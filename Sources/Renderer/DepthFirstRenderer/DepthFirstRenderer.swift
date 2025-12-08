import Foundation
@preconcurrency import Metal
import RendererTypes
import simd

public final class DepthFirstRenderer: GaussianRenderer, @unchecked Sendable {
    private static let maxSupportedGaussians = 30_000_000
    private static let tileWidth = 16
    private static let tileHeight = 16

    public let device: MTLDevice
    let library: MTLLibrary

    // Encoders
    let preprocessEncoder: DepthFirstPreprocessEncoder
    let dispatchEncoder: DepthFirstDispatchEncoder
    let depthSortEncoder: DepthSortEncoder
    let instanceExpansionEncoder: InstanceExpansionEncoder
    let tileSortEncoder: TileSortEncoder
    let tileRangeEncoder: TileRangeEncoder
    let renderEncoder: DepthFirstRenderEncoder
    let foveatedStereoEncoder: FoveatedStereoEncoder

    // Resources
    private let stereoResources: DepthFirstMultiViewResources

    // Debug access for testing
    var debugLeftFrame: DepthFirstViewResources { stereoResources.left }

    // Configuration
    private let config: RendererConfig
    private let limits: RendererLimits
    private let radixBlockSize = 256
    private let radixGrainSize = 4

    public private(set) var lastGPUTime: Double?

    public init(device: MTLDevice? = nil, config: RendererConfig = RendererConfig()) throws {
        guard config.maxGaussians <= DepthFirstRenderer.maxSupportedGaussians else {
            throw RendererError.invalidGaussianCount(
                provided: config.maxGaussians,
                maximum: DepthFirstRenderer.maxSupportedGaussians
            )
        }

        let device = device ?? MTLCreateSystemDefaultDevice()
        guard let device else {
            throw RendererError.deviceNotAvailable
        }
        self.device = device
        self.config = config

        guard let metallibURL = Bundle.module.url(forResource: "DepthFirstShaders", withExtension: "metallib") else {
            throw RendererError.failedToCreatePipeline("DepthFirstShaders.metallib not found in bundle")
        }
        let library = try device.makeLibrary(URL: metallibURL)
        self.library = library

        // Initialize encoders
        self.preprocessEncoder = try DepthFirstPreprocessEncoder(device: device, library: library)
        self.dispatchEncoder = try DepthFirstDispatchEncoder(device: device, library: library)
        self.depthSortEncoder = try DepthSortEncoder(device: device, library: library)
        self.instanceExpansionEncoder = try InstanceExpansionEncoder(device: device, library: library, maxGaussians: config.maxGaussians)
        self.tileSortEncoder = try TileSortEncoder(device: device, library: library)
        self.tileRangeEncoder = try TileRangeEncoder(device: device, library: library)
        self.renderEncoder = try DepthFirstRenderEncoder(device: device, library: library)
        self.foveatedStereoEncoder = try FoveatedStereoEncoder(device: device, library: library)

        // Compute limits
        self.limits = RendererLimits(
            from: config,
            tileWidth: DepthFirstRenderer.tileWidth,
            tileHeight: DepthFirstRenderer.tileHeight
        )

        self.stereoResources = try DepthFirstMultiViewResources(
            device: device,
            maxGaussians: limits.maxGaussians,
            maxWidth: limits.maxWidth,
            maxHeight: limits.maxHeight,
            tileWidth: limits.tileWidth,
            tileHeight: limits.tileHeight,
            radixBlockSize: radixBlockSize,
            radixGrainSize: radixGrainSize
        )
    }

    public func render(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        input: GaussianInput,
        camera: CameraParams,
        width: Int,
        height: Int
    ) {
        let cameraUniforms = CameraUniformsSwift(
            from: camera,
            width: width,
            height: height,
            gaussianCount: input.gaussianCount,
            shComponents: input.shComponents
        )

        let packedWorld = PackedWorldBuffers(
            packedGaussians: input.gaussians,
            harmonics: input.harmonics
        )

        encodeRender(
            commandBuffer: commandBuffer,
            gaussianCount: input.gaussianCount,
            packedWorldBuffers: packedWorld,
            cameraUniforms: cameraUniforms,
            targetColor: colorTexture,
            targetDepth: depthTexture,
            width: width,
            height: height,
            frame: stereoResources.left,
            shComponents: input.shComponents
        )
    }

    public func renderStereo(
        commandBuffer: MTLCommandBuffer,
        output: StereoRenderOutput,
        input: GaussianInput,
        camera: StereoCameraParams,
        width: Int,
        height: Int
    ) {
        let packedWorld = PackedWorldBuffers(
            packedGaussians: input.gaussians,
            harmonics: input.harmonics
        )

        // Left eye
        let leftCamera = CameraUniformsSwift(
            from: camera.leftEye,
            width: width,
            height: height,
            gaussianCount: input.gaussianCount,
            shComponents: input.shComponents
        )
        encodeRender(
            commandBuffer: commandBuffer,
            gaussianCount: input.gaussianCount,
            packedWorldBuffers: packedWorld,
            cameraUniforms: leftCamera,
            targetColor: output.leftColor,
            targetDepth: output.leftDepth,
            width: width,
            height: height,
            frame: stereoResources.left,
            shComponents: input.shComponents
        )

        // Right eye
        let rightCamera = CameraUniformsSwift(
            from: camera.rightEye,
            width: width,
            height: height,
            gaussianCount: input.gaussianCount,
            shComponents: input.shComponents
        )
        encodeRender(
            commandBuffer: commandBuffer,
            gaussianCount: input.gaussianCount,
            packedWorldBuffers: packedWorld,
            cameraUniforms: rightCamera,
            targetColor: output.rightColor,
            targetDepth: output.rightDepth,
            width: width,
            height: height,
            frame: stereoResources.right,
            shComponents: input.shComponents
        )
    }

    private func encodeRender(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        cameraUniforms: CameraUniformsSwift,
        targetColor: MTLTexture,
        targetDepth: MTLTexture?,
        width: Int,
        height: Int,
        frame: DepthFirstViewResources,
        shComponents: Int
    ) {
        guard gaussianCount > 0, gaussianCount <= limits.maxGaussians else { return }

        let tilesX = (width + DepthFirstRenderer.tileWidth - 1) / DepthFirstRenderer.tileWidth
        let tilesY = (height + DepthFirstRenderer.tileHeight - 1) / DepthFirstRenderer.tileHeight
        let tileCount = tilesX * tilesY

        // Build binning params
        let binningParams = limits.buildBinningParams(gaussianCount: gaussianCount)

        // Step 0: Reset state
        tileRangeEncoder.encodeResetState(
            commandBuffer: commandBuffer,
            header: frame.header,
            activeTileCount: frame.activeTileCount
        )

        // Clear only atomic counters - radix sort kernels use bounds checking from header
        // so no sentinel fills needed (saves ~8 large buffer fills per frame)
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "ClearCounters"
            blit.fill(buffer: frame.visibleCount, range: 0 ..< 4, value: 0)
            blit.fill(buffer: frame.totalInstances, range: 0 ..< 4, value: 0)
            blit.endEncoding()
        }

        // Step 1: Preprocess - project gaussians, compute depth keys, count tiles
        preprocessEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: cameraUniforms,
            renderData: frame.renderData,
            bounds: frame.bounds,
            depthKeys: frame.depthKeys,
            primitiveIndices: frame.primitiveIndices,
            nTouchedTiles: frame.nTouchedTiles,
            visibleCount: frame.visibleCount,
            totalInstances: frame.totalInstances,
            binningParams: binningParams,
            useHalfWorld: config.precision == .float16,
            shDegree: DepthFirstPreprocessEncoder.shDegree(from: shComponents)
        )

        // Step 1.5: Prepare indirect dispatch arguments (GPU computes dispatch sizes)
        let dispatchConfig = DepthFirstDispatchConfigSwift(
            maxGaussians: UInt32(limits.maxGaussians),
            maxInstances: UInt32(frame.maxInstances),
            radixBlockSize: UInt32(radixBlockSize),
            radixGrainSize: UInt32(radixGrainSize),
            instanceExpansionTGSize: UInt32(instanceExpansionEncoder.threadgroupSize),
            prefixSumTGSize: 256,
            tileRangeTGSize: UInt32(tileRangeEncoder.threadgroupSize),
            tileCount: UInt32(tileCount)
        )
        dispatchEncoder.encode(
            commandBuffer: commandBuffer,
            visibleCount: frame.visibleCount,
            totalInstances: frame.totalInstances,
            header: frame.header,
            dispatchArgs: frame.dispatchArgs,
            config: dispatchConfig
        )

        // Step 2: Sort by depth (uses indirect dispatch)
        let depthSortBuffers = DepthSortEncoder.SortBuffers(
            histogram: frame.depthSortHistogram,
            blockSums: frame.depthSortBlockSums,
            scannedHistogram: frame.depthSortScannedHist,
            scratchKeys: frame.depthSortScratchKeys,
            scratchPayload: frame.depthSortScratchPayload
        )
        depthSortEncoder.encode(
            commandBuffer: commandBuffer,
            depthKeys: frame.depthKeys,
            primitiveIndices: frame.primitiveIndices,
            sortBuffers: depthSortBuffers,
            header: frame.header,
            dispatchArgs: frame.dispatchArgs
        )

        // Step 3: Apply depth ordering - reorder tile counts (indirect dispatch)
        instanceExpansionEncoder.encodeApplyDepthOrdering(
            commandBuffer: commandBuffer,
            sortedPrimitiveIndices: frame.primitiveIndices,
            nTouchedTiles: frame.nTouchedTiles,
            orderedTileCounts: frame.orderedTileCounts,
            header: frame.header,
            dispatchArgs: frame.dispatchArgs
        )

        // Step 4: Prefix sum on ordered tile counts to get instance offsets (indirect dispatch)
        instanceExpansionEncoder.encodePrefixSum(
            commandBuffer: commandBuffer,
            dataBuffer: frame.orderedTileCounts,
            blockSumsBuffer: frame.prefixSumBlockSums,
            header: frame.header,
            dispatchArgs: frame.dispatchArgs
        )

        // Step 5: Create instances - expand depth-sorted gaussians to per-tile assignments
        let instanceParams = DepthFirstParamsSwift(
            gaussianCount: gaussianCount,
            visibleCount: 0, // Read from header on GPU
            tilesX: tilesX,
            tilesY: tilesY,
            tileWidth: DepthFirstRenderer.tileWidth,
            tileHeight: DepthFirstRenderer.tileHeight,
            maxAssignments: frame.maxInstances
        )
        instanceExpansionEncoder.encodeCreateInstances(
            commandBuffer: commandBuffer,
            sortedPrimitiveIndices: frame.primitiveIndices,
            instanceOffsets: frame.orderedTileCounts,
            tileBounds: frame.bounds,
            renderData: frame.renderData,
            instanceTileIds: frame.instanceTileIds,
            instanceGaussianIndices: frame.instanceGaussianIndices,
            params: instanceParams,
            header: frame.header,
            dispatchArgs: frame.dispatchArgs
        )

        // Step 6: Sort instances by tile ID (stable radix sort preserves depth order)
        let tileSortBuffers = TileSortEncoder.SortBuffers(
            histogram: frame.tileSortHistogram,
            blockSums: frame.tileSortBlockSums,
            scannedHistogram: frame.tileSortScannedHist,
            scratchTileIds: frame.tileSortScratchTileIds,
            scratchGaussianIndices: frame.tileSortScratchIndices
        )
        tileSortEncoder.encode(
            commandBuffer: commandBuffer,
            tileIds: frame.instanceTileIds,
            gaussianIndices: frame.instanceGaussianIndices,
            tileCount: tileCount,
            sortBuffers: tileSortBuffers,
            header: frame.header,
            dispatchArgs: frame.dispatchArgs
        )

        // Step 7: Extract tile ranges (reads totalInstances from header)
        tileRangeEncoder.encodeExtractRanges(
            commandBuffer: commandBuffer,
            sortedTileIds: frame.instanceTileIds,
            tileHeaders: frame.tileHeaders,
            activeTiles: frame.activeTiles,
            activeTileCount: frame.activeTileCount,
            header: frame.header,
            tileCount: tileCount
        )

        // Step 7.5: Prepare render dispatch based on actual active tile count
        dispatchEncoder.encodePrepareRenderDispatch(
            commandBuffer: commandBuffer,
            activeTileCount: frame.activeTileCount,
            dispatchArgs: frame.dispatchArgs,
            tileCount: UInt32(tileCount)
        )

        // Step 8: Clear and render
        renderEncoder.encodeClear(
            commandBuffer: commandBuffer,
            colorTexture: targetColor,
            depthTexture: targetDepth,
            width: width,
            height: height
        )

        let renderParams = RenderParams(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(DepthFirstRenderer.tileWidth),
            tileHeight: UInt32(DepthFirstRenderer.tileHeight),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            activeTileCount: 0,
            gaussianCount: UInt32(gaussianCount)
        )

        renderEncoder.encodeRender(
            commandBuffer: commandBuffer,
            tileHeaders: frame.tileHeaders,
            renderData: frame.renderData,
            sortedGaussianIndices: frame.instanceGaussianIndices,
            activeTiles: frame.activeTiles,
            activeTileCount: frame.activeTileCount,
            colorTexture: targetColor,
            depthTexture: targetDepth,
            params: renderParams,
            dispatchArgs: frame.dispatchArgs,
            dispatchOffset: DepthFirstDispatchSlot.render.offset
        )
    }

    // MARK: - Foveated Stereo Rendering

    /// Render gaussians to a foveated stereo drawable from Vision Pro Compositor Services.
    /// This uses a rasterization pipeline instead of compute, enabling:
    /// - Direct rendering to Compositor Services textures (which don't support compute writes)
    /// - Foveated rendering via MTLRasterizationRateMap
    /// - Single-pass stereo rendering with vertex amplification
    ///
    /// - Parameters:
    ///   - commandBuffer: The command buffer to encode rendering commands into
    ///   - drawable: The foveated stereo drawable from Compositor Services
    ///   - input: Gaussian splat data to render
    ///   - configuration: Stereo camera configuration for both eyes
    ///   - width: Render width in pixels (per eye)
    ///   - height: Render height in pixels (per eye)
    public func renderFoveatedStereo(
        commandBuffer: MTLCommandBuffer,
        drawable: FoveatedStereoDrawable,
        input: GaussianInput,
        configuration: FoveatedStereoConfiguration,
        width: Int,
        height: Int
    ) {
        let packedWorld = PackedWorldBuffers(
            packedGaussians: input.gaussians,
            harmonics: input.harmonics
        )

        switch configuration.layout {
        case .layered:
            // Single-pass stereo with vertex amplification
            renderFoveatedStereoLayered(
                commandBuffer: commandBuffer,
                drawable: drawable,
                packedWorldBuffers: packedWorld,
                configuration: configuration,
                input: input,
                width: width,
                height: height
            )

        case .dedicated:
            // Two separate passes, one per eye
            renderFoveatedStereoDedicated(
                commandBuffer: commandBuffer,
                drawable: drawable,
                packedWorldBuffers: packedWorld,
                configuration: configuration,
                input: input,
                width: width,
                height: height
            )

        case .shared:
            // Single texture with two viewports
            renderFoveatedStereoShared(
                commandBuffer: commandBuffer,
                drawable: drawable,
                packedWorldBuffers: packedWorld,
                configuration: configuration,
                input: input,
                width: width,
                height: height
            )
        }
    }

    /// Layered stereo rendering - single pass with vertex amplification
    private func renderFoveatedStereoLayered(
        commandBuffer: MTLCommandBuffer,
        drawable: FoveatedStereoDrawable,
        packedWorldBuffers: PackedWorldBuffers,
        configuration: FoveatedStereoConfiguration,
        input: GaussianInput,
        width: Int,
        height: Int
    ) {
        guard input.gaussianCount > 0, input.gaussianCount <= limits.maxGaussians else { return }

        // For layered rendering, we run compute pipeline once with combined camera
        // The sort position is the midpoint between eyes
        let sortCamera = CameraUniformsSwift(
            viewMatrix: configuration.leftEye.viewMatrix,  // Use left eye for culling
            projectionMatrix: configuration.leftEye.projectionMatrix,
            cameraCenter: configuration.sortPosition,
            pixelFactor: 1.0,
            focalX: configuration.leftEye.focalX,
            focalY: configuration.leftEye.focalY,
            width: Float(width),
            height: Float(height),
            nearPlane: configuration.leftEye.near,
            farPlane: configuration.leftEye.far,
            shComponents: UInt32(input.shComponents),
            gaussianCount: UInt32(input.gaussianCount)
        )

        let frame = stereoResources.left

        // Run compute pipeline (steps 0-7)
        encodeComputePipeline(
            commandBuffer: commandBuffer,
            gaussianCount: input.gaussianCount,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: sortCamera,
            width: width,
            height: height,
            frame: frame,
            shComponents: input.shComponents
        )

        // Encode foveated stereo render pass
        do {
            try foveatedStereoEncoder.encode(
                commandBuffer: commandBuffer,
                drawable: drawable,
                configuration: configuration,
                tileHeaders: frame.tileHeaders,
                renderData: frame.renderData,
                sortedGaussianIndices: frame.instanceGaussianIndices,
                width: width,
                height: height
            )
        } catch {
            // Log error but don't crash
            print("FoveatedStereo encode error: \(error)")
        }
    }

    /// Dedicated stereo rendering - two separate textures
    private func renderFoveatedStereoDedicated(
        commandBuffer: MTLCommandBuffer,
        drawable: FoveatedStereoDrawable,
        packedWorldBuffers: PackedWorldBuffers,
        configuration: FoveatedStereoConfiguration,
        input: GaussianInput,
        width: Int,
        height: Int
    ) {
        guard input.gaussianCount > 0, input.gaussianCount <= limits.maxGaussians else { return }

        // For dedicated layout, render each eye separately
        // Left eye
        let leftCamera = CameraUniformsSwift(
            viewMatrix: configuration.leftEye.viewMatrix,
            projectionMatrix: configuration.leftEye.projectionMatrix,
            cameraCenter: configuration.leftEye.cameraPosition,
            pixelFactor: 1.0,
            focalX: configuration.leftEye.focalX,
            focalY: configuration.leftEye.focalY,
            width: Float(width),
            height: Float(height),
            nearPlane: configuration.leftEye.near,
            farPlane: configuration.leftEye.far,
            shComponents: UInt32(input.shComponents),
            gaussianCount: UInt32(input.gaussianCount)
        )

        let leftFrame = stereoResources.left
        encodeComputePipeline(
            commandBuffer: commandBuffer,
            gaussianCount: input.gaussianCount,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: leftCamera,
            width: width,
            height: height,
            frame: leftFrame,
            shComponents: input.shComponents
        )

        // Encode left eye render
        do {
            try foveatedStereoEncoder.encodeSingleEye(
                commandBuffer: commandBuffer,
                colorTexture: drawable.colorTexture,
                depthTexture: drawable.depthTexture,
                rasterizationRateMap: drawable.rasterizationRateMap,
                viewport: configuration.leftEye.viewport,
                tileHeaders: leftFrame.tileHeaders,
                renderData: leftFrame.renderData,
                sortedGaussianIndices: leftFrame.instanceGaussianIndices,
                width: width,
                height: height,
                colorPixelFormat: drawable.colorPixelFormat,
                depthPixelFormat: drawable.depthPixelFormat
            )
        } catch {
            print("FoveatedStereo left eye encode error: \(error)")
        }

        // Right eye
        let rightCamera = CameraUniformsSwift(
            viewMatrix: configuration.rightEye.viewMatrix,
            projectionMatrix: configuration.rightEye.projectionMatrix,
            cameraCenter: configuration.rightEye.cameraPosition,
            pixelFactor: 1.0,
            focalX: configuration.rightEye.focalX,
            focalY: configuration.rightEye.focalY,
            width: Float(width),
            height: Float(height),
            nearPlane: configuration.rightEye.near,
            farPlane: configuration.rightEye.far,
            shComponents: UInt32(input.shComponents),
            gaussianCount: UInt32(input.gaussianCount)
        )

        let rightFrame = stereoResources.right
        encodeComputePipeline(
            commandBuffer: commandBuffer,
            gaussianCount: input.gaussianCount,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: rightCamera,
            width: width,
            height: height,
            frame: rightFrame,
            shComponents: input.shComponents
        )

        // For dedicated, we need a second texture - but the drawable only has one
        // This layout isn't fully supported with the current API
        // Users should use layered or shared layout instead
    }

    /// Shared stereo rendering - single texture with two viewports
    private func renderFoveatedStereoShared(
        commandBuffer: MTLCommandBuffer,
        drawable: FoveatedStereoDrawable,
        packedWorldBuffers: PackedWorldBuffers,
        configuration: FoveatedStereoConfiguration,
        input: GaussianInput,
        width: Int,
        height: Int
    ) {
        // For shared layout, we can use vertex amplification similar to layered
        // but writing to different viewports in the same texture slice
        renderFoveatedStereoLayered(
            commandBuffer: commandBuffer,
            drawable: drawable,
            packedWorldBuffers: packedWorldBuffers,
            configuration: configuration,
            input: input,
            width: width,
            height: height
        )
    }

    /// Encode compute pipeline stages (preprocess through tile range extraction)
    /// Used by foveated stereo rendering to prepare data before the raster pass
    private func encodeComputePipeline(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        cameraUniforms: CameraUniformsSwift,
        width: Int,
        height: Int,
        frame: DepthFirstViewResources,
        shComponents: Int
    ) {
        let tilesX = (width + DepthFirstRenderer.tileWidth - 1) / DepthFirstRenderer.tileWidth
        let tilesY = (height + DepthFirstRenderer.tileHeight - 1) / DepthFirstRenderer.tileHeight
        let tileCount = tilesX * tilesY

        // Build binning params
        let binningParams = limits.buildBinningParams(gaussianCount: gaussianCount)

        // Step 0: Reset state
        tileRangeEncoder.encodeResetState(
            commandBuffer: commandBuffer,
            header: frame.header,
            activeTileCount: frame.activeTileCount
        )

        // Clear atomic counters
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "ClearCounters"
            blit.fill(buffer: frame.visibleCount, range: 0 ..< 4, value: 0)
            blit.fill(buffer: frame.totalInstances, range: 0 ..< 4, value: 0)
            blit.endEncoding()
        }

        // Step 1: Preprocess
        preprocessEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: cameraUniforms,
            renderData: frame.renderData,
            bounds: frame.bounds,
            depthKeys: frame.depthKeys,
            primitiveIndices: frame.primitiveIndices,
            nTouchedTiles: frame.nTouchedTiles,
            visibleCount: frame.visibleCount,
            totalInstances: frame.totalInstances,
            binningParams: binningParams,
            useHalfWorld: config.precision == .float16,
            shDegree: DepthFirstPreprocessEncoder.shDegree(from: shComponents)
        )

        // Step 1.5: Prepare indirect dispatch arguments
        let dispatchConfig = DepthFirstDispatchConfigSwift(
            maxGaussians: UInt32(limits.maxGaussians),
            maxInstances: UInt32(frame.maxInstances),
            radixBlockSize: UInt32(radixBlockSize),
            radixGrainSize: UInt32(radixGrainSize),
            instanceExpansionTGSize: UInt32(instanceExpansionEncoder.threadgroupSize),
            prefixSumTGSize: 256,
            tileRangeTGSize: UInt32(tileRangeEncoder.threadgroupSize),
            tileCount: UInt32(tileCount)
        )
        dispatchEncoder.encode(
            commandBuffer: commandBuffer,
            visibleCount: frame.visibleCount,
            totalInstances: frame.totalInstances,
            header: frame.header,
            dispatchArgs: frame.dispatchArgs,
            config: dispatchConfig
        )

        // Step 2: Sort by depth
        let depthSortBuffers = DepthSortEncoder.SortBuffers(
            histogram: frame.depthSortHistogram,
            blockSums: frame.depthSortBlockSums,
            scannedHistogram: frame.depthSortScannedHist,
            scratchKeys: frame.depthSortScratchKeys,
            scratchPayload: frame.depthSortScratchPayload
        )
        depthSortEncoder.encode(
            commandBuffer: commandBuffer,
            depthKeys: frame.depthKeys,
            primitiveIndices: frame.primitiveIndices,
            sortBuffers: depthSortBuffers,
            header: frame.header,
            dispatchArgs: frame.dispatchArgs
        )

        // Step 3: Apply depth ordering
        instanceExpansionEncoder.encodeApplyDepthOrdering(
            commandBuffer: commandBuffer,
            sortedPrimitiveIndices: frame.primitiveIndices,
            nTouchedTiles: frame.nTouchedTiles,
            orderedTileCounts: frame.orderedTileCounts,
            header: frame.header,
            dispatchArgs: frame.dispatchArgs
        )

        // Step 4: Prefix sum
        instanceExpansionEncoder.encodePrefixSum(
            commandBuffer: commandBuffer,
            dataBuffer: frame.orderedTileCounts,
            blockSumsBuffer: frame.prefixSumBlockSums,
            header: frame.header,
            dispatchArgs: frame.dispatchArgs
        )

        // Step 5: Create instances
        let instanceParams = DepthFirstParamsSwift(
            gaussianCount: gaussianCount,
            visibleCount: 0,
            tilesX: tilesX,
            tilesY: tilesY,
            tileWidth: DepthFirstRenderer.tileWidth,
            tileHeight: DepthFirstRenderer.tileHeight,
            maxAssignments: frame.maxInstances
        )
        instanceExpansionEncoder.encodeCreateInstances(
            commandBuffer: commandBuffer,
            sortedPrimitiveIndices: frame.primitiveIndices,
            instanceOffsets: frame.orderedTileCounts,
            tileBounds: frame.bounds,
            renderData: frame.renderData,
            instanceTileIds: frame.instanceTileIds,
            instanceGaussianIndices: frame.instanceGaussianIndices,
            params: instanceParams,
            header: frame.header,
            dispatchArgs: frame.dispatchArgs
        )

        // Step 6: Sort instances by tile ID
        let tileSortBuffers = TileSortEncoder.SortBuffers(
            histogram: frame.tileSortHistogram,
            blockSums: frame.tileSortBlockSums,
            scannedHistogram: frame.tileSortScannedHist,
            scratchTileIds: frame.tileSortScratchTileIds,
            scratchGaussianIndices: frame.tileSortScratchIndices
        )
        tileSortEncoder.encode(
            commandBuffer: commandBuffer,
            tileIds: frame.instanceTileIds,
            gaussianIndices: frame.instanceGaussianIndices,
            tileCount: tileCount,
            sortBuffers: tileSortBuffers,
            header: frame.header,
            dispatchArgs: frame.dispatchArgs
        )

        // Step 7: Extract tile ranges
        tileRangeEncoder.encodeExtractRanges(
            commandBuffer: commandBuffer,
            sortedTileIds: frame.instanceTileIds,
            tileHeaders: frame.tileHeaders,
            activeTiles: frame.activeTiles,
            activeTileCount: frame.activeTileCount,
            header: frame.header,
            tileCount: tileCount
        )
    }
}

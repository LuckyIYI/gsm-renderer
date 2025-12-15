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
    let visibilityCompactionEncoder: VisibilityCompactionEncoder
    let dispatchEncoder: DepthFirstDispatchEncoder
    let depthSortEncoder: DepthRadixSortEncoder
    let instanceExpansionEncoder: InstanceExpansionEncoder
    let tileSortEncoder: TileSortEncoder
    let tileRangeEncoder: TileRangeEncoder
    let renderEncoder: DepthFirstRenderEncoder

    // Unified stereo encoders (tiled pipeline)
    let stereoUnifiedPreprocessEncoder: DepthFirstStereoUnifiedPreprocessEncoder
    let stereoComputeRenderEncoder: DepthFirstStereoComputeRenderEncoder
    let stereoCopyEncoder: DepthFirstStereoCopyEncoder

    // Resources (lazily allocated based on usage)
    private var _monoResources: DepthFirstViewResources?
    private var _stereoUnifiedResources: StereoUnifiedTiledResources?
    private var _stereoIntermediateColor: MTLTexture?

    // Debug access for testing
    var debugLeftFrame: DepthFirstViewResources? { _monoResources }

    // Configuration
    private let config: RendererConfig
    private let limits: RendererLimits
    private let radixBlockSize = 256
    private let radixGrainSize = 4

    public private(set) var lastGPUTime: Double?

    public init(
        device: MTLDevice? = nil,
        config: RendererConfig = RendererConfig(),
        depthSortKeyPrecision: RadixSortKeyPrecision = .bits32
    ) throws {
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
        self.visibilityCompactionEncoder = try VisibilityCompactionEncoder(
            device: device,
            library: library,
            depthKeyPrecision: depthSortKeyPrecision
        )
        self.dispatchEncoder = try DepthFirstDispatchEncoder(device: device, library: library)
        self.depthSortEncoder = try DepthRadixSortEncoder(device: device, library: library, precision: depthSortKeyPrecision)
        self.instanceExpansionEncoder = try InstanceExpansionEncoder(device: device, library: library, maxGaussians: config.maxGaussians)
        self.tileSortEncoder = try TileSortEncoder(device: device, library: library)
        self.tileRangeEncoder = try TileRangeEncoder(device: device, library: library)
        self.renderEncoder = try DepthFirstRenderEncoder(device: device, library: library)

        // Unified stereo encoders (tiled pipeline)
        self.stereoUnifiedPreprocessEncoder = try DepthFirstStereoUnifiedPreprocessEncoder(device: device, library: library)
        self.stereoComputeRenderEncoder = try DepthFirstStereoComputeRenderEncoder(device: device, library: library)
        self.stereoCopyEncoder = try DepthFirstStereoCopyEncoder(device: device, library: library, colorFormat: config.colorFormat)

        // Compute limits
        self.limits = RendererLimits(
            from: config,
            tileWidth: DepthFirstRenderer.tileWidth,
            tileHeight: DepthFirstRenderer.tileHeight
        )

        // Resources are lazily allocated on first use
    }

    // MARK: - Lazy Resource Accessors

    private func ensureMonoResources() throws -> DepthFirstViewResources {
        if let existing = _monoResources { return existing }
        let resources = try DepthFirstViewResources(
            device: device,
            maxGaussians: limits.maxGaussians,
            maxWidth: limits.maxWidth,
            maxHeight: limits.maxHeight,
            tileWidth: limits.tileWidth,
            tileHeight: limits.tileHeight,
            radixBlockSize: radixBlockSize,
            radixGrainSize: radixGrainSize
        )
        _monoResources = resources
        return resources
    }

    private func ensureStereoUnifiedResources() throws -> StereoUnifiedTiledResources {
        if let existing = _stereoUnifiedResources { return existing }
        let resources = try StereoUnifiedTiledResources(
            device: device,
            maxGaussians: limits.maxGaussians,
            maxWidth: limits.maxWidth,
            maxHeight: limits.maxHeight,
            tileWidth: limits.tileWidth,
            tileHeight: limits.tileHeight,
            radixBlockSize: radixBlockSize,
            radixGrainSize: radixGrainSize
        )
        _stereoUnifiedResources = resources
        return resources
    }

    private func ensureStereoIntermediateColor(width: Int, height: Int) -> MTLTexture? {
        if let existing = _stereoIntermediateColor,
           existing.width == width,
           existing.height == height,
           existing.arrayLength == 2,
           existing.pixelFormat == .rgba16Float
        {
            return existing
        }

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.textureType = .type2DArray
        desc.arrayLength = 2
        desc.usage = [.shaderWrite, .shaderRead]
        desc.storageMode = .private
        desc.resourceOptions = .storageModePrivate
        let tex = device.makeTexture(descriptor: desc)
        tex?.label = "DepthFirstStereoIntermediateColor"
        _stereoIntermediateColor = tex
        return tex
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
            shComponents: input.shComponents,
            inputIsSRGB: config.gaussianColorSpace == .srgb
        )

        let packedWorld = PackedWorldBuffers(
            packedGaussians: input.gaussians,
            harmonics: input.harmonics
        )

        guard let monoResources = try? ensureMonoResources() else { return }

        encodeRender(
            commandBuffer: commandBuffer,
            gaussianCount: input.gaussianCount,
            packedWorldBuffers: packedWorld,
            cameraUniforms: cameraUniforms,
            targetColor: colorTexture,
            targetDepth: depthTexture,
            width: width,
            height: height,
            frame: monoResources,
            shComponents: input.shComponents
        )
    }

    public func renderStereo(
        commandBuffer: MTLCommandBuffer,
        target: StereoRenderTarget,
        input: GaussianInput,
        camera: StereoCameraParams,
        width: Int,
        height: Int
    ) {
        switch target {
        case .sideBySide(let colorTexture, let depthTexture):
            renderStereoSideBySideRaster(
                commandBuffer: commandBuffer,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                input: input,
                camera: camera,
                width: width,
                height: height,
            )

        case .foveated(let drawable, let configuration):
            renderFoveatedStereoInternal(
                commandBuffer: commandBuffer,
                drawable: drawable,
                input: input,
                configuration: configuration,
                width: width,
                height: height,
            )
        }
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

        // Build binning params with actual render dimensions
        let binningParams = limits.buildBinningParams(gaussianCount: gaussianCount, width: width, height: height)

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
            preDepthKeys: frame.depthSortScratchKeys,
            nTouchedTiles: frame.nTouchedTiles,
            totalInstances: frame.totalInstances,
            binningParams: binningParams,
            useHalfWorld: config.precision == .float16,
            shDegree: DepthFirstPreprocessEncoder.shDegree(from: shComponents)
        )

        // Step 1.25: Deterministic visibility compaction into depth sort buffers (writes visibleCount).
        visibilityCompactionEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            nTouchedTiles: frame.nTouchedTiles,
            preDepthKeys: frame.depthSortScratchKeys,
            prefixSumsScratch: frame.prefixSumBlockSums,
            prefixOffsetsOut: frame.depthSortScratchPayload,
            depthKeysOut: frame.depthKeys,
            primitiveIndicesOut: frame.primitiveIndices,
            visibleCountOut: frame.visibleCount,
            maxOutCount: frame.maxGaussians,
            maxCountCapacity: frame.maxGaussians
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
        let depthSortBuffers = RadixSortBuffers(
            histogram: frame.depthSortHistogram,
            blockSums: frame.depthSortBlockSums,
            scannedHistogram: frame.depthSortScannedHist,
            scratchKeys: frame.depthSortScratchKeys,
            scratchIndices: frame.depthSortScratchPayload
        )
        let depthDispatchOffsets = DepthRadixSortEncoder.DispatchOffsets.fromSlots(
            histogram: DepthFirstDispatchSlot.depthHistogram.rawValue,
            scanBlocks: DepthFirstDispatchSlot.depthScanBlocks.rawValue,
            exclusive: DepthFirstDispatchSlot.depthExclusive.rawValue,
            apply: DepthFirstDispatchSlot.depthApply.rawValue,
            scatter: DepthFirstDispatchSlot.depthScatter.rawValue,
            stride: MemoryLayout<DispatchIndirectArgsSwift>.stride
        )
        depthSortEncoder.encode(
            commandBuffer: commandBuffer,
            depthKeys: frame.depthKeys,
            sortedIndices: frame.primitiveIndices,
            sortBuffers: depthSortBuffers,
            sortHeader: frame.header,
            dispatchBuffer: frame.dispatchArgs,
            offsets: depthDispatchOffsets,
            label: "DepthSort"
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
            maxAssignments: frame.maxInstances,
            alphaThreshold: binningParams.alphaThreshold
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

    // MARK: - Stereo Rendering (Center-Sort with Billboard Quads)

    private func renderStereoSideBySideRaster(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture _: MTLTexture?,
        input: GaussianInput,
        camera: StereoCameraParams,
        width: Int,
        height: Int
    ) {
        guard input.gaussianCount > 0, input.gaussianCount <= limits.maxGaussians else { return }

        let configuration = StereoConfiguration(
            from: camera,
            width: width,
            height: height,
            rightViewOrigin: .init(Float(width), 0)
        )

        let packedWorld = PackedWorldBuffers(
            packedGaussians: input.gaussians,
            harmonics: input.harmonics
        )

        let stereoCameraUniforms = makeStereoCameraUniforms(
            from: configuration,
            gaussianCount: input.gaussianCount,
            shComponents: input.shComponents,
            width: width,
            height: height
        )

        // Use unified tiled pipeline (same as foveated) → render directly to side-by-side target.
        encodeUnifiedStereoPipeline(
            commandBuffer: commandBuffer,
            gaussianCount: input.gaussianCount,
            packedWorldBuffers: packedWorld,
            stereoCameraUniforms: stereoCameraUniforms,
            configuration: configuration,
            rasterizationRateMap: nil,
            colorTexture: colorTexture,
            width: width,
            height: height,
            shComponents: input.shComponents
        )
    }

    // MARK: - Foveated Stereo Rendering (Unified Tiled Pipeline)

    private func renderFoveatedStereoInternal(
        commandBuffer: MTLCommandBuffer,
        drawable: FoveatedStereoDrawable,
        input: GaussianInput,
        configuration: StereoConfiguration,
        width: Int,
        height: Int
    ) {
        guard input.gaussianCount > 0, input.gaussianCount <= limits.maxGaussians else { return }

        let packedWorld = PackedWorldBuffers(
            packedGaussians: input.gaussians,
            harmonics: input.harmonics
        )

        let stereoCameraUniforms = makeStereoCameraUniforms(
            from: configuration,
            gaussianCount: input.gaussianCount,
            shComponents: input.shComponents,
            width: width,
            height: height
        )

        encodeUnifiedStereoPipeline(
            commandBuffer: commandBuffer,
            gaussianCount: input.gaussianCount,
            packedWorldBuffers: packedWorld,
            stereoCameraUniforms: stereoCameraUniforms,
            configuration: configuration,
            rasterizationRateMap: drawable.rasterizationRateMap,
            colorTexture: drawable.colorTexture,
            width: width,
            height: height,
            shComponents: input.shComponents
        )
    }

    /// Build StereoCameraUniforms from StereoConfiguration for unified stereo preprocess
    private func makeStereoCameraUniforms(
        from configuration: StereoConfiguration,
        gaussianCount: Int,
        shComponents: Int,
        width: Int,
        height: Int
    ) -> StereoCameraUniforms {
        let left = configuration.leftEye
        let right = configuration.rightEye

        return StereoCameraUniforms(
            leftViewMatrix: left.viewMatrix,
            leftProjectionMatrix: left.projectionMatrix,
            leftCameraCenterX: left.cameraPosition.x,
            leftCameraCenterY: left.cameraPosition.y,
            leftCameraCenterZ: left.cameraPosition.z,
            leftFocalX: left.focalX,
            leftFocalY: left.focalY,
            _padLeft0: 0, _padLeft1: 0, _padLeft2: 0,
            rightViewMatrix: right.viewMatrix,
            rightProjectionMatrix: right.projectionMatrix,
            rightCameraCenterX: right.cameraPosition.x,
            rightCameraCenterY: right.cameraPosition.y,
            rightCameraCenterZ: right.cameraPosition.z,
            rightFocalX: right.focalX,
            rightFocalY: right.focalY,
            _padRight0: 0, _padRight1: 0, _padRight2: 0,
            width: Float(width),
            height: Float(height),
            nearPlane: left.near,
            farPlane: left.far,
            shComponents: UInt32(shComponents),
            gaussianCount: UInt32(gaussianCount),
            inputIsSRGB: config.gaussianColorSpace == .srgb ? 1.0 : 0.0,
            _padShared1: 0,
            sceneTransform: configuration.sceneTransform
        )
    }

    // MARK: - Unified Stereo Tiled Pipeline

    /// Encode the full unified stereo tiled pipeline.
    /// This mirrors the mono pipeline but uses:
    /// - StereoTiledRenderData (both eye positions per gaussian)
    /// - Union tile bounds for tile assignment
    /// - Stereo render kernel that outputs to texture array
    private func encodeUnifiedStereoPipeline(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        stereoCameraUniforms: StereoCameraUniforms,
        configuration: StereoConfiguration,
        rasterizationRateMap: MTLRasterizationRateMap?,
        colorTexture: MTLTexture,
        width: Int,
        height: Int,
        shComponents: Int
    ) {
        guard gaussianCount > 0, gaussianCount <= limits.maxGaussians else { return }
        guard let res = try? ensureStereoUnifiedResources() else { return }

        let tilesX = (width + DepthFirstRenderer.tileWidth - 1) / DepthFirstRenderer.tileWidth
        let tilesY = (height + DepthFirstRenderer.tileHeight - 1) / DepthFirstRenderer.tileHeight
        let tileCount = tilesX * tilesY

        // Build binning params with actual render dimensions
        let binningParams = limits.buildBinningParams(gaussianCount: gaussianCount, width: width, height: height)

        // Step 0: Reset state
        tileRangeEncoder.encodeResetState(
            commandBuffer: commandBuffer,
            header: res.header,
            activeTileCount: res.activeTileCount
        )

        // Clear atomic counters
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "ClearStereoUnifiedCounters"
            blit.fill(buffer: res.visibleCount, range: 0 ..< 4, value: 0)
            blit.fill(buffer: res.totalInstances, range: 0 ..< 4, value: 0)
            blit.endEncoding()
        }

        // Step 1: Stereo preprocess - project both eyes, compute union bounds
        stereoUnifiedPreprocessEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            packedWorldBuffers: packedWorldBuffers,
            stereoCamera: stereoCameraUniforms,
            renderData: res.renderData,
            bounds: res.bounds,
            preDepthKeys: res.depthSortScratchKeys,
            nTouchedTiles: res.nTouchedTiles,
            totalInstances: res.totalInstances,
            binningParams: binningParams,
            useHalfWorld: config.precision == .float16,
            shDegree: DepthFirstStereoUnifiedPreprocessEncoder.shDegree(from: shComponents)
        )

        // Step 1.25: Deterministic visibility compaction into depth sort buffers (writes visibleCount).
        visibilityCompactionEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            nTouchedTiles: res.nTouchedTiles,
            preDepthKeys: res.depthSortScratchKeys,
            prefixSumsScratch: res.prefixSumBlockSums,
            prefixOffsetsOut: res.depthSortScratchPayload,
            depthKeysOut: res.depthKeys,
            primitiveIndicesOut: res.primitiveIndices,
            visibleCountOut: res.visibleCount,
            maxOutCount: res.maxGaussians,
            maxCountCapacity: res.maxGaussians
        )

        // Step 1.5: Prepare indirect dispatch arguments
        let dispatchConfig = DepthFirstDispatchConfigSwift(
            maxGaussians: UInt32(limits.maxGaussians),
            maxInstances: UInt32(res.maxInstances),
            radixBlockSize: UInt32(radixBlockSize),
            radixGrainSize: UInt32(radixGrainSize),
            instanceExpansionTGSize: UInt32(instanceExpansionEncoder.threadgroupSize),
            prefixSumTGSize: 256,
            tileRangeTGSize: UInt32(tileRangeEncoder.threadgroupSize),
            tileCount: UInt32(tileCount)
        )
        dispatchEncoder.encode(
            commandBuffer: commandBuffer,
            visibleCount: res.visibleCount,
            totalInstances: res.totalInstances,
            header: res.header,
            dispatchArgs: res.dispatchArgs,
            config: dispatchConfig
        )

        // Step 2: Sort by depth
        let depthSortBuffers = RadixSortBuffers(
            histogram: res.depthSortHistogram,
            blockSums: res.depthSortBlockSums,
            scannedHistogram: res.depthSortScannedHist,
            scratchKeys: res.depthSortScratchKeys,
            scratchIndices: res.depthSortScratchPayload
        )
        let depthDispatchOffsets = DepthRadixSortEncoder.DispatchOffsets.fromSlots(
            histogram: DepthFirstDispatchSlot.depthHistogram.rawValue,
            scanBlocks: DepthFirstDispatchSlot.depthScanBlocks.rawValue,
            exclusive: DepthFirstDispatchSlot.depthExclusive.rawValue,
            apply: DepthFirstDispatchSlot.depthApply.rawValue,
            scatter: DepthFirstDispatchSlot.depthScatter.rawValue,
            stride: MemoryLayout<DispatchIndirectArgsSwift>.stride
        )
        depthSortEncoder.encode(
            commandBuffer: commandBuffer,
            depthKeys: res.depthKeys,
            sortedIndices: res.primitiveIndices,
            sortBuffers: depthSortBuffers,
            sortHeader: res.header,
            dispatchBuffer: res.dispatchArgs,
            offsets: depthDispatchOffsets,
            label: "StereoDepthSort"
        )

        // Step 3: Apply depth ordering - reorder tile counts
        instanceExpansionEncoder.encodeApplyDepthOrdering(
            commandBuffer: commandBuffer,
            sortedPrimitiveIndices: res.primitiveIndices,
            nTouchedTiles: res.nTouchedTiles,
            orderedTileCounts: res.orderedTileCounts,
            header: res.header,
            dispatchArgs: res.dispatchArgs
        )

        // Step 4: Prefix sum on ordered tile counts
        instanceExpansionEncoder.encodePrefixSum(
            commandBuffer: commandBuffer,
            dataBuffer: res.orderedTileCounts,
            blockSumsBuffer: res.prefixSumBlockSums,
            header: res.header,
            dispatchArgs: res.dispatchArgs
        )

        // Step 5: Create instances - expand depth-sorted gaussians to per-tile assignments
        let instanceParams = DepthFirstParamsSwift(
            gaussianCount: gaussianCount,
            visibleCount: 0,
            tilesX: tilesX,
            tilesY: tilesY,
            tileWidth: DepthFirstRenderer.tileWidth,
            tileHeight: DepthFirstRenderer.tileHeight,
            maxAssignments: res.maxInstances,
            alphaThreshold: binningParams.alphaThreshold
        )
        instanceExpansionEncoder.encodeCreateInstancesStereo(
            commandBuffer: commandBuffer,
            sortedPrimitiveIndices: res.primitiveIndices,
            instanceOffsets: res.orderedTileCounts,
            tileBounds: res.bounds,
            instanceTileIds: res.instanceTileIds,
            instanceGaussianIndices: res.instanceGaussianIndices,
            params: instanceParams,
            header: res.header,
            dispatchArgs: res.dispatchArgs
        )

        // Step 6: Sort instances by tile ID
        let tileSortBuffers = TileSortEncoder.SortBuffers(
            histogram: res.tileSortHistogram,
            blockSums: res.tileSortBlockSums,
            scannedHistogram: res.tileSortScannedHist,
            scratchTileIds: res.tileSortScratchTileIds,
            scratchGaussianIndices: res.tileSortScratchIndices
        )
        tileSortEncoder.encode(
            commandBuffer: commandBuffer,
            tileIds: res.instanceTileIds,
            gaussianIndices: res.instanceGaussianIndices,
            tileCount: tileCount,
            sortBuffers: tileSortBuffers,
            header: res.header,
            dispatchArgs: res.dispatchArgs
        )

        // Step 7: Extract tile ranges
        tileRangeEncoder.encodeExtractRanges(
            commandBuffer: commandBuffer,
            sortedTileIds: res.instanceTileIds,
            tileHeaders: res.tileHeaders,
            activeTiles: res.activeTiles,
            activeTileCount: res.activeTileCount,
            header: res.header,
            tileCount: tileCount
        )

        // Step 8: Prepare render dispatch based on active tile count
        dispatchEncoder.encodePrepareRenderDispatch(
            commandBuffer: commandBuffer,
            activeTileCount: res.activeTileCount,
            dispatchArgs: res.dispatchArgs,
            tileCount: UInt32(tileCount)
        )

        // Step 9: Software blend for both eyes in a single compute pass (into intermediate array)
        guard let intermediate = ensureStereoIntermediateColor(width: width, height: height) else { return }

        stereoComputeRenderEncoder.encodeClear(
            commandBuffer: commandBuffer,
            colorTexture: intermediate,
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
            gaussianCount: UInt32(gaussianCount),
        )

        stereoComputeRenderEncoder.encodeRender(
            commandBuffer: commandBuffer,
            tileHeaders: res.tileHeaders,
            renderData: res.renderData,
            sortedGaussianIndices: res.instanceGaussianIndices,
            activeTiles: res.activeTiles,
            activeTileCount: res.activeTileCount,
            colorTexture: intermediate,
            params: renderParams,
            dispatchArgs: res.dispatchArgs,
            dispatchOffset: DepthFirstDispatchSlot.render.offset
        )

        // Step 10: Copy to the final target using rasterization rate map + vertex amplification
        stereoCopyEncoder.encodeRender(
            commandBuffer: commandBuffer,
            sourceTexture: intermediate,
            destinationTexture: colorTexture,
            rasterizationRateMap: rasterizationRateMap,
            configuration: configuration
        )
    }

    // MARK: - Mono Pipeline Support

    private func buildDispatchConfig(frame: DepthFirstViewResources, tileCount: Int) -> DepthFirstDispatchConfigSwift {
        DepthFirstDispatchConfigSwift(
            maxGaussians: UInt32(limits.maxGaussians),
            maxInstances: UInt32(frame.maxInstances),
            radixBlockSize: UInt32(radixBlockSize),
            radixGrainSize: UInt32(radixGrainSize),
            instanceExpansionTGSize: UInt32(instanceExpansionEncoder.threadgroupSize),
            prefixSumTGSize: 256,
            tileRangeTGSize: UInt32(tileRangeEncoder.threadgroupSize),
            tileCount: UInt32(tileCount)
        )
    }

    private func encodeDepthSort(commandBuffer: MTLCommandBuffer, frame: DepthFirstViewResources) {
        let depthSortBuffers = RadixSortBuffers(
            histogram: frame.depthSortHistogram,
            blockSums: frame.depthSortBlockSums,
            scannedHistogram: frame.depthSortScannedHist,
            scratchKeys: frame.depthSortScratchKeys,
            scratchIndices: frame.depthSortScratchPayload
        )
        let depthDispatchOffsets = DepthRadixSortEncoder.DispatchOffsets.fromSlots(
            histogram: DepthFirstDispatchSlot.depthHistogram.rawValue,
            scanBlocks: DepthFirstDispatchSlot.depthScanBlocks.rawValue,
            exclusive: DepthFirstDispatchSlot.depthExclusive.rawValue,
            apply: DepthFirstDispatchSlot.depthApply.rawValue,
            scatter: DepthFirstDispatchSlot.depthScatter.rawValue,
            stride: MemoryLayout<DispatchIndirectArgsSwift>.stride
        )
        depthSortEncoder.encode(
            commandBuffer: commandBuffer,
            depthKeys: frame.depthKeys,
            sortedIndices: frame.primitiveIndices,
            sortBuffers: depthSortBuffers,
            sortHeader: frame.header,
            dispatchBuffer: frame.dispatchArgs,
            offsets: depthDispatchOffsets,
            label: "DepthSort"
        )
    }

    /// Shared between stereo eyes (after preprocess and optional shared sorting).
    private func encodePostPreprocessPipeline(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        frame: DepthFirstViewResources,
        tilesX: Int,
        tilesY: Int,
        tileCount: Int,
        visibleCount: MTLBuffer,
        sortedPrimitiveIndices: MTLBuffer,
        doDepthSort: Bool,
        alphaThreshold: Float
    ) {
        // Step 1.5: Prepare indirect dispatch arguments
        let dispatchConfig = buildDispatchConfig(frame: frame, tileCount: tileCount)
        dispatchEncoder.encode(
            commandBuffer: commandBuffer,
            visibleCount: visibleCount,
            totalInstances: frame.totalInstances,
            header: frame.header,
            dispatchArgs: frame.dispatchArgs,
            config: dispatchConfig
        )

        // Step 2: Sort by depth (optional for center-sort mode)
        if doDepthSort {
            encodeDepthSort(commandBuffer: commandBuffer, frame: frame)
        }

        // Step 3: Apply depth ordering
        instanceExpansionEncoder.encodeApplyDepthOrdering(
            commandBuffer: commandBuffer,
            sortedPrimitiveIndices: sortedPrimitiveIndices,
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
            maxAssignments: frame.maxInstances,
            alphaThreshold: alphaThreshold
        )
        instanceExpansionEncoder.encodeCreateInstances(
            commandBuffer: commandBuffer,
            sortedPrimitiveIndices: sortedPrimitiveIndices,
            instanceOffsets: frame.orderedTileCounts,
            tileBounds: frame.bounds,
            renderData: frame.renderData,
            instanceTileIds: frame.instanceTileIds,
            instanceGaussianIndices: frame.instanceGaussianIndices,
            params: instanceParams,
            header: frame.header,
            dispatchArgs: frame.dispatchArgs
        )

        // Step 6: Sort instances by tile ID (stable, preserves order within each tile)
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

    private func encodeComputeRenderStage(
        commandBuffer: MTLCommandBuffer,
        frame: DepthFirstViewResources,
        targetColor: MTLTexture,
        targetDepth: MTLTexture?,
        gaussianCount: Int,
        width: Int,
        height: Int,
        tilesX: Int,
        tilesY: Int,
        tileCount: Int
    ) {
        dispatchEncoder.encodePrepareRenderDispatch(
            commandBuffer: commandBuffer,
            activeTileCount: frame.activeTileCount,
            dispatchArgs: frame.dispatchArgs,
            tileCount: UInt32(tileCount)
        )

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
}

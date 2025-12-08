import Foundation
@preconcurrency import Metal
import RendererTypes
import simd

public final class DepthFirstRenderer: GaussianRenderer, @unchecked Sendable {
    private static let maxSupportedGaussians = 10_000_000
    private static let tileWidth = 16
    private static let tileHeight = 16
    private static let maxPerTile = 2048

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

        // Compute limits and layout
        self.limits = RendererLimits(
            from: config,
            tileWidth: DepthFirstRenderer.tileWidth,
            tileHeight: DepthFirstRenderer.tileHeight,
            maxPerTile: DepthFirstRenderer.maxPerTile
        )

        let layout = DepthFirstRenderer.computeLayout(
            limits: limits,
            radixBlockSize: radixBlockSize,
            radixGrainSize: radixGrainSize
        )

        self.stereoResources = try DepthFirstMultiViewResources(device: device, layout: layout)
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
            maxPerTile: UInt32(DepthFirstRenderer.maxPerTile),
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

    private static func computeLayout(
        limits: RendererLimits,
        radixBlockSize: Int,
        radixGrainSize: Int
    ) -> DepthFirstResourceLayout {
        let maxGaussians = limits.maxGaussians
        let tileCount = limits.maxTileCount

        // Instance capacity: each gaussian can touch up to 32 tiles on average
        let maxInstances = min(maxGaussians * 32, tileCount * limits.maxPerTile)

        // Padded capacities for radix sort alignment
        let radixAlignment = radixBlockSize * radixGrainSize
        let paddedGaussianCapacity = ((maxGaussians + radixAlignment - 1) / radixAlignment) * radixAlignment
        let paddedInstanceCapacity = ((maxInstances + radixAlignment - 1) / radixAlignment) * radixAlignment

        // Radix sort histogram size
        let radix = 256
        let gaussianGridSize = max(1, (paddedGaussianCapacity + radixAlignment - 1) / radixAlignment)
        let instanceGridSize = max(1, (paddedInstanceCapacity + radixAlignment - 1) / radixAlignment)

        // Prefix sum block sums size
        let prefixBlockSize = 256
        let prefixNumBlocks = (maxGaussians + prefixBlockSize - 1) / prefixBlockSize
        let prefixLevel2Blocks = (prefixNumBlocks + prefixBlockSize - 1) / prefixBlockSize
        let prefixBlockSumsSize = (prefixNumBlocks + 1 + prefixLevel2Blocks + 1) * MemoryLayout<UInt32>.stride

        var bufferAllocations: [(label: String, length: Int, options: MTLResourceOptions)] = []
        func add(_ label: String, _ length: Int, _ opts: MTLResourceOptions = .storageModePrivate) {
            bufferAllocations.append((label, max(1, length), opts))
        }

        // Per-gaussian buffers
        add("RenderData", maxGaussians * MemoryLayout<GaussianRenderData>.stride)
        add("Bounds", maxGaussians * MemoryLayout<SIMD4<Int32>>.stride)
        add("DepthKeys", paddedGaussianCapacity * MemoryLayout<UInt32>.stride)
        add("PrimitiveIndices", paddedGaussianCapacity * MemoryLayout<Int32>.stride)
        add("NTouchedTiles", maxGaussians * MemoryLayout<UInt32>.stride)
        add("OrderedTileCounts", maxGaussians * MemoryLayout<UInt32>.stride)

        // Counters (shared for CPU readback)
        add("VisibleCount", MemoryLayout<UInt32>.stride, .storageModeShared)
        add("TotalInstances", MemoryLayout<UInt32>.stride, .storageModeShared)
        add("ActiveTileCount", MemoryLayout<UInt32>.stride, .storageModeShared)
        add("Header", 32, .storageModeShared) // DepthFirstHeader
        add("DispatchArgs", DepthFirstDispatchSlot.count * MemoryLayout<DispatchIndirectArgsSwift>.stride)

        // Per-instance buffers (tile IDs are 16-bit - max 65535 tiles)
        add("InstanceTileIds", paddedInstanceCapacity * MemoryLayout<UInt16>.stride)
        add("InstanceGaussianIndices", paddedInstanceCapacity * MemoryLayout<Int32>.stride)

        // Tile headers
        add("TileHeaders", tileCount * MemoryLayout<GaussianHeader>.stride)
        add("ActiveTiles", tileCount * MemoryLayout<UInt32>.stride)

        // Depth sort buffers
        add("DepthSortHistogram", gaussianGridSize * radix * MemoryLayout<UInt32>.stride)
        add("DepthSortBlockSums", gaussianGridSize * MemoryLayout<UInt32>.stride)
        add("DepthSortScannedHist", gaussianGridSize * radix * MemoryLayout<UInt32>.stride)
        add("DepthSortScratchKeys", paddedGaussianCapacity * MemoryLayout<UInt32>.stride)
        add("DepthSortScratchPayload", paddedGaussianCapacity * MemoryLayout<Int32>.stride)

        // Tile sort buffers (stable radix sort for tile IDs)
        add("TileSortHistogram", instanceGridSize * radix * MemoryLayout<UInt32>.stride)
        add("TileSortBlockSums", instanceGridSize * MemoryLayout<UInt32>.stride)
        add("TileSortScannedHist", instanceGridSize * radix * MemoryLayout<UInt32>.stride)
        add("TileSortScratchTileIds", paddedInstanceCapacity * MemoryLayout<UInt16>.stride)
        add("TileSortScratchIndices", paddedInstanceCapacity * MemoryLayout<Int32>.stride)

        // Prefix sum block sums
        add("PrefixSumBlockSums", prefixBlockSumsSize)

        return DepthFirstResourceLayout(
            limits: limits,
            maxGaussians: maxGaussians,
            maxInstances: maxInstances,
            paddedGaussianCapacity: paddedGaussianCapacity,
            paddedInstanceCapacity: paddedInstanceCapacity,
            bufferAllocations: bufferAllocations
        )
    }
}

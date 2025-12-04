import Metal
import simd

/// Depth-First Gaussian Splatting Renderer (FastGS/Splatshop style)
///
/// Single-phase sorting approach using optimized RadixSortEncoder:
/// 1. Project + compact visible gaussians
/// 2. Create instances with fused (tile, depth) keys
/// 3. Radix sort instances using RadixSortEncoder (same as GlobalSort)
/// 4. Render (already sorted by tile then depth!)
///
/// Key feature: ZERO waitUntilCompleted - all dispatches are indirect!
public final class DepthFirstRenderer: GaussianRenderer, @unchecked Sendable {
    public let device: MTLDevice
    private let library: MTLLibrary
    private let mainLibrary: MTLLibrary  // For RadixSortEncoder kernels
    private let config: RendererConfig

    // RadixSortEncoder (uses optimized kernels from main library)
    private let radixSortEncoder: RadixSortEncoder
    private let radixBuffers: RadixBufferSet

    // Pipeline states
    private let clearPipeline: MTLComputePipelineState
    private let projectFloatPipeline: MTLComputePipelineState
    private let projectHalfPipeline: MTLComputePipelineState?
    private let prepareVisibleDispatchPipeline: MTLComputePipelineState
    private let prepareRadixDispatchPipeline: MTLComputePipelineState
    private let storeVisibleInfoPipeline: MTLComputePipelineState
    private let computeTotalInstancesPipeline: MTLComputePipelineState
    private let genDepthKeysPipeline: MTLComputePipelineState
    private let radixHistogramPipeline: MTLComputePipelineState
    private let radixScanHistogramPipeline: MTLComputePipelineState
    private let radixScatterPipeline: MTLComputePipelineState
    private let applyOrderPipeline: MTLComputePipelineState
    private let prefixScanPipeline: MTLComputePipelineState
    private let scanPartialSumsPipeline: MTLComputePipelineState
    private let finalizeScanPipeline: MTLComputePipelineState
    private let createInstancesPipeline: MTLComputePipelineState
    private let createInstancesV2Pipeline: MTLComputePipelineState  // Outputs SIMD2<UInt32>
    private let createInstancesTileOnlyPipeline: MTLComputePipelineState  // FastGS: tile-only keys
    private let countTileInstancesPipeline: MTLComputePipelineState  // FastGS: count for counting sort
    private let scatterTileInstancesPipeline: MTLComputePipelineState  // FastGS: stable scatter
    private let extractRangesTileOnlyPipeline: MTLComputePipelineState  // FastGS: extract from tile keys
    private let countTilesPipeline: MTLComputePipelineState
    private let scatterTilesPipeline: MTLComputePipelineState
    private let zeroCountersPipeline: MTLComputePipelineState
    private let extractRangesPipeline: MTLComputePipelineState
    private let clearTileRangesPipeline: MTLComputePipelineState  // Clear tile ranges before extraction
    private let gatherGaussiansPipeline: MTLComputePipelineState  // Gather sorted Gaussians for sequential access
    private let renderSortedPipeline: MTLComputePipelineState     // Render with sequential access
    private let renderPipeline: MTLComputePipelineState           // Legacy render with indirection
    private let debugValidationPipeline: MTLComputePipelineState? // Debug pipeline (optional)

    // Buffers
    private let compactedHeader: MTLBuffer
    private let compactedGaussians: MTLBuffer

    // Indirect dispatch buffers
    private let visibleDispatchArgs: MTLBuffer      // For visible count based dispatch
    private let radixDispatchArgs: MTLBuffer        // For radix sort blocks (depth sort)
    private let instanceDispatchArgs: MTLBuffer    // For instance count based dispatch
    private let instanceRadixDispatchArgs: MTLBuffer // For instance radix sort blocks
    private let radixSortDispatchArgs: MTLBuffer   // For RadixSortEncoder (7 slots)
    private let instanceHeader: MTLBuffer          // TileAssignmentHeader for RadixSortEncoder
    private let visibleInfo: MTLBuffer             // [count, numBlocks, histogramSize]
    private let instanceInfo: MTLBuffer            // [total, numBlocks, histogramSize] for instance radix sort

    // Depth sort buffers
    private let depthKeys: MTLBuffer
    private let depthSortedIndices: MTLBuffer
    private let tempKeys: MTLBuffer
    private let tempVals: MTLBuffer
    private let radixHistogram: MTLBuffer

    // Tile counts and offsets
    private let gaussianTileCounts: MTLBuffer
    private let gaussianOffsets: MTLBuffer
    private let gaussianPartialSums: MTLBuffer

    // Instance buffers (uint32 combined keys: tileKey << 20 | depthOrder) - legacy
    private let instanceSortKeys: MTLBuffer
    private let instanceGaussianIdx: MTLBuffer
    private let sortedSortKeys: MTLBuffer
    private let sortedGaussianIdx: MTLBuffer
    private let tempInstanceKeys: MTLBuffer  // For radix sort ping-pong
    private let tempInstanceVals: MTLBuffer

    // FastGS instance buffers (tile-only keys - ushort)
    private let instanceTileKeys: MTLBuffer      // ushort tile keys
    private let instancePrimitiveIdx: MTLBuffer  // uint primitive indices
    private let sortedTileKeys: MTLBuffer        // Output of counting sort
    private let sortedPrimitiveIdx: MTLBuffer    // Output of counting sort

    // Tile sort buffers
    private let tileCounts: MTLBuffer
    private let tileOffsets: MTLBuffer
    private let tileCounters: MTLBuffer
    private let tilePartialSums: MTLBuffer
    private let tileRanges: MTLBuffer

    // Sorted Gaussian buffer for sequential render access
    private let sortedCompacted: MTLBuffer

    // Output textures
    private var colorTexture: MTLTexture?
    private var depthTexture: MTLTexture?

    // Constants
    private let radixBlockSize = 256
    private let radixGrainSize = 4
    private let prefixBlockSize = 256
    private let prefixGrainSize = 4
    // Use 32x16 tiles like GlobalSort for matching render kernel
    private let tileWidth = 32
    private let tileHeight = 16
    private var maxInstanceCapacity: UInt32 = 0

    public init(device: MTLDevice, config: RendererConfig = RendererConfig()) throws {
        self.device = device
        self.config = config

        // Load DepthFirst shader library
        guard let libraryURL = Bundle.module.url(forResource: "DepthFirstShaders", withExtension: "metallib"),
              let library = try? device.makeLibrary(URL: libraryURL) else {
            throw RendererError.failedToCreatePipeline("Failed to load DepthFirstShaders.metallib")
        }
        self.library = library

        // Load main library for RadixSortEncoder (optimized radix sort kernels)
        guard let mainLibraryURL = Bundle.module.url(forResource: "GaussianMetalRenderer", withExtension: "metallib"),
              let mainLib = try? device.makeLibrary(URL: mainLibraryURL) else {
            throw RendererError.failedToCreatePipeline("Failed to load GaussianMetalRenderer.metallib for radix sort")
        }
        self.mainLibrary = mainLib

        // Create RadixSortEncoder using optimized kernels
        self.radixSortEncoder = try RadixSortEncoder(device: device, library: mainLib)

        // Create function constant values for SH_DEGREE
        let functionConstants = MTLFunctionConstantValues()
        var defaultDegree: UInt32 = 3
        functionConstants.setConstantValue(&defaultDegree, type: .uint, index: 0)  // SH_DEGREE

        // Create pipeline states
        guard let clearFn = library.makeFunction(name: "dfClear"),
              let projectFloatFn = try? library.makeFunction(name: "dfProjectCompactFloat", constantValues: functionConstants),
              let prepareVisibleFn = library.makeFunction(name: "dfPrepareVisibleDispatch"),
              let prepareRadixFn = library.makeFunction(name: "dfPrepareRadixDispatch"),
              let storeInfoFn = library.makeFunction(name: "dfStoreVisibleInfo"),
              let computeInstFn = library.makeFunction(name: "dfComputeTotalInstances"),
              let genKeysFn = library.makeFunction(name: "dfGenDepthKeys"),
              let radixHistFn = library.makeFunction(name: "dfRadixHistogram"),
              let radixScanFn = library.makeFunction(name: "dfRadixScanHistogram"),
              let radixScatterFn = library.makeFunction(name: "dfRadixScatter"),
              let applyOrderFn = library.makeFunction(name: "dfApplyOrder"),
              let prefixScanFn = library.makeFunction(name: "dfPrefixScan"),
              let scanPartialsFn = library.makeFunction(name: "dfScanPartialSums"),
              let finalizeFn = library.makeFunction(name: "dfFinalizeScan"),
              let createInstFn = library.makeFunction(name: "dfCreateInstances"),
              let createInstV2Fn = library.makeFunction(name: "dfCreateInstancesV2"),
              let createInstTileOnlyFn = library.makeFunction(name: "dfCreateInstancesTileOnly"),
              let countTileInstFn = library.makeFunction(name: "dfCountTileInstances"),
              let scatterTileInstFn = library.makeFunction(name: "dfScatterTileInstances"),
              let extractRangesTileOnlyFn = library.makeFunction(name: "dfExtractRangesTileOnly"),
              let countTilesFn = library.makeFunction(name: "dfCountTiles"),
              let scatterTilesFn = library.makeFunction(name: "dfScatterTiles"),
              let zeroCountersFn = library.makeFunction(name: "dfZeroCounters"),
              let extractRangesFn = library.makeFunction(name: "dfExtractRanges"),
              let clearTileRangesFn = library.makeFunction(name: "dfClearTileRanges"),
              let gatherGaussiansFn = library.makeFunction(name: "dfGatherGaussians"),
              let renderSortedFn = library.makeFunction(name: "dfRenderSorted"),
              let renderFn = library.makeFunction(name: "dfRender") else {
            throw RendererError.failedToCreatePipeline("Missing depth-first kernel functions")
        }

        self.clearPipeline = try device.makeComputePipelineState(function: clearFn)
        self.projectFloatPipeline = try device.makeComputePipelineState(function: projectFloatFn)
        self.prepareVisibleDispatchPipeline = try device.makeComputePipelineState(function: prepareVisibleFn)
        self.prepareRadixDispatchPipeline = try device.makeComputePipelineState(function: prepareRadixFn)
        self.storeVisibleInfoPipeline = try device.makeComputePipelineState(function: storeInfoFn)
        self.computeTotalInstancesPipeline = try device.makeComputePipelineState(function: computeInstFn)
        self.genDepthKeysPipeline = try device.makeComputePipelineState(function: genKeysFn)
        self.radixHistogramPipeline = try device.makeComputePipelineState(function: radixHistFn)
        self.radixScanHistogramPipeline = try device.makeComputePipelineState(function: radixScanFn)
        self.radixScatterPipeline = try device.makeComputePipelineState(function: radixScatterFn)
        self.applyOrderPipeline = try device.makeComputePipelineState(function: applyOrderFn)
        self.prefixScanPipeline = try device.makeComputePipelineState(function: prefixScanFn)
        self.scanPartialSumsPipeline = try device.makeComputePipelineState(function: scanPartialsFn)
        self.finalizeScanPipeline = try device.makeComputePipelineState(function: finalizeFn)
        self.createInstancesPipeline = try device.makeComputePipelineState(function: createInstFn)
        self.createInstancesV2Pipeline = try device.makeComputePipelineState(function: createInstV2Fn)
        self.createInstancesTileOnlyPipeline = try device.makeComputePipelineState(function: createInstTileOnlyFn)
        self.countTileInstancesPipeline = try device.makeComputePipelineState(function: countTileInstFn)
        self.scatterTileInstancesPipeline = try device.makeComputePipelineState(function: scatterTileInstFn)
        self.extractRangesTileOnlyPipeline = try device.makeComputePipelineState(function: extractRangesTileOnlyFn)
        self.countTilesPipeline = try device.makeComputePipelineState(function: countTilesFn)
        self.scatterTilesPipeline = try device.makeComputePipelineState(function: scatterTilesFn)
        self.zeroCountersPipeline = try device.makeComputePipelineState(function: zeroCountersFn)
        self.extractRangesPipeline = try device.makeComputePipelineState(function: extractRangesFn)
        self.clearTileRangesPipeline = try device.makeComputePipelineState(function: clearTileRangesFn)
        self.gatherGaussiansPipeline = try device.makeComputePipelineState(function: gatherGaussiansFn)
        self.renderSortedPipeline = try device.makeComputePipelineState(function: renderSortedFn)
        self.renderPipeline = try device.makeComputePipelineState(function: renderFn)

        // Half precision project (optional)
        if let projectHalfFn = try? library.makeFunction(name: "dfProjectCompactHalf", constantValues: functionConstants) {
            self.projectHalfPipeline = try? device.makeComputePipelineState(function: projectHalfFn)
        } else {
            self.projectHalfPipeline = nil
        }

        // Debug validation pipeline (optional)
        if let debugFn = library.makeFunction(name: "dfDebugValidation") {
            self.debugValidationPipeline = try? device.makeComputePipelineState(function: debugFn)
        } else {
            self.debugValidationPipeline = nil
        }

        // Calculate buffer sizes
        let maxGaussians = config.maxGaussians
        let maxTiles = (config.maxWidth / tileWidth + 1) * (config.maxHeight / tileHeight + 1)
        // Max instances = gaussians * avg tiles per gaussian
        // Using 64 tiles per gaussian to handle zoomed-out views where many gaussians are visible
        // and each gaussian can cover many tiles. The previous value of 16 caused overflow when
        // viewing large scenes from a distance, resulting in random gaussians being dropped.
        let maxInstances = maxGaussians * 64

        let elementsPerBlock = radixBlockSize * radixGrainSize
        let numBlocks = (maxGaussians + elementsPerBlock - 1) / elementsPerBlock
        // Histogram needs to be sized for the LARGER of gaussians or instances
        let instanceNumBlocks = (maxInstances + elementsPerBlock - 1) / elementsPerBlock
        let histogramSize = 256 * max(numBlocks, instanceNumBlocks)

        let elementsPerGroup = prefixBlockSize * prefixGrainSize
        let numGroups = (maxGaussians + elementsPerGroup - 1) / elementsPerGroup
        let tileNumGroups = (maxTiles + elementsPerGroup - 1) / elementsPerGroup

        // Allocate buffers
        self.compactedHeader = device.makeBuffer(length: MemoryLayout<TileAssignmentHeader>.stride, options: .storageModeShared)!
        self.compactedGaussians = device.makeBuffer(length: maxGaussians * MemoryLayout<CompactedGaussian>.stride, options: .storageModePrivate)!

        // Indirect dispatch buffers (12 bytes for MTLDispatchThreadgroupsIndirectArguments)
        self.visibleDispatchArgs = device.makeBuffer(length: 12, options: .storageModePrivate)!
        self.radixDispatchArgs = device.makeBuffer(length: 12, options: .storageModePrivate)!
        self.instanceDispatchArgs = device.makeBuffer(length: 12, options: .storageModePrivate)!
        self.instanceRadixDispatchArgs = device.makeBuffer(length: 12, options: .storageModePrivate)!
        // RadixSortEncoder needs 7 dispatch arg slots (same as GlobalSort)
        self.radixSortDispatchArgs = device.makeBuffer(length: 7 * 12, options: .storageModePrivate)!
        // TileAssignmentHeader for RadixSortEncoder (instance count)
        self.instanceHeader = device.makeBuffer(length: MemoryLayout<TileAssignmentHeader>.stride, options: .storageModeShared)!
        self.visibleInfo = device.makeBuffer(length: 3 * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        self.instanceInfo = device.makeBuffer(length: 3 * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!

        self.depthKeys = device.makeBuffer(length: maxGaussians * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        self.depthSortedIndices = device.makeBuffer(length: maxGaussians * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        self.tempKeys = device.makeBuffer(length: maxGaussians * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        self.tempVals = device.makeBuffer(length: maxGaussians * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        self.radixHistogram = device.makeBuffer(length: histogramSize * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!

        self.gaussianTileCounts = device.makeBuffer(length: maxGaussians * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        self.gaussianOffsets = device.makeBuffer(length: maxGaussians * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        self.gaussianPartialSums = device.makeBuffer(length: numGroups * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!

        // Instance buffers - SIMD2<UInt32> keys for RadixSortEncoder (tileId, depth)
        self.maxInstanceCapacity = UInt32(maxInstances)
        self.instanceSortKeys = device.makeBuffer(length: maxInstances * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModePrivate)!
        self.instanceGaussianIdx = device.makeBuffer(length: maxInstances * MemoryLayout<Int32>.stride, options: .storageModePrivate)!
        self.sortedSortKeys = device.makeBuffer(length: maxInstances * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        self.sortedGaussianIdx = device.makeBuffer(length: maxInstances * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        self.tempInstanceKeys = device.makeBuffer(length: maxInstances * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        self.tempInstanceVals = device.makeBuffer(length: maxInstances * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!

        self.tileCounts = device.makeBuffer(length: maxTiles * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        self.tileOffsets = device.makeBuffer(length: maxTiles * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        self.tileCounters = device.makeBuffer(length: maxTiles * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        self.tilePartialSums = device.makeBuffer(length: tileNumGroups * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        self.tileRanges = device.makeBuffer(length: maxTiles * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModePrivate)!

        // Sorted Gaussian buffer for sequential render access
        self.sortedCompacted = device.makeBuffer(length: maxInstances * MemoryLayout<CompactedGaussian>.stride, options: .storageModePrivate)!

        // RadixBufferSet for RadixSortEncoder (same as GlobalSort)
        let radixHistSize = 256 * instanceNumBlocks * MemoryLayout<UInt32>.stride
        let blockSumsSize = ((256 * instanceNumBlocks + 255) / 256) * MemoryLayout<UInt32>.stride
        self.radixBuffers = RadixBufferSet(
            histogram: device.makeBuffer(length: radixHistSize, options: .storageModePrivate)!,
            blockSums: device.makeBuffer(length: blockSumsSize, options: .storageModePrivate)!,
            scannedHistogram: device.makeBuffer(length: radixHistSize, options: .storageModePrivate)!,
            fusedKeys: device.makeBuffer(length: maxInstances * MemoryLayout<UInt64>.stride, options: .storageModePrivate)!,
            scratchKeys: device.makeBuffer(length: maxInstances * MemoryLayout<UInt64>.stride, options: .storageModePrivate)!,
            scratchPayload: device.makeBuffer(length: maxInstances * MemoryLayout<Int32>.stride, options: .storageModePrivate)!
        )

        // FastGS instance buffers (tile-only keys - ushort for 16-bit tile indices)
        self.instanceTileKeys = device.makeBuffer(length: maxInstances * MemoryLayout<UInt16>.stride, options: .storageModePrivate)!
        self.instancePrimitiveIdx = device.makeBuffer(length: maxInstances * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
        self.sortedTileKeys = device.makeBuffer(length: maxInstances * MemoryLayout<UInt16>.stride, options: .storageModePrivate)!
        self.sortedPrimitiveIdx = device.makeBuffer(length: maxInstances * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!
    }

    // MARK: - GaussianRenderer Protocol

    public func render(
        toTexture commandBuffer: MTLCommandBuffer,
        input: GaussianInput,
        camera: CameraParams,
        width: Int,
        height: Int,
        whiteBackground: Bool,
        mortonSorted: Bool
    ) -> TextureRenderResult? {
        let tilesX = (width + tileWidth - 1) / tileWidth
        let tilesY = (height + tileHeight - 1) / tileHeight
        let nTiles = tilesX * tilesY

        // Ensure textures
        ensureTextures(width: width, height: height)
        guard let colorTex = colorTexture, let depthTex = depthTexture else { return nil }

        // Create camera uniforms
        var cameraUniforms = CameraUniforms(
            viewMatrix: camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            cameraCenter: camera.position,
            pixelFactor: 1.0,
            focalX: camera.focalX,
            focalY: camera.focalY,
            width: Float(width),
            height: Float(height),
            nearPlane: camera.near,
            farPlane: camera.far,
            shComponents: UInt32(input.shComponents),
            gaussianCount: UInt32(input.gaussianCount),
            padding0: 0,
            padding1: 0
        )

        var binningParams = TileBinningParamsLocal(
            gaussianCount: UInt32(input.gaussianCount),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            surfaceWidth: UInt32(width),
            surfaceHeight: UInt32(height),
            maxCapacity: UInt32(config.maxGaussians)
        )

        var elementsPerBlock = UInt32(radixBlockSize * radixGrainSize)
        _ = UInt32(prefixBlockSize * prefixGrainSize)  // elementsPerGroup (used in prefix scan)
        _ = UInt32(nTiles)  // nTilesU (passed to shaders via counting sort)

        // === 1. CLEAR HEADER ===
        var maxCompactedU = UInt32(config.maxGaussians)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_Clear"
            encoder.setComputePipelineState(clearPipeline)
            encoder.setBuffer(compactedHeader, offset: 0, index: 0)
            encoder.setBytes(&maxCompactedU, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // === 2. PROJECT + COMPACT ===
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_ProjectCompact"
            encoder.setComputePipelineState(projectFloatPipeline)
            encoder.setBuffer(input.gaussians, offset: 0, index: 0)
            encoder.setBuffer(input.harmonics, offset: 0, index: 1)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 2)
            encoder.setBuffer(compactedHeader, offset: 0, index: 3)
            encoder.setBytes(&cameraUniforms, length: MemoryLayout<CameraUniforms>.stride, index: 4)
            encoder.setBytes(&binningParams, length: MemoryLayout<TileBinningParamsLocal>.stride, index: 5)
            let tg = MTLSize(width: projectFloatPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(MTLSize(width: input.gaussianCount, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === 3. PREPARE INDIRECT DISPATCH ARGS ===
        var threadsPerGroup = UInt32(genDepthKeysPipeline.threadExecutionWidth)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_PrepareVisibleDispatch"
            encoder.setComputePipelineState(prepareVisibleDispatchPipeline)
            encoder.setBuffer(compactedHeader, offset: 0, index: 0)
            encoder.setBuffer(visibleDispatchArgs, offset: 0, index: 1)
            encoder.setBytes(&threadsPerGroup, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_PrepareRadixDispatch"
            encoder.setComputePipelineState(prepareRadixDispatchPipeline)
            encoder.setBuffer(compactedHeader, offset: 0, index: 0)
            encoder.setBuffer(radixDispatchArgs, offset: 0, index: 1)
            encoder.setBytes(&elementsPerBlock, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_StoreVisibleInfo"
            encoder.setComputePipelineState(storeVisibleInfoPipeline)
            encoder.setBuffer(compactedHeader, offset: 0, index: 0)
            encoder.setBuffer(visibleInfo, offset: 0, index: 1)
            encoder.setBytes(&elementsPerBlock, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // === 4. GENERATE DEPTH KEYS + INDICES (for depth sort) ===
        // FastGS algorithm: Sort primitives by depth FIRST, then use tile-only keys for instances
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_GenDepthKeys"
            encoder.setComputePipelineState(genDepthKeysPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(depthKeys, offset: 0, index: 1)
            encoder.setBuffer(depthSortedIndices, offset: 0, index: 2)  // Initialize as [0,1,2,...]
            encoder.setBuffer(compactedHeader, offset: 0, index: 3)
            let tg = MTLSize(width: Int(threadsPerGroup), height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: visibleDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === 5. RADIX SORT PRIMITIVES BY DEPTH (4 passes on N_visible - small!) ===
        // This is the key insight of FastGS: sort the PRIMITIVES by depth first,
        // then instances inherit depth order automatically!
        encodeDepthSort(commandBuffer)

        // === 6. APPLY DEPTH ORDERING → TILE COUNTS (indirect) ===
        // Now depthSortedIndices contains primitives sorted by depth (front-to-back)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_ApplyOrder"
            encoder.setComputePipelineState(applyOrderPipeline)
            encoder.setBuffer(depthSortedIndices, offset: 0, index: 0)  // Depth-sorted indices!
            encoder.setBuffer(compactedGaussians, offset: 0, index: 1)
            encoder.setBuffer(gaussianTileCounts, offset: 0, index: 2)
            encoder.setBuffer(compactedHeader, offset: 0, index: 3)
            let tg = MTLSize(width: applyOrderPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: visibleDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === 7. PREFIX SUM GAUSSIAN TILE COUNTS (indirect) ===
        encodePrefixScanIndirect(commandBuffer, input: gaussianTileCounts, output: gaussianOffsets,
                                  partials: gaussianPartialSums, dispatchArgs: radixDispatchArgs)

        // === 8. COMPUTE TOTAL INSTANCES + PREPARE DISPATCH ARGS ===
        var extractRangesThreadsPerGroup = UInt32(extractRangesTileOnlyPipeline.threadExecutionWidth)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_ComputeTotalInstances"
            encoder.setComputePipelineState(computeTotalInstancesPipeline)
            encoder.setBuffer(gaussianOffsets, offset: 0, index: 0)
            encoder.setBuffer(gaussianTileCounts, offset: 0, index: 1)
            encoder.setBuffer(compactedHeader, offset: 0, index: 2)
            encoder.setBuffer(instanceInfo, offset: 0, index: 3)  // [total, numBlocks, histogramSize]
            encoder.setBuffer(instanceDispatchArgs, offset: 0, index: 4)
            encoder.setBytes(&extractRangesThreadsPerGroup, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBuffer(instanceRadixDispatchArgs, offset: 0, index: 6)
            encoder.setBuffer(instanceHeader, offset: 0, index: 7)
            encoder.setBuffer(radixSortDispatchArgs, offset: 0, index: 8)
            var maxCap = maxInstanceCapacity
            encoder.setBytes(&maxCap, length: MemoryLayout<UInt32>.stride, index: 9)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // === 9. CREATE INSTANCES WITH FUSED (TILE, DEPTH) KEYS ===
        // Since primitives are depth-sorted, the depth component ensures correct ordering
        // Pre-fill with UINT32_MAX for tileId - garbage entries will sort to end
        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            blitEncoder.label = "DF_FillSentinel"
            blitEncoder.fill(buffer: instanceSortKeys, range: 0..<instanceSortKeys.length, value: 0xFF)
            blitEncoder.endEncoding()
        }

        var tilesXU = UInt32(tilesX)
        var tileW = Int32(tileWidth)
        var tileH = Int32(tileHeight)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_CreateInstancesV2"
            encoder.setComputePipelineState(createInstancesV2Pipeline)
            encoder.setBuffer(depthSortedIndices, offset: 0, index: 0)  // Depth-sorted primitive indices!
            encoder.setBuffer(compactedGaussians, offset: 0, index: 1)
            encoder.setBuffer(gaussianOffsets, offset: 0, index: 2)
            encoder.setBuffer(instanceSortKeys, offset: 0, index: 3)  // SIMD2<UInt32> keys (tile, depth)
            encoder.setBuffer(instanceGaussianIdx, offset: 0, index: 4)  // Int32 indices
            encoder.setBytes(&tilesXU, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&tileW, length: MemoryLayout<Int32>.stride, index: 6)
            encoder.setBytes(&tileH, length: MemoryLayout<Int32>.stride, index: 7)
            encoder.setBuffer(compactedHeader, offset: 0, index: 8)
            var maxCap2 = maxInstanceCapacity
            encoder.setBytes(&maxCap2, length: MemoryLayout<UInt32>.stride, index: 9)
            let tg = MTLSize(width: createInstancesV2Pipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: visibleDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === 10. RADIX SORT BY TILE+DEPTH (using RadixSortEncoder) ===
        // Only 2 passes needed for tile (16-bit), depth provides ordering within tile
        let offsets = (
            fuse: 0 * 12,
            unpack: 1 * 12,
            histogram: 2 * 12,
            scanBlocks: 3 * 12,
            exclusive: 4 * 12,
            apply: 5 * 12,
            scatter: 6 * 12
        )
        radixSortEncoder.encode(
            commandBuffer: commandBuffer,
            keyBuffer: instanceSortKeys,       // SIMD2<UInt32> keys (tile, depth)
            sortedIndices: instanceGaussianIdx, // Int32 indices (payload)
            header: instanceHeader,             // TileAssignmentHeader with instance count
            dispatchArgs: radixSortDispatchArgs,
            radixBuffers: radixBuffers,
            offsets: offsets,
            tileCount: nTiles
        )

        // === 11. ZERO TILE RANGES + EXTRACT RANGES ===
        var nTilesU = UInt32(nTiles)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_ClearTileRanges"
            encoder.setComputePipelineState(clearTileRangesPipeline)
            encoder.setBuffer(tileRanges, offset: 0, index: 0)
            encoder.setBytes(&nTilesU, length: MemoryLayout<UInt32>.stride, index: 1)
            let tpg = clearTileRangesPipeline.threadExecutionWidth
            let numGroups = (nTiles + tpg - 1) / tpg
            encoder.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // Extract tile ranges from sorted SIMD2<UInt32> keys
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_ExtractRangesV2"
            encoder.setComputePipelineState(extractRangesPipeline)
            encoder.setBuffer(instanceSortKeys, offset: 0, index: 0)  // Sorted SIMD2 keys
            encoder.setBuffer(tileRanges, offset: 0, index: 1)
            encoder.setBuffer(instanceInfo, offset: 0, index: 2)  // [0]=total
            encoder.setBytes(&nTilesU, length: MemoryLayout<UInt32>.stride, index: 3)
            let tg = MTLSize(width: extractRangesPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: instanceDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === 12. RENDER (using sorted gaussian indices) ===
        var renderParams = RenderParams(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            maxPerTile: 0,
            whiteBackground: whiteBackground ? 1 : 0,
            activeTileCount: 0,
            gaussianCount: 0
        )

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_RenderSorted"
            encoder.setComputePipelineState(renderSortedPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(tileRanges, offset: 0, index: 1)
            encoder.setBuffer(instanceGaussianIdx, offset: 0, index: 2)  // Sorted indices
            encoder.setTexture(colorTex, index: 0)
            encoder.setBytes(&renderParams, length: MemoryLayout<RenderParams>.stride, index: 3)
            encoder.dispatchThreadgroups(MTLSize(width: tilesX, height: tilesY, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            encoder.endEncoding()
        }

        return TextureRenderResult(color: colorTex, depth: depthTex)
    }

    public func render(
        toBuffer commandBuffer: MTLCommandBuffer,
        input: GaussianInput,
        camera: CameraParams,
        width: Int,
        height: Int,
        whiteBackground: Bool,
        mortonSorted: Bool
    ) -> BufferRenderResult? {
        // Not implemented - use texture render
        return nil
    }

    /// Get the number of visible gaussians from the last render
    public func getVisibleCount() -> UInt32 {
        let ptr = compactedHeader.contents().bindMemory(to: TileAssignmentHeader.self, capacity: 1)
        return ptr.pointee.totalAssignments
    }

    /// Get instance count from instanceInfo
    public func getInstanceCount() -> UInt32 {
        // instanceInfo is private, but we can get it from instanceHeader which is shared
        let ptr = instanceHeader.contents().bindMemory(to: TileAssignmentHeader.self, capacity: 1)
        return ptr.pointee.totalAssignments
    }

    /// Get max instance capacity (for overflow detection)
    public func getMaxInstanceCapacity() -> UInt32 {
        return maxInstanceCapacity
    }

    /// Check if instance buffer overflow occurred (call after render)
    public func hasOverflow() -> Bool {
        let ptr = instanceHeader.contents().bindMemory(to: TileAssignmentHeader.self, capacity: 1)
        return ptr.pointee.overflow != 0
    }

    /// Print comprehensive pipeline diagnostics (call after render)
    public func printDiagnostics() {
        let visibleCount = getVisibleCount()
        let instanceCount = getInstanceCount()
        let maxCapacity = getMaxInstanceCapacity()
        let overflow = hasOverflow()

        print("=== DepthFirstRenderer Diagnostics ===")
        print("Visible gaussians: \(visibleCount)")
        print("Total instances: \(instanceCount)")
        print("Max instance capacity: \(maxCapacity)")
        print("Capacity usage: \(String(format: "%.1f", Double(instanceCount) / Double(maxCapacity) * 100))%")

        if overflow {
            print("⚠️  WARNING: Instance buffer OVERFLOW! Some gaussians are missing tiles.")
            print("   This causes jittery/scattered rendering. Increase maxGaussians or zoom in.")
        }

        let avgTilesPerGaussian = visibleCount > 0 ? Double(instanceCount) / Double(visibleCount) : 0
        print("Avg tiles per gaussian: \(String(format: "%.2f", avgTilesPerGaussian))")
        print("=======================================")
    }

    /// Print detailed tile range diagnostics (call after render with queue)
    /// Shows distribution of gaussians per tile to identify issues
    public func printTileRangeDiagnostics(queue: MTLCommandQueue, tilesX: Int, tilesY: Int) {
        let nTiles = tilesX * tilesY
        let ranges = getTileRanges(queue: queue, count: nTiles)

        var emptyTiles = 0
        var singleGaussianTiles = 0
        var garbageTiles = 0
        var maxGaussiansPerTile: UInt32 = 0
        var totalNonEmpty = 0
        var totalGaussiansFromRanges: UInt32 = 0

        for i in 0..<nTiles {
            let range = ranges[i]
            let count = range.end >= range.start ? range.end - range.start : 0

            if range.end < range.start || count > 100000 {
                garbageTiles += 1
            } else if count == 0 {
                emptyTiles += 1
            } else {
                totalNonEmpty += 1
                totalGaussiansFromRanges += count
                if count == 1 {
                    singleGaussianTiles += 1
                }
                if count > maxGaussiansPerTile {
                    maxGaussiansPerTile = count
                }
            }
        }

        print("=== Tile Range Diagnostics ===")
        print("Total tiles: \(nTiles) (\(tilesX) x \(tilesY))")
        print("Empty tiles: \(emptyTiles)")
        print("Single-gaussian tiles: \(singleGaussianTiles)")
        print("Tiles with garbage ranges: \(garbageTiles)")
        print("Non-empty tiles: \(totalNonEmpty)")
        print("Max gaussians in one tile: \(maxGaussiansPerTile)")
        print("Total instances from ranges: \(totalGaussiansFromRanges)")
        print("Expected instances: \(getInstanceCount())")

        if totalGaussiansFromRanges != getInstanceCount() {
            print("⚠️  MISMATCH: Range sum (\(totalGaussiansFromRanges)) != instance count (\(getInstanceCount()))")
        }
        if garbageTiles > 0 {
            print("⚠️  WARNING: \(garbageTiles) tiles have garbage ranges!")
        }
        print("===============================")
    }

    /// Get tile ranges buffer for debugging (copies from GPU to CPU)
    public func getTileRanges(queue: MTLCommandQueue, count: Int) -> [(start: UInt32, end: UInt32)] {
        let size = count * MemoryLayout<SIMD2<UInt32>>.stride
        guard let readBuf = device.makeBuffer(length: size, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return [] }
        blit.copy(from: tileRanges, sourceOffset: 0, to: readBuf, destinationOffset: 0, size: size)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = readBuf.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: count)
        var result: [(UInt32, UInt32)] = []
        for i in 0..<count {
            result.append((ptr[i].x, ptr[i].y))
        }
        return result
    }

    /// Get sorted instance keys for debugging (copies from GPU to CPU)
    public func getInstanceSortKeys(queue: MTLCommandQueue, count: Int) -> [(tileId: UInt32, depth16: UInt32)] {
        let size = count * MemoryLayout<SIMD2<UInt32>>.stride
        guard let readBuf = device.makeBuffer(length: size, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return [] }
        blit.copy(from: instanceSortKeys, sourceOffset: 0, to: readBuf, destinationOffset: 0, size: size)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = readBuf.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: count)
        var result: [(UInt32, UInt32)] = []
        for i in 0..<count {
            result.append((ptr[i].x, ptr[i].y))
        }
        return result
    }

    /// Get sorted gaussian indices for debugging (copies from GPU to CPU)
    public func getInstanceGaussianIdx(queue: MTLCommandQueue, count: Int) -> [Int32] {
        let size = count * MemoryLayout<Int32>.stride
        guard let readBuf = device.makeBuffer(length: size, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return [] }
        blit.copy(from: instanceGaussianIdx, sourceOffset: 0, to: readBuf, destinationOffset: 0, size: size)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = readBuf.contents().bindMemory(to: Int32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Debug validation: Check if sorted keys are in order and count instances per tile
    /// Returns: (totalInstances, outOfOrderCount, firstOutOfOrderIdx, tileCounts[0..31], expectedStarts[0..31], expectedEnds[0..31])
    public func runDebugValidation(queue: MTLCommandQueue, maxTiles: Int) -> (total: UInt32, outOfOrder: UInt32, firstIdx: UInt32, tileCounts: [UInt32], expectedStarts: [UInt32], expectedEnds: [UInt32])? {
        guard let pipeline = debugValidationPipeline else {
            print("Warning: Debug validation pipeline not available")
            return nil
        }

        // Create debug output buffer (shared mode for CPU readback)
        // Layout: [0]=total, [1]=outOfOrder, [2]=firstIdx, [3]=tileAtFirst, [4]=tileBeforeFirst,
        //         [5-36]=tileCounts, [37-68]=starts, [69-100]=ends
        let debugSize = 101 * MemoryLayout<UInt32>.stride
        guard let debugBuffer = device.makeBuffer(length: debugSize, options: .storageModeShared) else {
            return nil
        }

        guard let cb = queue.makeCommandBuffer(),
              let encoder = cb.makeComputeCommandEncoder() else {
            return nil
        }

        var maxTilesU = UInt32(maxTiles)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(instanceSortKeys, offset: 0, index: 0)
        encoder.setBuffer(instanceInfo, offset: 0, index: 1)
        encoder.setBuffer(debugBuffer, offset: 0, index: 2)
        encoder.setBytes(&maxTilesU, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        // Read results
        let ptr = debugBuffer.contents().bindMemory(to: UInt32.self, capacity: 101)
        let total = ptr[0]
        let outOfOrder = ptr[1]
        let firstIdx = ptr[2]
        var tileCounts: [UInt32] = []
        var expectedStarts: [UInt32] = []
        var expectedEnds: [UInt32] = []
        for i in 0..<32 {
            tileCounts.append(ptr[5 + i])
            expectedStarts.append(ptr[37 + i])
            expectedEnds.append(ptr[69 + i])
        }

        return (total, outOfOrder, firstIdx, tileCounts, expectedStarts, expectedEnds)
    }

    /// Profile the renderer with per-step timing (for debugging)
    public func profileRender(
        queue: MTLCommandQueue,
        input: GaussianInput,
        camera: CameraParams,
        width: Int,
        height: Int
    ) -> [String: Double] {
        var timings: [String: Double] = [:]

        let tilesX = (width + tileWidth - 1) / tileWidth
        let tilesY = (height + tileHeight - 1) / tileHeight
        let nTiles = tilesX * tilesY

        ensureTextures(width: width, height: height)
        guard let colorTex = colorTexture else { return [:] }

        var cameraUniforms = CameraUniforms(
            viewMatrix: camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            cameraCenter: camera.position,
            pixelFactor: 1.0,
            focalX: camera.focalX,
            focalY: camera.focalY,
            width: Float(width),
            height: Float(height),
            nearPlane: camera.near,
            farPlane: camera.far,
            shComponents: UInt32(input.shComponents),
            gaussianCount: UInt32(input.gaussianCount),
            padding0: 0,
            padding1: 0
        )

        var binningParams = TileBinningParamsLocal(
            gaussianCount: UInt32(input.gaussianCount),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            surfaceWidth: UInt32(width),
            surfaceHeight: UInt32(height),
            maxCapacity: UInt32(config.maxGaussians)
        )

        var elementsPerBlock = UInt32(radixBlockSize * radixGrainSize)
        var threadsPerGroup = UInt32(genDepthKeysPipeline.threadExecutionWidth)
        var tilesXU = UInt32(tilesX)
        var tileW = Int32(tileWidth)
        var tileH = Int32(tileHeight)
        var extractRangesThreadsPerGroup = UInt32(extractRangesPipeline.threadExecutionWidth)
        var maxCompactedU = UInt32(config.maxGaussians)

        // STEP 1: Clear + Project + Compact
        if let cb = queue.makeCommandBuffer() {
            if let encoder = cb.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(clearPipeline)
                encoder.setBuffer(compactedHeader, offset: 0, index: 0)
                encoder.setBytes(&maxCompactedU, length: MemoryLayout<UInt32>.stride, index: 1)
                encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                encoder.endEncoding()
            }
            if let encoder = cb.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(projectFloatPipeline)
                encoder.setBuffer(input.gaussians, offset: 0, index: 0)
                encoder.setBuffer(input.harmonics, offset: 0, index: 1)
                encoder.setBuffer(compactedGaussians, offset: 0, index: 2)
                encoder.setBuffer(compactedHeader, offset: 0, index: 3)
                encoder.setBytes(&cameraUniforms, length: MemoryLayout<CameraUniforms>.stride, index: 4)
                encoder.setBytes(&binningParams, length: MemoryLayout<TileBinningParamsLocal>.stride, index: 5)
                let tg = MTLSize(width: projectFloatPipeline.threadExecutionWidth, height: 1, depth: 1)
                encoder.dispatchThreads(MTLSize(width: input.gaussianCount, height: 1, depth: 1), threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }
            cb.commit()
            cb.waitUntilCompleted()
            timings["1_Project+Compact"] = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
        }

        // STEP 2: Prepare dispatches + Identity indices + Tile counts
        if let cb = queue.makeCommandBuffer() {
            if let encoder = cb.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(prepareVisibleDispatchPipeline)
                encoder.setBuffer(compactedHeader, offset: 0, index: 0)
                encoder.setBuffer(visibleDispatchArgs, offset: 0, index: 1)
                encoder.setBytes(&threadsPerGroup, length: MemoryLayout<UInt32>.stride, index: 2)
                encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                encoder.endEncoding()
            }
            if let encoder = cb.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(prepareRadixDispatchPipeline)
                encoder.setBuffer(compactedHeader, offset: 0, index: 0)
                encoder.setBuffer(radixDispatchArgs, offset: 0, index: 1)
                encoder.setBytes(&elementsPerBlock, length: MemoryLayout<UInt32>.stride, index: 2)
                encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                encoder.endEncoding()
            }
            if let encoder = cb.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(storeVisibleInfoPipeline)
                encoder.setBuffer(compactedHeader, offset: 0, index: 0)
                encoder.setBuffer(visibleInfo, offset: 0, index: 1)
                encoder.setBytes(&elementsPerBlock, length: MemoryLayout<UInt32>.stride, index: 2)
                encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                encoder.endEncoding()
            }
            if let encoder = cb.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(genDepthKeysPipeline)
                encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
                encoder.setBuffer(depthKeys, offset: 0, index: 1)
                encoder.setBuffer(depthSortedIndices, offset: 0, index: 2)
                encoder.setBuffer(compactedHeader, offset: 0, index: 3)
                let tg = MTLSize(width: Int(threadsPerGroup), height: 1, depth: 1)
                encoder.dispatchThreadgroups(indirectBuffer: visibleDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }
            // Depth sort primitives (FastGS key insight!)
            encodeDepthSort(cb)
            if let encoder = cb.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(applyOrderPipeline)
                encoder.setBuffer(depthSortedIndices, offset: 0, index: 0)  // Now depth-sorted!
                encoder.setBuffer(compactedGaussians, offset: 0, index: 1)
                encoder.setBuffer(gaussianTileCounts, offset: 0, index: 2)
                encoder.setBuffer(compactedHeader, offset: 0, index: 3)
                let tg = MTLSize(width: applyOrderPipeline.threadExecutionWidth, height: 1, depth: 1)
                encoder.dispatchThreadgroups(indirectBuffer: visibleDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }
            cb.commit()
            cb.waitUntilCompleted()
            timings["2_DepthSort+TileCounts"] = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
        }

        // STEP 3: Prefix sum + compute total instances
        if let cb = queue.makeCommandBuffer() {
            // Use radixDispatchArgs (computed with elementsPerBlock=1024) which matches prefix scan block size
            encodePrefixScanIndirect(cb, input: gaussianTileCounts, output: gaussianOffsets,
                                     partials: gaussianPartialSums, dispatchArgs: radixDispatchArgs)
            if let encoder = cb.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(computeTotalInstancesPipeline)
                encoder.setBuffer(gaussianOffsets, offset: 0, index: 0)
                encoder.setBuffer(gaussianTileCounts, offset: 0, index: 1)
                encoder.setBuffer(compactedHeader, offset: 0, index: 2)
                encoder.setBuffer(instanceInfo, offset: 0, index: 3)
                encoder.setBuffer(instanceDispatchArgs, offset: 0, index: 4)
                encoder.setBytes(&extractRangesThreadsPerGroup, length: MemoryLayout<UInt32>.stride, index: 5)
                encoder.setBuffer(instanceRadixDispatchArgs, offset: 0, index: 6)
                encoder.setBuffer(instanceHeader, offset: 0, index: 7)
                encoder.setBuffer(radixSortDispatchArgs, offset: 0, index: 8)
                var maxCap = maxInstanceCapacity
                encoder.setBytes(&maxCap, length: MemoryLayout<UInt32>.stride, index: 9)
                encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                encoder.endEncoding()
            }
            cb.commit()
            cb.waitUntilCompleted()
            timings["3_PrefixSum+Total"] = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
        }

        // STEP 4: Create instances with fused (tile, depthOrder) keys
        if let cb = queue.makeCommandBuffer() {
            if let blitEncoder = cb.makeBlitCommandEncoder() {
                blitEncoder.fill(buffer: instanceSortKeys, range: 0..<instanceSortKeys.length, value: 0xFF)
                blitEncoder.endEncoding()
            }
            if let encoder = cb.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(createInstancesV2Pipeline)
                encoder.setBuffer(depthSortedIndices, offset: 0, index: 0)  // Depth-sorted!
                encoder.setBuffer(compactedGaussians, offset: 0, index: 1)
                encoder.setBuffer(gaussianOffsets, offset: 0, index: 2)
                encoder.setBuffer(instanceSortKeys, offset: 0, index: 3)  // SIMD2 keys
                encoder.setBuffer(instanceGaussianIdx, offset: 0, index: 4)
                encoder.setBytes(&tilesXU, length: MemoryLayout<UInt32>.stride, index: 5)
                encoder.setBytes(&tileW, length: MemoryLayout<Int32>.stride, index: 6)
                encoder.setBytes(&tileH, length: MemoryLayout<Int32>.stride, index: 7)
                encoder.setBuffer(compactedHeader, offset: 0, index: 8)
                var maxCap2 = maxInstanceCapacity
                encoder.setBytes(&maxCap2, length: MemoryLayout<UInt32>.stride, index: 9)
                let tg = MTLSize(width: createInstancesV2Pipeline.threadExecutionWidth, height: 1, depth: 1)
                encoder.dispatchThreadgroups(indirectBuffer: visibleDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }
            cb.commit()
            cb.waitUntilCompleted()
            timings["4_CreateInstances"] = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
        }

        // STEP 5: Radix sort by (tile, depthOrder)
        var nTilesU = UInt32(nTiles)
        if let cb = queue.makeCommandBuffer() {
            let offsets = (
                fuse: 0 * 12,
                unpack: 1 * 12,
                histogram: 2 * 12,
                scanBlocks: 3 * 12,
                exclusive: 4 * 12,
                apply: 5 * 12,
                scatter: 6 * 12
            )
            radixSortEncoder.encode(
                commandBuffer: cb,
                keyBuffer: instanceSortKeys,
                sortedIndices: instanceGaussianIdx,
                header: instanceHeader,
                dispatchArgs: radixSortDispatchArgs,
                radixBuffers: radixBuffers,
                offsets: offsets,
                tileCount: nTiles
            )
            cb.commit()
            cb.waitUntilCompleted()
            timings["5_RadixSort"] = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
        }

        // STEP 6: Extract ranges
        if let cb = queue.makeCommandBuffer() {
            if let encoder = cb.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(clearTileRangesPipeline)
                encoder.setBuffer(tileRanges, offset: 0, index: 0)
                encoder.setBytes(&nTilesU, length: MemoryLayout<UInt32>.stride, index: 1)
                let tpg = clearTileRangesPipeline.threadExecutionWidth
                let numGroups = (nTiles + tpg - 1) / tpg
                encoder.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
                encoder.endEncoding()
            }
            if let encoder = cb.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(extractRangesPipeline)
                encoder.setBuffer(instanceSortKeys, offset: 0, index: 0)
                encoder.setBuffer(tileRanges, offset: 0, index: 1)
                encoder.setBuffer(instanceInfo, offset: 0, index: 2)
                encoder.setBytes(&nTilesU, length: MemoryLayout<UInt32>.stride, index: 3)
                let tg = MTLSize(width: extractRangesPipeline.threadExecutionWidth, height: 1, depth: 1)
                encoder.dispatchThreadgroups(indirectBuffer: instanceDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }
            cb.commit()
            cb.waitUntilCompleted()
            timings["6_ExtractRanges"] = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
        }

        // STEP 7: Render
        if let cb = queue.makeCommandBuffer() {
            var renderParams = RenderParams(
                width: UInt32(width),
                height: UInt32(height),
                tileWidth: UInt32(tileWidth),
                tileHeight: UInt32(tileHeight),
                tilesX: UInt32(tilesX),
                tilesY: UInt32(tilesY),
                maxPerTile: 0,
                whiteBackground: 1,
                activeTileCount: 0,
                gaussianCount: 0
            )
            if let encoder = cb.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(renderSortedPipeline)
                encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
                encoder.setBuffer(tileRanges, offset: 0, index: 1)
                encoder.setBuffer(instanceGaussianIdx, offset: 0, index: 2)
                encoder.setTexture(colorTex, index: 0)
                encoder.setBytes(&renderParams, length: MemoryLayout<RenderParams>.stride, index: 3)
                encoder.dispatchThreadgroups(MTLSize(width: tilesX, height: tilesY, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
                encoder.endEncoding()
            }
            cb.commit()
            cb.waitUntilCompleted()
            timings["7_Render"] = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
        }

        return timings
    }

    // MARK: - Private Helpers

    private func ensureTextures(width: Int, height: Int) {
        if colorTexture?.width != width || colorTexture?.height != height {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float,
                width: width,
                height: height,
                mipmapped: false
            )
            desc.usage = [.shaderRead, .shaderWrite]
            desc.storageMode = .private
            colorTexture = device.makeTexture(descriptor: desc)

            desc.pixelFormat = .r16Float
            depthTexture = device.makeTexture(descriptor: desc)
        }
    }

    /// Encode 2-pass radix sort on upper 16 bits (sufficient for depth ordering)
    /// This is 2x faster than full 4-pass sort while maintaining good quality
    private func encodeDepthSort2Pass(_ commandBuffer: MTLCommandBuffer) {
        // Only sort upper 16 bits (passes 2 and 3) - 16-bit depth precision is enough
        for pass in 2..<4 {
            let isFirstPass = (pass == 2)
            let inKeys = isFirstPass ? depthKeys : tempKeys
            let outKeys = isFirstPass ? tempKeys : depthKeys
            let inVals = isFirstPass ? depthSortedIndices : tempVals
            let outVals = isFirstPass ? tempVals : depthSortedIndices

            var digitU = UInt32(pass)

            // Histogram (indirect)
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "DF_Histogram2_\(pass)"
                encoder.setComputePipelineState(radixHistogramPipeline)
                encoder.setBuffer(inKeys, offset: 0, index: 0)
                encoder.setBuffer(radixHistogram, offset: 0, index: 1)
                encoder.setBuffer(visibleInfo, offset: 0, index: 2)
                encoder.setBytes(&digitU, length: MemoryLayout<UInt32>.stride, index: 3)
                let tg = MTLSize(width: radixBlockSize, height: 1, depth: 1)
                encoder.dispatchThreadgroups(indirectBuffer: radixDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }

            // Scan histogram
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "DF_ScanHistogram2_\(pass)"
                encoder.setComputePipelineState(radixScanHistogramPipeline)
                encoder.setBuffer(radixHistogram, offset: 0, index: 0)
                encoder.setBuffer(visibleInfo, offset: 2 * MemoryLayout<UInt32>.stride, index: 1)
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: radixBlockSize, height: 1, depth: 1))
                encoder.endEncoding()
            }

            // Scatter (indirect)
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "DF_Scatter2_\(pass)"
                encoder.setComputePipelineState(radixScatterPipeline)
                encoder.setBuffer(inKeys, offset: 0, index: 0)
                encoder.setBuffer(outKeys, offset: 0, index: 1)
                encoder.setBuffer(inVals, offset: 0, index: 2)
                encoder.setBuffer(outVals, offset: 0, index: 3)
                encoder.setBuffer(radixHistogram, offset: 0, index: 4)
                encoder.setBuffer(visibleInfo, offset: 0, index: 5)
                encoder.setBytes(&digitU, length: MemoryLayout<UInt32>.stride, index: 6)
                let tg = MTLSize(width: radixBlockSize, height: 1, depth: 1)
                encoder.dispatchThreadgroups(indirectBuffer: radixDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }
        }
        // After 2 passes (even from 2), result is back in depthKeys/depthSortedIndices
    }

    /// Encode 4-pass radix sort using indirect dispatch (FULL 32-bit precision)
    /// NOTE: Replaced by encodeDepthSort2Pass for better performance
    private func encodeDepthSort(_ commandBuffer: MTLCommandBuffer) {
        for pass in 0..<4 {
            let isEven = (pass % 2 == 0)
            let inKeys = isEven ? depthKeys : tempKeys
            let outKeys = isEven ? tempKeys : depthKeys
            let inVals = isEven ? depthSortedIndices : tempVals
            let outVals = isEven ? tempVals : depthSortedIndices

            var digitU = UInt32(pass)

            // Histogram (indirect)
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "DF_Histogram_\(pass)"
                encoder.setComputePipelineState(radixHistogramPipeline)
                encoder.setBuffer(inKeys, offset: 0, index: 0)
                encoder.setBuffer(radixHistogram, offset: 0, index: 1)
                encoder.setBuffer(visibleInfo, offset: 0, index: 2)  // count from visibleInfo[0]
                encoder.setBytes(&digitU, length: MemoryLayout<UInt32>.stride, index: 3)
                let tg = MTLSize(width: radixBlockSize, height: 1, depth: 1)
                encoder.dispatchThreadgroups(indirectBuffer: radixDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }

            // Scan histogram
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "DF_ScanHistogram_\(pass)"
                encoder.setComputePipelineState(radixScanHistogramPipeline)
                encoder.setBuffer(radixHistogram, offset: 0, index: 0)
                encoder.setBuffer(visibleInfo, offset: 2 * MemoryLayout<UInt32>.stride, index: 1)  // histogramSize from visibleInfo[2]
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: radixBlockSize, height: 1, depth: 1))
                encoder.endEncoding()
            }

            // Scatter (indirect)
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "DF_Scatter_\(pass)"
                encoder.setComputePipelineState(radixScatterPipeline)
                encoder.setBuffer(inKeys, offset: 0, index: 0)
                encoder.setBuffer(outKeys, offset: 0, index: 1)
                encoder.setBuffer(inVals, offset: 0, index: 2)
                encoder.setBuffer(outVals, offset: 0, index: 3)
                encoder.setBuffer(radixHistogram, offset: 0, index: 4)
                encoder.setBuffer(visibleInfo, offset: 0, index: 5)  // count from visibleInfo[0]
                encoder.setBytes(&digitU, length: MemoryLayout<UInt32>.stride, index: 6)
                let tg = MTLSize(width: radixBlockSize, height: 1, depth: 1)
                encoder.dispatchThreadgroups(indirectBuffer: radixDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }
        }
    }

    /// Encode prefix scan with indirect dispatch for visible count
    private func encodePrefixScanIndirect(_ commandBuffer: MTLCommandBuffer, input: MTLBuffer,
                                           output: MTLBuffer, partials: MTLBuffer,
                                           dispatchArgs: MTLBuffer) {
        // Main scan pass (indirect)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_PrefixScan"
            encoder.setComputePipelineState(prefixScanPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBuffer(visibleInfo, offset: 0, index: 2)  // count from visibleInfo[0]
            encoder.setBuffer(partials, offset: 0, index: 3)
            let tg = MTLSize(width: prefixBlockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: dispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Scan partials (small, single threadgroup)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_ScanPartials"
            encoder.setComputePipelineState(scanPartialSumsPipeline)
            encoder.setBuffer(partials, offset: 0, index: 0)
            encoder.setBuffer(visibleInfo, offset: MemoryLayout<UInt32>.stride, index: 1)  // numGroups from visibleInfo[1]
            encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // Finalize scan (indirect)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_FinalizeScan"
            encoder.setComputePipelineState(finalizeScanPipeline)
            encoder.setBuffer(output, offset: 0, index: 0)
            encoder.setBuffer(visibleInfo, offset: 0, index: 1)  // count from visibleInfo[0]
            encoder.setBuffer(partials, offset: 0, index: 2)
            let tg = MTLSize(width: prefixBlockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: dispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Encode tile counting sort with indirect dispatch
    private func encodeTileSort(_ commandBuffer: MTLCommandBuffer, nTiles: Int) {
        var nTilesU = UInt32(nTiles)
        _ = UInt32(prefixBlockSize * prefixGrainSize)  // tileElemsPerGroup (used in encodeTilePrefixScan)

        // Zero tile counts
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_ZeroTileCounts"
            encoder.setComputePipelineState(zeroCountersPipeline)
            encoder.setBuffer(tileCounts, offset: 0, index: 0)
            encoder.setBytes(&nTilesU, length: MemoryLayout<UInt32>.stride, index: 1)
            let tg = MTLSize(width: zeroCountersPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(MTLSize(width: nTiles, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Count tiles (indirect) - NOTE: this counts from combined keys (upper 12 bits = tile)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_CountTiles"
            encoder.setComputePipelineState(countTilesPipeline)
            encoder.setBuffer(instanceSortKeys, offset: 0, index: 0)  // Combined keys
            encoder.setBuffer(tileCounts, offset: 0, index: 1)
            encoder.setBuffer(instanceInfo, offset: 0, index: 2)  // [0]=total
            let tg = MTLSize(width: countTilesPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: instanceDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Prefix scan tile counts (fixed size = nTiles)
        encodeTilePrefixScan(commandBuffer, input: tileCounts, output: tileOffsets, partials: tilePartialSums, count: nTiles)

        // Zero counters for scatter
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_ZeroCounters"
            encoder.setComputePipelineState(zeroCountersPipeline)
            encoder.setBuffer(tileCounters, offset: 0, index: 0)
            encoder.setBytes(&nTilesU, length: MemoryLayout<UInt32>.stride, index: 1)
            let tg = MTLSize(width: zeroCountersPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(MTLSize(width: nTiles, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Scatter tiles (indirect) - NOTE: this method is unused, replaced by radix sort
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_ScatterTiles"
            encoder.setComputePipelineState(scatterTilesPipeline)
            encoder.setBuffer(instanceSortKeys, offset: 0, index: 0)  // Combined keys
            encoder.setBuffer(instanceGaussianIdx, offset: 0, index: 1)
            encoder.setBuffer(sortedSortKeys, offset: 0, index: 2)  // Output combined keys
            encoder.setBuffer(sortedGaussianIdx, offset: 0, index: 3)
            encoder.setBuffer(tileOffsets, offset: 0, index: 4)
            encoder.setBuffer(tileCounters, offset: 0, index: 5)
            encoder.setBuffer(instanceInfo, offset: 0, index: 6)  // [0]=total
            let tg = MTLSize(width: scatterTilesPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: instanceDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Encode prefix scan for fixed-size tile counts
    private func encodeTilePrefixScan(_ commandBuffer: MTLCommandBuffer, input: MTLBuffer,
                                       output: MTLBuffer, partials: MTLBuffer, count: Int) {
        let elementsPerGroup = prefixBlockSize * prefixGrainSize
        let numGroups = (count + elementsPerGroup - 1) / elementsPerGroup
        var countU = UInt32(count)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_TilePrefixScan"
            encoder.setComputePipelineState(prefixScanPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&countU, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBuffer(partials, offset: 0, index: 3)
            encoder.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_TileScanPartials"
            encoder.setComputePipelineState(scanPartialSumsPipeline)
            encoder.setBuffer(partials, offset: 0, index: 0)
            var numPartialsU = UInt32(numGroups)
            encoder.setBytes(&numPartialsU, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_TileFinalizeScan"
            encoder.setComputePipelineState(finalizeScanPipeline)
            encoder.setBuffer(output, offset: 0, index: 0)
            encoder.setBytes(&countU, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.setBuffer(partials, offset: 0, index: 2)
            encoder.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    /// Encode 2-pass radix sort for instance keys (sort upper 16 bits only)
    /// Combined key format: (tileKey << 20) | depthOrder
    /// By only sorting bits 16-31, we preserve depth order (bits 0-19) within each tile
    /// This is 2x faster than full 4-pass radix sort!
    private func encodeInstanceRadixSort2Pass(_ commandBuffer: MTLCommandBuffer) {
        // Only do passes 2 and 3 (bits 16-23 and 24-31)
        // This sorts by tile while preserving relative order (depth order) within tiles
        for pass in 2..<4 {  // Skip passes 0 and 1
            let isFirstPass = (pass == 2)
            let inKeys = isFirstPass ? instanceSortKeys : tempInstanceKeys
            let outKeys = isFirstPass ? tempInstanceKeys : sortedSortKeys
            let inVals = isFirstPass ? instanceGaussianIdx : tempInstanceVals
            let outVals = isFirstPass ? tempInstanceVals : sortedGaussianIdx

            var digitU = UInt32(pass)

            // Histogram (indirect)
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "DF_InstHist2_\(pass)"
                encoder.setComputePipelineState(radixHistogramPipeline)
                encoder.setBuffer(inKeys, offset: 0, index: 0)
                encoder.setBuffer(radixHistogram, offset: 0, index: 1)
                encoder.setBuffer(instanceInfo, offset: 0, index: 2)
                encoder.setBytes(&digitU, length: MemoryLayout<UInt32>.stride, index: 3)
                let tg = MTLSize(width: radixBlockSize, height: 1, depth: 1)
                encoder.dispatchThreadgroups(indirectBuffer: instanceRadixDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }

            // Scan histogram
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "DF_InstScanHist2_\(pass)"
                encoder.setComputePipelineState(radixScanHistogramPipeline)
                encoder.setBuffer(radixHistogram, offset: 0, index: 0)
                encoder.setBuffer(instanceInfo, offset: 2 * MemoryLayout<UInt32>.stride, index: 1)
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: radixBlockSize, height: 1, depth: 1))
                encoder.endEncoding()
            }

            // Scatter (indirect)
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "DF_InstScatter2_\(pass)"
                encoder.setComputePipelineState(radixScatterPipeline)
                encoder.setBuffer(inKeys, offset: 0, index: 0)
                encoder.setBuffer(outKeys, offset: 0, index: 1)
                encoder.setBuffer(inVals, offset: 0, index: 2)
                encoder.setBuffer(outVals, offset: 0, index: 3)
                encoder.setBuffer(radixHistogram, offset: 0, index: 4)
                encoder.setBuffer(instanceInfo, offset: 0, index: 5)
                encoder.setBytes(&digitU, length: MemoryLayout<UInt32>.stride, index: 6)
                let tg = MTLSize(width: radixBlockSize, height: 1, depth: 1)
                encoder.dispatchThreadgroups(indirectBuffer: instanceRadixDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }
        }
        // After 2 passes (even from 2), result is in sortedSortKeys/sortedGaussianIdx (no copy needed!)
    }

    /// Encode counting sort for tile assignment (NOT stable with atomics!)
    /// NOTE: Replaced by encodeInstanceRadixSort2Pass which is stable
    private func encodeCountingSort(_ commandBuffer: MTLCommandBuffer, nTiles: Int) {
        var nTilesU = UInt32(nTiles)

        // 1. Zero tile counts
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_ZeroTileCounts"
            encoder.setComputePipelineState(zeroCountersPipeline)
            encoder.setBuffer(tileCounts, offset: 0, index: 0)
            encoder.setBytes(&nTilesU, length: MemoryLayout<UInt32>.stride, index: 1)
            let tg = MTLSize(width: zeroCountersPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(MTLSize(width: nTiles, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // 2. Count instances per tile (indirect dispatch based on instance count)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_CountTiles"
            encoder.setComputePipelineState(countTilesPipeline)
            encoder.setBuffer(instanceSortKeys, offset: 0, index: 0)  // Combined keys
            encoder.setBuffer(tileCounts, offset: 0, index: 1)
            encoder.setBuffer(instanceInfo, offset: 0, index: 2)  // [0]=total instances
            let tg = MTLSize(width: countTilesPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: instanceDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // 3. Prefix sum tile counts → offsets (fixed size = nTiles)
        encodeTilePrefixScan(commandBuffer, input: tileCounts, output: tileOffsets,
                              partials: tilePartialSums, count: nTiles)

        // 4. Zero tile counters for scatter
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_ZeroCounters"
            encoder.setComputePipelineState(zeroCountersPipeline)
            encoder.setBuffer(tileCounters, offset: 0, index: 0)
            encoder.setBytes(&nTilesU, length: MemoryLayout<UInt32>.stride, index: 1)
            let tg = MTLSize(width: zeroCountersPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(MTLSize(width: nTiles, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // 5. Scatter instances to sorted positions (stable - preserves depth order!)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DF_ScatterTiles"
            encoder.setComputePipelineState(scatterTilesPipeline)
            encoder.setBuffer(instanceSortKeys, offset: 0, index: 0)  // Input combined keys
            encoder.setBuffer(instanceGaussianIdx, offset: 0, index: 1)
            encoder.setBuffer(sortedSortKeys, offset: 0, index: 2)    // Output combined keys
            encoder.setBuffer(sortedGaussianIdx, offset: 0, index: 3)
            encoder.setBuffer(tileOffsets, offset: 0, index: 4)
            encoder.setBuffer(tileCounters, offset: 0, index: 5)
            encoder.setBuffer(instanceInfo, offset: 0, index: 6)  // [0]=total instances
            let tg = MTLSize(width: scatterTilesPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: instanceDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Encode radix sort for instance keys (stable tile sort via combined key)
    /// Combined key format: (tileKey << 20) | depthOrder
    /// Radix sort is inherently stable, so depth order is preserved within tiles
    /// NOTE: Replaced by encodeCountingSort which is much faster
    private func encodeInstanceRadixSort(_ commandBuffer: MTLCommandBuffer) {
        // Use 4 passes of 8-bit radix sort on 32-bit combined keys
        // This sorts by tile first (upper bits), then by depth order (lower bits)
        // After 4 passes: sortedSortKeys/sortedGaussianIdx contain tile+depth sorted data
        //
        // instanceInfo layout: [0]=total, [1]=numBlocks, [2]=histogramSize

        for pass in 0..<4 {
            let isEven = (pass % 2 == 0)
            let inKeys = isEven ? instanceSortKeys : tempInstanceKeys
            let outKeys = isEven ? tempInstanceKeys : instanceSortKeys
            let inVals = isEven ? instanceGaussianIdx : tempInstanceVals
            let outVals = isEven ? tempInstanceVals : instanceGaussianIdx

            var digitU = UInt32(pass)

            // Histogram (indirect) - uses instanceInfo[0] for count
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "DF_InstHist_\(pass)"
                encoder.setComputePipelineState(radixHistogramPipeline)
                encoder.setBuffer(inKeys, offset: 0, index: 0)
                encoder.setBuffer(radixHistogram, offset: 0, index: 1)
                encoder.setBuffer(instanceInfo, offset: 0, index: 2)  // [0]=count
                encoder.setBytes(&digitU, length: MemoryLayout<UInt32>.stride, index: 3)
                let tg = MTLSize(width: radixBlockSize, height: 1, depth: 1)
                encoder.dispatchThreadgroups(indirectBuffer: instanceRadixDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }

            // Scan histogram - uses instanceInfo[2] for histogramSize
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "DF_InstScanHist_\(pass)"
                encoder.setComputePipelineState(radixScanHistogramPipeline)
                encoder.setBuffer(radixHistogram, offset: 0, index: 0)
                encoder.setBuffer(instanceInfo, offset: 2 * MemoryLayout<UInt32>.stride, index: 1)  // Read [2]=histogramSize
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: radixBlockSize, height: 1, depth: 1))
                encoder.endEncoding()
            }

            // Scatter (indirect) - uses instanceInfo[0] for count
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "DF_InstScatter_\(pass)"
                encoder.setComputePipelineState(radixScatterPipeline)
                encoder.setBuffer(inKeys, offset: 0, index: 0)
                encoder.setBuffer(outKeys, offset: 0, index: 1)
                encoder.setBuffer(inVals, offset: 0, index: 2)
                encoder.setBuffer(outVals, offset: 0, index: 3)
                encoder.setBuffer(radixHistogram, offset: 0, index: 4)
                encoder.setBuffer(instanceInfo, offset: 0, index: 5)  // [0]=count
                encoder.setBytes(&digitU, length: MemoryLayout<UInt32>.stride, index: 6)
                let tg = MTLSize(width: radixBlockSize, height: 1, depth: 1)
                encoder.dispatchThreadgroups(indirectBuffer: instanceRadixDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }
        }

        // After 4 passes (even), result is back in instanceSortKeys/instanceGaussianIdx
        // Copy to sortedSortKeys/sortedGaussianIdx for consistency
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "DF_CopySorted"
            blit.copy(from: instanceSortKeys, sourceOffset: 0,
                     to: sortedSortKeys, destinationOffset: 0,
                     size: instanceSortKeys.length)
            blit.copy(from: instanceGaussianIdx, sourceOffset: 0,
                     to: sortedGaussianIdx, destinationOffset: 0,
                     size: instanceGaussianIdx.length)
            blit.endEncoding()
        }
    }
}

// MARK: - Supporting Types

fileprivate struct CameraUniforms {
    var viewMatrix: simd_float4x4
    var projectionMatrix: simd_float4x4
    var cameraCenter: SIMD3<Float>
    var pixelFactor: Float
    var focalX: Float
    var focalY: Float
    var width: Float
    var height: Float
    var nearPlane: Float
    var farPlane: Float
    var shComponents: UInt32
    var gaussianCount: UInt32
    var padding0: UInt32
    var padding1: UInt32
}

fileprivate struct TileBinningParamsLocal {
    var gaussianCount: UInt32
    var tilesX: UInt32
    var tilesY: UInt32
    var tileWidth: UInt32
    var tileHeight: UInt32
    var surfaceWidth: UInt32
    var surfaceHeight: UInt32
    var maxCapacity: UInt32
    var alphaThreshold: Float = 0.004
    var minCoverageRatio: Float = 0.02
}

fileprivate struct TileAssignmentHeader {
    var totalAssignments: UInt32
    var maxCapacity: UInt32
    var paddedCount: UInt32
    var overflow: UInt32
}

fileprivate struct CompactedGaussian {
    var covariance_depth: SIMD4<Float>
    var position_color: SIMD4<Float>
    var min_tile: SIMD2<Int32>
    var max_tile: SIMD2<Int32>
    var originalIdx: UInt32
    var _pad0: UInt32
}

fileprivate struct RenderParams {
    var width: UInt32
    var height: UInt32
    var tileWidth: UInt32
    var tileHeight: UInt32
    var tilesX: UInt32
    var tilesY: UInt32
    var maxPerTile: UInt32
    var whiteBackground: UInt32
    var activeTileCount: UInt32
    var gaussianCount: UInt32
}

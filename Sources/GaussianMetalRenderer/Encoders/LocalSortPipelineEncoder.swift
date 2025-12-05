import Metal

/// Local sort pipeline encoder - Memory-efficient deterministic Gaussian splatting
/// Uses two-pass projection: Visibility → PrefixSum → ProjectScatter
/// Pipeline: Clear → ProjectVisibility → PrefixScan → ProjectScatter → TilePrefixScan → Scatter → Sort → Render
public final class LocalSortPipelineEncoder {
    // Device and library
    private let device: MTLDevice
    private let localSortLibrary: MTLLibrary

    /// Public access to the Metal library for shared encoders (e.g., ClusterCullEncoder)
    public var library: MTLLibrary { localSortLibrary }

    // Core pipelines
    private let clearPipeline: MTLComputePipelineState
    private let writeVisibleCountPipeline: MTLComputePipelineState

    // Efficient projection pipelines (temp buffer approach - single projection pass)
    // Key: (SH degree, USE_CLUSTER_CULL)
    private var projectStoreFloatPipelines: [UInt64: MTLComputePipelineState] = [:]
    private var projectStoreHalfPipelines: [UInt64: MTLComputePipelineState] = [:]
    private let compactCountPipeline: MTLComputePipelineState
    private let prepareCompactDispatchPipeline: MTLComputePipelineState

    // Hierarchical prefix sum pipelines
    private let blockReducePipeline: MTLComputePipelineState
    private let singleBlockScanPipeline: MTLComputePipelineState
    private let blockScanPipeline: MTLComputePipelineState

    // Tile prefix scan pipelines
    private let prefixScanPipeline: MTLComputePipelineState
    private let scanPartialSumsPipeline: MTLComputePipelineState
    private let finalizeScanPipeline: MTLComputePipelineState
    private let finalizeScanAndZeroPipeline: MTLComputePipelineState

    // Scatter and sort pipelines
    private let prepareScatterDispatchPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState
    private let perTileSortPipeline: MTLComputePipelineState
    private let renderPipeline: MTLComputePipelineState
    private let zeroPipeline: MTLComputePipelineState

    // 16-bit sort experimental pipelines (optional)
    private let scatter16Pipeline: MTLComputePipelineState?
    private let sort16Pipeline: MTLComputePipelineState?
    private let render16Pipeline: MTLComputePipelineState?

    // Texture-cached render (optional)
    private let packRenderTexturePipeline: MTLComputePipelineState?
    private let renderTexturedPipeline: MTLComputePipelineState?

    private let prefixBlockSize = 256
    private let prefixGrainSize = 4

    /// Create pipeline key from SH degree and cluster cull flag
    private static func pipelineKey(shDegree: UInt32, useClusterCull: Bool) -> UInt64 {
        return UInt64(shDegree) | (useClusterCull ? 0x100 : 0)
    }

    /// Map shComponents count to SH degree (0-3)
    private static func shDegree(from shComponents: UInt32) -> UInt32 {
        switch shComponents {
        case 0, 1: return 0     // DC only
        case 2...4: return 1    // Degree 1 (4 coeffs)
        case 5...9: return 2    // Degree 2 (9 coeffs)
        default: return 3       // Degree 3 (16 coeffs)
        }
    }

    public init(device: MTLDevice) throws {
        self.device = device

        // Load Local sort shader library
        guard let libraryURL = Bundle.module.url(forResource: "LocalSortShaders", withExtension: "metallib"),
              let library = try? device.makeLibrary(URL: libraryURL) else {
            fatalError("Failed to load LocalSortShaders.metallib")
        }
        self.localSortLibrary = library

        // Default function constants
        let defaultConstants = MTLFunctionConstantValues()
        var defaultDegree: UInt32 = 3
        var defaultCull: Bool = false
        defaultConstants.setConstantValue(&defaultDegree, type: .uint, index: 0)
        defaultConstants.setConstantValue(&defaultCull, type: .bool, index: 1)

        // Load core kernels
        guard let clearFn = library.makeFunction(name: "localSortClear"),
              let writeVisibleCountFn = library.makeFunction(name: "localSortWriteVisibleCount"),
              let compactCountFn = library.makeFunction(name: "localSortCompactCount"),
              let prefixFn = library.makeFunction(name: "localSortPrefixScan"),
              let partialFn = library.makeFunction(name: "localSortScanPartialSums"),
              let finalizeFn = library.makeFunction(name: "localSortFinalizeScan"),
              let finalizeAndZeroFn = library.makeFunction(name: "localSortFinalizeScanAndZero"),
              let prepareDispatchFn = library.makeFunction(name: "localSortPrepareScatterDispatch"),
              let scatterFn = library.makeFunction(name: "localSortScatterSimd"),
              let sortFn = library.makeFunction(name: "localSortPerTileSort"),
              let renderFn = library.makeFunction(name: "localSortRender"),
              let zeroFn = library.makeFunction(name: "tileBinningZeroCountsKernel") else {
            fatalError("Missing required kernel functions in LocalSortShaders")
        }

        self.clearPipeline = try device.makeComputePipelineState(function: clearFn)
        self.writeVisibleCountPipeline = try device.makeComputePipelineState(function: writeVisibleCountFn)
        self.compactCountPipeline = try device.makeComputePipelineState(function: compactCountFn)
        self.prefixScanPipeline = try device.makeComputePipelineState(function: prefixFn)
        self.scanPartialSumsPipeline = try device.makeComputePipelineState(function: partialFn)
        self.finalizeScanPipeline = try device.makeComputePipelineState(function: finalizeFn)
        self.finalizeScanAndZeroPipeline = try device.makeComputePipelineState(function: finalizeAndZeroFn)
        self.prepareScatterDispatchPipeline = try device.makeComputePipelineState(function: prepareDispatchFn)
        self.prepareCompactDispatchPipeline = try device.makeComputePipelineState(function: prepareDispatchFn)  // Reuse same kernel with different args
        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)
        self.perTileSortPipeline = try device.makeComputePipelineState(function: sortFn)
        self.renderPipeline = try device.makeComputePipelineState(function: renderFn)
        self.zeroPipeline = try device.makeComputePipelineState(function: zeroFn)

        // Create projection+store pipeline variants: SH degree (0-3) x cluster cull (true/false)
        for degree: UInt32 in 0...3 {
            for useClusterCull in [false, true] {
                let constantValues = MTLFunctionConstantValues()
                var shDegree = degree
                var cullEnabled = useClusterCull
                constantValues.setConstantValue(&shDegree, type: .uint, index: 0)
                constantValues.setConstantValue(&cullEnabled, type: .bool, index: 1)

                let key = Self.pipelineKey(shDegree: degree, useClusterCull: useClusterCull)

                // ProjectStore - Float world, Float harmonics
                if let fn = try? library.makeFunction(name: "localSortProjectStoreFloat", constantValues: constantValues) {
                    self.projectStoreFloatPipelines[key] = try? device.makeComputePipelineState(function: fn)
                }

                // ProjectStore - Half world, Half harmonics
                if let fn = try? library.makeFunction(name: "localSortProjectStoreHalfHalfSh", constantValues: constantValues) {
                    self.projectStoreHalfPipelines[key] = try? device.makeComputePipelineState(function: fn)
                }
            }
        }

        // Load hierarchical prefix sum pipelines from main library
        guard let mainLibraryURL = Bundle.module.url(forResource: "GaussianMetalRenderer", withExtension: "metallib"),
              let mainLibrary = try? device.makeLibrary(URL: mainLibraryURL) else {
            fatalError("Failed to load GaussianMetalRenderer.metallib")
        }
        guard let reduceFn = mainLibrary.makeFunction(name: "blockReduceKernel"),
              let singleScanFn = mainLibrary.makeFunction(name: "singleBlockScanKernel"),
              let blockScanFn = mainLibrary.makeFunction(name: "blockScanKernel") else {
            fatalError("Missing hierarchical prefix sum kernels in GaussianMetalRenderer")
        }
        self.blockReducePipeline = try device.makeComputePipelineState(function: reduceFn)
        self.singleBlockScanPipeline = try device.makeComputePipelineState(function: singleScanFn)
        self.blockScanPipeline = try device.makeComputePipelineState(function: blockScanFn)

        // Optional: Texture-cached render pipelines
        if let packFn = library.makeFunction(name: "localSortPackRenderTexture"),
           let renderTexFn = library.makeFunction(name: "localSortRenderTextured") {
            self.packRenderTexturePipeline = try? device.makeComputePipelineState(function: packFn)
            self.renderTexturedPipeline = try? device.makeComputePipelineState(function: renderTexFn)
        } else {
            self.packRenderTexturePipeline = nil
            self.renderTexturedPipeline = nil
        }

        // Optional: 16-bit sort pipelines
        if let scatter16Fn = library.makeFunction(name: "localSortScatterSimd16"),
           let sort16Fn = library.makeFunction(name: "localSortPerTileSort16"),
           let render16Fn = library.makeFunction(name: "localSortRender16") {
            self.scatter16Pipeline = try? device.makeComputePipelineState(function: scatter16Fn)
            self.sort16Pipeline = try? device.makeComputePipelineState(function: sort16Fn)
            self.render16Pipeline = try? device.makeComputePipelineState(function: render16Fn)
        } else {
            self.scatter16Pipeline = nil
            self.sort16Pipeline = nil
            self.render16Pipeline = nil
        }
    }

    /// Legacy init for backwards compatibility
    public convenience init(device: MTLDevice, library: MTLLibrary) throws {
        try self.init(device: device)
    }

    /// Encode the full Local sort pipeline
    /// Efficient temp buffer approach: ProjectStore → PrefixSum → CompactCount → Scatter → Sort
    /// Temp buffer can alias with other buffers not used during projection
    public func encode(
        commandBuffer: MTLCommandBuffer,
        // Input
        worldGaussians: MTLBuffer,
        harmonics: MTLBuffer,
        camera: CameraUniformsSwift,
        gaussianCount: Int,
        // Tile params
        tilesX: Int,
        tilesY: Int,
        tileWidth: Int,
        tileHeight: Int,
        surfaceWidth: Int,
        surfaceHeight: Int,
        // Output buffers
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        tileCounts: MTLBuffer,
        tileOffsets: MTLBuffer,
        partialSums: MTLBuffer,
        sortKeys: MTLBuffer?,
        sortIndices: MTLBuffer,
        maxCompacted: Int,
        maxAssignments: Int,
        // Temp buffer for single-pass projection (can alias with sortKeys/sortIndices)
        tempProjectionBuffer: MTLBuffer,   // [gaussianCount] CompactedGaussian
        // Visibility prefix sum buffers
        visibilityMarks: MTLBuffer,        // [gaussianCount + 1] for prefix sum
        visibilityPartialSums: MTLBuffer,  // For hierarchical scan
        // Options
        useHalfWorld: Bool = false,
        skipSort: Bool = false,
        tempSortKeys: MTLBuffer? = nil,
        tempSortIndices: MTLBuffer? = nil,
        clusterVisibility: MTLBuffer? = nil,
        clusterSize: UInt32 = 1024,
        use16BitSort: Bool = false,
        depthKeys16: MTLBuffer? = nil
    ) {
        let tileCount = tilesX * tilesY
        let elementsPerGroup = prefixBlockSize * prefixGrainSize
        let actualGroups = max(1, (tileCount + elementsPerGroup - 1) / elementsPerGroup)

        var params = TileBinningParams(
            gaussianCount: UInt32(gaussianCount),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            surfaceWidth: UInt32(surfaceWidth),
            surfaceHeight: UInt32(surfaceHeight),
            maxCapacity: UInt32(maxCompacted)
        )

        var tileCountU = UInt32(tileCount)
        var maxCompactedU = UInt32(maxCompacted)
        var tilesXU = UInt32(tilesX)
        var maxAssignmentsU = UInt32(maxAssignments)
        var gaussianCountU = UInt32(gaussianCount)
        var cameraUniforms = camera

        let shDegree = Self.shDegree(from: camera.shComponents)
        let useClusterCull = clusterVisibility != nil
        let key = Self.pipelineKey(shDegree: shDegree, useClusterCull: useClusterCull)

        // === 1. CLEAR ===
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_Clear"
            encoder.setComputePipelineState(clearPipeline)
            encoder.setBuffer(tileCounts, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBytes(&tileCountU, length: 4, index: 2)
            encoder.setBytes(&maxCompactedU, length: 4, index: 3)
            let threads = MTLSize(width: max(tileCount, 1), height: 1, depth: 1)
            let tg = MTLSize(width: clearPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === 2a. PROJECT + STORE (single pass - stores full data to temp buffer) ===
        // Projects all gaussians, writes full CompactedGaussian to tempBuffer[gid], marks visibility
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_ProjectStore"
            let pipeline = useHalfWorld
                ? projectStoreHalfPipelines[key]!
                : projectStoreFloatPipelines[key]!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(worldGaussians, offset: 0, index: 0)
            encoder.setBuffer(harmonics, offset: 0, index: 1)
            encoder.setBuffer(tempProjectionBuffer, offset: 0, index: 2)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 3)
            encoder.setBytes(&cameraUniforms, length: MemoryLayout<CameraUniformsSwift>.stride, index: 4)
            encoder.setBytes(&params, length: MemoryLayout<TileBinningParams>.stride, index: 5)
            if let clusterVis = clusterVisibility {
                encoder.setBuffer(clusterVis, offset: 0, index: 6)
                var cs = clusterSize
                encoder.setBytes(&cs, length: 4, index: 7)
            }
            let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
            let tg = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === 2b. HIERARCHICAL PREFIX SUM ON VISIBILITY ===
        let visScanCount = gaussianCount + 1
        let visBlocks = (visScanCount + prefixBlockSize - 1) / prefixBlockSize
        var visScanCountU = UInt32(visScanCount)
        var visBlocksU = UInt32(visBlocks)

        // Block reduce
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_VisBlockReduce"
            encoder.setComputePipelineState(blockReducePipeline)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 0)
            encoder.setBuffer(visibilityPartialSums, offset: 0, index: 1)
            encoder.setBytes(&visScanCountU, length: 4, index: 2)
            let threads = MTLSize(width: visBlocks * prefixBlockSize, height: 1, depth: 1)
            let tg = MTLSize(width: prefixBlockSize, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Scan block sums
        if visBlocks <= prefixBlockSize {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "LocalSort_VisScanBlockSums"
                encoder.setComputePipelineState(singleBlockScanPipeline)
                encoder.setBuffer(visibilityPartialSums, offset: 0, index: 0)
                encoder.setBytes(&visBlocksU, length: 4, index: 1)
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
                encoder.endEncoding()
            }
        } else {
            // Two-level hierarchical scan
            let level2Blocks = (visBlocks + prefixBlockSize - 1) / prefixBlockSize
            let level2Offset = (visBlocks + 1) * 4
            var level2BlocksU = UInt32(level2Blocks)

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "LocalSort_VisLevel2Reduce"
                encoder.setComputePipelineState(blockReducePipeline)
                encoder.setBuffer(visibilityPartialSums, offset: 0, index: 0)
                encoder.setBuffer(visibilityPartialSums, offset: level2Offset, index: 1)
                encoder.setBytes(&visBlocksU, length: 4, index: 2)
                let threads = MTLSize(width: level2Blocks * prefixBlockSize, height: 1, depth: 1)
                encoder.dispatchThreads(threads, threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
                encoder.endEncoding()
            }

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "LocalSort_VisLevel2Scan"
                encoder.setComputePipelineState(singleBlockScanPipeline)
                encoder.setBuffer(visibilityPartialSums, offset: level2Offset, index: 0)
                encoder.setBytes(&level2BlocksU, length: 4, index: 1)
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
                encoder.endEncoding()
            }

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "LocalSort_VisLevel1Scan"
                encoder.setComputePipelineState(blockScanPipeline)
                encoder.setBuffer(visibilityPartialSums, offset: 0, index: 0)
                encoder.setBuffer(visibilityPartialSums, offset: 0, index: 1)
                encoder.setBuffer(visibilityPartialSums, offset: level2Offset, index: 2)
                encoder.setBytes(&visBlocksU, length: 4, index: 3)
                let threads = MTLSize(width: level2Blocks * prefixBlockSize, height: 1, depth: 1)
                encoder.dispatchThreads(threads, threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
                encoder.endEncoding()
            }
        }

        // Final scan - apply block offsets
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_VisFinalScan"
            encoder.setComputePipelineState(blockScanPipeline)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 0)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 1)
            encoder.setBuffer(visibilityPartialSums, offset: 0, index: 2)
            encoder.setBytes(&visScanCountU, length: 4, index: 3)
            let threads = MTLSize(width: visBlocks * prefixBlockSize, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // === 2c. WRITE VISIBLE COUNT TO HEADER ===
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_WriteVisibleCount"
            encoder.setComputePipelineState(writeVisibleCountPipeline)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBytes(&gaussianCountU, length: 4, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // === 2d. COMPACT + COUNT (copies visible from temp to compacted, counts tiles) ===
        // Uses prefix sum offsets for deterministic output ordering
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_CompactCount"
            encoder.setComputePipelineState(compactCountPipeline)
            encoder.setBuffer(tempProjectionBuffer, offset: 0, index: 0)
            encoder.setBuffer(visibilityMarks, offset: 0, index: 1)  // Prefix sum offsets
            encoder.setBuffer(compactedGaussians, offset: 0, index: 2)
            encoder.setBuffer(tileCounts, offset: 0, index: 3)
            encoder.setBytes(&gaussianCountU, length: 4, index: 4)
            encoder.setBytes(&tilesXU, length: 4, index: 5)
            let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
            let tg = MTLSize(width: compactCountPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === 3. PREFIX SCAN ===
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_PrefixScan"
            encoder.setComputePipelineState(prefixScanPipeline)
            encoder.setBuffer(tileCounts, offset: 0, index: 0)
            encoder.setBuffer(tileOffsets, offset: 0, index: 1)
            encoder.setBytes(&tileCountU, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBuffer(partialSums, offset: 0, index: 3)
            let tg = MTLSize(width: prefixBlockSize, height: 1, depth: 1)
            let tgs = MTLSize(width: actualGroups, height: 1, depth: 1)
            encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_ScanPartialSums"
            encoder.setComputePipelineState(scanPartialSumsPipeline)
            var numPartial = UInt32(actualGroups)
            encoder.setBuffer(partialSums, offset: 0, index: 0)
            encoder.setBytes(&numPartial, length: MemoryLayout<UInt32>.stride, index: 1)
            let tg = MTLSize(width: prefixBlockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === 3b + 4. FUSED FINALIZE SCAN + ZERO COUNTERS ===
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_FinalizeScanAndZero"
            encoder.setComputePipelineState(finalizeScanAndZeroPipeline)
            encoder.setBuffer(tileOffsets, offset: 0, index: 0)
            encoder.setBuffer(tileCounts, offset: 0, index: 1)
            encoder.setBytes(&tileCountU, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBuffer(partialSums, offset: 0, index: 3)
            let tg = MTLSize(width: prefixBlockSize, height: 1, depth: 1)
            let tgs = MTLSize(width: actualGroups, height: 1, depth: 1)
            encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === 5a. PREPARE INDIRECT DISPATCH for scatter (GPU-driven) ===
        // Create dispatch args buffer inline (12 bytes for MTLDispatchThreadgroupsIndirectArguments)
        // SIMD scatter: 4 gaussians per threadgroup (4 SIMD groups × 32 threads = 128 threads)
        let dispatchArgsBuffer = commandBuffer.device.makeBuffer(length: 12, options: .storageModeShared)!
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_PrepareScatterDispatch"
            encoder.setComputePipelineState(prepareScatterDispatchPipeline)
            encoder.setBuffer(compactedHeader, offset: 0, index: 0)
            encoder.setBuffer(dispatchArgsBuffer, offset: 0, index: 1)
            var tgWidth = UInt32(4)  // 4 gaussians per threadgroup (4 SIMD groups)
            encoder.setBytes(&tgWidth, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // === 5b. SCATTER (SIMD-cooperative, GPU-driven indirect dispatch) ===
        // Each SIMD group (32 threads) handles 1 gaussian cooperatively
        // 4 SIMD groups per threadgroup = 4 gaussians per threadgroup = 128 threads
        let effective16Bit = use16BitSort && scatter16Pipeline != nil && depthKeys16 != nil
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            if effective16Bit, let scatter16 = scatter16Pipeline, let depth16 = depthKeys16 {
                // 16-bit scatter: writes depthKeys16 (ushort) + sortIndices (uint) - sortInfo removed!
                encoder.label = "LocalSort_Scatter16_SIMD"
                encoder.setComputePipelineState(scatter16)
                encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
                encoder.setBuffer(compactedHeader, offset: 0, index: 1)
                encoder.setBuffer(tileCounts, offset: 0, index: 2)
                encoder.setBuffer(tileOffsets, offset: 0, index: 3)
                encoder.setBuffer(depth16, offset: 0, index: 4)      // ushort buffer for depth
                encoder.setBuffer(sortIndices, offset: 0, index: 5)  // uint buffer for global indices
                encoder.setBytes(&tilesXU, length: MemoryLayout<UInt32>.stride, index: 6)
                encoder.setBytes(&maxAssignmentsU, length: MemoryLayout<UInt32>.stride, index: 7)
                var tileW = Int32(tileWidth)
                var tileH = Int32(tileHeight)
                encoder.setBytes(&tileW, length: MemoryLayout<Int32>.stride, index: 8)
                encoder.setBytes(&tileH, length: MemoryLayout<Int32>.stride, index: 9)
                // buffer(10) removed - sortInfo no longer needed
            } else if let sortKeysBuf = sortKeys {
                // 32-bit scatter: writes combined sortKeys (depth16 << 16 | idx16) + sortIndices
                encoder.label = "LocalSort_Scatter_SIMD"
                encoder.setComputePipelineState(scatterPipeline)
                encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
                encoder.setBuffer(compactedHeader, offset: 0, index: 1)
                encoder.setBuffer(tileCounts, offset: 0, index: 2)
                encoder.setBuffer(tileOffsets, offset: 0, index: 3)
                encoder.setBuffer(sortKeysBuf, offset: 0, index: 4)
                encoder.setBuffer(sortIndices, offset: 0, index: 5)
                encoder.setBytes(&tilesXU, length: MemoryLayout<UInt32>.stride, index: 6)
                encoder.setBytes(&maxAssignmentsU, length: MemoryLayout<UInt32>.stride, index: 7)
                var tileW = Int32(tileWidth)
                var tileH = Int32(tileHeight)
                encoder.setBytes(&tileW, length: MemoryLayout<Int32>.stride, index: 8)
                encoder.setBytes(&tileH, length: MemoryLayout<Int32>.stride, index: 9)
            }

            // 128 threads per threadgroup (4 SIMD groups = 4 gaussians)
            let tg = MTLSize(width: 128, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: dispatchArgsBuffer, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === 6. PER-TILE SORT (32-bit only) ===
        if !skipSort, let sortKeysBuf = sortKeys, let tempK = tempSortKeys, let tempV = tempSortIndices {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "LocalSort_PerTileSort"
                encoder.setComputePipelineState(perTileSortPipeline)
                encoder.setBuffer(sortKeysBuf, offset: 0, index: 0)
                encoder.setBuffer(sortIndices, offset: 0, index: 1)
                encoder.setBuffer(tileOffsets, offset: 0, index: 2)
                encoder.setBuffer(tileCounts, offset: 0, index: 3)
                encoder.setBuffer(tempK, offset: 0, index: 4)
                encoder.setBuffer(tempV, offset: 0, index: 5)

                // One threadgroup per tile
                let tgs = MTLSize(width: tileCount, height: 1, depth: 1)
                let tg = MTLSize(width: 256, height: 1, depth: 1)
                encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }
        }
    }

    /// Encode just the render pass (after sorting is complete)
    public func encodeRender(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        tileOffsets: MTLBuffer,
        tileCounts: MTLBuffer,
        sortedIndices: MTLBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        tilesX: Int,
        tilesY: Int,
        width: Int,
        height: Int,
        tileWidth: Int,
        tileHeight: Int,
        whiteBackground: Bool = false
    ) {
        var params = LocalSortRenderParamsSwift(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            maxPerTile: 0,  // Not used in LocalSort render
            whiteBackground: whiteBackground ? 1 : 0,
            activeTileCount: 0,  // Not used in LocalSort render
            gaussianCount: 0  // Not used in LocalSort render
        )

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_Render"
            encoder.setComputePipelineState(renderPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(tileOffsets, offset: 0, index: 1)
            encoder.setBuffer(tileCounts, offset: 0, index: 2)
            encoder.setBuffer(sortedIndices, offset: 0, index: 3)
            encoder.setTexture(colorTexture, index: 0)
            encoder.setTexture(depthTexture, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<LocalSortRenderParamsSwift>.stride, index: 4)

            // 4×8 threadgroup for 16×16 tile (4×2 pixels per thread)
            let tgs = MTLSize(width: tilesX, height: tilesY, depth: 1)
            let tg = MTLSize(width: 4, height: 8, depth: 1)
            encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Texture width for render texture (must match RENDER_TEX_WIDTH in shader)
    public static let renderTexWidth = 4096

    /// Check if textured render is available
    public var hasTexturedRender: Bool {
        return packRenderTexturePipeline != nil && renderTexturedPipeline != nil
    }

    /// Check if 16-bit sort is available (experimental - reduces threadgroup memory by 50%)
    public var has16BitSort: Bool {
        return scatter16Pipeline != nil && sort16Pipeline != nil && render16Pipeline != nil
    }

    /// Encode the pack pass to copy gaussian data to texture
    public func encodePackRenderTexture(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        renderTexture: MTLTexture,
        maxGaussians: Int
    ) {
        guard let packPipeline = packRenderTexturePipeline else { return }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_PackRenderTexture"
            encoder.setComputePipelineState(packPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setTexture(renderTexture, index: 0)

            // One thread per gaussian
            let threadsPerGroup = 256
            let numGroups = (maxGaussians + threadsPerGroup - 1) / threadsPerGroup
            encoder.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    /// Encode render pass using texture-cached gaussian data (for TLB optimization)
    public func encodeRenderTextured(
        commandBuffer: MTLCommandBuffer,
        gaussianTexture: MTLTexture,
        tileOffsets: MTLBuffer,
        tileCounts: MTLBuffer,
        sortedIndices: MTLBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        tilesX: Int,
        tilesY: Int,
        width: Int,
        height: Int,
        tileWidth: Int,
        tileHeight: Int,
        whiteBackground: Bool = false
    ) {
        guard let renderTexPipeline = renderTexturedPipeline else {
            // Fall back to buffer-based render
            return
        }

        var params = LocalSortRenderParamsSwift(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            maxPerTile: 0,  // Not used in LocalSort render
            whiteBackground: whiteBackground ? 1 : 0,
            activeTileCount: 0,  // Not used in LocalSort render
            gaussianCount: 0  // Not used in LocalSort render
        )

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_RenderTextured"
            encoder.setComputePipelineState(renderTexPipeline)
            encoder.setTexture(gaussianTexture, index: 0)
            encoder.setBuffer(tileOffsets, offset: 0, index: 0)
            encoder.setBuffer(tileCounts, offset: 0, index: 1)
            encoder.setBuffer(sortedIndices, offset: 0, index: 2)
            encoder.setTexture(colorTexture, index: 1)
            encoder.setTexture(depthTexture, index: 2)
            encoder.setBytes(&params, length: MemoryLayout<LocalSortRenderParamsSwift>.stride, index: 3)

            // 4×8 threadgroup for 16×16 tile (4×2 pixels per thread)
            let tgs = MTLSize(width: tilesX, height: tilesY, depth: 1)
            let tg = MTLSize(width: 4, height: 8, depth: 1)
            encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Read visible count from header buffer (call after command buffer completes)
    public static func readVisibleCount(from header: MTLBuffer) -> UInt32 {
        let ptr = header.contents().bindMemory(to: CompactedHeaderSwift.self, capacity: 1)
        return ptr.pointee.visibleCount
    }

    /// Debug: Print first few compacted gaussians (call after command buffer completes)
    public static func debugPrintCompacted(from compacted: MTLBuffer, header: MTLBuffer, count: Int = 5) {
        let visibleCount = readVisibleCount(from: header)
        print("[LocalSort Debug] Visible count: \(visibleCount)")

        guard visibleCount > 0, compacted.storageMode == .shared else {
            print("[LocalSort Debug] Cannot read compacted buffer (private storage or empty)")
            return
        }

        let ptr = compacted.contents().bindMemory(to: CompactedGaussianSwift.self, capacity: Int(visibleCount))
        for i in 0..<min(count, Int(visibleCount)) {
            let g = ptr[i]
            print("  [\(i)] pos=(\(g.position_color.x), \(g.position_color.y)) depth=\(g.covariance_depth.w) tiles=(\(g.min_tile.x),\(g.min_tile.y))->(\(g.max_tile.x),\(g.max_tile.y))")
        }
    }

    /// Read overflow flag from header buffer
    public static func readOverflow(from header: MTLBuffer) -> Bool {
        let ptr = header.contents().bindMemory(to: CompactedHeaderSwift.self, capacity: 1)
        return ptr.pointee.overflow != 0
    }

    /// Clear just the header buffer (for temporal mode which clears separately)
    public func encodeClearHeader(
        commandBuffer: MTLCommandBuffer,
        header: MTLBuffer,
        maxCompacted: Int
    ) {
        // Zero out the header (visibleCount = 0, overflow = 0)
        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            blitEncoder.label = "LocalSort_ClearHeader"
            blitEncoder.fill(buffer: header, range: 0..<MemoryLayout<CompactedHeaderSwift>.stride, value: 0)
            blitEncoder.endEncoding()
        }
    }

    // MARK: - 16-bit Sort Experimental (50% less threadgroup memory)

    /// Encode 16-bit scatter (writes 16-bit depth keys + global indices mapping)
    /// Requires: depthKeys16 (ushort buffer), globalIndices (uint buffer)
    public func encodeScatter16(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        tileCounts: MTLBuffer,
        tileOffsets: MTLBuffer,
        depthKeys16: MTLBuffer,      // ushort buffer for 16-bit depth keys
        globalIndices: MTLBuffer,    // uint buffer for local→global mapping
        tilesX: Int,
        maxAssignments: Int,
        tileWidth: Int,
        tileHeight: Int,
        dispatchArgsBuffer: MTLBuffer
    ) {
        guard let scatterPipeline = scatter16Pipeline else { return }

        var tilesXU = UInt32(tilesX)
        var maxAssignmentsU = UInt32(maxAssignments)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_Scatter16_SIMD"
            encoder.setComputePipelineState(scatterPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBuffer(tileCounts, offset: 0, index: 2)
            encoder.setBuffer(tileOffsets, offset: 0, index: 3)
            encoder.setBuffer(depthKeys16, offset: 0, index: 4)
            encoder.setBuffer(globalIndices, offset: 0, index: 5)
            encoder.setBytes(&tilesXU, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&maxAssignmentsU, length: MemoryLayout<UInt32>.stride, index: 7)
            var tileW = Int32(tileWidth)
            var tileH = Int32(tileHeight)
            encoder.setBytes(&tileW, length: MemoryLayout<Int32>.stride, index: 8)
            encoder.setBytes(&tileH, length: MemoryLayout<Int32>.stride, index: 9)

            let tg = MTLSize(width: 128, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: dispatchArgsBuffer, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Encode 16-bit per-tile sort - reads depthKeys16 directly (sequential 2-byte reads!)
    /// Output: sortedLocalIdx (ushort buffer) - sorted local indices per tile
    public func encodeSort16(
        commandBuffer: MTLCommandBuffer,
        depthKeys16: MTLBuffer,          // ushort buffer - 16-bit depth keys from scatter
        globalIndices: MTLBuffer,        // uint buffer - local→global mapping (for render)
        sortedLocalIdx: MTLBuffer,       // ushort buffer - output sorted local indices
        tileOffsets: MTLBuffer,
        tileCounts: MTLBuffer,
        tileCount: Int
    ) {
        guard let sortPipeline = sort16Pipeline else { return }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_PerTileSort16"
            encoder.setComputePipelineState(sortPipeline)
            encoder.setBuffer(depthKeys16, offset: 0, index: 0)  // Sequential 2-byte reads!
            encoder.setBuffer(globalIndices, offset: 0, index: 1)
            encoder.setBuffer(sortedLocalIdx, offset: 0, index: 2)
            encoder.setBuffer(tileOffsets, offset: 0, index: 3)
            encoder.setBuffer(tileCounts, offset: 0, index: 4)

            // One threadgroup per tile
            let tgs = MTLSize(width: tileCount, height: 1, depth: 1)
            let tg = MTLSize(width: 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Encode 16-bit render (two-level indirection: sortedLocalIdx → globalIndices → compacted)
    public func encodeRender16(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        tileOffsets: MTLBuffer,
        tileCounts: MTLBuffer,
        sortedLocalIdx: MTLBuffer,   // ushort buffer - sorted local indices
        globalIndices: MTLBuffer,    // uint buffer - local→global mapping
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        tilesX: Int,
        tilesY: Int,
        width: Int,
        height: Int,
        tileWidth: Int,
        tileHeight: Int,
        whiteBackground: Bool = false
    ) {
        guard let renderPipeline = render16Pipeline else { return }

        var params = LocalSortRenderParamsSwift(
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
            encoder.label = "LocalSort_Render16"
            encoder.setComputePipelineState(renderPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(tileOffsets, offset: 0, index: 1)
            encoder.setBuffer(tileCounts, offset: 0, index: 2)
            encoder.setBuffer(sortedLocalIdx, offset: 0, index: 3)
            encoder.setBuffer(globalIndices, offset: 0, index: 4)
            encoder.setTexture(colorTexture, index: 0)
            encoder.setTexture(depthTexture, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<LocalSortRenderParamsSwift>.stride, index: 5)

            // 4×8 threadgroup for 16×16 tile (4×2 pixels per thread)
            let tgs = MTLSize(width: tilesX, height: tilesY, depth: 1)
            let tg = MTLSize(width: 4, height: 8, depth: 1)
            encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }
}

/// Render params struct for local sort pipeline (matches Metal RenderParams - 40 bytes)
public struct LocalSortRenderParamsSwift {
    public var width: UInt32
    public var height: UInt32
    public var tileWidth: UInt32
    public var tileHeight: UInt32
    public var tilesX: UInt32
    public var tilesY: UInt32
    public var maxPerTile: UInt32
    public var whiteBackground: UInt32
    public var activeTileCount: UInt32
    public var gaussianCount: UInt32
}

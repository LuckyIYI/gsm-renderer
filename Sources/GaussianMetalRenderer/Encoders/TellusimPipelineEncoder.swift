import Metal

/// Tellusim-style pipeline encoder
/// Uses dedicated TellusimShaders.metallib for clean, optimized implementation
/// Pipeline: Clear → Project+Compact+Count → PrefixScan → Scatter → PerTileSort → Render
public final class TellusimPipelineEncoder {
    // Pipeline states from TellusimShaders.metallib
    private let clearPipeline: MTLComputePipelineState
    private let projectCompactCountPipeline: MTLComputePipelineState
    private let projectCompactCountHalfPipeline: MTLComputePipelineState?
    private let prefixScanPipeline: MTLComputePipelineState
    private let scanPartialSumsPipeline: MTLComputePipelineState
    private let finalizeScanPipeline: MTLComputePipelineState
    private let finalizeScanAndZeroPipeline: MTLComputePipelineState
    private let prepareScatterDispatchPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState
    private let perTileSortPipeline: MTLComputePipelineState
    private let renderPipeline: MTLComputePipelineState
    private let zeroPipeline: MTLComputePipelineState

    private let prefixBlockSize = 256
    private let prefixGrainSize = 4

    public init(device: MTLDevice) throws {
        // Load Tellusim shader library
        guard let libraryURL = Bundle.module.url(forResource: "TellusimShaders", withExtension: "metallib"),
              let library = try? device.makeLibrary(URL: libraryURL) else {
            fatalError("Failed to load TellusimShaders.metallib")
        }

        // Load all kernel functions
        guard let clearFn = library.makeFunction(name: "tellusim_clear"),
              let projectFn = library.makeFunction(name: "tellusim_project_compact_count_float"),
              let prefixFn = library.makeFunction(name: "tellusim_prefix_scan"),
              let partialFn = library.makeFunction(name: "tellusim_scan_partial_sums"),
              let finalizeFn = library.makeFunction(name: "tellusim_finalize_scan"),
              let finalizeAndZeroFn = library.makeFunction(name: "tellusim_finalize_scan_and_zero"),
              let prepareDispatchFn = library.makeFunction(name: "tellusim_prepare_scatter_dispatch"),
              let scatterFn = library.makeFunction(name: "tellusim_scatter"),
              let sortFn = library.makeFunction(name: "tellusim_per_tile_sort"),
              let renderFn = library.makeFunction(name: "tellusim_render") else {
            fatalError("Missing required kernel functions in TellusimShaders")
        }

        self.clearPipeline = try device.makeComputePipelineState(function: clearFn)
        self.projectCompactCountPipeline = try device.makeComputePipelineState(function: projectFn)
        self.prefixScanPipeline = try device.makeComputePipelineState(function: prefixFn)
        self.scanPartialSumsPipeline = try device.makeComputePipelineState(function: partialFn)
        self.finalizeScanPipeline = try device.makeComputePipelineState(function: finalizeFn)
        self.finalizeScanAndZeroPipeline = try device.makeComputePipelineState(function: finalizeAndZeroFn)
        self.prepareScatterDispatchPipeline = try device.makeComputePipelineState(function: prepareDispatchFn)
        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)
        self.perTileSortPipeline = try device.makeComputePipelineState(function: sortFn)
        self.renderPipeline = try device.makeComputePipelineState(function: renderFn)

        // Half precision project kernel (optional)
        if let projectHalfFn = library.makeFunction(name: "tellusim_project_compact_count_half") {
            self.projectCompactCountHalfPipeline = try? device.makeComputePipelineState(function: projectHalfFn)
        } else {
            self.projectCompactCountHalfPipeline = nil
        }

        // Zero kernel - reuse from main library for now
        let mainLibraryURL = Bundle.module.url(forResource: "GaussianMetalRenderer", withExtension: "metallib")!
        let mainLibrary = try device.makeLibrary(URL: mainLibraryURL)
        guard let zeroFn = mainLibrary.makeFunction(name: "tileBinningZeroCountsKernel") else {
            fatalError("Missing tileBinningZeroCountsKernel")
        }
        self.zeroPipeline = try device.makeComputePipelineState(function: zeroFn)
    }

    /// Legacy init for backwards compatibility with existing tests
    public convenience init(device: MTLDevice, library: MTLLibrary) throws {
        try self.init(device: device)
    }

    /// Encode the full Tellusim-style pipeline (steps 1-5: Clear → Project → Scan → Scatter → Sort)
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
        // Output buffers (caller provides)
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        tileCounts: MTLBuffer,
        tileOffsets: MTLBuffer,
        partialSums: MTLBuffer,
        sortKeys: MTLBuffer,
        sortIndices: MTLBuffer,
        maxCompacted: Int,
        maxAssignments: Int,
        // Optional
        useHalfWorld: Bool = false,
        skipSort: Bool = false,
        // For sort
        tempSortKeys: MTLBuffer? = nil,
        tempSortIndices: MTLBuffer? = nil
    ) {
        let tileCount = tilesX * tilesY
        let elementsPerGroup = prefixBlockSize * prefixGrainSize
        let actualGroups = max(1, (tileCount + elementsPerGroup - 1) / elementsPerGroup)

        var params = ProjectCompactParamsSwift(
            gaussianCount: UInt32(gaussianCount),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            surfaceWidth: UInt32(surfaceWidth),
            surfaceHeight: UInt32(surfaceHeight),
            maxCompacted: UInt32(maxCompacted)
        )

        var tileCountU = UInt32(tileCount)
        var maxCompactedU = UInt32(maxCompacted)
        var tilesXU = UInt32(tilesX)
        var maxAssignmentsU = UInt32(maxAssignments)
        var cameraUniforms = camera

        // === 1. CLEAR ===
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Tellusim_Clear"
            encoder.setComputePipelineState(clearPipeline)
            encoder.setBuffer(tileCounts, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBytes(&tileCountU, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&maxCompactedU, length: MemoryLayout<UInt32>.stride, index: 3)

            let threads = MTLSize(width: max(tileCount, 1), height: 1, depth: 1)
            let tg = MTLSize(width: clearPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === 2. PROJECT + COMPACT + COUNT ===
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Tellusim_ProjectCompactCount"

            let pipeline: MTLComputePipelineState
            if useHalfWorld, let halfPipe = projectCompactCountHalfPipeline {
                pipeline = halfPipe
            } else {
                pipeline = projectCompactCountPipeline
            }
            encoder.setComputePipelineState(pipeline)

            encoder.setBuffer(worldGaussians, offset: 0, index: 0)
            encoder.setBuffer(harmonics, offset: 0, index: 1)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 2)
            encoder.setBuffer(compactedHeader, offset: 0, index: 3)
            encoder.setBuffer(tileCounts, offset: 0, index: 4)
            encoder.setBytes(&cameraUniforms, length: MemoryLayout<CameraUniformsSwift>.stride, index: 5)
            encoder.setBytes(&params, length: MemoryLayout<ProjectCompactParamsSwift>.stride, index: 6)

            let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
            let tg = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === 3. PREFIX SCAN ===
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Tellusim_PrefixScan"
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
            encoder.label = "Tellusim_ScanPartialSums"
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
            encoder.label = "Tellusim_FinalizeScanAndZero"
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
        let dispatchArgsBuffer = commandBuffer.device.makeBuffer(length: 12, options: .storageModeShared)!
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Tellusim_PrepareScatterDispatch"
            encoder.setComputePipelineState(prepareScatterDispatchPipeline)
            encoder.setBuffer(compactedHeader, offset: 0, index: 0)
            encoder.setBuffer(dispatchArgsBuffer, offset: 0, index: 1)
            var tgWidth = UInt32(scatterPipeline.threadExecutionWidth)
            encoder.setBytes(&tgWidth, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // === 5b. SCATTER (GPU-driven indirect dispatch) ===
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Tellusim_Scatter"
            encoder.setComputePipelineState(scatterPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBuffer(tileCounts, offset: 0, index: 2)
            encoder.setBuffer(tileOffsets, offset: 0, index: 3)
            encoder.setBuffer(sortKeys, offset: 0, index: 4)
            encoder.setBuffer(sortIndices, offset: 0, index: 5)
            encoder.setBytes(&tilesXU, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&maxAssignmentsU, length: MemoryLayout<UInt32>.stride, index: 7)
            var tileW = Int32(tileWidth)
            var tileH = Int32(tileHeight)
            encoder.setBytes(&tileW, length: MemoryLayout<Int32>.stride, index: 8)
            encoder.setBytes(&tileH, length: MemoryLayout<Int32>.stride, index: 9)

            let tg = MTLSize(width: scatterPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: dispatchArgsBuffer, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === 6. PER-TILE SORT ===
        if !skipSort, let tempK = tempSortKeys, let tempV = tempSortIndices {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Tellusim_PerTileSort"
                encoder.setComputePipelineState(perTileSortPipeline)
                encoder.setBuffer(sortKeys, offset: 0, index: 0)
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
        var params = TellusimRenderParamsSwift(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            whiteBackground: whiteBackground ? 1 : 0,
            _pad: 0
        )

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Tellusim_Render"
            encoder.setComputePipelineState(renderPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(tileOffsets, offset: 0, index: 1)
            encoder.setBuffer(tileCounts, offset: 0, index: 2)
            encoder.setBuffer(sortedIndices, offset: 0, index: 3)
            encoder.setTexture(colorTexture, index: 0)
            encoder.setTexture(depthTexture, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<TellusimRenderParamsSwift>.stride, index: 4)

            // 8x8 threadgroup, one per tile
            let tgs = MTLSize(width: tilesX, height: tilesY, depth: 1)
            let tg = MTLSize(width: 8, height: 8, depth: 1)
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
        print("[Tellusim Debug] Visible count: \(visibleCount)")

        guard visibleCount > 0, compacted.storageMode == .shared else {
            print("[Tellusim Debug] Cannot read compacted buffer (private storage or empty)")
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
}

// Render params struct for Tellusim pipeline
public struct TellusimRenderParamsSwift {
    public var width: UInt32
    public var height: UInt32
    public var tileWidth: UInt32
    public var tileHeight: UInt32
    public var tilesX: UInt32
    public var tilesY: UInt32
    public var whiteBackground: UInt32
    public var _pad: UInt32
}

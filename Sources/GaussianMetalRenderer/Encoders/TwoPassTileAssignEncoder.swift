import Metal

/// Two-pass tile assignment encoder with deterministic compaction:
/// MarkVisibility → PrefixSum → ScatterCompact → PrepareDispatch → Count → PrefixSum → Scatter
/// Uses prefix-sum based compaction for deterministic ordering (atomic-based was non-deterministic)
final class TwoPassTileAssignEncoder {
    // Deterministic compaction pipelines
    private let markVisibilityPipeline: MTLComputePipelineState
    private let scatterCompactPipeline: MTLComputePipelineState
    private let prepareIndirectPipeline: MTLComputePipelineState

    // Tile count/scatter pipelines
    private let countPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState

    // Prefix sum pipelines
    private let blockReducePipeline: MTLComputePipelineState
    private let singleBlockScanPipeline: MTLComputePipelineState
    private let blockScanPipeline: MTLComputePipelineState
    private let writeTotalPipeline: MTLComputePipelineState

    private let blockSize = 256
    let threadgroupSize: Int

    /// Buffers required for indirect dispatch
    struct IndirectBuffers {
        let visibleIndices: MTLBuffer     // Compacted visible gaussian indices [maxGaussians]
        let visibleCount: MTLBuffer       // Visible count (4 bytes)
        let indirectDispatchArgs: MTLBuffer  // Dispatch args (12 bytes)
    }

    init(device: MTLDevice, library: MTLLibrary) throws {
        // Deterministic compaction kernels
        guard let markFn = library.makeFunction(name: "markVisibilityKernel"),
              let scatterCompactFn = library.makeFunction(name: "scatterCompactKernel"),
              let prepareIndirectFn = library.makeFunction(name: "prepareIndirectDispatchKernel"),
              let countFn = library.makeFunction(name: "tileCountIndirectKernel"),
              let scatterFn = library.makeFunction(name: "tileScatterIndirectKernel")
        else {
            fatalError("Tile assign kernels not found in library")
        }

        // Prefix sum kernels
        guard let reduceFn = library.makeFunction(name: "blockReduceKernel"),
              let singleScanFn = library.makeFunction(name: "singleBlockScanKernel"),
              let blockScanFn = library.makeFunction(name: "blockScanKernel"),
              let writeTotalFn = library.makeFunction(name: "writeTotalCountKernel")
        else {
            fatalError("Prefix sum kernels not found in library")
        }

        self.markVisibilityPipeline = try device.makeComputePipelineState(function: markFn)
        self.scatterCompactPipeline = try device.makeComputePipelineState(function: scatterCompactFn)
        self.prepareIndirectPipeline = try device.makeComputePipelineState(function: prepareIndirectFn)
        self.countPipeline = try device.makeComputePipelineState(function: countFn)
        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)

        self.blockReducePipeline = try device.makeComputePipelineState(function: reduceFn)
        self.singleBlockScanPipeline = try device.makeComputePipelineState(function: singleScanFn)
        self.blockScanPipeline = try device.makeComputePipelineState(function: blockScanFn)
        self.writeTotalPipeline = try device.makeComputePipelineState(function: writeTotalFn)

        self.threadgroupSize = min(countPipeline.maxTotalThreadsPerThreadgroup, 256)
    }

    /// Encode hierarchical prefix sum on a buffer
    /// Returns the numBlocksU32 value needed for reading total from blockSums
    private func encodePrefixSum(
        commandBuffer: MTLCommandBuffer,
        dataBuffer: MTLBuffer,
        blockSumsBuffer: MTLBuffer,
        count: Int,
        labelPrefix: String
    ) -> UInt32 {
        let numBlocks = (count + blockSize - 1) / blockSize
        var countU32 = UInt32(count)
        var numBlocksU32 = UInt32(numBlocks)

        // Block-level reduce (dataBuffer → blockSums)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "\(labelPrefix)BlockReduce"
            encoder.setComputePipelineState(blockReducePipeline)
            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBuffer(blockSumsBuffer, offset: 0, index: 1)
            encoder.setBytes(&countU32, length: 4, index: 2)

            let threads = MTLSize(width: numBlocks * blockSize, height: 1, depth: 1)
            let tg = MTLSize(width: blockSize, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        if numBlocks <= blockSize {
            // Single-level scan: blockSums fits in one threadgroup
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "\(labelPrefix)SingleBlockScan"
                encoder.setComputePipelineState(singleBlockScanPipeline)
                encoder.setBuffer(blockSumsBuffer, offset: 0, index: 0)
                encoder.setBytes(&numBlocksU32, length: 4, index: 1)

                let tg = MTLSize(width: blockSize, height: 1, depth: 1)
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }
        } else {
            // Two-level scan for numBlocks > 256
            let level2Blocks = (numBlocks + blockSize - 1) / blockSize
            let level2Offset = (numBlocks + 1) * MemoryLayout<UInt32>.stride

            var numBlocksL2 = UInt32(numBlocks)
            var level2BlocksU32 = UInt32(level2Blocks)

            // Reduce blockSums → level2Sums
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "\(labelPrefix)Level2Reduce"
                encoder.setComputePipelineState(blockReducePipeline)
                encoder.setBuffer(blockSumsBuffer, offset: 0, index: 0)
                encoder.setBuffer(blockSumsBuffer, offset: level2Offset, index: 1)
                encoder.setBytes(&numBlocksL2, length: 4, index: 2)

                let threads = MTLSize(width: level2Blocks * blockSize, height: 1, depth: 1)
                let tg = MTLSize(width: blockSize, height: 1, depth: 1)
                encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }

            // Single-block scan of level2Sums
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "\(labelPrefix)Level2Scan"
                encoder.setComputePipelineState(singleBlockScanPipeline)
                encoder.setBuffer(blockSumsBuffer, offset: level2Offset, index: 0)
                encoder.setBytes(&level2BlocksU32, length: 4, index: 1)

                let tg = MTLSize(width: blockSize, height: 1, depth: 1)
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }

            // Block scan of blockSums using level2Sums as offsets
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "\(labelPrefix)Level1Scan"
                encoder.setComputePipelineState(blockScanPipeline)
                encoder.setBuffer(blockSumsBuffer, offset: 0, index: 0)
                encoder.setBuffer(blockSumsBuffer, offset: 0, index: 1)
                encoder.setBuffer(blockSumsBuffer, offset: level2Offset, index: 2)
                encoder.setBytes(&numBlocksU32, length: 4, index: 3)

                let threads = MTLSize(width: level2Blocks * blockSize, height: 1, depth: 1)
                let tg = MTLSize(width: blockSize, height: 1, depth: 1)
                encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }

            numBlocksU32 = UInt32((level2Offset / MemoryLayout<UInt32>.stride) + level2Blocks)
        }

        // Block-level scan with offsets
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "\(labelPrefix)BlockScan"
            encoder.setComputePipelineState(blockScanPipeline)
            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBuffer(dataBuffer, offset: 0, index: 1)
            encoder.setBuffer(blockSumsBuffer, offset: 0, index: 2)
            encoder.setBytes(&countU32, length: 4, index: 3)

            let threads = MTLSize(width: numBlocks * blockSize, height: 1, depth: 1)
            let tg = MTLSize(width: blockSize, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        return numBlocksU32
    }

    /// Encode two-pass tile assignment with deterministic compaction
    /// Flow: MarkVisibility → PrefixSum → ScatterCompact → PrepareDispatch →
    ///       ClearCoverage → IndirectCount → PrefixSum → WriteTotalCount → IndirectScatter
    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        tileWidth: Int,
        tileHeight: Int,
        tilesX: Int,
        maxAssignments: Int,
        boundsBuffer: MTLBuffer,
        coverageBuffer: MTLBuffer,  // Used for visibility marks, then tile counts
        renderData: MTLBuffer,
        tileIndicesBuffer: MTLBuffer,
        tileIdsBuffer: MTLBuffer,
        tileAssignmentHeader: MTLBuffer,
        blockSumsBuffer: MTLBuffer,
        indirectBuffers: IndirectBuffers
    ) {
        // =================================================================
        // Pass 0: Deterministic Compaction via Prefix Sum
        // =================================================================

        // 0a. Mark visibility: write 1/0 to coverageBuffer
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "MarkVisibility"
            encoder.setComputePipelineState(markVisibilityPipeline)
            encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 1)
            var count = UInt32(gaussianCount)
            encoder.setBytes(&count, length: 4, index: 2)

            let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
            let tg = MTLSize(width: threadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // 0b. Prefix sum on visibility marks → exclusive offsets in coverageBuffer
        _ = encodePrefixSum(
            commandBuffer: commandBuffer,
            dataBuffer: coverageBuffer,
            blockSumsBuffer: blockSumsBuffer,
            count: gaussianCount,
            labelPrefix: "Compact"
        )

        // 0c. Scatter visible indices using prefix sum offsets
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "ScatterCompact"
            encoder.setComputePipelineState(scatterCompactPipeline)
            encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 1)
            encoder.setBuffer(indirectBuffers.visibleIndices, offset: 0, index: 2)
            encoder.setBuffer(indirectBuffers.visibleCount, offset: 0, index: 3)
            var count = UInt32(gaussianCount)
            encoder.setBytes(&count, length: 4, index: 4)

            let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
            let tg = MTLSize(width: threadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // 0d. Prepare indirect dispatch args based on visible count
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "PrepareIndirectDispatch"
            encoder.setComputePipelineState(prepareIndirectPipeline)
            encoder.setBuffer(indirectBuffers.visibleCount, offset: 0, index: 0)
            encoder.setBuffer(indirectBuffers.indirectDispatchArgs, offset: 0, index: 1)
            var tgSize = UInt32(threadgroupSize)
            encoder.setBytes(&tgSize, length: 4, index: 2)

            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // =================================================================
        // Pass 1: Count tiles per visible gaussian (indirect dispatch)
        // =================================================================

        // Clear coverageBuffer for tile counts (prefix sum contaminated it)
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "ClearCoverage"
            let coverageSize = gaussianCount * MemoryLayout<UInt32>.stride
            blit.fill(buffer: coverageBuffer, range: 0..<coverageSize, value: 0)
            blit.endEncoding()
        }

        var params = TileAssignParamsSwift(
            gaussianCount: UInt32(gaussianCount),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            maxAssignments: UInt32(maxAssignments)
        )

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "TileCountIndirect"
            encoder.setComputePipelineState(countPipeline)
            encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
            encoder.setBuffer(renderData, offset: 0, index: 1)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 2)
            encoder.setBuffer(indirectBuffers.visibleIndices, offset: 0, index: 3)
            encoder.setBytes(&params, length: MemoryLayout<TileAssignParamsSwift>.stride, index: 4)

            let tg = MTLSize(width: threadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: indirectBuffers.indirectDispatchArgs,
                                         indirectBufferOffset: 0,
                                         threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // =================================================================
        // Pass 2: Hierarchical Prefix Sum for tile offsets
        // =================================================================
        let numBlocksU32 = encodePrefixSum(
            commandBuffer: commandBuffer,
            dataBuffer: coverageBuffer,
            blockSumsBuffer: blockSumsBuffer,
            count: gaussianCount,
            labelPrefix: "TileOffset"
        )

        // Write total count to header
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "WriteTotalCount"
            encoder.setComputePipelineState(writeTotalPipeline)
            encoder.setBuffer(blockSumsBuffer, offset: 0, index: 0)
            encoder.setBuffer(tileAssignmentHeader, offset: 0, index: 1)
            var numBlocksVar = numBlocksU32
            encoder.setBytes(&numBlocksVar, length: 4, index: 2)

            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // =================================================================
        // Pass 3: Scatter using precomputed offsets (indirect dispatch)
        // =================================================================
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "TileScatterIndirect"
            encoder.setComputePipelineState(scatterPipeline)
            encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
            encoder.setBuffer(renderData, offset: 0, index: 1)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 2)
            encoder.setBuffer(indirectBuffers.visibleIndices, offset: 0, index: 3)
            encoder.setBuffer(tileIndicesBuffer, offset: 0, index: 4)
            encoder.setBuffer(tileIdsBuffer, offset: 0, index: 5)
            encoder.setBytes(&params, length: MemoryLayout<TileAssignParamsSwift>.stride, index: 6)

            let tg = MTLSize(width: threadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: indirectBuffers.indirectDispatchArgs,
                                         indirectBufferOffset: 0,
                                         threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }
}

// Swift struct matching Metal TileAssignParams
struct TileAssignParamsSwift {
    var gaussianCount: UInt32
    var tileWidth: UInt32
    var tileHeight: UInt32
    var tilesX: UInt32
    var maxAssignments: UInt32
}

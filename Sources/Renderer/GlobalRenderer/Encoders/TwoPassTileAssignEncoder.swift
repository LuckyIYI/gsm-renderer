import Metal
import RendererTypes

/// Two-pass tile assignment encoder with deterministic compaction:
/// MarkVisibility → PrefixSum → ScatterCompact → PrepareDispatch → Count → PrefixSum → Scatter
final class TwoPassTileAssignEncoder {
    // Deterministic compaction pipelines
    private let markVisibilityPipeline: MTLComputePipelineState
    private let scatterCompactPipeline: MTLComputePipelineState
    private let prepareIndirectPipeline: MTLComputePipelineState

    // Tile count/scatter pipelines
    private let countPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState

    // Indirect prefix sum pipelines (count read from buffer)
    private let preparePrefixSumDispatchPipeline: MTLComputePipelineState
    private let blockReduceIndirectPipeline: MTLComputePipelineState
    private let singleBlockScanIndirectPipeline: MTLComputePipelineState
    private let blockScanIndirectPipeline: MTLComputePipelineState
    private let writeTotalIndirectPipeline: MTLComputePipelineState

    // Prefix sum pipelines
    private let blockReducePipeline: MTLComputePipelineState
    private let singleBlockScanPipeline: MTLComputePipelineState
    private let blockScanPipeline: MTLComputePipelineState
    private let writeTotalPipeline: MTLComputePipelineState

    private let blockSize = 256
    let threadgroupSize: Int

    /// Buffers required for indirect dispatch
    struct IndirectBuffers {
        let visibleIndices: MTLBuffer // Compacted visible gaussian indices [maxGaussians]
        let visibleCount: MTLBuffer // Visible count (4 bytes)
        let indirectDispatchArgs: MTLBuffer // Dispatch args (12 bytes)
    }

    init(device: MTLDevice, library: MTLLibrary) throws {
        // Deterministic compaction kernels
        guard let markFn = library.makeFunction(name: "markVisibilityKernel"),
              let scatterCompactFn = library.makeFunction(name: "scatterCompactKernel"),
              let prepareIndirectFn = library.makeFunction(name: "prepareIndirectDispatchKernel"),
              let countFn = library.makeFunction(name: "tileCountIndirectKernel"),
              let scatterFn = library.makeFunction(name: "tileScatterIndirectKernel")
        else {
            throw RendererError.failedToCreatePipeline("Tile assign kernels not found in library")
        }

        // Prefix sum kernels
        guard let reduceFn = library.makeFunction(name: "blockReduceKernel"),
              let singleScanFn = library.makeFunction(name: "singleBlockScanKernel"),
              let blockScanFn = library.makeFunction(name: "blockScanKernel"),
              let writeTotalFn = library.makeFunction(name: "writeTotalCountKernel")
        else {
            throw RendererError.failedToCreatePipeline("Prefix sum kernels not found in library")
        }

        // Indirect prefix sum kernels
        guard let prepPrefixFn = library.makeFunction(name: "preparePrefixSumDispatchKernel"),
              let reduceIndirectFn = library.makeFunction(name: "blockReduceIndirectKernel"),
              let singleScanIndirectFn = library.makeFunction(name: "singleBlockScanIndirectKernel"),
              let blockScanIndirectFn = library.makeFunction(name: "blockScanIndirectKernel"),
              let writeTotalIndirectFn = library.makeFunction(name: "writeTotalCountIndirectKernel")
        else {
            throw RendererError.failedToCreatePipeline("Indirect prefix sum kernels not found in library")
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

        self.preparePrefixSumDispatchPipeline = try device.makeComputePipelineState(function: prepPrefixFn)
        self.blockReduceIndirectPipeline = try device.makeComputePipelineState(function: reduceIndirectFn)
        self.singleBlockScanIndirectPipeline = try device.makeComputePipelineState(function: singleScanIndirectFn)
        self.blockScanIndirectPipeline = try device.makeComputePipelineState(function: blockScanIndirectFn)
        self.writeTotalIndirectPipeline = try device.makeComputePipelineState(function: writeTotalIndirectFn)

        self.threadgroupSize = min(self.countPipeline.maxTotalThreadsPerThreadgroup, 256)
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
        let numBlocks = (count + self.blockSize - 1) / self.blockSize
        var countU32 = UInt32(count)
        var numBlocksU32 = UInt32(numBlocks)

        // Block-level reduce (dataBuffer → blockSums)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "\(labelPrefix)BlockReduce"
            encoder.setComputePipelineState(self.blockReducePipeline)
            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBuffer(blockSumsBuffer, offset: 0, index: 1)
            encoder.setBytes(&countU32, length: 4, index: 2)

            let threads = MTLSize(width: numBlocks * self.blockSize, height: 1, depth: 1)
            let tg = MTLSize(width: blockSize, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        if numBlocks <= self.blockSize {
            // Single-level scan: blockSums fits in one threadgroup
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "\(labelPrefix)SingleBlockScan"
                encoder.setComputePipelineState(self.singleBlockScanPipeline)
                encoder.setBuffer(blockSumsBuffer, offset: 0, index: 0)
                encoder.setBytes(&numBlocksU32, length: 4, index: 1)

                let tg = MTLSize(width: blockSize, height: 1, depth: 1)
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }
        } else {
            // Two-level scan for numBlocks > 256
            let level2Blocks = (numBlocks + self.blockSize - 1) / self.blockSize
            let level2Offset = (numBlocks + 1) * MemoryLayout<UInt32>.stride

            var numBlocksL2 = UInt32(numBlocks)
            var level2BlocksU32 = UInt32(level2Blocks)

            // Reduce blockSums → level2Sums
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "\(labelPrefix)Level2Reduce"
                encoder.setComputePipelineState(self.blockReducePipeline)
                encoder.setBuffer(blockSumsBuffer, offset: 0, index: 0)
                encoder.setBuffer(blockSumsBuffer, offset: level2Offset, index: 1)
                encoder.setBytes(&numBlocksL2, length: 4, index: 2)

                let threads = MTLSize(width: level2Blocks * self.blockSize, height: 1, depth: 1)
                let tg = MTLSize(width: blockSize, height: 1, depth: 1)
                encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }

            // Single-block scan of level2Sums
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "\(labelPrefix)Level2Scan"
                encoder.setComputePipelineState(self.singleBlockScanPipeline)
                encoder.setBuffer(blockSumsBuffer, offset: level2Offset, index: 0)
                encoder.setBytes(&level2BlocksU32, length: 4, index: 1)

                let tg = MTLSize(width: blockSize, height: 1, depth: 1)
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }

            // Block scan of blockSums using level2Sums as offsets
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "\(labelPrefix)Level1Scan"
                encoder.setComputePipelineState(self.blockScanPipeline)
                encoder.setBuffer(blockSumsBuffer, offset: 0, index: 0)
                encoder.setBuffer(blockSumsBuffer, offset: 0, index: 1)
                encoder.setBuffer(blockSumsBuffer, offset: level2Offset, index: 2)
                encoder.setBytes(&numBlocksU32, length: 4, index: 3)

                let threads = MTLSize(width: level2Blocks * self.blockSize, height: 1, depth: 1)
                let tg = MTLSize(width: blockSize, height: 1, depth: 1)
                encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
                encoder.endEncoding()
            }

            numBlocksU32 = UInt32((level2Offset / MemoryLayout<UInt32>.stride) + level2Blocks)
        }

        // Block-level scan with offsets
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "\(labelPrefix)BlockScan"
            encoder.setComputePipelineState(self.blockScanPipeline)
            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBuffer(dataBuffer, offset: 0, index: 1)
            encoder.setBuffer(blockSumsBuffer, offset: 0, index: 2)
            encoder.setBytes(&countU32, length: 4, index: 3)

            let threads = MTLSize(width: numBlocks * self.blockSize, height: 1, depth: 1)
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
        coverageBuffer: MTLBuffer, // Used for visibility marks, then tile counts
        renderData: MTLBuffer,
        tileIndicesBuffer: MTLBuffer,
        tileIdsBuffer: MTLBuffer,
        tileAssignmentHeader: MTLBuffer,
        blockSumsBuffer: MTLBuffer,
        indirectBuffers: IndirectBuffers
    ) {
        // 0a. Mark visibility: write 1/0 to coverageBuffer
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "MarkVisibility"
            encoder.setComputePipelineState(self.markVisibilityPipeline)
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
        _ = self.encodePrefixSum(
            commandBuffer: commandBuffer,
            dataBuffer: coverageBuffer,
            blockSumsBuffer: blockSumsBuffer,
            count: gaussianCount,
            labelPrefix: "Compact"
        )

        // 0c. Scatter visible indices using prefix sum offsets
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "ScatterCompact"
            encoder.setComputePipelineState(self.scatterCompactPipeline)
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
            encoder.setComputePipelineState(self.prepareIndirectPipeline)
            encoder.setBuffer(indirectBuffers.visibleCount, offset: 0, index: 0)
            encoder.setBuffer(indirectBuffers.indirectDispatchArgs, offset: 0, index: 1)
            var tgSize = UInt32(threadgroupSize)
            encoder.setBytes(&tgSize, length: 4, index: 2)

            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // Clear coverageBuffer for tile counts (prefix sum contaminated it)
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "ClearCoverage"
            let coverageSize = gaussianCount * MemoryLayout<UInt32>.stride
            blit.fill(buffer: coverageBuffer, range: 0 ..< coverageSize, value: 0)
            blit.endEncoding()
        }

        var params = TileAssignParams(
            gaussianCount: UInt32(gaussianCount),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            maxAssignments: UInt32(maxAssignments)
        )

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "TileCountIndirect"
            encoder.setComputePipelineState(self.countPipeline)
            encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
            encoder.setBuffer(renderData, offset: 0, index: 1)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 2)
            encoder.setBuffer(indirectBuffers.visibleIndices, offset: 0, index: 3)
            encoder.setBuffer(indirectBuffers.visibleCount, offset: 0, index: 4)
            encoder.setBytes(&params, length: MemoryLayout<TileAssignParams>.stride, index: 5)

            let tg = MTLSize(width: threadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: indirectBuffers.indirectDispatchArgs,
                                         indirectBufferOffset: 0,
                                         threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        let numBlocksU32 = self.encodePrefixSum(
            commandBuffer: commandBuffer,
            dataBuffer: coverageBuffer,
            blockSumsBuffer: blockSumsBuffer,
            count: gaussianCount,
            labelPrefix: "TileOffset"
        )

        // Write total count to header
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "WriteTotalCount"
            encoder.setComputePipelineState(self.writeTotalPipeline)
            encoder.setBuffer(blockSumsBuffer, offset: 0, index: 0)
            encoder.setBuffer(tileAssignmentHeader, offset: 0, index: 1)
            var numBlocksVar = numBlocksU32
            encoder.setBytes(&numBlocksVar, length: 4, index: 2)

            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "TileScatterIndirect"
            encoder.setComputePipelineState(self.scatterPipeline)
            encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
            encoder.setBuffer(renderData, offset: 0, index: 1)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 2)
            encoder.setBuffer(indirectBuffers.visibleIndices, offset: 0, index: 3)
            encoder.setBuffer(indirectBuffers.visibleCount, offset: 0, index: 4)
            encoder.setBuffer(tileIndicesBuffer, offset: 0, index: 5)
            encoder.setBuffer(tileIdsBuffer, offset: 0, index: 6)
            encoder.setBytes(&params, length: MemoryLayout<TileAssignParams>.stride, index: 7)

            let tg = MTLSize(width: threadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: indirectBuffers.indirectDispatchArgs,
                                         indirectBufferOffset: 0,
                                         threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Encode tile assignment when `visibleIndices/visibleCount` are already populated.
    /// Flow: PrepareDispatch → IndirectCount → PrefixSum(count=visibleCount) → WriteTotal → IndirectScatter
    func encodeFromVisibleList(
        commandBuffer: MTLCommandBuffer,
        tileWidth: Int,
        tileHeight: Int,
        tilesX: Int,
        maxAssignments: Int,
        boundsBuffer: MTLBuffer,
        coverageBuffer: MTLBuffer, // Used as tileCounts, then overwritten with offsets
        renderData: MTLBuffer,
        tileIndicesBuffer: MTLBuffer,
        tileIdsBuffer: MTLBuffer,
        tileAssignmentHeader: MTLBuffer,
        blockSumsBuffer: MTLBuffer,
        blockSumsLevel2OffsetBytes: Int,
        prefixSumDispatchArgs: MTLBuffer,
        prefixSumBlockCountBuffer: MTLBuffer,
        indirectBuffers: IndirectBuffers
    ) {
        // 0) Prepare indirect dispatch args based on visible count
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "PrepareIndirectDispatch(Visible)"
            encoder.setComputePipelineState(self.prepareIndirectPipeline)
            encoder.setBuffer(indirectBuffers.visibleCount, offset: 0, index: 0)
            encoder.setBuffer(indirectBuffers.indirectDispatchArgs, offset: 0, index: 1)
            var tgSize = UInt32(threadgroupSize)
            encoder.setBytes(&tgSize, length: 4, index: 2)

            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // 1) TileCountIndirect: coverageBuffer[0..visibleCount) = tileCounts
        var params = TileAssignParams(
            gaussianCount: 0,
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            maxAssignments: UInt32(maxAssignments)
        )

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "TileCountIndirect(Visible)"
            encoder.setComputePipelineState(self.countPipeline)
            encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
            encoder.setBuffer(renderData, offset: 0, index: 1)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 2)
            encoder.setBuffer(indirectBuffers.visibleIndices, offset: 0, index: 3)
            encoder.setBuffer(indirectBuffers.visibleCount, offset: 0, index: 4)
            encoder.setBytes(&params, length: MemoryLayout<TileAssignParams>.stride, index: 5)

            let tg = MTLSize(width: threadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: indirectBuffers.indirectDispatchArgs,
                                         indirectBufferOffset: 0,
                                         threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // 2) Prefix sum on coverageBuffer with count = visibleCount (GPU-only, no clears)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "PreparePrefixSumDispatch(Visible)"
            encoder.setComputePipelineState(self.preparePrefixSumDispatchPipeline)
            encoder.setBuffer(indirectBuffers.visibleCount, offset: 0, index: 0)
            encoder.setBuffer(prefixSumDispatchArgs, offset: 0, index: 1)
            encoder.setBuffer(prefixSumBlockCountBuffer, offset: 0, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        let prefixBlockSize = 256
        let prefixTG = MTLSize(width: prefixBlockSize, height: 1, depth: 1)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "PrefixBlockReduce(Visible)"
            encoder.setComputePipelineState(self.blockReduceIndirectPipeline)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 0)
            encoder.setBuffer(blockSumsBuffer, offset: 0, index: 1)
            encoder.setBuffer(indirectBuffers.visibleCount, offset: 0, index: 2)
            encoder.dispatchThreadgroups(indirectBuffer: prefixSumDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: prefixTG)
            encoder.endEncoding()
        }

        // Reduce level-1 block sums into level-2 block sums (always, even if level2Blocks == 1).
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "PrefixBlockReduce(Level2)"
            encoder.setComputePipelineState(self.blockReduceIndirectPipeline)
            encoder.setBuffer(blockSumsBuffer, offset: 0, index: 0)
            encoder.setBuffer(blockSumsBuffer, offset: blockSumsLevel2OffsetBytes, index: 1)
            encoder.setBuffer(prefixSumBlockCountBuffer, offset: 0, index: 2) // level1Blocks
            encoder.dispatchThreadgroups(indirectBuffer: prefixSumDispatchArgs,
                                         indirectBufferOffset: MemoryLayout<DispatchIndirectArgsSwift>.stride,
                                         threadsPerThreadgroup: prefixTG)
            encoder.endEncoding()
        }

        // Scan level-2 block sums in a single threadgroup (level2Blocks <= 256 for our maxGaussians).
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "PrefixSingleBlockScan(Level2)"
            encoder.setComputePipelineState(self.singleBlockScanIndirectPipeline)
            encoder.setBuffer(blockSumsBuffer, offset: blockSumsLevel2OffsetBytes, index: 0)
            encoder.setBuffer(prefixSumBlockCountBuffer, offset: MemoryLayout<UInt32>.stride, index: 1) // level2Blocks
            encoder.dispatchThreads(MTLSize(width: prefixBlockSize, height: 1, depth: 1),
                                    threadsPerThreadgroup: prefixTG)
            encoder.endEncoding()
        }

        // Convert level-1 block sums to exclusive offsets using scanned level-2 offsets.
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "PrefixBlockScan(Level1)"
            encoder.setComputePipelineState(self.blockScanIndirectPipeline)
            encoder.setBuffer(blockSumsBuffer, offset: 0, index: 0)
            encoder.setBuffer(blockSumsBuffer, offset: 0, index: 1)
            encoder.setBuffer(blockSumsBuffer, offset: blockSumsLevel2OffsetBytes, index: 2)
            encoder.setBuffer(prefixSumBlockCountBuffer, offset: 0, index: 3) // level1Blocks
            encoder.dispatchThreadgroups(indirectBuffer: prefixSumDispatchArgs,
                                         indirectBufferOffset: MemoryLayout<DispatchIndirectArgsSwift>.stride,
                                         threadsPerThreadgroup: prefixTG)
            encoder.endEncoding()
        }

        // Scan original buffer using level-1 block offsets.
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "PrefixBlockScan(Visible)"
            encoder.setComputePipelineState(self.blockScanIndirectPipeline)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 0)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 1)
            encoder.setBuffer(blockSumsBuffer, offset: 0, index: 2) // level1Offsets
            encoder.setBuffer(indirectBuffers.visibleCount, offset: 0, index: 3)
            encoder.dispatchThreadgroups(indirectBuffer: prefixSumDispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: prefixTG)
            encoder.endEncoding()
        }

        // 3) Write total assignments to header from level-2 total.
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "WriteTotalCount(Visible)"
            encoder.setComputePipelineState(self.writeTotalIndirectPipeline)
            encoder.setBuffer(blockSumsBuffer, offset: blockSumsLevel2OffsetBytes, index: 0)
            encoder.setBuffer(tileAssignmentHeader, offset: 0, index: 1)
            encoder.setBuffer(prefixSumBlockCountBuffer, offset: MemoryLayout<UInt32>.stride, index: 2) // level2Blocks
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // 4) TileScatterIndirect
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "TileScatterIndirect(Visible)"
            encoder.setComputePipelineState(self.scatterPipeline)
            encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
            encoder.setBuffer(renderData, offset: 0, index: 1)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 2)
            encoder.setBuffer(indirectBuffers.visibleIndices, offset: 0, index: 3)
            encoder.setBuffer(indirectBuffers.visibleCount, offset: 0, index: 4)
            encoder.setBuffer(tileIndicesBuffer, offset: 0, index: 5)
            encoder.setBuffer(tileIdsBuffer, offset: 0, index: 6)
            encoder.setBytes(&params, length: MemoryLayout<TileAssignParams>.stride, index: 7)

            let tg = MTLSize(width: threadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: indirectBuffers.indirectDispatchArgs,
                                         indirectBufferOffset: 0,
                                         threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }
}

// TileAssignParams is defined in BridgingTypes.h

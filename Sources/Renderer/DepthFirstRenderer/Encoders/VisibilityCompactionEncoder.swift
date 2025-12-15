import Metal
import RendererTypes

/// Deterministically compacts visible gaussians into contiguous depth-sort buffers.
/// Uses a hierarchical exclusive prefix sum over `nTouchedTiles > 0` (gid order) and scatters:
/// - `depthKeys[outIdx] = preDepthKeys[gid]`
/// - `primitiveIndices[outIdx] = gid`
/// Also writes `visibleCount[0]`.
final class VisibilityCompactionEncoder {
    private let markReducePipeline: MTLComputePipelineState
    private let reducePipeline: MTLComputePipelineState
    private let scanPipeline: MTLComputePipelineState
    private let markScanPipeline: MTLComputePipelineState
    private let singleBlockScanPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState

    private let blockSize: Int = 256

    init(device: MTLDevice, library: MTLLibrary, depthKeyPrecision: RadixSortKeyPrecision = .bits32) throws {
        guard let markReduceFn = library.makeFunction(name: "visibilityBlockReduceKernel"),
              let reduceFn = library.makeFunction(name: "scanBlockReduceKernel"),
              let scanFn = library.makeFunction(name: "scanBlockScanKernel"),
              let markScanFn = library.makeFunction(name: "visibilityBlockScanKernel"),
              let singleFn = library.makeFunction(name: "singleBlockScanKernel")
        else {
            throw RendererError.failedToCreatePipeline("Visibility compaction kernels not found")
        }

        let scatterConstants = MTLFunctionConstantValues()
        var use16BitDepthKey = depthKeyPrecision == .bits16
        scatterConstants.setConstantValue(&use16BitDepthKey, type: .bool, index: 2)
        guard let scatterFn = try? library.makeFunction(name: "visibilityScatterCompactKernel", constantValues: scatterConstants) else {
            throw RendererError.failedToCreatePipeline("visibilityScatterCompactKernel not found")
        }

        self.markReducePipeline = try device.makeComputePipelineState(function: markReduceFn)
        self.reducePipeline = try device.makeComputePipelineState(function: reduceFn)
        self.scanPipeline = try device.makeComputePipelineState(function: scanFn)
        self.markScanPipeline = try device.makeComputePipelineState(function: markScanFn)
        self.singleBlockScanPipeline = try device.makeComputePipelineState(function: singleFn)
        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        nTouchedTiles: MTLBuffer,
        preDepthKeys: MTLBuffer,
        prefixSumsScratch: MTLBuffer,
        prefixOffsetsOut: MTLBuffer,
        depthKeysOut: MTLBuffer,
        primitiveIndicesOut: MTLBuffer,
        visibleCountOut: MTLBuffer,
        maxOutCount: Int,
        maxCountCapacity: Int
    ) {
        guard gaussianCount > 0 else { return }

        let count = UInt32(gaussianCount)
        let maxOut = UInt32(maxOutCount)

        let stride = MemoryLayout<UInt32>.stride
        let maxBlocks = (maxCountCapacity + blockSize - 1) / blockSize
        let maxLevel2Blocks = (maxBlocks + blockSize - 1) / blockSize
        let maxLevel3Blocks = (maxLevel2Blocks + blockSize - 1) / blockSize
        let requiredScratchCount = maxBlocks + 1 + maxLevel2Blocks + 1 + maxLevel3Blocks + 1
        let availableScratchCount = prefixSumsScratch.length / stride
        if availableScratchCount < requiredScratchCount {
            return
        }

        let level2OffsetBytes = (maxBlocks + 1) * stride
        let level3OffsetBytes = (maxBlocks + 1 + maxLevel2Blocks + 1) * stride

        let level1Blocks = (gaussianCount + blockSize - 1) / blockSize
        let level2Blocks = (level1Blocks + blockSize - 1) / blockSize
        let level3Blocks = (level2Blocks + blockSize - 1) / blockSize

        let tg = MTLSize(width: blockSize, height: 1, depth: 1)

        // Level 1 reduce (marks from nTouchedTiles)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "VisCompact_Level1Reduce"
            encoder.setComputePipelineState(markReducePipeline)
            encoder.setBuffer(nTouchedTiles, offset: 0, index: 0)
            encoder.setBuffer(prefixSumsScratch, offset: 0, index: 1)
            var c = count
            encoder.setBytes(&c, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreads(MTLSize(width: gaussianCount, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Level 2 reduce (reduce level1 block sums)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "VisCompact_Level2Reduce"
            encoder.setComputePipelineState(reducePipeline)
            encoder.setBuffer(prefixSumsScratch, offset: 0, index: 0)
            encoder.setBuffer(prefixSumsScratch, offset: level2OffsetBytes, index: 1)
            var c = UInt32(level1Blocks)
            encoder.setBytes(&c, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreads(MTLSize(width: level1Blocks, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Level 3 reduce (reduce level2 block sums)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "VisCompact_Level3Reduce"
            encoder.setComputePipelineState(reducePipeline)
            encoder.setBuffer(prefixSumsScratch, offset: level2OffsetBytes, index: 0)
            encoder.setBuffer(prefixSumsScratch, offset: level3OffsetBytes, index: 1)
            var c = UInt32(level2Blocks)
            encoder.setBytes(&c, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreads(MTLSize(width: level2Blocks, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Scan level 3 block sums in a single block
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "VisCompact_Level3Scan"
            encoder.setComputePipelineState(singleBlockScanPipeline)
            encoder.setBuffer(prefixSumsScratch, offset: level3OffsetBytes, index: 0)
            var bc = UInt32(level3Blocks)
            encoder.setBytes(&bc, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Scan level 2 block sums using scanned level 3 as block offsets
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "VisCompact_Level2Scan"
            encoder.setComputePipelineState(scanPipeline)
            encoder.setBuffer(prefixSumsScratch, offset: level2OffsetBytes, index: 0)
            encoder.setBuffer(prefixSumsScratch, offset: level2OffsetBytes, index: 1)
            encoder.setBuffer(prefixSumsScratch, offset: level3OffsetBytes, index: 2)
            var c = UInt32(level2Blocks)
            encoder.setBytes(&c, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.dispatchThreads(MTLSize(width: level2Blocks, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Scan level 1 block sums using scanned level 2 as block offsets
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "VisCompact_Level1Scan"
            encoder.setComputePipelineState(scanPipeline)
            encoder.setBuffer(prefixSumsScratch, offset: 0, index: 0)
            encoder.setBuffer(prefixSumsScratch, offset: 0, index: 1)
            encoder.setBuffer(prefixSumsScratch, offset: level2OffsetBytes, index: 2)
            var c = UInt32(level1Blocks)
            encoder.setBytes(&c, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.dispatchThreads(MTLSize(width: level1Blocks, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Scan the per-gaussian visibility marks (nTouchedTiles > 0) into `prefixOffsetsOut`
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "VisCompact_MarkScan"
            encoder.setComputePipelineState(markScanPipeline)
            encoder.setBuffer(nTouchedTiles, offset: 0, index: 0)
            encoder.setBuffer(prefixOffsetsOut, offset: 0, index: 1)
            encoder.setBuffer(prefixSumsScratch, offset: 0, index: 2)
            var c = count
            encoder.setBytes(&c, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.dispatchThreads(MTLSize(width: gaussianCount, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Scatter compaction into the depth-sort buffers and write visibleCountOut.
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "VisCompact_Scatter"
            encoder.setComputePipelineState(scatterPipeline)
            encoder.setBuffer(nTouchedTiles, offset: 0, index: 0)
            encoder.setBuffer(prefixOffsetsOut, offset: 0, index: 1)
            encoder.setBuffer(preDepthKeys, offset: 0, index: 2)
            encoder.setBuffer(depthKeysOut, offset: 0, index: 3)
            encoder.setBuffer(primitiveIndicesOut, offset: 0, index: 4)
            encoder.setBuffer(visibleCountOut, offset: 0, index: 5)
            var c = count
            var m = maxOut
            encoder.setBytes(&c, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&m, length: MemoryLayout<UInt32>.stride, index: 7)
            encoder.dispatchThreads(MTLSize(width: gaussianCount, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }
}

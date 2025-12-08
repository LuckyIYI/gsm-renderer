import Metal
import RendererTypes

/// Encoder for depth-first instance expansion using indirect dispatch
/// After depth sort, expands each gaussian into per-tile instances
final class InstanceExpansionEncoder {
    private let applyDepthOrderingPipeline: MTLComputePipelineState
    private let createInstancesPipeline: MTLComputePipelineState
    private let blockReducePipeline: MTLComputePipelineState
    private let level2ReducePipeline: MTLComputePipelineState
    private let level2ScanPipeline: MTLComputePipelineState
    private let level2ApplyPipeline: MTLComputePipelineState
    private let blockScanPipeline: MTLComputePipelineState

    let threadgroupSize: Int
    private let blockSize = 256
    let maxGaussians: Int

    init(device: MTLDevice, library: MTLLibrary, maxGaussians: Int) throws {
        guard let applyFn = library.makeFunction(name: "applyDepthOrderingKernel"),
              let createFn = library.makeFunction(name: "createInstancesKernel"),
              let reduceFn = library.makeFunction(name: "blockReduceKernel"),
              let level2ReduceFn = library.makeFunction(name: "level2ReduceKernel"),
              let level2ScanFn = library.makeFunction(name: "level2ScanKernel"),
              let level2ApplyFn = library.makeFunction(name: "level2ApplyAndScanKernel"),
              let blockScanFn = library.makeFunction(name: "blockScanKernel")
        else {
            throw RendererError.failedToCreatePipeline("Instance expansion kernels not found")
        }

        self.applyDepthOrderingPipeline = try device.makeComputePipelineState(function: applyFn)
        self.createInstancesPipeline = try device.makeComputePipelineState(function: createFn)
        self.blockReducePipeline = try device.makeComputePipelineState(function: reduceFn)
        self.level2ReducePipeline = try device.makeComputePipelineState(function: level2ReduceFn)
        self.level2ScanPipeline = try device.makeComputePipelineState(function: level2ScanFn)
        self.level2ApplyPipeline = try device.makeComputePipelineState(function: level2ApplyFn)
        self.blockScanPipeline = try device.makeComputePipelineState(function: blockScanFn)

        self.threadgroupSize = min(applyDepthOrderingPipeline.maxTotalThreadsPerThreadgroup, 256)
        self.maxGaussians = maxGaussians
    }

    /// Apply depth ordering - reorder tile counts by depth-sorted indices (indirect dispatch)
    func encodeApplyDepthOrdering(
        commandBuffer: MTLCommandBuffer,
        sortedPrimitiveIndices: MTLBuffer,
        nTouchedTiles: MTLBuffer,
        orderedTileCounts: MTLBuffer,
        header: MTLBuffer,
        dispatchArgs: MTLBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ApplyDepthOrdering"

        encoder.setComputePipelineState(applyDepthOrderingPipeline)
        encoder.setBuffer(sortedPrimitiveIndices, offset: 0, index: 0)
        encoder.setBuffer(nTouchedTiles, offset: 0, index: 1)
        encoder.setBuffer(orderedTileCounts, offset: 0, index: 2)
        encoder.setBuffer(header, offset: 0, index: 3)

        let tg = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreadgroups(
            indirectBuffer: dispatchArgs,
            indirectBufferOffset: DepthFirstDispatchSlot.applyDepthOrder.offset,
            threadsPerThreadgroup: tg
        )
        encoder.endEncoding()
    }

    /// Prefix sum on ordered tile counts to get instance offsets (indirect dispatch)
    /// Uses multi-level prefix sum to handle large numbers of blocks (>256)
    func encodePrefixSum(
        commandBuffer: MTLCommandBuffer,
        dataBuffer: MTLBuffer,
        blockSumsBuffer: MTLBuffer,
        header: MTLBuffer,
        dispatchArgs: MTLBuffer
    ) {
        let tg = MTLSize(width: blockSize, height: 1, depth: 1)

        // Calculate level 2 buffer offset (after level 1 block sums)
        // maxGaussians / 256 blocks at level 1, rounded up + 1 for safety
        let numBlocks = (maxGaussians + blockSize - 1) / blockSize
        let level2Offset = (numBlocks + 1) * MemoryLayout<UInt32>.stride

        // Step 1: Block-level reduce - computes sum for each block
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "PrefixBlockReduce"
            encoder.setComputePipelineState(blockReducePipeline)
            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBuffer(blockSumsBuffer, offset: 0, index: 1)
            encoder.setBuffer(header, offset: 0, index: 2)

            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: DepthFirstDispatchSlot.prefixSum.offset,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }

        // Step 2: Level 2 reduce - reduces every 256 block sums into super-block sums
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "PrefixLevel2Reduce"
            encoder.setComputePipelineState(level2ReducePipeline)
            encoder.setBuffer(blockSumsBuffer, offset: 0, index: 0)
            encoder.setBuffer(blockSumsBuffer, offset: level2Offset, index: 1)
            encoder.setBuffer(header, offset: 0, index: 2)

            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: DepthFirstDispatchSlot.prefixLevel2Reduce.offset,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }

        // Step 3: Level 2 scan - exclusive scan of super-block sums (single block)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "PrefixLevel2Scan"
            encoder.setComputePipelineState(level2ScanPipeline)
            encoder.setBuffer(blockSumsBuffer, offset: level2Offset, index: 0)
            encoder.setBuffer(header, offset: 0, index: 1)

            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: DepthFirstDispatchSlot.prefixLevel2Scan.offset,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }

        // Step 4: Level 2 apply + scan - apply super-block offsets to block sums and scan them
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "PrefixLevel2Apply"
            encoder.setComputePipelineState(level2ApplyPipeline)
            encoder.setBuffer(blockSumsBuffer, offset: 0, index: 0)
            encoder.setBuffer(blockSumsBuffer, offset: level2Offset, index: 1)
            encoder.setBuffer(header, offset: 0, index: 2)

            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: DepthFirstDispatchSlot.prefixLevel2Apply.offset,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }

        // Step 5: Final block scan - each block scans its elements and adds block offset
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "PrefixFinalScan"
            encoder.setComputePipelineState(blockScanPipeline)
            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBuffer(dataBuffer, offset: 0, index: 1)
            encoder.setBuffer(blockSumsBuffer, offset: 0, index: 2)
            encoder.setBuffer(header, offset: 0, index: 3)

            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: DepthFirstDispatchSlot.prefixFinalScan.offset,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }
    }

    /// Create instances - expand depth-sorted gaussians into per-tile assignments (indirect dispatch)
    func encodeCreateInstances(
        commandBuffer: MTLCommandBuffer,
        sortedPrimitiveIndices: MTLBuffer,
        instanceOffsets: MTLBuffer,
        tileBounds: MTLBuffer,
        renderData: MTLBuffer,
        instanceTileIds: MTLBuffer,
        instanceGaussianIndices: MTLBuffer,
        params: DepthFirstParamsSwift,
        header: MTLBuffer,
        dispatchArgs: MTLBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "CreateInstances"

        encoder.setComputePipelineState(createInstancesPipeline)
        encoder.setBuffer(sortedPrimitiveIndices, offset: 0, index: 0)
        encoder.setBuffer(instanceOffsets, offset: 0, index: 1)
        encoder.setBuffer(tileBounds, offset: 0, index: 2)
        encoder.setBuffer(renderData, offset: 0, index: 3)
        encoder.setBuffer(instanceTileIds, offset: 0, index: 4)
        encoder.setBuffer(instanceGaussianIndices, offset: 0, index: 5)

        var p = params
        encoder.setBytes(&p, length: MemoryLayout<DepthFirstParamsSwift>.stride, index: 6)
        encoder.setBuffer(header, offset: 0, index: 7)

        let tg = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreadgroups(
            indirectBuffer: dispatchArgs,
            indirectBufferOffset: DepthFirstDispatchSlot.createInstances.offset,
            threadsPerThreadgroup: tg
        )
        encoder.endEncoding()
    }
}

/// Swift struct matching DepthFirstParams in Metal
struct DepthFirstParamsSwift {
    var gaussianCount: UInt32
    var visibleCount: UInt32
    var tilesX: UInt32
    var tilesY: UInt32
    var tileWidth: UInt32
    var tileHeight: UInt32
    var maxAssignments: UInt32
    var padding: UInt32

    init(
        gaussianCount: Int,
        visibleCount: Int,
        tilesX: Int,
        tilesY: Int,
        tileWidth: Int,
        tileHeight: Int,
        maxAssignments: Int
    ) {
        self.gaussianCount = UInt32(gaussianCount)
        self.visibleCount = UInt32(visibleCount)
        self.tilesX = UInt32(tilesX)
        self.tilesY = UInt32(tilesY)
        self.tileWidth = UInt32(tileWidth)
        self.tileHeight = UInt32(tileHeight)
        self.maxAssignments = UInt32(maxAssignments)
        self.padding = 0
    }
}

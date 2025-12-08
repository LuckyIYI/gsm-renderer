import Metal
import RendererTypes

/// Encoder for extracting tile ranges after tile sort
/// Uses binary search to find start/end for each tile
final class TileRangeEncoder {
    private let extractRangesPipeline: MTLComputePipelineState
    private let resetStatePipeline: MTLComputePipelineState
    let threadgroupSize: Int

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let extractFn = library.makeFunction(name: "extractTileRangesKernel"),
              let resetFn = library.makeFunction(name: "resetDepthFirstStateKernel")
        else {
            throw RendererError.failedToCreatePipeline("Tile range kernels not found")
        }

        self.extractRangesPipeline = try device.makeComputePipelineState(function: extractFn)
        self.resetStatePipeline = try device.makeComputePipelineState(function: resetFn)
        self.threadgroupSize = min(extractRangesPipeline.maxTotalThreadsPerThreadgroup, 256)
    }

    /// Reset depth-first state (visible count, instance count, active tile count)
    func encodeResetState(
        commandBuffer: MTLCommandBuffer,
        header: MTLBuffer,
        activeTileCount: MTLBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ResetDepthFirstState"
        encoder.setComputePipelineState(resetStatePipeline)
        encoder.setBuffer(header, offset: 0, index: 0)
        encoder.setBuffer(activeTileCount, offset: 0, index: 1)
        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Extract tile ranges from sorted tile IDs (uses header for totalInstances)
    func encodeExtractRanges(
        commandBuffer: MTLCommandBuffer,
        sortedTileIds: MTLBuffer,
        tileHeaders: MTLBuffer,
        activeTiles: MTLBuffer,
        activeTileCount: MTLBuffer,
        header: MTLBuffer,
        tileCount: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ExtractTileRanges"

        encoder.setComputePipelineState(extractRangesPipeline)
        encoder.setBuffer(sortedTileIds, offset: 0, index: 0)
        encoder.setBuffer(tileHeaders, offset: 0, index: 1)
        encoder.setBuffer(activeTiles, offset: 0, index: 2)
        encoder.setBuffer(activeTileCount, offset: 0, index: 3)
        encoder.setBuffer(header, offset: 0, index: 4)

        var tiles = UInt32(tileCount)
        encoder.setBytes(&tiles, length: 4, index: 5)

        // Use fixed dispatch for tileCount (known at Swift side)
        let threads = MTLSize(width: tileCount, height: 1, depth: 1)
        let tg = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

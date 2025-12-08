import Metal

final class SortKeyGenEncoder {
    private let pipeline: MTLComputePipelineState
    let threadgroupSize: Int

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let function = library.makeFunction(name: "computeSortKeysKernel") else {
            throw RendererError.failedToCreatePipeline("computeSortKeysKernel not found")
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
        self.threadgroupSize = max(1, min(self.pipeline.maxTotalThreadsPerThreadgroup, 256))
    }

    /// Reads depth from packed GaussianRenderData
    func encode(
        commandBuffer: MTLCommandBuffer,
        tileIds: MTLBuffer,
        tileIndices: MTLBuffer,
        renderData: MTLBuffer,
        sortKeys: MTLBuffer,
        sortedIndices: MTLBuffer,
        header: MTLBuffer,
        dispatchArgs: MTLBuffer,
        dispatchOffset: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "SortKeys"

        encoder.setComputePipelineState(self.pipeline)
        encoder.setBuffer(tileIds, offset: 0, index: 0)
        encoder.setBuffer(tileIndices, offset: 0, index: 1)
        encoder.setBuffer(renderData, offset: 0, index: 2)
        encoder.setBuffer(sortKeys, offset: 0, index: 3)
        encoder.setBuffer(sortedIndices, offset: 0, index: 4)
        encoder.setBuffer(header, offset: 0, index: 5)

        let tg = MTLSize(width: self.threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreadgroups(indirectBuffer: dispatchArgs,
                                     indirectBufferOffset: dispatchOffset,
                                     threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

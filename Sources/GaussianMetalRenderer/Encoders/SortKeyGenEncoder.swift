import Metal

final class SortKeyGenEncoder {
    private let pipeline: MTLComputePipelineState
    private let pipelineHalf: MTLComputePipelineState?
    let threadgroupSize: Int

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let function = library.makeFunction(name: "computeSortKeysKernel_float") else {
            fatalError("computeSortKeysKernel_float not found")
        }
        self.pipeline = try device.makeComputePipelineState(function: function)

        if let functionHalf = library.makeFunction(name: "computeSortKeysKernel_half") {
            self.pipelineHalf = try? device.makeComputePipelineState(function: functionHalf)
        } else {
            self.pipelineHalf = nil
        }

        self.threadgroupSize = max(1, min(pipeline.maxTotalThreadsPerThreadgroup, 256))
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        tileIds: MTLBuffer,
        tileIndices: MTLBuffer,
        depths: MTLBuffer,
        sortKeys: MTLBuffer,
        sortedIndices: MTLBuffer,
        header: MTLBuffer,
        dispatchArgs: MTLBuffer,
        dispatchOffset: Int,
        precision: Precision = .float32
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "SortKeys"

        if precision == .float16, let halfPipe = self.pipelineHalf {
            encoder.setComputePipelineState(halfPipe)
        } else {
            encoder.setComputePipelineState(self.pipeline)
        }
        encoder.setBuffer(tileIds, offset: 0, index: 0)
        encoder.setBuffer(tileIndices, offset: 0, index: 1)
        encoder.setBuffer(depths, offset: 0, index: 2)
        encoder.setBuffer(sortKeys, offset: 0, index: 3)
        encoder.setBuffer(sortedIndices, offset: 0, index: 4)
        encoder.setBuffer(header, offset: 0, index: 5)
        // Indirect dispatch: dispatchArgs prepared by DispatchEncoder with paddedCount/totalAssignments from GPU header.
        let tg = MTLSize(width: self.threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreadgroups(indirectBuffer: dispatchArgs,
                                     indirectBufferOffset: dispatchOffset,
                                     threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

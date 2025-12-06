import Metal

final class DispatchEncoder {
    private let pipeline: MTLComputePipelineState
    private let config: AssignmentDispatchConfigSwift

    init(device: MTLDevice, library: MTLLibrary, config: AssignmentDispatchConfigSwift) throws {
        guard let function = library.makeFunction(name: "prepareAssignmentDispatchKernel") else {
            fatalError("prepareAssignmentDispatchKernel not found")
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
        self.config = config
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        header: MTLBuffer,
        dispatchArgs: MTLBuffer,
        maxAssignments: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PrepareDispatch"

        encoder.setComputePipelineState(self.pipeline)
        encoder.setBuffer(header, offset: 0, index: 0)
        encoder.setBuffer(dispatchArgs, offset: 0, index: 1)

        var cfg = self.config
        cfg.maxAssignments = UInt32(maxAssignments)
        encoder.setBytes(&cfg, length: MemoryLayout<AssignmentDispatchConfigSwift>.stride, index: 2)

        let threads = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: threads)
        encoder.endEncoding()
    }
}

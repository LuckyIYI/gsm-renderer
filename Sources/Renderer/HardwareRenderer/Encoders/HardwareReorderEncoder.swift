import Metal
import RendererTypes

/// Encoder for reordering projected Gaussian data based on sorted indices.
/// Handles hardware stereo and mono rendering modes.
final class HardwareReorderEncoder {
    static let threadgroupSize: Int = 256

    private let stereoPipeline: MTLComputePipelineState
    private let monoPipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let centerFn = library.makeFunction(name: "reorderStereoProjectedKernel"),
              let monoFn = library.makeFunction(name: "reorderMonoDataKernel")
        else {
            throw RendererError.failedToCreatePipeline("Reorder kernels not found")
        }

        self.stereoPipeline = try device.makeComputePipelineState(function: centerFn)
        self.monoPipeline = try device.makeComputePipelineState(function: monoFn)
    }

    /// Reorder data for hardware stereo
    func encodeStereo(
        commandBuffer: MTLCommandBuffer,
        projected: MTLBuffer,
        sortedIndices: MTLBuffer,
        projectedSorted: MTLBuffer,
        header: MTLBuffer,
        reorderDispatch: MTLBuffer,
        backToFront: Bool
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ReorderStereo"
        encoder.setComputePipelineState(self.stereoPipeline)

        encoder.setBuffer(projected, offset: 0, index: 0)
        encoder.setBuffer(sortedIndices, offset: 0, index: 1)
        encoder.setBuffer(projectedSorted, offset: 0, index: 2)
        encoder.setBuffer(header, offset: 0, index: 3)
        var backToFrontU32: UInt32 = backToFront ? 1 : 0
        encoder.setBytes(&backToFrontU32, length: 4, index: 4)

        let tg = MTLSize(width: Self.threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreadgroups(indirectBuffer: reorderDispatch, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }

    /// Reorder data for mono rendering
    func encodeMono(
        commandBuffer: MTLCommandBuffer,
        projected: MTLBuffer,
        sortedIndices: MTLBuffer,
        projectedSorted: MTLBuffer,
        header: MTLBuffer,
        reorderDispatch: MTLBuffer,
        backToFront: Bool
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ReorderMono"
        encoder.setComputePipelineState(self.monoPipeline)

        encoder.setBuffer(projected, offset: 0, index: 0)
        encoder.setBuffer(sortedIndices, offset: 0, index: 1)
        encoder.setBuffer(projectedSorted, offset: 0, index: 2)
        encoder.setBuffer(header, offset: 0, index: 3)
        var backToFrontU32: UInt32 = backToFront ? 1 : 0
        encoder.setBytes(&backToFrontU32, length: 4, index: 4)

        let tg = MTLSize(width: Self.threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreadgroups(indirectBuffer: reorderDispatch, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

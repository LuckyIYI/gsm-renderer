import Metal
import RendererTypes

/// Encoder for reordering projected Gaussian data based on sorted indices.
/// Handles center-sort (stereo) and mono rendering modes.
final class ReorderDataEncoder {
    static let threadgroupSize: Int = 256

    private let centerSortPipeline: MTLComputePipelineState
    private let monoPipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let centerFn = library.makeFunction(name: "reorderCenterSortDataKernel"),
              let monoFn = library.makeFunction(name: "reorderMonoDataKernel") else {
            throw RendererError.failedToCreatePipeline("Reorder kernels not found")
        }

        self.centerSortPipeline = try device.makeComputePipelineState(function: centerFn)
        self.monoPipeline = try device.makeComputePipelineState(function: monoFn)
    }

    /// Reorder data for center-sort stereo (single shared sort)
    func encodeCenterSort(
        commandBuffer: MTLCommandBuffer,
        projected: MTLBuffer,
        sortedIndices: MTLBuffer,
        projectedSorted: MTLBuffer,
        header: MTLBuffer,
        reorderDispatch: MTLBuffer,
        backToFront: Bool
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "CenterReorder"
        encoder.setComputePipelineState(centerSortPipeline)

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
        encoder.setComputePipelineState(monoPipeline)

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

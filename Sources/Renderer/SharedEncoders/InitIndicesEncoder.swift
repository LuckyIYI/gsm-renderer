import Metal
import RendererTypes

/// Encoder for initializing sorted indices arrays to sequential values [0, 1, 2, ..., n-1].
/// Handles center-sort (stereo) and mono rendering modes.
final class InitIndicesEncoder {
    private let centerSortPipeline: MTLComputePipelineState
    private let monoPipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let centerFn = library.makeFunction(name: "initCenterSortIndicesKernel"),
              let monoFn = library.makeFunction(name: "initMonoIndicesKernel") else {
            throw RendererError.failedToCreatePipeline("Init indices kernels not found")
        }

        self.centerSortPipeline = try device.makeComputePipelineState(function: centerFn)
        self.monoPipeline = try device.makeComputePipelineState(function: monoFn)
    }

    /// Initialize indices for center-sort stereo (single shared sort)
    func encodeCenterSort(
        commandBuffer: MTLCommandBuffer,
        sortedIndices: MTLBuffer,
        paddedMax: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "CenterInitIndices"
        encoder.setComputePipelineState(centerSortPipeline)

        encoder.setBuffer(sortedIndices, offset: 0, index: 0)
        var maxCount = UInt32(paddedMax)
        encoder.setBytes(&maxCount, length: 4, index: 1)

        let tg = MTLSize(width: min(256, centerSortPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encoder.dispatchThreads(MTLSize(width: paddedMax, height: 1, depth: 1), threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }

    /// Initialize indices for mono rendering
    func encodeMono(
        commandBuffer: MTLCommandBuffer,
        sortedIndices: MTLBuffer,
        paddedMax: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "InitMonoIndices"
        encoder.setComputePipelineState(monoPipeline)

        encoder.setBuffer(sortedIndices, offset: 0, index: 0)
        var maxCount = UInt32(paddedMax)
        encoder.setBytes(&maxCount, length: 4, index: 1)

        let tg = MTLSize(width: min(256, monoPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encoder.dispatchThreads(MTLSize(width: paddedMax, height: 1, depth: 1), threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

import Metal
import RendererTypes

/// Encoder for unified stereo preprocess kernel.
/// Projects both eyes in one pass, outputs single StereoTiledRenderData with union bounds.
final class DepthFirstStereoUnifiedPreprocessEncoder {
    /// Map shComponents count to SH degree (0-3)
    static func shDegree(from shComponents: Int) -> UInt32 {
        switch shComponents {
        case 0, 1: 0 // DC only
        case 2 ... 4: 1 // Degree 1 (4 coeffs)
        case 5 ... 9: 2 // Degree 2 (9 coeffs)
        default: 3 // Degree 3 (16 coeffs)
        }
    }

    private var floatPipelines: [UInt32: MTLComputePipelineState] = [:]
    private var halfPipelines: [UInt32: MTLComputePipelineState] = [:]
    let threadgroupSize: Int

    init(device: MTLDevice, library: MTLLibrary) throws {
        var maxThreads = 256

        for degree: UInt32 in 0 ... 3 {
            let constants = MTLFunctionConstantValues()
            var shDegree = degree
            constants.setConstantValue(&shDegree, type: .uint, index: 0)

            if let fn = try? library.makeFunction(name: "depthFirstStereoUnifiedPreprocessKernel", constantValues: constants) {
                let pipeline = try device.makeComputePipelineState(function: fn)
                self.floatPipelines[degree] = pipeline
                maxThreads = min(maxThreads, pipeline.maxTotalThreadsPerThreadgroup)
            }
            if let fn = try? library.makeFunction(name: "depthFirstStereoUnifiedPreprocessKernelHalf", constantValues: constants) {
                let pipeline = try device.makeComputePipelineState(function: fn)
                self.halfPipelines[degree] = pipeline
                maxThreads = min(maxThreads, pipeline.maxTotalThreadsPerThreadgroup)
            }
        }

        guard !floatPipelines.isEmpty else {
            throw RendererError.failedToCreatePipeline("depthFirstStereoUnifiedPreprocessKernel not found")
        }

        self.threadgroupSize = maxThreads
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        stereoCamera: StereoCameraUniforms,
        // Unified outputs
        renderData: MTLBuffer,       // StereoTiledRenderData
        bounds: MTLBuffer,           // int4 union bounds
        preDepthKeys: MTLBuffer,
        nTouchedTiles: MTLBuffer,
        totalInstances: MTLBuffer,
        // Params
        binningParams: TileBinningParams,
        useHalfWorld: Bool,
        shDegree: UInt32
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "DepthFirstStereoUnifiedPreprocess"

        let pipelines = useHalfWorld ? halfPipelines : floatPipelines
        guard let pipeline = pipelines[shDegree] ?? pipelines[0] else { return }
        encoder.setComputePipelineState(pipeline)

        // Input buffers
        encoder.setBuffer(packedWorldBuffers.packedGaussians, offset: 0, index: 0)
        encoder.setBuffer(packedWorldBuffers.harmonics, offset: 0, index: 1)

        // Unified outputs
        encoder.setBuffer(renderData, offset: 0, index: 2)
        encoder.setBuffer(bounds, offset: 0, index: 3)
        encoder.setBuffer(preDepthKeys, offset: 0, index: 4)
        encoder.setBuffer(nTouchedTiles, offset: 0, index: 5)
        encoder.setBuffer(totalInstances, offset: 0, index: 6)

        // Uniforms
        var camera = stereoCamera
        encoder.setBytes(&camera, length: MemoryLayout<StereoCameraUniforms>.stride, index: 7)

        var params = binningParams
        encoder.setBytes(&params, length: MemoryLayout<TileBinningParams>.stride, index: 8)

        let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

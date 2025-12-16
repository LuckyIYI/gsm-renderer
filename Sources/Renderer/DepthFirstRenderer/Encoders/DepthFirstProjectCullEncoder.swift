import Metal
import RendererTypes

/// Projects Gaussians to screen space, applies culling, and emits:
/// - per-Gaussian render data
/// - tile bounds and touched-tile counts
/// - a depth-sort key for later sorting
///
/// This encoder supports both mono and stereo projection paths; the output buffer
/// types differ, but the stage is the same (project + cull).
final class DepthFirstProjectCullEncoder {
    /// Map shComponents count to SH degree (0-3)
    static func shDegree(from shComponents: Int) -> UInt32 {
        switch shComponents {
        case 0, 1: 0
        case 2 ... 4: 1
        case 5 ... 9: 2
        default: 3
        }
    }

    private var monoFloatPipelines: [UInt32: MTLComputePipelineState] = [:]
    private var monoHalfPipelines: [UInt32: MTLComputePipelineState] = [:]
    private var stereoFloatPipelines: [UInt32: MTLComputePipelineState] = [:]
    private var stereoHalfPipelines: [UInt32: MTLComputePipelineState] = [:]

    let threadgroupSize: Int

    init(device: MTLDevice, library: MTLLibrary) throws {
        var maxThreads = 256

        for degree: UInt32 in 0 ... 3 {
            let constants = MTLFunctionConstantValues()
            var shDegree = degree
            constants.setConstantValue(&shDegree, type: .uint, index: 0)

            if let fn = try? library.makeFunction(name: "depthFirstProjectCullKernel", constantValues: constants) {
                let pipeline = try device.makeComputePipelineState(function: fn)
                self.monoFloatPipelines[degree] = pipeline
                maxThreads = min(maxThreads, pipeline.maxTotalThreadsPerThreadgroup)
            }
            if let fn = try? library.makeFunction(name: "depthFirstProjectCullKernelHalf", constantValues: constants) {
                let pipeline = try device.makeComputePipelineState(function: fn)
                self.monoHalfPipelines[degree] = pipeline
                maxThreads = min(maxThreads, pipeline.maxTotalThreadsPerThreadgroup)
            }

            if let fn = try? library.makeFunction(name: "depthFirstStereoProjectCullKernel", constantValues: constants) {
                let pipeline = try device.makeComputePipelineState(function: fn)
                self.stereoFloatPipelines[degree] = pipeline
                maxThreads = min(maxThreads, pipeline.maxTotalThreadsPerThreadgroup)
            }
            if let fn = try? library.makeFunction(name: "depthFirstStereoProjectCullKernelHalf", constantValues: constants) {
                let pipeline = try device.makeComputePipelineState(function: fn)
                self.stereoHalfPipelines[degree] = pipeline
                maxThreads = min(maxThreads, pipeline.maxTotalThreadsPerThreadgroup)
            }
        }

        guard self.monoFloatPipelines[0] != nil, self.stereoFloatPipelines[0] != nil else {
            throw RendererError.failedToCreatePipeline("Depth-first project+cull kernels not found")
        }

        self.threadgroupSize = maxThreads
    }

    func encodeMono(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        cameraUniforms: CameraUniformsSwift,
        renderData: MTLBuffer,
        bounds: MTLBuffer,
        preDepthKeys: MTLBuffer,
        nTouchedTiles: MTLBuffer,
        totalInstances: MTLBuffer,
        binningParams: TileBinningParams,
        useHalfWorld: Bool,
        shDegree: UInt32
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "DepthFirst_ProjectCull_Mono"

        let pipelines = useHalfWorld ? self.monoHalfPipelines : self.monoFloatPipelines
        guard let pipeline = pipelines[shDegree] ?? pipelines[0] else {
            encoder.endEncoding()
            return
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(packedWorldBuffers.packedGaussians, offset: 0, index: 0)
        encoder.setBuffer(packedWorldBuffers.harmonics, offset: 0, index: 1)
        encoder.setBuffer(renderData, offset: 0, index: 2)
        encoder.setBuffer(bounds, offset: 0, index: 3)
        encoder.setBuffer(preDepthKeys, offset: 0, index: 4)
        encoder.setBuffer(nTouchedTiles, offset: 0, index: 5)
        encoder.setBuffer(totalInstances, offset: 0, index: 6)

        var camera = cameraUniforms
        encoder.setBytes(&camera, length: MemoryLayout<CameraUniforms>.stride, index: 7)

        var params = binningParams
        encoder.setBytes(&params, length: MemoryLayout<TileBinningParams>.stride, index: 8)

        let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }

    func encodeStereo(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        stereoCamera: StereoCameraUniforms,
        renderData: MTLBuffer,
        bounds: MTLBuffer,
        preDepthKeys: MTLBuffer,
        nTouchedTiles: MTLBuffer,
        totalInstances: MTLBuffer,
        binningParams: TileBinningParams,
        useHalfWorld: Bool,
        shDegree: UInt32
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "DepthFirst_ProjectCull_Stereo"

        let pipelines = useHalfWorld ? self.stereoHalfPipelines : self.stereoFloatPipelines
        guard let pipeline = pipelines[shDegree] ?? pipelines[0] else {
            encoder.endEncoding()
            return
        }
        encoder.setComputePipelineState(pipeline)

        encoder.setBuffer(packedWorldBuffers.packedGaussians, offset: 0, index: 0)
        encoder.setBuffer(packedWorldBuffers.harmonics, offset: 0, index: 1)
        encoder.setBuffer(renderData, offset: 0, index: 2)
        encoder.setBuffer(bounds, offset: 0, index: 3)
        encoder.setBuffer(preDepthKeys, offset: 0, index: 4)
        encoder.setBuffer(nTouchedTiles, offset: 0, index: 5)
        encoder.setBuffer(totalInstances, offset: 0, index: 6)

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

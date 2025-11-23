import Metal

final class ProjectEncoder {
    private let pipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let function = library.makeFunction(name: "projectGaussiansKernel") else {
            fatalError("projectGaussiansKernel not found")
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        worldBuffers: WorldGaussianBuffers,
        cameraUniforms: CameraUniformsSwift,
        projectionBuffers: ProjectionReadbackBuffers
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ProjectGaussiansDebug"
        
        encoder.setComputePipelineState(self.pipeline)
        encoder.setBuffer(worldBuffers.positions, offset: 0, index: 0)
        encoder.setBuffer(worldBuffers.scales, offset: 0, index: 1)
        encoder.setBuffer(worldBuffers.rotations, offset: 0, index: 2)
        encoder.setBuffer(worldBuffers.opacities, offset: 0, index: 3)
        encoder.setBuffer(worldBuffers.harmonics, offset: 0, index: 4)
        
        encoder.setBuffer(projectionBuffers.meansOut, offset: 0, index: 5)
        encoder.setBuffer(projectionBuffers.conicsOut, offset: 0, index: 6)
        encoder.setBuffer(projectionBuffers.colorsOut, offset: 0, index: 7)
        encoder.setBuffer(projectionBuffers.opacitiesOut, offset: 0, index: 8)
        encoder.setBuffer(projectionBuffers.depthsOut, offset: 0, index: 9)
        encoder.setBuffer(projectionBuffers.radiiOut, offset: 0, index: 10)
        encoder.setBuffer(projectionBuffers.maskOut, offset: 0, index: 11)
        
        var cam = cameraUniforms
        encoder.setBytes(&cam, length: MemoryLayout<CameraUniformsSwift>.stride, index: 12)
        
        let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: self.pipeline.threadExecutionWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }

    func encodeForRender(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        worldBuffers: WorldGaussianBuffers,
        cameraUniforms: CameraUniformsSwift,
        gaussianBuffers: GaussianInputBuffers
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ProjectGaussians"
        
        encoder.setComputePipelineState(self.pipeline)
        encoder.setBuffer(worldBuffers.positions, offset: 0, index: 0)
        encoder.setBuffer(worldBuffers.scales, offset: 0, index: 1)
        encoder.setBuffer(worldBuffers.rotations, offset: 0, index: 2)
        encoder.setBuffer(worldBuffers.opacities, offset: 0, index: 3)
        encoder.setBuffer(worldBuffers.harmonics, offset: 0, index: 4)
        
        encoder.setBuffer(gaussianBuffers.means, offset: 0, index: 5)
        encoder.setBuffer(gaussianBuffers.conics, offset: 0, index: 6)
        encoder.setBuffer(gaussianBuffers.colors, offset: 0, index: 7)
        encoder.setBuffer(gaussianBuffers.opacities, offset: 0, index: 8)
        encoder.setBuffer(gaussianBuffers.depths, offset: 0, index: 9)
        encoder.setBuffer(gaussianBuffers.radii, offset: 0, index: 10)
        encoder.setBuffer(gaussianBuffers.mask, offset: 0, index: 11)
        
        var cam = cameraUniforms
        encoder.setBytes(&cam, length: MemoryLayout<CameraUniformsSwift>.stride, index: 12)
        
        let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: self.pipeline.threadExecutionWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}
import Metal

final class ProjectEncoder {
    private let pipelineFloat: MTLComputePipelineState
    private let pipelineHalf: MTLComputePipelineState?
    private let pipelineHalfInput: MTLComputePipelineState?

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let functionFloat = library.makeFunction(name: "projectGaussians_float") else {
            fatalError("projectGaussians_float not found")
        }

        self.pipelineFloat = try device.makeComputePipelineState(function: functionFloat)

        if let functionHalf = library.makeFunction(name: "projectGaussians_half") {
            self.pipelineHalf = try device.makeComputePipelineState(function: functionHalf)
        } else {
            self.pipelineHalf = nil
        }

        if let functionHalfInput = library.makeFunction(name: "projectGaussians_half_input") {
            self.pipelineHalfInput = try device.makeComputePipelineState(function: functionHalfInput)
        } else {
            self.pipelineHalfInput = nil
        }
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        worldBuffers: WorldGaussianBuffers,
        cameraUniforms: CameraUniformsSwift,
        projectionBuffers: ProjectionReadbackBuffers
    ) {
        // Debug always uses Float for readback ease
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ProjectGaussiansDebug"
        
        encoder.setComputePipelineState(self.pipelineFloat)
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
        let tg = MTLSize(width: self.pipelineFloat.threadExecutionWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }

    func encodeForRender(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        worldBuffers: WorldGaussianBuffers,
        cameraUniforms: CameraUniformsSwift,
        gaussianBuffers: GaussianInputBuffers,
        precision: Precision
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ProjectGaussians"

        if precision == .float16, let pipe = self.pipelineHalf {
            encoder.setComputePipelineState(pipe)
        } else {
            encoder.setComputePipelineState(self.pipelineFloat)
        }

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
        let tg = MTLSize(width: self.pipelineFloat.threadExecutionWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }

    /// Encode projection from half-precision world buffers to half-precision gaussian buffers.
    func encodeForRenderHalf(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        worldBuffers: WorldGaussianBuffersHalf,
        cameraUniforms: CameraUniformsSwift,
        gaussianBuffers: GaussianInputBuffers  // Uses same buffer struct, but with half layout
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ProjectGaussiansHalfInput"

        if let pipe = self.pipelineHalfInput {
            encoder.setComputePipelineState(pipe)
        } else {
            // Fallback: This should not happen if Metal file is compiled correctly
            fatalError("projectGaussians_half_input pipeline not available")
        }

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
        let tg = MTLSize(width: self.pipelineHalfInput!.threadExecutionWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

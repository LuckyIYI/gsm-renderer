import Metal

/// Packed world gaussian buffers (float32) - single interleaved buffer for optimal memory access
public struct PackedWorldBuffers {
    public let packedGaussians: MTLBuffer  // PackedWorldGaussian array (48 bytes each)
    public let harmonics: MTLBuffer         // Separate buffer for variable-size SH data

    public init(packedGaussians: MTLBuffer, harmonics: MTLBuffer) {
        self.packedGaussians = packedGaussians
        self.harmonics = harmonics
    }
}

/// Packed world gaussian buffers (float16) - half the memory of PackedWorldBuffers
public struct PackedWorldBuffersHalf {
    public let packedGaussians: MTLBuffer  // PackedWorldGaussianHalf array (24 bytes each)
    public let harmonics: MTLBuffer         // Separate buffer for variable-size SH data (still float for accuracy)

    public init(packedGaussians: MTLBuffer, harmonics: MTLBuffer) {
        self.packedGaussians = packedGaussians
        self.harmonics = harmonics
    }
}

final class ProjectEncoder {
    private let pipelineFloatHalf: MTLComputePipelineState   // float input -> half output
    private let pipelineFloatFloat: MTLComputePipelineState  // float input -> float output
    private let pipelineHalfHalf: MTLComputePipelineState    // half input -> half output

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let fnFloatHalf = library.makeFunction(name: "projectGaussiansPacked_float_half") else {
            fatalError("projectGaussiansPacked_float_half not found")
        }
        guard let fnFloatFloat = library.makeFunction(name: "projectGaussiansPacked_float_float") else {
            fatalError("projectGaussiansPacked_float_float not found")
        }
        guard let fnHalfHalf = library.makeFunction(name: "projectGaussiansPacked_half_half") else {
            fatalError("projectGaussiansPacked_half_half not found")
        }

        self.pipelineFloatHalf = try device.makeComputePipelineState(function: fnFloatHalf)
        self.pipelineFloatFloat = try device.makeComputePipelineState(function: fnFloatFloat)
        self.pipelineHalfHalf = try device.makeComputePipelineState(function: fnHalfHalf)
    }

    /// Encode projection from packed world buffers (optimized memory access)
    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        cameraUniforms: CameraUniformsSwift,
        gaussianBuffers: GaussianInputBuffers,
        precision: Precision
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ProjectGaussiansPacked"

        let pipeline = precision == .float16 ? pipelineFloatHalf : pipelineFloatFloat

        encoder.setComputePipelineState(pipeline)

        // Input buffers (packed format - single coalesced read per gaussian)
        encoder.setBuffer(packedWorldBuffers.packedGaussians, offset: 0, index: 0)
        encoder.setBuffer(packedWorldBuffers.harmonics, offset: 0, index: 1)

        // Output buffers
        encoder.setBuffer(gaussianBuffers.means, offset: 0, index: 2)
        encoder.setBuffer(gaussianBuffers.conics, offset: 0, index: 3)
        encoder.setBuffer(gaussianBuffers.colors, offset: 0, index: 4)
        encoder.setBuffer(gaussianBuffers.opacities, offset: 0, index: 5)
        encoder.setBuffer(gaussianBuffers.depths, offset: 0, index: 6)
        encoder.setBuffer(gaussianBuffers.radii, offset: 0, index: 7)
        encoder.setBuffer(gaussianBuffers.mask, offset: 0, index: 8)

        var cam = cameraUniforms
        encoder.setBytes(&cam, length: MemoryLayout<CameraUniformsSwift>.stride, index: 9)

        let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }

    /// Encode projection from half-precision packed world buffers (full half pipeline)
    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffersHalf: PackedWorldBuffersHalf,
        cameraUniforms: CameraUniformsSwift,
        gaussianBuffers: GaussianInputBuffers
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ProjectGaussiansPacked_HalfHalf"

        encoder.setComputePipelineState(pipelineHalfHalf)

        // Input buffers (half-precision packed format)
        encoder.setBuffer(packedWorldBuffersHalf.packedGaussians, offset: 0, index: 0)
        encoder.setBuffer(packedWorldBuffersHalf.harmonics, offset: 0, index: 1)

        // Output buffers (half precision)
        encoder.setBuffer(gaussianBuffers.means, offset: 0, index: 2)
        encoder.setBuffer(gaussianBuffers.conics, offset: 0, index: 3)
        encoder.setBuffer(gaussianBuffers.colors, offset: 0, index: 4)
        encoder.setBuffer(gaussianBuffers.opacities, offset: 0, index: 5)
        encoder.setBuffer(gaussianBuffers.depths, offset: 0, index: 6)
        encoder.setBuffer(gaussianBuffers.radii, offset: 0, index: 7)
        encoder.setBuffer(gaussianBuffers.mask, offset: 0, index: 8)

        var cam = cameraUniforms
        encoder.setBytes(&cam, length: MemoryLayout<CameraUniformsSwift>.stride, index: 9)

        let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: pipelineHalfHalf.threadExecutionWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

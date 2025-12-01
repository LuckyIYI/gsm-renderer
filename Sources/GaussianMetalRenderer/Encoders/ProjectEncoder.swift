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
    public let harmonics: MTLBuffer         // Half-precision SH data (Float16)

    public init(packedGaussians: MTLBuffer, harmonics: MTLBuffer) {
        self.packedGaussians = packedGaussians
        self.harmonics = harmonics
    }
}

final class ProjectEncoder {
    // SH degree-specific pipelines (function constants for unrolled loops)
    // Key: SH degree (0-3)
    private var pipelinesFloatHalfBySHDegree: [UInt32: MTLComputePipelineState] = [:]
    private var pipelinesFloatFloatBySHDegree: [UInt32: MTLComputePipelineState] = [:]
    private var pipelinesHalfHalfBySHDegree: [UInt32: MTLComputePipelineState] = [:]

    /// Map shComponents count to SH degree (0-3)
    private static func shDegree(from shComponents: UInt32) -> UInt32 {
        switch shComponents {
        case 0, 1: return 0     // DC only
        case 2...4: return 1    // Degree 1 (4 coeffs)
        case 5...9: return 2    // Degree 2 (9 coeffs)
        default: return 3       // Degree 3 (16 coeffs)
        }
    }

    init(device: MTLDevice, library: MTLLibrary) throws {
        // Create SH degree-specific pipeline variants (function constants for unrolled loops)
        // This gives significant speedup by allowing the compiler to fully unroll SH loops
        for degree: UInt32 in 0...3 {
            let constantValues = MTLFunctionConstantValues()
            var shDegree = degree
            constantValues.setConstantValue(&shDegree, type: .uint, index: 0) // SH_DEGREE at index 0

            // Float input -> Half output
            if let fn = try? library.makeFunction(name: "projectGaussiansPacked_float_half", constantValues: constantValues) {
                self.pipelinesFloatHalfBySHDegree[degree] = try? device.makeComputePipelineState(function: fn)
            }

            // Float input -> Float output
            if let fn = try? library.makeFunction(name: "projectGaussiansPacked_float_float", constantValues: constantValues) {
                self.pipelinesFloatFloatBySHDegree[degree] = try? device.makeComputePipelineState(function: fn)
            }

            // Half input -> Half output
            if let fn = try? library.makeFunction(name: "projectGaussiansPacked_half_half", constantValues: constantValues) {
                self.pipelinesHalfHalfBySHDegree[degree] = try? device.makeComputePipelineState(function: fn)
            }
        }

        // Verify we have at least degree 0 pipelines
        guard pipelinesFloatHalfBySHDegree[0] != nil,
              pipelinesFloatFloatBySHDegree[0] != nil,
              pipelinesHalfHalfBySHDegree[0] != nil else {
            fatalError("Failed to create projection pipelines")
        }
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

        // Select pipeline with correct SH degree for optimized unrolled loops
        let shDegree = Self.shDegree(from: cameraUniforms.shComponents)
        let pipeline: MTLComputePipelineState
        if precision == .float16 {
            pipeline = pipelinesFloatHalfBySHDegree[shDegree] ?? pipelinesFloatHalfBySHDegree[0]!
        } else {
            pipeline = pipelinesFloatFloatBySHDegree[shDegree] ?? pipelinesFloatFloatBySHDegree[0]!
        }

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

        // Select pipeline with correct SH degree for optimized unrolled loops
        let shDegree = Self.shDegree(from: cameraUniforms.shComponents)
        let pipeline = pipelinesHalfHalfBySHDegree[shDegree] ?? pipelinesHalfHalfBySHDegree[0]!

        encoder.setComputePipelineState(pipeline)

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
        let tg = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

import Metal

/// Packed world gaussian buffers - single interleaved buffer for optimal memory access
public struct PackedWorldBuffers {
    public let packedGaussians: MTLBuffer  // PackedWorldGaussian or PackedWorldGaussianHalf array
    public let harmonics: MTLBuffer         // Separate buffer for variable-size SH data (float or half)

    public init(packedGaussians: MTLBuffer, harmonics: MTLBuffer) {
        self.packedGaussians = packedGaussians
        self.harmonics = harmonics
    }
}

/// Output buffer set for projection (packed render data + radii + mask)
public struct ProjectionOutput {
    public let renderData: MTLBuffer   // GaussianRenderData array (packed)
    public let radii: MTLBuffer        // float array (for tileBounds)
    public let mask: MTLBuffer         // uchar array (for culling)

    public init(renderData: MTLBuffer, radii: MTLBuffer, mask: MTLBuffer) {
        self.renderData = renderData
        self.radii = radii
        self.mask = mask
    }
}

final class ProjectEncoder {
    // Float world input pipelines (by SH degree)
    private var pipelinesBySHDegree: [UInt32: MTLComputePipelineState] = [:]
    // Half world input pipelines (by SH degree)
    private var halfPipelinesBySHDegree: [UInt32: MTLComputePipelineState] = [:]

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
        for degree: UInt32 in 0...3 {
            let constantValues = MTLFunctionConstantValues()
            var shDegree = degree
            constantValues.setConstantValue(&shDegree, type: .uint, index: 0)

            // Float world input
            if let fn = try? library.makeFunction(name: "projectGaussiansKernel", constantValues: constantValues) {
                self.pipelinesBySHDegree[degree] = try? device.makeComputePipelineState(function: fn)
            }

            // Half world input
            if let fn = try? library.makeFunction(name: "projectGaussiansKernelHalf", constantValues: constantValues) {
                self.halfPipelinesBySHDegree[degree] = try? device.makeComputePipelineState(function: fn)
            }
        }

        guard pipelinesBySHDegree[0] != nil else {
            fatalError("Failed to create projection pipeline")
        }
    }

    /// Encode projection - outputs to packed GaussianRenderData struct + radii + mask
    /// - useHalfWorld: If true, interpret input as PackedWorldGaussianHalf + half harmonics
    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        cameraUniforms: CameraUniformsSwift,
        output: ProjectionOutput,
        useHalfWorld: Bool = false
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ProjectGaussians"

        let shDegree = Self.shDegree(from: cameraUniforms.shComponents)

        // Select pipeline based on input precision (like LocalSort)
        let pipeline: MTLComputePipelineState
        if useHalfWorld {
            pipeline = halfPipelinesBySHDegree[shDegree] ?? halfPipelinesBySHDegree[0] ?? pipelinesBySHDegree[0]!
        } else {
            pipeline = pipelinesBySHDegree[shDegree] ?? pipelinesBySHDegree[0]!
        }

        encoder.setComputePipelineState(pipeline)

        // Input buffers
        encoder.setBuffer(packedWorldBuffers.packedGaussians, offset: 0, index: 0)
        encoder.setBuffer(packedWorldBuffers.harmonics, offset: 0, index: 1)

        // Output buffers
        encoder.setBuffer(output.renderData, offset: 0, index: 2)
        encoder.setBuffer(output.radii, offset: 0, index: 3)
        encoder.setBuffer(output.mask, offset: 0, index: 4)

        var cam = cameraUniforms
        encoder.setBytes(&cam, length: MemoryLayout<CameraUniformsSwift>.stride, index: 5)

        let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

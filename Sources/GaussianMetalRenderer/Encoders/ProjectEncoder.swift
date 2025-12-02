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
    // Pipelines without cluster culling (USE_CLUSTER_CULL=false)
    private var floatPipelines: [UInt32: MTLComputePipelineState] = [:]
    private var halfPipelines: [UInt32: MTLComputePipelineState] = [:]

    // Pipelines with cluster culling (USE_CLUSTER_CULL=true)
    private var floatCullPipelines: [UInt32: MTLComputePipelineState] = [:]
    private var halfCullPipelines: [UInt32: MTLComputePipelineState] = [:]

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
        // Create pipelines for each SH degree, with and without cluster culling
        for degree: UInt32 in 0...3 {
            // Without cluster culling (USE_CLUSTER_CULL=false)
            let noCullConstants = MTLFunctionConstantValues()
            var shDegree = degree
            var useClusterCull = false
            noCullConstants.setConstantValue(&shDegree, type: .uint, index: 0)
            noCullConstants.setConstantValue(&useClusterCull, type: .bool, index: 1)

            if let fn = try? library.makeFunction(name: "projectGaussiansKernel", constantValues: noCullConstants) {
                self.floatPipelines[degree] = try? device.makeComputePipelineState(function: fn)
            }
            if let fn = try? library.makeFunction(name: "projectGaussiansKernelHalf", constantValues: noCullConstants) {
                self.halfPipelines[degree] = try? device.makeComputePipelineState(function: fn)
            }

            // With cluster culling (USE_CLUSTER_CULL=true)
            let cullConstants = MTLFunctionConstantValues()
            var useCull = true
            cullConstants.setConstantValue(&shDegree, type: .uint, index: 0)
            cullConstants.setConstantValue(&useCull, type: .bool, index: 1)

            if let fn = try? library.makeFunction(name: "projectGaussiansKernel", constantValues: cullConstants) {
                self.floatCullPipelines[degree] = try? device.makeComputePipelineState(function: fn)
            }
            if let fn = try? library.makeFunction(name: "projectGaussiansKernelHalf", constantValues: cullConstants) {
                self.halfCullPipelines[degree] = try? device.makeComputePipelineState(function: fn)
            }
        }

        guard floatPipelines[0] != nil else {
            fatalError("Failed to create projection pipeline")
        }
    }

    /// Encode projection - outputs to packed GaussianRenderData struct + radii + mask
    /// - Parameters:
    ///   - clusterVisibility: Optional - when provided, enables skip-based cluster culling
    ///   - clusterSize: Size of clusters (only used when clusterVisibility is provided)
    ///   - useHalfWorld: If true, interpret input as PackedWorldGaussianHalf + half harmonics
    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        cameraUniforms: CameraUniformsSwift,
        output: ProjectionOutput,
        clusterVisibility: MTLBuffer? = nil,
        clusterSize: UInt32 = 1024,
        useHalfWorld: Bool = false
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = clusterVisibility != nil ? "ProjectGaussians_Cull" : "ProjectGaussians"

        let shDegree = Self.shDegree(from: cameraUniforms.shComponents)
        let useClusterCull = clusterVisibility != nil

        // Select pipeline based on precision and culling mode
        let pipeline: MTLComputePipelineState
        if useClusterCull {
            if useHalfWorld {
                pipeline = halfCullPipelines[shDegree] ?? halfCullPipelines[0] ?? floatCullPipelines[0]!
            } else {
                pipeline = floatCullPipelines[shDegree] ?? floatCullPipelines[0]!
            }
        } else {
            if useHalfWorld {
                pipeline = halfPipelines[shDegree] ?? halfPipelines[0] ?? floatPipelines[0]!
            } else {
                pipeline = floatPipelines[shDegree] ?? floatPipelines[0]!
            }
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

        // Cluster visibility (only bound when USE_CLUSTER_CULL=true pipeline is used)
        if let visibility = clusterVisibility {
            encoder.setBuffer(visibility, offset: 0, index: 6)
            var size = clusterSize
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.stride, index: 7)
        }

        let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

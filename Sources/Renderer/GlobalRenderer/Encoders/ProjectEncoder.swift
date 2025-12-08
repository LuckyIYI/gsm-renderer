import Metal

struct PackedWorldBuffers {
    let packedGaussians: MTLBuffer // PackedWorldGaussian or PackedWorldGaussianHalf array
    let harmonics: MTLBuffer // Separate buffer for variable-size SH data (float or half)

    init(packedGaussians: MTLBuffer, harmonics: MTLBuffer) {
        self.packedGaussians = packedGaussians
        self.harmonics = harmonics
    }
}

struct ProjectionOutput {
    let renderData: MTLBuffer
    let bounds: MTLBuffer
    let mask: MTLBuffer

    init(renderData: MTLBuffer, bounds: MTLBuffer, mask: MTLBuffer) {
        self.renderData = renderData
        self.bounds = bounds
        self.mask = mask
    }
}

final class ProjectEncoder {
    // Pipelines without cluster culling (USE_CLUSTER_CULL=false)
    private var floatPipelines: [UInt32: MTLComputePipelineState] = [:]
    private var halfPipelines: [UInt32: MTLComputePipelineState] = [:]

    private static func shDegree(from shComponents: UInt32) -> UInt32 {
        switch shComponents {
        case 0, 1: 0 // DC only
        case 2 ... 4: 1 // Degree 1 (4 coeffs)
        case 5 ... 9: 2 // Degree 2 (9 coeffs)
        default: 3 // Degree 3 (16 coeffs)
        }
    }

    init(device: MTLDevice, library: MTLLibrary) throws {
        for degree: UInt32 in 0 ... 3 {
            // Without cluster culling (USE_CLUSTER_CULL=false)
            let noCullConstants = MTLFunctionConstantValues()
            var shDegree = degree
            noCullConstants.setConstantValue(&shDegree, type: .uint, index: 0)

            if let fn = try? library.makeFunction(name: "projectGaussiansFusedKernel", constantValues: noCullConstants) {
                self.floatPipelines[degree] = try? device.makeComputePipelineState(function: fn)
            }
            if let fn = try? library.makeFunction(name: "projectGaussiansFusedKernelHalf", constantValues: noCullConstants) {
                self.halfPipelines[degree] = try? device.makeComputePipelineState(function: fn)
            }
        }

        guard self.floatPipelines[0] != nil else {
            throw RendererError.failedToCreatePipeline("Failed to create fused projection pipeline")
        }
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        cameraUniforms: CameraUniformsSwift,
        output: ProjectionOutput,
        params: TileBinningParams,
        clusterVisibility: MTLBuffer? = nil,
        clusterSize: UInt32 = 1024,
        useHalfWorld: Bool = false
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = clusterVisibility != nil ? "ProjectGaussiansFused_Cull" : "ProjectGaussiansFused"

        let shDegree = Self.shDegree(from: cameraUniforms.shComponents)

        // Select pipeline based on precision and culling mode
        // Safe to force unwrap - we validated floatPipelines[0] exists in init
        let pipeline: MTLComputePipelineState =
            if useHalfWorld {
                self.halfPipelines[shDegree] ?? self.halfPipelines[0] ?? self.floatPipelines[0]!
            } else {
                self.floatPipelines[shDegree] ?? self.floatPipelines[0]!
            }

        encoder.setComputePipelineState(pipeline)

        // Input buffers
        encoder.setBuffer(packedWorldBuffers.packedGaussians, offset: 0, index: 0)
        encoder.setBuffer(packedWorldBuffers.harmonics, offset: 0, index: 1)

        // Output buffers
        encoder.setBuffer(output.renderData, offset: 0, index: 2)
        encoder.setBuffer(output.bounds, offset: 0, index: 3) // int4 tile bounds
        encoder.setBuffer(output.mask, offset: 0, index: 4)

        var cam = cameraUniforms
        encoder.setBytes(&cam, length: MemoryLayout<CameraUniformsSwift>.stride, index: 5)

        // Tile parameters for fused tile bounds computation
        var binParams = params
        encoder.setBytes(&binParams, length: MemoryLayout<TileBinningParams>.stride, index: 6)

        // Cluster visibility (only bound when USE_CLUSTER_CULL=true pipeline is used)
        if let visibility = clusterVisibility {
            encoder.setBuffer(visibility, offset: 0, index: 7)
            var size = clusterSize
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.stride, index: 8)
        }

        let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

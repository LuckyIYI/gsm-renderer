import Metal

/// Encodes projection stage: ProjectStore only (sparse mode - no compaction)
/// Visibility is encoded in tile bounds: minTile == maxTile == 0 means invisible
public final class LocalProjectEncoder {
    // Projection pipelines by (shDegree, useClusterCull)
    private var projectStoreFloatPipelines: [UInt64: MTLComputePipelineState] = [:]
    private var projectStoreHalfPipelines: [UInt64: MTLComputePipelineState] = [:]

    public init(LocalLibrary: MTLLibrary, mainLibrary: MTLLibrary, device: MTLDevice) throws {
        // Create projection pipeline variants: SH degree (0-3) x cluster cull (true/false)
        for degree: UInt32 in 0 ... 3 {
            for useClusterCull in [false, true] {
                let constantValues = MTLFunctionConstantValues()
                var shDegree = degree
                var cullEnabled = useClusterCull
                constantValues.setConstantValue(&shDegree, type: .uint, index: 0)
                constantValues.setConstantValue(&cullEnabled, type: .bool, index: 1)

                let key = Self.pipelineKey(shDegree: degree, useClusterCull: useClusterCull)

                if let fn = try? LocalLibrary.makeFunction(name: "LocalProjectStoreFloat", constantValues: constantValues) {
                    self.projectStoreFloatPipelines[key] = try? device.makeComputePipelineState(function: fn)
                }
                if let fn = try? LocalLibrary.makeFunction(name: "LocalProjectStoreHalfHalfSh", constantValues: constantValues) {
                    self.projectStoreHalfPipelines[key] = try? device.makeComputePipelineState(function: fn)
                }
            }
        }
    }

    private static func pipelineKey(shDegree: UInt32, useClusterCull: Bool) -> UInt64 {
        UInt64(shDegree) | (useClusterCull ? 0x100 : 0)
    }

    public static func shDegree(from shComponents: UInt32) -> UInt32 {
        switch shComponents {
        case 0, 1: 0
        case 2 ... 4: 1
        case 5 ... 9: 2
        default: 3
        }
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        worldGaussians: MTLBuffer,
        harmonics: MTLBuffer,
        camera: CameraUniformsSwift,
        params: TileBinningParams,
        gaussianCount: Int,
        // Output
        projectionBuffer: MTLBuffer,
        compactedHeader: MTLBuffer,
        // Options
        useHalfWorld: Bool,
        clusterVisibility: MTLBuffer?,
        clusterSize: UInt32
    ) {
        var cameraUniforms = camera
        var binParams = params

        let shDegree = Self.shDegree(from: camera.shComponents)
        let useClusterCull = clusterVisibility != nil
        let key = Self.pipelineKey(shDegree: shDegree, useClusterCull: useClusterCull)

        // === PROJECT + STORE ===
        // Visibility is encoded in tile bounds: minTile == maxTile == 0 means invisible
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_ProjectStore"
            let pipeline = useHalfWorld
                ? self.projectStoreHalfPipelines[key]!
                : self.projectStoreFloatPipelines[key]!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(worldGaussians, offset: 0, index: 0)
            encoder.setBuffer(harmonics, offset: 0, index: 1)
            encoder.setBuffer(projectionBuffer, offset: 0, index: 2)
            encoder.setBytes(&cameraUniforms, length: MemoryLayout<CameraUniformsSwift>.stride, index: 4)
            encoder.setBytes(&binParams, length: MemoryLayout<TileBinningParams>.stride, index: 5)
            if let clusterVis = clusterVisibility {
                encoder.setBuffer(clusterVis, offset: 0, index: 6)
                var cs = clusterSize
                encoder.setBytes(&cs, length: 4, index: 7)
            }
            let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
            let tg = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }
}

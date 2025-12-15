import Metal
import RendererTypes
import simd

/// Encoder for mono (single-eye) projection.
/// Projects Gaussians for single-texture rendering.
final class MonoProjectionEncoder {
    private var floatPipelines: [UInt32: MTLComputePipelineState] = [:]
    private var halfPipelines: [UInt32: MTLComputePipelineState] = [:]

    init(device: MTLDevice, library: MTLLibrary) throws {
        for degree: UInt32 in 0...3 {
            let constantValues = MTLFunctionConstantValues()
            var shDegreeValue = degree
            constantValues.setConstantValue(&shDegreeValue, type: .uint, index: 0)

            if let fn = try? library.makeFunction(name: "monoProjectKernel", constantValues: constantValues) {
                floatPipelines[degree] = try? device.makeComputePipelineState(function: fn)
            }
            if let fn = try? library.makeFunction(name: "monoProjectKernelHalf", constantValues: constantValues) {
                halfPipelines[degree] = try? device.makeComputePipelineState(function: fn)
            }
        }

        guard floatPipelines[0] != nil else {
            throw RendererError.failedToCreatePipeline("Failed to create mono projection pipeline")
        }
    }

    /// Convert shComponents count to shDegree (0-3)
    private static func shDegree(from shComponents: Int) -> UInt32 {
        switch shComponents {
        case 0, 1: return 0
        case 2...4: return 1
        case 5...9: return 2
        default: return 3
        }
    }

    /// Encode mono projection
    func encode(
        commandBuffer: MTLCommandBuffer,
        input: GaussianInput,
        projected: MTLBuffer,
        preDepthKeys: MTLBuffer,
        visibilityMarks: MTLBuffer,
        camera: CameraParams,
        width: Int,
        height: Int,
        inputIsSRGB: Bool,
        precision: RenderPrecision
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "MonoProject"

        let shDegree = Self.shDegree(from: input.shComponents)
        let pipeline: MTLComputePipelineState
        if precision == .float32 {
            pipeline = floatPipelines[shDegree] ?? floatPipelines[0]!
        } else {
            pipeline = halfPipelines[shDegree] ?? halfPipelines[0] ?? floatPipelines[0]!
        }
        encoder.setComputePipelineState(pipeline)

        guard input.gaussians.length > 0 else {
            encoder.endEncoding()
            return
        }

        var params = MonoProjectionParams(
            sceneTransform: matrix_identity_float4x4,
            viewMatrix: camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            cameraPosX: camera.position.x,
            cameraPosY: camera.position.y,
            cameraPosZ: camera.position.z,
            _pad0: 0,
            width: Float(width),
            height: Float(height),
            nearPlane: camera.near,
            farPlane: camera.far,
            gaussianCount: UInt32(input.gaussianCount),
            shComponents: UInt32(input.shComponents),
            totalInkThreshold: 2.0,
            inputIsSRGB: inputIsSRGB ? 1.0 : 0.0
        )


        encoder.setBuffer(input.gaussians, offset: 0, index: 0)
        encoder.setBuffer(input.harmonics, offset: 0, index: 1)
        encoder.setBuffer(projected, offset: 0, index: 2)
        encoder.setBuffer(preDepthKeys, offset: 0, index: 3)
        encoder.setBuffer(visibilityMarks, offset: 0, index: 4)
        encoder.setBytes(&params, length: MemoryLayout<MonoProjectionParams>.stride, index: 5)

        let threads = MTLSize(width: input.gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)

        encoder.endEncoding()
    }
}

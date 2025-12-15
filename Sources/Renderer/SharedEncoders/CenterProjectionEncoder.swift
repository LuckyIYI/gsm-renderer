import Metal
import RendererTypes
import simd

/// Encoder for center-sort stereo projection.
/// Projects Gaussians from a center viewpoint with data for both eyes.
final class CenterProjectionEncoder {
    private var floatPipelines: [UInt32: MTLComputePipelineState] = [:]
    private var halfPipelines: [UInt32: MTLComputePipelineState] = [:]

    init(device: MTLDevice, library: MTLLibrary) throws {
        for degree: UInt32 in 0...3 {
            let constantValues = MTLFunctionConstantValues()
            var shDegreeValue = degree
            constantValues.setConstantValue(&shDegreeValue, type: .uint, index: 0)

            if let fn = try? library.makeFunction(name: "centerSortProjectKernel", constantValues: constantValues) {
                floatPipelines[degree] = try? device.makeComputePipelineState(function: fn)
            }
            if let fn = try? library.makeFunction(name: "centerSortProjectKernelHalf", constantValues: constantValues) {
                halfPipelines[degree] = try? device.makeComputePipelineState(function: fn)
            }
        }

        guard floatPipelines[0] != nil else {
            throw RendererError.failedToCreatePipeline("Failed to create center projection pipeline")
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

    /// Encode center-sort projection
    func encode(
        commandBuffer: MTLCommandBuffer,
        input: GaussianInput,
        projected: MTLBuffer,
        preDepthKeys: MTLBuffer,
        visibilityMarks: MTLBuffer,
        configuration: StereoConfiguration,
        eyeWidth: Int,
        eyeHeight: Int,
        inputIsSRGB: Bool,
        precision: RenderPrecision,
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "CenterProjection"

        let shDegree = Self.shDegree(from: input.shComponents)
        let pipeline: MTLComputePipelineState
        if precision == .float32 {
            pipeline = floatPipelines[shDegree] ?? floatPipelines[0]!
        } else {
            pipeline = halfPipelines[shDegree] ?? halfPipelines[0] ?? floatPipelines[0]!
        }
        encoder.setComputePipelineState(pipeline)

        encoder.setBuffer(input.gaussians, offset: 0, index: 0)
        encoder.setBuffer(input.harmonics, offset: 0, index: 1)
        encoder.setBuffer(projected, offset: 0, index: 2)
        encoder.setBuffer(preDepthKeys, offset: 0, index: 3)
        encoder.setBuffer(visibilityMarks, offset: 0, index: 4)

        let leftEye = configuration.leftEye
        let rightEye = configuration.rightEye

        let centerCameraPos = (leftEye.cameraPosition + rightEye.cameraPosition) * 0.5

        var centerViewMatrix = leftEye.viewMatrix
        centerViewMatrix[3][0] = (leftEye.viewMatrix[3][0] + rightEye.viewMatrix[3][0]) * 0.5
        centerViewMatrix[3][1] = (leftEye.viewMatrix[3][1] + rightEye.viewMatrix[3][1]) * 0.5
        centerViewMatrix[3][2] = (leftEye.viewMatrix[3][2] + rightEye.viewMatrix[3][2]) * 0.5

        let centerProjectionMatrix = simd_float4x4(
            (leftEye.projectionMatrix.columns.0 + rightEye.projectionMatrix.columns.0) * 0.5,
            (leftEye.projectionMatrix.columns.1 + rightEye.projectionMatrix.columns.1) * 0.5,
            (leftEye.projectionMatrix.columns.2 + rightEye.projectionMatrix.columns.2) * 0.5,
            (leftEye.projectionMatrix.columns.3 + rightEye.projectionMatrix.columns.3) * 0.5
        )

        var params = CenterSortProjectionParams(
            sceneTransform: configuration.sceneTransform,
            centerViewMatrix: centerViewMatrix,
            centerProjectionMatrix: centerProjectionMatrix,
            leftViewMatrix: leftEye.viewMatrix,
            leftProjectionMatrix: leftEye.projectionMatrix,
            rightViewMatrix: rightEye.viewMatrix,
            rightProjectionMatrix: rightEye.projectionMatrix,
            centerCameraPosX: centerCameraPos.x,
            centerCameraPosY: centerCameraPos.y,
            centerCameraPosZ: centerCameraPos.z,
            _padCenter0: 0,
            leftCameraPosX: leftEye.cameraPosition.x,
            leftCameraPosY: leftEye.cameraPosition.y,
            leftCameraPosZ: leftEye.cameraPosition.z,
            _padLeft0: 0,
            rightCameraPosX: rightEye.cameraPosition.x,
            rightCameraPosY: rightEye.cameraPosition.y,
            rightCameraPosZ: rightEye.cameraPosition.z,
            _padRight0: 0,
            width: Float(eyeWidth),
            height: Float(eyeHeight),
            nearPlane: leftEye.near,
            farPlane: leftEye.far,
            gaussianCount: UInt32(input.gaussianCount),
            shComponents: UInt32(input.shComponents),
            totalInkThreshold: 2.0,
            inputIsSRGB: inputIsSRGB ? 1.0 : 0.0
        )
        encoder.setBytes(&params, length: MemoryLayout<CenterSortProjectionParams>.stride, index: 5)

        let threads = MTLSize(width: input.gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

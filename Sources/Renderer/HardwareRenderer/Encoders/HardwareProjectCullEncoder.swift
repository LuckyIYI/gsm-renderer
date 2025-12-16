import Metal
import RendererTypes
import simd

/// Project+cull stage for HardwareRenderer.
/// Provides mono and stereo variants with identical call-site ergonomics.
final class HardwareProjectCullEncoder {
    private var monoFloatPipelines: [UInt32: MTLComputePipelineState] = [:]
    private var monoHalfPipelines: [UInt32: MTLComputePipelineState] = [:]
    private var stereoFloatPipelines: [UInt32: MTLComputePipelineState] = [:]
    private var stereoHalfPipelines: [UInt32: MTLComputePipelineState] = [:]

    init(device: MTLDevice, library: MTLLibrary) throws {
        for degree: UInt32 in 0...3 {
            let constantValues = MTLFunctionConstantValues()
            var shDegreeValue = degree
            constantValues.setConstantValue(&shDegreeValue, type: .uint, index: 0)

            if let fn = try? library.makeFunction(name: "monoProjectCullKernel", constantValues: constantValues) {
                monoFloatPipelines[degree] = try? device.makeComputePipelineState(function: fn)
            }
            if let fn = try? library.makeFunction(name: "monoProjectCullKernelHalf", constantValues: constantValues) {
                monoHalfPipelines[degree] = try? device.makeComputePipelineState(function: fn)
            }

            if let fn = try? library.makeFunction(name: "stereoProjectCullKernel", constantValues: constantValues) {
                stereoFloatPipelines[degree] = try? device.makeComputePipelineState(function: fn)
            }
            if let fn = try? library.makeFunction(name: "stereoProjectCullKernelHalf", constantValues: constantValues) {
                stereoHalfPipelines[degree] = try? device.makeComputePipelineState(function: fn)
            }
        }

        guard monoFloatPipelines[0] != nil, stereoFloatPipelines[0] != nil else {
            throw RendererError.failedToCreatePipeline("Hardware project+cull pipelines not found")
        }
    }

    private static func shDegree(from shComponents: Int) -> UInt32 {
        switch shComponents {
        case 0, 1: return 0
        case 2...4: return 1
        case 5...9: return 2
        default: return 3
        }
    }

    func encodeMono(
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
        encoder.label = "Hardware_ProjectCull_Mono"

        let shDegree = Self.shDegree(from: input.shComponents)
        let pipeline: MTLComputePipelineState
        if precision == .float32 {
            pipeline = monoFloatPipelines[shDegree] ?? monoFloatPipelines[0]!
        } else {
            pipeline = monoHalfPipelines[shDegree] ?? monoHalfPipelines[0] ?? monoFloatPipelines[0]!
        }
        encoder.setComputePipelineState(pipeline)

        var params = HardwareMonoProjectionParams(
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
        encoder.setBytes(&params, length: MemoryLayout<HardwareMonoProjectionParams>.stride, index: 5)

        let threads = MTLSize(width: input.gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }

    func encodeStereo(
        commandBuffer: MTLCommandBuffer,
        input: GaussianInput,
        projected: MTLBuffer,
        preDepthKeys: MTLBuffer,
        visibilityMarks: MTLBuffer,
        configuration: StereoConfiguration,
        eyeWidth: Int,
        eyeHeight: Int,
        inputIsSRGB: Bool,
        precision: RenderPrecision
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Hardware_ProjectCull_Stereo"

        let shDegree = Self.shDegree(from: input.shComponents)
        let pipeline: MTLComputePipelineState
        if precision == .float32 {
            pipeline = stereoFloatPipelines[shDegree] ?? stereoFloatPipelines[0]!
        } else {
            pipeline = stereoHalfPipelines[shDegree] ?? stereoHalfPipelines[0] ?? stereoFloatPipelines[0]!
        }
        encoder.setComputePipelineState(pipeline)

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

        var params = HardwareStereoProjectionParams(
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

        encoder.setBuffer(input.gaussians, offset: 0, index: 0)
        encoder.setBuffer(input.harmonics, offset: 0, index: 1)
        encoder.setBuffer(projected, offset: 0, index: 2)
        encoder.setBuffer(preDepthKeys, offset: 0, index: 3)
        encoder.setBuffer(visibilityMarks, offset: 0, index: 4)
        encoder.setBytes(&params, length: MemoryLayout<HardwareStereoProjectionParams>.stride, index: 5)

        let threads = MTLSize(width: input.gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}


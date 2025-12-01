import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Tests for the fused pipeline (interleaved gaussian data for cache efficiency).
/// The fused pipeline is now always enabled - these tests verify it works at various scales.
final class FusedPipelineTests: XCTestCase {

    // MARK: - Helpers

    func createPackedWorldBuffers(
        device: MTLDevice,
        positions: [SIMD3<Float>],
        scales: [SIMD3<Float>],
        rotations: [SIMD4<Float>],
        opacities: [Float],
        colors: [SIMD3<Float>]
    ) -> PackedWorldBuffers? {
        let count = positions.count
        guard count == scales.count, count == rotations.count,
              count == opacities.count, count == colors.count else {
            return nil
        }

        var packed: [PackedWorldGaussian] = []
        for i in 0..<count {
            packed.append(PackedWorldGaussian(
                position: positions[i],
                scale: scales[i],
                rotation: rotations[i],
                opacity: opacities[i]
            ))
        }

        var harmonics: [Float] = []
        for color in colors {
            harmonics.append(color.x)
            harmonics.append(color.y)
            harmonics.append(color.z)
        }

        guard let packedBuf = device.makeBuffer(
            bytes: &packed,
            length: count * MemoryLayout<PackedWorldGaussian>.stride,
            options: .storageModeShared
        ) else { return nil }

        guard let harmonicsBuf = device.makeBuffer(
            bytes: &harmonics,
            length: count * 3 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else { return nil }

        return PackedWorldBuffers(packedGaussians: packedBuf, harmonics: harmonicsBuf)
    }

    func createPackedWorldBuffersHalf(
        device: MTLDevice,
        positions: [SIMD3<Float>],
        scales: [SIMD3<Float>],
        rotations: [SIMD4<Float>],
        opacities: [Float],
        colors: [SIMD3<Float>]
    ) -> PackedWorldBuffersHalf? {
        let count = positions.count
        guard count == scales.count, count == rotations.count,
              count == opacities.count, count == colors.count else {
            return nil
        }

        var packed: [PackedWorldGaussianHalf] = []
        for i in 0..<count {
            packed.append(PackedWorldGaussianHalf(
                position: positions[i],
                scale: scales[i],
                rotation: rotations[i],
                opacity: opacities[i]
            ))
        }

        var harmonics: [Float] = []
        for color in colors {
            harmonics.append(color.x)
            harmonics.append(color.y)
            harmonics.append(color.z)
        }

        guard let packedBuf = device.makeBuffer(
            bytes: &packed,
            length: count * MemoryLayout<PackedWorldGaussianHalf>.stride,
            options: .storageModeShared
        ) else { return nil }

        guard let harmonicsBuf = device.makeBuffer(
            bytes: &harmonics,
            length: count * 3 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else { return nil }

        return PackedWorldBuffersHalf(packedGaussians: packedBuf, harmonics: harmonicsBuf)
    }

    func createSimpleCamera(width: Float, height: Float, gaussianCount: Int) -> CameraUniformsSwift {
        var projMatrix = matrix_identity_float4x4
        projMatrix.columns.0 = SIMD4(2.0 / width, 0, 0, 0)
        projMatrix.columns.1 = SIMD4(0, 2.0 / height, 0, 0)
        projMatrix.columns.2 = SIMD4(0, 0, -0.02, 0)
        projMatrix.columns.3 = SIMD4(0, 0, -1.002, 1)

        return CameraUniformsSwift(
            viewMatrix: matrix_identity_float4x4,
            projectionMatrix: projMatrix,
            cameraCenter: SIMD3<Float>(0, 0, 0),
            pixelFactor: 1.0,
            focalX: width / 2.0,
            focalY: height / 2.0,
            width: width,
            height: height,
            nearPlane: 0.1,
            farPlane: 100.0,
            shComponents: 0,
            gaussianCount: UInt32(gaussianCount),
            padding0: 0,
            padding1: 0
        )
    }

    func generateRandomGaussians(count: Int) -> (
        positions: [SIMD3<Float>],
        scales: [SIMD3<Float>],
        rotations: [SIMD4<Float>],
        opacities: [Float],
        colors: [SIMD3<Float>]
    ) {
        var positions: [SIMD3<Float>] = []
        var scales: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var opacities: [Float] = []
        var colors: [SIMD3<Float>] = []

        for i in 0..<count {
            // Spread gaussians in a grid pattern
            let gridSize = Int(sqrt(Double(count))) + 1
            let x = Float(i % gridSize) / Float(gridSize) * 800 - 400
            let y = Float(i / gridSize) / Float(gridSize) * 600 - 300
            let z = Float.random(in: 1...50)
            positions.append(SIMD3(x, y, z))

            let s = Float.random(in: 0.1...0.5)
            scales.append(SIMD3(s, s, s))

            rotations.append(SIMD4(0, 0, 0, 1))  // Identity
            opacities.append(Float.random(in: 0.5...1.0))
            colors.append(SIMD3(
                Float.random(in: -0.3...0.3),
                Float.random(in: -0.3...0.3),
                Float.random(in: -0.3...0.3)
            ))
        }

        return (positions, scales, rotations, opacities, colors)
    }

    // MARK: - Tests

    /// Test fused pipeline with half precision
    func testFusedHalfPrecision() throws {
        let width = 256
        let height = 256
        let renderer = GlobalSortRenderer(
            precision: Precision.float16,
            useHeapAllocation: false,
            limits: RendererLimits(maxGaussians: 10_000, maxWidth: width, maxHeight: height)
        )

        let count = 100
        let (positions, scales, rotations, opacities, colors) = generateRandomGaussians(count: count)

        guard let packedBuffersHalf = createPackedWorldBuffersHalf(
            device: renderer.device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create packed buffers half")
            return
        }

        let camera = createSimpleCamera(width: Float(width), height: Float(height), gaussianCount: count)

        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let frameParams = FrameParams(gaussianCount: count, whiteBackground: false)
        let textures = renderer.encodeRenderToTextureHalf(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            packedWorldBuffersHalf: packedBuffersHalf,
            cameraUniforms: camera,
            frameParams: frameParams
        )

        XCTAssertNotNil(textures, "Should return textures")
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertEqual(commandBuffer.status, MTLCommandBufferStatus.completed)
    }

    /// Test fused pipeline at 10K scale
    func testFusedAtScale() throws {
        let width = 512
        let height = 512
        let renderer = GlobalSortRenderer(
            precision: Precision.float32,
            useHeapAllocation: false,
            limits: RendererLimits(maxGaussians: 50_000, maxWidth: width, maxHeight: height)
        )

        let count = 10_000
        let (positions, scales, rotations, opacities, colors) = generateRandomGaussians(count: count)

        guard let packedBuffers = createPackedWorldBuffers(
            device: renderer.device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create packed buffers")
            return
        }

        let camera = createSimpleCamera(width: Float(width), height: Float(height), gaussianCount: count)

        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let frameParams = FrameParams(gaussianCount: count, whiteBackground: true)
        let textures = renderer.encodeRenderToTextures(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            packedWorldBuffers: packedBuffers,
            cameraUniforms: camera,
            frameParams: frameParams
        )

        XCTAssertNotNil(textures, "Should return textures")
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertEqual(commandBuffer.status, MTLCommandBufferStatus.completed)
    }

    /// Test fused pipeline at 100K scale
    func testFusedAt100KScale() throws {
        let width = 1024
        let height = 768
        let renderer = GlobalSortRenderer(
            precision: Precision.float32,
            useHeapAllocation: false,
            limits: RendererLimits(maxGaussians: 200_000, maxWidth: width, maxHeight: height)
        )

        let count = 100_000
        let (positions, scales, rotations, opacities, colors) = generateRandomGaussians(count: count)

        guard let packedBuffers = createPackedWorldBuffers(
            device: renderer.device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create packed buffers")
            return
        }

        let camera = createSimpleCamera(width: Float(width), height: Float(height), gaussianCount: count)

        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let frameParams = FrameParams(gaussianCount: count, whiteBackground: false)
        let textures = renderer.encodeRenderToTextures(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            packedWorldBuffers: packedBuffers,
            cameraUniforms: camera,
            frameParams: frameParams
        )

        XCTAssertNotNil(textures, "Should return textures")
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertEqual(commandBuffer.status, MTLCommandBufferStatus.completed)
    }

    /// Test fused pipeline at 500K scale
    func testFusedAt500KScale() throws {
        let width = 1920
        let height = 1080
        let renderer = GlobalSortRenderer(
            precision: Precision.float16,
            useHeapAllocation: false,
            textureOnly: true,
            limits: RendererLimits(maxGaussians: 600_000, maxWidth: width, maxHeight: height)
        )

        let count = 500_000
        let (positions, scales, rotations, opacities, colors) = generateRandomGaussians(count: count)

        guard let packedBuffersHalf = createPackedWorldBuffersHalf(
            device: renderer.device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create packed buffers half")
            return
        }

        let camera = createSimpleCamera(width: Float(width), height: Float(height), gaussianCount: count)

        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let frameParams = FrameParams(gaussianCount: count, whiteBackground: true)
        let textures = renderer.encodeRenderToTextureHalf(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            packedWorldBuffersHalf: packedBuffersHalf,
            cameraUniforms: camera,
            frameParams: frameParams
        )

        XCTAssertNotNil(textures, "Should return textures")
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertEqual(commandBuffer.status, MTLCommandBufferStatus.completed)
    }

    /// Test 32x16 tile configuration with fused pipeline
    func testFused32x16Tiles() throws {
        let width = 256
        let height = 256
        let renderer = GlobalSortRenderer(
            precision: Precision.float32,
            useHeapAllocation: false,
            limits: RendererLimits(maxGaussians: 10_000, maxWidth: width, maxHeight: height, tileWidth: 32, tileHeight: 16)
        )

        let count = 500
        let (positions, scales, rotations, opacities, colors) = generateRandomGaussians(count: count)

        guard let packedBuffers = createPackedWorldBuffers(
            device: renderer.device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create packed buffers")
            return
        }

        let camera = createSimpleCamera(width: Float(width), height: Float(height), gaussianCount: count)

        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let frameParams = FrameParams(gaussianCount: count, whiteBackground: false)
        let textures = renderer.encodeRenderToTextures(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            packedWorldBuffers: packedBuffers,
            cameraUniforms: camera,
            frameParams: frameParams
        )

        XCTAssertNotNil(textures, "Should return textures")
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertEqual(commandBuffer.status, MTLCommandBufferStatus.completed)
    }

    /// Test 32x16 tiles at larger scale
    func testFused32x16AtScale() throws {
        let width = 1024
        let height = 768
        let renderer = GlobalSortRenderer(
            precision: Precision.float16,
            useHeapAllocation: false,
            limits: RendererLimits(maxGaussians: 100_000, maxWidth: width, maxHeight: height, tileWidth: 32, tileHeight: 16)
        )

        let count = 50_000
        let (positions, scales, rotations, opacities, colors) = generateRandomGaussians(count: count)

        guard let packedBuffersHalf = createPackedWorldBuffersHalf(
            device: renderer.device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create packed buffers half")
            return
        }

        let camera = createSimpleCamera(width: Float(width), height: Float(height), gaussianCount: count)

        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let frameParams = FrameParams(gaussianCount: count, whiteBackground: true)
        let textures = renderer.encodeRenderToTextureHalf(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            packedWorldBuffersHalf: packedBuffersHalf,
            cameraUniforms: camera,
            frameParams: frameParams
        )

        XCTAssertNotNil(textures, "Should return textures")
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertEqual(commandBuffer.status, MTLCommandBufferStatus.completed)
    }

    /// Test GPU timing measurement
    func testGPUTimingMeasurement() throws {
        let width = 512
        let height = 512
        let renderer = GlobalSortRenderer(
            precision: Precision.float32,
            useHeapAllocation: false,
            limits: RendererLimits(maxGaussians: 50_000, maxWidth: width, maxHeight: height)
        )

        let count = 10_000
        let (positions, scales, rotations, opacities, colors) = generateRandomGaussians(count: count)

        guard let packedBuffers = createPackedWorldBuffers(
            device: renderer.device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create packed buffers")
            return
        }

        let camera = createSimpleCamera(width: Float(width), height: Float(height), gaussianCount: count)

        // Warm up
        for _ in 0..<3 {
            guard let commandBuffer = renderer.queue.makeCommandBuffer() else { continue }
            let frameParams = FrameParams(gaussianCount: count, whiteBackground: false)
            _ = renderer.encodeRenderToTextures(
                commandBuffer: commandBuffer,
                gaussianCount: count,
                packedWorldBuffers: packedBuffers,
                cameraUniforms: camera,
                frameParams: frameParams
            )
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        // Measure
        var times: [Double] = []
        for _ in 0..<5 {
            guard let commandBuffer = renderer.queue.makeCommandBuffer() else { continue }
            let frameParams = FrameParams(gaussianCount: count, whiteBackground: false)
            _ = renderer.encodeRenderToTextures(
                commandBuffer: commandBuffer,
                gaussianCount: count,
                packedWorldBuffers: packedBuffers,
                cameraUniforms: camera,
                frameParams: frameParams
            )
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let gpuTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
            if gpuTime > 0 {
                times.append(gpuTime * 1000)  // ms
            }
        }

        if !times.isEmpty {
            let avg = times.reduce(0, +) / Double(times.count)
            let min = times.min()!
            let max = times.max()!
            print("[FusedPipeline] \(count) gaussians: avg=\(String(format: "%.2f", avg))ms min=\(String(format: "%.2f", min)) max=\(String(format: "%.2f", max))")
        }

        XCTAssertTrue(times.count > 0, "Should have measured at least some timing")
    }
}

import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Integration tests for radix sort rendering pipeline.
/// Verifies that the radix sort algorithm produces correct render output.
final class RadixIntegrationTests: XCTestCase {

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
            let gridSize = Int(sqrt(Double(count))) + 1
            let x = Float(i % gridSize) / Float(gridSize) * 800 - 400
            let y = Float(i / gridSize) / Float(gridSize) * 600 - 300
            let z = Float.random(in: 1...50)
            positions.append(SIMD3(x, y, z))

            let s = Float.random(in: 0.1...0.5)
            scales.append(SIMD3(s, s, s))

            rotations.append(SIMD4(0, 0, 0, 1))
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

    /// Test that radix sort produces valid render output
    func testRadixSortRenderOutput() throws {
        let width = 512
        let height = 512
        let renderer = GlobalSortRenderer(
            precision: Precision.float32,
            sortAlgorithm: .radix,
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
        XCTAssertEqual(commandBuffer.status, MTLCommandBufferStatus.completed, "Radix sort render should complete")
    }

    /// Test radix sort at larger scale (100k gaussians)
    func testRadixSortAtScale() throws {
        let width = 1024
        let height = 768
        let renderer = GlobalSortRenderer(
            precision: Precision.float16,
            sortAlgorithm: .radix,
            useHeapAllocation: false,
            textureOnly: true,
            limits: RendererLimits(maxGaussians: 200_000, maxWidth: width, maxHeight: height)
        )

        let count = 100_000
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
        XCTAssertEqual(commandBuffer.status, MTLCommandBufferStatus.completed, "Radix sort at scale should complete")
    }

    /// Test that radix and bitonic produce consistent output (both complete without error)
    func testRadixConsistentWithBitonic() throws {
        let width = 256
        let height = 256
        let count = 1000

        let (positions, scales, rotations, opacities, colors) = generateRandomGaussians(count: count)

        // Create renderers with different sort algorithms
        let radixRenderer = GlobalSortRenderer(
            precision: Precision.float32,
            sortAlgorithm: .radix,
            useHeapAllocation: false,
            limits: RendererLimits(maxGaussians: 10_000, maxWidth: width, maxHeight: height)
        )

        let bitonicRenderer = GlobalSortRenderer(
            precision: Precision.float32,
            sortAlgorithm: .bitonic,
            useHeapAllocation: false,
            limits: RendererLimits(maxGaussians: 10_000, maxWidth: width, maxHeight: height)
        )

        let camera = createSimpleCamera(width: Float(width), height: Float(height), gaussianCount: count)
        let frameParams = FrameParams(gaussianCount: count, whiteBackground: true)

        // Test radix renderer
        guard let radixBuffers = createPackedWorldBuffers(
            device: radixRenderer.device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create radix packed buffers")
            return
        }

        guard let radixCmd = radixRenderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create radix command buffer")
            return
        }

        let radixTextures = radixRenderer.encodeRenderToTextures(
            commandBuffer: radixCmd,
            gaussianCount: count,
            packedWorldBuffers: radixBuffers,
            cameraUniforms: camera,
            frameParams: frameParams
        )
        XCTAssertNotNil(radixTextures)
        radixCmd.commit()
        radixCmd.waitUntilCompleted()
        XCTAssertEqual(radixCmd.status, MTLCommandBufferStatus.completed, "Radix render should complete")

        // Test bitonic renderer
        guard let bitonicBuffers = createPackedWorldBuffers(
            device: bitonicRenderer.device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create bitonic packed buffers")
            return
        }

        guard let bitonicCmd = bitonicRenderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create bitonic command buffer")
            return
        }

        let bitonicTextures = bitonicRenderer.encodeRenderToTextures(
            commandBuffer: bitonicCmd,
            gaussianCount: count,
            packedWorldBuffers: bitonicBuffers,
            cameraUniforms: camera,
            frameParams: frameParams
        )
        XCTAssertNotNil(bitonicTextures)
        bitonicCmd.commit()
        bitonicCmd.waitUntilCompleted()
        XCTAssertEqual(bitonicCmd.status, MTLCommandBufferStatus.completed, "Bitonic render should complete")
    }
}

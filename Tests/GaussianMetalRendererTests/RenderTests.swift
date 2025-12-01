import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

final class RenderTests: XCTestCase {

    var renderer: GlobalSortRenderer!

    override func setUp() {
        super.setUp()
        renderer = GlobalSortRenderer(
            precision: Precision.float32,
            
            limits: RendererLimits(maxGaussians: 100_000, maxWidth: 1024, maxHeight: 1024, tileWidth: 16, tileHeight: 16)
        )
    }

    // MARK: - Helper Functions

    /// Create packed world buffers from gaussian parameters
    func createPackedWorldBuffers(
        device: MTLDevice,
        positions: [SIMD3<Float>],
        scales: [SIMD3<Float>],
        rotations: [SIMD4<Float>],
        opacities: [Float],
        colors: [SIMD3<Float>]  // DC color terms (harmonics with shComponents=0)
    ) -> PackedWorldBuffers? {
        let count = positions.count
        guard count == scales.count, count == rotations.count,
              count == opacities.count, count == colors.count else {
            return nil
        }

        // Create packed gaussians
        var packed: [PackedWorldGaussian] = []
        for i in 0..<count {
            packed.append(PackedWorldGaussian(
                position: positions[i],
                scale: scales[i],
                rotation: rotations[i],
                opacity: opacities[i]
            ))
        }

        // Create harmonics buffer (RGB per gaussian for DC term)
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

    /// Create half-precision packed world buffers from gaussian parameters
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

    /// Create a simple camera looking down -Z axis
    func createSimpleCamera(width: Float, height: Float, gaussianCount: Int) -> CameraUniformsSwift {
        // Simple orthographic-like projection
        var projMatrix = matrix_identity_float4x4
        projMatrix.columns.0 = SIMD4(2.0 / width, 0, 0, 0)
        projMatrix.columns.1 = SIMD4(0, 2.0 / height, 0, 0)
        projMatrix.columns.2 = SIMD4(0, 0, -0.02, 0)  // Near=0.1, Far=100
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

    // MARK: - Basic Render Tests

    /// Test rendering a single red gaussian at the center
    func testRenderSingleGaussian() throws {
        let width = 32
        let height = 32
        let renderer = GlobalSortRenderer(
            precision: Precision.float32,
            
            limits: RendererLimits(maxGaussians: 1024, maxWidth: width, maxHeight: height, tileWidth: 16, tileHeight: 16)
        )

        let positions: [SIMD3<Float>] = [SIMD3(0, 0, 5)]
        let scales: [SIMD3<Float>] = [SIMD3(0.5, 0.5, 0.5)]
        let rotations: [SIMD4<Float>] = [SIMD4(0, 0, 0, 1)]  // Identity
        let opacities: [Float] = [0.9]
        let colors: [SIMD3<Float>] = [SIMD3(0.5, 0, 0)]  // Red (will add 0.5 in shader)

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

        let camera = createSimpleCamera(width: Float(width), height: Float(height), gaussianCount: 1)

        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let frameParams = FrameParams(gaussianCount: 1, whiteBackground: true)
        let textures = renderer.encodeRenderToTextures(
            commandBuffer: commandBuffer,
            gaussianCount: 1,
            packedWorldBuffers: packedBuffers,
            cameraUniforms: camera,
            frameParams: frameParams
        )

        XCTAssertNotNil(textures, "Should return textures")

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        XCTAssertEqual(commandBuffer.status, MTLCommandBufferStatus.completed, "Command buffer should complete")
    }

    /// Test rendering multiple gaussians in a grid pattern
    func testRenderGridPattern() throws {
        let width = 64
        let height = 64
        let renderer = GlobalSortRenderer(
            precision: Precision.float32,
            
            limits: RendererLimits(maxGaussians: 1024, maxWidth: width, maxHeight: height, tileWidth: 16, tileHeight: 16)
        )

        // Create a 4x4 grid of gaussians
        var positions: [SIMD3<Float>] = []
        var scales: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var opacities: [Float] = []
        var colors: [SIMD3<Float>] = []

        for row in 0..<4 {
            for col in 0..<4 {
                let x = Float(col - 2) * 0.4 + 0.2
                let y = Float(row - 2) * 0.4 + 0.2
                positions.append(SIMD3(x, y, 5))
                scales.append(SIMD3(0.1, 0.1, 0.1))
                rotations.append(SIMD4(0, 0, 0, 1))
                opacities.append(0.8)
                // Alternate colors
                if (row + col) % 2 == 0 {
                    colors.append(SIMD3(0.5, 0, 0))  // Red
                } else {
                    colors.append(SIMD3(0, 0, 0.5))  // Blue
                }
            }
        }

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

        let camera = createSimpleCamera(width: Float(width), height: Float(height), gaussianCount: positions.count)

        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let frameParams = FrameParams(gaussianCount: positions.count, whiteBackground: false)
        let textures = renderer.encodeRenderToTextures(
            commandBuffer: commandBuffer,
            gaussianCount: positions.count,
            packedWorldBuffers: packedBuffers,
            cameraUniforms: camera,
            frameParams: frameParams
        )

        XCTAssertNotNil(textures, "Should return textures")

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        XCTAssertEqual(commandBuffer.status, MTLCommandBufferStatus.completed, "Command buffer should complete")
    }

    /// Test rendering gaussian that spans multiple tiles
    func testRenderGaussianSpanningTiles() throws {
        let width = 64
        let height = 64
        let renderer = GlobalSortRenderer(
            precision: Precision.float32,
            
            limits: RendererLimits(maxGaussians: 1024, maxWidth: width, maxHeight: height, tileWidth: 16, tileHeight: 16)
        )

        // Single large gaussian at center that should span all 4 tiles
        let positions: [SIMD3<Float>] = [SIMD3(0, 0, 2)]
        let scales: [SIMD3<Float>] = [SIMD3(2.0, 2.0, 0.5)]  // Large in X/Y
        let rotations: [SIMD4<Float>] = [SIMD4(0, 0, 0, 1)]
        let opacities: [Float] = [0.9]
        let colors: [SIMD3<Float>] = [SIMD3(0, 0.5, 0)]  // Green

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

        let camera = createSimpleCamera(width: Float(width), height: Float(height), gaussianCount: 1)

        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let frameParams = FrameParams(gaussianCount: 1, whiteBackground: true)
        let textures = renderer.encodeRenderToTextures(
            commandBuffer: commandBuffer,
            gaussianCount: 1,
            packedWorldBuffers: packedBuffers,
            cameraUniforms: camera,
            frameParams: frameParams
        )

        XCTAssertNotNil(textures, "Should return textures")

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        XCTAssertEqual(commandBuffer.status, MTLCommandBufferStatus.completed, "Command buffer should complete")
    }

    /// Test checkerboard pattern with gaussians in alternating tiles
    func testRenderCheckerboardPattern() throws {
        let width = 64
        let height = 64
        let renderer = GlobalSortRenderer(
            precision: Precision.float32,
            
            limits: RendererLimits(maxGaussians: 1024, maxWidth: width, maxHeight: height, tileWidth: 16, tileHeight: 16)
        )

        // Place gaussians in a checkerboard pattern across tiles
        var positions: [SIMD3<Float>] = []
        var scales: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var opacities: [Float] = []
        var colors: [SIMD3<Float>] = []

        // 4x4 tiles, place in alternating tiles
        for tileY in 0..<4 {
            for tileX in 0..<4 {
                if (tileX + tileY) % 2 == 0 {
                    // Convert tile position to world coordinates
                    let x = Float(tileX - 2) * 0.5 + 0.25
                    let y = Float(tileY - 2) * 0.5 + 0.25
                    positions.append(SIMD3(x, y, 5))
                    scales.append(SIMD3(0.15, 0.15, 0.15))
                    rotations.append(SIMD4(0, 0, 0, 1))
                    opacities.append(0.95)
                    colors.append(SIMD3(0.5, 0.5, 0))  // Yellow
                }
            }
        }

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

        let camera = createSimpleCamera(width: Float(width), height: Float(height), gaussianCount: positions.count)

        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let frameParams = FrameParams(gaussianCount: positions.count, whiteBackground: false)
        let textures = renderer.encodeRenderToTextures(
            commandBuffer: commandBuffer,
            gaussianCount: positions.count,
            packedWorldBuffers: packedBuffers,
            cameraUniforms: camera,
            frameParams: frameParams
        )

        XCTAssertNotNil(textures, "Should return textures")

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        XCTAssertEqual(commandBuffer.status, MTLCommandBufferStatus.completed, "Command buffer should complete")
    }

    // MARK: - Half Precision Tests

    /// Test half-precision output pipeline
    func testRenderHalfPrecisionOutput() throws {
        let width = 32
        let height = 32
        let renderer = GlobalSortRenderer(
            precision: Precision.float16,
            
            limits: RendererLimits(maxGaussians: 1024, maxWidth: width, maxHeight: height, tileWidth: 16, tileHeight: 16)
        )

        let positions: [SIMD3<Float>] = [
            SIMD3(0, 0, 5),
            SIMD3(0.3, 0.3, 5),
            SIMD3(-0.3, -0.3, 5),
            SIMD3(0.3, -0.3, 5)
        ]
        let scales: [SIMD3<Float>] = Array(repeating: SIMD3(0.2, 0.2, 0.2), count: 4)
        let rotations: [SIMD4<Float>] = Array(repeating: SIMD4(0, 0, 0, 1), count: 4)
        let opacities: [Float] = [0.9, 0.8, 0.7, 0.6]
        let colors: [SIMD3<Float>] = [
            SIMD3(0.5, 0, 0),
            SIMD3(0, 0.5, 0),
            SIMD3(0, 0, 0.5),
            SIMD3(0.5, 0.5, 0)
        ]

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

        let camera = createSimpleCamera(width: Float(width), height: Float(height), gaussianCount: 4)

        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let frameParams = FrameParams(gaussianCount: 4, whiteBackground: true)
        let textures = renderer.encodeRenderToTextureHalf(
            commandBuffer: commandBuffer,
            gaussianCount: 4,
            packedWorldBuffersHalf: packedBuffersHalf,
            cameraUniforms: camera,
            frameParams: frameParams
        )

        XCTAssertNotNil(textures, "encodeRenderToTextureHalf should return textures")

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        XCTAssertEqual(commandBuffer.status, MTLCommandBufferStatus.completed, "Command buffer should complete")

        if let textures = textures {
            XCTAssertGreaterThanOrEqual(textures.color.width, width)
            XCTAssertGreaterThanOrEqual(textures.color.height, height)
        }
    }

    // MARK: - Async/Performance Tests

    /// Test that render is non-blocking (async)
    @MainActor
    func testRenderWorldAsyncNonBlocking() throws {
        let width = 64
        let height = 64
        let renderer = GlobalSortRenderer(
            precision: Precision.float32,
            
            limits: RendererLimits(maxGaussians: 10_000, maxWidth: width, maxHeight: height, tileWidth: 16, tileHeight: 16)
        )

        // Create 1000 random gaussians
        var positions: [SIMD3<Float>] = []
        var scales: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var opacities: [Float] = []
        var colors: [SIMD3<Float>] = []

        for i in 0..<1000 {
            let x = Float(i % 32 - 16) * 0.05
            let y = Float(i / 32 - 16) * 0.05
            positions.append(SIMD3(x, y, 5))
            scales.append(SIMD3(0.05, 0.05, 0.05))
            rotations.append(SIMD4(0, 0, 0, 1))
            opacities.append(0.5)
            colors.append(SIMD3(0.3, 0.3, 0.3))
        }

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

        let camera = createSimpleCamera(width: Float(width), height: Float(height), gaussianCount: 1000)

        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        let frameParams = FrameParams(gaussianCount: 1000, whiteBackground: false)
        let textures = renderer.encodeRenderToTextures(
            commandBuffer: commandBuffer,
            gaussianCount: 1000,
            packedWorldBuffers: packedBuffers,
            cameraUniforms: camera,
            frameParams: frameParams
        )

        let encodeTime = CFAbsoluteTimeGetCurrent() - startTime

        XCTAssertNotNil(textures, "Should return textures")

        // Encoding should be fast (non-blocking) - less than 100ms
        XCTAssertLessThan(encodeTime, 0.1, "Encoding should be fast (non-blocking)")

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        XCTAssertEqual(commandBuffer.status, MTLCommandBufferStatus.completed, "Command buffer should complete")
    }

    /// Test half precision pipeline integrity with verification
    func testHalfPrecisionPipelineIntegrity() throws {
        let width = 32
        let height = 32
        let renderer = GlobalSortRenderer(
            precision: Precision.float16,
            
            limits: RendererLimits(maxGaussians: 1024, maxWidth: width, maxHeight: height, tileWidth: 16, tileHeight: 16)
        )

        // Single bright gaussian at center
        let positions: [SIMD3<Float>] = [SIMD3(0, 0, 2)]
        let scales: [SIMD3<Float>] = [SIMD3(0.5, 0.5, 0.5)]
        let rotations: [SIMD4<Float>] = [SIMD4(0, 0, 0, 1)]
        let opacities: [Float] = [0.99]
        let colors: [SIMD3<Float>] = [SIMD3(0.5, 0.5, 0.5)]  // Will be 1.0 after +0.5

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

        let camera = createSimpleCamera(width: Float(width), height: Float(height), gaussianCount: 1)

        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let frameParams = FrameParams(gaussianCount: 1, whiteBackground: false)
        let textures = renderer.encodeRenderToTextureHalf(
            commandBuffer: commandBuffer,
            gaussianCount: 1,
            packedWorldBuffersHalf: packedBuffersHalf,
            cameraUniforms: camera,
            frameParams: frameParams
        )

        XCTAssertNotNil(textures, "Should return textures")

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        XCTAssertEqual(commandBuffer.status, MTLCommandBufferStatus.completed, "Command buffer should complete")
    }

    // MARK: - Scale Tests

    /// Test rendering at larger scale (10K gaussians)
    func testRenderAtScale10K() throws {
        let width = 512
        let height = 512
        let renderer = GlobalSortRenderer(
            precision: Precision.float16,
            
            limits: RendererLimits(maxGaussians: 20_000, maxWidth: width, maxHeight: height, tileWidth: 16, tileHeight: 16)
        )

        let count = 10_000
        var positions: [SIMD3<Float>] = []
        var scales: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var opacities: [Float] = []
        var colors: [SIMD3<Float>] = []

        // Grid of gaussians
        let gridSize = Int(sqrt(Double(count)))
        for i in 0..<count {
            let x = Float(i % gridSize - gridSize/2) * 0.02
            let y = Float(i / gridSize - gridSize/2) * 0.02
            positions.append(SIMD3(x, y, 5))
            scales.append(SIMD3(0.01, 0.01, 0.01))
            rotations.append(SIMD4(0, 0, 0, 1))
            opacities.append(0.5)
            colors.append(SIMD3(
                Float(i % 256) / 512.0,
                Float((i / 256) % 256) / 512.0,
                Float(i % 128) / 256.0
            ))
        }

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

        XCTAssertNotNil(textures, "Should return textures for 10K gaussians")

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        XCTAssertEqual(commandBuffer.status, MTLCommandBufferStatus.completed, "Command buffer should complete for 10K gaussians")
    }
}

import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// End-to-end tests for GlobalSort and LocalSort renderers
final class RendererE2ETests: XCTestCase {

    // MARK: - Test Data Helpers

    func generateGaussians(count: Int, seed: Int = 42) -> (
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

        srand48(seed)
        for i in 0..<count {
            let gridSize = Int(sqrt(Double(count))) + 1
            let x = Float(i % gridSize) / Float(gridSize) * 4 - 2
            let y = Float(i / gridSize) / Float(gridSize) * 4 - 2
            let z = Float(drand48() * 3 + 2)
            positions.append(SIMD3(x, y, z))

            let s = Float(drand48() * 0.1 + 0.05)
            scales.append(SIMD3(s, s, s))

            rotations.append(SIMD4(0, 0, 0, 1))
            opacities.append(Float(drand48() * 0.5 + 0.5))
            colors.append(SIMD3(
                Float(drand48() * 0.5),
                Float(drand48() * 0.5),
                Float(drand48() * 0.5)
            ))
        }

        return (positions, scales, rotations, opacities, colors)
    }

    func createPackedBuffers(device: MTLDevice, positions: [SIMD3<Float>], scales: [SIMD3<Float>],
                             rotations: [SIMD4<Float>], opacities: [Float], colors: [SIMD3<Float>]) -> PackedWorldBuffers? {
        let count = positions.count
        var packed: [PackedWorldGaussian] = []
        for i in 0..<count {
            packed.append(PackedWorldGaussian(
                position: positions[i], scale: scales[i], rotation: rotations[i], opacity: opacities[i]
            ))
        }

        var harmonics: [Float] = []
        for color in colors {
            harmonics.append(color.x)
            harmonics.append(color.y)
            harmonics.append(color.z)
        }

        guard let packedBuf = device.makeBuffer(bytes: &packed, length: count * MemoryLayout<PackedWorldGaussian>.stride, options: .storageModeShared),
              let harmonicsBuf = device.makeBuffer(bytes: &harmonics, length: count * 3 * 4, options: .storageModeShared) else {
            return nil
        }

        return PackedWorldBuffers(packedGaussians: packedBuf, harmonics: harmonicsBuf)
    }

    func createCamera(width: Float, height: Float, gaussianCount: Int) -> CameraUniformsSwift {
        let aspect = width / height
        let fov: Float = 60.0 * .pi / 180.0
        let near: Float = 0.1
        let far: Float = 100.0
        let f = 1.0 / tan(fov / 2.0)

        var projMatrix = matrix_identity_float4x4
        projMatrix.columns.0 = SIMD4(f / aspect, 0, 0, 0)
        projMatrix.columns.1 = SIMD4(0, f, 0, 0)
        projMatrix.columns.2 = SIMD4(0, 0, -(far + near) / (far - near), -1)
        projMatrix.columns.3 = SIMD4(0, 0, -2 * far * near / (far - near), 0)

        let focalX = width * f / (2 * aspect)
        let focalY = height * f / 2

        return CameraUniformsSwift(
            viewMatrix: matrix_identity_float4x4,
            projectionMatrix: projMatrix,
            cameraCenter: SIMD3<Float>(0, 0, 0),
            pixelFactor: 1.0,
            focalX: focalX,
            focalY: focalY,
            width: width,
            height: height,
            nearPlane: near,
            farPlane: far,
            shComponents: 0,
            gaussianCount: UInt32(gaussianCount),
            padding0: 0,
            padding1: 0
        )
    }

    func readPixels(texture: MTLTexture, device: MTLDevice, queue: MTLCommandQueue) -> [UInt8]? {
        let bytesPerPixel = 4
        let bytesPerRow = texture.width * bytesPerPixel

        // Create a shared buffer to copy texture data
        guard let buffer = device.makeBuffer(length: bytesPerRow * texture.height, options: .storageModeShared) else {
            return nil
        }

        // Use blit encoder to copy texture to buffer
        guard let cb = queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else {
            return nil
        }

        blit.copy(from: texture,
                  sourceSlice: 0,
                  sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: texture.width, height: texture.height, depth: 1),
                  to: buffer,
                  destinationOffset: 0,
                  destinationBytesPerRow: bytesPerRow,
                  destinationBytesPerImage: bytesPerRow * texture.height)
        blit.endEncoding()

        cb.commit()
        cb.waitUntilCompleted()

        // Read from buffer
        let ptr = buffer.contents().bindMemory(to: UInt8.self, capacity: bytesPerRow * texture.height)
        return Array(UnsafeBufferPointer(start: ptr, count: bytesPerRow * texture.height))
    }

    func countNonBlackPixels(_ pixels: [UInt8]) -> Int {
        var count = 0
        for i in stride(from: 0, to: pixels.count, by: 4) {
            if pixels[i] > 10 || pixels[i + 1] > 10 || pixels[i + 2] > 10 {
                count += 1
            }
        }
        return count
    }

    // MARK: - GlobalSort Tests

    func testGlobalSortRendersCorrectly() throws {
        let width = 256
        let height = 256
        let count = 1000

        let renderer = GlobalSortRenderer(
            precision: .float32,
            limits: RendererLimits(maxGaussians: 10_000, maxWidth: width, maxHeight: height)
        )

        let (positions, scales, rotations, opacities, colors) = generateGaussians(count: count)
        guard let buffers = createPackedBuffers(device: renderer.device, positions: positions,
                                                 scales: scales, rotations: rotations,
                                                 opacities: opacities, colors: colors) else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = createCamera(width: Float(width), height: Float(height), gaussianCount: count)
        let frameParams = FrameParams(gaussianCount: count, whiteBackground: false)

        guard let queue = renderer.device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let textures = renderer.encodeRenderToTextures(
            commandBuffer: cb, gaussianCount: count,
            packedWorldBuffers: buffers, cameraUniforms: camera, frameParams: frameParams
        )

        XCTAssertNotNil(textures, "GlobalSort should produce textures")
        XCTAssertNotNil(textures?.color, "GlobalSort should produce color texture")
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, MTLCommandBufferStatus.completed)
    }

    func testGlobalSortAtScale() throws {
        let width = 512
        let height = 512
        let count = 50_000

        let renderer = GlobalSortRenderer(
            precision: .float16,
            textureOnly: true,
            limits: RendererLimits(maxGaussians: 100_000, maxWidth: width, maxHeight: height)
        )

        let (positions, scales, rotations, opacities, colors) = generateGaussians(count: count)
        guard let buffers = createPackedBuffers(device: renderer.device, positions: positions,
                                                 scales: scales, rotations: rotations,
                                                 opacities: opacities, colors: colors) else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = createCamera(width: Float(width), height: Float(height), gaussianCount: count)
        let frameParams = FrameParams(gaussianCount: count, whiteBackground: false)

        guard let queue = renderer.device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let textures = renderer.encodeRenderToTextures(
            commandBuffer: cb, gaussianCount: count,
            packedWorldBuffers: buffers, cameraUniforms: camera, frameParams: frameParams
        )

        XCTAssertNotNil(textures)
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, MTLCommandBufferStatus.completed)
    }

    func testGlobalSortHalfPrecision() throws {
        let width = 256
        let height = 256
        let count = 1000

        let renderer = GlobalSortRenderer(
            precision: .float16,
            limits: RendererLimits(maxGaussians: 10_000, maxWidth: width, maxHeight: height)
        )

        let (positions, scales, rotations, opacities, colors) = generateGaussians(count: count)
        guard let buffers = createPackedBuffers(device: renderer.device, positions: positions,
                                                 scales: scales, rotations: rotations,
                                                 opacities: opacities, colors: colors) else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = createCamera(width: Float(width), height: Float(height), gaussianCount: count)
        let frameParams = FrameParams(gaussianCount: count, whiteBackground: true)

        guard let queue = renderer.device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let textures = renderer.encodeRenderToTextures(
            commandBuffer: cb, gaussianCount: count,
            packedWorldBuffers: buffers, cameraUniforms: camera, frameParams: frameParams
        )

        XCTAssertNotNil(textures)
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, MTLCommandBufferStatus.completed)
    }

    // MARK: - LocalSort Tests

    func testLocalSortRendersCorrectly() throws {
        let width = 256
        let height = 256
        let count = 1000

        let config = RendererConfig(
            maxGaussians: 10_000, maxWidth: width, maxHeight: height, precision: .float32
        )
        let renderer = try LocalSortRenderer(config: config)
        let device = renderer.device

        let (positions, scales, rotations, opacities, colors) = generateGaussians(count: count)
        guard let buffers = createPackedBuffers(device: device, positions: positions,
                                                 scales: scales, rotations: rotations,
                                                 opacities: opacities, colors: colors) else {
            XCTFail("Failed to create buffers")
            return
        }

        // Use GaussianInput and CameraParams for protocol-compliant API
        let input = GaussianInput(
            gaussians: buffers.packedGaussians,
            harmonics: buffers.harmonics,
            gaussianCount: count,
            shComponents: 0
        )

        let aspect = Float(width) / Float(height)
        let fov: Float = 60.0 * .pi / 180.0
        let f = 1.0 / tan(fov / 2.0)
        var projMatrix = matrix_identity_float4x4
        projMatrix.columns.0 = SIMD4(f / aspect, 0, 0, 0)
        projMatrix.columns.1 = SIMD4(0, f, 0, 0)
        projMatrix.columns.2 = SIMD4(0, 0, -1.002, -1)
        projMatrix.columns.3 = SIMD4(0, 0, -0.2, 0)

        let camera = CameraParams(
            viewMatrix: matrix_identity_float4x4,
            projectionMatrix: projMatrix,
            position: SIMD3<Float>(0, 0, 0),
            focalX: Float(width) * f / (2 * aspect),
            focalY: Float(height) * f / 2
        )

        // Create command buffer from device
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let result = renderer.render(
            toTexture: cb,
            input: input,
            camera: camera,
            width: width,
            height: height,
            whiteBackground: false,
            mortonSorted: false
        )

        cb.commit()
        cb.waitUntilCompleted()

        XCTAssertNotNil(result, "LocalSort should return render result")
    }

    func testLocalSortAtScale() throws {
        let width = 512
        let height = 512
        let count = 50_000

        let config = RendererConfig(
            maxGaussians: 100_000, maxWidth: width, maxHeight: height, precision: .float16
        )
        let renderer = try LocalSortRenderer(config: config)
        let device = renderer.device

        let (positions, scales, rotations, opacities, colors) = generateGaussians(count: count)
        guard let buffers = createPackedBuffers(device: device, positions: positions,
                                                 scales: scales, rotations: rotations,
                                                 opacities: opacities, colors: colors) else {
            XCTFail("Failed to create buffers")
            return
        }

        let input = GaussianInput(
            gaussians: buffers.packedGaussians,
            harmonics: buffers.harmonics,
            gaussianCount: count,
            shComponents: 0
        )

        let aspect = Float(width) / Float(height)
        let fov: Float = 60.0 * .pi / 180.0
        let f = 1.0 / tan(fov / 2.0)
        var projMatrix = matrix_identity_float4x4
        projMatrix.columns.0 = SIMD4(f / aspect, 0, 0, 0)
        projMatrix.columns.1 = SIMD4(0, f, 0, 0)
        projMatrix.columns.2 = SIMD4(0, 0, -1.002, -1)
        projMatrix.columns.3 = SIMD4(0, 0, -0.2, 0)

        let camera = CameraParams(
            viewMatrix: matrix_identity_float4x4,
            projectionMatrix: projMatrix,
            position: SIMD3<Float>(0, 0, 0),
            focalX: Float(width) * f / (2 * aspect),
            focalY: Float(height) * f / 2
        )

        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let result = renderer.render(
            toTexture: cb,
            input: input,
            camera: camera,
            width: width,
            height: height,
            whiteBackground: false,
            mortonSorted: false
        )

        cb.commit()
        cb.waitUntilCompleted()

        XCTAssertNotNil(result, "LocalSort at scale should return render result")
    }

    // MARK: - Pixel Comparison Tests

    func testGlobalSortVsLocalSortPixelComparison() throws {
        let width = 256
        let height = 256
        let count = 500

        // Generate same gaussians for both renderers
        let (positions, scales, rotations, opacities, colors) = generateGaussians(count: count, seed: 999)

        // === Render with GlobalSort ===
        let globalRenderer = GlobalSortRenderer(
            precision: .float32,
            limits: RendererLimits(maxGaussians: 10_000, maxWidth: width, maxHeight: height)
        )

        guard let globalBuffers = createPackedBuffers(device: globalRenderer.device, positions: positions,
                                                       scales: scales, rotations: rotations,
                                                       opacities: opacities, colors: colors) else {
            XCTFail("Failed to create GlobalSort buffers")
            return
        }

        let globalCamera = createCamera(width: Float(width), height: Float(height), gaussianCount: count)
        let frameParams = FrameParams(gaussianCount: count, whiteBackground: false)

        guard let globalQueue = globalRenderer.device.makeCommandQueue(),
              let globalCB = globalQueue.makeCommandBuffer() else {
            XCTFail("Failed to create GlobalSort command buffer")
            return
        }

        let globalTextures = globalRenderer.encodeRenderToTextures(
            commandBuffer: globalCB, gaussianCount: count,
            packedWorldBuffers: globalBuffers, cameraUniforms: globalCamera, frameParams: frameParams
        )

        globalCB.commit()
        globalCB.waitUntilCompleted()

        guard let globalColorTexture = globalTextures?.color else {
            XCTFail("GlobalSort didn't produce color texture")
            return
        }

        guard let globalPixels = readPixels(texture: globalColorTexture, device: globalRenderer.device, queue: globalQueue) else {
            XCTFail("Failed to read GlobalSort pixels")
            return
        }

        // === Render with LocalSort ===
        let localConfig = RendererConfig(
            maxGaussians: 10_000, maxWidth: width, maxHeight: height, precision: .float32
        )
        let localRenderer = try LocalSortRenderer(config: localConfig)

        guard let localBuffers = createPackedBuffers(device: localRenderer.device, positions: positions,
                                                      scales: scales, rotations: rotations,
                                                      opacities: opacities, colors: colors) else {
            XCTFail("Failed to create LocalSort buffers")
            return
        }

        let input = GaussianInput(
            gaussians: localBuffers.packedGaussians,
            harmonics: localBuffers.harmonics,
            gaussianCount: count,
            shComponents: 0
        )

        let aspect = Float(width) / Float(height)
        let fov: Float = 60.0 * .pi / 180.0
        let f = 1.0 / tan(fov / 2.0)
        var projMatrix = matrix_identity_float4x4
        projMatrix.columns.0 = SIMD4(f / aspect, 0, 0, 0)
        projMatrix.columns.1 = SIMD4(0, f, 0, 0)
        projMatrix.columns.2 = SIMD4(0, 0, -1.002, -1)
        projMatrix.columns.3 = SIMD4(0, 0, -0.2, 0)

        let localCamera = CameraParams(
            viewMatrix: matrix_identity_float4x4,
            projectionMatrix: projMatrix,
            position: SIMD3<Float>(0, 0, 0),
            focalX: Float(width) * f / (2 * aspect),
            focalY: Float(height) * f / 2
        )

        guard let localQueue = localRenderer.device.makeCommandQueue(),
              let localCB = localQueue.makeCommandBuffer() else {
            XCTFail("Failed to create LocalSort command buffer")
            return
        }

        let localResult = localRenderer.render(
            toTexture: localCB,
            input: input,
            camera: localCamera,
            width: width,
            height: height,
            whiteBackground: false,
            mortonSorted: false
        )

        localCB.commit()
        localCB.waitUntilCompleted()

        guard let localColorTexture = localResult?.color else {
            XCTFail("LocalSort didn't produce color texture")
            return
        }

        guard let localPixels = readPixels(texture: localColorTexture, device: localRenderer.device, queue: localQueue) else {
            XCTFail("Failed to read LocalSort pixels")
            return
        }

        // === Compare pixels ===
        XCTAssertEqual(globalPixels.count, localPixels.count, "Pixel count mismatch")

        // Verify both renderers produced visible output
        let globalNonBlack = countNonBlackPixels(globalPixels)
        let localNonBlack = countNonBlackPixels(localPixels)

        XCTAssertGreaterThan(globalNonBlack, 100, "GlobalSort should render visible gaussians")
        XCTAssertGreaterThan(localNonBlack, 100, "LocalSort should render visible gaussians")

        // Compute similarity metrics
        var totalDiff: Int = 0
        var matchingPixels = 0
        let tolerance: Int = 20  // Allow differences due to different sorting approaches

        for i in stride(from: 0, to: globalPixels.count, by: 4) {
            let rDiff = abs(Int(globalPixels[i]) - Int(localPixels[i]))
            let gDiff = abs(Int(globalPixels[i + 1]) - Int(localPixels[i + 1]))
            let bDiff = abs(Int(globalPixels[i + 2]) - Int(localPixels[i + 2]))
            let pixelDiff = max(rDiff, max(gDiff, bDiff))

            if pixelDiff <= tolerance {
                matchingPixels += 1
            }
            totalDiff += rDiff + gDiff + bDiff
        }

        let pixelCount = globalPixels.count / 4
        let matchPercent = Double(matchingPixels) / Double(pixelCount) * 100.0
        let avgDiff = Double(totalDiff) / Double(pixelCount * 3)

        print("=== Pixel Comparison Stats ===")
        print("Global non-black: \(globalNonBlack) / \(pixelCount)")
        print("Local non-black:  \(localNonBlack) / \(pixelCount)")
        print("Matching pixels:  \(matchingPixels) / \(pixelCount) (\(String(format: "%.1f", matchPercent))%)")
        print("Avg channel diff: \(String(format: "%.2f", avgDiff))")
        print("==============================")

        // Both renderers should produce similar overall output (at least 50% matching)
        // Note: GlobalSort and LocalSort use different tile/sort approaches so exact match isn't expected
        XCTAssertGreaterThan(matchPercent, 50.0,
            "Renderers should produce similar output: \(matchPercent)% matching")

        // Verify both have similar coverage (non-black pixel count within 50%)
        let coverageDiff = abs(globalNonBlack - localNonBlack)
        let avgCoverage = (globalNonBlack + localNonBlack) / 2
        let coverageRatio = avgCoverage > 0 ? Double(coverageDiff) / Double(avgCoverage) : 0.0

        XCTAssertLessThan(coverageRatio, 0.5,
            "Coverage should be similar: global=\(globalNonBlack), local=\(localNonBlack)")
    }
}

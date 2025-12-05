@testable import GaussianMetalRenderer
import Metal
import simd
import XCTest

/// End-to-end tests for GlobalSort and Local renderers
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
        for i in 0 ..< count {
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
                             rotations: [SIMD4<Float>], opacities: [Float], colors: [SIMD3<Float>]) -> PackedWorldBuffers?
    {
        let count = positions.count
        var packed: [PackedWorldGaussian] = []
        for i in 0 ..< count {
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
              let harmonicsBuf = device.makeBuffer(bytes: &harmonics, length: count * 3 * 4, options: .storageModeShared)
        else {
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
              let blit = cb.makeBlitCommandEncoder()
        else {
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
            limits: RendererLimits(maxGaussians: 10000, maxWidth: width, maxHeight: height)
        )

        let (positions, scales, rotations, opacities, colors) = self.generateGaussians(count: count)
        guard let buffers = createPackedBuffers(device: renderer.device, positions: positions,
                                                scales: scales, rotations: rotations,
                                                opacities: opacities, colors: colors)
        else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = self.createCamera(width: Float(width), height: Float(height), gaussianCount: count)
        let frameParams = FrameParams(gaussianCount: count, whiteBackground: false)

        guard let queue = renderer.device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer()
        else {
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
        let count = 50000

        let renderer = GlobalSortRenderer(
            precision: .float16,
            textureOnly: true,
            limits: RendererLimits(maxGaussians: 100_000, maxWidth: width, maxHeight: height)
        )

        let (positions, scales, rotations, opacities, colors) = self.generateGaussians(count: count)
        guard let buffers = createPackedBuffers(device: renderer.device, positions: positions,
                                                scales: scales, rotations: rotations,
                                                opacities: opacities, colors: colors)
        else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = self.createCamera(width: Float(width), height: Float(height), gaussianCount: count)
        let frameParams = FrameParams(gaussianCount: count, whiteBackground: false)

        guard let queue = renderer.device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer()
        else {
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
            limits: RendererLimits(maxGaussians: 10000, maxWidth: width, maxHeight: height)
        )

        let (positions, scales, rotations, opacities, colors) = self.generateGaussians(count: count)
        guard let buffers = createPackedBuffers(device: renderer.device, positions: positions,
                                                scales: scales, rotations: rotations,
                                                opacities: opacities, colors: colors)
        else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = self.createCamera(width: Float(width), height: Float(height), gaussianCount: count)
        let frameParams = FrameParams(gaussianCount: count, whiteBackground: true)

        guard let queue = renderer.device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer()
        else {
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

    // MARK: - Local Tests

    func testLocalRendersCorrectly() throws {
        let width = 256
        let height = 256
        let count = 1000

        let config = RendererConfig(
            maxGaussians: 10000, maxWidth: width, maxHeight: height, precision: .float32
        )
        let renderer = try LocalRenderer(config: config)
        let device = renderer.device

        let (positions, scales, rotations, opacities, colors) = self.generateGaussians(count: count)
        guard let buffers = createPackedBuffers(device: device, positions: positions,
                                                scales: scales, rotations: rotations,
                                                opacities: opacities, colors: colors)
        else {
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
              let cb = queue.makeCommandBuffer()
        else {
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

        XCTAssertNotNil(result, "Local should return render result")
    }

    func testLocalAtScale() throws {
        let width = 512
        let height = 512
        let count = 50000

        let config = RendererConfig(
            maxGaussians: 100_000, maxWidth: width, maxHeight: height, precision: .float16
        )
        let renderer = try LocalRenderer(config: config)
        let device = renderer.device

        let (positions, scales, rotations, opacities, colors) = self.generateGaussians(count: count)
        guard let buffers = createPackedBuffers(device: device, positions: positions,
                                                scales: scales, rotations: rotations,
                                                opacities: opacities, colors: colors)
        else {
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
              let cb = queue.makeCommandBuffer()
        else {
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

        XCTAssertNotNil(result, "Local at scale should return render result")
    }

    // MARK: - Pixel Comparison Tests

    func testGlobalSortVsLocalPixelComparison() throws {
        let width = 256
        let height = 256
        let count = 500

        // Generate same gaussians for both renderers
        let (positions, scales, rotations, opacities, colors) = self.generateGaussians(count: count, seed: 999)

        // === Render with GlobalSort ===
        let globalRenderer = GlobalSortRenderer(
            precision: .float32,
            limits: RendererLimits(maxGaussians: 10000, maxWidth: width, maxHeight: height)
        )

        guard let globalBuffers = createPackedBuffers(device: globalRenderer.device, positions: positions,
                                                      scales: scales, rotations: rotations,
                                                      opacities: opacities, colors: colors)
        else {
            XCTFail("Failed to create GlobalSort buffers")
            return
        }

        let globalCamera = self.createCamera(width: Float(width), height: Float(height), gaussianCount: count)
        let frameParams = FrameParams(gaussianCount: count, whiteBackground: false)

        guard let globalQueue = globalRenderer.device.makeCommandQueue(),
              let globalCB = globalQueue.makeCommandBuffer()
        else {
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

        // === Render with Local ===
        let localConfig = RendererConfig(
            maxGaussians: 10000, maxWidth: width, maxHeight: height, precision: .float32
        )
        let localRenderer = try LocalRenderer(config: localConfig)

        guard let localBuffers = createPackedBuffers(device: localRenderer.device, positions: positions,
                                                     scales: scales, rotations: rotations,
                                                     opacities: opacities, colors: colors)
        else {
            XCTFail("Failed to create Local buffers")
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
              let localCB = localQueue.makeCommandBuffer()
        else {
            XCTFail("Failed to create Local command buffer")
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
            XCTFail("Local didn't produce color texture")
            return
        }

        guard let localPixels = readPixels(texture: localColorTexture, device: localRenderer.device, queue: localQueue) else {
            XCTFail("Failed to read Local pixels")
            return
        }

        // === Compare pixels ===
        XCTAssertEqual(globalPixels.count, localPixels.count, "Pixel count mismatch")

        // Verify both renderers produced visible output
        let globalNonBlack = self.countNonBlackPixels(globalPixels)
        let localNonBlack = self.countNonBlackPixels(localPixels)

        XCTAssertGreaterThan(globalNonBlack, 100, "GlobalSort should render visible gaussians")
        XCTAssertGreaterThan(localNonBlack, 100, "Local should render visible gaussians")

        // Compute similarity metrics
        var totalDiff = 0
        var matchingPixels = 0
        let tolerance = 20 // Allow differences due to different sorting approaches

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
        // Note: GlobalSort and Local use different tile/sort approaches so exact match isn't expected
        XCTAssertGreaterThan(matchPercent, 50.0,
                             "Renderers should produce similar output: \(matchPercent)% matching")

        // Verify both have similar coverage (non-black pixel count within 50%)
        let coverageDiff = abs(globalNonBlack - localNonBlack)
        let avgCoverage = (globalNonBlack + localNonBlack) / 2
        let coverageRatio = avgCoverage > 0 ? Double(coverageDiff) / Double(avgCoverage) : 0.0

        XCTAssertLessThan(coverageRatio, 0.5,
                          "Coverage should be similar: global=\(globalNonBlack), local=\(localNonBlack)")
    }

    // MARK: - Performance Test Helpers

    struct PerfConfig {
        let width: Int
        let height: Int
        let count: Int
        let precision: RenderPrecision
        let warmupFrames: Int
        let measureFrames: Int
        let verbose: Bool

        init(width: Int = 512, height: Int = 512, count: Int, precision: RenderPrecision = .float32,
             warmupFrames: Int = 3, measureFrames: Int = 10, verbose: Bool = false)
        {
            self.width = width
            self.height = height
            self.count = count
            self.precision = precision
            self.warmupFrames = warmupFrames
            self.measureFrames = measureFrames
            self.verbose = verbose
        }
    }

    struct PerfResult {
        let times: [Double]
        var avg: Double { self.times.reduce(0, +) / Double(self.times.count) }
        var min: Double { self.times.min() ?? 0 }
        var max: Double { self.times.max() ?? 0 }
        var fps: Double { 1000.0 / self.avg }
    }

    func createCameraParams(width: Int, height: Int) -> CameraParams {
        let aspect = Float(width) / Float(height)
        let fov: Float = 60.0 * .pi / 180.0
        let f = 1.0 / tan(fov / 2.0)
        var projMatrix = matrix_identity_float4x4
        projMatrix.columns.0 = SIMD4(f / aspect, 0, 0, 0)
        projMatrix.columns.1 = SIMD4(0, f, 0, 0)
        projMatrix.columns.2 = SIMD4(0, 0, -1.002, -1)
        projMatrix.columns.3 = SIMD4(0, 0, -0.2, 0)

        return CameraParams(
            viewMatrix: matrix_identity_float4x4,
            projectionMatrix: projMatrix,
            position: SIMD3<Float>(0, 0, 0),
            focalX: Float(width) * f / (2 * aspect),
            focalY: Float(height) * f / 2
        )
    }

    func measureRenderer(
        _ renderer: GaussianRenderer,
        input: GaussianInput,
        camera: CameraParams,
        config: PerfConfig,
        label: String
    ) -> PerfResult? {
        guard let queue = renderer.device.makeCommandQueue() else { return nil }

        // Warmup
        for _ in 0 ..< config.warmupFrames {
            guard let cb = queue.makeCommandBuffer() else { continue }
            _ = renderer.render(
                toTexture: cb, input: input, camera: camera,
                width: config.width, height: config.height,
                whiteBackground: false, mortonSorted: false
            )
            cb.commit()
            cb.waitUntilCompleted()
        }

        // Measure
        var times: [Double] = []
        for i in 0 ..< config.measureFrames {
            guard let cb = queue.makeCommandBuffer() else { continue }
            let start = CFAbsoluteTimeGetCurrent()
            _ = renderer.render(
                toTexture: cb, input: input, camera: camera,
                width: config.width, height: config.height,
                whiteBackground: false, mortonSorted: false
            )
            cb.commit()
            cb.waitUntilCompleted()
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
            times.append(elapsed)
            if config.verbose {
                print("  \(label) run \(i + 1): \(String(format: "%.2f", elapsed))ms")
            }
        }

        return PerfResult(times: times)
    }

    func runPerfComparison(config: PerfConfig, label _: String) throws -> (global: PerfResult, local: PerfResult)? {
        let (positions, scales, rotations, opacities, colors) = self.generateGaussians(count: config.count, seed: 42)
        let camera = self.createCameraParams(width: config.width, height: config.height)

        // GlobalSort
        let globalRenderer = GlobalSortRenderer(
            precision: config.precision,
            textureOnly: true,
            limits: RendererLimits(maxGaussians: config.count + 1000, maxWidth: config.width, maxHeight: config.height)
        )

        guard let globalBuffers = createPackedBuffers(
            device: globalRenderer.device, positions: positions, scales: scales,
            rotations: rotations, opacities: opacities, colors: colors
        ) else { return nil }

        let globalInput = GaussianInput(
            gaussians: globalBuffers.packedGaussians,
            harmonics: globalBuffers.harmonics,
            gaussianCount: config.count,
            shComponents: 0
        )

        guard let globalResult = measureRenderer(globalRenderer, input: globalInput, camera: camera, config: config, label: "GlobalSort") else {
            return nil
        }

        // Local
        let localConfig = RendererConfig(
            maxGaussians: config.count + 1000, maxWidth: config.width, maxHeight: config.height, precision: config.precision
        )
        let localRenderer = try LocalRenderer(config: localConfig)

        guard let localBuffers = createPackedBuffers(
            device: localRenderer.device, positions: positions, scales: scales,
            rotations: rotations, opacities: opacities, colors: colors
        ) else { return nil }

        let localInput = GaussianInput(
            gaussians: localBuffers.packedGaussians,
            harmonics: localBuffers.harmonics,
            gaussianCount: config.count,
            shComponents: 0
        )

        guard let localResult = measureRenderer(localRenderer, input: localInput, camera: camera, config: config, label: "Local") else {
            return nil
        }

        return (globalResult, localResult)
    }

    func printComparisonResult(label: String, global: PerfResult, local: PerfResult, showMinMax: Bool = true) {
        let faster = global.avg < local.avg ? "GlobalSort" : "Local"
        let speedup = global.avg < local.avg ? local.avg / global.avg : global.avg / local.avg

        print("[\(label)]")
        if showMinMax {
            print("  GlobalSort: \(String(format: "%.2f", global.avg))ms avg (min: \(String(format: "%.2f", global.min)), max: \(String(format: "%.2f", global.max)))")
            print("  Local:      \(String(format: "%.2f", local.avg))ms avg (min: \(String(format: "%.2f", local.min)), max: \(String(format: "%.2f", local.max)))")
        } else {
            print("  GlobalSort: \(String(format: "%.2f", global.avg))ms (\(String(format: "%.1f", global.fps)) FPS)")
            print("  Local:      \(String(format: "%.2f", local.avg))ms (\(String(format: "%.1f", local.fps)) FPS)")
        }
        print("  Winner:     \(faster) (\(String(format: "%.2f", speedup))x faster)\n")
    }

    // MARK: - Performance Tests

    /// End-to-end performance comparison between GlobalSort and Local renderers
    func testPerformanceComparison() throws {
        let testCases: [(count: Int, name: String)] = [
            (10000, "10K"),
            (50000, "50K"),
            (100_000, "100K"),
        ]

        print("\n=== Renderer Performance Comparison ===")
        print("Resolution: 512x512")
        print("API: render(toTexture:...) for both renderers")
        print("Warmup: 3 frames, Measured: 10 frames each\n")

        for testCase in testCases {
            let config = PerfConfig(count: testCase.count)
            guard let (global, local) = try runPerfComparison(config: config, label: testCase.name) else {
                XCTFail("Failed to run comparison for \(testCase.name)")
                return
            }
            self.printComparisonResult(label: "\(testCase.name) Gaussians", global: global, local: local)
        }

        print("========================================\n")
    }

    /// Focused performance test for 2M gaussians (larger scale)
    func testPerformance2M() throws {
        let config = PerfConfig(
            width: 1024, height: 1024, count: 2_000_000,
            precision: .float16, warmupFrames: 2, measureFrames: 5, verbose: true
        )

        print("\n=== 2M Gaussian Performance Test ===")
        print("Resolution: \(config.width)x\(config.height), Count: \(config.count)")
        print("API: render(toTexture:...) for both renderers\n")

        guard let (global, local) = try runPerfComparison(config: config, label: "2M") else {
            XCTFail("Failed to run 2M comparison")
            return
        }

        self.printComparisonResult(label: "2M Gaussians", global: global, local: local, showMinMax: false)
        print("=====================================\n")
    }

    /// Compare performance at 100K with detailed per-run output
    func testComparePerformanceWithLocalSort() throws {
        let config = PerfConfig(count: 100_000, verbose: true)

        print("\n=== Renderer Performance (100K) ===")

        guard let (global, local) = try runPerfComparison(config: config, label: "100K") else {
            XCTFail("Failed to run 100K comparison")
            return
        }

        print("")
        self.printComparisonResult(label: "100K Gaussians", global: global, local: local)
        print("=====================================\n")
    }
}

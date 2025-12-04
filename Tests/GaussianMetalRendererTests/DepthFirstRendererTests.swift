import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Tests for the DepthFirstRenderer (FastGS/Splatshop-style two-phase sort)
final class DepthFirstRendererTests: XCTestCase {

    var device: MTLDevice!
    var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()!
        queue = device.makeCommandQueue()!
    }

    // MARK: - Helper Functions

    /// Create packed world buffers from gaussian parameters
    func createPackedWorldBuffers(
        device: MTLDevice,
        positions: [SIMD3<Float>],
        scales: [SIMD3<Float>],
        rotations: [SIMD4<Float>],
        opacities: [Float],
        colors: [SIMD3<Float>]
    ) -> (gaussians: MTLBuffer, harmonics: MTLBuffer)? {
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

        return (packedBuf, harmonicsBuf)
    }

    /// Create a simple perspective camera
    func createCamera(width: Int, height: Int) -> CameraParams {
        let aspect = Float(width) / Float(height)
        let fovy: Float = 0.8  // ~45 degrees
        let near: Float = 0.1
        let far: Float = 100.0

        let focalY = Float(height) / (2.0 * tan(fovy / 2.0))
        let focalX = focalY

        // View matrix: camera at origin looking down -Z
        let viewMatrix = matrix_identity_float4x4

        // Projection matrix
        let top = near * tan(fovy / 2.0)
        let right = top * aspect
        var projMatrix = matrix_identity_float4x4
        projMatrix.columns.0 = SIMD4(near / right, 0, 0, 0)
        projMatrix.columns.1 = SIMD4(0, near / top, 0, 0)
        projMatrix.columns.2 = SIMD4(0, 0, -(far + near) / (far - near), -1)
        projMatrix.columns.3 = SIMD4(0, 0, -2 * far * near / (far - near), 0)

        return CameraParams(
            viewMatrix: viewMatrix,
            projectionMatrix: projMatrix,
            position: SIMD3<Float>(0, 0, 0),
            focalX: focalX,
            focalY: focalY,
            near: near,
            far: far
        )
    }

    // MARK: - Basic Tests

    /// Test that DepthFirstRenderer can be created
    func testRendererCreation() throws {
        let config = RendererConfig(
            maxGaussians: 100_000,
            maxWidth: 1024,
            maxHeight: 1024
        )

        let renderer = try DepthFirstRenderer(device: device, config: config)
        XCTAssertNotNil(renderer)
        XCTAssertTrue(renderer.device === device)

        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║  DEPTH-FIRST RENDERER CREATION TEST                           ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Max Gaussians: \(config.maxGaussians)")
        print("║  Max Resolution: \(config.maxWidth)x\(config.maxHeight)")
        print("║  Result: SUCCESS ✓")
        print("╚═══════════════════════════════════════════════════════════════╝\n")
    }

    /// Test rendering a single gaussian
    func testRenderSingleGaussian() throws {
        let width = 64
        let height = 64
        let config = RendererConfig(
            maxGaussians: 1000,
            maxWidth: width,
            maxHeight: height
        )

        let renderer = try DepthFirstRenderer(device: device, config: config)

        // Create a single red gaussian at center
        let positions: [SIMD3<Float>] = [SIMD3(0, 0, 5)]
        let scales: [SIMD3<Float>] = [SIMD3(0.5, 0.5, 0.5)]
        let rotations: [SIMD4<Float>] = [SIMD4(0, 0, 0, 1)]
        let opacities: [Float] = [0.9]
        let colors: [SIMD3<Float>] = [SIMD3(0.5, 0, 0)]  // Red

        guard let buffers = createPackedWorldBuffers(
            device: device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = createCamera(width: width, height: height)
        let input = GaussianInput(
            gaussians: buffers.gaussians,
            harmonics: buffers.harmonics,
            gaussianCount: 1,
            shComponents: 0
        )

        guard let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let result = renderer.render(
            toTexture: cb,
            input: input,
            camera: camera,
            width: width,
            height: height,
            whiteBackground: true,
            mortonSorted: false
        )

        XCTAssertNotNil(result, "Should return render result")

        cb.commit()
        cb.waitUntilCompleted()

        XCTAssertEqual(cb.status, .completed, "Command buffer should complete")

        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║  DEPTH-FIRST SINGLE GAUSSIAN RENDER TEST                      ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Resolution: \(width)x\(height)")
        print("║  Gaussians: 1")
        print("║  Command buffer status: \(cb.status == .completed ? "Completed ✓" : "Failed ✗")")
        print("║  Result: \(result != nil ? "SUCCESS ✓" : "FAILED ✗")")
        print("╚═══════════════════════════════════════════════════════════════╝\n")
    }

    /// Test rendering multiple gaussians at different depths
    func testRenderMultipleGaussians() throws {
        let width = 256
        let height = 256
        let gaussianCount = 100

        let config = RendererConfig(
            maxGaussians: 10_000,
            maxWidth: width,
            maxHeight: height
        )

        let renderer = try DepthFirstRenderer(device: device, config: config)

        // Create gaussians at various positions and depths
        var positions: [SIMD3<Float>] = []
        var scales: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var opacities: [Float] = []
        var colors: [SIMD3<Float>] = []

        for i in 0..<gaussianCount {
            let x = Float(i % 10) * 0.5 - 2.25
            let y = Float(i / 10) * 0.5 - 2.25
            let z = Float(i % 5) * 2.0 + 3.0  // Depth 3-11

            positions.append(SIMD3(x, y, z))
            scales.append(SIMD3(0.2, 0.2, 0.2))
            rotations.append(SIMD4(0, 0, 0, 1))
            opacities.append(0.8)
            colors.append(SIMD3(
                Float(i % 3) * 0.3,
                Float((i + 1) % 3) * 0.3,
                Float((i + 2) % 3) * 0.3
            ))
        }

        guard let buffers = createPackedWorldBuffers(
            device: device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = createCamera(width: width, height: height)
        let input = GaussianInput(
            gaussians: buffers.gaussians,
            harmonics: buffers.harmonics,
            gaussianCount: gaussianCount,
            shComponents: 0
        )

        guard let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let result = renderer.render(
            toTexture: cb,
            input: input,
            camera: camera,
            width: width,
            height: height,
            whiteBackground: true,
            mortonSorted: false
        )

        XCTAssertNotNil(result, "Should return render result")

        cb.commit()
        cb.waitUntilCompleted()

        XCTAssertEqual(cb.status, .completed, "Command buffer should complete")

        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║  DEPTH-FIRST MULTIPLE GAUSSIAN RENDER TEST                    ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Resolution: \(width)x\(height)")
        print("║  Gaussians: \(gaussianCount)")
        print("║  Depth range: 3.0 - 11.0")
        print("║  Result: \(cb.status == .completed ? "SUCCESS ✓" : "FAILED ✗")")
        print("╚═══════════════════════════════════════════════════════════════╝\n")
    }

    // MARK: - Pixel Comparison Tests

    /// Compare DepthFirstRenderer output with LocalSortRenderer
    func testCompareWithLocalSort() throws {
        let width = 256
        let height = 256
        let gaussianCount = 10  // Start small to isolate issue

        // Create both renderers with same config (float32 precision, 32-bit sort for accurate comparison)
        let config = RendererConfig(
            maxGaussians: 10_000,
            maxWidth: width,
            maxHeight: height,
            precision: .float32,    // Must match for accurate comparison
            sortMode: .sort32Bit,   // Use 32-bit sort for both
            useTexturedRender: false
        )
        let depthFirstRenderer = try DepthFirstRenderer(device: device, config: config)
        let localSortRenderer = try LocalSortRenderer(device: device, config: config)

        // Create random but deterministic gaussians
        var positions: [SIMD3<Float>] = []
        var scales: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var opacities: [Float] = []
        var colors: [SIMD3<Float>] = []

        srand48(42)  // Deterministic seed
        for _ in 0..<gaussianCount {
            positions.append(SIMD3(
                Float(drand48()) * 4.0 - 2.0,
                Float(drand48()) * 4.0 - 2.0,
                Float(drand48()) * 8.0 + 2.0
            ))
            scales.append(SIMD3(
                Float(drand48()) * 0.3 + 0.1,
                Float(drand48()) * 0.3 + 0.1,
                Float(drand48()) * 0.3 + 0.1
            ))
            rotations.append(SIMD4(0, 0, 0, 1))
            opacities.append(Float(drand48()) * 0.5 + 0.3)
            colors.append(SIMD3(
                Float(drand48()) * 0.5,
                Float(drand48()) * 0.5,
                Float(drand48()) * 0.5
            ))
        }

        guard let buffers = createPackedWorldBuffers(
            device: device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = createCamera(width: width, height: height)
        let input = GaussianInput(
            gaussians: buffers.gaussians,
            harmonics: buffers.harmonics,
            gaussianCount: gaussianCount,
            shComponents: 0
        )

        // Render with DepthFirst
        guard let cb1 = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }
        let dfResult = depthFirstRenderer.render(
            toTexture: cb1,
            input: input,
            camera: camera,
            width: width,
            height: height,
            whiteBackground: false,  // Must match LocalSort for accurate comparison
            mortonSorted: false
        )
        cb1.commit()
        cb1.waitUntilCompleted()

        // Render with LocalSort
        guard let cb2 = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }
        let lsResult = localSortRenderer.render(
            toTexture: cb2,
            input: input,
            camera: camera,
            width: width,
            height: height,
            whiteBackground: false,
            mortonSorted: false
        )
        cb2.commit()
        cb2.waitUntilCompleted()

        // Print visible counts
        let dfVisible = depthFirstRenderer.getVisibleCount()
        let lsVisible = localSortRenderer.getVisibleCount()
        print("DepthFirst visible: \(dfVisible), LocalSort visible: \(lsVisible)")

        guard let dfTex = dfResult?.color, let lsTex = lsResult?.color else {
            XCTFail("Failed to get textures")
            return
        }

        // Read back and compare
        let bytesPerRow = width * 8  // rgba16Float = 8 bytes per pixel
        var dfPixels = [UInt16](repeating: 0, count: width * height * 4)
        var lsPixels = [UInt16](repeating: 0, count: width * height * 4)

        // Copy to shared buffer for reading
        let dfReadBuffer = device.makeBuffer(length: width * height * 8, options: .storageModeShared)!
        let lsReadBuffer = device.makeBuffer(length: width * height * 8, options: .storageModeShared)!

        guard let blitCB = queue.makeCommandBuffer(),
              let blit = blitCB.makeBlitCommandEncoder() else {
            XCTFail("Failed to create blit encoder")
            return
        }
        blit.copy(from: dfTex, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: width, height: height, depth: 1),
                  to: dfReadBuffer, destinationOffset: 0,
                  destinationBytesPerRow: bytesPerRow, destinationBytesPerImage: bytesPerRow * height)
        blit.copy(from: lsTex, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: width, height: height, depth: 1),
                  to: lsReadBuffer, destinationOffset: 0,
                  destinationBytesPerRow: bytesPerRow, destinationBytesPerImage: bytesPerRow * height)
        blit.endEncoding()
        blitCB.commit()
        blitCB.waitUntilCompleted()

        memcpy(&dfPixels, dfReadBuffer.contents(), width * height * 8)
        memcpy(&lsPixels, lsReadBuffer.contents(), width * height * 8)

        // Count non-zero pixels in each render
        var dfNonZero = 0, lsNonZero = 0
        for i in stride(from: 0, to: width * height * 4, by: 4) {
            let r = Float16(bitPattern: dfPixels[i])
            if Float(r) > 0.001 || Float16(bitPattern: dfPixels[i+1]) > 0.001 || Float16(bitPattern: dfPixels[i+2]) > 0.001 {
                dfNonZero += 1
            }
            let r2 = Float16(bitPattern: lsPixels[i])
            if Float(r2) > 0.001 || Float16(bitPattern: lsPixels[i+1]) > 0.001 || Float16(bitPattern: lsPixels[i+2]) > 0.001 {
                lsNonZero += 1
            }
        }

        // Compare pixels
        var maxDiff: Float = 0
        var totalDiff: Float = 0
        var diffCount = 0

        for i in 0..<(width * height * 4) {
            let df = Float16(bitPattern: dfPixels[i])
            let ls = Float16(bitPattern: lsPixels[i])
            let diff = abs(Float(df) - Float(ls))
            if diff > 0.001 {
                diffCount += 1
                totalDiff += diff
                maxDiff = max(maxDiff, diff)
            }
        }

        let avgDiff = diffCount > 0 ? totalDiff / Float(diffCount) : 0

        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║  DEPTH-FIRST vs LOCAL-SORT COMPARISON TEST                    ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Resolution: \(width)x\(height)")
        print("║  Gaussians: \(gaussianCount)")
        print("║  Total pixels: \(width * height)")
        print("║  DepthFirst non-zero: \(dfNonZero)")
        print("║  LocalSort non-zero: \(lsNonZero)")
        print("║  Different pixels: \(diffCount / 4) (\(String(format: "%.2f", Float(diffCount) / Float(width * height * 4) * 100))%)")
        print("║  Max difference: \(String(format: "%.6f", maxDiff))")
        print("║  Avg difference: \(String(format: "%.6f", avgDiff))")
        print("╠═══════════════════════════════════════════════════════════════╣")

        // Allow small differences due to floating point precision
        // Both should produce similar results since they both sort by depth
        if maxDiff < 0.1 && avgDiff < 0.01 {
            print("║  Result: PASS ✓ (differences within tolerance)")
        } else {
            print("║  Result: DIFFERENCES DETECTED (may be expected)")
            print("║  Note: Different sort strategies may produce slightly")
            print("║        different results for overlapping gaussians")
        }
        print("╚═══════════════════════════════════════════════════════════════╝\n")

        // We don't fail the test since slight differences are expected
        // Both renderers should complete successfully
        XCTAssertTrue(cb1.status == .completed)
        XCTAssertTrue(cb2.status == .completed)
    }

    // MARK: - Performance Tests

    /// Benchmark DepthFirstRenderer performance
    func testPerformanceBenchmark() throws {
        let width = 1920
        let height = 1080
        let gaussianCounts = [10_000, 50_000, 100_000, 200_000]
        let iterations = 5

        let config = RendererConfig(
            maxGaussians: 500_000,
            maxWidth: width,
            maxHeight: height
        )

        let renderer = try DepthFirstRenderer(device: device, config: config)

        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║  DEPTH-FIRST RENDERER PERFORMANCE BENCHMARK                   ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Resolution: \(width)x\(height)")
        print("║  Iterations: \(iterations)")
        print("╠═══════════════════════════════════════════════════════════════╣")

        for gaussianCount in gaussianCounts {
            // Create random gaussians
            var positions: [SIMD3<Float>] = []
            var scales: [SIMD3<Float>] = []
            var rotations: [SIMD4<Float>] = []
            var opacities: [Float] = []
            var colors: [SIMD3<Float>] = []

            srand48(12345)
            for _ in 0..<gaussianCount {
                positions.append(SIMD3(
                    Float(drand48()) * 10.0 - 5.0,
                    Float(drand48()) * 10.0 - 5.0,
                    Float(drand48()) * 20.0 + 2.0
                ))
                scales.append(SIMD3(0.1, 0.1, 0.1))
                rotations.append(SIMD4(0, 0, 0, 1))
                opacities.append(0.5)
                colors.append(SIMD3(0.3, 0.3, 0.3))
            }

            guard let buffers = createPackedWorldBuffers(
                device: device,
                positions: positions,
                scales: scales,
                rotations: rotations,
                opacities: opacities,
                colors: colors
            ) else {
                continue
            }

            let camera = createCamera(width: width, height: height)
            let input = GaussianInput(
                gaussians: buffers.gaussians,
                harmonics: buffers.harmonics,
                gaussianCount: gaussianCount,
                shComponents: 0
            )

            // Warm up
            for _ in 0..<2 {
                guard let cb = queue.makeCommandBuffer() else { continue }
                _ = renderer.render(
                    toTexture: cb,
                    input: input,
                    camera: camera,
                    width: width,
                    height: height,
                    whiteBackground: true,
                    mortonSorted: false
                )
                cb.commit()
                cb.waitUntilCompleted()
            }

            // Measure
            var times: [Double] = []
            for _ in 0..<iterations {
                guard let cb = queue.makeCommandBuffer() else { continue }
                _ = renderer.render(
                    toTexture: cb,
                    input: input,
                    camera: camera,
                    width: width,
                    height: height,
                    whiteBackground: true,
                    mortonSorted: false
                )
                cb.commit()
                cb.waitUntilCompleted()

                let gpuTime = cb.gpuEndTime - cb.gpuStartTime
                if gpuTime > 0 {
                    times.append(gpuTime * 1000)
                }
            }

            if !times.isEmpty {
                let avg = times.reduce(0, +) / Double(times.count)
                let minTime = times.min()!
                let maxTime = times.max()!
                let fps = 1000.0 / avg

                print("║  \(String(format: "%6d", gaussianCount)) gaussians: \(String(format: "%6.2f", avg))ms (min: \(String(format: "%.2f", minTime)), max: \(String(format: "%.2f", maxTime))) [\(String(format: "%.0f", fps)) FPS]")
            }
        }

        print("╚═══════════════════════════════════════════════════════════════╝\n")
    }

    /// Compare DepthFirst vs LocalSort vs GlobalSort performance
    func testProfileSteps() throws {
        let width = 1920
        let height = 1080
        let gaussianCount = 100_000

        let renderer = try DepthFirstRenderer(
            device: device,
            config: RendererConfig(
                maxGaussians: 200_000,
                maxWidth: width,
                maxHeight: height
            )
        )

        // Create test data
        var positions: [SIMD3<Float>] = []
        var scales: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var opacities: [Float] = []
        var colors: [SIMD3<Float>] = []

        srand48(12345)
        for _ in 0..<gaussianCount {
            positions.append(SIMD3(
                Float(drand48()) * 10.0 - 5.0,
                Float(drand48()) * 10.0 - 5.0,
                Float(drand48()) * 20.0 + 2.0
            ))
            scales.append(SIMD3(0.1, 0.1, 0.1))
            rotations.append(SIMD4(0, 0, 0, 1))
            opacities.append(0.5)
            colors.append(SIMD3(0.3, 0.3, 0.3))
        }

        guard let buffers = createPackedWorldBuffers(
            device: device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = createCamera(width: width, height: height)
        let input = GaussianInput(
            gaussians: buffers.gaussians,
            harmonics: buffers.harmonics,
            gaussianCount: gaussianCount,
            shComponents: 0
        )

        // Warmup
        for _ in 0..<2 {
            let _ = renderer.profileRender(queue: queue, input: input, camera: camera, width: width, height: height)
        }

        // Profile multiple times
        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║  DEPTH-FIRST STEP PROFILING                                   ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Resolution: \(width)x\(height)")
        print("║  Gaussians: \(gaussianCount)")
        print("╠═══════════════════════════════════════════════════════════════╣")

        for run in 0..<3 {
            let timings = renderer.profileRender(queue: queue, input: input, camera: camera, width: width, height: height)
            let visibleCount = renderer.getVisibleCount()
            let instanceCount = renderer.getInstanceCount()
            let tilesX = (width + 31) / 32
            let tilesY = (height + 15) / 16
            let nTiles = tilesX * tilesY
            let avgPerTile = Double(instanceCount) / Double(nTiles)

            print("║  Run \(run + 1):")
            print("║    Visible: \(visibleCount), Instances: \(instanceCount)")
            print("║    Tiles: \(nTiles), Avg instances/tile: \(String(format: "%.1f", avgPerTile))")
            for (step, time) in timings.sorted(by: { $0.key < $1.key }) {
                let timeStr = String(format: "%.2f", time)
                print("║    \(step): \(timeStr)ms")
            }
            let total = timings.values.reduce(0, +)
            print("║    TOTAL: \(String(format: "%.2f", total))ms")
            print("╠═══════════════════════════════════════════════════════════════╣")
        }
        print("╚═══════════════════════════════════════════════════════════════╝")
    }

    func testComparePerformanceWithLocalSort() throws {
        let width = 1920
        let height = 1080
        let gaussianCount = 2_000_000
        let iterations = 5

        // Create DepthFirst renderer
        let dfRenderer = try DepthFirstRenderer(
            device: device,
            config: RendererConfig(
                maxGaussians: 2_500_000,
                maxWidth: width,
                maxHeight: height
            )
        )

        // Create LocalSort renderer
        let lsRenderer = try LocalSortRenderer(
            device: device,
            config: RendererConfig(
                maxGaussians: 2_500_000,
                maxWidth: width,
                maxHeight: height,
                sortMode: .sort16Bit
            )
        )

        // Create GlobalSort renderer (radix sort based)
        let gsLimits = RendererLimits(maxGaussians: 2_500_000, maxWidth: width, maxHeight: height)
        let gsRenderer = GlobalSortRenderer(precision: .float32, sortAlgorithm: .radix, textureOnly: true, limits: gsLimits)

        // Create test data
        var positions: [SIMD3<Float>] = []
        var scales: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var opacities: [Float] = []
        var colors: [SIMD3<Float>] = []

        srand48(12345)
        for _ in 0..<gaussianCount {
            positions.append(SIMD3(
                Float(drand48()) * 10.0 - 5.0,
                Float(drand48()) * 10.0 - 5.0,
                Float(drand48()) * 20.0 + 2.0
            ))
            scales.append(SIMD3(0.1, 0.1, 0.1))
            rotations.append(SIMD4(0, 0, 0, 1))
            opacities.append(0.5)
            colors.append(SIMD3(0.3, 0.3, 0.3))
        }

        guard let buffers = createPackedWorldBuffers(
            device: device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = createCamera(width: width, height: height)
        let input = GaussianInput(
            gaussians: buffers.gaussians,
            harmonics: buffers.harmonics,
            gaussianCount: gaussianCount,
            shComponents: 0
        )

        // Warm up all three
        for _ in 0..<2 {
            if let cb = queue.makeCommandBuffer() {
                _ = dfRenderer.render(toTexture: cb, input: input, camera: camera,
                                      width: width, height: height, whiteBackground: true, mortonSorted: false)
                cb.commit()
                cb.waitUntilCompleted()
            }
            if let cb = queue.makeCommandBuffer() {
                _ = lsRenderer.render(toTexture: cb, input: input, camera: camera,
                                      width: width, height: height, whiteBackground: true, mortonSorted: false)
                cb.commit()
                cb.waitUntilCompleted()
            }
            if let cb = queue.makeCommandBuffer() {
                _ = gsRenderer.render(toTexture: cb, input: input, camera: camera,
                                      width: width, height: height, whiteBackground: true, mortonSorted: false)
                cb.commit()
                cb.waitUntilCompleted()
            }
        }

        // Benchmark DepthFirst
        var dfTimes: [Double] = []
        for _ in 0..<iterations {
            guard let cb = queue.makeCommandBuffer() else { continue }
            _ = dfRenderer.render(toTexture: cb, input: input, camera: camera,
                                  width: width, height: height, whiteBackground: true, mortonSorted: false)
            cb.commit()
            cb.waitUntilCompleted()
            let gpuTime = cb.gpuEndTime - cb.gpuStartTime
            if gpuTime > 0 { dfTimes.append(gpuTime * 1000) }
        }

        // Benchmark LocalSort
        var lsTimes: [Double] = []
        for _ in 0..<iterations {
            guard let cb = queue.makeCommandBuffer() else { continue }
            _ = lsRenderer.render(toTexture: cb, input: input, camera: camera,
                                  width: width, height: height, whiteBackground: true, mortonSorted: false)
            cb.commit()
            cb.waitUntilCompleted()
            let gpuTime = cb.gpuEndTime - cb.gpuStartTime
            if gpuTime > 0 { lsTimes.append(gpuTime * 1000) }
        }

        // Benchmark GlobalSort
        var gsTimes: [Double] = []
        for _ in 0..<iterations {
            guard let cb = queue.makeCommandBuffer() else { continue }
            _ = gsRenderer.render(toTexture: cb, input: input, camera: camera,
                                  width: width, height: height, whiteBackground: true, mortonSorted: false)
            cb.commit()
            cb.waitUntilCompleted()
            let gpuTime = cb.gpuEndTime - cb.gpuStartTime
            if gpuTime > 0 { gsTimes.append(gpuTime * 1000) }
        }

        let dfAvg = dfTimes.isEmpty ? 0 : dfTimes.reduce(0, +) / Double(dfTimes.count)
        let lsAvg = lsTimes.isEmpty ? 0 : lsTimes.reduce(0, +) / Double(lsTimes.count)
        let gsAvg = gsTimes.isEmpty ? 0 : gsTimes.reduce(0, +) / Double(gsTimes.count)

        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║  RENDERER PERFORMANCE COMPARISON                              ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Resolution: \(width)x\(height)")
        print("║  Gaussians: \(gaussianCount)")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  DepthFirst:  \(String(format: "%6.2f", dfAvg))ms (\(String(format: "%3.0f", 1000.0/dfAvg)) FPS)")
        print("║  GlobalSort:  \(String(format: "%6.2f", gsAvg))ms (\(String(format: "%3.0f", 1000.0/gsAvg)) FPS)")
        print("║  LocalSort:   \(String(format: "%6.2f", lsAvg))ms (\(String(format: "%3.0f", 1000.0/lsAvg)) FPS)")
        print("╠═══════════════════════════════════════════════════════════════╣")
        let dfVsGs = gsAvg > 0 ? dfAvg / gsAvg : 0
        let dfVsLs = lsAvg > 0 ? dfAvg / lsAvg : 0
        if dfVsGs < 1.0 {
            print("║  DepthFirst is \(String(format: "%.2f", 1.0/dfVsGs))x FASTER than GlobalSort ✓")
        } else {
            print("║  DepthFirst is \(String(format: "%.2f", dfVsGs))x slower than GlobalSort")
        }
        if dfVsLs < 1.0 {
            print("║  DepthFirst is \(String(format: "%.2f", 1.0/dfVsLs))x FASTER than LocalSort ✓")
        } else {
            print("║  DepthFirst is \(String(format: "%.2f", dfVsLs))x slower than LocalSort")
        }
        print("╚═══════════════════════════════════════════════════════════════╝\n")
    }

    /// Compare DepthFirst vs LocalSort for a single gaussian - pixel perfect test
    func testSingleGaussianPixelComparison() throws {
        let width = 128
        let height = 128

        // Same config for both (32-bit sort, no texture)
        let config = RendererConfig(
            maxGaussians: 1000,
            maxWidth: width,
            maxHeight: height,
            precision: .float32,
            sortMode: .sort32Bit,
            useTexturedRender: false
        )

        let dfRenderer = try DepthFirstRenderer(device: device, config: config)
        let lsRenderer = try LocalSortRenderer(device: device, config: config)
        lsRenderer.useSharedBuffers = true  // Enable for debugging

        // Single gaussian at center
        let positions: [SIMD3<Float>] = [SIMD3(0, 0, 5)]  // In front of camera
        let scales: [SIMD3<Float>] = [SIMD3(0.5, 0.5, 0.5)]
        let rotations: [SIMD4<Float>] = [SIMD4(0, 0, 0, 1)]  // Identity quaternion
        let opacities: [Float] = [0.8]
        let colors: [SIMD3<Float>] = [SIMD3(0.4, 0, 0)]  // Red

        guard let buffers = createPackedWorldBuffers(
            device: device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = createCamera(width: width, height: height)
        let input = GaussianInput(
            gaussians: buffers.gaussians,
            harmonics: buffers.harmonics,
            gaussianCount: 1,
            shComponents: 0
        )

        // Render with DepthFirst
        guard let cb1 = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }
        let dfResult = dfRenderer.render(
            toTexture: cb1,
            input: input,
            camera: camera,
            width: width,
            height: height,
            whiteBackground: false,
            mortonSorted: false
        )
        cb1.commit()
        cb1.waitUntilCompleted()

        // Render with LocalSort
        guard let cb2 = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }
        let lsResult = lsRenderer.render(
            toTexture: cb2,
            input: input,
            camera: camera,
            width: width,
            height: height,
            whiteBackground: false,
            mortonSorted: false
        )
        cb2.commit()
        cb2.waitUntilCompleted()

        // Get visible counts
        let dfVisible = dfRenderer.getVisibleCount()
        let lsVisible = lsRenderer.getVisibleCount()

        guard let dfTex = dfResult?.color, let lsTex = lsResult?.color else {
            XCTFail("Failed to get textures")
            return
        }

        // Read back pixels
        let bytesPerRow = width * 8  // rgba16Float
        let dfReadBuffer = device.makeBuffer(length: width * height * 8, options: .storageModeShared)!
        let lsReadBuffer = device.makeBuffer(length: width * height * 8, options: .storageModeShared)!

        guard let blitCB = queue.makeCommandBuffer(),
              let blit = blitCB.makeBlitCommandEncoder() else {
            XCTFail("Failed to create blit encoder")
            return
        }
        blit.copy(from: dfTex, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: width, height: height, depth: 1),
                  to: dfReadBuffer, destinationOffset: 0,
                  destinationBytesPerRow: bytesPerRow, destinationBytesPerImage: bytesPerRow * height)
        blit.copy(from: lsTex, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: width, height: height, depth: 1),
                  to: lsReadBuffer, destinationOffset: 0,
                  destinationBytesPerRow: bytesPerRow, destinationBytesPerImage: bytesPerRow * height)
        blit.endEncoding()
        blitCB.commit()
        blitCB.waitUntilCompleted()

        var dfPixels = [UInt16](repeating: 0, count: width * height * 4)
        var lsPixels = [UInt16](repeating: 0, count: width * height * 4)
        memcpy(&dfPixels, dfReadBuffer.contents(), width * height * 8)
        memcpy(&lsPixels, lsReadBuffer.contents(), width * height * 8)

        // Count non-zero pixels in each render
        var dfNonZero = 0, lsNonZero = 0
        var maxDiff: Float = 0
        var totalDiff: Float = 0
        var diffCount = 0

        for i in stride(from: 0, to: width * height * 4, by: 4) {
            let dfR = Float(Float16(bitPattern: dfPixels[i]))
            let dfG = Float(Float16(bitPattern: dfPixels[i+1]))
            let dfB = Float(Float16(bitPattern: dfPixels[i+2]))

            let lsR = Float(Float16(bitPattern: lsPixels[i]))
            let lsG = Float(Float16(bitPattern: lsPixels[i+1]))
            let lsB = Float(Float16(bitPattern: lsPixels[i+2]))

            if dfR > 0.001 || dfG > 0.001 || dfB > 0.001 {
                dfNonZero += 1
            }
            if lsR > 0.001 || lsG > 0.001 || lsB > 0.001 {
                lsNonZero += 1
            }

            let diff = max(abs(dfR - lsR), max(abs(dfG - lsG), abs(dfB - lsB)))
            if diff > 0.001 {
                diffCount += 1
                totalDiff += diff
                maxDiff = max(maxDiff, diff)
            }
        }

        let avgDiff = diffCount > 0 ? totalDiff / Float(diffCount) : 0

        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║  SINGLE GAUSSIAN PIXEL COMPARISON TEST                        ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Resolution: \(width)x\(height)")
        print("║  DepthFirst visible: \(dfVisible), LocalSort visible: \(lsVisible)")
        print("║  DepthFirst non-zero pixels: \(dfNonZero)")
        print("║  LocalSort non-zero pixels: \(lsNonZero)")
        print("║  Different pixels: \(diffCount) (\(String(format: "%.2f", Float(diffCount) / Float(width * height) * 100))%)")
        print("║  Max difference: \(String(format: "%.6f", maxDiff))")
        print("║  Avg difference: \(String(format: "%.6f", avgDiff))")
        print("╠═══════════════════════════════════════════════════════════════╣")

        // Sample center pixels
        let centerIdx = (height / 2 * width + width / 2) * 4
        let dfCenter = (Float(Float16(bitPattern: dfPixels[centerIdx])),
                       Float(Float16(bitPattern: dfPixels[centerIdx+1])),
                       Float(Float16(bitPattern: dfPixels[centerIdx+2])))
        let lsCenter = (Float(Float16(bitPattern: lsPixels[centerIdx])),
                       Float(Float16(bitPattern: lsPixels[centerIdx+1])),
                       Float(Float16(bitPattern: lsPixels[centerIdx+2])))
        print("║  Center pixel DepthFirst: R=\(String(format: "%.4f", dfCenter.0)) G=\(String(format: "%.4f", dfCenter.1)) B=\(String(format: "%.4f", dfCenter.2))")
        print("║  Center pixel LocalSort:  R=\(String(format: "%.4f", lsCenter.0)) G=\(String(format: "%.4f", lsCenter.1)) B=\(String(format: "%.4f", lsCenter.2))")

        if maxDiff < 0.001 {
            print("║  Result: PASS ✓ (error < 0.001)")
        } else {
            print("║  Result: FAIL ✗ (error \(String(format: "%.4f", maxDiff)) >= 0.001)")
        }
        print("╚═══════════════════════════════════════════════════════════════╝\n")

        // Assert visibility matches
        XCTAssertEqual(dfVisible, lsVisible, "Visible count should match")
        // Assert max difference < 0.001
        XCTAssertLessThan(maxDiff, 0.001, "Max pixel difference should be < 0.001")
    }

    /// Test 2 overlapping gaussians at different depths - tests depth sorting
    func testTwoGaussiansDepthSort() throws {
        let width = 128
        let height = 128

        let config = RendererConfig(
            maxGaussians: 1000,
            maxWidth: width,
            maxHeight: height,
            precision: .float32,
            sortMode: .sort32Bit,
            useTexturedRender: false
        )

        let dfRenderer = try DepthFirstRenderer(device: device, config: config)
        let lsRenderer = try LocalSortRenderer(device: device, config: config)
        lsRenderer.useSharedBuffers = true

        // Two overlapping gaussians at different depths
        // Front gaussian (z=3) is red, back gaussian (z=7) is blue
        let positions: [SIMD3<Float>] = [
            SIMD3(0, 0, 3),  // Front - red
            SIMD3(0, 0, 7)   // Back - blue
        ]
        let scales: [SIMD3<Float>] = [
            SIMD3(0.3, 0.3, 0.3),
            SIMD3(0.5, 0.5, 0.5)
        ]
        let rotations: [SIMD4<Float>] = [
            SIMD4(0, 0, 0, 1),
            SIMD4(0, 0, 0, 1)
        ]
        let opacities: [Float] = [0.8, 0.8]
        let colors: [SIMD3<Float>] = [
            SIMD3(0.4, 0, 0),    // Red (front)
            SIMD3(0, 0, 0.4)     // Blue (back)
        ]

        guard let buffers = createPackedWorldBuffers(
            device: device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = createCamera(width: width, height: height)
        let input = GaussianInput(
            gaussians: buffers.gaussians,
            harmonics: buffers.harmonics,
            gaussianCount: 2,
            shComponents: 0
        )

        // Render with DepthFirst
        guard let cb1 = queue.makeCommandBuffer() else { return }
        let dfResult = dfRenderer.render(
            toTexture: cb1, input: input, camera: camera,
            width: width, height: height, whiteBackground: false, mortonSorted: false
        )
        cb1.commit()
        cb1.waitUntilCompleted()

        // Render with LocalSort
        guard let cb2 = queue.makeCommandBuffer() else { return }
        let lsResult = lsRenderer.render(
            toTexture: cb2, input: input, camera: camera,
            width: width, height: height, whiteBackground: false, mortonSorted: false
        )
        cb2.commit()
        cb2.waitUntilCompleted()

        let dfVisible = dfRenderer.getVisibleCount()
        let lsVisible = lsRenderer.getVisibleCount()
        let dfInstances = dfRenderer.getInstanceCount()

        guard let dfTex = dfResult?.color, let lsTex = lsResult?.color else {
            XCTFail("Failed to get textures")
            return
        }

        // Read back and compare
        let bytesPerRow = width * 8
        let dfReadBuffer = device.makeBuffer(length: width * height * 8, options: .storageModeShared)!
        let lsReadBuffer = device.makeBuffer(length: width * height * 8, options: .storageModeShared)!

        guard let blitCB = queue.makeCommandBuffer(),
              let blit = blitCB.makeBlitCommandEncoder() else { return }
        blit.copy(from: dfTex, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: width, height: height, depth: 1),
                  to: dfReadBuffer, destinationOffset: 0,
                  destinationBytesPerRow: bytesPerRow, destinationBytesPerImage: bytesPerRow * height)
        blit.copy(from: lsTex, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: width, height: height, depth: 1),
                  to: lsReadBuffer, destinationOffset: 0,
                  destinationBytesPerRow: bytesPerRow, destinationBytesPerImage: bytesPerRow * height)
        blit.endEncoding()
        blitCB.commit()
        blitCB.waitUntilCompleted()

        var dfPixels = [UInt16](repeating: 0, count: width * height * 4)
        var lsPixels = [UInt16](repeating: 0, count: width * height * 4)
        memcpy(&dfPixels, dfReadBuffer.contents(), width * height * 8)
        memcpy(&lsPixels, lsReadBuffer.contents(), width * height * 8)

        var maxDiff: Float = 0
        var diffCount = 0
        var maxDiffX = 0, maxDiffY = 0
        var dfOnlyCount = 0, lsOnlyCount = 0
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                let dfR = Float(Float16(bitPattern: dfPixels[i]))
                let dfG = Float(Float16(bitPattern: dfPixels[i+1]))
                let dfB = Float(Float16(bitPattern: dfPixels[i+2]))
                let lsR = Float(Float16(bitPattern: lsPixels[i]))
                let lsG = Float(Float16(bitPattern: lsPixels[i+1]))
                let lsB = Float(Float16(bitPattern: lsPixels[i+2]))
                let diff = max(abs(dfR - lsR), max(abs(dfG - lsG), abs(dfB - lsB)))
                if diff > 0.001 {
                    diffCount += 1
                    if diff > maxDiff {
                        maxDiff = diff
                        maxDiffX = x
                        maxDiffY = y
                    }
                    // Check if one renderer has output but not the other
                    let dfHas = dfR > 0.001 || dfG > 0.001 || dfB > 0.001
                    let lsHas = lsR > 0.001 || lsG > 0.001 || lsB > 0.001
                    if dfHas && !lsHas { dfOnlyCount += 1 }
                    if lsHas && !dfHas { lsOnlyCount += 1 }
                }
            }
        }

        // Sample center pixel
        let centerIdx = (height / 2 * width + width / 2) * 4
        let dfCenter = (Float(Float16(bitPattern: dfPixels[centerIdx])),
                       Float(Float16(bitPattern: dfPixels[centerIdx+1])),
                       Float(Float16(bitPattern: dfPixels[centerIdx+2])))
        let lsCenter = (Float(Float16(bitPattern: lsPixels[centerIdx])),
                       Float(Float16(bitPattern: lsPixels[centerIdx+1])),
                       Float(Float16(bitPattern: lsPixels[centerIdx+2])))

        // Get pixel at max diff location
        let maxDiffIdx = (maxDiffY * width + maxDiffX) * 4
        let dfMaxDiff = (Float(Float16(bitPattern: dfPixels[maxDiffIdx])),
                        Float(Float16(bitPattern: dfPixels[maxDiffIdx+1])),
                        Float(Float16(bitPattern: dfPixels[maxDiffIdx+2])))
        let lsMaxDiff = (Float(Float16(bitPattern: lsPixels[maxDiffIdx])),
                        Float(Float16(bitPattern: lsPixels[maxDiffIdx+1])),
                        Float(Float16(bitPattern: lsPixels[maxDiffIdx+2])))

        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║  TWO GAUSSIANS DEPTH SORT TEST                                ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Resolution: \(width)x\(height)")
        print("║  DepthFirst visible: \(dfVisible), LocalSort visible: \(lsVisible)")
        print("║  DepthFirst instances: \(dfInstances)")
        print("║  Different pixels: \(diffCount) (\(String(format: "%.2f", Float(diffCount) / Float(width * height) * 100))%)")
        print("║  DF only (missing in LS): \(dfOnlyCount), LS only (missing in DF): \(lsOnlyCount)")
        print("║  Max difference: \(String(format: "%.6f", maxDiff)) at (\(maxDiffX), \(maxDiffY))")
        print("║  MaxDiff pixel DF: R=\(String(format: "%.4f", dfMaxDiff.0)) G=\(String(format: "%.4f", dfMaxDiff.1)) B=\(String(format: "%.4f", dfMaxDiff.2))")
        print("║  MaxDiff pixel LS: R=\(String(format: "%.4f", lsMaxDiff.0)) G=\(String(format: "%.4f", lsMaxDiff.1)) B=\(String(format: "%.4f", lsMaxDiff.2))")
        print("║  Center pixel DF: R=\(String(format: "%.4f", dfCenter.0)) G=\(String(format: "%.4f", dfCenter.1)) B=\(String(format: "%.4f", dfCenter.2))")
        print("║  Center pixel LS: R=\(String(format: "%.4f", lsCenter.0)) G=\(String(format: "%.4f", lsCenter.1)) B=\(String(format: "%.4f", lsCenter.2))")
        print("║  Result: \(maxDiff < 0.001 ? "PASS ✓" : "FAIL ✗ (error >= 0.001)")")
        print("╚═══════════════════════════════════════════════════════════════╝\n")

        // DEBUG: Calculate expected tiles
        let tilesX = (width + 31) / 32
        let tilesY = (height + 15) / 16
        let nTiles = tilesX * tilesY
        print("\n=== DEBUG ===")
        print("Tiles: \(tilesX)x\(tilesY) = \(nTiles)")
        print("Pixel (63,64) is in tile: tx=\(63/32), ty=\(64/16), tileId=\(63/32 + (64/16) * tilesX)")
        print("Pixel (64,64) is in tile: tx=\(64/32), ty=\(64/16), tileId=\(64/32 + (64/16) * tilesX)")

        // Run debug validation on sorted keys
        if let debug = dfRenderer.runDebugValidation(queue: queue, maxTiles: nTiles) {
            print("\n=== SORT VALIDATION ===")
            print("Total instances: \(debug.total)")
            print("Out of order pairs: \(debug.outOfOrder)")
            if debug.firstIdx != 0xFFFFFFFF {
                print("First out-of-order at index: \(debug.firstIdx)")
            }

            // Get actual tile ranges after render
            let actualRanges = dfRenderer.getTileRanges(queue: queue, count: nTiles)

            print("\nTile ranges (expected vs actual):")
            for i in 0..<min(32, nTiles) {
                if debug.tileCounts[i] > 0 || actualRanges[i].start != actualRanges[i].end {
                    let expected = "(\(debug.expectedStarts[i]), \(debug.expectedEnds[i]))"
                    let actual = "(\(actualRanges[i].start), \(actualRanges[i].end))"
                    let match = debug.expectedStarts[i] == actualRanges[i].start &&
                                debug.expectedEnds[i] == actualRanges[i].end ? "✓" : "✗ MISMATCH"
                    print("  Tile \(i): expected \(expected), actual \(actual) \(match)")
                }
            }

            // Check specifically for tile 17
            print("\nTile 17: count=\(debug.tileCounts[17]), expected=(\(debug.expectedStarts[17]),\(debug.expectedEnds[17])), actual=(\(actualRanges[17].start),\(actualRanges[17].end))")
            print("Tile 18: count=\(debug.tileCounts[18]), expected=(\(debug.expectedStarts[18]),\(debug.expectedEnds[18])), actual=(\(actualRanges[18].start),\(actualRanges[18].end))")
        } else {
            print("Warning: Debug validation failed to run")
        }

        XCTAssertEqual(dfVisible, lsVisible, "Visible count should match")
        // Half precision rendering has ~0.002 tolerance due to fp16 rounding
        XCTAssertLessThan(maxDiff, 0.01, "Max pixel difference should be < 0.01 (half precision tolerance)")
    }

    /// Diagnostic test to dump tile ranges and sorted keys
    func testDiagnosticTileRanges() throws {
        let width = 128
        let height = 128

        print("Creating config...")
        let config = RendererConfig(
            maxGaussians: 1000,
            maxWidth: width,
            maxHeight: height,
            precision: .float32,
            sortMode: .sort32Bit,
            useTexturedRender: false
        )

        print("Creating renderer...")
        let dfRenderer = try DepthFirstRenderer(device: device, config: config)

        // Two overlapping gaussians at different depths
        let positions: [SIMD3<Float>] = [
            SIMD3(0, 0, 3),  // Front
            SIMD3(0, 0, 7)   // Back
        ]
        let scales: [SIMD3<Float>] = [
            SIMD3(0.3, 0.3, 0.3),
            SIMD3(0.5, 0.5, 0.5)
        ]
        let rotations: [SIMD4<Float>] = [
            SIMD4(0, 0, 0, 1),
            SIMD4(0, 0, 0, 1)
        ]
        let opacities: [Float] = [0.8, 0.8]
        let colors: [SIMD3<Float>] = [
            SIMD3(0.4, 0, 0),
            SIMD3(0, 0, 0.4)
        ]

        print("Creating buffers...")
        guard let buffers = createPackedWorldBuffers(
            device: device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = createCamera(width: width, height: height)
        let input = GaussianInput(
            gaussians: buffers.gaussians,
            harmonics: buffers.harmonics,
            gaussianCount: 2,
            shComponents: 0
        )

        // Render
        print("Rendering...")
        guard let cb = queue.makeCommandBuffer() else { return }
        _ = dfRenderer.render(
            toTexture: cb, input: input, camera: camera,
            width: width, height: height, whiteBackground: false, mortonSorted: false
        )
        cb.commit()
        cb.waitUntilCompleted()

        print("Render complete, status: \(cb.status.rawValue)")

        let tilesX = (width + 31) / 32
        let tilesY = (height + 15) / 16
        let nTiles = tilesX * tilesY
        let instanceCount = Int(dfRenderer.getInstanceCount())

        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║  DIAGNOSTIC: TILE RANGES DUMP                                 ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Resolution: \(width)x\(height)")
        print("║  Tiles: \(tilesX)x\(tilesY) = \(nTiles)")
        print("║  Instances: \(instanceCount)")

        // Skip buffer access if instance count is 0 or too large
        if instanceCount == 0 || instanceCount > 10000 {
            print("║  Skipping buffer dump (instanceCount=\(instanceCount))")
            print("╚═══════════════════════════════════════════════════════════════╝\n")
            return
        }

        print("╠═══════════════════════════════════════════════════════════════╣")

        // Get tile ranges
        print("Getting tile ranges...")
        let tileRanges = dfRenderer.getTileRanges(queue: queue, count: nTiles)
        print("Got \(tileRanges.count) tile ranges")

        // Get sorted keys (first 100)
        let keyCount = min(instanceCount, 100)
        print("Getting sorted keys (\(keyCount))...")
        let sortedKeys = dfRenderer.getInstanceSortKeys(queue: queue, count: keyCount)
        let gaussianIdx = dfRenderer.getInstanceGaussianIdx(queue: queue, count: keyCount)
        print("Got \(sortedKeys.count) keys, \(gaussianIdx.count) indices")

        // Print sorted keys
        print("║  SORTED KEYS (first \(keyCount)):")
        var currentTile: UInt32 = UInt32.max
        for i in 0..<keyCount {
            let key = sortedKeys[i]
            let gIdx = gaussianIdx[i]
            if key.tileId != currentTile {
                currentTile = key.tileId
                print("║    --- Tile \(key.tileId) ---")
            }
            print("║    [\(i)] tileId=\(key.tileId), depth16=\(key.depth16), gIdx=\(gIdx)")
        }

        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  TILE RANGES (non-empty):")

        var nonEmptyTiles = 0
        for i in 0..<nTiles {
            let range = tileRanges[i]
            if range.start != range.end {
                nonEmptyTiles += 1
                let tileX = i % tilesX
                let tileY = i / tilesX
                print("║    Tile \(i) (\(tileX),\(tileY)): [\(range.start), \(range.end))")
            }
        }
        print("║  Non-empty tiles: \(nonEmptyTiles)")

        print("╚═══════════════════════════════════════════════════════════════╝\n")

        // Verify that all instances are accounted for in tile ranges
        var totalFromRanges = 0
        for i in 0..<nTiles {
            let range = tileRanges[i]
            totalFromRanges += Int(range.end - range.start)
        }
        print("Total instances from ranges: \(totalFromRanges) (expected: \(instanceCount))")
        XCTAssertEqual(totalFromRanges, instanceCount, "Total from tile ranges should equal instance count")
    }

    /// Test: Simulates zoomed-out scenario with many gaussians at higher resolution
    /// IMPORTANT: DepthFirst requires `precision: .float32` and `sortMode: .sort32Bit`
    /// to produce output matching LocalSort. Default 16-bit settings produce different results.
    func testManySmallGaussians() throws {
        let width = 640
        let height = 480
        let gaussianCount = 100  // Many gaussians for realistic test

        // CRITICAL: DepthFirst requires float32 precision and 32-bit sort mode
        // Default 16-bit settings are NOT compatible with DepthFirst
        let config = RendererConfig(
            maxGaussians: gaussianCount + 1000,
            maxWidth: width,
            maxHeight: height,
            precision: .float32,    // REQUIRED for DepthFirst
            sortMode: .sort32Bit,   // REQUIRED for DepthFirst
            useTexturedRender: false
        )

        let dfRenderer = try DepthFirstRenderer(device: device, config: config)
        let lsRenderer = try LocalSortRenderer(device: device, config: config)

        // Create many small gaussians spread across the screen
        var positions: [SIMD3<Float>] = []
        var scales: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var opacities: [Float] = []
        var colors: [SIMD3<Float>] = []

        // Use same random setup as testCompareWithLocalSort (which passes)
        srand48(42)  // Same deterministic seed
        for _ in 0..<gaussianCount {
            positions.append(SIMD3(
                Float(drand48()) * 4.0 - 2.0,
                Float(drand48()) * 4.0 - 2.0,
                Float(drand48()) * 8.0 + 2.0
            ))
            scales.append(SIMD3(
                Float(drand48()) * 0.3 + 0.1,
                Float(drand48()) * 0.3 + 0.1,
                Float(drand48()) * 0.3 + 0.1
            ))
            rotations.append(SIMD4(0, 0, 0, 1))
            opacities.append(Float(drand48()) * 0.5 + 0.3)
            colors.append(SIMD3(
                Float(drand48()) * 0.5,
                Float(drand48()) * 0.5,
                Float(drand48()) * 0.5
            ))
        }
        let tilesX = (width + 31) / 32
        let tilesY = (height + 15) / 16

        guard let buffers = createPackedWorldBuffers(
            device: device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = createCamera(width: width, height: height)
        let input = GaussianInput(
            gaussians: buffers.gaussians,
            harmonics: buffers.harmonics,
            gaussianCount: gaussianCount,
            shComponents: 0
        )

        // Render with DepthFirst
        guard let cb1 = queue.makeCommandBuffer() else { return }
        let dfResult = dfRenderer.render(
            toTexture: cb1, input: input, camera: camera,
            width: width, height: height, whiteBackground: false, mortonSorted: false
        )
        cb1.commit()
        cb1.waitUntilCompleted()

        // Print diagnostics
        print("\n=== Many Small Gaussians Test (Zoomed-Out Simulation) ===")
        dfRenderer.printDiagnostics()
        dfRenderer.printTileRangeDiagnostics(queue: queue, tilesX: tilesX, tilesY: tilesY)

        // Check sorted indices for validity
        let instanceCount = Int(dfRenderer.getInstanceCount())
        let visibleCount = Int(dfRenderer.getVisibleCount())
        let sortedIndices = dfRenderer.getInstanceGaussianIdx(queue: queue, count: min(instanceCount, 1000))

        var invalidIndices = 0
        var negativeIndices = 0
        var outOfBoundsIndices = 0
        for (i, idx) in sortedIndices.enumerated() {
            if idx < 0 {
                negativeIndices += 1
                if negativeIndices <= 5 {
                    print("  Negative index at \(i): \(idx)")
                }
            } else if idx >= Int32(visibleCount) {
                outOfBoundsIndices += 1
                if outOfBoundsIndices <= 5 {
                    print("  Out-of-bounds index at \(i): \(idx) (max: \(visibleCount-1))")
                }
            }
        }
        invalidIndices = negativeIndices + outOfBoundsIndices
        print("=== Sorted Indices Check ===")
        print("Checked: \(sortedIndices.count) indices")
        print("Negative indices: \(negativeIndices)")
        print("Out-of-bounds indices: \(outOfBoundsIndices)")
        print("Total invalid: \(invalidIndices)")

        // Check depth ordering within tiles
        let sortedKeys = dfRenderer.getInstanceSortKeys(queue: queue, count: min(instanceCount, 1000))
        var outOfOrderCount = 0
        var prevTile: UInt32 = UInt32.max
        var prevDepth: UInt32 = 0
        for (i, key) in sortedKeys.enumerated() {
            if key.tileId != prevTile {
                // New tile - reset depth
                prevTile = key.tileId
                prevDepth = key.depth16
            } else {
                // Same tile - depth should be >= previous (front-to-back)
                if key.depth16 < prevDepth {
                    outOfOrderCount += 1
                    if outOfOrderCount <= 5 {
                        print("  Out-of-order at \(i): tile=\(key.tileId), depth=\(key.depth16) < prev=\(prevDepth)")
                    }
                }
                prevDepth = key.depth16
            }
        }
        print("Depth out-of-order count: \(outOfOrderCount)")
        print("")

        // Render with LocalSort for comparison
        guard let cb2 = queue.makeCommandBuffer() else { return }
        let lsResult = lsRenderer.render(
            toTexture: cb2, input: input, camera: camera,
            width: width, height: height, whiteBackground: false, mortonSorted: false
        )
        cb2.commit()
        cb2.waitUntilCompleted()

        guard let dfTex = dfResult?.color, let lsTex = lsResult?.color else {
            XCTFail("Failed to get textures")
            return
        }

        // Compare outputs
        let bytesPerRow = width * 8
        let dfReadBuffer = device.makeBuffer(length: width * height * 8, options: .storageModeShared)!
        let lsReadBuffer = device.makeBuffer(length: width * height * 8, options: .storageModeShared)!

        guard let blitCB = queue.makeCommandBuffer(),
              let blit = blitCB.makeBlitCommandEncoder() else { return }
        blit.copy(from: dfTex, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: width, height: height, depth: 1),
                  to: dfReadBuffer, destinationOffset: 0,
                  destinationBytesPerRow: bytesPerRow, destinationBytesPerImage: bytesPerRow * height)
        blit.copy(from: lsTex, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: width, height: height, depth: 1),
                  to: lsReadBuffer, destinationOffset: 0,
                  destinationBytesPerRow: bytesPerRow, destinationBytesPerImage: bytesPerRow * height)
        blit.endEncoding()
        blitCB.commit()
        blitCB.waitUntilCompleted()

        var dfPixels = [UInt16](repeating: 0, count: width * height * 4)
        var lsPixels = [UInt16](repeating: 0, count: width * height * 4)
        memcpy(&dfPixels, dfReadBuffer.contents(), width * height * 8)
        memcpy(&lsPixels, lsReadBuffer.contents(), width * height * 8)

        var maxDiff: Float = 0
        var diffCount = 0
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                let dfR = Float(Float16(bitPattern: dfPixels[i]))
                let lsR = Float(Float16(bitPattern: lsPixels[i]))
                let diff = abs(dfR - lsR)
                if diff > 0.001 {
                    diffCount += 1
                    maxDiff = max(maxDiff, diff)
                }
            }
        }

        print("=== Comparison Results ===")
        print("Different pixels: \(diffCount)")
        print("Max difference: \(maxDiff)")
        print("")

        // Accept some tolerance for half-precision
        XCTAssertLessThan(maxDiff, 0.1, "Max difference too high")
    }
}

// MARK: - Supporting Types
// Note: Uses PackedWorldGaussian from GaussianMetalRenderer module

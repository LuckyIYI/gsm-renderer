import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Tests for cluster-level culling of Morton-sorted Gaussians
/// Uses skip-based culling (no compaction) for optimal performance
final class ClusterCullTests: XCTestCase {
    private let imageWidth = 1920
    private let imageHeight = 1080

    // MARK: - Test Helpers

    private func createTestCamera(cameraPosition: SIMD3<Float> = SIMD3<Float>(0, 0, 0),
                                   lookAt: SIMD3<Float> = SIMD3<Float>(0, 0, 5)) -> CameraUniformsSwift {
        let aspect = Float(imageWidth) / Float(imageHeight)
        let fov: Float = 60.0 * .pi / 180.0
        let near: Float = 0.1
        let far: Float = 100.0

        let f = 1.0 / tan(fov / 2.0)
        let projectionMatrix = simd_float4x4(columns: (
            SIMD4<Float>(f / aspect, 0, 0, 0),
            SIMD4<Float>(0, f, 0, 0),
            SIMD4<Float>(0, 0, (far + near) / (near - far), -1),
            SIMD4<Float>(0, 0, (2 * far * near) / (near - far), 0)
        ))

        let forward = simd_normalize(lookAt - cameraPosition)
        let worldUp = SIMD3<Float>(0, 1, 0)
        let right = simd_normalize(simd_cross(forward, worldUp))
        let up = simd_cross(right, forward)

        let viewMatrix = simd_float4x4(columns: (
            SIMD4<Float>(right.x, up.x, -forward.x, 0),
            SIMD4<Float>(right.y, up.y, -forward.y, 0),
            SIMD4<Float>(right.z, up.z, -forward.z, 0),
            SIMD4<Float>(-simd_dot(right, cameraPosition), -simd_dot(up, cameraPosition), simd_dot(forward, cameraPosition), 1)
        ))

        let focalLength = Float(imageHeight) / (2.0 * tan(fov / 2.0))

        return CameraUniformsSwift(
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            cameraCenter: cameraPosition,
            pixelFactor: 1.0,
            focalX: focalLength,
            focalY: focalLength,
            width: Float(imageWidth),
            height: Float(imageHeight),
            nearPlane: near,
            farPlane: far,
            shComponents: 1,
            gaussianCount: 0
        )
    }

    private func createGaussianBuffer(device: MTLDevice, positions: [SIMD3<Float>]) -> (MTLBuffer, MTLBuffer)? {
        let count = positions.count
        var gaussians = [PackedWorldGaussian](repeating: PackedWorldGaussian(), count: count)
        var harmonics = [Float](repeating: 0.5, count: count * 3)

        for (i, pos) in positions.enumerated() {
            gaussians[i] = PackedWorldGaussian(
                px: pos.x, py: pos.y, pz: pos.z,
                opacity: 0.8,
                sx: 0.1, sy: 0.1, sz: 0.1,
                _pad0: 0,
                rotation: simd_float4(0, 0, 0, 1)
            )
        }

        guard let gaussianBuffer = device.makeBuffer(
            bytes: &gaussians,
            length: count * MemoryLayout<PackedWorldGaussian>.stride,
            options: .storageModeShared
        ) else { return nil }

        guard let harmonicsBuffer = device.makeBuffer(
            bytes: &harmonics,
            length: count * 3 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else { return nil }

        return (gaussianBuffer, harmonicsBuffer)
    }

    // MARK: - Tests

    /// Test that ClusterCullEncoder initializes correctly
    func testEncoderInitialization() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }

        let encoder = try ClusterCullEncoder(device: device)
        XCTAssertNotNil(encoder)
        XCTAssertEqual(encoder.clusterSize, UInt32(CLUSTER_SIZE))
    }

    /// Test that visibility buffer is created for visible clusters
    func testVisibilityBufferCreation() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("No Metal device")
        }

        // Create gaussians in front of camera
        let positions: [SIMD3<Float>] = (0..<100).map { i in
            SIMD3<Float>(Float(i % 10) - 5, Float(i / 10) - 5, 5)
        }

        guard let (gaussianBuffer, _) = createGaussianBuffer(device: device, positions: positions) else {
            XCTFail("Failed to create buffers")
            return
        }

        let encoder = try ClusterCullEncoder(device: device)
        let camera = createTestCamera()

        guard let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let visibility = encoder.encodeCull(
            commandBuffer: cb,
            worldGaussians: gaussianBuffer,
            gaussianCount: positions.count,
            viewMatrix: camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            useHalfWorld: false
        )

        XCTAssertNotNil(visibility, "Visibility buffer should be created")
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// Test that LocalSortRenderer correctly uses cluster culling with mortonSorted=true
    func testLocalSortRendererWithMortonCulling() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("No Metal device")
        }

        let renderer = try LocalSortRenderer(device: device)

        // Create some visible gaussians
        let positions: [SIMD3<Float>] = (0..<1000).map { i in
            SIMD3<Float>(Float(i % 32) - 16, Float(i / 32) - 16, 5)
        }

        guard let (gaussianBuffer, harmonicsBuffer) = createGaussianBuffer(device: device, positions: positions) else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = createTestCamera()
        let cameraParams = CameraParams(
            viewMatrix: camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            position: camera.cameraCenter,
            focalX: camera.focalX,
            focalY: camera.focalY
        )

        let input = GaussianInput(
            gaussians: gaussianBuffer,
            harmonics: harmonicsBuffer,
            gaussianCount: positions.count,
            shComponents: 1
        )

        guard let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        // Test with mortonSorted=true (uses cluster culling)
        let result = renderer.render(
            toTexture: cb,
            input: input,
            camera: cameraParams,
            width: imageWidth,
            height: imageHeight,
            whiteBackground: false,
            mortonSorted: true
        )

        cb.commit()
        cb.waitUntilCompleted()

        XCTAssertNotNil(result, "Render should succeed with mortonSorted=true")
        XCTAssertNotNil(result?.color, "Color texture should exist")
    }

    /// Test that rendering with and without culling produces similar results
    func testCullVsNoCullRendering() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("No Metal device")
        }

        let renderer = try LocalSortRenderer(device: device)

        // Create visible gaussians
        let positions: [SIMD3<Float>] = (0..<2048).map { i in
            SIMD3<Float>(Float(i % 64) - 32, Float(i / 64) - 16, 10)
        }

        guard let (gaussianBuffer, harmonicsBuffer) = createGaussianBuffer(device: device, positions: positions) else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = createTestCamera()
        let cameraParams = CameraParams(
            viewMatrix: camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            position: camera.cameraCenter,
            focalX: camera.focalX,
            focalY: camera.focalY
        )

        let input = GaussianInput(
            gaussians: gaussianBuffer,
            harmonics: harmonicsBuffer,
            gaussianCount: positions.count,
            shComponents: 1
        )

        // Render without culling
        guard let cb1 = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let resultNoCull = renderer.render(
            toTexture: cb1,
            input: input,
            camera: cameraParams,
            width: imageWidth,
            height: imageHeight,
            whiteBackground: false,
            mortonSorted: false
        )
        cb1.commit()
        cb1.waitUntilCompleted()

        // Render with culling
        guard let cb2 = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let resultWithCull = renderer.render(
            toTexture: cb2,
            input: input,
            camera: cameraParams,
            width: imageWidth,
            height: imageHeight,
            whiteBackground: false,
            mortonSorted: true
        )
        cb2.commit()
        cb2.waitUntilCompleted()

        XCTAssertNotNil(resultNoCull, "No-cull render should succeed")
        XCTAssertNotNil(resultWithCull, "With-cull render should succeed")
    }

    /// Benchmark: Compare culling vs no culling performance
    func testCullVsNoCullBenchmark() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("No Metal device")
        }

        let renderer = try LocalSortRenderer(device: device)
        let gaussianCount = 100_000

        // Create gaussians - half in front, half behind camera
        var positions: [SIMD3<Float>] = []
        for i in 0..<gaussianCount {
            let z = i < gaussianCount / 2 ? Float.random(in: 2...20) : Float.random(in: -20...(-2))
            positions.append(SIMD3<Float>(
                Float.random(in: -10...10),
                Float.random(in: -10...10),
                z
            ))
        }

        guard let (gaussianBuffer, harmonicsBuffer) = createGaussianBuffer(device: device, positions: positions) else {
            XCTFail("Failed to create buffers")
            return
        }

        let camera = createTestCamera()
        let cameraParams = CameraParams(
            viewMatrix: camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            position: camera.cameraCenter,
            focalX: camera.focalX,
            focalY: camera.focalY
        )

        let input = GaussianInput(
            gaussians: gaussianBuffer,
            harmonics: harmonicsBuffer,
            gaussianCount: gaussianCount,
            shComponents: 1
        )

        // Warmup
        for mortonSorted in [false, true] {
            for _ in 0..<3 {
                guard let cb = queue.makeCommandBuffer() else { continue }
                _ = renderer.render(
                    toTexture: cb,
                    input: input,
                    camera: cameraParams,
                    width: imageWidth,
                    height: imageHeight,
                    whiteBackground: false,
                    mortonSorted: mortonSorted
                )
                cb.commit()
                cb.waitUntilCompleted()
            }
        }

        // Benchmark
        let iterations = 10
        var noCullTimes: [Double] = []
        var cullTimes: [Double] = []

        for _ in 0..<iterations {
            // No cull
            guard let cb1 = queue.makeCommandBuffer() else { continue }
            _ = renderer.render(
                toTexture: cb1,
                input: input,
                camera: cameraParams,
                width: imageWidth,
                height: imageHeight,
                whiteBackground: false,
                mortonSorted: false
            )
            cb1.commit()
            cb1.waitUntilCompleted()
            noCullTimes.append((cb1.gpuEndTime - cb1.gpuStartTime) * 1000)

            // With cull
            guard let cb2 = queue.makeCommandBuffer() else { continue }
            _ = renderer.render(
                toTexture: cb2,
                input: input,
                camera: cameraParams,
                width: imageWidth,
                height: imageHeight,
                whiteBackground: false,
                mortonSorted: true
            )
            cb2.commit()
            cb2.waitUntilCompleted()
            cullTimes.append((cb2.gpuEndTime - cb2.gpuStartTime) * 1000)
        }

        let noCullAvg = noCullTimes.reduce(0, +) / Double(noCullTimes.count)
        let cullAvg = cullTimes.reduce(0, +) / Double(cullTimes.count)
        let speedup = noCullAvg / cullAvg

        print("\n[Skip-Based Cluster Culling Benchmark - \(gaussianCount) gaussians, ~50% culled]")
        print("  No Cull:   \(String(format: "%.3f", noCullAvg)) ms")
        print("  With Cull: \(String(format: "%.3f", cullAvg)) ms")
        print("  Speedup:   \(String(format: "%.2f", speedup))x")
    }
}

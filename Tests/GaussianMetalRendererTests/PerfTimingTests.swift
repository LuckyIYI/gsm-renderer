import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Performance timing tests for the rendering pipeline.
final class PerfTimingTests: XCTestCase {

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

    func generateRandomGaussians(count: Int, spreadX: Float = 800, spreadY: Float = 600) -> (
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
            let x = Float(i % gridSize) / Float(gridSize) * spreadX - spreadX/2
            let y = Float(i / gridSize) / Float(gridSize) * spreadY - spreadY/2
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

    /// Test renderer startup time
    func testStartupTiming() throws {
        let startTime = CFAbsoluteTimeGetCurrent()

        let renderer = Renderer(
            precision: .float32,
            usePreciseIntersection: true,
            useFusedCoverageScatter: true,
            useHeapAllocation: false,
            limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1920, maxHeight: 1080)
        )

        let endTime = CFAbsoluteTimeGetCurrent()
        let startupMs = (endTime - startTime) * 1000

        print("[PerfTiming] Renderer startup: \(String(format: "%.1f", startupMs))ms")

        // Verify renderer is functional
        XCTAssertNotNil(renderer.device)
        XCTAssertTrue(startupMs < 5000, "Startup should be under 5 seconds")
    }

    /// Test wall-clock render time
    func testRenderWallTime() throws {
        let width = 1024
        let height = 768
        let renderer = Renderer(
            precision: .float16,
            usePreciseIntersection: true,
            useFusedCoverageScatter: true,
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
        let frameParams = FrameParams(gaussianCount: count, whiteBackground: false)

        // Warm up
        for _ in 0..<3 {
            guard let commandBuffer = renderer.queue.makeCommandBuffer() else { continue }
            _ = renderer.encodeRenderToTextureHalf(
                commandBuffer: commandBuffer,
                gaussianCount: count,
                packedWorldBuffersHalf: packedBuffersHalf,
                cameraUniforms: camera,
                frameParams: frameParams
            )
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        // Measure wall time
        var wallTimes: [Double] = []
        var gpuTimes: [Double] = []

        for _ in 0..<10 {
            guard let commandBuffer = renderer.queue.makeCommandBuffer() else { continue }

            let wallStart = CFAbsoluteTimeGetCurrent()
            _ = renderer.encodeRenderToTextureHalf(
                commandBuffer: commandBuffer,
                gaussianCount: count,
                packedWorldBuffersHalf: packedBuffersHalf,
                cameraUniforms: camera,
                frameParams: frameParams
            )
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let wallEnd = CFAbsoluteTimeGetCurrent()

            wallTimes.append((wallEnd - wallStart) * 1000)

            let gpuTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
            if gpuTime > 0 {
                gpuTimes.append(gpuTime * 1000)
            }
        }

        let avgWall = wallTimes.reduce(0, +) / Double(wallTimes.count)
        let avgGPU = gpuTimes.isEmpty ? 0 : gpuTimes.reduce(0, +) / Double(gpuTimes.count)

        print("[PerfTiming] \(count) gaussians @ \(width)x\(height): wall=\(String(format: "%.2f", avgWall))ms gpu=\(String(format: "%.2f", avgGPU))ms")

        XCTAssertTrue(avgWall < 100, "Render should complete under 100ms")
    }

    /// Test precise scale timing at different counts
    func testPreciseScaleTiming() throws {
        let width = 1920
        let height = 1080
        let renderer = Renderer(
            precision: .float16,
            usePreciseIntersection: true,
            useFusedCoverageScatter: true,
            useHeapAllocation: false,
            textureOnly: true,
            limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: width, maxHeight: height)
        )

        let testCounts = [10_000, 50_000, 100_000, 500_000]
        var results: [(count: Int, avgMs: Double, minMs: Double, maxMs: Double)] = []

        for count in testCounts {
            let (positions, scales, rotations, opacities, colors) = generateRandomGaussians(
                count: count,
                spreadX: Float(width),
                spreadY: Float(height)
            )

            guard let packedBuffersHalf = createPackedWorldBuffersHalf(
                device: renderer.device,
                positions: positions,
                scales: scales,
                rotations: rotations,
                opacities: opacities,
                colors: colors
            ) else {
                continue
            }

            let camera = createSimpleCamera(width: Float(width), height: Float(height), gaussianCount: count)
            let frameParams = FrameParams(gaussianCount: count, whiteBackground: false)

            // Warm up
            for _ in 0..<2 {
                guard let commandBuffer = renderer.queue.makeCommandBuffer() else { continue }
                _ = renderer.encodeRenderToTextureHalf(
                    commandBuffer: commandBuffer,
                    gaussianCount: count,
                    packedWorldBuffersHalf: packedBuffersHalf,
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
                _ = renderer.encodeRenderToTextureHalf(
                    commandBuffer: commandBuffer,
                    gaussianCount: count,
                    packedWorldBuffersHalf: packedBuffersHalf,
                    cameraUniforms: camera,
                    frameParams: frameParams
                )
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()

                let gpuTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
                if gpuTime > 0 {
                    times.append(gpuTime * 1000)
                }
            }

            if !times.isEmpty {
                let avg = times.reduce(0, +) / Double(times.count)
                results.append((count, avg, times.min()!, times.max()!))
            }
        }

        // Print summary
        var summary = "[PerfTiming] Scale timing: "
        for r in results {
            summary += "\(r.count/1000)k: avg=\(String(format: "%.2f", r.avgMs))ms | "
        }
        print(summary)

        XCTAssertTrue(results.count > 0, "Should have timing results")
    }
}

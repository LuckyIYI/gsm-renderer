import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Performance comparison: OLD renderer vs NEW (Tellusim) renderer
final class PerfComparisonTests: XCTestCase {
    private let imageWidth = 1920
    private let imageHeight = 1080

    func testCompare1M() throws {
        try comparePerformance(gaussianCount: 1_000_000, label: "1M")
    }

    func testCompare2M() throws {
        try comparePerformance(gaussianCount: 2_000_000, label: "2M")
    }

    func testCompare4M() throws {
        try comparePerformance(gaussianCount: 4_000_000, label: "4M")
    }

    private func comparePerformance(gaussianCount: Int, label: String) throws {
        // Create OLD renderer with appropriate limits
        let oldRenderer = GlobalSortRenderer(
            precision: Precision.float32,
            
            limits: RendererLimits(
                maxGaussians: gaussianCount,
                maxWidth: imageWidth,
                maxHeight: imageHeight,
                tileWidth: 32,
                tileHeight: 16
            )
        )
        let device = oldRenderer.device
        let queue = oldRenderer.queue

        // Create test data
        var positions: [SIMD3<Float>] = []
        var scales: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var opacities: [Float] = []
        var colors: [SIMD3<Float>] = []

        positions.reserveCapacity(gaussianCount)
        scales.reserveCapacity(gaussianCount)
        rotations.reserveCapacity(gaussianCount)
        opacities.reserveCapacity(gaussianCount)
        colors.reserveCapacity(gaussianCount)

        for _ in 0..<gaussianCount {
            let x = Float.random(in: -10...10)
            let y = Float.random(in: -10...10)
            let z = Float.random(in: 2...20)

            positions.append(SIMD3<Float>(x, y, z))
            scales.append(SIMD3<Float>(0.02, 0.02, 0.02))
            rotations.append(SIMD4<Float>(0, 0, 0, 1))
            opacities.append(Float.random(in: 0.3...1.0))
            colors.append(SIMD3<Float>(
                Float.random(in: 0...1),
                Float.random(in: 0...1),
                Float.random(in: 0...1)
            ))
        }

        // Create packed buffers for OLD renderer
        guard let packedBuffers = createPackedWorldBuffers(
            device: device,
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        ) else {
            XCTFail("Failed to create packed buffers")
            return
        }

        // Camera setup
        let fov: Float = 60.0 * .pi / 180.0
        let aspect = Float(imageWidth) / Float(imageHeight)
        let near: Float = 0.2
        let far: Float = 100.0

        let f = 1.0 / tan(fov / 2.0)
        let projectionMatrix = simd_float4x4(columns: (
            SIMD4<Float>(f / aspect, 0, 0, 0),
            SIMD4<Float>(0, f, 0, 0),
            SIMD4<Float>(0, 0, (far + near) / (near - far), -1),
            SIMD4<Float>(0, 0, (2 * far * near) / (near - far), 0)
        ))
        let viewMatrix = simd_float4x4(1.0)

        let focalX = f * Float(imageWidth) / 2.0
        let focalY = f * Float(imageHeight) / 2.0

        let camera = CameraUniformsSwift(
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            cameraCenter: SIMD3<Float>(0, 0, 0),
            pixelFactor: 1.0,
            focalX: focalX,
            focalY: focalY,
            width: Float(imageWidth),
            height: Float(imageHeight),
            nearPlane: near,
            farPlane: far,
            shComponents: 0,
            gaussianCount: UInt32(gaussianCount)
        )

        let frameParams = FrameParams(gaussianCount: gaussianCount, whiteBackground: false)

        // ============ BENCHMARK OLD RENDERER ============
        let iterations = 10
        var oldTimes = [Double]()

        // Warmup + benchmark old renderer
        for i in 0..<iterations + 2 {
            guard let cb = queue.makeCommandBuffer() else { continue }
            let start = CACurrentMediaTime()

            _ = oldRenderer.encodeRenderToTextures(
                commandBuffer: cb,
                gaussianCount: gaussianCount,
                packedWorldBuffers: packedBuffers,
                cameraUniforms: camera,
                frameParams: frameParams
            )

            cb.commit()
            cb.waitUntilCompleted()
            if i >= 2 { oldTimes.append(CACurrentMediaTime() - start) }
        }

        // ============ BENCHMARK NEW (TELLUSIM) RENDERER ============
        let tellusim = try LocalSortRenderer(device: device)
        var newTimes = [Double]()

        // Warmup + benchmark Tellusim renderer
        for i in 0..<iterations + 2 {
            guard let cb = queue.makeCommandBuffer() else { continue }
            let start = CACurrentMediaTime()

            _ = tellusim.render(
                commandBuffer: cb,
                worldGaussians: packedBuffers.packedGaussians,
                harmonics: packedBuffers.harmonics,
                gaussianCount: gaussianCount,
                viewMatrix: viewMatrix,
                projectionMatrix: projectionMatrix,
                cameraPosition: SIMD3<Float>(0, 0, 0),
                focalX: focalX,
                focalY: focalY,
                width: imageWidth,
                height: imageHeight,
                shComponents: 0
            )

            cb.commit()
            cb.waitUntilCompleted()
            if i >= 2 { newTimes.append(CACurrentMediaTime() - start) }
        }

        // ============ RESULTS ============
        let oldAvg = oldTimes.reduce(0, +) / Double(oldTimes.count) * 1000
        let oldMin = oldTimes.min()! * 1000
        let oldMax = oldTimes.max()! * 1000

        let newAvg = newTimes.reduce(0, +) / Double(newTimes.count) * 1000
        let newMin = newTimes.min()! * 1000
        let newMax = newTimes.max()! * 1000

        let speedup = oldAvg / newAvg
        let winner = newAvg < oldAvg ? "NEW (Tellusim)" : "OLD"

        print("\n╔═══════════════════════════════════════════════════════════════════╗")
        print("║  PERFORMANCE COMPARISON: \(label) GAUSSIANS @ \(imageWidth)x\(imageHeight)")
        print("╠═══════════════════════════════════════════════════════════════════╣")
        print("║  OLD Renderer:")
        print("║    Avg: \(String(format: "%7.2f", oldAvg))ms  (\(String(format: "%5.1f", 1000/oldAvg)) FPS)")
        print("║    Min: \(String(format: "%7.2f", oldMin))ms  Max: \(String(format: "%.2f", oldMax))ms")
        print("╠───────────────────────────────────────────────────────────────────╣")
        print("║  NEW Renderer (Tellusim):")
        print("║    Avg: \(String(format: "%7.2f", newAvg))ms  (\(String(format: "%5.1f", 1000/newAvg)) FPS)")
        print("║    Min: \(String(format: "%7.2f", newMin))ms  Max: \(String(format: "%.2f", newMax))ms")
        print("╠═══════════════════════════════════════════════════════════════════╣")
        print("║  Speedup: \(String(format: "%.2fx", speedup)) - Winner: \(winner)")
        print("║  60 FPS (16.67ms): OLD \(oldAvg <= 16.67 ? "✓" : "✗")  NEW \(newAvg <= 16.67 ? "✓" : "✗")")
        print("║  120 FPS (8.33ms): OLD \(oldAvg <= 8.33 ? "✓" : "✗")  NEW \(newAvg <= 8.33 ? "✓" : "✗")")
        print("╚═══════════════════════════════════════════════════════════════════╝\n")

        // Assert new renderer is at least competitive (not more than 50% slower)
        XCTAssertLessThan(newAvg, oldAvg * 1.5, "New renderer should not be >50% slower than old")
    }

    // MARK: - Helper

    private func createPackedWorldBuffers(
        device: MTLDevice,
        positions: [SIMD3<Float>],
        scales: [SIMD3<Float>],
        rotations: [SIMD4<Float>],
        opacities: [Float],
        colors: [SIMD3<Float>]
    ) -> PackedWorldBuffers? {
        let count = positions.count

        var packed: [PackedWorldGaussian] = []
        packed.reserveCapacity(count)
        for i in 0..<count {
            packed.append(PackedWorldGaussian(
                position: positions[i],
                scale: scales[i],
                rotation: rotations[i],
                opacity: opacities[i]
            ))
        }

        var harmonics: [Float] = []
        harmonics.reserveCapacity(count * 3)
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
}

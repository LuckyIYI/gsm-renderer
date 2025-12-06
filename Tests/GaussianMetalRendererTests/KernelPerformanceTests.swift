@testable import GaussianMetalRenderer
import Metal
import simd
import XCTest

/// Isolated kernel performance tests for projection and scatter
/// Tests realistic scales (2-6M gaussians) to establish baselines
final class KernelPerformanceTests: XCTestCase {
    var device: MTLDevice!
    var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        self.device = MTLCreateSystemDefaultDevice()!
        self.queue = self.device.makeCommandQueue()!
    }

    // MARK: - Test Data Generation

    /// Generate gaussians that are actually visible in the default camera view
    func generateVisibleGaussians(count: Int, seed: Int = 42) -> (
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

        positions.reserveCapacity(count)
        scales.reserveCapacity(count)
        rotations.reserveCapacity(count)
        opacities.reserveCapacity(count)
        colors.reserveCapacity(count)

        srand48(seed)
        for i in 0 ..< count {
            // Position in view frustum (z between 1-10, xy scaled to be visible)
            let z = Float(drand48() * 8 + 1.5) // 1.5 to 9.5
            let spread = z * 0.6 // Spread increases with depth (frustum shape)
            let x = Float(drand48() * 2 - 1) * spread
            let y = Float(drand48() * 2 - 1) * spread
            positions.append(SIMD3(x, y, z))

            // Larger scales for visibility
            let s = Float(drand48() * 0.15 + 0.08)
            scales.append(SIMD3(s, s, s))

            rotations.append(SIMD4(0, 0, 0, 1))
            opacities.append(Float(drand48() * 0.5 + 0.5))
            colors.append(SIMD3(
                Float(drand48()),
                Float(drand48()),
                Float(drand48())
            ))
        }

        return (positions, scales, rotations, opacities, colors)
    }

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

        positions.reserveCapacity(count)
        scales.reserveCapacity(count)
        rotations.reserveCapacity(count)
        opacities.reserveCapacity(count)
        colors.reserveCapacity(count)

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

    func createPackedBuffers(positions: [SIMD3<Float>], scales: [SIMD3<Float>],
                             rotations: [SIMD4<Float>], opacities: [Float],
                             colors: [SIMD3<Float>]) -> PackedWorldBuffers?
    {
        let count = positions.count
        var packed: [PackedWorldGaussian] = []
        packed.reserveCapacity(count)
        for i in 0 ..< count {
            packed.append(PackedWorldGaussian(
                position: positions[i], scale: scales[i], rotation: rotations[i], opacity: opacities[i]
            ))
        }

        var harmonics: [Float] = []
        harmonics.reserveCapacity(count * 3)
        for color in colors {
            harmonics.append(color.x)
            harmonics.append(color.y)
            harmonics.append(color.z)
        }

        guard let packedBuf = self.device.makeBuffer(bytes: &packed,
                                                     length: count * MemoryLayout<PackedWorldGaussian>.stride,
                                                     options: .storageModeShared),
              let harmonicsBuf = self.device.makeBuffer(bytes: &harmonics,
                                                        length: count * 3 * 4,
                                                        options: .storageModeShared)
        else {
            return nil
        }

        return PackedWorldBuffers(packedGaussians: packedBuf, harmonics: harmonicsBuf)
    }

    // MARK: - Kernel Timing Helper

    struct KernelTiming {
        let name: String
        let times: [Double]

        var avg: Double { self.times.reduce(0, +) / Double(self.times.count) }
        var min: Double { self.times.min() ?? 0 }
        var max: Double { self.times.max() ?? 0 }
        var stdDev: Double {
            let mean = self.avg
            let variance = self.times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(self.times.count)
            return sqrt(variance)
        }
    }

    func measureKernel(name: String, warmup: Int = 3, iterations: Int = 10, _ block: () -> Void) -> KernelTiming {
        // Warmup
        for _ in 0 ..< warmup {
            block()
        }

        // Measure
        var times: [Double] = []
        for _ in 0 ..< iterations {
            let start = CFAbsoluteTimeGetCurrent()
            block()
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
            times.append(elapsed)
        }

        return KernelTiming(name: name, times: times)
    }

    func printTiming(_ timing: KernelTiming, gaussianCount: Int) {
        let throughput = Double(gaussianCount) / (timing.avg * 1000.0) // M gaussians/sec
        print("  \(timing.name):")
        print("    Avg: \(String(format: "%.3f", timing.avg))ms (stddev: \(String(format: "%.3f", timing.stdDev)))")
        print("    Min: \(String(format: "%.3f", timing.min))ms, Max: \(String(format: "%.3f", timing.max))ms")
        print("    Throughput: \(String(format: "%.2f", throughput))M gaussians/sec")
    }

    // MARK: - Full Pipeline Performance Test (Baseline)

    func testFullPipelineBaseline() throws {
        let testCases: [(count: Int, name: String)] = [
            (2_000_000, "2M"),
            (4_000_000, "4M"),
            (6_000_000, "6M"),
        ]

        let width = 1920
        let height = 1080

        print("\n" + String(repeating: "=", count: 60))
        print("FULL PIPELINE PERFORMANCE BASELINE")
        print("Resolution: \(width)x\(height)")
        print("Device: \(self.device.name)")
        print(String(repeating: "=", count: 60))

        for testCase in testCases {
            print("\n[\(testCase.name) Gaussians]")

            let config = RendererConfig(
                maxGaussians: testCase.count,
                maxWidth: width,
                maxHeight: height,
                precision: .float16
            )

            let renderer = try LocalRenderer(config: config)

            let (positions, scales, rotations, opacities, colors) = self.generateGaussians(count: testCase.count)
            guard let buffers = self.createPackedBuffers(positions: positions, scales: scales,
                                                         rotations: rotations, opacities: opacities, colors: colors)
            else {
                XCTFail("Failed to create buffers")
                return
            }

            let input = GaussianInput(
                gaussians: buffers.packedGaussians,
                harmonics: buffers.harmonics,
                gaussianCount: testCase.count,
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

            let timing = self.measureKernel(name: "Full Pipeline", warmup: 3, iterations: 10) {
                guard let cb = self.queue.makeCommandBuffer() else { return }
                _ = renderer.render(
                    toTexture: cb,
                    input: input,
                    camera: camera,
                    width: width,
                    height: height,
                    whiteBackground: false
                )
                cb.commit()
                cb.waitUntilCompleted()
            }

            self.printTiming(timing, gaussianCount: testCase.count)

            // Print visible count and FPS
            let visibleCount = renderer.getVisibleCount()
            let visibilityRate = Double(visibleCount) / Double(testCase.count) * 100.0
            let fps = 1000.0 / timing.avg
            print("    Visible: \(visibleCount) (\(String(format: "%.1f", visibilityRate))%)")
            print("    FPS: \(String(format: "%.1f", fps))")
        }

        print("\n" + String(repeating: "=", count: 60) + "\n")
    }

    // MARK: - Resolution Scaling Test

    func testResolutionScaling() throws {
        let gaussianCount = 4_000_000
        let resolutions: [(width: Int, height: Int, name: String)] = [
            (1280, 720, "720p"),
            (1920, 1080, "1080p"),
            (2560, 1440, "1440p"),
            (3840, 2160, "4K"),
        ]

        print("\n" + String(repeating: "=", count: 60))
        print("RESOLUTION SCALING TEST")
        print("Gaussians: \(gaussianCount / 1_000_000)M")
        print("Device: \(self.device.name)")
        print(String(repeating: "=", count: 60))

        let (positions, scales, rotations, opacities, colors) = self.generateGaussians(count: gaussianCount)

        for res in resolutions {
            print("\n[\(res.name) - \(res.width)x\(res.height)]")

            let config = RendererConfig(
                maxGaussians: gaussianCount,
                maxWidth: res.width,
                maxHeight: res.height,
                precision: .float16
            )

            let renderer = try LocalRenderer(config: config)

            guard let buffers = self.createPackedBuffers(positions: positions, scales: scales,
                                                         rotations: rotations, opacities: opacities, colors: colors)
            else {
                XCTFail("Failed to create buffers")
                return
            }

            let input = GaussianInput(
                gaussians: buffers.packedGaussians,
                harmonics: buffers.harmonics,
                gaussianCount: gaussianCount,
                shComponents: 0
            )

            let aspect = Float(res.width) / Float(res.height)
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
                focalX: Float(res.width) * f / (2 * aspect),
                focalY: Float(res.height) * f / 2
            )

            let timing = self.measureKernel(name: res.name, warmup: 3, iterations: 10) {
                guard let cb = self.queue.makeCommandBuffer() else { return }
                _ = renderer.render(
                    toTexture: cb,
                    input: input,
                    camera: camera,
                    width: res.width,
                    height: res.height,
                    whiteBackground: false
                )
                cb.commit()
                cb.waitUntilCompleted()
            }

            let fps = 1000.0 / timing.avg
            let tilesX = (res.width + 15) / 16
            let tilesY = (res.height + 15) / 16
            print("  Time: \(String(format: "%.2f", timing.avg))ms")
            print("  FPS: \(String(format: "%.1f", fps))")
            print("  Tiles: \(tilesX)x\(tilesY) = \(tilesX * tilesY)")
        }

        print("\n" + String(repeating: "=", count: 60) + "\n")
    }

    // MARK: - Sort Performance Test (16-bit only)

    func testSortModeComparison() throws {
        let testCases: [(count: Int, name: String)] = [
            (2_000_000, "2M"),
            (4_000_000, "4M"),
        ]

        let width = 1920
        let height = 1080

        print("\n" + String(repeating: "=", count: 60))
        print("16-BIT SORT PERFORMANCE")
        print("Resolution: \(width)x\(height)")
        print("Device: \(self.device.name)")
        print(String(repeating: "=", count: 60))

        for testCase in testCases {
            print("\n[\(testCase.name) Gaussians]")

            let (positions, scales, rotations, opacities, colors) = self.generateGaussians(count: testCase.count)

            let config = RendererConfig(
                maxGaussians: testCase.count,
                maxWidth: width,
                maxHeight: height,
                precision: .float16
            )

            let renderer = try LocalRenderer(config: config)

            guard let buffers = self.createPackedBuffers(positions: positions, scales: scales,
                                                         rotations: rotations, opacities: opacities, colors: colors)
            else {
                XCTFail("Failed to create buffers")
                return
            }

            let input = GaussianInput(
                gaussians: buffers.packedGaussians,
                harmonics: buffers.harmonics,
                gaussianCount: testCase.count,
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

            let timing = self.measureKernel(name: "16-bit", warmup: 3, iterations: 10) {
                guard let cb = self.queue.makeCommandBuffer() else { return }
                _ = renderer.render(
                    toTexture: cb,
                    input: input,
                    camera: camera,
                    width: width,
                    height: height,
                    whiteBackground: false
                )
                cb.commit()
                cb.waitUntilCompleted()
            }

            print("  Time: \(String(format: "%.2f", timing.avg))ms (\(String(format: "%.1f", 1000.0 / timing.avg)) FPS)")
        }

        print("\n" + String(repeating: "=", count: 60) + "\n")
    }

    // MARK: - Combined Baseline Test

    func testKernelBaseline() throws {
        print("\n")
        print(String(repeating: "=", count: 70))
        print("  KERNEL PERFORMANCE BASELINE - Local Renderer")
        print("  Device: \(self.device.name)")
        print(String(repeating: "=", count: 70))

        try self.testFullPipelineBaseline()
        try self.testResolutionScaling()
        try self.testSortModeComparison()
    }

}

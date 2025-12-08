import CoreGraphics
import ImageIO
import Metal
@testable import Renderer
import simd
import UniformTypeIdentifiers
import XCTest

/// Benchmark tests using real PLY scene data
final class PLYBenchmarkTests: XCTestCase {
    static let plyPath = "../point_cloud.ply"

    var device: MTLDevice!
    var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        self.device = MTLCreateSystemDefaultDevice()!
        self.queue = self.device.makeCommandQueue()!
    }

    // MARK: - Helper Functions

    /// Creates camera params looking at a scene center from a given distance.
    /// OpenCV convention: +X right, +Y down, +Z forward.
    func createSceneCameraParams(
        lookAt _: SIMD3<Float>,
        distance _: Float,
        width: Int,
        height: Int
    ) -> CameraParams {
        let viewMatrix = simd_float4x4([[0.7994084, -0.22484124, 0.557129, 0.0], [0.28039712, 0.9597669, -0.014999399, 0.0], [-0.53134143, 0.16820799, 0.8302905, 0.0], [-1.7380741, -13.886897, 11.441933, 1.0]])
        let projectionMatrix = makeProjectionMatrix(width: width, height: height, near: 0.1, far: 10.0)

        let aspect = Float(width) / Float(height)
        let fov: Float = TestCameraDefaults.fovDegrees * .pi / 180.0
        let f = 1.0 / tan(fov / 2.0)

        return CameraParams(
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            position: .zero,
            focalX: Float(width) * f / (2 * aspect),
            focalY: Float(height) * f / 2
        )
    }

    // MARK: - PLY Loading Test

    func testLoadPLYScene() throws {
        let url = URL(fileURLWithPath: Self.plyPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("PLY file not found at \(Self.plyPath)")
        }

        print("\n=== Loading PLY Scene ===")

        let start = CFAbsoluteTimeGetCurrent()
        let dataset = try PLYLoader.load(url: url)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        print("Loaded \(dataset.records.count) gaussians in \(String(format: "%.2f", elapsed))s")
        print("SH components: \(dataset.shComponents)")

        // Compute scene bounds
        let bounds = GaussianSceneBuilder.bounds(of: dataset.records)
        print("Scene center: \(bounds.center)")
        print("Scene radius: \(bounds.radius)")

        XCTAssertGreaterThan(dataset.records.count, 0, "Should load some gaussians")
    }

    // MARK: - Render Test with PLY data

    func testRenderPLYScene() throws {
        let url = URL(fileURLWithPath: Self.plyPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("PLY file not found at \(Self.plyPath)")
        }

        print("\n=== Render PLY Scene Test ===")

        // Load PLY with Y-flip for correct orientation
        let dataset = try PLYLoader.load(url: url)
        let gaussianCount = dataset.records.count
        print("Loaded \(gaussianCount) gaussians, SH: \(dataset.shComponents)")

        // Get scene bounds
        let bounds = GaussianSceneBuilder.bounds(of: dataset.records)
        print("Scene center: \(bounds.center), radius: \(bounds.radius)")

        let width = 1920
        let height = 1080

        // Create packed gaussians from PLY data
        var packed: [PackedWorldGaussian] = []
        packed.reserveCapacity(gaussianCount)
        for record in dataset.records {
            packed.append(PackedWorldGaussian(
                position: record.position,
                scale: record.scale,
                rotation: SIMD4<Float>(record.rotation.imag.x, record.rotation.imag.y, record.rotation.imag.z, record.rotation.real),
                opacity: record.opacity
            ))
        }

        // Extract DC coefficients from harmonics
        var harmonics = dataset.harmonics
        guard let gaussianBuf = device.makeBuffer(bytes: &packed, length: packed.count * MemoryLayout<PackedWorldGaussian>.stride, options: .storageModeShared),
              let harmonicsBuf = device.makeBuffer(bytes: &harmonics, length: harmonics.count * MemoryLayout<Float>.stride, options: .storageModeShared)
        else {
            XCTFail("Failed to create buffers")
            return
        }

        let config = RendererConfig(maxGaussians: gaussianCount, maxWidth: width, maxHeight: height, precision: .float32)
        let localRenderer = try LocalRenderer(config: config)
        let globalRenderer = try GlobalRenderer(config: config)
        let depthFirstRenderer = try DepthFirstRenderer(config: config)

        let cameraMargin: Float = 0.25
        let cameraDistance = bounds.radius * cameraMargin
        let camera = createSceneCameraParams(lookAt: bounds.center - .init(-6.0, -10.0, 0.0), distance: cameraDistance, width: width, height: height)

        let input = GaussianInput(
            gaussians: gaussianBuf,
            harmonics: harmonicsBuf,
            gaussianCount: gaussianCount,
            shComponents: dataset.shComponents
        )

        // Create output textures
        guard let colorTexture = makeColorTexture(device: device, width: width, height: height, pixelFormat: .rgba8Unorm),
              let depthTexture = makeDepthTexture(device: device, width: width, height: height)
        else {
            XCTFail("Failed to create textures")
            return
        }

        guard let q = localRenderer.device.makeCommandQueue() else {
            XCTFail("Failed to create command queue")
            return
        }

        print("\nBenchmarking \(gaussianCount) gaussians at \(width)x\(height)...")
        print("Warmup: 3 runs, Measurement: 10 runs\n")

        // Run benchmarks for Local renderer
        print("=== Local Renderer ===")
        let localTiming = benchmark(name: "Local") {
            guard let cb = q.makeCommandBuffer() else { return }
            localRenderer.render(
                commandBuffer: cb,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                input: input,
                camera: camera,
                width: width,
                height: height
            )
            cb.commit()
            cb.waitUntilCompleted()
        }
        print("Local: avg=\(String(format: "%.2f", localTiming.avg))ms, min=\(String(format: "%.2f", localTiming.min))ms, max=\(String(format: "%.2f", localTiming.max))ms")

        // Also benchmark Global renderer
        print("\n=== Global Renderer ===")

        let globalTiming = benchmark(name: "Global") {
            guard let cb = q.makeCommandBuffer() else { return }
            globalRenderer.render(
                commandBuffer: cb,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                input: input,
                camera: camera,
                width: width,
                height: height
            )
            cb.commit()
            cb.waitUntilCompleted()
        }
        print("Global: avg=\(String(format: "%.2f", globalTiming.avg))ms, min=\(String(format: "%.2f", globalTiming.min))ms, max=\(String(format: "%.2f", globalTiming.max))ms")

        // Benchmark DepthFirst renderer
        print("\n=== DepthFirst Renderer ===")

        let depthFirstTiming = benchmark(name: "DepthFirst") {
            guard let cb = q.makeCommandBuffer() else { return }
            depthFirstRenderer.render(
                commandBuffer: cb,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                input: input,
                camera: camera,
                width: width,
                height: height
            )
            cb.commit()
            cb.waitUntilCompleted()
        }
        print("DepthFirst: avg=\(String(format: "%.2f", depthFirstTiming.avg))ms, min=\(String(format: "%.2f", depthFirstTiming.min))ms, max=\(String(format: "%.2f", depthFirstTiming.max))ms")

        // Summary comparison
        print("\n=== Summary ===")
        print("Local: \(String(format: "%.2f", localTiming.avg))ms")
        print("Global: \(String(format: "%.2f", globalTiming.avg))ms")
        print("DepthFirst: \(String(format: "%.2f", depthFirstTiming.avg))ms")

        // Save final render for visual verification
        if let cb = q.makeCommandBuffer() {
            localRenderer.render(
                commandBuffer: cb,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                input: input,
                camera: camera,
                width: width,
                height: height
            )
            cb.commit()
            cb.waitUntilCompleted()
        }
        saveTextureToJPEG(texture: colorTexture, filename: "render_local_output.jpg")

        // Save Global render
        if let cb = q.makeCommandBuffer() {
            globalRenderer.render(
                commandBuffer: cb,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                input: input,
                camera: camera,
                width: width,
                height: height
            )
            cb.commit()
            cb.waitUntilCompleted()
        }
        saveTextureToJPEG(texture: colorTexture, filename: "render_global_output.jpg")

        // Save DepthFirst render
        if let cb = q.makeCommandBuffer() {
            depthFirstRenderer.render(
                commandBuffer: cb,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                input: input,
                camera: camera,
                width: width,
                height: height
            )
            cb.commit()
            cb.waitUntilCompleted()
        }
        saveTextureToJPEG(texture: colorTexture, filename: "render_depthfirst_output.jpg")
    }

    // MARK: - Debug Helpers

    /// Save texture as JPEG by compositing over a solid background (alpha treated as transparency)
    /// background: RGBA in [0,1] for the backdrop color to composite against (default black)
    func saveTextureToJPEG(texture: MTLTexture, filename: String, background: SIMD4<Float> = SIMD4<Float>(0, 0, 0, 1)) {
        guard let device = texture.device.makeCommandQueue()?.device else {
            print("Failed to access Metal device from texture")
            return
        }
        let ciContext = CIContext(mtlDevice: device)
        guard var ciImage = CIImage(mtlTexture: texture, options: [CIImageOption.colorSpace: CGColorSpaceCreateDeviceRGB()]) else {
            print("Failed to create CIImage from MTLTexture")
            return
        }
        // Orient to match prior PNG path
        ciImage = ciImage.oriented(.downMirrored)

        // Create a solid background color image
        let bgColor = CIColor(red: CGFloat(background.x), green: CGFloat(background.y), blue: CGFloat(background.z), alpha: CGFloat(background.w))
        guard let bgGenerator = CIFilter(name: "CIConstantColorGenerator") else {
            print("Failed to create CIConstantColorGenerator filter")
            return
        }
        bgGenerator.setValue(bgColor, forKey: kCIInputColorKey)
        guard let bgImage = bgGenerator.outputImage?.cropped(to: CGRect(x: 0, y: 0, width: texture.width, height: texture.height)) else {
            print("Failed to generate background image")
            return
        }

        // Composite source over background (alpha treated as transparency)
        guard let overFilter = CIFilter(name: "CISourceOverCompositing") else {
            print("Failed to create CISourceOverCompositing filter")
            return
        }
        overFilter.setValue(ciImage, forKey: kCIInputImageKey)
        overFilter.setValue(bgImage, forKey: kCIInputBackgroundImageKey)
        guard let composited = overFilter.outputImage else {
            print("Failed to composite image over background")
            return
        }

        let rect = CGRect(x: 0, y: 0, width: texture.width, height: texture.height)
        guard let cgImage = ciContext.createCGImage(composited, from: rect) else {
            print("Failed to create CGImage from composited CIImage")
            return
        }

        let desktopURL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Desktop/\(filename)")
        guard let dest = CGImageDestinationCreateWithURL(desktopURL as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
            print("Failed to create image destination for JPEG")
            return
        }
        // Optionally set JPEG quality
        let options: [CFString: Any] = [kCGImageDestinationLossyCompressionQuality: 0.95]
        CGImageDestinationAddImage(dest, cgImage, options as CFDictionary)
        if CGImageDestinationFinalize(dest) {
            print("Saved render (JPEG) to: \(desktopURL.path)")
        } else {
            print("Failed to save JPEG")
        }
    }
}

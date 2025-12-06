@testable import GaussianMetalRenderer
import CoreGraphics
import ImageIO
import Metal
import simd
import UniformTypeIdentifiers
import XCTest

/// Benchmark tests using real PLY scene data
final class PLYBenchmarkTests: XCTestCase {

    // Path to test PLY file - use smaller bonsai.ply for faster iteration
    static let plyPath = "/Users/laki/Documents/GitHub/renderer/gsm-renderer/point_cloud.ply"

    var device: MTLDevice!
    var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        self.device = MTLCreateSystemDefaultDevice()!
        self.queue = self.device.makeCommandQueue()!
    }

    // MARK: - Helper Functions

    func createCameraParams(
        lookAt center: SIMD3<Float>,
        distance: Float,
        width: Int,
        height: Int
    ) -> CameraParams {
        let aspect = Float(width) / Float(height)
        let fov: Float = 60.0 * .pi / 180.0
        let near: Float = 0.1
        let far: Float = 1000.0
        let f = 1.0 / tan(fov / 2.0)

        let cameraPos = center - SIMD3<Float>(0, 0, distance)

        // View matrix: translate world so camera is at origin, scene is at +Z
        var viewMatrix = matrix_identity_float4x4
        viewMatrix.columns.3 = SIMD4(-cameraPos.x, -cameraPos.y, -cameraPos.z, 1)

        // Standard perspective projection matrix (Metal NDC: z in [0, 1])
        var projMatrix = matrix_identity_float4x4
        projMatrix.columns.0 = SIMD4(f / aspect, 0, 0, 0)
        projMatrix.columns.1 = SIMD4(0, f, 0, 0)
        projMatrix.columns.2 = SIMD4(0, 0, -far / (far - near), -1)
        projMatrix.columns.3 = SIMD4(0, 0, -(far * near) / (far - near), 0)

        return CameraParams(
            viewMatrix: viewMatrix,
            projectionMatrix: projMatrix,
            position: cameraPos,
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
        let dataset = try PLYLoader.load(url: url, flipY: true)
        let gaussianCount = dataset.records.count
        print("Loaded \(gaussianCount) gaussians, SH: \(dataset.shComponents)")

        // Get scene bounds
        let bounds = GaussianSceneBuilder.bounds(of: dataset.records)
        print("Scene center: \(bounds.center), radius: \(bounds.radius)")

        // Print actual min/max to understand scene orientation
        var minP = dataset.records[0].position
        var maxP = dataset.records[0].position
        for r in dataset.records {
            minP = SIMD3<Float>(min(minP.x, r.position.x), min(minP.y, r.position.y), min(minP.z, r.position.z))
            maxP = SIMD3<Float>(max(maxP.x, r.position.x), max(maxP.y, r.position.y), max(maxP.z, r.position.z))
        }
        print("Scene min: \(minP), max: \(maxP)")

        let width = 1920
        let height = 1080

        // Create packed gaussians from PLY data (half precision to match default renderer config)
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

        // Extract DC coefficients from harmonics - planar layout is:
        // [R0..R15, G0..G15, B0..B15] per gaussian where shComponents=16
        // DC coefficients are at indices 0 (R), shComponents (G), shComponents*2 (B)
        var harmonics = [Float]()
        harmonics.reserveCapacity(gaussianCount * 3)
        let shComponents = dataset.shComponents
        let shCoeffsPerGaussian = shComponents * 3
        for i in 0..<gaussianCount {
            let baseIdx = i * shCoeffsPerGaussian
            if baseIdx + shComponents * 2 < dataset.harmonics.count {
                // DC_R at offset 0, DC_G at offset shComponents, DC_B at offset shComponents*2
                harmonics.append(dataset.harmonics[baseIdx + 0])              // R DC
                harmonics.append(dataset.harmonics[baseIdx + shComponents])   // G DC
                harmonics.append(dataset.harmonics[baseIdx + shComponents * 2]) // B DC
            } else {
                harmonics.append(0.5)
                harmonics.append(0.5)
                harmonics.append(0.5)
            }
        }

        guard let gaussianBuf = device.makeBuffer(bytes: &packed, length: packed.count * MemoryLayout<PackedWorldGaussian>.stride, options: .storageModeShared),
              let harmonicsBuf = device.makeBuffer(bytes: &harmonics, length: harmonics.count * MemoryLayout<Float>.stride, options: .storageModeShared)
        else {
            XCTFail("Failed to create buffers")
            return
        }

        print("Buffers: gaussians=\(gaussianBuf.length) bytes, harmonics=\(harmonicsBuf.length) bytes")

        let config = RendererConfig(maxGaussians: gaussianCount, maxWidth: width, maxHeight: height, precision: .float32)
        let renderer = try LocalRenderer(config: config)

        let cameraMargin: Float = 0.15
        let cameraDistance = bounds.radius * cameraMargin
        print("Camera at distance \(cameraDistance) (radius \(bounds.radius) + \(Int((cameraMargin - 1) * 100))% margin)")
        let camera = createCameraParams(lookAt: bounds.center - .init(6.0, 14.0, 0.0), distance: cameraDistance, width: width, height: height)

        let input = GaussianInput(
            gaussians: gaussianBuf,
            harmonics: harmonicsBuf,
            gaussianCount: gaussianCount,
            shComponents: 0  // Using DC only for now
        )

        print("Rendering \(gaussianCount) gaussians at \(width)x\(height)...")

        // Create output textures (caller's responsibility with new API)
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        colorDesc.usage = [.shaderRead, .shaderWrite]
        colorDesc.storageMode = .private
        guard let colorTexture = device.makeTexture(descriptor: colorDesc) else {
            XCTFail("Failed to create color texture")
            return
        }

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float, width: width, height: height, mipmapped: false)
        depthDesc.usage = [.shaderRead, .shaderWrite]
        depthDesc.storageMode = .private
        let depthTexture = device.makeTexture(descriptor: depthDesc)

        guard let q = renderer.device.makeCommandQueue(),
              let cb = q.makeCommandBuffer()
        else {
            XCTFail("Failed to create command buffer")
            return
        }

        let start = CFAbsoluteTimeGetCurrent()
        renderer.render(
            commandBuffer: cb,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            input: input,
            camera: camera,
            width: width,
            height: height,
            whiteBackground: false
        )
        
        cb.commit()
        cb.waitUntilCompleted()

        let renderTime = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        print("Render time: \(String(format: "%.2f", renderTime))ms")
        print("Output: \(colorTexture.width)x\(colorTexture.height) texture")

        // Save texture to PNG for debugging
        saveTextureToPNG(texture: colorTexture, filename: "render_output.png")
    }

    // MARK: - Debug Helpers

    func saveTextureToPNG(texture: MTLTexture, filename: String) {
        let width = texture.width
        let height = texture.height
        let bytesPerPixel = 8 // RGBA16Float = 8 bytes per pixel
        let bytesPerRow = width * bytesPerPixel

        // Create a staging buffer to copy texture data (texture may be .private storage)
        guard let stagingBuffer = device.makeBuffer(length: height * bytesPerRow, options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let blitEncoder = commandBuffer.makeBlitCommandEncoder()
        else {
            print("Failed to create staging resources")
            return
        }

        blitEncoder.copy(
            from: texture,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: width, height: height, depth: 1),
            to: stagingBuffer,
            destinationOffset: 0,
            destinationBytesPerRow: bytesPerRow,
            destinationBytesPerImage: height * bytesPerRow
        )
        blitEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let pixelData = stagingBuffer.contents().assumingMemoryBound(to: UInt8.self)

        // Convert RGBA16Float to RGBA8 for PNG
        var rgba8 = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let srcIdx = (y * width + x) * 8
                let dstIdx = (y * width + x) * 4

                // Read half-floats and convert to 0-255
                for c in 0..<4 {
                    let halfBits = UInt16(pixelData[srcIdx + c * 2]) | (UInt16(pixelData[srcIdx + c * 2 + 1]) << 8)
                    let floatVal = halfToFloat(halfBits)
                    rgba8[dstIdx + c] = UInt8(max(0, min(255, floatVal * 255)))
                }
            }
        }

        // Create CGImage
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &rgba8,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cgImage = context.makeImage() else {
            print("Failed to create CGImage")
            return
        }

        // Save to Desktop
        let desktopURL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Desktop/\(filename)")
        guard let dest = CGImageDestinationCreateWithURL(desktopURL as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            print("Failed to create image destination")
            return
        }
        CGImageDestinationAddImage(dest, cgImage, nil)
        if CGImageDestinationFinalize(dest) {
            print("Saved render to: \(desktopURL.path)")
        } else {
            print("Failed to save PNG")
        }
    }

    func halfToFloat(_ h: UInt16) -> Float {
        let sign = (h & 0x8000) >> 15
        let exp = (h & 0x7C00) >> 10
        let mant = h & 0x03FF

        if exp == 0 {
            if mant == 0 { return sign == 0 ? 0.0 : -0.0 }
            // Subnormal
            var m = Float(mant) / 1024.0
            m *= pow(2.0, -14.0)
            return sign == 0 ? m : -m
        } else if exp == 31 {
            if mant == 0 { return sign == 0 ? Float.infinity : -Float.infinity }
            return Float.nan
        }

        let f = Float(sign == 0 ? 1 : -1) * pow(2.0, Float(Int(exp) - 15)) * (1.0 + Float(mant) / 1024.0)
        return f
    }
}

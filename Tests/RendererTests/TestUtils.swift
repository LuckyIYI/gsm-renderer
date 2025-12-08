import Metal
@testable import Renderer
import simd
import XCTest

// MARK: - Standard Camera Parameters

/// Standard near/far planes for test cameras
enum TestCameraDefaults {
    static let near: Float = 0.1
    static let far: Float = 100.0
    static let fovDegrees: Float = 60.0
}

/// Creates a standard perspective projection matrix for tests.
/// Uses OpenCV convention: +X right, +Y down, +Z forward.
func makeProjectionMatrix(
    width: Int,
    height: Int,
    near: Float = TestCameraDefaults.near,
    far: Float = TestCameraDefaults.far,
    fovDegrees: Float = TestCameraDefaults.fovDegrees
) -> simd_float4x4 {
    let aspect = Float(width) / Float(height)
    let fov = fovDegrees * .pi / 180.0
    let f = 1.0 / tan(fov / 2.0)

    // OpenCV projection matrix (Metal NDC: z in [0, 1])
    // No Y-flip here - the view matrix uses "down" vector which handles orientation
    var proj = matrix_identity_float4x4
    proj.columns.0 = SIMD4(f / aspect, 0, 0, 0)
    proj.columns.1 = SIMD4(0, f, 0, 0)
    proj.columns.2 = SIMD4(0, 0, far / (far - near), 1)
    proj.columns.3 = SIMD4(0, 0, -(far * near) / (far - near), 0)
    return proj
}

/// Creates CameraParams for the GaussianRenderer protocol.
func makeCameraParams(
    width: Int,
    height: Int,
    position: SIMD3<Float> = .zero,
    viewMatrix: simd_float4x4 = matrix_identity_float4x4,
    near: Float = TestCameraDefaults.near,
    far: Float = TestCameraDefaults.far
) -> CameraParams {
    let aspect = Float(width) / Float(height)
    let fov: Float = TestCameraDefaults.fovDegrees * .pi / 180.0
    let f = 1.0 / tan(fov / 2.0)
    let projMatrix = makeProjectionMatrix(width: width, height: height, near: near, far: far)

    return CameraParams(
        viewMatrix: viewMatrix,
        projectionMatrix: projMatrix,
        position: position,
        focalX: Float(width) * f / (2 * aspect),
        focalY: Float(height) * f / 2
    )
}

/// Creates CameraUniformsSwift for internal renderer APIs.
func makeCameraUniforms(
    width: Int,
    height: Int,
    gaussianCount: Int,
    shComponents: Int = 0,
    position: SIMD3<Float> = .zero,
    viewMatrix: simd_float4x4 = matrix_identity_float4x4,
    near: Float = TestCameraDefaults.near,
    far: Float = TestCameraDefaults.far
) -> CameraUniformsSwift {
    let aspect = Float(width) / Float(height)
    let fov: Float = TestCameraDefaults.fovDegrees * .pi / 180.0
    let f = 1.0 / tan(fov / 2.0)
    let projMatrix = makeProjectionMatrix(width: width, height: height, near: near, far: far)

    return CameraUniformsSwift(
        viewMatrix: viewMatrix,
        projectionMatrix: projMatrix,
        cameraCenter: position,
        pixelFactor: 1.0,
        focalX: Float(width) * f / (2 * aspect),
        focalY: Float(height) * f / 2,
        width: Float(width),
        height: Float(height),
        nearPlane: near,
        farPlane: far,
        shComponents: UInt32(shComponents),
        gaussianCount: UInt32(gaussianCount)
    )
}

// MARK: - Gaussian Generation

/// Generated gaussian data arrays
struct GeneratedGaussians {
    let positions: [SIMD3<Float>]
    let scales: [SIMD3<Float>]
    let rotations: [SIMD4<Float>]
    let opacities: [Float]
    let colors: [SIMD3<Float>]

    var count: Int { positions.count }
}

/// Generates test gaussians on a grid pattern.
/// OpenCV convention: +X right, +Y down, +Z forward.
func generateGridGaussians(count: Int, seed: Int = 42) -> GeneratedGaussians {
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
        // OpenCV: objects in front of camera at positive Z
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

    return GeneratedGaussians(
        positions: positions,
        scales: scales,
        rotations: rotations,
        opacities: opacities,
        colors: colors
    )
}

/// Generates test gaussians in view frustum with larger sizes for visibility testing.
/// OpenCV convention: +X right, +Y down, +Z forward.
func generateVisibleGaussians(count: Int, seed: Int = 42) -> GeneratedGaussians {
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
    for _ in 0 ..< count {
        // OpenCV: objects in front of camera at positive Z (1.5 to 9.5)
        let z = Float(drand48() * 8 + 1.5)
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

    return GeneratedGaussians(
        positions: positions,
        scales: scales,
        rotations: rotations,
        opacities: opacities,
        colors: colors
    )
}

// MARK: - Buffer Creation

/// Creates PackedWorldBuffers from generated gaussians.
func makePackedBuffers(
    device: MTLDevice,
    gaussians: GeneratedGaussians,
    options: MTLResourceOptions = .storageModeShared
) -> PackedWorldBuffers? {
    let count = gaussians.count
    var packed: [PackedWorldGaussian] = []
    packed.reserveCapacity(count)
    for i in 0 ..< count {
        packed.append(PackedWorldGaussian(
            position: gaussians.positions[i],
            scale: gaussians.scales[i],
            rotation: gaussians.rotations[i],
            opacity: gaussians.opacities[i]
        ))
    }

    var harmonics: [Float] = []
    harmonics.reserveCapacity(count * 3)
    for color in gaussians.colors {
        harmonics.append(color.x)
        harmonics.append(color.y)
        harmonics.append(color.z)
    }

    guard let packedBuf = device.makeBuffer(
        bytes: &packed,
        length: count * MemoryLayout<PackedWorldGaussian>.stride,
        options: options
    ),
        let harmonicsBuf = device.makeBuffer(
            bytes: &harmonics,
            length: count * 3 * MemoryLayout<Float>.stride,
            options: options
        )
    else {
        return nil
    }

    return PackedWorldBuffers(packedGaussians: packedBuf, harmonics: harmonicsBuf)
}

/// Creates GaussianInput from generated gaussians.
func makeGaussianInput(
    device: MTLDevice,
    gaussians: GeneratedGaussians,
    shComponents: Int = 0
) -> GaussianInput? {
    guard let buffers = makePackedBuffers(device: device, gaussians: gaussians) else {
        return nil
    }
    return GaussianInput(
        gaussians: buffers.packedGaussians,
        harmonics: buffers.harmonics,
        gaussianCount: gaussians.count,
        shComponents: shComponents
    )
}

// MARK: - Texture Creation

/// Creates a color texture for rendering output.
func makeColorTexture(
    device: MTLDevice,
    width: Int,
    height: Int,
    pixelFormat: MTLPixelFormat = .rgba16Float,
    storageMode: MTLStorageMode = .private
) -> MTLTexture? {
    let desc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: pixelFormat,
        width: width,
        height: height,
        mipmapped: false
    )
    desc.usage = [.shaderRead, .shaderWrite]
    desc.storageMode = storageMode
    return device.makeTexture(descriptor: desc)
}

/// Creates a depth texture for rendering output.
func makeDepthTexture(
    device: MTLDevice,
    width: Int,
    height: Int,
    pixelFormat: MTLPixelFormat = .r16Float,
    storageMode: MTLStorageMode = .private
) -> MTLTexture? {
    let desc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: pixelFormat,
        width: width,
        height: height,
        mipmapped: false
    )
    desc.usage = [.shaderRead, .shaderWrite]
    desc.storageMode = storageMode
    return device.makeTexture(descriptor: desc)
}

/// Creates both color and depth textures for rendering.
func makeRenderTextures(
    device: MTLDevice,
    width: Int,
    height: Int
) -> (color: MTLTexture, depth: MTLTexture)? {
    guard let color = makeColorTexture(device: device, width: width, height: height),
          let depth = makeDepthTexture(device: device, width: width, height: height)
    else {
        return nil
    }
    return (color, depth)
}

// MARK: - Benchmark Helpers

/// Result of a benchmark measurement.
struct BenchmarkResult {
    let name: String
    let times: [Double]

    var avg: Double { times.reduce(0, +) / Double(times.count) }
    var min: Double { times.min() ?? 0 }
    var max: Double { times.max() ?? 0 }
    var fps: Double { 1000.0 / avg }

    var stdDev: Double {
        let mean = avg
        let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(times.count)
        return sqrt(variance)
    }

    func printSummary(gaussianCount: Int? = nil) {
        print("  \(name):")
        print("    Avg: \(String(format: "%.3f", avg))ms (stddev: \(String(format: "%.3f", stdDev)))")
        print("    Min: \(String(format: "%.3f", min))ms, Max: \(String(format: "%.3f", max))ms")
        print("    FPS: \(String(format: "%.1f", fps))")
        if let count = gaussianCount {
            let throughput = Double(count) / (avg * 1000.0) // M gaussians/sec
            print("    Throughput: \(String(format: "%.2f", throughput))M gaussians/sec")
        }
    }
}

/// Measures execution time of a block with warmup and multiple iterations.
func benchmark(
    name: String,
    warmup: Int = 3,
    iterations: Int = 10,
    _ block: () -> Void
) -> BenchmarkResult {
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

    return BenchmarkResult(name: name, times: times)
}

// MARK: - Metal Extensions

extension MTLDevice {
    /// Creates a buffer with type-safe count and stride calculation.
    func makeBuffer<T>(count: Int, type _: T.Type, options: MTLResourceOptions, label: String? = nil) throws -> MTLBuffer {
        let length = count * MemoryLayout<T>.stride
        guard let buffer = makeBuffer(length: length, options: options) else {
            throw TestError.bufferCreationFailed(label ?? "unnamed")
        }
        buffer.label = label
        return buffer
    }
}

enum TestError: Error {
    case bufferCreationFailed(String)
    case textureCreationFailed
    case deviceNotAvailable
}

// MARK: - Pixel Reading (for comparison tests)

/// Reads pixels from a texture into a byte array.
func readPixels(texture: MTLTexture, device: MTLDevice, queue: MTLCommandQueue) -> [UInt8]? {
    let bytesPerPixel = 4
    let bytesPerRow = texture.width * bytesPerPixel

    guard let buffer = device.makeBuffer(length: bytesPerRow * texture.height, options: .storageModeShared),
          let cb = queue.makeCommandBuffer(),
          let blit = cb.makeBlitCommandEncoder()
    else {
        return nil
    }

    blit.copy(
        from: texture,
        sourceSlice: 0,
        sourceLevel: 0,
        sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
        sourceSize: MTLSize(width: texture.width, height: texture.height, depth: 1),
        to: buffer,
        destinationOffset: 0,
        destinationBytesPerRow: bytesPerRow,
        destinationBytesPerImage: bytesPerRow * texture.height
    )
    blit.endEncoding()

    cb.commit()
    cb.waitUntilCompleted()

    let ptr = buffer.contents().bindMemory(to: UInt8.self, capacity: bytesPerRow * texture.height)
    return Array(UnsafeBufferPointer(start: ptr, count: bytesPerRow * texture.height))
}

/// Counts non-black pixels in RGBA8 pixel data.
func countNonBlackPixels(_ pixels: [UInt8], threshold: UInt8 = 10) -> Int {
    var count = 0
    for i in stride(from: 0, to: pixels.count, by: 4) {
        if pixels[i] > threshold || pixels[i + 1] > threshold || pixels[i + 2] > threshold {
            count += 1
        }
    }
    return count
}

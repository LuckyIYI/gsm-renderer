import simd
@preconcurrency import Metal

// MARK: - Precision

public enum RenderPrecision: Sendable {
    case float32
    case float16
}

// MARK: - Input Data

/// Unified input for Gaussian splatting renderers
/// Contains packed world-space gaussian data and spherical harmonics
public struct GaussianInput: Sendable {
    public let gaussians: MTLBuffer    // PackedWorldGaussian (48 bytes) or PackedWorldGaussianHalf (24 bytes)
    public let harmonics: MTLBuffer    // Spherical harmonics coefficients
    public let gaussianCount: Int
    public let shComponents: Int

    public init(
        gaussians: MTLBuffer,
        harmonics: MTLBuffer,
        gaussianCount: Int,
        shComponents: Int
    ) {
        self.gaussians = gaussians
        self.harmonics = harmonics
        self.gaussianCount = gaussianCount
        self.shComponents = shComponents
    }
}

// MARK: - Camera Parameters

public struct CameraParams: Sendable {
    public let viewMatrix: simd_float4x4
    public let projectionMatrix: simd_float4x4
    public let position: SIMD3<Float>
    public let focalX: Float
    public let focalY: Float
    public let near: Float
    public let far: Float

    public init(
        viewMatrix: simd_float4x4,
        projectionMatrix: simd_float4x4,
        position: SIMD3<Float>,
        focalX: Float,
        focalY: Float,
        near: Float = 0.1,
        far: Float = 100.0
    ) {
        self.viewMatrix = viewMatrix
        self.projectionMatrix = projectionMatrix
        self.position = position
        self.focalX = focalX
        self.focalY = focalY
        self.near = near
        self.far = far
    }
}

// MARK: - Render Output

/// Result from rendering to textures (GPU-only output)
public struct TextureRenderResult: Sendable {
    public let color: MTLTexture
    public let depth: MTLTexture?
    public let alpha: MTLTexture?

    public init(color: MTLTexture, depth: MTLTexture? = nil, alpha: MTLTexture? = nil) {
        self.color = color
        self.depth = depth
        self.alpha = alpha
    }
}

/// Result from rendering to buffers (CPU-readable output)
public struct BufferRenderResult: Sendable {
    public let color: MTLBuffer   // RGB float32, stride 12 bytes per pixel
    public let depth: MTLBuffer   // Float32, stride 4 bytes per pixel
    public let alpha: MTLBuffer   // Float32, stride 4 bytes per pixel

    public init(color: MTLBuffer, depth: MTLBuffer, alpha: MTLBuffer) {
        self.color = color
        self.depth = depth
        self.alpha = alpha
    }
}

// MARK: - Renderer Configuration

public struct RendererConfig: Sendable {
    public let maxGaussians: Int
    public let maxWidth: Int
    public let maxHeight: Int
    public let precision: RenderPrecision

    public init(
        maxGaussians: Int = 2_000_000,
        maxWidth: Int = 1920,
        maxHeight: Int = 1080,
        precision: RenderPrecision = .float16
    ) {
        self.maxGaussians = maxGaussians
        self.maxWidth = maxWidth
        self.maxHeight = maxHeight
        self.precision = precision
    }
}

// MARK: - Gaussian Renderer Protocol

/// Protocol for Gaussian splatting renderers
/// Exactly 2 render methods: one for textures, one for buffers
public protocol GaussianRenderer: AnyObject, Sendable {
    /// The Metal device used by this renderer
    var device: MTLDevice { get }

    /// Render to GPU textures (fastest path, no CPU readback)
    func render(
        toTexture commandBuffer: MTLCommandBuffer,
        input: GaussianInput,
        camera: CameraParams,
        width: Int,
        height: Int,
        whiteBackground: Bool
    ) -> TextureRenderResult?

    /// Render to CPU-readable buffers
    func render(
        toBuffer commandBuffer: MTLCommandBuffer,
        input: GaussianInput,
        camera: CameraParams,
        width: Int,
        height: Int,
        whiteBackground: Bool
    ) -> BufferRenderResult?
}

// MARK: - Errors

public enum RendererError: Error, Sendable {
    case deviceNotAvailable
    case failedToCreatePipeline(String)
    case invalidInput(String)
    case renderFailed(String)
    case bufferRenderNotSupported
}

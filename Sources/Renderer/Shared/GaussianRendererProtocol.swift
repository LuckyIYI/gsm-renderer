@preconcurrency import Metal
import simd

public enum RenderPrecision: Sendable {
    case float32
    case float16
}

public struct GaussianInput: Sendable {
    public let gaussians: MTLBuffer // PackedWorldGaussian (48 bytes) or PackedWorldGaussianHalf (24 bytes)
    public let harmonics: MTLBuffer // Spherical harmonics coefficients
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
        far: Float = 10.0
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

public struct StereoCameraParams: Sendable {
    public let leftEye: CameraParams
    public let rightEye: CameraParams

    public init(
        leftEye: CameraParams,
        rightEye: CameraParams
    ) {
        self.leftEye = leftEye
        self.rightEye = rightEye
    }
}

public struct EyeView: Sendable {
    public let viewport: MTLViewport
    public let viewMatrix: simd_float4x4
    public let projectionMatrix: simd_float4x4
    public let cameraPosition: SIMD3<Float>
    public let focalX: Float
    public let focalY: Float
    public let near: Float
    public let far: Float

    public init(
        viewport: MTLViewport,
        viewMatrix: simd_float4x4,
        projectionMatrix: simd_float4x4,
        cameraPosition: SIMD3<Float>,
        focalX: Float,
        focalY: Float,
        near: Float = 0.1,
        far: Float = 10.0
    ) {
        self.viewport = viewport
        self.viewMatrix = viewMatrix
        self.projectionMatrix = projectionMatrix
        self.cameraPosition = cameraPosition
        self.focalX = focalX
        self.focalY = focalY
        self.near = near
        self.far = far
    }
}

public struct StereoConfiguration: Sendable {
    /// Left eye view configuration
    public let leftEye: EyeView
    /// Right eye view configuration
    public let rightEye: EyeView
    /// Scene transform - transforms gaussian positions from scene space to world space
    public let sceneTransform: simd_float4x4

    public init(
        leftEye: EyeView,
        rightEye: EyeView,
        sceneTransform: simd_float4x4 = matrix_identity_float4x4
    ) {
        self.leftEye = leftEye
        self.rightEye = rightEye
        self.sceneTransform = sceneTransform
    }

    init(
        from camera: StereoCameraParams,
        width: Int,
        height: Int,
        leftViewOrigin: SIMD2<Float> = .zero,
        rightViewOrigin: SIMD2<Float> = .zero
    ) {
        let leftEyeView = EyeView(
            viewport: MTLViewport(
                originX: Double(leftViewOrigin.x),
                originY: Double(leftViewOrigin.y),
                width: Double(width),
                height: Double(height),
                znear: 0,
                zfar: 1
            ),
            viewMatrix: camera.leftEye.viewMatrix,
            projectionMatrix: camera.leftEye.projectionMatrix,
            cameraPosition: camera.leftEye.position,
            focalX: camera.leftEye.focalX,
            focalY: camera.leftEye.focalY,
            near: camera.leftEye.near,
            far: camera.leftEye.far
        )
        let rightEyeView = EyeView(
            viewport: MTLViewport(
                originX: Double(rightViewOrigin.x),
                originY: Double(rightViewOrigin.y),
                width: Double(width),
                height: Double(height),
                znear: 0,
                zfar: 1
            ),
            viewMatrix: camera.rightEye.viewMatrix,
            projectionMatrix: camera.rightEye.projectionMatrix,
            cameraPosition: camera.rightEye.position,
            focalX: camera.rightEye.focalX,
            focalY: camera.rightEye.focalY,
            near: camera.rightEye.near,
            far: camera.rightEye.far
        )

        self = StereoConfiguration(
            leftEye: leftEyeView,
            rightEye: rightEyeView
        )
    }
}

/// Drawable output for foveated stereo rendering (from Compositor Services)
public struct FoveatedStereoDrawable: Sendable {
    /// Color texture - either a texture array (layered) or single texture (shared/dedicated)
    public let colorTexture: MTLTexture
    /// Depth texture (optional) - matches colorTexture format
    public let depthTexture: MTLTexture?
    /// Rasterization rate map for foveated rendering (from Compositor Services)
    public let rasterizationRateMap: MTLRasterizationRateMap?
    /// Color texture pixel format
    public let colorPixelFormat: MTLPixelFormat
    /// Depth texture pixel format (if depth texture provided)
    public let depthPixelFormat: MTLPixelFormat

    public init(
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        rasterizationRateMap: MTLRasterizationRateMap?,
        colorPixelFormat: MTLPixelFormat = .bgra8Unorm_srgb,
        depthPixelFormat: MTLPixelFormat = .depth32Float
    ) {
        self.colorTexture = colorTexture
        self.depthTexture = depthTexture
        self.rasterizationRateMap = rasterizationRateMap
        self.colorPixelFormat = colorPixelFormat
        self.depthPixelFormat = depthPixelFormat
    }
}

public struct RendererConfig: Sendable {
    public enum GaussianColorSpace: UInt32, Sendable {
        /// Values are already linear.
        case linear = 0
        /// Values are encoded as sRGB and must be decoded to linear before blending/output.
        case srgb = 1
    }

    public let maxGaussians: Int
    public let maxWidth: Int
    public let maxHeight: Int
    public let precision: RenderPrecision
    public let colorFormat: MTLPixelFormat
    public let gaussianColorSpace: GaussianColorSpace
    public let backToFront: Bool

    public init(
        maxGaussians: Int = 6_000_000,
        maxWidth: Int = 1920,
        maxHeight: Int = 1080,
        precision: RenderPrecision = .float16,
        colorFormat: MTLPixelFormat = .bgra8Unorm_srgb,
        gaussianColorSpace: GaussianColorSpace = .srgb,
        backToFront: Bool = false
    ) {
        self.maxGaussians = maxGaussians
        self.maxWidth = maxWidth
        self.maxHeight = maxHeight
        self.precision = precision
        self.colorFormat = colorFormat
        self.gaussianColorSpace = gaussianColorSpace
        self.backToFront = backToFront
    }
}

// MARK: - Stereo Render Target

/// Unified stereo render target - all stereo rendering goes through this
public enum StereoRenderTarget: Sendable {
    /// Render side-by-side to a single texture (left eye on left half, right eye on right half)
    case sideBySide(colorTexture: MTLTexture, depthTexture: MTLTexture?)

    /// Render to foveated stereo drawable (Vision Pro Compositor Services)
    case foveated(drawable: FoveatedStereoDrawable, configuration: StereoConfiguration)
}

// MARK: - Protocol

public protocol GaussianRenderer: AnyObject, Sendable {
    var device: MTLDevice { get }
    var lastGPUTime: Double? { get }

    /// Render gaussians to a single texture (mono rendering)
    func render(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        input: GaussianInput,
        camera: CameraParams,
        width: Int,
        height: Int
    )

    /// Render stereo to the specified target
    /// - Parameters:
    ///   - target: Where to render (side-by-side or foveated)
    ///   - camera: Stereo camera parameters (used for sideBySide)
    ///   - width: Width per eye in pixels
    ///   - height: Height per eye in pixels
    func renderStereo(
        commandBuffer: MTLCommandBuffer,
        target: StereoRenderTarget,
        input: GaussianInput,
        camera: StereoCameraParams,
        width: Int,
        height: Int
    )
}

public enum RendererError: Error, Sendable, CustomStringConvertible {
    // Initialization errors
    case deviceNotAvailable
    case failedToCreateLibrary(String)
    case failedToCreatePipeline(String)
    case failedToAllocateBuffer(label: String, size: Int)
    case failedToAllocateTexture(label: String, width: Int, height: Int)

    // Input validation errors
    case invalidGaussianCount(provided: Int, maximum: Int)
    case invalidDimensions(width: Int, height: Int, maxWidth: Int, maxHeight: Int)
    case invalidBufferSize(buffer: String, expected: Int, actual: Int)
    case invalidTileCount(provided: Int, maximum: Int)
    case invalidAssignmentCapacity(required: Int, available: Int)

    // Runtime errors
    case renderFailed(String)
    case encoderCreationFailed(stage: String)
    case missingRequiredBuffer(String)

    public var description: String {
        switch self {
        case .deviceNotAvailable:
            "Metal device not available"
        case let .failedToCreateLibrary(name):
            "Failed to create Metal library: \(name)"
        case let .failedToCreatePipeline(name):
            "Failed to create pipeline: \(name)"
        case let .failedToAllocateBuffer(label, size):
            "Failed to allocate buffer '\(label)' with size \(size) bytes"
        case let .failedToAllocateTexture(label, width, height):
            "Failed to allocate texture '\(label)' (\(width)x\(height))"
        case let .invalidGaussianCount(provided, maximum):
            "Gaussian count \(provided) exceeds maximum \(maximum)"
        case let .invalidDimensions(width, height, maxWidth, maxHeight):
            "Dimensions \(width)x\(height) exceed maximum \(maxWidth)x\(maxHeight)"
        case let .invalidBufferSize(buffer, expected, actual):
            "Buffer '\(buffer)' has invalid size: expected \(expected) bytes, got \(actual) bytes"
        case let .invalidTileCount(provided, maximum):
            "Tile count \(provided) exceeds maximum \(maximum)"
        case let .invalidAssignmentCapacity(required, available):
            "Required tile assignment capacity \(required) exceeds available \(available)"
        case let .renderFailed(reason):
            "Render failed: \(reason)"
        case let .encoderCreationFailed(stage):
            "Failed to create compute encoder for stage: \(stage)"
        case let .missingRequiredBuffer(name):
            "Missing required buffer: \(name)"
        }
    }
}

enum BufferValidation {
    /// Size of PackedWorldGaussian (float32 precision) in bytes
    static let packedWorldGaussianSize = 48

    /// Size of PackedWorldGaussianHalf (float16 precision) in bytes
    static let packedWorldGaussianHalfSize = 32

    /// Size of a single SH coefficient (float) in bytes
    static let shCoefficientSize = 4

    /// Size of a single SH coefficient (half) in bytes
    static let shCoefficientHalfSize = 2

    /// Validate gaussian buffer size for given count and precision
    static func validateGaussianBuffer(
        _ buffer: MTLBuffer,
        gaussianCount: Int,
        precision: RenderPrecision
    ) throws {
        let elementSize = precision == .float32 ? packedWorldGaussianSize : packedWorldGaussianHalfSize
        let expectedSize = gaussianCount * elementSize
        guard buffer.length >= expectedSize else {
            throw RendererError.invalidBufferSize(
                buffer: "gaussians",
                expected: expectedSize,
                actual: buffer.length
            )
        }
    }

    /// Validate harmonics buffer size for given count and SH components
    static func validateHarmonicsBuffer(
        _ buffer: MTLBuffer,
        gaussianCount: Int,
        shComponents: Int,
        precision: RenderPrecision
    ) throws {
        // Each gaussian has shComponents * 3 coefficients (R, G, B channels)
        let coeffSize = precision == .float32 ? shCoefficientSize : shCoefficientHalfSize
        let expectedSize = gaussianCount * shComponents * 3 * coeffSize
        guard buffer.length >= expectedSize else {
            throw RendererError.invalidBufferSize(
                buffer: "harmonics",
                expected: expectedSize,
                actual: buffer.length
            )
        }
    }

    /// Validate complete GaussianInput
    static func validate(
        _ input: GaussianInput,
        precision: RenderPrecision,
        maxGaussians: Int
    ) throws {
        guard input.gaussianCount <= maxGaussians else {
            throw RendererError.invalidGaussianCount(
                provided: input.gaussianCount,
                maximum: maxGaussians
            )
        }

        try validateGaussianBuffer(input.gaussians, gaussianCount: input.gaussianCount, precision: precision)

        if input.shComponents > 0 {
            try validateHarmonicsBuffer(
                input.harmonics,
                gaussianCount: input.gaussianCount,
                shComponents: input.shComponents,
                precision: precision
            )
        }
    }

    /// Validate render dimensions
    static func validateDimensions(
        width: Int,
        height: Int,
        maxWidth: Int,
        maxHeight: Int
    ) throws {
        guard width <= maxWidth, height <= maxHeight else {
            throw RendererError.invalidDimensions(
                width: width,
                height: height,
                maxWidth: maxWidth,
                maxHeight: maxHeight
            )
        }
    }
}

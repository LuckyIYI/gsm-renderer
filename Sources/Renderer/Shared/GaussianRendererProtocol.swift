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
    public let sortPosition: SIMD3<Float>

    public init(
        leftEye: CameraParams,
        rightEye: CameraParams,
        sortPosition: SIMD3<Float>? = nil
    ) {
        self.leftEye = leftEye
        self.rightEye = rightEye
        self.sortPosition = sortPosition ?? (leftEye.position + rightEye.position) * 0.5
    }
}

public struct StereoRenderOutput: Sendable {
    public let leftColor: MTLTexture
    public let leftDepth: MTLTexture?
    public let rightColor: MTLTexture
    public let rightDepth: MTLTexture?

    public init(
        leftColor: MTLTexture,
        leftDepth: MTLTexture?,
        rightColor: MTLTexture,
        rightDepth: MTLTexture?
    ) {
        self.leftColor = leftColor
        self.leftDepth = leftDepth
        self.rightColor = rightColor
        self.rightDepth = rightDepth
    }
}

public struct RendererConfig: Sendable {
    public let maxGaussians: Int
    public let maxWidth: Int
    public let maxHeight: Int
    public let precision: RenderPrecision

    public init(
        maxGaussians: Int = 6_000_000,
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

public protocol GaussianRenderer: AnyObject, Sendable {
    var device: MTLDevice { get }
    var lastGPUTime: Double? { get }

    func render(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        input: GaussianInput,
        camera: CameraParams,
        width: Int,
        height: Int
    )

    func renderStereo(
        commandBuffer: MTLCommandBuffer,
        output: StereoRenderOutput,
        input: GaussianInput,
        camera: StereoCameraParams,
        width: Int,
        height: Int
    )
}

public extension GaussianRenderer {
    func renderStereo(
        commandBuffer: MTLCommandBuffer,
        output: StereoRenderOutput,
        input: GaussianInput,
        camera: StereoCameraParams,
        width: Int,
        height: Int
    ) {
        render(
            commandBuffer: commandBuffer,
            colorTexture: output.leftColor,
            depthTexture: output.leftDepth,
            input: input,
            camera: camera.leftEye,
            width: width,
            height: height
        )

        render(
            commandBuffer: commandBuffer,
            colorTexture: output.rightColor,
            depthTexture: output.rightDepth,
            input: input,
            camera: camera.rightEye,
            width: width,
            height: height
        )
    }
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

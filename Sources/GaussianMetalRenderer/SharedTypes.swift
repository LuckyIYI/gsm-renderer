import Foundation
import simd
import Metal

// MARK: - Common Types

public struct TileAssignmentHeaderSwift {
    public var totalAssignments: UInt32
    public var maxAssignments: UInt32
    public var paddedCount: UInt32
    public var overflow: UInt32

    public init(totalAssignments: UInt32 = 0, maxAssignments: UInt32 = 0, paddedCount: UInt32 = 0, overflow: UInt32 = 0) {
        self.totalAssignments = totalAssignments
        self.maxAssignments = maxAssignments
        self.paddedCount = paddedCount
        self.overflow = overflow
    }
}

public struct CameraUniformsSwift {
    public var viewMatrix: simd_float4x4
    public var projectionMatrix: simd_float4x4
    public var cameraCenter: SIMD3<Float>
    public var pixelFactor: Float
    public var focalX: Float
    public var focalY: Float
    public var width: Float
    public var height: Float
    public var nearPlane: Float
    public var farPlane: Float
    public var shComponents: UInt32
    public var gaussianCount: UInt32
    public var padding0: UInt32 = 0
    public var padding1: UInt32 = 0
    public init(
        viewMatrix: simd_float4x4,
        projectionMatrix: simd_float4x4,
        cameraCenter: SIMD3<Float>,
        pixelFactor: Float,
        focalX: Float,
        focalY: Float,
        width: Float,
        height: Float,
        nearPlane: Float,
        farPlane: Float,
        shComponents: UInt32,
        gaussianCount: UInt32,
        padding0: UInt32 = 0,
        padding1: UInt32 = 0
    ) {
        self.viewMatrix = viewMatrix
        self.projectionMatrix = projectionMatrix
        self.cameraCenter = cameraCenter
        self.pixelFactor = pixelFactor
        self.focalX = focalX
        self.focalY = focalY
        self.width = width
        self.height = height
        self.nearPlane = nearPlane
        self.farPlane = farPlane
        self.shComponents = shComponents
        self.gaussianCount = gaussianCount
        self.padding0 = padding0
        self.padding1 = padding1
    }
}

public struct DispatchIndirectArgsSwift {
    public var threadgroupsPerGridX: UInt32
    public var threadgroupsPerGridY: UInt32
    public var threadgroupsPerGridZ: UInt32
}

public struct AssignmentDispatchConfigSwift {
    public var sortThreadgroupSize: UInt32
    public var fuseThreadgroupSize: UInt32
    public var unpackThreadgroupSize: UInt32
    public var packThreadgroupSize: UInt32
    public var bitonicThreadgroupSize: UInt32
    public var radixBlockSize: UInt32
    public var radixGrainSize: UInt32
    public var maxAssignments: UInt32  // Clamp totalAssignments to this value
}

public enum DispatchSlot: Int, CaseIterable {
    case sortKeys = 0
    case fuseKeys = 1
    case unpackKeys = 2
    case pack = 3
    case bitonicFirst = 4
    case bitonicGeneral = 5
    case bitonicFinal = 6
    case radixHistogram = 7
    case radixScanBlocks = 8
    case radixExclusive = 9
    case radixApply = 10
    case radixScatter = 11
    case renderTiles = 12
}

public struct TileBoundsParamsSwift {
    public var width: UInt32
    public var height: UInt32
    public var tileWidth: UInt32
    public var tileHeight: UInt32
    public var tilesX: UInt32
    public var tilesY: UInt32
    public var gaussianCount: UInt32
}

public struct CoverageParamsSwift {
    public var gaussianCount: UInt32
    public var tileWidth: UInt32 = 0   // For precise intersection (optional, 0 = use AABB only)
    public var tileHeight: UInt32 = 0  // For precise intersection
}

public struct ScatterParamsSwift {
    public var gaussianCount: UInt32
    public var tilesX: UInt32
    public var tileWidth: UInt32 = 0   // For precise intersection
    public var tileHeight: UInt32 = 0  // For precise intersection
}

public struct FusedCoverageScatterParamsSwift {
    public var gaussianCount: UInt32
    public var tileWidth: UInt32
    public var tileHeight: UInt32
    public var tilesX: UInt32
    public var maxAssignments: UInt32

    public init(
        gaussianCount: UInt32,
        tileWidth: UInt32,
        tileHeight: UInt32,
        tilesX: UInt32,
        maxAssignments: UInt32
    ) {
        self.gaussianCount = gaussianCount
        self.tileWidth = tileWidth
        self.tileHeight = tileHeight
        self.tilesX = tilesX
        self.maxAssignments = maxAssignments
    }
}

public struct PackParamsSwift {
    public var totalAssignments: UInt32
    public var padding: UInt32 = 0
}

public struct SortKeyParamsSwift {
    public var maxAssignments: UInt32
    public var totalAssignments: UInt32
}

public struct ClearParamsSwift {
    public var pixelCount: UInt32
    public var whiteBackground: UInt32
}

public struct ClearTextureParamsSwift {
    public var width: UInt32
    public var height: UInt32
    public var whiteBackground: UInt32
}

public struct RenderDispatchParamsSwift {
    public var tileCount: UInt32
    public var totalAssignments: UInt32
    public var gaussianCount: UInt32
}

// MARK: - Buffer Sets

public struct GaussianInputBuffers {
    public let means: MTLBuffer      // float2
    public let radii: MTLBuffer      // float (always float for tile bounds)
    public let mask: MTLBuffer       // uchar
    public let depths: MTLBuffer     // float
    public let conics: MTLBuffer     // float4
    public let colors: MTLBuffer     // packed_float3
    public let opacities: MTLBuffer  // float
}

/// Half-precision gaussian input buffers for the half16 pipeline.
/// Note: radii stays float for accurate tile bounds computation.
public struct GaussianInputBuffersHalf {
    public let means: MTLBuffer      // half2
    public let radii: MTLBuffer      // float (always float for tile bounds)
    public let mask: MTLBuffer       // uchar
    public let depths: MTLBuffer     // half
    public let conics: MTLBuffer     // half4
    public let colors: MTLBuffer     // packed_half3
    public let opacities: MTLBuffer  // half
}

/// Unified gaussian input buffers supporting both precisions
public enum GaussianInputBuffersUnified {
    case float32(GaussianInputBuffers)
    case float16(GaussianInputBuffersHalf)

    public var precision: Precision {
        switch self {
        case .float32: return .float32
        case .float16: return .float16
        }
    }

    public var means: MTLBuffer {
        switch self {
        case .float32(let b): return b.means
        case .float16(let b): return b.means
        }
    }

    public var radii: MTLBuffer {
        switch self {
        case .float32(let b): return b.radii
        case .float16(let b): return b.radii
        }
    }

    public var mask: MTLBuffer {
        switch self {
        case .float32(let b): return b.mask
        case .float16(let b): return b.mask
        }
    }

    public var depths: MTLBuffer {
        switch self {
        case .float32(let b): return b.depths
        case .float16(let b): return b.depths
        }
    }

    public var conics: MTLBuffer {
        switch self {
        case .float32(let b): return b.conics
        case .float16(let b): return b.conics
        }
    }

    public var colors: MTLBuffer {
        switch self {
        case .float32(let b): return b.colors
        case .float16(let b): return b.colors
        }
    }

    public var opacities: MTLBuffer {
        switch self {
        case .float32(let b): return b.opacities
        case .float16(let b): return b.opacities
        }
    }
}

public struct TileAssignmentBuffers {
    public let tileCount: Int
    public let maxAssignments: Int
    public let tileIndices: MTLBuffer
    public let tileIds: MTLBuffer
    public let header: MTLBuffer
}

public struct TileBuilderResources {
    public let boundsBuffer: MTLBuffer
    public let coverageBuffer: MTLBuffer
    public let offsetsBuffer: MTLBuffer
    public let partialSumsBuffer: MTLBuffer
    public let scatterDispatchBuffer: MTLBuffer
    public let tileAssignmentHeader: MTLBuffer
    public var tileIndicesBuffer: MTLBuffer
    public var tileIdsBuffer: MTLBuffer
    public let dispatchArgsBuffer: MTLBuffer
    public let dispatchDebugBuffer: MTLBuffer?
}

public struct OrderedBufferSet {
    public let headers: MTLBuffer
    public let means: MTLBuffer
    public let conics: MTLBuffer
    public let colors: MTLBuffer
    public let opacities: MTLBuffer
    public let depths: MTLBuffer
}

public enum Precision {
    case float32
    case float16
}

public enum SortAlgorithm {
    case bitonic
    case radix
}

public struct OrderedGaussianBuffers {
    public let headers: MTLBuffer
    public let means: MTLBuffer
    public let conics: MTLBuffer
    public let colors: MTLBuffer
    public let opacities: MTLBuffer
    public let depths: MTLBuffer
    public let tileCount: Int
    public let activeTileIndices: MTLBuffer
    public let activeTileCount: MTLBuffer
    public let precision: Precision

    // Fused pipeline buffers (optional - for cache-efficient rendering)
    public let packedGaussiansFused: MTLBuffer?
}

public struct SortBufferSet {
    public let keys: MTLBuffer
    public let indices: MTLBuffer
}

public struct RadixBufferSet {
    public let histogram: MTLBuffer
    public let blockSums: MTLBuffer
    public let scannedHistogram: MTLBuffer
    public let fusedKeys: MTLBuffer
    public let scratchKeys: MTLBuffer
    public let scratchPayload: MTLBuffer
}

public struct RenderOutputBuffers {
    public let colorOutGPU: MTLBuffer
    public let depthOutGPU: MTLBuffer
    public let alphaOutGPU: MTLBuffer
}

public struct RenderOutputTextures {
    public let color: MTLTexture
    public let depth: MTLTexture
    public let alpha: MTLTexture
}

public struct RenderSubmission {
    public let commandBuffer: MTLCommandBuffer
    public let outputs: RenderOutputBuffers
    public let params: RenderParams
}

public struct WorldGaussianBuffers {
    public let positions: MTLBuffer
    public let scales: MTLBuffer
    public let rotations: MTLBuffer
    public let harmonics: MTLBuffer
    public let opacities: MTLBuffer
    public let shComponents: Int
    public init(
        positions: MTLBuffer,
        scales: MTLBuffer,
        rotations: MTLBuffer,
        harmonics: MTLBuffer,
        opacities: MTLBuffer,
        shComponents: Int
    ) {
        self.positions = positions
        self.scales = scales
        self.rotations = rotations
        self.harmonics = harmonics
        self.opacities = opacities
        self.shComponents = shComponents
    }
}

/// Half-precision world gaussian buffers for native float16 input data.
/// Layout: positions (half3), scales (half3), rotations (half4), harmonics (half), opacities (half)
public struct WorldGaussianBuffersHalf {
    public let positions: MTLBuffer    // half3 packed as 3 x UInt16
    public let scales: MTLBuffer       // half3 packed as 3 x UInt16
    public let rotations: MTLBuffer    // half4 packed as 4 x UInt16
    public let harmonics: MTLBuffer    // half (SH coefficients)
    public let opacities: MTLBuffer    // half
    public let shComponents: Int

    public init(
        positions: MTLBuffer,
        scales: MTLBuffer,
        rotations: MTLBuffer,
        harmonics: MTLBuffer,
        opacities: MTLBuffer,
        shComponents: Int
    ) {
        self.positions = positions
        self.scales = scales
        self.rotations = rotations
        self.harmonics = harmonics
        self.opacities = opacities
        self.shComponents = shComponents
    }
}

/// Unified wrapper for world gaussian buffers supporting both precisions.
public enum WorldGaussianBuffersUnified {
    case float32(WorldGaussianBuffers)
    case float16(WorldGaussianBuffersHalf)

    public var precision: Precision {
        switch self {
        case .float32: return .float32
        case .float16: return .float16
        }
    }

    public var shComponents: Int {
        switch self {
        case .float32(let b): return b.shComponents
        case .float16(let b): return b.shComponents
        }
    }
}

public struct ProjectionReadbackBuffers {
    public let meansOut: MTLBuffer
    public let conicsOut: MTLBuffer
    public let colorsOut: MTLBuffer
    public let opacitiesOut: MTLBuffer
    public let depthsOut: MTLBuffer
    public let radiiOut: MTLBuffer
    public let maskOut: MTLBuffer
}

public struct TilePipelineDebugOutPointers {
    public var tileAssignmentHeaderOutPtr: UnsafeMutableRawPointer?
    public var tileIndicesOutPtr: UnsafeMutablePointer<Int32>?
    public var tileIdsOutPtr: UnsafeMutablePointer<Int32>?
    public var totalAssignmentsOutPtr: UnsafeMutablePointer<Int32>?
    public var boundsOutPtr: UnsafeMutablePointer<Int32>?
    public var coverageOutPtr: UnsafeMutablePointer<UInt32>?
    public var offsetsOutPtr: UnsafeMutablePointer<UInt32>?
    
    public init(
        tileAssignmentHeaderOutPtr: UnsafeMutableRawPointer? = nil,
        tileIndicesOutPtr: UnsafeMutablePointer<Int32>? = nil,
        tileIdsOutPtr: UnsafeMutablePointer<Int32>? = nil,
        totalAssignmentsOutPtr: UnsafeMutablePointer<Int32>? = nil,
        boundsOutPtr: UnsafeMutablePointer<Int32>? = nil,
        coverageOutPtr: UnsafeMutablePointer<UInt32>? = nil,
        offsetsOutPtr: UnsafeMutablePointer<UInt32>? = nil
    ) {
        self.tileAssignmentHeaderOutPtr = tileAssignmentHeaderOutPtr
        self.tileIndicesOutPtr = tileIndicesOutPtr
        self.tileIdsOutPtr = tileIdsOutPtr
        self.totalAssignmentsOutPtr = totalAssignmentsOutPtr
        self.boundsOutPtr = boundsOutPtr
        self.coverageOutPtr = coverageOutPtr
        self.offsetsOutPtr = offsetsOutPtr
    }
}

public struct BufferCopyRequest {
    public let source: MTLBuffer
    public let destination: UnsafeMutableRawPointer
    public let length: Int
}

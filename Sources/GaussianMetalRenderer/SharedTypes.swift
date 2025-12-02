import Foundation
import simd
import Metal
import GaussianMetalRendererTypes

// MARK: - Type Aliases from BridgingTypes.h
// All types are defined in BridgingTypes.h and imported via GaussianMetalRendererTypes.
// Swift-side typealiases for API compatibility:

public typealias DispatchIndirectArgsSwift = DispatchIndirectArgs
public typealias AssignmentDispatchConfigSwift = AssignmentDispatchConfig
public typealias TileBoundsParamsSwift = TileBoundsParams
public typealias CoverageParamsSwift = CoverageParams
public typealias ScatterParamsSwift = ScatterParams
public typealias SortKeyParamsSwift = SortKeyParams
public typealias ClearParamsSwift = ClearParams
public typealias ClearTextureParamsSwift = ClearTextureParams
public typealias BitonicParamsSwift = BitonicParams

// DispatchSlot enum (Swift-native with static values matching C enum)
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

// RenderDispatchParamsSwift - Swift-only (no C equivalent)
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

    // Index-based render (like LocalSort): render reads via sortedIndices
    // renderData: AoS packed GaussianRenderData from projectGaussiansAoS
    public let renderData: MTLBuffer?
    public let sortedIndices: MTLBuffer?
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
}

public struct RenderSubmission {
    public let commandBuffer: MTLCommandBuffer
    public let outputs: RenderOutputBuffers
    public let params: RenderParams
}

/* REMOVED: WorldGaussianBuffers, WorldGaussianBuffersHalf, WorldGaussianBuffersUnified
   Use PackedWorldBuffers (defined in ProjectEncoder.swift) instead.
   PackedWorldBuffers contains a single interleaved buffer for optimal memory access.
*/

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

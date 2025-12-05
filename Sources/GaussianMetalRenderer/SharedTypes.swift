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

// DispatchSlot enum (Swift-native with static values matching C enum)
public enum DispatchSlot: Int, CaseIterable {
    case sortKeys = 0
    case pack = 3
    case radixHistogram = 7
    case radixScanBlocks = 8
    case radixExclusive = 9
    case radixApply = 10
    case radixScatter = 11
    case renderTiles = 12
}

// RenderDispatchParamsSwift - used for dispatch computation
public struct RenderDispatchParamsSwift {
    public var tileCount: UInt32
    public var totalAssignments: UInt32
    public var gaussianCount: UInt32
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

// MARK: - Buffer Sets

public struct TileAssignmentBuffers {
    public let tileCount: Int
    public let maxAssignments: Int
    public let tileIndices: MTLBuffer
    public let tileIds: MTLBuffer
    public let header: MTLBuffer
}

public struct OrderedBufferSet {
    public let headers: MTLBuffer
    public let means: MTLBuffer
    public let conics: MTLBuffer
    public let colors: MTLBuffer
    public let opacities: MTLBuffer
    public let depths: MTLBuffer
}

// Precision alias - use RenderPrecision from GaussianRendererProtocol
public typealias Precision = RenderPrecision

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

    // Index-based render (like Local): render reads via sortedIndices
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
    public let scratchKeys: MTLBuffer      // Scratch for ping-pong during radix sort
    public let scratchPayload: MTLBuffer   // Scratch for payload ping-pong
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

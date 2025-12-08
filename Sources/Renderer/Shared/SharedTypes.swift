import Foundation
@preconcurrency import Metal
import RendererTypes
import simd

typealias DispatchIndirectArgsSwift = DispatchIndirectArgs
typealias AssignmentDispatchConfigSwift = AssignmentDispatchConfig
typealias ClearTextureParamsSwift = ClearTextureParams

// DispatchSlot enum (Swift-native with static values matching C enum)
enum DispatchSlot: Int, CaseIterable {
    case sortKeys = 0
    case pack = 3
    case radixHistogram = 4
    case radixScanBlocks = 5
    case radixExclusive = 6
    case radixApply = 7
    case radixScatter = 8
    case renderTiles = 9
}

// RenderDispatchParamsSwift - used for dispatch computation
struct RenderDispatchParamsSwift {
    var tileCount: UInt32
    var totalAssignments: UInt32
    var gaussianCount: UInt32
}

struct TileAssignmentBuffers {
    let tileCount: Int
    let maxAssignments: Int
    let tileIndices: MTLBuffer
    let tileIds: MTLBuffer
    let header: MTLBuffer
}

struct OrderedBufferSet {
    let headers: MTLBuffer
    let means: MTLBuffer
    let conics: MTLBuffer
    let colors: MTLBuffer
    let opacities: MTLBuffer
    let depths: MTLBuffer
}

// Precision alias - use RenderPrecision from GaussianRendererProtocol
typealias Precision = RenderPrecision

struct OrderedGaussianBuffers {
    let headers: MTLBuffer
    let means: MTLBuffer
    let conics: MTLBuffer
    let colors: MTLBuffer
    let opacities: MTLBuffer
    let depths: MTLBuffer
    let tileCount: Int
    let activeTileIndices: MTLBuffer
    let activeTileCount: MTLBuffer
    let precision: Precision

    // Index-based render (like Local): render reads via sortedIndices
    // renderData: AoS packed GaussianRenderData from projectGaussiansAoS
    let renderData: MTLBuffer?
    let sortedIndices: MTLBuffer?
}

struct RadixBufferSet {
    let histogram: MTLBuffer
    let blockSums: MTLBuffer
    let scannedHistogram: MTLBuffer
    let scratchKeys: MTLBuffer // Scratch for ping-pong during radix sort
    let scratchPayload: MTLBuffer // Scratch for payload ping-pong
}

public struct RenderOutputTextures: Sendable {
    public let color: MTLTexture
    public let depth: MTLTexture

    public init(color: MTLTexture, depth: MTLTexture) {
        self.color = color
        self.depth = depth
    }
}

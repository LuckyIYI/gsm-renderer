import Foundation
import simd

public struct GaussianHeader {
    public var offset: UInt32
    public var count: UInt32
}

/// Simple per-frame render settings (caller only needs these at runtime)
public struct FrameParams {
    public var gaussianCount: Int
    public var whiteBackground: Bool

    public init(gaussianCount: Int, whiteBackground: Bool = false) {
        self.gaussianCount = gaussianCount
        self.whiteBackground = whiteBackground
    }
}

// =============================================================================
// PACKED WORLD GAUSSIAN - Interleaved input for projection kernel
// Caller must provide data in this format for optimal GPU memory access
// =============================================================================

/// Packed world gaussian (float32) - 48 bytes, matches Metal struct layout
/// Contains: position (12), opacity (4), scale (12), pad (4), rotation (16)
///
/// IMPORTANT: Uses raw floats for position/scale to match Metal's packed_float3 (12 bytes).
/// Swift's SIMD3<Float> is 16 bytes (padded), which would cause layout mismatch!
///
/// Usage:
/// ```swift
/// let packed = PackedWorldGaussian(
///     position: SIMD3<Float>(x, y, z),
///     scale: SIMD3<Float>(sx, sy, sz),
///     rotation: SIMD4<Float>(qx, qy, qz, qw),
///     opacity: alpha
/// )
/// ```
public struct PackedWorldGaussian {
    // Position as packed_float3 (12 bytes)
    public var px: Float
    public var py: Float
    public var pz: Float
    // Opacity (4 bytes)
    public var opacity: Float
    // Scale as packed_float3 (12 bytes)
    public var sx: Float
    public var sy: Float
    public var sz: Float
    // Padding (4 bytes)
    public var _pad0: Float
    // Rotation quaternion (16 bytes) - SIMD4 is 16 bytes, matches float4
    public var rotation: SIMD4<Float>

    public init(position: SIMD3<Float>, scale: SIMD3<Float>, rotation: SIMD4<Float>, opacity: Float) {
        self.px = position.x
        self.py = position.y
        self.pz = position.z
        self.opacity = opacity
        self.sx = scale.x
        self.sy = scale.y
        self.sz = scale.z
        self._pad0 = 0
        self.rotation = rotation
    }

    public var position: SIMD3<Float> {
        get { SIMD3(px, py, pz) }
        set { px = newValue.x; py = newValue.y; pz = newValue.z }
    }

    public var scale: SIMD3<Float> {
        get { SIMD3(sx, sy, sz) }
        set { sx = newValue.x; sy = newValue.y; sz = newValue.z }
    }
}

/// Packed world gaussian (half16) - 24 bytes, matches Metal struct layout
/// Contains: position (6), opacity (2), scale (6), pad (2), rotation (8)
///
/// IMPORTANT: Uses Float16 (half) for position/scale/rotation.
/// This struct is half the size of PackedWorldGaussian for memory-bound workloads.
///
/// Usage:
/// ```swift
/// let packed = PackedWorldGaussianHalf(
///     position: SIMD3<Float>(x, y, z),  // Converts to half
///     scale: SIMD3<Float>(sx, sy, sz),
///     rotation: SIMD4<Float>(qx, qy, qz, qw),
///     opacity: alpha
/// )
/// ```
public struct PackedWorldGaussianHalf {
    // Position as packed_half3 (6 bytes)
    public var px: Float16
    public var py: Float16
    public var pz: Float16
    // Opacity (2 bytes)
    public var opacity: Float16
    // Scale as packed_half3 (6 bytes)
    public var sx: Float16
    public var sy: Float16
    public var sz: Float16
    // Padding (2 bytes)
    public var _pad0: Float16
    // Rotation quaternion (8 bytes) - 4x Float16
    public var rx: Float16
    public var ry: Float16
    public var rz: Float16
    public var rw: Float16

    public init(position: SIMD3<Float>, scale: SIMD3<Float>, rotation: SIMD4<Float>, opacity: Float) {
        self.px = Float16(position.x)
        self.py = Float16(position.y)
        self.pz = Float16(position.z)
        self.opacity = Float16(opacity)
        self.sx = Float16(scale.x)
        self.sy = Float16(scale.y)
        self.sz = Float16(scale.z)
        self._pad0 = 0
        self.rx = Float16(rotation.x)
        self.ry = Float16(rotation.y)
        self.rz = Float16(rotation.z)
        self.rw = Float16(rotation.w)
    }

    /// Direct init from Float16 values (no conversion)
    public init(px: Float16, py: Float16, pz: Float16, opacity: Float16,
                sx: Float16, sy: Float16, sz: Float16,
                rx: Float16, ry: Float16, rz: Float16, rw: Float16) {
        self.px = px
        self.py = py
        self.pz = pz
        self.opacity = opacity
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self._pad0 = 0
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.rw = rw
    }

    public var position: SIMD3<Float> {
        get { SIMD3(Float(px), Float(py), Float(pz)) }
        set { px = Float16(newValue.x); py = Float16(newValue.y); pz = Float16(newValue.z) }
    }

    public var scale: SIMD3<Float> {
        get { SIMD3(Float(sx), Float(sy), Float(sz)) }
        set { sx = Float16(newValue.x); sy = Float16(newValue.y); sz = Float16(newValue.z) }
    }

    public var rotation: SIMD4<Float> {
        get { SIMD4(Float(rx), Float(ry), Float(rz), Float(rw)) }
        set { rx = Float16(newValue.x); ry = Float16(newValue.y); rz = Float16(newValue.z); rw = Float16(newValue.w) }
    }
}

/// Internal struct matching Metal RenderParams layout
/// Built automatically from RendererLimits + FrameParams
public struct RenderParams {
    public var width: UInt32
    public var height: UInt32
    public var tileWidth: UInt32
    public var tileHeight: UInt32
    public var tilesX: UInt32
    public var tilesY: UInt32
    public var maxPerTile: UInt32
    public var whiteBackground: UInt32
    public var activeTileCount: UInt32
    public var gaussianCount: UInt32

    public init(
        width: UInt32,
        height: UInt32,
        tileWidth: UInt32,
        tileHeight: UInt32,
        tilesX: UInt32,
        tilesY: UInt32,
        maxPerTile: UInt32,
        whiteBackground: UInt32,
        activeTileCount: UInt32,
        gaussianCount: UInt32
    ) {
        self.width = width
        self.height = height
        self.tileWidth = tileWidth
        self.tileHeight = tileHeight
        self.tilesX = tilesX
        self.tilesY = tilesY
        self.maxPerTile = maxPerTile
        self.whiteBackground = whiteBackground
        self.activeTileCount = activeTileCount
        self.gaussianCount = gaussianCount
    }
}

/// Swift struct matching Metal GaussianRenderData (24 bytes)
/// Layout: half2 mean (4) + half4 conic (8) + packed_half3 color (6) + half opacity (2) + half depth (2) + ushort pad (2)
/// Note: In Metal, half vectors only need 2-byte alignment (not size-based), so half4 packs directly after half2
public struct GaussianRenderDataSwift {
    // half2 mean - 4 bytes @ 0
    public var meanX: Float16
    public var meanY: Float16
    // half4 conic - 8 bytes @ 4
    public var conicA: Float16
    public var conicB: Float16
    public var conicC: Float16
    public var conicD: Float16
    // packed_half3 color - 6 bytes @ 12
    public var colorR: Float16
    public var colorG: Float16
    public var colorB: Float16
    // half opacity - 2 bytes @ 18
    public var opacity: Float16
    // half depth - 2 bytes @ 20
    public var depth: Float16
    // ushort _pad - 2 bytes @ 22
    public var _pad: UInt16

    public var mean: SIMD2<Float> {
        get { SIMD2(Float(meanX), Float(meanY)) }
    }
    public var conic: SIMD4<Float> {
        get { SIMD4(Float(conicA), Float(conicB), Float(conicC), Float(conicD)) }
    }
    public var color: SIMD3<Float> {
        get { SIMD3(Float(colorR), Float(colorG), Float(colorB)) }
    }
}

// =============================================================================
// LOCAL-STYLE PIPELINE TYPES
// =============================================================================

/// Compacted gaussian for local sort pipeline (matches Metal struct)
/// 48 bytes total
public struct CompactedGaussianSwift {
    public var covariance_depth: SIMD4<Float>  // conic.xyz + depth
    public var position_color: SIMD4<Float>    // pos.xy + packed half4(color, opacity)
    public var min_tile: SIMD2<Int32>
    public var max_tile: SIMD2<Int32>

    public init() {
        self.covariance_depth = .zero
        self.position_color = .zero
        self.min_tile = .zero
        self.max_tile = .zero
    }
}

/// Parameters for fused project+compact+count kernel
public struct ProjectCompactParamsSwift {
    public var gaussianCount: UInt32
    public var tilesX: UInt32
    public var tilesY: UInt32
    public var tileWidth: UInt32
    public var tileHeight: UInt32
    public var surfaceWidth: UInt32
    public var surfaceHeight: UInt32
    public var maxCompacted: UInt32

    public init(
        gaussianCount: UInt32,
        tilesX: UInt32,
        tilesY: UInt32,
        tileWidth: UInt32,
        tileHeight: UInt32,
        surfaceWidth: UInt32,
        surfaceHeight: UInt32,
        maxCompacted: UInt32
    ) {
        self.gaussianCount = gaussianCount
        self.tilesX = tilesX
        self.tilesY = tilesY
        self.tileWidth = tileWidth
        self.tileHeight = tileHeight
        self.surfaceWidth = surfaceWidth
        self.surfaceHeight = surfaceHeight
        self.maxCompacted = maxCompacted
    }
}

/// Header for compacted gaussians (matches TileAssignmentHeaderAtomic in Metal)
/// Layout must match: [visibleCount][maxCapacity][paddedCount][overflow]
public struct CompactedHeaderSwift {
    public var visibleCount: UInt32    // Atomic counter (read as UInt32 on CPU)
    public var maxCapacity: UInt32     // Max compacted capacity
    public var paddedCount: UInt32     // Padded count for radix sort
    public var overflow: UInt32        // Overflow flag

    public init(visibleCount: UInt32 = 0, maxCapacity: UInt32 = 0, paddedCount: UInt32 = 0, overflow: UInt32 = 0) {
        self.visibleCount = visibleCount
        self.maxCapacity = maxCapacity
        self.paddedCount = paddedCount
        self.overflow = overflow
    }
}

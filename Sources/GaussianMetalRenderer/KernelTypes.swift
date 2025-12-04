import Foundation
import simd
@_exported import GaussianMetalRendererTypes

// =============================================================================
// TYPE ALIASES FOR API COMPATIBILITY
// All types come from BridgingTypes.h via GaussianMetalRendererTypes
// =============================================================================

public typealias CameraUniformsSwift = CameraUniforms
public typealias RenderParamsSwift = RenderParams
public typealias TileBinningParamsSwift = TileBinningParams
public typealias TileAssignmentHeaderSwift = TileAssignmentHeader
public typealias CompactedHeaderSwift = TileAssignmentHeader
public typealias CompactedGaussianSwift = CompactedGaussian

// =============================================================================
// CONVENIENCE EXTENSIONS
// =============================================================================

extension PackedWorldGaussian {
    public init(position: SIMD3<Float>, scale: SIMD3<Float>, rotation: SIMD4<Float>, opacity: Float) {
        self.init()
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

extension CameraUniforms {
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
        self.init()
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

extension RenderParams {
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
        self.init()
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

extension TileAssignmentHeader {
    public init(visibleCount: UInt32 = 0, maxCapacity: UInt32 = 0, paddedCount: UInt32 = 0, overflow: UInt32 = 0) {
        self.init()
        self.totalAssignments = visibleCount
        self.maxCapacity = maxCapacity
        self.paddedCount = paddedCount
        self.overflow = overflow
    }

    public var visibleCount: UInt32 {
        get { totalAssignments }
        set { totalAssignments = newValue }
    }
}

extension TileBinningParams {
    public init(
        gaussianCount: UInt32,
        tilesX: UInt32,
        tilesY: UInt32,
        tileWidth: UInt32,
        tileHeight: UInt32,
        surfaceWidth: UInt32,
        surfaceHeight: UInt32,
        maxCapacity: UInt32,
        alphaThreshold: Float = 0.004,
        minCoverageRatio: Float = 0.05
    ) {
        self.init()
        self.gaussianCount = gaussianCount
        self.tilesX = tilesX
        self.tilesY = tilesY
        self.tileWidth = tileWidth
        self.tileHeight = tileHeight
        self.surfaceWidth = surfaceWidth
        self.surfaceHeight = surfaceHeight
        self.maxCapacity = maxCapacity
        self.alphaThreshold = alphaThreshold
        self.minCoverageRatio = minCoverageRatio
    }
}

// =============================================================================
// SWIFT-ONLY TYPES (Float16 - no C equivalent with correct alignment)
// =============================================================================

public struct FrameParams {
    public var gaussianCount: Int
    public var whiteBackground: Bool

    public init(gaussianCount: Int, whiteBackground: Bool = false) {
        self.gaussianCount = gaussianCount
        self.whiteBackground = whiteBackground
    }
}

/// Half-precision world gaussian (24 bytes) - Swift side
public struct PackedWorldGaussianHalf {
    public var px: Float16
    public var py: Float16
    public var pz: Float16
    public var opacity: Float16
    public var sx: Float16
    public var sy: Float16
    public var sz: Float16
    public var _pad0: Float16
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

/// Half-precision render data (32 bytes) - Swift side for reading GPU output
public struct GaussianRenderDataSwift {
    public var meanX: Float16
    public var meanY: Float16
    public var _alignPad: UInt32
    public var conicA: Float16
    public var conicB: Float16
    public var conicC: Float16
    public var conicD: Float16
    public var colorR: Float16
    public var colorG: Float16
    public var colorB: Float16
    public var opacity: Float16
    public var depth: Float16
    public var _pad: UInt16
    public var _structPad: UInt32

    public var mean: SIMD2<Float> { SIMD2(Float(meanX), Float(meanY)) }
    public var conic: SIMD4<Float> { SIMD4(Float(conicA), Float(conicB), Float(conicC), Float(conicD)) }
    public var color: SIMD3<Float> { SIMD3(Float(colorR), Float(colorG), Float(colorB)) }
}

// ProjectCompactParamsSwift removed - use TileBinningParams from C header instead

import Foundation
@_exported import RendererTypes
import simd

typealias CameraUniformsSwift = CameraUniforms
typealias RenderParamsSwift = RenderParams
typealias TileBinningParamsSwift = TileBinningParams
typealias TileAssignmentHeaderSwift = TileAssignmentHeader
typealias CompactedHeaderSwift = TileAssignmentHeader
typealias ProjectedGaussianSwift = ProjectedGaussian

public extension PackedWorldGaussian {
    init(position: SIMD3<Float>, scale: SIMD3<Float>, rotation: SIMD4<Float>, opacity: Float) {
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

    var position: SIMD3<Float> {
        get { SIMD3(px, py, pz) }
        set { px = newValue.x; py = newValue.y; pz = newValue.z }
    }

    var scale: SIMD3<Float> {
        get { SIMD3(sx, sy, sz) }
        set { sx = newValue.x; sy = newValue.y; sz = newValue.z }
    }
}

public extension PackedWorldGaussianHalf {
    init(position: SIMD3<Float>, scale: SIMD3<Float>, rotation: SIMD4<Float>, opacity: Float) {
        self.init()
        self.px = position.x
        self.py = position.y
        self.pz = position.z
        self.opacity = Float16(opacity)
        self.sx = Float16(scale.x)
        self.sy = Float16(scale.y)
        self.sz = Float16(scale.z)
        self.rx = Float16(rotation.x)
        self.ry = Float16(rotation.y)
        self.rz = Float16(rotation.z)
        self.rw = Float16(rotation.w)
        self._pad0 = 0
        self._pad1 = 0
    }

    var position: SIMD3<Float> {
        get { SIMD3(px, py, pz) }
        set { px = newValue.x; py = newValue.y; pz = newValue.z }
    }

    var scale: SIMD3<Float> {
        get { SIMD3(Float(sx), Float(sy), Float(sz)) }
        set { sx = Float16(newValue.x); sy = Float16(newValue.y); sz = Float16(newValue.z) }
    }

    var rotation: SIMD4<Float> {
        get { SIMD4(Float(rx), Float(ry), Float(rz), Float(rw)) }
        set { rx = Float16(newValue.x); ry = Float16(newValue.y); rz = Float16(newValue.z); rw = Float16(newValue.w) }
    }
}

extension CameraUniforms {
    init(
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

    /// Convenience initializer from CameraParams
    init(from camera: CameraParams, width: Int, height: Int, gaussianCount: Int, shComponents: Int) {
        self.init(
            viewMatrix: camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            cameraCenter: camera.position,
            pixelFactor: 1.0,
            focalX: camera.focalX,
            focalY: camera.focalY,
            width: Float(width),
            height: Float(height),
            nearPlane: camera.near,
            farPlane: camera.far,
            shComponents: UInt32(shComponents),
            gaussianCount: UInt32(gaussianCount)
        )
    }
}

extension RenderParams {
    init(
        width: UInt32,
        height: UInt32,
        tileWidth: UInt32,
        tileHeight: UInt32,
        tilesX: UInt32,
        tilesY: UInt32,
        maxPerTile: UInt32,
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
        self.activeTileCount = activeTileCount
        self.gaussianCount = gaussianCount
    }
}

extension TileAssignmentHeader {
    init(visibleCount: UInt32 = 0, maxCapacity: UInt32 = 0, paddedCount: UInt32 = 0, overflow: UInt32 = 0) {
        self.init()
        self.totalAssignments = visibleCount
        self.maxCapacity = maxCapacity
        self.paddedCount = paddedCount
        self.overflow = overflow
    }

    var visibleCount: UInt32 {
        get { totalAssignments }
        set { totalAssignments = newValue }
    }
}

extension TileBinningParams {
    init(
        gaussianCount: UInt32,
        tilesX: UInt32,
        tilesY: UInt32,
        tileWidth: UInt32,
        tileHeight: UInt32,
        surfaceWidth: UInt32,
        surfaceHeight: UInt32,
        maxCapacity: UInt32,
        alphaThreshold: Float = 0.005,
        totalInkThreshold: Float = 3.0
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
        self.totalInkThreshold = totalInkThreshold
    }
}

struct FrameParams {
    var gaussianCount: Int

    init(gaussianCount: Int) {
        self.gaussianCount = gaussianCount
    }
}

extension TileAssignParams {
    init(
        gaussianCount: UInt32,
        tileWidth: UInt32,
        tileHeight: UInt32,
        tilesX: UInt32,
        maxAssignments: UInt32
    ) {
        self.init()
        self.gaussianCount = gaussianCount
        self.tileWidth = tileWidth
        self.tileHeight = tileHeight
        self.tilesX = tilesX
        self.maxAssignments = maxAssignments
    }
}

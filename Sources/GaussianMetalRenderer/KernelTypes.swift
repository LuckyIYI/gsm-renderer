import Foundation

public struct GaussianHeader {
    public var offset: UInt32
    public var count: UInt32
}

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

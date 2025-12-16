import Metal
import RendererTypes

// MARK: - LocalRenderer Eye Resources

/// Resources for a single view in LocalRenderer's pipeline.
/// Allocates exactly what LocalRenderer needs - no more, no less.
public final class LocalViewResources {
    public let device: MTLDevice
    public let label: String

    // Pipeline control
    public let header: MTLBuffer // CompactedHeaderSwift
    public let tileCounts: MTLBuffer // Per-tile gaussian counts
    public let tileOffsets: MTLBuffer // Prefix sum of tile counts
    public let partialSums: MTLBuffer // Scratch for prefix sum
    public let activeTileIndices: MTLBuffer
    public let activeTileCount: MTLBuffer
    public let dispatchArgs: MTLBuffer

    // Projection & compaction
    public let tempProjection: MTLBuffer // Raw projected gaussians
    public let compactedGaussians: MTLBuffer // Compacted visible gaussians
    public let visibilityMarks: MTLBuffer // Visibility flags
    public let visibilityPartialSums: MTLBuffer

    // Per-tile sort (16-bit)
    public let sortIndices: MTLBuffer // Global indices per tile assignment
    public let depthKeys: MTLBuffer // 16-bit depth keys
    public let sortedLocalIdx: MTLBuffer // 16-bit sorted local indices

    public init(
        device: MTLDevice,
        maxGaussians: Int,
        maxTileCount: Int,
        maxAssignments: Int,
        label: String
    ) throws {
        self.device = device
        self.label = label

        let priv: MTLResourceOptions = .storageModePrivate
        let shared: MTLResourceOptions = .storageModeShared

        self.header = try device.makeBuffer(
            count: 1,
            type: CompactedHeaderSwift.self,
            options: shared,
            label: "\(label)/Header"
        )

        self.tileCounts = try device.makeBuffer(
            count: maxTileCount,
            type: UInt32.self,
            options: priv,
            label: "\(label)/TileCounts"
        )

        self.tileOffsets = try device.makeBuffer(
            count: maxTileCount + 1,
            type: UInt32.self,
            options: priv,
            label: "\(label)/TileOffsets"
        )

        self.partialSums = try device.makeBuffer(
            count: 1024,
            type: UInt32.self,
            options: priv,
            label: "\(label)/PartialSums"
        )

        self.activeTileIndices = try device.makeBuffer(
            count: maxTileCount,
            type: UInt32.self,
            options: priv,
            label: "\(label)/ActiveTileIndices"
        )

        self.activeTileCount = try device.makeBuffer(
            count: 1,
            type: UInt32.self,
            options: priv,
            label: "\(label)/ActiveTileCount"
        )

        self.dispatchArgs = try device.makeBuffer(
            count: 3,
            type: UInt32.self,
            options: priv,
            label: "\(label)/DispatchArgs"
        )

        self.tempProjection = try device.makeBuffer(
            count: maxGaussians,
            type: ProjectedGaussianSwift.self,
            options: priv,
            label: "\(label)/TempProjection"
        )

        self.compactedGaussians = try device.makeBuffer(
            count: maxGaussians,
            type: ProjectedGaussianSwift.self,
            options: priv,
            label: "\(label)/CompactedGaussians"
        )

        self.visibilityMarks = try device.makeBuffer(
            count: maxGaussians + 1,
            type: UInt32.self,
            options: priv,
            label: "\(label)/VisibilityMarks"
        )

        let visPartialCount = ((maxGaussians + 1 + 255) / 256) * 2 + 256
        self.visibilityPartialSums = try device.makeBuffer(
            count: visPartialCount,
            type: UInt32.self,
            options: priv,
            label: "\(label)/VisibilityPartialSums"
        )

        self.sortIndices = try device.makeBuffer(
            count: maxAssignments,
            type: UInt32.self,
            options: priv,
            label: "\(label)/SortIndices"
        )

        guard let depthKeys = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt16>.stride,
            options: priv
        ) else {
            throw RendererError.failedToAllocateBuffer(
                label: "\(label)/DepthKeys",
                size: maxAssignments * MemoryLayout<UInt16>.stride
            )
        }
        depthKeys.label = "\(label)/DepthKeys"
        self.depthKeys = depthKeys

        guard let sortedLocalIdx = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt16>.stride,
            options: priv
        ) else {
            throw RendererError.failedToAllocateBuffer(
                label: "\(label)/SortedLocalIdx",
                size: maxAssignments * MemoryLayout<UInt16>.stride
            )
        }
        sortedLocalIdx.label = "\(label)/SortedLocalIdx"
        self.sortedLocalIdx = sortedLocalIdx
    }
}

// MARK: - LocalRenderer Stereo Resources

/// Stereo resources for LocalRenderer - left and right view buffer sets.
public final class LocalMultiViewResources {
    public let device: MTLDevice
    public let left: LocalViewResources
    public let right: LocalViewResources

    public init(
        device: MTLDevice,
        maxGaussians: Int,
        maxWidth: Int,
        maxHeight: Int,
        tileWidth: Int = 16,
        tileHeight: Int = 16,
        maxPerTile: Int = 2048
    ) throws {
        self.device = device

        let tilesX = (maxWidth + tileWidth - 1) / tileWidth
        let tilesY = (maxHeight + tileHeight - 1) / tileHeight
        let maxTileCount = tilesX * tilesY
        let maxAssignments = maxTileCount * maxPerTile

        self.left = try LocalViewResources(
            device: device,
            maxGaussians: maxGaussians,
            maxTileCount: maxTileCount,
            maxAssignments: maxAssignments,
            label: "Left"
        )

        self.right = try LocalViewResources(
            device: device,
            maxGaussians: maxGaussians,
            maxTileCount: maxTileCount,
            maxAssignments: maxAssignments,
            label: "Right"
        )
    }
}

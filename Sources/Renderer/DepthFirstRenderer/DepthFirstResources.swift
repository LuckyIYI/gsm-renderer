import Metal
import RendererTypes

/// Resources for a single view in DepthFirstRenderer's pipeline.
/// Allocates exactly what DepthFirstRenderer needs - direct allocations like LocalViewResources.
final class DepthFirstViewResources {
    let device: MTLDevice

    // Per-gaussian buffers (sized to maxGaussians)
    let renderData: MTLBuffer
    let bounds: MTLBuffer
    let depthKeys: MTLBuffer
    let primitiveIndices: MTLBuffer
    let nTouchedTiles: MTLBuffer
    let orderedTileCounts: MTLBuffer

    // Counters (shared for CPU readback)
    let visibleCount: MTLBuffer
    let totalInstances: MTLBuffer
    let activeTileCount: MTLBuffer

    // Header for dispatch computation
    let header: MTLBuffer
    let dispatchArgs: MTLBuffer

    // Per-instance buffers (sized to maxInstances)
    let instanceTileIds: MTLBuffer
    let instanceGaussianIndices: MTLBuffer

    // Tile headers (sized to tileCount)
    let tileHeaders: MTLBuffer
    let activeTiles: MTLBuffer

    // Radix sort scratch buffers (for depth sort)
    let depthSortHistogram: MTLBuffer
    let depthSortBlockSums: MTLBuffer
    let depthSortScannedHist: MTLBuffer
    let depthSortScratchKeys: MTLBuffer
    let depthSortScratchPayload: MTLBuffer

    // Radix sort scratch buffers (for tile sort)
    let tileSortHistogram: MTLBuffer
    let tileSortBlockSums: MTLBuffer
    let tileSortScannedHist: MTLBuffer
    let tileSortScratchTileIds: MTLBuffer
    let tileSortScratchIndices: MTLBuffer

    // Prefix sum block sums
    let prefixSumBlockSums: MTLBuffer

    // Output textures
    var outputTextures: RenderOutputTextures

    let maxGaussians: Int
    let maxInstances: Int
    let tileCount: Int

    init(
        device: MTLDevice,
        maxGaussians: Int,
        maxWidth: Int,
        maxHeight: Int,
        tileWidth: Int,
        tileHeight: Int,
        radixBlockSize: Int,
        radixGrainSize: Int
    ) throws {
        self.device = device
        self.maxGaussians = maxGaussians

        let priv: MTLResourceOptions = .storageModePrivate
        let shared: MTLResourceOptions = .storageModeShared

        // Calculate derived sizes
        let tilesX = (maxWidth + tileWidth - 1) / tileWidth
        let tilesY = (maxHeight + tileHeight - 1) / tileHeight
        let tileCount = max(1, tilesX * tilesY)
        self.tileCount = tileCount

        let maxInstances = maxGaussians * 8
        self.maxInstances = maxInstances

        // Padded capacities for radix sort alignment
        let radixAlignment = radixBlockSize * radixGrainSize
        let paddedGaussianCapacity = ((maxGaussians + radixAlignment - 1) / radixAlignment) * radixAlignment
        let paddedInstanceCapacity = ((maxInstances + radixAlignment - 1) / radixAlignment) * radixAlignment

        // Radix sort histogram sizing
        let radix = 256
        let gaussianGridSize = max(1, (paddedGaussianCapacity + radixAlignment - 1) / radixAlignment)
        let instanceGridSize = max(1, (paddedInstanceCapacity + radixAlignment - 1) / radixAlignment)

        // Prefix sum block sums sizing
        let prefixBlockSize = 256
        let prefixNumBlocks = (maxGaussians + prefixBlockSize - 1) / prefixBlockSize
        let prefixLevel2Blocks = (prefixNumBlocks + prefixBlockSize - 1) / prefixBlockSize
        let prefixBlockSumsCount = prefixNumBlocks + 1 + prefixLevel2Blocks + 1

        // Per-gaussian buffers
        self.renderData = try device.makeBuffer(
            count: maxGaussians,
            type: GaussianRenderData.self,
            options: priv,
            label: "RenderData"
        )

        self.bounds = try device.makeBuffer(
            count: maxGaussians,
            type: SIMD4<Int32>.self,
            options: priv,
            label: "Bounds"
        )

        self.depthKeys = try device.makeBuffer(
            count: paddedGaussianCapacity,
            type: UInt32.self,
            options: priv,
            label: "DepthKeys"
        )

        self.primitiveIndices = try device.makeBuffer(
            count: paddedGaussianCapacity,
            type: Int32.self,
            options: priv,
            label: "PrimitiveIndices"
        )

        self.nTouchedTiles = try device.makeBuffer(
            count: maxGaussians,
            type: UInt32.self,
            options: priv,
            label: "NTouchedTiles"
        )

        self.orderedTileCounts = try device.makeBuffer(
            count: maxGaussians,
            type: UInt32.self,
            options: priv,
            label: "OrderedTileCounts"
        )

        // Counters (shared for CPU readback)
        self.visibleCount = try device.makeBuffer(
            count: 1,
            type: UInt32.self,
            options: shared,
            label: "VisibleCount"
        )

        self.totalInstances = try device.makeBuffer(
            count: 1,
            type: UInt32.self,
            options: shared,
            label: "TotalInstances"
        )

        self.activeTileCount = try device.makeBuffer(
            count: 1,
            type: UInt32.self,
            options: shared,
            label: "ActiveTileCount"
        )

        // Header (32 bytes for DepthFirstHeader)
        guard let header = device.makeBuffer(length: 32, options: shared) else {
            throw RendererError.failedToAllocateBuffer(label: "Header", size: 32)
        }
        header.label = "Header"
        self.header = header

        self.dispatchArgs = try device.makeBuffer(
            count: DepthFirstDispatchSlot.count,
            type: DispatchIndirectArgsSwift.self,
            options: priv,
            label: "DispatchArgs"
        )

        // Per-instance buffers (tile IDs are 16-bit)
        guard let instanceTileIds = device.makeBuffer(
            length: paddedInstanceCapacity * MemoryLayout<UInt16>.stride,
            options: priv
        ) else {
            throw RendererError.failedToAllocateBuffer(
                label: "InstanceTileIds",
                size: paddedInstanceCapacity * MemoryLayout<UInt16>.stride
            )
        }
        instanceTileIds.label = "InstanceTileIds"
        self.instanceTileIds = instanceTileIds

        self.instanceGaussianIndices = try device.makeBuffer(
            count: paddedInstanceCapacity,
            type: Int32.self,
            options: priv,
            label: "InstanceGaussianIndices"
        )

        // Tile headers
        self.tileHeaders = try device.makeBuffer(
            count: tileCount,
            type: GaussianHeader.self,
            options: priv,
            label: "TileHeaders"
        )

        self.activeTiles = try device.makeBuffer(
            count: tileCount,
            type: UInt32.self,
            options: priv,
            label: "ActiveTiles"
        )

        // Depth sort buffers
        self.depthSortHistogram = try device.makeBuffer(
            count: gaussianGridSize * radix,
            type: UInt32.self,
            options: priv,
            label: "DepthSortHistogram"
        )

        self.depthSortBlockSums = try device.makeBuffer(
            count: gaussianGridSize,
            type: UInt32.self,
            options: priv,
            label: "DepthSortBlockSums"
        )

        self.depthSortScannedHist = try device.makeBuffer(
            count: gaussianGridSize * radix,
            type: UInt32.self,
            options: priv,
            label: "DepthSortScannedHist"
        )

        self.depthSortScratchKeys = try device.makeBuffer(
            count: paddedGaussianCapacity,
            type: UInt32.self,
            options: priv,
            label: "DepthSortScratchKeys"
        )

        self.depthSortScratchPayload = try device.makeBuffer(
            count: paddedGaussianCapacity,
            type: Int32.self,
            options: priv,
            label: "DepthSortScratchPayload"
        )

        // Tile sort buffers
        self.tileSortHistogram = try device.makeBuffer(
            count: instanceGridSize * radix,
            type: UInt32.self,
            options: priv,
            label: "TileSortHistogram"
        )

        self.tileSortBlockSums = try device.makeBuffer(
            count: instanceGridSize,
            type: UInt32.self,
            options: priv,
            label: "TileSortBlockSums"
        )

        self.tileSortScannedHist = try device.makeBuffer(
            count: instanceGridSize * radix,
            type: UInt32.self,
            options: priv,
            label: "TileSortScannedHist"
        )

        guard let tileSortScratchTileIds = device.makeBuffer(
            length: paddedInstanceCapacity * MemoryLayout<UInt16>.stride,
            options: priv
        ) else {
            throw RendererError.failedToAllocateBuffer(
                label: "TileSortScratchTileIds",
                size: paddedInstanceCapacity * MemoryLayout<UInt16>.stride
            )
        }
        tileSortScratchTileIds.label = "TileSortScratchTileIds"
        self.tileSortScratchTileIds = tileSortScratchTileIds

        self.tileSortScratchIndices = try device.makeBuffer(
            count: paddedInstanceCapacity,
            type: Int32.self,
            options: priv,
            label: "TileSortScratchIndices"
        )

        // Prefix sum block sums
        self.prefixSumBlockSums = try device.makeBuffer(
            count: prefixBlockSumsCount,
            type: UInt32.self,
            options: priv,
            label: "PrefixSumBlockSums"
        )

        // Output textures
        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: maxWidth,
            height: maxHeight,
            mipmapped: false
        )
        texDesc.usage = [.shaderWrite, .shaderRead]
        texDesc.storageMode = .private
        guard let colorTex = device.makeTexture(descriptor: texDesc) else {
            throw RendererError.failedToAllocateTexture(
                label: "OutputColorTex",
                width: maxWidth,
                height: maxHeight
            )
        }
        colorTex.label = "DepthFirstOutputColorTex"

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: maxWidth,
            height: maxHeight,
            mipmapped: false
        )
        depthDesc.usage = [.shaderWrite, .shaderRead]
        depthDesc.storageMode = .private
        guard let depthTex = device.makeTexture(descriptor: depthDesc) else {
            throw RendererError.failedToAllocateTexture(
                label: "OutputDepthTex",
                width: maxWidth,
                height: maxHeight
            )
        }
        depthTex.label = "DepthFirstOutputDepthTex"

        self.outputTextures = RenderOutputTextures(color: colorTex, depth: depthTex)
    }
}

/// Stereo resources for DepthFirstRenderer - left and right view buffer sets.
final class DepthFirstMultiViewResources {
    let device: MTLDevice
    let left: DepthFirstViewResources
    let right: DepthFirstViewResources

    init(
        device: MTLDevice,
        maxGaussians: Int,
        maxWidth: Int,
        maxHeight: Int,
        tileWidth: Int,
        tileHeight: Int,
        radixBlockSize: Int,
        radixGrainSize: Int
    ) throws {
        self.device = device

        self.left = try DepthFirstViewResources(
            device: device,
            maxGaussians: maxGaussians,
            maxWidth: maxWidth,
            maxHeight: maxHeight,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            radixBlockSize: radixBlockSize,
            radixGrainSize: radixGrainSize
        )

        self.right = try DepthFirstViewResources(
            device: device,
            maxGaussians: maxGaussians,
            maxWidth: maxWidth,
            maxHeight: maxHeight,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            radixBlockSize: radixBlockSize,
            radixGrainSize: radixGrainSize
        )
    }
}

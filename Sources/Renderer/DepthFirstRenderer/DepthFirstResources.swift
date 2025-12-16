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
        radixGrainSize: Int,
        tileIdPrecision: RadixSortKeyPrecision
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

        let maxInstances = maxGaussians * 4
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
        let prefixLevel3Blocks = (prefixLevel2Blocks + prefixBlockSize - 1) / prefixBlockSize
        let prefixBlockSumsCount = prefixNumBlocks + 1 + prefixLevel2Blocks + 1 + prefixLevel3Blocks + 1

        // Per-gaussian buffers
        self.renderData = try device.makeBuffer(
            count: maxGaussians,
            type: GaussianRenderData.self,
            options: priv,
            label: "DepthFirstMono_RenderData"
        )

        self.bounds = try device.makeBuffer(
            count: maxGaussians,
            type: SIMD4<Int32>.self,
            options: priv,
            label: "DepthFirstMono_Bounds"
        )

        self.depthKeys = try device.makeBuffer(
            count: paddedGaussianCapacity,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstMono_DepthKeys"
        )

        self.primitiveIndices = try device.makeBuffer(
            count: paddedGaussianCapacity,
            type: Int32.self,
            options: priv,
            label: "DepthFirstMono_PrimitiveIndices"
        )

        self.nTouchedTiles = try device.makeBuffer(
            count: maxGaussians,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstMono_NTouchedTiles"
        )

        self.orderedTileCounts = try device.makeBuffer(
            count: maxGaussians,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstMono_OrderedTileCounts"
        )

        // Counters (shared for CPU readback)
        self.visibleCount = try device.makeBuffer(
            count: 1,
            type: UInt32.self,
            options: shared,
            label: "DepthFirstMono_VisibleCount"
        )

        self.totalInstances = try device.makeBuffer(
            count: 1,
            type: UInt32.self,
            options: shared,
            label: "DepthFirstMono_TotalInstances"
        )

        self.activeTileCount = try device.makeBuffer(
            count: 1,
            type: UInt32.self,
            options: shared,
            label: "DepthFirstMono_ActiveTileCount"
        )

        // Header (32 bytes for DepthFirstHeader)
        guard let header = device.makeBuffer(length: 32, options: shared) else {
            throw RendererError.failedToAllocateBuffer(label: "DepthFirstMono_Header", size: 32)
        }
        header.label = "DepthFirstMono_Header"
        self.header = header

        self.dispatchArgs = try device.makeBuffer(
            count: DepthFirstDispatchSlot.count,
            type: DispatchIndirectArgsSwift.self,
            options: priv,
            label: "DepthFirstMono_DispatchArgs"
        )

        let tileIdStride = tileIdPrecision == .bits32 ? MemoryLayout<UInt32>.stride : MemoryLayout<UInt16>.stride

        // Per-instance buffers (tile IDs are 16-bit or 32-bit depending on precision)
        guard let instanceTileIds = device.makeBuffer(
            length: paddedInstanceCapacity * tileIdStride,
            options: priv
        ) else {
            throw RendererError.failedToAllocateBuffer(
                label: "DepthFirstMono_InstanceTileIds",
                size: paddedInstanceCapacity * tileIdStride
            )
        }
        instanceTileIds.label = "DepthFirstMono_InstanceTileIds"
        self.instanceTileIds = instanceTileIds

        self.instanceGaussianIndices = try device.makeBuffer(
            count: paddedInstanceCapacity,
            type: Int32.self,
            options: priv,
            label: "DepthFirstMono_InstanceGaussianIndices"
        )

        // Tile headers
        self.tileHeaders = try device.makeBuffer(
            count: tileCount,
            type: GaussianHeader.self,
            options: priv,
            label: "DepthFirstMono_TileHeaders"
        )

        self.activeTiles = try device.makeBuffer(
            count: tileCount,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstMono_ActiveTiles"
        )

        // Depth sort buffers
        self.depthSortHistogram = try device.makeBuffer(
            count: gaussianGridSize * radix,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstMono_DepthSortHistogram"
        )

        self.depthSortBlockSums = try device.makeBuffer(
            count: gaussianGridSize,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstMono_DepthSortBlockSums"
        )

        self.depthSortScannedHist = try device.makeBuffer(
            count: gaussianGridSize * radix,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstMono_DepthSortScannedHist"
        )

        self.depthSortScratchKeys = try device.makeBuffer(
            count: paddedGaussianCapacity,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstMono_DepthSortScratchKeys"
        )

        self.depthSortScratchPayload = try device.makeBuffer(
            count: paddedGaussianCapacity,
            type: Int32.self,
            options: priv,
            label: "DepthFirstMono_DepthSortScratchPayload"
        )

        // Tile sort buffers
        self.tileSortHistogram = try device.makeBuffer(
            count: instanceGridSize * radix,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstMono_TileSortHistogram"
        )

        self.tileSortBlockSums = try device.makeBuffer(
            count: instanceGridSize,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstMono_TileSortBlockSums"
        )

        self.tileSortScannedHist = try device.makeBuffer(
            count: instanceGridSize * radix,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstMono_TileSortScannedHist"
        )

        guard let tileSortScratchTileIds = device.makeBuffer(
            length: paddedInstanceCapacity * tileIdStride,
            options: priv
        ) else {
            throw RendererError.failedToAllocateBuffer(
                label: "DepthFirstMono_TileSortScratchTileIds",
                size: paddedInstanceCapacity * tileIdStride
            )
        }
        tileSortScratchTileIds.label = "DepthFirstMono_TileSortScratchTileIds"
        self.tileSortScratchTileIds = tileSortScratchTileIds

        self.tileSortScratchIndices = try device.makeBuffer(
            count: paddedInstanceCapacity,
            type: Int32.self,
            options: priv,
            label: "DepthFirstMono_TileSortScratchIndices"
        )

        // Prefix sum block sums
        self.prefixSumBlockSums = try device.makeBuffer(
            count: prefixBlockSumsCount,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstMono_PrefixSumBlockSums"
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

final class StereoTiledResources {
    let device: MTLDevice

    // Per-gaussian stereo render data (32 bytes per gaussian)
    let renderData: MTLBuffer // StereoTiledRenderData
    let bounds: MTLBuffer // Union tile bounds (int4)
    let depthKeys: MTLBuffer
    let primitiveIndices: MTLBuffer
    let nTouchedTiles: MTLBuffer
    let orderedTileCounts: MTLBuffer

    // Counters
    let visibleCount: MTLBuffer
    let totalInstances: MTLBuffer
    let activeTileCount: MTLBuffer

    // Header for dispatch computation
    let header: MTLBuffer
    let dispatchArgs: MTLBuffer

    // Per-instance buffers
    let instanceTileIds: MTLBuffer
    let instanceGaussianIndices: MTLBuffer

    // Tile headers
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
        radixGrainSize: Int,
        tileIdPrecision: RadixSortKeyPrecision
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

        let maxInstances = maxGaussians * 4
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
        let prefixLevel3Blocks = (prefixLevel2Blocks + prefixBlockSize - 1) / prefixBlockSize
        let prefixBlockSumsCount = prefixNumBlocks + 1 + prefixLevel2Blocks + 1 + prefixLevel3Blocks + 1

        // Per-gaussian buffers - using StereoTiledRenderData (32 bytes)
        self.renderData = try device.makeBuffer(
            count: maxGaussians,
            type: StereoTiledRenderData.self,
            options: priv,
            label: "DepthFirstStereo_RenderData"
        )

        self.bounds = try device.makeBuffer(
            count: maxGaussians,
            type: SIMD4<Int32>.self,
            options: priv,
            label: "DepthFirstStereo_Bounds"
        )

        self.depthKeys = try device.makeBuffer(
            count: paddedGaussianCapacity,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstStereo_DepthKeys"
        )

        self.primitiveIndices = try device.makeBuffer(
            count: paddedGaussianCapacity,
            type: Int32.self,
            options: priv,
            label: "DepthFirstStereo_PrimitiveIndices"
        )

        self.nTouchedTiles = try device.makeBuffer(
            count: maxGaussians,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstStereo_NTouchedTiles"
        )

        self.orderedTileCounts = try device.makeBuffer(
            count: maxGaussians,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstStereo_OrderedTileCounts"
        )

        // Counters
        self.visibleCount = try device.makeBuffer(
            count: 1,
            type: UInt32.self,
            options: shared,
            label: "DepthFirstStereo_VisibleCount"
        )

        self.totalInstances = try device.makeBuffer(
            count: 1,
            type: UInt32.self,
            options: shared,
            label: "DepthFirstStereo_TotalInstances"
        )

        self.activeTileCount = try device.makeBuffer(
            count: 1,
            type: UInt32.self,
            options: shared,
            label: "DepthFirstStereo_ActiveTileCount"
        )

        // Header for dispatch
        self.header = try device.makeBuffer(
            count: 1,
            type: DepthFirstHeader.self,
            options: shared,
            label: "DepthFirstStereo_Header"
        )

        self.dispatchArgs = try device.makeBuffer(
            count: DepthFirstDispatchSlot.count,
            type: DispatchIndirectArgsSwift.self,
            options: priv,
            label: "DepthFirstStereo_DispatchArgs"
        )

        // Per-instance buffers
        if tileIdPrecision == .bits32 {
            self.instanceTileIds = try device.makeBuffer(
                count: maxInstances,
                type: UInt32.self,
                options: priv,
                label: "DepthFirstStereo_InstanceTileIds"
            )
        } else {
            self.instanceTileIds = try device.makeBuffer(
                count: maxInstances,
                type: UInt16.self,
                options: priv,
                label: "DepthFirstStereo_InstanceTileIds"
            )
        }

        self.instanceGaussianIndices = try device.makeBuffer(
            count: maxInstances,
            type: Int32.self,
            options: priv,
            label: "DepthFirstStereo_InstanceGaussianIndices"
        )

        // Tile headers
        self.tileHeaders = try device.makeBuffer(
            count: tileCount,
            type: GaussianHeader.self,
            options: priv,
            label: "DepthFirstStereo_TileHeaders"
        )

        self.activeTiles = try device.makeBuffer(
            count: tileCount,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstStereo_ActiveTiles"
        )

        // Depth sort scratch
        self.depthSortHistogram = try device.makeBuffer(
            count: gaussianGridSize * radix,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstStereo_DepthSortHistogram"
        )

        self.depthSortBlockSums = try device.makeBuffer(
            count: gaussianGridSize,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstStereo_DepthSortBlockSums"
        )

        self.depthSortScannedHist = try device.makeBuffer(
            count: gaussianGridSize * radix,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstStereo_DepthSortScannedHist"
        )

        self.depthSortScratchKeys = try device.makeBuffer(
            count: paddedGaussianCapacity,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstStereo_DepthSortScratchKeys"
        )

        self.depthSortScratchPayload = try device.makeBuffer(
            count: paddedGaussianCapacity,
            type: Int32.self,
            options: priv,
            label: "DepthFirstStereo_DepthSortScratchPayload"
        )

        // Tile sort scratch
        self.tileSortHistogram = try device.makeBuffer(
            count: instanceGridSize * radix,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstStereo_TileSortHistogram"
        )

        self.tileSortBlockSums = try device.makeBuffer(
            count: instanceGridSize,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstStereo_TileSortBlockSums"
        )

        self.tileSortScannedHist = try device.makeBuffer(
            count: instanceGridSize * radix,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstStereo_TileSortScannedHist"
        )

        if tileIdPrecision == .bits32 {
            self.tileSortScratchTileIds = try device.makeBuffer(
                count: paddedInstanceCapacity,
                type: UInt32.self,
                options: priv,
                label: "DepthFirstStereo_TileSortScratchTileIds"
            )
        } else {
            self.tileSortScratchTileIds = try device.makeBuffer(
                count: paddedInstanceCapacity,
                type: UInt16.self,
                options: priv,
                label: "DepthFirstStereo_TileSortScratchTileIds"
            )
        }

        self.tileSortScratchIndices = try device.makeBuffer(
            count: paddedInstanceCapacity,
            type: Int32.self,
            options: priv,
            label: "DepthFirstStereo_TileSortScratchIndices"
        )

        // Prefix sum block sums
        self.prefixSumBlockSums = try device.makeBuffer(
            count: prefixBlockSumsCount,
            type: UInt32.self,
            options: priv,
            label: "DepthFirstStereo_PrefixSumBlockSums"
        )
    }
}

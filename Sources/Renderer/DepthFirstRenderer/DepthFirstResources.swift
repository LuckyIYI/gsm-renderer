import Metal
import RendererTypes

/// Resource layout for depth-first renderer
struct DepthFirstResourceLayout {
    let limits: RendererLimits
    let maxGaussians: Int
    let maxInstances: Int
    let paddedGaussianCapacity: Int
    let paddedInstanceCapacity: Int
    let bufferAllocations: [(label: String, length: Int, options: MTLResourceOptions)]
}

/// Resources for a single view in DepthFirstRenderer's pipeline
final class DepthFirstViewResources {
    let device: MTLDevice

    // Per-gaussian buffers (sized to maxGaussians)
    let renderData: MTLBuffer // GaussianRenderData for each gaussian
    let bounds: MTLBuffer // int4 tile bounds per gaussian
    let depthKeys: MTLBuffer // uint depth keys for sorting
    let primitiveIndices: MTLBuffer // int indices (sorted by depth)
    let nTouchedTiles: MTLBuffer // uint tile count per gaussian
    let orderedTileCounts: MTLBuffer // uint reordered by depth sort

    // Counters (shared for CPU readback)
    let visibleCount: MTLBuffer // atomic uint
    let totalInstances: MTLBuffer // atomic uint
    let activeTileCount: MTLBuffer // atomic uint

    // Header for dispatch computation
    let header: MTLBuffer // DepthFirstHeader
    let dispatchArgs: MTLBuffer // DispatchIndirectArgs array

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

    init(device: MTLDevice, layout: DepthFirstResourceLayout) throws {
        self.device = device
        self.maxGaussians = layout.maxGaussians
        self.maxInstances = layout.maxInstances
        self.tileCount = layout.limits.maxTileCount

        // Allocate buffers
        var bufferMap: [String: MTLBuffer] = [:]
        for req in layout.bufferAllocations {
            guard let buf = device.makeBuffer(length: req.length, options: req.options) else {
                throw RendererError.failedToAllocateBuffer(label: req.label, size: req.length)
            }
            buf.label = req.label
            bufferMap[req.label] = buf
        }

        func buffer(_ label: String) throws -> MTLBuffer {
            guard let buf = bufferMap[label] else {
                throw RendererError.missingRequiredBuffer(label)
            }
            return buf
        }

        // Assign buffers
        self.renderData = try buffer("RenderData")
        self.bounds = try buffer("Bounds")
        self.depthKeys = try buffer("DepthKeys")
        self.primitiveIndices = try buffer("PrimitiveIndices")
        self.nTouchedTiles = try buffer("NTouchedTiles")
        self.orderedTileCounts = try buffer("OrderedTileCounts")

        self.visibleCount = try buffer("VisibleCount")
        self.totalInstances = try buffer("TotalInstances")
        self.activeTileCount = try buffer("ActiveTileCount")
        self.header = try buffer("Header")
        self.dispatchArgs = try buffer("DispatchArgs")

        self.instanceTileIds = try buffer("InstanceTileIds")
        self.instanceGaussianIndices = try buffer("InstanceGaussianIndices")

        self.tileHeaders = try buffer("TileHeaders")
        self.activeTiles = try buffer("ActiveTiles")

        self.depthSortHistogram = try buffer("DepthSortHistogram")
        self.depthSortBlockSums = try buffer("DepthSortBlockSums")
        self.depthSortScannedHist = try buffer("DepthSortScannedHist")
        self.depthSortScratchKeys = try buffer("DepthSortScratchKeys")
        self.depthSortScratchPayload = try buffer("DepthSortScratchPayload")

        self.tileSortHistogram = try buffer("TileSortHistogram")
        self.tileSortBlockSums = try buffer("TileSortBlockSums")
        self.tileSortScannedHist = try buffer("TileSortScannedHist")
        self.tileSortScratchTileIds = try buffer("TileSortScratchTileIds")
        self.tileSortScratchIndices = try buffer("TileSortScratchIndices")

        self.prefixSumBlockSums = try buffer("PrefixSumBlockSums")

        // Output textures
        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: layout.limits.maxWidth,
            height: layout.limits.maxHeight,
            mipmapped: false
        )
        texDesc.usage = [.shaderWrite, .shaderRead]
        texDesc.storageMode = .private
        guard let colorTex = device.makeTexture(descriptor: texDesc) else {
            throw RendererError.failedToAllocateTexture(
                label: "OutputColorTex",
                width: layout.limits.maxWidth,
                height: layout.limits.maxHeight
            )
        }
        colorTex.label = "DepthFirstOutputColorTex"

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: layout.limits.maxWidth,
            height: layout.limits.maxHeight,
            mipmapped: false
        )
        depthDesc.usage = [.shaderWrite, .shaderRead]
        depthDesc.storageMode = .private
        guard let depthTex = device.makeTexture(descriptor: depthDesc) else {
            throw RendererError.failedToAllocateTexture(
                label: "OutputDepthTex",
                width: layout.limits.maxWidth,
                height: layout.limits.maxHeight
            )
        }
        depthTex.label = "DepthFirstOutputDepthTex"

        self.outputTextures = RenderOutputTextures(color: colorTex, depth: depthTex)
    }
}

/// Stereo resources for DepthFirstRenderer
final class DepthFirstMultiViewResources {
    let device: MTLDevice
    let left: DepthFirstViewResources
    let right: DepthFirstViewResources

    init(device: MTLDevice, layout: DepthFirstResourceLayout) throws {
        self.device = device
        self.left = try DepthFirstViewResources(device: device, layout: layout)
        self.right = try DepthFirstViewResources(device: device, layout: layout)
    }
}

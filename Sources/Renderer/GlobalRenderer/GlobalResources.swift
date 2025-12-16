import Metal
import RendererTypes

/// Resources for a single view in GlobalRenderer's pipeline.
/// Allocates exactly what GlobalRenderer needs - direct allocations like LocalViewResources.
final class GlobalViewResources {
    let device: MTLDevice

    // Tile assignment buffers
    let boundsBuffer: MTLBuffer
    let coverageBuffer: MTLBuffer
    let offsetsBuffer: MTLBuffer
    let partialSumsBuffer: MTLBuffer
    let scatterDispatchBuffer: MTLBuffer
    let prefixSumBlockCountBuffer: MTLBuffer
    let tileAssignmentHeader: MTLBuffer
    let tileIndices: MTLBuffer
    let tileIds: MTLBuffer
    let tileAssignBlockSums: MTLBuffer
    var tileAssignmentMaxAssignments: Int
    var tileAssignmentPaddedCapacity: Int

    // Ordered/packed buffers
    let orderedHeaders: MTLBuffer
    let packedMeans: MTLBuffer
    let packedConics: MTLBuffer
    let packedColors: MTLBuffer
    let packedOpacities: MTLBuffer
    let packedDepths: MTLBuffer
    let activeTileIndices: MTLBuffer
    let activeTileCount: MTLBuffer

    // Sort buffers
    let sortKeys: MTLBuffer
    let sortedIndices: MTLBuffer

    // Radix sort scratch buffers
    let radixHistogram: MTLBuffer
    let radixBlockSums: MTLBuffer
    let radixScannedHistogram: MTLBuffer
    let radixKeysScratch: MTLBuffer
    let radixPayloadScratch: MTLBuffer

    // Interleaved gaussian data
    let interleavedGaussians: MTLBuffer

    // Indirect dispatch buffers
    let visibleIndices: MTLBuffer
    let visibleCount: MTLBuffer
    let tileAssignDispatchArgs: MTLBuffer

    // Dispatch args
    let dispatchArgs: MTLBuffer

    // Output textures
    var outputTextures: RenderOutputTextures

    init(
        device: MTLDevice,
        maxGaussians: Int,
        maxWidth: Int,
        maxHeight: Int,
        tileWidth: Int,
        tileHeight: Int,
        radixBlockSize: Int,
        radixGrainSize: Int,
        precision: Precision
    ) throws {
        self.device = device

        let priv: MTLResourceOptions = .storageModePrivate
        let shared: MTLResourceOptions = .storageModeShared

        // Calculate derived sizes
        let tilesX = (maxWidth + tileWidth - 1) / tileWidth
        let tilesY = (maxHeight + tileHeight - 1) / tileHeight
        let tileCount = max(1, tilesX * tilesY)

        let gaussianTileCapacity = maxGaussians * 4
        let maxAssignmentCapacity = gaussianTileCapacity
        self.tileAssignmentMaxAssignments = maxAssignmentCapacity

        let radixBlock = radixBlockSize * radixGrainSize
        let paddedCapacity = ((maxAssignmentCapacity + radixBlock - 1) / radixBlock) * radixBlock
        self.tileAssignmentPaddedCapacity = paddedCapacity

        // Stride calculations for precision
        let half2Stride = MemoryLayout<UInt16>.stride * 2
        let half4Stride = MemoryLayout<UInt16>.stride * 4
        let strideForMeans = precision == .float16 ? half2Stride : 8
        let strideForConics = precision == .float16 ? half4Stride : 16
        let strideForColors = precision == .float16 ? MemoryLayout<UInt16>.stride * 3 : 12
        let strideForOpacities = precision == .float16 ? MemoryLayout<UInt16>.stride : 4
        let strideForDepths = precision == .float16 ? MemoryLayout<UInt16>.stride : 4
        let strideForInterleaved = precision == .float16 ? 32 : 48

        // Prefix sum sizing
        let prefixBlockSize = 256
        let prefixGrainSize = 4
        let elementsPerGroup = prefixBlockSize * prefixGrainSize
        let groups = (maxGaussians + elementsPerGroup - 1) / elementsPerGroup

        // Radix sort sizing
        let valuesPerGroup = radixBlockSize * radixGrainSize
        let gridCapacity = max(1, (maxAssignmentCapacity + valuesPerGroup - 1) / valuesPerGroup)
        let histogramCapacity = gridCapacity * 256 // radix = 256

        // Two-pass tile assignment sizing
        let tileAssignBlockSize = 256
        let tileAssignNumBlocks = (maxGaussians + tileAssignBlockSize - 1) / tileAssignBlockSize
        let tileAssignLevel2Blocks = (tileAssignNumBlocks + tileAssignBlockSize - 1) / tileAssignBlockSize
        let tileAssignBlockSumsCount = tileAssignNumBlocks + 1 + tileAssignLevel2Blocks + 1

        // Dispatch args
        self.dispatchArgs = try device.makeBuffer(
            count: 256,
            type: UInt32.self,
            options: priv,
            label: "FrameDispatchArgs"
        )

        // Shared allocations
        self.tileAssignmentHeader = try device.makeBuffer(
            count: 1,
            type: TileAssignmentHeaderSwift.self,
            options: shared,
            label: "TileHeader"
        )
        // Initialize header
        let headerPtr = self.tileAssignmentHeader.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.maxCapacity = UInt32(maxAssignmentCapacity)
        headerPtr.pointee.paddedCount = UInt32(paddedCapacity)
        headerPtr.pointee.totalAssignments = 0
        headerPtr.pointee.overflow = 0

        self.activeTileCount = try device.makeBuffer(
            count: 1,
            type: UInt32.self,
            options: shared,
            label: "ActiveTileCount"
        )

        self.visibleCount = try device.makeBuffer(
            count: 1,
            type: UInt32.self,
            options: shared,
            label: "VisibleCount"
        )

        // Per-gaussian buffers
        self.boundsBuffer = try device.makeBuffer(
            count: maxGaussians,
            type: SIMD4<Int32>.self,
            options: priv,
            label: "Bounds"
        )

        self.coverageBuffer = try device.makeBuffer(
            count: maxGaussians,
            type: UInt32.self,
            options: priv,
            label: "Coverage"
        )

        self.offsetsBuffer = try device.makeBuffer(
            count: maxGaussians + 1,
            type: UInt32.self,
            options: priv,
            label: "Offsets"
        )

        self.partialSumsBuffer = try device.makeBuffer(
            count: max(groups, 1),
            type: UInt32.self,
            options: priv,
            label: "PartialSums"
        )

        self.scatterDispatchBuffer = try device.makeBuffer(
            count: 6,
            type: UInt32.self,
            options: priv,
            label: "ScatterDispatch"
        )

        self.prefixSumBlockCountBuffer = try device.makeBuffer(
            count: 2,
            type: UInt32.self,
            options: priv,
            label: "PrefixSumBlockCount"
        )

        // Tile assignment buffers
        self.tileIndices = try device.makeBuffer(
            count: paddedCapacity,
            type: Int32.self,
            options: priv,
            label: "TileIndices"
        )

        self.tileIds = try device.makeBuffer(
            count: paddedCapacity,
            type: Int32.self,
            options: priv,
            label: "TileIds"
        )

        self.tileAssignBlockSums = try device.makeBuffer(
            count: tileAssignBlockSumsCount,
            type: UInt32.self,
            options: priv,
            label: "TileAssignBlockSums"
        )

        // Ordered/packed buffers
        self.orderedHeaders = try device.makeBuffer(
            count: tileCount,
            type: GaussianHeader.self,
            options: priv,
            label: "OrderedHeaders"
        )

        guard let packedMeans = device.makeBuffer(length: maxAssignmentCapacity * strideForMeans, options: priv) else {
            throw RendererError.failedToAllocateBuffer(label: "PackedMeans", size: maxAssignmentCapacity * strideForMeans)
        }
        packedMeans.label = "PackedMeans"
        self.packedMeans = packedMeans

        guard let packedConics = device.makeBuffer(length: maxAssignmentCapacity * strideForConics, options: priv) else {
            throw RendererError.failedToAllocateBuffer(label: "PackedConics", size: maxAssignmentCapacity * strideForConics)
        }
        packedConics.label = "PackedConics"
        self.packedConics = packedConics

        guard let packedColors = device.makeBuffer(length: maxAssignmentCapacity * strideForColors, options: priv) else {
            throw RendererError.failedToAllocateBuffer(label: "PackedColors", size: maxAssignmentCapacity * strideForColors)
        }
        packedColors.label = "PackedColors"
        self.packedColors = packedColors

        guard let packedOpacities = device.makeBuffer(length: maxAssignmentCapacity * strideForOpacities, options: priv) else {
            throw RendererError.failedToAllocateBuffer(label: "PackedOpacities", size: maxAssignmentCapacity * strideForOpacities)
        }
        packedOpacities.label = "PackedOpacities"
        self.packedOpacities = packedOpacities

        guard let packedDepths = device.makeBuffer(length: maxAssignmentCapacity * strideForDepths, options: priv) else {
            throw RendererError.failedToAllocateBuffer(label: "PackedDepths", size: maxAssignmentCapacity * strideForDepths)
        }
        packedDepths.label = "PackedDepths"
        self.packedDepths = packedDepths

        self.activeTileIndices = try device.makeBuffer(
            count: tileCount,
            type: UInt32.self,
            options: priv,
            label: "ActiveTileIndices"
        )

        // Sort buffers
        self.sortKeys = try device.makeBuffer(
            count: paddedCapacity,
            type: UInt32.self,
            options: priv,
            label: "SortKeys"
        )

        self.sortedIndices = try device.makeBuffer(
            count: paddedCapacity,
            type: Int32.self,
            options: priv,
            label: "SortedIndices"
        )

        // Radix sort buffers
        self.radixHistogram = try device.makeBuffer(
            count: histogramCapacity,
            type: UInt32.self,
            options: priv,
            label: "RadixHist"
        )

        self.radixBlockSums = try device.makeBuffer(
            count: gridCapacity,
            type: UInt32.self,
            options: priv,
            label: "RadixBlockSums"
        )

        self.radixScannedHistogram = try device.makeBuffer(
            count: histogramCapacity,
            type: UInt32.self,
            options: priv,
            label: "RadixScanned"
        )

        self.radixKeysScratch = try device.makeBuffer(
            count: maxAssignmentCapacity,
            type: UInt32.self,
            options: priv,
            label: "RadixScratch"
        )

        self.radixPayloadScratch = try device.makeBuffer(
            count: maxAssignmentCapacity,
            type: UInt32.self,
            options: priv,
            label: "RadixPayload"
        )

        // Interleaved gaussian data
        guard let interleavedGaussians = device.makeBuffer(length: maxGaussians * strideForInterleaved, options: priv) else {
            throw RendererError.failedToAllocateBuffer(label: "InterleavedGaussians", size: maxGaussians * strideForInterleaved)
        }
        interleavedGaussians.label = "InterleavedGaussians"
        self.interleavedGaussians = interleavedGaussians

        // Indirect dispatch buffers
        self.visibleIndices = try device.makeBuffer(
            count: maxGaussians,
            type: UInt32.self,
            options: priv,
            label: "VisibleIndices"
        )

        self.tileAssignDispatchArgs = try device.makeBuffer(
            count: 3,
            type: UInt32.self,
            options: priv,
            label: "TileAssignDispatchArgs"
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
            throw RendererError.failedToAllocateTexture(label: "OutputColorTex", width: maxWidth, height: maxHeight)
        }
        colorTex.label = "OutputColorTex"

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: maxWidth,
            height: maxHeight,
            mipmapped: false
        )
        depthDesc.usage = [.shaderWrite, .shaderRead]
        depthDesc.storageMode = .private
        guard let depthTex = device.makeTexture(descriptor: depthDesc) else {
            throw RendererError.failedToAllocateTexture(label: "OutputDepthTex", width: maxWidth, height: maxHeight)
        }
        depthTex.label = "OutputDepthTex"

        self.outputTextures = RenderOutputTextures(color: colorTex, depth: depthTex)
    }
}

/// Stereo resources for GlobalRenderer - left and right view buffer sets.
final class GlobalMultiViewResources {
    let device: MTLDevice
    let left: GlobalViewResources
    let right: GlobalViewResources

    init(
        device: MTLDevice,
        maxGaussians: Int,
        maxWidth: Int,
        maxHeight: Int,
        tileWidth: Int,
        tileHeight: Int,
        radixBlockSize: Int,
        radixGrainSize: Int,
        precision: Precision
    ) throws {
        self.device = device

        self.left = try GlobalViewResources(
            device: device,
            maxGaussians: maxGaussians,
            maxWidth: maxWidth,
            maxHeight: maxHeight,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            radixBlockSize: radixBlockSize,
            radixGrainSize: radixGrainSize,
            precision: precision
        )

        self.right = try GlobalViewResources(
            device: device,
            maxGaussians: maxGaussians,
            maxWidth: maxWidth,
            maxHeight: maxHeight,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            radixBlockSize: radixBlockSize,
            radixGrainSize: radixGrainSize,
            precision: precision
        )
    }
}

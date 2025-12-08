import Metal

/// Resources for a single view in GlobalRenderer's pipeline.
final class GlobalViewResources {
    // Tile Builder Buffers
    let boundsBuffer: MTLBuffer
    let coverageBuffer: MTLBuffer
    let offsetsBuffer: MTLBuffer
    let partialSumsBuffer: MTLBuffer
    let scatterDispatchBuffer: MTLBuffer
    let tileAssignmentHeader: MTLBuffer
    let tileIndices: MTLBuffer
    let tileIds: MTLBuffer
    var tileAssignmentMaxAssignments: Int
    var tileAssignmentPaddedCapacity: Int

    // Ordered Buffers
    let orderedHeaders: MTLBuffer
    let packedMeans: MTLBuffer
    let packedConics: MTLBuffer
    let packedColors: MTLBuffer
    let packedOpacities: MTLBuffer
    let packedDepths: MTLBuffer
    let activeTileIndices: MTLBuffer
    let activeTileCount: MTLBuffer

    // Sort Buffers
    let sortKeys: MTLBuffer
    let sortedIndices: MTLBuffer

    // Radix Sort Buffers
    let radixHistogram: MTLBuffer?
    let radixBlockSums: MTLBuffer?
    let radixScannedHistogram: MTLBuffer?
    let radixKeysScratch: MTLBuffer?
    let radixPayloadScratch: MTLBuffer?

    // Interleaved gaussian data for cache efficiency
    let interleavedGaussians: MTLBuffer?

    // Two-pass tile assignment scratch buffer (prefix sum block sums)
    let tileAssignBlockSums: MTLBuffer?

    // Indirect dispatch buffers (for compacted visible gaussians)
    let visibleIndices: MTLBuffer // Compacted visible gaussian indices
    let visibleCount: MTLBuffer // Atomic counter for visible count
    let tileAssignDispatchArgs: MTLBuffer // Indirect dispatch args for tile count/scatter

    // Dispatch Args (Per-frame to avoid race on indirect dispatch)
    let dispatchArgs: MTLBuffer

    // Output Textures
    var outputTextures: RenderOutputTextures

    let device: MTLDevice

    init(device: MTLDevice, layout: GlobalResourceLayout) throws {
        self.device = device
        self.tileAssignmentMaxAssignments = layout.maxAssignmentCapacity
        self.tileAssignmentPaddedCapacity = layout.paddedCapacity

        guard let dispatchArgs = device.makeBuffer(length: 1024 * MemoryLayout<UInt32>.stride, options: .storageModePrivate) else {
            throw RendererError.failedToAllocateBuffer(label: "FrameDispatchArgs", size: 1024 * MemoryLayout<UInt32>.stride)
        }
        dispatchArgs.label = "FrameDispatchArgs"
        self.dispatchArgs = dispatchArgs

        // Buffer allocation helper
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
        self.boundsBuffer = try buffer("Bounds")
        self.coverageBuffer = try buffer("Coverage")
        self.offsetsBuffer = try buffer("Offsets")
        self.partialSumsBuffer = try buffer("PartialSums")
        self.scatterDispatchBuffer = try buffer("ScatterDispatch")
        self.tileAssignmentHeader = try buffer("TileHeader")

        // Initialize TileHeader constants ONCE at setup (not during render)
        // TileHeader is .storageModeShared so this CPU write at init is allowed
        let headerPtr = self.tileAssignmentHeader.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.maxCapacity = UInt32(layout.maxAssignmentCapacity)
        headerPtr.pointee.paddedCount = UInt32(layout.paddedCapacity)
        headerPtr.pointee.totalAssignments = 0
        headerPtr.pointee.overflow = 0

        self.tileIndices = try buffer("TileIndices")
        self.tileIds = try buffer("TileIds")
        self.orderedHeaders = try buffer("OrderedHeaders")
        self.packedMeans = try buffer("PackedMeans")
        self.packedConics = try buffer("PackedConics")
        self.packedColors = try buffer("PackedColors")
        self.packedOpacities = try buffer("PackedOpacities")
        self.packedDepths = try buffer("PackedDepths")
        self.activeTileIndices = try buffer("ActiveTileIndices")
        self.activeTileCount = try buffer("ActiveTileCount")
        self.sortKeys = try buffer("SortKeys")
        self.sortedIndices = try buffer("SortedIndices")

        // Conditionally allocate radix/fused buffers based on config
        func optionalBuffer(_ label: String) -> MTLBuffer? {
            bufferMap[label]
        }
        self.radixHistogram = optionalBuffer("RadixHist")
        self.radixBlockSums = optionalBuffer("RadixBlockSums")
        self.radixScannedHistogram = optionalBuffer("RadixScanned")
        self.radixKeysScratch = optionalBuffer("RadixScratch")
        self.radixPayloadScratch = optionalBuffer("RadixPayload")
        self.interleavedGaussians = optionalBuffer("InterleavedGaussians")
        self.tileAssignBlockSums = optionalBuffer("TileAssignBlockSums")

        // Indirect dispatch buffers
        self.visibleIndices = try buffer("VisibleIndices")
        self.visibleCount = try buffer("VisibleCount")
        self.tileAssignDispatchArgs = try buffer("TileAssignDispatchArgs")

        // Output textures sized to limits
        let texDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: layout.limits.maxWidth, height: layout.limits.maxHeight, mipmapped: false)
        texDesc.usage = [.shaderWrite, .shaderRead]
        texDesc.storageMode = .private
        guard let colorTex = device.makeTexture(descriptor: texDesc) else {
            throw RendererError.failedToAllocateTexture(label: "OutputColorTex", width: layout.limits.maxWidth, height: layout.limits.maxHeight)
        }
        colorTex.label = "OutputColorTex"

        // Depth texture uses half precision like Local
        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: layout.limits.maxWidth, height: layout.limits.maxHeight, mipmapped: false)
        depthDesc.usage = [.shaderWrite, .shaderRead]
        depthDesc.storageMode = .private
        guard let depthTex = device.makeTexture(descriptor: depthDesc) else {
            throw RendererError.failedToAllocateTexture(label: "OutputDepthTex", width: layout.limits.maxWidth, height: layout.limits.maxHeight)
        }
        depthTex.label = "OutputDepthTex"
        self.outputTextures = RenderOutputTextures(color: colorTex, depth: depthTex)
    }
}

// MARK: - GlobalRenderer Stereo Resources

/// Stereo resources for GlobalRenderer - left and right view buffer sets.
final class GlobalMultiViewResources {
    let device: MTLDevice
    let left: GlobalViewResources
    let right: GlobalViewResources

    init(device: MTLDevice, layout: GlobalResourceLayout) throws {
        self.device = device
        self.left = try GlobalViewResources(device: device, layout: layout)
        self.right = try GlobalViewResources(device: device, layout: layout)
    }
}

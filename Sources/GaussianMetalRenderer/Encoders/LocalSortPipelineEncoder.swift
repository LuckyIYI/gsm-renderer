import Metal

/// Local sort pipeline encoder - Memory-efficient deterministic Gaussian splatting
/// Uses two-pass projection: Visibility → PrefixSum → ProjectScatter
/// Pipeline: Clear → ProjectVisibility → PrefixScan → ProjectScatter → TilePrefixScan → Scatter → Sort → Render
public final class LocalSortPipelineEncoder {
    // Device and library
    private let device: MTLDevice
    private let localSortLibrary: MTLLibrary

    /// Public access to the Metal library for shared encoders
    public var library: MTLLibrary { localSortLibrary }

    // Per-stage encoders
    private let clearEncoder: LocalSortClearEncoder
    private let projectEncoder: LocalSortProjectEncoder
    private let prefixScanEncoder: LocalSortPrefixScanEncoder
    private let scatterEncoder: LocalSortScatterEncoder
    private let sortEncoder: LocalSortSortEncoder
    private let renderEncoder: LocalSortRenderEncoder

    public init(device: MTLDevice) throws {
        self.device = device

        // Load Local sort shader library
        guard let libraryURL = Bundle.module.url(forResource: "LocalSortShaders", withExtension: "metallib"),
              let library = try? device.makeLibrary(URL: libraryURL) else {
            fatalError("Failed to load LocalSortShaders.metallib")
        }
        self.localSortLibrary = library

        // Load main library for hierarchical prefix sum
        guard let mainLibraryURL = Bundle.module.url(forResource: "GaussianMetalRenderer", withExtension: "metallib"),
              let mainLibrary = try? device.makeLibrary(URL: mainLibraryURL) else {
            fatalError("Failed to load GaussianMetalRenderer.metallib")
        }

        // Initialize per-stage encoders
        self.clearEncoder = try LocalSortClearEncoder(library: library, device: device)
        self.projectEncoder = try LocalSortProjectEncoder(
            localSortLibrary: library,
            mainLibrary: mainLibrary,
            device: device
        )
        self.prefixScanEncoder = try LocalSortPrefixScanEncoder(library: library, device: device)
        self.scatterEncoder = try LocalSortScatterEncoder(library: library, device: device)
        self.sortEncoder = try LocalSortSortEncoder(library: library, device: device)
        self.renderEncoder = try LocalSortRenderEncoder(library: library, device: device)
    }

    /// Legacy init for backwards compatibility
    public convenience init(device: MTLDevice, library: MTLLibrary) throws {
        try self.init(device: device)
    }

    /// Encode the full Local sort pipeline
    /// Efficient temp buffer approach: ProjectStore → PrefixSum → CompactCount → Scatter → Sort
    public func encode(
        commandBuffer: MTLCommandBuffer,
        // Input
        worldGaussians: MTLBuffer,
        harmonics: MTLBuffer,
        camera: CameraUniformsSwift,
        gaussianCount: Int,
        // Tile params
        tilesX: Int,
        tilesY: Int,
        tileWidth: Int,
        tileHeight: Int,
        surfaceWidth: Int,
        surfaceHeight: Int,
        // Output buffers
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        tileCounts: MTLBuffer,
        tileOffsets: MTLBuffer,
        partialSums: MTLBuffer,
        sortKeys: MTLBuffer?,
        sortIndices: MTLBuffer,
        maxCompacted: Int,
        maxAssignments: Int,
        // Temp buffer for single-pass projection
        tempProjectionBuffer: MTLBuffer,
        // Visibility prefix sum buffers
        visibilityMarks: MTLBuffer,
        visibilityPartialSums: MTLBuffer,
        // Options
        useHalfWorld: Bool = false,
        skipSort: Bool = false,
        tempSortKeys: MTLBuffer? = nil,
        tempSortIndices: MTLBuffer? = nil,
        clusterVisibility: MTLBuffer? = nil,
        clusterSize: UInt32 = 1024,
        use16BitSort: Bool = false,
        depthKeys16: MTLBuffer? = nil
    ) {
        let tileCount = tilesX * tilesY

        let params = TileBinningParams(
            gaussianCount: UInt32(gaussianCount),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            surfaceWidth: UInt32(surfaceWidth),
            surfaceHeight: UInt32(surfaceHeight),
            maxCapacity: UInt32(maxCompacted)
        )

        // === 1. CLEAR ===
        clearEncoder.encode(
            commandBuffer: commandBuffer,
            tileCounts: tileCounts,
            header: compactedHeader,
            tileCount: tileCount,
            maxCompacted: maxCompacted
        )

        // === 2. PROJECT + VISIBILITY PREFIX SUM + COMPACT COUNT ===
        projectEncoder.encode(
            commandBuffer: commandBuffer,
            worldGaussians: worldGaussians,
            harmonics: harmonics,
            camera: camera,
            params: params,
            gaussianCount: gaussianCount,
            tilesX: tilesX,
            tempProjectionBuffer: tempProjectionBuffer,
            visibilityMarks: visibilityMarks,
            visibilityPartialSums: visibilityPartialSums,
            compactedGaussians: compactedGaussians,
            compactedHeader: compactedHeader,
            tileCounts: tileCounts,
            useHalfWorld: useHalfWorld,
            clusterVisibility: clusterVisibility,
            clusterSize: clusterSize
        )

        // === 3. TILE PREFIX SCAN ===
        prefixScanEncoder.encode(
            commandBuffer: commandBuffer,
            tileCounts: tileCounts,
            tileOffsets: tileOffsets,
            partialSums: partialSums,
            tileCount: tileCount
        )

        // === 4. SCATTER ===
        let effective16Bit = use16BitSort && scatterEncoder.has16BitScatter && depthKeys16 != nil
        if effective16Bit, let depth16 = depthKeys16 {
            scatterEncoder.encode16(
                commandBuffer: commandBuffer,
                compactedGaussians: compactedGaussians,
                compactedHeader: compactedHeader,
                tileCounts: tileCounts,
                tileOffsets: tileOffsets,
                depthKeys16: depth16,
                sortIndices: sortIndices,
                tilesX: tilesX,
                maxAssignments: maxAssignments,
                tileWidth: tileWidth,
                tileHeight: tileHeight
            )
        } else if let sortKeysBuf = sortKeys {
            scatterEncoder.encode(
                commandBuffer: commandBuffer,
                compactedGaussians: compactedGaussians,
                compactedHeader: compactedHeader,
                tileCounts: tileCounts,
                tileOffsets: tileOffsets,
                sortKeys: sortKeysBuf,
                sortIndices: sortIndices,
                tilesX: tilesX,
                maxAssignments: maxAssignments,
                tileWidth: tileWidth,
                tileHeight: tileHeight
            )
        }

        // === 5. PER-TILE SORT ===
        if !skipSort, let sortKeysBuf = sortKeys, let tempK = tempSortKeys, let tempV = tempSortIndices {
            sortEncoder.encode(
                commandBuffer: commandBuffer,
                sortKeys: sortKeysBuf,
                sortIndices: sortIndices,
                tileOffsets: tileOffsets,
                tileCounts: tileCounts,
                tempSortKeys: tempK,
                tempSortIndices: tempV,
                tileCount: tileCount
            )
        }
    }

    /// Encode just the render pass (after sorting is complete)
    public func encodeRender(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        tileOffsets: MTLBuffer,
        tileCounts: MTLBuffer,
        sortedIndices: MTLBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        tilesX: Int,
        tilesY: Int,
        width: Int,
        height: Int,
        tileWidth: Int,
        tileHeight: Int,
        whiteBackground: Bool = false
    ) {
        renderEncoder.encode(
            commandBuffer: commandBuffer,
            compactedGaussians: compactedGaussians,
            tileOffsets: tileOffsets,
            tileCounts: tileCounts,
            sortedIndices: sortedIndices,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            tilesX: tilesX,
            tilesY: tilesY,
            width: width,
            height: height,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            whiteBackground: whiteBackground
        )
    }

    /// Texture width for render texture (must match RENDER_TEX_WIDTH in shader)
    public static let renderTexWidth = LocalSortRenderEncoder.renderTexWidth

    /// Check if textured render is available
    public var hasTexturedRender: Bool { renderEncoder.hasTexturedRender }

    /// Check if 16-bit sort is available
    public var has16BitSort: Bool {
        scatterEncoder.has16BitScatter && sortEncoder.has16BitSort && renderEncoder.has16BitRender
    }

    /// Encode the pack pass to copy gaussian data to texture
    public func encodePackRenderTexture(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        renderTexture: MTLTexture,
        maxGaussians: Int
    ) {
        renderEncoder.encodePackRenderTexture(
            commandBuffer: commandBuffer,
            compactedGaussians: compactedGaussians,
            compactedHeader: compactedHeader,
            renderTexture: renderTexture,
            maxGaussians: maxGaussians
        )
    }

    /// Encode render pass using texture-cached gaussian data
    public func encodeRenderTextured(
        commandBuffer: MTLCommandBuffer,
        gaussianTexture: MTLTexture,
        tileOffsets: MTLBuffer,
        tileCounts: MTLBuffer,
        sortedIndices: MTLBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        tilesX: Int,
        tilesY: Int,
        width: Int,
        height: Int,
        tileWidth: Int,
        tileHeight: Int,
        whiteBackground: Bool = false
    ) {
        renderEncoder.encodeTextured(
            commandBuffer: commandBuffer,
            gaussianTexture: gaussianTexture,
            tileOffsets: tileOffsets,
            tileCounts: tileCounts,
            sortedIndices: sortedIndices,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            tilesX: tilesX,
            tilesY: tilesY,
            width: width,
            height: height,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            whiteBackground: whiteBackground
        )
    }

    /// Read visible count from header buffer (call after command buffer completes)
    public static func readVisibleCount(from header: MTLBuffer) -> UInt32 {
        let ptr = header.contents().bindMemory(to: CompactedHeaderSwift.self, capacity: 1)
        return ptr.pointee.visibleCount
    }

    /// Debug: Print first few compacted gaussians
    public static func debugPrintCompacted(from compacted: MTLBuffer, header: MTLBuffer, count: Int = 5) {
        let visibleCount = readVisibleCount(from: header)
        print("[LocalSort Debug] Visible count: \(visibleCount)")

        guard visibleCount > 0, compacted.storageMode == .shared else {
            print("[LocalSort Debug] Cannot read compacted buffer (private storage or empty)")
            return
        }

        let ptr = compacted.contents().bindMemory(to: CompactedGaussianSwift.self, capacity: Int(visibleCount))
        for i in 0..<min(count, Int(visibleCount)) {
            let g = ptr[i]
            print("  [\(i)] pos=(\(g.position_color.x), \(g.position_color.y)) depth=\(g.covariance_depth.w) tiles=(\(g.min_tile.x),\(g.min_tile.y))->(\(g.max_tile.x),\(g.max_tile.y))")
        }
    }

    /// Read overflow flag from header buffer
    public static func readOverflow(from header: MTLBuffer) -> Bool {
        let ptr = header.contents().bindMemory(to: CompactedHeaderSwift.self, capacity: 1)
        return ptr.pointee.overflow != 0
    }

    /// Clear just the header buffer
    public func encodeClearHeader(
        commandBuffer: MTLCommandBuffer,
        header: MTLBuffer,
        maxCompacted: Int
    ) {
        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            blitEncoder.label = "LocalSort_ClearHeader"
            blitEncoder.fill(buffer: header, range: 0..<MemoryLayout<CompactedHeaderSwift>.stride, value: 0)
            blitEncoder.endEncoding()
        }
    }

    // MARK: - 16-bit Sort Experimental

    /// Encode 16-bit scatter
    public func encodeScatter16(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        tileCounts: MTLBuffer,
        tileOffsets: MTLBuffer,
        depthKeys16: MTLBuffer,
        globalIndices: MTLBuffer,
        tilesX: Int,
        maxAssignments: Int,
        tileWidth: Int,
        tileHeight: Int,
        dispatchArgsBuffer: MTLBuffer
    ) {
        scatterEncoder.encode16(
            commandBuffer: commandBuffer,
            compactedGaussians: compactedGaussians,
            compactedHeader: compactedHeader,
            tileCounts: tileCounts,
            tileOffsets: tileOffsets,
            depthKeys16: depthKeys16,
            sortIndices: globalIndices,
            tilesX: tilesX,
            maxAssignments: maxAssignments,
            tileWidth: tileWidth,
            tileHeight: tileHeight
        )
    }

    /// Encode 16-bit per-tile sort
    public func encodeSort16(
        commandBuffer: MTLCommandBuffer,
        depthKeys16: MTLBuffer,
        globalIndices: MTLBuffer,
        sortedLocalIdx: MTLBuffer,
        tileOffsets: MTLBuffer,
        tileCounts: MTLBuffer,
        tileCount: Int
    ) {
        sortEncoder.encode16(
            commandBuffer: commandBuffer,
            depthKeys16: depthKeys16,
            globalIndices: globalIndices,
            sortedLocalIdx: sortedLocalIdx,
            tileOffsets: tileOffsets,
            tileCounts: tileCounts,
            tileCount: tileCount
        )
    }

    /// Encode 16-bit render
    public func encodeRender16(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        tileOffsets: MTLBuffer,
        tileCounts: MTLBuffer,
        sortedLocalIdx: MTLBuffer,
        globalIndices: MTLBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        tilesX: Int,
        tilesY: Int,
        width: Int,
        height: Int,
        tileWidth: Int,
        tileHeight: Int,
        whiteBackground: Bool = false
    ) {
        renderEncoder.encode16(
            commandBuffer: commandBuffer,
            compactedGaussians: compactedGaussians,
            tileOffsets: tileOffsets,
            tileCounts: tileCounts,
            sortedLocalIdx: sortedLocalIdx,
            globalIndices: globalIndices,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            tilesX: tilesX,
            tilesY: tilesY,
            width: width,
            height: height,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            whiteBackground: whiteBackground
        )
    }
}

/// Render params struct for local sort pipeline (matches Metal RenderParams - 40 bytes)
public struct LocalSortRenderParamsSwift {
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
}

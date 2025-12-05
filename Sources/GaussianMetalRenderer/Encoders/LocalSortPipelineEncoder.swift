import Metal

/// Local sort pipeline encoder - Memory-efficient deterministic Gaussian splatting
/// Uses two-pass projection: Visibility → PrefixSum → ProjectScatter
/// Pipeline: Clear → ProjectVisibility → PrefixScan → ProjectScatter → TilePrefixScan → Scatter → Sort → Render
public final class LocalPipelineEncoder {
    // Device and library
    private let device: MTLDevice
    private let LocalLibrary: MTLLibrary

    /// Public access to the Metal library for shared encoders
    public var library: MTLLibrary { self.LocalLibrary }

    // Per-stage encoders
    private let clearEncoder: LocalClearEncoder
    private let projectEncoder: LocalProjectEncoder
    private let prefixScanEncoder: LocalPrefixScanEncoder
    private let scatterEncoder: LocalScatterEncoder
    private let sortEncoder: LocalSortEncoder
    private let renderEncoder: LocalRenderEncoder

    public init(device: MTLDevice) throws {
        self.device = device

        // Load Local sort shader library
        guard let libraryURL = Bundle.module.url(forResource: "LocalShaders", withExtension: "metallib"),
              let library = try? device.makeLibrary(URL: libraryURL)
        else {
            fatalError("Failed to load LocalShaders.metallib")
        }
        self.LocalLibrary = library

        // Load main library for hierarchical prefix sum
        guard let mainLibraryURL = Bundle.module.url(forResource: "GaussianMetalRenderer", withExtension: "metallib"),
              let mainLibrary = try? device.makeLibrary(URL: mainLibraryURL)
        else {
            fatalError("Failed to load GaussianMetalRenderer.metallib")
        }

        // Initialize per-stage encoders
        self.clearEncoder = try LocalClearEncoder(library: library, device: device)
        self.projectEncoder = try LocalProjectEncoder(
            LocalLibrary: library,
            mainLibrary: mainLibrary,
            device: device
        )
        self.prefixScanEncoder = try LocalPrefixScanEncoder(library: library, device: device)
        self.scatterEncoder = try LocalScatterEncoder(library: library, device: device)
        self.sortEncoder = try LocalSortEncoder(library: library, device: device)
        self.renderEncoder = try LocalRenderEncoder(library: library, device: device)
    }

    /// Legacy init for backwards compatibility
    public convenience init(device: MTLDevice, library _: MTLLibrary) throws {
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
        maxAssignments _: Int,
        // Temp buffer for single-pass projection
        tempProjectionBuffer: MTLBuffer,
        // Visibility prefix sum buffers
        visibilityMarks: MTLBuffer,
        visibilityPartialSums: MTLBuffer,
        // Active tile buffers (populated during prefix scan)
        activeTileIndices: MTLBuffer,
        activeTileCount: MTLBuffer,
        // Options
        useHalfWorld: Bool = false,
        skipSort: Bool = false,
        tempSortKeys _: MTLBuffer? = nil,
        tempSortIndices _: MTLBuffer? = nil,
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
        self.clearEncoder.encode(
            commandBuffer: commandBuffer,
            tileCounts: tileCounts,
            header: compactedHeader,
            tileCount: tileCount,
            maxCompacted: maxCompacted
        )

        // === 2. PROJECT + VISIBILITY PREFIX SUM + COMPACT (no counting) ===
        self.projectEncoder.encode(
            commandBuffer: commandBuffer,
            worldGaussians: worldGaussians,
            harmonics: harmonics,
            camera: camera,
            params: params,
            gaussianCount: gaussianCount,
            tempProjectionBuffer: tempProjectionBuffer,
            visibilityMarks: visibilityMarks,
            visibilityPartialSums: visibilityPartialSums,
            compactedGaussians: compactedGaussians,
            compactedHeader: compactedHeader,
            useHalfWorld: useHalfWorld,
            clusterVisibility: clusterVisibility,
            clusterSize: clusterSize
        )

        // maxPerTile for fixed layout (same as SORT_MAX_SIZE in shader)
        let maxPerTile = 2048

        // === 3. SCATTER (fixed layout: tileId * maxPerTile, atomics give counts) ===
        let effective16Bit = use16BitSort && self.scatterEncoder.has16BitScatter && depthKeys16 != nil
        if effective16Bit, let depth16 = depthKeys16 {
            self.scatterEncoder.encode16(
                commandBuffer: commandBuffer,
                compactedGaussians: compactedGaussians,
                compactedHeader: compactedHeader,
                tileCounters: tileCounts,
                depthKeys16: depth16,
                globalIndices: sortIndices,
                tilesX: tilesX,
                maxPerTile: maxPerTile,
                tileWidth: tileWidth,
                tileHeight: tileHeight
            )
        } else if let sortKeysBuf = sortKeys {
            self.scatterEncoder.encode(
                commandBuffer: commandBuffer,
                compactedGaussians: compactedGaussians,
                compactedHeader: compactedHeader,
                tileCounters: tileCounts,
                sortKeys: sortKeysBuf,
                sortIndices: sortIndices,
                tilesX: tilesX,
                maxPerTile: maxPerTile,
                tileWidth: tileWidth,
                tileHeight: tileHeight
            )
        }

        // === 4. TILE PREFIX SCAN + ACTIVE TILE COMPACTION ===
        // (Now AFTER scatter - reads tile counts populated by scatter atomics)
        self.prefixScanEncoder.encode(
            commandBuffer: commandBuffer,
            tileCounts: tileCounts,
            tileOffsets: tileOffsets,
            partialSums: partialSums,
            tileCount: tileCount,
            activeTileIndices: activeTileIndices,
            activeTileCount: activeTileCount
        )

        // === 5. PER-TILE SORT (fixed layout: tileId * maxPerTile) ===
        if !skipSort, let sortKeysBuf = sortKeys {
            self.sortEncoder.encode(
                commandBuffer: commandBuffer,
                sortKeys: sortKeysBuf,
                sortIndices: sortIndices,
                tileCounts: tileCounts,
                maxPerTile: maxPerTile,
                tileCount: tileCount
            )
        }
    }

    /// Texture width for render texture (must match RENDER_TEX_WIDTH in shader)
    public static let renderTexWidth = LocalRenderEncoder.renderTexWidth

    /// Check if 16-bit sort is available
    public var has16BitSort: Bool {
        self.scatterEncoder.has16BitScatter && self.sortEncoder.has16BitSort && self.renderEncoder.has16BitRender
    }

    /// Read visible count from header buffer (call after command buffer completes)
    public static func readVisibleCount(from header: MTLBuffer) -> UInt32 {
        let ptr = header.contents().bindMemory(to: CompactedHeaderSwift.self, capacity: 1)
        return ptr.pointee.visibleCount
    }

    /// Read overflow flag from header buffer
    public static func readOverflow(from header: MTLBuffer) -> Bool {
        let ptr = header.contents().bindMemory(to: CompactedHeaderSwift.self, capacity: 1)
        return ptr.pointee.overflow != 0
    }

    // MARK: - 16-bit Sort Experimental

    /// Encode 16-bit scatter (fixed layout: tileId * maxPerTile)
    public func encodeScatter16(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        tileCounters: MTLBuffer,
        depthKeys16: MTLBuffer,
        globalIndices: MTLBuffer,
        tilesX: Int,
        maxPerTile: Int,
        tileWidth: Int,
        tileHeight: Int
    ) {
        self.scatterEncoder.encode16(
            commandBuffer: commandBuffer,
            compactedGaussians: compactedGaussians,
            compactedHeader: compactedHeader,
            tileCounters: tileCounters,
            depthKeys16: depthKeys16,
            globalIndices: globalIndices,
            tilesX: tilesX,
            maxPerTile: maxPerTile,
            tileWidth: tileWidth,
            tileHeight: tileHeight
        )
    }

    /// Encode 16-bit per-tile sort (fixed layout: tileId * maxPerTile)
    public func encodeSort16(
        commandBuffer: MTLCommandBuffer,
        depthKeys16: MTLBuffer,
        globalIndices: MTLBuffer,
        sortedLocalIdx: MTLBuffer,
        tileCounts: MTLBuffer,
        maxPerTile: Int,
        tileCount: Int
    ) {
        self.sortEncoder.encode16(
            commandBuffer: commandBuffer,
            depthKeys16: depthKeys16,
            globalIndices: globalIndices,
            sortedLocalIdx: sortedLocalIdx,
            tileCounts: tileCounts,
            maxPerTile: maxPerTile,
            tileCount: tileCount
        )
    }

    // MARK: - Indirect Dispatch Methods

    /// Clear color and depth textures before indirect render
    public func encodeClearTextures(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        width: Int,
        height: Int,
        whiteBackground: Bool
    ) {
        self.renderEncoder.encodeClearTextures(
            commandBuffer: commandBuffer,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            width: width,
            height: height,
            whiteBackground: whiteBackground
        )
    }

    /// Prepare indirect dispatch arguments
    public func encodePrepareRenderDispatch(
        commandBuffer: MTLCommandBuffer,
        activeTileCount: MTLBuffer,
        dispatchArgs: MTLBuffer
    ) {
        self.renderEncoder.encodePrepareRenderDispatch(
            commandBuffer: commandBuffer,
            activeTileCount: activeTileCount,
            dispatchArgs: dispatchArgs
        )
    }

    /// Render only active tiles using indirect dispatch (fixed layout)
    public func encodeRenderIndirect(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        tileCounts: MTLBuffer,
        maxPerTile: Int,
        sortedIndices: MTLBuffer,
        activeTileIndices: MTLBuffer,
        dispatchArgs: MTLBuffer,
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
        self.renderEncoder.encodeIndirect(
            commandBuffer: commandBuffer,
            compactedGaussians: compactedGaussians,
            tileCounts: tileCounts,
            maxPerTile: maxPerTile,
            sortedIndices: sortedIndices,
            activeTileIndices: activeTileIndices,
            dispatchArgs: dispatchArgs,
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

    /// Render only active tiles using indirect dispatch (16-bit path, fixed layout)
    public func encodeRenderIndirect16(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        tileCounts: MTLBuffer,
        maxPerTile: Int,
        sortedLocalIdx: MTLBuffer,
        globalIndices: MTLBuffer,
        activeTileIndices: MTLBuffer,
        dispatchArgs: MTLBuffer,
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
        self.renderEncoder.encodeIndirect16(
            commandBuffer: commandBuffer,
            compactedGaussians: compactedGaussians,
            tileCounts: tileCounts,
            maxPerTile: maxPerTile,
            sortedLocalIdx: sortedLocalIdx,
            globalIndices: globalIndices,
            activeTileIndices: activeTileIndices,
            dispatchArgs: dispatchArgs,
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
public struct LocalRenderParamsSwift {
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

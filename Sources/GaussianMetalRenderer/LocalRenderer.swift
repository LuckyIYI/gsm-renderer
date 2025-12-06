import Metal
import simd

/// Fast Gaussian splatting renderer using per-tile local sort
public final class LocalRenderer: GaussianRenderer, @unchecked Sendable {
    // MARK: - Constants

    private static let tileWidth = 16
    private static let tileHeight = 16
    private static let maxGaussiansPerTile32Bit = 2048
    private static let maxGaussiansPerTile16Bit = 4096
    private static let maxPerTile = 2048 // Fixed layout sort capacity
    private static let partialSumsCapacity = 1024
    private static let renderTexWidth = 4096

    // MARK: - Public Properties

    public let device: MTLDevice

    /// Last GPU execution time in seconds
    public private(set) var lastGPUTime: Double?

    /// Y-axis flip for coordinate system conversion
    public var flipY: Bool = false

    /// Use shared storage mode for debugging
    public var useSharedBuffers: Bool = false

    // MARK: - Private Properties

    private let config: RendererConfig

    // Stage encoders (internal helpers)
    private let clearEncoder: LocalClearEncoder
    private let projectEncoder: LocalProjectEncoder
    private let prefixScanEncoder: LocalPrefixScanEncoder
    private let scatterEncoder: LocalScatterEncoder
    private let sortEncoder: LocalSortEncoder
    private let renderEncoder: LocalRenderEncoder

    private var maxGaussiansPerTile: Int {
        self.config.sortMode == .sort16Bit ? LocalRenderer.maxGaussiansPerTile16Bit : LocalRenderer.maxGaussiansPerTile32Bit
    }

    // Core buffers
    private var compactedBuffer: MTLBuffer?
    private var headerBuffer: MTLBuffer?
    private var tileCountsBuffer: MTLBuffer?
    private var tileOffsetsBuffer: MTLBuffer?
    private var partialSumsBuffer: MTLBuffer?
    private var sortIndicesBuffer: MTLBuffer?
    private var colorTexture: MTLTexture?
    private var depthTexture: MTLTexture?

    // 32-bit sort buffers
    private var sortKeysBuffer: MTLBuffer?

    // 16-bit sort buffers
    private var depthKeys16Buffer: MTLBuffer?
    private var sortedLocalIdx16Buffer: MTLBuffer?

    // Projection buffers
    private var tempProjectionBuffer: MTLBuffer?
    private var visibilityMarksBuffer: MTLBuffer?
    private var visibilityPartialSumsBuffer: MTLBuffer?

    // Indirect dispatch buffers
    private var activeTileIndicesBuffer: MTLBuffer?
    private var activeTileCountBuffer: MTLBuffer?
    private var dispatchArgsBuffer: MTLBuffer?

    // Allocation tracking
    private var allocatedGaussianCapacity: Int = 0
    private var allocatedWidth: Int = 0
    private var allocatedHeight: Int = 0

    // MARK: - Initialization

    public init(device: MTLDevice? = nil, config: RendererConfig = RendererConfig()) throws {
        let device = device ?? MTLCreateSystemDefaultDevice()
        guard let device else {
            throw RendererError.deviceNotAvailable
        }
        self.device = device
        self.config = config

        // Load shader libraries
        guard let localLibraryURL = Bundle.module.url(forResource: "LocalShaders", withExtension: "metallib"),
              let localLibrary = try? device.makeLibrary(URL: localLibraryURL)
        else {
            throw RendererError.failedToCreatePipeline("LocalShaders.metallib not found")
        }

        guard let mainLibraryURL = Bundle.module.url(forResource: "GaussianMetalRenderer", withExtension: "metallib"),
              let mainLibrary = try? device.makeLibrary(URL: mainLibraryURL)
        else {
            throw RendererError.failedToCreatePipeline("GaussianMetalRenderer.metallib not found")
        }

        // Initialize stage encoders
        self.clearEncoder = try LocalClearEncoder(library: localLibrary, device: device)
        self.projectEncoder = try LocalProjectEncoder(LocalLibrary: localLibrary, mainLibrary: mainLibrary, device: device)
        self.prefixScanEncoder = try LocalPrefixScanEncoder(library: localLibrary, device: device)
        self.scatterEncoder = try LocalScatterEncoder(library: localLibrary, device: device)
        self.sortEncoder = try LocalSortEncoder(library: localLibrary, device: device)
        self.renderEncoder = try LocalRenderEncoder(library: localLibrary, device: device)

        // Enable sparse scatter by default (22-38% faster)
        self.projectEncoder.skipCompaction = true
        self.scatterEncoder.activeVariant = .sparse
    }

    // MARK: - GaussianRenderer Protocol Methods

    public func render(
        toTexture commandBuffer: MTLCommandBuffer,
        input: GaussianInput,
        camera: CameraParams,
        width: Int,
        height: Int,
        whiteBackground: Bool
    ) -> TextureRenderResult? {
        guard let colorTex = renderInternal(
            commandBuffer: commandBuffer,
            worldGaussians: input.gaussians,
            harmonics: input.harmonics,
            gaussianCount: input.gaussianCount,
            viewMatrix: camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            cameraPosition: camera.position,
            focalX: camera.focalX,
            focalY: camera.focalY,
            width: width,
            height: height,
            shComponents: input.shComponents,
            whiteBackground: whiteBackground,
            useHalfWorld: self.config.precision == .float16
        ) else { return nil }

        return TextureRenderResult(color: colorTex, depth: self.depthTexture, alpha: nil)
    }

    public func render(
        toBuffer _: MTLCommandBuffer,
        input _: GaussianInput,
        camera _: CameraParams,
        width _: Int,
        height _: Int,
        whiteBackground _: Bool
    ) -> BufferRenderResult? {
        nil // LocalRenderer only supports texture output
    }

    // MARK: - Public Helpers

    public func getVisibleCount() -> UInt32 {
        guard let header = headerBuffer else { return 0 }
        let ptr = header.contents().bindMemory(to: CompactedHeaderSwift.self, capacity: 1)
        return ptr.pointee.visibleCount
    }

    public func hadOverflow() -> Bool {
        guard let header = headerBuffer else { return false }
        let ptr = header.contents().bindMemory(to: CompactedHeaderSwift.self, capacity: 1)
        return ptr.pointee.overflow != 0
    }

    // MARK: - Internal Render Implementation

    private func renderInternal(
        commandBuffer: MTLCommandBuffer,
        worldGaussians: MTLBuffer,
        harmonics: MTLBuffer,
        gaussianCount: Int,
        viewMatrix: simd_float4x4,
        projectionMatrix: simd_float4x4,
        cameraPosition: SIMD3<Float>,
        focalX: Float,
        focalY: Float,
        width: Int,
        height: Int,
        shComponents: Int = 0,
        whiteBackground: Bool = false,
        useHalfWorld: Bool = false
    ) -> MTLTexture? {
        guard gaussianCount > 0, width > 0, height > 0 else { return nil }

        self.ensureBuffers(gaussianCount: gaussianCount, width: width, height: height)

        // Validate required buffers
        guard let compacted = compactedBuffer,
              let header = headerBuffer,
              let tileCounts = tileCountsBuffer,
              let tileOffsets = tileOffsetsBuffer,
              let partialSums = partialSumsBuffer,
              let sortIndices = sortIndicesBuffer,
              let tempProjection = tempProjectionBuffer,
              let visibilityMarks = visibilityMarksBuffer,
              let visibilityPartialSums = visibilityPartialSumsBuffer,
              let colorTex = colorTexture,
              let depthTex = depthTexture,
              let activeTileIndices = activeTileIndicesBuffer,
              let activeTileCount = activeTileCountBuffer,
              let dispatchArgs = dispatchArgsBuffer
        else {
            return nil
        }

        let tilesX = (width + LocalRenderer.tileWidth - 1) / LocalRenderer.tileWidth
        let tilesY = (height + LocalRenderer.tileHeight - 1) / LocalRenderer.tileHeight
        let tileCount = tilesX * tilesY
        let maxCompacted = gaussianCount

        var projMatrix = projectionMatrix
        if self.flipY {
            projMatrix[1][1] = -projMatrix[1][1]
        }

        let camera = CameraUniformsSwift(
            viewMatrix: viewMatrix,
            projectionMatrix: projMatrix,
            cameraCenter: cameraPosition,
            pixelFactor: 1.0,
            focalX: focalX,
            focalY: focalY,
            width: Float(width),
            height: Float(height),
            nearPlane: 0.1,
            farPlane: 100.0,
            shComponents: UInt32(shComponents),
            gaussianCount: UInt32(gaussianCount)
        )

        let params = TileBinningParams(
            gaussianCount: UInt32(gaussianCount),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            tileWidth: UInt32(LocalRenderer.tileWidth),
            tileHeight: UInt32(LocalRenderer.tileHeight),
            surfaceWidth: UInt32(width),
            surfaceHeight: UInt32(height),
            maxCapacity: UInt32(maxCompacted)
        )

        // Check if 16-bit sort is available and enabled
        let use16BitSort = self.config.sortMode == .sort16Bit &&
            self.scatterEncoder.has16BitScatter &&
            self.sortEncoder.has16BitSort &&
            self.renderEncoder.has16BitRender &&
            self.depthKeys16Buffer != nil &&
            self.sortedLocalIdx16Buffer != nil

        // === PIPELINE STAGE 1: CLEAR ===
        self.clearEncoder.encode(
            commandBuffer: commandBuffer,
            tileCounts: tileCounts,
            header: header,
            tileCount: tileCount,
            maxCompacted: maxCompacted
        )

        // === PIPELINE STAGE 2: PROJECT + VISIBILITY PREFIX SUM + COMPACT ===
        self.projectEncoder.encode(
            commandBuffer: commandBuffer,
            worldGaussians: worldGaussians,
            harmonics: harmonics,
            camera: camera,
            params: params,
            gaussianCount: gaussianCount,
            tempProjectionBuffer: tempProjection,
            visibilityMarks: visibilityMarks,
            visibilityPartialSums: visibilityPartialSums,
            compactedGaussians: compacted,
            compactedHeader: header,
            useHalfWorld: useHalfWorld,
            clusterVisibility: nil,
            clusterSize: UInt32(CLUSTER_SIZE)
        )

        // === PIPELINE STAGE 3: SCATTER (fixed layout: tileId * maxPerTile) ===
        // For sparse mode: set visibility mask and gaussian count
        let useSparseScatter = self.projectEncoder.skipCompaction
        if useSparseScatter {
            self.scatterEncoder.visibilityMask = visibilityMarks
            self.scatterEncoder.totalGaussianCount = gaussianCount
        }

        if use16BitSort, let depth16 = depthKeys16Buffer {
            let gaussianBuffer = useSparseScatter ? tempProjection : compacted
            self.scatterEncoder.encode16(
                commandBuffer: commandBuffer,
                compactedGaussians: gaussianBuffer,
                compactedHeader: header,
                tileCounters: tileCounts,
                depthKeys16: depth16,
                globalIndices: sortIndices,
                tilesX: tilesX,
                maxPerTile: LocalRenderer.maxPerTile,
                tileWidth: LocalRenderer.tileWidth,
                tileHeight: LocalRenderer.tileHeight
            )
        } else if let sortKeys = sortKeysBuffer {
            self.scatterEncoder.encode(
                commandBuffer: commandBuffer,
                compactedGaussians: compacted,
                compactedHeader: header,
                tileCounters: tileCounts,
                sortKeys: sortKeys,
                sortIndices: sortIndices,
                tilesX: tilesX,
                maxPerTile: LocalRenderer.maxPerTile,
                tileWidth: LocalRenderer.tileWidth,
                tileHeight: LocalRenderer.tileHeight
            )
        }

        // === PIPELINE STAGE 4: TILE PREFIX SCAN + ACTIVE TILE COMPACTION ===
        self.prefixScanEncoder.encode(
            commandBuffer: commandBuffer,
            tileCounts: tileCounts,
            tileOffsets: tileOffsets,
            partialSums: partialSums,
            tileCount: tileCount,
            activeTileIndices: activeTileIndices,
            activeTileCount: activeTileCount
        )

        // === PIPELINE STAGE 5: PER-TILE SORT ===
        if use16BitSort, let depth16 = depthKeys16Buffer, let sortedLocal16 = sortedLocalIdx16Buffer {
            self.sortEncoder.encode16(
                commandBuffer: commandBuffer,
                depthKeys16: depth16,
                globalIndices: sortIndices,
                sortedLocalIdx: sortedLocal16,
                tileCounts: tileCounts,
                maxPerTile: LocalRenderer.maxPerTile,
                tileCount: tileCount
            )
        } else if let sortKeys = sortKeysBuffer {
            self.sortEncoder.encode(
                commandBuffer: commandBuffer,
                sortKeys: sortKeys,
                sortIndices: sortIndices,
                tileCounts: tileCounts,
                maxPerTile: LocalRenderer.maxPerTile,
                tileCount: tileCount
            )
        }

        // === PIPELINE STAGE 6: CLEAR TEXTURES ===
        self.renderEncoder.encodeClearTextures(
            commandBuffer: commandBuffer,
            colorTexture: colorTex,
            depthTexture: depthTex,
            width: width,
            height: height,
            whiteBackground: whiteBackground
        )

        // === PIPELINE STAGE 7: PREPARE INDIRECT DISPATCH ===
        self.renderEncoder.encodePrepareRenderDispatch(
            commandBuffer: commandBuffer,
            activeTileCount: activeTileCount,
            dispatchArgs: dispatchArgs
        )

        // === PIPELINE STAGE 8: RENDER ===
        let gaussianBufferForRender = useSparseScatter ? tempProjection : compacted

        if use16BitSort, let sortedLocal16 = sortedLocalIdx16Buffer {
            self.renderEncoder.encodeIndirect16(
                commandBuffer: commandBuffer,
                compactedGaussians: gaussianBufferForRender,
                tileCounts: tileCounts,
                maxPerTile: LocalRenderer.maxPerTile,
                sortedLocalIdx: sortedLocal16,
                globalIndices: sortIndices,
                activeTileIndices: activeTileIndices,
                dispatchArgs: dispatchArgs,
                colorTexture: colorTex,
                depthTexture: depthTex,
                tilesX: tilesX,
                tilesY: tilesY,
                width: width,
                height: height,
                tileWidth: LocalRenderer.tileWidth,
                tileHeight: LocalRenderer.tileHeight,
                whiteBackground: whiteBackground
            )
        } else {
            self.renderEncoder.encodeIndirect(
                commandBuffer: commandBuffer,
                compactedGaussians: gaussianBufferForRender,
                tileCounts: tileCounts,
                maxPerTile: LocalRenderer.maxPerTile,
                sortedIndices: sortIndices,
                activeTileIndices: activeTileIndices,
                dispatchArgs: dispatchArgs,
                colorTexture: colorTex,
                depthTexture: depthTex,
                tilesX: tilesX,
                tilesY: tilesY,
                width: width,
                height: height,
                tileWidth: LocalRenderer.tileWidth,
                tileHeight: LocalRenderer.tileHeight,
                whiteBackground: whiteBackground
            )
        }

        return colorTex
    }

    // MARK: - Buffer Management

    private func ensureBuffers(gaussianCount: Int, width: Int, height: Int) {
        let tilesX = (width + LocalRenderer.tileWidth - 1) / LocalRenderer.tileWidth
        let tilesY = (height + LocalRenderer.tileHeight - 1) / LocalRenderer.tileHeight
        let tileCount = tilesX * tilesY
        let maxCompacted = gaussianCount
        let maxAssignments = tileCount * self.maxGaussiansPerTile

        let needsRealloc = gaussianCount > self.allocatedGaussianCapacity ||
            width != self.allocatedWidth ||
            height != self.allocatedHeight

        guard needsRealloc else { return }

        self.allocatedGaussianCapacity = gaussianCount
        self.allocatedWidth = width
        self.allocatedHeight = height

        let priv: MTLResourceOptions = self.useSharedBuffers ? .storageModeShared : .storageModePrivate
        let shared: MTLResourceOptions = .storageModeShared

        func makeBuffer(_ label: String, length: Int, options: MTLResourceOptions) -> MTLBuffer? {
            let buf = self.device.makeBuffer(length: length, options: options)
            buf?.label = label
            return buf
        }

        // Core buffers
        self.compactedBuffer = makeBuffer("CompactedGaussians",
            length: maxCompacted * MemoryLayout<CompactedGaussianSwift>.stride,
            options: self.useSharedBuffers ? shared : priv)
        self.headerBuffer = makeBuffer("CompactedHeader",
            length: MemoryLayout<CompactedHeaderSwift>.stride,
            options: shared)
        self.tileCountsBuffer = makeBuffer("TileCounts",
            length: tileCount * MemoryLayout<UInt32>.stride,
            options: priv)
        self.tileOffsetsBuffer = makeBuffer("TileOffsets",
            length: (tileCount + 1) * MemoryLayout<UInt32>.stride,
            options: priv)
        self.partialSumsBuffer = makeBuffer("PartialSums",
            length: LocalRenderer.partialSumsCapacity * MemoryLayout<UInt32>.stride,
            options: priv)
        self.sortIndicesBuffer = makeBuffer("SortIndices",
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: priv)

        // Projection buffers
        self.tempProjectionBuffer = makeBuffer("TempProjection",
            length: maxCompacted * MemoryLayout<CompactedGaussianSwift>.stride,
            options: priv)

        // Visibility prefix sum buffers
        let blockSize = 256
        let visBlocks = (gaussianCount + 1 + blockSize - 1) / blockSize
        let level2Blocks = (visBlocks + blockSize - 1) / blockSize
        let totalPartialSums = (visBlocks + 1) + (level2Blocks + 1)

        self.visibilityMarksBuffer = makeBuffer("VisibilityMarks",
            length: (gaussianCount + 1) * MemoryLayout<UInt32>.stride,
            options: priv)
        self.visibilityPartialSumsBuffer = makeBuffer("VisibilityPartialSums",
            length: totalPartialSums * MemoryLayout<UInt32>.stride,
            options: priv)

        // Mode-specific sort buffers
        switch self.config.sortMode {
        case .sort32Bit:
            self.sortKeysBuffer = makeBuffer("SortKeys",
                length: maxAssignments * MemoryLayout<UInt32>.stride,
                options: priv)
            self.depthKeys16Buffer = nil
            self.sortedLocalIdx16Buffer = nil

        case .sort16Bit:
            self.depthKeys16Buffer = makeBuffer("DepthKeys16",
                length: maxAssignments * MemoryLayout<UInt16>.stride,
                options: priv)
            self.sortedLocalIdx16Buffer = makeBuffer("SortedLocalIdx16",
                length: maxAssignments * MemoryLayout<UInt16>.stride,
                options: priv)
            self.sortKeysBuffer = nil
        }

        // Textures
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        colorDesc.usage = [.shaderRead, .shaderWrite]
        colorDesc.storageMode = .private
        self.colorTexture = self.device.makeTexture(descriptor: colorDesc)
        self.colorTexture?.label = "OutputColor"

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float, width: width, height: height, mipmapped: false)
        depthDesc.usage = [.shaderRead, .shaderWrite]
        depthDesc.storageMode = .private
        self.depthTexture = self.device.makeTexture(descriptor: depthDesc)
        self.depthTexture?.label = "OutputDepth"

        // Indirect dispatch buffers
        self.activeTileIndicesBuffer = makeBuffer("ActiveTileIndices",
            length: tileCount * MemoryLayout<UInt32>.stride,
            options: priv)
        self.activeTileCountBuffer = makeBuffer("ActiveTileCount",
            length: MemoryLayout<UInt32>.stride,
            options: priv)
        self.dispatchArgsBuffer = makeBuffer("DispatchArgs",
            length: 3 * MemoryLayout<UInt32>.stride,
            options: priv)
    }
}

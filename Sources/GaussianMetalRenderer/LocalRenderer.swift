import Metal
import simd

/// Fast Gaussian splatting renderer using per-tile local sort
public final class LocalRenderer: GaussianRenderer, @unchecked Sendable {
    // MARK: - Constants

    private static let tileWidth = 16
    private static let tileHeight = 16
    private static let maxGaussiansPerTile32Bit = 2048
    private static let maxGaussiansPerTile16Bit = 4096
    private static let partialSumsCapacity = 1024

    // MARK: - Public Properties

    public let device: MTLDevice
    public let encoder: LocalPipelineEncoder

    /// Last GPU execution time in seconds
    public private(set) var lastGPUTime: Double?

    /// Y-axis flip for coordinate system conversion
    public var flipY: Bool = false

    /// Use shared storage mode for debugging
    public var useSharedBuffers: Bool = false

    // MARK: - Private Properties

    private let config: RendererConfig

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
    private var tempSortKeysBuffer: MTLBuffer?
    private var tempSortIndicesBuffer: MTLBuffer?

    // 16-bit sort buffers
    private var depthKeys16Buffer: MTLBuffer?
    private var sortedLocalIdx16Buffer: MTLBuffer?

    // Projection buffers
    private var tempProjectionBuffer: MTLBuffer?
    private var visibilityMarksBuffer: MTLBuffer?
    private var visibilityPartialSumsBuffer: MTLBuffer?

    // Textured render
    private var gaussianRenderTexture: MTLTexture?

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
        self.encoder = try LocalPipelineEncoder(device: device)
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

        // Cluster culling disabled (archived)
        let clusterVisibility: MTLBuffer? = nil
        let clusterSize = UInt32(CLUSTER_SIZE)
        self.ensureBuffers(gaussianCount: gaussianCount, width: width, height: height)

        // Core buffers required for all modes
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
              let depthTex = depthTexture
        else {
            return nil
        }

        // Mode-specific buffer validation
        let sortKeys = self.sortKeysBuffer // May be nil in 16-bit mode
        let tempSortKeys = self.tempSortKeysBuffer
        let tempSortIndices = self.tempSortIndicesBuffer

        let tilesX = (width + LocalRenderer.tileWidth - 1) / LocalRenderer.tileWidth
        let tilesY = (height + LocalRenderer.tileHeight - 1) / LocalRenderer.tileHeight
        let maxCompacted = gaussianCount
        // Tile-bounded: tileCount × maxPerTile (matches ensureBuffers allocation)
        let maxAssignments = (tilesX * tilesY) * self.maxGaussiansPerTile

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

        let tileCount = tilesX * tilesY

        // Indirect dispatch buffers required for all render paths
        guard let activeTileIndices = activeTileIndicesBuffer,
              let activeTileCount = activeTileCountBuffer,
              let dispatchArgs = dispatchArgsBuffer
        else {
            return nil
        }

        // Check if 16-bit sort is available and enabled (based on config, not runtime flag)
        let effective16BitSort = self.config.sortMode == .sort16Bit && self.encoder.has16BitSort &&
            self.sortedLocalIdx16Buffer != nil && self.depthKeys16Buffer != nil

        // Efficient single-pass projection pipeline with optional cluster visibility
        // Also computes active tile list during prefix scan (fused)
        self.encoder.encode(
            commandBuffer: commandBuffer,
            worldGaussians: worldGaussians,
            harmonics: harmonics,
            camera: camera,
            gaussianCount: gaussianCount,
            tilesX: tilesX,
            tilesY: tilesY,
            tileWidth: LocalRenderer.tileWidth,
            tileHeight: LocalRenderer.tileHeight,
            surfaceWidth: width,
            surfaceHeight: height,
            compactedGaussians: compacted,
            compactedHeader: header,
            tileCounts: tileCounts,
            tileOffsets: tileOffsets,
            partialSums: partialSums,
            sortKeys: sortKeys,
            sortIndices: sortIndices,
            maxCompacted: maxCompacted,
            maxAssignments: maxAssignments,
            tempProjectionBuffer: tempProjection,
            visibilityMarks: visibilityMarks,
            visibilityPartialSums: visibilityPartialSums,
            activeTileIndices: activeTileIndices,
            activeTileCount: activeTileCount,
            useHalfWorld: useHalfWorld,
            skipSort: effective16BitSort,
            tempSortKeys: effective16BitSort ? nil : tempSortKeys,
            tempSortIndices: effective16BitSort ? nil : tempSortIndices,
            clusterVisibility: clusterVisibility,
            clusterSize: clusterSize,
            use16BitSort: effective16BitSort,
            depthKeys16: self.depthKeys16Buffer
        )

        // 1. Clear textures
        self.encoder.encodeClearTextures(
            commandBuffer: commandBuffer,
            colorTexture: colorTex,
            depthTexture: depthTex,
            width: width,
            height: height,
            whiteBackground: whiteBackground
        )

        // 2. Prepare indirect dispatch args (active tiles already computed in prefix scan)
        self.encoder.encodePrepareRenderDispatch(
            commandBuffer: commandBuffer,
            activeTileCount: activeTileCount,
            dispatchArgs: dispatchArgs
        )

        // maxPerTile for fixed layout (same as SORT_MAX_SIZE in shader)
        let maxPerTile = 2048

        // In sparse mode, render reads from tempProjection (sparse, original indices)
        // In normal mode, render reads from compacted (dense, compacted indices)
        let gaussianBufferForRender = self.encoder.useSparseScatter ? tempProjection : compacted

        // 16-bit sort path - reads depthKeys16 directly (sequential 2-byte reads!)
        if effective16BitSort, let sortedLocal16 = sortedLocalIdx16Buffer, let depth16 = depthKeys16Buffer {
            self.encoder.encodeSort16(
                commandBuffer: commandBuffer,
                depthKeys16: depth16, // Sequential 2-byte reads - much faster than sortInfo!
                globalIndices: sortIndices,
                sortedLocalIdx: sortedLocal16,
                tileCounts: tileCounts,
                maxPerTile: maxPerTile,
                tileCount: tileCount
            )

            // 16-bit render via indirect dispatch (two-level indirection)
            self.encoder.encodeRenderIndirect16(
                commandBuffer: commandBuffer,
                compactedGaussians: gaussianBufferForRender,
                tileCounts: tileCounts,
                maxPerTile: maxPerTile,
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
            // 32-bit render via indirect dispatch (fixed layout)
            self.encoder.encodeRenderIndirect(
                commandBuffer: commandBuffer,
                compactedGaussians: gaussianBufferForRender,
                tileCounts: tileCounts,
                maxPerTile: maxPerTile,
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

    public func getVisibleCount() -> UInt32 {
        guard let header = headerBuffer else { return 0 }
        return LocalPipelineEncoder.readVisibleCount(from: header)
    }

    public func hadOverflow() -> Bool {
        guard let header = headerBuffer else { return false }
        return LocalPipelineEncoder.readOverflow(from: header)
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

        // Helper to create and label buffer
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
            self.tempSortKeysBuffer = makeBuffer("TempSortKeys",
                length: maxAssignments * MemoryLayout<UInt32>.stride,
                options: priv)
            self.tempSortIndicesBuffer = makeBuffer("TempSortIndices",
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
            self.tempSortKeysBuffer = nil
            self.tempSortIndicesBuffer = nil
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

        // Textured render texture
        if self.config.useTexturedRender {
            let renderTexWidth = LocalPipelineEncoder.renderTexWidth
            let texelCount = maxCompacted * 2
            let renderTexHeight = (texelCount + renderTexWidth - 1) / renderTexWidth
            let gaussianTexDesc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba32Float, width: renderTexWidth, height: max(1, renderTexHeight), mipmapped: false)
            gaussianTexDesc.usage = [.shaderRead, .shaderWrite]
            gaussianTexDesc.storageMode = .private
            self.gaussianRenderTexture = self.device.makeTexture(descriptor: gaussianTexDesc)
            self.gaussianRenderTexture?.label = "GaussianRenderTexture"
        } else {
            self.gaussianRenderTexture = nil
        }

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

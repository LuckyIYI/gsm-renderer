import Metal
import simd

/// Fast Gaussian splatting renderer using per-tile local sort
public final class LocalRenderer: GaussianRenderer, @unchecked Sendable {
    // MARK: - Constants

    private static let tileWidth = 16
    private static let tileHeight = 16
    private static let maxGaussiansPerTile16Bit = 4096
    private static let maxPerTile = 2048 // Fixed layout sort capacity
    private static let partialSumsCapacity = 1024

    // MARK: - Public Properties

    public let device: MTLDevice

    /// Last GPU execution time in seconds
    public private(set) var lastGPUTime: Double?

    /// Y-axis flip for coordinate system conversion
    public var flipY: Bool = false

    // MARK: - Private Properties

    private let config: RendererConfig

    // Stage encoders (internal helpers)
    private let clearEncoder: LocalClearEncoder
    private let projectEncoder: LocalProjectEncoder
    private let prefixScanEncoder: LocalPrefixScanEncoder
    private let scatterEncoder: LocalScatterEncoder
    private let sortEncoder: LocalSortEncoder
    private let renderEncoder: LocalRenderEncoder

    private var maxGaussiansPerTile: Int { LocalRenderer.maxGaussiansPerTile16Bit }

    // Pre-allocated core buffers
    private let headerBuffer: MTLBuffer
    private let tileCountsBuffer: MTLBuffer
    private let tileOffsetsBuffer: MTLBuffer
    private let partialSumsBuffer: MTLBuffer
    private let sortIndicesBuffer: MTLBuffer

    // Pre-allocated 16-bit sort buffers (sparse mode)
    private let depthKeys16Buffer: MTLBuffer
    private let sortedLocalIdx16Buffer: MTLBuffer

    // Pre-allocated projection buffer (sparse mode - no compaction)
    private let projectionBuffer: MTLBuffer

    // Pre-allocated indirect dispatch buffers
    private let activeTileIndicesBuffer: MTLBuffer
    private let activeTileCountBuffer: MTLBuffer
    private let dispatchArgsBuffer: MTLBuffer

    // Pre-computed limits from config
    private let maxTileCount: Int
    private let maxAssignments: Int

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

        guard let mainLibraryURL = Bundle.module.url(forResource: "GlobalShaders", withExtension: "metallib"),
              let mainLibrary = try? device.makeLibrary(URL: mainLibraryURL)
        else {
            throw RendererError.failedToCreatePipeline("GlobalShaders.metallib not found")
        }

        // Initialize stage encoders
        self.clearEncoder = try LocalClearEncoder(library: localLibrary, device: device)
        self.projectEncoder = try LocalProjectEncoder(LocalLibrary: localLibrary, mainLibrary: mainLibrary, device: device)
        self.prefixScanEncoder = try LocalPrefixScanEncoder(library: localLibrary, device: device)
        self.scatterEncoder = try LocalScatterEncoder(library: localLibrary, device: device)
        self.sortEncoder = try LocalSortEncoder(library: localLibrary, device: device)
        self.renderEncoder = try LocalRenderEncoder(library: localLibrary, device: device)

        // Pre-compute limits from config
        let tilesX = (config.maxWidth + LocalRenderer.tileWidth - 1) / LocalRenderer.tileWidth
        let tilesY = (config.maxHeight + LocalRenderer.tileHeight - 1) / LocalRenderer.tileHeight
        let tileCount = tilesX * tilesY
        self.maxTileCount = tileCount
        self.maxAssignments = tileCount * LocalRenderer.maxGaussiansPerTile16Bit

        // Pre-allocate all buffers at init
        let priv: MTLResourceOptions = .storageModePrivate
        let shared: MTLResourceOptions = .storageModeShared

        self.headerBuffer = try device.makeBuffer(count: 1, type: CompactedHeaderSwift.self, options: shared, label: "CompactedHeader")
        self.tileCountsBuffer = try device.makeBuffer(count: tileCount, type: UInt32.self, options: priv, label: "TileCounts")
        self.tileOffsetsBuffer = try device.makeBuffer(count: tileCount + 1, type: UInt32.self, options: priv, label: "TileOffsets")
        self.partialSumsBuffer = try device.makeBuffer(count: LocalRenderer.partialSumsCapacity, type: UInt32.self, options: priv, label: "PartialSums")
        self.sortIndicesBuffer = try device.makeBuffer(count: self.maxAssignments, type: UInt32.self, options: priv, label: "SortIndices")
        self.projectionBuffer = try device.makeBuffer(count: config.maxGaussians, type: ProjectedGaussianSwift.self, options: priv, label: "Projection")
        self.depthKeys16Buffer = try device.makeBuffer(count: self.maxAssignments, type: UInt16.self, options: priv, label: "DepthKeys16")
        self.sortedLocalIdx16Buffer = try device.makeBuffer(count: self.maxAssignments, type: UInt16.self, options: priv, label: "SortedLocalIdx16")
        self.activeTileIndicesBuffer = try device.makeBuffer(count: tileCount, type: UInt32.self, options: priv, label: "ActiveTileIndices")
        self.activeTileCountBuffer = try device.makeBuffer(count: 1, type: UInt32.self, options: priv, label: "ActiveTileCount")
        self.dispatchArgsBuffer = try device.makeBuffer(count: 3, type: UInt32.self, options: priv, label: "DispatchArgs")
    }

    // MARK: - GaussianRenderer Protocol Methods

    public func render(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        input: GaussianInput,
        camera: CameraParams,
        width: Int,
        height: Int,
        whiteBackground: Bool
    ) {
        renderInternal(
            commandBuffer: commandBuffer,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
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
        )
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
        let ptr = headerBuffer.contents().bindMemory(to: CompactedHeaderSwift.self, capacity: 1)
        return ptr.pointee.visibleCount
    }

    public func hadOverflow() -> Bool {
        let ptr = headerBuffer.contents().bindMemory(to: CompactedHeaderSwift.self, capacity: 1)
        return ptr.pointee.overflow != 0
    }

    // MARK: - Internal Render Implementation

    private func renderInternal(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
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
    ) {
        guard gaussianCount > 0, width > 0, height > 0 else { return }

        // Validate limits
        precondition(gaussianCount <= config.maxGaussians,
                     "gaussianCount (\(gaussianCount)) exceeds config.maxGaussians (\(config.maxGaussians))")
        precondition(width <= config.maxWidth,
                     "width (\(width)) exceeds config.maxWidth (\(config.maxWidth))")
        precondition(height <= config.maxHeight,
                     "height (\(height)) exceeds config.maxHeight (\(config.maxHeight))")

        let tilesX = (width + LocalRenderer.tileWidth - 1) / LocalRenderer.tileWidth
        let tilesY = (height + LocalRenderer.tileHeight - 1) / LocalRenderer.tileHeight
        let tileCount = tilesX * tilesY

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
            maxCapacity: UInt32(gaussianCount)
        )

        // === PIPELINE STAGE 1: CLEAR ===
        self.clearEncoder.encode(
            commandBuffer: commandBuffer,
            tileCounts: tileCountsBuffer,
            header: headerBuffer,
            tileCount: tileCount,
            maxCompacted: gaussianCount
        )

        // === PIPELINE STAGE 2: PROJECT (sparse - no compaction) ===
        self.projectEncoder.encode(
            commandBuffer: commandBuffer,
            worldGaussians: worldGaussians,
            harmonics: harmonics,
            camera: camera,
            params: params,
            gaussianCount: gaussianCount,
            projectionBuffer: projectionBuffer,
            compactedHeader: headerBuffer,
            useHalfWorld: useHalfWorld,
            clusterVisibility: nil,
            clusterSize: UInt32(CLUSTER_SIZE)
        )

        // === PIPELINE STAGE 3: SCATTER (sparse mode - no compaction) ===
        self.scatterEncoder.totalGaussianCount = gaussianCount
        self.scatterEncoder.encode16(
            commandBuffer: commandBuffer,
            projectedGaussians: projectionBuffer,
            compactedHeader: headerBuffer,
            tileCounters: tileCountsBuffer,
            depthKeys16: depthKeys16Buffer,
            globalIndices: sortIndicesBuffer,
            tilesX: tilesX,
            maxPerTile: LocalRenderer.maxPerTile,
            tileWidth: LocalRenderer.tileWidth,
            tileHeight: LocalRenderer.tileHeight
        )

        // === PIPELINE STAGE 4: TILE PREFIX SCAN + ACTIVE TILE COMPACTION ===
        self.prefixScanEncoder.encode(
            commandBuffer: commandBuffer,
            tileCounts: tileCountsBuffer,
            tileOffsets: tileOffsetsBuffer,
            partialSums: partialSumsBuffer,
            tileCount: tileCount,
            activeTileIndices: activeTileIndicesBuffer,
            activeTileCount: activeTileCountBuffer
        )

        // === PIPELINE STAGE 5: PER-TILE SORT (16-bit only) ===
        self.sortEncoder.encode16(
            commandBuffer: commandBuffer,
            depthKeys16: depthKeys16Buffer,
            globalIndices: sortIndicesBuffer,
            sortedLocalIdx: sortedLocalIdx16Buffer,
            tileCounts: tileCountsBuffer,
            maxPerTile: LocalRenderer.maxPerTile,
            tileCount: tileCount
        )

        // === PIPELINE STAGE 6: CLEAR TEXTURES ===
        self.renderEncoder.encodeClearTextures(
            commandBuffer: commandBuffer,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            width: width,
            height: height,
            whiteBackground: whiteBackground
        )

        // === PIPELINE STAGE 7: PREPARE INDIRECT DISPATCH ===
        self.renderEncoder.encodePrepareRenderDispatch(
            commandBuffer: commandBuffer,
            activeTileCount: activeTileCountBuffer,
            dispatchArgs: dispatchArgsBuffer
        )

        // === PIPELINE STAGE 8: RENDER (16-bit only) ===
        self.renderEncoder.encodeIndirect16(
            commandBuffer: commandBuffer,
            projectedGaussians: projectionBuffer,
            tileCounts: tileCountsBuffer,
            maxPerTile: LocalRenderer.maxPerTile,
            sortedLocalIdx: sortedLocalIdx16Buffer,
            globalIndices: sortIndicesBuffer,
            activeTileIndices: activeTileIndicesBuffer,
            dispatchArgs: dispatchArgsBuffer,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            tilesX: tilesX,
            tilesY: tilesY,
            width: width,
            height: height,
            tileWidth: LocalRenderer.tileWidth,
            tileHeight: LocalRenderer.tileHeight,
            whiteBackground: whiteBackground
        )
    }
}

import Metal
import simd

public final class LocalRenderer: GaussianRenderer, @unchecked Sendable {
    private static let tileWidth = 16
    private static let tileHeight = 16
    private static let maxGaussiansPerTile16Bit = 2048
    private static let maxPerTile = 2048
    private static let partialSumsCapacity = 1024

    public let device: MTLDevice

    /// Last GPU execution time in seconds
    public private(set) var lastGPUTime: Double?

    private let config: RendererConfig

    // Stage encoders (internal helpers)
    private let clearEncoder: LocalClearEncoder
    private let projectCullEncoder: LocalProjectCullEncoder
    private let prefixScanEncoder: LocalPrefixScanEncoder
    private let scatterEncoder: LocalScatterEncoder
    private let sortEncoder: LocalSortEncoder
    private let renderEncoder: LocalRenderEncoder

    private var maxGaussiansPerTile: Int { LocalRenderer.maxGaussiansPerTile16Bit }

    // Pre-computed limits from config
    private let maxTileCount: Int
    private let maxAssignments: Int
    private let tilesX: Int
    private let tilesY: Int

    // Resource management: stereo resources (contains left and right view buffers)
    private let stereoResources: LocalMultiViewResources

    // Convenience accessors for mono rendering (uses left view resources)
    private var primaryResources: LocalViewResources { stereoResources.left }

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
        self.projectCullEncoder = try LocalProjectCullEncoder(LocalLibrary: localLibrary, mainLibrary: mainLibrary, device: device)
        self.prefixScanEncoder = try LocalPrefixScanEncoder(library: localLibrary, device: device)
        self.scatterEncoder = try LocalScatterEncoder(library: localLibrary, device: device)
        self.sortEncoder = try LocalSortEncoder(library: localLibrary, device: device)
        self.renderEncoder = try LocalRenderEncoder(library: localLibrary, device: device)

        // Pre-compute limits from config
        self.tilesX = (config.maxWidth + LocalRenderer.tileWidth - 1) / LocalRenderer.tileWidth
        self.tilesY = (config.maxHeight + LocalRenderer.tileHeight - 1) / LocalRenderer.tileHeight
        self.maxTileCount = tilesX * tilesY
        self.maxAssignments = maxTileCount * LocalRenderer.maxGaussiansPerTile16Bit

        // Create stereo resources (left/right view buffer sets)
        self.stereoResources = try LocalMultiViewResources(
            device: device,
            maxGaussians: config.maxGaussians,
            maxWidth: config.maxWidth,
            maxHeight: config.maxHeight,
            tileWidth: LocalRenderer.tileWidth,
            tileHeight: LocalRenderer.tileHeight,
            maxPerTile: LocalRenderer.maxPerTile
        )
    }

    public func render(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        input: GaussianInput,
        camera: CameraParams,
        width: Int,
        height: Int
    ) {
        renderView(
            commandBuffer: commandBuffer,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            worldGaussians: input.gaussians,
            harmonics: input.harmonics,
            gaussianCount: input.gaussianCount,
            camera: camera,
            width: width,
            height: height,
            shComponents: input.shComponents,
            useHalfWorld: config.precision == .float16,
            resources: primaryResources
        )
    }

    public func renderStereo(
        commandBuffer: MTLCommandBuffer,
        target: StereoRenderTarget,
        input: GaussianInput,
        camera: StereoCameraParams,
        width: Int,
        height: Int
    ) {
        switch target {
        case .sideBySide:
            fatalError("LocalRenderer does not support stereo rendering. Use HardwareRenderer or DepthFirstRenderer instead.")

        case .foveated:
            fatalError("LocalRenderer does not support stereo rendering. Use HardwareRenderer or DepthFirstRenderer instead.")
        }
    }

    private func renderView(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        worldGaussians: MTLBuffer,
        harmonics: MTLBuffer,
        gaussianCount: Int,
        camera: CameraParams,
        width: Int,
        height: Int,
        shComponents: Int,
        useHalfWorld: Bool,
        resources: LocalViewResources
    ) {
        guard gaussianCount > 0, width > 0, height > 0 else { return }
        guard gaussianCount <= config.maxGaussians else { return }
        guard width <= config.maxWidth, height <= config.maxHeight else { return }

        let tilesX = (width + LocalRenderer.tileWidth - 1) / LocalRenderer.tileWidth
        let tilesY = (height + LocalRenderer.tileHeight - 1) / LocalRenderer.tileHeight
        let tileCount = tilesX * tilesY

        let cameraUniforms = CameraUniformsSwift(
            from: camera,
            width: width,
            height: height,
            gaussianCount: gaussianCount,
            shComponents: shComponents,
            inputIsSRGB: config.gaussianColorSpace == .srgb
        )

        let params = TileBinningParams(
            gaussianCount: UInt32(gaussianCount),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            tileWidth: UInt32(LocalRenderer.tileWidth),
            tileHeight: UInt32(LocalRenderer.tileHeight),
            surfaceWidth: UInt32(width),
            surfaceHeight: UInt32(height),
            maxCapacity: UInt32(gaussianCount),
            alphaThreshold: 0.005,
            totalInkThreshold: 2.0
        )

        clearEncoder.encode(
            commandBuffer: commandBuffer,
            tileCounts: resources.tileCounts,
            header: resources.header,
            tileCount: tileCount,
            maxCompacted: gaussianCount
        )

        projectCullEncoder.encode(
            commandBuffer: commandBuffer,
            worldGaussians: worldGaussians,
            harmonics: harmonics,
            camera: cameraUniforms,
            params: params,
            gaussianCount: gaussianCount,
            tempProjectionBuffer: resources.tempProjection,
            visibilityMarks: resources.visibilityMarks,
            visibilityPartialSums: resources.visibilityPartialSums,
            compactedGaussians: resources.compactedGaussians,
            compactedHeader: resources.header,
            useHalfWorld: useHalfWorld
        )

        scatterEncoder.encode16(
            commandBuffer: commandBuffer,
            compactedGaussians: resources.compactedGaussians,
            compactedHeader: resources.header,
            tileCounters: resources.tileCounts,
            depthKeys16: resources.depthKeys,
            globalIndices: resources.sortIndices,
            tilesX: tilesX,
            maxPerTile: LocalRenderer.maxPerTile,
            tileWidth: LocalRenderer.tileWidth,
            tileHeight: LocalRenderer.tileHeight
        )

        prefixScanEncoder.encode(
            commandBuffer: commandBuffer,
            tileCounts: resources.tileCounts,
            tileOffsets: resources.tileOffsets,
            partialSums: resources.partialSums,
            tileCount: tileCount,
            activeTileIndices: resources.activeTileIndices,
            activeTileCount: resources.activeTileCount
        )

        sortEncoder.encode16(
            commandBuffer: commandBuffer,
            depthKeys16: resources.depthKeys,
            globalIndices: resources.sortIndices,
            sortedLocalIdx: resources.sortedLocalIdx,
            tileCounts: resources.tileCounts,
            maxPerTile: LocalRenderer.maxPerTile,
            tileCount: tileCount
        )

        renderEncoder.encodeClearTextures(
            commandBuffer: commandBuffer,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            width: width,
            height: height
        )

        renderEncoder.encodePrepareRenderDispatch(
            commandBuffer: commandBuffer,
            activeTileCount: resources.activeTileCount,
            dispatchArgs: resources.dispatchArgs
        )

        renderEncoder.encodeIndirect16(
            commandBuffer: commandBuffer,
            projectedGaussians: resources.compactedGaussians,
            tileCounts: resources.tileCounts,
            maxPerTile: LocalRenderer.maxPerTile,
            sortedLocalIdx: resources.sortedLocalIdx,
            globalIndices: resources.sortIndices,
            activeTileIndices: resources.activeTileIndices,
            dispatchArgs: resources.dispatchArgs,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            tilesX: tilesX,
            tilesY: tilesY,
            width: width,
            height: height,
            tileWidth: LocalRenderer.tileWidth,
            tileHeight: LocalRenderer.tileHeight
        )
    }

    // MARK: - Debug Helpers (internal)

    func getVisibleCount() -> UInt32 {
        let ptr = primaryResources.header.contents().bindMemory(to: CompactedHeaderSwift.self, capacity: 1)
        return ptr.pointee.visibleCount
    }

    func hadOverflow() -> Bool {
        let ptr = primaryResources.header.contents().bindMemory(to: CompactedHeaderSwift.self, capacity: 1)
        return ptr.pointee.overflow != 0
    }
}

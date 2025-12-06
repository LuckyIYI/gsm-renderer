import Metal
import simd

/// Fast Gaussian splatting renderer using per-tile local sort
/// Conforms to GaussianRenderer protocol with exactly 2 render methods
public final class LocalRenderer: GaussianRenderer, @unchecked Sendable {
    public let device: MTLDevice
    public let encoder: LocalPipelineEncoder  // Public for benchmarking variant access

    // Tile configuration (16×16 = 256 pixels per tile)
    private let tileWidth = 16
    private let tileHeight = 16

    // Max gaussians per tile (determines sort buffer allocation)
    // 32-bit: 2048 (16KB threadgroup), 16-bit: 4096 (12KB threadgroup)
    private var maxGaussiansPerTile: Int {
        self.config.sortMode == .sort16Bit ? 4096 : 2048
    }

    // Core buffers (always allocated)
    private var compactedBuffer: MTLBuffer?
    private var headerBuffer: MTLBuffer?
    private var tileCountsBuffer: MTLBuffer?
    private var tileOffsetsBuffer: MTLBuffer?
    private var partialSumsBuffer: MTLBuffer?
    private var sortIndicesBuffer: MTLBuffer? // Used by both 32-bit and 16-bit paths
    private var colorTexture: MTLTexture?
    private var depthTexture: MTLTexture?

    // 32-bit sort buffers (only allocated when sortMode == .sort32Bit)
    private var sortKeysBuffer: MTLBuffer?
    private var tempSortKeysBuffer: MTLBuffer?
    private var tempSortIndicesBuffer: MTLBuffer?

    // 16-bit sort buffers (only allocated when sortMode == .sort16Bit)
    private var depthKeys16Buffer: MTLBuffer? // ushort buffer for 16-bit depth keys
    private var sortedLocalIdx16Buffer: MTLBuffer? // ushort buffer for sorted local indices
    // sortInfoBuffer removed - sort reads depthKeys16 directly

    // Temp buffer for efficient single-pass projection
    // Can be aliased with other buffers not used during projection phase
    private var tempProjectionBuffer: MTLBuffer? // [gaussianCount] CompactedGaussian

    // Deterministic compaction buffers
    private var visibilityMarksBuffer: MTLBuffer? // Visibility marks for prefix sum [gaussianCount+1]
    private var visibilityPartialSumsBuffer: MTLBuffer? // Partial sums for hierarchical scan

    // Textured render buffer (only allocated when useTexturedRender == true)
    private var gaussianRenderTexture: MTLTexture?

    // Indirect dispatch buffers
    private var activeTileIndicesBuffer: MTLBuffer?
    private var activeTileCountBuffer: MTLBuffer?
    private var dispatchArgsBuffer: MTLBuffer?

    // Current allocation sizes
    private var allocatedGaussianCapacity: Int = 0
    private var allocatedWidth: Int = 0
    private var allocatedHeight: Int = 0

    // Settings (non-config)
    public var flipY: Bool = false
    public var useSharedBuffers: Bool = false

    // Renderer configuration (immutable - determines buffer allocation)
    private let config: RendererConfig

    public init(device: MTLDevice, config: RendererConfig = RendererConfig()) throws {
        self.device = device
        self.config = config
        self.encoder = try LocalPipelineEncoder(device: device)
    }

    public init(config: RendererConfig = RendererConfig()) throws {
        self.config = config
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw LocalError.failedToCreateQueue
        }
        self.device = device
        self.encoder = try LocalPipelineEncoder(device: device)
    }

    // MARK: - GaussianRenderer Protocol Methods

    public func render(
        toTexture commandBuffer: MTLCommandBuffer,
        input: GaussianInput,
        camera: CameraParams,
        width: Int,
        height: Int,
        whiteBackground: Bool,
        mortonSorted: Bool
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
            useHalfWorld: config.precision == .float16,
            mortonSorted: mortonSorted
        ) else { return nil }

        return TextureRenderResult(color: colorTex, depth: self.depthTexture, alpha: nil)
    }

    public func render(
        toBuffer _: MTLCommandBuffer,
        input _: GaussianInput,
        camera _: CameraParams,
        width _: Int,
        height _: Int,
        whiteBackground _: Bool,
        mortonSorted _: Bool
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
        useHalfWorld: Bool = false,
        mortonSorted _: Bool = false
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

        let tilesX = (width + self.tileWidth - 1) / self.tileWidth
        let tilesY = (height + self.tileHeight - 1) / self.tileHeight
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
            tileWidth: self.tileWidth,
            tileHeight: self.tileHeight,
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
                tileWidth: self.tileWidth,
                tileHeight: self.tileHeight,
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
                tileWidth: self.tileWidth,
                tileHeight: self.tileHeight,
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

    // MARK: - Private

    private func ensureBuffers(gaussianCount: Int, width: Int, height: Int) {
        let tilesX = (width + self.tileWidth - 1) / self.tileWidth
        let tilesY = (height + self.tileHeight - 1) / self.tileHeight
        let tileCount = tilesX * tilesY
        let maxCompacted = gaussianCount
        // Tile-bounded allocation: tileCount × maxPerTile (NOT gaussianCount × 64)
        // This reduces memory from ~1.5GB to ~67MB for sort buffers
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

        // === CORE BUFFERS (always allocated) ===
        self.compactedBuffer = self.device.makeBuffer(
            length: maxCompacted * MemoryLayout<CompactedGaussianSwift>.stride,
            options: self.useSharedBuffers ? shared : priv
        )
        self.headerBuffer = self.device.makeBuffer(
            length: MemoryLayout<CompactedHeaderSwift>.stride,
            options: shared
        )
        self.tileCountsBuffer = self.device.makeBuffer(
            length: tileCount * MemoryLayout<UInt32>.stride,
            options: priv
        )
        self.tileOffsetsBuffer = self.device.makeBuffer(
            length: (tileCount + 1) * MemoryLayout<UInt32>.stride,
            options: priv
        )
        self.partialSumsBuffer = self.device.makeBuffer(
            length: 1024 * MemoryLayout<UInt32>.stride,
            options: priv
        )
        self.sortIndicesBuffer = self.device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: priv
        )

        // === TEMP PROJECTION BUFFER (efficient single-pass projection) ===
        // Stores full CompactedGaussian at original gaussian index during projection
        self.tempProjectionBuffer = self.device.makeBuffer(
            length: maxCompacted * MemoryLayout<CompactedGaussianSwift>.stride,
            options: priv
        )

        // === VISIBILITY PREFIX SUM BUFFERS ===
        // Visibility marks: gaussianCount+1 elements for prefix sum (total at end)
        self.visibilityMarksBuffer = self.device.makeBuffer(
            length: (gaussianCount + 1) * MemoryLayout<UInt32>.stride,
            options: priv
        )
        // Partial sums for hierarchical scan (supports up to 16M gaussians)
        let blockSize = 256
        let visBlocks = (gaussianCount + 1 + blockSize - 1) / blockSize
        let level2Blocks = (visBlocks + blockSize - 1) / blockSize
        let totalPartialSums = (visBlocks + 1) + (level2Blocks + 1)
        self.visibilityPartialSumsBuffer = self.device.makeBuffer(
            length: totalPartialSums * MemoryLayout<UInt32>.stride,
            options: priv
        )

        // === MODE-SPECIFIC BUFFERS ===
        switch self.config.sortMode {
        case .sort32Bit:
            // 32-bit sort buffers
            self.sortKeysBuffer = self.device.makeBuffer(
                length: maxAssignments * MemoryLayout<UInt32>.stride,
                options: priv
            )
            self.tempSortKeysBuffer = self.device.makeBuffer(
                length: maxAssignments * MemoryLayout<UInt32>.stride,
                options: priv
            )
            self.tempSortIndicesBuffer = self.device.makeBuffer(
                length: maxAssignments * MemoryLayout<UInt32>.stride,
                options: priv
            )
            // Clear 16-bit buffers
            self.depthKeys16Buffer = nil
            self.sortedLocalIdx16Buffer = nil

        case .sort16Bit:
            // 16-bit sort buffers only (no sortInfo - sort reads depthKeys16 directly!)
            self.depthKeys16Buffer = self.device.makeBuffer(
                length: maxAssignments * MemoryLayout<UInt16>.stride,
                options: priv
            )
            self.sortedLocalIdx16Buffer = self.device.makeBuffer(
                length: maxAssignments * MemoryLayout<UInt16>.stride,
                options: priv
            )
            // Clear 32-bit buffers
            self.sortKeysBuffer = nil
            self.tempSortKeysBuffer = nil
            self.tempSortIndicesBuffer = nil
        }

        // === TEXTURES ===
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        colorDesc.usage = [.shaderRead, .shaderWrite]
        colorDesc.storageMode = .private
        self.colorTexture = self.device.makeTexture(descriptor: colorDesc)

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        depthDesc.usage = [.shaderRead, .shaderWrite]
        depthDesc.storageMode = .private
        self.depthTexture = self.device.makeTexture(descriptor: depthDesc)

        // Textured render texture (only if enabled)
        if self.config.useTexturedRender {
            let renderTexWidth = LocalPipelineEncoder.renderTexWidth
            let texelCount = maxCompacted * 2
            let renderTexHeight = (texelCount + renderTexWidth - 1) / renderTexWidth
            let gaussianTexDesc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba32Float,
                width: renderTexWidth,
                height: max(1, renderTexHeight),
                mipmapped: false
            )
            gaussianTexDesc.usage = [.shaderRead, .shaderWrite]
            gaussianTexDesc.storageMode = .private
            self.gaussianRenderTexture = self.device.makeTexture(descriptor: gaussianTexDesc)
        } else {
            self.gaussianRenderTexture = nil
        }

        // === INDIRECT DISPATCH BUFFERS ===
        // Active tile indices (max = tileCount)
        self.activeTileIndicesBuffer = self.device.makeBuffer(
            length: tileCount * MemoryLayout<UInt32>.stride,
            options: priv
        )
        // Active tile count (single atomic uint)
        self.activeTileCountBuffer = self.device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: priv
        )
        // Dispatch args (3 uints for MTLDispatchThreadgroupsIndirectArguments)
        self.dispatchArgsBuffer = self.device.makeBuffer(
            length: 3 * MemoryLayout<UInt32>.stride,
            options: priv
        )
    }
}

public enum LocalError: Error {
    case failedToCreateQueue
    case failedToCreateEncoder
}

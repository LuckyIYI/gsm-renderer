import simd
import Metal

/// Fast Gaussian splatting renderer using per-tile local sort
/// Conforms to GaussianRenderer protocol with exactly 2 render methods
public final class LocalSortRenderer: GaussianRenderer, @unchecked Sendable {
    public let device: MTLDevice
    private let queue: MTLCommandQueue
    private let encoder: LocalSortPipelineEncoder
    private var clusterCullEncoder: ClusterCullEncoder?

    // Tile configuration
    private let tileWidth = 32
    private let tileHeight = 16

    // Buffers - lazily allocated (standard pipeline)
    private var compactedBuffer: MTLBuffer?
    private var headerBuffer: MTLBuffer?
    private var tileCountsBuffer: MTLBuffer?
    private var tileOffsetsBuffer: MTLBuffer?
    private var partialSumsBuffer: MTLBuffer?
    private var sortKeysBuffer: MTLBuffer?
    private var sortIndicesBuffer: MTLBuffer?
    private var tempSortKeysBuffer: MTLBuffer?
    private var tempSortIndicesBuffer: MTLBuffer?
    private var colorTexture: MTLTexture?
    private var depthTexture: MTLTexture?
    private var gaussianRenderTexture: MTLTexture?

    // Current allocation sizes
    private var allocatedGaussianCapacity: Int = 0
    private var allocatedWidth: Int = 0
    private var allocatedHeight: Int = 0

    // Settings
    public var flipY: Bool = false
    public var debugPrint: Bool = false
    public var useSharedBuffers: Bool = false
    public var useTexturedRender: Bool = false

    // Renderer configuration
    private let config: RendererConfig

    public init(device: MTLDevice) throws {
        self.device = device
        self.config = RendererConfig()
        guard let queue = device.makeCommandQueue() else {
            throw LocalSortError.failedToCreateQueue
        }
        self.queue = queue
        self.encoder = try LocalSortPipelineEncoder(device: device)
    }

    public init(config: RendererConfig = RendererConfig()) throws {
        self.config = config
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw LocalSortError.failedToCreateQueue
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw LocalSortError.failedToCreateQueue
        }
        self.queue = queue
        self.encoder = try LocalSortPipelineEncoder(device: device)
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

        return TextureRenderResult(color: colorTex, depth: depthTexture, alpha: nil)
    }

    public func render(
        toBuffer commandBuffer: MTLCommandBuffer,
        input: GaussianInput,
        camera: CameraParams,
        width: Int,
        height: Int,
        whiteBackground: Bool,
        mortonSorted: Bool
    ) -> BufferRenderResult? {
        return nil  // LocalSortRenderer only supports texture output
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
        mortonSorted: Bool = false
    ) -> MTLTexture? {
        guard gaussianCount > 0, width > 0, height > 0 else { return nil }

        // Simple skip-based cluster culling (no compaction!)
        var clusterVisibility: MTLBuffer? = nil
        var clusterSize: UInt32 = UInt32(CLUSTER_SIZE)

        if mortonSorted {
            if clusterCullEncoder == nil {
                clusterCullEncoder = try? ClusterCullEncoder(device: device)
            }

            if let cullEncoder = clusterCullEncoder {
                let bytesPerGaussian = worldGaussians.length / gaussianCount
                let inputIsHalf = bytesPerGaussian <= 24

                // Just compute visibility buffer - NO compaction
                clusterVisibility = cullEncoder.encodeCull(
                    commandBuffer: commandBuffer,
                    worldGaussians: worldGaussians,
                    gaussianCount: gaussianCount,
                    viewMatrix: viewMatrix,
                    projectionMatrix: projectionMatrix,
                    useHalfWorld: inputIsHalf
                )
                clusterSize = cullEncoder.clusterSize
            }
        }

        // Continue with rendering using original buffers + visibility mask
        ensureBuffers(gaussianCount: gaussianCount, width: width, height: height)

        guard let compacted = compactedBuffer,
              let header = headerBuffer,
              let tileCounts = tileCountsBuffer,
              let tileOffsets = tileOffsetsBuffer,
              let partialSums = partialSumsBuffer,
              let sortKeys = sortKeysBuffer,
              let sortIndices = sortIndicesBuffer,
              let tempSortKeys = tempSortKeysBuffer,
              let tempSortIndices = tempSortIndicesBuffer,
              let colorTex = colorTexture,
              let depthTex = depthTexture else {
            return nil
        }

        let tilesX = (width + tileWidth - 1) / tileWidth
        let tilesY = (height + tileHeight - 1) / tileHeight
        let maxCompacted = gaussianCount
        let maxAssignments = gaussianCount * 64

        var projMatrix = projectionMatrix
        if flipY {
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

        if debugPrint {
            print("[LocalSortRenderer] Rendering \(gaussianCount) gaussians at \(width)x\(height)")
        }

        // Standard pipeline with optional cluster visibility
        encoder.encode(
            commandBuffer: commandBuffer,
            worldGaussians: worldGaussians,
            harmonics: harmonics,
            camera: camera,
            gaussianCount: gaussianCount,
            tilesX: tilesX,
            tilesY: tilesY,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
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
            useHalfWorld: useHalfWorld,
            skipSort: false,
            tempSortKeys: tempSortKeys,
            tempSortIndices: tempSortIndices,
            clusterVisibility: clusterVisibility,
            clusterSize: clusterSize
        )

        // Encode render
        if useTexturedRender, encoder.hasTexturedRender, let gaussianTex = gaussianRenderTexture {
            encoder.encodePackRenderTexture(
                commandBuffer: commandBuffer,
                compactedGaussians: compacted,
                compactedHeader: header,
                renderTexture: gaussianTex,
                maxGaussians: maxCompacted
            )

            encoder.encodeRenderTextured(
                commandBuffer: commandBuffer,
                gaussianTexture: gaussianTex,
                tileOffsets: tileOffsets,
                tileCounts: tileCounts,
                sortedIndices: sortIndices,
                colorTexture: colorTex,
                depthTexture: depthTex,
                tilesX: tilesX,
                tilesY: tilesY,
                width: width,
                height: height,
                tileWidth: tileWidth,
                tileHeight: tileHeight,
                whiteBackground: whiteBackground
            )
        } else {
            encoder.encodeRender(
                commandBuffer: commandBuffer,
                compactedGaussians: compacted,
                tileOffsets: tileOffsets,
                tileCounts: tileCounts,
                sortedIndices: sortIndices,
                colorTexture: colorTex,
                depthTexture: depthTex,
                tilesX: tilesX,
                tilesY: tilesY,
                width: width,
                height: height,
                tileWidth: tileWidth,
                tileHeight: tileHeight,
                whiteBackground: whiteBackground
            )
        }

        return colorTex
    }

    public func getVisibleCount() -> UInt32 {
        guard let header = headerBuffer else { return 0 }
        return LocalSortPipelineEncoder.readVisibleCount(from: header)
    }

    public func hadOverflow() -> Bool {
        guard let header = headerBuffer else { return false }
        return LocalSortPipelineEncoder.readOverflow(from: header)
    }

    // MARK: - Private

    private func ensureBuffers(gaussianCount: Int, width: Int, height: Int) {
        let tilesX = (width + tileWidth - 1) / tileWidth
        let tilesY = (height + tileHeight - 1) / tileHeight
        let tileCount = tilesX * tilesY
        let maxCompacted = gaussianCount
        let maxAssignments = gaussianCount * 64

        let needsRealloc = gaussianCount > allocatedGaussianCapacity ||
                          width != allocatedWidth ||
                          height != allocatedHeight

        guard needsRealloc else { return }

        allocatedGaussianCapacity = gaussianCount
        allocatedWidth = width
        allocatedHeight = height

        let priv: MTLResourceOptions = useSharedBuffers ? .storageModeShared : .storageModePrivate
        let shared: MTLResourceOptions = .storageModeShared

        compactedBuffer = device.makeBuffer(
            length: maxCompacted * MemoryLayout<CompactedGaussianSwift>.stride,
            options: useSharedBuffers ? shared : priv
        )
        headerBuffer = device.makeBuffer(
            length: MemoryLayout<CompactedHeaderSwift>.stride,
            options: shared
        )
        tileCountsBuffer = device.makeBuffer(
            length: tileCount * MemoryLayout<UInt32>.stride,
            options: priv
        )
        tileOffsetsBuffer = device.makeBuffer(
            length: (tileCount + 1) * MemoryLayout<UInt32>.stride,
            options: priv
        )
        partialSumsBuffer = device.makeBuffer(
            length: 1024 * MemoryLayout<UInt32>.stride,
            options: priv
        )
        sortKeysBuffer = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: priv
        )
        sortIndicesBuffer = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: priv
        )
        tempSortKeysBuffer = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: priv
        )
        tempSortIndicesBuffer = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: priv
        )

        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        colorDesc.usage = [.shaderRead, .shaderWrite]
        colorDesc.storageMode = .private
        colorTexture = device.makeTexture(descriptor: colorDesc)

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        depthDesc.usage = [.shaderRead, .shaderWrite]
        depthDesc.storageMode = .private
        depthTexture = device.makeTexture(descriptor: depthDesc)

        let renderTexWidth = LocalSortPipelineEncoder.renderTexWidth
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
        gaussianRenderTexture = device.makeTexture(descriptor: gaussianTexDesc)

        if debugPrint {
            print("[LocalSortRenderer] Allocated buffers for \(gaussianCount) gaussians, \(width)x\(height)")
        }
    }
}

public enum LocalSortError: Error {
    case failedToCreateQueue
    case failedToCreateEncoder
}

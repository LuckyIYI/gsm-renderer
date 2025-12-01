import simd
import Metal

/// Fast Gaussian splatting renderer using per-tile local sort
/// Conforms to GaussianRenderer protocol with exactly 2 render methods
public final class LocalSortRenderer: GaussianRenderer, @unchecked Sendable {
    public let device: MTLDevice
    private let queue: MTLCommandQueue
    private let encoder: LocalSortPipelineEncoder

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
    private var gaussianRenderTexture: MTLTexture?  // Texture cache for gaussian data

    // Current allocation sizes
    private var allocatedGaussianCapacity: Int = 0
    private var allocatedWidth: Int = 0
    private var allocatedHeight: Int = 0

    // Settings
    public var flipY: Bool = false  // Set to true if your projection has Y inverted
    public var debugPrint: Bool = false
    public var useSharedBuffers: Bool = false  // Set to true for debugging (slower but CPU-readable)
    public var useTexturedRender: Bool = false  // Use texture cache for render (TLB optimization)

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

    /// Initialize with configuration
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

    /// Render to GPU textures (protocol method)
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
            useHalfWorld: config.precision == .float16
        ) else { return nil }

        return TextureRenderResult(color: colorTex, depth: depthTexture, alpha: nil)
    }

    /// Render to CPU-readable buffers (protocol method)
    /// Note: LocalSortRenderer only supports texture output, not buffer output
    public func render(
        toBuffer commandBuffer: MTLCommandBuffer,
        input: GaussianInput,
        camera: CameraParams,
        width: Int,
        height: Int,
        whiteBackground: Bool
    ) -> BufferRenderResult? {
        // LocalSortRenderer does not support buffer output
        // Use GlobalSortRenderer for CPU-readable buffer output
        return nil
    }

    // MARK: - Direct Render Method

    /// Render with explicit parameters (for direct testing)
    public func render(
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
        return renderInternal(
            commandBuffer: commandBuffer,
            worldGaussians: worldGaussians,
            harmonics: harmonics,
            gaussianCount: gaussianCount,
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            cameraPosition: cameraPosition,
            focalX: focalX,
            focalY: focalY,
            width: width,
            height: height,
            shComponents: shComponents,
            whiteBackground: whiteBackground,
            useHalfWorld: useHalfWorld
        )
    }

    // MARK: - Internal Render Implementation

    /// Internal render implementation
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

        // Ensure buffers are allocated
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
        // Increase multiplier to prevent overflow with large/dense gaussians
        let maxAssignments = gaussianCount * 64

        // Optionally flip Y in projection matrix
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
            print("  flipY: \(flipY)")
            print("  focalX: \(focalX), focalY: \(focalY)")
        }

        // Standard pipeline
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
                tempSortIndices: tempSortIndices
            )

            // Encode render - use textured path if available for TLB optimization
            if useTexturedRender, encoder.hasTexturedRender, let gaussianTex = gaussianRenderTexture {
                // Pack gaussian data into texture
                encoder.encodePackRenderTexture(
                    commandBuffer: commandBuffer,
                    compactedGaussians: compacted,
                    compactedHeader: header,
                    renderTexture: gaussianTex,
                    maxGaussians: maxCompacted
                )

                // Render using texture cache
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
                // Standard buffer-based render
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

    /// Get visible count after render completes (call after waitUntilCompleted)
    public func getVisibleCount() -> UInt32 {
        guard let header = headerBuffer else { return 0 }
        return LocalSortPipelineEncoder.readVisibleCount(from: header)
    }

    /// Debug helper: print first few harmonics values and compacted colors
    public func debugPrintData(harmonics: MTLBuffer, harmonicsCount: Int) {
        print("\n[LocalSortRenderer DEBUG]")

        // Print first 3 harmonics (9 floats = 3 RGB values)
        let harmonicsPtr = harmonics.contents().bindMemory(to: Float.self, capacity: min(9, harmonicsCount))
        print("  First 3 harmonics (RGB values before +0.5):")
        for i in 0..<min(3, harmonicsCount / 3) {
            let r = harmonicsPtr[i * 3 + 0]
            let g = harmonicsPtr[i * 3 + 1]
            let b = harmonicsPtr[i * 3 + 2]
            print("    [\(i)] R=\(String(format: "%.4f", r)) G=\(String(format: "%.4f", g)) B=\(String(format: "%.4f", b))")
        }

        // Print compacted colors if available
        if let compacted = compactedBuffer, let header = headerBuffer {
            let visibleCount = LocalSortPipelineEncoder.readVisibleCount(from: header)
            print("  Visible count: \(visibleCount)")

            if visibleCount > 0 && compacted.storageMode == .shared {
                let compactedPtr = compacted.contents().bindMemory(to: CompactedGaussianSwift.self, capacity: Int(visibleCount))
                print("  First 3 compacted gaussian colors (after +0.5):")
                for i in 0..<min(3, Int(visibleCount)) {
                    let g = compactedPtr[i]
                    // Unpack half4 from position_color.zw
                    let packed = SIMD2<Float>(g.position_color.z, g.position_color.w)
                    let u0 = packed.x.bitPattern
                    let u1 = packed.y.bitPattern
                    let h0 = Float16(bitPattern: UInt16(u0 & 0xFFFF))
                    let h1 = Float16(bitPattern: UInt16((u0 >> 16) & 0xFFFF))
                    let h2 = Float16(bitPattern: UInt16(u1 & 0xFFFF))
                    let h3 = Float16(bitPattern: UInt16((u1 >> 16) & 0xFFFF))
                    print("    [\(i)] R=\(String(format: "%.4f", Float(h0))) G=\(String(format: "%.4f", Float(h1))) B=\(String(format: "%.4f", Float(h2))) A=\(String(format: "%.4f", Float(h3)))")
                }
            } else if compacted.storageMode == .private {
                print("  (compacted buffer is private - can't read directly)")
            }
        }
        print("")
    }

    /// Check if overflow occurred
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
        // Increase multiplier to prevent overflow with large/dense gaussians
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
            options: useSharedBuffers ? shared : priv  // Use shared for debug
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

        // Color texture
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        colorDesc.usage = [.shaderRead, .shaderWrite]
        colorDesc.storageMode = .private
        colorTexture = device.makeTexture(descriptor: colorDesc)

        // Depth texture
        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        depthDesc.usage = [.shaderRead, .shaderWrite]
        depthDesc.storageMode = .private
        depthTexture = device.makeTexture(descriptor: depthDesc)

        // Gaussian render texture for texture-cached rendering
        // Each gaussian uses 2 texels (covariance_depth + position_color)
        // Layout: width = RENDER_TEX_WIDTH (4096), height = ceil(gaussianCount * 2 / 4096)
        let renderTexWidth = LocalSortPipelineEncoder.renderTexWidth
        let texelCount = maxCompacted * 2
        let renderTexHeight = (texelCount + renderTexWidth - 1) / renderTexWidth
        let gaussianTexDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,  // float4 per texel
            width: renderTexWidth,
            height: max(1, renderTexHeight),
            mipmapped: false
        )
        gaussianTexDesc.usage = [.shaderRead, .shaderWrite]
        gaussianTexDesc.storageMode = .private
        gaussianRenderTexture = device.makeTexture(descriptor: gaussianTexDesc)

        if debugPrint {
            print("[LocalSortRenderer] Allocated buffers for \(gaussianCount) gaussians, \(width)x\(height)")
            print("  Gaussian render texture: \(renderTexWidth)x\(renderTexHeight)")
        }
    }
}

public enum LocalSortError: Error {
    case failedToCreateQueue
    case failedToCreateEncoder
}

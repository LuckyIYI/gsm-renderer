import simd
import Metal

/// Standalone Tellusim-style renderer backend
/// Drop-in replacement that uses the optimized Tellusim pipeline
public final class TellusimBackend {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let encoder: TellusimPipelineEncoder

    // Tile configuration
    private let tileWidth = 32
    private let tileHeight = 16

    // Buffers - lazily allocated
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

    // Current allocation sizes
    private var allocatedGaussianCapacity: Int = 0
    private var allocatedWidth: Int = 0
    private var allocatedHeight: Int = 0

    // Settings
    public var flipY: Bool = false  // Set to true if your projection has Y inverted
    public var debugPrint: Bool = false
    public var useSharedBuffers: Bool = false  // Set to true for debugging (slower but CPU-readable)

    public init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw TellusimError.failedToCreateQueue
        }
        self.queue = queue
        self.encoder = try TellusimPipelineEncoder(device: device)
    }

    /// Render gaussians to texture
    /// - Returns: Color texture with rendered output, or nil on failure
    /// - Parameter useHalfWorld: If true, expects worldGaussians in half16 format (24 bytes/gaussian)
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
            print("[TellusimBackend] Rendering \(gaussianCount) gaussians at \(width)x\(height)")
            print("  flipY: \(flipY)")
            print("  focalX: \(focalX), focalY: \(focalY)")
        }

        // Encode pipeline
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

        // Encode render
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

        return colorTex
    }

    /// Get visible count after render completes (call after waitUntilCompleted)
    public func getVisibleCount() -> UInt32 {
        guard let header = headerBuffer else { return 0 }
        return TellusimPipelineEncoder.readVisibleCount(from: header)
    }

    /// Debug helper: print first few harmonics values and compacted colors
    public func debugPrintData(harmonics: MTLBuffer, harmonicsCount: Int) {
        print("\n[TellusimBackend DEBUG]")

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
            let visibleCount = TellusimPipelineEncoder.readVisibleCount(from: header)
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
        return TellusimPipelineEncoder.readOverflow(from: header)
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

        if debugPrint {
            print("[TellusimBackend] Allocated buffers for \(gaussianCount) gaussians, \(width)x\(height)")
        }
    }
}

public enum TellusimError: Error {
    case failedToCreateQueue
    case failedToCreateEncoder
}

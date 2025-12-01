import Foundation
import simd
@preconcurrency import Metal

public struct RendererLimits: Sendable {
    public let maxGaussians: Int
    public let maxWidth: Int
    public let maxHeight: Int
    public let tileWidth: Int
    public let tileHeight: Int
    public let maxPerTile: Int

    public var tilesX: Int { (maxWidth + tileWidth - 1) / tileWidth }
    public var tilesY: Int { (maxHeight + tileHeight - 1) / tileHeight }
    public var maxTileCount: Int { max(1, tilesX * tilesY) }

    /// Initialize with viewport size - tile dimensions computed automatically
    /// - Parameters:
    ///   - maxGaussians: Maximum number of gaussians to render
    ///   - viewportSize: Viewport dimensions (width, height)
    ///   - tileSize: Optional custom tile size (default: 32x16 for multi-pixel rendering)
    ///   - maxPerTile: Optional max gaussians per tile (default: 1024)
    public init(maxGaussians: Int, viewportSize: (width: Int, height: Int), tileSize: (width: Int, height: Int)? = nil, maxPerTile: Int? = nil) {
        self.maxGaussians = max(1, maxGaussians)
        self.maxWidth = max(1, viewportSize.width)
        self.maxHeight = max(1, viewportSize.height)
        // Default tile size: 32x16 (optimized for multi-pixel rendering)
        self.tileWidth = max(1, tileSize?.width ?? 32)
        self.tileHeight = max(1, tileSize?.height ?? 16)
        self.maxPerTile = max(1, maxPerTile ?? 1024)
    }

    /// Legacy initializer for backwards compatibility
    public init(maxGaussians: Int, maxWidth: Int, maxHeight: Int, tileWidth: Int = 32, tileHeight: Int = 16, maxPerTile: Int? = nil) {
        self.maxGaussians = max(1, maxGaussians)
        self.maxWidth = max(1, maxWidth)
        self.maxHeight = max(1, maxHeight)
        self.tileWidth = max(1, tileWidth)
        self.tileHeight = max(1, tileHeight)
        self.maxPerTile = max(1, maxPerTile ?? 2048)
    }

    /// Build RenderParams from FrameParams (runtime settings only)
    func buildParams(from frame: FrameParams) -> RenderParams {
        RenderParams(
            width: UInt32(maxWidth),
            height: UInt32(maxHeight),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            maxPerTile: UInt32(maxPerTile),
            whiteBackground: frame.whiteBackground ? 1 : 0,
            activeTileCount: 0,
            gaussianCount: UInt32(frame.gaussianCount)
        )
    }

    /// Build RenderParams from individual values (internal helper)
    func buildParams(gaussianCount: Int, whiteBackground: Bool) -> RenderParams {
        buildParams(from: FrameParams(gaussianCount: gaussianCount, whiteBackground: whiteBackground))
    }
}

// MARK: - Frame Resources

/// Placement heap allocator with optional residency management
final class HeapAllocator {
    private let device: MTLDevice
    var residencySetProvider: (() -> (any MTLResidencySet)?)?
    private(set) var heap: MTLHeap?
    private(set) var heapSize: Int = 0
    private var currentOffset: Int = 0

    init(device: MTLDevice, residencySetProvider: (() -> (any MTLResidencySet)?)? = nil) {
        self.device = device
        self.residencySetProvider = residencySetProvider
    }

    func createHeap(size: Int, label: String) -> Bool {
        guard size > 0 else { return false }
        let descriptor = MTLHeapDescriptor()
        descriptor.type = .placement
        descriptor.storageMode = .private
        descriptor.hazardTrackingMode = .tracked
        descriptor.size = size

        guard let newHeap = device.makeHeap(descriptor: descriptor) else { return false }
        newHeap.label = label
        heap = newHeap
        heapSize = size
        currentOffset = 0

        if let resSet = residencySetProvider?() {
            resSet.addAllocation(newHeap)
            resSet.commit()
        }
        return true
    }

    func allocateBuffer(length: Int, options: MTLResourceOptions, label: String?) -> MTLBuffer? {
        guard let heap else { return nil }
        let sa = device.heapBufferSizeAndAlign(length: length, options: options)
        let align = max(Int(sa.align), 1)
        currentOffset = (currentOffset + align - 1) & ~(align - 1)
        guard currentOffset + Int(sa.size) <= heapSize else { return nil }
        guard let buffer = heap.makeBuffer(length: length, options: options, offset: currentOffset) else { return nil }
        buffer.label = label
        currentOffset += Int(sa.size)
        return buffer
    }
}

struct FrameResourceLayout {
    let limits: RendererLimits
    /// Maximum tile assignments capacity (tileCount * maxPerTile, clamped by gaussianCapacity)
    let maxAssignmentCapacity: Int
    /// Padded capacity for sort alignment (power-of-2 for bitonic, 1024-aligned for radix)
    let paddedCapacity: Int
    let heapAllocations: [(label: String, length: Int, options: MTLResourceOptions)]
    let sharedAllocations: [(label: String, length: Int, options: MTLResourceOptions)]
    let heapSize: Int
    let pixelCapacity: Int
    let precisionStride: Int
}

final class FrameResources {
    // Placement heap allocator for efficient buffer allocation
    let heapAllocator: HeapAllocator

    // Tile Builder Buffers
    let boundsBuffer: MTLBuffer
    let coverageBuffer: MTLBuffer
    let offsetsBuffer: MTLBuffer
    let partialSumsBuffer: MTLBuffer
    let scatterDispatchBuffer: MTLBuffer
    let tileAssignmentHeader: MTLBuffer
    let tileIndices: MTLBuffer
    let tileIds: MTLBuffer
    var tileAssignmentMaxAssignments: Int
    var tileAssignmentPaddedCapacity: Int

    // Ordered Buffers
    let orderedHeaders: MTLBuffer
    let packedMeans: MTLBuffer
    let packedConics: MTLBuffer
    let packedColors: MTLBuffer
    let packedOpacities: MTLBuffer
    let packedDepths: MTLBuffer
    let activeTileIndices: MTLBuffer
    let activeTileCount: MTLBuffer

    // Sort Buffers (used by both bitonic and radix)
    let sortKeys: MTLBuffer
    let sortedIndices: MTLBuffer

    // Radix Sort Buffers (nil when using bitonic)
    let radixHistogram: MTLBuffer?
    let radixBlockSums: MTLBuffer?
    let radixScannedHistogram: MTLBuffer?
    let radixFusedKeys: MTLBuffer?
    let radixKeysScratch: MTLBuffer?
    let radixPayloadScratch: MTLBuffer?

    // Fused Pipeline Buffers (interleaved data for cache efficiency)
    let interleavedGaussians: MTLBuffer?
    let packedGaussiansFused: MTLBuffer?

    // Dispatch Args (Per-frame to avoid race on indirect dispatch)
    let dispatchArgs: MTLBuffer

    // Output Buffers (nil when textureOnly mode)
    var outputBuffers: RenderOutputBuffers?
    // Output Textures (Alternative)
    var outputTextures: RenderOutputTextures

    let device: MTLDevice

    init(device: MTLDevice, layout: FrameResourceLayout, residencySetProvider: (() -> (any MTLResidencySet)?)?, useHeap: Bool, textureOnly: Bool = false) {
        self.device = device
        self.heapAllocator = HeapAllocator(device: device, residencySetProvider: residencySetProvider)
        self.tileAssignmentMaxAssignments = layout.maxAssignmentCapacity
        self.tileAssignmentPaddedCapacity = layout.paddedCapacity

        guard let dispatchArgs = device.makeBuffer(length: 4096, options: .storageModePrivate) else {
            fatalError("Failed to allocate dispatch args")
        }
        dispatchArgs.label = "FrameDispatchArgs"
        self.dispatchArgs = dispatchArgs

        func makeDeviceBuffer(_ req: (label: String, length: Int, options: MTLResourceOptions)) -> MTLBuffer {
            guard let buf = device.makeBuffer(length: req.length, options: req.options) else {
                fatalError("Failed to allocate buffer \(req.label)")
            }
            buf.label = req.label
            return buf
        }

        var heapBuffers: [String: MTLBuffer] = [:]
        let heapActive = useHeap
        if heapActive {
            guard heapAllocator.createHeap(size: layout.heapSize, label: "FrameResourcesHeap") else {
                fatalError("Failed to create placement heap of size \(layout.heapSize)")
            }
            for req in layout.heapAllocations {
                guard let buf = heapAllocator.allocateBuffer(length: req.length, options: req.options, label: req.label) else {
                    fatalError("Failed to allocate \(req.label) from heap")
                }
                heapBuffers[req.label] = buf
            }
        }

        func buffer(_ label: String) -> MTLBuffer {
            if let buf = heapBuffers[label] { return buf }
            if let match = layout.sharedAllocations.first(where: { $0.label == label }) {
                return makeDeviceBuffer(match)
            }
            if let match = layout.heapAllocations.first(where: { $0.label == label }) {
                if heapActive {
                    fatalError("Heap allocation for \(label) missing; heap failed or overflowed")
                }
                return makeDeviceBuffer(match)
            }
            fatalError("Unknown buffer label \(label)")
        }

        // Assign buffers
        self.boundsBuffer = buffer("Bounds")
        self.coverageBuffer = buffer("Coverage")
        self.offsetsBuffer = buffer("Offsets")
        self.partialSumsBuffer = buffer("PartialSums")
        self.scatterDispatchBuffer = buffer("ScatterDispatch")
        self.tileAssignmentHeader = buffer("TileHeader")

        // Initialize TileHeader constants ONCE at setup (not during render)
        // TileHeader is .storageModeShared so this CPU write at init is allowed
        let headerPtr = self.tileAssignmentHeader.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.maxAssignments = UInt32(layout.maxAssignmentCapacity)
        headerPtr.pointee.paddedCount = UInt32(layout.paddedCapacity)
        headerPtr.pointee.totalAssignments = 0
        headerPtr.pointee.overflow = 0

        self.tileIndices = buffer("TileIndices")
        self.tileIds = buffer("TileIds")
        self.orderedHeaders = buffer("OrderedHeaders")
        self.packedMeans = buffer("PackedMeans")
        self.packedConics = buffer("PackedConics")
        self.packedColors = buffer("PackedColors")
        self.packedOpacities = buffer("PackedOpacities")
        self.packedDepths = buffer("PackedDepths")
        self.activeTileIndices = buffer("ActiveTileIndices")
        self.activeTileCount = buffer("ActiveTileCount")
        self.sortKeys = buffer("SortKeys")
        self.sortedIndices = buffer("SortedIndices")

        // Conditionally allocate radix/fused buffers based on config
        func optionalBuffer(_ label: String) -> MTLBuffer? {
            layout.heapAllocations.contains(where: { $0.label == label }) ? buffer(label) : nil
        }
        self.radixHistogram = optionalBuffer("RadixHist")
        self.radixBlockSums = optionalBuffer("RadixBlockSums")
        self.radixScannedHistogram = optionalBuffer("RadixScanned")
        self.radixFusedKeys = optionalBuffer("RadixFused")
        self.radixKeysScratch = optionalBuffer("RadixScratch")
        self.radixPayloadScratch = optionalBuffer("RadixPayload")
        self.interleavedGaussians = optionalBuffer("InterleavedGaussians")
        self.packedGaussiansFused = optionalBuffer("PackedGaussiansFused")

        // Output buffers (skipped in textureOnly mode to save ~20MB)
        let pixelBytes = layout.pixelCapacity
        if textureOnly {
            self.outputBuffers = nil
        } else {
            guard
                let color = device.makeBuffer(length: pixelBytes * 12, options: .storageModeShared),
                let depth = device.makeBuffer(length: pixelBytes * 4, options: .storageModeShared),
                let alpha = device.makeBuffer(length: pixelBytes * 4, options: .storageModeShared)
            else { fatalError("Failed to allocate output buffers") }
            color.label = "RenderColorOutput"
            depth.label = "RenderDepthOutput"
            alpha.label = "RenderAlphaOutput"
            self.outputBuffers = RenderOutputBuffers(colorOutGPU: color, depthOutGPU: depth, alphaOutGPU: alpha)
        }

        // Output textures sized to limits
        let texDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: layout.limits.maxWidth, height: layout.limits.maxHeight, mipmapped: false)
        texDesc.usage = [.shaderWrite, .shaderRead]
        texDesc.storageMode = .private
        guard let colorTex = device.makeTexture(descriptor: texDesc) else { fatalError("Failed to allocate color texture") }
        colorTex.label = "OutputColorTex"

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: layout.limits.maxWidth, height: layout.limits.maxHeight, mipmapped: false)
        depthDesc.usage = [.shaderWrite, .shaderRead]
        depthDesc.storageMode = .private
        guard let depthTex = device.makeTexture(descriptor: depthDesc),
              let alphaTex = device.makeTexture(descriptor: depthDesc) else { fatalError("Failed to allocate depth/alpha textures") }
        depthTex.label = "OutputDepthTex"
        alphaTex.label = "OutputAlphaTex"
        self.outputTextures = RenderOutputTextures(color: colorTex, depth: depthTex, alpha: alphaTex)
    }
}

// MARK: - Global Sort Renderer

/// High-quality Gaussian splatting renderer using global radix sort
/// Conforms to GaussianRenderer protocol with exactly 2 render methods
public final class GlobalSortRenderer: GaussianRenderer, @unchecked Sendable {
    private static let supportedMaxGaussians = 10_000_000

    public let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    private var nextFrameCapturePath: String?
    
    // Encoders
    let tileBoundsEncoder: TileBoundsEncoder
    let sortKeyGenEncoder: SortKeyGenEncoder
    let bitonicSortEncoder: BitonicSortEncoder
    let radixSortEncoder: RadixSortEncoder
    let packEncoder: PackEncoder
    let renderEncoder: RenderEncoder
    let projectEncoder: ProjectEncoder
    let dispatchEncoder: DispatchEncoder

    // Fused pipeline encoder (for interleaved data optimization)
    let fusedPipelineEncoder: FusedPipelineEncoder

    // SIMD-optimized fused coverage+scatter V2 encoder
    let fusedCoverageScatterEncoderV2: FusedCoverageScatterEncoderV2

    // Reset tile builder state pipeline (replaces blit for lower overhead)
    private let resetTileBuilderStatePipeline: MTLComputePipelineState
    
    // Frame Resources (Single Buffering for now)
    private var frameResources: [FrameResources] = []
    private var frameInUse: [Bool] = []
    private var frameCursor: Int = 0
    private let maxInFlightFrames = 1 // Enforce single buffering initially
    private let frameLock = NSLock()
    
    // World Buffers (Input - Shared/Transient)
    private var worldPositionsCache: MTLBuffer?
    private var worldScalesCache: MTLBuffer?
    private var worldRotationsCache: MTLBuffer?
    private var worldHarmonicsCache: MTLBuffer?
    private var worldOpacitiesCache: MTLBuffer?
    
    // Gaussian input (projection stage) buffers (Shared/Transient)
    private var gaussianMeansCache: MTLBuffer?
    private var gaussianRadiiCache: MTLBuffer?
    private var gaussianMaskCache: MTLBuffer?
    private var gaussianDepthsCache: MTLBuffer?
    private var gaussianConicsCache: MTLBuffer?
    private var gaussianColorsCache: MTLBuffer?
    private var gaussianOpacitiesCache: MTLBuffer?

    // Half-precision gaussian buffers cache (for half16 pipeline)
    private var gaussianMeansHalfCache: MTLBuffer?
    private var gaussianDepthsHalfCache: MTLBuffer?
    private var gaussianConicsHalfCache: MTLBuffer?
    private var gaussianColorsHalfCache: MTLBuffer?
    private var gaussianOpacitiesHalfCache: MTLBuffer?

    private let sortAlgorithm: SortAlgorithm
    private let precision: Precision
    private let effectivePrecision: Precision
    public var precisionSetting: Precision { self.effectivePrecision }

    /// Last GPU execution time in seconds (from command buffer GPUStartTime/GPUEndTime)
    public private(set) var lastGPUTime: Double?

    /// Residency set for efficient GPU memory management (macOS 15+ / iOS 18+)
    private var residencySet: (any MTLResidencySet)?

    /// Use heap allocation for frame buffers (reduces TLB misses)
    public let useHeapAllocation: Bool
    private let limits: RendererLimits
    let frameLayout: FrameResourceLayout  // internal for tests

    /// Recommended tile size (32x16 for multi-pixel rendering)
    public var recommendedTileWidth: UInt32 { 32 }
    public var recommendedTileHeight: UInt32 { 16 }

    public init(
        precision: Precision = .float32,
        sortAlgorithm: SortAlgorithm = .radix,
        useHeapAllocation: Bool = false,
        textureOnly: Bool = false,  // Skip buffer output allocation to save ~20MB
        limits: RendererLimits = RendererLimits(maxGaussians: 1_000_000, maxWidth: 2048, maxHeight: 2048)
    ) {
        precondition(limits.maxGaussians <= GlobalSortRenderer.supportedMaxGaussians, "GlobalSortRenderer supports up to 10,000,000 gaussians; requested \(limits.maxGaussians)")
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device unavailable")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue")
        }
        self.queue = queue

        do {
            let currentFileURL = URL(fileURLWithPath: #filePath)
            let moduleDir = currentFileURL.deletingLastPathComponent()
            let metallibURL = moduleDir.appendingPathComponent("GaussianMetalRenderer.metallib")
            let library = try device.makeLibrary(URL: metallibURL)

            self.library = library
            self.effectivePrecision = precision

            self.tileBoundsEncoder = try TileBoundsEncoder(device: device, library: library)
            self.sortKeyGenEncoder = try SortKeyGenEncoder(device: device, library: library)
            self.bitonicSortEncoder = try BitonicSortEncoder(device: device, library: library, useIndirect: false)
            self.radixSortEncoder = try RadixSortEncoder(device: device, library: library)
            self.packEncoder = try PackEncoder(device: device, library: library)
            self.renderEncoder = try RenderEncoder(device: device, library: library)
            self.projectEncoder = try ProjectEncoder(device: device, library: library)

            let config = AssignmentDispatchConfigSwift(
                sortThreadgroupSize: UInt32(self.sortKeyGenEncoder.threadgroupSize),
                fuseThreadgroupSize: UInt32(self.radixSortEncoder.fuseThreadgroupSize),
                unpackThreadgroupSize: UInt32(self.radixSortEncoder.unpackThreadgroupSize),
                packThreadgroupSize: UInt32(self.packEncoder.packThreadgroupSize),
                bitonicThreadgroupSize: UInt32(self.bitonicSortEncoder.unitSize),
                radixBlockSize: UInt32(self.radixSortEncoder.blockSize),
                radixGrainSize: UInt32(self.radixSortEncoder.grainSize),
                maxAssignments: 0  // Set dynamically at encode time
            )
            self.dispatchEncoder = try DispatchEncoder(device: device, library: library, config: config)

            // Fused pipeline encoder (always enabled - better performance)
            self.fusedPipelineEncoder = try FusedPipelineEncoder(device: device, library: library)

            // SIMD-optimized V2 encoder for coverage+scatter
            self.fusedCoverageScatterEncoderV2 = try FusedCoverageScatterEncoderV2(device: device, library: library)

            // Reset tile builder state pipeline (replaces blit for lower overhead)
            guard let resetFn = library.makeFunction(name: "resetTileBuilderStateKernel") else {
                fatalError("resetTileBuilderStateKernel not found in library")
            }
            self.resetTileBuilderStatePipeline = try device.makeComputePipelineState(function: resetFn)

        } catch {
            fatalError("Failed to load library or initialize encoders: \(error)")
        }
        self.precision = precision
        self.sortAlgorithm = sortAlgorithm
        self.useHeapAllocation = useHeapAllocation
        self.limits = limits

        do {
            let descriptor = MTLResidencySetDescriptor()
            descriptor.label = "GaussianRendererResidencySet"
            descriptor.initialCapacity = 64
            let resSet = try device.makeResidencySet(descriptor: descriptor)
            self.residencySet = resSet
            queue.addResidencySet(resSet)
        } catch {
            self.residencySet = nil
        }

        // Compute layout once using max precision for safety
        self.frameLayout = GlobalSortRenderer.computeLayout(
            limits: limits,
            sortAlgorithm: sortAlgorithm,
            useFusedPipeline: true,  // Always enabled
            precision: Precision.float32,
            device: device,
            radixSortEncoder: self.radixSortEncoder
        )

        // Initialize frame slots with fully allocated resources
        for _ in 0..<maxInFlightFrames {
            let frame = FrameResources(
                device: device,
                layout: self.frameLayout,
                residencySetProvider: { [weak self] in self?.residencySet },
                useHeap: useHeapAllocation,
                textureOnly: textureOnly
            )
            self.frameResources.append(frame)
            self.frameInUse.append(false)
        }

        // Preallocate shared CPU-visible caches based on precision mode
        // float32 precision: allocate float32 gaussian caches
        // float16 precision: allocate half gaussian caches
        // World caches only needed for non-textureOnly mode (separate world->gaussian projection)
        let shared: MTLResourceOptions = .storageModeShared
        func makeShared(_ length: Int, _ label: String) -> MTLBuffer {
            guard let buf = device.makeBuffer(length: length, options: shared) else {
                fatalError("Failed to allocate shared buffer \(label)")
            }
            buf.label = label
            return buf
        }

        let g = limits.maxGaussians

        // World caches only needed for world->gaussian projection (not textureOnly mode)
        if !textureOnly {
            self.worldPositionsCache = makeShared(g * MemoryLayout<SIMD3<Float>>.stride, "WorldPositions")
            self.worldScalesCache = makeShared(g * MemoryLayout<SIMD3<Float>>.stride, "WorldScales")
            self.worldRotationsCache = makeShared(g * MemoryLayout<SIMD4<Float>>.stride, "WorldRotations")
            // SH coeffs worst-case: 9 coefficients (l=2) *3 channels -> 27 floats = 108MB!
            self.worldHarmonicsCache = makeShared(g * MemoryLayout<Float>.stride * 27, "WorldHarmonics")
            self.worldOpacitiesCache = makeShared(g * MemoryLayout<Float>.stride, "WorldOpacities")
        }

        // Gaussian input caches - allocate based on precision mode
        // encodeRenderToTextures uses float32 caches, encodeRenderToTextureHalf uses half caches
        if precision == .float32 {
            // Float32 gaussian caches for encodeRenderToTextures
            self.gaussianMeansCache = makeShared(g * 8, "GaussianMeans")
            self.gaussianDepthsCache = makeShared(g * 4, "GaussianDepths")
            self.gaussianConicsCache = makeShared(g * 16, "GaussianConics")
            self.gaussianColorsCache = makeShared(g * 12, "GaussianColors")
            self.gaussianOpacitiesCache = makeShared(g * 4, "GaussianOpacities")
        } else {
            // Half precision caches for encodeRenderToTextureHalf
            self.gaussianMeansHalfCache = makeShared(g * 4, "GaussianMeansHalf")
            self.gaussianDepthsHalfCache = makeShared(g * 2, "GaussianDepthsHalf")
            self.gaussianConicsHalfCache = makeShared(g * 8, "GaussianConicsHalf")
            self.gaussianColorsHalfCache = makeShared(g * 6, "GaussianColorsHalf")
            self.gaussianOpacitiesHalfCache = makeShared(g * 2, "GaussianOpacitiesHalf")
        }

        // Radii and Mask always float (used for tile bounds in all modes)
        self.gaussianRadiiCache = makeShared(g * 4, "GaussianRadii")
        self.gaussianMaskCache = makeShared(g, "GaussianMask")
    }

    /// Convenience initializer from RendererConfig
    public convenience init(config: RendererConfig) {
        let limits = RendererLimits(
            maxGaussians: config.maxGaussians,
            maxWidth: config.maxWidth,
            maxHeight: config.maxHeight
        )
        let precision: Precision = config.precision == .float16 ? .float16 : .float32
        self.init(
            precision: precision,
            sortAlgorithm: .radix,
            useHeapAllocation: config.useHeapAllocation,
            textureOnly: false,
            limits: limits
        )
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
        let cameraUniforms = CameraUniformsSwift(
            viewMatrix: camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            cameraCenter: camera.position,
            pixelFactor: 1.0,
            focalX: camera.focalX,
            focalY: camera.focalY,
            width: Float(width),
            height: Float(height),
            nearPlane: camera.near,
            farPlane: camera.far,
            shComponents: UInt32(input.shComponents),
            gaussianCount: UInt32(input.gaussianCount)
        )

        let frameParams = FrameParams(
            gaussianCount: input.gaussianCount,
            whiteBackground: whiteBackground
        )

        let output: RenderOutputTextures?

        if effectivePrecision == .float16 {
            let packedWorld = PackedWorldBuffersHalf(
                packedGaussians: input.gaussians,
                harmonics: input.harmonics
            )
            output = encodeRenderToTextureHalf(
                commandBuffer: commandBuffer,
                gaussianCount: input.gaussianCount,
                packedWorldBuffersHalf: packedWorld,
                cameraUniforms: cameraUniforms,
                frameParams: frameParams
            )
        } else {
            let packedWorld = PackedWorldBuffers(
                packedGaussians: input.gaussians,
                harmonics: input.harmonics
            )
            output = encodeRenderToTextures(
                commandBuffer: commandBuffer,
                gaussianCount: input.gaussianCount,
                packedWorldBuffers: packedWorld,
                cameraUniforms: cameraUniforms,
                frameParams: frameParams
            )
        }

        guard let result = output else { return nil }
        return TextureRenderResult(color: result.color, depth: result.depth, alpha: result.alpha)
    }

    /// Render to CPU-readable buffers (protocol method)
    public func render(
        toBuffer commandBuffer: MTLCommandBuffer,
        input: GaussianInput,
        camera: CameraParams,
        width: Int,
        height: Int,
        whiteBackground: Bool
    ) -> BufferRenderResult? {
        let cameraUniforms = CameraUniformsSwift(
            viewMatrix: camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            cameraCenter: camera.position,
            pixelFactor: 1.0,
            focalX: camera.focalX,
            focalY: camera.focalY,
            width: Float(width),
            height: Float(height),
            nearPlane: camera.near,
            farPlane: camera.far,
            shComponents: UInt32(input.shComponents),
            gaussianCount: UInt32(input.gaussianCount)
        )

        let frameParams = FrameParams(
            gaussianCount: input.gaussianCount,
            whiteBackground: whiteBackground
        )

        let packedWorld = PackedWorldBuffers(
            packedGaussians: input.gaussians,
            harmonics: input.harmonics
        )

        guard let output = encodeRender(
            commandBuffer: commandBuffer,
            gaussianCount: input.gaussianCount,
            packedWorldBuffers: packedWorld,
            cameraUniforms: cameraUniforms,
            frameParams: frameParams
        ) else { return nil }

        return BufferRenderResult(
            color: output.colorOutGPU,
            depth: output.depthOutGPU,
            alpha: output.alphaOutGPU
        )
    }

    // MARK: - Internal Methods

    func triggerCapture(path: String) {
        self.nextFrameCapturePath = path
    }

    private static func computeLayout(
        limits: RendererLimits,
        sortAlgorithm: SortAlgorithm,
        useFusedPipeline: Bool,
        precision: Precision,
        device: MTLDevice,
        radixSortEncoder: RadixSortEncoder
    ) -> FrameResourceLayout {
        let gaussianCapacity = limits.maxGaussians
        let tileCount = limits.maxTileCount

        // maxAssignmentCapacity: worst-case tile assignments we must support
        // Each assignment is a (gaussian, tile) pair where gaussian overlaps tile
        // Bound 1: tileCount * maxPerTile (per-tile limit)
        // Bound 2: gaussianCapacity * tilesPerGaussian (each gaussian spans limited tiles)
        let tileCapacity = tileCount * limits.maxPerTile
        let gaussianTileCapacity = gaussianCapacity * 8  // each gaussian spans ~8 tiles max
        let maxAssignmentCapacity = min(tileCapacity, gaussianTileCapacity)

        // paddedCapacity: aligned for sort dispatch
        // - Radix: 1024-aligned (blockSize * grainSize)
        // - Bitonic: power-of-2 (much larger, can be 2x maxAssignmentCapacity)
        let paddedCapacity: Int = {
            if sortAlgorithm == .radix {
                let block = radixSortEncoder.blockSize * radixSortEncoder.grainSize
                return ((maxAssignmentCapacity + block - 1) / block) * block
            } else {
                var value = 1
                while value < maxAssignmentCapacity { value <<= 1 }
                return value
            }
        }()

        let strideForPrecision: Int = (precision == .float16) ? 2 : 4
        let half2Stride = MemoryLayout<UInt16>.stride * 2
        let half4Stride = MemoryLayout<UInt16>.stride * 4
        func strideForMeans() -> Int { precision == .float16 ? half2Stride : 8 }
        func strideForConics() -> Int { precision == .float16 ? half4Stride : 16 }
        func strideForColors() -> Int { precision == .float16 ? MemoryLayout<UInt16>.stride * 3 : 12 }
        func strideForOpacities() -> Int { precision == .float16 ? MemoryLayout<UInt16>.stride : 4 }
        func strideForDepths() -> Int { precision == .float16 ? MemoryLayout<UInt16>.stride : 4 }
        func strideForInterleaved() -> Int { precision == .float16 ? 24 : 48 }

        let prefixBlockSize = 256
        let prefixGrainSize = 4
        let elementsPerGroup = prefixBlockSize * prefixGrainSize
        let groups = (gaussianCapacity + elementsPerGroup - 1) / elementsPerGroup

        var heapAllocations: [(label: String, length: Int, options: MTLResourceOptions)] = []
        func add(_ label: String, _ length: Int, _ opts: MTLResourceOptions = .storageModePrivate) {
            heapAllocations.append((label, max(1, length), opts))
        }

        // Per-gaussian buffers (sized to gaussianCapacity)
        add("Bounds", gaussianCapacity * MemoryLayout<SIMD4<Int32>>.stride)
        add("Coverage", gaussianCapacity * MemoryLayout<UInt32>.stride)
        add("Offsets", (gaussianCapacity + 1) * MemoryLayout<UInt32>.stride)
        add("PartialSums", max(groups, 1) * MemoryLayout<UInt32>.stride)
        add("ScatterDispatch", 3 * MemoryLayout<UInt32>.stride)

        // Tile assignment buffers (sized to paddedCapacity for sort alignment)
        add("TileIndices", paddedCapacity * MemoryLayout<Int32>.stride)
        add("TileIds", paddedCapacity * MemoryLayout<Int32>.stride)

        // Ordered/packed buffers (sized to maxAssignmentCapacity)
        add("OrderedHeaders", tileCount * MemoryLayout<GaussianHeader>.stride)
        add("PackedMeans", maxAssignmentCapacity * strideForMeans())
        add("PackedConics", maxAssignmentCapacity * strideForConics())
        add("PackedColors", maxAssignmentCapacity * strideForColors())
        add("PackedOpacities", maxAssignmentCapacity * strideForOpacities())
        add("PackedDepths", maxAssignmentCapacity * strideForDepths())
        add("ActiveTileIndices", tileCount * MemoryLayout<UInt32>.stride)

        // Sort key/index buffers (sized to paddedCapacity for sort alignment)
        add("SortKeys", paddedCapacity * MemoryLayout<SIMD2<UInt32>>.stride)
        add("SortedIndices", paddedCapacity * MemoryLayout<Int32>.stride)

        // Radix-specific buffers (only when using radix sort)
        // Note: gridCapacity is for worst-case; runtime uses ceil(totalAssignments/1024) blocks
        if sortAlgorithm == .radix {
            let valuesPerGroup = radixSortEncoder.blockSize * radixSortEncoder.grainSize
            let gridCapacity = max(1, (maxAssignmentCapacity + valuesPerGroup - 1) / valuesPerGroup)
            let histogramCapacity = gridCapacity * radixSortEncoder.radix
            add("RadixHist", histogramCapacity * MemoryLayout<UInt32>.stride)
            add("RadixBlockSums", gridCapacity * MemoryLayout<UInt32>.stride)
            add("RadixScanned", histogramCapacity * MemoryLayout<UInt32>.stride)
            // Key/payload scratch sized to maxAssignmentCapacity (not paddedCapacity)
            add("RadixFused", maxAssignmentCapacity * MemoryLayout<UInt64>.stride)
            add("RadixScratch", maxAssignmentCapacity * MemoryLayout<UInt64>.stride)
            add("RadixPayload", maxAssignmentCapacity * MemoryLayout<UInt32>.stride)
        }

        // Fused pipeline buffers (always enabled)
        add("InterleavedGaussians", gaussianCapacity * strideForInterleaved())
        add("PackedGaussiansFused", maxAssignmentCapacity * strideForInterleaved())

        func heapSize(for allocations: [(label: String, length: Int, options: MTLResourceOptions)]) -> Int {
            var offset = 0
            for req in allocations {
                let sa = device.heapBufferSizeAndAlign(length: req.length, options: req.options)
                let align = max(Int(sa.align), 1)
                offset = (offset + align - 1) & ~(align - 1)
                offset += Int(sa.size)
            }
            return (offset + 65535) & ~65535
        }

        let sharedAllocations: [(label: String, length: Int, options: MTLResourceOptions)] = [
            ("TileHeader", MemoryLayout<TileAssignmentHeaderSwift>.stride, .storageModeShared),
            ("ActiveTileCount", MemoryLayout<UInt32>.stride, .storageModeShared)
        ]

        return FrameResourceLayout(
            limits: limits,
            maxAssignmentCapacity: maxAssignmentCapacity,
            paddedCapacity: paddedCapacity,
            heapAllocations: heapAllocations,
            sharedAllocations: sharedAllocations,
            heapSize: heapSize(for: heapAllocations),
            pixelCapacity: limits.maxWidth * limits.maxHeight,
            precisionStride: strideForPrecision
        )
    }

    // MARK: - Public Render Methods (Packed Input Only)

    /* Internal C-API methods removed - use public encodeRender* methods with PackedWorldBuffers */

    /* REMOVED: render, renderRaw, renderWorld - these used old non-packed buffers
       and were for the disabled C API. Use encodeRender/encodeRenderToTextures instead.
    */

    public func encodeRender(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        cameraUniforms: CameraUniformsSwift,
        frameParams: FrameParams
    ) -> RenderOutputBuffers? {
        let params = limits.buildParams(from: frameParams)
        guard self.validateLimits(gaussianCount: gaussianCount) else { return nil }
        guard let (frame, slotIndex) = self.acquireFrame(width: limits.maxWidth, height: limits.maxHeight) else {
            return nil
        }

        if let capturePath = self.nextFrameCapturePath {
            let descriptor = MTLCaptureDescriptor()
            descriptor.captureObject = self.device
            descriptor.destination = .gpuTraceDocument
            descriptor.outputURL = URL(fileURLWithPath: capturePath)
            let manager = MTLCaptureManager.shared()
            try? manager.startCapture(with: descriptor)
            self.nextFrameCapturePath = nil // Consume the capture path
        }

        guard let gaussianBuffers = self.prepareGaussianBuffers(count: gaussianCount) else {
            self.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
            return nil
        }

        self.projectEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: cameraUniforms,
            gaussianBuffers: gaussianBuffers,
            precision: self.effectivePrecision
        )

        guard let assignment = self.buildTileAssignmentsGPU(commandBuffer: commandBuffer, gaussianCount: gaussianCount, gaussianBuffers: gaussianBuffers, params: params, frame: frame) else {
            self.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }

            return nil
        }

        guard let ordered = self.buildOrderedGaussians(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            assignment: assignment,
            gaussianBuffers: gaussianBuffers,
            params: params,
            frame: frame,
            precision: self.effectivePrecision
        ) else {
            self.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }

            return nil
        }

        let submission = self.submitRender(
            commandBuffer: commandBuffer,
            ordered: ordered,
            params: params,
            frame: frame,
            slotIndex: slotIndex,
            precision: self.effectivePrecision
        )
        
        guard submission else {
            self.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
            return nil
        }

        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
        }
        
        return frame.outputBuffers
    }
    
    // New API for Texture-based rendering
    public func encodeRenderToTextures(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        cameraUniforms: CameraUniformsSwift,
        frameParams: FrameParams
    ) -> RenderOutputTextures? {
        let params = limits.buildParams(from: frameParams)
        // validateLimits now crashes with descriptive message on failure
        guard self.validateLimits(gaussianCount: gaussianCount) else {
            fatalError("[encodeRenderToTextures] validateLimits failed - should have crashed with details")
        }

        guard let (frame, slotIndex) = self.acquireFrame(width: limits.maxWidth, height: limits.maxHeight) else {
            fatalError("[encodeRenderToTextures] acquireFrame failed for \(limits.maxWidth)x\(limits.maxHeight)")
        }

        if let capturePath = self.nextFrameCapturePath {
            let descriptor = MTLCaptureDescriptor()
            descriptor.captureObject = self.device
            descriptor.destination = .gpuTraceDocument
            descriptor.outputURL = URL(fileURLWithPath: capturePath)
            let manager = MTLCaptureManager.shared()
            try? manager.startCapture(with: descriptor)
            self.nextFrameCapturePath = nil
        }

        guard let gaussianBuffers = self.prepareGaussianBuffers(count: gaussianCount) else {
            fatalError("[encodeRenderToTextures] prepareGaussianBuffers failed for count \(gaussianCount)")
        }

        self.projectEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: cameraUniforms,
            gaussianBuffers: gaussianBuffers,
            precision: self.effectivePrecision
        )

        // buildTileAssignmentsGPU now crashes with descriptive message on failure
        guard let assignment = self.buildTileAssignmentsGPU(commandBuffer: commandBuffer, gaussianCount: gaussianCount, gaussianBuffers: gaussianBuffers, params: params, frame: frame) else {
            fatalError("[encodeRenderToTextures] buildTileAssignmentsGPU failed - should have crashed with details")
        }

        // buildOrderedGaussians now crashes with descriptive message on failure
        guard let ordered = self.buildOrderedGaussians(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            assignment: assignment,
            gaussianBuffers: gaussianBuffers,
            params: params,
            frame: frame,
            precision: self.effectivePrecision,
            textureOnlyFused: true  // Skip redundant pack for texture render
        ) else {
            fatalError("[encodeRenderToTextures] buildOrderedGaussians failed - should have crashed with details")
        }

        // Submit render to textures
        let textures = frame.outputTextures
        let dispatchArgs = frame.dispatchArgs

        // Fused render: single-struct reads for cache efficiency
        guard ordered.packedGaussiansFused != nil else {
            fatalError("[encodeRenderToTextures] Fused pipeline unavailable")
        }
        self.fusedPipelineEncoder.encodeCompleteFusedRender(
            commandBuffer: commandBuffer,
            orderedBuffers: ordered,
            outputTextures: textures,
            params: params,
            dispatchArgs: dispatchArgs,
            dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            precision: self.effectivePrecision
        )

        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
        }

        return textures
    }

    /// API for full half-precision pipeline (half input -> half output).
    /// Uses PackedWorldBuffersHalf for optimal memory bandwidth.
    public func encodeRenderToTextureHalf(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffersHalf: PackedWorldBuffersHalf,
        cameraUniforms: CameraUniformsSwift,
        frameParams: FrameParams
    ) -> RenderOutputTextures? {
        let params = limits.buildParams(from: frameParams)
        // validateLimits now crashes with descriptive message on failure
        guard self.validateLimits(gaussianCount: gaussianCount) else {
            fatalError("[encodeRenderToTextureHalf] validateLimits failed - should have crashed with details")
        }

        guard let (frame, slotIndex) = self.acquireFrame(width: limits.maxWidth, height: limits.maxHeight) else {
            fatalError("[encodeRenderToTextureHalf] acquireFrame failed for \(limits.maxWidth)x\(limits.maxHeight)")
        }

        if let capturePath = self.nextFrameCapturePath {
            let descriptor = MTLCaptureDescriptor()
            descriptor.captureObject = self.device
            descriptor.destination = .gpuTraceDocument
            descriptor.outputURL = URL(fileURLWithPath: capturePath)
            let manager = MTLCaptureManager.shared()
            try? manager.startCapture(with: descriptor)
            self.nextFrameCapturePath = nil
        }

        // Prepare gaussian buffers with half-precision layout
        guard let gaussianBuffers = self.prepareGaussianBuffersHalf(count: gaussianCount) else {
            fatalError("[encodeRenderToTextureHalf] prepareGaussianBuffersHalf failed for count \(gaussianCount)")
        }

        // Project from half-precision packed world data to half gaussian data (half_half pipeline)
        self.projectEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            packedWorldBuffersHalf: packedWorldBuffersHalf,
            cameraUniforms: cameraUniforms,
            gaussianBuffers: gaussianBuffers
        )

        // buildTileAssignmentsGPU now crashes with descriptive message on failure
        guard let assignment = self.buildTileAssignmentsGPU(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            gaussianBuffers: gaussianBuffers,
            params: params,
            frame: frame,
            precision: .float16  // Half precision for this path
        ) else {
            fatalError("[encodeRenderToTextureHalf] buildTileAssignmentsGPU failed - should have crashed with details")
        }

        // buildOrderedGaussians now crashes with descriptive message on failure
        guard let ordered = self.buildOrderedGaussians(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            assignment: assignment,
            gaussianBuffers: gaussianBuffers,
            params: params,
            frame: frame,
            precision: .float16,  // Always half for this path
            textureOnlyFused: true  // Skip redundant pack for texture render
        ) else {
            fatalError("[encodeRenderToTextureHalf] buildOrderedGaussians failed - should have crashed with details")
        }

        let textures = frame.outputTextures
        let dispatchArgs = frame.dispatchArgs

        // Fused render: single-struct reads for cache efficiency
        guard ordered.packedGaussiansFused != nil else {
            fatalError("[encodeRenderToTextureHalf] Fused pipeline unavailable")
        }
        self.fusedPipelineEncoder.encodeCompleteFusedRender(
            commandBuffer: commandBuffer,
            orderedBuffers: ordered,
            outputTextures: textures,
            params: params,
            dispatchArgs: dispatchArgs,
            dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            precision: .float16
        )

        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
        }

        return textures
    }

    /// Render to user-provided texture (most direct path).
    /// Allows GSViewer to pass its own drawable texture for zero-copy rendering.
    public func encodeRenderToTargetTexture(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        cameraUniforms: CameraUniformsSwift,
        frameParams: FrameParams,
        targetColor: MTLTexture,
        targetDepth: MTLTexture?,
        targetAlpha: MTLTexture?
    ) -> Bool {
        let params = limits.buildParams(from: frameParams)
        guard self.validateLimits(gaussianCount: gaussianCount) else {
            return false
        }
        guard let (frame, slotIndex) = self.acquireFrame(width: limits.maxWidth, height: limits.maxHeight) else {
            return false
        }

        guard let gaussianBuffers = self.prepareGaussianBuffersHalf(count: gaussianCount) else {
            self.releaseFrame(index: slotIndex)
            return false
        }

        self.projectEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: cameraUniforms,
            gaussianBuffers: gaussianBuffers,
            precision: .float16
        )

        guard let assignment = self.buildTileAssignmentsGPU(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            gaussianBuffers: gaussianBuffers,
            params: params,
            frame: frame,
            precision: .float16  // Half precision for this path
        ) else {
            self.releaseFrame(index: slotIndex)
            return false
        }

        guard let ordered = self.buildOrderedGaussians(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            assignment: assignment,
            gaussianBuffers: gaussianBuffers,
            params: params,
            frame: frame,
            precision: .float16,
            textureOnlyFused: true  // Skip redundant pack for texture render
        ) else {
            self.releaseFrame(index: slotIndex)
            return false
        }

        let dispatchArgs = frame.dispatchArgs

        // Use user's target textures
        let outputTextures = RenderOutputTextures(
            color: targetColor,
            depth: targetDepth ?? frame.outputTextures.depth,
            alpha: targetAlpha ?? frame.outputTextures.alpha
        )

        // Fused render: single-struct reads for cache efficiency
        guard ordered.packedGaussiansFused != nil else {
            fatalError("[encodeRenderToTargetTexture] Fused pipeline unavailable")
        }
        self.fusedPipelineEncoder.encodeCompleteFusedRender(
            commandBuffer: commandBuffer,
            orderedBuffers: ordered,
            outputTextures: outputTextures,
            params: params,
            dispatchArgs: dispatchArgs,
            dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            precision: .float16
        )

        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.releaseFrame(index: slotIndex)
        }

        return true
    }

    // MARK: - Helper Methods

    private func validateLimits(gaussianCount: Int) -> Bool {
        // CRASH on validation failures - silent returns cause black screens!
        // Since params are built from limits, we only need to validate gaussianCount
        precondition(gaussianCount <= limits.maxGaussians,
            "gaussianCount (\(gaussianCount)) exceeds limits.maxGaussians (\(limits.maxGaussians))")
        return true
    }

    // Legacy validation for tests that construct RenderParams manually
    private func validateLimits(gaussianCount: Int, params: RenderParams) -> Bool {
        precondition(gaussianCount <= limits.maxGaussians,
            "gaussianCount (\(gaussianCount)) exceeds limits.maxGaussians (\(limits.maxGaussians))")
        precondition(Int(params.tilesX * params.tilesY) <= limits.maxTileCount,
            "tileCount exceeds limits.maxTileCount (\(limits.maxTileCount))")
        return true
    }
    
    internal func buildTileAssignmentsGPU(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        gaussianBuffers: GaussianInputBuffers,
        params: RenderParams,
        frame: FrameResources,
        precision: Precision = .float32
    ) -> TileAssignmentBuffers? {
        // CRASH on failures - silent returns cause black screens!
        precondition(gaussianCount <= limits.maxGaussians,
            "[buildTileAssignmentsGPU] gaussianCount (\(gaussianCount)) > limits.maxGaussians (\(limits.maxGaussians))")

        let tileCount = Int(params.tilesX * params.tilesY)
        precondition(tileCount <= limits.maxTileCount,
            "[buildTileAssignmentsGPU] tileCount (\(tileCount)) > limits.maxTileCount (\(limits.maxTileCount))")

        // Validate capacity using same formula as computeLayout:
        // maxAssignmentCapacity = min(tileCapacity, gaussianTileCapacity)
        let perTileLimit = (params.maxPerTile == 0) ? UInt32(limits.maxPerTile) : min(params.maxPerTile, UInt32(limits.maxPerTile))
        let tileCapacity = tileCount * Int(perTileLimit)
        let gaussianTileCapacity = gaussianCount * 8  // each gaussian spans ~8 tiles max
        let requiredCapacity = min(tileCapacity, gaussianTileCapacity)

        precondition(requiredCapacity <= frame.tileAssignmentMaxAssignments,
            "[buildTileAssignmentsGPU] requiredCapacity (\(requiredCapacity)) > frame.tileAssignmentMaxAssignments (\(frame.tileAssignmentMaxAssignments)) - increase limits.maxPerTile or maxGaussians")

        resetTileBuilderState(commandBuffer: commandBuffer, frame: frame)

        self.tileBoundsEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianBuffers: gaussianBuffers,
            boundsBuffer: frame.boundsBuffer,
            params: params,
            gaussianCount: gaussianCount,
            precision: precision
        )

        // SIMD-optimized fused coverage+scatter V2
        self.fusedCoverageScatterEncoderV2.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            tileWidth: Int(params.tileWidth),
            tileHeight: Int(params.tileHeight),
            tilesX: Int(params.tilesX),
            maxAssignments: frame.tileAssignmentMaxAssignments,
            boundsBuffer: frame.boundsBuffer,
            coverageBuffer: frame.coverageBuffer,
            opacitiesBuffer: gaussianBuffers.opacities,
            meansBuffer: gaussianBuffers.means,
            conicsBuffer: gaussianBuffers.conics,
            tileIndicesBuffer: frame.tileIndices,
            tileIdsBuffer: frame.tileIds,
            tileAssignmentHeader: frame.tileAssignmentHeader,
            precision: precision
        )
        
        return TileAssignmentBuffers(
            tileCount: tileCount,
            maxAssignments: frame.tileAssignmentMaxAssignments,
            tileIndices: frame.tileIndices,
            tileIds: frame.tileIds,
            header: frame.tileAssignmentHeader
        )
    }
    
    
    internal func buildOrderedGaussians(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        assignment: TileAssignmentBuffers,
        gaussianBuffers: GaussianInputBuffers,
        params: RenderParams,
        frame: FrameResources,
        precision: Precision,
        textureOnlyFused: Bool = false  // When true, skip standard pack payload for fused texture render
    ) -> OrderedGaussianBuffers? {
        // CRASH on failures - silent returns cause black screens!
        precondition(gaussianCount <= limits.maxGaussians,
            "[buildOrderedGaussians] gaussianCount (\(gaussianCount)) > limits.maxGaussians (\(limits.maxGaussians))")
        precondition(assignment.maxAssignments <= frame.tileAssignmentMaxAssignments,
            "[buildOrderedGaussians] assignment.maxAssignments (\(assignment.maxAssignments)) > frame.tileAssignmentMaxAssignments (\(frame.tileAssignmentMaxAssignments))")

        let sortKeysBuffer = frame.sortKeys
        let sortedIndicesBuffer = frame.sortedIndices

        let paddedCount = frame.tileAssignmentPaddedCapacity

        let dispatchArgs = frame.dispatchArgs

        self.dispatchEncoder.encode(
            commandBuffer: commandBuffer,
            header: assignment.header,
            dispatchArgs: dispatchArgs,
            maxAssignments: frame.tileAssignmentMaxAssignments
        )
        
        self.sortKeyGenEncoder.encode(
            commandBuffer: commandBuffer,
            tileIds: assignment.tileIds,
            tileIndices: assignment.tileIndices,
            depths: gaussianBuffers.depths,
            sortKeys: sortKeysBuffer,
            sortedIndices: sortedIndicesBuffer,
            header: assignment.header,
            dispatchArgs: dispatchArgs,
            dispatchOffset: DispatchSlot.sortKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            precision: precision
        )

        if self.sortAlgorithm == .radix {
             let radixBuffers = RadixBufferSet(
                histogram: frame.radixHistogram!,
                blockSums: frame.radixBlockSums!,
                scannedHistogram: frame.radixScannedHistogram!,
                fusedKeys: frame.radixFusedKeys!,
                scratchKeys: frame.radixKeysScratch!,
                scratchPayload: frame.radixPayloadScratch!
            )
            
            let offsets = (
                fuse: DispatchSlot.fuseKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                unpack: DispatchSlot.unpackKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                histogram: DispatchSlot.radixHistogram.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                scanBlocks: DispatchSlot.radixScanBlocks.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                exclusive: DispatchSlot.radixExclusive.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                apply: DispatchSlot.radixApply.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                scatter: DispatchSlot.radixScatter.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
            )
            
            self.radixSortEncoder.encode(
                commandBuffer: commandBuffer,
                keyBuffer: sortKeysBuffer,
                sortedIndices: sortedIndicesBuffer,
                header: assignment.header,
                dispatchArgs: dispatchArgs,
                radixBuffers: radixBuffers,
                offsets: offsets,
                tileCount: assignment.tileCount
            )
        } else {
             let offsets = (
                first: DispatchSlot.bitonicFirst.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                general: DispatchSlot.bitonicGeneral.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                final: DispatchSlot.bitonicFinal.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
            )
            
            self.bitonicSortEncoder.encode(
                commandBuffer: commandBuffer,
                sortKeys: sortKeysBuffer,
                sortedIndices: sortedIndicesBuffer,
                header: assignment.header,
                dispatchArgs: dispatchArgs,
                offsets: offsets,
                paddedCapacity: paddedCount
            )
        }
        
        let orderedBuffers = OrderedBufferSet(
            headers: frame.orderedHeaders,
            means: frame.packedMeans,
            conics: frame.packedConics,
            colors: frame.packedColors,
            opacities: frame.packedOpacities,
            depths: frame.packedDepths
        )
        frame.activeTileCount.contents().storeBytes(of: UInt32(0), as: UInt32.self)

        // Check if we can use texture-only fused path (skip redundant standard pack payload)
        let canUseFusedOnly = textureOnlyFused &&
                              frame.interleavedGaussians != nil &&
                              frame.packedGaussiansFused != nil

        var fusedBuffer: MTLBuffer? = nil

        if canUseFusedOnly {
            // Texture-only fused path: skip standard pack payload, only build headers + fused data
            self.packEncoder.encodeHeadersAndActiveTiles(
                commandBuffer: commandBuffer,
                sortedKeys: sortKeysBuffer,
                assignment: assignment,
                orderedHeaders: orderedBuffers.headers,
                activeTileIndices: frame.activeTileIndices,
                activeTileCount: frame.activeTileCount
            )

            let fusedEncoder = self.fusedPipelineEncoder
            let interleavedBuffer = frame.interleavedGaussians!
            let packedFusedBuffer = frame.packedGaussiansFused!

            // 1. Interleave gaussian data into single struct per gaussian
            fusedEncoder.encodeInterleave(
                commandBuffer: commandBuffer,
                gaussianBuffers: gaussianBuffers,
                interleavedOutput: interleavedBuffer,
                gaussianCount: gaussianCount,
                precision: precision
            )

            // 2. Pack using interleaved data (single struct read for texture render)
            fusedEncoder.encodePackFused(
                commandBuffer: commandBuffer,
                sortedIndices: sortedIndicesBuffer,
                interleavedGaussians: interleavedBuffer,
                packedOutput: packedFusedBuffer,
                header: assignment.header,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.pack.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                precision: precision
            )

            fusedBuffer = packedFusedBuffer
        } else {
            // Standard path: full pack with separated buffers (needed for buffer render)
            // Don't run fused pack here - it's not used for buffer render
            self.packEncoder.encode(
                commandBuffer: commandBuffer,
                sortedIndices: sortedIndicesBuffer,
                sortedKeys: sortKeysBuffer,
                gaussianBuffers: gaussianBuffers,
                orderedBuffers: orderedBuffers,
                assignment: assignment,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.pack.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                activeTileIndices: frame.activeTileIndices,
                activeTileCount: frame.activeTileCount,
                precision: precision
            )
            // fusedBuffer stays nil - buffer render uses separated buffers
        }

        return OrderedGaussianBuffers(
            headers: orderedBuffers.headers,
            means: orderedBuffers.means,
            conics: orderedBuffers.conics,
            colors: orderedBuffers.colors,
            opacities: orderedBuffers.opacities,
            depths: orderedBuffers.depths,
            tileCount: assignment.tileCount,
            activeTileIndices: frame.activeTileIndices,
            activeTileCount: frame.activeTileCount,
            precision: precision,
            packedGaussiansFused: fusedBuffer
        )
    }
        
    
    private func submitRender(
        commandBuffer: MTLCommandBuffer,
        ordered: OrderedGaussianBuffers,
        params: RenderParams,
        frame: FrameResources,
        slotIndex: Int,
        precision: Precision
    ) -> Bool {
        guard let outputBuffers = frame.outputBuffers else {
            fatalError("[encodeRenderToBuffers] outputBuffers is nil - did you create the renderer with textureOnly: true?")
        }
        self.renderEncoder.encode(
            commandBuffer: commandBuffer,
            orderedBuffers: ordered,
            outputBuffers: outputBuffers,
            params: params,
            dispatchArgs: frame.dispatchArgs,
            dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            precision: precision
        )

        return true
    }
    
    private func encodeAndRunRender(
        commandBuffer: MTLCommandBuffer,
        ordered: OrderedGaussianBuffers,
        colorOutPtr: UnsafeMutablePointer<Float>?,
        depthOutPtr: UnsafeMutablePointer<Float>?,
        alphaOutPtr: UnsafeMutablePointer<Float>?,
        params: RenderParams,
        gaussianCount: Int = 0
    ) -> Int32 {
        guard let (frame, slotIndex) = self.acquireFrame(width: Int(params.width), height: Int(params.height)) else { return -5 }
        
        return self.encodeAndRunRenderWithFrame(
            commandBuffer: commandBuffer,
            ordered: ordered,
            frame: frame,
            slotIndex: slotIndex,
            colorOutPtr: colorOutPtr,
            depthOutPtr: depthOutPtr,
            alphaOutPtr: alphaOutPtr,
            params: params,
            gaussianCount: gaussianCount
        )
    }
    
    internal func encodeAndRunRenderWithFrame(
        commandBuffer: MTLCommandBuffer,
        ordered: OrderedGaussianBuffers,
        frame: FrameResources,
        slotIndex: Int,
        colorOutPtr: UnsafeMutablePointer<Float>?,
        depthOutPtr: UnsafeMutablePointer<Float>?,
        alphaOutPtr: UnsafeMutablePointer<Float>?,
        params: RenderParams,
        gaussianCount: Int = 0
    ) -> Int32 {
        let cpuReadback = !(colorOutPtr == nil && depthOutPtr == nil && alphaOutPtr == nil)
        guard self.submitRender(
            commandBuffer: commandBuffer,
            ordered: ordered,
            params: params,
            frame: frame,
            slotIndex: slotIndex,
            precision: ordered.precision  // Use the precision the buffers were packed with
        ) else { return -5 }

        if !cpuReadback {
            commandBuffer.addCompletedHandler { [weak self] _ in
                self?.releaseFrame(index: slotIndex)
            }
            commandBuffer.commit()
            return 0
        }
        
        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.releaseFrame(index: slotIndex)
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Capture GPU execution time from command buffer timestamps
        let gpuStart = commandBuffer.gpuStartTime
        let gpuEnd = commandBuffer.gpuEndTime
        if gpuStart > 0 && gpuEnd > gpuStart {
            self.lastGPUTime = gpuEnd - gpuStart
        }

        let pixelCount = Int(params.width * params.height)
        guard let outputBuffers = frame.outputBuffers else {
            fatalError("[renderRaw] outputBuffers is nil - did you create the renderer with textureOnly: true?")
        }
        if let p = colorOutPtr { memcpy(p, outputBuffers.colorOutGPU.contents(), pixelCount * 12) }
        if let p = depthOutPtr { memcpy(p, outputBuffers.depthOutGPU.contents(), pixelCount * 4) }
        if let p = alphaOutPtr { memcpy(p, outputBuffers.alphaOutGPU.contents(), pixelCount * 4) }

        return 0
    }
    
    // MARK: - Resource Management

    /* REMOVED: prepareWorldBuffers - was for C API with non-packed buffers */
    
    internal func prepareGaussianBuffers(count: Int) -> GaussianInputBuffers? {
        guard count <= limits.maxGaussians else { return nil }
        guard
            let means = self.gaussianMeansCache,
            let radii = self.gaussianRadiiCache,
            let mask = self.gaussianMaskCache,
            let depths = self.gaussianDepthsCache,
            let conics = self.gaussianConicsCache,
            let colors = self.gaussianColorsCache,
            let opacities = self.gaussianOpacitiesCache
        else { return nil }

        return GaussianInputBuffers(
            means: means,
            radii: radii,
            mask: mask,
            depths: depths,
            conics: conics,
            colors: colors,
            opacities: opacities
        )
    }

    internal func prepareGaussianBuffersHalf(count: Int) -> GaussianInputBuffers? {
        guard count <= limits.maxGaussians else { return nil }
        guard
            let means = self.gaussianMeansHalfCache,
            let radii = self.gaussianRadiiCache,
            let mask = self.gaussianMaskCache,
            let depths = self.gaussianDepthsHalfCache,
            let conics = self.gaussianConicsHalfCache,
            let colors = self.gaussianColorsHalfCache,
            let opacities = self.gaussianOpacitiesHalfCache
        else { return nil }

        return GaussianInputBuffers(
            means: means,
            radii: radii,
            mask: mask,
            depths: depths,
            conics: conics,
            colors: colors,
            opacities: opacities
        )
    }

    private func resetTileBuilderState(commandBuffer: MTLCommandBuffer, frame: FrameResources) {
        // GPU-only reset: use compute shader for lower overhead than blit encoder
        // Resets: totalAssignments, overflow, and activeTileCount in a single dispatch
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("[resetTileBuilderState] Failed to create compute encoder")
        }
        encoder.label = "ResetTileBuilderState"
        encoder.setComputePipelineState(resetTileBuilderStatePipeline)
        encoder.setBuffer(frame.tileAssignmentHeader, offset: 0, index: 0)
        encoder.setBuffer(frame.activeTileCount, offset: 0, index: 1)
        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
    }

    internal func acquireFrame(width: Int, height: Int) -> (FrameResources, Int)? {
        guard width <= limits.maxWidth, height <= limits.maxHeight else { return nil }
        frameLock.lock()
        defer { frameLock.unlock() }

        let available = min(maxInFlightFrames, frameResources.count)
        guard available > 0 else { return nil }

        for _ in 0..<available {
            let idx = frameCursor % available
            frameCursor = (frameCursor + 1) % max(available, 1)
            if frameInUse[idx] == false {
                frameInUse[idx] = true
                return (frameResources[idx], idx)
            }
        }
        return nil
    }
    
    internal func releaseFrame(index: Int) {
        frameLock.lock()
        defer { frameLock.unlock() }
        if index >= 0 && index < frameInUse.count {
            frameInUse[index] = false
        }
    }

    // MARK: - Debug/Test helpers
    internal func debugHeapSizeBytes() -> Int? {
        return frameResources.first?.heapAllocator.heap?.size
    }

    func makeBuffer<T>(ptr: UnsafePointer<T>, count: Int) -> MTLBuffer? {
        self.device.makeBuffer(bytes: ptr, length: count * MemoryLayout<T>.stride, options: .storageModeShared)
    }
    
    /// Choose padded assignment capacity based on sort algorithm to avoid over-allocation.
    private func paddedAssignmentCapacity(for totalAssignments: Int) -> Int {
        guard totalAssignments > 0 else { return 1 }
        if self.sortAlgorithm == .radix {
            let block = self.radixSortEncoder.blockSize * self.radixSortEncoder.grainSize
            return ((totalAssignments + block - 1) / block) * block
        } else {
            return self.nextPowerOfTwo(value: totalAssignments)
        }
    }
    internal func nextPowerOfTwo(value: Int) -> Int {
        var result = 1
        while result < value {
            result <<= 1
        }
        return result
    }
}

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

    public var maxTileCount: Int {
        let tilesX = (maxWidth + tileWidth - 1) / tileWidth
        let tilesY = (maxHeight + tileHeight - 1) / tileHeight
        return max(1, tilesX * tilesY)
    }

    public init(maxGaussians: Int, maxWidth: Int, maxHeight: Int, tileWidth: Int = 32, tileHeight: Int = 16, maxPerTile: Int? = nil) {
        self.maxGaussians = max(1, maxGaussians)
        self.maxWidth = max(1, maxWidth)
        self.maxHeight = max(1, maxHeight)
        self.tileWidth = max(1, tileWidth)
        self.tileHeight = max(1, tileHeight)
        // Default to a sane per-tile cap to prevent runaway heap sizing when callers omit it.
        // 256 per tile is reasonable: 16K tiles × 256 = 4M assignments = ~200MB (not 3GB!)
        self.maxPerTile = max(1, maxPerTile ?? 256)
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
    let maxAssignments: Int
    let paddedAssignments: Int
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

    // Sort Buffers
    let sortKeys: MTLBuffer
    let sortedIndices: MTLBuffer

    // Radix Buffers
    let radixHistogram: MTLBuffer
    let radixBlockSums: MTLBuffer
    let radixScannedHistogram: MTLBuffer
    let radixFusedKeys: MTLBuffer
    let radixKeysScratch: MTLBuffer
    let radixPayloadScratch: MTLBuffer

    // Fused Pipeline Buffers (interleaved data for cache efficiency)
    let interleavedGaussians: MTLBuffer?
    let packedGaussiansFused: MTLBuffer?

    // Dispatch Args (Per-frame to avoid race on indirect dispatch)
    let dispatchArgs: MTLBuffer

    // Output Buffers
    var outputBuffers: RenderOutputBuffers
    // Output Textures (Alternative)
    var outputTextures: RenderOutputTextures

    let device: MTLDevice

    init(device: MTLDevice, layout: FrameResourceLayout, residencySetProvider: (() -> (any MTLResidencySet)?)?, useHeap: Bool) {
        self.device = device
        self.heapAllocator = HeapAllocator(device: device, residencySetProvider: residencySetProvider)
        self.tileAssignmentMaxAssignments = layout.maxAssignments
        self.tileAssignmentPaddedCapacity = layout.paddedAssignments

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
        headerPtr.pointee.maxAssignments = UInt32(layout.maxAssignments)
        headerPtr.pointee.paddedCount = UInt32(layout.paddedAssignments)
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
        self.radixHistogram = buffer("RadixHist")
        self.radixBlockSums = buffer("RadixBlockSums")
        self.radixScannedHistogram = buffer("RadixScanned")
        self.radixFusedKeys = buffer("RadixFused")
        self.radixKeysScratch = buffer("RadixScratch")
        self.radixPayloadScratch = buffer("RadixPayload")
        self.interleavedGaussians = layout.heapAllocations.contains(where: { $0.label == "InterleavedGaussians" }) ? buffer("InterleavedGaussians") : nil
        self.packedGaussiansFused = layout.heapAllocations.contains(where: { $0.label == "PackedGaussiansFused" }) ? buffer("PackedGaussiansFused") : nil

        // Output buffers
        let pixelBytes = layout.pixelCapacity
        guard
            let color = device.makeBuffer(length: pixelBytes * 12, options: .storageModeShared),
            let depth = device.makeBuffer(length: pixelBytes * 4, options: .storageModeShared),
            let alpha = device.makeBuffer(length: pixelBytes * 4, options: .storageModeShared)
        else { fatalError("Failed to allocate output buffers") }
        color.label = "RenderColorOutput"
        depth.label = "RenderDepthOutput"
        alpha.label = "RenderAlphaOutput"
        self.outputBuffers = RenderOutputBuffers(colorOutGPU: color, depthOutGPU: depth, alphaOutGPU: alpha)

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

// MARK: - Renderer

public final class Renderer: @unchecked Sendable {
    public static let shared = Renderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))

    private static let supportedMaxGaussians = 10_000_000

    public let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    private var nextFrameCapturePath: String?
    
    // Encoders
    let tileBoundsEncoder: TileBoundsEncoder
    let coverageEncoder: CoverageEncoder
    let scatterEncoder: ScatterEncoder
    let sortKeyGenEncoder: SortKeyGenEncoder
    let bitonicSortEncoder: BitonicSortEncoder
    let radixSortEncoder: RadixSortEncoder
    let packEncoder: PackEncoder
    let renderEncoder: RenderEncoder
    let projectEncoder: ProjectEncoder
    let dispatchEncoder: DispatchEncoder

    // Fused pipeline encoder (optional - for interleaved data optimization)
    var fusedPipelineEncoder: FusedPipelineEncoder?
    
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

    /// Enable multi-pixel rendering (4x2 pixels per thread, 64 threads per 32x16 tile)
    /// This can improve performance by reducing thread divergence and memory loads
    /// When enabled, caller should use tileWidth=32, tileHeight=16 in RenderParams
    public let useMultiPixelRendering: Bool
    public var isMultiPixelAvailable: Bool { self.renderEncoder.isMultiPixelAvailable }

    /// FlashGS-style precise ellipse-tile intersection
    /// Eliminates tiles in AABB that don't actually intersect the gaussian ellipse
    /// Reduces wasted gaussian-tile pairs, especially for elongated/rotated gaussians
    public let usePreciseIntersection: Bool
    public var isPreciseIntersectionAvailable: Bool {
        self.coverageEncoder.isPreciseAvailable && self.scatterEncoder.isPreciseAvailable
    }

    /// Fused pipeline: interleaved data structures for cache-efficient rendering
    /// Uses single struct reads instead of 5 scattered buffer reads
    public let useFusedPipeline: Bool

    /// Residency set for efficient GPU memory management (macOS 15+ / iOS 18+)
    /// Attached to command queue for minimal CPU overhead
    private var residencySet: (any MTLResidencySet)?

    /// Use heap allocation for frame buffers (reduces TLB misses)
    public let useHeapAllocation: Bool
    private let limits: RendererLimits
    let frameLayout: FrameResourceLayout  // internal for tests
    public var isFusedPipelineAvailable: Bool { self.fusedPipelineEncoder != nil }

    /// Recommended tile size for current rendering mode
    public var recommendedTileWidth: UInt32 { useMultiPixelRendering ? 32 : 16 }
    public var recommendedTileHeight: UInt32 { useMultiPixelRendering ? 16 : 16 }

    public init(
        precision: Precision = .float32,
        useIndirectBitonic: Bool = false,
        sortAlgorithm: SortAlgorithm = .radix,
        useMultiPixelRendering: Bool = false,
        usePreciseIntersection: Bool = false,
        useFusedPipeline: Bool = false,
        useHeapAllocation: Bool = false,  // Disabled by default - placement heap has issues
        limits: RendererLimits = RendererLimits(maxGaussians: 1_000_000, maxWidth: 2048, maxHeight: 2048)
    ) {
        precondition(limits.maxGaussians <= Renderer.supportedMaxGaussians, "Renderer supports up to 10,000,000 gaussians; requested \(limits.maxGaussians)")
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
            self.coverageEncoder = try CoverageEncoder(device: device, library: library)
            self.scatterEncoder = try ScatterEncoder(device: device, library: library)
            self.sortKeyGenEncoder = try SortKeyGenEncoder(device: device, library: library)
            self.bitonicSortEncoder = try BitonicSortEncoder(device: device, library: library, useIndirect: useIndirectBitonic)
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
                radixGrainSize: UInt32(self.radixSortEncoder.grainSize)
            )
            self.dispatchEncoder = try DispatchEncoder(device: device, library: library, config: config)

            // Fused pipeline encoder (optional optimization)
            if useFusedPipeline {
                self.fusedPipelineEncoder = try FusedPipelineEncoder(device: device, library: library)
            }

        } catch {
            fatalError("Failed to load library or initialize encoders: \(error)")
        }
        self.precision = precision
        self.sortAlgorithm = sortAlgorithm
        self.useMultiPixelRendering = useMultiPixelRendering
        self.usePreciseIntersection = usePreciseIntersection
        self.useFusedPipeline = useFusedPipeline
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
        self.frameLayout = Renderer.computeLayout(
            limits: limits,
            sortAlgorithm: sortAlgorithm,
            useFusedPipeline: useFusedPipeline,
            precision: .float32,
            device: device,
            radixSortEncoder: self.radixSortEncoder
        )

        // Initialize frame slots with fully allocated resources
        for _ in 0..<maxInFlightFrames {
            let frame = FrameResources(
                device: device,
                layout: self.frameLayout,
                residencySetProvider: { [weak self] in self?.residencySet },
                useHeap: useHeapAllocation
            )
            self.frameResources.append(frame)
            self.frameInUse.append(false)
        }

        // Preallocate shared CPU-visible caches up to the configured limits
        let shared: MTLResourceOptions = .storageModeShared
        func makeShared(_ length: Int, _ label: String) -> MTLBuffer {
            guard let buf = device.makeBuffer(length: length, options: shared) else {
                fatalError("Failed to allocate shared buffer \(label)")
            }
            buf.label = label
            return buf
        }

        let g = limits.maxGaussians
        self.worldPositionsCache = makeShared(g * MemoryLayout<SIMD3<Float>>.stride, "WorldPositions")
        self.worldScalesCache = makeShared(g * MemoryLayout<SIMD3<Float>>.stride, "WorldScales")
        self.worldRotationsCache = makeShared(g * MemoryLayout<SIMD4<Float>>.stride, "WorldRotations")
        // SH coeffs worst-case: 9 coefficients (l=2) *3 channels -> 27 floats
        self.worldHarmonicsCache = makeShared(g * MemoryLayout<Float>.stride * 27, "WorldHarmonics")
        self.worldOpacitiesCache = makeShared(g * MemoryLayout<Float>.stride, "WorldOpacities")

        self.gaussianMeansCache = makeShared(g * 8, "GaussianMeans")
        self.gaussianRadiiCache = makeShared(g * 4, "GaussianRadii")
        self.gaussianMaskCache = makeShared(g, "GaussianMask")
        self.gaussianDepthsCache = makeShared(g * 4, "GaussianDepths")
        self.gaussianConicsCache = makeShared(g * 16, "GaussianConics")
        self.gaussianColorsCache = makeShared(g * 12, "GaussianColors")
        self.gaussianOpacitiesCache = makeShared(g * 4, "GaussianOpacities")

        self.gaussianMeansHalfCache = makeShared(g * 4, "GaussianMeansHalf")
        self.gaussianDepthsHalfCache = makeShared(g * 2, "GaussianDepthsHalf")
        self.gaussianConicsHalfCache = makeShared(g * 8, "GaussianConicsHalf")
        self.gaussianColorsHalfCache = makeShared(g * 6, "GaussianColorsHalf")
        self.gaussianOpacitiesHalfCache = makeShared(g * 2, "GaussianOpacitiesHalf")
    }
    
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
        let gaussianCount = limits.maxGaussians
        let tileCount = limits.maxTileCount
        // Clamp maxAssignments to a realistic bound to avoid huge heaps when tileCount * maxPerTile
        // is unreasonably large. But we must always allocate at least tileCount * maxPerTile since
        // that's the minimum needed by any valid render.
        let tileCapacity = tileCount * limits.maxPerTile
        let gaussianCapacity = gaussianCount * 8  // each Gaussian in ~8 tiles max
        // Use the smaller of unclamped and gaussianCapacity, but never below tileCapacity
        let unclamped = max(gaussianCount, tileCapacity)
        let maxAssignments = max(tileCapacity, min(unclamped, gaussianCapacity))
        let paddedAssignments: Int = {
            if sortAlgorithm == .radix {
                let block = radixSortEncoder.blockSize * radixSortEncoder.grainSize
                return ((maxAssignments + block - 1) / block) * block
            } else {
                var value = 1
                while value < maxAssignments { value <<= 1 }
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
        let groups = (gaussianCount + elementsPerGroup - 1) / elementsPerGroup

        var heapAllocations: [(label: String, length: Int, options: MTLResourceOptions)] = []
        func add(_ label: String, _ length: Int, _ opts: MTLResourceOptions = .storageModePrivate) {
            heapAllocations.append((label, max(1, length), opts))
        }

        add("Bounds", gaussianCount * MemoryLayout<SIMD4<Int32>>.stride)
        add("Coverage", gaussianCount * MemoryLayout<UInt32>.stride)
        add("Offsets", (gaussianCount + 1) * MemoryLayout<UInt32>.stride)
        add("PartialSums", max(groups, 1) * MemoryLayout<UInt32>.stride)
        add("ScatterDispatch", 3 * MemoryLayout<UInt32>.stride)
        add("TileIndices", paddedAssignments * MemoryLayout<Int32>.stride)
        add("TileIds", paddedAssignments * MemoryLayout<Int32>.stride)

        add("OrderedHeaders", tileCount * MemoryLayout<GaussianHeader>.stride)
        add("PackedMeans", maxAssignments * strideForMeans())
        add("PackedConics", maxAssignments * strideForConics())
        add("PackedColors", maxAssignments * strideForColors())
        add("PackedOpacities", maxAssignments * strideForOpacities())
        add("PackedDepths", maxAssignments * strideForDepths())
        add("ActiveTileIndices", tileCount * MemoryLayout<UInt32>.stride)

        add("SortKeys", paddedAssignments * MemoryLayout<SIMD2<UInt32>>.stride)
        add("SortedIndices", paddedAssignments * MemoryLayout<Int32>.stride)

        let valuesPerGroup = radixSortEncoder.blockSize * radixSortEncoder.grainSize
        let gridSize = max(1, (paddedAssignments + valuesPerGroup - 1) / valuesPerGroup)
        let histogramCount = gridSize * radixSortEncoder.radix
        add("RadixHist", histogramCount * MemoryLayout<UInt32>.stride)
        add("RadixBlockSums", gridSize * MemoryLayout<UInt32>.stride)
        add("RadixScanned", histogramCount * MemoryLayout<UInt32>.stride)
        add("RadixFused", paddedAssignments * MemoryLayout<UInt64>.stride)
        add("RadixScratch", paddedAssignments * MemoryLayout<UInt64>.stride)
        add("RadixPayload", paddedAssignments * MemoryLayout<UInt32>.stride)

        if useFusedPipeline {
            add("InterleavedGaussians", gaussianCount * strideForInterleaved())
            add("PackedGaussiansFused", maxAssignments * strideForInterleaved())
        }

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
            maxAssignments: maxAssignments,
            paddedAssignments: paddedAssignments,
            heapAllocations: heapAllocations,
            sharedAllocations: sharedAllocations,
            heapSize: heapSize(for: heapAllocations),
            pixelCapacity: limits.maxWidth * limits.maxHeight,
            precisionStride: strideForPrecision
        )
    }

    // MARK: - Render Methods
    
    func render(
        headersPtr: UnsafePointer<GaussianHeader>,
        headerCount: Int,
        meansPtr: UnsafePointer<Float>,
        conicsPtr: UnsafePointer<Float>,
        colorsPtr: UnsafePointer<Float>,
        opacityPtr: UnsafePointer<Float>,
        depthsPtr: UnsafePointer<Float>,
        colorOutPtr: UnsafeMutablePointer<Float>?,
        depthOutPtr: UnsafeMutablePointer<Float>?,
        alphaOutPtr: UnsafeMutablePointer<Float>?,
        params: RenderParams
    ) -> Int32 {
        guard headerCount > 0 else {
            return 0
        }
        
        // Helper to create buffers from pointers (Shared mode for CPU->GPU)
        func makeBuf<T>(_ ptr: UnsafePointer<T>, _ count: Int) -> MTLBuffer? {
            self.device.makeBuffer(bytes: ptr, length: count * MemoryLayout<T>.stride, options: .storageModeShared)
        }
        
        let headers = UnsafeBufferPointer(start: headersPtr, count: headerCount)
        guard let last = headers.last else { return 0 }
        let totalAssignments = Int(last.offset + last.count)
        guard totalAssignments <= limits.maxGaussians else { return -5 }
        guard self.validateLimits(gaussianCount: totalAssignments, params: params) else { return -5 }
        
        guard
            let headersBuf = makeBuf(headersPtr, headerCount),
            let meansBuf = makeBuf(meansPtr, totalAssignments * 2),
            let conicsBuf = makeBuf(conicsPtr, totalAssignments * 4),
            let colorsBuf = makeBuf(colorsPtr, totalAssignments * 3),
            let opacityBuf = makeBuf(opacityPtr, totalAssignments),
            let depthsBuf = makeBuf(depthsPtr, totalAssignments)
        else { return -1 }
        
        // Generate Active Tiles (All tiles)
        guard let activeTileIndices = self.device.makeBuffer(length: headerCount * 4, options: .storageModeShared),
              let activeTileCount = self.device.makeBuffer(length: 4, options: .storageModeShared)
        else { return -2 }
        
        let activeIndicesPtr = activeTileIndices.contents().bindMemory(to: UInt32.self, capacity: headerCount)
        for i in 0..<headerCount {
            activeIndicesPtr[i] = UInt32(i)
        }
        activeTileCount.contents().storeBytes(of: UInt32(headerCount), as: UInt32.self)
        
        let ordered = OrderedGaussianBuffers(
            headers: headersBuf,
            means: meansBuf,
            conics: conicsBuf,
            colors: colorsBuf,
            opacities: opacityBuf,
            depths: depthsBuf,
            tileCount: headerCount,
            activeTileIndices: activeTileIndices,
            activeTileCount: activeTileCount,
            precision: .float32,
            packedGaussiansFused: nil
        )
        
        guard let commandBuffer = self.queue.makeCommandBuffer() else { return -1 }
        return self.encodeAndRunRender(
            commandBuffer: commandBuffer,
            ordered: ordered,
            colorOutPtr: colorOutPtr,
            depthOutPtr: depthOutPtr,
            alphaOutPtr: alphaOutPtr,
            params: params,
            gaussianCount: totalAssignments
        )
    }
    
    func renderRaw(
        gaussianCount: Int,
        meansPtr: UnsafePointer<Float>,
        conicsPtr: UnsafePointer<Float>,
        colorsPtr: UnsafePointer<Float>,
        opacityPtr: UnsafePointer<Float>,
        depthsPtr: UnsafePointer<Float>,
        radiiPtr: UnsafePointer<Float>,
        colorOutPtr: UnsafeMutablePointer<Float>?,
        depthOutPtr: UnsafeMutablePointer<Float>?,
        alphaOutPtr: UnsafeMutablePointer<Float>?,
        params: RenderParams
    ) -> Int32 {
        let pixelCount = Int(params.width * params.height)
        if gaussianCount == 0 || pixelCount == 0 {
            return 0
        }
        
        let count = gaussianCount
        guard self.validateLimits(gaussianCount: count, params: params) else { return -5 }
        
        // Acquire a frame for rendering
        guard let (frame, slotIndex) = self.acquireFrame(width: Int(params.width), height: Int(params.height)) else { return -5 }

        guard
            let meansBuffer = self.gaussianMeansCache,
            let conicsBuffer = self.gaussianConicsCache,
            let colorsBuffer = self.gaussianColorsCache,
            let opacityBuffer = self.gaussianOpacitiesCache,
            let depthsBuffer = self.gaussianDepthsCache,
            let radiiBuffer = self.gaussianRadiiCache,
            let maskBuffer = self.gaussianMaskCache
        else {
            self.releaseFrame(index: slotIndex)
            return -2
        }

        memcpy(meansBuffer.contents(), meansPtr, count * 2 * MemoryLayout<Float>.stride)
        memcpy(conicsBuffer.contents(), conicsPtr, count * 4 * MemoryLayout<Float>.stride)
        memcpy(colorsBuffer.contents(), colorsPtr, count * 3 * MemoryLayout<Float>.stride)
        memcpy(opacityBuffer.contents(), opacityPtr, count * MemoryLayout<Float>.stride)
        memcpy(depthsBuffer.contents(), depthsPtr, count * MemoryLayout<Float>.stride)
        memcpy(radiiBuffer.contents(), radiiPtr, count * MemoryLayout<Float>.stride)
        memset(maskBuffer.contents(), 1, count)
        
        let inputs = GaussianInputBuffers(
            means: meansBuffer,
            radii: radiiBuffer,
            mask: maskBuffer,
            depths: depthsBuffer,
            conics: conicsBuffer,
            colors: colorsBuffer,
            opacities: opacityBuffer
        )
        
        guard let commandBuffer = self.queue.makeCommandBuffer() else { 
            self.releaseFrame(index: slotIndex)
            return -1 
        }
        
        let estimatedAssignments = self.estimateAssignmentCapacity(
            gaussianCount: count,
            meansPtr: meansPtr,
            radiiPtr: radiiPtr,
            params: params
        )
        
        // renderRaw always uses float32 precision because inputs are float32 pointers
        guard let assignment = self.buildTileAssignmentsGPU(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            gaussianBuffers: inputs,
            params: params,
            frame: frame,
            estimatedAssignments: estimatedAssignments,
            precision: .float32
        ) else {
            self.releaseFrame(index: slotIndex)
            return -3
        }

        guard let ordered = self.buildOrderedGaussians(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            assignment: assignment,
            gaussianBuffers: inputs,
            params: params,
            frame: frame,
            precision: .float32  // Always float32 for renderRaw (float input buffers)
        ) else {
            self.releaseFrame(index: slotIndex)
            return -4
        }

        return self.encodeAndRunRenderWithFrame(
            commandBuffer: commandBuffer,
            ordered: ordered,
            frame: frame,
            slotIndex: slotIndex,
            colorOutPtr: colorOutPtr,
            depthOutPtr: depthOutPtr,
            alphaOutPtr: alphaOutPtr,
            params: params,
            gaussianCount: count
        )
    }
    
    func renderWorld(
        gaussianCount: Int,
        worldBuffers: WorldGaussianBuffers,
        cameraUniforms: CameraUniformsSwift,
        colorOutPtr: UnsafeMutablePointer<Float>?,
        depthOutPtr: UnsafeMutablePointer<Float>?,
        alphaOutPtr: UnsafeMutablePointer<Float>?,
        radiiOutPtr: UnsafeMutablePointer<Float>?,
        maskOutPtr: UnsafeMutablePointer<UInt8>?,
        params: RenderParams
    ) -> Int32 {
         // Metal Capture Start
         if let capturePath = self.nextFrameCapturePath {
             let descriptor = MTLCaptureDescriptor()
             descriptor.captureObject = self.device
             descriptor.destination = .gpuTraceDocument
             descriptor.outputURL = URL(fileURLWithPath: capturePath)
             
             let manager = MTLCaptureManager.shared()
             try? manager.startCapture(with: descriptor)
             self.nextFrameCapturePath = nil
        }

        let count = gaussianCount
        guard self.validateLimits(gaussianCount: count, params: params) else { return -5 }
        
        // Acquire a frame for rendering
        guard let (frame, slotIndex) = self.acquireFrame(width: Int(params.width), height: Int(params.height)) else { return -5 }

        guard let gaussianBuffers = self.prepareGaussianBuffers(count: count) else {
            self.releaseFrame(index: slotIndex)
            return -2
        }
        
        guard let commandBuffer = self.queue.makeCommandBuffer() else {
            self.releaseFrame(index: slotIndex)
            return -1
        }

        self.projectEncoder.encodeForRender(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            worldBuffers: worldBuffers,
            cameraUniforms: cameraUniforms,
            gaussianBuffers: gaussianBuffers,
            precision: self.effectivePrecision
        )
        
        guard let assignment = self.buildTileAssignmentsGPU(commandBuffer: commandBuffer, gaussianCount: count, gaussianBuffers: gaussianBuffers, params: params, frame: frame) else {
            self.releaseFrame(index: slotIndex)
            return -3
        }
        
        guard let ordered = self.buildOrderedGaussians(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            assignment: assignment,
            gaussianBuffers: gaussianBuffers,
            params: params,
            frame: frame,
            precision: self.effectivePrecision
        ) else {
            self.releaseFrame(index: slotIndex)
            return -4
        }
        
        let result = self.encodeAndRunRenderWithFrame(
            commandBuffer: commandBuffer,
            ordered: ordered,
            frame: frame,
            slotIndex: slotIndex,
            colorOutPtr: colorOutPtr,
            depthOutPtr: depthOutPtr,
            alphaOutPtr: alphaOutPtr,
            params: params,
            gaussianCount: count
        )
        
        if MTLCaptureManager.shared().isCapturing {
            MTLCaptureManager.shared().stopCapture()
        }
        
        return result
    }
    public func encodeRender(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        worldBuffers: WorldGaussianBuffers,
        cameraUniforms: CameraUniformsSwift,
        params: RenderParams
    ) -> RenderOutputBuffers? {
        guard self.validateLimits(gaussianCount: gaussianCount, params: params) else { return nil }
        guard let (frame, slotIndex) = self.acquireFrame(width: Int(params.width), height: Int(params.height)) else {
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

        self.projectEncoder.encodeForRender(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            worldBuffers: worldBuffers,
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
        worldBuffers: WorldGaussianBuffers,
        cameraUniforms: CameraUniformsSwift,
        params: RenderParams
    ) -> RenderOutputTextures? {
        // validateLimits now crashes with descriptive message on failure
        guard self.validateLimits(gaussianCount: gaussianCount, params: params) else {
            fatalError("[encodeRenderToTextures] validateLimits failed - should have crashed with details")
        }

        guard let (frame, slotIndex) = self.acquireFrame(width: Int(params.width), height: Int(params.height)) else {
            fatalError("[encodeRenderToTextures] acquireFrame failed for \(params.width)x\(params.height)")
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

        self.projectEncoder.encodeForRender(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            worldBuffers: worldBuffers,
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

        // Select render path: fused > multi-pixel > standard
        if self.useFusedPipeline, let fusedEncoder = self.fusedPipelineEncoder, ordered.packedGaussiansFused != nil {
            // Fused render: single-struct reads for cache efficiency
            fusedEncoder.encodeCompleteFusedRender(
                commandBuffer: commandBuffer,
                orderedBuffers: ordered,
                outputTextures: textures,
                params: params,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                precision: self.effectivePrecision
            )
        } else if self.useMultiPixelRendering && self.renderEncoder.isMultiPixelAvailable {
            _ = self.renderEncoder.encodeMultiPixel(
                commandBuffer: commandBuffer,
                orderedBuffers: ordered,
                outputTextures: textures,
                params: params,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
            )
        } else {
            self.renderEncoder.encodeDirect(
                commandBuffer: commandBuffer,
                orderedBuffers: ordered,
                outputTextures: textures,
                params: params,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                precision: self.effectivePrecision
            )
        }

        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.releaseFrame(index: slotIndex)
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
        }

        return textures
    }

    /// New API for half-precision input rendering directly to textures.
    /// This is the fastest path for GSViewer: half16 world data → texture output.
    public func encodeRenderToTextureHalf(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        worldBuffers: WorldGaussianBuffersHalf,
        cameraUniforms: CameraUniformsSwift,
        params: RenderParams
    ) -> RenderOutputTextures? {
        // validateLimits now crashes with descriptive message on failure
        guard self.validateLimits(gaussianCount: gaussianCount, params: params) else {
            fatalError("[encodeRenderToTextureHalf] validateLimits failed - should have crashed with details")
        }

        guard let (frame, slotIndex) = self.acquireFrame(width: Int(params.width), height: Int(params.height)) else {
            fatalError("[encodeRenderToTextureHalf] acquireFrame failed for \(params.width)x\(params.height)")
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

        // Project from half world data to half gaussian data
        self.projectEncoder.encodeForRenderHalf(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            worldBuffers: worldBuffers,
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

        // Select render path: fused > multi-pixel > standard
        if self.useFusedPipeline, let fusedEncoder = self.fusedPipelineEncoder, ordered.packedGaussiansFused != nil {
            fusedEncoder.encodeCompleteFusedRender(
                commandBuffer: commandBuffer,
                orderedBuffers: ordered,
                outputTextures: textures,
                params: params,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                precision: .float16
            )
        } else if self.useMultiPixelRendering && self.renderEncoder.isMultiPixelAvailable {
            _ = self.renderEncoder.encodeMultiPixel(
                commandBuffer: commandBuffer,
                orderedBuffers: ordered,
                outputTextures: textures,
                params: params,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
            )
        } else {
            self.renderEncoder.encodeDirect(
                commandBuffer: commandBuffer,
                orderedBuffers: ordered,
                outputTextures: textures,
                params: params,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                precision: .float16
            )
        }

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
        worldBuffers: WorldGaussianBuffersHalf,
        cameraUniforms: CameraUniformsSwift,
        params: RenderParams,
        targetColor: MTLTexture,
        targetDepth: MTLTexture?,
        targetAlpha: MTLTexture?
    ) -> Bool {
        guard self.validateLimits(gaussianCount: gaussianCount, params: params) else {
            return false
        }
        guard let (frame, slotIndex) = self.acquireFrame(width: Int(params.width), height: Int(params.height)) else {
            return false
        }

        guard let gaussianBuffers = self.prepareGaussianBuffersHalf(count: gaussianCount) else {
            self.releaseFrame(index: slotIndex)
            return false
        }

        self.projectEncoder.encodeForRenderHalf(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            worldBuffers: worldBuffers,
            cameraUniforms: cameraUniforms,
            gaussianBuffers: gaussianBuffers
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

        // Select render path: fused > multi-pixel > standard
        if self.useFusedPipeline, let fusedEncoder = self.fusedPipelineEncoder, ordered.packedGaussiansFused != nil {
            fusedEncoder.encodeCompleteFusedRender(
                commandBuffer: commandBuffer,
                orderedBuffers: ordered,
                outputTextures: outputTextures,
                params: params,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                precision: .float16
            )
        } else if self.useMultiPixelRendering && self.renderEncoder.isMultiPixelAvailable {
            _ = self.renderEncoder.encodeMultiPixel(
                commandBuffer: commandBuffer,
                orderedBuffers: ordered,
                outputTextures: outputTextures,
                params: params,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
            )
        } else {
            self.renderEncoder.encodeDirect(
                commandBuffer: commandBuffer,
                orderedBuffers: ordered,
                outputTextures: outputTextures,
                params: params,
                dispatchArgs: dispatchArgs,
                dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                precision: .float16
            )
        }

        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.releaseFrame(index: slotIndex)
        }

        return true
    }

    // MARK: - Helper Methods
    
    private func estimateAssignmentCapacity(
        gaussianCount: Int,
        meansPtr: UnsafePointer<Float>,
        radiiPtr: UnsafePointer<Float>,
        params: RenderParams
    ) -> Int {
        guard gaussianCount > 0 else { return 1 }
        let tilesX = Int(params.tilesX)
        let tilesY = Int(params.tilesY)
        let tileW = Float(params.tileWidth)
        let tileH = Float(params.tileHeight)
        var total = 0
        for i in 0..<gaussianCount {
            let mx = meansPtr[i * 2 + 0]
            let my = meansPtr[i * 2 + 1]
            let r = radiiPtr[i]
            let minX = max(0, Int(floor((mx - r) / tileW)))
            let maxX = min(tilesX - 1, Int(ceil((mx + r) / tileW)))
            let minY = max(0, Int(floor((my - r) / tileH)))
            let maxY = min(tilesY - 1, Int(ceil((my + r) / tileH)))
            if maxX < minX || maxY < minY { continue }
            let spanX = maxX - minX + 1
            let spanY = maxY - minY + 1
            total += spanX * spanY
        }
        // Clamp to a reasonable upper bound to avoid runaway allocations.
        let conservativeCap = Int(params.tilesX * params.tilesY) * max(1, Int(params.maxPerTile == 0 ? UInt32(gaussianCount) : params.maxPerTile))
        let estimate = min(total, conservativeCap)
        return max(estimate, gaussianCount)
    }

    private func validateLimits(gaussianCount: Int, params: RenderParams) -> Bool {
        let tileCount = Int(params.tilesX * params.tilesY)

        // CRASH on validation failures - silent returns cause black screens!
        precondition(gaussianCount <= limits.maxGaussians,
            "gaussianCount (\(gaussianCount)) exceeds limits.maxGaussians (\(limits.maxGaussians))")
        precondition(tileCount <= limits.maxTileCount,
            "tileCount (\(tileCount)) exceeds limits.maxTileCount (\(limits.maxTileCount))")
        precondition(Int(params.tileWidth) == limits.tileWidth,
            "params.tileWidth (\(params.tileWidth)) != limits.tileWidth (\(limits.tileWidth)) - did you forget to set tileWidth in RendererLimits for multi-pixel rendering?")
        precondition(Int(params.tileHeight) == limits.tileHeight,
            "params.tileHeight (\(params.tileHeight)) != limits.tileHeight (\(limits.tileHeight)) - did you forget to set tileHeight in RendererLimits for multi-pixel rendering?")
        precondition(Int(params.width) <= limits.maxWidth,
            "params.width (\(params.width)) exceeds limits.maxWidth (\(limits.maxWidth))")
        precondition(Int(params.height) <= limits.maxHeight,
            "params.height (\(params.height)) exceeds limits.maxHeight (\(limits.maxHeight))")

        return true
    }
    
    internal func buildTileAssignmentsGPU(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        gaussianBuffers: GaussianInputBuffers,
        params: RenderParams,
        frame: FrameResources,
        estimatedAssignments: Int? = nil,
        precision: Precision = .float32
    ) -> TileAssignmentBuffers? {
        // CRASH on failures - silent returns cause black screens!
        precondition(gaussianCount <= limits.maxGaussians,
            "[buildTileAssignmentsGPU] gaussianCount (\(gaussianCount)) > limits.maxGaussians (\(limits.maxGaussians))")

        let tileCount = Int(params.tilesX * params.tilesY)
        precondition(tileCount <= limits.maxTileCount,
            "[buildTileAssignmentsGPU] tileCount (\(tileCount)) > limits.maxTileCount (\(limits.maxTileCount))")

        let perTileLimit = (params.maxPerTile == 0) ? UInt32(limits.maxPerTile) : min(params.maxPerTile, UInt32(limits.maxPerTile))
        let baseCapacity = max(tileCount * Int(perTileLimit), gaussianCount)

        precondition(baseCapacity <= frame.tileAssignmentMaxAssignments,
            "[buildTileAssignmentsGPU] baseCapacity (\(baseCapacity)) > frame.tileAssignmentMaxAssignments (\(frame.tileAssignmentMaxAssignments)) - increase limits.maxPerTile or reduce tile count")

        resetTileBuilderState(commandBuffer: commandBuffer, frame: frame)

        self.tileBoundsEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianBuffers: gaussianBuffers,
            boundsBuffer: frame.boundsBuffer,
            params: params,
            gaussianCount: gaussianCount,
            precision: precision
        )
        
        // Coverage: count tiles per gaussian
        if self.usePreciseIntersection && self.isPreciseIntersectionAvailable {
            // FlashGS precise ellipse-tile intersection
            self.coverageEncoder.encodePrecise(
                commandBuffer: commandBuffer,
                gaussianCount: gaussianCount,
                tileWidth: Int(params.tileWidth),
                tileHeight: Int(params.tileHeight),
                boundsBuffer: frame.boundsBuffer,
                opacitiesBuffer: gaussianBuffers.opacities,
                meansBuffer: gaussianBuffers.means,
                conicsBuffer: gaussianBuffers.conics,
                coverageBuffer: frame.coverageBuffer,
                offsetsBuffer: frame.offsetsBuffer,
                partialSumsBuffer: frame.partialSumsBuffer,
                tileAssignmentHeader: frame.tileAssignmentHeader,
                precision: precision
            )
        } else {
            self.coverageEncoder.encode(
                commandBuffer: commandBuffer,
                gaussianCount: gaussianCount,
                boundsBuffer: frame.boundsBuffer,
                opacitiesBuffer: gaussianBuffers.opacities,
                coverageBuffer: frame.coverageBuffer,
                offsetsBuffer: frame.offsetsBuffer,
                partialSumsBuffer: frame.partialSumsBuffer,
                tileAssignmentHeader: frame.tileAssignmentHeader,
                precision: precision
            )
        }

        // Scatter: write gaussian-tile pairs
        if self.usePreciseIntersection && self.isPreciseIntersectionAvailable {
            // FlashGS precise scatter (parallel: one threadgroup per gaussian)
            self.scatterEncoder.encodePrecise(
                commandBuffer: commandBuffer,
                gaussianCount: gaussianCount,
                tilesX: Int(params.tilesX),
                tileWidth: Int(params.tileWidth),
                tileHeight: Int(params.tileHeight),
                offsetsBuffer: frame.offsetsBuffer,
                dispatchBuffer: frame.scatterDispatchBuffer,
                boundsBuffer: frame.boundsBuffer,
                tileIndicesBuffer: frame.tileIndices,
                tileIdsBuffer: frame.tileIds,
                tileAssignmentHeader: frame.tileAssignmentHeader,
                meansBuffer: gaussianBuffers.means,
                conicsBuffer: gaussianBuffers.conics,
                opacitiesBuffer: gaussianBuffers.opacities,
                precision: precision
            )
        } else {
            // Load-balanced scatter (binary search) for better GPU utilization
            self.scatterEncoder.encodeBalanced(
                commandBuffer: commandBuffer,
                gaussianCount: gaussianCount,
                tilesX: Int(params.tilesX),
                offsetsBuffer: frame.offsetsBuffer,
                dispatchBuffer: frame.scatterDispatchBuffer,
                boundsBuffer: frame.boundsBuffer,
                tileIndicesBuffer: frame.tileIndices,
                tileIdsBuffer: frame.tileIds,
                tileAssignmentHeader: frame.tileAssignmentHeader
            )
        }
        
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
            dispatchArgs: dispatchArgs
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
                histogram: frame.radixHistogram,
                blockSums: frame.radixBlockSums,
                scannedHistogram: frame.radixScannedHistogram,
                fusedKeys: frame.radixFusedKeys,
                scratchKeys: frame.radixKeysScratch,
                scratchPayload: frame.radixPayloadScratch
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
                              self.useFusedPipeline &&
                              self.fusedPipelineEncoder != nil &&
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

            let fusedEncoder = self.fusedPipelineEncoder!
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
        self.renderEncoder.encode(
            commandBuffer: commandBuffer,
            orderedBuffers: ordered,
            outputBuffers: frame.outputBuffers,
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
        if let p = colorOutPtr { memcpy(p, frame.outputBuffers.colorOutGPU.contents(), pixelCount * 12) }
        if let p = depthOutPtr { memcpy(p, frame.outputBuffers.depthOutGPU.contents(), pixelCount * 4) }
        if let p = alphaOutPtr { memcpy(p, frame.outputBuffers.alphaOutGPU.contents(), pixelCount * 4) }

        return 0
    }
    
    // MARK: - Resource Management

    func prepareWorldBuffers(
        count: Int,
        meansPtr: UnsafePointer<Float>,
        scalesPtr: UnsafePointer<Float>,
        rotationsPtr: UnsafePointer<Float>,
        harmonicsPtr: UnsafePointer<Float>,
        opacitiesPtr: UnsafePointer<Float>,
        shComponents: Int
    ) -> WorldGaussianBuffers? {
        guard count <= limits.maxGaussians else { return nil }
        guard
            let positions = self.worldPositionsCache,
            let scales = self.worldScalesCache,
            let rotations = self.worldRotationsCache,
            let harmonics = self.worldHarmonicsCache,
            let opacities = self.worldOpacitiesCache
        else { return nil }

        let positionDest = positions.contents().bindMemory(to: SIMD3<Float>.self, capacity: count)
        let scalesDest = scales.contents().bindMemory(to: SIMD3<Float>.self, capacity: count)
        for i in 0..<count {
            let base = i * 3
            positionDest[i] = SIMD3(meansPtr[base + 0], meansPtr[base + 1], meansPtr[base + 2])
            scalesDest[i] = SIMD3(scalesPtr[base + 0], scalesPtr[base + 1], scalesPtr[base + 2])
        }

        let rotationLength = count * MemoryLayout<SIMD4<Float>>.stride
        let coeffs = max(shComponents, 0)
        let harmonicsLength = count * MemoryLayout<Float>.stride * (coeffs == 0 ? 3 : coeffs * 3)
        let opacityLength = count * MemoryLayout<Float>.stride
        guard harmonicsLength <= harmonics.length else { return nil }

        memcpy(rotations.contents(), rotationsPtr, rotationLength)
        memcpy(harmonics.contents(), harmonicsPtr, harmonicsLength)
        memcpy(opacities.contents(), opacitiesPtr, opacityLength)

        return WorldGaussianBuffers(
            positions: positions,
            scales: scales,
            rotations: rotations,
            harmonics: harmonics,
            opacities: opacities,
            shComponents: shComponents
        )
    }
    
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
        // GPU-only reset: use blit encoder to zero the counters
        // TileHeader layout: [totalAssignments: UInt32, maxAssignments: UInt32, paddedCount: UInt32, overflow: UInt32]
        // We only need to reset totalAssignments (offset 0) and overflow (offset 12)
        // maxAssignments and paddedCount are constants set once at init
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            fatalError("[resetTileBuilderState] Failed to create blit encoder")
        }
        blit.label = "ResetTileBuilderState"
        // Zero totalAssignments (first 4 bytes)
        blit.fill(buffer: frame.tileAssignmentHeader, range: 0..<4, value: 0)
        // Zero overflow (bytes 12-16)
        blit.fill(buffer: frame.tileAssignmentHeader, range: 12..<16, value: 0)
        blit.endEncoding()
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

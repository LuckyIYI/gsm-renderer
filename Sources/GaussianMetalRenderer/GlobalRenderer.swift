import Foundation
import RendererTypes
@preconcurrency import Metal
import simd

// MARK: - Internal Limits (computed from RendererConfig)

struct RendererLimits: Sendable {
    let maxGaussians: Int
    let maxWidth: Int
    let maxHeight: Int
    let tileWidth: Int
    let tileHeight: Int
    let maxPerTile: Int

    var tilesX: Int { (self.maxWidth + self.tileWidth - 1) / self.tileWidth }
    var tilesY: Int { (self.maxHeight + self.tileHeight - 1) / self.tileHeight }
    var maxTileCount: Int { max(1, self.tilesX * self.tilesY) }

    init(from config: RendererConfig, tileWidth: Int = 32, tileHeight: Int = 16, maxPerTile: Int = 2048) {
        self.maxGaussians = max(1, config.maxGaussians)
        self.maxWidth = max(1, config.maxWidth)
        self.maxHeight = max(1, config.maxHeight)
        self.tileWidth = max(1, tileWidth)
        self.tileHeight = max(1, tileHeight)
        self.maxPerTile = max(1, maxPerTile)
    }

    func buildParams(from frame: FrameParams) -> RenderParams {
        RenderParams(
            width: UInt32(self.maxWidth),
            height: UInt32(self.maxHeight),
            tileWidth: UInt32(self.tileWidth),
            tileHeight: UInt32(self.tileHeight),
            tilesX: UInt32(self.tilesX),
            tilesY: UInt32(self.tilesY),
            maxPerTile: UInt32(self.maxPerTile),
            whiteBackground: frame.whiteBackground ? 1 : 0,
            activeTileCount: 0,
            gaussianCount: UInt32(frame.gaussianCount)
        )
    }
}

// MARK: - Frame Resources

struct FrameResourceLayout {
    let limits: RendererLimits
    /// Maximum tile assignments capacity (tileCount * maxPerTile, clamped by gaussianCapacity)
    let maxAssignmentCapacity: Int
    /// Padded capacity for radix sort alignment (1024-aligned)
    let paddedCapacity: Int
    let bufferAllocations: [(label: String, length: Int, options: MTLResourceOptions)]
    let pixelCapacity: Int
    let precisionStride: Int
}

final class FrameResources {
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

    // Radix Sort Buffers
    let radixHistogram: MTLBuffer?
    let radixBlockSums: MTLBuffer?
    let radixScannedHistogram: MTLBuffer?
    let radixKeysScratch: MTLBuffer?
    let radixPayloadScratch: MTLBuffer?

    // Interleaved gaussian data for cache efficiency
    let interleavedGaussians: MTLBuffer?

    // Two-pass tile assignment scratch buffer (prefix sum block sums)
    let tileAssignBlockSums: MTLBuffer?

    // Indirect dispatch buffers (for compacted visible gaussians)
    let visibleIndices: MTLBuffer // Compacted visible gaussian indices
    let visibleCount: MTLBuffer // Atomic counter for visible count
    let tileAssignDispatchArgs: MTLBuffer // Indirect dispatch args for tile count/scatter

    // Dispatch Args (Per-frame to avoid race on indirect dispatch)
    let dispatchArgs: MTLBuffer

    // Output Buffers (nil when textureOnly mode)
    var outputBuffers: RenderOutputBuffers?
    // Output Textures (Alternative)
    var outputTextures: RenderOutputTextures

    let device: MTLDevice

    init(device: MTLDevice, layout: FrameResourceLayout, textureOnly: Bool = false) {
        self.device = device
        self.tileAssignmentMaxAssignments = layout.maxAssignmentCapacity
        self.tileAssignmentPaddedCapacity = layout.paddedCapacity

        self.dispatchArgs = try! device.makeBuffer(count: 1024, type: UInt32.self, options: .storageModePrivate, label: "FrameDispatchArgs")

        // Buffer allocation helper
        var bufferMap: [String: MTLBuffer] = [:]
        for req in layout.bufferAllocations {
            guard let buf = device.makeBuffer(length: req.length, options: req.options) else {
                fatalError("Failed to allocate buffer \(req.label)")
            }
            buf.label = req.label
            bufferMap[req.label] = buf
        }

        func buffer(_ label: String) -> MTLBuffer {
            guard let buf = bufferMap[label] else {
                fatalError("Unknown buffer label \(label)")
            }
            return buf
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
        headerPtr.pointee.maxCapacity = UInt32(layout.maxAssignmentCapacity)
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
            bufferMap[label]
        }
        self.radixHistogram = optionalBuffer("RadixHist")
        self.radixBlockSums = optionalBuffer("RadixBlockSums")
        self.radixScannedHistogram = optionalBuffer("RadixScanned")
        self.radixKeysScratch = optionalBuffer("RadixScratch")
        self.radixPayloadScratch = optionalBuffer("RadixPayload")
        self.interleavedGaussians = optionalBuffer("InterleavedGaussians")
        self.tileAssignBlockSums = optionalBuffer("TileAssignBlockSums")

        // Indirect dispatch buffers
        self.visibleIndices = buffer("VisibleIndices")
        self.visibleCount = buffer("VisibleCount")
        self.tileAssignDispatchArgs = buffer("TileAssignDispatchArgs")

        // Output buffers (skipped in textureOnly mode to save ~20MB)
        let pixelCount = layout.pixelCapacity
        if textureOnly {
            self.outputBuffers = nil
        } else {
            let color = try! device.makeBuffer(count: pixelCount * 3, type: Float.self, options: .storageModeShared, label: "RenderColorOutput")
            let depth = try! device.makeBuffer(count: pixelCount, type: Float.self, options: .storageModeShared, label: "RenderDepthOutput")
            let alpha = try! device.makeBuffer(count: pixelCount, type: Float.self, options: .storageModeShared, label: "RenderAlphaOutput")
            self.outputBuffers = RenderOutputBuffers(colorOutGPU: color, depthOutGPU: depth, alphaOutGPU: alpha)
        }

        // Output textures sized to limits
        let texDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: layout.limits.maxWidth, height: layout.limits.maxHeight, mipmapped: false)
        texDesc.usage = [.shaderWrite, .shaderRead]
        texDesc.storageMode = .private
        guard let colorTex = device.makeTexture(descriptor: texDesc) else { fatalError("Failed to allocate color texture") }
        colorTex.label = "OutputColorTex"

        // Depth texture uses half precision like Local
        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: layout.limits.maxWidth, height: layout.limits.maxHeight, mipmapped: false)
        depthDesc.usage = [.shaderWrite, .shaderRead]
        depthDesc.storageMode = .private
        guard let depthTex = device.makeTexture(descriptor: depthDesc) else { fatalError("Failed to allocate depth texture") }
        depthTex.label = "OutputDepthTex"
        self.outputTextures = RenderOutputTextures(color: colorTex, depth: depthTex)
    }
}

// MARK: - Global Sort Renderer

/// High-quality Gaussian splatting renderer using global radix sort
public final class GlobalRenderer: GaussianRenderer, @unchecked Sendable {
    private static let maxSupportedGaussians = 10_000_000
    private static let tileWidth = 32
    private static let tileHeight = 16
    private static let maxPerTile = 2048
    private static let dispatchArgsSize = 10 * MemoryLayout<DispatchIndirectArgsSwift>.stride

    public let device: MTLDevice
    let library: MTLLibrary

    // Encoders
    let sortKeyGenEncoder: SortKeyGenEncoder
    let radixSortEncoder: RadixSortEncoder
    let packEncoder: PackEncoder
    let renderEncoder: RenderEncoder
    let projectEncoder: ProjectEncoder
    let dispatchEncoder: DispatchEncoder
    let fusedPipelineEncoder: FusedPipelineEncoder
    let twoPassTileAssignEncoder: TwoPassTileAssignEncoder

    // Reset tile builder state pipeline
    private let resetTileBuilderStatePipeline: MTLComputePipelineState

    // Frame Resources
    private var frame: FrameResources!

    // Gaussian projection output caches
    private var gaussianBoundsCache: MTLBuffer!
    private var gaussianMaskCache: MTLBuffer!

    // Configuration
    private let config: RendererConfig
    private let limits: RendererLimits
    let frameLayout: FrameResourceLayout

    /// Last GPU execution time in seconds
    public private(set) var lastGPUTime: Double?

    /// Primary initializer
    public init(device: MTLDevice? = nil, config: RendererConfig = RendererConfig()) throws {
        guard config.maxGaussians <= GlobalRenderer.maxSupportedGaussians else {
            throw RendererError.invalidInput("maxGaussians (\(config.maxGaussians)) exceeds limit of \(GlobalRenderer.maxSupportedGaussians)")
        }

        let device = device ?? MTLCreateSystemDefaultDevice()
        guard let device else {
            throw RendererError.deviceNotAvailable
        }
        self.device = device
        self.config = config

        guard let metallibURL = Bundle.module.url(forResource: "GlobalShaders", withExtension: "metallib") else {
            throw RendererError.failedToCreatePipeline("GlobalShaders.metallib not found in bundle")
        }
        let library = try device.makeLibrary(URL: metallibURL)
        self.library = library

        // Initialize encoders
        self.sortKeyGenEncoder = try SortKeyGenEncoder(device: device, library: library)
        self.radixSortEncoder = try RadixSortEncoder(device: device, library: library)
        self.packEncoder = try PackEncoder(device: device, library: library)
        self.renderEncoder = try RenderEncoder(device: device, library: library)
        self.projectEncoder = try ProjectEncoder(device: device, library: library)
        self.fusedPipelineEncoder = try FusedPipelineEncoder(device: device, library: library)
        self.twoPassTileAssignEncoder = try TwoPassTileAssignEncoder(device: device, library: library)

        let dispatchConfig = AssignmentDispatchConfigSwift(
            sortThreadgroupSize: UInt32(self.sortKeyGenEncoder.threadgroupSize),
            fuseThreadgroupSize: 256,
            unpackThreadgroupSize: 256,
            packThreadgroupSize: UInt32(self.packEncoder.packThreadgroupSize),
            radixBlockSize: UInt32(self.radixSortEncoder.blockSize),
            radixGrainSize: UInt32(self.radixSortEncoder.grainSize),
            maxAssignments: 0
        )
        self.dispatchEncoder = try DispatchEncoder(device: device, library: library, config: dispatchConfig)

        guard let resetFn = library.makeFunction(name: "resetTileBuilderStateKernel") else {
            throw RendererError.failedToCreatePipeline("resetTileBuilderStateKernel not found")
        }
        self.resetTileBuilderStatePipeline = try device.makeComputePipelineState(function: resetFn)

        // Compute limits and layout
        self.limits = RendererLimits(
            from: config,
            tileWidth: GlobalRenderer.tileWidth,
            tileHeight: GlobalRenderer.tileHeight,
            maxPerTile: GlobalRenderer.maxPerTile
        )

        self.frameLayout = GlobalRenderer.computeLayout(
            limits: self.limits,
            precision: config.precision,
            radixSortEncoder: self.radixSortEncoder
        )

        // Initialize frame resources
        self.frame = FrameResources(
            device: device,
            layout: self.frameLayout,
            textureOnly: !config.useTexturedRender
        )

        // Allocate projection caches
        let g = self.limits.maxGaussians
        guard let boundsCache = device.makeBuffer(length: g * MemoryLayout<SIMD4<Int32>>.stride, options: .storageModeShared),
              let maskCache = device.makeBuffer(length: g, options: .storageModeShared)
        else {
            throw RendererError.failedToCreatePipeline("Failed to allocate projection caches")
        }
        boundsCache.label = "GaussianBounds"
        maskCache.label = "GaussianMask"
        self.gaussianBoundsCache = boundsCache
        self.gaussianMaskCache = maskCache
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

        _ = encodeRenderToTargetTexture(
            commandBuffer: commandBuffer,
            gaussianCount: input.gaussianCount,
            packedWorldBuffers: packedWorld,
            cameraUniforms: cameraUniforms,
            frameParams: frameParams,
            targetColor: colorTexture,
            targetDepth: depthTexture
        )
    }

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

    private static func computeLayout(
        limits: RendererLimits,
        precision: Precision,
        radixSortEncoder: RadixSortEncoder
    ) -> FrameResourceLayout {
        let gaussianCapacity = limits.maxGaussians
        let tileCount = limits.maxTileCount

        let tileCapacity = tileCount * limits.maxPerTile
        let gaussianTileCapacity = gaussianCapacity * 32
        let maxAssignmentCapacity = min(tileCapacity, gaussianTileCapacity)

        // paddedCapacity: 1024-aligned for radix sort (blockSize * grainSize)
        let block = radixSortEncoder.blockSize * radixSortEncoder.grainSize
        let paddedCapacity = ((maxAssignmentCapacity + block - 1) / block) * block

        let strideForPrecision: Int = (precision == .float16) ? 2 : 4
        let half2Stride = MemoryLayout<UInt16>.stride * 2
        let half4Stride = MemoryLayout<UInt16>.stride * 4
        func strideForMeans() -> Int { precision == .float16 ? half2Stride : 8 }
        func strideForConics() -> Int { precision == .float16 ? half4Stride : 16 }
        func strideForColors() -> Int { precision == .float16 ? MemoryLayout<UInt16>.stride * 3 : 12 }
        func strideForOpacities() -> Int { precision == .float16 ? MemoryLayout<UInt16>.stride : 4 }
        func strideForDepths() -> Int { precision == .float16 ? MemoryLayout<UInt16>.stride : 4 }
        // Metal half4 requires 8-byte alignment (28 bytes) + struct padding to 32 for stride
        func strideForInterleaved() -> Int { precision == .float16 ? 32 : 48 }

        let prefixBlockSize = 256
        let prefixGrainSize = 4
        let elementsPerGroup = prefixBlockSize * prefixGrainSize
        let groups = (gaussianCapacity + elementsPerGroup - 1) / elementsPerGroup

        var bufferAllocations: [(label: String, length: Int, options: MTLResourceOptions)] = []
        func add(_ label: String, _ length: Int, _ opts: MTLResourceOptions = .storageModePrivate) {
            bufferAllocations.append((label, max(1, length), opts))
        }

        // Shared allocations (CPU-visible)
        add("TileHeader", MemoryLayout<TileAssignmentHeaderSwift>.stride, .storageModeShared)
        add("ActiveTileCount", MemoryLayout<UInt32>.stride, .storageModeShared)

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
        // Keys are now single uint32: [tile:16][depth:16]
        add("SortKeys", paddedCapacity * MemoryLayout<UInt32>.stride)
        add("SortedIndices", paddedCapacity * MemoryLayout<Int32>.stride)

        // Radix sort buffers
        // Note: gridCapacity is for worst-case; runtime uses ceil(totalAssignments/1024) blocks
        let valuesPerGroup = radixSortEncoder.blockSize * radixSortEncoder.grainSize
        let gridCapacity = max(1, (maxAssignmentCapacity + valuesPerGroup - 1) / valuesPerGroup)
        let histogramCapacity = gridCapacity * radixSortEncoder.radix
        add("RadixHist", histogramCapacity * MemoryLayout<UInt32>.stride)
        add("RadixBlockSums", gridCapacity * MemoryLayout<UInt32>.stride)
        add("RadixScanned", histogramCapacity * MemoryLayout<UInt32>.stride)
        // Scratch buffers for ping-pong during radix sort
        add("RadixScratch", maxAssignmentCapacity * MemoryLayout<UInt32>.stride)
        add("RadixPayload", maxAssignmentCapacity * MemoryLayout<UInt32>.stride)

        // Interleaved gaussian data
        add("InterleavedGaussians", gaussianCapacity * strideForInterleaved())

        // Two-pass tile assignment scratch buffer (hierarchical prefix sum block sums)
        // Block size is 256, so numBlocks = ceil(gaussianCapacity / 256)
        // For hierarchical scan: level1 needs numBlocks+1, level2 needs level2Blocks+1
        let tileAssignBlockSize = 256
        let tileAssignNumBlocks = (gaussianCapacity + tileAssignBlockSize - 1) / tileAssignBlockSize
        let tileAssignLevel2Blocks = (tileAssignNumBlocks + tileAssignBlockSize - 1) / tileAssignBlockSize
        let tileAssignBlockSumsSize = (tileAssignNumBlocks + 1 + tileAssignLevel2Blocks + 1) * MemoryLayout<UInt32>.stride
        add("TileAssignBlockSums", tileAssignBlockSumsSize)

        // Indirect dispatch buffers for compacted visible gaussians
        add("VisibleIndices", gaussianCapacity * MemoryLayout<UInt32>.stride)
        add("VisibleCount", MemoryLayout<UInt32>.stride, .storageModeShared) // Atomic counter
        add("TileAssignDispatchArgs", 12) // MTLDispatchThreadgroupsIndirectArguments (3 x uint32)

        return FrameResourceLayout(
            limits: limits,
            maxAssignmentCapacity: maxAssignmentCapacity,
            paddedCapacity: paddedCapacity,
            bufferAllocations: bufferAllocations,
            pixelCapacity: limits.maxWidth * limits.maxHeight,
            precisionStride: strideForPrecision
        )
    }

    // MARK: - Internal Render Methods

    func encodeRender(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        cameraUniforms: CameraUniformsSwift,
        frameParams: FrameParams
    ) -> RenderOutputBuffers? {
        let params = self.limits.buildParams(from: frameParams)
        guard self.validateLimits(gaussianCount: gaussianCount) else { return nil }

        guard let projectionOutput = self.prepareProjectionOutput(frame: frame) else {
            return nil
        }

        // Fused projection + tile bounds (single pass)
        self.projectEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: cameraUniforms,
            output: projectionOutput,
            tileWidth: params.tileWidth,
            tileHeight: params.tileHeight,
            tilesX: params.tilesX,
            tilesY: UInt32(self.limits.tilesY),
            useHalfWorld: self.config.precision == .float16
        )

        guard let assignment = self.buildTileAssignments(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            projectionOutput: projectionOutput,
            params: params,
            frame: frame
        ) else {
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
            return nil
        }

        guard let ordered = self.buildOrderedGaussians(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            assignment: assignment,
            projectionOutput: projectionOutput,
            params: params,
            frame: frame
        ) else {
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
            return nil
        }

        let submission = self.submitRender(
            commandBuffer: commandBuffer,
            ordered: ordered,
            params: params,
            frame: frame,
            precision: self.config.precision
        )

        guard submission else {
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
            return nil
        }

        commandBuffer.addCompletedHandler { _ in
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
        }

        return frame.outputBuffers
    }

    /// Render to textures using packed GaussianRenderData pipeline
    func encodeRenderToTextures(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        cameraUniforms: CameraUniformsSwift,
        frameParams: FrameParams
    ) -> RenderOutputTextures? {
        let params = self.limits.buildParams(from: frameParams)
        guard self.validateLimits(gaussianCount: gaussianCount) else {
            fatalError("[encodeRenderToTextures] validateLimits failed")
        }

        // Prepare projection output (packed renderData + radii + mask)
        guard let projectionOutput = self.prepareProjectionOutput(frame: frame) else {
            fatalError("[encodeRenderToTextures] prepareProjectionOutput failed")
        }

        // Fused projection + tile bounds (single pass)
        self.projectEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: cameraUniforms,
            output: projectionOutput,
            tileWidth: params.tileWidth,
            tileHeight: params.tileHeight,
            tilesX: params.tilesX,
            tilesY: UInt32(self.limits.tilesY),
            useHalfWorld: self.config.precision == .float16
        )

        // Build tile assignments from packed render data
        guard let assignment = self.buildTileAssignments(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            projectionOutput: projectionOutput,
            params: params,
            frame: frame
        ) else {
            fatalError("[encodeRenderToTextures] buildTileAssignments failed")
        }

        // Build sorted indices for rendering
        guard let ordered = self.buildOrderedGaussians(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            assignment: assignment,
            projectionOutput: projectionOutput,
            params: params,
            frame: frame
        ) else {
            fatalError("[encodeRenderToTextures] buildOrderedGaussians failed")
        }

        // Submit render to textures
        let textures = frame.outputTextures
        let dispatchArgs = frame.dispatchArgs
        let dispatchOffset = DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride

        guard let renderData = ordered.renderData,
              let sortedIndices = ordered.sortedIndices
        else {
            fatalError("[encodeRenderToTextures] No renderData or sortedIndices available")
        }
        self.fusedPipelineEncoder.encodeCompleteRender(
            commandBuffer: commandBuffer,
            orderedBuffers: ordered,
            renderData: renderData,
            sortedIndices: sortedIndices,
            colorTexture: textures.color,
            depthTexture: textures.depth,
            params: params,
            dispatchArgs: dispatchArgs,
            dispatchOffset: dispatchOffset
        )

        commandBuffer.addCompletedHandler { _ in
            if MTLCaptureManager.shared().isCapturing { MTLCaptureManager.shared().stopCapture() }
        }

        return textures
    }

    /// Render to user-provided texture (most direct path).
    /// Allows GSViewer to pass its own drawable texture for zero-copy rendering.
    func encodeRenderToTargetTexture(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        cameraUniforms: CameraUniformsSwift,
        frameParams: FrameParams,
        targetColor: MTLTexture,
        targetDepth: MTLTexture?
    ) -> Bool {
        let params = self.limits.buildParams(from: frameParams)
        guard self.validateLimits(gaussianCount: gaussianCount) else {
            return false
        }

        guard let projectionOutput = self.prepareProjectionOutput(frame: frame) else {
            return false
        }

        // Fused projection + tile bounds (single pass)
        self.projectEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: cameraUniforms,
            output: projectionOutput,
            tileWidth: params.tileWidth,
            tileHeight: params.tileHeight,
            tilesX: params.tilesX,
            tilesY: UInt32(self.limits.tilesY),
            useHalfWorld: self.config.precision == .float16
        )

        guard let assignment = self.buildTileAssignments(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            projectionOutput: projectionOutput,
            params: params,
            frame: frame
        ) else {
            return false
        }

        guard let ordered = self.buildOrderedGaussians(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            assignment: assignment,
            projectionOutput: projectionOutput,
            params: params,
            frame: frame
        ) else {
            return false
        }

        let dispatchArgs = frame.dispatchArgs
        let dispatchOffset = DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride

        // Use user's target textures
        let colorTex = targetColor
        let depthTex = targetDepth ?? frame.outputTextures.depth

        guard let renderData = ordered.renderData,
              let sortedIndices = ordered.sortedIndices
        else {
            return false
        }
        self.fusedPipelineEncoder.encodeCompleteRender(
            commandBuffer: commandBuffer,
            orderedBuffers: ordered,
            renderData: renderData,
            sortedIndices: sortedIndices,
            colorTexture: colorTex,
            depthTexture: depthTex,
            params: params,
            dispatchArgs: dispatchArgs,
            dispatchOffset: dispatchOffset
        )

        return true
    }

    // MARK: - Helper Methods

    private func validateLimits(gaussianCount: Int) -> Bool {
        // CRASH on validation failures - silent returns cause black screens!
        // Since params are built from limits, we only need to validate gaussianCount
        precondition(gaussianCount <= self.limits.maxGaussians,
                     "gaussianCount (\(gaussianCount)) exceeds limits.maxGaussians (\(self.limits.maxGaussians))")
        return true
    }

    // Legacy validation for tests that construct RenderParams manually
    private func validateLimits(gaussianCount: Int, params: RenderParams) -> Bool {
        precondition(gaussianCount <= self.limits.maxGaussians,
                     "gaussianCount (\(gaussianCount)) exceeds limits.maxGaussians (\(self.limits.maxGaussians))")
        precondition(Int(params.tilesX * params.tilesY) <= self.limits.maxTileCount,
                     "tileCount exceeds limits.maxTileCount (\(self.limits.maxTileCount))")
        return true
    }

    /// Build tile assignments from packed GaussianRenderData
    func buildTileAssignments(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        projectionOutput: ProjectionOutput,
        params: RenderParams,
        frame: FrameResources
    ) -> TileAssignmentBuffers? {
        precondition(gaussianCount <= self.limits.maxGaussians,
                     "[buildTileAssignments] gaussianCount (\(gaussianCount)) > limits.maxGaussians (\(self.limits.maxGaussians))")

        let tileCount = Int(params.tilesX * params.tilesY)
        precondition(tileCount <= self.limits.maxTileCount,
                     "[buildTileAssignments] tileCount (\(tileCount)) > limits.maxTileCount (\(self.limits.maxTileCount))")

        let perTileLimit = (params.maxPerTile == 0) ? UInt32(self.limits.maxPerTile) : min(params.maxPerTile, UInt32(self.limits.maxPerTile))
        let tileCapacity = tileCount * Int(perTileLimit)
        let gaussianTileCapacity = gaussianCount * 8
        let requiredCapacity = min(tileCapacity, gaussianTileCapacity)

        precondition(requiredCapacity <= frame.tileAssignmentMaxAssignments,
                     "[buildTileAssignments] requiredCapacity (\(requiredCapacity)) > frame.tileAssignmentMaxAssignments (\(frame.tileAssignmentMaxAssignments))")

        self.resetTileBuilderState(commandBuffer: commandBuffer, frame: frame)

        // Note: tileBoundsEncoder removed - bounds already computed in fused projection kernel
        // Copy bounds from projection output to frame's boundsBuffer for coverage+scatter
        // Actually, we can use projectionOutput.bounds directly!

        // Two-pass tile assignment with indirect dispatch (compacts visible gaussians first)
        // Bounds are already computed in the fused projection kernel (projectionOutput.bounds)
        guard let blockSumsBuffer = frame.tileAssignBlockSums else {
            fatalError("[buildTileAssignments] tileAssignBlockSums buffer is nil")
        }

        let indirectBuffers = TwoPassTileAssignEncoder.IndirectBuffers(
            visibleIndices: frame.visibleIndices,
            visibleCount: frame.visibleCount,
            indirectDispatchArgs: frame.tileAssignDispatchArgs
        )

        self.twoPassTileAssignEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            tileWidth: Int(params.tileWidth),
            tileHeight: Int(params.tileHeight),
            tilesX: Int(params.tilesX),
            maxAssignments: frame.tileAssignmentMaxAssignments,
            boundsBuffer: projectionOutput.bounds,
            coverageBuffer: frame.coverageBuffer,
            renderData: projectionOutput.renderData,
            tileIndicesBuffer: frame.tileIndices,
            tileIdsBuffer: frame.tileIds,
            tileAssignmentHeader: frame.tileAssignmentHeader,
            blockSumsBuffer: blockSumsBuffer,
            indirectBuffers: indirectBuffers
        )

        return TileAssignmentBuffers(
            tileCount: tileCount,
            maxAssignments: frame.tileAssignmentMaxAssignments,
            tileIndices: frame.tileIndices,
            tileIds: frame.tileIds,
            header: frame.tileAssignmentHeader
        )
    }

    /// Build ordered gaussian buffers - generates sort keys, sorts, and prepares for rendering
    /// Uses index-based render (no pack step - render reads via sortedIndices into packed renderData)
    func buildOrderedGaussians(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        assignment: TileAssignmentBuffers,
        projectionOutput: ProjectionOutput,
        params _: RenderParams,
        frame: FrameResources
    ) -> OrderedGaussianBuffers? {
        precondition(gaussianCount <= self.limits.maxGaussians,
                     "[buildOrderedGaussians] gaussianCount (\(gaussianCount)) > limits.maxGaussians (\(self.limits.maxGaussians))")
        precondition(assignment.maxAssignments <= frame.tileAssignmentMaxAssignments,
                     "[buildOrderedGaussians] assignment.maxAssignments (\(assignment.maxAssignments)) > frame.tileAssignmentMaxAssignments (\(frame.tileAssignmentMaxAssignments))")

        let sortKeysBuffer = frame.sortKeys
        let sortedIndicesBuffer = frame.sortedIndices
        let dispatchArgs = frame.dispatchArgs

        self.dispatchEncoder.encode(
            commandBuffer: commandBuffer,
            header: assignment.header,
            dispatchArgs: dispatchArgs,
            maxAssignments: frame.tileAssignmentMaxAssignments
        )

        // Sort key generation - reads depth from packed GaussianRenderData
        self.sortKeyGenEncoder.encode(
            commandBuffer: commandBuffer,
            tileIds: assignment.tileIds,
            tileIndices: assignment.tileIndices,
            renderData: projectionOutput.renderData,
            sortKeys: sortKeysBuffer,
            sortedIndices: sortedIndicesBuffer,
            header: assignment.header,
            dispatchArgs: dispatchArgs,
            dispatchOffset: DispatchSlot.sortKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
        )

        // Radix sort
        let radixBuffers = RadixBufferSet(
            histogram: frame.radixHistogram!,
            blockSums: frame.radixBlockSums!,
            scannedHistogram: frame.radixScannedHistogram!,
            scratchKeys: frame.radixKeysScratch!,
            scratchPayload: frame.radixPayloadScratch!
        )

        let offsets = (
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

        // Build headers and active tile list (index-based render - no pack step)
        let orderedBuffers = OrderedBufferSet(
            headers: frame.orderedHeaders,
            means: frame.packedMeans,
            conics: frame.packedConics,
            colors: frame.packedColors,
            opacities: frame.packedOpacities,
            depths: frame.packedDepths
        )
        frame.activeTileCount.contents().storeBytes(of: UInt32(0), as: UInt32.self)

        self.packEncoder.encodeHeadersAndActiveTiles(
            commandBuffer: commandBuffer,
            sortedKeys: sortKeysBuffer,
            assignment: assignment,
            orderedHeaders: orderedBuffers.headers,
            activeTileIndices: frame.activeTileIndices,
            activeTileCount: frame.activeTileCount
        )

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
            precision: self.config.precision,
            renderData: projectionOutput.renderData,
            sortedIndices: sortedIndicesBuffer
        )
    }

    private func submitRender(
        commandBuffer: MTLCommandBuffer,
        ordered: OrderedGaussianBuffers,
        params: RenderParams,
        frame: FrameResources,
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

    // MARK: - Resource Management

    func prepareProjectionOutput(frame: FrameResources) -> ProjectionOutput? {
        guard
            let renderData = frame.interleavedGaussians,
            let bounds = self.gaussianBoundsCache,
            let mask = self.gaussianMaskCache
        else { return nil }

        return ProjectionOutput(
            renderData: renderData,
            bounds: bounds, // int4 tile bounds (computed directly in fused projection)
            mask: mask
        )
    }

    private func resetTileBuilderState(commandBuffer: MTLCommandBuffer, frame: FrameResources) {
        // GPU-only reset: use compute shader for lower overhead than blit encoder
        // Resets: totalAssignments, overflow, and activeTileCount in a single dispatch
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("[resetTileBuilderState] Failed to create compute encoder")
        }
        encoder.label = "ResetTileBuilderState"
        encoder.setComputePipelineState(self.resetTileBuilderStatePipeline)
        encoder.setBuffer(frame.tileAssignmentHeader, offset: 0, index: 0)
        encoder.setBuffer(frame.activeTileCount, offset: 0, index: 1)
        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Pad assignment capacity for radix sort alignment (1024-aligned)
    private func paddedAssignmentCapacity(for totalAssignments: Int) -> Int {
        guard totalAssignments > 0 else { return 1 }
        let block = self.radixSortEncoder.blockSize * self.radixSortEncoder.grainSize
        return ((totalAssignments + block - 1) / block) * block
    }
}

import Foundation
@preconcurrency import Metal
import RendererTypes
import simd

struct RendererLimits: Sendable {
    let maxGaussians: Int
    let maxWidth: Int
    let maxHeight: Int
    let tileWidth: Int
    let tileHeight: Int

    var tilesX: Int { (self.maxWidth + self.tileWidth - 1) / self.tileWidth }
    var tilesY: Int { (self.maxHeight + self.tileHeight - 1) / self.tileHeight }
    var maxTileCount: Int { max(1, self.tilesX * self.tilesY) }

    init(from config: RendererConfig, tileWidth: Int = 32, tileHeight: Int = 16,) {
        self.maxGaussians = max(1, config.maxGaussians)
        self.maxWidth = max(1, config.maxWidth)
        self.maxHeight = max(1, config.maxHeight)
        self.tileWidth = max(1, tileWidth)
        self.tileHeight = max(1, tileHeight)
    }

    func buildParams(from frame: FrameParams) -> RenderParams {
        RenderParams(
            width: UInt32(self.maxWidth),
            height: UInt32(self.maxHeight),
            tileWidth: UInt32(self.tileWidth),
            tileHeight: UInt32(self.tileHeight),
            tilesX: UInt32(self.tilesX),
            tilesY: UInt32(self.tilesY),
            activeTileCount: 0,
            gaussianCount: UInt32(frame.gaussianCount)
        )
    }

    func buildBinningParams(gaussianCount: Int) -> TileBinningParams {
        TileBinningParams(
            gaussianCount: UInt32(gaussianCount),
            tilesX: UInt32(self.tilesX),
            tilesY: UInt32(self.tilesY),
            tileWidth: UInt32(self.tileWidth),
            tileHeight: UInt32(self.tileHeight),
            surfaceWidth: UInt32(self.maxWidth),
            surfaceHeight: UInt32(self.maxHeight),
            maxCapacity: UInt32(gaussianCount),
            alphaThreshold: 0.005,
            totalInkThreshold: 3.0
        )
    }
}


public final class GlobalRenderer: GaussianRenderer, @unchecked Sendable {
    private static let maxSupportedGaussians = 30_000_000
    private static let tileWidth = 32
    private static let tileHeight = 16
    private static let dispatchArgsSize = 10 * MemoryLayout<DispatchIndirectArgsSwift>.stride

    public let device: MTLDevice
    let library: MTLLibrary

    // Encoders
    let sortKeyGenEncoder: SortKeyGenEncoder
    let radixSortEncoder: RadixSortEncoder
    let packEncoder: PackEncoder
    let projectEncoder: ProjectEncoder
    let dispatchEncoder: DispatchEncoder
    let fusedPipelineEncoder: FusedPipelineEncoder
    let twoPassTileAssignEncoder: TwoPassTileAssignEncoder

    // Reset tile builder state pipeline
    private let resetTileBuilderStatePipeline: MTLComputePipelineState

    // Stereo resources (left for mono, both for stereo)
    private let stereoResources: GlobalMultiViewResources

    // Gaussian projection output caches (per-eye)
    private let leftBoundsCache: MTLBuffer
    private let leftMaskCache: MTLBuffer
    private let rightBoundsCache: MTLBuffer
    private let rightMaskCache: MTLBuffer

    // Configuration
    private let config: RendererConfig
    private let limits: RendererLimits

    /// Last GPU execution time in seconds
    public private(set) var lastGPUTime: Double?

    /// Primary initializer
    public init(device: MTLDevice? = nil, config: RendererConfig = RendererConfig()) throws {
        guard config.maxGaussians <= GlobalRenderer.maxSupportedGaussians else {
            throw RendererError.invalidGaussianCount(provided: config.maxGaussians, maximum: GlobalRenderer.maxSupportedGaussians)
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

        // Compute limits
        self.limits = RendererLimits(
            from: config,
            tileWidth: GlobalRenderer.tileWidth,
            tileHeight: GlobalRenderer.tileHeight,
        )

        // Initialize stereo resources (left for mono, both for stereo)
        self.stereoResources = try GlobalMultiViewResources(
            device: device,
            maxGaussians: limits.maxGaussians,
            maxWidth: limits.maxWidth,
            maxHeight: limits.maxHeight,
            tileWidth: limits.tileWidth,
            tileHeight: limits.tileHeight,
            radixBlockSize: radixSortEncoder.blockSize,
            radixGrainSize: radixSortEncoder.grainSize,
            precision: config.precision
        )

        // Allocate projection caches (per-eye for stereo)
        let g = self.limits.maxGaussians
        let boundsSize = g * MemoryLayout<SIMD4<Int32>>.stride

        guard let leftBoundsCache = device.makeBuffer(length: boundsSize, options: .storageModeShared) else {
            throw RendererError.failedToAllocateBuffer(label: "LeftGaussianBounds", size: boundsSize)
        }
        guard let leftMaskCache = device.makeBuffer(length: g, options: .storageModeShared) else {
            throw RendererError.failedToAllocateBuffer(label: "LeftGaussianMask", size: g)
        }
        leftBoundsCache.label = "LeftGaussianBounds"
        leftMaskCache.label = "LeftGaussianMask"
        self.leftBoundsCache = leftBoundsCache
        self.leftMaskCache = leftMaskCache

        guard let rightBoundsCache = device.makeBuffer(length: boundsSize, options: .storageModeShared) else {
            throw RendererError.failedToAllocateBuffer(label: "RightGaussianBounds", size: boundsSize)
        }
        guard let rightMaskCache = device.makeBuffer(length: g, options: .storageModeShared) else {
            throw RendererError.failedToAllocateBuffer(label: "RightGaussianMask", size: g)
        }
        rightBoundsCache.label = "RightGaussianBounds"
        rightMaskCache.label = "RightGaussianMask"
        self.rightBoundsCache = rightBoundsCache
        self.rightMaskCache = rightMaskCache
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
        let cameraUniforms = CameraUniformsSwift(
            from: camera,
            width: width,
            height: height,
            gaussianCount: input.gaussianCount,
            shComponents: input.shComponents
        )

        let frameParams = FrameParams(gaussianCount: input.gaussianCount)

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
            targetDepth: depthTexture,
            frame: stereoResources.left,
            boundsCache: leftBoundsCache,
            maskCache: leftMaskCache
        )
    }

    public func renderStereo(
        commandBuffer: MTLCommandBuffer,
        output: StereoRenderOutput,
        input: GaussianInput,
        camera: StereoCameraParams,
        width: Int,
        height: Int
    ) {
        let frameParams = FrameParams(gaussianCount: input.gaussianCount)

        let packedWorld = PackedWorldBuffers(
            packedGaussians: input.gaussians,
            harmonics: input.harmonics
        )

        let leftCameraUniforms = CameraUniformsSwift(
            from: camera.leftEye,
            width: width,
            height: height,
            gaussianCount: input.gaussianCount,
            shComponents: input.shComponents
        )

        _ = encodeRenderToTargetTexture(
            commandBuffer: commandBuffer,
            gaussianCount: input.gaussianCount,
            packedWorldBuffers: packedWorld,
            cameraUniforms: leftCameraUniforms,
            frameParams: frameParams,
            targetColor: output.leftColor,
            targetDepth: output.leftDepth,
            frame: stereoResources.left,
            boundsCache: leftBoundsCache,
            maskCache: leftMaskCache
        )

        let rightCameraUniforms = CameraUniformsSwift(
            from: camera.rightEye,
            width: width,
            height: height,
            gaussianCount: input.gaussianCount,
            shComponents: input.shComponents
        )

        _ = encodeRenderToTargetTexture(
            commandBuffer: commandBuffer,
            gaussianCount: input.gaussianCount,
            packedWorldBuffers: packedWorld,
            cameraUniforms: rightCameraUniforms,
            frameParams: frameParams,
            targetColor: output.rightColor,
            targetDepth: output.rightDepth,
            frame: stereoResources.right,
            boundsCache: rightBoundsCache,
            maskCache: rightMaskCache
        )
    }

    func encodeRenderToTextures(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        cameraUniforms: CameraUniformsSwift,
        frameParams: FrameParams
    ) -> RenderOutputTextures? {
        let textures = stereoResources.left.outputTextures
        let success = encodeRenderToTargetTexture(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: cameraUniforms,
            frameParams: frameParams,
            targetColor: textures.color,
            targetDepth: textures.depth,
            frame: stereoResources.left,
            boundsCache: leftBoundsCache,
            maskCache: leftMaskCache
        )
        return success ? textures : nil
    }

    /// Render to user-provided texture (main render path)
    private func encodeRenderToTargetTexture(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        packedWorldBuffers: PackedWorldBuffers,
        cameraUniforms: CameraUniformsSwift,
        frameParams: FrameParams,
        targetColor: MTLTexture,
        targetDepth: MTLTexture?,
        frame: GlobalViewResources,
        boundsCache: MTLBuffer,
        maskCache: MTLBuffer
    ) -> Bool {
        let params = self.limits.buildParams(from: frameParams)

        do {
            try self.validateLimits(gaussianCount: gaussianCount)
        } catch {
            return false
        }

        let projectionOutput = self.prepareProjectionOutput(
            frame: frame,
            boundsCache: boundsCache,
            maskCache: maskCache
        )

        let binningParams = self.limits.buildBinningParams(gaussianCount: gaussianCount)
        self.projectEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: cameraUniforms,
            output: projectionOutput,
            params: binningParams,
            useHalfWorld: self.config.precision == .float16
        )

        let assignment: TileAssignmentBuffers
        do {
            assignment = try self.buildTileAssignments(
                commandBuffer: commandBuffer,
                gaussianCount: gaussianCount,
                projectionOutput: projectionOutput,
                params: params,
                frame: frame
            )
        } catch {
            return false
        }

        let ordered: OrderedGaussianBuffers
        do {
            ordered = try self.buildOrderedGaussians(
                commandBuffer: commandBuffer,
                gaussianCount: gaussianCount,
                assignment: assignment,
                projectionOutput: projectionOutput,
                params: params,
                frame: frame
            )
        } catch {
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

    private func validateLimits(gaussianCount: Int) throws {
        guard gaussianCount <= self.limits.maxGaussians else {
            throw RendererError.invalidGaussianCount(provided: gaussianCount, maximum: self.limits.maxGaussians)
        }
    }

    /// Build tile assignments from packed GaussianRenderData
    func buildTileAssignments(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        projectionOutput: ProjectionOutput,
        params: RenderParams,
        frame: GlobalViewResources
    ) throws -> TileAssignmentBuffers {
        guard gaussianCount <= self.limits.maxGaussians else {
            throw RendererError.invalidGaussianCount(provided: gaussianCount, maximum: self.limits.maxGaussians)
        }

        let tileCount = Int(params.tilesX * params.tilesY)
        guard tileCount <= self.limits.maxTileCount else {
            throw RendererError.invalidTileCount(provided: tileCount, maximum: self.limits.maxTileCount)
        }

        let requiredCapacity = gaussianCount * 4
        guard requiredCapacity <= frame.tileAssignmentMaxAssignments else {
            throw RendererError.invalidAssignmentCapacity(required: requiredCapacity, available: frame.tileAssignmentMaxAssignments)
        }

        self.resetTileBuilderState(commandBuffer: commandBuffer, frame: frame)

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
            blockSumsBuffer: frame.tileAssignBlockSums,
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
        frame: GlobalViewResources
    ) throws -> OrderedGaussianBuffers {
        guard gaussianCount <= self.limits.maxGaussians else {
            throw RendererError.invalidGaussianCount(provided: gaussianCount, maximum: self.limits.maxGaussians)
        }
        guard assignment.maxAssignments <= frame.tileAssignmentMaxAssignments else {
            throw RendererError.invalidAssignmentCapacity(required: assignment.maxAssignments, available: frame.tileAssignmentMaxAssignments)
        }

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

        let radixBuffers = RadixBufferSet(
            histogram: frame.radixHistogram,
            blockSums: frame.radixBlockSums,
            scannedHistogram: frame.radixScannedHistogram,
            scratchKeys: frame.radixKeysScratch,
            scratchPayload: frame.radixPayloadScratch
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

    func prepareProjectionOutput(
        frame: GlobalViewResources,
        boundsCache: MTLBuffer,
        maskCache: MTLBuffer
    ) -> ProjectionOutput {
        return ProjectionOutput(
            renderData: frame.interleavedGaussians,
            bounds: boundsCache,
            mask: maskCache
        )
    }

    private func resetTileBuilderState(commandBuffer: MTLCommandBuffer, frame: GlobalViewResources) {
        // GPU-only reset: use compute shader for lower overhead than blit encoder
        // Resets: totalAssignments, overflow, and activeTileCount in a single dispatch
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            // Log error but continue - this is a non-fatal condition during render
            return
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

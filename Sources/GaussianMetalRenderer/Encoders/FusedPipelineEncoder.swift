import Metal

/// Encoder for fused pipeline operations using interleaved data structures
/// Provides cache-efficient single-struct reads instead of scattered buffer access
final class FusedPipelineEncoder {
    // Pipeline states
    private let interleaveHalfPipeline: MTLComputePipelineState
    private let interleaveFloatPipeline: MTLComputePipelineState
    private let packFusedHalfPipeline: MTLComputePipelineState
    private let packFusedFloatPipeline: MTLComputePipelineState
    private let renderFusedHalfPipeline: MTLComputePipelineState
    private let renderFusedFloatPipeline: MTLComputePipelineState
    private let renderFusedMultiPixelHalfPipeline: MTLComputePipelineState
    private let renderFusedMultiPixelFloatPipeline: MTLComputePipelineState
    private let renderFusedV2HalfPipeline: MTLComputePipelineState?  // V2: lower register pressure
    private let renderFusedV3HalfPipeline: MTLComputePipelineState?  // V3: local sort style, no shared mem
    private let prepareDispatchPipeline: MTLComputePipelineState
    private let clearTexturesPipeline: MTLComputePipelineState

    // Threadgroup sizes
    let interleaveThreadgroupSize: Int
    let packThreadgroupSize: Int
    let renderThreadgroupSize: MTLSize
    let renderMultiPixelThreadgroupSize: MTLSize  // 8x8 for 32x16 tiles (4x2 pixels/thread)
    let renderV2ThreadgroupSize: MTLSize  // 16x8 for 32x16 tiles (2x2 pixels/thread)
    let renderV3ThreadgroupSize: MTLSize  // 8x8 for 32x16 tiles (4x2 pixels/thread) - local sort style

    /// Render kernel version: 1=original, 2=shared mem, 3=Local (no shared mem, 8 pixels/thread)
    public var renderVersion: Int = 2  // Default to V2 (shared mem, lower register pressure)

    init(device: MTLDevice, library: MTLLibrary) throws {
        // Interleave kernels
        guard let interleaveHalfFn = library.makeFunction(name: "interleaveGaussianDataKernel_half"),
              let interleaveFloatFn = library.makeFunction(name: "interleaveGaussianDataKernel_float")
        else {
            throw NSError(domain: "FusedPipelineEncoder", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Interleave kernel functions missing"])
        }

        // Pack fused kernels
        guard let packFusedHalfFn = library.makeFunction(name: "packFusedKernel_half"),
              let packFusedFloatFn = library.makeFunction(name: "packFusedKernel_float")
        else {
            throw NSError(domain: "FusedPipelineEncoder", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Pack fused kernel functions missing"])
        }

        // Render fused kernels
        guard let renderFusedHalfFn = library.makeFunction(name: "renderTilesFused_half"),
              let renderFusedFloatFn = library.makeFunction(name: "renderTilesFused_float")
        else {
            throw NSError(domain: "FusedPipelineEncoder", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "Render fused kernel functions missing"])
        }

        // Render fused multi-pixel kernels (32x16 tiles)
        guard let renderFusedMultiPixelHalfFn = library.makeFunction(name: "renderTilesFusedMultiPixel_half"),
              let renderFusedMultiPixelFloatFn = library.makeFunction(name: "renderTilesFusedMultiPixel_float")
        else {
            throw NSError(domain: "FusedPipelineEncoder", code: 5,
                          userInfo: [NSLocalizedDescriptionKey: "Render fused multi-pixel kernel functions missing"])
        }

        // Render fused V2 kernel (lower register pressure, higher occupancy)
        let renderFusedV2HalfFn = library.makeFunction(name: "renderTilesFusedV2_half")

        // Render fused V3 kernel (local sort style: no shared memory, 8 pixels/thread)
        let renderFusedV3HalfFn = library.makeFunction(name: "renderTilesFusedV3_half")

        // Prepare dispatch and clear textures kernels
        guard let prepFn = library.makeFunction(name: "prepareRenderDispatchKernel"),
              let clearTexFn = library.makeFunction(name: "clearRenderTexturesKernel")
        else {
            throw NSError(domain: "FusedPipelineEncoder", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "Prepare/Clear kernel functions missing"])
        }

        // Create pipeline states
        self.interleaveHalfPipeline = try device.makeComputePipelineState(function: interleaveHalfFn)
        self.interleaveFloatPipeline = try device.makeComputePipelineState(function: interleaveFloatFn)
        self.packFusedHalfPipeline = try device.makeComputePipelineState(function: packFusedHalfFn)
        self.packFusedFloatPipeline = try device.makeComputePipelineState(function: packFusedFloatFn)
        self.renderFusedHalfPipeline = try device.makeComputePipelineState(function: renderFusedHalfFn)
        self.renderFusedFloatPipeline = try device.makeComputePipelineState(function: renderFusedFloatFn)
        self.renderFusedMultiPixelHalfPipeline = try device.makeComputePipelineState(function: renderFusedMultiPixelHalfFn)
        self.renderFusedMultiPixelFloatPipeline = try device.makeComputePipelineState(function: renderFusedMultiPixelFloatFn)
        self.prepareDispatchPipeline = try device.makeComputePipelineState(function: prepFn)
        self.clearTexturesPipeline = try device.makeComputePipelineState(function: clearTexFn)

        // V2 render kernel (optional - may not exist in older metallibs)
        if let v2Fn = renderFusedV2HalfFn {
            self.renderFusedV2HalfPipeline = try device.makeComputePipelineState(function: v2Fn)
        } else {
            self.renderFusedV2HalfPipeline = nil
        }

        // V3 render kernel (local sort style)
        if let v3Fn = renderFusedV3HalfFn {
            self.renderFusedV3HalfPipeline = try device.makeComputePipelineState(function: v3Fn)
        } else {
            self.renderFusedV3HalfPipeline = nil
        }

        // Compute threadgroup sizes
        self.interleaveThreadgroupSize = max(1, min(interleaveHalfPipeline.maxTotalThreadsPerThreadgroup, 256))
        self.packThreadgroupSize = max(1, min(packFusedHalfPipeline.maxTotalThreadsPerThreadgroup, 256))
        // Render uses 16x16 threadgroup for 16x16 tiles
        self.renderThreadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        // Multi-pixel render uses 8x8 threadgroup for 32x16 tiles (4x2 pixels per thread)
        self.renderMultiPixelThreadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
        // V2 render uses 16x8 threadgroup for 32x16 tiles (2x2 pixels per thread)
        self.renderV2ThreadgroupSize = MTLSize(width: 16, height: 8, depth: 1)
        // V3 render uses 8x8 threadgroup for 32x16 tiles (4x2 pixels per thread) - local sort style
        self.renderV3ThreadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
    }

    // MARK: - Interleave

    /// Interleave separate gaussian buffers into single GaussianRenderData struct
    func encodeInterleave(
        commandBuffer: MTLCommandBuffer,
        gaussianBuffers: GaussianInputBuffers,
        interleavedOutput: MTLBuffer,
        gaussianCount: Int,
        precision: Precision
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "InterleaveGaussians"

        let pipeline = precision == .float16 ? interleaveHalfPipeline : interleaveFloatPipeline
        encoder.setComputePipelineState(pipeline)

        encoder.setBuffer(gaussianBuffers.means, offset: 0, index: 0)
        encoder.setBuffer(gaussianBuffers.conics, offset: 0, index: 1)
        encoder.setBuffer(gaussianBuffers.colors, offset: 0, index: 2)
        encoder.setBuffer(gaussianBuffers.opacities, offset: 0, index: 3)
        encoder.setBuffer(gaussianBuffers.depths, offset: 0, index: 4)
        encoder.setBuffer(interleavedOutput, offset: 0, index: 5)

        var count = UInt32(gaussianCount)
        encoder.setBytes(&count, length: 4, index: 6)

        let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: interleaveThreadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)

        encoder.endEncoding()
    }

    // MARK: - Pack Fused

    /// Pack interleaved gaussians using sorted indices
    /// Uses indirect dispatch from dispatchArgs buffer
    func encodePackFused(
        commandBuffer: MTLCommandBuffer,
        sortedIndices: MTLBuffer,
        interleavedGaussians: MTLBuffer,
        packedOutput: MTLBuffer,
        header: MTLBuffer,
        dispatchArgs: MTLBuffer,
        dispatchOffset: Int,
        precision: Precision
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PackFused"

        let pipeline = precision == .float16 ? packFusedHalfPipeline : packFusedFloatPipeline
        encoder.setComputePipelineState(pipeline)

        encoder.setBuffer(sortedIndices, offset: 0, index: 0)
        encoder.setBuffer(interleavedGaussians, offset: 0, index: 1)
        encoder.setBuffer(packedOutput, offset: 0, index: 2)
        encoder.setBuffer(header, offset: 0, index: 3)

        let tg = MTLSize(width: packThreadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreadgroups(
            indirectBuffer: dispatchArgs,
            indirectBufferOffset: dispatchOffset,
            threadsPerThreadgroup: tg
        )

        encoder.endEncoding()
    }

    // MARK: - Render Fused

    /// Render using interleaved PackedGaussian structs
    /// Uses indirect dispatch for active tiles only
    /// Automatically selects multi-pixel kernel for 32x16 tiles
    func encodeRenderFused(
        commandBuffer: MTLCommandBuffer,
        headers: MTLBuffer,
        packedGaussians: MTLBuffer,
        activeTileIndices: MTLBuffer,
        activeTileCount: MTLBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        alphaTexture: MTLTexture,
        params: RenderParams,
        dispatchArgs: MTLBuffer,
        dispatchOffset: Int,
        precision: Precision
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        // Detect multi-pixel mode: 32x16 tiles
        let isMultiPixel = params.tileWidth == 32 && params.tileHeight == 16

        // Select kernel and threadgroup size based on renderVersion
        let threadgroupSize: MTLSize

        if isMultiPixel {
            // Try V3 first (local sort style: no shared mem, 8 pixels/thread)
            if renderVersion >= 3, precision == .float16, let v3Pipeline = renderFusedV3HalfPipeline {
                encoder.label = "RenderFusedV3"
                encoder.setComputePipelineState(v3Pipeline)
                threadgroupSize = renderV3ThreadgroupSize  // 8x8 = 64 threads
            }
            // Then try V2 (shared mem, 4 pixels/thread)
            else if renderVersion >= 2, precision == .float16, let v2Pipeline = renderFusedV2HalfPipeline {
                encoder.label = "RenderFusedV2"
                encoder.setComputePipelineState(v2Pipeline)
                threadgroupSize = renderV2ThreadgroupSize  // 16x8 = 128 threads
            }
            // Fallback to original multi-pixel
            else {
                encoder.label = "RenderFusedMultiPixel"
                let pipeline = precision == .float16 ? renderFusedMultiPixelHalfPipeline : renderFusedMultiPixelFloatPipeline
                encoder.setComputePipelineState(pipeline)
                threadgroupSize = renderMultiPixelThreadgroupSize  // 8x8 = 64 threads
            }
        } else {
            encoder.label = "RenderFused"
            let pipeline = precision == .float16 ? renderFusedHalfPipeline : renderFusedFloatPipeline
            encoder.setComputePipelineState(pipeline)
            threadgroupSize = renderThreadgroupSize  // 16x16 = 256 threads
        }

        encoder.setBuffer(headers, offset: 0, index: 0)
        encoder.setBuffer(packedGaussians, offset: 0, index: 1)
        encoder.setBuffer(activeTileIndices, offset: 0, index: 2)
        encoder.setBuffer(activeTileCount, offset: 0, index: 3)

        encoder.setTexture(colorTexture, index: 0)
        encoder.setTexture(depthTexture, index: 1)
        encoder.setTexture(alphaTexture, index: 2)

        var renderParams = params
        encoder.setBytes(&renderParams, length: MemoryLayout<RenderParams>.stride, index: 4)

        // Dispatch one threadgroup per active tile
        encoder.dispatchThreadgroups(
            indirectBuffer: dispatchArgs,
            indirectBufferOffset: dispatchOffset,
            threadsPerThreadgroup: threadgroupSize
        )

        encoder.endEncoding()
    }

    // MARK: - Complete Fused Render (prep + clear + render)

    /// Complete fused render: prepares dispatch, clears textures, and renders
    func encodeCompleteFusedRender(
        commandBuffer: MTLCommandBuffer,
        orderedBuffers: OrderedGaussianBuffers,
        outputTextures: RenderOutputTextures,
        params: RenderParams,
        dispatchArgs: MTLBuffer,
        dispatchOffset: Int,
        precision: Precision
    ) {
        guard let packedFused = orderedBuffers.packedGaussiansFused else { return }

        // 1. Prepare dispatch args from active tile count
        prepareDispatch(
            commandBuffer: commandBuffer,
            activeTileCount: orderedBuffers.activeTileCount,
            dispatchArgs: dispatchArgs
        )

        // 2. Clear output textures
        // Note: Can't skip - inactive tiles would have garbage.
        // TODO: Could optimize by only clearing inactive tiles, or using blit instead of compute
        clearTextures(
            commandBuffer: commandBuffer,
            outputTextures: outputTextures,
            params: params
        )

        // 3. Fused render
        encodeRenderFused(
            commandBuffer: commandBuffer,
            headers: orderedBuffers.headers,
            packedGaussians: packedFused,
            activeTileIndices: orderedBuffers.activeTileIndices,
            activeTileCount: orderedBuffers.activeTileCount,
            colorTexture: outputTextures.color,
            depthTexture: outputTextures.depth,
            alphaTexture: outputTextures.alpha,
            params: params,
            dispatchArgs: dispatchArgs,
            dispatchOffset: dispatchOffset,
            precision: precision
        )
    }

    // MARK: - Private Helpers

    private func prepareDispatch(
        commandBuffer: MTLCommandBuffer,
        activeTileCount: MTLBuffer,
        dispatchArgs: MTLBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PrepareRenderDispatchFused"
        encoder.setComputePipelineState(prepareDispatchPipeline)
        encoder.setBuffer(activeTileCount, offset: 0, index: 0)
        encoder.setBuffer(dispatchArgs, offset: 0, index: 1)

        var dispatchParams = RenderDispatchParamsSwift(
            tileCount: 1000000, // Large enough to not clamp
            totalAssignments: 0,
            gaussianCount: 0
        )
        encoder.setBytes(&dispatchParams, length: MemoryLayout<RenderDispatchParamsSwift>.stride, index: 2)

        let threads = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: threads)
        encoder.endEncoding()
    }

    private func clearTextures(
        commandBuffer: MTLCommandBuffer,
        outputTextures: RenderOutputTextures,
        params: RenderParams
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ClearTexturesFused"
        encoder.setComputePipelineState(clearTexturesPipeline)
        encoder.setTexture(outputTextures.color, index: 0)
        encoder.setTexture(outputTextures.depth, index: 1)
        encoder.setTexture(outputTextures.alpha, index: 2)

        var clearParams = ClearTextureParamsSwift(
            width: params.width,
            height: params.height,
            whiteBackground: params.whiteBackground
        )
        encoder.setBytes(&clearParams, length: MemoryLayout<ClearTextureParamsSwift>.stride, index: 0)

        let w = Int(params.width)
        let h = Int(params.height)

        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroups = MTLSize(
            width: (w + 15) / 16,
            height: (h + 15) / 16,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}

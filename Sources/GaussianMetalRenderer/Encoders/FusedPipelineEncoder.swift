import Metal

/// Encoder for fused pipeline operations using interleaved data structures
/// Unified with LocalSort approach: index-based render, no alpha texture, half precision
final class FusedPipelineEncoder {
    // Pipeline states
    private let interleaveHalfPipeline: MTLComputePipelineState
    private let interleaveFloatPipeline: MTLComputePipelineState
    private let renderPipeline: MTLComputePipelineState  // Unified render (like LocalSort)
    private let prepareDispatchPipeline: MTLComputePipelineState
    private let clearTexturesPipeline: MTLComputePipelineState

    // Threadgroup sizes
    let interleaveThreadgroupSize: Int
    let renderThreadgroupSize: MTLSize  // 8x8 for 32x16 tiles (4x2 pixels/thread)

    init(device: MTLDevice, library: MTLLibrary) throws {
        // Interleave kernels
        guard let interleaveHalfFn = library.makeFunction(name: "interleaveGaussianDataKernelHalf"),
              let interleaveFloatFn = library.makeFunction(name: "interleaveGaussianDataKernelFloat")
        else {
            throw NSError(domain: "FusedPipelineEncoder", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Interleave kernel functions missing"])
        }

        // Unified render kernel (like LocalSort)
        guard let renderFn = library.makeFunction(name: "globalSortRender") else {
            throw NSError(domain: "FusedPipelineEncoder", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "Render kernel function missing"])
        }

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
        self.renderPipeline = try device.makeComputePipelineState(function: renderFn)
        self.prepareDispatchPipeline = try device.makeComputePipelineState(function: prepFn)
        self.clearTexturesPipeline = try device.makeComputePipelineState(function: clearTexFn)

        // Compute threadgroup sizes
        self.interleaveThreadgroupSize = max(1, min(interleaveHalfPipeline.maxTotalThreadsPerThreadgroup, 256))
        // Render uses 8x8 threadgroup for 32x16 tiles (4x2 pixels per thread)
        self.renderThreadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
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

    // MARK: - Render (unified, like LocalSort)

    /// Render using index-based access (like LocalSort)
    /// No Pack step needed - reads via sortedIndices directly
    /// Only outputs color + depth (alpha in color.a channel)
    func encodeRender(
        commandBuffer: MTLCommandBuffer,
        headers: MTLBuffer,
        interleavedGaussians: MTLBuffer,
        sortedIndices: MTLBuffer,
        activeTileIndices: MTLBuffer,
        activeTileCount: MTLBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        params: RenderParams,
        dispatchArgs: MTLBuffer,
        dispatchOffset: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.label = "GlobalSortRender"
        encoder.setComputePipelineState(renderPipeline)

        encoder.setBuffer(headers, offset: 0, index: 0)
        encoder.setBuffer(interleavedGaussians, offset: 0, index: 1)
        encoder.setBuffer(sortedIndices, offset: 0, index: 2)
        encoder.setBuffer(activeTileIndices, offset: 0, index: 3)
        encoder.setBuffer(activeTileCount, offset: 0, index: 4)

        encoder.setTexture(colorTexture, index: 0)
        encoder.setTexture(depthTexture, index: 1)

        var renderParams = params
        encoder.setBytes(&renderParams, length: MemoryLayout<RenderParams>.stride, index: 5)

        // Dispatch one threadgroup per active tile
        encoder.dispatchThreadgroups(
            indirectBuffer: dispatchArgs,
            indirectBufferOffset: dispatchOffset,
            threadsPerThreadgroup: renderThreadgroupSize
        )

        encoder.endEncoding()
    }

    // MARK: - Complete Render (prep + clear + render)

    /// Complete render: prepares dispatch, clears textures, and renders
    func encodeCompleteRender(
        commandBuffer: MTLCommandBuffer,
        orderedBuffers: OrderedGaussianBuffers,
        interleavedGaussians: MTLBuffer,
        sortedIndices: MTLBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        params: RenderParams,
        dispatchArgs: MTLBuffer,
        dispatchOffset: Int
    ) {
        // 1. Prepare dispatch args from active tile count
        prepareDispatch(
            commandBuffer: commandBuffer,
            activeTileCount: orderedBuffers.activeTileCount,
            dispatchArgs: dispatchArgs
        )

        // 2. Clear output textures
        clearTextures(
            commandBuffer: commandBuffer,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            params: params
        )

        // 3. Render
        encodeRender(
            commandBuffer: commandBuffer,
            headers: orderedBuffers.headers,
            interleavedGaussians: interleavedGaussians,
            sortedIndices: sortedIndices,
            activeTileIndices: orderedBuffers.activeTileIndices,
            activeTileCount: orderedBuffers.activeTileCount,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            params: params,
            dispatchArgs: dispatchArgs,
            dispatchOffset: dispatchOffset
        )
    }

    // MARK: - Private Helpers

    private func prepareDispatch(
        commandBuffer: MTLCommandBuffer,
        activeTileCount: MTLBuffer,
        dispatchArgs: MTLBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PrepareRenderDispatch"
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
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        params: RenderParams
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ClearTextures"
        encoder.setComputePipelineState(clearTexturesPipeline)
        encoder.setTexture(colorTexture, index: 0)
        encoder.setTexture(depthTexture, index: 1)

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

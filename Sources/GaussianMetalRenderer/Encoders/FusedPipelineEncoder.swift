import GaussianMetalRendererTypes
import Metal

/// Encoder for fused pipeline operations using AoS (Array of Structures) data
/// Unified with Local approach: index-based render, no alpha texture, half precision
/// NOTE: Interleave step is OBSOLETE - projection now outputs AoS directly via projectGaussiansAoS
final class FusedPipelineEncoder {
    // Pipeline states
    private let renderPipeline: MTLComputePipelineState // Unified render (like Local)
    private let prepareDispatchPipeline: MTLComputePipelineState
    private let clearTexturesPipeline: MTLComputePipelineState

    // Threadgroup sizes
    let renderThreadgroupSize: MTLSize // 8x8 for 32x16 tiles (4x2 pixels per thread)

    init(device: MTLDevice, library: MTLLibrary) throws {
        // Unified render kernel (like Local)
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
        self.renderPipeline = try device.makeComputePipelineState(function: renderFn)
        self.prepareDispatchPipeline = try device.makeComputePipelineState(function: prepFn)
        self.clearTexturesPipeline = try device.makeComputePipelineState(function: clearTexFn)

        // Render uses 8x8 threadgroup for 32x16 tiles (4x2 pixels per thread)
        self.renderThreadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
    }

    // MARK: - Render (unified, like Local)

    /// Render using index-based access (like Local)
    /// No Pack step needed - reads via sortedIndices directly
    /// Only outputs color + depth (alpha in color.a channel)
    /// renderData: AoS packed GaussianRenderData from projectGaussiansAoS
    func encodeRender(
        commandBuffer: MTLCommandBuffer,
        headers: MTLBuffer,
        renderData: MTLBuffer, // GaussianRenderData from AoS projection
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
        encoder.setComputePipelineState(self.renderPipeline)

        encoder.setBuffer(headers, offset: 0, index: 0)
        encoder.setBuffer(renderData, offset: 0, index: 1)
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
            threadsPerThreadgroup: self.renderThreadgroupSize
        )

        encoder.endEncoding()
    }

    // MARK: - Complete Render (prep + clear + render)

    /// Complete render: prepares dispatch, clears textures, and renders
    /// renderData: AoS packed GaussianRenderData from projectGaussiansAoS
    func encodeCompleteRender(
        commandBuffer: MTLCommandBuffer,
        orderedBuffers: OrderedGaussianBuffers,
        renderData: MTLBuffer, // GaussianRenderData from AoS projection
        sortedIndices: MTLBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        params: RenderParams,
        dispatchArgs: MTLBuffer,
        dispatchOffset: Int
    ) {
        // 1. Prepare dispatch args from active tile count
        self.prepareDispatch(
            commandBuffer: commandBuffer,
            activeTileCount: orderedBuffers.activeTileCount,
            dispatchArgs: dispatchArgs
        )

        // 2. Clear output textures
        self.clearTextures(
            commandBuffer: commandBuffer,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            params: params
        )

        // 3. Render
        self.encodeRender(
            commandBuffer: commandBuffer,
            headers: orderedBuffers.headers,
            renderData: renderData,
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
        encoder.setComputePipelineState(self.prepareDispatchPipeline)
        encoder.setBuffer(activeTileCount, offset: 0, index: 0)
        encoder.setBuffer(dispatchArgs, offset: 0, index: 1)

        var dispatchParams = RenderDispatchParamsSwift(
            tileCount: 1_000_000, // Large enough to not clamp
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
        encoder.setComputePipelineState(self.clearTexturesPipeline)
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

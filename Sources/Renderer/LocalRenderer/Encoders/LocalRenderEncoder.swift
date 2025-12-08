import Metal
import RendererTypes

final class LocalRenderEncoder {
    // Indirect dispatch pipelines
    private let clearTexturesPipeline: MTLComputePipelineState
    private let prepareRenderDispatchPipeline: MTLComputePipelineState
    private let renderIndirect16Pipeline: MTLComputePipelineState

    init(library: MTLLibrary, device: MTLDevice) throws {
        // Required indirect dispatch pipelines
        guard let clearFn = library.makeFunction(name: "localClearTextures"),
              let prepareFn = library.makeFunction(name: "localPrepareRenderDispatch")
        else {
            throw RendererError.failedToCreatePipeline("Missing required indirect render kernels")
        }
        self.clearTexturesPipeline = try device.makeComputePipelineState(function: clearFn)
        self.prepareRenderDispatchPipeline = try device.makeComputePipelineState(function: prepareFn)

        // 16-bit render pipeline
        guard let renderInd16Fn = library.makeFunction(name: "localRender16") else {
            throw RendererError.failedToCreatePipeline("Missing localRender16 kernel")
        }
        self.renderIndirect16Pipeline = try device.makeComputePipelineState(function: renderInd16Fn)
    }

    func encodeClearTextures(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        width: Int,
        height: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        var w = UInt32(width)
        var h = UInt32(height)

        encoder.label = "Local_ClearTextures"
        encoder.setComputePipelineState(self.clearTexturesPipeline)
        encoder.setTexture(colorTexture, index: 0)
        if let depthTexture { encoder.setTexture(depthTexture, index: 1) }
        encoder.setBytes(&w, length: MemoryLayout<UInt32>.stride, index: 0)
        encoder.setBytes(&h, length: MemoryLayout<UInt32>.stride, index: 1)

        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width: (width + 15) / 16,
            height: (height + 15) / 16,
            depth: 1
        )
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    /// Prepare indirect dispatch arguments from active tile count
    func encodePrepareRenderDispatch(
        commandBuffer: MTLCommandBuffer,
        activeTileCount: MTLBuffer,
        dispatchArgs: MTLBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.label = "Local_PrepareRenderDispatch"
        encoder.setComputePipelineState(self.prepareRenderDispatchPipeline)
        encoder.setBuffer(activeTileCount, offset: 0, index: 0)
        encoder.setBuffer(dispatchArgs, offset: 0, index: 1)

        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// Encode indirect render pass (16-bit indices, fixed layout)
    func encodeIndirect16(
        commandBuffer: MTLCommandBuffer,
        projectedGaussians: MTLBuffer,
        tileCounts: MTLBuffer,
        maxPerTile: Int,
        sortedLocalIdx: MTLBuffer,
        globalIndices: MTLBuffer,
        activeTileIndices: MTLBuffer,
        dispatchArgs: MTLBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        tilesX: Int,
        tilesY: Int,
        width: Int,
        height: Int,
        tileWidth: Int,
        tileHeight: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var maxPerTileU = UInt32(maxPerTile)

        var params = RenderParams(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            maxPerTile: UInt32(maxPerTile),
            activeTileCount: 0,
            gaussianCount: 0
        )

        encoder.label = "Local_RenderIndirect16"
        encoder.setComputePipelineState(renderIndirect16Pipeline)
        encoder.setBuffer(projectedGaussians, offset: 0, index: 0)
        encoder.setBuffer(tileCounts, offset: 0, index: 1)
        encoder.setBytes(&maxPerTileU, length: 4, index: 2)
        encoder.setBuffer(sortedLocalIdx, offset: 0, index: 3)
        encoder.setBuffer(globalIndices, offset: 0, index: 4)
        encoder.setBuffer(activeTileIndices, offset: 0, index: 5)
        encoder.setTexture(colorTexture, index: 0)
        if let depthTexture { encoder.setTexture(depthTexture, index: 1) }
        encoder.setBytes(&params, length: MemoryLayout<RenderParams>.stride, index: 6)

        // 4×8 threadgroup for 16×16 tile (4×2 pixels per thread)
        let tg = MTLSize(width: 4, height: 8, depth: 1)
        encoder.dispatchThreadgroups(indirectBuffer: dispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

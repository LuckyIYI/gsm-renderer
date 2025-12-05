import Metal

/// Encodes the render stage - blends sorted gaussians to output textures (indirect dispatch only)
public final class LocalRenderEncoder {
    // Indirect dispatch pipelines
    private let clearTexturesPipeline: MTLComputePipelineState
    private let prepareRenderDispatchPipeline: MTLComputePipelineState
    private let renderIndirectPipeline: MTLComputePipelineState
    private let renderIndirect16Pipeline: MTLComputePipelineState?

    /// Texture width for render texture (must match RENDER_TEX_WIDTH in shader)
    public static let renderTexWidth = 4096

    public init(library: MTLLibrary, device: MTLDevice) throws {
        // Required indirect dispatch pipelines
        guard let clearFn = library.makeFunction(name: "LocalClearTextures"),
              let prepareFn = library.makeFunction(name: "LocalPrepareRenderDispatch")
        else {
            fatalError("Missing required indirect render kernels")
        }
        self.clearTexturesPipeline = try device.makeComputePipelineState(function: clearFn)
        self.prepareRenderDispatchPipeline = try device.makeComputePipelineState(function: prepareFn)

        // Create function constant values for 32-bit render (USE_16BIT_RENDER = false)
        let constantValues32 = MTLFunctionConstantValues()
        var use16Bit = false
        constantValues32.setConstantValue(&use16Bit, type: .bool, index: 2)

        guard let renderIndFn = try? library.makeFunction(name: "LocalRenderIndirect", constantValues: constantValues32) else {
            fatalError("Failed to create LocalRenderIndirect function")
        }
        self.renderIndirectPipeline = try device.makeComputePipelineState(function: renderIndFn)

        // Create function constant values for 16-bit render (USE_16BIT_RENDER = true)
        let constantValues16 = MTLFunctionConstantValues()
        var use16BitTrue = true
        constantValues16.setConstantValue(&use16BitTrue, type: .bool, index: 2)

        if let renderInd16Fn = try? library.makeFunction(name: "LocalRenderIndirect16", constantValues: constantValues16) {
            self.renderIndirect16Pipeline = try? device.makeComputePipelineState(function: renderInd16Fn)
        } else {
            self.renderIndirect16Pipeline = nil
        }
    }

    /// Check if 16-bit render is available
    public var has16BitRender: Bool { self.renderIndirect16Pipeline != nil }

    // MARK: - Indirect Dispatch Methods

    /// Clear color and depth textures (for indirect dispatch path)
    public func encodeClearTextures(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        width: Int,
        height: Int,
        whiteBackground: Bool
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        var w = UInt32(width)
        var h = UInt32(height)
        var bg: UInt32 = whiteBackground ? 1 : 0

        encoder.label = "Local_ClearTextures"
        encoder.setComputePipelineState(self.clearTexturesPipeline)
        encoder.setTexture(colorTexture, index: 0)
        encoder.setTexture(depthTexture, index: 1)
        encoder.setBytes(&w, length: MemoryLayout<UInt32>.stride, index: 0)
        encoder.setBytes(&h, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBytes(&bg, length: MemoryLayout<UInt32>.stride, index: 2)

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
    public func encodePrepareRenderDispatch(
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

    /// Encode indirect render pass (only active tiles, 32-bit indices, fixed layout)
    public func encodeIndirect(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        tileCounts: MTLBuffer,
        maxPerTile: Int,
        sortedIndices: MTLBuffer,
        activeTileIndices: MTLBuffer,
        dispatchArgs: MTLBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        tilesX: Int,
        tilesY: Int,
        width: Int,
        height: Int,
        tileWidth: Int,
        tileHeight: Int,
        whiteBackground: Bool = false
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var maxPerTileU = UInt32(maxPerTile)

        var params = LocalRenderParamsSwift(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            maxPerTile: UInt32(maxPerTile),
            whiteBackground: whiteBackground ? 1 : 0,
            activeTileCount: 0,
            gaussianCount: 0
        )

        encoder.label = "Local_RenderIndirect"
        encoder.setComputePipelineState(self.renderIndirectPipeline)
        // Fixed layout buffer bindings:
        // buffer(0): compacted, buffer(1): counts, buffer(2): maxPerTile
        // buffer(3): sortBuffer, buffer(4): globalIndices (unused), buffer(5): activeTileIndices
        // buffer(6): params
        encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
        encoder.setBuffer(tileCounts, offset: 0, index: 1)
        encoder.setBytes(&maxPerTileU, length: 4, index: 2)
        encoder.setBuffer(sortedIndices, offset: 0, index: 3)
        // buffer(4) unused in 32-bit path
        encoder.setBuffer(activeTileIndices, offset: 0, index: 5)
        encoder.setTexture(colorTexture, index: 0)
        encoder.setTexture(depthTexture, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<LocalRenderParamsSwift>.stride, index: 6)

        // 4×8 threadgroup for 16×16 tile (4×2 pixels per thread)
        let tg = MTLSize(width: 4, height: 8, depth: 1)
        encoder.dispatchThreadgroups(indirectBuffer: dispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }

    /// Encode indirect render pass (only active tiles, 16-bit indices, fixed layout)
    public func encodeIndirect16(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        tileCounts: MTLBuffer,
        maxPerTile: Int,
        sortedLocalIdx: MTLBuffer,
        globalIndices: MTLBuffer,
        activeTileIndices: MTLBuffer,
        dispatchArgs: MTLBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        tilesX: Int,
        tilesY: Int,
        width: Int,
        height: Int,
        tileWidth: Int,
        tileHeight: Int,
        whiteBackground: Bool = false
    ) {
        guard let pipeline = renderIndirect16Pipeline else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var maxPerTileU = UInt32(maxPerTile)

        var params = LocalRenderParamsSwift(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            maxPerTile: UInt32(maxPerTile),
            whiteBackground: whiteBackground ? 1 : 0,
            activeTileCount: 0,
            gaussianCount: 0
        )

        encoder.label = "Local_RenderIndirect16"
        encoder.setComputePipelineState(pipeline)
        // Fixed layout buffer bindings:
        // buffer(0): compacted, buffer(1): counts, buffer(2): maxPerTile
        // buffer(3): sortBuffer (ushort*), buffer(4): globalIndices, buffer(5): activeTileIndices
        // buffer(6): params
        encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
        encoder.setBuffer(tileCounts, offset: 0, index: 1)
        encoder.setBytes(&maxPerTileU, length: 4, index: 2)
        encoder.setBuffer(sortedLocalIdx, offset: 0, index: 3)
        encoder.setBuffer(globalIndices, offset: 0, index: 4)
        encoder.setBuffer(activeTileIndices, offset: 0, index: 5)
        encoder.setTexture(colorTexture, index: 0)
        encoder.setTexture(depthTexture, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<LocalRenderParamsSwift>.stride, index: 6)

        // 4×8 threadgroup for 16×16 tile (4×2 pixels per thread)
        let tg = MTLSize(width: 4, height: 8, depth: 1)
        encoder.dispatchThreadgroups(indirectBuffer: dispatchArgs, indirectBufferOffset: 0, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

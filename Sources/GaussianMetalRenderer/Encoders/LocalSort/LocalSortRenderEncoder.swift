import Metal

/// Encodes the render stage - blends sorted gaussians to output textures
public final class LocalSortRenderEncoder {
    private let renderPipeline: MTLComputePipelineState
    private let render16Pipeline: MTLComputePipelineState?
    private let packRenderTexturePipeline: MTLComputePipelineState?
    private let renderTexturedPipeline: MTLComputePipelineState?

    /// Texture width for render texture (must match RENDER_TEX_WIDTH in shader)
    public static let renderTexWidth = 4096

    public init(library: MTLLibrary, device: MTLDevice) throws {
        guard let renderFn = library.makeFunction(name: "localSortRender") else {
            fatalError("Missing render kernel")
        }
        self.renderPipeline = try device.makeComputePipelineState(function: renderFn)

        // Optional 16-bit render
        if let render16Fn = library.makeFunction(name: "localSortRender16") {
            self.render16Pipeline = try? device.makeComputePipelineState(function: render16Fn)
        } else {
            self.render16Pipeline = nil
        }

        // Optional texture-cached render
        if let packFn = library.makeFunction(name: "localSortPackRenderTexture"),
           let renderTexFn = library.makeFunction(name: "localSortRenderTextured") {
            self.packRenderTexturePipeline = try? device.makeComputePipelineState(function: packFn)
            self.renderTexturedPipeline = try? device.makeComputePipelineState(function: renderTexFn)
        } else {
            self.packRenderTexturePipeline = nil
            self.renderTexturedPipeline = nil
        }
    }

    /// Check if 16-bit render is available
    public var has16BitRender: Bool { render16Pipeline != nil }

    /// Check if textured render is available
    public var hasTexturedRender: Bool {
        packRenderTexturePipeline != nil && renderTexturedPipeline != nil
    }

    /// Encode standard render pass
    public func encode(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        tileOffsets: MTLBuffer,
        tileCounts: MTLBuffer,
        sortedIndices: MTLBuffer,
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
        var params = LocalSortRenderParamsSwift(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            maxPerTile: 0,
            whiteBackground: whiteBackground ? 1 : 0,
            activeTileCount: 0,
            gaussianCount: 0
        )

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "LocalSort_Render"
        encoder.setComputePipelineState(renderPipeline)
        encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
        encoder.setBuffer(tileOffsets, offset: 0, index: 1)
        encoder.setBuffer(tileCounts, offset: 0, index: 2)
        encoder.setBuffer(sortedIndices, offset: 0, index: 3)
        encoder.setTexture(colorTexture, index: 0)
        encoder.setTexture(depthTexture, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<LocalSortRenderParamsSwift>.stride, index: 4)

        // 4×8 threadgroup for 16×16 tile (4×2 pixels per thread)
        let tgs = MTLSize(width: tilesX, height: tilesY, depth: 1)
        let tg = MTLSize(width: 4, height: 8, depth: 1)
        encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }

    /// Encode 16-bit render (two-level indirection)
    public func encode16(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        tileOffsets: MTLBuffer,
        tileCounts: MTLBuffer,
        sortedLocalIdx: MTLBuffer,
        globalIndices: MTLBuffer,
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
        guard let render16 = render16Pipeline else { return }

        var params = LocalSortRenderParamsSwift(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            maxPerTile: 0,
            whiteBackground: whiteBackground ? 1 : 0,
            activeTileCount: 0,
            gaussianCount: 0
        )

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "LocalSort_Render16"
        encoder.setComputePipelineState(render16)
        encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
        encoder.setBuffer(tileOffsets, offset: 0, index: 1)
        encoder.setBuffer(tileCounts, offset: 0, index: 2)
        encoder.setBuffer(sortedLocalIdx, offset: 0, index: 3)
        encoder.setBuffer(globalIndices, offset: 0, index: 4)
        encoder.setTexture(colorTexture, index: 0)
        encoder.setTexture(depthTexture, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<LocalSortRenderParamsSwift>.stride, index: 5)

        let tgs = MTLSize(width: tilesX, height: tilesY, depth: 1)
        let tg = MTLSize(width: 4, height: 8, depth: 1)
        encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }

    /// Encode pack pass to copy gaussian data to texture (for TLB optimization)
    public func encodePackRenderTexture(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        renderTexture: MTLTexture,
        maxGaussians: Int
    ) {
        guard let packPipeline = packRenderTexturePipeline else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.label = "LocalSort_PackRenderTexture"
        encoder.setComputePipelineState(packPipeline)
        encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
        encoder.setBuffer(compactedHeader, offset: 0, index: 1)
        encoder.setTexture(renderTexture, index: 0)

        let threadsPerGroup = 256
        let numGroups = (maxGaussians + threadsPerGroup - 1) / threadsPerGroup
        encoder.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Encode textured render (uses texture-cached gaussian data)
    public func encodeTextured(
        commandBuffer: MTLCommandBuffer,
        gaussianTexture: MTLTexture,
        tileOffsets: MTLBuffer,
        tileCounts: MTLBuffer,
        sortedIndices: MTLBuffer,
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
        guard let renderTexPipeline = renderTexturedPipeline else { return }

        var params = LocalSortRenderParamsSwift(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            maxPerTile: 0,
            whiteBackground: whiteBackground ? 1 : 0,
            activeTileCount: 0,
            gaussianCount: 0
        )

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "LocalSort_RenderTextured"
        encoder.setComputePipelineState(renderTexPipeline)
        encoder.setTexture(gaussianTexture, index: 0)
        encoder.setBuffer(tileOffsets, offset: 0, index: 0)
        encoder.setBuffer(tileCounts, offset: 0, index: 1)
        encoder.setBuffer(sortedIndices, offset: 0, index: 2)
        encoder.setTexture(colorTexture, index: 1)
        encoder.setTexture(depthTexture, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<LocalSortRenderParamsSwift>.stride, index: 3)

        let tgs = MTLSize(width: tilesX, height: tilesY, depth: 1)
        let tg = MTLSize(width: 4, height: 8, depth: 1)
        encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

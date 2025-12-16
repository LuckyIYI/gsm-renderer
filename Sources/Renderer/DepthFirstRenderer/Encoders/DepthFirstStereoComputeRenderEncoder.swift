import Metal
import RendererTypes

/// Encoder for depth-first stereo blending using a single compute pass.
/// Writes both eyes into a 2-slice intermediate texture array.
final class DepthFirstStereoComputeRenderEncoder {
    private let renderPipeline: MTLComputePipelineState
    private let clearPipeline: MTLComputePipelineState
    let threadgroupSize: MTLSize

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let renderFn = library.makeFunction(name: "depthFirstStereoRender"),
              let clearFn = library.makeFunction(name: "clearStereoRenderTextureKernel")
        else {
            throw RendererError.failedToCreatePipeline("Depth-first stereo render kernels not found")
        }

        self.renderPipeline = try device.makeComputePipelineState(function: renderFn)
        self.clearPipeline = try device.makeComputePipelineState(function: clearFn)
        self.threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
    }

    func encodeClear(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        width: Int,
        height: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ClearStereoIntermediate"
        encoder.setComputePipelineState(self.clearPipeline)
        encoder.setTexture(colorTexture, index: 0)

        var params = ClearTextureParams()
        params.width = UInt32(width)
        params.height = UInt32(height)
        encoder.setBytes(&params, length: MemoryLayout<ClearTextureParams>.stride, index: 0)

        let threads = MTLSize(width: width, height: height, depth: 2)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }

    func encodeRender(
        commandBuffer: MTLCommandBuffer,
        tileHeaders: MTLBuffer,
        renderData: MTLBuffer, // StereoTiledRenderData
        sortedGaussianIndices: MTLBuffer,
        activeTiles: MTLBuffer,
        activeTileCount: MTLBuffer,
        colorTexture: MTLTexture,
        params: RenderParams,
        dispatchArgs: MTLBuffer,
        dispatchOffset: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "DepthFirstStereoRender"

        encoder.setComputePipelineState(self.renderPipeline)
        encoder.setBuffer(tileHeaders, offset: 0, index: 0)
        encoder.setBuffer(renderData, offset: 0, index: 1)
        encoder.setBuffer(sortedGaussianIndices, offset: 0, index: 2)
        encoder.setBuffer(activeTiles, offset: 0, index: 3)
        encoder.setBuffer(activeTileCount, offset: 0, index: 4)

        encoder.setTexture(colorTexture, index: 0)

        var renderParams = params
        encoder.setBytes(&renderParams, length: MemoryLayout<RenderParams>.stride, index: 5)

        encoder.dispatchThreadgroups(
            indirectBuffer: dispatchArgs,
            indirectBufferOffset: dispatchOffset,
            threadsPerThreadgroup: self.threadgroupSize
        )
        encoder.endEncoding()
    }
}

import Metal
import RendererTypes

/// Encoder for depth-first rendering
final class DepthFirstRenderEncoder {
    private let renderPipeline: MTLComputePipelineState
    private let clearPipeline: MTLComputePipelineState
    let threadgroupSize: MTLSize

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let renderFn = library.makeFunction(name: "depthFirstRender"),
              let clearFn = library.makeFunction(name: "clearRenderTexturesKernel")
        else {
            throw RendererError.failedToCreatePipeline("Depth-first render kernels not found")
        }

        self.renderPipeline = try device.makeComputePipelineState(function: renderFn)
        self.clearPipeline = try device.makeComputePipelineState(function: clearFn)

        // 8x8 threads per tile (32x16 tile, 4x2 pixels per thread)
        self.threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
    }

    /// Clear render textures to background color
    func encodeClear(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        width: Int,
        height: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ClearRenderTextures"
        encoder.setComputePipelineState(self.clearPipeline)
        encoder.setTexture(colorTexture, index: 0)
        if let depthTex = depthTexture, depthTex.pixelFormat != .depth32Float {
            encoder.setTexture(depthTex, index: 1)
        }

        var params = ClearTextureParams()
        params.width = UInt32(width)
        params.height = UInt32(height)
        encoder.setBytes(&params, length: MemoryLayout<ClearTextureParams>.stride, index: 0)

        let threads = MTLSize(width: width, height: height, depth: 1)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }

    /// Render tiles using depth-sorted gaussian indices (indirect dispatch)
    func encodeRender(
        commandBuffer: MTLCommandBuffer,
        tileHeaders: MTLBuffer,
        renderData: MTLBuffer,
        sortedGaussianIndices: MTLBuffer,
        activeTiles: MTLBuffer,
        activeTileCount: MTLBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture?,
        params: RenderParams,
        dispatchArgs: MTLBuffer,
        dispatchOffset: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "DepthFirstRender"

        encoder.setComputePipelineState(self.renderPipeline)
        encoder.setBuffer(tileHeaders, offset: 0, index: 0)
        encoder.setBuffer(renderData, offset: 0, index: 1)
        encoder.setBuffer(sortedGaussianIndices, offset: 0, index: 2)
        encoder.setBuffer(activeTiles, offset: 0, index: 3)
        encoder.setBuffer(activeTileCount, offset: 0, index: 4)

        encoder.setTexture(colorTexture, index: 0)
        if let depthTex = depthTexture, depthTex.pixelFormat != .depth32Float {
            encoder.setTexture(depthTex, index: 1)
        }

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

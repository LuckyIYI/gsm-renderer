import Metal
import GaussianMetalRendererTypes

final class TileBoundsEncoder {
    private let pipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let function = library.makeFunction(name: "tileBoundsKernel") else {
            fatalError("tileBoundsKernel not found")
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    /// Reads mean from packed GaussianRenderData
    func encode(
        commandBuffer: MTLCommandBuffer,
        renderData: MTLBuffer,
        radii: MTLBuffer,
        mask: MTLBuffer,
        boundsBuffer: MTLBuffer,
        params: RenderParams,
        gaussianCount: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "TileBounds"

        var paramsStruct = TileBoundsParamsSwift(
            width: params.width,
            height: params.height,
            tileWidth: params.tileWidth,
            tileHeight: params.tileHeight,
            tilesX: params.tilesX,
            tilesY: params.tilesY,
            gaussianCount: UInt32(gaussianCount)
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(renderData, offset: 0, index: 0)
        encoder.setBuffer(radii, offset: 0, index: 1)
        encoder.setBuffer(mask, offset: 0, index: 2)
        encoder.setBuffer(boundsBuffer, offset: 0, index: 3)
        encoder.setBytes(&paramsStruct, length: MemoryLayout<TileBoundsParamsSwift>.stride, index: 4)

        let tgWidth = pipeline.threadExecutionWidth
        let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

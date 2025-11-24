import Metal

final class TileBoundsEncoder {
    private let pipeline: MTLComputePipelineState
    private let pipelineHalf: MTLComputePipelineState?

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let function = library.makeFunction(name: "tileBoundsKernel_float") else {
            fatalError("tileBoundsKernel_float not found")
        }
        self.pipeline = try device.makeComputePipelineState(function: function)

        if let functionHalf = library.makeFunction(name: "tileBoundsKernel_half") {
            self.pipelineHalf = try? device.makeComputePipelineState(function: functionHalf)
        } else {
            self.pipelineHalf = nil
        }
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianBuffers: GaussianInputBuffers,
        boundsBuffer: MTLBuffer,
        params: RenderParams,
        gaussianCount: Int,
        precision: Precision = .float32
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

        if precision == .float16, let halfPipe = self.pipelineHalf {
            encoder.setComputePipelineState(halfPipe)
        } else {
            encoder.setComputePipelineState(self.pipeline)
        }
        encoder.setBuffer(gaussianBuffers.means, offset: 0, index: 0)
        encoder.setBuffer(gaussianBuffers.radii, offset: 0, index: 1)
        encoder.setBuffer(gaussianBuffers.mask, offset: 0, index: 2)
        encoder.setBuffer(boundsBuffer, offset: 0, index: 3)
        encoder.setBytes(&paramsStruct, length: MemoryLayout<TileBoundsParamsSwift>.stride, index: 4)

        let activePipeline = (precision == .float16 && self.pipelineHalf != nil) ? self.pipelineHalf! : self.pipeline
        let tgWidth = activePipeline.threadExecutionWidth
        let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }

    // Overload to retain backwards compatibility when count is embedded in params.
    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianBuffers: GaussianInputBuffers,
        boundsBuffer: MTLBuffer,
        params: RenderParams,
    ) {
        let count = Int(params.activeTileCount)
        self.encode(
            commandBuffer: commandBuffer,
            gaussianBuffers: gaussianBuffers,
            boundsBuffer: boundsBuffer,
            params: params,
            gaussianCount: count,
            precision: .float32
        )
    }
}

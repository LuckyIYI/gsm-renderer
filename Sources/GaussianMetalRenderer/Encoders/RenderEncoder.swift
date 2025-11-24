import Metal

final class RenderEncoder {
    private let renderPipeline: MTLComputePipelineState
    private let renderPipelineHalf: MTLComputePipelineState?
    private let prepareDispatchPipeline: MTLComputePipelineState
    private let clearPipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard
            let renderFn = library.makeFunction(name: "renderTiles"),
            let renderHalfFn = library.makeFunction(name: "renderTilesHalf"),
            let prepFn = library.makeFunction(name: "prepareRenderDispatchKernel"),
            let clearFn = library.makeFunction(name: "clearRenderTargetsKernel")
        else {
            fatalError("Render functions missing")
        }
        
        self.renderPipeline = try device.makeComputePipelineState(function: renderFn)
        self.renderPipelineHalf = try? device.makeComputePipelineState(function: renderHalfFn)
        self.prepareDispatchPipeline = try device.makeComputePipelineState(function: prepFn)
        self.clearPipeline = try device.makeComputePipelineState(function: clearFn)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        orderedBuffers: OrderedGaussianBuffers,
        outputBuffers: RenderOutputBuffers,
        params: RenderParams,
        dispatchArgs: MTLBuffer,
        dispatchOffset: Int,
        precision: Precision
    ) {
        // 1. Prepare Dispatch (CPU-side check handled in SwiftRenderer, this is GPU dispatch preparation)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "PrepareRenderDispatch"
            encoder.setComputePipelineState(self.prepareDispatchPipeline)
            encoder.setBuffer(orderedBuffers.activeTileCount, offset: 0, index: 0)
            encoder.setBuffer(dispatchArgs, offset: 0, index: 1)
            
            var dispatchParams = RenderDispatchParamsSwift(
                tileCount: UInt32(orderedBuffers.tileCount),
                totalAssignments: 0, // unused in kernel
                gaussianCount: 0
            )
            encoder.setBytes(&dispatchParams, length: MemoryLayout<RenderDispatchParamsSwift>.stride, index: 2)
            
            let threads = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: threads)
            encoder.endEncoding()
        }

        // 2. Clear Targets
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "ClearTargets"
            encoder.setComputePipelineState(self.clearPipeline)
            encoder.setBuffer(outputBuffers.colorOutGPU, offset: 0, index: 0)
            encoder.setBuffer(outputBuffers.depthOutGPU, offset: 0, index: 1)
            encoder.setBuffer(outputBuffers.alphaOutGPU, offset: 0, index: 2)
            
            var clearParams = ClearParamsSwift(
                pixelCount: params.width * params.height,
                whiteBackground: params.whiteBackground
            )
            encoder.setBytes(&clearParams, length: MemoryLayout<ClearParamsSwift>.stride, index: 3)
            
            let threads = MTLSize(width: Int(clearParams.pixelCount), height: 1, depth: 1)
            let tg = MTLSize(width: self.clearPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // 3. Render Tiles
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "RenderTiles"
            if precision == .float16, let halfPipe = self.renderPipelineHalf {
                encoder.setComputePipelineState(halfPipe)
            } else {
                encoder.setComputePipelineState(self.renderPipeline)
            }
            encoder.setBuffer(orderedBuffers.headers, offset: 0, index: 0)
            encoder.setBuffer(orderedBuffers.means, offset: 0, index: 1)
            encoder.setBuffer(orderedBuffers.conics, offset: 0, index: 2)
            encoder.setBuffer(orderedBuffers.colors, offset: 0, index: 3)
            encoder.setBuffer(orderedBuffers.opacities, offset: 0, index: 4)
            encoder.setBuffer(orderedBuffers.depths, offset: 0, index: 5)
            
            encoder.setBuffer(outputBuffers.colorOutGPU, offset: 0, index: 6)
            encoder.setBuffer(outputBuffers.depthOutGPU, offset: 0, index: 7)
            encoder.setBuffer(outputBuffers.alphaOutGPU, offset: 0, index: 8)
            
            var p = params
            encoder.setBytes(&p, length: MemoryLayout<RenderParams>.stride, index: 9)
            
            encoder.setBuffer(orderedBuffers.activeTileIndices, offset: 0, index: 10)
            encoder.setBuffer(orderedBuffers.activeTileCount, offset: 0, index: 11)
            
            // Dispatch using indirect args
            let w = Int(params.tileWidth)
            let h = Int(params.tileHeight)
            let tg = MTLSize(width: w, height: h, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: dispatchOffset,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }
    }
}

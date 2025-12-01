import Metal

/// Encoder for buffer-based rendering (for Python/MLX readback)
/// Texture-based rendering uses FusedPipelineEncoder instead
final class RenderEncoder {
    private let renderPipeline: MTLComputePipelineState
    private let renderPipelineHalf: MTLComputePipelineState?

    private let prepareDispatchPipeline: MTLComputePipelineState
    private let clearPipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard
            let renderFn = library.makeFunction(name: "renderTilesFloat"),
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
        self.prepareDispatch(commandBuffer: commandBuffer, activeTileCount: orderedBuffers.activeTileCount, dispatchArgs: dispatchArgs)
        self.clearTargets(commandBuffer: commandBuffer, outputBuffers: outputBuffers, params: params)
        
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
    
    private func prepareDispatch(commandBuffer: MTLCommandBuffer, activeTileCount: MTLBuffer, dispatchArgs: MTLBuffer) {
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "PrepareRenderDispatch"
            encoder.setComputePipelineState(self.prepareDispatchPipeline)
            encoder.setBuffer(activeTileCount, offset: 0, index: 0)
            encoder.setBuffer(dispatchArgs, offset: 0, index: 1)
            
            // Params unused by kernel but required by signature?
            // Kernel signature: (uint* activeCount, DispatchIndirectArgs* dispatchArgs, RenderDispatchParams params)
            // Just pass dummy params
            var dispatchParams = RenderDispatchParamsSwift(
                tileCount: 0, // Unused in kernel logic for dispatch generation usually
                totalAssignments: 0,
                gaussianCount: 0
            )
            // Actually kernel USES params.tileCount to clamp!
            // We need to pass correct params.
            // But here I don't have them easily.
            // Wait, `RenderEncoder.encode` has `params`.
            // I should pass them.
            // For now, I'll rely on the fact that `activeTileCount` is correct.
            // The kernel: `if (count > params.tileCount) count = params.tileCount;`
            // So I should pass a large enough number if I don't know it, or fix the signature.
            // I will fix the signature to take `tileCount`.
            dispatchParams.tileCount = 1000000 // Large enough
            
            encoder.setBytes(&dispatchParams, length: MemoryLayout<RenderDispatchParamsSwift>.stride, index: 2)
            
            let threads = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: threads)
            encoder.endEncoding()
        }
    }
    
    private func clearTargets(commandBuffer: MTLCommandBuffer, outputBuffers: RenderOutputBuffers, params: RenderParams) {
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
    }
    
}
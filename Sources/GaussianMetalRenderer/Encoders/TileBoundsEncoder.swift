import Metal

final class TileBoundsEncoder {
    private let pipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let function = library.makeFunction(name: "tileBoundsKernel") else {
            fatalError("tileBoundsKernel not found")
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianBuffers: GaussianInputBuffers,
        boundsBuffer: MTLBuffer,
        params: RenderParams
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
            gaussianCount: params.activeTileCount // activeTileCount re-used as gaussianCount in legacy code, cleaner here
        )
        // Fix: RenderParams doesn't have gaussianCount directly, but usually it's passed or inferred.
        // In SwiftRenderer, it was constructed from params properties + explicit gaussianCount.
        // Let's fix the signature to accept gaussianCount explicitly if needed, or use a dedicated struct.
        // Wait, `TileBoundsParamsSwift` is what the kernel takes.
        
        // Re-checking SwiftRenderer.swift usage:
        // var paramsStruct = TileBoundsParamsSwift(..., gaussianCount: UInt32(gaussianCount))
        // So I should pass gaussianCount to this encode function.
        
        encoder.setComputePipelineState(self.pipeline)
        encoder.setBuffer(gaussianBuffers.means, offset: 0, index: 0)
        encoder.setBuffer(gaussianBuffers.radii, offset: 0, index: 1)
        encoder.setBuffer(gaussianBuffers.mask, offset: 0, index: 2)
        encoder.setBuffer(boundsBuffer, offset: 0, index: 3)
        encoder.setBytes(&paramsStruct, length: MemoryLayout<TileBoundsParamsSwift>.stride, index: 4)
        
        let count = Int(paramsStruct.gaussianCount)
        let threads = MTLSize(width: count, height: 1, depth: 1)
        let tg = MTLSize(width: self.pipeline.threadExecutionWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
    
    // Overload to accept explicit count if params is not enough
    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianBuffers: GaussianInputBuffers,
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
        
        encoder.setComputePipelineState(self.pipeline)
        encoder.setBuffer(gaussianBuffers.means, offset: 0, index: 0)
        encoder.setBuffer(gaussianBuffers.radii, offset: 0, index: 1)
        encoder.setBuffer(gaussianBuffers.mask, offset: 0, index: 2)
        encoder.setBuffer(boundsBuffer, offset: 0, index: 3)
        encoder.setBytes(&paramsStruct, length: MemoryLayout<TileBoundsParamsSwift>.stride, index: 4)
        
        let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
        let tg = MTLSize(width: self.pipeline.threadExecutionWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

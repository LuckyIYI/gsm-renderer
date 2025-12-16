import Metal

/// Encodes scatter stage: assigns gaussians to tiles using SIMD-cooperative approach
final class LocalScatterEncoder {
    private let prepareDispatchPipeline: MTLComputePipelineState
    private let scatter16Pipeline: MTLComputePipelineState
    private let scatter16OptimizedPipeline: MTLComputePipelineState?

    init(library: MTLLibrary, device: MTLDevice) throws {
        guard let prepareDispatchFn = library.makeFunction(name: "localPrepareScatterDispatch") else {
            throw RendererError.failedToCreatePipeline("Missing localPrepareScatterDispatch kernel")
        }
        self.prepareDispatchPipeline = try device.makeComputePipelineState(function: prepareDispatchFn)

        // 16-bit baseline scatter
        guard let scatter16Fn = library.makeFunction(name: "localScatterSimd16") else {
            throw RendererError.failedToCreatePipeline("Missing localScatterSimd16 kernel")
        }
        self.scatter16Pipeline = try device.makeComputePipelineState(function: scatter16Fn)

        if let scatter16OptFn = library.makeFunction(name: "localScatterSimd16V7") {
            self.scatter16OptimizedPipeline = try? device.makeComputePipelineState(function: scatter16OptFn)
        } else {
            self.scatter16OptimizedPipeline = nil
        }
    }

    /// Encode scatter with 16-bit depth keys (fixed layout: tileId * maxPerTile)
    /// Uses indirect dispatch based on visible count from compaction
    func encode16(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        tileCounters: MTLBuffer,
        depthKeys16: MTLBuffer,
        globalIndices: MTLBuffer,
        tilesX: Int,
        maxPerTile: Int,
        tileWidth: Int,
        tileHeight: Int
    ) {
        // Select pipeline - prefer optimized if available
        let scatter16: MTLComputePipelineState
        let threadgroupSize: Int
        let gaussiansPerTG: Int

        if let optimized = scatter16OptimizedPipeline {
            scatter16 = optimized
            threadgroupSize = 256 // 256 threads per TG
            gaussiansPerTG = 256 * 32 // Each thread processes 32 gaussians (8192 per TG)
        } else {
            scatter16 = self.scatter16Pipeline
            threadgroupSize = 128 // 4 SIMD groups
            gaussiansPerTG = 4
        }

        var tilesXU = UInt32(tilesX)
        var maxPerTileU = UInt32(maxPerTile)
        var tileW = Int32(tileWidth)
        var tileH = Int32(tileHeight)

        // Create dispatch args buffer
        guard let dispatchArgsBuffer = commandBuffer.device.makeBuffer(length: 12, options: .storageModeShared) else {
            return
        }

        // Prepare indirect dispatch
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_PrepareScatter16Dispatch"
            encoder.setComputePipelineState(self.prepareDispatchPipeline)
            encoder.setBuffer(compactedHeader, offset: 0, index: 0)
            encoder.setBuffer(dispatchArgsBuffer, offset: 0, index: 1)
            var tgWidth = UInt32(gaussiansPerTG)
            encoder.setBytes(&tgWidth, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // 16-bit scatter (fixed layout: tileId * maxPerTile + localIdx)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_Scatter16"
            encoder.setComputePipelineState(scatter16)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBuffer(tileCounters, offset: 0, index: 2)
            encoder.setBuffer(depthKeys16, offset: 0, index: 3)
            encoder.setBuffer(globalIndices, offset: 0, index: 4)
            encoder.setBytes(&tilesXU, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&maxPerTileU, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&tileW, length: MemoryLayout<Int32>.stride, index: 7)
            encoder.setBytes(&tileH, length: MemoryLayout<Int32>.stride, index: 8)

            let tg = MTLSize(width: threadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: dispatchArgsBuffer,
                                         indirectBufferOffset: 0,
                                         threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }
}

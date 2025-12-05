import Metal

/// Encodes scatter stage: assigns gaussians to tiles using SIMD-cooperative approach
public final class LocalScatterEncoder {
    private let prepareDispatchPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState
    private let scatter16Pipeline: MTLComputePipelineState?

    public init(library: MTLLibrary, device: MTLDevice) throws {
        guard let prepareDispatchFn = library.makeFunction(name: "LocalPrepareScatterDispatch"),
              let scatterFn = library.makeFunction(name: "LocalScatterSimd") else {
            fatalError("Missing scatter kernels")
        }
        self.prepareDispatchPipeline = try device.makeComputePipelineState(function: prepareDispatchFn)
        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)

        // Optional 16-bit scatter
        if let scatter16Fn = library.makeFunction(name: "LocalScatterSimd16") {
            self.scatter16Pipeline = try? device.makeComputePipelineState(function: scatter16Fn)
        } else {
            self.scatter16Pipeline = nil
        }
    }

    /// Check if 16-bit scatter is available
    public var has16BitScatter: Bool { scatter16Pipeline != nil }

    /// Encode scatter with 32-bit keys (fixed layout: tileId * maxPerTile)
    public func encode(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        tileCounters: MTLBuffer,
        sortKeys: MTLBuffer,
        sortIndices: MTLBuffer,
        tilesX: Int,
        maxPerTile: Int,
        tileWidth: Int,
        tileHeight: Int
    ) {
        var tilesXU = UInt32(tilesX)
        var maxPerTileU = UInt32(maxPerTile)
        var tileW = Int32(tileWidth)
        var tileH = Int32(tileHeight)

        // Create dispatch args buffer (12 bytes for MTLDispatchThreadgroupsIndirectArguments)
        let dispatchArgsBuffer = commandBuffer.device.makeBuffer(length: 12, options: .storageModeShared)!

        // Prepare indirect dispatch
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_PrepareScatterDispatch"
            encoder.setComputePipelineState(prepareDispatchPipeline)
            encoder.setBuffer(compactedHeader, offset: 0, index: 0)
            encoder.setBuffer(dispatchArgsBuffer, offset: 0, index: 1)
            var tgWidth = UInt32(4)  // 4 gaussians per threadgroup (4 SIMD groups)
            encoder.setBytes(&tgWidth, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // SIMD-cooperative scatter (fixed layout: tileId * maxPerTile + localIdx)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_Scatter_SIMD"
            encoder.setComputePipelineState(scatterPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBuffer(tileCounters, offset: 0, index: 2)
            encoder.setBuffer(sortKeys, offset: 0, index: 3)
            encoder.setBuffer(sortIndices, offset: 0, index: 4)
            encoder.setBytes(&tilesXU, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&maxPerTileU, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&tileW, length: MemoryLayout<Int32>.stride, index: 7)
            encoder.setBytes(&tileH, length: MemoryLayout<Int32>.stride, index: 8)

            // 128 threads per threadgroup (4 SIMD groups = 4 gaussians)
            let tg = MTLSize(width: 128, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: dispatchArgsBuffer,
                                         indirectBufferOffset: 0,
                                         threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Encode scatter with 16-bit depth keys (fixed layout: tileId * maxPerTile)
    public func encode16(
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
        guard let scatter16 = scatter16Pipeline else { return }

        var tilesXU = UInt32(tilesX)
        var maxPerTileU = UInt32(maxPerTile)
        var tileW = Int32(tileWidth)
        var tileH = Int32(tileHeight)

        // Create dispatch args buffer
        let dispatchArgsBuffer = commandBuffer.device.makeBuffer(length: 12, options: .storageModeShared)!

        // Prepare indirect dispatch
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_PrepareScatter16Dispatch"
            encoder.setComputePipelineState(prepareDispatchPipeline)
            encoder.setBuffer(compactedHeader, offset: 0, index: 0)
            encoder.setBuffer(dispatchArgsBuffer, offset: 0, index: 1)
            var tgWidth = UInt32(4)
            encoder.setBytes(&tgWidth, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // 16-bit scatter (fixed layout: tileId * maxPerTile + localIdx)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_Scatter16_SIMD"
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

            let tg = MTLSize(width: 128, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: dispatchArgsBuffer,
                                         indirectBufferOffset: 0,
                                         threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }
}

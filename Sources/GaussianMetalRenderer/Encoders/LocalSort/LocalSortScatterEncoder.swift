import Metal

/// Encodes scatter stage: assigns gaussians to tiles using SIMD-cooperative approach
public final class LocalSortScatterEncoder {
    private let prepareDispatchPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState
    private let scatter16Pipeline: MTLComputePipelineState?

    public init(library: MTLLibrary, device: MTLDevice) throws {
        guard let prepareDispatchFn = library.makeFunction(name: "localSortPrepareScatterDispatch"),
              let scatterFn = library.makeFunction(name: "localSortScatterSimd") else {
            fatalError("Missing scatter kernels")
        }
        self.prepareDispatchPipeline = try device.makeComputePipelineState(function: prepareDispatchFn)
        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)

        // Optional 16-bit scatter
        if let scatter16Fn = library.makeFunction(name: "localSortScatterSimd16") {
            self.scatter16Pipeline = try? device.makeComputePipelineState(function: scatter16Fn)
        } else {
            self.scatter16Pipeline = nil
        }
    }

    /// Check if 16-bit scatter is available
    public var has16BitScatter: Bool { scatter16Pipeline != nil }

    /// Encode scatter with 32-bit keys
    public func encode(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        tileCounts: MTLBuffer,
        tileOffsets: MTLBuffer,
        sortKeys: MTLBuffer,
        sortIndices: MTLBuffer,
        tilesX: Int,
        maxAssignments: Int,
        tileWidth: Int,
        tileHeight: Int
    ) {
        var tilesXU = UInt32(tilesX)
        var maxAssignmentsU = UInt32(maxAssignments)
        var tileW = Int32(tileWidth)
        var tileH = Int32(tileHeight)

        // Create dispatch args buffer (12 bytes for MTLDispatchThreadgroupsIndirectArguments)
        let dispatchArgsBuffer = commandBuffer.device.makeBuffer(length: 12, options: .storageModeShared)!

        // Prepare indirect dispatch
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_PrepareScatterDispatch"
            encoder.setComputePipelineState(prepareDispatchPipeline)
            encoder.setBuffer(compactedHeader, offset: 0, index: 0)
            encoder.setBuffer(dispatchArgsBuffer, offset: 0, index: 1)
            var tgWidth = UInt32(4)  // 4 gaussians per threadgroup (4 SIMD groups)
            encoder.setBytes(&tgWidth, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // SIMD-cooperative scatter
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_Scatter_SIMD"
            encoder.setComputePipelineState(scatterPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBuffer(tileCounts, offset: 0, index: 2)
            encoder.setBuffer(tileOffsets, offset: 0, index: 3)
            encoder.setBuffer(sortKeys, offset: 0, index: 4)
            encoder.setBuffer(sortIndices, offset: 0, index: 5)
            encoder.setBytes(&tilesXU, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&maxAssignmentsU, length: MemoryLayout<UInt32>.stride, index: 7)
            encoder.setBytes(&tileW, length: MemoryLayout<Int32>.stride, index: 8)
            encoder.setBytes(&tileH, length: MemoryLayout<Int32>.stride, index: 9)

            // 128 threads per threadgroup (4 SIMD groups = 4 gaussians)
            let tg = MTLSize(width: 128, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: dispatchArgsBuffer,
                                         indirectBufferOffset: 0,
                                         threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Encode scatter with 16-bit depth keys (experimental)
    public func encode16(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        tileCounts: MTLBuffer,
        tileOffsets: MTLBuffer,
        depthKeys16: MTLBuffer,
        sortIndices: MTLBuffer,
        tilesX: Int,
        maxAssignments: Int,
        tileWidth: Int,
        tileHeight: Int
    ) {
        guard let scatter16 = scatter16Pipeline else { return }

        var tilesXU = UInt32(tilesX)
        var maxAssignmentsU = UInt32(maxAssignments)
        var tileW = Int32(tileWidth)
        var tileH = Int32(tileHeight)

        // Create dispatch args buffer
        let dispatchArgsBuffer = commandBuffer.device.makeBuffer(length: 12, options: .storageModeShared)!

        // Prepare indirect dispatch
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_PrepareScatter16Dispatch"
            encoder.setComputePipelineState(prepareDispatchPipeline)
            encoder.setBuffer(compactedHeader, offset: 0, index: 0)
            encoder.setBuffer(dispatchArgsBuffer, offset: 0, index: 1)
            var tgWidth = UInt32(4)
            encoder.setBytes(&tgWidth, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // 16-bit scatter
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "LocalSort_Scatter16_SIMD"
            encoder.setComputePipelineState(scatter16)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBuffer(tileCounts, offset: 0, index: 2)
            encoder.setBuffer(tileOffsets, offset: 0, index: 3)
            encoder.setBuffer(depthKeys16, offset: 0, index: 4)
            encoder.setBuffer(sortIndices, offset: 0, index: 5)
            encoder.setBytes(&tilesXU, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&maxAssignmentsU, length: MemoryLayout<UInt32>.stride, index: 7)
            encoder.setBytes(&tileW, length: MemoryLayout<Int32>.stride, index: 8)
            encoder.setBytes(&tileH, length: MemoryLayout<Int32>.stride, index: 9)

            let tg = MTLSize(width: 128, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: dispatchArgsBuffer,
                                         indirectBufferOffset: 0,
                                         threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }
}

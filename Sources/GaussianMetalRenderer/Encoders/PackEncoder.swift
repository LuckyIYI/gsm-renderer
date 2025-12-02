import Metal

final class PackEncoder {
    private let headerFromSortedPipeline: MTLComputePipelineState
    private let compactActiveTilesPipeline: MTLComputePipelineState
    
    let packThreadgroupSize: Int

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard
            let headerFn = library.makeFunction(name: "buildHeadersFromSortedKernel"),
            let compactFn = library.makeFunction(name: "compactActiveTilesKernel")
        else {
            fatalError("Pack functions missing")
        }


        self.headerFromSortedPipeline = try device.makeComputePipelineState(function: headerFn)
        self.compactActiveTilesPipeline = try device.makeComputePipelineState(function: compactFn)
        
        self.packThreadgroupSize = max(1, 256) // TODO: delete, unused
    }


    /// Build per-tile headers and the active-tile list without packing payload buffers.
    func encodeHeadersAndActiveTiles(
        commandBuffer: MTLCommandBuffer,
        sortedKeys: MTLBuffer,
        assignment: TileAssignmentBuffers,
        orderedHeaders: MTLBuffer,
        activeTileIndices: MTLBuffer,
        activeTileCount: MTLBuffer
    ) {
        // Note: activeTileCount is reset in resetTileBuilderStateKernel before tile assignment

        // Headers From Sorted
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "HeadersFromSorted"
            var tileCount = UInt32(assignment.tileCount)
            encoder.setComputePipelineState(self.headerFromSortedPipeline)
            encoder.setBuffer(sortedKeys, offset: 0, index: 0)
            encoder.setBuffer(orderedHeaders, offset: 0, index: 1)
            encoder.setBuffer(assignment.header, offset: 0, index: 2)
            encoder.setBytes(&tileCount, length: MemoryLayout<UInt32>.size, index: 3)

            let threads = MTLSize(width: assignment.tileCount, height: 1, depth: 1)
            let tgWidth = self.headerFromSortedPipeline.threadExecutionWidth
            let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Compact Active Tiles
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "CompactActiveTiles"
            encoder.setComputePipelineState(self.compactActiveTilesPipeline)
            encoder.setBuffer(orderedHeaders, offset: 0, index: 0)
            encoder.setBuffer(activeTileIndices, offset: 0, index: 1)
            encoder.setBuffer(activeTileCount, offset: 0, index: 2)
            var tileCountValue = UInt32(assignment.tileCount)
            encoder.setBytes(&tileCountValue, length: MemoryLayout<UInt32>.stride, index: 3)

            let tgWidth = max(1, self.compactActiveTilesPipeline.threadExecutionWidth)
            let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
            let threads = MTLSize(width: assignment.tileCount, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }
}

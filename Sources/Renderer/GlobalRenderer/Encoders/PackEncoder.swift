import Metal

final class PackEncoder {
    private let headerFromSortedPipeline: MTLComputePipelineState
    let packThreadgroupSize: Int

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let headerFn = library.makeFunction(name: "buildHeadersFromSortedKernel") else {
            throw RendererError.failedToCreatePipeline("buildHeadersFromSortedKernel missing")
        }
        self.headerFromSortedPipeline = try device.makeComputePipelineState(function: headerFn)
        self.packThreadgroupSize = 256
    }

    /// Build per-tile headers and the active-tile list (single pass)
    func encodeHeadersAndActiveTiles(
        commandBuffer: MTLCommandBuffer,
        sortedKeys: MTLBuffer,
        assignment: TileAssignmentBuffers,
        orderedHeaders: MTLBuffer,
        activeTileIndices: MTLBuffer,
        activeTileCount: MTLBuffer
    ) {
        // Headers from sorted assignments + compact active tiles
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "HeadersAndActiveTiles"
            var tileCount = UInt32(assignment.tileCount)
            encoder.setComputePipelineState(self.headerFromSortedPipeline)
            encoder.setBuffer(sortedKeys, offset: 0, index: 0)
            encoder.setBuffer(orderedHeaders, offset: 0, index: 1)
            encoder.setBuffer(assignment.header, offset: 0, index: 2)
            encoder.setBytes(&tileCount, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBuffer(activeTileIndices, offset: 0, index: 4)
            encoder.setBuffer(activeTileCount, offset: 0, index: 5)

            let threads = MTLSize(width: assignment.tileCount, height: 1, depth: 1)
            let tgWidth = self.headerFromSortedPipeline.threadExecutionWidth
            let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }
}

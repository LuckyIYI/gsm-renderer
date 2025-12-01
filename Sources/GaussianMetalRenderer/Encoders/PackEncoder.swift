import Metal

final class PackEncoder {
    private let packPipeline: MTLComputePipelineState
    private let headerFromSortedPipeline: MTLComputePipelineState
    private let compactActiveTilesPipeline: MTLComputePipelineState
    private let packPipelineHalf: MTLComputePipelineState?
    
    let packThreadgroupSize: Int

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard
            let packFn = library.makeFunction(name: "packTileDataKernelFloat"),
            let headerFn = library.makeFunction(name: "buildHeadersFromSortedKernel"),
            let compactFn = library.makeFunction(name: "compactActiveTilesKernel")
        else {
            fatalError("Pack functions missing")
        }

        self.packPipeline = try device.makeComputePipelineState(function: packFn)
        if let packHalfFn = library.makeFunction(name: "packTileDataKernelHalf") {
            self.packPipelineHalf = try? device.makeComputePipelineState(function: packHalfFn)
        } else {
            self.packPipelineHalf = nil
        }
        self.headerFromSortedPipeline = try device.makeComputePipelineState(function: headerFn)
        self.compactActiveTilesPipeline = try device.makeComputePipelineState(function: compactFn)
        
        self.packThreadgroupSize = max(1, min(packPipeline.maxTotalThreadsPerThreadgroup, 256))
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        sortedIndices: MTLBuffer,
        sortedKeys: MTLBuffer,
        gaussianBuffers: GaussianInputBuffers,
        orderedBuffers: OrderedBufferSet,
        assignment: TileAssignmentBuffers,
        dispatchArgs: MTLBuffer,
        dispatchOffset: Int,
        activeTileIndices: MTLBuffer,
        activeTileCount: MTLBuffer,
        precision: Precision
    ) {
        // 1. Pack
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Pack"
            if precision == .float16, let halfPipe = self.packPipelineHalf {
                encoder.setComputePipelineState(halfPipe)
            } else {
                encoder.setComputePipelineState(self.packPipeline)
            }
            encoder.setBuffer(sortedIndices, offset: 0, index: 0)
            encoder.setBuffer(gaussianBuffers.means, offset: 0, index: 1)
            encoder.setBuffer(gaussianBuffers.conics, offset: 0, index: 2)
            encoder.setBuffer(gaussianBuffers.colors, offset: 0, index: 3)
            encoder.setBuffer(gaussianBuffers.opacities, offset: 0, index: 4)
            encoder.setBuffer(gaussianBuffers.depths, offset: 0, index: 5)
            encoder.setBuffer(orderedBuffers.means, offset: 0, index: 6)
            encoder.setBuffer(orderedBuffers.conics, offset: 0, index: 7)
            encoder.setBuffer(orderedBuffers.colors, offset: 0, index: 8)
            encoder.setBuffer(orderedBuffers.opacities, offset: 0, index: 9)
            encoder.setBuffer(orderedBuffers.depths, offset: 0, index: 10)
            encoder.setBuffer(assignment.header, offset: 0, index: 11)
            encoder.setBuffer(assignment.tileIndices, offset: 0, index: 12)
            encoder.setBuffer(assignment.tileIds, offset: 0, index: 13)
            
            
            let tg = MTLSize(width: self.packThreadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: dispatchOffset,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }
        self.encodeHeadersAndActiveTiles(
            commandBuffer: commandBuffer,
            sortedKeys: sortedKeys,
            assignment: assignment,
            orderedHeaders: orderedBuffers.headers,
            activeTileIndices: activeTileIndices,
            activeTileCount: activeTileCount
        )
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

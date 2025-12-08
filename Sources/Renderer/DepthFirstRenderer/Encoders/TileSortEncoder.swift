import Metal
import RendererTypes

/// Encoder for stable radix sort on tile IDs (preserves depth order within tiles)
/// Uses the same multi-block radix sort algorithm as depth sort
final class TileSortEncoder {
    private let histogramPipeline: MTLComputePipelineState
    private let scanBlocksPipeline: MTLComputePipelineState
    private let exclusiveScanPipeline: MTLComputePipelineState
    private let applyOffsetsPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState

    private let blockSize = 256
    private let grainSize = 4

    struct SortBuffers {
        let histogram: MTLBuffer
        let blockSums: MTLBuffer
        let scannedHistogram: MTLBuffer
        let scratchTileIds: MTLBuffer
        let scratchGaussianIndices: MTLBuffer
    }

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let histFn = library.makeFunction(name: "tileRadixHistogramKernel"),
              let scanFn = library.makeFunction(name: "tileRadixScanBlocksKernel"),
              let exclusiveFn = library.makeFunction(name: "tileRadixExclusiveScanKernel"),
              let applyFn = library.makeFunction(name: "tileRadixApplyOffsetsKernel"),
              let scatterFn = library.makeFunction(name: "tileRadixScatterKernel")
        else {
            throw RendererError.failedToCreatePipeline("Tile radix sort kernels not found")
        }

        self.histogramPipeline = try device.makeComputePipelineState(function: histFn)
        self.scanBlocksPipeline = try device.makeComputePipelineState(function: scanFn)
        self.exclusiveScanPipeline = try device.makeComputePipelineState(function: exclusiveFn)
        self.applyOffsetsPipeline = try device.makeComputePipelineState(function: applyFn)
        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)
    }

    /// Encode stable radix sort on tile IDs
    /// Sorts (tileIds, gaussianIndices) pairs by tileId while preserving order of equal keys
    func encode(
        commandBuffer: MTLCommandBuffer,
        tileIds: MTLBuffer,
        gaussianIndices: MTLBuffer,
        tileCount: Int,
        sortBuffers: SortBuffers,
        header: MTLBuffer,
        dispatchArgs: MTLBuffer
    ) {
        // Determine number of radix passes based on tileCount
        let bitsNeeded = tileCount > 0 ? (Int(log2(Double(max(tileCount - 1, 1)))) + 1) : 1
        let numPasses = (bitsNeeded + 7) / 8 // 8 bits per pass

        var inputTileIds = tileIds
        var inputIndices = gaussianIndices
        var outputTileIds = sortBuffers.scratchTileIds
        var outputIndices = sortBuffers.scratchGaussianIndices

        let tg = MTLSize(width: blockSize, height: 1, depth: 1)

        for pass in 0 ..< numPasses {
            let bitOffset = UInt32(pass * 8)

            // Step 1: Build histogram
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "TileRadixHistogram_\(pass)"
                encoder.setComputePipelineState(histogramPipeline)
                encoder.setBuffer(inputTileIds, offset: 0, index: 0)
                encoder.setBuffer(sortBuffers.histogram, offset: 0, index: 1)
                encoder.setBuffer(header, offset: 0, index: 2)
                var bo = bitOffset
                encoder.setBytes(&bo, length: 4, index: 3)

                encoder.dispatchThreadgroups(
                    indirectBuffer: dispatchArgs,
                    indirectBufferOffset: DepthFirstDispatchSlot.tileHistogram.offset,
                    threadsPerThreadgroup: tg
                )
                encoder.endEncoding()
            }

            // Step 2: Scan histogram blocks
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "TileRadixScanBlocks_\(pass)"
                encoder.setComputePipelineState(scanBlocksPipeline)
                encoder.setBuffer(sortBuffers.histogram, offset: 0, index: 0)
                encoder.setBuffer(sortBuffers.blockSums, offset: 0, index: 1)
                encoder.setBuffer(header, offset: 0, index: 2)

                encoder.dispatchThreadgroups(
                    indirectBuffer: dispatchArgs,
                    indirectBufferOffset: DepthFirstDispatchSlot.tileScanBlocks.offset,
                    threadsPerThreadgroup: tg
                )
                encoder.endEncoding()
            }

            // Step 3: Exclusive scan of block sums
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "TileRadixExclusive_\(pass)"
                encoder.setComputePipelineState(exclusiveScanPipeline)
                encoder.setBuffer(sortBuffers.blockSums, offset: 0, index: 0)
                encoder.setBuffer(sortBuffers.scannedHistogram, offset: 0, index: 1)
                encoder.setBuffer(sortBuffers.histogram, offset: 0, index: 2)
                encoder.setBuffer(header, offset: 0, index: 3)
                encoder.setThreadgroupMemoryLength(256 * MemoryLayout<UInt32>.stride, index: 0)

                encoder.dispatchThreadgroups(
                    indirectBuffer: dispatchArgs,
                    indirectBufferOffset: DepthFirstDispatchSlot.tileExclusive.offset,
                    threadsPerThreadgroup: tg
                )
                encoder.endEncoding()
            }

            // Step 4: Apply offsets to histogram
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "TileRadixApply_\(pass)"
                encoder.setComputePipelineState(applyOffsetsPipeline)
                encoder.setBuffer(sortBuffers.histogram, offset: 0, index: 0)
                encoder.setBuffer(sortBuffers.blockSums, offset: 0, index: 1)
                encoder.setBuffer(sortBuffers.scannedHistogram, offset: 0, index: 2)
                encoder.setBuffer(header, offset: 0, index: 3)

                encoder.dispatchThreadgroups(
                    indirectBuffer: dispatchArgs,
                    indirectBufferOffset: DepthFirstDispatchSlot.tileApply.offset,
                    threadsPerThreadgroup: tg
                )
                encoder.endEncoding()
            }

            // Step 5: Scatter to sorted positions
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "TileRadixScatter_\(pass)"
                encoder.setComputePipelineState(scatterPipeline)
                encoder.setBuffer(inputTileIds, offset: 0, index: 0)
                encoder.setBuffer(inputIndices, offset: 0, index: 1)
                encoder.setBuffer(sortBuffers.scannedHistogram, offset: 0, index: 2) // Use scanned histogram with offsets
                encoder.setBuffer(outputTileIds, offset: 0, index: 3)
                encoder.setBuffer(outputIndices, offset: 0, index: 4)
                encoder.setBuffer(header, offset: 0, index: 5)
                var bo = bitOffset
                encoder.setBytes(&bo, length: 4, index: 6)

                encoder.dispatchThreadgroups(
                    indirectBuffer: dispatchArgs,
                    indirectBufferOffset: DepthFirstDispatchSlot.tileScatter.offset,
                    threadsPerThreadgroup: tg
                )
                encoder.endEncoding()
            }

            // Swap buffers for next pass
            swap(&inputTileIds, &outputTileIds)
            swap(&inputIndices, &outputIndices)
        }

        // If odd number of passes, need to copy back to original buffers
        if numPasses % 2 == 1 {
            if let blit = commandBuffer.makeBlitCommandEncoder() {
                blit.label = "TileSortCopyBack"
                blit.copy(from: inputTileIds, sourceOffset: 0, to: tileIds, destinationOffset: 0, size: tileIds.length)
                blit.copy(from: inputIndices, sourceOffset: 0, to: gaussianIndices, destinationOffset: 0, size: gaussianIndices.length)
                blit.endEncoding()
            }
        }
    }
}

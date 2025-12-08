import Metal
import RendererTypes

/// Radix sort encoder for 16-bit depth keys using indirect dispatch
/// Sorts gaussians by depth first (front-to-back for proper alpha blending)
final class DepthSortEncoder {
    private let histogramPipeline: MTLComputePipelineState
    private let scanBlocksPipeline: MTLComputePipelineState
    private let exclusiveScanPipeline: MTLComputePipelineState
    private let applyOffsetsPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState

    let blockSize = 256
    let grainSize = 4
    let radix = 256

    init(device: MTLDevice, library: MTLLibrary) throws {
        // Use depth-sort-specific kernels that read from DepthFirstHeader correctly
        guard
            let histFn = library.makeFunction(name: "depthSortHistogramKernel"),
            let scanFn = library.makeFunction(name: "depthSortScanBlocksKernel"),
            let exclusiveFn = library.makeFunction(name: "depthSortExclusiveScanKernel"),
            let applyFn = library.makeFunction(name: "depthSortApplyOffsetsKernel"),
            let scatterFn = library.makeFunction(name: "depthSortScatterKernel")
        else {
            throw RendererError.failedToCreatePipeline("Depth sort radix functions missing from library")
        }

        self.histogramPipeline = try device.makeComputePipelineState(function: histFn)
        self.scanBlocksPipeline = try device.makeComputePipelineState(function: scanFn)
        self.exclusiveScanPipeline = try device.makeComputePipelineState(function: exclusiveFn)
        self.applyOffsetsPipeline = try device.makeComputePipelineState(function: applyFn)
        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)
    }

    struct SortBuffers {
        let histogram: MTLBuffer
        let blockSums: MTLBuffer
        let scannedHistogram: MTLBuffer
        let scratchKeys: MTLBuffer
        let scratchPayload: MTLBuffer
    }

    /// Encode radix sort for depth keys (16-bit, needs 2 passes) using indirect dispatch
    func encode(
        commandBuffer: MTLCommandBuffer,
        depthKeys: MTLBuffer,
        primitiveIndices: MTLBuffer,
        sortBuffers: SortBuffers,
        header: MTLBuffer,
        dispatchArgs: MTLBuffer
    ) {
        // 16-bit depth key needs 2 passes (bytes 0 and 1)
        let passCount = 2

        var sourceKeys = depthKeys
        var destKeys = sortBuffers.scratchKeys
        var sourcePayload = primitiveIndices
        var destPayload = sortBuffers.scratchPayload

        for digit in 0 ..< passCount {
            encodeOnePass(
                commandBuffer: commandBuffer,
                digit: digit,
                sourceKeys: sourceKeys,
                destKeys: destKeys,
                sourcePayload: sourcePayload,
                destPayload: destPayload,
                sortBuffers: sortBuffers,
                header: header,
                dispatchArgs: dispatchArgs
            )
            swap(&sourceKeys, &destKeys)
            swap(&sourcePayload, &destPayload)
        }

        // passCount is even, so results are in original buffers - no copy needed
    }

    private func encodeOnePass(
        commandBuffer: MTLCommandBuffer,
        digit: Int,
        sourceKeys: MTLBuffer,
        destKeys: MTLBuffer,
        sourcePayload: MTLBuffer,
        destPayload: MTLBuffer,
        sortBuffers: SortBuffers,
        header: MTLBuffer,
        dispatchArgs: MTLBuffer
    ) {
        var currentDigitUInt = UInt32(digit)
        let tg = MTLSize(width: blockSize, height: 1, depth: 1)

        // A. Histogram - uses DF_SLOT_DEPTH_HISTOGRAM
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthRadixHist_\(digit)"
            encoder.setComputePipelineState(histogramPipeline)
            encoder.setBuffer(sourceKeys, offset: 0, index: 0)
            encoder.setBuffer(sortBuffers.histogram, offset: 0, index: 1)
            encoder.setBytes(&currentDigitUInt, length: 4, index: 3)
            encoder.setBuffer(header, offset: 0, index: 4)

            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: DepthFirstDispatchSlot.depthHistogram.offset,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }

        // B. Scan blocks - uses DF_SLOT_DEPTH_SCAN_BLOCKS
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthRadixScanBlocks_\(digit)"
            encoder.setComputePipelineState(scanBlocksPipeline)
            encoder.setBuffer(sortBuffers.histogram, offset: 0, index: 0)
            encoder.setBuffer(sortBuffers.blockSums, offset: 0, index: 1)
            encoder.setBuffer(header, offset: 0, index: 2)

            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: DepthFirstDispatchSlot.depthScanBlocks.offset,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }

        // C. Exclusive scan - uses DF_SLOT_DEPTH_EXCLUSIVE
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthRadixExclusive_\(digit)"
            encoder.setComputePipelineState(exclusiveScanPipeline)
            encoder.setBuffer(sortBuffers.blockSums, offset: 0, index: 0)
            encoder.setBuffer(header, offset: 0, index: 1)
            encoder.setThreadgroupMemoryLength(blockSize * MemoryLayout<UInt32>.stride, index: 0)

            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: DepthFirstDispatchSlot.depthExclusive.offset,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }

        // D. Apply offsets - uses DF_SLOT_DEPTH_APPLY
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthRadixApply_\(digit)"
            encoder.setComputePipelineState(applyOffsetsPipeline)
            encoder.setBuffer(sortBuffers.histogram, offset: 0, index: 0)
            encoder.setBuffer(sortBuffers.blockSums, offset: 0, index: 1)
            encoder.setBuffer(sortBuffers.scannedHistogram, offset: 0, index: 2)
            encoder.setBuffer(header, offset: 0, index: 3)
            encoder.setThreadgroupMemoryLength(blockSize * MemoryLayout<UInt32>.stride, index: 0)

            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: DepthFirstDispatchSlot.depthApply.offset,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }

        // E. Scatter - uses DF_SLOT_DEPTH_SCATTER
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthRadixScatter_\(digit)"
            encoder.setComputePipelineState(scatterPipeline)
            encoder.setBuffer(destKeys, offset: 0, index: 0)
            encoder.setBuffer(sourceKeys, offset: 0, index: 1)
            encoder.setBuffer(destPayload, offset: 0, index: 2)
            encoder.setBuffer(sourcePayload, offset: 0, index: 3)
            encoder.setBuffer(sortBuffers.scannedHistogram, offset: 0, index: 5)
            encoder.setBytes(&currentDigitUInt, length: 4, index: 6)
            encoder.setBuffer(header, offset: 0, index: 7)

            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: DepthFirstDispatchSlot.depthScatter.offset,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }
    }
}

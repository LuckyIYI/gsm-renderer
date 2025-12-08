import Metal

/// Radix sort encoder for 32-bit keys (single uint)
/// Key format: [tile:16][depth:16] - tile in high bits for primary sort
final class RadixSortEncoder {
    let histogramPipeline: MTLComputePipelineState
    let scanBlocksPipeline: MTLComputePipelineState
    let exclusiveScanPipeline: MTLComputePipelineState
    let applyOffsetsPipeline: MTLComputePipelineState
    let scatterPipeline: MTLComputePipelineState

    let blockSize = 256
    let grainSize = 4
    let radix = 256

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard
            let histFn = library.makeFunction(name: "radixHistogramKernel"),
            let scanFn = library.makeFunction(name: "radixScanBlocksKernel"),
            let exclusiveFn = library.makeFunction(name: "radixExclusiveScanKernel"),
            let applyFn = library.makeFunction(name: "radixApplyScanOffsetsKernel"),
            let scatterFn = library.makeFunction(name: "radixScatterKernel")
        else {
            throw RendererError.failedToCreatePipeline("Radix sort functions missing from library")
        }

        self.histogramPipeline = try device.makeComputePipelineState(function: histFn)
        self.scanBlocksPipeline = try device.makeComputePipelineState(function: scanFn)
        self.exclusiveScanPipeline = try device.makeComputePipelineState(function: exclusiveFn)
        self.applyOffsetsPipeline = try device.makeComputePipelineState(function: applyFn)
        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)
    }

    /// Encode radix sort for 32-bit keys
    /// - Parameters:
    ///   - keyBuffer: Input/output 32-bit keys (uint)
    ///   - sortedIndices: Input/output payload indices (Int32)
    ///   - header: TileAssignmentHeader buffer
    ///   - dispatchArgs: Indirect dispatch arguments
    ///   - radixBuffers: Scratch buffers for radix sort
    ///   - offsets: Dispatch offsets for histogram, scanBlocks, exclusive, apply, scatter
    ///   - tileCount: Number of tiles (determines how many passes needed)
    func encode(
        commandBuffer: MTLCommandBuffer,
        keyBuffer: MTLBuffer, // Input 32-bit keys (uint), also output
        sortedIndices: MTLBuffer, // Input indices (Int32), also output indices
        header: MTLBuffer,
        dispatchArgs: MTLBuffer,
        radixBuffers: RadixBufferSet,
        offsets: (histogram: Int, scanBlocks: Int,
                  exclusive: Int, apply: Int, scatter: Int),
        tileCount: Int
    ) {
        // Determine number of passes needed for 32-bit key
        // Key layout: [tile:16][depth:16]
        // depth is in low 16 bits (2 bytes), tile in high 16 bits
        let depthBytes = 2 // 16-bit depth
        var tileBytes = 1
        var remainingTiles = max(tileCount - 1, 0)
        while remainingTiles >= 256 {
            tileBytes += 1
            remainingTiles >>= 8
        }
        // Max 4 passes for 32-bit key
        let passCount = min(4, depthBytes + tileBytes)

        // Ping-pong between keyBuffer/scratchKeys and sortedIndices/scratchPayload
        var sourceKeys = keyBuffer
        var destKeys = radixBuffers.scratchKeys
        var sourcePayload = sortedIndices
        var destPayload = radixBuffers.scratchPayload

        for digit in 0 ..< passCount {
            self.encodeOnePass(
                commandBuffer: commandBuffer,
                digit: digit,
                sourceKeys: sourceKeys,
                destKeys: destKeys,
                sourcePayload: sourcePayload,
                destPayload: destPayload,
                dispatchArgs: dispatchArgs,
                radixBuffers: radixBuffers,
                offsets: offsets,
                header: header
            )
            swap(&sourceKeys, &destKeys)
            swap(&sourcePayload, &destPayload)
        }

        // If passCount is odd, sorted data is in scratch buffers - copy back
        if passCount % 2 != 0 {
            if let blit = commandBuffer.makeBlitCommandEncoder() {
                blit.label = "CopyRadixResults"
                blit.copy(from: radixBuffers.scratchKeys, sourceOffset: 0,
                          to: keyBuffer, destinationOffset: 0,
                          size: keyBuffer.length)
                blit.copy(from: radixBuffers.scratchPayload, sourceOffset: 0,
                          to: sortedIndices, destinationOffset: 0,
                          size: sortedIndices.length)
                blit.endEncoding()
            }
        }
    }

    private func encodeOnePass(
        commandBuffer: MTLCommandBuffer,
        digit: Int,
        sourceKeys: MTLBuffer,
        destKeys: MTLBuffer,
        sourcePayload: MTLBuffer,
        destPayload: MTLBuffer,
        dispatchArgs: MTLBuffer,
        radixBuffers: RadixBufferSet,
        offsets: (histogram: Int, scanBlocks: Int,
                  exclusive: Int, apply: Int, scatter: Int),
        header: MTLBuffer
    ) {
        var currentDigitUInt = UInt32(digit)

        // A. Histogram
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "RadixHist_\(digit)"
            encoder.setComputePipelineState(self.histogramPipeline)
            encoder.setBuffer(sourceKeys, offset: 0, index: 0)
            encoder.setBuffer(radixBuffers.histogram, offset: 0, index: 1)

            encoder.setBytes(&currentDigitUInt, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBuffer(header, offset: 0, index: 4)

            let tg = MTLSize(width: blockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: offsets.histogram,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }

        // B. Scan blocks (reduce histogram into block sums)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "RadixScanBlocks_\(digit)"
            encoder.setComputePipelineState(self.scanBlocksPipeline)
            encoder.setBuffer(radixBuffers.histogram, offset: 0, index: 0)
            encoder.setBuffer(radixBuffers.blockSums, offset: 0, index: 1)
            encoder.setBuffer(header, offset: 0, index: 2)

            let tg = MTLSize(width: blockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: offsets.scanBlocks,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }

        // C. Exclusive scan of block sums
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "RadixExclusive_\(digit)"
            encoder.setComputePipelineState(self.exclusiveScanPipeline)
            encoder.setBuffer(radixBuffers.blockSums, offset: 0, index: 0)
            encoder.setBuffer(header, offset: 0, index: 1)

            encoder.setThreadgroupMemoryLength(self.blockSize * MemoryLayout<UInt32>.stride, index: 0)

            let threadsPerScanGroup = MTLSize(width: blockSize, height: 1, depth: 1)
            let threadgroupsScan = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroupsScan, threadsPerThreadgroup: threadsPerScanGroup)
            encoder.endEncoding()
        }

        // D. Apply scanned block offsets
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "RadixApply_\(digit)"
            encoder.setComputePipelineState(self.applyOffsetsPipeline)
            encoder.setBuffer(radixBuffers.histogram, offset: 0, index: 0)
            encoder.setBuffer(radixBuffers.blockSums, offset: 0, index: 1)
            encoder.setBuffer(radixBuffers.scannedHistogram, offset: 0, index: 2)
            encoder.setBuffer(header, offset: 0, index: 3)

            encoder.setThreadgroupMemoryLength(self.blockSize * MemoryLayout<UInt32>.stride, index: 0)

            let tg = MTLSize(width: blockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: offsets.apply,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }

        // E. Scatter
        // Shader buffer order: output_keys(0), input_keys(1), output_payload(2), input_payload(3),
        //                      offsets_flat(5), current_digit(6), header(7)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "RadixScatter_\(digit)"
            encoder.setComputePipelineState(self.scatterPipeline)
            encoder.setBuffer(destKeys, offset: 0, index: 0) // output_keys
            encoder.setBuffer(sourceKeys, offset: 0, index: 1) // input_keys
            encoder.setBuffer(destPayload, offset: 0, index: 2) // output_payload
            encoder.setBuffer(sourcePayload, offset: 0, index: 3) // input_payload
            encoder.setBuffer(radixBuffers.scannedHistogram, offset: 0, index: 5) // offsets_flat
            encoder.setBytes(&currentDigitUInt, length: MemoryLayout<UInt32>.stride, index: 6) // current_digit
            encoder.setBuffer(header, offset: 0, index: 7) // header

            encoder.setThreadgroupMemoryLength(self.radix * MemoryLayout<UInt32>.stride, index: 0)
            encoder.setThreadgroupMemoryLength(self.radix * MemoryLayout<UInt32>.stride, index: 1)

            let tg = MTLSize(width: blockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: offsets.scatter,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }
    }
}

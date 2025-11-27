import Metal

final class RadixSortEncoder {
    let histogramPipeline: MTLComputePipelineState
    let scanBlocksPipeline: MTLComputePipelineState
    let exclusiveScanPipeline: MTLComputePipelineState
    let applyOffsetsPipeline: MTLComputePipelineState
    let scatterPipeline: MTLComputePipelineState
    
    let fusePipeline: MTLComputePipelineState
    let unpackPipeline: MTLComputePipelineState
    
    let blockSize = 256
    let grainSize = 4
    let radix = 256
    
    let fuseThreadgroupSize: Int
    let unpackThreadgroupSize: Int

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard
            let fuseFn = library.makeFunction(name: "fuseSortKeysKernel"),
            let unpackFn = library.makeFunction(name: "unpackSortKeysKernel"),
            let histFn = library.makeFunction(name: "radixHistogramKernel"),
            let scanFn = library.makeFunction(name: "radixScanBlocksKernel"),
            let exclusiveFn = library.makeFunction(name: "radixExclusiveScanKernel"),
            let applyFn = library.makeFunction(name: "radixApplyScanOffsetsKernel"),
            let scatterFn = library.makeFunction(name: "radixScatterKernel")
        else {
            fatalError("Radix sort functions missing from library")
        }
        
        self.fusePipeline = try device.makeComputePipelineState(function: fuseFn)
        self.unpackPipeline = try device.makeComputePipelineState(function: unpackFn)
        self.histogramPipeline = try device.makeComputePipelineState(function: histFn)
        self.scanBlocksPipeline = try device.makeComputePipelineState(function: scanFn)
        self.exclusiveScanPipeline = try device.makeComputePipelineState(function: exclusiveFn)
        self.applyOffsetsPipeline = try device.makeComputePipelineState(function: applyFn)
        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)
        
        self.fuseThreadgroupSize = max(1, min(fusePipeline.maxTotalThreadsPerThreadgroup, 256))
        self.unpackThreadgroupSize = max(1, min(unpackPipeline.maxTotalThreadsPerThreadgroup, 256))
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        keyBuffer: MTLBuffer,          // Input keys (e.g. SIMD2<UInt32>)
        sortedIndices: MTLBuffer,      // Input indices (Int32), also output indices
        header: MTLBuffer,
        dispatchArgs: MTLBuffer,
        radixBuffers: RadixBufferSet,
        offsets: (fuse: Int, unpack: Int,
                  histogram: Int, scanBlocks: Int,
                  exclusive: Int, apply: Int, scatter: Int),
        tileCount: Int
    ) {
        // 1. Fuse keys
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "FuseKeys"
            encoder.setComputePipelineState(fusePipeline)
            encoder.setBuffer(keyBuffer, offset: 0, index: 0)
            encoder.setBuffer(radixBuffers.fusedKeys, offset: 0, index: 1)
            
            let tg = MTLSize(width: fuseThreadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: offsets.fuse,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }
        
        // Decide how many byte-wide passes we need.
        // Use 16-bit depth quantization (2 bytes) for faster sorting - works well for gaussian splatting
        // since relative depth order within a tile matters more than absolute precision
        let depthBytes = 2  // 16-bit depth = 65536 levels per tile (plenty for correct ordering)
        var tileBytes = 1
        var remainingTiles = max(tileCount - 1, 0)
        while remainingTiles >= 256 {
            tileBytes += 1
            remainingTiles >>= 8
        }
        // Only do as many tile bytes as actually needed - extra passes with all elements
        // in the same bin will scramble the sorted order due to atomic racing
        let passCount = min(8, depthBytes + tileBytes)

        var sourceKeys = radixBuffers.fusedKeys
        var destKeys = radixBuffers.scratchKeys
        var sourcePayload = sortedIndices
        var destPayload = radixBuffers.scratchPayload

        for digit in 0..<passCount {
            encodeOnePass(
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

        // If passCount is odd, the sorted payload is in scratchPayload, not sortedIndices.
        // Copy it back to sortedIndices.
        if passCount % 2 != 0 {
            if let blit = commandBuffer.makeBlitCommandEncoder() {
                blit.label = "CopyPayload"
                blit.copy(from: radixBuffers.scratchPayload, sourceOffset: 0,
                         to: sortedIndices, destinationOffset: 0,
                         size: sortedIndices.length)
                blit.endEncoding()
            }
        }

        // 3. Unpack keys back to original layout
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "UnpackKeys"
            encoder.setComputePipelineState(unpackPipeline)
            encoder.setBuffer(sourceKeys, offset: 0, index: 0)
            encoder.setBuffer(keyBuffer, offset: 0, index: 1)
            
            let tg = MTLSize(width: unpackThreadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: offsets.unpack,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
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
        offsets: (fuse: Int, unpack: Int,
                  histogram: Int, scanBlocks: Int,
                  exclusive: Int, apply: Int, scatter: Int),
        header: MTLBuffer
    ) {
        var currentDigitUInt = UInt32(digit)
        
        // A. Histogram
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "RadixHist_\(digit)"
            encoder.setComputePipelineState(histogramPipeline)
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
            encoder.setComputePipelineState(scanBlocksPipeline)
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
        
        // C. Exclusive scan of block sums (using ThreadgroupPrefixScan helper)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "RadixExclusive_\(digit)"
            encoder.setComputePipelineState(exclusiveScanPipeline)
            encoder.setBuffer(radixBuffers.blockSums, offset: 0, index: 0)
            encoder.setBuffer(header, offset: 0, index: 1)

            // Allocate threadgroup memory for prefix scan
            encoder.setThreadgroupMemoryLength(blockSize * MemoryLayout<UInt32>.stride, index: 0)

            // Dispatch one threadgroup with 256 threads
            let threadsPerScanGroup = MTLSize(width: blockSize, height: 1, depth: 1)
            let threadgroupsScan = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroupsScan, threadsPerThreadgroup: threadsPerScanGroup)
            encoder.endEncoding()
        }
        
        // D. Apply scanned block offsets → full scanned histogram
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "RadixApply_\(digit)"
            encoder.setComputePipelineState(applyOffsetsPipeline)
            encoder.setBuffer(radixBuffers.histogram, offset: 0, index: 0) // hist_flat
            encoder.setBuffer(radixBuffers.blockSums, offset: 0, index: 1) // scanned_blocks
            encoder.setBuffer(radixBuffers.scannedHistogram, offset: 0, index: 2)
            encoder.setBuffer(header, offset: 0, index: 3)
            
            encoder.setThreadgroupMemoryLength(blockSize * MemoryLayout<UInt32>.stride, index: 0) // shared_mem
            
            let tg = MTLSize(width: blockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: offsets.apply,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }
        
        // E. Scatter
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "RadixScatter_\(digit)"
            encoder.setComputePipelineState(scatterPipeline)
            encoder.setBuffer(destKeys, offset: 0, index: 0)      // output_keys
            encoder.setBuffer(sourceKeys, offset: 0, index: 1)    // input_keys
            encoder.setBuffer(destPayload, offset: 0, index: 2)   // output_payload
            encoder.setBuffer(sourcePayload, offset: 0, index: 3) // input_payload
            
            encoder.setBuffer(radixBuffers.scannedHistogram, offset: 0, index: 5) // offsets_flat
            encoder.setBytes(&currentDigitUInt, length: MemoryLayout<UInt32>.stride, index: 6) // current_digit
            encoder.setBuffer(header, offset: 0, index: 7)
            
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

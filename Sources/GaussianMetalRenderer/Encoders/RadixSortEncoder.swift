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
            fatalError("Radix sort functions missing")
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
        keyBuffer: MTLBuffer,          // Input keys (SIMD2<UInt32>)
        sortedIndices: MTLBuffer,      // Input indices (Int32), also Output indices
        header: MTLBuffer,
        dispatchArgs: MTLBuffer,
        radixBuffers: RadixBufferSet,
        offsets: (fuse: Int, unpack: Int, histogram: Int, scanBlocks: Int, exclusive: Int, apply: Int, scatter: Int)
    ) {        
        // 1. Fuse Keys
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "FuseKeys"
            encoder.setComputePipelineState(self.fusePipeline)
            encoder.setBuffer(keyBuffer, offset: 0, index: 0)
            encoder.setBuffer(radixBuffers.fusedKeys, offset: 0, index: 1)
                        
            let tg = MTLSize(width: self.fuseThreadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: offsets.fuse,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }
        //123
        // 2. Radix Passes (8 passes for 64-bit keys)
        var sourceKeys = radixBuffers.fusedKeys
        var destKeys = radixBuffers.scratchKeys
        
        var sourcePayload = sortedIndices
        var destPayload = radixBuffers.scratchPayload
        
        for shift in 0..<8 {
            self.encodeOnePass(
                commandBuffer: commandBuffer,
                shift: shift,
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
        
        // 3. Unpack Keys
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "UnpackKeys"
            encoder.setComputePipelineState(self.unpackPipeline)
            encoder.setBuffer(sourceKeys, offset: 0, index: 0)
            encoder.setBuffer(keyBuffer, offset: 0, index: 1)
            
            let tg = MTLSize(width: self.unpackThreadgroupSize, height: 1, depth: 1)
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
        shift: Int,
        sourceKeys: MTLBuffer,
        destKeys: MTLBuffer,
        sourcePayload: MTLBuffer,
        destPayload: MTLBuffer,
        dispatchArgs: MTLBuffer,
        radixBuffers: RadixBufferSet,
        offsets: (fuse: Int, unpack: Int, histogram: Int, scanBlocks: Int, exclusive: Int, apply: Int, scatter: Int),
        header: MTLBuffer
    ) {
        var shiftUInt = UInt32(shift)
        
        // A. Histogram
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "RadixHist_\(shift)"
            encoder.setComputePipelineState(self.histogramPipeline)
            encoder.setBuffer(sourceKeys, offset: 0, index: 0)
            encoder.setBuffer(radixBuffers.histogram, offset: 0, index: 1)
            
            encoder.setBytes(&shiftUInt, length: 4, index: 3)
            encoder.setBuffer(header, offset: 0, index: 4)
            
            let tg = MTLSize(width: self.blockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: offsets.histogram,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }
        
        // B. Scan Blocks
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "RadixScanBlocks_\(shift)"
            encoder.setComputePipelineState(self.scanBlocksPipeline)
            encoder.setBuffer(radixBuffers.histogram, offset: 0, index: 0)
            encoder.setBuffer(radixBuffers.blockSums, offset: 0, index: 1)
            
            
            let tg = MTLSize(width: self.blockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: offsets.scanBlocks,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }
        
        // C. Exclusive Scan
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "RadixExclusive_\(shift)"
            encoder.setComputePipelineState(self.exclusiveScanPipeline)
            encoder.setBuffer(radixBuffers.blockSums, offset: 0, index: 0)
            encoder.setBuffer(header, offset: 0, index: 1)
            
            let tg = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: offsets.exclusive,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }
        
        // D. Apply Offsets
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "RadixApply_\(shift)"
            encoder.setComputePipelineState(self.applyOffsetsPipeline)
            encoder.setBuffer(radixBuffers.histogram, offset: 0, index: 0) // Input (Histogram)
            encoder.setBuffer(radixBuffers.blockSums, offset: 0, index: 1) // Input (BlockSums)
            encoder.setBuffer(radixBuffers.scannedHistogram, offset: 0, index: 2)
                        
            encoder.setThreadgroupMemoryLength(self.blockSize * 4, index: 0)
            
            let tg = MTLSize(width: self.blockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: offsets.apply,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }
        
        // E. Scatter
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "RadixScatter_\(shift)"
            encoder.setComputePipelineState(self.scatterPipeline)
            encoder.setBuffer(destKeys, offset: 0, index: 0)      // output_keys
            encoder.setBuffer(sourceKeys, offset: 0, index: 1)    // input_keys
            encoder.setBuffer(destPayload, offset: 0, index: 2)   // output_payload
            encoder.setBuffer(sourcePayload, offset: 0, index: 3) // input_payload
            
            
            encoder.setBuffer(radixBuffers.scannedHistogram, offset: 0, index: 5) // offsets_data
            encoder.setBytes(&shiftUInt, length: MemoryLayout<UInt32>.stride, index: 6) // current_digit
            encoder.setBuffer(header, offset: 0, index: 7)
            
            let tg = MTLSize(width: self.blockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchArgs,
                indirectBufferOffset: offsets.scatter,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }
    }
}

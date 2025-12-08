import Metal

/// Encodes tile prefix scan stage: tileCounts → tileOffsets with partial sums
final class LocalPrefixScanEncoder {
    private let prefixScanPipeline: MTLComputePipelineState
    private let scanPartialSumsPipeline: MTLComputePipelineState
    private let finalizeScanAndZeroPipeline: MTLComputePipelineState

    private let prefixBlockSize = 256
    private let prefixGrainSize = 4

    init(library: MTLLibrary, device: MTLDevice) throws {
        guard let prefixFn = library.makeFunction(name: "localPrefixScan"),
              let partialFn = library.makeFunction(name: "localScanPartialSums"),
              let finalizeAndZeroFn = library.makeFunction(name: "localFinalizeScanAndZero")
        else {
            throw RendererError.failedToCreatePipeline("Missing prefix scan kernels")
        }
        self.prefixScanPipeline = try device.makeComputePipelineState(function: prefixFn)
        self.scanPartialSumsPipeline = try device.makeComputePipelineState(function: partialFn)
        self.finalizeScanAndZeroPipeline = try device.makeComputePipelineState(function: finalizeAndZeroFn)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        tileCounts: MTLBuffer,
        tileOffsets: MTLBuffer,
        partialSums: MTLBuffer,
        tileCount: Int,
        activeTileIndices: MTLBuffer,
        activeTileCount: MTLBuffer
    ) {
        let elementsPerGroup = self.prefixBlockSize * self.prefixGrainSize
        let actualGroups = max(1, (tileCount + elementsPerGroup - 1) / elementsPerGroup)
        var tileCountU = UInt32(tileCount)

        // Prefix scan per block
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_PrefixScan"
            encoder.setComputePipelineState(self.prefixScanPipeline)
            encoder.setBuffer(tileCounts, offset: 0, index: 0)
            encoder.setBuffer(tileOffsets, offset: 0, index: 1)
            encoder.setBytes(&tileCountU, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBuffer(partialSums, offset: 0, index: 3)
            let tg = MTLSize(width: prefixBlockSize, height: 1, depth: 1)
            let tgs = MTLSize(width: actualGroups, height: 1, depth: 1)
            encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Scan partial sums
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_ScanPartialSums"
            encoder.setComputePipelineState(self.scanPartialSumsPipeline)
            var numPartial = UInt32(actualGroups)
            encoder.setBuffer(partialSums, offset: 0, index: 0)
            encoder.setBytes(&numPartial, length: MemoryLayout<UInt32>.stride, index: 1)
            let tg = MTLSize(width: prefixBlockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Clear activeTileCount before finalize pass
        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            blitEncoder.label = "Local_ClearActiveTileCount"
            blitEncoder.fill(buffer: activeTileCount, range: 0 ..< MemoryLayout<UInt32>.stride, value: 0)
            blitEncoder.endEncoding()
        }

        // Finalize scan + zero counters + compact active tiles (fused)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_FinalizeScanAndZero"
            encoder.setComputePipelineState(self.finalizeScanAndZeroPipeline)
            encoder.setBuffer(tileOffsets, offset: 0, index: 0)
            encoder.setBuffer(tileCounts, offset: 0, index: 1)
            encoder.setBytes(&tileCountU, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBuffer(partialSums, offset: 0, index: 3)
            encoder.setBuffer(activeTileIndices, offset: 0, index: 4)
            encoder.setBuffer(activeTileCount, offset: 0, index: 5)
            let tg = MTLSize(width: prefixBlockSize, height: 1, depth: 1)
            let tgs = MTLSize(width: actualGroups, height: 1, depth: 1)
            encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }
}

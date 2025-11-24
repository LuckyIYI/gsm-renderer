import Metal

final class CoverageEncoder {
    private let coveragePipeline: MTLComputePipelineState
    private let coveragePipelineHalf: MTLComputePipelineState?
    private let prefixPipeline: MTLComputePipelineState
    private let partialPipeline: MTLComputePipelineState
    private let finalizePipeline: MTLComputePipelineState
    private let storeTotalPipeline: MTLComputePipelineState

    private let prefixBlockSize = 256
    private let prefixGrainSize = 4

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard
            let coverageFn = library.makeFunction(name: "gaussianCoverageKernel_float"),
            let prefixFn = library.makeFunction(name: "coveragePrefixScanKernel"),
            let partialFn = library.makeFunction(name: "coverageScanPartialSumsKernel"),
            let finalizeFn = library.makeFunction(name: "coverageFinalizeScanKernel"),
            let storeTotalFn = library.makeFunction(name: "coverageStoreTotalKernel")
        else {
            fatalError("Coverage functions missing")
        }

        self.coveragePipeline = try device.makeComputePipelineState(function: coverageFn)
        if let coverageHalfFn = library.makeFunction(name: "gaussianCoverageKernel_half") {
            self.coveragePipelineHalf = try? device.makeComputePipelineState(function: coverageHalfFn)
        } else {
            self.coveragePipelineHalf = nil
        }
        self.prefixPipeline = try device.makeComputePipelineState(function: prefixFn)
        self.partialPipeline = try device.makeComputePipelineState(function: partialFn)
        self.finalizePipeline = try device.makeComputePipelineState(function: finalizeFn)
        self.storeTotalPipeline = try device.makeComputePipelineState(function: storeTotalFn)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        boundsBuffer: MTLBuffer,
        opacitiesBuffer: MTLBuffer,
        coverageBuffer: MTLBuffer,
        offsetsBuffer: MTLBuffer,
        partialSumsBuffer: MTLBuffer,
        tileAssignmentHeader: MTLBuffer,
        precision: Precision = .float32
    ) {
        let elementsPerGroup = prefixBlockSize * prefixGrainSize
        let actualGroups = max(1, (gaussianCount + elementsPerGroup - 1) / elementsPerGroup)

        // 1. Calculate Coverage
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Coverage"
            var params = CoverageParamsSwift(gaussianCount: UInt32(gaussianCount))
            if precision == .float16, let halfPipe = self.coveragePipelineHalf {
                encoder.setComputePipelineState(halfPipe)
            } else {
                encoder.setComputePipelineState(self.coveragePipeline)
            }
            encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 1)
            encoder.setBuffer(opacitiesBuffer, offset: 0, index: 2)
            encoder.setBytes(&params, length: MemoryLayout<CoverageParamsSwift>.stride, index: 3)
            let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
            let activePipeline = (precision == .float16 && self.coveragePipelineHalf != nil) ? self.coveragePipelineHalf! : self.coveragePipeline
            let tgWidth = activePipeline.threadExecutionWidth
            let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
        
        // 2. Prefix Scan
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Prefix"
            var countUInt = UInt32(gaussianCount)
            encoder.setComputePipelineState(self.prefixPipeline)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 0)
            encoder.setBuffer(offsetsBuffer, offset: 0, index: 1)
            encoder.setBytes(&countUInt, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBuffer(partialSumsBuffer, offset: 0, index: 3)
            let threadgroups = MTLSize(width: actualGroups, height: 1, depth: 1)
            let threads = MTLSize(width: prefixBlockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threads)
            encoder.endEncoding()
        }
        
        // 3. Partial Scan
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "PartialScan"
            var numPartial = UInt32(actualGroups)
            encoder.setComputePipelineState(self.partialPipeline)
            encoder.setBuffer(partialSumsBuffer, offset: 0, index: 0)
            encoder.setBytes(&numPartial, length: MemoryLayout<UInt32>.stride, index: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            let threads = MTLSize(width: prefixBlockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threads)
            encoder.endEncoding()
        }
        
        // 4. Finalize Scan
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Finalize"
            var countUInt = UInt32(gaussianCount)
            encoder.setComputePipelineState(self.finalizePipeline)
            encoder.setBuffer(offsetsBuffer, offset: 0, index: 0)
            encoder.setBytes(&countUInt, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.setBuffer(partialSumsBuffer, offset: 0, index: 2)
            let threadgroups = MTLSize(width: actualGroups, height: 1, depth: 1)
            let threads = MTLSize(width: prefixBlockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threads)
            encoder.endEncoding()
        }
        
        // 5. Store Total
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "StoreTotal"
            var countUInt = UInt32(gaussianCount)
            encoder.setComputePipelineState(self.storeTotalPipeline)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 0)
            encoder.setBuffer(offsetsBuffer, offset: 0, index: 1)
            encoder.setBytes(&countUInt, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBuffer(tileAssignmentHeader, offset: 0, index: 3)
            let threads = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: threads)
            encoder.endEncoding()
        }
    }
}

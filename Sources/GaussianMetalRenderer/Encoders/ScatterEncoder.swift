import Metal

final class ScatterEncoder {
    private let scatterPipeline: MTLComputePipelineState
    private let scatterBalancedPipeline: MTLComputePipelineState
    private let dispatchPipeline: MTLComputePipelineState
    private let dispatchBalancedPipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard
            let scatterFn = library.makeFunction(name: "scatterAssignmentsKernel"),
            let scatterBalancedFn = library.makeFunction(name: "scatterAssignmentsBalancedKernel"),
            let dispatchFn = library.makeFunction(name: "scatterDispatchKernel"),
            let dispatchBalancedFn = library.makeFunction(name: "scatterBalancedDispatchKernel")
        else {
            fatalError("Scatter functions missing")
        }

        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)
        self.scatterBalancedPipeline = try device.makeComputePipelineState(function: scatterBalancedFn)
        self.dispatchPipeline = try device.makeComputePipelineState(function: dispatchFn)
        self.dispatchBalancedPipeline = try device.makeComputePipelineState(function: dispatchBalancedFn)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        tilesX: Int,
        offsetsBuffer: MTLBuffer,
        dispatchBuffer: MTLBuffer,
        boundsBuffer: MTLBuffer,
        tileIndicesBuffer: MTLBuffer,
        tileIdsBuffer: MTLBuffer,
        tileAssignmentHeader: MTLBuffer
    ) {
        // 1. Prepare Indirect Dispatch
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "ScatterDispatch"
            var dispatchParams = ScatterDispatchParamsSwift(
                threadgroupWidth: UInt32(max(1, self.scatterPipeline.threadExecutionWidth)),
                gaussianCount: UInt32(gaussianCount)
            )
            encoder.setComputePipelineState(self.dispatchPipeline)
            encoder.setBuffer(offsetsBuffer, offset: 0, index: 0)
            encoder.setBuffer(dispatchBuffer, offset: 0, index: 1)
            encoder.setBytes(&dispatchParams, length: MemoryLayout<ScatterDispatchParamsSwift>.stride, index: 2)
            let threads = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: threads)
            encoder.endEncoding()
        }
        
        // 2. Scatter
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Scatter"
            var scatterParams = ScatterParamsSwift(gaussianCount: UInt32(gaussianCount), tilesX: UInt32(tilesX))
            encoder.setComputePipelineState(self.scatterPipeline)
            encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
            encoder.setBuffer(offsetsBuffer, offset: 0, index: 1)
            encoder.setBuffer(tileIndicesBuffer, offset: 0, index: 2)
            encoder.setBuffer(tileIdsBuffer, offset: 0, index: 3)
            encoder.setBytes(&scatterParams, length: MemoryLayout<ScatterParamsSwift>.stride, index: 4)
            encoder.setBuffer(tileAssignmentHeader, offset: 0, index: 5)
            
            let tgWidth = max(1, self.scatterPipeline.threadExecutionWidth)
            let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchBuffer,
                indirectBufferOffset: 0,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }
    }

    /// Load-balanced scatter: each thread handles exactly one output slot
    /// Uses binary search - perfect load balancing, no nested loops
    func encodeBalanced(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        tilesX: Int,
        offsetsBuffer: MTLBuffer,
        dispatchBuffer: MTLBuffer,
        boundsBuffer: MTLBuffer,
        tileIndicesBuffer: MTLBuffer,
        tileIdsBuffer: MTLBuffer,
        tileAssignmentHeader: MTLBuffer
    ) {
        // 1. Prepare Indirect Dispatch (reads totalAssignments from header)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "ScatterBalancedDispatch"
            var tgWidth = UInt32(max(1, self.scatterBalancedPipeline.threadExecutionWidth))
            encoder.setComputePipelineState(self.dispatchBalancedPipeline)
            encoder.setBuffer(tileAssignmentHeader, offset: 0, index: 0)
            encoder.setBuffer(dispatchBuffer, offset: 0, index: 1)
            encoder.setBytes(&tgWidth, length: MemoryLayout<UInt32>.stride, index: 2)
            let threads = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: threads)
            encoder.endEncoding()
        }

        // 2. Balanced Scatter (one thread per output slot)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "ScatterBalanced"
            var scatterParams = ScatterParamsSwift(gaussianCount: UInt32(gaussianCount), tilesX: UInt32(tilesX))
            encoder.setComputePipelineState(self.scatterBalancedPipeline)
            encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
            encoder.setBuffer(offsetsBuffer, offset: 0, index: 1)
            encoder.setBuffer(tileIndicesBuffer, offset: 0, index: 2)
            encoder.setBuffer(tileIdsBuffer, offset: 0, index: 3)
            encoder.setBytes(&scatterParams, length: MemoryLayout<ScatterParamsSwift>.stride, index: 4)
            encoder.setBuffer(tileAssignmentHeader, offset: 0, index: 5)

            let tgWidth = max(1, self.scatterBalancedPipeline.threadExecutionWidth)
            let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                indirectBuffer: dispatchBuffer,
                indirectBufferOffset: 0,
                threadsPerThreadgroup: tg
            )
            encoder.endEncoding()
        }
    }
}

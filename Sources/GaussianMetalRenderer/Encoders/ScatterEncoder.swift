import Metal

final class ScatterEncoder {
    private let scatterBalancedPipeline: MTLComputePipelineState
    private let scatterPrecisePipeline: MTLComputePipelineState?
    private let scatterPrecisePipelineHalf: MTLComputePipelineState?
    private let scatterPreciseBalancedPipeline: MTLComputePipelineState?
    private let scatterPreciseBalancedPipelineHalf: MTLComputePipelineState?
    private let dispatchBalancedPipeline: MTLComputePipelineState

    public var isPreciseAvailable: Bool { scatterPrecisePipeline != nil }

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard
            let scatterBalancedFn = library.makeFunction(name: "scatterAssignmentsBalancedKernel"),
            let dispatchBalancedFn = library.makeFunction(name: "scatterBalancedDispatchKernel")
        else {
            fatalError("Scatter functions missing")
        }

        self.scatterBalancedPipeline = try device.makeComputePipelineState(function: scatterBalancedFn)
        self.dispatchBalancedPipeline = try device.makeComputePipelineState(function: dispatchBalancedFn)

        // FlashGS precise scatter kernels
        if let preciseFn = library.makeFunction(name: "scatterAssignmentsPreciseKernel_float") {
            self.scatterPrecisePipeline = try? device.makeComputePipelineState(function: preciseFn)
        } else {
            self.scatterPrecisePipeline = nil
        }
        if let preciseHalfFn = library.makeFunction(name: "scatterAssignmentsPreciseKernel_half") {
            self.scatterPrecisePipelineHalf = try? device.makeComputePipelineState(function: preciseHalfFn)
        } else {
            self.scatterPrecisePipelineHalf = nil
        }
        if let preciseBalancedFn = library.makeFunction(name: "scatterAssignmentsPreciseBalancedKernel_float") {
            self.scatterPreciseBalancedPipeline = try? device.makeComputePipelineState(function: preciseBalancedFn)
        } else {
            self.scatterPreciseBalancedPipeline = nil
        }
        if let preciseBalancedHalfFn = library.makeFunction(name: "scatterAssignmentsPreciseBalancedKernel_half") {
            self.scatterPreciseBalancedPipelineHalf = try? device.makeComputePipelineState(function: preciseBalancedHalfFn)
        } else {
            self.scatterPreciseBalancedPipelineHalf = nil
        }
    }

    /// Legacy encode() - redirects to encodeBalanced() for better load balancing
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
        // Redirect to balanced version - same interface, better performance
        encodeBalanced(
            commandBuffer: commandBuffer,
            gaussianCount: gaussianCount,
            tilesX: tilesX,
            offsetsBuffer: offsetsBuffer,
            dispatchBuffer: dispatchBuffer,
            boundsBuffer: boundsBuffer,
            tileIndicesBuffer: tileIndicesBuffer,
            tileIdsBuffer: tileIdsBuffer,
            tileAssignmentHeader: tileAssignmentHeader
        )
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

    /// FlashGS precise scatter - only writes tiles that actually intersect the gaussian ellipse
    /// Parallel version: one threadgroup per gaussian, threads cooperatively test and write
    func encodePrecise(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        tilesX: Int,
        tileWidth: Int,
        tileHeight: Int,
        offsetsBuffer: MTLBuffer,
        dispatchBuffer: MTLBuffer,
        boundsBuffer: MTLBuffer,
        tileIndicesBuffer: MTLBuffer,
        tileIdsBuffer: MTLBuffer,
        tileAssignmentHeader: MTLBuffer,
        meansBuffer: MTLBuffer,
        conicsBuffer: MTLBuffer,
        opacitiesBuffer: MTLBuffer,
        precision: Precision = .float32
    ) {
        // Precise Scatter - one threadgroup per gaussian
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "ScatterPrecise"
            var scatterParams = ScatterParamsSwift(
                gaussianCount: UInt32(gaussianCount),
                tilesX: UInt32(tilesX),
                tileWidth: UInt32(tileWidth),
                tileHeight: UInt32(tileHeight)
            )

            let pipeline: MTLComputePipelineState
            if precision == .float16, let halfPipe = self.scatterPrecisePipelineHalf {
                pipeline = halfPipe
            } else if let precisePipe = self.scatterPrecisePipeline {
                pipeline = precisePipe
            } else {
                // Fall back to balanced scatter
                encoder.endEncoding()
                encodeBalanced(
                    commandBuffer: commandBuffer,
                    gaussianCount: gaussianCount,
                    tilesX: tilesX,
                    offsetsBuffer: offsetsBuffer,
                    dispatchBuffer: dispatchBuffer,
                    boundsBuffer: boundsBuffer,
                    tileIndicesBuffer: tileIndicesBuffer,
                    tileIdsBuffer: tileIdsBuffer,
                    tileAssignmentHeader: tileAssignmentHeader
                )
                return
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
            encoder.setBuffer(offsetsBuffer, offset: 0, index: 1)
            encoder.setBuffer(tileIndicesBuffer, offset: 0, index: 2)
            encoder.setBuffer(tileIdsBuffer, offset: 0, index: 3)
            encoder.setBytes(&scatterParams, length: MemoryLayout<ScatterParamsSwift>.stride, index: 4)
            encoder.setBuffer(tileAssignmentHeader, offset: 0, index: 5)
            encoder.setBuffer(meansBuffer, offset: 0, index: 6)
            encoder.setBuffer(conicsBuffer, offset: 0, index: 7)
            encoder.setBuffer(opacitiesBuffer, offset: 0, index: 8)

            // One threadgroup per gaussian, 32 threads (single SIMD) - fastest
            let threadgroups = MTLSize(width: gaussianCount, height: 1, depth: 1)
            let threadsPerTG = MTLSize(width: 32, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
            encoder.endEncoding()
        }
    }

    /// FlashGS precise balanced scatter - load balanced with precise ellipse intersection
    func encodePreciseBalanced(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        tilesX: Int,
        tileWidth: Int,
        tileHeight: Int,
        offsetsBuffer: MTLBuffer,
        dispatchBuffer: MTLBuffer,
        boundsBuffer: MTLBuffer,
        tileIndicesBuffer: MTLBuffer,
        tileIdsBuffer: MTLBuffer,
        tileAssignmentHeader: MTLBuffer,
        meansBuffer: MTLBuffer,
        conicsBuffer: MTLBuffer,
        opacitiesBuffer: MTLBuffer,
        precision: Precision = .float32
    ) {
        let pipeline: MTLComputePipelineState
        if precision == .float16, let halfPipe = self.scatterPreciseBalancedPipelineHalf {
            pipeline = halfPipe
        } else if let precisePipe = self.scatterPreciseBalancedPipeline {
            pipeline = precisePipe
        } else {
            // Fall back to regular balanced scatter
            self.encodeBalanced(
                commandBuffer: commandBuffer,
                gaussianCount: gaussianCount,
                tilesX: tilesX,
                offsetsBuffer: offsetsBuffer,
                dispatchBuffer: dispatchBuffer,
                boundsBuffer: boundsBuffer,
                tileIndicesBuffer: tileIndicesBuffer,
                tileIdsBuffer: tileIdsBuffer,
                tileAssignmentHeader: tileAssignmentHeader
            )
            return
        }

        // 1. Prepare Indirect Dispatch (reads totalAssignments from header)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "ScatterPreciseBalancedDispatch"
            var tgWidth = UInt32(max(1, pipeline.threadExecutionWidth))
            encoder.setComputePipelineState(self.dispatchBalancedPipeline)
            encoder.setBuffer(tileAssignmentHeader, offset: 0, index: 0)
            encoder.setBuffer(dispatchBuffer, offset: 0, index: 1)
            encoder.setBytes(&tgWidth, length: MemoryLayout<UInt32>.stride, index: 2)
            let threads = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: threads)
            encoder.endEncoding()
        }

        // 2. Precise Balanced Scatter
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "ScatterPreciseBalanced"
            var scatterParams = ScatterParamsSwift(
                gaussianCount: UInt32(gaussianCount),
                tilesX: UInt32(tilesX),
                tileWidth: UInt32(tileWidth),
                tileHeight: UInt32(tileHeight)
            )

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
            encoder.setBuffer(offsetsBuffer, offset: 0, index: 1)
            encoder.setBuffer(tileIndicesBuffer, offset: 0, index: 2)
            encoder.setBuffer(tileIdsBuffer, offset: 0, index: 3)
            encoder.setBytes(&scatterParams, length: MemoryLayout<ScatterParamsSwift>.stride, index: 4)
            encoder.setBuffer(tileAssignmentHeader, offset: 0, index: 5)
            encoder.setBuffer(meansBuffer, offset: 0, index: 6)
            encoder.setBuffer(conicsBuffer, offset: 0, index: 7)
            encoder.setBuffer(opacitiesBuffer, offset: 0, index: 8)

            let tgWidth = max(1, pipeline.threadExecutionWidth)
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

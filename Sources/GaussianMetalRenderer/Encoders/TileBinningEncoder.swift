import Metal

/// Tile-binning pipeline inspired by Tellusim flow:
/// 1) Count gaussians per tile with precise intersection
/// 2) Prefix-scan counts to offsets (reuse coverage scan kernels)
/// 3) Scatter gaussian indices into contiguous per-tile segments
final class TileBinningEncoder {
    private let clearPipeline: MTLComputePipelineState
    private let zeroPipeline: MTLComputePipelineState
    private let countPipeline: MTLComputePipelineState
    private let countPipelineHalf: MTLComputePipelineState?
    private let scatterPipeline: MTLComputePipelineState
    private let scatterPipelineHalf: MTLComputePipelineState?

    // Reuse existing prefix/scan kernels
    private let prefixPipeline: MTLComputePipelineState
    private let partialPipeline: MTLComputePipelineState
    private let finalizePipeline: MTLComputePipelineState
    private let storeTotalPipeline: MTLComputePipelineState

    private let prefixBlockSize = 256
    private let prefixGrainSize = 4

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard
            let clearFn = library.makeFunction(name: "tileBinningClearKernel"),
            let zeroFn = library.makeFunction(name: "tileBinningZeroCountsKernel"),
            let countFn = library.makeFunction(name: "tileBinningCountKernel_float"),
            let scatterFn = library.makeFunction(name: "tileBinningScatterKernel_float"),
            let prefixFn = library.makeFunction(name: "coveragePrefixScanKernel"),
            let partialFn = library.makeFunction(name: "coverageScanPartialSumsKernel"),
            let finalizeFn = library.makeFunction(name: "coverageFinalizeScanKernel"),
            let storeTotalFn = library.makeFunction(name: "coverageStoreTotalKernel")
        else {
            fatalError("Missing tile binning functions in library")
        }

        self.clearPipeline = try device.makeComputePipelineState(function: clearFn)
        self.zeroPipeline = try device.makeComputePipelineState(function: zeroFn)
        self.countPipeline = try device.makeComputePipelineState(function: countFn)
        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)

        if let countHalfFn = library.makeFunction(name: "tileBinningCountKernel_half") {
            self.countPipelineHalf = try? device.makeComputePipelineState(function: countHalfFn)
        } else {
            self.countPipelineHalf = nil
        }
        if let scatterHalfFn = library.makeFunction(name: "tileBinningScatterKernel_half") {
            self.scatterPipelineHalf = try? device.makeComputePipelineState(function: scatterHalfFn)
        } else {
            self.scatterPipelineHalf = nil
        }

        self.prefixPipeline = try device.makeComputePipelineState(function: prefixFn)
        self.partialPipeline = try device.makeComputePipelineState(function: partialFn)
        self.finalizePipeline = try device.makeComputePipelineState(function: finalizeFn)
        self.storeTotalPipeline = try device.makeComputePipelineState(function: storeTotalFn)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        tilesX: Int,
        tilesY: Int,
        tileWidth: Int,
        tileHeight: Int,
        boundsBuffer: MTLBuffer,
        meansBuffer: MTLBuffer,
        conicsBuffer: MTLBuffer,
        opacitiesBuffer: MTLBuffer,
        coverageBuffer: MTLBuffer,
        offsetsBuffer: MTLBuffer,
        partialSumsBuffer: MTLBuffer,
        tileAssignmentHeader: MTLBuffer,
        tileIndicesBuffer: MTLBuffer,
        tileIdsBuffer: MTLBuffer,
        precision: Precision = .float32
    ) {
        let tileCount = tilesX * tilesY
        let elementsPerGroup = prefixBlockSize * prefixGrainSize
        let actualGroups = max(1, (tileCount + elementsPerGroup - 1) / elementsPerGroup)

        var tileCountU = UInt32(tileCount)
        var params = TileBinningParamsSwift(
            gaussianCount: UInt32(gaussianCount),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            maxAssignments: UInt32(tileAssignmentHeader.length / MemoryLayout<UInt32>.stride) // placeholder, overridden below
        )
        // Use header's maxAssignments instead of buffer length heuristic
        let headerPtr = tileAssignmentHeader.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        params.maxAssignments = headerPtr.pointee.maxAssignments

        // 1) Clear counts + header
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "TileBinClear"
            encoder.setComputePipelineState(clearPipeline)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 0)
            encoder.setBuffer(tileAssignmentHeader, offset: 0, index: 1)
            encoder.setBytes(&tileCountU, length: MemoryLayout<UInt32>.stride, index: 2)
            let tgWidth = clearPipeline.threadExecutionWidth
            let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
            let threads = MTLSize(width: tileCount, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // 2) Count gaussians per tile (precise intersection)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "TileBinCount"
            encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 1)
            encoder.setBuffer(opacitiesBuffer, offset: 0, index: 2)
            encoder.setBytes(&params, length: MemoryLayout<TileBinningParamsSwift>.stride, index: 3)
            encoder.setBuffer(meansBuffer, offset: 0, index: 4)
            encoder.setBuffer(conicsBuffer, offset: 0, index: 5)

            let pipeline: MTLComputePipelineState
            if precision == .float16, let halfPipe = self.countPipelineHalf {
                pipeline = halfPipe
            } else {
                pipeline = self.countPipeline
            }
            encoder.setComputePipelineState(pipeline)

            let tgWidth = pipeline.threadExecutionWidth
            let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
            let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // 3) Prefix scan counts -> offsets
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "TileBinPrefix"
            encoder.setComputePipelineState(prefixPipeline)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 0)
            encoder.setBuffer(offsetsBuffer, offset: 0, index: 1)
            encoder.setBytes(&tileCountU, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBuffer(partialSumsBuffer, offset: 0, index: 3)
            let tg = MTLSize(width: prefixBlockSize, height: 1, depth: 1)
            let tgs = MTLSize(width: actualGroups, height: 1, depth: 1)
            encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "TileBinPartial"
            encoder.setComputePipelineState(partialPipeline)
            var numPartial = UInt32(actualGroups)
            encoder.setBuffer(partialSumsBuffer, offset: 0, index: 0)
            encoder.setBytes(&numPartial, length: MemoryLayout<UInt32>.stride, index: 1)
            let tg = MTLSize(width: prefixBlockSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "TileBinFinalize"
            encoder.setComputePipelineState(finalizePipeline)
            encoder.setBuffer(offsetsBuffer, offset: 0, index: 0)
            encoder.setBytes(&tileCountU, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.setBuffer(partialSumsBuffer, offset: 0, index: 2)
            let tg = MTLSize(width: prefixBlockSize, height: 1, depth: 1)
            let tgs = MTLSize(width: actualGroups, height: 1, depth: 1)
            encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "TileBinStoreTotal"
            encoder.setComputePipelineState(storeTotalPipeline)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 0)
            encoder.setBuffer(offsetsBuffer, offset: 0, index: 1)
            encoder.setBytes(&tileCountU, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBuffer(tileAssignmentHeader, offset: 0, index: 3)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // 4) Reset counters for scatter (reuse coverageBuffer)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "TileBinZero"
            encoder.setComputePipelineState(zeroPipeline)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 0)
            encoder.setBytes(&tileCountU, length: MemoryLayout<UInt32>.stride, index: 1)
            let tgWidth = zeroPipeline.threadExecutionWidth
            let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
            let threads = MTLSize(width: tileCount, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // 5) Scatter into per-tile blocks
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "TileBinScatter"
            encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
            encoder.setBuffer(coverageBuffer, offset: 0, index: 1) // counters
            encoder.setBuffer(offsetsBuffer, offset: 0, index: 2)
            encoder.setBuffer(tileIndicesBuffer, offset: 0, index: 3)
            encoder.setBuffer(tileIdsBuffer, offset: 0, index: 4)
            encoder.setBuffer(tileAssignmentHeader, offset: 0, index: 5)
            encoder.setBytes(&params, length: MemoryLayout<TileBinningParamsSwift>.stride, index: 6)
            encoder.setBuffer(meansBuffer, offset: 0, index: 7)
            encoder.setBuffer(conicsBuffer, offset: 0, index: 8)
            encoder.setBuffer(opacitiesBuffer, offset: 0, index: 9)

            let pipeline: MTLComputePipelineState
            if precision == .float16, let halfPipe = self.scatterPipelineHalf {
                pipeline = halfPipe
            } else {
                pipeline = self.scatterPipeline
            }
            encoder.setComputePipelineState(pipeline)

            let tgWidth = pipeline.threadExecutionWidth
            let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
            let threads = MTLSize(width: gaussianCount, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }
}

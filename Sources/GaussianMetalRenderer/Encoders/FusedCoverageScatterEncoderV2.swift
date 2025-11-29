import Metal

/// Optimized encoder for fused coverage + scatter kernel V2
/// Key optimizations over V1:
/// - Reduced threadgroup memory: 128 tiles (512B) vs 2048 (8KB) = ~8x occupancy
/// - SIMD ballot/prefix for tile counting - no threadgroup atomics in hot loop
/// - Warp-cooperative global allocation - one atomic per SIMD group
/// - True half precision math for half variant
final class FusedCoverageScatterEncoderV2 {
    private let fusedFloatPipeline: MTLComputePipelineState
    private let fusedHalfPipeline: MTLComputePipelineState?

    // Threadgroup size matches FUSED_V2_TG_SIZE in Metal
    let threadgroupSize: Int = 64

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let fusedFloatFn = library.makeFunction(name: "fusedCoverageScatterKernelV2_float") else {
            fatalError("fusedCoverageScatterKernelV2_float not found in library")
        }
        self.fusedFloatPipeline = try device.makeComputePipelineState(function: fusedFloatFn)

        if let fusedHalfFn = library.makeFunction(name: "fusedCoverageScatterKernelV2_half") {
            self.fusedHalfPipeline = try? device.makeComputePipelineState(function: fusedHalfFn)
        } else {
            self.fusedHalfPipeline = nil
        }
    }

    /// Encode fused coverage + scatter V2 in single pass
    /// Uses SIMD optimizations for better occupancy and throughput
    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        tileWidth: Int,
        tileHeight: Int,
        tilesX: Int,
        maxAssignments: Int,
        boundsBuffer: MTLBuffer,
        coverageBuffer: MTLBuffer,
        opacitiesBuffer: MTLBuffer,
        meansBuffer: MTLBuffer,
        conicsBuffer: MTLBuffer,
        tileIndicesBuffer: MTLBuffer,
        tileIdsBuffer: MTLBuffer,
        tileAssignmentHeader: MTLBuffer,
        precision: Precision = .float32
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "FusedCoverageScatterV2"

        let pipeline: MTLComputePipelineState
        if precision == .float16, let halfPipe = fusedHalfPipeline {
            pipeline = halfPipe
        } else {
            pipeline = fusedFloatPipeline
        }
        encoder.setComputePipelineState(pipeline)

        // Buffer bindings match Metal kernel
        encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
        encoder.setBuffer(coverageBuffer, offset: 0, index: 1)
        encoder.setBuffer(opacitiesBuffer, offset: 0, index: 2)

        // Reuse existing params struct - same layout
        var params = FusedCoverageScatterParamsSwift(
            gaussianCount: UInt32(gaussianCount),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            maxAssignments: UInt32(maxAssignments)
        )
        encoder.setBytes(&params, length: MemoryLayout<FusedCoverageScatterParamsSwift>.stride, index: 3)

        encoder.setBuffer(meansBuffer, offset: 0, index: 4)
        encoder.setBuffer(conicsBuffer, offset: 0, index: 5)
        encoder.setBuffer(tileIndicesBuffer, offset: 0, index: 6)
        encoder.setBuffer(tileIdsBuffer, offset: 0, index: 7)
        encoder.setBuffer(tileAssignmentHeader, offset: 0, index: 8)

        // One threadgroup per gaussian
        let tgCount = MTLSize(width: gaussianCount, height: 1, depth: 1)
        let threadsPerTG = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreadgroups(tgCount, threadsPerThreadgroup: threadsPerTG)

        encoder.endEncoding()
    }

    /// Get pipeline state for external use (e.g., occupancy queries)
    func getPipeline(precision: Precision) -> MTLComputePipelineState {
        if precision == .float16, let halfPipe = fusedHalfPipeline {
            return halfPipe
        }
        return fusedFloatPipeline
    }
}

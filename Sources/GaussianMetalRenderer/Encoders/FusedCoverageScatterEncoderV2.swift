import Metal

/// Optimized encoder for fused coverage + scatter kernel V2
/// Key optimizations over V1:
/// - Reduced threadgroup memory: 128 tiles (512B) vs 2048 (8KB) = ~8x occupancy
/// - SIMD ballot/prefix for tile counting - no threadgroup atomics in hot loop
/// - Warp-cooperative global allocation - one atomic per SIMD group
/// - Reads from packed GaussianRenderData struct
final class FusedCoverageScatterEncoderV2 {
    private let pipeline: MTLComputePipelineState

    // Threadgroup size matches FUSED_V2_TG_SIZE in Metal
    let threadgroupSize: Int = 64

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let fn = library.makeFunction(name: "fusedCoverageScatterKernelV2") else {
            fatalError("fusedCoverageScatterKernelV2 not found in library")
        }
        self.pipeline = try device.makeComputePipelineState(function: fn)
    }

    /// Encode fused coverage + scatter V2 in single pass
    /// Reads mean/conic/opacity from packed GaussianRenderData
    func encode(
        commandBuffer: MTLCommandBuffer,
        gaussianCount: Int,
        tileWidth: Int,
        tileHeight: Int,
        tilesX: Int,
        maxAssignments: Int,
        boundsBuffer: MTLBuffer,
        coverageBuffer: MTLBuffer,
        renderData: MTLBuffer,
        tileIndicesBuffer: MTLBuffer,
        tileIdsBuffer: MTLBuffer,
        tileAssignmentHeader: MTLBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "FusedCoverageScatterV2"

        encoder.setComputePipelineState(pipeline)

        // Buffer bindings match Metal kernel
        encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
        encoder.setBuffer(coverageBuffer, offset: 0, index: 1)
        encoder.setBuffer(renderData, offset: 0, index: 2)

        var params = FusedCoverageScatterParamsSwift(
            gaussianCount: UInt32(gaussianCount),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            maxAssignments: UInt32(maxAssignments)
        )
        encoder.setBytes(&params, length: MemoryLayout<FusedCoverageScatterParamsSwift>.stride, index: 3)

        encoder.setBuffer(tileIndicesBuffer, offset: 0, index: 4)
        encoder.setBuffer(tileIdsBuffer, offset: 0, index: 5)
        encoder.setBuffer(tileAssignmentHeader, offset: 0, index: 6)

        // One threadgroup per gaussian
        let tgCount = MTLSize(width: gaussianCount, height: 1, depth: 1)
        let threadsPerTG = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreadgroups(tgCount, threadsPerThreadgroup: threadsPerTG)

        encoder.endEncoding()
    }

    /// Get pipeline state for external use (e.g., occupancy queries)
    func getPipeline() -> MTLComputePipelineState {
        return pipeline
    }
}

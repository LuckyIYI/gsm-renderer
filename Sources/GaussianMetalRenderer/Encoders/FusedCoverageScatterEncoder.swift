import Metal

/// Encoder for fused coverage + scatter kernel
/// Eliminates prefix scan by using atomic allocation
final class FusedCoverageScatterEncoder {
    private let fusedFloatPipeline: MTLComputePipelineState
    private let fusedHalfPipeline: MTLComputePipelineState?

    let threadgroupSize: Int = 32  // Single SIMD, matches FUSED_TG_SIZE in Metal

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let fusedFloatFn = library.makeFunction(name: "fusedCoverageScatterKernel_float") else {
            fatalError("fusedCoverageScatterKernel_float not found in library")
        }
        self.fusedFloatPipeline = try device.makeComputePipelineState(function: fusedFloatFn)

        if let fusedHalfFn = library.makeFunction(name: "fusedCoverageScatterKernel_half") {
            self.fusedHalfPipeline = try? device.makeComputePipelineState(function: fusedHalfFn)
        } else {
            self.fusedHalfPipeline = nil
        }
    }

    /// Encode fused coverage + scatter in single pass
    /// Replaces: Coverage (5 passes) + Scatter (2 passes) = 7 passes → 1 pass
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
        encoder.label = "FusedCoverageScatter"

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

        // One threadgroup per gaussian, 32 threads (single SIMD)
        let threadgroups = MTLSize(width: gaussianCount, height: 1, depth: 1)
        let threadsPerTG = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)

        encoder.endEncoding()
    }
}

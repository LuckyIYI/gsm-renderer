import Metal

/// Encodes scatter stage: assigns gaussians to tiles (sparse mode - no compaction)
/// Visibility encoded in zero tile bounds: minTile == maxTile == 0 means invisible
/// Uses ellipse intersection for precise tile filtering (benchmarked as optimal)
public final class LocalScatterEncoder {
    private let scatterSparsePipeline: MTLComputePipelineState

    /// Total gaussian count for sparse dispatch
    public var totalGaussianCount: Int = 0

    public init(library: MTLLibrary, device: MTLDevice) throws {
        guard let scatterSparseFn = library.makeFunction(name: "localScatterSimd16Sparse") else {
            fatalError("Missing localScatterSimd16Sparse kernel")
        }
        self.scatterSparsePipeline = try device.makeComputePipelineState(function: scatterSparseFn)
    }

    /// Encode scatter with 16-bit depth keys (sparse mode)
    /// Dispatches for ALL gaussians, skips invisible via zero tile bounds
    public func encode16(
        commandBuffer: MTLCommandBuffer,
        projectedGaussians: MTLBuffer,  // Sparse projection buffer
        compactedHeader: MTLBuffer,     // Unused in sparse mode
        tileCounters: MTLBuffer,
        depthKeys16: MTLBuffer,
        globalIndices: MTLBuffer,
        tilesX: Int,
        maxPerTile: Int,
        tileWidth: Int,
        tileHeight: Int
    ) {
        guard totalGaussianCount > 0 else { return }

        var tilesXU = UInt32(tilesX)
        var maxPerTileU = UInt32(maxPerTile)
        var tileW = Int32(tileWidth)
        var tileH = Int32(tileHeight)
        var gaussianCountU = UInt32(totalGaussianCount)

        // Direct dispatch for all gaussians (no indirect needed)
        let gaussiansPerTG = 256 * 32 // 8192 gaussians per threadgroup
        let threadgroups = (totalGaussianCount + gaussiansPerTG - 1) / gaussiansPerTG

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_Scatter16_sparse"
            encoder.setComputePipelineState(scatterSparsePipeline)
            encoder.setBuffer(projectedGaussians, offset: 0, index: 0)
            // index 1 unused (was visibilityMask)
            encoder.setBuffer(tileCounters, offset: 0, index: 2)
            encoder.setBuffer(depthKeys16, offset: 0, index: 3)
            encoder.setBuffer(globalIndices, offset: 0, index: 4)
            encoder.setBytes(&tilesXU, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&maxPerTileU, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&tileW, length: MemoryLayout<Int32>.stride, index: 7)
            encoder.setBytes(&tileH, length: MemoryLayout<Int32>.stride, index: 8)
            encoder.setBytes(&gaussianCountU, length: MemoryLayout<UInt32>.stride, index: 9)

            let tg = MTLSize(width: 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }
}

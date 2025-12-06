import Metal

/// Encodes per-tile bitonic sort stage (16-bit only)
public final class LocalSortEncoder {
    private let sort16Pipeline: MTLComputePipelineState

    public init(library: MTLLibrary, device: MTLDevice) throws {
        guard let sort16Fn = library.makeFunction(name: "localPerTileSort16") else {
            fatalError("Missing localPerTileSort16 kernel")
        }
        self.sort16Pipeline = try device.makeComputePipelineState(function: sort16Fn)
    }

    /// Encode 16-bit per-tile sort (fixed layout: tileId * maxPerTile)
    public func encode16(
        commandBuffer: MTLCommandBuffer,
        depthKeys16: MTLBuffer,
        globalIndices: MTLBuffer,
        sortedLocalIdx: MTLBuffer,
        tileCounts: MTLBuffer,
        maxPerTile: Int,
        tileCount: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var maxPerTileU = UInt32(maxPerTile)

        encoder.label = "Local_PerTileSort16"
        encoder.setComputePipelineState(sort16Pipeline)
        encoder.setBuffer(depthKeys16, offset: 0, index: 0)
        encoder.setBuffer(globalIndices, offset: 0, index: 1)
        encoder.setBuffer(sortedLocalIdx, offset: 0, index: 2)
        encoder.setBuffer(tileCounts, offset: 0, index: 3)
        encoder.setBytes(&maxPerTileU, length: 4, index: 4)

        // One threadgroup per tile
        let tgs = MTLSize(width: tileCount, height: 1, depth: 1)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

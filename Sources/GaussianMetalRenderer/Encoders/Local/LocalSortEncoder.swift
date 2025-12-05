import Metal

/// Encodes per-tile radix sort stage
public final class LocalSortEncoder {
    private let perTileSortPipeline: MTLComputePipelineState
    private let sort16Pipeline: MTLComputePipelineState?

    public init(library: MTLLibrary, device: MTLDevice) throws {
        guard let sortFn = library.makeFunction(name: "localPerTileSort") else {
            fatalError("Missing localPerTileSort kernel")
        }
        self.perTileSortPipeline = try device.makeComputePipelineState(function: sortFn)

        // Optional 16-bit sort
        if let sort16Fn = library.makeFunction(name: "localPerTileSort16") {
            self.sort16Pipeline = try? device.makeComputePipelineState(function: sort16Fn)
        } else {
            self.sort16Pipeline = nil
        }
    }

    /// Check if 16-bit sort is available
    public var has16BitSort: Bool { self.sort16Pipeline != nil }

    /// Encode 32-bit per-tile bitonic sort (fixed layout: tileId * maxPerTile)
    public func encode(
        commandBuffer: MTLCommandBuffer,
        sortKeys: MTLBuffer,
        sortIndices: MTLBuffer,
        tileCounts: MTLBuffer,
        maxPerTile: Int,
        tileCount: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var maxPerTileU = UInt32(maxPerTile)

        encoder.label = "Local_PerTileSort"
        encoder.setComputePipelineState(self.perTileSortPipeline)
        encoder.setBuffer(sortKeys, offset: 0, index: 0)
        encoder.setBuffer(sortIndices, offset: 0, index: 1)
        encoder.setBuffer(tileCounts, offset: 0, index: 2)
        encoder.setBytes(&maxPerTileU, length: 4, index: 3)

        // One threadgroup per tile
        let tgs = MTLSize(width: tileCount, height: 1, depth: 1)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
        encoder.endEncoding()
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
        guard let sortPipeline = sort16Pipeline else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var maxPerTileU = UInt32(maxPerTile)

        encoder.label = "Local_PerTileSort16"
        encoder.setComputePipelineState(sortPipeline)
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

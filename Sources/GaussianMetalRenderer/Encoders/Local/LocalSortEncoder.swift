import Metal

/// Encodes per-tile radix sort stage
public final class LocalSortEncoder {
    private let perTileSortPipeline: MTLComputePipelineState
    private let sort16Pipeline: MTLComputePipelineState?

    public init(library: MTLLibrary, device: MTLDevice) throws {
        guard let sortFn = library.makeFunction(name: "LocalPerTileSort") else {
            fatalError("Missing per-tile sort kernel")
        }
        self.perTileSortPipeline = try device.makeComputePipelineState(function: sortFn)

        // Optional 16-bit sort
        if let sort16Fn = library.makeFunction(name: "LocalPerTileSort16") {
            self.sort16Pipeline = try? device.makeComputePipelineState(function: sort16Fn)
        } else {
            self.sort16Pipeline = nil
        }
    }

    /// Check if 16-bit sort is available
    public var has16BitSort: Bool { sort16Pipeline != nil }

    /// Encode 32-bit per-tile radix sort
    public func encode(
        commandBuffer: MTLCommandBuffer,
        sortKeys: MTLBuffer,
        sortIndices: MTLBuffer,
        tileOffsets: MTLBuffer,
        tileCounts: MTLBuffer,
        tempSortKeys: MTLBuffer,
        tempSortIndices: MTLBuffer,
        tileCount: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Local_PerTileSort"
        encoder.setComputePipelineState(perTileSortPipeline)
        encoder.setBuffer(sortKeys, offset: 0, index: 0)
        encoder.setBuffer(sortIndices, offset: 0, index: 1)
        encoder.setBuffer(tileOffsets, offset: 0, index: 2)
        encoder.setBuffer(tileCounts, offset: 0, index: 3)
        encoder.setBuffer(tempSortKeys, offset: 0, index: 4)
        encoder.setBuffer(tempSortIndices, offset: 0, index: 5)

        // One threadgroup per tile
        let tgs = MTLSize(width: tileCount, height: 1, depth: 1)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }

    /// Encode 16-bit per-tile sort (experimental - 50% less threadgroup memory)
    public func encode16(
        commandBuffer: MTLCommandBuffer,
        depthKeys16: MTLBuffer,
        globalIndices: MTLBuffer,
        sortedLocalIdx: MTLBuffer,
        tileOffsets: MTLBuffer,
        tileCounts: MTLBuffer,
        tileCount: Int
    ) {
        guard let sortPipeline = sort16Pipeline else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.label = "Local_PerTileSort16"
        encoder.setComputePipelineState(sortPipeline)
        encoder.setBuffer(depthKeys16, offset: 0, index: 0)
        encoder.setBuffer(globalIndices, offset: 0, index: 1)
        encoder.setBuffer(sortedLocalIdx, offset: 0, index: 2)
        encoder.setBuffer(tileOffsets, offset: 0, index: 3)
        encoder.setBuffer(tileCounts, offset: 0, index: 4)

        // One threadgroup per tile
        let tgs = MTLSize(width: tileCount, height: 1, depth: 1)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

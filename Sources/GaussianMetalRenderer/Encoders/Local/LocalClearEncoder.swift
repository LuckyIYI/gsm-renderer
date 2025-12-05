import Metal

/// Encodes the clear stage - zeros tile counts and initializes header
public final class LocalClearEncoder {
    private let clearPipeline: MTLComputePipelineState

    public init(library: MTLLibrary, device: MTLDevice) throws {
        guard let clearFn = library.makeFunction(name: "localClear") else {
            fatalError("Missing localClear kernel")
        }
        self.clearPipeline = try device.makeComputePipelineState(function: clearFn)
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        tileCounts: MTLBuffer,
        header: MTLBuffer,
        tileCount: Int,
        maxCompacted: Int
    ) {
        var tileCountU = UInt32(tileCount)
        var maxCompactedU = UInt32(maxCompacted)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Local_Clear"
        encoder.setComputePipelineState(self.clearPipeline)
        encoder.setBuffer(tileCounts, offset: 0, index: 0)
        encoder.setBuffer(header, offset: 0, index: 1)
        encoder.setBytes(&tileCountU, length: 4, index: 2)
        encoder.setBytes(&maxCompactedU, length: 4, index: 3)
        let threads = MTLSize(width: max(tileCount, 1), height: 1, depth: 1)
        let tg = MTLSize(width: clearPipeline.threadExecutionWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}

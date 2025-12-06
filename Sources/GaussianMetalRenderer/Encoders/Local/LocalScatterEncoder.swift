import Metal

/// Scatter kernel variant for benchmarking
public enum ScatterVariant: Int, CaseIterable {
    case baseline = 0 // Original 4 gaussians/TG (SIMD-cooperative)
    case optimized = 1 // 32 gaussians per thread (3.9x faster)
    case sparse = 2 // No compaction, dispatch for all gaussians, skip invisible
}

/// Encodes scatter stage: assigns gaussians to tiles using SIMD-cooperative approach
public final class LocalScatterEncoder {
    private let prepareDispatchPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState
    private let scatter16Pipeline: MTLComputePipelineState?
    private let scatter16OptimizedPipeline: MTLComputePipelineState?
    private let scatter16SparsePipeline: MTLComputePipelineState?

    /// Currently active scatter variant
    /// Default: .optimized (32 gaussians/thread) - provides 3.9x speedup over baseline
    public var activeVariant: ScatterVariant = .optimized

    /// For sparse variant: visibility mask and gaussian count (set before encode16Sparse)
    public var visibilityMask: MTLBuffer?
    public var totalGaussianCount: Int = 0

    public init(library: MTLLibrary, device: MTLDevice) throws {
        guard let prepareDispatchFn = library.makeFunction(name: "localPrepareScatterDispatch"),
              let scatterFn = library.makeFunction(name: "localScatterSimd")
        else {
            fatalError("Missing scatter kernels")
        }
        self.prepareDispatchPipeline = try device.makeComputePipelineState(function: prepareDispatchFn)
        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)

        // 16-bit baseline scatter
        if let scatter16Fn = library.makeFunction(name: "localScatterSimd16") {
            self.scatter16Pipeline = try? device.makeComputePipelineState(function: scatter16Fn)
        } else {
            self.scatter16Pipeline = nil
        }

        // Optimized scatter: 32 gaussians per thread (3.9x speedup)
        if let scatter16OptFn = library.makeFunction(name: "localScatterSimd16V7") {
            self.scatter16OptimizedPipeline = try? device.makeComputePipelineState(function: scatter16OptFn)
        } else {
            self.scatter16OptimizedPipeline = nil
        }

        // Sparse scatter: no compaction, dispatch for all gaussians
        if let scatter16SparseFn = library.makeFunction(name: "localScatterSimd16Sparse") {
            self.scatter16SparsePipeline = try? device.makeComputePipelineState(function: scatter16SparseFn)
        } else {
            self.scatter16SparsePipeline = nil
        }
    }

    /// Check if 16-bit scatter is available
    public var has16BitScatter: Bool { self.scatter16Pipeline != nil }

    /// Check which variants are available
    public var availableVariants: [ScatterVariant] {
        var variants: [ScatterVariant] = []
        if self.scatter16Pipeline != nil { variants.append(.baseline) }
        if self.scatter16OptimizedPipeline != nil { variants.append(.optimized) }
        return variants
    }

    /// Encode scatter with 32-bit keys (fixed layout: tileId * maxPerTile)
    public func encode(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        tileCounters: MTLBuffer,
        sortKeys: MTLBuffer,
        sortIndices: MTLBuffer,
        tilesX: Int,
        maxPerTile: Int,
        tileWidth: Int,
        tileHeight: Int
    ) {
        var tilesXU = UInt32(tilesX)
        var maxPerTileU = UInt32(maxPerTile)
        var tileW = Int32(tileWidth)
        var tileH = Int32(tileHeight)

        // Create dispatch args buffer (12 bytes for MTLDispatchThreadgroupsIndirectArguments)
        let dispatchArgsBuffer = commandBuffer.device.makeBuffer(length: 12, options: .storageModeShared)!

        // Prepare indirect dispatch
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_PrepareScatterDispatch"
            encoder.setComputePipelineState(self.prepareDispatchPipeline)
            encoder.setBuffer(compactedHeader, offset: 0, index: 0)
            encoder.setBuffer(dispatchArgsBuffer, offset: 0, index: 1)
            var tgWidth = UInt32(4) // 4 gaussians per threadgroup (4 SIMD groups)
            encoder.setBytes(&tgWidth, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // SIMD-cooperative scatter (fixed layout: tileId * maxPerTile + localIdx)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_Scatter_SIMD"
            encoder.setComputePipelineState(self.scatterPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBuffer(tileCounters, offset: 0, index: 2)
            encoder.setBuffer(sortKeys, offset: 0, index: 3)
            encoder.setBuffer(sortIndices, offset: 0, index: 4)
            encoder.setBytes(&tilesXU, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&maxPerTileU, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&tileW, length: MemoryLayout<Int32>.stride, index: 7)
            encoder.setBytes(&tileH, length: MemoryLayout<Int32>.stride, index: 8)

            // 128 threads per threadgroup (4 SIMD groups = 4 gaussians)
            let tg = MTLSize(width: 128, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: dispatchArgsBuffer,
                                         indirectBufferOffset: 0,
                                         threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Encode scatter with 16-bit depth keys (fixed layout: tileId * maxPerTile)
    /// Uses activeVariant to select kernel
    public func encode16(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        tileCounters: MTLBuffer,
        depthKeys16: MTLBuffer,
        globalIndices: MTLBuffer,
        tilesX: Int,
        maxPerTile: Int,
        tileWidth: Int,
        tileHeight: Int
    ) {
        // Sparse variant uses different kernel with different buffer layout
        if self.activeVariant == .sparse {
            self.encode16Sparse(
                commandBuffer: commandBuffer,
                projectedGaussians: compactedGaussians, // Actually tempProjectionBuffer in sparse mode
                tileCounters: tileCounters,
                depthKeys16: depthKeys16,
                globalIndices: globalIndices,
                tilesX: tilesX,
                maxPerTile: maxPerTile,
                tileWidth: tileWidth,
                tileHeight: tileHeight
            )
            return
        }

        // Select pipeline based on active variant
        let scatter16: MTLComputePipelineState
        let threadgroupSize: Int
        let gaussiansPerTG: Int

        switch self.activeVariant {
        case .baseline:
            guard let pipeline = scatter16Pipeline else { return }
            scatter16 = pipeline
            threadgroupSize = 128 // 4 SIMD groups
            gaussiansPerTG = 4
        case .optimized:
            guard let pipeline = scatter16OptimizedPipeline else {
                // Fallback to baseline
                guard let fallback = scatter16Pipeline else { return }
                scatter16 = fallback
                threadgroupSize = 128
                gaussiansPerTG = 4
                break
            }
            scatter16 = pipeline
            threadgroupSize = 256 // 256 threads per TG
            gaussiansPerTG = 256 * 32 // Each thread processes 32 gaussians (8192 per TG)
        case .sparse:
            return // Handled above
        }

        var tilesXU = UInt32(tilesX)
        var maxPerTileU = UInt32(maxPerTile)
        var tileW = Int32(tileWidth)
        var tileH = Int32(tileHeight)

        // Create dispatch args buffer
        let dispatchArgsBuffer = commandBuffer.device.makeBuffer(length: 12, options: .storageModeShared)!

        // Prepare indirect dispatch
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_PrepareScatter16Dispatch"
            encoder.setComputePipelineState(self.prepareDispatchPipeline)
            encoder.setBuffer(compactedHeader, offset: 0, index: 0)
            encoder.setBuffer(dispatchArgsBuffer, offset: 0, index: 1)
            var tgWidth = UInt32(gaussiansPerTG)
            encoder.setBytes(&tgWidth, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // 16-bit scatter (fixed layout: tileId * maxPerTile + localIdx)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Local_Scatter16_\(self.activeVariant)"
            encoder.setComputePipelineState(scatter16)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBuffer(tileCounters, offset: 0, index: 2)
            encoder.setBuffer(depthKeys16, offset: 0, index: 3)
            encoder.setBuffer(globalIndices, offset: 0, index: 4)
            encoder.setBytes(&tilesXU, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&maxPerTileU, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&tileW, length: MemoryLayout<Int32>.stride, index: 7)
            encoder.setBytes(&tileH, length: MemoryLayout<Int32>.stride, index: 8)

            let tg = MTLSize(width: threadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(indirectBuffer: dispatchArgsBuffer,
                                         indirectBufferOffset: 0,
                                         threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Encode sparse scatter - dispatches for ALL gaussians, skips invisible via visibility mask
    /// No compaction needed - reads from sparse projectedGaussians buffer
    private func encode16Sparse(
        commandBuffer: MTLCommandBuffer,
        projectedGaussians: MTLBuffer,
        tileCounters: MTLBuffer,
        depthKeys16: MTLBuffer,
        globalIndices: MTLBuffer,
        tilesX: Int,
        maxPerTile: Int,
        tileWidth: Int,
        tileHeight: Int
    ) {
        guard let sparsePipeline = scatter16SparsePipeline,
              let visMask = visibilityMask,
              totalGaussianCount > 0
        else { return }

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
            encoder.setComputePipelineState(sparsePipeline)
            encoder.setBuffer(projectedGaussians, offset: 0, index: 0)
            encoder.setBuffer(visMask, offset: 0, index: 1)
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

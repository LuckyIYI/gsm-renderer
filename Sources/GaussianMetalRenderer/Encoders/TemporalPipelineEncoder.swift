import Metal

/// Standalone Temporal Pipeline Encoder
/// Archive of temporal coherent sorting experiments
/// Uses TemporalShaders.metallib for all temporal kernels
public final class TemporalPipelineEncoder {
    private let device: MTLDevice
    private let library: MTLLibrary

    // Core temporal pipelines
    private let clearTemporalPipeline: MTLComputePipelineState
    private let initSortedPipeline: MTLComputePipelineState
    private let scatterCandidatesDepthPipeline: MTLComputePipelineState
    private let scatterCandidatesHonestPipeline: MTLComputePipelineState
    private let recomputeSortedScoresPipeline: MTLComputePipelineState
    private let recomputeSortedDepthsPipeline: MTLComputePipelineState
    private let resortAfterRecomputePipeline: MTLComputePipelineState
    private let insertionSortPipeline: MTLComputePipelineState
    private let spatialReusePipeline: MTLComputePipelineState
    private let renderTemporalPipeline: MTLComputePipelineState
    private let renderStochasticPipeline: MTLComputePipelineState
    private let randomSampleTilesPipeline: MTLComputePipelineState

    /// Default max gaussians per tile for temporal mode
    public static let temporalMaxPerTile: UInt32 = 512

    /// Max candidates per tile per frame
    public static let maxCandidatesPerTile: UInt32 = 64

    /// Stochastic mode tile dimensions
    public static let stochasticTileWidth: Int = 8
    public static let stochasticTileHeight: Int = 8

    /// Standard tile dimensions
    public static let tileWidth: Int = 32
    public static let tileHeight: Int = 16

    public init(device: MTLDevice) throws {
        self.device = device

        // Load Temporal shader library
        guard let libraryURL = Bundle.module.url(forResource: "TemporalShaders", withExtension: "metallib"),
              let lib = try? device.makeLibrary(URL: libraryURL) else {
            fatalError("Failed to load TemporalShaders.metallib")
        }
        self.library = lib

        // Load all kernel functions
        guard let clearFn = lib.makeFunction(name: "temporal_clear"),
              let initFn = lib.makeFunction(name: "temporal_init_sorted"),
              let scatterDepthFn = lib.makeFunction(name: "temporal_scatter_candidates_depth"),
              let scatterHonestFn = lib.makeFunction(name: "temporal_scatter_candidates_honest"),
              let recomputeScoresFn = lib.makeFunction(name: "temporal_recompute_sorted_scores"),
              let recomputeDepthsFn = lib.makeFunction(name: "temporal_recompute_sorted_depths"),
              let resortFn = lib.makeFunction(name: "temporal_resort_after_recompute"),
              let insertionFn = lib.makeFunction(name: "temporal_insertion_sort"),
              let spatialFn = lib.makeFunction(name: "temporal_spatial_reuse"),
              let renderFn = lib.makeFunction(name: "temporal_render"),
              let stochasticFn = lib.makeFunction(name: "temporal_render_stochastic"),
              let randomSampleFn = lib.makeFunction(name: "temporal_random_sample_tiles") else {
            fatalError("Missing required kernel functions in TemporalShaders")
        }

        self.clearTemporalPipeline = try device.makeComputePipelineState(function: clearFn)
        self.initSortedPipeline = try device.makeComputePipelineState(function: initFn)
        self.scatterCandidatesDepthPipeline = try device.makeComputePipelineState(function: scatterDepthFn)
        self.scatterCandidatesHonestPipeline = try device.makeComputePipelineState(function: scatterHonestFn)
        self.recomputeSortedScoresPipeline = try device.makeComputePipelineState(function: recomputeScoresFn)
        self.recomputeSortedDepthsPipeline = try device.makeComputePipelineState(function: recomputeDepthsFn)
        self.resortAfterRecomputePipeline = try device.makeComputePipelineState(function: resortFn)
        self.insertionSortPipeline = try device.makeComputePipelineState(function: insertionFn)
        self.spatialReusePipeline = try device.makeComputePipelineState(function: spatialFn)
        self.renderTemporalPipeline = try device.makeComputePipelineState(function: renderFn)
        self.renderStochasticPipeline = try device.makeComputePipelineState(function: stochasticFn)
        self.randomSampleTilesPipeline = try device.makeComputePipelineState(function: randomSampleFn)
    }

    // MARK: - Buffer Initialization

    /// Clear candidate buffers, worldToCompacted mapping, AND header for temporal mode
    public func encodeClearTemporal(
        commandBuffer: MTLCommandBuffer,
        candidateCounts: MTLBuffer,
        worldToCompacted: MTLBuffer,
        compactedHeader: MTLBuffer,
        candidateIndices: MTLBuffer,
        tileCount: Int,
        maxGaussians: Int,
        maxCandidates: UInt32
    ) {
        // Clear header first
        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            blitEncoder.label = "Temporal_ClearHeader"
            blitEncoder.fill(buffer: compactedHeader, range: 0..<16, value: 0)
            blitEncoder.endEncoding()
        }

        var tileCountU = UInt32(tileCount)
        var maxGaussiansU = UInt32(maxGaussians)
        var maxCandU = maxCandidates
        let totalCandidateSlots = tileCount * Int(maxCandidates)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Temporal_Clear"
            encoder.setComputePipelineState(clearTemporalPipeline)
            encoder.setBuffer(candidateCounts, offset: 0, index: 0)
            encoder.setBuffer(worldToCompacted, offset: 0, index: 1)
            encoder.setBuffer(candidateIndices, offset: 0, index: 2)
            encoder.setBytes(&tileCountU, length: 4, index: 3)
            encoder.setBytes(&maxGaussiansU, length: 4, index: 4)
            encoder.setBytes(&maxCandU, length: 4, index: 5)

            let threads = MTLSize(width: max(maxGaussians, totalCandidateSlots), height: 1, depth: 1)
            let tg = MTLSize(width: clearTemporalPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Initialize sorted buffers (call once at start or when scene changes)
    public func encodeInitSorted(
        commandBuffer: MTLCommandBuffer,
        sortedKeys: MTLBuffer,
        sortedIndices: MTLBuffer,
        sortedCounts: MTLBuffer,
        tileCount: Int,
        maxPerTile: UInt32
    ) {
        var tileCountU = UInt32(tileCount)
        var maxPerTileU = maxPerTile
        let totalSlots = tileCount * Int(maxPerTile)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Temporal_InitSorted"
            encoder.setComputePipelineState(initSortedPipeline)
            encoder.setBuffer(sortedKeys, offset: 0, index: 0)
            encoder.setBuffer(sortedIndices, offset: 0, index: 1)
            encoder.setBuffer(sortedCounts, offset: 0, index: 2)
            encoder.setBytes(&tileCountU, length: 4, index: 3)
            encoder.setBytes(&maxPerTileU, length: 4, index: 4)

            let threads = MTLSize(width: max(totalSlots, tileCount), height: 1, depth: 1)
            let tg = MTLSize(width: initSortedPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    // MARK: - Scatter Candidates

    /// Scatter candidates with DEPTH keys using stochastic hash-based slot selection
    public func encodeScatterCandidatesDepth(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        compactedToWorld: MTLBuffer,
        candidateKeys: MTLBuffer,
        candidateIndices: MTLBuffer,
        tilesX: Int,
        maxCandidates: UInt32,
        tileWidth: Int,
        tileHeight: Int,
        maxGaussians: Int,
        frameId: UInt32
    ) {
        var tilesXU = UInt32(tilesX)
        var maxCandU = maxCandidates
        var frameIdU = frameId
        var tileW = Int32(tileWidth)
        var tileH = Int32(tileHeight)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Temporal_ScatterCandidatesDepth"
            encoder.setComputePipelineState(scatterCandidatesDepthPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBuffer(compactedToWorld, offset: 0, index: 2)
            encoder.setBuffer(candidateKeys, offset: 0, index: 3)
            encoder.setBuffer(candidateIndices, offset: 0, index: 4)
            encoder.setBytes(&tilesXU, length: 4, index: 5)
            encoder.setBytes(&maxCandU, length: 4, index: 6)
            encoder.setBytes(&frameIdU, length: 4, index: 7)
            encoder.setBytes(&tileW, length: 4, index: 8)
            encoder.setBytes(&tileH, length: 4, index: 9)

            let threads = MTLSize(width: maxGaussians, height: 1, depth: 1)
            let tg = MTLSize(width: scatterCandidatesDepthPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Scatter candidates with atomic counters (honest, no hash collisions)
    public func encodeScatterCandidatesHonest(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        compactedToWorld: MTLBuffer,
        candidateCounts: MTLBuffer,
        candidateKeys: MTLBuffer,
        candidateIndices: MTLBuffer,
        tilesX: Int,
        maxCandidates: UInt32,
        tileWidth: Int,
        tileHeight: Int,
        maxGaussians: Int
    ) {
        var tilesXU = UInt32(tilesX)
        var maxCandU = maxCandidates
        var tileW = Int32(tileWidth)
        var tileH = Int32(tileHeight)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Temporal_ScatterCandidatesHonest"
            encoder.setComputePipelineState(scatterCandidatesHonestPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBuffer(compactedToWorld, offset: 0, index: 2)
            encoder.setBuffer(candidateCounts, offset: 0, index: 3)
            encoder.setBuffer(candidateKeys, offset: 0, index: 4)
            encoder.setBuffer(candidateIndices, offset: 0, index: 5)
            encoder.setBytes(&tilesXU, length: 4, index: 6)
            encoder.setBytes(&maxCandU, length: 4, index: 7)
            encoder.setBytes(&tileW, length: 4, index: 8)
            encoder.setBytes(&tileH, length: 4, index: 9)

            let threads = MTLSize(width: maxGaussians, height: 1, depth: 1)
            let tg = MTLSize(width: scatterCandidatesHonestPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    // MARK: - Recompute Keys (for camera movement)

    /// Recompute SCORE keys for sorted gaussians
    public func encodeRecomputeSortedScores(
        commandBuffer: MTLCommandBuffer,
        sortedKeys: MTLBuffer,
        sortedIndices: MTLBuffer,
        sortedCounts: MTLBuffer,
        compactedGaussians: MTLBuffer,
        worldToCompacted: MTLBuffer,
        maxPerTile: UInt32,
        tileCount: Int
    ) {
        var maxPerTileU = maxPerTile

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Temporal_RecomputeSortedScores"
            encoder.setComputePipelineState(recomputeSortedScoresPipeline)
            encoder.setBuffer(sortedKeys, offset: 0, index: 0)
            encoder.setBuffer(sortedIndices, offset: 0, index: 1)
            encoder.setBuffer(sortedCounts, offset: 0, index: 2)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 3)
            encoder.setBuffer(worldToCompacted, offset: 0, index: 4)
            encoder.setBytes(&maxPerTileU, length: 4, index: 5)

            let threads = MTLSize(width: tileCount, height: 1, depth: 1)
            let tg = MTLSize(width: recomputeSortedScoresPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Recompute DEPTH keys and check tile intersection for sorted gaussians
    public func encodeRecomputeSortedDepths(
        commandBuffer: MTLCommandBuffer,
        sortedKeys: MTLBuffer,
        sortedIndices: MTLBuffer,
        sortedCounts: MTLBuffer,
        compactedGaussians: MTLBuffer,
        worldToCompacted: MTLBuffer,
        maxPerTile: UInt32,
        tileCount: Int,
        tilesX: Int
    ) {
        var maxPerTileU = maxPerTile
        var tileCountU = UInt32(tileCount)
        var tilesXU = UInt32(tilesX)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Temporal_RecomputeSortedDepths"
            encoder.setComputePipelineState(recomputeSortedDepthsPipeline)
            encoder.setBuffer(sortedKeys, offset: 0, index: 0)
            encoder.setBuffer(sortedIndices, offset: 0, index: 1)
            encoder.setBuffer(sortedCounts, offset: 0, index: 2)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 3)
            encoder.setBuffer(worldToCompacted, offset: 0, index: 4)
            encoder.setBytes(&maxPerTileU, length: 4, index: 5)
            encoder.setBytes(&tileCountU, length: 4, index: 6)
            encoder.setBytes(&tilesXU, length: 4, index: 7)

            let totalThreads = tileCount * Int(maxPerTile)
            let threads = MTLSize(width: totalThreads, height: 1, depth: 1)
            let tg = MTLSize(width: recomputeSortedDepthsPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Resort sorted list after recompute, rebuild bloom filter
    public func encodeResortAfterRecompute(
        commandBuffer: MTLCommandBuffer,
        sortedKeys: MTLBuffer,
        sortedIndices: MTLBuffer,
        sortedCounts: MTLBuffer,
        bloomFilters: MTLBuffer,
        maxPerTile: UInt32,
        tileCount: Int
    ) {
        var maxPerTileU = maxPerTile

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Temporal_ResortAfterRecompute"
            encoder.setComputePipelineState(resortAfterRecomputePipeline)
            encoder.setBuffer(sortedKeys, offset: 0, index: 0)
            encoder.setBuffer(sortedIndices, offset: 0, index: 1)
            encoder.setBuffer(sortedCounts, offset: 0, index: 2)
            encoder.setBytes(&maxPerTileU, length: 4, index: 3)
            encoder.setBuffer(bloomFilters, offset: 0, index: 4)

            let threads = MTLSize(width: tileCount, height: 1, depth: 1)
            let tg = MTLSize(width: min(resortAfterRecomputePipeline.threadExecutionWidth, tileCount), height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    // MARK: - Insertion Sort

    /// Insertion sort of candidates into sorted list with bloom filter
    public func encodeInsertionSort(
        commandBuffer: MTLCommandBuffer,
        sortedKeys: MTLBuffer,
        sortedIndices: MTLBuffer,
        sortedCounts: MTLBuffer,
        bloomFilters: MTLBuffer,
        candidateIndices: MTLBuffer,
        candidateCounts: MTLBuffer,
        compactedGaussians: MTLBuffer,
        worldToCompacted: MTLBuffer,
        maxPerTile: UInt32,
        maxCandidates: UInt32,
        tileCount: Int
    ) {
        var maxPerTileU = maxPerTile
        var maxCandU = maxCandidates

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Temporal_InsertionSort"
            encoder.setComputePipelineState(insertionSortPipeline)
            encoder.setBuffer(sortedKeys, offset: 0, index: 0)
            encoder.setBuffer(sortedIndices, offset: 0, index: 1)
            encoder.setBuffer(sortedCounts, offset: 0, index: 2)
            encoder.setBuffer(bloomFilters, offset: 0, index: 3)
            encoder.setBuffer(candidateIndices, offset: 0, index: 4)
            encoder.setBuffer(candidateCounts, offset: 0, index: 5)
            encoder.setBytes(&maxPerTileU, length: 4, index: 6)
            encoder.setBytes(&maxCandU, length: 4, index: 7)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 8)
            encoder.setBuffer(worldToCompacted, offset: 0, index: 9)

            let threads = MTLSize(width: tileCount, height: 1, depth: 1)
            let tg = MTLSize(width: insertionSortPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    // MARK: - Spatial Reuse

    /// Steal top candidates from neighboring tiles
    public func encodeSpatialReuse(
        commandBuffer: MTLCommandBuffer,
        sortedKeys: MTLBuffer,
        sortedIndices: MTLBuffer,
        sortedCounts: MTLBuffer,
        compactedGaussians: MTLBuffer,
        worldToCompacted: MTLBuffer,
        maxPerTile: UInt32,
        tilesX: Int,
        tilesY: Int,
        numToSteal: UInt32 = 8
    ) {
        var maxPerTileU = maxPerTile
        var tilesXU = UInt32(tilesX)
        var tilesYU = UInt32(tilesY)
        var numToStealU = numToSteal
        let tileCount = tilesX * tilesY

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Temporal_SpatialReuse"
            encoder.setComputePipelineState(spatialReusePipeline)
            encoder.setBuffer(sortedKeys, offset: 0, index: 0)
            encoder.setBuffer(sortedIndices, offset: 0, index: 1)
            encoder.setBuffer(sortedCounts, offset: 0, index: 2)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 3)
            encoder.setBuffer(worldToCompacted, offset: 0, index: 4)
            encoder.setBytes(&maxPerTileU, length: 4, index: 5)
            encoder.setBytes(&tilesXU, length: 4, index: 6)
            encoder.setBytes(&tilesYU, length: 4, index: 7)
            encoder.setBytes(&numToStealU, length: 4, index: 8)

            let threads = MTLSize(width: tileCount, height: 1, depth: 1)
            let tg = MTLSize(width: spatialReusePipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    // MARK: - Render

    /// Render using temporal sorted list
    public func encodeRender(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        sortedCounts: MTLBuffer,
        sortedWorldIndices: MTLBuffer,
        worldToCompacted: MTLBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        tilesX: Int,
        tilesY: Int,
        width: Int,
        height: Int,
        tileWidth: Int,
        tileHeight: Int,
        maxPerTile: UInt32,
        whiteBackground: Bool = false
    ) {
        var params = TemporalRenderParams(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            whiteBackground: whiteBackground ? 1 : 0,
            _pad: 0
        )
        var maxPerTileU = maxPerTile

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Temporal_Render"
            encoder.setComputePipelineState(renderTemporalPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(sortedCounts, offset: 0, index: 1)
            encoder.setBuffer(sortedWorldIndices, offset: 0, index: 2)
            encoder.setBuffer(worldToCompacted, offset: 0, index: 3)
            encoder.setTexture(colorTexture, index: 0)
            encoder.setTexture(depthTexture, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<TemporalRenderParams>.stride, index: 4)
            encoder.setBytes(&maxPerTileU, length: 4, index: 5)

            let tgs = MTLSize(width: tilesX, height: tilesY, depth: 1)
            let tg = MTLSize(width: 8, height: 8, depth: 1)
            encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    /// Stochastic render with temporal accumulation
    public func encodeRenderStochastic(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        sortedCounts: MTLBuffer,
        sortedWorldIndices: MTLBuffer,
        worldToCompacted: MTLBuffer,
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        tilesX: Int,
        tilesY: Int,
        width: Int,
        height: Int,
        maxPerTile: UInt32,
        frameIndex: UInt32,
        whiteBackground: Bool = false
    ) {
        var params = TemporalRenderParams(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(Self.stochasticTileWidth),
            tileHeight: UInt32(Self.stochasticTileHeight),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            whiteBackground: whiteBackground ? 1 : 0,
            _pad: 0
        )
        var maxPerTileU = maxPerTile
        var frameIdx = frameIndex

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Temporal_RenderStochastic"
            encoder.setComputePipelineState(renderStochasticPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(sortedCounts, offset: 0, index: 1)
            encoder.setBuffer(sortedWorldIndices, offset: 0, index: 2)
            encoder.setBuffer(worldToCompacted, offset: 0, index: 3)
            encoder.setTexture(colorTexture, index: 0)
            encoder.setTexture(depthTexture, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<TemporalRenderParams>.stride, index: 4)
            encoder.setBytes(&maxPerTileU, length: 4, index: 5)
            encoder.setBytes(&frameIdx, length: 4, index: 6)

            let tgs = MTLSize(width: tilesX, height: tilesY, depth: 1)
            let tg = MTLSize(width: 4, height: 4, depth: 1)
            encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    // MARK: - Random Sampling

    /// Random sample M gaussians per tile
    public func encodeRandomSampleTiles(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        compactedToWorld: MTLBuffer,
        candidateKeys: MTLBuffer,
        candidateIndices: MTLBuffer,
        samplesPerTile: UInt32,
        frameIndex: UInt32,
        tilesX: Int,
        tilesY: Int,
        tileWidth: Int,
        tileHeight: Int
    ) {
        var samples = samplesPerTile
        var frame = frameIndex
        var tilesXU = UInt32(tilesX)
        var tileW = UInt32(tileWidth)
        var tileH = UInt32(tileHeight)
        let tileCount = tilesX * tilesY

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Temporal_RandomSampleTiles"
            encoder.setComputePipelineState(randomSampleTilesPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBuffer(compactedToWorld, offset: 0, index: 2)
            encoder.setBuffer(candidateKeys, offset: 0, index: 3)
            encoder.setBuffer(candidateIndices, offset: 0, index: 4)
            encoder.setBytes(&samples, length: 4, index: 5)
            encoder.setBytes(&frame, length: 4, index: 6)
            encoder.setBytes(&tilesXU, length: 4, index: 7)
            encoder.setBytes(&tileW, length: 4, index: 8)
            encoder.setBytes(&tileH, length: 4, index: 9)

            let tgs = MTLSize(width: tileCount, height: 1, depth: 1)
            let tg = MTLSize(width: 64, height: 1, depth: 1)
            encoder.dispatchThreads(tgs, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }
}

// MARK: - Supporting Types

public struct TemporalRenderParams {
    public var width: UInt32
    public var height: UInt32
    public var tileWidth: UInt32
    public var tileHeight: UInt32
    public var tilesX: UInt32
    public var tilesY: UInt32
    public var whiteBackground: UInt32
    public var _pad: UInt32

    public init(width: UInt32, height: UInt32, tileWidth: UInt32, tileHeight: UInt32,
                tilesX: UInt32, tilesY: UInt32, whiteBackground: UInt32, _pad: UInt32) {
        self.width = width
        self.height = height
        self.tileWidth = tileWidth
        self.tileHeight = tileHeight
        self.tilesX = tilesX
        self.tilesY = tilesY
        self.whiteBackground = whiteBackground
        self._pad = _pad
    }
}

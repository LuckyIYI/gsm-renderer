import Metal

/// Depth-First Sort Encoder (FastGS/Splatshop style two-phase sort)
///
/// Flow:
/// 1. Project + Compact visible gaussians → N_visible
/// 2. Generate depth keys (32-bit) → depthKeys[N_visible]
/// 3. Radix sort by depth (4 passes) → depth-sorted indices
/// 4. Apply ordering + compute tile counts → gaussianTileCounts[N_visible]
/// 5. Prefix sum → gaussianOffsets[N_visible]
/// 6. Create instances → instanceTileKeys[N_instances], instanceGaussianIdx[N_instances]
/// 7. Counting sort by tile (1 pass) → tile-sorted instances
/// 8. Extract tile ranges → tileRanges[N_tiles]
/// 9. Render (no per-tile sort needed - already depth-ordered!)
///
/// Benefits over current approach:
/// - Depth sort on N_visible (much smaller than N_instances)
/// - 32-bit keys = 4 radix passes (vs 8 for 64-bit fused keys)
/// - 16-bit tile sort = 1 counting sort pass
/// - No per-tile bitonic sort needed!
public final class DepthFirstSortEncoder {
    private let device: MTLDevice
    private let library: MTLLibrary

    // Radix sort pipelines (32-bit depth sort)
    private let histogramPipeline: MTLComputePipelineState
    private let scanHistogramPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState

    // Counting sort pipelines (16-bit tile sort)
    private let countTilesPipeline: MTLComputePipelineState
    private let scatterTilesPipeline: MTLComputePipelineState
    private let zeroCountersPipeline: MTLComputePipelineState

    // Depth-first specific pipelines
    private let genKeysPipeline: MTLComputePipelineState
    private let applyOrderPipeline: MTLComputePipelineState
    private let createInstancesPipeline: MTLComputePipelineState
    private let extractRangesPipeline: MTLComputePipelineState
    private let renderPipeline: MTLComputePipelineState

    // Prefix scan pipelines (reuse from local sort)
    private let prefixScanPipeline: MTLComputePipelineState
    private let scanPartialSumsPipeline: MTLComputePipelineState
    private let finalizeScanPipeline: MTLComputePipelineState

    // Constants
    private let radixBlockSize = 256
    private let radixGrainSize = 4
    private let prefixBlockSize = 256
    private let prefixGrainSize = 4

    public init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        self.library = library

        // Load radix sort kernels
        guard let histogramFn = library.makeFunction(name: "depthFirstHistogram"),
              let scanHistFn = library.makeFunction(name: "depthFirstScanHistogram"),
              let scatterFn = library.makeFunction(name: "depthFirstScatter") else {
            throw RendererError.failedToCreatePipeline("Missing depth-first radix sort kernels")
        }
        self.histogramPipeline = try device.makeComputePipelineState(function: histogramFn)
        self.scanHistogramPipeline = try device.makeComputePipelineState(function: scanHistFn)
        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)

        // Load counting sort kernels
        guard let countTilesFn = library.makeFunction(name: "depthFirstCountTiles"),
              let scatterTilesFn = library.makeFunction(name: "depthFirstScatterTiles"),
              let zeroCountersFn = library.makeFunction(name: "depthFirstZeroTileCounters") else {
            throw RendererError.failedToCreatePipeline("Missing depth-first counting sort kernels")
        }
        self.countTilesPipeline = try device.makeComputePipelineState(function: countTilesFn)
        self.scatterTilesPipeline = try device.makeComputePipelineState(function: scatterTilesFn)
        self.zeroCountersPipeline = try device.makeComputePipelineState(function: zeroCountersFn)

        // Load depth-first specific kernels
        guard let genKeysFn = library.makeFunction(name: "depthFirstGenKeys"),
              let applyOrderFn = library.makeFunction(name: "depthFirstApplyOrder"),
              let createInstFn = library.makeFunction(name: "depthFirstCreateInstances"),
              let extractRangesFn = library.makeFunction(name: "depthFirstExtractRanges"),
              let renderFn = library.makeFunction(name: "depthFirstRender") else {
            throw RendererError.failedToCreatePipeline("Missing depth-first pipeline kernels")
        }
        self.genKeysPipeline = try device.makeComputePipelineState(function: genKeysFn)
        self.applyOrderPipeline = try device.makeComputePipelineState(function: applyOrderFn)
        self.createInstancesPipeline = try device.makeComputePipelineState(function: createInstFn)
        self.extractRangesPipeline = try device.makeComputePipelineState(function: extractRangesFn)
        self.renderPipeline = try device.makeComputePipelineState(function: renderFn)

        // Load prefix scan kernels
        guard let prefixFn = library.makeFunction(name: "localSortPrefixScan"),
              let partialFn = library.makeFunction(name: "localSortScanPartialSums"),
              let finalizeFn = library.makeFunction(name: "localSortFinalizeScan") else {
            throw RendererError.failedToCreatePipeline("Missing prefix scan kernels")
        }
        self.prefixScanPipeline = try device.makeComputePipelineState(function: prefixFn)
        self.scanPartialSumsPipeline = try device.makeComputePipelineState(function: partialFn)
        self.finalizeScanPipeline = try device.makeComputePipelineState(function: finalizeFn)
    }

    // MARK: - 32-bit Radix Sort (Depth Keys)

    /// Encode a single radix sort pass (histogram + scan + scatter)
    public func encodeRadixPass(
        commandBuffer: MTLCommandBuffer,
        inKeys: MTLBuffer,
        outKeys: MTLBuffer,
        inVals: MTLBuffer,
        outVals: MTLBuffer,
        histogram: MTLBuffer,
        count: Int,
        digit: Int  // Which byte to sort (0-3)
    ) {
        let elementsPerBlock = radixBlockSize * radixGrainSize
        let numBlocks = (count + elementsPerBlock - 1) / elementsPerBlock
        let histogramSize = 256 * numBlocks

        var countU = UInt32(count)
        var digitU = UInt32(digit)

        // Step 1: Histogram
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_Histogram_\(digit)"
            encoder.setComputePipelineState(histogramPipeline)
            encoder.setBuffer(inKeys, offset: 0, index: 0)
            encoder.setBuffer(histogram, offset: 0, index: 1)
            encoder.setBytes(&countU, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&digitU, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.dispatchThreadgroups(MTLSize(width: numBlocks, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: radixBlockSize, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // Step 2: Exclusive scan of histogram
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_ScanHistogram_\(digit)"
            encoder.setComputePipelineState(scanHistogramPipeline)
            encoder.setBuffer(histogram, offset: 0, index: 0)
            var histSizeU = UInt32(histogramSize)
            encoder.setBytes(&histSizeU, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: radixBlockSize, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // Step 3: Scatter
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_Scatter_\(digit)"
            encoder.setComputePipelineState(scatterPipeline)
            encoder.setBuffer(inKeys, offset: 0, index: 0)
            encoder.setBuffer(outKeys, offset: 0, index: 1)
            encoder.setBuffer(inVals, offset: 0, index: 2)
            encoder.setBuffer(outVals, offset: 0, index: 3)
            encoder.setBuffer(histogram, offset: 0, index: 4)
            encoder.setBytes(&countU, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&digitU, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.dispatchThreadgroups(MTLSize(width: numBlocks, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: radixBlockSize, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    /// Encode full 32-bit radix sort (4 passes)
    /// After sorting, output is in outKeys/outVals (for even number of passes)
    public func encodeDepthSort(
        commandBuffer: MTLCommandBuffer,
        keys: MTLBuffer,       // Input/output: 32-bit depth keys
        values: MTLBuffer,     // Input/output: gaussian indices
        tempKeys: MTLBuffer,   // Temp buffer for ping-pong
        tempVals: MTLBuffer,   // Temp buffer for ping-pong
        histogram: MTLBuffer,  // Size: 256 * numBlocks
        count: Int
    ) {
        // 4 passes for 32-bit keys (8-bit radix)
        // Pass 0: keys → tempKeys, vals → tempVals
        // Pass 1: tempKeys → keys, tempVals → vals
        // Pass 2: keys → tempKeys, vals → tempVals
        // Pass 3: tempKeys → keys, tempVals → vals

        for pass in 0..<4 {
            let isEven = (pass % 2 == 0)
            encodeRadixPass(
                commandBuffer: commandBuffer,
                inKeys: isEven ? keys : tempKeys,
                outKeys: isEven ? tempKeys : keys,
                inVals: isEven ? values : tempVals,
                outVals: isEven ? tempVals : values,
                histogram: histogram,
                count: count,
                digit: pass
            )
        }
        // After 4 passes, result is back in keys/values
    }

    // MARK: - 16-bit Counting Sort (Tile Keys)

    /// Encode counting sort for tile keys
    public func encodeTileSort(
        commandBuffer: MTLCommandBuffer,
        inTileKeys: MTLBuffer,      // Input: ushort tile keys
        inGaussianIdx: MTLBuffer,   // Input: gaussian indices
        outTileKeys: MTLBuffer,     // Output: sorted tile keys
        outGaussianIdx: MTLBuffer,  // Output: sorted gaussian indices
        tileCounts: MTLBuffer,      // Temp: count per tile (uint)
        tileOffsets: MTLBuffer,     // Temp: exclusive prefix sum
        tileCounters: MTLBuffer,    // Temp: atomic counters for scatter
        partialSums: MTLBuffer,     // Temp: for prefix scan
        nInstances: Int,
        nTiles: Int
    ) {
        var nInstU = UInt32(nInstances)
        var nTilesU = UInt32(nTiles)
        let elementsPerGroup = prefixBlockSize * prefixGrainSize
        let numGroups = max(1, (nTiles + elementsPerGroup - 1) / elementsPerGroup)

        // Step 1: Zero tile counts
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_ZeroTileCounts"
            encoder.setComputePipelineState(zeroCountersPipeline)
            encoder.setBuffer(tileCounts, offset: 0, index: 0)
            encoder.setBytes(&nTilesU, length: MemoryLayout<UInt32>.stride, index: 1)
            let tg = MTLSize(width: zeroCountersPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(MTLSize(width: nTiles, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Step 2: Count tile occurrences
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_CountTiles"
            encoder.setComputePipelineState(countTilesPipeline)
            encoder.setBuffer(inTileKeys, offset: 0, index: 0)
            encoder.setBuffer(tileCounts, offset: 0, index: 1)
            encoder.setBytes(&nInstU, length: MemoryLayout<UInt32>.stride, index: 2)
            let tg = MTLSize(width: countTilesPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(MTLSize(width: nInstances, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Step 3: Prefix scan tile counts → offsets
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_PrefixScan"
            encoder.setComputePipelineState(prefixScanPipeline)
            encoder.setBuffer(tileCounts, offset: 0, index: 0)
            encoder.setBuffer(tileOffsets, offset: 0, index: 1)
            encoder.setBytes(&nTilesU, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBuffer(partialSums, offset: 0, index: 3)
            encoder.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_ScanPartials"
            encoder.setComputePipelineState(scanPartialSumsPipeline)
            encoder.setBuffer(partialSums, offset: 0, index: 0)
            var numPartialsU = UInt32(numGroups)
            encoder.setBytes(&numPartialsU, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_FinalizeScan"
            encoder.setComputePipelineState(finalizeScanPipeline)
            encoder.setBuffer(tileOffsets, offset: 0, index: 0)
            encoder.setBytes(&nTilesU, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.setBuffer(partialSums, offset: 0, index: 2)
            encoder.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // Step 4: Zero tile counters for scatter
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_ZeroCounters"
            encoder.setComputePipelineState(zeroCountersPipeline)
            encoder.setBuffer(tileCounters, offset: 0, index: 0)
            encoder.setBytes(&nTilesU, length: MemoryLayout<UInt32>.stride, index: 1)
            let tg = MTLSize(width: zeroCountersPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(MTLSize(width: nTiles, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Step 5: Scatter instances to sorted positions
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_ScatterTiles"
            encoder.setComputePipelineState(scatterTilesPipeline)
            encoder.setBuffer(inTileKeys, offset: 0, index: 0)
            encoder.setBuffer(inGaussianIdx, offset: 0, index: 1)
            encoder.setBuffer(outTileKeys, offset: 0, index: 2)
            encoder.setBuffer(outGaussianIdx, offset: 0, index: 3)
            encoder.setBuffer(tileOffsets, offset: 0, index: 4)
            encoder.setBuffer(tileCounters, offset: 0, index: 5)
            encoder.setBytes(&nInstU, length: MemoryLayout<UInt32>.stride, index: 6)
            let tg = MTLSize(width: scatterTilesPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(MTLSize(width: nInstances, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    // MARK: - Full Pipeline

    /// Encode the complete depth-first sorting pipeline
    /// Assumes project+compact has already been done by LocalSortPipelineEncoder
    public func encodeSort(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        compactedHeader: MTLBuffer,
        // Depth sort buffers
        depthKeys: MTLBuffer,           // uint[maxGaussians]
        depthSortedIndices: MTLBuffer,  // uint[maxGaussians]
        tempKeys: MTLBuffer,            // uint[maxGaussians]
        tempVals: MTLBuffer,            // uint[maxGaussians]
        radixHistogram: MTLBuffer,      // uint[256 * numBlocks]
        // Tile counts
        gaussianTileCounts: MTLBuffer,  // uint[maxGaussians]
        gaussianOffsets: MTLBuffer,     // uint[maxGaussians]
        gaussianPartialSums: MTLBuffer, // uint[numGroups]
        // Instance buffers
        instanceTileKeys: MTLBuffer,    // ushort[maxInstances]
        instanceGaussianIdx: MTLBuffer, // uint[maxInstances]
        sortedTileKeys: MTLBuffer,      // ushort[maxInstances]
        sortedGaussianIdx: MTLBuffer,   // uint[maxInstances]
        // Tile sort buffers
        tileCounts: MTLBuffer,          // uint[nTiles]
        tileOffsets: MTLBuffer,         // uint[nTiles]
        tileCounters: MTLBuffer,        // uint[nTiles]
        tilePartialSums: MTLBuffer,     // uint[numGroups]
        // Tile ranges output
        tileRanges: MTLBuffer,          // uint2[nTiles]
        // Parameters
        visibleCount: Int,
        maxInstances: Int,
        tilesX: Int,
        tilesY: Int,
        tileWidth: Int,
        tileHeight: Int
    ) {
        let nTiles = tilesX * tilesY
        let elementsPerGroup = prefixBlockSize * prefixGrainSize
        let gaussianNumGroups = max(1, (visibleCount + elementsPerGroup - 1) / elementsPerGroup)

        var visibleCountU = UInt32(visibleCount)
        var tilesXU = UInt32(tilesX)
        var tileW = Int32(tileWidth)
        var tileH = Int32(tileHeight)
        var nTilesU = UInt32(nTiles)

        // === Step 1: Generate depth keys ===
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_GenKeys"
            encoder.setComputePipelineState(genKeysPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(compactedHeader, offset: 0, index: 1)
            encoder.setBuffer(depthKeys, offset: 0, index: 2)
            encoder.setBuffer(depthSortedIndices, offset: 0, index: 3)
            let tg = MTLSize(width: genKeysPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(MTLSize(width: visibleCount, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === Step 2: Radix sort by depth (4 passes) ===
        encodeDepthSort(
            commandBuffer: commandBuffer,
            keys: depthKeys,
            values: depthSortedIndices,
            tempKeys: tempKeys,
            tempVals: tempVals,
            histogram: radixHistogram,
            count: visibleCount
        )

        // === Step 3: Apply depth ordering → get tile counts ===
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_ApplyOrder"
            encoder.setComputePipelineState(applyOrderPipeline)
            encoder.setBuffer(depthSortedIndices, offset: 0, index: 0)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 1)
            encoder.setBuffer(gaussianTileCounts, offset: 0, index: 2)
            encoder.setBytes(&visibleCountU, length: MemoryLayout<UInt32>.stride, index: 3)
            let tg = MTLSize(width: applyOrderPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(MTLSize(width: visibleCount, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // === Step 4: Prefix sum gaussian tile counts ===
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_PrefixScanGaussian"
            encoder.setComputePipelineState(prefixScanPipeline)
            encoder.setBuffer(gaussianTileCounts, offset: 0, index: 0)
            encoder.setBuffer(gaussianOffsets, offset: 0, index: 1)
            encoder.setBytes(&visibleCountU, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBuffer(gaussianPartialSums, offset: 0, index: 3)
            encoder.dispatchThreadgroups(MTLSize(width: gaussianNumGroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_ScanPartialsGaussian"
            encoder.setComputePipelineState(scanPartialSumsPipeline)
            encoder.setBuffer(gaussianPartialSums, offset: 0, index: 0)
            var numPartialsU = UInt32(gaussianNumGroups)
            encoder.setBytes(&numPartialsU, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_FinalizeScanGaussian"
            encoder.setComputePipelineState(finalizeScanPipeline)
            encoder.setBuffer(gaussianOffsets, offset: 0, index: 0)
            encoder.setBytes(&visibleCountU, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.setBuffer(gaussianPartialSums, offset: 0, index: 2)
            encoder.dispatchThreadgroups(MTLSize(width: gaussianNumGroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: prefixBlockSize, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // === Step 5: Create instances (NO ATOMICS - uses prefix sum!) ===
        // Note: Need to get total instance count from last element of prefix sum
        // For now, use maxInstances as upper bound
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_CreateInstances"
            encoder.setComputePipelineState(createInstancesPipeline)
            encoder.setBuffer(depthSortedIndices, offset: 0, index: 0)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 1)
            encoder.setBuffer(gaussianOffsets, offset: 0, index: 2)
            encoder.setBuffer(instanceTileKeys, offset: 0, index: 3)
            encoder.setBuffer(instanceGaussianIdx, offset: 0, index: 4)
            encoder.setBytes(&tilesXU, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&tileW, length: MemoryLayout<Int32>.stride, index: 6)
            encoder.setBytes(&tileH, length: MemoryLayout<Int32>.stride, index: 7)
            encoder.setBytes(&visibleCountU, length: MemoryLayout<UInt32>.stride, index: 8)
            let tg = MTLSize(width: createInstancesPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(MTLSize(width: visibleCount, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }

        // Get total instance count (would need CPU readback or indirect dispatch)
        // For now, use maxInstances as the count for tile sort
        let estimatedInstances = maxInstances

        // === Step 6: Counting sort by tile key ===
        encodeTileSort(
            commandBuffer: commandBuffer,
            inTileKeys: instanceTileKeys,
            inGaussianIdx: instanceGaussianIdx,
            outTileKeys: sortedTileKeys,
            outGaussianIdx: sortedGaussianIdx,
            tileCounts: tileCounts,
            tileOffsets: tileOffsets,
            tileCounters: tileCounters,
            partialSums: tilePartialSums,
            nInstances: estimatedInstances,
            nTiles: nTiles
        )

        // === Step 7: Zero tile ranges and extract ===
        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            blitEncoder.label = "DepthFirst_ZeroRanges"
            blitEncoder.fill(buffer: tileRanges, range: 0..<(nTiles * MemoryLayout<SIMD2<UInt32>>.stride), value: 0)
            blitEncoder.endEncoding()
        }

        var nInstU = UInt32(estimatedInstances)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_ExtractRanges"
            encoder.setComputePipelineState(extractRangesPipeline)
            encoder.setBuffer(sortedTileKeys, offset: 0, index: 0)
            encoder.setBuffer(tileRanges, offset: 0, index: 1)
            encoder.setBytes(&nInstU, length: MemoryLayout<UInt32>.stride, index: 2)
            let tg = MTLSize(width: extractRangesPipeline.threadExecutionWidth, height: 1, depth: 1)
            encoder.dispatchThreads(MTLSize(width: estimatedInstances, height: 1, depth: 1), threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    // MARK: - Render

    /// Encode depth-first render (no per-tile sort needed!)
    public func encodeRender(
        commandBuffer: MTLCommandBuffer,
        compactedGaussians: MTLBuffer,
        tileRanges: MTLBuffer,
        instanceGaussianIdx: MTLBuffer,
        colorTexture: MTLTexture,
        tilesX: Int,
        tilesY: Int,
        width: Int,
        height: Int,
        tileWidth: Int,
        tileHeight: Int,
        whiteBackground: Bool = false
    ) {
        var params = RenderParams(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            maxPerTile: 0,
            whiteBackground: whiteBackground ? 1 : 0,
            activeTileCount: 0,
            gaussianCount: 0
        )

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "DepthFirst_Render"
            encoder.setComputePipelineState(renderPipeline)
            encoder.setBuffer(compactedGaussians, offset: 0, index: 0)
            encoder.setBuffer(tileRanges, offset: 0, index: 1)
            encoder.setBuffer(instanceGaussianIdx, offset: 0, index: 2)
            encoder.setTexture(colorTexture, index: 0)
            encoder.setBytes(&params, length: MemoryLayout<RenderParams>.stride, index: 3)

            let tgs = MTLSize(width: tilesX, height: tilesY, depth: 1)
            let tg = MTLSize(width: 8, height: 8, depth: 1)
            encoder.dispatchThreadgroups(tgs, threadsPerThreadgroup: tg)
            encoder.endEncoding()
        }
    }

    // MARK: - Buffer Size Calculations

    /// Calculate required histogram buffer size
    public func histogramBufferSize(maxGaussians: Int) -> Int {
        let elementsPerBlock = radixBlockSize * radixGrainSize
        let numBlocks = (maxGaussians + elementsPerBlock - 1) / elementsPerBlock
        return 256 * numBlocks * MemoryLayout<UInt32>.stride
    }

    /// Calculate required partial sums buffer size
    public func partialSumsBufferSize(maxElements: Int) -> Int {
        let elementsPerGroup = prefixBlockSize * prefixGrainSize
        let numGroups = (maxElements + elementsPerGroup - 1) / elementsPerGroup
        return numGroups * MemoryLayout<UInt32>.stride
    }
}

/// RenderParams struct matching Metal (40 bytes)
fileprivate struct RenderParams {
    var width: UInt32
    var height: UInt32
    var tileWidth: UInt32
    var tileHeight: UInt32
    var tilesX: UInt32
    var tilesY: UInt32
    var maxPerTile: UInt32
    var whiteBackground: UInt32
    var activeTileCount: UInt32
    var gaussianCount: UInt32
}

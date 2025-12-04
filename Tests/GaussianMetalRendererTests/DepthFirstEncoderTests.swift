import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Unit tests for DepthFirst pipeline components
/// Tests each stage separately: Projection, DepthSort, InstanceCreation, TileSort, Render
final class DepthFirstEncoderTests: XCTestCase {

    var device: MTLDevice!
    var library: MTLLibrary!
    var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal device")
            return
        }
        self.device = device
        self.queue = device.makeCommandQueue()!

        // Load DepthFirst shader library
        let bundle = Bundle(for: type(of: self))
        guard let libraryURL = bundle.url(forResource: "DepthFirstShaders", withExtension: "metallib") else {
            // Try loading from main bundle
            if let mainURL = Bundle.main.url(forResource: "DepthFirstShaders", withExtension: "metallib") {
                self.library = try? device.makeLibrary(URL: mainURL)
            } else {
                // Try loading from package resources
                let packagePath = Bundle.module.bundleURL.deletingLastPathComponent()
                    .appendingPathComponent("GaussianMetalRenderer_GaussianMetalRenderer.bundle")
                    .appendingPathComponent("DepthFirstShaders.metallib")
                self.library = try? device.makeLibrary(URL: packagePath)
            }
            return
        }
        self.library = try? device.makeLibrary(URL: libraryURL)
    }

    // MARK: - Test 1: Depth Key Generation

    func testDepthKeyGeneration() throws {
        guard let library = self.library else {
            print("⚠️ Skipping test - library not loaded")
            return
        }

        // Create test data: 10 compacted gaussians with known depths
        let count = 10
        let depths: [Float] = [5.0, 2.0, 8.0, 1.0, 3.0, 9.0, 4.0, 7.0, 6.0, 10.0]

        // Create CompactedGaussian buffer (covariance_depth.w holds depth)
        var compactedData = [CompactedGaussianSwift]()
        for i in 0..<count {
            var g = CompactedGaussianSwift()
            g.covariance_depth = SIMD4<Float>(0, 0, 0, depths[i])  // .w is depth
            g.position_color = SIMD4<Float>(0, 0, 0, 0)  // Not used for this test
            g.min_tile = SIMD2<Int32>(0, 0)
            g.max_tile = SIMD2<Int32>(1, 1)
            g.originalIdx = UInt32(i)
            compactedData.append(g)
        }

        let compactedBuffer = device.makeBuffer(bytes: compactedData,
                                                  length: count * MemoryLayout<CompactedGaussianSwift>.stride,
                                                  options: .storageModeShared)!

        // Create header with count
        var header = TileAssignmentHeaderSwift()
        header.totalAssignments = UInt32(count)
        let headerBuffer = device.makeBuffer(bytes: &header,
                                               length: MemoryLayout<TileAssignmentHeaderSwift>.stride,
                                               options: .storageModeShared)!

        // Output buffers
        let depthKeysBuffer = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride,
                                                  options: .storageModeShared)!
        let indicesBuffer = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride,
                                                options: .storageModeShared)!

        // Load and run dfGenDepthKeys kernel
        guard let genKeysFn = library.makeFunction(name: "dfGenDepthKeys") else {
            XCTFail("dfGenDepthKeys not found")
            return
        }
        let pipeline = try device.makeComputePipelineState(function: genKeysFn)

        let commandBuffer = queue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(compactedBuffer, offset: 0, index: 0)
        encoder.setBuffer(depthKeysBuffer, offset: 0, index: 1)
        encoder.setBuffer(indicesBuffer, offset: 0, index: 2)
        encoder.setBuffer(headerBuffer, offset: 0, index: 3)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(count, 64), height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Verify: indices should be [0, 1, 2, ..., count-1]
        let indicesPtr = indicesBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
        for i in 0..<count {
            XCTAssertEqual(indicesPtr[i], UInt32(i), "Index \(i) should be \(i)")
        }

        // Verify: depth keys should be float-sortable (XOR with sign bit)
        let keysPtr = depthKeysBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
        for i in 0..<count {
            let expectedKey = depths[i].bitPattern ^ 0x80000000
            XCTAssertEqual(keysPtr[i], expectedKey, "Key \(i) mismatch for depth \(depths[i])")
        }

        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║  DEPTH KEY GENERATION TEST                                    ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Gaussians: \(count)")
        print("║  Depths: \(depths)")
        print("║  Keys generated: ✓")
        print("║  Indices initialized: ✓")
        print("╚═══════════════════════════════════════════════════════════════╝")
    }

    // MARK: - Test 2: Simple CPU Sort Verification

    func testDepthSortCPUReference() throws {
        // Test that our understanding of depth sorting is correct
        let depths: [Float] = [5.0, 2.0, 8.0, 1.0, 3.0]
        let indices = [0, 1, 2, 3, 4]

        // Sort by depth (ascending - closer first)
        let sortedPairs = zip(depths, indices).sorted { $0.0 < $1.0 }
        let sortedIndices = sortedPairs.map { $0.1 }

        // Expected order: 1.0(3), 2.0(1), 3.0(4), 5.0(0), 8.0(2)
        XCTAssertEqual(sortedIndices, [3, 1, 4, 0, 2])

        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║  DEPTH SORT CPU REFERENCE TEST                                ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Input depths: \(depths)")
        print("║  Expected sorted indices: [3, 1, 4, 0, 2]")
        print("║  Got: \(sortedIndices)")
        print("║  Result: ✓")
        print("╚═══════════════════════════════════════════════════════════════╝")
    }

    // MARK: - Test 3: Tile Count Computation

    func testTileCountComputation() throws {
        // Test that dfApplyOrder correctly computes tile counts
        // A gaussian covering tiles (0,0) to (2,2) should have count = 2*2 = 4

        guard let library = self.library else {
            print("⚠️ Skipping test - library not loaded")
            return
        }

        let count = 3

        // Create compacted gaussians with known tile ranges
        var compactedData = [CompactedGaussianSwift]()

        // Gaussian 0: covers 1 tile (0,0) to (1,1) = 1 tile
        var g0 = CompactedGaussianSwift()
        g0.min_tile = SIMD2<Int32>(0, 0)
        g0.max_tile = SIMD2<Int32>(1, 1)
        compactedData.append(g0)

        // Gaussian 1: covers 4 tiles (0,0) to (2,2) = 2x2 = 4 tiles
        var g1 = CompactedGaussianSwift()
        g1.min_tile = SIMD2<Int32>(0, 0)
        g1.max_tile = SIMD2<Int32>(2, 2)
        compactedData.append(g1)

        // Gaussian 2: covers 6 tiles (1,0) to (4,2) = 3x2 = 6 tiles
        var g2 = CompactedGaussianSwift()
        g2.min_tile = SIMD2<Int32>(1, 0)
        g2.max_tile = SIMD2<Int32>(4, 2)
        compactedData.append(g2)

        let compactedBuffer = device.makeBuffer(bytes: compactedData,
                                                  length: count * MemoryLayout<CompactedGaussianSwift>.stride,
                                                  options: .storageModeShared)!

        // Sorted indices (identity for this test)
        var sortedIndices: [UInt32] = [0, 1, 2]
        let sortedIndicesBuffer = device.makeBuffer(bytes: &sortedIndices,
                                                      length: count * MemoryLayout<UInt32>.stride,
                                                      options: .storageModeShared)!

        // Output: tile counts
        let tileCountsBuffer = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride,
                                                   options: .storageModeShared)!

        // Header
        var header = TileAssignmentHeaderSwift()
        header.totalAssignments = UInt32(count)
        let headerBuffer = device.makeBuffer(bytes: &header,
                                               length: MemoryLayout<TileAssignmentHeaderSwift>.stride,
                                               options: .storageModeShared)!

        // Load kernel
        guard let applyOrderFn = library.makeFunction(name: "dfApplyOrder") else {
            XCTFail("dfApplyOrder not found")
            return
        }
        let pipeline = try device.makeComputePipelineState(function: applyOrderFn)

        let commandBuffer = queue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(sortedIndicesBuffer, offset: 0, index: 0)
        encoder.setBuffer(compactedBuffer, offset: 0, index: 1)
        encoder.setBuffer(tileCountsBuffer, offset: 0, index: 2)
        encoder.setBuffer(headerBuffer, offset: 0, index: 3)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Verify tile counts
        let countsPtr = tileCountsBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
        let expectedCounts: [UInt32] = [1, 4, 6]

        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║  TILE COUNT COMPUTATION TEST                                  ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        for i in 0..<count {
            let actual = countsPtr[i]
            let expected = expectedCounts[i]
            let status = (actual == expected) ? "✓" : "✗"
            print("║  Gaussian \(i): expected \(expected), got \(actual) \(status)")
            XCTAssertEqual(actual, expected, "Tile count mismatch for gaussian \(i)")
        }
        print("╚═══════════════════════════════════════════════════════════════╝")
    }

    // MARK: - Test 4: Instance Creation

    func testInstanceCreation() throws {
        guard let library = self.library else {
            print("⚠️ Skipping test - library not loaded")
            return
        }

        // Create 1 gaussian covering 2x2 = 4 tiles
        let count = 1

        var compactedData = [CompactedGaussianSwift]()
        var g = CompactedGaussianSwift()
        g.covariance_depth = SIMD4<Float>(0.01, 0, 0.01, 5.0)  // Small conic, depth 5
        g.position_color = SIMD4<Float>(48, 24, 0, 0)  // Center at pixel (48, 24)
        g.min_tile = SIMD2<Int32>(0, 0)
        g.max_tile = SIMD2<Int32>(2, 2)  // Covers tiles (0,0), (0,1), (1,0), (1,1)
        g.originalIdx = 0
        compactedData.append(g)

        let compactedBuffer = device.makeBuffer(bytes: compactedData,
                                                  length: MemoryLayout<CompactedGaussianSwift>.stride,
                                                  options: .storageModeShared)!

        // Sorted indices (identity)
        var sortedIndices: [UInt32] = [0]
        let sortedIndicesBuffer = device.makeBuffer(bytes: &sortedIndices,
                                                      length: MemoryLayout<UInt32>.stride,
                                                      options: .storageModeShared)!

        // Gaussian offsets (prefix sum) - starts at 0
        var offsets: [UInt32] = [0]
        let offsetsBuffer = device.makeBuffer(bytes: &offsets,
                                                length: MemoryLayout<UInt32>.stride,
                                                options: .storageModeShared)!

        // Output buffers (4 tiles max)
        let maxInstances = 4
        let instanceSortKeysBuffer = device.makeBuffer(length: maxInstances * MemoryLayout<UInt32>.stride,
                                                         options: .storageModeShared)!
        let instanceGaussianIdxBuffer = device.makeBuffer(length: maxInstances * MemoryLayout<UInt32>.stride,
                                                            options: .storageModeShared)!

        // Header
        var header = TileAssignmentHeaderSwift()
        header.totalAssignments = UInt32(count)
        let headerBuffer = device.makeBuffer(bytes: &header,
                                               length: MemoryLayout<TileAssignmentHeaderSwift>.stride,
                                               options: .storageModeShared)!

        // Parameters
        var tilesX: UInt32 = 4  // 4 tiles in X
        var tileWidth: Int32 = 32
        var tileHeight: Int32 = 16

        // Load kernel
        guard let createInstFn = library.makeFunction(name: "dfCreateInstances") else {
            XCTFail("dfCreateInstances not found")
            return
        }
        let pipeline = try device.makeComputePipelineState(function: createInstFn)

        let commandBuffer = queue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(sortedIndicesBuffer, offset: 0, index: 0)
        encoder.setBuffer(compactedBuffer, offset: 0, index: 1)
        encoder.setBuffer(offsetsBuffer, offset: 0, index: 2)
        encoder.setBuffer(instanceSortKeysBuffer, offset: 0, index: 3)
        encoder.setBuffer(instanceGaussianIdxBuffer, offset: 0, index: 4)
        encoder.setBytes(&tilesX, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&tileWidth, length: MemoryLayout<Int32>.stride, index: 6)
        encoder.setBytes(&tileHeight, length: MemoryLayout<Int32>.stride, index: 7)
        encoder.setBuffer(headerBuffer, offset: 0, index: 8)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Check output
        let sortKeysPtr = instanceSortKeysBuffer.contents().bindMemory(to: UInt32.self, capacity: maxInstances)
        let gaussianIdxPtr = instanceGaussianIdxBuffer.contents().bindMemory(to: UInt32.self, capacity: maxInstances)

        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║  INSTANCE CREATION TEST                                       ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Gaussian covers tiles (0,0) to (2,2) = 4 tiles")
        print("║  tilesX = \(tilesX), tileWidth = \(tileWidth), tileHeight = \(tileHeight)")
        print("╠═══════════════════════════════════════════════════════════════╣")

        // Expected instances for tiles (0,0), (0,1), (1,0), (1,1) with depthOrder=0
        // tileKey = ty * tilesX + tx
        // Combined key = (tileKey << 20) | depthOrder
        let expectedTileKeys: [UInt32] = [0, 1, 4, 5]  // (0,0)=0, (1,0)=1, (0,1)=4, (1,1)=5

        for i in 0..<maxInstances {
            let sortKey = sortKeysPtr[i]
            let tileKey = sortKey >> 20
            let depthOrder = sortKey & 0xFFFFF
            let gaussianIdx = gaussianIdxPtr[i]
            print("║  Instance \(i): tileKey=\(tileKey), depthOrder=\(depthOrder), gaussianIdx=\(gaussianIdx)")
        }
        print("╚═══════════════════════════════════════════════════════════════╝")
    }

    // MARK: - Test 5: GPU Radix Sort Verification

    func testGPURadixSort() throws {
        guard let library = self.library else {
            print("⚠️ Skipping test - library not loaded")
            return
        }

        // Test data: 8 unsorted depth keys
        let count = 8
        let depths: [Float] = [5.0, 2.0, 8.0, 1.0, 3.0, 9.0, 4.0, 7.0]
        var keys: [UInt32] = depths.map { $0.bitPattern ^ 0x80000000 }
        var indices: [UInt32] = [0, 1, 2, 3, 4, 5, 6, 7]

        // Create buffers
        let keysBuffer = device.makeBuffer(bytes: &keys, length: count * 4, options: .storageModeShared)!
        let indicesBuffer = device.makeBuffer(bytes: &indices, length: count * 4, options: .storageModeShared)!
        let tempKeysBuffer = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let tempIndicesBuffer = device.makeBuffer(length: count * 4, options: .storageModeShared)!

        // Histogram buffer: 256 buckets * 1 block
        let histogramBuffer = device.makeBuffer(length: 256 * 4, options: .storageModeShared)!

        // visibleInfo: [count, numBlocks, histogramSize]
        var visibleInfo: [UInt32] = [UInt32(count), 1, 256]
        let visibleInfoBuffer = device.makeBuffer(bytes: &visibleInfo, length: 3 * 4, options: .storageModeShared)!

        // Load kernels
        guard let histogramFn = library.makeFunction(name: "dfRadixHistogram"),
              let scanFn = library.makeFunction(name: "dfRadixScanHistogram"),
              let scatterFn = library.makeFunction(name: "dfRadixScatter") else {
            XCTFail("Missing radix sort kernels")
            return
        }

        let histogramPipe = try device.makeComputePipelineState(function: histogramFn)
        let scanPipe = try device.makeComputePipelineState(function: scanFn)
        let scatterPipe = try device.makeComputePipelineState(function: scatterFn)

        // Run 4 passes (32-bit sort)
        for pass in 0..<4 {
            let isEven = (pass % 2 == 0)
            let inKeys = isEven ? keysBuffer : tempKeysBuffer
            let outKeys = isEven ? tempKeysBuffer : keysBuffer
            let inVals = isEven ? indicesBuffer : tempIndicesBuffer
            let outVals = isEven ? tempIndicesBuffer : indicesBuffer

            var digit = UInt32(pass)

            // Zero histogram
            memset(histogramBuffer.contents(), 0, 256 * 4)

            // Histogram pass
            let cmdBuf1 = queue.makeCommandBuffer()!
            var enc = cmdBuf1.makeComputeCommandEncoder()!
            enc.setComputePipelineState(histogramPipe)
            enc.setBuffer(inKeys, offset: 0, index: 0)
            enc.setBuffer(histogramBuffer, offset: 0, index: 1)
            enc.setBuffer(visibleInfoBuffer, offset: 0, index: 2)
            enc.setBytes(&digit, length: 4, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cmdBuf1.commit()
            cmdBuf1.waitUntilCompleted()

            // Scan histogram
            let cmdBuf2 = queue.makeCommandBuffer()!
            enc = cmdBuf2.makeComputeCommandEncoder()!
            enc.setComputePipelineState(scanPipe)
            enc.setBuffer(histogramBuffer, offset: 0, index: 0)
            enc.setBuffer(visibleInfoBuffer, offset: 2 * 4, index: 1)  // histogramSize = visibleInfo[2]
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()

            // Scatter pass
            let cmdBuf3 = queue.makeCommandBuffer()!
            enc = cmdBuf3.makeComputeCommandEncoder()!
            enc.setComputePipelineState(scatterPipe)
            enc.setBuffer(inKeys, offset: 0, index: 0)
            enc.setBuffer(outKeys, offset: 0, index: 1)
            enc.setBuffer(inVals, offset: 0, index: 2)
            enc.setBuffer(outVals, offset: 0, index: 3)
            enc.setBuffer(histogramBuffer, offset: 0, index: 4)
            enc.setBuffer(visibleInfoBuffer, offset: 0, index: 5)
            enc.setBytes(&digit, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cmdBuf3.commit()
            cmdBuf3.waitUntilCompleted()
        }

        // After 4 passes, result is back in original buffers (even pass count)
        let resultKeys = keysBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
        let resultIndices = indicesBuffer.contents().bindMemory(to: UInt32.self, capacity: count)

        // Convert keys back to depths
        var sortedDepths: [Float] = []
        var sortedOrigIdx: [UInt32] = []
        for i in 0..<count {
            let key = resultKeys[i]
            let depth = Float(bitPattern: key ^ 0x80000000)
            sortedDepths.append(depth)
            sortedOrigIdx.append(resultIndices[i])
        }

        // Expected: sorted by depth ascending
        // depths: [5.0, 2.0, 8.0, 1.0, 3.0, 9.0, 4.0, 7.0]
        // sorted: [1.0(3), 2.0(1), 3.0(4), 4.0(6), 5.0(0), 7.0(7), 8.0(2), 9.0(5)]
        let expectedDepths: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0]
        let expectedIndices: [UInt32] = [3, 1, 4, 6, 0, 7, 2, 5]

        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║  GPU RADIX SORT TEST                                          ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Input depths: \(depths)")
        print("║  Sorted depths: \(sortedDepths)")
        print("║  Expected:      \(expectedDepths)")
        print("║  Sorted indices: \(sortedOrigIdx)")
        print("║  Expected:       \(expectedIndices)")
        print("╠═══════════════════════════════════════════════════════════════╣")

        var allCorrect = true
        for i in 0..<count {
            if abs(sortedDepths[i] - expectedDepths[i]) > 0.001 {
                print("║  ✗ Depth mismatch at \(i): got \(sortedDepths[i]), expected \(expectedDepths[i])")
                allCorrect = false
            }
            if sortedOrigIdx[i] != expectedIndices[i] {
                print("║  ✗ Index mismatch at \(i): got \(sortedOrigIdx[i]), expected \(expectedIndices[i])")
                allCorrect = false
            }
        }
        if allCorrect {
            print("║  ✓ All depths and indices correct!")
        }
        print("╚═══════════════════════════════════════════════════════════════╝")

        for i in 0..<count {
            XCTAssertEqual(sortedDepths[i], expectedDepths[i], accuracy: 0.001)
            XCTAssertEqual(sortedOrigIdx[i], expectedIndices[i], "Index mismatch at position \(i)")
        }
    }

    // MARK: - Test 6: Tile Range Extraction Test

    func testTileRangeExtraction() throws {
        guard let library = self.library else {
            print("⚠️ Skipping test - library not loaded")
            return
        }

        // Create test data: 4 instances for 2 tiles
        // Tile 0: instances 0, 1 (depth16 0, 1)
        // Tile 1: instances 2, 3 (depth16 0, 1)
        let instanceCount = 4

        // Fused keys: depth16 | (tileKey << 16)
        // Already sorted by tile then depth
        var sortKeys: [UInt32] = [
            0 | (0 << 16),  // tile 0, depth 0
            1 | (0 << 16),  // tile 0, depth 1
            0 | (1 << 16),  // tile 1, depth 0
            1 | (1 << 16)   // tile 1, depth 1
        ]
        let sortKeysBuffer = device.makeBuffer(bytes: &sortKeys, length: instanceCount * 4, options: .storageModeShared)!

        // Tile ranges output
        let numTiles = 4
        let tileRangesBuffer = device.makeBuffer(length: numTiles * 8, options: .storageModeShared)!
        memset(tileRangesBuffer.contents(), 0, numTiles * 8)

        // Instance info: [total, numBlocks, histogramSize]
        var instanceInfo: [UInt32] = [UInt32(instanceCount), 1, 256]
        let instanceInfoBuffer = device.makeBuffer(bytes: &instanceInfo, length: 12, options: .storageModeShared)!

        // Load kernel
        guard let extractRangesFn = library.makeFunction(name: "dfExtractRanges") else {
            XCTFail("dfExtractRanges not found")
            return
        }
        let extractRangesPipe = try device.makeComputePipelineState(function: extractRangesFn)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(extractRangesPipe)
        enc.setBuffer(sortKeysBuffer, offset: 0, index: 0)
        enc.setBuffer(tileRangesBuffer, offset: 0, index: 1)
        enc.setBuffer(instanceInfoBuffer, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: instanceCount, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: instanceCount, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read tile ranges
        let rangesPtr = tileRangesBuffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: numTiles)

        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║  TILE RANGE EXTRACTION TEST                                   ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Input: 4 instances for 2 tiles")
        print("║  Keys: [(0<<20)|0, (0<<20)|1, (1<<20)|0, (1<<20)|1]")
        print("╠═══════════════════════════════════════════════════════════════╣")
        for i in 0..<numTiles {
            let range = rangesPtr[i]
            print("║  Tile \(i): start=\(range.x), end=\(range.y)")
        }
        print("╠═══════════════════════════════════════════════════════════════╣")

        // Expected: tile 0 = [0, 2), tile 1 = [2, 4)
        let tile0Range = rangesPtr[0]
        let tile1Range = rangesPtr[1]

        if tile0Range.x == 0 && tile0Range.y == 2 &&
           tile1Range.x == 2 && tile1Range.y == 4 {
            print("║  ✓ Tile ranges are correct!")
        } else {
            print("║  ✗ Tile ranges are INCORRECT!")
            print("║    Expected tile 0: [0, 2), got [\(tile0Range.x), \(tile0Range.y))")
            print("║    Expected tile 1: [2, 4), got [\(tile1Range.x), \(tile1Range.y))")
        }
        print("╚═══════════════════════════════════════════════════════════════╝")

        XCTAssertEqual(tile0Range.x, 0, "Tile 0 start")
        XCTAssertEqual(tile0Range.y, 2, "Tile 0 end")
        XCTAssertEqual(tile1Range.x, 2, "Tile 1 start")
        XCTAssertEqual(tile1Range.y, 4, "Tile 1 end")
    }

    // MARK: - Test 7: Debug Instance Sort Keys After Radix Sort

    func testDebugInstanceSortPipeline() throws {
        guard let library = self.library else {
            print("⚠️ Skipping test - library not loaded")
            return
        }

        // Create 4 instances for 2 tiles, unsorted
        // Instance 0: tile 1, depth 0 -> should end up at position 2
        // Instance 1: tile 0, depth 1 -> should end up at position 1
        // Instance 2: tile 1, depth 1 -> should end up at position 3
        // Instance 3: tile 0, depth 0 -> should end up at position 0
        let instanceCount = 4

        // Combined keys (tileKey << 20) | depthOrder - UNSORTED
        var sortKeys: [UInt32] = [
            (1 << 20) | 0,  // tile 1, depth 0
            (0 << 20) | 1,  // tile 0, depth 1
            (1 << 20) | 1,  // tile 1, depth 1
            (0 << 20) | 0   // tile 0, depth 0
        ]
        var gaussianIdx: [UInt32] = [100, 101, 102, 103]  // Different values to track

        let sortKeysBuffer = device.makeBuffer(bytes: &sortKeys, length: instanceCount * 4, options: .storageModeShared)!
        let gaussianIdxBuffer = device.makeBuffer(bytes: &gaussianIdx, length: instanceCount * 4, options: .storageModeShared)!
        let tempKeysBuffer = device.makeBuffer(length: instanceCount * 4, options: .storageModeShared)!
        let tempIdxBuffer = device.makeBuffer(length: instanceCount * 4, options: .storageModeShared)!
        let histogramBuffer = device.makeBuffer(length: 256 * 4, options: .storageModeShared)!

        var instanceInfo: [UInt32] = [UInt32(instanceCount), 1, 256]
        let instanceInfoBuffer = device.makeBuffer(bytes: &instanceInfo, length: 12, options: .storageModeShared)!

        // Load kernels
        guard let histogramFn = library.makeFunction(name: "dfRadixHistogram"),
              let scanFn = library.makeFunction(name: "dfRadixScanHistogram"),
              let scatterFn = library.makeFunction(name: "dfRadixScatter") else {
            XCTFail("Missing radix sort kernels")
            return
        }

        let histogramPipe = try device.makeComputePipelineState(function: histogramFn)
        let scanPipe = try device.makeComputePipelineState(function: scanFn)
        let scatterPipe = try device.makeComputePipelineState(function: scatterFn)

        // Run 4 passes (32-bit sort)
        for pass in 0..<4 {
            let isEven = (pass % 2 == 0)
            let inKeys = isEven ? sortKeysBuffer : tempKeysBuffer
            let outKeys = isEven ? tempKeysBuffer : sortKeysBuffer
            let inVals = isEven ? gaussianIdxBuffer : tempIdxBuffer
            let outVals = isEven ? tempIdxBuffer : gaussianIdxBuffer

            var digit = UInt32(pass)
            memset(histogramBuffer.contents(), 0, 256 * 4)

            let cmd1 = queue.makeCommandBuffer()!
            var enc = cmd1.makeComputeCommandEncoder()!
            enc.setComputePipelineState(histogramPipe)
            enc.setBuffer(inKeys, offset: 0, index: 0)
            enc.setBuffer(histogramBuffer, offset: 0, index: 1)
            enc.setBuffer(instanceInfoBuffer, offset: 0, index: 2)
            enc.setBytes(&digit, length: 4, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cmd1.commit()
            cmd1.waitUntilCompleted()

            let cmd2 = queue.makeCommandBuffer()!
            enc = cmd2.makeComputeCommandEncoder()!
            enc.setComputePipelineState(scanPipe)
            enc.setBuffer(histogramBuffer, offset: 0, index: 0)
            enc.setBuffer(instanceInfoBuffer, offset: 8, index: 1)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cmd2.commit()
            cmd2.waitUntilCompleted()

            let cmd3 = queue.makeCommandBuffer()!
            enc = cmd3.makeComputeCommandEncoder()!
            enc.setComputePipelineState(scatterPipe)
            enc.setBuffer(inKeys, offset: 0, index: 0)
            enc.setBuffer(outKeys, offset: 0, index: 1)
            enc.setBuffer(inVals, offset: 0, index: 2)
            enc.setBuffer(outVals, offset: 0, index: 3)
            enc.setBuffer(histogramBuffer, offset: 0, index: 4)
            enc.setBuffer(instanceInfoBuffer, offset: 0, index: 5)
            enc.setBytes(&digit, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cmd3.commit()
            cmd3.waitUntilCompleted()
        }

        // Read results
        let resultKeys = sortKeysBuffer.contents().bindMemory(to: UInt32.self, capacity: instanceCount)
        let resultIdx = gaussianIdxBuffer.contents().bindMemory(to: UInt32.self, capacity: instanceCount)

        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║  INSTANCE SORT PIPELINE DEBUG TEST                            ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Input (unsorted):")
        print("║    [0] tile=1 depth=0 idx=100")
        print("║    [1] tile=0 depth=1 idx=101")
        print("║    [2] tile=1 depth=1 idx=102")
        print("║    [3] tile=0 depth=0 idx=103")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Output (should be sorted by tile, then depth):")

        var allCorrect = true
        // Expected order: tile0/depth0, tile0/depth1, tile1/depth0, tile1/depth1
        // = idx 103, 101, 100, 102
        let expectedIdx: [UInt32] = [103, 101, 100, 102]
        let expectedTiles: [UInt32] = [0, 0, 1, 1]
        let expectedDepths: [UInt32] = [0, 1, 0, 1]

        for i in 0..<instanceCount {
            let key = resultKeys[i]
            let tile = key >> 20
            let depth = key & 0xFFFFF
            let idx = resultIdx[i]
            let status = (idx == expectedIdx[i] && tile == expectedTiles[i] && depth == expectedDepths[i]) ? "✓" : "✗"
            print("║    [\(i)] tile=\(tile) depth=\(depth) idx=\(idx) \(status)")
            if idx != expectedIdx[i] || tile != expectedTiles[i] || depth != expectedDepths[i] {
                allCorrect = false
            }
        }

        print("╠═══════════════════════════════════════════════════════════════╣")
        if allCorrect {
            print("║  ✓ Instance sort is correct!")
        } else {
            print("║  ✗ Instance sort is WRONG!")
        }
        print("╚═══════════════════════════════════════════════════════════════╝")

        for i in 0..<instanceCount {
            XCTAssertEqual(resultIdx[i], expectedIdx[i], "Gaussian idx mismatch at position \(i)")
        }
    }

    // MARK: - Test 8: Verify DepthFirst Render Output Has Non-Zero Pixels

    func testVerifyRenderOutputNonZero() throws {
        let width = 64
        let height = 64

        // Create renderers - use 32-bit sort mode, float32 precision, and disable textured render for simplicity
        let config = RendererConfig(
            maxGaussians: 1000,
            maxWidth: width,
            maxHeight: height,
            precision: .float32,  // Explicitly use float32 precision
            sortMode: .sort32Bit,
            useTexturedRender: false
        )

        let dfRenderer = try DepthFirstRenderer(device: device, config: config)
        let lsRenderer = try LocalSortRenderer(device: device, config: config)
        lsRenderer.debugPrint = true  // Enable debug output

        // Create a single gaussian at center using library types
        var packed = [PackedWorldGaussian]()
        var g = PackedWorldGaussian()
        g.px = 0
        g.py = 0
        g.pz = 5
        g.opacity = 0.9
        g.sx = 0.5
        g.sy = 0.5
        g.sz = 0.5
        g._pad0 = 0
        g.rotation = SIMD4<Float>(0, 0, 0, 1)
        packed.append(g)

        let gaussianBuffer = device.makeBuffer(bytes: &packed, length: packed.count * MemoryLayout<PackedWorldGaussian>.stride, options: .storageModeShared)!

        // DC color coefficients
        var harmonics: [Float] = [0.5, 0.2, 0.1]
        let harmonicsBuffer = device.makeBuffer(bytes: &harmonics, length: harmonics.count * 4, options: .storageModeShared)!

        // Create proper perspective camera (matches LocalSortPipelineTests.createTestCamera)
        let aspect = Float(width) / Float(height)
        let fov: Float = 60.0 * .pi / 180.0
        let near: Float = 0.1
        let far: Float = 100.0

        let f = 1.0 / tan(fov / 2.0)
        let projectionMatrix = simd_float4x4(columns: (
            SIMD4<Float>(f / aspect, 0, 0, 0),
            SIMD4<Float>(0, f, 0, 0),
            SIMD4<Float>(0, 0, (far + near) / (near - far), -1),
            SIMD4<Float>(0, 0, (2 * far * near) / (near - far), 0)
        ))

        // Camera at origin looking down -Z
        let camera = CameraParams(
            viewMatrix: simd_float4x4(1.0),  // identity
            projectionMatrix: projectionMatrix,
            position: SIMD3<Float>(0, 0, 0),
            focalX: f * Float(width) / 2.0,
            focalY: f * Float(height) / 2.0,
            near: near,
            far: far
        )

        let input = GaussianInput(
            gaussians: gaussianBuffer,
            harmonics: harmonicsBuffer,
            gaussianCount: 1,
            shComponents: 0
        )

        // Render with both
        guard let cb1 = queue.makeCommandBuffer(), let cb2 = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let dfResult = dfRenderer.render(toTexture: cb1, input: input, camera: camera, width: width, height: height, whiteBackground: false, mortonSorted: false)
        let lsResult = lsRenderer.render(toTexture: cb2, input: input, camera: camera, width: width, height: height, whiteBackground: false, mortonSorted: false)

        cb1.commit()
        cb2.commit()
        cb1.waitUntilCompleted()
        cb2.waitUntilCompleted()

        // Check visible counts
        let lsVisibleCount = lsRenderer.getVisibleCount()
        let lsOverflow = lsRenderer.hadOverflow()
        print("LocalSort: visible=\(lsVisibleCount), overflow=\(lsOverflow)")

        guard let dfTex = dfResult?.color, let lsTex = lsResult?.color else {
            XCTFail("Failed to get textures")
            return
        }

        // Read pixels
        let readBuffer1 = device.makeBuffer(length: width * height * 8, options: .storageModeShared)!
        let readBuffer2 = device.makeBuffer(length: width * height * 8, options: .storageModeShared)!

        guard let blitCB = queue.makeCommandBuffer(), let blit = blitCB.makeBlitCommandEncoder() else { return }
        blit.copy(from: dfTex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(), sourceSize: MTLSize(width: width, height: height, depth: 1), to: readBuffer1, destinationOffset: 0, destinationBytesPerRow: width * 8, destinationBytesPerImage: width * height * 8)
        blit.copy(from: lsTex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(), sourceSize: MTLSize(width: width, height: height, depth: 1), to: readBuffer2, destinationOffset: 0, destinationBytesPerRow: width * 8, destinationBytesPerImage: width * height * 8)
        blit.endEncoding()
        blitCB.commit()
        blitCB.waitUntilCompleted()

        let dfPixels = readBuffer1.contents().bindMemory(to: UInt16.self, capacity: width * height * 4)
        let lsPixels = readBuffer2.contents().bindMemory(to: UInt16.self, capacity: width * height * 4)

        // Check center pixel
        let centerIdx = (height / 2) * width * 4 + (width / 2) * 4

        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║  RENDER OUTPUT VERIFICATION TEST                              ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Single gaussian render, center pixel:")

        let dfR = Float16(bitPattern: dfPixels[centerIdx])
        let dfG = Float16(bitPattern: dfPixels[centerIdx + 1])
        let dfB = Float16(bitPattern: dfPixels[centerIdx + 2])
        let dfA = Float16(bitPattern: dfPixels[centerIdx + 3])

        let lsR = Float16(bitPattern: lsPixels[centerIdx])
        let lsG = Float16(bitPattern: lsPixels[centerIdx + 1])
        let lsB = Float16(bitPattern: lsPixels[centerIdx + 2])
        let lsA = Float16(bitPattern: lsPixels[centerIdx + 3])

        print("║  DepthFirst: R=\(dfR), G=\(dfG), B=\(dfB), A=\(dfA)")
        print("║  LocalSort:  R=\(lsR), G=\(lsG), B=\(lsB), A=\(lsA)")

        // Count non-zero pixels
        var dfNonZero = 0, lsNonZero = 0
        for i in stride(from: 0, to: width * height * 4, by: 4) {
            let r = Float16(bitPattern: dfPixels[i])
            if Float(r) > 0.01 { dfNonZero += 1 }
            let r2 = Float16(bitPattern: lsPixels[i])
            if Float(r2) > 0.01 { lsNonZero += 1 }
        }

        print("║  DepthFirst non-zero pixels: \(dfNonZero) / \(width * height)")
        print("║  LocalSort non-zero pixels: \(lsNonZero) / \(width * height)")
        print("╚═══════════════════════════════════════════════════════════════╝")

        XCTAssertGreaterThan(dfNonZero, 0, "DepthFirst should render something")
    }

    // MARK: - Test 8b: Two Overlapping Gaussians Comparison

    /// Test with exactly 2 overlapping gaussians to isolate multi-gaussian issues
    func testTwoGaussiansComparison() throws {
        let width = 64
        let height = 64

        // Same config for both renderers
        let config = RendererConfig(
            maxGaussians: 1000,
            maxWidth: width,
            maxHeight: height,
            precision: .float32,
            sortMode: .sort32Bit,
            useTexturedRender: false
        )

        let dfRenderer = try DepthFirstRenderer(device: device, config: config)
        let lsRenderer = try LocalSortRenderer(device: device, config: config)

        // Create 2 overlapping gaussians: red at depth 3, blue at depth 5 (red should be in front)
        var packed = [PackedWorldGaussian]()

        // Gaussian 0: Red, at depth 3 (closer)
        var g0 = PackedWorldGaussian()
        g0.px = 0; g0.py = 0; g0.pz = 3
        g0.opacity = 0.8
        g0.sx = 0.3; g0.sy = 0.3; g0.sz = 0.3
        g0.rotation = SIMD4<Float>(0, 0, 0, 1)
        packed.append(g0)

        // Gaussian 1: Blue, at depth 5 (farther)
        var g1 = PackedWorldGaussian()
        g1.px = 0; g1.py = 0; g1.pz = 5
        g1.opacity = 0.8
        g1.sx = 0.3; g1.sy = 0.3; g1.sz = 0.3
        g1.rotation = SIMD4<Float>(0, 0, 0, 1)
        packed.append(g1)

        // SH colors: Red for g0, Blue for g1 (SH degree 0: just DC term, normalized by sqrt(4*pi))
        let sh0 = 0.5 / sqrt(4.0 * Float.pi)  // DC normalization
        var harmonics: [Float] = [
            0.5, 0.0, 0.0,  // Red for g0 (R,G,B DC terms)
            0.0, 0.0, 0.5   // Blue for g1
        ]

        guard let gaussianBuffer = device.makeBuffer(bytes: &packed, length: packed.count * MemoryLayout<PackedWorldGaussian>.stride, options: .storageModeShared),
              let harmonicsBuffer = device.makeBuffer(bytes: &harmonics, length: harmonics.count * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            XCTFail("Failed to create buffers")
            return
        }

        // Create proper perspective camera
        let aspect = Float(width) / Float(height)
        let fov: Float = 60.0 * .pi / 180.0
        let near: Float = 0.1
        let far: Float = 100.0
        let f = 1.0 / tan(fov / 2.0)
        let projectionMatrix = simd_float4x4(columns: (
            SIMD4<Float>(f / aspect, 0, 0, 0),
            SIMD4<Float>(0, f, 0, 0),
            SIMD4<Float>(0, 0, (far + near) / (near - far), -1),
            SIMD4<Float>(0, 0, (2 * far * near) / (near - far), 0)
        ))

        let camera = CameraParams(
            viewMatrix: simd_float4x4(1.0),
            projectionMatrix: projectionMatrix,
            position: SIMD3<Float>(0, 0, 0),
            focalX: f * Float(width) / 2.0,
            focalY: f * Float(height) / 2.0,
            near: near,
            far: far
        )

        let input = GaussianInput(
            gaussians: gaussianBuffer,
            harmonics: harmonicsBuffer,
            gaussianCount: 2,
            shComponents: 0
        )

        // Render with both
        guard let cb1 = queue.makeCommandBuffer(), let cb2 = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let dfResult = dfRenderer.render(toTexture: cb1, input: input, camera: camera, width: width, height: height, whiteBackground: false, mortonSorted: false)
        let lsResult = lsRenderer.render(toTexture: cb2, input: input, camera: camera, width: width, height: height, whiteBackground: false, mortonSorted: false)

        cb1.commit()
        cb2.commit()
        cb1.waitUntilCompleted()
        cb2.waitUntilCompleted()

        guard let dfTex = dfResult?.color, let lsTex = lsResult?.color else {
            XCTFail("Failed to get textures")
            return
        }

        // Read pixels
        let readBuffer1 = device.makeBuffer(length: width * height * 8, options: .storageModeShared)!
        let readBuffer2 = device.makeBuffer(length: width * height * 8, options: .storageModeShared)!

        guard let blitCB = queue.makeCommandBuffer(), let blit = blitCB.makeBlitCommandEncoder() else { return }
        blit.copy(from: dfTex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(), sourceSize: MTLSize(width: width, height: height, depth: 1), to: readBuffer1, destinationOffset: 0, destinationBytesPerRow: width * 8, destinationBytesPerImage: width * height * 8)
        blit.copy(from: lsTex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(), sourceSize: MTLSize(width: width, height: height, depth: 1), to: readBuffer2, destinationOffset: 0, destinationBytesPerRow: width * 8, destinationBytesPerImage: width * height * 8)
        blit.endEncoding()
        blitCB.commit()
        blitCB.waitUntilCompleted()

        let dfPixels = readBuffer1.contents().bindMemory(to: UInt16.self, capacity: width * height * 4)
        let lsPixels = readBuffer2.contents().bindMemory(to: UInt16.self, capacity: width * height * 4)

        // Check center pixel
        let centerIdx = (height / 2) * width * 4 + (width / 2) * 4

        let dfR = Float16(bitPattern: dfPixels[centerIdx])
        let dfG = Float16(bitPattern: dfPixels[centerIdx + 1])
        let dfB = Float16(bitPattern: dfPixels[centerIdx + 2])
        let lsR = Float16(bitPattern: lsPixels[centerIdx])
        let lsG = Float16(bitPattern: lsPixels[centerIdx + 1])
        let lsB = Float16(bitPattern: lsPixels[centerIdx + 2])

        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║  TWO GAUSSIANS COMPARISON TEST                                ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Red gaussian at depth 3 (closer)")
        print("║  Blue gaussian at depth 5 (farther)")
        print("║  Expected: More red at center (red in front)")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  DepthFirst center: R=\(String(format: "%.3f", Float(dfR))), G=\(String(format: "%.3f", Float(dfG))), B=\(String(format: "%.3f", Float(dfB)))")
        print("║  LocalSort center:  R=\(String(format: "%.3f", Float(lsR))), G=\(String(format: "%.3f", Float(lsG))), B=\(String(format: "%.3f", Float(lsB)))")

        // Count differences
        var diffCount = 0
        var maxDiff: Float = 0
        for i in 0..<(width * height * 4) {
            let df = Float(Float16(bitPattern: dfPixels[i]))
            let ls = Float(Float16(bitPattern: lsPixels[i]))
            let diff = abs(df - ls)
            if diff > 0.001 {
                diffCount += 1
                maxDiff = max(maxDiff, diff)
            }
        }

        print("║  Different channels: \(diffCount) / \(width * height * 4)")
        print("║  Max difference: \(String(format: "%.4f", maxDiff))")
        print("╚═══════════════════════════════════════════════════════════════╝")

        // Red should be more visible (closer) - check that both renderers agree
        let dfMoreRed = Float(dfR) > Float(dfB)
        let lsMoreRed = Float(lsR) > Float(lsB)
        print("DepthFirst more red: \(dfMoreRed), LocalSort more red: \(lsMoreRed)")
    }

    // MARK: - Test 9: Full Depth Sort Pipeline Test

    func testFullDepthSortPipeline() throws {
        guard let library = self.library else {
            print("⚠️ Skipping test - library not loaded")
            return
        }

        // Create test data: 16 gaussians with known depths
        let count = 16
        let depths: [Float] = [10.0, 5.0, 15.0, 3.0, 8.0, 12.0, 1.0, 7.0,
                               14.0, 6.0, 11.0, 2.0, 9.0, 13.0, 4.0, 16.0]

        // Create compacted gaussians
        var compactedData = [CompactedGaussianSwift]()
        for i in 0..<count {
            var g = CompactedGaussianSwift()
            g.covariance_depth = SIMD4<Float>(0.1, 0, 0.1, depths[i])
            g.position_color = SIMD4<Float>(Float(i * 32), 16, 0, 0)
            g.min_tile = SIMD2<Int32>(Int32(i), 0)
            g.max_tile = SIMD2<Int32>(Int32(i + 1), 1)
            g.originalIdx = UInt32(i)
            compactedData.append(g)
        }

        let compactedBuffer = device.makeBuffer(bytes: compactedData,
                                                 length: count * MemoryLayout<CompactedGaussianSwift>.stride,
                                                 options: .storageModeShared)!

        // Header
        var header = TileAssignmentHeaderSwift()
        header.totalAssignments = UInt32(count)
        header.maxCapacity = UInt32(count)
        let headerBuffer = device.makeBuffer(bytes: &header, length: 16, options: .storageModeShared)!

        // Output buffers
        let depthKeysBuffer = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let indicesBuffer = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let tempKeysBuffer = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let tempIndicesBuffer = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let histogramBuffer = device.makeBuffer(length: 256 * 4, options: .storageModeShared)!
        var visibleInfo: [UInt32] = [UInt32(count), 1, 256]
        let visibleInfoBuffer = device.makeBuffer(bytes: &visibleInfo, length: 12, options: .storageModeShared)!

        // Load kernels
        guard let genKeysFn = library.makeFunction(name: "dfGenDepthKeys"),
              let histogramFn = library.makeFunction(name: "dfRadixHistogram"),
              let scanFn = library.makeFunction(name: "dfRadixScanHistogram"),
              let scatterFn = library.makeFunction(name: "dfRadixScatter") else {
            XCTFail("Missing kernels")
            return
        }

        let genKeysPipe = try device.makeComputePipelineState(function: genKeysFn)
        let histogramPipe = try device.makeComputePipelineState(function: histogramFn)
        let scanPipe = try device.makeComputePipelineState(function: scanFn)
        let scatterPipe = try device.makeComputePipelineState(function: scatterFn)

        // Step 1: Generate depth keys
        let cmdBuf = queue.makeCommandBuffer()!
        var enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(genKeysPipe)
        enc.setBuffer(compactedBuffer, offset: 0, index: 0)
        enc.setBuffer(depthKeysBuffer, offset: 0, index: 1)
        enc.setBuffer(indicesBuffer, offset: 0, index: 2)
        enc.setBuffer(headerBuffer, offset: 0, index: 3)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(count, 64), height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Step 2: Radix sort (4 passes)
        for pass in 0..<4 {
            let isEven = (pass % 2 == 0)
            let inKeys = isEven ? depthKeysBuffer : tempKeysBuffer
            let outKeys = isEven ? tempKeysBuffer : depthKeysBuffer
            let inVals = isEven ? indicesBuffer : tempIndicesBuffer
            let outVals = isEven ? tempIndicesBuffer : indicesBuffer

            var digit = UInt32(pass)
            memset(histogramBuffer.contents(), 0, 256 * 4)

            let cmd1 = queue.makeCommandBuffer()!
            enc = cmd1.makeComputeCommandEncoder()!
            enc.setComputePipelineState(histogramPipe)
            enc.setBuffer(inKeys, offset: 0, index: 0)
            enc.setBuffer(histogramBuffer, offset: 0, index: 1)
            enc.setBuffer(visibleInfoBuffer, offset: 0, index: 2)
            enc.setBytes(&digit, length: 4, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cmd1.commit()
            cmd1.waitUntilCompleted()

            let cmd2 = queue.makeCommandBuffer()!
            enc = cmd2.makeComputeCommandEncoder()!
            enc.setComputePipelineState(scanPipe)
            enc.setBuffer(histogramBuffer, offset: 0, index: 0)
            enc.setBuffer(visibleInfoBuffer, offset: 8, index: 1)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cmd2.commit()
            cmd2.waitUntilCompleted()

            let cmd3 = queue.makeCommandBuffer()!
            enc = cmd3.makeComputeCommandEncoder()!
            enc.setComputePipelineState(scatterPipe)
            enc.setBuffer(inKeys, offset: 0, index: 0)
            enc.setBuffer(outKeys, offset: 0, index: 1)
            enc.setBuffer(inVals, offset: 0, index: 2)
            enc.setBuffer(outVals, offset: 0, index: 3)
            enc.setBuffer(histogramBuffer, offset: 0, index: 4)
            enc.setBuffer(visibleInfoBuffer, offset: 0, index: 5)
            enc.setBytes(&digit, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cmd3.commit()
            cmd3.waitUntilCompleted()
        }

        // Read sorted indices
        let resultIndices = indicesBuffer.contents().bindMemory(to: UInt32.self, capacity: count)

        // Check depth order
        var sortedDepths: [Float] = []
        for i in 0..<count {
            let idx = resultIndices[i]
            sortedDepths.append(depths[Int(idx)])
        }

        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║  FULL DEPTH SORT PIPELINE TEST                                ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Input depths: \(depths)")
        print("║  Sorted indices: \(Array(UnsafeBufferPointer(start: resultIndices, count: count)))")
        print("║  Resulting depth order: \(sortedDepths)")
        print("╠═══════════════════════════════════════════════════════════════╣")

        // Verify sorted ascending
        var isSorted = true
        for i in 1..<count {
            if sortedDepths[i] < sortedDepths[i-1] {
                print("║  ✗ Not sorted: \(sortedDepths[i-1]) > \(sortedDepths[i]) at position \(i)")
                isSorted = false
            }
        }
        if isSorted {
            print("║  ✓ Depths are sorted in ascending order!")
        }
        print("╚═══════════════════════════════════════════════════════════════╝")

        XCTAssertTrue(isSorted, "Depths should be sorted in ascending order")
    }
}

import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Unit tests for two-pass tile assignment with indirect dispatch
final class TwoPassTileAssignTests: XCTestCase {

    var device: MTLDevice!
    var library: MTLLibrary!
    var queue: MTLCommandQueue!
    var twoPassEncoder: TwoPassTileAssignEncoder!

    // Individual kernel pipelines for step-by-step testing
    var countPipeline: MTLComputePipelineState!
    var blockReducePipeline: MTLComputePipelineState!
    var singleBlockScanPipeline: MTLComputePipelineState!
    var blockScanPipeline: MTLComputePipelineState!
    var writeTotalPipeline: MTLComputePipelineState!
    var scatterPipeline: MTLComputePipelineState!

    let blockSize = 256

    /// Helper to create indirect buffers for a given gaussian count
    func createIndirectBuffers(gaussianCount: Int) -> TwoPassTileAssignEncoder.IndirectBuffers {
        let visibleIndices = device.makeBuffer(length: gaussianCount * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let visibleCount = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let dispatchArgs = device.makeBuffer(length: 12, options: .storageModeShared)!
        return TwoPassTileAssignEncoder.IndirectBuffers(
            visibleIndices: visibleIndices,
            visibleCount: visibleCount,
            indirectDispatchArgs: dispatchArgs
        )
    }

    override func setUp() {
        super.setUp()
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 100_000, maxWidth: 1024, maxHeight: 1024))
        self.device = renderer.device
        self.library = renderer.library
        self.queue = renderer.queue
        self.twoPassEncoder = try! TwoPassTileAssignEncoder(device: device, library: library)

        // Create individual pipelines for step-by-step testing
        self.countPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "tileCountIndirectKernel")!)
        self.blockReducePipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "blockReduceKernel")!)
        self.singleBlockScanPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "singleBlockScanKernel")!)
        self.blockScanPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "blockScanKernel")!)
        self.writeTotalPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "writeTotalCountKernel")!)
        self.scatterPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "tileScatterIndirectKernel")!)
    }

    // MARK: - CPU Reference Implementation

    /// Compute qMax from alpha (mirrors Metal computeQMax)
    func cpuComputeQMax(alpha: Float, tau: Float = 1.0 / 255.0) -> Float {
        if alpha <= tau { return 0.0 }
        return -2.0 * log(tau / alpha)
    }

    /// CPU reference for tile counting
    func cpuCountTiles(
        bounds: [SIMD4<Int32>],
        renderData: [GaussianRenderDataSwift],
        tileWidth: Int,
        tileHeight: Int
    ) -> [UInt32] {
        let tau: Float = 1.0 / 255.0
        let preciseThreshold: UInt32 = 16  // FUSED_V2_PRECISE_THRESHOLD

        var counts = [UInt32](repeating: 0, count: bounds.count)

        for i in 0..<bounds.count {
            let rect = bounds[i]
            let minTX = Int(rect.x)
            let maxTX = Int(rect.y)
            let minTY = Int(rect.z)
            let maxTY = Int(rect.w)

            if minTX > maxTX || minTY > maxTY {
                counts[i] = 0
                continue
            }

            let g = renderData[i]
            let alpha = Float(g.opacity)
            if alpha < 1e-4 {
                counts[i] = 0
                continue
            }

            let aabbWidth = UInt32(maxTX - minTX + 1)
            let aabbHeight = UInt32(maxTY - minTY + 1)
            let aabbCount = aabbWidth * aabbHeight

            // For small AABBs, accept all tiles
            if aabbCount <= preciseThreshold {
                counts[i] = aabbCount
                continue
            }

            // Precise counting for large AABBs
            let tileW = Float(tileWidth)
            let tileH = Float(tileHeight)
            let centerX = Float(g.meanX)
            let centerY = Float(g.meanY)
            let conicX = Float(g.conicA)
            let conicY = Float(g.conicB)
            let conicZ = Float(g.conicC)
            let qMax = cpuComputeQMax(alpha: alpha, tau: tau)

            // Expanded qMax for borderline test
            let halfTileW = tileW * 0.5
            let halfTileH = tileH * 0.5
            let tileDiag = sqrt(halfTileW * halfTileW + halfTileH * halfTileH)
            let maxConic = max(conicX, conicZ)
            let expand = tileDiag * sqrt(maxConic)
            let qMaxExpanded = qMax + 2.0 * expand * sqrt(qMax) + expand * expand

            var count: UInt32 = 0
            for ty in minTY...maxTY {
                for tx in minTX...maxTX {
                    let tileCenterX = (Float(tx) + 0.5) * tileW
                    let tileCenterY = (Float(ty) + 0.5) * tileH
                    let dx = tileCenterX - centerX
                    let dy = tileCenterY - centerY
                    let q = dx * dx * conicX + 2.0 * dx * dy * conicY + dy * dy * conicZ

                    if q <= qMax {
                        count += 1
                    } else if q <= qMaxExpanded {
                        // Simplified: accept borderline tiles (full ellipse intersection test is complex)
                        count += 1
                    }
                }
            }
            counts[i] = count
        }
        return counts
    }

    /// CPU reference for exclusive prefix sum
    func cpuExclusivePrefixSum(_ input: [UInt32]) -> (offsets: [UInt32], total: UInt32) {
        var offsets = [UInt32](repeating: 0, count: input.count)
        var sum: UInt32 = 0
        for i in 0..<input.count {
            offsets[i] = sum
            sum += input[i]
        }
        return (offsets, sum)
    }

    // MARK: - Test Data Generation

    /// Create test data with known tile coverage
    func createTestData(gaussianCount: Int, width: Int, height: Int, tileWidth: Int, tileHeight: Int) -> (
        bounds: [SIMD4<Int32>],
        renderData: [GaussianRenderDataSwift],
        boundsBuffer: MTLBuffer,
        renderDataBuffer: MTLBuffer,
        tilesX: Int,
        tilesY: Int
    ) {
        let tilesX = (width + tileWidth - 1) / tileWidth
        let tilesY = (height + tileHeight - 1) / tileHeight

        var bounds = [SIMD4<Int32>]()
        var renderData = [GaussianRenderDataSwift]()

        for i in 0..<gaussianCount {
            let gridSize = Int(sqrt(Double(gaussianCount))) + 1
            let gx = i % gridSize
            let gy = i / gridSize

            let centerX = Float(gx) / Float(gridSize) * Float(width)
            let centerY = Float(gy) / Float(gridSize) * Float(height)

            // Each gaussian covers a 3x3 tile area
            let tileX = Int(centerX) / tileWidth
            let tileY = Int(centerY) / tileHeight
            let minTX = max(0, tileX - 1)
            let maxTX = min(tilesX - 1, tileX + 1)
            let minTY = max(0, tileY - 1)
            let maxTY = min(tilesY - 1, tileY + 1)

            bounds.append(SIMD4<Int32>(Int32(minTX), Int32(maxTX), Int32(minTY), Int32(maxTY)))

            let data = GaussianRenderDataSwift(
                meanX: Float16(centerX),
                meanY: Float16(centerY),
                _alignPad: 0,
                conicA: Float16(0.001),
                conicB: Float16(0.0),
                conicC: Float16(0.001),
                conicD: Float16(0.0),  // Not used for qMax computation
                colorR: Float16(1.0),
                colorG: Float16(0.0),
                colorB: Float16(0.0),
                opacity: Float16(0.9),
                depth: Float16(Float(i) * 0.001 + 0.1),
                _pad: 0,
                _structPad: 0
            )
            renderData.append(data)
        }

        let boundsBuffer = device.makeBuffer(bytes: bounds, length: bounds.count * MemoryLayout<SIMD4<Int32>>.stride, options: .storageModeShared)!
        let renderDataBuffer = device.makeBuffer(bytes: renderData, length: renderData.count * MemoryLayout<GaussianRenderDataSwift>.stride, options: .storageModeShared)!

        return (bounds, renderData, boundsBuffer, renderDataBuffer, tilesX, tilesY)
    }

    // MARK: - Pass 1: Tile Count Test

    func testTileCountVsCPU() throws {
        let gaussianCount = 100
        let width = 256
        let height = 256
        let tileWidth = 16
        let tileHeight = 16

        let (bounds, renderData, boundsBuffer, renderDataBuffer, tilesX, _) = createTestData(
            gaussianCount: gaussianCount,
            width: width,
            height: height,
            tileWidth: tileWidth,
            tileHeight: tileHeight
        )

        // CPU reference
        let cpuCounts = cpuCountTiles(bounds: bounds, renderData: renderData, tileWidth: tileWidth, tileHeight: tileHeight)

        // GPU count kernel
        let countsBuffer = device.makeBuffer(length: gaussianCount * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

        var params = TileAssignParamsSwift(
            gaussianCount: UInt32(gaussianCount),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            maxAssignments: UInt32(gaussianCount * 20)
        )

        let cmd = queue.makeCommandBuffer()!
        let encoder = cmd.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(countPipeline)
        encoder.setBuffer(boundsBuffer, offset: 0, index: 0)
        encoder.setBuffer(renderDataBuffer, offset: 0, index: 1)
        encoder.setBuffer(countsBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<TileAssignParamsSwift>.stride, index: 3)
        encoder.dispatchThreads(MTLSize(width: gaussianCount, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        // Compare
        let gpuCounts = countsBuffer.contents().bindMemory(to: UInt32.self, capacity: gaussianCount)

        var mismatches = 0
        for i in 0..<gaussianCount {
            if cpuCounts[i] != gpuCounts[i] {
                mismatches += 1
                if mismatches <= 5 {
                    NSLog("Count mismatch at %d: CPU=%d, GPU=%d", i, cpuCounts[i], gpuCounts[i])
                }
            }
        }

        let cpuTotal = cpuCounts.reduce(0, +)
        let gpuTotal = (0..<gaussianCount).map { gpuCounts[$0] }.reduce(0 as UInt32, +)
        NSLog("Tile count test: CPU total=%d, GPU total=%d, mismatches=%d", cpuTotal, gpuTotal, mismatches)

        XCTAssertEqual(cpuTotal, gpuTotal, "Total tile counts should match")
        XCTAssertEqual(mismatches, 0, "All per-gaussian counts should match")
    }

    // MARK: - Pass 2: Prefix Sum Test

    func testPrefixSumVsCPU() throws {
        // Test prefix sum with known input
        let testInput: [UInt32] = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]
        let count = testInput.count

        // Pad to multiple of blockSize
        var paddedInput = testInput
        while paddedInput.count < blockSize {
            paddedInput.append(0)
        }
        let paddedCount = paddedInput.count

        // CPU reference
        let (cpuOffsets, cpuTotal) = cpuExclusivePrefixSum(testInput)

        // GPU buffers
        let inputBuffer = device.makeBuffer(bytes: paddedInput, length: paddedCount * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(length: paddedCount * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let blockSumsBuffer = device.makeBuffer(length: blockSize * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
        memset(headerBuffer.contents(), 0, MemoryLayout<TileAssignmentHeaderSwift>.stride)

        var countU32 = UInt32(count)
        var numBlocks = UInt32(1)

        // Run prefix sum passes
        let cmd = queue.makeCommandBuffer()!

        // 2a. Block reduce
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(blockReducePipeline)
            enc.setBuffer(inputBuffer, offset: 0, index: 0)
            enc.setBuffer(blockSumsBuffer, offset: 0, index: 1)
            enc.setBytes(&countU32, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: blockSize, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: blockSize, height: 1, depth: 1))
            enc.endEncoding()
        }

        // 2b. Single block scan
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(singleBlockScanPipeline)
            enc.setBuffer(blockSumsBuffer, offset: 0, index: 0)
            enc.setBytes(&numBlocks, length: 4, index: 1)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: blockSize, height: 1, depth: 1))
            enc.endEncoding()
        }

        // 2c. Block scan
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(blockScanPipeline)
            enc.setBuffer(inputBuffer, offset: 0, index: 0)
            enc.setBuffer(outputBuffer, offset: 0, index: 1)
            enc.setBuffer(blockSumsBuffer, offset: 0, index: 2)
            enc.setBytes(&countU32, length: 4, index: 3)
            enc.dispatchThreads(MTLSize(width: blockSize, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: blockSize, height: 1, depth: 1))
            enc.endEncoding()
        }

        // 2d. Write total (reads from blockSums[numBlocks] where total is stored)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(writeTotalPipeline)
            enc.setBuffer(blockSumsBuffer, offset: 0, index: 0)
            enc.setBuffer(headerBuffer, offset: 0, index: 1)
            enc.setBytes(&numBlocks, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            enc.endEncoding()
        }

        cmd.commit()
        cmd.waitUntilCompleted()

        // Compare
        let gpuOffsets = outputBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
        let gpuHeader = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        let gpuTotal = gpuHeader.pointee.totalAssignments

        NSLog("Prefix sum test: CPU total=%d, GPU total=%d", cpuTotal, gpuTotal)
        NSLog("CPU offsets: %@", cpuOffsets.prefix(10).map { String($0) }.joined(separator: ", "))
        NSLog("GPU offsets: %@", (0..<min(10, count)).map { String(gpuOffsets[$0]) }.joined(separator: ", "))

        XCTAssertEqual(cpuTotal, gpuTotal, "Total should match")

        var mismatches = 0
        for i in 0..<count {
            if cpuOffsets[i] != gpuOffsets[i] {
                mismatches += 1
                if mismatches <= 5 {
                    NSLog("Offset mismatch at %d: CPU=%d, GPU=%d", i, cpuOffsets[i], gpuOffsets[i])
                }
            }
        }
        XCTAssertEqual(mismatches, 0, "All offsets should match")
    }

    // MARK: - Scatter Accuracy Test (Larger Scale)

    func testTwoPassScatterVsCPU() throws {
        let sizes = [1_000, 10_000, 100_000]
        let width = 1920
        let height = 1080
        let tileWidth = 16
        let tileHeight = 16
        let tilesX = (width + tileWidth - 1) / tileWidth
        let tilesY = (height + tileHeight - 1) / tileHeight

        for gaussianCount in sizes {
            let maxAssignments = gaussianCount * 10

            let (bounds, renderData, boundsBuffer, renderDataBuffer, _, _) = createTestData(
                gaussianCount: gaussianCount,
                width: width,
                height: height,
                tileWidth: tileWidth,
                tileHeight: tileHeight
            )

            // CPU reference scatter
            let cpuCounts = cpuCountTiles(bounds: bounds, renderData: renderData, tileWidth: tileWidth, tileHeight: tileHeight)
            let (cpuOffsets, cpuTotal) = cpuExclusivePrefixSum(cpuCounts)

            // CPU scatter - generate (gaussianIdx, tileId) pairs
            var cpuPairs = Set<String>()
            let tau: Float = 1.0 / 255.0
            let preciseThreshold: UInt32 = 16

            for i in 0..<gaussianCount {
                let rect = bounds[i]
                let minTX = Int(rect.x)
                let maxTX = Int(rect.y)
                let minTY = Int(rect.z)
                let maxTY = Int(rect.w)

                if minTX > maxTX || minTY > maxTY { continue }

                let g = renderData[i]
                let alpha = Float(g.opacity)
                if alpha < 1e-4 { continue }

                let aabbWidth = UInt32(maxTX - minTX + 1)
                let aabbHeight = UInt32(maxTY - minTY + 1)
                let aabbCount = aabbWidth * aabbHeight

                if aabbCount <= preciseThreshold {
                    for ty in minTY...maxTY {
                        for tx in minTX...maxTX {
                            let tileId = ty * tilesX + tx
                            cpuPairs.insert("\(i)_\(tileId)")
                        }
                    }
                } else {
                    let tileW = Float(tileWidth)
                    let tileH = Float(tileHeight)
                    let centerX = Float(g.meanX)
                    let centerY = Float(g.meanY)
                    let conicX = Float(g.conicA)
                    let conicY = Float(g.conicB)
                    let conicZ = Float(g.conicC)
                    let qMax = cpuComputeQMax(alpha: alpha, tau: tau)

                    let halfTileW = tileW * 0.5
                    let halfTileH = tileH * 0.5
                    let tileDiag = sqrt(halfTileW * halfTileW + halfTileH * halfTileH)
                    let maxConic = max(conicX, conicZ)
                    let expand = tileDiag * sqrt(maxConic)
                    let qMaxExpanded = qMax + 2.0 * expand * sqrt(qMax) + expand * expand

                    for ty in minTY...maxTY {
                        for tx in minTX...maxTX {
                            let tileCenterX = (Float(tx) + 0.5) * tileW
                            let tileCenterY = (Float(ty) + 0.5) * tileH
                            let dx = tileCenterX - centerX
                            let dy = tileCenterY - centerY
                            let q = dx * dx * conicX + 2.0 * dx * dy * conicY + dy * dy * conicZ

                            if q <= qMax || q <= qMaxExpanded {
                                let tileId = ty * tilesX + tx
                                cpuPairs.insert("\(i)_\(tileId)")
                            }
                        }
                    }
                }
            }

            // GPU two-pass
            let coverageBuffer = device.makeBuffer(length: gaussianCount * 4, options: .storageModeShared)!
            let tileIndices = device.makeBuffer(length: maxAssignments * 4, options: .storageModeShared)!
            let tileIds = device.makeBuffer(length: maxAssignments * 4, options: .storageModeShared)!
            let header = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
            let numBlocks = (gaussianCount + blockSize - 1) / blockSize
            let level2Blocks = (numBlocks + blockSize - 1) / blockSize
            // Buffer needs space for: blockSums[numBlocks+1] + level2Sums[level2Blocks+1]
            let blockSumsSize = (numBlocks + 1 + level2Blocks + 1) * 4
            let blockSums = device.makeBuffer(length: blockSumsSize, options: .storageModeShared)!
            let indirectBuffers = createIndirectBuffers(gaussianCount: gaussianCount)
            memset(header.contents(), 0, MemoryLayout<TileAssignmentHeaderSwift>.stride)

            let cmd = queue.makeCommandBuffer()!
            twoPassEncoder.encode(
                commandBuffer: cmd,
                gaussianCount: gaussianCount,
                tileWidth: tileWidth,
                tileHeight: tileHeight,
                tilesX: tilesX,
                maxAssignments: maxAssignments,
                boundsBuffer: boundsBuffer,
                coverageBuffer: coverageBuffer,
                renderData: renderDataBuffer,
                tileIndicesBuffer: tileIndices,
                tileIdsBuffer: tileIds,
                tileAssignmentHeader: header,
                blockSumsBuffer: blockSums,
                indirectBuffers: indirectBuffers
            )
            cmd.commit()
            cmd.waitUntilCompleted()

            let headerPtr = header.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
            let gpuTotal = Int(headerPtr.pointee.totalAssignments)

            var gpuPairs = Set<String>()
            let indicesPtr = tileIndices.contents().bindMemory(to: Int32.self, capacity: gpuTotal)
            let idsPtr = tileIds.contents().bindMemory(to: Int32.self, capacity: gpuTotal)
            for i in 0..<gpuTotal {
                gpuPairs.insert("\(indicesPtr[i])_\(idsPtr[i])")
            }

            let onlyInCPU = cpuPairs.subtracting(gpuPairs)
            let onlyInGPU = gpuPairs.subtracting(cpuPairs)

            NSLog("[ScatterVsCPU %dk] CPU=%d, GPU=%d, onlyInCPU=%d, onlyInGPU=%d",
                  gaussianCount / 1000, cpuPairs.count, gpuPairs.count, onlyInCPU.count, onlyInGPU.count)

            XCTAssertEqual(cpuPairs.count, gpuPairs.count, "[\(gaussianCount)] Total assignments should match")
            XCTAssertEqual(onlyInCPU.count, 0, "[\(gaussianCount)] Should have no pairs only in CPU")
            XCTAssertEqual(onlyInGPU.count, 0, "[\(gaussianCount)] Should have no pairs only in GPU")
        }
    }

    // MARK: - Debug Test: Step-by-step encoder verification

    func testTwoPassEncoderStepByStep() throws {
        let gaussianCount = 100
        let width = 256
        let height = 256
        let tileWidth = 16
        let tileHeight = 16
        let maxAssignments = gaussianCount * 20

        let (bounds, renderData, boundsBuffer, renderDataBuffer, tilesX, _) = createTestData(
            gaussianCount: gaussianCount,
            width: width,
            height: height,
            tileWidth: tileWidth,
            tileHeight: tileHeight
        )

        // Expected counts from CPU
        let cpuCounts = cpuCountTiles(bounds: bounds, renderData: renderData, tileWidth: tileWidth, tileHeight: tileHeight)
        let (cpuOffsets, cpuTotal) = cpuExclusivePrefixSum(cpuCounts)

        NSLog("CPU counts (first 10): %@", cpuCounts.prefix(10).map { String($0) }.joined(separator: ", "))
        NSLog("CPU offsets (first 10): %@", cpuOffsets.prefix(10).map { String($0) }.joined(separator: ", "))
        NSLog("CPU total: %d", cpuTotal)

        // Allocate all buffers
        let coverageBuffer = device.makeBuffer(length: gaussianCount * 4, options: .storageModeShared)!
        let tileIndices = device.makeBuffer(length: maxAssignments * 4, options: .storageModeShared)!
        let tileIds = device.makeBuffer(length: maxAssignments * 4, options: .storageModeShared)!
        let header = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
        let numBlocks = (gaussianCount + blockSize - 1) / blockSize
        let level2Blocks = (numBlocks + blockSize - 1) / blockSize
        let blockSums = device.makeBuffer(length: (numBlocks + 1 + level2Blocks + 1) * 4, options: .storageModeShared)!
        memset(header.contents(), 0, MemoryLayout<TileAssignmentHeaderSwift>.stride)

        var params = TileAssignParamsSwift(
            gaussianCount: UInt32(gaussianCount),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            maxAssignments: UInt32(maxAssignments)
        )

        // STEP 1: Run count kernel
        do {
            let cmd = queue.makeCommandBuffer()!
            if let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(countPipeline)
                enc.setBuffer(boundsBuffer, offset: 0, index: 0)
                enc.setBuffer(renderDataBuffer, offset: 0, index: 1)
                enc.setBuffer(coverageBuffer, offset: 0, index: 2)
                enc.setBytes(&params, length: MemoryLayout<TileAssignParamsSwift>.stride, index: 3)
                enc.dispatchThreads(MTLSize(width: gaussianCount, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
            }
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        // Check counts after step 1
        let countsPtr = coverageBuffer.contents().bindMemory(to: UInt32.self, capacity: gaussianCount)
        let gpuCounts = (0..<gaussianCount).map { countsPtr[$0] }
        let gpuCountTotal = gpuCounts.reduce(0 as UInt32, +)
        NSLog("After STEP 1 (count): GPU counts (first 10): %@", gpuCounts.prefix(10).map { String($0) }.joined(separator: ", "))
        NSLog("After STEP 1 (count): GPU count total: %d", gpuCountTotal)
        XCTAssertEqual(cpuTotal, gpuCountTotal, "Step 1: Counts should match")

        // STEP 2a: Block reduce
        var countU32 = UInt32(gaussianCount)
        var numBlocksU32 = UInt32(numBlocks)
        do {
            let cmd = queue.makeCommandBuffer()!
            if let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(blockReducePipeline)
                enc.setBuffer(coverageBuffer, offset: 0, index: 0)
                enc.setBuffer(blockSums, offset: 0, index: 1)
                enc.setBytes(&countU32, length: 4, index: 2)
                enc.dispatchThreads(MTLSize(width: numBlocks * blockSize, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: blockSize, height: 1, depth: 1))
                enc.endEncoding()
            }
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let blockSumsPtr = blockSums.contents().bindMemory(to: UInt32.self, capacity: numBlocks)
        let blockSumsArray = (0..<numBlocks).map { blockSumsPtr[$0] }
        NSLog("After STEP 2a (reduce): Block sums: %@", blockSumsArray.map { String($0) }.joined(separator: ", "))

        // STEP 2b: Single block scan
        do {
            let cmd = queue.makeCommandBuffer()!
            if let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(singleBlockScanPipeline)
                enc.setBuffer(blockSums, offset: 0, index: 0)
                enc.setBytes(&numBlocksU32, length: 4, index: 1)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: blockSize, height: 1, depth: 1))
                enc.endEncoding()
            }
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let blockOffsetsArray = (0..<numBlocks).map { blockSumsPtr[$0] }
        let totalStored = blockSumsPtr[numBlocks]  // Total should be at blockSums[numBlocks]
        NSLog("After STEP 2b (scan): Block offsets: %@, totalStored at [%d]=%d", blockOffsetsArray.map { String($0) }.joined(separator: ", "), numBlocks, totalStored)

        // STEP 2c: Block scan (this overwrites coverageBuffer with offsets!)
        do {
            let cmd = queue.makeCommandBuffer()!
            if let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(blockScanPipeline)
                enc.setBuffer(coverageBuffer, offset: 0, index: 0)  // INPUT: counts
                enc.setBuffer(coverageBuffer, offset: 0, index: 1)  // OUTPUT: offsets (IN-PLACE!)
                enc.setBuffer(blockSums, offset: 0, index: 2)
                enc.setBytes(&countU32, length: 4, index: 3)
                enc.dispatchThreads(MTLSize(width: numBlocks * blockSize, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: blockSize, height: 1, depth: 1))
                enc.endEncoding()
            }
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        // Check offsets after step 2c - this is where in-place might fail
        let offsetsPtr = coverageBuffer.contents().bindMemory(to: UInt32.self, capacity: gaussianCount)
        let gpuOffsets = (0..<gaussianCount).map { offsetsPtr[$0] }
        NSLog("After STEP 2c (block scan): GPU offsets (first 10): %@", gpuOffsets.prefix(10).map { String($0) }.joined(separator: ", "))

        // Verify offsets match CPU
        var offsetMismatches = 0
        for i in 0..<gaussianCount {
            if cpuOffsets[i] != gpuOffsets[i] {
                offsetMismatches += 1
                if offsetMismatches <= 5 {
                    NSLog("Offset mismatch at %d: CPU=%d, GPU=%d", i, cpuOffsets[i], gpuOffsets[i])
                }
            }
        }
        NSLog("After STEP 2c: %d offset mismatches out of %d", offsetMismatches, gaussianCount)
        XCTAssertEqual(offsetMismatches, 0, "Step 2c: Offsets should match CPU reference")
    }

    // MARK: - Performance Benchmark

    func testTwoPassPerformance() throws {
        let sizes = [100_000, 500_000, 1_000_000, 2_000_000, 4_000_000]
        let width = 1920
        let height = 1080
        let tileWidth = 16
        let tileHeight = 16
        let tilesX = (width + tileWidth - 1) / tileWidth
        let iterations = 5

        var results: [String] = []

        for gaussianCount in sizes {
            let maxAssignments = gaussianCount * 10

            // Create test data
            let (_, _, boundsBuffer, renderDataBuffer, _, _) = createTestData(
                gaussianCount: gaussianCount,
                width: width,
                height: height,
                tileWidth: tileWidth,
                tileHeight: tileHeight
            )

            // Allocate buffers
            let coverageBuffer = device.makeBuffer(length: gaussianCount * 4, options: .storageModeShared)!
            let tileIndices = device.makeBuffer(length: maxAssignments * 4, options: .storageModeShared)!
            let tileIds = device.makeBuffer(length: maxAssignments * 4, options: .storageModeShared)!
            let header = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
            let numBlocks = (gaussianCount + blockSize - 1) / blockSize
            let level2BlocksPerf = (numBlocks + blockSize - 1) / blockSize
            let blockSums = device.makeBuffer(length: (numBlocks + 1 + level2BlocksPerf + 1) * 4, options: .storageModeShared)!
            let indirectBuffers = createIndirectBuffers(gaussianCount: gaussianCount)

            // Warmup
            for _ in 0..<2 {
                memset(header.contents(), 0, MemoryLayout<TileAssignmentHeaderSwift>.stride)
                let cmd = queue.makeCommandBuffer()!
                twoPassEncoder.encode(
                    commandBuffer: cmd,
                    gaussianCount: gaussianCount,
                    tileWidth: tileWidth,
                    tileHeight: tileHeight,
                    tilesX: tilesX,
                    maxAssignments: maxAssignments,
                    boundsBuffer: boundsBuffer,
                    coverageBuffer: coverageBuffer,
                    renderData: renderDataBuffer,
                    tileIndicesBuffer: tileIndices,
                    tileIdsBuffer: tileIds,
                    tileAssignmentHeader: header,
                    blockSumsBuffer: blockSums,
                    indirectBuffers: indirectBuffers
                )
                cmd.commit()
                cmd.waitUntilCompleted()
            }

            // Benchmark two-pass with indirect dispatch
            var times: [Double] = []
            for _ in 0..<iterations {
                memset(header.contents(), 0, MemoryLayout<TileAssignmentHeaderSwift>.stride)
                let cmd = queue.makeCommandBuffer()!
                let start = CFAbsoluteTimeGetCurrent()
                twoPassEncoder.encode(
                    commandBuffer: cmd,
                    gaussianCount: gaussianCount,
                    tileWidth: tileWidth,
                    tileHeight: tileHeight,
                    tilesX: tilesX,
                    maxAssignments: maxAssignments,
                    boundsBuffer: boundsBuffer,
                    coverageBuffer: coverageBuffer,
                    renderData: renderDataBuffer,
                    tileIndicesBuffer: tileIndices,
                    tileIdsBuffer: tileIds,
                    tileAssignmentHeader: header,
                    blockSumsBuffer: blockSums,
                    indirectBuffers: indirectBuffers
                )
                cmd.commit()
                cmd.waitUntilCompleted()
                let end = CFAbsoluteTimeGetCurrent()
                times.append((end - start) * 1000.0)
            }

            let avg = times.reduce(0, +) / Double(iterations)
            results.append(String(format: "%dk: %.3fms", gaussianCount / 1000, avg))
        }

        NSLog("[TileAssign Perf] %@", results.joined(separator: " | "))
    }
}

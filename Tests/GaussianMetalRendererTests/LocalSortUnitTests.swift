@testable import GaussianMetalRenderer
import Metal
import simd
import XCTest

/// Unit tests for Local pipeline stages
final class LocalUnitTests: XCTestCase {
    var device: MTLDevice!
    var library: MTLLibrary!
    var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        self.device = MTLCreateSystemDefaultDevice()!
        self.queue = self.device.makeCommandQueue()!

        // Load Local library
        guard let libraryURL = Bundle.module.url(forResource: "LocalShaders", withExtension: "metallib"),
              let lib = try? device.makeLibrary(URL: libraryURL)
        else {
            XCTFail("Failed to load LocalShaders.metallib")
            return
        }
        self.library = lib
    }

    // MARK: - Prefix Scan Tests

    func testPrefixScanCorrectness() throws {
        let prefixScanEncoder = try LocalPrefixScanEncoder(library: library, device: device)

        // Test with 64 tiles (4x16 grid)
        let tileCount = 64
        var tileCounts = [UInt32](repeating: 0, count: tileCount)

        // Fill with known values: tile i has i gaussians
        for i in 0 ..< tileCount {
            tileCounts[i] = UInt32(i + 1)
        }

        // CPU reference: exclusive prefix sum
        var expectedOffsets = [UInt32](repeating: 0, count: tileCount)
        var sum: UInt32 = 0
        for i in 0 ..< tileCount {
            expectedOffsets[i] = sum
            sum += tileCounts[i]
        }
        let expectedTotal = sum

        // GPU buffers
        let tileCountsBuffer = self.device.makeBuffer(bytes: tileCounts, length: tileCount * 4, options: .storageModeShared)!
        let tileOffsetsBuffer = self.device.makeBuffer(length: tileCount * 4, options: .storageModeShared)!
        let partialSumsBuffer = self.device.makeBuffer(length: 1024 * 4, options: .storageModeShared)!
        let activeTileIndicesBuffer = self.device.makeBuffer(length: tileCount * 4, options: .storageModeShared)!
        let activeTileCountBuffer = self.device.makeBuffer(length: 4, options: .storageModeShared)!

        // Encode
        let cb = self.queue.makeCommandBuffer()!
        prefixScanEncoder.encode(
            commandBuffer: cb,
            tileCounts: tileCountsBuffer,
            tileOffsets: tileOffsetsBuffer,
            partialSums: partialSumsBuffer,
            tileCount: tileCount,
            activeTileIndices: activeTileIndicesBuffer,
            activeTileCount: activeTileCountBuffer
        )
        cb.commit()
        cb.waitUntilCompleted()

        // Verify offsets
        let gpuOffsets = tileOffsetsBuffer.contents().bindMemory(to: UInt32.self, capacity: tileCount)
        for i in 0 ..< tileCount {
            XCTAssertEqual(gpuOffsets[i], expectedOffsets[i], "Offset mismatch at tile \(i): got \(gpuOffsets[i]), expected \(expectedOffsets[i])")
        }

        // Verify last offset + last count = total
        let lastOffset = gpuOffsets[tileCount - 1]
        let lastCount = UInt32(tileCount) // Last tile has tileCount gaussians
        XCTAssertEqual(lastOffset + lastCount, expectedTotal, "Total assignment count mismatch")
    }

    func testPrefixScanLargeScale() throws {
        let prefixScanEncoder = try LocalPrefixScanEncoder(library: library, device: device)

        // Test with 4096 tiles (64x64 grid)
        let tileCount = 4096
        var tileCounts = [UInt32](repeating: 0, count: tileCount)

        srand48(42)
        for i in 0 ..< tileCount {
            tileCounts[i] = UInt32(drand48() * 100)
        }

        // CPU reference
        var expectedOffsets = [UInt32](repeating: 0, count: tileCount)
        var sum: UInt32 = 0
        for i in 0 ..< tileCount {
            expectedOffsets[i] = sum
            sum += tileCounts[i]
        }

        // GPU
        let tileCountsBuffer = self.device.makeBuffer(bytes: tileCounts, length: tileCount * 4, options: .storageModeShared)!
        let tileOffsetsBuffer = self.device.makeBuffer(length: tileCount * 4, options: .storageModeShared)!
        let partialSumsBuffer = self.device.makeBuffer(length: 4096 * 4, options: .storageModeShared)!
        let activeTileIndicesBuffer = self.device.makeBuffer(length: tileCount * 4, options: .storageModeShared)!
        let activeTileCountBuffer = self.device.makeBuffer(length: 4, options: .storageModeShared)!

        let cb = self.queue.makeCommandBuffer()!
        prefixScanEncoder.encode(
            commandBuffer: cb,
            tileCounts: tileCountsBuffer,
            tileOffsets: tileOffsetsBuffer,
            partialSums: partialSumsBuffer,
            tileCount: tileCount,
            activeTileIndices: activeTileIndicesBuffer,
            activeTileCount: activeTileCountBuffer
        )
        cb.commit()
        cb.waitUntilCompleted()

        // Verify
        let gpuOffsets = tileOffsetsBuffer.contents().bindMemory(to: UInt32.self, capacity: tileCount)
        for i in 0 ..< tileCount {
            XCTAssertEqual(gpuOffsets[i], expectedOffsets[i], "Offset mismatch at tile \(i)")
        }
    }

    // MARK: - Per-Tile Sort Tests

    func testPerTileSortCorrectness() throws {
        let sortEncoder = try LocalSortEncoder(library: library, device: device)

        // Test 4 tiles with known data
        let tileCount = 4
        let maxPerTile = 64
        let totalAssignments = tileCount * maxPerTile

        // Generate sort keys: [depth:32] for each tile
        // Data layout: tile i's data at i * maxPerTile
        var sortKeys = [UInt32](repeating: 0, count: totalAssignments)
        var sortIndices = [Int32](repeating: 0, count: totalAssignments)
        let tileCounts = [UInt32](repeating: UInt32(maxPerTile), count: tileCount)

        // Fill each tile with random depths (fixed layout: tile * maxPerTile)
        srand48(42)
        for tile in 0 ..< tileCount {
            let offset = tile * maxPerTile
            for j in 0 ..< maxPerTile {
                let depth = UInt32(drand48() * 65535.0)
                sortKeys[offset + j] = depth
                sortIndices[offset + j] = Int32(offset + j)
            }
        }

        // CPU reference: sort each tile
        var expectedKeys = sortKeys
        for tile in 0 ..< tileCount {
            let offset = tile * maxPerTile
            let tileSlice = Array(expectedKeys[offset ..< (offset + maxPerTile)])
            let sorted = tileSlice.sorted()
            for j in 0 ..< maxPerTile {
                expectedKeys[offset + j] = sorted[j]
            }
        }

        // GPU buffers
        let sortKeysBuffer = self.device.makeBuffer(bytes: sortKeys, length: totalAssignments * 4, options: .storageModeShared)!
        let sortIndicesBuffer = self.device.makeBuffer(bytes: sortIndices, length: totalAssignments * 4, options: .storageModeShared)!
        let tileCountsBuffer = self.device.makeBuffer(bytes: tileCounts, length: tileCount * 4, options: .storageModeShared)!

        let cb = self.queue.makeCommandBuffer()!
        sortEncoder.encode(
            commandBuffer: cb,
            sortKeys: sortKeysBuffer,
            sortIndices: sortIndicesBuffer,
            tileCounts: tileCountsBuffer,
            maxPerTile: maxPerTile,
            tileCount: tileCount
        )
        cb.commit()
        cb.waitUntilCompleted()

        // Verify each tile is sorted
        let gpuKeys = sortKeysBuffer.contents().bindMemory(to: UInt32.self, capacity: totalAssignments)
        for tile in 0 ..< tileCount {
            let offset = tile * maxPerTile
            for j in 0 ..< (maxPerTile - 1) {
                XCTAssertLessThanOrEqual(gpuKeys[offset + j], gpuKeys[offset + j + 1],
                                         "Tile \(tile) not sorted at position \(j): \(gpuKeys[offset + j]) > \(gpuKeys[offset + j + 1])")
            }
        }

        // Verify keys match expected
        for i in 0 ..< totalAssignments {
            XCTAssertEqual(gpuKeys[i], expectedKeys[i], "Key mismatch at \(i)")
        }
    }

    func testPerTileSortVariableCounts() throws {
        let sortEncoder = try LocalSortEncoder(library: library, device: device)

        // Test with variable tile counts (fixed layout: maxPerTile per tile)
        let tileCount = 8
        let maxPerTile = 128 // Fixed layout - each tile gets maxPerTile slots
        let tileCounts: [UInt32] = [10, 50, 5, 100, 25, 0, 75, 30]
        let totalSize = tileCount * maxPerTile

        // Generate data using fixed layout
        var sortKeys = [UInt32](repeating: 0xFFFF_FFFF, count: totalSize)
        var sortIndices = [Int32](repeating: 0, count: totalSize)

        srand48(123)
        for tile in 0 ..< tileCount {
            let offset = tile * maxPerTile
            let count = Int(tileCounts[tile])
            for j in 0 ..< count {
                sortKeys[offset + j] = UInt32(drand48() * 65535.0)
                sortIndices[offset + j] = Int32(offset + j)
            }
        }

        // GPU buffers
        let sortKeysBuffer = self.device.makeBuffer(bytes: sortKeys, length: totalSize * 4, options: .storageModeShared)!
        let sortIndicesBuffer = self.device.makeBuffer(bytes: sortIndices, length: totalSize * 4, options: .storageModeShared)!
        let tileCountsBuffer = self.device.makeBuffer(bytes: tileCounts, length: tileCount * 4, options: .storageModeShared)!

        let cb = self.queue.makeCommandBuffer()!
        sortEncoder.encode(
            commandBuffer: cb,
            sortKeys: sortKeysBuffer,
            sortIndices: sortIndicesBuffer,
            tileCounts: tileCountsBuffer,
            maxPerTile: maxPerTile,
            tileCount: tileCount
        )
        cb.commit()
        cb.waitUntilCompleted()

        // Verify each tile is sorted
        let gpuKeys = sortKeysBuffer.contents().bindMemory(to: UInt32.self, capacity: totalSize)
        for tile in 0 ..< tileCount {
            let tileOffset = tile * maxPerTile // Fixed layout
            let count = Int(tileCounts[tile])
            guard count > 1 else { continue }
            for j in 0 ..< (count - 1) {
                XCTAssertLessThanOrEqual(gpuKeys[tileOffset + j], gpuKeys[tileOffset + j + 1],
                                         "Tile \(tile) not sorted at position \(j)")
            }
        }
    }
}

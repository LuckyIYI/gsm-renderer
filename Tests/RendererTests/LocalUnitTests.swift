import Metal
@testable import Renderer
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

        let tileCount = 64
        var tileCounts = [UInt32](repeating: 0, count: tileCount)

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

        let tileCountsBuffer = self.device.makeBuffer(bytes: tileCounts, length: tileCount * 4, options: .storageModeShared)!
        let tileOffsetsBuffer = self.device.makeBuffer(length: tileCount * 4, options: .storageModeShared)!
        let partialSumsBuffer = self.device.makeBuffer(length: 1024 * 4, options: .storageModeShared)!
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

        let gpuOffsets = tileOffsetsBuffer.contents().bindMemory(to: UInt32.self, capacity: tileCount)
        for i in 0 ..< tileCount {
            XCTAssertEqual(gpuOffsets[i], expectedOffsets[i], "Offset mismatch at tile \(i): got \(gpuOffsets[i]), expected \(expectedOffsets[i])")
        }

        let lastOffset = gpuOffsets[tileCount - 1]
        let lastCount = UInt32(tileCount) // Last tile has tileCount gaussians
        XCTAssertEqual(lastOffset + lastCount, expectedTotal, "Total assignment count mismatch")
    }

    func testPrefixScanLargeScale() throws {
        let prefixScanEncoder = try LocalPrefixScanEncoder(library: library, device: device)

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

        let gpuOffsets = tileOffsetsBuffer.contents().bindMemory(to: UInt32.self, capacity: tileCount)
        for i in 0 ..< tileCount {
            XCTAssertEqual(gpuOffsets[i], expectedOffsets[i], "Offset mismatch at tile \(i)")
        }
    }

    // MARK: - Per-Tile Sort Tests (16-bit)

    func testPerTileSortCorrectness() throws {
        let sortEncoder = try LocalSortEncoder(library: library, device: device)

        let tileCount = 4
        let maxPerTile = 64
        let totalAssignments = tileCount * maxPerTile

        var depthKeys16 = [UInt16](repeating: 0, count: totalAssignments)
        var globalIndices = [UInt32](repeating: 0, count: totalAssignments)
        let tileCounts = [UInt32](repeating: UInt32(maxPerTile), count: tileCount)

        srand48(42)
        for tile in 0 ..< tileCount {
            let offset = tile * maxPerTile
            for j in 0 ..< maxPerTile {
                depthKeys16[offset + j] = UInt16(drand48() * 65535.0)
                globalIndices[offset + j] = UInt32(offset + j)
            }
        }

        // CPU reference: compute expected sorted order
        // The 16-bit sort creates 32-bit keys as (depth16 << 16) | localIdx
        var expectedSortedIdx = [UInt16](repeating: 0, count: totalAssignments)
        for tile in 0 ..< tileCount {
            let offset = tile * maxPerTile
            var keysWithIdx = [(key: UInt32, localIdx: UInt16)]()
            for j in 0 ..< maxPerTile {
                let key = (UInt32(depthKeys16[offset + j]) << 16) | UInt32(j)
                keysWithIdx.append((key, UInt16(j)))
            }
            keysWithIdx.sort { $0.key < $1.key }
            for j in 0 ..< maxPerTile {
                expectedSortedIdx[offset + j] = keysWithIdx[j].localIdx
            }
        }

        let depthKeys16Buffer = self.device.makeBuffer(bytes: depthKeys16, length: totalAssignments * 2, options: .storageModeShared)!
        let globalIndicesBuffer = self.device.makeBuffer(bytes: globalIndices, length: totalAssignments * 4, options: .storageModeShared)!
        let sortedLocalIdxBuffer = self.device.makeBuffer(length: totalAssignments * 2, options: .storageModeShared)!
        let tileCountsBuffer = self.device.makeBuffer(bytes: tileCounts, length: tileCount * 4, options: .storageModeShared)!

        let cb = self.queue.makeCommandBuffer()!
        sortEncoder.encode16(
            commandBuffer: cb,
            depthKeys16: depthKeys16Buffer,
            globalIndices: globalIndicesBuffer,
            sortedLocalIdx: sortedLocalIdxBuffer,
            tileCounts: tileCountsBuffer,
            maxPerTile: maxPerTile,
            tileCount: tileCount
        )
        cb.commit()
        cb.waitUntilCompleted()

        let gpuSortedIdx = sortedLocalIdxBuffer.contents().bindMemory(to: UInt16.self, capacity: totalAssignments)
        for i in 0 ..< totalAssignments {
            XCTAssertEqual(gpuSortedIdx[i], expectedSortedIdx[i], "Sorted index mismatch at \(i)")
        }
    }

    func testPerTileSortVariableCounts() throws {
        let sortEncoder = try LocalSortEncoder(library: library, device: device)

        let tileCount = 8
        let maxPerTile = 128
        let tileCounts: [UInt32] = [10, 50, 5, 100, 25, 0, 75, 30]
        let totalSize = tileCount * maxPerTile

        var depthKeys16 = [UInt16](repeating: 0xFFFF, count: totalSize)
        var globalIndices = [UInt32](repeating: 0, count: totalSize)

        srand48(123)
        for tile in 0 ..< tileCount {
            let offset = tile * maxPerTile
            let count = Int(tileCounts[tile])
            for j in 0 ..< count {
                depthKeys16[offset + j] = UInt16(drand48() * 65535.0)
                globalIndices[offset + j] = UInt32(offset + j)
            }
        }

        let depthKeys16Buffer = self.device.makeBuffer(bytes: depthKeys16, length: totalSize * 2, options: .storageModeShared)!
        let globalIndicesBuffer = self.device.makeBuffer(bytes: globalIndices, length: totalSize * 4, options: .storageModeShared)!
        let sortedLocalIdxBuffer = self.device.makeBuffer(length: totalSize * 2, options: .storageModeShared)!
        let tileCountsBuffer = self.device.makeBuffer(bytes: tileCounts, length: tileCount * 4, options: .storageModeShared)!

        let cb = self.queue.makeCommandBuffer()!
        sortEncoder.encode16(
            commandBuffer: cb,
            depthKeys16: depthKeys16Buffer,
            globalIndices: globalIndicesBuffer,
            sortedLocalIdx: sortedLocalIdxBuffer,
            tileCounts: tileCountsBuffer,
            maxPerTile: maxPerTile,
            tileCount: tileCount
        )
        cb.commit()
        cb.waitUntilCompleted()

        let gpuSortedIdx = sortedLocalIdxBuffer.contents().bindMemory(to: UInt16.self, capacity: totalSize)
        for tile in 0 ..< tileCount {
            let offset = tile * maxPerTile
            let count = Int(tileCounts[tile])
            guard count > 1 else { continue }
            for j in 0 ..< (count - 1) {
                let idx1 = Int(gpuSortedIdx[offset + j])
                let idx2 = Int(gpuSortedIdx[offset + j + 1])
                let depth1 = depthKeys16[offset + idx1]
                let depth2 = depthKeys16[offset + idx2]
                XCTAssertLessThanOrEqual(depth1, depth2,
                                         "Tile \(tile) not sorted at position \(j): depth \(depth1) > \(depth2)")
            }
        }
    }
}

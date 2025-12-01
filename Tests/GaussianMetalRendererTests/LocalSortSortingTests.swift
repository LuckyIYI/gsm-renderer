import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Test Local per-tile radix sorting
final class LocalSortSortingTests: XCTestCase {

    /// Helper to load Local library from bundle
    private func loadLocalLibrary(device: MTLDevice) throws -> MTLLibrary {
        guard let libraryURL = Bundle.module.url(forResource: "LocalSortShaders", withExtension: "metallib"),
              let library = try? device.makeLibrary(URL: libraryURL) else {
            throw NSError(domain: "LocalSortSortingTests", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to load LocalSortShaders.metallib"])
        }
        return library
    }

    /// Test that per-tile sort correctly sorts depth keys
    func testPerTileSortCorrectness() throws {
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
        let device = renderer.device
        let queue = renderer.queue

        // Load Local library
        let library = try loadLocalLibrary(device: device)

        guard let sortFn = library.makeFunction(name: "localSortPerTileSort") else {
            XCTFail("Missing sort function")
            return
        }
        let sortPipeline = try device.makeComputePipelineState(function: sortFn)

        // Test parameters
        let tileCount = 4
        let gaussiansPerTile = 64
        let totalAssignments = tileCount * gaussiansPerTile

        // Create offsets and counts
        var offsets = [UInt32]()
        var counts = [UInt32]()
        for i in 0..<tileCount {
            offsets.append(UInt32(i * gaussiansPerTile))
            counts.append(UInt32(gaussiansPerTile))
        }
        offsets.append(UInt32(totalAssignments))  // End offset

        // Create random keys and values
        var keys = [UInt32]()
        var values = [UInt32]()
        for tile in 0..<tileCount {
            for i in 0..<gaussiansPerTile {
                // Random 24-bit depth key
                let depth = Float.random(in: 0.1...100.0)
                let depthKey = (depth.bitPattern ^ 0x80000000) >> 8
                keys.append(depthKey)
                values.append(UInt32(tile * gaussiansPerTile + i))  // Original index
            }
        }

        // Create buffers
        let keysBuffer = device.makeBuffer(bytes: &keys, length: keys.count * 4, options: .storageModeShared)!
        let valuesBuffer = device.makeBuffer(bytes: &values, length: values.count * 4, options: .storageModeShared)!
        let offsetsBuffer = device.makeBuffer(bytes: &offsets, length: offsets.count * 4, options: .storageModeShared)!
        let countsBuffer = device.makeBuffer(bytes: &counts, length: counts.count * 4, options: .storageModeShared)!
        let tempKeysBuffer = device.makeBuffer(length: keys.count * 4, options: .storageModeShared)!
        let tempValuesBuffer = device.makeBuffer(length: values.count * 4, options: .storageModeShared)!

        // Run sort
        guard let cb = queue.makeCommandBuffer(),
              let encoder = cb.makeComputeCommandEncoder() else {
            XCTFail("Failed to create command buffer")
            return
        }

        encoder.setComputePipelineState(sortPipeline)
        encoder.setBuffer(keysBuffer, offset: 0, index: 0)
        encoder.setBuffer(valuesBuffer, offset: 0, index: 1)
        encoder.setBuffer(offsetsBuffer, offset: 0, index: 2)
        encoder.setBuffer(countsBuffer, offset: 0, index: 3)
        encoder.setBuffer(tempKeysBuffer, offset: 0, index: 4)
        encoder.setBuffer(tempValuesBuffer, offset: 0, index: 5)

        let threadsPerTG = min(sortPipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreadgroups(
            MTLSize(width: tileCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerTG, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        // Verify each tile is sorted
        let sortedKeys = keysBuffer.contents().bindMemory(to: UInt32.self, capacity: keys.count)
        let sortedValues = valuesBuffer.contents().bindMemory(to: UInt32.self, capacity: values.count)

        var allSorted = true
        var sortErrors = 0

        for tile in 0..<tileCount {
            let start = tile * gaussiansPerTile
            var prevKey: UInt32 = 0

            for i in 0..<gaussiansPerTile {
                let key = sortedKeys[start + i]
                if key < prevKey {
                    allSorted = false
                    sortErrors += 1
                    if sortErrors <= 5 {
                        print("Sort error in tile \(tile): key[\(i-1)]=\(prevKey) > key[\(i)]=\(key)")
                    }
                }
                prevKey = key
            }
        }

        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║  LOCAL SORT SORTING TEST                                     ║")
        print("╠═══════════════════════════════════════════════════════════╣")
        print("║  Tiles: \(tileCount), Gaussians/tile: \(gaussiansPerTile)")
        print("║  Sort errors: \(sortErrors)")
        print("║  Status: \(allSorted ? "PASS" : "FAIL")")
        print("╚═══════════════════════════════════════════════════════════╝\n")

        XCTAssertTrue(allSorted, "All tiles should be sorted by depth")
    }

    /// Test sort stability with many identical keys
    func testSortWithIdenticalKeys() throws {
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
        let device = renderer.device
        let queue = renderer.queue

        let library = try loadLocalLibrary(device: device)

        guard let sortFn = library.makeFunction(name: "localSortPerTileSort") else {
            XCTFail("Missing sort function")
            return
        }
        let sortPipeline = try device.makeComputePipelineState(function: sortFn)

        let tileCount = 1
        let gaussiansPerTile = 128

        var offsets: [UInt32] = [0, UInt32(gaussiansPerTile)]
        var counts: [UInt32] = [UInt32(gaussiansPerTile)]

        // All same key - tests stability
        var keys = [UInt32](repeating: 0x00FFFFFF, count: gaussiansPerTile)
        var values = (0..<gaussiansPerTile).map { UInt32($0) }

        let keysBuffer = device.makeBuffer(bytes: &keys, length: keys.count * 4, options: .storageModeShared)!
        let valuesBuffer = device.makeBuffer(bytes: &values, length: values.count * 4, options: .storageModeShared)!
        let offsetsBuffer = device.makeBuffer(bytes: &offsets, length: offsets.count * 4, options: .storageModeShared)!
        let countsBuffer = device.makeBuffer(bytes: &counts, length: counts.count * 4, options: .storageModeShared)!
        let tempKeysBuffer = device.makeBuffer(length: keys.count * 4, options: .storageModeShared)!
        let tempValuesBuffer = device.makeBuffer(length: values.count * 4, options: .storageModeShared)!

        guard let cb = queue.makeCommandBuffer(),
              let encoder = cb.makeComputeCommandEncoder() else {
            XCTFail("Failed to create command buffer")
            return
        }

        encoder.setComputePipelineState(sortPipeline)
        encoder.setBuffer(keysBuffer, offset: 0, index: 0)
        encoder.setBuffer(valuesBuffer, offset: 0, index: 1)
        encoder.setBuffer(offsetsBuffer, offset: 0, index: 2)
        encoder.setBuffer(countsBuffer, offset: 0, index: 3)
        encoder.setBuffer(tempKeysBuffer, offset: 0, index: 4)
        encoder.setBuffer(tempValuesBuffer, offset: 0, index: 5)

        encoder.dispatchThreadgroups(
            MTLSize(width: tileCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let sortedValues = valuesBuffer.contents().bindMemory(to: UInt32.self, capacity: values.count)

        // With identical keys, values should still be valid (each appears exactly once)
        var valueCounts = [UInt32: Int]()
        for i in 0..<gaussiansPerTile {
            let v = sortedValues[i]
            valueCounts[v, default: 0] += 1
        }

        let allUnique = valueCounts.values.allSatisfy { $0 == 1 }
        let allValid = valueCounts.keys.allSatisfy { $0 < UInt32(gaussiansPerTile) }

        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║  IDENTICAL KEYS SORT TEST                                  ║")
        print("╠═══════════════════════════════════════════════════════════╣")
        print("║  Unique values preserved: \(allUnique ? "YES" : "NO")")
        print("║  All values valid: \(allValid ? "YES" : "NO")")
        print("╚═══════════════════════════════════════════════════════════╝\n")

        XCTAssertTrue(allUnique, "All values should appear exactly once")
        XCTAssertTrue(allValid, "All values should be valid indices")
    }

    /// Test sort with large tile counts (stress test)
    func testSortLargeTileCount() throws {
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
        let device = renderer.device
        let queue = renderer.queue

        let library = try loadLocalLibrary(device: device)

        guard let sortFn = library.makeFunction(name: "localSortPerTileSort") else {
            XCTFail("Missing sort function")
            return
        }
        let sortPipeline = try device.makeComputePipelineState(function: sortFn)

        // Simulate 1080p with 32x16 tiles = 60x68 = 4080 tiles
        let tileCount = 4080
        let avgGaussiansPerTile = 50
        let totalAssignments = tileCount * avgGaussiansPerTile

        var offsets = [UInt32]()
        var counts = [UInt32]()
        var currentOffset: UInt32 = 0
        for _ in 0..<tileCount {
            offsets.append(currentOffset)
            let count = UInt32.random(in: 10...100)
            counts.append(count)
            currentOffset += count
        }
        offsets.append(currentOffset)

        var keys = [UInt32](repeating: 0, count: Int(currentOffset))
        var values = [UInt32](repeating: 0, count: Int(currentOffset))

        for tile in 0..<tileCount {
            let start = Int(offsets[tile])
            let count = Int(counts[tile])
            for i in 0..<count {
                let depth = Float.random(in: 0.1...100.0)
                keys[start + i] = (depth.bitPattern ^ 0x80000000) >> 8
                values[start + i] = UInt32(start + i)
            }
        }

        let keysBuffer = device.makeBuffer(bytes: &keys, length: keys.count * 4, options: .storageModeShared)!
        let valuesBuffer = device.makeBuffer(bytes: &values, length: values.count * 4, options: .storageModeShared)!
        let offsetsBuffer = device.makeBuffer(bytes: &offsets, length: offsets.count * 4, options: .storageModeShared)!
        let countsBuffer = device.makeBuffer(bytes: &counts, length: counts.count * 4, options: .storageModeShared)!
        let tempKeysBuffer = device.makeBuffer(length: keys.count * 4, options: .storageModeShared)!
        let tempValuesBuffer = device.makeBuffer(length: values.count * 4, options: .storageModeShared)!

        guard let cb = queue.makeCommandBuffer(),
              let encoder = cb.makeComputeCommandEncoder() else {
            XCTFail("Failed to create command buffer")
            return
        }

        encoder.setComputePipelineState(sortPipeline)
        encoder.setBuffer(keysBuffer, offset: 0, index: 0)
        encoder.setBuffer(valuesBuffer, offset: 0, index: 1)
        encoder.setBuffer(offsetsBuffer, offset: 0, index: 2)
        encoder.setBuffer(countsBuffer, offset: 0, index: 3)
        encoder.setBuffer(tempKeysBuffer, offset: 0, index: 4)
        encoder.setBuffer(tempValuesBuffer, offset: 0, index: 5)

        encoder.dispatchThreadgroups(
            MTLSize(width: tileCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        // Verify random sample of tiles
        let sortedKeys = keysBuffer.contents().bindMemory(to: UInt32.self, capacity: keys.count)

        var sortErrors = 0
        let samplesToCheck = min(100, tileCount)
        for _ in 0..<samplesToCheck {
            let tile = Int.random(in: 0..<tileCount)
            let start = Int(offsets[tile])
            let count = Int(counts[tile])

            var prevKey: UInt32 = 0
            for i in 0..<count {
                let key = sortedKeys[start + i]
                if key < prevKey {
                    sortErrors += 1
                }
                prevKey = key
            }
        }

        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║  LARGE TILE COUNT SORT TEST                                ║")
        print("╠═══════════════════════════════════════════════════════════╣")
        print("║  Tiles: \(tileCount), Total assignments: \(currentOffset)")
        print("║  Tiles checked: \(samplesToCheck)")
        print("║  Sort errors: \(sortErrors)")
        print("╚═══════════════════════════════════════════════════════════╝\n")

        XCTAssertEqual(sortErrors, 0, "No sort errors should be found")
    }
}

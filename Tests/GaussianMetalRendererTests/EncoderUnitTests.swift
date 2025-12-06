@testable import GaussianMetalRenderer
import Metal
import simd
import XCTest

/// Isolated unit tests for individual encoders
/// Tests both correctness (output matching) and performance of optimized variants
final class EncoderUnitTests: XCTestCase {
    var device: MTLDevice!
    var queue: MTLCommandQueue!
    var library: MTLLibrary!

    override func setUp() {
        super.setUp()
        self.device = MTLCreateSystemDefaultDevice()!
        self.queue = self.device.makeCommandQueue()!

        // Load LocalShaders library directly
        guard let libraryURL = Bundle.module.url(forResource: "LocalShaders", withExtension: "metallib"),
              let lib = try? self.device.makeLibrary(URL: libraryURL)
        else {
            XCTFail("Failed to load LocalShaders.metallib")
            return
        }
        self.library = lib
    }

    // MARK: - Timing Helper

    func measureEncoder(name: String, warmup: Int = 3, iterations: Int = 10, _ block: () -> Void) -> (avg: Double, stddev: Double, min: Double, max: Double) {
        for _ in 0 ..< warmup { block() }

        var times: [Double] = []
        for _ in 0 ..< iterations {
            let start = CFAbsoluteTimeGetCurrent()
            block()
            times.append((CFAbsoluteTimeGetCurrent() - start) * 1000.0)
        }

        let avg = times.reduce(0, +) / Double(times.count)
        let variance = times.map { ($0 - avg) * ($0 - avg) }.reduce(0, +) / Double(times.count)
        return (avg, sqrt(variance), times.min()!, times.max()!)
    }

    // MARK: - Scatter Encoder Unit Tests

    func testScatterEncoderVariants() throws {
        print("\n" + String(repeating: "=", count: 60))
        print("SCATTER ENCODER UNIT TEST")
        print("Device: \(self.device.name)")
        print(String(repeating: "=", count: 60))

        let scatterEncoder = try LocalScatterEncoder(library: self.library, device: self.device)

        // Test parameters - use 2M gaussians to match real-world profiling
        let visibleCount = 2_000_000 // Match profiling data
        let tilesX = 120 // 1920/16
        let tilesY = 68 // 1080/16
        let totalTiles = tilesX * tilesY
        let maxPerTile = 4096

        // Create test buffers
        let compactedSize = visibleCount * MemoryLayout<CompactedGaussian>.stride
        guard let compactedBuffer = self.device.makeBuffer(length: compactedSize, options: .storageModeShared),
              let headerBuffer = self.device.makeBuffer(length: 16, options: .storageModeShared),
              let tileCountersBuffer = self.device.makeBuffer(length: totalTiles * 4, options: .storageModeShared),
              let depthKeys16Buffer = self.device.makeBuffer(length: totalTiles * maxPerTile * 2, options: .storageModeShared),
              let globalIndicesBuffer = self.device.makeBuffer(length: totalTiles * maxPerTile * 4, options: .storageModeShared)
        else {
            XCTFail("Failed to create buffers")
            return
        }

        // Initialize compacted gaussians with CLUSTERED distribution
        // Simulate looking at an object - many gaussians overlapping the same tiles
        let compactedPtr = compactedBuffer.contents().bindMemory(to: CompactedGaussian.self, capacity: visibleCount)
        srand48(42)

        // Create 100 clusters spread across the screen
        let numClusters = 100
        var clusterCenters: [(Float, Float)] = []
        for _ in 0 ..< numClusters {
            clusterCenters.append((Float(drand48()) * 1920, Float(drand48()) * 1080))
        }

        for i in 0 ..< visibleCount {
            // Assign to a random cluster with gaussian distribution around center
            let clusterIdx = Int(drand48() * Double(numClusters))
            let (cx, cy) = clusterCenters[clusterIdx]

            // Position with gaussian distribution (stddev ~50 pixels)
            let stddev: Float = 50.0
            let angle = Float(drand48() * .pi * 2)
            let dist = Float(sqrt(-2.0 * log(max(drand48(), 1e-10)))) * stddev
            let px = max(0, min(1919, cx + dist * cos(angle)))
            let py = max(0, min(1079, cy + dist * sin(angle)))

            // Larger radius to create more tile overlaps
            let radius = Float(drand48() * 40 + 16) // 16-56 pixels
            let minTX = max(0, Int((px - radius) / 16))
            let minTY = max(0, Int((py - radius) / 16))
            let maxTX = min(tilesX, Int((px + radius) / 16) + 1)
            let maxTY = min(tilesY, Int((py + radius) / 16) + 1)

            var g = CompactedGaussian()
            g.positionColor = SIMD4<Float>(px, py, 0, 0)
            g.covarianceDepth = SIMD4<Float>(0.01, 0, 0.01, Float(i) / Float(visibleCount)) // Simple conic, depth
            g.minTile = SIMD2<Int32>(Int32(minTX), Int32(minTY))
            g.maxTile = SIMD2<Int32>(Int32(maxTX), Int32(maxTY))
            g.originalIdx = UInt32(i)
            compactedPtr[i] = g
        }

        // Set header with visible count
        let headerPtr = headerBuffer.contents().bindMemory(to: UInt32.self, capacity: 4)
        headerPtr[0] = UInt32(visibleCount)

        // Test each variant
        var results: [(name: String, time: Double, totalAssigned: Int)] = []

        for variant in ScatterVariant.allCases {
            scatterEncoder.activeVariant = variant

            let variantName: String
            switch variant {
            case .baseline: variantName = "Baseline (4 gauss/TG)"
            case .optimized: variantName = "Optimized (32 gauss/thread)"
            case .sparse: continue // Skip sparse in this test (needs different setup)
            }

            // Clear counters before each test
            memset(tileCountersBuffer.contents(), 0, totalTiles * 4)

            let timing = self.measureEncoder(name: variantName, warmup: 3, iterations: 10) {
                // Clear counters
                memset(tileCountersBuffer.contents(), 0, totalTiles * 4)

                guard let cb = self.queue.makeCommandBuffer() else { return }
                scatterEncoder.encode16(
                    commandBuffer: cb,
                    compactedGaussians: compactedBuffer,
                    compactedHeader: headerBuffer,
                    tileCounters: tileCountersBuffer,
                    depthKeys16: depthKeys16Buffer,
                    globalIndices: globalIndicesBuffer,
                    tilesX: tilesX,
                    maxPerTile: maxPerTile,
                    tileWidth: 16,
                    tileHeight: 16
                )
                cb.commit()
                cb.waitUntilCompleted()
            }

            // Count total assignments
            let countersPtr = tileCountersBuffer.contents().bindMemory(to: UInt32.self, capacity: totalTiles)
            var totalAssigned = 0
            for t in 0 ..< totalTiles {
                totalAssigned += Int(countersPtr[t])
            }

            results.append((variantName, timing.avg, totalAssigned))

            print("\n[\(variantName)]")
            print("  Time: \(String(format: "%.3f", timing.avg))ms (stddev: \(String(format: "%.3f", timing.stddev)))")
            print("  Total assignments: \(totalAssigned)")
        }

        // Verify correctness: both variants should produce same assignment count
        let baselineAssignments = results[0].totalAssigned
        for result in results {
            XCTAssertEqual(result.totalAssigned, baselineAssignments,
                           "\(result.name) produced different assignment count: \(result.totalAssigned) vs \(baselineAssignments)")
        }

        // Summary
        print("\n" + String(repeating: "-", count: 40))
        print("SUMMARY (vs baseline):")
        let baselineTime = results[0].time
        for result in results {
            let speedup = baselineTime / result.time
            let marker = speedup > 1.05 ? "✓" : (speedup < 0.95 ? "✗" : "~")
            print("  \(marker) \(result.name): \(String(format: "%.2f", speedup))x")
        }
        print(String(repeating: "=", count: 60) + "\n")
    }

    // MARK: - Run All Encoder Tests

    func testAllEncoderVariants() throws {
        try self.testScatterEncoderVariants()
    }
}

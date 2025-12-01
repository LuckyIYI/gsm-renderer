import XCTest
@testable import GaussianMetalRenderer

final class HeapAllocationTests: XCTestCase {
    func testHeapSizingWithinBounds() {
        // Configure a large-but-reasonable scene to verify heap sizing stays finite.
        // With maxAssignments clamped to 4× maxGaussians, heap size is bounded.
        let limits = RendererLimits(
            maxGaussians: 1_000_000,
            maxWidth: 1024,
            maxHeight: 1024,
            tileWidth: 16,
            tileHeight: 16,
            maxPerTile: 512
        )
        // Explicitly enable heap allocation for this test (it's disabled by default)
        let renderer = GlobalSortRenderer(useHeapAllocation: true, limits: limits)

        // Heap should exist and be non-zero.
        guard let heapSize = renderer.debugHeapSizeBytes() else {
            XCTFail("Heap not allocated")
            return
        }
        XCTAssertGreaterThan(heapSize, 0, "Heap size should be positive")

        // With the above limits, heap should fit comfortably under 8 GB.
        let eightGB = 8 * 1024 * 1024 * 1024
        XCTAssertLessThan(heapSize, eightGB, "Heap size unreasonably large (\(heapSize) bytes)")
    }
}

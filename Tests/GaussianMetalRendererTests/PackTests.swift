import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

final class PackTests: XCTestCase {
    
    var device: MTLDevice!
    var library: MTLLibrary!
    var queue: MTLCommandQueue!
    
    override func setUp() {
        super.setUp()
        let renderer = Renderer.shared
        self.device = renderer.device
        self.library = renderer.library
        self.queue = renderer.queue
    }
    
    func testBuildHeaders() throws {
        let encoder = try PackEncoder(device: device, library: library)
        
        // Scenario: 10 assignments.
        // Tile 0: 3 items
        // Tile 1: 0 items
        // Tile 2: 2 items
        // Tile 3: 5 items
        // Keys: 0,0,0, 2,2, 3,3,3,3,3
        
        let keys: [SIMD2<UInt32>] = [
            SIMD2(0, 10), SIMD2(0, 20), SIMD2(0, 30),
            SIMD2(2, 10), SIMD2(2, 20),
            SIMD2(3, 10), SIMD2(3, 11), SIMD2(3, 12), SIMD2(3, 13), SIMD2(3, 14)
        ]
        let count = keys.count
        let tileCount = 5
        
        let keysBuffer = device.makeBuffer(bytes: keys, length: count * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
        
        let orderedHeaders = device.makeBuffer(length: tileCount * MemoryLayout<GaussianHeader>.stride, options: .storageModeShared)!
        let assignmentHeader = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
        
        let headerPtr = assignmentHeader.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.totalAssignments = UInt32(count)
        
        // We can't easily run *just* the header kernel via the public `encode` method because it requires all buffers.
        // However, we can invoke the pipeline directly if we access the internal pipeline, OR we can pass dummy buffers for the rest.
        
        // Let's pass dummy buffers to `encode`.
        let dummyGaussian = GaussianInputBuffers(
            means: device.makeBuffer(length: 16, options: .storageModePrivate)!,
            radii: device.makeBuffer(length: 16, options: .storageModePrivate)!,
            mask: device.makeBuffer(length: 16, options: .storageModePrivate)!,
            depths: device.makeBuffer(length: 16, options: .storageModePrivate)!,
            conics: device.makeBuffer(length: 16, options: .storageModePrivate)!,
            colors: device.makeBuffer(length: 16, options: .storageModePrivate)!,
            opacities: device.makeBuffer(length: 16, options: .storageModePrivate)!
        )
        
        let dummyOrdered = OrderedBufferSet(
            headers: orderedHeaders,
            means: device.makeBuffer(length: 16, options: .storageModePrivate)!,
            conics: device.makeBuffer(length: 16, options: .storageModePrivate)!,
            colors: device.makeBuffer(length: 16, options: .storageModePrivate)!,
            opacities: device.makeBuffer(length: 16, options: .storageModePrivate)!,
            depths: device.makeBuffer(length: 16, options: .storageModePrivate)!
        )
        
        let dummyAssignment = TileAssignmentBuffers(
            tileCount: tileCount,
            maxAssignments: count,
            tileIndices: device.makeBuffer(length: 4 * count, options: .storageModePrivate)!,
            tileIds: device.makeBuffer(length: 4 * count, options: .storageModePrivate)!,
            header: assignmentHeader
        )
        
        let activeTileIndices = device.makeBuffer(length: tileCount * 4, options: .storageModePrivate)!
        let activeTileCount = device.makeBuffer(length: 4, options: .storageModePrivate)!
        
        let dispatchArgs = device.makeBuffer(length: 1024, options: .storageModePrivate)!
        
        let commandBuffer = queue.makeCommandBuffer()!
        
        // We only care about `encode` running `buildHeadersFromSortedKernel`.
        // It runs: Pack -> Headers -> Clear -> Compact.
        // Pack might crash if we give it small dummy buffers but tell it `totalAssignments` is 10.
        // `packTileDataKernel` reads `sortedIndices`[0..9].
        // We need `sortedIndices` to be valid size.
        let sortedIndices = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        // And `gaussianBuffers` need to be large enough to read from `sortedIndices` indices?
        // Actually `sortedIndices` values will be 0.. (assumed). Pack reads `means[sortedIndices[i]]`.
        // If sortedIndices contains 0s, means must have at least 1 element.
        // Our dummy buffers have length 16, so index 0 is safe.
        
        encoder.encode(
            commandBuffer: commandBuffer,
            sortedIndices: sortedIndices,
            sortedKeys: keysBuffer,
            gaussianBuffers: dummyGaussian,
            orderedBuffers: dummyOrdered,
            assignment: dummyAssignment,
            totalAssignments: count,
            dispatchArgs: dispatchArgs,
            dispatchOffset: 0,
            activeTileIndices: activeTileIndices,
            activeTileCount: activeTileCount
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Verify Headers
        let outHeaders = orderedHeaders.contents().bindMemory(to: GaussianHeader.self, capacity: tileCount)
        
        // Expected:
        // Tile 0: offset 0, count 3
        XCTAssertEqual(outHeaders[0].offset, 0)
        XCTAssertEqual(outHeaders[0].count, 3)
        
        // Tile 1: offset 3, count 0 (empty)
        XCTAssertEqual(outHeaders[1].offset, 3)
        XCTAssertEqual(outHeaders[1].count, 0) // Offset 3 is where it "would" start
        
        // Tile 2: offset 3, count 2
        XCTAssertEqual(outHeaders[2].offset, 3)
        XCTAssertEqual(outHeaders[2].count, 2)
        
        // Tile 3: offset 5, count 5
        XCTAssertEqual(outHeaders[3].offset, 5)
        XCTAssertEqual(outHeaders[3].count, 5)
        
        // Tile 4: offset 10, count 0
        XCTAssertEqual(outHeaders[4].offset, 10)
        XCTAssertEqual(outHeaders[4].count, 0)
    }
}

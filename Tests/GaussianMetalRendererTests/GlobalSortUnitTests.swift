import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Unit tests for GlobalSort pipeline stages
final class GlobalSortUnitTests: XCTestCase {

    var device: MTLDevice!
    var library: MTLLibrary!
    var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 100_000, maxWidth: 512, maxHeight: 512))
        self.device = renderer.device
        self.library = renderer.library
        self.queue = device.makeCommandQueue()!
    }

    // MARK: - Radix Sort Tests

    func testRadixSortCorrectness() throws {
        let encoder = try RadixSortEncoder(device: device, library: library)
        let count = 1024

        // Generate test data: 32-bit keys [tile:16][depth:16]
        var keys = [UInt32](repeating: 0, count: count)
        var indices = [Int32](repeating: 0, count: count)

        srand48(42)
        for i in 0..<count {
            let tileId = UInt32(drand48() * 10)
            let depth = Float(drand48() * 100.0)
            let halfDepth = Float16(depth)
            let depthBits = UInt32(halfDepth.bitPattern) ^ 0x8000
            keys[i] = (tileId << 16) | (depthBits & 0xFFFF)
            indices[i] = Int32(i)
        }

        // CPU reference sort
        let cpuSorted = zip(keys, indices).sorted { $0.0 < $1.0 }

        // GPU sort
        let keyBuffer = device.makeBuffer(bytes: keys, length: count * 4, options: .storageModeShared)!
        let indicesBuffer = device.makeBuffer(bytes: indices, length: count * 4, options: .storageModeShared)!
        let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
        let dispatchArgs = device.makeBuffer(length: 4096, options: .storageModeShared)!

        let headerPtr = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.totalAssignments = UInt32(count)

        let dispatchEncoder = try DispatchEncoder(
            device: device, library: library,
            config: AssignmentDispatchConfigSwift(
                sortThreadgroupSize: 256, fuseThreadgroupSize: 256, unpackThreadgroupSize: 256,
                packThreadgroupSize: 256, bitonicThreadgroupSize: 256,
                radixBlockSize: 256, radixGrainSize: 4, maxAssignments: UInt32(count * 10)
            )
        )

        let gridSize = max(1, (count + 1023) / 1024)
        let histogramCount = gridSize * 256
        let radixBuffers = RadixBufferSet(
            histogram: device.makeBuffer(length: histogramCount * 4, options: .storageModeShared)!,
            blockSums: device.makeBuffer(length: gridSize * 4, options: .storageModeShared)!,
            scannedHistogram: device.makeBuffer(length: histogramCount * 4, options: .storageModeShared)!,
            scratchKeys: device.makeBuffer(length: count * 4, options: .storageModeShared)!,
            scratchPayload: device.makeBuffer(length: count * 4, options: .storageModeShared)!
        )

        let offsets = (
            histogram: DispatchSlot.radixHistogram.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scanBlocks: DispatchSlot.radixScanBlocks.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            exclusive: DispatchSlot.radixExclusive.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            apply: DispatchSlot.radixApply.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scatter: DispatchSlot.radixScatter.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
        )

        // Dispatch setup
        let cb1 = queue.makeCommandBuffer()!
        dispatchEncoder.encode(commandBuffer: cb1, header: headerBuffer, dispatchArgs: dispatchArgs, maxAssignments: count * 10)
        cb1.commit()
        cb1.waitUntilCompleted()

        // Sort
        let cb2 = queue.makeCommandBuffer()!
        encoder.encode(
            commandBuffer: cb2, keyBuffer: keyBuffer, sortedIndices: indicesBuffer,
            header: headerBuffer, dispatchArgs: dispatchArgs, radixBuffers: radixBuffers,
            offsets: offsets, tileCount: 10
        )
        cb2.commit()
        cb2.waitUntilCompleted()

        // Verify
        let gpuKeys = keyBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
        for i in 0..<count-1 {
            XCTAssertLessThanOrEqual(gpuKeys[i], gpuKeys[i+1], "Keys not sorted at \(i)")
        }

        for i in 0..<count {
            XCTAssertEqual(gpuKeys[i], cpuSorted[i].0, "Key mismatch at \(i)")
        }
    }

    func testRadixSortLargeScale() throws {
        let encoder = try RadixSortEncoder(device: device, library: library)
        let count = 50_000

        var keys = [UInt32](repeating: 0, count: count)
        var indices = [Int32](repeating: 0, count: count)

        srand48(123)
        for i in 0..<count {
            let tileId = UInt32(drand48() * 100)
            let depth = Float(drand48() * 100.0)
            let halfDepth = Float16(depth)
            let depthBits = UInt32(halfDepth.bitPattern) ^ 0x8000
            keys[i] = (tileId << 16) | (depthBits & 0xFFFF)
            indices[i] = Int32(i)
        }

        let keyBuffer = device.makeBuffer(bytes: keys, length: count * 4, options: .storageModeShared)!
        let indicesBuffer = device.makeBuffer(bytes: indices, length: count * 4, options: .storageModeShared)!
        let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
        let dispatchArgs = device.makeBuffer(length: 4096, options: .storageModePrivate)!

        let headerPtr = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.totalAssignments = UInt32(count)

        let dispatchEncoder = try DispatchEncoder(
            device: device, library: library,
            config: AssignmentDispatchConfigSwift(
                sortThreadgroupSize: 256, fuseThreadgroupSize: 256, unpackThreadgroupSize: 256,
                packThreadgroupSize: 256, bitonicThreadgroupSize: 256,
                radixBlockSize: 256, radixGrainSize: 4, maxAssignments: UInt32(count * 10)
            )
        )

        let gridSize = max(1, (count + 1023) / 1024)
        let histogramCount = gridSize * 256
        let radixBuffers = RadixBufferSet(
            histogram: device.makeBuffer(length: histogramCount * 4, options: .storageModePrivate)!,
            blockSums: device.makeBuffer(length: gridSize * 4, options: .storageModePrivate)!,
            scannedHistogram: device.makeBuffer(length: histogramCount * 4, options: .storageModePrivate)!,
            scratchKeys: device.makeBuffer(length: count * 4, options: .storageModePrivate)!,
            scratchPayload: device.makeBuffer(length: count * 4, options: .storageModePrivate)!
        )

        let offsets = (
            histogram: DispatchSlot.radixHistogram.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scanBlocks: DispatchSlot.radixScanBlocks.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            exclusive: DispatchSlot.radixExclusive.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            apply: DispatchSlot.radixApply.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scatter: DispatchSlot.radixScatter.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
        )

        let cb1 = queue.makeCommandBuffer()!
        dispatchEncoder.encode(commandBuffer: cb1, header: headerBuffer, dispatchArgs: dispatchArgs, maxAssignments: count * 10)
        cb1.commit()
        cb1.waitUntilCompleted()

        let cb2 = queue.makeCommandBuffer()!
        encoder.encode(
            commandBuffer: cb2, keyBuffer: keyBuffer, sortedIndices: indicesBuffer,
            header: headerBuffer, dispatchArgs: dispatchArgs, radixBuffers: radixBuffers,
            offsets: offsets, tileCount: 100
        )
        cb2.commit()
        cb2.waitUntilCompleted()

        // Verify sorted
        let gpuKeys = keyBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
        for i in 0..<count-1 {
            XCTAssertLessThanOrEqual(gpuKeys[i], gpuKeys[i+1], "Keys not sorted at \(i)")
        }
    }
}

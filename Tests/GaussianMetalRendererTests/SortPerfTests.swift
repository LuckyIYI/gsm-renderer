import XCTest
import Metal
import simd
import Foundation
@testable import GaussianMetalRenderer

/// Lightweight perf sanity for radix vs bitonic encoders.
final class SortPerfTests: XCTestCase {
    private let sizes = [100_000, 1_000_000, 4_000_000]
    private let tileCount = 4096
    private let blockSize = 256
    private let grainSize = 4
    private let radix = 256

    private func makeDispatchEncoder(device: MTLDevice, library: MTLLibrary) throws -> DispatchEncoder {
        try DispatchEncoder(
            device: device,
            library: library,
            config: AssignmentDispatchConfigSwift(
                sortThreadgroupSize: UInt32(blockSize),
                fuseThreadgroupSize: UInt32(blockSize),
                unpackThreadgroupSize: UInt32(blockSize),
                packThreadgroupSize: UInt32(blockSize),
                bitonicThreadgroupSize: UInt32(blockSize),
                radixBlockSize: UInt32(blockSize),
                radixGrainSize: UInt32(grainSize),
                maxAssignments: 0  // Set dynamically at encode time
            )
        )
    }

    func testSortPerfRadixVsBitonic() throws {
        let renderer = Renderer.shared
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue
        let dispatchArgs = device.makeBuffer(length: 4096, options: .storageModePrivate)!
        let dispatchEncoder = try makeDispatchEncoder(device: device, library: library)
        let radixEncoder = try RadixSortEncoder(device: device, library: library)
        let bitonicEncoder = try BitonicSortEncoder(device: device, library: library)

        let radixOffsets = (
            fuse: DispatchSlot.fuseKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            unpack: DispatchSlot.unpackKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            histogram: DispatchSlot.radixHistogram.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scanBlocks: DispatchSlot.radixScanBlocks.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            exclusive: DispatchSlot.radixExclusive.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            apply: DispatchSlot.radixApply.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scatter: DispatchSlot.radixScatter.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
        )

        let bitonicOffsets = (
            first: DispatchSlot.bitonicFirst.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            general: DispatchSlot.bitonicGeneral.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            final: DispatchSlot.bitonicFinal.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
        )

        var summaries: [String] = []

        for count in sizes {
            let paddedCount = nextPow2(count)
            let valuesPerGroup = blockSize * grainSize
            let gridSize = max(1, (paddedCount + valuesPerGroup - 1) / valuesPerGroup)
            let histogramCount = gridSize * radix

            let sortKeys = device.makeBuffer(length: paddedCount * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
            let sortIndices = device.makeBuffer(length: paddedCount * MemoryLayout<Int32>.stride, options: .storageModeShared)!
            let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!

            let radixBuffers = RadixBufferSet(
                histogram: device.makeBuffer(length: histogramCount * 4, options: .storageModePrivate)!,
                blockSums: device.makeBuffer(length: gridSize * 4, options: .storageModePrivate)!,
                scannedHistogram: device.makeBuffer(length: histogramCount * 4, options: .storageModePrivate)!,
                fusedKeys: device.makeBuffer(length: paddedCount * 8, options: .storageModePrivate)!,
                scratchKeys: device.makeBuffer(length: paddedCount * 8, options: .storageModePrivate)!,
                scratchPayload: device.makeBuffer(length: paddedCount * 4, options: .storageModePrivate)!
            )

            let (keys, indices) = makeKeys(count: count, maxTile: tileCount)
            let expectedKeys = keys
            let expectedIdx = indices

            func fillInput() {
                let keysBuf = sortKeys.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: paddedCount)
                keys.withUnsafeBufferPointer { src in
                    keysBuf.update(from: src.baseAddress!, count: keys.count)
                }
                let idxBuf = sortIndices.contents().bindMemory(to: Int32.self, capacity: paddedCount)
                indices.withUnsafeBufferPointer { src in
                    idxBuf.update(from: src.baseAddress!, count: indices.count)
                }
                let padKey = SIMD2<UInt32>(UInt32.max, UInt32.max)
                for i in count..<paddedCount {
                    keysBuf[i] = padKey
                    idxBuf[i] = -1
                }
                let headerPtr = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
                headerPtr.pointee.totalAssignments = UInt32(count)
                headerPtr.pointee.paddedCount = UInt32(paddedCount)
                headerPtr.pointee.maxAssignments = UInt32(paddedCount)
            }


            func runRadix() -> Double {
                fillInput()
                guard let cb = queue.makeCommandBuffer() else { return .infinity }
                dispatchEncoder.encode(commandBuffer: cb, header: headerBuffer, dispatchArgs: dispatchArgs, maxAssignments: paddedCount)
                cb.commit()
                cb.waitUntilCompleted()

                guard let sortCb = queue.makeCommandBuffer() else { return .infinity }
                let start = CFAbsoluteTimeGetCurrent()
                radixEncoder.encode(
                    commandBuffer: sortCb,
                    keyBuffer: sortKeys,
                    sortedIndices: sortIndices,
                    header: headerBuffer,
                    dispatchArgs: dispatchArgs,
                    radixBuffers: radixBuffers,
                    offsets: radixOffsets,
                    tileCount: tileCount
                )
                sortCb.commit()
                sortCb.waitUntilCompleted()
                let end = CFAbsoluteTimeGetCurrent()
                return (end - start) * 1000.0
            }

            func runBitonic() -> Double {
                fillInput()
                guard let cb = queue.makeCommandBuffer() else { return .infinity }
                dispatchEncoder.encode(commandBuffer: cb, header: headerBuffer, dispatchArgs: dispatchArgs, maxAssignments: paddedCount)
                cb.commit()
                cb.waitUntilCompleted()

                guard let sortCb = queue.makeCommandBuffer() else { return .infinity }
                let start = CFAbsoluteTimeGetCurrent()
                bitonicEncoder.encode(
                    commandBuffer: sortCb,
                    sortKeys: sortKeys,
                    sortedIndices: sortIndices,
                    header: headerBuffer,
                    dispatchArgs: dispatchArgs,
                    offsets: bitonicOffsets,
                    paddedCapacity: paddedCount
                )
                sortCb.commit()
                sortCb.waitUntilCompleted()
                let end = CFAbsoluteTimeGetCurrent()
                return (end - start) * 1000.0
            }

            // Warmup once per size.
            _ = runRadix()
            _ = runBitonic()

            let radixMs = runRadix()
            let bitonicMs = runBitonic()
            summaries.append("\(count): radix=\(String(format: "%.3f", radixMs))ms, bitonic=\(String(format: "%.3f", bitonicMs))ms")
        }

        print("[SortPerf] \(summaries.joined(separator: " | "))")
    }

    private func makeKeys(count: Int, maxTile: Int) -> ([SIMD2<UInt32>], [Int32]) {
        var keys = [SIMD2<UInt32>](repeating: .zero, count: count)
        var indices = [Int32](repeating: 0, count: count)
        for i in 0..<count {
            let tile = UInt32(i / max(1, count / maxTile))
            let depth = Float(i) * 0.0001 + 0.1
            keys[i] = SIMD2<UInt32>(tile, depth.bitPattern)
            indices[i] = Int32(i)
        }
        return (keys, indices)
    }

    private func nextPow2(_ value: Int) -> Int {
        var v = value - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        v += 1
        return max(v, 1)
    }
}

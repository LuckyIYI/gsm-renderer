import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

final class EncoderTests: XCTestCase {
    
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
    
    func testRadixSort() throws {
        let encoder = try RadixSortEncoder(device: device, library: library)
        
        let count = 1024
        
        var keys = [SIMD2<UInt32>](repeating: .zero, count: count)
        var indices = [Int32](repeating: 0, count: count)
        
        for i in 0..<count {
            let tileId = UInt32.random(in: 0..<10)
            let depth = Float.random(in: 0.1...100.0)
            let depthBits = depth.bitPattern
            keys[i] = SIMD2<UInt32>(tileId, depthBits)
            indices[i] = Int32(i)
        }
        
        struct Item {
            let key: SIMD2<UInt32>
            let index: Int32
        }
        
        let items = zip(keys, indices).map { Item(key: $0, index: $1) }
        let sortedItems = items.sorted { a, b in
            if a.key.x != b.key.x {
                return a.key.x < b.key.x
            }
            return a.key.y < b.key.y
        }
        
        let keyBuffer = device.makeBuffer(bytes: keys, length: count * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
        let indicesBuffer = device.makeBuffer(bytes: indices, length: count * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
        
        let dispatchArgs = device.makeBuffer(length: 1024, options: .storageModePrivate)!
        
        let dispatchEncoder = try DispatchEncoder(
            device: device, 
            library: library, 
            config: AssignmentDispatchConfigSwift(
                sortThreadgroupSize: 256,
                fuseThreadgroupSize: 256,
                unpackThreadgroupSize: 256,
                packThreadgroupSize: 256,
                bitonicThreadgroupSize: 256,
                radixBlockSize: 256,
                radixGrainSize: 4
            )
        )
        
        let headerPtr = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.totalAssignments = UInt32(count)
        
        let valuesPerGroup = 256 * 4
        let gridSize = max(1, (count + valuesPerGroup - 1) / valuesPerGroup)
        let histogramCount = gridSize * 256
        
        let histBuf = device.makeBuffer(length: histogramCount * 4, options: .storageModePrivate)!
        let blockSumsBuf = device.makeBuffer(length: gridSize * 4, options: .storageModePrivate)!
        let scannedHistBuf = device.makeBuffer(length: histogramCount * 4, options: .storageModePrivate)!
        let fusedKeysBuf = device.makeBuffer(length: count * 8, options: .storageModePrivate)!
        let scratchKeysBuf = device.makeBuffer(length: count * 8, options: .storageModePrivate)!
        let scratchPayloadBuf = device.makeBuffer(length: count * 4, options: .storageModePrivate)!
        
        let radixBuffers = RadixBufferSet(
            histogram: histBuf,
            blockSums: blockSumsBuf,
            scannedHistogram: scannedHistBuf,
            fusedKeys: fusedKeysBuf,
            scratchKeys: scratchKeysBuf,
            scratchPayload: scratchPayloadBuf
        )
        
        let offsets = (
            fuse: DispatchSlot.fuseKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            unpack: DispatchSlot.unpackKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            histogram: DispatchSlot.radixHistogram.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scanBlocks: DispatchSlot.radixScanBlocks.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            exclusive: DispatchSlot.radixExclusive.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            apply: DispatchSlot.radixApply.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scatter: DispatchSlot.radixScatter.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
        )
        
        let commandBuffer = queue.makeCommandBuffer()!
        
        dispatchEncoder.encode(commandBuffer: commandBuffer, header: headerBuffer, dispatchArgs: dispatchArgs)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let commandBuffer2 = queue.makeCommandBuffer()!
        
        encoder.encode(
            commandBuffer: commandBuffer2,
            keyBuffer: keyBuffer,
            sortedIndices: indicesBuffer,
            header: headerBuffer,
            dispatchArgs: dispatchArgs,
            radixBuffers: radixBuffers,
            offsets: offsets
        )
        
        commandBuffer2.commit()
        commandBuffer2.waitUntilCompleted()
        
        let outIndices = indicesBuffer.contents().bindMemory(to: Int32.self, capacity: count)
        let outKeys = keyBuffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: count)

        // Verify Sorted Order
        for i in 0..<count {
            let cpuIdx = sortedItems[i].index
            let gpuIdx = outIndices[i]
            
            if cpuIdx != gpuIdx {
                // Allow stable sort variation if keys are identical
                let cpuKey = sortedItems[i].key
                let gpuKey = outKeys[i]
                
                if cpuKey != gpuKey {
                     XCTFail("Mismatch at \(i): CPU Key \(cpuKey) Idx \(cpuIdx) vs GPU Key \(gpuKey) Idx \(gpuIdx)")
                     break
                }
            }
        }
        
        // Verify Keys Strictly Sorted
        for i in 0..<count-1 {
            let k1 = outKeys[i]
            let k2 = outKeys[i+1]
            let k1Val = (UInt64(k1.x) << 32) | UInt64(k1.y)
            let k2Val = (UInt64(k2.x) << 32) | UInt64(k2.y)
            XCTAssertLessThanOrEqual(k1Val, k2Val, "Keys not sorted at \(i)")
        }
    }

    func testBitonicSort() throws {
        let encoder = try BitonicSortEncoder(device: device, library: library)
        
//        let count = 2048 // Reduced size for debugging
//        let count = 65536 // workd
        let count = 131072 // works
//        let count = 262144 

        var keys = [SIMD2<UInt32>](repeating: .zero, count: count)
        var indices = [Int32](repeating: 0, count: count)
        
        for i in 0..<count {
            let tileId = UInt32.random(in: 0..<50)
            let depth = Float.random(in: 0.1...100.0)
            let depthBits = depth.bitPattern
            keys[i] = SIMD2<UInt32>(tileId, depthBits)
            indices[i] = Int32(i)
        }
        
        struct Item {
            let key: SIMD2<UInt32>
            let index: Int32
        }
        
        let items = zip(keys, indices).map { Item(key: $0, index: $1) }
        let sortedItems = items.sorted { a, b in
            if a.key.x != b.key.x {
                return a.key.x < b.key.x
            }
            return a.key.y < b.key.y
        }
        
        let keyBuffer = device.makeBuffer(bytes: keys, length: count * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
        let indicesBuffer = device.makeBuffer(bytes: indices, length: count * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
        
        let dispatchArgs = device.makeBuffer(length: 4096, options: .storageModePrivate)!
        
        // Config for dispatch
        let dispatchEncoder = try DispatchEncoder(
            device: device, 
            library: library, 
            config: AssignmentDispatchConfigSwift(
                sortThreadgroupSize: 256,
                fuseThreadgroupSize: 256,
                unpackThreadgroupSize: 256,
                packThreadgroupSize: 256,
                // Match the encoder's chosen threadgroup size to keep indirect dispatch in sync.
                bitonicThreadgroupSize: UInt32(encoder.unitSize),
                radixBlockSize: 256,
                radixGrainSize: 4
            )
        )
        
        let headerPtr = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.totalAssignments = UInt32(count)
        var padded = 1
        while padded < count { padded <<= 1 }
        headerPtr.pointee.paddedCount = UInt32(padded)
        
        let offsets = (
            first: DispatchSlot.bitonicFirst.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            general: DispatchSlot.bitonicGeneral.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            final: DispatchSlot.bitonicFinal.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
        )
        
        let commandBuffer = queue.makeCommandBuffer()!
        
        dispatchEncoder.encode(commandBuffer: commandBuffer, header: headerBuffer, dispatchArgs: dispatchArgs)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let commandBuffer2 = queue.makeCommandBuffer()!
        
        encoder.encode(
            commandBuffer: commandBuffer2,
            sortKeys: keyBuffer,
            sortedIndices: indicesBuffer,
            header: headerBuffer,
            dispatchArgs: dispatchArgs,
            offsets: offsets,
            paddedCapacity: padded
        )
        
        commandBuffer2.commit()
        commandBuffer2.waitUntilCompleted()

        let outIndices = indicesBuffer.contents().bindMemory(to: Int32.self, capacity: count)
        let outKeys = keyBuffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: count)
        
        // Verify Sorted Order
        for i in 0..<count {
            let cpuIdx = sortedItems[i].index
            let gpuIdx = outIndices[i]
            
            if cpuIdx != gpuIdx {
                let cpuKey = sortedItems[i].key
                let gpuKey = outKeys[i]
                
                if cpuKey != gpuKey {
                     print("First 10 GPU Keys:")
                     for k in 0..<min(10, count) {
                         print("\(k): \(outKeys[k]) Idx \(outIndices[k])")
                     }
                     XCTFail("Bitonic Mismatch at \(i): CPU Key \(cpuKey) Idx \(cpuIdx) vs GPU Key \(gpuKey) Idx \(gpuIdx)")
                     break
                }
            }
        }
        
        // Verify Keys Strictly Sorted
        for i in 0..<count-1 {
            let k1 = outKeys[i]
            let k2 = outKeys[i+1]
            let k1Val = (UInt64(k1.x) << 32) | UInt64(k1.y)
            let k2Val = (UInt64(k2.x) << 32) | UInt64(k2.y)
            XCTAssertLessThanOrEqual(k1Val, k2Val, "Bitonic Keys not sorted at \(i)")
            if k1Val < k2Val {
                break
            }
        }
    }

    func testRadixSortLarge() throws {
        let encoder = try RadixSortEncoder(device: device, library: library)

        let count = 131_072
        
        var keys = [SIMD2<UInt32>](repeating: .zero, count: count)
        var indices = [Int32](repeating: 0, count: count)
        
        // Deterministic seed or random? Random is fine for sorting.
        for i in 0..<count {
            let tileId = UInt32.random(in: 0..<50) // More tiles
            let depth = Float.random(in: 0.1...100.0)
            let depthBits = depth.bitPattern
            keys[i] = SIMD2<UInt32>(tileId, depthBits)
            indices[i] = Int32(i)
        }
        
        // CPU Reference Sort
        struct Item {
            let key: SIMD2<UInt32>
            let index: Int32
        }
        
        let items = zip(keys, indices).map { Item(key: $0, index: $1) }
        let sortedItems = items.sorted { a, b in
            if a.key.x != b.key.x {
                return a.key.x < b.key.x
            }
            return a.key.y < b.key.y
        }
        
        let keyBuffer = device.makeBuffer(bytes: keys, length: count * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
        let indicesBuffer = device.makeBuffer(bytes: indices, length: count * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
        
        let dispatchArgs = device.makeBuffer(length: 4096, options: .storageModePrivate)!
        
        let dispatchEncoder = try DispatchEncoder(
            device: device, 
            library: library, 
            config: AssignmentDispatchConfigSwift(
                sortThreadgroupSize: 256,
                fuseThreadgroupSize: 256,
                unpackThreadgroupSize: 256,
                packThreadgroupSize: 256,
                bitonicThreadgroupSize: 256,
                radixBlockSize: 256,
                radixGrainSize: 4
            )
        )
        
        let headerPtr = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.totalAssignments = UInt32(count)
        
        let valuesPerGroup = 256 * 4
        let gridSize = max(1, (count + valuesPerGroup - 1) / valuesPerGroup)
        let histogramCount = gridSize * 256
        
        let histBuf = device.makeBuffer(length: histogramCount * 4, options: .storageModePrivate)!
        let blockSumsBuf = device.makeBuffer(length: gridSize * 4, options: .storageModePrivate)!
        let scannedHistBuf = device.makeBuffer(length: histogramCount * 4, options: .storageModePrivate)!
        let fusedKeysBuf = device.makeBuffer(length: count * 8, options: .storageModePrivate)!
        let scratchKeysBuf = device.makeBuffer(length: count * 8, options: .storageModePrivate)!
        let scratchPayloadBuf = device.makeBuffer(length: count * 4, options: .storageModePrivate)!
        
        let radixBuffers = RadixBufferSet(
            histogram: histBuf,
            blockSums: blockSumsBuf,
            scannedHistogram: scannedHistBuf,
            fusedKeys: fusedKeysBuf,
            scratchKeys: scratchKeysBuf,
            scratchPayload: scratchPayloadBuf
        )
        
        let offsets = (
            fuse: DispatchSlot.fuseKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            unpack: DispatchSlot.unpackKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            histogram: DispatchSlot.radixHistogram.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scanBlocks: DispatchSlot.radixScanBlocks.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            exclusive: DispatchSlot.radixExclusive.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            apply: DispatchSlot.radixApply.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scatter: DispatchSlot.radixScatter.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
        )
        
        let commandBuffer = queue.makeCommandBuffer()!
        
        dispatchEncoder.encode(commandBuffer: commandBuffer, header: headerBuffer, dispatchArgs: dispatchArgs)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let commandBuffer2 = queue.makeCommandBuffer()!
        
        encoder.encode(
            commandBuffer: commandBuffer2,
            keyBuffer: keyBuffer,
            sortedIndices: indicesBuffer,
            header: headerBuffer,
            dispatchArgs: dispatchArgs,
            radixBuffers: radixBuffers,
            offsets: offsets
        )
        
        commandBuffer2.commit()
        commandBuffer2.waitUntilCompleted()
        
        let outIndices = indicesBuffer.contents().bindMemory(to: Int32.self, capacity: count)
        let outKeys = keyBuffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: count)
        
        // Verify Sorted Order
        for i in 0..<count {
            let cpuIdx = sortedItems[i].index
            let gpuIdx = outIndices[i]
            
            if cpuIdx != gpuIdx {
                let cpuKey = sortedItems[i].key
                let gpuKey = outKeys[i]
                
                if cpuKey != gpuKey {
                     XCTFail("Mismatch at \(i): CPU Key \(cpuKey) Idx \(cpuIdx) vs GPU Key \(gpuKey) Idx \(gpuIdx)")
                     break
                }
            }
        }
        
        // Verify Keys Strictly Sorted
        for i in 0..<count-1 {
            let k1 = outKeys[i]
            let k2 = outKeys[i+1]
            let k1Val = (UInt64(k1.x) << 32) | UInt64(k1.y)
            let k2Val = (UInt64(k2.x) << 32) | UInt64(k2.y)
            XCTAssertLessThanOrEqual(k1Val, k2Val, "Keys not sorted at \(i)")
        }
    
    }
}

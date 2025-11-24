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

        // Use deterministic values for debugging
        for i in 0..<count {
            let tileId = UInt32(i % 10)  // Tiles 0-9 in order
            let depth = Float(i)  // Simple increasing depth
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
        let tileCount = 10

        let keyBuffer = device.makeBuffer(bytes: keys, length: count * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
        let indicesBuffer = device.makeBuffer(bytes: indices, length: count * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
        
        let dispatchArgs = device.makeBuffer(length: 1024, options: .storageModeShared)!

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

        // Use shared storage to inspect intermediate results
        let histBuf = device.makeBuffer(length: histogramCount * 4, options: .storageModeShared)!
        let blockSumsBuf = device.makeBuffer(length: gridSize * 4, options: .storageModeShared)!
        let scannedHistBuf = device.makeBuffer(length: histogramCount * 4, options: .storageModeShared)!
        let fusedKeysBuf = device.makeBuffer(length: count * 8, options: .storageModeShared)!
        let scratchKeysBuf = device.makeBuffer(length: count * 8, options: .storageModeShared)!
        let scratchPayloadBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        
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

        // Debug: Print header after dispatch encoder
        let headerAfter = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        NSLog("After dispatch: totalAssignments=%d paddedCount=%d", headerAfter.pointee.totalAssignments, headerAfter.pointee.paddedCount)

        // Debug: Print dispatch args
        let dispatchArgsPtr = dispatchArgs.contents().bindMemory(to: DispatchIndirectArgsSwift.self, capacity: 20)
        let fuseSlot = DispatchSlot.fuseKeys.rawValue
        let histSlot = DispatchSlot.radixHistogram.rawValue
        let scatterSlot = DispatchSlot.radixScatter.rawValue
        NSLog("Dispatch args: fuse=(%d,%d,%d) histogram=(%d,%d,%d) scatter=(%d,%d,%d)",
              dispatchArgsPtr[fuseSlot].threadgroupsPerGridX,
              dispatchArgsPtr[fuseSlot].threadgroupsPerGridY,
              dispatchArgsPtr[fuseSlot].threadgroupsPerGridZ,
              dispatchArgsPtr[histSlot].threadgroupsPerGridX,
              dispatchArgsPtr[histSlot].threadgroupsPerGridY,
              dispatchArgsPtr[histSlot].threadgroupsPerGridZ,
              dispatchArgsPtr[scatterSlot].threadgroupsPerGridX,
              dispatchArgsPtr[scatterSlot].threadgroupsPerGridY,
              dispatchArgsPtr[scatterSlot].threadgroupsPerGridZ)

        let commandBuffer2 = queue.makeCommandBuffer()!

        // Only run the fuse step first to check the fused keys
        let fuseEncoder = try RadixSortEncoder(device: device, library: library)

        // Manually encode just the fuse step
        if let enc = commandBuffer2.makeComputeCommandEncoder() {
            enc.label = "FuseOnly"
            guard let fuseFn = library.makeFunction(name: "fuseSortKeysKernel") else {
                XCTFail("fuseSortKeysKernel not found")
                return
            }
            let fusePipeline = try device.makeComputePipelineState(function: fuseFn)
            enc.setComputePipelineState(fusePipeline)
            enc.setBuffer(keyBuffer, offset: 0, index: 0)
            enc.setBuffer(fusedKeysBuf, offset: 0, index: 1)
            enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(count, 256), height: 1, depth: 1))
            enc.endEncoding()
        }
        commandBuffer2.commit()
        commandBuffer2.waitUntilCompleted()

        // Debug: Print fused keys BEFORE sort
        let fusedBeforePtr = fusedKeysBuf.contents().bindMemory(to: UInt64.self, capacity: count)
        var fuseCheckMsg = "Fused keys BEFORE sort:\n"
        for i in [0, 1, 2, 10, 11, 100, 101] {
            let fk = fusedBeforePtr[i]
            fuseCheckMsg += "  [\(i)] fused=\(fk) tile=\(fk >> 32)\n"
        }
        NSLog("%@", fuseCheckMsg)

        // Now run the full sort
        let commandBuffer3 = queue.makeCommandBuffer()!

        encoder.encode(
            commandBuffer: commandBuffer3,
            keyBuffer: keyBuffer,
            sortedIndices: indicesBuffer,
            header: headerBuffer,
            dispatchArgs: dispatchArgs,
            radixBuffers: radixBuffers,
            offsets: offsets,
            tileCount: tileCount
        )

        commandBuffer3.commit()
        commandBuffer3.waitUntilCompleted()

        // Debug: Print fused keys (after all passes)
        let fusedKeysPtr = fusedKeysBuf.contents().bindMemory(to: UInt64.self, capacity: count)
        let scratchKeysPtr = scratchKeysBuf.contents().bindMemory(to: UInt64.self, capacity: count)
        var debugMsg = "First 10 fusedKeys after sort:\n"
        for i in 0..<10 {
            let fk = fusedKeysPtr[i]
            let sk = scratchKeysPtr[i]
            debugMsg += "  [\(i)] fused=\(fk) tile=\(fk >> 32) | scratch=\(sk) tile=\(sk >> 32)\n"
        }
        NSLog("%@", debugMsg)

        let outIndices = indicesBuffer.contents().bindMemory(to: Int32.self, capacity: count)
        let outKeys = keyBuffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: count)

        // Debug: Print first 20 output keys as an assertion message
        var outputMsg = "First 20 GPU sorted keys:\n"
        for i in 0..<20 {
            outputMsg += "  [\(i)] tile=\(outKeys[i].x) depth=\(outKeys[i].y) idx=\(outIndices[i])\n"
        }
        NSLog("%@", outputMsg)

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

    // Test just histogram and first pass to debug
    func testRadixSortSinglePass() throws {
        // Simple test: 16 elements, 4 unique bins (tiles 0-3)
        let count = 16

        var keys = [SIMD2<UInt32>](repeating: .zero, count: count)
        var indices = [Int32](repeating: 0, count: count)

        // Create simple pattern: indices 0,4,8,12 have tile=0; 1,5,9,13 have tile=1; etc.
        for i in 0..<count {
            let tileId = UInt32(i % 4)
            let depth = Float(i)
            keys[i] = SIMD2<UInt32>(tileId, depth.bitPattern)
            indices[i] = Int32(i)
        }

        // Expected after sorting by tile: elements with tile 0 first, then tile 1, etc.
        NSLog("Input: %@", keys.map { "(\($0.x), \($0.y))" }.joined(separator: ", "))

        let keyBuffer = device.makeBuffer(bytes: keys, length: count * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
        let indicesBuffer = device.makeBuffer(bytes: indices, length: count * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!

        let headerPtr = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.totalAssignments = UInt32(count)
        var padded = 1
        while padded < count { padded <<= 1 }
        headerPtr.pointee.paddedCount = UInt32(padded)

        // Fuse the keys manually for inspection
        let fusedKeysBuf = device.makeBuffer(length: count * 8, options: .storageModeShared)!

        guard let fuseFn = library.makeFunction(name: "fuseSortKeysKernel"),
              let histFn = library.makeFunction(name: "radixHistogramKernel"),
              let scanFn = library.makeFunction(name: "radixScanBlocksKernel"),
              let exclusiveFn = library.makeFunction(name: "radixExclusiveScanKernel"),
              let applyFn = library.makeFunction(name: "radixApplyScanOffsetsKernel"),
              let scatterFn = library.makeFunction(name: "radixScatterKernel") else {
            XCTFail("Missing kernel functions")
            return
        }

        let fusePipeline = try device.makeComputePipelineState(function: fuseFn)
        let histPipeline = try device.makeComputePipelineState(function: histFn)
        let scanPipeline = try device.makeComputePipelineState(function: scanFn)
        let exclusivePipeline = try device.makeComputePipelineState(function: exclusiveFn)
        let applyPipeline = try device.makeComputePipelineState(function: applyFn)
        let scatterPipeline = try device.makeComputePipelineState(function: scatterFn)

        // Allocate histogram buffers (1 threadgroup for 16 elements)
        let valuesPerGroup = 256 * 4  // BLOCK_SIZE * GRAIN_SIZE
        let gridSize = max(1, (count + valuesPerGroup - 1) / valuesPerGroup)
        let histogramCount = gridSize * 256

        let histBuf = device.makeBuffer(length: histogramCount * 4, options: .storageModeShared)!
        let blockSumsBuf = device.makeBuffer(length: max(gridSize, 256) * 4, options: .storageModeShared)!
        let scannedHistBuf = device.makeBuffer(length: histogramCount * 4, options: .storageModeShared)!
        let scratchKeysBuf = device.makeBuffer(length: count * 8, options: .storageModeShared)!
        let scratchPayloadBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!

        let commandBuffer = queue.makeCommandBuffer()!

        // Step 1: Fuse keys
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Fuse"
            encoder.setComputePipelineState(fusePipeline)
            encoder.setBuffer(keyBuffer, offset: 0, index: 0)
            encoder.setBuffer(fusedKeysBuf, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(count, 256), height: 1, depth: 1))
            encoder.endEncoding()
        }

        // Step 2: One histogram pass (digit 0)
        var digit: UInt32 = 4  // Start with byte 4 (first byte of tile)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Histogram"
            encoder.setComputePipelineState(histPipeline)
            encoder.setBuffer(fusedKeysBuf, offset: 0, index: 0)
            encoder.setBuffer(histBuf, offset: 0, index: 1)
            encoder.setBytes(&digit, length: 4, index: 3)
            encoder.setBuffer(headerBuffer, offset: 0, index: 4)
            encoder.dispatchThreadgroups(MTLSize(width: gridSize, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // Step 3: Scan blocks
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "ScanBlocks"
            encoder.setComputePipelineState(scanPipeline)
            encoder.setBuffer(histBuf, offset: 0, index: 0)
            encoder.setBuffer(blockSumsBuf, offset: 0, index: 1)
            encoder.setBuffer(headerBuffer, offset: 0, index: 2)
            encoder.dispatchThreadgroups(MTLSize(width: gridSize, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // Step 4: Exclusive scan
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "ExclusiveScan"
            encoder.setComputePipelineState(exclusivePipeline)
            encoder.setBuffer(blockSumsBuf, offset: 0, index: 0)
            encoder.setBuffer(headerBuffer, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // Step 5: Apply offsets
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "ApplyOffsets"
            encoder.setComputePipelineState(applyPipeline)
            encoder.setBuffer(histBuf, offset: 0, index: 0)
            encoder.setBuffer(blockSumsBuf, offset: 0, index: 1)
            encoder.setBuffer(scannedHistBuf, offset: 0, index: 2)
            encoder.setBuffer(headerBuffer, offset: 0, index: 3)
            encoder.setThreadgroupMemoryLength(256 * 4, index: 0)
            encoder.dispatchThreadgroups(MTLSize(width: gridSize, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // Step 6: Scatter
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Scatter"
            encoder.setComputePipelineState(scatterPipeline)
            encoder.setBuffer(scratchKeysBuf, offset: 0, index: 0)      // output keys
            encoder.setBuffer(fusedKeysBuf, offset: 0, index: 1)        // input keys
            encoder.setBuffer(scratchPayloadBuf, offset: 0, index: 2)   // output payload
            encoder.setBuffer(indicesBuffer, offset: 0, index: 3)       // input payload
            encoder.setBuffer(scannedHistBuf, offset: 0, index: 5)      // offsets
            encoder.setBytes(&digit, length: 4, index: 6)
            encoder.setBuffer(headerBuffer, offset: 0, index: 7)
            encoder.dispatchThreadgroups(MTLSize(width: gridSize, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Print results
        let fusedPtr = fusedKeysBuf.contents().bindMemory(to: UInt64.self, capacity: count)
        let histPtr = histBuf.contents().bindMemory(to: UInt32.self, capacity: histogramCount)
        let scannedPtr = scannedHistBuf.contents().bindMemory(to: UInt32.self, capacity: histogramCount)
        let scratchPtr = scratchKeysBuf.contents().bindMemory(to: UInt64.self, capacity: count)
        let scratchPayloadPtr = scratchPayloadBuf.contents().bindMemory(to: UInt32.self, capacity: count)

        NSLog("Fused keys: %@", (0..<count).map { "[\($0)]=\(fusedPtr[$0]) (tile=\(fusedPtr[$0] >> 32))" }.joined(separator: ", "))

        // Print non-zero histogram bins
        var histMsg = "Histogram (non-zero bins): "
        for i in 0..<256 {
            let val = histPtr[i]
            if val > 0 {
                histMsg += "bin[\(i)]=\(val) "
            }
        }
        NSLog("%@", histMsg)

        // Print scanned histogram for bins 0-3
        NSLog("Scanned histogram bins 0-3: %@", (0..<4).map { "[\($0)]=\(scannedPtr[$0])" }.joined(separator: ", "))

        // Print sorted output
        NSLog("After scatter: %@", (0..<count).map {
            let k = scratchPtr[$0]
            let p = scratchPayloadPtr[$0]
            return "[\($0)] key=\(k) tile=\(k >> 32) idx=\(p)"
        }.joined(separator: ", "))

        // Verify: elements should be grouped by tile (bins 0,1,2,3)
        var prevTile: UInt64 = 0
        for i in 0..<count {
            let tile = scratchPtr[i] >> 32
            XCTAssertGreaterThanOrEqual(tile, prevTile, "Tile at \(i) should be >= previous tile")
            prevTile = tile
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
        let tileCount = 50

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
            offsets: offsets,
            tileCount: tileCount
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

    func testRadixSortNonPowerOfTwo() throws {
        let encoder = try RadixSortEncoder(device: device, library: library)
        let count = 12_345
        var paddedCount = 1
        while paddedCount < count { paddedCount <<= 1 }

        var keys = [SIMD2<UInt32>](repeating: SIMD2<UInt32>(0xFFFFFFFF, 0xFFFFFFFF), count: paddedCount)
        var indices = [Int32](repeating: -1, count: paddedCount)

        for i in 0..<count {
            let tileId = UInt32.random(in: 0..<50)
            let depth = Float.random(in: 0.1...100.0)
            keys[i] = SIMD2<UInt32>(tileId, depth.bitPattern)
            indices[i] = Int32(i)
        }

        let cpuSorted = zip(keys, indices).sorted { a, b in
            if a.0.x != b.0.x { return a.0.x < b.0.x }
            return a.0.y < b.0.y
        }
        let tileCount = 50

        let keyBuffer = device.makeBuffer(bytes: keys, length: paddedCount * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
        let indicesBuffer = device.makeBuffer(bytes: indices, length: paddedCount * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
        let dispatchArgs = device.makeBuffer(length: 4096, options: .storageModePrivate)!

        let headerPtr = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.totalAssignments = UInt32(count)
        headerPtr.pointee.paddedCount = UInt32(paddedCount)

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

        let valuesPerGroup = 256 * 4
        let gridSize = max(1, (paddedCount + valuesPerGroup - 1) / valuesPerGroup)
        let histogramCount = gridSize * 256

        let radixBuffers = RadixBufferSet(
            histogram: device.makeBuffer(length: histogramCount * 4, options: .storageModePrivate)!,
            blockSums: device.makeBuffer(length: gridSize * 4, options: .storageModePrivate)!,
            scannedHistogram: device.makeBuffer(length: histogramCount * 4, options: .storageModePrivate)!,
            fusedKeys: device.makeBuffer(length: paddedCount * 8, options: .storageModePrivate)!,
            scratchKeys: device.makeBuffer(length: paddedCount * 8, options: .storageModePrivate)!,
            scratchPayload: device.makeBuffer(length: paddedCount * 4, options: .storageModePrivate)!
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

        let sortCommand = queue.makeCommandBuffer()!
        encoder.encode(
            commandBuffer: sortCommand,
            keyBuffer: keyBuffer,
            sortedIndices: indicesBuffer,
            header: headerBuffer,
            dispatchArgs: dispatchArgs,
            radixBuffers: radixBuffers,
            offsets: offsets,
            tileCount: tileCount
        )
        sortCommand.commit()
        sortCommand.waitUntilCompleted()

        let gpuKeys = keyBuffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: count)
        let gpuIndices = indicesBuffer.contents().bindMemory(to: Int32.self, capacity: count)

        for i in 0..<count {
            let cpu = cpuSorted[i]
            let gpuKey = gpuKeys[i]
            let gpuIdx = gpuIndices[i]
            if cpu.0 != gpuKey || cpu.1 != gpuIdx {
                XCTFail("Mismatch at \(i): cpuKey \(cpu.0) gpuKey \(gpuKey) cpuIdx \(cpu.1) gpuIdx \(gpuIdx)")
                break
            }
        }
    }
}

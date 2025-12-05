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
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
        self.device = renderer.device
        self.library = renderer.library
        self.queue = renderer.queue
    }
    
    func testRadixSort() throws {
        let encoder = try RadixSortEncoder(device: device, library: library)

        let count = 1024

        // 32-bit keys: [tile:16][depth:16]
        var keys = [UInt32](repeating: 0, count: count)
        var indices = [Int32](repeating: 0, count: count)

        // Use deterministic values with 16-bit depth (half precision)
        for i in 0..<count {
            let tileId = UInt32(i % 10)  // Tiles 0-9 in order
            let depth = Float(i)  // Simple increasing depth
            // Convert to half precision
            let halfDepth = Float16(depth)
            let depthBits = UInt32(halfDepth.bitPattern) ^ 0x8000  // IEEE 754 sign fix
            // Pack as [tile:16][depth:16]
            keys[i] = (tileId << 16) | (depthBits & 0xFFFF)
            indices[i] = Int32(i)
        }

        struct Item {
            let key: UInt32
            let index: Int32
        }

        let items = zip(keys, indices).map { Item(key: $0, index: $1) }
        let sortedItems = items.sorted { $0.key < $1.key }
        let tileCount = 10

        let keyBuffer = device.makeBuffer(bytes: keys, length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let indicesBuffer = device.makeBuffer(bytes: indices, length: count * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!

        let dispatchArgs = device.makeBuffer(length: 1024, options: .storageModeShared)!

        let maxAssignments = count * 10
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
                radixGrainSize: 4,
                maxAssignments: UInt32(maxAssignments)
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
        // 32-bit scratch keys (not 64-bit fused keys anymore)
        let scratchKeysBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let scratchPayloadBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!

        let radixBuffers = RadixBufferSet(
            histogram: histBuf,
            blockSums: blockSumsBuf,
            scannedHistogram: scannedHistBuf,
            scratchKeys: scratchKeysBuf,
            scratchPayload: scratchPayloadBuf
        )

        // New offset tuple without fuse/unpack
        let offsets = (
            histogram: DispatchSlot.radixHistogram.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scanBlocks: DispatchSlot.radixScanBlocks.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            exclusive: DispatchSlot.radixExclusive.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            apply: DispatchSlot.radixApply.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scatter: DispatchSlot.radixScatter.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
        )

        let commandBuffer = queue.makeCommandBuffer()!

        dispatchEncoder.encode(commandBuffer: commandBuffer, header: headerBuffer, dispatchArgs: dispatchArgs, maxAssignments: maxAssignments)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Debug: Print header after dispatch encoder
        let headerAfter = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        NSLog("After dispatch: totalAssignments=%d paddedCount=%d", headerAfter.pointee.totalAssignments, headerAfter.pointee.paddedCount)

        // Debug: Print dispatch args
        let dispatchArgsPtr = dispatchArgs.contents().bindMemory(to: DispatchIndirectArgsSwift.self, capacity: 20)
        let histSlot = DispatchSlot.radixHistogram.rawValue
        let scatterSlot = DispatchSlot.radixScatter.rawValue
        NSLog("Dispatch args: histogram=(%d,%d,%d) scatter=(%d,%d,%d)",
              dispatchArgsPtr[histSlot].threadgroupsPerGridX,
              dispatchArgsPtr[histSlot].threadgroupsPerGridY,
              dispatchArgsPtr[histSlot].threadgroupsPerGridZ,
              dispatchArgsPtr[scatterSlot].threadgroupsPerGridX,
              dispatchArgsPtr[scatterSlot].threadgroupsPerGridY,
              dispatchArgsPtr[scatterSlot].threadgroupsPerGridZ)

        // Run the sort directly (no fuse step needed - keys already 32-bit)
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
        let outKeys = keyBuffer.contents().bindMemory(to: UInt32.self, capacity: count)

        // Debug: Print first 20 output keys as an assertion message
        var outputMsg = "First 20 GPU sorted keys:\n"
        for i in 0..<20 {
            let key = outKeys[i]
            outputMsg += "  [\(i)] tile=\(key >> 16) depth=\(key & 0xFFFF) idx=\(outIndices[i])\n"
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
            XCTAssertLessThanOrEqual(k1, k2, "Keys not sorted at \(i)")
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
        let maxAssignments = count * 10

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
                radixGrainSize: 4,
                maxAssignments: UInt32(maxAssignments)
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

        dispatchEncoder.encode(commandBuffer: commandBuffer, header: headerBuffer, dispatchArgs: dispatchArgs, maxAssignments: maxAssignments)
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
    // Now tests 32-bit keys directly (no fuse step needed)
    func testRadixSortSinglePass() throws {
        // Simple test: 16 elements, 4 unique bins (tiles 0-3)
        let count = 16

        // 32-bit keys: [tile:16][depth:16]
        var keys = [UInt32](repeating: 0, count: count)
        var indices = [Int32](repeating: 0, count: count)

        // Create simple pattern: indices 0,4,8,12 have tile=0; 1,5,9,13 have tile=1; etc.
        for i in 0..<count {
            let tileId = UInt32(i % 4)
            let depthBits = UInt32(i) & 0xFFFF  // Simple depth value
            keys[i] = (tileId << 16) | depthBits
            indices[i] = Int32(i)
        }

        // Expected after sorting by tile: elements with tile 0 first, then tile 1, etc.
        NSLog("Input: %@", keys.map { "tile=\($0 >> 16) depth=\($0 & 0xFFFF)" }.joined(separator: ", "))

        let keyBuffer = device.makeBuffer(bytes: keys, length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let indicesBuffer = device.makeBuffer(bytes: indices, length: count * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!

        let headerPtr = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.totalAssignments = UInt32(count)
        var padded = 1
        while padded < count { padded <<= 1 }
        headerPtr.pointee.paddedCount = UInt32(padded)

        guard let histFn = library.makeFunction(name: "radixHistogramKernel"),
              let scanFn = library.makeFunction(name: "radixScanBlocksKernel"),
              let exclusiveFn = library.makeFunction(name: "radixExclusiveScanKernel"),
              let applyFn = library.makeFunction(name: "radixApplyScanOffsetsKernel"),
              let scatterFn = library.makeFunction(name: "radixScatterKernel") else {
            XCTFail("Missing kernel functions")
            return
        }

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
        let scratchKeysBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let scratchPayloadBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!

        let commandBuffer = queue.makeCommandBuffer()!

        // Step 1: One histogram pass (digit 2 = high byte of 32-bit key = tile bits)
        var digit: UInt32 = 2  // Byte 2 contains tile info (key layout: [tile:16][depth:16])
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Histogram"
            encoder.setComputePipelineState(histPipeline)
            encoder.setBuffer(keyBuffer, offset: 0, index: 0)
            encoder.setBuffer(histBuf, offset: 0, index: 1)
            encoder.setBytes(&digit, length: 4, index: 3)
            encoder.setBuffer(headerBuffer, offset: 0, index: 4)
            encoder.dispatchThreadgroups(MTLSize(width: gridSize, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // Step 2: Scan blocks
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

        // Step 3: Exclusive scan
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "ExclusiveScan"
            encoder.setComputePipelineState(exclusivePipeline)
            encoder.setBuffer(blockSumsBuf, offset: 0, index: 0)
            encoder.setBuffer(headerBuffer, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // Step 4: Apply offsets
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

        // Step 5: Scatter
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Scatter"
            encoder.setComputePipelineState(scatterPipeline)
            encoder.setBuffer(scratchKeysBuf, offset: 0, index: 0)      // output keys
            encoder.setBuffer(keyBuffer, offset: 0, index: 1)           // input keys (32-bit)
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
        let histPtr = histBuf.contents().bindMemory(to: UInt32.self, capacity: histogramCount)
        let scannedPtr = scannedHistBuf.contents().bindMemory(to: UInt32.self, capacity: histogramCount)
        let scratchPtr = scratchKeysBuf.contents().bindMemory(to: UInt32.self, capacity: count)
        let scratchPayloadPtr = scratchPayloadBuf.contents().bindMemory(to: UInt32.self, capacity: count)

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
            return "[\($0)] key=\(k) tile=\(k >> 16) idx=\(p)"
        }.joined(separator: ", "))

        // Verify: elements should be grouped by tile (bins 0,1,2,3)
        var prevTile: UInt32 = 0
        for i in 0..<count {
            let tile = scratchPtr[i] >> 16
            XCTAssertGreaterThanOrEqual(tile, prevTile, "Tile at \(i) should be >= previous tile")
            prevTile = tile
        }
    }

    func testRadixSortLarge() throws {
        let encoder = try RadixSortEncoder(device: device, library: library)

        let count = 131_072

        // 32-bit keys: [tile:16][depth:16]
        var keys = [UInt32](repeating: 0, count: count)
        var indices = [Int32](repeating: 0, count: count)

        // Deterministic seed for reproducibility
        srand48(42)
        for i in 0..<count {
            let tileId = UInt32(drand48() * 50) // More tiles
            let depth = Float(drand48() * 100.0 + 0.1)
            let halfDepth = Float16(depth)
            let depthBits = UInt32(halfDepth.bitPattern) ^ 0x8000  // IEEE 754 sign fix
            keys[i] = (tileId << 16) | (depthBits & 0xFFFF)
            indices[i] = Int32(i)
        }
        let tileCount = 50

        // CPU Reference Sort
        struct Item {
            let key: UInt32
            let index: Int32
        }

        let items = zip(keys, indices).map { Item(key: $0, index: $1) }
        let sortedItems = items.sorted { $0.key < $1.key }

        let keyBuffer = device.makeBuffer(bytes: keys, length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let indicesBuffer = device.makeBuffer(bytes: indices, length: count * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!

        let dispatchArgs = device.makeBuffer(length: 4096, options: .storageModePrivate)!
        let maxAssignments = count * 10

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
                radixGrainSize: 4,
                maxAssignments: UInt32(maxAssignments)
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
        let scratchKeysBuf = device.makeBuffer(length: count * 4, options: .storageModePrivate)!
        let scratchPayloadBuf = device.makeBuffer(length: count * 4, options: .storageModePrivate)!

        let radixBuffers = RadixBufferSet(
            histogram: histBuf,
            blockSums: blockSumsBuf,
            scannedHistogram: scannedHistBuf,
            scratchKeys: scratchKeysBuf,
            scratchPayload: scratchPayloadBuf
        )

        let offsets = (
            histogram: DispatchSlot.radixHistogram.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scanBlocks: DispatchSlot.radixScanBlocks.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            exclusive: DispatchSlot.radixExclusive.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            apply: DispatchSlot.radixApply.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scatter: DispatchSlot.radixScatter.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
        )

        let commandBuffer = queue.makeCommandBuffer()!

        dispatchEncoder.encode(commandBuffer: commandBuffer, header: headerBuffer, dispatchArgs: dispatchArgs, maxAssignments: maxAssignments)
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
        let outKeys = keyBuffer.contents().bindMemory(to: UInt32.self, capacity: count)

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
            XCTAssertLessThanOrEqual(k1, k2, "Keys not sorted at \(i)")
        }

    }

    func testRadixSortNonPowerOfTwo() throws {
        let encoder = try RadixSortEncoder(device: device, library: library)
        let count = 12_345
        var paddedCount = 1
        while paddedCount < count { paddedCount <<= 1 }

        // Use max value for sentinel keys (32-bit key: [tile:16][depth:16])
        var keys = [UInt32](repeating: 0xFFFFFFFF, count: paddedCount)
        var indices = [Int32](repeating: -1, count: paddedCount)

        // Deterministic seed for reproducibility
        srand48(123)
        for i in 0..<count {
            let tileId = UInt32(drand48() * 50)
            let depth = Float(drand48() * 100.0 + 0.1)
            let halfDepth = Float16(depth)
            let depthBits = UInt32(halfDepth.bitPattern) ^ 0x8000  // IEEE 754 sign fix
            keys[i] = (tileId << 16) | (depthBits & 0xFFFF)
            indices[i] = Int32(i)
        }

        let cpuSorted = zip(keys, indices).sorted { $0.0 < $1.0 }
        let tileCount = 50

        let keyBuffer = device.makeBuffer(bytes: keys, length: paddedCount * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let indicesBuffer = device.makeBuffer(bytes: indices, length: paddedCount * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
        let dispatchArgs = device.makeBuffer(length: 4096, options: .storageModePrivate)!

        let headerPtr = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.totalAssignments = UInt32(count)
        headerPtr.pointee.paddedCount = UInt32(paddedCount)
        let maxAssignments = paddedCount * 2

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
                radixGrainSize: 4,
                maxAssignments: UInt32(maxAssignments)
            )
        )

        let valuesPerGroup = 256 * 4
        let gridSize = max(1, (paddedCount + valuesPerGroup - 1) / valuesPerGroup)
        let histogramCount = gridSize * 256

        let radixBuffers = RadixBufferSet(
            histogram: device.makeBuffer(length: histogramCount * 4, options: .storageModePrivate)!,
            blockSums: device.makeBuffer(length: gridSize * 4, options: .storageModePrivate)!,
            scannedHistogram: device.makeBuffer(length: histogramCount * 4, options: .storageModePrivate)!,
            scratchKeys: device.makeBuffer(length: paddedCount * 4, options: .storageModePrivate)!,
            scratchPayload: device.makeBuffer(length: paddedCount * 4, options: .storageModePrivate)!
        )

        let offsets = (
            histogram: DispatchSlot.radixHistogram.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scanBlocks: DispatchSlot.radixScanBlocks.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            exclusive: DispatchSlot.radixExclusive.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            apply: DispatchSlot.radixApply.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scatter: DispatchSlot.radixScatter.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
        )

        let commandBuffer = queue.makeCommandBuffer()!
        dispatchEncoder.encode(commandBuffer: commandBuffer, header: headerBuffer, dispatchArgs: dispatchArgs, maxAssignments: maxAssignments)
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

        let gpuKeys = keyBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
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

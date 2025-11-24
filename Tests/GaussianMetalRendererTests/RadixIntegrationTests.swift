import XCTest
@testable import GaussianMetalRenderer

final class RadixIntegrationTests: XCTestCase {
    func testRadixMatchesBitonicRenderOutput() {
        let width: UInt32 = 320
        let height: UInt32 = 240
        let tile: UInt32 = 16
        let count = 4_096

        var means: [Float] = []
        var conics: [Float] = []
        var colors: [Float] = []
        var opacities: [Float] = []
        var depths: [Float] = []
        var radii: [Float] = []

        var rng = SeededGenerator(seed: 42)
        for i in 0..<count {
            let mx = Float.random(in: 0..<Float(width), using: &rng)
            let my = Float.random(in: 0..<Float(height), using: &rng)
            means.append(mx)
            means.append(my)

            // Simple, well-conditioned conic.
            conics.append(contentsOf: [1.0, 0.0, 1.0, 0.0])

            colors.append(Float.random(in: 0...1, using: &rng))
            colors.append(Float.random(in: 0...1, using: &rng))
            colors.append(Float.random(in: 0...1, using: &rng))

            opacities.append(Float.random(in: 0.1...0.9, using: &rng))
            depths.append(0.5 + Float(i) * 1e-4) // Unique depths to avoid key ties.
            radii.append(6.0)
        }

        let params = RenderParams(
            width: width,
            height: height,
            tileWidth: tile,
            tileHeight: tile,
            tilesX: (width + tile - 1) / tile,
            tilesY: (height + tile - 1) / tile,
            maxPerTile: 0,
            whiteBackground: 0,
            activeTileCount: 0,
            gaussianCount: UInt32(count)
        )

        var bitonicColor = [Float](repeating: 0, count: Int(width * height) * 3)
        var bitonicDepth = [Float](repeating: 0, count: Int(width * height))
        var bitonicAlpha = [Float](repeating: 0, count: Int(width * height))

        var radixColor = bitonicColor
        var radixDepth = bitonicDepth
        var radixAlpha = bitonicAlpha

        let bitonicRenderer = Renderer(useIndirectBitonic: false, sortAlgorithm: .bitonic)
        let radixRenderer = Renderer(useIndirectBitonic: false, sortAlgorithm: .radix)

        let (radixKeys, radixIndices, radixPre, cpuKeys, cpuIdx) = gatherSortedKeys(
            renderer: radixRenderer,
            params: params,
            means: means,
            conics: conics,
            colors: colors,
            opacities: opacities,
            depths: depths,
            radii: radii,
            algorithm: .radix
        )

        // Ensure key generation itself produced non-zero values.
        let zeroVec = SIMD2<UInt32>(0, 0)
        let preRadix = Array(radixPre.prefix(4))
        print("[RadixTest] totalAssignments=\(cpuKeys.count) preRadix=\(preRadix)")
        XCTAssertFalse(preRadix.allSatisfy { $0 == zeroVec }, "Pre-sort keys are zeros; keygen failed (radix)")

        // Compare against CPU reference.
        let total = cpuKeys.count
        var radixCpuMismatch: [(Int, SIMD2<UInt32>, SIMD2<UInt32>, Int32, Int32)] = []
        for i in 0..<min(total, radixKeys.count) {
            if radixKeys[i] != cpuKeys[i] || radixIndices[i] != cpuIdx[i] {
                radixCpuMismatch.append((i, radixKeys[i], cpuKeys[i], radixIndices[i], cpuIdx[i]))
                if radixCpuMismatch.count >= 8 { break }
            }
        }
        XCTAssertTrue(radixCpuMismatch.isEmpty, "Radix output diverges from CPU reference. Sample: \(radixCpuMismatch)")

    }

    private func gatherSortedKeys(
        renderer: Renderer,
        params: RenderParams,
        means: [Float],
        conics: [Float],
        colors: [Float],
        opacities: [Float],
        depths: [Float],
        radii: [Float],
        algorithm: SortAlgorithm
    ) -> ([SIMD2<UInt32>], [Int32], [SIMD2<UInt32>], [SIMD2<UInt32>], [Int32]) {
        let count = means.count / 2

        guard let (frame, slot) = renderer.acquireFrame(width: Int(params.width), height: Int(params.height)) else {
            XCTFail("Failed to acquire frame")
            return ([], [], [], [], [])
        }
        defer { renderer.releaseFrame(index: slot) }

        guard let gaussianBuffers = renderer.prepareGaussianBuffers(count: count) else {
            XCTFail("Failed to allocate gaussian buffers")
            return ([], [], [], [], [])
        }

        _ = means.withUnsafeBytes { src in memcpy(gaussianBuffers.means.contents(), src.baseAddress!, src.count) }
        _ = conics.withUnsafeBytes { src in memcpy(gaussianBuffers.conics.contents(), src.baseAddress!, src.count) }
        _ = colors.withUnsafeBytes { src in memcpy(gaussianBuffers.colors.contents(), src.baseAddress!, src.count) }
        _ = opacities.withUnsafeBytes { src in memcpy(gaussianBuffers.opacities.contents(), src.baseAddress!, src.count) }
        _ = depths.withUnsafeBytes { src in memcpy(gaussianBuffers.depths.contents(), src.baseAddress!, src.count) }
        _ = radii.withUnsafeBytes { src in memcpy(gaussianBuffers.radii.contents(), src.baseAddress!, src.count) }
        memset(gaussianBuffers.mask.contents(), 1, count)

        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Command buffer unavailable")
            return ([], [], [], [], [])
        }

        guard let assignment = renderer.buildTileAssignmentsGPU(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            gaussianBuffers: gaussianBuffers,
            params: params,
            frame: frame,
            estimatedAssignments: nil
        ) else {
            XCTFail("Tile assignment failed")
            return ([], [], [], [], [])
        }

        guard
            let sortKeysBuffer = renderer.ensureBuffer(&frame.sortKeys, length: frame.tileAssignmentPaddedCapacity * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModePrivate, label: "SortKeys"),
            let sortedIndicesBuffer = renderer.ensureBuffer(&frame.sortedIndices, length: frame.tileAssignmentPaddedCapacity * MemoryLayout<Int32>.stride, options: .storageModePrivate, label: "SortedIndices"),
            let dispatchArgs = frame.dispatchArgs
        else {
            XCTFail("Sort buffer allocation failed")
            return ([], [], [], [], [])
        }

        let paddedCount = frame.tileAssignmentPaddedCapacity
        let tileIdsCPU = renderer.device.makeBuffer(length: paddedCount * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let tileIdxCPU = renderer.device.makeBuffer(length: paddedCount * MemoryLayout<Int32>.stride, options: .storageModeShared)!

        renderer.dispatchEncoder.encode(
            commandBuffer: commandBuffer,
            header: assignment.header,
            dispatchArgs: dispatchArgs
        )

        renderer.sortKeyGenEncoder.encode(
            commandBuffer: commandBuffer,
            tileIds: assignment.tileIds,
            tileIndices: assignment.tileIndices,
            depths: gaussianBuffers.depths,
            sortKeys: sortKeysBuffer,
            sortedIndices: sortedIndicesBuffer,
            header: assignment.header,
            dispatchArgs: dispatchArgs,
            dispatchOffset: DispatchSlot.sortKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
        )

        // Snapshot keys/indices right after key generation.
        let preKeysOut = renderer.device.makeBuffer(length: paddedCount * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
        let preIdxOut = renderer.device.makeBuffer(length: paddedCount * MemoryLayout<Int32>.stride, options: .storageModeShared)!

        if let preBlit = commandBuffer.makeBlitCommandEncoder() {
            preBlit.copy(from: sortKeysBuffer, sourceOffset: 0, to: preKeysOut, destinationOffset: 0, size: paddedCount * MemoryLayout<SIMD2<UInt32>>.stride)
            preBlit.copy(from: sortedIndicesBuffer, sourceOffset: 0, to: preIdxOut, destinationOffset: 0, size: paddedCount * MemoryLayout<Int32>.stride)
            preBlit.copy(from: assignment.tileIds, sourceOffset: 0, to: tileIdsCPU, destinationOffset: 0, size: paddedCount * MemoryLayout<Int32>.stride)
            preBlit.copy(from: assignment.tileIndices, sourceOffset: 0, to: tileIdxCPU, destinationOffset: 0, size: paddedCount * MemoryLayout<Int32>.stride)
            preBlit.endEncoding()
        }

        let offsetsRadix = (
            fuse: DispatchSlot.fuseKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            unpack: DispatchSlot.unpackKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            histogram: DispatchSlot.radixHistogram.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scanBlocks: DispatchSlot.radixScanBlocks.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            exclusive: DispatchSlot.radixExclusive.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            apply: DispatchSlot.radixApply.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            scatter: DispatchSlot.radixScatter.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
        )
        let tileCount = Int(params.tilesX * params.tilesY)

        let offsetsBitonic = (
            first: DispatchSlot.bitonicFirst.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            general: DispatchSlot.bitonicGeneral.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
            final: DispatchSlot.bitonicFinal.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
        )

        if algorithm == .radix {
            _ = renderer.ensureRadixBuffers(frame: frame, paddedCapacity: paddedCount)
            let radixBuffers = RadixBufferSet(
                histogram: frame.radixHistogram!,
                blockSums: frame.radixBlockSums!,
                scannedHistogram: frame.radixScannedHistogram!,
                fusedKeys: frame.radixFusedKeys!,
                scratchKeys: frame.radixKeysScratch!,
                scratchPayload: frame.radixPayloadScratch!
            )
            
            renderer.radixSortEncoder.encode(
                commandBuffer: commandBuffer,
                keyBuffer: sortKeysBuffer,
                sortedIndices: sortedIndicesBuffer,
                header: assignment.header,
                dispatchArgs: dispatchArgs,
                radixBuffers: radixBuffers,
                offsets: offsetsRadix,
                tileCount: tileCount
            )
        } else {
            renderer.bitonicSortEncoder.encode(
                commandBuffer: commandBuffer,
                sortKeys: sortKeysBuffer,
                sortedIndices: sortedIndicesBuffer,
                header: assignment.header,
                dispatchArgs: dispatchArgs,
                offsets: offsetsBitonic,
                paddedCapacity: paddedCount
            )
        }

        let keysOut = renderer.device.makeBuffer(length: paddedCount * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
        let indicesOut = renderer.device.makeBuffer(length: paddedCount * MemoryLayout<Int32>.stride, options: .storageModeShared)!

        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.copy(from: sortKeysBuffer, sourceOffset: 0, to: keysOut, destinationOffset: 0, size: paddedCount * MemoryLayout<SIMD2<UInt32>>.stride)
            blit.copy(from: sortedIndicesBuffer, sourceOffset: 0, to: indicesOut, destinationOffset: 0, size: paddedCount * MemoryLayout<Int32>.stride)
            blit.endEncoding()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let headerAfter = assignment.header.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        let totalAssignments = Int(headerAfter.pointee.totalAssignments)
        if totalAssignments == 0 {
            return ([], [], [], [], [])
        }

        let keysPtr = keysOut.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: paddedCount)
        let idxPtr = indicesOut.contents().bindMemory(to: Int32.self, capacity: paddedCount)
        let keysArray = Array(UnsafeBufferPointer<SIMD2<UInt32>>(start: keysPtr, count: paddedCount))
        let indicesArray = Array(UnsafeBufferPointer<Int32>(start: idxPtr, count: paddedCount))
        let preKeysPtr = preKeysOut.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: paddedCount)
        let preKeysArray = Array(UnsafeBufferPointer<SIMD2<UInt32>>(start: preKeysPtr, count: paddedCount))
        let tileIdsArray = Array(UnsafeBufferPointer<Int32>(start: tileIdsCPU.contents().bindMemory(to: Int32.self, capacity: totalAssignments), count: totalAssignments))
        let tileIdxArray = Array(UnsafeBufferPointer<Int32>(start: tileIdxCPU.contents().bindMemory(to: Int32.self, capacity: totalAssignments), count: totalAssignments))
        let depthPtr = gaussianBuffers.depths.contents().bindMemory(to: Float.self, capacity: count)
        var cpuPairs: [(SIMD2<UInt32>, Int32)] = []
        cpuPairs.reserveCapacity(totalAssignments)
        for a in 0..<totalAssignments {
            let idx = Int(tileIdxArray[a])
            let tile = UInt32(bitPattern: tileIdsArray[a])
            let depthBits = depthPtr[idx].bitPattern
            cpuPairs.append((SIMD2(tile, depthBits), Int32(idx)))
        }
        cpuPairs.sort {
            if $0.0.x != $1.0.x { return $0.0.x < $1.0.x }
            return $0.0.y < $1.0.y
        }
        let cpuKeys = cpuPairs.map { $0.0 }
        let cpuIdx = cpuPairs.map { $0.1 }
        return (keysArray, indicesArray, preKeysArray, cpuKeys, cpuIdx)
    }
}

// Deterministic RNG for reproducible tests.
struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        return state &* 2685821657736338717
    }
}

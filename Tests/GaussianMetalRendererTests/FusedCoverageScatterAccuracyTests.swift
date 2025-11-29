import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Validates fused coverage+scatter output against the precise two-pass pipeline.
final class FusedCoverageScatterAccuracyTests: XCTestCase {
    private let tileWidth = 16
    private let tileHeight = 16
    private let imageWidth = 512
    private let imageHeight = 512
    private let gaussianCount = 512

    func testFusedMatchesPrecisePipeline() throws {
        let renderer = Renderer.shared
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let coverageEncoder = try CoverageEncoder(device: device, library: library)
        let scatterEncoder = try ScatterEncoder(device: device, library: library)
        let fusedEncoder = try FusedCoverageScatterEncoder(device: device, library: library)

        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight
        let tileCount = tilesX * tilesY

        // Shared inputs
        let boundsBuffer = device.makeBuffer(length: gaussianCount * MemoryLayout<SIMD4<Int32>>.stride, options: .storageModeShared)!
        let meansBuffer = device.makeBuffer(length: gaussianCount * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
        let conicsBuffer = device.makeBuffer(length: gaussianCount * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)!
        let opacitiesBuffer = device.makeBuffer(length: gaussianCount * MemoryLayout<Float>.stride, options: .storageModeShared)!

        fillTestData(
            boundsBuffer: boundsBuffer,
            meansBuffer: meansBuffer,
            conicsBuffer: conicsBuffer,
            opacitiesBuffer: opacitiesBuffer,
            tilesX: tilesX,
            tilesY: tilesY
        )

        // Precise pipeline buffers
        let coverageBuffer = device.makeBuffer(length: gaussianCount * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let offsetsBuffer = device.makeBuffer(length: (gaussianCount + 1) * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let prefixBlockSize = 256
        let prefixGrainSize = 4
        let elementsPerGroup = prefixBlockSize * prefixGrainSize
        let actualGroups = max(1, (gaussianCount + elementsPerGroup - 1) / elementsPerGroup)
        let partialSumsBuffer = device.makeBuffer(length: actualGroups * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let maxAssignments = gaussianCount * 64
        let preciseTileIds = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let preciseTileIndices = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let preciseHeader = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
        let dispatchBuffer = device.makeBuffer(length: 3 * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

        // Fused pipeline buffers
        let fusedCoverage = device.makeBuffer(length: gaussianCount * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let fusedTileIds = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let fusedTileIndices = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let fusedHeader = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!

        // Precise run
        guard let cbPrecise = queue.makeCommandBuffer() else { XCTFail("No command buffer"); return }
        let preciseHeaderPtr = preciseHeader.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        preciseHeaderPtr.pointee = TileAssignmentHeaderSwift(totalAssignments: 0, maxAssignments: UInt32(maxAssignments))

        coverageEncoder.encodePrecise(
            commandBuffer: cbPrecise,
            gaussianCount: gaussianCount,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            boundsBuffer: boundsBuffer,
            opacitiesBuffer: opacitiesBuffer,
            meansBuffer: meansBuffer,
            conicsBuffer: conicsBuffer,
            coverageBuffer: coverageBuffer,
            offsetsBuffer: offsetsBuffer,
            partialSumsBuffer: partialSumsBuffer,
            tileAssignmentHeader: preciseHeader,
            precision: .float32
        )

        scatterEncoder.encodePrecise(
            commandBuffer: cbPrecise,
            gaussianCount: gaussianCount,
            tilesX: tilesX,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            offsetsBuffer: offsetsBuffer,
            dispatchBuffer: dispatchBuffer,
            boundsBuffer: boundsBuffer,
            tileIndicesBuffer: preciseTileIndices,
            tileIdsBuffer: preciseTileIds,
            tileAssignmentHeader: preciseHeader,
            meansBuffer: meansBuffer,
            conicsBuffer: conicsBuffer,
            opacitiesBuffer: opacitiesBuffer,
            precision: .float32
        )

        cbPrecise.commit()
        cbPrecise.waitUntilCompleted()

        // Fused run
        guard let cbFused = queue.makeCommandBuffer() else { XCTFail("No command buffer"); return }
        let fusedHeaderPtr = fusedHeader.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        fusedHeaderPtr.pointee = TileAssignmentHeaderSwift(totalAssignments: 0, maxAssignments: UInt32(maxAssignments))

        fusedEncoder.encode(
            commandBuffer: cbFused,
            gaussianCount: gaussianCount,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            tilesX: tilesX,
            maxAssignments: maxAssignments,
            boundsBuffer: boundsBuffer,
            coverageBuffer: fusedCoverage,
            opacitiesBuffer: opacitiesBuffer,
            meansBuffer: meansBuffer,
            conicsBuffer: conicsBuffer,
            tileIndicesBuffer: fusedTileIndices,
            tileIdsBuffer: fusedTileIds,
            tileAssignmentHeader: fusedHeader,
            precision: .float32
        )

        cbFused.commit()
        cbFused.waitUntilCompleted()

        // Compare totals
        let preciseTotal = Int(preciseHeaderPtr.pointee.totalAssignments)
        let fusedTotal = Int(fusedHeaderPtr.pointee.totalAssignments)
        XCTAssertEqual(preciseTotal, fusedTotal, "Assignment totals differ")

        // Compare sets of (tileId, gaussianIdx)
        let preciseIds = preciseTileIds.contents().bindMemory(to: Int32.self, capacity: maxAssignments)
        let preciseIdx = preciseTileIndices.contents().bindMemory(to: Int32.self, capacity: maxAssignments)
        var precisePairs: [(Int32, Int32)] = []
        for i in 0..<preciseTotal {
            precisePairs.append((preciseIds[i], preciseIdx[i]))
        }

        let fusedIdsPtr = fusedTileIds.contents().bindMemory(to: Int32.self, capacity: maxAssignments)
        let fusedIdxPtr = fusedTileIndices.contents().bindMemory(to: Int32.self, capacity: maxAssignments)
        var fusedPairs: [(Int32, Int32)] = []
        for i in 0..<fusedTotal {
            fusedPairs.append((fusedIdsPtr[i], fusedIdxPtr[i]))
        }

        precisePairs.sort { $0.0 == $1.0 ? $0.1 < $1.1 : $0.0 < $1.0 }
        fusedPairs.sort { $0.0 == $1.0 ? $0.1 < $1.1 : $0.0 < $1.0 }

        XCTAssertEqual(precisePairs.count, fusedPairs.count, "Pair counts differ")
        for i in 0..<precisePairs.count {
            XCTAssertEqual(precisePairs[i].0, fusedPairs[i].0, "Tile id mismatch at \(i)")
            XCTAssertEqual(precisePairs[i].1, fusedPairs[i].1, "Gaussian index mismatch at \(i)")
        }
    }

    private func fillTestData(
        boundsBuffer: MTLBuffer,
        meansBuffer: MTLBuffer,
        conicsBuffer: MTLBuffer,
        opacitiesBuffer: MTLBuffer,
        tilesX: Int,
        tilesY: Int
    ) {
        let boundsPtr = boundsBuffer.contents().bindMemory(to: SIMD4<Int32>.self, capacity: gaussianCount)
        let meansPtr = meansBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: gaussianCount)
        let conicsPtr = conicsBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: gaussianCount)
        let opacitiesPtr = opacitiesBuffer.contents().bindMemory(to: Float.self, capacity: gaussianCount)

        for i in 0..<gaussianCount {
            let cx = Float.random(in: 0..<Float(imageWidth))
            let cy = Float.random(in: 0..<Float(imageHeight))
            meansPtr[i] = SIMD2<Float>(cx, cy)

            let radius = Float.random(in: 8..<80)
            let minTX = max(0, Int32((cx - radius) / Float(tileWidth)))
            let maxTX = min(Int32(tilesX - 1), Int32((cx + radius) / Float(tileWidth)))
            let minTY = max(0, Int32((cy - radius) / Float(tileHeight)))
            let maxTY = min(Int32(tilesY - 1), Int32((cy + radius) / Float(tileHeight)))
            boundsPtr[i] = SIMD4<Int32>(minTX, maxTX, minTY, maxTY)

            let invR2 = 1.0 / max(radius * radius, 1.0)
            conicsPtr[i] = SIMD4<Float>(invR2, 0, invR2, 0)

            opacitiesPtr[i] = Float.random(in: 0.6..<1.0)
        }
    }
}

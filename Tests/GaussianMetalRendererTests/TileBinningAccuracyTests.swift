import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Validates tile-binning pipeline against the precise coverage+scatter pipeline.
final class TileBinningAccuracyTests: XCTestCase {
    private let tileWidth = 32
    private let tileHeight = 16
    private let imageWidth = 512
    private let imageHeight = 512
    private let gaussianCount = 512

    func testTileBinningMatchesPrecisePipeline() throws {
        let renderer = Renderer(
            precision: .float32,
            useMultiPixelRendering: true,
            usePreciseIntersection: true,
            useTileBinningPipeline: true,
            useFusedCoverageScatter: false
        )
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let coverageEncoder = try CoverageEncoder(device: device, library: library)
        let scatterEncoder = try ScatterEncoder(device: device, library: library)
        let tileBinningEncoder = try TileBinningEncoder(device: device, library: library)

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
            tilesY: tilesY,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            tileWidth: tileWidth,
            tileHeight: tileHeight
        )

        // Precise pipeline buffers
        let coverageBufferPrecise = device.makeBuffer(length: gaussianCount * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let offsetsBufferPrecise = device.makeBuffer(length: (gaussianCount + 1) * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let prefixBlockSize = 256
        let prefixGrainSize = 4
        let elementsPerGroup = prefixBlockSize * prefixGrainSize
        let actualGroupsPrecise = max(1, (gaussianCount + elementsPerGroup - 1) / elementsPerGroup)
        let partialSumsPrecise = device.makeBuffer(length: actualGroupsPrecise * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let maxAssignments = gaussianCount * 64
        let preciseTileIds = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let preciseTileIndices = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let preciseHeader = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
        let dispatchBuffer = device.makeBuffer(length: 3 * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

        // Tile-binning buffers
        let tileCountsBuffer = device.makeBuffer(length: tileCount * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let tileOffsetsBuffer = device.makeBuffer(length: (tileCount + 1) * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let actualGroupsTile = max(1, (tileCount + elementsPerGroup - 1) / elementsPerGroup)
        let partialSumsTile = device.makeBuffer(length: actualGroupsTile * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let tileIdsBinned = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let tileIndicesBinned = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let headerBinned = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!

        // Precise reference
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
            coverageBuffer: coverageBufferPrecise,
            offsetsBuffer: offsetsBufferPrecise,
            partialSumsBuffer: partialSumsPrecise,
            tileAssignmentHeader: preciseHeader,
            precision: .float32
        )

        scatterEncoder.encodePrecise(
            commandBuffer: cbPrecise,
            gaussianCount: gaussianCount,
            tilesX: tilesX,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            offsetsBuffer: offsetsBufferPrecise,
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

        // Tile-binning run
        guard let cbTile = queue.makeCommandBuffer() else { XCTFail("No command buffer"); return }
        let headerPtr = headerBinned.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee = TileAssignmentHeaderSwift(totalAssignments: 0, maxAssignments: UInt32(maxAssignments))

        tileBinningEncoder.encode(
            commandBuffer: cbTile,
            gaussianCount: gaussianCount,
            tilesX: tilesX,
            tilesY: tilesY,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            boundsBuffer: boundsBuffer,
            meansBuffer: meansBuffer,
            conicsBuffer: conicsBuffer,
            opacitiesBuffer: opacitiesBuffer,
            coverageBuffer: tileCountsBuffer,
            offsetsBuffer: tileOffsetsBuffer,
            partialSumsBuffer: partialSumsTile,
            tileAssignmentHeader: headerBinned,
            tileIndicesBuffer: tileIndicesBinned,
            tileIdsBuffer: tileIdsBinned,
            precision: .float32
        )

        cbTile.commit()
        cbTile.waitUntilCompleted()

        // Compare totals
        let preciseTotal = Int(preciseHeaderPtr.pointee.totalAssignments)
        let binnedTotal = Int(headerPtr.pointee.totalAssignments)
        XCTAssertEqual(preciseTotal, binnedTotal, "Assignment totals differ")

        // Compare sets of (tileId, gaussianIdx)
        let preciseIds = preciseTileIds.contents().bindMemory(to: Int32.self, capacity: maxAssignments)
        let preciseIdx = preciseTileIndices.contents().bindMemory(to: Int32.self, capacity: maxAssignments)
        var precisePairs: [(Int32, Int32)] = []
        for i in 0..<preciseTotal {
            precisePairs.append((preciseIds[i], preciseIdx[i]))
        }

        let binnedIdsPtr = tileIdsBinned.contents().bindMemory(to: Int32.self, capacity: maxAssignments)
        let binnedIdxPtr = tileIndicesBinned.contents().bindMemory(to: Int32.self, capacity: maxAssignments)
        var binnedPairs: [(Int32, Int32)] = []
        for i in 0..<binnedTotal {
            binnedPairs.append((binnedIdsPtr[i], binnedIdxPtr[i]))
        }

        precisePairs.sort { $0.0 == $1.0 ? $0.1 < $1.1 : $0.0 < $1.0 }
        binnedPairs.sort { $0.0 == $1.0 ? $0.1 < $1.1 : $0.0 < $1.0 }

        XCTAssertEqual(precisePairs.count, binnedPairs.count, "Pair counts differ")
        for i in 0..<precisePairs.count {
            XCTAssertEqual(precisePairs[i].0, binnedPairs[i].0, "Tile id mismatch at \(i)")
            XCTAssertEqual(precisePairs[i].1, binnedPairs[i].1, "Gaussian index mismatch at \(i)")
        }
    }
}

private func fillTestData(
    boundsBuffer: MTLBuffer,
    meansBuffer: MTLBuffer,
    conicsBuffer: MTLBuffer,
    opacitiesBuffer: MTLBuffer,
    tilesX: Int,
    tilesY: Int,
    imageWidth: Int,
    imageHeight: Int,
    tileWidth: Int,
    tileHeight: Int
) {
    let gaussianCount = boundsBuffer.length / MemoryLayout<SIMD4<Int32>>.stride
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

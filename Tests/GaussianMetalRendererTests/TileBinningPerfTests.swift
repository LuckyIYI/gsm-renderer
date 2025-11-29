import XCTest
import Metal
import simd
import Foundation
@testable import GaussianMetalRenderer

/// Micro-benchmarks for the tile-binning + scatter pipeline.
/// Compares GPU time against the fused coverage+scatter baseline.
final class TileBinningPerfTests: XCTestCase {
    private let sizes = [50_000, 100_000, 250_000]
    private let iterations = 6
    private let tileWidth = 32
    private let tileHeight = 16
    private let imageWidth = 1920
    private let imageHeight = 1080

    func testTileBinningGPUTime() throws {
        let renderer = Renderer.shared
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let tileBinning = try TileBinningEncoder(device: device, library: library)
        let fused = try FusedCoverageScatterEncoder(device: device, library: library)

        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight
        let tileCount = tilesX * tilesY

        var summaries: [String] = []

        for count in sizes {
            // Shared inputs
            let boundsBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD4<Int32>>.stride, options: .storageModeShared)!
            let meansBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
            let conicsBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)!
            let opacitiesBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared)!

            // Tile-binning buffers
            let tileCounts = device.makeBuffer(length: tileCount * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
            let tileOffsets = device.makeBuffer(length: (tileCount + 1) * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
            let prefixBlockSize = 256
            let prefixGrainSize = 4
            let elementsPerGroup = prefixBlockSize * prefixGrainSize
            let groups = max(1, (tileCount + elementsPerGroup - 1) / elementsPerGroup)
            let partialSums = device.makeBuffer(length: groups * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

            // Generous cap to avoid overflow in synthetic test data
            let maxAssignments = max(count * 96, tileCount * 64)
            let tileIndices = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModePrivate)!
            let tileIds = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModePrivate)!
            let header = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!

            // Fused buffers
            let fusedCoverage = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
            let fusedTileIndices = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModePrivate)!
            let fusedTileIds = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModePrivate)!
            let fusedHeader = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!

            populateTestData(
                count: count,
                boundsBuffer: boundsBuffer,
                meansBuffer: meansBuffer,
                conicsBuffer: conicsBuffer,
                opacitiesBuffer: opacitiesBuffer,
                headerBuffer: header,
                maxAssignments: maxAssignments,
                tilesX: tilesX,
                tilesY: tilesY
            )
            populateTestData(
                count: count,
                boundsBuffer: boundsBuffer,
                meansBuffer: meansBuffer,
                conicsBuffer: conicsBuffer,
                opacitiesBuffer: opacitiesBuffer,
                headerBuffer: fusedHeader,
                maxAssignments: maxAssignments,
                tilesX: tilesX,
                tilesY: tilesY
            )

            // Warmup to avoid shader compilation noise.
            for _ in 0..<2 {
                guard let cb = queue.makeCommandBuffer() else { continue }
                tileBinning.encode(
                    commandBuffer: cb,
                    gaussianCount: count,
                    tilesX: tilesX,
                    tilesY: tilesY,
                    tileWidth: tileWidth,
                    tileHeight: tileHeight,
                    boundsBuffer: boundsBuffer,
                    meansBuffer: meansBuffer,
                    conicsBuffer: conicsBuffer,
                    opacitiesBuffer: opacitiesBuffer,
                    coverageBuffer: tileCounts,
                    offsetsBuffer: tileOffsets,
                    partialSumsBuffer: partialSums,
                    tileAssignmentHeader: header,
                    tileIndicesBuffer: tileIndices,
                    tileIdsBuffer: tileIds,
                    precision: .float32
                )
                cb.commit()
                cb.waitUntilCompleted()
            }

            var binTimes: [Double] = []
            var fusedTimes: [Double] = []

            for _ in 0..<iterations {
                let headerPtr = header.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
                headerPtr.pointee.totalAssignments = 0
                headerPtr.pointee.overflow = 0

                guard let cb = queue.makeCommandBuffer() else { continue }
                tileBinning.encode(
                    commandBuffer: cb,
                    gaussianCount: count,
                    tilesX: tilesX,
                    tilesY: tilesY,
                    tileWidth: tileWidth,
                    tileHeight: tileHeight,
                    boundsBuffer: boundsBuffer,
                    meansBuffer: meansBuffer,
                    conicsBuffer: conicsBuffer,
                    opacitiesBuffer: opacitiesBuffer,
                    coverageBuffer: tileCounts,
                    offsetsBuffer: tileOffsets,
                    partialSumsBuffer: partialSums,
                    tileAssignmentHeader: header,
                    tileIndicesBuffer: tileIndices,
                    tileIdsBuffer: tileIds,
                    precision: .float32
                )
                cb.commit()
                cb.waitUntilCompleted()
                let gpuTime = cb.gpuEndTime - cb.gpuStartTime
                if gpuTime > 0 {
                    binTimes.append(gpuTime * 1000)
                }
            }

            for _ in 0..<iterations {
                let headerPtr = fusedHeader.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
                headerPtr.pointee.totalAssignments = 0
                headerPtr.pointee.overflow = 0

                guard let cb = queue.makeCommandBuffer() else { continue }
                fused.encode(
                    commandBuffer: cb,
                    gaussianCount: count,
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
                cb.commit()
                cb.waitUntilCompleted()
                let gpuTime = cb.gpuEndTime - cb.gpuStartTime
                if gpuTime > 0 {
                    fusedTimes.append(gpuTime * 1000)
                }
            }

            guard let binAvg = binTimes.average, let fusedAvg = fusedTimes.average else { continue }

            summaries.append("\(count/1000)k: tileBin \(String(format: "%.2f", binAvg))ms vs fused \(String(format: "%.2f", fusedAvg))ms (\(String(format: "%.2fx", fusedAvg / binAvg)))")

            let headerPtr = header.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
            XCTAssertEqual(headerPtr.pointee.overflow, 0, "Tile binning overflowed assignments")
            let fusedHeaderPtr = fusedHeader.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
            XCTAssertEqual(fusedHeaderPtr.pointee.overflow, 0, "Fused overflowed assignments")
        }

        XCTAssertFalse(summaries.isEmpty, "GPU timings should be recorded")
        print("[TileBinPerf] \(summaries.joined(separator: " | "))")
    }

    private func populateTestData(
        count: Int,
        boundsBuffer: MTLBuffer,
        meansBuffer: MTLBuffer,
        conicsBuffer: MTLBuffer,
        opacitiesBuffer: MTLBuffer,
        headerBuffer: MTLBuffer,
        maxAssignments: Int,
        tilesX: Int,
        tilesY: Int
    ) {
        let boundsPtr = boundsBuffer.contents().bindMemory(to: SIMD4<Int32>.self, capacity: count)
        let meansPtr = meansBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
        let conicsPtr = conicsBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        let opacitiesPtr = opacitiesBuffer.contents().bindMemory(to: Float.self, capacity: count)
        let headerPtr = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)

        for i in 0..<count {
            let cx = Float.random(in: 0..<Float(imageWidth))
            let cy = Float.random(in: 0..<Float(imageHeight))
            meansPtr[i] = SIMD2<Float>(cx, cy)

            // Mix of small and medium gaussians to stress occupancy without overflowing buffers.
            let radius = Float.random(in: 6..<96)

            let minTX = max(0, Int32((cx - radius) / Float(tileWidth)))
            let maxTX = min(Int32(tilesX - 1), Int32((cx + radius) / Float(tileWidth)))
            let minTY = max(0, Int32((cy - radius) / Float(tileHeight)))
            let maxTY = min(Int32(tilesY - 1), Int32((cy + radius) / Float(tileHeight)))
            boundsPtr[i] = SIMD4<Int32>(minTX, maxTX, minTY, maxTY)

            let invR2 = 1.0 / max(radius * radius, 1.0)
            conicsPtr[i] = SIMD4<Float>(invR2, 0, invR2, 0)

            opacitiesPtr[i] = Float.random(in: 0.35..<1.0)
        }

        headerPtr.pointee.totalAssignments = 0
        headerPtr.pointee.maxAssignments = UInt32(maxAssignments)
        headerPtr.pointee.paddedCount = 0
        headerPtr.pointee.overflow = 0
    }
}

private extension Array where Element == Double {
    var average: Double? {
        guard !isEmpty else { return nil }
        return reduce(0, +) / Double(count)
    }
}

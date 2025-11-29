import XCTest
import Metal
import simd
import Foundation
@testable import GaussianMetalRenderer

/// Comprehensive performance comparison of all scatter methods:
/// 1. Fused V1 (original) - large TG memory, threadgroup atomics
/// 2. Fused V2 (SIMD optimized) - reduced TG memory, SIMD ballot
/// 3. Tile Binning (Tellusim-style) - per-tile counts + prefix scan
final class AllScatterMethodsPerfTests: XCTestCase {
    private let sizes = [50_000, 100_000, 250_000]
    private let iterations = 8
    private let tileWidth = 16
    private let tileHeight = 16
    private let imageWidth = 1920
    private let imageHeight = 1080

    func testAllMethodsComparison() throws {
        let renderer = Renderer.shared
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let encoderV1 = try FusedCoverageScatterEncoder(device: device, library: library)
        let encoderV2 = try FusedCoverageScatterEncoderV2(device: device, library: library)
        let tileBinningEncoder = try TileBinningEncoder(device: device, library: library)

        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight
        let tileCount = tilesX * tilesY

        var summaries: [String] = []

        for count in sizes {
            let boundsBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD4<Int32>>.stride, options: .storageModeShared)!
            let meansBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
            let conicsBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)!
            let opacitiesBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared)!

            // Fused encoders use this as coverage output
            let coverageBuffer = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

            // Tile binning uses this as per-tile counts
            let tileCountsBuffer = device.makeBuffer(length: tileCount * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

            // Prefix sum buffers for tile binning
            let offsetsBuffer = device.makeBuffer(length: (tileCount + 1) * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
            let partialSumsBuffer = device.makeBuffer(length: 1024 * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

            let maxAssignments = count * 24
            let tileIndicesBuffer = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModePrivate)!
            let tileIdsBuffer = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModePrivate)!
            let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!

            populateTestData(
                count: count,
                boundsBuffer: boundsBuffer,
                meansBuffer: meansBuffer,
                conicsBuffer: conicsBuffer,
                opacitiesBuffer: opacitiesBuffer,
                headerBuffer: headerBuffer,
                maxAssignments: maxAssignments,
                tilesX: tilesX,
                tilesY: tilesY
            )

            // Warm up all encoders
            for _ in 0..<2 {
                // V1
                if let cb = queue.makeCommandBuffer() {
                    resetHeader(headerBuffer, maxAssignments: maxAssignments)
                    encoderV1.encode(
                        commandBuffer: cb, gaussianCount: count, tileWidth: tileWidth, tileHeight: tileHeight,
                        tilesX: tilesX, maxAssignments: maxAssignments, boundsBuffer: boundsBuffer,
                        coverageBuffer: coverageBuffer, opacitiesBuffer: opacitiesBuffer, meansBuffer: meansBuffer,
                        conicsBuffer: conicsBuffer, tileIndicesBuffer: tileIndicesBuffer, tileIdsBuffer: tileIdsBuffer,
                        tileAssignmentHeader: headerBuffer, precision: .float32
                    )
                    cb.commit(); cb.waitUntilCompleted()
                }
                // V2
                if let cb = queue.makeCommandBuffer() {
                    resetHeader(headerBuffer, maxAssignments: maxAssignments)
                    encoderV2.encode(
                        commandBuffer: cb, gaussianCount: count, tileWidth: tileWidth, tileHeight: tileHeight,
                        tilesX: tilesX, maxAssignments: maxAssignments, boundsBuffer: boundsBuffer,
                        coverageBuffer: coverageBuffer, opacitiesBuffer: opacitiesBuffer, meansBuffer: meansBuffer,
                        conicsBuffer: conicsBuffer, tileIndicesBuffer: tileIndicesBuffer, tileIdsBuffer: tileIdsBuffer,
                        tileAssignmentHeader: headerBuffer, precision: .float32
                    )
                    cb.commit(); cb.waitUntilCompleted()
                }
                // Tile Binning
                if let cb = queue.makeCommandBuffer() {
                    resetHeader(headerBuffer, maxAssignments: maxAssignments)
                    clearBuffer(tileCountsBuffer, count: tileCount)
                    tileBinningEncoder.encode(
                        commandBuffer: cb, gaussianCount: count, tilesX: tilesX, tilesY: tilesY,
                        tileWidth: tileWidth, tileHeight: tileHeight, boundsBuffer: boundsBuffer,
                        meansBuffer: meansBuffer, conicsBuffer: conicsBuffer, opacitiesBuffer: opacitiesBuffer,
                        coverageBuffer: tileCountsBuffer, offsetsBuffer: offsetsBuffer,
                        partialSumsBuffer: partialSumsBuffer, tileAssignmentHeader: headerBuffer,
                        tileIndicesBuffer: tileIndicesBuffer, tileIdsBuffer: tileIdsBuffer, precision: .float32
                    )
                    cb.commit(); cb.waitUntilCompleted()
                }
            }

            // Benchmark V1
            var v1Times: [Double] = []
            for _ in 0..<iterations {
                resetHeader(headerBuffer, maxAssignments: maxAssignments)
                guard let cb = queue.makeCommandBuffer() else { continue }
                encoderV1.encode(
                    commandBuffer: cb, gaussianCount: count, tileWidth: tileWidth, tileHeight: tileHeight,
                    tilesX: tilesX, maxAssignments: maxAssignments, boundsBuffer: boundsBuffer,
                    coverageBuffer: coverageBuffer, opacitiesBuffer: opacitiesBuffer, meansBuffer: meansBuffer,
                    conicsBuffer: conicsBuffer, tileIndicesBuffer: tileIndicesBuffer, tileIdsBuffer: tileIdsBuffer,
                    tileAssignmentHeader: headerBuffer, precision: .float32
                )
                cb.commit(); cb.waitUntilCompleted()
                let gpuTime = cb.gpuEndTime - cb.gpuStartTime
                if gpuTime > 0 { v1Times.append(gpuTime * 1000) }
            }

            // Benchmark V2
            var v2Times: [Double] = []
            for _ in 0..<iterations {
                resetHeader(headerBuffer, maxAssignments: maxAssignments)
                guard let cb = queue.makeCommandBuffer() else { continue }
                encoderV2.encode(
                    commandBuffer: cb, gaussianCount: count, tileWidth: tileWidth, tileHeight: tileHeight,
                    tilesX: tilesX, maxAssignments: maxAssignments, boundsBuffer: boundsBuffer,
                    coverageBuffer: coverageBuffer, opacitiesBuffer: opacitiesBuffer, meansBuffer: meansBuffer,
                    conicsBuffer: conicsBuffer, tileIndicesBuffer: tileIndicesBuffer, tileIdsBuffer: tileIdsBuffer,
                    tileAssignmentHeader: headerBuffer, precision: .float32
                )
                cb.commit(); cb.waitUntilCompleted()
                let gpuTime = cb.gpuEndTime - cb.gpuStartTime
                if gpuTime > 0 { v2Times.append(gpuTime * 1000) }
            }

            // Benchmark Tile Binning
            var tileBinTimes: [Double] = []
            for _ in 0..<iterations {
                resetHeader(headerBuffer, maxAssignments: maxAssignments)
                clearBuffer(tileCountsBuffer, count: tileCount)
                guard let cb = queue.makeCommandBuffer() else { continue }
                tileBinningEncoder.encode(
                    commandBuffer: cb, gaussianCount: count, tilesX: tilesX, tilesY: tilesY,
                    tileWidth: tileWidth, tileHeight: tileHeight, boundsBuffer: boundsBuffer,
                    meansBuffer: meansBuffer, conicsBuffer: conicsBuffer, opacitiesBuffer: opacitiesBuffer,
                    coverageBuffer: tileCountsBuffer, offsetsBuffer: offsetsBuffer,
                    partialSumsBuffer: partialSumsBuffer, tileAssignmentHeader: headerBuffer,
                    tileIndicesBuffer: tileIndicesBuffer, tileIdsBuffer: tileIdsBuffer, precision: .float32
                )
                cb.commit(); cb.waitUntilCompleted()
                let gpuTime = cb.gpuEndTime - cb.gpuStartTime
                if gpuTime > 0 { tileBinTimes.append(gpuTime * 1000) }
            }

            guard !v1Times.isEmpty && !v2Times.isEmpty && !tileBinTimes.isEmpty else { continue }

            let v1Avg = v1Times.reduce(0, +) / Double(v1Times.count)
            let v2Avg = v2Times.reduce(0, +) / Double(v2Times.count)
            let tileBinAvg = tileBinTimes.reduce(0, +) / Double(tileBinTimes.count)

            let v2VsV1 = v1Avg / v2Avg
            let v2VsTileBin = tileBinAvg / v2Avg

            summaries.append("""
            \(count/1000)k: V1=\(String(format: "%.2f", v1Avg))ms V2=\(String(format: "%.2f", v2Avg))ms TileBin=\(String(format: "%.2f", tileBinAvg))ms | V2 vs V1: \(String(format: "%.2fx", v2VsV1)) | V2 vs TileBin: \(String(format: "%.2fx", v2VsTileBin))
            """)
        }

        XCTAssertFalse(summaries.isEmpty, "GPU timings should be recorded")
        print("\n[AllMethods Comparison]")
        for summary in summaries {
            print(summary)
        }
        print("")
    }

    private func resetHeader(_ headerBuffer: MTLBuffer, maxAssignments: Int) {
        let headerPtr = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.totalAssignments = 0
        headerPtr.pointee.maxAssignments = UInt32(maxAssignments)
        headerPtr.pointee.paddedCount = 0
        headerPtr.pointee.overflow = 0
    }

    private func clearBuffer(_ buffer: MTLBuffer, count: Int) {
        let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: count)
        for i in 0..<count { ptr[i] = 0 }
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

            let klass = Float.random(in: 0..<1)
            let radius: Float
            if klass < 0.25 {
                radius = Float.random(in: 6..<20)
            } else if klass < 0.7 {
                radius = Float.random(in: 20..<80)
            } else {
                radius = Float.random(in: 80..<220)
            }

            let minTX = max(0, Int32((cx - radius) / Float(tileWidth)))
            let maxTX = min(Int32(tilesX - 1), Int32((cx + radius) / Float(tileWidth)))
            let minTY = max(0, Int32((cy - radius) / Float(tileHeight)))
            let maxTY = min(Int32(tilesY - 1), Int32((cy + radius) / Float(tileHeight)))
            boundsPtr[i] = SIMD4<Int32>(minTX, maxTX, minTY, maxTY)

            let invR2 = 1.0 / max(radius * radius, 1.0)
            conicsPtr[i] = SIMD4<Float>(invR2, 0, invR2, 0)

            opacitiesPtr[i] = Float.random(in: 0.4..<1.0)
        }

        headerPtr.pointee.totalAssignments = 0
        headerPtr.pointee.maxAssignments = UInt32(maxAssignments)
        headerPtr.pointee.paddedCount = 0
        headerPtr.pointee.overflow = 0
    }
}

import XCTest
import Metal
import simd
import Foundation
@testable import GaussianMetalRenderer

/// Performance comparison between V1 and V2 fused coverage scatter kernels.
/// V2 optimizations:
/// - Reduced threadgroup memory: 128 tiles (512B) vs 2048 (8KB) = ~8x occupancy
/// - SIMD ballot/prefix for tile counting - no threadgroup atomics in hot loop
/// - Warp-cooperative global allocation - one atomic per SIMD group
final class FusedCoverageScatterV2PerfTests: XCTestCase {
    private let sizes = [50_000, 100_000, 250_000]
    private let iterations = 8
    private let tileWidth = 16
    private let tileHeight = 16
    private let imageWidth = 1920
    private let imageHeight = 1080

    func testV1vsV2Performance() throws {
        let renderer = Renderer.shared
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let encoderV1 = try FusedCoverageScatterEncoder(device: device, library: library)
        let encoderV2 = try FusedCoverageScatterEncoderV2(device: device, library: library)

        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight

        var summaries: [String] = []

        for count in sizes {
            let boundsBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD4<Int32>>.stride, options: .storageModeShared)!
            let meansBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
            let conicsBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)!
            let opacitiesBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared)!
            let coverageBuffer = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

            let maxAssignments = count * 24
            let tileIndicesBuffer = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModePrivate)!
            let tileIdsBuffer = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModePrivate)!
            let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!

            populateFusedTestData(
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

            // Warm up both encoders
            for _ in 0..<2 {
                if let cb = queue.makeCommandBuffer() {
                    resetHeader(headerBuffer, maxAssignments: maxAssignments)
                    encoderV1.encode(
                        commandBuffer: cb,
                        gaussianCount: count,
                        tileWidth: tileWidth,
                        tileHeight: tileHeight,
                        tilesX: tilesX,
                        maxAssignments: maxAssignments,
                        boundsBuffer: boundsBuffer,
                        coverageBuffer: coverageBuffer,
                        opacitiesBuffer: opacitiesBuffer,
                        meansBuffer: meansBuffer,
                        conicsBuffer: conicsBuffer,
                        tileIndicesBuffer: tileIndicesBuffer,
                        tileIdsBuffer: tileIdsBuffer,
                        tileAssignmentHeader: headerBuffer,
                        precision: .float32
                    )
                    cb.commit()
                    cb.waitUntilCompleted()
                }
                if let cb = queue.makeCommandBuffer() {
                    resetHeader(headerBuffer, maxAssignments: maxAssignments)
                    encoderV2.encode(
                        commandBuffer: cb,
                        gaussianCount: count,
                        tileWidth: tileWidth,
                        tileHeight: tileHeight,
                        tilesX: tilesX,
                        maxAssignments: maxAssignments,
                        boundsBuffer: boundsBuffer,
                        coverageBuffer: coverageBuffer,
                        opacitiesBuffer: opacitiesBuffer,
                        meansBuffer: meansBuffer,
                        conicsBuffer: conicsBuffer,
                        tileIndicesBuffer: tileIndicesBuffer,
                        tileIdsBuffer: tileIdsBuffer,
                        tileAssignmentHeader: headerBuffer,
                        precision: .float32
                    )
                    cb.commit()
                    cb.waitUntilCompleted()
                }
            }

            // Benchmark V1
            var v1Times: [Double] = []
            for _ in 0..<iterations {
                resetHeader(headerBuffer, maxAssignments: maxAssignments)
                guard let cb = queue.makeCommandBuffer() else { continue }
                encoderV1.encode(
                    commandBuffer: cb,
                    gaussianCount: count,
                    tileWidth: tileWidth,
                    tileHeight: tileHeight,
                    tilesX: tilesX,
                    maxAssignments: maxAssignments,
                    boundsBuffer: boundsBuffer,
                    coverageBuffer: coverageBuffer,
                    opacitiesBuffer: opacitiesBuffer,
                    meansBuffer: meansBuffer,
                    conicsBuffer: conicsBuffer,
                    tileIndicesBuffer: tileIndicesBuffer,
                    tileIdsBuffer: tileIdsBuffer,
                    tileAssignmentHeader: headerBuffer,
                    precision: .float32
                )
                cb.commit()
                cb.waitUntilCompleted()
                let gpuTime = cb.gpuEndTime - cb.gpuStartTime
                if gpuTime > 0 { v1Times.append(gpuTime * 1000) }
            }

            // Benchmark V2
            var v2Times: [Double] = []
            for _ in 0..<iterations {
                resetHeader(headerBuffer, maxAssignments: maxAssignments)
                guard let cb = queue.makeCommandBuffer() else { continue }
                encoderV2.encode(
                    commandBuffer: cb,
                    gaussianCount: count,
                    tileWidth: tileWidth,
                    tileHeight: tileHeight,
                    tilesX: tilesX,
                    maxAssignments: maxAssignments,
                    boundsBuffer: boundsBuffer,
                    coverageBuffer: coverageBuffer,
                    opacitiesBuffer: opacitiesBuffer,
                    meansBuffer: meansBuffer,
                    conicsBuffer: conicsBuffer,
                    tileIndicesBuffer: tileIndicesBuffer,
                    tileIdsBuffer: tileIdsBuffer,
                    tileAssignmentHeader: headerBuffer,
                    precision: .float32
                )
                cb.commit()
                cb.waitUntilCompleted()
                let gpuTime = cb.gpuEndTime - cb.gpuStartTime
                if gpuTime > 0 { v2Times.append(gpuTime * 1000) }
            }

            guard !v1Times.isEmpty && !v2Times.isEmpty else { continue }

            let v1Avg = v1Times.reduce(0, +) / Double(v1Times.count)
            let v2Avg = v2Times.reduce(0, +) / Double(v2Times.count)
            let speedup = v1Avg / v2Avg
            summaries.append("\(count/1000)k: V1=\(String(format: "%.2f", v1Avg))ms V2=\(String(format: "%.2f", v2Avg))ms speedup=\(String(format: "%.2fx", speedup))")
        }

        XCTAssertFalse(summaries.isEmpty, "GPU timings should be recorded")
        print("[V1vsV2] \(summaries.joined(separator: " | "))")
    }

    private func resetHeader(_ headerBuffer: MTLBuffer, maxAssignments: Int) {
        let headerPtr = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
        headerPtr.pointee.totalAssignments = 0
        headerPtr.pointee.maxAssignments = UInt32(maxAssignments)
        headerPtr.pointee.paddedCount = 0
        headerPtr.pointee.overflow = 0
    }

    private func populateFusedTestData(
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

            // Mix of tiny (fast path) and medium/large (precise) gaussians.
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

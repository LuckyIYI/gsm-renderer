import XCTest
import Metal
import simd
import Foundation
@testable import GaussianMetalRenderer

/// Performance tests for precise coverage/scatter kernels (simd_sum vs threadgroup reduction)
final class PrecisePerfTests: XCTestCase {
    private let sizes = [50_000, 100_000, 500_000, 1_000_000]
    private let iterations = 10
    private let tileWidth = 16
    private let tileHeight = 16
    private let imageWidth = 1920
    private let imageHeight = 1080

    func testPreciseCoverageScatterPerf() throws {
        let renderer = Renderer.shared
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let coverageEncoder = try CoverageEncoder(device: device, library: library)
        let scatterEncoder = try ScatterEncoder(device: device, library: library)

        guard coverageEncoder.isPreciseAvailable && scatterEncoder.isPreciseAvailable else {
            print("[PrecisePerf] Precise kernels not available, skipping test")
            return
        }

        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight
        let tileCount = tilesX * tilesY

        var summaries: [String] = []

        for count in sizes {
            // Create buffers
            let boundsBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD4<Int32>>.stride, options: .storageModeShared)!
            let meansBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
            let conicsBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)!
            let opacitiesBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared)!
            let coverageBuffer = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
            let offsetsBuffer = device.makeBuffer(length: (count + 1) * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

            let prefixBlockSize = 256
            let prefixGrainSize = 4
            let elementsPerGroup = prefixBlockSize * prefixGrainSize
            let actualGroups = max(1, (count + elementsPerGroup - 1) / elementsPerGroup)
            let partialSumsBuffer = device.makeBuffer(length: actualGroups * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

            let maxAssignments = count * 20  // Assume avg 20 tiles per gaussian
            let tileIndicesBuffer = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModePrivate)!
            let tileIdsBuffer = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModePrivate)!
            let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
            let dispatchBuffer = device.makeBuffer(length: 3 * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

            // Fill test data - gaussians with varying AABB sizes
            fillTestData(
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

            // Warmup (multiple times to stabilize)
            for _ in 0..<3 {
                runPrecisePipeline(
                queue: queue,
                coverageEncoder: coverageEncoder,
                scatterEncoder: scatterEncoder,
                count: count,
                tilesX: tilesX,
                boundsBuffer: boundsBuffer,
                meansBuffer: meansBuffer,
                conicsBuffer: conicsBuffer,
                opacitiesBuffer: opacitiesBuffer,
                coverageBuffer: coverageBuffer,
                offsetsBuffer: offsetsBuffer,
                partialSumsBuffer: partialSumsBuffer,
                tileIndicesBuffer: tileIndicesBuffer,
                tileIdsBuffer: tileIdsBuffer,
                headerBuffer: headerBuffer,
                dispatchBuffer: dispatchBuffer
            )
            }

            // Measure
            var times: [Double] = []
            for _ in 0..<iterations {
                // Reset header
                let headerPtr = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
                headerPtr.pointee.totalAssignments = 0
                headerPtr.pointee.maxAssignments = UInt32(maxAssignments)
                headerPtr.pointee.paddedCount = 0
                headerPtr.pointee.overflow = 0

                let start = CFAbsoluteTimeGetCurrent()
                runPrecisePipeline(
                    queue: queue,
                    coverageEncoder: coverageEncoder,
                    scatterEncoder: scatterEncoder,
                    count: count,
                    tilesX: tilesX,
                    boundsBuffer: boundsBuffer,
                    meansBuffer: meansBuffer,
                    conicsBuffer: conicsBuffer,
                    opacitiesBuffer: opacitiesBuffer,
                    coverageBuffer: coverageBuffer,
                    offsetsBuffer: offsetsBuffer,
                    partialSumsBuffer: partialSumsBuffer,
                    tileIndicesBuffer: tileIndicesBuffer,
                    tileIdsBuffer: tileIdsBuffer,
                    headerBuffer: headerBuffer,
                    dispatchBuffer: dispatchBuffer
                )
                let end = CFAbsoluteTimeGetCurrent()
                times.append((end - start) * 1000.0)
            }

            let avgMs = times.reduce(0, +) / Double(times.count)
            let minMs = times.min() ?? 0
            let maxMs = times.max() ?? 0
            summaries.append("\(count/1000)k: avg=\(String(format: "%.2f", avgMs))ms min=\(String(format: "%.2f", minMs)) max=\(String(format: "%.2f", maxMs))")
        }

        print("[PrecisePerf] \(summaries.joined(separator: " | "))")
    }

    private func runPrecisePipeline(
        queue: MTLCommandQueue,
        coverageEncoder: CoverageEncoder,
        scatterEncoder: ScatterEncoder,
        count: Int,
        tilesX: Int,
        boundsBuffer: MTLBuffer,
        meansBuffer: MTLBuffer,
        conicsBuffer: MTLBuffer,
        opacitiesBuffer: MTLBuffer,
        coverageBuffer: MTLBuffer,
        offsetsBuffer: MTLBuffer,
        partialSumsBuffer: MTLBuffer,
        tileIndicesBuffer: MTLBuffer,
        tileIdsBuffer: MTLBuffer,
        headerBuffer: MTLBuffer,
        dispatchBuffer: MTLBuffer
    ) {
        guard let cb = queue.makeCommandBuffer() else { return }

        coverageEncoder.encodePrecise(
            commandBuffer: cb,
            gaussianCount: count,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            boundsBuffer: boundsBuffer,
            opacitiesBuffer: opacitiesBuffer,
            meansBuffer: meansBuffer,
            conicsBuffer: conicsBuffer,
            coverageBuffer: coverageBuffer,
            offsetsBuffer: offsetsBuffer,
            partialSumsBuffer: partialSumsBuffer,
            tileAssignmentHeader: headerBuffer,
            precision: .float32
        )

        scatterEncoder.encodePrecise(
            commandBuffer: cb,
            gaussianCount: count,
            tilesX: tilesX,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            offsetsBuffer: offsetsBuffer,
            dispatchBuffer: dispatchBuffer,
            boundsBuffer: boundsBuffer,
            tileIndicesBuffer: tileIndicesBuffer,
            tileIdsBuffer: tileIdsBuffer,
            tileAssignmentHeader: headerBuffer,
            meansBuffer: meansBuffer,
            conicsBuffer: conicsBuffer,
            opacitiesBuffer: opacitiesBuffer,
            precision: .float32
        )

        cb.commit()
        cb.waitUntilCompleted()
    }

    private func fillTestData(
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

        // Mix of small (fast path) and large (precise test) gaussians
        for i in 0..<count {
            let centerX = Float.random(in: 0..<Float(imageWidth))
            let centerY = Float.random(in: 0..<Float(imageHeight))
            meansPtr[i] = SIMD2<Float>(centerX, centerY)

            // Varying radius - 30% small, 50% medium, 20% large
            let radiusClass = Float.random(in: 0..<1)
            let radius: Float
            if radiusClass < 0.3 {
                radius = Float.random(in: 8..<32)  // Small - fast path
            } else if radiusClass < 0.8 {
                radius = Float.random(in: 32..<128)  // Medium
            } else {
                radius = Float.random(in: 128..<256)  // Large
            }

            // Compute tile bounds
            let minTX = max(0, Int32((centerX - radius) / Float(tileWidth)))
            let maxTX = min(Int32(tilesX - 1), Int32((centerX + radius) / Float(tileWidth)))
            let minTY = max(0, Int32((centerY - radius) / Float(tileHeight)))
            let maxTY = min(Int32(tilesY - 1), Int32((centerY + radius) / Float(tileHeight)))
            boundsPtr[i] = SIMD4<Int32>(minTX, maxTX, minTY, maxTY)

            // Conic matrix for ellipse (simplified circular for test)
            let invR2 = 1.0 / (radius * radius)
            conicsPtr[i] = SIMD4<Float>(invR2, 0, invR2, 0)

            opacitiesPtr[i] = Float.random(in: 0.5..<1.0)
        }

        headerPtr.pointee.totalAssignments = 0
        headerPtr.pointee.maxAssignments = UInt32(maxAssignments)
        headerPtr.pointee.paddedCount = 0
        headerPtr.pointee.overflow = 0
    }
}

import XCTest
import Metal
import simd
import Foundation
@testable import GaussianMetalRenderer

/// Performance comparison between V1 (4x2 pixels/thread) and V2 (2x2 pixels/thread) render kernels.
/// V2 optimizations:
/// - 2x2 pixels per thread vs 4x2 = fewer registers
/// - 16x8 TG (128 threads) vs 8x8 (64 threads) = better occupancy
/// - Smaller SIMD batches with early exit
final class RenderV2PerfTests: XCTestCase {
    private let sizes = [500_000, 1_000_000, 2_000_000, 4_000_000]
    private let iterations = 5
    private let tileWidth = 32
    private let tileHeight = 16
    private let imageWidth = 1920
    private let imageHeight = 1080

    func testRenderV1vsV2Performance() throws {
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let fusedPipelineEncoder = try FusedPipelineEncoder(device: device, library: library)
        let scatterEncoder = try FusedCoverageScatterEncoderV2(device: device, library: library)

        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight
        let tileCount = tilesX * tilesY

        // Create output textures
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: imageWidth,
            height: imageHeight,
            mipmapped: false
        )
        colorDesc.usage = [.shaderRead, .shaderWrite]
        colorDesc.storageMode = .private

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: imageWidth,
            height: imageHeight,
            mipmapped: false
        )
        depthDesc.usage = [.shaderRead, .shaderWrite]
        depthDesc.storageMode = .private

        let alphaDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: imageWidth,
            height: imageHeight,
            mipmapped: false
        )
        alphaDesc.usage = [.shaderRead, .shaderWrite]
        alphaDesc.storageMode = .private

        guard let colorTex = device.makeTexture(descriptor: colorDesc),
              let depthTex = device.makeTexture(descriptor: depthDesc),
              let alphaTex = device.makeTexture(descriptor: alphaDesc) else {
            XCTFail("Failed to create textures")
            return
        }

        var summaries: [String] = []

        for count in sizes {
            // Allocate buffers - use smaller multiplier for large counts to avoid OOM
            let maxAssignments = count <= 1_000_000 ? count * 16 : count * 8

            let boundsBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD4<Int32>>.stride, options: .storageModeShared)!
            let meansBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
            let conicsBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)!
            let opacitiesBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared)!
            let coverageBuffer = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

            let tileIndicesBuffer = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModePrivate)!
            let tileIdsBuffer = device.makeBuffer(length: maxAssignments * MemoryLayout<Int32>.stride, options: .storageModePrivate)!
            let headerBuffer = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!

            // Per-tile headers (start/count)
            let headersBuffer = device.makeBuffer(length: tileCount * 8, options: .storageModeShared)!
            let activeTileIndicesBuffer = device.makeBuffer(length: tileCount * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
            let activeTileCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!

            // Packed gaussians (16 bytes each for half)
            let packedBuffer = device.makeBuffer(length: maxAssignments * 16, options: .storageModePrivate)!

            // Dispatch args
            let dispatchArgsBuffer = device.makeBuffer(length: 12, options: .storageModeShared)!

            // Populate test data
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

            // Build tile assignments once
            if let cb = queue.makeCommandBuffer() {
                resetHeader(headerBuffer, maxAssignments: maxAssignments)
                scatterEncoder.encode(
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
                    precision: Precision.float16
                )
                cb.commit()
                cb.waitUntilCompleted()
            }

            let headerPtr = headerBuffer.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
            let totalAssignments = Int(headerPtr.pointee.totalAssignments)

            // Build per-tile headers (mock data for perf test)
            let headers = headersBuffer.contents().bindMemory(to: UInt32.self, capacity: tileCount * 2)
            let activeTileIndices = activeTileIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: tileCount)
            let activeTileCount = activeTileCountBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)

            var activeCount: UInt32 = 0
            var offset: UInt32 = 0
            let avgPerTile = UInt32(max(1, totalAssignments / tileCount))
            for t in 0..<tileCount {
                headers[t * 2] = offset
                headers[t * 2 + 1] = avgPerTile
                if avgPerTile > 0 {
                    activeTileIndices[Int(activeCount)] = UInt32(t)
                    activeCount += 1
                }
                offset += avgPerTile
            }
            activeTileCount.pointee = activeCount

            // Prepare dispatch args
            let dispatchPtr = dispatchArgsBuffer.contents().bindMemory(to: UInt32.self, capacity: 3)
            dispatchPtr[0] = activeCount
            dispatchPtr[1] = 1
            dispatchPtr[2] = 1

            let params = RenderParams(
                width: UInt32(imageWidth),
                height: UInt32(imageHeight),
                tileWidth: UInt32(tileWidth),
                tileHeight: UInt32(tileHeight),
                tilesX: UInt32(tilesX),
                tilesY: UInt32(tilesY),
                maxPerTile: UInt32(max(1, totalAssignments / tileCount)),
                whiteBackground: 0,
                activeTileCount: activeCount,
                gaussianCount: UInt32(count)
            )

            // Warm up
            for _ in 0..<2 {
                // V1 (multi-pixel)
                fusedPipelineEncoder.renderVersion = 1
                if let cb = queue.makeCommandBuffer() {
                    fusedPipelineEncoder.encodeRenderFused(
                        commandBuffer: cb,
                        headers: headersBuffer,
                        packedGaussians: packedBuffer,
                        activeTileIndices: activeTileIndicesBuffer,
                        activeTileCount: activeTileCountBuffer,
                        colorTexture: colorTex,
                        depthTexture: depthTex,
                        alphaTexture: alphaTex,
                        params: params,
                        dispatchArgs: dispatchArgsBuffer,
                        dispatchOffset: 0,
                        precision: Precision.float16
                    )
                    cb.commit()
                    cb.waitUntilCompleted()
                }
                // V2
                fusedPipelineEncoder.renderVersion = 2
                if let cb = queue.makeCommandBuffer() {
                    fusedPipelineEncoder.encodeRenderFused(
                        commandBuffer: cb,
                        headers: headersBuffer,
                        packedGaussians: packedBuffer,
                        activeTileIndices: activeTileIndicesBuffer,
                        activeTileCount: activeTileCountBuffer,
                        colorTexture: colorTex,
                        depthTexture: depthTex,
                        alphaTexture: alphaTex,
                        params: params,
                        dispatchArgs: dispatchArgsBuffer,
                        dispatchOffset: 0,
                        precision: Precision.float16
                    )
                    cb.commit()
                    cb.waitUntilCompleted()
                }
            }

            // Benchmark V1 (multi-pixel)
            var v1Times: [Double] = []
            fusedPipelineEncoder.renderVersion = 1
            for _ in 0..<iterations {
                guard let cb = queue.makeCommandBuffer() else { continue }
                fusedPipelineEncoder.encodeRenderFused(
                    commandBuffer: cb,
                    headers: headersBuffer,
                    packedGaussians: packedBuffer,
                    activeTileIndices: activeTileIndicesBuffer,
                    activeTileCount: activeTileCountBuffer,
                    colorTexture: colorTex,
                    depthTexture: depthTex,
                    alphaTexture: alphaTex,
                    params: params,
                    dispatchArgs: dispatchArgsBuffer,
                    dispatchOffset: 0,
                    precision: Precision.float16
                )
                cb.commit()
                cb.waitUntilCompleted()
                let gpuTime = cb.gpuEndTime - cb.gpuStartTime
                if gpuTime > 0 { v1Times.append(gpuTime * 1000) }
            }

            // Benchmark V2
            var v2Times: [Double] = []
            fusedPipelineEncoder.renderVersion = 2
            for _ in 0..<iterations {
                guard let cb = queue.makeCommandBuffer() else { continue }
                fusedPipelineEncoder.encodeRenderFused(
                    commandBuffer: cb,
                    headers: headersBuffer,
                    packedGaussians: packedBuffer,
                    activeTileIndices: activeTileIndicesBuffer,
                    activeTileCount: activeTileCountBuffer,
                    colorTexture: colorTex,
                    depthTexture: depthTex,
                    alphaTexture: alphaTex,
                    params: params,
                    dispatchArgs: dispatchArgsBuffer,
                    dispatchOffset: 0,
                    precision: Precision.float16
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

            summaries.append("\(count/1000)k (\(totalAssignments/1000)k assigns): V1=\(String(format: "%.2f", v1Avg))ms V2=\(String(format: "%.2f", v2Avg))ms speedup=\(String(format: "%.2fx", speedup))")
        }

        XCTAssertFalse(summaries.isEmpty, "GPU timings should be recorded")
        print("\n[RenderV1vsV2]")
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

        // Realistic 3DGS distribution: mostly tiny splats, few medium, rare large
        for i in 0..<count {
            let cx = Float.random(in: 0..<Float(imageWidth))
            let cy = Float.random(in: 0..<Float(imageHeight))
            meansPtr[i] = SIMD2<Float>(cx, cy)

            let klass = Float.random(in: 0..<1)
            let radius: Float
            if klass < 0.70 {
                // 70% tiny splats (1-4 tiles)
                radius = Float.random(in: 4..<16)
            } else if klass < 0.92 {
                // 22% small splats (4-16 tiles)
                radius = Float.random(in: 16..<40)
            } else if klass < 0.98 {
                // 6% medium splats
                radius = Float.random(in: 40..<80)
            } else {
                // 2% large splats
                radius = Float.random(in: 80..<150)
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

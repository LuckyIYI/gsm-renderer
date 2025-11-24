import XCTest
@testable import GaussianMetalRenderer

final class PerfTimingTests: XCTestCase {
    /// Simple wall-clock measurement helper.
    private func timeMillis(_ block: () -> Void) -> Double {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        let end = CFAbsoluteTimeGetCurrent()
        return (end - start) * 1000.0
    }

    func testRenderRawTimings() {
        let renderer = Renderer(useIndirectBitonic: false)

        let configs: [(label: String, width: Int32, height: Int32, tile: Int32, count: Int, perTile: Int)] = [
            ("100k", 256, 256, 16, 100_000, 512),
            ("1M", 256, 256, 16, 1_000_000, 512),
            ("5M", 256, 256, 16, 5_000_000, 512),
        ]

        var results: [(String, Double)] = []

        for cfg in configs {
            // Build deterministic grid of gaussians.
            var means: [Float] = []
            var conics: [Float] = []
            var colors: [Float] = []
            var opacities: [Float] = []
            var depths: [Float] = []
            var radii: [Float] = []

            let tilesX = cfg.width / cfg.tile
            let tilesY = cfg.height / cfg.tile
            let perTileTarget = max(cfg.perTile, Int(ceil(Double(cfg.count) / Double(max(Int(tilesX * tilesY), 1)))))
            var placed = 0
            for ty in 0..<tilesY {
                for tx in 0..<tilesX {
                    let cx = Float(tx) * Float(cfg.tile) + Float(cfg.tile) / 2.0
                    let cy = Float(ty) * Float(cfg.tile) + Float(cfg.tile) / 2.0
                    for i in 0..<perTileTarget {
                        let jitter = Float(i) * 0.01
                        means.append(cx + jitter)
                        means.append(cy + jitter)
                        conics.append(contentsOf: [1.0, 0.0, 1.0, 0.0])
                        colors.append(contentsOf: [1.0, 1.0, 1.0])
                        opacities.append(1.0)
                        depths.append(Float(i % 8) * 0.1)
                        radii.append(Float(cfg.tile) * 0.25)
                        placed += 1
                        if placed >= cfg.count { break }
                    }
                    if placed >= cfg.count { break }
                }
                if placed >= cfg.count { break }
            }

            let count = means.count / 2
            var colorOut = [Float](repeating: 0, count: Int(cfg.width * cfg.height) * 3)
            var depthOut = [Float](repeating: 0, count: Int(cfg.width * cfg.height))
            var alphaOut = [Float](repeating: 0, count: Int(cfg.width * cfg.height))

            let params = RenderParams(
                width: UInt32(cfg.width),
                height: UInt32(cfg.height),
                tileWidth: UInt32(cfg.tile),
                tileHeight: UInt32(cfg.tile),
                tilesX: UInt32(max(Int(cfg.width / cfg.tile), 1)),
                tilesY: UInt32(max(Int(cfg.height / cfg.tile), 1)),
                maxPerTile: UInt32(perTileTarget),
                whiteBackground: 0,
                activeTileCount: 0,
                gaussianCount: UInt32(count)
            )

            // Warmup once to avoid JIT noise.
            _ = means.withUnsafeBufferPointer { meansBuf in
                conics.withUnsafeBufferPointer { conicsBuf in
                    colors.withUnsafeBufferPointer { colorsBuf in
                        opacities.withUnsafeBufferPointer { opacitiesBuf in
                            depths.withUnsafeBufferPointer { depthsBuf in
                                radii.withUnsafeBufferPointer { radiiBuf in
                                    colorOut.withUnsafeMutableBufferPointer { colorOutBuf in
                                        depthOut.withUnsafeMutableBufferPointer { depthOutBuf in
                                            alphaOut.withUnsafeMutableBufferPointer { alphaOutBuf in
                                                renderer.renderRaw(
                                                    gaussianCount: count,
                                                    meansPtr: meansBuf.baseAddress!,
                                                    conicsPtr: conicsBuf.baseAddress!,
                                                    colorsPtr: colorsBuf.baseAddress!,
                                                    opacityPtr: opacitiesBuf.baseAddress!,
                                                    depthsPtr: depthsBuf.baseAddress!,
                                                    radiiPtr: radiiBuf.baseAddress!,
                                                    colorOutPtr: colorOutBuf.baseAddress!,
                                                    depthOutPtr: depthOutBuf.baseAddress!,
                                                    alphaOutPtr: alphaOutBuf.baseAddress!,
                                                    params: params
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let elapsed = timeMillis {
                _ = means.withUnsafeBufferPointer { meansBuf in
                    conics.withUnsafeBufferPointer { conicsBuf in
                        colors.withUnsafeBufferPointer { colorsBuf in
                            opacities.withUnsafeBufferPointer { opacitiesBuf in
                                depths.withUnsafeBufferPointer { depthsBuf in
                                    radii.withUnsafeBufferPointer { radiiBuf in
                                        colorOut.withUnsafeMutableBufferPointer { colorOutBuf in
                                            depthOut.withUnsafeMutableBufferPointer { depthOutBuf in
                                                alphaOut.withUnsafeMutableBufferPointer { alphaOutBuf in
                                                    renderer.renderRaw(
                                                        gaussianCount: count,
                                                        meansPtr: meansBuf.baseAddress!,
                                                        conicsPtr: conicsBuf.baseAddress!,
                                                        colorsPtr: colorsBuf.baseAddress!,
                                                        opacityPtr: opacitiesBuf.baseAddress!,
                                                        depthsPtr: depthsBuf.baseAddress!,
                                                        radiiPtr: radiiBuf.baseAddress!,
                                                        colorOutPtr: colorOutBuf.baseAddress!,
                                                        depthOutPtr: depthOutBuf.baseAddress!,
                                                        alphaOutPtr: alphaOutBuf.baseAddress!,
                                                        params: params
                                                    )
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            results.append((cfg.label, elapsed))
        }

        // Log concise timing table.
        let table = results.map { label, ms in "\(label): \(String(format: "%.3f", ms)) ms" }
        print("[PerfTiming] renderRaw wall-times -> \(table.joined(separator: " | "))")
        XCTAssertFalse(results.isEmpty)
    }

    func testEncoderStageTimings() {
            let renderer = Renderer(useIndirectBitonic: false)
        let configs: [(label: String, width: Int32, height: Int32, tile: Int32, count: Int, perTile: Int)] = [
            ("100k", 256, 256, 16, 100_000, 512),
            ("1M", 256, 256, 16, 1_000_000, 512),
            ("5M", 256, 256, 16, 5_000_000, 512),
        ]

        var summaries: [String] = []

        for cfg in configs {
            // Build inputs
            var means: [Float] = []
            var conics: [Float] = []
            var colors: [Float] = []
            var opacities: [Float] = []
            var depths: [Float] = []
            var radii: [Float] = []

            let tilesX = cfg.width / cfg.tile
            let tilesY = cfg.height / cfg.tile
            let perTileTarget = max(cfg.perTile, Int(ceil(Double(cfg.count) / Double(max(Int(tilesX * tilesY), 1)))))
            var placed = 0
            for ty in 0..<tilesY {
                for tx in 0..<tilesX {
                    let cx = Float(tx) * Float(cfg.tile) + Float(cfg.tile) / 2.0
                    let cy = Float(ty) * Float(cfg.tile) + Float(cfg.tile) / 2.0
                    for i in 0..<perTileTarget {
                        let jitter = Float(i) * 0.01
                        means.append(cx + jitter)
                        means.append(cy + jitter)
                        conics.append(contentsOf: [1.0, 0.0, 1.0, 0.0])
                        colors.append(contentsOf: [1.0, 1.0, 1.0])
                        opacities.append(1.0)
                        depths.append(Float(i % 8) * 0.1)
                        radii.append(Float(cfg.tile) * 0.25)
                        placed += 1
                        if placed >= cfg.count { break }
                    }
                    if placed >= cfg.count { break }
                }
                if placed >= cfg.count { break }
            }
            let count = means.count / 2

            let params = RenderParams(
                width: UInt32(cfg.width),
                height: UInt32(cfg.height),
                tileWidth: UInt32(cfg.tile),
                tileHeight: UInt32(cfg.tile),
                tilesX: UInt32(max(Int(cfg.width / cfg.tile), 1)),
                tilesY: UInt32(max(Int(cfg.height / cfg.tile), 1)),
                maxPerTile: UInt32(perTileTarget),
                whiteBackground: 0,
                activeTileCount: 0,
                gaussianCount: UInt32(count)
            )

            guard let gaussianBuffers = renderer.prepareGaussianBuffers(count: count) else {
                XCTFail("Failed to prepare gaussian buffers")
                return
            }

            // Copy data into buffers
            means.withUnsafeBytes { src in
                memcpy(gaussianBuffers.means.contents(), src.baseAddress!, src.count)
            }
            radii.withUnsafeBytes { src in
                memcpy(gaussianBuffers.radii.contents(), src.baseAddress!, src.count)
            }
            opacities.withUnsafeBytes { src in
                memcpy(gaussianBuffers.opacities.contents(), src.baseAddress!, src.count)
            }
            depths.withUnsafeBytes { src in
                memcpy(gaussianBuffers.depths.contents(), src.baseAddress!, src.count)
            }
            conics.withUnsafeBytes { src in
                memcpy(gaussianBuffers.conics.contents(), src.baseAddress!, src.count)
            }
            colors.withUnsafeBytes { src in
                memcpy(gaussianBuffers.colors.contents(), src.baseAddress!, src.count)
            }

            let frame = FrameResources(device: renderer.device)
            let tileCount = Int(params.tilesX * params.tilesY)
            let perTileLimit = (params.maxPerTile == 0) ? UInt32(max(count, 1)) : params.maxPerTile
            let baseCapacity = max(tileCount * Int(perTileLimit), 1)
            let overlapCapacity = count * 8
            let forcedCapacity = max(baseCapacity, overlapCapacity)
            XCTAssertTrue(renderer.prepareTileBuilderResources(frame: frame, gaussianCount: count, tileCount: tileCount, maxPerTile: Int(perTileLimit), forcedCapacity: forcedCapacity))

            var timings: [(String, Double)] = []

            func measure(label: String, _ encode: (MTLCommandBuffer) -> Void) {
                guard let cb = renderer.queue.makeCommandBuffer() else { return }
                let ms = timeMillis {
                    encode(cb)
                    cb.commit()
                    cb.waitUntilCompleted()
                }
                timings.append((label, ms))
            }

            // Tile bounds
            measure(label: "tileBounds") { cb in
                renderer.tileBoundsEncoder.encode(commandBuffer: cb, gaussianBuffers: gaussianBuffers, boundsBuffer: frame.boundsBuffer!, params: params, gaussianCount: count)
            }

            // Coverage + offsets
            measure(label: "coverage") { cb in
                renderer.coverageEncoder.encode(commandBuffer: cb, gaussianCount: count, boundsBuffer: frame.boundsBuffer!, opacitiesBuffer: gaussianBuffers.opacities, coverageBuffer: frame.coverageBuffer!, offsetsBuffer: frame.offsetsBuffer!, partialSumsBuffer: frame.partialSumsBuffer!, tileAssignmentHeader: frame.tileAssignmentHeader!)
            }

            // Scatter
            measure(label: "scatter") { cb in
                renderer.scatterEncoder.encode(commandBuffer: cb, gaussianCount: count, tilesX: Int(params.tilesX), offsetsBuffer: frame.offsetsBuffer!, dispatchBuffer: frame.scatterDispatchBuffer!, boundsBuffer: frame.boundsBuffer!, tileIndicesBuffer: frame.tileIndices!, tileIdsBuffer: frame.tileIds!, tileAssignmentHeader: frame.tileAssignmentHeader!)
            }

            // Prepare ordered buffers
            let headerPtr = frame.tileAssignmentHeader!.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1)
            let padded = Int(headerPtr.pointee.paddedCount)
            XCTAssertTrue(renderer.prepareOrderedBuffers(frame: frame, maxAssignments: frame.tileAssignmentMaxAssignments, tileCount: tileCount, precision: .float32))
            _ = renderer.ensureBuffer(&frame.sortKeys, length: padded * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared, label: "SortKeys")
            _ = renderer.ensureBuffer(&frame.sortedIndices, length: padded * MemoryLayout<Int32>.stride, options: .storageModeShared, label: "SortedIndices")

            // Sort (dispatch + sortKeyGen + bitonic)
            measure(label: "sort") { cb in
                renderer.dispatchEncoder.encode(commandBuffer: cb, header: frame.tileAssignmentHeader!, dispatchArgs: frame.dispatchArgs!)
                if let blit = cb.makeBlitCommandEncoder() {
                    let sortBytes = padded * MemoryLayout<SIMD2<UInt32>>.stride
                    blit.fill(buffer: frame.sortKeys!, range: 0..<sortBytes, value: 0xFF)
                    let idxBytes = padded * MemoryLayout<Int32>.stride
                    blit.fill(buffer: frame.sortedIndices!, range: 0..<idxBytes, value: 0xFF)
                    blit.endEncoding()
                }
                renderer.sortKeyGenEncoder.encode(commandBuffer: cb, tileIds: frame.tileIds!, tileIndices: frame.tileIndices!, depths: gaussianBuffers.depths, sortKeys: frame.sortKeys!, sortedIndices: frame.sortedIndices!, header: frame.tileAssignmentHeader!, dispatchArgs: frame.dispatchArgs!, dispatchOffset: DispatchSlot.sortKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride)
                let offsets = (
                    first: DispatchSlot.bitonicFirst.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                    general: DispatchSlot.bitonicGeneral.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                    final: DispatchSlot.bitonicFinal.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
                )
                renderer.bitonicSortEncoder.encode(commandBuffer: cb, sortKeys: frame.sortKeys!, sortedIndices: frame.sortedIndices!, header: frame.tileAssignmentHeader!, dispatchArgs: frame.dispatchArgs!, offsets: offsets, paddedCapacity: padded)
            }

            // Pack
            measure(label: "pack") { cb in
                let orderedBuffers = OrderedBufferSet(headers: frame.orderedHeaders!, means: frame.packedMeans!, conics: frame.packedConics!, colors: frame.packedColors!, opacities: frame.packedOpacities!, depths: frame.packedDepths!)
                renderer.packEncoder.encode(commandBuffer: cb, sortedIndices: frame.sortedIndices!, sortedKeys: frame.sortKeys!, gaussianBuffers: gaussianBuffers, orderedBuffers: orderedBuffers, assignment: TileAssignmentBuffers(tileCount: tileCount, maxAssignments: frame.tileAssignmentMaxAssignments, tileIndices: frame.tileIndices!, tileIds: frame.tileIds!, header: frame.tileAssignmentHeader!), totalAssignments: Int(headerPtr.pointee.totalAssignments), dispatchArgs: frame.dispatchArgs!, dispatchOffset: DispatchSlot.pack.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride, activeTileIndices: frame.activeTileIndices!, activeTileCount: frame.activeTileCount!, precision: .float32)
            }

            // Render
            measure(label: "render") { cb in
                let pixelCount = Int(params.width * params.height)
                let color = renderer.device.makeBuffer(length: pixelCount * 12, options: .storageModeShared)!
                let depth = renderer.device.makeBuffer(length: pixelCount * 4, options: .storageModeShared)!
                let alpha = renderer.device.makeBuffer(length: pixelCount * 4, options: .storageModeShared)!
                frame.outputBuffers = RenderOutputBuffers(colorOutGPU: color, depthOutGPU: depth, alphaOutGPU: alpha)
                renderer.renderEncoder.encode(commandBuffer: cb, orderedBuffers: OrderedGaussianBuffers(headers: frame.orderedHeaders!, means: frame.packedMeans!, conics: frame.packedConics!, colors: frame.packedColors!, opacities: frame.packedOpacities!, depths: frame.packedDepths!, tileCount: tileCount, activeTileIndices: frame.activeTileIndices!, activeTileCount: frame.activeTileCount!, precision: .float32), outputBuffers: frame.outputBuffers!, params: params, dispatchArgs: frame.dispatchArgs!, dispatchOffset: DispatchSlot.renderTiles.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride, precision: .float32)
            }

            let summary = timings.map { "\($0.0)=\(String(format: "%.3f", $0.1))ms" }.joined(separator: ", ")
            summaries.append("\(cfg.label): \(summary)")
        }

        print("[PerfTiming] stage wall-times -> \(summaries.joined(separator: " | "))")
    }
}

import XCTest
@testable import GaussianMetalRenderer

/// Measures sort-stage timings (dispatch + sortKeyGen + sort) without packing/render.
final class SortStageTimingTests: XCTestCase {
    private func timeMillis(_ block: () -> Void) -> Double {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        return (CFAbsoluteTimeGetCurrent() - start) * 1000.0
    }

    func testBitonicVsRadixSortStageTimings() {
        let widths: [Int32] = [256, 256]
        let heights: [Int32] = [256, 256]
        let counts = [100_000, 1_000_000]
        let tile: Int32 = 16

        var summaries: [String] = []

        for (idx, count) in counts.enumerated() {
            let width = widths[idx]
            let height = heights[idx]
            let params = RenderParams(
                width: UInt32(width),
                height: UInt32(height),
                tileWidth: UInt32(tile),
                tileHeight: UInt32(tile),
                tilesX: UInt32(max(Int(width / tile), 1)),
                tilesY: UInt32(max(Int(height / tile), 1)),
                maxPerTile: 0,
                whiteBackground: 0,
                activeTileCount: 0,
                gaussianCount: UInt32(count)
            )

            let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
            let (means, conics, colors, opacities, depths, radii) = makeInputs(count: count, width: width, height: height, tile: tile)

            func prepareInputs() -> GaussianInputBuffers? {
                guard let buffers = renderer.prepareGaussianBuffers(count: count) else { return nil }
                _ = means.withUnsafeBytes { src in memcpy(buffers.means.contents(), src.baseAddress!, src.count) }
                _ = conics.withUnsafeBytes { src in memcpy(buffers.conics.contents(), src.baseAddress!, src.count) }
                _ = colors.withUnsafeBytes { src in memcpy(buffers.colors.contents(), src.baseAddress!, src.count) }
                _ = opacities.withUnsafeBytes { src in memcpy(buffers.opacities.contents(), src.baseAddress!, src.count) }
                _ = depths.withUnsafeBytes { src in memcpy(buffers.depths.contents(), src.baseAddress!, src.count) }
                _ = radii.withUnsafeBytes { src in memcpy(buffers.radii.contents(), src.baseAddress!, src.count) }
                memset(buffers.mask.contents(), 1, count)
                return buffers
            }

            func runSort(algorithm: SortAlgorithm) -> (ms: Double, assignments: Int, padded: Int) {
                guard let (frame, slot) = renderer.acquireFrame(width: Int(params.width), height: Int(params.height)),
                      let inputs = prepareInputs() else {
                    XCTFail("Setup failed")
                    return (.infinity, 0, 0)
                }

                defer { renderer.releaseFrame(index: slot) }

                // Tile assignment
                guard let cbAssign = renderer.queue.makeCommandBuffer() else {
                    XCTFail("No command buffer")
                    return (.infinity, 0, 0)
                }

                guard let assignment = renderer.buildTileAssignmentsGPU(
                    commandBuffer: cbAssign,
                    gaussianCount: count,
                    gaussianBuffers: inputs,
                    params: params,
                    frame: frame
                ) else {
                    XCTFail("Tile assignment failed")
                    return (.infinity, 0, 0)
                }

                cbAssign.commit()
                cbAssign.waitUntilCompleted()

                // Sort buffers
                let sortKeysBuffer = frame.sortKeys
                let sortedIndicesBuffer = frame.sortedIndices

                // Dispatch prep + keygen + sort
                guard let cbSort = renderer.queue.makeCommandBuffer() else {
                    XCTFail("No sort CB")
                    return (.infinity, 0, 0)
                }

                renderer.dispatchEncoder.encode(
                    commandBuffer: cbSort,
                    header: assignment.header,
                    dispatchArgs: frame.dispatchArgs,
                    maxAssignments: frame.tileAssignmentMaxAssignments
                )

                renderer.sortKeyGenEncoder.encode(
                    commandBuffer: cbSort,
                    tileIds: assignment.tileIds,
                    tileIndices: assignment.tileIndices,
                    depths: inputs.depths,
                    sortKeys: sortKeysBuffer,
                    sortedIndices: sortedIndicesBuffer,
                    header: assignment.header,
                    dispatchArgs: frame.dispatchArgs,
                    dispatchOffset: DispatchSlot.sortKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
                )

                let elapsed = timeMillis {
                    if algorithm == .radix {
                        let radixBuffers = RadixBufferSet(
                            histogram: frame.radixHistogram!,
                            blockSums: frame.radixBlockSums!,
                            scannedHistogram: frame.radixScannedHistogram!,
                            fusedKeys: frame.radixFusedKeys!,
                            scratchKeys: frame.radixKeysScratch!,
                            scratchPayload: frame.radixPayloadScratch!
                        )
                        let offsets = (
                            fuse: DispatchSlot.fuseKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                            unpack: DispatchSlot.unpackKeys.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                            histogram: DispatchSlot.radixHistogram.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                            scanBlocks: DispatchSlot.radixScanBlocks.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                            exclusive: DispatchSlot.radixExclusive.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                            apply: DispatchSlot.radixApply.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                            scatter: DispatchSlot.radixScatter.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
                        )
                        renderer.radixSortEncoder.encode(
                            commandBuffer: cbSort,
                            keyBuffer: sortKeysBuffer,
                            sortedIndices: sortedIndicesBuffer,
                            header: assignment.header,
                            dispatchArgs: frame.dispatchArgs,
                            radixBuffers: radixBuffers,
                            offsets: offsets,
                            tileCount: assignment.tileCount
                        )
                    } else {
                        let offsets = (
                            first: DispatchSlot.bitonicFirst.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                            general: DispatchSlot.bitonicGeneral.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride,
                            final: DispatchSlot.bitonicFinal.rawValue * MemoryLayout<DispatchIndirectArgsSwift>.stride
                        )
                        renderer.bitonicSortEncoder.encode(
                            commandBuffer: cbSort,
                            sortKeys: sortKeysBuffer,
                            sortedIndices: sortedIndicesBuffer,
                            header: assignment.header,
                            dispatchArgs: frame.dispatchArgs,
                            offsets: offsets,
                            paddedCapacity: Int(frame.tileAssignmentPaddedCapacity)
                        )
                    }
                    cbSort.commit()
                    cbSort.waitUntilCompleted()
                }

                let hdr = assignment.header.contents().bindMemory(to: TileAssignmentHeaderSwift.self, capacity: 1).pointee
                return (elapsed, Int(hdr.totalAssignments), Int(hdr.paddedCount))
            }

            let bitonic = runSort(algorithm: .bitonic)
            let radix = runSort(algorithm: .radix)
            summaries.append("\(count) assign \(radix.assignments)/pad \(radix.padded): bitonic=\(String(format: "%.3f", bitonic.ms))ms, radix=\(String(format: "%.3f", radix.ms))ms")
        }

        print("[SortStageTiming] \(summaries.joined(separator: " | "))")
    }

    private func makeInputs(count: Int, width: Int32, height: Int32, tile: Int32) -> ([Float], [Float], [Float], [Float], [Float], [Float]) {
        var means: [Float] = []
        var conics: [Float] = []
        var colors: [Float] = []
        var opacities: [Float] = []
        var depths: [Float] = []
        var radii: [Float] = []

        let tilesX = max(1, width / tile)
        let tilesY = max(1, height / tile)
        for i in 0..<count {
            let tx = i % Int(tilesX)
            let ty = (i / Int(tilesX)) % Int(tilesY)
            let cx = Float(tx) * Float(tile) + Float(tile) / 2.0
            let cy = Float(ty) * Float(tile) + Float(tile) / 2.0
            let jitter = Float(i % 8) * 0.01
            means.append(cx + jitter)
            means.append(cy + jitter)
            conics.append(contentsOf: [1.0, 0.0, 1.0, 0.0])
            colors.append(contentsOf: [1.0, 1.0, 1.0])
            opacities.append(1.0)
            depths.append(Float(i % 8) * 0.1)
            radii.append(Float(tile) * 0.25)
        }

        return (means, conics, colors, opacities, depths, radii)
    }
}

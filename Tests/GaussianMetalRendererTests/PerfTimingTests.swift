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
}

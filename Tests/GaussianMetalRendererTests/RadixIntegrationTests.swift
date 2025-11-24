import XCTest
@testable import GaussianMetalRenderer

final class RadixIntegrationTests: XCTestCase {
    func testRadixMatchesBitonicRenderOutput() {
        let width: UInt32 = 320
        let height: UInt32 = 240
        let tile: UInt32 = 16
        let count = 4_096

        var means: [Float] = []
        var conics: [Float] = []
        var colors: [Float] = []
        var opacities: [Float] = []
        var depths: [Float] = []
        var radii: [Float] = []

        var rng = SeededGenerator(seed: 42)
        for i in 0..<count {
            let mx = Float.random(in: 0..<Float(width), using: &rng)
            let my = Float.random(in: 0..<Float(height), using: &rng)
            means.append(mx)
            means.append(my)

            conics.append(contentsOf: [1.0, 0.0, 1.0, 0.0])

            colors.append(Float.random(in: 0...1, using: &rng))
            colors.append(Float.random(in: 0...1, using: &rng))
            colors.append(Float.random(in: 0...1, using: &rng))

            opacities.append(Float.random(in: 0.1...0.9, using: &rng))
            depths.append(0.5 + Float(i) * 1e-4) // Unique depths to avoid key ties.
            radii.append(6.0)
        }

        let params = RenderParams(
            width: width,
            height: height,
            tileWidth: tile,
            tileHeight: tile,
            tilesX: (width + tile - 1) / tile,
            tilesY: (height + tile - 1) / tile,
            maxPerTile: 0,
            whiteBackground: 0,
            activeTileCount: 0,
            gaussianCount: UInt32(count)
        )

        let bitonicRenderer = Renderer(useIndirectBitonic: false, sortAlgorithm: .bitonic)
        let radixRenderer = Renderer(useIndirectBitonic: false, sortAlgorithm: .radix)

        let pixelCount = Int(width * height)
        var bitonicColor = [Float](repeating: 0, count: pixelCount * 3)
        var bitonicDepth = [Float](repeating: 0, count: pixelCount)
        var bitonicAlpha = [Float](repeating: 0, count: pixelCount)

        var radixColor = bitonicColor
        var radixDepth = bitonicDepth
        var radixAlpha = bitonicAlpha

        let bitonicResult = means.withUnsafeBufferPointer { meansBuf in
            conics.withUnsafeBufferPointer { conicsBuf in
                colors.withUnsafeBufferPointer { colorsBuf in
                    opacities.withUnsafeBufferPointer { opacitiesBuf in
                        depths.withUnsafeBufferPointer { depthsBuf in
                            radii.withUnsafeBufferPointer { radiiBuf in
                                bitonicColor.withUnsafeMutableBufferPointer { colorOutBuf in
                                    bitonicDepth.withUnsafeMutableBufferPointer { depthOutBuf in
                                        bitonicAlpha.withUnsafeMutableBufferPointer { alphaOutBuf in
                                            bitonicRenderer.renderRaw(
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

        let radixResult = means.withUnsafeBufferPointer { meansBuf in
            conics.withUnsafeBufferPointer { conicsBuf in
                colors.withUnsafeBufferPointer { colorsBuf in
                    opacities.withUnsafeBufferPointer { opacitiesBuf in
                        depths.withUnsafeBufferPointer { depthsBuf in
                            radii.withUnsafeBufferPointer { radiiBuf in
                                radixColor.withUnsafeMutableBufferPointer { colorOutBuf in
                                    radixDepth.withUnsafeMutableBufferPointer { depthOutBuf in
                                        radixAlpha.withUnsafeMutableBufferPointer { alphaOutBuf in
                                            radixRenderer.renderRaw(
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

        XCTAssertEqual(bitonicResult, 0, "Bitonic render failed (\(bitonicResult))")
        XCTAssertEqual(radixResult, 0, "Radix render failed (\(radixResult))")

        // Keep a strict comparison so failures remain visible if accumulations diverge.
        func assertArraysClose(_ a: [Float], _ b: [Float], label: String, maxTolerance: Float = 1e-3, meanTolerance: Float = 1e-5, spikeThreshold: Float = 1e-4, spikeFraction: Float = 1e-4) {
            XCTAssertEqual(a.count, b.count, "\(label) size mismatch")
            var maxDiff: Float = 0
            var sumDiff: Double = 0
            var spikes = 0
            for i in 0..<a.count {
                let diff = abs(a[i] - b[i])
                if diff > maxDiff { maxDiff = diff }
                sumDiff += Double(diff)
                if diff > spikeThreshold { spikes += 1 }
            }
            let meanDiff = Float(sumDiff / Double(a.count))
            let frac = Float(spikes) / Float(a.count)
            XCTAssertLessThanOrEqual(maxDiff, maxTolerance, "\(label) max diff \(maxDiff)")
            XCTAssertLessThanOrEqual(meanDiff, meanTolerance, "\(label) mean diff \(meanDiff)")
            XCTAssertLessThanOrEqual(frac, spikeFraction, "\(label) spike fraction \(frac)")
        }

        assertArraysClose(bitonicColor, radixColor, label: "color")
        assertArraysClose(bitonicDepth, radixDepth, label: "depth")
        assertArraysClose(bitonicAlpha, radixAlpha, label: "alpha")
    }
}

// Deterministic RNG for reproducible tests.
struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        return state &* 2685821657736338717
    }
}

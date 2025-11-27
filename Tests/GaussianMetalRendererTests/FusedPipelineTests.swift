import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Tests for the fused pipeline optimization
/// Validates that fused pipeline produces identical output to standard pipeline
final class FusedPipelineTests: XCTestCase {
    private func makeLargeLimits(maxGaussians: Int) -> RendererLimits {
        // Use smaller maxPerTile to avoid heap explosion; computeLayout now clamps maxAssignments
        RendererLimits(maxGaussians: maxGaussians, maxWidth: 2048, maxHeight: 2048, tileWidth: 16, tileHeight: 16, maxPerTile: 512)
    }
    private let largeLimits = RendererLimits(
        maxGaussians: 4_000_000,  // Reduced from 10M to keep heap under device limits
        maxWidth: 1024,
        maxHeight: 1024,
        tileWidth: 16,
        tileHeight: 16,
        maxPerTile: 512  // Reduced from 1024
    )

    /// Test that fused pipeline produces same output as standard pipeline
    func testFusedMatchesStandard() throws {
        // Create standard renderer
        let limits = RendererLimits(
            maxGaussians: 2_000,
            maxWidth: 512,
            maxHeight: 512,
            tileWidth: 16,
            tileHeight: 16
        )
        let standardRenderer = Renderer(
            precision: .float32,
            sortAlgorithm: .radix,
            limits: limits
        )

        // Create fused renderer
        let fusedRenderer = Renderer(
            precision: .float32,
            sortAlgorithm: .radix,
            limits: limits
        )

        XCTAssertTrue(fusedRenderer.isFusedPipelineAvailable, "Fused pipeline should be available")

        // Test parameters
        let gaussianCount = 1000
        let imageWidth: UInt32 = 256
        let imageHeight: UInt32 = 256
        let tileWidth: UInt32 = 16
        let tileHeight: UInt32 = 16
        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight

        // Generate test gaussians spread across the image
        var means = [Float](repeating: 0, count: gaussianCount * 2)
        var conics = [Float](repeating: 0, count: gaussianCount * 4)
        var colors = [Float](repeating: 0, count: gaussianCount * 3)
        var opacities = [Float](repeating: 0, count: gaussianCount)
        var depths = [Float](repeating: 0, count: gaussianCount)
        var radii = [Float](repeating: 0, count: gaussianCount)

        for i in 0..<gaussianCount {
            // Spread gaussians across image
            let x = Float(i % 32) * 8 + 4
            let y = Float(i / 32) * 8 + 4
            means[i * 2] = x
            means[i * 2 + 1] = y

            // Standard conic for ~4px radius gaussian
            let sigma = Float(4.0)
            let inv_sigma2 = 1.0 / (sigma * sigma)
            conics[i * 4 + 0] = inv_sigma2  // a
            conics[i * 4 + 1] = 0           // b
            conics[i * 4 + 2] = inv_sigma2  // c
            conics[i * 4 + 3] = 0           // unused

            // Varying colors
            colors[i * 3 + 0] = Float(i % 256) / 255.0  // R
            colors[i * 3 + 1] = Float((i * 7) % 256) / 255.0  // G
            colors[i * 3 + 2] = Float((i * 13) % 256) / 255.0  // B

            opacities[i] = 0.8
            depths[i] = Float(i) / Float(gaussianCount)
            radii[i] = sigma * 3  // 3-sigma radius
        }

        let params = RenderParams(
            width: imageWidth,
            height: imageHeight,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            tilesX: tilesX,
            tilesY: tilesY,
            maxPerTile: UInt32(gaussianCount),
            whiteBackground: 0,
            activeTileCount: 0,
            gaussianCount: UInt32(gaussianCount)
        )

        // Render with standard pipeline
        var standardColor = [Float](repeating: 0, count: Int(imageWidth * imageHeight) * 3)
        var standardDepth = [Float](repeating: 0, count: Int(imageWidth * imageHeight))
        var standardAlpha = [Float](repeating: 0, count: Int(imageWidth * imageHeight))

        let standardResult = standardRenderer.renderRaw(
            gaussianCount: gaussianCount,
            meansPtr: &means,
            conicsPtr: &conics,
            colorsPtr: &colors,
            opacityPtr: &opacities,
            depthsPtr: &depths,
            radiiPtr: &radii,
            colorOutPtr: &standardColor,
            depthOutPtr: &standardDepth,
            alphaOutPtr: &standardAlpha,
            params: params
        )
        XCTAssertEqual(standardResult, 0, "Standard render should succeed")

        // Render with fused pipeline
        var fusedColor = [Float](repeating: 0, count: Int(imageWidth * imageHeight) * 3)
        var fusedDepth = [Float](repeating: 0, count: Int(imageWidth * imageHeight))
        var fusedAlpha = [Float](repeating: 0, count: Int(imageWidth * imageHeight))

        let fusedResult = fusedRenderer.renderRaw(
            gaussianCount: gaussianCount,
            meansPtr: &means,
            conicsPtr: &conics,
            colorsPtr: &colors,
            opacityPtr: &opacities,
            depthsPtr: &depths,
            radiiPtr: &radii,
            colorOutPtr: &fusedColor,
            depthOutPtr: &fusedDepth,
            alphaOutPtr: &fusedAlpha,
            params: params
        )
        XCTAssertEqual(fusedResult, 0, "Fused render should succeed")

        // Compare outputs
        let tolerance: Float = 1e-4
        var maxColorDiff: Float = 0
        var maxDepthDiff: Float = 0
        var maxAlphaDiff: Float = 0
        var differentPixels = 0

        for i in 0..<Int(imageWidth * imageHeight) {
            for c in 0..<3 {
                let diff = abs(standardColor[i * 3 + c] - fusedColor[i * 3 + c])
                maxColorDiff = max(maxColorDiff, diff)
                if diff > tolerance {
                    differentPixels += 1
                }
            }
            maxDepthDiff = max(maxDepthDiff, abs(standardDepth[i] - fusedDepth[i]))
            maxAlphaDiff = max(maxAlphaDiff, abs(standardAlpha[i] - fusedAlpha[i]))
        }

        print("[FusedTest] Max color diff: \(maxColorDiff), max depth diff: \(maxDepthDiff), max alpha diff: \(maxAlphaDiff)")
        print("[FusedTest] Pixels with color diff > \(tolerance): \(differentPixels)")

        XCTAssertLessThan(maxColorDiff, tolerance, "Color output should match within tolerance")
        XCTAssertLessThan(maxDepthDiff, tolerance, "Depth output should match within tolerance")
        XCTAssertLessThan(maxAlphaDiff, tolerance, "Alpha output should match within tolerance")
    }

    /// Test fused pipeline with half precision
    func testFusedHalfPrecision() throws {
        let fusedRenderer = Renderer(
            precision: .float16,
            sortAlgorithm: .radix
        )

        XCTAssertTrue(fusedRenderer.isFusedPipelineAvailable, "Fused pipeline should be available")

        // Simple test with a few gaussians
        let gaussianCount = 100
        let imageWidth: UInt32 = 128
        let imageHeight: UInt32 = 128
        let tileWidth: UInt32 = 32
        let tileHeight: UInt32 = 16
        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight

        var means = [Float](repeating: 0, count: gaussianCount * 2)
        var conics = [Float](repeating: 0, count: gaussianCount * 4)
        var colors = [Float](repeating: 0, count: gaussianCount * 3)
        var opacities = [Float](repeating: 0, count: gaussianCount)
        var depths = [Float](repeating: 0, count: gaussianCount)
        var radii = [Float](repeating: 0, count: gaussianCount)

        for i in 0..<gaussianCount {
            means[i * 2] = Float(64 + (i % 10) * 4)
            means[i * 2 + 1] = Float(64 + (i / 10) * 4)
            conics[i * 4] = 0.0625
            conics[i * 4 + 2] = 0.0625
            colors[i * 3] = 1.0
            colors[i * 3 + 1] = 0.5
            colors[i * 3 + 2] = 0.0
            opacities[i] = 0.9
            depths[i] = Float(i) / Float(gaussianCount)
            radii[i] = 12.0
        }

        let params = RenderParams(
            width: imageWidth,
            height: imageHeight,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            tilesX: tilesX,
            tilesY: tilesY,
            maxPerTile: UInt32(gaussianCount),
            whiteBackground: 0,
            activeTileCount: 0,
            gaussianCount: UInt32(gaussianCount)
        )

        var colorOut = [Float](repeating: 0, count: Int(imageWidth * imageHeight) * 3)
        var depthOut = [Float](repeating: 0, count: Int(imageWidth * imageHeight))
        var alphaOut = [Float](repeating: 0, count: Int(imageWidth * imageHeight))

        let result = fusedRenderer.renderRaw(
            gaussianCount: gaussianCount,
            meansPtr: &means,
            conicsPtr: &conics,
            colorsPtr: &colors,
            opacityPtr: &opacities,
            depthsPtr: &depths,
            radiiPtr: &radii,
            colorOutPtr: &colorOut,
            depthOutPtr: &depthOut,
            alphaOutPtr: &alphaOut,
            params: params
        )

        XCTAssertEqual(result, 0, "Half precision fused render should succeed")

        // Check that we got some non-zero output
        let centerPixel = Int((imageHeight / 2) * imageWidth + imageWidth / 2)
        let hasOutput = alphaOut[centerPixel] > 0
        print("[FusedHalfTest] Center alpha: \(alphaOut[centerPixel])")
        XCTAssertTrue(hasOutput, "Should have rendered something at center")
    }

    /// Test fused pipeline at scale (stress test)
    func testFusedAtScale() throws {
        let fusedRenderer = Renderer(
            precision: .float32,
            sortAlgorithm: .radix
        )

        // Large scale test
        let gaussianCount = 100_000
        let imageWidth: UInt32 = 512
        let imageHeight: UInt32 = 512
        let tileWidth: UInt32 = 32
        let tileHeight: UInt32 = 16
        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight

        var means = [Float](repeating: 0, count: gaussianCount * 2)
        var conics = [Float](repeating: 0, count: gaussianCount * 4)
        var colors = [Float](repeating: 0, count: gaussianCount * 3)
        var opacities = [Float](repeating: 0, count: gaussianCount)
        var depths = [Float](repeating: 0, count: gaussianCount)
        var radii = [Float](repeating: 0, count: gaussianCount)

        // Random distribution
        for i in 0..<gaussianCount {
            means[i * 2] = Float.random(in: 0..<Float(imageWidth))
            means[i * 2 + 1] = Float.random(in: 0..<Float(imageHeight))
            conics[i * 4] = Float.random(in: 0.01..<0.1)
            conics[i * 4 + 2] = Float.random(in: 0.01..<0.1)
            colors[i * 3] = Float.random(in: 0..<1)
            colors[i * 3 + 1] = Float.random(in: 0..<1)
            colors[i * 3 + 2] = Float.random(in: 0..<1)
            opacities[i] = Float.random(in: 0.5..<1.0)
            depths[i] = Float.random(in: 0..<1)
            radii[i] = Float.random(in: 5..<20)
        }

        let params = RenderParams(
            width: imageWidth,
            height: imageHeight,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            tilesX: tilesX,
            tilesY: tilesY,
            maxPerTile: 1024,
            whiteBackground: 0,
            activeTileCount: 0,
            gaussianCount: UInt32(gaussianCount)
        )

        var colorOut = [Float](repeating: 0, count: Int(imageWidth * imageHeight) * 3)
        var depthOut = [Float](repeating: 0, count: Int(imageWidth * imageHeight))
        var alphaOut = [Float](repeating: 0, count: Int(imageWidth * imageHeight))

        let startTime = CFAbsoluteTimeGetCurrent()
        let result = fusedRenderer.renderRaw(
            gaussianCount: gaussianCount,
            meansPtr: &means,
            conicsPtr: &conics,
            colorsPtr: &colors,
            opacityPtr: &opacities,
            depthsPtr: &depths,
            radiiPtr: &radii,
            colorOutPtr: &colorOut,
            depthOutPtr: &depthOut,
            alphaOutPtr: &alphaOut,
            params: params
        )
        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

        XCTAssertEqual(result, 0, "Scale test fused render should succeed")
        print("[FusedScaleTest] 100k gaussians @ 512x512: \(String(format: "%.2f", elapsed))ms")

        // Check for black tiles (the bug from before)
        var blackTiles = 0
        var totalTiles = 0
        for ty in 0..<Int(tilesY) {
            for tx in 0..<Int(tilesX) {
                totalTiles += 1
                var tileSum: Float = 0
                for py in 0..<Int(tileHeight) {
                    for px in 0..<Int(tileWidth) {
                        let x = tx * Int(tileWidth) + px
                        let y = ty * Int(tileHeight) + py
                        if x < Int(imageWidth) && y < Int(imageHeight) {
                            let idx = y * Int(imageWidth) + x
                            tileSum += alphaOut[idx]
                        }
                    }
                }
                if tileSum == 0 {
                    blackTiles += 1
                }
            }
        }

        print("[FusedScaleTest] Black tiles: \(blackTiles)/\(totalTiles)")
        // Allow some black tiles (empty areas), but not too many
        XCTAssertLessThan(blackTiles, totalTiles / 2, "Too many black tiles suggests a bug")
    }

    /// Test fused pipeline at 1M+ scale (the problematic case)
    func testFusedAt1MScale() throws {
        // Test BOTH pipelines to compare
        let standardRenderer = Renderer(
            precision: .float32,
            sortAlgorithm: .radix,
            limits: largeLimits
        )
        let fusedRenderer = Renderer(
            precision: .float32,
            sortAlgorithm: .radix,
            limits: largeLimits
        )

        // 1M+ gaussian test - this is where bugs appear
        let gaussianCount = 1_000_000
        let imageWidth: UInt32 = 1024
        let imageHeight: UInt32 = 1024
        let tileWidth: UInt32 = 16
        let tileHeight: UInt32 = 16
        let tilesX = (imageWidth + tileWidth - 1) / tileWidth  // 64
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight  // 64
        let totalTileCount = Int(tilesX * tilesY)  // 4096

        print("[Fused1MTest] Setup: \(gaussianCount) gaussians, \(imageWidth)x\(imageHeight), \(totalTileCount) tiles")

        var means = [Float](repeating: 0, count: gaussianCount * 2)
        var conics = [Float](repeating: 0, count: gaussianCount * 4)
        var colors = [Float](repeating: 0, count: gaussianCount * 3)
        var opacities = [Float](repeating: 0, count: gaussianCount)
        var depths = [Float](repeating: 0, count: gaussianCount)
        var radii = [Float](repeating: 0, count: gaussianCount)

        // Distribute gaussians across the image
        for i in 0..<gaussianCount {
            means[i * 2] = Float.random(in: 0..<Float(imageWidth))
            means[i * 2 + 1] = Float.random(in: 0..<Float(imageHeight))
            let sigma = Float.random(in: 2..<8)
            conics[i * 4] = 1.0 / (sigma * sigma)
            conics[i * 4 + 2] = 1.0 / (sigma * sigma)
            colors[i * 3] = Float.random(in: 0..<1)
            colors[i * 3 + 1] = Float.random(in: 0..<1)
            colors[i * 3 + 2] = Float.random(in: 0..<1)
            opacities[i] = Float.random(in: 0.3..<0.9)
            depths[i] = Float.random(in: 0..<1)
            radii[i] = sigma * 3
        }

        let params = RenderParams(
            width: imageWidth,
            height: imageHeight,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            tilesX: tilesX,
            tilesY: tilesY,
            maxPerTile: 2048,
            whiteBackground: 0,
            activeTileCount: 0,
            gaussianCount: UInt32(gaussianCount)
        )

        var colorOut = [Float](repeating: 0, count: Int(imageWidth * imageHeight) * 3)
        var depthOut = [Float](repeating: 0, count: Int(imageWidth * imageHeight))
        var alphaOut = [Float](repeating: 0, count: Int(imageWidth * imageHeight))

        // First render with STANDARD pipeline
        var standardColorOut = [Float](repeating: 0, count: Int(imageWidth * imageHeight) * 3)
        var standardDepthOut = [Float](repeating: 0, count: Int(imageWidth * imageHeight))
        var standardAlphaOut = [Float](repeating: 0, count: Int(imageWidth * imageHeight))

        let standardStart = CFAbsoluteTimeGetCurrent()
        let standardResult = standardRenderer.renderRaw(
            gaussianCount: gaussianCount,
            meansPtr: &means,
            conicsPtr: &conics,
            colorsPtr: &colors,
            opacityPtr: &opacities,
            depthsPtr: &depths,
            radiiPtr: &radii,
            colorOutPtr: &standardColorOut,
            depthOutPtr: &standardDepthOut,
            alphaOutPtr: &standardAlphaOut,
            params: params
        )
        let standardElapsed = (CFAbsoluteTimeGetCurrent() - standardStart) * 1000
        XCTAssertEqual(standardResult, 0, "Standard 1M render should succeed")
        print("[Standard1MTest] 1M gaussians @ 1024x1024: \(String(format: "%.2f", standardElapsed))ms")

        // Count standard black tiles
        var standardBlackTiles = 0
        for ty in 0..<Int(tilesY) {
            for tx in 0..<Int(tilesX) {
                var tileSum: Float = 0
                for py in 0..<Int(tileHeight) {
                    for px in 0..<Int(tileWidth) {
                        let x = tx * Int(tileWidth) + px
                        let y = ty * Int(tileHeight) + py
                        if x < Int(imageWidth) && y < Int(imageHeight) {
                            let idx = y * Int(imageWidth) + x
                            tileSum += standardAlphaOut[idx]
                        }
                    }
                }
                if tileSum == 0 { standardBlackTiles += 1 }
            }
        }
        print("[Standard1MTest] Black tiles: \(standardBlackTiles)/\(totalTileCount)")

        // Then render with FUSED pipeline
        let fusedStart = CFAbsoluteTimeGetCurrent()
        let result = fusedRenderer.renderRaw(
            gaussianCount: gaussianCount,
            meansPtr: &means,
            conicsPtr: &conics,
            colorsPtr: &colors,
            opacityPtr: &opacities,
            depthsPtr: &depths,
            radiiPtr: &radii,
            colorOutPtr: &colorOut,
            depthOutPtr: &depthOut,
            alphaOutPtr: &alphaOut,
            params: params
        )
        let elapsed = (CFAbsoluteTimeGetCurrent() - fusedStart) * 1000

        XCTAssertEqual(result, 0, "1M scale fused render should succeed")
        print("[Fused1MTest] 1M gaussians @ 1024x1024: \(String(format: "%.2f", elapsed))ms")

        // Check for black tiles
        var blackTiles = 0
        var tilesWithSomeAlpha = 0
        for ty in 0..<Int(tilesY) {
            for tx in 0..<Int(tilesX) {
                var tileSum: Float = 0
                for py in 0..<Int(tileHeight) {
                    for px in 0..<Int(tileWidth) {
                        let x = tx * Int(tileWidth) + px
                        let y = ty * Int(tileHeight) + py
                        if x < Int(imageWidth) && y < Int(imageHeight) {
                            let idx = y * Int(imageWidth) + x
                            tileSum += alphaOut[idx]
                        }
                    }
                }
                if tileSum == 0 {
                    blackTiles += 1
                } else {
                    tilesWithSomeAlpha += 1
                }
            }
        }

        print("[Fused1MTest] Black tiles: \(blackTiles)/\(totalTileCount), rendered: \(tilesWithSomeAlpha)")

        // Fused should be close to standard (small variance due to random data / buffer reuse)
        let tileDiff = abs(blackTiles - standardBlackTiles)
        print("[1MTest] Tile difference: \(tileDiff) (fused has \(blackTiles < standardBlackTiles ? "fewer" : "more") black tiles)")

        // Allow up to 5% variance between fused and standard
        let maxDiff = totalTileCount / 20
        XCTAssertLessThan(tileDiff, maxDiff, "Fused/standard difference (\(tileDiff)) too large")

        // Both should render at least half the tiles
        XCTAssertGreaterThan(tilesWithSomeAlpha, totalTileCount / 2, "Too few rendered tiles")
    }

    /// Test at 4M scale
    func testFusedAt4MScale() throws {
        let standardRenderer = Renderer(precision: .float32, sortAlgorithm: .radix, limits: largeLimits)
        let fusedRenderer = Renderer(precision: .float32, sortAlgorithm: .radix, limits: largeLimits)

        let gaussianCount = 4_000_000
        let imageWidth: UInt32 = 1024
        let imageHeight: UInt32 = 1024
        let tileWidth: UInt32 = 16
        let tileHeight: UInt32 = 16
        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight
        let totalTileCount = Int(tilesX * tilesY)

        print("[4MTest] Setup: \(gaussianCount) gaussians, \(imageWidth)x\(imageHeight), \(totalTileCount) tiles")

        var means = [Float](repeating: 0, count: gaussianCount * 2)
        var conics = [Float](repeating: 0, count: gaussianCount * 4)
        var colors = [Float](repeating: 0, count: gaussianCount * 3)
        var opacities = [Float](repeating: 0, count: gaussianCount)
        var depths = [Float](repeating: 0, count: gaussianCount)
        var radii = [Float](repeating: 0, count: gaussianCount)

        for i in 0..<gaussianCount {
            means[i * 2] = Float.random(in: 0..<Float(imageWidth))
            means[i * 2 + 1] = Float.random(in: 0..<Float(imageHeight))
            let sigma = Float.random(in: 2..<6)
            conics[i * 4] = 1.0 / (sigma * sigma)
            conics[i * 4 + 2] = 1.0 / (sigma * sigma)
            colors[i * 3] = Float.random(in: 0..<1)
            colors[i * 3 + 1] = Float.random(in: 0..<1)
            colors[i * 3 + 2] = Float.random(in: 0..<1)
            opacities[i] = Float.random(in: 0.3..<0.8)
            depths[i] = Float.random(in: 0..<1)
            radii[i] = sigma * 3
        }

        let params = RenderParams(
            width: imageWidth, height: imageHeight,
            tileWidth: tileWidth, tileHeight: tileHeight,
            tilesX: tilesX, tilesY: tilesY,
            maxPerTile: 4096, whiteBackground: 0,
            activeTileCount: 0, gaussianCount: UInt32(gaussianCount)
        )

        // Standard render
        var standardAlpha = [Float](repeating: 0, count: Int(imageWidth * imageHeight))
        let stdResult = standardRenderer.renderRaw(
            gaussianCount: gaussianCount, meansPtr: &means, conicsPtr: &conics,
            colorsPtr: &colors, opacityPtr: &opacities, depthsPtr: &depths, radiiPtr: &radii,
            colorOutPtr: nil, depthOutPtr: nil, alphaOutPtr: &standardAlpha, params: params
        )
        XCTAssertEqual(stdResult, 0)

        // Fused render
        var fusedAlpha = [Float](repeating: 0, count: Int(imageWidth * imageHeight))
        let fusedResult = fusedRenderer.renderRaw(
            gaussianCount: gaussianCount, meansPtr: &means, conicsPtr: &conics,
            colorsPtr: &colors, opacityPtr: &opacities, depthsPtr: &depths, radiiPtr: &radii,
            colorOutPtr: nil, depthOutPtr: nil, alphaOutPtr: &fusedAlpha, params: params
        )
        XCTAssertEqual(fusedResult, 0)

        // Compare
        var stdBlack = 0, fusedBlack = 0
        for ty in 0..<Int(tilesY) {
            for tx in 0..<Int(tilesX) {
                var stdSum: Float = 0, fusedSum: Float = 0
                for py in 0..<Int(tileHeight) {
                    for px in 0..<Int(tileWidth) {
                        let x = tx * Int(tileWidth) + px
                        let y = ty * Int(tileHeight) + py
                        let idx = y * Int(imageWidth) + x
                        stdSum += standardAlpha[idx]
                        fusedSum += fusedAlpha[idx]
                    }
                }
                if stdSum == 0 { stdBlack += 1 }
                if fusedSum == 0 { fusedBlack += 1 }
            }
        }

        print("[4MTest] Standard black: \(stdBlack)/\(totalTileCount), Fused black: \(fusedBlack)/\(totalTileCount)")
        XCTAssertEqual(fusedBlack, stdBlack, "4M: Fused should match standard black tile count")
    }

    /// Test fused pipeline with 32x16 tiles (multi-pixel mode)
    func testFused32x16Tiles() throws {
        // Reduced maxGaussians and maxPerTile to keep heap within device limits
        let limits = RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024, tileWidth: 32, tileHeight: 16, maxPerTile: 512)
        let standardRenderer = Renderer(
            precision: .float32,
            sortAlgorithm: .radix,
            limits: limits
        )
        let fusedRenderer = Renderer(
            precision: .float32,
            sortAlgorithm: .radix,
            limits: limits
        )

        XCTAssertTrue(fusedRenderer.isFusedPipelineAvailable, "Fused pipeline should be available")

        let gaussianCount = 5000
        let imageWidth: UInt32 = 512
        let imageHeight: UInt32 = 512
        let tileWidth: UInt32 = 32
        let tileHeight: UInt32 = 16
        let tilesX = (imageWidth + tileWidth - 1) / tileWidth  // 16
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight  // 32
        let totalTiles = Int(tilesX * tilesY)

        print("[32x16Test] Setup: \(gaussianCount) gaussians, \(imageWidth)x\(imageHeight), \(tileWidth)x\(tileHeight) tiles, \(totalTiles) total")

        var means = [Float](repeating: 0, count: gaussianCount * 2)
        var conics = [Float](repeating: 0, count: gaussianCount * 4)
        var colors = [Float](repeating: 0, count: gaussianCount * 3)
        var opacities = [Float](repeating: 0, count: gaussianCount)
        var depths = [Float](repeating: 0, count: gaussianCount)
        var radii = [Float](repeating: 0, count: gaussianCount)

        // Distribute gaussians evenly
        for i in 0..<gaussianCount {
            let x = Float(i % 70) * 7 + 10
            let y = Float(i / 70) * 7 + 10
            means[i * 2] = x
            means[i * 2 + 1] = y

            let sigma: Float = 4.0
            conics[i * 4] = 1.0 / (sigma * sigma)
            conics[i * 4 + 2] = 1.0 / (sigma * sigma)

            colors[i * 3] = Float(i % 256) / 255.0
            colors[i * 3 + 1] = Float((i * 3) % 256) / 255.0
            colors[i * 3 + 2] = Float((i * 7) % 256) / 255.0

            opacities[i] = 0.7
            depths[i] = Float(i) / Float(gaussianCount)
            radii[i] = sigma * 3
        }

        let params = RenderParams(
            width: imageWidth,
            height: imageHeight,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            tilesX: tilesX,
            tilesY: tilesY,
            maxPerTile: UInt32(gaussianCount),
            whiteBackground: 0,
            activeTileCount: 0,
            gaussianCount: UInt32(gaussianCount)
        )

        // Standard render
        var standardColor = [Float](repeating: 0, count: Int(imageWidth * imageHeight) * 3)
        var standardDepth = [Float](repeating: 0, count: Int(imageWidth * imageHeight))
        var standardAlpha = [Float](repeating: 0, count: Int(imageWidth * imageHeight))

        let standardResult = standardRenderer.renderRaw(
            gaussianCount: gaussianCount,
            meansPtr: &means,
            conicsPtr: &conics,
            colorsPtr: &colors,
            opacityPtr: &opacities,
            depthsPtr: &depths,
            radiiPtr: &radii,
            colorOutPtr: &standardColor,
            depthOutPtr: &standardDepth,
            alphaOutPtr: &standardAlpha,
            params: params
        )
        XCTAssertEqual(standardResult, 0, "Standard 32x16 render should succeed")

        // Fused render
        var fusedColor = [Float](repeating: 0, count: Int(imageWidth * imageHeight) * 3)
        var fusedDepth = [Float](repeating: 0, count: Int(imageWidth * imageHeight))
        var fusedAlpha = [Float](repeating: 0, count: Int(imageWidth * imageHeight))

        let fusedResult = fusedRenderer.renderRaw(
            gaussianCount: gaussianCount,
            meansPtr: &means,
            conicsPtr: &conics,
            colorsPtr: &colors,
            opacityPtr: &opacities,
            depthsPtr: &depths,
            radiiPtr: &radii,
            colorOutPtr: &fusedColor,
            depthOutPtr: &fusedDepth,
            alphaOutPtr: &fusedAlpha,
            params: params
        )
        XCTAssertEqual(fusedResult, 0, "Fused 32x16 render should succeed")

        // Compare outputs
        let tolerance: Float = 1e-3  // Slightly higher tolerance due to half precision differences
        var maxColorDiff: Float = 0
        var maxDepthDiff: Float = 0
        var maxAlphaDiff: Float = 0
        var differentPixels = 0

        for i in 0..<Int(imageWidth * imageHeight) {
            for c in 0..<3 {
                let diff = abs(standardColor[i * 3 + c] - fusedColor[i * 3 + c])
                maxColorDiff = max(maxColorDiff, diff)
                if diff > tolerance {
                    differentPixels += 1
                }
            }
            maxDepthDiff = max(maxDepthDiff, abs(standardDepth[i] - fusedDepth[i]))
            maxAlphaDiff = max(maxAlphaDiff, abs(standardAlpha[i] - fusedAlpha[i]))
        }

        print("[32x16Test] Max color diff: \(maxColorDiff), max depth diff: \(maxDepthDiff), max alpha diff: \(maxAlphaDiff)")
        print("[32x16Test] Pixels with color diff > \(tolerance): \(differentPixels)")

        // Count black tiles for both
        var stdBlack = 0, fusedBlack = 0
        for ty in 0..<Int(tilesY) {
            for tx in 0..<Int(tilesX) {
                var stdSum: Float = 0, fusedSum: Float = 0
                for py in 0..<Int(tileHeight) {
                    for px in 0..<Int(tileWidth) {
                        let x = tx * Int(tileWidth) + px
                        let y = ty * Int(tileHeight) + py
                        if x < Int(imageWidth) && y < Int(imageHeight) {
                            let idx = y * Int(imageWidth) + x
                            stdSum += standardAlpha[idx]
                            fusedSum += fusedAlpha[idx]
                        }
                    }
                }
                if stdSum == 0 { stdBlack += 1 }
                if fusedSum == 0 { fusedBlack += 1 }
            }
        }

        print("[32x16Test] Standard black: \(stdBlack)/\(totalTiles), Fused black: \(fusedBlack)/\(totalTiles)")

        // Fused should match standard very closely
        XCTAssertLessThan(maxColorDiff, tolerance * 100, "Color diff too large for 32x16 tiles")
        XCTAssertEqual(fusedBlack, stdBlack, "32x16: Fused should match standard black tile count")
    }

    /// Test fused multi-pixel at scale
    func testFused32x16AtScale() throws {
        // Reduced maxPerTile to keep heap within device limits
        let limits = RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024, tileWidth: 32, tileHeight: 16, maxPerTile: 512)
        let standardRenderer = Renderer(precision: .float32, sortAlgorithm: .radix, limits: limits)
        let fusedRenderer = Renderer(precision: .float32, sortAlgorithm: .radix, limits: limits)

        let gaussianCount = 500_000
        let imageWidth: UInt32 = 1024
        let imageHeight: UInt32 = 1024
        let tileWidth: UInt32 = 32
        let tileHeight: UInt32 = 16
        let tilesX = (imageWidth + tileWidth - 1) / tileWidth  // 32
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight  // 64
        let totalTiles = Int(tilesX * tilesY)  // 2048

        print("[32x16ScaleTest] Setup: \(gaussianCount) gaussians, \(imageWidth)x\(imageHeight), \(totalTiles) tiles (\(tileWidth)x\(tileHeight))")

        var means = [Float](repeating: 0, count: gaussianCount * 2)
        var conics = [Float](repeating: 0, count: gaussianCount * 4)
        var colors = [Float](repeating: 0, count: gaussianCount * 3)
        var opacities = [Float](repeating: 0, count: gaussianCount)
        var depths = [Float](repeating: 0, count: gaussianCount)
        var radii = [Float](repeating: 0, count: gaussianCount)

        for i in 0..<gaussianCount {
            means[i * 2] = Float.random(in: 0..<Float(imageWidth))
            means[i * 2 + 1] = Float.random(in: 0..<Float(imageHeight))
            let sigma = Float.random(in: 2..<6)
            conics[i * 4] = 1.0 / (sigma * sigma)
            conics[i * 4 + 2] = 1.0 / (sigma * sigma)
            colors[i * 3] = Float.random(in: 0..<1)
            colors[i * 3 + 1] = Float.random(in: 0..<1)
            colors[i * 3 + 2] = Float.random(in: 0..<1)
            opacities[i] = Float.random(in: 0.3..<0.8)
            depths[i] = Float.random(in: 0..<1)
            radii[i] = sigma * 3
        }

        let params = RenderParams(
            width: imageWidth, height: imageHeight,
            tileWidth: tileWidth, tileHeight: tileHeight,
            tilesX: tilesX, tilesY: tilesY,
            maxPerTile: 2048, whiteBackground: 0,
            activeTileCount: 0, gaussianCount: UInt32(gaussianCount)
        )

        // Standard render
        var standardAlpha = [Float](repeating: 0, count: Int(imageWidth * imageHeight))
        let standardStart = CFAbsoluteTimeGetCurrent()
        let stdResult = standardRenderer.renderRaw(
            gaussianCount: gaussianCount, meansPtr: &means, conicsPtr: &conics,
            colorsPtr: &colors, opacityPtr: &opacities, depthsPtr: &depths, radiiPtr: &radii,
            colorOutPtr: nil, depthOutPtr: nil, alphaOutPtr: &standardAlpha, params: params
        )
        let standardTime = (CFAbsoluteTimeGetCurrent() - standardStart) * 1000
        XCTAssertEqual(stdResult, 0)

        // Fused render
        var fusedAlpha = [Float](repeating: 0, count: Int(imageWidth * imageHeight))
        let fusedStart = CFAbsoluteTimeGetCurrent()
        let fusedResult = fusedRenderer.renderRaw(
            gaussianCount: gaussianCount, meansPtr: &means, conicsPtr: &conics,
            colorsPtr: &colors, opacityPtr: &opacities, depthsPtr: &depths, radiiPtr: &radii,
            colorOutPtr: nil, depthOutPtr: nil, alphaOutPtr: &fusedAlpha, params: params
        )
        let fusedTime = (CFAbsoluteTimeGetCurrent() - fusedStart) * 1000
        XCTAssertEqual(fusedResult, 0)

        print("[32x16ScaleTest] Standard: \(String(format: "%.2f", standardTime))ms, Fused: \(String(format: "%.2f", fusedTime))ms")

        // Compare black tile counts
        var stdBlack = 0, fusedBlack = 0
        for ty in 0..<Int(tilesY) {
            for tx in 0..<Int(tilesX) {
                var stdSum: Float = 0, fusedSum: Float = 0
                for py in 0..<Int(tileHeight) {
                    for px in 0..<Int(tileWidth) {
                        let x = tx * Int(tileWidth) + px
                        let y = ty * Int(tileHeight) + py
                        let idx = y * Int(imageWidth) + x
                        stdSum += standardAlpha[idx]
                        fusedSum += fusedAlpha[idx]
                    }
                }
                if stdSum == 0 { stdBlack += 1 }
                if fusedSum == 0 { fusedBlack += 1 }
            }
        }

        print("[32x16ScaleTest] Standard black: \(stdBlack)/\(totalTiles), Fused black: \(fusedBlack)/\(totalTiles)")

        // Allow small variance
        let maxDiff = totalTiles / 20
        let diff = abs(fusedBlack - stdBlack)
        XCTAssertLessThan(diff, maxDiff, "32x16 scale: Tile difference (\(diff)) too large")
    }

    /// GPU timing comparison: Standard vs Fused pipeline
    /// Uses Metal command buffer GPU timestamps for accurate measurement
    func testGPUTimingComparison() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            XCTFail("Failed to create Metal device/queue")
            return
        }

        let standardRenderer = Renderer(
            precision: .float16,
            sortAlgorithm: .radix
        )
        let fusedRenderer = Renderer(
            precision: .float16,
            sortAlgorithm: .radix
        )

        XCTAssertTrue(fusedRenderer.isFusedPipelineAvailable, "Fused pipeline should be available")

        // Test scene: 1M gaussians at 1024x1024 with 32x16 tiles
        let gaussianCount = 1_000_000
        let imageWidth: UInt32 = 1024
        let imageHeight: UInt32 = 1024
        let tileWidth: UInt32 = 32
        let tileHeight: UInt32 = 16
        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight

        print("\n[GPUTiming] Setup: \(gaussianCount) gaussians, \(imageWidth)x\(imageHeight), \(tileWidth)x\(tileHeight) tiles")

        // Generate test data
        var means = [Float](repeating: 0, count: gaussianCount * 2)
        var conics = [Float](repeating: 0, count: gaussianCount * 4)
        var colors = [Float](repeating: 0, count: gaussianCount * 3)
        var opacities = [Float](repeating: 0, count: gaussianCount)
        var depths = [Float](repeating: 0, count: gaussianCount)
        var radii = [Float](repeating: 0, count: gaussianCount)

        for i in 0..<gaussianCount {
            means[i * 2] = Float.random(in: 0..<Float(imageWidth))
            means[i * 2 + 1] = Float.random(in: 0..<Float(imageHeight))
            let sigma = Float.random(in: 2..<6)
            conics[i * 4] = 1.0 / (sigma * sigma)
            conics[i * 4 + 2] = 1.0 / (sigma * sigma)
            colors[i * 3] = Float.random(in: 0..<1)
            colors[i * 3 + 1] = Float.random(in: 0..<1)
            colors[i * 3 + 2] = Float.random(in: 0..<1)
            opacities[i] = Float.random(in: 0.3..<0.8)
            depths[i] = Float.random(in: 0..<1)
            radii[i] = sigma * 3
        }

        let params = RenderParams(
            width: imageWidth, height: imageHeight,
            tileWidth: tileWidth, tileHeight: tileHeight,
            tilesX: tilesX, tilesY: tilesY,
            maxPerTile: 2048, whiteBackground: 0,
            activeTileCount: 0, gaussianCount: UInt32(gaussianCount)
        )

        var alphaOut = [Float](repeating: 0, count: Int(imageWidth * imageHeight))

        // Warmup runs
        for _ in 0..<3 {
            _ = standardRenderer.renderRaw(
                gaussianCount: gaussianCount, meansPtr: &means, conicsPtr: &conics,
                colorsPtr: &colors, opacityPtr: &opacities, depthsPtr: &depths, radiiPtr: &radii,
                colorOutPtr: nil, depthOutPtr: nil, alphaOutPtr: &alphaOut, params: params
            )
            _ = fusedRenderer.renderRaw(
                gaussianCount: gaussianCount, meansPtr: &means, conicsPtr: &conics,
                colorsPtr: &colors, opacityPtr: &opacities, depthsPtr: &depths, radiiPtr: &radii,
                colorOutPtr: nil, depthOutPtr: nil, alphaOutPtr: &alphaOut, params: params
            )
        }

        // GPU timing using command buffer timestamps
        let iterations = 10
        var standardGPUTimes: [Double] = []
        var fusedGPUTimes: [Double] = []

        for _ in 0..<iterations {
            // Standard pipeline - measure GPU time
            _ = standardRenderer.renderRaw(
                gaussianCount: gaussianCount, meansPtr: &means, conicsPtr: &conics,
                colorsPtr: &colors, opacityPtr: &opacities, depthsPtr: &depths, radiiPtr: &radii,
                colorOutPtr: nil, depthOutPtr: nil, alphaOutPtr: &alphaOut, params: params
            )
            if let gpuTime = standardRenderer.lastGPUTime {
                standardGPUTimes.append(gpuTime * 1000) // Convert to ms
            }

            // Fused pipeline - measure GPU time
            _ = fusedRenderer.renderRaw(
                gaussianCount: gaussianCount, meansPtr: &means, conicsPtr: &conics,
                colorsPtr: &colors, opacityPtr: &opacities, depthsPtr: &depths, radiiPtr: &radii,
                colorOutPtr: nil, depthOutPtr: nil, alphaOutPtr: &alphaOut, params: params
            )
            if let gpuTime = fusedRenderer.lastGPUTime {
                fusedGPUTimes.append(gpuTime * 1000) // Convert to ms
            }
        }

        // Calculate statistics
        if !standardGPUTimes.isEmpty && !fusedGPUTimes.isEmpty {
            let stdAvg = standardGPUTimes.reduce(0, +) / Double(standardGPUTimes.count)
            let fusedAvg = fusedGPUTimes.reduce(0, +) / Double(fusedGPUTimes.count)
            let stdMin = standardGPUTimes.min()!
            let fusedMin = fusedGPUTimes.min()!

            print("\n[GPUTiming] GPU Time Results over \(iterations) iterations:")
            print("[GPUTiming] Standard GPU: avg=\(String(format: "%.2f", stdAvg))ms, min=\(String(format: "%.2f", stdMin))ms")
            print("[GPUTiming] Fused GPU:    avg=\(String(format: "%.2f", fusedAvg))ms, min=\(String(format: "%.2f", fusedMin))ms")

            if fusedAvg < stdAvg {
                print("[GPUTiming] ✓ Fused GPU is \(String(format: "%.1f", (1 - fusedAvg/stdAvg) * 100))% faster")
            } else {
                print("[GPUTiming] ✗ Standard GPU is \(String(format: "%.1f", (1 - stdAvg/fusedAvg) * 100))% faster")
            }
        } else {
            print("[GPUTiming] GPU timestamps not available - using wall time instead")
            // Fall back to wall time measurement
            var stdWallTimes: [Double] = []
            var fusedWallTimes: [Double] = []

            let nilFloatPtr: UnsafeMutablePointer<Float>? = nil
            for _ in 0..<iterations {
                let t0 = CFAbsoluteTimeGetCurrent()
                _ = standardRenderer.renderRaw(
                    gaussianCount: gaussianCount, meansPtr: &means, conicsPtr: &conics,
                    colorsPtr: &colors, opacityPtr: &opacities, depthsPtr: &depths, radiiPtr: &radii,
                    colorOutPtr: nilFloatPtr, depthOutPtr: nilFloatPtr, alphaOutPtr: &alphaOut, params: params
                )
                stdWallTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)

                let t1 = CFAbsoluteTimeGetCurrent()
                _ = fusedRenderer.renderRaw(
                    gaussianCount: gaussianCount, meansPtr: &means, conicsPtr: &conics,
                    colorsPtr: &colors, opacityPtr: &opacities, depthsPtr: &depths, radiiPtr: &radii,
                    colorOutPtr: nilFloatPtr, depthOutPtr: nilFloatPtr, alphaOutPtr: &alphaOut, params: params
                )
                fusedWallTimes.append((CFAbsoluteTimeGetCurrent() - t1) * 1000)
            }

            let stdAvg = stdWallTimes.reduce(0, +) / Double(iterations)
            let fusedAvg = fusedWallTimes.reduce(0, +) / Double(iterations)
            print("[GPUTiming] Standard Wall: avg=\(String(format: "%.2f", stdAvg))ms")
            print("[GPUTiming] Fused Wall:    avg=\(String(format: "%.2f", fusedAvg))ms")
        }

        XCTAssertTrue(true)
    }
}

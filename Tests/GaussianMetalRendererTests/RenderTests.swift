import XCTest
import simd
@testable import GaussianMetalRenderer

final class RenderTests: XCTestCase {
    
    var renderer: Renderer!
    
    override func setUp() {
        super.setUp()
        renderer = Renderer.shared
    }
    
    func testRenderRawSimpleGaussian() {
        let width: Int32 = 16
        let height: Int32 = 16
        let tileWidth: Int32 = 16
        let tileHeight: Int32 = 16
        
        let count = 1
        
                // Center of 16x16
                let means: [Float] = [8.0, 8.0] 
                let conics: [Float] = [1.0, 0.0, 1.0, 0.0] 
                let colors: [Float] = [1.0, 0.0, 0.0] // Red
                let opacities: [Float] = [1.0]
                let depths: [Float] = [1.0]
                let radii: [Float] = [8.0]
                
                var colorOut = [Float](repeating: 0, count: Int(width * height) * 3)
                var depthOut = [Float](repeating: 0, count: Int(width * height))
                var alphaOut = [Float](repeating: 0, count: Int(width * height))
                
                let params = RenderParams(
                    width: UInt32(width),
                    height: UInt32(height),
                    tileWidth: UInt32(tileWidth),
                    tileHeight: UInt32(tileHeight),
                    tilesX: 1,
                    tilesY: 1,
                    maxPerTile: 100,
                    whiteBackground: 0,
                    activeTileCount: 0,
                    gaussianCount: UInt32(count)
                )        
        let result = means.withUnsafeBufferPointer { meansBuf in
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
        
        XCTAssertEqual(result, 0, "Render failed with code \(result)")
        
        let centerIdx = 136
        let r = colorOut[centerIdx * 3 + 0]
        let g = colorOut[centerIdx * 3 + 1]
        let b = colorOut[centerIdx * 3 + 2]
        let a = alphaOut[centerIdx]
        
        XCTAssertGreaterThan(a, 0.1, "Pixel alpha should be non-zero")
        XCTAssertGreaterThan(r, 0.1, "Red channel should be active")
        XCTAssertLessThan(g, 0.1, "Green channel should be inactive")
        XCTAssertLessThan(b, 0.1, "Blue channel should be inactive")
    }
    
    func testRenderRawSimpleGaussianHalf() {
        let renderer = Renderer(precision: .float16)
        let width: Int32 = 16
        let height: Int32 = 16
        let tileWidth: Int32 = 16
        let tileHeight: Int32 = 16
        let count = 1
        
        let means: [Float] = [8.0, 8.0]
        let conics: [Float] = [1.0, 0.0, 1.0, 0.0]
        let colors: [Float] = [1.0, 0.0, 0.0]
        let opacities: [Float] = [1.0]
        let depths: [Float] = [1.0]
        let radii: [Float] = [8.0]
        
        var colorOut = [Float](repeating: 0, count: Int(width * height) * 3)
        var depthOut = [Float](repeating: 0, count: Int(width * height))
        var alphaOut = [Float](repeating: 0, count: Int(width * height))
        
        let params = RenderParams(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: 1,
            tilesY: 1,
            maxPerTile: 100,
            whiteBackground: 0,
            activeTileCount: 0,
            gaussianCount: UInt32(count)
        )
        
        let result = means.withUnsafeBufferPointer { meansBuf in
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
        
        XCTAssertEqual(result, 0, "Render failed with code \(result)")
        let centerIdx = 136
        XCTAssertGreaterThan(alphaOut[centerIdx], 0.1, "Pixel alpha should be non-zero in half mode")
    }
    
    func testRenderGridPattern() {
        let width: Int32 = 256
        let height: Int32 = 256
        let tileWidth: Int32 = 16
        let tileHeight: Int32 = 16
        
        let tilesX = width / tileWidth
        let tilesY = height / tileHeight
        let totalTiles = Int(tilesX * tilesY)
        
        var means: [Float] = []
        var conics: [Float] = []
        var colors: [Float] = []
        var opacities: [Float] = []
        var depths: [Float] = []
        var radii: [Float] = []
        
        // Create tile coordinates and shuffle them
        var tileCoords: [(Int, Int)] = []
        for ty in 0..<tilesY {
            for tx in 0..<tilesX {
                tileCoords.append((Int(tx), Int(ty)))
            }
        }
        // Shuffle deterministically
        tileCoords.sort { (a, b) -> Bool in
            let h1 = (a.0 * 31 + a.1 * 17) % 7
            let h2 = (b.0 * 31 + b.1 * 17) % 7
            if h1 != h2 { return h1 < h2 }
            return a.0 > b.0
        }
        
        // 300 per tile * 4096 tiles = 1.2M assignments
        for (tx, ty) in tileCoords {
            let cx = Float(tx) * Float(tileWidth) + Float(tileWidth) / 2.0
            let cy = Float(ty) * Float(tileHeight) + Float(tileHeight) / 2.0
            
            for i in 0..<300 {
                means.append(cx + Float(i)*0.01)
                means.append(cy + Float(i)*0.01)
                
                conics.append(contentsOf: [1.0, 0.0, 1.0, 0.0])
                colors.append(contentsOf: [1.0, 1.0, 1.0])
                opacities.append(1.0)
                depths.append(Float.random(in: 0.1...10.0))
                // Keep radius confined to the home tile to avoid multi-tile spill for this stress test.
                radii.append(Float(tileWidth) / 4.0)
            }
        }
        
        let count = means.count / 2
        print("Total Gaussians: \(count)")
        
        var colorOut = [Float](repeating: 0, count: Int(width * height) * 3)
        var depthOut = [Float](repeating: 0, count: Int(width * height))
        var alphaOut = [Float](repeating: 0, count: Int(width * height))
        
        let params = RenderParams(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: UInt32(tilesX),
            tilesY: UInt32(tilesY),
            maxPerTile: 500,
            whiteBackground: 0,
            activeTileCount: 0,
            gaussianCount: UInt32(count)
        )
        
        let result = means.withUnsafeBufferPointer { meansBuf in
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
        
        XCTAssertEqual(result, 0, "Render failed")

        // Verify every tile has content
        var missingTiles = 0
        for ty in 0..<tilesY {
            for tx in 0..<tilesX {
                let cx = Int(tx * tileWidth) + Int(tileWidth) / 2
                let cy = Int(ty * tileHeight) + Int(tileHeight) / 2
                let idx = cy * Int(width) + cx
                
                let alpha = alphaOut[idx]
                if alpha < 0.1 {
                    missingTiles += 1
                }
            }
        }
        
        XCTAssertEqual(missingTiles, 0, "Found \(missingTiles) empty tiles out of \(totalTiles)")
    }
    
    func testRenderGaussianSpanningTiles() {
        let width: Int32 = 1024
        let height: Int32 = 1024
        let tileWidth: Int32 = 16
        let tileHeight: Int32 = 16
        
        // 2 tiles side-by-side: [0, 1]
        
        // Gaussian at x=16 (boundary). Should cover both tile 0 (0-15) and tile 1 (16-31)
        // Radius = 4. x range [12, 20].
        // Tile 0: x < 16. Intersects.
        // Tile 1: x >= 16. Intersects.
        
        let means: [Float] = [16.0, 8.0]
        let conics: [Float] = [1.0, 0.0, 1.0, 0.0]
        let colors: [Float] = [1.0, 0.0, 0.0]
        let opacities: [Float] = [1.0]
        let depths: [Float] = [1.0]
        let radii: [Float] = [4.0]
        
        var colorOut = [Float](repeating: 0, count: Int(width * height) * 3)
        var depthOut = [Float](repeating: 0, count: Int(width * height))
        var alphaOut = [Float](repeating: 0, count: Int(width * height))
        
        let params = RenderParams(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: 2,
            tilesY: 1,
            maxPerTile: 100,
            whiteBackground: 0,
            activeTileCount: 0,
            gaussianCount: 1
        )
        
        let result = means.withUnsafeBufferPointer { meansBuf in
            conics.withUnsafeBufferPointer { conicsBuf in
                colors.withUnsafeBufferPointer { colorsBuf in
                    opacities.withUnsafeBufferPointer { opacitiesBuf in
                        depths.withUnsafeBufferPointer { depthsBuf in
                            radii.withUnsafeBufferPointer { radiiBuf in
                                colorOut.withUnsafeMutableBufferPointer { colorOutBuf in
                                    depthOut.withUnsafeMutableBufferPointer { depthOutBuf in
                                        alphaOut.withUnsafeMutableBufferPointer { alphaOutBuf in
                                            renderer.renderRaw(
                                                gaussianCount: 1,
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
        
        XCTAssertEqual(result, 0, "Render failed")
        
        // Check pixel in Tile 0 (x=15)
        let idx0 = 8 * Int(width) + 15
        XCTAssertGreaterThan(alphaOut[idx0], 0.1, "Tile 0 should have content")
        
        // Check pixel in Tile 1 (x=16)
        let idx1 = 8 * Int(width) + 16
        XCTAssertGreaterThan(alphaOut[idx1], 0.1, "Tile 1 should have content")
    }
    
    func testRenderCheckerboardPattern() {
        let width: Int32 = 32
        let height: Int32 = 32
        let tileWidth: Int32 = 16
        let tileHeight: Int32 = 16
        
        // 2x2 tiles. IDs: 0, 1, 2, 3.
        // We place one gaussian in the center of each tile.
        // Tile 0 center: (8, 8)
        // Tile 1 center: (24, 8)
        // Tile 2 center: (8, 24)
        // Tile 3 center: (24, 24)
        
        let means: [Float] = [
            8.0, 8.0,   // Tile 0
            24.0, 8.0,  // Tile 1
            8.0, 24.0,  // Tile 2
            24.0, 24.0  // Tile 3
        ]
        let conics: [Float] = Array(repeating: [1.0, 0.0, 1.0, 0.0], count: 4).flatMap { $0 }
        let colors: [Float] = [
            1.0, 0.0, 0.0, // Red
            0.0, 1.0, 0.0, // Green
            0.0, 0.0, 1.0, // Blue
            1.0, 1.0, 0.0  // Yellow
        ]
        let opacities: [Float] = [1.0, 1.0, 1.0, 1.0]
        let depths: [Float] = [1.0, 1.0, 1.0, 1.0]
        let radii: [Float] = [4.0, 4.0, 4.0, 4.0]
        
        var colorOut = [Float](repeating: 0, count: Int(width * height) * 3)
        var depthOut = [Float](repeating: 0, count: Int(width * height))
        var alphaOut = [Float](repeating: 0, count: Int(width * height))
        
        let params = RenderParams(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tileWidth),
            tileHeight: UInt32(tileHeight),
            tilesX: 2,
            tilesY: 2,
            maxPerTile: 100,
            whiteBackground: 0,
            activeTileCount: 0,
            gaussianCount: 4
        )
        
        let result = means.withUnsafeBufferPointer { meansBuf in
            conics.withUnsafeBufferPointer { conicsBuf in
                colors.withUnsafeBufferPointer { colorsBuf in
                    opacities.withUnsafeBufferPointer { opacitiesBuf in
                        depths.withUnsafeBufferPointer { depthsBuf in
                            radii.withUnsafeBufferPointer { radiiBuf in
                                colorOut.withUnsafeMutableBufferPointer { colorOutBuf in
                                    depthOut.withUnsafeMutableBufferPointer { depthOutBuf in
                                        alphaOut.withUnsafeMutableBufferPointer { alphaOutBuf in
                                            renderer.renderRaw(
                                                gaussianCount: 4,
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
        
        XCTAssertEqual(result, 0, "Render failed")
        
        // Debug inspection removed
        
        // Check Center of Tile 0 (8, 8)
        let idx0 = 8 * 32 + 8
        XCTAssertGreaterThan(alphaOut[idx0], 0.1, "Tile 0 (Top-Left) missing")
        
        // Check Center of Tile 1 (8, 24)
        let idx1 = 8 * 32 + 24
        XCTAssertGreaterThan(alphaOut[idx1], 0.1, "Tile 1 (Top-Right) missing")
        
        // Check Center of Tile 2 (24, 8) -> y=24, x=8. Index 24*32 + 8
        let idx2 = 24 * 32 + 8
        XCTAssertGreaterThan(alphaOut[idx2], 0.1, "Tile 2 (Bottom-Left) missing")
        
        // Check Center of Tile 3 (24, 24) -> y=24, x=24. Index 24*32 + 24
        let idx3 = 24 * 32 + 24
        XCTAssertGreaterThan(alphaOut[idx3], 0.1, "Tile 3 (Bottom-Right) missing")
    }
    
    /// Test the native half16 input pipeline (WorldGaussianBuffersHalf -> encodeRenderToTextureHalf)
    @MainActor
    func testRenderHalfInputToTexture() {
        let renderer = Renderer(precision: .float16)
        let width: Int32 = 32
        let height: Int32 = 32
        let tile: Int32 = 16
        let count = 4

        // Helper to convert float32 to float16
        func floatToHalf(_ value: Float) -> UInt16 {
            let bits = value.bitPattern
            let sign = (bits >> 16) & 0x8000
            let exp = Int((bits >> 23) & 0xFF) - 127 + 15
            let frac = bits & 0x007FFFFF
            if exp <= 0 {
                if exp < -10 { return UInt16(sign) }
                let shift = 14 - exp
                return UInt16(sign | ((frac | 0x00800000) >> (shift + 1)))
            } else if exp >= 31 {
                return UInt16(sign | 0x7C00 | (frac != 0 ? 0x0200 : 0))
            }
            return UInt16(sign | UInt32(exp << 10) | (frac >> 13))
        }

        // Create half-precision world data
        var positionsHalf = [UInt16](repeating: 0, count: count * 3)
        var scalesHalf = [UInt16](repeating: 0, count: count * 3)
        var rotationsHalf = [UInt16](repeating: 0, count: count * 4)
        var harmonicsHalf = [UInt16](repeating: 0, count: count * 3)
        var opacitiesHalf = [UInt16](repeating: 0, count: count)

        // 4 gaussians at positions along z=1.0
        for i in 0..<count {
            let pBase = i * 3
            positionsHalf[pBase] = floatToHalf(Float(i) * 0.1)
            positionsHalf[pBase + 1] = floatToHalf(0.0)
            positionsHalf[pBase + 2] = floatToHalf(1.0)

            scalesHalf[pBase] = floatToHalf(0.05)
            scalesHalf[pBase + 1] = floatToHalf(0.05)
            scalesHalf[pBase + 2] = floatToHalf(0.05)

            let rBase = i * 4
            rotationsHalf[rBase] = floatToHalf(1.0)  // Identity quaternion
            rotationsHalf[rBase + 1] = floatToHalf(0.0)
            rotationsHalf[rBase + 2] = floatToHalf(0.0)
            rotationsHalf[rBase + 3] = floatToHalf(0.0)

            // Simple RGB colors for harmonics (DC term)
            harmonicsHalf[pBase] = floatToHalf(0.5)
            harmonicsHalf[pBase + 1] = floatToHalf(0.0)
            harmonicsHalf[pBase + 2] = floatToHalf(0.0)

            opacitiesHalf[i] = floatToHalf(0.9)
        }

        let device = renderer.device
        let shared: MTLResourceOptions = .storageModeShared

        guard
            let posBuffer = device.makeBuffer(bytes: positionsHalf, length: positionsHalf.count * 2, options: shared),
            let scaBuffer = device.makeBuffer(bytes: scalesHalf, length: scalesHalf.count * 2, options: shared),
            let rotBuffer = device.makeBuffer(bytes: rotationsHalf, length: rotationsHalf.count * 2, options: shared),
            let harmBuffer = device.makeBuffer(bytes: harmonicsHalf, length: harmonicsHalf.count * 2, options: shared),
            let opacBuffer = device.makeBuffer(bytes: opacitiesHalf, length: opacitiesHalf.count * 2, options: shared)
        else {
            XCTFail("Failed to create half-precision buffers")
            return
        }

        let worldBuffersHalf = WorldGaussianBuffersHalf(
            positions: posBuffer,
            scales: scaBuffer,
            rotations: rotBuffer,
            harmonics: harmBuffer,
            opacities: opacBuffer,
            shComponents: 0
        )

        let params = RenderParams(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tile),
            tileHeight: UInt32(tile),
            tilesX: UInt32((width + tile - 1) / tile),
            tilesY: UInt32((height + tile - 1) / tile),
            maxPerTile: 64,
            whiteBackground: 1,
            activeTileCount: 0,
            gaussianCount: UInt32(count)
        )

        let camera = CameraUniformsSwift(
            viewMatrix: matrix_identity_float4x4,
            projectionMatrix: matrix_identity_float4x4,
            cameraCenter: SIMD3<Float>(0, 0, 0),
            pixelFactor: 1.0,
            focalX: Float(width),
            focalY: Float(height),
            width: Float(width),
            height: Float(height),
            nearPlane: 0.1,
            farPlane: 10.0,
            shComponents: 0,
            gaussianCount: UInt32(count)
        )

        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let textures = renderer.encodeRenderToTextureHalf(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            worldBuffers: worldBuffersHalf,
            cameraUniforms: camera,
            params: params
        )

        XCTAssertNotNil(textures, "encodeRenderToTextureHalf should return textures")

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        XCTAssertEqual(commandBuffer.status, .completed, "Command buffer should complete successfully")

        // Verify textures were created with correct dimensions
        if let textures = textures {
            XCTAssertEqual(textures.color.width, Int(width), "Color texture width mismatch")
            XCTAssertEqual(textures.color.height, Int(height), "Color texture height mismatch")
        }
    }

    @MainActor
    func testRenderWorldAsyncNonBlocking() {
        let width: Int32 = 16
        let height: Int32 = 16
        let tile: Int32 = 16
        let count = 4
        
        var positions: [Float] = []
        var scales: [Float] = []
        var rotations = [Float](repeating: 0, count: count * 4)
        let harmonics = [Float](repeating: 0, count: count * 3)
        let opacities = [Float](repeating: 0.8, count: count)
        
        for i in 0..<count {
            positions.append(Float(i) * 0.1)
            positions.append(0)
            positions.append(1.0)
            
            scales.append(contentsOf: [0.05, 0.05, 0.05])
            rotations[i * 4] = 1.0 // identity quaternion real component
        }
        
        let params = RenderParams(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tile),
            tileHeight: UInt32(tile),
            tilesX: 1,
            tilesY: 1,
            maxPerTile: 64,
            whiteBackground: 1,
            activeTileCount: 0,
            gaussianCount: UInt32(count)
        )
        
        let worldBuffers = positions.withUnsafeBufferPointer { posPtr in
            scales.withUnsafeBufferPointer { scalePtr in
                rotations.withUnsafeBufferPointer { rotPtr in
                    harmonics.withUnsafeBufferPointer { harmPtr in
                        opacities.withUnsafeBufferPointer { opaPtr in
                            renderer.prepareWorldBuffers(
                                count: count,
                                meansPtr: posPtr.baseAddress!,
                                scalesPtr: scalePtr.baseAddress!,
                                rotationsPtr: rotPtr.baseAddress!,
                                harmonicsPtr: harmPtr.baseAddress!,
                                opacitiesPtr: opaPtr.baseAddress!,
                                shComponents: 0
                            )
                        }
                    }
                }
            }
        }
        
        XCTAssertNotNil(worldBuffers, "Failed to create world buffers")
        guard let worldBuffers else { return }
        
        let camera = CameraUniformsSwift(
            viewMatrix: matrix_identity_float4x4,
            projectionMatrix: matrix_identity_float4x4,
            cameraCenter: SIMD3<Float>(0, 0, 0),
            pixelFactor: 1.0,
            focalX: 1.0,
            focalY: 1.0,
            width: Float(width),
            height: Float(height),
            nearPlane: 0.1,
            farPlane: 10.0,
            shComponents: 0,
            gaussianCount: UInt32(count)
        )
        
        let start = CFAbsoluteTimeGetCurrent()
        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }
        
        let outputBuffers = renderer.encodeRender(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            worldBuffers: worldBuffers,
            cameraUniforms: camera,
            params: params
        )
        
        XCTAssertNotNil(outputBuffers, "encodeRender failed")
        
        commandBuffer.commit()
        
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        XCTAssertLessThan(elapsed, 0.05, "Async render should return immediately without blocking")

        commandBuffer.waitUntilCompleted()
        XCTAssertEqual(commandBuffer.status, .completed)
    }

    /// Tests that the half-precision pipeline produces valid, non-garbage output.
    /// This test catches half/float type mismatches that cause kernels to read
    /// the wrong data types, resulting in garbage values or white screens.
    func testHalfPrecisionPipelineIntegrity() {
        let renderer = Renderer(precision: .float16)
        let width: Int32 = 64
        let height: Int32 = 64
        let tile: Int32 = 16
        let count = 16  // Multiple gaussians to stress the pipeline

        // Helper to convert float32 to float16
        func floatToHalf(_ value: Float) -> UInt16 {
            let bits = value.bitPattern
            let sign = (bits >> 16) & 0x8000
            let exp = Int((bits >> 23) & 0xFF) - 127 + 15
            let frac = bits & 0x007FFFFF
            if exp <= 0 {
                if exp < -10 { return UInt16(sign) }
                let shift = 14 - exp
                return UInt16(sign | ((frac | 0x00800000) >> (shift + 1)))
            } else if exp >= 31 {
                return UInt16(sign | 0x7C00 | (frac != 0 ? 0x0200 : 0))
            }
            return UInt16(sign | UInt32(exp << 10) | (frac >> 13))
        }

        // Create half-precision world data with gaussians spread across screen
        var positionsHalf = [UInt16](repeating: 0, count: count * 3)
        var scalesHalf = [UInt16](repeating: 0, count: count * 3)
        var rotationsHalf = [UInt16](repeating: 0, count: count * 4)
        var harmonicsHalf = [UInt16](repeating: 0, count: count * 3)
        var opacitiesHalf = [UInt16](repeating: 0, count: count)

        for i in 0..<count {
            let row = i / 4
            let col = i % 4
            let pBase = i * 3
            // Position gaussians in a grid pattern at z=1.0
            positionsHalf[pBase] = floatToHalf(Float(col) * 0.2 - 0.3)
            positionsHalf[pBase + 1] = floatToHalf(Float(row) * 0.2 - 0.3)
            positionsHalf[pBase + 2] = floatToHalf(1.0)

            scalesHalf[pBase] = floatToHalf(0.08)
            scalesHalf[pBase + 1] = floatToHalf(0.08)
            scalesHalf[pBase + 2] = floatToHalf(0.08)

            let rBase = i * 4
            rotationsHalf[rBase] = floatToHalf(1.0)  // Identity quaternion
            rotationsHalf[rBase + 1] = floatToHalf(0.0)
            rotationsHalf[rBase + 2] = floatToHalf(0.0)
            rotationsHalf[rBase + 3] = floatToHalf(0.0)

            // Varying colors to detect if color channel reads are correct
            harmonicsHalf[pBase] = floatToHalf(Float(i % 4) / 4.0 + 0.2)
            harmonicsHalf[pBase + 1] = floatToHalf(Float((i + 1) % 4) / 4.0 + 0.2)
            harmonicsHalf[pBase + 2] = floatToHalf(Float((i + 2) % 4) / 4.0 + 0.2)

            opacitiesHalf[i] = floatToHalf(0.8)
        }

        let device = renderer.device
        let shared: MTLResourceOptions = .storageModeShared

        guard
            let posBuffer = device.makeBuffer(bytes: positionsHalf, length: positionsHalf.count * 2, options: shared),
            let scaBuffer = device.makeBuffer(bytes: scalesHalf, length: scalesHalf.count * 2, options: shared),
            let rotBuffer = device.makeBuffer(bytes: rotationsHalf, length: rotationsHalf.count * 2, options: shared),
            let harmBuffer = device.makeBuffer(bytes: harmonicsHalf, length: harmonicsHalf.count * 2, options: shared),
            let opacBuffer = device.makeBuffer(bytes: opacitiesHalf, length: opacitiesHalf.count * 2, options: shared)
        else {
            XCTFail("Failed to create half-precision buffers")
            return
        }

        let worldBuffersHalf = WorldGaussianBuffersHalf(
            positions: posBuffer,
            scales: scaBuffer,
            rotations: rotBuffer,
            harmonics: harmBuffer,
            opacities: opacBuffer,
            shComponents: 0
        )

        let params = RenderParams(
            width: UInt32(width),
            height: UInt32(height),
            tileWidth: UInt32(tile),
            tileHeight: UInt32(tile),
            tilesX: UInt32((width + tile - 1) / tile),
            tilesY: UInt32((height + tile - 1) / tile),
            maxPerTile: 64,
            whiteBackground: 0,
            activeTileCount: 0,
            gaussianCount: UInt32(count)
        )

        let camera = CameraUniformsSwift(
            viewMatrix: matrix_identity_float4x4,
            projectionMatrix: matrix_identity_float4x4,
            cameraCenter: SIMD3<Float>(0, 0, 0),
            pixelFactor: 1.0,
            focalX: Float(width),
            focalY: Float(height),
            width: Float(width),
            height: Float(height),
            nearPlane: 0.1,
            farPlane: 10.0,
            shComponents: 0,
            gaussianCount: UInt32(count)
        )

        guard let commandBuffer = renderer.queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let textures = renderer.encodeRenderToTextureHalf(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            worldBuffers: worldBuffersHalf,
            cameraUniforms: camera,
            params: params
        )

        XCTAssertNotNil(textures, "encodeRenderToTextureHalf should return textures")
        guard let textures = textures else { return }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        XCTAssertEqual(commandBuffer.status, .completed, "Command buffer should complete successfully")

        // Copy from private texture to readable buffer using a blit encoder
        let bytesPerRow = Int(width) * 8  // RGBA16F = 8 bytes per pixel
        let bufferSize = bytesPerRow * Int(height)
        guard let readbackBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            XCTFail("Failed to create readback buffer")
            return
        }

        guard let blitCommandBuffer = renderer.queue.makeCommandBuffer(),
              let blitEncoder = blitCommandBuffer.makeBlitCommandEncoder() else {
            XCTFail("Failed to create blit command buffer")
            return
        }

        blitEncoder.copy(
            from: textures.color,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: Int(width), height: Int(height), depth: 1),
            to: readbackBuffer,
            destinationOffset: 0,
            destinationBytesPerRow: bytesPerRow,
            destinationBytesPerImage: bufferSize
        )
        blitEncoder.endEncoding()
        blitCommandBuffer.commit()
        blitCommandBuffer.waitUntilCompleted()

        let outputData = readbackBuffer.contents().bindMemory(to: UInt16.self, capacity: Int(width * height) * 4)

        // Convert half floats back to check for valid values
        func halfToFloat(_ h: UInt16) -> Float {
            let sign = Float((h >> 15) & 1) * -2.0 + 1.0
            let exp = Int((h >> 10) & 0x1F)
            let frac = Float(h & 0x3FF) / 1024.0
            if exp == 0 {
                return sign * frac * pow(2, -14)
            } else if exp == 31 {
                return frac == 0 ? (sign * .infinity) : .nan
            }
            return sign * (1.0 + frac) * pow(2, Float(exp - 15))
        }

        // Check for non-zero alpha in the output (indicates gaussians were rendered)
        var nonZeroAlphaCount = 0
        var nanCount = 0
        var infCount = 0
        var totalAlpha: Float = 0.0

        for y in 0..<Int(height) {
            for x in 0..<Int(width) {
                let idx = (y * Int(width) + x) * 4
                let r = halfToFloat(outputData[idx])
                let g = halfToFloat(outputData[idx + 1])
                let b = halfToFloat(outputData[idx + 2])
                let a = halfToFloat(outputData[idx + 3])

                // Check for NaN (indicates type mismatch reading garbage)
                if r.isNaN || g.isNaN || b.isNaN || a.isNaN {
                    nanCount += 1
                }
                // Check for infinity
                if r.isInfinite || g.isInfinite || b.isInfinite || a.isInfinite {
                    infCount += 1
                }

                if a > 0.01 {
                    nonZeroAlphaCount += 1
                    totalAlpha += a
                }
            }
        }

        XCTAssertEqual(nanCount, 0, "Half-precision pipeline produced \(nanCount) NaN values - likely type mismatch")
        XCTAssertEqual(infCount, 0, "Half-precision pipeline produced \(infCount) Inf values - likely type mismatch")
        XCTAssertGreaterThan(nonZeroAlphaCount, 10, "Half-precision pipeline should render visible gaussians (got \(nonZeroAlphaCount) non-zero pixels)")

        // Average alpha should be reasonable (not 0)
        let avgAlpha = nonZeroAlphaCount > 0 ? totalAlpha / Float(nonZeroAlphaCount) : 0
        XCTAssertGreaterThan(avgAlpha, 0.1, "Average alpha too low (\(avgAlpha)) - possible type mismatch causing dimming")
        // Alpha can legitimately be 1.0 if gaussians are opaque, so we just check it's not > 1
        XCTAssertLessThanOrEqual(avgAlpha, 1.001, "Average alpha (\(avgAlpha)) should be <= 1.0")
    }
}

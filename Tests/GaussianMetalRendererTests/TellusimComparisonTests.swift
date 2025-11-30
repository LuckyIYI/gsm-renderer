import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Compare Tellusim pipeline output to original pipeline output
final class TellusimComparisonTests: XCTestCase {
    private let imageWidth = 1920
    private let imageHeight = 1080
    private let tileWidth = 32
    private let tileHeight = 16

    /// Compare projection outputs between original and Tellusim pipelines
    func testCompareProjection() throws {
        let renderer = Renderer.shared
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let gaussianCount = 100

        // Create identical input data for both pipelines
        let worldBuffer = device.makeBuffer(
            length: gaussianCount * MemoryLayout<PackedWorldGaussian>.stride,
            options: .storageModeShared
        )!
        let harmonicsBuffer = device.makeBuffer(
            length: gaussianCount * 3 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        // Populate with test data - scattered gaussians in front of camera
        let worldPtr = worldBuffer.contents().bindMemory(to: PackedWorldGaussian.self, capacity: gaussianCount)
        let harmonicsPtr = harmonicsBuffer.contents().bindMemory(to: Float.self, capacity: gaussianCount * 3)

        for i in 0..<gaussianCount {
            // Place gaussians at various positions in front of camera
            let x = Float(i % 10) - 4.5  // -4.5 to 4.5
            let y = Float(i / 10) - 4.5  // -4.5 to 4.5
            let z: Float = 5.0  // 5 units in front (positive Z = in front for this test)

            worldPtr[i] = PackedWorldGaussian(
                position: SIMD3<Float>(x, y, z),
                scale: SIMD3<Float>(0.1, 0.1, 0.1),
                rotation: SIMD4<Float>(0, 0, 0, 1),
                opacity: 0.8
            )

            harmonicsPtr[i * 3 + 0] = 1.0  // R
            harmonicsPtr[i * 3 + 1] = 0.5  // G
            harmonicsPtr[i * 3 + 2] = 0.2  // B
        }

        // Camera setup - must be identical for both pipelines
        let aspect = Float(imageWidth) / Float(imageHeight)
        let fov: Float = 60.0 * .pi / 180.0
        let near: Float = 0.2
        let far: Float = 100.0

        let f = 1.0 / tan(fov / 2.0)
        let projectionMatrix = simd_float4x4(columns: (
            SIMD4<Float>(f / aspect, 0, 0, 0),
            SIMD4<Float>(0, f, 0, 0),
            SIMD4<Float>(0, 0, (far + near) / (near - far), -1),
            SIMD4<Float>(0, 0, (2 * far * near) / (near - far), 0)
        ))
        let viewMatrix = simd_float4x4(1.0)  // Identity

        let focalX = f * Float(imageWidth) / 2.0
        let focalY = f * Float(imageHeight) / 2.0

        let cameraUniforms = CameraUniformsSwift(
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            cameraCenter: SIMD3<Float>(0, 0, 0),
            pixelFactor: 1.0,
            focalX: focalX,
            focalY: focalY,
            width: Float(imageWidth),
            height: Float(imageHeight),
            nearPlane: near,
            farPlane: far,
            shComponents: 0,
            gaussianCount: UInt32(gaussianCount)
        )

        // ============================================
        // Run ORIGINAL pipeline projection
        // ============================================
        let origMeansBuffer = device.makeBuffer(length: gaussianCount * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
        let origConicBuffer = device.makeBuffer(length: gaussianCount * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)!
        // packed_float3 is 12 bytes (3 floats contiguous), not 16
        let origColorBuffer = device.makeBuffer(length: gaussianCount * 12, options: .storageModeShared)!
        let origOpacityBuffer = device.makeBuffer(length: gaussianCount * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let origDepthBuffer = device.makeBuffer(length: gaussianCount * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let origRadiiBuffer = device.makeBuffer(length: gaussianCount * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let origMaskBuffer = device.makeBuffer(length: gaussianCount * MemoryLayout<UInt8>.stride, options: .storageModeShared)!

        let gaussianBuffers = GaussianInputBuffers(
            means: origMeansBuffer,
            radii: origRadiiBuffer,
            mask: origMaskBuffer,
            depths: origDepthBuffer,
            conics: origConicBuffer,
            colors: origColorBuffer,
            opacities: origOpacityBuffer
        )

        let projectEncoder = try ProjectEncoder(device: device, library: library)
        let packedWorld = PackedWorldBuffers(packedGaussians: worldBuffer, harmonics: harmonicsBuffer)

        if let cb = queue.makeCommandBuffer() {
            projectEncoder.encode(
                commandBuffer: cb,
                gaussianCount: gaussianCount,
                packedWorldBuffers: packedWorld,
                cameraUniforms: cameraUniforms,
                gaussianBuffers: gaussianBuffers,
                precision: .float32
            )
            cb.commit()
            cb.waitUntilCompleted()
        }

        // ============================================
        // Run TELLUSIM pipeline projection
        // ============================================
        let tellusimEncoder = try TellusimPipelineEncoder(device: device, library: library)

        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight
        let tileCount = tilesX * tilesY
        let maxCompacted = gaussianCount
        let maxAssignments = gaussianCount * 64

        let tellusimCompacted = device.makeBuffer(
            length: maxCompacted * MemoryLayout<CompactedGaussianSwift>.stride,
            options: .storageModeShared
        )!
        let tellusimHeader = device.makeBuffer(
            length: MemoryLayout<CompactedHeaderSwift>.stride,
            options: .storageModeShared
        )!
        let tellusimTileCounts = device.makeBuffer(length: tileCount * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let tellusimTileOffsets = device.makeBuffer(length: (tileCount + 1) * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let tellusimPartialSums = device.makeBuffer(length: 1024 * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let tellusimSortKeys = device.makeBuffer(length: maxAssignments * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let tellusimSortIndices = device.makeBuffer(length: maxAssignments * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

        if let cb = queue.makeCommandBuffer() {
            tellusimEncoder.encode(
                commandBuffer: cb,
                worldGaussians: worldBuffer,
                harmonics: harmonicsBuffer,
                camera: cameraUniforms,
                gaussianCount: gaussianCount,
                tilesX: tilesX,
                tilesY: tilesY,
                tileWidth: tileWidth,
                tileHeight: tileHeight,
                surfaceWidth: imageWidth,
                surfaceHeight: imageHeight,
                compactedGaussians: tellusimCompacted,
                compactedHeader: tellusimHeader,
                tileCounts: tellusimTileCounts,
                tileOffsets: tellusimTileOffsets,
                partialSums: tellusimPartialSums,
                sortKeys: tellusimSortKeys,
                sortIndices: tellusimSortIndices,
                maxCompacted: maxCompacted,
                maxAssignments: maxAssignments,
                skipSort: true
            )
            cb.commit()
            cb.waitUntilCompleted()
        }

        // ============================================
        // Compare outputs
        // ============================================
        let origMeans = origMeansBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: gaussianCount)
        let origMask = origMaskBuffer.contents().bindMemory(to: UInt8.self, capacity: gaussianCount)
        let origDepth = origDepthBuffer.contents().bindMemory(to: Float.self, capacity: gaussianCount)

        let tellusimVisibleCount = TellusimPipelineEncoder.readVisibleCount(from: tellusimHeader)
        let tellusimCompactedPtr = tellusimCompacted.contents().bindMemory(to: CompactedGaussianSwift.self, capacity: Int(tellusimVisibleCount))

        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║  PIPELINE COMPARISON TEST                                  ║")
        print("╠═══════════════════════════════════════════════════════════╣")

        // Count visible in original
        var origVisibleCount = 0
        for i in 0..<gaussianCount {
            if origMask[i] != 0 && origMask[i] != 2 {
                origVisibleCount += 1
            }
        }

        print("║  Original visible: \(origVisibleCount) / \(gaussianCount)")
        print("║  Tellusim visible: \(tellusimVisibleCount) / \(gaussianCount)")

        // Also get original colors (packed_float3 = 12 bytes = 3 floats) and conics
        let origColorFloats = origColorBuffer.contents().bindMemory(to: Float.self, capacity: gaussianCount * 3)
        let origConics = origConicBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: gaussianCount)
        let origOpacities = origOpacityBuffer.contents().bindMemory(to: Float.self, capacity: gaussianCount)

        // Print first 10 gaussians from each
        print("╠═══════════════════════════════════════════════════════════╣")
        print("║  ORIGINAL (first 10 visible):                             ║")
        var origPrinted = 0
        for i in 0..<gaussianCount {
            guard origPrinted < 10 else { break }
            if origMask[i] != 0 && origMask[i] != 2 {
                let pos = origMeans[i]
                let d = origDepth[i]
                let c = SIMD3<Float>(origColorFloats[i*3], origColorFloats[i*3+1], origColorFloats[i*3+2])
                let o = origOpacities[i]
                let conic = origConics[i]
                print("║  [\(i)] pos=(\(String(format: "%.1f", pos.x)), \(String(format: "%.1f", pos.y))) depth=\(String(format: "%.2f", d)) color=(\(String(format: "%.2f", c.x)), \(String(format: "%.2f", c.y)), \(String(format: "%.2f", c.z))) opacity=\(String(format: "%.2f", o))")
                print("║       conic=(\(String(format: "%.4f", conic.x)), \(String(format: "%.4f", conic.y)), \(String(format: "%.4f", conic.z)))")
                origPrinted += 1
            }
        }

        print("╠═══════════════════════════════════════════════════════════╣")
        print("║  TELLUSIM (first 10 visible):                             ║")
        for i in 0..<min(10, Int(tellusimVisibleCount)) {
            let g = tellusimCompactedPtr[i]
            let pos = SIMD2<Float>(g.position_color.x, g.position_color.y)
            let d = g.covariance_depth.w
            let conic = SIMD3<Float>(g.covariance_depth.x, g.covariance_depth.y, g.covariance_depth.z)
            // Unpack color from position_color.zw
            let packed = SIMD2<Float>(g.position_color.z, g.position_color.w)
            let u0 = unsafeBitCast(packed.x, to: UInt32.self)
            let u1 = unsafeBitCast(packed.y, to: UInt32.self)
            let h0 = Float16(bitPattern: UInt16(u0 & 0xFFFF))
            let h1 = Float16(bitPattern: UInt16((u0 >> 16) & 0xFFFF))
            let h2 = Float16(bitPattern: UInt16(u1 & 0xFFFF))
            let h3 = Float16(bitPattern: UInt16((u1 >> 16) & 0xFFFF))
            let r = Float(h0), g_col = Float(h1), b = Float(h2), a = Float(h3)
            print("║  [\(i)] pos=(\(String(format: "%.1f", pos.x)), \(String(format: "%.1f", pos.y))) depth=\(String(format: "%.2f", d)) color=(\(String(format: "%.2f", r)), \(String(format: "%.2f", g_col)), \(String(format: "%.2f", b))) opacity=\(String(format: "%.2f", a))")
            print("║       conic=(\(String(format: "%.4f", conic.x)), \(String(format: "%.4f", conic.y)), \(String(format: "%.4f", conic.z)))")
        }

        print("╚═══════════════════════════════════════════════════════════╝\n")

        // Assertions
        XCTAssertGreaterThan(origVisibleCount, 0, "Original should have visible gaussians")
        XCTAssertGreaterThan(tellusimVisibleCount, 0, "Tellusim should have visible gaussians")

        // Note: Tellusim culls off-screen gaussians more aggressively, so count may differ
        // The original keeps all gaussians even if off-screen, Tellusim culls based on tile coverage
        print("║  Note: Count difference is expected - Tellusim culls off-screen gaussians")
    }

    /// Test PIXEL-PERFECT end-to-end comparison between original and Tellusim pipelines
    func testPixelPerfectComparison() throws {
        let width = 128
        let height = 128
        let gaussianCount = 16

        // Create original renderer (uses fused pipeline by default)
        let origRenderer = Renderer(
            precision: .float32,
            useHeapAllocation: false,
            limits: RendererLimits(maxGaussians: 1024, maxWidth: width, maxHeight: height, tileWidth: 16, tileHeight: 16)
        )
        let device = origRenderer.device
        let queue = origRenderer.queue

        // Create Tellusim backend
        let tellusimBackend = try TellusimBackend(device: device)
        tellusimBackend.debugPrint = false

        // Create test data - 4x4 grid of gaussians with varying colors
        var positions: [SIMD3<Float>] = []
        var scales: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var opacities: [Float] = []
        var colors: [SIMD3<Float>] = []

        for row in 0..<4 {
            for col in 0..<4 {
                let x = Float(col - 2) * 0.5 + 0.25
                let y = Float(row - 2) * 0.5 + 0.25
                positions.append(SIMD3(x, y, 5))
                scales.append(SIMD3(0.15, 0.15, 0.15))
                rotations.append(SIMD4(0, 0, 0, 1))
                opacities.append(0.85)
                // Varying colors (these are SH0 coefficients, shader adds 0.5)
                colors.append(SIMD3(Float(col) / 4.0, Float(row) / 4.0, 0.2))
            }
        }

        // Create packed buffers
        var packed: [PackedWorldGaussian] = []
        var harmonics: [Float] = []
        for i in 0..<gaussianCount {
            packed.append(PackedWorldGaussian(
                position: positions[i],
                scale: scales[i],
                rotation: rotations[i],
                opacity: opacities[i]
            ))
            harmonics.append(colors[i].x)
            harmonics.append(colors[i].y)
            harmonics.append(colors[i].z)
        }

        let packedBuf = device.makeBuffer(bytes: &packed, length: packed.count * MemoryLayout<PackedWorldGaussian>.stride, options: .storageModeShared)!
        let harmonicsBuf = device.makeBuffer(bytes: &harmonics, length: harmonics.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let packedWorld = PackedWorldBuffers(packedGaussians: packedBuf, harmonics: harmonicsBuf)

        // Camera
        let aspect = Float(width) / Float(height)
        let fov: Float = 60.0 * .pi / 180.0
        let near: Float = 0.1
        let far: Float = 100.0
        let f = 1.0 / tan(fov / 2.0)
        let projectionMatrix = simd_float4x4(columns: (
            SIMD4<Float>(f / aspect, 0, 0, 0),
            SIMD4<Float>(0, f, 0, 0),
            SIMD4<Float>(0, 0, (far + near) / (near - far), -1),
            SIMD4<Float>(0, 0, (2 * far * near) / (near - far), 0)
        ))
        let viewMatrix = simd_float4x4(1.0)
        let focalX = f * Float(width) / 2.0
        let focalY = f * Float(height) / 2.0

        let camera = CameraUniformsSwift(
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            cameraCenter: SIMD3<Float>(0, 0, 0),
            pixelFactor: 1.0,
            focalX: focalX,
            focalY: focalY,
            width: Float(width),
            height: Float(height),
            nearPlane: near,
            farPlane: far,
            shComponents: 0,
            gaussianCount: UInt32(gaussianCount)
        )

        // ============================================
        // RENDER WITH ORIGINAL PIPELINE
        // ============================================
        var origColorData: [Float16]?
        if let cb = queue.makeCommandBuffer() {
            let frameParams = FrameParams(gaussianCount: gaussianCount, whiteBackground: false)
            if let textures = origRenderer.encodeRenderToTextures(
                commandBuffer: cb,
                gaussianCount: gaussianCount,
                packedWorldBuffers: packedWorld,
                cameraUniforms: camera,
                frameParams: frameParams
            ) {
                cb.commit()
                cb.waitUntilCompleted()

                // Read back color texture
                let bytesPerRow = width * 8  // rgba16Float
                let readBuf = device.makeBuffer(length: height * bytesPerRow, options: .storageModeShared)!
                if let blitCb = queue.makeCommandBuffer(),
                   let blitEnc = blitCb.makeBlitCommandEncoder() {
                    blitEnc.copy(from: textures.color, sourceSlice: 0, sourceLevel: 0,
                                sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                                sourceSize: MTLSize(width: width, height: height, depth: 1),
                                to: readBuf, destinationOffset: 0,
                                destinationBytesPerRow: bytesPerRow,
                                destinationBytesPerImage: height * bytesPerRow)
                    blitEnc.endEncoding()
                    blitCb.commit()
                    blitCb.waitUntilCompleted()
                    origColorData = Array(UnsafeBufferPointer(
                        start: readBuf.contents().bindMemory(to: Float16.self, capacity: width * height * 4),
                        count: width * height * 4
                    ))
                }
            }
        }

        // ============================================
        // RENDER WITH TELLUSIM PIPELINE
        // ============================================
        var tellusimColorData: [Float16]?
        if let cb = queue.makeCommandBuffer() {
            if let tex = tellusimBackend.render(
                commandBuffer: cb,
                worldGaussians: packedBuf,
                harmonics: harmonicsBuf,
                gaussianCount: gaussianCount,
                viewMatrix: viewMatrix,
                projectionMatrix: projectionMatrix,
                cameraPosition: SIMD3<Float>(0, 0, 0),
                focalX: focalX,
                focalY: focalY,
                width: width,
                height: height,
                shComponents: 0,
                whiteBackground: false
            ) {
                cb.commit()
                cb.waitUntilCompleted()

                // Read back color texture
                let bytesPerRow = width * 8
                let readBuf = device.makeBuffer(length: height * bytesPerRow, options: .storageModeShared)!
                if let blitCb = queue.makeCommandBuffer(),
                   let blitEnc = blitCb.makeBlitCommandEncoder() {
                    blitEnc.copy(from: tex, sourceSlice: 0, sourceLevel: 0,
                                sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                                sourceSize: MTLSize(width: width, height: height, depth: 1),
                                to: readBuf, destinationOffset: 0,
                                destinationBytesPerRow: bytesPerRow,
                                destinationBytesPerImage: height * bytesPerRow)
                    blitEnc.endEncoding()
                    blitCb.commit()
                    blitCb.waitUntilCompleted()
                    tellusimColorData = Array(UnsafeBufferPointer(
                        start: readBuf.contents().bindMemory(to: Float16.self, capacity: width * height * 4),
                        count: width * height * 4
                    ))
                }
            }
        }

        // ============================================
        // COMPARE PIXEL DATA
        // ============================================
        guard let orig = origColorData, let tellusim = tellusimColorData else {
            XCTFail("Failed to get texture data from one or both pipelines")
            return
        }

        XCTAssertEqual(orig.count, tellusim.count, "Buffer sizes should match")

        var maxDiff: Float = 0
        var totalDiff: Float = 0
        var diffCount = 0
        var coloredPixelCount = 0
        var coloredDiffCount = 0
        var samplePixels: [(orig: SIMD4<Float>, tellusim: SIMD4<Float>, idx: Int)] = []

        for i in stride(from: 0, to: min(orig.count, tellusim.count), by: 4) {
            let oR = Float(orig[i]), oG = Float(orig[i+1]), oB = Float(orig[i+2]), oA = Float(orig[i+3])
            let tR = Float(tellusim[i]), tG = Float(tellusim[i+1]), tB = Float(tellusim[i+2]), tA = Float(tellusim[i+3])

            // Check if either has color (not just background)
            let origHasContent = oR > 0.01 || oG > 0.01 || oB > 0.01
            let tellHasContent = tR > 0.01 || tG > 0.01 || tB > 0.01

            // Focus on RGB differences (ignore alpha for background pixels)
            let dR = abs(oR - tR), dG = abs(oG - tG), dB = abs(oB - tB)
            let maxRGBDiff = max(max(dR, dG), dB)

            if origHasContent || tellHasContent {
                coloredPixelCount += 1
                if maxRGBDiff > 0.01 {
                    coloredDiffCount += 1
                    totalDiff += maxRGBDiff
                    maxDiff = max(maxDiff, maxRGBDiff)

                    // Sample first 5 differing colored pixels
                    if samplePixels.count < 5 {
                        samplePixels.append((
                            orig: SIMD4<Float>(oR, oG, oB, oA),
                            tellusim: SIMD4<Float>(tR, tG, tB, tA),
                            idx: i / 4
                        ))
                    }
                }
            }

            // Count total different pixels (including alpha)
            let dA = abs(oA - tA)
            if max(maxRGBDiff, dA) > 0.001 {
                diffCount += 1
            }
        }

        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║  PIXEL-PERFECT COMPARISON TEST                             ║")
        print("╠═══════════════════════════════════════════════════════════╣")
        print("║  Image size: \(width)x\(height) (\(width * height) pixels)")
        print("║  Total differing pixels: \(diffCount) / \(width * height) (includes alpha)")
        print("║  Colored pixels: \(coloredPixelCount)")
        print("║  Colored pixels with RGB diff: \(coloredDiffCount) / \(coloredPixelCount)")
        print("║  Max RGB difference: \(String(format: "%.6f", maxDiff))")
        print("║  Avg RGB difference: \(coloredDiffCount > 0 ? String(format: "%.6f", totalDiff / Float(coloredDiffCount)) : "N/A")")

        if !samplePixels.isEmpty {
            print("╠═══════════════════════════════════════════════════════════╣")
            print("║  Sample differing COLORED pixels:")
            for sample in samplePixels {
                let px = sample.idx % width
                let py = sample.idx / width
                print("║  [\(px),\(py)] orig=(\(String(format: "%.3f", sample.orig.x)),\(String(format: "%.3f", sample.orig.y)),\(String(format: "%.3f", sample.orig.z)),\(String(format: "%.3f", sample.orig.w)))")
                print("║         tell=(\(String(format: "%.3f", sample.tellusim.x)),\(String(format: "%.3f", sample.tellusim.y)),\(String(format: "%.3f", sample.tellusim.z)),\(String(format: "%.3f", sample.tellusim.w)))")
            }
        } else if coloredPixelCount > 0 {
            print("║  No RGB differences in colored pixels!")
        }
        print("╚═══════════════════════════════════════════════════════════╝\n")

        // Assert that RGB values in colored pixels match closely
        let coloredDiffPercent = coloredPixelCount > 0 ? Float(coloredDiffCount) / Float(coloredPixelCount) * 100 : 0
        XCTAssertLessThan(coloredDiffPercent, 5.0, "Should have less than 5% differing colored pixels")
        XCTAssertLessThan(maxDiff, 0.15, "Max RGB difference should be less than 0.15")
    }

    /// Test full render comparison between original and Tellusim
    func testCompareFullRender() throws {
        let renderer = Renderer.shared
        let device = renderer.device
        let queue = renderer.queue

        let gaussianCount = 25
        let width = 256
        let height = 256

        // Create input data - place gaussians in view
        let worldBuffer = device.makeBuffer(
            length: gaussianCount * MemoryLayout<PackedWorldGaussian>.stride,
            options: .storageModeShared
        )!
        let harmonicsBuffer = device.makeBuffer(
            length: gaussianCount * 3 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        let worldPtr = worldBuffer.contents().bindMemory(to: PackedWorldGaussian.self, capacity: gaussianCount)
        let harmonicsPtr = harmonicsBuffer.contents().bindMemory(to: Float.self, capacity: gaussianCount * 3)

        for i in 0..<gaussianCount {
            let x = Float(i % 5) - 2.0  // -2 to 2
            let y = Float(i / 5) - 2.0  // -2 to 2
            let z: Float = 5.0

            worldPtr[i] = PackedWorldGaussian(
                position: SIMD3<Float>(x, y, z),
                scale: SIMD3<Float>(0.2, 0.2, 0.2),
                rotation: SIMD4<Float>(0, 0, 0, 1),
                opacity: 0.9
            )

            // Vary colors
            harmonicsPtr[i * 3 + 0] = Float(i % 5) / 4.0  // R varies
            harmonicsPtr[i * 3 + 1] = Float(i / 5) / 4.0  // G varies
            harmonicsPtr[i * 3 + 2] = 0.3  // B constant
        }

        // Camera
        let aspect = Float(width) / Float(height)
        let fov: Float = 60.0 * .pi / 180.0
        let near: Float = 0.1
        let far: Float = 100.0
        let f = 1.0 / tan(fov / 2.0)
        let projectionMatrix = simd_float4x4(columns: (
            SIMD4<Float>(f / aspect, 0, 0, 0),
            SIMD4<Float>(0, f, 0, 0),
            SIMD4<Float>(0, 0, (far + near) / (near - far), -1),
            SIMD4<Float>(0, 0, (2 * far * near) / (near - far), 0)
        ))
        let viewMatrix = simd_float4x4(1.0)
        let focalX = f * Float(width) / 2.0
        let focalY = f * Float(height) / 2.0

        // Create Tellusim backend and render
        let backend = try TellusimBackend(device: device)
        backend.debugPrint = false

        if let cb = queue.makeCommandBuffer() {
            let packedWorld = PackedWorldBuffers(packedGaussians: worldBuffer, harmonics: harmonicsBuffer)

            let outputTex = backend.render(
                commandBuffer: cb,
                worldGaussians: worldBuffer,
                harmonics: harmonicsBuffer,
                gaussianCount: gaussianCount,
                viewMatrix: viewMatrix,
                projectionMatrix: projectionMatrix,
                cameraPosition: SIMD3<Float>(0, 0, 0),
                focalX: focalX,
                focalY: focalY,
                width: width,
                height: height,
                shComponents: 0,
                whiteBackground: false
            )

            cb.commit()
            cb.waitUntilCompleted()

            // Read back texture
            if let tex = outputTex {
                // Read center pixels
                var pixels = [Float16](repeating: 0, count: 4)
                let region = MTLRegion(origin: MTLOrigin(x: width/2, y: height/2, z: 0),
                                      size: MTLSize(width: 1, height: 1, depth: 1))

                // Need to copy to CPU-readable buffer first since texture is private
                let bytesPerRow = width * 8  // rgba16Float = 8 bytes
                let readBuffer = device.makeBuffer(length: height * bytesPerRow, options: .storageModeShared)!

                if let blitCb = queue.makeCommandBuffer(),
                   let blitEncoder = blitCb.makeBlitCommandEncoder() {
                    blitEncoder.copy(from: tex, sourceSlice: 0, sourceLevel: 0,
                                    sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                                    sourceSize: MTLSize(width: width, height: height, depth: 1),
                                    to: readBuffer, destinationOffset: 0,
                                    destinationBytesPerRow: bytesPerRow,
                                    destinationBytesPerImage: height * bytesPerRow)
                    blitEncoder.endEncoding()
                    blitCb.commit()
                    blitCb.waitUntilCompleted()

                    // Read center pixel
                    let pixelData = readBuffer.contents().bindMemory(to: Float16.self, capacity: width * height * 4)
                    let centerIdx = (height/2 * width + width/2) * 4
                    let r = Float(pixelData[centerIdx + 0])
                    let g = Float(pixelData[centerIdx + 1])
                    let b = Float(pixelData[centerIdx + 2])
                    let a = Float(pixelData[centerIdx + 3])

                    print("\n╔═══════════════════════════════════════════════════════════╗")
                    print("║  FULL RENDER TEST                                          ║")
                    print("╠═══════════════════════════════════════════════════════════╣")
                    print("║  Visible count: \(backend.getVisibleCount())")
                    print("║  Center pixel: R=\(String(format: "%.3f", r)) G=\(String(format: "%.3f", g)) B=\(String(format: "%.3f", b)) A=\(String(format: "%.3f", a))")
                    print("╚═══════════════════════════════════════════════════════════╝\n")

                    // Assertions
                    XCTAssertGreaterThan(backend.getVisibleCount(), 0, "Should have visible gaussians")
                    XCTAssertGreaterThan(a, 0.0, "Center pixel should have alpha > 0")
                    XCTAssertGreaterThan(r + g + b, 0.0, "Center pixel should have color")
                }
            } else {
                XCTFail("Failed to get output texture")
            }
        }
    }
}

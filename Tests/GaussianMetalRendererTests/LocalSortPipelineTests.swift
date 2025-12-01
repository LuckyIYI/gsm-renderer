import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Tests for the Tellusim-style pipeline (fused project + compact + count + scatter)
final class LocalSortPipelineTests: XCTestCase {
    private let tileWidth = 32
    private let tileHeight = 16
    private let imageWidth = 1920
    private let imageHeight = 1080

    /// Test basic pipeline execution and verify visible count
    func testBasicPipelineExecution() throws {
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let encoder = try LocalSortPipelineEncoder(device: device, library: library)

        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight
        let tileCount = tilesX * tilesY

        let gaussianCount = 10_000
        let maxCompacted = gaussianCount  // At most all are visible
        let maxAssignments = gaussianCount * 16  // Avg 16 tiles per gaussian

        // Create buffers
        let worldBuffer = device.makeBuffer(
            length: gaussianCount * MemoryLayout<PackedWorldGaussian>.stride,
            options: .storageModeShared
        )!
        let harmonicsBuffer = device.makeBuffer(
            length: gaussianCount * 3 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let compactedBuffer = device.makeBuffer(
            length: maxCompacted * MemoryLayout<CompactedGaussianSwift>.stride,
            options: .storageModeShared
        )!
        let headerBuffer = device.makeBuffer(
            length: MemoryLayout<CompactedHeaderSwift>.stride,
            options: .storageModeShared
        )!
        let tileCountsBuffer = device.makeBuffer(
            length: tileCount * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )!
        let tileOffsetsBuffer = device.makeBuffer(
            length: (tileCount + 1) * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )!
        let partialSumsBuffer = device.makeBuffer(
            length: 1024 * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )!
        let sortKeysBuffer = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )!
        let sortIndicesBuffer = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )!

        // Populate test data - gaussians in front of camera
        populateTestData(
            worldBuffer: worldBuffer,
            harmonicsBuffer: harmonicsBuffer,
            count: gaussianCount
        )

        // Create camera uniforms
        let camera = createTestCamera()

        // Run pipeline
        guard let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        encoder.encode(
            commandBuffer: cb,
            worldGaussians: worldBuffer,
            harmonics: harmonicsBuffer,
            camera: camera,
            gaussianCount: gaussianCount,
            tilesX: tilesX,
            tilesY: tilesY,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            surfaceWidth: imageWidth,
            surfaceHeight: imageHeight,
            compactedGaussians: compactedBuffer,
            compactedHeader: headerBuffer,
            tileCounts: tileCountsBuffer,
            tileOffsets: tileOffsetsBuffer,
            partialSums: partialSumsBuffer,
            sortKeys: sortKeysBuffer,
            sortIndices: sortIndicesBuffer,
            maxCompacted: maxCompacted,
            maxAssignments: maxAssignments
        )

        cb.commit()
        cb.waitUntilCompleted()

        // Read results
        let visibleCount = LocalSortPipelineEncoder.readVisibleCount(from: headerBuffer)
        let overflow = LocalSortPipelineEncoder.readOverflow(from: headerBuffer)

        print("Visible count: \(visibleCount) / \(gaussianCount)")
        print("Overflow: \(overflow)")

        // Verify some gaussians are visible (not all will be due to culling)
        XCTAssertGreaterThan(visibleCount, 0, "Should have some visible gaussians")
        XCTAssertLessThanOrEqual(Int(visibleCount), gaussianCount, "Can't have more visible than total")
        XCTAssertFalse(overflow, "Should not overflow")

        // Verify tile counts sum to total assignments
        let tileCounts = tileCountsBuffer.contents().bindMemory(to: UInt32.self, capacity: tileCount)
        var totalAssignments: UInt64 = 0
        for i in 0..<tileCount {
            totalAssignments += UInt64(tileCounts[i])
        }
        print("Total tile assignments: \(totalAssignments)")
        XCTAssertGreaterThan(totalAssignments, 0, "Should have some tile assignments")

        // Verify compacted data looks reasonable
        let compacted = compactedBuffer.contents().bindMemory(to: CompactedGaussianSwift.self, capacity: Int(visibleCount))
        for i in 0..<min(5, Int(visibleCount)) {
            let g = compacted[i]
            XCTAssertGreaterThan(g.covariance_depth.w, 0, "Depth should be positive")
            XCTAssertLessThan(g.min_tile.x, g.max_tile.x, "Min tile x should be less than max")
            XCTAssertLessThan(g.min_tile.y, g.max_tile.y, "Min tile y should be less than max")
        }
    }

    /// Test full pipeline including render and verify output
    func testFullPipelineWithRender() throws {
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let encoder = try LocalSortPipelineEncoder(device: device, library: library)

        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight
        let tileCount = tilesX * tilesY

        let gaussianCount = 50_000
        let maxCompacted = gaussianCount
        let maxAssignments = gaussianCount * 16

        // Create buffers
        let worldBuffer = device.makeBuffer(
            length: gaussianCount * MemoryLayout<PackedWorldGaussian>.stride,
            options: .storageModeShared
        )!
        let harmonicsBuffer = device.makeBuffer(
            length: gaussianCount * 3 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let compactedBuffer = device.makeBuffer(
            length: maxCompacted * MemoryLayout<CompactedGaussianSwift>.stride,
            options: .storageModeShared
        )!
        let headerBuffer = device.makeBuffer(
            length: MemoryLayout<CompactedHeaderSwift>.stride,
            options: .storageModeShared
        )!
        let tileCountsBuffer = device.makeBuffer(
            length: tileCount * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )!
        let tileOffsetsBuffer = device.makeBuffer(
            length: (tileCount + 1) * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )!
        let partialSumsBuffer = device.makeBuffer(
            length: 1024 * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )!
        let sortKeysBuffer = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )!
        let sortIndicesBuffer = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )!
        // Temp buffers for per-tile sort
        let tempSortKeys = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!
        let tempSortIndices = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!

        // Create output textures
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: imageWidth,
            height: imageHeight,
            mipmapped: false
        )
        colorDesc.usage = [.shaderRead, .shaderWrite]
        colorDesc.storageMode = .shared
        let colorTexture = device.makeTexture(descriptor: colorDesc)!

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: imageWidth,
            height: imageHeight,
            mipmapped: false
        )
        depthDesc.usage = [.shaderRead, .shaderWrite]
        depthDesc.storageMode = .shared
        let depthTexture = device.makeTexture(descriptor: depthDesc)!

        // Populate test data - dense cluster of gaussians
        populateTestData(
            worldBuffer: worldBuffer,
            harmonicsBuffer: harmonicsBuffer,
            count: gaussianCount
        )

        let camera = createTestCamera()

        // Run full pipeline
        guard let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        // Step 1-6: Project, scan, scatter, sort
        encoder.encode(
            commandBuffer: cb,
            worldGaussians: worldBuffer,
            harmonics: harmonicsBuffer,
            camera: camera,
            gaussianCount: gaussianCount,
            tilesX: tilesX,
            tilesY: tilesY,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            surfaceWidth: imageWidth,
            surfaceHeight: imageHeight,
            compactedGaussians: compactedBuffer,
            compactedHeader: headerBuffer,
            tileCounts: tileCountsBuffer,
            tileOffsets: tileOffsetsBuffer,
            partialSums: partialSumsBuffer,
            sortKeys: sortKeysBuffer,
            sortIndices: sortIndicesBuffer,
            maxCompacted: maxCompacted,
            maxAssignments: maxAssignments,
            useHalfWorld: false,
            skipSort: false,
            tempSortKeys: tempSortKeys,
            tempSortIndices: tempSortIndices
        )

        // Step 7: Render
        encoder.encodeRender(
            commandBuffer: cb,
            compactedGaussians: compactedBuffer,
            tileOffsets: tileOffsetsBuffer,
            tileCounts: tileCountsBuffer,
            sortedIndices: sortIndicesBuffer,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            tilesX: tilesX,
            tilesY: tilesY,
            width: imageWidth,
            height: imageHeight,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            whiteBackground: false
        )

        cb.commit()
        cb.waitUntilCompleted()

        // Verify results
        let visibleCount = LocalSortPipelineEncoder.readVisibleCount(from: headerBuffer)
        print("Full pipeline - Visible: \(visibleCount) / \(gaussianCount)")

        XCTAssertGreaterThan(visibleCount, 0, "Should have visible gaussians")

        // Check render output - sample center pixels
        var nonBlackPixels = 0
        let centerX = imageWidth / 2
        let centerY = imageHeight / 2
        let sampleSize = 100

        // Read pixels from center region
        for dy in -sampleSize/2..<sampleSize/2 {
            for dx in -sampleSize/2..<sampleSize/2 {
                let x = centerX + dx
                let y = centerY + dy
                guard x >= 0 && x < imageWidth && y >= 0 && y < imageHeight else { continue }

                var pixel = SIMD4<Float16>(0, 0, 0, 0)
                colorTexture.getBytes(
                    &pixel,
                    bytesPerRow: imageWidth * 8,
                    from: MTLRegion(origin: MTLOrigin(x: x, y: y, z: 0), size: MTLSize(width: 1, height: 1, depth: 1)),
                    mipmapLevel: 0
                )

                // Check if pixel has any color (alpha > 0 or RGB > 0)
                if pixel.w > 0.01 || pixel.x > 0.01 || pixel.y > 0.01 || pixel.z > 0.01 {
                    nonBlackPixels += 1
                }
            }
        }

        print("Non-black pixels in center region: \(nonBlackPixels) / \(sampleSize * sampleSize)")
        XCTAssertGreaterThan(nonBlackPixels, 0, "Render should produce visible output in center")
    }

    /// Full end-to-end performance test including render
    func testEndToEndPerformance() throws {
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let encoder = try LocalSortPipelineEncoder(device: device, library: library)

        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight
        let tileCount = tilesX * tilesY

        let sizes = [100_000, 500_000, 1_000_000]
        let iterations = 10

        var summaries: [String] = []

        // Create textures once (reused)
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: imageWidth,
            height: imageHeight,
            mipmapped: false
        )
        colorDesc.usage = [.shaderRead, .shaderWrite]
        colorDesc.storageMode = .private
        let colorTexture = device.makeTexture(descriptor: colorDesc)!

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: imageWidth,
            height: imageHeight,
            mipmapped: false
        )
        depthDesc.usage = [.shaderRead, .shaderWrite]
        depthDesc.storageMode = .private
        let depthTexture = device.makeTexture(descriptor: depthDesc)!

        for gaussianCount in sizes {
            let maxCompacted = gaussianCount
            let maxAssignments = gaussianCount * 32  // Need more for uniform test data

            // Create buffers
            let worldBuffer = device.makeBuffer(
                length: gaussianCount * MemoryLayout<PackedWorldGaussian>.stride,
                options: .storageModeShared
            )!
            let harmonicsBuffer = device.makeBuffer(
                length: gaussianCount * 3 * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )!
            let compactedBuffer = device.makeBuffer(
                length: maxCompacted * MemoryLayout<CompactedGaussianSwift>.stride,
                options: .storageModePrivate
            )!
            let headerBuffer = device.makeBuffer(
                length: MemoryLayout<CompactedHeaderSwift>.stride,
                options: .storageModeShared
            )!
            let tileCountsBuffer = device.makeBuffer(
                length: tileCount * MemoryLayout<UInt32>.stride,
                options: .storageModePrivate
            )!
            let tileOffsetsBuffer = device.makeBuffer(
                length: (tileCount + 1) * MemoryLayout<UInt32>.stride,
                options: .storageModePrivate
            )!
            let partialSumsBuffer = device.makeBuffer(
                length: 1024 * MemoryLayout<UInt32>.stride,
                options: .storageModePrivate
            )!
            let sortKeysBuffer = device.makeBuffer(
                length: maxAssignments * MemoryLayout<UInt32>.stride,
                options: .storageModePrivate
            )!
            let sortIndicesBuffer = device.makeBuffer(
                length: maxAssignments * MemoryLayout<UInt32>.stride,
                options: .storageModePrivate
            )!
            let tempSortKeys = device.makeBuffer(
                length: maxAssignments * MemoryLayout<UInt32>.stride,
                options: .storageModePrivate
            )!
            let tempSortIndices = device.makeBuffer(
                length: maxAssignments * MemoryLayout<UInt32>.stride,
                options: .storageModePrivate
            )!

            populateTestData(
                worldBuffer: worldBuffer,
                harmonicsBuffer: harmonicsBuffer,
                count: gaussianCount
            )

            let camera = createTestCamera()

            // Warm up
            for _ in 0..<3 {
                if let cb = queue.makeCommandBuffer() {
                    encoder.encode(
                        commandBuffer: cb,
                        worldGaussians: worldBuffer,
                        harmonics: harmonicsBuffer,
                        camera: camera,
                        gaussianCount: gaussianCount,
                        tilesX: tilesX,
                        tilesY: tilesY,
                        tileWidth: tileWidth,
                        tileHeight: tileHeight,
                        surfaceWidth: imageWidth,
                        surfaceHeight: imageHeight,
                        compactedGaussians: compactedBuffer,
                        compactedHeader: headerBuffer,
                        tileCounts: tileCountsBuffer,
                        tileOffsets: tileOffsetsBuffer,
                        partialSums: partialSumsBuffer,
                        sortKeys: sortKeysBuffer,
                        sortIndices: sortIndicesBuffer,
                        maxCompacted: maxCompacted,
                        maxAssignments: maxAssignments,
                        useHalfWorld: false,
                        skipSort: false,
                        tempSortKeys: tempSortKeys,
                        tempSortIndices: tempSortIndices
                    )
                    encoder.encodeRender(
                        commandBuffer: cb,
                        compactedGaussians: compactedBuffer,
                        tileOffsets: tileOffsetsBuffer,
                        tileCounts: tileCountsBuffer,
                        sortedIndices: sortIndicesBuffer,
                        colorTexture: colorTexture,
                        depthTexture: depthTexture,
                        tilesX: tilesX,
                        tilesY: tilesY,
                        width: imageWidth,
                        height: imageHeight,
                        tileWidth: tileWidth,
                        tileHeight: tileHeight,
                        whiteBackground: false
                    )
                    cb.commit()
                    cb.waitUntilCompleted()
                }
            }

            // Benchmark full pipeline including render
            var times: [Double] = []
            for _ in 0..<iterations {
                guard let cb = queue.makeCommandBuffer() else { continue }
                encoder.encode(
                    commandBuffer: cb,
                    worldGaussians: worldBuffer,
                    harmonics: harmonicsBuffer,
                    camera: camera,
                    gaussianCount: gaussianCount,
                    tilesX: tilesX,
                    tilesY: tilesY,
                    tileWidth: tileWidth,
                    tileHeight: tileHeight,
                    surfaceWidth: imageWidth,
                    surfaceHeight: imageHeight,
                    compactedGaussians: compactedBuffer,
                    compactedHeader: headerBuffer,
                    tileCounts: tileCountsBuffer,
                    tileOffsets: tileOffsetsBuffer,
                    partialSums: partialSumsBuffer,
                    sortKeys: sortKeysBuffer,
                    sortIndices: sortIndicesBuffer,
                    maxCompacted: maxCompacted,
                    maxAssignments: maxAssignments,
                    useHalfWorld: false,
                    skipSort: false,
                    tempSortKeys: tempSortKeys,
                    tempSortIndices: tempSortIndices
                )
                encoder.encodeRender(
                    commandBuffer: cb,
                    compactedGaussians: compactedBuffer,
                    tileOffsets: tileOffsetsBuffer,
                    tileCounts: tileCountsBuffer,
                    sortedIndices: sortIndicesBuffer,
                    colorTexture: colorTexture,
                    depthTexture: depthTexture,
                    tilesX: tilesX,
                    tilesY: tilesY,
                    width: imageWidth,
                    height: imageHeight,
                    tileWidth: tileWidth,
                    tileHeight: tileHeight,
                    whiteBackground: false
                )
                cb.commit()
                cb.waitUntilCompleted()
                let gpuTime = cb.gpuEndTime - cb.gpuStartTime
                if gpuTime > 0 { times.append(gpuTime * 1000) }
            }

            guard !times.isEmpty else { continue }

            let visibleCount = LocalSortPipelineEncoder.readVisibleCount(from: headerBuffer)
            let avg = times.reduce(0, +) / Double(times.count)
            let fps = 1000.0 / avg

            summaries.append("\(gaussianCount/1000)k: \(String(format: "%.2f", avg))ms (\(String(format: "%.1f", fps)) FPS, visible: \(visibleCount))")
        }

        print("\n╔═════════════════════════════════════════════╗")
        print("║  TELLUSIM PIPELINE - FULL END-TO-END        ║")
        print("║  (Project + Scan + Scatter + Sort + Render) ║")
        print("╠═════════════════════════════════════════════╣")
        for summary in summaries {
            print("║  \(summary)")
        }
        print("╠═════════════════════════════════════════════╣")
        print("║  Target: 8.33ms (120 FPS)                   ║")
        print("╚═════════════════════════════════════════════╝\n")

        XCTAssertFalse(summaries.isEmpty, "Should have timing results")
    }

    /// Test pipeline stages separately to identify bottleneck
    func testPipelineStageBreakdown() throws {
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let encoder = try LocalSortPipelineEncoder(device: device, library: library)

        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight
        let tileCount = tilesX * tilesY

        let gaussianCount = 500_000
        let maxCompacted = gaussianCount
        let maxAssignments = gaussianCount * 32  // Need more for uniform test data
        let iterations = 5

        // Create all buffers
        let worldBuffer = device.makeBuffer(
            length: gaussianCount * MemoryLayout<PackedWorldGaussian>.stride,
            options: .storageModeShared
        )!
        let harmonicsBuffer = device.makeBuffer(
            length: gaussianCount * 3 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let compactedBuffer = device.makeBuffer(
            length: maxCompacted * MemoryLayout<CompactedGaussianSwift>.stride,
            options: .storageModePrivate
        )!
        let headerBuffer = device.makeBuffer(
            length: MemoryLayout<CompactedHeaderSwift>.stride,
            options: .storageModeShared
        )!
        let tileCountsBuffer = device.makeBuffer(
            length: tileCount * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )!
        let tileOffsetsBuffer = device.makeBuffer(
            length: (tileCount + 1) * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )!
        let partialSumsBuffer = device.makeBuffer(
            length: 1024 * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!
        let sortKeysBuffer = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )!
        let sortIndicesBuffer = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )!
        let tempSortKeys = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!
        let tempSortIndices = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!

        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: imageWidth,
            height: imageHeight,
            mipmapped: false
        )
        colorDesc.usage = [.shaderRead, .shaderWrite]
        colorDesc.storageMode = .private
        let colorTexture = device.makeTexture(descriptor: colorDesc)!

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: imageWidth,
            height: imageHeight,
            mipmapped: false
        )
        depthDesc.usage = [.shaderRead, .shaderWrite]
        depthDesc.storageMode = .private
        let depthTexture = device.makeTexture(descriptor: depthDesc)!

        populateTestData(worldBuffer: worldBuffer, harmonicsBuffer: harmonicsBuffer, count: gaussianCount)
        let camera = createTestCamera()

        // Warm up
        for _ in 0..<2 {
            if let cb = queue.makeCommandBuffer() {
                encoder.encode(
                    commandBuffer: cb,
                    worldGaussians: worldBuffer,
                    harmonics: harmonicsBuffer,
                    camera: camera,
                    gaussianCount: gaussianCount,
                    tilesX: tilesX,
                    tilesY: tilesY,
                    tileWidth: tileWidth,
                    tileHeight: tileHeight,
                    surfaceWidth: imageWidth,
                    surfaceHeight: imageHeight,
                    compactedGaussians: compactedBuffer,
                    compactedHeader: headerBuffer,
                    tileCounts: tileCountsBuffer,
                    tileOffsets: tileOffsetsBuffer,
                    partialSums: partialSumsBuffer,
                    sortKeys: sortKeysBuffer,
                    sortIndices: sortIndicesBuffer,
                    maxCompacted: maxCompacted,
                    maxAssignments: maxAssignments,
                    useHalfWorld: false,
                    skipSort: false,
                    tempSortKeys: tempSortKeys,
                    tempSortIndices: tempSortIndices
                )
                cb.commit()
                cb.waitUntilCompleted()
            }
        }

        // Test 1: Pipeline WITHOUT sort (skipSort=true)
        var noSortTimes: [Double] = []
        for _ in 0..<iterations {
            guard let cb = queue.makeCommandBuffer() else { continue }
            encoder.encode(
                commandBuffer: cb,
                worldGaussians: worldBuffer,
                harmonics: harmonicsBuffer,
                camera: camera,
                gaussianCount: gaussianCount,
                tilesX: tilesX,
                tilesY: tilesY,
                tileWidth: tileWidth,
                tileHeight: tileHeight,
                surfaceWidth: imageWidth,
                surfaceHeight: imageHeight,
                compactedGaussians: compactedBuffer,
                compactedHeader: headerBuffer,
                tileCounts: tileCountsBuffer,
                tileOffsets: tileOffsetsBuffer,
                partialSums: partialSumsBuffer,
                sortKeys: sortKeysBuffer,
                sortIndices: sortIndicesBuffer,
                maxCompacted: maxCompacted,
                maxAssignments: maxAssignments,
                useHalfWorld: false,
                skipSort: true  // No sort
            )
            cb.commit()
            cb.waitUntilCompleted()
            let gpuTime = cb.gpuEndTime - cb.gpuStartTime
            if gpuTime > 0 { noSortTimes.append(gpuTime * 1000) }
        }

        // Test 2: Pipeline WITH sort (skipSort=false)
        var withSortTimes: [Double] = []
        for _ in 0..<iterations {
            guard let cb = queue.makeCommandBuffer() else { continue }
            encoder.encode(
                commandBuffer: cb,
                worldGaussians: worldBuffer,
                harmonics: harmonicsBuffer,
                camera: camera,
                gaussianCount: gaussianCount,
                tilesX: tilesX,
                tilesY: tilesY,
                tileWidth: tileWidth,
                tileHeight: tileHeight,
                surfaceWidth: imageWidth,
                surfaceHeight: imageHeight,
                compactedGaussians: compactedBuffer,
                compactedHeader: headerBuffer,
                tileCounts: tileCountsBuffer,
                tileOffsets: tileOffsetsBuffer,
                partialSums: partialSumsBuffer,
                sortKeys: sortKeysBuffer,
                sortIndices: sortIndicesBuffer,
                maxCompacted: maxCompacted,
                maxAssignments: maxAssignments,
                useHalfWorld: false,
                skipSort: false,
                tempSortKeys: tempSortKeys,
                tempSortIndices: tempSortIndices
            )
            cb.commit()
            cb.waitUntilCompleted()
            let gpuTime = cb.gpuEndTime - cb.gpuStartTime
            if gpuTime > 0 { withSortTimes.append(gpuTime * 1000) }
        }

        // Test 3: Render only
        var renderTimes: [Double] = []
        for _ in 0..<iterations {
            guard let cb = queue.makeCommandBuffer() else { continue }
            encoder.encodeRender(
                commandBuffer: cb,
                compactedGaussians: compactedBuffer,
                tileOffsets: tileOffsetsBuffer,
                tileCounts: tileCountsBuffer,
                sortedIndices: sortIndicesBuffer,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                tilesX: tilesX,
                tilesY: tilesY,
                width: imageWidth,
                height: imageHeight,
                tileWidth: tileWidth,
                tileHeight: tileHeight,
                whiteBackground: false
            )
            cb.commit()
            cb.waitUntilCompleted()
            let gpuTime = cb.gpuEndTime - cb.gpuStartTime
            if gpuTime > 0 { renderTimes.append(gpuTime * 1000) }
        }

        let noSort = noSortTimes.reduce(0, +) / Double(noSortTimes.count)
        let withSort = withSortTimes.reduce(0, +) / Double(withSortTimes.count)
        let render = renderTimes.reduce(0, +) / Double(renderTimes.count)
        let sortOnly = withSort - noSort

        // Check tile distribution
        let tileCounts = tileCountsBuffer.contents().bindMemory(to: UInt32.self, capacity: tileCount)
        var maxTileCount: UInt32 = 0
        var totalAssignments: UInt64 = 0
        var nonEmptyTiles = 0
        for i in 0..<tileCount {
            let c = tileCounts[i]
            totalAssignments += UInt64(c)
            if c > maxTileCount { maxTileCount = c }
            if c > 0 { nonEmptyTiles += 1 }
        }
        let avgPerTile = Double(totalAssignments) / Double(nonEmptyTiles)

        let visibleCount = LocalSortPipelineEncoder.readVisibleCount(from: headerBuffer)

        print("\n╔═══════════════════════════════════════════════╗")
        print("║  PIPELINE STAGE BREAKDOWN (500k gaussians)    ║")
        print("╠═══════════════════════════════════════════════╣")
        print("║  Visible: \(visibleCount) gaussians")
        print("║  Total assignments: \(totalAssignments)")
        print("║  Max gaussians/tile: \(maxTileCount)")
        print("║  Avg gaussians/tile: \(String(format: "%.1f", avgPerTile))")
        print("║  Non-empty tiles: \(nonEmptyTiles) / \(tileCount)")
        print("╠═══════════════════════════════════════════════╣")
        print("║  Project+Scan+Scatter (no sort): \(String(format: "%.2f", noSort))ms")
        print("║  Project+Scan+Scatter+Sort: \(String(format: "%.2f", withSort))ms")
        print("║  Per-tile sort only: \(String(format: "%.2f", sortOnly))ms")
        print("║  Render only: \(String(format: "%.2f", render))ms")
        print("╠═══════════════════════════════════════════════╣")
        print("║  Bottleneck: \(sortOnly > render ? "PER-TILE SORT" : "RENDER")")
        print("╚═══════════════════════════════════════════════╝\n")
    }

    /// Stage breakdown at 2M gaussians to identify bottlenecks
    func testStageBreakdown2M() throws {
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let encoder = try LocalSortPipelineEncoder(device: device, library: library)

        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight
        let tileCount = tilesX * tilesY

        let gaussianCount = 2_000_000
        let maxCompacted = gaussianCount
        let maxAssignments = gaussianCount * 8
        let iterations = 5

        // Create buffers
        let worldBuffer = device.makeBuffer(
            length: gaussianCount * MemoryLayout<PackedWorldGaussian>.stride,
            options: .storageModeShared
        )!
        let harmonicsBuffer = device.makeBuffer(
            length: gaussianCount * 3 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let compactedBuffer = device.makeBuffer(
            length: maxCompacted * MemoryLayout<CompactedGaussianSwift>.stride,
            options: .storageModePrivate
        )!
        let headerBuffer = device.makeBuffer(
            length: MemoryLayout<CompactedHeaderSwift>.stride,
            options: .storageModeShared
        )!
        let tileCountsBuffer = device.makeBuffer(
            length: tileCount * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )!
        let tileOffsetsBuffer = device.makeBuffer(
            length: (tileCount + 1) * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!
        let partialSumsBuffer = device.makeBuffer(
            length: 1024 * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!
        let sortKeysBuffer = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!
        let sortIndicesBuffer = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!
        let tempSortKeys = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!
        let tempSortIndices = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!

        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: imageWidth, height: imageHeight, mipmapped: false)
        colorDesc.usage = [.shaderRead, .shaderWrite]
        colorDesc.storageMode = .private
        let colorTexture = device.makeTexture(descriptor: colorDesc)!

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float, width: imageWidth, height: imageHeight, mipmapped: false)
        depthDesc.usage = [.shaderRead, .shaderWrite]
        depthDesc.storageMode = .private
        let depthTexture = device.makeTexture(descriptor: depthDesc)!

        // Create realistic clustered data
        let worldPtr = worldBuffer.contents().bindMemory(to: PackedWorldGaussian.self, capacity: gaussianCount)
        let harmonicsPtr = harmonicsBuffer.contents().bindMemory(to: Float.self, capacity: gaussianCount * 3)

        for i in 0..<gaussianCount {
            let theta = Float.random(in: 0..<Float.pi * 2)
            let phi = Float.random(in: 0..<Float.pi)
            let r = Float.random(in: 1..<5)
            let x = r * sin(phi) * cos(theta)
            let y = r * sin(phi) * sin(theta)
            let z = 5.0 + r * cos(phi)  // Positive Z for identity view matrix

            worldPtr[i] = PackedWorldGaussian(
                position: SIMD3<Float>(x, y, z),
                scale: SIMD3<Float>(0.01, 0.01, 0.01),
                rotation: SIMD4<Float>(0, 0, 0, 1),
                opacity: Float.random(in: 0.3..<1.0)
            )
            harmonicsPtr[i * 3 + 0] = Float.random(in: 0..<1)
            harmonicsPtr[i * 3 + 1] = Float.random(in: 0..<1)
            harmonicsPtr[i * 3 + 2] = Float.random(in: 0..<1)
        }

        let camera = createTestCamera()

        // Warm up
        for _ in 0..<2 {
            if let cb = queue.makeCommandBuffer() {
                encoder.encode(
                    commandBuffer: cb,
                    worldGaussians: worldBuffer,
                    harmonics: harmonicsBuffer,
                    camera: camera,
                    gaussianCount: gaussianCount,
                    tilesX: tilesX, tilesY: tilesY,
                    tileWidth: tileWidth, tileHeight: tileHeight,
                    surfaceWidth: imageWidth, surfaceHeight: imageHeight,
                    compactedGaussians: compactedBuffer,
                    compactedHeader: headerBuffer,
                    tileCounts: tileCountsBuffer,
                    tileOffsets: tileOffsetsBuffer,
                    partialSums: partialSumsBuffer,
                    sortKeys: sortKeysBuffer,
                    sortIndices: sortIndicesBuffer,
                    maxCompacted: maxCompacted,
                    maxAssignments: maxAssignments,
                    skipSort: false,
                    tempSortKeys: tempSortKeys,
                    tempSortIndices: tempSortIndices
                )
                cb.commit()
                cb.waitUntilCompleted()
            }
        }

        // Test 1: Pipeline WITHOUT sort
        var noSortTimes: [Double] = []
        for _ in 0..<iterations {
            guard let cb = queue.makeCommandBuffer() else { continue }
            encoder.encode(
                commandBuffer: cb,
                worldGaussians: worldBuffer,
                harmonics: harmonicsBuffer,
                camera: camera,
                gaussianCount: gaussianCount,
                tilesX: tilesX, tilesY: tilesY,
                tileWidth: tileWidth, tileHeight: tileHeight,
                surfaceWidth: imageWidth, surfaceHeight: imageHeight,
                compactedGaussians: compactedBuffer,
                compactedHeader: headerBuffer,
                tileCounts: tileCountsBuffer,
                tileOffsets: tileOffsetsBuffer,
                partialSums: partialSumsBuffer,
                sortKeys: sortKeysBuffer,
                sortIndices: sortIndicesBuffer,
                maxCompacted: maxCompacted,
                maxAssignments: maxAssignments,
                skipSort: true
            )
            cb.commit()
            cb.waitUntilCompleted()
            let gpuTime = cb.gpuEndTime - cb.gpuStartTime
            if gpuTime > 0 { noSortTimes.append(gpuTime * 1000) }
        }

        // Test 2: Pipeline WITH sort
        var withSortTimes: [Double] = []
        for _ in 0..<iterations {
            guard let cb = queue.makeCommandBuffer() else { continue }
            encoder.encode(
                commandBuffer: cb,
                worldGaussians: worldBuffer,
                harmonics: harmonicsBuffer,
                camera: camera,
                gaussianCount: gaussianCount,
                tilesX: tilesX, tilesY: tilesY,
                tileWidth: tileWidth, tileHeight: tileHeight,
                surfaceWidth: imageWidth, surfaceHeight: imageHeight,
                compactedGaussians: compactedBuffer,
                compactedHeader: headerBuffer,
                tileCounts: tileCountsBuffer,
                tileOffsets: tileOffsetsBuffer,
                partialSums: partialSumsBuffer,
                sortKeys: sortKeysBuffer,
                sortIndices: sortIndicesBuffer,
                maxCompacted: maxCompacted,
                maxAssignments: maxAssignments,
                skipSort: false,
                tempSortKeys: tempSortKeys,
                tempSortIndices: tempSortIndices
            )
            cb.commit()
            cb.waitUntilCompleted()
            let gpuTime = cb.gpuEndTime - cb.gpuStartTime
            if gpuTime > 0 { withSortTimes.append(gpuTime * 1000) }
        }

        // Test 3: Render only
        var renderTimes: [Double] = []
        for _ in 0..<iterations {
            guard let cb = queue.makeCommandBuffer() else { continue }
            encoder.encodeRender(
                commandBuffer: cb,
                compactedGaussians: compactedBuffer,
                tileOffsets: tileOffsetsBuffer,
                tileCounts: tileCountsBuffer,
                sortedIndices: sortIndicesBuffer,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                tilesX: tilesX, tilesY: tilesY,
                width: imageWidth, height: imageHeight,
                tileWidth: tileWidth, tileHeight: tileHeight,
                whiteBackground: false
            )
            cb.commit()
            cb.waitUntilCompleted()
            let gpuTime = cb.gpuEndTime - cb.gpuStartTime
            if gpuTime > 0 { renderTimes.append(gpuTime * 1000) }
        }

        let noSort = noSortTimes.reduce(0, +) / Double(noSortTimes.count)
        let withSort = withSortTimes.reduce(0, +) / Double(withSortTimes.count)
        let render = renderTimes.reduce(0, +) / Double(renderTimes.count)
        let sortOnly = withSort - noSort

        // Check tile distribution
        let tileCounts = tileCountsBuffer.contents().bindMemory(to: UInt32.self, capacity: tileCount)
        var maxTileCount: UInt32 = 0
        var totalAssignments: UInt64 = 0
        var nonEmptyTiles = 0
        for i in 0..<tileCount {
            let c = tileCounts[i]
            totalAssignments += UInt64(c)
            if c > maxTileCount { maxTileCount = c }
            if c > 0 { nonEmptyTiles += 1 }
        }
        let avgPerTile = Double(totalAssignments) / Double(max(1, nonEmptyTiles))

        let visibleCount = LocalSortPipelineEncoder.readVisibleCount(from: headerBuffer)

        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║  PIPELINE STAGE BREAKDOWN (2M gaussians)                   ║")
        print("╠═══════════════════════════════════════════════════════════╣")
        print("║  Visible: \(visibleCount) / \(gaussianCount)")
        print("║  Total assignments: \(totalAssignments)")
        print("║  Max gaussians/tile: \(maxTileCount)")
        print("║  Avg gaussians/tile: \(String(format: "%.1f", avgPerTile))")
        print("║  Non-empty tiles: \(nonEmptyTiles) / \(tileCount)")
        print("╠═══════════════════════════════════════════════════════════╣")
        print("║  Project+Scan+Scatter (no sort): \(String(format: "%.2f", noSort))ms")
        print("║  Project+Scan+Scatter+Sort: \(String(format: "%.2f", withSort))ms")
        print("║  Per-tile sort only: \(String(format: "%.2f", sortOnly))ms")
        print("║  Render only: \(String(format: "%.2f", render))ms")
        print("╠═══════════════════════════════════════════════════════════╣")
        print("║  Total estimate: \(String(format: "%.2f", withSort + render))ms")
        print("║  Target 120 FPS gap: \(String(format: "%.2f", (withSort + render) - 8.33))ms")
        print("╚═══════════════════════════════════════════════════════════╝\n")
    }

    /// Performance test at 2M gaussians (target: 60 FPS / 16.67ms, stretch: 120 FPS / 8.33ms)
    func testPerformance2M() throws {
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let encoder = try LocalSortPipelineEncoder(device: device, library: library)

        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight
        let tileCount = tilesX * tilesY

        let gaussianCount = 2_000_000
        let maxCompacted = gaussianCount
        let maxAssignments = gaussianCount * 8  // 8x is typical for real scenes
        let iterations = 10

        print("\n=== Allocating buffers for 2M gaussians ===")

        // Create buffers
        let worldBuffer = device.makeBuffer(
            length: gaussianCount * MemoryLayout<PackedWorldGaussian>.stride,
            options: .storageModeShared
        )!
        let harmonicsBuffer = device.makeBuffer(
            length: gaussianCount * 3 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let compactedBuffer = device.makeBuffer(
            length: maxCompacted * MemoryLayout<CompactedGaussianSwift>.stride,
            options: .storageModePrivate
        )!
        let headerBuffer = device.makeBuffer(
            length: MemoryLayout<CompactedHeaderSwift>.stride,
            options: .storageModeShared
        )!
        let tileCountsBuffer = device.makeBuffer(
            length: tileCount * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!
        let tileOffsetsBuffer = device.makeBuffer(
            length: (tileCount + 1) * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!
        let partialSumsBuffer = device.makeBuffer(
            length: 1024 * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!
        let sortKeysBuffer = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!
        let sortIndicesBuffer = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!
        let tempSortKeys = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!
        let tempSortIndices = device.makeBuffer(
            length: maxAssignments * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!

        // Output textures
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: imageWidth, height: imageHeight, mipmapped: false)
        colorDesc.usage = [.shaderRead, .shaderWrite]
        colorDesc.storageMode = .private
        let colorTexture = device.makeTexture(descriptor: colorDesc)!

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float, width: imageWidth, height: imageHeight, mipmapped: false)
        depthDesc.usage = [.shaderRead, .shaderWrite]
        depthDesc.storageMode = .private
        let depthTexture = device.makeTexture(descriptor: depthDesc)!

        print("Populating \(gaussianCount) gaussians...")

        // Create realistic test data - clustered distribution (like real scenes)
        let worldPtr = worldBuffer.contents().bindMemory(to: PackedWorldGaussian.self, capacity: gaussianCount)
        let harmonicsPtr = harmonicsBuffer.contents().bindMemory(to: Float.self, capacity: gaussianCount * 3)

        for i in 0..<gaussianCount {
            // Cluster positions in a sphere (positive Z for identity view matrix)
            let theta = Float.random(in: 0..<Float.pi * 2)
            let phi = Float.random(in: 0..<Float.pi)
            let r = Float.random(in: 1..<5)

            let x = r * sin(phi) * cos(theta)
            let y = r * sin(phi) * sin(theta)
            let z = 5.0 + r * cos(phi)  // Centered at z=5 (positive for identity view)

            worldPtr[i] = PackedWorldGaussian(
                position: SIMD3<Float>(x, y, z),
                scale: SIMD3<Float>(0.01, 0.01, 0.01),  // Small splats
                rotation: SIMD4<Float>(0, 0, 0, 1),
                opacity: Float.random(in: 0.3..<1.0)
            )

            harmonicsPtr[i * 3 + 0] = Float.random(in: 0..<1)
            harmonicsPtr[i * 3 + 1] = Float.random(in: 0..<1)
            harmonicsPtr[i * 3 + 2] = Float.random(in: 0..<1)
        }

        let camera = createTestCamera()

        print("Warming up...")

        // Warm up
        for _ in 0..<3 {
            if let cb = queue.makeCommandBuffer() {
                encoder.encode(
                    commandBuffer: cb,
                    worldGaussians: worldBuffer,
                    harmonics: harmonicsBuffer,
                    camera: camera,
                    gaussianCount: gaussianCount,
                    tilesX: tilesX,
                    tilesY: tilesY,
                    tileWidth: tileWidth,
                    tileHeight: tileHeight,
                    surfaceWidth: imageWidth,
                    surfaceHeight: imageHeight,
                    compactedGaussians: compactedBuffer,
                    compactedHeader: headerBuffer,
                    tileCounts: tileCountsBuffer,
                    tileOffsets: tileOffsetsBuffer,
                    partialSums: partialSumsBuffer,
                    sortKeys: sortKeysBuffer,
                    sortIndices: sortIndicesBuffer,
                    maxCompacted: maxCompacted,
                    maxAssignments: maxAssignments,
                    skipSort: false,
                    tempSortKeys: tempSortKeys,
                    tempSortIndices: tempSortIndices
                )
                encoder.encodeRender(
                    commandBuffer: cb,
                    compactedGaussians: compactedBuffer,
                    tileOffsets: tileOffsetsBuffer,
                    tileCounts: tileCountsBuffer,
                    sortedIndices: sortIndicesBuffer,
                    colorTexture: colorTexture,
                    depthTexture: depthTexture,
                    tilesX: tilesX,
                    tilesY: tilesY,
                    width: imageWidth,
                    height: imageHeight,
                    tileWidth: tileWidth,
                    tileHeight: tileHeight,
                    whiteBackground: false
                )
                cb.commit()
                cb.waitUntilCompleted()
            }
        }

        print("Benchmarking \(iterations) iterations...")

        // Benchmark
        var times: [Double] = []
        for _ in 0..<iterations {
            guard let cb = queue.makeCommandBuffer() else { continue }
            encoder.encode(
                commandBuffer: cb,
                worldGaussians: worldBuffer,
                harmonics: harmonicsBuffer,
                camera: camera,
                gaussianCount: gaussianCount,
                tilesX: tilesX,
                tilesY: tilesY,
                tileWidth: tileWidth,
                tileHeight: tileHeight,
                surfaceWidth: imageWidth,
                surfaceHeight: imageHeight,
                compactedGaussians: compactedBuffer,
                compactedHeader: headerBuffer,
                tileCounts: tileCountsBuffer,
                tileOffsets: tileOffsetsBuffer,
                partialSums: partialSumsBuffer,
                sortKeys: sortKeysBuffer,
                sortIndices: sortIndicesBuffer,
                maxCompacted: maxCompacted,
                maxAssignments: maxAssignments,
                skipSort: false,
                tempSortKeys: tempSortKeys,
                tempSortIndices: tempSortIndices
            )
            encoder.encodeRender(
                commandBuffer: cb,
                compactedGaussians: compactedBuffer,
                tileOffsets: tileOffsetsBuffer,
                tileCounts: tileCountsBuffer,
                sortedIndices: sortIndicesBuffer,
                colorTexture: colorTexture,
                depthTexture: depthTexture,
                tilesX: tilesX,
                tilesY: tilesY,
                width: imageWidth,
                height: imageHeight,
                tileWidth: tileWidth,
                tileHeight: tileHeight,
                whiteBackground: false
            )
            cb.commit()
            cb.waitUntilCompleted()
            let gpuTime = cb.gpuEndTime - cb.gpuStartTime
            if gpuTime > 0 { times.append(gpuTime * 1000) }
        }

        let visibleCount = LocalSortPipelineEncoder.readVisibleCount(from: headerBuffer)
        let overflow = LocalSortPipelineEncoder.readOverflow(from: headerBuffer)

        guard !times.isEmpty else {
            XCTFail("No timing results")
            return
        }

        let avg = times.reduce(0, +) / Double(times.count)
        let minT = times.min()!
        let maxT = times.max()!
        let fps = 1000.0 / avg

        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║  TELLUSIM PIPELINE @ 2M GAUSSIANS                         ║")
        print("╠═══════════════════════════════════════════════════════════╣")
        print("║  Visible: \(visibleCount) / \(gaussianCount) (\(String(format: "%.1f", Double(visibleCount) / Double(gaussianCount) * 100))%)")
        print("║  Overflow: \(overflow)")
        print("╠═══════════════════════════════════════════════════════════╣")
        print("║  Avg: \(String(format: "%.2f", avg))ms")
        print("║  Min: \(String(format: "%.2f", minT))ms")
        print("║  Max: \(String(format: "%.2f", maxT))ms")
        print("║  FPS: \(String(format: "%.1f", fps))")
        print("╠═══════════════════════════════════════════════════════════╣")
        print("║  Target 60 FPS: \(avg <= 16.67 ? "PASS ✓" : "FAIL ✗") (need ≤16.67ms)")
        print("║  Target 120 FPS: \(avg <= 8.33 ? "PASS ✓" : "FAIL ✗") (need ≤8.33ms)")
        print("╚═══════════════════════════════════════════════════════════╝\n")

        // This test documents current performance, doesn't fail
    }

    /// Performance test comparing old pipeline vs Tellusim pipeline
    func testPipelinePerformance() throws {
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let encoder = try LocalSortPipelineEncoder(device: device, library: library)

        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight
        let tileCount = tilesX * tilesY

        let sizes = [100_000, 500_000, 1_000_000]
        let iterations = 5

        var summaries: [String] = []

        for gaussianCount in sizes {
            let maxCompacted = gaussianCount
            let maxAssignments = gaussianCount * 16

            // Create buffers
            let worldBuffer = device.makeBuffer(
                length: gaussianCount * MemoryLayout<PackedWorldGaussian>.stride,
                options: .storageModeShared
            )!
            let harmonicsBuffer = device.makeBuffer(
                length: gaussianCount * 3 * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )!
            let compactedBuffer = device.makeBuffer(
                length: maxCompacted * MemoryLayout<CompactedGaussianSwift>.stride,
                options: .storageModePrivate
            )!
            let headerBuffer = device.makeBuffer(
                length: MemoryLayout<CompactedHeaderSwift>.stride,
                options: .storageModeShared
            )!
            let tileCountsBuffer = device.makeBuffer(
                length: tileCount * MemoryLayout<UInt32>.stride,
                options: .storageModeShared
            )!
            let tileOffsetsBuffer = device.makeBuffer(
                length: (tileCount + 1) * MemoryLayout<UInt32>.stride,
                options: .storageModeShared
            )!
            let partialSumsBuffer = device.makeBuffer(
                length: 1024 * MemoryLayout<UInt32>.stride,
                options: .storageModeShared
            )!
            let sortKeysBuffer = device.makeBuffer(
                length: maxAssignments * MemoryLayout<UInt32>.stride,
                options: .storageModePrivate
            )!
            let sortIndicesBuffer = device.makeBuffer(
                length: maxAssignments * MemoryLayout<UInt32>.stride,
                options: .storageModePrivate
            )!

            populateTestData(
                worldBuffer: worldBuffer,
                harmonicsBuffer: harmonicsBuffer,
                count: gaussianCount
            )

            let camera = createTestCamera()

            // Warm up
            for _ in 0..<2 {
                if let cb = queue.makeCommandBuffer() {
                    encoder.encode(
                        commandBuffer: cb,
                        worldGaussians: worldBuffer,
                        harmonics: harmonicsBuffer,
                        camera: camera,
                        gaussianCount: gaussianCount,
                        tilesX: tilesX,
                        tilesY: tilesY,
                        tileWidth: tileWidth,
                        tileHeight: tileHeight,
                        surfaceWidth: imageWidth,
                        surfaceHeight: imageHeight,
                        compactedGaussians: compactedBuffer,
                        compactedHeader: headerBuffer,
                        tileCounts: tileCountsBuffer,
                        tileOffsets: tileOffsetsBuffer,
                        partialSums: partialSumsBuffer,
                        sortKeys: sortKeysBuffer,
                        sortIndices: sortIndicesBuffer,
                        maxCompacted: maxCompacted,
                        maxAssignments: maxAssignments
                    )
                    cb.commit()
                    cb.waitUntilCompleted()
                }
            }

            // Benchmark
            var times: [Double] = []
            for _ in 0..<iterations {
                guard let cb = queue.makeCommandBuffer() else { continue }
                encoder.encode(
                    commandBuffer: cb,
                    worldGaussians: worldBuffer,
                    harmonics: harmonicsBuffer,
                    camera: camera,
                    gaussianCount: gaussianCount,
                    tilesX: tilesX,
                    tilesY: tilesY,
                    tileWidth: tileWidth,
                    tileHeight: tileHeight,
                    surfaceWidth: imageWidth,
                    surfaceHeight: imageHeight,
                    compactedGaussians: compactedBuffer,
                    compactedHeader: headerBuffer,
                    tileCounts: tileCountsBuffer,
                    tileOffsets: tileOffsetsBuffer,
                    partialSums: partialSumsBuffer,
                    sortKeys: sortKeysBuffer,
                    sortIndices: sortIndicesBuffer,
                    maxCompacted: maxCompacted,
                    maxAssignments: maxAssignments
                )
                cb.commit()
                cb.waitUntilCompleted()
                let gpuTime = cb.gpuEndTime - cb.gpuStartTime
                if gpuTime > 0 { times.append(gpuTime * 1000) }
            }

            guard !times.isEmpty else { continue }

            let visibleCount = LocalSortPipelineEncoder.readVisibleCount(from: headerBuffer)
            let avg = times.reduce(0, +) / Double(times.count)

            summaries.append("\(gaussianCount/1000)k: \(String(format: "%.2f", avg))ms (visible: \(visibleCount))")
        }

        print("\n[Tellusim Pipeline Performance]")
        for summary in summaries {
            print(summary)
        }
        print("")

        XCTAssertFalse(summaries.isEmpty, "Should have timing results")
    }

    // MARK: - Helpers

    private func populateTestData(
        worldBuffer: MTLBuffer,
        harmonicsBuffer: MTLBuffer,
        count: Int
    ) {
        let worldPtr = worldBuffer.contents().bindMemory(to: PackedWorldGaussian.self, capacity: count)
        let harmonicsPtr = harmonicsBuffer.contents().bindMemory(to: Float.self, capacity: count * 3)

        for i in 0..<count {
            // Position: scattered in front of camera (z = 2 to 10, positive Z for identity view matrix)
            let x = Float.random(in: -5..<5)
            let y = Float.random(in: -3..<3)
            let z = Float.random(in: 2..<10)

            // Scale: small splats
            let scale = Float.random(in: 0.01..<0.1)

            // Random rotation quaternion
            let angle = Float.random(in: 0..<Float.pi * 2)
            let axis = simd_normalize(SIMD3<Float>(
                Float.random(in: -1..<1),
                Float.random(in: -1..<1),
                Float.random(in: -1..<1)
            ))
            let quat = SIMD4<Float>(
                axis.x * sin(angle/2),
                axis.y * sin(angle/2),
                axis.z * sin(angle/2),
                cos(angle/2)
            )

            worldPtr[i] = PackedWorldGaussian(
                position: SIMD3<Float>(x, y, z),
                scale: SIMD3<Float>(scale, scale, scale),
                rotation: quat,
                opacity: Float.random(in: 0.5..<1.0)
            )

            // Simple RGB color
            harmonicsPtr[i * 3 + 0] = Float.random(in: 0..<1)
            harmonicsPtr[i * 3 + 1] = Float.random(in: 0..<1)
            harmonicsPtr[i * 3 + 2] = Float.random(in: 0..<1)
        }
    }


    private func createTestCamera() -> CameraUniformsSwift {
        // Simple perspective camera looking down -Z
        let aspect = Float(imageWidth) / Float(imageHeight)
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

        // Identity view matrix (camera at origin looking down -Z)
        let viewMatrix = simd_float4x4(1.0)

        let focalX = f * Float(imageWidth) / 2.0
        let focalY = f * Float(imageHeight) / 2.0

        return CameraUniformsSwift(
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
            shComponents: 0,  // Use simple RGB, not SH
            gaussianCount: 0  // Will be set per-frame
        )
    }
}

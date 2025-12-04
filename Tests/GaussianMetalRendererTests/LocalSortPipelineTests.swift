import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Tests for the Local-style pipeline (fused project + compact + count + scatter)
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
        print("║  LOCAL SORT PIPELINE - FULL END-TO-END        ║")
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
        print("║  LOCAL SORT PIPELINE @ 2M GAUSSIANS                         ║")
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

    /// Performance test comparing old pipeline vs Local pipeline
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

        print("\n[Local Pipeline Performance]")
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

    // MARK: - 16-bit Sort Experimental Tests

    /// Test 16-bit sort pipeline availability
    func test16BitSortAvailable() throws {
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
        let device = renderer.device
        let library = renderer.library

        let encoder = try LocalSortPipelineEncoder(device: device, library: library)

        if encoder.has16BitSort {
            print("[16-bit Sort] Available ✓")
        } else {
            print("[16-bit Sort] Not available - kernels not found")
        }

        // This test just checks availability, doesn't fail
    }

    /// Test that 16-bit sort doesn't crash and produces valid output
    func test16BitSortCorrectness() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!

        // Create renderer with 16-bit sort enabled via config
        let config = RendererConfig(sortMode: .sort16Bit)
        let renderer = try LocalSortRenderer(device: device, config: config)
        renderer.useSharedBuffers = true  // Enable shared for debugging

        let gaussianCount = 10_000
        let width = 512
        let height = 384

        // Create shared input buffers
        let worldBuffer = device.makeBuffer(
            length: gaussianCount * MemoryLayout<PackedWorldGaussian>.stride,
            options: .storageModeShared
        )!
        let harmonicsBuffer = device.makeBuffer(
            length: gaussianCount * 3 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        // Populate test data with deterministic values
        let worldPtr = worldBuffer.contents().bindMemory(to: PackedWorldGaussian.self, capacity: gaussianCount)
        let harmonicsPtr = harmonicsBuffer.contents().bindMemory(to: Float.self, capacity: gaussianCount * 3)

        srand48(42)  // Deterministic seed
        for i in 0..<gaussianCount {
            let x = Float(drand48()) * 10 - 5
            let y = Float(drand48()) * 6 - 3
            let z = Float(drand48()) * 8 + 2
            let scale = Float(drand48()) * 0.09 + 0.01

            worldPtr[i] = PackedWorldGaussian(
                position: SIMD3<Float>(x, y, z),
                scale: SIMD3<Float>(scale, scale, scale),
                rotation: SIMD4<Float>(0, 0, 0, 1),
                opacity: Float(drand48()) * 0.5 + 0.5
            )

            harmonicsPtr[i * 3 + 0] = Float(drand48())
            harmonicsPtr[i * 3 + 1] = Float(drand48())
            harmonicsPtr[i * 3 + 2] = Float(drand48())
        }

        let camera = CameraParams(
            viewMatrix: simd_float4x4(1.0),
            projectionMatrix: simd_float4x4(columns: (
                SIMD4<Float>(2.0 / Float(width), 0, 0, 0),
                SIMD4<Float>(0, 2.0 / Float(height), 0, 0),
                SIMD4<Float>(0, 0, -0.02, 0),
                SIMD4<Float>(0, 0, -1.0, 1)
            )),
            position: SIMD3<Float>(0, 0, 0),
            focalX: Float(width) / 2.0,
            focalY: Float(height) / 2.0
        )

        let input = GaussianInput(
            gaussians: worldBuffer,
            harmonics: harmonicsBuffer,
            gaussianCount: gaussianCount,
            shComponents: 0
        )

        // Render with 16-bit sort
        guard let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        let result = renderer.render(
            toTexture: cb,
            input: input,
            camera: camera,
            width: width,
            height: height,
            whiteBackground: false,
            mortonSorted: false
        )

        cb.commit()
        cb.waitUntilCompleted()

        // Check that render succeeded
        XCTAssertNotNil(result, "16-bit sort render should succeed")

        // Check command buffer status
        XCTAssertEqual(cb.status, .completed, "Command buffer should complete without error")

        let visibleCount = renderer.getVisibleCount()
        let hadOverflow = renderer.hadOverflow()

        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║  16-BIT SORT CORRECTNESS TEST                                  ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Gaussians: \(gaussianCount)")
        print("║  Resolution: \(width)x\(height)")
        print("║  Visible count: \(visibleCount)")
        print("║  Overflow: \(hadOverflow)")
        print("║  Result: \(result != nil ? "SUCCESS" : "FAILED")")
        print("╚═══════════════════════════════════════════════════════════════╝\n")

        XCTAssertGreaterThan(visibleCount, 0, "Should have visible gaussians")
        XCTAssertFalse(hadOverflow, "Should not overflow")
    }

    /// Benchmark 16-bit vs 32-bit full pipeline using LocalSortRenderer
    func testSort16vs32Benchmark() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!

        // Create two renderers - one 32-bit, one 16-bit via config
        let config32 = RendererConfig(sortMode: .sort32Bit)
        let config16 = RendererConfig(sortMode: .sort16Bit)
        let renderer32 = try LocalSortRenderer(device: device, config: config32)
        let renderer16 = try LocalSortRenderer(device: device, config: config16)

        let gaussianCount = 500_000
        let width = 1920
        let height = 1080
        let iterations = 10

        // Create shared input buffers
        let worldBuffer = device.makeBuffer(
            length: gaussianCount * MemoryLayout<PackedWorldGaussian>.stride,
            options: .storageModeShared
        )!
        let harmonicsBuffer = device.makeBuffer(
            length: gaussianCount * 3 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        // Populate test data
        populateTestData(worldBuffer: worldBuffer, harmonicsBuffer: harmonicsBuffer, count: gaussianCount)
        let camera = createTestCamera()

        let input = GaussianInput(
            gaussians: worldBuffer,
            harmonics: harmonicsBuffer,
            gaussianCount: gaussianCount,
            shComponents: 0
        )

        let cameraParams = CameraParams(
            viewMatrix: camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix,
            position: camera.cameraCenter,
            focalX: camera.focalX,
            focalY: camera.focalY
        )

        // Warm up both renderers
        for _ in 0..<3 {
            if let cb = queue.makeCommandBuffer() {
                _ = renderer32.render(toTexture: cb, input: input, camera: cameraParams,
                                      width: width, height: height, whiteBackground: false, mortonSorted: false)
                cb.commit()
                cb.waitUntilCompleted()
            }
            if let cb = queue.makeCommandBuffer() {
                _ = renderer16.render(toTexture: cb, input: input, camera: cameraParams,
                                      width: width, height: height, whiteBackground: false, mortonSorted: false)
                cb.commit()
                cb.waitUntilCompleted()
            }
        }

        // Benchmark 32-bit full pipeline
        var times32: [Double] = []
        for _ in 0..<iterations {
            guard let cb = queue.makeCommandBuffer() else { continue }
            _ = renderer32.render(toTexture: cb, input: input, camera: cameraParams,
                                  width: width, height: height, whiteBackground: false, mortonSorted: false)
            cb.commit()
            cb.waitUntilCompleted()
            let gpuTime = cb.gpuEndTime - cb.gpuStartTime
            if gpuTime > 0 { times32.append(gpuTime * 1000) }
        }

        // Benchmark 16-bit full pipeline
        var times16: [Double] = []
        for _ in 0..<iterations {
            guard let cb = queue.makeCommandBuffer() else { continue }
            _ = renderer16.render(toTexture: cb, input: input, camera: cameraParams,
                                  width: width, height: height, whiteBackground: false, mortonSorted: false)
            cb.commit()
            cb.waitUntilCompleted()
            let gpuTime = cb.gpuEndTime - cb.gpuStartTime
            if gpuTime > 0 { times16.append(gpuTime * 1000) }
        }

        // Results
        let avg32 = times32.reduce(0, +) / Double(max(1, times32.count))
        let avg16 = times16.reduce(0, +) / Double(max(1, times16.count))
        let speedup = avg32 > 0 ? ((avg32 - avg16) / avg32 * 100) : 0

        let visibleCount32 = renderer32.getVisibleCount()
        let visibleCount16 = renderer16.getVisibleCount()

        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║  16-BIT vs 32-BIT FULL PIPELINE BENCHMARK                      ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Gaussians: \(gaussianCount)")
        print("║  Resolution: \(width)x\(height)")
        print("║  Visible (32-bit): \(visibleCount32)")
        print("║  Visible (16-bit): \(visibleCount16)")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  32-bit pipeline: \(String(format: "%.2f", avg32))ms")
        print("║  16-bit pipeline: \(String(format: "%.2f", avg16))ms")
        print("║  Speedup: \(String(format: "%.1f", speedup))%")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Capacity & memory:")
        print("║    32-bit: 2K/tile, 16KB TG (2048 × 4B × 2)")
        print("║    16-bit: 4K/tile, 24KB TG (4096 × 4B + 4096 × 2B)")
        print("╚═══════════════════════════════════════════════════════════════╝\n")

        // Verify both produce same visible count
        XCTAssertEqual(visibleCount32, visibleCount16, "Both paths should produce same visible count")
    }

    // MARK: - Scatter Analysis

    /// Analyze scatter distribution to determine if small/large split would help
    func testScatterDistributionAnalysis() throws {
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
        let maxAssignments = gaussianCount * 32

        // Create buffers - compactedBuffer MUST be .storageModeShared to read back
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
            options: .storageModeShared  // Shared to read back tile bounds
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

        populateTestData(worldBuffer: worldBuffer, harmonicsBuffer: harmonicsBuffer, count: gaussianCount)
        let camera = createTestCamera()

        // Run pipeline to populate compacted gaussians
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
            maxAssignments: maxAssignments,
            skipSort: true  // Skip sort, we just want the scatter distribution
        )

        cb.commit()
        cb.waitUntilCompleted()

        let visibleCount = LocalSortPipelineEncoder.readVisibleCount(from: headerBuffer)

        // Read back compacted gaussians and analyze tile distribution
        let compactedPtr = compactedBuffer.contents().bindMemory(to: CompactedGaussian.self, capacity: Int(visibleCount))

        // Tile count buckets
        var bucket1 = 0      // 1 tile
        var bucket2to4 = 0   // 2-4 tiles
        var bucket5to16 = 0  // 5-16 tiles
        var bucket17to64 = 0 // 17-64 tiles
        var bucket65plus = 0 // 65+ tiles

        var totalTiles: UInt64 = 0
        var maxTileCount: Int32 = 0

        for i in 0..<Int(visibleCount) {
            let g = compactedPtr[i]
            let rangeX = g.max_tile.x - g.min_tile.x
            let rangeY = g.max_tile.y - g.min_tile.y
            let tiles = rangeX * rangeY

            totalTiles += UInt64(tiles)
            if tiles > maxTileCount { maxTileCount = tiles }

            switch tiles {
            case 1:
                bucket1 += 1
            case 2...4:
                bucket2to4 += 1
            case 5...16:
                bucket5to16 += 1
            case 17...64:
                bucket17to64 += 1
            default:
                bucket65plus += 1
            }
        }

        let avgTilesPerGaussian = Double(totalTiles) / Double(visibleCount)
        let smallCount = bucket1 + bucket2to4  // Tiles <= 4
        let smallPct = Double(smallCount) / Double(visibleCount) * 100

        // Estimate cost: assume SIMD scatter overhead is 32x for small gaussians
        // Small gaussian: 1 thread could do it, but we use 32 threads
        // Savings: if 80% of gaussians are small, we save ~75% of scatter cost for them
        let simdOverhead = 32.0
        let smallSavingsFactor = (simdOverhead - 1) / simdOverhead  // ~96% savings per small gaussian
        let potentialSpeedup = smallPct * smallSavingsFactor / 100.0

        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║  SCATTER DISTRIBUTION ANALYSIS                                 ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Gaussians: \(gaussianCount) (visible: \(visibleCount))")
        print("║  Resolution: \(imageWidth)x\(imageHeight)")
        print("║  Tile size: \(tileWidth)x\(tileHeight)")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  TILE COUNT DISTRIBUTION:")
        print("║    1 tile:      \(bucket1) (\(String(format: "%.1f", Double(bucket1)/Double(visibleCount)*100))%)")
        print("║    2-4 tiles:   \(bucket2to4) (\(String(format: "%.1f", Double(bucket2to4)/Double(visibleCount)*100))%)")
        print("║    5-16 tiles:  \(bucket5to16) (\(String(format: "%.1f", Double(bucket5to16)/Double(visibleCount)*100))%)")
        print("║    17-64 tiles: \(bucket17to64) (\(String(format: "%.1f", Double(bucket17to64)/Double(visibleCount)*100))%)")
        print("║    65+ tiles:   \(bucket65plus) (\(String(format: "%.1f", Double(bucket65plus)/Double(visibleCount)*100))%)")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  STATISTICS:")
        print("║    Total tile assignments: \(totalTiles)")
        print("║    Avg tiles/gaussian: \(String(format: "%.1f", avgTilesPerGaussian))")
        print("║    Max tiles/gaussian: \(maxTileCount)")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  SMALL/LARGE SPLIT ANALYSIS (threshold: 4 tiles):")
        print("║    Small gaussians (≤4 tiles): \(smallCount) (\(String(format: "%.1f", smallPct))%)")
        print("║    Large gaussians (>4 tiles): \(Int(visibleCount) - smallCount) (\(String(format: "%.1f", 100-smallPct))%)")
        print("║    Potential scatter speedup: ~\(String(format: "%.0f", potentialSpeedup * 100))%")
        print("║    (if small use sequential, large use SIMD)")
        print("╚═══════════════════════════════════════════════════════════════╝\n")

        // Analysis verdict
        if smallPct > 70 {
            print("✓ RECOMMENDATION: Small/large split would likely help significantly")
            print("  Most gaussians (\(String(format: "%.0f", smallPct))%) touch ≤4 tiles")
        } else if smallPct > 40 {
            print("~ RECOMMENDATION: Small/large split might help moderately")
        } else {
            print("✗ RECOMMENDATION: Small/large split unlikely to help much")
            print("  Most gaussians touch many tiles - SIMD scatter is appropriate")
        }
    }

    // MARK: - Determinism Tests

    /// Test compaction determinism - run multiple times and check if compacted order is consistent
    func testCompactionDeterminism() throws {
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let encoder = try LocalSortPipelineEncoder(device: device, library: library)

        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight
        let tileCount = tilesX * tilesY

        let gaussianCount = 10_000  // Smaller for faster testing
        let maxCompacted = gaussianCount
        let maxAssignments = gaussianCount * 32
        let runs = 5

        // Create buffers
        let worldBuffer = device.makeBuffer(
            length: gaussianCount * MemoryLayout<PackedWorldGaussian>.stride,
            options: .storageModeShared
        )!
        let harmonicsBuffer = device.makeBuffer(
            length: gaussianCount * 3 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        // Populate ONCE with deterministic data
        let worldPtr = worldBuffer.contents().bindMemory(to: PackedWorldGaussian.self, capacity: gaussianCount)
        let harmonicsPtr = harmonicsBuffer.contents().bindMemory(to: Float.self, capacity: gaussianCount * 3)
        srand48(12345)  // Fixed seed
        for i in 0..<gaussianCount {
            let x = Float(drand48()) * 10 - 5
            let y = Float(drand48()) * 6 - 3
            let z = Float(drand48()) * 8 + 2
            let scale = Float(drand48()) * 0.09 + 0.01
            worldPtr[i] = PackedWorldGaussian(
                position: SIMD3<Float>(x, y, z),
                scale: SIMD3<Float>(scale, scale, scale),
                rotation: SIMD4<Float>(0, 0, 0, 1),
                opacity: Float(drand48()) * 0.5 + 0.5
            )
            harmonicsPtr[i * 3 + 0] = Float(drand48())
            harmonicsPtr[i * 3 + 1] = Float(drand48())
            harmonicsPtr[i * 3 + 2] = Float(drand48())
        }

        let camera = createTestCamera()

        // Store results from each run
        var compactedResults: [[Float]] = []  // Store depths from compacted buffer

        for run in 0..<runs {
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
                skipSort: true  // Skip sort to isolate compaction
            )

            cb.commit()
            cb.waitUntilCompleted()

            let visibleCount = LocalSortPipelineEncoder.readVisibleCount(from: headerBuffer)

            // Read compacted depths (order matters!)
            let compactedPtr = compactedBuffer.contents().bindMemory(to: CompactedGaussian.self, capacity: Int(visibleCount))
            var depths: [Float] = []
            for i in 0..<min(Int(visibleCount), 1000) {  // First 1000
                depths.append(compactedPtr[i].covariance_depth.w)
            }
            compactedResults.append(depths)

            if run == 0 {
                print("Run \(run): visible=\(visibleCount), first 5 depths: \(depths.prefix(5))")
            }
        }

        // Compare runs
        var compactionDeterministic = true
        for run in 1..<runs {
            if compactedResults[run] != compactedResults[0] {
                compactionDeterministic = false
                // Find first difference
                for i in 0..<min(compactedResults[0].count, compactedResults[run].count) {
                    if compactedResults[0][i] != compactedResults[run][i] {
                        print("COMPACTION DIFFERS at index \(i): run0=\(compactedResults[0][i]) vs run\(run)=\(compactedResults[run][i])")
                        break
                    }
                }
                break
            }
        }

        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║  COMPACTION DETERMINISM TEST                                   ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Runs: \(runs)")
        print("║  Gaussians: \(gaussianCount)")
        print("║  Result: \(compactionDeterministic ? "DETERMINISTIC ✓" : "NON-DETERMINISTIC ✗")")
        print("╚═══════════════════════════════════════════════════════════════╝\n")

        if !compactionDeterministic {
            print("→ COMPACTION is the source of non-determinism!")
            print("  Fix: Use original world index for tie-breaking, not compacted index")
        }
    }

    /// Test scatter determinism (after compaction)
    func testScatterDeterminism() throws {
        // Debug: Check struct size
        print("CompactedGaussian size: \(MemoryLayout<CompactedGaussian>.size)")
        print("CompactedGaussian stride: \(MemoryLayout<CompactedGaussian>.stride)")
        // Expected: 56 bytes after adding originalIdx + pad

        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
        let device = renderer.device
        let library = renderer.library
        let queue = renderer.queue

        let encoder = try LocalSortPipelineEncoder(device: device, library: library)

        let tilesX = (imageWidth + tileWidth - 1) / tileWidth
        let tilesY = (imageHeight + tileHeight - 1) / tileHeight
        let tileCount = tilesX * tilesY

        let gaussianCount = 10_000
        let maxCompacted = gaussianCount
        let maxAssignments = gaussianCount * 32
        let runs = 5

        // Create buffers
        let worldBuffer = device.makeBuffer(
            length: gaussianCount * MemoryLayout<PackedWorldGaussian>.stride,
            options: .storageModeShared
        )!
        let harmonicsBuffer = device.makeBuffer(
            length: gaussianCount * 3 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        // Populate with deterministic data
        let worldPtr = worldBuffer.contents().bindMemory(to: PackedWorldGaussian.self, capacity: gaussianCount)
        let harmonicsPtr = harmonicsBuffer.contents().bindMemory(to: Float.self, capacity: gaussianCount * 3)
        srand48(12345)
        for i in 0..<gaussianCount {
            let x = Float(drand48()) * 10 - 5
            let y = Float(drand48()) * 6 - 3
            let z = Float(drand48()) * 8 + 2
            let scale = Float(drand48()) * 0.09 + 0.01
            worldPtr[i] = PackedWorldGaussian(
                position: SIMD3<Float>(x, y, z),
                scale: SIMD3<Float>(scale, scale, scale),
                rotation: SIMD4<Float>(0, 0, 0, 1),
                opacity: Float(drand48()) * 0.5 + 0.5
            )
            harmonicsPtr[i * 3 + 0] = Float(drand48())
            harmonicsPtr[i * 3 + 1] = Float(drand48())
            harmonicsPtr[i * 3 + 2] = Float(drand48())
        }

        let camera = createTestCamera()

        // Store sort keys from each run (before sorting)
        var sortKeyResults: [[UInt32]] = []

        for run in 0..<runs {
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
                skipSort: true  // Skip sort to see scatter output
            )

            cb.commit()
            cb.waitUntilCompleted()

            // Read sort keys (unsorted, direct from scatter)
            let keysPtr = sortKeysBuffer.contents().bindMemory(to: UInt32.self, capacity: maxAssignments)
            var keys: [UInt32] = []
            for i in 0..<min(5000, maxAssignments) {
                if keysPtr[i] != 0 {
                    keys.append(keysPtr[i])
                }
            }
            sortKeyResults.append(keys)

            if run == 0 {
                print("Run \(run): \(keys.count) non-zero keys, first 5: \(keys.prefix(5))")
            }
        }

        // Compare - for scatter, we expect order to differ but SET should be same
        var scatterOrderDeterministic = true
        var scatterSetDeterministic = true

        let set0 = Set(sortKeyResults[0])
        for run in 1..<runs {
            if sortKeyResults[run] != sortKeyResults[0] {
                scatterOrderDeterministic = false
            }
            let setN = Set(sortKeyResults[run])
            if setN != set0 {
                scatterSetDeterministic = false
            }
        }

        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║  SCATTER DETERMINISM TEST                                      ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print("║  Runs: \(runs)")
        print("║  Order deterministic: \(scatterOrderDeterministic ? "YES ✓" : "NO ✗")")
        print("║  Set deterministic: \(scatterSetDeterministic ? "YES ✓" : "NO ✗")")
        print("╚═══════════════════════════════════════════════════════════════╝\n")

        if !scatterOrderDeterministic && scatterSetDeterministic {
            print("→ SCATTER ORDER is non-deterministic (expected - atomics)")
            print("  The SET of keys is the same, just in different order")
            print("  This is OK if sort produces deterministic output")
        }
        if !scatterSetDeterministic {
            print("→ SCATTER SET differs! This indicates depth key non-determinism")
            print("  Likely cause: compacted index used in depth key tie-breaking")
        }
    }

    // NOTE: Depth-First pipeline tests have been moved to a standalone DepthFirstRenderer.
    // See DepthFirstRenderer.swift for the new two-phase sorting implementation.
}

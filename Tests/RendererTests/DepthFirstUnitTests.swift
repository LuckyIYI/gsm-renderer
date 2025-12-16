import CoreGraphics
import ImageIO
import Metal
@testable import Renderer
import simd
import UniformTypeIdentifiers
import XCTest

/// Unit tests for DepthFirst renderer pipeline stages
final class DepthFirstUnitTests: XCTestCase {
    var device: MTLDevice!
    var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        self.device = MTLCreateSystemDefaultDevice()!
        self.queue = self.device.makeCommandQueue()!
    }

    /// Test with a simple synthetic scene to verify pipeline correctness
    func testDepthFirstPipelineStages() throws {
        // Create a small scene with 100 gaussians
        let gaussianCount = 1000
        let width = 640
        let height = 480

        // Create packed gaussians in a grid pattern
        var packed: [PackedWorldGaussian] = []
        var harmonics: [Float] = []

        for i in 0 ..< gaussianCount {
            let row = i / 32
            let col = i % 32
            let x = Float(col) * 0.1 - 1.6
            let y = Float(row) * 0.1 - 1.6
            let z = 2.0 + Float(i) * 0.001 // In front of OpenCV camera, away from near plane

            packed.append(PackedWorldGaussian(
                position: SIMD3<Float>(x, y, z),
                scale: SIMD3<Float>(0.01, 0.01, 0.01),
                rotation: SIMD4<Float>(0, 0, 0, 1),
                opacity: 0.8
            ))

            // SH DC coefficients (RGB)
            let r = Float(i % 10) / 10.0
            let g = Float((i / 10) % 10) / 10.0
            let b = Float((i / 100) % 10) / 10.0
            harmonics.append(contentsOf: [r, g, b])
        }

        guard let gaussianBuf = device.makeBuffer(bytes: &packed, length: packed.count * MemoryLayout<PackedWorldGaussian>.stride, options: .storageModeShared),
              let harmonicsBuf = device.makeBuffer(bytes: &harmonics, length: harmonics.count * MemoryLayout<Float>.stride, options: .storageModeShared)
        else {
            XCTFail("Failed to create buffers")
            return
        }

        let config = RendererConfig(maxGaussians: gaussianCount, maxWidth: width, maxHeight: height, precision: .float32)
        let renderer = try DepthFirstRenderer(config: config)

        // Create camera looking at the scene
        let viewMatrix = matrix_identity_float4x4
        let projectionMatrix = makeProjectionMatrix(width: width, height: height, near: 0.1, far: 10.0)

        let camera = CameraParams(
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            position: .zero,
            focalX: Float(width) * 1.5,
            focalY: Float(height) * 1.5
        )

        let input = GaussianInput(
            gaussians: gaussianBuf,
            harmonics: harmonicsBuf,
            gaussianCount: gaussianCount,
            shComponents: 1
        )

        // Create output textures
        guard let colorTexture = makeColorTexture(device: device, width: width, height: height, pixelFormat: .rgba8Unorm),
              let depthTexture = makeDepthTexture(device: device, width: width, height: height)
        else {
            XCTFail("Failed to create textures")
            return
        }

        // Render and wait
        guard let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        renderer.render(
            commandBuffer: cb,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            input: input,
            camera: camera,
            width: width,
            height: height
        )

        cb.commit()
        cb.waitUntilCompleted()

        // Read back header to check values
        let header = renderer.debugReadHeader()
        print("\n=== DepthFirst Pipeline Debug ===")
        print("Visible count: \(header.visibleCount)")
        print("Total instances: \(header.totalInstances)")
        print("Padded visible count: \(header.paddedVisibleCount)")
        print("Padded instance count: \(header.paddedInstanceCount)")
        print("Overflow: \(header.overflow)")

        // Check for overflow
        XCTAssertEqual(header.overflow, 0, "Should not overflow with small scene")

        // Check visible count is reasonable
        XCTAssertGreaterThan(header.visibleCount, 0, "Should have some visible gaussians")
        XCTAssertLessThanOrEqual(Int(header.visibleCount), gaussianCount, "Visible count should not exceed total")

        // Check instances are created
        XCTAssertGreaterThan(header.totalInstances, 0, "Should have some instances")

        print("Test passed!")
    }

    /// Simple unit test for depth sort with known input
    func testDepthSortSimple() throws {
        // Create a simple test case: 10 visible gaussians with known depths
        // We'll manually create compacted depth keys and primitive indices
        // and verify the sort produces correct output

        let visibleCount = 10

        // Create depth keys in REVERSE order (10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
        // After sorting, they should be (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        var depthKeys: [UInt32] = []
        var primitiveIndices: [Int32] = []
        for i in 0 ..< visibleCount {
            // Depth key: higher = further back
            // Using simple values, not XOR encoded for clarity
            depthKeys.append(UInt32(visibleCount - i)) // 10, 9, 8, ..., 1
            primitiveIndices.append(Int32(i * 100)) // 0, 100, 200, ..., 900
        }

        print("\n=== Depth Sort Unit Test ===")
        print("Input depth keys: \(depthKeys)")
        print("Input primitive indices: \(primitiveIndices)")

        // Expected output after sorting by depth key (ascending):
        // depthKeys: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        // primitiveIndices: 900, 800, 700, 600, 500, 400, 300, 200, 100, 0
        let expectedIndices: [Int32] = [900, 800, 700, 600, 500, 400, 300, 200, 100, 0]

        // Allocate buffers
        guard let depthKeysBuf = device.makeBuffer(bytes: &depthKeys, length: depthKeys.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let primitiveIndicesBuf = device.makeBuffer(bytes: &primitiveIndices, length: primitiveIndices.count * MemoryLayout<Int32>.stride, options: .storageModeShared)
        else {
            XCTFail("Failed to create input buffers")
            return
        }

        // Create header with known values
        var header = DepthFirstHeaderSwift(
            visibleCount: UInt32(visibleCount),
            totalInstances: 0,
            paddedVisibleCount: UInt32((visibleCount + 1023) / 1024 * 1024), // Align to 1024
            paddedInstanceCount: 0,
            overflow: 0,
            padding0: 0,
            padding1: 0,
            padding2: 0
        )

        guard let headerBuf = device.makeBuffer(bytes: &header, length: MemoryLayout<DepthFirstHeaderSwift>.stride, options: .storageModeShared)
        else {
            XCTFail("Failed to create header buffer")
            return
        }

        print("Header: visibleCount=\(header.visibleCount), paddedVisibleCount=\(header.paddedVisibleCount)")

        // Create scratch buffers for radix sort
        let radixAlignment = 256 * 4 // blockSize * grainSize
        let paddedCount = Int(header.paddedVisibleCount)
        let gridSize = (paddedCount + radixAlignment - 1) / radixAlignment
        let histogramSize = gridSize * 256 * MemoryLayout<UInt32>.stride
        let blockSumsSize = ((gridSize * 256 + 255) / 256) * MemoryLayout<UInt32>.stride

        // Use shared mode for debugging intermediate values
        guard let histogramBuf = device.makeBuffer(length: max(histogramSize, 1024), options: .storageModeShared),
              let blockSumsBuf = device.makeBuffer(length: max(blockSumsSize, 1024), options: .storageModeShared),
              let scannedHistBuf = device.makeBuffer(length: max(histogramSize, 1024), options: .storageModeShared),
              let scratchKeysBuf = device.makeBuffer(length: paddedCount * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let scratchPayloadBuf = device.makeBuffer(length: paddedCount * MemoryLayout<Int32>.stride, options: .storageModeShared)
        else {
            XCTFail("Failed to create scratch buffers")
            return
        }

        // Create dispatch args buffer
        var dispatchArgs = [UInt32](repeating: 0, count: 16 * 3) // 16 slots * 3 uints each
        // Set up depth histogram dispatch: gridSize threadgroups
        let histogramSlot = 0
        dispatchArgs[histogramSlot * 3 + 0] = UInt32(max(gridSize, 1))
        dispatchArgs[histogramSlot * 3 + 1] = 1
        dispatchArgs[histogramSlot * 3 + 2] = 1

        // Scan blocks dispatch
        let histBlocks = (gridSize * 256 + 255) / 256
        let scanBlocksSlot = 1
        dispatchArgs[scanBlocksSlot * 3 + 0] = UInt32(max(histBlocks, 1))
        dispatchArgs[scanBlocksSlot * 3 + 1] = 1
        dispatchArgs[scanBlocksSlot * 3 + 2] = 1

        // Exclusive scan dispatch
        let exclusiveSlot = 2
        dispatchArgs[exclusiveSlot * 3 + 0] = 1
        dispatchArgs[exclusiveSlot * 3 + 1] = 1
        dispatchArgs[exclusiveSlot * 3 + 2] = 1

        // Apply offsets dispatch
        let applySlot = 3
        dispatchArgs[applySlot * 3 + 0] = UInt32(max(histBlocks, 1))
        dispatchArgs[applySlot * 3 + 1] = 1
        dispatchArgs[applySlot * 3 + 2] = 1

        // Scatter dispatch
        let scatterSlot = 4
        dispatchArgs[scatterSlot * 3 + 0] = UInt32(max(gridSize, 1))
        dispatchArgs[scatterSlot * 3 + 1] = 1
        dispatchArgs[scatterSlot * 3 + 2] = 1

        guard let dispatchArgsBuf = device.makeBuffer(bytes: &dispatchArgs, length: dispatchArgs.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else {
            XCTFail("Failed to create dispatch args buffer")
            return
        }

        print("Dispatch: gridSize=\(gridSize), histBlocks=\(histBlocks)")

        // Load the library and create depth sort encoder
        guard let libPath = Bundle.module.url(forResource: "DepthFirstShaders", withExtension: "metallib"),
              let library = try? device.makeLibrary(URL: libPath)
        else {
            XCTFail("Failed to load DepthFirstShaders library")
            return
        }

        let sortBuffers = RadixSortBuffers(
            histogram: histogramBuf,
            blockSums: blockSumsBuf,
            scannedHistogram: scannedHistBuf,
            scratchKeys: scratchKeysBuf,
            scratchIndices: scratchPayloadBuf
        )

        let encoder: DepthRadixSortEncoder
        do {
            encoder = try DepthRadixSortEncoder(device: self.device, library: library, precision: .bits32)
        } catch {
            XCTFail("Failed to create DepthRadixSortEncoder: \(error)")
            return
        }

        // Create dispatch offsets for test (slots 0-4)
        let depthDispatchOffsets = DepthRadixSortEncoder.DispatchOffsets.fromSlots(
            histogram: 0,
            scanBlocks: 1,
            exclusive: 2,
            apply: 3,
            scatter: 4,
            stride: MemoryLayout<UInt32>.stride * 3
        )

        // Run the sort
        guard let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        encoder.encode(
            commandBuffer: cb,
            depthKeys: depthKeysBuf,
            sortedIndices: primitiveIndicesBuf,
            sortBuffers: sortBuffers,
            sortHeader: headerBuf,
            dispatchBuffer: dispatchArgsBuf,
            offsets: depthDispatchOffsets,
            label: "TestDepthSort"
        )

        cb.commit()
        cb.waitUntilCompleted()

        // Debug: Print scanned histogram for bins 0-15
        let scannedHistPtr = scannedHistBuf.contents().bindMemory(to: UInt32.self, capacity: 256)
        print("Scanned histogram for bins 0-15: \(Array(UnsafeBufferPointer(start: scannedHistPtr, count: 16)))")

        // Debug: Print scratch keys after passes
        let scratchKeysPtr = scratchKeysBuf.contents().bindMemory(to: UInt32.self, capacity: min(20, paddedCount))
        print("Scratch keys (first 20): \(Array(UnsafeBufferPointer(start: scratchKeysPtr, count: min(20, paddedCount))))")

        // Read back results
        let sortedKeysPtr = depthKeysBuf.contents().bindMemory(to: UInt32.self, capacity: visibleCount)
        let sortedIndicesPtr = primitiveIndicesBuf.contents().bindMemory(to: Int32.self, capacity: visibleCount)

        let sortedKeys = Array(UnsafeBufferPointer(start: sortedKeysPtr, count: visibleCount))
        let sortedIndices = Array(UnsafeBufferPointer(start: sortedIndicesPtr, count: visibleCount))

        print("Output depth keys: \(sortedKeys)")
        print("Output primitive indices: \(sortedIndices)")
        print("Expected primitive indices: \(expectedIndices)")

        // Verify depth keys are sorted
        var isSorted = true
        for i in 1 ..< sortedKeys.count {
            if sortedKeys[i] < sortedKeys[i - 1] {
                isSorted = false
                print("NOT SORTED at index \(i): \(sortedKeys[i - 1]) > \(sortedKeys[i])")
            }
        }
        XCTAssertTrue(isSorted, "Depth keys should be sorted")

        // Verify primitive indices match expected
        XCTAssertEqual(sortedIndices, expectedIndices, "Primitive indices should match expected order")
    }

    /// Unit test for depth sort at scale (to verify multi-block radix sort works)
    func testDepthSortAtScale() throws {
        let visibleCount = 1_000_000 // 1 million elements

        // Create depth keys - values are 16-bit (0-65535) but we have millions of them
        // Generate random-ish distribution in 16-bit range
        var depthKeys: [UInt32] = []
        var primitiveIndices: [Int32] = []
        for i in 0 ..< visibleCount {
            // Depth value in 16-bit range, spread across the range
            let key = UInt32((i * 37 + 12345) & 0xFFFF) // Simple hash to spread values
            depthKeys.append(key)
            primitiveIndices.append(Int32(i)) // Payload is original index
        }

        print("\n=== Depth Sort Unit Test (Scale) ===")
        print("Visible count: \(visibleCount)")
        print("First 10 input depth keys: \(depthKeys.prefix(10).map { String($0) }.joined(separator: ", "))")

        // Allocate buffers
        guard let depthKeysBuf = device.makeBuffer(bytes: &depthKeys, length: depthKeys.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let primitiveIndicesBuf = device.makeBuffer(bytes: &primitiveIndices, length: primitiveIndices.count * MemoryLayout<Int32>.stride, options: .storageModeShared)
        else {
            XCTFail("Failed to create input buffers")
            return
        }

        // Create header
        let radixAlignment = 256 * 4
        let paddedCount = ((visibleCount + radixAlignment - 1) / radixAlignment) * radixAlignment
        var header = DepthFirstHeaderSwift(
            visibleCount: UInt32(visibleCount),
            totalInstances: 0,
            paddedVisibleCount: UInt32(paddedCount),
            paddedInstanceCount: 0,
            overflow: 0,
            padding0: 0,
            padding1: 0,
            padding2: 0
        )

        guard let headerBuf = device.makeBuffer(bytes: &header, length: MemoryLayout<DepthFirstHeaderSwift>.stride, options: .storageModeShared)
        else {
            XCTFail("Failed to create header buffer")
            return
        }

        // Create scratch buffers
        let gridSize = (paddedCount + radixAlignment - 1) / radixAlignment
        let histogramSize = gridSize * 256 * MemoryLayout<UInt32>.stride
        let histBlocks = (gridSize * 256 + 255) / 256
        let blockSumsSize = histBlocks * MemoryLayout<UInt32>.stride

        guard let histogramBuf = device.makeBuffer(length: max(histogramSize, 1024), options: .storageModePrivate),
              let blockSumsBuf = device.makeBuffer(length: max(blockSumsSize, 1024), options: .storageModePrivate),
              let scannedHistBuf = device.makeBuffer(length: max(histogramSize, 1024), options: .storageModePrivate),
              let scratchKeysBuf = device.makeBuffer(length: paddedCount * MemoryLayout<UInt32>.stride, options: .storageModePrivate),
              let scratchPayloadBuf = device.makeBuffer(length: paddedCount * MemoryLayout<Int32>.stride, options: .storageModePrivate)
        else {
            XCTFail("Failed to create scratch buffers")
            return
        }

        // Create dispatch args (using depth sort slot indices 0-4)
        var dispatchArgs = [UInt32](repeating: 0, count: 32 * 3)
        dispatchArgs[0 * 3 + 0] = UInt32(max(gridSize, 1))
        dispatchArgs[0 * 3 + 1] = 1
        dispatchArgs[0 * 3 + 2] = 1
        dispatchArgs[1 * 3 + 0] = UInt32(max(histBlocks, 1))
        dispatchArgs[1 * 3 + 1] = 1
        dispatchArgs[1 * 3 + 2] = 1
        dispatchArgs[2 * 3 + 0] = 1
        dispatchArgs[2 * 3 + 1] = 1
        dispatchArgs[2 * 3 + 2] = 1
        dispatchArgs[3 * 3 + 0] = UInt32(max(histBlocks, 1))
        dispatchArgs[3 * 3 + 1] = 1
        dispatchArgs[3 * 3 + 2] = 1
        dispatchArgs[4 * 3 + 0] = UInt32(max(gridSize, 1))
        dispatchArgs[4 * 3 + 1] = 1
        dispatchArgs[4 * 3 + 2] = 1

        guard let dispatchArgsBuf = device.makeBuffer(bytes: &dispatchArgs, length: dispatchArgs.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else {
            XCTFail("Failed to create dispatch args buffer")
            return
        }

        print("Dispatch: gridSize=\(gridSize), histBlocks=\(histBlocks), paddedCount=\(paddedCount)")

        // Load library
        guard let libPath = Bundle.module.url(forResource: "DepthFirstShaders", withExtension: "metallib"),
              let library = try? device.makeLibrary(URL: libPath)
        else {
            XCTFail("Failed to load library")
            return
        }

        let sortBuffers = RadixSortBuffers(
            histogram: histogramBuf,
            blockSums: blockSumsBuf,
            scannedHistogram: scannedHistBuf,
            scratchKeys: scratchKeysBuf,
            scratchIndices: scratchPayloadBuf
        )

        let encoder = try DepthRadixSortEncoder(device: device, library: library, precision: .bits32)

        // Create dispatch offsets for test (slots 0-4)
        let depthDispatchOffsets = DepthRadixSortEncoder.DispatchOffsets.fromSlots(
            histogram: 0,
            scanBlocks: 1,
            exclusive: 2,
            apply: 3,
            scatter: 4,
            stride: MemoryLayout<UInt32>.stride * 3
        )

        // Initialize scratch buffers
        guard let cb1 = queue.makeCommandBuffer(),
              let blit = cb1.makeBlitCommandEncoder()
        else {
            XCTFail("Failed to create blit encoder")
            return
        }
        blit.fill(buffer: depthKeysBuf, range: visibleCount * 4 ..< depthKeysBuf.length, value: 0xFF)
        blit.fill(buffer: primitiveIndicesBuf, range: visibleCount * 4 ..< primitiveIndicesBuf.length, value: 0xFF)
        blit.fill(buffer: scratchKeysBuf, range: 0 ..< scratchKeysBuf.length, value: 0xFF)
        blit.fill(buffer: scratchPayloadBuf, range: 0 ..< scratchPayloadBuf.length, value: 0xFF)
        blit.endEncoding()
        cb1.commit()
        cb1.waitUntilCompleted()

        // Run sort
        guard let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        encoder.encode(
            commandBuffer: cb,
            depthKeys: depthKeysBuf,
            sortedIndices: primitiveIndicesBuf,
            sortBuffers: sortBuffers,
            sortHeader: headerBuf,
            dispatchBuffer: dispatchArgsBuf,
            offsets: depthDispatchOffsets,
            label: "TestDepthSortAtScale"
        )

        cb.commit()
        cb.waitUntilCompleted()

        // Read back results
        let sortedKeysPtr = depthKeysBuf.contents().bindMemory(to: UInt32.self, capacity: visibleCount)
        let sortedKeys = Array(UnsafeBufferPointer(start: sortedKeysPtr, count: visibleCount))

        print("First 20 output depth keys: \(sortedKeys.prefix(20).map { String($0) }.joined(separator: ", "))")

        // Verify sorted
        var isSorted = true
        var unsortedCount = 0
        for i in 1 ..< sortedKeys.count {
            if sortedKeys[i] < sortedKeys[i - 1] {
                isSorted = false
                unsortedCount += 1
                if unsortedCount <= 10 {
                    print("NOT SORTED at index \(i): \(sortedKeys[i - 1]) > \(sortedKeys[i])")
                }
            }
        }

        print("Total unsorted pairs: \(unsortedCount) out of \(visibleCount - 1)")
        XCTAssertTrue(isSorted, "Depth keys should be fully sorted (found \(unsortedCount) unsorted pairs)")
    }

    /// Unit test for tile sort at scale (now uses counting sort instead of radix sort)
    func testTileSortAtScale() throws {
        // Counting sort replaced radix sort for tile IDs
        // This test is now covered by the full pipeline test (testRenderPLYScene)
        throw XCTSkip("Tile radix sort replaced with counting sort - use testRenderPLYScene for validation")
    }

    /// Debug test to trace depth distribution through the pipeline
    func testDepthDistributionDebug() throws {
        let plyPath = "/Users/lakiiinbor/Downloads/models-2/garden/point_cloud/iteration_30000/point_cloud.ply"
        let url = URL(fileURLWithPath: plyPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("PLY file not found")
        }

        let dataset = try PLYLoader.load(url: url)
        let gaussianCount = dataset.records.count
        print("\n=== Depth Distribution Debug Test (Simple) ===")
        print("Loaded \(gaussianCount) gaussians")

        let width = 1920
        let height = 1080

        var packed: [PackedWorldGaussian] = []
        for record in dataset.records {
            packed.append(PackedWorldGaussian(
                position: record.position,
                scale: record.scale,
                rotation: SIMD4<Float>(record.rotation.imag.x, record.rotation.imag.y, record.rotation.imag.z, record.rotation.real),
                opacity: record.opacity
            ))
        }

        var harmonics = dataset.harmonics

        guard let gaussianBuf = device.makeBuffer(bytes: &packed, length: packed.count * MemoryLayout<PackedWorldGaussian>.stride, options: .storageModeShared),
              let harmonicsBuf = device.makeBuffer(bytes: &harmonics, length: harmonics.count * MemoryLayout<Float>.stride, options: .storageModeShared)
        else {
            XCTFail("Failed to create buffers")
            return
        }

        let config = RendererConfig(maxGaussians: gaussianCount, maxWidth: width, maxHeight: height, precision: .float32)
        let renderer = try DepthFirstRenderer(config: config)

        let viewMatrix = simd_float4x4([[0.7994084, -0.22484124, 0.557129, 0.0], [0.28039712, 0.9597669, -0.014999399, 0.0], [-0.53134143, 0.16820799, 0.8302905, 0.0], [-1.7380741, -13.886897, 11.441933, 1.0]])
        let projectionMatrix = makeProjectionMatrix(width: width, height: height, near: 0.1, far: 10.0)

        let aspect = Float(width) / Float(height)
        let fov: Float = 60.0 * .pi / 180.0
        let f = 1.0 / tan(fov / 2.0)

        let camera = CameraParams(
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            position: .zero,
            focalX: Float(width) * f / (2 * aspect),
            focalY: Float(height) * f / 2
        )

        let input = GaussianInput(
            gaussians: gaussianBuf,
            harmonics: harmonicsBuf,
            gaussianCount: gaussianCount,
            shComponents: dataset.shComponents
        )

        guard let colorTexture = makeColorTexture(device: device, width: width, height: height, pixelFormat: .rgba8Unorm),
              let depthTexture = makeDepthTexture(device: device, width: width, height: height)
        else {
            XCTFail("Failed to create textures")
            return
        }

        guard let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        renderer.render(
            commandBuffer: cb,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            input: input,
            camera: camera,
            width: width,
            height: height
        )

        cb.commit()
        cb.waitUntilCompleted()

        // Read header
        let header = renderer.debugReadHeader()
        print("\n=== Header Values ===")
        print("Visible count: \(header.visibleCount)")
        print("Total instances: \(header.totalInstances)")

        // Read tile headers to check which tiles have gaussians
        let tilesX = (width + 31) / 32
        let tilesY = (height + 15) / 16
        let tileCount = tilesX * tilesY
        let tileHeaders = renderer.debugReadTileHeaders(count: tileCount)

        var tilesWithGaussians = 0
        var maxGaussiansPerTile: UInt32 = 0
        var totalTileInstances: UInt64 = 0

        for h in tileHeaders {
            if h.count > 0 {
                tilesWithGaussians += 1
                maxGaussiansPerTile = max(maxGaussiansPerTile, h.count)
                totalTileInstances += UInt64(h.count)
            }
        }

        print("\n=== Tile Statistics ===")
        print("Tiles with gaussians: \(tilesWithGaussians) / \(tileCount)")
        print("Max gaussians per tile: \(maxGaussiansPerTile)")
        print("Total tile instances (from headers): \(totalTileInstances)")
        print("Expected total instances: \(header.totalInstances)")

        let instanceRatio = Double(totalTileInstances) / Double(header.totalInstances)
        print("\n=== CRITICAL CHECK ===")
        print("Instance ratio (headers/expected): \(instanceRatio * 100)%")

        // Check prefix sum result (orderedTileCounts after prefix sum)
        let offsetsCount = min(Int(header.visibleCount), 1000)
        let offsets = renderer.debugReadInstanceOffsets(count: offsetsCount)
        print("\n=== Prefix Sum (Instance Offsets) Analysis ===")
        print("First 20 offsets: \(offsets.prefix(20).map { String($0) }.joined(separator: ", "))")

        // The offsets should be monotonically increasing
        var offsetsValid = true
        var lastOffset: UInt32 = 0
        for (i, offset) in offsets.enumerated() {
            if offset < lastOffset {
                print("WARNING: Non-monotonic offset at \(i): \(lastOffset) -> \(offset)")
                offsetsValid = false
            }
            lastOffset = offset
        }
        print("Offsets monotonic: \(offsetsValid)")

        // Also check offset spacing (should roughly equal tile counts)
        var offsetDiffs: [UInt32] = []
        for i in 1 ..< min(50, offsets.count) {
            offsetDiffs.append(offsets[i] - offsets[i - 1])
        }
        print("First 20 offset differences (tile counts): \(offsetDiffs.prefix(20).map { String($0) }.joined(separator: ", "))")

        // Check sorted primitive indices at start
        let primIndices = renderer.debugReadSortedPrimitiveIndices(count: 50)
        print("\n=== First 20 Sorted Primitive Indices ===")
        print(primIndices.prefix(20).map { String($0) }.joined(separator: ", "))

        // Check bounds for first few sorted gaussians
        print("\n=== Bounds for First 10 Sorted Gaussians ===")
        for i in 0 ..< min(10, primIndices.count) {
            let origIdx = primIndices[i]
            if origIdx >= 0 {
                let bounds = renderer.debugReadSingleBounds(at: Int(origIdx))
                let tileCount = (bounds.y - bounds.x + 1) * (bounds.w - bounds.z + 1)
                print("  Depth pos \(i): origIdx=\(origIdx), bounds=(\(bounds.x),\(bounds.y),\(bounds.z),\(bounds.w)), tiles=\(tileCount)")
            } else {
                print("  Depth pos \(i): origIdx=\(origIdx) - SENTINEL")
            }
        }

        // Read the sorted tile IDs to check what was actually written
        let tileIdsSample = min(Int(header.totalInstances), 10000)
        let sortedTileIds = renderer.debugReadSortedTileIds(count: tileIdsSample)

        var validTileIds = 0
        var sentinelCount = 0
        for tid in sortedTileIds {
            if tid == 0xFFFF_FFFF {
                sentinelCount += 1
            } else {
                validTileIds += 1
            }
        }
        print("\n=== Sorted Tile IDs Analysis ===")
        print("Sample size: \(tileIdsSample)")
        print("Valid tile IDs: \(validTileIds), Sentinel values: \(sentinelCount)")
        print("First 20 tile IDs: \(sortedTileIds.prefix(20).map { String($0) }.joined(separator: ", "))")

        // Find where sentinels start
        if let firstSentinel = sortedTileIds.firstIndex(where: { $0 == 0xFFFF_FFFF }) {
            print("First sentinel at position: \(firstSentinel)")
        }

        XCTAssertGreaterThan(header.visibleCount, 0, "Should have visible gaussians")
        XCTAssertGreaterThan(tilesWithGaussians, 0, "Should have tiles with gaussians")

        if instanceRatio < 0.9 {
            print("\nWARNING: Significant instance loss detected!")
        }
    }

    /// Test with PLY data to check for overflow
    func testDepthFirstWithPLYData() throws {
        let plyPath = "/Users/lakiiinbor/Downloads/models-2/garden/point_cloud/iteration_30000/point_cloud.ply"
        let url = URL(fileURLWithPath: plyPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("PLY file not found")
        }

        let dataset = try PLYLoader.load(url: url)
        let gaussianCount = dataset.records.count
        print("\n=== DepthFirst PLY Debug Test ===")
        print("Loaded \(gaussianCount) gaussians")

        let width = 1920
        let height = 1080

        var packed: [PackedWorldGaussian] = []
        for record in dataset.records {
            packed.append(PackedWorldGaussian(
                position: record.position,
                scale: record.scale,
                rotation: SIMD4<Float>(record.rotation.imag.x, record.rotation.imag.y, record.rotation.imag.z, record.rotation.real),
                opacity: record.opacity
            ))
        }

        var harmonics = dataset.harmonics

        guard let gaussianBuf = device.makeBuffer(bytes: &packed, length: packed.count * MemoryLayout<PackedWorldGaussian>.stride, options: .storageModeShared),
              let harmonicsBuf = device.makeBuffer(bytes: &harmonics, length: harmonics.count * MemoryLayout<Float>.stride, options: .storageModeShared)
        else {
            XCTFail("Failed to create buffers")
            return
        }

        let config = RendererConfig(maxGaussians: gaussianCount, maxWidth: width, maxHeight: height, precision: .float32)
        let renderer = try DepthFirstRenderer(config: config)

        // Same camera as benchmark test
        let viewMatrix = simd_float4x4([[0.7994084, -0.22484124, 0.557129, 0.0], [0.28039712, 0.9597669, -0.014999399, 0.0], [-0.53134143, 0.16820799, 0.8302905, 0.0], [-1.7380741, -13.886897, 11.441933, 1.0]])
        let projectionMatrix = makeProjectionMatrix(width: width, height: height, near: 0.1, far: 10.0)

        let aspect = Float(width) / Float(height)
        let fov: Float = 60.0 * .pi / 180.0
        let f = 1.0 / tan(fov / 2.0)

        let camera = CameraParams(
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            position: .zero,
            focalX: Float(width) * f / (2 * aspect),
            focalY: Float(height) * f / 2
        )

        let input = GaussianInput(
            gaussians: gaussianBuf,
            harmonics: harmonicsBuf,
            gaussianCount: gaussianCount,
            shComponents: dataset.shComponents
        )

        guard let colorTexture = makeColorTexture(device: device, width: width, height: height, pixelFormat: .rgba8Unorm),
              let depthTexture = makeDepthTexture(device: device, width: width, height: height)
        else {
            XCTFail("Failed to create textures")
            return
        }

        guard let cb = queue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }

        renderer.render(
            commandBuffer: cb,
            colorTexture: colorTexture,
            depthTexture: depthTexture,
            input: input,
            camera: camera,
            width: width,
            height: height
        )

        cb.commit()
        cb.waitUntilCompleted()

        // Read back header
        let header = renderer.debugReadHeader()
        print("\n=== Header Values ===")
        print("Visible count: \(header.visibleCount)")
        print("Total instances: \(header.totalInstances)")
        print("Padded visible count: \(header.paddedVisibleCount)")
        print("Padded instance count: \(header.paddedInstanceCount)")
        print("Overflow: \(header.overflow)")

        // Calculate max instances
        let maxInstances = gaussianCount * 32
        print("\nMax instances: \(maxInstances)")
        print("Instance utilization: \(Float(header.totalInstances) / Float(maxInstances) * 100)%")

        if header.overflow != 0 {
            print("WARNING: Instance overflow detected!")
        }

        // Print active tile count
        let activeTileCountPtr = renderer.debugReadActiveTileCount()
        print("Active tile count: \(activeTileCountPtr)")

        // Read more sorted tile IDs to verify sorting
        let sampleSize = min(10000, Int(header.totalInstances))
        let sortedTileIds = renderer.debugReadSortedTileIds(count: sampleSize)
        print("\n=== First 50 Sorted Tile IDs ===")
        print(sortedTileIds.prefix(50).map { String($0) }.joined(separator: ", "))

        // Check distribution of tile IDs
        var tileIdCounts: [UInt32: Int] = [:]
        for tid in sortedTileIds {
            tileIdCounts[tid, default: 0] += 1
        }
        print("\n=== Tile ID Distribution (top 20) ===")
        let sortedCounts = tileIdCounts.sorted { $0.value > $1.value }.prefix(20)
        for (tid, count) in sortedCounts {
            print("Tile \(tid): \(count) instances")
        }

        // Check if sorted
        var sortedCount = 0
        var unsortedCount = 0
        for i in 1 ..< sortedTileIds.count {
            if sortedTileIds[i] >= sortedTileIds[i - 1] {
                sortedCount += 1
            } else {
                unsortedCount += 1
                if unsortedCount <= 5 {
                    print("NOT SORTED at index \(i): \(sortedTileIds[i - 1]) > \(sortedTileIds[i])")
                }
            }
        }
        print("\nSorted pairs: \(sortedCount), Unsorted pairs: \(unsortedCount)")
        print("Sort quality: \(Float(sortedCount) / Float(sortedCount + unsortedCount) * 100)%")

        // Count unique tile IDs
        print("Unique tile IDs in sample: \(tileIdCounts.count)")
        print("Min tile ID: \(sortedTileIds.min() ?? 0), Max tile ID: \(sortedTileIds.max() ?? 0)")

        // Read tile bounds to debug
        let tileBounds = renderer.debugReadTileBounds(count: min(100, Int(header.visibleCount)))
        print("\n=== Sample Tile Bounds (first 20 non-empty) ===")
        var nonEmptyCount = 0
        for (i, bounds) in tileBounds.enumerated() {
            if bounds.x <= bounds.y, bounds.z <= bounds.w {
                print("Gaussian \(i): minTX=\(bounds.x), maxTX=\(bounds.y), minTY=\(bounds.z), maxTY=\(bounds.w)")
                nonEmptyCount += 1
                if nonEmptyCount >= 20 { break }
            }
        }

        // Count bounds statistics
        var emptyBounds = 0
        var validBounds = 0
        for bounds in tileBounds {
            if bounds.x > bounds.y || bounds.z > bounds.w {
                emptyBounds += 1
            } else {
                validBounds += 1
            }
        }
        print("\nBounds stats: \(validBounds) valid, \(emptyBounds) empty/culled out of \(tileBounds.count)")

        // Read sorted primitive indices to trace the indirection
        let sortedPrimitiveIndices = renderer.debugReadSortedPrimitiveIndices(count: min(50, Int(header.visibleCount)))
        print("\n=== First 20 Sorted Primitive Indices ===")
        print(sortedPrimitiveIndices.prefix(20).map { String($0) }.joined(separator: ", "))

        // Now trace: for depth position 0, what original index, what bounds, what tile ID should be?
        print("\n=== Trace first 10 depth positions ===")
        for i in 0 ..< min(10, sortedPrimitiveIndices.count) {
            let originalIdx = sortedPrimitiveIndices[i]
            if originalIdx >= 0, originalIdx < tileBounds.count {
                let bounds = tileBounds[Int(originalIdx)]
                let minTX = bounds.x
                let maxTX = bounds.y
                let minTY = bounds.z
                let maxTY = bounds.w
                if minTX <= maxTX, minTY <= maxTY {
                    // Expected tile ID for (minTY, minTX)
                    let expectedTileId = minTY * 60 + minTX // tilesX = 60
                    print("Depth pos \(i): originalIdx=\(originalIdx), bounds=(\(minTX),\(maxTX),\(minTY),\(maxTY)), expected first tileID=\(expectedTileId)")
                } else {
                    print("Depth pos \(i): originalIdx=\(originalIdx), bounds INVALID")
                }
            } else {
                print("Depth pos \(i): originalIdx=\(originalIdx) out of range or negative")
            }
        }

        // Check instance offsets (prefix sum result)
        let instanceOffsets = renderer.debugReadInstanceOffsets(count: min(20, Int(header.visibleCount)))
        print("\n=== First 20 Instance Offsets (prefix sum results) ===")
        print(instanceOffsets.prefix(20).map { String($0) }.joined(separator: ", "))

        // Check nTouchedTiles (original tile counts before depth ordering)
        let nTouchedTiles = renderer.debugReadNTouchedTiles(count: min(20, Int(header.visibleCount)))
        print("\n=== First 20 nTouchedTiles (original counts) ===")
        print(nTouchedTiles.prefix(20).map { String($0) }.joined(separator: ", "))

        // Check instanceGaussianIndices (should be valid gaussian indices if createInstances worked)
        let gaussianIndices = renderer.debugReadInstanceGaussianIndices(count: min(100, Int(header.totalInstances)))
        print("\n=== First 50 Instance Gaussian Indices ===")
        print(gaussianIndices.prefix(50).map { String($0) }.joined(separator: ", "))

        // Count unique gaussian indices
        var uniqueGaussians: Set<Int32> = []
        var negativeCount = 0
        var outOfRangeCount = 0
        for idx in gaussianIndices {
            if idx < 0 { negativeCount += 1 }
            else if idx >= header.visibleCount { outOfRangeCount += 1 }
            uniqueGaussians.insert(idx)
        }
        print("Unique gaussian indices: \(uniqueGaussians.count), negative: \(negativeCount), out of range: \(outOfRangeCount)")

        // Read depth keys to check compaction
        let depthKeysSample = renderer.debugReadDepthKeys(count: min(50, Int(header.visibleCount)))
        print("\n=== First 50 Depth Keys (after sort, should be sorted) ===")
        print(depthKeysSample.prefix(50).map { String($0) }.joined(separator: ", "))

        // Check if depth keys are sorted
        var depthSorted = 0
        var depthUnsorted = 0
        for i in 1 ..< depthKeysSample.count {
            if depthKeysSample[i] >= depthKeysSample[i - 1] {
                depthSorted += 1
            } else {
                depthUnsorted += 1
            }
        }
        print("Depth keys: sorted pairs=\(depthSorted), unsorted pairs=\(depthUnsorted)")

        // Read scratch buffers to see intermediate sort state
        let scratchKeys = renderer.debugReadScratchDepthKeys(count: min(50, Int(header.visibleCount)))
        print("\n=== First 50 Scratch Depth Keys (after pass 0) ===")
        print(scratchKeys.prefix(50).map { String($0) }.joined(separator: ", "))

        let nonZeroScratchKeys = scratchKeys.filter { $0 != 0 }.count
        print("Non-zero scratch depth keys: \(nonZeroScratchKeys) out of \(scratchKeys.count)")

        let scratchIndices = renderer.debugReadScratchPrimitiveIndices(count: min(50, Int(header.visibleCount)))
        print("\n=== First 50 Scratch Primitive Indices (after pass 0) ===")
        print(scratchIndices.prefix(50).map { String($0) }.joined(separator: ", "))

        let nonZeroScratchIndices = scratchIndices.filter { $0 != 0 }.count
        print("Non-zero scratch primitive indices: \(nonZeroScratchIndices) out of \(scratchIndices.count)")

        // Also read from a range in the middle of the buffer to check if data is there
        let midStart = max(0, Int(header.visibleCount) / 2 - 25)
        let midDepthKeys = renderer.debugReadDepthKeys(count: min(50, Int(header.visibleCount) - midStart))
        print("\n=== Depth Keys around position \(midStart) ===")
        print(midDepthKeys.prefix(20).map { String($0) }.joined(separator: ", "))

        let midIndices = renderer.debugReadSortedPrimitiveIndices(count: min(50, Int(header.visibleCount) - midStart))
        print("\n=== Primitive Indices around position \(midStart) ===")
        print(midIndices.prefix(20).map { String($0) }.joined(separator: ", "))
    }
}

// MARK: - Extensions

extension DepthFirstRenderer {
    /// Debug method to read header values from left view
    func debugReadHeader() -> DepthFirstHeaderSwift {
        guard let frame = debugLeftFrame else {
            return DepthFirstHeaderSwift(visibleCount: 0, totalInstances: 0, paddedVisibleCount: 0, paddedInstanceCount: 0, overflow: 0, padding0: 0, padding1: 0, padding2: 0)
        }
        let ptr = frame.header.contents().bindMemory(to: DepthFirstHeaderSwift.self, capacity: 1)
        return ptr.pointee
    }

    /// Debug method to read active tile count from left view
    func debugReadActiveTileCount() -> UInt32 {
        guard let frame = debugLeftFrame else { return 0 }
        let ptr = frame.activeTileCount.contents().bindMemory(to: UInt32.self, capacity: 1)
        return ptr.pointee
    }

    /// Debug method to read sorted tile IDs (requires blit copy for private buffers)
    func debugReadSortedTileIds(count: Int) -> [UInt32] {
        guard let frame = debugLeftFrame else { return [] }
        // instanceTileIds is private, need to copy to shared buffer
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let sharedBuffer = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else {
            return []
        }

        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: frame.instanceTileIds, sourceOffset: 0,
                      to: sharedBuffer, destinationOffset: 0,
                      size: min(count * MemoryLayout<UInt32>.stride, frame.instanceTileIds.length))
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = sharedBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Debug method to read a single bounds entry at a specific index
    func debugReadSingleBounds(at index: Int) -> SIMD4<Int32> {
        guard let frame = debugLeftFrame else { return SIMD4<Int32>(0, -1, 0, -1) }
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let sharedBuffer = device.makeBuffer(length: MemoryLayout<SIMD4<Int32>>.stride, options: .storageModeShared)
        else {
            return SIMD4<Int32>(0, -1, 0, -1)
        }

        let sourceOffset = index * MemoryLayout<SIMD4<Int32>>.stride
        if sourceOffset >= frame.bounds.length {
            return SIMD4<Int32>(0, -1, 0, -1)
        }

        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: frame.bounds, sourceOffset: sourceOffset,
                      to: sharedBuffer, destinationOffset: 0,
                      size: MemoryLayout<SIMD4<Int32>>.stride)
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = sharedBuffer.contents().bindMemory(to: SIMD4<Int32>.self, capacity: 1)
        return ptr.pointee
    }

    /// Debug method to read tile bounds (requires blit copy for private buffers)
    func debugReadTileBounds(count: Int) -> [SIMD4<Int32>] {
        guard let frame = debugLeftFrame else { return [] }
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let sharedBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD4<Int32>>.stride, options: .storageModeShared)
        else {
            return []
        }

        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: frame.bounds, sourceOffset: 0,
                      to: sharedBuffer, destinationOffset: 0,
                      size: min(count * MemoryLayout<SIMD4<Int32>>.stride, frame.bounds.length))
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = sharedBuffer.contents().bindMemory(to: SIMD4<Int32>.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Debug method to read sorted primitive indices
    func debugReadSortedPrimitiveIndices(count: Int) -> [Int32] {
        self.debugReadSortedPrimitiveIndicesRange(start: 0, count: count)
    }

    /// Debug method to read sorted primitive indices from a specific range
    func debugReadSortedPrimitiveIndicesRange(start: Int, count: Int) -> [Int32] {
        guard let frame = debugLeftFrame else { return [] }
        let sourceOffset = start * MemoryLayout<Int32>.stride
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let sharedBuffer = device.makeBuffer(length: count * MemoryLayout<Int32>.stride, options: .storageModeShared)
        else {
            return []
        }

        if let blit = cb.makeBlitCommandEncoder() {
            let copySize = min(count * MemoryLayout<Int32>.stride, frame.primitiveIndices.length - sourceOffset)
            if copySize > 0 {
                blit.copy(from: frame.primitiveIndices, sourceOffset: sourceOffset,
                          to: sharedBuffer, destinationOffset: 0,
                          size: copySize)
            }
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = sharedBuffer.contents().bindMemory(to: Int32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Debug method to read instance offsets (prefix sum result in orderedTileCounts)
    func debugReadInstanceOffsets(count: Int) -> [UInt32] {
        guard let frame = debugLeftFrame else { return [] }
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let sharedBuffer = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else {
            return []
        }

        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: frame.orderedTileCounts, sourceOffset: 0,
                      to: sharedBuffer, destinationOffset: 0,
                      size: min(count * MemoryLayout<UInt32>.stride, frame.orderedTileCounts.length))
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = sharedBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Debug method to read nTouchedTiles (original tile counts per gaussian)
    func debugReadNTouchedTiles(count: Int) -> [UInt32] {
        guard let frame = debugLeftFrame else { return [] }
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let sharedBuffer = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else {
            return []
        }

        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: frame.nTouchedTiles, sourceOffset: 0,
                      to: sharedBuffer, destinationOffset: 0,
                      size: min(count * MemoryLayout<UInt32>.stride, frame.nTouchedTiles.length))
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = sharedBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Debug method to read instanceGaussianIndices (after tile sort)
    func debugReadInstanceGaussianIndices(count: Int) -> [Int32] {
        guard let frame = debugLeftFrame else { return [] }
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let sharedBuffer = device.makeBuffer(length: count * MemoryLayout<Int32>.stride, options: .storageModeShared)
        else {
            return []
        }

        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: frame.instanceGaussianIndices, sourceOffset: 0,
                      to: sharedBuffer, destinationOffset: 0,
                      size: min(count * MemoryLayout<Int32>.stride, frame.instanceGaussianIndices.length))
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = sharedBuffer.contents().bindMemory(to: Int32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Debug method to read depthKeys (after depth sort, should be sorted)
    func debugReadDepthKeys(count: Int) -> [UInt32] {
        guard let frame = debugLeftFrame else { return [] }
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let sharedBuffer = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else {
            return []
        }

        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: frame.depthKeys, sourceOffset: 0,
                      to: sharedBuffer, destinationOffset: 0,
                      size: min(count * MemoryLayout<UInt32>.stride, frame.depthKeys.length))
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = sharedBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Debug method to read scratch depth keys (intermediate state from pass 0)
    func debugReadScratchDepthKeys(count: Int) -> [UInt32] {
        guard let frame = debugLeftFrame else { return [] }
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let sharedBuffer = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else {
            return []
        }

        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: frame.depthSortScratchKeys, sourceOffset: 0,
                      to: sharedBuffer, destinationOffset: 0,
                      size: min(count * MemoryLayout<UInt32>.stride, frame.depthSortScratchKeys.length))
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = sharedBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Debug method to read scratch primitive indices (intermediate state from pass 0)
    func debugReadScratchPrimitiveIndices(count: Int) -> [Int32] {
        guard let frame = debugLeftFrame else { return [] }
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let sharedBuffer = device.makeBuffer(length: count * MemoryLayout<Int32>.stride, options: .storageModeShared)
        else {
            return []
        }

        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: frame.depthSortScratchPayload, sourceOffset: 0,
                      to: sharedBuffer, destinationOffset: 0,
                      size: min(count * MemoryLayout<Int32>.stride, frame.depthSortScratchPayload.length))
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = sharedBuffer.contents().bindMemory(to: Int32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Debug method to read render data to check stored depth values
    func debugReadRenderData(count: Int) -> [(depth: Float, meanX: Float, meanY: Float, opacity: Float)] {
        guard let frame = debugLeftFrame else { return [] }
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let sharedBuffer = device.makeBuffer(length: count * MemoryLayout<GaussianRenderData>.stride, options: .storageModeShared)
        else {
            return []
        }

        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: frame.renderData, sourceOffset: 0,
                      to: sharedBuffer, destinationOffset: 0,
                      size: min(count * MemoryLayout<GaussianRenderData>.stride, frame.renderData.length))
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = sharedBuffer.contents().bindMemory(to: GaussianRenderData.self, capacity: count)
        var result: [(depth: Float, meanX: Float, meanY: Float, opacity: Float)] = []
        for i in 0 ..< count {
            let g = ptr[i]
            result.append((depth: Float(g.depth), meanX: Float(g.meanX), meanY: Float(g.meanY), opacity: Float(g.opacity)))
        }
        return result
    }

    /// Debug method to read ordered tile counts (before prefix sum modifies them)
    /// Note: After prefix sum, this buffer contains offsets, not counts!
    func debugReadOrderedTileCounts(count: Int) -> [UInt32] {
        guard let frame = debugLeftFrame else { return [] }
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let sharedBuffer = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else {
            return []
        }

        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: frame.orderedTileCounts, sourceOffset: 0,
                      to: sharedBuffer, destinationOffset: 0,
                      size: min(count * MemoryLayout<UInt32>.stride, frame.orderedTileCounts.length))
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = sharedBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Debug method to read tile headers
    func debugReadTileHeaders(count: Int) -> [(offset: UInt32, count: UInt32)] {
        guard let frame = debugLeftFrame else { return [] }
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let sharedBuffer = device.makeBuffer(length: count * MemoryLayout<GaussianHeader>.stride, options: .storageModeShared)
        else {
            return []
        }

        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: frame.tileHeaders, sourceOffset: 0,
                      to: sharedBuffer, destinationOffset: 0,
                      size: min(count * MemoryLayout<GaussianHeader>.stride, frame.tileHeaders.length))
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = sharedBuffer.contents().bindMemory(to: GaussianHeader.self, capacity: count)
        var result: [(offset: UInt32, count: UInt32)] = []
        for i in 0 ..< count {
            let h = ptr[i]
            result.append((offset: h.offset, count: h.count))
        }
        return result
    }
}

/// Swift struct matching DepthFirstHeader in Metal
struct DepthFirstHeaderSwift {
    var visibleCount: UInt32
    var totalInstances: UInt32
    var paddedVisibleCount: UInt32
    var paddedInstanceCount: UInt32
    var overflow: UInt32
    var padding0: UInt32
    var padding1: UInt32
    var padding2: UInt32
}

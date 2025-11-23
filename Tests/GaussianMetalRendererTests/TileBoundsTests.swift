import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

final class TileBoundsTests: XCTestCase {
    
    var device: MTLDevice!
    var library: MTLLibrary!
    var queue: MTLCommandQueue!
    
    override func setUp() {
        super.setUp()
        let renderer = Renderer.shared
        self.device = renderer.device
        self.library = renderer.library
        self.queue = renderer.queue
    }
    
    func testTileBoundsAndScatter() throws {
        let tileBoundsEncoder = try TileBoundsEncoder(device: device, library: library)
        let coverageEncoder = try CoverageEncoder(device: device, library: library)
        let scatterEncoder = try ScatterEncoder(device: device, library: library)
        
        let width: UInt32 = 100
        let height: UInt32 = 100
        let tileW: UInt32 = 16
        let tileH: UInt32 = 16
        let tilesX = (width + tileW - 1) / tileW // 7
        let tilesY = (height + tileH - 1) / tileH // 7
        let gaussianCount = 3
        
        let params = RenderParams(
            width: width,
            height: height,
            tileWidth: tileW,
            tileHeight: tileH,
            tilesX: tilesX,
            tilesY: tilesY,
            maxPerTile: 0,
            whiteBackground: 0,
            activeTileCount: 0,
            gaussianCount: UInt32(gaussianCount)
        )
        
        // Scenario:
        // Gaussian 0: At (20, 20), radius 5. Tile (1, 1). 
        //   Bounds: (20-5, 20+5) = (15, 25).
        //   X tiles: 15/16=0, 25/16=1. -> minX=0, maxX=1.
        //   Y tiles: 15/16=0, 25/16=1. -> minY=0, maxY=1.
        //   Should cover 4 tiles: (0,0), (0,1), (1,0), (1,1).
        
        // Gaussian 1: At (50, 50), radius 2. Tile (3, 3).
        //   Bounds: (48, 52).
        //   X tiles: 48/16=3, 52/16=3. -> minX=3, maxX=3.
        //   Y tiles: 3. -> minY=3, maxY=3.
        //   Should cover 1 tile: (3,3).
        
        // Gaussian 2: Masked out (mask=0). Should cover 0 tiles.
        
        let means: [SIMD2<Float>] = [SIMD2(20, 20), SIMD2(50, 50), SIMD2(1000, 1000)]
        let radii: [Float] = [5.0, 2.0, 100.0]
        let mask: [UInt8] = [1, 1, 0] // 3rd is masked
        
        let meansBuf = device.makeBuffer(bytes: means, length: 3 * 8, options: .storageModeShared)!
        let radiiBuf = device.makeBuffer(bytes: radii, length: 3 * 4, options: .storageModeShared)!
        let maskBuf = device.makeBuffer(bytes: mask, length: 3, options: .storageModeShared)!
        
        let dummy = device.makeBuffer(length: 16, options: .storageModePrivate)!
        let inputs = GaussianInputBuffers(
            means: meansBuf,
            radii: radiiBuf,
            mask: maskBuf,
            depths: dummy,
            conics: dummy,
            colors: dummy,
            opacities: dummy
        )
        
        let boundsBuf = device.makeBuffer(length: gaussianCount * MemoryLayout<SIMD4<Int32>>.stride, options: .storageModeShared)!
        
        let commandBuffer = queue.makeCommandBuffer()!
        
        // 1. Tile Bounds
        tileBoundsEncoder.encode(
            commandBuffer: commandBuffer,
            gaussianBuffers: inputs,
            boundsBuffer: boundsBuf,
            params: params,
            gaussianCount: gaussianCount
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let bounds = boundsBuf.contents().bindMemory(to: SIMD4<Int32>.self, capacity: gaussianCount)
        
        // Verify Gaussian 0
        // minX=0, maxX=1, minY=0, maxY=1
        XCTAssertEqual(bounds[0], SIMD4<Int32>(0, 1, 0, 1))
        
        // Verify Gaussian 1
        // minX=3, maxX=3, minY=3, maxY=3
        XCTAssertEqual(bounds[1], SIMD4<Int32>(3, 3, 3, 3))
        
        // Verify Gaussian 2 (Masked)
        // Should be (0, -1, 0, -1) or similar invalid range
        XCTAssertTrue(bounds[2].y < bounds[2].x || bounds[2].w < bounds[2].z)
        
        // 2. Coverage
        let coverageBuf = device.makeBuffer(length: gaussianCount * 4, options: .storageModeShared)!
        let offsetsBuf = device.makeBuffer(length: (gaussianCount + 1) * 4, options: .storageModeShared)!
        let partialSums = device.makeBuffer(length: 1024, options: .storageModePrivate)!
        let headerBuf = device.makeBuffer(length: MemoryLayout<TileAssignmentHeaderSwift>.stride, options: .storageModeShared)!
        let opacities = [Float](repeating: 1.0, count: gaussianCount)
        let opacitiesBuf = device.makeBuffer(bytes: opacities, length: gaussianCount * MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        let commandBuffer2 = queue.makeCommandBuffer()!
        
        coverageEncoder.encode(
            commandBuffer: commandBuffer2,
            gaussianCount: gaussianCount,
            boundsBuffer: boundsBuf,
            opacitiesBuffer: opacitiesBuf,
            coverageBuffer: coverageBuf,
            offsetsBuffer: offsetsBuf,
            partialSumsBuffer: partialSums,
            tileAssignmentHeader: headerBuf
        )
        
        commandBuffer2.commit()
        commandBuffer2.waitUntilCompleted()
        
        let coverage = coverageBuf.contents().bindMemory(to: UInt32.self, capacity: gaussianCount)
        let offsets = offsetsBuf.contents().bindMemory(to: UInt32.self, capacity: gaussianCount + 1)
        
        XCTAssertEqual(coverage[0], 4) // (1-0+1) * (1-0+1) = 2*2=4
        XCTAssertEqual(coverage[1], 1) // 1*1 = 1
        XCTAssertEqual(coverage[2], 0) // Masked
        
        XCTAssertEqual(offsets[0], 0)
        XCTAssertEqual(offsets[1], 4)
        XCTAssertEqual(offsets[2], 5)
        XCTAssertEqual(offsets[3], 5) // Total = 5
        
        // 3. Scatter
        let totalAssignments = Int(offsets[3])
        let tileIndices = device.makeBuffer(length: totalAssignments * 4, options: .storageModeShared)!
        let tileIds = device.makeBuffer(length: totalAssignments * 4, options: .storageModeShared)!
        let dispatchBuf = device.makeBuffer(length: 12, options: .storageModePrivate)! // 3 * 4
        
        let commandBuffer3 = queue.makeCommandBuffer()!
        
        scatterEncoder.encode(
            commandBuffer: commandBuffer3,
            gaussianCount: gaussianCount,
            tilesX: Int(tilesX),
            offsetsBuffer: offsetsBuf,
            dispatchBuffer: dispatchBuf,
            boundsBuffer: boundsBuf,
            tileIndicesBuffer: tileIndices,
            tileIdsBuffer: tileIds,
            tileAssignmentHeader: headerBuf
        )
        
        commandBuffer3.commit()
        commandBuffer3.waitUntilCompleted()
        
        let outIndices = tileIndices.contents().bindMemory(to: Int32.self, capacity: totalAssignments)
        
        // Expected Assignments:
        // Gaussian 0 (Tiles 0,1,7,8 -> (0,0),(1,0),(0,1),(1,1) if tilesX=7)
        // (0,0) -> id 0
        // (1,0) -> id 1
        // (0,1) -> id 7
        // (1,1) -> id 8
        // The scatter kernel loop:
        // for ty in minY..maxY:
        //   for tx in minX..maxX:
        //     ...
        
        // G0: minX=0, maxX=1, minY=0, maxY=1
        // y=0: x=0 (id 0), x=1 (id 1)
        // y=1: x=0 (id 7), x=1 (id 8)
        
        // Assignments should be:
        // 0: G0, Tile 0
        // 1: G0, Tile 1
        // 2: G0, Tile 7
        // 3: G0, Tile 8
        // 4: G1, Tile (3,3) -> 3*7 + 3 = 24
        
        var histogram: [Int32: Int] = [:]
        for i in 0..<totalAssignments {
            histogram[outIndices[i], default: 0] += 1
        }
        let assigned = histogram.values.reduce(0, +)
        XCTAssertEqual(assigned, totalAssignments)
        XCTAssertGreaterThanOrEqual(histogram[Int32(0)] ?? 0, 4)
    }
}

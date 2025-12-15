import Metal
@testable import Renderer
import simd
import XCTest

/// Unit tests for LOD (Level of Detail) tree system
final class LODUnitTests: XCTestCase {
    var device: MTLDevice!
    var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        self.device = MTLCreateSystemDefaultDevice()!
        self.queue = self.device.makeCommandQueue()!
    }

    // MARK: - LODTreeBuilder Tests

    /// Test building LOD tree from simple Gaussian records
    func testLODTreeBuilderBasic() throws {
        let count = 100
        var records: [GaussianRecord] = []
        var harmonics: [Float] = []

        // Create simple grid of Gaussians
        for i in 0..<count {
            let x = Float(i % 10) * 0.1 - 0.5
            let y = Float(i / 10) * 0.1 - 0.5
            let z = Float(i) * 0.01

            records.append(GaussianRecord(
                position: SIMD3(x, y, z),
                scale: SIMD3(0.02, 0.02, 0.02),
                rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
                opacity: 0.8,
                sh: []
            ))

            // DC SH coefficients (RGB)
            harmonics.append(contentsOf: [0.5, 0.5, 0.5])
        }

        // Build tree
        let tree = LODTreeBuilder.build(from: records, harmonics: harmonics, shComponents: 1)

        // Verify tree structure
        XCTAssertEqual(Int(tree.header.leafCount), count, "Leaf count should match input")
        XCTAssertEqual(Int(tree.header.interiorCount), count - 1, "Interior count should be N-1")
        XCTAssertEqual(Int(tree.header.nodeCount), 2 * count - 1, "Total nodes should be 2N-1")
        XCTAssertEqual(tree.nodes.count, 2 * count - 1, "Node array size should match")
        XCTAssertEqual(tree.mergedGaussians.count, count - 1, "Merged Gaussians should be N-1")

        // Verify leaf nodes reference original Gaussians (not necessarily in order with top-down BVH)
        var referencedGaussians = Set<UInt32>()
        for i in 0..<count {
            let leafIndex = tree.header.interiorCount + UInt32(i)
            let node = tree.nodes[Int(leafIndex)]
            XCTAssertEqual(node.isLeaf, 1, "Should be marked as leaf")
            XCTAssertLessThan(node.gaussianIndex, UInt32(count), "Should reference a valid original Gaussian")
            referencedGaussians.insert(node.gaussianIndex)
        }
        XCTAssertEqual(referencedGaussians.count, count, "All original Gaussians should be referenced exactly once")

        // Verify root node
        let root = tree.nodes[0]
        XCTAssertEqual(root.parentIndex, UInt32.max, "Root should have no parent")
        XCTAssertEqual(root.isLeaf, 0, "Root should not be a leaf")
        XCTAssertEqual(Int(root.subtreeCount), count, "Root should contain all Gaussians")

        print("LOD Tree built successfully:")
        print("  Nodes: \(tree.header.nodeCount)")
        print("  Leaves: \(tree.header.leafCount)")
        print("  Interior: \(tree.header.interiorCount)")
        print("  Max depth: \(tree.header.maxDepth)")
    }

    /// Test single Gaussian edge case
    func testLODTreeBuilderSingleGaussian() throws {
        let records = [GaussianRecord(
            position: SIMD3(0, 0, 1),
            scale: SIMD3(0.1, 0.1, 0.1),
            rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
            opacity: 1.0,
            sh: []
        )]
        let harmonics: [Float] = [0.5, 0.5, 0.5]

        let tree = LODTreeBuilder.build(from: records, harmonics: harmonics, shComponents: 1)

        XCTAssertEqual(Int(tree.header.leafCount), 1)
        XCTAssertEqual(Int(tree.header.interiorCount), 0)
        XCTAssertEqual(Int(tree.header.nodeCount), 1)
        XCTAssertEqual(tree.nodes[0].isLeaf, 1)
    }

    /// Test empty input edge case
    func testLODTreeBuilderEmpty() throws {
        let tree = LODTreeBuilder.build(from: [], harmonics: [], shComponents: 0)

        XCTAssertEqual(Int(tree.header.nodeCount), 0)
        XCTAssertEqual(tree.nodes.count, 0)
    }

    // MARK: - LODResources Tests

    /// Test GPU resource allocation
    func testLODResourcesAllocation() throws {
        let count = 1000
        var records: [GaussianRecord] = []
        var harmonics: [Float] = []

        for i in 0..<count {
            records.append(GaussianRecord(
                position: SIMD3(Float(i % 32) * 0.1, Float(i / 32) * 0.1, Float(i) * 0.001),
                scale: SIMD3(0.02, 0.02, 0.02),
                rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
                opacity: 0.8,
                sh: []
            ))
            harmonics.append(contentsOf: [0.5, 0.5, 0.5])
        }

        let tree = LODTreeBuilder.build(from: records, harmonics: harmonics, shComponents: 1)
        let resources = try LODResources(device: device, tree: tree)

        XCTAssertEqual(resources.header.nodeCount, tree.header.nodeCount)
        XCTAssertEqual(resources.header.leafCount, tree.header.leafCount)

        // Verify buffer sizes
        let expectedNodeSize = MemoryLayout<LODNode>.stride * Int(tree.header.nodeCount)
        XCTAssertGreaterThanOrEqual(resources.nodeBuffer.length, expectedNodeSize)

        let expectedMergedSize = MemoryLayout<PackedWorldGaussian>.stride * Int(tree.header.interiorCount)
        XCTAssertGreaterThanOrEqual(resources.mergedGaussiansBuffer.length, max(expectedMergedSize, 1))

        print("LOD Resources allocated:")
        print("  Node buffer: \(resources.nodeBuffer.length) bytes")
        print("  Merged Gaussians: \(resources.mergedGaussiansBuffer.length) bytes")
    }

    // MARK: - LODManager Tests

    /// Test LODManager initialization and tree building
    func testLODManagerInit() throws {
        let lodManager = try LODManager(device: device)
        XCTAssertFalse(lodManager.isReady, "Should not be ready before tree is built")

        // Build tree
        let count = 500
        var records: [GaussianRecord] = []
        var harmonics: [Float] = []

        for i in 0..<count {
            records.append(GaussianRecord(
                position: SIMD3(Float(i % 25) * 0.1 - 1.25, Float(i / 25) * 0.1 - 1.0, Float(i) * 0.002 + 2.0),
                scale: SIMD3(0.03, 0.03, 0.03),
                rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
                opacity: 0.7,
                sh: []
            ))
            harmonics.append(contentsOf: [0.5, 0.3, 0.2])
        }

        let header = try lodManager.buildTree(from: records, harmonics: harmonics, shComponents: 1)

        XCTAssertTrue(lodManager.isReady, "Should be ready after tree is built")
        XCTAssertEqual(Int(header.leafCount), count)
        XCTAssertNotNil(lodManager.treeHeader)

        print("LODManager initialized and tree built:")
        print("  Total nodes: \(header.nodeCount)")
        print("  Leaf nodes: \(header.leafCount)")
        print("  Interior nodes: \(header.interiorCount)")
    }

    // MARK: - AABB Tests

    /// Test that AABBs are computed correctly
    func testAABBComputation() throws {
        // Create a single Gaussian and verify its AABB
        let pos = SIMD3<Float>(1.0, 2.0, 3.0)
        let scale = SIMD3<Float>(0.1, 0.2, 0.3)

        let records = [GaussianRecord(
            position: pos,
            scale: scale,
            rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
            opacity: 1.0,
            sh: []
        )]
        let harmonics: [Float] = [0.5, 0.5, 0.5]

        let tree = LODTreeBuilder.build(from: records, harmonics: harmonics, shComponents: 1)
        let node = tree.nodes[0]

        // AABB should be position ± 3σ
        let expectedMin = pos - scale * 3.0
        let expectedMax = pos + scale * 3.0

        XCTAssertEqual(node.aabbMinX, expectedMin.x, accuracy: 0.001)
        XCTAssertEqual(node.aabbMinY, expectedMin.y, accuracy: 0.001)
        XCTAssertEqual(node.aabbMinZ, expectedMin.z, accuracy: 0.001)
        XCTAssertEqual(node.aabbMaxX, expectedMax.x, accuracy: 0.001)
        XCTAssertEqual(node.aabbMaxY, expectedMax.y, accuracy: 0.001)
        XCTAssertEqual(node.aabbMaxZ, expectedMax.z, accuracy: 0.001)
    }

    /// Test that parent AABBs contain children
    func testAABBHierarchy() throws {
        let count = 100
        var records: [GaussianRecord] = []
        var harmonics: [Float] = []

        for i in 0..<count {
            records.append(GaussianRecord(
                position: SIMD3(Float(i % 10) * 0.5, Float(i / 10) * 0.5, Float(i) * 0.05),
                scale: SIMD3(0.05, 0.05, 0.05),
                rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
                opacity: 0.8,
                sh: []
            ))
            harmonics.append(contentsOf: [0.5, 0.5, 0.5])
        }

        let tree = LODTreeBuilder.build(from: records, harmonics: harmonics, shComponents: 1)

        // Check that each interior node's AABB contains its children's AABBs
        for i in 0..<Int(tree.header.interiorCount) {
            let node = tree.nodes[i]
            if node.leftChild != UInt32.max && node.rightChild != UInt32.max {
                let left = tree.nodes[Int(node.leftChild)]
                let right = tree.nodes[Int(node.rightChild)]

                // Parent min should be <= children mins
                XCTAssertLessThanOrEqual(node.aabbMinX, left.aabbMinX)
                XCTAssertLessThanOrEqual(node.aabbMinX, right.aabbMinX)
                XCTAssertLessThanOrEqual(node.aabbMinY, left.aabbMinY)
                XCTAssertLessThanOrEqual(node.aabbMinY, right.aabbMinY)
                XCTAssertLessThanOrEqual(node.aabbMinZ, left.aabbMinZ)
                XCTAssertLessThanOrEqual(node.aabbMinZ, right.aabbMinZ)

                // Parent max should be >= children maxs
                XCTAssertGreaterThanOrEqual(node.aabbMaxX, left.aabbMaxX)
                XCTAssertGreaterThanOrEqual(node.aabbMaxX, right.aabbMaxX)
                XCTAssertGreaterThanOrEqual(node.aabbMaxY, left.aabbMaxY)
                XCTAssertGreaterThanOrEqual(node.aabbMaxY, right.aabbMaxY)
                XCTAssertGreaterThanOrEqual(node.aabbMaxZ, left.aabbMaxZ)
                XCTAssertGreaterThanOrEqual(node.aabbMaxZ, right.aabbMaxZ)
            }
        }
    }

    // MARK: - Merged Gaussian Tests

    /// Test that merged Gaussians have reasonable values
    func testMergedGaussianValues() throws {
        // Create two Gaussians with known positions
        let pos1 = SIMD3<Float>(-1.0, 0.0, 5.0)
        let pos2 = SIMD3<Float>(1.0, 0.0, 5.0)
        let scale = SIMD3<Float>(0.1, 0.1, 0.1)
        let opacity: Float = 0.8

        let records = [
            GaussianRecord(position: pos1, scale: scale, rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), opacity: opacity, sh: []),
            GaussianRecord(position: pos2, scale: scale, rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), opacity: opacity, sh: [])
        ]
        let harmonics: [Float] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        let tree = LODTreeBuilder.build(from: records, harmonics: harmonics, shComponents: 1)

        // Should have 1 interior node (the root)
        XCTAssertEqual(tree.mergedGaussians.count, 1)

        let merged = tree.mergedGaussians[0]

        // Merged position should be average (since equal weights)
        let expectedPos = (pos1 + pos2) * 0.5
        XCTAssertEqual(merged.px, expectedPos.x, accuracy: 0.01)
        XCTAssertEqual(merged.py, expectedPos.y, accuracy: 0.01)
        XCTAssertEqual(merged.pz, expectedPos.z, accuracy: 0.01)

        // Merged opacity is a "falloff" value that can exceed 1.0 per the paper
        // (gets clamped during rendering). Should still be positive.
        XCTAssertGreaterThan(merged.opacity, 0, "Merged falloff should be positive")

        print("Merged Gaussian:")
        print("  Position: (\(merged.px), \(merged.py), \(merged.pz))")
        print("  Scale: (\(merged.sx), \(merged.sy), \(merged.sz))")
        print("  Opacity: \(merged.opacity)")
    }

    // MARK: - Helper Functions

    /// Create camera parameters for testing
    /// Uses Metal-style projection where w = z_view (positive for objects in front)
    private func makeCameraParams(width: Int, height: Int, position: SIMD3<Float>) -> CameraParams {
        let fovY: Float = 60.0 * .pi / 180.0
        let aspect = Float(width) / Float(height)
        let near: Float = 0.1
        let far: Float = 100.0

        // View matrix: camera at position, looking down +Z
        // Transform world coords to view coords where +Z is forward
        let viewMatrix = simd_float4x4(
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(-position.x, -position.y, -position.z, 1)
        )

        // Metal-style projection matrix where w = z (positive for objects in front)
        // Maps [near, far] to [0, 1] NDC depth
        let tanHalfFov = tan(fovY / 2)
        let projectionMatrix = simd_float4x4(
            SIMD4<Float>(1 / (aspect * tanHalfFov), 0, 0, 0),
            SIMD4<Float>(0, 1 / tanHalfFov, 0, 0),
            SIMD4<Float>(0, 0, far / (far - near), 1),  // w = z (positive)
            SIMD4<Float>(0, 0, -near * far / (far - near), 0)
        )

        let focalY = Float(height) / (2 * tanHalfFov)
        let focalX = focalY * aspect

        return CameraParams(
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            position: position,
            focalX: focalX,
            focalY: focalY,
            near: near,
            far: far
        )
    }
}

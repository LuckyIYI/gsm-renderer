import Foundation
import Metal
@testable import Renderer
import simd
import XCTest

/// Tests for mesh extraction from Gaussian Opacity Fields
final class MeshExtractionTests: XCTestCase {
    // Use same PLY path as PLYBenchmarkTests
    static let plyPath = "/Users/laki/Desktop/point_cloud_garden.ply"

    var device: MTLDevice!

    override func setUp() {
        super.setUp()
        self.device = MTLCreateSystemDefaultDevice()!
    }

    // MARK: - OBJ Export Helper

    /// Saves an ExtractedMesh to an OBJ file
    func saveMeshToOBJ(mesh: ExtractedMesh, filename: String) {
        var objContent = "# OBJ file exported from Gaussian Opacity Fields mesh extraction\n"
        objContent += "# Vertices: \(mesh.vertexCount), Triangles: \(mesh.triangleCount)\n\n"

        // Write vertices
        for position in mesh.positions {
            objContent += "v \(position.x) \(position.y) \(position.z)\n"
        }

        objContent += "\n"

        // Write normals
        for normal in mesh.normals {
            objContent += "vn \(normal.x) \(normal.y) \(normal.z)\n"
        }

        objContent += "\n"

        // Write faces (OBJ uses 1-based indexing)
        for i in stride(from: 0, to: mesh.indices.count, by: 3) {
            let v0 = mesh.indices[i] + 1
            let v1 = mesh.indices[i + 1] + 1
            let v2 = mesh.indices[i + 2] + 1
            // Format: f v0//n0 v1//n1 v2//n2 (vertex//normal)
            objContent += "f \(v0)//\(v0) \(v1)//\(v1) \(v2)//\(v2)\n"
        }

        let desktopURL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Desktop/\(filename)")

        do {
            try objContent.write(to: desktopURL, atomically: true, encoding: .utf8)
            print("Saved mesh to: \(desktopURL.path)")
        } catch {
            print("Failed to save OBJ file: \(error)")
        }
    }

    // MARK: - Mesh Extraction Tests

    /// Test mesh extraction with synthetic gaussian data
    func testMeshExtractionSynthetic() throws {
        print("\n=== Mesh Extraction Test (Synthetic Data) ===")

        // Generate a simple sphere-like arrangement of gaussians
        let gaussians = generateSphereGaussians(count: 100, radius: 1.0, center: .zero)

        var packed: [PackedWorldGaussian] = []
        packed.reserveCapacity(gaussians.count)
        for i in 0..<gaussians.count {
            packed.append(PackedWorldGaussian(
                position: gaussians.positions[i],
                scale: gaussians.scales[i],
                rotation: gaussians.rotations[i],
                opacity: gaussians.opacities[i]
            ))
        }

        guard let gaussianBuffer = device.makeBuffer(
            bytes: &packed,
            length: packed.count * MemoryLayout<PackedWorldGaussian>.stride,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create gaussian buffer")
            return
        }

        // Create mesh extractor
        let extractor: MeshExtractor
        do {
            extractor = try MeshExtractor(device: device)
        } catch {
            throw XCTSkip("MeshExtractor initialization failed (Metal library may not be compiled): \(error)")
        }

        // Extract mesh with lower resolution for speed
        let config = MeshExtractionConfig(
            gridResolution: 64,
            isoLevel: 0.5,
            margin: 0.2,
            sigmaCutoff: 3.0
        )

        let startTime = CFAbsoluteTimeGetCurrent()
        let mesh = try extractor.extractMesh(
            gaussians: gaussianBuffer,
            gaussianCount: gaussians.count,
            useHalfPrecision: false,
            config: config
        )
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        print("Extracted mesh in \(String(format: "%.2f", elapsed * 1000))ms")
        print("Vertices: \(mesh.vertexCount)")
        print("Triangles: \(mesh.triangleCount)")

        // Save to OBJ
        if mesh.triangleCount > 0 {
            saveMeshToOBJ(mesh: mesh, filename: "mesh_synthetic.obj")
        }

        XCTAssertGreaterThan(mesh.triangleCount, 0, "Should extract some triangles from sphere arrangement")
    }

    /// Test mesh extraction from PLY scene data
    func testMeshExtractionFromPLY() throws {
        let url = URL(fileURLWithPath: Self.plyPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("PLY file not found at \(Self.plyPath)")
        }

        print("\n=== Mesh Extraction Test (PLY Scene) ===")

        // Load PLY
        let loadStart = CFAbsoluteTimeGetCurrent()
        let dataset = try PLYLoader.load(url: url)
        let loadElapsed = CFAbsoluteTimeGetCurrent() - loadStart
        print("Loaded \(dataset.records.count) gaussians in \(String(format: "%.2f", loadElapsed))s")

        let bounds = GaussianSceneBuilder.bounds(of: dataset.records)
        print("Scene center: \(bounds.center)")
        print("Scene radius: \(bounds.radius)")

        // Convert to packed format
        var packed: [PackedWorldGaussian] = []
        packed.reserveCapacity(dataset.records.count)
        for record in dataset.records {
            packed.append(PackedWorldGaussian(
                position: record.position,
                scale: record.scale,
                rotation: SIMD4<Float>(record.rotation.imag.x, record.rotation.imag.y, record.rotation.imag.z, record.rotation.real),
                opacity: record.opacity
            ))
        }

        guard let gaussianBuffer = device.makeBuffer(
            bytes: &packed,
            length: packed.count * MemoryLayout<PackedWorldGaussian>.stride,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create gaussian buffer")
            return
        }

        // Create mesh extractor
        let extractor: MeshExtractor
        do {
            extractor = try MeshExtractor(device: device)
        } catch {
            throw XCTSkip("MeshExtractor initialization failed (Metal library may not be compiled): \(error)")
        }

        // Extract mesh - use moderate resolution for balance of quality and speed
        let config = MeshExtractionConfig(
            gridResolution: 128,
            isoLevel: 0.5,
            margin: bounds.radius * 0.1,
            sigmaCutoff: 3.0,
            opacityCutoff: 1.0 / 255.0
        )

        print("\nExtracting mesh (grid: \(config.gridResolution)³)...")
        let extractStart = CFAbsoluteTimeGetCurrent()
        let mesh = try extractor.extractMesh(
            gaussians: gaussianBuffer,
            gaussianCount: dataset.records.count,
            useHalfPrecision: false,
            config: config
        )
        let extractElapsed = CFAbsoluteTimeGetCurrent() - extractStart

        print("Extraction completed in \(String(format: "%.2f", extractElapsed))s")
        print("Vertices: \(mesh.vertexCount)")
        print("Triangles: \(mesh.triangleCount)")

        // Calculate mesh statistics
        if mesh.vertexCount > 0 {
            var minPos = mesh.positions[0]
            var maxPos = mesh.positions[0]
            for pos in mesh.positions {
                minPos = SIMD3<Float>(min(minPos.x, pos.x), min(minPos.y, pos.y), min(minPos.z, pos.z))
                maxPos = SIMD3<Float>(max(maxPos.x, pos.x), max(maxPos.y, pos.y), max(maxPos.z, pos.z))
            }
            print("Mesh bounds: [\(minPos)] to [\(maxPos)]")
            print("Mesh extent: \(maxPos - minPos)")
        }

        // Save to OBJ
        if mesh.triangleCount > 0 {
            saveMeshToOBJ(mesh: mesh, filename: "mesh_from_ply.obj")

            // Also save a high-resolution version if not too slow
            if dataset.records.count < 500000 {
                print("\nExtracting high-resolution mesh (grid: 256³)...")
                let hiResConfig = MeshExtractionConfig(
                    gridResolution: 256,
                    isoLevel: 0.5,
                    margin: bounds.radius * 0.1,
                    sigmaCutoff: 3.0
                )

                let hiResStart = CFAbsoluteTimeGetCurrent()
                let hiResMesh = try extractor.extractMesh(
                    gaussians: gaussianBuffer,
                    gaussianCount: dataset.records.count,
                    useHalfPrecision: false,
                    config: hiResConfig
                )
                let hiResElapsed = CFAbsoluteTimeGetCurrent() - hiResStart

                print("High-res extraction completed in \(String(format: "%.2f", hiResElapsed))s")
                print("High-res vertices: \(hiResMesh.vertexCount)")
                print("High-res triangles: \(hiResMesh.triangleCount)")

                if hiResMesh.triangleCount > 0 {
                    saveMeshToOBJ(mesh: hiResMesh, filename: "mesh_from_ply_hires.obj")
                }
            }
        }

        XCTAssertGreaterThan(mesh.vertexCount, 0, "Should extract some vertices from PLY scene")
    }

    /// Test mesh extraction with different iso levels
    func testMeshExtractionIsoLevels() throws {
        let url = URL(fileURLWithPath: Self.plyPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("PLY file not found at \(Self.plyPath)")
        }

        print("\n=== Mesh Extraction Iso-Level Comparison ===")

        let dataset = try PLYLoader.load(url: url)
        let bounds = GaussianSceneBuilder.bounds(of: dataset.records)

        var packed: [PackedWorldGaussian] = []
        packed.reserveCapacity(dataset.records.count)
        for record in dataset.records {
            packed.append(PackedWorldGaussian(
                position: record.position,
                scale: record.scale,
                rotation: SIMD4<Float>(record.rotation.imag.x, record.rotation.imag.y, record.rotation.imag.z, record.rotation.real),
                opacity: record.opacity
            ))
        }

        guard let gaussianBuffer = device.makeBuffer(
            bytes: &packed,
            length: packed.count * MemoryLayout<PackedWorldGaussian>.stride,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create gaussian buffer")
            return
        }

        let extractor: MeshExtractor
        do {
            extractor = try MeshExtractor(device: device)
        } catch {
            throw XCTSkip("MeshExtractor initialization failed: \(error)")
        }

        // Test different iso levels
        let isoLevels: [Float] = [0.3, 0.5, 0.7]

        for isoLevel in isoLevels {
            let config = MeshExtractionConfig(
                gridResolution: 128,
                isoLevel: isoLevel,
                margin: bounds.radius * 0.1,
                sigmaCutoff: 3.0
            )

            let mesh = try extractor.extractMesh(
                gaussians: gaussianBuffer,
                gaussianCount: dataset.records.count,
                useHalfPrecision: false,
                config: config
            )

            print("Iso-level \(isoLevel): \(mesh.triangleCount) triangles")

            if mesh.triangleCount > 0 {
                saveMeshToOBJ(mesh: mesh, filename: "mesh_iso\(String(format: "%.1f", isoLevel)).obj")
            }
        }
    }

    // MARK: - Helper: Generate Sphere Gaussians

    /// Generates gaussians arranged on a sphere surface (for testing)
    func generateSphereGaussians(count: Int, radius: Float, center: SIMD3<Float>) -> GeneratedGaussians {
        var positions: [SIMD3<Float>] = []
        var scales: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var opacities: [Float] = []
        var colors: [SIMD3<Float>] = []

        // Golden angle for uniform sphere distribution
        let goldenAngle = Float.pi * (3.0 - sqrt(5.0))

        for i in 0..<count {
            let y = 1.0 - (Float(i) / Float(count - 1)) * 2.0  // -1 to 1
            let radiusAtY = sqrt(1.0 - y * y)
            let theta = goldenAngle * Float(i)

            let x = cos(theta) * radiusAtY
            let z = sin(theta) * radiusAtY

            positions.append(center + SIMD3<Float>(x, y, z) * radius)
            scales.append(SIMD3<Float>(repeating: radius * 0.15))  // Scale relative to sphere
            rotations.append(SIMD4<Float>(0, 0, 0, 1))  // Identity quaternion
            opacities.append(0.9)  // High opacity for solid surface
            colors.append(SIMD3<Float>(0.7, 0.7, 0.7))  // Gray
        }

        return GeneratedGaussians(
            positions: positions,
            scales: scales,
            rotations: rotations,
            opacities: opacities,
            colors: colors
        )
    }
}

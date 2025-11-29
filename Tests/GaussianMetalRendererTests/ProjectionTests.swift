import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

final class ProjectionTests: XCTestCase {

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

    /// Test packed projection with a single gaussian
    func testPackedProjection() throws {
        let encoder = try ProjectEncoder(device: device, library: library)

        let count = 1

        // Camera setup - simple perspective projection
        let viewMatrix = matrix_identity_float4x4
        var projMatrix = matrix_identity_float4x4
        projMatrix.columns.0 = SIMD4(1, 0, 0, 0)
        projMatrix.columns.1 = SIMD4(0, 1, 0, 0)
        projMatrix.columns.2 = SIMD4(0, 0, 1, 1)
        projMatrix.columns.3 = SIMD4(0, 0, 0, 0)

        let width: Float = 100.0
        let height: Float = 100.0

        let camera = CameraUniformsSwift(
            viewMatrix: viewMatrix,
            projectionMatrix: projMatrix,
            cameraCenter: SIMD3(0, 0, 0),
            pixelFactor: 1.0,
            focalX: width / 2.0,
            focalY: height / 2.0,
            width: width,
            height: height,
            nearPlane: 0.1,
            farPlane: 100.0,
            shComponents: 0,
            gaussianCount: UInt32(count),
            padding0: 0,
            padding1: 0
        )

        // Create test gaussian
        let position = SIMD3<Float>(0, 0, 10)
        let scale = SIMD3<Float>(1, 1, 1)
        let rotation = SIMD4<Float>(0, 0, 0, 1)  // Identity quaternion (scalar-last)
        let opacity: Float = 1.0

        // Create packed world buffer
        var packed = [PackedWorldGaussian(
            position: position,
            scale: scale,
            rotation: rotation,
            opacity: opacity
        )]
        var harmonics = [Float](repeating: 0.5, count: count * 3)  // DC term only

        let packedBuf = device.makeBuffer(bytes: &packed, length: count * MemoryLayout<PackedWorldGaussian>.stride, options: .storageModeShared)!
        let harmonicsBuf = device.makeBuffer(bytes: &harmonics, length: count * 12, options: .storageModeShared)!
        let packedWorldBuffers = PackedWorldBuffers(packedGaussians: packedBuf, harmonics: harmonicsBuf)

        // Output buffers
        let meansOut = device.makeBuffer(length: count * 8, options: .storageModeShared)!
        let conicsOut = device.makeBuffer(length: count * 16, options: .storageModeShared)!
        let colorsOut = device.makeBuffer(length: count * 12, options: .storageModeShared)!
        let opacitiesOut = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let depthsOut = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let radiiOut = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let maskOut = device.makeBuffer(length: count, options: .storageModeShared)!

        let gaussianBuffers = GaussianInputBuffers(
            means: meansOut,
            radii: radiiOut,
            mask: maskOut,
            depths: depthsOut,
            conics: conicsOut,
            colors: colorsOut,
            opacities: opacitiesOut
        )

        let commandBuffer = queue.makeCommandBuffer()!

        encoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: camera,
            gaussianBuffers: gaussianBuffers,
            precision: .float32
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let means = meansOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)
        let radii = radiiOut.contents().bindMemory(to: Float.self, capacity: 1)
        let mask = maskOut.contents().bindMemory(to: UInt8.self, capacity: 1)
        let depths = depthsOut.contents().bindMemory(to: Float.self, capacity: 1)

        XCTAssertEqual(mask[0], 1, "Gaussian should be visible")
        XCTAssertEqual(depths[0], 10.0, accuracy: 0.001)

        // Expected means ~49.5 (center of 100x100 viewport)
        XCTAssertEqual(means[0].x, 49.5, accuracy: 0.1)
        XCTAssertEqual(means[0].y, 49.5, accuracy: 0.1)

        // Check Radii (should be non-zero)
        XCTAssertGreaterThan(radii[0], 0.0)
    }

    /// Test packed projection with multiple gaussians
    func testPackedProjectionMultiple() throws {
        let encoder = try ProjectEncoder(device: device, library: library)

        let count = 4

        // Camera setup
        let viewMatrix = matrix_identity_float4x4
        var projMatrix = matrix_identity_float4x4
        projMatrix.columns.0 = SIMD4(1, 0, 0, 0)
        projMatrix.columns.1 = SIMD4(0, 1, 0, 0)
        projMatrix.columns.2 = SIMD4(0, 0, 1, 1)
        projMatrix.columns.3 = SIMD4(0, 0, 0, 0)

        let width: Float = 100.0
        let height: Float = 100.0

        let camera = CameraUniformsSwift(
            viewMatrix: viewMatrix,
            projectionMatrix: projMatrix,
            cameraCenter: SIMD3(0, 0, 0),
            pixelFactor: 1.0,
            focalX: width / 2.0,
            focalY: height / 2.0,
            width: width,
            height: height,
            nearPlane: 0.1,
            farPlane: 100.0,
            shComponents: 0,
            gaussianCount: UInt32(count),
            padding0: 0,
            padding1: 0
        )

        // Create test gaussians with varied positions
        let positions: [SIMD3<Float>] = [
            SIMD3(0, 0, 10),
            SIMD3(5, 5, 10),
            SIMD3(-3, 2, 15),
            SIMD3(1, -1, 8)
        ]
        let scales: [SIMD3<Float>] = [
            SIMD3(1, 1, 1),
            SIMD3(0.5, 0.5, 0.5),
            SIMD3(2, 1, 1),
            SIMD3(1, 2, 0.5)
        ]
        // Scalar-last quaternion convention: (x, y, z, w) where w is scalar
        let rotations: [SIMD4<Float>] = [
            SIMD4(0, 0, 0, 1),            // Identity
            SIMD4(0.707, 0, 0, 0.707),    // 90° around x-axis
            SIMD4(0, 0, 0, 1),            // Identity
            SIMD4(0.5, 0.5, 0.5, 0.5)     // 120° around (1,1,1) - same in both conventions
        ]
        let opacities: [Float] = [1.0, 0.8, 0.5, 0.9]

        // Create packed world buffers
        var packed: [PackedWorldGaussian] = []
        for i in 0..<count {
            packed.append(PackedWorldGaussian(
                position: positions[i],
                scale: scales[i],
                rotation: rotations[i],
                opacity: opacities[i]
            ))
        }

        var harmonics = [Float](repeating: 0.5, count: count * 3)

        let packedBuf = device.makeBuffer(bytes: &packed, length: count * MemoryLayout<PackedWorldGaussian>.stride, options: .storageModeShared)!
        let harmonicsBuf = device.makeBuffer(bytes: &harmonics, length: count * 12, options: .storageModeShared)!
        let packedWorldBuffers = PackedWorldBuffers(packedGaussians: packedBuf, harmonics: harmonicsBuf)

        // Output buffers
        let meansOut = device.makeBuffer(length: count * 8, options: .storageModeShared)!
        let conicsOut = device.makeBuffer(length: count * 16, options: .storageModeShared)!
        let colorsOut = device.makeBuffer(length: count * 12, options: .storageModeShared)!
        let opacitiesOut = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let depthsOut = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let radiiOut = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let maskOut = device.makeBuffer(length: count, options: .storageModeShared)!

        let gaussianBuffers = GaussianInputBuffers(
            means: meansOut,
            radii: radiiOut,
            mask: maskOut,
            depths: depthsOut,
            conics: conicsOut,
            colors: colorsOut,
            opacities: opacitiesOut
        )

        let commandBuffer = queue.makeCommandBuffer()!

        encoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: camera,
            gaussianBuffers: gaussianBuffers,
            precision: .float32
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let means = meansOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
        let depths = depthsOut.contents().bindMemory(to: Float.self, capacity: count)
        let mask = maskOut.contents().bindMemory(to: UInt8.self, capacity: count)
        let radii = radiiOut.contents().bindMemory(to: Float.self, capacity: count)

        // Check all gaussians are visible and have reasonable values
        for i in 0..<count {
            XCTAssertEqual(mask[i], 1, "Gaussian \(i) should be visible")
            XCTAssertGreaterThan(depths[i], 0, "Gaussian \(i) should have positive depth")
            XCTAssertGreaterThan(radii[i], 0, "Gaussian \(i) should have positive radius")
        }

        // Check specific depth values
        XCTAssertEqual(depths[0], 10.0, accuracy: 0.001, "Gaussian 0 depth")
        XCTAssertEqual(depths[1], 10.0, accuracy: 0.001, "Gaussian 1 depth")
        XCTAssertEqual(depths[2], 15.0, accuracy: 0.001, "Gaussian 2 depth")
        XCTAssertEqual(depths[3], 8.0, accuracy: 0.001, "Gaussian 3 depth")
    }

    /// Test half-precision packed projection (PackedWorldGaussianHalf -> half outputs)
    func testPackedProjectionHalf() throws {
        let encoder = try ProjectEncoder(device: device, library: library)

        let count = 4

        // Camera setup
        let viewMatrix = matrix_identity_float4x4
        var projMatrix = matrix_identity_float4x4
        projMatrix.columns.0 = SIMD4(1, 0, 0, 0)
        projMatrix.columns.1 = SIMD4(0, 1, 0, 0)
        projMatrix.columns.2 = SIMD4(0, 0, 1, 1)
        projMatrix.columns.3 = SIMD4(0, 0, 0, 0)

        let width: Float = 100.0
        let height: Float = 100.0

        let camera = CameraUniformsSwift(
            viewMatrix: viewMatrix,
            projectionMatrix: projMatrix,
            cameraCenter: SIMD3(0, 0, 0),
            pixelFactor: 1.0,
            focalX: width / 2.0,
            focalY: height / 2.0,
            width: width,
            height: height,
            nearPlane: 0.1,
            farPlane: 100.0,
            shComponents: 0,
            gaussianCount: UInt32(count),
            padding0: 0,
            padding1: 0
        )

        // Create test gaussians with varied positions
        let positions: [SIMD3<Float>] = [
            SIMD3(0, 0, 10),
            SIMD3(5, 5, 10),
            SIMD3(-3, 2, 15),
            SIMD3(1, -1, 8)
        ]
        let scales: [SIMD3<Float>] = [
            SIMD3(1, 1, 1),
            SIMD3(0.5, 0.5, 0.5),
            SIMD3(2, 1, 1),
            SIMD3(1, 2, 0.5)
        ]
        // Scalar-last quaternion convention: (x, y, z, w) where w is scalar
        let rotations: [SIMD4<Float>] = [
            SIMD4(0, 0, 0, 1),            // Identity
            SIMD4(0.707, 0, 0, 0.707),    // 90° around x-axis
            SIMD4(0, 0, 0, 1),            // Identity
            SIMD4(0.5, 0.5, 0.5, 0.5)     // 120° around (1,1,1) - same in both conventions
        ]
        let opacities: [Float] = [1.0, 0.8, 0.5, 0.9]

        // Create HALF-PRECISION packed world buffers
        var packed: [PackedWorldGaussianHalf] = []
        for i in 0..<count {
            packed.append(PackedWorldGaussianHalf(
                position: positions[i],
                scale: scales[i],
                rotation: rotations[i],
                opacity: opacities[i]
            ))
        }

        // Verify struct size is correct (24 bytes)
        XCTAssertEqual(MemoryLayout<PackedWorldGaussianHalf>.stride, 24,
                       "PackedWorldGaussianHalf should be 24 bytes")

        var harmonics = [Float](repeating: 0.5, count: count * 3)

        let packedBuf = device.makeBuffer(bytes: &packed,
                                          length: count * MemoryLayout<PackedWorldGaussianHalf>.stride,
                                          options: .storageModeShared)!
        let harmonicsBuf = device.makeBuffer(bytes: &harmonics,
                                              length: count * 12,
                                              options: .storageModeShared)!
        let packedWorldBuffersHalf = PackedWorldBuffersHalf(packedGaussians: packedBuf, harmonics: harmonicsBuf)

        // Output buffers (half precision: half2=4 bytes, half4=8 bytes, half=2 bytes)
        let meansOut = device.makeBuffer(length: count * 4, options: .storageModeShared)!       // half2
        let conicsOut = device.makeBuffer(length: count * 8, options: .storageModeShared)!      // half4
        let colorsOut = device.makeBuffer(length: count * 6, options: .storageModeShared)!      // packed_half3
        let opacitiesOut = device.makeBuffer(length: count * 2, options: .storageModeShared)!   // half
        let depthsOut = device.makeBuffer(length: count * 2, options: .storageModeShared)!      // half
        let radiiOut = device.makeBuffer(length: count * 4, options: .storageModeShared)!       // float (always float)
        let maskOut = device.makeBuffer(length: count, options: .storageModeShared)!            // uchar

        let gaussianBuffers = GaussianInputBuffers(
            means: meansOut,
            radii: radiiOut,
            mask: maskOut,
            depths: depthsOut,
            conics: conicsOut,
            colors: colorsOut,
            opacities: opacitiesOut
        )

        let commandBuffer = queue.makeCommandBuffer()!

        encoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            packedWorldBuffersHalf: packedWorldBuffersHalf,
            cameraUniforms: camera,
            gaussianBuffers: gaussianBuffers
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read back results (half precision)
        let depthsPtr = depthsOut.contents().bindMemory(to: Float16.self, capacity: count)
        let maskPtr = maskOut.contents().bindMemory(to: UInt8.self, capacity: count)
        let radiiPtr = radiiOut.contents().bindMemory(to: Float.self, capacity: count)

        // Check all gaussians are visible and have reasonable values
        for i in 0..<count {
            XCTAssertEqual(maskPtr[i], 1, "Gaussian \(i) should be visible")
            XCTAssertGreaterThan(Float(depthsPtr[i]), 0, "Gaussian \(i) should have positive depth")
            XCTAssertGreaterThan(radiiPtr[i], 0, "Gaussian \(i) should have positive radius")
        }

        // Check specific depth values (with half precision tolerance ~0.1)
        XCTAssertEqual(Float(depthsPtr[0]), 10.0, accuracy: 0.1, "Gaussian 0 depth")
        XCTAssertEqual(Float(depthsPtr[1]), 10.0, accuracy: 0.1, "Gaussian 1 depth")
        XCTAssertEqual(Float(depthsPtr[2]), 15.0, accuracy: 0.1, "Gaussian 2 depth")
        XCTAssertEqual(Float(depthsPtr[3]), 8.0, accuracy: 0.1, "Gaussian 3 depth")
    }

    /// Test that PackedWorldGaussianHalf converts values correctly
    func testPackedWorldGaussianHalfConversion() throws {
        let position = SIMD3<Float>(1.5, 2.5, 10.0)
        let scale = SIMD3<Float>(0.5, 1.0, 0.25)
        let rotation = SIMD4<Float>(0.5, 0.5, 0.5, 0.5)
        let opacity: Float = 0.75

        let packed = PackedWorldGaussianHalf(
            position: position,
            scale: scale,
            rotation: rotation,
            opacity: opacity
        )

        // Check size
        XCTAssertEqual(MemoryLayout<PackedWorldGaussianHalf>.stride, 24)

        // Verify round-trip conversion (with half precision tolerance)
        let convertedPos = packed.position
        let convertedScale = packed.scale
        let convertedRot = packed.rotation

        XCTAssertEqual(convertedPos.x, position.x, accuracy: 0.01)
        XCTAssertEqual(convertedPos.y, position.y, accuracy: 0.01)
        XCTAssertEqual(convertedPos.z, position.z, accuracy: 0.01)

        XCTAssertEqual(convertedScale.x, scale.x, accuracy: 0.01)
        XCTAssertEqual(convertedScale.y, scale.y, accuracy: 0.01)
        XCTAssertEqual(convertedScale.z, scale.z, accuracy: 0.01)

        XCTAssertEqual(convertedRot.x, rotation.x, accuracy: 0.01)
        XCTAssertEqual(convertedRot.y, rotation.y, accuracy: 0.01)
        XCTAssertEqual(convertedRot.z, rotation.z, accuracy: 0.01)
        XCTAssertEqual(convertedRot.w, rotation.w, accuracy: 0.01)

        XCTAssertEqual(Float(packed.opacity), opacity, accuracy: 0.01)
    }
}

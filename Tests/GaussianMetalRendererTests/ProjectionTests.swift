import XCTest
import Metal
import simd
@testable import GaussianMetalRenderer

/// Tests for ProjectEncoder - verifies projection from world to render data
final class ProjectionTests: XCTestCase {

    var device: MTLDevice!
    var library: MTLLibrary!
    var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        let renderer = GlobalSortRenderer(limits: RendererLimits(maxGaussians: 1_000_000, maxWidth: 1024, maxHeight: 1024))
        self.device = renderer.device
        self.library = renderer.library
        self.queue = renderer.queue
    }

    /// Test packed projection with a single gaussian (float world input)
    func testPackedProjection() throws {
        let encoder = try ProjectEncoder(device: device, library: library)
        let count = 1

        let camera = makeTestCamera(width: 100, height: 100, gaussianCount: count)
        let (packedWorldBuffers, _) = makeTestGaussians(
            device: device,
            positions: [SIMD3(0, 0, 10)],
            scales: [SIMD3(1, 1, 1)],
            rotations: [SIMD4(0, 0, 0, 1)],
            opacities: [1.0],
            useHalf: false
        )

        // Output buffers - packed GaussianRenderData (32 bytes stride with half4 alignment)
        let stride = MemoryLayout<GaussianRenderDataSwift>.stride
        XCTAssertEqual(stride, 32, "GaussianRenderDataSwift should be 32 bytes stride")

        let renderDataOut = device.makeBuffer(length: count * stride, options: .storageModeShared)!
        let radiiOut = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let maskOut = device.makeBuffer(length: count, options: .storageModeShared)!

        let projectionOutput = ProjectionOutput(
            renderData: renderDataOut,
            radii: radiiOut,
            mask: maskOut
        )

        let commandBuffer = queue.makeCommandBuffer()!
        encoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: camera,
            output: projectionOutput,
            useHalfWorld: false
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let renderData = renderDataOut.contents().bindMemory(to: GaussianRenderDataSwift.self, capacity: count)
        let radii = radiiOut.contents().bindMemory(to: Float.self, capacity: 1)
        let mask = maskOut.contents().bindMemory(to: UInt8.self, capacity: 1)

        XCTAssertEqual(mask[0], 1, "Gaussian should be visible")
        XCTAssertGreaterThan(radii[0], 0.0, "Radius should be positive")

        let g = renderData[0]
        XCTAssertEqual(g.mean.x, 49.5, accuracy: 0.5, "Mean X should be ~49.5")
        XCTAssertEqual(g.mean.y, 49.5, accuracy: 0.5, "Mean Y should be ~49.5")
        XCTAssertEqual(Float(g.depth), 10.0, accuracy: 0.1, "Depth should be ~10.0")
    }

    /// Test packed projection with multiple gaussians (float world input)
    func testPackedProjectionMultiple() throws {
        let encoder = try ProjectEncoder(device: device, library: library)
        let count = 4

        let camera = makeTestCamera(width: 100, height: 100, gaussianCount: count)
        let (packedWorldBuffers, _) = makeTestGaussians(
            device: device,
            positions: [
                SIMD3(0, 0, 10),
                SIMD3(5, 5, 10),
                SIMD3(-3, 2, 15),
                SIMD3(1, -1, 8)
            ],
            scales: [
                SIMD3(1, 1, 1),
                SIMD3(0.5, 0.5, 0.5),
                SIMD3(2, 1, 1),
                SIMD3(1, 2, 0.5)
            ],
            rotations: [
                SIMD4(0, 0, 0, 1),
                SIMD4(0.707, 0, 0, 0.707),
                SIMD4(0, 0, 0, 1),
                SIMD4(0.5, 0.5, 0.5, 0.5)
            ],
            opacities: [1.0, 0.8, 0.5, 0.9],
            useHalf: false
        )

        let stride = MemoryLayout<GaussianRenderDataSwift>.stride
        let renderDataOut = device.makeBuffer(length: count * stride, options: .storageModeShared)!
        let radiiOut = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let maskOut = device.makeBuffer(length: count, options: .storageModeShared)!

        let projectionOutput = ProjectionOutput(
            renderData: renderDataOut,
            radii: radiiOut,
            mask: maskOut
        )

        let commandBuffer = queue.makeCommandBuffer()!
        encoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: camera,
            output: projectionOutput,
            useHalfWorld: false
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let renderData = renderDataOut.contents().bindMemory(to: GaussianRenderDataSwift.self, capacity: count)
        let mask = maskOut.contents().bindMemory(to: UInt8.self, capacity: count)
        let radii = radiiOut.contents().bindMemory(to: Float.self, capacity: count)

        let expectedDepths: [Float] = [10.0, 10.0, 15.0, 8.0]
        for i in 0..<count {
            XCTAssertEqual(mask[i], 1, "Gaussian \(i) should be visible")
            XCTAssertGreaterThan(radii[i], 0, "Gaussian \(i) should have positive radius")
            XCTAssertEqual(Float(renderData[i].depth), expectedDepths[i], accuracy: 0.5, "Gaussian \(i) depth")
        }
    }

    /// Test half-precision input (PackedWorldGaussianHalf -> GaussianRenderData)
    func testPackedProjectionHalf() throws {
        let encoder = try ProjectEncoder(device: device, library: library)
        let count = 4

        let camera = makeTestCamera(width: 100, height: 100, gaussianCount: count)
        let (packedWorldBuffers, _) = makeTestGaussians(
            device: device,
            positions: [
                SIMD3(0, 0, 10),
                SIMD3(5, 5, 10),
                SIMD3(-3, 2, 15),
                SIMD3(1, -1, 8)
            ],
            scales: [
                SIMD3(1, 1, 1),
                SIMD3(0.5, 0.5, 0.5),
                SIMD3(2, 1, 1),
                SIMD3(1, 2, 0.5)
            ],
            rotations: [
                SIMD4(0, 0, 0, 1),
                SIMD4(0.707, 0, 0, 0.707),
                SIMD4(0, 0, 0, 1),
                SIMD4(0.5, 0.5, 0.5, 0.5)
            ],
            opacities: [1.0, 0.8, 0.5, 0.9],
            useHalf: true
        )

        // Verify struct sizes
        XCTAssertEqual(MemoryLayout<PackedWorldGaussianHalf>.stride, 24,
                       "PackedWorldGaussianHalf should be 24 bytes")

        let stride = MemoryLayout<GaussianRenderDataSwift>.stride
        let renderDataOut = device.makeBuffer(length: count * stride, options: .storageModeShared)!
        let radiiOut = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let maskOut = device.makeBuffer(length: count, options: .storageModeShared)!

        let projectionOutput = ProjectionOutput(
            renderData: renderDataOut,
            radii: radiiOut,
            mask: maskOut
        )

        let commandBuffer = queue.makeCommandBuffer()!
        encoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            packedWorldBuffers: packedWorldBuffers,
            cameraUniforms: camera,
            output: projectionOutput,
            useHalfWorld: true
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let renderData = renderDataOut.contents().bindMemory(to: GaussianRenderDataSwift.self, capacity: count)
        let mask = maskOut.contents().bindMemory(to: UInt8.self, capacity: count)
        let radii = radiiOut.contents().bindMemory(to: Float.self, capacity: count)

        let expectedDepths: [Float] = [10.0, 10.0, 15.0, 8.0]
        for i in 0..<count {
            XCTAssertEqual(mask[i], 1, "Gaussian \(i) should be visible")
            XCTAssertGreaterThan(radii[i], 0, "Gaussian \(i) should have positive radius")
            XCTAssertEqual(Float(renderData[i].depth), expectedDepths[i], accuracy: 0.5, "Gaussian \(i) depth")
        }
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

        XCTAssertEqual(MemoryLayout<PackedWorldGaussianHalf>.stride, 24)

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

    // MARK: - Helpers

    private func makeTestCamera(width: Float, height: Float, gaussianCount: Int) -> CameraUniformsSwift {
        let viewMatrix = matrix_identity_float4x4
        var projMatrix = matrix_identity_float4x4
        projMatrix.columns.0 = SIMD4(1, 0, 0, 0)
        projMatrix.columns.1 = SIMD4(0, 1, 0, 0)
        projMatrix.columns.2 = SIMD4(0, 0, 1, 1)
        projMatrix.columns.3 = SIMD4(0, 0, 0, 0)

        return CameraUniformsSwift(
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
            gaussianCount: UInt32(gaussianCount),
            padding0: 0,
            padding1: 0
        )
    }

    private func makeTestGaussians(
        device: MTLDevice,
        positions: [SIMD3<Float>],
        scales: [SIMD3<Float>],
        rotations: [SIMD4<Float>],
        opacities: [Float],
        useHalf: Bool
    ) -> (PackedWorldBuffers, [Float]) {
        let count = positions.count
        var harmonics = [Float](repeating: 0.5, count: count * 3)

        if useHalf {
            var packed: [PackedWorldGaussianHalf] = []
            for i in 0..<count {
                packed.append(PackedWorldGaussianHalf(
                    position: positions[i],
                    scale: scales[i],
                    rotation: rotations[i],
                    opacity: opacities[i]
                ))
            }
            let packedBuf = device.makeBuffer(bytes: &packed, length: count * MemoryLayout<PackedWorldGaussianHalf>.stride, options: .storageModeShared)!
            let harmonicsBuf = device.makeBuffer(bytes: &harmonics, length: count * 12, options: .storageModeShared)!
            return (PackedWorldBuffers(packedGaussians: packedBuf, harmonics: harmonicsBuf), harmonics)
        } else {
            var packed: [PackedWorldGaussian] = []
            for i in 0..<count {
                packed.append(PackedWorldGaussian(
                    position: positions[i],
                    scale: scales[i],
                    rotation: rotations[i],
                    opacity: opacities[i]
                ))
            }
            let packedBuf = device.makeBuffer(bytes: &packed, length: count * MemoryLayout<PackedWorldGaussian>.stride, options: .storageModeShared)!
            let harmonicsBuf = device.makeBuffer(bytes: &harmonics, length: count * 12, options: .storageModeShared)!
            return (PackedWorldBuffers(packedGaussians: packedBuf, harmonics: harmonicsBuf), harmonics)
        }
    }
}

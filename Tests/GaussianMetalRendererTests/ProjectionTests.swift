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
    
    func testProjection() throws {
        let encoder = try ProjectEncoder(device: device, library: library)
        
        let count = 1
        
        // 1. Setup Camera
        // Simple perspective projection
        // FOV 90 degrees (tan(45) = 1)
        // Aspect 1.0
        // Near 0.1, Far 100.0
        // Matrix col-major
        // Z-range [0, 1] for Metal
        // P * V * Pos
        
        // Let's manually construct a projection that maps (0,0,10) to (0,0) NDC.
        // And maps (10,10,10) to (1,1) NDC.
        
        // Identity view matrix (Camera at origin, looking down +Z presumably based on kernel logic)
        var viewMatrix = matrix_identity_float4x4
        
        // Projection:
        // x = x / z
        // y = y / z
        // z = z
        // w = z
        var projMatrix = matrix_identity_float4x4
        projMatrix.columns.0 = SIMD4(1, 0, 0, 0)
        projMatrix.columns.1 = SIMD4(0, 1, 0, 0)
        projMatrix.columns.2 = SIMD4(0, 0, 1, 1) // Z goes to Z and W
        projMatrix.columns.3 = SIMD4(0, 0, 0, 0)
        
        let width: Float = 100.0
        let height: Float = 100.0
        
        let camera = CameraUniformsSwift(
            viewMatrix: viewMatrix,
            projectionMatrix: projMatrix,
            cameraCenter: SIMD3(0,0,0),
            pixelFactor: 1.0,
            focalX: width / 2.0, // Should result in 1.0 scale at depth?
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
        
        // 2. Setup Gaussian
        // Position at (0, 0, 10).
        // NDC: x=0/10=0, y=0/10=0.
        // Screen: 
        //   px = ((0 + 1) * 100 - 1) * 0.5 = 99 * 0.5 = 49.5
        //   py = ((0 + 1) * 100 - 1) * 0.5 = 49.5
        
        let position = SIMD3<Float>(0, 0, 10)
        let scale = SIMD3<Float>(1, 1, 1)
        let rotation = SIMD4<Float>(1, 0, 0, 0) // Identity quaternion
        let opacity: Float = 1.0
        
        let posBuf = device.makeBuffer(bytes: [position], length: 12, options: .storageModeShared)!
        let scaleBuf = device.makeBuffer(bytes: [scale], length: 12, options: .storageModeShared)!
        let rotBuf = device.makeBuffer(bytes: [rotation], length: 16, options: .storageModeShared)!
        let opacBuf = device.makeBuffer(bytes: [opacity], length: 4, options: .storageModeShared)!
        let shBuf = device.makeBuffer(length: 12, options: .storageModeShared)! // Dummy SH
        
        let worldBuffers = WorldGaussianBuffers(
            positions: posBuf,
            scales: scaleBuf,
            rotations: rotBuf,
            harmonics: shBuf,
            opacities: opacBuf,
            shComponents: 0
        )
        
        // Outputs
        let meansOut = device.makeBuffer(length: 8, options: .storageModeShared)!
        let radiiOut = device.makeBuffer(length: 4, options: .storageModeShared)!
        let maskOut = device.makeBuffer(length: 1, options: .storageModeShared)!
        let depthsOut = device.makeBuffer(length: 4, options: .storageModeShared)!
        let conicsOut = device.makeBuffer(length: 16, options: .storageModeShared)!
        let colorsOut = device.makeBuffer(length: 12, options: .storageModeShared)!
        let opacitiesOut = device.makeBuffer(length: 4, options: .storageModeShared)!
        
        let projBuffers = ProjectionReadbackBuffers(
            meansOut: meansOut,
            conicsOut: conicsOut,
            colorsOut: colorsOut,
            opacitiesOut: opacitiesOut,
            depthsOut: depthsOut,
            radiiOut: radiiOut,
            maskOut: maskOut
        )
        
        let commandBuffer = queue.makeCommandBuffer()!
        
        encoder.encode(
            commandBuffer: commandBuffer,
            gaussianCount: count,
            worldBuffers: worldBuffers,
            cameraUniforms: camera,
            projectionBuffers: projBuffers
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let means = meansOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)
        let radii = radiiOut.contents().bindMemory(to: Float.self, capacity: 1)
        let mask = maskOut.contents().bindMemory(to: UInt8.self, capacity: 1)
        let depths = depthsOut.contents().bindMemory(to: Float.self, capacity: 1)
        
        XCTAssertEqual(mask[0], 1, "Gaussian should be visible")
        XCTAssertEqual(depths[0], 10.0, accuracy: 0.001)
        
        // Check Means
        // Expected ~49.5
        XCTAssertEqual(means[0].x, 49.5, accuracy: 0.1)
        XCTAssertEqual(means[0].y, 49.5, accuracy: 0.1)
        
        // Check Radii (should be non-zero)
        XCTAssertGreaterThan(radii[0], 0.0)
    }
}

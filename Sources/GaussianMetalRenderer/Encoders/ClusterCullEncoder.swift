import Metal
import simd
import GaussianMetalRendererTypes

/// Simple skip-based cluster culling for Morton-sorted Gaussians.
/// - Precomputes AABBs once at load time
/// - At runtime: just checks frustum vs AABBs, writes visibility buffer
/// - Projection kernel checks visibility and early-returns for culled clusters
/// - NO COMPACTION NEEDED - saves massive memory bandwidth
public final class ClusterCullEncoder {
    private let device: MTLDevice
    private let clusterCullLibrary: MTLLibrary

    // Pipeline states
    private let precomputeAABBsFloatPipeline: MTLComputePipelineState
    private let precomputeAABBsHalfPipeline: MTLComputePipelineState
    private let cullClustersSimplePipeline: MTLComputePipelineState

    // Cached buffers
    private var clusterAABBBuffer: MTLBuffer?
    private var clusterVisibleBuffer: MTLBuffer?  // 1 byte per cluster
    private var paramsBuffer: MTLBuffer?

    // Track state
    private var aabbsPrecomputed: Bool = false
    private var cachedGaussianBuffer: MTLBuffer? = nil
    private var cachedGaussianCount: Int = 0
    private var cachedUseHalf: Bool = false

    /// Initialize from device
    public convenience init(device: MTLDevice) throws {
        let currentFileURL = URL(fileURLWithPath: #filePath)
        let moduleDir = currentFileURL.deletingLastPathComponent().deletingLastPathComponent()
        let metallibURL = moduleDir.appendingPathComponent("ClusterCullShaders.metallib")
        let library = try device.makeLibrary(URL: metallibURL)
        try self.init(device: device, library: library)
    }

    public init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        self.clusterCullLibrary = library

        guard let precomputeFloat = library.makeFunction(name: "precomputeClusterAABBsFloat"),
              let precomputeHalf = library.makeFunction(name: "precomputeClusterAABBsHalf"),
              let cullSimple = library.makeFunction(name: "cullClustersSimple")
        else {
            throw NSError(domain: "ClusterCullEncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to load cluster cull shader functions"])
        }

        precomputeAABBsFloatPipeline = try device.makeComputePipelineState(function: precomputeFloat)
        precomputeAABBsHalfPipeline = try device.makeComputePipelineState(function: precomputeHalf)
        cullClustersSimplePipeline = try device.makeComputePipelineState(function: cullSimple)
    }

    /// Get the cluster visibility buffer for use with projection kernel
    public var visibilityBuffer: MTLBuffer? { clusterVisibleBuffer }

    /// Get cluster size constant
    public var clusterSize: UInt32 { UInt32(CLUSTER_SIZE) }

    /// Ensure buffers are allocated for given gaussian count
    private func ensureBuffers(gaussianCount: Int, useHalf: Bool) {
        let needsRealloc = gaussianCount != cachedGaussianCount || useHalf != cachedUseHalf

        guard needsRealloc else { return }

        cachedGaussianCount = gaussianCount
        cachedUseHalf = useHalf
        aabbsPrecomputed = false

        let clusterSize = Int(CLUSTER_SIZE)
        let clusterCount = (gaussianCount + clusterSize - 1) / clusterSize

        // Cluster AABBs (32 bytes each)
        clusterAABBBuffer = device.makeBuffer(
            length: clusterCount * MemoryLayout<ClusterAABB>.size,
            options: .storageModePrivate
        )

        // Visibility buffer (1 byte per cluster)
        clusterVisibleBuffer = device.makeBuffer(
            length: clusterCount,
            options: .storageModePrivate
        )

        // Params buffer
        paramsBuffer = device.makeBuffer(
            length: MemoryLayout<ClusterCullParams>.size,
            options: .storageModeShared
        )
    }

    /// Precompute AABBs (call once when data is loaded, or automatically on first cull)
    public func precomputeAABBs(
        commandBuffer: MTLCommandBuffer,
        worldGaussians: MTLBuffer,
        gaussianCount: Int,
        useHalfWorld: Bool
    ) {
        guard gaussianCount > 0 else { return }

        let clusterSize = Int(CLUSTER_SIZE)
        let clusterCount = (gaussianCount + clusterSize - 1) / clusterSize

        ensureBuffers(gaussianCount: gaussianCount, useHalf: useHalfWorld)

        guard let clusterAABBBuffer = clusterAABBBuffer,
              let paramsBuffer = paramsBuffer else { return }

        var params = ClusterCullParams(
            totalGaussians: UInt32(gaussianCount),
            clusterCount: UInt32(clusterCount),
            gaussiansPerCluster: UInt32(clusterSize),
            shComponents: 0
        )
        paramsBuffer.contents().copyMemory(from: &params, byteCount: MemoryLayout<ClusterCullParams>.size)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "ClusterCull: Precompute AABBs"
            let pipeline = useHalfWorld ? precomputeAABBsHalfPipeline : precomputeAABBsFloatPipeline
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(worldGaussians, offset: 0, index: 0)
            encoder.setBuffer(clusterAABBBuffer, offset: 0, index: 1)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 2)

            let threadgroupSize = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
            let threadgroups = (clusterCount + threadgroupSize - 1) / threadgroupSize
            encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
            encoder.endEncoding()
        }

        aabbsPrecomputed = true
        cachedGaussianBuffer = worldGaussians
    }

    /// Encode cluster visibility culling. Returns visibility buffer for projection kernel.
    /// Call this before projection, then pass visibilityBuffer to projection kernel.
    public func encodeCull(
        commandBuffer: MTLCommandBuffer,
        worldGaussians: MTLBuffer,
        gaussianCount: Int,
        viewMatrix: simd_float4x4,
        projectionMatrix: simd_float4x4,
        useHalfWorld: Bool
    ) -> MTLBuffer? {
        guard gaussianCount > 0 else { return nil }

        let clusterSize = Int(CLUSTER_SIZE)
        let clusterCount = (gaussianCount + clusterSize - 1) / clusterSize

        ensureBuffers(gaussianCount: gaussianCount, useHalf: useHalfWorld)

        guard let clusterAABBBuffer = clusterAABBBuffer,
              let clusterVisibleBuffer = clusterVisibleBuffer else { return nil }

        // Auto-precompute AABBs if needed
        let needsAABBRecompute = !aabbsPrecomputed || cachedGaussianBuffer !== worldGaussians
        if needsAABBRecompute {
            precomputeAABBs(commandBuffer: commandBuffer, worldGaussians: worldGaussians,
                          gaussianCount: gaussianCount, useHalfWorld: useHalfWorld)
        }

        // Extract frustum planes
        let viewProj = projectionMatrix * viewMatrix
        var frustum = extractFrustumPlanes(from: viewProj)

        guard let frustumBuffer = device.makeBuffer(bytes: &frustum, length: MemoryLayout<FrustumPlanes>.size, options: .storageModeShared) else {
            return nil
        }

        // Simple cull pass - just writes 0 or 1 to visibility buffer
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "ClusterCull: Simple"
            encoder.setComputePipelineState(cullClustersSimplePipeline)
            encoder.setBuffer(clusterAABBBuffer, offset: 0, index: 0)
            encoder.setBuffer(clusterVisibleBuffer, offset: 0, index: 1)
            encoder.setBuffer(frustumBuffer, offset: 0, index: 2)
            var count = UInt32(clusterCount)
            encoder.setBytes(&count, length: 4, index: 3)

            let threadgroupSize = min(cullClustersSimplePipeline.maxTotalThreadsPerThreadgroup, 256)
            let threadgroups = (clusterCount + threadgroupSize - 1) / threadgroupSize
            encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
            encoder.endEncoding()
        }

        return clusterVisibleBuffer
    }

    // MARK: - Private Helpers

    private func extractFrustumPlanes(from viewProj: simd_float4x4) -> FrustumPlanes {
        let m = viewProj
        var planes = FrustumPlanes()

        // Left plane: row3 + row0
        planes.planes.0 = simd_float4(m[0][3] + m[0][0], m[1][3] + m[1][0], m[2][3] + m[2][0], m[3][3] + m[3][0])
        // Right plane: row3 - row0
        planes.planes.1 = simd_float4(m[0][3] - m[0][0], m[1][3] - m[1][0], m[2][3] - m[2][0], m[3][3] - m[3][0])
        // Bottom plane: row3 + row1
        planes.planes.2 = simd_float4(m[0][3] + m[0][1], m[1][3] + m[1][1], m[2][3] + m[2][1], m[3][3] + m[3][1])
        // Top plane: row3 - row1
        planes.planes.3 = simd_float4(m[0][3] - m[0][1], m[1][3] - m[1][1], m[2][3] - m[2][1], m[3][3] - m[3][1])
        // Near plane: row3 + row2
        planes.planes.4 = simd_float4(m[0][3] + m[0][2], m[1][3] + m[1][2], m[2][3] + m[2][2], m[3][3] + m[3][2])
        // Far plane: row3 - row2
        planes.planes.5 = simd_float4(m[0][3] - m[0][2], m[1][3] - m[1][2], m[2][3] - m[2][2], m[3][3] - m[3][2])

        // Normalize planes
        for i in 0..<6 {
            let plane = planes[i]
            let len = simd_length(simd_make_float3(plane.x, plane.y, plane.z))
            if len > 0 {
                planes[i] = plane / len
            }
        }

        return planes
    }
}

// Extension to access tuple elements by index
extension FrustumPlanes {
    subscript(index: Int) -> simd_float4 {
        get {
            switch index {
            case 0: return planes.0
            case 1: return planes.1
            case 2: return planes.2
            case 3: return planes.3
            case 4: return planes.4
            case 5: return planes.5
            default: fatalError("Index out of bounds")
            }
        }
        set {
            switch index {
            case 0: planes.0 = newValue
            case 1: planes.1 = newValue
            case 2: planes.2 = newValue
            case 3: planes.3 = newValue
            case 4: planes.4 = newValue
            case 5: planes.5 = newValue
            default: fatalError("Index out of bounds")
            }
        }
    }
}

// MeshExtractor.swift - Swift interface for mesh extraction from Gaussian Splatting
//
// Implements the Gaussian Opacity Fields (GOF) mesh extraction algorithm
// Reference: Yu et al., "Gaussian Opacity Fields", arXiv:2404.10772 (2024)

import Foundation
import Metal
import simd

/// Configuration for mesh extraction
public struct MeshExtractionConfig {
    /// Grid resolution in each dimension (default: 256)
    public var gridResolution: Int

    /// Isosurface threshold for opacity (default: 0.5)
    /// Points with opacity > isoLevel are considered inside the surface
    public var isoLevel: Float

    /// Margin to add around the Gaussian bounding box (default: 0.1)
    public var margin: Float

    /// Cutoff distance in sigma units (default: 3.0)
    /// Gaussians beyond this distance don't contribute
    public var sigmaCutoff: Float

    /// Minimum opacity to consider (default: 1/255)
    public var opacityCutoff: Float

    public init(
        gridResolution: Int = 256,
        isoLevel: Float = 0.5,
        margin: Float = 0.1,
        sigmaCutoff: Float = 3.0,
        opacityCutoff: Float = 1.0 / 255.0
    ) {
        self.gridResolution = gridResolution
        self.isoLevel = isoLevel
        self.margin = margin
        self.sigmaCutoff = sigmaCutoff
        self.opacityCutoff = opacityCutoff
    }
}

/// Result of mesh extraction
public struct ExtractedMesh {
    /// Vertex positions in world space
    public var positions: [SIMD3<Float>]

    /// Vertex normals (per-face normals)
    public var normals: [SIMD3<Float>]

    /// Vertex colors (if available)
    public var colors: [SIMD3<Float>]

    /// Triangle indices (3 indices per triangle)
    public var indices: [UInt32]

    /// Number of triangles
    public var triangleCount: Int { indices.count / 3 }

    /// Number of vertices
    public var vertexCount: Int { positions.count }

    public init() {
        positions = []
        normals = []
        colors = []
        indices = []
    }
}

/// Error types for mesh extraction
public enum MeshExtractionError: Error {
    case deviceNotSupported(String)
    case pipelineCreationFailed(String)
    case bufferAllocationFailed(String)
    case computeError(String)
    case overflow(String)
}

/// Extracts triangle meshes from 3D Gaussian representations
public class MeshExtractor {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary

    // Compute pipelines
    private let initBoundsPipeline: MTLComputePipelineState
    private let computeBoundsPipeline: MTLComputePipelineState
    private let finalizeBoundsPipeline: MTLComputePipelineState
    private let computeOpacityPipeline: MTLComputePipelineState
    private let computeOpacityHalfPipeline: MTLComputePipelineState
    private let initMeshHeaderPipeline: MTLComputePipelineState
    private let marchingCubesCountPipeline: MTLComputePipelineState
    private let meshBlockReducePipeline: MTLComputePipelineState
    private let meshBlockScanPipeline: MTLComputePipelineState
    private let meshSingleBlockScanPipeline: MTLComputePipelineState
    private let finalizeMeshHeaderPipeline: MTLComputePipelineState
    private let marchingCubesGeneratePipeline: MTLComputePipelineState

    // Reusable buffers
    private var minBoundsBuffer: MTLBuffer?
    private var maxBoundsBuffer: MTLBuffer?
    private var boundsResultBuffer: MTLBuffer?
    private var opacityGridBuffer: MTLBuffer?
    private var triangleCountsBuffer: MTLBuffer?
    private var triangleOffsetsBuffer: MTLBuffer?
    private var blockSumsBuffer: MTLBuffer?
    private var vertexBuffer: MTLBuffer?
    private var indexBuffer: MTLBuffer?
    private var meshHeaderBuffer: MTLBuffer?

    public init(device: MTLDevice) throws {
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            throw MeshExtractionError.deviceNotSupported("Failed to create command queue")
        }
        self.commandQueue = commandQueue

        // Load library from default bundle
        guard let library = device.makeDefaultLibrary() else {
            throw MeshExtractionError.pipelineCreationFailed("Failed to load default Metal library")
        }
        self.library = library

        // Create compute pipelines
        self.initBoundsPipeline = try Self.makePipeline(device: device, library: library, name: "initBoundsKernel")
        self.computeBoundsPipeline = try Self.makePipeline(device: device, library: library, name: "computeGaussianBoundsKernel")
        self.finalizeBoundsPipeline = try Self.makePipeline(device: device, library: library, name: "finalizeBoundsKernel")
        self.computeOpacityPipeline = try Self.makePipeline(device: device, library: library, name: "computeOpacityFieldKernel")
        self.computeOpacityHalfPipeline = try Self.makePipeline(device: device, library: library, name: "computeOpacityFieldKernelHalf")
        self.initMeshHeaderPipeline = try Self.makePipeline(device: device, library: library, name: "initMeshHeaderKernel")
        self.marchingCubesCountPipeline = try Self.makePipeline(device: device, library: library, name: "marchingCubesCountKernel")
        self.meshBlockReducePipeline = try Self.makePipeline(device: device, library: library, name: "meshBlockReduceKernel")
        self.meshBlockScanPipeline = try Self.makePipeline(device: device, library: library, name: "meshBlockScanKernel")
        self.meshSingleBlockScanPipeline = try Self.makePipeline(device: device, library: library, name: "meshSingleBlockScanKernel")
        self.finalizeMeshHeaderPipeline = try Self.makePipeline(device: device, library: library, name: "finalizeMeshHeaderKernel")
        self.marchingCubesGeneratePipeline = try Self.makePipeline(device: device, library: library, name: "marchingCubesGenerateKernel")
    }

    private static func makePipeline(device: MTLDevice, library: MTLLibrary, name: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw MeshExtractionError.pipelineCreationFailed("Function '\(name)' not found")
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            throw MeshExtractionError.pipelineCreationFailed("Failed to create pipeline for '\(name)': \(error)")
        }
    }

    /// Extract a mesh from Gaussian data
    /// - Parameters:
    ///   - gaussians: Buffer containing PackedWorldGaussian or PackedWorldGaussianHalf structures
    ///   - gaussianCount: Number of Gaussians in the buffer
    ///   - useHalfPrecision: Whether the input uses half precision (PackedWorldGaussianHalf)
    ///   - config: Extraction configuration
    /// - Returns: Extracted mesh with vertices, normals, and indices
    public func extractMesh(
        gaussians: MTLBuffer,
        gaussianCount: Int,
        useHalfPrecision: Bool = false,
        config: MeshExtractionConfig = MeshExtractionConfig()
    ) throws -> ExtractedMesh {
        guard gaussianCount > 0 else {
            return ExtractedMesh()
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MeshExtractionError.computeError("Failed to create command buffer")
        }

        // Step 1: Compute bounds
        let bounds = try computeBounds(
            commandBuffer: commandBuffer,
            gaussians: gaussians,
            gaussianCount: gaussianCount,
            margin: config.margin
        )

        // Commit and wait for bounds computation
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read bounds result
        guard let boundsResult = boundsResultBuffer else {
            throw MeshExtractionError.bufferAllocationFailed("Bounds result buffer not allocated")
        }

        let boundsPtr = boundsResult.contents().bindMemory(to: BoundsResult.self, capacity: 1)
        let minBound = SIMD3<Float>(boundsPtr.pointee.minBound.x, boundsPtr.pointee.minBound.y, boundsPtr.pointee.minBound.z)
        let maxBound = SIMD3<Float>(boundsPtr.pointee.maxBound.x, boundsPtr.pointee.maxBound.y, boundsPtr.pointee.maxBound.z)

        // Compute grid dimensions
        let gridExtent = maxBound - minBound
        let maxExtent = max(gridExtent.x, max(gridExtent.y, gridExtent.z))
        let voxelSize = maxExtent / Float(config.gridResolution)

        let gridDims = SIMD3<UInt32>(
            UInt32(ceil(gridExtent.x / voxelSize)) + 1,
            UInt32(ceil(gridExtent.y / voxelSize)) + 1,
            UInt32(ceil(gridExtent.z / voxelSize)) + 1
        )

        // Create params
        var params = MeshExtractionParams()
        params.gridMin = simd_float3(minBound.x, minBound.y, minBound.z)
        params.gridMax = simd_float3(maxBound.x, maxBound.y, maxBound.z)
        params.voxelSize = voxelSize
        params.isoLevel = config.isoLevel
        params.gridDimX = gridDims.x
        params.gridDimY = gridDims.y
        params.gridDimZ = gridDims.z
        params.gaussianCount = UInt32(gaussianCount)
        params.opacityCutoff = config.opacityCutoff
        params.sigmaCutoff = config.sigmaCutoff

        // Step 2: Allocate opacity grid
        let gridSize = Int(gridDims.x) * Int(gridDims.y) * Int(gridDims.z)
        try allocateOpacityGrid(size: gridSize)

        // Step 3: Compute opacity field
        guard let commandBuffer2 = commandQueue.makeCommandBuffer() else {
            throw MeshExtractionError.computeError("Failed to create command buffer")
        }

        try computeOpacityField(
            commandBuffer: commandBuffer2,
            gaussians: gaussians,
            useHalfPrecision: useHalfPrecision,
            params: params
        )

        commandBuffer2.commit()
        commandBuffer2.waitUntilCompleted()

        // Step 4: Marching cubes - count triangles
        let cellCount = Int(gridDims.x - 1) * Int(gridDims.y - 1) * Int(gridDims.z - 1)
        try allocateTriangleCounts(cellCount: cellCount)

        guard let commandBuffer3 = commandQueue.makeCommandBuffer() else {
            throw MeshExtractionError.computeError("Failed to create command buffer")
        }

        try countTriangles(commandBuffer: commandBuffer3, params: params)

        commandBuffer3.commit()
        commandBuffer3.waitUntilCompleted()

        // Step 5: Prefix sum to get triangle offsets
        guard let commandBuffer4 = commandQueue.makeCommandBuffer() else {
            throw MeshExtractionError.computeError("Failed to create command buffer")
        }

        let totalTriangles = try prefixSumTriangleCounts(
            commandBuffer: commandBuffer4,
            cellCount: cellCount
        )

        commandBuffer4.commit()
        commandBuffer4.waitUntilCompleted()

        guard totalTriangles > 0 else {
            return ExtractedMesh()
        }

        // Step 6: Allocate mesh buffers
        let maxTriangles = min(totalTriangles, 10_000_000)  // Cap at 10M triangles
        let maxVertices = maxTriangles * 3
        try allocateMeshBuffers(maxVertices: maxVertices, maxTriangles: maxTriangles)

        // Step 7: Generate triangles
        guard let commandBuffer5 = commandQueue.makeCommandBuffer() else {
            throw MeshExtractionError.computeError("Failed to create command buffer")
        }

        try generateTriangles(
            commandBuffer: commandBuffer5,
            params: params,
            maxVertices: maxVertices,
            maxTriangles: maxTriangles
        )

        commandBuffer5.commit()
        commandBuffer5.waitUntilCompleted()

        // Step 8: Read results
        return try readMeshResults(maxVertices: maxVertices, maxTriangles: maxTriangles)
    }

    // MARK: - Private Methods

    private func computeBounds(
        commandBuffer: MTLCommandBuffer,
        gaussians: MTLBuffer,
        gaussianCount: Int,
        margin: Float
    ) throws {
        // Allocate bounds buffers if needed
        if minBoundsBuffer == nil {
            minBoundsBuffer = device.makeBuffer(length: 3 * MemoryLayout<Int32>.stride, options: .storageModeShared)
            maxBoundsBuffer = device.makeBuffer(length: 3 * MemoryLayout<Int32>.stride, options: .storageModeShared)
            boundsResultBuffer = device.makeBuffer(length: MemoryLayout<BoundsResult>.stride, options: .storageModeShared)
        }

        guard let minBounds = minBoundsBuffer,
              let maxBounds = maxBoundsBuffer,
              let boundsResult = boundsResultBuffer else {
            throw MeshExtractionError.bufferAllocationFailed("Failed to allocate bounds buffers")
        }

        // Initialize bounds
        guard let encoder1 = commandBuffer.makeComputeCommandEncoder() else {
            throw MeshExtractionError.computeError("Failed to create compute encoder")
        }
        encoder1.setComputePipelineState(initBoundsPipeline)
        encoder1.setBuffer(minBounds, offset: 0, index: 0)
        encoder1.setBuffer(maxBounds, offset: 0, index: 1)
        encoder1.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder1.endEncoding()

        // Compute bounds from Gaussians
        guard let encoder2 = commandBuffer.makeComputeCommandEncoder() else {
            throw MeshExtractionError.computeError("Failed to create compute encoder")
        }
        encoder2.setComputePipelineState(computeBoundsPipeline)
        encoder2.setBuffer(gaussians, offset: 0, index: 0)
        encoder2.setBuffer(minBounds, offset: 0, index: 1)
        encoder2.setBuffer(maxBounds, offset: 0, index: 2)
        var count = UInt32(gaussianCount)
        encoder2.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)

        let threadgroupSize = min(computeBoundsPipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadgroups = (gaussianCount + threadgroupSize - 1) / threadgroupSize
        encoder2.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
        encoder2.endEncoding()

        // Finalize bounds with margin
        guard let encoder3 = commandBuffer.makeComputeCommandEncoder() else {
            throw MeshExtractionError.computeError("Failed to create compute encoder")
        }
        encoder3.setComputePipelineState(finalizeBoundsPipeline)
        encoder3.setBuffer(minBounds, offset: 0, index: 0)
        encoder3.setBuffer(maxBounds, offset: 0, index: 1)
        encoder3.setBuffer(boundsResult, offset: 0, index: 2)
        var marginValue = margin
        encoder3.setBytes(&marginValue, length: MemoryLayout<Float>.stride, index: 3)
        encoder3.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder3.endEncoding()
    }

    private func allocateOpacityGrid(size: Int) throws {
        let requiredSize = size * MemoryLayout<Float>.stride
        if opacityGridBuffer == nil || opacityGridBuffer!.length < requiredSize {
            opacityGridBuffer = device.makeBuffer(length: requiredSize, options: .storageModeShared)
        }
        guard opacityGridBuffer != nil else {
            throw MeshExtractionError.bufferAllocationFailed("Failed to allocate opacity grid (\(requiredSize) bytes)")
        }
    }

    private func computeOpacityField(
        commandBuffer: MTLCommandBuffer,
        gaussians: MTLBuffer,
        useHalfPrecision: Bool,
        params: MeshExtractionParams
    ) throws {
        guard let opacityGrid = opacityGridBuffer else {
            throw MeshExtractionError.bufferAllocationFailed("Opacity grid not allocated")
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MeshExtractionError.computeError("Failed to create compute encoder")
        }

        let pipeline = useHalfPrecision ? computeOpacityHalfPipeline : computeOpacityPipeline
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gaussians, offset: 0, index: 0)
        encoder.setBuffer(opacityGrid, offset: 0, index: 1)

        var paramsCopy = params
        encoder.setBytes(&paramsCopy, length: MemoryLayout<MeshExtractionParams>.stride, index: 2)

        let gridWidth = Int(params.gridDimX)
        let gridHeight = Int(params.gridDimY)
        let gridDepth = Int(params.gridDimZ)

        encoder.dispatchThreads(MTLSize(width: gridWidth, height: gridHeight, depth: gridDepth),
                                 threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 4))
        encoder.endEncoding()
    }

    private func allocateTriangleCounts(cellCount: Int) throws {
        let countsSize = (cellCount + 1) * MemoryLayout<UInt32>.stride  // +1 for total
        if triangleCountsBuffer == nil || triangleCountsBuffer!.length < countsSize {
            triangleCountsBuffer = device.makeBuffer(length: countsSize, options: .storageModeShared)
            triangleOffsetsBuffer = device.makeBuffer(length: countsSize, options: .storageModeShared)
        }

        let blockCount = (cellCount + 255) / 256
        let blockSumsSize = (blockCount + 1) * MemoryLayout<UInt32>.stride
        if blockSumsBuffer == nil || blockSumsBuffer!.length < blockSumsSize {
            blockSumsBuffer = device.makeBuffer(length: blockSumsSize, options: .storageModeShared)
        }

        guard triangleCountsBuffer != nil, triangleOffsetsBuffer != nil, blockSumsBuffer != nil else {
            throw MeshExtractionError.bufferAllocationFailed("Failed to allocate triangle count buffers")
        }
    }

    private func countTriangles(
        commandBuffer: MTLCommandBuffer,
        params: MeshExtractionParams
    ) throws {
        guard let opacityGrid = opacityGridBuffer,
              let triangleCounts = triangleCountsBuffer else {
            throw MeshExtractionError.bufferAllocationFailed("Required buffers not allocated")
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MeshExtractionError.computeError("Failed to create compute encoder")
        }

        encoder.setComputePipelineState(marchingCubesCountPipeline)
        encoder.setBuffer(opacityGrid, offset: 0, index: 0)
        encoder.setBuffer(triangleCounts, offset: 0, index: 1)

        var paramsCopy = params
        encoder.setBytes(&paramsCopy, length: MemoryLayout<MeshExtractionParams>.stride, index: 2)

        let cellWidth = Int(params.gridDimX - 1)
        let cellHeight = Int(params.gridDimY - 1)
        let cellDepth = Int(params.gridDimZ - 1)

        encoder.dispatchThreads(MTLSize(width: cellWidth, height: cellHeight, depth: cellDepth),
                                 threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 4))
        encoder.endEncoding()
    }

    private func prefixSumTriangleCounts(
        commandBuffer: MTLCommandBuffer,
        cellCount: Int
    ) throws -> Int {
        guard let triangleCounts = triangleCountsBuffer,
              let triangleOffsets = triangleOffsetsBuffer,
              let blockSums = blockSumsBuffer else {
            throw MeshExtractionError.bufferAllocationFailed("Required buffers not allocated")
        }

        let blockSize = 256
        let blockCount = (cellCount + blockSize - 1) / blockSize

        // Block reduce
        guard let encoder1 = commandBuffer.makeComputeCommandEncoder() else {
            throw MeshExtractionError.computeError("Failed to create compute encoder")
        }
        encoder1.setComputePipelineState(meshBlockReducePipeline)
        encoder1.setBuffer(triangleCounts, offset: 0, index: 0)
        encoder1.setBuffer(blockSums, offset: 0, index: 1)
        var count = UInt32(cellCount)
        encoder1.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder1.dispatchThreadgroups(MTLSize(width: blockCount, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: blockSize, height: 1, depth: 1))
        encoder1.endEncoding()

        // Single block scan of block sums
        guard let encoder2 = commandBuffer.makeComputeCommandEncoder() else {
            throw MeshExtractionError.computeError("Failed to create compute encoder")
        }
        encoder2.setComputePipelineState(meshSingleBlockScanPipeline)
        encoder2.setBuffer(blockSums, offset: 0, index: 0)
        var blockCountU32 = UInt32(blockCount)
        encoder2.setBytes(&blockCountU32, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder2.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: blockSize, height: 1, depth: 1))
        encoder2.endEncoding()

        // Block scan to produce offsets
        guard let encoder3 = commandBuffer.makeComputeCommandEncoder() else {
            throw MeshExtractionError.computeError("Failed to create compute encoder")
        }
        encoder3.setComputePipelineState(meshBlockScanPipeline)
        encoder3.setBuffer(triangleCounts, offset: 0, index: 0)
        encoder3.setBuffer(triangleOffsets, offset: 0, index: 1)
        encoder3.setBuffer(blockSums, offset: 0, index: 2)
        encoder3.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder3.dispatchThreadgroups(MTLSize(width: blockCount, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: blockSize, height: 1, depth: 1))
        encoder3.endEncoding()

        // Read total from blockSums[blockCount] after completion
        commandBuffer.addCompletedHandler { [weak self] _ in
            // Total will be read after completion
        }

        // Return 0 for now, actual reading happens in extractMesh
        // We need to read from blockSums[blockCount] after this command buffer completes
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let blockSumsPtr = blockSums.contents().bindMemory(to: UInt32.self, capacity: blockCount + 1)
        return Int(blockSumsPtr[blockCount])
    }

    private func allocateMeshBuffers(maxVertices: Int, maxTriangles: Int) throws {
        let vertexSize = maxVertices * MemoryLayout<MeshVertex>.stride
        let indexSize = maxTriangles * 3 * MemoryLayout<UInt32>.stride
        let headerSize = MemoryLayout<MeshExtractionHeader>.stride

        if vertexBuffer == nil || vertexBuffer!.length < vertexSize {
            vertexBuffer = device.makeBuffer(length: vertexSize, options: .storageModeShared)
        }
        if indexBuffer == nil || indexBuffer!.length < indexSize {
            indexBuffer = device.makeBuffer(length: indexSize, options: .storageModeShared)
        }
        if meshHeaderBuffer == nil {
            meshHeaderBuffer = device.makeBuffer(length: headerSize, options: .storageModeShared)
        }

        guard vertexBuffer != nil, indexBuffer != nil, meshHeaderBuffer != nil else {
            throw MeshExtractionError.bufferAllocationFailed("Failed to allocate mesh buffers")
        }

        // Initialize header
        let headerPtr = meshHeaderBuffer!.contents().bindMemory(to: MeshExtractionHeader.self, capacity: 1)
        headerPtr.pointee.totalVertices = 0
        headerPtr.pointee.totalTriangles = 0
        headerPtr.pointee.activeCells = 0
        headerPtr.pointee.overflow = 0
    }

    private func generateTriangles(
        commandBuffer: MTLCommandBuffer,
        params: MeshExtractionParams,
        maxVertices: Int,
        maxTriangles: Int
    ) throws {
        guard let opacityGrid = opacityGridBuffer,
              let triangleOffsets = triangleOffsetsBuffer,
              let vertices = vertexBuffer,
              let indices = indexBuffer,
              let header = meshHeaderBuffer else {
            throw MeshExtractionError.bufferAllocationFailed("Required buffers not allocated")
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MeshExtractionError.computeError("Failed to create compute encoder")
        }

        encoder.setComputePipelineState(marchingCubesGeneratePipeline)
        encoder.setBuffer(opacityGrid, offset: 0, index: 0)
        encoder.setBuffer(triangleOffsets, offset: 0, index: 1)
        encoder.setBuffer(vertices, offset: 0, index: 2)
        encoder.setBuffer(indices, offset: 0, index: 3)
        encoder.setBuffer(header, offset: 0, index: 4)

        var paramsCopy = params
        encoder.setBytes(&paramsCopy, length: MemoryLayout<MeshExtractionParams>.stride, index: 5)

        var maxVerts = UInt32(maxVertices)
        var maxTris = UInt32(maxTriangles)
        encoder.setBytes(&maxVerts, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&maxTris, length: MemoryLayout<UInt32>.stride, index: 7)

        let cellWidth = Int(params.gridDimX - 1)
        let cellHeight = Int(params.gridDimY - 1)
        let cellDepth = Int(params.gridDimZ - 1)

        encoder.dispatchThreads(MTLSize(width: cellWidth, height: cellHeight, depth: cellDepth),
                                 threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 4))
        encoder.endEncoding()
    }

    private func readMeshResults(maxVertices: Int, maxTriangles: Int) throws -> ExtractedMesh {
        guard let vertices = vertexBuffer,
              let indices = indexBuffer,
              let header = meshHeaderBuffer else {
            throw MeshExtractionError.bufferAllocationFailed("Mesh buffers not allocated")
        }

        // Read header
        let headerPtr = header.contents().bindMemory(to: MeshExtractionHeader.self, capacity: 1)
        let totalVertices = Int(headerPtr.pointee.totalVertices)
        let totalTriangles = Int(headerPtr.pointee.totalTriangles)

        if headerPtr.pointee.overflow != 0 {
            throw MeshExtractionError.overflow("Mesh buffer overflow - too many triangles")
        }

        guard totalVertices > 0 && totalTriangles > 0 else {
            return ExtractedMesh()
        }

        // Read vertices
        let vertexPtr = vertices.contents().bindMemory(to: MeshVertex.self, capacity: totalVertices)
        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var colors: [SIMD3<Float>] = []

        positions.reserveCapacity(totalVertices)
        normals.reserveCapacity(totalVertices)
        colors.reserveCapacity(totalVertices)

        for i in 0..<totalVertices {
            let v = vertexPtr[i]
            positions.append(SIMD3<Float>(v.position.x, v.position.y, v.position.z))
            normals.append(SIMD3<Float>(v.normal.x, v.normal.y, v.normal.z))
            colors.append(SIMD3<Float>(v.color.x, v.color.y, v.color.z))
        }

        // Read indices
        let indexPtr = indices.contents().bindMemory(to: UInt32.self, capacity: totalTriangles * 3)
        var indexArray: [UInt32] = []
        indexArray.reserveCapacity(totalTriangles * 3)

        for i in 0..<(totalTriangles * 3) {
            indexArray.append(indexPtr[i])
        }

        var mesh = ExtractedMesh()
        mesh.positions = positions
        mesh.normals = normals
        mesh.colors = colors
        mesh.indices = indexArray

        return mesh
    }
}

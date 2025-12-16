// MeshExtractionShaders.metal - Compute shaders for mesh extraction from Gaussian Opacity Fields
//
// This implements the GOF (Gaussian Opacity Fields) mesh extraction algorithm:
// 1. Compute opacity field from 3D Gaussians at each grid point
// 2. Convert opacity to SDF: SDF = 0.5 - opacity
// 3. Use marching cubes to extract isosurface at SDF = 0
//
// Reference: Yu et al., "Gaussian Opacity Fields", arXiv:2404.10772 (2024)

#include <metal_stdlib>
#include "BridgingTypes.h"
#include "GaussianShared.h"
#include "MarchingCubesTables.h"
using namespace metal;

// ============================================================================
// STAGE 1: Compute Bounds
// Computes the axis-aligned bounding box of all Gaussians
// ============================================================================

kernel void computeGaussianBoundsKernel(
    const device PackedWorldGaussian* gaussians [[buffer(0)]],
    device atomic_int* minBoundsAtomic [[buffer(1)]],  // 3 ints for x,y,z (scaled by 1e6)
    device atomic_int* maxBoundsAtomic [[buffer(2)]],  // 3 ints for x,y,z (scaled by 1e6)
    constant uint& gaussianCount [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= gaussianCount) return;

    PackedWorldGaussian g = gaussians[gid];
    float3 pos = float3(g.px, g.py, g.pz);
    float3 scale = float3(g.sx, g.sy, g.sz);

    // Extend bounds by 3 sigma (99.7% of Gaussian mass)
    float maxScale = max(scale.x, max(scale.y, scale.z));
    float3 extent = float3(maxScale * 3.0f);

    float3 minP = pos - extent;
    float3 maxP = pos + extent;

    // Scale to integer for atomic operations (micron precision)
    int3 minI = int3(minP * 1e6f);
    int3 maxI = int3(maxP * 1e6f);

    // Atomic min/max
    atomic_fetch_min_explicit(&minBoundsAtomic[0], minI.x, memory_order_relaxed);
    atomic_fetch_min_explicit(&minBoundsAtomic[1], minI.y, memory_order_relaxed);
    atomic_fetch_min_explicit(&minBoundsAtomic[2], minI.z, memory_order_relaxed);
    atomic_fetch_max_explicit(&maxBoundsAtomic[0], maxI.x, memory_order_relaxed);
    atomic_fetch_max_explicit(&maxBoundsAtomic[1], maxI.y, memory_order_relaxed);
    atomic_fetch_max_explicit(&maxBoundsAtomic[2], maxI.z, memory_order_relaxed);
}

kernel void finalizeBoundsKernel(
    const device int* minBoundsAtomic [[buffer(0)]],
    const device int* maxBoundsAtomic [[buffer(1)]],
    device BoundsResult* result [[buffer(2)]],
    constant float& margin [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    float3 minB = float3(minBoundsAtomic[0], minBoundsAtomic[1], minBoundsAtomic[2]) * 1e-6f;
    float3 maxB = float3(maxBoundsAtomic[0], maxBoundsAtomic[1], maxBoundsAtomic[2]) * 1e-6f;

    // Add margin
    result->minBound = minB - float3(margin);
    result->maxBound = maxB + float3(margin);
}

// ============================================================================
// STAGE 2: Compute Opacity Field
// For each grid point, accumulate opacity from all nearby Gaussians
// ============================================================================

// Compute the inverse covariance matrix (precision matrix) for a Gaussian
inline float3x3 computeInverseCovariance(float3 scale, float4 quat) {
    // Σ⁻¹ = R · S⁻² · Rᵀ
    float4 q = normalizeQuaternion(quat);
    float3x3 R = quaternionToMatrix(q);

    // S⁻² = diag(1/sx², 1/sy², 1/sz²)
    float3 invScale2 = 1.0f / max(scale * scale, float3(1e-12f));

    // Compute R · S⁻² · Rᵀ
    // For efficiency, compute column by column
    float3 col0 = R[0] * invScale2.x;
    float3 col1 = R[1] * invScale2.y;
    float3 col2 = R[2] * invScale2.z;

    // Result = [col0 col1 col2] · Rᵀ
    float3x3 result;
    result[0] = col0 * R[0].x + col1 * R[1].x + col2 * R[2].x;
    result[1] = col0 * R[0].y + col1 * R[1].y + col2 * R[2].y;
    result[2] = col0 * R[0].z + col1 * R[1].z + col2 * R[2].z;

    return result;
}

// Compute Mahalanobis distance squared: (p - μ)ᵀ · Σ⁻¹ · (p - μ)
inline float mahalanobisDistanceSquared(float3 delta, float3x3 invCov) {
    float3 transformed = invCov * delta;
    return dot(delta, transformed);
}

// Compute opacity at a single 3D point from all Gaussians
// Uses brute force evaluation - every Gaussian is tested
kernel void computeOpacityFieldKernel(
    const device PackedWorldGaussian* gaussians [[buffer(0)]],
    device float* opacityGrid [[buffer(1)]],
    constant MeshExtractionParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Check bounds
    if (gid.x >= params.gridDimX || gid.y >= params.gridDimY || gid.z >= params.gridDimZ) {
        return;
    }

    // Compute world position of this grid point
    float3 gridPos = float3(gid);
    float3 worldPos = params.gridMin + gridPos * params.voxelSize;

    // Accumulate transmittance (1 - alpha) from all Gaussians
    // T = ∏ᵢ (1 - αᵢ(p))
    float transmittance = 1.0f;

    float sigmaCutoff2 = params.sigmaCutoff * params.sigmaCutoff;

    for (uint i = 0; i < params.gaussianCount; i++) {
        PackedWorldGaussian g = gaussians[i];

        float3 pos = float3(g.px, g.py, g.pz);
        float3 scale = float3(g.sx, g.sy, g.sz);
        float opacity = g.opacity;

        // Skip low-opacity Gaussians
        if (opacity < params.opacityCutoff) continue;

        // Quick distance check using maximum scale
        float3 delta = worldPos - pos;
        float maxScale = max(scale.x, max(scale.y, scale.z));
        float roughDist2 = dot(delta, delta);
        float cutoffDist = maxScale * params.sigmaCutoff;
        if (roughDist2 > cutoffDist * cutoffDist) continue;

        // Compute inverse covariance
        float4 quat = getRotation(g);
        float3x3 invCov = computeInverseCovariance(scale, quat);

        // Compute Mahalanobis distance squared
        float d2 = mahalanobisDistanceSquared(delta, invCov);

        // Skip if beyond sigma cutoff (Gaussian contribution negligible)
        if (d2 > sigmaCutoff2) continue;

        // Compute Gaussian contribution: α_i(p) = α_i · exp(-0.5 · d²)
        float gaussianValue = exp(-0.5f * d2);
        float alpha = min(opacity * gaussianValue, 0.99f);

        // Accumulate: T *= (1 - α)
        transmittance *= (1.0f - alpha);

        // Early exit if transmittance is negligible
        if (transmittance < 1e-6f) {
            transmittance = 0.0f;
            break;
        }
    }

    // Final opacity: α(p) = 1 - T
    float finalOpacity = 1.0f - transmittance;

    // Write to grid (linearized 3D index)
    uint gridIndex = gid.x + gid.y * params.gridDimX + gid.z * params.gridDimX * params.gridDimY;
    opacityGrid[gridIndex] = finalOpacity;
}

// Half-precision version for bandwidth optimization
kernel void computeOpacityFieldKernelHalf(
    const device PackedWorldGaussianHalf* gaussians [[buffer(0)]],
    device float* opacityGrid [[buffer(1)]],
    constant MeshExtractionParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Check bounds
    if (gid.x >= params.gridDimX || gid.y >= params.gridDimY || gid.z >= params.gridDimZ) {
        return;
    }

    // Compute world position of this grid point
    float3 gridPos = float3(gid);
    float3 worldPos = params.gridMin + gridPos * params.voxelSize;

    // Accumulate transmittance
    float transmittance = 1.0f;
    float sigmaCutoff2 = params.sigmaCutoff * params.sigmaCutoff;

    for (uint i = 0; i < params.gaussianCount; i++) {
        PackedWorldGaussianHalf g = gaussians[i];

        float3 pos = float3(g.px, g.py, g.pz);
        float3 scale = float3(float(g.sx), float(g.sy), float(g.sz));
        float opacity = float(g.opacity);

        if (opacity < params.opacityCutoff) continue;

        float3 delta = worldPos - pos;
        float maxScale = max(scale.x, max(scale.y, scale.z));
        float roughDist2 = dot(delta, delta);
        float cutoffDist = maxScale * params.sigmaCutoff;
        if (roughDist2 > cutoffDist * cutoffDist) continue;

        float4 quat = getRotation(g);
        float3x3 invCov = computeInverseCovariance(scale, quat);
        float d2 = mahalanobisDistanceSquared(delta, invCov);

        if (d2 > sigmaCutoff2) continue;

        float gaussianValue = exp(-0.5f * d2);
        float alpha = min(opacity * gaussianValue, 0.99f);
        transmittance *= (1.0f - alpha);

        if (transmittance < 1e-6f) {
            transmittance = 0.0f;
            break;
        }
    }

    float finalOpacity = 1.0f - transmittance;
    uint gridIndex = gid.x + gid.y * params.gridDimX + gid.z * params.gridDimX * params.gridDimY;
    opacityGrid[gridIndex] = finalOpacity;
}

// ============================================================================
// STAGE 3: Marching Cubes - Count Triangles
// Count the number of triangles each cell will produce
// ============================================================================

// Get opacity at a grid position (with bounds checking)
inline float sampleOpacity(
    const device float* opacityGrid,
    uint3 pos,
    uint3 dims
) {
    if (pos.x >= dims.x || pos.y >= dims.y || pos.z >= dims.z) {
        return 0.0f;
    }
    uint index = pos.x + pos.y * dims.x + pos.z * dims.x * dims.y;
    return opacityGrid[index];
}

// Convert opacity to SDF: SDF = isoLevel - opacity
// Points with opacity > isoLevel are "inside" (SDF < 0)
inline float opacityToSDF(float opacity, float isoLevel) {
    return isoLevel - opacity;
}

kernel void marchingCubesCountKernel(
    const device float* opacityGrid [[buffer(0)]],
    device uint* triangleCounts [[buffer(1)]],
    constant MeshExtractionParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Each cell is defined by 8 corner vertices
    // Cell (x,y,z) uses vertices at (x,y,z), (x+1,y,z), etc.
    // So we process cells 0..dim-2 in each dimension
    if (gid.x >= params.gridDimX - 1 || gid.y >= params.gridDimY - 1 || gid.z >= params.gridDimZ - 1) {
        return;
    }

    uint3 dims = uint3(params.gridDimX, params.gridDimY, params.gridDimZ);

    // Sample opacity at 8 corners
    float opacity[8];
    opacity[0] = sampleOpacity(opacityGrid, gid + uint3(0, 0, 0), dims);
    opacity[1] = sampleOpacity(opacityGrid, gid + uint3(1, 0, 0), dims);
    opacity[2] = sampleOpacity(opacityGrid, gid + uint3(1, 1, 0), dims);
    opacity[3] = sampleOpacity(opacityGrid, gid + uint3(0, 1, 0), dims);
    opacity[4] = sampleOpacity(opacityGrid, gid + uint3(0, 0, 1), dims);
    opacity[5] = sampleOpacity(opacityGrid, gid + uint3(1, 0, 1), dims);
    opacity[6] = sampleOpacity(opacityGrid, gid + uint3(1, 1, 1), dims);
    opacity[7] = sampleOpacity(opacityGrid, gid + uint3(0, 1, 1), dims);

    // Convert to SDF
    float sdf[8];
    for (int i = 0; i < 8; i++) {
        sdf[i] = opacityToSDF(opacity[i], params.isoLevel);
    }

    // Compute cube index (which corners are inside the surface)
    int cubeIndex = 0;
    if (sdf[0] < 0) cubeIndex |= 1;
    if (sdf[1] < 0) cubeIndex |= 2;
    if (sdf[2] < 0) cubeIndex |= 4;
    if (sdf[3] < 0) cubeIndex |= 8;
    if (sdf[4] < 0) cubeIndex |= 16;
    if (sdf[5] < 0) cubeIndex |= 32;
    if (sdf[6] < 0) cubeIndex |= 64;
    if (sdf[7] < 0) cubeIndex |= 128;

    // Count triangles for this configuration
    int triCount = getTriangleCount(cubeIndex);

    // Write count to cell index
    uint cellIndex = gid.x + gid.y * (params.gridDimX - 1) + gid.z * (params.gridDimX - 1) * (params.gridDimY - 1);
    triangleCounts[cellIndex] = (uint)triCount;
}

// ============================================================================
// STAGE 4: Marching Cubes - Generate Triangles
// Actually generate the triangle vertices after prefix sum
// ============================================================================

kernel void marchingCubesGenerateKernel(
    const device float* opacityGrid [[buffer(0)]],
    const device uint* triangleOffsets [[buffer(1)]],  // Prefix sum of triangle counts
    device MeshVertex* vertices [[buffer(2)]],
    device uint* indices [[buffer(3)]],                // Triangle indices (3 per triangle)
    device MeshExtractionHeader* header [[buffer(4)]],
    constant MeshExtractionParams& params [[buffer(5)]],
    constant uint& maxVertices [[buffer(6)]],
    constant uint& maxTriangles [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.gridDimX - 1 || gid.y >= params.gridDimY - 1 || gid.z >= params.gridDimZ - 1) {
        return;
    }

    uint3 dims = uint3(params.gridDimX, params.gridDimY, params.gridDimZ);

    // Sample opacity at 8 corners
    float opacity[8];
    opacity[0] = sampleOpacity(opacityGrid, gid + uint3(0, 0, 0), dims);
    opacity[1] = sampleOpacity(opacityGrid, gid + uint3(1, 0, 0), dims);
    opacity[2] = sampleOpacity(opacityGrid, gid + uint3(1, 1, 0), dims);
    opacity[3] = sampleOpacity(opacityGrid, gid + uint3(0, 1, 0), dims);
    opacity[4] = sampleOpacity(opacityGrid, gid + uint3(0, 0, 1), dims);
    opacity[5] = sampleOpacity(opacityGrid, gid + uint3(1, 0, 1), dims);
    opacity[6] = sampleOpacity(opacityGrid, gid + uint3(1, 1, 1), dims);
    opacity[7] = sampleOpacity(opacityGrid, gid + uint3(0, 1, 1), dims);

    // Convert to SDF
    float sdf[8];
    for (int i = 0; i < 8; i++) {
        sdf[i] = opacityToSDF(opacity[i], params.isoLevel);
    }

    // Compute cube index
    int cubeIndex = 0;
    if (sdf[0] < 0) cubeIndex |= 1;
    if (sdf[1] < 0) cubeIndex |= 2;
    if (sdf[2] < 0) cubeIndex |= 4;
    if (sdf[3] < 0) cubeIndex |= 8;
    if (sdf[4] < 0) cubeIndex |= 16;
    if (sdf[5] < 0) cubeIndex |= 32;
    if (sdf[6] < 0) cubeIndex |= 64;
    if (sdf[7] < 0) cubeIndex |= 128;

    // Skip if no triangles
    if (edgeTable[cubeIndex] == 0) return;

    // Cell world position (lower corner)
    float3 cellPos = params.gridMin + float3(gid) * params.voxelSize;

    // Compute vertex positions at 8 corners
    float3 cornerPos[8];
    for (int i = 0; i < 8; i++) {
        cornerPos[i] = cellPos + vertexOffsets[i] * params.voxelSize;
    }

    // Interpolate vertices on edges
    float3 vertexPositions[12];
    int edges = edgeTable[cubeIndex];

    for (int e = 0; e < 12; e++) {
        if (edges & (1 << e)) {
            int2 ev = edgeVertices[e];
            vertexPositions[e] = interpolateVertex(
                cornerPos[ev.x], cornerPos[ev.y],
                sdf[ev.x], sdf[ev.y],
                0.0f  // ISO level is 0 in SDF space
            );
        }
    }

    // Get triangle offset from prefix sum
    uint cellIndex = gid.x + gid.y * (params.gridDimX - 1) + gid.z * (params.gridDimX - 1) * (params.gridDimY - 1);
    uint triangleOffset = triangleOffsets[cellIndex];

    // Generate triangles
    for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
        uint triIdx = triangleOffset + i / 3;

        // Check bounds
        uint vertexBase = triIdx * 3;
        if (vertexBase + 2 >= maxVertices || triIdx >= maxTriangles) {
            // Signal overflow
            atomic_fetch_or_explicit((device atomic_uint*)&header->overflow, 1u, memory_order_relaxed);
            return;
        }

        // Get edge indices for this triangle
        int e0 = triTable[cubeIndex][i];
        int e1 = triTable[cubeIndex][i + 1];
        int e2 = triTable[cubeIndex][i + 2];

        // Write vertices (3 per triangle, no indexing for simplicity)
        float3 p0 = vertexPositions[e0];
        float3 p1 = vertexPositions[e1];
        float3 p2 = vertexPositions[e2];

        // Compute face normal
        float3 edge1 = p1 - p0;
        float3 edge2 = p2 - p0;
        float3 normal = normalize(cross(edge1, edge2));

        // Write vertices
        vertices[vertexBase + 0].position = p0;
        vertices[vertexBase + 0].normal = normal;
        vertices[vertexBase + 0].color = float3(0.7f);  // Default gray
        vertices[vertexBase + 0].opacity = 1.0f;

        vertices[vertexBase + 1].position = p1;
        vertices[vertexBase + 1].normal = normal;
        vertices[vertexBase + 1].color = float3(0.7f);
        vertices[vertexBase + 1].opacity = 1.0f;

        vertices[vertexBase + 2].position = p2;
        vertices[vertexBase + 2].normal = normal;
        vertices[vertexBase + 2].color = float3(0.7f);
        vertices[vertexBase + 2].opacity = 1.0f;

        // Write indices
        indices[triIdx * 3 + 0] = vertexBase + 0;
        indices[triIdx * 3 + 1] = vertexBase + 1;
        indices[triIdx * 3 + 2] = vertexBase + 2;
    }
}

// ============================================================================
// UTILITY KERNELS
// ============================================================================

// Initialize bounds atomics to extreme values
kernel void initBoundsKernel(
    device atomic_int* minBoundsAtomic [[buffer(0)]],
    device atomic_int* maxBoundsAtomic [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0) {
        atomic_store_explicit(&minBoundsAtomic[0], INT_MAX, memory_order_relaxed);
        atomic_store_explicit(&minBoundsAtomic[1], INT_MAX, memory_order_relaxed);
        atomic_store_explicit(&minBoundsAtomic[2], INT_MAX, memory_order_relaxed);
        atomic_store_explicit(&maxBoundsAtomic[0], INT_MIN, memory_order_relaxed);
        atomic_store_explicit(&maxBoundsAtomic[1], INT_MIN, memory_order_relaxed);
        atomic_store_explicit(&maxBoundsAtomic[2], INT_MIN, memory_order_relaxed);
    }
}

// Initialize mesh extraction header
kernel void initMeshHeaderKernel(
    device MeshExtractionHeader* header [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0) {
        header->totalVertices = 0;
        header->totalTriangles = 0;
        header->activeCells = 0;
        header->overflow = 0;
    }
}

// Finalize mesh header with totals from prefix sum
kernel void finalizeMeshHeaderKernel(
    const device uint* prefixSum [[buffer(0)]],
    device MeshExtractionHeader* header [[buffer(1)]],
    constant uint& cellCount [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0) {
        uint totalTriangles = prefixSum[cellCount];
        header->totalTriangles = totalTriangles;
        header->totalVertices = totalTriangles * 3;  // 3 vertices per triangle
    }
}

// ============================================================================
// PREFIX SUM KERNELS (Reuse pattern from GlobalShaders)
// ============================================================================

#define MESH_SCAN_BLOCK_SIZE 256u

kernel void meshBlockReduceKernel(
    const device uint* input [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]]
) {
    threadgroup uint shared[MESH_SCAN_BLOCK_SIZE];

    uint value = (gid < count) ? input[gid] : 0u;

    // Simple reduction in shared memory
    shared[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = MESH_SCAN_BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        blockSums[groupId] = shared[0];
    }
}

kernel void meshBlockScanKernel(
    const device uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    const device uint* blockOffsets [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]]
) {
    threadgroup uint shared[MESH_SCAN_BLOCK_SIZE];

    uint value = (gid < count) ? input[gid] : 0u;

    // Exclusive scan within block
    shared[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Blelloch scan
    // Up-sweep
    for (uint d = 1; d < MESH_SCAN_BLOCK_SIZE; d *= 2) {
        uint ai = (lid + 1) * d * 2 - 1;
        if (ai < MESH_SCAN_BLOCK_SIZE) {
            shared[ai] += shared[ai - d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Set last to 0 for exclusive scan
    if (lid == 0) {
        shared[MESH_SCAN_BLOCK_SIZE - 1] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint d = MESH_SCAN_BLOCK_SIZE / 2; d >= 1; d /= 2) {
        uint ai = (lid + 1) * d * 2 - 1;
        if (ai < MESH_SCAN_BLOCK_SIZE) {
            uint t = shared[ai - d];
            shared[ai - d] = shared[ai];
            shared[ai] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Add block offset
    if (gid < count) {
        output[gid] = shared[lid] + blockOffsets[groupId];
    }
}

kernel void meshSingleBlockScanKernel(
    device uint* blockSums [[buffer(0)]],
    constant uint& blockCount [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup uint shared[MESH_SCAN_BLOCK_SIZE];

    uint value = (lid < blockCount) ? blockSums[lid] : 0u;
    shared[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep
    for (uint d = 1; d < MESH_SCAN_BLOCK_SIZE; d *= 2) {
        uint ai = (lid + 1) * d * 2 - 1;
        if (ai < MESH_SCAN_BLOCK_SIZE) {
            shared[ai] += shared[ai - d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store total before zeroing
    uint total = shared[MESH_SCAN_BLOCK_SIZE - 1];

    if (lid == 0) {
        shared[MESH_SCAN_BLOCK_SIZE - 1] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint d = MESH_SCAN_BLOCK_SIZE / 2; d >= 1; d /= 2) {
        uint ai = (lid + 1) * d * 2 - 1;
        if (ai < MESH_SCAN_BLOCK_SIZE) {
            uint t = shared[ai - d];
            shared[ai - d] = shared[ai];
            shared[ai] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid < blockCount) {
        blockSums[lid] = shared[lid];
    }
    // Store total at blockSums[blockCount]
    if (lid == 0) {
        blockSums[blockCount] = total;
    }
}

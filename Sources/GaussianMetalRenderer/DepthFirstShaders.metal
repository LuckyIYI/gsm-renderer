// DepthFirstShaders.metal - Standalone depth-first Gaussian splatting pipeline
// FastGS/Splatshop-style two-phase sort: Depth sort first, then tile sort
//
// Flow:
// 1. Project + Compact visible gaussians → N_visible
// 2. Generate depth keys (32-bit)
// 3. Radix sort by depth (4 passes) → depth-sorted indices
// 4. Apply ordering + compute tile counts
// 5. Prefix sum → gaussian offsets
// 6. Create instances (depth-ordered)
// 7. Counting sort by tile (1 pass)
// 8. Extract tile ranges
// 9. Render (no per-tile sort - already depth-ordered!)

#include <metal_stdlib>
#include <metal_atomic>
#include "BridgingTypes.h"
#include "GaussianShared.h"
#include "GaussianHelpers.h"
using namespace metal;

// =============================================================================
// CONSTANTS
// =============================================================================

#define DF_RADIX_BITS 8
#define DF_RADIX_SIZE 256
#define DF_BLOCK_SIZE 256
#define DF_GRAIN_SIZE 4
#define DF_PREFIX_BLOCK_SIZE 256
#define DF_PREFIX_GRAIN_SIZE 4

// Tile dimensions for DepthFirst pipeline (32x16 to match GlobalSort)
#define DF_TILE_WIDTH 32
#define DF_TILE_HEIGHT 16
#define DF_TG_WIDTH 8
#define DF_TG_HEIGHT 8
#define DF_PIXELS_X 4
#define DF_PIXELS_Y 2

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Get rotation from PackedWorldGaussian
inline float4 dfGetRotation(PackedWorldGaussian g) { return g.rotation; }

// Get rotation from PackedWorldGaussianHalf
inline float4 dfGetRotation(PackedWorldGaussianHalf g) {
    return float4(half(g.rx), half(g.ry), half(g.rz), half(g.rw));
}

// Pack half4 to float2
inline float2 dfPackHalf4(half4 v) {
    uint2 u = uint2(as_type<uint>(half2(v.xy)), as_type<uint>(half2(v.zw)));
    return float2(as_type<float>(u.x), as_type<float>(u.y));
}

// Unpack float2 to half4
inline half4 dfUnpackHalf4(float2 v) {
    uint2 u = uint2(as_type<uint>(v.x), as_type<uint>(v.y));
    return half4(as_type<half2>(u.x), as_type<half2>(u.y));
}

// Compute power from opacity (FlashGS exact)
constant float DF_LN2 = 0.693147180559945f;
inline float dfComputePower(float opacity) {
    return DF_LN2 * 8.0f + DF_LN2 * log2(max(opacity, 1e-6f));
}

// Block-ellipse intersection
inline bool dfSegmentIntersectEllipse(float a, float b, float c, float d, float l, float r) {
    float delta = b * b - 4.0f * a * c;
    float t1 = (l - d) * (2.0f * a) + b;
    float t2 = (r - d) * (2.0f * a) + b;
    return delta >= 0.0f && (t1 <= 0.0f || t1 * t1 <= delta) && (t2 >= 0.0f || t2 * t2 <= delta);
}

inline bool dfBlockIntersectEllipse(int2 pix_min, int2 pix_max, float2 center, float3 conic, float power) {
    float w = 2.0f * power;
    float dx, dy, a, b, c;

    if (center.x * 2.0f < float(pix_min.x + pix_max.x)) {
        dx = center.x - float(pix_min.x);
    } else {
        dx = center.x - float(pix_max.x);
    }
    a = conic.z;
    b = -2.0f * conic.y * dx;
    c = conic.x * dx * dx - w;
    if (dfSegmentIntersectEllipse(a, b, c, center.y, float(pix_min.y), float(pix_max.y))) return true;

    if (center.y * 2.0f < float(pix_min.y + pix_max.y)) {
        dy = center.y - float(pix_min.y);
    } else {
        dy = center.y - float(pix_max.y);
    }
    a = conic.x;
    b = -2.0f * conic.y * dy;
    c = conic.z * dy * dy - w;
    return dfSegmentIntersectEllipse(a, b, c, center.x, float(pix_min.x), float(pix_max.x));
}

inline bool dfBlockContainsCenter(int2 pix_min, int2 pix_max, float2 center) {
    return center.x >= float(pix_min.x) && center.x <= float(pix_max.x) &&
           center.y >= float(pix_min.y) && center.y <= float(pix_max.y);
}

// =============================================================================
// KERNEL 1: CLEAR
// =============================================================================

kernel void dfClear(
    device TileAssignmentHeader* header [[buffer(0)]],
    constant uint& maxCompacted [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        atomic_store_explicit((device atomic_uint*)&header->totalAssignments, 0u, memory_order_relaxed);
        header->maxCapacity = maxCompacted;
        header->overflow = 0u;
    }
}

// =============================================================================
// KERNEL 2: PROJECT + COMPACT
// =============================================================================

template <typename PackedWorldT, typename HarmonicsT>
kernel void dfProjectCompact(
    const device PackedWorldT* worldGaussians [[buffer(0)]],
    const device HarmonicsT* harmonics [[buffer(1)]],
    device CompactedGaussian* compacted [[buffer(2)]],
    device TileAssignmentHeader* header [[buffer(3)]],
    constant CameraUniforms& camera [[buffer(4)]],
    constant TileBinningParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.gaussianCount) return;

    PackedWorldT g = worldGaussians[gid];

    // Early cull: tiny scale
    float3 scale = float3(g.sx, g.sy, g.sz);
    float maxScale = max(scale.x, max(scale.y, scale.z));
    if (maxScale < 0.0005f) return;

    float3 pos = float3(g.px, g.py, g.pz);
    float4 viewPos = camera.viewMatrix * float4(pos, 1.0f);

    // Depth culling
    float depth = viewPos.z;
    if (fabs(depth) < 1e-4f) depth = (depth >= 0) ? 1e-4f : -1e-4f;
    if (depth <= camera.nearPlane) return;

    float opacity = float(g.opacity);
    if (opacity < 1e-4f) return;

    // Project
    float4 clip = camera.projectionMatrix * viewPos;
    if (fabs(clip.w) < 1e-6f) return;

    float ndcX = clip.x / clip.w;
    float ndcY = clip.y / clip.w;
    float px = ((ndcX + 1.0f) * camera.width - 1.0f) * 0.5f;
    float py = ((ndcY + 1.0f) * camera.height - 1.0f) * 0.5f;

    // Covariance
    float4 quat = normalizeQuaternion(dfGetRotation(g));
    float3x3 cov3d = buildCovariance3D(scale, quat);

    float3 vr0 = float3(camera.viewMatrix[0][0], camera.viewMatrix[1][0], camera.viewMatrix[2][0]);
    float3 vr1 = float3(camera.viewMatrix[0][1], camera.viewMatrix[1][1], camera.viewMatrix[2][1]);
    float3 vr2 = float3(camera.viewMatrix[0][2], camera.viewMatrix[1][2], camera.viewMatrix[2][2]);
    float3x3 viewRot = matrixFromRows(vr0, vr1, vr2);

    float2x2 cov2d = projectCovariance(cov3d, viewPos.xyz, viewRot,
                                        camera.focalX, camera.focalY,
                                        camera.width, camera.height);

    float4 conic4;
    float radius;
    computeConicAndRadius(cov2d, conic4, radius);
    float3 conic = conic4.xyz;

    if (radius < 0.5f) return;

    // Tile bounds
    float maxW = float(params.surfaceWidth - 1);
    float maxH = float(params.surfaceHeight - 1);
    float xmin = clamp(px - radius, 0.0f, maxW);
    float xmax = clamp(px + radius, 0.0f, maxW);
    float ymin = clamp(py - radius, 0.0f, maxH);
    float ymax = clamp(py + radius, 0.0f, maxH);

    int tileW = int(params.tileWidth);
    int tileH = int(params.tileHeight);
    int2 minTile = int2(max(0, int(floor(xmin / float(tileW)))),
                        max(0, int(floor(ymin / float(tileH)))));
    int2 maxTile = int2(min(int(params.tilesX), int(ceil(xmax / float(tileW)))),
                        min(int(params.tilesY), int(ceil(ymax / float(tileH)))));

    if (minTile.x >= maxTile.x || minTile.y >= maxTile.y) return;

    // Compute SH color
    float3 color = computeSHColor(harmonics, gid, pos, camera.cameraCenter, camera.shComponents);
    color = max(color + 0.5f, float3(0.0f));

    // Compact
    uint idx = atomic_fetch_add_explicit((device atomic_uint*)&header->totalAssignments, 1u, memory_order_relaxed);
    if (idx >= params.maxCapacity) {
        header->overflow = 1u;
        return;
    }

    compacted[idx].covariance_depth = float4(conic, depth);
    compacted[idx].position_color = float4(px, py, dfPackHalf4(half4(half3(color), half(opacity))));
    compacted[idx].min_tile = minTile;
    compacted[idx].max_tile = maxTile;
    compacted[idx].originalIdx = gid;
}

// Instantiate templates
template [[host_name("dfProjectCompactFloat")]]
kernel void dfProjectCompact<PackedWorldGaussian, float>(
    const device PackedWorldGaussian*, const device float*,
    device CompactedGaussian*, device TileAssignmentHeader*,
    constant CameraUniforms&, constant TileBinningParams&, uint);

template [[host_name("dfProjectCompactHalf")]]
kernel void dfProjectCompact<PackedWorldGaussianHalf, half>(
    const device PackedWorldGaussianHalf*, const device half*,
    device CompactedGaussian*, device TileAssignmentHeader*,
    constant CameraUniforms&, constant TileBinningParams&, uint);

// =============================================================================
// INDIRECT DISPATCH PREPARATION KERNELS
// =============================================================================

// Prepare dispatch args based on visible count from header
// Used after project+compact to dispatch subsequent kernels
kernel void dfPrepareVisibleDispatch(
    const device TileAssignmentHeader* header [[buffer(0)]],
    device MTLDispatchThreadgroupsIndirectArguments* dispatchArgs [[buffer(1)]],
    constant uint& threadsPerGroup [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    uint visibleCount = atomic_load_explicit((const device atomic_uint*)&header->totalAssignments, memory_order_relaxed);
    uint numGroups = (visibleCount + threadsPerGroup - 1) / threadsPerGroup;

    dispatchArgs->threadgroupsPerGrid[0] = max(numGroups, 1u);
    dispatchArgs->threadgroupsPerGrid[1] = 1;
    dispatchArgs->threadgroupsPerGrid[2] = 1;
}

// Prepare radix sort dispatch args based on visible count
kernel void dfPrepareRadixDispatch(
    const device TileAssignmentHeader* header [[buffer(0)]],
    device MTLDispatchThreadgroupsIndirectArguments* dispatchArgs [[buffer(1)]],
    constant uint& elementsPerBlock [[buffer(2)]],  // BLOCK_SIZE * GRAIN_SIZE
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    uint visibleCount = atomic_load_explicit((const device atomic_uint*)&header->totalAssignments, memory_order_relaxed);
    uint numBlocks = (visibleCount + elementsPerBlock - 1) / elementsPerBlock;

    dispatchArgs->threadgroupsPerGrid[0] = max(numBlocks, 1u);
    dispatchArgs->threadgroupsPerGrid[1] = 1;
    dispatchArgs->threadgroupsPerGrid[2] = 1;
}

// Store visible count and radix block count for later use
kernel void dfStoreVisibleInfo(
    const device TileAssignmentHeader* header [[buffer(0)]],
    device uint* visibleInfo [[buffer(1)]],  // [0]=count, [1]=numBlocks, [2]=histogramSize
    constant uint& elementsPerBlock [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    uint visibleCount = atomic_load_explicit((const device atomic_uint*)&header->totalAssignments, memory_order_relaxed);
    uint numBlocks = (visibleCount + elementsPerBlock - 1) / elementsPerBlock;

    visibleInfo[0] = visibleCount;
    visibleInfo[1] = numBlocks;
    visibleInfo[2] = 256 * numBlocks;  // histogram size
}

// Read total instances from last gaussian offset + last tile count
// After prefix scan of gaussianTileCounts, the total is at gaussianOffsets[n-1] + gaussianTileCounts[n-1]
// Also sets up instanceHeader for RadixSortEncoder
kernel void dfComputeTotalInstances(
    const device uint* gaussianOffsets [[buffer(0)]],
    const device uint* gaussianTileCounts [[buffer(1)]],
    const device TileAssignmentHeader* header [[buffer(2)]],
    device uint* instanceInfo [[buffer(3)]],  // Output: [0]=total, [1]=radixNumBlocks, [2]=radixHistogramSize
    device MTLDispatchThreadgroupsIndirectArguments* dispatchArgs [[buffer(4)]],
    constant uint& threadsPerGroup [[buffer(5)]],
    device MTLDispatchThreadgroupsIndirectArguments* radixDispatchArgs [[buffer(6)]],  // For radix sort
    device TileAssignmentHeader* instanceHeader [[buffer(7)]],  // For RadixSortEncoder
    device MTLDispatchThreadgroupsIndirectArguments* radixSortDispatchArgs [[buffer(8)]],  // 7 slots for RadixSortEncoder
    constant uint& maxInstanceCapacity [[buffer(9)]],  // Max instances buffer can hold
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    uint visibleCount = atomic_load_explicit((const device atomic_uint*)&header->totalAssignments, memory_order_relaxed);

    if (visibleCount == 0) {
        instanceInfo[0] = 0;
        instanceInfo[1] = 0;
        instanceInfo[2] = 0;
        dispatchArgs->threadgroupsPerGrid[0] = 1;
        dispatchArgs->threadgroupsPerGrid[1] = 1;
        dispatchArgs->threadgroupsPerGrid[2] = 1;
        radixDispatchArgs->threadgroupsPerGrid[0] = 1;
        radixDispatchArgs->threadgroupsPerGrid[1] = 1;
        radixDispatchArgs->threadgroupsPerGrid[2] = 1;
        // Zero out instanceHeader
        instanceHeader->totalAssignments = 0;
        instanceHeader->paddedCount = 0;
        instanceHeader->maxCapacity = 0;
        instanceHeader->overflow = 0;
        // Zero out radixSortDispatchArgs (7 slots)
        for (int i = 0; i < 7; ++i) {
            radixSortDispatchArgs[i].threadgroupsPerGrid[0] = 1;
            radixSortDispatchArgs[i].threadgroupsPerGrid[1] = 1;
            radixSortDispatchArgs[i].threadgroupsPerGrid[2] = 1;
        }
        return;
    }

    // Total instances = last offset + last count
    uint lastIdx = visibleCount - 1;
    uint totalNeeded = gaussianOffsets[lastIdx] + gaussianTileCounts[lastIdx];

    // Detect overflow and clamp to max capacity
    uint overflow = (totalNeeded > maxInstanceCapacity) ? 1u : 0u;
    uint total = min(totalNeeded, maxInstanceCapacity);

    // Compute radix sort parameters (for custom radix sort)
    uint elementsPerBlock = DF_BLOCK_SIZE * DF_GRAIN_SIZE;  // 256 * 4 = 1024
    uint numBlocks = (total + elementsPerBlock - 1) / elementsPerBlock;
    uint histogramSize = DF_RADIX_SIZE * numBlocks;  // 256 * numBlocks

    instanceInfo[0] = total;
    instanceInfo[1] = numBlocks;
    instanceInfo[2] = histogramSize;

    uint numGroups = (total + threadsPerGroup - 1) / threadsPerGroup;
    dispatchArgs->threadgroupsPerGrid[0] = max(numGroups, 1u);
    dispatchArgs->threadgroupsPerGrid[1] = 1;
    dispatchArgs->threadgroupsPerGrid[2] = 1;

    // Radix dispatch = numBlocks
    radixDispatchArgs->threadgroupsPerGrid[0] = max(numBlocks, 1u);
    radixDispatchArgs->threadgroupsPerGrid[1] = 1;
    radixDispatchArgs->threadgroupsPerGrid[2] = 1;

    // Set up instanceHeader for RadixSortEncoder (uses same block size 256*4=1024)
    instanceHeader->totalAssignments = total;
    instanceHeader->paddedCount = numBlocks * elementsPerBlock;  // Padded to block size
    instanceHeader->maxCapacity = maxInstanceCapacity;  // Store actual capacity
    instanceHeader->overflow = overflow;  // Flag overflow condition

    // Set up radixSortDispatchArgs (7 slots for RadixSortEncoder)
    // Slot 0: fuse - 1 group per 256 threads
    // Slot 1: unpack - 1 group per 256 threads
    // Slots 2-6: histogram, scanBlocks, exclusive, apply, scatter - numBlocks groups
    uint fuseGroups = (total + 255) / 256;
    radixSortDispatchArgs[0].threadgroupsPerGrid[0] = max(fuseGroups, 1u);  // fuse
    radixSortDispatchArgs[0].threadgroupsPerGrid[1] = 1;
    radixSortDispatchArgs[0].threadgroupsPerGrid[2] = 1;
    radixSortDispatchArgs[1].threadgroupsPerGrid[0] = max(fuseGroups, 1u);  // unpack
    radixSortDispatchArgs[1].threadgroupsPerGrid[1] = 1;
    radixSortDispatchArgs[1].threadgroupsPerGrid[2] = 1;
    for (int i = 2; i < 7; ++i) {
        radixSortDispatchArgs[i].threadgroupsPerGrid[0] = max(numBlocks, 1u);
        radixSortDispatchArgs[i].threadgroupsPerGrid[1] = 1;
        radixSortDispatchArgs[i].threadgroupsPerGrid[2] = 1;
    }
}

// =============================================================================
// KERNEL 3: GENERATE DEPTH KEYS (reads count from header indirectly)
// =============================================================================

kernel void dfGenDepthKeys(
    const device CompactedGaussian* compacted [[buffer(0)]],
    device uint* depthKeys [[buffer(1)]],
    device uint* indices [[buffer(2)]],
    const device TileAssignmentHeader* header [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint count = atomic_load_explicit((const device atomic_uint*)&header->totalAssignments, memory_order_relaxed);
    if (gid >= count) return;
    float depth = compacted[gid].covariance_depth.w;
    depthKeys[gid] = as_type<uint>(depth) ^ 0x80000000u;  // Float-sortable
    indices[gid] = gid;
}

// =============================================================================
// KERNEL 4: RADIX SORT (32-bit, 4 passes)
// =============================================================================

kernel void dfRadixHistogram(
    const device uint* keys [[buffer(0)]],
    device uint* histogram [[buffer(1)]],
    const device uint* visibleInfo [[buffer(2)]],  // [0]=count, [1]=numBlocks, [2]=histogramSize
    constant uint& digit [[buffer(3)]],
    uint groupId [[threadgroup_position_in_grid]],
    uint gridSize [[threadgroups_per_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    uint count = visibleInfo[0];

    threadgroup uint localHist[DF_RADIX_SIZE];
    localHist[localId] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint elementsPerBlock = DF_BLOCK_SIZE * DF_GRAIN_SIZE;
    uint baseIdx = groupId * elementsPerBlock;
    uint shift = digit * DF_RADIX_BITS;

    for (uint g = 0; g < DF_GRAIN_SIZE; ++g) {
        uint idx = baseIdx + localId + g * DF_BLOCK_SIZE;
        if (idx < count) {
            uint key = keys[idx];
            uint bucket = (key >> shift) & 0xFFu;
            atomic_fetch_add_explicit((threadgroup atomic_uint*)&localHist[bucket], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    histogram[localId * gridSize + groupId] = localHist[localId];
}

kernel void dfRadixScanHistogram(
    device uint* histogram [[buffer(0)]],
    const device uint* visibleInfo [[buffer(1)]],  // Read histogramSize from visibleInfo[2] or pass offset
    ushort localId [[thread_position_in_threadgroup]]
) {
    uint numElements = visibleInfo[0];  // Actually we need histogramSize, which is 256 * numBlocks
    // But we're passed with offset to visibleInfo[2], so just read [0]

    threadgroup uint shared[DF_BLOCK_SIZE];

    uint elemsPerThread = (numElements + DF_BLOCK_SIZE - 1) / DF_BLOCK_SIZE;
    uint threadBase = localId * elemsPerThread;

    uint localSum = 0;
    uint localVals[32];
    for (uint i = 0; i < elemsPerThread && i < 32; ++i) {
        uint idx = threadBase + i;
        uint val = (idx < numElements) ? histogram[idx] : 0;
        localVals[i] = val;
        localSum += val;
    }

    shared[localId] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (localId == 0) {
        uint sum = 0;
        for (uint i = 0; i < DF_BLOCK_SIZE; ++i) {
            uint val = shared[i];
            shared[i] = sum;
            sum += val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint prefix = shared[localId];
    uint running = prefix;
    for (uint i = 0; i < elemsPerThread && i < 32; ++i) {
        uint idx = threadBase + i;
        if (idx < numElements) {
            histogram[idx] = running;
            running += localVals[i];
        }
    }
}

kernel void dfRadixScatter(
    const device uint* inKeys [[buffer(0)]],
    device uint* outKeys [[buffer(1)]],
    const device uint* inVals [[buffer(2)]],
    device uint* outVals [[buffer(3)]],
    device uint* histogram [[buffer(4)]],
    const device uint* visibleInfo [[buffer(5)]],  // [0]=count
    constant uint& digit [[buffer(6)]],
    uint groupId [[threadgroup_position_in_grid]],
    uint gridSize [[threadgroups_per_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    uint count = visibleInfo[0];

    threadgroup uint binOffsets[DF_RADIX_SIZE];
    threadgroup uint localCounts[DF_RADIX_SIZE];

    binOffsets[localId] = histogram[localId * gridSize + groupId];
    localCounts[localId] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint elementsPerBlock = DF_BLOCK_SIZE * DF_GRAIN_SIZE;
    uint baseIdx = groupId * elementsPerBlock;
    uint shift = digit * DF_RADIX_BITS;

    for (uint g = 0; g < DF_GRAIN_SIZE; ++g) {
        uint idx = baseIdx + localId + g * DF_BLOCK_SIZE;
        if (idx < count) {
            uint key = inKeys[idx];
            uint val = inVals[idx];
            uint bucket = (key >> shift) & 0xFFu;

            uint localOffset = atomic_fetch_add_explicit((threadgroup atomic_uint*)&localCounts[bucket], 1u, memory_order_relaxed);
            uint writePos = binOffsets[bucket] + localOffset;

            outKeys[writePos] = key;
            outVals[writePos] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (localId < DF_RADIX_SIZE) {
            binOffsets[localId] += localCounts[localId];
            localCounts[localId] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// =============================================================================
// KERNEL 5: APPLY DEPTH ORDER → TILE COUNTS (reads count from header)
// =============================================================================

kernel void dfApplyOrder(
    const device uint* sortedIndices [[buffer(0)]],
    const device CompactedGaussian* compacted [[buffer(1)]],
    device uint* gaussianTileCounts [[buffer(2)]],
    const device TileAssignmentHeader* header [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint count = atomic_load_explicit((const device atomic_uint*)&header->totalAssignments, memory_order_relaxed);
    if (gid >= count) return;

    uint compactedIdx = sortedIndices[gid];
    CompactedGaussian g = compacted[compactedIdx];
    int2 minT = g.min_tile;
    int2 maxT = g.max_tile;
    gaussianTileCounts[gid] = uint(maxT.x - minT.x) * uint(maxT.y - minT.y);
}

// =============================================================================
// KERNEL 6: PREFIX SCAN
// =============================================================================

kernel void dfPrefixScan(
    const device uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    const device uint* countPtr [[buffer(2)]],  // Can be visibleInfo or direct count buffer
    device uint* partialSums [[buffer(3)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    uint count = countPtr[0];

    threadgroup uint shared[DF_PREFIX_BLOCK_SIZE * DF_PREFIX_GRAIN_SIZE];

    uint elementsPerGroup = DF_PREFIX_BLOCK_SIZE * DF_PREFIX_GRAIN_SIZE;
    uint globalOffset = groupId * elementsPerGroup;

    for (ushort i = 0; i < DF_PREFIX_GRAIN_SIZE; ++i) {
        uint idx = globalOffset + localId * DF_PREFIX_GRAIN_SIZE + i;
        shared[localId * DF_PREFIX_GRAIN_SIZE + i] = (idx < count) ? input[idx] : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint localSum = 0;
    for (ushort i = 0; i < DF_PREFIX_GRAIN_SIZE; ++i) {
        uint val = shared[localId * DF_PREFIX_GRAIN_SIZE + i];
        shared[localId * DF_PREFIX_GRAIN_SIZE + i] = localSum;
        localSum += val;
    }

    threadgroup uint blockSums[DF_PREFIX_BLOCK_SIZE];
    blockSums[localId] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (localId == 0) {
        uint sum = 0;
        for (ushort i = 0; i < DF_PREFIX_BLOCK_SIZE; ++i) {
            uint val = blockSums[i];
            blockSums[i] = sum;
            sum += val;
        }
        partialSums[groupId] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint blockSum = blockSums[localId];
    for (ushort i = 0; i < DF_PREFIX_GRAIN_SIZE; ++i) {
        uint idx = globalOffset + localId * DF_PREFIX_GRAIN_SIZE + i;
        if (idx < count) {
            output[idx] = shared[localId * DF_PREFIX_GRAIN_SIZE + i] + blockSum;
        }
    }
}

kernel void dfScanPartialSums(
    device uint* partialSums [[buffer(0)]],
    const device uint* numPartialsPtr [[buffer(1)]],  // Can be visibleInfo[1] or direct buffer
    ushort localId [[thread_position_in_threadgroup]]
) {
    uint numPartials = numPartialsPtr[0];

    // Simple sequential exclusive scan - handles ANY number of partials
    // This is run with a single thread (thread 0), which is fine because
    // numPartials is at most ~2000 for 2M gaussians (fast sequential scan)
    if (localId == 0) {
        uint sum = 0;
        for (uint i = 0; i < numPartials; ++i) {
            uint val = partialSums[i];
            partialSums[i] = sum;
            sum += val;
        }
    }
}

kernel void dfFinalizeScan(
    device uint* output [[buffer(0)]],
    const device uint* countPtr [[buffer(1)]],  // Can be visibleInfo or direct count buffer
    const device uint* partialSums [[buffer(2)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    if (groupId == 0) return;

    uint count = countPtr[0];

    uint elementsPerGroup = DF_PREFIX_BLOCK_SIZE * DF_PREFIX_GRAIN_SIZE;
    uint globalOffset = groupId * elementsPerGroup;
    uint blockSum = partialSums[groupId];

    for (ushort i = 0; i < DF_PREFIX_GRAIN_SIZE; ++i) {
        uint idx = globalOffset + localId * DF_PREFIX_GRAIN_SIZE + i;
        if (idx < count) {
            output[idx] += blockSum;
        }
    }
}

// =============================================================================
// KERNEL 7: CREATE INSTANCES (reads count from header)
// =============================================================================

// Create instances with fused key: [depth16 0-15, tile 16-31]
// Layout matches GlobalSort for efficient radix sort (LSB-first)
// NO prior depth sort needed - radix sort handles both dimensions
kernel void dfCreateInstances(
    const device uint* gaussianIndices [[buffer(0)]],  // Identity: [0,1,2,...] (no depth sort needed!)
    const device CompactedGaussian* compacted [[buffer(1)]],
    const device uint* gaussianOffsets [[buffer(2)]],
    device uint* instanceSortKeys [[buffer(3)]],    // Fused key: [depth16 0-15, tile 16-31]
    device uint* instanceGaussianIdx [[buffer(4)]],
    constant uint& tilesX [[buffer(5)]],
    constant int& tileWidth [[buffer(6)]],
    constant int& tileHeight [[buffer(7)]],
    const device TileAssignmentHeader* header [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint count = atomic_load_explicit((const device atomic_uint*)&header->totalAssignments, memory_order_relaxed);
    if (gid >= count) return;

    uint compactedIdx = gaussianIndices[gid];  // Just gid (identity) - no depth sort!
    CompactedGaussian g = compacted[compactedIdx];

    int2 minT = g.min_tile;
    int2 maxT = g.max_tile;

    uint writeOffset = gaussianOffsets[gid];

    // Quantize depth to 16 bits (same as GlobalSort)
    float depth = g.covariance_depth.w;
    uint depthBits = as_type<uint>(depth);
    // Flip sign bit for correct float sorting (positive values sort correctly)
    uint depth16 = (depthBits ^ 0x80000000u) >> 16;  // Top 16 bits, sign-flipped

    // Write ALL tiles in the range
    for (int ty = minT.y; ty < maxT.y; ++ty) {
        for (int tx = minT.x; tx < maxT.x; ++tx) {
            uint tileKey = uint(ty * int(tilesX) + tx);
            // Fused key: [depth16 0-15, tile 16-31]
            // Radix sort (LSB-first) will sort by depth first, then group by tile
            instanceSortKeys[writeOffset] = depth16 | (tileKey << 16);
            instanceGaussianIdx[writeOffset] = compactedIdx;
            writeOffset++;
        }
    }
}

// =============================================================================
// KERNEL 7b: CREATE INSTANCES V2 - outputs SIMD2<UInt32> for RadixSortEncoder
// =============================================================================

// Create instances with SIMD2<UInt32> keys for RadixSortEncoder
// Format: (tileId, depth16) - uses 16-bit quantized depth like GlobalSort
// The depth value is taken directly from the compacted gaussian data.
// 16-bit depth provides sufficient precision for visual correctness (same as GlobalSort).
kernel void dfCreateInstancesV2(
    const device uint* depthSortedIndices [[buffer(0)]],  // Depth-sorted primitive indices
    const device CompactedGaussian* compacted [[buffer(1)]],
    const device uint* gaussianOffsets [[buffer(2)]],
    device uint2* instanceSortKeys [[buffer(3)]],    // (tileId, depth16) for RadixSortEncoder
    device int* instanceGaussianIdx [[buffer(4)]],   // Int32 indices for RadixSortEncoder payload
    constant uint& tilesX [[buffer(5)]],
    constant int& tileWidth [[buffer(6)]],
    constant int& tileHeight [[buffer(7)]],
    const device TileAssignmentHeader* header [[buffer(8)]],
    constant uint& maxInstanceCapacity [[buffer(9)]],  // Max instances buffer can hold
    uint gid [[thread_position_in_grid]]
) {
    uint count = atomic_load_explicit((const device atomic_uint*)&header->totalAssignments, memory_order_relaxed);
    if (gid >= count) return;

    uint primitiveIdx = depthSortedIndices[gid];
    CompactedGaussian g = compacted[primitiveIdx];

    int2 minT = g.min_tile;
    int2 maxT = g.max_tile;

    uint writeOffset = gaussianOffsets[gid];

    // Use actual depth value quantized to 16 bits (same approach as GlobalSort)
    // This provides sufficient precision for visual correctness
    float depth = g.covariance_depth.w;
    // Quantize depth using same method as GlobalSort: XOR flip for float comparison, top 16 bits
    // This ensures consistent ordering with GlobalSort
    uint depth16 = ((as_type<uint>(depth) ^ 0x80000000u) >> 16u) & 0xFFFFu;

    // Write tiles in the range, with bounds checking
    for (int ty = minT.y; ty < maxT.y; ++ty) {
        for (int tx = minT.x; tx < maxT.x; ++tx) {
            if (writeOffset >= maxInstanceCapacity) return;

            uint tileKey = uint(ty * int(tilesX) + tx);
            // SIMD2 format: (tileId, depth16) - primary sort by tile, secondary by depth
            instanceSortKeys[writeOffset] = uint2(tileKey, depth16);
            instanceGaussianIdx[writeOffset] = int(primitiveIdx);
            writeOffset++;
        }
    }
}

// =============================================================================
// KERNEL 7c: CREATE INSTANCES (TILE-ONLY KEYS) - for FastGS-style depth-first sort
// =============================================================================
// When primitives are already sorted by depth, we only need tile keys for instances.
// The depth order is preserved because we iterate over depth-sorted primitives.

kernel void dfCreateInstancesTileOnly(
    const device uint* depthSortedIndices [[buffer(0)]],  // Depth-sorted primitive indices
    const device CompactedGaussian* compacted [[buffer(1)]],
    const device uint* gaussianOffsets [[buffer(2)]],
    device ushort* instanceTileKeys [[buffer(3)]],    // Tile-only keys (16-bit)
    device uint* instancePrimitiveIdx [[buffer(4)]],  // Primitive indices (for rendering)
    constant uint& tilesX [[buffer(5)]],
    const device TileAssignmentHeader* header [[buffer(6)]],
    constant uint& maxInstanceCapacity [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint count = atomic_load_explicit((const device atomic_uint*)&header->totalAssignments, memory_order_relaxed);
    if (gid >= count) return;

    // Get depth-sorted primitive index
    uint primitiveIdx = depthSortedIndices[gid];
    CompactedGaussian g = compacted[primitiveIdx];

    int2 minT = g.min_tile;
    int2 maxT = g.max_tile;

    uint writeOffset = gaussianOffsets[gid];

    // Write tiles in the range (in depth order since we iterate depth-sorted primitives)
    for (int ty = minT.y; ty < maxT.y; ++ty) {
        for (int tx = minT.x; tx < maxT.x; ++tx) {
            if (writeOffset >= maxInstanceCapacity) return;

            ushort tileKey = ushort(ty * int(tilesX) + tx);
            instanceTileKeys[writeOffset] = tileKey;
            instancePrimitiveIdx[writeOffset] = primitiveIdx;
            writeOffset++;
        }
    }
}

// =============================================================================
// KERNEL 8a: COUNT TILE INSTANCES (for counting sort)
// =============================================================================

kernel void dfCountTileInstances(
    const device ushort* instanceTileKeys [[buffer(0)]],
    device atomic_uint* tileCounts [[buffer(1)]],
    const device uint* totalInstances [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint nInstances = totalInstances[0];
    if (gid >= nInstances) return;

    ushort tileKey = instanceTileKeys[gid];
    atomic_fetch_add_explicit(&tileCounts[tileKey], 1u, memory_order_relaxed);
}

// =============================================================================
// KERNEL 8b: SCATTER TILE INSTANCES (stable counting sort)
// =============================================================================
// Scatters instances to their sorted positions while preserving depth order.
// Uses atomic counters per tile to ensure stable ordering.

kernel void dfScatterTileInstances(
    const device ushort* inTileKeys [[buffer(0)]],
    const device uint* inPrimitiveIdx [[buffer(1)]],
    device ushort* outTileKeys [[buffer(2)]],
    device uint* outPrimitiveIdx [[buffer(3)]],
    const device uint* tileOffsets [[buffer(4)]],      // Prefix sum of tile counts
    device atomic_uint* tileCounters [[buffer(5)]],    // Per-tile atomic counters
    const device uint* totalInstances [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint nInstances = totalInstances[0];
    if (gid >= nInstances) return;

    ushort tileKey = inTileKeys[gid];
    uint primitiveIdx = inPrimitiveIdx[gid];

    // Get position within this tile's range (atomic increment for stability)
    uint localOffset = atomic_fetch_add_explicit(&tileCounters[tileKey], 1u, memory_order_relaxed);
    uint globalOffset = tileOffsets[tileKey] + localOffset;

    outTileKeys[globalOffset] = tileKey;
    outPrimitiveIdx[globalOffset] = primitiveIdx;
}

// =============================================================================
// KERNEL 8c: EXTRACT TILE RANGES (from tile-sorted instances)
// =============================================================================

kernel void dfExtractRangesTileOnly(
    const device ushort* instanceTileKeys [[buffer(0)]],
    device uint* tileRangesFlat [[buffer(1)]],  // [start0, end0, start1, end1, ...]
    const device uint* totalInstances [[buffer(2)]],
    constant uint& maxTiles [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint nInstances = totalInstances[0];
    if (gid >= nInstances) return;

    ushort tileKey = instanceTileKeys[gid];
    if (tileKey >= maxTiles) return;

    device atomic_uint* rangeStart = (device atomic_uint*)&tileRangesFlat[tileKey * 2];
    device atomic_uint* rangeEnd = (device atomic_uint*)&tileRangesFlat[tileKey * 2 + 1];

    if (gid == 0) {
        atomic_store_explicit(rangeStart, 0, memory_order_relaxed);
    } else {
        ushort prevKey = instanceTileKeys[gid - 1];
        if (tileKey != prevKey) {
            if (prevKey < maxTiles) {
                device atomic_uint* prevEnd = (device atomic_uint*)&tileRangesFlat[prevKey * 2 + 1];
                atomic_store_explicit(prevEnd, gid, memory_order_relaxed);
            }
            atomic_store_explicit(rangeStart, gid, memory_order_relaxed);
        }
    }

    if (gid == nInstances - 1) {
        atomic_store_explicit(rangeEnd, nInstances, memory_order_relaxed);
    }
}

// =============================================================================
// KERNEL 8 (LEGACY): COUNTING SORT (16-bit tile keys) - reads nInstances from buffer
// =============================================================================

// Count tiles from fused keys [depth16 0-15, tile 16-31]
kernel void dfCountTiles(
    const device uint* instanceSortKeys [[buffer(0)]],  // Fused keys: depth16 | (tile << 16)
    device atomic_uint* tileCounts [[buffer(1)]],
    const device uint* totalInstances [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint nInstances = totalInstances[0];
    if (gid >= nInstances) return;
    uint tileKey = instanceSortKeys[gid] >> 16;  // Extract tile from upper 16 bits
    atomic_fetch_add_explicit(&tileCounts[tileKey], 1u, memory_order_relaxed);
}

// Scatter instances by tile (stable counting sort)
// Preserves relative order of instances within each tile (depth order!)
kernel void dfScatterTiles(
    const device uint* inSortKeys [[buffer(0)]],        // Fused keys: depth16 | (tile << 16)
    const device uint* inGaussianIdx [[buffer(1)]],
    device uint* outSortKeys [[buffer(2)]],             // Output fused keys
    device uint* outGaussianIdx [[buffer(3)]],
    const device uint* tileOffsets [[buffer(4)]],
    device atomic_uint* tileCounters [[buffer(5)]],
    const device uint* totalInstances [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint nInstances = totalInstances[0];
    if (gid >= nInstances) return;

    uint sortKey = inSortKeys[gid];
    uint tileKey = sortKey >> 16;  // Extract tile from upper 16 bits
    uint gaussianIdx = inGaussianIdx[gid];

    uint localIdx = atomic_fetch_add_explicit(&tileCounters[tileKey], 1u, memory_order_relaxed);
    uint writePos = tileOffsets[tileKey] + localIdx;

    outSortKeys[writePos] = sortKey;
    outGaussianIdx[writePos] = gaussianIdx;
}

kernel void dfZeroCounters(
    device uint* counters [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        counters[gid] = 0;
    }
}

// =============================================================================
// KERNEL 9: EXTRACT TILE RANGES (reads nInstances from buffer)
// =============================================================================

// Extract tile ranges from sorted SIMD2<UInt32> keys
// After RadixSortEncoder, keys are (tileId, depth16)
// Clear tile ranges before extraction (prevents garbage in unused tiles)
kernel void dfClearTileRanges(
    device uint2* tileRanges [[buffer(0)]],
    constant uint& tileCount [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= tileCount) return;
    tileRanges[gid] = uint2(0, 0);  // Empty range
}

// NOTE: Writing to .x and .y of a device uint2 is NOT atomic - the read-modify-write
// of the whole 8-byte vector can cause race conditions. We use separate atomic stores
// to the underlying uint buffer to fix this.
kernel void dfExtractRanges(
    const device uint2* instanceSortKeys [[buffer(0)]],  // SIMD2: (tileId, depth16)
    device uint* tileRangesFlat [[buffer(1)]],  // Flat uint array: [start0, end0, start1, end1, ...]
    const device uint* totalInstances [[buffer(2)]],
    constant uint& maxTiles [[buffer(3)]],  // Bounds check for tile indices
    uint gid [[thread_position_in_grid]]
) {
    uint nInstances = totalInstances[0];
    if (gid >= nInstances) return;

    // Extract tile key from .x of SIMD2
    uint tileKey = instanceSortKeys[gid].x;

    // Bounds check: skip invalid tile indices (garbage data protection)
    if (tileKey >= maxTiles) return;

    // Use atomic writes to avoid race conditions on uint2 component writes
    // Layout: tileRangesFlat[tileKey*2] = start, tileRangesFlat[tileKey*2 + 1] = end
    device atomic_uint* rangeStart = (device atomic_uint*)&tileRangesFlat[tileKey * 2];
    device atomic_uint* rangeEnd = (device atomic_uint*)&tileRangesFlat[tileKey * 2 + 1];

    if (gid == 0) {
        atomic_store_explicit(rangeStart, 0, memory_order_relaxed);
    } else {
        uint prevKey = instanceSortKeys[gid - 1].x;  // Extract from .x
        if (tileKey != prevKey) {
            // Bounds check prevKey too
            if (prevKey < maxTiles) {
                device atomic_uint* prevEnd = (device atomic_uint*)&tileRangesFlat[prevKey * 2 + 1];
                atomic_store_explicit(prevEnd, gid, memory_order_relaxed);
            }
            atomic_store_explicit(rangeStart, gid, memory_order_relaxed);
        }
    }

    if (gid == nInstances - 1) {
        atomic_store_explicit(rangeEnd, nInstances, memory_order_relaxed);
    }
}

// =============================================================================
// KERNEL 9a: DEBUG VALIDATION - Check sorted order and count tiles
// =============================================================================
// Writes to a shared buffer:
// [0]: total instances
// [1]: number of out-of-order pairs (should be 0)
// [2]: first out-of-order index (or 0xFFFFFFFF if none)
// [3]: tile ID at first out-of-order index
// [4]: tile ID at index before first out-of-order
// [5-36]: count of instances for tiles 0-31
// [37-68]: start index for tiles 0-31 (computed from sorted keys)
// [69-100]: end index for tiles 0-31 (computed from sorted keys)

kernel void dfDebugValidation(
    const device uint2* instanceSortKeys [[buffer(0)]],  // SIMD2: (tileId, depth16)
    const device uint* totalInstances [[buffer(1)]],
    device uint* debugOutput [[buffer(2)]],  // Shared buffer for debug output (101 uints)
    constant uint& maxTiles [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        uint nInstances = totalInstances[0];
        debugOutput[0] = nInstances;
        debugOutput[1] = 0;  // out-of-order count
        debugOutput[2] = 0xFFFFFFFF;  // first out-of-order index
        debugOutput[3] = 0;  // tile at first out-of-order
        debugOutput[4] = 0;  // tile before first out-of-order
        // Initialize tile counts and ranges
        for (uint i = 0; i < 32; ++i) {
            debugOutput[5 + i] = 0;       // count
            debugOutput[37 + i] = 0xFFFFFFFF;  // start (invalid)
            debugOutput[69 + i] = 0;      // end
        }

        // Check sort order, count tiles, and compute expected ranges
        for (uint i = 0; i < nInstances; ++i) {
            uint tileKey = instanceSortKeys[i].x;
            uint depth16 = instanceSortKeys[i].y;

            // Count this tile and track range
            if (tileKey < 32) {
                debugOutput[5 + tileKey]++;  // count
                if (debugOutput[37 + tileKey] == 0xFFFFFFFF) {
                    debugOutput[37 + tileKey] = i;  // first occurrence = start
                }
                debugOutput[69 + tileKey] = i + 1;  // end = last occurrence + 1
            }

            // Check order (compare with previous)
            if (i > 0) {
                uint prevTile = instanceSortKeys[i - 1].x;
                uint prevDepth = instanceSortKeys[i - 1].y;

                // Keys should be sorted by (tileKey, depth16)
                // tileKey is primary sort key
                bool outOfOrder = false;
                if (tileKey < prevTile) {
                    outOfOrder = true;  // Tile went backwards
                } else if (tileKey == prevTile && depth16 < prevDepth) {
                    outOfOrder = true;  // Same tile, depth went backwards
                }

                if (outOfOrder) {
                    debugOutput[1]++;  // Increment out-of-order count
                    if (debugOutput[2] == 0xFFFFFFFF) {
                        // First out-of-order - record details
                        debugOutput[2] = i;
                        debugOutput[3] = tileKey;
                        debugOutput[4] = prevTile;
                    }
                }
            }
        }
    }
}

// =============================================================================
// KERNEL 9b: GATHER SORTED GAUSSIANS (reorder data for sequential access)
// =============================================================================

kernel void dfGatherGaussians(
    const device CompactedGaussian* compacted [[buffer(0)]],
    const device int* sortedIndices [[buffer(1)]],  // Int32 indices from RadixSortEncoder
    device CompactedGaussian* sortedCompacted [[buffer(2)]],  // Output: reordered Gaussians
    const device uint* instanceInfo [[buffer(3)]],  // [0]=total
    uint gid [[thread_position_in_grid]]
) {
    uint total = instanceInfo[0];
    if (gid >= total) return;

    uint gIdx = uint(sortedIndices[gid]);
    sortedCompacted[gid] = compacted[gIdx];
}

// =============================================================================
// KERNEL 10: RENDER WITH COOPERATIVE LOADING (like GlobalSort)
// =============================================================================

#define DF_RENDER_BATCH_SIZE 64  // 8x8 threadgroup size

// Simple render kernel - 1 pixel per thread like GlobalSort (for performance testing)
// Uses 16x16 threadgroup with batch size 256
#define DF_RENDER_BATCH_256 256

kernel void dfRenderSorted(
    const device CompactedGaussian* compacted [[buffer(0)]],  // Original compacted gaussians
    const device uint2* tileRanges [[buffer(1)]],
    const device int* sortedIndices [[buffer(2)]],  // Sorted indices from RadixSortEncoder
    texture2d<half, access::write> colorOut [[texture(0)]],
    constant RenderParams& params [[buffer(3)]],
    uint2 groupId [[threadgroup_position_in_grid]],
    uint2 localId [[thread_position_in_threadgroup]]
) {
    uint tileId = groupId.y * params.tilesX + groupId.x;
    uint2 range = tileRanges[tileId];
    // Safety check: if range looks like garbage (underflow or huge), skip tile
    if (range.y < range.x || range.y - range.x > 100000u) return;
    uint gaussCount = range.y - range.x;
    if (gaussCount == 0) return;

    // 8x8 threads, 4x2 pixels per thread (matches GlobalSort exactly)
    uint baseX = groupId.x * DF_TILE_WIDTH + localId.x * DF_PIXELS_X;
    uint baseY = groupId.y * DF_TILE_HEIGHT + localId.y * DF_PIXELS_Y;

    // Pixel positions (8 pixels: 4 wide x 2 tall)
    half2 pos00 = half2(half(baseX + 0), half(baseY + 0));
    half2 pos10 = half2(half(baseX + 1), half(baseY + 0));
    half2 pos20 = half2(half(baseX + 2), half(baseY + 0));
    half2 pos30 = half2(half(baseX + 3), half(baseY + 0));
    half2 pos01 = half2(half(baseX + 0), half(baseY + 1));
    half2 pos11 = half2(half(baseX + 1), half(baseY + 1));
    half2 pos21 = half2(half(baseX + 2), half(baseY + 1));
    half2 pos31 = half2(half(baseX + 3), half(baseY + 1));

    // Accumulators for 8 pixels (4x2)
    half trans00 = 1.0h, trans10 = 1.0h, trans20 = 1.0h, trans30 = 1.0h;
    half trans01 = 1.0h, trans11 = 1.0h, trans21 = 1.0h, trans31 = 1.0h;

    half3 color00 = 0, color10 = 0, color20 = 0, color30 = 0;
    half3 color01 = 0, color11 = 0, color21 = 0, color31 = 0;

    // Direct loop with indirection (like GlobalSort)
    for (uint i = 0; i < gaussCount; ++i) {
        // Early exit when all pixels are saturated
        half maxTrans0 = max(max(trans00, trans10), max(trans20, trans30));
        half maxTrans1 = max(max(trans01, trans11), max(trans21, trans31));
        if (max(maxTrans0, maxTrans1) < half(1.0h/255.0h)) break;

        // Direct read from compacted via sorted indices
        uint gaussianIdx = uint(sortedIndices[range.x + i]);
        CompactedGaussian g = compacted[gaussianIdx];

        half2 gMean = half2(g.position_color.xy);
        half3 cov = half3(g.covariance_depth.xyz);
        half4 colorOp = dfUnpackHalf4(g.position_color.zw);
        half3 gColor = colorOp.xyz;
        half opacity = colorOp.w;

        half cxx = cov.x, cyy = cov.z, cxy2 = 2.0h * cov.y;

        // Direction vectors for 8 pixels
        half2 d00 = pos00 - gMean;
        half2 d10 = pos10 - gMean;
        half2 d20 = pos20 - gMean;
        half2 d30 = pos30 - gMean;
        half2 d01 = pos01 - gMean;
        half2 d11 = pos11 - gMean;
        half2 d21 = pos21 - gMean;
        half2 d31 = pos31 - gMean;

        // Quadratic form
        half p00 = d00.x*d00.x*cxx + d00.y*d00.y*cyy + d00.x*d00.y*cxy2;
        half p10 = d10.x*d10.x*cxx + d10.y*d10.y*cyy + d10.x*d10.y*cxy2;
        half p20 = d20.x*d20.x*cxx + d20.y*d20.y*cyy + d20.x*d20.y*cxy2;
        half p30 = d30.x*d30.x*cxx + d30.y*d30.y*cyy + d30.x*d30.y*cxy2;
        half p01 = d01.x*d01.x*cxx + d01.y*d01.y*cyy + d01.x*d01.y*cxy2;
        half p11 = d11.x*d11.x*cxx + d11.y*d11.y*cyy + d11.x*d11.y*cxy2;
        half p21 = d21.x*d21.x*cxx + d21.y*d21.y*cyy + d21.x*d21.y*cxy2;
        half p31 = d31.x*d31.x*cxx + d31.y*d31.y*cyy + d31.x*d31.y*cxy2;

        // Alpha
        half a00 = min(opacity * exp(-0.5h * p00), 0.99h);
        half a10 = min(opacity * exp(-0.5h * p10), 0.99h);
        half a20 = min(opacity * exp(-0.5h * p20), 0.99h);
        half a30 = min(opacity * exp(-0.5h * p30), 0.99h);
        half a01 = min(opacity * exp(-0.5h * p01), 0.99h);
        half a11 = min(opacity * exp(-0.5h * p11), 0.99h);
        half a21 = min(opacity * exp(-0.5h * p21), 0.99h);
        half a31 = min(opacity * exp(-0.5h * p31), 0.99h);

        // Blend color
        color00 += gColor * (a00 * trans00);
        color10 += gColor * (a10 * trans10);
        color20 += gColor * (a20 * trans20);
        color30 += gColor * (a30 * trans30);
        color01 += gColor * (a01 * trans01);
        color11 += gColor * (a11 * trans11);
        color21 += gColor * (a21 * trans21);
        color31 += gColor * (a31 * trans31);

        // Update transparency
        trans00 *= (1.0h - a00); trans10 *= (1.0h - a10);
        trans20 *= (1.0h - a20); trans30 *= (1.0h - a30);
        trans01 *= (1.0h - a01); trans11 *= (1.0h - a11);
        trans21 *= (1.0h - a21); trans31 *= (1.0h - a31);
    }

    // Apply background
    half bg = params.whiteBackground ? 1.0h : 0.0h;
    color00 += trans00 * bg; color10 += trans10 * bg;
    color20 += trans20 * bg; color30 += trans30 * bg;
    color01 += trans01 * bg; color11 += trans11 * bg;
    color21 += trans21 * bg; color31 += trans31 * bg;

    // Write output (bounds check)
    uint w = params.width, h = params.height;
    if (baseX + 0 < w && baseY + 0 < h) colorOut.write(half4(color00, 1.0h - trans00), uint2(baseX + 0, baseY + 0));
    if (baseX + 1 < w && baseY + 0 < h) colorOut.write(half4(color10, 1.0h - trans10), uint2(baseX + 1, baseY + 0));
    if (baseX + 2 < w && baseY + 0 < h) colorOut.write(half4(color20, 1.0h - trans20), uint2(baseX + 2, baseY + 0));
    if (baseX + 3 < w && baseY + 0 < h) colorOut.write(half4(color30, 1.0h - trans30), uint2(baseX + 3, baseY + 0));
    if (baseX + 0 < w && baseY + 1 < h) colorOut.write(half4(color01, 1.0h - trans01), uint2(baseX + 0, baseY + 1));
    if (baseX + 1 < w && baseY + 1 < h) colorOut.write(half4(color11, 1.0h - trans11), uint2(baseX + 1, baseY + 1));
    if (baseX + 2 < w && baseY + 1 < h) colorOut.write(half4(color21, 1.0h - trans21), uint2(baseX + 2, baseY + 1));
    if (baseX + 3 < w && baseY + 1 < h) colorOut.write(half4(color31, 1.0h - trans31), uint2(baseX + 3, baseY + 1));
}

// Legacy render kernel (uses indirection - slower)
kernel void dfRender(
    const device CompactedGaussian* compacted [[buffer(0)]],
    const device uint2* tileRanges [[buffer(1)]],
    const device int* instanceGaussianIdx [[buffer(2)]],  // Int32 indices from RadixSortEncoder
    texture2d<half, access::write> colorOut [[texture(0)]],
    constant RenderParams& params [[buffer(3)]],
    uint2 groupId [[threadgroup_position_in_grid]],
    uint2 localId [[thread_position_in_threadgroup]],
    uint threadIdx [[thread_index_in_threadgroup]]
) {
    // Shared memory for cooperative loading (like GlobalSort)
    threadgroup half2 shPos[DF_RENDER_BATCH_SIZE];
    threadgroup half3 shCov[DF_RENDER_BATCH_SIZE];
    threadgroup half3 shColor[DF_RENDER_BATCH_SIZE];
    threadgroup half shOpacity[DF_RENDER_BATCH_SIZE];

    uint tileId = groupId.y * params.tilesX + groupId.x;
    uint2 range = tileRanges[tileId];
    uint gaussCount = range.y - range.x;
    if (gaussCount == 0) return;

    half px0 = half(groupId.x * DF_TILE_WIDTH + localId.x * DF_PIXELS_X);
    half py0 = half(groupId.y * DF_TILE_HEIGHT + localId.y * DF_PIXELS_Y);

    half3 c00 = 0, c10 = 0, c20 = 0, c30 = 0;
    half3 c01 = 0, c11 = 0, c21 = 0, c31 = 0;
    half t00 = 1, t10 = 1, t20 = 1, t30 = 1;
    half t01 = 1, t11 = 1, t21 = 1, t31 = 1;

    // Process in batches using cooperative loading
    for (uint batch = 0; batch < gaussCount; batch += DF_RENDER_BATCH_SIZE) {
        uint batchCount = min(uint(DF_RENDER_BATCH_SIZE), gaussCount - batch);

        // Cooperative load: each thread loads one Gaussian into shared memory
        if (threadIdx < batchCount) {
            uint instanceIdx = range.x + batch + threadIdx;
            uint gIdx = uint(instanceGaussianIdx[instanceIdx]);
            CompactedGaussian g = compacted[gIdx];

            shPos[threadIdx] = half2(g.position_color.xy);
            shCov[threadIdx] = half3(g.covariance_depth.xyz);
            half4 colorOp = dfUnpackHalf4(g.position_color.zw);
            shColor[threadIdx] = colorOp.xyz;
            shOpacity[threadIdx] = colorOp.w;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Check if this thread's pixels are already saturated
        half maxT0 = max(max(t00, t10), max(t20, t30));
        half maxT1 = max(max(t01, t11), max(t21, t31));
        bool stillActive = max(maxT0, maxT1) >= half(1.0h/255.0h);

        // Process all Gaussians in this batch from shared memory
        if (stillActive) {
            for (uint i = 0; i < batchCount; ++i) {
                half2 gPos = shPos[i];
                half3 cov = shCov[i];
                half3 gColor = shColor[i];
                half opacity = shOpacity[i];

                half cxx = cov.x, cyy = cov.z, cxy2 = 2.0h * cov.y;

                half dx0 = px0 - gPos.x, dx1 = px0 + 1.0h - gPos.x;
                half dx2 = px0 + 2.0h - gPos.x, dx3 = px0 + 3.0h - gPos.x;
                half dy0 = py0 - gPos.y, dy1 = py0 + 1.0h - gPos.y;

                half p00 = dx0*dx0*cxx + dy0*dy0*cyy + dx0*dy0*cxy2;
                half p10 = dx1*dx1*cxx + dy0*dy0*cyy + dx1*dy0*cxy2;
                half p20 = dx2*dx2*cxx + dy0*dy0*cyy + dx2*dy0*cxy2;
                half p30 = dx3*dx3*cxx + dy0*dy0*cyy + dx3*dy0*cxy2;
                half p01 = dx0*dx0*cxx + dy1*dy1*cyy + dx0*dy1*cxy2;
                half p11 = dx1*dx1*cxx + dy1*dy1*cyy + dx1*dy1*cxy2;
                half p21 = dx2*dx2*cxx + dy1*dy1*cyy + dx2*dy1*cxy2;
                half p31 = dx3*dx3*cxx + dy1*dy1*cyy + dx3*dy1*cxy2;

                half a00 = min(opacity * exp(-0.5h * p00), 0.99h);
                half a10 = min(opacity * exp(-0.5h * p10), 0.99h);
                half a20 = min(opacity * exp(-0.5h * p20), 0.99h);
                half a30 = min(opacity * exp(-0.5h * p30), 0.99h);
                half a01 = min(opacity * exp(-0.5h * p01), 0.99h);
                half a11 = min(opacity * exp(-0.5h * p11), 0.99h);
                half a21 = min(opacity * exp(-0.5h * p21), 0.99h);
                half a31 = min(opacity * exp(-0.5h * p31), 0.99h);

                c00 += gColor * (a00 * t00); c10 += gColor * (a10 * t10);
                c20 += gColor * (a20 * t20); c30 += gColor * (a30 * t30);
                c01 += gColor * (a01 * t01); c11 += gColor * (a11 * t11);
                c21 += gColor * (a21 * t21); c31 += gColor * (a31 * t31);

                t00 *= (1.0h - a00); t10 *= (1.0h - a10);
                t20 *= (1.0h - a20); t30 *= (1.0h - a30);
                t01 *= (1.0h - a01); t11 *= (1.0h - a11);
                t21 *= (1.0h - a21); t31 *= (1.0h - a31);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tile-level early exit using simd_all
        maxT0 = max(max(t00, t10), max(t20, t30));
        maxT1 = max(max(t01, t11), max(t21, t31));
        if (simd_all(max(maxT0, maxT1) < half(1.0h/255.0h))) break;
    }

    half bg = params.whiteBackground ? 1.0h : 0.0h;
    c00 += t00 * bg; c10 += t10 * bg; c20 += t20 * bg; c30 += t30 * bg;
    c01 += t01 * bg; c11 += t11 * bg; c21 += t21 * bg; c31 += t31 * bg;

    uint w = params.width, h = params.height;
    uint bx = uint(px0), by = uint(py0);

    if (bx < w && by < h)     colorOut.write(half4(c00, 1.0h - t00), uint2(bx, by));
    if (bx+1 < w && by < h)   colorOut.write(half4(c10, 1.0h - t10), uint2(bx+1, by));
    if (bx+2 < w && by < h)   colorOut.write(half4(c20, 1.0h - t20), uint2(bx+2, by));
    if (bx+3 < w && by < h)   colorOut.write(half4(c30, 1.0h - t30), uint2(bx+3, by));
    if (bx < w && by+1 < h)   colorOut.write(half4(c01, 1.0h - t01), uint2(bx, by+1));
    if (bx+1 < w && by+1 < h) colorOut.write(half4(c11, 1.0h - t11), uint2(bx+1, by+1));
    if (bx+2 < w && by+1 < h) colorOut.write(half4(c21, 1.0h - t21), uint2(bx+2, by+1));
    if (bx+3 < w && by+1 < h) colorOut.write(half4(c31, 1.0h - t31), uint2(bx+3, by+1));
}

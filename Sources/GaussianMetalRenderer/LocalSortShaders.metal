// LocalSortShaders.metal - Per-tile local sort Gaussian splatting pipeline
// Uses bitonic sort within each tile for efficient rendering

#include <metal_stdlib>
#include <metal_atomic>
#include "GaussianShared.h"
#include "GaussianHelpers.h"
#include "GaussianStructs.h"
using namespace metal;

// SH constants and computeSHColor function now defined in GaussianShared.h
// Math helpers (matrixFromRows, normalizeQuaternion, buildCovariance3D, projectCovariance,
// computeConicAndRadius) now defined in GaussianHelpers.h
// Shared structures (CameraUniforms, PackedWorldGaussian, PackedWorldGaussianHalf)
// now defined in GaussianStructs.h

// =============================================================================
// DATA STRUCTURES (LocalSort-specific)
// =============================================================================

/// Compacted gaussian (after projection + culling) - 48 bytes
/// Matches LocalSort's layout for optimal render performance
struct CompactedGaussian {
    float4 covariance_depth;    // conic.xyz + depth (16 bytes)
    float4 position_color;      // pos.xy + packed_half4(color,opacity) (16 bytes)
    int2 min_tile;              // 8 bytes
    int2 max_tile;              // 8 bytes
};

// TileBinningParams from GaussianStructs.h used for project params
// TileAssignmentHeaderAtomic from GaussianStructs.h used for compacted header
// RenderParams from GaussianStructs.h used for render params

// =============================================================================
// HELPER FUNCTIONS (LocalSort-specific - shared math helpers in GaussianHelpers.h)
// =============================================================================

// Pack half4 to float2
inline float2 localSortPackHalf4(half4 v) {
    uint2 u = uint2(as_type<uint>(half2(v.xy)), as_type<uint>(half2(v.zw)));
    return float2(as_type<float>(u.x), as_type<float>(u.y));
}

// Unpack float2 to half4
inline half4 localSortUnpackHalf4(float2 v) {
    uint2 u = uint2(as_type<uint>(v.x), as_type<uint>(v.y));
    return half4(as_type<half2>(u.x), as_type<half2>(u.y));
}

// =============================================================================
// FLASHGS-STYLE ELLIPSE-TILE INTERSECTION
// =============================================================================
// Exact port of FlashGS intersection test
// power = ln2 * (8 + log2(opacity)) = ln(256 * opacity)
// Threshold w = 2 * power in the quadratic form: conic.x*dx^2 + 2*conic.y*dx*dy + conic.z*dy^2 <= w

// Segment-ellipse intersection (FlashGS exact)
inline bool localSortSegmentIntersectEllipse(float a, float b, float c, float d, float l, float r) {
    float delta = b * b - 4.0f * a * c;
    float t1 = (l - d) * (2.0f * a) + b;
    float t2 = (r - d) * (2.0f * a) + b;
    return delta >= 0.0f && (t1 <= 0.0f || t1 * t1 <= delta) && (t2 >= 0.0f || t2 * t2 <= delta);
}

// Block-ellipse intersection (FlashGS exact)
// Tests closest vertical and horizontal edges
inline bool localSortBlockIntersectEllipse(int2 pix_min, int2 pix_max, float2 center, float3 conic, float power) {
    float w = 2.0f * power;
    float dx, dy;
    float a, b, c;

    // Check closest vertical edge
    if (center.x * 2.0f < float(pix_min.x + pix_max.x)) {
        dx = center.x - float(pix_min.x);
    } else {
        dx = center.x - float(pix_max.x);
    }
    a = conic.z;
    b = -2.0f * conic.y * dx;
    c = conic.x * dx * dx - w;

    if (localSortSegmentIntersectEllipse(a, b, c, center.y, float(pix_min.y), float(pix_max.y))) {
        return true;
    }

    // Check closest horizontal edge
    if (center.y * 2.0f < float(pix_min.y + pix_max.y)) {
        dy = center.y - float(pix_min.y);
    } else {
        dy = center.y - float(pix_max.y);
    }
    a = conic.x;
    b = -2.0f * conic.y * dy;
    c = conic.z * dy * dy - w;

    if (localSortSegmentIntersectEllipse(a, b, c, center.x, float(pix_min.x), float(pix_max.x))) {
        return true;
    }

    return false;
}

// Block contains center (FlashGS exact)
inline bool localSortBlockContainsCenter(int2 pix_min, int2 pix_max, float2 center) {
    return center.x >= float(pix_min.x) && center.x <= float(pix_max.x) &&
           center.y >= float(pix_min.y) && center.y <= float(pix_max.y);
}

// Compute power from opacity (FlashGS exact)
// power = ln2 * 8 + ln2 * log2(opacity) = ln(256 * opacity)
constant float LN2 = 0.693147180559945f;
inline float localSortComputePower(float opacity) {
    return LN2 * 8.0f + LN2 * log2(max(opacity, 1e-6f));
}

// =============================================================================
// KERNEL: ZERO TILE COUNTS (simple version for external use)
// =============================================================================

kernel void tileBinningZeroCountsKernel(
    device uint* counts [[buffer(0)]],
    constant uint& tileCount [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < tileCount) {
        counts[gid] = 0;
    }
}

// =============================================================================
// KERNEL 1: CLEAR (full reset including header)
// =============================================================================

kernel void localSortClear(
    device atomic_uint* tileCounts [[buffer(0)]],
    device TileAssignmentHeaderAtomic* header [[buffer(1)]],
    constant uint& tileCount [[buffer(2)]],
    constant uint& maxCapacity [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < tileCount) {
        atomic_store_explicit(&tileCounts[gid], 0u, memory_order_relaxed);
    }
    if (gid == 0u) {
        atomic_store_explicit(&header->visibleCount, 0u, memory_order_relaxed);
        header->maxCapacity = maxCapacity;
        header->overflow = 0u;
    }
}

// =============================================================================
// KERNEL 2: FUSED PROJECT + COMPACT + COUNT
// =============================================================================
// Templated on world gaussian type and harmonics type for maximum flexibility

template <typename PackedWorldT, typename HarmonicsT>
kernel void localSortProjectCompactCount(
    const device PackedWorldT* worldGaussians [[buffer(0)]],
    const device HarmonicsT* harmonics [[buffer(1)]],
    device CompactedGaussian* compacted [[buffer(2)]],
    device TileAssignmentHeaderAtomic* header [[buffer(3)]],
    device atomic_uint* tileCounts [[buffer(4)]],
    constant CameraUniforms& camera [[buffer(5)]],
    constant TileBinningParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.gaussianCount) return;

    // Load
    PackedWorldT g = worldGaussians[gid];

    // Early cull: skip tiny gaussians (scale < 0.001)
    float3 scale = float3(g.scale);
    float maxScale = max(scale.x, max(scale.y, scale.z));
    if (maxScale < 0.0005f) return;  // Skip near-invisible gaussians

    float3 pos = float3(g.position);
    float4 viewPos = camera.viewMatrix * float4(pos, 1.0f);

    // Cull: behind camera - match original pipeline EXACTLY
    // safeDepthComponent: keeps sign, just handles near-zero values
    float depth = viewPos.z;
    float absDepth = fabs(depth);
    if (absDepth < 1e-4f) {
        depth = (depth >= 0) ? 1e-4f : -1e-4f;
    }

    if (depth <= camera.nearPlane) return;
    float opacity = float(g.opacity);

    // Cull: low opacity - match original pipeline
    if (opacity < 1e-4f) return;

    // Project - match original pipeline convention exactly
    float4 clip = camera.projectionMatrix * viewPos;
    if (fabs(clip.w) < 1e-6f) return;

    float ndcX = clip.x / clip.w;
    float ndcY = clip.y / clip.w;
    float px = ((ndcX + 1.0f) * camera.width - 1.0f) * 0.5f;
    float py = ((ndcY + 1.0f) * camera.height - 1.0f) * 0.5f;

    // Covariance (scale already loaded in early cull) - using shared functions from GaussianHelpers.h
    float4 quat = normalizeQuaternion(float4(g.rotation));
    float3x3 cov3d = buildCovariance3D(scale, quat);

    float3 vr0 = float3(camera.viewMatrix[0][0], camera.viewMatrix[1][0], camera.viewMatrix[2][0]);
    float3 vr1 = float3(camera.viewMatrix[0][1], camera.viewMatrix[1][1], camera.viewMatrix[2][1]);
    float3 vr2 = float3(camera.viewMatrix[0][2], camera.viewMatrix[1][2], camera.viewMatrix[2][2]);
    float3x3 viewRot = matrixFromRows(vr0, vr1, vr2);

    float2x2 cov2d = projectCovariance(cov3d, viewPos.xyz, viewRot,
                                        camera.focalX, camera.focalY,
                                        camera.width, camera.height);

    // Compute conic and radius using shared function
    float4 conic4;
    float radius;
    computeConicAndRadius(cov2d, conic4, radius);
    float3 conic = conic4.xyz;

    if (radius < 0.5f) return;

    // Tile bounds - ALIGNED WITH TEMPORAL PIPELINE
    // 1. Clamp pixel coords to screen bounds first
    float maxW = float(params.surfaceWidth - 1);
    float maxH = float(params.surfaceHeight - 1);
    float xmin = clamp(px - radius, 0.0f, maxW);
    float xmax = clamp(px + radius, 0.0f, maxW);
    float ymin = clamp(py - radius, 0.0f, maxH);
    float ymax = clamp(py + radius, 0.0f, maxH);

    // 2. Compute tile bounds (EXCLUSIVE max - matches scatter_simd interpretation)
    int tileW = int(params.tileWidth);
    int tileH = int(params.tileHeight);
    int2 minTile = int2(
        max(0, int(floor(xmin / float(tileW)))),
        max(0, int(floor(ymin / float(tileH))))
    );
    int2 maxTile = int2(
        min(int(params.tilesX), int(ceil(xmax / float(tileW)))),
        min(int(params.tilesY), int(ceil(ymax / float(tileH))))
    );

    // Cull: zero coverage
    if (minTile.x >= maxTile.x || minTile.y >= maxTile.y) return;

    // Count per tile (AABB - simple and reliable)
    for (int ty = minTile.y; ty < maxTile.y; ++ty) {
        for (int tx = minTile.x; tx < maxTile.x; ++tx) {
            uint tileId = uint(ty * int(params.tilesX) + tx);
            atomic_fetch_add_explicit(&tileCounts[tileId], 1u, memory_order_relaxed);
        }
    }

    // Compute color (SH) - uses function constant for compile-time optimization
    float3 color = computeSHColor(harmonics, gid, pos, camera.cameraCenter, camera.shComponents);
    color = max(color + 0.5f, float3(0.0f));

    // Compact: write visible gaussian
    uint idx = atomic_fetch_add_explicit(&header->visibleCount, 1u, memory_order_relaxed);
    if (idx >= params.maxCapacity) {
        header->overflow = 1u;
        return;
    }

    compacted[idx].covariance_depth = float4(conic, depth);
    // Use the computed color from harmonics (debug line removed)
    compacted[idx].position_color = float4(px, py, localSortPackHalf4(half4(half3(color), half(opacity))));
    compacted[idx].min_tile = minTile;
    compacted[idx].max_tile = maxTile;
}

// Instantiate all combinations: World(float/half) x Harmonics(float/half)
// Uses shared structures from GaussianStructs.h: PackedWorldGaussian, PackedWorldGaussianHalf, CameraUniforms

// Float world, float harmonics (original)
template [[host_name("localSortProjectCompactCountFloat")]]
kernel void localSortProjectCompactCount<PackedWorldGaussian, float>(
    const device PackedWorldGaussian*, const device float*,
    device CompactedGaussian*, device TileAssignmentHeaderAtomic*,
    device atomic_uint*, constant CameraUniforms&,
    constant TileBinningParams&, uint);

// Half world, float harmonics
template [[host_name("localSortProjectCompactCountHalf")]]
kernel void localSortProjectCompactCount<PackedWorldGaussianHalf, float>(
    const device PackedWorldGaussianHalf*, const device float*,
    device CompactedGaussian*, device TileAssignmentHeaderAtomic*,
    device atomic_uint*, constant CameraUniforms&,
    constant TileBinningParams&, uint);

// Float world, half harmonics (memory bandwidth optimization)
template [[host_name("localSortProjectCompactCountFloatHalfSh")]]
kernel void localSortProjectCompactCount<PackedWorldGaussian, half>(
    const device PackedWorldGaussian*, const device half*,
    device CompactedGaussian*, device TileAssignmentHeaderAtomic*,
    device atomic_uint*, constant CameraUniforms&,
    constant TileBinningParams&, uint);

// Half world, half harmonics (full half precision pipeline)
template [[host_name("localSortProjectCompactCountHalfHalfSh")]]
kernel void localSortProjectCompactCount<PackedWorldGaussianHalf, half>(
    const device PackedWorldGaussianHalf*, const device half*,
    device CompactedGaussian*, device TileAssignmentHeaderAtomic*,
    device atomic_uint*, constant CameraUniforms&,
    constant TileBinningParams&, uint);

// =============================================================================
// KERNEL 3: PREFIX SCAN (reuse from main file or inline simple version)
// =============================================================================

#define LOCAL_SORT_PREFIX_BLOCK_SIZE 256
#define LOCAL_SORT_PREFIX_GRAIN_SIZE 4

kernel void localSortPrefixScan(
    const device uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    device uint* partialSums [[buffer(3)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    threadgroup uint shared[LOCAL_SORT_PREFIX_BLOCK_SIZE * LOCAL_SORT_PREFIX_GRAIN_SIZE];

    uint elementsPerGroup = LOCAL_SORT_PREFIX_BLOCK_SIZE * LOCAL_SORT_PREFIX_GRAIN_SIZE;
    uint globalOffset = groupId * elementsPerGroup;

    // Load
    for (ushort i = 0; i < LOCAL_SORT_PREFIX_GRAIN_SIZE; ++i) {
        uint idx = globalOffset + localId * LOCAL_SORT_PREFIX_GRAIN_SIZE + i;
        shared[localId * LOCAL_SORT_PREFIX_GRAIN_SIZE + i] = (idx < count) ? input[idx] : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Local prefix sum
    uint localSum = 0;
    for (ushort i = 0; i < LOCAL_SORT_PREFIX_GRAIN_SIZE; ++i) {
        uint val = shared[localId * LOCAL_SORT_PREFIX_GRAIN_SIZE + i];
        shared[localId * LOCAL_SORT_PREFIX_GRAIN_SIZE + i] = localSum;
        localSum += val;
    }

    // Threadgroup scan (simple serial for now)
    threadgroup uint blockSums[LOCAL_SORT_PREFIX_BLOCK_SIZE];
    blockSums[localId] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (localId == 0) {
        uint sum = 0;
        for (ushort i = 0; i < LOCAL_SORT_PREFIX_BLOCK_SIZE; ++i) {
            uint val = blockSums[i];
            blockSums[i] = sum;
            sum += val;
        }
        partialSums[groupId] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Add block sum to local values
    uint blockSum = blockSums[localId];
    for (ushort i = 0; i < LOCAL_SORT_PREFIX_GRAIN_SIZE; ++i) {
        uint idx = globalOffset + localId * LOCAL_SORT_PREFIX_GRAIN_SIZE + i;
        if (idx < count) {
            output[idx] = shared[localId * LOCAL_SORT_PREFIX_GRAIN_SIZE + i] + blockSum;
        }
    }
}

kernel void localSortScanPartialSums(
    device uint* partialSums [[buffer(0)]],
    constant uint& numPartials [[buffer(1)]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    threadgroup uint shared[LOCAL_SORT_PREFIX_BLOCK_SIZE];
    shared[localId] = (localId < numPartials) ? partialSums[localId] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (localId == 0) {
        uint sum = 0;
        for (ushort i = 0; i < LOCAL_SORT_PREFIX_BLOCK_SIZE; ++i) {
            uint val = shared[i];
            shared[i] = sum;
            sum += val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (localId < numPartials) {
        partialSums[localId] = shared[localId];
    }
}

kernel void localSortFinalizeScan(
    device uint* output [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    const device uint* partialSums [[buffer(2)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    if (groupId == 0) return;  // First group doesn't need adjustment

    uint elementsPerGroup = LOCAL_SORT_PREFIX_BLOCK_SIZE * LOCAL_SORT_PREFIX_GRAIN_SIZE;
    uint globalOffset = groupId * elementsPerGroup;
    uint blockSum = partialSums[groupId];

    for (ushort i = 0; i < LOCAL_SORT_PREFIX_GRAIN_SIZE; ++i) {
        uint idx = globalOffset + localId * LOCAL_SORT_PREFIX_GRAIN_SIZE + i;
        if (idx < count) {
            output[idx] += blockSum;
        }
    }
}

// Fused finalize scan + zero counters (one dispatch instead of two)
kernel void localSortFinalizeScanAndZero(
    device uint* output [[buffer(0)]],
    device atomic_uint* counters [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    const device uint* partialSums [[buffer(3)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    uint elementsPerGroup = LOCAL_SORT_PREFIX_BLOCK_SIZE * LOCAL_SORT_PREFIX_GRAIN_SIZE;
    uint globalOffset = groupId * elementsPerGroup;
    uint blockSum = (groupId > 0) ? partialSums[groupId] : 0u;

    for (ushort i = 0; i < LOCAL_SORT_PREFIX_GRAIN_SIZE; ++i) {
        uint idx = globalOffset + localId * LOCAL_SORT_PREFIX_GRAIN_SIZE + i;
        if (idx < count) {
            // Finalize scan (add block sum)
            if (groupId > 0) {
                output[idx] += blockSum;
            }
            // Zero the counter for scatter
            atomic_store_explicit(&counters[idx], 0u, memory_order_relaxed);
        }
    }
}

// =============================================================================
// KERNEL: PREPARE INDIRECT DISPATCH (GPU-driven dispatch args)
// =============================================================================
// Writes MTLDispatchThreadgroupsIndirectArguments based on visibleCount
// Uses Metal's built-in MTLDispatchThreadgroupsIndirectArguments struct

kernel void localSortPrepareScatterDispatch(
    const device TileAssignmentHeaderAtomic* header [[buffer(0)]],
    device MTLDispatchThreadgroupsIndirectArguments* dispatchArgs [[buffer(1)]],
    constant uint& threadgroupWidth [[buffer(2)]]
) {
    uint visibleCount = atomic_load_explicit(&header->visibleCount, memory_order_relaxed);
    uint threadgroups = (visibleCount + threadgroupWidth - 1) / threadgroupWidth;
    dispatchArgs->threadgroupsPerGrid[0] = max(threadgroups, 1u);
    dispatchArgs->threadgroupsPerGrid[1] = 1;
    dispatchArgs->threadgroupsPerGrid[2] = 1;
}

// =============================================================================
// KERNEL 4a: COMPUTE PER-GAUSSIAN TILE COUNTS (for balanced scatter)
// =============================================================================

kernel void localSortComputeGaussianTileCounts(
    const device CompactedGaussian* compacted [[buffer(0)]],
    const device TileAssignmentHeaderAtomic* header [[buffer(1)]],
    device uint* gaussianTileCounts [[buffer(2)]],  // Output: tiles per gaussian
    uint gid [[thread_position_in_grid]]
) {
    uint visibleCount = atomic_load_explicit(&header->visibleCount, memory_order_relaxed);
    if (gid >= visibleCount) {
        gaussianTileCounts[gid] = 0;
        return;
    }

    CompactedGaussian g = compacted[gid];
    int2 minTile = g.min_tile;
    int2 maxTile = g.max_tile;

    uint tileCount = uint(maxTile.x - minTile.x) * uint(maxTile.y - minTile.y);
    gaussianTileCounts[gid] = tileCount;
}


// =============================================================================
// KERNEL 4: SIMD-COOPERATIVE SCATTER (FlashGS-style)
// =============================================================================
// All 32 threads in SIMD group work on ONE gaussian cooperatively
// - Broadcasts gaussian data via simd_shuffle
// - Threads divide up tile range
// - Warp-level atomic batching (only lane 0 does atomic)
// - Each thread computes its write position via popcount

kernel void localSortScatterSimd(
    const device CompactedGaussian* compacted [[buffer(0)]],
    const device TileAssignmentHeaderAtomic* header [[buffer(1)]],
    device atomic_uint* tileCounters [[buffer(2)]],
    const device uint* offsets [[buffer(3)]],
    device uint* sortKeys [[buffer(4)]],
    device uint* sortIndices [[buffer(5)]],
    constant uint& tilesX [[buffer(6)]],
    constant uint& maxCapacity [[buffer(7)]],
    constant int& tileWidth [[buffer(8)]],
    constant int& tileHeight [[buffer(9)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    // Each SIMD group (32 threads) processes gaussians cooperatively
    // Thread group: multiple SIMD groups, each processing different gaussians
    uint visibleCount = atomic_load_explicit(&header->visibleCount, memory_order_relaxed);

    // Calculate which gaussian this SIMD group is responsible for
    uint simd_groups_per_tg = 4;  // 128 threads / 32 = 4 SIMD groups per threadgroup
    uint global_simd_id = tg_id * simd_groups_per_tg + simd_group_id;
    uint gaussianIdx = global_simd_id;

    if (gaussianIdx >= visibleCount) return;

    // Lane 0 loads gaussian data
    CompactedGaussian g;
    float3 conic;
    float2 center;
    float power;
    uint depthKey;
    int minTX, minTY, maxTX, maxTY;

    if (simd_lane == 0) {
        g = compacted[gaussianIdx];
        conic = g.covariance_depth.xyz;
        center = g.position_color.xy;
        half4 colorOp = localSortUnpackHalf4(g.position_color.zw);
        power = localSortComputePower(float(colorOp.w));
        // 32-bit stable key: (depth16 << 16) | compactedIdx for deterministic tie-breaking
        uint depth16 = ((as_type<uint>(g.covariance_depth.w) ^ 0x80000000u) >> 16u) & 0xFFFFu;
        depthKey = (depth16 << 16) | (gaussianIdx & 0xFFFFu);
        minTX = g.min_tile.x;
        minTY = g.min_tile.y;
        maxTX = g.max_tile.x;
        maxTY = g.max_tile.y;
    }

    // Broadcast to all lanes in SIMD group
    conic.x = simd_shuffle(conic.x, 0);
    conic.y = simd_shuffle(conic.y, 0);
    conic.z = simd_shuffle(conic.z, 0);
    center.x = simd_shuffle(center.x, 0);
    center.y = simd_shuffle(center.y, 0);
    power = simd_shuffle(power, 0);
    depthKey = simd_shuffle(depthKey, 0);
    minTX = simd_shuffle(minTX, 0);
    minTY = simd_shuffle(minTY, 0);
    maxTX = simd_shuffle(maxTX, 0);
    maxTY = simd_shuffle(maxTY, 0);

    // Tile range
    int tileRangeX = maxTX - minTX;
    int tileRangeY = maxTY - minTY;
    int totalTiles = tileRangeX * tileRangeY;

    // Each lane handles a subset of tiles
    for (int tileIdx = int(simd_lane); tileIdx < totalTiles; tileIdx += 32) {
        int localY = tileIdx / tileRangeX;
        int localX = tileIdx % tileRangeX;
        int tx = minTX + localX;
        int ty = minTY + localY;

        // Pixel bounds for intersection test
        int2 pix_min = int2(tx * tileWidth, ty * tileHeight);
        int2 pix_max = int2(pix_min.x + tileWidth - 1, pix_min.y + tileHeight - 1);

        // FlashGS intersection test 
        bool valid = localSortBlockContainsCenter(pix_min, pix_max, center) ||
                     localSortBlockIntersectEllipse(pix_min, pix_max, center, conic, power);

        // SIMD early-out: skip iteration if no lanes are valid
        simd_vote ballot = simd_ballot(valid);
        if (simd_vote::vote_t(ballot) == 0) continue;

        // Valid lanes write to their tiles (parallel atomics across SIMD)
        if (valid) {
            uint tileId = uint(ty) * tilesX + uint(tx);
            uint localIdx = atomic_fetch_add_explicit(&tileCounters[tileId], 1u, memory_order_relaxed);
            uint writePos = offsets[tileId] + localIdx;

            if (writePos < maxCapacity) {
                sortKeys[writePos] = depthKey;
                sortIndices[writePos] = gaussianIdx;
            }
        }
    }
}

// =============================================================================
// KERNEL 5: PER-TILE BITONIC SORT (Optimized)
// =============================================================================
// Fast bitonic sort with SIMD optimizations
// Uses packed key-value pairs to reduce memory traffic
// 4096 elements * 4 bytes * 2 arrays = 32KB (fits in threadgroup memory)

#define SORT_MAX_SIZE 4096

kernel void localSortPerTileSort(
    device uint* keys [[buffer(0)]],
    device uint* values [[buffer(1)]],
    const device uint* offsets [[buffer(2)]],
    const device uint* counts [[buffer(3)]],
    device uint* tempKeys [[buffer(4)]],
    device uint* tempValues [[buffer(5)]],
    uint tileId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]],
    ushort tgSize [[threads_per_threadgroup]]
) {
    uint start = offsets[tileId];
    uint count = counts[tileId];

    if (count <= 1) return;

    uint n = min(count, uint(SORT_MAX_SIZE));

    // 32-bit keys for stable sorting (depth16 << 16 | compactedIdx16)
    // This allows deterministic tie-breaking when depths are equal
    threadgroup uint sharedKeys[SORT_MAX_SIZE];
    threadgroup uint sharedVals[SORT_MAX_SIZE];

    // Pad to next power of 2
    uint padded = 1;
    while (padded < n) padded <<= 1;

    // Parallel load with padding
    for (uint i = localId; i < padded; i += tgSize) {
        if (i < n) {
            sharedKeys[i] = keys[start + i];  // Full 32-bit key
            sharedVals[i] = values[start + i];
        } else {
            sharedKeys[i] = 0xFFFFFFFFu;  // Max key for padding
            sharedVals[i] = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort - each thread handles multiple elements per step
    for (uint k = 2; k <= padded; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            // Each thread processes elements strided by tgSize
            for (uint i = localId; i < padded; i += tgSize) {
                uint ixj = i ^ j;
                if (ixj > i) {
                    uint aKey = sharedKeys[i];
                    uint bKey = sharedKeys[ixj];
                    uint aVal = sharedVals[i];
                    uint bVal = sharedVals[ixj];

                    bool ascending = ((i & k) == 0);
                    bool needSwap = ascending ? (aKey > bKey) : (aKey < bKey);

                    if (needSwap) {
                        sharedKeys[i] = bKey;
                        sharedKeys[ixj] = aKey;
                        sharedVals[i] = bVal;
                        sharedVals[ixj] = aVal;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Parallel write back
    for (uint i = localId; i < n; i += tgSize) {
        keys[start + i] = uint(sharedKeys[i]);
        values[start + i] = sharedVals[i];
    }
}

// =============================================================================
// KERNEL 6: RENDER (LocalSort-style, 8 pixels per thread)
// =============================================================================
// Optimized: No arrays (prevents spilling), half precision, direct memory access

#define LOCAL_SORT_TILE_WIDTH 32
#define LOCAL_SORT_TILE_HEIGHT 16
#define LOCAL_SORT_TG_WIDTH 8
#define LOCAL_SORT_TG_HEIGHT 8
#define LOCAL_SORT_PIXELS_X 4
#define LOCAL_SORT_PIXELS_Y 2

kernel void localSortRender(
    const device CompactedGaussian* compacted [[buffer(0)]],
    const device uint* offsets [[buffer(1)]],
    const device uint* counts [[buffer(2)]],
    const device uint* sortedIndices [[buffer(3)]],
    texture2d<half, access::write> colorOut [[texture(0)]],
    texture2d<half, access::write> depthOut [[texture(1)]],
    constant RenderParams& params [[buffer(4)]],
    uint2 groupId [[threadgroup_position_in_grid]],
    uint2 localId [[thread_position_in_threadgroup]],
    uint localIdx [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Shared memory for tile info only
    threadgroup uint tileGaussianCount;
    threadgroup uint tileDataOffset;

    if (localIdx == 0) {
        uint tileId = groupId.y * params.tilesX + groupId.x;
        tileGaussianCount = counts[tileId];
        tileDataOffset = offsets[tileId];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint gaussCount = tileGaussianCount;
    if (gaussCount == 0) return;

    uint offset = tileDataOffset;

    // Base pixel position
    half px0 = half(groupId.x * LOCAL_SORT_TILE_WIDTH + localId.x * LOCAL_SORT_PIXELS_X);
    half py0 = half(groupId.y * LOCAL_SORT_TILE_HEIGHT + localId.y * LOCAL_SORT_PIXELS_Y);

    // Explicit accumulators - no arrays to prevent register spilling
    half3 c00 = 0, c10 = 0, c20 = 0, c30 = 0;
    half3 c01 = 0, c11 = 0, c21 = 0, c31 = 0;
    half t00 = 1, t10 = 1, t20 = 1, t30 = 1;
    half t01 = 1, t11 = 1, t21 = 1, t31 = 1;

    // Process all gaussians
    for (uint i = 0; i < gaussCount; ++i) {
        uint gIdx = sortedIndices[offset + i];
        CompactedGaussian g = compacted[gIdx];

        half2 gPos = half2(g.position_color.xy);
        half3 cov = half3(g.covariance_depth.xyz);
        half4 colorOp = localSortUnpackHalf4(g.position_color.zw);

        // Match original pipeline's formula exactly
        half cxx = cov.x, cyy = cov.z, cxy2 = 2.0h * cov.y;
        half3 gColor = colorOp.xyz;
        half opacity = colorOp.w;

        // Compute distances for all 8 pixels
        half dx0 = px0 - gPos.x, dx1 = px0 + 1.0h - gPos.x;
        half dx2 = px0 + 2.0h - gPos.x, dx3 = px0 + 3.0h - gPos.x;
        half dy0 = py0 - gPos.y, dy1 = py0 + 1.0h - gPos.y;

        // Quadratic form: power = cxx*dx² + cyy*dy² + cxy2*dx*dy
        // Then alpha = opacity * exp(-0.5 * power)
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

        // SIMD-coordinated early exit - no warp divergence
        // Exit only when ALL threads in SIMD group are saturated
        half maxT = max(max(max(t00, t10), max(t20, t30)),
                       max(max(t01, t11), max(t21, t31)));
        bool threadSaturated = maxT < 0.004h;
        if (simd_all(threadSaturated)) break;
    }

    // Background
    half bg = params.whiteBackground ? 1.0h : 0.0h;
    c00 += t00 * bg; c10 += t10 * bg; c20 += t20 * bg; c30 += t30 * bg;
    c01 += t01 * bg; c11 += t11 * bg; c21 += t21 * bg; c31 += t31 * bg;

    // Write output
    uint w = params.width, h = params.height;
    uint bx = uint(px0), by = uint(py0);

    if (bx < w && by < h)     { colorOut.write(half4(c00, 1.0h - t00), uint2(bx, by));     depthOut.write(half4(0), uint2(bx, by)); }
    if (bx+1 < w && by < h)   { colorOut.write(half4(c10, 1.0h - t10), uint2(bx+1, by));   depthOut.write(half4(0), uint2(bx+1, by)); }
    if (bx+2 < w && by < h)   { colorOut.write(half4(c20, 1.0h - t20), uint2(bx+2, by));   depthOut.write(half4(0), uint2(bx+2, by)); }
    if (bx+3 < w && by < h)   { colorOut.write(half4(c30, 1.0h - t30), uint2(bx+3, by));   depthOut.write(half4(0), uint2(bx+3, by)); }
    if (bx < w && by+1 < h)   { colorOut.write(half4(c01, 1.0h - t01), uint2(bx, by+1));   depthOut.write(half4(0), uint2(bx, by+1)); }
    if (bx+1 < w && by+1 < h) { colorOut.write(half4(c11, 1.0h - t11), uint2(bx+1, by+1)); depthOut.write(half4(0), uint2(bx+1, by+1)); }
    if (bx+2 < w && by+1 < h) { colorOut.write(half4(c21, 1.0h - t21), uint2(bx+2, by+1)); depthOut.write(half4(0), uint2(bx+2, by+1)); }
    if (bx+3 < w && by+1 < h) { colorOut.write(half4(c31, 1.0h - t31), uint2(bx+3, by+1)); depthOut.write(half4(0), uint2(bx+3, by+1)); }
}


// LocalShaders.metal - Per-tile local sort Gaussian splatting pipeline
// Uses radix sort within each tile for efficient rendering

#include <metal_stdlib>
#include <metal_atomic>
#include "BridgingTypes.h"
#include "GaussianShared.h"
#include "GaussianHelpers.h"
#include "GaussianPrefixScan.h"
using namespace metal;

// Function constant for optional cluster culling (index 1, matches GaussianMetalRenderer.metal)
constant bool USE_CLUSTER_CULL [[function_constant(1)]];

// =============================================================================
// HELPER FUNCTIONS (Local-specific)
// =============================================================================

// Pack half4 to float2
inline float2 LocalPackHalf4(half4 v) {
    uint2 u = uint2(as_type<uint>(half2(v.xy)), as_type<uint>(half2(v.zw)));
    return float2(as_type<float>(u.x), as_type<float>(u.y));
}

// Unpack float2 to half4
inline half4 LocalUnpackHalf4(float2 v) {
    uint2 u = uint2(as_type<uint>(v.x), as_type<uint>(v.y));
    return half4(as_type<half2>(u.x), as_type<half2>(u.y));
}

// Ellipse intersection is now in GaussianShared.h - use gaussianIntersectsTile()

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

kernel void LocalClear(
    device atomic_uint* tileCounts [[buffer(0)]],
    device TileAssignmentHeader* header [[buffer(1)]],
    constant uint& tileCount [[buffer(2)]],
    constant uint& maxCapacity [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < tileCount) {
        atomic_store_explicit(&tileCounts[gid], 0u, memory_order_relaxed);
    }
    if (gid == 0u) {
        atomic_store_explicit((device atomic_uint*)&header->totalAssignments, 0u, memory_order_relaxed);
        header->maxCapacity = maxCapacity;
        header->overflow = 0u;
    }
}

// =============================================================================
// KERNEL 1b: CLEAR TEXTURES
// =============================================================================
// Clears color and depth textures before rendering (for indirect dispatch)

kernel void LocalClearTextures(
    texture2d<half, access::write> colorTex [[texture(0)]],
    texture2d<half, access::write> depthTex [[texture(1)]],
    constant uint& width [[buffer(0)]],
    constant uint& height [[buffer(1)]],
    constant uint& whiteBackground [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= width || gid.y >= height) return;
    half bg = whiteBackground ? 1.0h : 0.0h;
    colorTex.write(half4(bg, bg, bg, 0.0h), gid);
    depthTex.write(half4(0), gid);
}

// =============================================================================
// KERNEL 1c: PREPARE RENDER DISPATCH
// =============================================================================
// Prepares indirect dispatch arguments from active tile count

kernel void LocalPrepareRenderDispatch(
    const device uint* activeTileCount [[buffer(0)]],
    device uint* dispatchArgs [[buffer(1)]],  // MTLDispatchThreadgroupsIndirectArguments
    uint tid [[thread_position_in_grid]]
) {
    if (tid > 0) return;
    uint count = activeTileCount[0];
    dispatchArgs[0] = count;  // threadgroupsPerGrid.x
    dispatchArgs[1] = 1;      // threadgroupsPerGrid.y
    dispatchArgs[2] = 1;      // threadgroupsPerGrid.z
}

// =============================================================================
// KERNEL 2a: PROJECT + STORE (efficient temp buffer approach)
// =============================================================================
// Projects all gaussians, stores full results to temp buffer, marks visibility.
// Temp buffer can share memory with other buffers not used during projection.

template <typename PackedWorldT, typename HarmonicsT>
kernel void LocalProjectStore(
    const device PackedWorldT* worldGaussians [[buffer(0)]],
    const device HarmonicsT* harmonics [[buffer(1)]],
    device CompactedGaussian* tempBuffer [[buffer(2)]],     // Temp storage at gid
    device uint* visibilityMarks [[buffer(3)]],             // 1 if visible, 0 if culled
    constant CameraUniforms& camera [[buffer(4)]],
    constant TileBinningParams& params [[buffer(5)]],
    const device uchar* clusterVisible [[buffer(6), function_constant(USE_CLUSTER_CULL)]],
    constant uint& clusterSize [[buffer(7), function_constant(USE_CLUSTER_CULL)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.gaussianCount) return;

    // Cluster culling
    if (USE_CLUSTER_CULL) {
        uint clusterId = gid / clusterSize;
        if (clusterVisible[clusterId] == 0) {
            visibilityMarks[gid] = 0u;
            return;
        }
    }

    PackedWorldT g = worldGaussians[gid];

    // Early cull: tiny gaussians
    float3 scale = float3(g.sx, g.sy, g.sz);
    float maxScale = max(scale.x, max(scale.y, scale.z));
    if (maxScale < 0.0005f) {
        visibilityMarks[gid] = 0u;
        return;
    }

    float3 pos = float3(g.px, g.py, g.pz);
    float4 viewPos = camera.viewMatrix * float4(pos, 1.0f);

    float depth = viewPos.z;
    if (fabs(depth) < 1e-4f) depth = (depth >= 0) ? 1e-4f : -1e-4f;

    if (depth <= camera.nearPlane) {
        visibilityMarks[gid] = 0u;
        return;
    }

    float opacity = float(g.opacity);
    if (opacity < 0.004f) {
        visibilityMarks[gid] = 0u;
        return;
    }

    float4 clip = camera.projectionMatrix * viewPos;
    if (fabs(clip.w) < 1e-6f) {
        visibilityMarks[gid] = 0u;
        return;
    }

    float ndcX = clip.x / clip.w;
    float ndcY = clip.y / clip.w;
    float px = ((ndcX + 1.0f) * camera.width - 1.0f) * 0.5f;
    float py = ((ndcY + 1.0f) * camera.height - 1.0f) * 0.5f;

    float4 quat = normalizeQuaternion(getRotation(g));
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

    if (radius < 0.5f) {
        visibilityMarks[gid] = 0u;
        return;
    }

    float det = conic.x * conic.z - conic.y * conic.y;
    float totalInk = opacity * 6.283185f / sqrt(max(det, 1e-6f));
    if (totalInk < 3.0f) {
        visibilityMarks[gid] = 0u;
        return;
    }

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

    if (minTile.x >= maxTile.x || minTile.y >= maxTile.y) {
        visibilityMarks[gid] = 0u;
        return;
    }

    // Compute color
    float3 color = computeSHColor(harmonics, gid, pos, camera.cameraCenter, camera.shComponents);
    color = max(color + 0.5f, float3(0.0f));

    // Store to temp buffer at original index
    tempBuffer[gid].covariance_depth = float4(conic, depth);
    tempBuffer[gid].position_color = float4(px, py, LocalPackHalf4(half4(half3(color), half(opacity))));
    tempBuffer[gid].min_tile = minTile;
    tempBuffer[gid].max_tile = maxTile;
    tempBuffer[gid].originalIdx = gid;

    visibilityMarks[gid] = 1u;
}

// Template instantiations for ProjectStore
template [[host_name("LocalProjectStoreFloat")]]
kernel void LocalProjectStore<PackedWorldGaussian, float>(
    const device PackedWorldGaussian*, const device float*,
    device CompactedGaussian*, device uint*,
    constant CameraUniforms&, constant TileBinningParams&,
    const device uchar*, constant uint&, uint);

template [[host_name("LocalProjectStoreHalf")]]
kernel void LocalProjectStore<PackedWorldGaussianHalf, float>(
    const device PackedWorldGaussianHalf*, const device float*,
    device CompactedGaussian*, device uint*,
    constant CameraUniforms&, constant TileBinningParams&,
    const device uchar*, constant uint&, uint);

template [[host_name("LocalProjectStoreHalfHalfSh")]]
kernel void LocalProjectStore<PackedWorldGaussianHalf, half>(
    const device PackedWorldGaussianHalf*, const device half*,
    device CompactedGaussian*, device uint*,
    constant CameraUniforms&, constant TileBinningParams&,
    const device uchar*, constant uint&, uint);

// =============================================================================
// KERNEL 2b: COMPACT (copies visible from temp to compacted, NO counting)
// =============================================================================
// Uses prefix sum offsets to deterministically copy visible gaussians.
// Counting eliminated - scatter atomics give us count for free.

kernel void LocalCompact(
    const device CompactedGaussian* tempBuffer [[buffer(0)]],
    const device uint* prefixOffsets [[buffer(1)]],  // Exclusive prefix sum
    device CompactedGaussian* compacted [[buffer(2)]],
    constant uint& gaussianCount [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= gaussianCount) return;

    // Check visibility via prefix sum
    uint myOffset = prefixOffsets[gid];
    uint nextOffset = prefixOffsets[gid + 1];
    if (nextOffset <= myOffset) return;  // Not visible

    // Copy from temp to compacted (deterministic index from prefix sum)
    compacted[myOffset] = tempBuffer[gid];
}

// Write visible count to header
kernel void LocalWriteVisibleCount(
    const device uint* prefixOffsets [[buffer(0)]],
    device TileAssignmentHeader* header [[buffer(1)]],
    constant uint& gaussianCount [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    header->totalAssignments = prefixOffsets[gaussianCount];
}

// =============================================================================
// KERNEL 3: PREFIX SCAN (reuse from main file or inline simple version)
// =============================================================================

#define LOCAL_SORT_PREFIX_BLOCK_SIZE 256
#define LOCAL_SORT_PREFIX_GRAIN_SIZE 4

kernel void LocalPrefixScan(
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

kernel void LocalScanPartialSums(
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

kernel void LocalFinalizeScan(
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

// Fused finalize scan + compact active tiles
// Note: No longer zeros counters - clear stage does this before scatter.
// Now runs AFTER scatter, so counters contain real tile counts that sort/render need.
kernel void LocalFinalizeScanAndZero(
    device uint* output [[buffer(0)]],
    const device uint* counters [[buffer(1)]],  // Read-only now
    constant uint& count [[buffer(2)]],
    const device uint* partialSums [[buffer(3)]],
    device uint* activeTileIndices [[buffer(4)]],
    device atomic_uint* activeTileCount [[buffer(5)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    uint elementsPerGroup = LOCAL_SORT_PREFIX_BLOCK_SIZE * LOCAL_SORT_PREFIX_GRAIN_SIZE;
    uint globalOffset = groupId * elementsPerGroup;
    uint blockSum = (groupId > 0) ? partialSums[groupId] : 0u;

    for (ushort i = 0; i < LOCAL_SORT_PREFIX_GRAIN_SIZE; ++i) {
        uint idx = globalOffset + localId * LOCAL_SORT_PREFIX_GRAIN_SIZE + i;
        if (idx < count) {
            // Finalize scan (add block sum) - vestigial but harmless
            if (groupId > 0) {
                output[idx] += blockSum;
            }
            // Read count - if > 0, this tile is active
            uint tileCount = counters[idx];
            if (tileCount > 0) {
                uint slot = atomic_fetch_add_explicit(activeTileCount, 1u, memory_order_relaxed);
                activeTileIndices[slot] = idx;
            }
            // No zeroing - counters needed by sort/render
        }
    }
}

// =============================================================================
// KERNEL: PREPARE INDIRECT DISPATCH (GPU-driven dispatch args)
// =============================================================================
// Writes MTLDispatchThreadgroupsIndirectArguments based on visibleCount
// Uses Metal's built-in MTLDispatchThreadgroupsIndirectArguments struct

kernel void LocalPrepareScatterDispatch(
    const device TileAssignmentHeader* header [[buffer(0)]],
    device MTLDispatchThreadgroupsIndirectArguments* dispatchArgs [[buffer(1)]],
    constant uint& threadgroupWidth [[buffer(2)]]
) {
    uint visibleCount = atomic_load_explicit((const device atomic_uint*)&header->totalAssignments, memory_order_relaxed);
    uint threadgroups = (visibleCount + threadgroupWidth - 1) / threadgroupWidth;
    dispatchArgs->threadgroupsPerGrid[0] = max(threadgroups, 1u);
    dispatchArgs->threadgroupsPerGrid[1] = 1;
    dispatchArgs->threadgroupsPerGrid[2] = 1;
}


// =============================================================================
// KERNEL 4: SIMD-COOPERATIVE SCATTER (FlashGS-style)
// =============================================================================
// All 32 threads in SIMD group work on ONE gaussian cooperatively
// - Broadcasts gaussian data via simd_shuffle
// - Threads divide up tile range
// - Warp-level atomic batching (only lane 0 does atomic)
// - Each thread computes its write position via popcount

kernel void LocalScatterSimd(
    const device CompactedGaussian* compacted [[buffer(0)]],
    const device TileAssignmentHeader* header [[buffer(1)]],
    device atomic_uint* tileCounters [[buffer(2)]],
    device uint* sortKeys [[buffer(3)]],
    device uint* sortIndices [[buffer(4)]],
    constant uint& tilesX [[buffer(5)]],
    constant uint& maxPerTile [[buffer(6)]],
    constant int& tileWidth [[buffer(7)]],
    constant int& tileHeight [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    // Each SIMD group (32 threads) processes gaussians cooperatively
    // Thread group: multiple SIMD groups, each processing different gaussians
    uint visibleCount = atomic_load_explicit((const device atomic_uint*)&header->totalAssignments, memory_order_relaxed);

    // Calculate which gaussian this SIMD group is responsible for
    uint simd_groups_per_tg = 4;  // 128 threads / 32 = 4 SIMD groups per threadgroup
    uint global_simd_id = tg_id * simd_groups_per_tg + simd_group_id;
    uint gaussianIdx = global_simd_id;

    if (gaussianIdx >= visibleCount) return;

    // Lane 0 loads gaussian data
    CompactedGaussian g;
    float3 conic = 0;
    float2 center = 0;
    float power = 0;
    uint depthKey = 0;
    int minTX = 0, minTY = 0, maxTX = 0, maxTY = 0;

    float opacity = 0;
    if (simd_lane == 0) {
        g = compacted[gaussianIdx];
        conic = g.covariance_depth.xyz;
        center = g.position_color.xy;
        half4 colorOp = LocalUnpackHalf4(g.position_color.zw);
        opacity = float(colorOp.w);
        power = gaussianComputePower(opacity);
        // 32-bit stable key: (depth16 << 16) | originalIdx for deterministic tie-breaking
        // Use originalIdx (world buffer index) instead of compactedIdx to ensure determinism
        uint depth16 = ((as_type<uint>(g.covariance_depth.w) ^ 0x80000000u) >> 16u) & 0xFFFFu;
        depthKey = (depth16 << 16) | (g.originalIdx & 0xFFFFu);
        minTX = g.min_tile.x;
        minTY = g.min_tile.y;
        maxTX = g.max_tile.x;
        maxTY = g.max_tile.y;
    }
    opacity = simd_shuffle(opacity, 0);

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

        // Shared ellipse intersection test (from GaussianShared.h)
        bool valid = gaussianIntersectsTile(pix_min, pix_max, center, conic, power);

        // SIMD early-out: skip iteration if no lanes are valid
        simd_vote ballot = simd_ballot(valid);
        if (simd_vote::vote_t(ballot) == 0) continue;

        // Valid lanes write to their tiles (parallel atomics across SIMD)
        // Fixed layout: tileId * maxPerTile + localIdx (no prefix sum needed)
        if (valid) {
            uint tileId = uint(ty) * tilesX + uint(tx);
            uint localIdx = atomic_fetch_add_explicit(&tileCounters[tileId], 1u, memory_order_relaxed);

            if (localIdx < maxPerTile) {
                uint writePos = tileId * maxPerTile + localIdx;
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
// 2048 elements * 4 bytes * 2 arrays = 16KB (safely fits in threadgroup memory)
// Reduced from 4096 to avoid potential memory issues on some GPUs

#define SORT_MAX_SIZE 2048      // 32-bit path: 16KB threadgroup (2048 × 4B × 2)
#define SORT_MAX_SIZE_16 2048   // 16-bit path: 24KB threadgroup (4096 × 4B + 4096 × 2B)

// Tile dimensions for render kernels
// Each thread processes 4×2 = 8 pixels (hardcoded in render kernel)
#define LOCAL_SORT_TILE_WIDTH 16
#define LOCAL_SORT_TILE_HEIGHT 16
#define LOCAL_SORT_TG_WIDTH 4    // 4 threads × 4 pixels = 16
#define LOCAL_SORT_TG_HEIGHT 8   // 8 threads × 2 pixels = 16
#define LOCAL_SORT_PIXELS_X 4
#define LOCAL_SORT_PIXELS_Y 2

// =============================================================================
// 16-BIT SORT WITH PER-TILE DEPTH NORMALIZATION + DETERMINISM
// =============================================================================
// Benefits:
// - 4K max per tile (vs 2K for 32-bit) - handles denser scenes
// - Full 16-bit precision within each tile's depth range
// - Deterministic via originalIdx tie-breaking
// - 24KB threadgroup memory (vs 16KB for 32-bit)

kernel void LocalPerTileSort16(
    const device ushort* depthKeys16 [[buffer(0)]],    // 16-bit depth keys from scatter (sequential reads!)
    const device uint* globalIndices [[buffer(1)]],    // local idx → compacted gaussian idx (for render)
    device ushort* sortedLocalIdx [[buffer(2)]],       // output: sorted local indices
    const device uint* counts [[buffer(3)]],
    constant uint& maxPerTile [[buffer(4)]],
    uint tileId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]],
    ushort tgSize [[threads_per_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id [[simdgroup_index_in_threadgroup]]
) {
    // Fixed layout: each tile's data at tileId * maxPerTile
    uint start = tileId * maxPerTile;
    uint count = counts[tileId];

    if (count <= 1) {
        if (count == 1 && localId == 0) {
            sortedLocalIdx[start] = 0;
        }
        return;
    }

    uint n = min(count, uint(SORT_MAX_SIZE_16));

    // Threadgroup memory: 32-bit keys (16KB) + 16-bit vals (8KB) = 24KB (fits in 32KB limit)
    // Keys: (depth16 << 16) | localIdx for stable sort
    threadgroup uint sharedKeys[SORT_MAX_SIZE_16];
    threadgroup ushort sharedVals[SORT_MAX_SIZE_16];    // local indices (output)

    // Pad to next power of 2
    uint padded = 1;
    while (padded < n) padded <<= 1;

    // =========================================================================
    // PHASE 1: Load depthKeys16 directly (sequential 2-byte reads!)
    // No per-tile normalization - depthKeys16 already has correct sort order
    // =========================================================================
    for (uint i = localId; i < padded; i += tgSize) {
        if (i < n) {
            // Sequential 2-byte read - very cache friendly
            ushort depth16 = depthKeys16[start + i];
            // Sort key: (depth16 << 16) | localIdx for stable ordering
            sharedKeys[i] = (uint(depth16) << 16) | i;
            sharedVals[i] = ushort(i);  // Local index
        } else {
            sharedKeys[i] = 0xFFFFFFFF;  // Max value for padding
            sharedVals[i] = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // PHASE 2: Bitonic sort on 32-bit keys
    // =========================================================================
    for (uint k = 2; k <= padded; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = localId; i < padded; i += tgSize) {
                uint ixj = i ^ j;
                if (ixj > i) {
                    uint aKey = sharedKeys[i];
                    uint bKey = sharedKeys[ixj];
                    ushort aVal = sharedVals[i];
                    ushort bVal = sharedVals[ixj];

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

    // =========================================================================
    // PHASE 4: Write sorted local indices
    // =========================================================================
    for (uint i = localId; i < n; i += tgSize) {
        sortedLocalIdx[start + i] = sharedVals[i];
    }
}

// Scatter kernel variant that writes 16-bit depth keys for 16-bit sort
kernel void LocalScatterSimd16(
    const device CompactedGaussian* compacted [[buffer(0)]],
    const device TileAssignmentHeader* header [[buffer(1)]],
    device atomic_uint* tileCounters [[buffer(2)]],
    device ushort* depthKeys16 [[buffer(3)]],      // 16-bit depth keys - used directly by sort16
    device uint* globalIndices [[buffer(4)]],       // local → global mapping for render
    constant uint& tilesX [[buffer(5)]],
    constant uint& maxPerTile [[buffer(6)]],
    constant int& tileWidth [[buffer(7)]],
    constant int& tileHeight [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    uint visibleCount = atomic_load_explicit((const device atomic_uint*)&header->totalAssignments, memory_order_relaxed);

    uint simd_groups_per_tg = 4;
    uint global_simd_id = tg_id * simd_groups_per_tg + simd_group_id;
    uint gaussianIdx = global_simd_id;

    if (gaussianIdx >= visibleCount) return;

    // Lane 0 loads gaussian data
    CompactedGaussian g;
    float3 conic = 0;
    float2 center = 0;
    float power = 0;
    ushort depth16 = 0;
    uint depthBits = 0;
    uint originalIdx = 0;
    int minTX = 0, minTY = 0, maxTX = 0, maxTY = 0;

    float opacity16 = 0;
    if (simd_lane == 0) {
        g = compacted[gaussianIdx];
        conic = g.covariance_depth.xyz;
        center = g.position_color.xy;
        half4 colorOp = LocalUnpackHalf4(g.position_color.zw);
        opacity16 = float(colorOp.w);
        power = gaussianComputePower(opacity16);
        // Full depth bits for sort
        depthBits = as_type<uint>(g.covariance_depth.w);
        originalIdx = g.originalIdx;
        // 16-bit depth (for backwards compat)
        depth16 = ushort((depthBits ^ 0x80000000u) >> 16u);
        minTX = g.min_tile.x;
        minTY = g.min_tile.y;
        maxTX = g.max_tile.x;
        maxTY = g.max_tile.y;
    }

    // Broadcast to all lanes
    opacity16 = simd_shuffle(opacity16, 0);
    conic.x = simd_shuffle(conic.x, 0);
    conic.y = simd_shuffle(conic.y, 0);
    conic.z = simd_shuffle(conic.z, 0);
    center.x = simd_shuffle(center.x, 0);
    center.y = simd_shuffle(center.y, 0);
    power = simd_shuffle(power, 0);
    depth16 = simd_shuffle(depth16, 0);
    depthBits = simd_shuffle(depthBits, 0);
    originalIdx = simd_shuffle(originalIdx, 0);
    minTX = simd_shuffle(minTX, 0);
    minTY = simd_shuffle(minTY, 0);
    maxTX = simd_shuffle(maxTX, 0);
    maxTY = simd_shuffle(maxTY, 0);

    int tileRangeX = maxTX - minTX;
    int tileRangeY = maxTY - minTY;
    int totalTiles = tileRangeX * tileRangeY;

    for (int tileIdx = int(simd_lane); tileIdx < totalTiles; tileIdx += 32) {
        int localY = tileIdx / tileRangeX;
        int localX = tileIdx % tileRangeX;
        int tx = minTX + localX;
        int ty = minTY + localY;

        int2 pix_min = int2(tx * tileWidth, ty * tileHeight);
        int2 pix_max = int2(pix_min.x + tileWidth - 1, pix_min.y + tileHeight - 1);

        // Shared ellipse intersection test (from GaussianShared.h)
        bool valid = gaussianIntersectsTile(pix_min, pix_max, center, conic, power);

        simd_vote ballot = simd_ballot(valid);
        if (simd_vote::vote_t(ballot) == 0) continue;

        // Fixed layout: tileId * maxPerTile + localIdx
        if (valid) {
            uint tileId = uint(ty) * tilesX + uint(tx);
            uint localIdx = atomic_fetch_add_explicit(&tileCounters[tileId], 1u, memory_order_relaxed);

            if (localIdx < maxPerTile) {
                uint writePos = tileId * maxPerTile + localIdx;
                depthKeys16[writePos] = depth16;
                globalIndices[writePos] = gaussianIdx;
            }
        }
    }
}

kernel void LocalPerTileSort(
    device uint* keys [[buffer(0)]],
    device uint* values [[buffer(1)]],
    const device uint* counts [[buffer(2)]],
    constant uint& maxPerTile [[buffer(3)]],
    uint tileId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]],
    ushort tgSize [[threads_per_threadgroup]]
) {
    // Fixed layout: each tile's data starts at tileId * maxPerTile
    uint start = tileId * maxPerTile;
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
// KERNEL 6: RENDER (Templated - supports 32/16-bit, indirect dispatch)
// =============================================================================
// USE_16BIT: uses sortedLocalIdx + globalIndices instead of sortedIndices
// Indirect dispatch: uses activeTileIndices to get tileId, framebuffer pre-cleared

constant bool USE_16BIT_RENDER [[function_constant(2)]];

template <bool USE_16BIT>
kernel void LocalRenderImpl(
    const device CompactedGaussian* compacted [[buffer(0)]],
    const device uint* counts [[buffer(1)]],
    constant uint& maxPerTile [[buffer(2)]],
    // 32-bit: sortedIndices, 16-bit: sortedLocalIdx
    const device void* sortBuffer [[buffer(3)]],
    // 16-bit only: globalIndices (unused in 32-bit path)
    const device uint* globalIndices [[buffer(4), function_constant(USE_16BIT_RENDER)]],
    // Indirect dispatch: active tile indices
    const device uint* activeTileIndices [[buffer(5)]],
    texture2d<half, access::write> colorOut [[texture(0)]],
    texture2d<half, access::write> depthOut [[texture(1)]],
    constant RenderParams& params [[buffer(6)]],
    uint2 groupId [[threadgroup_position_in_grid]],
    uint2 localId [[thread_position_in_threadgroup]],
    uint localIdx [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Get tile ID from indirect dispatch (1D grid, active tiles only)
    uint groupIdx = groupId.x;

    threadgroup uint tileId;
    threadgroup uint tileGaussianCount;
    threadgroup uint tileDataOffset;

    if (localIdx == 0) {
        tileId = activeTileIndices[groupIdx];
        tileGaussianCount = counts[tileId];
        // Fixed layout: each tile's data at tileId * maxPerTile
        tileDataOffset = tileId * maxPerTile;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint gaussCount = tileGaussianCount;
    uint offset = tileDataOffset;

    // Compute tile position from tileId
    uint tileX = tileId % params.tilesX;
    uint tileY = tileId / params.tilesX;
    half px0 = half(tileX * LOCAL_SORT_TILE_WIDTH + localId.x * LOCAL_SORT_PIXELS_X);
    half py0 = half(tileY * LOCAL_SORT_TILE_HEIGHT + localId.y * LOCAL_SORT_PIXELS_Y);

    // Cast sort buffer to appropriate type
    const device uint* sortedIndices32 = (const device uint*)sortBuffer;
    const device ushort* sortedLocalIdx16 = (const device ushort*)sortBuffer;

    // Explicit accumulators - no arrays to prevent register spilling
    half3 c00 = 0, c10 = 0, c20 = 0, c30 = 0;
    half3 c01 = 0, c11 = 0, c21 = 0, c31 = 0;
    half t00 = 1, t10 = 1, t20 = 1, t30 = 1;
    half t01 = 1, t11 = 1, t21 = 1, t31 = 1;

    // Process all gaussians
    for (uint i = 0; i < gaussCount; ++i) {
        uint gIdx;
        if (USE_16BIT) {
            // Two-level indirection: sortedLocalIdx → globalIndices → compacted
            ushort localSortedIdx = sortedLocalIdx16[offset + i];
            gIdx = globalIndices[offset + localSortedIdx];
        } else {
            gIdx = sortedIndices32[offset + i];
        }
        CompactedGaussian g = compacted[gIdx];

        half2 gPos = half2(g.position_color.xy);
        half3 cov = half3(g.covariance_depth.xyz);
        half4 colorOp = LocalUnpackHalf4(g.position_color.zw);

        half cxx = cov.x, cyy = cov.z, cxy2 = 2.0h * cov.y;
        half3 gColor = colorOp.xyz;
        half opacity = colorOp.w;

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

        if (all(half4(a00, a10, a20, a30) == 0.0h) && all(half4(a01, a11, a21, a31) == 0.0h)) continue;

        c00 += gColor * (a00 * t00); c10 += gColor * (a10 * t10);
        c20 += gColor * (a20 * t20); c30 += gColor * (a30 * t30);
        c01 += gColor * (a01 * t01); c11 += gColor * (a11 * t11);
        c21 += gColor * (a21 * t21); c31 += gColor * (a31 * t31);

        t00 *= (1.0h - a00); t10 *= (1.0h - a10);
        t20 *= (1.0h - a20); t30 *= (1.0h - a30);
        t01 *= (1.0h - a01); t11 *= (1.0h - a11);
        t21 *= (1.0h - a21); t31 *= (1.0h - a31);

        half maxT = max(max(max(t00, t10), max(t20, t30)),
                       max(max(t01, t11), max(t21, t31)));
        if (simd_all(maxT < 0.004h)) break;
    }

    // Background (framebuffer pre-cleared, just blend)
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

// Explicit instantiations
template [[host_name("LocalRenderIndirect")]]
kernel void LocalRenderImpl<false>(
    const device CompactedGaussian*, const device uint*, constant uint&,
    const device void*, const device uint*, const device uint*,
    texture2d<half, access::write>, texture2d<half, access::write>,
    constant RenderParams&, uint2, uint2, uint, uint);

template [[host_name("LocalRenderIndirect16")]]
kernel void LocalRenderImpl<true>(
    const device CompactedGaussian*, const device uint*, constant uint&,
    const device void*, const device uint*, const device uint*,
    texture2d<half, access::write>, texture2d<half, access::write>,
    constant RenderParams&, uint2, uint2, uint, uint);


#include <metal_stdlib>
#include <metal_atomic>
#include "BridgingTypes.h"
#include "GaussianShared.h"
#include "GaussianPrefixScan.h"
using namespace metal;

constant bool USE_CLUSTER_CULL [[function_constant(1)]];

kernel void localClear(
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

kernel void localClearTextures(
    texture2d<half, access::write> colorTex [[texture(0)]],
    texture2d<half, access::write> depthTex [[texture(1)]],
    constant uint& width [[buffer(0)]],
    constant uint& height [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= width || gid.y >= height) return;
    colorTex.write(half4(0.0h, 0.0h, 0.0h, 0.0h), gid);
    if (!is_null_texture(depthTex)) {
        depthTex.write(half4(0), gid);
    }
}

kernel void localPrepareRenderDispatch(
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

template <typename PackedWorldT, typename HarmonicsT>
kernel void localProjectStore(
    const device PackedWorldT* worldGaussians [[buffer(0)]],
    const device HarmonicsT* harmonics [[buffer(1)]],
    device ProjectedGaussian* tempBuffer [[buffer(2)]],
    device uint* visibilityMarks [[buffer(3)]],
    constant CameraUniforms& camera [[buffer(4)]],
    constant TileBinningParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.gaussianCount) return;

    // Zero tile bounds = invisible (scatter loops won't process)
    constexpr ushort2 zeroTile = ushort2(0, 0);

    PackedWorldT g = worldGaussians[gid];

    // Early cull: tiny gaussians
    float3 scale = float3(g.sx, g.sy, g.sz);
    float maxScale = max(scale.x, max(scale.y, scale.z));
    if (maxScale < 0.0005f) {
        tempBuffer[gid].minTile = zeroTile;
        tempBuffer[gid].maxTile = zeroTile;
        visibilityMarks[gid] = 0;
        return;
    }

    float3 pos = float3(g.px, g.py, g.pz);
    float4 viewPos = camera.viewMatrix * float4(pos, 1.0f);

    float depth = safeDepthComponent(viewPos.z);
    if (fabs(depth) < 1e-4f) depth = (depth >= 0) ? 1e-4f : -1e-4f;

    if (depth <= camera.nearPlane) {
        tempBuffer[gid].minTile = zeroTile;
        tempBuffer[gid].maxTile = zeroTile;
        visibilityMarks[gid] = 0;
        return;
    }

    float opacity = float(g.opacity);
    if (opacity < params.alphaThreshold) {
        tempBuffer[gid].minTile = zeroTile;
        tempBuffer[gid].maxTile = zeroTile;
        visibilityMarks[gid] = 0;
        return;
    }

    float4 clip = camera.projectionMatrix * viewPos;
    if (fabs(clip.w) < 1e-6f) {
        tempBuffer[gid].minTile = zeroTile;
        tempBuffer[gid].maxTile = zeroTile;
        visibilityMarks[gid] = 0;
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
        tempBuffer[gid].minTile = zeroTile;
        tempBuffer[gid].maxTile = zeroTile;
        visibilityMarks[gid] = 0;
        return;
    }

    // Total ink filter with depth-adaptive threshold
    // Close gaussians need more ink (higher threshold), far ones can be tiny
    float det = conic.x * conic.z - conic.y * conic.y;
    float totalInk = opacity * 6.283185f / sqrt(max(det, 1e-6f));
    float depthFactor = 1.0 - pow(saturate((camera.farPlane - depth) / (camera.farPlane - camera.nearPlane)), 4.0);
    float adjustedThreshold = depthFactor * params.totalInkThreshold;
    if (totalInk < adjustedThreshold) {
        tempBuffer[gid].minTile = zeroTile;
        tempBuffer[gid].maxTile = zeroTile;
        visibilityMarks[gid] = 0;
        return;
    }

    float2 obbExtents = computeOBBExtents(cov2d, 3.0f);

    float maxW = float(params.surfaceWidth - 1);
    float maxH = float(params.surfaceHeight - 1);
    float xmin = clamp(px - obbExtents.x, 0.0f, maxW);
    float xmax = clamp(px + obbExtents.x, 0.0f, maxW);
    float ymin = clamp(py - obbExtents.y, 0.0f, maxH);
    float ymax = clamp(py + obbExtents.y, 0.0f, maxH);

    int tileW = int(params.tileWidth);
    int tileH = int(params.tileHeight);
    int minTileX = max(0, int(floor(xmin / float(tileW))));
    int minTileY = max(0, int(floor(ymin / float(tileH))));
    int maxTileX = min(int(params.tilesX), int(ceil(xmax / float(tileW))));
    int maxTileY = min(int(params.tilesY), int(ceil(ymax / float(tileH))));

    if (minTileX >= maxTileX || minTileY >= maxTileY) {
        tempBuffer[gid].minTile = zeroTile;
        tempBuffer[gid].maxTile = zeroTile;
        visibilityMarks[gid] = 0;
        return;
    }

    // Compute color
    float3 color = computeSHColor(harmonics, gid, pos, camera.cameraCenter, camera.shComponents);
    color = max(color + 0.5f, float3(0.0f));

    tempBuffer[gid].posX = px;
    tempBuffer[gid].posY = py;
    tempBuffer[gid].conicA = conic.x;
    tempBuffer[gid].conicB = conic.y;
    tempBuffer[gid].conicC = conic.z;
    tempBuffer[gid].depth = half(depth);
    tempBuffer[gid].colorR = half(color.x);
    tempBuffer[gid].colorG = half(color.y);
    tempBuffer[gid].colorB = half(color.z);
    tempBuffer[gid].opacity = half(opacity);

    tempBuffer[gid].minTile = ushort2(minTileX, minTileY);
    tempBuffer[gid].maxTile = ushort2(maxTileX, maxTileY);
    tempBuffer[gid].obbExtentX = half(obbExtents.x);
    tempBuffer[gid].obbExtentY = half(obbExtents.y);

    // Mark as visible for compaction
    visibilityMarks[gid] = 1;
}

// Template instantiations for ProjectStore
template [[host_name("LocalProjectStoreFloat")]]
kernel void localProjectStore<PackedWorldGaussian, float>(
    const device PackedWorldGaussian*, const device float*,
    device ProjectedGaussian*, device uint*,
    constant CameraUniforms&, constant TileBinningParams&, uint
);

template [[host_name("LocalProjectStoreHalfHalfSh")]]
kernel void localProjectStore<PackedWorldGaussianHalf, half>(
    const device PackedWorldGaussianHalf*, const device half*,
    device ProjectedGaussian*, device uint*,
    constant CameraUniforms&, constant TileBinningParams&, uint
 );

kernel void localCompact(
    const device ProjectedGaussian* tempBuffer [[buffer(0)]],
    const device uint* prefixOffsets [[buffer(1)]],
    device ProjectedGaussian* compacted [[buffer(2)]],
    constant uint& gaussianCount [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= gaussianCount) return;

    uint myOffset = prefixOffsets[gid];
    uint nextOffset = prefixOffsets[gid + 1];
    if (nextOffset <= myOffset) return;
    compacted[myOffset] = tempBuffer[gid];
}

kernel void localWriteVisibleCount(
    const device uint* prefixOffsets [[buffer(0)]],
    device TileAssignmentHeader* header [[buffer(1)]],
    constant uint& gaussianCount [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    header->totalAssignments = prefixOffsets[gaussianCount];
}

#define LOCAL_SORT_PREFIX_BLOCK_SIZE 256
#define LOCAL_SORT_PREFIX_GRAIN_SIZE 4

kernel void localPrefixScan(
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

kernel void localScanPartialSums(
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

// Fused finalize scan + compact active tiles
kernel void localFinalizeScanAndZero(
    device uint* output [[buffer(0)]],
    const device uint* counters [[buffer(1)]],
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

kernel void localPrepareScatterDispatch(
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

#define SORT_MAX_SIZE 2048
#define SORT_MAX_SIZE_16 2048

#define LOCAL_SORT_TILE_WIDTH 16
#define LOCAL_SORT_TILE_HEIGHT 16
#define LOCAL_SORT_TG_WIDTH 4
#define LOCAL_SORT_TG_HEIGHT 8
#define LOCAL_SORT_PIXELS_X 4
#define LOCAL_SORT_PIXELS_Y 2

kernel void localPerTileSort16(
    const device ushort* depthKeys16 [[buffer(0)]],
    const device uint* globalIndices [[buffer(1)]],
    device ushort* sortedLocalIdx [[buffer(2)]],
    const device uint* counts [[buffer(3)]],
    constant uint& maxPerTile [[buffer(4)]],
    uint tileId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]],
    ushort tgSize [[threads_per_threadgroup]],
    ushort simdLane [[thread_index_in_simdgroup]],
    ushort simdId [[simdgroup_index_in_threadgroup]]
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

    threadgroup uint sharedKeys[SORT_MAX_SIZE_16];
    threadgroup ushort sharedVals[SORT_MAX_SIZE_16];

    // Pad to next power of 2
    uint padded = 1;
    while (padded < n) padded <<= 1;

    // Load keys
    for (uint i = localId; i < padded; i += tgSize) {
        if (i < n) {
            ushort depth16 = depthKeys16[start + i];
            sharedKeys[i] = (uint(depth16) << 16) | i;
            sharedVals[i] = ushort(i);
        } else {
            sharedKeys[i] = 0xFFFFFFFF;
            sharedVals[i] = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort
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

    // Write sorted indices
    for (uint i = localId; i < n; i += tgSize) {
        sortedLocalIdx[start + i] = sharedVals[i];
    }
}


kernel void localRender16(
    const device ProjectedGaussian* compacted [[buffer(0)]],
    const device uint* counts [[buffer(1)]],
    constant uint& maxPerTile [[buffer(2)]],
    const device ushort* sortedLocalIdx [[buffer(3)]],
    const device uint* globalIndices [[buffer(4)]],
    const device uint* activeTileIndices [[buffer(5)]],
    texture2d<half, access::write> colorOut [[texture(0)]],
    texture2d<half, access::write> depthOut [[texture(1)]],
    constant RenderParams& params [[buffer(6)]],
    uint2 groupId [[threadgroup_position_in_grid]],
    uint2 localId [[thread_position_in_threadgroup]],
    uint localIdx [[thread_index_in_threadgroup]],
    uint simdLane [[thread_index_in_simdgroup]]
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

    // Compute tile position from tileId (compute in uint like Global to avoid half precision loss)
    uint tileX = tileId % params.tilesX;
    uint tileY = tileId / params.tilesX;
    uint baseX = tileX * LOCAL_SORT_TILE_WIDTH + localId.x * LOCAL_SORT_PIXELS_X;
    uint baseY = tileY * LOCAL_SORT_TILE_HEIGHT + localId.y * LOCAL_SORT_PIXELS_Y;

    // Explicit accumulators - no arrays to prevent register spilling
    half3 c00 = 0, c10 = 0, c20 = 0, c30 = 0;
    half3 c01 = 0, c11 = 0, c21 = 0, c31 = 0;
    half t00 = 1, t10 = 1, t20 = 1, t30 = 1;
    half t01 = 1, t11 = 1, t21 = 1, t31 = 1;

    // Depth accumulators - store depth of first significant contribution
    float d00 = 0, d10 = 0, d20 = 0, d30 = 0;
    float d01 = 0, d11 = 0, d21 = 0, d31 = 0;

    // Process all gaussians
    for (uint i = 0; i < gaussCount; ++i) {
        // Two-level indirection: sortedLocalIdx → globalIndices → compacted
        ushort localSortedIdx = sortedLocalIdx[offset + i];
        uint gIdx = globalIndices[offset + localSortedIdx];
        ProjectedGaussian g = compacted[gIdx];

        half2 gPos = half2(g.posX, g.posY);
        half3 cov = half3(g.conicA, g.conicB, g.conicC);
        half3 gColor = half3(g.colorR, g.colorG, g.colorB);
        half opacity = g.opacity;
        float gDepth = g.depth;

        half cxx = cov.x, cyy = cov.z, cxy2 = 2.0h * cov.y;

        // Compute pixel positions in uint first, then subtract (like Global)
        half dx0 = half(baseX + 0) - gPos.x, dx1 = half(baseX + 1) - gPos.x;
        half dx2 = half(baseX + 2) - gPos.x, dx3 = half(baseX + 3) - gPos.x;
        half dy0 = half(baseY + 0) - gPos.y, dy1 = half(baseY + 1) - gPos.y;

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

        // Record depth of first significant contribution (alpha > 0.1)
        if (d00 == 0 && a00 > 0.1h) d00 = gDepth;
        if (d10 == 0 && a10 > 0.1h) d10 = gDepth;
        if (d20 == 0 && a20 > 0.1h) d20 = gDepth;
        if (d30 == 0 && a30 > 0.1h) d30 = gDepth;
        if (d01 == 0 && a01 > 0.1h) d01 = gDepth;
        if (d11 == 0 && a11 > 0.1h) d11 = gDepth;
        if (d21 == 0 && a21 > 0.1h) d21 = gDepth;
        if (d31 == 0 && a31 > 0.1h) d31 = gDepth;

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

    half bg = 0.0h;
    c00 += t00 * bg; c10 += t10 * bg; c20 += t20 * bg; c30 += t30 * bg;
    c01 += t01 * bg; c11 += t11 * bg; c21 += t21 * bg; c31 += t31 * bg;

    // Write output
    uint w = params.width, h = params.height;
    uint bx = baseX, by = baseY;
    bool hasDepth = !is_null_texture(depthOut);

    if (bx < w && by < h)     { colorOut.write(half4(c00, 1.0h - t00), uint2(bx, by));     if (hasDepth) depthOut.write(half4(half(d00)), uint2(bx, by)); }
    if (bx+1 < w && by < h)   { colorOut.write(half4(c10, 1.0h - t10), uint2(bx+1, by));   if (hasDepth) depthOut.write(half4(half(d10)), uint2(bx+1, by)); }
    if (bx+2 < w && by < h)   { colorOut.write(half4(c20, 1.0h - t20), uint2(bx+2, by));   if (hasDepth) depthOut.write(half4(half(d20)), uint2(bx+2, by)); }
    if (bx+3 < w && by < h)   { colorOut.write(half4(c30, 1.0h - t30), uint2(bx+3, by));   if (hasDepth) depthOut.write(half4(half(d30)), uint2(bx+3, by)); }
    if (bx < w && by+1 < h)   { colorOut.write(half4(c01, 1.0h - t01), uint2(bx, by+1));   if (hasDepth) depthOut.write(half4(half(d01)), uint2(bx, by+1)); }
    if (bx+1 < w && by+1 < h) { colorOut.write(half4(c11, 1.0h - t11), uint2(bx+1, by+1)); if (hasDepth) depthOut.write(half4(half(d11)), uint2(bx+1, by+1)); }
    if (bx+2 < w && by+1 < h) { colorOut.write(half4(c21, 1.0h - t21), uint2(bx+2, by+1)); if (hasDepth) depthOut.write(half4(half(d21)), uint2(bx+2, by+1)); }
    if (bx+3 < w && by+1 < h) { colorOut.write(half4(c31, 1.0h - t31), uint2(bx+3, by+1)); if (hasDepth) depthOut.write(half4(half(d31)), uint2(bx+3, by+1)); }
}

kernel void localScatterSimd16(
    const device ProjectedGaussian* compacted [[buffer(0)]],
    const device TileAssignmentHeader* header [[buffer(1)]],
    device atomic_uint* tileCounters [[buffer(2)]],
    device ushort* depthKeys16 [[buffer(3)]],      // 16-bit depth keys
    device uint* globalIndices [[buffer(4)]],       // local → compacted gaussian idx for render
    constant uint& tilesX [[buffer(5)]],
    constant uint& maxPerTile [[buffer(6)]],
    constant int& tileWidth [[buffer(7)]],
    constant int& tileHeight [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint simdLane [[thread_index_in_simdgroup]],
    uint simdGroupId [[simdgroup_index_in_threadgroup]],
    uint tgId [[threadgroup_position_in_grid]]
) {
    uint visibleCount = atomic_load_explicit((const device atomic_uint*)&header->totalAssignments, memory_order_relaxed);

    uint simdGroupsPerTg = 4;
    uint globalSimdId = tgId * simdGroupsPerTg + simdGroupId;
    uint gaussianIdx = globalSimdId;

    if (gaussianIdx >= visibleCount) return;

    // Lane 0 loads gaussian data
    float3 conic = 0;
    float2 center = 0;
    float opacity = 0;
    float2 obbExtents = 0;
    float radius = 0;
    ushort depth16 = 0;
    int minTX = 0, minTY = 0, maxTX = 0, maxTY = 0;

    if (simdLane == 0) {
        ProjectedGaussian g = compacted[gaussianIdx];
        conic = float3(g.conicA, g.conicB, g.conicC);
        center = float2(g.posX, g.posY);
        opacity = float(g.opacity);
        // Read pre-computed OBB extents
        obbExtents = float2(g.obbExtentX, g.obbExtentY);
        radius = max(obbExtents.x, obbExtents.y);
        // 16-bit depth key: reinterpret half bits, XOR sign for correct unsigned sort order
        // (matches GlobalShaders.metal approach)
        depth16 = as_type<ushort>(g.depth) ^ 0x8000u;
        minTX = g.minTile.x;
        minTY = g.minTile.y;
        maxTX = g.maxTile.x;
        maxTY = g.maxTile.y;
    }

    // Broadcast to all lanes via simd_shuffle
    conic.x = simd_shuffle(conic.x, 0);
    conic.y = simd_shuffle(conic.y, 0);
    conic.z = simd_shuffle(conic.z, 0);
    center.x = simd_shuffle(center.x, 0);
    center.y = simd_shuffle(center.y, 0);
    opacity = simd_shuffle(opacity, 0);
    obbExtents.x = simd_shuffle(obbExtents.x, 0);
    obbExtents.y = simd_shuffle(obbExtents.y, 0);
    radius = simd_shuffle(radius, 0);
    depth16 = simd_shuffle(depth16, 0);
    minTX = simd_shuffle(minTX, 0);
    minTY = simd_shuffle(minTY, 0);
    maxTX = simd_shuffle(maxTX, 0);
    maxTY = simd_shuffle(maxTY, 0);

    int tileRangeX = maxTX - minTX;
    int tileRangeY = maxTY - minTY;
    int totalTiles = tileRangeX * tileRangeY;

    // Process tiles in strided pattern for better cache locality
    for (int tileIdx = int(simdLane); tileIdx < totalTiles; tileIdx += 32) {
        int localY = tileIdx / tileRangeX;
        int localX = tileIdx % tileRangeX;
        int tx = minTX + localX;
        int ty = minTY + localY;

        int2 pixMin = int2(tx * tileWidth, ty * tileHeight);
        int2 pixMax = int2(pixMin.x + tileWidth - 1, pixMin.y + tileHeight - 1);

        bool valid = intersectsTile(pixMin, pixMax, center, conic, opacity, radius, obbExtents);

        // SIMD early-out: skip iteration if no lanes are valid
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

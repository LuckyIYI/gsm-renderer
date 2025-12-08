#include <metal_stdlib>
#include <metal_atomic>
#include "BridgingTypes.h"
#include "GaussianShared.h"
#include "GaussianPrefixScan.h"
#include "RadixSortHelpers.h"
using namespace metal;

inline half2 getMean(const device GaussianRenderData& g) { return half2(g.meanX, g.meanY); }
inline half4 getConic(const device GaussianRenderData& g) { return half4(g.conicA, g.conicB, g.conicC, g.conicD); }
inline half3 getColor(const device GaussianRenderData& g) { return half3(g.colorR, g.colorG, g.colorB); }

inline half2 getMean(const thread GaussianRenderData& g) { return half2(g.meanX, g.meanY); }
inline half4 getConic(const thread GaussianRenderData& g) { return half4(g.conicA, g.conicB, g.conicC, g.conicD); }
inline half3 getColor(const thread GaussianRenderData& g) { return half3(g.colorR, g.colorG, g.colorB); }


struct DepthFirstParams {
    uint gaussianCount;
    uint visibleCount;
    uint tilesX;
    uint tilesY;
    uint tileWidth;
    uint tileHeight;
    uint maxAssignments;
    uint padding;
};

struct DepthFirstHeader {
    uint visibleCount;
    uint totalInstances;
    uint paddedVisibleCount;
    uint paddedInstanceCount;
    uint overflow;
    uint padding0;
    uint padding1;
    uint padding2;
};


template <typename PackedWorldT, typename HarmonicT, typename RenderDataT>
kernel void depthFirstPreprocessKernel(
    const device PackedWorldT* worldGaussians [[buffer(0)]],
    const device HarmonicT* harmonics [[buffer(1)]],
    device RenderDataT* outRenderData [[buffer(2)]],
    device int4* outBounds [[buffer(3)]],
    device uint* depthKeys [[buffer(4)]],
    device int* primitiveIndices [[buffer(5)]],
    device uint* nTouchedTiles [[buffer(6)]],
    device atomic_uint* visibleCount [[buffer(7)]],
    device atomic_uint* totalInstances [[buffer(8)]],
    constant CameraUniforms& camera [[buffer(9)]],
    constant TileBinningParams& params [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= camera.gaussianCount) { return; }

    PackedWorldT g = worldGaussians[gid];

    // Early cull: skip tiny gaussians
    float3 scale = float3(g.sx, g.sy, g.sz);
    float maxScale = max(scale.x, max(scale.y, scale.z));
    if (maxScale < 0.0005f) {
        // NOTE: Do NOT write to depthKeys/primitiveIndices here!
        // Those buffers use compacted indexing (0..visibleCount-1)
        // Writing at gid would race with compacted writes from visible gaussians
        nTouchedTiles[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    float3 position = float3(g.px, g.py, g.pz);
    float4 viewPos4 = camera.viewMatrix * float4(position, 1.0f);
    float depth = safeDepthComponent(viewPos4.z);

    if (depth <= camera.nearPlane) {
        nTouchedTiles[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    float4 clip = camera.projectionMatrix * viewPos4;
    if (fabs(clip.w) < 1e-6f) {
        nTouchedTiles[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    float ndcX = clip.x / clip.w;
    float ndcY = clip.y / clip.w;
    float px = ((ndcX + 1.0f) * camera.width - 1.0f) * 0.5f;
    float py = ((ndcY + 1.0f) * camera.height - 1.0f) * 0.5f;

    float opacity = float(g.opacity);
    if (opacity < params.alphaThreshold) {
        nTouchedTiles[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    float4 quat = normalizeQuaternion(getRotation(g));
    float3x3 cov3d = buildCovariance3D(scale, quat);

    float3 viewRow0 = float3(camera.viewMatrix[0][0], camera.viewMatrix[1][0], camera.viewMatrix[2][0]);
    float3 viewRow1 = float3(camera.viewMatrix[0][1], camera.viewMatrix[1][1], camera.viewMatrix[2][1]);
    float3 viewRow2 = float3(camera.viewMatrix[0][2], camera.viewMatrix[1][2], camera.viewMatrix[2][2]);
    float3x3 viewRot = matrixFromRows(viewRow0, viewRow1, viewRow2);

    float2x2 cov2d = projectCovariance(cov3d, viewPos4.xyz, viewRot, camera.focalX, camera.focalY, camera.width, camera.height);
    float4 conic;
    float radius;
    computeConicAndRadius(cov2d, conic, radius);

    if (radius < 0.5f) {
        nTouchedTiles[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    // Total ink filter
    float det = conic.x * conic.z - conic.y * conic.y;
    float totalInk = opacity * 6.283185f / sqrt(max(det, 1e-6f));
    float depthFactor = 1.0 - pow(saturate((camera.farPlane - depth) / (camera.farPlane - camera.nearPlane)), 2.0);
    float adjustedThreshold = depthFactor * params.totalInkThreshold;
    if (totalInk < adjustedThreshold) {
        nTouchedTiles[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    // OBB extents for tighter bounds
    float2 obbExtents = computeOBBExtents(cov2d, 3.0f);

    // Off-screen culling
    if (px + obbExtents.x < 0.0f || px - obbExtents.x > camera.width ||
        py + obbExtents.y < 0.0f || py - obbExtents.y > camera.height) {
        nTouchedTiles[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    // Compute SH color
    float3 color = computeSHColor(harmonics, gid, position, camera.cameraCenter, camera.shComponents);
    color = max(color + float3(0.5f), float3(0.0f));

    // Write render data
    RenderDataT renderData;
    renderData.meanX = half(px);
    renderData.meanY = half(py);
    renderData.conicA = half(conic.x);
    renderData.conicB = half(conic.y);
    renderData.conicC = half(conic.z);
    renderData.conicD = half(conic.w);
    renderData.colorR = half(color.x);
    renderData.colorG = half(color.y);
    renderData.colorB = half(color.z);
    renderData.opacity = half(opacity);
    renderData.depth = half(depth);
    outRenderData[gid] = renderData;

    // Compute tile bounds
    float xmin = px - obbExtents.x;
    float xmax = px + obbExtents.x;
    float ymin = py - obbExtents.y;
    float ymax = py + obbExtents.y;

    float maxW = camera.width - 1.0f;
    float maxH = camera.height - 1.0f;

    xmin = clamp(xmin, 0.0f, maxW);
    xmax = clamp(xmax, 0.0f, maxW);
    ymin = clamp(ymin, 0.0f, maxH);
    ymax = clamp(ymax, 0.0f, maxH);

    int minTileX = int(floor(xmin / float(params.tileWidth)));
    int maxTileX = int(ceil(xmax / float(params.tileWidth))) - 1;
    int minTileY = int(floor(ymin / float(params.tileHeight)));
    int maxTileY = int(ceil(ymax / float(params.tileHeight))) - 1;

    minTileX = max(minTileX, 0);
    minTileY = max(minTileY, 0);
    maxTileX = min(maxTileX, int(params.tilesX) - 1);
    maxTileY = min(maxTileY, int(params.tilesY) - 1);

    outBounds[gid] = int4(minTileX, maxTileX, minTileY, maxTileY);

    // Count touched tiles from AABB (fast)
    // Note: Could add intersection testing but it's expensive in preprocess
    // since this runs for ALL gaussians, not just visible ones
    uint tilesX_count = (minTileX <= maxTileX) ? uint(maxTileX - minTileX + 1) : 0;
    uint tilesY_count = (minTileY <= maxTileY) ? uint(maxTileY - minTileY + 1) : 0;
    uint touchedCount = tilesX_count * tilesY_count;

    // Atomic compaction: get compacted write position for depth sort buffers
    // This ensures visible gaussians are contiguous in the first visibleCount positions
    uint compactIdx = atomic_fetch_add_explicit(visibleCount, 1u, memory_order_relaxed);

    // Write depth key and primitive index to COMPACTED positions (for depth sort)
    half depthHalf = half(depth);
    uint depthBits = uint(as_type<ushort>(depthHalf)) ^ 0x8000u;  // XOR sign bit for unsigned ordering
    depthKeys[compactIdx] = depthBits;
    primitiveIndices[compactIdx] = int(gid);  // Store original gaussian index

    // Write nTouchedTiles to ORIGINAL position (indexed by original gid for applyDepthOrdering)
    nTouchedTiles[gid] = touchedCount;

    // Count total instances
    atomic_fetch_add_explicit(totalInstances, touchedCount, memory_order_relaxed);
}

// Template instantiations
template [[host_name("depthFirstPreprocessKernel")]]
kernel void depthFirstPreprocessKernel<PackedWorldGaussian, float, GaussianRenderData>(
    const device PackedWorldGaussian*, const device float*,
    device GaussianRenderData*, device int4*, device uint*, device int*,
    device uint*, device atomic_uint*, device atomic_uint*,
    constant CameraUniforms&, constant TileBinningParams&, uint);

template [[host_name("depthFirstPreprocessKernelHalf")]]
kernel void depthFirstPreprocessKernel<PackedWorldGaussianHalf, half, GaussianRenderData>(
    const device PackedWorldGaussianHalf*, const device half*,
    device GaussianRenderData*, device int4*, device uint*, device int*,
    device uint*, device atomic_uint*, device atomic_uint*,
    constant CameraUniforms&, constant TileBinningParams&, uint);

kernel void applyDepthOrderingKernel(
    const device int* sortedPrimitiveIndices [[buffer(0)]],
    const device uint* nTouchedTiles [[buffer(1)]],
    device uint* orderedTileCounts [[buffer(2)]],
    constant DepthFirstHeader& header [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint visibleCount = header.visibleCount;
    if (gid >= visibleCount) { return; }

    int originalIdx = sortedPrimitiveIndices[gid];
    if (originalIdx < 0) {
        orderedTileCounts[gid] = 0;
        return;
    }

    orderedTileCounts[gid] = nTouchedTiles[originalIdx];
}

kernel void createInstancesKernel(
    const device int* sortedPrimitiveIndices [[buffer(0)]],
    const device uint* instanceOffsets [[buffer(1)]],
    const device int4* tileBounds [[buffer(2)]],
    const device GaussianRenderData* renderData [[buffer(3)]],
    device ushort* instanceTileIds [[buffer(4)]],  // ushort is enough for tiles (max 65535)
    device int* instanceGaussianIndices [[buffer(5)]],
    constant DepthFirstParams& params [[buffer(6)]],
    constant DepthFirstHeader& header [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint visibleCount = header.visibleCount;
    if (gid >= visibleCount) { return; }

    int originalIdx = sortedPrimitiveIndices[gid];
    if (originalIdx < 0) { return; }

    int4 bounds = tileBounds[originalIdx];
    int minTX = bounds.x;
    int maxTX = bounds.y;
    int minTY = bounds.z;
    int maxTY = bounds.w;

    if (minTX > maxTX || minTY > maxTY) { return; }

    // Get the write offset for this gaussian (from prefix sum)
    uint writeOffset = instanceOffsets[gid];

    // Write instances for all tiles in bounding box (must match nTouchedTiles count)
    for (int ty = minTY; ty <= maxTY; ty++) {
        for (int tx = minTX; tx <= maxTX; tx++) {
            if (writeOffset < params.maxAssignments) {
                ushort tileId = ushort(ty * int(params.tilesX) + tx);
                instanceTileIds[writeOffset] = tileId;
                instanceGaussianIndices[writeOffset] = originalIdx;
                writeOffset++;
            }
        }
    }
}

kernel void tileRadixHistogramKernel(
    device const ushort* inputTileIds [[buffer(0)]],
    device uint* histFlat [[buffer(1)]],
    constant DepthFirstHeader& header [[buffer(2)]],
    constant uint& currentDigit [[buffer(3)]],  // 0 or 8 for 16-bit keys
    uint gridSize [[threadgroups_per_grid]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint elementsPerBlock = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint baseId = groupId * elementsPerBlock;

    uint paddedCount = header.paddedInstanceCount;
    uint count = header.totalInstances;

    uint bufferRemaining = (baseId < paddedCount) ? (paddedCount - baseId) : 0;
    uint assignmentsRemain = (baseId < count) ? (count - baseId) : 0;
    uint available = min(bufferRemaining, assignmentsRemain);

    threadgroup uint localHist[RADIX_BUCKETS];

    for (uint i = 0; i < (RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE; ++i) {
        uint bin = localId + i * RADIX_BLOCK_SIZE;
        if (bin < RADIX_BUCKETS) {
            localHist[bin] = 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ushort keys[RADIX_GRAIN_SIZE];
    ushort sentinelKey = 0xFFFFu;

    load_blocked_local_from_global<RADIX_GRAIN_SIZE>(
        keys,
        &inputTileIds[baseId],
        localId,
        available,
        sentinelKey);

    volatile threadgroup atomic_uint* atomicHist =
        reinterpret_cast<volatile threadgroup atomic_uint*>(localHist);

    for (ushort i = 0; i < RADIX_GRAIN_SIZE; ++i) {
        uint offsetInBlock = localId * RADIX_GRAIN_SIZE + i;
        uint globalIdx = baseId + offsetInBlock;

        if (globalIdx < count) {
            uchar bin = (uchar)((keys[i] >> currentDigit) & 0xFFu);
            atomic_fetch_add_explicit(&atomicHist[bin], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write histogram in TRANSPOSED layout
    for (uint i = 0; i < (RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE; ++i) {
        uint bin = localId + i * RADIX_BLOCK_SIZE;
        if (bin < RADIX_BUCKETS) {
            histFlat[bin * gridSize + groupId] = localHist[bin];
        }
    }
}

// Tile scan blocks kernel
kernel void tileRadixScanBlocksKernel(
    device uint* histFlat [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant DepthFirstHeader& header [[buffer(2)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    uint paddedCount = header.paddedInstanceCount;
    const uint valuesPerGroup = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint gridSize = (paddedCount + valuesPerGroup - 1u) / valuesPerGroup;
    uint histLength = gridSize * RADIX_BUCKETS;

    uint baseIndex = groupId * RADIX_BLOCK_SIZE;
    threadgroup uint shared[RADIX_BLOCK_SIZE];

    uint value = (baseIndex + localId < histLength) ? histFlat[baseIndex + localId] : 0u;
    uint scanned = threadgroup_raking_prefix_exclusive_sum<RADIX_BLOCK_SIZE>(value, shared, localId);

    if (baseIndex + localId < histLength) {
        histFlat[baseIndex + localId] = scanned;
    }
    if (localId == RADIX_BLOCK_SIZE - 1) {
        blockSums[groupId] = scanned + value;
    }
}

// Tile exclusive scan kernel (handles up to 16384 block sums = ~16M instances)
#define TILE_RADIX_SCAN_GRAIN 64

kernel void tileRadixExclusiveScanKernel(
    device uint* blockSums [[buffer(0)]],
    device uint* scannedHist [[buffer(1)]],
    device const uint* histFlat [[buffer(2)]],
    constant DepthFirstHeader& header [[buffer(3)]],
    ushort localId [[thread_position_in_threadgroup]],
    threadgroup uint* shared [[threadgroup(0)]]
) {
    uint paddedCount = header.paddedInstanceCount;
    const uint valuesPerGroup = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint gridSize = (paddedCount + valuesPerGroup - 1u) / valuesPerGroup;
    uint numBlockSums = (gridSize * RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE;

    uint elemsPerThread = (numBlockSums + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE;
    elemsPerThread = min(elemsPerThread, (uint)TILE_RADIX_SCAN_GRAIN);

    uint localVals[TILE_RADIX_SCAN_GRAIN];
    uint localSum = 0u;
    uint threadBase = localId * elemsPerThread;

    for (uint i = 0u; i < elemsPerThread; ++i) {
        uint idx = threadBase + i;
        uint val = (idx < numBlockSums) ? blockSums[idx] : 0u;
        localVals[i] = val;
        localSum += val;
    }

    uint scannedSum = radix_threadgroup_prefix_scan<RADIX_SCAN_TYPE_EXCLUSIVE>(localSum, shared, localId, radix_sum_op<uint>());

    uint running = scannedSum;
    for (uint i = 0u; i < elemsPerThread; ++i) {
        uint idx = threadBase + i;
        if (idx < numBlockSums) {
            blockSums[idx] = running;
            running += localVals[i];
        }
    }
}

// Tile apply offsets kernel
kernel void tileRadixApplyOffsetsKernel(
    device const uint* histFlat [[buffer(0)]],
    device const uint* blockSums [[buffer(1)]],
    device uint* scannedHistFlat [[buffer(2)]],
    constant DepthFirstHeader& header [[buffer(3)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    uint paddedCount = header.paddedInstanceCount;
    const uint valuesPerGroup = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint gridSize = (paddedCount + valuesPerGroup - 1u) / valuesPerGroup;
    uint histLength = gridSize * RADIX_BUCKETS;

    uint baseIndex = groupId * RADIX_BLOCK_SIZE;
    uint blockOffset = blockSums[groupId];

    if (baseIndex + localId < histLength) {
        scannedHistFlat[baseIndex + localId] = histFlat[baseIndex + localId] + blockOffset;
    }
}

// Tile scatter kernel - stable scatter preserving depth order (16-bit tile IDs)
kernel void tileRadixScatterKernel(
    device const ushort* inputTileIds [[buffer(0)]],
    device const int* inputIndices [[buffer(1)]],
    device const uint* offsetsFlat [[buffer(2)]],
    device ushort* outputTileIds [[buffer(3)]],
    device int* outputIndices [[buffer(4)]],
    constant DepthFirstHeader& header [[buffer(5)]],
    constant uint& currentDigit [[buffer(6)]],  // 0 or 8 for 16-bit keys
    uint groupId [[threadgroup_position_in_grid]],
    uint gridSize [[threadgroups_per_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint elementsPerBlock = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint baseId = groupId * elementsPerBlock;

    uint paddedCount = header.paddedInstanceCount;
    uint count = header.totalInstances;

    uint bufferRemaining = (baseId < paddedCount) ? (paddedCount - baseId) : 0;
    uint assignmentsRemain = (baseId < count) ? (count - baseId) : 0;
    uint available = min(bufferRemaining, assignmentsRemain);

    uint keySentinel = 0xFFFFu;  // 16-bit sentinel

    // Load as uint for sorting (ushort zero-extended)
    uint keys[RADIX_GRAIN_SIZE];
    int payloads[RADIX_GRAIN_SIZE];

    // Manual load for ushort -> uint conversion
    for (ushort i = 0; i < RADIX_GRAIN_SIZE; ++i) {
        uint idx = baseId + localId + i * RADIX_BLOCK_SIZE;
        if (idx < available + baseId && idx < count) {
            keys[i] = uint(inputTileIds[idx]);
            payloads[i] = inputIndices[idx];
        } else {
            keys[i] = keySentinel;
            payloads[i] = -1;
        }
    }

    threadgroup uint globalBinBase[RADIX_BUCKETS];
    threadgroup RadixKeyPayload tgKp[RADIX_BLOCK_SIZE];
    threadgroup ushort tgShort[RADIX_BLOCK_SIZE];

    // Read from TRANSPOSED histogram layout
    for (uint i = 0; i < (RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE; ++i) {
        uint bin = localId + i * RADIX_BLOCK_SIZE;
        if (bin < RADIX_BUCKETS) {
            globalBinBase[bin] = offsetsFlat[bin * gridSize + groupId];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort chunk = 0; chunk < RADIX_GRAIN_SIZE; ++chunk) {
        RadixKeyPayload kp;
        kp.key = keys[chunk];
        kp.payload = uint(payloads[chunk]);

        // Use the existing radix_partial_sort with digit extraction
        // Note: currentDigit is in bits (0 or 8), convert to digit index (0 or 1)
        ushort digitIndex = (ushort)(currentDigit / 8);
        kp = radix_partial_sort(kp, tgKp, localId, digitIndex);

        ushort myBin = (ushort)((kp.key >> currentDigit) & 0xFFu);

        uchar headFlag = flag_head_discontinuity<RADIX_BLOCK_SIZE>(myBin, tgShort, localId);
        ushort runStart = radix_threadgroup_prefix_scan<RADIX_SCAN_TYPE_INCLUSIVE>(
            headFlag ? localId : (ushort)0, tgShort, localId, radix_max_op<ushort>());
        ushort localOffset = localId - runStart;

        uchar tailFlag = flag_tail_discontinuity<RADIX_BLOCK_SIZE>(myBin, tgShort, localId);

        bool isValid = (kp.key != keySentinel);
        if (isValid) {
            uint dst = globalBinBase[myBin] + localOffset;
            outputTileIds[dst] = ushort(kp.key);
            outputIndices[dst] = int(kp.payload);
        }

        if (tailFlag && isValid) {
            globalBinBase[myBin] += localOffset + 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void extractTileRangesKernel(
    const device ushort* sortedTileIds [[buffer(0)]],  // 16-bit tile IDs
    device GaussianHeader* tileHeaders [[buffer(1)]],
    device uint* activeTiles [[buffer(2)]],
    device atomic_uint* activeTileCount [[buffer(3)]],
    constant DepthFirstHeader& dfHeader [[buffer(4)]],
    constant uint& tileCount [[buffer(5)]],
    uint tile [[thread_position_in_grid]]
) {
    if (tile >= tileCount) { return; }

    uint totalInstances = dfHeader.totalInstances;
    if (totalInstances == 0) {
        GaussianHeader emptyHeader;
        emptyHeader.offset = 0;
        emptyHeader.count = 0;
        tileHeaders[tile] = emptyHeader;
        return;
    }

    // Binary search for start of tile
    uint left = 0;
    uint right = totalInstances;
    while (left < right) {
        uint mid = (left + right) >> 1;
        uint midTile = uint(sortedTileIds[mid]);
        if (midTile < tile) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    uint start = left;

    // Binary search for end of tile
    left = start;
    right = totalInstances;
    while (left < right) {
        uint mid = (left + right) >> 1;
        uint midTile = uint(sortedTileIds[mid]);
        if (midTile <= tile) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    uint end = left;

    GaussianHeader tileHeader;
    tileHeader.offset = start;
    tileHeader.count = end > start ? (end - start) : 0;
    tileHeaders[tile] = tileHeader;

    // Compact active tiles
    if (tileHeader.count > 0) {
        uint index = atomic_fetch_add_explicit(activeTileCount, 1u, memory_order_relaxed);
        activeTiles[index] = tile;
    }
}

kernel void resetDepthFirstStateKernel(
    device DepthFirstHeader* header [[buffer(0)]],
    device uint* activeTileCount [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        header->visibleCount = 0;
        header->totalInstances = 0;
        header->paddedVisibleCount = 0;
        header->paddedInstanceCount = 0;
        header->overflow = 0;
        *activeTileCount = 0;
    }
}

kernel void depthSortHistogramKernel(
    device const RadixKeyType* inputKeys [[buffer(0)]],
    device uint* histFlat [[buffer(1)]],
    constant uint& currentDigit [[buffer(3)]],
    constant DepthFirstHeader& header [[buffer(4)]],
    uint gridSize [[threadgroups_per_grid]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint elementsPerBlock = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint baseId = groupId * elementsPerBlock;

    uint paddedCount = header.paddedVisibleCount;  // For depth sort
    uint count = header.visibleCount;              // For depth sort

    uint bufferRemaining = (baseId < paddedCount) ? (paddedCount - baseId) : 0;
    uint assignmentsRemain = (baseId < count) ? (count - baseId) : 0;
    uint available = min(bufferRemaining, assignmentsRemain);

    threadgroup uint localHist[RADIX_BUCKETS];

    for (uint i = 0; i < (RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE; ++i) {
        uint bin = localId + i * RADIX_BLOCK_SIZE;
        if (bin < RADIX_BUCKETS) {
            localHist[bin] = 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    RadixKeyType keys[RADIX_GRAIN_SIZE];
    RadixKeyType sentinelKey = (RadixKeyType)0xFFFFFFFFu;

    load_blocked_local_from_global<RADIX_GRAIN_SIZE>(
        keys,
        &inputKeys[baseId],
        localId,
        available,
        sentinelKey);

    volatile threadgroup atomic_uint* atomicHist =
        reinterpret_cast<volatile threadgroup atomic_uint*>(localHist);

    for (ushort i = 0; i < RADIX_GRAIN_SIZE; ++i) {
        uint offsetInBlock = localId * RADIX_GRAIN_SIZE + i;
        uint globalIdx = baseId + offsetInBlock;

        if (globalIdx < count) {
            uchar bin = (uchar)value_to_key_at_digit(keys[i], (ushort)currentDigit);
            atomic_fetch_add_explicit(&atomicHist[bin], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write histogram in TRANSPOSED layout: [bin0_groups..., bin1_groups..., ...]
    // This allows linear scan to produce correct global offsets for multi-block radix sort
    for (uint i = 0; i < (RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE; ++i) {
        uint bin = localId + i * RADIX_BLOCK_SIZE;
        if (bin < RADIX_BUCKETS) {
            histFlat[bin * gridSize + groupId] = localHist[bin];
        }
    }
}

// Depth-sort-specific scan blocks kernel (reads visibleCount/paddedVisibleCount from DepthFirstHeader)
kernel void depthSortScanBlocksKernel(
    device uint* histFlat [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant DepthFirstHeader& header [[buffer(2)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    uint paddedCount = header.paddedVisibleCount;  // For depth sort, use padded visible count
    const uint valuesPerGroup = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint gridSize = (paddedCount + valuesPerGroup - 1u) / valuesPerGroup;
    uint histLength = gridSize * RADIX_BUCKETS;

    uint baseIndex = groupId * RADIX_BLOCK_SIZE;
    threadgroup uint shared[RADIX_BLOCK_SIZE];

    uint value = (baseIndex + localId < histLength) ? histFlat[baseIndex + localId] : 0u;
    uint scanned = threadgroup_raking_prefix_exclusive_sum<RADIX_BLOCK_SIZE>(value, shared, localId);

    if (baseIndex + localId < histLength) {
        histFlat[baseIndex + localId] = scanned;
    }
    if (localId == RADIX_BLOCK_SIZE - 1) {
        blockSums[groupId] = scanned + value;
    }
}

// Depth-sort-specific exclusive scan kernel (handles up to 8192 block sums)
#define RADIX_SCAN_GRAIN 32

kernel void depthSortExclusiveScanKernel(
    device uint* blockSums [[buffer(0)]],
    constant DepthFirstHeader& header [[buffer(1)]],
    ushort localId [[thread_position_in_threadgroup]],
    threadgroup uint* shared [[threadgroup(0)]]
) {
    uint paddedCount = header.paddedVisibleCount;  // For depth sort
    const uint valuesPerGroup = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint gridSize = (paddedCount + valuesPerGroup - 1u) / valuesPerGroup;
    uint numBlockSums = (gridSize * RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE;

    // Calculate elements per thread (up to RADIX_SCAN_GRAIN)
    uint elemsPerThread = (numBlockSums + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE;
    elemsPerThread = min(elemsPerThread, (uint)RADIX_SCAN_GRAIN);

    // Load multiple elements and compute local sum
    uint localVals[RADIX_SCAN_GRAIN];
    uint localSum = 0u;
    uint threadBase = localId * elemsPerThread;

    for (uint i = 0u; i < elemsPerThread; ++i) {
        uint idx = threadBase + i;
        uint val = (idx < numBlockSums) ? blockSums[idx] : 0u;
        localVals[i] = val;
        localSum += val;
    }

    // Scan the per-thread sums using helper (exclusive scan of 256 values)
    uint scannedSum = radix_threadgroup_prefix_scan<RADIX_SCAN_TYPE_EXCLUSIVE>(localSum, shared, localId, radix_sum_op<uint>());

    // Write back: each element gets scannedSum + prefix of local values
    uint running = scannedSum;
    for (uint i = 0u; i < elemsPerThread; ++i) {
        uint idx = threadBase + i;
        if (idx < numBlockSums) {
            blockSums[idx] = running;
            running += localVals[i];
        }
    }
}

// Depth-sort-specific apply offsets kernel
kernel void depthSortApplyOffsetsKernel(
    device const uint* histFlat [[buffer(0)]],
    device const uint* blockSums [[buffer(1)]],
    device uint* scannedHistFlat [[buffer(2)]],
    constant DepthFirstHeader& header [[buffer(3)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]],
    threadgroup uint* shared [[threadgroup(0)]]
) {
    uint paddedCount = header.paddedVisibleCount;  // For depth sort
    const uint valuesPerGroup = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint gridSize = (paddedCount + valuesPerGroup - 1u) / valuesPerGroup;
    uint histLength = gridSize * RADIX_BUCKETS;

    uint baseIndex = groupId * RADIX_BLOCK_SIZE;
    uint blockOffset = blockSums[groupId];

    if (baseIndex + localId < histLength) {
        scannedHistFlat[baseIndex + localId] = histFlat[baseIndex + localId] + blockOffset;
    }
}

// Depth-sort-specific scatter kernel
kernel void depthSortScatterKernel(
    device RadixKeyType* outputKeys [[buffer(0)]],
    device const RadixKeyType* inputKeys [[buffer(1)]],
    device int* outputPayload [[buffer(2)]],
    device const int* inputPayload [[buffer(3)]],
    device const uint* offsetsFlat [[buffer(5)]],
    constant uint& currentDigit [[buffer(6)]],
    constant DepthFirstHeader& header [[buffer(7)]],
    uint groupId [[threadgroup_position_in_grid]],
    uint gridSize [[threadgroups_per_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint elementsPerBlock = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint baseId = groupId * elementsPerBlock;

    uint paddedCount = header.paddedVisibleCount;  // For depth sort
    uint count = header.visibleCount;              // For depth sort

    uint bufferRemaining = (baseId < paddedCount) ? (paddedCount - baseId) : 0;
    uint assignmentsRemain = (baseId < count) ? (count - baseId) : 0;
    uint available = min(bufferRemaining, assignmentsRemain);

    RadixKeyType keySentinel = (RadixKeyType)0xFFFFFFFFu;

    RadixKeyType keys[RADIX_GRAIN_SIZE];
    int payloads[RADIX_GRAIN_SIZE];
    load_striped_local_from_global<RADIX_GRAIN_SIZE, RADIX_BLOCK_SIZE>(keys, &inputKeys[baseId], localId, available, keySentinel);
    load_striped_local_from_global<RADIX_GRAIN_SIZE, RADIX_BLOCK_SIZE>(payloads, &inputPayload[baseId], localId, available, -1);

    threadgroup uint globalBinBase[RADIX_BUCKETS];
    threadgroup RadixKeyPayload tgKp[RADIX_BLOCK_SIZE];
    threadgroup ushort tgShort[RADIX_BLOCK_SIZE];

    // Read from TRANSPOSED histogram layout: [bin0_groups..., bin1_groups..., ...]
    for (uint i = 0; i < (RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE; ++i) {
        uint bin = localId + i * RADIX_BLOCK_SIZE;
        if (bin < RADIX_BUCKETS) {
            globalBinBase[bin] = offsetsFlat[bin * gridSize + groupId];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort chunk = 0; chunk < RADIX_GRAIN_SIZE; ++chunk) {
        RadixKeyPayload kp;
        kp.key = keys[chunk];
        kp.payload = uint(payloads[chunk]);

        ushort myBin = value_to_key_at_digit(kp.key, (ushort)currentDigit);

        kp = radix_partial_sort(kp, tgKp, localId, (ushort)currentDigit);

        myBin = value_to_key_at_digit(kp.key, (ushort)currentDigit);

        uchar headFlag = flag_head_discontinuity<RADIX_BLOCK_SIZE>(myBin, tgShort, localId);
        ushort runStart = radix_threadgroup_prefix_scan<RADIX_SCAN_TYPE_INCLUSIVE>(
            headFlag ? localId : (ushort)0, tgShort, localId, radix_max_op<ushort>());
        ushort localOffset = localId - runStart;

        uchar tailFlag = flag_tail_discontinuity<RADIX_BLOCK_SIZE>(myBin, tgShort, localId);

        bool isValid = (kp.key != keySentinel);
        if (isValid) {
            uint dst = globalBinBase[myBin] + localOffset;
            outputKeys[dst] = kp.key;
            outputPayload[dst] = int(kp.payload);
        }

        if (tailFlag && isValid) {
            globalBinBase[myBin] += localOffset + 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define RENDER_DF_TG_WIDTH 8
#define RENDER_DF_TG_HEIGHT 8
#define RENDER_DF_PIXELS_X 2
#define RENDER_DF_PIXELS_Y 2

kernel void depthFirstRender(
    const device GaussianHeader* headers [[buffer(0)]],
    const device GaussianRenderData* gaussians [[buffer(1)]],
    const device int* sortedGaussianIndices [[buffer(2)]],
    const device uint* activeTiles [[buffer(3)]],
    const device uint* activeTileCount [[buffer(4)]],
    texture2d<half, access::write> colorOut [[texture(0)]],
    texture2d<half, access::write> depthOut [[texture(1)]],
    constant RenderParams& params [[buffer(5)]],
    uint2 groupId [[threadgroup_position_in_grid]],
    uint2 localId [[thread_position_in_threadgroup]]
) {
    uint tileIdx = groupId.x;
    if (tileIdx >= *activeTileCount) return;

    uint tile = activeTiles[tileIdx];
    GaussianHeader hdr = headers[tile];

    uint tileX = tile % params.tilesX;
    uint tileY = tile / params.tilesX;

    // 16x16 tile, 8x8 threads, 2x2 pixels per thread
    uint baseX = tileX * 16 + localId.x * RENDER_DF_PIXELS_X;
    uint baseY = tileY * 16 + localId.y * RENDER_DF_PIXELS_Y;

    half2 pos00 = half2(half(baseX + 0), half(baseY + 0));
    half2 pos10 = half2(half(baseX + 1), half(baseY + 0));
    half2 pos01 = half2(half(baseX + 0), half(baseY + 1));
    half2 pos11 = half2(half(baseX + 1), half(baseY + 1));

    half trans00 = 1.0h, trans10 = 1.0h;
    half trans01 = 1.0h, trans11 = 1.0h;

    half3 color00 = 0, color10 = 0;
    half3 color01 = 0, color11 = 0;

    half depth00 = 0, depth10 = 0;
    half depth01 = 0, depth11 = 0;

    uint start = hdr.offset;
    uint count = hdr.count;

    for (uint i = 0; i < count; i++) {
        half maxTrans = max(max(trans00, trans10), max(trans01, trans11));
        if (maxTrans < half(1.0h/255.0h)) break;

        int gaussianIdx = sortedGaussianIndices[start + i];
        if (gaussianIdx < 0) continue;
        GaussianRenderData g = gaussians[gaussianIdx];

        half4 gConic = getConic(g);
        half cxx = gConic.x;
        half cyy = gConic.z;
        half cxy2 = 2.0h * gConic.y;
        half opacity = g.opacity;
        half3 gColor = getColor(g);
        half gDepth = g.depth;

        half2 gMean = getMean(g);
        half2 d00 = pos00 - gMean;
        half2 d10 = pos10 - gMean;
        half2 d01 = pos01 - gMean;
        half2 d11 = pos11 - gMean;

        half p00 = d00.x*d00.x*cxx + d00.y*d00.y*cyy + d00.x*d00.y*cxy2;
        half p10 = d10.x*d10.x*cxx + d10.y*d10.y*cyy + d10.x*d10.y*cxy2;
        half p01 = d01.x*d01.x*cxx + d01.y*d01.y*cyy + d01.x*d01.y*cxy2;
        half p11 = d11.x*d11.x*cxx + d11.y*d11.y*cyy + d11.x*d11.y*cxy2;

        half a00 = min(opacity * exp(-0.5h * p00), 0.99h);
        half a10 = min(opacity * exp(-0.5h * p10), 0.99h);
        half a01 = min(opacity * exp(-0.5h * p01), 0.99h);
        half a11 = min(opacity * exp(-0.5h * p11), 0.99h);

        half4 alphas = half4(a00, a10, a01, a11);
        if (all(alphas == 0.0h)) continue;

        color00 += gColor * (a00 * trans00); depth00 += gDepth * (a00 * trans00);
        color10 += gColor * (a10 * trans10); depth10 += gDepth * (a10 * trans10);
        color01 += gColor * (a01 * trans01); depth01 += gDepth * (a01 * trans01);
        color11 += gColor * (a11 * trans11); depth11 += gDepth * (a11 * trans11);

        trans00 *= (1.0h - a00); trans10 *= (1.0h - a10);
        trans01 *= (1.0h - a01); trans11 *= (1.0h - a11);
    }

    uint w = params.width, h = params.height;
    bool hasDepth = !is_null_texture(depthOut);

    if (baseX + 0 < w && baseY + 0 < h) {
        colorOut.write(half4(color00, 1.0h - trans00), uint2(baseX + 0, baseY + 0));
        if (hasDepth) depthOut.write(half4(depth00, 0, 0, 0), uint2(baseX + 0, baseY + 0));
    }
    if (baseX + 1 < w && baseY + 0 < h) {
        colorOut.write(half4(color10, 1.0h - trans10), uint2(baseX + 1, baseY + 0));
        if (hasDepth) depthOut.write(half4(depth10, 0, 0, 0), uint2(baseX + 1, baseY + 0));
    }
    if (baseX + 0 < w && baseY + 1 < h) {
        colorOut.write(half4(color01, 1.0h - trans01), uint2(baseX + 0, baseY + 1));
        if (hasDepth) depthOut.write(half4(depth01, 0, 0, 0), uint2(baseX + 0, baseY + 1));
    }
    if (baseX + 1 < w && baseY + 1 < h) {
        colorOut.write(half4(color11, 1.0h - trans11), uint2(baseX + 1, baseY + 1));
        if (hasDepth) depthOut.write(half4(depth11, 0, 0, 0), uint2(baseX + 1, baseY + 1));
    }
}

kernel void clearRenderTexturesKernel(
    texture2d<half, access::write> colorTex [[texture(0)]],
    texture2d<half, access::write> depthTex [[texture(1)]],
    constant ClearTextureParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    half bg = 0.0h;
    colorTex.write(half4(bg, bg, bg, 1.0h), gid);
    if (!is_null_texture(depthTex)) {
        depthTex.write(half4(0.0h), gid);
    }
}

kernel void blockReduceKernel(
    device const uint* input [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant DepthFirstHeader& header [[buffer(2)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint BLOCK_SIZE = 256;
    uint baseIndex = groupId * BLOCK_SIZE;
    threadgroup uint shared[BLOCK_SIZE];

    uint count = header.visibleCount;
    uint value = (baseIndex + localId < count) ? input[baseIndex + localId] : 0u;
    uint sum = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(value, shared, localId);

    if (localId == 0) {
        blockSums[groupId] = sum;
    }
}

// Level 2 reduce: reduce every 256 block sums into a super-block sum
kernel void level2ReduceKernel(
    device const uint* blockSums [[buffer(0)]],
    device uint* level2Sums [[buffer(1)]],
    constant DepthFirstHeader& header [[buffer(2)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint BLOCK_SIZE = 256;
    uint numBlocks = (header.visibleCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint baseIndex = groupId * BLOCK_SIZE;

    threadgroup uint shared[BLOCK_SIZE];

    uint value = (baseIndex + localId < numBlocks) ? blockSums[baseIndex + localId] : 0u;
    uint sum = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(value, shared, localId);

    if (localId == 0) {
        level2Sums[groupId] = sum;
    }
}

// Level 2 scan: exclusive scan of super-block sums (fits in single block for up to 256 super-blocks = 65536 blocks = 16M elements)
kernel void level2ScanKernel(
    device uint* level2Sums [[buffer(0)]],
    constant DepthFirstHeader& header [[buffer(1)]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint BLOCK_SIZE = 256;
    threadgroup uint shared[BLOCK_SIZE];

    uint numBlocks = (header.visibleCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint numLevel2Blocks = (numBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

    uint value = (localId < numLevel2Blocks) ? level2Sums[localId] : 0u;
    uint scanned = threadgroup_raking_prefix_exclusive_sum<BLOCK_SIZE>(value, shared, localId);
    if (localId < numLevel2Blocks) {
        level2Sums[localId] = scanned;
    }
}

// Level 2 apply + block scan: apply level 2 offsets to block sums, then scan within each level 2 group
kernel void level2ApplyAndScanKernel(
    device uint* blockSums [[buffer(0)]],
    device const uint* level2Sums [[buffer(1)]],
    constant DepthFirstHeader& header [[buffer(2)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint BLOCK_SIZE = 256;
    uint numBlocks = (header.visibleCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint baseIndex = groupId * BLOCK_SIZE;

    threadgroup uint shared[BLOCK_SIZE];

    uint value = (baseIndex + localId < numBlocks) ? blockSums[baseIndex + localId] : 0u;
    uint scanned = threadgroup_raking_prefix_exclusive_sum<BLOCK_SIZE>(value, shared, localId);

    uint level2Offset = level2Sums[groupId];
    if (baseIndex + localId < numBlocks) {
        blockSums[baseIndex + localId] = scanned + level2Offset;
    }
}

kernel void blockScanKernel(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device const uint* blockOffsets [[buffer(2)]],
    constant DepthFirstHeader& header [[buffer(3)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint BLOCK_SIZE = 256;
    uint baseIndex = groupId * BLOCK_SIZE;
    threadgroup uint shared[BLOCK_SIZE];

    uint count = header.visibleCount;
    uint value = (baseIndex + localId < count) ? input[baseIndex + localId] : 0u;
    uint scanned = threadgroup_raking_prefix_exclusive_sum<BLOCK_SIZE>(value, shared, localId);

    uint blockOffset = blockOffsets[groupId];
    if (baseIndex + localId < count) {
        output[baseIndex + localId] = scanned + blockOffset;
    }
}

struct DepthFirstDispatchConfig {
    uint maxGaussians;
    uint maxInstances;
    uint radixBlockSize;
    uint radixGrainSize;
    uint instanceExpansionTGSize;
    uint prefixSumTGSize;
    uint tileRangeTGSize;
    uint tileCount;
};

// Dispatch slot indices for depth-first pipeline
constant uint DF_SLOT_DEPTH_HISTOGRAM = 0;
constant uint DF_SLOT_DEPTH_SCAN_BLOCKS = 1;
constant uint DF_SLOT_DEPTH_EXCLUSIVE = 2;
constant uint DF_SLOT_DEPTH_APPLY = 3;
constant uint DF_SLOT_DEPTH_SCATTER = 4;
constant uint DF_SLOT_APPLY_DEPTH_ORDER = 5;
constant uint DF_SLOT_PREFIX_SUM = 6;
constant uint DF_SLOT_PREFIX_LEVEL2_REDUCE = 7;
constant uint DF_SLOT_PREFIX_LEVEL2_SCAN = 8;
constant uint DF_SLOT_PREFIX_LEVEL2_APPLY = 9;
constant uint DF_SLOT_PREFIX_FINAL_SCAN = 10;
constant uint DF_SLOT_CREATE_INSTANCES = 11;
// Stable radix sort for tile IDs (preserves depth order within tiles)
constant uint DF_SLOT_TILE_HISTOGRAM = 12;
constant uint DF_SLOT_TILE_SCAN_BLOCKS = 13;
constant uint DF_SLOT_TILE_EXCLUSIVE = 14;
constant uint DF_SLOT_TILE_APPLY = 15;
constant uint DF_SLOT_TILE_SCATTER = 16;
constant uint DF_SLOT_EXTRACT_RANGES = 17;
constant uint DF_SLOT_RENDER = 18;

kernel void prepareDepthFirstDispatchKernel(
    device const uint* visibleCountAtomic [[buffer(0)]],
    device const uint* totalInstancesAtomic [[buffer(1)]],
    device DepthFirstHeader* header [[buffer(2)]],
    device DispatchIndirectArgs* dispatchArgs [[buffer(3)]],
    constant DepthFirstDispatchConfig& config [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid > 0) return;

    uint visibleCount = visibleCountAtomic[0];
    uint totalInstances = totalInstancesAtomic[0];

    // Clamp to max
    if (visibleCount > config.maxGaussians) {
        visibleCount = config.maxGaussians;
        header->overflow = 1u;
    }
    if (totalInstances > config.maxInstances) {
        totalInstances = config.maxInstances;
        header->overflow = 1u;
    }

    // Compute padded counts for radix alignment
    uint radixAlignment = config.radixBlockSize * config.radixGrainSize;
    uint paddedVisibleCount = ((visibleCount + radixAlignment - 1u) / radixAlignment) * radixAlignment;
    uint paddedInstanceCount = ((totalInstances + radixAlignment - 1u) / radixAlignment) * radixAlignment;

    // Store in header
    header->visibleCount = visibleCount;
    header->totalInstances = totalInstances;
    header->paddedVisibleCount = paddedVisibleCount;
    header->paddedInstanceCount = paddedInstanceCount;

    // === Depth Sort Dispatch (based on visibleCount) ===
    uint depthRadixGrid = (visibleCount > 0u) ? ((paddedVisibleCount + radixAlignment - 1u) / radixAlignment) : 0u;
    uint depthHistBlocks = (depthRadixGrid * 256u + 255u) / 256u; // RADIX_BUCKETS = 256

    dispatchArgs[DF_SLOT_DEPTH_HISTOGRAM].threadgroupsPerGridX = depthRadixGrid;
    dispatchArgs[DF_SLOT_DEPTH_HISTOGRAM].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_DEPTH_HISTOGRAM].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_DEPTH_SCAN_BLOCKS].threadgroupsPerGridX = depthHistBlocks;
    dispatchArgs[DF_SLOT_DEPTH_SCAN_BLOCKS].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_DEPTH_SCAN_BLOCKS].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_DEPTH_EXCLUSIVE].threadgroupsPerGridX = (depthRadixGrid > 0u) ? 1u : 0u;
    dispatchArgs[DF_SLOT_DEPTH_EXCLUSIVE].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_DEPTH_EXCLUSIVE].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_DEPTH_APPLY].threadgroupsPerGridX = depthHistBlocks;
    dispatchArgs[DF_SLOT_DEPTH_APPLY].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_DEPTH_APPLY].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_DEPTH_SCATTER].threadgroupsPerGridX = depthRadixGrid;
    dispatchArgs[DF_SLOT_DEPTH_SCATTER].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_DEPTH_SCATTER].threadgroupsPerGridZ = 1u;

    // === Instance Expansion Dispatch (based on visibleCount) ===
    uint instanceTG = config.instanceExpansionTGSize;
    uint instanceGroups = (visibleCount > 0u) ? ((visibleCount + instanceTG - 1u) / instanceTG) : 0u;

    dispatchArgs[DF_SLOT_APPLY_DEPTH_ORDER].threadgroupsPerGridX = instanceGroups;
    dispatchArgs[DF_SLOT_APPLY_DEPTH_ORDER].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_APPLY_DEPTH_ORDER].threadgroupsPerGridZ = 1u;

    // Prefix sum dispatch (multi-level for large block counts)
    uint prefixTG = config.prefixSumTGSize;  // 256
    uint prefixGroups = (visibleCount > 0u) ? ((visibleCount + prefixTG - 1u) / prefixTG) : 0u;
    uint prefixLevel2Groups = (prefixGroups + prefixTG - 1u) / prefixTG;

    // Level 1: block reduce (same dispatch as before)
    dispatchArgs[DF_SLOT_PREFIX_SUM].threadgroupsPerGridX = prefixGroups;
    dispatchArgs[DF_SLOT_PREFIX_SUM].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_PREFIX_SUM].threadgroupsPerGridZ = 1u;

    // Level 2: reduce block sums into super-block sums
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_REDUCE].threadgroupsPerGridX = prefixLevel2Groups;
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_REDUCE].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_REDUCE].threadgroupsPerGridZ = 1u;

    // Level 2: scan super-block sums (single block, fits 256 super-blocks = 65536 blocks = 16M elements)
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_SCAN].threadgroupsPerGridX = (prefixLevel2Groups > 0u) ? 1u : 0u;
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_SCAN].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_SCAN].threadgroupsPerGridZ = 1u;

    // Level 2: apply super-block offsets and scan block sums
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_APPLY].threadgroupsPerGridX = prefixLevel2Groups;
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_APPLY].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_APPLY].threadgroupsPerGridZ = 1u;

    // Final block scan with block offsets
    dispatchArgs[DF_SLOT_PREFIX_FINAL_SCAN].threadgroupsPerGridX = prefixGroups;
    dispatchArgs[DF_SLOT_PREFIX_FINAL_SCAN].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_PREFIX_FINAL_SCAN].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_CREATE_INSTANCES].threadgroupsPerGridX = instanceGroups;
    dispatchArgs[DF_SLOT_CREATE_INSTANCES].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_CREATE_INSTANCES].threadgroupsPerGridZ = 1u;

    // === Tile Sort Dispatch (stable radix sort based on totalInstances) ===
    uint tileRadixGrid = (totalInstances > 0u) ? ((paddedInstanceCount + radixAlignment - 1u) / radixAlignment) : 0u;
    uint tileHistBlocks = (tileRadixGrid * 256u + 255u) / 256u;

    dispatchArgs[DF_SLOT_TILE_HISTOGRAM].threadgroupsPerGridX = tileRadixGrid;
    dispatchArgs[DF_SLOT_TILE_HISTOGRAM].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_TILE_HISTOGRAM].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_TILE_SCAN_BLOCKS].threadgroupsPerGridX = tileHistBlocks;
    dispatchArgs[DF_SLOT_TILE_SCAN_BLOCKS].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_TILE_SCAN_BLOCKS].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_TILE_EXCLUSIVE].threadgroupsPerGridX = (tileRadixGrid > 0u) ? 1u : 0u;
    dispatchArgs[DF_SLOT_TILE_EXCLUSIVE].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_TILE_EXCLUSIVE].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_TILE_APPLY].threadgroupsPerGridX = tileHistBlocks;
    dispatchArgs[DF_SLOT_TILE_APPLY].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_TILE_APPLY].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_TILE_SCATTER].threadgroupsPerGridX = tileRadixGrid;
    dispatchArgs[DF_SLOT_TILE_SCATTER].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_TILE_SCATTER].threadgroupsPerGridZ = 1u;

    // === Tile Range Extraction (based on tileCount - iterates over all tiles) ===
    uint rangeTG = config.tileRangeTGSize;
    uint rangeGroups = (config.tileCount + rangeTG - 1u) / rangeTG;

    dispatchArgs[DF_SLOT_EXTRACT_RANGES].threadgroupsPerGridX = rangeGroups;
    dispatchArgs[DF_SLOT_EXTRACT_RANGES].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_EXTRACT_RANGES].threadgroupsPerGridZ = 1u;

    // Render dispatch will be set by prepareRenderDispatchKernel after extractTileRanges
    // (activeTileCount not known until then)
    dispatchArgs[DF_SLOT_RENDER].threadgroupsPerGridX = 0u;
    dispatchArgs[DF_SLOT_RENDER].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_RENDER].threadgroupsPerGridZ = 1u;
}

// Prepare render dispatch based on actual active tile count
// Must run AFTER extractTileRangesKernel which populates activeTileCount
kernel void prepareRenderDispatchKernel(
    device const uint* activeTileCount [[buffer(0)]],
    device DispatchIndirectArgs* dispatchArgs [[buffer(1)]],
    constant uint& tileCount [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid > 0) return;

    uint count = activeTileCount[0];
    if (count > tileCount) count = tileCount;

    dispatchArgs[DF_SLOT_RENDER].threadgroupsPerGridX = count;
    dispatchArgs[DF_SLOT_RENDER].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_RENDER].threadgroupsPerGridZ = 1u;
}

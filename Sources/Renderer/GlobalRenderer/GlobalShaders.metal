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

constant bool USE_CLUSTER_CULL [[function_constant(1)]];

template <typename PackedWorldT, typename HarmonicT, typename RenderDataT>
kernel void projectGaussiansFused(
    const device PackedWorldT* worldGaussians [[buffer(0)]],
    const device HarmonicT* harmonics [[buffer(1)]],
    device RenderDataT* outRenderData [[buffer(2)]],      // AoS packed output
    device int4* outBounds [[buffer(3)]],                 // Tile bounds
    device uchar* outMask [[buffer(4)]],
    constant CameraUniforms& camera [[buffer(5)]],
    constant TileBinningParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= camera.gaussianCount) { return; }

    // SINGLE coalesced read for all core gaussian data
    PackedWorldT g = worldGaussians[gid];

    // Early cull: skip tiny gaussians (scale < 0.001)
    float3 scale = float3(g.sx, g.sy, g.sz);
    float maxScale = max(scale.x, max(scale.y, scale.z));
    if (maxScale < 0.0005f) {
        outMask[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    // Extract and promote to float for computation accuracy
    float3 position = float3(g.px, g.py, g.pz);
    float4 viewPos4 = camera.viewMatrix * float4(position, 1.0f);
    float depth = safeDepthComponent(viewPos4.z);

    if (depth <= camera.nearPlane) {
        outMask[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    float4 clip = camera.projectionMatrix * viewPos4;
    if (fabs(clip.w) < 1e-6f) {
        outMask[gid] = 2;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    float ndcX = clip.x / clip.w;
    float ndcY = clip.y / clip.w;
    float px = ((ndcX + 1.0f) * camera.width - 1.0f) * 0.5f;
    float py = ((ndcY + 1.0f) * camera.height - 1.0f) * 0.5f;

    float opacity = float(g.opacity);
    if (opacity < 1e-4f) {
        outMask[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    if (opacity < params.alphaThreshold) {
        outMask[gid] = 0;
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
        outMask[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    float det = conic.x * conic.z - conic.y * conic.y;
    float totalInk = opacity * 6.283185f / sqrt(max(det, 1e-6f));
    float depthFactor = 1.0 - pow(saturate((camera.farPlane - depth) / (camera.farPlane - camera.nearPlane)), 4.0);
    float adjustedThreshold = depthFactor * params.totalInkThreshold;
    if (totalInk < adjustedThreshold) {
        outMask[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    float2 obbExtents = computeOBBExtents(cov2d, 3.0f);

    // Off-screen culling using OBB extents (tighter than radius-based)
    if (px + obbExtents.x < 0.0f || px - obbExtents.x > camera.width ||
        py + obbExtents.y < 0.0f || py - obbExtents.y > camera.height) {
        outMask[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    // Compute SH color
    float3 color = computeSHColor(harmonics, gid, position, camera.cameraCenter, camera.shComponents);
    color = max(color + float3(0.5f), float3(0.0f));

    // Write AoS output - single struct write
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
    outMask[gid] = 1;
}

// Fused projection + tile bounds kernel instantiation
// Float world input
template [[host_name("projectGaussiansFusedKernel")]]
kernel void projectGaussiansFused<PackedWorldGaussian, float, GaussianRenderData>(
    const device PackedWorldGaussian*, const device float*,
    device GaussianRenderData*, device int4*, device uchar*,
    constant CameraUniforms&, constant TileBinningParams&, uint);

// Half world input (for bandwidth optimization)
template [[host_name("projectGaussiansFusedKernelHalf")]]
kernel void projectGaussiansFused<PackedWorldGaussianHalf, half, GaussianRenderData>(
    const device PackedWorldGaussianHalf*, const device half*,
    device GaussianRenderData*, device int4*, device uchar*,
    constant CameraUniforms&, constant TileBinningParams&, uint);

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

// Reset tile builder state
kernel void resetTileBuilderStateKernel(
    device TileAssignmentHeader* header [[buffer(0)]],
    device uint* activeTileCount [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        header->totalAssignments = 0;
        header->overflow = 0;
        *activeTileCount = 0;
    }
}

kernel void markVisibilityKernel(
    const device int4* bounds [[buffer(0)]],       // Tile bounds from projection
    device uint* visibilityMarks [[buffer(1)]],    // Output: 1 if visible, 0 if culled
    constant uint& gaussianCount [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= gaussianCount) return;

    // Check if gaussian is visible (valid bounds: minTX <= maxTX)
    int4 rect = bounds[gid];
    bool visible = (rect.x <= rect.y && rect.z <= rect.w);
    visibilityMarks[gid] = visible ? 1u : 0u;
}

kernel void scatterCompactKernel(
    const device int4* bounds [[buffer(0)]],           // Tile bounds from projection
    const device uint* prefixSumOffsets [[buffer(1)]], // Exclusive prefix sum of visibility marks
    device uint* visibleIndices [[buffer(2)]],         // Output: compacted visible indices
    device uint* visibleCount [[buffer(3)]],           // Output: number of visible gaussians
    constant uint& gaussianCount [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= gaussianCount) return;

    // Check if gaussian is visible
    int4 rect = bounds[gid];
    bool visible = (rect.x <= rect.y && rect.z <= rect.w);

    if (visible) {
        // Write to position given by exclusive prefix sum
        uint writePos = prefixSumOffsets[gid];
        visibleIndices[writePos] = gid;
    }

    // Last thread writes total visible count
    if (gid == gaussianCount - 1u) {
        // Total = offset of last element + its visibility mark
        *visibleCount = prefixSumOffsets[gid] + (visible ? 1u : 0u);
    }
}

kernel void prepareIndirectDispatchKernel(
    const device uint* visibleCount [[buffer(0)]],
    device DispatchIndirectArgs* dispatchArgs [[buffer(1)]],
    constant uint& threadgroupSize [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    uint count = *visibleCount;
    uint groups = (count + threadgroupSize - 1u) / threadgroupSize;

    dispatchArgs->threadgroupsPerGridX = groups;
    dispatchArgs->threadgroupsPerGridY = 1u;
    dispatchArgs->threadgroupsPerGridZ = 1u;
}

// Sort key: [tile:16][depth:16] - radix processes LSB first so depth in low bits
template <typename RenderDataT>
kernel void computeSortKeysKernel(
    const device int* tileIds [[buffer(0)]],
    const device int* tileIndices [[buffer(1)]],
    const device RenderDataT* renderData [[buffer(2)]],
    device uint* sortKeys [[buffer(3)]],
    device int* sortedIndices [[buffer(4)]],
    const device TileAssignmentHeader& header [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint totalAssignments = header.totalAssignments;
    uint padded = header.paddedCount;
    if (gid < padded) {
        if (gid < totalAssignments) {
            int g = tileIndices[gid];
            uint tileId = (uint)tileIds[gid];
            float depthFloat = float(renderData[g].depth);
            half depthHalf = half(depthFloat);
            // XOR with sign bit for correct IEEE 754 sorting as unsigned
            uint depthBits = uint(as_type<ushort>(depthHalf)) ^ 0x8000u;
            // Pack: [tile:16][depth:16] - tile in high bits for primary sort
            sortKeys[gid] = (tileId << 16) | (depthBits & 0xFFFFu);
            sortedIndices[gid] = g;
        } else {
            sortKeys[gid] = 0xFFFFFFFFu;
            sortedIndices[gid] = -1;
        }
    }
}

template [[host_name("computeSortKeysKernel")]]
kernel void computeSortKeysKernel<GaussianRenderData>(
    const device int*, const device int*, const device GaussianRenderData*,
    device uint*, device int*, const device TileAssignmentHeader&, uint);


// Fused: build headers + compact active tiles in one pass
kernel void buildHeadersFromSortedKernel(
    const device uint* sortedKeys [[buffer(0)]],  // 32-bit keys: [tile:16][depth:16]
    device GaussianHeader* headers [[buffer(1)]],
    const device TileAssignmentHeader* headerInfo [[buffer(2)]],
    constant uint& tileCount [[buffer(3)]],
    device uint* activeTiles [[buffer(4)]],
    device atomic_uint* activeCount [[buffer(5)]],
    uint tile [[thread_position_in_grid]]
) {
    if (tile >= tileCount) {
        return;
    }
    uint total = headerInfo->totalAssignments;
    if (total == 0) {
        GaussianHeader emptyHeader;
        emptyHeader.offset = 0u;
        emptyHeader.count = 0u;
        headers[tile] = emptyHeader;
        return;
    }

    // Binary search for start of tile (tile ID is in high 16 bits)
    uint left = 0;
    uint right = total;
    while (left < right) {
        uint mid = (left + right) >> 1;
        uint midTile = sortedKeys[mid] >> 16;  // Extract tile from high bits
        if (midTile < tile) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    uint start = left;

    // Binary search for end of tile
    left = start;
    right = total;
    while (left < right) {
        uint mid = (left + right) >> 1;
        uint midTile = sortedKeys[mid] >> 16;  // Extract tile from high bits
        if (midTile <= tile) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    uint end = left;

    GaussianHeader header;
    header.offset = start;
    header.count = end > start ? (end - start) : 0u;
    headers[tile] = header;

    // Compact active tiles (fused - no separate pass needed)
    if (header.count > 0u) {
        uint index = atomic_fetch_add_explicit(activeCount, 1u, memory_order_relaxed);
        activeTiles[index] = tile;
    }
}

struct RenderDispatchParams {
    uint tileCount;
    uint totalAssignments;
};

kernel void prepareRenderDispatchKernel(
    const device uint* activeCount [[buffer(0)]],
    device DispatchIndirectArgs* dispatchArgs [[buffer(1)]],
    constant RenderDispatchParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid > 0) {
        return;
    }
    uint count = activeCount[0];
    if (count > params.tileCount) { count = params.tileCount; }
    dispatchArgs[DispatchSlotRenderTiles].threadgroupsPerGridX = count;
    dispatchArgs[DispatchSlotRenderTiles].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotRenderTiles].threadgroupsPerGridZ = 1u;
}

#define SCAN_BLOCK_SIZE 256u

// Block-level reduce: compute sum of each block using SIMD-optimized reduction
kernel void blockReduceKernel(
    const device uint* input [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]]
) {
    threadgroup uint shared[SCAN_BLOCK_SIZE];

    uint value = (gid < count) ? input[gid] : 0u;

    // Use SIMD-optimized cooperative reduction
    uint blockSum = threadgroup_cooperative_reduce_sum<SCAN_BLOCK_SIZE>(value, shared, ushort(lid));

    if (lid == 0u) {
        blockSums[groupId] = blockSum;
    }
}

kernel void blockScanKernel(
    const device uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    const device uint* blockOffsets [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]]
) {
    threadgroup uint shared[SCAN_BLOCK_SIZE];

    uint value = (gid < count) ? input[gid] : 0u;

    // Use SIMD-optimized raking prefix exclusive sum
    uint exclusive = threadgroup_raking_prefix_exclusive_sum<SCAN_BLOCK_SIZE>(value, shared, ushort(lid));

    // Add block offset
    if (gid < count) {
        output[gid] = exclusive + blockOffsets[groupId];
    }
}

kernel void singleBlockScanKernel(
    device uint* blockSums [[buffer(0)]],
    constant uint& blockCount [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup uint shared[SCAN_BLOCK_SIZE];

    // Read input value
    uint value = (lid < blockCount) ? blockSums[lid] : 0u;

    // Compute total using SIMD reduction
    uint total = threadgroup_cooperative_reduce_sum<SCAN_BLOCK_SIZE>(value, shared, ushort(lid));

    // Compute exclusive prefix sum
    uint exclusive = threadgroup_raking_prefix_exclusive_sum<SCAN_BLOCK_SIZE>(value, shared, ushort(lid));

    // Write back results
    if (lid < blockCount) {
        blockSums[lid] = exclusive;
    }
    // Store total at blockSums[blockCount]
    if (lid == 0u) {
        blockSums[blockCount] = total;
    }
}

kernel void writeTotalCountKernel(
    const device uint* blockSums [[buffer(0)]],
    device TileAssignmentHeader* header [[buffer(1)]],
    constant uint& blockCount [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0u) return;

    // Total is stored at blockSums[blockCount] by singleBlockScanKernel
    header->totalAssignments = blockSums[blockCount];
}

template <typename RenderDataT>
kernel void tileCountIndirectKernel(
    const device int4* bounds [[buffer(0)]],
    const device RenderDataT* renderData [[buffer(1)]],
    device uint* tileCounts [[buffer(2)]],
    const device uint* visibleIndices [[buffer(3)]],
    constant TileAssignParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.gaussianCount) return;

    uint originalIdx = visibleIndices[gid];
    const int4 rect = bounds[originalIdx];
    const int minTX = rect.x;
    const int maxTX = rect.y;
    const int minTY = rect.z;
    const int maxTY = rect.w;

    if (minTX > maxTX || minTY > maxTY) {
        tileCounts[gid] = 0u;
        return;
    }

    const RenderDataT g = renderData[originalIdx];
    const float alpha = float(g.opacity);
    if (alpha < 1e-4f) {
        tileCounts[gid] = 0u;
        return;
    }

    // Extract gaussian properties for intersection test
    float2 center = float2(g.meanX, g.meanY);
    float3 conic = float3(g.conicA, g.conicB, g.conicC);
    float radius = float(g.conicD);  // conicD stores radius

    uint count = 0;
    const int tileW = int(params.tileWidth);
    const int tileH = int(params.tileHeight);

    for (int ty = minTY; ty <= maxTY; ty++) {
        for (int tx = minTX; tx <= maxTX; tx++) {
            int2 pixMin = int2(tx * tileW, ty * tileH);
            int2 pixMax = int2(pixMin.x + tileW - 1, pixMin.y + tileH - 1);

            if (intersectsTile(pixMin, pixMax, center, conic, alpha, radius, float2(0.0f))) {
                count++;
            }
        }
    }

    tileCounts[gid] = count;
}

template [[host_name("tileCountIndirectKernel")]]
kernel void tileCountIndirectKernel<GaussianRenderData>(
    const device int4*, const device GaussianRenderData*, device uint*,
    const device uint*, constant TileAssignParams&, uint);

template <typename RenderDataT>
kernel void tileScatterIndirectKernel(
    const device int4* bounds [[buffer(0)]],
    const device RenderDataT* renderData [[buffer(1)]],
    const device uint* offsets [[buffer(2)]],
    const device uint* visibleIndices [[buffer(3)]],
    device int* tileIndices [[buffer(4)]],
    device int* tileIds [[buffer(5)]],
    constant TileAssignParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.gaussianCount) return;

    uint originalIdx = visibleIndices[gid];
    const int4 rect = bounds[originalIdx];
    const int minTX = rect.x;
    const int maxTX = rect.y;
    const int minTY = rect.z;
    const int maxTY = rect.w;

    if (minTX > maxTX || minTY > maxTY) return;

    const RenderDataT g = renderData[originalIdx];
    const float alpha = float(g.opacity);
    if (alpha < 1e-4f) return;

    // Extract gaussian properties for intersection test
    float2 center = float2(g.meanX, g.meanY);
    float3 conic = float3(g.conicA, g.conicB, g.conicC);
    float radius = float(g.conicD);  // conicD stores radius

    const uint tilesX = params.tilesX;
    const uint maxAssignments = params.maxAssignments;
    const int tileW = int(params.tileWidth);
    const int tileH = int(params.tileHeight);

    uint writePos = offsets[gid];
    const int gaussianIdx = int(originalIdx);

    for (int ty = minTY; ty <= maxTY; ty++) {
        for (int tx = minTX; tx <= maxTX; tx++) {
            int2 pixMin = int2(tx * tileW, ty * tileH);
            int2 pixMax = int2(pixMin.x + tileW - 1, pixMin.y + tileH - 1);

            if (intersectsTile(pixMin, pixMax, center, conic, alpha, radius, float2(0.0f))) {
                if (writePos < maxAssignments) {
                    tileIds[writePos] = ty * int(tilesX) + tx;
                    tileIndices[writePos] = gaussianIdx;
                    writePos++;
                }
            }
        }
    }
}

template [[host_name("tileScatterIndirectKernel")]]
kernel void tileScatterIndirectKernel<GaussianRenderData>(
    const device int4*, const device GaussianRenderData*, const device uint*,
    const device uint*, device int*, device int*, constant TileAssignParams&, uint);

kernel void prepareAssignmentDispatchKernel(
    device TileAssignmentHeader* header [[buffer(0)]],
    device DispatchIndirectArgs* dispatchArgs [[buffer(1)]],
    constant AssignmentDispatchConfig& config [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid > 0) {
        return;
    }
    uint total = header[0].totalAssignments;

    // Clamp totalAssignments to maxAssignments (fix for fused scatter overflow)
    if (total > config.maxAssignments) {
        total = config.maxAssignments;
        header[0].totalAssignments = total;
        header[0].overflow = 1u;
    }

    uint sortTG = max(config.sortThreadgroupSize, 1u);
    uint fuseTG = max(config.fuseThreadgroupSize, 1u);
    uint unpackTG = max(config.unpackThreadgroupSize, 1u);
    uint packTG = max(config.packThreadgroupSize, 1u);
    uint radixBlockSize = max(config.radixBlockSize, 1u);
    uint radixGrainSize = max(config.radixGrainSize, 1u);

    // Radix sort alignment: blockSize * grainSize = 1024
    uint radixAlignment = radixBlockSize * radixGrainSize;
    header[0].paddedCount = ((total + radixAlignment - 1u) / radixAlignment) * radixAlignment;

    // Sort key generation uses total count (not padded)
    uint sortGroups = (total > 0u) ? ((total + sortTG - 1u) / sortTG) : 0u;
    dispatchArgs[DispatchSlotSortKeys].threadgroupsPerGridX = sortGroups;
    dispatchArgs[DispatchSlotSortKeys].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotSortKeys].threadgroupsPerGridZ = 1u;

    // Fuse/unpack use total count for radix path
    uint fuseGroups = (total > 0u) ? ((total + fuseTG - 1u) / fuseTG) : 0u;
    dispatchArgs[DispatchSlotFuseKeys].threadgroupsPerGridX = fuseGroups;
    dispatchArgs[DispatchSlotFuseKeys].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotFuseKeys].threadgroupsPerGridZ = 1u;

    uint unpackGroups = (total > 0u) ? ((total + unpackTG - 1u) / unpackTG) : 0u;
    dispatchArgs[DispatchSlotUnpackKeys].threadgroupsPerGridX = unpackGroups;
    dispatchArgs[DispatchSlotUnpackKeys].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotUnpackKeys].threadgroupsPerGridZ = 1u;

    uint packGroups = (total + packTG - 1u) / packTG;
    dispatchArgs[DispatchSlotPack].threadgroupsPerGridX = (total > 0u) ? packGroups : 0u;
    dispatchArgs[DispatchSlotPack].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotPack].threadgroupsPerGridZ = 1u;

    uint valuesPerGroup = max(radixBlockSize * radixGrainSize, 1u);
    uint radixGrid = (total > 0u) ? ((total + valuesPerGroup - 1u) / valuesPerGroup) : 0u;
    uint histogramGroups = radixGrid;
    uint applyGroups = radixGrid;
    uint scatterGroups = radixGrid;
    uint blockCount = radixGrid;
    uint exclusiveGroups = (blockCount > 0u) ? 1u : 0u;

    dispatchArgs[DispatchSlotRadixHistogram].threadgroupsPerGridX = histogramGroups;
    dispatchArgs[DispatchSlotRadixHistogram].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotRadixHistogram].threadgroupsPerGridZ = 1u;

    dispatchArgs[DispatchSlotRadixScanBlocks].threadgroupsPerGridX = applyGroups;
    dispatchArgs[DispatchSlotRadixScanBlocks].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotRadixScanBlocks].threadgroupsPerGridZ = 1u;

    dispatchArgs[DispatchSlotRadixExclusive].threadgroupsPerGridX = exclusiveGroups;
    dispatchArgs[DispatchSlotRadixExclusive].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotRadixExclusive].threadgroupsPerGridZ = 1u;

    dispatchArgs[DispatchSlotRadixApply].threadgroupsPerGridX = applyGroups;
    dispatchArgs[DispatchSlotRadixApply].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotRadixApply].threadgroupsPerGridZ = 1u;

    dispatchArgs[DispatchSlotRadixScatter].threadgroupsPerGridX = scatterGroups;
    dispatchArgs[DispatchSlotRadixScatter].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotRadixScatter].threadgroupsPerGridZ = 1u;
    dispatchArgs[DispatchSlotRenderTiles].threadgroupsPerGridX = 0u;
    dispatchArgs[DispatchSlotRenderTiles].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotRenderTiles].threadgroupsPerGridZ = 1u;
}

kernel void radixHistogramKernel(
    device const RadixKeyType* inputKeys [[buffer(0)]],
    device uint* histFlat [[buffer(1)]],
    constant uint& currentDigit [[buffer(3)]],
    const device TileAssignmentHeader* header [[buffer(4)]],
    uint gridSize [[threadgroups_per_grid]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint elementsPerBlock = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;

    uint baseId = groupId * elementsPerBlock;
    uint totalAssignments = header[0].totalAssignments;
    uint paddedCount = header[0].paddedCount;

    uint bufferRemaining = (baseId < paddedCount) ? (paddedCount - baseId) : 0;
    uint assignmentsRemain = (baseId < totalAssignments) ? (totalAssignments - baseId) : 0;
    uint available = min(bufferRemaining, assignmentsRemain);

    // 1) Per-group local histogram in LDS
    threadgroup uint localHist[RADIX_BUCKETS];

    for (uint i = 0; i < (RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE; ++i) {
        uint bin = localId + i * RADIX_BLOCK_SIZE;
        if (bin < RADIX_BUCKETS) {
            localHist[bin] = 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) Load keys in blocked fashion per group into registers
    RadixKeyType keys[RADIX_GRAIN_SIZE];
    RadixKeyType sentinelKey = (RadixKeyType)0;

    load_blocked_local_from_global<RADIX_GRAIN_SIZE>(
        keys,
        &inputKeys[baseId],
        localId,
        available,
        sentinelKey);

    // 3) Build local histogram using threadgroup atomics
    volatile threadgroup atomic_uint* atomicHist =
        reinterpret_cast<volatile threadgroup atomic_uint*>(localHist);

    for (ushort i = 0; i < RADIX_GRAIN_SIZE; ++i) {
        uint offsetInBlock = localId * RADIX_GRAIN_SIZE + i;
        uint globalIdx = baseId + offsetInBlock;

        if (globalIdx < totalAssignments) {
            uchar bin = (uchar)value_to_key_at_digit(keys[i], (ushort)currentDigit);
            atomic_fetch_add_explicit(&atomicHist[bin], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4) Write out in bin-major layout: [bin][block]
    for (uint i = 0; i < (RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE; ++i) {
        uint bin = localId + i * RADIX_BLOCK_SIZE;
        if (bin < RADIX_BUCKETS) {
            histFlat[bin * gridSize + groupId] = localHist[bin];
        }
    }
}

kernel void radixScanBlocksKernel(
    device const uint* histFlat [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    const device TileAssignmentHeader* header [[buffer(2)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    threadgroup uint sharedMem[RADIX_BLOCK_SIZE];

    const uint elementsPerBlock = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint paddedCount = header[0].paddedCount;
    uint histGridSize = (paddedCount + elementsPerBlock - 1u) / elementsPerBlock;
    uint numHistElem = histGridSize * RADIX_BUCKETS;

    uint blockBase = groupId * RADIX_BLOCK_SIZE;
    uint idx = blockBase + localId;

    uint val = (idx < numHistElem) ? histFlat[idx] : 0u;
    sharedMem[localId] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = RADIX_BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
        if (localId < stride) {
            sharedMem[localId] += sharedMem[localId + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (localId == 0) {
        blockSums[groupId] = sharedMem[0];
    }
}

// Parallel exclusive scan - 256 threads, up to 8192 elements
#define RADIX_SCAN_GRAIN 32

// Each thread handles RADIX_SCAN_GRAIN elements, sums them, scans sums, then applies back
kernel void radixExclusiveScanKernel(
    device uint* blockSums [[buffer(0)]],
    const device TileAssignmentHeader* header [[buffer(1)]],
    threadgroup uint* sharedMem [[threadgroup(0)]],
    ushort lid [[thread_position_in_threadgroup]]
) {

    const uint elementsPerBlock = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint paddedCount = header[0].paddedCount;
    uint histGridSize = (paddedCount + elementsPerBlock - 1u) / elementsPerBlock;
    uint numHistElem = histGridSize * RADIX_BUCKETS;
    uint numBlockSums = (numHistElem + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE;

    // Calculate elements per thread (up to RADIX_SCAN_GRAIN)
    uint elemsPerThread = (numBlockSums + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE;
    elemsPerThread = min(elemsPerThread, (uint)RADIX_SCAN_GRAIN);

    // Load multiple elements and compute local sum
    uint localVals[RADIX_SCAN_GRAIN];
    uint localSum = 0u;
    uint threadBase = lid * elemsPerThread;

    for (uint i = 0u; i < elemsPerThread; ++i) {
        uint idx = threadBase + i;
        uint val = (idx < numBlockSums) ? blockSums[idx] : 0u;
        localVals[i] = val;
        localSum += val;
    }

    // Scan the per-thread sums using helper (exclusive scan of 256 values)
    uint scannedSum = radix_threadgroup_prefix_scan<RADIX_SCAN_TYPE_EXCLUSIVE>(localSum, sharedMem, lid, radix_sum_op<uint>());

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


kernel void radixApplyScanOffsetsKernel(
    device const uint* histFlat [[buffer(0)]],
    device const uint* scannedBlocks [[buffer(1)]],
    device uint* offsetsFlat [[buffer(2)]],
    const device TileAssignmentHeader* header [[buffer(3)]],
    threadgroup uint* sharedMem [[threadgroup(0)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint elementsPerBlock = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint paddedCount = header[0].paddedCount;
    uint histGridSize = (paddedCount + elementsPerBlock - 1u) / elementsPerBlock;
    uint numHistElem = histGridSize * RADIX_BUCKETS;

    uint blockBase = groupId * RADIX_BLOCK_SIZE;
    uint idx = blockBase + localId;

    uint val = (idx < numHistElem) ? histFlat[idx] : 0u;

    uint localScanned = radix_threadgroup_prefix_scan<RADIX_SCAN_TYPE_EXCLUSIVE>(
        val, sharedMem, localId, radix_sum_op<uint>());

    uint blockOffset = scannedBlocks[groupId];

    if (idx < numHistElem) {
        offsetsFlat[idx] = localScanned + blockOffset;
    }
}

kernel void radixScatterKernel(
    device RadixKeyType* outputKeys [[buffer(0)]],
    device const RadixKeyType* inputKeys [[buffer(1)]],
    device uint* outputPayload [[buffer(2)]],
    device const uint* inputPayload [[buffer(3)]],
    device const uint* offsetsFlat [[buffer(5)]],
    constant uint& currentDigit [[buffer(6)]],
    const device TileAssignmentHeader* header [[buffer(7)]],
    uint groupId [[threadgroup_position_in_grid]],
    uint gridSize [[threadgroups_per_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint elementsPerBlock = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;

    uint baseId = groupId * elementsPerBlock;
    uint totalAssignments = header[0].totalAssignments;
    uint paddedCount = header[0].paddedCount;

    uint bufferRemaining = (baseId < paddedCount) ? (paddedCount - baseId) : 0;
    uint assignmentsRemain = (baseId < totalAssignments) ? (totalAssignments - baseId) : 0;
    uint available = min(bufferRemaining, assignmentsRemain);

    // Use max key as sentinel
    constexpr RadixKeyType keySentinel = ~(RadixKeyType)0;

    // 1) Load keys / payloads in STRIPED fashion
    RadixKeyType keys[RADIX_GRAIN_SIZE];
    uint payloads[RADIX_GRAIN_SIZE];
    load_striped_local_from_global<RADIX_GRAIN_SIZE, RADIX_BLOCK_SIZE>(keys, &inputKeys[baseId], localId, available, keySentinel);
    load_striped_local_from_global<RADIX_GRAIN_SIZE, RADIX_BLOCK_SIZE>(payloads, &inputPayload[baseId], localId, available, UINT_MAX);

    // 2) Shared memory for bin offsets and ranking
    threadgroup uint globalBinBase[RADIX_BUCKETS];
    threadgroup RadixKeyPayload tgKp[RADIX_BLOCK_SIZE];
    threadgroup ushort tgShort[RADIX_BLOCK_SIZE];

    // Load global bin offsets
    for (uint i = 0; i < (RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE; ++i) {
        uint bin = localId + i * RADIX_BLOCK_SIZE;
        if (bin < RADIX_BUCKETS) {
            globalBinBase[bin] = offsetsFlat[bin * gridSize + groupId];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) Process chunks with optimized ranking
    for (ushort chunk = 0; chunk < RADIX_GRAIN_SIZE; ++chunk) {
        RadixKeyPayload kp;
        kp.key = keys[chunk];
        kp.payload = payloads[chunk];

        // Partial radix sort: groups same-bin elements together, preserving relative order
        kp = radix_partial_sort(kp, tgKp, localId, (ushort)currentDigit);

        // Extract bin after sorting
        ushort myBin = value_to_key_at_digit(kp.key, (ushort)currentDigit);

        // Head discontinuity + max scan to find run start
        uchar headFlag = flag_head_discontinuity<RADIX_BLOCK_SIZE>(myBin, tgShort, localId);
        ushort runStart = radix_threadgroup_prefix_scan<RADIX_SCAN_TYPE_INCLUSIVE>(
            headFlag ? localId : (ushort)0, tgShort, localId, radix_max_op<ushort>());
        ushort localOffset = localId - runStart;

        // Tail discontinuity for offset updates
        uchar tailFlag = flag_tail_discontinuity<RADIX_BLOCK_SIZE>(myBin, tgShort, localId);

        // Check validity and scatter
        bool isValid = (kp.payload != UINT_MAX);
        if (isValid) {
            uint dst = globalBinBase[myBin] + localOffset;
            outputKeys[dst] = kp.key;
            outputPayload[dst] = kp.payload;
        }

        // Update global offsets at run boundaries
        if (tailFlag && isValid) {
            globalBinBase[myBin] += localOffset + 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// 32x16 tile, 8x8 threadgroup (64 threads), 4x2 pixels per thread
#define RENDER_V3_TG_WIDTH 8
#define RENDER_V3_TG_HEIGHT 8
#define RENDER_V3_PIXELS_X 4
#define RENDER_V3_PIXELS_Y 2

kernel void globalRender(
    const device GaussianHeader* headers [[buffer(0)]],
    const device GaussianRenderData* gaussians [[buffer(1)]],
    const device int* sortedIndices [[buffer(2)]],
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

    // 32x16 tile, 8x8 threads, 4x2 pixels per thread
    uint baseX = tileX * 32 + localId.x * RENDER_V3_PIXELS_X;
    uint baseY = tileY * 16 + localId.y * RENDER_V3_PIXELS_Y;

    // Pixel positions (8 pixels: 4x2)
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

    half depth00 = 0, depth10 = 0, depth20 = 0, depth30 = 0;
    half depth01 = 0, depth11 = 0, depth21 = 0, depth31 = 0;

    uint start = hdr.offset;
    uint count = hdr.count;

    for (uint i = 0; i < count; i++) {
        // Early exit when all pixels are saturated
        half maxTrans0 = max(max(trans00, trans10), max(trans20, trans30));
        half maxTrans1 = max(max(trans01, trans11), max(trans21, trans31));
        if (max(maxTrans0, maxTrans1) < half(1.0h/255.0h)) break;

        int gaussianIdx = sortedIndices[start + i];
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
        half2 d20 = pos20 - gMean;
        half2 d30 = pos30 - gMean;
        half2 d01 = pos01 - gMean;
        half2 d11 = pos11 - gMean;
        half2 d21 = pos21 - gMean;
        half2 d31 = pos31 - gMean;

        half p00 = d00.x*d00.x*cxx + d00.y*d00.y*cyy + d00.x*d00.y*cxy2;
        half p10 = d10.x*d10.x*cxx + d10.y*d10.y*cyy + d10.x*d10.y*cxy2;
        half p20 = d20.x*d20.x*cxx + d20.y*d20.y*cyy + d20.x*d20.y*cxy2;
        half p30 = d30.x*d30.x*cxx + d30.y*d30.y*cyy + d30.x*d30.y*cxy2;
        half p01 = d01.x*d01.x*cxx + d01.y*d01.y*cyy + d01.x*d01.y*cxy2;
        half p11 = d11.x*d11.x*cxx + d11.y*d11.y*cyy + d11.x*d11.y*cxy2;
        half p21 = d21.x*d21.x*cxx + d21.y*d21.y*cyy + d21.x*d21.y*cxy2;
        half p31 = d31.x*d31.x*cxx + d31.y*d31.y*cyy + d31.x*d31.y*cxy2;

        half a00 = min(opacity * exp(-0.5h * p00), 0.99h);
        half a10 = min(opacity * exp(-0.5h * p10), 0.99h);
        half a20 = min(opacity * exp(-0.5h * p20), 0.99h);
        half a30 = min(opacity * exp(-0.5h * p30), 0.99h);
        half a01 = min(opacity * exp(-0.5h * p01), 0.99h);
        half a11 = min(opacity * exp(-0.5h * p11), 0.99h);
        half a21 = min(opacity * exp(-0.5h * p21), 0.99h);
        half a31 = min(opacity * exp(-0.5h * p31), 0.99h);

        half4 alphaRow0 = half4(a00, a10, a20, a30);
        half4 alphaRow1 = half4(a01, a11, a21, a31);
        if (all(alphaRow0 == 0.0h) && all(alphaRow1 == 0.0h)) continue;

        color00 += gColor * (a00 * trans00); depth00 += gDepth * (a00 * trans00);
        color10 += gColor * (a10 * trans10); depth10 += gDepth * (a10 * trans10);
        color20 += gColor * (a20 * trans20); depth20 += gDepth * (a20 * trans20);
        color30 += gColor * (a30 * trans30); depth30 += gDepth * (a30 * trans30);
        color01 += gColor * (a01 * trans01); depth01 += gDepth * (a01 * trans01);
        color11 += gColor * (a11 * trans11); depth11 += gDepth * (a11 * trans11);
        color21 += gColor * (a21 * trans21); depth21 += gDepth * (a21 * trans21);
        color31 += gColor * (a31 * trans31); depth31 += gDepth * (a31 * trans31);

        trans00 *= (1.0h - a00); trans10 *= (1.0h - a10);
        trans20 *= (1.0h - a20); trans30 *= (1.0h - a30);
        trans01 *= (1.0h - a01); trans11 *= (1.0h - a11);
        trans21 *= (1.0h - a21); trans31 *= (1.0h - a31);
    }

    // Write 8 pixels
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
    if (baseX + 2 < w && baseY + 0 < h) {
        colorOut.write(half4(color20, 1.0h - trans20), uint2(baseX + 2, baseY + 0));
        if (hasDepth) depthOut.write(half4(depth20, 0, 0, 0), uint2(baseX + 2, baseY + 0));
    }
    if (baseX + 3 < w && baseY + 0 < h) {
        colorOut.write(half4(color30, 1.0h - trans30), uint2(baseX + 3, baseY + 0));
        if (hasDepth) depthOut.write(half4(depth30, 0, 0, 0), uint2(baseX + 3, baseY + 0));
    }
    if (baseX + 0 < w && baseY + 1 < h) {
        colorOut.write(half4(color01, 1.0h - trans01), uint2(baseX + 0, baseY + 1));
        if (hasDepth) depthOut.write(half4(depth01, 0, 0, 0), uint2(baseX + 0, baseY + 1));
    }
    if (baseX + 1 < w && baseY + 1 < h) {
        colorOut.write(half4(color11, 1.0h - trans11), uint2(baseX + 1, baseY + 1));
        if (hasDepth) depthOut.write(half4(depth11, 0, 0, 0), uint2(baseX + 1, baseY + 1));
    }
    if (baseX + 2 < w && baseY + 1 < h) {
        colorOut.write(half4(color21, 1.0h - trans21), uint2(baseX + 2, baseY + 1));
        if (hasDepth) depthOut.write(half4(depth21, 0, 0, 0), uint2(baseX + 2, baseY + 1));
    }
    if (baseX + 3 < w && baseY + 1 < h) {
        colorOut.write(half4(color31, 1.0h - trans31), uint2(baseX + 3, baseY + 1));
        if (hasDepth) depthOut.write(half4(depth31, 0, 0, 0), uint2(baseX + 3, baseY + 1));
    }
}

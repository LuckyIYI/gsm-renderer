#include <metal_stdlib>
#include <metal_atomic>
#include "BridgingTypes.h"
#include "GaussianShared.h"
#include "GaussianPrefixScan.h"
#include "GaussianHelpers.h"
using namespace metal;

// =============================================================================
// STRUCT ACCESSOR HELPERS - GaussianRenderData (specific to GlobalSort)
// =============================================================================

// GaussianRenderData accessors (device address space)
inline half2 getMean(const device GaussianRenderData& g) { return half2(g.meanX, g.meanY); }
inline half4 getConic(const device GaussianRenderData& g) { return half4(g.conicA, g.conicB, g.conicC, g.conicD); }
inline half3 getColor(const device GaussianRenderData& g) { return half3(g.colorR, g.colorG, g.colorB); }

// GaussianRenderData accessors (thread address space)
inline half2 getMean(const thread GaussianRenderData& g) { return half2(g.meanX, g.meanY); }
inline half4 getConic(const thread GaussianRenderData& g) { return half4(g.conicA, g.conicB, g.conicC, g.conicD); }
inline half3 getColor(const thread GaussianRenderData& g) { return half3(g.colorR, g.colorG, g.colorB); }

// =============================================================================
// FUSED PROJECTION + TILE BOUNDS KERNEL
// Combines projection and tile bounds computation in a single pass
// Eliminates separate radii buffer and tileBounds kernel dispatch
// =============================================================================

// Function constant for cluster culling (index 1, index 0 is SH_DEGREE)
constant bool USE_CLUSTER_CULL [[function_constant(1)]];

struct ProjectFusedParams {
    uint tileWidth;
    uint tileHeight;
    uint tilesX;
    uint tilesY;
};

template <typename PackedWorldT, typename HarmonicT, typename RenderDataT>
kernel void projectGaussiansFused(
    const device PackedWorldT* worldGaussians [[buffer(0)]],
    const device HarmonicT* harmonics [[buffer(1)]],
    device RenderDataT* outRenderData [[buffer(2)]],      // AoS packed output
    device int4* outBounds [[buffer(3)]],                 // Tile bounds directly (replaces radii)
    device uchar* outMask [[buffer(4)]],
    constant CameraUniforms& camera [[buffer(5)]],
    constant ProjectFusedParams& tileParams [[buffer(6)]],
    // Optional cluster culling buffers (only bound when USE_CLUSTER_CULL is true)
    const device uchar* clusterVisible [[buffer(7), function_constant(USE_CLUSTER_CULL)]],
    constant uint& clusterSize [[buffer(8), function_constant(USE_CLUSTER_CULL)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= camera.gaussianCount) { return; }

    // Skip-based cluster culling (only when enabled via function constant)
    if (USE_CLUSTER_CULL) {
        uint clusterId = gid / clusterSize;
        if (clusterVisible[clusterId] == 0) {
            outMask[gid] = 0;
            outBounds[gid] = int4(0, -1, 0, -1);  // Empty bounds
            return;
        }
    }

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

    // ==========================================================================
    // OPACITY CONTRIBUTION FILTER - Skip gaussians that can't contribute meaningfully
    // Even at gaussian center (power=0), alpha = opacity * exp(0) = opacity
    // If opacity is too low, no pixel will get meaningful contribution
    // Threshold ~1/255 = 0.004 (below visible quantization level)
    // ==========================================================================
    if (opacity < 0.005f) {
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

    // ==========================================================================
    // TOTAL INK FILTER - Skip gaussians with negligible total contribution
    // Total integrated alpha = opacity × 2π / √det  (the gaussian's "total ink")
    // If total ink is too small, the gaussian can't meaningfully contribute anywhere
    // ==========================================================================
    float det = conic.x * conic.z - conic.y * conic.y;
    float totalInk = opacity * 6.283185f / sqrt(max(det, 1e-6f));
    if (totalInk < 3.0f) {
        outMask[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    // Off-screen culling
    if (px + radius < 0.0f || px - radius > camera.width ||
        py + radius < 0.0f || py - radius > camera.height) {
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

    // =========================================================================
    // FUSED TILE BOUNDS COMPUTATION (previously separate tileBoundsKernel)
    // =========================================================================
    float xmin = px - radius;
    float xmax = px + radius;
    float ymin = py - radius;
    float ymax = py + radius;

    float maxW = camera.width - 1.0f;
    float maxH = camera.height - 1.0f;

    xmin = clamp(xmin, 0.0f, maxW);
    xmax = clamp(xmax, 0.0f, maxW);
    ymin = clamp(ymin, 0.0f, maxH);
    ymax = clamp(ymax, 0.0f, maxH);

    int minTileX = int(floor(xmin / float(tileParams.tileWidth)));
    int maxTileX = int(ceil(xmax / float(tileParams.tileWidth))) - 1;
    int minTileY = int(floor(ymin / float(tileParams.tileHeight)));
    int maxTileY = int(ceil(ymax / float(tileParams.tileHeight))) - 1;

    minTileX = max(minTileX, 0);
    minTileY = max(minTileY, 0);
    maxTileX = min(maxTileX, int(tileParams.tilesX) - 1);
    maxTileY = min(maxTileY, int(tileParams.tilesY) - 1);

    outBounds[gid] = int4(minTileX, maxTileX, minTileY, maxTileY);
    outMask[gid] = 1;
}

// Fused projection + tile bounds kernel instantiation
// Float world input
template [[host_name("projectGaussiansFusedKernel")]]
kernel void projectGaussiansFused<PackedWorldGaussian, float, GaussianRenderData>(
    const device PackedWorldGaussian*, const device float*,
    device GaussianRenderData*, device int4*, device uchar*,
    constant CameraUniforms&, constant ProjectFusedParams&,
    const device uchar*, constant uint&, uint);

// Half world input (for bandwidth optimization)
template [[host_name("projectGaussiansFusedKernelHalf")]]
kernel void projectGaussiansFused<PackedWorldGaussianHalf, half, GaussianRenderData>(
    const device PackedWorldGaussianHalf*, const device half*,
    device GaussianRenderData*, device int4*, device uchar*,
    constant CameraUniforms&, constant ProjectFusedParams&,
    const device uchar*, constant uint&, uint);

// Fast exp approximation for Gaussian weight calculation
// Uses Schraudolph's method - accurate enough for alpha blending
inline float fast_exp_neg(float x) {
    // x should be negative (from -0.5 * quad)
    // Clamp to prevent overflow
    x = max(x, -10.0f);  // exp(-10) ≈ 0, below our threshold anyway
    // Schraudolph's approximation: exp(x) via bit manipulation
    // f = 2^(x * log2(e)) = 2^(x * 1.4427)
    // Scale factor: 2^23 / ln(2) = 12102203.16
    // Bias: 127 * 2^23 = 1065353216, adjusted = 1064866805
    union { float f; int i; } u;
    u.i = int(12102203.0f * x + 1064866805.0f);
    return u.f;
}

// Batch size for shared memory loading (buffer version)
constant uint RENDER_BATCH_SIZE_BUF = 32;

// HYBRID PRECISION: half for per-gaussian math, float for accumulation
template <typename T, typename Vec2, typename Vec4, typename Packed3>
kernel void renderTiles(
    const device GaussianHeader* headers [[buffer(0)]],
    const device Vec2* means [[buffer(1)]],
    const device Vec4* conics [[buffer(2)]],
    const device Packed3* colors [[buffer(3)]],
    const device T* opacities [[buffer(4)]],
    const device T* depths [[buffer(5)]],
    device float* colorOut [[buffer(6)]],
    device float* depthOut [[buffer(7)]],
    device float* alphaOut [[buffer(8)]],
    constant RenderParams& params [[buffer(9)]],
    const device uint* activeTiles [[buffer(10)]],
    const device uint* activeTileCount [[buffer(11)]],
    uint3 localPos3 [[thread_position_in_threadgroup]],
    uint3 tileCoord [[threadgroup_position_in_grid]],
    uint threadIdx [[thread_index_in_threadgroup]]
) {
    // Shared memory in half precision - saves bandwidth
    threadgroup half2 shMeans[RENDER_BATCH_SIZE_BUF];
    threadgroup half4 shConics[RENDER_BATCH_SIZE_BUF];
    threadgroup half3 shColors[RENDER_BATCH_SIZE_BUF];
    threadgroup half shOpacities[RENDER_BATCH_SIZE_BUF];
    threadgroup half shDepths[RENDER_BATCH_SIZE_BUF];
    // Per-simd-group done flags (8 simd groups in 16x16 tile)
    threadgroup bool simdDoneFlags[8];

    uint activeCount = activeTileCount[0];
    uint activeIdx = tileCoord.x;
    if (activeIdx >= activeCount) { return; }
    uint tileId = activeTiles[activeIdx];
    uint tilesX = params.tilesX;
    uint tileWidth = params.tileWidth;
    uint tileHeight = params.tileHeight;
    uint localX = localPos3.x;
    uint localY = localPos3.y;
    uint tileX = tileId % tilesX;
    uint tileY = tileId / tilesX;
    uint px = tileX * tileWidth + localX;
    uint py = tileY * tileHeight + localY;
    bool inBounds = (px < params.width) && (py < params.height);

    // Pixel position in half (sufficient for coordinates up to 2048)
    half hx = half(px);
    half hy = half(py);

    // Accumulation in float for stability
    float3 accumColor = float3(0.0f);
    float accumDepth = 0.0f;
    float accumAlpha = 0.0f;
    float trans = 1.0f;

    GaussianHeader header = headers[tileId];
    uint start = header.offset;
    uint count = header.count;

    // Get simd group index (0-7 for 256 threads)
    uint simdGroupIdx = threadIdx / 32;

    // Initialize done flags
    if (threadIdx < 8) {
        simdDoneFlags[threadIdx] = false;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process gaussians in batches using shared memory
    for (uint batch = 0; batch < count; batch += RENDER_BATCH_SIZE_BUF) {
        uint batchCount = min(RENDER_BATCH_SIZE_BUF, count - batch);

        // Cooperative loading (half)
        if (threadIdx < batchCount) {
            uint gIdx = start + batch + threadIdx;
            shMeans[threadIdx] = half2(means[gIdx]);
            shConics[threadIdx] = half4(conics[gIdx]);
            shColors[threadIdx] = half3(colors[gIdx]);
            shOpacities[threadIdx] = min(half(opacities[gIdx]), half(0.99h));
            shDepths[threadIdx] = half(depths[gIdx]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process all gaussians in this batch - BRANCHLESS half precision math
        // Opacity > 0 guaranteed by projection filtering
        if (inBounds && trans > 1e-3f) {
            for (uint i = 0; i < batchCount; ++i) {
                half2 mean = shMeans[i];
                half4 conic = shConics[i];

                // Half precision gaussian evaluation
                // Clamp dx/dy to prevent half overflow (250^2 = 62500 < 65504 max half)
                half dx = clamp(hx - mean.x, half(-250.0h), half(250.0h));
                half dy = clamp(hy - mean.y, half(-250.0h), half(250.0h));
                half quad = dx * dx * conic.x + dy * dy * conic.z + 2.0h * dx * dy * conic.y;

                // Branchless: clamp quad (already bounded by dx/dy clamp, but be safe)
                half clampedQuad = min(quad, half(20.0h));
                half weight = exp(-0.5h * clampedQuad);
                half hAlpha = weight * shOpacities[i];

                // Branchless mask: zero out if quad >= 20 or alpha too small
                half mask = half(quad < 20.0h) * half(hAlpha > 1e-4h);
                hAlpha *= mask;

                // Convert to float for stable accumulation
                float alpha = float(hAlpha);
                float contrib = trans * alpha;
                trans *= (1.0f - alpha);
                accumAlpha += contrib;
                accumColor += float3(shColors[i]) * contrib;
                accumDepth += float(shDepths[i]) * contrib;

                // Early exit when fully opaque (this branch is worth keeping)
                if (trans < 1e-3f) { break; }
            }
        }

        // Check if all simd groups are done (tile-level early exit)
        bool pixelDone = (trans < 1e-3f) || !inBounds;
        bool simdDone = simd_all(pixelDone);
        if (simdDone && simd_is_first()) {
            simdDoneFlags[simdGroupIdx] = true;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Check all flags (small unrolled loop)
        bool allDone = simdDoneFlags[0] && simdDoneFlags[1] && simdDoneFlags[2] && simdDoneFlags[3] &&
                       simdDoneFlags[4] && simdDoneFlags[5] && simdDoneFlags[6] && simdDoneFlags[7];
        if (allDone) {
            break;
        }
    }

    if (inBounds) {
        if (params.whiteBackground != 0) {
            accumColor += float3(trans);
        }
        uint pixelIndex = py * params.width + px;
        uint base = pixelIndex * 3;
        colorOut[base + 0] = accumColor.x;
        colorOut[base + 1] = accumColor.y;
        colorOut[base + 2] = accumColor.z;
        depthOut[pixelIndex] = accumDepth;
        alphaOut[pixelIndex] = accumAlpha;
    }
}

#define instantiate_renderTiles(name, hostName, T, Vec2, Vec4, Packed3) \
    template [[host_name(hostName)]] \
    kernel void renderTiles<T, Vec2, Vec4, Packed3>( \
        const device GaussianHeader* headers [[buffer(0)]], \
        const device Vec2* means [[buffer(1)]], \
        const device Vec4* conics [[buffer(2)]], \
        const device Packed3* colors [[buffer(3)]], \
        const device T* opacities [[buffer(4)]], \
        const device T* depths [[buffer(5)]], \
        device float* colorOut [[buffer(6)]], \
        device float* depthOut [[buffer(7)]], \
        device float* alphaOut [[buffer(8)]], \
        constant RenderParams& params [[buffer(9)]], \
        const device uint* activeTiles [[buffer(10)]], \
        const device uint* activeTileCount [[buffer(11)]], \
        uint3 localPos3 [[thread_position_in_threadgroup]], \
        uint3 tileCoord [[threadgroup_position_in_grid]], \
        uint threadIdx [[thread_index_in_threadgroup]] \
    );

instantiate_renderTiles(float, "renderTilesFloat", float, float2, float4, packed_float3)
instantiate_renderTiles(half, "renderTilesHalf", half, half2, half4, packed_half3)

#undef instantiate_renderTiles

///////////////////////////////////////////////////////////////////////////////
kernel void clearRenderTargetsKernel(
    device float* colorOut [[buffer(0)]],
    device float* depthOut [[buffer(1)]],
    device float* alphaOut [[buffer(2)]],
    constant ClearParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.pixelCount) {
        return;
    }
    float bg = (params.whiteBackground != 0u) ? 1.0f : 0.0f;
    uint base = gid * 3u;
    colorOut[base + 0] = bg;
    colorOut[base + 1] = bg;
    colorOut[base + 2] = bg;
    depthOut[gid] = 0.0f;
    alphaOut[gid] = 0.0f;
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
    half bg = (params.whiteBackground != 0u) ? 1.0h : 0.0h;
    colorTex.write(half4(bg, bg, bg, 1.0h), gid);
    depthTex.write(half4(0.0h), gid);
}

// Reset tile builder state - replaces blit with single-thread compute for lower overhead
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

// Note: tileBoundsKernel has been removed - tile bounds are now computed
// directly in projectGaussiansFused kernel, eliminating the separate pass

// -----------------------------------------------------------------------------
// Visible Gaussian Compaction (Deterministic Prefix-Sum Based)
// Creates a compact list of visible gaussian indices for efficient dispatch
// Uses prefix sum for deterministic ordering (atomic-based compaction was non-deterministic)
// -----------------------------------------------------------------------------

// Pass 1: Mark visibility - write 1 for visible, 0 for culled
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

// Pass 2: After prefix sum on visibilityMarks, scatter visible indices
// The prefix sum gives each visible gaussian a unique write position
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

// Prepare indirect dispatch args based on visible count
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

// Sort key generation - outputs single 32-bit key: [tile:16][depth:16]
// Radix sort processes LSB first, so depth (secondary) in low bits, tile (primary) in high bits
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

///////////////////////////////////////////////////////////////////////////////
//  Fast bitonic sort (shared-memory first/final pass)
///////////////////////////////////////////////////////////////////////////////

static constexpr int genLeftIndex(const uint position, const uint blockSize) {
    const uint blockMask = blockSize - 1;
    const uint no = position & blockMask;
    return ((position & ~blockMask) << 1) | no;
}

static inline bool bitonicKeyLess(uint2 left, uint2 right) {
    return (left.x < right.x) || ((left.x == right.x) && (left.y < right.y));
}

static void bitonicSwap(bool reverse, threadgroup uint2& keyL, threadgroup int& valL, threadgroup uint2& keyR, threadgroup
   int& valR) {
    bool lt = bitonicKeyLess(keyL, keyR);
    bool swap = (!lt) ^ reverse;
    if (swap) {
        uint2 tk = keyL; keyL = keyR; keyR = tk;
        int tv = valL; valL = valR; valR = tv;
    }
}

static void loadShared(const uint threadGroupSize, const uint indexInThreadgroup, const uint position,
                       device uint2* keys, device int* values,
                       threadgroup uint2* sharedKeys, threadgroup int* sharedVals) {
    const uint index = genLeftIndex(position, threadGroupSize);
    sharedKeys[indexInThreadgroup] = keys[index];
    sharedVals[indexInThreadgroup] = values[index];

    uint index2 = index | threadGroupSize;
    sharedKeys[indexInThreadgroup | threadGroupSize] = keys[index2];
    sharedVals[indexInThreadgroup | threadGroupSize] = values[index2];

}

static void storeShared(const uint threadGroupSize, const uint indexInThreadgroup, const uint position,
                        device uint2* keys, device int* values,
                        threadgroup uint2* sharedKeys, threadgroup int* sharedVals) {
    const uint index = genLeftIndex(position, threadGroupSize);
    keys[index] = sharedKeys[indexInThreadgroup];
    values[index] = sharedVals[indexInThreadgroup];

    uint index2 = index | threadGroupSize;
    keys[index2] = sharedKeys[indexInThreadgroup | threadGroupSize];
    values[index2] = sharedVals[indexInThreadgroup | threadGroupSize];

}

kernel void bitonicSortFirstPass(
    device uint2* keys [[buffer(0)]],
    device int* values [[buffer(1)]],
    const device TileAssignmentHeader* header [[buffer(2)]],
    constant uint& logicalBlockSize [[buffer(3)]],
    threadgroup uint2* sharedKeys [[threadgroup(0)]],
    threadgroup int* sharedVals [[threadgroup(1)]],
    const uint threadgroupSize [[threads_per_threadgroup]],
    const uint indexInThreadgroup [[thread_index_in_threadgroup]],
    const uint position [[thread_position_in_grid]]
) {
    uint gridSize = header->paddedCount;
    if (position >= gridSize) { return; }

    loadShared(threadgroupSize, indexInThreadgroup, position, keys, values, sharedKeys, sharedVals);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint unitSize = 1; unitSize <= logicalBlockSize; unitSize <<= 1) {
        bool reverse = (position & unitSize) != 0;
        for (uint blockSize = unitSize; blockSize > 0; blockSize >>= 1) {
            const uint left = genLeftIndex(indexInThreadgroup, blockSize);
            bitonicSwap(reverse, sharedKeys[left], sharedVals[left], sharedKeys[left | blockSize], sharedVals[left | blockSize]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    storeShared(threadgroupSize, indexInThreadgroup, position, keys, values, sharedKeys, sharedVals);
}

kernel void bitonicSortGeneralPass(
    device uint2* keys [[buffer(0)]],
    device int* values [[buffer(1)]],
    const device TileAssignmentHeader* header [[buffer(2)]],
    constant uint2& params [[buffer(3)]],
    const uint position [[thread_position_in_grid]]
) {
    uint gridSize = header->paddedCount;
    if (position >= gridSize) { return; }

    bool reverse = (position & (params.x >> 1)) != 0;
    uint blockSize = params.y;
    const uint left = genLeftIndex(position, blockSize);

    uint2 keyL = keys[left];
    uint2 keyR = keys[left | blockSize];
    int valL = values[left];
    int valR = values[left | blockSize];

    bool lt = bitonicKeyLess(keyL, keyR);
    bool swap = (!lt) ^ reverse;

    if (swap) {
        keys[left] = keyR;
        keys[left | blockSize] = keyL;
        values[left] = valR;
        values[left | blockSize] = valL;
    }
}

kernel void bitonicSortFinalPass(
    device uint2* keys [[buffer(0)]],
    device int* values [[buffer(1)]],
    const device TileAssignmentHeader* header [[buffer(2)]],
    constant uint2& params [[buffer(3)]],
    threadgroup uint2* sharedKeys [[threadgroup(0)]],
    threadgroup int* sharedVals [[threadgroup(1)]],
    const uint threadgroupSize [[threads_per_threadgroup]],
    const uint indexInThreadgroup [[thread_index_in_threadgroup]],
    const uint position [[thread_position_in_grid]]
) {
    uint gridSize = header->paddedCount;
    if (position >= gridSize) { return; }

    loadShared(threadgroupSize, indexInThreadgroup, position, keys, values, sharedKeys, sharedVals);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint unitSize = params.x;
    uint blockSize = params.y;
    bool reverse = (position & (unitSize >> 1)) != 0;

    for (uint width = blockSize; width > 0; width >>= 1) {
        const uint left = genLeftIndex(indexInThreadgroup, width);
        bitonicSwap(reverse, sharedKeys[left], sharedVals[left], sharedKeys[left | width], sharedVals[left | width]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    storeShared(threadgroupSize, indexInThreadgroup, position, keys, values, sharedKeys, sharedVals);
}

struct BitonicICBContainer {
    command_buffer commandBuffer [[ id(0) ]];
};

kernel void encodeBitonicICB(
    constant BitonicICBContainer& icb_container [[buffer(0)]],
    const device TileAssignmentHeader* header [[buffer(1)]],
    const device DispatchIndirectArgs* dispatchArgs [[buffer(2)]],
    const device uint* passTypes [[buffer(3)]], // 0=first,1=general,2=final,3=unused
    constant uint& unitSize [[buffer(4)]],
    constant uint& maxCommands [[buffer(5)]],
    constant uint& firstOffset [[buffer(6)]],
    constant uint& generalOffset [[buffer(7)]],
    constant uint& finalOffset [[buffer(8)]],
    device uint* firstParams [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint firstStride = 256u;
    if (tid != 0) { return; }

    uint pc = max(1u, header[0].paddedCount);
    uint m = 31u - clz(pc);
    uint u = 31u - clz(unitSize);
    uint g = (m > u) ? (m - u) : 0u;
    uint passesNeeded = 1u + g + (g * (g - 1u)) / 2u;
    passesNeeded = min(passesNeeded, maxCommands);

    // First-pass logical block size depends on the actual padded count.
    if (firstParams != nullptr && maxCommands > 0) {
        uint logical = min(unitSize, pc / 2u);
        device uint* firstPtr = (device uint*)((device char*)firstParams + 0u * firstStride);
        *firstPtr = max(1u, logical);
    }

    const device DispatchIndirectArgs* firstArgs = (const device DispatchIndirectArgs*)((const device char*)dispatchArgs + firstOffset);
    const device DispatchIndirectArgs* generalArgs = (const device DispatchIndirectArgs*)((const device char*)dispatchArgs + generalOffset);
    const device DispatchIndirectArgs* finalArgs = (const device DispatchIndirectArgs*)((const device char*)dispatchArgs + finalOffset);

    uint firstGroupsX = firstArgs->threadgroupsPerGridX;
    uint generalGroupsX = generalArgs->threadgroupsPerGridX;
    uint finalGroupsX = finalArgs->threadgroupsPerGridX;

    uint3 tgSize(unitSize, 1, 1);

    for (uint slot = 0; slot < maxCommands; ++slot) {
        compute_command cmd(icb_container.commandBuffer, slot);
        uint3 tgCount(0, 1, 1);

        if (slot < passesNeeded) {
            uint t = passTypes[slot];
            if (t == 0u) {
                tgCount = uint3(firstGroupsX, 1, 1);
            } else if (t == 1u) {
                tgCount = uint3(generalGroupsX, 1, 1);
            } else if (t == 2u) {
                tgCount = uint3(finalGroupsX, 1, 1);
            }
        }

        cmd.concurrent_dispatch_threadgroups(tgCount, tgSize);
    }
}

// =============================================================================
// MARK: - Two-Pass Tile Assignment with Indirect Dispatch
// =============================================================================

// Pass 0: Compact visible gaussians
// Pass 1: Count tiles per visible gaussian (indirect dispatch)
// Pass 2: Prefix sum on counts → offsets
// Pass 3: Scatter tiles using precomputed offsets (indirect dispatch)
// =============================================================================


struct TileAssignParams {
    uint gaussianCount;
    uint tileWidth;
    uint tileHeight;
    uint tilesX;
    uint maxAssignments;
};

// -----------------------------------------------------------------------------
// Pass 2: Prefix sum (block-level reduce + scan)
// Uses optimized primitives from GaussianPrefixScan.h
// -----------------------------------------------------------------------------
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

// Block-level exclusive scan with block sum offsets
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

// Single-block scan for block sums (assumes <= 256 blocks)
// Also computes total and stores it at blockSums[blockCount]
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

// Write total count to header
// Total was precomputed in singleBlockScanKernel and stored at blockSums[blockCount]
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

// -----------------------------------------------------------------------------
// Pass 1 & 3: Indirect Tile Count/Scatter - work with compacted visible indices
// Only processes visible gaussians via indirection through visibleIndices
// -----------------------------------------------------------------------------

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

    const uint aabbWidth = uint(maxTX - minTX + 1);
    const uint aabbHeight = uint(maxTY - minTY + 1);
    const uint aabbCount = aabbWidth * aabbHeight;

    // Use AABB directly - ellipse check disabled for simplicity
    // (Ellipse check was duplicating work between count and scatter)
    tileCounts[gid] = aabbCount;
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

    const uint tilesX = params.tilesX;
    const uint maxAssignments = params.maxAssignments;

    uint writePos = offsets[gid];
    const int gaussianIdx = int(originalIdx);

    // Use AABB directly - ellipse check disabled for simplicity
    // (Ellipse check was duplicating work between count and scatter)
    for (int ty = minTY; ty <= maxTY; ty++) {
        for (int tx = minTX; tx <= maxTX; tx++) {
            if (writePos < maxAssignments) {
                tileIds[writePos] = ty * int(tilesX) + tx;
                tileIndices[writePos] = gaussianIdx;
                writePos++;
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

    header[0].paddedCount = nextPowerOfTwo(total);
    uint padded = header[0].paddedCount;

    uint sortTG = max(config.sortThreadgroupSize, 1u);
    uint fuseTG = max(config.fuseThreadgroupSize, 1u);
    uint unpackTG = max(config.unpackThreadgroupSize, 1u);
    uint packTG = max(config.packThreadgroupSize, 1u);
    uint bitonicTG = max(config.bitonicThreadgroupSize, 1u);
    uint radixBlockSize = max(config.radixBlockSize, 1u);
    uint radixGrainSize = max(config.radixGrainSize, 1u);

    // Sort key generation uses total (radix) or padded (bitonic)
    // For radix path: use total count to avoid processing padding
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

    uint bitonicItems = padded;
    uint bitonicThreads = bitonicItems / 2u;
    uint bitonicGroups = (bitonicThreads > 0u) ? ((bitonicThreads + bitonicTG - 1u) / bitonicTG) : 0u;
    dispatchArgs[DispatchSlotBitonicFirst].threadgroupsPerGridX = bitonicGroups;
    dispatchArgs[DispatchSlotBitonicFirst].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotBitonicFirst].threadgroupsPerGridZ = 1u;
    dispatchArgs[DispatchSlotBitonicGeneral] = dispatchArgs[DispatchSlotBitonicFirst];
    dispatchArgs[DispatchSlotBitonicFinal] = dispatchArgs[DispatchSlotBitonicFirst];

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
#define BLOCK_SIZE 256
#define HISTOGRAM_SCAN_BLOCK_SIZE 256
#define GRAIN_SIZE 4
#define RADIX 256
#define BITS_PER_PASS 8
#define NUM_PASSES (64 / BITS_PER_PASS)
#define SCAN_TYPE_INCLUSIVE (0)
#define SCAN_TYPE_EXCLUSIVE (1)

using KeyType = uint;  // 32-bit key: [tile:16][depth:16]

struct KeyPayload {
    KeyType key;
    uint payload;
};

template <typename T>
struct SumOp {
    inline T operator()(thread const T& a, thread const T& b) const{return a + b;}
    inline T operator()(threadgroup const T& a, thread const T& b) const{return a + b;}
    inline T operator()(threadgroup const T& a, threadgroup const T& b) const{return a + b;}
    inline T operator()(volatile threadgroup const T& a, volatile threadgroup const T& b) const{return a + b;}
    constexpr T identity(){return static_cast<T>(0);}
};

template <typename T>
struct MaxOp {
    inline T operator()(thread const T& a, thread const T& b) const{return max(a,b);}
    inline T operator()(threadgroup const T& a, thread const T& b) const{return max(a,b);}
    inline T operator()(threadgroup const T& a, threadgroup const T& b) const{return max(a,b);}
    inline T operator()(volatile threadgroup const T& a, volatile threadgroup const T& b) const{return max(a,b);}
    constexpr T identity(){ return metal::numeric_limits<T>::min(); }
};

static constexpr ushort RadixToBits(ushort n) {
    return (n-1<2)?1:
    (n-1<4)?2:
    (n-1<8)?3:
    (n-1<16)?4:
    (n-1<32)?5:
    (n-1<64)?6:
    (n-1<128)?7:
    (n-1<256)?8:
    (n-1<512)?9:
    (n-1<1024)?10:
    (n-1<2048)?11:
    (n-1<4096)?12:
    (n-1<8192)?13:
    (n-1<16384)?14:
    (n-1<32768)?15:0;
}

template <ushort R, typename T> static inline ushort
ValueToKeyAtBit(T value, ushort current_bit){
    return (value >> current_bit) & (R - 1);
}

static inline ushort ValueToKeyAtDigit(KeyType value, ushort current_digit){
    ushort bits_to_shift = RadixToBits(RADIX) * current_digit;
    return ValueToKeyAtBit<RADIX>(value, bits_to_shift);
}

template <ushort R> static inline ushort
ValueToKeyAtBit(KeyPayload value, ushort current_bit){
    return ValueToKeyAtBit<R>(value.key, current_bit);
}

template<ushort LENGTH, int SCAN_TYPE, typename BinaryOp, typename T>
static inline T ThreadScan(threadgroup T* values, BinaryOp Op){
    for (ushort i = 1; i < LENGTH; i++){
        values[i] = Op(values[i],values[i - 1]);
    }
    T result = values[LENGTH - 1];
    if (SCAN_TYPE == SCAN_TYPE_EXCLUSIVE){
        for (ushort i = LENGTH - 1; i > 0; i--){
            values[i] = values[i - 1];
        }
        values[0] = 0;
    }
    return result;
}

template <int SCAN_TYPE, typename BinaryOp, typename T> static inline T
SimdgroupScan(T value, ushort local_id, BinaryOp Op){
    const ushort lane_id = local_id % 32;
    T temp = simd_shuffle_up(value, 1);
    if (lane_id >= 1) value = Op(value,temp);
    temp = simd_shuffle_up(value, 2);
    if (lane_id >= 2) value = Op(value,temp);
    temp = simd_shuffle_up(value, 4);
    if (lane_id >= 4) value = Op(value,temp);
    temp = simd_shuffle_up(value, 8);
    if (lane_id >= 8) value = Op(value,temp);
    temp = simd_shuffle_up(value, 16);
    if (lane_id >= 16) value = Op(value,temp);
    if (SCAN_TYPE == SCAN_TYPE_EXCLUSIVE){
        temp = simd_shuffle_up(value, 1);
        value = (lane_id == 0) ? 0 : temp;
    }
    return value;
}

template<ushort LENGTH, typename BinaryOp, typename T> static inline void
ThreadUniformApply(threadgroup T* values, T uni, BinaryOp Op){
    for (ushort i = 0; i < LENGTH; i++){
        values[i] = Op(values[i],uni);
    }
}

template<int SCAN_TYPE, typename BinaryOp, typename T> static T
ThreadgroupPrefixScanStoreSum(T value, thread T& inclusive_sum, threadgroup T* shared, const ushort local_id, BinaryOp Op) {
    shared[local_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_id < 32){
        T partial_sum = ThreadScan<BLOCK_SIZE / 32, SCAN_TYPE>(&shared[local_id * (BLOCK_SIZE / 32)], Op);
        T prefix = SimdgroupScan<SCAN_TYPE_EXCLUSIVE>(partial_sum, local_id, Op);
        ThreadUniformApply<BLOCK_SIZE / 32>(&shared[local_id * (BLOCK_SIZE / 32)], prefix, Op);
        if (local_id == 31) shared[0] = prefix + partial_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (SCAN_TYPE == SCAN_TYPE_INCLUSIVE) value = (local_id == 0) ? value : shared[local_id];
    else value = (local_id == 0) ? 0 : shared[local_id];
    inclusive_sum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return value;
}

template<int SCAN_TYPE, typename BinaryOp, typename T> static T
ThreadgroupPrefixScan(T value, threadgroup T* shared, const ushort local_id, BinaryOp Op) {
    shared[local_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id < 32){
        T partial_sum = ThreadScan<BLOCK_SIZE / 32, SCAN_TYPE>(&shared[local_id * (BLOCK_SIZE / 32)], Op);
        T prefix = SimdgroupScan<SCAN_TYPE_EXCLUSIVE>(partial_sum, local_id, Op);
        ThreadUniformApply<BLOCK_SIZE / 32>(&shared[local_id * (BLOCK_SIZE / 32)], prefix, Op);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = shared[local_id];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return value;
}

template <typename T>
static T SortByTwoBits(const T value, threadgroup T* shared, const ushort local_id, const uchar current_bit){
    uchar mask = ValueToKeyAtBit<4>(value, current_bit);

    uchar4 partial_sum;
    uchar4 scan = {0};
    scan[mask] = 1;
    scan = ThreadgroupPrefixScanStoreSum<SCAN_TYPE_EXCLUSIVE>(scan,
                                                              partial_sum,
                                                              reinterpret_cast<threadgroup uchar4*>(shared),
                                                              local_id,
                                                              SumOp<uchar4>());

    ushort4 offset;
    offset[0] = 0;
    offset[1] = offset[0] + partial_sum[0];
    offset[2] = offset[1] + partial_sum[1];
    offset[3] = offset[2] + partial_sum[2];

    shared[scan[mask] + offset[mask]] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    T result = shared[local_id];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return result;
}

template <typename T>
static T SortByBit(const T value, threadgroup T* shared, const ushort local_id, const uchar current_bit){
    uchar mask = ValueToKeyAtBit<2>(value, current_bit);
    
    uchar2 partial_sum;
    uchar2 scan = {0};
    scan[mask] = 1;
    scan = ThreadgroupPrefixScanStoreSum<SCAN_TYPE_EXCLUSIVE>(scan,
                                                              partial_sum,
                                                              reinterpret_cast<threadgroup uchar2*>(shared),
                                                              local_id,
                                                              SumOp<uchar2>());
    
    ushort2 offset;
    offset[0] = 0;
    offset[1] = offset[0] + partial_sum[0];

    shared[scan[mask] + offset[mask]] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    T result = shared[local_id];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return result;
}

template <typename T>
static T PartialRadixSort(const T value, threadgroup T* shared, const ushort local_id, const ushort current_digit){
    T result = value;
    ushort current_bit = current_digit * RadixToBits(RADIX);
    const ushort key_bits = (ushort)(sizeof(KeyType) * 8);
    const ushort range_end = (ushort)(current_bit + RadixToBits(RADIX));
    const ushort last_bit = min(range_end, key_bits);
    while (current_bit < last_bit){
        ushort remaining = last_bit - current_bit;
        if (remaining >= 2){
            result = SortByTwoBits(result, shared, local_id, current_bit);
            current_bit += 2;
        } else {
            result = SortByBit(result, shared, local_id, current_bit);
            current_bit += 1;
        }
    }
    return result;
}

kernel void radixHistogramKernel(
    device const KeyType*               input_keys     [[buffer(0)]],
    device uint*                        hist_flat      [[buffer(1)]],
    constant uint&                      current_digit  [[buffer(3)]],
    const device TileAssignmentHeader*  header         [[buffer(4)]],
    uint                                grid_size      [[threadgroups_per_grid]],
    uint                                group_id       [[threadgroup_position_in_grid]],
    ushort                              local_id       [[thread_position_in_threadgroup]]
) {
    const uint elementsPerBlock = BLOCK_SIZE * GRAIN_SIZE;

    uint base_id          = group_id * elementsPerBlock;
    uint totalAssignments = header[0].totalAssignments;
    uint paddedCount      = header[0].paddedCount;

    uint bufferRemaining   = (base_id < paddedCount)      ? (paddedCount      - base_id) : 0;
    uint assignmentsRemain = (base_id < totalAssignments) ? (totalAssignments - base_id) : 0;
    uint available         = min(bufferRemaining, assignmentsRemain);

    // 1) Per-group local histogram in LDS
    threadgroup uint local_hist[RADIX];

    for (uint i = 0; i < (RADIX + BLOCK_SIZE - 1u) / BLOCK_SIZE; ++i) {
        uint bin = local_id + i * BLOCK_SIZE;
        if (bin < RADIX) {
            local_hist[bin] = 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) Load keys in blocked fashion per group into registers
    KeyType keys[GRAIN_SIZE];
    KeyType sentinelKey = (KeyType)0;

    load_blocked_local_from_global<GRAIN_SIZE>(
        keys,
        &input_keys[base_id],
        local_id,
        available,
        sentinelKey);

    // 3) Build local histogram using threadgroup atomics
    volatile threadgroup atomic_uint* atomic_hist =
        reinterpret_cast<volatile threadgroup atomic_uint*>(local_hist);

    for (ushort i = 0; i < GRAIN_SIZE; ++i) {
        uint offset_in_block = local_id * GRAIN_SIZE + i;
        uint global_idx      = base_id + offset_in_block;

        if (global_idx < totalAssignments) {
            uchar bin = (uchar)ValueToKeyAtDigit(keys[i], (ushort)current_digit);
            atomic_fetch_add_explicit(&atomic_hist[bin], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4) Write out in bin-major layout: [bin][block]
    for (uint i = 0; i < (RADIX + BLOCK_SIZE - 1u) / BLOCK_SIZE; ++i) {
        uint bin = local_id + i * BLOCK_SIZE;
        if (bin < RADIX) {
            hist_flat[bin * grid_size + group_id] = local_hist[bin];
        }
    }
}

kernel void radixScanBlocksKernel(
    device const uint*                 hist_flat   [[buffer(0)]],
    device uint*                       block_sums  [[buffer(1)]],
    const device TileAssignmentHeader* header      [[buffer(2)]],
    uint                               group_id    [[threadgroup_position_in_grid]],
    ushort                             local_id    [[thread_position_in_threadgroup]]
) {
    threadgroup uint shared_mem[BLOCK_SIZE];

    const uint elementsPerBlock = BLOCK_SIZE * GRAIN_SIZE;
    uint paddedCount   = header[0].paddedCount;
    uint histGridSize  = (paddedCount + elementsPerBlock - 1u) / elementsPerBlock;
    uint num_hist_elem = histGridSize * RADIX;

    uint block_base = group_id * BLOCK_SIZE;
    uint idx        = block_base + local_id;

    uint val = (idx < num_hist_elem) ? hist_flat[idx] : 0u;
    shared_mem[local_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_mem[local_id] += shared_mem[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_id == 0) {
        block_sums[group_id] = shared_mem[0];
    }
}

// Parallel exclusive scan - 256 threads, up to 8192 elements
#define SCAN_GRAIN 32

// Each thread handles SCAN_GRAIN elements, sums them, scans sums, then applies back
kernel void radixExclusiveScanKernel(
    device uint*                       block_sums  [[buffer(0)]],
    const device TileAssignmentHeader* header      [[buffer(1)]],
    threadgroup uint*                  shared_mem  [[threadgroup(0)]],
    ushort                             lid         [[thread_position_in_threadgroup]]
) {

    const uint elementsPerBlock = BLOCK_SIZE * GRAIN_SIZE;
    uint paddedCount   = header[0].paddedCount;
    uint histGridSize  = (paddedCount + elementsPerBlock - 1u) / elementsPerBlock;
    uint num_hist_elem = histGridSize * RADIX;
    uint num_block_sums = (num_hist_elem + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    // Calculate elements per thread (up to SCAN_GRAIN)
    uint elems_per_thread = (num_block_sums + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    elems_per_thread = min(elems_per_thread, (uint)SCAN_GRAIN);

    // Load multiple elements and compute local sum
    uint local_vals[SCAN_GRAIN];
    uint local_sum = 0u;
    uint thread_base = lid * elems_per_thread;

    for (uint i = 0u; i < elems_per_thread; ++i) {
        uint idx = thread_base + i;
        uint val = (idx < num_block_sums) ? block_sums[idx] : 0u;
        local_vals[i] = val;
        local_sum += val;
    }

    // Scan the per-thread sums using helper (exclusive scan of 256 values)
    uint scanned_sum = ThreadgroupPrefixScan<SCAN_TYPE_EXCLUSIVE>(local_sum, shared_mem, lid, SumOp<uint>());

    // Write back: each element gets scanned_sum + prefix of local values
    uint running = scanned_sum;
    for (uint i = 0u; i < elems_per_thread; ++i) {
        uint idx = thread_base + i;
        if (idx < num_block_sums) {
            block_sums[idx] = running;
            running += local_vals[i];
        }
    }
}


kernel void radixApplyScanOffsetsKernel(
    device const uint*                 hist_flat       [[buffer(0)]],
    device const uint*                 scanned_blocks  [[buffer(1)]],
    device uint*                       offsets_flat    [[buffer(2)]],
    const device TileAssignmentHeader* header          [[buffer(3)]],
    threadgroup uint*                  shared_mem      [[threadgroup(0)]],
    uint                               group_id        [[threadgroup_position_in_grid]],
    ushort                             local_id        [[thread_position_in_threadgroup]]
) {
    const uint elementsPerBlock = BLOCK_SIZE * GRAIN_SIZE;
    uint paddedCount   = header[0].paddedCount;
    uint histGridSize  = (paddedCount + elementsPerBlock - 1u) / elementsPerBlock;
    uint num_hist_elem = histGridSize * RADIX;

    uint block_base = group_id * BLOCK_SIZE;
    uint idx        = block_base + local_id;

    uint val = (idx < num_hist_elem) ? hist_flat[idx] : 0u;

    uint local_scanned = ThreadgroupPrefixScan<SCAN_TYPE_EXCLUSIVE>(
        val, shared_mem, local_id, SumOp<uint>());

    uint block_offset = scanned_blocks[group_id];

    if (idx < num_hist_elem) {
        offsets_flat[idx] = local_scanned + block_offset;
    }
}

kernel void radixScatterKernel(
    device KeyType*                    output_keys     [[buffer(0)]],
    device const KeyType*              input_keys      [[buffer(1)]],
    device uint*                       output_payload  [[buffer(2)]],
    device const uint*                 input_payload   [[buffer(3)]],
    device const uint*                 offsets_flat    [[buffer(5)]],
    constant uint&                     current_digit   [[buffer(6)]],
    const device TileAssignmentHeader* header          [[buffer(7)]],
    uint                               group_id        [[threadgroup_position_in_grid]],
    uint                               grid_size       [[threadgroups_per_grid]],
    ushort                             local_id        [[thread_position_in_threadgroup]]
) {
    const uint elementsPerBlock = BLOCK_SIZE * GRAIN_SIZE;

    uint base_id          = group_id * elementsPerBlock;
    uint totalAssignments = header[0].totalAssignments;
    uint paddedCount      = header[0].paddedCount;

    uint bufferRemaining   = (base_id < paddedCount)      ? (paddedCount      - base_id) : 0;
    uint assignmentsRemain = (base_id < totalAssignments) ? (totalAssignments - base_id) : 0;
    uint available         = min(bufferRemaining, assignmentsRemain);

    // Use max key as sentinel
    constexpr KeyType keySentinel = ~(KeyType)0;

    // 1) Load keys / payloads in STRIPED fashion
    KeyType keys[GRAIN_SIZE];
    uint payloads[GRAIN_SIZE];
    load_striped_local_from_global<GRAIN_SIZE, BLOCK_SIZE>(keys, &input_keys[base_id], local_id, available, keySentinel);
    load_striped_local_from_global<GRAIN_SIZE, BLOCK_SIZE>(payloads, &input_payload[base_id], local_id, available, UINT_MAX);

    // 2) Shared memory for bin offsets and ranking
    threadgroup uint global_bin_base[RADIX];
    threadgroup KeyPayload tg_kp[BLOCK_SIZE];
    threadgroup ushort tg_short[BLOCK_SIZE];

    // Load global bin offsets
    for (uint i = 0; i < (RADIX + BLOCK_SIZE - 1u) / BLOCK_SIZE; ++i) {
        uint bin = local_id + i * BLOCK_SIZE;
        if (bin < RADIX) {
            global_bin_base[bin] = offsets_flat[bin * grid_size + group_id];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) Process chunks with optimized ranking
    for (ushort chunk = 0; chunk < GRAIN_SIZE; ++chunk) {
        KeyPayload kp;
        kp.key = keys[chunk];
        kp.payload = payloads[chunk];

        // Partial radix sort: groups same-bin elements together, preserving relative order
        kp = PartialRadixSort(kp, tg_kp, local_id, (ushort)current_digit);

        // Extract bin after sorting
        ushort my_bin = ValueToKeyAtDigit(kp.key, (ushort)current_digit);

        // Head discontinuity + max scan to find run start
        uchar head_flag = flag_head_discontinuity<BLOCK_SIZE>(my_bin, tg_short, local_id);
        ushort run_start = ThreadgroupPrefixScan<SCAN_TYPE_INCLUSIVE>(
            head_flag ? local_id : (ushort)0, tg_short, local_id, MaxOp<ushort>());
        ushort local_offset = local_id - run_start;

        // Tail discontinuity for offset updates
        uchar tail_flag = flag_tail_discontinuity<BLOCK_SIZE>(my_bin, tg_short, local_id);

        // Check validity and scatter
        bool is_valid = (kp.payload != UINT_MAX);
        if (is_valid) {
            uint dst = global_bin_base[my_bin] + local_offset;
            output_keys[dst] = kp.key;
            output_payload[dst] = kp.payload;
        }

        // Update global offsets at run boundaries
        if (tail_flag && is_valid) {
            global_bin_base[my_bin] += local_offset + 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// =============================================================================
// MARK: - GlobalSort Render Kernel
// =============================================================================
// 32x16 tile, 8x8 threadgroup (64 threads), 4x2 pixels per thread = 512 pixels
#define RENDER_V3_TG_WIDTH 8
#define RENDER_V3_TG_HEIGHT 8
#define RENDER_V3_PIXELS_X 4
#define RENDER_V3_PIXELS_Y 2

// =============================================================================
// Unified Index-based render (like Local)
// =============================================================================
// - Only outputs color + depth (alpha stored in color.a, like Local)
// - Uses index-based access via sortedIndices (no Pack step needed)
// - Half precision only (matching Local)

kernel void globalSortRender(
    const device GaussianHeader*     headers         [[buffer(0)]],
    const device GaussianRenderData* gaussians       [[buffer(1)]],  // interleavedGaussians
    const device int*                sortedIndices   [[buffer(2)]],  // indices into gaussians
    const device uint*               activeTiles     [[buffer(3)]],
    const device uint*               activeTileCount [[buffer(4)]],
    texture2d<half, access::write>   colorOut        [[texture(0)]],
    texture2d<half, access::write>   depthOut        [[texture(1)]],  // half like Local
    constant RenderParams&           params          [[buffer(5)]],
    uint2                            group_id        [[threadgroup_position_in_grid]],
    uint2                            local_id        [[thread_position_in_threadgroup]]
) {
    uint tileIdx = group_id.x;
    if (tileIdx >= *activeTileCount) return;

    uint tile = activeTiles[tileIdx];
    GaussianHeader hdr = headers[tile];

    uint tileX = tile % params.tilesX;
    uint tileY = tile / params.tilesX;

    // 32x16 tile, 8x8 threads, 4x2 pixels per thread
    uint baseX = tileX * 32 + local_id.x * RENDER_V3_PIXELS_X;
    uint baseY = tileY * 16 + local_id.y * RENDER_V3_PIXELS_Y;

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

    half depth00 = 0, depth10 = 0, depth20 = 0, depth30 = 0;
    half depth01 = 0, depth11 = 0, depth21 = 0, depth31 = 0;

    uint start = hdr.offset;
    uint count = hdr.count;

    // Direct loop - using sortedIndices for indirect access (like Local)
    for (uint i = 0; i < count; i++) {
        // Early exit when all pixels are saturated
        half maxTrans0 = max(max(trans00, trans10), max(trans20, trans30));
        half maxTrans1 = max(max(trans01, trans11), max(trans21, trans31));
        if (max(maxTrans0, maxTrans1) < half(1.0h/255.0h)) break;

        // Read via sortedIndices (like Local - no Pack step)
        int gaussianIdx = sortedIndices[start + i];
        if (gaussianIdx < 0) continue;
        GaussianRenderData g = gaussians[gaussianIdx];

        // Precompute covariance terms (using accessor functions)
        half4 gConic = getConic(g);
        half cxx = gConic.x;
        half cyy = gConic.z;
        half cxy2 = 2.0h * gConic.y;
        half opacity = g.opacity;
        half3 gColor = getColor(g);
        half gDepth = g.depth;

        // Direction vectors for 8 pixels
        half2 gMean = getMean(g);
        half2 d00 = pos00 - gMean;
        half2 d10 = pos10 - gMean;
        half2 d20 = pos20 - gMean;
        half2 d30 = pos30 - gMean;
        half2 d01 = pos01 - gMean;
        half2 d11 = pos11 - gMean;
        half2 d21 = pos21 - gMean;
        half2 d31 = pos31 - gMean;

        // Quadratic form: power = cxx*dx*dx + cyy*dy*dy + cxy2*dx*dy
        half p00 = d00.x*d00.x*cxx + d00.y*d00.y*cyy + d00.x*d00.y*cxy2;
        half p10 = d10.x*d10.x*cxx + d10.y*d10.y*cyy + d10.x*d10.y*cxy2;
        half p20 = d20.x*d20.x*cxx + d20.y*d20.y*cyy + d20.x*d20.y*cxy2;
        half p30 = d30.x*d30.x*cxx + d30.y*d30.y*cyy + d30.x*d30.y*cxy2;
        half p01 = d01.x*d01.x*cxx + d01.y*d01.y*cyy + d01.x*d01.y*cxy2;
        half p11 = d11.x*d11.x*cxx + d11.y*d11.y*cyy + d11.x*d11.y*cxy2;
        half p21 = d21.x*d21.x*cxx + d21.y*d21.y*cyy + d21.x*d21.y*cxy2;
        half p31 = d31.x*d31.x*cxx + d31.y*d31.y*cyy + d31.x*d31.y*cxy2;

        // Alpha = opacity * exp(-0.5 * power)
        half a00 = min(opacity * exp(-0.5h * p00), 0.99h);
        half a10 = min(opacity * exp(-0.5h * p10), 0.99h);
        half a20 = min(opacity * exp(-0.5h * p20), 0.99h);
        half a30 = min(opacity * exp(-0.5h * p30), 0.99h);
        half a01 = min(opacity * exp(-0.5h * p01), 0.99h);
        half a11 = min(opacity * exp(-0.5h * p11), 0.99h);
        half a21 = min(opacity * exp(-0.5h * p21), 0.99h);
        half a31 = min(opacity * exp(-0.5h * p31), 0.99h);

        // Early exit: skip if all alphas are zero (no contribution to any pixel)
        half4 alphaRow0 = half4(a00, a10, a20, a30);
        half4 alphaRow1 = half4(a01, a11, a21, a31);
        if (all(alphaRow0 == 0.0h) && all(alphaRow1 == 0.0h)) continue;

        // Blend color and depth
        color00 += gColor * (a00 * trans00); depth00 += gDepth * (a00 * trans00);
        color10 += gColor * (a10 * trans10); depth10 += gDepth * (a10 * trans10);
        color20 += gColor * (a20 * trans20); depth20 += gDepth * (a20 * trans20);
        color30 += gColor * (a30 * trans30); depth30 += gDepth * (a30 * trans30);
        color01 += gColor * (a01 * trans01); depth01 += gDepth * (a01 * trans01);
        color11 += gColor * (a11 * trans11); depth11 += gDepth * (a11 * trans11);
        color21 += gColor * (a21 * trans21); depth21 += gDepth * (a21 * trans21);
        color31 += gColor * (a31 * trans31); depth31 += gDepth * (a31 * trans31);

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

    // Write 8 pixels (color with alpha in .a channel, depth in separate texture)
    uint w = params.width, h = params.height;
    if (baseX + 0 < w && baseY + 0 < h) {
        colorOut.write(half4(color00, 1.0h - trans00), uint2(baseX + 0, baseY + 0));
        depthOut.write(half4(depth00, 0, 0, 0), uint2(baseX + 0, baseY + 0));
    }
    if (baseX + 1 < w && baseY + 0 < h) {
        colorOut.write(half4(color10, 1.0h - trans10), uint2(baseX + 1, baseY + 0));
        depthOut.write(half4(depth10, 0, 0, 0), uint2(baseX + 1, baseY + 0));
    }
    if (baseX + 2 < w && baseY + 0 < h) {
        colorOut.write(half4(color20, 1.0h - trans20), uint2(baseX + 2, baseY + 0));
        depthOut.write(half4(depth20, 0, 0, 0), uint2(baseX + 2, baseY + 0));
    }
    if (baseX + 3 < w && baseY + 0 < h) {
        colorOut.write(half4(color30, 1.0h - trans30), uint2(baseX + 3, baseY + 0));
        depthOut.write(half4(depth30, 0, 0, 0), uint2(baseX + 3, baseY + 0));
    }
    if (baseX + 0 < w && baseY + 1 < h) {
        colorOut.write(half4(color01, 1.0h - trans01), uint2(baseX + 0, baseY + 1));
        depthOut.write(half4(depth01, 0, 0, 0), uint2(baseX + 0, baseY + 1));
    }
    if (baseX + 1 < w && baseY + 1 < h) {
        colorOut.write(half4(color11, 1.0h - trans11), uint2(baseX + 1, baseY + 1));
        depthOut.write(half4(depth11, 0, 0, 0), uint2(baseX + 1, baseY + 1));
    }
    if (baseX + 2 < w && baseY + 1 < h) {
        colorOut.write(half4(color21, 1.0h - trans21), uint2(baseX + 2, baseY + 1));
        depthOut.write(half4(depth21, 0, 0, 0), uint2(baseX + 2, baseY + 1));
    }
    if (baseX + 3 < w && baseY + 1 < h) {
        colorOut.write(half4(color31, 1.0h - trans31), uint2(baseX + 3, baseY + 1));
        depthOut.write(half4(depth31, 0, 0, 0), uint2(baseX + 3, baseY + 1));
    }
}

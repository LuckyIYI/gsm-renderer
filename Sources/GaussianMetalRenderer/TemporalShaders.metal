// Copyright (C) 2024 - Temporal Coherent Gaussian Splatting Pipeline
// Standalone archive of temporal sorting experiments
// Key insight: Insertion sort M candidates per tile per frame into persistent list
// Over multiple frames, the top-K closest gaussians emerge

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// =============================================================================
// SHARED CONSTANTS
// =============================================================================

constant float LN2 = 0.693147180559945f;

// Temporal buffer size constants
#define TEMPORAL_MAX_PER_TILE 512
#define TEMPORAL_MAX_CANDIDATES 64

// Render tile sizes
#define TEMPORAL_TILE_WIDTH 32
#define TEMPORAL_TILE_HEIGHT 16
#define TEMPORAL_TG_WIDTH 8
#define TEMPORAL_TG_HEIGHT 8
#define TEMPORAL_PIXELS_X 4
#define TEMPORAL_PIXELS_Y 2

// Stochastic mode uses smaller tiles (8x8) for better efficiency
#define STOCHASTIC_TILE_WIDTH 8
#define STOCHASTIC_TILE_HEIGHT 8
#define STOCHASTIC_TG_WIDTH 4
#define STOCHASTIC_TG_HEIGHT 4
#define STOCHASTIC_PIXELS_X 2
#define STOCHASTIC_PIXELS_Y 2

// =============================================================================
// DATA STRUCTURES
// =============================================================================

/// Camera uniforms (matches Swift CameraUniformsSwift)
struct TemporalCameraUniforms {
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float3 cameraCenter;
    float pixelFactor;
    float focalX;
    float focalY;
    float width;
    float height;
    float nearPlane;
    float farPlane;
    uint shComponents;
    uint gaussianCount;
    uint padding0;
    uint padding1;
};

/// Packed world gaussian input (float32) - 48 bytes
struct TemporalPackedWorld {
    packed_float3 position;
    float         opacity;
    packed_float3 scale;
    float         _pad0;
    float4        rotation;
};

/// Packed world gaussian input (half16) - 24 bytes
struct TemporalPackedWorldHalf {
    packed_half3  position;
    half          opacity;
    packed_half3  scale;
    half          _pad0;
    half4         rotation;
};

/// Compacted gaussian (after projection + culling) - 48 bytes
struct TemporalCompactedGaussian {
    float4 covariance_depth;    // conic.xyz + depth
    float4 position_color;      // pos.xy + packed_half4(color,opacity)
    int2 min_tile;
    int2 max_tile;
};

/// Parameters for project+compact kernel
struct TemporalProjectParams {
    uint gaussianCount;
    uint tilesX;
    uint tilesY;
    uint tileWidth;
    uint tileHeight;
    uint surfaceWidth;
    uint surfaceHeight;
    uint maxCompacted;
};

/// Header for compacted gaussians
struct TemporalCompactedHeader {
    atomic_uint visibleCount;
    uint maxCompacted;
    uint overflow;
    uint _pad;
};

/// Render parameters
struct TemporalRenderParams {
    uint width;
    uint height;
    uint tileWidth;
    uint tileHeight;
    uint tilesX;
    uint tilesY;
    uint whiteBackground;
    uint _pad;
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

inline float3x3 temporal_matrixFromRows(float3 r0, float3 r1, float3 r2) {
    return float3x3(
        float3(r0.x, r1.x, r2.x),
        float3(r0.y, r1.y, r2.y),
        float3(r0.z, r1.z, r2.z)
    );
}

inline float4 temporal_normalizeQuat(float4 q) {
    float d = dot(q, q);
    float r = rsqrt(max(d, 1e-8f));
    return q * r;
}

inline float3x3 temporal_buildCov3D(float3 scale, float4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;

    float3x3 R = float3x3(
        float3(1 - (yy + zz), xy + wz, xz - wy),
        float3(xy - wz, 1 - (xx + zz), yz + wx),
        float3(xz + wy, yz - wx, 1 - (xx + yy))
    );

    float3x3 M = float3x3(R[0] * scale.x, R[1] * scale.y, R[2] * scale.z);
    return M * transpose(M);
}

inline float2x2 temporal_projectCov(float3x3 cov3d, float3 viewPos, float3x3 viewRot,
                                     float fx, float fy, float W, float H) {
    float z = viewPos.z;
    float limX = 1.3f * W / (2.0f * fx);
    float limY = 1.3f * H / (2.0f * fy);

    float invZ = 1.0f / max(1e-4f, z);
    float invZ2 = invZ * invZ;

    float x = clamp(viewPos.x * invZ, -limX, limX) * z;
    float y = clamp(viewPos.y * invZ, -limY, limY) * z;

    float3 col0 = float3(fx * invZ, 0.0f, 0.0f);
    float3 col1 = float3(0.0f, fy * invZ, 0.0f);
    float3 col2 = float3(-(fx * x) * invZ2, -(fy * y) * invZ2, 0.0f);
    float3x3 J = float3x3(col0, col1, col2);

    float3x3 T = J * viewRot;
    float3x3 cov2d_full = T * cov3d * transpose(T);

    return float2x2(
        cov2d_full[0][0] + 0.3f, cov2d_full[0][1],
        cov2d_full[0][1], cov2d_full[1][1] + 0.3f
    );
}

// Pack half4 to float2
inline float2 temporal_packHalf4(half4 v) {
    uint2 u = uint2(as_type<uint>(half2(v.xy)), as_type<uint>(half2(v.zw)));
    return float2(as_type<float>(u.x), as_type<float>(u.y));
}

// Unpack float2 to half4
inline half4 temporal_unpackHalf4(float2 v) {
    uint2 u = uint2(as_type<uint>(v.x), as_type<uint>(v.y));
    return half4(as_type<half2>(u.x), as_type<half2>(u.y));
}

// =============================================================================
// FLASHGS-STYLE ELLIPSE-TILE INTERSECTION
// =============================================================================

inline bool temporal_segmentIntersectEllipse(float a, float b, float c, float d, float l, float r) {
    float delta = b * b - 4.0f * a * c;
    float t1 = (l - d) * (2.0f * a) + b;
    float t2 = (r - d) * (2.0f * a) + b;
    return delta >= 0.0f && (t1 <= 0.0f || t1 * t1 <= delta) && (t2 >= 0.0f || t2 * t2 <= delta);
}

inline bool temporal_blockIntersectEllipse(int2 pix_min, int2 pix_max, float2 center, float3 conic, float power) {
    float w = 2.0f * power;
    float dx, dy;
    float a, b, c;

    if (center.x * 2.0f < float(pix_min.x + pix_max.x)) {
        dx = center.x - float(pix_min.x);
    } else {
        dx = center.x - float(pix_max.x);
    }
    a = conic.z;
    b = -2.0f * conic.y * dx;
    c = conic.x * dx * dx - w;

    if (temporal_segmentIntersectEllipse(a, b, c, center.y, float(pix_min.y), float(pix_max.y))) {
        return true;
    }

    if (center.y * 2.0f < float(pix_min.y + pix_max.y)) {
        dy = center.y - float(pix_min.y);
    } else {
        dy = center.y - float(pix_max.y);
    }
    a = conic.x;
    b = -2.0f * conic.y * dy;
    c = conic.z * dy * dy - w;

    if (temporal_segmentIntersectEllipse(a, b, c, center.x, float(pix_min.x), float(pix_max.x))) {
        return true;
    }

    return false;
}

inline bool temporal_blockContainsCenter(int2 pix_min, int2 pix_max, float2 center) {
    return center.x >= float(pix_min.x) && center.x <= float(pix_max.x) &&
           center.y >= float(pix_min.y) && center.y <= float(pix_max.y);
}

inline float temporal_computePower(float opacity) {
    return LN2 * 8.0f + LN2 * log2(max(opacity, 1e-6f));
}

// =============================================================================
// HASH FUNCTIONS
// =============================================================================

inline float temporal_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) / 4294967295.0f;
}

inline uint temporal_hash_uint(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// =============================================================================
// KEY COMPUTATION HELPERS
// =============================================================================

/// Compute DEPTH key for front-to-back sorting
inline uint temporal_compute_depth_key(TemporalCompactedGaussian g) {
    return ((as_type<uint>(g.covariance_depth.w) ^ 0x80000000u) >> 16u) & 0xFFFFu;
}

/// Compute stable depth key with world index for tie-breaking
inline uint temporal_compute_stable_depth_key(TemporalCompactedGaussian g, uint worldIdx) {
    uint depth16 = ((as_type<uint>(g.covariance_depth.w) ^ 0x80000000u) >> 16u) & 0xFFFFu;
    return (depth16 << 16) | (worldIdx & 0xFFFFu);
}

/// Compute importance score for a gaussian (for score-based sorting)
inline uint temporal_compute_score_key(TemporalCompactedGaussian g) {
    half4 colorOp = temporal_unpackHalf4(g.position_color.zw);
    float opacity = float(colorOp.w);
    float3 conic = g.covariance_depth.xyz;
    float detConic = conic.x * conic.z - conic.y * conic.y;
    float area = (detConic > 1e-6f) ? 1.0f / sqrt(detConic) : 0.0f;
    float score = clamp(opacity * area, 0.0f, 1e6f);
    return 0xFFFFFFFFu - as_type<uint>(score);
}

// =============================================================================
// KERNEL: CLEAR TEMPORAL BUFFERS
// =============================================================================

kernel void temporal_clear(
    device uint* candidateCounts [[buffer(0)]],
    device uint* worldToCompacted [[buffer(1)]],
    device uint* candidateIndices [[buffer(2)]],
    constant uint& tileCount [[buffer(3)]],
    constant uint& maxGaussians [[buffer(4)]],
    constant uint& maxCandidates [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < maxGaussians) {
        worldToCompacted[gid] = 0xFFFFFFFF;
    }
    uint totalCandidateSlots = tileCount * maxCandidates;
    if (gid < totalCandidateSlots) {
        candidateIndices[gid] = 0xFFFFFFFF;
    }
}

// =============================================================================
// KERNEL: INITIALIZE SORTED BUFFERS
// =============================================================================

kernel void temporal_init_sorted(
    device uint* sortedKeys [[buffer(0)]],
    device uint* sortedIndices [[buffer(1)]],
    device uint* sortedCounts [[buffer(2)]],
    constant uint& tileCount [[buffer(3)]],
    constant uint& maxPerTile [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint totalSlots = tileCount * maxPerTile;
    if (gid < totalSlots) {
        sortedKeys[gid] = 0xFFFFFFFF;
        sortedIndices[gid] = 0xFFFFFFFF;
    }
    if (gid < tileCount) {
        sortedCounts[gid] = 0;
    }
}

// =============================================================================
// KERNEL: SCATTER CANDIDATES (STOCHASTIC HASH-BASED)
// =============================================================================

kernel void temporal_scatter_candidates_depth(
    const device TemporalCompactedGaussian* compacted [[buffer(0)]],
    const device TemporalCompactedHeader* header [[buffer(1)]],
    const device uint* compactedToWorld [[buffer(2)]],
    device uint* candidateKeys [[buffer(3)]],
    device uint* candidateIndices [[buffer(4)]],
    constant uint& tilesX [[buffer(5)]],
    constant uint& maxCandidates [[buffer(6)]],
    constant uint& frameId [[buffer(7)]],
    constant int& tileWidth [[buffer(8)]],
    constant int& tileHeight [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint visibleCount = atomic_load_explicit(&header->visibleCount, memory_order_relaxed);
    if (gid >= visibleCount) return;

    TemporalCompactedGaussian g = compacted[gid];
    uint compactedIdx = gid;
    uint worldIdx = compactedToWorld[compactedIdx];

    int minTX = g.min_tile.x, minTY = g.min_tile.y;
    int maxTX = g.max_tile.x, maxTY = g.max_tile.y;

    if (minTX >= maxTX || minTY >= maxTY) return;

    uint depth16 = ((as_type<uint>(g.covariance_depth.w) ^ 0x80000000u) >> 16u) & 0xFFFFu;
    uint stableKey = (depth16 << 16) | (compactedIdx & 0xFFFFu);

    float3 conic = g.covariance_depth.xyz;
    float2 center = g.position_color.xy;
    half4 colorOp = temporal_unpackHalf4(g.position_color.zw);
    float power = temporal_computePower(float(colorOp.w));

    for (int ty = minTY; ty < maxTY; ++ty) {
        for (int tx = minTX; tx < maxTX; ++tx) {
            int2 pix_min = int2(tx * tileWidth, ty * tileHeight);
            int2 pix_max = int2(pix_min.x + tileWidth - 1, pix_min.y + tileHeight - 1);

            bool valid = temporal_blockContainsCenter(pix_min, pix_max, center) ||
                         temporal_blockIntersectEllipse(pix_min, pix_max, center, conic, power);

            if (!valid) continue;

            uint tileId = uint(ty) * tilesX + uint(tx);

            uint h = worldIdx;
            h ^= frameId * 0x85ebca6bu;
            h ^= tileId * 0xc2b2ae35u;
            h ^= h >> 16;
            h *= 0x85ebca6bu;
            h ^= h >> 13;
            uint slot = h % maxCandidates;
            uint offset = tileId * maxCandidates + slot;

            candidateKeys[offset] = stableKey;
            candidateIndices[offset] = worldIdx;
        }
    }
}

// =============================================================================
// KERNEL: SCATTER CANDIDATES (HONEST - ATOMIC COUNTERS)
// =============================================================================

kernel void temporal_scatter_candidates_honest(
    const device TemporalCompactedGaussian* compacted [[buffer(0)]],
    const device TemporalCompactedHeader* header [[buffer(1)]],
    const device uint* compactedToWorld [[buffer(2)]],
    device atomic_uint* candidateCounts [[buffer(3)]],
    device uint* candidateKeys [[buffer(4)]],
    device uint* candidateIndices [[buffer(5)]],
    constant uint& tilesX [[buffer(6)]],
    constant uint& maxCandidates [[buffer(7)]],
    constant int& tileWidth [[buffer(8)]],
    constant int& tileHeight [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint visibleCount = atomic_load_explicit(&header->visibleCount, memory_order_relaxed);
    if (gid >= visibleCount) return;

    TemporalCompactedGaussian g = compacted[gid];
    uint indexToStore = gid;

    int minTX = g.min_tile.x, minTY = g.min_tile.y;
    int maxTX = g.max_tile.x, maxTY = g.max_tile.y;

    if (minTX >= maxTX || minTY >= maxTY) return;

    uint depthKey = temporal_compute_stable_depth_key(g, gid);

    float3 conic = g.covariance_depth.xyz;
    float2 center = g.position_color.xy;
    half4 colorOp = temporal_unpackHalf4(g.position_color.zw);
    float power = temporal_computePower(float(colorOp.w));

    for (int ty = minTY; ty < maxTY; ++ty) {
        for (int tx = minTX; tx < maxTX; ++tx) {
            int2 pix_min = int2(tx * tileWidth, ty * tileHeight);
            int2 pix_max = int2(pix_min.x + tileWidth - 1, pix_min.y + tileHeight - 1);

            bool valid = temporal_blockContainsCenter(pix_min, pix_max, center) ||
                         temporal_blockIntersectEllipse(pix_min, pix_max, center, conic, power);

            if (!valid) continue;

            uint tileId = uint(ty) * tilesX + uint(tx);
            uint slot = atomic_fetch_add_explicit(&candidateCounts[tileId], 1, memory_order_relaxed);

            if (slot < maxCandidates) {
                uint offset = tileId * maxCandidates + slot;
                candidateKeys[offset] = depthKey;
                candidateIndices[offset] = indexToStore;
            }
        }
    }
}

// =============================================================================
// KERNEL: RECOMPUTE SORTED SCORES
// =============================================================================

kernel void temporal_recompute_sorted_scores(
    device uint* sortedKeys [[buffer(0)]],
    device uint* sortedIndices [[buffer(1)]],
    device uint* sortedCounts [[buffer(2)]],
    const device TemporalCompactedGaussian* compacted [[buffer(3)]],
    const device uint* worldToCompacted [[buffer(4)]],
    constant uint& maxPerTile [[buffer(5)]],
    uint tileId [[thread_position_in_grid]]
) {
    uint sortedOffset = tileId * maxPerTile;
    uint count = sortedCounts[tileId];

    if (count == 0) return;

    uint newCount = 0;
    for (uint i = 0; i < count; ++i) {
        uint worldIdx = sortedIndices[sortedOffset + i];
        uint compactedIdx = worldToCompacted[worldIdx];
        if (compactedIdx == 0xFFFFFFFF) {
            sortedKeys[sortedOffset + i] = 0xFFFFFFFFu;
            continue;
        }
        TemporalCompactedGaussian g = compacted[compactedIdx];
        uint scoreKey = temporal_compute_score_key(g);
        sortedKeys[sortedOffset + i] = scoreKey;
        newCount++;
    }

    uint writePos = 0;
    for (uint i = 0; i < count; ++i) {
        if (sortedKeys[sortedOffset + i] != 0xFFFFFFFFu) {
            if (writePos != i) {
                sortedKeys[sortedOffset + writePos] = sortedKeys[sortedOffset + i];
                sortedIndices[sortedOffset + writePos] = sortedIndices[sortedOffset + i];
            }
            writePos++;
        }
    }
    count = writePos;

    for (uint i = 1; i < count; ++i) {
        uint key = sortedKeys[sortedOffset + i];
        uint idx = sortedIndices[sortedOffset + i];
        uint j = i;
        while (j > 0 && sortedKeys[sortedOffset + j - 1] > key) {
            sortedKeys[sortedOffset + j] = sortedKeys[sortedOffset + j - 1];
            sortedIndices[sortedOffset + j] = sortedIndices[sortedOffset + j - 1];
            j--;
        }
        sortedKeys[sortedOffset + j] = key;
        sortedIndices[sortedOffset + j] = idx;
    }

    sortedCounts[tileId] = count;
}

// =============================================================================
// KERNEL: RECOMPUTE SORTED DEPTHS (PARALLEL)
// =============================================================================

kernel void temporal_recompute_sorted_depths(
    device uint* sortedKeys [[buffer(0)]],
    const device uint* sortedIndices [[buffer(1)]],
    const device uint* sortedCounts [[buffer(2)]],
    const device TemporalCompactedGaussian* compacted [[buffer(3)]],
    const device uint* worldToCompacted [[buffer(4)]],
    constant uint& maxPerTile [[buffer(5)]],
    constant uint& tileCount [[buffer(6)]],
    constant uint& tilesX [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint tileId = gid / maxPerTile;
    uint entryIdx = gid % maxPerTile;

    if (tileId >= tileCount) return;

    uint count = sortedCounts[tileId];
    if (entryIdx >= count) return;

    uint sortedOffset = tileId * maxPerTile;
    uint worldIdx = sortedIndices[sortedOffset + entryIdx];

    if (worldIdx == 0xFFFFFFFF) {
        sortedKeys[sortedOffset + entryIdx] = 0xFFFFFFFFu;
        return;
    }

    uint compactedIdx = worldToCompacted[worldIdx];
    if (compactedIdx == 0xFFFFFFFF) {
        sortedKeys[sortedOffset + entryIdx] = 0xFFFFFFFFu;
        return;
    }

    TemporalCompactedGaussian g = compacted[compactedIdx];

    uint tileY = tileId / tilesX;
    uint tileX = tileId % tilesX;
    if (int(tileX) < g.min_tile.x || int(tileX) >= g.max_tile.x ||
        int(tileY) < g.min_tile.y || int(tileY) >= g.max_tile.y) {
        sortedKeys[sortedOffset + entryIdx] = 0xFFFFFFFFu;
        return;
    }

    uint depth16 = ((as_type<uint>(g.covariance_depth.w) ^ 0x80000000u) >> 16u) & 0xFFFFu;
    uint depthKey = (depth16 << 16) | (worldIdx & 0xFFFFu);
    sortedKeys[sortedOffset + entryIdx] = depthKey;
}

// =============================================================================
// KERNEL: RESORT AFTER RECOMPUTE
// =============================================================================

kernel void temporal_resort_after_recompute(
    device uint* sortedKeys [[buffer(0)]],
    device uint* sortedIndices [[buffer(1)]],
    device uint* sortedCounts [[buffer(2)]],
    constant uint& maxPerTile [[buffer(3)]],
    device uint* bloomFilters [[buffer(4)]],
    uint tileId [[thread_position_in_grid]]
) {
    uint offset = tileId * maxPerTile;
    uint bloomOffset = tileId * 64;
    uint count = sortedCounts[tileId];

    if (count == 0) {
        for (uint b = 0; b < 64; ++b) {
            bloomFilters[bloomOffset + b] = 0;
        }
        return;
    }

    uint keys[256];
    uint indices[256];
    for (uint i = 0; i < count && i < 256; ++i) {
        keys[i] = sortedKeys[offset + i];
        indices[i] = sortedIndices[offset + i];
    }
    count = min(count, 256u);

    for (uint i = 1; i < count; ++i) {
        uint key = keys[i];
        uint idx = indices[i];
        uint j = i;
        while (j > 0 && keys[j - 1] > key) {
            keys[j] = keys[j - 1];
            indices[j] = indices[j - 1];
            j--;
        }
        keys[j] = key;
        indices[j] = idx;
    }

    uint bloom[64];
    for (uint b = 0; b < 64; ++b) bloom[b] = 0;

    uint validCount = 0;
    for (uint i = 0; i < count; ++i) {
        if (keys[i] < 0xFFFFFFFFu) {
            validCount++;
            uint worldIdx = indices[i];
            uint h1 = worldIdx & 0x7FF;
            uint h2 = ((worldIdx * 2654435761u) >> 21) & 0x7FF;
            bloom[h1 >> 5] |= (1u << (h1 & 31));
            bloom[h2 >> 5] |= (1u << (h2 & 31));
        }
    }

    for (uint i = 0; i < count; ++i) {
        sortedKeys[offset + i] = keys[i];
        sortedIndices[offset + i] = indices[i];
    }

    for (uint b = 0; b < 64; ++b) {
        bloomFilters[bloomOffset + b] = bloom[b];
    }

    sortedCounts[tileId] = validCount;
}

// =============================================================================
// KERNEL: TEMPORAL INSERTION SORT
// =============================================================================

kernel void temporal_insertion_sort(
    device uint* sortedKeys [[buffer(0)]],
    device uint* sortedIndices [[buffer(1)]],
    device uint* sortedCounts [[buffer(2)]],
    device uint* bloomFilters [[buffer(3)]],
    const device uint* candidateIndices [[buffer(4)]],
    const device uint* candidateCounts [[buffer(5)]],
    constant uint& maxPerTile [[buffer(6)]],
    constant uint& maxCandidates [[buffer(7)]],
    const device TemporalCompactedGaussian* compacted [[buffer(8)]],
    const device uint* worldToCompacted [[buffer(9)]],
    uint tileId [[thread_position_in_grid]]
) {
    uint sortedOffset = tileId * maxPerTile;
    uint candOffset = tileId * maxCandidates;
    uint bloomOffset = tileId * 64;
    uint count = sortedCounts[tileId];

    uint bloom[64];
    for (uint b = 0; b < 64; ++b) {
        bloom[b] = bloomFilters[bloomOffset + b];
    }

    uint worstKey = (count >= maxPerTile) ? sortedKeys[sortedOffset + count - 1] : 0xFFFFFFFFu;

    uint newKeys[32];
    uint newWorldIdx[32];
    uint numNew = 0;

    for (uint c = 0; c < maxCandidates && numNew < 32; ++c) {
        uint worldIdx = candidateIndices[candOffset + c];
        if (worldIdx == 0xFFFFFFFF) continue;

        uint h1 = worldIdx & 0x7FF;
        uint h2 = ((worldIdx * 2654435761u) >> 21) & 0x7FF;
        bool maybeExists = ((bloom[h1 >> 5] >> (h1 & 31)) & 1) &&
                          ((bloom[h2 >> 5] >> (h2 & 31)) & 1);
        if (maybeExists) {
            bool actuallyExists = false;
            for (uint i = 0; i < count && !actuallyExists; ++i) {
                if (sortedIndices[sortedOffset + i] == worldIdx) actuallyExists = true;
            }
            if (actuallyExists) continue;
        }

        uint compactedIdx = worldToCompacted[worldIdx];
        if (compactedIdx == 0xFFFFFFFF) continue;

        TemporalCompactedGaussian g = compacted[compactedIdx];
        uint depth16 = ((as_type<uint>(g.covariance_depth.w) ^ 0x80000000u) >> 16u) & 0xFFFFu;
        uint newKey = (depth16 << 16) | (worldIdx & 0xFFFFu);

        if (newKey >= worstKey && count >= maxPerTile) continue;

        newKeys[numNew] = newKey;
        newWorldIdx[numNew] = worldIdx;

        bloom[h1 >> 5] |= (1u << (h1 & 31));
        bloom[h2 >> 5] |= (1u << (h2 & 31));

        numNew++;
    }

    if (numNew == 0) {
        for (uint b = 0; b < 64; ++b) {
            bloomFilters[bloomOffset + b] = bloom[b];
        }
        return;
    }

    for (uint i = 1; i < numNew; ++i) {
        uint key = newKeys[i];
        uint idx = newWorldIdx[i];
        uint j = i;
        while (j > 0 && newKeys[j - 1] > key) {
            newKeys[j] = newKeys[j - 1];
            newWorldIdx[j] = newWorldIdx[j - 1];
            j--;
        }
        newKeys[j] = key;
        newWorldIdx[j] = idx;
    }

    uint finalCount = min(count + numNew, maxPerTile);
    int si = int(count) - 1;
    int ci = int(numNew) - 1;
    int wi = int(finalCount) - 1;

    while (wi >= 0 && (si >= 0 || ci >= 0)) {
        bool takeSorted = false;
        if (si >= 0 && ci >= 0) {
            takeSorted = (sortedKeys[sortedOffset + uint(si)] >= newKeys[uint(ci)]);
        } else {
            takeSorted = (si >= 0);
        }

        if (takeSorted) {
            sortedKeys[sortedOffset + uint(wi)] = sortedKeys[sortedOffset + uint(si)];
            sortedIndices[sortedOffset + uint(wi)] = sortedIndices[sortedOffset + uint(si)];
            si--;
        } else {
            sortedKeys[sortedOffset + uint(wi)] = newKeys[uint(ci)];
            sortedIndices[sortedOffset + uint(wi)] = newWorldIdx[uint(ci)];
            ci--;
        }
        wi--;
    }

    for (uint b = 0; b < 64; ++b) {
        bloomFilters[bloomOffset + b] = bloom[b];
    }

    sortedCounts[tileId] = finalCount;
}

// =============================================================================
// KERNEL: SPATIAL REUSE
// =============================================================================

kernel void temporal_spatial_reuse(
    device uint* sortedKeys [[buffer(0)]],
    device uint* sortedIndices [[buffer(1)]],
    device uint* sortedCounts [[buffer(2)]],
    const device TemporalCompactedGaussian* compacted [[buffer(3)]],
    const device uint* worldToCompacted [[buffer(4)]],
    constant uint& maxPerTile [[buffer(5)]],
    constant uint& tilesX [[buffer(6)]],
    constant uint& tilesY [[buffer(7)]],
    constant uint& numToSteal [[buffer(8)]],
    uint tileId [[thread_position_in_grid]]
) {
    uint tileY = tileId / tilesX;
    uint tileX = tileId % tilesX;

    uint sortedOffset = tileId * maxPerTile;
    uint count = sortedCounts[tileId];

    const int2 neighbors[8] = {
        int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1),
        int2(-1, -1), int2(1, -1), int2(-1, 1), int2(1, 1)
    };

    for (uint n = 0; n < 8; ++n) {
        int nx = int(tileX) + neighbors[n].x;
        int ny = int(tileY) + neighbors[n].y;

        if (nx < 0 || nx >= int(tilesX) || ny < 0 || ny >= int(tilesY)) continue;

        uint neighborId = uint(ny) * tilesX + uint(nx);
        uint neighborOffset = neighborId * maxPerTile;
        uint neighborCount = sortedCounts[neighborId];

        uint toSteal = min(numToSteal, neighborCount);
        for (uint s = 0; s < toSteal; ++s) {
            uint worldIdx = sortedIndices[neighborOffset + s];
            if (worldIdx == 0xFFFFFFFF) continue;

            uint compactedIdx = worldToCompacted[worldIdx];
            if (compactedIdx == 0xFFFFFFFF) continue;

            TemporalCompactedGaussian g = compacted[compactedIdx];
            if (int(tileX) < g.min_tile.x || int(tileX) >= g.max_tile.x ||
                int(tileY) < g.min_tile.y || int(tileY) >= g.max_tile.y) continue;

            uint newKey = temporal_compute_stable_depth_key(g, worldIdx);

            bool duplicate = false;
            for (uint i = 0; i < count && !duplicate; ++i) {
                if (sortedIndices[sortedOffset + i] == worldIdx) duplicate = true;
            }
            if (duplicate) continue;

            uint lo = 0, hi = count;
            while (lo < hi) {
                uint mid = (lo + hi) / 2;
                if (sortedKeys[sortedOffset + mid] < newKey) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }

            if (count < maxPerTile) {
                for (uint i = count; i > lo; --i) {
                    sortedKeys[sortedOffset + i] = sortedKeys[sortedOffset + i - 1];
                    sortedIndices[sortedOffset + i] = sortedIndices[sortedOffset + i - 1];
                }
                sortedKeys[sortedOffset + lo] = newKey;
                sortedIndices[sortedOffset + lo] = worldIdx;
                count++;
            } else if (lo < maxPerTile) {
                for (uint i = maxPerTile - 1; i > lo; --i) {
                    sortedKeys[sortedOffset + i] = sortedKeys[sortedOffset + i - 1];
                    sortedIndices[sortedOffset + i] = sortedIndices[sortedOffset + i - 1];
                }
                sortedKeys[sortedOffset + lo] = newKey;
                sortedIndices[sortedOffset + lo] = worldIdx;
            }
        }
    }

    sortedCounts[tileId] = count;
}

// =============================================================================
// KERNEL: TEMPORAL RENDER
// =============================================================================

kernel void temporal_render(
    const device TemporalCompactedGaussian* compacted [[buffer(0)]],
    const device uint* sortedCounts [[buffer(1)]],
    const device uint* sortedWorldIndices [[buffer(2)]],
    const device uint* worldToCompacted [[buffer(3)]],
    texture2d<half, access::write> colorOut [[texture(0)]],
    texture2d<half, access::write> depthOut [[texture(1)]],
    constant TemporalRenderParams& params [[buffer(4)]],
    constant uint& maxPerTile [[buffer(5)]],
    uint2 groupId [[threadgroup_position_in_grid]],
    uint2 localId [[thread_position_in_threadgroup]],
    uint localIdx [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    threadgroup uint tileGaussianCount = 0;
    threadgroup uint tileDataOffset = 0;

    if (localIdx == 0) {
        uint tileId = groupId.y * params.tilesX + groupId.x;
        tileGaussianCount = min(sortedCounts[tileId], maxPerTile);
        tileDataOffset = tileId * maxPerTile;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint gaussCount = tileGaussianCount;
    uint offset = tileDataOffset;

    half px0 = half(groupId.x * TEMPORAL_TILE_WIDTH + localId.x * TEMPORAL_PIXELS_X);
    half py0 = half(groupId.y * TEMPORAL_TILE_HEIGHT + localId.y * TEMPORAL_PIXELS_Y);

    half3 c00 = 0, c10 = 0, c20 = 0, c30 = 0;
    half3 c01 = 0, c11 = 0, c21 = 0, c31 = 0;
    half t00 = 1, t10 = 1, t20 = 1, t30 = 1;
    half t01 = 1, t11 = 1, t21 = 1, t31 = 1;

    for (uint i = 0; i < gaussCount; ++i) {
        uint worldIdx = sortedWorldIndices[offset + i];
        if (worldIdx == 0xFFFFFFFF) continue;

        uint compactedIdx = worldToCompacted[worldIdx];
        if (compactedIdx == 0xFFFFFFFF) continue;

        TemporalCompactedGaussian g = compacted[compactedIdx];

        half2 gPos = half2(g.position_color.xy);
        half3 cov = half3(g.covariance_depth.xyz);
        half4 colorOp = temporal_unpackHalf4(g.position_color.zw);

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

    half bg = params.whiteBackground ? 1.0h : 0.0h;
    c00 += t00 * bg; c10 += t10 * bg; c20 += t20 * bg; c30 += t30 * bg;
    c01 += t01 * bg; c11 += t11 * bg; c21 += t21 * bg; c31 += t31 * bg;

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

// =============================================================================
// KERNEL: STOCHASTIC RENDER
// =============================================================================

kernel void temporal_render_stochastic(
    const device TemporalCompactedGaussian* compacted [[buffer(0)]],
    const device uint* sortedCounts [[buffer(1)]],
    const device uint* sortedWorldIndices [[buffer(2)]],
    const device uint* worldToCompacted [[buffer(3)]],
    texture2d<half, access::write> colorOut [[texture(0)]],
    texture2d<half, access::write> depthOut [[texture(1)]],
    constant TemporalRenderParams& params [[buffer(4)]],
    constant uint& maxPerTile [[buffer(5)]],
    constant uint& frameIndex [[buffer(6)]],
    uint2 groupId [[threadgroup_position_in_grid]],
    uint2 localId [[thread_position_in_threadgroup]],
    uint localIdx [[thread_index_in_threadgroup]]
) {
    threadgroup uint tileGaussianCount = 0;
    threadgroup uint tileDataOffset = 0;

    if (localIdx == 0) {
        uint tileId = groupId.y * params.tilesX + groupId.x;
        tileGaussianCount = min(sortedCounts[tileId], maxPerTile);
        tileDataOffset = tileId * maxPerTile;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint gaussCount = tileGaussianCount;
    uint offset = tileDataOffset;

    uint px0 = groupId.x * STOCHASTIC_TILE_WIDTH + localId.x * STOCHASTIC_PIXELS_X;
    uint py0 = groupId.y * STOCHASTIC_TILE_HEIGHT + localId.y * STOCHASTIC_PIXELS_Y;

    half3 win00 = 0, win10 = 0;
    half3 win01 = 0, win11 = 0;
    float dep00 = 1e10f, dep10 = 1e10f;
    float dep01 = 1e10f, dep11 = 1e10f;

    uint seed00 = (py0 * params.width + px0) ^ (frameIndex * 1664525u);
    uint seed10 = (py0 * params.width + px0 + 1) ^ (frameIndex * 1664525u);
    uint seed01 = ((py0 + 1) * params.width + px0) ^ (frameIndex * 1664525u);
    uint seed11 = ((py0 + 1) * params.width + px0 + 1) ^ (frameIndex * 1664525u);

    for (uint i = 0; i < gaussCount; ++i) {
        uint worldIdx = sortedWorldIndices[offset + i];
        if (worldIdx == 0xFFFFFFFF) continue;

        uint compactedIdx = worldToCompacted[worldIdx];
        if (compactedIdx == 0xFFFFFFFF) continue;

        TemporalCompactedGaussian g = compacted[compactedIdx];

        half2 gPos = half2(g.position_color.xy);
        half3 cov = half3(g.covariance_depth.xyz);
        half4 colorOp = temporal_unpackHalf4(g.position_color.zw);
        float depth = g.covariance_depth.w;

        half cxx = cov.x, cyy = cov.z, cxy2 = 2.0h * cov.y;
        half3 gColor = colorOp.xyz;
        half opacity = colorOp.w;

        half hpx0 = half(px0), hpy0 = half(py0);
        half dx0 = hpx0 - gPos.x, dx1 = hpx0 + 1.0h - gPos.x;
        half dy0 = hpy0 - gPos.y, dy1 = hpy0 + 1.0h - gPos.y;

        half p00 = dx0*dx0*cxx + dy0*dy0*cyy + dx0*dy0*cxy2;
        half p10 = dx1*dx1*cxx + dy0*dy0*cyy + dx1*dy0*cxy2;
        half p01 = dx0*dx0*cxx + dy1*dy1*cyy + dx0*dy1*cxy2;
        half p11 = dx1*dx1*cxx + dy1*dy1*cyy + dx1*dy1*cxy2;

        float a00 = min(float(opacity) * exp(-0.5f * float(p00)), 0.99f);
        float a10 = min(float(opacity) * exp(-0.5f * float(p10)), 0.99f);
        float a01 = min(float(opacity) * exp(-0.5f * float(p01)), 0.99f);
        float a11 = min(float(opacity) * exp(-0.5f * float(p11)), 0.99f);

        float r00 = temporal_hash(seed00 + worldIdx * 1013904223u);
        float r10 = temporal_hash(seed10 + worldIdx * 1013904223u);
        float r01 = temporal_hash(seed01 + worldIdx * 1013904223u);
        float r11 = temporal_hash(seed11 + worldIdx * 1013904223u);

        if (r00 < a00 && depth < dep00) { win00 = gColor; dep00 = depth; }
        if (r10 < a10 && depth < dep10) { win10 = gColor; dep10 = depth; }
        if (r01 < a01 && depth < dep01) { win01 = gColor; dep01 = depth; }
        if (r11 < a11 && depth < dep11) { win11 = gColor; dep11 = depth; }
    }

    half bg = params.whiteBackground ? 1.0h : 0.0h;
    half3 bgColor = half3(bg);
    if (dep00 > 1e9f) win00 = bgColor;
    if (dep10 > 1e9f) win10 = bgColor;
    if (dep01 > 1e9f) win01 = bgColor;
    if (dep11 > 1e9f) win11 = bgColor;

    uint w = params.width, h = params.height;
    if (px0 < w && py0 < h)     { colorOut.write(half4(win00, 1.0h), uint2(px0, py0));     depthOut.write(half4(half(dep00), 0, 0, 0), uint2(px0, py0)); }
    if (px0+1 < w && py0 < h)   { colorOut.write(half4(win10, 1.0h), uint2(px0+1, py0));   depthOut.write(half4(half(dep10), 0, 0, 0), uint2(px0+1, py0)); }
    if (px0 < w && py0+1 < h)   { colorOut.write(half4(win01, 1.0h), uint2(px0, py0+1));   depthOut.write(half4(half(dep01), 0, 0, 0), uint2(px0, py0+1)); }
    if (px0+1 < w && py0+1 < h) { colorOut.write(half4(win11, 1.0h), uint2(px0+1, py0+1)); depthOut.write(half4(half(dep11), 0, 0, 0), uint2(px0+1, py0+1)); }
}

// =============================================================================
// KERNEL: RANDOM SAMPLE TILES
// =============================================================================

kernel void temporal_random_sample_tiles(
    const device TemporalCompactedGaussian* compacted [[buffer(0)]],
    const device uint* compactedHeader [[buffer(1)]],
    const device uint* compactedToWorld [[buffer(2)]],
    device uint* candidateKeys [[buffer(3)]],
    device uint* candidateIndices [[buffer(4)]],
    constant uint& samplesPerTile [[buffer(5)]],
    constant uint& frameIndex [[buffer(6)]],
    constant uint& tilesX [[buffer(7)]],
    constant uint& tileWidth [[buffer(8)]],
    constant uint& tileHeight [[buffer(9)]],
    uint tileId [[thread_position_in_grid]]
) {
    uint visibleCount = compactedHeader[0];
    uint offset = tileId * samplesPerTile;

    for (uint i = 0; i < samplesPerTile; ++i) {
        candidateKeys[offset + i] = 0xFFFFFFFFu;
        candidateIndices[offset + i] = 0xFFFFFFFFu;
    }

    if (visibleCount == 0) return;

    uint tileX = tileId % tilesX;
    uint tileY = tileId / tilesX;
    float tilePxMinX = float(tileX * tileWidth);
    float tilePxMinY = float(tileY * tileHeight);
    float tilePxMaxX = tilePxMinX + float(tileWidth);
    float tilePxMaxY = tilePxMinY + float(tileHeight);

    uint validCount = 0;
    for (uint s = 0; s < samplesPerTile && validCount < samplesPerTile; ++s) {
        uint seed = temporal_hash_uint(tileId);
        seed = temporal_hash_uint(seed + frameIndex);
        seed = temporal_hash_uint(seed + s);
        uint rndIdx = seed % visibleCount;

        TemporalCompactedGaussian g = compacted[rndIdx];

        float2 pos = g.position_color.xy;
        float3 cov = g.covariance_depth.xyz;

        float detConic = cov.x * cov.z - cov.y * cov.y;
        float radius = (detConic > 1e-12f) ? 3.0f / sqrt(sqrt(detConic)) : 100.0f;

        if (pos.x + radius >= tilePxMinX && pos.x - radius <= tilePxMaxX &&
            pos.y + radius >= tilePxMinY && pos.y - radius <= tilePxMaxY) {
            uint worldIdx = compactedToWorld[rndIdx];
            uint scoreKey = temporal_compute_score_key(g);

            candidateKeys[offset + validCount] = scoreKey;
            candidateIndices[offset + validCount] = worldIdx;
            validCount++;
        }
    }
}

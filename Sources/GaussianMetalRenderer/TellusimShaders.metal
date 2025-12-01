// Copyright (C) 2024 - Tellusim-style Gaussian Splatting Pipeline
// Clean, optimized implementation inspired by Tellusim's approach

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// =============================================================================
// SHARED CONSTANTS
// =============================================================================

// SH coefficients
constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2_0 = 1.0925484305920792f;
constant float SH_C2_1 = -1.0925484305920792f;
constant float SH_C2_2 = 0.31539156525252005f;
constant float SH_C2_3 = -1.0925484305920792f;
constant float SH_C2_4 = 0.5462742152960396f;
constant float SH_C3_0 = -0.5900435899266435f;
constant float SH_C3_1 = 2.890611442640554f;
constant float SH_C3_2 = -0.4570457994644658f;
constant float SH_C3_3 = 0.3731763325901154f;
constant float SH_C3_4 = -0.4570457994644658f;
constant float SH_C3_5 = 1.445305721320277f;
constant float SH_C3_6 = -0.5900435899266435f;

// =============================================================================
// FUNCTION CONSTANTS FOR SH DEGREE
// Compiler eliminates branches and unrolls loops at compile time
// =============================================================================
// SH degree: 0 = DC only (1 coeff), 1 = 4 coeffs, 2 = 9 coeffs, 3 = 16 coeffs
constant uint SH_DEGREE [[function_constant(0)]];
constant bool SH_DEGREE_0 = (SH_DEGREE == 0);
constant bool SH_DEGREE_1 = (SH_DEGREE == 1);
constant bool SH_DEGREE_2 = (SH_DEGREE == 2);
constant bool SH_DEGREE_3 = (SH_DEGREE == 3);

// =============================================================================
// OPTIMIZED SH COMPUTATION (compile-time branching, fully unrolled)
// =============================================================================
template<typename HarmonicsT>
inline float3 tellusim_computeSHColor(
    const device HarmonicsT* harmonics,
    uint gid,
    float3 pos,
    float3 cameraCenter,
    uint shComponents  // runtime fallback if function constant not set
) {
    // DC only - no direction needed
    if (SH_DEGREE_0 || shComponents == 0) {
        uint base = gid * 3u;
        return float3(harmonics[base], harmonics[base + 1], harmonics[base + 2]);
    }

    // Compute view direction
    float3 dir = normalize(cameraCenter - pos);
    float xx = dir.x * dir.x, yy = dir.y * dir.y, zz = dir.z * dir.z;
    float xy = dir.x * dir.y, yz = dir.y * dir.z, xz = dir.x * dir.z;

    // SH basis (compute only what we need based on degree)
    float shBasis[16];
    shBasis[0] = SH_C0;

    if (SH_DEGREE_1 || SH_DEGREE_2 || SH_DEGREE_3 || shComponents >= 4) {
        shBasis[1] = -SH_C1 * dir.y;
        shBasis[2] = SH_C1 * dir.z;
        shBasis[3] = -SH_C1 * dir.x;
    }

    if (SH_DEGREE_2 || SH_DEGREE_3 || shComponents >= 9) {
        shBasis[4] = SH_C2_0 * xy;
        shBasis[5] = SH_C2_1 * yz;
        shBasis[6] = SH_C2_2 * (2.0f * zz - xx - yy);
        shBasis[7] = SH_C2_3 * xz;
        shBasis[8] = SH_C2_4 * (xx - yy);
    }

    if (SH_DEGREE_3 || shComponents >= 16) {
        shBasis[9] = SH_C3_0 * dir.y * (3.0f * xx - yy);
        shBasis[10] = SH_C3_1 * xy * dir.z;
        shBasis[11] = SH_C3_2 * dir.y * (4.0f * zz - xx - yy);
        shBasis[12] = SH_C3_3 * dir.z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
        shBasis[13] = SH_C3_4 * dir.x * (4.0f * zz - xx - yy);
        shBasis[14] = SH_C3_5 * dir.z * (xx - yy);
        shBasis[15] = SH_C3_6 * dir.x * (xx - 3.0f * yy);
    }

    // Accumulate color - fully unrolled based on function constant
    float3 color = float3(0.0f);
    uint coeffs = SH_DEGREE_0 ? 1 : (SH_DEGREE_1 ? 4 : (SH_DEGREE_2 ? 9 : (SH_DEGREE_3 ? 16 : shComponents)));
    uint base = gid * coeffs * 3u;

    // Unrolled loops for each degree - compiler eliminates dead branches
    if (SH_DEGREE_1) {
        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
            color.x += float(harmonics[base + i]) * shBasis[i];
            color.y += float(harmonics[base + 4 + i]) * shBasis[i];
            color.z += float(harmonics[base + 8 + i]) * shBasis[i];
        }
    } else if (SH_DEGREE_2) {
        #pragma unroll
        for (uint i = 0; i < 9; ++i) {
            color.x += float(harmonics[base + i]) * shBasis[i];
            color.y += float(harmonics[base + 9 + i]) * shBasis[i];
            color.z += float(harmonics[base + 18 + i]) * shBasis[i];
        }
    } else if (SH_DEGREE_3) {
        #pragma unroll
        for (uint i = 0; i < 16; ++i) {
            color.x += float(harmonics[base + i]) * shBasis[i];
            color.y += float(harmonics[base + 16 + i]) * shBasis[i];
            color.z += float(harmonics[base + 32 + i]) * shBasis[i];
        }
    } else {
        // Runtime fallback (shouldn't happen with proper function constants)
        for (uint i = 0; i < min(coeffs, 16u); ++i) {
            color.x += float(harmonics[base + i]) * shBasis[i];
            color.y += float(harmonics[base + coeffs + i]) * shBasis[i];
            color.z += float(harmonics[base + coeffs * 2u + i]) * shBasis[i];
        }
    }

    return color;
}

// =============================================================================
// DATA STRUCTURES
// =============================================================================

/// Camera uniforms (matches Swift CameraUniformsSwift)
struct TellusimCameraUniforms {
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
struct TellusimPackedWorld {
    packed_float3 position;   // 12 bytes
    float         opacity;    // 4 bytes
    packed_float3 scale;      // 12 bytes
    float         _pad0;      // 4 bytes
    float4        rotation;   // 16 bytes
};

/// Packed world gaussian input (half16) - 24 bytes
struct TellusimPackedWorldHalf {
    packed_half3  position;   // 6 bytes
    half          opacity;    // 2 bytes
    packed_half3  scale;      // 6 bytes
    half          _pad0;      // 2 bytes
    half4         rotation;   // 8 bytes
};

/// Compacted gaussian (after projection + culling) - 48 bytes
/// Matches Tellusim's layout for optimal render performance
struct TellusimCompactedGaussian {
    float4 covariance_depth;    // conic.xyz + depth (16 bytes)
    float4 position_color;      // pos.xy + packed_half4(color,opacity) (16 bytes)
    int2 min_tile;              // 8 bytes
    int2 max_tile;              // 8 bytes
};

/// Parameters for fused project+compact+count kernel
struct TellusimProjectParams {
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
struct TellusimCompactedHeader {
    atomic_uint visibleCount;
    uint maxCompacted;
    uint overflow;
    uint _pad;
};

/// Render parameters
struct TellusimRenderParams {
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

inline float3x3 tellusim_matrixFromRows(float3 r0, float3 r1, float3 r2) {
    return float3x3(
        float3(r0.x, r1.x, r2.x),
        float3(r0.y, r1.y, r2.y),
        float3(r0.z, r1.z, r2.z)
    );
}

inline float4 tellusim_normalizeQuat(float4 q) {
    // Fast rsqrt-based normalization
    float d = dot(q, q);
    float r = rsqrt(max(d, 1e-8f));
    return q * r;
}

// Optimized: Fused quaternion to rotation + scale + covariance
inline void tellusim_buildCov3D_fast(float3 scale, float4 q,
                                      thread float& cov00, thread float& cov01, thread float& cov02,
                                      thread float& cov11, thread float& cov12, thread float& cov22) {
    // Quaternion to rotation matrix elements
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;

    // R = rotation matrix columns
    float3 R0 = float3(1 - (yy + zz), xy + wz, xz - wy);
    float3 R1 = float3(xy - wz, 1 - (xx + zz), yz + wx);
    float3 R2 = float3(xz + wy, yz - wx, 1 - (xx + yy));

    // M = S * R^T (scale applied to rows)
    float3 M0 = R0 * scale.x;
    float3 M1 = R1 * scale.y;
    float3 M2 = R2 * scale.z;

    // Covariance = M * M^T (only upper triangle needed)
    cov00 = dot(M0, M0);
    cov01 = dot(M0, M1);
    cov02 = dot(M0, M2);
    cov11 = dot(M1, M1);
    cov12 = dot(M1, M2);
    cov22 = dot(M2, M2);
}

// Optimized: Direct 2D covariance from 3D elements
inline float3 tellusim_projectCov_fast(float cov00, float cov01, float cov02,
                                        float cov11, float cov12, float cov22,
                                        float3 viewPos, float3x3 viewRot,
                                        float fx, float fy, float W, float H) {
    float z = viewPos.z;
    float z2 = z * z;
    // Match original: tanHalfFovX = W / (2 * fx), limit = 1.3 * tanHalfFovX
    float limX = 1.3f * W / (2.0f * fx);
    float limY = 1.3f * H / (2.0f * fy);

    float tx = clamp(viewPos.x / z, -limX, limX) * z;
    float ty = clamp(viewPos.y / z, -limY, limY) * z;

    // J * viewRot = T (only first two rows matter)
    float fxz = fx / z, fyz = fy / z;
    float fxtx = -fx * tx / z2, fyty = -fy * ty / z2;

    // T[0] = fxz * viewRot[0] + fxtx * viewRot[2]
    // T[1] = fyz * viewRot[1] + fyty * viewRot[2]
    float3 T0 = fxz * viewRot[0] + fxtx * viewRot[2];
    float3 T1 = fyz * viewRot[1] + fyty * viewRot[2];

    // cov2d = T * cov3d * T^T (only 2x2 result needed)
    // cov3d is symmetric, represented as upper triangle
    float3 cov3d_col0 = float3(cov00, cov01, cov02);
    float3 cov3d_col1 = float3(cov01, cov11, cov12);
    float3 cov3d_col2 = float3(cov02, cov12, cov22);

    // T * cov3d
    float3 TC0 = T0.x * cov3d_col0 + T0.y * cov3d_col1 + T0.z * cov3d_col2;
    float3 TC1 = T1.x * cov3d_col0 + T1.y * cov3d_col1 + T1.z * cov3d_col2;

    // (T * cov3d) * T^T
    float c00 = dot(TC0, T0) + 0.3f;
    float c01 = dot(TC0, T1);
    float c11 = dot(TC1, T1) + 0.3f;

    return float3(c00, c01, c11);
}

inline float3x3 tellusim_buildCov3D(float3 scale, float4 q) {
    // Quaternion to rotation matrix - must match original pipeline exactly
    // Convention: q = (x, y, z, w) where w is scalar/real part
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;

    // Column-major rotation matrix (matches original quaternionToMatrix + matrixFromRows)
    float3x3 R = float3x3(
        float3(1 - (yy + zz), xy + wz, xz - wy),  // col0
        float3(xy - wz, 1 - (xx + zz), yz + wx),  // col1
        float3(xz + wy, yz - wx, 1 - (xx + yy))   // col2
    );

    // RS = R * diag(scale) - scale each column
    float3x3 M = float3x3(
        R[0] * scale.x,
        R[1] * scale.y,
        R[2] * scale.z
    );

    // Cov = RS * RS^T
    return M * transpose(M);
}

inline float2x2 tellusim_projectCov(float3x3 cov3d, float3 viewPos, float3x3 viewRot,
                                     float fx, float fy, float W, float H) {
    float z = viewPos.z;
    // Match original: tanHalfFovX = W / (2 * fx), limit = 1.3 * tanHalfFovX
    float limX = 1.3f * W / (2.0f * fx);
    float limY = 1.3f * H / (2.0f * fy);

    float invZ = 1.0f / max(1e-4f, z);
    float invZ2 = invZ * invZ;

    float x = clamp(viewPos.x * invZ, -limX, limX) * z;
    float y = clamp(viewPos.y * invZ, -limY, limY) * z;

    // Jacobian columns (must match original exactly)
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
inline float2 tellusim_packHalf4(half4 v) {
    uint2 u = uint2(as_type<uint>(half2(v.xy)), as_type<uint>(half2(v.zw)));
    return float2(as_type<float>(u.x), as_type<float>(u.y));
}

// Unpack float2 to half4
inline half4 tellusim_unpackHalf4(float2 v) {
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
inline bool tellusim_segmentIntersectEllipse(float a, float b, float c, float d, float l, float r) {
    float delta = b * b - 4.0f * a * c;
    float t1 = (l - d) * (2.0f * a) + b;
    float t2 = (r - d) * (2.0f * a) + b;
    return delta >= 0.0f && (t1 <= 0.0f || t1 * t1 <= delta) && (t2 >= 0.0f || t2 * t2 <= delta);
}

// Block-ellipse intersection (FlashGS exact)
// Tests closest vertical and horizontal edges
inline bool tellusim_blockIntersectEllipse(int2 pix_min, int2 pix_max, float2 center, float3 conic, float power) {
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

    if (tellusim_segmentIntersectEllipse(a, b, c, center.y, float(pix_min.y), float(pix_max.y))) {
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

    if (tellusim_segmentIntersectEllipse(a, b, c, center.x, float(pix_min.x), float(pix_max.x))) {
        return true;
    }

    return false;
}

// Block contains center (FlashGS exact)
inline bool tellusim_blockContainsCenter(int2 pix_min, int2 pix_max, float2 center) {
    return center.x >= float(pix_min.x) && center.x <= float(pix_max.x) &&
           center.y >= float(pix_min.y) && center.y <= float(pix_max.y);
}

// Compute power from opacity (FlashGS exact)
// power = ln2 * 8 + ln2 * log2(opacity) = ln(256 * opacity)
constant float LN2 = 0.693147180559945f;
inline float tellusim_computePower(float opacity) {
    return LN2 * 8.0f + LN2 * log2(max(opacity, 1e-6f));
}

// =============================================================================
// KERNEL 1: CLEAR
// =============================================================================

kernel void tellusim_clear(
    device atomic_uint* tileCounts [[buffer(0)]],
    device TellusimCompactedHeader* header [[buffer(1)]],
    constant uint& tileCount [[buffer(2)]],
    constant uint& maxCompacted [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < tileCount) {
        atomic_store_explicit(&tileCounts[gid], 0u, memory_order_relaxed);
    }
    if (gid == 0u) {
        atomic_store_explicit(&header->visibleCount, 0u, memory_order_relaxed);
        header->maxCompacted = maxCompacted;
        header->overflow = 0u;
    }
}

// =============================================================================
// KERNEL 2: FUSED PROJECT + COMPACT + COUNT
// =============================================================================
// Templated on world gaussian type and harmonics type for maximum flexibility

template <typename PackedWorldT, typename HarmonicsT>
kernel void tellusim_project_compact_count(
    const device PackedWorldT* worldGaussians [[buffer(0)]],
    const device HarmonicsT* harmonics [[buffer(1)]],
    device TellusimCompactedGaussian* compacted [[buffer(2)]],
    device TellusimCompactedHeader* header [[buffer(3)]],
    device atomic_uint* tileCounts [[buffer(4)]],
    constant TellusimCameraUniforms& camera [[buffer(5)]],
    constant TellusimProjectParams& params [[buffer(6)]],
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

    // Covariance (scale already loaded in early cull)
    float4 quat = tellusim_normalizeQuat(float4(g.rotation));
    float3x3 cov3d = tellusim_buildCov3D(scale, quat);

    float3 vr0 = float3(camera.viewMatrix[0][0], camera.viewMatrix[1][0], camera.viewMatrix[2][0]);
    float3 vr1 = float3(camera.viewMatrix[0][1], camera.viewMatrix[1][1], camera.viewMatrix[2][1]);
    float3 vr2 = float3(camera.viewMatrix[0][2], camera.viewMatrix[1][2], camera.viewMatrix[2][2]);
    float3x3 viewRot = tellusim_matrixFromRows(vr0, vr1, vr2);

    float2x2 cov2d = tellusim_projectCov(cov3d, viewPos.xyz, viewRot,
                                          camera.focalX, camera.focalY,
                                          camera.width, camera.height);

    float a = cov2d[0][0], b = cov2d[0][1], c = cov2d[1][1];
    float det = a * c - b * b;
    if (det <= 1e-6f) return;

    float idet = 1.0f / det;
    float3 conic = float3(c * idet, -b * idet, a * idet);

    float mid = 0.5f * (a + c);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float radius = ceil(3.0f * sqrt(lambda1));

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
    float3 color = tellusim_computeSHColor(harmonics, gid, pos, camera.cameraCenter, camera.shComponents);
    color = max(color + 0.5f, float3(0.0f));

    // Compact: write visible gaussian
    uint idx = atomic_fetch_add_explicit(&header->visibleCount, 1u, memory_order_relaxed);
    if (idx >= params.maxCompacted) {
        header->overflow = 1u;
        return;
    }

    compacted[idx].covariance_depth = float4(conic, depth);
    // Use the computed color from harmonics (debug line removed)
    compacted[idx].position_color = float4(px, py, tellusim_packHalf4(half4(half3(color), half(opacity))));
    compacted[idx].min_tile = minTile;
    compacted[idx].max_tile = maxTile;
}

// Instantiate all combinations: World(float/half) x Harmonics(float/half)
// Float world, float harmonics (original)
template [[host_name("tellusim_project_compact_count_float")]]
kernel void tellusim_project_compact_count<TellusimPackedWorld, float>(
    const device TellusimPackedWorld*, const device float*,
    device TellusimCompactedGaussian*, device TellusimCompactedHeader*,
    device atomic_uint*, constant TellusimCameraUniforms&,
    constant TellusimProjectParams&, uint);

// Half world, float harmonics
template [[host_name("tellusim_project_compact_count_half")]]
kernel void tellusim_project_compact_count<TellusimPackedWorldHalf, float>(
    const device TellusimPackedWorldHalf*, const device float*,
    device TellusimCompactedGaussian*, device TellusimCompactedHeader*,
    device atomic_uint*, constant TellusimCameraUniforms&,
    constant TellusimProjectParams&, uint);

// Float world, half harmonics (memory bandwidth optimization)
template [[host_name("tellusim_project_compact_count_float_halfsh")]]
kernel void tellusim_project_compact_count<TellusimPackedWorld, half>(
    const device TellusimPackedWorld*, const device half*,
    device TellusimCompactedGaussian*, device TellusimCompactedHeader*,
    device atomic_uint*, constant TellusimCameraUniforms&,
    constant TellusimProjectParams&, uint);

// Half world, half harmonics (full half precision pipeline)
template [[host_name("tellusim_project_compact_count_half_halfsh")]]
kernel void tellusim_project_compact_count<TellusimPackedWorldHalf, half>(
    const device TellusimPackedWorldHalf*, const device half*,
    device TellusimCompactedGaussian*, device TellusimCompactedHeader*,
    device atomic_uint*, constant TellusimCameraUniforms&,
    constant TellusimProjectParams&, uint);

// =============================================================================
// KERNEL 3: PREFIX SCAN (reuse from main file or inline simple version)
// =============================================================================

#define TELLUSIM_PREFIX_BLOCK_SIZE 256
#define TELLUSIM_PREFIX_GRAIN_SIZE 4

kernel void tellusim_prefix_scan(
    const device uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    device uint* partialSums [[buffer(3)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    threadgroup uint shared[TELLUSIM_PREFIX_BLOCK_SIZE * TELLUSIM_PREFIX_GRAIN_SIZE];

    uint elementsPerGroup = TELLUSIM_PREFIX_BLOCK_SIZE * TELLUSIM_PREFIX_GRAIN_SIZE;
    uint globalOffset = groupId * elementsPerGroup;

    // Load
    for (ushort i = 0; i < TELLUSIM_PREFIX_GRAIN_SIZE; ++i) {
        uint idx = globalOffset + localId * TELLUSIM_PREFIX_GRAIN_SIZE + i;
        shared[localId * TELLUSIM_PREFIX_GRAIN_SIZE + i] = (idx < count) ? input[idx] : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Local prefix sum
    uint localSum = 0;
    for (ushort i = 0; i < TELLUSIM_PREFIX_GRAIN_SIZE; ++i) {
        uint val = shared[localId * TELLUSIM_PREFIX_GRAIN_SIZE + i];
        shared[localId * TELLUSIM_PREFIX_GRAIN_SIZE + i] = localSum;
        localSum += val;
    }

    // Threadgroup scan (simple serial for now)
    threadgroup uint blockSums[TELLUSIM_PREFIX_BLOCK_SIZE];
    blockSums[localId] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (localId == 0) {
        uint sum = 0;
        for (ushort i = 0; i < TELLUSIM_PREFIX_BLOCK_SIZE; ++i) {
            uint val = blockSums[i];
            blockSums[i] = sum;
            sum += val;
        }
        partialSums[groupId] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Add block sum to local values
    uint blockSum = blockSums[localId];
    for (ushort i = 0; i < TELLUSIM_PREFIX_GRAIN_SIZE; ++i) {
        uint idx = globalOffset + localId * TELLUSIM_PREFIX_GRAIN_SIZE + i;
        if (idx < count) {
            output[idx] = shared[localId * TELLUSIM_PREFIX_GRAIN_SIZE + i] + blockSum;
        }
    }
}

kernel void tellusim_scan_partial_sums(
    device uint* partialSums [[buffer(0)]],
    constant uint& numPartials [[buffer(1)]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    threadgroup uint shared[TELLUSIM_PREFIX_BLOCK_SIZE];
    shared[localId] = (localId < numPartials) ? partialSums[localId] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (localId == 0) {
        uint sum = 0;
        for (ushort i = 0; i < TELLUSIM_PREFIX_BLOCK_SIZE; ++i) {
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

kernel void tellusim_finalize_scan(
    device uint* output [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    const device uint* partialSums [[buffer(2)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    if (groupId == 0) return;  // First group doesn't need adjustment

    uint elementsPerGroup = TELLUSIM_PREFIX_BLOCK_SIZE * TELLUSIM_PREFIX_GRAIN_SIZE;
    uint globalOffset = groupId * elementsPerGroup;
    uint blockSum = partialSums[groupId];

    for (ushort i = 0; i < TELLUSIM_PREFIX_GRAIN_SIZE; ++i) {
        uint idx = globalOffset + localId * TELLUSIM_PREFIX_GRAIN_SIZE + i;
        if (idx < count) {
            output[idx] += blockSum;
        }
    }
}

// Fused finalize scan + zero counters (one dispatch instead of two)
kernel void tellusim_finalize_scan_and_zero(
    device uint* output [[buffer(0)]],
    device atomic_uint* counters [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    const device uint* partialSums [[buffer(3)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    uint elementsPerGroup = TELLUSIM_PREFIX_BLOCK_SIZE * TELLUSIM_PREFIX_GRAIN_SIZE;
    uint globalOffset = groupId * elementsPerGroup;
    uint blockSum = (groupId > 0) ? partialSums[groupId] : 0u;

    for (ushort i = 0; i < TELLUSIM_PREFIX_GRAIN_SIZE; ++i) {
        uint idx = globalOffset + localId * TELLUSIM_PREFIX_GRAIN_SIZE + i;
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

kernel void tellusim_prepare_scatter_dispatch(
    const device TellusimCompactedHeader* header [[buffer(0)]],
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

kernel void tellusim_compute_gaussian_tile_counts(
    const device TellusimCompactedGaussian* compacted [[buffer(0)]],
    const device TellusimCompactedHeader* header [[buffer(1)]],
    device uint* gaussianTileCounts [[buffer(2)]],  // Output: tiles per gaussian
    uint gid [[thread_position_in_grid]]
) {
    uint visibleCount = atomic_load_explicit(&header->visibleCount, memory_order_relaxed);
    if (gid >= visibleCount) {
        gaussianTileCounts[gid] = 0;
        return;
    }

    TellusimCompactedGaussian g = compacted[gid];
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

kernel void tellusim_scatter_simd(
    const device TellusimCompactedGaussian* compacted [[buffer(0)]],
    const device TellusimCompactedHeader* header [[buffer(1)]],
    device atomic_uint* tileCounters [[buffer(2)]],
    const device uint* offsets [[buffer(3)]],
    device uint* sortKeys [[buffer(4)]],
    device uint* sortIndices [[buffer(5)]],
    constant uint& tilesX [[buffer(6)]],
    constant uint& maxAssignments [[buffer(7)]],
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
    TellusimCompactedGaussian g;
    float3 conic;
    float2 center;
    float power;
    uint depthKey;
    int minTX, minTY, maxTX, maxTY;

    if (simd_lane == 0) {
        g = compacted[gaussianIdx];
        conic = g.covariance_depth.xyz;
        center = g.position_color.xy;
        half4 colorOp = tellusim_unpackHalf4(g.position_color.zw);
        power = tellusim_computePower(float(colorOp.w));
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
        bool valid = tellusim_blockContainsCenter(pix_min, pix_max, center) ||
                     tellusim_blockIntersectEllipse(pix_min, pix_max, center, conic, power);

        // SIMD early-out: skip iteration if no lanes are valid
        simd_vote ballot = simd_ballot(valid);
        if (simd_vote::vote_t(ballot) == 0) continue;

        // Valid lanes write to their tiles (parallel atomics across SIMD)
        if (valid) {
            uint tileId = uint(ty) * tilesX + uint(tx);
            uint localIdx = atomic_fetch_add_explicit(&tileCounters[tileId], 1u, memory_order_relaxed);
            uint writePos = offsets[tileId] + localIdx;

            if (writePos < maxAssignments) {
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
// Limit 2048 elements to fit in 32KB threadgroup memory

#define SORT_MAX_SIZE 2048

kernel void tellusim_per_tile_sort(
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
// KERNEL 6: RENDER (Tellusim-style, 8 pixels per thread)
// =============================================================================
// Optimized: No arrays (prevents spilling), half precision, direct memory access

#define TELLUSIM_TILE_WIDTH 32
#define TELLUSIM_TILE_HEIGHT 16
#define TELLUSIM_TG_WIDTH 8
#define TELLUSIM_TG_HEIGHT 8
#define TELLUSIM_PIXELS_X 4
#define TELLUSIM_PIXELS_Y 2

kernel void tellusim_render(
    const device TellusimCompactedGaussian* compacted [[buffer(0)]],
    const device uint* offsets [[buffer(1)]],
    const device uint* counts [[buffer(2)]],
    const device uint* sortedIndices [[buffer(3)]],
    texture2d<half, access::write> colorOut [[texture(0)]],
    texture2d<half, access::write> depthOut [[texture(1)]],
    constant TellusimRenderParams& params [[buffer(4)]],
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
    half px0 = half(groupId.x * TELLUSIM_TILE_WIDTH + localId.x * TELLUSIM_PIXELS_X);
    half py0 = half(groupId.y * TELLUSIM_TILE_HEIGHT + localId.y * TELLUSIM_PIXELS_Y);

    // Explicit accumulators - no arrays to prevent register spilling
    half3 c00 = 0, c10 = 0, c20 = 0, c30 = 0;
    half3 c01 = 0, c11 = 0, c21 = 0, c31 = 0;
    half t00 = 1, t10 = 1, t20 = 1, t30 = 1;
    half t01 = 1, t11 = 1, t21 = 1, t31 = 1;

    // Process all gaussians
    for (uint i = 0; i < gaussCount; ++i) {
        uint gIdx = sortedIndices[offset + i];
        TellusimCompactedGaussian g = compacted[gIdx];

        half2 gPos = half2(g.position_color.xy);
        half3 cov = half3(g.covariance_depth.xyz);
        half4 colorOp = tellusim_unpackHalf4(g.position_color.zw);

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


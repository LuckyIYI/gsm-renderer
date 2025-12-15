// GaussianShared.h - Shared functions and constants for Gaussian splatting
// Used by both GlobalShaders.metal and LocalShaders.metal

#ifndef GAUSSIAN_SHARED_H
#define GAUSSIAN_SHARED_H

#include <metal_stdlib>
#include "BridgingTypes.h"
using namespace metal;



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

// Function constants for SH degree (compile-time branching)
constant uint SH_DEGREE [[function_constant(0)]];
constant bool SH_DEGREE_0 = (SH_DEGREE == 0);
constant bool SH_DEGREE_1 = (SH_DEGREE == 1);
constant bool SH_DEGREE_2 = (SH_DEGREE == 2);
constant bool SH_DEGREE_3 = (SH_DEGREE == 3);


template<typename HarmonicsT>
inline float3 computeSHColor(
    const device HarmonicsT* harmonics,
    uint gid,
    float3 pos,
    float3 cameraCenter,
    uint shComponents
) {
    if (SH_DEGREE_0 || shComponents == 0) {
        uint base = gid * 3u;
        return float3(harmonics[base], harmonics[base + 1], harmonics[base + 2]) * SH_C0;
    }

    float3 dir = normalize(cameraCenter - pos);
    float xx = dir.x * dir.x, yy = dir.y * dir.y, zz = dir.z * dir.z;
    float xy = dir.x * dir.y, yz = dir.y * dir.z, xz = dir.x * dir.z;

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

    float3 color = float3(0.0f);
    uint coeffs = SH_DEGREE_0 ? 1 : (SH_DEGREE_1 ? 4 : (SH_DEGREE_2 ? 9 : (SH_DEGREE_3 ? 16 : shComponents)));
    uint base = gid * coeffs * 3u;

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
        for (uint i = 0; i < min(coeffs, 16u); ++i) {
            color.x += float(harmonics[base + i]) * shBasis[i];
            color.y += float(harmonics[base + coeffs + i]) * shBasis[i];
            color.z += float(harmonics[base + coeffs * 2u + i]) * shBasis[i];
        }
    }

    return color;
}

// ============================================================================
// Color Space Helpers
// ============================================================================

inline float srgbToLinearChannel(float c) {
    c = clamp(c, 0.0f, 1.0f);
    return (c <= 0.04045f) ? (c / 12.92f) : fast::powr((c + 0.055f) / 1.055f, 2.4f);
}

inline float3 srgbToLinear(float3 c) {
    return float3(
        srgbToLinearChannel(c.x),
        srgbToLinearChannel(c.y),
        srgbToLinearChannel(c.z)
    );
}

inline float3 maybeDecodeSRGBToLinear(float3 c, float inputIsSRGB) {
    return (inputIsSRGB > 0.5f) ? srgbToLinear(c) : c;
}

inline float4 getRotation(PackedWorldGaussian g) { return g.rotation; }

inline float4 getRotation(PackedWorldGaussianHalf g) {
    return float4(half(g.rx), half(g.ry), half(g.rz), half(g.rw));
}

inline float3x3 matrixFromRows(float3 r0, float3 r1, float3 r2) {
    return float3x3(
        float3(r0.x, r1.x, r2.x),
        float3(r0.y, r1.y, r2.y),
        float3(r0.z, r1.z, r2.z)
    );
}

// NDC to screen coordinates (float precision)
inline float2 ndcToScreen(float2 ndc, float width, float height) {
    return float2(
        (ndc.x + 1.0f) * 0.5f * width,
        (ndc.y + 1.0f) * 0.5f * height
    );
}

// Screen to NDC coordinates (float precision)
inline float2 screenToNDC(float2 screen, float width, float height) {
    return float2(
        (screen.x / width) * 2.0f - 1.0f,
        (screen.y / height) * 2.0f - 1.0f
    );
}

// NDC to screen coordinates (half precision input, float output)
inline float2 ndcToScreen(half2 ndc, float width, float height) {
    return float2(
        (float(ndc.x) + 1.0f) * 0.5f * width,
        (float(ndc.y) + 1.0f) * 0.5f * height
    );
}

// Screen to NDC coordinates (half precision input, float output)
inline float2 screenToNDC(half2 screen, float width, float height) {
    return float2(
        (float(screen.x) / width) * 2.0f - 1.0f,
        (float(screen.y) / height) * 2.0f - 1.0f
    );
}

// NDC to screen coordinates with pixel centering (float precision)
// Maps NDC [-1,+1] to pixel centers [-0.5, size-0.5]
// Use this for tiled renderers that need pixel-centered coordinates
inline float2 ndcToScreenCentered(float2 ndc, float width, float height) {
    return float2(
        ((ndc.x + 1.0f) * width - 1.0f) * 0.5f,
        ((ndc.y + 1.0f) * height - 1.0f) * 0.5f
    );
}

// ============================================================================
// Unified Projection Helpers - Z-sign agnostic
// Works with both OpenCV (+Z forward) and OpenGL (-Z forward) conventions
// ============================================================================

// Result of projecting a point to screen space
struct ProjectedPoint {
    float2 screen;      // Screen position in pixels
    float depth;        // Positive depth (for sorting/culling)
    bool visible;       // True if in front of camera and past near plane
};

// Project a world position to screen space (Z-sign agnostic)
// This is THE canonical projection function - use it everywhere for consistency
inline ProjectedPoint projectToScreen(
    float3 worldPos,
    float4x4 viewMatrix,
    float4x4 projMatrix,
    float nearPlane,
    float width,
    float height
) {
    ProjectedPoint result;
    result.visible = false;
    result.depth = 0.0f;
    result.screen = float2(0.0f);

    // Transform to view space then clip space
    float4 viewPos = viewMatrix * float4(worldPos, 1.0f);
    float4 clip = projMatrix * viewPos;

    // Z-sign agnostic culling: clip.w > 0 means in front for BOTH conventions
    // clip.w also serves as depth (positive for visible points)
    result.depth = clip.w;

    if (clip.w <= nearPlane) {
        return result;  // Behind camera or too close
    }

    // Perspective divide to NDC
    float2 ndc = clip.xy / clip.w;

    // NDC to screen using shared helper (convention-agnostic)
    result.screen = ndcToScreen(ndc, width, height);

    result.visible = true;
    return result;
}

// Overload that takes float4 viewPos (when view transform already done)
inline ProjectedPoint projectToScreenFromView(
    float4 viewPos,
    float4x4 projMatrix,
    float nearPlane,
    float width,
    float height
) {
    ProjectedPoint result;
    result.visible = false;
    result.depth = 0.0f;
    result.screen = float2(0.0f);

    float4 clip = projMatrix * viewPos;

    result.depth = clip.w;

    if (clip.w <= nearPlane) {
        return result;
    }

    float2 ndc = clip.xy / clip.w;

    // NDC to screen using shared helper (convention-agnostic)
    result.screen = ndcToScreen(ndc, width, height);

    result.visible = true;
    return result;
}

// Check if point is in front of camera using clip.w
// This is Z-sign agnostic: projection matrix encodes the convention
// clip.w > 0 means in front for BOTH OpenCV and OpenGL
inline bool isInFrontOfCameraClipW(float clipW, float nearPlane) {
    return clipW > nearPlane;
}

// Compute depth factor for LOD/culling (0 at far, 1 at near)
// depth should be positive (from clip.w or ProjectedPoint.depth)
inline float computeDepthFactor(float depth, float nearPlane, float farPlane) {
    float adjustedFarPlane = farPlane * 0.02;
    return 1.0f - pow(saturate((adjustedFarPlane - depth) / (adjustedFarPlane - nearPlane)), 2.0f);
}

// DEPRECATED: Use extractDepth() instead
inline float safeDepthComponent(float value) {
    float absVal = fabs(value);
    if (absVal < 1e-4f) {
        return value >= 0 ? 1e-4f : -1e-4f;
    }
    return value;
}

inline float4 normalizeQuaternion(float4 quat) {
    float norm = sqrt(max(dot(quat, quat), 1e-8f));
    if (norm < 1e-8f) {
        return float4(1.0f, 0.0f, 0.0f, 0.0f);
    }
    return quat / norm;
}

inline float3x3 quaternionToMatrix(float4 quat) {
    float x = quat.x, y = quat.y, z = quat.z, r = quat.w;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float3 row0 = float3(1.0f - 2.0f * (yy + zz), 2.0f * (xy - r * z), 2.0f * (xz + r * y));
    float3 row1 = float3(2.0f * (xy + r * z), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - r * x));
    float3 row2 = float3(2.0f * (xz - r * y), 2.0f * (yz + r * x), 1.0f - 2.0f * (xx + yy));
    return matrixFromRows(row0, row1, row2);
}

inline float3x3 buildCovariance3D(float3 scale, float4 quat) {
    float4 q = normalizeQuaternion(quat);
    float3x3 R = quaternionToMatrix(q);
    float3 RS0 = R[0] * scale.x;
    float3 RS1 = R[1] * scale.y;
    float3 RS2 = R[2] * scale.z;
    return float3x3(
        float3(RS0.x*RS0.x + RS1.x*RS1.x + RS2.x*RS2.x,
               RS0.x*RS0.y + RS1.x*RS1.y + RS2.x*RS2.y,
               RS0.x*RS0.z + RS1.x*RS1.z + RS2.x*RS2.z),
        float3(RS0.y*RS0.x + RS1.y*RS1.x + RS2.y*RS2.x,
               RS0.y*RS0.y + RS1.y*RS1.y + RS2.y*RS2.y,
               RS0.y*RS0.z + RS1.y*RS1.z + RS2.y*RS2.z),
        float3(RS0.z*RS0.x + RS1.z*RS1.x + RS2.z*RS2.x,
               RS0.z*RS0.y + RS1.z*RS1.y + RS2.z*RS2.y,
               RS0.z*RS0.z + RS1.z*RS1.z + RS2.z*RS2.z)
    );
}

// ============================================================================
// Unified Covariance Projection - THE canonical function
// Z-sign agnostic: works with both OpenCV (+Z forward) and OpenGL (-Z forward)
// Derives focal length from projection matrix for correctness
// ============================================================================

inline float2x2 projectCovariance2D(
    float3x3 cov3d,
    float3 viewPos,
    float3x3 viewRotation,
    float4x4 projMatrix,
    float2 screenSize
) {
    // Z-sign agnostic projection (works for OpenCV +Z and OpenGL -Z conventions)
    // Proven mathematically: 2D covariance is identical for both conventions
    float absZ = abs(viewPos.z);
    float signZ = (viewPos.z >= 0.0f) ? 1.0f : -1.0f;
    float safeAbsZ = max(absZ, 1e-4f);
    float invAbsZ = 1.0f / safeAbsZ;
    float invAbsZ2 = invAbsZ * invAbsZ;

    // Derive frustum limits from projection matrix
    float tanHalfFovX = 1.0f / max(abs(projMatrix[0][0]), 1e-4f);
    float tanHalfFovY = 1.0f / max(abs(projMatrix[1][1]), 1e-4f);
    float limX = 1.3f * tanHalfFovX;
    float limY = 1.3f * tanHalfFovY;

    // Compute tangent values (position / |z|) and clamp to frustum
    float tx = viewPos.x * invAbsZ;
    float ty = viewPos.y * invAbsZ;
    float xClamped = clamp(tx, -limX, limX) * safeAbsZ;
    float yClamped = clamp(ty, -limY, limY) * safeAbsZ;

    // Derive focal length from projection matrix: focal = screenSize * proj[i][i] / 2
    float focalX = screenSize.x * abs(projMatrix[0][0]) * 0.5f;
    float focalY = screenSize.y * abs(projMatrix[1][1]) * 0.5f;

    // Z-sign agnostic Jacobian:
    // J[0][2] and J[1][2] include signZ for correct derivative of 1/|z|
    // d(1/|z|)/dz = -sign(z)/|z|^2
    float3x3 J = float3x3(
        float3(focalX * invAbsZ, 0.0f, 0.0f),
        float3(0.0f, focalY * invAbsZ, 0.0f),
        float3(-focalX * xClamped * signZ * invAbsZ2, -focalY * yClamped * signZ * invAbsZ2, 0.0f)
    );

    // Transform covariance: cov2d = J * W * cov3d * W^T * J^T
    float3x3 T = J * viewRotation;
    float3x3 covFull = T * cov3d * transpose(T);

    // Low-pass filter: every Gaussian should be at least one pixel wide/high
    float2x2 cov2d = float2x2(covFull[0][0], covFull[0][1], covFull[1][0], covFull[1][1]);
    cov2d[0][0] += 0.3f;
    cov2d[1][1] += 0.3f;
    return cov2d;
}

// Convenience overload: extract view rotation from view matrix
inline float2x2 projectCovariance2D(
    float3x3 cov3d,
    float3 viewPos,
    float4x4 viewMatrix,
    float4x4 projMatrix,
    float2 screenSize
) {
    // Extract rotation (upper-left 3x3, stored as columns)
    float3x3 viewRotation = float3x3(viewMatrix[0].xyz, viewMatrix[1].xyz, viewMatrix[2].xyz);
    return projectCovariance2D(cov3d, viewPos, viewRotation, projMatrix, screenSize);
}

// ============================================================================
// DEPRECATED: Legacy function signatures for backward compatibility
// These all delegate to projectCovariance2D
// ============================================================================

// DEPRECATED: Use projectCovariance2D instead
inline float2x2 projectCovarianceFromMatrices(
    float3x3 cov3d,
    float3 viewPos,
    float4x4 viewMatrix,
    float4x4 projMatrix,
    uint2 screenSize
) {
    return projectCovariance2D(cov3d, viewPos, viewMatrix, projMatrix, float2(screenSize));
}

// DEPRECATED: Use projectCovariance2D instead
// This version requires caller to build a synthetic projection matrix or use the new API
// Z-sign agnostic: works with both OpenCV (+Z) and OpenGL (-Z) conventions
inline float2x2 projectCovariance(
    float3x3 cov3d,
    float3 viewPos,
    float3x3 viewRotation,
    float focalX,
    float focalY,
    float width,
    float height
) {
    float tanHalfFovX = width / max(2.0f * max(focalX, 1e-4f), 1e-4f);
    float tanHalfFovY = height / max(2.0f * max(focalY, 1e-4f), 1e-4f);
    float limX = 1.3f * tanHalfFovX;
    float limY = 1.3f * tanHalfFovY;

    // Z-sign agnostic projection (works for OpenCV +Z and OpenGL -Z conventions)
    float absZ = abs(viewPos.z);
    float signZ = (viewPos.z >= 0.0f) ? 1.0f : -1.0f;
    float safeAbsZ = max(absZ, 1e-4f);
    float invAbsZ = 1.0f / safeAbsZ;
    float invAbsZ2 = invAbsZ * invAbsZ;

    // Compute tangent values (position / |z|) and clamp to frustum
    float tx = viewPos.x * invAbsZ;
    float ty = viewPos.y * invAbsZ;
    float xClamped = clamp(tx, -limX, limX) * safeAbsZ;
    float yClamped = clamp(ty, -limY, limY) * safeAbsZ;

    // Z-sign agnostic Jacobian:
    // J[0][2] and J[1][2] include signZ for correct derivative of 1/|z|
    float3 col0 = float3(focalX * invAbsZ, 0.0f, 0.0f);
    float3 col1 = float3(0.0f, focalY * invAbsZ, 0.0f);
    float3 col2 = float3(-focalX * xClamped * signZ * invAbsZ2, -focalY * yClamped * signZ * invAbsZ2, 0.0f);
    float3x3 J = float3x3(col0, col1, col2);

    float3x3 T = J * viewRotation;
    float3x3 covFull = T * cov3d * transpose(T);

    float2x2 cov2d = float2x2(covFull[0][0], covFull[0][1], covFull[1][0], covFull[1][1]);
    cov2d[0][0] += 0.3f;
    cov2d[1][1] += 0.3f;
    return cov2d;
}

// DEPRECATED: Use projectCovariance2D instead
inline float2x2 projectCovarianceWithProjMatrix(
    float3x3 cov3d,
    float3 viewPos,
    float3x3 viewRotation,
    float4x4 projMatrix,
    float width,
    float height
) {
    return projectCovariance2D(cov3d, viewPos, viewRotation, projMatrix, float2(width, height));
}

inline void computeConicAndRadius(float2x2 cov, thread float4& conic, thread float& radius) {
    float a = cov[0][0], b = cov[0][1], c = cov[1][0], d = cov[1][1];
    float det = a * d - b * c;
    float invDet = 1.0f / max(det, 1e-8f);
    conic = float4(d * invDet, -b * invDet, a * invDet, 0.0f);
    float mid = 0.5f * (a + d);
    float delta = max(mid * mid - det, 1e-5f);
    float sqrtDelta = sqrt(delta);
    float maxEig = mid + sqrtDelta;
    radius = 3.0f * ceil(sqrt(max(maxEig, 1e-5f)));
}

inline float4 invertCovariance2D(float2x2 cov) {
    float det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    float invDet = 1.0f / max(det, 1e-8f);
    return float4(cov[1][1] * invDet, -cov[0][1] * invDet, cov[0][0] * invDet, 0.0f);
}

inline float radiusFromCovariance(float2x2 cov) {
    float det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    float mid = 0.5f * (cov[0][0] + cov[1][1]);
    float delta = max(mid * mid - det, 1e-5f);
    return 3.0f * ceil(sqrt(max(mid + sqrt(delta), 1e-5f)));
}

inline float2 computeOBBExtents(float2x2 cov, float sigma_multiplier) {
    float a = cov[0][0], b = cov[0][1], d = cov[1][1];
    float det = a * d - b * b;
    float mid = 0.5f * (a + d);
    float disc = max(mid * mid - det, 1e-6f);
    float sqrtDisc = sqrt(disc);

    float lambda1 = mid + sqrtDisc;
    float lambda2 = max(mid - sqrtDisc, 1e-6f);

    float e1 = sigma_multiplier * sqrt(max(lambda1, 1e-6f));
    float e2 = sigma_multiplier * sqrt(max(lambda2, 1e-6f));

    float2 v1;
    if (abs(b) > 1e-6f) {
        float vx = b, vy = lambda1 - a;
        float vlen = sqrt(vx * vx + vy * vy);
        v1 = float2(vx, vy) / max(vlen, 1e-6f);
    } else {
        v1 = (a >= d) ? float2(1.0f, 0.0f) : float2(0.0f, 1.0f);
    }

    float xExtent = abs(v1.x) * e1 + abs(v1.y) * e2;
    float yExtent = abs(v1.y) * e1 + abs(v1.x) * e2;
    return float2(xExtent, yExtent);
}

inline float2 computeOBBExtentsFromComponents(float a, float b, float d, float sigma_multiplier) {
    float det = a * d - b * b;
    float mid = 0.5f * (a + d);
    float disc = max(mid * mid - det, 1e-6f);
    float sqrtDisc = sqrt(disc);

    float lambda1 = mid + sqrtDisc;
    float lambda2 = max(mid - sqrtDisc, 1e-6f);

    float e1 = sigma_multiplier * sqrt(max(lambda1, 1e-6f));
    float e2 = sigma_multiplier * sqrt(max(lambda2, 1e-6f));

    float2 v1;
    if (abs(b) > 1e-6f) {
        float vx = b, vy = lambda1 - a;
        float vlen = sqrt(vx * vx + vy * vy);
        v1 = float2(vx, vy) / max(vlen, 1e-6f);
    } else {
        v1 = (a >= d) ? float2(1.0f, 0.0f) : float2(0.0f, 1.0f);
    }

    float xExtent = abs(v1.x) * e1 + abs(v1.y) * e2;
    float yExtent = abs(v1.y) * e1 + abs(v1.x) * e2;
    return float2(xExtent, yExtent);
}

inline float2 computeOBBExtentsFromConic(float3 conic, float sigma_multiplier) {
    float invA = conic.x, invB = conic.y, invD = conic.z;
    float conicDet = invA * invD - invB * invB;
    if (abs(conicDet) < 1e-8f) {
        return float2(sigma_multiplier * 10.0f, sigma_multiplier * 10.0f);
    }
    float a = invD / conicDet;
    float b = -invB / conicDet;
    float d = invA / conicDet;
    return computeOBBExtentsFromComponents(a, b, d, sigma_multiplier);
}

constant float GAUSSIAN_TAU = 1.0f / 255.0f;
constant float GAUSSIAN_LN2 = 0.693147180559945f;

// ============================================================================
// Ellipse packing helpers (theta + sigma1/sigma2) for 16-byte GaussianRenderData
// ============================================================================

constant float kPiF = 3.14159265358979323846f;

inline UINT16 packThetaPi(float theta) {
    // Store theta in [0, pi) because ellipse orientation is pi-periodic.
    theta = fmod(theta, kPiF);
    if (theta < 0.0f) theta += kPiF;
    float u = theta * (65535.0f / kPiF);
    return UINT16(clamp(u + 0.5f, 0.0f, 65535.0f));
}

inline float unpackThetaPi(UINT16 packed) {
    return (float(packed) * (kPiF / 65535.0f));
}

inline bool covarianceToThetaSigmas(
    float2x2 cov,
    thread float& theta,
    thread float& sigma1,
    thread float& sigma2
) {
    // Eigen decomposition for symmetric 2x2 covariance.
    float a = cov[0][0];
    float b = 0.5f * (cov[0][1] + cov[1][0]);
    float d = cov[1][1];
    if (!isfinite(a) || !isfinite(b) || !isfinite(d)) return false;

    // Ensure positive (covariance should already be SPD after filtering).
    a = max(a, 1e-8f);
    d = max(d, 1e-8f);

    float det = a * d - b * b;
    if (!isfinite(det) || det <= 0.0f) return false;

    float mid = 0.5f * (a + d);
    float disc = max(mid * mid - det, 0.0f);
    float sqrtDisc = sqrt(disc);
    float lambda1 = max(mid + sqrtDisc, 1e-8f);
    float lambda2 = max(mid - sqrtDisc, 1e-8f);

    // Principal eigenvector for lambda1.
    float2 v1;
    if (abs(b) > 1e-8f) {
        v1 = normalize(float2(b, lambda1 - a));
    } else {
        v1 = (a >= d) ? float2(1.0f, 0.0f) : float2(0.0f, 1.0f);
    }

    theta = atan2(v1.y, v1.x);
    // Map to [0, pi)
    theta = fmod(theta, kPiF);
    if (theta < 0.0f) theta += kPiF;
    if (theta >= kPiF) theta -= kPiF;

    sigma1 = sqrt(lambda1);
    sigma2 = sqrt(lambda2);
    return isfinite(theta) && isfinite(sigma1) && isfinite(sigma2);
}

inline float3 conicFromThetaSigmas(float theta, float sigma1, float sigma2) {
    // Returns invCov (A,B,C) in pixel^-2 such that:
    //   q = A*dx^2 + 2B*dx*dy + C*dy^2
    // where dx,dy are pixel deltas from the mean.
    float c;
    float s = fast::sincos(theta, c);

    float sig1 = max(sigma1, 1e-4f);
    float sig2 = max(sigma2, 1e-4f);
    float invVar1 = 1.0f / (sig1 * sig1);
    float invVar2 = 1.0f / (sig2 * sig2);

    float cc = c * c;
    float ss = s * s;
    float cs = c * s;

    float A = cc * invVar1 + ss * invVar2;
    float B = cs * (invVar1 - invVar2);
    float C = ss * invVar1 + cc * invVar2;
    return float3(A, B, C);
}

inline float computeQMax(float alpha, float tau) {
    if (alpha <= tau) return 0.0f;
    return -2.0f * log(tau / alpha);
}

// Evaluate quadratic q(x,y) = a*x² + 2*b*x*y + c*y²
inline float evalQuad(float x, float y, float a, float b, float c) {
    return a * x * x + 2.0f * b * x * y + c * y * y;
}

// Compute exact minimum of q(x,y) = a*x² + 2*b*x*y + c*y² over axis-aligned rectangle
// Rectangle coords are relative to mean: [xmin,xmax] × [ymin,ymax]
// Returns the minimum value of the quadratic over the rectangle
inline float minQuadRect(float xmin, float xmax, float ymin, float ymax,
                         float a, float b, float c) {
    // If mean (origin) is inside tile, min is at mean => q=0
    if (xmin <= 0.0f && 0.0f <= xmax && ymin <= 0.0f && 0.0f <= ymax) {
        return 0.0f;
    }

    // Avoid div-by-zero (a,c > 0 for valid ellipse)
    float invA = 1.0f / max(a, 1e-20f);
    float invC = 1.0f / max(c, 1e-20f);

    float qmin = INFINITY;

    // Vertical edge x = xmin: optimal y* = -(b/c) * x, clamped to [ymin, ymax]
    {
        float x = xmin;
        float y = clamp(-(b * invC) * x, ymin, ymax);
        qmin = min(qmin, evalQuad(x, y, a, b, c));
    }
    // Vertical edge x = xmax
    {
        float x = xmax;
        float y = clamp(-(b * invC) * x, ymin, ymax);
        qmin = min(qmin, evalQuad(x, y, a, b, c));
    }
    // Horizontal edge y = ymin: optimal x* = -(b/a) * y, clamped to [xmin, xmax]
    {
        float y = ymin;
        float x = clamp(-(b * invA) * y, xmin, xmax);
        qmin = min(qmin, evalQuad(x, y, a, b, c));
    }
    // Horizontal edge y = ymax
    {
        float y = ymax;
        float x = clamp(-(b * invA) * y, xmin, xmax);
        qmin = min(qmin, evalQuad(x, y, a, b, c));
    }

    return qmin;
}

// Build inverse covariance (conic A,B,C) from sigma1, sigma2, theta
// Returns (a, b, c) such that q(dx,dy) = a*dx² + 2*b*dx*dy + c*dy²
// where dx,dy are pixel offsets from mean
inline float3 conicFromSigmaTheta(float sigma1, float sigma2, float theta) {
    float c, s;
    s = fast::sincos(theta, c);

    float invS1sq = 1.0f / max(sigma1 * sigma1, 1e-12f);
    float invS2sq = 1.0f / max(sigma2 * sigma2, 1e-12f);

    float cc = c * c;
    float ss = s * s;
    float cs = c * s;

    float a = cc * invS1sq + ss * invS2sq;
    float conic_c = ss * invS1sq + cc * invS2sq;
    float b = cs * (invS1sq - invS2sq);

    return float3(a, b, conic_c);
}

// Test if max alpha in tile >= tau using exact quadratic minimization
// Returns d²_cutoff for the test: if minQuadRect(...) <= d²_cutoff, tile passes
// Returns negative value if opacity < tau (nothing can pass)
inline float computeD2Cutoff(float opacity, float tau) {
    if (opacity < tau) return -1.0f;
    return -2.0f * log(tau / opacity);
}

inline float gaussianComputePower(float opacity) {
    return GAUSSIAN_LN2 * 8.0f + GAUSSIAN_LN2 * log2(max(opacity, 1e-6f));
}

inline bool gaussianSegmentIntersectEllipse(float a, float b, float c, float d, float l, float r) {
    float delta = b * b - 4.0f * a * c;
    float t1 = (l - d) * (2.0f * a) + b;
    float t2 = (r - d) * (2.0f * a) + b;
    return delta >= 0.0f && (t1 <= 0.0f || t1 * t1 <= delta) && (t2 >= 0.0f || t2 * t2 <= delta);
}

inline bool gaussianTileContainsCenter(int2 pix_min, int2 pix_max, float2 center) {
    return center.x >= float(pix_min.x) && center.x <= float(pix_max.x) &&
           center.y >= float(pix_min.y) && center.y <= float(pix_max.y);
}

inline bool gaussianTileIntersectsEllipse(int2 pix_min, int2 pix_max, float2 center, float3 conic, float power) {
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
    if (gaussianSegmentIntersectEllipse(a, b, c, center.y, float(pix_min.y), float(pix_max.y))) {
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
    if (gaussianSegmentIntersectEllipse(a, b, c, center.x, float(pix_min.x), float(pix_max.x))) {
        return true;
    }

    return false;
}

inline bool gaussianIntersectsTile(int2 pix_min, int2 pix_max, float2 center, float3 conic, float power) {
    return gaussianTileContainsCenter(pix_min, pix_max, center) ||
           gaussianTileIntersectsEllipse(pix_min, pix_max, center, conic, power);
}

inline bool intersectsTile(
    int2 pix_min, int2 pix_max,
    float2 center, float3 conic, float opacity, float2 obbExtents
) {
    float power = gaussianComputePower(opacity);
    return gaussianIntersectsTile(pix_min, pix_max, center, conic, power);
}

inline bool intersectsTileSimple(
    int2 pix_min, int2 pix_max,
    float2 center, float3 conic, float opacity, float radius
) {
    float power = gaussianComputePower(opacity);
    return gaussianIntersectsTile(pix_min, pix_max, center, conic, power);
}

inline bool intersectsTileHalf(
    int2 pix_min, int2 pix_max,
    half2 center, half3 conic, half opacity, half2 obbExtents
) {
    float power = gaussianComputePower(float(opacity));
    return gaussianIntersectsTile(pix_min, pix_max, float2(center), float3(conic), power);
}

// ============================================================================
// Unified Covariance Stabilization
// Ensures numerical stability for half-precision packing and prevents
// degenerate conics (needles/lines). Use this before packing to half.
// ============================================================================

constant float kCovStabilize_MinVar = 1e-4f;
constant float kCovStabilize_MinDet = 1e-8f;
constant float kCovStabilize_MaxAxisRatio = 256.0f;
constant float kCovStabilize_BoundsRadius = 3.0f;

inline float2x2 stabilizeCovariance2D(float2x2 cov, float2 screenSize) {
    float maxCond = kCovStabilize_MaxAxisRatio * kCovStabilize_MaxAxisRatio;

    float maxDim = max(screenSize.x, screenSize.y);
    float maxExtentPx = maxDim * 2.0f;
    float maxEig = (maxExtentPx / kCovStabilize_BoundsRadius);
    maxEig = maxEig * maxEig;

    float a = cov[0][0];
    float b = 0.5f * (cov[0][1] + cov[1][0]);
    float d = cov[1][1];

    if (!isfinite(a) || !isfinite(b) || !isfinite(d)) {
        return float2x2(1.0f, 0.0f, 0.0f, 1.0f);
    }

    a = max(a, kCovStabilize_MinVar);
    d = max(d, kCovStabilize_MinVar);

    float det = a * d - b * b;
    if (!isfinite(det) || det < kCovStabilize_MinDet) {
        float bump = (kCovStabilize_MinDet - det) + kCovStabilize_MinVar;
        a += bump;
        d += bump;
        det = a * d - b * b;
    }

    float mid = 0.5f * (a + d);
    float disc = max(mid * mid - det, 0.0f);
    float sqrtDisc = sqrt(disc);
    float lambda1 = mid + sqrtDisc;
    float lambda2 = max(mid - sqrtDisc, kCovStabilize_MinVar);

    float2 v1;
    if (abs(b) > 1e-8f) {
        float vx = b;
        float vy = lambda1 - a;
        float vlen = sqrt(vx * vx + vy * vy);
        v1 = float2(vx, vy) / max(vlen, 1e-8f);
    } else {
        v1 = (a >= d) ? float2(1.0f, 0.0f) : float2(0.0f, 1.0f);
    }
    float2 v2 = float2(v1.y, -v1.x);

    lambda1 = min(lambda1, maxEig);
    lambda2 = max(lambda2, lambda1 / maxCond);

    float2x2 covOut =
        lambda1 * float2x2(v1.x * v1.x, v1.x * v1.y,
                           v1.y * v1.x, v1.y * v1.y) +
        lambda2 * float2x2(v2.x * v2.x, v2.x * v2.y,
                           v2.y * v2.x, v2.y * v2.y);

    return covOut;
}

// ============================================================================
// Unified Culling Helpers
// These consolidate the duplicated culling logic across all projection kernels
// ============================================================================

// Scale culling: reject Gaussians that are too small to matter
constant float kMinGaussianScale = 0.0005f;

inline bool cullByScale(float3 scale) {
    float maxScale = max(scale.x, max(scale.y, scale.z));
    return maxScale < kMinGaussianScale;
}

// Radius culling: reject projected Gaussians smaller than half a pixel
constant float kMinProjectedRadius = 0.5f;

inline bool cullByRadius(float radius) {
    return radius < kMinProjectedRadius;
}

// Far plane culling
inline bool cullByFarPlane(float depth, float farPlane) {
    return depth > farPlane;
}

// Total ink culling: reject Gaussians with negligible contribution
// Formula: totalInk = opacity * 2π * sqrt(det(cov2d))
// Threshold is depth-adaptive: closer Gaussians have higher threshold
inline bool cullByTotalInk(
    float opacity,
    float detCov2d,
    float depth,
    float nearPlane,
    float farPlane,
    float totalInkThreshold
) {
    if (totalInkThreshold <= 0.0f) return false;
    float totalInk = opacity * 6.283185f * sqrt(max(detCov2d, 1e-12f));
    float depthFactor = computeDepthFactor(depth, nearPlane, farPlane);
    float adjustedThreshold = depthFactor * totalInkThreshold;
    return totalInk < adjustedThreshold;
}

// Convenience: compute det from cov2d and call cullByTotalInk
inline bool cullByTotalInkFromCov(
    float opacity,
    float2x2 cov2d,
    float depth,
    float nearPlane,
    float farPlane,
    float totalInkThreshold
) {
    float a = cov2d[0][0];
    float b = 0.5f * (cov2d[0][1] + cov2d[1][0]);
    float d = cov2d[1][1];
    float detCov = a * d - b * b;
    return cullByTotalInk(opacity, detCov, depth, nearPlane, farPlane, totalInkThreshold);
}

// Off-screen culling using OBB extents
inline bool cullByScreenBounds(
    float2 screenPos,
    float2 obbExtents,
    float width,
    float height
) {
    return (screenPos.x + obbExtents.x < 0.0f ||
            screenPos.x - obbExtents.x > width ||
            screenPos.y + obbExtents.y < 0.0f ||
            screenPos.y - obbExtents.y > height);
}

// ============================================================================
// Unified Tile Bounds Computation
// ============================================================================

struct TileBounds {
    int minTileX;
    int maxTileX;
    int minTileY;
    int maxTileY;
    bool valid;
};

inline TileBounds computeTileBounds(
    float2 screenPos,
    float2 obbExtents,
    float width,
    float height,
    int tileWidth,
    int tileHeight,
    int tilesX,
    int tilesY
) {
    TileBounds result;

    float xmin = screenPos.x - obbExtents.x;
    float xmax = screenPos.x + obbExtents.x;
    float ymin = screenPos.y - obbExtents.y;
    float ymax = screenPos.y + obbExtents.y;

    float maxW = width - 1.0f;
    float maxH = height - 1.0f;

    xmin = clamp(xmin, 0.0f, maxW);
    xmax = clamp(xmax, 0.0f, maxW);
    ymin = clamp(ymin, 0.0f, maxH);
    ymax = clamp(ymax, 0.0f, maxH);

    result.minTileX = int(floor(xmin / float(tileWidth)));
    result.maxTileX = int(ceil(xmax / float(tileWidth))) - 1;
    result.minTileY = int(floor(ymin / float(tileHeight)));
    result.maxTileY = int(ceil(ymax / float(tileHeight))) - 1;

    result.minTileX = max(result.minTileX, 0);
    result.minTileY = max(result.minTileY, 0);
    result.maxTileX = min(result.maxTileX, tilesX - 1);
    result.maxTileY = min(result.maxTileY, tilesY - 1);

    result.valid = (result.minTileX <= result.maxTileX && result.minTileY <= result.maxTileY);
    return result;
}

// ============================================================================
// Unified Gaussian Projection Result
// Contains all computed values from projection + culling
// ============================================================================

struct GaussianProjection {
    // Screen space
    float2 screenPos;
    float depth;

    // Covariance decomposition
    float2x2 cov2d;
    float theta;
    float sigma1;
    float sigma2;
    float radius;
    float2 obbExtents;

    // Tile bounds
    TileBounds tileBounds;

    // Status
    bool visible;
    bool culled;        // true if any cull check failed
    uint cullReason;    // bitmask of which culls failed (for debugging)
};

// Cull reason flags
constant uint CULL_NONE = 0;
constant uint CULL_SCALE = 1 << 0;
constant uint CULL_NEAR_PLANE = 1 << 1;
constant uint CULL_FAR_PLANE = 1 << 2;
constant uint CULL_OPACITY = 1 << 3;
constant uint CULL_COVARIANCE = 1 << 4;
constant uint CULL_RADIUS = 1 << 5;
constant uint CULL_TOTAL_INK = 1 << 6;
constant uint CULL_OFF_SCREEN = 1 << 7;

#endif // GAUSSIAN_SHARED_H

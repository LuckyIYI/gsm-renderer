// GaussianShared.h - Shared constants for Gaussian rendering
// Included by both GlobalShaders.metal and LocalShaders.metal

#ifndef GAUSSIAN_SHARED_H
#define GAUSSIAN_SHARED_H

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// SPHERICAL HARMONICS CONSTANTS
// =============================================================================

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
// FUNCTION CONSTANTS FOR SH DEGREE (compile-time branching)
// =============================================================================
// SH degree: 0 = DC only (1 coeff), 1 = 4 coeffs, 2 = 9 coeffs, 3 = 16 coeffs
// These must be set when creating the pipeline state via MTLFunctionConstantValues

constant uint SH_DEGREE [[function_constant(0)]];
constant bool SH_DEGREE_0 = (SH_DEGREE == 0);
constant bool SH_DEGREE_1 = (SH_DEGREE == 1);
constant bool SH_DEGREE_2 = (SH_DEGREE == 2);
constant bool SH_DEGREE_3 = (SH_DEGREE == 3);

// =============================================================================
// SPHERICAL HARMONICS COLOR COMPUTATION (shared between all renderers)
// =============================================================================
// Template supports both float and half harmonics for memory bandwidth optimization
// Uses function constants for compile-time branching (eliminates runtime overhead)

template<typename HarmonicsT>
inline float3 computeSHColor(
    const device HarmonicsT* harmonics,
    uint gid,
    float3 pos,
    float3 cameraCenter,
    uint shComponents  // runtime fallback if function constant not set
) {
    // DC only - no direction needed, but still need SH_C0 coefficient
    if (SH_DEGREE_0 || shComponents == 0) {
        uint base = gid * 3u;
        return float3(harmonics[base], harmonics[base + 1], harmonics[base + 2]) * SH_C0;
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
// SHARED ELLIPSE-TILE INTERSECTION (used by both Local and Global pipelines)
// =============================================================================
// FlashGS-style intersection: exact ellipse-tile test
// power = ln2 * (8 + log2(opacity)) = ln(256 * opacity)
// Threshold w = 2 * power in quadratic form: conic.x*dx^2 + 2*conic.y*dx*dy + conic.z*dy^2 <= w

constant float GAUSSIAN_LN2 = 0.693147180559945f;

/// Compute power threshold from opacity (FlashGS exact)
inline float gaussianComputePower(float opacity) {
    return GAUSSIAN_LN2 * 8.0f + GAUSSIAN_LN2 * log2(max(opacity, 1e-6f));
}

/// Segment-ellipse intersection helper (FlashGS exact)
inline bool gaussianSegmentIntersectEllipse(float a, float b, float c, float d, float l, float r) {
    float delta = b * b - 4.0f * a * c;
    float t1 = (l - d) * (2.0f * a) + b;
    float t2 = (r - d) * (2.0f * a) + b;
    return delta >= 0.0f && (t1 <= 0.0f || t1 * t1 <= delta) && (t2 >= 0.0f || t2 * t2 <= delta);
}

/// Check if tile contains gaussian center
inline bool gaussianTileContainsCenter(int2 pix_min, int2 pix_max, float2 center) {
    return center.x >= float(pix_min.x) && center.x <= float(pix_max.x) &&
           center.y >= float(pix_min.y) && center.y <= float(pix_max.y);
}

/// Full ellipse-tile intersection test (FlashGS exact)
/// Tests closest vertical and horizontal edges
inline bool gaussianTileIntersectsEllipse(int2 pix_min, int2 pix_max, float2 center, float3 conic, float power) {
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

    if (gaussianSegmentIntersectEllipse(a, b, c, center.y, float(pix_min.y), float(pix_max.y))) {
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

    if (gaussianSegmentIntersectEllipse(a, b, c, center.x, float(pix_min.x), float(pix_max.x))) {
        return true;
    }

    return false;
}

/// Combined intersection test: center containment OR ellipse intersection
inline bool gaussianIntersectsTile(int2 pix_min, int2 pix_max, float2 center, float3 conic, float power) {
    return gaussianTileContainsCenter(pix_min, pix_max, center) ||
           gaussianTileIntersectsEllipse(pix_min, pix_max, center, conic, power);
}

// =============================================================================
// AABB-TILE INTERSECTION (simpler, faster, less precise)
// =============================================================================
// Use when ellipse precision isn't needed or for early-out culling

/// Check if gaussian AABB intersects tile (simple box overlap)
inline bool gaussianAABBIntersectsTile(int2 tileMin, int2 tileMax, int2 gaussianMinTile, int2 gaussianMaxTile, int tileX, int tileY) {
    return tileX >= gaussianMinTile.x && tileX < gaussianMaxTile.x &&
           tileY >= gaussianMinTile.y && tileY < gaussianMaxTile.y;
}

/// Check if pixel-space AABB intersects tile
inline bool gaussianPixelAABBIntersectsTile(int2 pix_min, int2 pix_max, float2 center, float radius) {
    float gaussianMinX = center.x - radius;
    float gaussianMaxX = center.x + radius;
    float gaussianMinY = center.y - radius;
    float gaussianMaxY = center.y + radius;

    return gaussianMaxX >= float(pix_min.x) && gaussianMinX <= float(pix_max.x) &&
           gaussianMaxY >= float(pix_min.y) && gaussianMinY <= float(pix_max.y);
}

// =============================================================================
// OBB (ORIENTED BOUNDING BOX) - GSCore-style tighter bounds
// =============================================================================
// Computes AABB of the oriented bounding box aligned to ellipse axes.
// Tighter than circular AABB, faster than exact ellipse intersection.

/// Compute OBB extents from conic (inverse covariance)
/// conic = (inv_cov[0][0], inv_cov[0][1], inv_cov[1][1])
inline float2 computeOBBExtentsFromConic(float3 conic, float sigma_multiplier) {
    // conic is inverse of covariance, invert back
    float invA = conic.x;
    float invB = conic.y;
    float invD = conic.z;

    float conicDet = invA * invD - invB * invB;
    if (abs(conicDet) < 1e-8f) {
        return float2(sigma_multiplier * 10.0f, sigma_multiplier * 10.0f);
    }

    // Invert: cov = (invD, -invB, invA) / conicDet
    float a = invD / conicDet;
    float b = -invB / conicDet;
    float d = invA / conicDet;

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
        float vx = b;
        float vy = lambda1 - a;
        float vlen = sqrt(vx * vx + vy * vy);
        v1 = float2(vx, vy) / max(vlen, 1e-6f);
    } else {
        v1 = (a >= d) ? float2(1.0f, 0.0f) : float2(0.0f, 1.0f);
    }

    return float2(abs(v1.x) * e1 + abs(v1.y) * e2,
                  abs(v1.y) * e1 + abs(v1.x) * e2);
}

/// Check if pixel-space OBB intersects tile
inline bool gaussianPixelOBBIntersectsTile(int2 pix_min, int2 pix_max, float2 center, float2 obbExtents) {
    float gaussianMinX = center.x - obbExtents.x;
    float gaussianMaxX = center.x + obbExtents.x;
    float gaussianMinY = center.y - obbExtents.y;
    float gaussianMaxY = center.y + obbExtents.y;

    return gaussianMaxX >= float(pix_min.x) && gaussianMinX <= float(pix_max.x) &&
           gaussianMaxY >= float(pix_min.y) && gaussianMinY <= float(pix_max.y);
}

// =============================================================================
// UNIFIED TILE INTERSECTION - Single entry point with method selection
// =============================================================================
// Function constant INTERSECTION_MODE selects the method at compile time:
//   0 = AABB (fastest, least precise - circular radius)
//   1 = OBB  (medium - oriented bounding box, GSCore-style)
//   2 = Ellipse (slowest, exact - FlashGS-style)

constant uint INTERSECTION_MODE [[function_constant(10)]];
constant bool USE_AABB_MODE = (INTERSECTION_MODE == 0);
constant bool USE_OBB_MODE = (INTERSECTION_MODE == 1);
constant bool USE_ELLIPSE_MODE = (INTERSECTION_MODE == 2);

/// Unified tile intersection test - single entry point
/// Automatically selects AABB, OBB, or Ellipse based on INTERSECTION_MODE
/// @param pix_min Tile minimum pixel coordinates
/// @param pix_max Tile maximum pixel coordinates
/// @param center Gaussian center in pixel coordinates
/// @param conic Inverse covariance (conicA, conicB, conicC)
/// @param opacity Gaussian opacity (for ellipse power threshold)
/// @param radius AABB radius (only used in AABB mode)
/// @param obbExtents Pre-computed OBB extents (pass float2(0) to compute on-the-fly)
inline bool intersectsTile(
    int2 pix_min,
    int2 pix_max,
    float2 center,
    float3 conic,
    float opacity,
    float radius,
    float2 obbExtents
) {
    if (USE_AABB_MODE) {
        // AABB mode - fastest, uses circular radius
        return gaussianPixelAABBIntersectsTile(pix_min, pix_max, center, radius);
    }
    else if (USE_OBB_MODE) {
        // OBB mode - medium speed, uses oriented bounding box
        float2 extents = (obbExtents.x > 0.0f) ? obbExtents : computeOBBExtentsFromConic(conic, 3.0f);
        return gaussianPixelOBBIntersectsTile(pix_min, pix_max, center, extents);
    }
    else {
        // Ellipse mode - exact intersection (FlashGS)
        float power = gaussianComputePower(opacity);
        return gaussianIntersectsTile(pix_min, pix_max, center, conic, power);
    }
}

/// Simplified unified intersection (computes OBB on-the-fly if needed)
inline bool intersectsTileSimple(
    int2 pix_min,
    int2 pix_max,
    float2 center,
    float3 conic,
    float opacity,
    float radius
) {
    return intersectsTile(pix_min, pix_max, center, conic, opacity, radius, float2(0.0f));
}

// =============================================================================
// HALF-PRECISION VARIANTS
// =============================================================================

inline half2 computeOBBExtentsFromConicHalf(half3 conic, half sigma_multiplier) {
    half invA = conic.x;
    half invB = conic.y;
    half invD = conic.z;

    half conicDet = invA * invD - invB * invB;
    if (abs(conicDet) < half(1e-6h)) {
        return half2(sigma_multiplier * half(10.0h), sigma_multiplier * half(10.0h));
    }

    half a = invD / conicDet;
    half b = -invB / conicDet;
    half d = invA / conicDet;

    half det = a * d - b * b;
    half mid = half(0.5h) * (a + d);
    half disc = max(mid * mid - det, half(1e-6h));
    half sqrtDisc = sqrt(disc);

    half lambda1 = mid + sqrtDisc;
    half lambda2 = max(mid - sqrtDisc, half(1e-6h));

    half e1 = sigma_multiplier * sqrt(max(lambda1, half(1e-6h)));
    half e2 = sigma_multiplier * sqrt(max(lambda2, half(1e-6h)));

    half2 v1;
    if (abs(b) > half(1e-6h)) {
        half vx = b;
        half vy = lambda1 - a;
        half vlen = sqrt(vx * vx + vy * vy);
        v1 = half2(vx, vy) / max(vlen, half(1e-6h));
    } else {
        v1 = (a >= d) ? half2(half(1.0h), half(0.0h)) : half2(half(0.0h), half(1.0h));
    }

    return half2(abs(v1.x) * e1 + abs(v1.y) * e2,
                 abs(v1.y) * e1 + abs(v1.x) * e2);
}

inline bool intersectsTileHalf(
    int2 pix_min,
    int2 pix_max,
    half2 center,
    half3 conic,
    half opacity,
    half radius,
    half2 obbExtents
) {
    if (USE_AABB_MODE) {
        half gaussianMinX = center.x - radius;
        half gaussianMaxX = center.x + radius;
        half gaussianMinY = center.y - radius;
        half gaussianMaxY = center.y + radius;
        return gaussianMaxX >= half(pix_min.x) && gaussianMinX <= half(pix_max.x) &&
               gaussianMaxY >= half(pix_min.y) && gaussianMinY <= half(pix_max.y);
    }
    else if (USE_OBB_MODE) {
        half2 extents = (obbExtents.x > half(0.0h)) ? obbExtents : computeOBBExtentsFromConicHalf(conic, half(3.0h));
        half gaussianMinX = center.x - extents.x;
        half gaussianMaxX = center.x + extents.x;
        half gaussianMinY = center.y - extents.y;
        half gaussianMaxY = center.y + extents.y;
        return gaussianMaxX >= half(pix_min.x) && gaussianMinX <= half(pix_max.x) &&
               gaussianMaxY >= half(pix_min.y) && gaussianMinY <= half(pix_max.y);
    }
    else {
        // Ellipse mode - use float for precision
        float power = gaussianComputePower(float(opacity));
        return gaussianIntersectsTile(pix_min, pix_max, float2(center), float3(conic), power);
    }
}

#endif // GAUSSIAN_SHARED_H

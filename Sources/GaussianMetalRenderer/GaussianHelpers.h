// GaussianHelpers.h - Shared helper functions for Gaussian splatting
// Used by both global radix sort (GaussianMetalRenderer) and local sort (Local) pipelines

#ifndef GAUSSIAN_HELPERS_H
#define GAUSSIAN_HELPERS_H

#include <metal_stdlib>
#include "BridgingTypes.h"
using namespace metal;

// =============================================================================
// STRUCT ACCESSOR HELPERS
// =============================================================================

// Get rotation from PackedWorldGaussian (has simd_float4 rotation)
inline float4 getRotation(PackedWorldGaussian g) { return g.rotation; }

// Get rotation from PackedWorldGaussianHalf (has individual rx,ry,rz,rw)
inline float4 getRotation(PackedWorldGaussianHalf g) {
    return float4(half(g.rx), half(g.ry), half(g.rz), half(g.rw));
}

// =============================================================================
// MATH HELPERS
// =============================================================================

/// Matrix from row vectors (converts row-major input to column-major)
inline float3x3 matrixFromRows(float3 r0, float3 r1, float3 r2) {
    return float3x3(
        float3(r0.x, r1.x, r2.x),
        float3(r0.y, r1.y, r2.y),
        float3(r0.z, r1.z, r2.z)
    );
}

/// Safe depth component (avoid division by zero)
inline float safeDepthComponent(float value) {
    float absVal = fabs(value);
    if (absVal < 1e-4f) {
        return value >= 0 ? 1e-4f : -1e-4f;
    }
    return value;
}

/// Normalize quaternion with fallback for degenerate cases
inline float4 normalizeQuaternion(float4 quat) {
    float norm = sqrt(max(dot(quat, quat), 1e-8f));
    if (norm < 1e-8f) {
        return float4(1.0f, 0.0f, 0.0f, 0.0f);
    }
    return quat / norm;
}

/// Quaternion to rotation matrix (3DGS convention: xyzw where w is scalar)
inline float3x3 quaternionToMatrix(float4 quat) {
    float x = quat.x;
    float y = quat.y;
    float z = quat.z;
    float r = quat.w;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float xz = x * z;
    float yz = y * z;
    float3 row0 = float3(
        1.0f - 2.0f * (yy + zz),
        2.0f * (xy - r * z),
        2.0f * (xz + r * y)
    );
    float3 row1 = float3(
        2.0f * (xy + r * z),
        1.0f - 2.0f * (xx + zz),
        2.0f * (yz - r * x)
    );
    float3 row2 = float3(
        2.0f * (xz - r * y),
        2.0f * (yz + r * x),
        1.0f - 2.0f * (xx + yy)
    );
    return matrixFromRows(row0, row1, row2);
}

/// Build 3D covariance matrix from scale and quaternion rotation
inline float3x3 buildCovariance3D(float3 scale, float4 quat) {
    float4 q = normalizeQuaternion(quat);
    float3x3 R = quaternionToMatrix(q);

    // RS = [R[:,0]*s.x, R[:,1]*s.y, R[:,2]*s.z]
    float3 RS0 = R[0] * scale.x;
    float3 RS1 = R[1] * scale.y;
    float3 RS2 = R[2] * scale.z;

    // Cov = RS * RS^T
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

/// Project 3D covariance to 2D screen space
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

    float invZ = 1.0f / max(1e-4f, viewPos.z);
    float invZ2 = invZ * invZ;

    float x = clamp(viewPos.x * invZ, -limX, limX) * viewPos.z;
    float y = clamp(viewPos.y * invZ, -limY, limY) * viewPos.z;

    float3 col0 = float3(focalX * invZ, 0.0f, 0.0f);
    float3 col1 = float3(0.0f, focalY * invZ, 0.0f);
    float3 col2 = float3(-(focalX * x) * invZ2, -(focalY * y) * invZ2, 0.0f);
    float3x3 J = float3x3(col0, col1, col2);

    float3x3 T = J * viewRotation;
    float3x3 covFull = T * cov3d * transpose(T);

    float2x2 cov2d = float2x2(
        covFull[0][0], covFull[0][1],
        covFull[1][0], covFull[1][1]
    );
    cov2d[0][0] += 0.3f;
    cov2d[1][1] += 0.3f;
    return cov2d;
}

/// Compute conic (inverse covariance) and radius from 2D covariance
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

/// Invert 2D covariance to get conic
inline float4 invertCovariance2D(float2x2 cov) {
    float det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    float invDet = 1.0f / max(det, 1e-8f);
    return float4(cov[1][1] * invDet, -cov[0][1] * invDet, cov[0][0] * invDet, 0.0f);
}

/// Compute radius from covariance eigenvalues
inline float radiusFromCovariance(float2x2 cov) {
    float det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    float mid = 0.5f * (cov[0][0] + cov[1][1]);
    float delta = max(mid * mid - det, 1e-5f);
    return 3.0f * ceil(sqrt(max(mid + sqrt(delta), 1e-5f)));
}

// =============================================================================
// OBB (ORIENTED BOUNDING BOX) COMPUTATION - GSCore-style
// =============================================================================
// Computes tighter bounds by using ellipse orientation instead of circular AABB.
// Reduces tile assignments by ~30-50% for elongated gaussians.
// Reference: GSCore (ASPLOS'24) - https://dl.acm.org/doi/10.1145/3620666.3651385

/// Compute OBB extents (half-widths along X and Y axes) from 2D covariance matrix
/// Returns float2(xExtent, yExtent) - the half-widths of the AABB of the OBB
/// sigma_multiplier: typically 3.0 for 3-sigma bounds
inline float2 computeOBBExtents(float2x2 cov, float sigma_multiplier) {
    float a = cov[0][0];
    float b = cov[0][1];  // = cov[1][0], symmetric
    float d = cov[1][1];

    float det = a * d - b * b;
    float mid = 0.5f * (a + d);
    float disc = max(mid * mid - det, 1e-6f);
    float sqrtDisc = sqrt(disc);

    // Eigenvalues (variances along principal axes)
    float lambda1 = mid + sqrtDisc;  // major axis variance
    float lambda2 = max(mid - sqrtDisc, 1e-6f);  // minor axis variance

    // Semi-axes lengths (standard deviations * sigma_multiplier)
    float e1 = sigma_multiplier * sqrt(max(lambda1, 1e-6f));  // major half-extent
    float e2 = sigma_multiplier * sqrt(max(lambda2, 1e-6f));  // minor half-extent

    // Eigenvector for major axis (v1)
    // For symmetric 2x2: if b != 0, v1 = normalize(b, lambda1 - a)
    // else v1 = (1,0) if a > d, else (0,1)
    float2 v1;
    if (abs(b) > 1e-6f) {
        float vx = b;
        float vy = lambda1 - a;
        float vlen = sqrt(vx * vx + vy * vy);
        v1 = float2(vx, vy) / max(vlen, 1e-6f);
    } else {
        // Diagonal matrix - eigenvectors are axis-aligned
        v1 = (a >= d) ? float2(1.0f, 0.0f) : float2(0.0f, 1.0f);
    }

    // Minor axis is perpendicular: v2 = (-v1.y, v1.x)
    // |v2.x| = |v1.y|, |v2.y| = |v1.x|

    // AABB of OBB: extent along each axis = |v1.component| * e1 + |v2.component| * e2
    float xExtent = abs(v1.x) * e1 + abs(v1.y) * e2;
    float yExtent = abs(v1.y) * e1 + abs(v1.x) * e2;

    return float2(xExtent, yExtent);
}

/// Compute OBB extents from individual covariance components
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
        float vx = b;
        float vy = lambda1 - a;
        float vlen = sqrt(vx * vx + vy * vy);
        v1 = float2(vx, vy) / max(vlen, 1e-6f);
    } else {
        v1 = (a >= d) ? float2(1.0f, 0.0f) : float2(0.0f, 1.0f);
    }

    float xExtent = abs(v1.x) * e1 + abs(v1.y) * e2;
    float yExtent = abs(v1.y) * e1 + abs(v1.x) * e2;

    return float2(xExtent, yExtent);
}

// =============================================================================
// ELLIPSE-TILE INTERSECTION (FlashGS-style)
// =============================================================================

/// tau = minimum opacity threshold for visibility (typically 1/255)
constant float GAUSSIAN_TAU = 1.0f / 255.0f;

/// Compute qMax threshold for ellipse intersection
/// weight = alpha * exp(-0.5 * q) >= tau  =>  q <= -2 * ln(tau / alpha)
inline float computeQMax(float alpha, float tau) {
    if (alpha <= tau) return 0.0f;
    return -2.0f * log(tau / alpha);
}

/// Check if point is inside ellipse: (p - center)^T * conic * (p - center) <= qMax
inline bool ellipseContainsPoint(float2 center, float3 conic, float qMax, float2 p) {
    float2 d = p - center;
    float q = d.x * d.x * conic.x + 2.0f * d.x * d.y * conic.y + d.y * d.y * conic.z;
    return q <= qMax;
}

/// Check if ellipse intersects line segment p0->p1
inline bool ellipseIntersectsEdge(float2 center, float3 conic, float qMax, float2 p0, float2 p1) {
    float2 dir = p1 - p0;
    float2 c = p0 - center;

    float a = conic.x * dir.x * dir.x + 2.0f * conic.y * dir.x * dir.y + conic.z * dir.y * dir.y;
    float b = 2.0f * (conic.x * c.x * dir.x + conic.y * (c.x * dir.y + c.y * dir.x) + conic.z * c.y * dir.y);
    float c0 = conic.x * c.x * c.x + 2.0f * conic.y * c.x * c.y + conic.z * c.y * c.y - qMax;

    if (abs(a) < 1e-6f) {
        if (abs(b) < 1e-6f) return false;
        float t = -c0 / b;
        return t >= 0.0f && t <= 1.0f;
    }

    float disc = b * b - 4.0f * a * c0;
    if (disc < 0.0f) return false;

    float sqrtDisc = sqrt(disc);
    float inv2a = 0.5f / a;
    float t0 = (-b - sqrtDisc) * inv2a;
    float t1 = (-b + sqrtDisc) * inv2a;

    return (t0 >= 0.0f && t0 <= 1.0f) || (t1 >= 0.0f && t1 <= 1.0f) ||
           (t0 < 0.0f && t1 > 1.0f);
}

/// Full ellipse-tile intersection test
inline bool ellipseIntersectsTile(float2 center, float3 conic, float qMax,
                                   float tileMinX, float tileMinY, float tileMaxX, float tileMaxY) {
    float2 p0 = float2(tileMinX, tileMinY);
    float2 p1 = float2(tileMaxX, tileMinY);
    float2 p2 = float2(tileMaxX, tileMaxY);
    float2 p3 = float2(tileMinX, tileMaxY);

    // 1) Any corner inside ellipse?
    if (ellipseContainsPoint(center, conic, qMax, p0) ||
        ellipseContainsPoint(center, conic, qMax, p1) ||
        ellipseContainsPoint(center, conic, qMax, p2) ||
        ellipseContainsPoint(center, conic, qMax, p3)) {
        return true;
    }

    // 2) Ellipse center inside tile?
    if (center.x >= tileMinX && center.x <= tileMaxX &&
        center.y >= tileMinY && center.y <= tileMaxY) {
        return true;
    }

    // 3) Ellipse intersects any edge?
    if (ellipseIntersectsEdge(center, conic, qMax, p0, p1)) return true;
    if (ellipseIntersectsEdge(center, conic, qMax, p1, p2)) return true;
    if (ellipseIntersectsEdge(center, conic, qMax, p2, p3)) return true;
    if (ellipseIntersectsEdge(center, conic, qMax, p3, p0)) return true;

    return false;
}

// =============================================================================
// HALF-PRECISION VARIANTS
// =============================================================================

inline half computeQMaxHalf(half alpha, half tau) {
    if (alpha <= tau) return half(0.0h);
    return half(-2.0h) * log(tau / alpha);
}

inline bool ellipseContainsPointHalf(half2 center, half3 conic, half qMax, half2 p) {
    half2 d = p - center;
    half q = d.x * d.x * conic.x + half(2.0h) * d.x * d.y * conic.y + d.y * d.y * conic.z;
    return q <= qMax;
}

inline bool ellipseIntersectsEdgeHalf(half2 center, half3 conic, half qMax, half2 p0, half2 p1) {
    half2 dir = p1 - p0;
    half2 c = p0 - center;
    half a = conic.x * dir.x * dir.x + half(2.0h) * conic.y * dir.x * dir.y + conic.z * dir.y * dir.y;
    half b = half(2.0h) * (conic.x * c.x * dir.x + conic.y * (c.x * dir.y + c.y * dir.x) + conic.z * c.y * dir.y);
    half c0 = conic.x * c.x * c.x + half(2.0h) * conic.y * c.x * c.y + conic.z * c.y * c.y - qMax;
    half disc = b * b - half(4.0h) * a * c0;
    if (disc < half(0.0h)) return false;
    half sqrtDisc = sqrt(disc);
    half t1 = (-b - sqrtDisc) / (half(2.0h) * a);
    half t2 = (-b + sqrtDisc) / (half(2.0h) * a);
    return (t1 >= half(0.0h) && t1 <= half(1.0h)) || (t2 >= half(0.0h) && t2 <= half(1.0h));
}

inline bool ellipseIntersectsTileHalf(half2 center, half3 conic, half qMax,
                                      half tileMinX, half tileMinY, half tileMaxX, half tileMaxY) {
    half2 p0 = half2(tileMinX, tileMinY);
    half2 p1 = half2(tileMaxX, tileMinY);
    half2 p2 = half2(tileMaxX, tileMaxY);
    half2 p3 = half2(tileMinX, tileMaxY);
    if (ellipseContainsPointHalf(center, conic, qMax, p0) ||
        ellipseContainsPointHalf(center, conic, qMax, p1) ||
        ellipseContainsPointHalf(center, conic, qMax, p2) ||
        ellipseContainsPointHalf(center, conic, qMax, p3)) {
        return true;
    }
    if (center.x >= tileMinX && center.x <= tileMaxX &&
        center.y >= tileMinY && center.y <= tileMaxY) {
        return true;
    }
    if (ellipseIntersectsEdgeHalf(center, conic, qMax, p0, p1) ||
        ellipseIntersectsEdgeHalf(center, conic, qMax, p1, p2) ||
        ellipseIntersectsEdgeHalf(center, conic, qMax, p2, p3) ||
        ellipseIntersectsEdgeHalf(center, conic, qMax, p3, p0)) {
        return true;
    }
    return false;
}

// =============================================================================
// FLASHGS-STYLE ELLIPSE INTERSECTION (faster edge-only check)
// =============================================================================
// Alternative ellipse intersection using closest-edge approach.
// Faster than full corner+edge check for tile-based rendering.

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

/// Full ellipse-tile intersection test (FlashGS closest-edge)
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

/// Combined ellipse intersection: center containment OR edge intersection
inline bool gaussianIntersectsTile(int2 pix_min, int2 pix_max, float2 center, float3 conic, float power) {
    return gaussianTileContainsCenter(pix_min, pix_max, center) ||
           gaussianTileIntersectsEllipse(pix_min, pix_max, center, conic, power);
}

// =============================================================================
// AABB-TILE INTERSECTION (simple box overlap)
// =============================================================================

/// Check if pixel-space circular AABB intersects tile
inline bool gaussianPixelAABBIntersectsTile(int2 pix_min, int2 pix_max, float2 center, float radius) {
    float gaussianMinX = center.x - radius;
    float gaussianMaxX = center.x + radius;
    float gaussianMinY = center.y - radius;
    float gaussianMaxY = center.y + radius;

    return gaussianMaxX >= float(pix_min.x) && gaussianMinX <= float(pix_max.x) &&
           gaussianMaxY >= float(pix_min.y) && gaussianMinY <= float(pix_max.y);
}

/// Check if pixel-space OBB (as AABB) intersects tile
inline bool gaussianPixelOBBIntersectsTile(int2 pix_min, int2 pix_max, float2 center, float2 obbExtents) {
    float gaussianMinX = center.x - obbExtents.x;
    float gaussianMaxX = center.x + obbExtents.x;
    float gaussianMinY = center.y - obbExtents.y;
    float gaussianMaxY = center.y + obbExtents.y;

    return gaussianMaxX >= float(pix_min.x) && gaussianMinX <= float(pix_max.x) &&
           gaussianMaxY >= float(pix_min.y) && gaussianMinY <= float(pix_max.y);
}

// =============================================================================
// OBB FROM CONIC (for scatter stage where only conic is available)
// =============================================================================

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

    // Reuse existing OBB computation
    return computeOBBExtentsFromComponents(a, b, d, sigma_multiplier);
}

// =============================================================================
// TILE INTERSECTION - Ellipse-based (FlashGS style, benchmarked as optimal)
// =============================================================================

/// Tile intersection test using ellipse (best precision/performance tradeoff)
inline bool intersectsTile(
    int2 pix_min,
    int2 pix_max,
    float2 center,
    float3 conic,
    float opacity,
    float radius,
    float2 obbExtents
) {
    float power = gaussianComputePower(opacity);
    return gaussianIntersectsTile(pix_min, pix_max, center, conic, power);
}

/// Simplified intersection (same signature for compatibility)
inline bool intersectsTileSimple(
    int2 pix_min,
    int2 pix_max,
    float2 center,
    float3 conic,
    float opacity,
    float radius
) {
    float power = gaussianComputePower(opacity);
    return gaussianIntersectsTile(pix_min, pix_max, center, conic, power);
}

/// Half-precision variant (promotes to float for ellipse math)
inline bool intersectsTileHalf(
    int2 pix_min,
    int2 pix_max,
    half2 center,
    half3 conic,
    half opacity,
    half radius,
    half2 obbExtents
) {
    float power = gaussianComputePower(float(opacity));
    return gaussianIntersectsTile(pix_min, pix_max, float2(center), float3(conic), power);
}

#endif // GAUSSIAN_HELPERS_H

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

    float2x2 cov2d = float2x2(covFull[0][0], covFull[0][1], covFull[1][0], covFull[1][1]);
    cov2d[0][0] += 0.3f;
    cov2d[1][1] += 0.3f;
    return cov2d;
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

inline float computeQMax(float alpha, float tau) {
    if (alpha <= tau) return 0.0f;
    return -2.0f * log(tau / alpha);
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
    float2 center, float3 conic, float opacity,
    float radius, float2 obbExtents
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
    half2 center, half3 conic, half opacity,
    half radius, half2 obbExtents
) {
    float power = gaussianComputePower(float(opacity));
    return gaussianIntersectsTile(pix_min, pix_max, float2(center), float3(conic), power);
}

#endif // GAUSSIAN_SHARED_H

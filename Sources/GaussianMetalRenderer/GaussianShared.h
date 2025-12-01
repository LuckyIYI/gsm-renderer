// GaussianShared.h - Shared constants for Gaussian rendering
// Included by both GaussianMetalRenderer.metal and TellusimShaders.metal

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

#endif // GAUSSIAN_SHARED_H

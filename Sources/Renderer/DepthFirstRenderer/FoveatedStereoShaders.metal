// FoveatedStereoShaders.metal
// Vertex and fragment shaders for foveated stereo rendering on Vision Pro
// Uses rasterization pipeline instead of compute for Compositor Services compatibility

#include <metal_stdlib>
#include "BridgingTypes.h"
using namespace metal;

// MARK: - Uniforms and Types

struct FoveatedStereoUniforms {
    uint width;
    uint height;
    uint tileWidth;
    uint tileHeight;
    uint tilesX;
    uint tilesY;
    uint viewIndex;      // 0 = left, 1 = right
    uint padding;
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
    uint viewIndex [[render_target_array_index]];
};

// Per-view uniforms for stereo (passed as buffer)
struct StereoViewUniforms {
    float4 viewport;     // x, y, width, height
    uint renderTargetArrayIndex;
    uint padding0;
    uint padding1;
    uint padding2;
};

// MARK: - Vertex Shader

// Fullscreen quad vertex shader with vertex amplification for stereo
// Renders a quad covering the full render target
// With amplification_count=2, this draws once for each eye
vertex VertexOut foveatedStereoVertex(
    uint vertexID [[vertex_id]],
    uint amplificationID [[amplification_id]],
    constant StereoViewUniforms* viewUniforms [[buffer(0)]]
) {
    // Generate fullscreen triangle (3 vertices cover the screen)
    // Using a single triangle is more efficient than a quad
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0)
    };

    float2 texCoords[3] = {
        float2(0.0, 1.0),
        float2(2.0, 1.0),
        float2(0.0, -1.0)
    };

    VertexOut out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.texCoord = texCoords[vertexID];
    out.viewIndex = viewUniforms[amplificationID].renderTargetArrayIndex;

    return out;
}

// MARK: - Fragment Shader Helper Functions

inline half2 getMean(const device GaussianRenderData& g) {
    return half2(g.meanX, g.meanY);
}

inline half4 getConic(const device GaussianRenderData& g) {
    return half4(g.conicA, g.conicB, g.conicC, g.conicD);
}

inline half3 getColor(const device GaussianRenderData& g) {
    return half3(g.colorR, g.colorG, g.colorB);
}

// MARK: - Fragment Shader

// Fragment shader that performs gaussian splatting blending
// Each fragment blends all gaussians that affect its pixel
// This replaces the compute kernel for Compositor Services compatibility
fragment half4 foveatedStereoFragment(
    VertexOut in [[stage_in]],
    constant FoveatedStereoUniforms& params [[buffer(0)]],
    const device GaussianHeader* tileHeaders [[buffer(1)]],
    const device GaussianRenderData* gaussians [[buffer(2)]],
    const device int* sortedGaussianIndices [[buffer(3)]],
    constant StereoViewUniforms* viewUniforms [[buffer(4)]]
) {
    // Get pixel coordinates from fragment position
    uint2 pixelCoord = uint2(in.position.xy);

    // Bounds check
    if (pixelCoord.x >= params.width || pixelCoord.y >= params.height) {
        return half4(0.0h, 0.0h, 0.0h, 1.0h);
    }

    // Compute which tile this pixel belongs to
    uint tileX = pixelCoord.x / params.tileWidth;
    uint tileY = pixelCoord.y / params.tileHeight;
    uint tileIdx = tileY * params.tilesX + tileX;

    // Get tile header
    GaussianHeader hdr = tileHeaders[tileIdx];

    // Initialize accumulation
    half3 color = half3(0.0h);
    half transmittance = 1.0h;
    half depth = 0.0h;

    // Pixel position as half for gaussian evaluation
    half2 pixelPos = half2(half(pixelCoord.x), half(pixelCoord.y));

    uint start = hdr.offset;
    uint count = hdr.count;

    // Blend gaussians front-to-back (depth sorted)
    for (uint i = 0; i < count; i++) {
        // Early termination when fully opaque
        if (transmittance < half(1.0h / 255.0h)) {
            break;
        }

        int gaussianIdx = sortedGaussianIndices[start + i];
        if (gaussianIdx < 0) {
            continue;
        }

        GaussianRenderData g = gaussians[gaussianIdx];

        // Get gaussian parameters
        half4 gConic = getConic(g);
        half cxx = gConic.x;
        half cyy = gConic.z;
        half cxy2 = 2.0h * gConic.y;
        half opacity = g.opacity;
        half3 gColor = getColor(g);
        half gDepth = g.depth;

        // Compute distance from pixel to gaussian center
        half2 gMean = getMean(g);
        half2 d = pixelPos - gMean;

        // Evaluate gaussian (quadratic form)
        half power = d.x * d.x * cxx + d.y * d.y * cyy + d.x * d.y * cxy2;

        // Compute alpha with gaussian falloff
        half alpha = min(opacity * exp(-0.5h * power), 0.99h);

        // Skip if alpha is negligible
        if (alpha < 1.0h / 255.0h) {
            continue;
        }

        // Accumulate color and depth (front-to-back blending)
        color += gColor * (alpha * transmittance);
        depth += gDepth * (alpha * transmittance);

        // Update transmittance
        transmittance *= (1.0h - alpha);
    }

    // Output color with alpha (1 - remaining transmittance)
    return half4(color, 1.0h - transmittance);
}

// MARK: - Depth Output Fragment Shader

// Fragment output structure for color + depth
struct FragmentOutput {
    half4 color [[color(0)]];
    float depth [[depth(any)]];
};

// Fragment shader variant that also outputs depth
fragment FragmentOutput foveatedStereoFragmentWithDepth(
    VertexOut in [[stage_in]],
    constant FoveatedStereoUniforms& params [[buffer(0)]],
    const device GaussianHeader* tileHeaders [[buffer(1)]],
    const device GaussianRenderData* gaussians [[buffer(2)]],
    const device int* sortedGaussianIndices [[buffer(3)]],
    constant StereoViewUniforms* viewUniforms [[buffer(4)]]
) {
    // Get pixel coordinates from fragment position
    uint2 pixelCoord = uint2(in.position.xy);

    FragmentOutput out;

    // Bounds check
    if (pixelCoord.x >= params.width || pixelCoord.y >= params.height) {
        out.color = half4(0.0h, 0.0h, 0.0h, 1.0h);
        out.depth = 1.0;
        return out;
    }

    // Compute which tile this pixel belongs to
    uint tileX = pixelCoord.x / params.tileWidth;
    uint tileY = pixelCoord.y / params.tileHeight;
    uint tileIdx = tileY * params.tilesX + tileX;

    // Get tile header
    GaussianHeader hdr = tileHeaders[tileIdx];

    // Initialize accumulation
    half3 color = half3(0.0h);
    half transmittance = 1.0h;
    half accumulatedDepth = 0.0h;

    // Pixel position as half for gaussian evaluation
    half2 pixelPos = half2(half(pixelCoord.x), half(pixelCoord.y));

    uint start = hdr.offset;
    uint count = hdr.count;

    // Blend gaussians front-to-back (depth sorted)
    for (uint i = 0; i < count; i++) {
        // Early termination when fully opaque
        if (transmittance < half(1.0h / 255.0h)) {
            break;
        }

        int gaussianIdx = sortedGaussianIndices[start + i];
        if (gaussianIdx < 0) {
            continue;
        }

        GaussianRenderData g = gaussians[gaussianIdx];

        // Get gaussian parameters
        half4 gConic = getConic(g);
        half cxx = gConic.x;
        half cyy = gConic.z;
        half cxy2 = 2.0h * gConic.y;
        half opacity = g.opacity;
        half3 gColor = getColor(g);
        half gDepth = g.depth;

        // Compute distance from pixel to gaussian center
        half2 gMean = getMean(g);
        half2 d = pixelPos - gMean;

        // Evaluate gaussian (quadratic form)
        half power = d.x * d.x * cxx + d.y * d.y * cyy + d.x * d.y * cxy2;

        // Compute alpha with gaussian falloff
        half alpha = min(opacity * exp(-0.5h * power), 0.99h);

        // Skip if alpha is negligible
        if (alpha < 1.0h / 255.0h) {
            continue;
        }

        // Accumulate color and depth (front-to-back blending)
        color += gColor * (alpha * transmittance);
        accumulatedDepth += gDepth * (alpha * transmittance);

        // Update transmittance
        transmittance *= (1.0h - alpha);
    }

    // Output color with alpha
    out.color = half4(color, 1.0h - transmittance);
    // Convert accumulated depth to hardware depth (0-1 range, assuming linear depth)
    out.depth = float(accumulatedDepth);

    return out;
}

// MARK: - Clear Shader for Render Pass

// Simple clear vertex shader
vertex float4 clearVertex(uint vertexID [[vertex_id]]) {
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0)
    };
    return float4(positions[vertexID], 0.0, 1.0);
}

// Clear fragment shader
fragment half4 clearFragment() {
    return half4(0.0h, 0.0h, 0.0h, 1.0h);
}

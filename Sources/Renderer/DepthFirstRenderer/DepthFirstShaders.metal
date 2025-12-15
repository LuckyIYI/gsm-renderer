#include <metal_stdlib>
#include <metal_atomic>
#include "BridgingTypes.h"
#include "GaussianShared.h"
#include "GaussianPrefixScan.h"
#include "RadixSortHelpers.h"
using namespace metal;

inline half2 getMean(const device GaussianRenderData& g) { return half2(g.meanX, g.meanY); }
inline half3 getColor(const device GaussianRenderData& g) { return half3(half(g.colorR) / 255.0h, half(g.colorG) / 255.0h, half(g.colorB) / 255.0h); }
inline half getOpacity(const device GaussianRenderData& g) { return half(g.opacity) / 255.0h; }

inline half2 getMean(const thread GaussianRenderData& g) { return half2(g.meanX, g.meanY); }
inline half3 getColor(const thread GaussianRenderData& g) { return half3(half(g.colorR) / 255.0h, half(g.colorG) / 255.0h, half(g.colorB) / 255.0h); }
inline half getOpacity(const thread GaussianRenderData& g) { return half(g.opacity) / 255.0h; }


struct DepthFirstParams {
    uint gaussianCount;
    uint visibleCount;
    uint tilesX;
    uint tilesY;
    uint tileWidth;
    uint tileHeight;
    uint maxAssignments;
    float alphaThreshold;
};

// DepthFirstHeader is now defined in BridgingTypes.h

constant bool DF_DEPTH_KEY_16 [[function_constant(2)]]; // default = false

inline uint float_to_sortable_uint(float v) {
    uint bits = as_type<uint>(v);
    uint mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

inline float sortable_uint_to_float(uint v) {
    // Inverse of float_to_sortable_uint (reversible mapping).
    uint bits = (v & 0x80000000u) ? (v ^ 0x80000000u) : ~v;
    return as_type<float>(bits);
}


template <typename PackedWorldT, typename HarmonicT, typename RenderDataT>
	kernel void depthFirstPreprocessKernel(
	    const device PackedWorldT* worldGaussians [[buffer(0)]],
	    const device HarmonicT* harmonics [[buffer(1)]],
	    device RenderDataT* outRenderData [[buffer(2)]],
	    device int4* outBounds [[buffer(3)]],
	    device uint* preDepthKeys [[buffer(4)]],
	    device uint* nTouchedTiles [[buffer(5)]],
	    device atomic_uint* totalInstances [[buffer(6)]],
	    constant CameraUniforms& camera [[buffer(7)]],
	    constant TileBinningParams& params [[buffer(8)]],
	    uint gid [[thread_position_in_grid]]
	) {
    if (gid >= camera.gaussianCount) { return; }

    PackedWorldT g = worldGaussians[gid];

	float3 scale = float3(g.sx, g.sy, g.sz);
	if (cullByScale(scale)) {
	    nTouchedTiles[gid] = 0;
	    outBounds[gid] = int4(0, -1, 0, -1);
	    preDepthKeys[gid] = 0xFFFFFFFFu;
	    return;
	}

    float3 position = float3(g.px, g.py, g.pz);
    float4 viewPos4 = camera.viewMatrix * float4(position, 1.0f);
    float4 clip = camera.projectionMatrix * viewPos4;

    float depth = clip.w;
	if (!isInFrontOfCameraClipW(clip.w, camera.nearPlane)) {
	    nTouchedTiles[gid] = 0;
	    outBounds[gid] = int4(0, -1, 0, -1);
	    preDepthKeys[gid] = 0xFFFFFFFFu;
	    return;
	}
	if (cullByFarPlane(depth, camera.farPlane)) {
	    nTouchedTiles[gid] = 0;
	    outBounds[gid] = int4(0, -1, 0, -1);
	    preDepthKeys[gid] = 0xFFFFFFFFu;
	    return;
	}

    float ndcX = clip.x / clip.w;
    float ndcY = clip.y / clip.w;
    float2 screenPos = ndcToScreen(float2(ndcX, ndcY), camera.width, camera.height);

	float opacity = float(g.opacity);
	if (opacity < params.alphaThreshold) {
	    nTouchedTiles[gid] = 0;
	    outBounds[gid] = int4(0, -1, 0, -1);
	    preDepthKeys[gid] = 0xFFFFFFFFu;
	    return;
	}

    float4 quat = normalizeQuaternion(getRotation(g));
    float3x3 cov3d = buildCovariance3D(scale, quat);

    float2x2 cov2d = projectCovariance2D(cov3d, viewPos4.xyz, camera.viewMatrix,
                                          camera.projectionMatrix,
                                          float2(camera.width, camera.height));

    cov2d = stabilizeCovariance2D(cov2d, float2(camera.width, camera.height));

    float theta, sigma1, sigma2;
    if (!covarianceToThetaSigmas(cov2d, theta, sigma1, sigma2)) {
        nTouchedTiles[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }
    float radius = 3.0f * max(sigma1, sigma2);

    if (cullByRadius(radius)) {
        nTouchedTiles[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    if (cullByTotalInkFromCov(opacity, cov2d, depth, camera.nearPlane, camera.farPlane, params.totalInkThreshold)) {
        nTouchedTiles[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        preDepthKeys[gid] = 0xFFFFFFFFu;
        return;
    }

    float2 obbExtents = computeOBBExtents(cov2d, 3.0f);

    if (cullByScreenBounds(screenPos, obbExtents, camera.width, camera.height)) {
        nTouchedTiles[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        return;
    }

    float3 color = computeSHColor(harmonics, gid, position, camera.cameraCenter, camera.shComponents);
    color = max(color + float3(0.5f), float3(0.0f));
    color = maybeDecodeSRGBToLinear(color, camera.inputIsSRGB);

    RenderDataT renderData;
    renderData.meanX = half(screenPos.x);
    renderData.meanY = half(screenPos.y);
    renderData.theta = packThetaPi(theta);
    renderData.sigma1 = half(sigma1);
    renderData.sigma2 = half(sigma2);
    renderData.depth = half(depth);
    renderData.colorR = uchar(clamp(color.x * 255.0f, 0.0f, 255.0f));
    renderData.colorG = uchar(clamp(color.y * 255.0f, 0.0f, 255.0f));
    renderData.colorB = uchar(clamp(color.z * 255.0f, 0.0f, 255.0f));
    renderData.opacity = uchar(clamp(opacity * 255.0f, 0.0f, 255.0f));
    outRenderData[gid] = renderData;

    // Compute tile bounds using unified helper
    TileBounds tb = computeTileBounds(
        screenPos, obbExtents,
        camera.width, camera.height,
        int(params.tileWidth), int(params.tileHeight),
        int(params.tilesX), int(params.tilesY)
    );

    outBounds[gid] = int4(tb.minTileX, tb.maxTileX, tb.minTileY, tb.maxTileY);

    float meanX_q = float(renderData.meanX);
    float meanY_q = float(renderData.meanY);
    float theta_q = unpackThetaPi(renderData.theta);
    float sigma1_q = float(renderData.sigma1);
    float sigma2_q = float(renderData.sigma2);
    float opacity_q = float(renderData.opacity) * (1.0f / 255.0f);

    float3 conic = conicFromSigmaTheta(sigma1_q, sigma2_q, theta_q);
    float conic_a = conic.x;
    float conic_b = conic.y;
    float conic_c = conic.z;

    float tau = max(params.alphaThreshold, 1e-12f);
    float d2Cutoff = computeD2Cutoff(opacity_q, tau);

    uint touchedCount = 0;
    if (d2Cutoff >= 0.0f) {
        float tileW = float(params.tileWidth);
        float tileH = float(params.tileHeight);

        for (int ty = tb.minTileY; ty <= tb.maxTileY; ++ty) {
            float tileMinY = float(ty) * tileH;
            float tileMaxY = tileMinY + tileH;
            float tile_ymin = tileMinY - meanY_q;
            float tile_ymax = tileMaxY - meanY_q;

            for (int tx = tb.minTileX; tx <= tb.maxTileX; ++tx) {
                float tileMinX = float(tx) * tileW;
                float tileMaxX = tileMinX + tileW;
                float tile_xmin = tileMinX - meanX_q;
                float tile_xmax = tileMaxX - meanX_q;

                float d2min = minQuadRect(tile_xmin, tile_xmax, tile_ymin, tile_ymax,
                                          conic_a, conic_b, conic_c);
                if (d2min <= d2Cutoff) {
                    touchedCount++;
                }
            }
        }
    }

    if (touchedCount == 0) {
        nTouchedTiles[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        preDepthKeys[gid] = 0xFFFFFFFFu;
        return;
    }

    preDepthKeys[gid] = float_to_sortable_uint(depth);

    nTouchedTiles[gid] = touchedCount;

    atomic_fetch_add_explicit(totalInstances, touchedCount, memory_order_relaxed);
}

template [[host_name("depthFirstPreprocessKernel")]]
kernel void depthFirstPreprocessKernel<PackedWorldGaussian, float, GaussianRenderData>(
    const device PackedWorldGaussian*, const device float*,
    device GaussianRenderData*, device int4*, device uint*,
    device uint*, device atomic_uint*,
    constant CameraUniforms&, constant TileBinningParams&, uint);

template [[host_name("depthFirstPreprocessKernelHalf")]]
kernel void depthFirstPreprocessKernel<PackedWorldGaussianHalf, half, GaussianRenderData>(
    const device PackedWorldGaussianHalf*, const device half*,
    device GaussianRenderData*, device int4*, device uint*,
    device uint*, device atomic_uint*,
    constant CameraUniforms&, constant TileBinningParams&, uint);


struct EyeProjectionResult {
    float2 screenPos;      // px, py in screen coordinates
    float theta;           // ellipse angle in [0, pi)
    float sigma1;          // major axis sigma in pixels
    float sigma2;          // minor axis sigma in pixels
    float detCov;          // determinant of covariance (pixel^4)
    float2 obbExtents;     // OBB half-extents for tight bounds
    float depth;           // positive depth for sorting
    int4 tileBounds;       // minTileX, maxTileX, minTileY, maxTileY
    uint touchedTiles;     // number of tiles touched
    bool visible;          // whether visible in this eye
};

inline EyeProjectionResult projectToEye(
    float3 scenePos,       // gaussian position in scene space
    float3 scale,          // gaussian scale (sx, sy, sz)
    float4 quat,           // gaussian rotation quaternion
    float4x4 sceneTransform, // scene → world transform
    float4x4 viewMatrix,   // world → eye transform
    float4x4 projMatrix,
    float focalX,
    float focalY,
    float width,
    float height,
    float nearPlane,
    float farPlane,
    uint tileWidth,
    uint tileHeight,
    uint tilesX,
    uint tilesY
) {
    EyeProjectionResult result;
    result.visible = false;
    result.touchedTiles = 0;
    result.tileBounds = int4(0, -1, 0, -1);
    result.theta = 0.0f;
    result.sigma1 = 0.0f;
    result.sigma2 = 0.0f;
    result.detCov = 0.0f;

    float4 worldPos4 = sceneTransform * float4(scenePos, 1.0f);
    float4 viewPos4 = viewMatrix * worldPos4;
    float4 clip = projMatrix * viewPos4;
    float depth = clip.w;
    result.depth = depth;

    if (!isInFrontOfCameraClipW(clip.w, nearPlane)) {
        return result;
    }
    if (cullByFarPlane(depth, farPlane)) {
        return result;
    }

    float ndcX = clip.x / clip.w;
    float ndcY = clip.y / clip.w;
    result.screenPos = ndcToScreen(float2(ndcX, ndcY), width, height);

    float sceneScale = length(float3(sceneTransform[0][0], sceneTransform[0][1], sceneTransform[0][2]));
    float3 scaledGaussianScale = scale * sceneScale;
    float3x3 cov3d = buildCovariance3D(scaledGaussianScale, quat);

    float2x2 cov2d = projectCovariance2D(cov3d, viewPos4.xyz, viewMatrix, projMatrix, float2(width, height));

    cov2d = stabilizeCovariance2D(cov2d, float2(width, height));

    float theta, sigma1, sigma2;
    if (!covarianceToThetaSigmas(cov2d, theta, sigma1, sigma2)) {
        return result;
    }
    result.theta = theta;
    result.sigma1 = sigma1;
    result.sigma2 = sigma2;
    float a = cov2d[0][0];
    float b = 0.5f * (cov2d[0][1] + cov2d[1][0]);
    float d = cov2d[1][1];
    result.detCov = max(a * d - b * b, 0.0f);
    float radius = 3.0f * max(sigma1, sigma2);

    if (cullByRadius(radius)) {
        return result;
    }

    result.obbExtents = computeOBBExtents(cov2d, 3.0f);

    if (cullByScreenBounds(result.screenPos, result.obbExtents, width, height)) {
        return result;
    }

    TileBounds tb = computeTileBounds(
        result.screenPos, result.obbExtents,
        width, height,
        int(tileWidth), int(tileHeight),
        int(tilesX), int(tilesY)
    );

    result.tileBounds = int4(tb.minTileX, tb.maxTileX, tb.minTileY, tb.maxTileY);

    uint tilesX_count = tb.valid ? uint(tb.maxTileX - tb.minTileX + 1) : 0;
    uint tilesY_count = tb.valid ? uint(tb.maxTileY - tb.minTileY + 1) : 0;
    result.touchedTiles = tilesX_count * tilesY_count;

    result.visible = true;
    return result;
}

template <typename PackedWorldT, typename HarmonicT>
	kernel void depthFirstStereoUnifiedPreprocessKernel(
	    const device PackedWorldT* worldGaussians [[buffer(0)]],
	    const device HarmonicT* harmonics [[buffer(1)]],
	    device StereoTiledRenderData* outRenderData [[buffer(2)]],
	    device int4* outBounds [[buffer(3)]],
	    device uint* preDepthKeys [[buffer(4)]],
	    device uint* nTouchedTiles [[buffer(5)]],
	    device atomic_uint* totalInstances [[buffer(6)]],
	    constant StereoCameraUniforms& camera [[buffer(7)]],
	    constant TileBinningParams& params [[buffer(8)]],
	    uint gid [[thread_position_in_grid]]
) {
    if (gid >= camera.gaussianCount) { return; }

    PackedWorldT g = worldGaussians[gid];

    // Early cull: skip tiny gaussians
	float3 scale = float3(g.sx, g.sy, g.sz);
	if (cullByScale(scale)) {
	    nTouchedTiles[gid] = 0;
	    outBounds[gid] = int4(0, -1, 0, -1);
	    preDepthKeys[gid] = 0xFFFFFFFFu;
	    return;
	}

    float3 position = float3(g.px, g.py, g.pz);

	float opacity = float(g.opacity);
	if (opacity < params.alphaThreshold) {
	    nTouchedTiles[gid] = 0;
	    outBounds[gid] = int4(0, -1, 0, -1);
	    preDepthKeys[gid] = 0xFFFFFFFFu;
	    return;
	}

    float4 quat = normalizeQuaternion(getRotation(g));

    EyeProjectionResult leftResult = projectToEye(
        position, scale, quat,
        camera.sceneTransform,
        camera.leftViewMatrix, camera.leftProjectionMatrix,
        camera.leftFocalX, camera.leftFocalY,
        camera.width, camera.height, camera.nearPlane, camera.farPlane,
        params.tileWidth, params.tileHeight, params.tilesX, params.tilesY
    );

    EyeProjectionResult rightResult = projectToEye(
        position, scale, quat,
        camera.sceneTransform,
        camera.rightViewMatrix, camera.rightProjectionMatrix,
        camera.rightFocalX, camera.rightFocalY,
        camera.width, camera.height, camera.nearPlane, camera.farPlane,
        params.tileWidth, params.tileHeight, params.tilesX, params.tilesY
    );

	if (!leftResult.visible && !rightResult.visible) {
	    nTouchedTiles[gid] = 0;
	    outBounds[gid] = int4(0, -1, 0, -1);
	    preDepthKeys[gid] = 0xFFFFFFFFu;
	    return;
	}

    float checkDepth = leftResult.visible ? leftResult.depth : rightResult.depth;
    if (leftResult.visible && rightResult.visible) {
        checkDepth = (leftResult.depth + rightResult.depth) * 0.5f;
    }
    float detCov = leftResult.visible ? leftResult.detCov : rightResult.detCov;
    if (leftResult.visible && rightResult.visible) {
        detCov = max(leftResult.detCov, rightResult.detCov);
    }
	if (cullByTotalInk(opacity, detCov, checkDepth, camera.nearPlane, camera.farPlane, params.totalInkThreshold)) {
	    nTouchedTiles[gid] = 0;
	    outBounds[gid] = int4(0, -1, 0, -1);
	    preDepthKeys[gid] = 0xFFFFFFFFu;
	    return;
	}

    float3 leftCameraCenter = float3(camera.leftCameraCenterX, camera.leftCameraCenterY, camera.leftCameraCenterZ);
    float3 rightCameraCenter = float3(camera.rightCameraCenterX, camera.rightCameraCenterY, camera.rightCameraCenterZ);
    float3 midCamera = (leftCameraCenter + rightCameraCenter) * 0.5f;
    float3 color = computeSHColor(harmonics, gid, position, midCamera, camera.shComponents);
    color = max(color + float3(0.5f), float3(0.0f));
    color = maybeDecodeSRGBToLinear(color, camera.inputIsSRGB);

    int4 unionBounds;
    if (leftResult.visible && rightResult.visible) {
        unionBounds = int4(
            min(leftResult.tileBounds.x, rightResult.tileBounds.x),
            max(leftResult.tileBounds.y, rightResult.tileBounds.y),
            min(leftResult.tileBounds.z, rightResult.tileBounds.z),
            max(leftResult.tileBounds.w, rightResult.tileBounds.w)
        );
    } else if (leftResult.visible) {
        unionBounds = leftResult.tileBounds;
    } else {
        unionBounds = rightResult.tileBounds;
    }

    int unionTilesX = max(unionBounds.y - unionBounds.x + 1, 0);
    int unionTilesY = max(unionBounds.w - unionBounds.z + 1, 0);
    uint touchedCount = uint(unionTilesX * unionTilesY);

    if (touchedCount == 0) {
        nTouchedTiles[gid] = 0;
        outBounds[gid] = int4(0, -1, 0, -1);
        preDepthKeys[gid] = 0xFFFFFFFFu;
        return;
    }

    StereoTiledRenderData renderData;
    if (leftResult.visible) {
        renderData.leftMeanX = half(leftResult.screenPos.x);
        renderData.leftMeanY = half(leftResult.screenPos.y);
        float3 conicLF = conicFromThetaSigmas(leftResult.theta, leftResult.sigma1, leftResult.sigma2);
        renderData.leftCxx = half(conicLF.x);
        renderData.leftCyy = half(conicLF.z);
        renderData.leftCxy2 = half(2.0f * conicLF.y);
        renderData.leftDepth = half(leftResult.depth);
    } else {
        renderData.leftMeanX = half(-1e10f);
        renderData.leftMeanY = half(-1e10f);
        renderData.leftCxx = half(0);
        renderData.leftCyy = half(0);
        renderData.leftCxy2 = half(0);
        renderData.leftDepth = half(0);
    }
    if (rightResult.visible) {
        renderData.rightMeanX = half(rightResult.screenPos.x);
        renderData.rightMeanY = half(rightResult.screenPos.y);
        float3 conicRF = conicFromThetaSigmas(rightResult.theta, rightResult.sigma1, rightResult.sigma2);
        renderData.rightCxx = half(conicRF.x);
        renderData.rightCyy = half(conicRF.z);
        renderData.rightCxy2 = half(2.0f * conicRF.y);
        renderData.rightDepth = half(rightResult.depth);
    } else {
        renderData.rightMeanX = half(-1e10f);
        renderData.rightMeanY = half(-1e10f);
        renderData.rightCxx = half(0);
        renderData.rightCyy = half(0);
        renderData.rightCxy2 = half(0);
        renderData.rightDepth = half(0);
    }

    renderData.colorR = uchar(clamp(color.x * 255.0f, 0.0f, 255.0f));
    renderData.colorG = uchar(clamp(color.y * 255.0f, 0.0f, 255.0f));
    renderData.colorB = uchar(clamp(color.z * 255.0f, 0.0f, 255.0f));
    renderData.opacity = uchar(clamp(opacity * 255.0f, 0.0f, 255.0f));
    renderData.centerDepth = half(checkDepth);
    renderData._pad0 = 0;

    outRenderData[gid] = renderData;
    outBounds[gid] = unionBounds;
    nTouchedTiles[gid] = touchedCount;

    preDepthKeys[gid] = float_to_sortable_uint(checkDepth);

    atomic_fetch_add_explicit(totalInstances, touchedCount, memory_order_relaxed);
}

template [[host_name("depthFirstStereoUnifiedPreprocessKernel")]]
kernel void depthFirstStereoUnifiedPreprocessKernel<PackedWorldGaussian, float>(
    const device PackedWorldGaussian*, const device float*,
    device StereoTiledRenderData*, device int4*, device uint*,
    device uint*, device atomic_uint*,
    constant StereoCameraUniforms&, constant TileBinningParams&, uint);

template [[host_name("depthFirstStereoUnifiedPreprocessKernelHalf")]]
kernel void depthFirstStereoUnifiedPreprocessKernel<PackedWorldGaussianHalf, half>(
    const device PackedWorldGaussianHalf*, const device half*,
    device StereoTiledRenderData*, device int4*, device uint*,
    device uint*, device atomic_uint*,
    constant StereoCameraUniforms&, constant TileBinningParams&, uint);


#define DF_SCAN_BLOCK_SIZE 256u

kernel void visibilityBlockReduceKernel(
    const device uint* nTouchedTiles [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]]
) {
    threadgroup uint shared[DF_SCAN_BLOCK_SIZE];
    uint value = (gid < count && nTouchedTiles[gid] > 0u) ? 1u : 0u;
    uint blockSum = threadgroup_cooperative_reduce_sum<DF_SCAN_BLOCK_SIZE>(value, shared, ushort(lid));
    if (lid == 0u) { blockSums[groupId] = blockSum; }
}

kernel void scanBlockReduceKernel(
    const device uint* input [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]]
) {
    threadgroup uint shared[DF_SCAN_BLOCK_SIZE];
    uint value = (gid < count) ? input[gid] : 0u;
    uint blockSum = threadgroup_cooperative_reduce_sum<DF_SCAN_BLOCK_SIZE>(value, shared, ushort(lid));
    if (lid == 0u) { blockSums[groupId] = blockSum; }
}

kernel void scanBlockScanKernel(
    const device uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    const device uint* blockOffsets [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]]
) {
    threadgroup uint shared[DF_SCAN_BLOCK_SIZE];
    uint value = (gid < count) ? input[gid] : 0u;
    uint exclusive = threadgroup_raking_prefix_exclusive_sum<DF_SCAN_BLOCK_SIZE>(value, shared, ushort(lid));
    if (gid < count) { output[gid] = exclusive + blockOffsets[groupId]; }
}

kernel void visibilityBlockScanKernel(
    const device uint* nTouchedTiles [[buffer(0)]],
    device uint* outputOffsets [[buffer(1)]],
    const device uint* blockOffsets [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]]
) {
    threadgroup uint shared[DF_SCAN_BLOCK_SIZE];
    uint value = (gid < count && nTouchedTiles[gid] > 0u) ? 1u : 0u;
    uint exclusive = threadgroup_raking_prefix_exclusive_sum<DF_SCAN_BLOCK_SIZE>(value, shared, ushort(lid));
    if (gid < count) { outputOffsets[gid] = exclusive + blockOffsets[groupId]; }
}

kernel void singleBlockScanKernel(
    device uint* blockSums [[buffer(0)]],
    constant uint& blockCount [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup uint shared[DF_SCAN_BLOCK_SIZE];
    uint value = (lid < blockCount) ? blockSums[lid] : 0u;
    uint total = threadgroup_cooperative_reduce_sum<DF_SCAN_BLOCK_SIZE>(value, shared, ushort(lid));
    uint exclusive = threadgroup_raking_prefix_exclusive_sum<DF_SCAN_BLOCK_SIZE>(value, shared, ushort(lid));
    if (lid < blockCount) { blockSums[lid] = exclusive; }
    if (lid == 0u) { blockSums[blockCount] = total; }
}

kernel void visibilityScatterCompactKernel(
    const device uint* nTouchedTiles [[buffer(0)]],
    const device uint* prefixOffsets [[buffer(1)]],
    const device uint* preDepthKeys [[buffer(2)]],
    device uint* depthKeys [[buffer(3)]],
    device int* primitiveIndices [[buffer(4)]],
    device uint* visibleCount [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    constant uint& maxOutCount [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) { return; }

    bool isVisible = (nTouchedTiles[gid] > 0u);
    if (isVisible) {
        uint outIdx = prefixOffsets[gid];
        if (outIdx < maxOutCount) {
            uint key = preDepthKeys[gid];
            if (DF_DEPTH_KEY_16) {
                float depth = sortable_uint_to_float(key);
                half depthHalf = half(depth);
                ushort depthBits = as_type<ushort>(depthHalf) ^ 0x8000u;
                key = uint(depthBits);
            }
            depthKeys[outIdx] = key;
            primitiveIndices[outIdx] = int(gid);
        }
    }

    if (gid == count - 1u) {
        visibleCount[0] = prefixOffsets[gid] + (isVisible ? 1u : 0u);
    }
}

kernel void applyDepthOrderingKernel(
    const device int* sortedPrimitiveIndices [[buffer(0)]],
    const device uint* nTouchedTiles [[buffer(1)]],
    device uint* orderedTileCounts [[buffer(2)]],
    constant DepthFirstHeader& header [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint visibleCount = header.visibleCount;
    if (gid >= visibleCount) { return; }

    int originalIdx = sortedPrimitiveIndices[gid];
    if (originalIdx < 0) {
        orderedTileCounts[gid] = 0;
        return;
    }

    orderedTileCounts[gid] = nTouchedTiles[originalIdx];
}

kernel void createInstancesKernel(
    const device int* sortedPrimitiveIndices [[buffer(0)]],
    const device uint* instanceOffsets [[buffer(1)]],
    const device int4* tileBounds [[buffer(2)]],
    const device GaussianRenderData* renderData [[buffer(3)]],
    device ushort* instanceTileIds [[buffer(4)]],  // ushort is enough for tiles (max 65535)
    device int* instanceGaussianIndices [[buffer(5)]],
    constant DepthFirstParams& params [[buffer(6)]],
    constant DepthFirstHeader& header [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint visibleCount = header.visibleCount;
    if (gid >= visibleCount) { return; }

    int originalIdx = sortedPrimitiveIndices[gid];
    if (originalIdx < 0) { return; }

    int4 bounds = tileBounds[originalIdx];
    int minTX = bounds.x;
    int maxTX = bounds.y;
    int minTY = bounds.z;
    int maxTY = bounds.w;

    if (minTX > maxTX || minTY > maxTY) { return; }

    // Read quantized gaussian data (same values used in preprocess counting)
    GaussianRenderData g = renderData[originalIdx];
    float meanX_q = float(g.meanX);
    float meanY_q = float(g.meanY);
    float theta_q = unpackThetaPi(g.theta);
    float sigma1_q = float(g.sigma1);
    float sigma2_q = float(g.sigma2);
    float opacity_q = float(g.opacity) * (1.0f / 255.0f);

    // Build conic from quantized ellipse params
    float3 conic = conicFromSigmaTheta(sigma1_q, sigma2_q, theta_q);
    float conic_a = conic.x;
    float conic_b = conic.y;
    float conic_c = conic.z;

    // Compute d² cutoff: if minQuadRect <= d2Cutoff, tile passes
    float tau = max(params.alphaThreshold, 1e-12f);
    float d2Cutoff = computeD2Cutoff(opacity_q, tau);

    // Get the write offset for this gaussian (from prefix sum)
    uint writeOffset = instanceOffsets[gid];

    float tileW = float(params.tileWidth);
    float tileH = float(params.tileHeight);

    if (d2Cutoff >= 0.0f) {
        for (int ty = minTY; ty <= maxTY; ty++) {
            float tileMinY = float(ty) * tileH;
            float tileMaxY = tileMinY + tileH;
            float tile_ymin = tileMinY - meanY_q;
            float tile_ymax = tileMaxY - meanY_q;

            for (int tx = minTX; tx <= maxTX; tx++) {
                float tileMinX = float(tx) * tileW;
                float tileMaxX = tileMinX + tileW;
                float tile_xmin = tileMinX - meanX_q;
                float tile_xmax = tileMaxX - meanX_q;

                float d2min = minQuadRect(tile_xmin, tile_xmax, tile_ymin, tile_ymax,
                                          conic_a, conic_b, conic_c);
                if (d2min <= d2Cutoff && writeOffset < params.maxAssignments) {
                    ushort tileId = ushort(ty * int(params.tilesX) + tx);
                    instanceTileIds[writeOffset] = tileId;
                    instanceGaussianIndices[writeOffset] = originalIdx;
                    writeOffset++;
                }
            }
        }
    }
}

kernel void createInstancesStereoKernel(
    const device int* sortedPrimitiveIndices [[buffer(0)]],
    const device uint* instanceOffsets [[buffer(1)]],
    const device int4* tileBounds [[buffer(2)]],
    device ushort* instanceTileIds [[buffer(3)]],
    device int* instanceGaussianIndices [[buffer(4)]],
    constant DepthFirstParams& params [[buffer(5)]],
    constant DepthFirstHeader& header [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint visibleCount = header.visibleCount;
    if (gid >= visibleCount) { return; }

    int originalIdx = sortedPrimitiveIndices[gid];
    if (originalIdx < 0) { return; }

    int4 bounds = tileBounds[originalIdx];
    int minTX = bounds.x;
    int maxTX = bounds.y;
    int minTY = bounds.z;
    int maxTY = bounds.w;

    if (minTX > maxTX || minTY > maxTY) { return; }

    uint writeOffset = instanceOffsets[gid];

    for (int ty = minTY; ty <= maxTY; ty++) {
        for (int tx = minTX; tx <= maxTX; tx++) {
            if (writeOffset < params.maxAssignments) {
                ushort tileId = ushort(ty * int(params.tilesX) + tx);
                instanceTileIds[writeOffset] = tileId;
                instanceGaussianIndices[writeOffset] = originalIdx;
                writeOffset++;
            }
        }
    }
}

kernel void tileRadixHistogramKernel(
    device const ushort* inputTileIds [[buffer(0)]],
    device uint* histFlat [[buffer(1)]],
    constant DepthFirstHeader& header [[buffer(2)]],
    constant uint& currentDigit [[buffer(3)]],
    uint gridSize [[threadgroups_per_grid]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint elementsPerBlock = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint baseId = groupId * elementsPerBlock;

    uint paddedCount = header.paddedInstanceCount;
    uint count = header.totalInstances;

    uint bufferRemaining = (baseId < paddedCount) ? (paddedCount - baseId) : 0;
    uint assignmentsRemain = (baseId < count) ? (count - baseId) : 0;
    uint available = min(bufferRemaining, assignmentsRemain);

    threadgroup atomic_uint localHist[RADIX_BUCKETS];

    for (uint i = 0; i < (RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE; ++i) {
        uint bin = localId + i * RADIX_BLOCK_SIZE;
        if (bin < RADIX_BUCKETS) {
            atomic_store_explicit(&localHist[bin], 0u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ushort keys[RADIX_GRAIN_SIZE];
    ushort sentinelKey = 0xFFFFu;

    load_blocked_local_from_global<RADIX_GRAIN_SIZE>(
        keys,
        &inputTileIds[baseId],
        localId,
        available,
        sentinelKey);

    for (ushort i = 0; i < RADIX_GRAIN_SIZE; ++i) {
        uint offsetInBlock = localId * RADIX_GRAIN_SIZE + i;
        uint globalIdx = baseId + offsetInBlock;

        if (globalIdx < count && keys[i] != sentinelKey) {
            uchar bin = (uchar)((keys[i] >> currentDigit) & 0xFFu);
            atomic_fetch_add_explicit(&localHist[bin], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < (RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE; ++i) {
        uint bin = localId + i * RADIX_BLOCK_SIZE;
        if (bin < RADIX_BUCKETS) {
            histFlat[bin * gridSize + groupId] = atomic_load_explicit(&localHist[bin], memory_order_relaxed);
        }
    }
}

kernel void tileRadixScanBlocksKernel(
    device uint* histFlat [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant DepthFirstHeader& header [[buffer(2)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    uint paddedCount = header.paddedInstanceCount;
    const uint valuesPerGroup = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint gridSize = (paddedCount + valuesPerGroup - 1u) / valuesPerGroup;
    uint histLength = gridSize * RADIX_BUCKETS;

    uint baseIndex = groupId * RADIX_BLOCK_SIZE;
    threadgroup uint shared[RADIX_BLOCK_SIZE];

    uint value = (baseIndex + localId < histLength) ? histFlat[baseIndex + localId] : 0u;
    uint scanned = threadgroup_raking_prefix_exclusive_sum<RADIX_BLOCK_SIZE>(value, shared, localId);

    if (baseIndex + localId < histLength) {
        histFlat[baseIndex + localId] = scanned;
    }
    if (localId == RADIX_BLOCK_SIZE - 1) {
        blockSums[groupId] = scanned + value;
    }
}

// Tile exclusive scan kernel (handles up to 16384 block sums = ~16M instances)
#define TILE_RADIX_SCAN_GRAIN 64

kernel void tileRadixExclusiveScanKernel(
    device uint* blockSums [[buffer(0)]],
    device uint* scannedHist [[buffer(1)]],
    device const uint* histFlat [[buffer(2)]],
    constant DepthFirstHeader& header [[buffer(3)]],
    ushort localId [[thread_position_in_threadgroup]],
    threadgroup uint* shared [[threadgroup(0)]]
) {
    uint paddedCount = header.paddedInstanceCount;
    const uint valuesPerGroup = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint gridSize = (paddedCount + valuesPerGroup - 1u) / valuesPerGroup;
    uint numBlockSums = (gridSize * RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE;

    uint elemsPerThread = (numBlockSums + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE;
    elemsPerThread = min(elemsPerThread, (uint)TILE_RADIX_SCAN_GRAIN);

    uint localVals[TILE_RADIX_SCAN_GRAIN];
    uint localSum = 0u;
    uint threadBase = localId * elemsPerThread;

    for (uint i = 0u; i < elemsPerThread; ++i) {
        uint idx = threadBase + i;
        uint val = (idx < numBlockSums) ? blockSums[idx] : 0u;
        localVals[i] = val;
        localSum += val;
    }

    uint scannedSum = radix_threadgroup_prefix_scan<RADIX_SCAN_TYPE_EXCLUSIVE>(localSum, shared, localId, radix_sum_op<uint>());

    uint running = scannedSum;
    for (uint i = 0u; i < elemsPerThread; ++i) {
        uint idx = threadBase + i;
        if (idx < numBlockSums) {
            blockSums[idx] = running;
            running += localVals[i];
        }
    }
}

kernel void tileRadixApplyOffsetsKernel(
    device const uint* histFlat [[buffer(0)]],
    device const uint* blockSums [[buffer(1)]],
    device uint* scannedHistFlat [[buffer(2)]],
    constant DepthFirstHeader& header [[buffer(3)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    uint paddedCount = header.paddedInstanceCount;
    const uint valuesPerGroup = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint gridSize = (paddedCount + valuesPerGroup - 1u) / valuesPerGroup;
    uint histLength = gridSize * RADIX_BUCKETS;

    uint baseIndex = groupId * RADIX_BLOCK_SIZE;
    uint blockOffset = blockSums[groupId];

    if (baseIndex + localId < histLength) {
        scannedHistFlat[baseIndex + localId] = histFlat[baseIndex + localId] + blockOffset;
    }
}

kernel void tileRadixScatterKernel(
    device const ushort* inputTileIds [[buffer(0)]],
    device const int* inputIndices [[buffer(1)]],
    device const uint* offsetsFlat [[buffer(2)]],
    device ushort* outputTileIds [[buffer(3)]],
    device int* outputIndices [[buffer(4)]],
    constant DepthFirstHeader& header [[buffer(5)]],
    constant uint& currentDigit [[buffer(6)]],  // 0 or 8 for 16-bit keys
    uint groupId [[threadgroup_position_in_grid]],
    uint gridSize [[threadgroups_per_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint elementsPerBlock = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint baseId = groupId * elementsPerBlock;

    uint paddedCount = header.paddedInstanceCount;
    uint count = header.totalInstances;

    uint bufferRemaining = (baseId < paddedCount) ? (paddedCount - baseId) : 0;
    uint assignmentsRemain = (baseId < count) ? (count - baseId) : 0;
    uint available = min(bufferRemaining, assignmentsRemain);

    uint keySentinel = 0xFFFFu;  // 16-bit sentinel

    uint keys[RADIX_GRAIN_SIZE];
    int payloads[RADIX_GRAIN_SIZE];

    for (ushort i = 0; i < RADIX_GRAIN_SIZE; ++i) {
        uint idx = baseId + localId + i * RADIX_BLOCK_SIZE;
        if (idx < available + baseId && idx < count) {
            keys[i] = uint(inputTileIds[idx]);
            payloads[i] = inputIndices[idx];
        } else {
            keys[i] = keySentinel;
            payloads[i] = -1;
        }
    }

    threadgroup uint globalBinBase[RADIX_BUCKETS];
    threadgroup RadixKeyPayload tgKp[RADIX_BLOCK_SIZE];
    threadgroup ushort tgShort[RADIX_BLOCK_SIZE];

    for (uint i = 0; i < (RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE; ++i) {
        uint bin = localId + i * RADIX_BLOCK_SIZE;
        if (bin < RADIX_BUCKETS) {
            globalBinBase[bin] = offsetsFlat[bin * gridSize + groupId];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

	    for (ushort chunk = 0; chunk < RADIX_GRAIN_SIZE; ++chunk) {
	        RadixKeyPayload kp;
	        kp.key = keys[chunk];
	        kp.payload = uint(payloads[chunk]);

        ushort digitIndex = (ushort)(currentDigit / 8);
        kp = radix_partial_sort(kp, tgKp, localId, digitIndex);

	        ushort myBin = (ushort)((kp.key >> currentDigit) & 0xFFu);

	        bool isValid = (kp.key != keySentinel);

	        // Invalid lanes must not participate in run detection.
	        // Use a bin value that can't match any valid bin (0..255) so adjacent valid lanes
	        // next to invalid lanes still correctly see discontinuities.
	        ushort headBin = isValid ? myBin : (ushort)0xFFFF;
	        uchar headFlag = flag_head_discontinuity<RADIX_BLOCK_SIZE>(headBin, tgShort, localId);
	        if (!isValid) headFlag = 0;

        ushort runStart = radix_threadgroup_prefix_scan<RADIX_SCAN_TYPE_INCLUSIVE>(
            headFlag ? localId : (ushort)0, tgShort, localId, radix_max_op<ushort>());
        ushort localOffset = localId - runStart;

	        ushort tailBin = isValid ? myBin : (ushort)0xFFFF;
	        uchar tailFlag = flag_tail_discontinuity<RADIX_BLOCK_SIZE>(tailBin, tgShort, localId);
	        if (!isValid) tailFlag = 0;

	        if (isValid) {
	            uint dst = globalBinBase[myBin] + localOffset;
	            if (dst < count) {
	                outputTileIds[dst] = ushort(kp.key);
	                outputIndices[dst] = int(kp.payload);
	            }
	        }

	        // Prevent any SIMD-group from updating the shared bin base while others are still
	        // reading it to compute `dst` for this chunk.
	        threadgroup_barrier(mem_flags::mem_threadgroup);

	        if (tailFlag && isValid) {
	            globalBinBase[myBin] += localOffset + 1;
	        }
	        threadgroup_barrier(mem_flags::mem_threadgroup);
	    }
	}

kernel void extractTileRangesKernel(
    const device ushort* sortedTileIds [[buffer(0)]],  // 16-bit tile IDs
    device GaussianHeader* tileHeaders [[buffer(1)]],
    device uint* activeTiles [[buffer(2)]],
    device atomic_uint* activeTileCount [[buffer(3)]],
    constant DepthFirstHeader& dfHeader [[buffer(4)]],
    constant uint& tileCount [[buffer(5)]],
    uint tile [[thread_position_in_grid]]
) {
    if (tile >= tileCount) { return; }

    uint totalInstances = dfHeader.totalInstances;
    if (totalInstances == 0) {
        GaussianHeader emptyHeader;
        emptyHeader.offset = 0;
        emptyHeader.count = 0;
        tileHeaders[tile] = emptyHeader;
        return;
    }

    uint left = 0;
    uint right = totalInstances;
    while (left < right) {
        uint mid = (left + right) >> 1;
        uint midTile = uint(sortedTileIds[mid]);
        if (midTile < tile) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    uint start = left;

    left = start;
    right = totalInstances;
    while (left < right) {
        uint mid = (left + right) >> 1;
        uint midTile = uint(sortedTileIds[mid]);
        if (midTile <= tile) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    uint end = left;

    GaussianHeader tileHeader;
    tileHeader.offset = start;
    tileHeader.count = end > start ? (end - start) : 0;
    tileHeaders[tile] = tileHeader;

    if (tileHeader.count > 0) {
        uint index = atomic_fetch_add_explicit(activeTileCount, 1u, memory_order_relaxed);
        activeTiles[index] = tile;
    }
}

kernel void resetDepthFirstStateKernel(
    device DepthFirstHeader* header [[buffer(0)]],
    device uint* activeTileCount [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        header->visibleCount = 0;
        header->totalInstances = 0;
        header->paddedVisibleCount = 0;
        header->paddedInstanceCount = 0;
        header->overflow = 0;
        *activeTileCount = 0;
    }
}

template <typename KeyT>
void depthSortHistogramImpl(
    device const KeyT* inputKeys,
    device uint* histFlat,
    threadgroup atomic_uint* localHist,
    uint currentDigit,
    uint paddedCount,
    uint count,
    uint gridSize,
    uint groupId,
    ushort localId,
    KeyT sentinelKey
) {
    const uint elementsPerBlock = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint baseId = groupId * elementsPerBlock;

    uint bufferRemaining = (baseId < paddedCount) ? (paddedCount - baseId) : 0;
    uint assignmentsRemain = (baseId < count) ? (count - baseId) : 0;
    uint available = min(bufferRemaining, assignmentsRemain);

    for (uint i = 0; i < (RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE; ++i) {
        uint bin = localId + i * RADIX_BLOCK_SIZE;
        if (bin < RADIX_BUCKETS) { atomic_store_explicit(&localHist[bin], 0u, memory_order_relaxed); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    KeyT keys[RADIX_GRAIN_SIZE];
    load_blocked_local_from_global<RADIX_GRAIN_SIZE>(keys, &inputKeys[baseId], localId, available, sentinelKey);

    for (ushort i = 0; i < RADIX_GRAIN_SIZE; ++i) {
        uint offsetInBlock = localId * RADIX_GRAIN_SIZE + i;
        uint globalIdx = baseId + offsetInBlock;

        if (globalIdx < count && keys[i] != sentinelKey) {
            uchar bin = (uchar)value_to_key_at_digit_t(keys[i], (ushort)currentDigit);
            atomic_fetch_add_explicit(&localHist[bin], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < (RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE; ++i) {
        uint bin = localId + i * RADIX_BLOCK_SIZE;
        if (bin < RADIX_BUCKETS) { histFlat[bin * gridSize + groupId] = atomic_load_explicit(&localHist[bin], memory_order_relaxed); }
    }
}

kernel void depthSortHistogramKernel16(
    device const ushort* inputKeys [[buffer(0)]],
    device uint* histFlat [[buffer(1)]],
    constant uint& currentDigit [[buffer(3)]],
    constant DepthFirstHeader& header [[buffer(4)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint valuesPerGroup = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint gridSize = (header.paddedVisibleCount + valuesPerGroup - 1u) / valuesPerGroup;

    if (groupId >= gridSize) return;

    threadgroup atomic_uint localHist[RADIX_BUCKETS];
    depthSortHistogramImpl<ushort>(
        inputKeys, histFlat, localHist, currentDigit,
        header.paddedVisibleCount, header.visibleCount,
        gridSize, groupId, localId, (ushort)0xFFFFu
    );
}

kernel void depthSortHistogramKernel32(
    device const uint* inputKeys [[buffer(0)]],
    device uint* histFlat [[buffer(1)]],
    constant uint& currentDigit [[buffer(3)]],
    constant DepthFirstHeader& header [[buffer(4)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint valuesPerGroup = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint gridSize = (header.paddedVisibleCount + valuesPerGroup - 1u) / valuesPerGroup;

    if (groupId >= gridSize) return;

    threadgroup atomic_uint localHist[RADIX_BUCKETS];
    depthSortHistogramImpl<uint>(
        inputKeys, histFlat, localHist, currentDigit,
        header.paddedVisibleCount, header.visibleCount,
        gridSize, groupId, localId, 0xFFFFFFFFu
    );
}

kernel void depthSortScanBlocksKernel(
    device uint* histFlat [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant DepthFirstHeader& header [[buffer(2)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    uint paddedCount = header.paddedVisibleCount;  // For depth sort, use padded visible count
    const uint valuesPerGroup = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint gridSize = (paddedCount + valuesPerGroup - 1u) / valuesPerGroup;
    uint histLength = gridSize * RADIX_BUCKETS;
    uint numBlocks = (histLength + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE;

    if (groupId >= numBlocks) return;

    uint baseIndex = groupId * RADIX_BLOCK_SIZE;
    threadgroup uint shared[RADIX_BLOCK_SIZE];

    uint value = (baseIndex + localId < histLength) ? histFlat[baseIndex + localId] : 0u;
    uint scanned = threadgroup_raking_prefix_exclusive_sum<RADIX_BLOCK_SIZE>(value, shared, localId);

    if (baseIndex + localId < histLength) {
        histFlat[baseIndex + localId] = scanned;
    }
    if (localId == RADIX_BLOCK_SIZE - 1) {
        blockSums[groupId] = scanned + value;
    }
}

#define RADIX_SCAN_GRAIN 32
kernel void depthSortExclusiveScanKernel(
    device uint* blockSums [[buffer(0)]],
    constant DepthFirstHeader& header [[buffer(1)]],
    ushort localId [[thread_position_in_threadgroup]],
    threadgroup uint* shared [[threadgroup(0)]]
) {
    uint paddedCount = header.paddedVisibleCount;  // For depth sort
    const uint valuesPerGroup = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint gridSize = (paddedCount + valuesPerGroup - 1u) / valuesPerGroup;
    uint numBlockSums = (gridSize * RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE;

    uint elemsPerThread = (numBlockSums + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE;
    elemsPerThread = min(elemsPerThread, (uint)RADIX_SCAN_GRAIN);

    uint localVals[RADIX_SCAN_GRAIN];
    uint localSum = 0u;
    uint threadBase = localId * elemsPerThread;

    for (uint i = 0u; i < elemsPerThread; ++i) {
        uint idx = threadBase + i;
        uint val = (idx < numBlockSums) ? blockSums[idx] : 0u;
        localVals[i] = val;
        localSum += val;
    }

    uint scannedSum = radix_threadgroup_prefix_scan<RADIX_SCAN_TYPE_EXCLUSIVE>(localSum, shared, localId, radix_sum_op<uint>());

    uint running = scannedSum;
    for (uint i = 0u; i < elemsPerThread; ++i) {
        uint idx = threadBase + i;
        if (idx < numBlockSums) {
            blockSums[idx] = running;
            running += localVals[i];
        }
    }
}

kernel void depthSortApplyOffsetsKernel(
    device const uint* histFlat [[buffer(0)]],
    device const uint* blockSums [[buffer(1)]],
    device uint* scannedHistFlat [[buffer(2)]],
    constant DepthFirstHeader& header [[buffer(3)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]],
    threadgroup uint* shared [[threadgroup(0)]]
) {
    uint paddedCount = header.paddedVisibleCount;  // For depth sort
    const uint valuesPerGroup = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint gridSize = (paddedCount + valuesPerGroup - 1u) / valuesPerGroup;
    uint histLength = gridSize * RADIX_BUCKETS;
    uint numBlocks = (histLength + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE;

    if (groupId >= numBlocks) return;

    uint baseIndex = groupId * RADIX_BLOCK_SIZE;
    uint blockOffset = blockSums[groupId];

    if (baseIndex + localId < histLength) {
        scannedHistFlat[baseIndex + localId] = histFlat[baseIndex + localId] + blockOffset;
    }
}

template <typename KeyT, typename KpT>
void depthSortScatterImpl(
    device KeyT* outputKeys,
    device const KeyT* inputKeys,
    device int* outputPayload,
    device const int* inputPayload,
    device const uint* offsetsFlat,
    threadgroup uint* globalBinBase,
    threadgroup KpT* tgKp,
    threadgroup ushort* tgShort,
    uint currentDigit,
    uint paddedCount,
    uint count,
    uint groupId,
    uint gridSize,
    ushort localId,
    KeyT keySentinel
) {
    const uint elementsPerBlock = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint baseId = groupId * elementsPerBlock;

    uint bufferRemaining = (baseId < paddedCount) ? (paddedCount - baseId) : 0;
    uint assignmentsRemain = (baseId < count) ? (count - baseId) : 0;
    uint available = min(bufferRemaining, assignmentsRemain);

    KeyT keys[RADIX_GRAIN_SIZE];
    int payloads[RADIX_GRAIN_SIZE];
    load_striped_local_from_global<RADIX_GRAIN_SIZE, RADIX_BLOCK_SIZE>(keys, &inputKeys[baseId], localId, available, keySentinel);
    load_striped_local_from_global<RADIX_GRAIN_SIZE, RADIX_BLOCK_SIZE>(payloads, &inputPayload[baseId], localId, available, -1);

    for (uint i = 0; i < (RADIX_BUCKETS + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE; ++i) {
        uint bin = localId + i * RADIX_BLOCK_SIZE;
        if (bin < RADIX_BUCKETS) { globalBinBase[bin] = offsetsFlat[bin * gridSize + groupId]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort chunk = 0; chunk < RADIX_GRAIN_SIZE; ++chunk) {
        KpT kp;
        kp.key = keys[chunk];
        kp.payload = uint(payloads[chunk]);

        ushort myBin = value_to_key_at_digit_t(kp.key, (ushort)currentDigit);
        kp = radix_partial_sort_t<KeyT>(kp, tgKp, localId, (ushort)currentDigit);
        myBin = value_to_key_at_digit_t(kp.key, (ushort)currentDigit);

        bool isValid = (kp.key != keySentinel);

        ushort headBin = isValid ? myBin : (ushort)0xFFFF;
        uchar headFlag = flag_head_discontinuity<RADIX_BLOCK_SIZE>(headBin, tgShort, localId);
        if (!isValid) headFlag = 0;

        ushort runStart = radix_threadgroup_prefix_scan<RADIX_SCAN_TYPE_INCLUSIVE>(
            headFlag ? localId : (ushort)0, tgShort, localId, radix_max_op<ushort>());
        ushort localOffset = localId - runStart;

        ushort tailBin = isValid ? myBin : (ushort)0xFFFF;
        uchar tailFlag = flag_tail_discontinuity<RADIX_BLOCK_SIZE>(tailBin, tgShort, localId);
        if (!isValid) tailFlag = 0;

        if (isValid) {
            uint dst = globalBinBase[myBin] + localOffset;
            if (dst < count) {
                outputKeys[dst] = kp.key;
                outputPayload[dst] = int(kp.payload);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tailFlag && isValid) { globalBinBase[myBin] += localOffset + 1; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// 16-bit key scatter kernel
kernel void depthSortScatterKernel16(
    device ushort* outputKeys [[buffer(0)]],
    device const ushort* inputKeys [[buffer(1)]],
    device int* outputPayload [[buffer(2)]],
    device const int* inputPayload [[buffer(3)]],
    device const uint* offsetsFlat [[buffer(5)]],
    constant uint& currentDigit [[buffer(6)]],
    constant DepthFirstHeader& header [[buffer(7)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint valuesPerGroup = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint gridSize = (header.paddedVisibleCount + valuesPerGroup - 1u) / valuesPerGroup;

    if (groupId >= gridSize) return;

    threadgroup uint globalBinBase[RADIX_BUCKETS];
    threadgroup RadixKeyPayloadT<ushort> tgKp[RADIX_BLOCK_SIZE];
    threadgroup ushort tgShort[RADIX_BLOCK_SIZE];

    depthSortScatterImpl<ushort, RadixKeyPayloadT<ushort>>(
        outputKeys, inputKeys, outputPayload, inputPayload, offsetsFlat,
        globalBinBase, tgKp, tgShort,
        currentDigit, header.paddedVisibleCount, header.visibleCount,
        groupId, gridSize, localId, (ushort)0xFFFFu
    );
}

kernel void depthSortScatterKernel32(
    device uint* outputKeys [[buffer(0)]],
    device const uint* inputKeys [[buffer(1)]],
    device int* outputPayload [[buffer(2)]],
    device const int* inputPayload [[buffer(3)]],
    device const uint* offsetsFlat [[buffer(5)]],
    constant uint& currentDigit [[buffer(6)]],
    constant DepthFirstHeader& header [[buffer(7)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint valuesPerGroup = RADIX_BLOCK_SIZE * RADIX_GRAIN_SIZE;
    uint gridSize = (header.paddedVisibleCount + valuesPerGroup - 1u) / valuesPerGroup;

    if (groupId >= gridSize) return;

    threadgroup uint globalBinBase[RADIX_BUCKETS];
    threadgroup RadixKeyPayloadT<uint> tgKp[RADIX_BLOCK_SIZE];
    threadgroup ushort tgShort[RADIX_BLOCK_SIZE];

    depthSortScatterImpl<uint, RadixKeyPayloadT<uint>>(
        outputKeys, inputKeys, outputPayload, inputPayload, offsetsFlat,
        globalBinBase, tgKp, tgShort,
        currentDigit, header.paddedVisibleCount, header.visibleCount,
        groupId, gridSize, localId, 0xFFFFFFFFu
    );
}

#define RENDER_DF_TG_WIDTH 8
#define RENDER_DF_TG_HEIGHT 8
#define RENDER_DF_PIXELS_X 2
#define RENDER_DF_PIXELS_Y 2

kernel void depthFirstRender(
    const device GaussianHeader* headers [[buffer(0)]],
    const device GaussianRenderData* gaussians [[buffer(1)]],
    const device int* sortedGaussianIndices [[buffer(2)]],
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

    // 16x16 tile, 8x8 threads, 2x2 pixels per thread
    uint baseX = tileX * 16 + localId.x * RENDER_DF_PIXELS_X;
    uint baseY = tileY * 16 + localId.y * RENDER_DF_PIXELS_Y;

    half2 pos00 = half2(half(baseX + 0), half(baseY + 0));
    half2 pos10 = half2(half(baseX + 1), half(baseY + 0));
    half2 pos01 = half2(half(baseX + 0), half(baseY + 1));
    half2 pos11 = half2(half(baseX + 1), half(baseY + 1));

    half trans00 = 1.0h, trans10 = 1.0h;
    half trans01 = 1.0h, trans11 = 1.0h;

    half3 color00 = 0, color10 = 0;
    half3 color01 = 0, color11 = 0;

    half depth00 = 0, depth10 = 0;
    half depth01 = 0, depth11 = 0;

    uint start = hdr.offset;
    uint count = hdr.count;

    for (uint i = 0; i < count; i++) {
        half maxTrans = max(max(trans00, trans10), max(trans01, trans11));
        if (maxTrans < half(1.0h/255.0h)) break;

        int gaussianIdx = sortedGaussianIndices[start + i];
        if (gaussianIdx < 0) continue;
        GaussianRenderData g = gaussians[gaussianIdx];

        float theta = unpackThetaPi(g.theta);
        float sig1 = float(g.sigma1);
        float sig2 = float(g.sigma2);
        float3 conicF = conicFromThetaSigmas(theta, sig1, sig2);
        half cxx = half(conicF.x);
        half cyy = half(conicF.z);
        half cxy2 = half(2.0f * conicF.y);
        half opacity = getOpacity(g);
        half3 gColor = getColor(g);
        half gDepth = g.depth;

        half2 gMean = getMean(g);
        half2 d00 = pos00 - gMean;
        half2 d10 = pos10 - gMean;
        half2 d01 = pos01 - gMean;
        half2 d11 = pos11 - gMean;

        half p00 = d00.x*d00.x*cxx + d00.y*d00.y*cyy + d00.x*d00.y*cxy2;
        half p10 = d10.x*d10.x*cxx + d10.y*d10.y*cyy + d10.x*d10.y*cxy2;
        half p01 = d01.x*d01.x*cxx + d01.y*d01.y*cyy + d01.x*d01.y*cxy2;
        half p11 = d11.x*d11.x*cxx + d11.y*d11.y*cyy + d11.x*d11.y*cxy2;

        half a00 = min(opacity * exp(-0.5h * p00), 0.99h);
        half a10 = min(opacity * exp(-0.5h * p10), 0.99h);
        half a01 = min(opacity * exp(-0.5h * p01), 0.99h);
        half a11 = min(opacity * exp(-0.5h * p11), 0.99h);

        half4 alphas = half4(a00, a10, a01, a11);
        if (all(alphas == 0.0h)) continue;

        color00 += gColor * (a00 * trans00); depth00 += gDepth * (a00 * trans00);
        color10 += gColor * (a10 * trans10); depth10 += gDepth * (a10 * trans10);
        color01 += gColor * (a01 * trans01); depth01 += gDepth * (a01 * trans01);
        color11 += gColor * (a11 * trans11); depth11 += gDepth * (a11 * trans11);

        trans00 *= (1.0h - a00); trans10 *= (1.0h - a10);
        trans01 *= (1.0h - a01); trans11 *= (1.0h - a11);
    }

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
    if (baseX + 0 < w && baseY + 1 < h) {
        colorOut.write(half4(color01, 1.0h - trans01), uint2(baseX + 0, baseY + 1));
        if (hasDepth) depthOut.write(half4(depth01, 0, 0, 0), uint2(baseX + 0, baseY + 1));
    }
    if (baseX + 1 < w && baseY + 1 < h) {
        colorOut.write(half4(color11, 1.0h - trans11), uint2(baseX + 1, baseY + 1));
        if (hasDepth) depthOut.write(half4(depth11, 0, 0, 0), uint2(baseX + 1, baseY + 1));
    }
}

kernel void clearStereoRenderTextureKernel(
    texture2d_array<half, access::write> colorTex [[texture(0)]],
    constant ClearTextureParams& params [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= 2) {
        return;
    }
    half bg = 0.0h;
    colorTex.write(half4(bg, bg, bg, 1.0h), uint2(gid.xy), gid.z);
}

kernel void depthFirstStereoRender(
    const device GaussianHeader* headers [[buffer(0)]],
    const device StereoTiledRenderData* gaussians [[buffer(1)]],
    const device int* sortedGaussianIndices [[buffer(2)]],
    const device uint* activeTiles [[buffer(3)]],
    const device uint* activeTileCount [[buffer(4)]],
    texture2d_array<half, access::write> colorOut [[texture(0)]],
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

    // 16x16 tile, 8x8 threads, 2x2 pixels per thread
    uint baseX = tileX * 16 + localId.x * RENDER_DF_PIXELS_X;
    uint baseY = tileY * 16 + localId.y * RENDER_DF_PIXELS_Y;

    half2 pos00 = half2(half(baseX + 0), half(baseY + 0));
    half2 pos10 = half2(half(baseX + 1), half(baseY + 0));
    half2 pos01 = half2(half(baseX + 0), half(baseY + 1));
    half2 pos11 = half2(half(baseX + 1), half(baseY + 1));

    half transL00 = 1.0h, transL10 = 1.0h;
    half transL01 = 1.0h, transL11 = 1.0h;
    half transR00 = 1.0h, transR10 = 1.0h;
    half transR01 = 1.0h, transR11 = 1.0h;

    half3 colorL00 = 0, colorL10 = 0;
    half3 colorL01 = 0, colorL11 = 0;
    half3 colorR00 = 0, colorR10 = 0;
    half3 colorR01 = 0, colorR11 = 0;

    uint start = hdr.offset;
    uint count = hdr.count;

    for (uint i = 0; i < count; i++) {
        half maxTransL = max(max(transL00, transL10), max(transL01, transL11));
        half maxTransR = max(max(transR00, transR10), max(transR01, transR11));
        half maxTrans = max(maxTransL, maxTransR);
        if (maxTrans < half(1.0h/255.0h)) break;

        int gaussianIdx = sortedGaussianIndices[start + i];
        if (gaussianIdx < 0) continue;

        StereoTiledRenderData g = gaussians[gaussianIdx];

        half opacity = half(g.opacity) / 255.0h;
        half3 gColor = half3(half(g.colorR), half(g.colorG), half(g.colorB)) / 255.0h;

        if (maxTransL >= half(1.0h/255.0h)) {
            half2 gMeanL = half2(g.leftMeanX, g.leftMeanY);
            if (gMeanL.x >= -60000.0h) {
                half cxxL = g.leftCxx;
                half cyyL = g.leftCyy;
                half cxy2L = g.leftCxy2;

                half2 dL00 = pos00 - gMeanL;
                half2 dL10 = pos10 - gMeanL;
                half2 dL01 = pos01 - gMeanL;
                half2 dL11 = pos11 - gMeanL;

                half pL00 = dL00.x*dL00.x*cxxL + dL00.y*dL00.y*cyyL + dL00.x*dL00.y*cxy2L;
                half pL10 = dL10.x*dL10.x*cxxL + dL10.y*dL10.y*cyyL + dL10.x*dL10.y*cxy2L;
                half pL01 = dL01.x*dL01.x*cxxL + dL01.y*dL01.y*cyyL + dL01.x*dL01.y*cxy2L;
                half pL11 = dL11.x*dL11.x*cxxL + dL11.y*dL11.y*cyyL + dL11.x*dL11.y*cxy2L;

                const half r2Max = half(9.0f);
                half aL00 = 0.0h, aL10 = 0.0h, aL01 = 0.0h, aL11 = 0.0h;
                if (!(pL00 > r2Max && pL10 > r2Max && pL01 > r2Max && pL11 > r2Max)) {
                    aL00 = (pL00 > r2Max) ? 0.0h : min(opacity * exp(-0.5h * pL00), 0.99h);
                    aL10 = (pL10 > r2Max) ? 0.0h : min(opacity * exp(-0.5h * pL10), 0.99h);
                    aL01 = (pL01 > r2Max) ? 0.0h : min(opacity * exp(-0.5h * pL01), 0.99h);
                    aL11 = (pL11 > r2Max) ? 0.0h : min(opacity * exp(-0.5h * pL11), 0.99h);
                }

                half4 alphasL = half4(aL00, aL10, aL01, aL11);
                if (!all(alphasL == 0.0h)) {
                    colorL00 += gColor * (aL00 * transL00);
                    colorL10 += gColor * (aL10 * transL10);
                    colorL01 += gColor * (aL01 * transL01);
                    colorL11 += gColor * (aL11 * transL11);

                    transL00 *= (1.0h - aL00);
                    transL10 *= (1.0h - aL10);
                    transL01 *= (1.0h - aL01);
                    transL11 *= (1.0h - aL11);
                }
            }
        }

        if (maxTransR >= half(1.0h/255.0h)) {
            half2 gMeanR = half2(g.rightMeanX, g.rightMeanY);
            if (gMeanR.x >= -60000.0h) {
                half cxxR = g.rightCxx;
                half cyyR = g.rightCyy;
                half cxy2R = g.rightCxy2;

                half2 dR00 = pos00 - gMeanR;
                half2 dR10 = pos10 - gMeanR;
                half2 dR01 = pos01 - gMeanR;
                half2 dR11 = pos11 - gMeanR;

                half pR00 = dR00.x*dR00.x*cxxR + dR00.y*dR00.y*cyyR + dR00.x*dR00.y*cxy2R;
                half pR10 = dR10.x*dR10.x*cxxR + dR10.y*dR10.y*cyyR + dR10.x*dR10.y*cxy2R;
                half pR01 = dR01.x*dR01.x*cxxR + dR01.y*dR01.y*cyyR + dR01.x*dR01.y*cxy2R;
                half pR11 = dR11.x*dR11.x*cxxR + dR11.y*dR11.y*cyyR + dR11.x*dR11.y*cxy2R;

                const half r2Max = half(9.0f);
                half aR00 = 0.0h, aR10 = 0.0h, aR01 = 0.0h, aR11 = 0.0h;
                if (!(pR00 > r2Max && pR10 > r2Max && pR01 > r2Max && pR11 > r2Max)) {
                    aR00 = (pR00 > r2Max) ? 0.0h : min(opacity * exp(-0.5h * pR00), 0.99h);
                    aR10 = (pR10 > r2Max) ? 0.0h : min(opacity * exp(-0.5h * pR10), 0.99h);
                    aR01 = (pR01 > r2Max) ? 0.0h : min(opacity * exp(-0.5h * pR01), 0.99h);
                    aR11 = (pR11 > r2Max) ? 0.0h : min(opacity * exp(-0.5h * pR11), 0.99h);
                }

                half4 alphasR = half4(aR00, aR10, aR01, aR11);
                if (!all(alphasR == 0.0h)) {
                    colorR00 += gColor * (aR00 * transR00);
                    colorR10 += gColor * (aR10 * transR10);
                    colorR01 += gColor * (aR01 * transR01);
                    colorR11 += gColor * (aR11 * transR11);

                    transR00 *= (1.0h - aR00);
                    transR10 *= (1.0h - aR10);
                    transR01 *= (1.0h - aR01);
                    transR11 *= (1.0h - aR11);
                }
            }
        }
    }

    uint w = params.width, h = params.height;

    if (baseX + 0 < w && baseY + 0 < h) {
        colorOut.write(half4(colorL00, 1.0h - transL00), uint2(baseX + 0, baseY + 0), 0);
        colorOut.write(half4(colorR00, 1.0h - transR00), uint2(baseX + 0, baseY + 0), 1);
    }
    if (baseX + 1 < w && baseY + 0 < h) {
        colorOut.write(half4(colorL10, 1.0h - transL10), uint2(baseX + 1, baseY + 0), 0);
        colorOut.write(half4(colorR10, 1.0h - transR10), uint2(baseX + 1, baseY + 0), 1);
    }
    if (baseX + 0 < w && baseY + 1 < h) {
        colorOut.write(half4(colorL01, 1.0h - transL01), uint2(baseX + 0, baseY + 1), 0);
        colorOut.write(half4(colorR01, 1.0h - transR01), uint2(baseX + 0, baseY + 1), 1);
    }
    if (baseX + 1 < w && baseY + 1 < h) {
        colorOut.write(half4(colorL11, 1.0h - transL11), uint2(baseX + 1, baseY + 1), 0);
        colorOut.write(half4(colorR11, 1.0h - transR11), uint2(baseX + 1, baseY + 1), 1);
    }
}

struct StereoCopyVertexOut {
    float4 position [[position]];
    float2 uv;
    uint eyeIndex [[flat]];
};

vertex StereoCopyVertexOut stereoCopyVertex(
    uint vertexId [[vertex_id]],
    ushort ampId [[amplification_id]]
) {
    const float2 positions[3] = {
        float2(-1.0f, -1.0f),
        float2( 3.0f, -1.0f),
        float2(-1.0f,  3.0f)
    };
    const float2 uvs[3] = {
        float2(0.0f, 0.0f),
        float2(2.0f, 0.0f),
        float2(0.0f, 2.0f)
    };
    StereoCopyVertexOut out;
    out.position = float4(positions[vertexId], 0.0f, 1.0f);
    out.uv = uvs[vertexId];
    out.eyeIndex = uint(ampId);
    return out;
}

fragment half4 stereoCopyFragment(
    StereoCopyVertexOut in [[stage_in]],
    texture2d_array<half, access::sample> src [[texture(0)]]
) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = clamp(in.uv, 0.0f, 1.0f);
    return src.sample(s, uv, in.eyeIndex);
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
    half bg = 0.0h;
    colorTex.write(half4(bg, bg, bg, 1.0h), gid);
    if (!is_null_texture(depthTex)) {
        depthTex.write(half4(0.0h), gid);
    }
}

kernel void blockReduceKernel(
    device const uint* input [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant DepthFirstHeader& header [[buffer(2)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint BLOCK_SIZE = 256;
    uint baseIndex = groupId * BLOCK_SIZE;
    threadgroup uint shared[BLOCK_SIZE];

    uint count = header.visibleCount;
    uint value = (baseIndex + localId < count) ? input[baseIndex + localId] : 0u;
    uint sum = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(value, shared, localId);

    if (localId == 0) {
        blockSums[groupId] = sum;
    }
}

// Level 2 reduce: reduce every 256 block sums into a super-block sum
kernel void level2ReduceKernel(
    device const uint* blockSums [[buffer(0)]],
    device uint* level2Sums [[buffer(1)]],
    constant DepthFirstHeader& header [[buffer(2)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint BLOCK_SIZE = 256;
    uint numBlocks = (header.visibleCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint baseIndex = groupId * BLOCK_SIZE;

    threadgroup uint shared[BLOCK_SIZE];

    uint value = (baseIndex + localId < numBlocks) ? blockSums[baseIndex + localId] : 0u;
    uint sum = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(value, shared, localId);

    if (localId == 0) {
        level2Sums[groupId] = sum;
    }
}

kernel void level2ScanKernel(
    device uint* level2Sums [[buffer(0)]],
    constant DepthFirstHeader& header [[buffer(1)]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint BLOCK_SIZE = 256;
    threadgroup uint shared[BLOCK_SIZE];

    uint numBlocks = (header.visibleCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint numLevel2Blocks = (numBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

    uint value = (localId < numLevel2Blocks) ? level2Sums[localId] : 0u;
    uint scanned = threadgroup_raking_prefix_exclusive_sum<BLOCK_SIZE>(value, shared, localId);
    if (localId < numLevel2Blocks) {
        level2Sums[localId] = scanned;
    }
}

// Level 2 apply + block scan: apply level 2 offsets to block sums, then scan within each level 2 group
kernel void level2ApplyAndScanKernel(
    device uint* blockSums [[buffer(0)]],
    device const uint* level2Sums [[buffer(1)]],
    constant DepthFirstHeader& header [[buffer(2)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint BLOCK_SIZE = 256;
    uint numBlocks = (header.visibleCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint baseIndex = groupId * BLOCK_SIZE;

    threadgroup uint shared[BLOCK_SIZE];

    uint value = (baseIndex + localId < numBlocks) ? blockSums[baseIndex + localId] : 0u;
    uint scanned = threadgroup_raking_prefix_exclusive_sum<BLOCK_SIZE>(value, shared, localId);

    uint level2Offset = level2Sums[groupId];
    if (baseIndex + localId < numBlocks) {
        blockSums[baseIndex + localId] = scanned + level2Offset;
    }
}

kernel void blockScanKernel(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device const uint* blockOffsets [[buffer(2)]],
    constant DepthFirstHeader& header [[buffer(3)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    const uint BLOCK_SIZE = 256;
    uint baseIndex = groupId * BLOCK_SIZE;
    threadgroup uint shared[BLOCK_SIZE];

    uint count = header.visibleCount;
    uint value = (baseIndex + localId < count) ? input[baseIndex + localId] : 0u;
    uint scanned = threadgroup_raking_prefix_exclusive_sum<BLOCK_SIZE>(value, shared, localId);

    uint blockOffset = blockOffsets[groupId];
    if (baseIndex + localId < count) {
        output[baseIndex + localId] = scanned + blockOffset;
    }
}

struct DepthFirstDispatchConfig {
    uint maxGaussians;
    uint maxInstances;
    uint radixBlockSize;
    uint radixGrainSize;
    uint instanceExpansionTGSize;
    uint prefixSumTGSize;
    uint tileRangeTGSize;
    uint tileCount;
};

// Dispatch slot indices for depth-first pipeline
constant uint DF_SLOT_DEPTH_HISTOGRAM = 0;
constant uint DF_SLOT_DEPTH_SCAN_BLOCKS = 1;
constant uint DF_SLOT_DEPTH_EXCLUSIVE = 2;
constant uint DF_SLOT_DEPTH_APPLY = 3;
constant uint DF_SLOT_DEPTH_SCATTER = 4;
constant uint DF_SLOT_APPLY_DEPTH_ORDER = 5;
constant uint DF_SLOT_PREFIX_SUM = 6;
constant uint DF_SLOT_PREFIX_LEVEL2_REDUCE = 7;
constant uint DF_SLOT_PREFIX_LEVEL2_SCAN = 8;
constant uint DF_SLOT_PREFIX_LEVEL2_APPLY = 9;
constant uint DF_SLOT_PREFIX_FINAL_SCAN = 10;
constant uint DF_SLOT_CREATE_INSTANCES = 11;
// Stable radix sort for tile IDs (preserves depth order within tiles)
constant uint DF_SLOT_TILE_HISTOGRAM = 12;
constant uint DF_SLOT_TILE_SCAN_BLOCKS = 13;
constant uint DF_SLOT_TILE_EXCLUSIVE = 14;
constant uint DF_SLOT_TILE_APPLY = 15;
constant uint DF_SLOT_TILE_SCATTER = 16;
constant uint DF_SLOT_EXTRACT_RANGES = 17;
constant uint DF_SLOT_RENDER = 18;

kernel void prepareDepthFirstDispatchKernel(
    device const uint* visibleCountAtomic [[buffer(0)]],
    device const uint* totalInstancesAtomic [[buffer(1)]],
    device DepthFirstHeader* header [[buffer(2)]],
    device DispatchIndirectArgs* dispatchArgs [[buffer(3)]],
    constant DepthFirstDispatchConfig& config [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid > 0) return;

    uint visibleCount = visibleCountAtomic[0];
    uint totalInstances = totalInstancesAtomic[0];

    if (visibleCount > config.maxGaussians) {
        visibleCount = config.maxGaussians;
        header->overflow = 1u;
    }
    if (totalInstances > config.maxInstances) {
        totalInstances = config.maxInstances;
        header->overflow = 1u;
    }

    uint radixAlignment = config.radixBlockSize * config.radixGrainSize;
    uint paddedVisibleCount = ((visibleCount + radixAlignment - 1u) / radixAlignment) * radixAlignment;
    uint paddedInstanceCount = ((totalInstances + radixAlignment - 1u) / radixAlignment) * radixAlignment;

    header->visibleCount = visibleCount;
    header->totalInstances = totalInstances;
    header->paddedVisibleCount = paddedVisibleCount;
    header->paddedInstanceCount = paddedInstanceCount;

    // === Depth Sort Dispatch (based on visibleCount) ===
    uint depthRadixGrid = (visibleCount > 0u) ? ((paddedVisibleCount + radixAlignment - 1u) / radixAlignment) : 0u;
    uint depthHistBlocks = (depthRadixGrid * 256u + 255u) / 256u; // RADIX_BUCKETS = 256

    dispatchArgs[DF_SLOT_DEPTH_HISTOGRAM].threadgroupsPerGridX = depthRadixGrid;
    dispatchArgs[DF_SLOT_DEPTH_HISTOGRAM].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_DEPTH_HISTOGRAM].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_DEPTH_SCAN_BLOCKS].threadgroupsPerGridX = depthHistBlocks;
    dispatchArgs[DF_SLOT_DEPTH_SCAN_BLOCKS].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_DEPTH_SCAN_BLOCKS].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_DEPTH_EXCLUSIVE].threadgroupsPerGridX = (depthRadixGrid > 0u) ? 1u : 0u;
    dispatchArgs[DF_SLOT_DEPTH_EXCLUSIVE].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_DEPTH_EXCLUSIVE].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_DEPTH_APPLY].threadgroupsPerGridX = depthHistBlocks;
    dispatchArgs[DF_SLOT_DEPTH_APPLY].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_DEPTH_APPLY].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_DEPTH_SCATTER].threadgroupsPerGridX = depthRadixGrid;
    dispatchArgs[DF_SLOT_DEPTH_SCATTER].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_DEPTH_SCATTER].threadgroupsPerGridZ = 1u;

    // === Instance Expansion Dispatch (based on visibleCount) ===
    uint instanceTG = config.instanceExpansionTGSize;
    uint instanceGroups = (visibleCount > 0u) ? ((visibleCount + instanceTG - 1u) / instanceTG) : 0u;

    dispatchArgs[DF_SLOT_APPLY_DEPTH_ORDER].threadgroupsPerGridX = instanceGroups;
    dispatchArgs[DF_SLOT_APPLY_DEPTH_ORDER].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_APPLY_DEPTH_ORDER].threadgroupsPerGridZ = 1u;

    // Prefix sum dispatch (multi-level for large block counts)
    uint prefixTG = config.prefixSumTGSize;  // 256
    uint prefixGroups = (visibleCount > 0u) ? ((visibleCount + prefixTG - 1u) / prefixTG) : 0u;
    uint prefixLevel2Groups = (prefixGroups + prefixTG - 1u) / prefixTG;

    // Level 1: block reduce (same dispatch as before)
    dispatchArgs[DF_SLOT_PREFIX_SUM].threadgroupsPerGridX = prefixGroups;
    dispatchArgs[DF_SLOT_PREFIX_SUM].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_PREFIX_SUM].threadgroupsPerGridZ = 1u;

    // Level 2: reduce block sums into super-block sums
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_REDUCE].threadgroupsPerGridX = prefixLevel2Groups;
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_REDUCE].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_REDUCE].threadgroupsPerGridZ = 1u;

    // Level 2: scan super-block sums (single block, fits 256 super-blocks = 65536 blocks = 16M elements)
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_SCAN].threadgroupsPerGridX = (prefixLevel2Groups > 0u) ? 1u : 0u;
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_SCAN].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_SCAN].threadgroupsPerGridZ = 1u;

    // Level 2: apply super-block offsets and scan block sums
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_APPLY].threadgroupsPerGridX = prefixLevel2Groups;
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_APPLY].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_PREFIX_LEVEL2_APPLY].threadgroupsPerGridZ = 1u;

    // Final block scan with block offsets
    dispatchArgs[DF_SLOT_PREFIX_FINAL_SCAN].threadgroupsPerGridX = prefixGroups;
    dispatchArgs[DF_SLOT_PREFIX_FINAL_SCAN].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_PREFIX_FINAL_SCAN].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_CREATE_INSTANCES].threadgroupsPerGridX = instanceGroups;
    dispatchArgs[DF_SLOT_CREATE_INSTANCES].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_CREATE_INSTANCES].threadgroupsPerGridZ = 1u;

    // === Tile Sort Dispatch (stable radix sort based on totalInstances) ===
    uint tileRadixGrid = (totalInstances > 0u) ? ((paddedInstanceCount + radixAlignment - 1u) / radixAlignment) : 0u;
    uint tileHistBlocks = (tileRadixGrid * 256u + 255u) / 256u;

    dispatchArgs[DF_SLOT_TILE_HISTOGRAM].threadgroupsPerGridX = tileRadixGrid;
    dispatchArgs[DF_SLOT_TILE_HISTOGRAM].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_TILE_HISTOGRAM].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_TILE_SCAN_BLOCKS].threadgroupsPerGridX = tileHistBlocks;
    dispatchArgs[DF_SLOT_TILE_SCAN_BLOCKS].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_TILE_SCAN_BLOCKS].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_TILE_EXCLUSIVE].threadgroupsPerGridX = (tileRadixGrid > 0u) ? 1u : 0u;
    dispatchArgs[DF_SLOT_TILE_EXCLUSIVE].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_TILE_EXCLUSIVE].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_TILE_APPLY].threadgroupsPerGridX = tileHistBlocks;
    dispatchArgs[DF_SLOT_TILE_APPLY].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_TILE_APPLY].threadgroupsPerGridZ = 1u;

    dispatchArgs[DF_SLOT_TILE_SCATTER].threadgroupsPerGridX = tileRadixGrid;
    dispatchArgs[DF_SLOT_TILE_SCATTER].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_TILE_SCATTER].threadgroupsPerGridZ = 1u;

    // === Tile Range Extraction (based on tileCount - iterates over all tiles) ===
    uint rangeTG = config.tileRangeTGSize;
    uint rangeGroups = (config.tileCount + rangeTG - 1u) / rangeTG;

    dispatchArgs[DF_SLOT_EXTRACT_RANGES].threadgroupsPerGridX = rangeGroups;
    dispatchArgs[DF_SLOT_EXTRACT_RANGES].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_EXTRACT_RANGES].threadgroupsPerGridZ = 1u;

    // Render dispatch will be set by prepareRenderDispatchKernel after extractTileRanges
    // (activeTileCount not known until then)
    dispatchArgs[DF_SLOT_RENDER].threadgroupsPerGridX = 0u;
    dispatchArgs[DF_SLOT_RENDER].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_RENDER].threadgroupsPerGridZ = 1u;
}

kernel void prepareRenderDispatchKernel(
    device const uint* activeTileCount [[buffer(0)]],
    device DispatchIndirectArgs* dispatchArgs [[buffer(1)]],
    constant uint& tileCount [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid > 0) return;

    uint count = activeTileCount[0];
    if (count > tileCount) count = tileCount;

    dispatchArgs[DF_SLOT_RENDER].threadgroupsPerGridX = count;
    dispatchArgs[DF_SLOT_RENDER].threadgroupsPerGridY = 1u;
    dispatchArgs[DF_SLOT_RENDER].threadgroupsPerGridZ = 1u;
}

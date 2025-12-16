#include <metal_stdlib>
#include <metal_atomic>
#include <metal_mesh>
#include "BridgingTypes.h"
#include "GaussianShared.h"

using namespace metal;

inline uint float_to_sortable_uint(float v) {
    uint bits = as_type<uint>(v);
    uint mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

constant float MIN_OPACITY = 1.0f / 255.0f;
constant uint MAX_INDEXED_SPLAT_COUNT = 1024;
constant float BOUNDS_RADIUS = 3.0f;
constant float BOUNDS_RADIUS_SQUARED = BOUNDS_RADIUS * BOUNDS_RADIUS;

typedef struct {
    half4 color [[raster_order_group(0)]];
    float depth [[raster_order_group(0)]];
} DepthFragmentValues;

typedef struct {
    DepthFragmentValues values [[imageblock_data]];
} DepthFragmentStore;

typedef struct {
    half4 color [[color(0)]];
    float depth [[depth(any)]];
} DepthFragmentOut;

kernel void initializeFragmentStore(
    imageblock<DepthFragmentValues, imageblock_layout_explicit> blockData,
    ushort2 localThreadID [[thread_position_in_threadgroup]]
) {
    threadgroup_imageblock DepthFragmentValues* values = blockData.data(localThreadID);
    values->color = half4(0.0h);
    values->depth = 0.0f;
}

struct FullscreenVertexOut {
    float4 position [[position]];
};

vertex FullscreenVertexOut postprocessVertexShader(uint vertexID [[vertex_id]]) {
    FullscreenVertexOut out;

    float4 position;
    position.x = (vertexID == 2) ? 3.0 : -1.0;
    position.y = (vertexID == 0) ? -3.0 : 1.0;
    position.zw = 1.0;

    out.position = position;
    return out;
}

fragment DepthFragmentOut postprocessFragmentShader(DepthFragmentValues values [[imageblock_data]]) {
    DepthFragmentOut out;
    float alpha = float(values.color.a);
    out.depth = (alpha == 0.0f) ? 1.0f : (values.depth / alpha);
    out.color = values.color;
    return out;
}

fragment half4 postprocessFragmentShaderNoDepth(DepthFragmentValues values [[imageblock_data]]) {
    return values.color;
}

inline bool thetaSigmasFromCovariance2D(
    float2x2 cov,
    thread float& theta,
    thread float& sigma1,
    thread float& sigma2
) {
    // Use shared helper (stable eigen decomposition) and keep packing logic consistent
    // with Global/DepthFirst pipelines.
    return covarianceToThetaSigmas(cov, theta, sigma1, sigma2);
}

struct EyeProjectionParams {
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float3 cameraPos;
};

struct EyeProjected {
    float2 screenPos;
    float theta;
    float sigma1;
    float sigma2;
    float2 obbExtents;
    float radius;
    float depth;
    float detCov;
    bool inFront;
};

// Note: stabilizeCovariance2D is now provided by GaussianShared.h

inline EyeProjected projectGaussianForEye(
    float3x3 cov3d,
    float4 worldPos4,
    EyeProjectionParams eye,
    float2 screenSize,
    float nearPlane
) {
    EyeProjected out;
    out.screenPos = float2(0.0f);
    out.theta = 0.0f;
    out.sigma1 = 0.0f;
    out.sigma2 = 0.0f;
    out.radius = 0.0f;
    out.obbExtents = float2(0.0f);
    out.depth = 0.0f;
    out.detCov = 0.0f;
    out.inFront = false;

    float4 viewPos4 = eye.viewMatrix * worldPos4;
    float4 clipPos = eye.projectionMatrix * viewPos4;

    out.depth = clipPos.w;
    out.inFront = isInFrontOfCameraClipW(clipPos.w, nearPlane);
    if (!out.inFront) {
        return out;
    }

    float2x2 cov2d = projectCovariance2D(
        cov3d,
        viewPos4.xyz,
        eye.viewMatrix,
        eye.projectionMatrix,
        screenSize
    );

    cov2d = stabilizeCovariance2D(cov2d, screenSize);

    float theta, sigma1, sigma2;
    if (!thetaSigmasFromCovariance2D(cov2d, theta, sigma1, sigma2)) {
        out.inFront = false;
        return out;
    }

    out.theta = theta;
    out.sigma1 = sigma1;
    out.sigma2 = sigma2;

    float major = max(sigma1, sigma2);
    out.radius = BOUNDS_RADIUS * major;
    // det(cov) = (sigma1^2) * (sigma2^2)
    float sigProd = max(sigma1 * sigma2, 0.0f);
    out.detCov = sigProd * sigProd;

    // OBB extents from axes: axis1 = sigma1*[c,s], axis2 = sigma2*[-s,c]
    float c;
    float s = fast::sincos(theta, c);
    float2 axis1 = sigma1 * float2(c, s);
    float2 axis2 = sigma2 * float2(-s, c);
    out.obbExtents = BOUNDS_RADIUS * (abs(axis1) + abs(axis2));

    float2 ndc = clipPos.xy / clipPos.w;
    out.screenPos = ndcToScreen(ndc, screenSize.x, screenSize.y);
    return out;
}

// NOTE: HardwareRenderer compacts visibility deterministically (prefix sum + scatter),
// so projection kernels must write per-gaussian outputs at `gid` (not atomic append).

template<typename GaussianType, typename HarmonicType>
void stereoProjectCullImpl(
    const device GaussianType* gaussians,
    const device HarmonicType* harmonics,
    device HardwareProjectedGaussian* projected,
    device uint* preDepthKeys,
    device uint* visibilityMarks,
    constant HardwareStereoProjectionParams& params,
    uint gid
) {
    if (gid >= params.gaussianCount) return;
    visibilityMarks[gid] = 0u;
    preDepthKeys[gid] = 0xFFFFFFFFu;

    GaussianType g = gaussians[gid];
    float opacity = float(g.opacity);

    if (opacity < MIN_OPACITY) return;

    float3 scenePos = float3(g.px, g.py, g.pz);
    float4 worldPos4 = params.sceneTransform * float4(scenePos, 1.0f);
    float3 worldPos = worldPos4.xyz;

    float sceneScale = length(float3(params.sceneTransform[0][0], params.sceneTransform[0][1], params.sceneTransform[0][2]));
    float3 scale = float3(float(g.sx), float(g.sy), float(g.sz)) * sceneScale;
    float4 quat = normalizeQuaternion(getRotation(g));
    float3x3 cov3d = buildCovariance3D(scale, quat);

    float3x3 sceneRot = float3x3(
        normalize(params.sceneTransform[0].xyz),
        normalize(params.sceneTransform[1].xyz),
        normalize(params.sceneTransform[2].xyz)
    );
    cov3d = sceneRot * cov3d * transpose(sceneRot);

    float4 centerViewPos4 = params.centerViewMatrix * worldPos4;
    float4 centerClipPos = params.centerProjectionMatrix * centerViewPos4;

    float centerDepth = centerClipPos.w;
    if (!isInFrontOfCameraClipW(centerClipPos.w, params.nearPlane)) return;
    if (cullByFarPlane(centerDepth, params.farPlane)) return;

    float2 screenSize = float2(params.width, params.height);
    float2x2 centerCov2d = projectCovariance2D(
        cov3d,
        centerViewPos4.xyz,
        params.centerViewMatrix,
        params.centerProjectionMatrix,
        screenSize
    );
    centerCov2d = stabilizeCovariance2D(centerCov2d, screenSize);

    float centerTheta, centerSigma1, centerSigma2;
    if (!thetaSigmasFromCovariance2D(centerCov2d, centerTheta, centerSigma1, centerSigma2)) return;
    float centerRadius = BOUNDS_RADIUS * max(centerSigma1, centerSigma2);
    if (cullByRadius(centerRadius)) return;

    // Total ink filter using unified helper
    float sigProd = max(centerSigma1 * centerSigma2, 0.0f);
    float detCov = sigProd * sigProd;
    if (cullByTotalInk(opacity, detCov, centerDepth, params.nearPlane, params.farPlane, params.totalInkThreshold)) return;

    EyeProjectionParams leftEye;
    leftEye.viewMatrix = params.leftViewMatrix;
    leftEye.projectionMatrix = params.leftProjectionMatrix;
    leftEye.cameraPos = float3(params.leftCameraPosX, params.leftCameraPosY, params.leftCameraPosZ);

    EyeProjectionParams rightEye;
    rightEye.viewMatrix = params.rightViewMatrix;
    rightEye.projectionMatrix = params.rightProjectionMatrix;
    rightEye.cameraPos = float3(params.rightCameraPosX, params.rightCameraPosY, params.rightCameraPosZ);

    EyeProjected left = projectGaussianForEye(cov3d, worldPos4, leftEye, screenSize, params.nearPlane);
    EyeProjected right = projectGaussianForEye(cov3d, worldPos4, rightEye, screenSize, params.nearPlane);

    // Avoid emitting tiny splats that won't contribute.
    if (max(left.radius, right.radius) < 0.5f) return;

    bool leftVisible = left.inFront && (left.screenPos.x + left.obbExtents.x >= 0.0f && left.screenPos.x - left.obbExtents.x < params.width &&
                        left.screenPos.y + left.obbExtents.y >= 0.0f && left.screenPos.y - left.obbExtents.y < params.height);
    bool rightVisible = right.inFront && (right.screenPos.x + right.obbExtents.x >= 0.0f && right.screenPos.x - right.obbExtents.x < params.width &&
                         right.screenPos.y + right.obbExtents.y >= 0.0f && right.screenPos.y - right.obbExtents.y < params.height);

    if (!leftVisible && !rightVisible) return;

    float3 centerCameraPos = float3(params.centerCameraPosX, params.centerCameraPosY, params.centerCameraPosZ);
    float3 color = computeSHColor(harmonics, gid, worldPos, centerCameraPos, params.shComponents);
    color = max(color + 0.5f, 0.0f);
    color = maybeDecodeSRGBToLinear(color, params.inputIsSRGB);

    HardwareProjectedGaussian data;
    data.leftScreenX = half(left.screenPos.x);
    data.leftScreenY = half(left.screenPos.y);
    data.rightScreenX = half(right.screenPos.x);
    data.rightScreenY = half(right.screenPos.y);
    data.leftTheta = packThetaPi(left.theta);
    data.rightTheta = packThetaPi(right.theta);
    data.leftSigma1 = half(left.sigma1);
    data.leftSigma2 = half(left.sigma2);
    data.rightSigma1 = half(right.sigma1);
    data.rightSigma2 = half(right.sigma2);
    data.depth = half(centerDepth);
    data.colorR = uchar(clamp(color.r * 255.0f, 0.0f, 255.0f));
    data.colorG = uchar(clamp(color.g * 255.0f, 0.0f, 255.0f));
    data.colorB = uchar(clamp(color.b * 255.0f, 0.0f, 255.0f));
    data.opacity = uchar(clamp(opacity * 255.0f, 0.0f, 255.0f));
    projected[gid] = data;
    preDepthKeys[gid] = float_to_sortable_uint(centerDepth);
    visibilityMarks[gid] = 1u;
}

kernel void stereoProjectCullKernel(
    const device PackedWorldGaussian* gaussians [[buffer(0)]],
    const device float* harmonics [[buffer(1)]],
    device HardwareProjectedGaussian* projected [[buffer(2)]],
    device uint* preDepthKeys [[buffer(3)]],
    device uint* visibilityMarks [[buffer(4)]],
    constant HardwareStereoProjectionParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    stereoProjectCullImpl(gaussians, harmonics, projected, preDepthKeys, visibilityMarks, params, gid);
}

kernel void stereoProjectCullKernelHalf(
    const device PackedWorldGaussianHalf* gaussians [[buffer(0)]],
    const device half* harmonics [[buffer(1)]],
    device HardwareProjectedGaussian* projected [[buffer(2)]],
    device uint* preDepthKeys [[buffer(3)]],
    device uint* visibilityMarks [[buffer(4)]],
    constant HardwareStereoProjectionParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    stereoProjectCullImpl(gaussians, harmonics, projected, preDepthKeys, visibilityMarks, params, gid);
}

kernel void prepareStereoSortKernel(
    device const uint* count [[buffer(0)]],
    device HardwareStereoHeader* header [[buffer(1)]],
    device DepthFirstHeader* sortHeader [[buffer(2)]],
    device DispatchIndirectArgs* sortDispatch [[buffer(3)]],
    device DispatchIndirectArgs* scanDispatch [[buffer(4)]],
    device DispatchIndirectArgs* reorderDispatch [[buffer(5)]],
    constant uint& radixAlignment [[buffer(6)]],
    constant uint& sortThreadgroupSize [[buffer(7)]],
    constant uint& reorderThreadgroupSize [[buffer(8)]],
    constant uint& maxNumBlocks [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint visible = *count;
    uint padded = ((visible + radixAlignment - 1) / radixAlignment) * radixAlignment;
    uint numBlocks = min(padded / radixAlignment, maxNumBlocks);
    padded = numBlocks * radixAlignment;

    header->visibleCount = min(visible, padded);
    header->paddedCount = padded;
    header->overflow = (visible > padded) ? 1 : 0;

    sortHeader->visibleCount = min(visible, padded);
    sortHeader->totalInstances = 0;
    sortHeader->paddedVisibleCount = padded;
    sortHeader->paddedInstanceCount = 0;
    sortHeader->overflow = (visible > padded) ? 1 : 0;

    sortDispatch->threadgroupsPerGridX = numBlocks > 0 ? numBlocks : 1;
    sortDispatch->threadgroupsPerGridY = 1;
    sortDispatch->threadgroupsPerGridZ = 1;

    scanDispatch->threadgroupsPerGridX = numBlocks > 0 ? numBlocks : 1;
    scanDispatch->threadgroupsPerGridY = 1;
    scanDispatch->threadgroupsPerGridZ = 1;

    reorderDispatch->threadgroupsPerGridX = visible > 0 ? ((visible + reorderThreadgroupSize - 1) / reorderThreadgroupSize) : 1;
    reorderDispatch->threadgroupsPerGridY = 1;
    reorderDispatch->threadgroupsPerGridZ = 1;
}

kernel void reorderStereoProjectedKernel(
    const device HardwareProjectedGaussian* srcData [[buffer(0)]],
    const device uint* sortedIndices [[buffer(1)]],
    device HardwareProjectedGaussian* dstData [[buffer(2)]],
    device const uint* countPtr [[buffer(3)]],
    constant uint& backToFront [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint count = *countPtr;
    if (gid >= count) return;

    uint outIdx = gid;
    uint inIdx = (backToFront != 0) ? (count - 1 - outIdx) : outIdx;
    uint srcIdx = sortedIndices[inIdx];
    dstData[gid] = srcData[srcIdx];
}

struct HardwareStereoVertexOut {
    float4 position [[position]];
    half2 relativePosition;
    half3 color;
    half opacity;
};

kernel void prepareStereoDrawArgsKernel(
    device const uint* count [[buffer(0)]],
    device DrawIndexedIndirectArgs* drawArgs [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint visible = *count;

    uint indexedCount = min(visible, MAX_INDEXED_SPLAT_COUNT);
    uint instanceCount = visible > 0 ? ((visible + MAX_INDEXED_SPLAT_COUNT - 1) / MAX_INDEXED_SPLAT_COUNT) : 0;

    drawArgs->indexCount = indexedCount * 6;
    drawArgs->instanceCount = instanceCount;
    drawArgs->indexStart = 0;
    drawArgs->baseVertex = 0;
    drawArgs->baseInstance = 0;
}

// Indirect args for mesh shader dispatch (3 x uint32 = threadgroups per grid)
struct MeshDrawIndirectArgs {
    uint threadgroupsPerGridX;
    uint threadgroupsPerGridY;
    uint threadgroupsPerGridZ;
};

constant uint MESH_GAUSSIANS_PER_OBJECT_TG = 64;

kernel void prepareMeshDrawArgsKernel(
    device const uint* visibleCount [[buffer(0)]],
    device MeshDrawIndirectArgs* drawArgs [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint count = *visibleCount;
    uint threadgroups = (count + MESH_GAUSSIANS_PER_OBJECT_TG - 1) / MESH_GAUSSIANS_PER_OBJECT_TG;

    drawArgs->threadgroupsPerGridX = threadgroups;
    drawArgs->threadgroupsPerGridY = 1;
    drawArgs->threadgroupsPerGridZ = 1;
}

vertex HardwareStereoVertexOut stereoInstancedVertex(
    uint vertexId [[vertex_id]],
    uint instanceId [[instance_id]],
    ushort ampId [[amplification_id]],
    const device HardwareProjectedGaussian* gaussians [[buffer(0)]],
    constant HardwareRenderUniforms& uniforms [[buffer(1)]],
    constant uint& visibleCount [[buffer(2)]]
) {
    uint localSplatIndex = vertexId / 4;
    uint cornerIndex = vertexId % 4;
    uint gaussianIndex = instanceId * MAX_INDEXED_SPLAT_COUNT + localSplatIndex;

    HardwareStereoVertexOut out;
    if (gaussianIndex >= visibleCount) {
        // Clip immediately (avoid undefined behavior with w=0)
        out.position = float4(2.0f, 2.0f, 1.0f, 1.0f);
        out.relativePosition = half2(0.0h);
        out.color = half3(0.0h);
        out.opacity = 0.0h;
        return out;
    }

    HardwareProjectedGaussian g = gaussians[gaussianIndex];

    half2 center = (ampId == 0)
        ? half2(g.leftScreenX, g.leftScreenY)
        : half2(g.rightScreenX, g.rightScreenY);

    float theta = (ampId == 0)
        ? unpackThetaPi(g.leftTheta)
        : unpackThetaPi(g.rightTheta);
    float sigma1 = (ampId == 0) ? float(g.leftSigma1) : float(g.rightSigma1);
    float sigma2 = (ampId == 0) ? float(g.leftSigma2) : float(g.rightSigma2);
    if (!isfinite(theta) || !isfinite(sigma1) || !isfinite(sigma2)) {
        out.position = float4(2.0f, 2.0f, 1.0f, 1.0f);
        out.relativePosition = half2(0.0h);
        out.color = half3(0.0h);
        out.opacity = 0.0h;
        return out;
    }
    float c;
    float s = fast::sincos(theta, c);
    float2 axis1 = sigma1 * float2(c, s);
    float2 axis2 = sigma2 * float2(-s, c);

    // Triangle strip quad order: (0,1,2) and (2,1,3)
    const half2 corners[4] = {
        half2(-1.0h, -1.0h),
        half2( 1.0h, -1.0h),
        half2(-1.0h,  1.0h),
        half2( 1.0h,  1.0h)
    };

    half2 corner = corners[cornerIndex];
    float2 u = float2(corner) * BOUNDS_RADIUS;
    float2 pixelOffset = axis1 * u.x + axis2 * u.y;
    float2 screenPos = float2(center) + pixelOffset;

    float2 ndc = screenToNDC(screenPos, uniforms.width, uniforms.height);
    float normalizedDepth = clamp(float(g.depth) / uniforms.farPlane, 0.0f, 1.0f);
    out.position = float4(ndc, normalizedDepth, 1.0f);
    out.relativePosition = half2(u);
    out.color = half3(half(g.colorR) / 255.0h, half(g.colorG) / 255.0h, half(g.colorB) / 255.0h);
    out.opacity = half(g.opacity) / 255.0h;

    return out;
}

fragment DepthFragmentStore stereoInstancedFragment(
    HardwareStereoVertexOut in [[stage_in]],
    DepthFragmentValues previous [[imageblock_data]]
) {
    DepthFragmentStore out;

    half remaining = 1.0h - previous.color.a;
    if (remaining < half(MIN_OPACITY)) discard_fragment();

    half2 rp = in.relativePosition;
    half r2 = dot(rp, rp);
    if (r2 > half(BOUNDS_RADIUS_SQUARED)) discard_fragment();

    half alpha = in.opacity * exp(-0.5h * r2);
    if (alpha < half(MIN_OPACITY)) discard_fragment();

    half4 premultiplied = half4(in.color * alpha, alpha);
    out.values.color = previous.color + premultiplied * remaining;
    out.values.depth = previous.depth + (in.position.z * float(alpha) * float(remaining));

    return out;
}

fragment DepthFragmentStore stereoInstancedFragmentBackToFront(
    HardwareStereoVertexOut in [[stage_in]],
    DepthFragmentValues previous [[imageblock_data]]
) {
    DepthFragmentStore out;

    half2 rp = in.relativePosition;
    half r2 = dot(rp, rp);
    if (r2 > half(BOUNDS_RADIUS_SQUARED)) discard_fragment();

    half alpha = in.opacity * exp(-0.5h * r2);
    if (alpha < half(MIN_OPACITY)) discard_fragment();

    half4 premultiplied = half4(in.color * alpha, alpha);
    half oneMinusAlpha = 1.0h - alpha;

    out.values.color = previous.color * oneMinusAlpha + premultiplied;
    out.values.depth = previous.depth * float(oneMinusAlpha) + (in.position.z * float(alpha));

    return out;
}

template<typename GaussianType, typename HarmonicType>
void monoProjectCullImpl(
    const device GaussianType* gaussians,
    const device HarmonicType* harmonics,
    device InstancedGaussianData* projected,
    device uint* preDepthKeys,
    device uint* visibilityMarks,
    constant HardwareMonoProjectionParams& params,
    uint gid
) {
    if (gid >= params.gaussianCount) return;
    visibilityMarks[gid] = 0u;
    preDepthKeys[gid] = 0xFFFFFFFFu;

    GaussianType g = gaussians[gid];
    float opacity = float(g.opacity);

    if (opacity < MIN_OPACITY) return;

    float3 scenePos = float3(g.px, g.py, g.pz);
    float4 worldPos4 = params.sceneTransform * float4(scenePos, 1.0f);
    float3 worldPos = worldPos4.xyz;

    float sceneScale = length(float3(params.sceneTransform[0][0], params.sceneTransform[0][1], params.sceneTransform[0][2]));
    float3 scale = float3(float(g.sx), float(g.sy), float(g.sz)) * sceneScale;
    float4 quat = normalizeQuaternion(getRotation(g));
    float3x3 cov3d = buildCovariance3D(scale, quat);

    float3x3 sceneRot = float3x3(
        normalize(params.sceneTransform[0].xyz),
        normalize(params.sceneTransform[1].xyz),
        normalize(params.sceneTransform[2].xyz)
    );
    cov3d = sceneRot * cov3d * transpose(sceneRot);

    EyeProjectionParams eye;
    eye.viewMatrix = params.viewMatrix;
    eye.projectionMatrix = params.projectionMatrix;
    eye.cameraPos = float3(params.cameraPosX, params.cameraPosY, params.cameraPosZ);

    uint2 screenSize = uint2(uint(params.width), uint(params.height));

    float2 screenSizeF = float2(screenSize);
    EyeProjected p = projectGaussianForEye(cov3d, worldPos4, eye, screenSizeF, params.nearPlane);
    if (!p.inFront) return;
    if (cullByFarPlane(p.depth, params.farPlane)) return;
    if (cullByRadius(p.radius)) return;
    if (cullByTotalInk(opacity, p.detCov, p.depth, params.nearPlane, params.farPlane, params.totalInkThreshold)) return;

    float width = float(screenSize.x);
    float height = float(screenSize.y);
    float2 screenPos = p.screenPos;
    float2 obbExtents = p.obbExtents;
    if (screenPos.x + obbExtents.x < 0.0f || screenPos.x - obbExtents.x >= width ||
        screenPos.y + obbExtents.y < 0.0f || screenPos.y - obbExtents.y >= height) {
        return;
    }

    float3 color = computeSHColor(harmonics, gid, worldPos, eye.cameraPos, params.shComponents);
    color = max(color + 0.5f, 0.0f);
    color = maybeDecodeSRGBToLinear(color, params.inputIsSRGB);

    InstancedGaussianData data;
    data.screenX = half(screenPos.x);
    data.screenY = half(screenPos.y);
    data.theta = packThetaPi(p.theta);
    data.sigma1 = half(p.sigma1);
    data.sigma2 = half(p.sigma2);
    data.depth = half(p.depth);
    data.colorR = uchar(clamp(color.r * 255.0f, 0.0f, 255.0f));
    data.colorG = uchar(clamp(color.g * 255.0f, 0.0f, 255.0f));
    data.colorB = uchar(clamp(color.b * 255.0f, 0.0f, 255.0f));
    data.opacity = uchar(clamp(opacity * 255.0f, 0.0f, 255.0f));
    projected[gid] = data;

    preDepthKeys[gid] = float_to_sortable_uint(p.depth);
    visibilityMarks[gid] = 1u;
}

kernel void monoProjectCullKernel(
    const device PackedWorldGaussian* gaussians [[buffer(0)]],
    const device float* harmonics [[buffer(1)]],
    device InstancedGaussianData* projected [[buffer(2)]],
    device uint* preDepthKeys [[buffer(3)]],
    device uint* visibilityMarks [[buffer(4)]],
    constant HardwareMonoProjectionParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    monoProjectCullImpl(gaussians, harmonics, projected, preDepthKeys, visibilityMarks, params, gid);
}

kernel void monoProjectCullKernelHalf(
    const device PackedWorldGaussianHalf* gaussians [[buffer(0)]],
    const device half* harmonics [[buffer(1)]],
    device InstancedGaussianData* projected [[buffer(2)]],
    device uint* preDepthKeys [[buffer(3)]],
    device uint* visibilityMarks [[buffer(4)]],
    constant HardwareMonoProjectionParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    monoProjectCullImpl(gaussians, harmonics, projected, preDepthKeys, visibilityMarks, params, gid);
}

kernel void prepareMonoSortKernel(
    device const uint* count [[buffer(0)]],
    device HardwareMonoHeader* header [[buffer(1)]],
    device DepthFirstHeader* sortHeader [[buffer(2)]],
    device DispatchIndirectArgs* sortDispatch [[buffer(3)]],
    device DispatchIndirectArgs* scanDispatch [[buffer(4)]],
    device DispatchIndirectArgs* reorderDispatch [[buffer(5)]],
    device DrawIndexedIndirectArgs* drawArgs [[buffer(6)]],
    constant uint& radixAlignment [[buffer(7)]],
    constant uint& sortThreadgroupSize [[buffer(8)]],
    constant uint& reorderThreadgroupSize [[buffer(9)]],
    constant uint& maxNumBlocks [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint visible = *count;
    uint padded = ((visible + radixAlignment - 1) / radixAlignment) * radixAlignment;
    uint numBlocks = min(padded / radixAlignment, maxNumBlocks);
    padded = numBlocks * radixAlignment;

    header->visibleCount = min(visible, padded);
    header->paddedCount = padded;
    header->overflow = (visible > padded) ? 1 : 0;

    sortHeader->visibleCount = min(visible, padded);
    sortHeader->totalInstances = 0;
    sortHeader->paddedVisibleCount = padded;
    sortHeader->paddedInstanceCount = 0;
    sortHeader->overflow = (visible > padded) ? 1 : 0;

    sortDispatch->threadgroupsPerGridX = numBlocks > 0 ? numBlocks : 1;
    sortDispatch->threadgroupsPerGridY = 1;
    sortDispatch->threadgroupsPerGridZ = 1;

    scanDispatch->threadgroupsPerGridX = numBlocks > 0 ? numBlocks : 1;
    scanDispatch->threadgroupsPerGridY = 1;
    scanDispatch->threadgroupsPerGridZ = 1;

    reorderDispatch->threadgroupsPerGridX = visible > 0 ? ((visible + reorderThreadgroupSize - 1) / reorderThreadgroupSize) : 1;
    reorderDispatch->threadgroupsPerGridY = 1;
    reorderDispatch->threadgroupsPerGridZ = 1;

    // Prepare draw args
    uint indexedCount = min(visible, MAX_INDEXED_SPLAT_COUNT);
    uint instanceCount = visible > 0 ? ((visible + MAX_INDEXED_SPLAT_COUNT - 1) / MAX_INDEXED_SPLAT_COUNT) : 0;
    drawArgs->indexCount = indexedCount * 6;
    drawArgs->instanceCount = instanceCount;
    drawArgs->indexStart = 0;
    drawArgs->baseVertex = 0;
    drawArgs->baseInstance = 0;
}

kernel void reorderMonoDataKernel(
    const device InstancedGaussianData* srcData [[buffer(0)]],
    const device uint* sortedIndices [[buffer(1)]],
    device InstancedGaussianData* dstData [[buffer(2)]],
    device const uint* countPtr [[buffer(3)]],
    constant uint& backToFront [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint count = *countPtr;
    if (gid >= count) return;

    uint outIdx = gid;
    uint inIdx = (backToFront != 0) ? (count - 1 - outIdx) : outIdx;
    uint srcIdx = sortedIndices[inIdx];
    dstData[gid] = srcData[srcIdx];
}

// Mono vertex shader - simpler version without render target array index
struct MonoVertexOut {
    float4 position [[position]];
    half2 relativePosition;
    half3 color;
    half opacity;
};

vertex MonoVertexOut monoGaussianVertex(
    uint vertexId [[vertex_id]],
    uint instanceId [[instance_id]],
    const device InstancedGaussianData* gaussians [[buffer(0)]],
    constant HardwareRenderUniforms& uniforms [[buffer(1)]],
    constant uint& visibleCount [[buffer(2)]]
) {
    uint localSplatIndex = vertexId / 4;
    uint cornerIndex = vertexId % 4;
    uint gaussianIndex = instanceId * MAX_INDEXED_SPLAT_COUNT + localSplatIndex;

    MonoVertexOut out;
    if (gaussianIndex >= visibleCount) {
        // Clip immediately (avoid undefined behavior with w=0)
        out.position = float4(2.0f, 2.0f, 1.0f, 1.0f);
        out.relativePosition = half2(0.0h);
        out.color = half3(0.0h);
        out.opacity = 0.0h;
        return out;
    }

    InstancedGaussianData g = gaussians[gaussianIndex];

    half2 center = half2(g.screenX, g.screenY);
    float theta = unpackThetaPi(g.theta);
    float sigma1 = float(g.sigma1);
    float sigma2 = float(g.sigma2);
    if (!isfinite(theta) || !isfinite(sigma1) || !isfinite(sigma2)) {
        out.position = float4(2.0f, 2.0f, 1.0f, 1.0f);
        out.relativePosition = half2(0.0h);
        out.color = half3(0.0h);
        out.opacity = 0.0h;
        return out;
    }
    float c;
    float s = fast::sincos(theta, c);
    float2 axis1 = sigma1 * float2(c, s);
    float2 axis2 = sigma2 * float2(-s, c);

    // Triangle strip quad order: (0,1,2) and (2,1,3)
    const half2 corners[4] = {
        half2(-1.0h, -1.0h),
        half2( 1.0h, -1.0h),
        half2(-1.0h,  1.0h),
        half2( 1.0h,  1.0h)
    };

    half2 corner = corners[cornerIndex];
    float2 u = float2(corner) * BOUNDS_RADIUS;
    float2 pixelOffset = axis1 * u.x + axis2 * u.y;
    float2 screenPos = float2(center) + pixelOffset;

    float2 ndc = screenToNDC(screenPos, uniforms.width, uniforms.height);
    float normalizedDepth = clamp(float(g.depth) / uniforms.farPlane, 0.0f, 1.0f);
    out.position = float4(ndc, normalizedDepth, 1.0f);
    out.relativePosition = half2(u);
    out.color = half3(half(g.colorR) / 255.0h, half(g.colorG) / 255.0h, half(g.colorB) / 255.0h);
    out.opacity = half(g.opacity) / 255.0h;

    return out;
}

fragment half4 monoGaussianFragment(MonoVertexOut in [[stage_in]]) {
    half2 rp = in.relativePosition;
    half r2 = dot(rp, rp);
    if (r2 > half(BOUNDS_RADIUS_SQUARED)) discard_fragment();

    half alpha = in.opacity * exp(-0.5h * r2);
    if (alpha < half(MIN_OPACITY)) discard_fragment();

    return half4(in.color * alpha, alpha);
}

constant uint VERTICES_PER_GAUSSIAN = 4;
constant uint TRIANGLES_PER_GAUSSIAN = 2;
constant uint GAUSSIANS_PER_OBJECT_TG = 64;
constant uint GAUSSIANS_PER_MESH_TG = 16;
constant uint MESH_THREADS = GAUSSIANS_PER_MESH_TG * VERTICES_PER_GAUSSIAN;
constant uint MESH_VERTICES = GAUSSIANS_PER_MESH_TG * VERTICES_PER_GAUSSIAN;
constant uint MESH_TRIANGLES = GAUSSIANS_PER_MESH_TG * TRIANGLES_PER_GAUSSIAN;


struct VertexOut {
    float4 position [[position]];
    half2 relativePosition;
    half3 color;
    half opacity;
};

struct PrimitiveOut {
    uint renderTargetArrayIndex [[render_target_array_index]];
};

fragment DepthFragmentStore gaussianFragmentShader(
    VertexOut in [[stage_in]],
    DepthFragmentValues previous [[imageblock_data]]
) {
    DepthFragmentStore out;

    half remaining = 1.0h - previous.color.a;
    if (remaining < half(MIN_OPACITY)) discard_fragment();

    half2 rp = in.relativePosition;
    half r2 = dot(rp, rp);
    if (r2 > half(BOUNDS_RADIUS_SQUARED)) discard_fragment();

    half alpha = in.opacity * exp(-0.5h * r2);
    if (alpha < half(MIN_OPACITY)) discard_fragment();

    half4 premultiplied = half4(in.color * alpha, alpha);
    out.values.color = previous.color + premultiplied * remaining;
    out.values.depth = previous.depth + (in.position.z * float(alpha) * float(remaining));

    return out;
}

fragment DepthFragmentStore gaussianFragmentShaderBackToFront(
    VertexOut in [[stage_in]],
    DepthFragmentValues previous [[imageblock_data]]
) {
    DepthFragmentStore out;

    half2 rp = in.relativePosition;
    half r2 = dot(rp, rp);
    if (r2 > half(BOUNDS_RADIUS_SQUARED)) discard_fragment();

    half alpha = in.opacity * exp(-0.5h * r2);
    if (alpha < half(MIN_OPACITY)) discard_fragment();

    half4 premultiplied = half4(in.color * alpha, alpha);
    half oneMinusAlpha = 1.0h - alpha;

    out.values.color = previous.color * oneMinusAlpha + premultiplied;
    out.values.depth = previous.depth * float(oneMinusAlpha) + (in.position.z * float(alpha));

    return out;
}

constant uint CENTER_GAUSSIANS_PER_OBJECT_TG = 64;
constant uint CENTER_GAUSSIANS_PER_MESH_TG = 16;
constant uint CENTER_MESH_THREADS = CENTER_GAUSSIANS_PER_MESH_TG * VERTICES_PER_GAUSSIAN;
constant uint CENTER_MESH_VERTICES = CENTER_GAUSSIANS_PER_MESH_TG * VERTICES_PER_GAUSSIAN;
constant uint CENTER_MESH_TRIANGLES = CENTER_GAUSSIANS_PER_MESH_TG * TRIANGLES_PER_GAUSSIAN;

struct HardwareStereoPayload {
    HardwareProjectedGaussian gaussians[CENTER_GAUSSIANS_PER_OBJECT_TG];
    uint validCount;
};

[[object, max_total_threads_per_threadgroup(CENTER_GAUSSIANS_PER_OBJECT_TG)]]
void stereoObjectShader(
    object_data HardwareStereoPayload& payload [[payload]],
    const device HardwareProjectedGaussian* projectedSorted [[buffer(0)]],
    constant HardwareRenderUniforms& uniforms [[buffer(1)]],
    const device uint* visibleCountPtr [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    mesh_grid_properties mgp
) {
    uint visibleCount = *visibleCountPtr;
    uint globalIdx = tgid * CENTER_GAUSSIANS_PER_OBJECT_TG + tid;

    if (globalIdx < visibleCount && tid < CENTER_GAUSSIANS_PER_OBJECT_TG) {
        payload.gaussians[tid] = projectedSorted[globalIdx];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint remaining = visibleCount > tgid * CENTER_GAUSSIANS_PER_OBJECT_TG
            ? visibleCount - tgid * CENTER_GAUSSIANS_PER_OBJECT_TG
            : 0;
        payload.validCount = min(remaining, CENTER_GAUSSIANS_PER_OBJECT_TG);

        uint meshTGs = (payload.validCount + CENTER_GAUSSIANS_PER_MESH_TG - 1) / CENTER_GAUSSIANS_PER_MESH_TG;
        mgp.set_threadgroups_per_grid(uint3(meshTGs, 1, 1));
    }
}

using HardwareStereoMesh = metal::mesh<VertexOut, PrimitiveOut, CENTER_MESH_VERTICES, CENTER_MESH_TRIANGLES, metal::topology::triangle>;

[[mesh, max_total_threads_per_threadgroup(CENTER_MESH_THREADS)]]
void stereoMeshShader(
    HardwareStereoMesh output,
    const object_data HardwareStereoPayload& payload [[payload]],
    constant HardwareRenderUniforms& uniforms [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort ampId [[amplification_id]]
) {
    uint localGaussianIdx = tid / VERTICES_PER_GAUSSIAN;
    uint vertexIdx = tid % VERTICES_PER_GAUSSIAN;
    uint payloadGaussianIdx = tgid * CENTER_GAUSSIANS_PER_MESH_TG + localGaussianIdx;
    bool validGaussian = payloadGaussianIdx < payload.validCount;

    VertexOut v;
    if (validGaussian) {
        HardwareProjectedGaussian g = payload.gaussians[payloadGaussianIdx];

        half2 center = (ampId == 0)
            ? half2(g.leftScreenX, g.leftScreenY)
            : half2(g.rightScreenX, g.rightScreenY);

        float theta = (ampId == 0)
            ? unpackThetaPi(g.leftTheta)
            : unpackThetaPi(g.rightTheta);
        float sigma1 = (ampId == 0) ? float(g.leftSigma1) : float(g.rightSigma1);
        float sigma2 = (ampId == 0) ? float(g.leftSigma2) : float(g.rightSigma2);
        if (!isfinite(theta) || !isfinite(sigma1) || !isfinite(sigma2)) {
            v.position = float4(2.0f, 2.0f, 1.0f, 1.0f);
            v.relativePosition = half2(0.0h);
            v.color = half3(0.0h);
            v.opacity = 0.0h;
            output.set_vertex(tid, v);
            return;
        }
        float c;
        float s = fast::sincos(theta, c);
        float2 axis1 = sigma1 * float2(c, s);
        float2 axis2 = sigma2 * float2(-s, c);

        const half2 corners[4] = {
            half2(-1.0h, -1.0h),
            half2( 1.0h, -1.0h),
            half2(-1.0h,  1.0h),
            half2( 1.0h,  1.0h)
        };

        half2 corner = corners[vertexIdx];
        float2 u = float2(corner) * BOUNDS_RADIUS;
        float2 pixelOffset = axis1 * u.x + axis2 * u.y;
        float2 screenPos = float2(center) + pixelOffset;

        float2 ndc = screenToNDC(screenPos, uniforms.width, uniforms.height);
        float normalizedDepth = clamp(float(g.depth) / uniforms.farPlane, 0.0f, 1.0f);
        v.position = float4(ndc, normalizedDepth, 1.0f);
        v.relativePosition = half2(u);
        v.color = half3(half(g.colorR) / 255.0h, half(g.colorG) / 255.0h, half(g.colorB) / 255.0h);
        v.opacity = half(g.opacity) / 255.0h;
    } else {
        v.position = float4(2.0f, 2.0f, 1.0f, 1.0f);
        v.relativePosition = half2(0.0h);
        v.color = half3(0.0h);
        v.opacity = 0.0h;
    }
    output.set_vertex(tid, v);

    if (tid == 0) {
        uint startGaussian = tgid * CENTER_GAUSSIANS_PER_MESH_TG;
        uint endGaussian = min(startGaussian + CENTER_GAUSSIANS_PER_MESH_TG, payload.validCount);
        uint gaussiansInThisTG = endGaussian > startGaussian ? endGaussian - startGaussian : 0;

        for (uint i = 0; i < gaussiansInThisTG; i++) {
            uint baseVertex = i * VERTICES_PER_GAUSSIAN;
            uint baseIndex = i * 6;

            output.set_index(baseIndex + 0, baseVertex + 0);
            output.set_index(baseIndex + 1, baseVertex + 1);
            output.set_index(baseIndex + 2, baseVertex + 2);
            output.set_index(baseIndex + 3, baseVertex + 2);
            output.set_index(baseIndex + 4, baseVertex + 1);
            output.set_index(baseIndex + 5, baseVertex + 3);

            // Vertex amplification viewMappings already selects the target slice (offset 0/1),
            // so keep this at 0 to avoid double-offsetting into a non-existent slice.
            PrimitiveOut prim;
            prim.renderTargetArrayIndex = 0;
            output.set_primitive(i * 2, prim);
            output.set_primitive(i * 2 + 1, prim);
        }

        output.set_primitive_count(gaussiansInThisTG * TRIANGLES_PER_GAUSSIAN);
    }
}

struct MonoObjectPayload {
    InstancedGaussianData gaussians[GAUSSIANS_PER_OBJECT_TG];
    uint validCount;
};

[[object, max_total_threads_per_threadgroup(GAUSSIANS_PER_OBJECT_TG)]]
void monoObjectShader(
    object_data MonoObjectPayload& payload [[payload]],
    const device InstancedGaussianData* projectedSorted [[buffer(0)]],
    constant HardwareRenderUniforms& uniforms [[buffer(1)]],
    const device uint* visibleCountPtr [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    mesh_grid_properties mgp
) {
    uint visibleCount = *visibleCountPtr;
    uint globalIdx = tgid * GAUSSIANS_PER_OBJECT_TG + tid;

    if (globalIdx < visibleCount && tid < GAUSSIANS_PER_OBJECT_TG) {
        payload.gaussians[tid] = projectedSorted[globalIdx];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint remaining = visibleCount > tgid * GAUSSIANS_PER_OBJECT_TG
            ? visibleCount - tgid * GAUSSIANS_PER_OBJECT_TG
            : 0;
        payload.validCount = min(remaining, GAUSSIANS_PER_OBJECT_TG);

        uint meshTGs = (payload.validCount + GAUSSIANS_PER_MESH_TG - 1) / GAUSSIANS_PER_MESH_TG;
        mgp.set_threadgroups_per_grid(uint3(meshTGs, 1, 1));
    }
}

using MonoMesh = metal::mesh<MonoVertexOut, void, MESH_VERTICES, MESH_TRIANGLES, metal::topology::triangle>;

[[mesh, max_total_threads_per_threadgroup(MESH_THREADS)]]
void monoMeshShader(
    MonoMesh output,
    const object_data MonoObjectPayload& payload [[payload]],
    constant HardwareRenderUniforms& uniforms [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint localGaussianIdx = tid / VERTICES_PER_GAUSSIAN;
    uint vertexIdx = tid % VERTICES_PER_GAUSSIAN;
    uint payloadGaussianIdx = tgid * GAUSSIANS_PER_MESH_TG + localGaussianIdx;
    bool validGaussian = payloadGaussianIdx < payload.validCount;

    MonoVertexOut v;
    if (validGaussian) {
        InstancedGaussianData g = payload.gaussians[payloadGaussianIdx];

        half2 center = half2(g.screenX, g.screenY);
        float theta = unpackThetaPi(g.theta);
        float sigma1 = float(g.sigma1);
        float sigma2 = float(g.sigma2);
        if (!isfinite(theta) || !isfinite(sigma1) || !isfinite(sigma2)) {
            v.position = float4(2.0f, 2.0f, 1.0f, 1.0f);
            v.relativePosition = half2(0.0h);
            v.color = half3(0.0h);
            v.opacity = 0.0h;
            output.set_vertex(tid, v);
            return;
        }
        float c;
        float s = fast::sincos(theta, c);
        float2 axis1 = sigma1 * float2(c, s);
        float2 axis2 = sigma2 * float2(-s, c);

        const half2 corners[4] = {
            half2(-1.0h, -1.0h),
            half2( 1.0h, -1.0h),
            half2(-1.0h,  1.0h),
            half2( 1.0h,  1.0h)
        };

        half2 corner = corners[vertexIdx];
        float2 u = float2(corner) * BOUNDS_RADIUS;
        float2 pixelOffset = axis1 * u.x + axis2 * u.y;
        float2 screenPos = float2(center) + pixelOffset;

        float2 ndc = screenToNDC(screenPos, uniforms.width, uniforms.height);
        float normalizedDepth = clamp(float(g.depth) / uniforms.farPlane, 0.0f, 1.0f);
        v.position = float4(ndc, normalizedDepth, 1.0f);
        v.relativePosition = half2(u);
        v.color = half3(half(g.colorR) / 255.0h, half(g.colorG) / 255.0h, half(g.colorB) / 255.0h);
        v.opacity = half(g.opacity) / 255.0h;
    } else {
        v.position = float4(2.0f, 2.0f, 1.0f, 1.0f);
        v.relativePosition = half2(0.0h);
        v.color = half3(0.0h);
        v.opacity = 0.0h;
    }
    output.set_vertex(tid, v);

    if (tid == 0) {
        uint startGaussian = tgid * GAUSSIANS_PER_MESH_TG;
        uint endGaussian = min(startGaussian + GAUSSIANS_PER_MESH_TG, payload.validCount);
        uint gaussiansInThisTG = endGaussian > startGaussian ? endGaussian - startGaussian : 0;

        for (uint i = 0; i < gaussiansInThisTG; i++) {
            uint baseVertex = i * VERTICES_PER_GAUSSIAN;
            uint baseIndex = i * 6;

            output.set_index(baseIndex + 0, baseVertex + 0);
            output.set_index(baseIndex + 1, baseVertex + 1);
            output.set_index(baseIndex + 2, baseVertex + 2);
            output.set_index(baseIndex + 3, baseVertex + 2);
            output.set_index(baseIndex + 4, baseVertex + 1);
            output.set_index(baseIndex + 5, baseVertex + 3);
        }

        output.set_primitive_count(gaussiansInThisTG * TRIANGLES_PER_GAUSSIAN);
    }
}

[[fragment]]
half4 monoFragmentShader(MonoVertexOut in [[stage_in]]) {
    half2 rp = in.relativePosition;
    half r2 = dot(rp, rp);
    if (r2 > half(BOUNDS_RADIUS_SQUARED)) discard_fragment();

    half alpha = in.opacity * exp(-0.5h * r2);
    if (alpha < half(MIN_OPACITY)) discard_fragment();

    return half4(in.color * alpha, alpha);
}

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

#define TILE_PREFIX_BLOCK_SIZE 256
#define TILE_PREFIX_GRAIN_SIZE 4


template <ushort BLOCK_SIZE, typename T> static uchar
FlagHeadDiscontinuity(const T value, threadgroup T* shared, const ushort local_id){
    shared[local_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uchar result = (local_id == 0) ? 1 : shared[local_id] != shared[local_id - 1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

template <ushort BLOCK_SIZE, typename T> static uchar
FlagTailDiscontinuity(const T value, threadgroup T* shared, const ushort local_id){
    shared[local_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uchar result = (local_id == BLOCK_SIZE - 1) ? 1 : shared[local_id] != shared[local_id + 1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

template<ushort LENGTH, typename T> static void
LoadStripedLocalFromGlobal(thread T (&value)[LENGTH],
                           const device T* input_data,
                           const ushort local_id,
                           const ushort local_size) {
    for (ushort i = 0; i < LENGTH; i++){
        value[i] = input_data[local_id + i * local_size];
    }
}

template<ushort LENGTH, typename T> static void
LoadStripedLocalFromGlobal(thread T (&value)[LENGTH],
                           const device T* input_data,
                           const ushort local_id,
                           const ushort local_size,
                           const uint n,
                           const T substitution_value){
    for (ushort i = 0; i < LENGTH; i++){
        value[i] = (local_id + i * local_size < n) ? input_data[local_id + i * local_size] : substitution_value;
    }
}

template<ushort GRAIN_SIZE, typename T> static void
LoadBlockedLocalFromGlobal(thread T (&value)[GRAIN_SIZE],
                           const device T* input_data,
                           const ushort local_id) {
    for (ushort i = 0; i < GRAIN_SIZE; i++){
        value[i] = input_data[local_id * GRAIN_SIZE + i];
    }
}

template<ushort GRAIN_SIZE, typename T> static void
LoadBlockedLocalFromGlobal(thread T (&value)[GRAIN_SIZE],
                           const device T* input_data,
                           const ushort local_id,
                           const uint n,
                           const T substitution_value) {
    for (ushort i = 0; i < GRAIN_SIZE; i++){
        value[i] = (local_id * GRAIN_SIZE + i < n) ? input_data[local_id * GRAIN_SIZE + i] : substitution_value;
    }
}

template<ushort GRAIN_SIZE, typename T> static void
StoreBlockedLocalToGlobal(device T *output_data,
                          thread const T (&value)[GRAIN_SIZE],
                          const ushort local_id) {
    for (ushort i = 0; i < GRAIN_SIZE; i++){
        output_data[local_id * GRAIN_SIZE + i] = value[i];
    }
}

template<ushort GRAIN_SIZE, typename T> static void
StoreBlockedLocalToGlobal(device T *output_data,
                          thread const T (&value)[GRAIN_SIZE],
                          const ushort local_id,
                          const uint n) {
    for (ushort i = 0; i < GRAIN_SIZE; i++){
        if (local_id * GRAIN_SIZE + i < n) {
            output_data[local_id * GRAIN_SIZE + i] = value[i];
        }
    }
}

template<ushort LENGTH, typename T>
static inline T ThreadPrefixInclusiveSum(thread T (&values)[LENGTH]){
    for (ushort i = 1; i < LENGTH; i++){
        values[i] += values[i - 1];
    }
    return values[LENGTH - 1];
}

template<ushort LENGTH, typename T>
static inline T ThreadPrefixInclusiveSum(threadgroup T* values){
    for (ushort i = 1; i < LENGTH; i++){
        values[i] += values[i - 1];
    }
    return values[LENGTH - 1];
}

template<ushort LENGTH, typename T>
static inline T ThreadPrefixExclusiveSum(thread T (&values)[LENGTH]){
    T inclusive_prefix = ThreadPrefixInclusiveSum<LENGTH>(values);
    for (ushort i = LENGTH - 1; i > 0; i--){
        values[i] = values[i - 1];
    }
    values[0] = 0;
    return inclusive_prefix;
}

template<ushort LENGTH, typename T>
static inline T ThreadPrefixExclusiveSum(threadgroup T* values){
    T inclusive_prefix = ThreadPrefixInclusiveSum<LENGTH>(values);
    for (ushort i = LENGTH - 1; i > 0; i--){
        values[i] = values[i - 1];
    }
    values[0] = 0;
    return inclusive_prefix;
}

template<ushort LENGTH, typename T> static inline void
ThreadUniformAdd(thread T (&values)[LENGTH], T uni){
    for (ushort i = 0; i < LENGTH; i++){
        values[i] += uni;
    }
}

template<ushort LENGTH, typename T> static inline void
ThreadUniformAdd(threadgroup T* values, T uni){
    for (ushort i = 0; i < LENGTH; i++){
        values[i] += uni;
    }
}

template<ushort BLOCK_SIZE, typename T> static T
ThreadgroupBlellochPrefixExclusiveSum(T value, threadgroup T* sdata, const ushort lid) {
    sdata[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ushort ai = 2 * lid + 1;
    const ushort bi = 2 * lid + 2;

    if (BLOCK_SIZE >=    2) {if (lid < (BLOCK_SIZE >>  1) ) {sdata[   1 * bi - 1] += sdata[   1 * ai - 1];} if ((BLOCK_SIZE >>  0) > 32) threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >=    4) {if (lid < (BLOCK_SIZE >>  2) ) {sdata[   2 * bi - 1] += sdata[   2 * ai - 1];} if ((BLOCK_SIZE >>  1) > 32) threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >=    8) {if (lid < (BLOCK_SIZE >>  3) ) {sdata[   4 * bi - 1] += sdata[   4 * ai - 1];} if ((BLOCK_SIZE >>  2) > 32) threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >=   16) {if (lid < (BLOCK_SIZE >>  4) ) {sdata[   8 * bi - 1] += sdata[   8 * ai - 1];} if ((BLOCK_SIZE >>  3) > 32) threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >=   32) {if (lid < (BLOCK_SIZE >>  5) ) {sdata[  16 * bi - 1] += sdata[  16 * ai - 1];} if ((BLOCK_SIZE >>  4) > 32) threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >=   64) {if (lid < (BLOCK_SIZE >>  6) ) {sdata[  32 * bi - 1] += sdata[  32 * ai - 1];} }
    if (BLOCK_SIZE >=  128) {if (lid < (BLOCK_SIZE >>  7) ) {sdata[  64 * bi - 1] += sdata[  64 * ai - 1];} }
    if (BLOCK_SIZE >=  256) {if (lid < (BLOCK_SIZE >>  8) ) {sdata[ 128 * bi - 1] += sdata[ 128 * ai - 1];} }
    if (BLOCK_SIZE >=  512) {if (lid < (BLOCK_SIZE >>  9) ) {sdata[ 256 * bi - 1] += sdata[ 256 * ai - 1];} }
    if (BLOCK_SIZE >= 1024) {if (lid < (BLOCK_SIZE >> 10) ) {sdata[ 512 * bi - 1] += sdata[ 512 * ai - 1];} }

    if (lid == 0){
        sdata[BLOCK_SIZE - 1] = 0;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (BLOCK_SIZE >=    2){
        if (lid <    1) {
            sdata[(BLOCK_SIZE >>  1) * bi - 1] += sdata[(BLOCK_SIZE >>  1) * ai - 1];
            sdata[(BLOCK_SIZE >>  1) * ai - 1] = sdata[(BLOCK_SIZE >>  1) * bi - 1] - sdata[(BLOCK_SIZE >>  1) * ai - 1];
        }
    }
    if (BLOCK_SIZE >=    4){ if (lid <    2) {sdata[(BLOCK_SIZE >>  2) * bi - 1] += sdata[(BLOCK_SIZE >>  2) * ai - 1]; sdata[(BLOCK_SIZE >>  2) * ai - 1] = sdata[(BLOCK_SIZE >>  2) * bi - 1] - sdata[(BLOCK_SIZE >>  2) * ai - 1];} }
    if (BLOCK_SIZE >=    8){ if (lid <    4) {sdata[(BLOCK_SIZE >>  3) * bi - 1] += sdata[(BLOCK_SIZE >>  3) * ai - 1]; sdata[(BLOCK_SIZE >>  3) * ai - 1] = sdata[(BLOCK_SIZE >>  3) * bi - 1] - sdata[(BLOCK_SIZE >>  3) * ai - 1];} }
    if (BLOCK_SIZE >=   16){ if (lid <    8) {sdata[(BLOCK_SIZE >>  4) * bi - 1] += sdata[(BLOCK_SIZE >>  4) * ai - 1]; sdata[(BLOCK_SIZE >>  4) * ai - 1] = sdata[(BLOCK_SIZE >>  4) * bi - 1] - sdata[(BLOCK_SIZE >>  4) * ai - 1];} }
    if (BLOCK_SIZE >=   32){ if (lid <   16) {sdata[(BLOCK_SIZE >>  5) * bi - 1] += sdata[(BLOCK_SIZE >>  5) * ai - 1]; sdata[(BLOCK_SIZE >>  5) * ai - 1] = sdata[(BLOCK_SIZE >>  5) * bi - 1] - sdata[(BLOCK_SIZE >>  5) * ai - 1];} }
    if (BLOCK_SIZE >=   64){ if (lid <   32) {sdata[(BLOCK_SIZE >>  6) * bi - 1] += sdata[(BLOCK_SIZE >>  6) * ai - 1]; sdata[(BLOCK_SIZE >>  6) * ai - 1] = sdata[(BLOCK_SIZE >>  6) * bi - 1] - sdata[(BLOCK_SIZE >>  6) * ai - 1];} threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >=  128){ if (lid <   64) {sdata[(BLOCK_SIZE >>  7) * bi - 1] += sdata[(BLOCK_SIZE >>  7) * ai - 1]; sdata[(BLOCK_SIZE >>  7) * ai - 1] = sdata[(BLOCK_SIZE >>  7) * bi - 1] - sdata[(BLOCK_SIZE >>  7) * ai - 1];} threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >=  256){ if (lid <  128) {sdata[(BLOCK_SIZE >>  8) * bi - 1] += sdata[(BLOCK_SIZE >>  8) * ai - 1]; sdata[(BLOCK_SIZE >>  8) * ai - 1] = sdata[(BLOCK_SIZE >>  8) * bi - 1] - sdata[(BLOCK_SIZE >>  8) * ai - 1];} threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >=  512){ if (lid <  256) {sdata[(BLOCK_SIZE >>  9) * bi - 1] += sdata[(BLOCK_SIZE >>  9) * ai - 1]; sdata[(BLOCK_SIZE >>  9) * ai - 1] = sdata[(BLOCK_SIZE >>  9) * bi - 1] - sdata[(BLOCK_SIZE >>  9) * ai - 1];} threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >= 1024){ if (lid <  512) {sdata[(BLOCK_SIZE >> 10) * bi - 1] += sdata[(BLOCK_SIZE >> 10) * ai - 1]; sdata[(BLOCK_SIZE >> 10) * ai - 1] = sdata[(BLOCK_SIZE >> 10) * bi - 1] - sdata[(BLOCK_SIZE >> 10) * ai - 1];} threadgroup_barrier(mem_flags::mem_threadgroup); }

    return sdata[lid];
}

template<ushort BLOCK_SIZE, typename T> static T
ThreadgroupRakingPrefixExclusiveSum(T value, threadgroup T* shared, const ushort lid) {
    shared[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < 32){
        const short values_per_thread = BLOCK_SIZE / 32;
        const short first_index = lid * values_per_thread;
        for (short i = first_index + 1; i < first_index + values_per_thread; i++){
            shared[i] += shared[i - 1];
        }
        T partial_sum = shared[first_index + values_per_thread - 1];
        for (short i = first_index + values_per_thread - 1; i > first_index; i--){
            shared[i] = shared[i - 1];
        }
        shared[first_index] = 0;
        
        T prefix = simd_prefix_exclusive_sum(partial_sum);
        
        for (short i = first_index; i < first_index + values_per_thread; i++){
            shared[i] += prefix;
        }
        
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return shared[lid];
}


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

struct CameraUniforms {
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

struct RenderParams {
    uint width;
    uint height;
    uint tileWidth;
    uint tileHeight;
    uint tilesX;
    uint tilesY;
    uint maxPerTile;
    uint whiteBackground;
    uint activeTileCount;
    uint gaussianCount;
};

struct GaussianHeader {
    uint offset;
    uint count;
};

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
    float r = quat.x;
    float x = quat.y;
    float y = quat.z;
    float z = quat.w;
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

inline float3x3 buildCovariance3D(float3 scale, float4 quat) {
    float3x3 scaleMat = float3x3(
        float3(scale.x, 0.0f, 0.0f),
        float3(0.0f, scale.y, 0.0f),
        float3(0.0f, 0.0f, scale.z)
    );
    float4 normalized = normalizeQuaternion(quat);
    float3x3 rotation = quaternionToMatrix(normalized);
    float3x3 composed = rotation * scaleMat;
    return composed * transpose(composed);
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
    float tz = safeDepthComponent(viewPos.z);
    float tanHalfY = height / max(2.0f * max(focalY, 1e-4f), 1e-4f);
    float tanHalfX = width / max(2.0f * max(focalX, 1e-4f), 1e-4f);
    float clampX = clamp(tz, -tanHalfX * 1.3f, tanHalfX * 1.3f);
    float clampY = clamp(tz, -tanHalfY * 1.3f, tanHalfY * 1.3f);
    float tx = viewPos.x / clampX * tz;
    float ty = viewPos.y / clampY * tz;
    float invTZ = 1.0f / tz;
    float invTZ2 = invTZ * invTZ;
    float3 row0 = float3(focalX * invTZ, 0.0f, -tx * invTZ2 * focalX);
    float3 row1 = float3(0.0f, focalY * invTZ, -ty * invTZ2 * focalY);
    float3 row2 = float3(0.0f);
    float3x3 J = transpose(matrixFromRows(row0, row1, row2));
    float3x3 temp = J * viewRotation;
    temp = temp * cov3d;
    temp = temp * transpose(viewRotation);
    float3x3 covFull = temp * transpose(J);
    float2x2 cov2d = float2x2(
        covFull[0][0], covFull[0][1],
        covFull[1][0], covFull[1][1]
    );
    cov2d[0][0] += 0.3f;
    cov2d[1][1] += 0.3f;
    return cov2d;
}

inline float4 invertCovariance2D(float2x2 cov) {
    float det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    float invDet = 1.0f / max(det, 1e-8f);
    float m00 = cov[1][1] * invDet;
    float m01 = -cov[0][1] * invDet;
    float m11 = cov[0][0] * invDet;
    return float4(m00, m01, m11, 0.0f);
}

inline float radiusFromCovariance(float2x2 cov) {
    float det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    float mid = 0.5f * (cov[0][0] + cov[1][1]);
    float delta = max(mid * mid - det, 1e-5f);
    float sqrtDelta = sqrt(delta);
    float lam1 = mid + sqrtDelta;
    float lam2 = mid - sqrtDelta;
    float maxCov = max(lam1, lam2);
    float radius = sqrt(max(maxCov, 1e-5f));
    return 3.0f * ceil(radius);
}

// =============================================================================
// Templated projection kernel - single implementation for all precision combos
// =============================================================================
// InPos3: packed_float3 or packed_half3 (world positions)
// InT: float or half (world scalars: opacities, harmonics)
// InQuat: float4 or half4 (quaternions)
// OutVec2: float2 or half2 (screen means)
// OutVec4: float4 or half4 (conics)
// OutT: float or half (output scalars: opacities, depths, colors)
// OutPacked3: packed_float3 or packed_half3 (output colors)
template <typename InPos3, typename InT, typename InQuat,
          typename OutVec2, typename OutVec4, typename OutT, typename OutPacked3>
kernel void projectGaussiansImpl(
    const device InPos3* positions [[buffer(0)]],
    const device InPos3* scales [[buffer(1)]],
    const device InQuat* rotations [[buffer(2)]],
    const device InT* opacities [[buffer(3)]],
    const device InT* harmonics [[buffer(4)]],
    device OutVec2* outMeans [[buffer(5)]],
    device OutVec4* outConics [[buffer(6)]],
    device OutPacked3* outColors [[buffer(7)]],
    device OutT* outOpacities [[buffer(8)]],
    device OutT* outDepths [[buffer(9)]],
    device float* outRadii [[buffer(10)]],  // Always float for tile bounds accuracy
    device uchar* outMask [[buffer(11)]],
    constant CameraUniforms& camera [[buffer(12)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= camera.gaussianCount) { return; }

    // Read input and promote to float for computation accuracy
    float3 position = float3(positions[gid]);
    float4 viewPos4 = camera.viewMatrix * float4(position, 1.0f);
    float depth = safeDepthComponent(viewPos4.z);

    if (depth <= camera.nearPlane) {
        outMask[gid] = 0;
        outRadii[gid] = 0.0f;
        return;
    }

    float4 clip = camera.projectionMatrix * viewPos4;
    if (fabs(clip.w) < 1e-6f) {
        outMask[gid] = 2;
        outRadii[gid] = 0.0f;
        return;
    }

    float ndcX = clip.x / clip.w;
    float ndcY = clip.y / clip.w;
    float px = ((ndcX + 1.0f) * camera.width - 1.0f) * 0.5f;
    float py = ((ndcY + 1.0f) * camera.height - 1.0f) * 0.5f;

    float3 scale = float3(scales[gid]);
    float4 quat = float4(rotations[gid]);
    float3x3 cov3d = buildCovariance3D(scale, quat);

    float3 viewRow0 = float3(camera.viewMatrix[0][0], camera.viewMatrix[1][0], camera.viewMatrix[2][0]);
    float3 viewRow1 = float3(camera.viewMatrix[0][1], camera.viewMatrix[1][1], camera.viewMatrix[2][1]);
    float3 viewRow2 = float3(camera.viewMatrix[0][2], camera.viewMatrix[1][2], camera.viewMatrix[2][2]);
    float3x3 viewRot = matrixFromRows(viewRow0, viewRow1, viewRow2);

    float2x2 cov2d = projectCovariance(cov3d, viewPos4.xyz, viewRot, camera.focalX, camera.focalY, camera.width, camera.height);
    float4 conic = invertCovariance2D(cov2d);
    float radius = radiusFromCovariance(cov2d);

    // Compute SH color
    float3 color = float3(0.0f);
    if (camera.shComponents == 0) {
        uint baseColor = gid * 3u;
        color = float3(float(harmonics[baseColor + 0]), float(harmonics[baseColor + 1]), float(harmonics[baseColor + 2]));
    } else {
        float3 dir = normalize(camera.cameraCenter - position);
        float shBasis[16];
        shBasis[0] = SH_C0;
        shBasis[1] = -SH_C1 * dir.y;
        shBasis[2] = SH_C1 * dir.z;
        shBasis[3] = -SH_C1 * dir.x;
        float xx = dir.x * dir.x, yy = dir.y * dir.y, zz = dir.z * dir.z;
        float xy = dir.x * dir.y, yz = dir.y * dir.z, xz = dir.x * dir.z;
        shBasis[4] = SH_C2_0 * xy;
        shBasis[5] = SH_C2_1 * yz;
        shBasis[6] = SH_C2_2 * (2.0f * zz - xx - yy);
        shBasis[7] = SH_C2_3 * xz;
        shBasis[8] = SH_C2_4 * (xx - yy);
        shBasis[9] = SH_C3_0 * dir.y * (3.0f * xx - yy);
        shBasis[10] = SH_C3_1 * xy * dir.z;
        shBasis[11] = SH_C3_2 * dir.y * (4.0f * zz - xx - yy);
        shBasis[12] = SH_C3_3 * dir.z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
        shBasis[13] = SH_C3_4 * dir.x * (4.0f * zz - xx - yy);
        shBasis[14] = SH_C3_5 * dir.z * (xx - yy);
        shBasis[15] = SH_C3_6 * dir.x * (xx - 3.0f * yy);

        uint coeffs = camera.shComponents;
        uint base = gid * coeffs * 3u;
        for (uint i = 0; i < min(coeffs, 16u); ++i) {
            color.x += float(harmonics[base + i]) * shBasis[i];
            color.y += float(harmonics[base + coeffs + i]) * shBasis[i];
            color.z += float(harmonics[base + coeffs * 2u + i]) * shBasis[i];
        }
    }
    color = max(color + float3(0.5f), float3(0.0f));

    // Write outputs (convert to output precision)
    outMeans[gid] = OutVec2(px, py);
    outConics[gid] = OutVec4(conic);
    outOpacities[gid] = OutT(opacities[gid]);
    outDepths[gid] = OutT(depth);
    outRadii[gid] = radius;  // Always float
    outColors[gid] = OutPacked3(color);
    outMask[gid] = 1;
}

// Instantiation macro for projectGaussians variants
#define instantiate_projectGaussians(name, InPos3, InT, InQuat, OutVec2, OutVec4, OutT, OutPacked3) \
    template [[host_name("projectGaussians_" #name)]] \
    kernel void projectGaussiansImpl<InPos3, InT, InQuat, OutVec2, OutVec4, OutT, OutPacked3>( \
        const device InPos3* positions [[buffer(0)]], \
        const device InPos3* scales [[buffer(1)]], \
        const device InQuat* rotations [[buffer(2)]], \
        const device InT* opacities [[buffer(3)]], \
        const device InT* harmonics [[buffer(4)]], \
        device OutVec2* outMeans [[buffer(5)]], \
        device OutVec4* outConics [[buffer(6)]], \
        device OutPacked3* outColors [[buffer(7)]], \
        device OutT* outOpacities [[buffer(8)]], \
        device OutT* outDepths [[buffer(9)]], \
        device float* outRadii [[buffer(10)]], \
        device uchar* outMask [[buffer(11)]], \
        constant CameraUniforms& camera [[buffer(12)]], \
        uint gid [[thread_position_in_grid]] \
    );

// Float input -> Float output (standard path)
instantiate_projectGaussians(float, packed_float3, float, float4, float2, float4, float, packed_float3)

// Float input -> Half output (for half pipeline with float world data)
instantiate_projectGaussians(half, packed_float3, float, float4, half2, half4, half, packed_half3)

// Half input -> Half output (full half pipeline)
instantiate_projectGaussians(half_input, packed_half3, half, half4, half2, half4, half, packed_half3)

#undef instantiate_projectGaussians

template <typename T, typename Vec2, typename Vec4, typename Packed3>
kernel void renderTiles(
    const device GaussianHeader* headers [[buffer(0)]],
    const device Vec2* means [[buffer(1)]],
    const device Vec4* conics [[buffer(2)]],
    const device Packed3* colors [[buffer(3)]],
    const device T* opacities [[buffer(4)]],
    const device T* depths [[buffer(5)]],
    device float* colorOut [[buffer(6)]],
    device float* depthOut [[buffer(7)]],
    device float* alphaOut [[buffer(8)]],
    constant RenderParams& params [[buffer(9)]],
    const device uint* activeTiles [[buffer(10)]],
    const device uint* activeTileCount [[buffer(11)]],
    uint3 localPos3 [[thread_position_in_threadgroup]],
    uint3 tileCoord [[threadgroup_position_in_grid]]
) {
    uint activeCount = activeTileCount[0];
    uint activeIdx = tileCoord.x;
    if (activeIdx >= activeCount) { return; }
    uint tileId = activeTiles[activeIdx];
    uint tilesX = params.tilesX;
    uint tileWidth = params.tileWidth;
    uint tileHeight = params.tileHeight;
    uint localX = localPos3.x;
    uint localY = localPos3.y;
    uint tileX = tileId % tilesX;
    uint tileY = tileId / tilesX;
    uint px = tileX * tileWidth + localX;
    uint py = tileY * tileHeight + localY;
    bool inBounds = (px < params.width) && (py < params.height);
    float3 accumColor = float3(0.0f);
    float accumDepth = 0.0f;
    float accumAlpha = 0.0f;
    float trans = 1.0f;
    GaussianHeader header = headers[tileId];
    uint start = header.offset;
    uint count = header.count;
    if (count == 0) { return; }
    for (uint i = 0; i < count; ++i) {
        uint gIdx = start + i;
        if (inBounds) {
            float2 mean = float2(means[gIdx]);
            float4 conic = float4(conics[gIdx]);
            float3 color = float3(colors[gIdx]);
            float baseOpacity = metal::min(float(opacities[gIdx]), 0.99f);
            if (baseOpacity > 0.0f) {
                float fx = float(px);
                float fy = float(py);
                float dx = fx - mean.x;
                float dy = fy - mean.y;
                float quad = dx * dx * conic.x + dy * dy * conic.z + 2.0f * dx * dy * conic.y;
                if (quad < 20.0f && (conic.x != 0.0f || conic.z != 0.0f)) {
                    float weight = metal::exp(-0.5f * quad);
                    float alpha = weight * baseOpacity;
                    if (alpha > 1e-4f) {
                        float contrib = trans * alpha;
                        trans *= (1.0f - alpha);
                        accumAlpha += contrib;
                        accumColor += color * contrib;
                        accumDepth += float(depths[gIdx]) * contrib;
                        if (trans < 1e-3f) { break; }
                    }
                }
            }
        }
    }
    if (inBounds) {
        if (params.whiteBackground != 0) {
            accumColor += float3(trans);
        }
        uint pixelIndex = py * params.width + px;
        uint base = pixelIndex * 3;
        colorOut[base + 0] = accumColor.x;
        colorOut[base + 1] = accumColor.y;
        colorOut[base + 2] = accumColor.z;
        depthOut[pixelIndex] = accumDepth;
        alphaOut[pixelIndex] = accumAlpha;
    }
}

#define instantiate_renderTiles(name, T, Vec2, Vec4, Packed3) \
    template [[host_name("renderTiles_" #name)]] \
    kernel void renderTiles<T, Vec2, Vec4, Packed3>( \
        const device GaussianHeader* headers [[buffer(0)]], \
        const device Vec2* means [[buffer(1)]], \
        const device Vec4* conics [[buffer(2)]], \
        const device Packed3* colors [[buffer(3)]], \
        const device T* opacities [[buffer(4)]], \
        const device T* depths [[buffer(5)]], \
        device float* colorOut [[buffer(6)]], \
        device float* depthOut [[buffer(7)]], \
        device float* alphaOut [[buffer(8)]], \
        constant RenderParams& params [[buffer(9)]], \
        const device uint* activeTiles [[buffer(10)]], \
        const device uint* activeTileCount [[buffer(11)]], \
        uint3 localPos3 [[thread_position_in_threadgroup]], \
        uint3 tileCoord [[threadgroup_position_in_grid]] \
    );

instantiate_renderTiles(float, float, float2, float4, packed_float3)
instantiate_renderTiles(half, half, half2, half4, packed_half3)

#undef instantiate_renderTiles

// Texture-based rendering template
template <typename T, typename Vec2, typename Vec4, typename Packed3>
kernel void renderTilesDirect(
    const device GaussianHeader* headers [[buffer(0)]],
    const device Vec2* means [[buffer(1)]],
    const device Vec4* conics [[buffer(2)]],
    const device Packed3* colors [[buffer(3)]],
    const device T* opacities [[buffer(4)]],
    const device T* depths [[buffer(5)]],
    texture2d<float, access::write> colorTex [[texture(0)]],
    texture2d<float, access::write> depthTex [[texture(1)]],
    texture2d<float, access::write> alphaTex [[texture(2)]],
    constant RenderParams& params [[buffer(9)]],
    const device uint* activeTiles [[buffer(10)]],
    const device uint* activeTileCount [[buffer(11)]],
    uint3 localPos3 [[thread_position_in_threadgroup]],
    uint3 tileCoord [[threadgroup_position_in_grid]]
) {
    uint activeCount = activeTileCount[0];
    uint activeIdx = tileCoord.x;
    if (activeIdx >= activeCount) { return; }
    uint tileId = activeTiles[activeIdx];
    uint tilesX = params.tilesX;
    uint tileWidth = params.tileWidth;
    uint tileHeight = params.tileHeight;
    uint localX = localPos3.x;
    uint localY = localPos3.y;
    uint tileX = tileId % tilesX;
    uint tileY = tileId / tilesX;
    uint px = tileX * tileWidth + localX;
    uint py = tileY * tileHeight + localY;
    bool inBounds = (px < params.width) && (py < params.height);
    float3 accumColor = float3(0.0f);
    float accumDepth = 0.0f;
    float accumAlpha = 0.0f;
    float trans = 1.0f;
    GaussianHeader header = headers[tileId];
    uint start = header.offset;
    uint count = header.count;
    
    if (count > 0) {
        for (uint i = 0; i < count; ++i) {
            uint gIdx = start + i;
            if (inBounds) {
                float2 mean = float2(means[gIdx]);
                float4 conic = float4(conics[gIdx]);
                float3 color = float3(colors[gIdx]);
                float baseOpacity = metal::min(float(opacities[gIdx]), 0.99f);
                if (baseOpacity > 0.0f) {
                    float fx = float(px);
                    float fy = float(py);
                    float dx = fx - mean.x;
                    float dy = fy - mean.y;
                    float quad = dx * dx * conic.x + dy * dy * conic.z + 2.0f * dx * dy * conic.y;
                    if (quad < 20.0f && (conic.x != 0.0f || conic.z != 0.0f)) {
                        float weight = metal::exp(-0.5f * quad);
                        float alpha = weight * baseOpacity;
                        if (alpha > 1e-4f) {
                            float contrib = trans * alpha;
                            trans *= (1.0f - alpha);
                            accumAlpha += contrib;
                            accumColor += color * contrib;
                            accumDepth += float(depths[gIdx]) * contrib;
                            if (trans < 1e-3f) { break; }
                        }
                    }
                }
            }
        }
    }
    
    if (inBounds) {
        if (params.whiteBackground != 0) {
            accumColor += float3(trans);
        }
        
        uint2 coords = uint2(px, py);
        colorTex.write(float4(accumColor, 1.0f), coords);
        depthTex.write(float4(accumDepth, 0, 0, 0), coords); 
        alphaTex.write(float4(accumAlpha, 0, 0, 0), coords);
    }
}

#define instantiate_renderTilesDirect(name, T, Vec2, Vec4, Packed3) \
    template [[host_name("renderTilesDirect_" #name)]] \
    kernel void renderTilesDirect<T, Vec2, Vec4, Packed3>( \
        const device GaussianHeader* headers [[buffer(0)]], \
        const device Vec2* means [[buffer(1)]], \
        const device Vec4* conics [[buffer(2)]], \
        const device Packed3* colors [[buffer(3)]], \
        const device T* opacities [[buffer(4)]], \
        const device T* depths [[buffer(5)]], \
        texture2d<float, access::write> colorTex [[texture(0)]], \
        texture2d<float, access::write> depthTex [[texture(1)]], \
        texture2d<float, access::write> alphaTex [[texture(2)]], \
        constant RenderParams& params [[buffer(9)]], \
        const device uint* activeTiles [[buffer(10)]], \
        const device uint* activeTileCount [[buffer(11)]], \
        uint3 localPos3 [[thread_position_in_threadgroup]], \
        uint3 tileCoord [[threadgroup_position_in_grid]] \
    );

instantiate_renderTilesDirect(float, float, float2, float4, packed_float3)
instantiate_renderTilesDirect(half, half, half2, half4, packed_half3)

#undef instantiate_renderTilesDirect

struct ClearParams {
    uint pixelCount;
    uint whiteBackground;
};

kernel void clearRenderTargetsKernel(
    device float* colorOut [[buffer(0)]],
    device float* depthOut [[buffer(1)]],
    device float* alphaOut [[buffer(2)]],
    constant ClearParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.pixelCount) {
        return;
    }
    float bg = (params.whiteBackground != 0u) ? 1.0f : 0.0f;
    uint base = gid * 3u;
    colorOut[base + 0] = bg;
    colorOut[base + 1] = bg;
    colorOut[base + 2] = bg;
    depthOut[gid] = 0.0f;
    alphaOut[gid] = 0.0f;
}

struct ClearTextureParams {
    uint width;
    uint height;
    uint whiteBackground;
};

kernel void clearRenderTexturesKernel(
    texture2d<float, access::write> colorTex [[texture(0)]],
    texture2d<float, access::write> depthTex [[texture(1)]],
    texture2d<float, access::write> alphaTex [[texture(2)]],
    constant ClearTextureParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    float bg = (params.whiteBackground != 0u) ? 1.0f : 0.0f;
    colorTex.write(float4(bg, bg, bg, 1.0f), gid);
    depthTex.write(float4(0.0f), gid);
    alphaTex.write(float4(0.0f), gid);
}

struct TileBoundsParams {
    uint width;
    uint height;
    uint tileWidth;
    uint tileHeight;
    uint tilesX;
    uint tilesY;
    uint gaussianCount;
};

struct CoverageParams {
    uint gaussianCount;
};

struct ScatterParams {
    uint gaussianCount;
    uint tilesX;
};

struct ScatterDispatchParams {
    uint threadgroupWidth;
    uint gaussianCount;
};

struct TileAssignmentHeader {
    uint totalAssignments;
    uint maxAssignments;
    uint paddedCount;
    uint overflow;
};

struct SortKeyParams {
    uint maxAssignments;
    uint totalAssignments;
};

struct DispatchIndirectArgs {
    uint threadgroupsPerGridX;
    uint threadgroupsPerGridY;
    uint threadgroupsPerGridZ;
};

struct AssignmentDispatchConfig {
    uint sortThreadgroupSize;
    uint fuseThreadgroupSize;
    uint unpackThreadgroupSize;
    uint packThreadgroupSize;
    uint bitonicThreadgroupSize;
    uint radixBlockSize;
    uint radixGrainSize;
};

enum DispatchSlots {
    DispatchSlotSortKeys = 0,
    DispatchSlotFuseKeys = 1,
    DispatchSlotUnpackKeys = 2,
    DispatchSlotPack = 3,
    DispatchSlotBitonicFirst = 4,
    DispatchSlotBitonicGeneral = 5,
    DispatchSlotBitonicFinal = 6,
    DispatchSlotRadixHistogram = 7,
    DispatchSlotRadixScanBlocks = 8,
    DispatchSlotRadixExclusive = 9,
    DispatchSlotRadixApply = 10,
    DispatchSlotRadixScatter = 11,
    DispatchSlotRenderTiles = 12
};

struct BitonicParams {
    uint j;
    uint k;
    uint total;
};

struct HeaderFromSortedParams {
    uint tileCount;
    uint totalAssignments;
};

static inline uint nextPowerOfTwo(uint value) {
    value = max(value, 1u);
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value++;
    return value;
}

template <typename MeansT>
kernel void tileBoundsKernel(
    const device MeansT* means [[buffer(0)]],
    const device float* radii [[buffer(1)]],
    const device uchar* mask [[buffer(2)]],
    device int4* bounds [[buffer(3)]],
    constant TileBoundsParams& params [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.gaussianCount) {
        return;
    }
    if (mask[idx] == 0) {
        bounds[idx] = int4(0, -1, 0, -1);
        return;
    }

    float2 mean = float2(means[idx]);
    float radius = radii[idx];

    float xmin = mean.x - radius;
    float xmax = mean.x + radius;
    float ymin = mean.y - radius;
    float ymax = mean.y + radius;

    float maxW = float(params.width - 1);
    float maxH = float(params.height - 1);

    xmin = clamp(xmin, 0.0f, maxW);
    xmax = clamp(xmax, 0.0f, maxW);
    ymin = clamp(ymin, 0.0f, maxH);
    ymax = clamp(ymax, 0.0f, maxH);

    int minX = (int)floor(xmin / float(params.tileWidth));
    int maxX = (int)ceil(xmax / float(params.tileWidth)) - 1;
    int minY = (int)floor(ymin / float(params.tileHeight));
    int maxY = (int)ceil(ymax / float(params.tileHeight)) - 1;

    minX = max(minX, 0);
    minY = max(minY, 0);
    maxX = min(maxX, int(params.tilesX) - 1);
    maxY = min(maxY, int(params.tilesY) - 1);

    bounds[idx] = int4(minX, maxX, minY, maxY);
}

#define instantiate_tileBoundsKernel(name, MeansT) \
    template [[host_name("tileBoundsKernel_" #name)]] \
    kernel void tileBoundsKernel<MeansT>( \
        const device MeansT* means [[buffer(0)]], \
        const device float* radii [[buffer(1)]], \
        const device uchar* mask [[buffer(2)]], \
        device int4* bounds [[buffer(3)]], \
        constant TileBoundsParams& params [[buffer(4)]], \
        uint idx [[thread_position_in_grid]]);

instantiate_tileBoundsKernel(float, float2)
instantiate_tileBoundsKernel(half, half2)

#undef instantiate_tileBoundsKernel


template <typename DepthT>
kernel void computeSortKeysKernel(
    const device int* tileIds [[buffer(0)]],
    const device int* tileIndices [[buffer(1)]],
    const device DepthT* depths [[buffer(2)]],
    device uint2* sortKeys [[buffer(3)]],
    device int* sortedIndices [[buffer(4)]],
    const device TileAssignmentHeader& header [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint totalAssignments = header.totalAssignments;
    uint padded = header.paddedCount;
    if (gid < padded) {
        if (gid < totalAssignments) {
            int g = tileIndices[gid];
            uint tileId = (uint)tileIds[gid];
            // Convert to float for consistent bit representation when sorting
            float depthFloat = float(depths[g]);
            uint depthBits = as_type<uint>(depthFloat);
            sortKeys[gid] = uint2(tileId, depthBits);
            sortedIndices[gid] = g;
        } else {
            sortKeys[gid] = uint2(0xFFFFFFFFu, 0xFFFFFFFFu);
            sortedIndices[gid] = -1;
        }
    }
}

#define instantiate_computeSortKeysKernel(name, DepthT) \
    template [[host_name("computeSortKeysKernel_" #name)]] \
    kernel void computeSortKeysKernel<DepthT>( \
        const device int* tileIds [[buffer(0)]], \
        const device int* tileIndices [[buffer(1)]], \
        const device DepthT* depths [[buffer(2)]], \
        device uint2* sortKeys [[buffer(3)]], \
        device int* sortedIndices [[buffer(4)]], \
        const device TileAssignmentHeader& header [[buffer(5)]], \
        uint gid [[thread_position_in_grid]]);

instantiate_computeSortKeysKernel(float, float)
instantiate_computeSortKeysKernel(half, half)

#undef instantiate_computeSortKeysKernel

struct PackParams {
    uint totalAssignments;
    uint padding;
};

// Template for packTileDataKernel:
// InT, InVec2, InVec4, InPacked3 - Input types (from gaussian buffers)
// OutT, OutVec2, OutVec4, OutPacked3 - Output types (to packed buffers)
template <typename InT, typename InVec2, typename InVec4, typename InPacked3,
          typename OutT, typename OutVec2, typename OutVec4, typename OutPacked3>
kernel void packTileDataKernel(
    const device int* sortedIndices [[buffer(0)]],
    const device InVec2* means [[buffer(1)]],
    const device InVec4* conics [[buffer(2)]],
    const device InPacked3* colors [[buffer(3)]],
    const device InT* opacities [[buffer(4)]],
    const device InT* depths [[buffer(5)]],
    device OutVec2* outMeans [[buffer(6)]],
    device OutVec4* outConics [[buffer(7)]],
    device OutPacked3* outColors [[buffer(8)]],
    device OutT* outOpacities [[buffer(9)]],
    device OutT* outDepths [[buffer(10)]],
    const device TileAssignmentHeader* header [[buffer(11)]],
    const device int* tileIndices [[buffer(12)]],
    const device int* tileIds [[buffer(13)]],
    constant PackParams& params [[buffer(14)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = header->totalAssignments;
    if (gid >= total) { return; }
    int src = sortedIndices[gid];
    if (src < 0) { return; }
    outMeans[gid] = OutVec2(float2(means[src]));
    outConics[gid] = OutVec4(float4(conics[src]));
    outColors[gid] = OutPacked3(float3(colors[src]));
    outOpacities[gid] = OutT(float(opacities[src]));
    outDepths[gid] = OutT(float(depths[src]));
}

#define instantiate_packTileDataKernel(name, InT, InVec2, InVec4, InPacked3, OutT, OutVec2, OutVec4, OutPacked3) \
    template [[host_name("packTileDataKernel_" #name)]] \
    kernel void packTileDataKernel<InT, InVec2, InVec4, InPacked3, OutT, OutVec2, OutVec4, OutPacked3>( \
        const device int* sortedIndices [[buffer(0)]], \
        const device InVec2* means [[buffer(1)]], \
        const device InVec4* conics [[buffer(2)]], \
        const device InPacked3* colors [[buffer(3)]], \
        const device InT* opacities [[buffer(4)]], \
        const device InT* depths [[buffer(5)]], \
        device OutVec2* outMeans [[buffer(6)]], \
        device OutVec4* outConics [[buffer(7)]], \
        device OutPacked3* outColors [[buffer(8)]], \
        device OutT* outOpacities [[buffer(9)]], \
        device OutT* outDepths [[buffer(10)]], \
        const device TileAssignmentHeader* header [[buffer(11)]], \
        const device int* tileIndices [[buffer(12)]], \
        const device int* tileIds [[buffer(13)]], \
        constant PackParams& params [[buffer(14)]], \
        uint gid [[thread_position_in_grid]] \
    );

// float input -> float output
instantiate_packTileDataKernel(float, float, float2, float4, packed_float3, float, float2, float4, packed_float3)
// half input -> half output
instantiate_packTileDataKernel(half, half, half2, half4, packed_half3, half, half2, half4, packed_half3)

#undef instantiate_packTileDataKernel

kernel void buildHeadersFromSortedKernel(
    const device uint2* sortedKeys [[buffer(0)]],
    device GaussianHeader* headers [[buffer(1)]],
    const device TileAssignmentHeader* headerInfo [[buffer(2)]],
    constant HeaderFromSortedParams& params [[buffer(3)]],
    uint tile [[thread_position_in_grid]]
) {
    if (tile >= params.tileCount) {
        return;
    }
    uint total = headerInfo->totalAssignments;
    if (total == 0) {
        GaussianHeader emptyHeader;
        emptyHeader.offset = 0u;
        emptyHeader.count = 0u;
        headers[tile] = emptyHeader;
        return;
    }

    uint left = 0;
    uint right = total;
    while (left < right) {
        uint mid = (left + right) >> 1;
        uint midTile = sortedKeys[mid].x;
        if (midTile < tile) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    uint start = left;

    left = start;
    right = total;
    while (left < right) {
        uint mid = (left + right) >> 1;
        uint midTile = sortedKeys[mid].x;
        if (midTile <= tile) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    uint end = left;

    GaussianHeader header;
    header.offset = start;
    header.count = end > start ? (end - start) : 0u;
    headers[tile] = header;
}

kernel void compactActiveTilesKernel(
    const device GaussianHeader* headers [[buffer(0)]],
    device uint* activeTiles [[buffer(1)]],
    device atomic_uint* activeCount [[buffer(2)]],
    constant uint& tileCount [[buffer(3)]],
    uint tile [[thread_position_in_grid]]
) {
    if (tile >= tileCount) {
        return;
    }
    if (headers[tile].count > 0u) {
        uint index = atomic_fetch_add_explicit(activeCount, 1u, memory_order_relaxed);
        activeTiles[index] = tile;
    }
}

struct RenderDispatchParams {
    uint tileCount;
    uint totalAssignments;
};

kernel void prepareRenderDispatchKernel(
    const device uint* activeCount [[buffer(0)]],
    device DispatchIndirectArgs* dispatchArgs [[buffer(1)]],
    constant RenderDispatchParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid > 0) {
        return;
    }
    uint count = activeCount[0];
    if (count > params.tileCount) { count = params.tileCount; }
    dispatchArgs[DispatchSlotRenderTiles].threadgroupsPerGridX = count;
    dispatchArgs[DispatchSlotRenderTiles].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotRenderTiles].threadgroupsPerGridZ = 1u;
}

///////////////////////////////////////////////////////////////////////////////
//  Fast bitonic sort (shared-memory first/final pass)
///////////////////////////////////////////////////////////////////////////////

static constexpr int genLeftIndex(const uint position, const uint blockSize) {
    const uint blockMask = blockSize - 1;
    const uint no = position & blockMask;
    return ((position & ~blockMask) << 1) | no;
}

static inline bool bitonicKeyLess(uint2 left, uint2 right) {
    return (left.x < right.x) || ((left.x == right.x) && (left.y < right.y));
}

static void bitonicSwap(bool reverse, threadgroup uint2& keyL, threadgroup int& valL, threadgroup uint2& keyR, threadgroup
   int& valR) {
    bool lt = bitonicKeyLess(keyL, keyR);
    bool swap = (!lt) ^ reverse;
    if (swap) {
        uint2 tk = keyL; keyL = keyR; keyR = tk;
        int tv = valL; valL = valR; valR = tv;
    }
}

static void loadShared(const uint threadGroupSize, const uint indexInThreadgroup, const uint position,
                       device uint2* keys, device int* values,
                       threadgroup uint2* sharedKeys, threadgroup int* sharedVals) {
    const uint index = genLeftIndex(position, threadGroupSize);
    sharedKeys[indexInThreadgroup] = keys[index];
    sharedVals[indexInThreadgroup] = values[index];

    uint index2 = index | threadGroupSize;
    sharedKeys[indexInThreadgroup | threadGroupSize] = keys[index2];
    sharedVals[indexInThreadgroup | threadGroupSize] = values[index2];

}

static void storeShared(const uint threadGroupSize, const uint indexInThreadgroup, const uint position,
                        device uint2* keys, device int* values,
                        threadgroup uint2* sharedKeys, threadgroup int* sharedVals) {
    const uint index = genLeftIndex(position, threadGroupSize);
    keys[index] = sharedKeys[indexInThreadgroup];
    values[index] = sharedVals[indexInThreadgroup];

    uint index2 = index | threadGroupSize;
    keys[index2] = sharedKeys[indexInThreadgroup | threadGroupSize];
    values[index2] = sharedVals[indexInThreadgroup | threadGroupSize];

}

kernel void bitonicSortFirstPass(
    device uint2* keys [[buffer(0)]],
    device int* values [[buffer(1)]],
    const device TileAssignmentHeader* header [[buffer(2)]],
    constant uint& logicalBlockSize [[buffer(3)]],
    threadgroup uint2* sharedKeys [[threadgroup(0)]],
    threadgroup int* sharedVals [[threadgroup(1)]],
    const uint threadgroupSize [[threads_per_threadgroup]],
    const uint indexInThreadgroup [[thread_index_in_threadgroup]],
    const uint position [[thread_position_in_grid]]
) {
    uint gridSize = header->paddedCount;
    if (position >= gridSize) { return; }

    loadShared(threadgroupSize, indexInThreadgroup, position, keys, values, sharedKeys, sharedVals);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint unitSize = 1; unitSize <= logicalBlockSize; unitSize <<= 1) {
        bool reverse = (position & unitSize) != 0;
        for (uint blockSize = unitSize; blockSize > 0; blockSize >>= 1) {
            const uint left = genLeftIndex(indexInThreadgroup, blockSize);
            bitonicSwap(reverse, sharedKeys[left], sharedVals[left], sharedKeys[left | blockSize], sharedVals[left | blockSize]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    storeShared(threadgroupSize, indexInThreadgroup, position, keys, values, sharedKeys, sharedVals);
}

kernel void bitonicSortGeneralPass(
    device uint2* keys [[buffer(0)]],
    device int* values [[buffer(1)]],
    const device TileAssignmentHeader* header [[buffer(2)]],
    constant uint2& params [[buffer(3)]],
    const uint position [[thread_position_in_grid]]
) {
    uint gridSize = header->paddedCount;
    if (position >= gridSize) { return; }

    bool reverse = (position & (params.x >> 1)) != 0;
    uint blockSize = params.y;
    const uint left = genLeftIndex(position, blockSize);

    uint2 keyL = keys[left];
    uint2 keyR = keys[left | blockSize];
    int valL = values[left];
    int valR = values[left | blockSize];

    bool lt = bitonicKeyLess(keyL, keyR);
    bool swap = (!lt) ^ reverse;

    if (swap) {
        keys[left] = keyR;
        keys[left | blockSize] = keyL;
        values[left] = valR;
        values[left | blockSize] = valL;
    }
}

kernel void bitonicSortFinalPass(
    device uint2* keys [[buffer(0)]],
    device int* values [[buffer(1)]],
    const device TileAssignmentHeader* header [[buffer(2)]],
    constant uint2& params [[buffer(3)]],
    threadgroup uint2* sharedKeys [[threadgroup(0)]],
    threadgroup int* sharedVals [[threadgroup(1)]],
    const uint threadgroupSize [[threads_per_threadgroup]],
    const uint indexInThreadgroup [[thread_index_in_threadgroup]],
    const uint position [[thread_position_in_grid]]
) {
    uint gridSize = header->paddedCount;
    if (position >= gridSize) { return; }

    loadShared(threadgroupSize, indexInThreadgroup, position, keys, values, sharedKeys, sharedVals);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint unitSize = params.x;
    uint blockSize = params.y;
    bool reverse = (position & (unitSize >> 1)) != 0;

    for (uint width = blockSize; width > 0; width >>= 1) {
        const uint left = genLeftIndex(indexInThreadgroup, width);
        bitonicSwap(reverse, sharedKeys[left], sharedVals[left], sharedKeys[left | width], sharedVals[left | width]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    storeShared(threadgroupSize, indexInThreadgroup, position, keys, values, sharedKeys, sharedVals);
}

struct BitonicICBContainer {
    command_buffer commandBuffer [[ id(0) ]];
};

kernel void encodeBitonicICB(
    constant BitonicICBContainer& icb_container [[buffer(0)]],
    const device TileAssignmentHeader* header [[buffer(1)]],
    const device DispatchIndirectArgs* dispatchArgs [[buffer(2)]],
    const device uint* passTypes [[buffer(3)]], // 0=first,1=general,2=final,3=unused
    constant uint& unitSize [[buffer(4)]],
    constant uint& maxCommands [[buffer(5)]],
    constant uint& firstOffset [[buffer(6)]],
    constant uint& generalOffset [[buffer(7)]],
    constant uint& finalOffset [[buffer(8)]],
    device uint* firstParams [[buffer(9)]],
    device uint4* debugParams [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint firstStride = 256u;
    if (tid != 0) { return; }

    uint pc = max(1u, header[0].paddedCount);
    uint m = 31u - clz(pc);
    uint u = 31u - clz(unitSize);
    uint g = (m > u) ? (m - u) : 0u;
    uint passesNeeded = 1u + g + (g * (g - 1u)) / 2u;
    passesNeeded = min(passesNeeded, maxCommands);

    // First-pass logical block size depends on the actual padded count.
    if (firstParams != nullptr && maxCommands > 0) {
        uint logical = min(unitSize, pc / 2u);
        device uint* firstPtr = (device uint*)((device char*)firstParams + 0u * firstStride);
        *firstPtr = max(1u, logical);
    }

    const device DispatchIndirectArgs* firstArgs = (const device DispatchIndirectArgs*)((const device char*)dispatchArgs + firstOffset);
    const device DispatchIndirectArgs* generalArgs = (const device DispatchIndirectArgs*)((const device char*)dispatchArgs + generalOffset);
    const device DispatchIndirectArgs* finalArgs = (const device DispatchIndirectArgs*)((const device char*)dispatchArgs + finalOffset);

    uint firstGroupsX = firstArgs->threadgroupsPerGridX;
    uint generalGroupsX = generalArgs->threadgroupsPerGridX;
    uint finalGroupsX = finalArgs->threadgroupsPerGridX;

    uint3 tgSize(unitSize, 1, 1);

    for (uint slot = 0; slot < maxCommands; ++slot) {
        compute_command cmd(icb_container.commandBuffer, slot);
        uint3 tgCount(0, 1, 1);

        if (slot < passesNeeded) {
            uint t = passTypes[slot];
            if (t == 0u) {
                tgCount = uint3(firstGroupsX, 1, 1);
            } else if (t == 1u) {
                tgCount = uint3(generalGroupsX, 1, 1);
            } else if (t == 2u) {
                tgCount = uint3(finalGroupsX, 1, 1);
            }
            if (debugParams != nullptr && slot < maxCommands) {
                debugParams[slot] = uint4(t, pc, passesNeeded, tgCount.x);
            }
        } else if (debugParams != nullptr && slot < maxCommands) {
            debugParams[slot] = uint4(3u, 0u, passesNeeded, 0u);
        }

        cmd.concurrent_dispatch_threadgroups(tgCount, tgSize);
    }
}


///////////////////////////////////////////////////////////////////////////////
//  Radix sort helpers (ported from mlx/Radix)
///////////////////////////////////////////////////////////////////////////////

template <typename OpacityT>
kernel void gaussianCoverageKernel(
    const device int4* bounds [[buffer(0)]],
    device uint* coverageCounts [[buffer(1)]],
    const device OpacityT* opacities [[buffer(2)]],
    constant CoverageParams& params [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.gaussianCount) {
        return;
    }
    int4 rect = bounds[idx];
    // Cull negligible opacities
    float op = float(opacities[idx]);
    if (op < 1e-4f) {
        coverageCounts[idx] = 0u;
        return;
    }
    // Coverage is per-tile, not per-pixel.
    int width = max(0, rect.y - rect.x + 1);
    int height = max(0, rect.w - rect.z + 1);
    uint count = (width > 0 && height > 0) ? uint(width * height) : 0u;
    coverageCounts[idx] = count;
}

#define instantiate_gaussianCoverageKernel(name, OpacityT) \
    template [[host_name("gaussianCoverageKernel_" #name)]] \
    kernel void gaussianCoverageKernel<OpacityT>( \
        const device int4* bounds [[buffer(0)]], \
        device uint* coverageCounts [[buffer(1)]], \
        const device OpacityT* opacities [[buffer(2)]], \
        constant CoverageParams& params [[buffer(3)]], \
        uint idx [[thread_position_in_grid]]);

instantiate_gaussianCoverageKernel(float, float)
instantiate_gaussianCoverageKernel(half, half)

#undef instantiate_gaussianCoverageKernel

kernel void coveragePrefixScanKernel(
    const device uint* input_data [[buffer(0)]],
    device uint* output_data [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    device uint* partial_sums [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    ushort local_id [[thread_position_in_threadgroup]]
) {
    uint elements_per_group = TILE_PREFIX_BLOCK_SIZE * TILE_PREFIX_GRAIN_SIZE;
    uint base_id = group_id * elements_per_group;
    if (base_id >= count) {
        return;
    }

    uint values[TILE_PREFIX_GRAIN_SIZE];
    LoadBlockedLocalFromGlobal<TILE_PREFIX_GRAIN_SIZE>(values, &input_data[base_id], local_id, count, 0u);

    uint aggregate = ThreadPrefixExclusiveSum<TILE_PREFIX_GRAIN_SIZE>(values);

    threadgroup uint scratch[TILE_PREFIX_BLOCK_SIZE];
    uint prefix = ThreadgroupRakingPrefixExclusiveSum<TILE_PREFIX_BLOCK_SIZE, uint>(aggregate, scratch, local_id);

    if (local_id == TILE_PREFIX_BLOCK_SIZE - 1) {
        partial_sums[group_id] = aggregate + prefix;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ThreadUniformAdd<TILE_PREFIX_GRAIN_SIZE>(values, prefix);
    StoreBlockedLocalToGlobal<TILE_PREFIX_GRAIN_SIZE>(&output_data[base_id], values, local_id, count);
}

kernel void coverageScanPartialSumsKernel(
    device uint* partial_sums [[buffer(0)]],
    constant uint& num_partial_sums [[buffer(1)]],
    ushort local_id [[thread_position_in_threadgroup]]
) {
    threadgroup uint shared_mem[TILE_PREFIX_BLOCK_SIZE];
    uint prefix = 0u;
    uint chunkCount = (num_partial_sums + TILE_PREFIX_BLOCK_SIZE - 1u) / TILE_PREFIX_BLOCK_SIZE;
    for (uint chunk = 0; chunk < chunkCount; ++chunk) {
        uint idx = chunk * TILE_PREFIX_BLOCK_SIZE + local_id;
        uint value = (idx < num_partial_sums) ? partial_sums[idx] : 0u;
        shared_mem[local_id] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        ThreadgroupRakingPrefixExclusiveSum<TILE_PREFIX_BLOCK_SIZE, uint>(shared_mem[local_id], shared_mem, local_id);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (idx < num_partial_sums) {
            partial_sums[idx] = shared_mem[local_id] + prefix;
        }
        if (local_id == TILE_PREFIX_BLOCK_SIZE - 1) {
            uint inclusiveLast = shared_mem[local_id] + value;
            shared_mem[local_id] = inclusiveLast;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint blockTotal = shared_mem[TILE_PREFIX_BLOCK_SIZE - 1];
        prefix += blockTotal;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void coverageFinalizeScanKernel(
    device uint* output_data [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    const device uint* partial_sums [[buffer(2)]],
    uint group_id [[threadgroup_position_in_grid]],
    ushort local_id [[thread_position_in_threadgroup]]
) {
    uint elements_per_group = TILE_PREFIX_BLOCK_SIZE * TILE_PREFIX_GRAIN_SIZE;
    uint base_id = group_id * elements_per_group;
    if (base_id >= count) {
        return;
    }

    uint values[TILE_PREFIX_GRAIN_SIZE];
    LoadBlockedLocalFromGlobal<TILE_PREFIX_GRAIN_SIZE>(values, &output_data[base_id], local_id, count, 0u);

    uint prefix = partial_sums[group_id];
    ThreadUniformAdd<TILE_PREFIX_GRAIN_SIZE>(values, prefix);

    StoreBlockedLocalToGlobal<TILE_PREFIX_GRAIN_SIZE>(&output_data[base_id], values, local_id, count);
}

kernel void coverageStoreTotalKernel(
    const device uint* coverageCounts [[buffer(0)]],
    device uint* offsets [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    device TileAssignmentHeader* header [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid > 0 || count == 0) {
        if (tid == 0 && count == 0) {
            offsets[0] = 0u;
            header[0].totalAssignments = 0u;
        }
        return;
    }
    uint lastIndex = count - 1;
    uint total = offsets[lastIndex] + coverageCounts[lastIndex];
    offsets[count] = total;
    bool overflow = total > header[0].maxAssignments;
    uint cappedTotal = overflow ? header[0].maxAssignments : total;
    header[0].totalAssignments = cappedTotal;
    uint padded = nextPowerOfTwo(cappedTotal);
    if (padded == 0u) { padded = 1u; }
    header[0].paddedCount = padded;
    header[0].overflow = overflow ? 1u : 0u;
}

kernel void scatterDispatchKernel(
    const device uint* offsets [[buffer(0)]],
    device uint* dispatchArgs [[buffer(1)]],
    constant ScatterDispatchParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) {
        return;
    }
    uint totalAssignments = (params.gaussianCount > 0) ? offsets[params.gaussianCount] : 0u;
    uint width = max(params.threadgroupWidth, 1u);
    uint groups = (width > 0u) ? ((totalAssignments + width - 1u) / width) : 0u;
    dispatchArgs[0] = groups;
    dispatchArgs[1] = 1u;
    dispatchArgs[2] = 1u;
}

kernel void scatterAssignmentsKernel(
    const device int4* bounds [[buffer(0)]],
    const device uint* offsets [[buffer(1)]],
    device int* tileIndices [[buffer(2)]],
    device int* tileIds [[buffer(3)]],
    constant ScatterParams& params [[buffer(4)]],
    const device TileAssignmentHeader* header [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.gaussianCount) {
        return;
    }
    int4 rect = bounds[idx];
    int minX = rect.x;
    int maxX = rect.y;
    int minY = rect.z;
    int maxY = rect.w;
    if (maxX < minX || maxY < minY) {
        return;
    }

    uint writeIndex = offsets[idx];
    uint tilesX = params.tilesX;
    uint maxAssignments = header[0].maxAssignments;

    for (int ty = minY; ty <= maxY; ++ty) {
        int rowBase = ty * int(tilesX);
        for (int tx = minX; tx <= maxX; ++tx) {
            if (writeIndex < maxAssignments) {
                tileIds[writeIndex] = rowBase + tx;
                tileIndices[writeIndex] = int(idx);
            }
            writeIndex++;
        }
    }
}

kernel void prepareAssignmentDispatchKernel(
    device TileAssignmentHeader* header [[buffer(0)]],
    device DispatchIndirectArgs* dispatchArgs [[buffer(1)]],
    constant AssignmentDispatchConfig& config [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid > 0) {
        return;
    }
    uint total = header[0].totalAssignments;
    header[0].paddedCount = nextPowerOfTwo(total);
    uint padded = header[0].paddedCount;

    uint sortTG = max(config.sortThreadgroupSize, 1u);
    uint fuseTG = max(config.fuseThreadgroupSize, 1u);
    uint unpackTG = max(config.unpackThreadgroupSize, 1u);
    uint packTG = max(config.packThreadgroupSize, 1u);
    uint bitonicTG = max(config.bitonicThreadgroupSize, 1u);
    uint radixBlockSize = max(config.radixBlockSize, 1u);
    uint radixGrainSize = max(config.radixGrainSize, 1u);

    uint sortGroups = (padded > 0u) ? ((padded + sortTG - 1u) / sortTG) : 0u;
    dispatchArgs[DispatchSlotSortKeys].threadgroupsPerGridX = sortGroups;
    dispatchArgs[DispatchSlotSortKeys].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotSortKeys].threadgroupsPerGridZ = 1u;

    uint fuseGroups = (padded > 0u) ? ((padded + fuseTG - 1u) / fuseTG) : 0u;
    dispatchArgs[DispatchSlotFuseKeys].threadgroupsPerGridX = fuseGroups;
    dispatchArgs[DispatchSlotFuseKeys].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotFuseKeys].threadgroupsPerGridZ = 1u;

    uint unpackGroups = (padded > 0u) ? ((padded + unpackTG - 1u) / unpackTG) : 0u;
    dispatchArgs[DispatchSlotUnpackKeys].threadgroupsPerGridX = unpackGroups;
    dispatchArgs[DispatchSlotUnpackKeys].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotUnpackKeys].threadgroupsPerGridZ = 1u;

    uint packGroups = (total + packTG - 1u) / packTG;
    dispatchArgs[DispatchSlotPack].threadgroupsPerGridX = (total > 0u) ? packGroups : 0u;
    dispatchArgs[DispatchSlotPack].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotPack].threadgroupsPerGridZ = 1u;

    uint bitonicItems = padded;
    uint bitonicThreads = bitonicItems / 2u;
    uint bitonicGroups = (bitonicThreads > 0u) ? ((bitonicThreads + bitonicTG - 1u) / bitonicTG) : 0u;
    dispatchArgs[DispatchSlotBitonicFirst].threadgroupsPerGridX = bitonicGroups;
    dispatchArgs[DispatchSlotBitonicFirst].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotBitonicFirst].threadgroupsPerGridZ = 1u;
    dispatchArgs[DispatchSlotBitonicGeneral] = dispatchArgs[DispatchSlotBitonicFirst];
    dispatchArgs[DispatchSlotBitonicFinal] = dispatchArgs[DispatchSlotBitonicFirst];

    uint valuesPerGroup = max(radixBlockSize * radixGrainSize, 1u);
    uint radixGrid = (padded > 0u) ? ((padded + valuesPerGroup - 1u) / valuesPerGroup) : 0u;
    uint histogramGroups = radixGrid;
    uint applyGroups = radixGrid;
    uint scatterGroups = radixGrid;
    uint blockCount = radixGrid;
    uint exclusiveGroups = (blockCount > 0u) ? 1u : 0u;

    dispatchArgs[DispatchSlotRadixHistogram].threadgroupsPerGridX = histogramGroups;
    dispatchArgs[DispatchSlotRadixHistogram].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotRadixHistogram].threadgroupsPerGridZ = 1u;

    dispatchArgs[DispatchSlotRadixScanBlocks].threadgroupsPerGridX = applyGroups;
    dispatchArgs[DispatchSlotRadixScanBlocks].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotRadixScanBlocks].threadgroupsPerGridZ = 1u;

    dispatchArgs[DispatchSlotRadixExclusive].threadgroupsPerGridX = exclusiveGroups;
    dispatchArgs[DispatchSlotRadixExclusive].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotRadixExclusive].threadgroupsPerGridZ = 1u;

    dispatchArgs[DispatchSlotRadixApply].threadgroupsPerGridX = applyGroups;
    dispatchArgs[DispatchSlotRadixApply].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotRadixApply].threadgroupsPerGridZ = 1u;

    dispatchArgs[DispatchSlotRadixScatter].threadgroupsPerGridX = scatterGroups;
    dispatchArgs[DispatchSlotRadixScatter].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotRadixScatter].threadgroupsPerGridZ = 1u;
    dispatchArgs[DispatchSlotRenderTiles].threadgroupsPerGridX = 0u;
    dispatchArgs[DispatchSlotRenderTiles].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotRenderTiles].threadgroupsPerGridZ = 1u;
}
#define BLOCK_SIZE 256
#define HISTOGRAM_SCAN_BLOCK_SIZE 256
#define GRAIN_SIZE 4
#define RADIX 256
#define BITS_PER_PASS 8
#define NUM_PASSES (64 / BITS_PER_PASS)
#define SCAN_TYPE_INCLUSIVE (0)
#define SCAN_TYPE_EXCLUSIVE (1)

using KeyType = ulong;

struct KeyPayload {
    KeyType key;
    uint payload;
};

template <typename T>
struct SumOp {
    inline T operator()(thread const T& a, thread const T& b) const{return a + b;}
    inline T operator()(threadgroup const T& a, thread const T& b) const{return a + b;}
    inline T operator()(threadgroup const T& a, threadgroup const T& b) const{return a + b;}
    inline T operator()(volatile threadgroup const T& a, volatile threadgroup const T& b) const{return a + b;}
    constexpr T identity(){return static_cast<T>(0);}
};

template <typename T>
struct MaxOp {
    inline T operator()(thread const T& a, thread const T& b) const{return max(a,b);}
    inline T operator()(threadgroup const T& a, thread const T& b) const{return max(a,b);}
    inline T operator()(threadgroup const T& a, threadgroup const T& b) const{return max(a,b);}
    inline T operator()(volatile threadgroup const T& a, volatile threadgroup const T& b) const{return max(a,b);}
    constexpr T identity(){ return metal::numeric_limits<T>::min(); }
};

static constexpr ushort RadixToBits(ushort n) {
    return (n-1<2)?1:
    (n-1<4)?2:
    (n-1<8)?3:
    (n-1<16)?4:
    (n-1<32)?5:
    (n-1<64)?6:
    (n-1<128)?7:
    (n-1<256)?8:
    (n-1<512)?9:
    (n-1<1024)?10:
    (n-1<2048)?11:
    (n-1<4096)?12:
    (n-1<8192)?13:
    (n-1<16384)?14:
    (n-1<32768)?15:0;
}

template <ushort R, typename T> static inline ushort
ValueToKeyAtBit(T value, ushort current_bit){
    return (value >> current_bit) & (R - 1);
}

static inline ushort ValueToKeyAtDigit(KeyType value, ushort current_digit){
    ushort bits_to_shift = RadixToBits(RADIX) * current_digit;
    return ValueToKeyAtBit<RADIX>(value, bits_to_shift);
}

template <ushort R> static inline ushort
ValueToKeyAtBit(KeyPayload value, ushort current_bit){
    return ValueToKeyAtBit<R>(value.key, current_bit);
}

static inline ushort ValueToKeyAtDigit(KeyPayload value, ushort current_digit){
    return ValueToKeyAtDigit(value.key, current_digit);
}

static inline bool KeyIsSentinel(KeyType value) {
    return value == KeyType(0xFFFFFFFFFFFFFFFFull);
}

template<ushort LENGTH, int SCAN_TYPE, typename BinaryOp, typename T>
static inline T ThreadScan(threadgroup T* values, BinaryOp Op){
    for (ushort i = 1; i < LENGTH; i++){
        values[i] = Op(values[i],values[i - 1]);
    }
    T result = values[LENGTH - 1];
    if (SCAN_TYPE == SCAN_TYPE_EXCLUSIVE){
        for (ushort i = LENGTH - 1; i > 0; i--){
            values[i] = values[i - 1];
        }
        values[0] = 0;
    }
    return result;
}

template <int SCAN_TYPE, typename BinaryOp, typename T> static inline T
SimdgroupScan(T value, ushort local_id, BinaryOp Op){
    const ushort lane_id = local_id % 32;
    T temp = simd_shuffle_up(value, 1);
    if (lane_id >= 1) value = Op(value,temp);
    temp = simd_shuffle_up(value, 2);
    if (lane_id >= 2) value = Op(value,temp);
    temp = simd_shuffle_up(value, 4);
    if (lane_id >= 4) value = Op(value,temp);
    temp = simd_shuffle_up(value, 8);
    if (lane_id >= 8) value = Op(value,temp);
    temp = simd_shuffle_up(value, 16);
    if (lane_id >= 16) value = Op(value,temp);
    if (SCAN_TYPE == SCAN_TYPE_EXCLUSIVE){
        temp = simd_shuffle_up(value, 1);
        value = (lane_id == 0) ? 0 : temp;
    }
    return value;
}

template<ushort LENGTH, typename BinaryOp, typename T> static inline void
ThreadUniformApply(threadgroup T* values, T uni, BinaryOp Op){
    for (ushort i = 0; i < LENGTH; i++){
        values[i] = Op(values[i],uni);
    }
}

template<int SCAN_TYPE, typename BinaryOp, typename T> static T
ThreadgroupPrefixScanStoreSum(T value, thread T& inclusive_sum, threadgroup T* shared, const ushort local_id, BinaryOp Op) {
    shared[local_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_id < 32){
        T partial_sum = ThreadScan<BLOCK_SIZE / 32, SCAN_TYPE>(&shared[local_id * (BLOCK_SIZE / 32)], Op);
        T prefix = SimdgroupScan<SCAN_TYPE_EXCLUSIVE>(partial_sum, local_id, Op);
        ThreadUniformApply<BLOCK_SIZE / 32>(&shared[local_id * (BLOCK_SIZE / 32)], prefix, Op);
        if (local_id == 31) shared[0] = prefix + partial_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (SCAN_TYPE == SCAN_TYPE_INCLUSIVE) value = (local_id == 0) ? value : shared[local_id];
    else value = (local_id == 0) ? 0 : shared[local_id];
    inclusive_sum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return value;
}

template<int SCAN_TYPE, typename BinaryOp, typename T> static T
ThreadgroupPrefixScan(T value, threadgroup T* shared, const ushort local_id, BinaryOp Op) {
    shared[local_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id < 32){
        T partial_sum = ThreadScan<BLOCK_SIZE / 32, SCAN_TYPE>(&shared[local_id * (BLOCK_SIZE / 32)], Op);
        T prefix = SimdgroupScan<SCAN_TYPE_EXCLUSIVE>(partial_sum, local_id, Op);
        ThreadUniformApply<BLOCK_SIZE / 32>(&shared[local_id * (BLOCK_SIZE / 32)], prefix, Op);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = shared[local_id];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return value;
}

template <typename T>
static T SortByTwoBits(const T value, threadgroup T* shared, const ushort local_id, const uchar current_bit){
    uchar mask = ValueToKeyAtBit<4>(value, current_bit);

    uchar4 partial_sum;
    uchar4 scan = {0};
    scan[mask] = 1;
    scan = ThreadgroupPrefixScanStoreSum<SCAN_TYPE_EXCLUSIVE>(scan,
                                                              partial_sum,
                                                              reinterpret_cast<threadgroup uchar4*>(shared),
                                                              local_id,
                                                              SumOp<uchar4>());

    ushort4 offset;
    offset[0] = 0;
    offset[1] = offset[0] + partial_sum[0];
    offset[2] = offset[1] + partial_sum[1];
    offset[3] = offset[2] + partial_sum[2];

    shared[scan[mask] + offset[mask]] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    T result = shared[local_id];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return result;
}

template <typename T>
static T SortByBit(const T value, threadgroup T* shared, const ushort local_id, const uchar current_bit){
    uchar mask = ValueToKeyAtBit<2>(value, current_bit);
    
    uchar2 partial_sum;
    uchar2 scan = {0};
    scan[mask] = 1;
    scan = ThreadgroupPrefixScanStoreSum<SCAN_TYPE_EXCLUSIVE>(scan,
                                                              partial_sum,
                                                              reinterpret_cast<threadgroup uchar2*>(shared),
                                                              local_id,
                                                              SumOp<uchar2>());
    
    ushort2 offset;
    offset[0] = 0;
    offset[1] = offset[0] + partial_sum[0];

    shared[scan[mask] + offset[mask]] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    T result = shared[local_id];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return result;
}

template <typename T>
static T PartialRadixSort(const T value, threadgroup T* shared, const ushort local_id, const ushort current_digit){
    T result = value;
    ushort current_bit = current_digit * RadixToBits(RADIX);
    const ushort key_bits = (ushort)(sizeof(KeyType) * 8);
    const ushort range_end = (ushort)(current_bit + RadixToBits(RADIX));
    const ushort last_bit = min(range_end, key_bits);
    while (current_bit < last_bit){
        ushort remaining = last_bit - current_bit;
        if (remaining >= 2){
            result = SortByTwoBits(result, shared, local_id, current_bit);
            current_bit += 2;
        } else {
            result = SortByBit(result, shared, local_id, current_bit);
            current_bit += 1;
        }
    }
    return result;
}

kernel void radixHistogramKernel(
    device const KeyType*               input_keys     [[buffer(0)]],
    device uint*                        hist_flat      [[buffer(1)]],
    constant uint&                      current_digit  [[buffer(3)]],
    const device TileAssignmentHeader*  header         [[buffer(4)]],
    uint                                grid_size      [[threadgroups_per_grid]],
    uint                                group_id       [[threadgroup_position_in_grid]],
    ushort                              local_id       [[thread_position_in_threadgroup]]
) {
    const uint elementsPerBlock = BLOCK_SIZE * GRAIN_SIZE;

    uint base_id          = group_id * elementsPerBlock;
    uint totalAssignments = header[0].totalAssignments;
    uint paddedCount      = header[0].paddedCount;

    uint bufferRemaining   = (base_id < paddedCount)      ? (paddedCount      - base_id) : 0;
    uint assignmentsRemain = (base_id < totalAssignments) ? (totalAssignments - base_id) : 0;
    uint available         = min(bufferRemaining, assignmentsRemain);

    // 1) Per-group local histogram in LDS
    threadgroup uint local_hist[RADIX];

    for (uint i = 0; i < (RADIX + BLOCK_SIZE - 1u) / BLOCK_SIZE; ++i) {
        uint bin = local_id + i * BLOCK_SIZE;
        if (bin < RADIX) {
            local_hist[bin] = 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) Load keys in blocked fashion per group into registers
    KeyType keys[GRAIN_SIZE];
    KeyType sentinelKey = (KeyType)0;

    LoadBlockedLocalFromGlobal<GRAIN_SIZE>(
        keys,
        &input_keys[base_id],
        local_id,
        available,
        sentinelKey);

    // 3) Build local histogram using threadgroup atomics
    volatile threadgroup atomic_uint* atomic_hist =
        reinterpret_cast<volatile threadgroup atomic_uint*>(local_hist);

    for (ushort i = 0; i < GRAIN_SIZE; ++i) {
        uint offset_in_block = local_id * GRAIN_SIZE + i;
        uint global_idx      = base_id + offset_in_block;

        if (global_idx < totalAssignments) {
            uchar bin = (uchar)ValueToKeyAtDigit(keys[i], (ushort)current_digit);
            atomic_fetch_add_explicit(&atomic_hist[bin], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4) Write out in bin-major layout: [bin][block]
    for (uint i = 0; i < (RADIX + BLOCK_SIZE - 1u) / BLOCK_SIZE; ++i) {
        uint bin = local_id + i * BLOCK_SIZE;
        if (bin < RADIX) {
            hist_flat[bin * grid_size + group_id] = local_hist[bin];
        }
    }
}

kernel void radixScanBlocksKernel(
    device const uint*                 hist_flat   [[buffer(0)]],
    device uint*                       block_sums  [[buffer(1)]],
    const device TileAssignmentHeader* header      [[buffer(2)]],
    uint                               group_id    [[threadgroup_position_in_grid]],
    ushort                             local_id    [[thread_position_in_threadgroup]]
) {
    threadgroup uint shared_mem[BLOCK_SIZE];

    const uint elementsPerBlock = BLOCK_SIZE * GRAIN_SIZE;
    uint paddedCount   = header[0].paddedCount;
    uint histGridSize  = (paddedCount + elementsPerBlock - 1u) / elementsPerBlock;
    uint num_hist_elem = histGridSize * RADIX;

    uint block_base = group_id * BLOCK_SIZE;
    uint idx        = block_base + local_id;

    uint val = (idx < num_hist_elem) ? hist_flat[idx] : 0u;
    shared_mem[local_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_mem[local_id] += shared_mem[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_id == 0) {
        block_sums[group_id] = shared_mem[0];
    }
}

kernel void radixExclusiveScanKernel(
    device uint*                       block_sums  [[buffer(0)]],
    const device TileAssignmentHeader* header      [[buffer(1)]],
    ushort                             local_id    [[thread_index_in_threadgroup]]
) {
    if (local_id != 0) {
        return;
    }

    const uint elementsPerBlock = BLOCK_SIZE * GRAIN_SIZE;
    uint paddedCount   = header[0].paddedCount;
    uint histGridSize  = (paddedCount + elementsPerBlock - 1u) / elementsPerBlock;
    uint num_hist_elem = histGridSize * RADIX;
    uint num_block_sums = (num_hist_elem + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    uint running = 0u;
    for (uint i = 0u; i < num_block_sums; ++i) {
        uint v = block_sums[i];
        block_sums[i] = running;
        running += v;
    }
}

kernel void radixApplyScanOffsetsKernel(
    device const uint*                 hist_flat       [[buffer(0)]],
    device const uint*                 scanned_blocks  [[buffer(1)]],
    device uint*                       offsets_flat    [[buffer(2)]],
    const device TileAssignmentHeader* header          [[buffer(3)]],
    threadgroup uint*                  shared_mem      [[threadgroup(0)]],
    uint                               group_id        [[threadgroup_position_in_grid]],
    ushort                             local_id        [[thread_position_in_threadgroup]]
) {
    const uint elementsPerBlock = BLOCK_SIZE * GRAIN_SIZE;
    uint paddedCount   = header[0].paddedCount;
    uint histGridSize  = (paddedCount + elementsPerBlock - 1u) / elementsPerBlock;
    uint num_hist_elem = histGridSize * RADIX;

    uint block_base = group_id * BLOCK_SIZE;
    uint idx        = block_base + local_id;

    uint val = (idx < num_hist_elem) ? hist_flat[idx] : 0u;

    uint local_scanned = ThreadgroupPrefixScan<SCAN_TYPE_EXCLUSIVE>(
        val, shared_mem, local_id, SumOp<uint>());

    uint block_offset = scanned_blocks[group_id];

    if (idx < num_hist_elem) {
        offsets_flat[idx] = local_scanned + block_offset;
    }
}

kernel void radixScatterKernel(
    device KeyType*                    output_keys     [[buffer(0)]],
    device const KeyType*              input_keys      [[buffer(1)]],
    device uint*                       output_payload  [[buffer(2)]],
    device const uint*                 input_payload   [[buffer(3)]],
    device const uint*                 offsets_flat    [[buffer(5)]],
    constant uint&                     current_digit   [[buffer(6)]],
    const device TileAssignmentHeader* header          [[buffer(7)]],
    uint                               group_id        [[threadgroup_position_in_grid]],
    uint                               grid_size       [[threadgroups_per_grid]],
    ushort                             local_id        [[thread_position_in_threadgroup]]
) {
    const uint elementsPerBlock = BLOCK_SIZE * GRAIN_SIZE;

    uint base_id          = group_id * elementsPerBlock;
    uint totalAssignments = header[0].totalAssignments;
    uint paddedCount      = header[0].paddedCount;

    uint bufferRemaining   = (base_id < paddedCount)      ? (paddedCount      - base_id) : 0;
    uint assignmentsRemain = (base_id < totalAssignments) ? (totalAssignments - base_id) : 0;
    uint available         = min(bufferRemaining, assignmentsRemain);

    // Use max key as sentinel
    constexpr KeyType keySentinel = ~(KeyType)0;

    // 1) Load keys / payloads in STRIPED fashion
    KeyType keys[GRAIN_SIZE];
    uint payloads[GRAIN_SIZE];
    LoadStripedLocalFromGlobal<GRAIN_SIZE>(keys, &input_keys[base_id], local_id, BLOCK_SIZE, available, keySentinel);
    LoadStripedLocalFromGlobal<GRAIN_SIZE>(payloads, &input_payload[base_id], local_id, BLOCK_SIZE, available, UINT_MAX);

    // 2) Shared memory for bin offsets and ranking
    threadgroup uint global_bin_base[RADIX];
    threadgroup KeyPayload tg_kp[BLOCK_SIZE];
    threadgroup ushort tg_short[BLOCK_SIZE];

    // Load global bin offsets
    for (uint i = 0; i < (RADIX + BLOCK_SIZE - 1u) / BLOCK_SIZE; ++i) {
        uint bin = local_id + i * BLOCK_SIZE;
        if (bin < RADIX) {
            global_bin_base[bin] = offsets_flat[bin * grid_size + group_id];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) Process chunks with optimized ranking
    for (ushort chunk = 0; chunk < GRAIN_SIZE; ++chunk) {
        KeyPayload kp;
        kp.key = keys[chunk];
        kp.payload = payloads[chunk];

        // Partial radix sort: groups same-bin elements together, preserving relative order
        kp = PartialRadixSort(kp, tg_kp, local_id, (ushort)current_digit);

        // Extract bin after sorting
        ushort my_bin = ValueToKeyAtDigit(kp.key, (ushort)current_digit);

        // Head discontinuity + max scan to find run start
        uchar head_flag = FlagHeadDiscontinuity<BLOCK_SIZE>(my_bin, tg_short, local_id);
        ushort run_start = ThreadgroupPrefixScan<SCAN_TYPE_INCLUSIVE>(
            head_flag ? local_id : (ushort)0, tg_short, local_id, MaxOp<ushort>());
        ushort local_offset = local_id - run_start;

        // Tail discontinuity for offset updates
        uchar tail_flag = FlagTailDiscontinuity<BLOCK_SIZE>(my_bin, tg_short, local_id);

        // Check validity and scatter
        bool is_valid = (kp.payload != UINT_MAX);
        if (is_valid) {
            uint dst = global_bin_base[my_bin] + local_offset;
            output_keys[dst] = kp.key;
            output_payload[dst] = kp.payload;
        }

        // Update global offsets at run boundaries
        if (tail_flag && is_valid) {
            global_bin_base[my_bin] += local_offset + 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void fuseSortKeysKernel(
    const device uint2* input_keys [[buffer(0)]],
    device KeyType* output_keys [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint2 key = input_keys[gid];
    KeyType fused = (KeyType(key.x) << 32) | KeyType(key.y);
    output_keys[gid] = fused;
}

kernel void unpackSortKeysKernel(
    const device KeyType* input_keys [[buffer(0)]],
    device uint2* output_keys [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    KeyType fused = input_keys[gid];
    uint tile = uint(fused >> 32);
    uint depthBits = uint(fused & 0xFFFFFFFFull);
    output_keys[gid] = uint2(tile, depthBits);
}
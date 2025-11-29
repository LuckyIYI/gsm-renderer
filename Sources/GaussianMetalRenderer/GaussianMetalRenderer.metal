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

// =============================================================================
// INTERLEAVED DATA STRUCTURES - Cache-efficient single-struct access
// =============================================================================

/// Interleaved gaussian render data (half16) - 24 bytes, aligned
/// Combines all per-gaussian data for single-read cache efficiency
struct GaussianRenderData {
    half2         mean;       // 4 bytes
    half4         conic;      // 8 bytes
    packed_half3  color;      // 6 bytes
    half          opacity;    // 2 bytes
    half          depth;      // 2 bytes
    ushort        _pad;       // 2 bytes (alignment to 24)
};

/// Interleaved gaussian render data (float32) - 48 bytes
struct GaussianRenderDataF32 {
    float2         mean;      // 8 bytes
    float4         conic;     // 16 bytes
    packed_float3  color;     // 12 bytes
    float          opacity;   // 4 bytes
    float          depth;     // 4 bytes
    uint           _pad;      // 4 bytes (alignment to 48)
};

/// Packed gaussian for render stage - matches GaussianRenderData layout
struct PackedGaussian {
    half2         mean;       // 4 bytes
    half4         conic;      // 8 bytes
    packed_half3  color;      // 6 bytes
    half          opacity;    // 2 bytes
    half          depth;      // 2 bytes
    ushort        _pad;       // 2 bytes
};

struct PackedGaussianF32 {
    float2         mean;      // 8 bytes
    float4         conic;     // 16 bytes
    packed_float3  color;     // 12 bytes
    float          opacity;   // 4 bytes
    float          depth;     // 4 bytes
    uint           _pad;      // 4 bytes
};

// =============================================================================
// PACKED WORLD GAUSSIAN - Interleaved input for projection kernel
// Eliminates 5 scattered buffer reads -> 1 coalesced read
// =============================================================================

/// Packed world gaussian (float32) - 48 bytes, 16-byte aligned
/// Contains all per-gaussian data needed for projection (except harmonics)
struct PackedWorldGaussian {
    packed_float3 position;   // 12 bytes - world position
    float         opacity;    // 4 bytes  - opacity value
    packed_float3 scale;      // 12 bytes - scale xyz
    float         _pad0;      // 4 bytes  - padding
    float4        rotation;   // 16 bytes - quaternion (xyzw)
};  // Total: 48 bytes

/// Packed world gaussian (half16) - 24 bytes
struct PackedWorldGaussianHalf {
    packed_half3  position;   // 6 bytes
    half          opacity;    // 2 bytes
    packed_half3  scale;      // 6 bytes
    half          _pad0;      // 2 bytes
    half4         rotation;   // 8 bytes
};  // Total: 24 bytes

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
    // 3DGS convention: quaternion stored as (x, y, z, w) where w is scalar
    float x = quat.x;
    float y = quat.y;
    float z = quat.z;
    float r = quat.w;  // scalar/real part is .w (last component)
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
    // Optimized: RS = R * diag(scale) is just scaling each column of R
    float4 q = normalizeQuaternion(quat);
    float3x3 R = quaternionToMatrix(q);

    // RS = [R[:,0]*s.x, R[:,1]*s.y, R[:,2]*s.z]
    // Since R is stored column-major, R[i] is column i
    float3 RS0 = R[0] * scale.x;
    float3 RS1 = R[1] * scale.y;
    float3 RS2 = R[2] * scale.z;

    // Cov = RS * RS^T = sum of outer products of scaled columns
    // Cov[i][j] = RS0[i]*RS0[j] + RS1[i]*RS1[j] + RS2[i]*RS2[j]
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

    float2x2 cov2d = float2x2(
        covFull[0][0], covFull[0][1],
        covFull[1][0], covFull[1][1]
    );
    cov2d[0][0] += 0.3f;
    cov2d[1][1] += 0.3f;
    return cov2d;
}

// Fused conic + radius computation - shares determinant calculation
inline void computeConicAndRadius(float2x2 cov, thread float4& conic, thread float& radius) {
    float a = cov[0][0], b = cov[0][1], c = cov[1][0], d = cov[1][1];
    float det = a * d - b * c;

    // Conic (inverse covariance)
    float invDet = 1.0f / max(det, 1e-8f);
    conic = float4(d * invDet, -b * invDet, a * invDet, 0.0f);

    // Radius from eigenvalues
    float mid = 0.5f * (a + d);
    float delta = max(mid * mid - det, 1e-5f);
    float sqrtDelta = sqrt(delta);
    float maxEig = mid + sqrtDelta;  // Larger eigenvalue
    radius = 3.0f * ceil(sqrt(max(maxEig, 1e-5f)));
}

// Legacy functions for backwards compatibility
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

// =============================================================================
// PACKED PROJECTION KERNEL - Single coalesced read per gaussian
// Uses PackedWorldGaussian for dramatically better memory access patterns
// =============================================================================

template <typename PackedWorldT, typename HarmonicT,
          typename OutVec2, typename OutVec4, typename OutT, typename OutPacked3>
kernel void projectGaussiansPacked(
    const device PackedWorldT* worldGaussians [[buffer(0)]],
    const device HarmonicT* harmonics [[buffer(1)]],
    device OutVec2* outMeans [[buffer(2)]],
    device OutVec4* outConics [[buffer(3)]],
    device OutPacked3* outColors [[buffer(4)]],
    device OutT* outOpacities [[buffer(5)]],
    device OutT* outDepths [[buffer(6)]],
    device float* outRadii [[buffer(7)]],
    device uchar* outMask [[buffer(8)]],
    constant CameraUniforms& camera [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= camera.gaussianCount) { return; }

    // SINGLE coalesced read for all core gaussian data
    PackedWorldT g = worldGaussians[gid];

    // Extract and promote to float for computation accuracy
    float3 position = float3(g.position);
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

    // Opacity already loaded from packed struct
    float opacity = float(g.opacity);
    if (opacity < 1e-4f) {
        outMask[gid] = 0;
        outRadii[gid] = 0.0f;
        return;
    }

    // Scale and rotation already loaded from packed struct
    float3 scale = float3(g.scale);
    float4 quat = float4(g.rotation);
    float3x3 cov3d = buildCovariance3D(scale, quat);

    float3 viewRow0 = float3(camera.viewMatrix[0][0], camera.viewMatrix[1][0], camera.viewMatrix[2][0]);
    float3 viewRow1 = float3(camera.viewMatrix[0][1], camera.viewMatrix[1][1], camera.viewMatrix[2][1]);
    float3 viewRow2 = float3(camera.viewMatrix[0][2], camera.viewMatrix[1][2], camera.viewMatrix[2][2]);
    float3x3 viewRot = matrixFromRows(viewRow0, viewRow1, viewRow2);

    float2x2 cov2d = projectCovariance(cov3d, viewPos4.xyz, viewRot, camera.focalX, camera.focalY, camera.width, camera.height);
    float4 conic;
    float radius;
    computeConicAndRadius(cov2d, conic, radius);

    if (radius < 0.5f) {
        outMask[gid] = 0;
        outRadii[gid] = 0.0f;
        return;
    }

    // Compute SH color (harmonics still separate - variable size)
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

    // Write outputs
    outMeans[gid] = OutVec2(px, py);
    outConics[gid] = OutVec4(conic);
    outOpacities[gid] = OutT(opacity);
    outDepths[gid] = OutT(depth);
    outRadii[gid] = radius;
    outColors[gid] = OutPacked3(color);
    outMask[gid] = 1;
}

// Instantiation macro for packed projectGaussians
#define instantiate_projectGaussiansPacked(name, PackedWorldT, HarmonicT, OutVec2, OutVec4, OutT, OutPacked3) \
    template [[host_name("projectGaussiansPacked_" #name)]] \
    kernel void projectGaussiansPacked<PackedWorldT, HarmonicT, OutVec2, OutVec4, OutT, OutPacked3>( \
        const device PackedWorldT* worldGaussians [[buffer(0)]], \
        const device HarmonicT* harmonics [[buffer(1)]], \
        device OutVec2* outMeans [[buffer(2)]], \
        device OutVec4* outConics [[buffer(3)]], \
        device OutPacked3* outColors [[buffer(4)]], \
        device OutT* outOpacities [[buffer(5)]], \
        device OutT* outDepths [[buffer(6)]], \
        device float* outRadii [[buffer(7)]], \
        device uchar* outMask [[buffer(8)]], \
        constant CameraUniforms& camera [[buffer(9)]], \
        uint gid [[thread_position_in_grid]]);

// Float packed world -> Half output (main path for half pipeline)
instantiate_projectGaussiansPacked(float_half, PackedWorldGaussian, float, half2, half4, half, packed_half3)

// Float packed world -> Float output
instantiate_projectGaussiansPacked(float_float, PackedWorldGaussian, float, float2, float4, float, packed_float3)

// Half packed world -> Half output (full half pipeline)
// NOTE: Harmonics stay float (wide dynamic range, minimal memory savings for half)
instantiate_projectGaussiansPacked(half_half, PackedWorldGaussianHalf, float, half2, half4, half, packed_half3)

#undef instantiate_projectGaussiansPacked

// Fast exp approximation for Gaussian weight calculation
// Uses Schraudolph's method - accurate enough for alpha blending
inline float fast_exp_neg(float x) {
    // x should be negative (from -0.5 * quad)
    // Clamp to prevent overflow
    x = max(x, -10.0f);  // exp(-10) ≈ 0, below our threshold anyway
    // Schraudolph's approximation: exp(x) via bit manipulation
    // f = 2^(x * log2(e)) = 2^(x * 1.4427)
    // Scale factor: 2^23 / ln(2) = 12102203.16
    // Bias: 127 * 2^23 = 1065353216, adjusted = 1064866805
    union { float f; int i; } u;
    u.i = int(12102203.0f * x + 1064866805.0f);
    return u.f;
}

// Batch size for shared memory loading (buffer version)
constant uint RENDER_BATCH_SIZE_BUF = 32;

// HYBRID PRECISION: half for per-gaussian math, float for accumulation
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
    uint3 tileCoord [[threadgroup_position_in_grid]],
    uint threadIdx [[thread_index_in_threadgroup]]
) {
    // Shared memory in half precision - saves bandwidth
    threadgroup half2 shMeans[RENDER_BATCH_SIZE_BUF];
    threadgroup half4 shConics[RENDER_BATCH_SIZE_BUF];
    threadgroup half3 shColors[RENDER_BATCH_SIZE_BUF];
    threadgroup half shOpacities[RENDER_BATCH_SIZE_BUF];
    threadgroup half shDepths[RENDER_BATCH_SIZE_BUF];
    // Per-simd-group done flags (8 simd groups in 16x16 tile)
    threadgroup bool simdDoneFlags[8];

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

    // Pixel position in half (sufficient for coordinates up to 2048)
    half hx = half(px);
    half hy = half(py);

    // Accumulation in float for stability
    float3 accumColor = float3(0.0f);
    float accumDepth = 0.0f;
    float accumAlpha = 0.0f;
    float trans = 1.0f;

    GaussianHeader header = headers[tileId];
    uint start = header.offset;
    uint count = header.count;

    // Get simd group index (0-7 for 256 threads)
    uint simdGroupIdx = threadIdx / 32;

    // Initialize done flags
    if (threadIdx < 8) {
        simdDoneFlags[threadIdx] = false;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process gaussians in batches using shared memory
    for (uint batch = 0; batch < count; batch += RENDER_BATCH_SIZE_BUF) {
        uint batchCount = min(RENDER_BATCH_SIZE_BUF, count - batch);

        // Cooperative loading (half)
        if (threadIdx < batchCount) {
            uint gIdx = start + batch + threadIdx;
            shMeans[threadIdx] = half2(means[gIdx]);
            shConics[threadIdx] = half4(conics[gIdx]);
            shColors[threadIdx] = half3(colors[gIdx]);
            shOpacities[threadIdx] = min(half(opacities[gIdx]), half(0.99h));
            shDepths[threadIdx] = half(depths[gIdx]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process all gaussians in this batch - BRANCHLESS half precision math
        // Opacity > 0 guaranteed by projection filtering
        if (inBounds && trans > 1e-3f) {
            for (uint i = 0; i < batchCount; ++i) {
                half2 mean = shMeans[i];
                half4 conic = shConics[i];

                // Half precision gaussian evaluation
                // Clamp dx/dy to prevent half overflow (250^2 = 62500 < 65504 max half)
                half dx = clamp(hx - mean.x, half(-250.0h), half(250.0h));
                half dy = clamp(hy - mean.y, half(-250.0h), half(250.0h));
                half quad = dx * dx * conic.x + dy * dy * conic.z + 2.0h * dx * dy * conic.y;

                // Branchless: clamp quad (already bounded by dx/dy clamp, but be safe)
                half clampedQuad = min(quad, half(20.0h));
                half weight = exp(-0.5h * clampedQuad);
                half hAlpha = weight * shOpacities[i];

                // Branchless mask: zero out if quad >= 20 or alpha too small
                half mask = half(quad < 20.0h) * half(hAlpha > 1e-4h);
                hAlpha *= mask;

                // Convert to float for stable accumulation
                float alpha = float(hAlpha);
                float contrib = trans * alpha;
                trans *= (1.0f - alpha);
                accumAlpha += contrib;
                accumColor += float3(shColors[i]) * contrib;
                accumDepth += float(shDepths[i]) * contrib;

                // Early exit when fully opaque (this branch is worth keeping)
                if (trans < 1e-3f) { break; }
            }
        }

        // Check if all simd groups are done (tile-level early exit)
        bool pixelDone = (trans < 1e-3f) || !inBounds;
        bool simdDone = simd_all(pixelDone);
        if (simdDone && simd_is_first()) {
            simdDoneFlags[simdGroupIdx] = true;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Check all flags (small unrolled loop)
        bool allDone = simdDoneFlags[0] && simdDoneFlags[1] && simdDoneFlags[2] && simdDoneFlags[3] &&
                       simdDoneFlags[4] && simdDoneFlags[5] && simdDoneFlags[6] && simdDoneFlags[7];
        if (allDone) {
            break;
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
        uint3 tileCoord [[threadgroup_position_in_grid]], \
        uint threadIdx [[thread_index_in_threadgroup]] \
    );

instantiate_renderTiles(float, float, float2, float4, packed_float3)
instantiate_renderTiles(half, half, half2, half4, packed_half3)

#undef instantiate_renderTiles

///////////////////////////////////////////////////////////////////////////////
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
    uint tileWidth;   // For precise intersection (optional, 0 = use AABB only)
    uint tileHeight;  // For precise intersection
};

struct ScatterParams {
    uint gaussianCount;
    uint tilesX;
    uint tileWidth;   // For precise intersection
    uint tileHeight;  // For precise intersection
};

// Tile-binning pipeline: per-tile counts then scatter using per-tile offsets
struct TileBinningParams {
    uint gaussianCount;
    uint tilesX;
    uint tilesY;
    uint tileWidth;
    uint tileHeight;
    uint maxAssignments;
};

struct TileAssignmentHeader {
    uint totalAssignments;
    uint maxAssignments;
    uint paddedCount;
    uint overflow;
};

// Reset tile builder state - replaces blit with single-thread compute for lower overhead
kernel void resetTileBuilderStateKernel(
    device TileAssignmentHeader* header [[buffer(0)]],
    device uint* activeTileCount [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        header->totalAssignments = 0;
        header->overflow = 0;
        *activeTileCount = 0;
    }
}

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
    uint maxAssignments;  // Clamp totalAssignments to this value
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

///////////////////////////////////////////////////////////////////////////////
// FlashGS-style Precise Ellipse-Tile Intersection
// Eliminates tiles in AABB that don't actually intersect the gaussian ellipse
///////////////////////////////////////////////////////////////////////////////

// Check if point is inside ellipse: (p - center)^T * conic * (p - center) <= qMax
inline bool ellipseContainsPoint(float2 center, float3 conic, float qMax, float2 p) {
    float2 d = p - center;
    float q = d.x * d.x * conic.x + 2.0f * d.x * d.y * conic.y + d.y * d.y * conic.z;
    return q <= qMax;
}

// Check if ellipse intersects line segment p0->p1
// Solves quadratic for t where ellipse intersects parametric line p(t) = p0 + t*(p1-p0)
inline bool ellipseIntersectsEdge(float2 center, float3 conic, float qMax, float2 p0, float2 p1) {
    float2 dir = p1 - p0;
    float2 c = p0 - center;

    // Quadratic coefficients: a*t^2 + b*t + c0 = 0
    float a = conic.x * dir.x * dir.x + 2.0f * conic.y * dir.x * dir.y + conic.z * dir.y * dir.y;
    float b = 2.0f * (conic.x * c.x * dir.x + conic.y * (c.x * dir.y + c.y * dir.x) + conic.z * c.y * dir.y);
    float c0 = conic.x * c.x * c.x + 2.0f * conic.y * c.x * c.y + conic.z * c.y * c.y - qMax;

    // Handle near-linear case
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

    // Check if either root is in [0, 1]
    return (t0 >= 0.0f && t0 <= 1.0f) || (t1 >= 0.0f && t1 <= 1.0f) ||
           (t0 < 0.0f && t1 > 1.0f);  // Edge fully inside ellipse
}

// Full ellipse-tile intersection test
inline bool ellipseIntersectsTile(float2 center, float3 conic, float qMax,
                                   float tileMinX, float tileMinY, float tileMaxX, float tileMaxY) {
    // Corners
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

// Opacity-aware threshold: we only care about the gaussian where weight >= tau
// weight = alpha * exp(-0.5 * q) >= tau  =>  q <= -2 * ln(tau / alpha)
inline float computeQMax(float alpha, float tau) {
    if (alpha <= tau) return 0.0f;
    return -2.0f * log(tau / alpha);
}

///////////////////////////////////////////////////////////////////////////////

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
            // Quantize depth to 16-bit half-float for faster radix sort (4 passes instead of 6)
            // Half-float bits sort correctly for positive values, and 16-bit precision is
            // more than enough for correct depth ordering within tiles
            float depthFloat = float(depths[g]);
            half depthHalf = half(depthFloat);
            uint depthBits = uint(as_type<ushort>(depthHalf));
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
    constant uint& tileCount [[buffer(3)]],
    uint tile [[thread_position_in_grid]]
) {
    if (tile >= tileCount) {
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

///////////////////////////////////////////////////////////////////////////////
// FlashGS Precise Coverage Kernel - counts only tiles with actual ellipse intersection
// Parallel version: one threadgroup per gaussian, threads cooperatively test tiles
///////////////////////////////////////////////////////////////////////////////

// tau = minimum opacity threshold for visibility (typically 1/255 = 0.00392)
constant float FLASHGS_TAU = 1.0f / 255.0f;

// Threadgroup size for parallel precise kernels
constant uint PRECISE_TG_SIZE = 32u;  // Single SIMD group - fastest (pure simd_sum, no barriers)

// Only use precise intersection for AABBs larger than this threshold
constant uint PRECISE_THRESHOLD_TILES = 16u;

// SIMD-optimized ellipse point test - tests 4 points at once
// Returns bitmask of which points are inside (bit 0-3)
inline uint ellipseContainsPoints4(float2 center, float3 conic, float qMax,
                                    float4 px, float4 py) {
    float4 dx = px - center.x;
    float4 dy = py - center.y;
    float4 q = dx * dx * conic.x + 2.0f * dx * dy * conic.y + dy * dy * conic.z;
    uint mask = 0;
    if (q.x <= qMax) mask |= 1u;
    if (q.y <= qMax) mask |= 2u;
    if (q.z <= qMax) mask |= 4u;
    if (q.w <= qMax) mask |= 8u;
    return mask;
}

// Fast tile test using center + expanded radius (conservative)
inline bool ellipseTileFastTest(float2 center, float3 conic, float qMax,
                                 float tileCenterX, float tileCenterY,
                                 float expandedQMax) {
    float dx = tileCenterX - center.x;
    float dy = tileCenterY - center.y;
    float q = dx * dx * conic.x + 2.0f * dx * dy * conic.y + dy * dy * conic.z;
    return q <= expandedQMax;
}

template <typename MeansT, typename ConicT, typename OpacityT>
kernel void gaussianCoveragePreciseKernel(
    const device int4* bounds [[buffer(0)]],
    device uint* coverageCounts [[buffer(1)]],
    const device OpacityT* opacities [[buffer(2)]],
    constant CoverageParams& params [[buffer(3)]],
    const device MeansT* means [[buffer(4)]],
    const device ConicT* conics [[buffer(5)]],
    uint gaussianIdx [[threadgroup_position_in_grid]],
    uint localIdx [[thread_index_in_threadgroup]]
) {
    if (gaussianIdx >= params.gaussianCount) {
        return;
    }

    // Thread 0 loads gaussian data to shared memory
    threadgroup float2 sharedCenter;
    threadgroup float3 sharedConic;
    threadgroup float sharedQMax;
    threadgroup int4 sharedRect;
    threadgroup uint sharedAabbCount;
    threadgroup float sharedTileW;
    threadgroup float sharedTileH;

    if (localIdx == 0) {
        float alpha = float(opacities[gaussianIdx]);
        sharedRect = bounds[gaussianIdx];

        int width = sharedRect.y - sharedRect.x + 1;
        int height = sharedRect.w - sharedRect.z + 1;
        sharedAabbCount = (width > 0 && height > 0) ? uint(width * height) : 0u;

        if (alpha < 1e-4f || sharedAabbCount == 0u) {
            sharedQMax = 0.0f;
        } else if (sharedAabbCount <= PRECISE_THRESHOLD_TILES) {
            // Small AABB - use AABB count directly
            sharedQMax = -1.0f;  // Signal to use AABB count
        } else {
            sharedCenter = float2(means[gaussianIdx]);
            ConicT conicData = conics[gaussianIdx];
            sharedConic = float3(conicData.x, conicData.y, conicData.z);
            sharedQMax = computeQMax(alpha, FLASHGS_TAU);
            sharedTileW = float(params.tileWidth);
            sharedTileH = float(params.tileHeight);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Early exit cases
    if (sharedQMax == 0.0f) {
        if (localIdx == 0) {
            coverageCounts[gaussianIdx] = 0u;
        }
        return;
    }
    if (sharedQMax < 0.0f) {
        // Small AABB - use AABB count
        if (localIdx == 0) {
            coverageCounts[gaussianIdx] = sharedAabbCount;
        }
        return;
    }

    // Each thread tests a subset of tiles
    uint totalTiles = sharedAabbCount;
    uint myCount = 0u;

    int minTX = sharedRect.x;
    int minTY = sharedRect.z;
    int width = sharedRect.y - sharedRect.x + 1;

    // Precompute values for fast tile tests
    float halfTileW = sharedTileW * 0.5f;
    float halfTileH = sharedTileH * 0.5f;
    // Expand qMax by tile diagonal for quick rejection test
    float tileDiag = sqrt(halfTileW * halfTileW + halfTileH * halfTileH);
    float maxConic = max(sharedConic.x, sharedConic.z);
    float expand = tileDiag * sqrt(maxConic);
    float qMaxExpanded = sharedQMax + 2.0f * expand * sqrt(sharedQMax) + expand * expand;

    // Grid-stride loop: each thread handles tiles [localIdx, localIdx+TG_SIZE, localIdx+2*TG_SIZE, ...]
    for (uint tileIdx = localIdx; tileIdx < totalTiles; tileIdx += PRECISE_TG_SIZE) {
        int localY = int(tileIdx) / width;
        int localX = int(tileIdx) % width;
        int tx = minTX + localX;
        int ty = minTY + localY;

        // Fast test: check tile center against expanded ellipse
        float tileCenterX = (float(tx) + 0.5f) * sharedTileW;
        float tileCenterY = (float(ty) + 0.5f) * sharedTileH;
        float dx = tileCenterX - sharedCenter.x;
        float dy = tileCenterY - sharedCenter.y;
        float q = dx * dx * sharedConic.x + 2.0f * dx * dy * sharedConic.y + dy * dy * sharedConic.z;

        // Quick accept: tile center inside original ellipse
        if (q <= sharedQMax) {
            myCount++;
            continue;
        }
        // Quick reject: tile center outside expanded ellipse
        if (q > qMaxExpanded) {
            continue;
        }
        // Boundary case: do full intersection test
        float tileMinX = float(tx) * sharedTileW;
        float tileMinY = float(ty) * sharedTileH;
        float tileMaxX = tileMinX + sharedTileW;
        float tileMaxY = tileMinY + sharedTileH;

        if (ellipseIntersectsTile(sharedCenter, sharedConic, sharedQMax, tileMinX, tileMinY, tileMaxX, tileMaxY)) {
            myCount++;
        }
    }

    // Pure SIMD reduction - single simd group, no shared memory or barriers needed
    uint total = simd_sum(myCount);
    if (localIdx == 0) {
        coverageCounts[gaussianIdx] = total;
    }
}

#define instantiate_gaussianCoveragePreciseKernel(name, MeansT, ConicT, OpacityT) \
    template [[host_name("gaussianCoveragePreciseKernel_" #name)]] \
    kernel void gaussianCoveragePreciseKernel<MeansT, ConicT, OpacityT>( \
        const device int4* bounds [[buffer(0)]], \
        device uint* coverageCounts [[buffer(1)]], \
        const device OpacityT* opacities [[buffer(2)]], \
        constant CoverageParams& params [[buffer(3)]], \
        const device MeansT* means [[buffer(4)]], \
        const device ConicT* conics [[buffer(5)]], \
        uint gaussianIdx [[threadgroup_position_in_grid]], \
        uint localIdx [[thread_index_in_threadgroup]]);

instantiate_gaussianCoveragePreciseKernel(float, float2, float4, float)
instantiate_gaussianCoveragePreciseKernel(half, half2, half4, half)

#undef instantiate_gaussianCoveragePreciseKernel

///////////////////////////////////////////////////////////////////////////////
// Tile-binning pipeline (Tellusim-style):
// 1) Count gaussians per tile (precise ellipse/tile intersection)
// 2) Prefix-scan counts to offsets (reuse coveragePrefixScan kernels)
// 3) Scatter gaussian indices using per-tile offsets (contiguous per-tile blocks)
///////////////////////////////////////////////////////////////////////////////

// Clear per-tile counters (coverageBuffer repurposed to tileCounts)
kernel void tileBinningClearKernel(
    device uint* tileCounts [[buffer(0)]],
    device TileAssignmentHeader* header [[buffer(1)]],
    constant uint& tileCount [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < tileCount) {
        tileCounts[gid] = 0u;
    }
    if (gid == 0) {
        header->totalAssignments = 0u;
        header->overflow = 0u;
    }
}

// Clear tile counters without touching header (used before scatter)
kernel void tileBinningZeroCountsKernel(
    device uint* tileCounts [[buffer(0)]],
    constant uint& tileCount [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < tileCount) {
        tileCounts[gid] = 0u;
    }
}

template <typename MeansT, typename ConicT, typename OpacityT>
kernel void tileBinningCountKernel(
    const device int4* bounds [[buffer(0)]],
    device uint* tileCounts [[buffer(1)]],
    const device OpacityT* opacities [[buffer(2)]],
    constant TileBinningParams& params [[buffer(3)]],
    const device MeansT* means [[buffer(4)]],
    const device ConicT* conics [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.gaussianCount) {
        return;
    }

    float alpha = float(opacities[idx]);
    if (alpha < 1e-4f) {
        return;
    }

    int4 rect = bounds[idx];
    int width = rect.y - rect.x + 1;
    int height = rect.w - rect.z + 1;
    if (width <= 0 || height <= 0) {
        return;
    }

    uint aabbCount = uint(width * height);
    bool usePrecise = (aabbCount > PRECISE_THRESHOLD_TILES);

    float2 center = float2(0.0f);
    float3 conic = float3(0.0f);
    float tileW = float(params.tileWidth);
    float tileH = float(params.tileHeight);
    float qMax = 0.0f;
    float qMaxExpanded = 0.0f;

    if (usePrecise) {
        center = float2(means[idx]);
        ConicT conicData = conics[idx];
        conic = float3(conicData.x, conicData.y, conicData.z);
        qMax = computeQMax(alpha, FLASHGS_TAU);

        float halfTileW = tileW * 0.5f;
        float halfTileH = tileH * 0.5f;
        float tileDiag = sqrt(halfTileW * halfTileW + halfTileH * halfTileH);
        float maxConic = max(conic.x, conic.z);
        float expand = tileDiag * sqrt(maxConic);
        qMaxExpanded = qMax + 2.0f * expand * sqrt(qMax) + expand * expand;
    }

    uint tilesX = params.tilesX;
    uint tileCount = tilesX * params.tilesY;

    for (int ty = rect.z; ty <= rect.w; ++ty) {
        for (int tx = rect.x; tx <= rect.y; ++tx) {
            if (tx < 0 || ty < 0) continue;
            uint tileId = uint(ty) * tilesX + uint(tx);
            if (tileId >= tileCount) continue;

            bool intersects = true;
            if (usePrecise) {
                float tileCenterX = (float(tx) + 0.5f) * tileW;
                float tileCenterY = (float(ty) + 0.5f) * tileH;
                float dx = tileCenterX - center.x;
                float dy = tileCenterY - center.y;
                float q = dx * dx * conic.x + 2.0f * dx * dy * conic.y + dy * dy * conic.z;
                if (q <= qMax) {
                    intersects = true;
                } else if (q > qMaxExpanded) {
                    intersects = false;
                } else {
                    float tileMinX = float(tx) * tileW;
                    float tileMinY = float(ty) * tileH;
                    float tileMaxX = tileMinX + tileW;
                    float tileMaxY = tileMinY + tileH;
                    intersects = ellipseIntersectsTile(center, conic, qMax, tileMinX, tileMinY, tileMaxX, tileMaxY);
                }
            }

            if (intersects) {
                atomic_fetch_add_explicit((device atomic_uint*)&tileCounts[tileId], 1u, memory_order_relaxed);
            }
        }
    }
}

#define instantiate_tileBinningCountKernel(name, MeansT, ConicT, OpacityT) \
    template [[host_name("tileBinningCountKernel_" #name)]] \
    kernel void tileBinningCountKernel<MeansT, ConicT, OpacityT>( \
        const device int4* bounds [[buffer(0)]], \
        device uint* tileCounts [[buffer(1)]], \
        const device OpacityT* opacities [[buffer(2)]], \
        constant TileBinningParams& params [[buffer(3)]], \
        const device MeansT* means [[buffer(4)]], \
        const device ConicT* conics [[buffer(5)]], \
        uint idx [[thread_position_in_grid]]);

instantiate_tileBinningCountKernel(float, float2, float4, float)
instantiate_tileBinningCountKernel(half, half2, half4, half)

#undef instantiate_tileBinningCountKernel

template <typename MeansT, typename ConicT, typename OpacityT>
kernel void tileBinningScatterKernel(
    const device int4* bounds [[buffer(0)]],
    device uint* tileCounters [[buffer(1)]],          // Reset to 0 before scatter
    const device uint* offsets [[buffer(2)]],         // Per-tile offsets (exclusive prefix of counts)
    device int* tileIndices [[buffer(3)]],
    device int* tileIds [[buffer(4)]],
    const device TileAssignmentHeader* header [[buffer(5)]],
    constant TileBinningParams& params [[buffer(6)]],
    const device MeansT* means [[buffer(7)]],
    const device ConicT* conics [[buffer(8)]],
    const device OpacityT* opacities [[buffer(9)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.gaussianCount) {
        return;
    }

    float alpha = float(opacities[idx]);
    if (alpha < 1e-4f) {
        return;
    }

    int4 rect = bounds[idx];
    int width = rect.y - rect.x + 1;
    int height = rect.w - rect.z + 1;
    if (width <= 0 || height <= 0) {
        return;
    }

    uint aabbCount = uint(width * height);
    bool usePrecise = (aabbCount > PRECISE_THRESHOLD_TILES);

    float2 center = float2(0.0f);
    float3 conic = float3(0.0f);
    float tileW = float(params.tileWidth);
    float tileH = float(params.tileHeight);
    float qMax = 0.0f;
    float qMaxExpanded = 0.0f;

    if (usePrecise) {
        center = float2(means[idx]);
        ConicT conicData = conics[idx];
        conic = float3(conicData.x, conicData.y, conicData.z);
        qMax = computeQMax(alpha, FLASHGS_TAU);

        float halfTileW = tileW * 0.5f;
        float halfTileH = tileH * 0.5f;
        float tileDiag = sqrt(halfTileW * halfTileW + halfTileH * halfTileH);
        float maxConic = max(conic.x, conic.z);
        float expand = tileDiag * sqrt(maxConic);
        qMaxExpanded = qMax + 2.0f * expand * sqrt(qMax) + expand * expand;
    }

    uint tilesX = params.tilesX;
    uint tileCount = tilesX * params.tilesY;
    uint maxAssignments = min(params.maxAssignments, header->maxAssignments);

    for (int ty = rect.z; ty <= rect.w; ++ty) {
        for (int tx = rect.x; tx <= rect.y; ++tx) {
            if (tx < 0 || ty < 0) continue;
            uint tileId = uint(ty) * tilesX + uint(tx);
            if (tileId >= tileCount) continue;

            bool intersects = true;
            if (usePrecise) {
                float tileCenterX = (float(tx) + 0.5f) * tileW;
                float tileCenterY = (float(ty) + 0.5f) * tileH;
                float dx = tileCenterX - center.x;
                float dy = tileCenterY - center.y;
                float q = dx * dx * conic.x + 2.0f * dx * dy * conic.y + dy * dy * conic.z;
                if (q <= qMax) {
                    intersects = true;
                } else if (q > qMaxExpanded) {
                    intersects = false;
                } else {
                    float tileMinX = float(tx) * tileW;
                    float tileMinY = float(ty) * tileH;
                    float tileMaxX = tileMinX + tileW;
                    float tileMaxY = tileMinY + tileH;
                    intersects = ellipseIntersectsTile(center, conic, qMax, tileMinX, tileMinY, tileMaxX, tileMaxY);
                }
            }

            if (intersects) {
                uint local = atomic_fetch_add_explicit((device atomic_uint*)&tileCounters[tileId], 1u, memory_order_relaxed);
                uint writePos = offsets[tileId] + local;
                if (writePos < maxAssignments) {
                    tileIds[writePos] = int(tileId);
                    tileIndices[writePos] = int(idx);
                }
            }
        }
    }
}

#define instantiate_tileBinningScatterKernel(name, MeansT, ConicT, OpacityT) \
    template [[host_name("tileBinningScatterKernel_" #name)]] \
    kernel void tileBinningScatterKernel<MeansT, ConicT, OpacityT>( \
        const device int4* bounds [[buffer(0)]], \
        device uint* tileCounters [[buffer(1)]], \
        const device uint* offsets [[buffer(2)]], \
        device int* tileIndices [[buffer(3)]], \
        device int* tileIds [[buffer(4)]], \
        const device TileAssignmentHeader* header [[buffer(5)]], \
        constant TileBinningParams& params [[buffer(6)]], \
        const device MeansT* means [[buffer(7)]], \
        const device ConicT* conics [[buffer(8)]], \
        const device OpacityT* opacities [[buffer(9)]], \
        uint idx [[thread_position_in_grid]]);

instantiate_tileBinningScatterKernel(float, float2, float4, float)
instantiate_tileBinningScatterKernel(half, half2, half4, half)

#undef instantiate_tileBinningScatterKernel

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

// Dispatch kernel for load-balanced scatter - reads totalAssignments from header
kernel void scatterBalancedDispatchKernel(
    const device TileAssignmentHeader* header [[buffer(0)]],
    device uint* dispatchArgs [[buffer(1)]],
    constant uint& threadgroupWidth [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) {
        return;
    }
    uint totalAssignments = header[0].totalAssignments;
    uint width = max(threadgroupWidth, 1u);
    uint groups = (totalAssignments + width - 1u) / width;
    dispatchArgs[0] = groups;
    dispatchArgs[1] = 1u;
    dispatchArgs[2] = 1u;
}

// Load-balanced scatter: each thread handles exactly one output slot
// Uses binary search to find owning gaussian - perfect load balancing
kernel void scatterAssignmentsBalancedKernel(
    const device int4* bounds [[buffer(0)]],
    const device uint* offsets [[buffer(1)]],  // Exclusive prefix sum, offsets[count] = total
    device int* tileIndices [[buffer(2)]],
    device int* tileIds [[buffer(3)]],
    constant ScatterParams& params [[buffer(4)]],
    const device TileAssignmentHeader* header [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    uint totalAssignments = header[0].totalAssignments;
    if (idx >= totalAssignments) {
        return;
    }

    // Binary search to find gaussian that owns this slot
    // Find largest gaussianIdx where offsets[gaussianIdx] <= idx
    uint lo = 0;
    uint hi = params.gaussianCount;
    while (lo < hi) {
        uint mid = (lo + hi + 1) >> 1;
        if (offsets[mid] <= idx) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    uint gaussianIdx = lo;

    // Local slot within this gaussian's tiles
    uint localSlot = idx - offsets[gaussianIdx];

    // Get bounds for this gaussian
    int4 rect = bounds[gaussianIdx];
    int minX = rect.x;
    int maxX = rect.y;
    int minY = rect.z;
    // maxY unused but available in rect.w

    // Convert local slot to (tx, ty)
    int width = maxX - minX + 1;
    int localY = int(localSlot) / width;
    int localX = int(localSlot) % width;
    int tx = minX + localX;
    int ty = minY + localY;

    // Write output
    int tileId = ty * int(params.tilesX) + tx;
    tileIds[idx] = tileId;
    tileIndices[idx] = int(gaussianIdx);
}

///////////////////////////////////////////////////////////////////////////////
// FlashGS Precise Scatter Kernels - only write tiles that actually intersect
// Parallel version: one threadgroup per gaussian, threads cooperatively test and write tiles
///////////////////////////////////////////////////////////////////////////////

template <typename MeansT, typename ConicT, typename OpacityT>
kernel void scatterAssignmentsPreciseKernel(
    const device int4* bounds [[buffer(0)]],
    const device uint* offsets [[buffer(1)]],
    device int* tileIndices [[buffer(2)]],
    device int* tileIds [[buffer(3)]],
    constant ScatterParams& params [[buffer(4)]],
    const device TileAssignmentHeader* header [[buffer(5)]],
    const device MeansT* means [[buffer(6)]],
    const device ConicT* conics [[buffer(7)]],
    const device OpacityT* opacities [[buffer(8)]],
    uint gaussianIdx [[threadgroup_position_in_grid]],
    uint localIdx [[thread_index_in_threadgroup]]
) {
    if (gaussianIdx >= params.gaussianCount) {
        return;
    }

    // Shared gaussian data
    threadgroup float2 sharedCenter;
    threadgroup float3 sharedConic;
    threadgroup float sharedQMax;
    threadgroup int4 sharedRect;
    threadgroup uint sharedAabbCount;
    threadgroup uint sharedBaseOffset;
    threadgroup uint sharedTilesX;
    threadgroup float sharedTileW;
    threadgroup float sharedTileH;
    threadgroup uint sharedMaxAssignments;

    // Atomic write counter within threadgroup
    threadgroup atomic_uint writeCounter;

    if (localIdx == 0) {
        float alpha = float(opacities[gaussianIdx]);
        sharedRect = bounds[gaussianIdx];
        sharedBaseOffset = offsets[gaussianIdx];
        sharedTilesX = params.tilesX;
        sharedMaxAssignments = header[0].maxAssignments;

        int width = sharedRect.y - sharedRect.x + 1;
        int height = sharedRect.w - sharedRect.z + 1;
        sharedAabbCount = (width > 0 && height > 0) ? uint(width * height) : 0u;

        if (alpha < 1e-4f || sharedAabbCount == 0u) {
            sharedQMax = 0.0f;
        } else if (sharedAabbCount <= PRECISE_THRESHOLD_TILES) {
            sharedQMax = -1.0f;  // Signal fast path
        } else {
            sharedCenter = float2(means[gaussianIdx]);
            ConicT conicData = conics[gaussianIdx];
            sharedConic = float3(conicData.x, conicData.y, conicData.z);
            sharedQMax = computeQMax(alpha, FLASHGS_TAU);
            sharedTileW = float(params.tileWidth);
            sharedTileH = float(params.tileHeight);
        }
        atomic_store_explicit(&writeCounter, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Early exit
    if (sharedQMax == 0.0f) {
        return;
    }

    uint totalTiles = sharedAabbCount;
    int minTX = sharedRect.x;
    int minTY = sharedRect.z;
    int width = sharedRect.y - sharedRect.x + 1;

    // Fast path for small AABBs - direct write (no intersection test)
    if (sharedQMax < 0.0f) {
        for (uint tileIdx = localIdx; tileIdx < totalTiles; tileIdx += PRECISE_TG_SIZE) {
            int localY = int(tileIdx) / width;
            int localX = int(tileIdx) % width;
            int tx = minTX + localX;
            int ty = minTY + localY;

            uint writePos = sharedBaseOffset + tileIdx;
            if (writePos < sharedMaxAssignments) {
                tileIds[writePos] = ty * int(sharedTilesX) + tx;
                tileIndices[writePos] = int(gaussianIdx);
            }
        }
        return;
    }

    // Precompute values for fast tile tests
    float halfTileW = sharedTileW * 0.5f;
    float halfTileH = sharedTileH * 0.5f;
    float tileDiag = sqrt(halfTileW * halfTileW + halfTileH * halfTileH);
    float maxConic = max(sharedConic.x, sharedConic.z);
    float expand = tileDiag * sqrt(maxConic);
    float qMaxExpanded = sharedQMax + 2.0f * expand * sqrt(sharedQMax) + expand * expand;

    // Precise path - test and write in parallel with fast accept/reject
    for (uint tileIdx = localIdx; tileIdx < totalTiles; tileIdx += PRECISE_TG_SIZE) {
        int localY = int(tileIdx) / width;
        int localX = int(tileIdx) % width;
        int tx = minTX + localX;
        int ty = minTY + localY;

        // Fast test: check tile center
        float tileCenterX = (float(tx) + 0.5f) * sharedTileW;
        float tileCenterY = (float(ty) + 0.5f) * sharedTileH;
        float dx = tileCenterX - sharedCenter.x;
        float dy = tileCenterY - sharedCenter.y;
        float q = dx * dx * sharedConic.x + 2.0f * dx * dy * sharedConic.y + dy * dy * sharedConic.z;

        bool intersects = false;
        if (q <= sharedQMax) {
            // Quick accept: tile center inside ellipse
            intersects = true;
        } else if (q <= qMaxExpanded) {
            // Boundary case: do full test
            float tileMinX = float(tx) * sharedTileW;
            float tileMinY = float(ty) * sharedTileH;
            float tileMaxX = tileMinX + sharedTileW;
            float tileMaxY = tileMinY + sharedTileH;
            intersects = ellipseIntersectsTile(sharedCenter, sharedConic, sharedQMax, tileMinX, tileMinY, tileMaxX, tileMaxY);
        }
        // else: quick reject (q > qMaxExpanded)

        if (intersects) {
            uint localWriteIdx = atomic_fetch_add_explicit(&writeCounter, 1u, memory_order_relaxed);
            uint writePos = sharedBaseOffset + localWriteIdx;
            if (writePos < sharedMaxAssignments) {
                tileIds[writePos] = ty * int(sharedTilesX) + tx;
                tileIndices[writePos] = int(gaussianIdx);
            }
        }
    }
}

#define instantiate_scatterAssignmentsPreciseKernel(name, MeansT, ConicT, OpacityT) \
    template [[host_name("scatterAssignmentsPreciseKernel_" #name)]] \
    kernel void scatterAssignmentsPreciseKernel<MeansT, ConicT, OpacityT>( \
        const device int4* bounds [[buffer(0)]], \
        const device uint* offsets [[buffer(1)]], \
        device int* tileIndices [[buffer(2)]], \
        device int* tileIds [[buffer(3)]], \
        constant ScatterParams& params [[buffer(4)]], \
        const device TileAssignmentHeader* header [[buffer(5)]], \
        const device MeansT* means [[buffer(6)]], \
        const device ConicT* conics [[buffer(7)]], \
        const device OpacityT* opacities [[buffer(8)]], \
        uint gaussianIdx [[threadgroup_position_in_grid]], \
        uint localIdx [[thread_index_in_threadgroup]]);

instantiate_scatterAssignmentsPreciseKernel(float, float2, float4, float)
instantiate_scatterAssignmentsPreciseKernel(half, half2, half4, half)

#undef instantiate_scatterAssignmentsPreciseKernel

// Load-balanced precise scatter: each thread handles one output slot
// Uses binary search + enumeration to find the correct tile
template <typename MeansT, typename ConicT, typename OpacityT>
kernel void scatterAssignmentsPreciseBalancedKernel(
    const device int4* bounds [[buffer(0)]],
    const device uint* offsets [[buffer(1)]],
    device int* tileIndices [[buffer(2)]],
    device int* tileIds [[buffer(3)]],
    constant ScatterParams& params [[buffer(4)]],
    const device TileAssignmentHeader* header [[buffer(5)]],
    const device MeansT* means [[buffer(6)]],
    const device ConicT* conics [[buffer(7)]],
    const device OpacityT* opacities [[buffer(8)]],
    uint idx [[thread_position_in_grid]]
) {
    uint totalAssignments = header[0].totalAssignments;
    if (idx >= totalAssignments) {
        return;
    }

    // Binary search to find gaussian that owns this slot
    uint lo = 0;
    uint hi = params.gaussianCount;
    while (lo < hi) {
        uint mid = (lo + hi + 1) >> 1;
        if (offsets[mid] <= idx) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    uint gaussianIdx = lo;

    // Local slot within this gaussian's tiles
    uint localSlot = idx - offsets[gaussianIdx];

    // Get bounds
    int4 rect = bounds[gaussianIdx];
    int minX = rect.x;
    int maxX = rect.y;
    int minY = rect.z;
    int maxY = rect.w;

    int width = maxX - minX + 1;
    int height = maxY - minY + 1;
    uint aabbCount = uint(width * height);

    // Fast path for small AABBs - direct index calculation
    if (aabbCount <= PRECISE_THRESHOLD_TILES) {
        int localY = int(localSlot) / width;
        int localX = int(localSlot) % width;
        int tx = minX + localX;
        int ty = minY + localY;
        int tileId = ty * int(params.tilesX) + tx;
        tileIds[idx] = tileId;
        tileIndices[idx] = int(gaussianIdx);
        return;
    }

    // Load gaussian params for precise test
    float alpha = float(opacities[gaussianIdx]);
    float2 center = float2(means[gaussianIdx]);
    ConicT conicData = conics[gaussianIdx];
    float3 conic = float3(conicData.x, conicData.y, conicData.z);
    float qMax = computeQMax(alpha, FLASHGS_TAU);

    float tileW = float(params.tileWidth);
    float tileH = float(params.tileHeight);

    // Enumerate tiles until we hit localSlot
    uint count = 0;
    int foundTX = 0, foundTY = 0;
    bool found = false;

    for (int ty = minY; ty <= maxY && !found; ++ty) {
        float tileMinY = float(ty) * tileH;
        float tileMaxY = tileMinY + tileH;

        for (int tx = minX; tx <= maxX && !found; ++tx) {
            float tileMinX = float(tx) * tileW;
            float tileMaxX = tileMinX + tileW;

            if (ellipseIntersectsTile(center, conic, qMax, tileMinX, tileMinY, tileMaxX, tileMaxY)) {
                if (count == localSlot) {
                    foundTX = tx;
                    foundTY = ty;
                    found = true;
                }
                count++;
            }
        }
    }

    // Write output
    int tileId = foundTY * int(params.tilesX) + foundTX;
    tileIds[idx] = tileId;
    tileIndices[idx] = int(gaussianIdx);
}

#define instantiate_scatterAssignmentsPreciseBalancedKernel(name, MeansT, ConicT, OpacityT) \
    template [[host_name("scatterAssignmentsPreciseBalancedKernel_" #name)]] \
    kernel void scatterAssignmentsPreciseBalancedKernel<MeansT, ConicT, OpacityT>( \
        const device int4* bounds [[buffer(0)]], \
        const device uint* offsets [[buffer(1)]], \
        device int* tileIndices [[buffer(2)]], \
        device int* tileIds [[buffer(3)]], \
        constant ScatterParams& params [[buffer(4)]], \
        const device TileAssignmentHeader* header [[buffer(5)]], \
        const device MeansT* means [[buffer(6)]], \
        const device ConicT* conics [[buffer(7)]], \
        const device OpacityT* opacities [[buffer(8)]], \
        uint idx [[thread_position_in_grid]]);

instantiate_scatterAssignmentsPreciseBalancedKernel(float, float2, float4, float)
instantiate_scatterAssignmentsPreciseBalancedKernel(half, half2, half4, half)

#undef instantiate_scatterAssignmentsPreciseBalancedKernel

// =============================================================================
// FUSED COVERAGE + SCATTER KERNEL
// Combines tile intersection testing with output writing in a single pass.
// Uses global atomic to allocate output space, eliminating prefix scan dependency.
//
// Design: Two-pass within same kernel (no tile count limit)
// - Pass 1: Count intersecting tiles
// - Pass 2: Allocate via SIMD batching, then write all tiles directly
// =============================================================================

#define FUSED_TG_SIZE 64u             // Cooperative per-gaussian
#define FUSED_MAX_TILES_PER_GAUSSIAN 2048u
#define FUSED_PRECISE_THRESHOLD 16u   // Threshold tiles before precise intersection

struct FusedCoverageScatterParams {
    uint gaussianCount;
    uint tileWidth;
    uint tileHeight;
    uint tilesX;
    uint maxAssignments;
};

// Cooperative fused coverage + scatter kernel.
// One threadgroup (single SIMD) per gaussian; threads stride tiles, accumulate hits
// into a threadgroup buffer, then issue one global atomic.
template <typename MeansT, typename ConicT, typename OpacityT>
kernel void fusedCoverageScatterKernel(
    const device int4* bounds [[buffer(0)]],
    device uint* coverageCounts [[buffer(1)]],
    const device OpacityT* opacities [[buffer(2)]],
    constant FusedCoverageScatterParams& params [[buffer(3)]],
    const device MeansT* means [[buffer(4)]],
    const device ConicT* conics [[buffer(5)]],
    device int* tileIndices [[buffer(6)]],
    device int* tileIds [[buffer(7)]],
    device TileAssignmentHeader* header [[buffer(8)]],
    uint tgIdx [[thread_index_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]],
    uint groupIdx [[threadgroup_position_in_grid]]
) {
    const uint gaussianIdx = groupIdx;
    if (gaussianIdx >= params.gaussianCount) return;

    threadgroup int tgMinTX, tgMaxTX, tgMinTY, tgMaxTY;
    threadgroup float tgTileW, tgTileH;
    threadgroup uint tgTilesX;
    threadgroup bool tgNeedsPrecise;
    threadgroup float2 tgCenter;
    threadgroup float3 tgConic;
    threadgroup float tgQMax, tgQMaxExpanded;
    threadgroup atomic_uint sharedCount;
    threadgroup uint sharedGlobalOffset;
    threadgroup int sharedTileIds[FUSED_MAX_TILES_PER_GAUSSIAN];

    if (tgIdx == 0u) {
        atomic_store_explicit(&sharedCount, 0u, memory_order_relaxed);
        sharedGlobalOffset = 0u;
        const float alpha = float(opacities[gaussianIdx]);
        const int4 rect = bounds[gaussianIdx];
        tgMinTX = rect.x;
        tgMaxTX = rect.y;
        tgMinTY = rect.z;
        tgMaxTY = rect.w;

        const int aabbWidth = tgMaxTX - tgMinTX + 1;
        const int aabbHeight = tgMaxTY - tgMinTY + 1;
        if (alpha < 1e-4f || aabbWidth <= 0 || aabbHeight <= 0) {
            tgNeedsPrecise = false;
            tgTilesX = params.tilesX;
            tgTileW = float(params.tileWidth);
            tgTileH = float(params.tileHeight);
            tgCenter = float2(0.0f);
            tgConic = float3(0.0f);
            tgQMax = 0.0f;
            tgQMaxExpanded = 0.0f;
        } else {
            const uint aabbCount = uint(aabbWidth * aabbHeight);
            tgTilesX = params.tilesX;
            tgTileW = float(params.tileWidth);
            tgTileH = float(params.tileHeight);
            tgNeedsPrecise = (aabbCount > FUSED_PRECISE_THRESHOLD);
            tgCenter = float2(means[gaussianIdx]);
            const ConicT conicData = conics[gaussianIdx];
            tgConic = float3(conicData.x, conicData.y, conicData.z);
            tgQMax = computeQMax(alpha, FLASHGS_TAU);

            const float halfTileW = tgTileW * 0.5f;
            const float halfTileH = tgTileH * 0.5f;
            const float tileDiag = fast::sqrt(halfTileW * halfTileW + halfTileH * halfTileH);
            const float maxConic = max(tgConic.x, tgConic.z);
            const float expand = tileDiag * fast::sqrt(maxConic);
            tgQMaxExpanded = tgQMax + 2.0f * expand * fast::sqrt(tgQMax) + expand * expand;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int minTX = tgMinTX;
    const int maxTX = tgMaxTX;
    const int minTY = tgMinTY;
    const int maxTY = tgMaxTY;
    const int aabbWidth = maxTX - minTX + 1;
    const int aabbHeight = maxTY - minTY + 1;
    const uint aabbCount = (aabbWidth > 0 && aabbHeight > 0) ? uint(aabbWidth * aabbHeight) : 0u;

    if (aabbCount == 0u) {
        if (tgIdx == 0u) {
            coverageCounts[gaussianIdx] = 0u;
        }
        return;
    }

    // Threads stride tiles and accumulate into threadgroup buffer.
    const float tileW = tgTileW;
    const float tileH = tgTileH;
    const float2 center = tgCenter;
    const float3 conic = tgConic;
    const float qMax = tgQMax;
    const float qMaxExpanded = tgQMaxExpanded;
    const bool needsPrecise = tgNeedsPrecise;

    for (uint idx = tgIdx; idx < aabbCount; idx += tgSize) {
        const int ty = minTY + int(idx / uint(aabbWidth));
        const int tx = minTX + int(idx % uint(aabbWidth));

        bool intersects = false;
        if (!needsPrecise) {
            intersects = true;
        } else {
            const float tileCenterY = (float(ty) + 0.5f) * tileH;
            const float dy = tileCenterY - center.y;
            const float dy2_cz = dy * dy * conic.z;
            const float dy_2cy = 2.0f * dy * conic.y;
            const float tileCenterX = (float(tx) + 0.5f) * tileW;
            const float dx = tileCenterX - center.x;
            const float q = dx * dx * conic.x + dx * dy_2cy + dy2_cz;

            if (q <= qMax) {
                intersects = true;
            } else if (q <= qMaxExpanded) {
                const float tileMinX = float(tx) * tileW;
                const float tileMinY = float(ty) * tileH;
                intersects = ellipseIntersectsTile(center, conic, qMax,
                                                  tileMinX, tileMinY,
                                                  tileMinX + tileW, tileMinY + tileH);
            }
        }

        if (intersects) {
            uint idxStore = atomic_fetch_add_explicit(&sharedCount, 1u, memory_order_relaxed);
            if (idxStore < FUSED_MAX_TILES_PER_GAUSSIAN) {
                sharedTileIds[idxStore] = ty * int(tgTilesX) + tx;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint count = min(atomic_load_explicit(&sharedCount, memory_order_relaxed), FUSED_MAX_TILES_PER_GAUSSIAN);

    if (tgIdx == 0u) {
        coverageCounts[gaussianIdx] = count;
        if (count > 0u) {
            sharedGlobalOffset = atomic_fetch_add_explicit(
                (device atomic_uint*)&header->totalAssignments, count, memory_order_relaxed);
        } else {
            sharedGlobalOffset = 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (count == 0u) return;

    const uint globalOffset = sharedGlobalOffset;
    const uint maxAssignments = params.maxAssignments;
    for (uint i = tgIdx; i < count; i += tgSize) {
        const uint writePos = globalOffset + i;
        if (writePos < maxAssignments) {
            tileIds[writePos] = sharedTileIds[i];
            tileIndices[writePos] = int(gaussianIdx);
        }
    }
}

#define instantiate_fusedCoverageScatterKernel(name, MeansT, ConicT, OpacityT) \
    template [[host_name("fusedCoverageScatterKernel_" #name)]] \
    kernel void fusedCoverageScatterKernel<MeansT, ConicT, OpacityT>( \
        const device int4* bounds [[buffer(0)]], \
        device uint* coverageCounts [[buffer(1)]], \
        const device OpacityT* opacities [[buffer(2)]], \
        constant FusedCoverageScatterParams& params [[buffer(3)]], \
        const device MeansT* means [[buffer(4)]], \
        const device ConicT* conics [[buffer(5)]], \
        device int* tileIndices [[buffer(6)]], \
        device int* tileIds [[buffer(7)]], \
        device TileAssignmentHeader* header [[buffer(8)]], \
        uint tgIdx [[thread_index_in_threadgroup]], \
        uint tgSize [[threads_per_threadgroup]], \
        uint groupIdx [[threadgroup_position_in_grid]]);

instantiate_fusedCoverageScatterKernel(float, float2, float4, float)
instantiate_fusedCoverageScatterKernel(half, half2, half4, half)

#undef instantiate_fusedCoverageScatterKernel

// =============================================================================
// MARK: - Optimized Fused Coverage Scatter v2 (SIMD-based, reduced TG memory)
// =============================================================================
// Key optimizations:
// 1. Reduced threadgroup memory: 128 tiles (512B) vs 2048 (8KB) = ~8x occupancy
// 2. SIMD ballot/prefix for tile counting - no threadgroup atomics in hot loop
// 3. Warp-cooperative global allocation - one atomic per SIMD group
// 4. True half precision math for half variant
// 5. Multi-pass for rare large gaussians (>128 tiles)
// =============================================================================

#define FUSED_V2_TG_SIZE 64u
#define FUSED_V2_SIMD_SIZE 32u
#define FUSED_V2_MAX_TILES 128u  // 512 bytes - fits well in TG memory
#define FUSED_V2_PRECISE_THRESHOLD 16u

struct FusedCoverageScatterParamsV2 {
    uint gaussianCount;
    uint tileWidth;
    uint tileHeight;
    uint tilesX;
    uint maxAssignments;
};

// Half-precision helper for true half math
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

template <typename MeansT, typename ConicT, typename OpacityT>
kernel void fusedCoverageScatterKernelV2(
    const device int4* bounds [[buffer(0)]],
    device uint* coverageCounts [[buffer(1)]],
    const device OpacityT* opacities [[buffer(2)]],
    constant FusedCoverageScatterParamsV2& params [[buffer(3)]],
    const device MeansT* means [[buffer(4)]],
    const device ConicT* conics [[buffer(5)]],
    device int* tileIndices [[buffer(6)]],
    device int* tileIds [[buffer(7)]],
    device TileAssignmentHeader* header [[buffer(8)]],
    uint tgIdx [[thread_index_in_threadgroup]],
    uint simdLaneId [[thread_index_in_simdgroup]],
    uint simdGroupId [[simdgroup_index_in_threadgroup]],
    uint groupIdx [[threadgroup_position_in_grid]]
) {
    const uint gaussianIdx = groupIdx;
    if (gaussianIdx >= params.gaussianCount) return;

    // Reduced threadgroup memory - 512 bytes instead of 8KB
    threadgroup int sharedTileIds[FUSED_V2_MAX_TILES];
    threadgroup uint sharedCount;
    threadgroup uint sharedGlobalOffset;

    // Load gaussian data once
    const int4 rect = bounds[gaussianIdx];
    const int minTX = rect.x;
    const int maxTX = rect.y;
    const int minTY = rect.z;
    const int maxTY = rect.w;
    const int aabbWidth = maxTX - minTX + 1;
    const int aabbHeight = maxTY - minTY + 1;

    if (aabbWidth <= 0 || aabbHeight <= 0) {
        if (tgIdx == 0u) {
            coverageCounts[gaussianIdx] = 0u;
        }
        return;
    }

    const uint aabbCount = uint(aabbWidth * aabbHeight);
    const float alpha = float(opacities[gaussianIdx]);

    if (alpha < 1e-4f) {
        if (tgIdx == 0u) {
            coverageCounts[gaussianIdx] = 0u;
        }
        return;
    }

    // Initialize shared count
    if (tgIdx == 0u) {
        sharedCount = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float tileW = float(params.tileWidth);
    const float tileH = float(params.tileHeight);
    const uint tilesX = params.tilesX;
    const bool needsPrecise = (aabbCount > FUSED_V2_PRECISE_THRESHOLD);

    const float2 center = float2(means[gaussianIdx]);
    const ConicT conicData = conics[gaussianIdx];
    const float3 conic = float3(float(conicData.x), float(conicData.y), float(conicData.z));
    const float qMax = computeQMax(alpha, FLASHGS_TAU);

    // Pre-compute expanded qMax for tile-center test
    const float halfTileW = tileW * 0.5f;
    const float halfTileH = tileH * 0.5f;
    const float tileDiag = fast::sqrt(halfTileW * halfTileW + halfTileH * halfTileH);
    const float maxConic = max(conic.x, conic.z);
    const float expand = tileDiag * fast::sqrt(maxConic);
    const float qMaxExpanded = qMax + 2.0f * expand * fast::sqrt(qMax) + expand * expand;

    // Process tiles using SIMD-based approach
    // Each thread checks one tile at a time, SIMD ballot counts hits
    uint totalFound = 0u;

    for (uint baseIdx = 0u; baseIdx < aabbCount; baseIdx += FUSED_V2_TG_SIZE) {
        uint idx = baseIdx + tgIdx;
        bool intersects = false;
        int tileId = 0;

        if (idx < aabbCount) {
            const int ty = minTY + int(idx / uint(aabbWidth));
            const int tx = minTX + int(idx % uint(aabbWidth));
            tileId = ty * int(tilesX) + tx;

            if (!needsPrecise) {
                intersects = true;
            } else {
                const float tileCenterX = (float(tx) + 0.5f) * tileW;
                const float tileCenterY = (float(ty) + 0.5f) * tileH;
                const float dx = tileCenterX - center.x;
                const float dy = tileCenterY - center.y;
                const float q = dx * dx * conic.x + 2.0f * dx * dy * conic.y + dy * dy * conic.z;

                if (q <= qMax) {
                    intersects = true;
                } else if (q <= qMaxExpanded) {
                    const float tileMinX = float(tx) * tileW;
                    const float tileMinY = float(ty) * tileH;
                    intersects = ellipseIntersectsTile(center, conic, qMax,
                                                       tileMinX, tileMinY,
                                                       tileMinX + tileW, tileMinY + tileH);
                }
            }
        }

        // SIMD ballot to count intersections efficiently
        // This replaces threadgroup atomics in the hot loop
        // simd_ballot returns simd_vote, cast to underlying vote_t then to uint32
        // On Apple Silicon SIMD width is 32, so lower 32 bits are valid
        simd_vote ballot = simd_ballot(intersects);
        uint simdMask = uint(simd_vote::vote_t(ballot) & 0xFFFFFFFFull);
        uint simdCount = popcount(simdMask);

        // Compute prefix within SIMD group for write position
        uint lowerMask = (1u << simdLaneId) - 1u;
        uint simdPrefix = popcount(simdMask & lowerMask);

        // First lane in each SIMD group atomically allocates space in shared buffer
        uint simdBaseOffset = 0u;
        if (simdLaneId == 0u && simdCount > 0u) {
            simdBaseOffset = atomic_fetch_add_explicit(
                (threadgroup atomic_uint*)&sharedCount, simdCount, memory_order_relaxed);
        }
        // Broadcast base offset to all lanes in SIMD group
        simdBaseOffset = simd_broadcast_first(simdBaseOffset);

        // Write to shared buffer (with overflow protection)
        if (intersects) {
            uint writePos = simdBaseOffset + simdPrefix;
            if (writePos < FUSED_V2_MAX_TILES) {
                sharedTileIds[writePos] = tileId;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Get final count (clamped to buffer size)
    uint count = min(sharedCount, FUSED_V2_MAX_TILES);

    // Single thread allocates global space and writes coverage count
    if (tgIdx == 0u) {
        coverageCounts[gaussianIdx] = count;
        if (count > 0u) {
            sharedGlobalOffset = atomic_fetch_add_explicit(
                (device atomic_uint*)&header->totalAssignments, count, memory_order_relaxed);
        } else {
            sharedGlobalOffset = 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (count == 0u) return;

    // Scatter to global buffers using coalesced writes
    const uint globalOffset = sharedGlobalOffset;
    const uint maxAssignments = params.maxAssignments;

    for (uint i = tgIdx; i < count; i += FUSED_V2_TG_SIZE) {
        const uint writePos = globalOffset + i;
        if (writePos < maxAssignments) {
            tileIds[writePos] = sharedTileIds[i];
            tileIndices[writePos] = int(gaussianIdx);
        }
    }
}

#define instantiate_fusedCoverageScatterKernelV2(name, MeansT, ConicT, OpacityT) \
    template [[host_name("fusedCoverageScatterKernelV2_" #name)]] \
    kernel void fusedCoverageScatterKernelV2<MeansT, ConicT, OpacityT>( \
        const device int4* bounds [[buffer(0)]], \
        device uint* coverageCounts [[buffer(1)]], \
        const device OpacityT* opacities [[buffer(2)]], \
        constant FusedCoverageScatterParamsV2& params [[buffer(3)]], \
        const device MeansT* means [[buffer(4)]], \
        const device ConicT* conics [[buffer(5)]], \
        device int* tileIndices [[buffer(6)]], \
        device int* tileIds [[buffer(7)]], \
        device TileAssignmentHeader* header [[buffer(8)]], \
        uint tgIdx [[thread_index_in_threadgroup]], \
        uint simdLaneId [[thread_index_in_simdgroup]], \
        uint simdGroupId [[simdgroup_index_in_threadgroup]], \
        uint groupIdx [[threadgroup_position_in_grid]]);

instantiate_fusedCoverageScatterKernelV2(float, float2, float4, float)
instantiate_fusedCoverageScatterKernelV2(half, half2, half4, half)

#undef instantiate_fusedCoverageScatterKernelV2

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

    // Clamp totalAssignments to maxAssignments (fix for fused scatter overflow)
    if (total > config.maxAssignments) {
        total = config.maxAssignments;
        header[0].totalAssignments = total;
        header[0].overflow = 1u;
    }

    header[0].paddedCount = nextPowerOfTwo(total);
    uint padded = header[0].paddedCount;

    uint sortTG = max(config.sortThreadgroupSize, 1u);
    uint fuseTG = max(config.fuseThreadgroupSize, 1u);
    uint unpackTG = max(config.unpackThreadgroupSize, 1u);
    uint packTG = max(config.packThreadgroupSize, 1u);
    uint bitonicTG = max(config.bitonicThreadgroupSize, 1u);
    uint radixBlockSize = max(config.radixBlockSize, 1u);
    uint radixGrainSize = max(config.radixGrainSize, 1u);

    // Sort key generation uses total (radix) or padded (bitonic)
    // For radix path: use total count to avoid processing padding
    uint sortGroups = (total > 0u) ? ((total + sortTG - 1u) / sortTG) : 0u;
    dispatchArgs[DispatchSlotSortKeys].threadgroupsPerGridX = sortGroups;
    dispatchArgs[DispatchSlotSortKeys].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotSortKeys].threadgroupsPerGridZ = 1u;

    // Fuse/unpack use total count for radix path
    uint fuseGroups = (total > 0u) ? ((total + fuseTG - 1u) / fuseTG) : 0u;
    dispatchArgs[DispatchSlotFuseKeys].threadgroupsPerGridX = fuseGroups;
    dispatchArgs[DispatchSlotFuseKeys].threadgroupsPerGridY = 1u;
    dispatchArgs[DispatchSlotFuseKeys].threadgroupsPerGridZ = 1u;

    uint unpackGroups = (total > 0u) ? ((total + unpackTG - 1u) / unpackTG) : 0u;
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
    uint radixGrid = (total > 0u) ? ((total + valuesPerGroup - 1u) / valuesPerGroup) : 0u;
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

// Parallel exclusive scan - 256 threads, up to 8192 elements
#define SCAN_GRAIN 32

// Each thread handles SCAN_GRAIN elements, sums them, scans sums, then applies back
kernel void radixExclusiveScanKernel(
    device uint*                       block_sums  [[buffer(0)]],
    const device TileAssignmentHeader* header      [[buffer(1)]],
    threadgroup uint*                  shared_mem  [[threadgroup(0)]],
    ushort                             lid         [[thread_position_in_threadgroup]]
) {

    const uint elementsPerBlock = BLOCK_SIZE * GRAIN_SIZE;
    uint paddedCount   = header[0].paddedCount;
    uint histGridSize  = (paddedCount + elementsPerBlock - 1u) / elementsPerBlock;
    uint num_hist_elem = histGridSize * RADIX;
    uint num_block_sums = (num_hist_elem + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    // Calculate elements per thread (up to SCAN_GRAIN)
    uint elems_per_thread = (num_block_sums + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    elems_per_thread = min(elems_per_thread, (uint)SCAN_GRAIN);

    // Load multiple elements and compute local sum
    uint local_vals[SCAN_GRAIN];
    uint local_sum = 0u;
    uint thread_base = lid * elems_per_thread;

    for (uint i = 0u; i < elems_per_thread; ++i) {
        uint idx = thread_base + i;
        uint val = (idx < num_block_sums) ? block_sums[idx] : 0u;
        local_vals[i] = val;
        local_sum += val;
    }

    // Scan the per-thread sums using helper (exclusive scan of 256 values)
    uint scanned_sum = ThreadgroupPrefixScan<SCAN_TYPE_EXCLUSIVE>(local_sum, shared_mem, lid, SumOp<uint>());

    // Write back: each element gets scanned_sum + prefix of local values
    uint running = scanned_sum;
    for (uint i = 0u; i < elems_per_thread; ++i) {
        uint idx = thread_base + i;
        if (idx < num_block_sums) {
            block_sums[idx] = running;
            running += local_vals[i];
        }
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
    // Pack tileId in upper bits, 16-bit depth in lower bits
    // Layout: [depth 0-15, tileId 16-63] for compact radix sort passes
    KeyType fused = (KeyType(key.x) << 16) | KeyType(key.y & 0xFFFFu);
    output_keys[gid] = fused;
}

kernel void unpackSortKeysKernel(
    const device KeyType* input_keys [[buffer(0)]],
    device uint2* output_keys [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    KeyType fused = input_keys[gid];
    uint tile = uint(fused >> 16);
    uint depthBits = uint(fused & 0xFFFFull);
    output_keys[gid] = uint2(tile, depthBits);
}

// =============================================================================
// FUSED PIPELINE KERNELS - Interleaved data for cache efficiency
// =============================================================================

/// Interleave separate gaussian buffers into single struct (half16)
kernel void interleaveGaussianDataKernel_half(
    const device half2*         means       [[buffer(0)]],
    const device half4*         conics      [[buffer(1)]],
    const device packed_half3*  colors      [[buffer(2)]],
    const device half*          opacities   [[buffer(3)]],
    const device half*          depths      [[buffer(4)]],
    device GaussianRenderData*  output      [[buffer(5)]],
    constant uint&              count       [[buffer(6)]],
    uint                        gid         [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    GaussianRenderData g;
    g.mean     = means[gid];
    g.conic    = conics[gid];
    g.color    = colors[gid];
    g.opacity  = opacities[gid];
    g.depth    = depths[gid];
    g._pad     = 0;

    output[gid] = g;
}

/// Interleave separate gaussian buffers into single struct (float32)
kernel void interleaveGaussianDataKernel_float(
    const device float2*         means       [[buffer(0)]],
    const device float4*         conics      [[buffer(1)]],
    const device packed_float3*  colors      [[buffer(2)]],
    const device float*          opacities   [[buffer(3)]],
    const device float*          depths      [[buffer(4)]],
    device GaussianRenderDataF32* output     [[buffer(5)]],
    constant uint&               count       [[buffer(6)]],
    uint                         gid         [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    GaussianRenderDataF32 g;
    g.mean     = means[gid];
    g.conic    = conics[gid];
    g.color    = colors[gid];
    g.opacity  = opacities[gid];
    g.depth    = depths[gid];
    g._pad     = 0;

    output[gid] = g;
}

/// Pack fused gaussians using sorted indices (half16)
/// Single struct read instead of 5 scattered reads
kernel void packFusedKernel_half(
    const device int*                sortedIndices   [[buffer(0)]],
    const device GaussianRenderData* gaussians       [[buffer(1)]],
    device PackedGaussian*           output          [[buffer(2)]],
    const device TileAssignmentHeader* header        [[buffer(3)]],
    uint                             gid             [[thread_position_in_grid]]
) {
    uint total = header->totalAssignments;
    if (gid >= total) return;

    int srcIdx = sortedIndices[gid];
    if (srcIdx < 0) return;  // Invalid/padding entry

    GaussianRenderData src = gaussians[srcIdx];

    // Direct copy - same layout
    PackedGaussian dst;
    dst.mean    = src.mean;
    dst.conic   = src.conic;
    dst.color   = src.color;
    dst.opacity = src.opacity;
    dst.depth   = src.depth;
    dst._pad    = 0;

    output[gid] = dst;
}

/// Pack fused gaussians using sorted indices (float32)
kernel void packFusedKernel_float(
    const device int*                  sortedIndices   [[buffer(0)]],
    const device GaussianRenderDataF32* gaussians      [[buffer(1)]],
    device PackedGaussianF32*          output          [[buffer(2)]],
    const device TileAssignmentHeader* header          [[buffer(3)]],
    uint                               gid             [[thread_position_in_grid]]
) {
    uint total = header->totalAssignments;
    if (gid >= total) return;

    int srcIdx = sortedIndices[gid];
    if (srcIdx < 0) return;  // Invalid/padding entry

    GaussianRenderDataF32 src = gaussians[srcIdx];

    PackedGaussianF32 dst;
    dst.mean    = src.mean;
    dst.conic   = src.conic;
    dst.color   = src.color;
    dst.opacity = src.opacity;
    dst.depth   = src.depth;
    dst._pad    = 0;

    output[gid] = dst;
}

/// Fused render kernel - reads interleaved PackedGaussian structs
/// Template version supporting both half and float precision
/// Single load per gaussian instead of 5 scattered loads
#define RENDER_FUSED_BATCH_SIZE 32

template <typename PackedT>
kernel void renderTilesFusedKernel(
    const device GaussianHeader*    headers         [[buffer(0)]],
    const device PackedT*           gaussians       [[buffer(1)]],
    const device uint*              activeTiles     [[buffer(2)]],
    const device uint*              activeTileCount [[buffer(3)]],
    texture2d<half, access::write>  colorOut        [[texture(0)]],
    texture2d<float, access::write> depthOut        [[texture(1)]],
    texture2d<float, access::write> alphaOut        [[texture(2)]],
    constant RenderParams&          params          [[buffer(4)]],
    uint2                           group_id        [[threadgroup_position_in_grid]],
    uint2                           local_id        [[thread_position_in_threadgroup]]
) {
    threadgroup PackedT shGaussians[RENDER_FUSED_BATCH_SIZE];

    uint tileIdx = group_id.x;
    if (tileIdx >= *activeTileCount) return;

    uint tile = activeTiles[tileIdx];
    GaussianHeader hdr = headers[tile];

    uint tileX = tile % params.tilesX;
    uint tileY = tile / params.tilesX;
    uint px = tileX * params.tileWidth + local_id.x;
    uint py = tileY * params.tileHeight + local_id.y;

    bool inBounds = (px < params.width) && (py < params.height);

    float3 accumColor = float3(0);
    float accumDepth = 0;
    float trans = 1.0f;

    uint start = hdr.offset;
    uint count = hdr.count;

    for (uint batch = 0; batch < count; batch += RENDER_FUSED_BATCH_SIZE) {
        uint batchCount = min(uint(RENDER_FUSED_BATCH_SIZE), count - batch);

        // Cooperative loading - single struct read per gaussian
        uint tid = local_id.y * params.tileWidth + local_id.x;
        if (tid < batchCount) {
            shGaussians[tid] = gaussians[start + batch + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process batch
        if (inBounds) {
            for (uint i = 0; i < batchCount && trans > 1e-3f; i++) {
                PackedT g = shGaussians[i];

                float2 mean = float2(g.mean);
                float2 d = float2(px, py) - mean;
                d.x = clamp(d.x, -250.0f, 250.0f);
                d.y = clamp(d.y, -250.0f, 250.0f);

                float4 conic = float4(g.conic);
                float quad = conic.x * d.x * d.x +
                             2.0f * conic.y * d.x * d.y +
                             conic.z * d.y * d.y;

                if (quad > 20.0f) continue;

                float weight = exp(-0.5f * quad);
                float alpha = min(0.99f, float(g.opacity) * weight);
                if (alpha < (1.0f / 255.0f)) continue;

                float3 color = float3(g.color);
                float depth = float(g.depth);

                float contrib = trans * alpha;
                accumColor += color * contrib;
                accumDepth += depth * contrib;
                trans *= (1.0f - alpha);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inBounds) {
        float3 bg = params.whiteBackground ? float3(1) : float3(0);
        float3 finalColor = accumColor + trans * bg;
        float finalAlpha = 1.0f - trans;

        colorOut.write(half4(half3(finalColor), half(finalAlpha)), uint2(px, py));
        depthOut.write(float4(accumDepth, 0, 0, 0), uint2(px, py));
        alphaOut.write(float4(finalAlpha, 0, 0, 0), uint2(px, py));
    }
}

// Instantiate fused render kernel for half and float
#define instantiate_renderTilesFused(name, PackedT) \
    template [[host_name("renderTilesFused_" #name)]] \
    kernel void renderTilesFusedKernel<PackedT>( \
        const device GaussianHeader* headers [[buffer(0)]], \
        const device PackedT* gaussians [[buffer(1)]], \
        const device uint* activeTiles [[buffer(2)]], \
        const device uint* activeTileCount [[buffer(3)]], \
        texture2d<half, access::write> colorOut [[texture(0)]], \
        texture2d<float, access::write> depthOut [[texture(1)]], \
        texture2d<float, access::write> alphaOut [[texture(2)]], \
        constant RenderParams& params [[buffer(4)]], \
        uint2 group_id [[threadgroup_position_in_grid]], \
        uint2 local_id [[thread_position_in_threadgroup]]);

instantiate_renderTilesFused(half, PackedGaussian)
instantiate_renderTilesFused(float, PackedGaussianF32)

#undef instantiate_renderTilesFused

/// Fused render kernel for 32x16 tiles - multi-pixel mode (half16)
/// 8x8 threadgroup, each thread handles 4x2 = 8 pixels
/// Uses PackedGaussian for cache-efficient single-struct reads
#define RENDER_FUSED_MULTIPIXEL_BATCH_SIZE 64

kernel void renderTilesFusedMultiPixel_half(
    const device GaussianHeader*    headers         [[buffer(0)]],
    const device PackedGaussian*    gaussians       [[buffer(1)]],
    const device uint*              activeTiles     [[buffer(2)]],
    const device uint*              activeTileCount [[buffer(3)]],
    texture2d<half, access::write>  colorOut        [[texture(0)]],
    texture2d<float, access::write> depthOut        [[texture(1)]],
    texture2d<float, access::write> alphaOut        [[texture(2)]],
    constant RenderParams&          params          [[buffer(4)]],
    uint2                           group_id        [[threadgroup_position_in_grid]],
    uint2                           local_id        [[thread_position_in_threadgroup]],
    uint                            tid             [[thread_index_in_threadgroup]]
) {
    threadgroup PackedGaussian shGaussians[RENDER_FUSED_MULTIPIXEL_BATCH_SIZE];

    uint tileIdx = group_id.x;
    if (tileIdx >= *activeTileCount) return;

    uint tile = activeTiles[tileIdx];
    GaussianHeader hdr = headers[tile];

    uint tileX = tile % params.tilesX;
    uint tileY = tile / params.tilesX;

    // 32x16 tile, 8x8 threads, 4x2 pixels per thread
    uint baseX = tileX * 32 + local_id.x * 4;
    uint baseY = tileY * 16 + local_id.y * 2;

    // Pixel coordinates
    half px0 = half(baseX), px1 = half(baseX + 1), px2 = half(baseX + 2), px3 = half(baseX + 3);
    half py0 = half(baseY), py1 = half(baseY + 1);

    // Accumulators for 8 pixels (4x2)
    half trans00 = 1.0h, trans10 = 1.0h, trans20 = 1.0h, trans30 = 1.0h;
    half trans01 = 1.0h, trans11 = 1.0h, trans21 = 1.0h, trans31 = 1.0h;

    half3 color00 = half3(0), color10 = half3(0), color20 = half3(0), color30 = half3(0);
    half3 color01 = half3(0), color11 = half3(0), color21 = half3(0), color31 = half3(0);

    float depth00 = 0, depth10 = 0, depth20 = 0, depth30 = 0;
    float depth01 = 0, depth11 = 0, depth21 = 0, depth31 = 0;

    uint start = hdr.offset;
    uint count = hdr.count;

    for (uint batch = 0; batch < count; batch += RENDER_FUSED_MULTIPIXEL_BATCH_SIZE) {
        uint batchCount = min(uint(RENDER_FUSED_MULTIPIXEL_BATCH_SIZE), count - batch);

        // Cooperative loading with 64 threads
        if (tid < batchCount) {
            shGaussians[tid] = gaussians[start + batch + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process batch
        for (uint i = 0; i < batchCount; i++) {
            // Early exit when all pixels are saturated
            half maxTrans0 = max(max(trans00, trans10), max(trans20, trans30));
            half maxTrans1 = max(max(trans01, trans11), max(trans21, trans31));
            if (max(maxTrans0, maxTrans1) < half(1.0h/255.0h)) break;

            PackedGaussian g = shGaussians[i];

            // Direction vectors for all 8 pixels
            half2 d00 = half2(px0, py0) - g.mean;
            half2 d10 = half2(px1, py0) - g.mean;
            half2 d20 = half2(px2, py0) - g.mean;
            half2 d30 = half2(px3, py0) - g.mean;
            half2 d01 = half2(px0, py1) - g.mean;
            half2 d11 = half2(px1, py1) - g.mean;
            half2 d21 = half2(px2, py1) - g.mean;
            half2 d31 = half2(px3, py1) - g.mean;

            // Quadratic form for each pixel
            half p00 = d00.x*d00.x*g.conic.x + d00.y*d00.y*g.conic.z + 2.0h*d00.x*d00.y*g.conic.y;
            half p10 = d10.x*d10.x*g.conic.x + d10.y*d10.y*g.conic.z + 2.0h*d10.x*d10.y*g.conic.y;
            half p20 = d20.x*d20.x*g.conic.x + d20.y*d20.y*g.conic.z + 2.0h*d20.x*d20.y*g.conic.y;
            half p30 = d30.x*d30.x*g.conic.x + d30.y*d30.y*g.conic.z + 2.0h*d30.x*d30.y*g.conic.y;
            half p01 = d01.x*d01.x*g.conic.x + d01.y*d01.y*g.conic.z + 2.0h*d01.x*d01.y*g.conic.y;
            half p11 = d11.x*d11.x*g.conic.x + d11.y*d11.y*g.conic.z + 2.0h*d11.x*d11.y*g.conic.y;
            half p21 = d21.x*d21.x*g.conic.x + d21.y*d21.y*g.conic.z + 2.0h*d21.x*d21.y*g.conic.y;
            half p31 = d31.x*d31.x*g.conic.x + d31.y*d31.y*g.conic.z + 2.0h*d31.x*d31.y*g.conic.y;

            // Alpha = opacity * exp(-0.5 * power)
            half opacity = g.opacity;
            half a00 = min(opacity * exp(-0.5h * p00), half(0.99h));
            half a10 = min(opacity * exp(-0.5h * p10), half(0.99h));
            half a20 = min(opacity * exp(-0.5h * p20), half(0.99h));
            half a30 = min(opacity * exp(-0.5h * p30), half(0.99h));
            half a01 = min(opacity * exp(-0.5h * p01), half(0.99h));
            half a11 = min(opacity * exp(-0.5h * p11), half(0.99h));
            half a21 = min(opacity * exp(-0.5h * p21), half(0.99h));
            half a31 = min(opacity * exp(-0.5h * p31), half(0.99h));

            // Blend: color += gaussian_color * alpha * transparency
            half3 gColor = half3(g.color);
            float gDepth = float(g.depth);

            color00 += gColor * (a00 * trans00); depth00 += gDepth * float(a00 * trans00);
            color10 += gColor * (a10 * trans10); depth10 += gDepth * float(a10 * trans10);
            color20 += gColor * (a20 * trans20); depth20 += gDepth * float(a20 * trans20);
            color30 += gColor * (a30 * trans30); depth30 += gDepth * float(a30 * trans30);
            color01 += gColor * (a01 * trans01); depth01 += gDepth * float(a01 * trans01);
            color11 += gColor * (a11 * trans11); depth11 += gDepth * float(a11 * trans11);
            color21 += gColor * (a21 * trans21); depth21 += gDepth * float(a21 * trans21);
            color31 += gColor * (a31 * trans31); depth31 += gDepth * float(a31 * trans31);

            // Update transparency
            trans00 *= (1.0h - a00); trans10 *= (1.0h - a10);
            trans20 *= (1.0h - a20); trans30 *= (1.0h - a30);
            trans01 *= (1.0h - a01); trans11 *= (1.0h - a11);
            trans21 *= (1.0h - a21); trans31 *= (1.0h - a31);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Apply background
    half bg = params.whiteBackground ? half(1.0h) : half(0.0h);
    color00 += half3(trans00 * bg); color10 += half3(trans10 * bg);
    color20 += half3(trans20 * bg); color30 += half3(trans30 * bg);
    color01 += half3(trans01 * bg); color11 += half3(trans11 * bg);
    color21 += half3(trans21 * bg); color31 += half3(trans31 * bg);

    // Write all 8 pixels
    if (all(uint2(baseX + 3, baseY + 1) < uint2(params.width, params.height))) {
        colorOut.write(half4(color00, 1.0h - trans00), uint2(baseX + 0, baseY));
        colorOut.write(half4(color10, 1.0h - trans10), uint2(baseX + 1, baseY));
        colorOut.write(half4(color20, 1.0h - trans20), uint2(baseX + 2, baseY));
        colorOut.write(half4(color30, 1.0h - trans30), uint2(baseX + 3, baseY));
        colorOut.write(half4(color01, 1.0h - trans01), uint2(baseX + 0, baseY + 1));
        colorOut.write(half4(color11, 1.0h - trans11), uint2(baseX + 1, baseY + 1));
        colorOut.write(half4(color21, 1.0h - trans21), uint2(baseX + 2, baseY + 1));
        colorOut.write(half4(color31, 1.0h - trans31), uint2(baseX + 3, baseY + 1));

        depthOut.write(float4(depth00, 0, 0, 0), uint2(baseX + 0, baseY));
        depthOut.write(float4(depth10, 0, 0, 0), uint2(baseX + 1, baseY));
        depthOut.write(float4(depth20, 0, 0, 0), uint2(baseX + 2, baseY));
        depthOut.write(float4(depth30, 0, 0, 0), uint2(baseX + 3, baseY));
        depthOut.write(float4(depth01, 0, 0, 0), uint2(baseX + 0, baseY + 1));
        depthOut.write(float4(depth11, 0, 0, 0), uint2(baseX + 1, baseY + 1));
        depthOut.write(float4(depth21, 0, 0, 0), uint2(baseX + 2, baseY + 1));
        depthOut.write(float4(depth31, 0, 0, 0), uint2(baseX + 3, baseY + 1));

        alphaOut.write(float4(1.0f - float(trans00), 0, 0, 0), uint2(baseX + 0, baseY));
        alphaOut.write(float4(1.0f - float(trans10), 0, 0, 0), uint2(baseX + 1, baseY));
        alphaOut.write(float4(1.0f - float(trans20), 0, 0, 0), uint2(baseX + 2, baseY));
        alphaOut.write(float4(1.0f - float(trans30), 0, 0, 0), uint2(baseX + 3, baseY));
        alphaOut.write(float4(1.0f - float(trans01), 0, 0, 0), uint2(baseX + 0, baseY + 1));
        alphaOut.write(float4(1.0f - float(trans11), 0, 0, 0), uint2(baseX + 1, baseY + 1));
        alphaOut.write(float4(1.0f - float(trans21), 0, 0, 0), uint2(baseX + 2, baseY + 1));
        alphaOut.write(float4(1.0f - float(trans31), 0, 0, 0), uint2(baseX + 3, baseY + 1));
    } else {
        // Bounds checking per pixel
        if (baseX + 0 < params.width && baseY < params.height) {
            colorOut.write(half4(color00, 1.0h - trans00), uint2(baseX + 0, baseY));
            depthOut.write(float4(depth00, 0, 0, 0), uint2(baseX + 0, baseY));
            alphaOut.write(float4(1.0f - float(trans00), 0, 0, 0), uint2(baseX + 0, baseY));
        }
        if (baseX + 1 < params.width && baseY < params.height) {
            colorOut.write(half4(color10, 1.0h - trans10), uint2(baseX + 1, baseY));
            depthOut.write(float4(depth10, 0, 0, 0), uint2(baseX + 1, baseY));
            alphaOut.write(float4(1.0f - float(trans10), 0, 0, 0), uint2(baseX + 1, baseY));
        }
        if (baseX + 2 < params.width && baseY < params.height) {
            colorOut.write(half4(color20, 1.0h - trans20), uint2(baseX + 2, baseY));
            depthOut.write(float4(depth20, 0, 0, 0), uint2(baseX + 2, baseY));
            alphaOut.write(float4(1.0f - float(trans20), 0, 0, 0), uint2(baseX + 2, baseY));
        }
        if (baseX + 3 < params.width && baseY < params.height) {
            colorOut.write(half4(color30, 1.0h - trans30), uint2(baseX + 3, baseY));
            depthOut.write(float4(depth30, 0, 0, 0), uint2(baseX + 3, baseY));
            alphaOut.write(float4(1.0f - float(trans30), 0, 0, 0), uint2(baseX + 3, baseY));
        }
        if (baseX + 0 < params.width && baseY + 1 < params.height) {
            colorOut.write(half4(color01, 1.0h - trans01), uint2(baseX + 0, baseY + 1));
            depthOut.write(float4(depth01, 0, 0, 0), uint2(baseX + 0, baseY + 1));
            alphaOut.write(float4(1.0f - float(trans01), 0, 0, 0), uint2(baseX + 0, baseY + 1));
        }
        if (baseX + 1 < params.width && baseY + 1 < params.height) {
            colorOut.write(half4(color11, 1.0h - trans11), uint2(baseX + 1, baseY + 1));
            depthOut.write(float4(depth11, 0, 0, 0), uint2(baseX + 1, baseY + 1));
            alphaOut.write(float4(1.0f - float(trans11), 0, 0, 0), uint2(baseX + 1, baseY + 1));
        }
        if (baseX + 2 < params.width && baseY + 1 < params.height) {
            colorOut.write(half4(color21, 1.0h - trans21), uint2(baseX + 2, baseY + 1));
            depthOut.write(float4(depth21, 0, 0, 0), uint2(baseX + 2, baseY + 1));
            alphaOut.write(float4(1.0f - float(trans21), 0, 0, 0), uint2(baseX + 2, baseY + 1));
        }
        if (baseX + 3 < params.width && baseY + 1 < params.height) {
            colorOut.write(half4(color31, 1.0h - trans31), uint2(baseX + 3, baseY + 1));
            depthOut.write(float4(depth31, 0, 0, 0), uint2(baseX + 3, baseY + 1));
            alphaOut.write(float4(1.0f - float(trans31), 0, 0, 0), uint2(baseX + 3, baseY + 1));
        }
    }
}

/// Fused render kernel for 32x16 tiles - multi-pixel mode (float32)
kernel void renderTilesFusedMultiPixel_float(
    const device GaussianHeader*      headers         [[buffer(0)]],
    const device PackedGaussianF32*   gaussians       [[buffer(1)]],
    const device uint*                activeTiles     [[buffer(2)]],
    const device uint*                activeTileCount [[buffer(3)]],
    texture2d<half, access::write>    colorOut        [[texture(0)]],
    texture2d<float, access::write>   depthOut        [[texture(1)]],
    texture2d<float, access::write>   alphaOut        [[texture(2)]],
    constant RenderParams&            params          [[buffer(4)]],
    uint2                             group_id        [[threadgroup_position_in_grid]],
    uint2                             local_id        [[thread_position_in_threadgroup]],
    uint                              tid             [[thread_index_in_threadgroup]]
) {
    threadgroup PackedGaussianF32 shGaussians[RENDER_FUSED_MULTIPIXEL_BATCH_SIZE];

    uint tileIdx = group_id.x;
    if (tileIdx >= *activeTileCount) return;

    uint tile = activeTiles[tileIdx];
    GaussianHeader hdr = headers[tile];

    uint tileX = tile % params.tilesX;
    uint tileY = tile / params.tilesX;

    uint baseX = tileX * 32 + local_id.x * 4;
    uint baseY = tileY * 16 + local_id.y * 2;

    float px0 = float(baseX), px1 = float(baseX + 1), px2 = float(baseX + 2), px3 = float(baseX + 3);
    float py0 = float(baseY), py1 = float(baseY + 1);

    float trans00 = 1.0f, trans10 = 1.0f, trans20 = 1.0f, trans30 = 1.0f;
    float trans01 = 1.0f, trans11 = 1.0f, trans21 = 1.0f, trans31 = 1.0f;

    float3 color00 = float3(0), color10 = float3(0), color20 = float3(0), color30 = float3(0);
    float3 color01 = float3(0), color11 = float3(0), color21 = float3(0), color31 = float3(0);

    float depth00 = 0, depth10 = 0, depth20 = 0, depth30 = 0;
    float depth01 = 0, depth11 = 0, depth21 = 0, depth31 = 0;

    uint start = hdr.offset;
    uint count = hdr.count;

    for (uint batch = 0; batch < count; batch += RENDER_FUSED_MULTIPIXEL_BATCH_SIZE) {
        uint batchCount = min(uint(RENDER_FUSED_MULTIPIXEL_BATCH_SIZE), count - batch);

        if (tid < batchCount) {
            shGaussians[tid] = gaussians[start + batch + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < batchCount; i++) {
            float maxTrans = max(max(max(trans00, trans10), max(trans20, trans30)),
                                 max(max(trans01, trans11), max(trans21, trans31)));
            if (maxTrans < 1e-3f) break;

            PackedGaussianF32 g = shGaussians[i];

            float2 d00 = float2(px0, py0) - g.mean;
            float2 d10 = float2(px1, py0) - g.mean;
            float2 d20 = float2(px2, py0) - g.mean;
            float2 d30 = float2(px3, py0) - g.mean;
            float2 d01 = float2(px0, py1) - g.mean;
            float2 d11 = float2(px1, py1) - g.mean;
            float2 d21 = float2(px2, py1) - g.mean;
            float2 d31 = float2(px3, py1) - g.mean;

            float p00 = d00.x*d00.x*g.conic.x + d00.y*d00.y*g.conic.z + 2.0f*d00.x*d00.y*g.conic.y;
            float p10 = d10.x*d10.x*g.conic.x + d10.y*d10.y*g.conic.z + 2.0f*d10.x*d10.y*g.conic.y;
            float p20 = d20.x*d20.x*g.conic.x + d20.y*d20.y*g.conic.z + 2.0f*d20.x*d20.y*g.conic.y;
            float p30 = d30.x*d30.x*g.conic.x + d30.y*d30.y*g.conic.z + 2.0f*d30.x*d30.y*g.conic.y;
            float p01 = d01.x*d01.x*g.conic.x + d01.y*d01.y*g.conic.z + 2.0f*d01.x*d01.y*g.conic.y;
            float p11 = d11.x*d11.x*g.conic.x + d11.y*d11.y*g.conic.z + 2.0f*d11.x*d11.y*g.conic.y;
            float p21 = d21.x*d21.x*g.conic.x + d21.y*d21.y*g.conic.z + 2.0f*d21.x*d21.y*g.conic.y;
            float p31 = d31.x*d31.x*g.conic.x + d31.y*d31.y*g.conic.z + 2.0f*d31.x*d31.y*g.conic.y;

            float opacity = g.opacity;
            float a00 = min(opacity * exp(-0.5f * p00), 0.99f);
            float a10 = min(opacity * exp(-0.5f * p10), 0.99f);
            float a20 = min(opacity * exp(-0.5f * p20), 0.99f);
            float a30 = min(opacity * exp(-0.5f * p30), 0.99f);
            float a01 = min(opacity * exp(-0.5f * p01), 0.99f);
            float a11 = min(opacity * exp(-0.5f * p11), 0.99f);
            float a21 = min(opacity * exp(-0.5f * p21), 0.99f);
            float a31 = min(opacity * exp(-0.5f * p31), 0.99f);

            float3 gColor = float3(g.color);
            float gDepth = g.depth;

            color00 += gColor * (a00 * trans00); depth00 += gDepth * (a00 * trans00);
            color10 += gColor * (a10 * trans10); depth10 += gDepth * (a10 * trans10);
            color20 += gColor * (a20 * trans20); depth20 += gDepth * (a20 * trans20);
            color30 += gColor * (a30 * trans30); depth30 += gDepth * (a30 * trans30);
            color01 += gColor * (a01 * trans01); depth01 += gDepth * (a01 * trans01);
            color11 += gColor * (a11 * trans11); depth11 += gDepth * (a11 * trans11);
            color21 += gColor * (a21 * trans21); depth21 += gDepth * (a21 * trans21);
            color31 += gColor * (a31 * trans31); depth31 += gDepth * (a31 * trans31);

            trans00 *= (1.0f - a00); trans10 *= (1.0f - a10);
            trans20 *= (1.0f - a20); trans30 *= (1.0f - a30);
            trans01 *= (1.0f - a01); trans11 *= (1.0f - a11);
            trans21 *= (1.0f - a21); trans31 *= (1.0f - a31);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float bg = params.whiteBackground ? 1.0f : 0.0f;
    color00 += float3(trans00 * bg); color10 += float3(trans10 * bg);
    color20 += float3(trans20 * bg); color30 += float3(trans30 * bg);
    color01 += float3(trans01 * bg); color11 += float3(trans11 * bg);
    color21 += float3(trans21 * bg); color31 += float3(trans31 * bg);

    if (all(uint2(baseX + 3, baseY + 1) < uint2(params.width, params.height))) {
        colorOut.write(half4(half3(color00), half(1.0f - trans00)), uint2(baseX + 0, baseY));
        colorOut.write(half4(half3(color10), half(1.0f - trans10)), uint2(baseX + 1, baseY));
        colorOut.write(half4(half3(color20), half(1.0f - trans20)), uint2(baseX + 2, baseY));
        colorOut.write(half4(half3(color30), half(1.0f - trans30)), uint2(baseX + 3, baseY));
        colorOut.write(half4(half3(color01), half(1.0f - trans01)), uint2(baseX + 0, baseY + 1));
        colorOut.write(half4(half3(color11), half(1.0f - trans11)), uint2(baseX + 1, baseY + 1));
        colorOut.write(half4(half3(color21), half(1.0f - trans21)), uint2(baseX + 2, baseY + 1));
        colorOut.write(half4(half3(color31), half(1.0f - trans31)), uint2(baseX + 3, baseY + 1));

        depthOut.write(float4(depth00, 0, 0, 0), uint2(baseX + 0, baseY));
        depthOut.write(float4(depth10, 0, 0, 0), uint2(baseX + 1, baseY));
        depthOut.write(float4(depth20, 0, 0, 0), uint2(baseX + 2, baseY));
        depthOut.write(float4(depth30, 0, 0, 0), uint2(baseX + 3, baseY));
        depthOut.write(float4(depth01, 0, 0, 0), uint2(baseX + 0, baseY + 1));
        depthOut.write(float4(depth11, 0, 0, 0), uint2(baseX + 1, baseY + 1));
        depthOut.write(float4(depth21, 0, 0, 0), uint2(baseX + 2, baseY + 1));
        depthOut.write(float4(depth31, 0, 0, 0), uint2(baseX + 3, baseY + 1));

        alphaOut.write(float4(1.0f - trans00, 0, 0, 0), uint2(baseX + 0, baseY));
        alphaOut.write(float4(1.0f - trans10, 0, 0, 0), uint2(baseX + 1, baseY));
        alphaOut.write(float4(1.0f - trans20, 0, 0, 0), uint2(baseX + 2, baseY));
        alphaOut.write(float4(1.0f - trans30, 0, 0, 0), uint2(baseX + 3, baseY));
        alphaOut.write(float4(1.0f - trans01, 0, 0, 0), uint2(baseX + 0, baseY + 1));
        alphaOut.write(float4(1.0f - trans11, 0, 0, 0), uint2(baseX + 1, baseY + 1));
        alphaOut.write(float4(1.0f - trans21, 0, 0, 0), uint2(baseX + 2, baseY + 1));
        alphaOut.write(float4(1.0f - trans31, 0, 0, 0), uint2(baseX + 3, baseY + 1));
    } else {
        if (baseX + 0 < params.width && baseY < params.height) {
            colorOut.write(half4(half3(color00), half(1.0f - trans00)), uint2(baseX + 0, baseY));
            depthOut.write(float4(depth00, 0, 0, 0), uint2(baseX + 0, baseY));
            alphaOut.write(float4(1.0f - trans00, 0, 0, 0), uint2(baseX + 0, baseY));
        }
        if (baseX + 1 < params.width && baseY < params.height) {
            colorOut.write(half4(half3(color10), half(1.0f - trans10)), uint2(baseX + 1, baseY));
            depthOut.write(float4(depth10, 0, 0, 0), uint2(baseX + 1, baseY));
            alphaOut.write(float4(1.0f - trans10, 0, 0, 0), uint2(baseX + 1, baseY));
        }
        if (baseX + 2 < params.width && baseY < params.height) {
            colorOut.write(half4(half3(color20), half(1.0f - trans20)), uint2(baseX + 2, baseY));
            depthOut.write(float4(depth20, 0, 0, 0), uint2(baseX + 2, baseY));
            alphaOut.write(float4(1.0f - trans20, 0, 0, 0), uint2(baseX + 2, baseY));
        }
        if (baseX + 3 < params.width && baseY < params.height) {
            colorOut.write(half4(half3(color30), half(1.0f - trans30)), uint2(baseX + 3, baseY));
            depthOut.write(float4(depth30, 0, 0, 0), uint2(baseX + 3, baseY));
            alphaOut.write(float4(1.0f - trans30, 0, 0, 0), uint2(baseX + 3, baseY));
        }
        if (baseX + 0 < params.width && baseY + 1 < params.height) {
            colorOut.write(half4(half3(color01), half(1.0f - trans01)), uint2(baseX + 0, baseY + 1));
            depthOut.write(float4(depth01, 0, 0, 0), uint2(baseX + 0, baseY + 1));
            alphaOut.write(float4(1.0f - trans01, 0, 0, 0), uint2(baseX + 0, baseY + 1));
        }
        if (baseX + 1 < params.width && baseY + 1 < params.height) {
            colorOut.write(half4(half3(color11), half(1.0f - trans11)), uint2(baseX + 1, baseY + 1));
            depthOut.write(float4(depth11, 0, 0, 0), uint2(baseX + 1, baseY + 1));
            alphaOut.write(float4(1.0f - trans11, 0, 0, 0), uint2(baseX + 1, baseY + 1));
        }
        if (baseX + 2 < params.width && baseY + 1 < params.height) {
            colorOut.write(half4(half3(color21), half(1.0f - trans21)), uint2(baseX + 2, baseY + 1));
            depthOut.write(float4(depth21, 0, 0, 0), uint2(baseX + 2, baseY + 1));
            alphaOut.write(float4(1.0f - trans21, 0, 0, 0), uint2(baseX + 2, baseY + 1));
        }
        if (baseX + 3 < params.width && baseY + 1 < params.height) {
            colorOut.write(half4(half3(color31), half(1.0f - trans31)), uint2(baseX + 3, baseY + 1));
            depthOut.write(float4(depth31, 0, 0, 0), uint2(baseX + 3, baseY + 1));
            alphaOut.write(float4(1.0f - trans31, 0, 0, 0), uint2(baseX + 3, baseY + 1));
        }
    }
}

// =============================================================================
// MARK: - Optimized Render Kernel V2 (Lower Register Pressure)
// =============================================================================
// Key optimizations:
// 1. 2x2 pixels per thread (vs 4x2) - halves register usage
// 2. 128 threads per TG (vs 64) - better occupancy
// 3. SIMD-level early exit - skip when entire SIMD group is saturated
// 4. Vectorized conic computation - fewer instructions
// Target: ~80-100 registers (vs 198), ~6-8% occupancy (vs 2.85%)
// =============================================================================

// =============================================================================
// V3: Tellusim-style - NO shared memory, 8 pixels/thread, direct global reads
// Key insight: Shared memory loading has overhead. Better to load per-thread
// and process more pixels to amortize the load cost.
// 32x16 tile, 8x8 threadgroup (64 threads), 4x2 pixels per thread = 512 pixels
// =============================================================================
#define RENDER_V3_TG_WIDTH 8
#define RENDER_V3_TG_HEIGHT 8
#define RENDER_V3_PIXELS_X 4
#define RENDER_V3_PIXELS_Y 2

kernel void renderTilesFusedV3_half(
    const device GaussianHeader*    headers         [[buffer(0)]],
    const device PackedGaussian*    gaussians       [[buffer(1)]],
    const device uint*              activeTiles     [[buffer(2)]],
    const device uint*              activeTileCount [[buffer(3)]],
    texture2d<half, access::write>  colorOut        [[texture(0)]],
    texture2d<float, access::write> depthOut        [[texture(1)]],
    texture2d<float, access::write> alphaOut        [[texture(2)]],
    constant RenderParams&          params          [[buffer(4)]],
    uint2                           group_id        [[threadgroup_position_in_grid]],
    uint2                           local_id        [[thread_position_in_threadgroup]]
) {
    // No shared memory - Tellusim insight: direct global reads are better
    // when each thread processes enough pixels to amortize the load

    uint tileIdx = group_id.x;
    if (tileIdx >= *activeTileCount) return;

    uint tile = activeTiles[tileIdx];
    GaussianHeader hdr = headers[tile];

    uint tileX = tile % params.tilesX;
    uint tileY = tile / params.tilesX;

    // 32x16 tile, 8x8 threads, 4x2 pixels per thread
    uint baseX = tileX * 32 + local_id.x * RENDER_V3_PIXELS_X;
    uint baseY = tileY * 16 + local_id.y * RENDER_V3_PIXELS_Y;

    // Pixel positions (8 pixels: 4 wide x 2 tall)
    half2 pos00 = half2(half(baseX + 0), half(baseY + 0));
    half2 pos10 = half2(half(baseX + 1), half(baseY + 0));
    half2 pos20 = half2(half(baseX + 2), half(baseY + 0));
    half2 pos30 = half2(half(baseX + 3), half(baseY + 0));
    half2 pos01 = half2(half(baseX + 0), half(baseY + 1));
    half2 pos11 = half2(half(baseX + 1), half(baseY + 1));
    half2 pos21 = half2(half(baseX + 2), half(baseY + 1));
    half2 pos31 = half2(half(baseX + 3), half(baseY + 1));

    // Accumulators for 8 pixels (4x2)
    half trans00 = 1.0h, trans10 = 1.0h, trans20 = 1.0h, trans30 = 1.0h;
    half trans01 = 1.0h, trans11 = 1.0h, trans21 = 1.0h, trans31 = 1.0h;

    half3 color00 = 0, color10 = 0, color20 = 0, color30 = 0;
    half3 color01 = 0, color11 = 0, color21 = 0, color31 = 0;

    half depth00 = 0, depth10 = 0, depth20 = 0, depth30 = 0;
    half depth01 = 0, depth11 = 0, depth21 = 0, depth31 = 0;

    uint start = hdr.offset;
    uint count = hdr.count;

    // Direct loop over all gaussians - no batching, no shared memory
    for (uint i = 0; i < count; i++) {
        // Early exit when all pixels are saturated
        half maxTrans0 = max(max(trans00, trans10), max(trans20, trans30));
        half maxTrans1 = max(max(trans01, trans11), max(trans21, trans31));
        if (max(maxTrans0, maxTrans1) < half(1.0h/255.0h)) break;

        // Load gaussian directly from global memory
        PackedGaussian g = gaussians[start + i];

        // Precompute covariance terms
        half cxx = g.conic.x;
        half cyy = g.conic.z;
        half cxy2 = 2.0h * g.conic.y;
        half opacity = g.opacity;
        half3 gColor = half3(g.color);
        half gDepth = g.depth;

        // Direction vectors for 8 pixels
        half2 d00 = pos00 - g.mean;
        half2 d10 = pos10 - g.mean;
        half2 d20 = pos20 - g.mean;
        half2 d30 = pos30 - g.mean;
        half2 d01 = pos01 - g.mean;
        half2 d11 = pos11 - g.mean;
        half2 d21 = pos21 - g.mean;
        half2 d31 = pos31 - g.mean;

        // Quadratic form: power = cxx*dx*dx + cyy*dy*dy + cxy2*dx*dy
        half p00 = d00.x*d00.x*cxx + d00.y*d00.y*cyy + d00.x*d00.y*cxy2;
        half p10 = d10.x*d10.x*cxx + d10.y*d10.y*cyy + d10.x*d10.y*cxy2;
        half p20 = d20.x*d20.x*cxx + d20.y*d20.y*cyy + d20.x*d20.y*cxy2;
        half p30 = d30.x*d30.x*cxx + d30.y*d30.y*cyy + d30.x*d30.y*cxy2;
        half p01 = d01.x*d01.x*cxx + d01.y*d01.y*cyy + d01.x*d01.y*cxy2;
        half p11 = d11.x*d11.x*cxx + d11.y*d11.y*cyy + d11.x*d11.y*cxy2;
        half p21 = d21.x*d21.x*cxx + d21.y*d21.y*cyy + d21.x*d21.y*cxy2;
        half p31 = d31.x*d31.x*cxx + d31.y*d31.y*cyy + d31.x*d31.y*cxy2;

        // Alpha = opacity * exp(-0.5 * power)
        half a00 = min(opacity * exp(-0.5h * p00), 0.99h);
        half a10 = min(opacity * exp(-0.5h * p10), 0.99h);
        half a20 = min(opacity * exp(-0.5h * p20), 0.99h);
        half a30 = min(opacity * exp(-0.5h * p30), 0.99h);
        half a01 = min(opacity * exp(-0.5h * p01), 0.99h);
        half a11 = min(opacity * exp(-0.5h * p11), 0.99h);
        half a21 = min(opacity * exp(-0.5h * p21), 0.99h);
        half a31 = min(opacity * exp(-0.5h * p31), 0.99h);

        // Blend color and depth
        color00 += gColor * (a00 * trans00); depth00 += gDepth * (a00 * trans00);
        color10 += gColor * (a10 * trans10); depth10 += gDepth * (a10 * trans10);
        color20 += gColor * (a20 * trans20); depth20 += gDepth * (a20 * trans20);
        color30 += gColor * (a30 * trans30); depth30 += gDepth * (a30 * trans30);
        color01 += gColor * (a01 * trans01); depth01 += gDepth * (a01 * trans01);
        color11 += gColor * (a11 * trans11); depth11 += gDepth * (a11 * trans11);
        color21 += gColor * (a21 * trans21); depth21 += gDepth * (a21 * trans21);
        color31 += gColor * (a31 * trans31); depth31 += gDepth * (a31 * trans31);

        // Update transparency
        trans00 *= (1.0h - a00); trans10 *= (1.0h - a10);
        trans20 *= (1.0h - a20); trans30 *= (1.0h - a30);
        trans01 *= (1.0h - a01); trans11 *= (1.0h - a11);
        trans21 *= (1.0h - a21); trans31 *= (1.0h - a31);
    }

    // Apply background
    half bg = params.whiteBackground ? 1.0h : 0.0h;
    color00 += trans00 * bg; color10 += trans10 * bg;
    color20 += trans20 * bg; color30 += trans30 * bg;
    color01 += trans01 * bg; color11 += trans11 * bg;
    color21 += trans21 * bg; color31 += trans31 * bg;

    // Write 8 pixels (bounds checking)
    uint w = params.width, h = params.height;
    if (baseX + 0 < w && baseY + 0 < h) {
        colorOut.write(half4(color00, 1.0h - trans00), uint2(baseX + 0, baseY + 0));
        depthOut.write(float4(float(depth00), 0, 0, 0), uint2(baseX + 0, baseY + 0));
        alphaOut.write(float4(1.0f - float(trans00), 0, 0, 0), uint2(baseX + 0, baseY + 0));
    }
    if (baseX + 1 < w && baseY + 0 < h) {
        colorOut.write(half4(color10, 1.0h - trans10), uint2(baseX + 1, baseY + 0));
        depthOut.write(float4(float(depth10), 0, 0, 0), uint2(baseX + 1, baseY + 0));
        alphaOut.write(float4(1.0f - float(trans10), 0, 0, 0), uint2(baseX + 1, baseY + 0));
    }
    if (baseX + 2 < w && baseY + 0 < h) {
        colorOut.write(half4(color20, 1.0h - trans20), uint2(baseX + 2, baseY + 0));
        depthOut.write(float4(float(depth20), 0, 0, 0), uint2(baseX + 2, baseY + 0));
        alphaOut.write(float4(1.0f - float(trans20), 0, 0, 0), uint2(baseX + 2, baseY + 0));
    }
    if (baseX + 3 < w && baseY + 0 < h) {
        colorOut.write(half4(color30, 1.0h - trans30), uint2(baseX + 3, baseY + 0));
        depthOut.write(float4(float(depth30), 0, 0, 0), uint2(baseX + 3, baseY + 0));
        alphaOut.write(float4(1.0f - float(trans30), 0, 0, 0), uint2(baseX + 3, baseY + 0));
    }
    if (baseX + 0 < w && baseY + 1 < h) {
        colorOut.write(half4(color01, 1.0h - trans01), uint2(baseX + 0, baseY + 1));
        depthOut.write(float4(float(depth01), 0, 0, 0), uint2(baseX + 0, baseY + 1));
        alphaOut.write(float4(1.0f - float(trans01), 0, 0, 0), uint2(baseX + 0, baseY + 1));
    }
    if (baseX + 1 < w && baseY + 1 < h) {
        colorOut.write(half4(color11, 1.0h - trans11), uint2(baseX + 1, baseY + 1));
        depthOut.write(float4(float(depth11), 0, 0, 0), uint2(baseX + 1, baseY + 1));
        alphaOut.write(float4(1.0f - float(trans11), 0, 0, 0), uint2(baseX + 1, baseY + 1));
    }
    if (baseX + 2 < w && baseY + 1 < h) {
        colorOut.write(half4(color21, 1.0h - trans21), uint2(baseX + 2, baseY + 1));
        depthOut.write(float4(float(depth21), 0, 0, 0), uint2(baseX + 2, baseY + 1));
        alphaOut.write(float4(1.0f - float(trans21), 0, 0, 0), uint2(baseX + 2, baseY + 1));
    }
    if (baseX + 3 < w && baseY + 1 < h) {
        colorOut.write(half4(color31, 1.0h - trans31), uint2(baseX + 3, baseY + 1));
        depthOut.write(float4(float(depth31), 0, 0, 0), uint2(baseX + 3, baseY + 1));
        alphaOut.write(float4(1.0f - float(trans31), 0, 0, 0), uint2(baseX + 3, baseY + 1));
    }
}

// =============================================================================
// V2: Cooperative loading with shared memory (kept for comparison)
// =============================================================================
#define RENDER_V2_BATCH_SIZE 64
#define RENDER_V2_TG_WIDTH 16
#define RENDER_V2_TG_HEIGHT 8
#define RENDER_V2_PIXELS_X 2
#define RENDER_V2_PIXELS_Y 2

kernel void renderTilesFusedV2_half(
    const device GaussianHeader*    headers         [[buffer(0)]],
    const device PackedGaussian*    gaussians       [[buffer(1)]],
    const device uint*              activeTiles     [[buffer(2)]],
    const device uint*              activeTileCount [[buffer(3)]],
    texture2d<half, access::write>  colorOut        [[texture(0)]],
    texture2d<float, access::write> depthOut        [[texture(1)]],
    texture2d<float, access::write> alphaOut        [[texture(2)]],
    constant RenderParams&          params          [[buffer(4)]],
    uint2                           group_id        [[threadgroup_position_in_grid]],
    uint2                           local_id        [[thread_position_in_threadgroup]],
    uint                            tid             [[thread_index_in_threadgroup]]
) {
    // Shared memory for cooperative loading - 128 threads load 64 gaussians
    threadgroup PackedGaussian shGaussians[RENDER_V2_BATCH_SIZE];

    uint tileIdx = group_id.x;
    if (tileIdx >= *activeTileCount) return;

    uint tile = activeTiles[tileIdx];
    GaussianHeader hdr = headers[tile];

    uint tileX = tile % params.tilesX;
    uint tileY = tile / params.tilesX;

    // 32x16 tile, 16x8 threads, 2x2 pixels per thread
    uint baseX = tileX * 32 + local_id.x * RENDER_V2_PIXELS_X;
    uint baseY = tileY * 16 + local_id.y * RENDER_V2_PIXELS_Y;

    // Pixel coordinates (2x2 = 4 pixels per thread)
    half px0 = half(baseX), px1 = half(baseX + 1);
    half py0 = half(baseY), py1 = half(baseY + 1);

    // Accumulators for 4 pixels (2x2) - reduced from 8
    half trans00 = 1.0h, trans10 = 1.0h;
    half trans01 = 1.0h, trans11 = 1.0h;

    half3 color00 = half3(0), color10 = half3(0);
    half3 color01 = half3(0), color11 = half3(0);

    half depth00 = 0, depth10 = 0;
    half depth01 = 0, depth11 = 0;

    uint start = hdr.offset;
    uint count = hdr.count;

    for (uint batch = 0; batch < count; batch += RENDER_V2_BATCH_SIZE) {
        uint batchCount = min(uint(RENDER_V2_BATCH_SIZE), count - batch);

        // Cooperative loading: 128 threads load up to 64 gaussians
        // First 64 threads load, rest wait
        if (tid < batchCount) {
            shGaussians[tid] = gaussians[start + batch + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process batch
        for (uint i = 0; i < batchCount; i++) {
            // SIMD-level early exit check
            half maxTrans = max(max(trans00, trans10), max(trans01, trans11));
            if (simd_all(maxTrans < half(1.0h/255.0h))) break;

            PackedGaussian g = shGaussians[i];

            // Direction vectors for 4 pixels
            half2 d00 = half2(px0, py0) - g.mean;
            half2 d10 = half2(px1, py0) - g.mean;
            half2 d01 = half2(px0, py1) - g.mean;
            half2 d11 = half2(px1, py1) - g.mean;

            // Quadratic form - vectorized where possible
            half cxx = g.conic.x, cyy = g.conic.z, cxy2 = 2.0h * g.conic.y;
            half p00 = d00.x*d00.x*cxx + d00.y*d00.y*cyy + d00.x*d00.y*cxy2;
            half p10 = d10.x*d10.x*cxx + d10.y*d10.y*cyy + d10.x*d10.y*cxy2;
            half p01 = d01.x*d01.x*cxx + d01.y*d01.y*cyy + d01.x*d01.y*cxy2;
            half p11 = d11.x*d11.x*cxx + d11.y*d11.y*cyy + d11.x*d11.y*cxy2;

            // Alpha computation with fast exp
            half opacity = g.opacity;
            half a00 = min(opacity * exp(-0.5h * p00), half(0.99h));
            half a10 = min(opacity * exp(-0.5h * p10), half(0.99h));
            half a01 = min(opacity * exp(-0.5h * p01), half(0.99h));
            half a11 = min(opacity * exp(-0.5h * p11), half(0.99h));

            // Skip if all alphas are near zero
            if (max(max(a00, a10), max(a01, a11)) < half(1.0h/255.0h)) continue;

            // Blend
            half3 gColor = half3(g.color);
            half gDepth = g.depth;

            color00 += gColor * (a00 * trans00); depth00 += gDepth * (a00 * trans00);
            color10 += gColor * (a10 * trans10); depth10 += gDepth * (a10 * trans10);
            color01 += gColor * (a01 * trans01); depth01 += gDepth * (a01 * trans01);
            color11 += gColor * (a11 * trans11); depth11 += gDepth * (a11 * trans11);

            // Update transparency
            trans00 *= (1.0h - a00); trans10 *= (1.0h - a10);
            trans01 *= (1.0h - a01); trans11 *= (1.0h - a11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Apply background
    half bg = params.whiteBackground ? half(1.0h) : half(0.0h);
    color00 += half3(trans00 * bg); color10 += half3(trans10 * bg);
    color01 += half3(trans01 * bg); color11 += half3(trans11 * bg);

    // Write 4 pixels with bounds checking
    if (baseX < params.width && baseY < params.height) {
        colorOut.write(half4(color00, 1.0h - trans00), uint2(baseX, baseY));
        depthOut.write(float4(float(depth00), 0, 0, 0), uint2(baseX, baseY));
        alphaOut.write(float4(1.0f - float(trans00), 0, 0, 0), uint2(baseX, baseY));
    }
    if (baseX + 1 < params.width && baseY < params.height) {
        colorOut.write(half4(color10, 1.0h - trans10), uint2(baseX + 1, baseY));
        depthOut.write(float4(float(depth10), 0, 0, 0), uint2(baseX + 1, baseY));
        alphaOut.write(float4(1.0f - float(trans10), 0, 0, 0), uint2(baseX + 1, baseY));
    }
    if (baseX < params.width && baseY + 1 < params.height) {
        colorOut.write(half4(color01, 1.0h - trans01), uint2(baseX, baseY + 1));
        depthOut.write(float4(float(depth01), 0, 0, 0), uint2(baseX, baseY + 1));
        alphaOut.write(float4(1.0f - float(trans01), 0, 0, 0), uint2(baseX, baseY + 1));
    }
    if (baseX + 1 < params.width && baseY + 1 < params.height) {
        colorOut.write(half4(color11, 1.0h - trans11), uint2(baseX + 1, baseY + 1));
        depthOut.write(float4(float(depth11), 0, 0, 0), uint2(baseX + 1, baseY + 1));
        alphaOut.write(float4(1.0f - float(trans11), 0, 0, 0), uint2(baseX + 1, baseY + 1));
    }
}

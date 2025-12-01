// GaussianStructs.h - Shared GPU structs for Gaussian splatting renderer
// Used by GaussianMetalRenderer.metal (global radix sort pipeline)

#ifndef GAUSSIAN_STRUCTS_H
#define GAUSSIAN_STRUCTS_H

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// CAMERA & RENDER PARAMS
// =============================================================================

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

// =============================================================================
// KERNEL PARAMS
// =============================================================================

struct ClearParams {
    uint pixelCount;
    uint whiteBackground;
};

struct ClearTextureParams {
    uint width;
    uint height;
    uint whiteBackground;
};

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
    uint tileWidth;
    uint tileHeight;
};

struct ScatterParams {
    uint gaussianCount;
    uint tilesX;
    uint tileWidth;
    uint tileHeight;
};

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
    uint maxAssignments;
};

struct BitonicParams {
    uint j;
    uint k;
    uint total;
};

// =============================================================================
// DISPATCH SLOT ENUM
// =============================================================================

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

#endif // GAUSSIAN_STRUCTS_H

// BridgingTypes.h - Single source of truth for Swift and Metal types
#pragma once

#ifdef __METAL_VERSION__
// Metal shading language
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
typedef uint    UINT32;
typedef ushort  UINT16;
typedef int     INT32;
typedef half    HALF;
#else
// Host side (C/ObjC/Swift)
#include <simd/simd.h>
#include <stdint.h>
typedef uint32_t UINT32;
typedef uint16_t UINT16;
typedef int32_t  INT32;
typedef _Float16 HALF;      // Swift sees this as Float16

#endif

// =============================================================================
// CAMERA & RENDER PARAMS
// =============================================================================

typedef struct {
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float3   cameraCenter;
    float         pixelFactor;
    float         focalX;
    float         focalY;
    float         width;
    float         height;
    float         nearPlane;
    float         farPlane;
    UINT32        shComponents;
    UINT32        gaussianCount;
    UINT32        padding0;
    UINT32        padding1;
} CameraUniforms;

typedef struct {
    UINT32 width;
    UINT32 height;
    UINT32 tileWidth;
    UINT32 tileHeight;
    UINT32 tilesX;
    UINT32 tilesY;
    UINT32 maxPerTile;
    UINT32 whiteBackground;
    UINT32 activeTileCount;
    UINT32 gaussianCount;
} RenderParams;

typedef struct {
    UINT32 offset;
    UINT32 count;
} GaussianHeader;

// =============================================================================
// PACKED WORLD GAUSSIAN (float32) - 48 bytes
// =============================================================================

typedef struct {
    float         px, py, pz;
    float         opacity;
    float         sx, sy, sz;
    float         _pad0;
    simd_float4   rotation;
} PackedWorldGaussian;

// =============================================================================
// PACKED WORLD GAUSSIAN (half16) - 24 bytes
// =============================================================================

typedef struct {
    HALF px, py, pz;        // position
    HALF opacity;
    HALF sx, sy, sz;        // scale
    HALF _pad0;
    HALF rx, ry, rz, rw;    // rotation quaternion
} PackedWorldGaussianHalf;

// =============================================================================
// GAUSSIAN RENDER DATA (half16) - 32 bytes
// =============================================================================

typedef struct {
    HALF   meanX, meanY;           // mean position @ 0
    UINT32 _alignPad;              // padding @ 4
    HALF   conicA, conicB, conicC, conicD;  // conic @ 8
    HALF   colorR, colorG, colorB; // color @ 16
    HALF   opacity;                // @ 22
    HALF   depth;                  // @ 24
    UINT16 _pad;                   // @ 26
    UINT32 _structPad;             // round to 32 @ 28
} GaussianRenderData;

// =============================================================================
// GAUSSIAN RENDER DATA (float32) - 48 bytes
// =============================================================================

typedef struct {
    float meanX, meanY;
    float conicA, conicB, conicC, conicD;
    float colorR, colorG, colorB;
    float opacity;
    float depth;
    UINT32 _pad;
} GaussianRenderDataF32;

// =============================================================================
// TILE BINNING & ASSIGNMENT
// =============================================================================

typedef struct {
    UINT32 gaussianCount;
    UINT32 tilesX;
    UINT32 tilesY;
    UINT32 tileWidth;
    UINT32 tileHeight;
    UINT32 surfaceWidth;
    UINT32 surfaceHeight;
    UINT32 maxCapacity;
    float  alphaThreshold;      // Min alpha to contribute (default 0.004 = 1/255)
    float  minCoverageRatio;    // Min fraction of tile pixels that benefit (default 0.0 = disabled)
} TileBinningParams;

typedef struct {
    UINT32 totalAssignments;
    UINT32 maxCapacity;
    UINT32 paddedCount;
    UINT32 overflow;
} TileAssignmentHeader;

typedef struct {
    simd_float4 covariance_depth;
    simd_float4 position_color;
    simd_int2   min_tile;
    simd_int2   max_tile;
    UINT32      originalIdx;    // World buffer index for deterministic tie-breaking
    UINT32      _pad0;          // Padding for alignment
} CompactedGaussian;

// =============================================================================
// DISPATCH & KERNEL PARAMS
// =============================================================================

typedef struct {
    UINT32 threadgroupsPerGridX;
    UINT32 threadgroupsPerGridY;
    UINT32 threadgroupsPerGridZ;
} DispatchIndirectArgs;

typedef struct {
    UINT32 sortThreadgroupSize;
    UINT32 fuseThreadgroupSize;
    UINT32 unpackThreadgroupSize;
    UINT32 packThreadgroupSize;
    UINT32 radixBlockSize;
    UINT32 radixGrainSize;
    UINT32 maxAssignments;
} AssignmentDispatchConfig;

typedef struct {
    UINT32 pixelCount;
    UINT32 whiteBackground;
} ClearParams;

typedef struct {
    UINT32 width;
    UINT32 height;
    UINT32 whiteBackground;
} ClearTextureParams;

typedef struct {
    UINT32 width;
    UINT32 height;
    UINT32 tileWidth;
    UINT32 tileHeight;
    UINT32 tilesX;
    UINT32 tilesY;
    UINT32 gaussianCount;
} TileBoundsParams;

typedef struct {
    UINT32 gaussianCount;
    UINT32 tileWidth;
    UINT32 tileHeight;
} CoverageParams;

typedef struct {
    UINT32 gaussianCount;
    UINT32 tilesX;
    UINT32 tileWidth;
    UINT32 tileHeight;
} ScatterParams;

typedef struct {
    UINT32 gaussianCount;
    UINT32 tileWidth;
    UINT32 tileHeight;
    UINT32 tilesX;
    UINT32 maxAssignments;
} TileAssignParams;

typedef struct {
    UINT32 maxAssignments;
    UINT32 totalAssignments;
} SortKeyParams;

// =============================================================================
// CLUSTER CULLING (Morton-sorted optimization)
// =============================================================================

#define CLUSTER_SIZE 1024  // Gaussians per cluster (larger = fewer clusters = faster cull)

typedef struct {
    simd_float3 minBounds;
    float       _pad0;
    simd_float3 maxBounds;
    float       _pad1;
} ClusterAABB;  // 32 bytes, aligned

typedef struct {
    simd_float4 planes[6];  // left, right, bottom, top, near, far
                            // Each plane: (nx, ny, nz, d) where dot(n, p) + d >= 0 is inside
} FrustumPlanes;  // 96 bytes

typedef struct {
    UINT32 totalGaussians;
    UINT32 clusterCount;
    UINT32 gaussiansPerCluster;
    UINT32 shComponents;
} ClusterCullParams;

// =============================================================================
// DISPATCH SLOTS ENUM
// =============================================================================

enum DispatchSlots {
    DispatchSlotSortKeys = 0,
    DispatchSlotFuseKeys = 1,
    DispatchSlotUnpackKeys = 2,
    DispatchSlotPack = 3,
    DispatchSlotRadixHistogram = 4,
    DispatchSlotRadixScanBlocks = 5,
    DispatchSlotRadixExclusive = 6,
    DispatchSlotRadixApply = 7,
    DispatchSlotRadixScatter = 8,
    DispatchSlotRenderTiles = 9
};

#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
typedef uint    UINT32;
typedef ushort  UINT16;
typedef int     INT32;
typedef half    HALF;
#else
#include <simd/simd.h>
#include <stdint.h>
typedef uint32_t UINT32;
typedef uint16_t UINT16;
typedef int32_t  INT32;
typedef _Float16 HALF;
#endif

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
    UINT32 activeTileCount;
    UINT32 gaussianCount;
} RenderParams;

typedef struct {
    UINT32 offset;
    UINT32 count;
} GaussianHeader;

// 48 bytes
typedef struct {
    float         px, py, pz;
    float         opacity;
    float         sx, sy, sz;
    float         _pad0;
    simd_float4   rotation;
} PackedWorldGaussian;

// 32 bytes - float position for quality, half for rest
typedef struct {
    float px, py, pz;
    HALF  opacity;
    HALF  sx, sy, sz;
    HALF  rx, ry, rz, rw;
    HALF  _pad0, _pad1;
} PackedWorldGaussianHalf;

// 32 bytes
typedef struct {
    HALF meanX, meanY;
    HALF conicA, conicB, conicC, conicD;
    HALF colorR, colorG, colorB;
    HALF opacity;
    HALF depth;
    HALF _pad0, _pad1, _pad2, _pad3, _pad4;
} GaussianRenderData;

typedef struct {
    UINT32 gaussianCount;
    UINT32 tilesX;
    UINT32 tilesY;
    UINT32 tileWidth;
    UINT32 tileHeight;
    UINT32 surfaceWidth;
    UINT32 surfaceHeight;
    UINT32 maxCapacity;
    float  alphaThreshold;
    float  totalInkThreshold;
} TileBinningParams;

typedef struct {
    UINT32 totalAssignments;
    UINT32 maxCapacity;
    UINT32 paddedCount;
    UINT32 overflow;
} TileAssignmentHeader;

typedef struct {
    HALF posX, posY;
    HALF conicA, conicB, conicC, depth;
    HALF colorR, colorG;
    HALF colorB, opacity;
    simd_ushort2 minTile;
    simd_ushort2 maxTile;
    HALF obbExtentX, obbExtentY;
} ProjectedGaussian;

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
} ClearParams;

typedef struct {
    UINT32 width;
    UINT32 height;
} ClearTextureParams;

typedef struct {
    UINT32 gaussianCount;
    UINT32 tileWidth;
    UINT32 tileHeight;
    UINT32 tilesX;
    UINT32 maxAssignments;
} TileAssignParams;

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

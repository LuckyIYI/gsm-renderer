#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
typedef uint    UINT32;
typedef ushort  UINT16;
typedef int     INT32;
typedef half    HALF;
typedef uchar   UCHAR;
#else
#include <simd/simd.h>
#include <stdint.h>
typedef uint32_t UINT32;
typedef uint16_t UINT16;
typedef int32_t  INT32;
typedef _Float16 HALF;
typedef uint8_t  UCHAR;
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
    float         inputIsSRGB;
    float         _pad1;
    float         _pad2;
    float         _pad3;
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

// 32 bytes
typedef struct {
    float px, py, pz;
    HALF  opacity;
    HALF  sx, sy, sz;
    HALF  rx, ry, rz, rw;
    HALF  _pad0, _pad1;
} PackedWorldGaussianHalf;

// 16 bytes - compact render data for global/depth-first tiled renderers.
typedef struct {
    HALF meanX, meanY;                    // 4 bytes - screen position
    UINT16 theta;                         // 2 bytes - angle in [0, pi) packed to 0..65535
    HALF sigma1;                          // 2 bytes - major axis sigma (pixels)
    HALF sigma2;                          // 2 bytes - minor axis sigma (pixels)
    HALF depth;                           // 2 bytes - depth for output
    UCHAR colorR, colorG, colorB;         // 3 bytes - 8-bit color
    UCHAR opacity;                        // 1 byte - 8-bit opacity
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


typedef struct {
    // Left eye (offset 0)
    simd_float4x4 leftViewMatrix;        // 0-63
    simd_float4x4 leftProjectionMatrix;  // 64-127
    // Camera center + focal (offset 128, 32 bytes to align next float4x4)
    float         leftCameraCenterX;     // 128
    float         leftCameraCenterY;     // 132
    float         leftCameraCenterZ;     // 136
    float         leftFocalX;            // 140
    float         leftFocalY;            // 144
    float         _padLeft0;             // 148
    float         _padLeft1;             // 152
    float         _padLeft2;             // 156
    // End left: 160 bytes (16-byte aligned)

    // Right eye (offset 160)
    simd_float4x4 rightViewMatrix;       // 160-223
    simd_float4x4 rightProjectionMatrix; // 224-287
    // Camera center + focal (offset 288, 32 bytes to align next section)
    float         rightCameraCenterX;    // 288
    float         rightCameraCenterY;    // 292
    float         rightCameraCenterZ;    // 296
    float         rightFocalX;           // 300
    float         rightFocalY;           // 304
    float         _padRight0;            // 308
    float         _padRight1;            // 312
    float         _padRight2;            // 316
    // End right: 320 bytes (16-byte aligned)

    // Shared parameters (offset 320)
    float         width;                 // 320
    float         height;                // 324
    float         nearPlane;             // 328
    float         farPlane;              // 332
    UINT32        shComponents;          // 336
    UINT32        gaussianCount;         // 340
    float         inputIsSRGB;           // 344
    float         _padShared1;           // 348
    // End shared: 352 bytes (16-byte aligned)

    // Scene transform (offset 352)
    simd_float4x4 sceneTransform;        // 352-415
} StereoCameraUniforms;
// Total size: 416 bytes


// Depth-first header for single-eye rendering and radix sort
typedef struct {
    UINT32 visibleCount;
    UINT32 totalInstances;
    UINT32 paddedVisibleCount;
    UINT32 paddedInstanceCount;
    UINT32 overflow;
    UINT32 padding0;
    UINT32 padding1;
    UINT32 padding2;
} DepthFirstHeader;


typedef struct {
    UINT32 vertexCount;      // 6 for two triangles (quad)
    UINT32 instanceCount;    // Number of visible gaussians
    UINT32 vertexStart;      // 0
    UINT32 baseInstance;     // 0
} DrawIndirectArgs;

typedef struct {
    UINT32 indexCount;
    UINT32 instanceCount;
    UINT32 indexStart;
    INT32  baseVertex;
    UINT32 baseInstance;
} DrawIndexedIndirectArgs;


typedef struct {
    HALF   screenX, screenY;            // 4 bytes - Screen position (pixels)
    UINT16 theta;                       // 2 bytes - angle in [0, pi) packed to 0..65535
    HALF   sigma1;                      // 2 bytes - major axis sigma (pixels)
    HALF   sigma2;                      // 2 bytes - minor axis sigma (pixels)
    HALF   depth;                       // 2 bytes - For sorting
    UCHAR  colorR, colorG, colorB;      // 3 bytes - 8-bit color
    UCHAR  opacity;                     // 1 byte - 8-bit opacity
} InstancedGaussianData;

typedef struct {
    float width;
    float height;
    float farPlane;
    float _pad0;
} HardwareRenderUniforms;


typedef struct {
    // Left eye (12 bytes)
    HALF leftMeanX, leftMeanY;       // 4 bytes - screen position
    HALF leftCxx;                    // 2 bytes - conic.xx
    HALF leftCyy;                    // 2 bytes - conic.yy
    HALF leftCxy2;                   // 2 bytes - 2*conic.xy
    HALF leftDepth;                  // 2 bytes

    // Right eye (12 bytes)
    HALF rightMeanX, rightMeanY;     // 4 bytes - screen position
    HALF rightCxx;                   // 2 bytes - conic.xx
    HALF rightCyy;                   // 2 bytes - conic.yy
    HALF rightCxy2;                  // 2 bytes - 2*conic.xy
    HALF rightDepth;                 // 2 bytes

    // Shared (8 bytes)
    UCHAR colorR, colorG, colorB;    // 3 bytes - 8-bit color
    UCHAR opacity;                   // 1 byte - 8-bit opacity
    HALF centerDepth;                // 2 bytes - for depth sort
    UINT16 _pad0;                    // 2 bytes padding to 32
} StereoTiledRenderData;


typedef struct {
    // Per-eye screen positions (differ due to IPD)
    HALF leftScreenX, leftScreenY;   // 4 bytes
    HALF rightScreenX, rightScreenY; // 4 bytes
    // Per-eye ellipse params (theta + sigmas). Theta in [0, pi) packed to 0..65535.
    UINT16 leftTheta;                // 2 bytes
    UINT16 rightTheta;               // 2 bytes
    HALF leftSigma1, leftSigma2;     // 4 bytes
    HALF rightSigma1, rightSigma2;   // 4 bytes
    HALF depth;                      // 2 bytes - Center eye depth (for sorting)
    UCHAR colorR, colorG, colorB;    // 3 bytes - 8-bit color
    UCHAR opacity;                   // 1 byte - 8-bit opacity
} HardwareProjectedGaussian;

typedef struct {
    UINT32 visibleCount;
    UINT32 paddedCount;
    UINT32 overflow;
    UINT32 _pad0;
} HardwareStereoHeader;

typedef struct {
    simd_float4x4 sceneTransform;
    simd_float4x4 centerViewMatrix;       // Average of left/right view
    simd_float4x4 centerProjectionMatrix; // For covariance projection
    simd_float4x4 leftViewMatrix;
    simd_float4x4 leftProjectionMatrix;
    simd_float4x4 rightViewMatrix;
    simd_float4x4 rightProjectionMatrix;
    float centerCameraPosX, centerCameraPosY, centerCameraPosZ;
    float _padCenter0;  // focalX/Y no longer needed - derived from projection matrix
    float leftCameraPosX, leftCameraPosY, leftCameraPosZ;
    float _padLeft0;
    float rightCameraPosX, rightCameraPosY, rightCameraPosZ;
    float _padRight0;
    float width, height;
    float nearPlane, farPlane;
    UINT32 gaussianCount;
    UINT32 shComponents;
    float totalInkThreshold;
    float inputIsSRGB;
} HardwareStereoProjectionParams;

typedef struct {
    simd_float4x4 sceneTransform;
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    float cameraPosX, cameraPosY, cameraPosZ;
    float _pad0;
    float width, height;
    float nearPlane, farPlane;
    UINT32 gaussianCount;
    UINT32 shComponents;
    float totalInkThreshold;
    float inputIsSRGB;
} HardwareMonoProjectionParams;

typedef struct {
    UINT32 visibleCount;
    UINT32 paddedCount;
    UINT32 overflow;
    UINT32 _pad0;
} HardwareMonoHeader;

// ============================================================================
// MESH EXTRACTION TYPES
// ============================================================================

// Parameters for mesh extraction from Gaussian opacity field
typedef struct {
    simd_float3 gridMin;          // Grid lower bound in world space
    float       voxelSize;        // Size of each voxel
    simd_float3 gridMax;          // Grid upper bound in world space
    float       isoLevel;         // Isosurface threshold (typically 0.5)
    UINT32      gridDimX;         // Grid dimensions
    UINT32      gridDimY;
    UINT32      gridDimZ;
    UINT32      gaussianCount;    // Number of Gaussians
    float       opacityCutoff;    // Min opacity to consider (e.g., 1/255)
    float       sigmaCutoff;      // Distance cutoff in sigma units (e.g., 3.0)
    UINT32      _pad0;
    UINT32      _pad1;
} MeshExtractionParams;

// Mesh extraction header with counts
typedef struct {
    UINT32 totalVertices;
    UINT32 totalTriangles;
    UINT32 activeCells;          // Cells that produce triangles
    UINT32 overflow;             // Set if buffers exceeded
} MeshExtractionHeader;

// Output vertex from mesh extraction
typedef struct {
    simd_float3 position;
    float       _pad0;
    simd_float3 normal;
    float       _pad1;
    simd_float3 color;           // Optional: color from opacity field
    float       opacity;         // Opacity at this vertex
} MeshVertex;

// Per-cell triangle count for prefix sum
typedef struct {
    UINT32 triangleCount;
    UINT32 vertexOffset;        // After prefix sum: starting vertex index
} CellTriangleInfo;

// Spatial hash entry for accelerating Gaussian lookups
typedef struct {
    UINT32 offset;              // Offset into gaussianIndices array
    UINT32 count;               // Number of Gaussians in this cell
} SpatialHashEntry;

// Bounds reduction result
typedef struct {
    simd_float3 minBound;
    float       _pad0;
    simd_float3 maxBound;
    float       _pad1;
} BoundsResult;

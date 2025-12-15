#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
typedef uint    UINT32;
typedef int     INT32;
typedef half    HALF;
#else
#include <simd/simd.h>
#include <stdint.h>
typedef uint32_t UINT32;
typedef int32_t  INT32;
typedef _Float16 HALF;
#endif

// ============================================================================
// LOD Tree Types for Hierarchical Gaussian Splatting
// Based on "A Hierarchical 3D Gaussian Representation" (SIGGRAPH 2024)
// ============================================================================

// Invalid index marker
#define LOD_INVALID_INDEX 0xFFFFFFFF

// LOD Tree Node - 64 bytes, GPU-aligned
// Binary BVH node with AABB bounds and tree structure
// For N leaves (original Gaussians), tree has 2N-1 total nodes:
// - Nodes 0..N-2: interior nodes (merged Gaussians)
// - Nodes N-1..2N-2: leaf nodes (original Gaussians)
typedef struct {
    // AABB bounds (24 bytes)
    float aabbMinX, aabbMinY, aabbMinZ;
    float aabbMaxX, aabbMaxY, aabbMaxZ;

    // Tree structure (16 bytes)
    UINT32 leftChild;      // Child index or LOD_INVALID_INDEX for leaf
    UINT32 rightChild;     // Child index or LOD_INVALID_INDEX for leaf
    UINT32 parentIndex;    // Parent index or LOD_INVALID_INDEX for root
    UINT32 gaussianIndex;  // For leaves: index in original buffer
                           // For interior: index in merged buffer

    // Node info (8 bytes)
    UINT32 isLeaf;         // 1 if leaf node, 0 if interior node
    UINT32 subtreeCount;   // Number of original Gaussians in subtree

    // Padding to 64 bytes (16 bytes)
    float _pad0, _pad1, _pad2, _pad3;
} LODNode;

// LOD Tree Header - metadata for the tree
typedef struct {
    UINT32 nodeCount;        // Total nodes in tree (2N-1 for N leaves)
    UINT32 leafCount;        // Number of leaf nodes (original Gaussians)
    UINT32 interiorCount;    // Number of interior nodes (N-1)
    UINT32 maxDepth;         // Maximum tree depth
} LODTreeHeader;

// LOD Selection Parameters - passed to GPU for cut selection
typedef struct {
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    float screenWidth;
    float screenHeight;
    float granularityThreshold;  // Target pixel size τ (e.g., 1.0 = 1 pixel)
    float nearPlane;
    UINT32 nodeCount;
    UINT32 leafCount;
    UINT32 enableTransitions;    // 1 to enable smooth transitions, 0 for hard cuts
    float transitionRange;       // Range for smooth blending (default: 0.5)
} LODSelectionParams;

// Per-node granularity computed on GPU
typedef struct {
    float granularity;      // Projected AABB diagonal in pixels
    float parentGranularity;// Parent's granularity (for cut selection)
    float transitionWeight; // Interpolation weight t ∈ [0,1] for smooth LOD
    UINT32 selected;        // 1 if node is in the cut, 0 otherwise
} LODNodeGranularity;

// LOD Selection Output Header
typedef struct {
    UINT32 selectedCount;    // Number of selected nodes
    UINT32 leafCount;        // Leaves among selected (use original Gaussians)
    UINT32 interiorCount;    // Interior nodes (use merged Gaussians)
    UINT32 _pad0;
} LODSelectionHeader;

// Selected node info - written by selection kernel
typedef struct {
    UINT32 nodeIndex;        // Index in LODNode array
    UINT32 gaussianIndex;    // Index in appropriate Gaussian buffer
    UINT32 isLeaf;           // 1 if from original buffer, 0 if merged
    float transitionWeight;  // For smooth LOD transitions
} LODSelectedNode;

// Merged Gaussian for interior nodes - same format as PackedWorldGaussian
// Computed via KL-divergence minimization: μ = Σwᵢμᵢ, Σ = Σwᵢ(Σᵢ + δμδμᵀ)
// Weight wᵢ = opacity × sqrt(|projected covariance|)
// Stored in separate buffer, indexed by (nodeIndex - leafCount)

// Merged SH coefficients for interior nodes
// Same format as original harmonics buffer
// Averaged using same weights as Gaussian merging

// LOD-Fused Projection Parameters - for fused LOD+projection kernel
typedef struct {
    float granularityThreshold;  // Target pixel size τ
    UINT32 nodeCount;            // Total LOD nodes
    UINT32 leafCount;            // Number of leaf nodes (for indexing merged buffer)
    UINT32 _pad0;                // Padding for alignment
} LODProjectParams;

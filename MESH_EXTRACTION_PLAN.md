# Mesh Extraction from Gaussian Splatting - Implementation Plan

## Overview

This document describes the implementation of mesh extraction based on **Gaussian Opacity Fields (GOF)** paper (Yu et al., 2024). The algorithm extracts triangle meshes from 3D Gaussian representations by computing an opacity field and using marching cubes to extract the isosurface.

**Reference:** https://github.com/autonomousvision/gaussian-opacity-fields

---

## 1. Mathematical Foundation

### 1.1 3D Gaussian Definition

Each Gaussian is defined by:
- **Position:** μ ∈ ℝ³
- **Covariance:** Σ = R · S · Sᵀ · Rᵀ where:
  - R is the rotation matrix from quaternion q
  - S = diag(sx, sy, sz) is the scale matrix
- **Opacity:** α ∈ [0, 1]

### 1.2 Gaussian Evaluation at a 3D Point

For a query point **p** ∈ ℝ³, the Gaussian contribution is:

```
G(p) = exp(-½ · (p - μ)ᵀ · Σ⁻¹ · (p - μ))
```

The weighted opacity contribution from Gaussian i at point p:

```
αᵢ(p) = αᵢ · G(p)
```

### 1.3 Opacity Field Computation

The opacity at a 3D point is computed by accumulating contributions from all Gaussians using alpha compositing:

```
α(p) = 1 - ∏ᵢ (1 - αᵢ(p))
```

In log-space for numerical stability:

```
α(p) = 1 - exp(∑ᵢ log(1 - αᵢ(p)))
```

Where αᵢ(p) is clamped to [0, 0.99] before accumulation.

### 1.4 SDF Conversion

The opacity field is converted to a signed distance field (SDF) for isosurface extraction:

```
SDF(p) = 0.5 - α(p)
```

- SDF < 0: Inside the surface (α > 0.5)
- SDF > 0: Outside the surface (α < 0.5)
- SDF = 0: On the surface (α = 0.5)

### 1.5 Inverse Covariance (Precision Matrix)

For efficient computation, we use the inverse covariance (precision matrix):

```
Σ⁻¹ = (R · S · Sᵀ · Rᵀ)⁻¹ = R · S⁻² · Rᵀ
```

Where S⁻² = diag(1/sx², 1/sy², 1/sz²)

The Mahalanobis distance squared:

```
d²(p) = (p - μ)ᵀ · Σ⁻¹ · (p - μ)
```

---

## 2. Algorithm Pipeline

### 2.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     MESH EXTRACTION PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. GRID GENERATION                                              │
│     ├── Define 3D grid bounds (AABB of Gaussians + margin)      │
│     └── Create uniform grid of query points                      │
│                                                                  │
│  2. OPACITY FIELD COMPUTATION (Compute Shader)                   │
│     ├── For each grid point p:                                   │
│     │   ├── For each Gaussian i:                                 │
│     │   │   ├── Compute d² = Mahalanobis distance²              │
│     │   │   ├── Compute αᵢ(p) = αᵢ · exp(-0.5 · d²)            │
│     │   │   └── Accumulate: T *= (1 - αᵢ(p))                    │
│     │   └── Final: α(p) = 1 - T                                 │
│     └── Output: 3D opacity grid                                  │
│                                                                  │
│  3. SDF CONVERSION                                               │
│     └── SDF(p) = 0.5 - α(p)                                     │
│                                                                  │
│  4. MARCHING CUBES (Compute Shader)                              │
│     ├── For each grid cell:                                      │
│     │   ├── Classify 8 corners (inside/outside)                 │
│     │   ├── Lookup triangle configuration                        │
│     │   └── Generate vertices via linear interpolation           │
│     └── Output: Triangle mesh                                    │
│                                                                  │
│  5. VERTEX REFINEMENT (Optional)                                 │
│     └── Binary search to refine vertex positions                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Grid Generation

The grid bounds are computed from the Gaussian point cloud:

```
bounds_min = min(μᵢ - 3·max(sᵢ)) - margin
bounds_max = max(μᵢ + 3·max(sᵢ)) + margin
```

Grid resolution is typically 256³ or 512³ depending on scene complexity.

### 2.3 Spatial Acceleration Structure

To avoid O(N·M) complexity where N = grid points and M = Gaussians, we use a uniform spatial hash:

1. **Voxelize Gaussians:** Each Gaussian covers voxels within its 3σ radius
2. **Build Hash Table:** Map voxel IDs to lists of overlapping Gaussians
3. **Query:** For each grid point, only evaluate Gaussians in nearby voxels

This reduces complexity to approximately O(N·k) where k is the average number of Gaussians per voxel.

---

## 3. Marching Cubes Algorithm

### 3.1 Cell Classification

Each cell has 8 corners. A corner is "inside" if SDF < 0 (opacity > 0.5).

The 8 corners form a binary index (0-255):
```
cornerIndex = (c0 < 0) | ((c1 < 0) << 1) | ... | ((c7 < 0) << 7)
```

### 3.2 Edge Table

256-entry lookup table mapping corner configurations to edge intersections.

```
edgeTable[cornerIndex] = 12-bit mask indicating which edges are crossed
```

### 3.3 Triangle Table

256×16 table mapping configurations to triangle vertex sequences:

```
triTable[cornerIndex][0..15] = edge indices forming triangles (-1 = end)
```

### 3.4 Vertex Interpolation

For an edge with endpoints p1, p2 and values v1, v2:

```
t = v1 / (v1 - v2)
vertex = p1 + t · (p2 - p1)
```

This places the vertex where SDF = 0.

---

## 4. Implementation Details

### 4.1 Data Structures

```metal
// Mesh extraction parameters
struct MeshExtractionParams {
    float3 gridMin;        // Grid lower bound
    float3 gridMax;        // Grid upper bound
    uint3 gridDims;        // Grid dimensions (e.g., 256×256×256)
    float3 voxelSize;      // Size of each voxel
    float isoLevel;        // Isosurface threshold (0.5)
    uint gaussianCount;    // Number of Gaussians
};

// Output vertex
struct MeshVertex {
    float3 position;
    float3 normal;
    float3 color;
};

// Output triangle
struct MeshTriangle {
    uint v0, v1, v2;
};
```

### 4.2 Compute Shader Stages

#### Stage 1: Compute Gaussian AABBs and Grid Bounds

```metal
kernel void computeGaussianBounds(
    const device PackedWorldGaussian* gaussians,
    device float3* minBounds,
    device float3* maxBounds,
    device atomic_uint* counter,
    uint gid [[thread_position_in_grid]]
);
```

#### Stage 2: Build Spatial Hash

```metal
kernel void buildSpatialHash(
    const device PackedWorldGaussian* gaussians,
    device uint* hashTable,
    device uint* gaussianLists,
    constant MeshExtractionParams& params,
    uint gid [[thread_position_in_grid]]
);
```

#### Stage 3: Compute Opacity Field

```metal
kernel void computeOpacityField(
    const device PackedWorldGaussian* gaussians,
    const device uint* hashTable,
    const device uint* gaussianLists,
    device float* opacityGrid,
    constant MeshExtractionParams& params,
    uint3 gid [[thread_position_in_grid]]
);
```

#### Stage 4: Marching Cubes - Count Triangles

```metal
kernel void marchingCubesCount(
    const device float* opacityGrid,
    device uint* triangleCounts,
    constant MeshExtractionParams& params,
    uint3 gid [[thread_position_in_grid]]
);
```

#### Stage 5: Prefix Sum for Triangle Offsets

Reuse existing `GaussianPrefixScan.h` infrastructure.

#### Stage 6: Marching Cubes - Generate Triangles

```metal
kernel void marchingCubesGenerate(
    const device float* opacityGrid,
    const device uint* triangleOffsets,
    device MeshVertex* vertices,
    device MeshTriangle* triangles,
    constant MeshExtractionParams& params,
    uint3 gid [[thread_position_in_grid]]
);
```

### 4.3 Optimizations

1. **Threadgroup Memory:** Cache opacity values for neighboring voxels
2. **Coalesced Access:** Process voxels in Morton order for better cache locality
3. **Early Exit:** Skip cells with all corners inside or outside
4. **LOD:** Multi-resolution extraction for large scenes

---

## 5. Swift Interface

```swift
public struct MeshExtractionConfig {
    public var gridResolution: Int = 256
    public var isoLevel: Float = 0.5
    public var margin: Float = 0.1
    public var refinementIterations: Int = 4
}

public struct ExtractedMesh {
    public var vertices: [SIMD3<Float>]
    public var normals: [SIMD3<Float>]
    public var colors: [SIMD3<Float>]
    public var triangles: [SIMD3<UInt32>]
}

public class MeshExtractor {
    public init(device: MTLDevice) throws

    public func extractMesh(
        gaussians: MTLBuffer,
        gaussianCount: Int,
        harmonics: MTLBuffer?,
        config: MeshExtractionConfig
    ) async throws -> ExtractedMesh
}
```

---

## 6. Memory Requirements

For a 256³ grid:
- Opacity grid: 256³ × 4 bytes = 64 MB
- Spatial hash: ~16 MB (depends on scene)
- Triangle buffer: ~100 MB (worst case)
- Total: ~200 MB

For a 512³ grid:
- Opacity grid: 512³ × 4 bytes = 512 MB
- Spatial hash: ~32 MB
- Triangle buffer: ~400 MB
- Total: ~1 GB

---

## 7. Validation

### 7.1 Unit Tests

1. Single Gaussian: Should produce sphere-like mesh
2. Two overlapping Gaussians: Should merge correctly
3. Separated Gaussians: Should produce distinct components

### 7.2 Numerical Validation

Compare opacity values against reference Python implementation:
```python
def compute_opacity(point, gaussians):
    transmittance = 1.0
    for g in gaussians:
        d2 = mahalanobis_squared(point, g.mean, g.cov_inv)
        alpha = g.opacity * exp(-0.5 * d2)
        alpha = min(alpha, 0.99)
        transmittance *= (1 - alpha)
    return 1 - transmittance
```

---

## 8. File Structure

```
Sources/Renderer/
├── MeshExtraction/
│   ├── MeshExtractor.swift           # Main Swift interface
│   ├── MeshExtractionResources.swift # Buffer management
│   ├── MeshExtractionShaders.metal   # All compute shaders
│   ├── MarchingCubesTables.h         # Edge/triangle lookup tables
│   └── Encoders/
│       ├── ComputeBoundsEncoder.swift
│       ├── BuildHashEncoder.swift
│       ├── OpacityFieldEncoder.swift
│       └── MarchingCubesEncoder.swift
└── Shared/
    └── MeshExtractionTypes.h         # Shared type definitions
```

---

## 9. Implementation Order

1. **Phase 1: Core Types** (This PR)
   - Add MeshExtractionTypes.h to BridgingTypes.h
   - Create marching cubes lookup tables

2. **Phase 2: Opacity Field** (This PR)
   - Implement computeOpacityField shader
   - No spatial acceleration (brute force for correctness)

3. **Phase 3: Marching Cubes** (This PR)
   - Implement count and generate shaders
   - Reuse prefix sum infrastructure

4. **Phase 4: Swift Interface** (This PR)
   - MeshExtractor class
   - Memory management

5. **Phase 5: Optimizations** (Future)
   - Spatial hash acceleration
   - Multi-resolution support
   - Binary search refinement

---

## 10. References

1. Yu, Z., et al. "Gaussian Opacity Fields: Efficient and Compact Surface Reconstruction in Unbounded Scenes." arXiv:2404.10772 (2024)
2. Lorensen, W. E., & Cline, H. E. "Marching cubes: A high resolution 3D surface construction algorithm." ACM SIGGRAPH (1987)
3. Kerbl, B., et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." SIGGRAPH (2023)

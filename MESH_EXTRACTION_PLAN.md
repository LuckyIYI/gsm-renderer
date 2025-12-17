# Mesh Extraction from Gaussian Splatting - Implementation Plan

## Overview

This document describes the implementation of mesh extraction based on **Gaussian Opacity Fields (GOF)** paper (Yu et al., 2024). The algorithm extracts triangle meshes from 3D Gaussian representations by computing an opacity field and using marching cubes/tetrahedra to extract the isosurface.

**Paper:** [arXiv:2404.10772](https://arxiv.org/abs/2404.10772)
**Code:** https://github.com/autonomousvision/gaussian-opacity-fields
**Project:** https://niujinshuchong.github.io/gaussian-opacity-fields/

---

## 1. Mathematical Foundation (Complete from Paper)

### 1.1 3D Gaussian Parameterization

Each 3D Gaussian is defined by:
- **Position (mean):** μ ∈ ℝ³
- **Scale:** S = diag(sₓ, sᵧ, sᵤ) - diagonal scale matrix
- **Rotation:** R ∈ SO(3) - rotation matrix from quaternion q = (qₓ, qᵧ, qᵤ, qᵥ)
- **Opacity:** α ∈ [0, 1]
- **Spherical Harmonics:** SH coefficients for view-dependent color

**Covariance Matrix:**
```
Σ = R · S · Sᵀ · Rᵀ = R · S² · Rᵀ
```

**Inverse Covariance (Precision Matrix):**
```
Σ⁻¹ = R · S⁻² · Rᵀ

where S⁻² = diag(1/sₓ², 1/sᵧ², 1/sᵤ²)
```

### 1.2 Ray-Gaussian Intersection (GOF Key Contribution)

Unlike standard 3DGS which projects Gaussians to 2D, GOF uses **ray-tracing based volume rendering**.

**Ray Definition:**
```
r(t) = o + t·d

where:
  o = ray origin (camera position)
  d = ray direction (normalized)
  t = distance along ray
```

**Gaussian in Ray Coordinates:**

The GOF paper transforms the Gaussian into a quadratic form along the ray. Given the view-to-Gaussian transformation, the evaluation becomes:

```
Q(t) = AA·t² + BB·t + CC
```

Where (from the CUDA implementation):
```
ray_point = (ray.x, ray.y, 1.0)  // homogeneous ray direction
n = Σ⁻¹_view · ray_point         // transformed by view-space precision

AA = ray.x · n[0] + ray.y · n[1] + n[2]
BB = 2 · (b[0]·ray.x + b[1]·ray.y + b[2])
CC = c
```

The precomputed values:
```
Σ⁻¹_view[6] = Rᵀ · S⁻² · R      // symmetric 3x3, upper triangle (6 values)
b[3] = S⁻² · R · t_cam           // translation component
c = t_camᵀ · S⁻² · t_cam         // constant term
```

**Optimal Depth (intersection point):**
```
t* = -BB / (2·AA)
```

**Minimum Quadratic Value (Gaussian exponent):**
```
min_value = -(BB/AA)·(BB/4) + CC = CC - BB²/(4·AA)
```

### 1.3 Alpha/Opacity Computation

**Per-Gaussian Alpha at Intersection:**
```
power = -0.5 · min_value
G = exp(power)
α = min(0.99, opacity · G)
```

The 0.99 clamp prevents numerical issues with fully opaque regions.

**Validity Check:**
```
if α < 1/255: skip this Gaussian (negligible contribution)
```

### 1.4 Transmittance-Based Alpha Compositing

**Front-to-Back Compositing:**
```
C = Σᵢ (colorᵢ · αᵢ · Tᵢ)
T_{i+1} = Tᵢ · (1 - αᵢ)

where T₀ = 1.0 (initial transmittance)
```

**Final Pixel Color:**
```
C_final = C + T_final · C_background
```

**Accumulated Alpha (Opacity Field Value):**
```
α_accumulated = 1 - T_final = 1 - ∏ᵢ(1 - αᵢ)
```

### 1.5 Multi-View Opacity Field (for Mesh Extraction)

The GOF paper evaluates opacity at 3D query points across **multiple camera views** and takes the **minimum**:

```
α(p) = 1 - min_view[ T_view(p) ]

where T_view(p) = accumulated transmittance at point p from view
```

This multi-view minimum ensures conservative surface estimation.

**Simplified Single-View Version (our implementation):**
```
For query point p:
  T = 1.0
  for each Gaussian i:
    d² = (p - μᵢ)ᵀ · Σᵢ⁻¹ · (p - μᵢ)   // Mahalanobis distance²
    αᵢ = min(0.99, opacityᵢ · exp(-0.5 · d²))
    T *= (1 - αᵢ)
  α(p) = 1 - T
```

### 1.6 SDF Conversion

**Opacity to Signed Distance Field:**
```
SDF(p) = α(p) - 0.5
```

Note: GOF uses `alpha - 0.5`, not `0.5 - alpha`, so:
- **SDF > 0:** Inside surface (α > 0.5, high opacity)
- **SDF < 0:** Outside surface (α < 0.5, low opacity)
- **SDF = 0:** On surface (α = 0.5, isosurface)

Our implementation uses `isoLevel - α(p)` for flexibility.

### 1.7 Surface Normal Estimation

**From Ray-Gaussian Intersection:**

The GOF paper approximates surface normals using the ray-Gaussian intersection plane:
```
n_surface ≈ n_ray-gaussian = normalize(∇Q(t*))
```

**From Depth Map (training regularization):**
```
n_depth = depth_to_normal(rendered_depth)
```

**From Rendered Normals:**
```
n_render = transform_to_world(rendered_normal_image)
```

---

## 2. Training Loss Functions (from Paper)

### 2.1 RGB Reconstruction Loss
```
L_rgb = (1 - λ_dssim) · L1(render, gt) + λ_dssim · (1 - SSIM(render, gt))
```

### 2.2 Depth Distortion Loss
```
L_distortion = mean(distortion_map)
```

The distortion map penalizes spread-out Gaussian contributions along rays.

### 2.3 Depth-Normal Consistency Loss
```
L_depth_normal = mean(1 - n_render · n_depth)
```

Enforces alignment between:
- Normals computed from rendered depth
- Directly rendered normal vectors

### 2.4 Complete Training Objective
```
L = L_rgb + λ_depth_normal · L_depth_normal + λ_distortion · L_distortion
```

Both geometric losses are activated after warmup iterations.

---

## 3. Mesh Extraction Algorithm

### 3.1 Overview Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     MESH EXTRACTION PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. TETRAHEDRAL GRID GENERATION                                  │
│     ├── Compute AABB of Gaussians + margin                       │
│     ├── Create uniform point grid                                │
│     └── Triangulate points into tetrahedra                       │
│                                                                  │
│  2. OPACITY FIELD EVALUATION                                     │
│     ├── For each grid vertex:                                    │
│     │   ├── For each view (multi-view version):                 │
│     │   │   └── Render/integrate alpha at vertex                │
│     │   └── α(p) = 1 - min_view(T_view)                         │
│     └── Output: per-vertex opacity values                        │
│                                                                  │
│  3. SDF CONVERSION                                               │
│     └── SDF(p) = α(p) - 0.5                                     │
│                                                                  │
│  4. MARCHING TETRAHEDRA                                          │
│     ├── For each tetrahedron:                                    │
│     │   ├── Classify 4 vertices (inside/outside)                │
│     │   ├── Identify crossing edges (sign change)               │
│     │   └── Generate 0-2 triangles via lookup table              │
│     └── Output: initial mesh                                     │
│                                                                  │
│  5. BINARY SEARCH REFINEMENT (8 iterations)                      │
│     ├── For each edge vertex:                                    │
│     │   ├── p_mid = (p_left + p_right) / 2                      │
│     │   ├── Evaluate SDF(p_mid)                                  │
│     │   └── Update left/right based on sign                      │
│     └── Output: refined vertex positions                         │
│                                                                  │
│  6. MESH FILTERING                                               │
│     ├── Filter vertices by distance threshold                    │
│     └── Remove faces with invalid vertices                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Tetrahedra vs Marching Cubes

GOF uses **Marching Tetrahedra** (from Tetra-NeRF) instead of Marching Cubes:
- Tetrahedra have only 16 configurations (vs 256 for cubes)
- No ambiguous cases
- Better suited for adaptive refinement
- Each cube subdivides into 5-6 tetrahedra

**Tetrahedron Edge Indices:**
```
base_tet_edges = [0,1], [0,2], [0,3], [1,2], [1,3], [2,3]  // 6 edges
```

**Valid Tetrahedra (produce triangles):**
```
0 < occ_sum < 4   // Some vertices inside, some outside
```

### 3.3 Vertex Interpolation

**Linear Interpolation on Edges:**
```
For edge with vertices p₁, p₂ and SDF values v₁, v₂:
  t = v₁ / (v₁ - v₂)
  p_surface = p₁ + t · (p₂ - p₁)
```

### 3.4 Binary Search Refinement

**8 Iterations of Bisection:**
```python
for iteration in range(8):
    p_mid = (p_left + p_right) / 2
    sdf_mid = evaluate_opacity(p_mid) - 0.5

    # Update based on sign consistency
    if sign(sdf_mid) == sign(sdf_left):
        p_left = p_mid
        sdf_left = sdf_mid
    else:
        p_right = p_mid
        sdf_right = sdf_mid

p_final = (p_left + p_right) / 2
```

### 3.5 Mesh Filtering

```python
# Filter by distance to nearest Gaussian
mask = (distance_to_gaussian <= scale_threshold)
valid_faces = faces where all 3 vertices satisfy mask
```

---

## 4. Implementation Details (Our Version)

### 4.1 Key Differences from Paper

| Paper (GOF) | Our Implementation |
|-------------|-------------------|
| Multi-view opacity evaluation | Single evaluation at grid points |
| Marching Tetrahedra | Marching Cubes (simpler) |
| Tetra-NeRF triangulation | Uniform cubic grid |
| CPU-based extraction | GPU compute shaders |
| Binary search refinement | Linear interpolation only |

### 4.2 Data Structures

```metal
// Mesh extraction parameters
struct MeshExtractionParams {
    float3 gridMin;          // Grid lower bound
    float  voxelSize;        // Voxel size
    float3 gridMax;          // Grid upper bound
    float  isoLevel;         // Isosurface threshold (0.5)
    uint3  gridDims;         // Grid dimensions
    uint   gaussianCount;    // Number of Gaussians
    float  opacityCutoff;    // Min opacity (1/255)
    float  sigmaCutoff;      // Distance cutoff (3.0)
};
```

### 4.3 Compute Shader: Opacity Field

```metal
kernel void computeOpacityFieldKernel(...) {
    float3 worldPos = gridMin + float3(gid) * voxelSize;

    float transmittance = 1.0;
    for (uint i = 0; i < gaussianCount; i++) {
        // Compute inverse covariance
        float3x3 invCov = R · diag(1/s²) · Rᵀ;

        // Mahalanobis distance squared
        float3 delta = worldPos - position;
        float d2 = dot(delta, invCov * delta);

        // Skip if too far
        if (d2 > sigmaCutoff²) continue;

        // Gaussian contribution
        float alpha = min(0.99, opacity * exp(-0.5 * d2));
        transmittance *= (1 - alpha);

        // Early exit
        if (transmittance < 1e-6) break;
    }

    opacityGrid[index] = 1.0 - transmittance;
}
```

### 4.4 Compute Shader: Marching Cubes

```metal
kernel void marchingCubesGenerateKernel(...) {
    // Sample 8 corners
    float sdf[8];
    for (int i = 0; i < 8; i++) {
        sdf[i] = isoLevel - sampleOpacity(corner[i]);
    }

    // Compute cube index
    int cubeIndex = 0;
    for (int i = 0; i < 8; i++) {
        if (sdf[i] < 0) cubeIndex |= (1 << i);
    }

    // Generate triangles from lookup table
    for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
        // Interpolate vertices on edges
        // Write triangle
    }
}
```

---

## 5. Mathematical Validation

### 5.1 Opacity Field Validation

Compare against reference Python:
```python
def compute_opacity_reference(point, gaussians):
    """Reference implementation matching GOF paper."""
    transmittance = 1.0
    for g in gaussians:
        # Build precision matrix: Σ⁻¹ = R · S⁻² · Rᵀ
        R = quaternion_to_matrix(g.rotation)
        S_inv2 = np.diag(1.0 / (g.scale ** 2))
        precision = R @ S_inv2 @ R.T

        # Mahalanobis distance squared
        delta = point - g.position
        d2 = delta @ precision @ delta

        # Gaussian contribution
        gaussian_value = np.exp(-0.5 * d2)
        alpha = min(0.99, g.opacity * gaussian_value)

        transmittance *= (1 - alpha)

    return 1.0 - transmittance
```

### 5.2 Expected Results

| Test Case | Expected |
|-----------|----------|
| Single spherical Gaussian | Spherical mesh |
| Two overlapping Gaussians | Merged blob |
| Two separated Gaussians | Two components |
| Opacity = 0.5 Gaussian | Mesh at 1σ from center |
| Opacity = 0.99 Gaussian | Mesh at ~3σ from center |

---

## 6. Memory Requirements

| Grid Resolution | Opacity Grid | Triangle Buffer | Total |
|-----------------|--------------|-----------------|-------|
| 64³             | 1 MB         | ~10 MB          | ~15 MB |
| 128³            | 8 MB         | ~50 MB          | ~70 MB |
| 256³            | 64 MB        | ~200 MB         | ~300 MB |
| 512³            | 512 MB       | ~800 MB         | ~1.5 GB |

---

## 7. References

1. Yu, Z., Sattler, T., & Geiger, A. (2024). "Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes." arXiv:2404.10772

2. Kerbl, B., et al. (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." SIGGRAPH

3. Lorensen, W. E., & Cline, H. E. (1987). "Marching cubes: A high resolution 3D surface construction algorithm." ACM SIGGRAPH

4. Kulhanek, J., & Sattler, T. (2023). "Tetra-NeRF: Representing Neural Radiance Fields Using Tetrahedra." ICCV

---

## 8. Summary of Key Equations

| Concept | Formula |
|---------|---------|
| Covariance | Σ = R · S² · Rᵀ |
| Precision | Σ⁻¹ = R · S⁻² · Rᵀ |
| Mahalanobis d² | d² = (p-μ)ᵀ · Σ⁻¹ · (p-μ) |
| Gaussian value | G(p) = exp(-0.5 · d²) |
| Per-Gaussian α | α = min(0.99, opacity · G(p)) |
| Transmittance | T = ∏(1 - αᵢ) |
| Total opacity | α(p) = 1 - T |
| SDF | SDF = α - 0.5 |
| Interpolation | t = v₁/(v₁-v₂), p = p₁ + t·(p₂-p₁) |

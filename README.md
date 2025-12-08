# GSM Renderer

> **Work in progress** - Not production ready.

A high-performance Gaussian Splatting renderer for Apple platforms, written in Swift and Metal.

## Features

- **Three rendering strategies:** Global (radix sort), Local (per-tile bitonic sort), and DepthFirst (hybrid)
- **Float16/Float32 precision:** Configurable for performance vs accuracy trade-off
- **Spherical Harmonics:** Supports SH degrees 0-3 for view-dependent color
- **Optimized for Apple Silicon:** Indirect dispatch, SIMD operations, minimal memory bandwidth
- **Scales to millions of Gaussians:** Tested with 6M+ Gaussians at real-time rates

## Requirements

- macOS 14.0+ / iOS 17.0+
- Swift 6.0+
- Metal-capable device

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/LuckyIYI/gsm-renderer.git", from: "1.0.0")
]
```

## Usage

```swift
import Renderer

// Initialize renderer
let config = RendererConfig(
    maxGaussians: 1_000_000,
    maxWidth: 1920,
    maxHeight: 1080,
    precision: .float16
)
let renderer = try DepthFirstRenderer(device: device, config: config)
// Or: try GlobalRenderer(device: device, config: config)
// Or: try LocalRenderer(device: device, config: config)

// Prepare input data
let input = GaussianInput(
    gaussians: gaussianBuffer,    // PackedWorldGaussian or PackedWorldGaussianHalf
    harmonics: harmonicsBuffer,   // SH coefficients
    gaussianCount: 50_000,
    shComponents: 4               // SH degree 1 = 4 components
)

let camera = CameraParams(
    viewMatrix: viewMatrix,
    projectionMatrix: projectionMatrix,
    position: SIMD3(0, 0, -2),
    focalX: 1000,
    focalY: 1000
)

// Render
let commandBuffer = commandQueue.makeCommandBuffer()!
renderer.render(
    commandBuffer: commandBuffer,
    colorTexture: colorTexture,      // rgba16Float
    depthTexture: depthTexture,      // r16Float (optional)
    input: input,
    camera: camera,
    width: 1920,
    height: 1080
)
commandBuffer.commit()
```

## Renderers

### GlobalRenderer

Uses a global radix sort to order all tile-gaussian assignments by (tile, depth).

**Pipeline:**
1. **Project** - Transform 3D gaussians to 2D screen space, compute conic matrices and colors from SH
2. **Tile Assignment** - Each gaussian writes to all tiles it overlaps, creating (tileId, depth, gaussianId) tuples
3. **Radix Sort** - Sort all assignments globally by (tileId << 32 | depth) using GPU radix sort
4. **Build Headers** - Scan sorted assignments to find offset/count for each tile
5. **Render** - Each tile blends its gaussians front-to-back using sorted order

**Characteristics:**
- Single global sort handles all tiles at once
- Best for scenes with many overlapping gaussians
- Memory: O(total tile assignments)

### LocalRenderer

Sorts gaussians independently per tile using threadgroup memory.

**Pipeline:**
1. **Project** - Transform 3D gaussians to 2D screen space, compute conic matrices and colors from SH
2. **Compact** - Stream compact visible gaussians, removing culled ones
3. **Scatter** - Each gaussian atomically increments tile counters and writes its index to tile lists
4. **Per-Tile Sort** - Each tile sorts its own gaussian list by depth using bitonic sort in threadgroup memory
5. **Render** - Each tile blends its sorted gaussians front-to-back

**Characteristics:**
- Per-tile sorting is cache-friendly
- Efficient for moderate gaussian counts per tile
- Memory: O(tiles × maxPerTile)

### DepthFirstRenderer

Hybrid approach that projects and packs data, then renders with per-tile sorting.

**Pipeline:**
1. **Project** - Transform 3D gaussians to 2D screen space, compute conic matrices and colors from SH
2. **Tile Assignment** - Assign gaussians to tiles with atomic counters
3. **Fuse Keys** - Build (tileId, depth) sort keys for each assignment
4. **Radix Sort** - Sort all assignments globally
5. **Unpack** - Extract sorted gaussian indices per tile
6. **Render** - Each tile blends gaussians front-to-back, 2×2 pixels per thread

**Characteristics:**
- Combines global sorting with efficient packed render data
- 16×16 pixel tiles, 8×8 threadgroup

## Data Formats

### Gaussian Input (Float32 - 48 bytes)
```c
struct PackedWorldGaussian {
    float px, py, pz;           // Position
    float opacity;
    float sx, sy, sz;           // Scale
    float _pad;
    float4 rotation;            // Quaternion
};
```

### Gaussian Input (Float16 - 32 bytes)
```c
struct PackedWorldGaussianHalf {
    float px, py, pz;           // Position (float for quality)
    half opacity;
    half sx, sy, sz;            // Scale
    half rx, ry, rz, rw;        // Rotation quaternion
    half _pad0, _pad1;
};
```

### Spherical Harmonics

| SH Degree | Components | Coefficients per Gaussian |
|-----------|------------|---------------------------|
| 0 (DC)    | 1          | 3 (RGB)                   |
| 1         | 4          | 12                        |
| 2         | 9          | 27                        |
| 3         | 16         | 48                        |

## Building Shaders

Pre-compiled `.metallib` files are included. To rebuild after modifying shaders:

```bash
./compile_shaders.sh          # macOS
./compile_shaders.sh ios      # iOS device
./compile_shaders.sh ios-sim  # iOS Simulator
```

## Architecture

```
Sources/
├── Renderer/
│   ├── GlobalRenderer/
│   │   ├── GlobalRenderer.swift      # Global radix sort renderer
│   │   ├── GlobalShaders.metal
│   │   └── Encoders/                 # Radix sort, projection, tile assignment
│   ├── LocalRenderer/
│   │   ├── LocalRenderer.swift       # Per-tile sort renderer
│   │   ├── LocalShaders.metal
│   │   └── Encoders/                 # Compaction, scatter, per-tile sort
│   ├── DepthFirstRenderer/
│   │   ├── DepthFirstRenderer.swift  # Hybrid renderer
│   │   ├── DepthFirstShaders.metal
│   │   └── Encoders/                 # Fused pipeline encoders
│   ├── GaussianRendererProtocol.swift
│   └── Utils/
│       ├── PLYLoader.swift           # Point cloud loading
│       └── Scene.swift               # Scene utilities
└── RendererTypes/
    └── include/
        └── BridgingTypes.h           # Swift/Metal shared types
```

## TODO

- [ ] Release viewer app
- [ ] Fix numeric precision bugs
- [ ] Efficient stereo rendering
- [ ] Cleanup memory allocation
- [ ] Proper LOD and culling
- [ ] SPZ import support
- [ ] Async rendering (sort/render overlap)
- [ ] Migrate to Metal 4

## References

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Kerbl et al., SIGGRAPH 2023
- [Splatshop: Editing Gaussian Splats](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70214) - Depth-first sorting approach used in DepthFirstRenderer
- [FlashGS: Efficient 3D Gaussian Splatting](https://arxiv.org/abs/2408.07967) - Tile-based rendering optimizations

# GSM Renderer

> **Work in progress** - Not production ready.

A high-performance Gaussian Splatting renderer for Apple platforms, written in Swift and Metal.

## Features

- **Multiple rendering strategies:** Global, Local, DepthFirst, and Hardware (mesh shaders / instanced)
- **Spherical Harmonics:** Supports SH degrees 0-3 for view-dependent color
- **Stereo rendering:** Side-by-side and visionOS foveated targets

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
// Or: try HardwareRenderer(device: device, config: config)

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

## Stereo Rendering

All renderers implement `GaussianRenderer.renderStereo(...)` with a `StereoRenderTarget`:

- `.sideBySide(colorTexture:depthTexture:)`: left eye on the left half, right eye on the right half.
- `.foveated(drawable:configuration:)`: visionOS Compositor Services output (rasterization rate map supported).

## Renderers

### GlobalRenderer

Uses a global radix sort to order all tile-gaussian assignments by (tile, depth).

**Pipeline:**
1. **Project + Cull** - Transform 3D gaussians to 2D screen space, cull, compute conics and colors from SH
2. **Tile Assignment** - Two-pass: count tiles per gaussian, prefix sum, then write (tileId, gaussianId) tuples
3. **Generate Sort Keys** - Build 64-bit keys as (tileId << 32 | depth) for each assignment
4. **Radix Sort** - Sort all assignments globally by sort key using 8-pass radix sort
5. **Build Headers** - Scan sorted keys to find offset/count for each tile, build active tile list
6. **Render** - Each active tile blends its gaussians front-to-back using indirect dispatch

### LocalRenderer

Sorts gaussians independently per tile using threadgroup memory.

**Pipeline:**
1. **Project + Cull** - Transform 3D to 2D, compute conics/colors, compact visible gaussians
2. **Scatter** - Each gaussian atomically increments tile counters, writes depth key and index to tile slots
3. **Prefix Scan** - Compute tile offsets from counts, build active tile list
4. **Per-Tile Sort** - Bitonic sort within each tile's slot range (16-bit depth keys, threadgroup memory)
5. **Render** - Each active tile blends its sorted gaussians front-to-back using indirect dispatch


### DepthFirstRenderer

Sorts gaussians by depth first (globally), then by tile. This ensures correct back-to-front ordering is established before tile assignment, using a stable tile sort to preserve depth order within each tile.

**Pipeline:**
1. **Project + Cull** - Project 3D to 2D, compute conics/colors, depth keys, and tile touch counts per gaussian
2. **Depth Sort** - Radix sort all visible gaussians globally by depth (back-to-front)
3. **Apply Depth Order** - Reorder tile counts according to depth-sorted gaussian order
4. **Prefix Sum** - Compute instance offsets from ordered tile counts
5. **Instance Expansion** - Expand depth-sorted gaussians into per-tile instances (one instance per tile touched)
6. **Tile Sort** - Stable radix sort instances by tile ID (preserves depth order within each tile)
7. **Extract Tile Ranges** - Scan sorted tile IDs to find offset/count for each tile
8. **Render** - Each active tile blends its gaussians front-to-back, 2×2 pixels per thread

### HardwareRenderer

Single renderer that supports two draw backends:

- **Mesh shaders** (object/mesh stages) when supported.
- **Instanced** draw fallback (indexed indirect draw).

**Pipeline (high level):**
1. **Project + Cull** - Project gaussians for mono or stereo views (viewport-driven).
2. **Visibility Compaction** - Compact visible gaussians.
3. **Depth Sort** - Radix sort visible gaussians by depth.
4. **Reorder** - Reorder projected gaussians into sorted order.
5. **Draw** - Encode either instanced or mesh-shader draw.


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
│   │   └── Encoders/
│   ├── HardwareRenderer/
│   │   ├── HardwareRenderer.swift    # Hardware raster renderer (mesh/instanced)
│   │   ├── HardwareGaussianShaders.metal
│   │   └── Encoders/
│   ├── Shared/
│   │   └── GaussianRendererProtocol.swift
│   └── Utils/
│       ├── PLYLoader.swift           # Point cloud loading
│       └── Scene.swift               # Scene utilities
└── RendererTypes/
    └── include/
        └── BridgingTypes.h           # Swift/Metal shared types
```

## Formatting

- Swift: `swiftformat .` (repo includes `gsm-renderer/.swiftformat` with `--self insert`).
- Metal: use `clang-format` for `.metal` files.

## TODO

- [ ] Release viewer app
- [x] Efficient stereo rendering
- [ ] Cleanup memory allocation
- [ ] Proper LOD and culling
- [ ] SPZ import support
- [ ] Async rendering (sort/render overlap)
- [ ] Migrate to Metal 4

## References

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Kerbl et al., SIGGRAPH 2023
- [Splatshop: Editing Gaussian Splats](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70214) - Depth-first sorting approach used in DepthFirstRenderer
- [FlashGS: Efficient 3D Gaussian Splatting](https://arxiv.org/abs/2408.07967) - Tile-based rendering optimizations

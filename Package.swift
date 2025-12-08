// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Renderer",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "Renderer",
            targets: ["Renderer"]
        ),
    ],
    targets: [
        // C target exposing shared types for Swift/Metal interop
        .target(
            name: "RendererTypes",
            path: "Sources/RendererTypes",
            publicHeadersPath: "include"
        ),
        .target(
            name: "Renderer",
            dependencies: ["RendererTypes"],
            path: "Sources/Renderer",
            exclude: [
                // Exclude intermediate build artifacts
                "GlobalRenderer/GlobalShaders.air",
                "LocalRenderer/LocalShaders.air",
                "DepthFirstRenderer/DepthFirstShaders.air",
                // Metal source files (kept for reference, compiled separately)
                "GlobalRenderer/GlobalShaders.metal",
                "LocalRenderer/LocalShaders.metal",
                "DepthFirstRenderer/DepthFirstShaders.metal",
            ],
            resources: [
                // Pre-compiled Metal libraries (run ./compile_shaders.sh to rebuild)
                .copy("GlobalRenderer/GlobalShaders.metallib"),
                .copy("LocalRenderer/LocalShaders.metallib"),
                .copy("DepthFirstRenderer/DepthFirstShaders.metallib"),
            ]
        ),
        .testTarget(
            name: "RendererTests",
            dependencies: ["Renderer"]
        ),
    ]
)

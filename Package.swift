// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "GaussianMetalRenderer",
    platforms: [
        .macOS(.v15),
        .iOS(.v18)
    ],
    products: [
        .library(
            name: "GaussianMetalRenderer",
            type: .dynamic,
            targets: ["GaussianMetalRenderer"]
        ),
    ],
    dependencies: [],
    targets: [
        // C target exposing shared types for Swift/Metal interop
        .target(
            name: "GaussianMetalRendererTypes",
            path: "Sources/GaussianMetalRendererTypes",
            publicHeadersPath: "include"
        ),
        .target(
            name: "GaussianMetalRenderer",
            dependencies: ["GaussianMetalRendererTypes"],
            path: "Sources/GaussianMetalRenderer",
            resources: [
                .process("GaussianMetalRenderer.metallib"),
                .process("GaussianMetalRenderer.metal"),
                .process("LocalShaders.metallib"),
                .process("LocalShaders.metal")
            ]
        ),
        .testTarget(
            name: "GaussianMetalRendererTests",
            dependencies: ["GaussianMetalRenderer"]
        ),
    ]
)

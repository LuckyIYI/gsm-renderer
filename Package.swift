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
        .target(
            name: "GaussianMetalRenderer",
            path: "Sources",
            resources: [
                .process("GaussianMetalRenderer/GaussianMetalRenderer.metallib"),
                .process("GaussianMetalRenderer/GaussianMetalRenderer.metal"),
                .process("GaussianMetalRenderer/LocalSortShaders.metallib"),
                .process("GaussianMetalRenderer/LocalSortShaders.metal"),
                .process("GaussianMetalRenderer/TemporalShaders.metallib"),
                .process("GaussianMetalRenderer/TemporalShaders.metal")
            ]
        ),
        .testTarget(
            name: "GaussianMetalRendererTests",
            dependencies: ["GaussianMetalRenderer"]
        ),
    ]
)

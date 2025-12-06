// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "GaussianMetalRenderer",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
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
            name: "RendererTypes",
            path: "Sources/RendererTypes",
            publicHeadersPath: "include"
        ),
        .target(
            name: "GaussianMetalRenderer",
            dependencies: ["RendererTypes"],
            path: "Sources/GaussianMetalRenderer",
            exclude: [
                "GlobalShaders.air",
                "LocalShaders.air",
            ],
            resources: [
                .process("GlobalShaders.metallib"),
                .process("GlobalShaders.metal"),
                .process("LocalShaders.metallib"),
                .process("LocalShaders.metal"),
            ]
        ),
        .testTarget(
            name: "GaussianMetalRendererTests",
            dependencies: ["GaussianMetalRenderer"]
        ),
    ]
)

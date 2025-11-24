// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "GaussianMetalRenderer",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .library(
            name: "GaussianMetalRenderer",
            type: .dynamic,
            targets: ["GaussianMetalRenderer"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/weichsel/ZIPFoundation", from: "0.9.17"),
    ],
    targets: [
        .target(
            name: "GaussianMetalRenderer",
            path: "Sources",
            resources: [
                .process("GaussianMetalRenderer/GaussianMetalRenderer.metallib"),
                .process("GaussianMetalRenderer/GaussianMetalRenderer.metal")
            ]
        ),
        .testTarget(
            name: "GaussianMetalRendererTests",
            dependencies: ["GaussianMetalRenderer"]
        ),
    ]
)

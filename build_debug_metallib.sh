#!/bin/bash
set -e

echo "Compiling Metal shaders with debug info..."

SDK_PATH=$(xcrun --sdk macosx --show-sdk-path)

# Compile .metal to .air with debug flags
xcrun -sdk macosx metal \
  -frecord-sources -gline-tables-only \
  -c Sources/GaussianMetalRenderer/GaussianMetalRenderer.metal \
  -o /tmp/GaussianMetalRenderer.air

# Create .metallib from .air
xcrun -sdk macosx metallib \
  /tmp/GaussianMetalRenderer.air \
  -o Sources/GaussianMetalRenderer/GaussianMetalRenderer.metallib

echo "Success! Created Sources/GaussianMetalRenderer/GaussianMetalRenderer.metallib"


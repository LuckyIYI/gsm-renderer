#!/bin/bash
set -e

SDK="macosx"

# Include path for shared types
INCLUDE="-I Sources/RendererTypes/include"

# Global shaders (radix sort pipeline)
SRC="Sources/GaussianMetalRenderer/GlobalShaders.metal"
AIR="Sources/GaussianMetalRenderer/GlobalShaders.air"
LIB="Sources/GaussianMetalRenderer/GlobalShaders.metallib"

echo "Compiling Metal Shaders (Debug Mode)..."
xcrun -sdk $SDK metal \
  $INCLUDE \
  -frecord-sources -gline-tables-only \
  -ffast-math \
  -c "$SRC" \
  -o "$AIR"

echo "Linking Metal Library..."
xcrun -sdk $SDK metallib \
  "$AIR" \
  -o "$LIB"

echo "Done: $LIB"

# Local shaders (per-tile bitonic sort)
SRC2="Sources/GaussianMetalRenderer/LocalShaders.metal"
AIR2="Sources/GaussianMetalRenderer/LocalShaders.air"
LIB2="Sources/GaussianMetalRenderer/LocalShaders.metallib"

echo ""
echo "Compiling Local Shaders..."
xcrun -sdk $SDK metal \
  $INCLUDE \
  -frecord-sources -gline-tables-only \
  -ffast-math \
  -c "$SRC2" \
  -o "$AIR2"

echo "Linking Local Metal Library..."
xcrun -sdk $SDK metallib \
  "$AIR2" \
  -o "$LIB2"

echo "Done: $LIB2"

#!/bin/bash
set -e

SDK="macosx"

# Main shaders
SRC="Sources/GaussianMetalRenderer/GaussianMetalRenderer.metal"
AIR="Sources/GaussianMetalRenderer/GaussianMetalRenderer.air"
LIB="Sources/GaussianMetalRenderer/GaussianMetalRenderer.metallib"

echo "Compiling Metal Shaders (Debug Mode)..."
xcrun -sdk $SDK metal \
  -frecord-sources -gline-tables-only \
  -c "$SRC" \
  -o "$AIR"

echo "Linking Metal Library..."
xcrun -sdk $SDK metallib \
  "$AIR" \
  -o "$LIB"

echo "Done: $LIB"

# Tellusim shaders
SRC2="Sources/GaussianMetalRenderer/TellusimShaders.metal"
AIR2="Sources/GaussianMetalRenderer/TellusimShaders.air"
LIB2="Sources/GaussianMetalRenderer/TellusimShaders.metallib"

echo ""
echo "Compiling Tellusim Shaders..."
xcrun -sdk $SDK metal \
  -frecord-sources -gline-tables-only \
  -c "$SRC2" \
  -o "$AIR2"

echo "Linking Tellusim Metal Library..."
xcrun -sdk $SDK metallib \
  "$AIR2" \
  -o "$LIB2"

echo "Done: $LIB2"


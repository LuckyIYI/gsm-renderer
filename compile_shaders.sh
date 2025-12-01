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

# LocalSort shaders (per-tile bitonic sort)
SRC2="Sources/GaussianMetalRenderer/LocalSortShaders.metal"
AIR2="Sources/GaussianMetalRenderer/LocalSortShaders.air"
LIB2="Sources/GaussianMetalRenderer/LocalSortShaders.metallib"

echo ""
echo "Compiling LocalSort Shaders..."
xcrun -sdk $SDK metal \
  -frecord-sources -gline-tables-only \
  -c "$SRC2" \
  -o "$AIR2"

echo "Linking LocalSort Metal Library..."
xcrun -sdk $SDK metallib \
  "$AIR2" \
  -o "$LIB2"

echo "Done: $LIB2"

# Temporal shaders (standalone archive)
SRC3="Sources/GaussianMetalRenderer/TemporalShaders.metal"
AIR3="Sources/GaussianMetalRenderer/TemporalShaders.air"
LIB3="Sources/GaussianMetalRenderer/TemporalShaders.metallib"

echo ""
echo "Compiling Temporal Shaders..."
xcrun -sdk $SDK metal \
  -frecord-sources -gline-tables-only \
  -c "$SRC3" \
  -o "$AIR3"

echo "Linking Temporal Metal Library..."
xcrun -sdk $SDK metallib \
  "$AIR3" \
  -o "$LIB3"

echo "Done: $LIB3"


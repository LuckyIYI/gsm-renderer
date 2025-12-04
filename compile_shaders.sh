#!/bin/bash
set -e

SDK="macosx"

# Include path for shared types
INCLUDE="-I Sources/GaussianMetalRendererTypes/include"

# Main shaders
SRC="Sources/GaussianMetalRenderer/GaussianMetalRenderer.metal"
AIR="Sources/GaussianMetalRenderer/GaussianMetalRenderer.air"
LIB="Sources/GaussianMetalRenderer/GaussianMetalRenderer.metallib"

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

# LocalSort shaders (per-tile bitonic sort)
SRC2="Sources/GaussianMetalRenderer/LocalSortShaders.metal"
AIR2="Sources/GaussianMetalRenderer/LocalSortShaders.air"
LIB2="Sources/GaussianMetalRenderer/LocalSortShaders.metallib"

echo ""
echo "Compiling LocalSort Shaders..."
xcrun -sdk $SDK metal \
  $INCLUDE \
  -frecord-sources -gline-tables-only \
  -ffast-math \
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
  $INCLUDE \
  -frecord-sources -gline-tables-only \
  -ffast-math \
  -c "$SRC3" \
  -o "$AIR3"

echo "Linking Temporal Metal Library..."
xcrun -sdk $SDK metallib \
  "$AIR3" \
  -o "$LIB3"

echo "Done: $LIB3"

# ClusterCull shaders (cluster-level culling for Morton-sorted gaussians)
SRC4="Sources/GaussianMetalRenderer/ClusterCullShaders.metal"
AIR4="Sources/GaussianMetalRenderer/ClusterCullShaders.air"
LIB4="Sources/GaussianMetalRenderer/ClusterCullShaders.metallib"

echo ""
echo "Compiling ClusterCull Shaders..."
xcrun -sdk $SDK metal \
  $INCLUDE \
  -frecord-sources -gline-tables-only \
  -ffast-math \
  -c "$SRC4" \
  -o "$AIR4"

echo "Linking ClusterCull Metal Library..."
xcrun -sdk $SDK metallib \
  "$AIR4" \
  -o "$LIB4"

echo "Done: $LIB4"

# DepthFirst shaders (FastGS/Splatshop-style two-phase sort)
SRC5="Sources/GaussianMetalRenderer/DepthFirstShaders.metal"
AIR5="Sources/GaussianMetalRenderer/DepthFirstShaders.air"
LIB5="Sources/GaussianMetalRenderer/DepthFirstShaders.metallib"

echo ""
echo "Compiling DepthFirst Shaders..."
xcrun -sdk $SDK metal \
  $INCLUDE \
  -frecord-sources -gline-tables-only \
  -ffast-math \
  -c "$SRC5" \
  -o "$AIR5"

echo "Linking DepthFirst Metal Library..."
xcrun -sdk $SDK metallib \
  "$AIR5" \
  -o "$LIB5"

echo "Done: $LIB5"


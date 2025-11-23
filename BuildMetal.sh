#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
METAL_BIN=$(xcrun --sdk macosx --find metal)
METALLIB_BIN=$(xcrun --sdk macosx --find metallib)
METALLIB="Sources/GaussianMetalRenderer/GaussianMetalRenderer.metallib"
SRC="Sources/GaussianMetalRenderer/GaussianMetalRenderer.metal"
TMP_AIR="tmp.$$.air"
"$METAL_BIN" -c "$SRC" -o "$TMP_AIR"
"$METALLIB_BIN" "$TMP_AIR" -o "$METALLIB"
rm "$TMP_AIR"

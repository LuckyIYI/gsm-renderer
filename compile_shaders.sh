#!/bin/bash
set -e

# Default to macOS, can override with: ./compile_shaders.sh ios
PLATFORM="${1:-macos}"

case "$PLATFORM" in
    macos)
        SDK="macosx"
        MIN_VERSION="-mmacosx-version-min=14.0"
        ;;
    ios)
        SDK="iphoneos"
        MIN_VERSION="-mios-version-min=17.0"
        ;;
    ios-sim)
        SDK="iphonesimulator"
        MIN_VERSION="-mios-simulator-version-min=17.0"
        ;;
    *)
        echo "Usage: $0 [macos|ios|ios-sim]"
        exit 1
        ;;
esac

RENDERER_DIR="Sources/Renderer"
INCLUDE="-I Sources/RendererTypes/include -I $RENDERER_DIR/Shared"

compile_shader() {
    local NAME=$1
    local SUBDIR=$2
    local SRC="$RENDERER_DIR/$SUBDIR/${NAME}.metal"
    local AIR="$RENDERER_DIR/$SUBDIR/${NAME}.air"
    local LIB="$RENDERER_DIR/$SUBDIR/${NAME}.metallib"

    echo "[$PLATFORM] Compiling $NAME.metal..."
    xcrun -sdk "$SDK" metal \
        $INCLUDE \
        $MIN_VERSION \
        -frecord-sources \
        -gline-tables-only \
        -ffast-math \
        -c "$SRC" \
        -o "$AIR"

    echo "[$PLATFORM] Linking $NAME.metallib..."
    xcrun -sdk "$SDK" metallib \
        "$AIR" \
        -o "$LIB"

    echo "[$PLATFORM] Done: $LIB"
}

compile_shader "GlobalShaders" "GlobalRenderer"
compile_shader "LocalShaders" "LocalRenderer"
compile_shader "DepthFirstShaders" "DepthFirstRenderer"

echo ""
echo "Metal shaders compiled successfully for $PLATFORM"

#!/usr/bin/env bash
# CASAS – Build C++ DA extension
# ================================
# Place this script alongside:
#   casas_da.hpp   (merged header)
#   bindings.cpp   (pybind11 glue)
#   CMakeLists.txt
#
# Run from that directory:  bash build.sh
# The resulting casas_cpp*.so is copied one level up (project root).

set -e
echo "=== CASAS C++ Build ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

# ── Dependency check ──────────────────────────────────────────
check_deps() {
    if ! python3 -c "import pybind11" 2>/dev/null; then
        echo "ERROR: pybind11 not found."
        echo "  Fix: pip install pybind11"
        exit 1
    fi
}

# ── Method 1: CMake (preferred) ───────────────────────────────
build_cmake() {
    echo "→ Building with CMake + pybind11"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    cmake "$SCRIPT_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"

    cmake --build . --parallel "$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"

    # Copy .so one level up to project root
    EXT="$(python3-config --extension-suffix 2>/dev/null || echo '.so')"
    find "$BUILD_DIR" -name "casas_cpp*$EXT" -exec cp {} "$PROJECT_ROOT/" \;

    echo "✅ CMake build complete → $PROJECT_ROOT/casas_cpp$EXT"
}

# ── Method 2: Direct c++ (fallback) ───────────────────────────
build_direct() {
    echo "→ Building with direct c++ command"
    cd "$PROJECT_ROOT"

    PY_INCLUDES="$(python3 -m pybind11 --includes)"
    PY_EXT="$(python3-config --extension-suffix)"

    c++ -std=c++17 -O3 -march=native -ffast-math \
        -shared -fPIC \
        $PY_INCLUDES \
        -I "$SCRIPT_DIR" \
        "$SCRIPT_DIR/bindings.cpp" \
        -o "casas_cpp${PY_EXT}"

    echo "✅ Direct build complete → casas_cpp${PY_EXT}"
}

# ── Main ──────────────────────────────────────────────────────
check_deps

if command -v cmake &>/dev/null; then
    build_cmake
elif command -v c++ &>/dev/null; then
    build_direct
else
    echo "ERROR: Neither cmake nor c++ compiler found."
    echo "  On Ubuntu/Debian: sudo apt install build-essential cmake"
    echo "  On macOS:         xcode-select --install && brew install cmake"
    exit 1
fi
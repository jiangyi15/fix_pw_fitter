#!/usr/bin/env bash
# Build portable ampfit AppImage: Python 3.10 + CUDA 12 + nvcc + all deps.
# Usage: ./build.sh [--no-appimage]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_APPIMAGE="$SCRIPT_DIR/python3.10.8-cp310-cp310-manylinux2014_x86_64.AppImage"
APPIMAGETOOL="$SCRIPT_DIR/appimagetool-x86_64.AppImage"
APP_DIR="$SCRIPT_DIR/AppDir"
OUTPUT_APPIMAGE="$SCRIPT_DIR/ampfit-python3.10-cuda12.AppImage"
CONDA="/home/jiangy/miniconda3"
CUDA12_ENV="$CONDA/envs/code2"  # has CUDA 12.8 w/ nvvm

step() { echo "=== $1 ==="; }

# ── 1. Extract base Python 3.10 AppImage ──────────────────────
step "Extract base Python 3.10 AppImage"
rm -rf "$APP_DIR"
cd /tmp
"$BASE_APPIMAGE" --appimage-extract > /dev/null 2>&1
mv /tmp/squashfs-root "$APP_DIR"

# ── 2. Install Python dependencies ─────────────────────────────
step "Install Python dependencies"
"$APP_DIR/AppRun" -m ensurepip --upgrade
PIP="$APP_DIR/AppRun -m pip install --no-input"
$PIP --upgrade pip wheel setuptools
$PIP numpy scipy pyyaml matplotlib cffi sympy
$PIP onnx onnxruntime pytest
rm -rf "$APP_DIR/root/.cache/pip"

# ── 3. Bundle CUDA 12 runtime libraries ───────────────────────
step "Bundle CUDA 12 runtime"
CUDA_LIB="$APP_DIR/usr/lib/cuda"
mkdir -p "$CUDA_LIB"
for lib in libcudart.so libcublas.so libcublasLt.so libcufft.so libcurand.so; do
    find "$CONDA/lib" -maxdepth 1 -name "${lib}*" -not -name "*.a" 2>/dev/null | while read f; do
        cp -aL "$f" "$CUDA_LIB/" 2>/dev/null || true
    done
done
# Strip
find "$CUDA_LIB" -name "*.so*" -exec strip --strip-unneeded {} \; 2>/dev/null || true

# ── 4. Bundle CUDA 12 toolkit (nvcc + headers + nvvm) ─────────
step "Bundle CUDA 12 toolkit"
CUDA_TK="$APP_DIR/usr/local/cuda"
mkdir -p "$CUDA_TK/bin" "$CUDA_TK/lib64" "$CUDA_TK/include" "$CUDA_TK/nvvm/lib64" "$CUDA_TK/nvvm/libdevice"

# nvcc binary (from CUDA 12 conda env)
cp -aL "$CUDA12_ENV/bin/nvcc" "$CUDA_TK/bin/"

# nvvm (from CUDA 12 env)
cp -aL "$CUDA12_ENV/nvvm/lib64/"libnvvm* "$CUDA_TK/nvvm/lib64/" 2>/dev/null || true
# libdevice (from CUDA 12 env)
cp -aL "$CUDA12_ENV/nvvm/libdevice/"* "$CUDA_TK/nvvm/libdevice/" 2>/dev/null || true

# CUDA headers
if [ -d "$CUDA12_ENV/targets/x86_64-linux/include" ]; then
    cp -r "$CUDA12_ENV/targets/x86_64-linux/include/"* "$CUDA_TK/include/"
fi

# Compiler support libs (libnvJitLink)
find "$CUDA12_ENV" -name "libnvJitLink*" -not -name "*.a" 2>/dev/null | while read f; do
    cp -aL "$f" "$CUDA_TK/lib64/" 2>/dev/null || true
done

# ── 5. Pre-compile CUDA kernels ───────────────────────────────
step "Pre-compile CUDA kernels"
export PATH="$CUDA_TK/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_TK/lib64:$CUDA_TK/nvvm/lib64:$CUDA_LIB:${LD_LIBRARY_PATH:-}"
TMP_SRC="/tmp/ampfit_build_$$"
mkdir -p "$TMP_SRC"
cp -r "$PROJECT_DIR/src/ampfit" "$TMP_SRC/"
"$APP_DIR/AppRun" -c "
import sys; sys.path.insert(0, '$TMP_SRC')
from ampfit.cuda.build import set_arch, build
set_arch('sm_70,sm_75,sm_86,sm_89')
build()
" 2>&1 | grep -v '^$'
mkdir -p "$APP_DIR/usr/lib/ampfit/cuda"
cp "$TMP_SRC/ampfit/cuda/"libcuda_kernels_*.so "$APP_DIR/usr/lib/ampfit/cuda/" 2>/dev/null || true
rm -rf "$TMP_SRC"

# ── 6. AppRun entry point ─────────────────────────────────────
step "Create AppRun"
cat > "$APP_DIR/AppRun" << 'APPRUN'
#!/bin/bash
SELF="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
export LD_LIBRARY_PATH="$SELF/usr/lib/cuda:$SELF/usr/local/cuda/lib64:$SELF/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$SELF/usr/local/cuda/bin:${PATH:-}"
export CUDA_PATH="$SELF/usr/local/cuda"
export CUDA_HOME="$CUDA_PATH"
export AMPFIT_CUDA_DIR="$SELF/usr/lib/ampfit/cuda"
PYTHON="$SELF/opt/python3.10.8/bin/python3.10"
[ -f "$PYTHON" ] || PYTHON="$SELF/usr/bin/python3"
exec "$PYTHON" "$@"
APPRUN
chmod +x "$APP_DIR/AppRun"

# ── 7. Desktop file ───────────────────────────────────────────
cat > "$APP_DIR/ampfit.desktop" << EOF
[Desktop Entry]
Name=ampfit (Python 3.10 + CUDA 12)
Exec=AppRun
Icon=ampfit
Terminal=true
Type=Application
Categories=Science;Physics;
EOF

# ── 8. Smoke tests ────────────────────────────────────────────
step "Smoke tests"
echo "  Python: $("$APP_DIR/AppRun" --version)"
echo "  numpy: $("$APP_DIR/AppRun" -c "import numpy; print(numpy.__version__)")"
echo "  scipy: $("$APP_DIR/AppRun" -c "import scipy; print(scipy.__version__)")"
echo -n "  cudart: "
"$APP_DIR/AppRun" -c "import ctypes; ctypes.CDLL('libcudart.so'); print('OK')" 2>&1
echo -n "  cublas: "
"$APP_DIR/AppRun" -c "import ctypes; ctypes.CDLL('libcublas.so'); print('OK')" 2>&1
echo -n "  nvcc: "
"$APP_DIR/AppRun" -c "
import subprocess, os
os.environ['PATH'] = '$CUDA_TK/bin:' + os.environ.get('PATH', '')
r = subprocess.run(['nvcc','--version'], capture_output=True, text=True)
print([l for l in r.stdout.split(chr(10)) if 'release' in l][0].strip())
" 2>&1

# ── 9. Create AppImage ────────────────────────────────────────
if [ "${1:-}" != "--no-appimage" ] && [ -f "$APPIMAGETOOL" ]; then
    step "Create AppImage"
    chmod +x "$APPIMAGETOOL"
    ARCH=x86_64 "$APPIMAGETOOL" "$APP_DIR" "$OUTPUT_APPIMAGE"
    ls -lh "$OUTPUT_APPIMAGE"
else
    echo "AppDir ready at: $APP_DIR"
    echo "Skip AppImage packaging."
fi

echo ""
echo "Done."
echo "Usage: PYTHONPATH=/path/to/ampfit/src $OUTPUT_APPIMAGE your_script.py"

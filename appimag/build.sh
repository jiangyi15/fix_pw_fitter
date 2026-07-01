#!/usr/bin/env bash
# Build portable ampfit AppImage: Python 3.10 + CUDA 12 runtime + nvcc + deps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_APPIMAGE="$SCRIPT_DIR/python3.10.8-cp310-cp310-manylinux2014_x86_64.AppImage"
APPIMAGETOOL="$SCRIPT_DIR/appimagetool-x86_64.AppImage"
APP_DIR="$SCRIPT_DIR/AppDir"
OUTPUT="$SCRIPT_DIR/ampfit-python3.10-cuda12.AppImage"
CONDA="/home/jiangy/miniconda3"
CUDA12_ENV="$CONDA/envs/code2"  # has full CUDA toolkit with nvcc

step() { echo "=== $1 ==="; }

rm -rf "$APP_DIR"

# 1. Extract base Python AppImage
step "Extract base Python 3.10"
cd /tmp && "$BASE_APPIMAGE" --appimage-extract > /dev/null 2>&1
mv /tmp/squashfs-root "$APP_DIR"

# 2. Install Python dependencies (Tsinghua mirror for speed)
step "Install Python deps"
"$APP_DIR/AppRun" -m ensurepip --upgrade > /dev/null 2>&1
PIP="$APP_DIR/AppRun -m pip install --no-input -i https://pypi.tuna.tsinghua.edu.cn/simple"
$PIP --upgrade pip wheel setuptools > /dev/null 2>&1
$PIP numpy scipy pyyaml matplotlib cffi sympy onnx onnxruntime pytest > /dev/null 2>&1
rm -rf "$APP_DIR/root/.cache/pip"

# 3. Bundle CUDA 12 runtime (from base conda — libcudart.so.12)
step "Bundle CUDA 12 runtime"
CUDA_LIB="$APP_DIR/usr/lib/cuda"
mkdir -p "$CUDA_LIB"
for lib in libcudart.so libcublas.so libcublasLt.so libcufft.so libcurand.so; do
    find "$CONDA/lib" -maxdepth 1 -name "${lib}*" -not -name "*.a" 2>/dev/null | while read f; do
        cp -aL "$f" "$CUDA_LIB/"
    done
done
find "$CUDA_LIB" -name "*.so*" -exec strip --strip-unneeded {} \; 2>/dev/null || true

# 4. Bundle nvcc + headers + compilation tools
step "Bundle nvcc + CUDA toolkit"
CUDA_TK="$APP_DIR/usr/local/cuda"
mkdir -p "$CUDA_TK/bin" "$CUDA_TK/nvvm/bin" "$CUDA_TK/nvvm/lib64"
mkdir -p "$CUDA_TK/nvvm/libdevice" "$CUDA_TK/lib64"

# nvcc + its profile (needed for include path resolution)
cp -aL "$CUDA12_ENV/bin/nvcc" "$CUDA12_ENV/bin/nvcc.profile" "$CUDA_TK/bin/"

# Headers (targets structure expected by nvcc)
cp -rL "$CUDA12_ENV/targets" "$CUDA_TK/"

# nvvm (jit compiler)
cp -rL "$CUDA12_ENV/nvvm/lib64/"libnvvm* "$CUDA_TK/nvvm/lib64/"
cp -rL "$CUDA12_ENV/nvvm/libdevice/"* "$CUDA_TK/nvvm/libdevice/"
cp -rL "$CUDA12_ENV/nvvm/bin/"* "$CUDA_TK/nvvm/bin/"

# Companion tools (ptxas, crt, nvlink, bin2c)
for tool in ptxas crt nvlink bin2c; do
    cp -aL "$CUDA12_ENV/bin/$tool" "$CUDA_TK/bin/" 2>/dev/null || true
done

# libnvJitLink
find "$CUDA12_ENV" -name "libnvJitLink*" -not -name "*.a" 2>/dev/null | while read f; do
    cp -aL "$f" "$CUDA_TK/lib64/"
done

# 5. Pre-compile CUDA kernels (sm_70, sm_75, sm_86, sm_89)
step "Pre-compile CUDA kernels"
export PATH="$CUDA_TK/bin:$CUDA_TK/nvvm/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_TK/lib64:$CUDA_TK/nvvm/lib64:$CUDA_LIB:${LD_LIBRARY_PATH:-}"
TMP_SRC="/tmp/ampfit_build_$$"
mkdir -p "$TMP_SRC"
cp -r "$PROJECT_DIR/src/ampfit" "$TMP_SRC/"
"$APP_DIR/AppRun" -c "
import sys; sys.path.insert(0, '$TMP_SRC')
from ampfit.cuda.build import set_arch, build
set_arch('sm_70,sm_75,sm_86,sm_89')
print('Building kernels...', end='')
ok = build()
print(' OK' if ok else ' FAILED')
" 2>&1
mkdir -p "$APP_DIR/usr/lib/ampfit/cuda"
cp "$TMP_SRC/ampfit/cuda/"libcuda_kernels_*.so "$APP_DIR/usr/lib/ampfit/cuda/" 2>/dev/null || true
rm -rf "$TMP_SRC"

# 6. AppRun entry point
step "Create AppRun"
cat > "$APP_DIR/AppRun" << 'APPRUN'
#!/bin/bash
SELF="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$SELF/usr/lib/cuda:$SELF/usr/local/cuda/lib64:$SELF/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$SELF/usr/local/cuda/bin:$SELF/usr/local/cuda/nvvm/bin:${PATH:-}"
export CUDA_PATH="$SELF/usr/local/cuda"
export CUDA_HOME="$CUDA_PATH"
export AMPFIT_CUDA_DIR="$SELF/usr/lib/ampfit/cuda"
PYTHON="$SELF/opt/python3.10/bin/python3.10"
exec "$PYTHON" "$@"
APPRUN
chmod +x "$APP_DIR/AppRun"

# 7. Desktop file
cat > "$APP_DIR/ampfit.desktop" << EOF
[Desktop Entry]
Name=ampfit (Python 3.10 + CUDA 12)
Exec=AppRun
Icon=ampfit
Terminal=true
Type=Application
Categories=Science;Physics;
EOF

# 8. Smoke tests
step "Smoke tests"
echo -n "  Python: "; $APP_DIR/AppRun --version
echo -n "  numpy: "; $APP_DIR/AppRun -c "import numpy; print(numpy.__version__)"
echo -n "  cudart: "
$APP_DIR/AppRun -c "import ctypes; ctypes.CDLL('libcudart.so')
v=ctypes.c_int(); ctypes.CDLL('libcudart.so').cudaRuntimeGetVersion(ctypes.byref(v))
print(f'CUDA {v.value//1000}.{(v.value%1000)//10}')" 2>&1
echo -n "  compile: "
echo '#include <cuda_runtime.h>
__global__ void k(){}' > /tmp/_test.cu
$APP_DIR/AppRun -c "
import subprocess, os
os.environ.update(PATH='/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:'+os.environ.get('PATH',''),
    LD_LIBRARY_PATH='/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64:/usr/lib/cuda')
r=subprocess.run(['nvcc','-c','/tmp/_test.cu','-o','/tmp/_test.o'])
print('OK' if r.returncode==0 else f'FAIL({r.returncode})')
" 2>&1
rm -f /tmp/_test.cu /tmp/_test.o 2>/dev/null

# 9. Package AppImage
if [ -f "$APPIMAGETOOL" ]; then
    step "Create AppImage"
    chmod +x "$APPIMAGETOOL"
    ARCH=x86_64 "$APPIMAGETOOL" "$APP_DIR" "$OUTPUT"
    ls -lh "$OUTPUT"
else
    echo "AppDir ready at $APP_DIR (no appimagetool)"
fi
echo "Done."

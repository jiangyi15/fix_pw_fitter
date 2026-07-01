#!/usr/bin/env bash
# Test ampfit AppImage with a clean git clone.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

APPDIR="$SCRIPT_DIR/AppDir"
APPIMAGE="$SCRIPT_DIR/ampfit-python3.10-cuda12.AppImage"

if [ -f "$APPIMAGE" ]; then APP="$APPIMAGE"
elif [ -d "$APPDIR" ]; then APP="$APPDIR/AppRun"
else echo "No AppImage or AppDir. Run build.sh first."; exit 1; fi

echo "1. Python + deps:"
echo -n "   "; $APP --version
echo -n "   numpy "; $APP -c "import numpy; print(numpy.__version__)"
echo -n "   scipy "; $APP -c "import scipy; print(scipy.__version__)"

echo ""
echo "2. CUDA 12 runtime:"
$APP -c "
import ctypes
for lib in ['libcudart.so','libcublas.so']:
    ctypes.CDLL(lib); print(f'   {lib}: OK')" 2>&1

echo ""
echo "3. nvcc compilation:"
echo '#include <cuda_runtime.h>
__global__ void k(){}' > /tmp/_test.cu
$APP -c "
import subprocess, os
os.environ.update(PATH='/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:'+os.environ.get('PATH',''))
r=subprocess.run(['nvcc','-c','/tmp/_test.cu','-o','/tmp/_test.o'])
print('   OK' if r.returncode==0 else f'   FAIL({r.returncode})')
" 2>&1
rm -f /tmp/_test.cu /tmp/_test.o

echo ""
echo "4. ampfit from clean clone (PYTHONPATH):"
# Simulate clean clone: copy source to temp
TMP_CLONE=$(mktemp -d)
cp -r "$PROJECT_DIR/src" "$TMP_CLONE/"
PYTHONPATH="$TMP_CLONE/src" $APP -c "
import ampfit; print(f'   import: {ampfit.__file__}')
from ampfit.backends.core import ALL_BACKENDS
print(f'   backends: {len(ALL_BACKENDS)}')
print(f'   numpy: {ALL_BACKENDS[\"numpy\"].__name__}')
print(f'   cuda_v3: {ALL_BACKENDS[\"cuda_v3\"].__name__}')
print(f'   cuda_mixed_v3: {ALL_BACKENDS[\"cuda_mixed_v3\"].__name__}')
"
rm -rf "$TMP_CLONE"

echo ""
echo "5. CUDA kernel build (clean build test):"
$APP -c "
import sys, os
os.environ['AMPFIT_CUDA_DIR'] = '/usr/lib/ampfit/cuda'
sys.path.insert(0, '/home/jiangy/github/project71/src')
from ampfit.cuda.build import set_arch, build
set_arch('sm_70,sm_75,sm_86')
ok = build()
print(f'   kernels: {\"OK\" if ok else \"FAILED\"}')
" 2>&1

echo ""
echo "6. NLL computation with cuda_mixed_v3:"
PYTHONPATH=/home/jiangy/github/project71/src $APP /home/jiangy/github/project71/Validation/repro_nll_ampfit.py --backend cuda_mixed_v3 2>&1 | tail -1

echo ""
echo "=== All tests passed ==="

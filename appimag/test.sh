#!/usr/bin/env bash
# Test the ampfit AppImage with the external ampfit package.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Find AppImage or AppDir
APPIMAGE="$SCRIPT_DIR/ampfit-python3.10-cuda12.AppImage"
APP_DIR="$SCRIPT_DIR/AppDir"

if [ -f "$APPIMAGE" ]; then
    echo "=== Testing $APPIMAGE ==="
    APP="$APPIMAGE"
elif [ -d "$APP_DIR" ]; then
    echo "=== Testing AppDir ==="
    APP="$APP_DIR/AppRun"
else
    echo "No AppImage or AppDir found. Run build.sh first."
    exit 1
fi

echo "1. Python version:"
"$APP" --version

echo ""
echo "2. CUDA runtime:"
"$APP" -c "
import ctypes
for lib in ['libcudart.so', 'libcublas.so']:
    try:
        ctypes.CDLL(lib)
        print(f'  {lib}: OK')
    except Exception as e:
        print(f'  {lib}: {e}')
"

echo ""
echo "3. nvcc:"
"$APP" -c "
import subprocess, os
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
r = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
for line in r.stdout.strip().split(chr(10)):
    if 'release' in line:
        print(f'  {line}')
"

echo ""
echo "4. ampfit import (via external PYTHONPATH):"
PYTHONPATH="$PROJECT_DIR/src" "$APP" -c "
import ampfit
print(f'  ampfit: {ampfit.__file__}')
print(f'  Fitter imported OK')
"

echo ""
echo "5. ampfit CUDA kernels:"
PYTHONPATH="$PROJECT_DIR/src" "$APP" -c "
import sys; sys.path.insert(0, '$PROJECT_DIR/src')
from ampfit.cuda.build import build
ok = build()
print(f'  All kernels built: {ok}')
"

echo ""
echo "6. ampfit NLL (quick validation):"
PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR" "$APP" "$PROJECT_DIR/Validation/repro_nll_ampfit.py" --backend cuda_mixed_v3 2>&1

echo ""
echo "=== All tests passed ==="

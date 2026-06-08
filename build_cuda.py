#!/usr/bin/env python3
"""
Build script for CUDA kernel with CFFI.

Usage:
    python build_cuda.py

This compiles the CUDA kernels and creates a shared library.
"""

import os
import subprocess
import sys

def find_cuda():
    """Find CUDA installation"""
    cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')

    if not cuda_path:
        # Try common locations
        common_paths = [
            '/usr/local/cuda',
            '/usr/local/cuda-11.0',
            '/usr/local/cuda-12.0',
            '/opt/cuda',
        ]
        for path in common_paths:
            if os.path.exists(path):
                cuda_path = path
                break

    if cuda_path:
        nvcc = os.path.join(cuda_path, 'bin', 'nvcc')
        if os.path.exists(nvcc):
            return cuda_path, nvcc

    return None, None

def build_cuda_library():
    """Build CUDA shared library"""
    cuda_path, nvcc = find_cuda()

    if not nvcc:
        print("ERROR: CUDA not found!")
        print("\nPlease install CUDA Toolkit or set CUDA_PATH environment variable:")
        print("  export CUDA_PATH=/usr/local/cuda")
        print("\nOr install CUDA from: https://developer.nvidia.com/cuda-downloads")
        return False

    print(f"Found CUDA at: {cuda_path}")
    print(f"Using nvcc: {nvcc}")

    # Check nvcc version
    try:
        result = subprocess.run([nvcc, '--version'], capture_output=True, text=True)
        print(f"NVCC version:\n{result.stdout}")
    except Exception as e:
        print(f"Error running nvcc: {e}")
        return False

    # Compile CUDA code
    output_lib = "libcuda_kernels.so"

    cmd = [
        nvcc,
        '-shared',
        '-Xcompiler', '-fPIC',
        '-o', output_lib,
        'cuda_kernels_cffi.cu',
        '-lcudart'
    ]

    print(f"\nCompiling: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}")
            return False

        print(f"✓ Successfully built {output_lib}")
        return True

    except Exception as e:
        print(f"Compilation error: {e}")
        return False

if __name__ == "__main__":
    print("="*70)
    print("CUDA KERNEL BUILD SCRIPT")
    print("="*70)

    success = build_cuda_library()

    if success:
        print("\n" + "="*70)
        print("✓ Build complete!")
        print("="*70)
        print("\nShared library created: libcuda_kernels.so")
        print("You can now use the CUDA kernel from Python via CFFI.")
    else:
        print("\n" + "="*70)
        print("✗ Build failed")
        print("="*70)
        sys.exit(1)

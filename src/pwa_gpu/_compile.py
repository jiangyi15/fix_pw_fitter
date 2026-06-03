"""
CUDA compilation — compiles the .cu kernel into a shared library on first use.

The shared library is cached alongside the .cu file and rebuilt when the
source is modified.
"""

import os
import subprocess


def compile_cuda():
    """Compile CUDA kernel to shared library. Returns path to .so file."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    cu_file = os.path.join(pkg_dir, '_kernels.cu')
    so_file = os.path.join(pkg_dir, 'libpwa_gpu.so')

    if os.path.exists(so_file):
        if os.path.getmtime(cu_file) <= os.path.getmtime(so_file):
            return so_file

    cmd = [
        'nvcc', '-shared', '-Xcompiler', '-fPIC',
        '-o', so_file, cu_file,
        '-lcudart', '--ptxas-options=-v'
    ]

    print("Compiling CUDA code...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}")
        raise RuntimeError("CUDA compilation failed")

    print(f"Compiled to {so_file}")
    return so_file

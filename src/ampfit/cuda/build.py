#!/usr/bin/env python3
"""Build the CUDA shared library for ampfit.

Usage:
    python -m ampfit.cuda.build

This compiles kernels.cu and creates libcuda_kernels.so in the same directory.
"""
import os
import subprocess
import sys


def find_cuda():
    import shutil
    nvcc_path = shutil.which('nvcc')
    if nvcc_path:
        return os.path.dirname(os.path.dirname(nvcc_path)), nvcc_path

    cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
    if cuda_path:
        nvcc = os.path.join(cuda_path, 'bin', 'nvcc')
        if os.path.exists(nvcc):
            return cuda_path, nvcc

    common = ['/usr/local/cuda', '/usr/local/cuda-13.2', '/usr/local/cuda-12.0',
              '/usr/local/cuda-11.0', '/opt/cuda']
    for path in common:
        nvcc = os.path.join(path, 'bin', 'nvcc')
        if os.path.exists(nvcc):
            return path, nvcc
    return None, None


VARIANTS = [
    ("kernels.cu", "libcuda_kernels.so"),
    ("kernels_v2.cu", "libcuda_kernels_v2.so"),
    ("kernels_v3.cu", "libcuda_kernels_v3.so"),
]


def build():
    cuda_path, nvcc = find_cuda()
    if not nvcc:
        print("ERROR: CUDA not found. Install CUDA Toolkit or set CUDA_PATH.")
        return False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_ok = True

    for src_name, lib_name in VARIANTS:
        src_file = os.path.join(script_dir, src_name)
        out_file = os.path.join(script_dir, lib_name)

        cmd = [nvcc, '-shared', '-Xcompiler', '-fPIC',
               '-o', out_file, src_file, '-lcudart']

        print(f"Building: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAILED:\n{result.stderr}")
            all_ok = False
        else:
            print(f"  ✓ Built {out_file}")

    return all_ok


if __name__ == "__main__":
    sys.exit(0 if build() else 1)

#!/usr/bin/env python3
"""Build the CUDA shared libraries for ampfit.

Usage:
    python -m ampfit.cuda.build

Auto-detects nvcc and required compiler flags.  Builds all 4 kernel
variants: v2/v3 × f64/f32.
"""
import os
import subprocess
import sys


def find_nvcc():
    import shutil
    nvcc_path = shutil.which('nvcc')
    if nvcc_path:
        return nvcc_path

    cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
    if cuda_path:
        nvcc = os.path.join(cuda_path, 'bin', 'nvcc')
        if os.path.exists(nvcc):
            return nvcc

    common = ['/usr/local/cuda', '/usr/local/cuda-13.2', '/usr/local/cuda-12.0',
              '/usr/local/cuda-11.0', '/opt/cuda']
    for path in common:
        nvcc = os.path.join(path, 'bin', 'nvcc')
        if os.path.exists(nvcc):
            return nvcc
    return None


def detect_gcc():
    """Find a gcc compatible with this nvcc."""
    import shutil
    # Try specific versioned gcc first (nvcc is picky about minor versions)
    for ver in ['-14', '-13', '-12', '-11']:
        p = shutil.which(f'gcc{ver}')
        if p:
            return p
    p = shutil.which('gcc')
    if p:
        return p
    return None


VARIANTS = [
    ("kernels_v2.cu",     "libcuda_kernels_v2.so"),
    ("kernels_v2_f32.cu", "libcuda_kernels_v2_f32.so"),
    ("kernels_v3.cu",     "libcuda_kernels_v3.so"),
    ("kernels_v3_f32.cu", "libcuda_kernels_v3_f32.so"),
]


def _arch_flag(nvcc):
    """Detect GPU architecture for atomicAdd(double) support."""
    import subprocess, re
    r = subprocess.run([nvcc, '--version'], capture_output=True, text=True)
    m = re.search(r'release (\d+\.\d+)', r.stdout)
    if m and float(m.group(1)) >= 11:
        return '-arch=sm_86'  # Ampere+ (RTX 30xx)
    return ''


def build():
    nvcc = find_nvcc()
    if not nvcc:
        print("ERROR: CUDA not found. Install CUDA Toolkit or set CUDA_PATH.")
        return False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gcc = detect_gcc()

    # Build flags: try plain first, fall back to compat flags
    base = [nvcc, '-shared', '-Xcompiler', '-fPIC', '-lcudart', '-lm', '-O2']
    arch = _arch_flag(nvcc)
    if arch:
        base.insert(1, arch)

    # Probe one file to detect required extra flags
    probe_file = os.path.join(script_dir, VARIANTS[0][0])
    probe_out = os.path.join(script_dir, '_probe.so')

    extra = []
    r = subprocess.run(base + ['-o', probe_out, probe_file],
                       capture_output=True, text=True)
    if r.returncode != 0:
        err = r.stderr
        if 'unsupported' in err:
            extra.append('-allow-unsupported-compiler')
        if gcc:
            extra.extend(['-ccbin', gcc])
        r2 = subprocess.run(base + extra + ['-o', probe_out, probe_file],
                            capture_output=True, text=True)
        if r2.returncode != 0:
            print(f"ERROR: probe compiler:\n{r2.stderr[:500]}")
            if os.path.exists(probe_out):
                os.remove(probe_out)
            return False
    if os.path.exists(probe_out):
        os.remove(probe_out)

    print(f"Using nvcc: {nvcc}")
    if extra:
        print(f"Extra flags: {' '.join(extra)}")

    all_ok = True
    for src_name, lib_name in VARIANTS:
        src_file = os.path.join(script_dir, src_name)
        out_file = os.path.join(script_dir, lib_name)
        cmd = base + extra + ['-o', out_file, src_file]
        print(f"  {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED:\n{r.stderr[:500]}")
            all_ok = False
        else:
            print(f"  ✓ Built {os.path.basename(out_file)}")

    return all_ok


if __name__ == "__main__":
    sys.exit(0 if build() else 1)

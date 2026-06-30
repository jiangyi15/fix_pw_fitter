#!/usr/bin/env python3
"""Build the CUDA shared libraries for ampfit.

Usage:
    python -m ampfit.cuda.build          # force rebuild all
    from ampfit.cuda.build import ensure; ensure()  # auto-update on import

Auto-detects nvcc and required compiler flags.  Builds all 4 kernel
variants: v2/v3 × f64/f32.

Each .so has a companion .hash file (SHA-256 of the .cu source).
On import, the loader checks the hash and auto-rebuilds if the source changed.
"""
import os, hashlib, subprocess, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

VARIANTS = [
    ("kernels_v2.cu",     "libcuda_kernels_v2.so"),
    ("kernels_v2_f32.cu", "libcuda_kernels_v2_f32.so"),
    ("kernels_v3.cu",     "libcuda_kernels_v3.so"),
    ("kernels_v3_f32.cu", "libcuda_kernels_v3_f32.so"),
]


# ── helpers ─────────────────────────────────────────────────────

def _cu_hash(src_name):
    """SHA-256 hex digest of a .cu source file."""
    path = os.path.join(SCRIPT_DIR, src_name)
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def _hash_path(lib_name):
    return os.path.join(SCRIPT_DIR, lib_name + ".hash")


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
    for path in ['/usr/local/cuda', '/usr/local/cuda-13.2',
                 '/usr/local/cuda-12.0', '/usr/local/cuda-11.0', '/opt/cuda']:
        nvcc = os.path.join(path, 'bin', 'nvcc')
        if os.path.exists(nvcc):
            return nvcc
    return None


def detect_gcc():
    import shutil
    for ver in ['-14', '-13', '-12', '-11']:
        p = shutil.which(f'gcc{ver}')
        if p:
            return p
    p = shutil.which('gcc')
    return p


def _arch_flag(nvcc):
    import re
    r = subprocess.run([nvcc, '--version'], capture_output=True, text=True)
    m = re.search(r'release (\d+\.\d+)', r.stdout)
    if m and float(m.group(1)) >= 11:
        return '-arch=sm_86'
    return ''


# ── build one variant ───────────────────────────────────────────

def _build_one(src_name, lib_name):
    """Compile a single .cu → .so. Returns True on success."""
    nvcc = find_nvcc()
    if not nvcc:
        return False

    gcc = detect_gcc()
    script_dir = SCRIPT_DIR
    base = [nvcc, '-shared', '-Xcompiler', '-fPIC', '-lcudart', '-lm', '-O2']
    arch = _arch_flag(nvcc)
    if arch:
        base.insert(1, arch)

    # Probe flags
    probe_file = os.path.join(script_dir, VARIANTS[0][0])
    probe_out = os.path.join(script_dir, '_probe.so')
    extra = []
    r = subprocess.run(base + ['-o', probe_out, probe_file],
                       capture_output=True, text=True)
    if r.returncode != 0:
        if 'unsupported' in r.stderr:
            extra.append('-allow-unsupported-compiler')
        if gcc:
            extra.extend(['-ccbin', gcc])
        r2 = subprocess.run(base + extra + ['-o', probe_out, probe_file],
                            capture_output=True, text=True)
        if r2.returncode != 0:
            if os.path.exists(probe_out):
                os.remove(probe_out)
            return False
    if os.path.exists(probe_out):
        os.remove(probe_out)

    src_file = os.path.join(script_dir, src_name)
    out_file = os.path.join(script_dir, lib_name)
    cmd = base + extra + ['-o', out_file, src_file]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


# ── public API ──────────────────────────────────────────────────

def ensure(src_name, lib_name):
    """Rebuild *lib_name* if *src_name* changed or .hash is missing.

    Returns True if .so is ready (up-to-date or freshly built), False on failure.
    """
    lib_path = os.path.join(SCRIPT_DIR, lib_name)
    hash_path = _hash_path(lib_name)
    current = _cu_hash(src_name)

    if os.path.exists(lib_path) and os.path.exists(hash_path):
        stored = open(hash_path).read().strip()
        if stored == current:
            return True

    print(f"ampfit CUDA: rebuilding {lib_name} ({src_name} changed)")
    ok = _build_one(src_name, lib_name)
    if ok:
        open(hash_path, 'w').write(current)
    return ok


def build():
    """Build all 4 kernel variants.  Returns True if all succeeded."""
    all_ok = True
    for src_name, lib_name in VARIANTS:
        print(f"  Building {lib_name}...", end=' ')
        sys.stdout.flush()
        ok = ensure(src_name, lib_name)
        if ok:
            print("✓")
        else:
            print("FAILED")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if build() else 1)

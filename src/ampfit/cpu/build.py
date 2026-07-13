#!/usr/bin/env python3
"""Build the CPU shared libraries for ampfit.

Usage:
    python -m ampfit.cpu.build          # force rebuild
    from ampfit.cpu.build import ensure; ensure()  # auto-update on import

Compiles kernels_cpu_v3.c → libcpu_kernels_v3.so via gcc.
Uses SHA-256 hash tracking to auto-rebuild when source changes.
"""
import os, hashlib, subprocess, sys, glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Auto-discover all kernel source files: kernels_*.c → libcpu_kernels_*.so
VARIANTS = []
for c_path in sorted(glob.glob(os.path.join(SCRIPT_DIR, "kernels_*.c"))):
    src_name = os.path.basename(c_path)
    lib_name = "libcpu_" + os.path.splitext(src_name)[0] + ".so"
    VARIANTS.append((src_name, lib_name))


def _src_hash(src_name):
    """SHA-256 hex digest of a C source file."""
    path = os.path.join(SCRIPT_DIR, src_name)
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def _hash_path(src_name):
    return os.path.join(SCRIPT_DIR, src_name + ".hash")


def detect_gcc():
    import shutil
    for ver in ['-14', '-13', '-12', '-11']:
        p = shutil.which(f'gcc{ver}')
        if p:
            return p
    p = shutil.which('gcc')
    return p


def _build_one(src_name, lib_name):
    """Compile a single .c → .so. Returns True on success."""
    gcc = detect_gcc()
    if not gcc:
        return False

    src_file = os.path.join(SCRIPT_DIR, src_name)
    out_file = os.path.join(SCRIPT_DIR, lib_name)

    # Base flags
    base = [
        gcc, '-shared', '-fPIC', '-O3', '-march=native',
        '-fopenmp',
        '-o', out_file, src_file,
    ]

    # Probe: try with -lmvec (GNU math vector library for auto-vectorized cos/sin)
    probe_out = os.path.join(SCRIPT_DIR, '_probe_cpu.so')
    probe_cmd = base + ['-lm', '-lmvec', '_probe_cpu.so']
    # Use a simple test
    r = subprocess.run(
        base + ['-lm', '-lmvec', '-o', probe_out, src_file],
        capture_output=True, text=True)
    if r.returncode == 0:
        cmd = base + ['-lm', '-lmvec']
    else:
        # Fallback: disable auto-vectorization (cos/sin not in libmvec)
        cmd = base + ['-lm', '-fno-tree-vectorize']
    if os.path.exists(probe_out):
        os.remove(probe_out)

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"CPU build FAILED for {lib_name}:", file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        return False
    return True


def ensure(src_name, lib_name):
    """Rebuild *lib_name* if *src_name* changed or .hash is missing.

    Returns True if .so is ready (up-to-date or freshly built), False on failure.
    """
    lib_path = os.path.join(SCRIPT_DIR, lib_name)
    hash_path = _hash_path(src_name)
    current = _src_hash(src_name)

    if os.path.exists(lib_path) and os.path.exists(hash_path):
        stored = open(hash_path).read().strip()
        if stored == current:
            return True

    print(f"ampfit CPU: rebuilding {lib_name} ({src_name} changed)")
    ok = _build_one(src_name, lib_name)
    if ok:
        open(hash_path, 'w').write(current)
    return ok


def build():
    """Build all discovered kernel variants.  Returns True if all succeeded."""
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
    print("ampfit CPU: force rebuilding all kernels")
    for src_name, lib_name in VARIANTS:
        hash_path = _hash_path(src_name)
        if os.path.exists(hash_path):
            os.remove(hash_path)
        print(f"  Building {lib_name}...", end=' ')
        sys.stdout.flush()
        ok = ensure(src_name, lib_name)
        print("✓" if ok else "FAILED")
    print("Done.")

#!/usr/bin/env python3
"""Build the CUDA shared libraries for tabpwa.

Usage:
    python -m tabpwa.cuda.build                # force rebuild all
    python -m tabpwa.cuda.build --tag sm86     # build one variation
    python -m tabpwa.cuda.build --arch sm_70,sm_86   # fat binary

Auto-detects nvcc and required compiler flags.  Auto-discovers all
``kernels_*.cu`` files and builds each into a shared library named by
the machine's **variation tag**:

    tag "" (default)   -> lib<base>.so          (plain, backward compatible)
    tag "sm86"         -> lib<base>.sm86.so
    tag "sm70_sm86"    -> lib<base>.sm70_sm86.so

so one disk can carry binaries for several variations/machines side by
side; set TABPWA_LIB_TAG per machine (or --tag) to pick a variation,
leave it unset for the plain names.

The tag comes from ONE source: the ``TABPWA_LIB_TAG`` environment
variable (or ``--tag`` / :func:`set_tag`); unset means the empty tag
and plain library names.  The compile itself always targets the local
GPU (auto-detected arch flags), and each binary's companion hash file
covers source + tag + nvcc version + schema, so a different
toolchain under the same tag simply rebuilds.
"""
import os, re, hashlib, subprocess, sys, glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_SCHEMA = "tabpwa-kernel-v2"        # bump to invalidate every cache

# Auto-discover all kernel source files: kernels_*.cu -> lib<base>.so
VARIANTS = []
for cu_path in sorted(glob.glob(os.path.join(SCRIPT_DIR, "kernels_*.cu"))):
    src_name = os.path.basename(cu_path)
    VARIANTS.append((src_name,
                     "libcuda_" + os.path.splitext(src_name)[0] + ".so"))

_override_arch = None   # set via set_arch() or --arch
_override_tag = None    # set via set_tag() or --tag
_tag_cache = {}


# ── helpers ─────────────────────────────────────────────────────

def _cu_hash(src_name):
    """SHA-256 hex digest of a .cu source file."""
    path = os.path.join(SCRIPT_DIR, src_name)
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


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


def set_arch(arch):
    """Override GPU architecture for compilation.

    *arch* can be a single SM like ``"sm_86"`` or a comma-separated
    list like ``"sm_70,sm_86"``.  Each entry becomes a separate
    ``-gencode`` flag, producing a fat binary.
    """
    global _override_arch
    _override_arch = arch
    _tag_cache.clear()


def set_tag(tag):
    """Override the variation tag (any string; ``""`` = plain names)."""
    global _override_tag
    _override_tag = tag
    _tag_cache.clear()


def _arch_flags(nvcc):
    """Auto-detect GPU compute capability from nvidia-smi.

    Falls back to sm_86 (Ampere+) with optional sm_70 (Volta) for
    CUDA < 13, when nvidia-smi is not available.
    """
    if _override_arch:
        sms = _override_arch.replace('compute_', 'sm_').split(',')
        flags = []
        for sm in sms:
            sm = sm.strip()
            flags.extend(['-gencode', f'arch=compute_{sm[3:]},code={sm}'])
        return flags
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            ver = r.stdout.strip()
            sm = f'sm_{ver.replace(".", "")}'
            return [f'-arch={sm}']
    except Exception:
        pass
    r = subprocess.run([nvcc, '--version'], capture_output=True, text=True)
    m = re.search(r'release (\d+\.\d+)', r.stdout)
    cuda_ver = float(m.group(1)) if m else 0
    flags = ['-gencode', 'arch=compute_86,code=sm_86']  # Ampere+
    if cuda_ver < 13:
        flags = ['-gencode', 'arch=compute_70,code=sm_70'] + flags  # +Volta
    return flags


def resolve_tag():
    """The variation tag (opaque string; ``""`` = plain library names).

    Purely label-based — NO device detection: ``set_tag()``/``--tag``
    wins, else the ``TABPWA_LIB_TAG`` environment variable, else
    ``""``.  (The compile flags still auto-target the local GPU; the
    tag only namespaces the produced files.)
    """
    key = (_override_tag, os.environ.get("TABPWA_LIB_TAG"))
    if key not in _tag_cache:
        if _override_tag is not None:
            _tag_cache[key] = _override_tag
        else:
            _tag_cache[key] = os.environ.get("TABPWA_LIB_TAG", "")
    return _tag_cache[key]


def lib_file_name(lib_name, tag=None):
    """Shared-library file name for a variation tag.

    ``tag=""`` keeps the plain name (backward compatible); a non-empty
    tag is inserted before the extension.
    """
    if tag is None:
        tag = resolve_tag()
    stem, ext = os.path.splitext(lib_name)
    return f"{stem}.{tag}{ext}" if tag else lib_name


def _hash_path(src_name, tag=None):
    if tag is None:
        tag = resolve_tag()
    return (os.path.join(SCRIPT_DIR, f"{src_name}.{tag}.hash") if tag
            else os.path.join(SCRIPT_DIR, src_name + ".hash"))


def _build_key(src_name, tag, nvcc):
    """Digest of everything the binary depends on."""
    h = hashlib.sha256()
    h.update(open(os.path.join(SCRIPT_DIR, src_name), 'rb').read())
    h.update((tag or "").encode())
    h.update(KEY_SCHEMA.encode())
    if nvcc:
        r = subprocess.run([nvcc, '--version'], capture_output=True,
                           text=True)
        m = re.search(r"release (\S+)", r.stdout or "")
        h.update((m.group(1) if m else "unknown").encode())
        h.update(nvcc.encode())
    return h.hexdigest()


def detect_gcc_compat(base, probe_file, probe_src):
    """Probe-compile; returns the extra flags needed (compiler fallbacks)."""
    extra = []
    r = subprocess.run(base + extra + ['-o', probe_file, probe_src],
                       capture_output=True, text=True)
    if r.returncode != 0:
        if 'unsupported' in r.stderr:
            extra.append('-allow-unsupported-compiler')
        gcc = detect_gcc()
        if gcc:
            extra.extend(['-ccbin', gcc])
    return extra


# ── build one variant ───────────────────────────────────────────

def _build_one(src_name, out_file):
    """Compile a single .cu -> the (already tag-suffixed) out_file."""
    nvcc = find_nvcc()
    if not nvcc:
        return False

    base = [nvcc, '-shared', '-Xcompiler', '-fPIC', '-lcudart', '-lm', '-O2']
    base.extend(_arch_flags(nvcc))

    probe_file = os.path.join(SCRIPT_DIR, '_probe.so')
    probe_src = os.path.join(SCRIPT_DIR, VARIANTS[0][0])
    try:
        extra = detect_gcc_compat(base, probe_file, probe_src)
        # probe once with the extra flags to validate the toolchain
        r = subprocess.run(base + extra + ['-o', probe_file, probe_src],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False
        src_file = os.path.join(SCRIPT_DIR, src_name)
        r = subprocess.run(base + extra + ['-o', out_file, src_file],
                           capture_output=True, text=True)
        return r.returncode == 0
    finally:
        if os.path.exists(probe_file):
            os.remove(probe_file)


# ── public API ──────────────────────────────────────────────────

def ensure(src_name, lib_name):
    """Rebuild the tagged *lib_name* if the build key changed.

    The key covers the .cu source AND the variation identity (tag +
    nvcc version + schema), so a shared disk never accepts another
    machine's binary.  ``lib_name`` is the BASE name; the actual file
    is :func:`lib_file_name` of it (plain name for the default tag).

    Returns the full path of the ready shared library, or None on failure.
    """
    nvcc = find_nvcc()
    tag = resolve_tag()
    lib_path = os.path.join(SCRIPT_DIR, lib_file_name(lib_name, tag))
    hash_path = _hash_path(src_name, tag)
    current = _build_key(src_name, tag, nvcc)

    if os.path.exists(lib_path) and os.path.exists(hash_path):
        stored = open(hash_path).read().strip()
        if stored == current:
            return lib_path

    print(f"tabpwa CUDA: rebuilding {os.path.basename(lib_path)} "
          f"({src_name} changed)")
    if _build_one(src_name, lib_path):
        open(hash_path, 'w').write(current)
        return lib_path
    return None


def build():
    """Build all discovered kernel variants for this machine's tag."""
    all_ok = True
    for src_name, lib_name in VARIANTS:
        print(f"  Building {lib_file_name(lib_name)}...", end=' ')
        sys.stdout.flush()
        ok = ensure(src_name, lib_name)
        print("✓" if ok else "FAILED")
        all_ok = all_ok and bool(ok)
    return all_ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build CUDA kernels for tabpwa")
    ap.add_argument("--arch", default=None,
                    help="Override GPU arch (e.g. 'sm_86' or 'sm_70,sm_86')")
    ap.add_argument("--tag", default=None,
                    help="Override the variation tag (labels the .so)")
    args = ap.parse_args()

    if args.arch:
        set_arch(args.arch)
    if args.tag:
        set_tag(args.tag)

    # Force rebuild: drop the tag's hash files, rebuild everything
    print("tabpwa CUDA: force rebuilding all kernels")
    tag = resolve_tag()
    pattern = (f"*.{'*' if tag else ''}{tag}.hash" if tag
               else "*.hash")
    for hash_path in glob.glob(os.path.join(SCRIPT_DIR, pattern)):
        os.remove(hash_path)
    build()
    print("Done.")

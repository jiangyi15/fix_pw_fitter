#!/usr/bin/env python3
"""Build the CUDA shared libraries for tabpwa.

Usage:
    python -m tabpwa.cuda.build                # build what is STALE only
    python -m tabpwa.cuda.build --force        # rebuild all
    python -m tabpwa.cuda.build pwa_gen        # only matching kernels
    python -m tabpwa.cuda.build 'v5_*' --force # glob, forced
    python -m tabpwa.cuda.build --list         # show what is discovered
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

The default is **incremental**: a kernel is compiled only when its build key
changed (source, tag, nvcc version, schema), so the common case — nothing
changed — compiles nothing and costs milliseconds.  ``--force`` rebuilds the
selection unconditionally.

The tag comes from ONE source: the ``TABPWA_LIB_TAG`` environment
variable (or ``--tag`` / :func:`set_tag`); unset means the empty tag
and plain library names.  The compile itself always targets the local
GPU (auto-detected arch flags), and each binary's companion hash file
covers source + tag + nvcc version + schema, so a different
toolchain under the same tag simply rebuilds.
"""
import os, re, fnmatch, hashlib, subprocess, sys, glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_SCHEMA = "tabpwa-kernel-v2"        # bump to invalidate every cache

# Where the ``.hash`` companions go.  Default: next to the sources.  Set
# ``TABPWA_HASH_DIR`` when the source tree is not writable (a read-only or
# packaged install) — the binaries still land next to the sources, only the
# bookkeeping moves.  Not shared across machines: the key already covers the
# nvcc identity, but a hash is cheap to lose and expensive to get wrong.
HASH_DIR = (os.environ.get("TABPWA_HASH_DIR") or SCRIPT_DIR)


def _discover():
    """All kernel source files, by glob: ``kernels_*.cu`` -> ``lib<base>.so``.

    Re-run on demand (:func:`variants`) rather than frozen at import, so a
    newly added kernel is picked up without a process restart.
    """
    out = []
    for cu_path in sorted(glob.glob(os.path.join(SCRIPT_DIR, "kernels_*.cu"))):
        src_name = os.path.basename(cu_path)
        out.append((src_name,
                    "libcuda_" + os.path.splitext(src_name)[0] + ".so"))
    return out


#: Snapshot of the discovered kernels (kept for backward compatibility;
#: :func:`variants` is the live view).
VARIANTS = _discover()

_override_arch = None   # set via set_arch() or --arch
_override_tag = None    # set via set_tag() or --tag
_tag_cache = {}
_flags_cache = None     # compiler flag prefix, probed once (see _compiler_flags)
_flags_key = None       # (nvcc, arch) that prefix belongs to


def variants():
    """The kernels currently present on disk (``[(src_name, lib_name)]``)."""
    global VARIANTS
    VARIANTS = _discover()
    return VARIANTS


def select(patterns=(), all_variants=None):
    """Kernels matching any of *patterns* (fnmatch globs on the source name).

    An empty selection means **everything**, so ``select()`` and
    ``select([])`` are the same as no filter at all.  A pattern that matches
    nothing is an error rather than a silent no-op — a typo in
    ``python -m tabpwa.cuda.build pwa_genn`` should not look like success.

    Patterns are matched against the source name with and without the
    ``.cu`` suffix and against the library's base name, so ``pwa_gen``,
    ``kernels_v5_pwa_gen*`` and ``libcuda_kernels_v5_pwa_gen*`` all work.
    """
    items = list(all_variants if all_variants is not None else variants())
    pats = [p for p in patterns if p]
    if not pats:
        return items

    def _cands(src, lib):
        stem = os.path.splitext(src)[0]
        return (src, stem, lib, os.path.splitext(lib)[0])

    def _match(pat, cands):
        # A bare word is a SUBSTRING match, so ``pwa_gen`` finds
        # ``kernels_v5_pwa_gen_bw.cu`` without the caller having to know the
        # full name or type wildcards; anything with glob metacharacters is
        # used as given, which keeps ``v5_*`` / ``kernels_v3`` precise.
        if any(c in pat for c in "*?["):
            return any(fnmatch.fnmatchcase(c, pat) for c in cands)
        return any(pat in c for c in cands)

    chosen, unmatched = [], list(pats)
    for item in items:
        cands = _cands(*item)
        hit = [p for p in pats if _match(p, cands)]
        if hit:
            chosen.append(item)
            for p in hit:
                if p in unmatched:
                    unmatched.remove(p)
    if unmatched:
        raise ValueError(
            f"no kernel matches {unmatched}; available: "
            f"{sorted(s for s, _l in items)}")
    return chosen


# ── helpers ─────────────────────────────────────────────────────

def _cu_hash(src_name):
    """SHA-256 hex digest of a .cu source file."""
    path = os.path.join(SCRIPT_DIR, src_name)
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def _nvcc_candidates():
    """Every nvcc this machine can plausibly use, best guess first.

    Order matters: an explicit ``TABPWA_NVCC`` / ``CUDA_HOME`` outranks
    whatever happens to be on ``PATH``, because a project that pins a toolkit
    (e.g. CUDA 12 for ``sm_70`` support) must not be overridden by a newer
    one earlier in the environment.
    """
    import shutil
    out = []
    explicit = os.environ.get("TABPWA_NVCC")
    if explicit:
        out.append(explicit)
    for var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(var)
        if root:
            out.append(os.path.join(root, "bin", "nvcc"))
    on_path = shutil.which("nvcc")
    if on_path:
        out.append(on_path)
    for root in ("/usr/local/cuda", "/usr/local/cuda-13.2",
                 "/usr/local/cuda-12.9", "/usr/local/cuda-12.0",
                 "/usr/local/cuda-11.8", "/opt/cuda"):
        out.append(os.path.join(root, "bin", "nvcc"))
    seen, uniq = set(), []
    for p in out:
        if p and p not in seen and os.path.exists(p):
            seen.add(p)
            uniq.append(p)
    return uniq


def find_nvcc():
    """The nvcc to compile with.

    ``TABPWA_NVCC`` > ``CUDA_HOME``/``CUDA_PATH`` > ``PATH`` > well-known
    prefixes.  If an explicit ``--arch`` was requested and the preferred
    toolchain cannot target it, the next candidate that CAN is used — a
    dropped architecture (CUDA 13 removed ``compute_70``) should select the
    toolkit that still supports it rather than fail.
    """
    cands = _nvcc_candidates()
    if not cands:
        return None
    if _override_arch:
        wanted = {"compute_" + s.strip().replace("sm_", "").replace(
            "compute_", "") for s in _override_arch.split(",") if s.strip()}
        for nvcc in cands:
            supported = _supported_sms(nvcc)
            if not supported or wanted <= supported:
                return nvcc
        # none can: return the preferred one and let validate_arch() explain
    return cands[0]


def detect_gcc():
    """A host compiler nvcc can actually use.

    Preference order: ``TABPWA_CC`` / ``CC`` env, then versioned system gccs
    (oldest first — CUDA rejects FUTURE gcc majors, so ``gcc-13`` is often the
    only usable one when the default is 15), then the conda-forge triple
    compiler (``x86_64-conda-linux-gnu-g++``, which is how a cu12 env ships a
    supported host compiler), then plain ``gcc``.
    """
    import shutil
    for var in ("TABPWA_CC", "CC"):
        p = os.environ.get(var)
        if p and os.path.exists(p):
            return p
    for ver in ['-14', '-13', '-12', '-11']:
        p = shutil.which(f'gcc{ver}')
        if p:
            return p
    for triple in ("x86_64-conda-linux-gnu", "x86_64-conda_cos6-linux-gnu"):
        p = shutil.which(f"{triple}-g++")
        if p:
            return p
    p = shutil.which('gcc')
    return p


def _supported_sms(nvcc):
    """The SM targets this nvcc can emit, best effort."""
    r = subprocess.run([nvcc, '--list-gpu-arch'], capture_output=True,
                       text=True)
    if r.returncode != 0:
        return set()
    return {ln.strip() for ln in r.stdout.split()
            if ln.strip().startswith("compute_")}


def validate_arch(nvcc=None):
    """Raise with a clear message when an explicit arch is not supported.

    A toolchain that dropped an architecture (CUDA 13 dropped ``compute_70``)
    used to surface as every kernel quietly printing ``FAILED``; the packaging
    script then shipped whatever binaries happened to be on disk.  This makes
    the mismatch a hard error at the point the arch is chosen.
    """
    if not _override_arch:
        return
    nvcc = nvcc or find_nvcc()
    if not nvcc:
        return
    supported = _supported_sms(nvcc)
    if not supported:
        return                      # older nvcc: cannot check, do not guess
    asked = ["compute_" + s.strip().replace("sm_", "").replace("compute_", "")
             for s in _override_arch.split(",") if s.strip()]
    missing = [a for a in asked if a not in supported]
    if missing:
        avail = ", ".join(sorted(supported, key=lambda s: int(s.split("_")[1])))
        raise ValueError(
            f"arch {', '.join(missing)} not supported by {nvcc} "
            f"(supported: {avail})")


def set_arch(arch):
    """Override GPU architecture for compilation.

    *arch* can be a single SM like ``"sm_86"`` or a comma-separated
    list like ``"sm_70,sm_86"``.  Each entry becomes a separate
    ``-gencode`` flag, producing a fat binary.

    Raises ``ValueError`` if the nvcc in use cannot target one of the
    requested architectures (see :func:`validate_arch`).
    """
    global _override_arch, _flags_cache, _flags_key
    _override_arch = arch
    _flags_cache = None       # the arch flags are part of the prefix
    _flags_key = None
    _tag_cache.clear()
    validate_arch()


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
    if tag:
        return os.path.join(HASH_DIR, f"{src_name}.{tag}.hash")
    return os.path.join(HASH_DIR, src_name + ".hash")


def _arch_signature():
    """The architecture the compile targets, as a stable string.

    Part of the build key: an `sm_120`-only binary is NOT interchangeable
    with a fat `sm_70,sm_75,sm_86` one, so a packaging run with
    ``set_arch(...)`` must not accept a previously built native binary (and
    vice versa).  Without this the AppImage build silently shipped
    locally-only cubins.
    """
    if _override_arch:
        return "arch:" + ",".join(
            s.strip() for s in _override_arch.split(",") if s.strip())
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return "auto:" + r.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return "auto:unknown"


def _build_key(src_name, tag, nvcc):
    """Digest of everything the binary depends on."""
    h = hashlib.sha256()
    h.update(open(os.path.join(SCRIPT_DIR, src_name), 'rb').read())
    h.update((tag or "").encode())
    h.update(KEY_SCHEMA.encode())
    # the target arch changes the cubin even for identical source
    flag_arch = None
    try:
        flag_arch = _arch_signature()
    except Exception:
        pass
    h.update((flag_arch or "").encode())
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


def _compiler_flags():
    """The nvcc flag prefix, with the compiler-compat probe done ONCE.

    ``detect_gcc_compat`` compiles a throwaway object to find out whether the
    system gcc needs ``-allow-unsupported-compiler`` / ``-ccbin``.  That is a
    property of the toolchain, not of the kernel, so it is probed once per
    process and cached; doing it per kernel cost two extra nvcc invocations
    (one of them compiling a whole unrelated kernel) on every rebuild.

    The cache key is the resolved nvcc plus the requested arch, so changing
    ``TABPWA_NVCC``/``CUDA_HOME`` or ``--arch`` mid-process cannot reuse
    flags that belong to another toolkit.
    """
    global _flags_cache, _flags_key
    nvcc = find_nvcc()
    if not nvcc:
        raise RuntimeError(
            "no nvcc found — set TABPWA_NVCC, CUDA_HOME or put nvcc on PATH")
    key = (nvcc, _override_arch or "", _arch_signature())
    if _flags_cache is not None and _flags_key == key:
        return _flags_cache
    base = [nvcc, '-shared', '-Xcompiler', '-fPIC', '-lcudart', '-lm', '-O2']
    base.extend(_arch_flags(nvcc))
    probe_file = os.path.join(HASH_DIR, '_probe.so')
    os.makedirs(HASH_DIR, exist_ok=True)
    # any real source works as a probe; which one is irrelevant, the flags
    # are a property of the toolchain
    probe_src = os.path.join(SCRIPT_DIR, VARIANTS[0][0])
    try:
        _flags_cache = base + detect_gcc_compat(base, probe_file, probe_src)
        _flags_key = key
    finally:
        if os.path.exists(probe_file):
            os.remove(probe_file)
    return _flags_cache


# ── build one variant ───────────────────────────────────────────

def _build_one(src_name, out_file):
    """Compile a single .cu -> the (already tag-suffixed) out_file."""
    if not find_nvcc():
        return False
    cmd = _compiler_flags() + ['-o', out_file,
                               os.path.join(SCRIPT_DIR, src_name)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"\n  nvcc failed on {src_name}:\n{r.stderr[-2000:]}",
              file=sys.stderr)
    return r.returncode == 0


# ── public API ──────────────────────────────────────────────────

def ensure(src_name, lib_name, force=False):
    """Rebuild the tagged *lib_name* if the build key changed.

    The key covers the .cu source, the target architecture, and the variation
    identity (tag + nvcc version + schema), so a shared disk never accepts
    another machine's binary.  ``lib_name`` is the BASE name; the actual file
    is :func:`lib_file_name` of it (plain name for the default tag).
    ``force=True`` skips the staleness check and always recompiles.

    Returns the full path of the ready shared library, or None on failure.
    """
    nvcc = find_nvcc()
    tag = resolve_tag()
    lib_path = os.path.join(SCRIPT_DIR, lib_file_name(lib_name, tag))
    hash_path = _hash_path(src_name, tag)
    current = _build_key(src_name, tag, nvcc)

    if not force and os.path.exists(lib_path) and os.path.exists(hash_path):
        stored = open(hash_path).read().strip()
        if stored == current:
            return lib_path

    print(f"tabpwa CUDA: rebuilding {os.path.basename(lib_path)} "
          f"({src_name} changed)")
    if _build_one(src_name, lib_path):
        os.makedirs(HASH_DIR, exist_ok=True)
        open(hash_path, 'w').write(current)
        return lib_path
    return None


def is_current(src_name, lib_name=None):
    """True when the tagged library exists and its build key matches.

    The check :func:`build` uses to skip a kernel without compiling it; it
    never touches the compiler, so it costs a hash of the source.
    """
    if lib_name is None:
        lib_name = "libcuda_" + os.path.splitext(src_name)[0] + ".so"
    tag = resolve_tag()
    lib_path = os.path.join(SCRIPT_DIR, lib_file_name(lib_name, tag))
    hash_path = _hash_path(src_name, tag)
    if not (os.path.exists(lib_path) and os.path.exists(hash_path)):
        return False
    try:
        stored = open(hash_path).read().strip()
    except OSError:
        return False
    return stored == _build_key(src_name, tag, find_nvcc())


def build(patterns=(), force=False, quiet=False):
    """Build the kernels matching *patterns* (default: every discovered one).

    Incremental by default: a kernel whose build key is unchanged is skipped,
    so the common case compiles nothing.  ``force=True`` rebuilds the whole
    selection.  An empty selection means all kernels; a pattern that matches
    nothing raises rather than silently building the wrong set.

    Returns ``True`` when every selected kernel is ready.
    """
    items = select(patterns)
    if not items:
        raise ValueError(
            f"no kernels found in {SCRIPT_DIR} (expected kernels_*.cu)")
    all_ok = True
    for src_name, lib_name in items:
        label = lib_file_name(lib_name)
        if not force and is_current(src_name, lib_name):
            if not quiet:
                print(f"  {label}: up to date")
            continue
        if not quiet:
            print(f"  Building {label}...", end=' ')
            sys.stdout.flush()
        ok = ensure(src_name, lib_name, force=force)
        all_ok = all_ok and bool(ok)
        if not quiet:
            print("✓" if ok else "FAILED")
    return all_ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Build CUDA kernels for tabpwa (incremental by default)")
    ap.add_argument("patterns", nargs="*", metavar="PATTERN",
                    help="kernel name globs to build (e.g. pwa_gen, 'v5_*'); "
                         "default: every kernels_*.cu")
    ap.add_argument("--force", action="store_true",
                    help="rebuild the selection even when up to date")
    ap.add_argument("--list", action="store_true",
                    help="list the discovered kernels and exit")
    ap.add_argument("--arch", default=None,
                    help="Override GPU arch (e.g. 'sm_86' or 'sm_70,sm_86')")
    ap.add_argument("--tag", default=None,
                    help="Override the variation tag (labels the .so)")
    args = ap.parse_args()

    if args.arch:
        set_arch(args.arch)
    if args.tag:
        set_tag(args.tag)

    try:
        chosen = select(args.patterns)
    except ValueError as exc:
        sys.exit(f"tabpwa CUDA: {exc}")

    if args.list:
        for src_name, lib_name in chosen:
            mark = "up to date" if is_current(src_name, lib_name) else "stale"
            print(f"  {src_name:38s} -> {lib_file_name(lib_name):46s} {mark}")
        sys.exit(0)

    print(f"tabpwa CUDA: building {len(chosen)} kernel(s)"
          + (" (forced)" if args.force else " (incremental)"))
    ok = build(args.patterns, force=args.force)
    print("Done." if ok else "FAILED")
    sys.exit(0 if ok else 1)

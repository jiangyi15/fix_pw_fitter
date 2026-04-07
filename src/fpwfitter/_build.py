"""
_build.py  –  Automatic CUDA shared-library build.

Called transparently by ``fpwfitter.__init__``.  If
``libfpwfitter.so`` is missing (or stale), it is compiled
in-place via ``nvcc``.  A file-lock prevents concurrent builds.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import subprocess
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PKG_DIR = Path(__file__).parent.resolve()          # src/fpwfitter
_LIB     = _PKG_DIR / "libfpwfitter.so"
_CU      = _PKG_DIR / "fpwfitter.cu"
_HASH    = _PKG_DIR / ".cu_hash"                     # cache key


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ensure_lib() -> Path:
    """Return the path to the compiled .so, building it if needed."""
    if _LIB.exists() and not _needs_rebuild():
        return _LIB
    _build()
    return _LIB


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _needs_rebuild() -> bool:
    if not _LIB.exists():
        return True
    if not _HASH.exists():
        return True
    return _HASH.read_text() != _cu_hash()


def _cu_hash() -> str:
    """SHA-256 of the .cu file (fast change detection)."""
    return hashlib.sha256(_CU.read_bytes()).hexdigest()


def _build() -> None:
    nvcc = os.environ.get("NVCC", "nvcc")
    cmd = [
        nvcc,
        "-Xcompiler", "-fPIC",
        "-shared",
        "-O3",
        "-arch=native",
        str(_CU),
        "-lcublas",
        "-o", str(_LIB),
    ]
    print(f"[fpwfitter] Building CUDA library ({_CU.name}) …", flush=True)
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr, file=__import__("sys").stderr)
        raise RuntimeError(f"nvcc failed (exit {exc.returncode})") from exc
    elapsed = time.perf_counter() - t0
    print(f"[fpwfitter] Built {_LIB.name} in {elapsed:.1f}s", flush=True)
    _HASH.write_text(_cu_hash())

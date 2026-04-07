"""
_build.py  –  Automatic CUDA shared-library build.

Called transparently by ``fpwfitter.__init__``.  If
``libfpwfitter.so`` is missing (or stale), it is compiled
in-place via ``nvcc``.  The target architecture (``sm_XX``)
is auto-detected from the current GPU.
"""

from __future__ import annotations

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
# GPU architecture detection
# ---------------------------------------------------------------------------

def _detect_sm() -> str:
    """Return ``sm_XX`` for the current GPU via nvidia-smi, or ``sm_75``."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            cap = out.stdout.strip().split("\n")[0].strip().replace(".", "")
            return f"sm_{cap}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return "sm_75"   # safe fallback (Turing)


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
    mp_lib = _PKG_DIR / "libfpwfitter_mp.so"
    if not mp_lib.exists():
        return True
    if not _HASH.exists():
        return True
    return _HASH.read_text() != _cu_hash()


def _cu_hash() -> str:
    """SHA-256 of the .cu file (fast change detection)."""
    return hashlib.sha256(_CU.read_bytes()).hexdigest()


def _build() -> None:
    sm = _detect_sm()
    nvcc = os.environ.get("NVCC", "nvcc")
    for name, sources, extra in [
        ("libfpwfitter.so", ["fpwfitter.cu"], "-lcublas"),
        ("libfpwfitter_mp.so", ["fpwfitter_mp.cu"], "-lcublas"),
    ]:
        cmd = [
            nvcc,
            "-Xcompiler", "-fPIC",
            "-shared",
            "-O3",
            f"-arch={sm}",
        ]
        for src in sources:
            cmd.append(str(_PKG_DIR / src))
        cmd.extend(extra.split())
        cmd.extend(["-o", str(_PKG_DIR / name)])
        print(f"[fpwfitter] Building {name} for {sm} …", flush=True)
        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            print(exc.stderr, file=__import__("sys").stderr)
            raise RuntimeError(f"nvcc failed on {name} (exit {exc.returncode})") from exc
        elapsed = time.perf_counter() - t0
        print(f"[fpwfitter] Built {name} in {elapsed:.1f}s", flush=True)
    # Hash based on both .cu files
    h = hashlib.sha256(_CU.read_bytes())
    mp_cu = _PKG_DIR / "fpwfitter_mp.cu"
    if mp_cu.exists():
        h.update(mp_cu.read_bytes())
    _HASH.write_text(h.hexdigest())

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
import re
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
    """Return ``sm_XX`` for the current GPU, or fallback ``sm_75``."""
    # Try nvidia-smi first
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

    # Try a tiny CUDA program
    cuda_src = """
#include <stdio.h>
#include <cuda_runtime.h>
int main(){
    int dev; cudaGetDevice(&dev);
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    printf("sm_%d%d\\n", major, minor);
    return 0;
}
"""
    try:
        tmp_src = Path("/tmp/_fpw_detect_cuda.cu")
        tmp_bin = Path("/tmp/_fpw_detect_cuda")
        tmp_src.write_text(cuda_src)
        nvcc = os.environ.get("NVCC", "nvcc")
        subprocess.run([nvcc, "-o", str(tmp_bin), str(tmp_src)],
                       capture_output=True, timeout=30, check=True)
        out = subprocess.run([str(tmp_bin)], capture_output=True,
                             text=True, timeout=5, check=True)
        sm = out.stdout.strip()
        if re.match(r"sm_\d+", sm):
            return sm
    except Exception:
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
    if not _HASH.exists():
        return True
    return _HASH.read_text() != _cu_hash()


def _cu_hash() -> str:
    """SHA-256 of the .cu file (fast change detection)."""
    return hashlib.sha256(_CU.read_bytes()).hexdigest()


def _build() -> None:
    sm = _detect_sm()
    nvcc = os.environ.get("NVCC", "nvcc")
    cmd = [
        nvcc,
        "-Xcompiler", "-fPIC",
        "-shared",
        "-O3",
        f"-arch={sm}",
        str(_CU),
        "-lcublas",
        "-o", str(_LIB),
    ]
    print(f"[fpwfitter] Building CUDA library ({_CU.name}) "
          f"for {sm} …", flush=True)
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr, file=__import__("sys").stderr)
        raise RuntimeError(f"nvcc failed (exit {exc.returncode})") from exc
    elapsed = time.perf_counter() - t0
    print(f"[fpwfitter] Built {_LIB.name} in {elapsed:.1f}s", flush=True)
    _HASH.write_text(_cu_hash())

"""Free-VRAM probe through ``libcudart`` — NEVER through ``torch``.

Two tests here need to know how much device memory is free before and after a
call (a leak / an allocation the load must not make).  ``torch`` was the
convenient way to ask, but importing it is not free:

MEASURED on this box: ``torch.cuda.init()`` makes every LATER raw-CUDA call in
the same process 5.1x slower and it never recovers — one s0e2
create/load/compute/free cycle goes from 5.71 ms to 29.38 ms.  That is how
``tests/test_flat_compute.py`` became 31 s alone, 102 s right after
``tests/test_fixed_coeff.py``, and 169 s inside the full suite.

So the probe is the CUDA runtime API called directly through ``ctypes``:
``cudaDeviceSynchronize`` / ``cudaMemGetInfo``, exact BYTES (not MiB), and no
new module in ``sys.modules``.  ``tests/_flat_compute.py`` carries a gate that
fails if anything ever loads ``torch`` again.

Semantics are the ones the torch probe had: synchronise, read the free bytes,
compare.  A missing ``libcudart`` (or no usable device) skips, exactly as
``pytest.importorskip("torch")`` did.
"""
import ctypes
import ctypes.util

import pytest

#: the loaded library (``None`` once we know it cannot be loaded)
_LIB = None
#: one probe per process — a failed ``dlopen`` must not be retried per call
_TRIED = False

#: the skip every caller used, in ONE place
NO_DEVICE = "no CUDA device for the free-bytes probe"


def _load():
    """``ctypes.CDLL`` for the CUDA runtime, or ``None`` (cached)."""
    for cand in ("libcudart.so", "libcudart.so.13", "libcudart.so.12",
                 ctypes.util.find_library("cudart")):
        if not cand:
            continue
        try:
            lib = ctypes.CDLL(cand)
        except OSError:
            continue
        lib.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t),
                                      ctypes.POINTER(ctypes.c_size_t)]
        lib.cudaMemGetInfo.restype = ctypes.c_int
        lib.cudaDeviceSynchronize.restype = ctypes.c_int
        lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cudaGetDeviceCount.restype = ctypes.c_int
        return lib
    return None


def lib():
    """The CUDA runtime handle, or ``None``.  Never raises."""
    global _LIB, _TRIED
    if not _TRIED:
        _TRIED = True
        _LIB = _load()
    return _LIB


def require():
    """``lib()``, or skip — the ``importorskip("torch")``/``is_available()`` pair
    these probes used, moved into one call."""
    handle = lib()
    if handle is None:
        pytest.skip(NO_DEVICE)
    n = ctypes.c_int(0)
    if handle.cudaGetDeviceCount(ctypes.byref(n)) != 0 or n.value <= 0:
        pytest.skip(NO_DEVICE)
    return handle


def sync(handle):
    """Wait for every queued kernel — a probe must not race the work."""
    rc = handle.cudaDeviceSynchronize()
    assert rc == 0, rc


def free_bytes(handle):
    """Free device memory in BYTES (``torch`` reported bytes too)."""
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    rc = handle.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    assert rc == 0, rc
    return free.value

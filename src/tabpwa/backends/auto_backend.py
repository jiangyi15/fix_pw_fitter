"""The ``"auto"`` backend: the MACHINE's state and the config's shape pick.

A backend cannot know which data it will be given, so routing on an event
count (or on a per-event memory estimate) is wrong: the same fitter is built
before ``set_phsp`` / ``set_data`` and then reused.  What *is* knowable at
construction is the host (is CUDA usable, how much GPU memory is free) and
the kernel config (``n_res``, the tile size its shape implies).

There is no shared selection table and no generic dispatcher: each model's
``auto`` class owns its rule in ``AutoBackend.choose`` (the class policy —
a model's policy lives with that model), and every branch in those rules is
a fact stated by the code: a shape guard the kernels themselves enforce, or
whether CUDA is usable.  Free GPU memory is REPORTED by ``describe()`` but
never decides — the sample size is unknown at construction, so a memory
threshold could only ever be a guess.
"""
import subprocess

import numpy as np

from tabpwa.cuda.build import find_nvcc

from .core import (ComputeBackend, backends_for_model, create_backend,
                   register_backend)



_QUERY = "memory.free,memory.total,compute_cap"
_MIB = 1024**2          # nvidia-smi --format=csv,nounits reports MiB
_KIB = 1024

#: Conservative published per-block shared-memory opt-in limits, by compute
#: capability.  A wrong entry only costs performance (the s0e `create` still
#: refuses an oversized tile loudly); unknown CCs get the 96 KiB floor.
_SMEM_OPTIN_KIB = {
    (7, 0): 96, (7, 5): 96,
    (8, 0): 99, (8, 6): 99, (8, 9): 99, (12, 0): 99,
    (9, 0): 227,
}
_SMEM_OPTIN_DEFAULT_KIB = 96      # KiB; unknown compute capability

# the s0e tile rule, mirrored from kernels_flat_pwa_gen_s0e.cu
_TILE_DEF_TE = 8
_TILE_SMEM_BUDGET = 48 * 1024

_state = None


def _int_or_none(text):
    try:
        return int(float(text.strip()))
    except (TypeError, ValueError):
        return None


def _mib_or_none(text):
    """``nounits`` are MiB; the state speaks bytes."""
    mib = _int_or_none(text)
    return None if mib is None else mib * _MIB


def _cc_or_none(text):
    try:
        major, minor = text.strip().split(".", 1)
        return (int(major), int(minor))
    except (AttributeError, ValueError):
        return None


def cuda_state():
    """Cache one ``nvidia-smi`` probe plus the nvcc lookup.

    Returns ``dict(has_cuda, free_bytes, total_bytes, cc, smem_optin_bytes)``.
    Any failure of the probe (no binary, non-zero exit, timeout, unusable
    output) or a missing nvcc gives ``has_cuda=False`` with the rest ``None``.
    Never raises; cached per process (see :func:`reset_cuda_state`).
    """
    global _state
    if _state is not None:
        return _state
    empty = dict(has_cuda=False, free_bytes=None, total_bytes=None, cc=None,
                 smem_optin_bytes=None)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + _QUERY,
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False)
    except Exception:
        _state = empty
        return _state
    if out.returncode != 0 or not out.stdout.strip():
        _state = empty
        return _state
    try:
        nvcc = find_nvcc()
    except Exception:
        nvcc = None
    if nvcc is None:
        _state = empty
        return _state
    fields = out.stdout.strip().splitlines()[0].split(",")
    free = _mib_or_none(fields[0])
    total = _mib_or_none(fields[1]) if len(fields) > 1 else None
    cc = _cc_or_none(fields[2]) if len(fields) > 2 else None
    optin = _KIB * _SMEM_OPTIN_KIB.get(cc, _SMEM_OPTIN_DEFAULT_KIB)
    _state = dict(has_cuda=True, free_bytes=free, total_bytes=total, cc=cc,
                  smem_optin_bytes=optin)
    return _state


def reset_cuda_state():
    """Drop the cached probe (tests, or after the GPU state changed)."""
    global _state
    _state = None


def flat_shape(kernel_config):
    """``(n_wave, n_res, n_basis)`` of a flat ``pwa_gen`` kernel config.

    MIRRORS the three lines in ``tabpwa/cuda/_flat_pwa_gen_s0e.py``::

        self.n_wave  = fl_table.shape[0]
        self.n_res   = num_table.shape[1]
        self.n_basis = basis_table.shape[0]

    That wrapper is a working kernel module (kernel policy: immutable), so the
    mirror lives here rather than being imported; ``tests/test_auto_backend.py``
    pins the two together against a real backend instance.
    """
    return (int(np.asarray(kernel_config["fl_table"]).shape[0]),
            int(np.asarray(kernel_config["num_table"]).shape[1]),
            int(np.asarray(kernel_config["basis_table"]).shape[0]))


def tile_bytes(n_wave, n_basis, n_proj, rsize=1, budget=_TILE_SMEM_BUDGET):
    """Shared memory the s0e tile needs, as ``gps0_layout`` sizes it.

    ``te`` follows ``gps0_pick_te``: GPS0_DEF_TE (8), rounded UP to whole
    log-sum groups when ``rsize <= 8``, then shrunk one event at a time while
    the tile exceeds *budget*.  ``gps0_layout``'s double count is::

        te·(2W + 5N + 4P + 3) + 4N + 2NB + 3,     W = P·N

    (2W: cr/ci; 5N: D/rd/pv/gr/gi; 4P: A/dA; 3: pe/q/sc; 4N + 2NB + 3: the
    block-reduction buffers).  This is an UPPER bound for the shape: the C side
    may pick a smaller ``te`` (a hint, or a stricter budget), which only lowers
    the bytes — so ``tile_bytes(...) > optin`` is certainly refused, and a
    false negative merely means the s0e ``create`` says so loudly.
    """
    N = int(n_wave)
    NB = int(n_basis)
    P = max(1, int(n_proj))
    W = P * N
    te = _TILE_DEF_TE
    if int(rsize) > 1 and int(rsize) <= _TILE_DEF_TE:
        te = int(rsize) * max(1, -(-_TILE_DEF_TE // int(rsize)))
    while te > 1:
        need = 8 * (te * (2 * W + 5 * N + 4 * P + 3) + 4 * N + 2 * NB + 3)
        if need <= int(budget):
            break
        te -= 1
    return 8 * (te * (2 * W + 5 * N + 4 * P + 3) + 4 * N + 2 * NB + 3)


GIB = 1024**3


class AutoBackend(ComputeBackend):
    """``auto`` for ONE amplitude model: resolve once, then forward.

    Registered through a per-model subclass (``AutoFlatPwaGenBackend`` /
    ``AutoPWABackend``), so the registration scope says which family the
    instance belongs to and the class — not the caller — owns the model name.
    A model with no ``auto`` registration is simply not eligible; the registry
    error lists what is.

    The choice is made EAGERLY in :meth:`__init__` from the host state and the
    kernel config — never from the data, which is not loaded yet and may be
    several datasets — and every call is forwarded to the concrete backend in
    ``self._delegate``.  Nothing intercepts ``load_data``: the decision needs
    neither the sample nor its size.

    ``chosen`` names the delegate; ``describe()`` says why, in one line::

        auto -> cuda_flat_pwa_gen_s0e (free 11.2 GiB, cc 12.0, n_wave 2, …)

    Extra kwargs (``resolution_size``, ``batch_size``, …) go to the delegate,
    and ``__getattr__`` forwards its attributes (``resolution_size``,
    ``kernel_stats``, …) so the wrapper stays transparent.
    """

    MODEL = None                             # set by the per-model subclass

    def choose(self, kernel_config, **kw):
        """The concrete backend name for THIS model — override.

        ``**kw`` are the constructor kwargs the caller passed (only
        ``resolution_size`` is consulted, by the pwa rule); most rules ignore
        them.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement choose(kernel_config)")

    def __init__(self, kernel_config, **kw):
        self.name = "auto"
        self.kernel_config = kernel_config
        self.chosen = self.choose(kernel_config, **kw)
        if self.chosen == "auto":
            raise RuntimeError(
                "auto resolved to itself — choose() must name a concrete "
                "backend for %r" % (self.MODEL,))
        # A hard-coded name can go stale (a rename/unregistration).  Fail
        # loudly here, exactly like the registry's own typo guard — never
        # degrade to another backend.
        allowed = set(backends_for_model(self.MODEL))
        if self.chosen not in allowed:
            raise RuntimeError(
                f"auto ({self.MODEL}): choose() returned {self.chosen!r}, "
                f"which is not registered for this model — "
                f"allowed: {sorted(allowed)}")
        self._delegate = create_backend(self.chosen, kernel_config,
                                        model=self.MODEL, **kw)

    def load_data(self, data_np):
        return self._delegate.load_data(data_np)

    def compute(self, params, handle, norm=None, return_p=True):
        return self._delegate.compute(params, handle, norm=norm,
                                      return_p=return_p)

    def free(self):
        """Idempotent; safe from ``__del__`` and from a half-built instance."""
        delegate = self.__dict__.get("_delegate")
        self.__dict__["_delegate"] = None
        if delegate is not None:
            delegate.free()

    def __getattr__(self, item):
        # Only called when normal lookup fails (so the attributes above always
        # win); ``_``-names are not forwarded, which is what keeps a missing
        # ``_delegate`` from recursing into itself.
        if item.startswith("_"):
            raise AttributeError(item)
        delegate = self.__dict__.get("_delegate")
        if delegate is None:
            raise AttributeError(item)
        return getattr(delegate, item)

    def describe(self):
        """One-line banner: the choice plus the state and shape behind it."""
        st = cuda_state()
        free = st.get("free_bytes")
        free_txt = "n/a" if free is None else f"{free / GIB:.1f} GiB"
        cc = st.get("cc")
        cc_txt = "n/a" if cc is None else "%d.%d" % cc
        try:
            n_wave, n_res, n_basis = flat_shape(self.kernel_config)
            shape_txt = (f"n_wave {n_wave}, n_res {n_res}, "
                         f"n_basis {n_basis}")
        except Exception:
            shape_txt = "shape n/a"
        return (f"auto -> {self.chosen} (free {free_txt}, cc {cc_txt}, "
                f"{shape_txt})")


@register_backend("auto", model="pwa_gen")
class AutoFlatPwaGenBackend(AutoBackend):
    """``auto`` for the flat ``pwa_gen`` model.

    Shape first (the kernels' own guards cannot be negotiated), then
    availability: multi-BW chain -> the general backend; a tile that would not
    fit the device -> the light one; no CUDA -> NumPy; otherwise the fast tile.
    Free memory is reported by ``describe()``, never decisive.
    """

    MODEL = "pwa_gen"

    def choose(self, kernel_config, **kw):
        # availability only; the flat family ALL takes `resolution_size`, so
        # the kwargs need no fork here (every pwa_gen backend accepts it)
        st = cuda_state()
        n_wave, n_res, n_basis = flat_shape(kernel_config)
        if n_res != 1:
            return "cuda_flat_pwa_gen2"      # s0e/s0c raise on multi-BW chains
        if (st["has_cuda"] and st.get("smem_optin_bytes")
                and tile_bytes(n_wave, n_basis,
                               int(kernel_config.get("n_proj", 1) or 1))
                > st["smem_optin_bytes"]):
            return "cuda_flat_pwa_gen_s0c"   # the s0e create would refuse it
        if not st["has_cuda"]:
            return "numpy_flat"
        return "cuda_flat_pwa_gen_s0e"


@register_backend("auto", model="pwa")
class AutoPWABackend(AutoBackend):
    """``auto`` for the plain projection-sum PWA model.

    The model's registered default on a GPU (``cuda_v4_pwa``) and the NumPy
    reference without one — no preference order, no faster variant.  Every
    faster member of this family is CONDITIONAL: ``cuda_v4_pwa_cache`` is
    strictly fixed m0/g0 (it raises otherwise), and ``cuda_v4_pwa_s0d`` /
    ``s0e`` / ``1bw`` all guard ``n_res == 1`` — which both of the suite's own
    configs are NOT (`tests/config_pwa.yml` and `tests/config_angle.yml` both
    have `n_res = 2`), so those would raise at construction on them.
    ``cuda_v4_pwa`` is the only unconditional one (it records `n_res` and
    handles it), and the s0 family's sample-size ordering is not knowable here
    anyway.  ``resolution_size > 1`` forks to ``cuda_v5_pwa`` / ``cpu_v5_pwa``:
    the v4 backend and the NumPy reference have no grouped objective and
    raise ``TypeError`` on that kwarg.  Free memory is reported by
    ``describe()``, never decisive.
    """

    MODEL = "pwa"

    def choose(self, kernel_config, **kw):
        # `resolution_size` is only ever a CONSTRUCTOR kwarg, never a config
        # key, so it is read from `**kw`.  Measured: `cuda_v4_pwa` and
        # `numpy_pwa` have no grouped objective at all — `create_backend(...,
        # resolution_size=4)` raises TypeError on the kwarg — while `cuda_v5_pwa`
        # / `cpu_v5_pwa` implement it.  A grouped fit therefore cannot use the
        # defaults.
        rs = int(kw.get("resolution_size", 1) or 1)
        if not cuda_state()["has_cuda"]:
            return "cpu_v5_pwa" if rs > 1 else "numpy_pwa"
        return "cuda_v5_pwa" if rs > 1 else "cuda_v4_pwa"

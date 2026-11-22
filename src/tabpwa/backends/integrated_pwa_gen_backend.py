"""``integrated_pwa_gen`` — Gram-matrix norm for the flat ``pwa_gen`` config.

Nested spec: ``base`` (the data-NLL backend, default
``cuda_flat_pwa_gen_s0e2``) and ``mc_batch`` (the Gram's slice size).

The Gram is built over the PHASE-SPACE bundle only, at a PINNED ``coeff``:
``D[k,k'] = Σ_e w_e Σ_p conj(c_{e,p,k})·c_{e,p,k'}`` — same algebra, TWO
builders:

* **GPU, preferred**: when the base's kernel exposes ``compute_gram`` (that is
  ``cuda_flat_pwa_gen_ampcache``, whose cache already holds the finished
  per-wave amplitude ``Afix = Amp·conj(D)/|D|²``), the Gram is reduced ON the
  device from that cache — ``cuda_v4_pwa_cache.compute_gram``'s role.  The
  cache has to be filled for the data NLL anyway, so the build adds no
  amplitude work, only the reduction.
* **numpy, the oracle and the fallback**: :class:`tabpwa.integrated_pwa_gen.
  IntegratedPWAFlat`, which calls ``numpy_flat``'s own forward chain, so the
  algebra is not written twice.  It runs for a base with no GPU Gram AND is
  what the GPU Gram is checked against.

**The build STREAMS the phase space** (``integrated_pwa``'s pattern): the
bundle is sliced into ``mc_batch``-event batches, each batch is loaded into the
base, Gram-reduced and FREED before the next, so the peak device memory is ONE
batch's cache (``mc_batch · 8·n_proj·n_wave`` B — 2.7 MB at 5000 events on the
CTPWA shape) plus the N×N accumulator, never ``ne·n_wave·n_proj·8``.  The
weight vector is sliced with the same bounds; the full array is never uploaded.
The numpy path keeps the whole sample because it is host memory.

A batch boundary must not change the answer: the partials are summed in order
and the last step is ``(D + D†)/2``, so ``mc_batch=5000`` and ``mc_batch=137``
agree to the cache's floor (gated in ``tests/test_integrated_pwa_gen.py``).

The restriction is enforced, not assumed. ``coeff`` is what ``D`` is a function
of, so a caller that frees a basis weight would get a norm computed at the old
weights and a gradient of zero — a silently wrong fit. This backend therefore
compares ``params["coeff"]`` against the vector the Gram was built at — from
the SECOND call on, on BOTH branches (the norm branch in ``_ensure_gram`` and
the data/plot branch in ``_check_pinned``, one backend-level snapshot) — and
raises (the ``cuda_v4_pwa_cache`` pattern) instead of zeroing. The FIRST call
of a fresh backend DEFINES the point, so it cannot raise; that is the fitter
starting up, not a weight that moved. ``integrated_pwa`` zeroes ``m0/g0`` by
comparison; that is fine there only because its registration scopes it to a
model where those are pinned by construction.

Which BUILDER runs is a separate rule: the GPU Gram only when the base's kernel
has ``compute_gram`` AND its own ``gram_ok`` (its shared tile is
``2·N² + 8·N·P`` doubles, so a wide wave axis exceeds the device's opt-in
ceiling); otherwise — or when a GPU call fails at run time — the numpy oracle
builds ``D`` and a warning names the path taken.

``resolution_size`` is accepted and FORWARDED to the base: grouping changes
the data NLL (``Q = -Σ_g W_g·log(S_g/W_g)``), not the normalisation — ``norm``
is the same phase-space integral ``Σ_e w_e P_e``, and it is the base that
applies the group.  Refusing it would be a restriction the maths does not have
(the registry-wide contract test caught exactly that).
"""

import warnings

import numpy as np

from tabpwa.integrated_pwa_gen import IntegratedPWAFlat

from . import create_backend
from .core import ComputeBackend, register_backend

DEFAULT_BASE = "cuda_flat_pwa_gen_s0e2"


@register_backend("integrated_pwa_gen", model="pwa_gen", tags=("delegate",))
class IntegratedPWAFlatBackend(ComputeBackend):
    """Hyper backend: wave Gram norm + a flat ``pwa_gen`` base for the data NLL."""

    def __init__(self, kernel_config, base=DEFAULT_BASE, mc_batch=5000,
                 model=None, resolution_size=1, **kw):
        rsize = int(resolution_size)
        if rsize < 1:
            raise ValueError(
                f"integrated_pwa_gen: resolution_size={rsize} is invalid — a "
                f"group has at least one event")
        self.resolution_size = rsize
        self._kernel_config = kernel_config
        self.mc_batch = int(mc_batch)
        self._gram = IntegratedPWAFlat(kernel_config, chunk=self.mc_batch)
        self._last_dnorm = None
        #: the ``coeff`` vector the Gram was built at — the BACKEND-level
        #: snapshot, so the data branch guards against the same point the norm
        #: branch does (``None`` = no Gram yet: the first call defines it)
        self._gram_coeff = None
        if isinstance(base, ComputeBackend):
            self.base = base
        else:
            self.base = create_backend(base, kernel_config, model=model,
                                       resolution_size=self.resolution_size)

    # ── per-load bundle (never the context: gotcha 15) ───────────────────
    class _Bundle:
        def __init__(self, data_np):
            self.data = dict(data_np)
            self.handle = None          # the BASE handle, lazily created
            self.D = None
            self.coeff = None           # the vector D was built at

        def free(self):
            if self.handle is not None and hasattr(self.handle, "free"):
                self.handle.free()
                self.handle = None
            self.D = None

    def load_data(self, data_np):
        return self._Bundle(data_np)

    def _base_handle(self, bundle):
        if bundle.handle is None:
            bundle.handle = self.base.load_data(bundle.data)
        return bundle.handle

    # ── the Gram (phsp bundle only) ──────────────────────────────────────
    def _ensure_gram(self, bundle, coeff):
        if bundle.D is not None and np.array_equal(coeff, bundle.coeff):
            return bundle.D
        if bundle.D is not None:
            raise ValueError(
                "integrated_pwa_gen: coeff moved after the Gram was built — the "
                "basis weights must stay pinned; use "
                f"{DEFAULT_BASE} when they float")
        # the vector the Gram (and, on the GPU path, the base's cache) is
        # pinned to: the SAME pinning the base enforces, so the two cannot
        # disagree about which point the norm belongs to.
        bundle.D = self._build_gram(bundle.data, coeff)
        bundle.coeff = np.array(coeff, np.float64, copy=True)
        # ... and the BACKEND-level snapshot: the data branch must hold the
        # SAME point, or a `P` at the new weights would be fitted against a
        # `norm` at the old ones with nothing raising.
        self._gram_coeff = bundle.coeff.copy()
        return bundle.D

    def _build_gram(self, data, coeff):
        """``D`` from the base's GPU Gram when it CAN run, else the numpy oracle.

        The GPU builder is only offered ``gram_ok`` shapes: its tile is
        ``2·N² + 8·N·P`` doubles of SHARED, so a wide wave axis (``N = 76``
        needs 102 144 B on a 101 376 B box; ``N = 200`` no device) is refused at
        construction and this falls back rather than raising — the norm is the
        same integral either way, the fp64 oracle is just slower.

        A shape can also be refused BETWEEN the check and the call (a kernel
        that fails to opt in on this device, a launch error): that raises, and
        is caught here for the same reason, with the numpy path NAMED in the
        message so the slowdown is never silent.
        """
        kern = getattr(self.base, "kernel", None)
        gpu_gram = getattr(kern, "compute_gram", None)
        if gpu_gram is not None and bool(getattr(kern, "gram_ok", True)):
            try:
                return self._gpu_gram(data, coeff, gpu_gram)
            except RuntimeError as exc:
                warnings.warn(
                    "integrated_pwa_gen: the GPU Gram build failed "
                    f"({exc}); falling back to the numpy builder "
                    "tabpwa.integrated_pwa_gen.IntegratedPWAFlat — the norm is "
                    "the same integral, the build is slower",
                    RuntimeWarning, stacklevel=2)
        return self._gram.gram(data, coeff)

    def _gpu_gram(self, data, coeff, gpu_gram):
        """Build ``D`` from the base's OWN cache, ONE batch on the device.

        ``integrated_pwa._ensure_gram``'s pattern, for the same reason: the
        whole phase space never resides on the device.  Each batch is loaded,
        reduced and freed before the next, so the peak is one batch's cache
        plus the ``N×N`` accumulator.  ``coeff`` travels with every call: the
        base's cache is filled on a handle's FIRST compute (its single
        lifecycle), which for a streamed build is the first batch's Gram call.
        """
        ne = (len(np.asarray(data["weight"])) if "weight" in data
              else len(np.asarray(data["mass"])))
        w = (np.asarray(data["weight"], np.float64) if "weight" in data
             else None)
        batch = max(1, int(self.mc_batch))
        D = None
        for b0 in range(0, ne, batch):
            b1 = min(b0 + batch, ne)
            sub = {k: (v[b0:b1] if isinstance(v, np.ndarray) and v.ndim >= 1
                       and v.shape[0] == ne else v)
                   for k, v in data.items()}
            h = self.base.load_data(sub)
            try:
                M = gpu_gram(h, None if w is None else w[b0:b1], coeff)
            finally:
                h.free()
            D = M if D is None else D + M
        if D is None:
            n = int(self._gram.n_wave)
            D = np.zeros((n, n), np.complex128)
        return (D + D.conj().T) / 2.0

    # ── ComputeBackend interface ─────────────────────────────────────────
    @property
    def dnorm_on_gpu(self):
        kern = getattr(self.base, "kernel", None)
        return bool(getattr(kern, "dnorm_on_gpu", False))

    def compute(self, params, data_handle, norm=None, return_p=True):
        coeff = np.asarray(params["coeff"], np.float64)
        if norm is not None or return_p:
            # data NLL (norm given) or the per-event P (plot path): the BASE.
            # The guard runs first — a cheap `array_equal` against the vector
            # the Gram was built at, NOT a Gram build from a data handle (a
            # data bundle's `D` is never set, and building one here would
            # integrate the DATA where the norm must integrate the phase
            # space). A moved `coeff` reaching this branch would hand the
            # fitter a `P` at the new weights and a `norm` at the old ones,
            # silently.
            self._check_pinned(coeff)
            h = self._base_handle(data_handle)
            Q, grads, P = self.base.compute(params, h, norm=norm,
                                            return_p=return_p)
            if "coeff" in grads:
                grads["coeff"] = np.zeros_like(np.asarray(grads["coeff"]))
            return Q, grads, P

        # the Gram norm over the phsp bundle
        D = self._ensure_gram(data_handle, coeff)
        ck = np.asarray(params["ck"], np.complex128)
        norm_val = self._gram.norm(ck, D)
        self._last_dnorm = self._gram.dnorm(ck, D)
        # ∂norm/∂coeff = 0: the weights are pinned by contract, and the guard
        # above is what makes that true rather than assumed.
        grads = {"ck": self._last_dnorm,
                 "coeff": np.zeros_like(coeff)}
        return float(norm_val), grads, None

    def _check_pinned(self, coeff):
        """Raise when ``coeff`` is not the vector the Gram was built at.

        ``_gram_coeff`` is None until the first Gram exists, so a fresh backend
        cannot raise: its FIRST call DEFINES the point (from the second call on
        the snapshot is strict).  This is the data-branch half of the guard;
        ``_ensure_gram`` is the norm-branch half and they share the snapshot.
        """
        if self._gram_coeff is None:
            return
        if not np.array_equal(np.asarray(coeff, np.float64), self._gram_coeff):
            raise ValueError(
                "integrated_pwa_gen: coeff moved after the Gram was built — the "
                "basis weights must stay pinned; use "
                f"{DEFAULT_BASE} when they float")

    def free(self):
        try:
            self.base.free()
        except Exception:
            pass

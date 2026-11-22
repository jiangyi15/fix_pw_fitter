"""``integrated_pwa_gen`` — the flat config's Gram-matrix norm.

Gates:

1. **the value is the reference's own integral** — ``Re(ck† D ck)`` equals
   ``numpy_flat``'s ``Σ_e w_e P_e`` on the same phase-space sample, at a
   pinned ``coeff``.  The Gram is built from ``numpy_flat``'s forward chain,
   so this is an identity to check, not a second implementation to trust.
2. **the gradient** — ``∂norm/∂ck`` equals ``numpy_flat``'s ``grads["ck"]`` on
   the same norm-only call (the Wirtinger convention both kernels publish).
3. **the guard** — a basis weight that moves AFTER the Gram exists raises on
   BOTH branches (the norm branch and the data/plot branch), from the SECOND
   call on: the first call of a fresh backend DEFINES the point, so it cannot
   raise.  Zeroing instead would let a fit run with a dead parameter and a
   stale norm.
4. **``resolution_size`` is accepted and FORWARDED** to the base (grouping
   changes the data NLL, not the normalisation; a test pins rsize 1 == 17).
5. **end to end through ``get_nll``** — the delegate's norm matches the base's
   ``Σ w P`` at the fp32 floor, and a real ``fit(maxiter=3)`` through the
   delegate tracks the same fit on ``cuda_flat_pwa_gen_s0e2`` (the
   ``dNLL/dnorm × dnorm/dck`` chain has no other gate).
6. **the GPU Gram is PREFERRED** when the base has one: it streams one
   ``mc_batch`` batch per call, a batch boundary is invisible, and ``D`` / the
   norm agree with the numpy oracle at the cache's floor.
7. **``gram_ok`` gates the builder** — a wave count whose Gram tile exceeds the
   device's shared-memory opt-in takes the numpy path (the GPU builder is not
   called at all) and still produces the right norm.
"""

import numpy as np
import pytest

from tabpwa.backends import create_backend
from tabpwa.backends.core import ComputeBackend
from tabpwa.integrated_pwa_gen import IntegratedPWAFlat
from tabpwa import Fitter
from _flat_grid import _in_grid

CFG = "tests/config_pwa_gen.yml"
CFG_KM = "tests/config_pwa_gen_km.yml"
CFG_MB = "tests/config_pwa_gen_mb.yml"


def _setup(cfg=CFG, n=64, seed=0):
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    return f, kc, params, _in_grid(kc, n, seed)


def test_the_gram_reproduces_the_reference_integral():
    """``Re(ck† D ck)`` == numpy_flat's ``Σ w P`` — the same forward chain."""
    _f, kc, params, data = _setup()
    ref = create_backend("numpy_flat", kc, model="pwa_gen")
    try:
        q_ref, _g, _P = ref.compute(params, ref.load_data(data), norm=None)
    finally:
        ref.free()
    gram = IntegratedPWAFlat(kc)
    D = gram.gram(data, params["coeff"])
    ck = np.asarray(params["ck"], complex)
    q_gram = gram.norm(ck, D)
    assert abs(q_gram - q_ref) <= 1e-12 * max(abs(q_ref), 1.0), (
        q_gram, q_ref, abs(q_gram - q_ref) / max(abs(q_ref), 1.0))


def test_the_gram_gradient_is_the_reference_ck_gradient():
    """``∂norm/∂ck`` in the convention both kernels publish."""
    _f, kc, params, data = _setup(CFG_KM)
    ref = create_backend("numpy_flat", kc, model="pwa_gen")
    try:
        _q, g_ref, _P = ref.compute(params, ref.load_data(data), norm=None)
    finally:
        ref.free()
    gram = IntegratedPWAFlat(kc)
    D = gram.gram(data, params["coeff"])
    ck = np.asarray(params["ck"], complex)
    g = gram.dnorm(ck, D)
    worst = float(np.abs(g - g_ref["ck"]).max())
    scale = max(float(np.abs(g_ref["ck"]).max()), 1e-300)
    assert worst / scale < 1e-10, (worst / scale, g[:3], g_ref["ck"][:3])


def test_a_moved_basis_weight_raises_instead_of_reusing_the_gram():
    """The Gram is a function of ``coeff``: a freed weight must not be silent."""
    f, kc, params, data = _setup()
    be = create_backend("integrated_pwa_gen", kc, model="pwa_gen")
    h = be.load_data(data)
    try:
        be.compute(params, h, norm=None, return_p=False)   # the fitter's norm call
        moved = dict(params)
        c = np.asarray(params["coeff"], np.float64).copy()
        c[0] += 1e-6
        moved["coeff"] = c
        with pytest.raises(ValueError) as ei:
            be.compute(moved, h, norm=None, return_p=False)
        msg = str(ei.value)
        assert "coeff moved after the Gram was built" in msg, msg
        assert "must stay pinned" in msg, msg
        # ... and strictly from the SECOND call: a fresh backend's first
        # compute DEFINES the point, so it cannot raise.  Here it builds the
        # Gram first, THEN moves — the fitter-starting-over case is the
        # norm-only call above, unchanged.
        be2 = create_backend("integrated_pwa_gen", kc, model="pwa_gen")
        h2 = be2.load_data(data)
        try:
            be2.compute(params, h2, norm=None, return_p=False)
            with pytest.raises(ValueError):
                be2.compute(moved, h2, norm=None, return_p=False)
        finally:
            be2.free()
        # ... so a FRESH backend handed the moved vector on its first call
        # legitimately defines the Gram there (nothing has been pinned yet)
        be3 = create_backend("integrated_pwa_gen", kc, model="pwa_gen")
        h3 = be3.load_data(data)
        try:
            Q3 = be3.compute(moved, h3, norm=None, return_p=False)[0]
            assert np.isfinite(Q3) and Q3 > 0.0, Q3
            assert np.array_equal(be3._gram_coeff,
                                  np.asarray(moved["coeff"], np.float64))
        finally:
            be3.free()
    finally:
        be.free()


def test_the_data_branch_holds_the_same_point_as_the_norm():
    """A moved ``coeff`` on the DATA (``return_p``) branch raises too.

    The data bundle has no ``D`` of its own — the Gram integrates the PHASE
    SPACE — so the guard is a cheap ``array_equal`` against the backend-level
    snapshot, not a Gram build.  Without it a plot/``return_p`` call at the new
    weights would be drawn against a ``norm`` at the old ones, silently.
    """
    f, kc, params, data = _setup()
    be = create_backend("integrated_pwa_gen", kc, model="pwa_gen")
    try:
        h_ph = be.load_data(data)
        h_data = be.load_data(data)
        be.compute(params, h_ph, norm=None, return_p=False)     # builds the Gram
        assert be._gram_coeff is not None
        moved = dict(params)
        c = np.asarray(params["coeff"], np.float64).copy()
        c[0] += 1e-9
        moved["coeff"] = c
        for kw in ({"norm": 3.0, "return_p": True}, {"norm": None,
                                                     "return_p": True}):
            with pytest.raises(ValueError) as ei:
                be.compute(moved, h_data, **kw)
            assert "coeff moved after the Gram" in str(ei.value)
        # the pinned vector still works on that branch
        Q, grads, P = be.compute(params, h_data, norm=3.0, return_p=True)
        assert np.isfinite(Q) and P is not None and P.size == len(P)
        assert np.array_equal(grads["coeff"], np.zeros_like(grads["coeff"]))
        h_ph.free()
        h_data.free()
    finally:
        be.free()


def test_resolution_size_is_accepted_and_reaches_the_base():
    """Grouping changes the DATA NLL, not the norm — so it is forwarded."""
    _f, kc, _params, _data = _setup()
    be = create_backend("integrated_pwa_gen", kc, model="pwa_gen",
                        resolution_size=17)
    try:
        assert be.resolution_size == 17
        assert be.base.resolution_size == 17
    finally:
        be.free()
    # rsize < 1 is an error, like every other pwa_gen backend
    with pytest.raises(ValueError):
        create_backend("integrated_pwa_gen", kc, model="pwa_gen",
                       resolution_size=0)


def test_the_gram_norm_is_unchanged_by_grouping():
    """``norm`` is the same phase-space integral at rsize 1 and 17."""
    f, kc, params, data = _setup()
    n = {}
    for rs in (1, 17):
        be = create_backend("integrated_pwa_gen", kc, model="pwa_gen",
                            resolution_size=rs)
        try:
            n[rs] = be.compute(params, be.load_data(data), norm=None,
                               return_p=False)[0]
        finally:
            be.free()
    assert abs(n[1] - n[17]) <= 1e-12 * max(abs(n[1]), 1.0), n


def test_get_nll_through_the_delegate_matches_the_base():
    """End to end: norm from the Gram, data NLL from the base."""
    f, kc, params, data = _setup()
    x = f.initial_values(seed=1)
    be = create_backend("integrated_pwa_gen", kc, model="pwa_gen")
    base = create_backend("cuda_flat_pwa_gen_s0e2", kc, model="pwa_gen")
    try:
        hd = be.load_data(data)
        hv = base.load_data(data)
        n_gram, _g, _p = be.compute(params, hd, norm=None, return_p=False)
        n_base, _g2, _p2 = base.compute(params, hv, norm=None, return_p=False)
        rel = abs(n_gram - n_base) / max(abs(n_base), 1e-300)
        assert rel < 5e-6, (n_gram, n_base, rel)   # the fp32 filter set
    finally:
        be.free()
        base.free()


# ── the GPU Gram: preferred when the base has one ──────────────────────
#
# ``cuda_flat_pwa_gen_ampcache`` caches the finished per-wave amplitude, so its
# ``compute_gram`` reduces that cache instead of walking ``numpy_flat``'s
# forward chain.  The delegate prefers it and keeps the numpy builder as the
# fallback AND the oracle.  The build STREAMS the phase space (one
# ``mc_batch``-event batch on the device at a time), so a batch boundary must
# not change the answer and the peak memory must be one batch, not one sample.

GPU_BASE = "cuda_flat_pwa_gen_ampcache"
#: the fp32 cache's floor on ``D`` (MEASURED 7.2e-8 relative at this anchor;
#: ~4x margin) and on the norm the family already gates at (5e-6)
GRAM_FLOOR = 1e-6


def _gpu_delegate(kc, mc_batch=5000, base=GPU_BASE):
    return create_backend("integrated_pwa_gen", kc, model="pwa_gen",
                          base=base, mc_batch=mc_batch)


def test_the_delegate_prefers_the_bases_gpu_gram():
    """The GPU path RUNS when the base has ``compute_gram`` — and agrees with
    the numpy builder to the cache's floor."""
    _f, kc, params, data = _setup(n=200)
    calls = []
    be = _gpu_delegate(kc, mc_batch=64)
    kernel = be.base.kernel
    assert hasattr(kernel, "compute_gram"), "the base lost its GPU Gram"
    orig = kernel.compute_gram

    def spy(*a, **kw):
        calls.append(a[0].ne)
        return orig(*a, **kw)

    kernel.compute_gram = spy
    try:
        h = be.load_data(data)
        n_gpu = be.compute(params, h, norm=None, return_p=False)[0]
        assert calls, "the delegate used the numpy builder with a GPU base"
        # the streamed build asks for one batch per call: 64 events at a time,
        # the last batch being the remainder (200 = 64 + 64 + 64 + 8)
        exp, b0, ne = [], 0, len(np.asarray(data["weight"]))
        while b0 < ne:
            exp.append(min(64, ne - b0))
            b0 += 64
        assert calls == exp, (calls, exp)
        n_np = IntegratedPWAFlat(kc).norm(
            np.asarray(params["ck"], complex),
            IntegratedPWAFlat(kc).gram(data, params["coeff"]))
        assert abs(n_gpu - n_np) / abs(n_np) < 1e-6, (n_gpu, n_np)
        h.free()
    finally:
        be.free()


def test_the_delegate_falls_back_to_numpy_without_a_gpu_gram():
    """A base with no ``compute_gram`` (a stub) takes the oracle path."""
    _f, kc, params, data = _setup()

    class _Stub(ComputeBackend):
        """The minimum a delegate base must offer; NO ``compute_gram``."""

        class kernel:
            pass

        def load_data(self, d):
            raise AssertionError("the fallback must not load the base")

        def compute(self, *a, **kw):
            raise AssertionError("the fallback must not compute the base")

        def free(self):
            pass

    be = _gpu_delegate(kc, base=_Stub())
    try:
        h = be.load_data(data)
        n_stub = be.compute(params, h, norm=None, return_p=False)[0]
        n_np = IntegratedPWAFlat(kc).norm(
            np.asarray(params["ck"], complex),
            IntegratedPWAFlat(kc).gram(data, params["coeff"]))
        assert abs(n_stub - n_np) <= 1e-12 * max(abs(n_np), 1.0), (n_stub, n_np)
    finally:
        be.free()


def test_a_gpu_gram_failure_at_run_time_falls_back_to_numpy():
    """A shape refused BETWEEN the ``gram_ok`` check and the call falls back.

    ``gram_ok`` is the pre-check, but a kernel can still fail (an opt-in that
    did not take, a launch error): that raises ``RuntimeError``, and the
    backend must take the numpy builder it holds with a warning that NAMES it,
    not propagate — the norm is the same integral either way.
    """
    _f, kc, params, data = _setup()

    class _H:
        def __init__(self, ne):
            self.ne = ne

        def free(self):
            pass

    class _Stub(ComputeBackend):
        class kernel:
            gram_ok = True                     # the pre-check passes ...

            @staticmethod
            def compute_gram(h, w, coeff):
                raise RuntimeError("tile refused at launch")   # ... the call fails

        def load_data(self, d):
            return _H(len(np.asarray(d["weight"])))

        def compute(self, *a, **kw):
            raise AssertionError("the fallback must not compute the base")

        def free(self):
            pass

    be = _gpu_delegate(kc, base=_Stub())
    try:
        h = be.load_data(data)
        with pytest.warns(RuntimeWarning, match="IntegratedPWAFlat"):
            n_stub = be.compute(params, h, norm=None, return_p=False)[0]
        n_np = IntegratedPWAFlat(kc).norm(
            np.asarray(params["ck"], complex),
            IntegratedPWAFlat(kc).gram(data, params["coeff"]))
        assert abs(n_stub - n_np) <= 1e-12 * max(abs(n_np), 1.0), (n_stub, n_np)
    finally:
        be.free()


def test_a_batch_boundary_does_not_change_the_gram():
    """``mc_batch`` 5000 and 137 build the SAME ``D`` — the build is a sum of
    per-batch partials, and the batching is not allowed to be visible."""
    _f, kc, params, data = _setup(n=400)
    D = {}
    for batch in (5000, 137):
        be = _gpu_delegate(kc, mc_batch=batch)
        try:
            b = be.load_data(data)
            D[batch] = be._ensure_gram(b, np.asarray(params["coeff"], float))
            b.free()
        finally:
            be.free()
    ne = len(np.asarray(data["weight"]))
    assert ne % 137 != 0, "the sample must actually straddle a 137 batch"
    rel = np.abs(D[5000] - D[137]).max() / np.abs(D[5000]).max()
    assert rel < 1e-12, rel


def test_the_streamed_gram_is_the_numpy_gram():
    """The delegate's GPU ``D`` against the fp64 oracle, at the cache's floor."""
    _f, kc, params, data = _setup()
    be = _gpu_delegate(kc, mc_batch=64)
    try:
        b = be.load_data(data)
        Dg = be._ensure_gram(b, np.asarray(params["coeff"], float))
        Dn = IntegratedPWAFlat(kc).gram(data, params["coeff"])
        rel = np.abs(Dg - Dn).max() / np.abs(Dn).max()
        assert rel < GRAM_FLOOR, rel
        assert np.abs(Dg - Dg.conj().T).max() / np.abs(Dg).max() < 1e-12
        b.free()
    finally:
        be.free()


def test_the_gpu_gram_norm_matches_the_free_coefficient_kernel():
    """``integrated_pwa_gen``'s norm (GPU built) == ``s0e2``'s ``Σ w P``.

    This is the number the fit consumes; MEASURED 1.8e-8 relative at this
    anchor, gated at the family's fp32 floor.
    """
    _f, kc, params, data = _setup()
    be = _gpu_delegate(kc)
    base = create_backend("cuda_flat_pwa_gen_s0e2", kc, model="pwa_gen")
    try:
        n_gpu = be.compute(params, be.load_data(data), norm=None,
                           return_p=False)[0]
        n_ref = base.compute(params, base.load_data(data), norm=None,
                             return_p=False)[0]
        rel = abs(n_gpu - n_ref) / abs(n_ref)
        assert rel < 5e-6, (n_gpu, n_ref, rel)   # the fp32 filter set
    finally:
        be.free()
        base.free()


def test_a_moved_coefficient_raises_through_the_gpu_gram_too():
    """The pinning guard is the delegate's, unchanged by which builder ran."""
    _f, kc, params, data = _setup()
    be = _gpu_delegate(kc, mc_batch=64)
    try:
        h = be.load_data(data)
        be.compute(params, h, norm=None, return_p=False)
        moved = dict(params)
        c = np.asarray(params["coeff"], np.float64).copy()
        c[0] += 1e-6
        moved["coeff"] = c
        with pytest.raises(ValueError, match="coeff moved after the Gram"):
            be.compute(moved, h, norm=None, return_p=False)
        h.free()
    finally:
        be.free()


def test_n_res_two_parity_through_the_gpu_gram():
    """``config_pwa_gen_mb.yml``: the Gram path is denominator-agnostic."""
    f = Fitter(CFG_MB)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    data = _in_grid(kc, 300, seed=3)
    be = _gpu_delegate(kc, mc_batch=100)
    base = create_backend("cuda_flat_pwa_gen2", kc, model="pwa_gen")
    try:
        n_gpu = be.compute(params, be.load_data(data), norm=None,
                           return_p=False)[0]
        n_ref = base.compute(params, base.load_data(data), norm=None,
                             return_p=False)[0]
        rel = abs(n_gpu - n_ref) / abs(n_ref)
        assert rel < 5e-6, (n_gpu, n_ref, rel)
    finally:
        be.free()
        base.free()


# ── gram_ok: the Gram's shared tile has a CEILING ──────────────────────
#
# The GPU Gram's tile is ``2·N² + 8·N·P`` doubles of SHARED memory, so a wave
# axis wide enough exceeds the device's opt-in limit (101 376 B on this box:
# N=34/2 needs 22 848 B and fits, N=76/2 needs 102 144 B and does not).  The
# Gram is OPTIONAL for the ampcache kernel, so the shape only disables it — and
# the delegate must consult that and take the numpy builder it holds, instead
# of paying a cache fill to discover the refusal from a failed call.

def _too_wide_shape(kc, n_proj):
    """The smallest whole multiple of this config's wave count above the box."""
    from tabpwa.backends.auto_backend import cuda_state
    optin = (cuda_state() or {}).get("smem_optin_bytes")
    assert optin, "no CUDA device / opt-in limit to test the ceiling against"
    n0 = int(np.asarray(kc["fl_table"]).shape[0])
    n = n0
    while ((2 * n * n + 8 * n * int(n_proj)) * 8) <= int(optin):
        n += n0
    return n, int(optin)


def test_a_shape_above_the_smem_ceiling_takes_the_numpy_builder():
    """``gram_ok`` False => the numpy oracle, and the norm is still right."""
    from benchmark_flat_vs_pwa import tile_flat
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    n_wave0 = int(np.asarray(kc["fl_table"]).shape[0])
    n_proj = (int(np.asarray(kc["matrix_angle"]).shape[1]) // n_wave0)
    n_target, optin = _too_wide_shape(kc, n_proj)
    big = tile_flat(kc, n_target, n_proj)
    n_wave = int(np.asarray(big["fl_table"]).shape[0])
    assert n_wave == n_target, (n_wave, n_target)
    # the tiled config is the same denominators repeated, so `ck` tiles too
    reps = n_wave // n_wave0
    bparams = {"ck": np.tile(np.asarray(params["ck"], complex), reps),
               "coeff": np.asarray(params["coeff"], np.float64)}
    data = _in_grid(big, 64, seed=0)

    be = _gpu_delegate(big, mc_batch=64)
    kernel = be.base.kernel
    try:
        assert kernel.gram_bytes() > optin, (kernel.gram_bytes(), optin)
        assert kernel.gram_ok is False, (
            "the shape fits — this box's ceiling moved, so the gate did not "
            "exercise the fallback")
        calls = []
        orig = kernel.compute_gram

        def spy(*a, **kw):
            calls.append(a[0].ne)
            return orig(*a, **kw)

        kernel.compute_gram = spy
        n_be = be.compute(bparams, be.load_data(data), norm=None,
                          return_p=False)[0]
        assert calls == [], "the GPU builder ran on a shape it cannot serve"
        # the numpy path ran instead, and its answer is the real integral:
        # compare against a FREE-coefficient kernel's Σ w P
        ref = create_backend("cuda_flat_pwa_gen_s0e2", big, model="pwa_gen")
        try:
            n_ref = ref.compute(bparams, ref.load_data(data), norm=None,
                                return_p=False)[0]
        finally:
            ref.free()
        rel = abs(n_be - n_ref) / abs(n_ref)
        assert rel < 5e-6, (n_be, n_ref, rel)      # the fp32 filter set
    finally:
        be.free()


# ── end to end: the fitters' own chain ─────────────────────────────────

def test_a_real_fit_runs_through_the_delegate():
    """``Fitter(backend="integrated_pwa_gen")`` + BFGS vs plain ``s0e2``.

    The one place ``dNLL/dnorm × dnorm/dck`` is exercised: the norm comes from
    the Gram (``∂norm/∂ck = conj(D ck)``), the data NLL and its gradient from
    the base, and ``grads["coeff"]`` must be zero on both.  A sign or
    convention error in the Gram's gradient shows up as a fit that walks the
    other way, which no single-``get_nll`` gate can see.
    """
    import warnings

    from scipy.optimize import OptimizeWarning
    FIX = ("k2_mass0", "k2_mass1", "k2_width0", "k2_width1")

    def build(name):
        f = Fitter(CFG_KM, backend=name)
        f.apply_constrains()
        res = f.build_params(f.initial_values(seed=1))[1]
        f.set_fixed({n: float(res[n]) for n in FIX})
        x0 = f.initial_values(seed=1)
        kc = f.kernel_config
        f.set_phsp(_in_grid(kc, 300, seed=7))
        f.set_data(_in_grid(kc, 300, seed=8))
        return f, x0

    fa, x0 = build("integrated_pwa_gen")
    fb, xb = build("cuda_flat_pwa_gen_s0e2")
    try:
        assert len(x0) == 2, list(fa.fwd_names())
        na, ga = fa.get_nll(x0)
        nb, gb = fb.get_nll(xb)
        assert np.isfinite(ga).all() and np.abs(ga).max() > 0.0
        rel = abs(na - nb) / max(abs(nb), 1.0)
        assert rel < 1e-7, (na, nb, rel)
        assert np.abs(ga - gb).max() < 1e-6 * max(np.abs(gb).max(), 1.0), (
            ga, gb)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            ra = fa.fit(x0, maxiter=3)
            rb = fb.fit(xb, maxiter=3)
        assert np.isfinite(ra.fun)
        # MEASURED: |Δfun| 1.2e-4 at NLL ~614 (2.0e-7 relative), max|Δx|
        # 2.7e-4, |ΔNLL| 1.4e-6 after three iterations.
        #
        # What this gap IS: the two sides differ in the norm-gradient path (the
        # Gram's analytic ∂norm/∂ck in fp64 against `s0e2`'s own fp32-cached
        # dnorm), and three iterations leave |grad| ~ 0.6, so the endpoint gap
        # is |grad|·|Δx| on a trajectory that has not converged — the last
        # digit of x is not determined yet.
        #
        # What it is NOT: a bias the oracle could cancel.  It was +7.6e-5 while
        # the numpy Gram was built on `numpy_flat`'s LINEAR interpolation and is
        # -1.2e-4 now that it is built on Catmull-Rom — the sign FLIPPED while
        # the Gram's agreement with `s0e2`'s own ΣwP improved 5.7x (2.9e-8 ->
        # 5.0e-9).  A residual error that flips sign with a 6x better norm
        # cannot be the thing setting the bound, so the bound (3e-4, 2.5x) is
        # sized for a non-converged endpoint, not tuned to either oracle.
        assert abs(ra.fun - rb.fun) < 3e-4, (ra.fun, rb.fun)
        assert np.abs(np.asarray(ra.x) - np.asarray(rb.x)).max() < 1e-3
    finally:
        fa.backend.free()
        fb.backend.free()


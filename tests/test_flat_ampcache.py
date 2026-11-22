"""The ``ampcache`` variants' own claims — the dtype statement, and nothing else.

``cuda_flat_pwa_gen_ampcache`` and ``cuda_flat_pwa_gen_ampcache32`` are one
kernel at two contraction widths:

    FILL        fp64 (both)   Afix[e,k,p] = Amp[e,slot(k),p]·conj(D_k)/|D_k|²,
                              once per handle; the stored cache is identical
    ITERATION   fp64 / fp32   A[e,p] = Σ_k ck_k·Afix[e,k,p] and
                              dck_k = Σ_{e,p} dA[e,p]·Afix[e,k,p]
    REDUCTIONS  fp64 (both)   Q/the NLL, P_e's Σ_p, the group sums, dnorm, g_ck
    GRAM        fp64 (both)   the parent's reduction over the same cache

NO conjugation on ``Afix`` in ``dck``: ``dA`` is already ``dQ/dP·conj(A)``
(``numpy_flat``'s ``dQ_dA = dQ_dP·conj(A)``), so a second conjugate would be
``|A|²`` and the fit would descend in the wrong direction.

The PINNED contract — registration/tag, case-1 isolation, the parity floors on
the shared fixtures, ``grads["coeff"] == 0``, the moved-``coeff`` guard, the one
lifecycle, the grouped objective (``rsize > te`` included), ``n_res == 2``, the
empty sample, the Gram and the generic guards — is ``tests/_flat_pinned.py``
and is NOT restated here.  Its ``TABLES`` is imported, so the parity floors and
the FD step (including the fp32 ``1e-2``) have ONE definition; the
free-coefficient contract of ``tests/_flat_compute.py`` does not apply (it
subtracts the tag).

Every claim here is a number about this config:

1. **the second rounding** — fp32 vs its fp64 PARENT, same cache, same
   parameters.  Two implementations, one difference, so the numbers are the
   dtype claim itself and the reason the variant exists.
2. **parity on ``config_pwa_gen_same.yml``** — the one one-BW fixture the
   shared matrix excludes on purpose (``_flat_pinned.ONE_BW``: no anchor was
   measured on it there).  One resonance drives two propagators that read
   DIFFERENT mass columns, so it is a new shape, not a second copy.
3. **the DEVICE Gram's norm** — ``Re(ck† D_gpu ck)`` and ``∂norm/∂ck`` against
   a free-coefficient kernel's own ``Σ w P`` / ``∂/∂ck``, plus the Hermitian
   asymmetry bound, on BOTH variants (the Gram is fp64 in each).  This is the
   anchor the ``integrated_pwa_gen`` row rests on: the norm a fit consumes is
   the number the production kernel computes.  ``_flat_pinned`` compares the
   Gram against ``IntegratedPWAFlat`` (the oracle), never against a kernel; and
   the fp32 variant's Gram being the fp64 PARENT's reduction is its own gate,
   so "the Gram was not traded to fp32" is a number.
4. **a shape above the Gram's shared-memory ceiling still returns the norm** —
   the delegate's documented numpy fallback is a slowdown, never a wrong
   answer: the value matches a free-coefficient kernel's ``Σ w P``.  (The
   DEVICE reduction is claim 3; here ``gram_ok`` is False by construction.)
5. **the two ``Fitter`` runs** — NLL, 3 BFGS iterations and ``max|Δx|``.

plus two more that are validation rather than measurement: the ``s0f``
mask-all sibling (≤1e-13 — what makes ``benchmark_ampcache.py`` a
like-for-like) and ``den_runs``' three refusals of a ``gamma_matrix`` that
cannot be a factor product (input validation, the same kind of contract as
``resolution_size < 1``).  And ``batch_size``, the named regression behind the
retired launch-count audit: it may change how many launches a call takes and
must not re-associate the objective.

Everything else the two per-kernel files asserted (``te`` ladders, smem/tile
budgets, launch counts, CSR internals, ``kernel_stats``/0-spill, helper
source identity, symbol lists) was an implementation audit and is gone; the
parity/guard/lifecycle/Gram half of it now lives once in ``_flat_pinned``.
The one survivor of that list is the helper pin, and it changed KIND: instead of
comparing source text it compares OUTPUTS (``np.array_equal`` on a real kernel
config, ``tests/_flat_isolation.py``'s ``HELPER_COPIES``), which is what the
kernel policy actually requires of a verbatim copy.
"""
import warnings

import numpy as np
import pytest
from scipy.optimize import OptimizeWarning

from tabpwa import Fitter
from tabpwa.backends import create_backend
from tabpwa.cuda._flat_pwa_gen_ampcache import den_runs as den_runs_64
from tabpwa.cuda._flat_pwa_gen_ampcache32 import den_runs as den_runs_32
from _flat_compute import _rel, _replicate
from _flat_grid import CFG_CP, CFG_KM, CFG_MB, CFG_ONE, CFG_SAME, _in_grid
#: the shared floors, FD steps and Gram bounds — imported, never restated
from _flat_pinned import GRAM_HERM, GRAM_NORM, PARITY, TABLES

PARENT = "fp64"
FP32 = "fp32"
BOTH = (PARENT, FP32)

#: the shared matrix PLUS the fixture it excludes, which is this file's new
#: config gate (see claim 2)
MATRIX = PARITY + (CFG_SAME,)

#: this kernel vs ``s0f`` all-fixed: the SAME fused arithmetic on the SAME
#: cache bytes, which is what makes ``benchmark_ampcache.py`` a like-for-like.
#: MEASURED bit-identical at the 64-event anchor (one block either way) and
#: <= 1e-13 when the block split differs.
SIBLING_TOL = 1e-13

#: Per-variant facts.  ``float_bits`` is the table's identity column: the
#: second-rounding claim only means something BETWEEN the two widths.
VARIANTS = {
    "fp64": {
        "name": "cuda_flat_pwa_gen_ampcache",
        "float_bits": 64,
        #: the copy's own CSR builder — its refusals are its input validation
        "den_runs": den_runs_64,
        #: no second rounding to measure — this IS the fp64 side of the claim
        "second_rounding": None,
        "fitter": {
            #: an INDEPENDENT free-coefficient kernel.  This variant's parent
            #: (s0f all-fixed) cannot be driven as a ``Fitter`` backend: its
            #: ``fixed_values`` ARE the parameters, which do not exist when the
            #: backend is built — so the reference is the free-coefficient
            #: kernel the kernel's own file compared with.
            "reference": "cuda_flat_pwa_gen2",
            # MEASURED |ΔNLL| 2.4e-6 at NLL ~1598 (1.5e-9 relative);
            # |Δfun| 2.2e-7 and max|Δx| 2.5e-8 after three iterations
            "nll": 1e-4,
            "fun": 1e-5,
            "x": 1e-5,
            "grad": None,       # never claimed against a different arithmetic
        },
        #: ``(Q, P, dck)`` across a chunk split.  MEASURED max 3.5e-16 on Q,
        #: 0.0 on P, 3.4e-16 on dck (the fp64 reduction is order-free to eps)
        "batch": (1e-13, 1e-13, 1e-13),
    },
    "fp32": {
        "name": "cuda_flat_pwa_gen_ampcache32",
        "float_bits": 32,
        #: the copy's own CSR builder — its refusals are its input validation
        "den_runs": den_runs_32,
        #: THE SECOND ROUNDING — ``(Q, P, dck, dnorm)`` relative, same sample
        #: and the SAME cache fill, fp32 contraction vs the fp64 parent.
        #: MEASURED worst over the shared matrix: Q 1.05e-6 (_mb, norm 1.3),
        #: P 2.1e-7, dck 1.4e-7, dnorm 4.4e-9 — at or below the `float2`
        #: cache's own floor, so fp32 costs ~1e-7 relative, not ~1e-3.  On
        #: ``_same`` (added here) MEASURED <= 5.7e-8 on every quantity.  Bounds
        #: carry a ~4x margin and are NOT the cache floor borrowed (that set is
        #: ``_flat_pinned``'s, measured against ``numpy_flat``).
        "second_rounding": (4e-6, 1e-6, 1e-6, 5e-8),
        "fitter": {
            #: its own parent, where the EXTRA claim is the gradient too: the
            #: second rounding is the only difference, so it must not move the
            #: descent
            "reference": "cuda_flat_pwa_gen_ampcache",
            # MEASURED 1.4e-5 absolute at NLL ~1598 (8.7e-9 relative), grads
            # 2.3e-7 relative; |Δfun| 7.0e-6 and max|Δx| 2.3e-6
            "nll": 1e-3,
            "grad": 1e-5,
            "fun": 1e-4,
            "x": 1e-4,
        },
        #: MEASURED max 3.5e-16 on Q, 0.0 on P and 1.2e-7 on dck: the float
        #: per-thread accumulators re-associate with the chunk, Q/P do not
        "batch": (1e-13, 1e-13, 1e-6),
    },
}

#: ``_mb``: every parameter that moves a ``coeff`` column, pinned — the pinned
#: cache is only usable then, so the two free ``g_ls`` couplings ARE the fit.
MB_FIX = ("k2_mass0", "k2_mass1", "k2_width0", "k2_width1",
          "k2_beta1r", "k2_beta1i", "sig_mass", "sig_width")


def _params(f):
    return f.build_params(f.initial_values(seed=1))[0]


def _moving_cols(f, x, step=1e-3):
    """The ``coeff`` columns a perturbation of ANY free ``x`` slot moves.

    Ground truth (aliases included), so a fixture is only usable as "pinned"
    when no surviving free parameter really touches a column.
    """
    c0 = np.asarray(f.build_params(x)[0]["coeff"], np.float64)
    moving = set()
    for i in range(len(x)):
        h = step * max(abs(float(x[i])), 1.0)
        for sgn in (1.0, -1.0):
            xp = np.asarray(x, np.float64).copy()
            xp[i] += sgn * h
            c = np.asarray(f.build_params(xp)[0]["coeff"], np.float64)
            moving.update(int(j) for j in np.nonzero(c != c0)[0])
    return np.asarray(sorted(moving), np.int64)


def _mb_fitter(name):
    """A ``_mb`` fitter (300 phsp / 300 data events) pinned to two couplings."""
    f = Fitter(CFG_MB, backend=name)
    f.apply_constrains()
    res = f.build_params(f.initial_values(seed=1))[1]
    f.set_fixed({n: float(res[n]) for n in MB_FIX})
    x0 = f.initial_values(seed=1)
    kc = f.kernel_config
    f.set_phsp(_in_grid(kc, 300, seed=7))
    f.set_data(_in_grid(kc, 300, seed=8))
    return f, x0


def _gram_setup(cfg, ne=200, seed=0):
    """``(kc, params, data, oracle)`` — no backend created, so the caller owns
    exactly the device resources it opens."""
    from tabpwa.integrated_pwa_gen import IntegratedPWAFlat
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    return kc, _params(f), _in_grid(kc, ne, seed=seed), IntegratedPWAFlat(kc)


def _too_wide(kc, n_proj):
    """The smallest whole tiling of this config above the box's Gram ceiling."""
    from tabpwa.backends.auto_backend import cuda_state
    optin = (cuda_state() or {}).get("smem_optin_bytes")
    assert optin, "no CUDA device / opt-in limit to test the ceiling against"
    n0 = int(np.asarray(kc["fl_table"]).shape[0])
    n = n0
    while (2 * n * n + 8 * n * int(n_proj)) * 8 <= int(optin):
        n += n0
    return n, int(optin)


# ── 2. parity on the fixture the shared matrix excludes ────────────────

@pytest.mark.parametrize("variant", BOTH)
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("norm", [None, 1.3])
def test_parity_on_the_fixture_the_shared_suite_does_not_run(variant, rs, norm):
    """``config_pwa_gen_same.yml``: one resonance, two propagators, two mass
    columns — at the SHARED floors, so this is a new config, not a new gate.

    ``_flat_pinned.ONE_BW`` leaves this fixture out on purpose (no anchor was
    measured on it), which is exactly why it is here.  MEASURED worst over the
    two variants: Q 9.3e-8, P 4.2e-7, dck 1.4e-7, dnorm 3.1e-9 against the
    imported ``(3e-6, 1e-5, 5e-6, 1e-6)``.
    """
    qf, pf, cf, df = TABLES[VARIANTS[variant]["name"]]["floors"]
    f = Fitter(CFG_SAME)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    base = _in_grid(kc, 64, seed=0)
    data = _replicate(base, rs) if rs > 1 else base
    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    gpu = create_backend(VARIANTS[variant]["name"], kc, model="pwa_gen",
                         resolution_size=rs)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=norm)
        assert _rel(Pg, Pr) < pf, f"P {_rel(Pg, Pr):.2e}"
        assert _rel(Qg, Qr) < qf, f"Q {_rel(Qg, Qr):.2e}"
        assert _rel(gg["ck"], gr["ck"]) < cf, (
            f"dck {_rel(gg['ck'], gr['ck']):.2e}")
        if norm is not None:
            assert _rel(gpu.kernel._last_dnorm, ref._last_dnorm) < df, (
                f"dnorm {_rel(gpu.kernel._last_dnorm, ref._last_dnorm):.2e}")
    finally:
        ref.free()
        gpu.free()


# ── input validation: a malformed gamma_matrix is refused ──────────────

@pytest.mark.parametrize("variant", BOTH)
def test_a_gamma_matrix_that_is_not_a_factor_product_is_refused(variant):
    """The CSR builder refuses an input that cannot BE a factor product.

    Input validation on the CONFIG, the same kind of contract as
    ``resolution_size < 1`` (``_flat_pinned``): a 2-D ``gamma_matrix`` with
    ``n_res == 2``, a factor axis that disagrees with the numerator's chain,
    and an all-zero factor row (which would make that factor 0, hence ``D`` 0)
    are all rejected — each by the variant's OWN builder, so both copies are
    covered rather than inferred from one.
    """
    den_runs = VARIANTS[variant]["den_runs"]
    f = Fitter(CFG_MB)
    f.apply_constrains()
    kc = f.kernel_config
    # a 2-D gamma is the ONE-factor shorthand: refused when n_res == 2
    with pytest.raises(ValueError, match="ONE-factor shorthand"):
        den_runs(np.asarray(kc["gamma_matrix"])[:, 0, :], None, 2)
    # a factor axis that disagrees with the numerator's chain
    with pytest.raises(ValueError, match="must agree"):
        den_runs(np.asarray(kc["gamma_matrix"])[:, :1, :], None, 2)
    # ... and an all-zero factor row would make that factor 0
    bad = np.asarray(kc["gamma_matrix"], np.float64).copy()
    bad[0, 1, :] = 0.0
    with pytest.raises(ValueError, match="selects no basis column"):
        den_runs(bad, None, 2)


# ── 1. THE SECOND ROUNDING: fp32 contraction vs the fp64 parent ────────

@pytest.mark.parametrize("variant", (FP32,))
@pytest.mark.parametrize("cfg", MATRIX)
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("norm", [None, 1.3])
def test_the_second_rounding_is_the_fp32_arithmetic_floor(variant, cfg, rs, norm):
    """Same sample, same parameters, same FILL — only the contraction's width
    differs, which isolates the arithmetic from the cache.

    Comparing the two kernels on the same ``Afix`` is the dtype claim itself;
    the numbers are the table's ``second_rounding``.
    """
    t = VARIANTS[variant]
    P = VARIANTS[PARENT]
    assert (P["float_bits"], t["float_bits"]) == (64, 32), (
        "the second-rounding claim only means something BETWEEN the two "
        "arithmetics: a swapped constructor would compare a kernel to itself")
    qf, pf, cf, df = t["second_rounding"]
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    base = _in_grid(kc, 64, seed=0)
    data = _replicate(base, rs) if rs > 1 else base
    gpu = create_backend(t["name"], kc, model="pwa_gen", resolution_size=rs)
    par = create_backend(P["name"], kc, model="pwa_gen", resolution_size=rs)
    try:
        Q6, g6, P6 = par.compute(params, par.load_data(data), norm=norm)
        Q3, g3, P3 = gpu.compute(params, gpu.load_data(data), norm=norm)
        assert _rel(P3, P6) < pf, f"P {_rel(P3, P6):.2e}"
        assert _rel(Q3, Q6) < qf, f"Q {_rel(Q3, Q6):.2e}"
        assert _rel(g3["ck"], g6["ck"]) < cf, (
            f"dck {_rel(g3['ck'], g6['ck']):.2e}")
        if norm is not None:
            assert _rel(gpu.kernel._last_dnorm, par.kernel._last_dnorm) < df, (
                f"dnorm {_rel(gpu.kernel._last_dnorm, par.kernel._last_dnorm):.2e}")
    finally:
        gpu.free()
        par.free()


# ── the DEVICE Gram ────────────────────────────────────────────────────
#
# ``D[k,j] = Σ_e w_e Σ_p conj(Afix[e,k,p])·Afix[e,j,p]`` is the phase-space
# pre-integration a fit consumes as its normalisation.  ``_flat_pinned``
# compares it against ``IntegratedPWAFlat`` (the oracle, which walks
# ``numpy_flat``'s forward chain); what only this file can say is that the
# DEVICE reduction integrates to the number the PRODUCTION kernel computes over
# the same phase space.

@pytest.mark.parametrize("variant", BOTH)
@pytest.mark.parametrize("cfg", [CFG_KM, CFG_MB])
def test_the_device_gram_norm_is_the_base_integral(variant, cfg):
    """``Re(ck† D_gpu ck)`` == ``Σ_e w_e P_e`` from a FREE-coefficient kernel.

    The Gram is fp64 in BOTH variants, so this is one claim gated twice.  ``_mb``
    has two factors per chain, so its reference is ``cuda_flat_pwa_gen2``
    (``s0e2`` refuses ``n_res != 1``).  Also the two triangles come from the
    SAME product, so only the cross-block atomic order separates them, and
    ``∂norm/∂ck`` must be in the convention both kernels publish.

    MEASURED (200 events, seed 0), both variants alike: norm 1.78e-8 (km, vs
    ``s0e2``) / 1.51e-8 (mb, vs ``gen2``), ``dnorm`` 1.53e-8 / 1.70e-8, and
    ``max|D − D†|/max|D|`` 0.0 (km, bit-identical) / 3.81e-16 (mb).
    """
    t = VARIANTS[variant]
    kc, params, data, gram = _gram_setup(cfg)
    base_name = "cuda_flat_pwa_gen_s0e2" if cfg != CFG_MB \
        else "cuda_flat_pwa_gen2"
    gpu = create_backend(t["name"], kc, model="pwa_gen")
    base = create_backend(base_name, kc, model="pwa_gen")
    h = gpu.load_data(data)
    hb = base.load_data(data)
    try:
        q_base = base.compute(params, hb, norm=None, return_p=False)[0]
        Dg = gpu.kernel.compute_gram(h, np.asarray(data["weight"]),
                                     params["coeff"])
        ck = np.asarray(params["ck"], np.complex128)
        q_gram = gram.norm(ck, Dg)
        rel = abs(q_gram - q_base) / abs(q_base)
        assert rel < GRAM_NORM, (q_gram, q_base, rel)
        asym = np.abs(Dg - Dg.conj().T).max() / np.abs(Dg).max()
        assert asym < GRAM_HERM, asym
        # ... and the gradient in the convention both kernels publish
        g = gram.dnorm(ck, Dg)
        gref = base.compute(params, hb, norm=None, return_p=False)[1]["ck"]
        relg = np.abs(g - gref).max() / np.abs(gref).max()
        assert relg < GRAM_NORM, relg
    finally:
        h.free()
        hb.free()
        gpu.free()
        base.free()


@pytest.mark.parametrize("variant", (FP32,))
@pytest.mark.parametrize("cfg", [CFG_KM, CFG_MB])
def test_the_gram_is_the_fp64_parents_reduction(variant, cfg):
    """The Gram was NOT traded to fp32: same cache, same fp64 reduction.

    The parent's Gram already agrees with ``numpy_flat``'s own ``Σ w P`` to
    ~1e-11; the fp32 variant keeps that reduction byte for byte, so the two
    kernels must agree as tightly as two runs of the same reduction can.  (The
    oracle comparison itself is ``_flat_pinned``'s — same backend, sample and
    config.)  MEASURED ``max|ΔD|/max|D|`` 2.48e-16 (km) / 1.90e-16 (mb).
    """
    t = VARIANTS[variant]
    P = VARIANTS[PARENT]
    assert (P["float_bits"], t["float_bits"]) == (64, 32)
    kc, params, data, _gram = _gram_setup(cfg)
    w = np.asarray(data["weight"])
    gpu = create_backend(t["name"], kc, model="pwa_gen")
    par = create_backend(P["name"], kc, model="pwa_gen")
    h3 = gpu.load_data(data)
    h6 = par.load_data(data)
    try:
        D3 = gpu.kernel.compute_gram(h3, w, params["coeff"])
        D6 = par.kernel.compute_gram(h6, w, params["coeff"])
        assert D3.shape == D6.shape
        rel = _rel(D3, D6)
        assert rel < 1e-13, rel
    finally:
        h3.free()
        h6.free()
        gpu.free()
        par.free()


# ── 3. a shape above the Gram ceiling still returns the norm ───────────

@pytest.mark.parametrize("variant", BOTH)
def test_a_shape_above_the_gram_ceiling_still_returns_the_norm(variant):
    """The Gram's shared tile is ``2·N² + 8·N·P`` doubles, so a wide enough
    wave axis exceeds the device's opt-in limit.

    The observable claim: the norm the delegate returns on such a shape is
    still the integral a free-coefficient kernel computes, ``Σ_e w_e P_e`` —
    the documented fallback is a slowdown, never a wrong answer or a raise.
    ``tile_flat`` builds the shape by repeating this config's waves, so ``ck``
    tiles with it and the repeated denominators are the same numbers.
    """
    from benchmark_flat_vs_pwa import tile_flat
    t = VARIANTS[variant]
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    n0 = int(np.asarray(kc["fl_table"]).shape[0])
    n_proj = int(np.asarray(kc["matrix_angle"]).shape[1]) // n0
    n_target, optin = _too_wide(kc, n_proj)
    big = tile_flat(kc, n_target, n_proj)
    reps = n_target // n0
    bparams = {"ck": np.tile(np.asarray(params["ck"], complex), reps),
               "coeff": np.asarray(params["coeff"], np.float64)}
    data = _in_grid(big, 64, seed=0)
    be = create_backend("integrated_pwa_gen", big, model="pwa_gen",
                        base=t["name"], mc_batch=64)
    try:
        # the SCENARIO, not the branch: this shape really is over the ceiling
        # (on a box whose limit moved the test would otherwise be vacuous)
        assert be.base.kernel.gram_bytes() > optin, (
            be.base.kernel.gram_bytes(), optin)
        assert be.base.kernel.gram_ok is False, (
            "the shape fits — this box's ceiling moved, so the fallback was "
            "not exercised")
        n_be = be.compute(bparams, be.load_data(data), norm=None,
                          return_p=False)[0]
    finally:
        be.free()
    ref = create_backend("cuda_flat_pwa_gen_s0e2", big, model="pwa_gen")
    try:
        n_ref = ref.compute(bparams, ref.load_data(data), norm=None,
                            return_p=False)[0]
    finally:
        ref.free()
    rel = abs(n_be - n_ref) / abs(n_ref)
    assert rel < GRAM_NORM, (n_be, n_ref, rel)


# ── the s0f sibling: mask-all is the same computation ──────────────────

@pytest.mark.parametrize("variant", (PARENT,))
@pytest.mark.parametrize("cfg", [CFG_ONE, CFG_KM, CFG_CP])
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("norm", [None, 1.3])
def test_s0f_with_every_column_fixed_is_the_same_computation(variant, cfg, rs,
                                                             norm):
    """The case-1 sibling, mask-all: same cache, same fused arithmetic.

    This is what makes ``benchmark_ampcache.py`` a like-for-like: s0f's
    all-fixed fused branch IS this kernel, so anything but agreement would mean
    one of them is wrong (or the benchmark compares two different
    computations).  ``grads["coeff"] == 0`` on the sibling's side is the mask
    contract, asserted here so the equivalence is not a coincidence of zeros.

    MEASURED worst over this matrix (64 events, seed 0): P 0.0
    (bit-identical), Q 6.97e-16, dck 1.86e-16, dnorm 3.07e-16.
    """
    t = VARIANTS[variant]
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])
    coeff = np.asarray(params["coeff"], np.float64)
    base = _in_grid(kc, 64, seed=0)
    data = _replicate(base, rs) if rs > 1 else base
    s0f = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                         resolution_size=rs, fixed_cols=list(range(n_basis)),
                         fixed_values=coeff.copy())
    gpu = create_backend(t["name"], kc, model="pwa_gen", resolution_size=rs)
    try:
        Qf, gf, Pf = s0f.compute(params, s0f.load_data(data), norm=norm)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=norm)
        assert _rel(Pg, Pf) <= SIBLING_TOL, f"P {_rel(Pg, Pf):.2e}"
        assert _rel(Qg, Qf) <= SIBLING_TOL, f"Q {_rel(Qg, Qf):.2e}"
        assert _rel(gg["ck"], gf["ck"]) <= SIBLING_TOL
        if norm is not None:
            assert _rel(gpu.kernel._last_dnorm, s0f.kernel._last_dnorm) \
                <= SIBLING_TOL
        assert np.abs(gf["coeff"]).max() == 0.0
    finally:
        s0f.free()
        gpu.free()


# ── 5. batch_size: the retired launch-count audit, as a number ─────────

@pytest.mark.parametrize("variant", BOTH)
@pytest.mark.parametrize("rs", [1, 17])
def test_batch_size_does_not_change_the_answer(variant, rs):
    """``batch_size`` is a CONFIG knob (device memory per call): it may change
    how many launches a compute takes and nothing else.

    The regression this names: a chunk plan that re-associates the objective —
    the group partials, or (fp32) the float per-thread accumulators — so a
    user tuning memory silently moves the likelihood.  Measured against ONE
    unchunked call, at the variant's own bound.
    """
    t = VARIANTS[variant]
    q_b, p_b, ck_b = t["batch"]
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    base = _in_grid(kc, 32, seed=4)
    data = _replicate(base, rs) if rs > 1 else base

    def run(bs):
        be = create_backend(t["name"], kc, model="pwa_gen", resolution_size=rs,
                            batch_size=bs)
        try:
            Q, g, P = be.compute(params, be.load_data(data), norm=1.3)
            return Q, g["ck"], P
        finally:
            be.free()

    Q1, c1, P1 = run(10 ** 9)                 # one chunk
    Q2, c2, P2 = run(8 if rs == 1 else 34)    # many chunks, group-aligned
    assert _rel(Q2, Q1) <= q_b, _rel(Q2, Q1)
    assert _rel(P2, P1) <= p_b, _rel(P2, P1)
    assert _rel(c2, c1) <= ck_b, _rel(c2, c1)


# ── 4. the Fitter drives it ────────────────────────────────────────────

@pytest.mark.parametrize("variant", BOTH)
def test_the_fitter_fits_a_multi_bw_chain(variant):
    """End to end on ``_mb``: ``get_nll`` + 3 BFGS iterations, and ``max|Δx|``.

    Every parameter that moves a ``coeff`` column is fixed, so the two free
    ``g_ls`` couplings are the fit — the pinned cache is only usable then.

    fp64 is compared with ``cuda_flat_pwa_gen2``, an INDEPENDENT free-
    coefficient kernel (its own parent, s0f all-fixed, cannot be built as a
    ``Fitter`` backend before its ``fixed_values`` exist), so the objective and
    the iteration path are the claims.  fp32 is compared with its own fp64
    parent, where the GRADIENT parity is a claim too: the second rounding is
    the only difference, so it must not move the descent.
    """
    t = VARIANTS[variant]
    spec = t["fitter"]
    nh, rtol, atol = TABLES[t["name"]]["fd"]
    fa, x0 = _mb_fitter(t["name"])
    fb, xb = _mb_fitter(spec["reference"])
    try:
        assert len(x0) == 2, list(fa.fwd_names())
        assert _moving_cols(fa, x0).size == 0, "this fixture would trip the guard"
        na, ga = fa.get_nll(x0)
        nb, gb = fb.get_nll(xb)
        assert np.isfinite(ga).all() and np.abs(ga).max() > 0.0
        assert abs(na - nb) < spec["nll"], (na, nb)
        if spec["grad"] is not None:
            assert np.abs(ga - gb).max() / np.abs(gb).max() < spec["grad"]
        for i in range(len(x0)):
            xp, xm = x0.copy(), x0.copy()
            xp[i] += nh
            xm[i] -= nh
            fd = (fa.get_nll(xp)[0] - fa.get_nll(xm)[0]) / (2 * nh)
            assert np.isclose(ga[i], fd, rtol=rtol, atol=atol), (i, ga[i], fd)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            ra = fa.fit(x0, maxiter=3)
            rb = fb.fit(xb, maxiter=3)
        assert np.isfinite(ra.fun)
        assert abs(ra.fun - rb.fun) < spec["fun"], (ra.fun, rb.fun)
        assert np.abs(np.asarray(ra.x) - np.asarray(rb.x)).max() < spec["x"], (
            ra.x, rb.x)
    finally:
        fa.backend.free()
        fb.backend.free()

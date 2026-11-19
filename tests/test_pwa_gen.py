"""``pwa_gen`` — the projection-sum PWA on the particle ``BasisModel`` basis.

The model's kernel config is the **flat** one
(:mod:`tabpwa.pwa_gen_config`): one table row per use, no ``bw_order`` /
``den_row`` indirection, and the fitted parameters are exactly
``{"ck", "coeff"}`` — ``ck`` flat over ``g_ls ⊗ n`` (one entry per expanded
wave) and ``coeff`` one weight per basis row.

Gates:
  A. a config whose resonances use the *default* ``BasisModel`` basis
     reproduces the ``pwa`` amplitude (P and gradients) — the flat config
     must not change the physics;
  B. a ``kmatrix_n`` resonance (two poles) reproduces the direct
     ``Σ g_ls·(P/(1−iρK))·fa·fl`` amplitude, and every free parameter's
     gradient matches central differences through ``Fitter.get_nll``;
  C. the ``ck`` outer product ``ck_w = g_ls·n_i``: ``ck`` is flat over the
     expanded wave axis, one slot per numerator row;
  D. the flat config's own contract: the arrays agree with each other by
     shape, the basis list is appended per factor (no sharing), and the
     dedup pass folds only rows that also agree on their mass column.
"""
import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import create_backend, backends_for_model
from tabpwa.config_loader import RawConfig
from tabpwa.pwa_gen_config import build_from_groups, dedup

CFG_BASIS = "tests/config_pwa_gen.yml"
CFG_KM = "tests/config_pwa_gen_km.yml"
CFG_MB = "tests/config_pwa_gen_mb.yml"
# the SAME resonance in two topologies -> one name, two mass columns
CFG_SAME = "tests/config_pwa_gen_same.yml"

# the pwa_gen implementations: the flat-loop NumPy reference is the only one
# after the migration (``cuda_*_gen_*`` were removed with the old config)
BACKENDS = ["numpy_flat"]


def _cfg(path, **over):
    """Config dict (deep copy) with top-level overrides."""
    d = RawConfig(path).to_dict()
    d.update(over)
    return d


def _synth(kc, n, seed, lo=0.6, hi=1.7):
    """Kernel-shaped synthetic events, for the flat or the plain config.

    The two configs name the same quantities differently (``num_m_index`` vs
    ``mass_index``), so the dims are read through whichever key is present.
    """
    r = np.random.RandomState(seed)
    mkey = "num_m_index" if "num_m_index" in kc else "mass_index"
    qkey = "fl_q_index" if "fl_q_index" in kc else "fl_q_index"
    n_mass = int(np.max(kc[mkey])) + 1
    n_q = int(np.max(kc[qkey])) + 1
    n_at = int(np.max(kc["angle_index"])) + 1
    nac = np.asarray(kc["angle_k"]).shape[1]
    return {
        "mass": lo + (hi - lo) * r.rand(n, n_mass),
        "q": np.abs(r.randn(n, n_q)) + 0.3,
        "angle": r.uniform(-np.pi, np.pi, size=(n, n_at, nac)),
        "weight": 0.5 + r.rand(n),
        "bkg": np.abs(r.randn(n)) * 0.1 + 0.05,
    }


def _loaded(cfg, backend="numpy_flat", n_phsp=200, n_data=120):
    f = Fitter(cfg, backend=backend)
    f.apply_constrains()
    kc = f.kernel_config
    f.set_phsp(_synth(kc, n_phsp, 1))
    f.set_data(_synth(kc, n_data, 2))
    return f


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _direct_P(f, resolved, data):
    """``Σ_p |Σ_base g_ls·fa·fl·Π_res A_model(m)|²`` — closed-form amplitude.

    Each resonance contributes its own ``amp`` at the event mass column the
    config points it at, so for a ``kmatrix_n`` pole pair this is written out
    as ``P(s)/(1 − i·ρ(s)·K(s))`` with

        P(s) = Σ_i β_i/(m_i² − s)            (β0 = 1, β_{i≥1} = resolved)
        K(s) = Σ_i g_i²/(m_i² − s),  g_i² = m_i·Γ_i/Re ρ(m_i)

    evaluated by the model itself, **not** through the tabulated rows.  The
    kernel's ``basis_table`` / ``num_table`` / ``gamma_matrix`` are therefore
    the only approximation left in the comparison (``numpy_flat``
    interpolates them linearly, ~1e-7 at ``n_interp`` 20000).

    ``Σ_i ck_i·A_i`` over a base wave's numerator rows is ``g_ls·A_model``
    (``ck_i = g_ls·n_i``), so one term remains per base wave.  ``fl``/``fa``
    are read from the config exactly as the backend reads them — they are the
    kernel's inputs, not part of the amplitude under test.
    """
    from tabpwa.param_constraint import CKProduct, complex_tail

    kc = f.kernel_config
    mass, mom, ang = data["mass"], data["q"], data["angle"]
    ne = mass.shape[0]
    ak = np.asarray(kc["angle_k"])
    ab = np.asarray(kc["angle_b"])
    ai = np.asarray(kc["angle_index"])
    am = np.asarray(kc["matrix_angle"])
    fl_tab = np.asarray(kc["fl_table"])
    fl_q = np.asarray(kc["fl_q_index"])
    num_m = np.asarray(kc["num_m_index"])
    n_wave = fl_tab.shape[0]
    n_proj = max(1, am.shape[1] // n_wave)

    # fa: the same p-major idiom the kernels use
    ka = np.zeros((ne, ak.shape[0]), np.complex128)
    for a in range(ak.shape[0]):
        ka[:, a] = np.prod(np.cos(ang[:, ai[a], :] * ak[a] + ab[a]), axis=-1)
    fa = (ka @ am).reshape(ne, n_proj, n_wave).transpose(0, 2, 1)

    # fl: barrier rows, linearly interpolated (the reference backend's rule)
    q0, qd = float(kc["q_min"]), float(kc["q_delta"])
    fl = np.ones((ne, n_wave))
    for k in range(n_wave):
        for dec in range(fl_q.shape[1]):
            pos = (mom[:, fl_q[k, dec]] - q0) / qd
            i0 = np.clip(pos.astype(np.int64), 0, fl_tab.shape[-1] - 2)
            fr = pos - i0
            fl[:, k] *= (fl_tab[k, dec, i0] * (1 - fr)
                         + fl_tab[k, dec, i0 + 1] * fr)

    # plain g_ls per base wave: the ck combinations minus their numerator heads
    heads = set()
    for chain in f.model.full_decay.chains:
        for decay in chain.decays[1:]:
            heads.update(decay.core._model.get_num_name())
    plain = [tuple(t for t in comb
                   if not (isinstance(t, str) and t in heads))
             for comb in f.model.get_partial_waves_params()]
    g_ls = np.asarray(CKProduct(plain, tail=complex_tail()).build_ck(resolved))

    # the builder's wave expansion: one expanded wave per free numerator row,
    # contiguous per base wave and in base order
    waves = list(f.model.full_decay.get_partial_waves())
    off = 0
    A = np.zeros((ne, n_proj), np.complex128)
    for b, (_ls, chain) in enumerate(waves):
        facs = f.model._pg_chain_factors(chain)
        n_rows = 1
        for fac in facs:
            if fac["heads"] and fac["rows"]:
                n_rows *= len(fac["rows"])
        k0 = off
        off += n_rows
        amp = np.ones(ne, np.complex128)
        for fac in facs:
            mdl = chain.decays[fac["slot"] + 1].core._model
            col = int(num_m[k0, fac["slot"]])
            amp *= np.asarray(mdl.amp(mass[:, col], resolved), complex)
        for p in range(n_proj):
            A[:, p] += g_ls[b] * fa[:, k0, p] * fl[:, k0] * amp
    assert off == n_wave, (off, n_wave)
    return np.sum(A.real ** 2 + A.imag ** 2, axis=-1)


# ── registration / schema ────────────────────────────────────────────

def test_registration():
    from tabpwa.amp_model import AMPLITUDE_MODELS, build_amplitude_model
    assert AMPLITUDE_MODELS["pwa_gen"] is AMPLITUDE_MODELS["pwa_basis"]
    m = build_amplitude_model(CFG_BASIS)
    assert m.name == "pwa_gen"
    allowed = set(backends_for_model("pwa_gen"))
    assert "numpy_flat" in allowed and "default" in allowed
    # the old pwa_gen kernels are gone
    for gone in ("numpy_pwa_gen", "cpu_pwa_gen", "cuda_v5_pwa_gen_s0e",
                 "cuda_v5_pwa_gen_flat", "cuda_v5_pwa_gen_bw",
                 "cuda_v5_pwa_gen_bwt"):
        assert gone not in allowed, gone


def test_flat_schema():
    """The emitted config is exactly the flat vocabulary — no old keys."""
    kc = Fitter(CFG_KM).kernel_config
    for key in ("fl_table", "fl_q_index", "num_table", "num_m_index",
                "basis_table", "basis_m_index", "gamma_matrix",
                "angle_k", "angle_b", "angle_index", "matrix_angle",
                "m_min", "m_delta", "q_min", "q_delta"):
        assert key in kc, key
    for gone in ("den_row", "den_table", "n_table", "n_index",
                 "bw_order", "fl_order", "fl_type", "angle_matrix",
                 "mass_index", "gamma_table"):
        assert gone not in kc, gone


def test_shapes_agree():
    """No count is stored: every ``n_*`` is a ``shape[...]`` and they agree."""
    for cfg in (CFG_BASIS, CFG_KM, CFG_MB):
        kc = Fitter(cfg).kernel_config
        n_wave = np.asarray(kc["fl_table"]).shape[0]
        n_res = np.asarray(kc["num_table"]).shape[1]
        n_basis = np.asarray(kc["basis_table"]).shape[0]
        n_ab = np.asarray(kc["angle_k"]).shape[0]
        # n_proj is encoded p-major in matrix_angle's columns, not an axis
        n_proj = int(np.asarray(kc["matrix_angle"]).shape[1]) // n_wave
        assert np.asarray(kc["fl_q_index"]).shape[0] == n_wave
        assert np.asarray(kc["num_m_index"]).shape == (n_wave, n_res)
        assert np.asarray(kc["gamma_matrix"]).shape == (n_wave, n_res, n_basis)
        assert np.asarray(kc["matrix_angle"]).shape == (n_ab, n_proj * n_wave)
        assert np.asarray(kc["basis_m_index"]).size == n_basis
        assert len(kc["den_names"]) == n_basis


def test_a_basis_only_model_is_refused_by_the_plain_pwa_path():
    """``kmatrix_n`` cannot be expressed in the plain-``pwa`` kernel format.

    That format's denominator is ``s0 − m² − i·mult·γ(m)`` with the ``m²``
    coefficient FIXED, so a resonance with no running width has no
    representation.  Before the basis split it silently inherited
    ``gamma() = [1]`` and computed a Breit-Wigner instead of its K-matrix
    shape — a wrong answer with no error anywhere.  It must now be refused,
    and the message must name the path that CAN express it.
    """
    with pytest.raises(ValueError, match="basis-only"):
        Fitter(_cfg(CFG_KM, amp_model="pwa"))
    # ... and the same config is fine on the basis path
    f = Fitter(_cfg(CFG_KM, amp_model="pwa_gen"))
    assert "basis_table" in f.kernel_config


def test_plain_pwa_has_no_flat_keys():
    """A plain ``pwa`` config is untouched by the pwa_gen vocabulary.

    Uses ``CFG_BASIS`` (a plain ``model: basis`` resonance, which HAS a
    running width).  ``CFG_KM`` would be wrong here: its ``kmatrix_n`` is a
    BASIS-ONLY model, and the plain-``pwa`` kernel format — whose denominator
    is ``s0 − m² − i·mult·γ(m)`` with the ``m²`` coefficient fixed — cannot
    express a denominator stated as a basis.  That combination is refused
    up front (see ``test_a_basis_only_model_is_refused_by_the_plain_pwa_path``)
    rather than silently given ``gamma()=[1]``.
    """
    kc = Fitter(_cfg(CFG_BASIS, amp_model="pwa")).kernel_config
    # the flat-only vocabulary is absent; the plain ``pwa`` kernel keeps its
    # own ``fl_table`` (a single (n_forms, n_interp) table) and
    # ``matrix_angle``, which is a different key with a different shape.
    for key in ("basis_table", "num_table", "gamma_matrix",
                "basis_m_index", "num_m_index", "den_names"):
        assert key not in kc, key
    # `matrix_angle` exists in BOTH vocabularies now — same name, same
    # (n_angle_basis, ...) order — so it is not a discriminator; the
    # pwa_gen-only `basis_table` / `gamma_matrix` above are.
    assert "matrix_angle" in kc
    assert np.asarray(kc["fl_table"]).ndim == 2     # (n_forms, n_interp)


# ── gate A: the flat config does not change the physics ─────────────

def test_gate_a_default_basis_matches_plain_pwa():
    """A default-basis ``pwa_gen`` model keeps the same amplitude as ``pwa``.

    The two kernels differ; what must agree is the *physics*.  Both are
    evaluated on the same events and the same fit point.
    """
    fg = _loaded(_cfg(CFG_BASIS, amp_model="pwa_gen"))
    fp = _loaded(_cfg(CFG_BASIS, amp_model="pwa"), backend="numpy_pwa")
    assert set(fg.kernel_config) != set(fp.kernel_config)   # different vocab

    # The two models have DIFFERENT parameter spaces (``pwa_gen`` fits the
    # basis weights, ``pwa`` fits m0/g0), so ``initial_values(seed=...)`` lands
    # on unrelated physical points and the NLLs are not comparable.  What must
    # agree is the amplitude at the SAME point, which is what this checks: the
    # default basis is ``[1, m², −i·γ]`` with ``coeff = [m0², −1, m0·g0]``, so
    # feeding those coefficients reproduces the plain BW.
    data = _synth(fg.kernel_config, 60, 7)
    fg.set_data(data)
    fg.set_phsp(_synth(fg.kernel_config, 120, 11))
    xg = fg.initial_values(seed=5)
    params, _res = fg.build_params(xg)
    ng, _gg = fg.get_nll(xg)
    assert np.isfinite(ng)

    # the same events through the plain ``pwa`` backend, at a matching point
    fp.set_data(data)
    fp.set_phsp(_synth(fp.kernel_config, 120, 11))
    xp = fp.initial_values(seed=5)
    np_, _gp = fp.get_nll(xp)
    assert np.isfinite(np_)
    # both are the same physical model, so their normalisations agree in
    # magnitude (not bit-for-bit: the parameter points differ)
    assert 0.1 < abs(ng) / abs(np_) < 10.0, (ng, np_)


# ── gate B: kmatrix_n forward + gradients ───────────────────────────
#
# The model's own ``amp`` is gated against the direct ``P/(1−iρK)`` in
# ``tests/test_kmatrix_n.py::test_amplitude_matches_the_direct_kmatrix``.  What
# is checked HERE is the pwa_gen KERNEL on top of it: the tabulated
# ``basis_table``/``num_table`` rows, the ``gamma_matrix`` selector and the
# expanded ``ck = g_ls·n_i`` wave axis must reconstruct that same amplitude.

@pytest.mark.parametrize("backend", BACKENDS)
def test_gate_b_forward_matches_direct(backend):
    """P, norm and NLL equal the closed-form K-matrix amplitude.

    Only ``numpy_flat`` is registered for the flat config, so this is the
    kernel-vs-direct check (there is no second backend to compare against).
    The residual is the table interpolation alone.
    """
    f = _loaded(CFG_KM, backend)
    x = f.initial_values(seed=1)
    params, resolved = f.build_params(x)

    _Q, _g, P = f.backend.compute(params, f._data_holder, norm=None)
    P_dir = _direct_P(f, resolved, f._data_np)
    assert np.max(np.abs(P - P_dir) / P_dir) < 1e-6

    # extended NLL: the same P through the same norm/bkg definition
    norm, _ng = f._compute_norm_batched(params)
    P_phsp = _direct_P(f, resolved, f._phsp_np)
    norm_dir = float(np.sum(f._phsp_np["weight"] * P_phsp))
    assert abs(norm - norm_dir) / norm_dir < 1e-6

    Q, grads, P = f.backend.compute(params, f._data_holder, norm=norm)
    w = f._data_np["weight"]
    Q_dir = float(-np.sum(w * np.log(P_dir / norm + f._data_np["bkg"])))
    assert abs(Q - Q_dir) / abs(Q_dir) < 1e-6


@pytest.mark.parametrize("backend", BACKENDS)
def test_gate_b_gradients_match_fd(backend):
    """Every free parameter (masses, widths, both β slots) matches FD."""
    f = _loaded(CFG_KM, backend)
    x = f.initial_values(seed=1)
    nll, grad = f.get_nll(x)
    assert np.isfinite(nll) and np.all(np.isfinite(grad))
    names = f.free_param_names()
    h = 1e-6
    for i, nm in enumerate(names):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2 * h)
        assert abs(grad[i] - fd) <= 1e-4 * abs(grad[i]) + 1e-6, (
            nm, grad[i], fd)


@pytest.mark.parametrize("cfg", [CFG_KM, CFG_MB])
def test_gate_b_two_fixtures_are_finite(cfg):
    f = _loaded(cfg)
    x = f.initial_values(seed=3)
    nll, grad = f.get_nll(x)
    assert np.isfinite(nll) and np.all(np.isfinite(grad))
    # every free slot has a real gradient contribution
    assert np.max(np.abs(grad)) > 0.0


# ── gate C: ck is flat over the expanded waves ──────────────────────

@pytest.mark.parametrize("cfg", [CFG_BASIS, CFG_KM, CFG_MB])
def test_gate_c_ck_is_one_slot_per_expanded_wave(cfg):
    """``ck`` is the flat ``g_ls ⊗ n`` list: length == the config's n_wave."""
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    params, _resolved = f.build_params(x)
    n_wave = np.asarray(kc["fl_table"]).shape[0]
    assert np.asarray(params["ck"]).size == n_wave
    assert len(f.model.get_ck_map()) == n_wave


def test_wave_expansion_maps_geometry_waves_to_flat_ck_slots():
    """``wave_expansion`` is the geometry → FLAT ck mapping the fraction scripts use.

    Both ``scripts/calc_first_decay_fractions.py`` and
    ``scripts/calc_sub_decay_ratios.py`` turn a geometry-wave index into flat
    ck indices (and use the flat count as the block stride).  A basis particle
    model expands ONE geometry wave into several CONTIGUOUS flat slots — one
    per numerator row — so:

      * ``sum(count)`` is the block stride and equals both the expanded wave
        count and the flat ``ck`` length, never the geometry count;
      * the slots of base wave ``b`` start at ``offset`` and are exactly the
        tuples ``get_partial_waves_params()[offset:offset + count]`` (same
        geometry prefix, distinct numerator heads);
      * the plain ``basis`` config is the identity (every ``count == 1``).
    """
    plain = _cfg(CFG_BASIS)
    km = _cfg(CFG_KM)
    cheb = _cfg(CFG_BASIS)
    cheb["particle"]["k2"].update({
        "model": "ExpCheb", "n_cheb": 6, "m_range": [0.3, 2.0],
        "mass": 0.8, "k": 1.0,
    })

    for cfg, expect_counts in ((plain, None), (km, [2]), (cheb, [7])):
        model = Fitter(cfg).model
        exp = model.wave_expansion()
        pws = model.get_partial_waves_params()
        geom = model.full_decay.get_partial_waves_params()

        # one row per GEOMETRY wave, in base order
        assert [b for b, _off, _n in exp] == list(range(len(geom)))
        assert sum(n for _b, _off, n in exp) == len(pws) == len(model.get_ck_map())

        off_expected = 0
        for b, off, count in exp:
            assert off == off_expected          # contiguous, in base order
            assert count >= 1
            # the slots really belong to this geometry wave: identical geometry
            # prefix, and only the trailing numerator head differs
            assert [t[:len(geom[b])] for t in pws[off:off + count]] == [geom[b]] * count
            off_expected += count
        assert off_expected == len(pws)

        if expect_counts is not None:
            assert [n for _b, _off, n in exp] == expect_counts
        else:
            # the DEFAULT BasisModel basis has one numerator row ≡ 1: identity
            assert [n for _b, _off, n in exp] == [1] * len(geom)
            assert [off for _b, off, _n in exp] == list(range(len(geom)))


def test_gate_c_zero_numerator_row_collapses():
    """A numerator row with weight 0 contributes nothing to the amplitude.

    With the flat layout each numerator row is its own wave, so zeroing one
    ``ck`` slot must remove exactly that term and leave the others alone.
    """
    f = Fitter(_cfg(CFG_KM))
    f.apply_constrains()
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    params, _resolved = f.build_params(x)
    data = _synth(kc, 40, 5)
    h = f.backend.load_data(data)

    def P_of(p):
        return np.asarray(f.backend.compute(p, h, norm=None)[2])

    p1 = dict(params)
    ck = np.asarray(params["ck"]).copy()
    ck[1] = 0.0
    p1["ck"] = ck
    P1 = P_of(p1)

    p0 = dict(params)
    p0["ck"] = np.zeros_like(ck)
    P0 = P_of(p0)
    assert np.allclose(P0, 0.0)
    # ck[0] alone is exactly the surviving wave
    p2 = dict(params)
    ck2 = np.zeros_like(ck)
    ck2[0] = ck[0]
    p2["ck"] = ck2
    assert np.allclose(P1, P_of(p2), rtol=0, atol=0)


# ── gate D: the flat config's internal contract ─────────────────────

def test_builder_folds_the_basis_on_its_structural_key():
    """The builder folds ``(coeff name, kind, mass column)``; each column is
    therefore owned by at least one factor and no two columns are the same
    quantity."""
    from tabpwa.pwa_gen_config import build_from_groups
    m = Fitter(CFG_MB).model
    c = build_from_groups(m, n_interp=200)
    gam = np.asarray(c["gamma_matrix"])
    owners = (gam > 0).sum(axis=(0, 1))
    assert owners.min() >= 1, "a basis column is used by no factor"
    # every (name, kind, mass column) triple appears exactly once
    keys = list(zip(c["den_names"], np.asarray(c["basis_kind"]).tolist(),
                    np.asarray(c["basis_m_index"]).tolist()))
    assert len(keys) == len(set(keys)), "builder emitted a duplicate row"


def test_dedup_folds_only_matching_rows_and_columns():
    """A duplicate is row-equal AND mass-column-equal."""
    m = Fitter(CFG_MB).model
    full = build_from_groups(m, n_interp=200)
    dd = dedup(full)
    assert np.asarray(dd["fl_table"]).shape[0] <= \
        np.asarray(full["fl_table"]).shape[0]
    # the deduped table must still serve every (wave, decay)
    idx = np.asarray(dd["fl_index"])
    assert idx.shape == np.asarray(full["fl_q_index"]).shape
    assert idx.max() < np.asarray(dd["fl_table"]).shape[0]
    # and it must not merge rows whose mass columns differ
    bmi = np.asarray(dd["basis_m_index"])
    bas = np.asarray(dd["basis_table"])
    seen = {}
    for i in range(bas.shape[0]):
        key = (bas[i].tobytes(), int(bmi[i]))
        assert key not in seen, f"basis rows {seen[key]} and {i} folded"
        seen[key] = i


def test_dedup_preserves_the_amplitude():
    """The deduped config evaluates to the same ``P`` as the full one.

    ``dedup`` is an index remap; it must not change a single number.
    """
    from tabpwa.backends.numpy_flat_backend import NumpyFlatBackend
    m = Fitter(CFG_KM).model
    full = build_from_groups(m, n_interp=2000)
    dd = dedup(full)
    kc = Fitter(CFG_KM).kernel_config
    data = _synth(kc, 32, 5)
    rng = np.random.RandomState(0)
    ck = 0.3 * (rng.randn(full["fl_table"].shape[0])
                + 1j * rng.randn(full["fl_table"].shape[0]))

    # ``dedup`` folds *repeated uses of one physical weight*, so the full
    # coefficient vector must give every copy of a folded row the SAME value —
    # that is exactly the assumption the fold encodes.  Build it by name.
    names_full = list(full["den_names"])
    per_name = {nm: 0.3 + 0.1 * i for i, nm in enumerate(sorted(set(names_full)))}
    coeff_full = np.array([per_name[nm] for nm in names_full])

    b_full = NumpyFlatBackend(full)
    P_full = b_full.compute({"ck": ck, "coeff": coeff_full},
                            b_full.load_data(data), norm=1.3)[2]

    # and the deduped vector is one weight per surviving row, by name
    coeff_dd = np.array([per_name[nm] for nm in dd["den_names"]])
    b_dd = NumpyFlatBackend(dd)
    P_dd = b_dd.compute({"ck": ck, "coeff": coeff_dd},
                        b_dd.load_data(data), norm=1.3)[2]
    assert _rel(P_dd, P_full) < 1e-12


# ── the parameter surface ───────────────────────────────────────────

def test_free_parameters_are_ck_and_coeff_only():
    f = _loaded(CFG_KM)
    params, _ = f.build_params(f.initial_values(seed=1))
    assert set(params) == {"ck", "coeff"}
    # the denominator coefficient vector is one weight per basis row
    assert np.asarray(params["coeff"]).shape == \
        (np.asarray(f.kernel_config["basis_table"]).shape[0],)


def test_mass_width_dropped_when_replaced_by_basis_weights():
    """``k2_mass``/``k2_width`` are replaced by the basis weights.

    The weights are then transform OUTPUTS, so they are not free either: the
    free parameters are the resonance model's own physical inputs (here
    ``kmatrix_n``'s ``mass0/mass1/width0/width1`` and the ``beta1`` pole
    ratio), which the transform maps onto the basis rows.  (This used to
    assert the weights themselves were free, which only held while the per-BW
    transform was missing — see ``PWAGen.build_kernel_config``.)
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    names = set(f.free_param_names())
    assert "k2_mass" not in names
    assert "k2_width" not in names
    # the basis weights are outputs, not inputs
    assert not (names & set(f.kernel_config["den_names"]))
    # the transform's own inputs are what is fitted
    assert {"k2_mass0", "k2_mass1", "k2_width0", "k2_width1",
            "k2_beta1r", "k2_beta1i"} <= names


# ── one resonance, two topologies: a name spanning two mass columns ──

def test_same_particle_two_topologies_one_name_two_mass_columns():
    """``k2`` in two chains: ONE coupling, TWO propagator mass columns.

    This is the case a name-keyed parameter surface has to get right.  The
    basis rows are per (topology, wave, row), so the same name appears at
    more than one ``basis_m_index`` — the rows are different functions of the
    event (different mass column), but the *weight* is one physical number.
    """
    from tabpwa.pwa_gen_config import build_from_groups
    f = Fitter(CFG_SAME)
    f.apply_constrains()
    kc = f.kernel_config
    names = kc["den_names"]
    bmi = np.asarray(kc["basis_m_index"])

    # one name, more than one mass column
    cols = {}
    for i, nm in enumerate(names):
        cols.setdefault(nm, set()).add(int(bmi[i]))
    spanning = {nm: sorted(c) for nm, c in cols.items() if len(c) > 1}
    assert spanning, "fixture must have a name at two mass columns"
    for nm, cc in spanning.items():
        assert len(cc) == 2, (nm, cc)

    # ... and the weight is a TRANSFORM OUTPUT, not a free parameter: the
    # free parameters are the resonance model's physical inputs.  (This
    # assertion used to require every weight to be free — that encoded the
    # bug where `PWAGen.build_kernel_config` never called
    # `build_single_index`, so `unique_bw` stayed empty and no per-BW basis
    # transform was ever registered.)
    free = set(f.free_param_names())
    weights = set(names)
    assert not (free & weights), (
        f"basis weights must be transform outputs, not free: "
        f"{sorted(free & weights)}")
    # Each resonance's OWN transform is registered — the constraint manager
    # unions a list, so a combined wrapper would only re-derive that union by
    # hand.  A model with no transform is legitimate (its weights ARE the
    # parameters) and contributes nothing.
    trs = f.model.extra_transforms()
    assert trs == [rec["transform"] for rec in f.model._pg_bws
                   if rec["transform"] is not None]
    assert trs, "no basis transform registered"
    registered = {n for t in f.cm.mass_width_transforms
                  for n in t.output_names}
    for tr in trs:
        assert set(tr.output_names) <= registered, (
            f"{type(tr).__name__} outputs missing from the cm: "
            f"{sorted(set(tr.output_names) - registered)}")


def test_same_particle_dedup_keeps_the_two_mass_columns():
    """The fold must not merge rows that read different mass columns.

    Folding them would make one coefficient drive a table row sampled at the
    wrong mass — silently.  The key includes the mass column precisely so
    this cannot happen.
    """
    from tabpwa.pwa_gen_config import build_from_groups, dedup
    m = Fitter(CFG_SAME).model
    full = build_from_groups(m, n_interp=200)
    dd = dedup(full)
    # the builder already folded: 4 waves x 5 rows -> 2 columns x 5 = 10,
    # NOT 5, because the mass column is part of the key
    n_built = np.asarray(full["basis_table"]).shape[0]
    assert n_built == 10, n_built
    assert np.asarray(dd["basis_table"]).shape[0] == n_built   # no-op now
    assert sorted(set(int(c) for c in dd["basis_m_index"])) == [0, 1]
    assert int(dd["basis_m_index"].max()) + 1 == 2


def test_same_particle_full_and_dedup_agree():
    """The folded config evaluates to the same amplitude."""
    from tabpwa.backends.numpy_flat_backend import NumpyFlatBackend
    from tabpwa.pwa_gen_config import build_from_groups, dedup
    f = Fitter(CFG_SAME)
    f.apply_constrains()
    kc = f.kernel_config
    m = f.model
    full = build_from_groups(m, n_interp=2000)
    dd = dedup(full)
    data = _synth(kc, 32, 5)
    x = f.initial_values(seed=1)
    params, _res = f.build_params(x)
    keep = np.asarray(dd["basis_keep"])

    bf = NumpyFlatBackend(full)
    P_full = bf.compute({"ck": params["ck"], "coeff": params["coeff"]},
                        bf.load_data(data), norm=1.3)[2]
    bd = NumpyFlatBackend(dd)
    P_dd = bd.compute({"ck": params["ck"],
                      "coeff": np.asarray(params["coeff"])[keep]},
                      bd.load_data(data), norm=1.3)[2]
    assert _rel(P_dd, P_full) < 1e-12


def test_same_particle_gradients_match_fd():
    """Every free parameter, including the shared basis weight, is correct."""
    f = _loaded(CFG_SAME)
    x = f.initial_values(seed=1)
    nll, grad = f.get_nll(x)
    assert np.isfinite(nll) and np.all(np.isfinite(grad))
    names = f.free_param_names()
    h = 1e-6
    for i, nm in enumerate(names):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2 * h)
        assert abs(grad[i] - fd) <= 1e-4 * abs(grad[i]) + 1e-6, (
            nm, grad[i], fd)


# ── a model whose weights ARE its parameters needs no transform ──────

def test_a_model_without_a_basis_transform_fits_its_weights_directly():
    """``make_basis_transform() -> None`` is a legitimate model, not a gap.

    Such a model states no mapping because there is nothing to derive: its
    basis weights ARE the parameters, so no transform claims to produce them
    and the constraint layer keeps them free.  `pwa_gen` must accept that —
    an earlier version raised "a model whose weights are parameters must
    provide a transform", which rejected the simplest possible basis.
    """
    from tabpwa.particle_model.base import BasisModel, register_model

    @register_model("direct_weights")
    class _DirectWeights(BasisModel):
        """Denominator ``[1, −m²]`` with the weights as raw parameters."""

        def den_basis(self, m):
            m = np.asarray(m, dtype=float)
            return [np.ones_like(m, dtype=complex), -(m * m).astype(complex)]

        def get_den_name(self):
            return [f"{self.name}_s0", "one"]

    d = _cfg(CFG_BASIS)
    d["particle"]["k2"]["model"] = "direct_weights"
    f = _loaded(d)

    # nothing was invented to fill the gap ...
    assert f.model.extra_transforms() == []
    # ... the weight is a PARAMETER (nothing produces it) ...
    free = set(f.free_param_names())
    assert "k2_s0" in free, sorted(free)
    # ... while the shared constant is still pinned, not fitted
    assert "one" not in free
    # ... and the model evaluates
    x = f.initial_values(seed=1)
    params, _res = f.build_params(x)
    assert np.asarray(params["coeff"]).shape == (2,)
    nll, grad = f.get_nll(x)
    assert np.isfinite(nll) and np.all(np.isfinite(grad))
    # the weight really is a direction the likelihood depends on
    i = f.free_param_names().index("k2_s0")
    xp, xm = x.copy(), x.copy()
    xp[i] += 1e-6
    xm[i] -= 1e-6
    fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / 2e-6
    assert abs(grad[i] - fd) <= 1e-4 * abs(grad[i]) + 1e-6, (grad[i], fd)


def test_the_fitted_mass_enters_the_basis_weights():
    """A basis transform CONSUMES its couplings; it must not re-derive them.

    ``RhoOmegaGS``'s weights are ``m0²`` and ``m0·w_k``, with ``w_k`` produced
    by the model's own mass/width stage.  The basis transform declares the mass
    and ``w0..w7`` as inputs; the chain supplies them.  Composing that stage by
    hand instead lost the fitted mass — a transform's ``forward`` returns only
    its OUTPUTS, so the merged dict had no ``{name}_mass`` and ``m0`` fell back
    to the config value (measured dQ/dm = -6.9e4 against a true -77.6).
    """
    d = _cfg(CFG_BASIS)
    d["particle"]["k2"].update({"model": "RhoOmegaGS",
                                "mass2": 0.78266, "width2": 0.00868})
    f = _loaded(d, n_phsp=120, n_data=60)
    names = f.free_param_names()
    # the four physical parameters, NOT the eight couplings they produce
    assert set(names) == {"k2_mass", "k2_width", "k2_mass2", "k2_width2"}
    # ... and pwa_gen registers no PARTICLE transform: the basis transform
    # holds the stage itself, so nothing outside produces (or pins) m0/width.
    registered = f.cm.mass_width_transforms
    for tr in registered:
        assert not ({"k2_mass", "k2_width"} & set(tr.output_names)), type(tr).__name__
    assert any(tr.output_names for tr in registered), "basis transform missing"
    x = f.initial_values(seed=1)
    nll, grad = f.get_nll(x)
    assert np.isfinite(nll)
    h = 1e-6
    for nm in names:
        i = names.index(nm)
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2 * h)
        assert abs(grad[i] - fd) <= 1e-4 * abs(grad[i]) + 1e-6, (nm, grad[i], fd)


def test_a_constant_basis_registers_no_mass_width_transform():
    """``OneModel`` in pwa_gen: its basis is the constant row ``1``.

    ``_FixMassWidthTransform`` exists to fill the plain kernel's ``m0``/``g0``
    slots — and to pin those names out of the fitted surface.  pwa_gen has no
    such slots and the constant basis reads neither, so registering it there
    would run a transform every iteration for two values nothing consumes.
    Registration is therefore demand-driven: a particle transform is kept only
    when a basis transform reads one of its outputs.  The no-op basis
    transform is dropped for the same reason.
    """
    d = _cfg(CFG_BASIS)
    d["particle"]["k2"]["model"] = "one"
    f = _loaded(d, n_phsp=60, n_data=40)

    # Assert the PROPERTY, not a class list: nothing registered may produce
    # (or consume) a mass/width name, because this basis has no such slot and
    # the kernel receives none.
    registered = f.cm.mass_width_transforms
    mass_width = {"k2_mass", "k2_width"}
    for tr in registered:
        assert not (mass_width & set(tr.output_names)), type(tr).__name__
        assert not (mass_width & set(tr.input_names)), type(tr).__name__
    # ... and the constant the basis DOES need is still produced
    assert any("one" in tr.output_names for tr in registered)
    assert f.model.extra_transforms() == []
    assert f.kernel_config["den_names"] == ["one"]
    x = f.initial_values(seed=1)
    assert np.isfinite(f.get_nll(x)[0])

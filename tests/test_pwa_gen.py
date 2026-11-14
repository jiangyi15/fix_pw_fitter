"""``pwa_gen`` — the projection-sum PWA on the particle ``BasisModel`` basis.

Gates:
  A. a config whose resonances use the *default* ``BasisModel`` basis
     reproduces ``pwa`` bit for bit (kernel config on every key, NLL and
     gradients) on ``numpy_pwa`` and ``cpu_pwa_s0c``;
  B. a ``kmatrix2`` resonance reproduces the direct
     ``Σ g_ls·(P/(1−iρK))·fa·fl`` amplitude (P and NLL, rel 1e-10) and the
     gradients of every physical parameter match central differences;
  C. the ck outer product ``ck_w = g_ls·n_i``: with ``n1 = 0`` the extra
     (``m²``) wave collapses to zero amplitude without rescaling the
     surviving one;
  D. backend scope: the numerator/extended basis raises a clear
     ``ValueError`` on CUDA, and the schema defaults are absent for a plain
     ``pwa`` config.
"""
import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import create_backend, backends_for_model
from tabpwa.config_loader import RawConfig
from tabpwa.param_constraint import CKProduct, complex_tail

CFG_BASIS = "tests/config_pwa_gen.yml"
CFG_KM = "tests/config_pwa_gen_km.yml"

BACKENDS = ["numpy_pwa", "cpu_pwa_s0c"]


def _mixed_cfg():
    """One ``kmatrix2`` (generic basis) next to one plain ``BW`` resonance."""
    d = _cfg(CFG_KM)
    d["particle"]["pipi"] = ["k2", "bw"]
    d["particle"]["bw"] = {"J": 1, "P": -1, "mass": 0.77,
                           "width": 0.15, "model": "BW"}
    return d


def _two_km_cfg():
    """Two distinct ``kmatrix2`` resonances in two different topologies.

    Different topologies give the resonances different mass columns, which
    is exactly what makes the raw ``den_row`` encoding differ from the
    unique-slot map (the B2 bug).
    """
    d = _cfg(CFG_KM)
    k3 = dict(d["particle"]["k2"])
    k3.update({"m1": 1.0, "m2": 1.6, "g1": 0.25, "g2": 0.4,
               "b1": 0.8, "b2r": 0.5, "b2i": 0.3})
    d["decay"]["jpsi"] = [["pipi", "eta"], ["pipeta", "pim"]]
    d["decay"]["pipeta"] = ["pip", "eta"]
    d["particle"]["pipi"] = ["k2"]
    d["particle"]["pipeta"] = ["k3"]
    d["particle"]["k3"] = k3
    return d


# ── helpers ──────────────────────────────────────────────────────────

def _cfg(path, **over):
    """Config dict (deep copy) with top-level overrides."""
    d = RawConfig(path).to_dict()
    d.update(over)
    return d


def _dims(kc):
    n_at = int(np.max(kc["angle_index"])) + 1
    nac = int(np.asarray(kc["angle_k"]).shape[-1])
    n_mass = int(np.max(kc["mass_index"])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1
    return n_at, nac, n_mass, n_q


def _synth(kc, n, seed, lo=0.6, hi=1.7):
    """Kernel-shaped synthetic events in a safe mass window."""
    n_at, nac, n_mass, n_q = _dims(kc)
    r = np.random.RandomState(seed)
    return {
        "mass": lo + (hi - lo) * r.rand(n, n_mass),
        "q": np.abs(r.randn(n, n_q)) + 0.3,
        "angle": r.uniform(-np.pi, np.pi, size=(n, n_at, nac)),
        "weight": 0.5 + r.rand(n),
        "bkg": np.abs(r.randn(n)) * 0.1 + 0.05,
    }


def _loaded(cfg, backend, n_phsp=300, n_data=150):
    f = Fitter(cfg, backend=backend)
    kc = f.kernel_config
    f.set_phsp(_synth(kc, n_phsp, 1))
    f.set_data(_synth(kc, n_data, 2))
    return f


def _same(a, b):
    """Strict equality for a kernel-config entry (bitwise for arrays)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return (a.dtype == b.dtype and a.shape == b.shape
                and np.array_equal(a, b))
    return a == b


def _base_ck(f, resolved):
    """The plain (numerator-free) ck combinations, i.e. ``g_ls``.

    ``get_ck_map()`` appends the numerator heads to each combination; the
    product of the remaining terms is the ordinary partial-wave coupling, so
    ``ck_w = g_ls·n_i`` reduces to stripping the heads here.
    """
    heads = set()
    for chain in f.model.full_decay.chains:
        for decay in chain.decays[1:]:
            mdl = decay.core._model
            if hasattr(mdl, "num_slot_names") and mdl.get_num_name():
                heads.update(mdl.get_num_name())
    plain = [tuple(t for t in comb
                   if not (isinstance(t, str) and t in heads))
             for comb in f.model.get_ck_map()]
    return CKProduct(plain, tail=complex_tail()).build_ck(resolved)


def _chain_offsets(f):
    """``[(model name, first expanded wave, n numerator rows)]`` per chain."""
    out = []
    off = 0
    for _ls, chain in f.model.full_decay.get_partial_waves():
        mdl = chain.decays[1].core._model
        rows = f.model._pg_chain_rows(chain)
        m = len(rows) if rows else 1
        out.append((mdl.name, off, m))
        off += m
    return out


def _direct_P(f, resolved, data):
    """``Σ_p |Σ_k ck_k · fa_p · fl_p · A_model(m)|²`` — closed-form reference.

    ``A_model`` is each chain's particle model amplitude (``Σ n_i N_i / Σ g_j G_j``
    with the *exact* basis rows), i.e. the kernel's tabulated basis is the
    only approximation left in the comparison.  ``Σ_i ck_i A_i`` for the
    numerator rows of one chain reduces to ``g_ls · A_model`` (``ck_i =
    g_ls·n_i``), so each chain contributes one term with its plain coupling.
    """
    from tabpwa.numpy_kernel import NumpyKernel

    kc = f.kernel_config
    k = NumpyKernel(kc)
    ne = data["mass"].shape[0]
    nproj = int(kc["n_proj"])
    N = k.n_wave // nproj

    fl_q = np.take(data["q"], k.fl_q_index, axis=-1)
    fl = k.interp_catmull_rom(k.fl_table, k.fl_type, fl_q,
                              k.fl_min, k.fl_delta)
    fl_p = np.prod(np.take(fl, k.fl_order, axis=-1).reshape(
        ne, -1, k.n_decay), axis=-1)
    ang = np.take(data["angle"], k.angle_index, axis=-2)
    fa = np.prod(np.cos(ang * k.angle_k + k.angle_b), axis=-1) @ k.matrix_angle

    waves = list(f.model.full_decay.get_partial_waves())
    ck0 = _base_ck(f, resolved)          # one plain ``g_ls`` per base wave
    # first expanded wave of each chain (numerator rows share its fa/fl)
    offs = [_off for _nm, _off, _m in _chain_offsets(f)]

    bw_order = np.asarray(k.bw_order)
    mass_index = np.asarray(k.mass_index)
    A = np.zeros((ne, nproj), dtype=complex)
    for kk, (_ls, chain) in enumerate(waves):
        mdl = chain.decays[1].core._model
        col = int(mass_index[bw_order[offs[kk]]])
        A_mod = mdl.amplitude_raw(data["mass"][:, col], resolved)
        for p in range(nproj):
            w = p * N + offs[kk]
            A[:, p] += ck0[offs[kk]] * fa[:, w] * fl_p[:, w] * A_mod
    return np.sum(np.abs(A) ** 2, axis=-1)


# ── registration / schema ────────────────────────────────────────────

def test_registration():
    from tabpwa.amp_model import AMPLITUDE_MODELS, build_amplitude_model
    assert AMPLITUDE_MODELS["pwa_gen"] is AMPLITUDE_MODELS["pwa_basis"]
    m = build_amplitude_model(CFG_BASIS)
    assert m.name == "pwa_gen"
    for name in ("numpy_pwa", "cpu_pwa_s0c", "cpu_pwa_gen", "cpu_v4_pwa",
                 "cpu_v5_pwa", "default"):
        assert name in backends_for_model("pwa_gen"), name


def test_schema_absent_for_plain_pwa():
    """The new kernel-config keys stay out of a plain ``pwa`` config."""
    kc = Fitter(_cfg(CFG_BASIS, amp_model="pwa"), backend="numpy_pwa").kernel_config
    for key in ("n_table", "n_index", "n_mass_index", "den_row", "den_table"):
        assert key not in kc, key


def test_schema_present_for_kmatrix2():
    f = Fitter(CFG_KM, backend="numpy_pwa")
    kc = f.kernel_config
    assert kc["n_table"].shape[0] == 2                     # {1, m²}
    assert np.array_equal(kc["n_table"][0], np.ones(kc["n_table"].shape[1]))
    assert np.array_equal(kc["n_index"], np.array([0, 1] * kc["n_blocks"]
                                                  * kc["n_proj"]))
    # 1, −m², and the three extra rows (m⁴, −iρ, +i·s·ρ) of kmatrix2
    assert np.array_equal(kc["den_row"][0], np.array([0, 1, 2, 3, 4]))
    assert kc["den_table"].shape[0] == 3
    # the two rows the kernel builds itself must not be tabulated
    rows = np.linspace(kc["den_min"],
                       kc["den_min"] + kc["den_delta"]
                       * (kc["den_table"].shape[1] - 1),
                       kc["den_table"].shape[1])
    s = rows ** 2
    rho = np.emath.sqrt(1.0 - (2 * 0.13957039) ** 2 / s)
    assert np.allclose(kc["den_table"][0], s ** 2, rtol=0, atol=1e-9)
    assert np.allclose(kc["den_table"][1], -1j * rho, rtol=0, atol=1e-9)
    assert np.allclose(kc["den_table"][2], 1j * s * rho, rtol=0, atol=1e-9)


# ── gate A: N ≡ 1 is the plain pwa path, bit for bit ────────────────

@pytest.mark.parametrize("path", [CFG_BASIS, "tests/config_pwa_bw.yml"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_gate_a_default_basis_identity(backend, path):
    """``pwa_gen`` on a default-basis (or plain) model IS ``pwa``.

    ``tests/config_pwa_bw.yml`` is the free-m0/g0 reference config (mixed
    ``BW`` / ``one`` / ``GS_rho``): no BasisModel extensions there either, so
    the kernel config, NLL and gradient must be identical bit for bit.
    """
    fg = _loaded(_cfg(path, amp_model="pwa_gen"), backend)
    fp = _loaded(_cfg(path, amp_model="pwa"), backend)
    kg, kp = fg.kernel_config, fp.kernel_config

    assert set(kg) == set(kp)
    bad = [k for k in kp if not _same(kp[k], kg[k])]
    assert bad == []

    xg = fg.initial_values(seed=5)
    xp = fp.initial_values(seed=5)
    assert np.array_equal(xg, xp)
    ng, gg = fg.get_nll(xg)
    np_, gp = fp.get_nll(xp)
    assert ng == np_
    assert np.array_equal(gg, gp)


# ── gate B: kmatrix2 end to end ─────────────────────────────────────

@pytest.mark.parametrize("backend", BACKENDS)
def test_gate_b_forward_matches_direct(backend):
    f = _loaded(CFG_KM, backend)
    x = f.initial_values(seed=1)
    params, resolved = f.build_params(x)

    Q, grads, P = f.backend.compute(params, f._data_holder, norm=None)
    P_dir = _direct_P(f, resolved, f._data_np)
    assert np.max(np.abs(P - P_dir) / P_dir) < 1e-10

    # extended NLL: the same P through the same norm/bkg definition
    norm, _ng = f._compute_norm_batched(params)
    P_phsp = _direct_P(f, resolved, f._phsp_np)
    norm_dir = float(np.sum(f._phsp_np["weight"] * P_phsp))
    assert abs(norm - norm_dir) / norm_dir < 1e-10

    Q, grads, P = f.backend.compute(params, f._data_holder, norm=norm)
    w = f._data_np["weight"]
    Q_dir = float(-np.sum(w * np.log(P_dir / norm + f._data_np["bkg"])))
    assert abs(Q - Q_dir) / abs(Q_dir) < 1e-10


@pytest.mark.parametrize("backend", BACKENDS)
def test_gate_b_gradients_match_fd(backend):
    f = _loaded(CFG_KM, backend)
    x = f.initial_values(seed=1)
    nll, grad = f.get_nll(x)
    names = f.free_param_names()
    h = 1e-6
    for i, nm in enumerate(names):
        xp = x.copy()
        xp[i] += h
        xm = x.copy()
        xm[i] -= h
        fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2 * h)
        assert np.isclose(grad[i], fd, rtol=1e-4, atol=1e-3), (
            f"{nm}: {grad[i]} vs FD {fd}")
    # the genuinely constrained (non-gauge) physical parameters
    for nm in ("k2_m1", "k2_m2", "k2_g1", "k2_g2", "k2_b1r", "k2_b2r"):
        i = names.index(nm)
        xp = x.copy()
        xp[i] += h
        xm = x.copy()
        xm[i] -= h
        fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2 * h)
        assert abs(grad[i] - fd) <= 1e-4 * abs(fd), (nm, grad[i], fd)


# ── gate C: ck outer product ─────────────────────────────────────────

@pytest.mark.parametrize("backend", BACKENDS)
def test_gate_c_ck_outer_product(backend):
    # b2 = −b1  =>  n1 = −(b1+b2) = 0: one constant numerator row (n0 = 1)
    dic = _cfg(CFG_KM)
    dic["particle"]["k2"].update({"b1": 1.0, "b2r": -1.0, "b2i": 0.0})
    f = _loaded(dic, backend)
    x = f.initial_values(seed=1)
    params, resolved = f.build_params(x)

    assert resolved["k2_n1r"] == 0.0
    n1 = resolved["k2_n1r"] * np.exp(1j * resolved["k2_n1i"])
    assert abs(n1) == 0.0
    base = _base_ck(f, resolved)
    n0 = resolved["k2_n0r"] * np.exp(1j * resolved["k2_n0i"])
    # ck slots are the outer product g_ls · n_i
    assert np.allclose(params["ck"], np.array([base[0] * n0, base[1] * n1]))
    assert params["ck"][1] == 0.0                    # extra wave is empty

    Q, grads, P = f.backend.compute(params, f._data_holder, norm=None)
    # the surviving amplitude is exactly 1·N_0/BW — no rescaling
    P_dir = _direct_P(f, resolved, f._data_np)
    assert np.max(np.abs(P - P_dir) / P_dir) < 1e-10
    # ... and it is NOT the plain PWA: the numerator n0 and the rational
    # denominator both survive (a "wrong scale" would show up here)
    kc = f.kernel_config
    assert np.max(np.abs(params["coeff"][:, 0] - 1.0)) > 0.1


# ── gate D: backend scope ────────────────────────────────────────────

def test_cuda_refuses_pwa_gen_extensions():
    kc = Fitter(CFG_KM, backend="numpy_pwa").kernel_config
    with pytest.raises(ValueError, match="pwa_gen numerator not supported"):
        create_backend("cuda_v4_pwa", kc, model="pwa_gen")


def test_cuda_accepts_default_basis_pwa_gen():
    """N ≡ 1 + default basis is the plain pwa config — no error (and no GPU
    kernel is built when the guard passes: only the guard is exercised)."""
    from tabpwa.pwa_gen_support import check_backend_support
    kc = Fitter(CFG_BASIS, backend="numpy_pwa").kernel_config
    check_backend_support(kc, "cuda_v4_pwa")          # must not raise


def test_cpu_backends_agree():
    out = {}
    for be in ("numpy_pwa", "cpu_pwa_s0c", "cpu_pwa_gen", "cpu_v5_pwa"):
        f = _loaded(CFG_KM, be, n_phsp=200, n_data=120)
        x = f.initial_values(seed=3)
        out[be] = (f.get_nll(x)[0], f.get_nll(x)[1])
    ref_nll, ref_g = out["numpy_pwa"]
    for be, (nll, g) in out.items():
        assert abs(nll - ref_nll) / abs(ref_nll) < 1e-12, be
        assert np.max(np.abs(g - ref_g)) < 1e-10 * max(1.0, np.max(np.abs(ref_g))), be


# ── B1: mixed generic + plain resonance ──────────────────────────────

@pytest.mark.parametrize("backend", BACKENDS)
def test_mixed_kmatrix2_and_plain_resonance(backend):
    """A ``kmatrix2`` next to an ordinary ``BW`` builds and matches the
    direct basis amplitude.

    Before the fix the plain resonance's extra denominator rows were never
    tabulated, so the config raised ``KeyError ('bw', 2)``.
    """
    f = _loaded(_mixed_cfg(), backend, n_phsp=300, n_data=150)
    kc = f.kernel_config
    assert f.model._pg_generic
    assert kc["den_row"].shape[0] == 2
    x = f.initial_values(seed=1)
    params, resolved = f.build_params(x)
    _Q, _grads, P = f.backend.compute(params, f._data_holder, norm=None)
    P_dir = _direct_P(f, resolved, f._data_np)
    assert np.max(np.abs(P - P_dir) / P_dir) < 1e-10


def test_mixed_plain_resonance_is_bit_identical():
    """The plain resonance alone reproduces the non-generic (``m0``/``g0``)
    path bit for bit.

    ``chi``: the generic path tabulates ``−i·γ`` on the *same* gamma grid the
    plain path interpolates, so switching the k-matrix partner on must not
    change the BW wave at all.
    """
    from tabpwa.numpy_pwa import NumpyPWA

    f = _loaded(_mixed_cfg(), "numpy_pwa", n_phsp=200, n_data=120)
    x = f.initial_values(seed=1)
    params, resolved = f.build_params(x)
    offsets = {nm: (off, mult) for nm, off, mult in _chain_offsets(f)}
    bw_off, bw_mult = offsets["bw"]

    # switch the k-matrix chain off (its ck slots -> 0)
    params = dict(params)
    ck = np.array(params["ck"], dtype=complex)
    keep = np.zeros(len(ck), bool)
    keep[bw_off:bw_off + bw_mult] = True
    ck[~keep] = 0.0
    params["ck"] = ck
    k_mix = f.backend.kernel
    Qm, _gm, Pm = k_mix.compute(params, k_mix.load_data(f._data_np), norm=None)

    # the same config/recoil without the k-matrix resonance: the non-generic
    # plain path (m0/g0, no den_row)
    d0 = _mixed_cfg()
    d0["particle"]["pipi"] = ["bw"]
    f0 = Fitter(d0, backend="numpy_pwa")
    assert not f0.model._pg_generic
    k0 = NumpyPWA(f0.kernel_config)
    p0 = {"ck": np.array([ck[bw_off]]),
          "m0": np.array([resolved["bw_mass"]]),
          "g0": np.array([resolved["bw_width"]])}
    Q0, _g0, P0 = k0.compute(p0, k0.load_data(f._data_np), norm=None)

    assert np.array_equal(Pm, P0)
    assert Qm == pytest.approx(Q0, rel=1e-14)


# ── B2: two distinct kmatrix2 resonances (slot-map indexing) ─────────

def test_two_kmatrix2_resonances_match_cpu_and_direct():
    """Two distinct k-matrices: ``numpy_pwa`` / ``cpu_pwa_s0c`` / the
    particle models' own ``Σ g_j G_j`` agree.

    With one unique BW the raw ``den_row`` encoding happens to equal the
    unique-slot map; two resonances in different topologies (different mass
    columns) break that coincidence, which is what the assertion pins down.
    """
    from tabpwa.numpy_kernel import NumpyKernel
    from tabpwa.pwa_gen_support import build_den_slots

    d = _two_km_cfg()
    fn = _loaded(d, "numpy_pwa", n_phsp=200, n_data=120)
    fc = _loaded(d, "cpu_pwa_s0c", n_phsp=200, n_data=120)
    kc = fn.kernel_config
    assert kc["den_row"].shape == (2, 5)

    k = NumpyKernel(kc)
    _kinds, _rows, _cols, slot_flat, _nu = build_den_slots(
        np.asarray(kc["den_row"]), np.asarray(k.mass_index))
    slots = slot_flat.reshape(kc["den_row"].shape)
    assert not np.array_equal(slots, np.asarray(kc["den_row"]))  # B2 guard

    xn = fn.initial_values(seed=3)
    xc = fc.initial_values(seed=3)
    assert np.array_equal(xn, xc)
    nll_n, g_n = fn.get_nll(xn)
    nll_c, g_c = fc.get_nll(xc)
    assert abs(nll_n - nll_c) / abs(nll_c) < 1e-10
    assert np.max(np.abs(g_n - g_c)) < 1e-9 * max(1.0, np.max(np.abs(g_c)))

    params, resolved = fn.build_params(xn)
    _Q, _g, P = fn.backend.compute(params, fn._data_holder, norm=None)
    P_dir = _direct_P(fn, resolved, fn._data_np)
    assert np.max(np.abs(P - P_dir) / P_dir) < 1e-10


# ── M1: n_blocks > 1 coefficient assembly ────────────────────────────

def test_cp_blocks_build_and_equal_the_block_free_nll():
    """``cp_particles`` (n_blocks = 2) must not break the generic
    coefficient assembly.

    One top spin state (``spins: [0]``) keeps ``n_proj = 1`` so the CP
    block is a pure second copy.  With the partner's ``ck`` switched off
    the block config is exactly the block-free config.
    """
    from tabpwa.numpy_pwa import NumpyPWA

    d0 = _cfg(CFG_KM)
    d0["particle"]["jpsi"]["spins"] = [0]
    dcp = _cfg(CFG_KM)
    dcp["particle"]["jpsi"]["spins"] = [0]
    dcp["data"]["cp_particles"] = [["pip", "pim"]]

    f0 = _loaded(d0, "numpy_pwa", n_phsp=200, n_data=120)
    fc = _loaded(dcp, "numpy_pwa", n_phsp=200, n_data=120)
    assert int(fc.kernel_config["n_blocks"]) == 2
    assert int(f0.kernel_config.get("n_blocks", 1)) == 1

    # the assembly itself: forward/backward must not raise (IndexError before)
    xc = fc.initial_values(seed=2)
    nll_c, g_c = fc.get_nll(xc)
    assert np.isfinite(nll_c) and np.all(np.isfinite(g_c))

    params, _resolved = fc.build_params(xc)
    nb = len(params["ck"]) // 2
    assert np.array_equal(params["coeff"][:nb], params["coeff"][nb:])

    # block-free equivalent: first block only, partner ck = 0
    params = dict(params)
    params["ck"] = np.concatenate([params["ck"][:nb],
                                   np.zeros(nb, dtype=complex)])
    k_cp = fc.backend.kernel
    Qc, _gc, Pc = k_cp.compute(params, k_cp.load_data(fc._data_np), norm=None)

    p0 = {"ck": params["ck"][:nb], "coeff": params["coeff"][:nb],
          "m0": params["m0"], "g0": params["g0"]}
    k0 = NumpyPWA(f0.kernel_config)
    Q0, _g0, P0 = k0.compute(p0, k0.load_data(fc._data_np), norm=None)
    assert Qc == pytest.approx(Q0, rel=1e-12)
    assert np.allclose(Pc, P0, rtol=1e-12, atol=0)


# ── M2: generic config drops unused m0/g0 names ──────────────────────

def test_generic_drops_mass_width_not_consumed_by_the_kernel():
    """A generic fit must have no identically-zero flat direction.

    The generic kernels consume only ``coeff``; ``k2_mass``/``k2_width`` are
    not basis-transform inputs, so they used to be free with a zero
    gradient.  The parameter surface is exactly the ck slots + transform
    inputs now.
    """
    from tabpwa.param_constraint import complex_tail

    f = Fitter(CFG_KM, backend="numpy_pwa")
    names = f.free_param_names()
    assert "k2_mass" not in names
    assert "k2_width" not in names
    assert {"k2_mass", "k2_width"} & set(f.model._coef_transform().input_names) == set()

    tail = complex_tail()
    heads = {h for comb in f.model.get_ck_map() for h in comb
             if isinstance(h, str)}
    ck_slots = {h + t for h in heads for t in tail}
    tr = f.model._coef_transform()
    # the numerator slots are ck inputs AND transform outputs: the cm
    # resolves them, so they are not independent flat variables
    expected = (ck_slots | set(tr.input_names)) - set(tr.output_names)
    assert set(names) == expected

    # the removed names are not silently kept as zero-gradient entries
    x = f.initial_values(seed=1)
    assert len(x) == len(names)
    params, resolved = f.build_params(x)
    assert params["m0"].size == 0 and params["g0"].size == 0


# ── M4: numpy must refuse a generic config without ``coeff`` ─────────

def test_numpy_generic_requires_coeff():
    from tabpwa.numpy_pwa import NumpyPWA

    kc = Fitter(CFG_KM, backend="numpy_pwa").kernel_config
    kern = NumpyPWA(kc)
    data = _synth(kc, 10, 1)
    params = {"ck": np.ones(kern.n_wave_base, dtype=complex),
              "m0": np.zeros(1), "g0": np.zeros(1)}
    with pytest.raises(ValueError, match="requires params\\['coeff'\\]"):
        kern.compute(params, kern.load_data(data), norm=None)


# ── minor: zero-padding sentinel is distinct from the table encoding ──

def test_zero_kind_is_a_negative_sentinel():
    from tabpwa.pwa_gen_support import (ZERO_KIND, TABLE_KIND, CONST_KIND,
                                        MSQ_KIND, build_den_slots)
    assert ZERO_KIND < 0
    assert ZERO_KIND not in (CONST_KIND, MSQ_KIND, TABLE_KIND)
    # encoding ``3`` is a real table row (2 + 1), padding is the sentinel
    enc = np.array([[-1, 3]])
    kinds, rows, _cols, slot_flat, n_uslot = build_den_slots(enc, [0])
    slots = slot_flat.reshape(enc.shape)
    assert n_uslot == 2
    assert kinds[slots[0, 0]] == ZERO_KIND
    assert kinds[slots[0, 1]] == TABLE_KIND
    assert rows[slots[0, 1]] == 1


# ── minor: one gamma-grid accessor ───────────────────────────────────

def test_gamma_grid_accessor_is_shared():
    f = Fitter(CFG_KM, backend="numpy_pwa")
    grid, g_min, g_delta = f.model.gamma_grid()
    _table, t_min, t_delta = f.model.build_gamma_table()
    assert g_min == t_min
    assert g_delta == t_delta
    assert np.array_equal(np.asarray(grid), np.linspace(
        f.model.get_max_mass_range()[0] - 0.01,
        f.model.get_max_mass_range()[1] + 0.01, grid.size))
    pg_grid, pg_min, pg_delta = f.model._pg_grid()
    assert np.array_equal(pg_grid, grid)
    assert (pg_min, pg_delta) == (g_min, g_delta)

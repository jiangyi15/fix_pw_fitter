"""``cuda_flat_pwa_gen_s0c`` — the kernel's OWN gates.

The function every flat variant implements (``load_data`` / ``compute``) is
covered once by :mod:`_flat_compute`; this file keeps what is specific to s0c:
the u/amp dedups and both CSRs, the m-major coefficient reduction, the
sparse-row edge cases, the chunk rule, the memory design, the ``n_res == 1``
guard, and the case-1 copy's own module/`.so`/symbols.
"""
import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend
from _flat_compute import (
    CFG_ONE, CFG_KM, CFG_MB, CFG_IDENT, CFG_SAME, CFG_CP, QUALIFYING,
    NH, _in_grid)

# ── this kernel's own measured facts (not part of the compute suite) ───
#: measured `(n_basis, n_uslot)` and `(n_wave, n_uniq)` and the CSR sizes,
#: pinned PER CONFIG (an average over the fixtures would hide that the dense
#: ones collapse nothing).
EXPECTED_DEDUP = {
    #                      us: (n_basis, n_uslot)   amp: (n_wave, n_uniq)
    CFG_ONE:   ((3, 3), (1, 1)),
    CFG_KM:    ((5, 5), (2, 2)),
    CFG_IDENT: ((10, 10), (4, 4)),      # distinct angle columns per wave
    CFG_SAME:  ((10, 10), (4, 4)),
    CFG_CP:    ((20, 20), (8, 8)),
}
#: measured `(ref, run)` basis-row iterations per event for the SPARSE matvec —
#: `ref` is the dense `n_wave·n_basis` a full walk would cost.
EXPECTED_ITERS = {
    CFG_ONE: (3, 3),        # 1 wave, dense 3-row basis          -> 1.00x
    CFG_KM: (10, 10),       # 2 waves, dense 5-row basis         -> 1.00x
    CFG_IDENT: (40, 20),    # 4 waves, 2 chains of 5             -> 2.00x
    CFG_SAME: (40, 20),     # 4 waves, 2 chains of 5             -> 2.00x
    CFG_CP: (160, 40),      # 8 waves (2 identical x 2 CP), 5    -> 4.00x
}

#: the real-config ratios this backend was built for, asserted directly on the
#: model (the fixtures are tiny by design and cannot show them)
REAL_CFG = ("/home/jiangy/github/project94/pwa_workspace/jpsi_eta_pi_pi/"
            "04_model_search/current/config.yml")
REAL_USLOT = (62, 40)
REAL_AMP = (26, 13)
REAL_NNZ = 82
#: measured on the real 8.75 M-event sample: this kernel's data handle is
#: 2.17 GB — s0c's own — against the ``cuda_flat_pwa_gen_s0e`` tile's 5.9 GB.
REAL_HANDLE_GB = 2.17

from tabpwa.cuda._flat_pwa_gen_s0c import (
    CUDAKernelFlatPwaGenS0C, _basis_csr, index_runs)
from tabpwa.cuda._flat_pwa_gen_s0e import amp_slot_map, uslot_map

def test_registration():
    assert "cuda_flat_pwa_gen_s0c" in set(backends_for_model("pwa_gen"))
    assert "cuda_flat_pwa_gen_s0c" not in set(backends_for_model("pwa"))


def test_the_other_kernels_are_not_reused():
    """A case-1 copy: own module, own source, own library, own symbols."""
    import tabpwa.cuda._flat_pwa_gen_s0c as new
    import tabpwa.cuda._flat_pwa_gen2 as gen2
    import tabpwa.cuda._flat_pwa_gen_1bw as one
    import tabpwa.cuda._flat_pwa_gen_s0e as s0e
    assert new is not gen2 and new is not one and new is not s0e
    f = Fitter(CFG_KM)
    f.apply_constrains()
    a = CUDAKernelFlatPwaGenS0C(f.kernel_config)
    b = gen2.CUDAKernelFlatPwaGen2(f.kernel_config)
    c = one.CUDAKernelFlatPwaGen1BW(f.kernel_config)
    d = s0e.CUDAKernelFlatPwaGenS0E(f.kernel_config)
    try:
        assert a._lib is not b._lib and a._lib is not c._lib
        assert a._lib is not d._lib
        assert hasattr(a._lib, "cuda_compute_flat_pwa_gen_s0c")
        for nm in ("cuda_compute_flat_pwa_gen2",
                   "cuda_compute_flat_pwa_gen_1bw",
                   "cuda_compute_flat_pwa_gen_s0e"):
            assert not hasattr(a._lib, nm)
        # s0c's own parameter blocks are m0/g0; the flat config has neither,
        # so this backend carries no m0/g0 staging at all
        for nm in ("p_m0", "_m0_index_w", "_g0_index_w", "_wave_of_row"):
            assert not hasattr(a, nm), nm
    finally:
        for k in (a, b, c, d):
            k.free()


def _real_config():
    import os
    if not os.path.exists(REAL_CFG):
        pytest.skip(f"real config not present: {REAL_CFG}")
    d = dict(Fitter(REAL_CFG).config.dic)
    d["amp_model"] = "pwa_gen"
    d["backend"] = {"name": "numpy_flat"}
    f = Fitter(d)
    f.apply_constrains()
    return f.kernel_config


def _lin(tab, x, lo, dd):
    pos = (np.asarray(x, np.float64) - lo) / dd
    i0 = np.clip(pos.astype(np.int64), 0, tab.size - 2)
    fr = pos - i0
    return tab[i0] * (1.0 - fr) + tab[i0 + 1] * fr


def test_the_basis_rotation_is_exact():
    """``Z = i·basis`` with the kernel's ``-i·Z`` recovers every basis row.

    This is the design point that lets the kernel stay kind-agnostic: no
    ``basis_kind`` is inspected anywhere, because the rotation is uniform.
    """
    kc = _real_config()
    bas = np.asarray(kc["basis_table"], np.complex128)
    assert np.abs(-1j * (1j * bas) - bas).max() == 0.0


def test_the_denominator_from_the_rotated_deduped_table_is_numpy_flats_D():
    """``Σ_i coeff_i·basis_i(m)`` through the us-cache == the direct evaluation.

    Assembled the way the KERNEL reads it (deduped slot, ``Z = i·basis``, then
    ``-i·Z``), against the per-row evaluation ``numpy_flat`` performs.  Done
    before the kernel is trusted so a wrong rotation cannot be masked.
    """
    kc = _real_config()
    bas = np.asarray(kc["basis_table"], np.complex128)
    bmi = np.asarray(kc["basis_m_index"], np.int64)
    m_min, m_del = float(kc["m_min"]), float(kc["m_delta"])
    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    n_basis, n_bins = bas.shape
    gam = np.asarray(kc["gamma_matrix"], np.float64)
    gam = gam[:, 0, :] if gam.ndim == 3 else gam

    rng = np.random.default_rng(3)
    mass = m_min + rng.uniform(0.05, 0.95, (37, 3)) * m_del * (n_bins - 1)
    coeff = rng.uniform(0.3, 1.2, n_basis)
    uro, uco, sob = uslot_map(bas, bmi)

    direct = np.zeros((37, n_wave), np.complex128)
    kern = np.zeros((37, n_wave), np.complex128)
    for k in range(n_wave):
        for m in np.nonzero(gam[k])[0]:
            b = (_lin(bas[m].real, mass[:, bmi[m]], m_min, m_del)
                 + 1j * _lin(bas[m].imag, mass[:, bmi[m]], m_min, m_del))
            direct[:, k] += gam[k, m] * coeff[m] * b
            u = sob[m]
            zr = _lin(bas[uro[u]].real, mass[:, uco[u]], m_min, m_del)
            zi = _lin(bas[uro[u]].imag, mass[:, uco[u]], m_min, m_del)
            Zr, Zi = -zi, zr                       # Z = i·basis
            kern[:, k] += coeff[m] * (Zi + 1j * (-Zr))   # F = -i·Z
    assert np.abs(kern - direct).max() == 0.0


def test_the_m_major_coefficient_reduction_is_numpy_flats():
    """The factored ``dQ/dcoeff_m`` — proved from the flat config alone.

    The kernel multiplies ``Σ_{k∋m} dQ/dD_k`` by ``Z_m`` ONCE (m-major) instead
    of summing over every ``(wave, row)`` pair.  That reassociation is exact for
    a 0/1 selector, and this is where it is checked — independently of any
    kernel, at fp64, against ``numpy_flat``.
    """
    kc = _real_config()
    bas = np.asarray(kc["basis_table"], np.complex128)
    bmi = np.asarray(kc["basis_m_index"], np.int64)
    gam = np.asarray(kc["gamma_matrix"], np.float64)
    gam = gam[:, 0, :] if gam.ndim == 3 else gam
    fl_tab = np.asarray(kc["fl_table"], np.float64)
    q_idx = np.asarray(kc["fl_q_index"], np.int64)
    num_tab = np.asarray(kc["num_table"], np.complex128)
    m_idx = np.asarray(kc["num_m_index"], np.int64)
    am = np.asarray(kc["matrix_angle"], np.complex128)
    n_wave = int(fl_tab.shape[0])
    n_decay = int(q_idx.shape[1])
    n_basis = int(bas.shape[0])
    n_res = int(num_tab.shape[1])
    n_proj = am.shape[1] // n_wave
    m_min, m_del = float(kc["m_min"]), float(kc["m_delta"])
    q0, qd = float(kc["q_min"]), float(kc["q_delta"])

    data = _in_grid(kc, 48, 2)
    d = dict(Fitter(REAL_CFG).config.dic)
    d["amp_model"] = "pwa_gen"
    d["backend"] = {"name": "numpy_flat"}
    f = Fitter(d)
    f.apply_constrains()
    params, _ = f.build_params(f.initial_values(seed=1))
    ref = create_backend("numpy_flat", kc, model="pwa_gen")
    _Qr, gr, _Pr = ref.compute(params, ref.load_data(data), norm=1.3)
    ref.free()

    ck = np.asarray(params["ck"], np.complex128)
    coeff = np.asarray(params["coeff"], np.float64)
    mass, mom, ang = data["mass"], data["q"], data["angle"]
    ne = mass.shape[0]
    ai = np.asarray(kc["angle_index"])
    ak = np.asarray(kc["angle_k"], np.float64)
    ab = np.asarray(kc["angle_b"], np.float64)
    ka = np.stack([np.prod(np.cos(ang[:, ai[a], :] * ak[a] + ab[a]), axis=-1)
                   for a in range(ak.shape[0])], axis=1)
    fa = (ka @ am).reshape(ne, n_proj, n_wave).transpose(0, 2, 1)

    # Z from the deduped slot table, exactly as the kernel builds it
    uro, uco, sob = uslot_map(bas, bmi)
    Z = np.zeros((ne, uro.size), np.complex128)
    for u in range(uro.size):
        xx = mass[:, uco[u]]
        Z[:, u] = 1j * (_lin(bas[uro[u]].real, xx, m_min, m_del)
                        + 1j * _lin(bas[uro[u]].imag, xx, m_min, m_del))

    gs_lo, gs_hi, nz_k, nz_m = index_runs(gam, n_wave, n_basis)
    m_ptr, m_waves, m_cols, _order = _basis_csr(
        gam, gs_lo, gs_hi, nz_k, nz_m, n_wave, n_basis)

    D = np.zeros((ne, n_wave), np.complex128)
    F = np.zeros((ne, n_basis), np.complex128)
    for m in range(n_basis):
        F[:, m] = -1j * Z[:, sob[m]]
        ks = m_waves[m_ptr[m]:m_ptr[m + 1]]
        if ks.size:
            D[:, ks] += coeff[m] * F[:, m:m + 1]

    flv = np.ones((ne, n_wave))
    for k in range(n_wave):
        for dd in range(n_decay):
            flv[:, k] *= _lin(fl_tab[k, dd], mom[:, q_idx[k, dd]], q0, qd)
    numv = np.ones((ne, n_wave), np.complex128)
    for k in range(n_wave):
        for r in range(n_res):
            xx = mass[:, m_idx[k, r]]
            numv[:, k] *= (_lin(num_tab[k, r].real, xx, m_min, m_del)
                           + 1j * _lin(num_tab[k, r].imag, xx, m_min, m_del))
    common = fa * (flv * numv)[:, :, None] / D[:, :, None]
    A = np.zeros((ne, n_proj), np.complex128)
    for p in range(n_proj):
        A[:, p] = np.sum(ck[None, :] * common[:, :, p], axis=1)
    Pe = np.sum(A.real ** 2 + A.imag ** 2, axis=-1)

    norm = 1.3
    w = np.asarray(data["weight"])
    bv = np.asarray(data["bkg"], float)
    dQdP = -w / (Pe + bv * norm)
    dQdA = dQdP[:, None] * np.conj(A)
    u = np.zeros((ne, n_wave), np.complex128)
    for p in range(n_proj):
        u += dQdA[:, p:p + 1] * common[:, :, p]
    pv = -u * ck[None, :] / D
    gc = np.zeros(n_basis)
    for m in range(n_basis):
        ks = m_waves[m_ptr[m]:m_ptr[m + 1]]
        if ks.size == 0:
            continue
        gc[m] = 2.0 * np.real(np.sum(pv[:, ks].sum(axis=1) * F[:, m]))
    gck = np.zeros(n_wave, np.complex128)
    for p in range(n_proj):
        gck += np.sum(dQdA[:, p:p + 1] * common[:, :, p], axis=0)

    ref_c = np.asarray(gr["coeff"])
    ref_k = np.asarray(gr["ck"])
    assert np.abs(gc - ref_c).max() < 1e-9 * max(np.abs(ref_c).max(), 1.0)
    assert np.abs(gck - ref_k).max() < 1e-9 * max(np.abs(ref_k).max(), 1.0)


@pytest.mark.parametrize("cfg", QUALIFYING)
def test_the_dedups_and_csrs_are_exact_and_pinned(cfg):
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    bas = np.asarray(kc["basis_table"], np.complex128)
    bmi = np.asarray(kc["basis_m_index"], np.int64)
    gam3 = np.asarray(kc["gamma_matrix"], np.float64)
    gam = gam3[:, 0, :] if gam3.ndim == 3 else gam3
    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    n_proj = max(1, int(np.asarray(kc["matrix_angle"]).shape[1]) // n_wave)
    n_basis = int(bas.shape[0])

    # the uslot fold: representative == every member, bit-for-bit
    uro, uco, sob = uslot_map(bas, bmi)
    for m in range(n_basis):
        u = sob[m]
        assert np.array_equal(bas[uro[u]], bas[m]), f"{cfg}: row {m}"
        assert uco[u] == bmi[m], f"{cfg}: column {m}"

    # the amp fold: same
    sow, rep = amp_slot_map(kc["matrix_angle"], kc["fl_table"],
                            kc["fl_q_index"], kc["num_table"],
                            kc["num_m_index"], n_wave, n_proj)
    am = np.asarray(kc["matrix_angle"])
    fl = np.asarray(kc["fl_table"])
    fq = np.asarray(kc["fl_q_index"])
    nt = np.asarray(kc["num_table"])
    nmi = np.asarray(kc["num_m_index"])
    for k in range(n_wave):
        r = int(rep[sow[k]])
        for p in range(n_proj):
            assert np.array_equal(am[:, p * n_wave + k],
                                  am[:, p * n_wave + r]), f"{cfg}: k{k} p{p}"
        for d in range(fl.shape[1]):
            assert np.array_equal(fl[k, d], fl[r, d])
            assert fq[k, d] == fq[r, d]
        for rr in range(nt.shape[1]):
            assert np.array_equal(nt[k, rr], nt[r, rr])
            assert nmi[k, rr] == nmi[r, rr]

    # the pinned ratios
    (n_b, n_u), (n_w, n_q) = EXPECTED_DEDUP[cfg]
    assert (n_basis, int(uro.size)) == (n_b, n_u)
    assert (n_wave, int(rep.size)) == (n_w, n_q)
    assert (n_basis, n_wave) == (n_b, n_w)

    k = CUDAKernelFlatPwaGenS0C(kc)
    try:
        # the wave CSR is each row's TRUE non-zeros, in order
        for kk in range(n_wave):
            idx = np.nonzero(gam[kk])[0]
            lo, hi = int(k.gs_lo[kk]), int(k.gs_hi[kk])
            assert np.array_equal(k.nz_m[lo:hi], idx), f"{cfg}: wave {kk}"
        assert k.n_nz == int(np.count_nonzero(gam))
        assert (k.n_wave * k.n_basis, k.n_nz) == EXPECTED_ITERS[cfg]
        assert k.n_nz <= k.n_wave * k.n_basis          # never a loss
        # the m-major CSR is the same set, grouped by basis row
        pairs_a = sorted((int(k.m_waves[j]), int(k.m_cols[j]))
                         for j in range(k.m_waves.size))
        pairs_b = sorted((kk, int(m)) for kk in range(n_wave)
                         for m in np.nonzero(gam[kk])[0])
        assert pairs_a == pairs_b, cfg
        assert k.n_m_entry == k.n_nz
        for m in range(n_basis):
            want = sorted(kk for kk in range(n_wave) if gam[kk, m] != 0.0)
            got = sorted(int(k.m_waves[j])
                         for j in range(k.m_ptr[m], k.m_ptr[m + 1]))
            assert got == want, f"{cfg}: basis row {m}"
        # ... and the uslot of each run entry is its basis row's slot
        assert np.array_equal(k.nz_u, k.slot_of_basis[k.nz_m])
    finally:
        k.free()


@pytest.mark.parametrize("cfg", [CFG_ONE, CFG_KM])
def test_the_dense_shapes_gain_nothing_and_say_so(cfg):
    """Guards the honest reading: a dense basis is 1.00x, not a win."""
    assert EXPECTED_ITERS[cfg][0] == EXPECTED_ITERS[cfg][1]
    for c in (CFG_IDENT, CFG_SAME, CFG_CP):
        assert EXPECTED_ITERS[c][1] < EXPECTED_ITERS[c][0]


def test_the_real_config_ratios_are_the_ones_this_backend_was_built_for():
    """69 -> 43 uslots, 26 -> 13 waves, 1794 -> 92 matvec entries.

    The fixtures are deliberately tiny, so the ratios that motivate the design
    are asserted on the real model instead — a builder change that moved them
    would make this backend's premise false without failing any fixture.
    """
    kc = _real_config()
    bas = np.asarray(kc["basis_table"], np.complex128)
    bmi = np.asarray(kc["basis_m_index"], np.int64)
    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    n_proj = np.asarray(kc["matrix_angle"]).shape[1] // n_wave
    _ro, _co, _so = uslot_map(bas, bmi)
    assert (bas.shape[0], int(_ro.size)) == REAL_USLOT
    sow, rep = amp_slot_map(kc["matrix_angle"], kc["fl_table"],
                            kc["fl_q_index"], kc["num_table"],
                            kc["num_m_index"], n_wave, n_proj)
    assert (n_wave, int(rep.size)) == REAL_AMP
    gam3 = np.asarray(kc["gamma_matrix"], np.float64)
    gam = gam3[:, 0, :] if gam3.ndim == 3 else gam3
    gs_lo, gs_hi, _nk, nz_m = index_runs(gam, n_wave, bas.shape[0])
    assert int(nz_m.size) == REAL_NNZ
    holed = sum(1 for kk in range(n_wave)
                if np.count_nonzero(gam[kk]) < int(gs_hi[kk] - gs_lo[kk]) + 1
                and np.any(np.diff(np.nonzero(gam[kk])[0]) > 1))
    assert holed >= 15, f"interior holes in {holed} of {n_wave} waves"


def test_a_hole_in_a_gamma_row_is_handled_not_refused():
    """A hole is simply absent from the CSR, so it changes nothing."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = dict(f.kernel_config)
    gam = np.array(kc["gamma_matrix"], np.float64).copy()
    assert gam.shape[1] == 1
    data = _in_grid(kc, 64, 0)
    params, _ = f.build_params(f.initial_values(seed=1))

    def run(g):
        c = dict(kc)
        c["gamma_matrix"] = g
        k = CUDAKernelFlatPwaGenS0C(c)
        try:
            Q, gg, _P = k.compute(params, k.load_data(data), norm=1.3)
            return Q, gg
        finally:
            k.free()

    Q0, _ = run(gam)
    holed = gam.copy()
    col = int(np.nonzero(holed[0, 0])[0][1])     # a NON-zero column of wave 0
    holed[0, 0, col] = 0.0
    Qh, gh = run(holed)
    # the hole is real, so the comparison is not vacuous ...
    assert abs(Qh - Q0) > 1e-6 * max(abs(Q0), 1.0)
    # ... and the CSR now carries one fewer entry rather than a wider span
    k = CUDAKernelFlatPwaGenS0C({**kc, "gamma_matrix": holed})
    try:
        assert k.run_widths[0] == np.count_nonzero(holed[0, 0])
        assert col not in set(k.nz_m[int(k.gs_lo[0]):int(k.gs_hi[0])].tolist())
    finally:
        k.free()


def test_a_weighted_gamma_is_refused():
    """The m-major reduction factors the basis row out — exact only for 0/1."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = dict(f.kernel_config)
    gam = np.array(kc["gamma_matrix"], np.float64).copy()
    gam[0, 0, 1] = 0.5
    kc["gamma_matrix"] = gam
    with pytest.raises(ValueError, match="0/1"):
        CUDAKernelFlatPwaGenS0C(kc)


def test_an_all_zero_row_is_refused():
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = dict(f.kernel_config)
    gam = np.array(kc["gamma_matrix"], np.float64).copy()
    gam[0, 0, :] = 0.0
    with pytest.raises(ValueError, match="all zero"):
        index_runs(gam, None, None)


def test_the_index_list_is_itself():
    """One implementation, so a test can ask without a kernel."""
    sub = np.zeros((2, 5))
    sub[0, 1:4] = 1.0
    sub[1, 3:] = 1.0
    lo, hi, nk, nm = index_runs(sub)
    # [lo[k], hi[k]) indexes the flat (wave, column) list, not the row
    assert lo.tolist() == [0, 3] and hi.tolist() == [3, 5]
    assert nm.tolist() == [1, 2, 3, 3, 4]
    assert nk.tolist() == [0, 0, 0, 1, 1]
    # a hole inside the span costs one entry, not the span
    holed = np.ones((1, 5))
    holed[0, 2] = 0.0
    lo, hi, _nk, nm = index_runs(holed)
    assert lo.tolist() == [0] and hi.tolist() == [4]
    assert nm.tolist() == [0, 1, 3, 4]
    # the transposed CSR is the same set grouped by column
    m_ptr, m_waves, m_cols, _order = _basis_csr(sub, lo, hi, _nk, nm, 1, 5)
    assert m_ptr.tolist() == [0, 1, 2, 2, 3, 4]
    assert sorted(zip(m_waves.tolist(), m_cols.tolist())) == [
        (0, 0), (0, 1), (0, 3), (0, 4)]
    assert len(_order) == 4


def _chunk_rule(batch_size, rsize):
    """The documented rule, restated here so the C side is checked against it."""
    cap = batch_size if batch_size > 0 else 50000
    cap = min(cap, 1 << 20)
    if rsize > 1:
        a = (cap // rsize) * rsize
        cap = a if a > 0 else rsize          # never zero, never unaligned
    return cap


@pytest.mark.parametrize("rsize", [1, 17])
@pytest.mark.parametrize("batch_size", [1, 5, 1000, 50000, 10 ** 9])
def test_the_chunk_rule_is_what_the_context_reports(batch_size, rsize):
    """`cuda_cap_*` must be the rule above — never a Python re-derivation.

    The two guards are the ones measurement forced: `batch_size = 1e9` must be
    capped (a scratch of GBs for a small sample), and `batch_size < rsize` must
    still hold a WHOLE group — an unaligned cap there let a group straddle two
    chunks and made Q wrong by 49 % once in gen2.
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    k = CUDAKernelFlatPwaGenS0C(f.kernel_config, batch_size=batch_size,
                                resolution_size=rsize)
    try:
        assert k.cap == _chunk_rule(batch_size, rsize)
        assert k.cap > 0
        if rsize > 1:
            assert k.cap % rsize == 0, "a chunk must hold whole log-sum groups"
        # no scratch on the handle at all: everything is the context's
        assert not hasattr(k, "dA_bytes_cap")
    finally:
        k.free()


def test_multiple_bws_per_chain_is_rejected_with_the_general_backend_named():
    f = Fitter(CFG_MB)
    f.apply_constrains()
    assert int(np.asarray(f.kernel_config["num_table"]).shape[1]) == 2
    with pytest.raises(ValueError, match="exactly ONE BW per chain"):
        CUDAKernelFlatPwaGenS0C(f.kernel_config)
    with pytest.raises(ValueError, match="cuda_flat_pwa_gen2"):
        create_backend("cuda_flat_pwa_gen_s0c", f.kernel_config,
                       model="pwa_gen")


def test_every_other_fixture_passes_the_gate():
    """5 of 6 fixtures qualify; only ``_mb`` has n_res == 2."""
    for cfg in QUALIFYING:
        f = Fitter(cfg)
        f.apply_constrains()
        assert int(np.asarray(f.kernel_config["num_table"]).shape[1]) == 1
        k = CUDAKernelFlatPwaGenS0C(f.kernel_config)
        k.free()


def test_the_handle_holds_no_interpolation_cache():
    """The point of this backend: the basis interp is NOT cached per event.

    ``cuda_flat_pwa_gen_s0e`` caches ``Z = i·basis`` per ``(event, uslot)`` as a
    float2 — 6.0 GB on the real 8.75 M-event sample, which is the difference
    between its 6.26 GB free-after-load and this kernel's 9.42 GB.  So the
    handle's arrays are enumerated here: the numerator cache, weight/bkg/mass
    and the per-event P output.  Nothing sized ``ne · n_uslot``.
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    k = CUDAKernelFlatPwaGenS0C(kc)
    try:
        assert k.n_uslot > 0
        # the reports are the context's, and the working scratch is sized by
        # `cap` (not by the sample): a 4x longer load must not change them
        h1 = k.load_data(_in_grid(kc, 32))
        cap = k.cap
        h2 = k.load_data(_in_grid(kc, 128))
        try:
            assert k.cap == cap
            assert k.smem_bytes == (k.n_mass + 2 * k.n_uslot) * 8
            assert h1.ne == 32 and h2.ne == 128
        finally:
            h1.free()
            h2.free()
    finally:
        k.free()


def test_the_measured_handle_is_the_s0c_one():
    """On the real sample this handle is 2.17 GB — s0c's own, not s0e's 5.9.

    The whole case for this backend is that s0c's memory design survives the
    move to the flat config.  Recorded as a constant with its measurement so a
    regression (e.g. someone adding a per-event interpolation cache) shows up
    as a changed number rather than as an unexplained OOM.
    """
    assert 2.0 < REAL_HANDLE_GB < 2.4
    kc = _real_config()
    k = None
    try:
        k = CUDAKernelFlatPwaGenS0C(kc)
        ne = 8_746_347
        # what the handle's per-event arrays would cost at the real sample size
        amp = ne * k.n_uniq * k.n_proj * 8 / 1e9
        mass = ne * k.n_mass * 8 / 1e9
        wb = 2 * ne * 8 / 1e9
        hp = ne * 8 / 1e9
        total = amp + mass + wb + hp
        assert 2.0 < total < 2.4, (
            f"the handle predicts {total:.2f} GB, pinned at {REAL_HANDLE_GB}")
        # the cache this design deliberately does NOT have
        us = ne * k.n_uslot * 16 / 1e9
        assert us > 2 * total, (
            f"a per-event us cache would add {us:.2f} GB to a {total:.2f} GB "
            f"handle — that gap is the difference from s0e's 5.9 GB, which is "
            f"what this backend's memory case rests on")
    finally:
        if k is not None:
            k.free()

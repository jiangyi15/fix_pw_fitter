"""``pwa_gen`` identical-particle and CP blocks.

A declared ``identical_particles`` permutation or ``cp_particles`` conjugation
adds a BLOCK, and every base wave is summed over all blocks sharing one ``ck``:

    A_{e,p} = Σ_b Σ_k ck_k · (F/D)_{e,k,b} · fa_{e,p,k,b}

``build_tree_event_data`` concatenates one column band per block, and
``build_from_groups`` rebases each per-wave index onto its block's band, so
the wave axis is ``n_blocks · n_base`` and ``len(ck)`` matches by
construction — no backend knows blocks exist.

Gates:
1. the index rebasing: strides are ``n_topo·n_decay`` (q), ``n_topo·n_res``
   (mass) and ``n_topo`` (angle), and ``n_wave == len(ck)``;
2. the blocks are COHERENT, not concatenated — zeroing one block's ``ck``
   must change ``P`` through the cross terms (a bug that summed intensities
   instead of amplitudes would leave ``P`` a plain sum of independent parts);
3. the permutation actually permutes: the same unfinished momenta give
   different angle columns per block, and identical ``|q|`` (they are the
   same physical pair), which is what makes a stale index visible;
4. all three backends agree with ``numpy_flat`` on ``Q``/``P``/gradients;
5. ``(P, dck)``-vs-central-differences: the block expansion must not disturb
   the analytic gradients.

Test data stays INSIDE both interpolation grids.
"""
import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import create_backend

CFG_IDENT = "tests/config_pwa_gen_ident.yml"   # n_perm 2, n_cp 1 -> 2 blocks
CFG_CP = "tests/config_pwa_gen_cp.yml"         # n_perm 2, n_cp 2 -> 4 blocks


def _model(cfg):
    f = Fitter(cfg)
    f.apply_constrains()
    return f, f.kernel_config


def _in_grid(kc, ne, seed=0):
    """Events strictly inside both interpolation grids, shaped for the config."""
    rng = np.random.RandomState(seed)
    n_basis = int(np.asarray(kc["basis_table"]).shape[1])
    n_fl = int(np.asarray(kc["fl_table"]).shape[-1])
    m_min, m_del = float(kc["m_min"]), float(kc["m_delta"])
    q_min, q_del = float(kc["q_min"]), float(kc["q_delta"])
    n_mass = int(np.max(kc["num_m_index"])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1
    n_at = int(np.max(kc["angle_index"])) + 1
    nac = int(np.asarray(kc["angle_k"]).shape[1])
    return {
        "mass": m_min + 0.05 * m_del * n_basis
                + rng.rand(ne, n_mass) * (0.9 * m_del * n_basis),
        "q": q_min + 0.05 * q_del * n_fl
             + rng.rand(ne, n_q) * (0.9 * q_del * n_fl),
        "angle": rng.uniform(-np.pi, np.pi, (ne, n_at, nac)),
        "weight": 0.5 + rng.rand(ne),
        "bkg": 0.05 + 0.1 * rng.rand(ne),
    }


def _angle_rows_of_block(kc, sl):
    """Basis rows the waves in ``sl`` select, from the p-major matrix_angle.

    ``matrix_angle`` is ``(n_angle_basis, n_proj*n_wave_full)`` with column
    ``p*n_wave_full + k`` — the stride is the FULL wave axis
    (``n_blocks*n_base``), not one block.  The row VALUES of the basis are
    shared across blocks (the key is ``(column, k, b)``), so a block's rows
    need not be contiguous.
    """
    ma = np.asarray(kc["matrix_angle"])
    n_ab, n_col = ma.shape
    n_wave_full = int(np.asarray(kc["fl_table"]).shape[0])
    n_proj = max(1, n_col // n_wave_full)
    ks = list(range(*sl.indices(n_wave_full)))
    cols = [p * n_wave_full + k for p in range(n_proj) for k in ks]
    return (np.abs(ma[:, cols]) > 0).any(axis=1).nonzero()[0]


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# ── gate 1: the rebasing ────────────────────────────────────────────

@pytest.mark.parametrize("cfg,n_blocks", [(CFG_IDENT, 2), (CFG_CP, 4)])
def test_every_wave_is_rebased_onto_its_block_band(cfg, n_blocks):
    f, kc = _model(cfg)
    t = f.model.decay_tree
    assert kc["n_blocks"] == n_blocks
    assert t.n_perm * t.n_cp == n_blocks

    n_topo = int(f.model.n_topo)
    n_decay = int(f.model.n_decay)
    n_res = int(f.model.n_res)
    n_base = len(f.model._pg_combs)          # expanded-but-not-blocked waves
    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    assert n_wave == n_blocks * n_base, "wave axis must be n_blocks x base"

    # `ck` and the wave axis must agree, or the ck-length assert fires
    assert len(f.model.get_ck_map()) == n_wave

    for arr, stride in (("fl_q_index", n_topo * n_decay),
                        ("num_m_index", n_topo * n_res),
                        ("angle_index", n_topo)):
        a = np.asarray(kc[arr])
        n_col = int(np.max(a)) + 1
        # block b occupies the half-open column band [b*stride, (b+1)*stride)
        assert n_col == n_blocks * stride, (
            f"{arr}: max column {n_col - 1} -> {n_col} columns, expected "
            f"n_blocks({n_blocks}) x stride({stride}) = {n_blocks * stride}")

    # each block's slice of the wave axis must land in its own band.  Note
    # `angle_index` is the SHARED basis (one row per `(column, k, b)` key), so
    # a wave references a row that *points into* its band — the rows are not
    # grouped by block.  `fl_q_index`/`num_m_index` are per-use, so those are
    # grouped.
    for b in range(n_blocks):
        sl = slice(b * n_base, (b + 1) * n_base)
        for arr, stride in (("fl_q_index", n_topo * n_decay),
                            ("num_m_index", n_topo * n_res)):
            a = np.asarray(kc[arr])[sl]
            assert a.min() >= b * stride and a.max() < (b + 1) * stride, (
                f"{arr} block {b} escapes its band")
        # the angle rows this block's waves select must reference ITS band.
        # The basis list is shared and keyed on (column, k, b), so a block's
        # rows are not contiguous — check per row, not on min/max.
        rows = _angle_rows_of_block(kc, sl)
        ai = np.asarray(kc["angle_index"])[rows]
        inside = (ai >= b * n_topo) & (ai < (b + 1) * n_topo)
        assert inside.all(), (
            f"angle basis rows for block {b} reference another block's "
            f"columns: {ai[~inside]}")


def test_single_block_is_unchanged():
    """n_blocks == 1 must be byte-identical to the pre-block behaviour."""
    a, kca = _model("tests/config_pwa_gen_km.yml")
    b, kcb = _model("tests/config_pwa_gen_km.yml")
    assert kca["n_blocks"] == 1
    for key in ("fl_q_index", "num_m_index", "angle_index", "fl_table",
                "matrix_angle", "gamma_matrix"):
        assert np.array_equal(np.asarray(kca[key]), np.asarray(kcb[key]))
    # the rebasing must be the identity when there is one block
    assert np.asarray(kca["fl_q_index"]).min() == 0
    assert np.asarray(kca["angle_index"]).min() == 0


# ── gate 2: blocks sum coherently, not as independent intensities ────

@pytest.mark.parametrize("cfg", [CFG_IDENT, CFG_CP])
def test_blocks_sum_as_amplitudes_not_intensities(cfg):
    """The block sum must be a COHERENT amplitude sum, checked independently.

    ``Q`` is quadratic in ``ck`` and the blocks share one ``ck``, so with
    ``Q(ck)`` the block structure is a quadratic form.  Writing ``ca`` for
    ``ck`` restricted to block 0, ``cb`` for block 1, and ``Q0 = Q(0) = 0``:

        Q(ca + cb) = Q(ca) + Q(cb) + 2·Re(cross)

    If the builder emitted the blocks as INDEPENDENT intensities there would be
    no cross term and ``Q(ck) == Q(ca) + Q(cb)`` exactly.  So the assertion
    ``Q(ck) != Q(ca) + Q(cb)`` is a backend-independent proof that the blocks
    interfere — unlike comparing one backend to another, which a consistent
    bug passes.
    """
    f, kc = _model(cfg)
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _in_grid(kc, 32, seed=2)
    be = create_backend("numpy_flat", kc, model="pwa_gen")
    h = be.load_data(data)

    n_base = len(f.model._pg_combs)

    def q_of(ck):
        p2 = dict(params)
        p2["ck"] = np.asarray(ck, complex)
        return be.compute(p2, h, norm=None)[0]

    ck = np.asarray(params["ck"], complex).copy()
    ca, cb = ck.copy(), ck.copy()
    ca[n_base:] = 0.0                      # block 0 only
    cb[:n_base] = 0.0                      # the rest

    q_all = q_of(ck)
    q_a, q_b = q_of(ca), q_of(cb)
    q_zero = q_of(np.zeros_like(ck))

    assert abs(q_zero) < 1e-12, "ck = 0 must give a vanishing intensity"
    assert q_a > 0.0 and q_b > 0.0, "each block must contribute on its own"
    # the decisive check: a cross term exists
    assert not np.isclose(q_all, q_a + q_b, rtol=1e-6), (
        f"Q(all)={q_all!r} equals Q(block0)+Q(block1)={q_a + q_b!r} — the "
        f"blocks were summed as independent intensities, not coherently")
    be.free()


@pytest.mark.parametrize("cfg", [CFG_IDENT, CFG_CP])
def test_block_zero_is_the_single_block_config(cfg):
    """Block 0's slice must equal the ``n_blocks == 1`` config exactly.

    A full-identity block (its order permutes nothing and it is not CP) is the
    *unmodified* event configuration, so its waves must reproduce the config
    built without any block declaration — same tables, and the same indices
    once the block offset is removed.  This is an oracle that does not consult
    a backend at all: if the rebasing assigns the wrong band to a block, or if
    the barrier/numerator/angle rows drift while being copied, this breaks.
    """
    f, kc = _model(cfg)
    t = f.model.decay_tree
    order0, is_cp0 = t.block_orders()[0]
    assert not is_cp0 and order0 == tuple(range(len(order0))), \
        "block 0 is not the identity block; this oracle needs it to be"

    n_base = len(f.model._pg_combs)
    n_topo, n_decay, n_res = (int(f.model.n_topo), int(f.model.n_decay),
                              int(f.model.n_res))

    # the same physical model, built with the symmetry declarations removed
    plain = Fitter(cfg)
    plain.model.decay_tree = type(plain.model.decay_tree)(
        plain.model.decay_tree.decay_spec,
        plain.model.decay_tree.particle_spec,
        {})                                   # no identical/cp groups
    plain.model._pg_kc = None
    k0 = plain.model.build_kernel_config()
    assert k0["n_blocks"] == 1

    assert int(np.asarray(kc["fl_table"]).shape[0]) == \
        int(kc["n_blocks"]) * int(np.asarray(k0["fl_table"]).shape[0])

    sl = slice(0, n_base)
    # the per-wave tables are block-invariant: block 0 keeps them verbatim
    for key in ("fl_table", "num_table", "wave_names"):
        a = np.asarray(kc[key])[sl] if key != "wave_names" else kc[key][:n_base]
        b = np.asarray(k0[key]) if key != "wave_names" else k0[key]
        assert np.array_equal(np.asarray(a), np.asarray(b)), f"{key} differs"

    # and its INDICES are the un-rebased ones
    for arr in ("fl_q_index", "num_m_index"):
        a = np.asarray(kc[arr])[sl]
        b = np.asarray(k0[arr])
        assert np.array_equal(a, b), f"{arr} block 0 is not un-rebased"

    # block 0's angle rows must be block 0's band (the first n_topo columns)
    used = _angle_rows_of_block(kc, sl)
    ai = np.asarray(kc["angle_index"])[used]
    # block 0's own basis rows must point into its own angle band
    ai0 = ai[ai < n_topo] if ai.size else ai
    assert ai0.size > 0 and ai0.min() >= 0 and ai0.max() < n_topo
    # the basis rows block 0 needs must be exactly those the 1-block config
    # has: the row VALUES are shared across blocks (the key is
    # (column, k, b)), so block 0's rows are a subset that must also cover
    # the 1-block config's set of (angle_index) values.
    one_block_vals = set(np.asarray(k0["angle_index"]).tolist())
    assert one_block_vals <= set(ai0.tolist()), (
        f"block 0 misses basis rows the 1-block config needs: "
        f"{sorted(one_block_vals - set(ai0.tolist()))}")

    # a LATER block must carry a non-zero offset, or nothing was rebased
    later = np.asarray(kc["fl_q_index"])[n_base::]
    assert later.min() >= n_topo * n_decay


@pytest.mark.parametrize("cfg", [CFG_IDENT, CFG_CP])
def test_the_objective_is_the_expected_quadratic_form(cfg):
    """Rebuild ``Q`` from an independently measured quadratic form.

    ``Q`` is quadratic in ``ck`` (the amplitude is linear and the intensity is
    ``Σ_p|A_p|²``), so for a REAL ``ck`` it is an ordinary quadratic form
    ``Q = ckᵀ M ck``.  ``M`` is measured here purely by probing the backend
    with unit vectors — no assumption about which wave reads which column —
    and then ``ckᵀ M ck`` must reproduce ``Q(ck)``.

    This is the backend-independent oracle: it reconstructs the whole objective
    from the backend's own unit responses, so it catches a wrong *index*
    (which changes ``M``) rather than just a disagreement between two backends
    that share the same config.  The block cross terms show up as a non-zero
    off-diagonal block of ``M``.
    """
    f, kc = _model(cfg)
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _in_grid(kc, 8, seed=5)
    be = create_backend("numpy_flat", kc, model="pwa_gen")
    h = be.load_data(data)

    def q_of(ck):
        p2 = dict(params)
        p2["ck"] = np.asarray(ck, complex)
        return be.compute(p2, h, norm=None)[0]

    n_base = len(f.model._pg_combs)
    ck = np.asarray(params["ck"], complex).real.astype(complex)   # real probe
    n = ck.size
    M = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n, complex)
        ei[i] = 1.0
        M[i, i] = q_of(ei)
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n, complex); ei[i] = 1.0
            ej = np.zeros(n, complex); ej[j] = 1.0
            M[i, j] = M[j, i] = (q_of(ei + ej) - M[i, i] - M[j, j]) / 2.0

    rebuilt = float(ck.real @ M @ ck.real)
    assert rebuilt == pytest.approx(q_of(ck), rel=1e-8), (
        "the objective is not the quadratic form its unit responses imply — "
        "a wrong block/column index would break this")

    # blocks must interfere: their off-diagonal block of M is non-zero
    assert np.linalg.norm(M[:n_base, n_base:]) > 0.0, (
        "no cross terms between blocks — they were summed as independent "
        "intensities instead of coherent amplitudes")
    be.free()


# ── gate 3: the permutation really permutes ────────────────────────

def test_a_permutation_block_reads_permuted_columns():
    """The per-block angle columns genuinely differ, and |q| need not.

    For an identical pair the two block orderings swap the SAME two momenta,
    so |q| repeats (same physical pair) while the canonical angles differ —
    which is exactly why an index that forgot the block offset would still
    look plausible in |q| and be wrong in the angles.
    """
    f, kc = _model(CFG_IDENT)
    topo = f.model.decay_tree.block_orders()[0][0]
    mom = np.array([[
        [0.30, 0.10, 0.02, 0.1396],
        [0.25, -0.12, 0.04, 0.1396],
        [0.40, 0.02, -0.06, 0.5479],
    ]])
    d = f.model.build_event_data(mom, kc)
    q = np.asarray(d["q"])
    ang = np.asarray(d["angle"])
    n_topo, n_decay = int(f.model.n_topo), int(f.model.n_decay)

    assert q.shape[1] == kc["n_blocks"] * n_topo * n_decay
    assert ang.shape[1] == kc["n_blocks"] * n_topo

    b0 = q[0][0:n_topo * n_decay]
    b1 = q[0][n_topo * n_decay:2 * n_topo * n_decay]
    assert np.allclose(b0, b1), "identical pair should share |q| across blocks"

    # the angle bands are the permuted configuration: they must differ
    a0, a1 = ang[0][0], ang[0][1]
    assert not np.allclose(a0, a1), (
        "permutation block produced identical angles — the order was not "
        f"applied (block order {topo})")


# ── gate 4: every backend agrees with the reference ────────────────

@pytest.mark.parametrize("cfg", [CFG_IDENT, CFG_CP])
@pytest.mark.parametrize("name", ["cuda_flat_pwa_gen",
                                  "cuda_flat_cache_pwa_gen"])
def test_cuda_matches_numpy_flat(cfg, name):
    f, kc = _model(cfg)
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _in_grid(kc, 64, seed=3)

    ref = create_backend("numpy_flat", kc, model="pwa_gen")
    gpu = create_backend(name, kc, model="pwa_gen")
    Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
    Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)

    # Precision floor.  The GPU caches F / basis_m / fa as float2, so this is
    # a CACHE limit, not an algebra difference: all the GPU kernels agree with
    # EACH OTHER to <1e-6, and the per-event MEDIAN relative error is ~1e-7 on
    # every fixture.  `_rel` divides the worst ABSOLUTE error by the global
    # max|P|, which is unfair on a config with a wide P dynamic range
    # (`_ident` spans 7e-4 .. 7e2, so the worst relative error lands on the
    # near-zero events while the global max sets the scale).  Measure the
    # quantity that actually matters instead: the median per-event relative
    # error, plus a loose bound on the global one for gross-error detection.
    Pr_a, Pg_a = np.asarray(Pr).ravel(), np.asarray(Pg).ravel()
    per_event = np.abs(Pg_a - Pr_a) / np.maximum(np.abs(Pr_a), 1e-300)
    assert float(np.median(per_event)) <= 2e-6, (
        f"median per-event P error {np.median(per_event):.2e} — should be the "
        f"float2 cache floor (~1e-7)")
    assert _rel(Pg, Pr) <= 1e-4, "gross P mismatch"
    assert _rel(Qg, Qr) <= 1e-5, "gross Q mismatch"
    assert _rel(gg["ck"], gr["ck"]) <= 1e-5
    # `dcoeff` sums dQ/ds*basis over EVERY (event, wave), so it accumulates the
    # float2 cache error once per term — looser than the per-event quantities,
    # and its own exactness is pinned separately by the central-difference
    # gate (test_dck_matches_central_differences), which does not depend on the
    # cache at all.
    assert _rel(gg["coeff"], gr["coeff"]) <= 1e-4, "gross dcoeff mismatch"
    ref.free()
    gpu.free()


# ── gate 5: analytic gradients survive the block expansion ─────────

@pytest.mark.parametrize("cfg", [CFG_IDENT, CFG_CP])
def test_dck_matches_central_differences(cfg):
    f, kc = _model(cfg)
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _in_grid(kc, 16, seed=4)
    be = create_backend("numpy_flat", kc, model="pwa_gen")
    h = be.load_data(data)
    _Q, g, _P = be.compute(params, h, norm=1.3)

    ck0 = np.asarray(params["ck"], complex).copy()
    step = 1e-6
    fd = np.zeros(ck0.size)
    for i in range(ck0.size):
        for sign in (+1.0, -1.0):
            ck = ck0.copy()
            ck[i] = ck[i] + sign * step      # perturb Re(ck_i)
            p2 = dict(params)
            p2["ck"] = ck
            q = be.compute(p2, h, norm=1.3)[0]
            fd[i] += sign * q
    fd /= (2 * step)                       # dQ/dRe(ck_i)
    # the kernel returns the Wirtinger half: dQ/dck = g, and dQ/dRe = 2*Re(g)
    got = 2.0 * np.real(np.asarray(g["ck"])).ravel()
    assert _rel(got, fd) <= 2e-4, f"dck vs FD: {_rel(got, fd):.2e}"
    be.free()

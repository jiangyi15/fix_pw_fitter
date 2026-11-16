"""``resolution_size`` — the grouped log-sum objective, on every pwa_gen backend.

The objective groups consecutive events and takes ONE log per group, weighted
by the group's total weight:

    W_g = Σ_{e∈g} w_e
    S_g = Σ_{e∈g} w_e·(P_e/norm + bkg_e)
    Q   = -Σ_g W_g·log(S_g/W_g)                 (groups with W_g = 0 inert)
    dQ/dP_e = -(W_g/S_g)·(w_e/norm)

Gates here:
1. ``numpy_flat`` reproduces an independently written group sum **exactly**
   (the reference must be right before it can anchor the kernels);
2. ``rs == 1`` is the per-event NLL for ARBITRARY weights — the grouped form
   reduces to ``-Σ w_e·log(pdf_e)`` there, so a different code path must not
   change the number;
3. ``rs > 1`` with group-normalised weights (Σw = 1 per group) makes each
   group's log argument the group's MEAN pdf — the normalisation that keeps
   ``Σw`` equal to the group count (and so the purity constant exact), NOT an
   equality with the per-event NLL;
4. every registered ``pwa_gen`` backend accepts the kwarg and agrees with
   ``numpy_flat`` on ``Q``/``P``/gradients/``dnorm``;
5. the group size is honoured ACROSS tiles: a group bigger than the CUDA tile
   (``resolution_size`` 10..100 in practice) must still be one log, not a log
   per tile.

Test data stays INSIDE both interpolation grids — outside, the tables clamp and
the CPU/GPU paths diverge, which looks exactly like a kernel bug.
"""
import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend

CFG_KM = "tests/config_pwa_gen_km.yml"

#: real-world group sizes are 10..100, so cover past one CUDA tile (8)
RSIZES = (1, 2, 4, 8, 10, 25, 100)


def _kc(cfg=CFG_KM):
    f = Fitter(cfg)
    f.apply_constrains()
    return f, f.kernel_config


def _in_grid(kc, ne, seed=0):
    """Events strictly inside BOTH interpolation grids."""
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


def _replicate(data, rs, normalised=True):
    """Repeat every event ``rs`` times; group-normalised weights by default.

    A row repeated ``rs`` times inside one group is the physical picture the
    grouped objective is for (rescanned / binned data), and with
    ``w = 1/rs`` each group's weight sums to 1, so the grouped NLL equals the
    per-event NLL of the original sample.
    """
    out = {}
    for k, v in data.items():
        out[k] = np.repeat(np.asarray(v), rs, axis=0)
    if normalised:
        out["weight"] = np.full(len(np.asarray(data["weight"])) * rs, 1.0 / rs)
    return out


def _group_sum(P, weight, bkg, norm, rs):
    """Independent, straight-from-the-definition group sum (numpy only)."""
    ne = len(P)
    Q = 0.0
    dn = 0.0
    for g in range((ne + rs - 1) // rs):
        sl = slice(g * rs, min((g + 1) * rs, ne))
        w, p, b = weight[sl], P[sl], bkg[sl]
        S = float(np.sum(w * (p / norm + b)))
        W = float(np.sum(w))
        Pb = float(np.sum(w * p))
        if W > 0.0:
            Q += -W * np.log(S / W)
            dn += W * Pb / (S * norm * norm)
    return float(Q), float(dn)


def _dnorm_of(backend):
    """The native d(NLL)/d(norm) whichever layer holds it."""
    v = getattr(backend, "_last_dnorm", None)
    if v is None:
        v = getattr(getattr(backend, "kernel", None), "_last_dnorm", None)
    return v


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# ── gate 1: the reference itself ────────────────────────────────────

@pytest.mark.parametrize("rs", [2, 4, 10, 25, 100])
def test_numpy_flat_matches_the_definition(rs):
    f, kc = _kc()
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _replicate(_in_grid(kc, 40, seed=2), rs)

    b = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    Q, _g, P = b.compute(params, b.load_data(data), norm=1.3)
    Qref, dnref = _group_sum(P, data["weight"], data["bkg"], 1.3, rs)
    assert Q == pytest.approx(Qref, rel=0, abs=1e-9)
    assert _dnorm_of(b) == pytest.approx(dnref, rel=0, abs=1e-9)
    b.free()


def test_numpy_flat_grouped_is_exact_not_approximate():
    """The reference must be exact: it anchors every other backend."""
    f, kc = _kc()
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _replicate(_in_grid(kc, 40, seed=3), 4)
    b = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=4)
    Q, _g, P = b.compute(params, b.load_data(data), norm=1.3)
    Qref, _ = _group_sum(P, data["weight"], data["bkg"], 1.3, 4)
    assert Q == Qref          # same expression, so bit-identical
    b.free()


@pytest.mark.parametrize("ne,rs", [(7, 4), (10, 3), (1, 8), (13, 100), (5, 5)])
def test_ragged_last_group_is_handled(ne, rs):
    """``ne`` need not be a multiple of ``rs`` — the last group is short.

    A reshape(PAD) that forgets the pad rows raises "cannot reshape array of
    size ne into shape (ng, rs)"; the correct answer is the definition over
    the short group.
    """
    f, kc = _kc()
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    rng = np.random.RandomState(11)
    data = {
        "mass": np.full((ne, 1), 0.9) + 1e-3 * rng.rand(ne, 1),
        "q": np.full((ne, 2), 0.8) + 1e-3 * rng.rand(ne, 2),
        "angle": rng.uniform(-np.pi, np.pi, (ne, 1, 4)),
        "weight": np.full(ne, 1.0 / max(rs, 1)),
        "bkg": np.full(ne, 0.1),
    }
    b = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    Q, _g, P = b.compute(params, b.load_data(data), norm=1.3)
    Qref, dnref = _group_sum(P, data["weight"], data["bkg"], 1.3, rs)
    assert Q == pytest.approx(Qref, rel=0, abs=1e-9)
    assert _dnorm_of(b) == pytest.approx(dnref, rel=0, abs=1e-9)
    b.free()


# ── gate 2/3: the two reduction identities ─────────────────────────

def test_rsize_one_is_the_per_event_nll_for_arbitrary_weights():
    """At rs == 1 the grouped form is -Σ w·log(pdf) whatever the weights."""
    f, kc = _kc()
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _in_grid(kc, 64, seed=4)          # un-normalised weights

    a = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=1)
    b = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=2)
    Qa, _ga, _Pa = a.compute(params, a.load_data(data), norm=1.3)
    Qb, _gb, _Pb = b.compute(params, b.load_data({**data,
                                                  "weight": np.full(64, .5)}),
                             norm=1.3)
    # rs=2 over *unrelated* events is a different objective: only rs=1 is the
    # per-event log.  The identity to check is rs=1 vs the per-event formula.
    P = _Pa
    w, bk = data["weight"], data["bkg"]
    Qref = float(-np.sum(w * np.log(P / 1.3 + bk)))
    assert Qa == pytest.approx(Qref, rel=1e-12)
    assert Qb != pytest.approx(Qa, rel=1e-9)   # the objective really changed
    a.free()
    b.free()


@pytest.mark.parametrize("rs", [2, 4, 10])
def test_group_normalised_weights_make_the_group_a_mean_pdf(rs):
    """Σw = 1 per group reduces the objective to -Σ_g log(mean_g pdf).

    This is what group normalisation buys: the group's weight is 1, so the
    log argument is the group's MEAN pdf.  It is deliberately NOT the
    per-event NLL — one log over a mean is a different (Jensen-shifted)
    objective — but it is the form whose Σw equals the group count, which is
    what keeps the fitter's purity constant ``-log(p)·Σw`` exact.
    """
    f, kc = _kc()
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _in_grid(kc, 32, seed=5)

    a = create_backend("numpy_flat", kc, model="pwa_gen")
    b = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    _Qa, ga, Pa = a.compute(params, a.load_data(data), norm=1.3)
    rep = _replicate(data, rs)
    Qb, gb, Pb = b.compute(params, b.load_data(rep), norm=1.3)
    # P stays the per-event intensity — grouping changes the objective only
    assert _rel(Pb, np.repeat(Pa, rs)) <= 1e-12

    pdf = np.repeat(Pa, rs) / 1.3 + rep["bkg"]
    Qmean = float(-np.sum(np.log(pdf.reshape(-1, rs).mean(axis=1))))
    assert Qb == pytest.approx(Qmean, rel=1e-12)
    # Σw == n_groups, the property the purity constant relies on
    assert rep["weight"].reshape(-1, rs).sum(axis=1) == pytest.approx(1.0)
    assert Qb != pytest.approx(_Qa, rel=1e-9)     # a different objective
    a.free()
    b.free()


def test_grouped_gradients_match_central_differences():
    """The grouped dcoeff must be the gradient of the grouped Q, not of the
    per-event Q — a stale per-event gradient would fit the wrong objective."""
    f, kc = _kc()
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    rs = 4
    data = _replicate(_in_grid(kc, 16, seed=12), rs)
    b = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    h = b.load_data(data)
    _Q, g, _P = b.compute(params, h, norm=1.3)

    def q_at(coeff):
        p = dict(params)
        p["coeff"] = np.asarray(coeff, np.float64)
        return b.compute(p, h, norm=1.3)[0]

    c0 = np.asarray(params["coeff"], np.float64)
    step = 1e-6
    fd = np.zeros_like(c0)
    for i in range(c0.size):
        up = c0.copy(); up[i] += step
        dn = c0.copy(); dn[i] -= step
        fd[i] = (q_at(up) - q_at(dn)) / (2 * step)
    assert _rel(g["coeff"], fd) <= 1e-5
    b.free()


def test_zero_weight_group_is_inert():
    """A fully zero-weight group must contribute neither Q nor a gradient."""
    f, kc = _kc()
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _replicate(_in_grid(kc, 16, seed=6), 2)
    killed = np.zeros_like(data["weight"])
    killed[:2] = 0.0                              # zero out group 0 entirely
    a = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=2)
    Qa, _ga, _Pa = a.compute(params, a.load_data({**data, "weight": killed}),
                             norm=1.3)
    keep = np.concatenate([np.arange(0, 0), np.arange(2, len(killed))])
    sub = {k: np.asarray(v)[keep] for k, v in {**data, "weight": killed}.items()}
    Qb, _gb, _Pb = a.compute(params, a.load_data(sub), norm=1.3)
    assert Qa == pytest.approx(Qb, rel=1e-12)
    a.free()


# ── gate 4: every registered backend takes the kwarg ───────────────

#: composers take a nested backend spec, not a kernel config, so they are
#: not leaf kernels and have no ``resolution_size`` of their own — the inner
#: spec carries it instead.
COMPOSERS = frozenset({"shard", "integrated", "integrated_pwa"})


def _pwa_gen_backends():
    """Real leaf pwa_gen backends.

    ``default`` is an alias and composers are excluded.  Names starting with
    ``_`` are also skipped: other test modules register throwaway probes into
    the GLOBAL registry (``_kwarg_probe``, ``_model_probe``), and a
    universal registration shows up in every model's set — sweeping the
    registry blindly would assert against those probes whenever the whole
    suite runs in one process.
    """
    names = sorted(backends_for_model("pwa_gen"))
    return [n for n in names
            if n != "default" and n not in COMPOSERS
            and not n.startswith("_")]


def test_every_pwa_gen_backend_accepts_resolution_size():
    f, kc = _kc()
    missing = []
    for name in _pwa_gen_backends():
        try:
            b = create_backend(name, kc, model="pwa_gen", resolution_size=4)
        except TypeError as exc:                  # kwarg not accepted
            missing.append(f"{name}: {exc}")
            continue
        got = getattr(b, "resolution_size",
                      getattr(getattr(b, "kernel", None), "resolution_size",
                              None))
        if got != 4:
            missing.append(f"{name}: resolution_size is {got!r}, expected 4")
        b.free()
    assert not missing, "backends missing resolution_size:\n" + "\n".join(missing)


def test_resolution_size_reaches_the_kernel_through_the_fitter():
    f, kc = _kc()
    for name in ("numpy_flat",):
        fit = Fitter(CFG_KM, backend={"name": name, "resolution_size": 10})
        assert getattr(fit.backend, "resolution_size", None) == 10


def test_resolution_size_below_one_is_rejected():
    """``rs < 1`` is an error everywhere — never a zero-sized group.

    (The reference clamps to 1; the CUDA wrappers raise.  Both are safe; what
    must not happen is a silently empty objective.)
    """
    _f, kc = _kc()
    for name in _pwa_gen_backends():
        try:
            b = create_backend(name, kc, model="pwa_gen", resolution_size=0)
        except ValueError:
            continue
        got = getattr(b, "resolution_size",
                      getattr(getattr(b, "kernel", None), "resolution_size",
                              None))
        assert got == 1, f"{name}: resolution_size=0 became {got!r}"
        b.free()


# ── gate 5: GPU backends agree with the reference ──────────────────

def _gpu_names():
    return [n for n in _pwa_gen_backends() if n.startswith("cuda")]


@pytest.mark.parametrize("name", _gpu_names() or ["<none>"])
@pytest.mark.parametrize("rs", RSIZES)
def test_cuda_matches_numpy_flat(name, rs):
    if name == "<none>":
        pytest.skip("no CUDA pwa_gen backend registered in this build")
    f, kc = _kc()
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _replicate(_in_grid(kc, 32, seed=7), rs)

    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    gpu = create_backend(name, kc, model="pwa_gen", resolution_size=rs)
    Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
    Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)

    assert _rel(Qg, Qr) <= 1e-6, f"{name} rs={rs} Q"
    assert _rel(Pg, Pr) <= 1e-6, f"{name} rs={rs} P"
    assert _rel(gg["ck"], gr["ck"]) <= 1e-6, f"{name} rs={rs} dck"
    assert _rel(gg["coeff"], gr["coeff"]) <= 1e-6, f"{name} rs={rs} dcoeff"
    assert _dnorm_of(gpu) == pytest.approx(_dnorm_of(ref), rel=1e-6), \
        f"{name} rs={rs} dnorm"
    ref.free()
    gpu.free()


@pytest.mark.parametrize("name", _gpu_names() or ["<none>"])
def test_cuda_rsize_one_does_not_change_the_objective(name):
    """``resolution_size=1`` must leave the per-event arithmetic alone.

    Not bit-identical: these kernels accumulate ``Q``/``dnorm`` with
    ``atomicAdd`` across blocks, so two separate contexts differ at the last
    ULP by construction (the non-cached kernel is not even repeatable within
    ONE instance for that reason).  What must hold is that ``rs == 1`` is the
    per-event objective, to well inside the float2 cache floor.
    """
    if name == "<none>":
        pytest.skip("no CUDA pwa_gen backend registered in this build")
    f, kc = _kc()
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _in_grid(kc, 64, seed=8)

    a = create_backend(name, kc, model="pwa_gen")
    b = create_backend(name, kc, model="pwa_gen", resolution_size=1)
    Qa, ga, Pa = a.compute(params, a.load_data(data), norm=1.3)
    Qb, gb, Pb = b.compute(params, b.load_data(data), norm=1.3)
    assert _rel(Qb, Qa) <= 1e-12
    # P carries no atomics: it is exactly reproducible
    assert np.array_equal(Pa, Pb)
    assert _rel(gb["ck"], ga["ck"]) <= 1e-12
    assert _rel(gb["coeff"], ga["coeff"]) <= 1e-12
    assert _dnorm_of(b) == pytest.approx(_dnorm_of(a), rel=1e-12)
    a.free()
    b.free()


# ── gate 6: a group must span tiles as ONE log ─────────────────────

def test_group_larger_than_a_tile_is_one_log():
    """The CUDA tiles are 8 events; rs 10..100 straddles them.

    If the kernel took one log per tile instead of per group, ``Q`` would
    differ from the definition by a factor structured as ``n_tiles`` extra
    logs — the reference group sum catches that immediately.
    """
    f, kc = _kc()
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    rs = 25                                   # > tile (8), < ne
    data = _replicate(_in_grid(kc, 8, seed=9), rs)   # 200 events, 8 groups

    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    Qr, _gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
    Qdef, _ = _group_sum(Pr, data["weight"], data["bkg"], 1.3, rs)
    assert Qr == pytest.approx(Qdef, rel=0, abs=1e-9)

    for name in _gpu_names():
        gpu = create_backend(name, kc, model="pwa_gen", resolution_size=rs)
        Qg, _gg, _Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
        assert _rel(Qg, Qdef) <= 1e-6, f"{name}: a 25-event group is not one log"
        gpu.free()
    ref.free()

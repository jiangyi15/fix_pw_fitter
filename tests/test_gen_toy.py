"""``Fitter.generate_toy`` — the one-call toy entry point, every model shape.

Four models, one signature: ``pwa``, the basis model ``pwa_gen``, the
``polarised_pwa`` rho-matrix model, and ``flavour_tag_mix`` — the last one
supplies its own generator through ``AmplitudeModel.toy_generator()`` because it
needs the ``frac``/``time`` columns.

The toy returns ``data`` (``(data, stats)`` with ``stats=True``): the buffer
:meth:`Fitter.set_data` takes, and the truth, the acceptance ceiling and the
counts.  NO norm sample is produced — generation needs only ``P`` and its
ceiling — so a toy study is::

    x0 = f.initial_values(seed=1)
    data = f.generate_toy(x0, 4000, seed=7)          # bkg = ones (uniform)
    f.set_phsp(f.generate_phsp(40_000, seed=7))      # the norm: SEPARATE
    f.set_data(data)

The ``bkg`` column is the background SHAPE, not its amount: the toy takes it from
``fbg`` and the norm from ``bkg`` (both uniform by default, both evaluated by
``toy_pwa.bkg_values``), and ``set_data`` applies ``(1-p)/p/N_b`` from the config
purity.  So a config with purity < 1 must be given the SAME function on both
calls, or the fit is mis-normalised.
"""

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.phasespace import fix_beam

CFG_PWA = "tests/config_pwa.yml"
CFG_GEN = "tests/config_pwa_gen.yml"
CFG_TD = "tests/config_amp.yml"


def _fitter(path, backend):
    f = Fitter(path, backend=backend)
    f.apply_constrains()
    return f


def test_generate_toy_returns_settable_buffers():
    """``pwa``: the buffers go straight into set_phsp/set_data and the NLL."""
    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=1)
    data = f.generate_toy(x0, 300, seed=5, batch_size=2000)
    phsp = f.generate_phsp(200, seed=5)

    assert len(data["mass"]) == 300 and len(phsp["mass"]) == 200
    for key in ("mass", "q", "angle", "weight"):
        assert key in data and key in phsp
    # a toy is distributed like |A|^2, so the weight column is 1 (unweighted)
    assert np.allclose(data["weight"], 1.0)

    f.set_phsp(phsp)
    f.set_data(data)
    nll, grad = f.get_nll(x0)
    assert np.isfinite(nll) and np.isfinite(np.asarray(grad)).all()


def test_the_toy_bkg_column_comes_from_fbg():
    """``bkg = fbg(events)`` — UNIFORM (ones) by default, the norm's shape.

    The column is the background SHAPE, the same argument ``generate_phsp`` takes
    as ``bkg``; the AMOUNT comes from the config purity in ``set_data``.  A strict
    signal-only column is ``fbg=zeros``, and such a sample is purity-invariant
    apart from ``-log(p)·Σw``.
    """
    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=1)
    data, _ = f.generate_toy(x0, 200, seed=5, stats=True)
    assert np.all(data["bkg"] == 1.0)                    # uniform default

    def shape(x):
        return 1.0 + 0.5 * np.cos(x["mass"][:, 0])

    graded = f.generate_toy(x0, 200, seed=5, fbg=shape)
    assert np.allclose(graded["bkg"], shape(graded))
    phsp = f.generate_phsp(150, seed=11, bkg=shape)      # the same function
    assert np.allclose(phsp["bkg"], shape(phsp))

    sig = f.generate_toy(x0, 200, seed=5,
                         fbg=lambda x: np.zeros(len(x["mass"])))
    assert np.all(sig["bkg"] == 0.0)

    def nll_at(purity):
        g = _fitter(CFG_PWA, "numpy_pwa")
        g.config.dic["data"]["purity"] = purity
        g.set_phsp(g.generate_phsp(300, seed=11))
        g.set_data(sig)
        return g.get_nll(x0)

    q1, g1 = nll_at(1.0)
    q9, g9 = nll_at(0.9)
    n = len(sig["mass"])
    assert abs((q9 - q1) + np.log(0.9) * n) < 1e-8 * max(1.0, abs(q1))
    assert np.allclose(g9, g1, atol=1e-9)


def test_generate_toy_bg_splits_the_counts_poisson():
    """``n_sig ~ Poisson(n·p)`` and ``n_bg ~ Poisson(n·(1-p))``, over many seeds.

    One draw at 4 sigma would pass a 5 % mean error most of the time, so this
    checks the MEAN and the scatters of 20 independent runs.
    """
    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=1)
    n, p, seeds = 400, 0.3, range(20)
    sig, bgc = [], []
    for sd in seeds:
        d, st = f.generate_toy_bg(x0, n, purity=p, seed=100 + sd, stats=True)
        assert st.n_signal + st.n_background == len(d["mass"])
        assert np.allclose(d["bkg"], 1.0)     # additive column, uniform b
        sig.append(st.n_signal)
        bgc.append(st.n_background)
    sig, bgc = np.asarray(sig, float), np.asarray(bgc, float)
    se_sig = np.sqrt(n * p / len(seeds))
    se_bg = np.sqrt(n * (1 - p) / len(seeds))
    assert abs(sig.mean() - n * p) < 4 * se_sig, (sig.mean(), n * p)
    assert abs(bgc.mean() - n * (1 - p)) < 4 * se_bg, (bgc.mean(), n * (1 - p))
    assert sig.std() > 0.5 * np.sqrt(n * p)    # a real Poisson scatter, not fixed


def test_generate_toy_bg_takes_the_purity_from_data_purity():
    """Production configs nest it: ``data.purity`` / ``data.bg_frac``.

    ``set_data`` reads top-level first, then ``data.purity``/``data.bg_frac``; the
    toy must use the SAME reader, or it is generated at one purity and fitted at
    another.
    """
    from tabpwa.config_loader import RawConfig

    for key in ("purity", "bg_frac"):
        dic = RawConfig(CFG_PWA).to_dict()
        dic.setdefault("data", {})[key] = 0.25
        f = Fitter(dic, backend="numpy_pwa")
        f.apply_constrains()
        assert f.config_purity() == 0.25
        x0 = f.initial_values(seed=1)
        _d, st = f.generate_toy_bg(x0, 400, seed=5, stats=True)
        assert abs(st.n_signal - 0.25 * 400) < 4 * np.sqrt(0.25 * 400), (key, st)


def test_generate_toy_bg_works_for_the_time_dependent_model():
    """TD: the signal comes from the model's own generator, the background from
    its ``build_bkg_event_data`` (flat mixing/time), and the columns survive.

    ``frac`` ∈ {0, 1} and ``time`` ~ U(0, 10) on BOTH sides, because a background
    that is flat in the mixing variables has exactly the signal's flat law.
    """
    f = _fitter(CFG_TD, "numpy")
    x0 = f.initial_values(seed=4)
    data, stats = f.generate_toy_bg(x0, 20, purity=0.6, seed=3, oversample=1,
                                    stats=True)
    assert {"mass", "q", "angle", "frac", "time", "weight", "bkg"} <= set(data)
    assert set(np.unique(data["frac"])) <= {0.0, 1.0}   # declared: choice(0, 1)
    assert data["time"].min() >= 0.0 and data["time"].max() <= 10.0
    assert np.allclose(data["bkg"], 1.0)
    assert stats.n_signal + stats.n_background == len(data["mass"])

    # the norm sample carries the background SHAPE (uniform), so N_b = 1 and the
    # fitter accepts the pair
    phsp = f.generate_phsp(20, seed=9)
    assert abs(phsp["bkg"].mean() - 1.0) < 1e-12
    f.set_phsp(phsp)
    f.set_data(data)
    assert np.isfinite(f.get_nll(x0)[0])

    # top_p/maxp stay refused: this model owns its phase space
    with pytest.raises(ValueError, match="do not apply"):
        # a SEED: an unseeded Poisson(2.5) draw is 0 about 8 % of the time,
        # and with no signal events the model's generator is never called
        # (so nothing can refuse `top_p`) — the test would flake
        f.generate_toy_bg(x0, 5, purity=0.5, seed=3,
                          top_p=fix_beam(0.034))


def test_generate_toy_bg_survives_a_narrow_support():
    """A batch that MISSES a narrow support must not be read as "b == 0"."""
    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=1)

    def narrow(ev):
        m = ev["mass"][:, 1]
        return ((m > 1.90) & (m < 1.97)).astype(float)

    data, st = f.generate_toy_bg(x0, 40, purity=0.0, bg=narrow, seed=3,
                                 stats=True)
    # n_background is the POISSON DRAW (around 40), and the data matches it
    assert st.n_background == len(data["mass"]) > 0 and st.bg_tried > 0
    m = data["mass"][:, 1]
    assert np.all((m > 1.90) & (m < 1.97))

    def zero(ev):
        return np.zeros(len(ev["mass"]))

    # the refusal is the claim; its cost is the engine's (a 1e6-event all-zero
    # budget, MEASURED 32.7 s before the adaptive batch and 4.4 s after — NOT
    # `stall_loops`, which measures nothing here)
    with pytest.raises(RuntimeError, match="zero over"):
        f.generate_toy_bg(x0, 10, purity=0.0, bg=zero, seed=1)


def test_generate_toy_bg_custom_background_is_sampled_and_normalised():
    """A cut background: no event outside its support, and the phsp carries it.

    ``N_b`` is the phase-space MEAN of the same column, so a non-uniform ``b``
    normalises without any manual factor: the phsp mean must equal the support
    fraction of the phase space.
    """
    from scipy.stats import ks_2samp

    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=1)

    def halfbg(ev):
        return (ev["mass"][:, 1] > 1.6).astype(float)

    data, st = f.generate_toy_bg(x0, 600, purity=0.0, bg=halfbg, seed=4,
                                 stats=True)
    phsp = f.generate_phsp(400, seed=11, bkg=halfbg)   # the SAME column
    assert st.n_signal == 0 and st.purity == 0.0
    assert np.all(data["mass"][:, 1] > 1.6)          # only inside the support
    assert np.allclose(data["bkg"], 1.0)
    ref = f.generate_phsp(4000, seed=9)
    assert abs(phsp["bkg"].mean() - np.mean(ref["mass"][:, 1] > 1.6)) < 0.05

    f.set_phsp(phsp)
    f.set_data(data)
    assert np.isfinite(f.get_nll(x0)[0])

    # a UNIFORM background is flat in the measure
    flat, _st = f.generate_toy_bg(x0, 2000, purity=0.0, seed=5, stats=True)
    plain = f.generate_phsp(2000, seed=5)
    assert ks_2samp(flat["mass"][:, 1], plain["mass"][:, 1]).pvalue > 1e-3


def test_generate_toy_bg_custom_background_follows_its_density():
    """Checked against a WEIGHTED reference (draw with probabilities ∝ b).

    A different method from the sampler's accept-reject, so this is not
    self-confirming; and ``purity=1`` with a ``bg`` given must simply produce no
    background events.
    """
    from scipy.stats import ks_2samp

    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=1)

    def peak(ev):
        m = ev["mass"][:, 1]
        return np.exp(-0.5 * ((m - 1.9) / 0.15) ** 2) + 0.05

    data, st = f.generate_toy_bg(x0, 2000, purity=0.0, bg=peak, seed=7,
                                 stats=True)
    assert st.n_background == len(data["mass"]) and st.n_signal == 0
    ref = f.generate_phsp(60_000, seed=8)
    v = peak(ref)
    rng = np.random.RandomState(1)
    idx = rng.choice(len(v), size=st.n_background, p=v / v.sum())
    assert ks_2samp(data["mass"][:, 1], ref["mass"][idx, 1]).pvalue > 1e-3
    assert not np.allclose(data["bkg"], 1.0)      # the column IS b(x) here

    pure, pst = f.generate_toy_bg(x0, 200, purity=1.0, bg=peak, seed=9,
                                  stats=True)
    assert pst.n_background == 0 and len(pure["mass"]) == pst.n_signal


def test_generate_toy_bg_purity_defaults_to_the_config():
    """The toy uses the SAME purity ``set_data`` will read from the config."""
    from tabpwa.config_loader import RawConfig

    dic = RawConfig(CFG_PWA).to_dict()
    dic["purity"] = 0.8
    f = Fitter(dic, backend="numpy_pwa")
    f.apply_constrains()
    x0 = f.initial_values(seed=1)
    _d, st = f.generate_toy_bg(x0, 1200, seed=6, stats=True)
    assert abs(st.n_signal - 0.8 * 1200) < 4 * np.sqrt(0.8 * 1200)


def test_generate_refusals():
    """The documented refusals, each with its own message."""
    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=1)
    with pytest.raises(ValueError, match="n_events must be >= 1"):
        f.generate_toy(x0, 0)
    with pytest.raises(ValueError, match="n_data must be >= 1"):
        f.generate_toy_bg(x0, 0)
    with pytest.raises(ValueError, match="maxp"):
        f.generate_toy(x0, 10, maxp=float("nan"))
    with pytest.raises(ValueError, match="momenta"):
        f.generate_phsp(10, bkg=lambda ev: ev["mass"][:, 1], momenta=True)


def test_stats_is_a_flag_on_both_generators():
    """Default: the sample alone (``set_*`` takes it directly); ``stats=True``: the pair."""
    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=1)
    data = f.generate_toy(x0, 50, seed=1, batch_size=1000)
    assert isinstance(data, dict) and "mass" in data
    pair = f.generate_toy(x0, 50, seed=1, batch_size=1000, stats=True)
    assert isinstance(pair, tuple) and pair[0] is not None and pair[1].maxp > 0
    phsp = f.generate_phsp(50, seed=2)
    assert isinstance(phsp, dict) and "mass" in phsp
    phsp2, pst = f.generate_phsp(50, seed=2, stats=True)
    assert pst.n_events == 50 and pst.n_finals == len(f.model.finals)


def test_generate_toy_works_for_the_basis_model():
    """``pwa_gen``: same call, the model's own event builder."""
    f = _fitter(CFG_GEN, "numpy_flat")
    x0 = f.initial_values(seed=2)
    data = f.generate_toy(x0, 200, seed=6, batch_size=2000)
    phsp = f.generate_phsp(100, seed=6)
    assert len(data["mass"]) == 200 and len(phsp["mass"]) == 100
    f.set_phsp(phsp)
    f.set_data(data)
    assert np.isfinite(f.get_nll(x0)[0])


def test_generate_phsp_is_settable_and_model_general():
    """``f.generate_phsp(n)`` = ``generate_pwa_phsp`` + the model's builder.

    It is the norm setup, and it is topology/model general: the flat sample
    depends only on the final masses, and the rows come from the model's own
    ``build_event_data``.
    """
    from tabpwa.pwa_build import generate_pwa_phsp

    f = _fitter(CFG_PWA, "numpy_pwa")
    mom = f.generate_phsp(300, seed=4, momenta=True)
    chain = f.decay_tree.partial_waves()[0][1]
    assert np.array_equal(mom, generate_pwa_phsp(f.model, chain, 300, seed=4))
    buf = f.generate_phsp(300, seed=4)
    assert np.array_equal(
        buf["mass"], f.model.build_event_data(mom, f.kernel_config)["mass"])
    f.set_phsp(buf)                       # the norm accepts it as-is

    g = _fitter(CFG_GEN, "numpy_flat")    # a different model shape
    assert len(g.generate_phsp(50, seed=1)["mass"]) == 50


def test_the_adaptive_ceiling_is_unbiased():
    """A tight-but-valid ceiling must not reshape the toy.

    The ceiling is ``maxp_scale * max(batch P)``, raised when a later batch
    exceeds it, and the accepted pool is thinned by ``old/new`` — exact because
    the raise happens before its own batch is accepted, so every accepted event
    has ``P <= ceiling``.  MEASURED on a deliberately tight ceiling (4 raises):
    top-decile share 0.100 vs 0.104 and KS p = 0.52 against a generous one.
    ``maxp_scale < 1`` is refused: ``new = scale*max`` could never exceed the
    maximum, so the peak would stay capped however the pool is thinned.
    """
    from scipy.stats import ks_2samp

    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=1)
    tight = f.generate_toy(x0, 800, seed=5, batch_size=400,
                                maxp_scale=1.0)
    loose = f.generate_toy(x0, 800, seed=5, batch_size=4000,
                                maxp_scale=2.0)
    assert ks_2samp(tight["mass"][:, 0], loose["mass"][:, 0]).pvalue > 1e-3
    with pytest.raises(ValueError, match="maxp_scale"):
        f.generate_toy(x0, 100, seed=1, maxp_scale=0.5)


def test_the_ceiling_is_returned_and_can_be_reused():
    """``maxp`` goes out and back in: the next run needs no estimate.

    The ceiling belongs to (parameter point, topology), so feeding it back gives
    the same value at a tiny fraction of the work; an under-estimate is still
    raised, so a stale value costs time and never correctness.
    """
    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=1)
    _d1, s1 = f.generate_toy(x0, 200, seed=1, batch_size=2000, stats=True)
    assert s1.maxp > 0 and s1.n_events == 200 and 0 < s1.acceptance <= 1
    assert np.allclose(s1.truth, x0)          # the truth comes back with it
    _d2, s2 = f.generate_toy(x0, 200, seed=2, batch_size=2000,
                             maxp=s1.maxp, stats=True)
    assert s2.maxp == s1.maxp                 # nothing needed raising
    _d3, s3 = f.generate_toy(x0, 200, seed=3, batch_size=2000,
                             maxp=s1.maxp / 10.0, stats=True)
    assert s3.maxp > s1.maxp / 10.0           # an under-estimate is raised
    assert abs(s3.maxp - s1.maxp) < 0.5 * s1.maxp


def test_the_runaway_guard_is_a_no_progress_window():
    """A dead pool fails in BOUNDED BATCHES; a progressing one is never capped.

    ``stall_loops`` counts CONSECUTIVE batches with no acceptance, so a peaked
    model that keeps accepting runs as long as it needs — the old
    ``max_loops=1000`` total budget (20 M candidates) cut it off.  A ceiling far
    above the true maximum accepts nothing, so the guard must fire; with
    ``stall_loops=0`` the guard is off and an explicit ``max_loops`` is the cap.
    """
    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=1)
    with pytest.raises(RuntimeError, match="no event accepted for 5 consecutive"):
        f.generate_toy(x0, 10, seed=1, batch_size=50, stall_loops=5, maxp=1e300)
    with pytest.raises(RuntimeError, match="accepted only 0 of 10 events"):
        f.generate_toy(x0, 10, seed=1, batch_size=50, max_loops=3, maxp=1e300)
    data = f.generate_toy(x0, 200, seed=1, batch_size=100, maxp_scale=1.0)
    assert len(data["mass"]) == 200
    with pytest.raises(ValueError, match="stall_loops"):
        f.generate_toy(x0, 5, seed=1, stall_loops=-1)
    with pytest.raises(ValueError, match="batch_size"):
        f.generate_toy(x0, 5, seed=1, batch_size=0)


def test_a_cached_run_resumes_exactly(tmp_path):
    """A killed run continues in the cache dir — and IS the run it interrupted.

    Every accepted batch is a ``chunk_NNNNNN.npz`` and ``state.json`` carries the
    counters, the ceiling and the RNG state, so restoring them continues the same
    sub-seed stream: the resumed sample is bit-identical to an uninterrupted run
    at the same seed.  ``max_loops=2`` here is how a wall-clock kill is imitated
    — the partial pool is on disk and the state is not marked done.
    """
    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=1)
    part_dir = str(tmp_path / "part")
    with pytest.raises(RuntimeError, match="accepted only"):
        f.generate_toy(x0, 2000, seed=7, batch_size=100, max_loops=2,
                       cache_dir=part_dir)
    assert (tmp_path / "part" / "state.json").exists()
    resumed = f.generate_toy(x0, 2000, seed=7, batch_size=100,
                             cache_dir=part_dir)
    full = f.generate_toy(x0, 2000, seed=7, batch_size=100,
                          cache_dir=str(tmp_path / "fresh"))
    assert len(resumed["mass"]) == 2000
    for k in ("mass", "q", "angle", "weight"):
        assert np.array_equal(resumed[k], full[k]), k

    # a finished cache is the SAME run (same signature == same sample), so a
    # second call starts over instead of replaying or blending it
    again = f.generate_toy(x0, 2000, seed=7, batch_size=100,
                           cache_dir=part_dir)
    assert np.array_equal(again["mass"], full["mass"])


def test_a_cache_killed_before_its_first_event_resumes(tmp_path):
    """An empty pool still resumes — the first accepted batch defines the columns.

    A loose ``maxp=`` makes an early batch accept nothing, so ``state.json`` can
    hold ``have=0`` with no chunk file; the resume must then take its column set
    from the next batch instead of from an empty dict.
    """
    import json

    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=1)
    d = str(tmp_path / "empty")
    with pytest.raises(RuntimeError, match="accepted only 0 of 40"):
        f.generate_toy(x0, 40, seed=11, batch_size=1, maxp_scale=10.0,
                       max_loops=1, cache_dir=d)
    with open(tmp_path / "empty" / "state.json", encoding="utf-8") as fh:
        st = json.load(fh)
    assert st["have"] == 0 and st["n_chunks"] == 0 and not st["done"]
    data = f.generate_toy(x0, 40, seed=11, batch_size=5, maxp_scale=10.0,
                          cache_dir=d)
    assert len(data["mass"]) == 40


def test_a_cache_from_another_run_is_refused(tmp_path):
    """Reusing a directory for a different point would BLEND two samples."""
    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=1)
    d = str(tmp_path / "cache")
    f.generate_toy(x0, 30, seed=1, batch_size=200, cache_dir=d)
    with pytest.raises(ValueError, match="another run"):
        f.generate_toy(x0, 30, seed=2, batch_size=200, cache_dir=d)
    with pytest.raises(ValueError, match="another run"):
        f.generate_toy(x0, 40, seed=1, batch_size=200, cache_dir=d)
    with pytest.raises(ValueError, match="explicit seed"):
        f.generate_toy(x0, 30, batch_size=200, cache_dir=str(tmp_path / "x"))


def test_progress_reports_a_live_run(capsys):
    """``progress=`` shows the run is alive; the default is AUTO (off off-tty)."""
    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=1)
    f.generate_toy(x0, 30, seed=1, batch_size=500)
    assert "toy:" not in capsys.readouterr().out      # AUTO: not a terminal here
    f.generate_toy(x0, 30, seed=1, batch_size=500, progress=True)
    out = capsys.readouterr().out
    assert "toy: 30/30 accepted" in out and "batches" in out
    f.generate_toy(x0, 30, seed=1, batch_size=500, progress=False)
    assert "toy:" not in capsys.readouterr().out


def test_generate_toy_works_for_the_polarised_model():
    """A THIRD model shape, generic path: the ρ-matrix PWA.

    Nothing in the toy knows about ``pwa``: it asks the model for a generator
    (``None`` here → the generic accept-reject), the model's own
    ``build_event_data``, and the fitter's parameter layer + backend.  So a
    model that is not plain PWA works with no change to the generator.
    """
    from tabpwa.config_loader import RawConfig

    dic = RawConfig(CFG_PWA).to_dict()
    dic["amp_model"] = "polarised_pwa"
    f = Fitter(dic, backend="numpy_polarised")
    f.apply_constrains()
    x0 = f.initial_values(seed=1)
    data, stats = f.generate_toy(x0, 60, seed=2, batch_size=500, stats=True)
    assert len(data["mass"]) == 60 and stats.maxp > 0

    params, _ = f.build_params(x0)
    _, _, P = f.backend.compute(params, f.backend.load_data(data),
                                norm=None, return_p=True)
    assert np.isfinite(P).all() and P.max() <= stats.maxp


def test_the_extra_var_declaration_defines_the_toy_law():
    """``data.extra_var.<name>.random`` is the ONE definition of the law.

    Changing it changes the toy — no second copy in Python — and the same
    declaration fills a background sample, so signal and background agree by
    construction.
    """
    from tabpwa.config_loader import RawConfig

    dic = RawConfig(CFG_TD).to_dict()
    dic["data"]["extra_var"]["time"] = {"default": 1.0,
                                        "random": {"dist": "uniform", "low": 2.0,
                                                   "high": 3.0}}
    dic["data"]["extra_var"]["frac"] = {"default": 0.0,
                                        "random": {"dist": "choice", "a": [1]}}
    f = Fitter(dic, backend="numpy")
    f.apply_constrains()
    x0 = f.initial_values(seed=1)
    data = f.generate_toy(x0, 6, seed=2, oversample=1)
    assert data["time"].min() >= 2.0 and data["time"].max() <= 3.0
    assert np.all(data["frac"] == 1.0)

    # a declared DEFAULT (no random:) means a constant
    dic["data"]["extra_var"]["time"] = {"default": 4.0}
    g = Fitter(dic, backend="numpy")
    g.apply_constrains()
    d2 = g.generate_toy(g.initial_values(seed=1), 3, seed=2, oversample=1)
    assert np.all(d2["time"] == 4.0)


def test_the_td_toy_generates_at_x0():
    """The hook must feed the FLAT x0 to build_params, not a resolved dict.

    ``param_dict`` is a RESOLVED dict and the constraint manager re-resolves it,
    which moved the generating point (measured: rhoA_mass 0.769 → 1.46) while
    ``ToyStats.truth`` still reported x0.
    """
    from tabpwa.toy_generator import _build_params

    f = _fitter(CFG_TD, "numpy")
    x0 = f.initial_values(seed=4)
    want = f.build_params(x0)[0]["m0"]
    assert np.allclose(_build_params(f, x0=x0)["m0"], want)
    # the OLD route (an already-resolved dict) does move the point: the guard
    moved = _build_params(f, param_dict=f.build_params(x0)[1])["m0"]
    assert not np.allclose(moved, want)


def test_the_td_ceiling_is_raised_not_fixed():
    """A later batch above the calibration max RAISES the TD ceiling.

    The old rule was a FIXED ``1.3·max(calibration)`` bound, and an event above
    it is accepted with probability 1 instead of P/P_max — silently biased.  With
    ``p_max_margin=1`` and a 100-event calibration against 1000-event proposal
    batches a later batch exceeds it, so ``p_max >= max accepted P`` is exactly
    the invariant the raise buys.
    """
    from tabpwa.toy_generator import toy_generate

    f = _fitter(CFG_TD, "numpy")
    # 40 accepted, not 300: at this amplitude's 0.09 % efficiency that is 452
    # proposal batches, i.e. the "later batch above the calibration max" is
    # exercised many times over, while the accepted count only buys time
    # (MEASURED: 300 -> 42.5 s, 40 -> 7.2 s, same p_max).
    ev = toy_generate(f, 40, seed=3, compute_batch=100, p_max_margin=1.0,
                      progress=False)
    assert ev["p_max"] >= float(ev["P"].max()) * (1 - 1e-12)
    with pytest.raises(ValueError, match="p_max_margin"):
        toy_generate(f, 5, seed=1, p_max_margin=0.9)
    with pytest.raises(ValueError, match="compute_batch"):
        toy_generate(f, 5, seed=1, compute_batch=0)


def test_the_td_engine_resumes_from_its_cache(tmp_path):
    """The time-dependent engine checkpoints the same way, and resumes exactly."""
    f = _fitter(CFG_TD, "numpy")
    x0 = f.initial_values(seed=4)
    d = str(tmp_path / "td")
    with pytest.raises(RuntimeError, match="accepted only"):
        f.generate_toy(x0, 500, seed=7, compute_batch=100, max_loops=1,
                       cache_dir=d)
    resumed = f.generate_toy(x0, 500, seed=7, compute_batch=100, cache_dir=d)
    full = f.generate_toy(x0, 500, seed=7, compute_batch=100,
                          cache_dir=str(tmp_path / "fresh"))
    assert len(resumed["mass"]) == 500
    for k in ("mass", "q", "angle", "frac", "time"):
        assert np.array_equal(resumed[k], full[k]), k


def test_the_td_runaway_guard(monkeypatch):
    """The time-dependent engine stops on NO progress, like the generic one."""
    from tabpwa.toy_generator import toy_generate

    f = _fitter(CFG_TD, "numpy")

    def dead(params, handle, norm=None, return_p=True):
        mass = handle["mass"] if isinstance(handle, dict) else handle.mass
        return 0.0, {}, np.zeros(len(mass))

    monkeypatch.setattr(f.backend, "compute", dead)
    with pytest.raises(RuntimeError, match="no event accepted for 3 consecutive"):
        toy_generate(f, 5, seed=1, compute_batch=50, stall_loops=3)


def test_the_td_norm_sample_carries_the_sample_law():
    """The norm is integrated over the SAMPLE's law, not the builder's defaults.

    With ``frac = 0.5, time = 0`` the integral is a point measure in the mixing
    variables — measured 4.7 % off the law-integrated one.
    """
    f = _fitter(CFG_TD, "numpy")
    phsp = f.generate_phsp(40, seed=5)
    assert set(np.unique(phsp["frac"])) <= {0.0, 1.0}
    assert phsp["time"].max() > 0.0
    assert np.allclose(phsp["bkg"], 1.0)          # the background SHAPE


def test_extra_var_errors_name_the_variable():
    """A bad declaration must say WHICH column, and be a mapping."""
    from tabpwa.config_loader import RawConfig

    dic = RawConfig(CFG_TD).to_dict()
    f = Fitter(dic, backend="numpy")
    f.apply_constrains()
    x0 = f.initial_values(seed=1)

    dic["data"]["extra_var"]["time"] = {"default": 1.0,
                                        "random": {"dist": "not_a_dist"}}
    with pytest.raises(ValueError, match="data.extra_var.time"):
        Fitter(dic, backend="numpy").generate_toy(x0, 4, seed=2, oversample=1)

    dic["data"]["extra_var"]["time"] = 3.0          # not a mapping
    with pytest.raises(ValueError, match="must be a mapping"):
        Fitter(dic, backend="numpy").generate_toy(x0, 4, seed=2, oversample=1)


def test_the_extra_var_declaration_fills_the_background_too():
    """One declaration, both halves of the mixture."""
    from tabpwa.config_loader import RawConfig

    dic = RawConfig(CFG_TD).to_dict()
    dic["data"]["extra_var"]["time"] = {"default": 1.0,
                                        "random": {"dist": "uniform", "low": 2.0,
                                                   "high": 3.0}}
    f = Fitter(dic, backend="numpy")
    f.apply_constrains()
    x0 = f.initial_values(seed=1)
    data, st = f.generate_toy_bg(x0, 6, purity=0.0, seed=3, stats=True)
    assert st.n_background == len(data["mass"]) > 0
    assert data["time"].min() >= 2.0 and data["time"].max() <= 3.0


def test_the_poisson_split_can_be_empty_and_says_so():
    """``n_sig = n_bg = 0`` is possible (37 % at n_data=1) — refuse, don't return {}."""
    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=1)
    for seed in range(20):
        try:
            f.generate_toy_bg(x0, 1, purity=0.5, seed=seed)
        except ValueError as exc:
            assert "realised 0 events" in str(exc)
            break
    else:                                          # pragma: no cover
        pytest.skip("no empty draw in 20 seeds")


def test_the_frac_time_overrides_are_scalars():
    """A batch-length array cannot be an override; the message says why."""
    f = _fitter(CFG_TD, "numpy")
    x0 = f.initial_values(seed=1)
    with pytest.raises(ValueError, match="override has 4 values"):
        f.generate_toy(x0, 5, seed=1, time=np.arange(4.0))
    d = f.generate_toy(x0, 3, seed=1, oversample=1, time=7.5, frac=1.0)
    assert np.all(d["time"] == 7.5) and np.all(d["frac"] == 1.0)


def test_generate_toy_carries_a_top_momentum():
    """``top_p`` reaches the generator — and the SAME frame for data and phsp."""
    f = _fitter(CFG_PWA, "numpy_pwa")
    x0 = f.initial_values(seed=3)
    beam = fix_beam(0.034, 0.0, 0.0)
    data = f.generate_toy(x0, 100, seed=7, batch_size=2000, top_p=beam)
    phsp = f.generate_phsp(100, seed=7, top_p=beam)
    assert len(data["mass"]) == 100 and len(phsp["mass"]) == 100
    # a boosted event still satisfies the model's own invariants: the mass row
    # carries every internal invariant mass, so it is finite and positive
    assert np.isfinite(data["mass"]).all() and (data["mass"] > 0).all()


def test_generate_toy_dispatches_the_time_dependent_model():
    """``flavour_tag_mix``: frac/time columns, and no norm sample from here."""
    f = _fitter(CFG_TD, "numpy")
    x0 = f.initial_values(seed=4)
    data, stats = f.generate_toy(x0, 5, seed=3, oversample=2, stats=True)
    assert stats.maxp is None
    assert len(data["mass"]) == 5
    assert {"mass", "q", "angle", "frac", "time", "weight"} <= set(data)
    assert set(np.unique(data["frac"])) <= {0.0, 1.0}   # declared: choice(0, 1)
    assert data["time"].min() >= 0.0

    with pytest.raises(ValueError, match="top_p"):
        f.generate_toy(x0, 5, top_p=fix_beam(0.034))
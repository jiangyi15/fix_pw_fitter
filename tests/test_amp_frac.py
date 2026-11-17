"""AmplitudeFractions: kernel-param handling is delegated to the transform."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CONFIG = os.path.join(ROOT, "tests", "config_pwa.yml")


class _StubResult:
    """Minimal OptimizeResult stand-in (only ``x`` / ``hess_inv`` are read)."""

    def __init__(self, x):
        self.x = x
        self.hess_inv = None


def _make():
    from tabpwa import Fitter
    from tabpwa.amp_frac import AmplitudeFractions

    f = Fitter(CONFIG, backend="numpy_pwa")
    f.apply_constrains()
    x = f.initial_values(seed=3)
    af = AmplitudeFractions(f, _StubResult(x))          # sample not needed
    return f, af


def test_default_param_names_are_the_free_variables():
    f, af = _make()
    try:
        names = af._default_param_names()
        assert names == list(f.cm.var_registry.flat_names)
        # m0/g0 must be included iff they are actually free.  This config
        # fixes them, so the old code's unconditional re-listing of
        # m0_phys_name/g0_phys_name would have (wrongly) propagated them.
        m0g0 = set(f.model.m0_phys_name) | set(f.model.g0_phys_name)
        assert m0g0.isdisjoint(names)
    finally:
        f.backend.free()


def test_phys_to_params_matches_the_transform_baseline():
    f, af = _make()
    try:
        base = af._phys_to_params({})
        assert set(base) >= {"ck", "m0", "g0"}
        assert np.allclose(base["ck"], af._params["ck"])
        assert np.allclose(base["m0"], af._params["m0"])
        assert np.allclose(base["g0"], af._params["g0"])
    finally:
        f.backend.free()


def test_phys_to_params_writes_m0_slot():
    f, af = _make()
    try:
        names = f.model.m0_phys_name
        name = names[0]
        idx = names.index(name)
        base = af._phys_to_params({})
        p = af._phys_to_params({name: 1.234})
        assert p["m0"][idx] == 1.234
        assert np.allclose(np.delete(p["m0"], idx), np.delete(base["m0"], idx))
    finally:
        f.backend.free()


def test_phys_to_params_now_applies_ck_overrides():
    """ck overrides used to be silently ignored; they must now take effect."""
    f, af = _make()
    try:
        m0g0 = set(f.model.m0_phys_name) | set(f.model.g0_phys_name)
        ck_names = [n for n in af._resolved
                    if n not in m0g0 and n.endswith("r")]
        assert ck_names, "expected some ck parameter names"
        name = ck_names[0]
        base = af._phys_to_params({})
        p = af._phys_to_params({name: float(af._resolved[name]) + 0.75})
        assert not np.allclose(p["ck"], base["ck"])
    finally:
        f.backend.free()


def test_compute_masks_ck_gradient_and_passes_other_groups_through(monkeypatch):
    f, af = _make()
    monkeypatch.setattr(af, "_get_handle", lambda: object())   # compute is faked
    try:
        fake = {
            "ck": np.ones(af._n_ck, dtype=complex),
            "m0": np.zeros(len(f.model.m0_phys_name)),
            "g0": np.zeros(len(f.model.g0_phys_name)),
        }
        monkeypatch.setattr(f.backend, "compute",
                            lambda p, d, norm=None: (2.0, dict(fake), None))

        seen = {}
        real_backward = f._kernel_builder.backward

        def spy(grads, resolved):
            seen["keys"] = set(grads)
            seen["ck"] = np.asarray(grads["ck"]).copy()
            return real_backward(grads, resolved)

        monkeypatch.setattr(f._kernel_builder, "backward", spy)

        total, _ = af._compute_total_and_grad(af._params, mask=[0])
        assert total == 2.0
        # it must not invent parameter groups (e.g. a fake "scalar")
        assert seen["keys"] == {"ck", "m0", "g0"}
        # inside the mask kept, outside the mask zeroed
        assert seen["ck"][0] == 1.0
        assert all(seen["ck"][i] == 0.0 for i in range(1, af._n_ck))
    finally:
        f.backend.free()


def test_fwd_integrate_takes_a_handle():
    """AmpFrac integrates one handle; sum_stream does the chunking."""
    import numpy as np
    from tabpwa import Fitter
    from tabpwa.amp_frac import AmplitudeFractions
    from test_pwa_gen import _synth

    f = Fitter(CONFIG, backend="numpy_pwa")
    f.apply_constrains()
    kc = f.kernel_config
    f.set_phsp(_synth(kc, 120, 1))
    x = f.initial_values(seed=3)

    class _R(_StubResult):
        def __init__(self, x):
            super().__init__(x)
            self.hess_inv = np.eye(len(x)) * 0.02

    r = _R(x)
    af = AmplitudeFractions(f, r)
    data = _synth(kc, 200, 5)
    masks = [[1], [2], [1, 2]]

    # integrand = masked kernel params at the best fit
    spec = af.integrand([1])
    assert spec["resolved"] is not None
    assert np.all(spec["params"]["ck"][[0, 2]] == 0)

    # one handle in, one Fwd per mask out
    one = af.fwd_integrate(masks, f.backend.load_data(data))
    assert len(one) == 3
    assert all(o.grad.shape == (len(af._default_param_names()),) for o in one)
    assert np.any(one[0].grad != 0.0)

    # the streaming (chunking, loading, accumulation) is the fitter's
    streamed = f.sum_stream(lambda h: af.fwd_integrate(masks, h),
                            data=data, batch=64)
    for a, b in zip(one, streamed):
        assert a.value == pytest.approx(b.value, rel=1e-10)
        np.testing.assert_allclose(a.grad, b.grad, rtol=1e-10, atol=1e-14)

    # [1,2] includes the 1-2 interference, so it is not the plain sum
    assert one[2].value != pytest.approx(one[0].value + one[1].value)
    frac = streamed[0] / streamed[2]
    assert 0.0 <= frac.value <= 1.0
    assert frac.sigma(r.hess_inv) > 0.0

    # a mask over a *fixed* coupling legitimately has a zero gradient
    fixed = af.fwd_integrate([[0]], f.backend.load_data(data))[0]
    assert np.all(fixed.grad == 0.0)


def test_fractions_batch_streams_the_same_values():
    """`batch=` routes through sum_stream; values/errors are unchanged."""
    import numpy as np
    from tabpwa import Fitter
    from tabpwa.amp_frac import AmplitudeFractions
    from test_pwa_gen import _synth

    f = Fitter(CONFIG, backend="numpy_pwa")
    f.apply_constrains()
    kc = f.kernel_config
    f.set_phsp(_synth(kc, 120, 1))
    x = f.initial_values(seed=3)

    class _R(_StubResult):
        def __init__(self, x):
            super().__init__(x)
            self.hess_inv = np.eye(len(x)) * 0.02

    af = AmplitudeFractions(f, _R(x))
    masks = [[1], [2], [1, 2]]

    v0, e0 = af.fractions(masks)                  # whole sample, one pass
    v1, e1 = af.fractions(masks, batch=64)        # streamed through sum_stream
    np.testing.assert_allclose(v1, v0, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(e1, e0, rtol=1e-10, atol=1e-15)

    # a batch larger than the sample is a single chunk
    v2, e2 = af.fractions(masks, batch=10 ** 6)
    np.testing.assert_allclose(v2, v0, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(e2, e0, rtol=1e-10, atol=1e-15)

    # the finite-difference path works too, batched and not — and agrees with
    # the analytic one (it used to return zero errors: the mask cache went
    # stale across perturbations and the perturbed ck was discarded)
    for b in (None, 64):
        v_fd, e_fd = af.fractions(masks, jac=False, batch=b)
        np.testing.assert_allclose(v_fd, v0, rtol=1e-12, atol=1e-15)
        assert np.all(e_fd > 0)
        np.testing.assert_allclose(e_fd, e0, rtol=1e-6, atol=1e-12)


def test_data_form_is_the_sample_that_gets_integrated():
    """Arrays given to the constructor are what a batched run slices."""
    import numpy as np
    from tabpwa import Fitter
    from tabpwa.amp_frac import AmplitudeFractions
    from test_pwa_gen import _synth

    f = Fitter(CONFIG, backend="numpy_pwa")
    f.apply_constrains()
    kc = f.kernel_config
    ph = _synth(kc, 120, 1)
    f.set_phsp(ph)
    x = f.initial_values(seed=3)

    class _R(_StubResult):
        def __init__(self, x):
            super().__init__(x)
            self.hess_inv = np.eye(len(x)) * 0.02

    masks = [[1], [2], [1, 2]]

    # the fitter's phase space (data=None)
    af_default = AmplitudeFractions(f, _R(x))
    v0, e0 = af_default.fractions(masks)
    v1, _ = af_default.fractions(masks, batch=64)
    np.testing.assert_allclose(v1, v0, rtol=1e-12, atol=1e-15)

    # a different sample given as arrays: used by both routes, and different
    other = _synth(kc, 90, 11)
    af_other = AmplitudeFractions(f, _R(x), data=other)
    v_arr, _ = af_other.fractions(masks)
    v_arr_b, _ = af_other.fractions(masks, batch=32)
    np.testing.assert_allclose(v_arr_b, v_arr, rtol=1e-12, atol=1e-15)
    assert not np.allclose(v_arr, v0, rtol=1e-6)

    # the sample is arrays only — a handle is refused, and the handle this
    # instance builds is freed by close()
    with pytest.raises(TypeError, match="must be a numpy dict"):
        AmplitudeFractions(f, _R(x), data=f.backend.load_data(other))
    af_other._get_handle()
    af_other.close()                                   # idempotent
    af_other.close()


def test_handle_resolution_order():
    """data arrays > fitter._phsp_holder > load of _phsp_np."""
    import numpy as np
    from tabpwa import Fitter
    from tabpwa.amp_frac import AmplitudeFractions
    from test_pwa_gen import _synth

    f = Fitter(CONFIG, backend="numpy_pwa")
    f.apply_constrains()
    kc = f.kernel_config
    f.set_phsp(_synth(kc, 60, 1))
    x = f.initial_values(seed=3)

    class _R(_StubResult):
        def __init__(self, x):
            super().__init__(x)
            self.hess_inv = None

    af = AmplitudeFractions(f, _R(x))
    assert af._get_handle() is f._phsp_holder          # the fitter's handle
    assert af._own is False

    other = _synth(kc, 30, 3)
    af2 = AmplitudeFractions(f, _R(x), data=other)
    h = af2._get_handle()
    assert h is not f._phsp_holder and af2._own is True
    af2.close()
    assert af2._handle is None
    assert f._phsp_holder is not None                  # fitter untouched

    # holder gone but arrays present -> load them
    f2 = Fitter(CONFIG, backend="numpy_pwa")
    f2.apply_constrains()
    f2.set_phsp(_synth(kc, 60, 1))
    f2._phsp_holder = None
    af3 = AmplitudeFractions(f2, _R(x))
    assert af3._get_handle() is not None and af3._own is True
    af3.close()


def test_batched_run_never_loads_the_whole_sample():
    """batch= must not build the full-sample handle (that would defeat it)."""
    import numpy as np
    from tabpwa import Fitter
    from tabpwa.amp_frac import AmplitudeFractions
    from test_pwa_gen import _synth

    f = Fitter(CONFIG, backend="numpy_pwa")
    f.apply_constrains()
    f.set_phsp(_synth(f.kernel_config, 120, 1))
    x = f.initial_values(seed=3)

    class _R(_StubResult):
        def __init__(self, x):
            super().__init__(x)
            self.hess_inv = np.eye(len(x)) * 0.02

    af = AmplitudeFractions(f, _R(x))
    assert af._handle is None

    # count the loads: one per chunk, never one for the whole sample
    loads = []
    real_load = f.backend.load_data
    f.backend.load_data = lambda d: (loads.append(len(d["mass"])), real_load(d))[1]
    try:
        af.fractions([[1], [2]], batch=32)
    finally:
        f.backend.load_data = real_load
    assert loads == [32, 32, 32, 24]          # 120 events, 4 chunks, no full load
    assert af._handle is None                 # and nothing cached on the instance

    # the unbatched call does resolve a handle (the fitter's, not a new load)
    af.fractions([[1]])
    assert af._handle is f._phsp_holder


def test_fractions_without_hess_inv_report_zero_errors():
    """No covariance -> zero errors, on every route (long-standing behaviour).

    ``cal_uncertainties_multi`` returns a zero covariance when ``hess_inv`` is
    ``None``; ``fractions`` must keep passing that through rather than
    inventing an error, because the values are still well defined.
    """
    import numpy as np
    from tabpwa import Fitter
    from tabpwa.amp_frac import AmplitudeFractions
    from test_pwa_gen import _synth

    f = Fitter(CONFIG, backend="numpy_pwa")
    f.apply_constrains()
    f.set_phsp(_synth(f.kernel_config, 90, 1))
    x = f.initial_values(seed=3)

    af = AmplitudeFractions(f, _StubResult(x))          # hess_inv is None
    for kw in ({}, {"batch": 32}, {"jac": False}):
        v, e = af.fractions([[1], [2]], **kw)
        assert np.all(np.isfinite(v))
        np.testing.assert_array_equal(np.asarray(e), 0.0)

    # with a covariance the same call returns non-zero errors
    class _R(_StubResult):
        def __init__(self, x):
            super().__init__(x)
            self.hess_inv = np.eye(len(x)) * 0.02

    v, e = AmplitudeFractions(f, _R(x)).fractions([[1], [2]])
    assert np.all(e > 0)


def test_uncertainties_agree_with_x_space_finite_differences():
    """All routes must match FD in ``x`` — the space ``hess_inv`` describes.

    ``config_pwa.yml`` has no transform whose input is not a flat name, so it
    cannot catch a chain that drops aliases or the bounds Jacobian.  This
    fixture (bounded K-matrix couplings, ``config_pwa_gen_mb.yml``) can.
    """
    import numpy as np
    from tabpwa import Fitter
    from tabpwa.amp_frac import AmplitudeFractions
    from test_pwa_gen import _synth

    f = Fitter("tests/config_pwa_gen_mb.yml", backend="numpy_flat")
    f.apply_constrains()
    f.set_phsp(_synth(f.kernel_config, 120, 1))
    f.set_data(_synth(f.kernel_config, 80, 2))
    res = f.fit(f.initial_values(seed=3), maxiter=2)
    assert res.hess_inv is not None

    af = AmplitudeFractions(f, res)
    x0 = np.asarray(res.x, float)
    masks = [[0], [1]]
    H = res.hess_inv

    def obs(x):
        """Fractions at x, without touching the instance's cached state."""
        params, _ = f.build_params(x)
        vals = []
        for m in [None] + masks:
            keep = np.zeros(params["ck"].shape[0], dtype=bool)
            if m is not None:
                keep[list(m)] = True
            ck = np.array(params["ck"], copy=True)
            ck[~keep if m is not None else np.zeros_like(keep)] = 0.0j
            vals.append(f.backend.compute(dict(params, ck=ck),
                                          af._get_handle(), norm=None)[0])
        return np.array([v / vals[0] for v in vals[1:]])

    J = np.zeros((len(masks), len(x0)))
    for j in range(len(x0)):
        h = 1e-6 * max(1.0, abs(x0[j]))
        up = x0.copy(); up[j] += h
        dn = x0.copy(); dn[j] -= h
        J[:, j] = (obs(up) - obs(dn)) / (2 * h)
    truth = np.array([np.sqrt(max(float(row @ H @ row), 0.0)) for row in J])

    v, e = af.fractions(masks)
    np.testing.assert_allclose(np.asarray(v), obs(x0), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(e), truth, rtol=1e-4, atol=1e-10)

    # the streamed routes must give the same numbers, not merely "close"
    for b in (32, 500):
        vb, eb = af.fractions(masks, batch=b)
        np.testing.assert_allclose(np.asarray(vb), v, rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(np.asarray(eb), e, rtol=1e-5, atol=1e-10)

    # jac=False is a real x-space sweep on both routes
    vf, ef = af.fractions(masks, jac=False)
    np.testing.assert_allclose(np.asarray(vf), v, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(ef), truth, rtol=1e-4, atol=1e-10)
    vfb, efb = af.fractions(masks, jac=False, batch=32)
    np.testing.assert_allclose(np.asarray(vfb), vf, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(efb), ef, rtol=1e-6, atol=1e-12)

    f.free()


def test_named_sweep_equals_the_x_sweep_for_the_names_it_covers():
    """``fd_jacobian(names=…)`` is the same derivative, restricted to a column.

    The named basis exists for observables with a small, genuinely complete
    name set (a Breit-Wigner mass/width).  For one such name its column must
    equal the x-basis column after the analytic chain — that is what makes the
    cheap sweep trustworthy.
    """
    import numpy as np
    from tabpwa import Fitter
    from test_pwa_gen import _synth

    f = Fitter("tests/config_pwa_gen_mb.yml", backend="numpy_flat")
    f.apply_constrains()
    f.set_phsp(_synth(f.kernel_config, 60, 1))
    x = f.initial_values(seed=3)
    flat = list(f.cm.var_registry.flat_names)

    # A name that exists in the pwa_gen parameter surface: the basis weights
    # replace the physical mass/width, and they are TRANSFORM OUTPUTS, so what
    # is fitted is the resonance model's own inputs (`kmatrix2`'s m1/m2/g1/...).
    # (This used to pick `k2_m4`, a basis weight, which was only free while
    # the per-BW transform was missing -- see PWAGen.build_kernel_config.)
    name = "k2_m1"
    j = flat.index(name)
    fn = lambda d: [float(d[name])]

    v_x, J_x = f.fd_jacobian(fn, x=x)
    v_n, J_n = f.fd_jacobian(fn, x=x, names=[name])
    assert J_n.shape == (1, len(x))
    np.testing.assert_allclose(v_n, v_x, rtol=1e-12)
    np.testing.assert_allclose(J_n[0], J_x[0], rtol=1e-6,
                               atol=1e-8 * max(1.0, np.abs(J_x).max()))
    assert abs(J_x[0, j]) > 0                     # the column is not trivial

    # an unresolvable name is reported, not silently dropped
    import pytest
    with pytest.warns(RuntimeWarning, match="not resolved"):
        f.fd_jacobian(fn, x=x, names=[name, "no_such_name"])
    with pytest.raises(ValueError, match="none of the requested names"):
        f.fd_jacobian(fn, x=x, names=["no_such_name"])
    f.free()


def test_fractions_evaluate_the_model_frame_not_the_display_frame():
    """Blinding is display-only: derivatives must be taken where the fit is.

    The program works with the real physics (``build_params(x)[1]``); the
    blinded frame only exists for printing/saving.  Evaluating the model at
    the report frame answers a different question — it describes a parameter
    point the fit never fitted — so ``fractions`` must not use it.
    """
    import numpy as np
    from tabpwa import Fitter
    from tabpwa.amp_frac import AmplitudeFractions
    from test_pwa_gen import _synth

    f = Fitter("tests/config_pwa_gen_mb.yml", backend="numpy_flat")
    f.apply_constrains()
    f.set_phsp(_synth(f.kernel_config, 120, 1))
    f.set_data(_synth(f.kernel_config, 80, 2))
    res = f.fit(f.initial_values(seed=3), maxiter=2)
    f.cm.set_blind(list(f.cm.var_registry.flat_names), seed=1234, scale=0.05)
    af = AmplitudeFractions(f, res)
    x = np.asarray(res.x, float)

    model = f.build_params(x)[1]
    report = f.report_params(x)
    assert any(model[n] != report[n] for n in report if n in model)

    def frac_at(phys):
        p = f._kernel_builder.forward(phys)
        h = af._get_handle()
        s = [float(f.backend.compute(
            dict(p, ck=af._ck_masked(m, p["ck"])), h, norm=None)[0])
            for m in (None, [0])]
        return s[1] / s[0]

    v_model = frac_at(model)
    v_report = frac_at(report)
    assert v_model != v_report                  # the two frames do differ

    v, _ = af.fractions([[0]])
    assert np.asarray(v)[0] == pytest.approx(v_model, rel=1e-10)
    assert not np.isclose(np.asarray(v)[0], v_report, rtol=1e-6)
    f.free()


def test_param_names_means_the_same_on_every_route():
    """``param_names`` must not change the answer just because ``batch=`` is set.

    A route that ignored it (or masked the wrong vocabulary) would return the
    full error for an empty set, or a different error for a subset.
    """
    import numpy as np
    from tabpwa import Fitter
    from tabpwa.amp_frac import AmplitudeFractions
    from test_pwa_gen import _synth

    f = Fitter("tests/config_pwa_gen_mb.yml", backend="numpy_flat")
    f.apply_constrains()
    f.set_phsp(_synth(f.kernel_config, 120, 1))
    f.set_data(_synth(f.kernel_config, 80, 2))
    res = f.fit(f.initial_values(seed=3), maxiter=2)
    af = AmplitudeFractions(f, res)

    full = np.asarray(af.fractions([[0]])[1])
    assert np.all(full > 0)

    # an empty / unknown set means "nothing to vary" on every route
    for names in ([], ["no_such_parameter"]):
        for kw in ({}, {"batch": 32}):
            v, e = af.fractions([[0]], param_names=names, **kw)
            np.testing.assert_array_equal(np.asarray(e), 0.0)

    # a non-empty set is accepted on every route and does not blow up
    for kw in ({}, {"batch": 32}):
        v, e = af.fractions([[0]], param_names=["k2_m1"], **kw)
        assert np.all(np.isfinite(e)) and np.all(np.asarray(e) > 0)
    f.free()


def test_fractions_sees_constraint_changes_after_construction():
    """A ``set_blind`` after construction must move all three routes alike."""
    import numpy as np
    from tabpwa import Fitter
    from tabpwa.amp_frac import AmplitudeFractions
    from test_pwa_gen import _synth

    f = Fitter("tests/config_pwa_gen_mb.yml", backend="numpy_flat")
    f.apply_constrains()
    f.set_phsp(_synth(f.kernel_config, 120, 1))
    f.set_data(_synth(f.kernel_config, 80, 2))
    res = f.fit(f.initial_values(seed=3), maxiter=2)
    af = AmplitudeFractions(f, res)
    before = af.fractions([[0]])[0][0]

    f.cm.set_blind(list(f.cm.var_registry.flat_names), seed=1234, scale=0.05)
    v_unb = af.fractions([[0]])[0][0]
    v_bat = af.fractions([[0]], batch=32)[0][0]
    v_fd = af.fractions([[0]], jac=False)[0][0]
    v_fresh = AmplitudeFractions(f, res).fractions([[0]])[0][0]

    assert not np.isclose(before, v_fresh)      # blinding really moved it
    np.testing.assert_allclose([v_unb, v_bat, v_fd], v_fresh,
                               rtol=1e-9, atol=1e-12)
    f.free()


def test_fractions_follows_set_phsp_after_construction():
    """``set_phsp`` replaces the fitter's handle: a cached one would be stale."""
    import numpy as np
    from tabpwa import Fitter
    from tabpwa.amp_frac import AmplitudeFractions
    from test_pwa_gen import _synth

    f = Fitter("tests/config_pwa_gen_mb.yml", backend="numpy_flat")
    f.apply_constrains()
    f.set_phsp(_synth(f.kernel_config, 120, 1))
    f.set_data(_synth(f.kernel_config, 80, 2))
    res = f.fit(f.initial_values(seed=3), maxiter=2)
    af = AmplitudeFractions(f, res)
    v_a = af.fractions([[0]])[0][0]

    f.set_phsp(_synth(f.kernel_config, 90, 7))
    v_unb = af.fractions([[0]])[0][0]
    v_bat = af.fractions([[0]], batch=32)[0][0]
    v_fresh = AmplitudeFractions(f, res).fractions([[0]])[0][0]

    assert not np.isclose(v_a, v_fresh)         # the sample really changed
    np.testing.assert_allclose([v_unb, v_bat], v_fresh, rtol=1e-9, atol=1e-12)
    f.free()

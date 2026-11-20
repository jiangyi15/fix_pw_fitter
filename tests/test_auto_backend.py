"""``backend="auto"`` — per-model classes that choose once, eagerly.

Each model's ``auto`` class owns its rule in ``choose``; the unit tests feed a
fake ``cuda_state`` (no GPU work) and the shape tests read a real
``Fitter.kernel_config``.  The backend-level tests build real instances and
are guarded by the box's state (a CPU-only box lands on the CPU backend).
"""
import numpy as np
import pytest

import tabpwa.backends.auto_backend as ab
from tabpwa import Fitter
from tabpwa.backends import (
    ComputeBackend, backends_for_model, backend_class, create_backend,
    resolve_backend_spec)
from tabpwa.backends.auto_backend import (
    AutoBackend, AutoFlatPwaGenBackend, AutoPWABackend, flat_shape,
    tile_bytes)

CFG = "tests/config_pwa_gen_km.yml"     # n_res == 1
CFG_MB = "tests/config_pwa_gen_mb.yml"  # n_res == 2 (two resonances per chain)
CFG_PWA = "tests/config_pwa.yml"        # the plain projection-sum PWA model
GB = 1024**3
STATE_AMPLE = dict(has_cuda=True, free_bytes=32 * GB,
                   smem_optin_bytes=99 * 1024)


def _choose(cls, kc, **kw):
    """The decision alone — no host probe, no delegate construction."""
    return cls.choose(cls.__new__(cls), kc, **kw)


def _km_kc():
    return Fitter(CFG).kernel_config


# ── the flat pwa_gen rule (shape first, then the host) ─────────────────

def test_flat_choose_without_cuda(monkeypatch):
    monkeypatch.setattr(ab, "cuda_state", lambda: dict(has_cuda=False))
    assert _choose(AutoFlatPwaGenBackend, _km_kc()) == "numpy_flat"


def test_flat_choose_is_availability_only(monkeypatch):
    """Free memory is reported, never decisive — the sample size is unknown."""
    kc = _km_kc()
    monkeypatch.setattr(
        ab, "cuda_state",
        lambda: dict(has_cuda=True, free_bytes=1 * GB,
                     smem_optin_bytes=99 * 1024))
    tight = _choose(AutoFlatPwaGenBackend, kc)
    monkeypatch.setattr(ab, "cuda_state", lambda: dict(STATE_AMPLE))
    ample = _choose(AutoFlatPwaGenBackend, kc)
    assert tight == ample == "cuda_flat_pwa_gen_s0e"


def test_flat_choose_multi_res_picks_the_general_backend(monkeypatch):
    monkeypatch.setattr(ab, "cuda_state", lambda: dict(STATE_AMPLE))
    kc = Fitter(CFG_MB).kernel_config
    assert flat_shape(kc)[1] == 2
    assert _choose(AutoFlatPwaGenBackend, kc) == "cuda_flat_pwa_gen2"


def test_flat_choose_tile_too_big_picks_the_light_backend(monkeypatch):
    kc = _km_kc()
    n_wave, _n_res, n_basis = flat_shape(kc)
    assert tile_bytes(n_wave, n_basis, int(kc.get("n_proj", 1) or 1)) > 1024
    monkeypatch.setattr(
        ab, "cuda_state",
        lambda: dict(has_cuda=True, free_bytes=32 * GB,
                     smem_optin_bytes=1024))
    assert _choose(AutoFlatPwaGenBackend, kc) == "cuda_flat_pwa_gen_s0c"


# ── the plain PWA rule (host only) ─────────────────────────────────────

def test_pwa_choose_without_cuda(monkeypatch):
    monkeypatch.setattr(ab, "cuda_state", lambda: dict(has_cuda=False))
    kc = Fitter(CFG_PWA, backend="numpy_pwa").kernel_config
    assert _choose(AutoPWABackend, kc) == "numpy_pwa"


SPECIALISED = {"cuda_v4_pwa_cache", "cuda_v4_pwa_s0d", "cuda_v4_pwa_s0e",
               "cuda_v4_pwa_1bw"}


def test_pwa_choose_is_availability_only(monkeypatch):
    """Free memory is reported, never decisive."""
    kc = Fitter(CFG_PWA, backend="numpy_pwa").kernel_config
    monkeypatch.setattr(
        ab, "cuda_state",
        lambda: dict(has_cuda=True, free_bytes=1 * GB,
                     smem_optin_bytes=99 * 1024))
    tight = _choose(AutoPWABackend, kc)
    monkeypatch.setattr(ab, "cuda_state", lambda: dict(STATE_AMPLE))
    ample = _choose(AutoPWABackend, kc)
    assert tight == ample == "cuda_v4_pwa"
    assert tight not in SPECIALISED


def test_pwa_choose_follows_resolution_size(monkeypatch):
    """`resolution_size > 1` needs v5: v4/NumPy have no grouped objective."""
    kc = Fitter(CFG_PWA, backend="numpy_pwa").kernel_config
    monkeypatch.setattr(ab, "cuda_state", lambda: dict(STATE_AMPLE))
    assert _choose(AutoPWABackend, kc) == "cuda_v4_pwa"
    assert _choose(AutoPWABackend, kc, resolution_size=1) == "cuda_v4_pwa"
    assert _choose(AutoPWABackend, kc, resolution_size=4) == "cuda_v5_pwa"
    monkeypatch.setattr(ab, "cuda_state", lambda: dict(has_cuda=False))
    assert _choose(AutoPWABackend, kc, resolution_size=1) == "numpy_pwa"
    assert _choose(AutoPWABackend, kc, resolution_size=4) == "cpu_v5_pwa"


def test_auto_follows_resolution_size_end_to_end():
    f = Fitter(CFG_PWA, backend={"name": "auto", "resolution_size": 2})
    try:
        expected = ("cuda_v5_pwa" if ab.cuda_state()["has_cuda"]
                    else "cpu_v5_pwa")
        assert f.backend.chosen == expected
        if not ab.cuda_state()["has_cuda"]:
            return
        ref = create_backend("cuda_v5_pwa", f.kernel_config, model="pwa",
                             resolution_size=2)
        try:
            f.apply_constrains()
            params, _resolved = f.build_params(f.initial_values(seed=1))
            data = _synth(f.kernel_config, 60, 11)
            Qa, ga, Pa = f.backend.compute(params, f.backend.load_data(data),
                                           norm=1.3)
            Qb, gb, Pb = ref.compute(params, ref.load_data(data), norm=1.3)
            assert abs(Qa - Qb) <= 1e-13
            assert np.max(np.abs(np.asarray(Pa) - np.asarray(Pb))) <= 1e-13
            assert np.max(np.abs(np.asarray(ga["ck"])
                                 - np.asarray(gb["ck"]))) <= 1e-13
        finally:
            ref.free()
    finally:
        f.backend.free()


def test_pwa_auto_builds_on_the_suite_fixture():
    """``config_pwa.yml`` is ``n_res == 1`` for the KERNELS (``// n_wave``).

    ``n_wave`` is ``matrix_angle.shape[1] = P·N``, not ``len(all_comb) = N`` —
    dividing by N overcounts by the projection factor and once made this
    fixture look like ``n_res = 2``, which it is not: the ``n_res == 1``
    variants (``s0d``/``s0e``/``1bw``) all construct on it.
    """
    f = Fitter(CFG_PWA, backend="auto")
    try:
        kc = f.kernel_config
        n_res = int(np.asarray(kc["bw_order"]).size
                    // int(np.asarray(kc["matrix_angle"]).shape[1]))
        assert n_res == 1, n_res
        if ab.cuda_state()["has_cuda"]:
            assert f.backend.chosen == "cuda_v4_pwa"
        else:
            assert f.backend.chosen == "numpy_pwa"
    finally:
        f.backend.free()


# ── the shared shape facts ─────────────────────────────────────────────

def test_flat_shape_mirrors_the_wrapper():
    f = Fitter(CFG)
    kc = f.kernel_config
    shape = flat_shape(kc)
    assert shape == (int(np.asarray(kc["fl_table"]).shape[0]),
                     int(np.asarray(kc["num_table"]).shape[1]),
                     int(np.asarray(kc["basis_table"]).shape[0]))
    if not ab.cuda_state()["has_cuda"]:
        pytest.skip("the mirror is pinned against a real CUDA wrapper")
    k = create_backend("cuda_flat_pwa_gen_s0e", kc, model="pwa_gen")
    try:
        assert shape == (k.kernel.n_wave, k.kernel.n_res, k.kernel.n_basis)
        n_wave, _n_res, n_basis = shape
        assert tile_bytes(n_wave, n_basis, k.kernel.n_proj) \
            == k.kernel.tile_bytes()
    finally:
        k.free()


# ── it is a backend ────────────────────────────────────────────────────

def test_choose_must_return_a_registered_backend(monkeypatch):
    """A stale choice fails loudly — never degrades to another backend."""
    kc = Fitter(CFG_PWA, backend="numpy_pwa").kernel_config
    monkeypatch.setattr(AutoPWABackend, "choose",
                        lambda self, kc, **kw: "cuda_v4_pwa_nonexistent")
    with pytest.raises(RuntimeError) as ei:
        create_backend("auto", kc, model="pwa")
    msg = str(ei.value)
    assert "cuda_v4_pwa_nonexistent" in msg
    assert "allowed" in msg and "cuda_v4_pwa" in msg


def test_every_choice_is_registered():
    """The standing invariant: every rule's answer exists for its model."""
    cases = [
        (AutoFlatPwaGenBackend, "pwa_gen", _km_kc(), {}),
        (AutoFlatPwaGenBackend, "pwa_gen",
         Fitter(CFG_MB).kernel_config, {}),
        (AutoPWABackend, "pwa",
         Fitter(CFG_PWA, backend="numpy_pwa").kernel_config, {}),
        (AutoPWABackend, "pwa",
         Fitter(CFG_PWA, backend="numpy_pwa").kernel_config,
         {"resolution_size": 2}),
    ]
    for cls, model, kc, kw in cases:
        chosen = _choose(cls, kc, **kw)
        assert chosen in backends_for_model(model), (model, chosen, kw)


def test_a_large_sample_uses_the_measured_ordering(monkeypatch):
    """The record is keyed by SHAPE and device, not by the event count alone.

    Same shape as the production config it was measured on (26 waves, 64
    basis, 2 projections, cc 12.0) and a large sample -> the measured order.
    """
    import tabpwa.backends.auto_backend as ab
    sig = ((12, 0), 26, 64, 2)
    assert sig in ab.FLAT_RATE_MEV_S
    best = max(ab.FLAT_RATE_MEV_S[sig], key=ab.FLAT_RATE_MEV_S[sig].get)
    assert best == "cuda_flat_pwa_gen_s0e2"

    kc = dict(_km_kc())
    kc["n_proj"] = 2
    # the recorded shape: 26 waves, 64 basis, 2 projections
    monkeypatch.setattr(ab, "flat_shape", lambda kc: (26, 1, 64))
    monkeypatch.setattr(ab, "cuda_state",
                        lambda: dict(STATE_AMPLE, cc=(12, 0)))
    assert _choose(AutoFlatPwaGenBackend, kc,
                   n_events=8_746_347) == best


def test_a_different_shape_is_not_ordered(monkeypatch):
    """A shape that was never measured gets no ordering — availability stands."""
    import tabpwa.backends.auto_backend as ab
    monkeypatch.setattr(ab, "cuda_state",
                        lambda: dict(STATE_AMPLE, cc=(12, 0)))
    kc = _km_kc()                       # 2 waves, 5 basis: NOT in the record
    assert _choose(AutoFlatPwaGenBackend, kc,
                   n_events=8_746_347) == "cuda_flat_pwa_gen_s0e"


def test_a_small_sample_keeps_the_availability_answer(monkeypatch):
    import tabpwa.backends.auto_backend as ab
    kc = _km_kc()
    monkeypatch.setattr(ab, "cuda_state", lambda: dict(STATE_AMPLE))
    assert _choose(AutoFlatPwaGenBackend, kc,
                   n_events=ab.LARGE_SAMPLE - 1) == "cuda_flat_pwa_gen_s0e"
    assert _choose(AutoFlatPwaGenBackend, kc) == "cuda_flat_pwa_gen_s0e"


def test_another_device_falls_back_to_availability(monkeypatch):
    """The record is for CC 12.0; elsewhere it is not extrapolated."""
    import tabpwa.backends.auto_backend as ab
    kc = _km_kc()
    monkeypatch.setattr(ab, "cuda_state",
                        lambda: dict(STATE_AMPLE, cc=(8, 0)))
    assert _choose(AutoFlatPwaGenBackend, kc,
                   n_events=8_746_347) == "cuda_flat_pwa_gen_s0e"


def test_a_multi_bw_sample_is_still_the_general_backend(monkeypatch):
    import tabpwa.backends.auto_backend as ab
    monkeypatch.setattr(ab, "cuda_state", lambda: dict(STATE_AMPLE))
    kc = Fitter(CFG_MB).kernel_config
    assert _choose(AutoFlatPwaGenBackend, kc,
                   n_events=8_746_347) == "cuda_flat_pwa_gen2"


def test_auto_is_a_registered_backend():
    assert "auto" in backends_for_model("pwa_gen")
    assert "auto" in backends_for_model("pwa")
    # per model — a model with no `auto` registration is simply not eligible,
    # and the registry says so (it is NOT universal)
    assert "auto" not in backends_for_model("flavour_tag_mix")
    with pytest.raises(ValueError, match="unknown backend 'auto'"):
        backend_class("auto", "flavour_tag_mix")
    with pytest.raises(ValueError, match="not registered"):
        resolve_backend_spec("auto",
                             allowed=backends_for_model("flavour_tag_mix"))


def test_create_backend_returns_an_auto_backend():
    kc = _km_kc()
    be = create_backend("auto", kc, model="pwa_gen")
    try:
        assert isinstance(be, AutoBackend)
        assert isinstance(be._delegate, ComputeBackend)
        assert be.chosen == be.choose(kc)     # __init__ used the class's rule
        assert be.describe().startswith(f"auto -> {be.chosen} (")
    finally:
        be.free()


def test_fitter_backend_is_auto_with_the_chosen_delegate():
    f = Fitter(CFG, backend="auto")
    try:
        assert type(f.backend).__name__ == "AutoFlatPwaGenBackend"
        assert isinstance(f.backend, AutoBackend)
        assert f.backend.chosen == f.backend.choose(f.kernel_config)
        if ab.cuda_state()["has_cuda"]:
            assert f.backend.chosen == "cuda_flat_pwa_gen_s0e"
        else:
            assert f.backend.chosen == "numpy_flat"
        assert type(f.backend._delegate) is backend_class(f.backend.chosen,
                                                         "pwa_gen")
    finally:
        f.backend.free()


def test_mb_config_chooses_the_general_backend():
    f = Fitter(CFG_MB, backend="auto")
    try:
        expected = ("cuda_flat_pwa_gen2" if ab.cuda_state()["has_cuda"]
                    else "numpy_flat")
        assert f.backend.chosen == expected
        assert type(f.backend._delegate) is backend_class(f.backend.chosen,
                                                         "pwa_gen")
        if not ab.cuda_state()["has_cuda"]:
            return
        f.apply_constrains()
        params, _resolved = f.build_params(f.initial_values(seed=1))
        data = _synth(f.kernel_config, 40, 7)
        Q, _grads, P = f.backend.compute(params, f.backend.load_data(data),
                                         norm=None)
        assert np.all(np.isfinite(np.asarray(P)))
        assert np.isfinite(Q)
    finally:
        f.backend.free()


def test_config_backend_auto_is_the_auto_backend():
    from tabpwa.config_loader import RawConfig
    d = RawConfig(CFG).to_dict()
    d["backend"] = "auto"
    f = Fitter(d)
    try:
        assert isinstance(f.backend, AutoBackend)
        assert type(f.backend).__name__ == "AutoFlatPwaGenBackend"
        assert f.config.dic["backend"] == "auto"
    finally:
        f.backend.free()


def test_auto_backend_is_not_swept_into_a_broken_state():
    """The registry sweeps construct every pwa_gen name; ``auto`` survives one.

    Constructing ``auto`` must not raise on this box, and every other
    registered leaf builds too (on a CPU-only box ``auto`` lands on the CPU
    backend, and the CUDA siblings are not constructible at all).
    """
    kc = _km_kc()
    names = sorted(backends_for_model("pwa_gen"))
    assert "auto" in names

    be = create_backend("auto", kc, model="pwa_gen")
    try:
        if ab.cuda_state()["has_cuda"]:
            assert be.chosen == be.choose(kc)
        else:
            assert be.chosen == "numpy_flat"
    finally:
        be.free()

    if not ab.cuda_state()["has_cuda"]:
        return
    for name in names:
        if name.startswith("_") or name == "default":
            continue
        b = create_backend(name, kc, model="pwa_gen")
        try:
            assert isinstance(b, ComputeBackend)
        finally:
            b.free()


def test_compute_matches_the_explicit_backend():
    if not ab.cuda_state()["has_cuda"]:
        pytest.skip("the box's choice here is the CPU backend")
    a = Fitter(CFG, backend="auto")
    b = Fitter(CFG, backend=a.backend.chosen)
    try:
        a.apply_constrains()
        params, _resolved = a.build_params(a.initial_values(seed=1))
        data = _synth(a.kernel_config, 60, 3)
        Qa, ga, Pa = a.backend.compute(params, a.backend.load_data(data),
                                       norm=1.3)
        Qb, gb, Pb = b.backend.compute(params, b.backend.load_data(data),
                                      norm=1.3)
        assert abs(Qa - Qb) <= 1e-13
        assert np.max(np.abs(np.asarray(Pa) - np.asarray(Pb))) <= 1e-13
        assert np.max(np.abs(np.asarray(ga["ck"])
                             - np.asarray(gb["ck"]))) <= 1e-13
    finally:
        a.backend.free()
        b.backend.free()


def _synth(kc, n, seed, lo=0.6, hi=1.7):
    """Kernel-shaped synthetic events for the flat ``pwa_gen`` config."""
    r = np.random.RandomState(seed)
    mkey = "num_m_index" if "num_m_index" in kc else "mass_index"
    n_mass = int(np.max(kc[mkey])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1
    n_at = int(np.max(kc["angle_index"])) + 1
    nac = np.asarray(kc["angle_k"]).shape[1]
    return {
        "mass": lo + (hi - lo) * r.rand(n, n_mass),
        "q": np.abs(r.randn(n, n_q)) + 0.3,
        "angle": r.uniform(-np.pi, np.pi, size=(n, n_at, nac)),
        "weight": 0.5 + r.rand(n),
        "bkg": np.abs(r.randn(n)) * 0.1 + 0.05,
    }

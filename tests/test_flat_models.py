"""Axis B of the flat suite: every particle MODEL, as a unit of a config.

See ``_flat_models.py`` for the axis argument (a leaf receives a kernel config,
never a model, so a model that yields an existing structure adds no backend
case).  This file is the other half of that split:

* the model's DECLARATION is checked with no backend at all (rows, weight
  names, slots, `ck` count, `wave_expansion`);
* the model's NUMBERS go through the family's REFERENCE path
  (``Fitter.get_nll``, i.e. ``numpy_flat``) — a model is not a backend case, so
  there is no per-leaf matrix (see ``_flat_models``' docstring for the two
  measurements that make a model-on-every-leaf sweep measure the leaf).  NOTE:
  no absolute value is pinned per model, so a wrong-but-self-consistent forward
  passes here; the FD gate catches a wrong VJP, and a model's absolute numbers
  live in its own file where one exists;
* its GRADIENT is checked by central differences on the fitter's flat vector,
  over one `ck` slot and one slot that is NOT `ck` (a mass, a width or a
  basis weight — whichever the model owns).

The live gates (``test_every_registered_model_is_covered``,
``test_every_fixture_has_an_owner``) fail when a particle model or a
``config_pwa_gen*`` fixture is added without a decision here — the same
property ``_flat_compute.test_the_live_registry_is_fully_swept`` gives the
backend axis.
"""
import glob
import os

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend
from tabpwa.param_constraint import complex_tail
from tabpwa.particle_model import ALL_MODELS

import _flat_models as fm
from _flat_grid import CFG_ONE, _in_grid

MODEL_NAMES = sorted(fm.MODELS)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), 1e-300))


def _fitter(name):
    return Fitter(fm.config(name))


def _grad_blocks(g):
    """The gradient as ``{block: array}`` whatever ``get_nll`` returned."""
    if isinstance(g, dict):
        out = {}
        for k, v in g.items():
            if isinstance(v, tuple):
                out[k] = np.concatenate([np.ravel(np.asarray(a)) for a in v])
            else:
                out[k] = np.ravel(np.asarray(v))
        return out
    arr = np.ravel(np.asarray(g))
    return {"flat": arr}


# ── the live gates ────────────────────────────────────────────────────────

def test_every_registered_model_is_covered():
    """A registered model is in ``MODELS`` or ``EXCLUDED`` — never ignored.

    ``EXCLUDED`` is not a soft skip: each entry must name the reason and a
    test file that covers it instead, and that file must exist.
    """
    mods, excluded = set(fm.MODELS), set(fm.EXCLUDED)
    assert not (mods & excluded), f"both axes list {sorted(mods & excluded)}"
    # a TEST-SIDE stub (`@register_model` inside a test module, e.g.
    # `test_pwa_gen`'s `direct_weights`) is not part of the product: a model
    # SHIPPED by the package is.  The class's defining module is the fact that
    # separates them, not its name or a list.
    shipped = {n for n, cls in ALL_MODELS.items()
               if cls.__module__.split(".")[0] == "tabpwa"}
    # coverage is by the CLASS an entry instantiates, not by its label: a model
    # may legitimately appear twice (`kmatrix_n_2pole` is a second parameter
    # point of `kmatrix_n`), and its label is not a registered name.
    covered = {fm.MODELS[n].block.get("model", "basis") for n in mods}
    missing = shipped - covered - excluded
    assert not missing, (
        f"particle model(s) {sorted(missing)} are neither in _flat_models.MODELS "
        f"nor in EXCLUDED (a new model needs a decision, not a silent gap)")
    stale = covered - shipped
    assert not stale, f"table instantiates unregistered model(s) {sorted(stale)}"
    for name, (why, owner) in fm.EXCLUDED.items():
        assert why and owner, f"EXCLUDED[{name!r}] needs a reason and an owner"
        assert os.path.exists(owner), f"EXCLUDED[{name!r}] owner {owner} is gone"


def test_every_fixture_has_an_owner():
    """Every flat fixture is a STRUCTURE (axis A) or names the file that owns it.

    A fixture is not a leaf case because a MODEL changed: a kernel branches on
    `n_res` / `n_blocks` / the basis width, never on which model produced the
    arrays.  So a model-only fixture (``_cidx``, ``_cidx_2f``) is owned by the
    model's own file, and a fixture with no claim at all is a gap.
    """
    on_disk = sorted(glob.glob("tests/config_pwa_gen*.yml"))
    assert on_disk, "no flat fixtures found"
    unclaimed = [p for p in on_disk
                 if p not in fm.CONFIGS and p not in fm.OWNED_ELSEWHERE]
    assert not unclaimed, (
        f"{unclaimed} are neither a structural class (_flat_models.CONFIGS) "
        f"nor owned by a file (OWNED_ELSEWHERE): a model-only fixture belongs "
        f"to its model, not to the leaf sweep")
    for path, owner in fm.OWNED_ELSEWHERE.items():
        assert os.path.exists(owner), f"{path} owner {owner} is gone"
        assert path not in fm.CONFIGS, f"{path} is both a class and owned"


def test_the_declared_needs_match_the_built_kernel_config():
    """ Each ``CONFIGS`` entry's ``needs`` is what its BUILT kc actually has.

    The declaration is not prose: ``needs_of`` reads the kc (``n_res > 1`` is a
    multi-BW chain), so a fixture that grows a second factor without saying so
    fails here instead of being swept onto leaves that cannot take it.
    """
    for path, fix in fm.CONFIGS.items():
        kc = Fitter(path).kernel_config
        assert fm.needs_of(kc) == fix.needs, path


def test_axis_a_is_one_fixture_per_kc_class():
    """Two axis-A fixtures may share a class only where the docstring says so.

    The class is shapes + `n_proj`: two fixtures with the same class drive the
    same backend code on the same shapes, so a second one is a model case (axis
    B).  ``_ident``/``_same`` are the documented exception — the same shape,
    two different index mechanisms that a shape check cannot see.
    """
    classes = {}
    for path in fm.AXIS_A:
        f = Fitter(path)
        classes.setdefault(fm.kc_class(f.kernel_config), []).append(path)
    for cls, paths in sorted(classes.items()):
        if len(paths) > 1:
            assert set(paths) == {"tests/config_pwa_gen_ident.yml",
                                  "tests/config_pwa_gen_same.yml"}, (
                f"{paths} share the kc class {cls} — the second one adds no "
                f"backend case; put it in MODELS or state the exception")


def test_the_template_renders_the_fixture_for_the_default_basis():
    """The in-memory template IS the shipped fixture for the default basis.

    The model axis must not depend on a fixture (a model is a unit of a config,
    so a minimal template is enough) — but "enough" has to mean the same kernel
    config, key by key, or the axis would test a different structure than the
    rest of the family.  The fixture is the CROSS-CHECK here, not the source.
    """
    a = Fitter(fm.config("basis")).kernel_config
    b = Fitter(CFG_ONE).kernel_config
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    for key in a:
        va, vb = np.asarray(a[key]), np.asarray(b[key])
        assert va.shape == vb.shape, f"{key}: {va.shape} != {vb.shape}"
        assert np.array_equal(va, vb), (
            f"{key} differs between the template and {CFG_ONE}")


# ── the declaration, with no backend at all ───────────────────────────────

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_declares_its_kernel_config(name):
    """Rows, weight names, heads, slots and the expansion, as declared."""
    want = fm.MODELS[name]
    f = _fitter(name)
    kc = f.kernel_config
    n_ck = len(f.model.get_ck_map())

    assert n_ck == want.n_wave, f"{name}: len(ck) {n_ck} != {want.n_wave}"
    assert tuple(kc["den_names"]) == want.den_names, f"{name}: denominator rows"
    assert int(np.asarray(kc["basis_table"]).shape[0]) == want.n_basis, \
        f"{name}: basis rows"
    assert tuple(np.asarray(kc["num_table"]).shape) == want.num_shape, \
        f"{name}: num_table shape"
    assert f.model.wave_expansion() == [tuple(e) for e in want.expansion], \
        f"{name}: wave expansion"

    # ONE flat `ck` slot per free numerator row, contiguous, in base order
    total = sum(count for _b, _o, count in want.expansion)
    assert n_ck >= total, f"{name}: {n_ck} slots < {total}"
    for base, offset, count in want.expansion:
        assert 0 <= offset and count >= 1
    # the fixed constant weight is produced once for the WHOLE fit, never
    # per resonance (`one` is served by PWAGen.constant_weight_values)
    assert "k2_one" not in want.den_names or name in ("ExpCheb",), \
        f"{name}: a per-resonance `one` weight would stay free"


def test_the_model_axis_runs_on_the_reference_path():
    """``pwa_gen``'s own default is the reference, so the FD path IS ``numpy_flat``.

    Pinned rather than assumed: if the model's registered default ever moved to
    a cached leaf, the FD tolerance would silently start measuring that leaf's
    floor (the reason this axis has no per-leaf matrix at all).
    """
    f = _fitter("basis")
    try:
        ref = create_backend(fm.REFERENCE, f.kernel_config, model="pwa_gen")
        try:
            assert isinstance(f.backend, type(ref)), (
                f"the model axis expected {fm.REFERENCE} as pwa_gen's default, "
                f"got {type(f.backend).__name__}")
        finally:
            ref.free()
    finally:
        f.backend.free()


# ── the gradient chain ───────────────────────────────────────────────────

def _slots(f):
    """FD slots: the first `ck` slot, plus the first slot that is NOT `ck`.

    A model whose weights ARE its parameters (`one`, `skip`, `spline_c_idx`, a
    fixed-shape row) legitimately has no second kind of slot, so the second
    slot falls back to another `ck` one — what must not happen is an empty
    sweep, which is why the caller asserts what it got.
    """
    mag, phase = complex_tail()
    heads = set()
    for combo in f.model.get_ck_map():
        for nm in np.atleast_1d(combo).ravel():
            heads.add(str(nm))
    ck_names = {h + s for h in heads for s in (mag, phase)}
    names = list(f.cm.var_registry.flat_names)
    free = set(f.free_param_names())
    ck_free = [n for n in names if n in ck_names and n in free]
    assert ck_free, f"{type(f.model).__name__}: no free `ck` slot"
    other = [n for n in names if n not in ck_names and n in free]
    return ck_free[0], (other[0] if other else ck_free[1 if len(ck_free) > 1 else 0])


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_gradient_matches_finite_differences(name):
    """Central differences on the flat vector: one `ck` slot and one weight.

    The second slot is the model's OWN: a mass, a width, a K-matrix beta or a
    spline knot — whichever its transform publishes — so the chain
    parameter -> basis weight -> `coeff`/`ck` is exercised, not just the `ck`
    block the kernel differentiates directly.
    """
    f = _fitter(name)
    kc = f.kernel_config
    f.set_phsp(_in_grid(kc, 200, seed=3))
    f.set_data(_in_grid(kc, 64, seed=2))
    x0 = f.initial_values(seed=1)
    nll, grad = f.get_nll(x0)
    assert np.isfinite(nll)
    blocks = _grad_blocks(grad)
    flat = blocks.get("flat")
    assert flat is not None and flat.shape == x0.shape, \
        f"{name}: get_nll did not return a flat gradient"
    assert np.all(np.isfinite(flat)), f"{name}: non-finite gradient"

    names = list(f.cm.var_registry.flat_names)
    h = 1e-6
    worst = 0.0
    for slot in _slots(f):
        i = names.index(slot)
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h
        xm[i] -= h
        fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2.0 * h)
        assert fd != 0.0 or abs(flat[i]) < 1e-12, f"{name}: {slot} has no slope"
        err = abs(flat[i] - fd)
        # MEASURED over this table: worst relative gap 1.1e-6 (`Exp`'s 52-row
        # basis, whose amplitude is sharply curved in its weights), i.e. the
        # central-difference truncation at h = 1e-6, not a gradient defect.
        assert err <= 1e-5 * abs(fd) + 1e-6, (
            f"{name}: d/d{slot} analytic {flat[i]:.6g} vs fd {fd:.6g}")
        worst = max(worst, err)

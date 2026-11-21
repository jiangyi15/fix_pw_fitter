"""The fixed-coefficient mask is DERIVED, not measured.

The ``fixed_coeff`` backend wants the columns of the flat ``coeff`` vector that
no free parameter can move, so it can cache/skip them.  This suite pins the
derivation and checks it against an independent one:

* ``AmplitudeModel.fixed_coeff_columns`` — the SYMBOLIC route: the constraint
  graph (free names, alias groups, transform edges), the public API;
* ``AmplitudeModel.fixed_coeff_columns_jacobian`` — the EXACT-JACOBIAN route:
  every basis column's ``d coeff[m] / d x``, taken by running the parameter map
  that produces ``coeff`` in reverse.  A GATE, not a product API.

The benchmark model is ``benchmarks/ctpwa_kketa/config_gen.yml`` (the CTPWA
J/psi -> K+ K- eta shape: 34 waves, 60 basis columns).  Its ``same`` section
aliases every ``...m_mass`` / ``...m_width`` to the ``...p_...`` slot, which is
why the aliasing gates below are the ones that matter: a derivation that reads
only the x slots' own names silently reports the second charge sector as fixed.

One property to keep in mind when reading the two answers: the symbolic route
sees a transform as ``(inputs -> outputs)`` as a SET, so when one parameter of
a pair is freed it cannot say which of that transform's outputs reads which
input and keeps all of them moving.  It is therefore CONSERVATIVE — it never
masks a column that moves — while the Jacobian is per-column exact.  The gates
below say exactly where they agree and demonstrate the difference.
"""
import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import create_backend
from tabpwa.param_constraint import FixedValueTransform, Transform

CFG = "benchmarks/ctpwa_kketa/config_gen.yml"
CFG_KM = "tests/config_pwa_gen_km.yml"
CFG_PWA = "tests/config_pwa.yml"

# The paper's ten-parameter scenario (benchmarks/ctpwa_kketa/README.md §4).
FREE_TEN = ["phi1680_mass", "phi1680_width",
            "X1750_mass", "X1750_width",
            "omega2220_mass", "omega2220_width",
            "K3st1780p_mass", "K3st1780p_width",
            "K4st2045p_mass", "K4st2045p_width"]

# The columns those ten move: K3st1780p / K4st2045p drag their EXACT
# charge-conjugate columns along (an alias, not a second free slot).
MOVING = [5, 6, 7, 8, 9, 10, 35, 36, 37, 38, 53, 54, 55, 56]
FREE_WAVES = [2, 3, 4, 19, 20, 29, 30]
N_NNZ = 88
FIXED_NNZ = 67                      # the non-zeros of the 27 fixed waves


def _fit(cfg=CFG):
    f = Fitter(cfg)
    f.apply_constrains()
    return f


def _mass_width(f):
    """Every fitted mass/width name (charge-conjugate aliases are not slots)."""
    return {n: float(f.defaults[n])
            for n in dict.fromkeys(f.var_registry.flat_names)
            if ("_mass" in n or "_width" in n) and n in f.defaults}


def _n_basis(f):
    return len(f.model.build_kernel_config()["den_names"])


def _symbolic_fixed(f):
    return f.model.fixed_coeff_columns(f.cm)


def _jacobian_fixed(f, x):
    """``how="jacobian"`` through the unified API."""
    return f.model.fixed_coeff_columns(f.cm, x, how="jacobian")


def _jacobian_fixed_fd(f, x, **kw):
    """The FD debugging route (``exact=False``), kept as the thin alias."""
    return f.model.fixed_coeff_columns_jacobian(f.cm, x, exact=False, **kw)


def _symbolic_moving(f):
    fixed = set(_symbolic_fixed(f))
    return [i for i in range(_n_basis(f)) if i not in fixed]


def _jacobian_moving(f, x, **kw):
    if kw.get("exact", True) is False:
        fixed = set(_jacobian_fixed_fd(f, x))
    else:
        fixed = set(_jacobian_fixed(f, x))
    return [i for i in range(_n_basis(f)) if i not in fixed]


def _probe_moving(f, x, step=1e-3):
    """Move every free x slot and collect the ``coeff`` columns that changed."""
    c0 = np.asarray(f.build_params(x)[0]["coeff"], np.float64)
    moved = set()
    for i in range(len(x)):
        h = step * max(abs(float(x[i])), 1.0)
        for sgn in (1.0, -1.0):
            xp = np.asarray(x, np.float64).copy()
            xp[i] += sgn * h
            c = np.asarray(f.build_params(xp)[0]["coeff"], np.float64)
            moved.update(int(j) for j in np.nonzero(c != c0)[0])
    return sorted(moved)


class _Fn(Transform):
    """A throwaway transform: *out* is a function of the declared inputs.

    Deliberately NOT a ``FixedValueTransform`` — an empty ``input_names`` is
    the whole definition of "constant", and the derivation must not depend on
    a class.
    """

    def __init__(self, inputs, out, fn, dfn=None):
        self.input_names = list(inputs)
        self.output_names = [out]
        self._fn = fn
        self._dfn = dfn

    def forward(self, d):
        d = dict(d or {})
        return {**d, self.output_names[0]: self._fn(d)}

    def backward(self, grad_out, d_in=None):
        out = self.output_names[0]
        g = grad_out.get(out, 0.0)
        if self._dfn is None:
            return super().backward(grad_out, d_in)          # numeric VJP
        return {n: g * self._dfn(n, dict(d_in or {}))
                for n in self.input_names}


# ── 1. the benchmark scenario: both derivations, three x ────────────────

def test_the_default_is_the_transform_walk():
    f = _fit()
    f.set_fixed({n: v for n, v in _mass_width(f).items() if n not in FREE_TEN})
    default = f.model.fixed_coeff_columns(f.cm)
    explicit = f.model.fixed_coeff_columns(f.cm, how="transform")
    assert default == explicit
    assert len(default) == _n_basis(f) - len(MOVING)
    # and the default needs no x at all
    assert f.model.fixed_coeff_columns(f.cm, x=None) == explicit


def test_how_is_validated_and_jacobian_needs_a_point():
    f = _fit()
    with pytest.raises(TypeError, match="needs a point"):
        f.model.fixed_coeff_columns(f.cm, how="jacobian")
    with pytest.raises(TypeError, match="needs a point"):
        f.model.fixed_coeff_columns_jacobian(f.cm, None)
    with pytest.raises(ValueError, match="(?s)transform.*jacobian"):
        f.model.fixed_coeff_columns(f.cm, how="bogus")
    # a plain-`pwa` model has no basis at all, whatever the derivation
    g = _fit(CFG_PWA)
    with pytest.raises(TypeError, match="pwa_gen"):
        g.model.fixed_coeff_columns(g.cm, how="transform")
    with pytest.raises(TypeError, match="pwa_gen"):
        g.model.fixed_coeff_columns(g.cm, x=np.zeros(3), how="jacobian")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_benchmark_mask_agrees_with_the_exact_jacobian(seed):
    f = _fit()
    f.set_fixed({n: v for n, v in _mass_width(f).items() if n not in FREE_TEN})
    n = _n_basis(f)
    x = f.initial_values(seed=seed)

    symbolic = _symbolic_fixed(f)
    jacobian = _jacobian_fixed(f, x)

    assert symbolic == jacobian, (
        "the symbolic graph and the exact Jacobian disagree on the fixed set")
    assert [i for i in range(n) if i not in jacobian] == MOVING
    assert [i for i in range(n) if i not in symbolic] == MOVING
    assert _probe_moving(f, x) == MOVING
    # the debug (central-difference, two-draw) variant agrees too
    assert _jacobian_fixed_fd(f, x) == symbolic

    # the same statement at wave resolution, which is how a mask-aware kernel
    # partitions its work: 27 fixed waves, 67 of 88 non-zeros
    gamma = np.asarray(f.model.build_kernel_config()["gamma_matrix"])
    moving_any = [k for k in range(gamma.shape[0])
                  if any(gamma[k, 0, c] for c in MOVING)]
    assert moving_any == FREE_WAVES
    nnz = [(int(k), int(c)) for k, c in np.argwhere(gamma[:, 0, :] > 0)]
    assert len(nnz) == N_NNZ
    assert sum(1 for k, _c in nnz if k not in FREE_WAVES) == FIXED_NNZ


# ── 2. alias groups ─────────────────────────────────────────────────────

def test_an_alias_of_a_free_parameter_moves_both_charge_sectors():
    f = _fit()
    f.set_fixed({n: v for n, v in _mass_width(f).items()
                 if n != "K3st1780p_mass"})

    # the charge-conjugate names have no x slot of their own — they are one
    # parameter with the p sector — and still resolve to a free value
    assert f.cm.same_params["K3st1780m_mass"] == "K3st1780p_mass"
    assert f.cm.is_free("K3st1780m_mass")

    x = f.initial_values(seed=1)
    both_sectors = [35, 36, 53, 54]
    assert _symbolic_moving(f) == both_sectors
    assert _jacobian_moving(f, x) == both_sectors
    assert _probe_moving(f, x) == both_sectors


def test_a_width_only_free_moves_the_coupled_column_in_both_sectors():
    f = _fit()
    f.set_fixed({n: v for n, v in _mass_width(f).items()
                 if n != "K3st1780p_width"})
    x = f.initial_values(seed=1)

    measured = _probe_moving(f, x)
    exact = _jacobian_moving(f, x)
    symbolic = _symbolic_moving(f)

    # the alias point, which is what a slot-name derivation gets wrong: the
    # charge-conjugate width column (54) moves even though that sector has no
    # free x slot of its own
    assert measured == [36, 54]
    assert exact == measured
    assert set(exact) <= set(symbolic)

    # The symbolic route is CONSERVATIVE here: ``_s0 = m0²`` and
    # ``_width_t = m0·g`` are both declared outputs of the pair (mass, width),
    # so with one of the pair free it cannot attribute outputs to inputs and
    # keeps BOTH moving (column 35 does not actually move).  Never the unsafe
    # direction — a moving column is never masked — and a config that frees a
    # parameter alone should expect exactly that.
    assert symbolic == [35, 36, 53, 54]


# ── 3. constants, everything fixed, everything free ─────────────────────

def test_one_is_fixed_because_its_producer_has_no_inputs():
    f = _fit()
    names = f.model.build_kernel_config()["den_names"]
    fixed = set(_symbolic_fixed(f))
    assert fixed == set(_jacobian_fixed(f, f.initial_values(seed=1)))

    ones = [i for i, n in enumerate(names) if n == "one"]
    assert ones and set(ones) <= fixed, "a constant basis weight must be fixed"

    # the REASON: a producer with no inputs generates no dependency.  No name
    # and no class is consulted — "one" is just the instance present here.
    producers = [tr for tr in f.cm.mass_width_transforms
                 if "one" in tr.output_names]
    assert producers
    assert all(not tr.input_names for tr in producers)

    # ... which is also true for a constant of any other class
    f.cm.add_transform(_Fn([], "my_constant", lambda d: 7.0))
    assert f.cm.depends_on_free("my_constant") is False
    assert not isinstance(
        [tr for tr in f.cm.custom_transforms
         if "my_constant" in tr.output_names][0], FixedValueTransform)


def test_every_mass_width_fixed_masks_every_column():
    f = _fit()
    f.set_fixed(_mass_width(f))
    n = _n_basis(f)
    assert n > 0
    everything = list(range(n))
    assert _symbolic_fixed(f) == everything
    assert _jacobian_fixed(f, f.initial_values(seed=1)) == everything
    assert _jacobian_fixed(f, f.initial_values(seed=2)) == everything


def test_everything_free_masks_no_shared_constant_row():
    # the kmatrix config has no "one" row, and every column belongs to a
    # transform that declares a free input (its weights transform declares the
    # numerator couplings too), so the symbolic mask is empty — the safe
    # direction, since the exact route only retires a genuinely constant row
    f = _fit(CFG_KM)
    assert _n_basis(f) == 5
    assert _symbolic_fixed(f) == []
    assert set(_probe_moving(f, f.initial_values(seed=1))) <= set(
        _symbolic_moving(f))

    # the benchmark does have a shared constant row: everything free leaves
    # EXACTLY the "one" columns masked in both derivations
    g = _fit()
    names = g.model.build_kernel_config()["den_names"]
    ones = [i for i, n in enumerate(names) if n == "one"]
    for seed in (1, 2):
        x = g.initial_values(seed=seed)
        assert _symbolic_fixed(g) == ones
        assert _jacobian_fixed(g, x) == ones


def test_a_constant_no_transform_produces_is_blind_to_the_transform_walk():
    """The documented blind spot of the default: ``k2_m4``.

    ``kmatrix_n``'s weights transform emits ``{name}_m4`` with the LITERAL
    weight 1, but as one output of a transform whose declared inputs are the
    free masses/widths/couplings.  The transform walk therefore cannot retire
    it; the exact Jacobian sees the zero derivative.
    """
    f = _fit(CFG_KM)
    x = f.initial_values(seed=1)
    m4 = f.model.build_kernel_config()["den_names"].index("k2_m4")

    assert m4 not in _symbolic_fixed(f), (
        "the transform walk cannot retire a constant that is an output of a "
        "transform fed by free parameters")
    assert m4 in _jacobian_fixed(f, x), "the Jacobian sees the zero derivative"
    # the safe direction holds: whatever the walk masks, the Jacobian masks too
    assert set(_symbolic_fixed(f)) <= set(_jacobian_fixed(f, x))


# ── 4. transitive production ────────────────────────────────────────────

def test_a_transform_consuming_another_transforms_output_agrees():
    f = Fitter(CFG)
    # two stages, in order: root_x -> X1750_mass -> (the basis transform)
    # -> coeff.  Registered AHEAD of the model's own transforms, because a
    # producer has to run before its consumer.
    f.cm.set_mass_width_transforms(
        [_Fn(["root_x"], "X1750_mass", lambda d: d["root_x"] ** 2,
             lambda n, d: 2.0 * d["root_x"])]
        + list(f.model.particle_transforms()), reset=True)
    f.apply_constrains()
    assert not f.cm.is_free("X1750_mass")        # now it is produced …
    assert f.cm.is_free("root_x")                # … from a free root

    x = f.initial_values(seed=1)
    derived = _symbolic_moving(f)
    exact = _jacobian_moving(f, x)
    assert {7, 8} <= set(derived), "the leaf's route to x is through the chain"
    assert derived == exact

    # ... and pinning the ROOT ends the chain: the leaf column stops moving
    f.set_fixed({"root_x": 0.9})
    assert f.cm.depends_on_free("X1750_mass") is False
    x = f.initial_values(seed=1)
    assert 7 not in _jacobian_moving(f, x)
    # the symbolic route still keeps 8 moving — its other input (the width) is
    # free — so pin that too, and both derivations agree on the whole set
    f.set_fixed({"X1750_width": float(f.defaults["X1750_width"])})
    x = f.initial_values(seed=1)
    assert 7 not in _symbolic_moving(f)
    assert _symbolic_moving(f) == _jacobian_moving(f, x)


# ── 5. a derivative that vanishes at one x ──────────────────────────────

def test_a_degenerate_point_does_not_make_the_exact_route_miss_a_column():
    """``exact=True`` is exempt from the FD caveat by construction.

    ``k2_poly = −(m0² + m1²)``: at ``m0 = 0`` its PARTIAL w.r.t. ``m0`` is
    exactly zero while the ``m1`` partial is not, so the column still moves.
    The exact route ORs over slots, so it keeps the column moving; a central
    difference on the ``m0`` slot would cancel to zero (second order), which is
    the caveat the two-draw ``exact=False`` variant carries and this one does
    not.  The mass ranges are removed so the point is reachable at all — a
    bound maps ``x`` to the interior, never to zero.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        f = _fit(CFG_KM)
        f.unset_range("k2_mass0")
        f.unset_range("k2_mass1")
        names = list(f.var_registry.flat_names)
        x = f.initial_values(seed=1)
        x[names.index("k2_mass0")] = 0.0

        exact = _jacobian_moving(f, x)
        assert 1 in exact, "the m1 partial still moves k2_poly"
        assert set(exact) <= set(_symbolic_moving(f))

        # the mechanism: the m0 partial IS zero here, the m1 partial is not
        kc = f.model.build_kernel_config()
        tr = f.model.params_transform_cls(f.model, kc)
        res = f.cm.flat_resolve(x)
        unit = np.zeros(len(kc["den_names"]))
        unit[kc["den_names"].index("k2_poly")] = 1.0
        grad = tr.backward({"coeff": unit,
                            "ck": np.zeros_like(tr.forward(res)["ck"])}, res)
        gx = np.asarray(f.cm.full_gradient(grad, res, x), np.float64)
        slots = dict(zip(f.var_registry.flat_names, gx))
        assert slots["k2_mass0"] == 0.0
        assert slots["k2_mass1"] != 0.0


# ── 6. plain `pwa` has no coefficient basis ─────────────────────────────

def test_plain_pwa_points_at_pwa_gen():
    f = _fit(CFG_PWA)
    with pytest.raises(TypeError, match="pwa_gen"):
        f.model.fixed_coeff_columns(f.cm)
    with pytest.raises(TypeError, match="pwa_gen"):
        f.model.fixed_coeff_columns(f.cm, x=np.zeros(3), how="jacobian")
    with pytest.raises(TypeError, match="pwa_gen"):
        f.model.fixed_coeff_columns_jacobian(f.cm, np.zeros(3))


def test_the_fitter_does_not_own_the_derivation():
    """The flat ``coeff`` axis is ``pwa_gen``'s — the Fitter must not know it.

    The model answers for itself, so the mask is asked of the model and handed
    to the backend by the caller; a ``Fitter.fixed_coeff_columns`` would put
    model-specific knowledge back in the orchestrator (class policy).
    """
    assert not hasattr(Fitter, "fixed_coeff_columns")


def test_the_model_derived_mask_drives_the_fixed_coeff_backend():
    """End to end: model method → ``fixed_coeff.set_fixed_cols``."""
    f = _fit()
    f.set_fixed({n: v for n, v in _mass_width(f).items() if n not in FREE_TEN})
    cols = f.model.fixed_coeff_columns(f.cm, x=f.initial_values(seed=1),
                                      how="jacobian")
    assert len(cols) == _n_basis(f) - len(MOVING)

    spec = {"name": "fixed_coeff", "base": "numpy_flat"}
    # the wrapper's mask comes WITH its values (the cache is filled at load,
    # so the coefficients must be part of the declaration); `numpy_flat`
    # ignores both — the mask is what this test checks
    params = f.build_params(f.initial_values(seed=1))[0]
    vals = {int(c): float(np.asarray(params["coeff"])[int(c)]) for c in cols}
    w = create_backend(spec, f.kernel_config, model="pwa_gen")
    try:
        w.set_fixed_cols(cols, vals)
        assert w.mask is not None
        # the installed mask IS the derived one (bit for bit)
        expected = np.zeros(_n_basis(f), np.int8)
        expected[np.asarray(cols, int)] = 1
        assert np.array_equal(np.asarray(w.mask, np.int8), expected)
        # ... and the same mask a direct construction would install
        direct = create_backend(spec, f.kernel_config, model="pwa_gen")
        try:
            direct.set_fixed_cols(cols, vals)
            assert np.array_equal(w.mask, direct.mask)
        finally:
            direct.free()
    finally:
        w.free()


# ── 7. the cache follows the constraint state ───────────────────────────

def test_the_mask_is_rederived_after_a_constraint_changes():
    f = _fit()
    f.set_fixed({n: v for n, v in _mass_width(f).items() if n not in FREE_TEN})
    assert _symbolic_moving(f) == MOVING

    assert 7 in _symbolic_moving(f)          # X1750_s0, moved by X1750_mass
    # the whole transform unit: a derived mask follows the declared
    # (inputs -> outputs) contract, so pinning BOTH inputs is what retires the
    # pair of columns
    f.set_fixed({"X1750_mass": float(f.defaults["X1750_mass"]),
                 "X1750_width": float(f.defaults["X1750_width"])})

    after = _symbolic_moving(f)
    assert after == [c for c in MOVING if c not in (7, 8)]
    x = f.initial_values(seed=1)
    assert _jacobian_moving(f, x) == after
    assert _probe_moving(f, x) == after

"""Flat phase space for ANY topology — ``phasespace.generate_phsp`` (mass tree)
and ``pwa_build.generate_pwa_phsp`` (config adapter).

``phasespace`` is the TFPWA-style entry, ``generate_phsp(m0, mass_tree, n)``: a
child of the tree is a final mass (float) or a PINNED intermediate
``(m, [...])``.  A mass-less nested list is REFUSED rather than read as a
grouping — flat phase space is a flat list of final masses, the grouping does
not enter the measure, and accepting it would only invite the wrong reading.
Nothing here knows about configs.

It used to be ``top -> (sub -> a b) + c`` only, so a four-body or ternary
vertex died in ``np.stack`` with a bare ``KeyError``.  That was a SHAPE limit,
not a model one: a config with ``b: model: skip`` at an internal node failed
identically to one with an ordinary ``BW`` there, because the generator never
reads the particle models at all.

One entry point now dispatches on the shape: the closed-form three-body
sampler for the shape every fixture uses (bit-for-bit unchanged, pinned by a
digest against ``HEAD``), and the general two-body fold otherwise.  The gates:

  A. momentum/energy conservation and the on-shell condition, for 3-, 4- and
     5-body tops and for nested internal nodes;
  B. agreement with the EXISTING three-body generator on the shape both can do
     (same density, not merely conserved momenta);
  C. the flatness of the sampled angles at every vertex;
  D. ``model: skip`` is not special-cased — swapping it for an ordinary BW at
     the same node gives BIT-IDENTICAL events;
  E. the sample flows through ``pwa_event_data_tree`` into a kernel, and the
     ``skip`` node's mass column is never READ (its factor was dropped);
  F. the adapter's own contract: ``weights_out``, ``top_p`` routing, ``batch``
     chunking, and the frozen closed form the dispatch must keep.

Exactness argument (why a two-body fold samples the flat n-body measure):
``dΦ_n(P) = dΦ(M; m_{c1}, m_{c2})·dΦ(m_{c1})·dΦ(m_{c2}) · dm²/2π`` is an
identity, so folding any vertex into two-body splits with isotropic directions
reproduces the same density.  The proposal is UNIFORM ON THE SLACK SIMPLEX, so
the importance weight is exactly ``Π q`` and no window factor appears; the older
nested-``U`` proposal needed ``Π q · Π W`` and is history (see the ``phasespace``
header).  The gates check the law numerically rather than trusting the argument.
"""
import hashlib

import numpy as np
import pytest

from tabpwa.amp_model import build_amplitude_model
from tabpwa.config_loader import RawConfig
from tabpwa.phasespace import two_body_momentum as ps_two_body_momentum
from tabpwa.pwa_build import generate_pwa_phsp

CFG_SKIP = "tests/config_skip.yml"        # 4-body: a -> (b:skip) (c:BW) d
CFG_ALL = "tests/config_skip_all.yml"     # 3-body: a -> (b:skip) d
CFG_PWA_GEN = "tests/config_pwa_gen.yml"  # 3-body, ordinary BW


def _gen(tree, n, **kw):
    """``generate_phsp`` in the ``(m0, children)`` form the tests carry."""
    from tabpwa.phasespace import generate_phsp
    return generate_phsp(tree[0], tree[1], n, **kw)


def _model(path):
    return build_amplitude_model(path)


def _chain(model):
    return model.full_decay.chains[0]


# ── A. conservation and on-shell, for several topologies ──────────────

@pytest.mark.parametrize("path", [CFG_ALL, CFG_SKIP, CFG_PWA_GEN])
def test_conservation_and_on_shell(path):
    """Σp = 0, ΣE = M_top, and every final is on its own mass shell."""
    m = _model(path)
    ch = _chain(m)
    mom = generate_pwa_phsp(m, ch, 400, seed=1)
    assert mom.shape == (400, len(m.finals), 4)

    tot = mom.sum(axis=1)
    assert np.abs(tot[:, 1:]).max() < 1e-9, "momentum not conserved"
    M = float(m.dic["particle"][m.top]["mass"])
    assert np.abs(tot[:, 0] - M).max() < 1e-9, "energy not conserved"

    masses = {o.name: float(o.mass) for d in ch.decays for o in d.outs
              if hasattr(o, "mass")}
    for j, name in enumerate(m.finals):
        p = mom[:, j]
        m2 = p[:, 0] ** 2 - np.sum(p[:, 1:] ** 2, axis=-1)
        assert np.abs(m2 - masses[name] ** 2).max() < 1e-9, name
        assert (p[:, 0] > 0).all(), f"{name}: negative energy"


def test_a_five_daughter_top_and_a_depth_three_chain():
    """A 5-way vertex and a depth-3 chain: shapes the old code cannot express.

    ``a -> (b -> e f) (c -> (j -> g h) i) d`` — five top daughters, and one
    branch two levels deep.  The invariant-mass chain has to stay consistent
    across both, which is where a scalar/array mix shows up.
    """
    dic = RawConfig(CFG_SKIP).to_dict()
    dic["particle"]["a"] = {"J": 1, "P": -1, "spins": [-1, 1], "mass": 4.2}
    # a -> (b -> e f) (c -> (j -> g h) i) d: five top daughters, depth 3
    dic["decay"]["a"] = [["b", "c", "d"], ["c", "j", "i"]]
    dic["decay"]["c"] = ["j", "i"]
    dic["decay"]["j"] = ["g", "h"]
    del dic["decay"]["b"]
    dic["decay"]["b"] = ["e", "f"]
    dic["particle"]["c"] = {"J": 0, "P": 1, "mass": 1.6}
    dic["particle"]["j"] = {"J": 0, "P": 1, "mass": 1.2}
    dic["particle"]["i"] = {"J": 0, "P": -1, "mass": 0.2}
    dic["particle"]["$finals"] = ["e", "f", "g", "h", "i", "d"]
    dic["data"] = {"dat_order": ["e", "f", "g", "h", "i", "d"]}
    m = build_amplitude_model(dic)
    ch = _chain(m)
    mom = generate_pwa_phsp(m, ch, 200, seed=5)
    tot = mom.sum(axis=1)
    assert np.abs(tot[:, 1:]).max() < 1e-9, "momentum not conserved"
    assert np.abs(tot[:, 0] - 4.2).max() < 1e-9, "energy not conserved"


# ── B. the shape both generators can do: same DENSITY ─────────────────

def test_matches_the_three_body_generator():
    """The GENERAL path reproduces the closed form on the shape both can do.

    This is the density gate for the new code: forcing the fold on a three-body
    chain must give the same pair-mass spectra as the closed-form sampler the
    framework has used for years (KS per pair mass — a wrong mass weight or a
    missing proposal Jacobian moves these).
    """
    from tabpwa.pwa_build import _generate_general_phsp, \
        _generate_three_body_phsp
    from scipy.stats import ks_2samp

    m = _model(CFG_PWA_GEN)
    ch = _chain(m)
    new = _generate_general_phsp(m, ch, 20000, seed=11)
    old = _generate_three_body_phsp(m, ch, 20000, seed=11)

    fi = {k: v for v, k in enumerate(m.finals)}
    for i in range(len(m.finals)):
        for j in range(i + 1, len(m.finals)):
            def pair(mom, a, b):
                s = mom[:, fi[a]] + mom[:, fi[b]]
                return np.sqrt(np.maximum(
                    s[:, 0] ** 2 - np.sum(s[:, 1:] ** 2, axis=-1), 0.0))
            a = pair(new, m.finals[i], m.finals[j])
            b = pair(old, m.finals[i], m.finals[j])
            assert ks_2samp(a, b).pvalue > 1e-3, (m.finals[i], m.finals[j])

    cos = new[:, 0, 3] / np.linalg.norm(new[:, 0, 1:], axis=-1)
    assert abs(cos.mean()) < 0.03 and abs(cos.std() - 1 / np.sqrt(3)) < 0.03


def test_four_identical_masses_are_symmetric():
    """REFERENCE-FREE density gate: interchangeable pairs must coincide.

    With ``a -> e f g h`` and all four masses equal, ``m(ef)`` and ``m(gh)``
    are the same random variable, so their laws must agree.  A wrong proposal
    Jacobian or a wrong weight moves this: the OLD nested-``U`` proposal without
    its ``Π W`` correction biased the first pair heavy (measured 1.38 vs 1.29,
    KS p = 0), which is what this reference-free gate exists to catch.
    """
    from scipy.stats import ks_2samp

    mu = 0.4937
    dic = {"amp_model": "pwa", "decay": {"a": [["e", "f", "g", "h"]]},
           "particle": {"$top": "a", "$finals": ["e", "f", "g", "h"],
                        "a": {"J": 1, "P": -1, "spins": [-1, 1], "mass": 3.0},
                        **{n: {"J": 0, "P": -1, "mass": mu} for n in "efgh"}}}
    m = build_amplitude_model(dic)
    mom = generate_pwa_phsp(m, _chain(m), 40000, seed=7)
    fi = {k: v for v, k in enumerate(m.finals)}

    def pair(a, b):
        s = mom[:, fi[a]] + mom[:, fi[b]]
        return np.sqrt(np.maximum(s[:, 0] ** 2 - np.sum(s[:, 1:] ** 2,
                                                        axis=-1), 0.0))

    for x, y in (("e f", "g h"), ("e g", "f h"), ("e f", "e g")):
        a = pair(*x.split())
        b = pair(*y.split())
        assert ks_2samp(a, b).pvalue > 1e-3, (x, y, a.mean(), b.mean())
        assert abs(a.mean() - b.mean()) < 0.01, (x, y)


def test_the_fold_order_does_not_matter():
    """Writing the daughters in another order is the SAME final state.

    A phase-space density cannot depend on how the decay is grouped, so
    ``a: [b,c,d]`` and ``a: [c,b,d]`` must agree.  This is the gate that
    catches "the last daughter is pinned" (its mass would become a delta).
    """
    from scipy.stats import ks_2samp

    dic = RawConfig(CFG_SKIP).to_dict()
    dic["decay"]["a"] = [["b", "c", "d"], ["b", "e", "f"], ["c", "g", "h"]]
    dic2 = RawConfig(CFG_SKIP).to_dict()
    dic2["decay"]["a"] = [["c", "b", "d"], ["b", "e", "f"], ["c", "g", "h"]]

    a = build_amplitude_model(dic)
    b = build_amplitude_model(dic2)
    mom_a = generate_pwa_phsp(a, _chain(a), 20000, seed=3)
    mom_b = generate_pwa_phsp(b, _chain(b), 20000, seed=3)

    def pair(mom, model):
        fi = {k: v for v, k in enumerate(model.finals)}
        s = mom[:, fi["e"]] + mom[:, fi["f"]]
        return np.sqrt(np.maximum(s[:, 0] ** 2 - np.sum(s[:, 1:] ** 2,
                                                        axis=-1), 0.0))

    pa, pb = pair(mom_a, a), pair(mom_b, b)
    assert ks_2samp(pa, pb).pvalue > 1e-3, (pa.mean(), pb.mean())
    assert abs(pa.mean() - pb.mean()) < 0.01


# ── D. `skip` is not special-cased ────────────────────────────────────

def test_a_node_model_does_not_enter_the_phase_space():
    """``skip`` vs an ordinary BW at the SAME node: bit-identical events.

    This is the point of the request: a ``skip`` node needs no code path.  The
    generator reads topology + final masses only, so the two configs must
    produce the same momenta from the same seed.
    """
    a = build_amplitude_model(RawConfig(CFG_SKIP).to_dict())
    dic = RawConfig(CFG_SKIP).to_dict()
    dic["particle"]["b"] = {"J": 0, "P": 1, "mass": 1.0, "width": 0.1}
    b = build_amplitude_model(dic)

    ma = generate_pwa_phsp(a, _chain(a), 400, seed=7)
    mb = generate_pwa_phsp(b, _chain(b), 400, seed=7)
    assert np.array_equal(ma, mb)

    # ... and the same for the three-body skip config
    c = build_amplitude_model(RawConfig(CFG_ALL).to_dict())
    dic2 = RawConfig(CFG_ALL).to_dict()
    dic2["particle"]["b"] = {"J": 0, "P": 1, "mass": 1.0, "width": 0.1}
    d = build_amplitude_model(dic2)
    assert np.array_equal(generate_pwa_phsp(c, _chain(c), 200, seed=3),
                          generate_pwa_phsp(d, _chain(d), 200, seed=3))


# ── E. through the event builder into a kernel ────────────────────────

def test_the_sample_flows_through_the_event_builder_into_a_kernel():
    """``generate_pwa_phsp`` → ``pwa_event_data_tree`` → a real compute.

    The ``skip`` node's mass column is never READ (it was dropped from the
    factor axis), which is why the generated mass row for it is harmless.
    """
    from tabpwa import Fitter
    from tabpwa.pwa_build import pwa_event_data_tree

    d = RawConfig(CFG_SKIP).to_dict()
    d["amp_model"] = "pwa_gen"
    f = Fitter(d, backend="numpy_flat")
    f.apply_constrains()
    ch = f.model.full_decay.chains[0]
    mom = generate_pwa_phsp(f.model, ch, 300, seed=3)
    kc = f.kernel_config

    data = pwa_event_data_tree(f.model, kc, {ch.topo_id: ch}, mom)
    assert np.asarray(data["mass"]).shape[0] == 300

    params, _ = f.build_params(f.initial_values(seed=1))
    Q, grads, P = f.backend.compute(params, f.backend.load_data(data),
                                    norm=None, return_p=True)
    assert np.isfinite(Q) and np.isfinite(P).all() and (P >= 0).all()
    assert P.sum() > 0

    # the dropped factor's mass column is never READ: put a wild value in every
    # column no factor references and the compute must be bit-identical
    used = set(np.unique(np.asarray(kc["num_m_index"])).tolist())
    dropped = [c for c in range(np.asarray(data["mass"]).shape[1])
               if c not in used]
    assert dropped, "the drop left no unread column to probe"
    wild = {k: np.array(v, copy=True) for k, v in data.items()}
    for c in dropped:
        wild["mass"][:, c] = 1e3
    Q2, g2, P2 = f.backend.compute(params, f.backend.load_data(wild),
                                   norm=None, return_p=True)
    assert np.array_equal(P2, P) and np.array_equal(g2["coeff"], grads["coeff"])
    assert Q2 == Q


# ── the mass-tree API (TFPWA's shape) ────────────────────────────────

def test_mass_tree_api_basic_shapes():
    """The tree alone is a valid input — no config, no amplitudes."""
    from tabpwa.phasespace import generate_phsp

    cases = {
        "3 flat":        (3.0, [0.139, 0.139, 0.547]),
        "pinned 2+1":    (3.0, [(0.9, [0.139, 0.139]), 0.547]),
        "5 flat":        (3.0, [0.139, 0.139, 0.493, 0.493, 0.547]),
        "7 flat":        (3.0, [0.139, 0.139, 0.493, 0.493, 0.547, 0.2, 0.3]),
        "pinned nested": (4.2, [(1.2, [0.139, 0.139]), 0.493, 0.493, 0.547]),
    }
    for label, tree in cases.items():
        mom = _gen(tree, 1500, seed=1)
        assert isinstance(mom, list), label
        tot = sum(mom)
        assert np.abs(tot[:, 1:]).max() < 1e-9, label
        assert np.abs(tot[:, 0] - tree[0]).max() < 1e-9, label
        assert all(np.isfinite(m).all() for m in mom), label


def test_mass_tree_pinned_vs_flat():
    """``(m, [...])`` pins the invariant mass; a flat list integrates it."""
    from tabpwa.phasespace import generate_phsp

    def sub_mass(tree):
        mom = _gen(tree, 3000, seed=2)
        s = mom[0] + mom[1]
        return np.sqrt(np.maximum(s[:, 0] ** 2 - np.sum(s[:, 1:] ** 2,
                                                        axis=-1), 0.0))

    pinned = sub_mass((3.0, [(0.9, [0.139, 0.139]), 0.547]))
    assert pinned.std() < 1e-9 and abs(pinned.mean() - 0.9) < 1e-9

    flat = sub_mass((3.0, [0.139, 0.139, 0.547]))
    assert flat.std() > 0.1             # a real distribution, not a delta


def test_mass_tree_leaf_order_is_the_listed_order():
    """The returned list follows the tree, so the masses line up positionally."""
    from tabpwa.phasespace import generate_phsp

    tree = (3.0, [0.139, 0.139, 0.493, 0.493, 0.547])
    mom = _gen(tree, 50, seed=1)
    assert len(mom) == 5
    for arr, want in zip(mom, (0.139, 0.139, 0.493, 0.493, 0.547)):
        m2 = arr[:, 0] ** 2 - np.sum(arr[:, 1:] ** 2, axis=-1)
        assert np.abs(m2 - want ** 2).max() < 1e-9


def test_mass_tree_refusals():
    from tabpwa.phasespace import generate_phsp

    with pytest.raises(ValueError, match="m0"):
        generate_phsp(0.0, [0.0, 0.0], 10, seed=1)      # massless top
    with pytest.raises(ValueError, match="no phase space"):
        generate_phsp(1.0, [0.6, 0.6], 10, seed=1)      # below threshold
    with pytest.raises(ValueError, match=">= 2 daughters"):
        generate_phsp(3.0, [0.139], 10, seed=1)         # one daughter
    with pytest.raises(ValueError, match=">= 2 daughters"):
        generate_phsp(3.0, 0.139, 10, seed=1)           # children not a list
    with pytest.raises(ValueError, match="must carry its mass"):
        generate_phsp(3.0, [[0.139, 0.139], 0.547], 10, seed=1)  # no mass


def test_a_pinned_chain_boosts_level_by_level():
    """Depth 3 must compose the boosts, not apply a single one.

    ``a -> (b -> (c -> e f) g) h`` with ``c`` and ``b`` pinned.  Each level is
    generated in ITS OWN rest frame and then boosted by that daughter's
    velocity in the frame it was produced in (``_node_momenta`` recurses, so
    the composition is structural: no function ever holds the top frame while
    generating a sub-decay).

    What this pins: every pinned mass is an exact delta at its own level, the
    innermost momentum matches the analytic two-body value, and the innermost
    direction is isotropic in its parent's rest frame and one level up.  (A
    single-boost variant differs from the composition only by a Wigner
    rotation, which leaves these marginals unchanged — so this gates the
    invariants, which a wrong composition does break.)
    """
    from tabpwa.phasespace import boost, two_body_momentum, generate_phsp

    tree = (3.0, [(1.5, [(1.0, [0.2, 0.3]), 0.4]), 0.547])
    mom = _gen(tree, 20000, seed=1)
    e, f, g, h = mom
    tot = e + f + g + h
    assert np.abs(tot[:, 1:]).max() < 1e-9
    assert np.abs(tot[:, 0] - 3.0).max() < 1e-9

    def mass(x):
        return np.sqrt(np.maximum(x[:, 0] ** 2 - np.sum(x[:, 1:] ** 2,
                                                        axis=-1), 0.0))

    c = e + f            # pinned at 1.0
    b = c + g            # pinned at 1.5
    assert abs(mass(c).mean() - 1.0) < 1e-9 and mass(c).std() < 1e-9
    assert abs(mass(b).mean() - 1.5) < 1e-9 and mass(b).std() < 1e-9

    # in b's rest frame e is isotropic, with the analytic two-body momentum
    beta = b[:, 1:] / b[:, 0:1]
    e_b = boost(e, -beta)
    cos = e_b[:, 3] / np.linalg.norm(e_b[:, 1:], axis=-1)
    assert abs(cos.mean()) < 0.02 and abs(cos.std() - 1 / np.sqrt(3)) < 0.02
    p_analytic = two_body_momentum(np.full(len(e), 1.0), 0.2, 0.3)
    p_here = np.linalg.norm(boost(e, -(c[:, 1:] / c[:, 0:1]))[:, 1:], axis=-1)
    assert np.abs(p_here - p_analytic).max() < 1e-9


def _tree_as_chain(tree):
    """``(m0, [children])`` -> a chain object the angle builder accepts."""
    import types

    decays = []

    def add(kids, name):
        outs = []
        for i, ch in enumerate(kids):
            nm = f"{name}{i}"
            outs.append((nm,
                         float(ch[0]) if isinstance(ch, (tuple, list)) else float(ch),
                         list(ch[1]) if isinstance(ch, (tuple, list)) else None))
        decays.append((name, outs))
        for nm, _m, sub in outs:
            if sub is not None:
                add(sub, nm)

    add(list(tree[1]), "t")
    parsed = [types.SimpleNamespace(
        core=types.SimpleNamespace(name=nm),
        outs=[types.SimpleNamespace(name=o, mass=m) for o, m, _ in kids])
        for nm, kids in decays]
    return types.SimpleNamespace(decays=parsed, top="t")


@pytest.mark.parametrize("tree", [
    (3.0, [0.139, 0.139, 0.547]),                        # 1 vertex
    (3.0, [(1.5, [(1.0, [0.2, 0.3]), 0.4]), 0.547]),     # 3 vertices
])
def test_the_angles_are_flat_at_every_vertex(tree):
    """Momenta -> angles, through the FRAMEWORK's own reconstruction.

    This is the end-to-end shape of the flatness claim: `decay_angles_vectorized`
    boosts down the tree level by level, so a generation whose sub-decays were
    not in their own rest frames would show up as a non-flat `cos(theta)` or an
    azimuthal bias.  Every vertex must be isotropic: `cos(theta)` uniform on
    [-1, 1] and `phi` uniform on [0, 2pi).

    (It does NOT discriminate a *single-boost* generation from the composed one:
    those differ by a Wigner rotation, which re-orients an already isotropic
    configuration and so leaves every marginal flat.  The composition itself is
    structural — `_node_momenta` only ever knows its own parent's frame — and
    the invariants are gated by `test_a_pinned_chain_boosts_level_by_level`.)
    """
    from tabpwa.momenta_to_angles import decay_angles_vectorized
    from tabpwa.phasespace import generate_phsp

    ch = _tree_as_chain(tree)
    mom = _gen(tree, 20000, seed=1)
    phi, th = decay_angles_vectorized(ch, mom)
    assert phi.shape == (20000, len(ch.decays))

    for v in range(phi.shape[1]):
        c = np.cos(th[:, v])
        p = phi[:, v]
        name = ch.decays[v].core.name
        assert abs(c.mean()) < 0.02, (name, c.mean())
        assert abs(c.std() - 1 / np.sqrt(3)) < 0.02, (name, c.std())
        assert abs(np.sin(p).mean()) < 0.02, (name, "phi")
        assert abs(p.std() - np.pi / np.sqrt(3)) < 0.03, (name, p.std())


# ── a moving top (`top_p`) ────────────────────────────────────

def test_a_boosted_event_gives_flat_angles_in_its_own_cm():
    """After the per-event CM boost the decay angles are flat again."""
    from tabpwa.momenta_to_angles import decay_angles_vectorized
    from tabpwa.phasespace import boost

    tree = (3.0969, [0.139, 0.139, 0.547])
    ch = _tree_as_chain(tree)

    def beam(n, rng):
        return rng.normal(scale=0.004, size=(n, 3))

    mom = _gen(tree, 20000, seed=3, top_p=beam)
    tot = sum(mom)
    beta = tot[:, 1:] / tot[:, 0:1]
    cm = [boost(m, -beta) for m in mom]
    phi, th = decay_angles_vectorized(ch, cm)
    c = np.cos(th[:, 0])
    assert abs(c.mean()) < 0.02 and abs(c.std() - 1 / np.sqrt(3)) < 0.02


def test_fix_beam_is_the_fixed_momentum_callable():
    """``fix_beam(px, py, pz)`` — the top's MOMENTUM, in ``top_p``'s form.

    Momentum rather than a velocity because it is the readable quantity: the
    event then sums to exactly the momentum given, and to
    ``E = sqrt(p^2 + m0^2)`` — both stated at the call site, nothing to invert.
    ``top_p`` is a per-event callable, so a fixed momentum is one too:
    the same value for every event, no draw.
    """
    from tabpwa.phasespace import generate_phsp, fix_beam

    b = fix_beam(0.034, 0, 0)
    assert callable(b)
    out = b(1000, None)
    assert out.shape == (1000, 3)
    assert np.ptp(out, axis=0).max() == 0.0        # the same value everywhere
    assert out[0, 1] == 0.0 and out[0, 2] == 0.0
    assert b(0, None).shape == (0, 3)              # n_events = 0 is fine
    assert np.array_equal(fix_beam(0.034)(7, None)[:, 0], np.full(7, 0.034))

    m0, p = 3.097, np.array([0.034, 0.0, 0.0])
    mom = generate_phsp(m0, [0.139, 0.139, 0.547], 2000, seed=1,
                        top_p=fix_beam(*p))
    tot = sum(mom)
    assert np.abs(tot[:, 1:] - p).max() < 1e-12              # Σp IS the input
    assert np.abs(tot[:, 0] - np.sqrt(p @ p + m0 * m0)).max() < 1e-12   # ΣE
    m2 = tot[:, 0] ** 2 - np.sum(tot[:, 1:] ** 2, axis=-1)
    assert np.abs(np.sqrt(m2) - m0).max() < 1e-9

    for bad in ((np.inf, 0.0, 0.0), (np.nan, 0.0, 0.0)):
        with pytest.raises(ValueError, match="fix_beam"):
            fix_beam(*bad)


def test_a_general_momentum_direction():
    """Any direction: Σp = p, ΣE = sqrt(p²+m0²), and the inverse is exact."""
    from tabpwa.phasespace import boost, generate_phsp, fix_beam

    m0, p = 3.0, np.array([0.03, -0.06, 0.09])
    mom = generate_phsp(m0, [0.139, 0.139, 0.547], 2000, seed=1,
                        top_p=fix_beam(*p))
    tot = sum(mom)
    assert np.abs(tot[:, 1:] - p).max() < 1e-12
    assert np.abs(tot[:, 0] - np.sqrt(p @ p + m0 * m0)).max() < 1e-12
    m2 = tot[:, 0] ** 2 - np.sum(tot[:, 1:] ** 2, axis=-1)
    assert np.abs(np.sqrt(m2) - m0).max() < 1e-9

    back = boost(tot, -p / tot[:, 0:1])            # beta = p/E, undone
    assert np.abs(back[:, 1:]).max() < 1e-9
    assert np.abs(back[:, 0] - m0).max() < 1e-9


def test_a_callable_momentum_varies_per_event():
    """When the top's momentum varies, that is simply another callable."""
    from tabpwa.phasespace import generate_phsp

    def spread(n, rng):
        return np.stack([rng.normal(0.034, 1e-3, n), np.zeros(n), np.zeros(n)],
                        axis=1)

    mom = generate_phsp(3.097, [0.139, 0.139, 0.547], 4000, seed=2,
                        top_p=spread)
    tot = sum(mom)
    assert tot[:, 1].std() > 1e-4
    assert abs(tot[:, 1].mean() - 0.034) < 1e-4
    assert np.abs(tot[:, 2:]).max() < 1e-12        # py = pz = 0
    m2 = tot[:, 0] ** 2 - np.sum(tot[:, 1:] ** 2, axis=-1)
    assert np.abs(np.sqrt(m2) - 3.097).max() < 1e-9


@pytest.mark.parametrize("bad, match", [
    ((0.034, 0, 0), "fix_beam"),                   # a vector is not a callable
    (0.034, "fix_beam"),                           # nor a scalar
    (lambda n, rng: np.zeros((n, 2)), "must give"),
    (lambda n, rng: np.full((n, 3), np.nan), "must be finite"),
])
def test_top_p_refusals(bad, match):
    from tabpwa.phasespace import generate_phsp

    with pytest.raises(ValueError, match=match):
        generate_phsp(3.0969, [0.139, 0.139, 0.547], 100, seed=1,
                      top_p=bad)


# ── massless particles ───────────────────────────────────────────────

_MASS0 = {"J": 0, "P": -1, "mass": 0.0}


def test_massless_finals_both_paths():
    """``m = 0`` is legal: ``E = |p|``, on-shell, conserved, finite.

    A massless parent has no phase space, and ``q = sqrt(λ)/(2M)`` is ``0/0``
    there.  That single ``nan`` used to poison the three-body grid (its first
    node is ``s_min = 0`` when both sub-daughters are massless) and made the
    sampler REFUSE a physical config.  ``_two_body_p`` now returns 0 for
    ``M == 0``, which is the physically right value.
    """
    top = {"J": 1, "P": -1, "spins": [-1, 1], "mass": 3.0}
    dic3 = {"amp_model": "pwa", "decay": {"a": ["b", "d"], "b": ["e", "f"]},
            "particle": {"$top": "a", "$finals": ["e", "f", "d"], "a": top,
                         "b": {"J": 1, "P": -1, "mass": 0.5, "width": 0.1},
                         "d": {"J": 0, "P": -1, "mass": 0.547},
                         "e": _MASS0, "f": _MASS0}}
    dic4 = {"amp_model": "pwa",
            "decay": {"a": ["b", "c"], "b": ["e", "f"], "c": ["g", "h"]},
            "particle": {"$top": "a", "$finals": ["e", "f", "g", "h"],
                         "a": top, "b": {"J": 0, "P": 1, "mass": 1.0},
                         "c": {"J": 0, "P": 1, "mass": 1.0},
                         "e": _MASS0, "f": _MASS0, "g": _MASS0, "h": _MASS0}}

    for dic, label in ((dic3, "3-body closed form"), (dic4, "4-body general")):
        m = build_amplitude_model(dic)
        mom = generate_pwa_phsp(m, _chain(m), 2000, seed=1)
        assert np.isfinite(mom).all(), label
        tot = mom.sum(axis=1)
        assert np.abs(tot[:, 1:]).max() < 1e-9, label
        assert np.abs(tot[:, 0] - 3.0).max() < 1e-9, label
        # every final has its declared mass -- 0 for the massless ones
        for j, name in enumerate(m.finals):
            want = float(m.dic["particle"][name].get("mass", 0.0))
            m2 = mom[:, j, 0] ** 2 - np.sum(mom[:, j, 1:] ** 2, axis=-1)
            assert np.abs(m2 - want ** 2).max() < 1e-9, (label, name)
            assert (mom[:, j, 0] > 0).all(), (label, name, "E must be > 0")


def test_massless_density_is_still_flat():
    """``m = 0`` does not change the measure: general == closed == analytic.

    With ``a -> (b -> e f) + d`` and ``m_e = m_f = 0`` the sub-system mass
    follows ``q_top(M; s, m_d) · s/2`` (since ``q_sub(s; 0, 0) = s/2``).
    """
    from scipy.stats import ks_2samp
    import tabpwa.pwa_build as pb

    dic = {"amp_model": "pwa", "decay": {"a": ["b", "d"], "b": ["e", "f"]},
           "particle": {"$top": "a", "$finals": ["e", "f", "d"],
                        "a": {"J": 1, "P": -1, "spins": [-1, 1], "mass": 3.0},
                        "b": {"J": 1, "P": -1, "mass": 0.5, "width": 0.1},
                        "d": {"J": 0, "P": -1, "mass": 0.547},
                        "e": _MASS0, "f": _MASS0}}
    m = build_amplitude_model(dic)
    ch = _chain(m)
    closed = pb._generate_three_body_phsp(m, ch, 20000, seed=1)
    general = pb._generate_general_phsp(m, ch, 20000, seed=2)

    fi = {k: v for v, k in enumerate(m.finals)}

    def msub(mom):
        s = mom[:, fi["e"]] + mom[:, fi["f"]]
        return np.sqrt(np.maximum(s[:, 0] ** 2 - np.sum(s[:, 1:] ** 2,
                                                        axis=-1), 0.0))

    a, b = msub(closed), msub(general)
    assert ks_2samp(a, b).pvalue > 1e-3, (a.mean(), b.mean())

    M, m_d = 3.0, 0.547
    grid = np.linspace(1e-6, M - m_d, 4000)
    w = pb._two_body_p(M, grid, m_d) * grid / 2
    w = w / np.trapezoid(w, grid)
    analytic = float(np.trapezoid(grid * w, grid))
    assert abs(b.mean() - analytic) < 0.03, (b.mean(), analytic)


def test_a_massless_top_is_refused():
    """A massless parent has no phase space: refuse, do not hand back nan."""
    dic = {"amp_model": "pwa", "decay": {"a": ["e", "f"]},
           "particle": {"$top": "a", "$finals": ["e", "f"],
                        "a": {"J": 1, "P": -1, "spins": [-1, 1], "mass": 0.0},
                        "e": _MASS0, "f": _MASS0}}
    m = build_amplitude_model(dic)
    with pytest.raises(ValueError, match="no phase space"):
        generate_pwa_phsp(m, _chain(m), 10, seed=1)


def test_each_vertex_envelope_bounds_its_own_weight():
    """Every vertex accepts on its OWN envelope, so each must bound its weight.

    A ceiling below the true maximum CAPS that vertex's masses (bias); a loose
    one only costs acceptance.  The naive ``Π(per-factor maximum)`` is exact but
    200x (5 daughters) to 35 000x (7) above the real maximum, so the ceiling is
    a nested maximum over the fold instead — checked here per vertex, since the
    acceptance is per vertex.
    """
    import tabpwa.phasespace as ps

    cases = [
        ("3 flat", (3.0, [0.139, 0.139, 0.547])),
        ("4 flat", (3.0, [0.139, 0.139, 0.493, 0.493])),
        ("7 flat", (3.0, [0.139, 0.139, 0.493, 0.493, 0.547, 0.2, 0.3])),
        ("chain", (4.0, [(1.5, [0.139, 0.139, 0.3]),
                         (1.2, [0.493, 0.493, 0.2])])),
    ]
    for label, (m0, kids) in cases:
        top, dmap, fixed, leaves = ps._parse_tree(m0, kids)
        space = ps.PhaseSpace(m0, kids)
        for node in dmap:                       # every vertex of the tree
            plan = space._plan[node]
            env = plan.ceiling
            _steps, w = ps._vertex_masses(plan, np.random.RandomState(0),
                                          200_000)
            assert w.max() <= env, (label, node, w.max(), env)
            assert env <= 4.0 * w.max(), (label, node, env, w.max())


def test_a_compiled_topology_does_not_resolve(monkeypatch):
    """``PhaseSpace`` parses and solves ONCE — ``generate`` calls neither again.

    The fixed cost is real (MEASURED 0.2 / 5.7 / 9.6 / 13.7 ms for 3 / 5 / 7 / 9
    daughters, up to ~56 % of a 400-event call at n >= 5), so the envelope must
    not be recomputed per call.  Same seed still gives the same events, and
    ``rng=`` continues one stream.
    """
    import tabpwa.phasespace as ps

    calls = {"n": 0}
    real = ps._fold_envelope

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ps, "_fold_envelope", counted)
    space = ps.PhaseSpace(3.0, [0.139, 0.139, 0.493, 0.493])
    solved = calls["n"]
    assert solved >= 1                       # one per vertex, at compile time
    a = space.generate(300, seed=1)
    b = space.generate(300, seed=1)
    c = space.generate(300, seed=2)
    assert calls["n"] == solved              # generate solved nothing
    assert all(np.array_equal(x, y) for x, y in zip(a, b))      # same seed
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))  # a real draw

    rng = np.random.RandomState(5)
    d = space.generate(100, rng=rng)
    e = space.generate(100, rng=rng)         # the same stream continues
    assert not all(np.array_equal(x, y) for x, y in zip(d, e))
    with pytest.raises(ValueError, match="not both"):
        space.generate(10, seed=1, rng=np.random.RandomState(0))


def test_a_pinned_sub_vertex_reproduces_its_standalone_law():
    """Per-vertex acceptance, end to end: the sub-decay knows nothing else.

    4.0 -> (1.2 -> 0.139 0.139 0.3) + 0.547: boost the sub-triplet back to its
    own frame and its pair mass must follow the SAME law as generating that
    vertex alone — the only coupling between vertices is the boost.
    """
    from scipy.stats import ks_2samp

    from tabpwa.phasespace import boost, generate_phsp

    n = 20000
    alone = generate_phsp(1.2, [0.139, 0.139, 0.3], n, seed=11)
    chain = generate_phsp(4.0, [(1.2, [0.139, 0.139, 0.3]), 0.547], n, seed=7)
    tot = sum(chain)
    beta = tot[:, 1:] / tot[:, 0:1]
    sub = [boost(m, -beta) for m in chain[:3]]

    def pair(m3):
        p = m3[0] + m3[1]
        return np.sqrt(np.maximum(p[:, 0] ** 2 - np.sum(p[:, 1:] ** 2, axis=-1),
                                  0.0))

    a, b = pair(alone), pair(sub)
    assert abs(a.mean() - b.mean()) < 0.01 * a.mean()
    assert ks_2samp(a, b).pvalue > 1e-3


def test_zero_events_and_describe_on_the_compiled_api():
    from tabpwa.phasespace import PhaseSpace

    space = PhaseSpace(3.0, [0.139, 0.139, 0.493, 0.493, 0.547])
    with pytest.raises(ValueError, match="not both"):
        space.generate(1, seed=1, rng=np.random.RandomState(0))
    mom, w = space.generate(0, weights_out=True)
    assert w.shape == (0,) and (w == 1.0).all()
    assert [m.shape for m in mom] == [(0, 4)] * 5
    text = space.describe()
    assert "finals" in text and "ceiling" in text


def _simplex_logw(plan, m0):
    """``log w`` and its gradient on the slack simplex (softmax coordinates).

    The primitive gradient is per MASS — each cluster sits in exactly two q's::

        ∂log w/∂S_k = q'_right(k)/q_k + q'_parent(k+1)/q_{k+1}

    and the simplex only adds one prefix sum on top of it.
    """
    md = np.array(plan.md)
    n = len(md)
    suf = np.array(plan.suffix, float)
    slack = plan.slack

    def parts(z):
        e = np.exp(np.append(z, 0.0) - np.max(np.append(z, 0.0)))
        return slack * e / e.sum()

    def value_grad(z):
        c = parts(z)
        above = np.cumsum(c[::-1])[::-1]
        p, logw, qs, dA, dR = m0, 0.0, [], [], []
        for k in range(n - 1):
            Sk = md[n - 1] if k == n - 2 else suf[k + 1] + above[k + 1]
            qk = float(ps_two_body_momentum(p, md[k], Sk))
            lam = (p * p - (md[k] + Sk) ** 2) * (p * p - (md[k] - Sk) ** 2)
            r = np.sqrt(max(lam, 1e-300))
            qs.append(max(qk, 1e-300))
            dA.append((p * p - md[k] ** 2 - Sk ** 2) / r - qk / p)   # the -q/A term
            dR.append(Sk * (Sk ** 2 - p * p - md[k] ** 2) / (p * r))
            logw += np.log(max(qk, 1e-300))
            p = Sk
        dS = np.array([dR[k] / qs[k] + dA[k + 1] / qs[k + 1]
                       for k in range(n - 2)])
        g_c = np.zeros(n - 1)
        g_c[1:] = np.cumsum(dS)
        return logw, c[:len(z)] * (g_c[:len(z)] - (g_c @ c) / slack)
    return value_grad


@pytest.mark.parametrize("kids", [
    [0.139, 0.139, 0.493, 0.493],
    [0.139, 0.139, 0.493, 0.493, 0.547],
    [0.139, 0.139, 0.493, 0.493, 0.547, 0.2, 0.3],
])
def test_bfgs_on_the_simplex_reproduces_the_solved_ceiling(kids):
    """One BFGS run with the ANALYTIC gradient against the solved ceiling.

    Unconstrained softmax over the simplex (no bounds, no projection — with
    L-BFGS-B on the box the same gradient stopped after one iteration, which is
    what a bounded quasi-Newton does when its projected-step test fires, not a
    property of the gradient).

    MEASURED: 6 / 7 / 9 iterations from ONE centroid start, 1.1 / 1.2 / 1.8 ms,
    landing on 0.999000 / 0.999001 / 0.998985 of the ceiling — i.e. the ceiling
    over its 1.001 margin.  ``log w`` is strictly concave here (see the module
    header), so this stationary point IS the maximum; it is still a LOWER bound,
    so it is an oracle for the ceiling, never a replacement for it.
    """
    from scipy.optimize import minimize

    import tabpwa.phasespace as ps

    m0 = 3.0
    space = ps.PhaseSpace(m0, kids)
    plan = space._plan[space.top]
    value_grad = _simplex_logw(plan, m0)

    def objective(z):
        logw, g = value_grad(z)
        return -logw, -g

    res = minimize(objective, np.zeros(len(plan.md) - 2), jac=True,
                   method="BFGS", options=dict(maxiter=4000))
    found = np.exp(-float(res.fun))
    assert res.nit <= 40, res.nit
    assert found <= plan.ceiling, (found, plan.ceiling)
    assert plan.ceiling / found < 1.01, (plan.ceiling, found)


# ── F. the adapter's contract ─────────────────────────────────────────

def test_weights_out_is_ones_on_every_path():
    """``weights_out=True`` gives ``(momenta, ones)`` — on 3-body AND general."""
    for path in (CFG_ALL, CFG_PWA_GEN, CFG_SKIP):
        m, ch = _model(path), _chain(_model(path))
        mom, w = generate_pwa_phsp(m, ch, 120, seed=4, weights_out=True)
        ref = generate_pwa_phsp(m, ch, 120, seed=4)
        assert w.shape == (120,) and (w == 1.0).all(), path
        assert np.array_equal(mom, ref), path


def test_the_three_body_dispatch_stays_on_the_closed_form():
    """A 3-body chain must NOT drift onto the general fold.

    The closed form is the shape every fixture uses; its numbers must not move
    when the general sampler changes.  The digest is what ``HEAD``'s sampler
    produced for this chain (200 events, seed 5), so a regression that reroutes
    production samples through the fold fails here.
    """
    from tabpwa.pwa_build import _generate_three_body_phsp

    m = _model(CFG_PWA_GEN)
    ch = _chain(m)
    mom = generate_pwa_phsp(m, ch, 200, seed=5)
    assert np.array_equal(mom, _generate_three_body_phsp(m, ch, 200, seed=5))
    assert hashlib.sha256(mom.tobytes()).hexdigest() == \
        "fe009010fbc139f359fd905975bd26ed5f001986e07ca5fac50cfecc3d601a33"


def test_a_top_momentum_routes_a_three_body_chain_through_the_general_fold():
    """``top_p`` must not be silently ignored on the frozen fast path."""
    from tabpwa.phasespace import fix_beam

    m, ch = _model(CFG_PWA_GEN), _chain(_model(CFG_PWA_GEN))
    p = np.array([0.034, 0.0, 0.0])
    mom = generate_pwa_phsp(m, ch, 300, seed=2, top_p=fix_beam(*p))
    tot = mom.sum(axis=1)
    assert np.abs(tot[:, 1:] - p).max() < 1e-9
    M = float(m.dic["particle"][m.top]["mass"])
    assert np.abs(tot[:, 0] - np.sqrt(p @ p + M * M)).max() < 1e-9
    assert not np.array_equal(mom, generate_pwa_phsp(m, ch, 300, seed=2))


@pytest.mark.parametrize("batch", [7, 4096])
def test_batch_chunking_conserves_and_boosts(batch):
    """The chunk loop must lose neither events nor the boost at a boundary."""
    from tabpwa.phasespace import fix_beam

    p = np.array([0.034, 0.0, 0.0])
    mom = _gen((3.0, [0.139, 0.139, 0.493, 0.493, 0.547]), 50, seed=1,
               batch=batch, top_p=fix_beam(*p))
    tot = sum(mom)
    assert tot.shape[0] == 50
    assert np.abs(tot[:, 1:] - p).max() < 1e-9
    m2 = tot[:, 0] ** 2 - np.sum(tot[:, 1:] ** 2, axis=-1)
    assert np.abs(np.sqrt(m2) - 3.0).max() < 1e-9


# ── the refusals ──────────────────────────────────────────────────────

def test_a_final_without_a_mass_is_refused():
    dic = RawConfig(CFG_ALL).to_dict()
    del dic["particle"]["d"]["mass"]
    with pytest.raises(KeyError, match="no 'mass'"):
        generate_pwa_phsp(build_amplitude_model(dic),
                           _chain(build_amplitude_model(dic)), 10, seed=1)


def test_repeated_final_names_are_refused():
    """Two daughters with one name cannot both be stored — say so.

    Both samplers build ``{name: 4-vector}``, so a duplicate would silently
    keep one daughter and produce an event that violates conservation
    (measured: |Σp| = 2.98 with no error before this guard).
    """
    dic = {"amp_model": "pwa", "decay": {"a": [["e", "e"]]},
           "particle": {"$top": "a", "$finals": ["e", "e"],
                        "a": {"J": 1, "P": -1, "spins": [-1, 1], "mass": 3.0},
                        "e": {"J": 0, "P": -1, "mass": 0.139}}}
    m = build_amplitude_model(dic)
    with pytest.raises(ValueError, match="repeated final name"):
        generate_pwa_phsp(m, _chain(m), 10, seed=1)


@pytest.mark.parametrize("bad", [0, -5])
def test_a_batch_that_cannot_advance_is_refused(bad):
    """``batch = 0`` used to spin forever: the chunk is empty, nothing advances."""
    from tabpwa.phasespace import generate_phsp

    with pytest.raises(ValueError, match="batch"):
        generate_phsp(3.0, [0.139, 0.139, 0.547], 10, seed=1, batch=bad)


def test_a_negative_count_is_refused():
    from tabpwa.phasespace import generate_phsp

    with pytest.raises(ValueError, match="n_events"):
        generate_phsp(3.0, [0.139, 0.139, 0.547], -1, seed=1)
    m = _model(CFG_ALL)
    with pytest.raises(ValueError, match="n_events"):
        generate_pwa_phsp(m, _chain(m), -1, seed=1)


def test_an_infeasible_pinned_node_is_refused_immediately():
    """Not "every mass combination has zero weight" 24 000 draws later.

    AT threshold counts as infeasible, exactly as it does for the top: the two
    daughters are at rest, so every weight of that vertex is 0.
    """
    from tabpwa.phasespace import generate_phsp

    with pytest.raises(ValueError, match="pinned node"):
        generate_phsp(3.0, [(0.5, [0.4, 0.4]), 0.547], 10, seed=1)
    with pytest.raises(ValueError, match="no phase space"):
        generate_phsp(3.0, [(0.278, [0.139, 0.139]), 0.547], 10, seed=1)


def test_zero_events_is_empty_not_a_crash():
    m = _model(CFG_ALL)
    mom = generate_pwa_phsp(m, _chain(m), 0, seed=1)
    assert mom.shape == (0, len(m.finals), 4)


def test_a_top_at_threshold_is_refused():
    """At Σm the weight is zero everywhere — refuse instead of looping."""
    dic = RawConfig(CFG_ALL).to_dict()
    s = sum(float(dic["particle"][n]["mass"])
            for n in ("e", "f", "d"))
    dic["particle"]["a"]["mass"] = s
    m = build_amplitude_model(dic)
    with pytest.raises(ValueError, match="no phase space"):
        generate_pwa_phsp(m, _chain(m), 10, seed=1)

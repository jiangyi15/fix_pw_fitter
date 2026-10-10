"""Tests for the BWRS model (width parameter transform).

The ``BWRS`` resonance model is identical in shape to ``BWR``; its
width parameter is reparametrised so the fitted value means the width
AT the fitted mass (a "normal BW" width).  The kernel receives
``width * gamma(M) / gamma(m0)`` through the particle-special
``WidthTransform``.

Verified:
1. the transform is produced by ``BWRS.make_mass_width_transform()``
   (and not by plain ``BWR``) and registers through the standard
   ``setup_mass_width_transforms`` bridge;
2. the transform forward/inverse round-trips exactly;
3. identical kernel params give identical NLL through the BWRS and
   BWR fitters (same likelihood, reparametrised);
4. the analytic gradients of the full BWRS-fitter pipeline (transform
   included) match centered finite differences.

Following the repo's model-test conventions: unit tests build the
particle directly via ``build_particle``; the fitter-level tests patch
``tests/config_pwa_bwr.yml`` in memory (no dedicated fixture config).
"""

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.pwa_build import build_tree_event_data, generate_pwa_phsp
from tabpwa.particle_model import build_particle
from tabpwa.particle_model.bwr_model import WidthTransform

RES = ("MI1m", "MI2p", "MI3m")
STD_CFG = "tests/config_pwa_bwr.yml"


# ── unit level: the particle model itself ────────────────────────

def test_transform_registration():
    # class-level: BWRS overrides the seam, plain BWR does not
    from tabpwa.particle_model.bwr_model import BWRModel, BWRSModel
    assert "make_mass_width_transform" in BWRSModel.__dict__
    assert "make_mass_width_transform" not in BWRModel.__dict__
    assert build_particle("MI1m", model="BWRS",
                          mass=0.769, width=0.15, L=1) is not None


def test_forward_inverse_roundtrip(env):
    # the transform needs the decay tree (q0), so it is built through
    # the fitter bridge rather than a standalone particle
    t = env["tfm"]["MI1m"]
    # distinct output name: the fit parameter is not overwritten
    assert t.input_names == ["MI1m_mass", "MI1m_width"]
    assert t.output_names == ["MI1m_width_t"]
    m0, w = 1.2, 0.3
    fwd = t.forward({t.mass_name: m0, t.width_name: w})
    assert fwd[t.out_name] == pytest.approx(
        w * t._gamma_M / float(np.asarray(
            t.model.gamma(np.array([m0]))).reshape(-1)[0]), rel=1e-12)
    inv = t.inverse({t.mass_name: m0, t.out_name: fwd[t.out_name]})
    assert inv[t.width_name] == pytest.approx(w, rel=1e-12)


# ── fitter level: the full pipeline ──────────────────────────────

@pytest.fixture(scope="module")
def env(tmp_path_factory):
    out = {}
    # BWRS config: the plain-BWR fixture with the model name patched
    text = open(STD_CFG).read().replace("model: BWR", "model: BWRS")
    cfg = tmp_path_factory.mktemp("bwrs") / "config_pwa_bwrs.yml"
    cfg.write_text(text)
    for kind, path in (("std", STD_CFG), ("width", str(cfg))):
        f = Fitter(path, backend="cuda_v4_pwa_s0c")
        f.apply_constrains()
        model, kc = f.model, f.kernel_config
        tree = f.decay_tree
        chain = tree.partial_waves()[0][1]
        byt = {tree.topo_index[ch.topo_id()]: ch
               for _, ch in tree.partial_waves()}
        phsp = build_tree_event_data(
            model, kc, byt, generate_pwa_phsp(model, chain, 300, seed=71))
        data = build_tree_event_data(
            model, kc, byt, generate_pwa_phsp(model, chain, 300, seed=72))
        f.set_phsp(phsp); f.set_data(data)
        out[kind] = f
    out["x0"] = out["std"].initial_values(seed=73)
    tfms = [t for t in out["width"].cm.mass_width_transforms
            if isinstance(t, WidthTransform)]
    assert [t.model.name for t in tfms] == list(RES)
    out["tfm"] = {t.model.name: t for t in tfms}
    return out


def test_transforms_registered_via_fitter(env):
    """The standard bridge registers the transforms on the BWRS fitter."""
    assert len(env["tfm"]) == 3
    # the std (plain BWR) fitter has none
    assert not [t for t in env["std"].cm.mass_width_transforms
                if isinstance(t, WidthTransform)]


def test_same_params_same_nll(env):
    """Identical kernel params -> identical NLL (same likelihood)."""
    f_std, f_w = env["std"], env["width"]
    x = env["x0"].copy()
    _, resolved = f_std.build_params(x)
    nll_std = float(f_std.get_nll_raw(
        f_std._kernel_builder.forward(resolved))[0])
    # the BWRS kernel reads {name}_width_t: feed the same values there
    resolved_w = dict(resolved)
    for name in RES:
        resolved_w[f"{name}_width_t"] = resolved[f"{name}_width"]
    nll_w = float(f_w.get_nll_raw(
        f_w._kernel_builder.forward(resolved_w))[0])
    assert nll_w == pytest.approx(nll_std, rel=1e-12)


def test_bwrs_fitter_gradients_match_fd(env):
    """The full BWRS pipeline (transform included) has consistent grads."""
    f = env["width"]
    x = env["x0"].copy()
    nll, grad = f.get_nll(x)
    names = f.cm.var_registry.flat_names
    gdict = dict(zip(names, np.asarray(grad)))
    rng = np.random.RandomState(7)
    for name in rng.choice(names, size=min(6, len(names)), replace=False):
        i = names.index(name)
        h = 1e-6 * (1.0 + abs(x[i]))
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2 * h)
        assert gdict[name] == pytest.approx(fd, rel=1e-5, abs=1e-6), name

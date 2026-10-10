"""All-topologies PWA: the m² basis across genuinely different subsystems.

``tests/config_pwa_topo.yml`` activates ALL THREE declared topology slots
with different resonance subsystems (ππ, π⁺η, π⁻η) — three DISTINCT mass
columns (``n_mass = n_topo·n_res = 3``), unlike ``config_pwa_bw.yml``
where the identical-particle blocks share one column.

The s0c uniform basis includes an m² table row interpolated at each
wave's own BW mass column; this test verifies the m² basis (and the
whole model) stays correct when the columns differ per topology:
numpy_pwa is the reference, cuda_v4_pwa_s0b (no uniform basis) and
cuda_v4_pwa_s0c (uniform basis incl. the m² row) must match.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CFG = "tests/config_pwa_topo.yml"


def _run(backend, ph, da, x0):
    from tabpwa import Fitter
    f = Fitter(CFG, backend=backend)
    f.apply_constrains()
    f.set_phsp(ph)
    f.set_data(da)
    nll, grad = f.get_nll(x0)
    _, resolved = f.build_params(x0)
    kp = f._kernel_builder.forward(resolved)
    q, _g, P = f.backend.kernel.compute(kp, f._data_holder, norm=None,
                                        return_p=True)
    f.free()
    return float(nll), np.asarray(grad), np.asarray(P)


@pytest.fixture(scope="module")
def data():
    from tabpwa import Fitter
    from tabpwa.pwa_build import build_tree_event_data, generate_pwa_phsp
    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    model, kc = f.model, f.kernel_config
    tree = f.decay_tree
    chain = tree.partial_waves()[0][1]
    byt = {tree.topo_index[ch.topo_id()]: ch for _, ch in tree.partial_waves()}
    ph = build_tree_event_data(model, kc, byt,
                               generate_pwa_phsp(model, chain, 400, seed=71))
    da = build_tree_event_data(model, kc, byt,
                               generate_pwa_phsp(model, chain, 300, seed=72))
    x0 = f.initial_values(seed=73)
    f.free()
    return ph, da, x0


def test_config_has_three_active_mass_columns(data):
    from tabpwa.amp_model import build_amplitude_model
    m = build_amplitude_model(CFG)
    kc = m.build_kernel_config()
    bw = np.asarray(kc["bw_order"]).reshape(-1)
    mi_w = np.asarray(kc["mass_index"])[bw]
    # three distinct subsystems -> three distinct mass columns
    assert len(set(mi_w.tolist())) == 3
    assert int(np.max(kc["mass_index"])) + 1 == 3


def test_s0b_matches_numpy(data):
    ph, da, x0 = data
    try:
        nll_ref, grad_ref, P_ref = _run("numpy_pwa", ph, da, x0)
        nll, grad, P = _run("cuda_v4_pwa_s0b", ph, da, x0)
    except RuntimeError as e:
        pytest.skip(f"unavailable: {e}")
    assert nll == pytest.approx(nll_ref, rel=1e-8)
    np.testing.assert_allclose(P, P_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grad, grad_ref, rtol=1e-4, atol=1e-5)


def test_s0c_uniform_m2_matches_numpy(data):
    ph, da, x0 = data
    try:
        nll_ref, grad_ref, P_ref = _run("numpy_pwa", ph, da, x0)
        nll, grad, P = _run("cuda_v4_pwa_s0c", ph, da, x0)
    except RuntimeError as e:
        pytest.skip(f"unavailable: {e}")
    assert nll == pytest.approx(nll_ref, rel=1e-8)
    np.testing.assert_allclose(P, P_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grad, grad_ref, rtol=1e-4, atol=1e-5)

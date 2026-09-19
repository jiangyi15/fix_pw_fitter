"""Multi-topology PWA: kernel strides must match the declared row widths.

The kc index tables only cover topologies that have partial waves; a
declared-but-empty pairing therefore leaves ``max(index)+1`` smaller than the
real per-event array width.  The CUDA v4/v5 kernels index with the declared
widths, and must agree with the numpy reference.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CFG = "tests/config_pwa.yml"


def _data(nph=800, nd=300):
    from tabpwa import Fitter
    from tabpwa.pwa_build import generate_pwa_phsp, pwa_event_data_tree

    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    kc, tree = f.kernel_config, f.decay_tree
    byt = {tree.topo_index[ch.topo_id()]: ch for _, ch in tree.partial_waves()}
    ch0 = tree.partial_waves()[0][1]
    ph = pwa_event_data_tree(f.model, kc, byt,
                             generate_pwa_phsp(f.model, ch0, nph, seed=11))
    da = pwa_event_data_tree(f.model, kc, byt,
                             generate_pwa_phsp(f.model, ch0, nd, seed=22))
    f.free()
    return kc, ph, da


def test_config_has_empty_declared_pairings():
    from tabpwa.amp_model import build_amplitude_model

    m = build_amplitude_model(CFG)
    kc = m.build_kernel_config()
    C = kc["n_blocks"]
    # The index tables only reach topologies that actually carry partial
    # waves, so a declared-but-empty pairing leaves max(index)+1 below the
    # full topology*block array width.  The CUDA wrappers therefore compact
    # the uploaded arrays to max(index)+1 (compact stride) instead of
    # assuming the full width.
    assert int(np.max(kc["mass_index"])) + 1 < m.n_topo * m.n_res * C
    assert int(np.max(kc["fl_q_index"])) + 1 < m.n_topo * m.n_decay * C
    assert int(np.max(kc["angle_index"])) + 1 < m.n_topo * C


def test_cuda_pwa_matches_numpy_on_multitopo():
    from tabpwa import Fitter

    kc, ph, da = _data()
    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    f.set_phsp(ph)
    f.set_data(da)
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    Qn, gn, Pn = f.backend.compute(params, f._phsp_holder, norm=None,
                                   return_p=True)
    f.free()

    for be in ("cuda_v4_pwa", "cuda_v5_pwa"):
        try:
            g = Fitter(CFG, backend=be)
        except Exception as e:                       # no CUDA / plugin
            pytest.skip(f"{be} unavailable: {e}")
        g.apply_constrains()
        # the wrapper must use the compact (referenced) stride, and the full
        # topology*block arrays handed in below must be sliced to it
        kern = getattr(g.backend, "kernel")
        assert getattr(kern, "n_mass") == int(np.max(kc["mass_index"])) + 1
        assert (getattr(kern, "n_momentum")
                == int(np.max(kc["fl_q_index"])) + 1)
        assert (getattr(kern, "n_angle_total")
                == int(np.max(kc["angle_index"])) + 1)
        g.set_phsp(ph)
        g.set_data(da)
        Qc, gc, Pc = g.backend.compute(params, g._phsp_holder, norm=None,
                                       return_p=True)
        g.free()
        assert Qn == pytest.approx(Qc, rel=1e-5), be
        np.testing.assert_allclose(np.asarray(Pn), np.asarray(Pc),
                                   rtol=1e-5, atol=1e-6, err_msg=be)
        for key in ("ck", "m0", "g0"):
            np.testing.assert_allclose(np.asarray(gn[key]), np.asarray(gc[key]),
                                       rtol=1e-4, atol=1e-5, err_msg=(be, key))

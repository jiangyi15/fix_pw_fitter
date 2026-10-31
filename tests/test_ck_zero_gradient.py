"""Regression: the gradient must stay finite at a zero ck magnitude.

``ck = r·exp(iθ)``; at ``r = 0`` the magnitude slot is a 0/0 point for the
``ck/r`` chain form, while the true derivative is finite.  The analytic
gradient is checked against central finite differences.
"""

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.pwa_build import build_tree_event_data, generate_pwa_phsp


@pytest.fixture(scope="module")
def fitter():
    f = Fitter("tests/config_pwa.yml", backend="numpy_pwa")
    f.apply_constrains()
    tree, kc = f.decay_tree, f.kernel_config
    chain = tree.partial_waves()[0][1]
    byt = {tree.topo_index[ch.topo_id()]: ch for _, ch in tree.partial_waves()}
    f.set_phsp(build_tree_event_data(
        f.model, kc, byt, generate_pwa_phsp(f.model, chain, 600, seed=41)))
    f.set_data(build_tree_event_data(
        f.model, kc, byt, generate_pwa_phsp(f.model, chain, 400, seed=42)))
    yield f
    f.free()


def test_zero_total_r_gradient_matches_fd(fitter):
    f = fitter
    x = f.initial_values(seed=3)
    names = f.cm.var_registry.flat_names
    i = next(k for k, n in enumerate(names) if n.endswith("_total_0r"))

    x[i] = 0.0                            # set one total_r = 0
    _, resolved = f.build_params(x)
    assert resolved[names[i]] == 0.0

    _, g = f.get_nll(x)
    assert np.all(np.isfinite(g)), "analytic gradient has non-finite values"

    eps = 1e-6
    for j in (i, i + 1):                  # magnitude and its phase slot
        xp, xm = x.copy(), x.copy()
        xp[j] += eps
        xm[j] -= eps
        fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2 * eps)
        assert g[j] == pytest.approx(fd, rel=1e-5, abs=1e-7), names[j]

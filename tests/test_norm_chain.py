"""The dNLL/dnorm chain is driven by the gradient keys actually provided.

`Fitter.get_nll_raw` combines the direct data gradient with
``dNLL/dnorm * d(norm)/dparam``.  That combination must not depend on a
hard-coded list of parameter blocks: whatever the backend returns in
``grads`` and whatever the norm path returns in ``norm_grads`` must be
chained, so a new block (``coeff``, ``spin``, a future basis weight) works
without touching the fitter.
"""
import numpy as np
import pytest

from tabpwa import Fitter
from test_cpu_pwa_s0c import _synth

EXTRA_DATA = {"mystery": [1.0, 2.0, -3.0]}
EXTRA_NORM = {"mystery": 0.25, "orphan": [1.5, -0.5]}


@pytest.fixture()
def patched():
    """A real fitter whose backend/norm carry two extra gradient keys.

    ``mystery`` is present in both dicts; ``orphan`` only enters the norm.
    """
    f = Fitter("tests/config_pwa.yml", backend="numpy_pwa")
    kc = f.kernel_config
    f.set_phsp(_synth(kc, 60, seed=1))
    f.set_data(_synth(kc, 40, seed=2))

    real_norm = f._compute_norm_batched
    real_compute = f.backend.compute

    def norm_patched(params):
        norm, g = real_norm(params)
        for key, val in EXTRA_NORM.items():
            g[key] = np.asarray(val, dtype=float)
        return norm, g

    def compute_patched(params, handle, norm=None, return_p=True):
        q, g, p = real_compute(params, handle, norm=norm, return_p=return_p)
        if norm is not None:                    # the normed (data) pass
            for key, val in EXTRA_DATA.items():
                g[key] = np.asarray(val, dtype=float)
        return q, g, p

    f._compute_norm_batched = norm_patched
    f.backend.compute = compute_patched
    return f, real_norm, real_compute


def test_norm_chain_uses_provided_keys(patched):
    f, real_norm, real_compute = patched
    x = f.initial_values(seed=3)
    params, _ = f.build_params(x)

    # the dNLL/dnorm the fitter will use, obtained independently
    norm, _ = real_norm(params)
    _, g_ref, _ = real_compute(params, f._data_holder, norm=norm,
                               return_p=False)
    dn = float(g_ref["norm"])

    _, grads = f.get_nll_raw(params)

    assert "mystery" in grads, "a provided key must not be dropped"
    np.testing.assert_allclose(np.asarray(grads["mystery"]),
                               np.asarray(EXTRA_DATA["mystery"]) + dn * 0.25,
                               rtol=1e-12)

    # a block that only enters the normalisation still contributes
    assert "orphan" in grads, "norm-only keys must still be chained"
    np.testing.assert_allclose(np.asarray(grads["orphan"]),
                               dn * np.asarray(EXTRA_NORM["orphan"]),
                               rtol=1e-12)


def test_norm_chain_keeps_unknown_keys_untouched(patched):
    """A data-only key (no norm counterpart) is passed through unchanged."""
    f, _, _ = patched
    x = f.initial_values(seed=4)
    params, _ = f.build_params(x)
    f.backend.compute_orig = f.backend.compute

    real = f.backend.compute

    def compute_patched(params, handle, norm=None, return_p=True):
        q, g, p = real(params, handle, norm=norm, return_p=return_p)
        if norm is not None:
            g["data_only"] = np.asarray([7.0, -7.0])
        return q, g, p

    f.backend.compute = compute_patched
    _, grads = f.get_nll_raw(params)
    np.testing.assert_array_equal(np.asarray(grads["data_only"]),
                                  np.asarray([7.0, -7.0]))

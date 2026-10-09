"""Parity tests: cuda_v4_pwa_1bw (one BW per chain) vs cuda_v4_pwa.

The 1bw kernel drops the per-wave BW product loop (n_res == 1 → the
propagator IS the single denominator) and gives every wave its OWN BW
slot — no bw_order indirection (the wrapper merges matrix_gamma with
bw_order into per-wave slots), contiguous arrays, no atomics.  NLL is
bit-exact vs the generic kernel; gradients agree to <=1 ulp (the two
kernels are separate compilations, so nvcc scheduling/FMA differs
slightly) plus host-side regrouping of the m0/g0 blocks.
Run on both a fixed-shape config (m0/g0 fixed → ck-only gradients) and a
BW config (free m0/g0 → the m0/g0 blocks are exercised).

Also checks the guard: multi-BW chains (n_res > 1) must raise ValueError.
"""

import numpy as np
import pytest

from tabpwa import Fitter


def _make_env(config, seed=60):
    f = Fitter(config, backend="cuda_v4_pwa")
    f.apply_constrains()
    model, kc = f.model, f.kernel_config
    tree = f.decay_tree
    chain = tree.partial_waves()[0][1]
    byt = {tree.topo_index[ch.topo_id()]: ch
           for _, ch in tree.partial_waves()}
    from tabpwa.pwa_build import build_tree_event_data, generate_pwa_phsp
    phsp = build_tree_event_data(
        model, kc, byt, generate_pwa_phsp(model, chain, 3000, seed=seed))
    data = build_tree_event_data(
        model, kc, byt, generate_pwa_phsp(model, chain, 3000, seed=seed + 1))
    return f, data, phsp


def _run(config, backend, data, phsp, x0):
    f = Fitter(config, backend=backend)
    f.apply_constrains()
    f.set_phsp(phsp)
    f.set_data(data)
    return f.get_nll(x0)


@pytest.mark.parametrize("sorted_backend", ["cuda_v4_pwa_1bw",
                                            "cuda_v4_pwa_1bws"])
@pytest.mark.parametrize("config", ["tests/config_pwa.yml",
                                    "tests/config_pwa_bw.yml"])
def test_one_bw_per_chain_parity(config, sorted_backend):
    f, data, phsp = _make_env(config)
    assert f.kernel_config["bw_order"].ndim == 1          # one slot per wave
    x0 = f.initial_values(seed=63)

    nll_ref, grads_ref = _run(config, "cuda_v4_pwa", data, phsp, x0)
    nll_1bw, grads_1bw = _run(config, sorted_backend, data, phsp, x0)

    if sorted_backend == "cuda_v4_pwa_1bws":
        from tabpwa.cuda._v4_pwa_1bws import CUDAKernelV4PWA1BWS
        kk = CUDAKernelV4PWA1BWS(f.kernel_config)
        assert np.all(np.diff(kk._mass_index_w) >= 0)     # sorted runs
        assert np.array_equal(np.sort(kk.ck_order),
                              np.arange(kk.n_wave_base))
        kk.free()

    nll_ref, nll_1bw = float(nll_ref), float(nll_1bw)
    assert nll_1bw == nll_ref                             # bit-exact NLL
    g_ref = np.asarray(grads_ref)
    g_1bw = np.asarray(grads_1bw)
    # gradients: same per-wave arithmetic, but the two kernels are distinct
    # compilations (nvcc scheduling / FMA contraction) -> <=1-ulp scatter;
    # the m0/g0 blocks additionally regroup per-wave slots on the host
    np.testing.assert_allclose(g_1bw, g_ref, rtol=1e-12, atol=1e-12)


def test_one_bw_rejects_multi_bw_chains():
    """n_res > 1 (two resonances in a chain) must fail fast in the wrapper."""
    f = Fitter("tests/config_pwa.yml", backend="cuda_v4_pwa_1bw")
    from tabpwa.cuda._v4_pwa_1bw import CUDAKernelV4PWA1BW
    kc = dict(f.kernel_config)
    # forge a 2-resonance bw_order (n_res = 2): must raise
    kc2 = dict(kc)
    kc2["bw_order"] = np.repeat(kc["bw_order"], 2)
    with pytest.raises(ValueError, match="ONE BW per chain"):
        CUDAKernelV4PWA1BW(kc2)

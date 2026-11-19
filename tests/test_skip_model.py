"""`model: skip` is a factor that is 1 — and `drop_unit_factors(kc)` drops it.

`tests/config_skip.yml` is ``a -> (b -> e f) (c -> g h) d`` with
``b: model: skip``: structurally two BW factors, but ``b``'s denominator basis
is the single constant row ``[1]``, so the BUILT flat config lets
:func:`tabpwa.pwa_gen_config.drop_unit_factors` drop that factor (2 -> 1) — and the
``n_res == 1`` kernels then accept the chain.

The decision is dict-only: no model, no flag, no name.  The model merely STATES
the constant factor; the config's arrays decide.
"""
import hashlib

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import create_backend
from tabpwa.config_loader import RawConfig
from tabpwa.amp_model import build_amplitude_model
from tabpwa.pwa_gen_config import drop_unit_factors
from test_cuda_flat_cache_pwa_gen import _in_grid

CFG = "tests/config_skip.yml"
CFG_ONE = "tests/config_skip_one.yml"
CFG_ALL = "tests/config_skip_all.yml"


def _cfg(path, **over):
    d = RawConfig(path).to_dict()
    d.update(over)
    return d


def _flat(path):
    return Fitter(_cfg(path, amp_model="pwa_gen"), backend="numpy_flat")


def _digest(v):
    arr = np.asarray(v)
    if arr.dtype == object:
        return hashlib.sha256(repr(v).encode()).hexdigest()[:16]
    return hashlib.sha256(
        np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


# ── the transform itself ───────────────────────────────────────────────

def test_drop_unit_drops_the_identity_factor():
    kc = _flat(CFG).kernel_config
    assert kc["num_table"].shape[1] == 1          # 2 -> 1
    assert kc["num_m_index"].shape[1] == 1
    assert kc["gamma_matrix"].shape[1] == 1
    # the shared "one" basis row survives, and the real BW's rows with it
    assert "one" in kc["den_names"]
    assert "c_s0" in kc["den_names"]
    # STRUCTURAL column: the survivor `c` is factor 1, so its mass column is 1
    assert set(np.unique(np.asarray(kc["num_m_index"]))) == {1}
    # nothing left to drop: the function is idempotent and returns kc itself
    assert drop_unit_factors(kc) is kc
    # the structural axis is untouched on the model
    f = _flat(CFG)
    assert f.model.n_res == 2 and f.model.decay_tree.n_res == 2


@pytest.mark.parametrize("path", ["tests/config_pwa_gen.yml",
                                  "tests/config_pwa_gen_km.yml",
                                  "tests/config_pwa_gen_mb.yml",
                                  "tests/config_pwa_gen_ident.yml",
                                  "tests/config_pwa_gen_cp.yml"])
def test_drop_unit_returns_the_same_object_for_a_real_config(path):
    kc = Fitter(path, backend="numpy_flat").kernel_config
    assert drop_unit_factors(kc) is kc


def test_a_partially_constant_axis_is_left_alone():
    """One factor constant in only SOME waves: the shared axis stays put."""
    kc = _flat(CFG).kernel_config
    g = np.asarray(kc["gamma_matrix"]).copy()
    nt = np.asarray(kc["num_table"]).copy()
    den = list(kc["den_names"])
    one = den.index("one")
    kc2 = dict(kc)
    kc2["gamma_matrix"] = np.concatenate(
        [g, np.zeros((g.shape[0], 1, g.shape[2]))], axis=1)
    kc2["gamma_matrix"][1, 1, one] = 1.0          # factor 1 = 1 in wave 1 only
    kc2["num_table"] = np.concatenate(
        [nt, np.ones((nt.shape[0], 1, nt.shape[2]), nt.dtype)], axis=1)
    kc2["num_m_index"] = np.concatenate(
        [np.asarray(kc["num_m_index"]), np.zeros((g.shape[0], 1), np.int32)],
        axis=1)
    assert drop_unit_factors(kc2) is kc2


# ── the consumers ──────────────────────────────────────────────────────

def test_n_res1_kernels_accept_the_optimised_config():
    from tabpwa.backends.auto_backend import cuda_state
    if not cuda_state()["has_cuda"]:
        pytest.skip("CUDA unusable here")
    f = _flat(CFG)
    for name in ("cuda_flat_pwa_gen_1bw", "cuda_flat_pwa_gen_s0e",
                 "cuda_flat_pwa_gen_s0c", "cuda_flat_pwa_gen_s0c2"):
        create_backend(name, f.kernel_config, model="pwa_gen").free()


@pytest.mark.parametrize("path", [CFG, "tests/config_pwa_gen_mb.yml"])
def test_the_mass_indices_stay_structural_and_point_into_the_buffer(path):
    """`optimize` renumbers NOTHING: both index arrays keep pointing into the
    event `mass` columns the builder writes (a column per factor)."""
    f = _flat(path)
    kc = f.kernel_config
    f.set_phsp(_in_grid(kc, 12, 1))
    f.set_data(_in_grid(kc, 24, 3))
    width = f._data_np["mass"].shape[1]
    num = np.asarray(kc["num_m_index"])
    bas = np.asarray(kc["basis_m_index"])
    assert int(num.max()) < width
    assert int(bas.max()) < width
    # every mass column the numerator reads is also a basis mass column
    assert set(np.unique(num)) <= set(np.unique(bas))
    # the factor AXIS is the compact one; the columns are not
    assert num.shape[1] == kc["gamma_matrix"].shape[1]


def test_drop_unit_on_and_off_give_the_same_numbers():
    """The drop is an OPTIMISATION: `optimize: false` must agree exactly.

    Same config, same params, same sample — one built with the factor axis
    compacted (that factor's denominator is 1), one without.  P is a product
    over factors, so losing a factor of exactly 1 is exact; Q carries one more
    summand, so it may differ by a rounding.
    """
    from tabpwa.backends import create_backend

    off = Fitter(_cfg(CFG, amp_model="pwa_gen", drop_unit=False),
                 backend="numpy_flat")
    on = Fitter(_cfg(CFG, amp_model="pwa_gen", drop_unit=True),
                backend="numpy_flat")
    try:
        kc_off, kc_on = off.kernel_config, on.kernel_config
        assert kc_off["num_table"].shape[1] == 2      # kept
        assert kc_on["num_table"].shape[1] == 1       # dropped
        on.apply_constrains()
        params_on, _ = on.build_params(on.initial_values(seed=1))
        # the SAME parameter point, mapped by basis NAME: the unoptimised config
        # carries one extra coefficient (the dropped factor's "one", which is
        # fixed to 1), everything else must line up name for name
        names_on = [str(n) for n in kc_on["den_names"]]
        names_off = [str(n) for n in kc_off["den_names"]]
        by_name = dict(zip(names_on, np.asarray(params_on["coeff"])))
        params_off = dict(params_on)
        params_off["coeff"] = np.array([by_name.get(n, 1.0) for n in names_off])
        data = _in_grid(kc_off, 48, 2)
        a = create_backend("numpy_flat", kc_off, model="pwa_gen")
        b = create_backend("numpy_flat", kc_on, model="pwa_gen")
        try:
            Qa, ga, Pa = a.compute(params_off, a.load_data(data), norm=1.3)
            Qb, gb, Pb = b.compute(params_on, b.load_data(data), norm=1.3)
        finally:
            a.free()
            b.free()
        assert np.array_equal(np.asarray(Pa), np.asarray(Pb)), "P must be exact"
        assert abs(Qa - Qb) <= 1e-13 * max(abs(Qa), 1.0)
        assert np.allclose(np.asarray(ga["ck"]), np.asarray(gb["ck"]),
                           rtol=1e-13, atol=1e-15)
    finally:
        off.backend.free()
        on.backend.free()


def test_the_plain_pwa_path_refuses_a_skip_factor():
    """No running width -> the existing guard names `pwa_gen`."""
    with pytest.raises(ValueError, match="declare none"):
        Fitter(CFG, backend="numpy_pwa")


def test_the_chain_of_only_constant_factors_has_nothing_to_fit():
    # the builder keeps the structural axis; with no factor left the *plain*
    # guard and the flat kernels both have nothing to evaluate
    with pytest.raises(ValueError):
        Fitter(CFG_ALL, backend="numpy_pwa")


# ── equivalence with the slot-occupying `one` ─────────────────────────

def test_the_skip_factor_is_numerically_the_one_twin():
    a = _flat(CFG)
    b = _flat(CFG_ONE)
    a.apply_constrains()
    b.apply_constrains()
    pa, _ = a.build_params(a.initial_values(seed=1))
    pb = dict(b.build_params(b.initial_values(seed=1))[0])
    pb["ck"] = pa["ck"]
    # one event set, every mass column given the same values, so both kernels
    # evaluate the real BW at the same masses (the dropped column is not read)
    data = _in_grid(b.kernel_config, 40, 3)
    data["mass"] = np.repeat(data["mass"][:, :1], data["mass"].shape[1], axis=1)
    Qa, ga, Pa = a.backend.compute(pa, a.backend.load_data(data), norm=1.3)
    Qb, gb, Pb = b.backend.compute(pb, b.backend.load_data(data), norm=1.3)
    assert np.allclose(Pa, Pb, rtol=1e-13, atol=0.0)
    assert abs(Qa - Qb) <= 1e-13 * max(abs(Qa), abs(Qb), 1.0)
    assert np.allclose(ga["ck"], gb["ck"], rtol=1e-13, atol=0.0)


# ── no regression: the existing kernel configs are byte-identical ─────

KC_BEFORE = {
    'tests/config_pwa.yml': {
        'angle_b': '2e00600b4936e712',
        'angle_index': '2ea9ab9198d16380',
        'angle_k': '063048130d6ff739',
        'bw_order': '8a813d53c6ae983e',
        'ck_map': 'c244036f04d485c3',
        'fl_delta': 'fd7c802da31cc7f4',
        'fl_min': 'af5570f5a1810b7a',
        'fl_order': 'c72bb1645a3511dc',
        'fl_q_index': '741b91ec0d673aba',
        'fl_table': 'e394210c2d9a5e5b',
        'fl_type': 'b8d04b8e4644977d',
        'g0_index': 'ab25350e3e65efeb',
        'g0_mass_index': '9d908ecfb6b256de',
        'g0_names': '99c021e40cf366eb',
        'gamma_delta': '1231a9cc4af79980',
        'gamma_min': '41b6e4e4b1b96d71',
        'gamma_table': 'c4c981a1b1e5a518',
        'm0_index': 'ab25350e3e65efeb',
        'm0_names': '00a709d4b71eee8a',
        'mass_index': '9d908ecfb6b256de',
        'matrix_angle': 'f5cac9da5a7c3af3',
        'matrix_gamma': '1332df685c45a61a',
        'n_blocks': '7c9fa136d4413fa6',
        'n_cp': '7c9fa136d4413fa6',
        'n_identical': '7c9fa136d4413fa6',
        'n_perm': '7c9fa136d4413fa6',
        'n_proj': 'd86e8112f3c4c444',
        'n_proj_base': 'd86e8112f3c4c444',
        'top_j0': '6e340b9cffb37a98',
        'top_states': 'c9cb04ba987a95a5',
        'variables': 'ac033248e9dc2457',
        'wave_names': 'ec984e340c015ef5',
    },
    'tests/config_pwa_gen.yml': {
        'angle_b': 'f39f733d6c9bd617',
        'angle_index': 'de47c9b27eb8d300',
        'angle_k': 'ef06b25337a7908e',
        'basis_key': '47ac873a6869b245',
        'basis_kind': 'ad5dc1478de06a4c',
        'basis_m_index': '15ec7bf0b50732b4',
        'basis_table': '2ed8aa4f882b0535',
        'den_names': '6e4e972d6442fe0a',
        'fl_q_index': '01acecb507abfe1a',
        'fl_table': 'c8c7f68b823276c8',
        'gamma_matrix': 'cc143326a2646c60',
        'm_delta': '1231a9cc4af79980',
        'm_min': '41b6e4e4b1b96d71',
        'matrix_angle': '0765df40294d98b2',
        'n_blocks': '7c9fa136d4413fa6',
        'n_cp': '7c9fa136d4413fa6',
        'n_identical': '7c9fa136d4413fa6',
        'n_perm': '7c9fa136d4413fa6',
        'n_proj': '35be322d094f9d15',
        'n_proj_base': '35be322d094f9d15',
        'num_m_index': 'df3f619804a92fdb',
        'num_table': '0a124928e4766265',
        'q_delta': 'fd7c802da31cc7f4',
        'q_min': 'af5570f5a1810b7a',
        'top_j0': '6e340b9cffb37a98',
        'top_states': 'd6c3f800c1b53a78',
        'variables': 'ac033248e9dc2457',
        'wave_names': 'd74d6da8731c4b1e',
    },
    'tests/config_pwa_gen_km.yml': {
        'angle_b': 'f39f733d6c9bd617',
        'angle_index': 'de47c9b27eb8d300',
        'angle_k': 'ef06b25337a7908e',
        'basis_key': '9010637893d81947',
        'basis_kind': 'e528f4309e1413e6',
        'basis_m_index': 'de47c9b27eb8d300',
        'basis_table': '7b25169663e36296',
        'den_names': '2b7181b451f17abf',
        'fl_q_index': '7605713c8dae5653',
        'fl_table': 'c0e52703af82171f',
        'gamma_matrix': 'ad8a323c573d41bd',
        'm_delta': 'c3c1a1723a988a31',
        'm_min': '41b6e4e4b1b96d71',
        'matrix_angle': '2c2387a3ecf458b4',
        'n_blocks': '7c9fa136d4413fa6',
        'n_cp': '7c9fa136d4413fa6',
        'n_identical': '7c9fa136d4413fa6',
        'n_perm': '7c9fa136d4413fa6',
        'n_proj': '35be322d094f9d15',
        'n_proj_base': '35be322d094f9d15',
        'num_m_index': 'af5570f5a1810b7a',
        'num_table': 'cbfcc5996137908d',
        'q_delta': '0e71c7190faef7be',
        'q_min': 'af5570f5a1810b7a',
        'top_j0': '6e340b9cffb37a98',
        'top_states': 'd6c3f800c1b53a78',
        'variables': 'ac033248e9dc2457',
        'wave_names': '10a5ae8889c590a6',
    },
    'tests/config_amp.yml': {
        'angle_b': '03a6a3bd1ba793dd',
        'angle_index': '4137577a02179090',
        'angle_k': 'c6224973865f5c52',
        'bw_order': 'f11ea22a28f07d00',
        'ck_map': 'de542d2b5db45b93',
        'fl_delta': '40897fc6b6753885',
        'fl_min': 'af5570f5a1810b7a',
        'fl_order': '9353d4b31fd6dc1f',
        'fl_q_index': '0d8b3c8609a94740',
        'fl_table': 'c242e7ec51fc72cc',
        'fl_type': 'c9f4bc269ccae78e',
        'g0_index': '0d257a8e6b9da2f9',
        'g0_mass_index': 'c103e0b94555df1c',
        'gamma_delta': '82c19ab53f20e01b',
        'gamma_min': '3beef8283adfe7b7',
        'gamma_table': '37ada2ef0f24d0f7',
        'm0_index': 'b7f07dd0fc416252',
        'mass_index': 'a27e6e2512d0d2cb',
        'matrix_angle': '356f471fcf243a10',
        'matrix_gamma': 'a4a17ba5af488410',
        'n_blocks': '6cc16abd70eefb90',
        'n_cp': 'd86e8112f3c4c444',
        'n_perm': 'f0a0278e4372459c',
        'n_proj': '7c9fa136d4413fa6',
        'n_proj_base': '7c9fa136d4413fa6',
        'top_j0': '4bf5122f344554c5',
        'variables': 'db695389e134f7ec',
    },
}


def test_existing_kernel_configs_are_unchanged():
    for path, hashes in KC_BEFORE.items():
        kc = build_amplitude_model(RawConfig(path)).build_kernel_config()
        got = {k: _digest(kc[k]) for k in sorted(kc)}
        assert got == hashes, path

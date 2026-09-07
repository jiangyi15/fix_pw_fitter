"""Momentum -> alignment euler angles (spinful finals, multi-topology)."""
import numpy as np

from ampfit.config_loader import Config
from ampfit.momenta_to_angles import (
    aligned_euler_from_momenta, _chain_total_rotations,
)

LC_CONFIG = """data_order:
    dat_order: [Lambda, pip, eta]
    data: /tmp/does-not-exist.npy
    phsp: /tmp/does-not-exist.npy

decay:
    Lc:
    - [Sigmapi, eta]
    - [pieta, Lambda]
    Sigmapi: [Lambda, pip]
    pieta: [pip, eta]

particle:
    $top: Lc
    $finals: [Lambda, pip, eta]
    Sigmapi: [ Sig1385p ]
    pieta: [ a098 ]
    Lc:
        J: 0.5
        P: +1
        spins: [-0.5, 0.5]
        mass: 2.28646
    Lambda:
        J: 0.5
        P: +1
        spins: [-0.5, 0.5]
        mass: 1.11568
    pip:
        J: 0
        P: -1
        mass: 0.13957
    eta:
        J: 0
        P: -1
        mass: 0.54786
    Sig1385p:
        J: 1.5
        P: +1
        mass: 1.3828
        width: 0.037
        model: BW
    a098:
        J: 0
        P: +1
        mass: 0.98
        width: 0.075
        model: BW
"""


def _cm_momenta(n_events=3, seed=7):
    """Random Lambda_c -> Lambda + pi + eta final momenta in the CM."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.4, (n_events, 3, 3))
    m = {"Lambda": 1.11568, "pip": 0.13957, "eta": 0.54786}
    names = ["Lambda", "pip", "eta"]
    tot = p.sum(axis=1)
    p = p - tot[:, None, :] / 3
    mom = {}
    for j, nm in enumerate(names):
        pj = p[:, j]
        Ej = np.sqrt(m[nm] ** 2 + np.sum(pj * pj, axis=-1))
        mom[nm] = np.stack([Ej, pj[:, 0], pj[:, 1], pj[:, 2]], axis=-1)
    return mom


def _chains(cfg):
    return [dc for _, dc in cfg.full_decay.get_partial_waves()]


def _cfg(tmp_path):
    cfg = tmp_path / "lc.yml"
    cfg.write_text(LC_CONFIG)
    c = Config(str(cfg))
    assert c.build_all_index()["angle_k"].shape[1] == 7  # 4 vertex + 3 align
    return c


def test_multi_topo_returns_columns(tmp_path):
    c = _cfg(tmp_path)
    chains = _chains(c)
    assert len(chains) == 2
    mom = _cm_momenta()
    out = aligned_euler_from_momenta(chains, mom, ["Lambda"])
    assert out is not None
    a = out["Lambda"]
    assert a.shape == (3, 2, 3)                 # (n_events, n_chains, euler)
    assert np.all(a[:, :, 1] >= -1e-12) and np.all(a[:, :, 1] <= np.pi + 1e-12)


def test_single_topo_returns_none(tmp_path):
    c = _cfg(tmp_path)
    chains = _chains(c)
    mom = _cm_momenta()
    assert aligned_euler_from_momenta(chains[:1], mom, ["Lambda"]) is None


def test_total_rotation_unit(tmp_path):
    c = _cfg(tmp_path)
    dc = _chains(c)[0]
    mom = _cm_momenta()
    R = _chain_total_rotations(dc, mom)
    # each stored rotation is orthogonal
    for name, Rv in R.items():
        Id = np.matmul(Rv, np.swapaxes(Rv, -1, -2))
        assert np.allclose(Id, np.eye(3), atol=1e-9), name
        det = np.linalg.det(Rv)
        assert np.allclose(det, 1.0, atol=1e-9), name


def test_reference_chain_zero(tmp_path):
    """The reference chain (Lambda produced directly from the top in chain
    ``Lc -> [pieta, Lambda]``) contributes zero alignment for itself."""
    c = _cfg(tmp_path)
    chains = _chains(c)
    mom = _cm_momenta()
    out = aligned_euler_from_momenta(chains, mom, ["Lambda"])
    # find which chain has Lambda as a direct top child
    ref = None
    for i, dc in enumerate(chains):
        if any(o.name == "Lambda" for o in dc.decays[0].outs):
            ref = i
    assert ref is not None
    # reference-chain slice is identically zero; the other chain carries the
    # real alignment euler.
    a = out["Lambda"]
    assert np.allclose(a[:, ref, :], 0.0, atol=1e-12)
    other = a[:, 1 - ref, :]
    assert np.all(np.abs(other[:, 0]) > 1e-6) or np.all(
        np.abs(other[:, 2]) > 1e-6)

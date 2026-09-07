"""Momentum -> alignment euler angles (spinful finals, multi-topology)."""
import numpy as np

from ampfit.config_loader import Config
from ampfit.momenta_to_angles import (
    aligned_euler_from_momenta, _chain_total_rotations,
)

LC_CONFIG = """data:
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


def test_center_mass_default_independent_of_chain(tmp_path):
    """center_mass (default): reference from lab axes + the final's own CM
    momentum, so both chains carry real alignment and swapping the chain
    order leaves each chain's euler unchanged."""
    c = _cfg(tmp_path)
    chains = _chains(c)
    mom = _cm_momenta()
    out = aligned_euler_from_momenta(chains, mom, ["Lambda"])
    a = out["Lambda"]
    assert a.shape == (3, 2, 3)
    # both chains non-trivial
    assert np.any(np.abs(a[:, 0, :]) > 1e-6)
    assert np.any(np.abs(a[:, 1, :]) > 1e-6)
    # chain-order independence (each slice determined by its own chain only)
    rev = aligned_euler_from_momenta([chains[1], chains[0]], mom,
                                     ["Lambda"])["Lambda"]
    assert np.allclose(a[:, 0, :], rev[:, 1, :], atol=1e-9)
    assert np.allclose(a[:, 1, :], rev[:, 0, :], atol=1e-9)


def test_rule1_chain_reference_zero(tmp_path):
    """align_ref="chain" (rule1): the chain where Lambda is a direct top
    child gives a zero slice, the other one the real alignment."""
    c = _cfg(tmp_path)
    chains = _chains(c)
    mom = _cm_momenta()
    out = aligned_euler_from_momenta(chains, mom, ["Lambda"],
                                     align_ref="chain")
    ref = None
    for i, dc in enumerate(chains):
        if any(o.name == "Lambda" for o in dc.decays[0].outs):
            ref = i
    assert ref is not None
    a = out["Lambda"]
    assert np.allclose(a[:, ref, :], 0.0, atol=1e-12)
    assert np.any(np.abs(a[:, 1 - ref, :]) > 1e-6)


def test_pwa_event_data_tree_fill(tmp_path):
    """Tree-shape event fill: only angles need the tree (already tree-based
    decay_angles_vectorized); mass/q/angle columns come from FLAT per-decay
    loops and match independent physics for every topology row."""
    from ampfit.pwa_build import pwa_event_data_tree, _two_body_p
    from ampfit.momenta_to_angles import decay_angles_vectorized
    from ampfit.helicity_angle import decay_chain_leaves
    c = _cfg(tmp_path)
    kc = c.build_all_index()
    byt, chains = {}, []
    for _, dc in c.full_decay.get_partial_waves():
        tid = c.topo_index[dc.topo_id()]
        byt[tid] = dc
        chains.append(dc)
    mom_d = _cm_momenta(n_events=5)
    names = ["Lambda", "pip", "eta"]
    mom = np.stack([mom_d[nm] for nm in names], axis=1)
    d = pwa_event_data_tree(c, kc, byt, mom, spinful_names=["Lambda"])
    assert d["mass"].shape == (5, c.n_topo * c.n_res)
    assert d["q"].shape == (5, c.n_topo * c.n_decay)
    nv = c.n_decay
    assert d["angle"].shape == (5, c.n_topo, 2 * nv + 3)
    for tid in sorted(byt):
        ch = byt[tid]
        lm = [o.name for o in decay_chain_leaves(ch)]
        mm = np.stack([mom[:, names.index(x)] for x in lm], axis=1)
        ph, th = decay_angles_vectorized(ch, mm)
        assert np.allclose(d["angle"][:, tid, :nv], ph, atol=1e-12)
        assert np.allclose(d["angle"][:, tid, nv:2 * nv], th, atol=1e-12)
        outs = [o.name for o in ch.decays[1].outs]
        s4 = np.zeros((5, 4))
        for o in outs:
            s4 = s4 + mom_d[o]
        mi = np.sqrt(np.clip((s4 ** 2) @ np.array([1, -1, -1, -1.]), 0, None))
        assert np.allclose(d["mass"][:, tid], mi, atol=1e-9)
        m0 = float(c.dic["particle"][outs[0]]["mass"])
        m1 = float(c.dic["particle"][outs[1]]["mass"])
        assert np.allclose(d["q"][:, c.n_decay * tid + 1],
                           _two_body_p(mi, m0, m1), atol=1e-9)
        # alignment slice equals the analytic angle/|p|-based function
        # (engine-consistent zyz(gamma=0) frames; independently re-derived)
        from ampfit.momenta_to_angles import aligned_euler_from_chain
        outs = [o.name for o in ch.decays[1].outs]
        s4 = np.zeros((5, 4))
        for o in outs:
            s4 = s4 + mom_d[o]
        mi = np.sqrt(np.clip((s4 ** 2) @ np.array([1, -1, -1, -1.]), 0, None))
        m_node = {nm: np.full(5, float(c.dic["particle"][nm]["mass"]))
                  for nm in lm}
        m_node[ch.decays[1].core.name] = mi
        ao = aligned_euler_from_chain(
            ch, ph, th, d["q"][:, c.n_decay * tid:c.n_decay * (tid + 1)],
            m_node, mom_d, ["Lambda"])
        assert np.allclose(d["angle"][:, tid, 2 * nv:2 * nv + 3],
                           ao["Lambda"], atol=1e-12)
        assert np.all(np.isfinite(d["angle"][:, tid, 2 * nv:2 * nv + 3]))

"""Identical-particle / CP block expansion for tree event data."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CONFIG = os.path.join(ROOT, "tests", "config_pwa.yml")


def _cfg(ident=None, cp=None):
    from tabpwa.config_loader import Config, load_config

    dic = load_config(CONFIG)
    dic["data"] = dict(dic.get("data") or {})
    if ident:
        dic["data"]["identical_particles"] = ident
    if cp:
        dic["data"]["cp_particles"] = cp
    return Config(dic)


def _mom(n, seed=3):
    """Physical pion/eta 4-momenta (E from mass), like test_loader_utils."""
    rs = np.random.RandomState(seed)
    p3 = rs.uniform(-0.4, 0.4, size=(n, 3, 3))
    m = {"pip": 0.13957, "pim": 0.13957, "eta": 0.54786}
    out = np.empty((n, 3, 4))
    for j, nm in enumerate(["pip", "pim", "eta"]):
        E = np.sqrt(m[nm] ** 2 + (p3[:, j] ** 2).sum(-1))
        out[:, j] = np.concatenate([E[:, None], p3[:, j]], axis=-1)
    return out


def _byt(tree):
    return {tree.topo_index[ch.topo_id()]: ch for _, ch in tree.partial_waves()}


def test_block_orders_counts():
    from tabpwa.pwa_build import block_orders

    f = ["pip", "pim", "eta"]
    assert block_orders(f, {}) == [((0, 1, 2), False)]

    b = block_orders(f, {"identical_particles": [["pip", "pim"]]})
    assert len(b) == 2 and all(not is_cp for _, is_cp in b)

    b = block_orders(f, {"cp_particles": [["pip", "pim"]]})
    assert len(b) == 2 and sorted(is_cp for _, is_cp in b) == [False, True]

    b = block_orders(f, {"identical_particles": [["pip", "pim"]],
                         "cp_particles": [["pip", "pim"]]})
    assert len(b) == 4 and sum(is_cp for _, is_cp in b) == 2


def test_build_tree_event_data_single_block_is_unchanged():
    from tabpwa.pwa_build import build_tree_event_data, pwa_event_data_tree

    cfg = _cfg()
    tree = cfg.decay_tree
    byt = _byt(tree)
    mom = _mom(5, 0)
    a = build_tree_event_data(tree, None, byt, mom)
    b = pwa_event_data_tree(tree, None, byt, mom)
    for k in ("mass", "q", "angle"):
        assert np.allclose(a[k], b[k])


def test_build_tree_event_data_blocks_shapes_and_composition():
    from tabpwa.pwa_build import (block_orders, build_tree_event_data,
                                  pwa_event_data_tree)

    cfg = _cfg(ident=[["pip", "pim"]], cp=[["pip", "pim"]])
    tree = cfg.decay_tree
    byt = _byt(tree)
    mom = _mom(5, 1)
    blocks = block_orders(tree.finals, cfg.dic["data"])
    assert len(blocks) == 4

    d = build_tree_event_data(tree, None, byt, mom, blocks=blocks)
    n_topo, n_res, n_decay, n_base = tree.n_topo, tree.n_res, tree.n_decay, 4
    assert d["mass"].shape == (5, 4 * n_topo * n_res)
    assert d["q"].shape == (5, 4 * n_topo * n_decay)
    assert d["angle"].shape == (5, 4 * n_topo, n_base)

    # each block equals a direct per-block tree walk on its momentum order
    for b, (order, _) in enumerate(blocks):
        direct = pwa_event_data_tree(tree, None, byt, mom[:, list(order)])
        sl = slice(b * n_topo, (b + 1) * n_topo)
        assert np.allclose(d["mass"][:, b * n_topo * n_res:(b + 1) * n_topo * n_res],
                           direct["mass"])


def test_cp_block_reverses_three_momentum():
    from tabpwa.pwa_build import (block_orders, build_tree_event_data,
                                  pwa_event_data_tree)

    cfg = _cfg(cp=[["pip", "pim"]])
    tree = cfg.decay_tree
    byt = _byt(tree)
    mom = _mom(5, 2)
    blocks = block_orders(tree.finals, cfg.dic["data"])
    d = build_tree_event_data(tree, None, byt, mom, blocks=blocks)

    from tabpwa.pwa_build import _boost_to_cm

    cp_b = [i for i, (_, is_cp) in enumerate(blocks) if is_cp][0]
    order = list(blocks[cp_b][0])
    mom_cm = _boost_to_cm(mom)
    mom_cp = mom_cm[:, order].copy()
    mom_cp[:, :, 1:] *= -1.0                       # CP: reverse 3-momentum (CM)
    direct = pwa_event_data_tree(tree, None, byt, mom_cp, cm_boost=False)
    lo, hi = cp_b * tree.n_topo, (cp_b + 1) * tree.n_topo
    assert np.allclose(d["angle"][:, lo:hi, :], direct["angle"])
    nres, ndec = tree.n_res, tree.n_decay
    assert np.allclose(d["mass"][:, cp_b * tree.n_topo * nres:(cp_b + 1) * tree.n_topo * nres],
                       direct["mass"])

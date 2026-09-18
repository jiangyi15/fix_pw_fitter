"""DecayTree: standalone decay-tree value object + Config composition."""

import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CONFIG = os.path.join(ROOT, "tests", "config_pwa.yml")


def test_config_composes_a_decay_tree():
    from ampfit.config_loader import Config
    from ampfit.decay_tree import DecayTree

    c = Config(CONFIG)
    assert isinstance(c.decay_tree, DecayTree)
    # legacy attribute names are aliases of the tree's data
    assert c.full_decay is c.decay_tree.full
    assert c.decay_struct is c.decay_tree.struct
    assert c.topo_index is c.decay_tree.topo_index
    assert c.n_topo == c.decay_tree.n_topo
    assert c.n_decay == c.decay_tree.n_decay
    assert c.n_res == c.decay_tree.n_res
    assert c.finals == c.decay_tree.finals


def test_decay_tree_builds_from_declarations_only():
    from ampfit.config_loader import Config, load_config
    from ampfit.decay_tree import DecayTree

    dic = load_config(CONFIG)
    tree = DecayTree(dic["decay"], dic["particle"])
    c = Config(CONFIG)
    assert tree.topo_index == c.topo_index
    assert len(tree.partial_waves()) == len(c.full_decay.get_partial_waves())
    assert tree.topo_index_from_name("pipi") == c.topo_index_from_name("pipi")


def test_decay_classes_reexported_for_compat():
    from ampfit.config_loader import Particle, Decay, DecayChain, DecayGroup
    from ampfit import decay_tree

    assert Particle is decay_tree.Particle
    assert Decay is decay_tree.Decay
    assert DecayChain is decay_tree.DecayChain
    assert DecayGroup is decay_tree.DecayGroup


def test_decay_tree_module_is_leaf():
    """decay_tree must not import config_loader (layering stays acyclic)."""
    import ast

    from ampfit import decay_tree

    mods = []
    for node in ast.walk(ast.parse(inspect.getsource(decay_tree))):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
    assert not any("config_loader" in m for m in mods), mods


def test_decay_tree_display_helpers_are_model_free():
    """Particle / decay-chain labels live on the tree, not on a model."""
    from ampfit.config_loader import load_config
    from ampfit.decay_tree import DecayTree

    dic = load_config(CONFIG)
    tree = DecayTree(dic["decay"], dic["particle"])
    names = tree.name_display_map()
    assert names["pip"]                      # non-empty LaTeX label
    chain = tree.full.chains[0]
    assert r"\to" in tree.display_decay(chain.decays[0])
    assert tree.display_chain(chain)


def test_decay_tree_owns_symmetry_declarations():
    """identical/cp declarations are decay information on the tree."""
    from ampfit.config_loader import load_config
    from ampfit.decay_tree import DecayTree

    dic = load_config(CONFIG)
    dic["data"] = dict(dic.get("data") or {})
    dic["data"]["identical_particles"] = [["pip", "pim"]]
    dic["data"]["cp_particles"] = [["pip", "pim"]]
    tree = DecayTree(dic["decay"], dic["particle"], dic["data"])

    assert tree.n_perm == 2 and tree.n_cp == 2 and tree.n_blocks == 4
    assert tree.identical_groups == [["pip", "pim"]]
    orders = tree.block_orders()
    assert len(orders) == 4 and sum(is_cp for _, is_cp in orders) == 2


def test_finals_order_follows_dat_order():
    """dat_order defines the particle/column order ($finals is just a list)."""
    from ampfit.config_loader import Config, load_config

    dic = load_config(CONFIG)
    dic["data"] = dict(dic.get("data") or {})
    dic["data"]["dat_order"] = ["eta", "pip", "pim"]
    assert Config(dic).decay_tree.finals == ["eta", "pip", "pim"]

    # no dat_order -> fall back to the declared $finals order
    dic2 = load_config(CONFIG)
    dic2["data"] = {k: v for k, v in (dic2.get("data") or {}).items()
                    if k != "dat_order"}
    c2 = Config(dic2)
    assert c2.decay_tree.finals == list(c2.dic["particle"]["$finals"])

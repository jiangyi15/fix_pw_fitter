"""ConfigBroadcast — one template config → N configs (pure dict level)."""
import copy

import pytest
import yaml

from tabpwa.config_broadcast import ConfigBroadcast, expand_configs

BASE = {
    "data": {
        "dat_order": ["pip", "pim", "eta"],
        "phsp_arr": "shared.npz",
        "purity": 1.0,
    },
    "particle": {"$top": "jpsi"},
}


def test_no_broadcast_passthrough():
    bc = ConfigBroadcast(BASE)
    assert bc.n_configs == 1
    assert not bc.is_broadcast
    assert bc.expand() == [BASE]
    assert expand_configs(BASE) == [BASE]


def test_diff_list_deep_merge_and_labels():
    base = copy.deepcopy(BASE)
    base["broadcast"] = [
        {"data": {"data_arr": "a.npz"}},
        {"data": {"data_arr": "b.npz"}, "data.purity": 0.93,
         "label": "runB"},
    ]
    bc = ConfigBroadcast(base)
    assert bc.is_broadcast and bc.n_configs == 2
    assert bc.labels() == ["cfg0", "runB"]

    c0, c1 = bc.expand()
    # deep merge: shared base keys survive, the entry's keys are added
    assert c0["data"]["phsp_arr"] == "shared.npz"
    assert c0["data"]["data_arr"] == "a.npz"
    assert c1["data"]["purity"] == 0.93          # dotted key applied
    assert c1["data"]["dat_order"] == ["pip", "pim", "eta"]
    # metadata never lands in the config; blocks are removed
    assert "label" not in c1
    assert "broadcast" not in c0 and "broadcast" not in c1
    # the template itself is never mutated
    assert "broadcast" in base and "data_arr" not in base["data"]
    assert ConfigBroadcast.items(base) == [
        ("0:cfg0", bc.expand()[0]),
        ("1:runB", bc.expand()[1]),
    ]


def test_dotted_key_matches_nested_form():
    base = copy.deepcopy(BASE)
    base["broadcast"] = [{"data.data_arr": "x.npz", "data.purity": 0.5}]
    nested = copy.deepcopy(BASE)
    nested["broadcast"] = [{"data": {"data_arr": "x.npz", "purity": 0.5}}]
    assert (ConfigBroadcast(base).expand()
            == ConfigBroadcast(nested).expand())


def test_relative_paths_resolved_against_config_dir(tmp_path):
    cfg = copy.deepcopy(BASE)
    cfg["data"]["phsp_arr"] = "mc/phsp.npz"
    cfg["broadcast"] = [{"data": {"data_arr": "run1.npz"}}]
    p = tmp_path / "multi.yml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))

    c = ConfigBroadcast(str(p)).expand()[0]
    assert c["data"]["data_arr"] == str(tmp_path / "run1.npz")
    assert c["data"]["phsp_arr"] == str(tmp_path / "mc" / "phsp.npz")
    # a dict source has no directory: paths are left untouched
    c2 = ConfigBroadcast(copy.deepcopy(cfg)).expand()[0]
    assert c2["data"]["data_arr"] == "run1.npz"


def test_errors():
    with pytest.raises(ValueError, match="empty 'broadcast'"):
        ConfigBroadcast({"broadcast": []})
    with pytest.raises(ValueError, match="list of dict diffs"):
        ConfigBroadcast({"broadcast": ["a.npz", "b.npz"]})


def test_has_broadcast_flag():
    """The explicit has/has-not test consumers branch on."""
    assert ConfigBroadcast.has_broadcast(BASE) is False
    with_b = copy.deepcopy(BASE)
    with_b["broadcast"] = [{"data": {"data_arr": "a.npz"}}]
    assert ConfigBroadcast.has_broadcast(with_b) is True
    empty = copy.deepcopy(BASE)
    empty["broadcast"] = []
    assert ConfigBroadcast.has_broadcast(empty) is True   # present → error at build


def test_forbidden_keys_setting():
    """The forbidden key set is an input setting (default: `decay`)."""
    # default: decay is forbidden (nested and dotted forms)
    with pytest.raises(ValueError, match="forbidden key"):
        ConfigBroadcast({**BASE, "broadcast": [{"decay": {"a": 1}}]})
    with pytest.raises(ValueError, match="forbidden key"):
        ConfigBroadcast({**BASE, "broadcast": [{"data.data_arr": "x.npz"},
                                               {"decay.jpsi": [["pip", "pim"]]}]})
    # disabled with forbidden=()
    cb = ConfigBroadcast({**BASE, "broadcast": [{"decay": {"a": 1}}]},
                         forbidden=())
    assert cb.expand()[0]["decay"] == {"a": 1}
    # a custom forbidden set protects other sections instead
    with pytest.raises(ValueError, match="forbidden key"):
        ConfigBroadcast({**BASE, "broadcast": [{"particle": {"jpsi": {}}}]},
                        forbidden=("particle",))
    # non-forbidden keys are unaffected
    cb2 = ConfigBroadcast({**BASE, "broadcast": [{"particle": {"jpsi": {}}}]},
                          forbidden=("decay",))
    assert cb2.expand()[0]["particle"]["jpsi"] == {}


def test_class_items_entry_point():
    base = copy.deepcopy(BASE)
    base["broadcast"] = [
        {"data": {"data_arr": "a.npz"}},
        {"data": {"data_arr": "b.npz"}, "label": "runB"},
    ]
    items = ConfigBroadcast.items(base)
    assert [lbl for lbl, _ in items] == ["0:cfg0", "1:runB"]
    assert items[1][1]["data"]["data_arr"] == "b.npz"
    # plain source: one unlabeled entry, the source itself
    assert ConfigBroadcast.items(BASE) == [("", BASE)]


def test_scan_grid_precompute():
    """`scan` is a pure pre-compute: it builds the template's diff list."""
    from tabpwa.config_broadcast import scan

    diffs = scan({"data.purity": [1.0, 0.9],
                  "data.phsp_arr": ["a.npz", "b.npz"]})
    assert len(diffs) == 4                       # cartesian product
    assert diffs[0]["label"] == "purity=1.0&phsp_arr=a.npz"

    cb = ConfigBroadcast({**BASE, "broadcast": diffs})
    assert len(cb) == 4
    assert cb.is_broadcast and ConfigBroadcast.has_broadcast(cb)
    assert ConfigBroadcast.has_broadcast(BASE) is False

    labels = cb.labels()
    assert labels == ["purity=1.0&phsp_arr=a.npz",
                      "purity=1.0&phsp_arr=b.npz",
                      "purity=0.9&phsp_arr=a.npz",
                      "purity=0.9&phsp_arr=b.npz"]
    # container protocol
    assert list(cb) == cb.expand()
    assert cb[0] == cb.expand()[0] == cb.config(0)
    assert cb.pairs() == list(zip(labels, cb.expand()))

    # each variant is the template + the scanned keys
    c = cb[3]
    assert c["data"]["purity"] == 0.9 and c["data"]["phsp_arr"] == "b.npz"
    assert c["data"]["dat_order"] == ["pip", "pim", "eta"]

    # scalar values are treated as one-point lists
    cb1 = ConfigBroadcast({**BASE, "broadcast": scan({"data.purity": 0.5})})
    assert len(cb1) == 1 and cb1[0]["data"]["purity"] == 0.5
    assert ConfigBroadcast.scan is scan            # same pre-compute helper


def test_template_diff_list_and_order():
    """The diff list lives in the template (`broadcast:`)."""
    base = copy.deepcopy(BASE)
    base["broadcast"] = [{"data.purity": 0.9},
                         {"data.purity": 0.95, "label": "hi"}]
    cb = ConfigBroadcast(base)
    assert len(cb) == 2 and cb.labels() == ["cfg0", "hi"]
    assert [c["data"]["purity"] for c in cb] == [0.9, 0.95]
    # a template with no broadcast key is a single config
    assert ConfigBroadcast(BASE).expand() == [BASE]


def test_module_helpers_exported():
    from tabpwa.config_broadcast import scan, expand_configs  # noqa: F401
    assert callable(scan) and callable(expand_configs)


def test_helpers_reusable():
    d = {"data": {"a": 1}}
    ConfigBroadcast.merge_patch(d, {"data": {"b": 2}, "c.d": 3})
    assert d == {"data": {"a": 1, "b": 2}, "c": {"d": 3}}
    assert ConfigBroadcast.get_dotted(d, "c.d") == 3
    assert ConfigBroadcast.get_dotted(d, "c.x", default=-1) == -1
    assert ConfigBroadcast.resolve_value("f.npz", "/base") == "/base/f.npz"
    assert ConfigBroadcast.resolve_value("/abs/f.npz", "/base") == "/abs/f.npz"
    assert ConfigBroadcast.resolve_value(["a.npz", "b.npz"], "/base") == \
        ["/base/a.npz", "/base/b.npz"]

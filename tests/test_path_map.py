"""paths — global file-path remapping (bind_paths / path_scope).

The map is GLOBAL and external: configs (and expanded config dicts) are
never modified, and every load site sees the same bindings.
"""

import copy
import os

import numpy as np
import pytest
import yaml

from tabpwa import paths
from tabpwa.paths import (apply_path_map, merge_path_maps, parse_path_map,
                          resolve_path)

CFG = "tests/config_pwa_bw.yml"          # pure-PWA, no particle-model files
FINALS = ["pip", "pim", "eta"]
MASS = {"pip": 0.139567, "pim": 0.139567, "eta": 0.5478621409593663}


@pytest.fixture(autouse=True)
def _clean_paths(monkeypatch):
    """Isolate every test: empty registry, no env, no leftover scope."""
    paths.clear_paths()
    monkeypatch.delenv("TABPWA_PATH_MAP", raising=False)
    yield
    paths.clear_paths()


# ── helpers ───────────────────────────────────────────────────────

def _base_cfg():
    return yaml.safe_load(open(CFG))


def _momentum(n, seed=3):
    rs = np.random.RandomState(seed)
    p3 = rs.uniform(-0.4, 0.4, size=(n, 3, 3))
    out = np.empty((n, 3, 4))
    for j, nm in enumerate(FINALS):
        E = np.sqrt(MASS[nm] ** 2 + (p3[:, j] ** 2).sum(-1))
        out[:, j] = np.concatenate([E[:, None], p3[:, j]], axis=-1)
    return out


def _synth(fitter, n, seed):
    """Kernel-shaped synthetic arrays for the fitter's kernel config."""
    kc = fitter.kernel_config
    rs = np.random.RandomState(seed)
    n_at = int(np.max(kc["angle_index"])) + 1 if len(kc["angle_index"]) else 1
    nac = int(np.asarray(kc["angle_k"]).shape[-1])
    n_mass = int(np.max(kc["mass_index"])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1
    gmin = float(kc["gamma_min"])
    gspan = float(kc["gamma_delta"]) * (np.asarray(kc["gamma_table"]).shape[-1])
    return {"mass": (gmin + 0.05) + (0.9 * gspan - 0.1) * rs.rand(n, n_mass),
            "q": np.abs(rs.randn(n, n_q)) + 0.3,
            "angle": rs.uniform(-np.pi, np.pi, size=(n, n_at * nac)),
            "weight": np.ones(n), "bkg": np.ones(n)}


def _save_npz(path, d):
    np.savez(path, mass=d["mass"], q=d["q"], angle=d["angle"],
             weight=d["weight"], bkg_raw=d["bkg"])


def _write_gamma_table(path):
    """A ``(N, 3) = [m, Im Pi, Re Pi]`` table for ``width_linear_npy``."""
    m = np.linspace(0.0, 3.0, 500)
    np.save(path, np.column_stack(
        [m, 0.8 * np.exp(-((m - 0.769) / 0.3) ** 2) + 0.05,
         np.zeros_like(m)]))


# ── unit: parse forms ────────────────────────────────────────────

def test_parse_forms():
    assert parse_path_map(None) == {}
    assert parse_path_map("") == {}
    assert parse_path_map({"/a": "/b"}) == {"/a": "/b"}
    assert parse_path_map("/a:/b") == {"/a": "/b"}
    assert parse_path_map("/a:/b;/c:/d") == {"/a": "/b", "/c": "/d"}
    assert parse_path_map("/a:/b\n/c:/d") == {"/a": "/b", "/c": "/d"}
    assert parse_path_map(["/a:/b", "/c:/d"]) == {"/a": "/b", "/c": "/d"}
    assert parse_path_map(("/a:/b", "/c:/d")) == {"/a": "/b", "/c": "/d"}
    # a list entry may itself carry several ';'-separated pairs
    assert parse_path_map(["/a:/b;/c:/d"]) == {"/a": "/b", "/c": "/d"}
    # JSON is the escape hatch (e.g. ':' inside a path)
    assert parse_path_map(['{"/a": "/b"}']) == {"/a": "/b"}
    assert parse_path_map('{"/a": "/b", "/c": "/d"}') == {"/a": "/b",
                                                         "/c": "/d"}
    assert parse_path_map('{"C:\\\\data": "/mnt/c"}') == {"C:\\data": "/mnt/c"}
    # first ':' splits the pair; the rest belongs to the target
    assert parse_path_map("/a:/b:/c") == {"/a": "/b:/c"}
    # comments / blank lines in a file-style string
    assert parse_path_map("# c\n\n/a:/b\n") == {"/a": "/b"}
    # later duplicate FROM keys win
    assert parse_path_map(["/a:/1", "/a:/2"]) == {"/a": "/2"}

    for bad in ({"/a": ""}, {"/a": None}, {"": "/b"}, "/noequals",
                "/a", "/a:", ":/b"):
        with pytest.raises((ValueError, TypeError)):
            parse_path_map(bad)


def test_merge_path_maps_later_wins():
    assert merge_path_maps(None, {}, {"/a": "/1"},
                           {"/a": "/2", "/b": "/3"}) == {
        "/a": "/2", "/b": "/3"}
    assert merge_path_maps("/a:/1", {"/b": "/2"}) == {"/a": "/1", "/b": "/2"}
    # one delimited string == several CLI flags == a list
    assert merge_path_maps("/remote/a:/local/a;/remote/b:/local/b") == \
        merge_path_maps(["/remote/a:/local/a", "/remote/b:/local/b"])
    assert merge_path_maps(parse_path_map(["/remote/a:/local/a",
                                           "/remote/b:/local/b"])) == {
        "/remote/a": "/local/a", "/remote/b": "/local/b"}


def test_multi_map_longest_prefix_across_entries():
    pm = parse_path_map(["/remote/a:/local/a", "/remote/b:/local/b",
                         "/remote/a/deep:/local/deep"])
    assert apply_path_map("/remote/a/x.npz", pm) == "/local/a/x.npz"
    assert apply_path_map("/remote/b/y.npz", pm) == "/local/b/y.npz"
    # longest FROM prefix wins over a shorter one from another entry
    assert apply_path_map("/remote/a/deep/z.npz", pm) == "/local/deep/z.npz"
    # duplicate FROM: later map wins
    assert parse_path_map(["/remote/a:/first", "/remote/a:/second"]) == {
        "/remote/a": "/second"}
    assert merge_path_maps({"/remote/a": "/env"},
                           ["/remote/a:/cli"]) == {"/remote/a": "/cli"}


def test_longest_prefix_boundary_exact_and_no_match():
    pm = {"/data": "/local", "/data/run1": "/local/run1"}
    # longest prefix wins
    assert apply_path_map("/data/run1/a.npz", pm) == "/local/run1/a.npz"
    assert apply_path_map("/data/run2/a.npz", pm) == "/local/run2/a.npz"
    # component boundary: /database is NOT under /data
    assert apply_path_map("/database/a.npz", pm) == "/database/a.npz"
    # exact match
    assert apply_path_map("/data", pm) == "/local"
    # no match → unchanged
    assert apply_path_map("/other/a.npz", pm) == "/other/a.npz"
    # empty map / non-string pass through
    assert apply_path_map("/data/a", {}) == "/data/a"
    assert apply_path_map(None, pm) is None


def test_resolve_path_relative_base_expandvars(monkeypatch):
    monkeypatch.setenv("PM_TEST_ROOT", "/mnt/pm")
    assert resolve_path("in.npz", base_dir="/cfg") == "/cfg/in.npz"
    assert resolve_path("/abs/in.npz", base_dir="/cfg") == "/abs/in.npz"
    assert resolve_path("$PM_TEST_ROOT/in.npz") == "/mnt/pm/in.npz"
    assert resolve_path("~/in.npz") == os.path.join(
        os.path.expanduser("~"), "in.npz")
    assert resolve_path("/mnt/pm/in.npz",
                        path_map={"/mnt/pm": "/local"}) == "/local/in.npz"
    assert resolve_path(None) is None


# ── global registry: bind_paths / clear_paths / path_scope ───────

def test_bind_paths_and_clear_paths():
    assert paths.current_path_map() == {}
    paths.bind_paths("/remote/a:/local/a")
    assert paths.current_path_map() == {"/remote/a": "/local/a"}
    # later wins per key, new keys accumulate
    paths.bind_paths(["/remote/b:/local/b", "/remote/a:/local/a2"])
    assert paths.current_path_map() == {"/remote/a": "/local/a2",
                                        "/remote/b": "/local/b"}
    assert resolve_path("/remote/a/x.npz") == "/local/a2/x.npz"
    assert resolve_path("/remote/b/x.npz") == "/local/b/x.npz"
    paths.clear_paths()
    assert paths.current_path_map() == {}
    assert resolve_path("/remote/a/x.npz") == "/remote/a/x.npz"


def test_path_scope_layers_and_restores():
    paths.bind_paths({"/global": "/g"})
    assert paths.current_path_map() == {"/global": "/g"}
    with paths.path_scope({"/scoped": "/s"}):
        assert paths.current_path_map() == {"/global": "/g",
                                            "/scoped": "/s"}
        with paths.path_scope("/global:/g2"):
            assert paths.current_path_map()["/global"] == "/g2"
        # nested scope restored
        assert paths.current_path_map()["/global"] == "/g"
    assert paths.current_path_map() == {"/global": "/g"}
    # scope wins over global; both are gone/restored after the block
    assert resolve_path("/global/x") == "/g/x"


def test_precedence_explicit_gt_scope_gt_global_gt_env(monkeypatch):
    monkeypatch.setenv("TABPWA_PATH_MAP", "/a:/env")
    assert resolve_path("/a/x") == "/env/x"                 # env alone
    paths.bind_paths({"/a": "/global"})
    assert resolve_path("/a/x") == "/global/x"              # global > env
    with paths.path_scope({"/a": "/scope"}):
        assert resolve_path("/a/x") == "/scope/x"           # scope > global
        # an explicit argument wins over everything
        assert resolve_path("/a/x", path_map={"/a": "/explicit"}) == \
            "/explicit/x"
    assert resolve_path("/a/x") == "/global/x"


# ── RawConfig: directory anchoring, no config-side map ───────────

def test_rawconfig_resolves_against_config_dir(tmp_path):
    from tabpwa.config_loader import RawConfig

    p = tmp_path / "c.yml"
    p.write_text("data:\n  data: ./m.npy\n")
    rc = RawConfig(str(p))
    assert rc.resolve("./m.npy") == str(tmp_path / "m.npy")
    # a dict config has no anchor: relative paths stay relative
    assert RawConfig({"data": {"data": "./m.npy"}}).resolve("./m.npy") \
        == "m.npy"


def test_config_path_map_key_is_ignored(tmp_path):
    """No ``path_map`` support inside configs — external settings only."""
    from tabpwa.config_loader import RawConfig

    dic = {"path_map": {"/a": "/from_config"},
           "config": {"path_map": {"/a": "/from_config2"}}}
    rc = RawConfig(dic)
    assert rc.resolve("/a/x.npy") == "/a/x.npy"
    assert rc.dic is dic                      # nothing added / rewritten
    assert not hasattr(rc, "path_map")
    # even with a global binding, only the global map applies
    paths.bind_paths({"/a": "/global"})
    assert rc.resolve("/a/x.npy") == "/global/x.npy"


# ── Fitter integration: data arrays ─────────────────────────────

def test_fitter_dict_config_remaps_npz_data(tmp_path):
    from tabpwa import Fitter

    local = tmp_path / "local"
    local.mkdir()
    cfg = _base_cfg()
    proto = Fitter(cfg, backend="numpy_pwa")
    data = _synth(proto, 25, 2)
    phsp = _synth(proto, 40, 3)
    _save_npz(local / "data.npz", data)
    _save_npz(local / "phsp.npz", phsp)
    proto.free()

    cfg["data"] = {"dat_order": FINALS,
                   "data_arr": "/remote/data.npz",
                   "phsp_arr": "/remote/phsp.npz"}
    before = copy.deepcopy(cfg)
    with paths.path_scope({"/remote": str(local)}):
        f = Fitter(cfg, backend="numpy_pwa")
        d, p = f.load_all_data()
        assert np.allclose(d["weight"], data["weight"])
        assert np.allclose(d["mass"], data["mass"])
        assert np.allclose(p["mass"], phsp["mass"])
        f.free()
    # the config dict is never modified
    assert cfg == before
    assert f.config.dic is cfg


def test_fitter_uses_bound_global_map(tmp_path):
    from tabpwa import Fitter

    local = tmp_path / "local"
    local.mkdir()
    proto = Fitter(_base_cfg(), backend="numpy_pwa")
    d = _synth(proto, 12, 4)
    _save_npz(local / "data.npz", d)
    proto.free()

    cfg = _base_cfg()
    cfg["data"] = {"dat_order": FINALS, "data_arr": "/env_remote/data.npz"}
    paths.bind_paths({"/env_remote": str(local)})
    f = Fitter(cfg, backend="numpy_pwa")
    assert np.allclose(f.load_dataset("data")["mass"], d["mass"])
    f.free()


def test_load_momenta_conf_remaps_prefix_weight_and_bg(tmp_path):
    from tabpwa import Fitter

    local = tmp_path / "local"
    local.mkdir()
    mom = _momentum(30)
    w = np.linspace(0.5, 1.5, 30)
    bg = np.linspace(0.0, 0.1, 30)
    np.save(local / "mom.npy", mom)
    np.save(local / "w.npy", w)
    np.save(local / "bg.npy", bg)

    cfg = _base_cfg()
    cfg["data"] = {"dat_order": FINALS,
                   "data": "/remote/mom.npy",
                   "data_weight": "/remote/w.npy",
                   "data_bg_value": "/remote/bg.npy"}
    with paths.path_scope({"/remote": str(local)}):
        f = Fitter(cfg, backend="numpy_pwa")
        ev = f.load_dataset("data")
        assert ev["mass"].shape[0] == 30
        assert np.allclose(ev["weight"], w)
        assert np.allclose(ev["bkg"], bg)
        f.free()


def test_missing_path_error_names_original_and_mapped(tmp_path):
    from tabpwa import Fitter

    local = tmp_path / "local"
    local.mkdir()
    cfg = _base_cfg()
    cfg["data"] = {"dat_order": FINALS,
                   "data_arr": "/remote/missing.npz"}
    with paths.path_scope({"/remote": str(local)}):
        f = Fitter(cfg, backend="numpy_pwa")
        with pytest.raises(FileNotFoundError) as ei:
            f.load_dataset("data")
        msg = str(ei.value)
        assert "/remote/missing.npz" in msg
        assert str(local / "missing.npz") in msg
        assert "/remote" in msg                 # the map itself is shown
        f.free()


def test_one_global_map_fixes_data_weight_and_particle_file(tmp_path):
    """One binding covers a data prefix, its weight sidecar and a particle
    model's ``file`` — the single ``resolve_path`` helper serves all."""
    from tabpwa import Fitter

    local = tmp_path / "local"
    local.mkdir()
    remote = str(tmp_path / "remote_tree")      # never exists
    mom = _momentum(20)
    w = np.linspace(0.4, 1.4, 20)
    np.save(local / "mom.npy", mom)
    np.save(local / "w.npy", w)
    _write_gamma_table(local / "mi0p_gamma.npy")

    cfg = _base_cfg()
    cfg["particle"]["MI0p"].update({"model": "width_linear_npy",
                                    "width": 0.15,
                                    "file": f"{remote}/mi0p_gamma.npy"})
    cfg["data"] = {"dat_order": FINALS,
                   "data": f"{remote}/mom.npy",
                   "data_weight": f"{remote}/w.npy"}
    before = copy.deepcopy(cfg)

    paths.bind_paths({remote: str(local)})
    f = Fitter(cfg, backend="numpy_pwa")
    ev = f.load_dataset("data")
    assert np.allclose(ev["weight"], w)
    assert f.model.dic["particle"]["MI0p"]["file"] == \
        str(local / "mi0p_gamma.npy")
    # the config dict itself is untouched (no materialised paths)
    assert cfg == before
    f.free()


# ── ConfigBroadcast / load_fit see the same global map ───────────

def test_config_broadcast_remaps_variants_without_touching_config(tmp_path):
    from tabpwa.config_broadcast import ConfigBroadcast

    local = tmp_path / "local"
    cfg = _base_cfg()
    cfg["data"] = {"dat_order": FINALS,
                   "phsp_arr": "/remote/phsp.npz"}
    cfg["broadcast"] = [{"data": {"data_arr": "/remote/a.npz"}},
                        {"data": {"data_arr": "/remote/b.npz"},
                         "label": "B"}]
    before = copy.deepcopy(cfg)

    with paths.path_scope({"/remote": str(local)}):
        cb = ConfigBroadcast(cfg)
        v0, v1 = cb.expand()
        assert v0["data"]["data_arr"] == str(local / "a.npz")
        assert v1["data"]["data_arr"] == str(local / "b.npz")
        assert v0["data"]["phsp_arr"] == str(local / "phsp.npz")
        # no injection of the map into the expanded configs
        assert "path_map" not in v0 and "path_map" not in v1
        # items() is the pass-through the joint fitter uses
        pairs = ConfigBroadcast.items(cfg)
        assert pairs[0][1]["data"]["data_arr"] == str(local / "a.npz")
    # the template itself is untouched
    assert cfg == before


def test_load_fit_uses_the_global_map(tmp_path):
    from tabpwa import load_fit

    local = tmp_path / "local"
    cfg = _base_cfg()
    cfg["data"] = {"dat_order": FINALS}
    cfg["broadcast"] = [{"data": {"data_arr": "/remote/a.npz"}}]
    paths.bind_paths({"/remote": str(local)})
    joint = load_fit(cfg, backend="numpy_pwa", load_data=False)
    assert joint.fitters[0].config.dic["data"]["data_arr"] == \
        str(local / "a.npz")
    joint.free()


def test_extensionless_config_value_finds_npz_suffix(tmp_path):
    """A config value without suffix resolves to the real ``.npz``/``.npy``."""
    from tabpwa import Fitter

    local = tmp_path / "local"
    local.mkdir()
    proto = Fitter(_base_cfg(), backend="numpy_pwa")
    d = _synth(proto, 8, 5)
    _save_npz(local / "data.npz", d)
    proto.free()

    cfg = _base_cfg()
    cfg["data"] = {"dat_order": FINALS, "data_arr": "/remote/data"}
    with paths.path_scope({"/remote": str(local)}):
        f = Fitter(cfg, backend="numpy_pwa")
        ev = f.load_dataset("data")
        assert np.allclose(ev["mass"], d["mass"])
        f.free()

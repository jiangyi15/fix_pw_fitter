"""Fitter.load_dataset / load_npz: prefix loader, sidecars, ordering."""

import os

import numpy as np
import pytest

CONFIG_TEMPLATE = "tests/config_pwa.yml"     # repo-root pure-PWA config (read-only)
F = ["pip", "pim", "eta"]


def _momentum(n, seed=3):
    rs = np.random.RandomState(seed)
    p3 = rs.uniform(-0.4, 0.4, size=(n, 3, 3))
    m = {"pip": 0.13957, "pim": 0.13957, "eta": 0.54786}
    out = np.empty((n, 3, 4))
    for j, nm in enumerate(F):
        E = np.sqrt(m[nm] ** 2 + (p3[:, j] ** 2).sum(-1))
        out[:, j] = np.concatenate([E[:, None], p3[:, j]], axis=-1)
    return out


@pytest.fixture()
def cfg(tmp_path):
    text = open(CONFIG_TEMPLATE).read()
    mom = _momentum(40)
    data = tmp_path / "data_mom.npy"
    phsp = tmp_path / "phsp_mom.npy"
    np.save(data, mom)
    np.save(phsp, _momentum(60, seed=5))
    np.save(tmp_path / "w.npy", np.linspace(0.5, 1.5, 40))
    np.save(tmp_path / "bg.npy", np.linspace(0.0, 0.1, 40))

    def strip_old(yml):
        yml = yml.replace("    data_weight: ./phsp_weight_slice.npy\n", "")
        return yml

    def patch(yml, extra=""):
        yml = strip_old(yml)
        return yml.replace(
            "    data: ./data_slice.npy",
            f"    data: {data}\n    data_weight: {tmp_path}/w.npy\n"
            f"    data_bg_value: {tmp_path}/bg.npy{extra}") \
            .replace("    phsp: ./phsp_slice.npy", f"    phsp: {phsp}")

    cfg_file = tmp_path / "c.yml"
    cfg_file.write_text(patch(text))
    from tabpwa.config_loader import Config
    return Config(str(cfg_file))


def _fitter(cfg):
    from tabpwa import Fitter
    return Fitter(cfg._config_path, backend="numpy_pwa")


def test_load_dataset_momenta_prefix(cfg):
    f = _fitter(cfg)
    ev = f.load_dataset("data")
    assert ev["mass"].shape[0] == 40
    assert set(("mass", "q", "angle", "weight", "bkg")) <= set(ev)
    assert np.allclose(ev["weight"], np.linspace(0.5, 1.5, 40))
    assert np.allclose(ev["bkg"], np.linspace(0.0, 0.1, 40))
    f.free()


def test_load_dataset_bg_length_mismatch(cfg, tmp_path):
    np.save(tmp_path / "short.npy", np.zeros(7))
    text = open(CONFIG_TEMPLATE).read()
    mom = _momentum(20)
    data = tmp_path / "d2.npy"
    np.save(data, mom)
    cfg_file = tmp_path / "c2.yml"
    text = text.replace("    data_weight: ./phsp_weight_slice.npy\n", "")
    cfg_file.write_text(text.replace("    data: ./data_slice.npy",
                                     f"    data: {data}\n"
                                     f"    data_bg_value: {tmp_path}/short.npy"))
    from tabpwa.config_loader import Config
    f = _fitter(Config(str(cfg_file)))
    with pytest.raises(ValueError):
        f.load_dataset("data")
    f.free()


def test_load_dataset_dat_order_permutation(cfg, tmp_path):
    """A permuted momenta file + matching dat_order reloads identically."""
    from tabpwa.config_loader import Config
    from tabpwa.pwa_build import pwa_event_data_tree

    f1 = _fitter(cfg)
    base = f1.load_dataset("data")

    perm = ["pim", "pip", "eta"]           # columns 0,1 swapped
    text = open(CONFIG_TEMPLATE).read()
    mom = _momentum(30, seed=9)
    permuted = mom[:, [1, 0, 2]]
    data = tmp_path / "d3.npy"
    np.save(data, permuted)
    cfg_file = tmp_path / "c3.yml"
    text = text.replace("    data_weight: ./phsp_weight_slice.npy\n", "")
    cfg_file.write_text(text.replace("    dat_order: [pip, pim, eta]",
                                     "    dat_order: [pim, pip, eta]")
                        .replace("    data: ./data_slice.npy",
                                 f"    data: {data}"))
    c3 = Config(str(cfg_file))
    f3 = _fitter(c3)
    ev3 = f3.load_dataset("data")

    # direct tree conversion of the correctly-ordered momenta
    kc = f1.kernel_config
    pws = list(f1.full_decay.get_partial_waves())
    byt = {f1.decay_tree.topo_index[ch.topo_id()]: ch for _, ch in pws}
    expect = pwa_event_data_tree(f1.model, kc, byt, mom)
    assert np.allclose(ev3["mass"], expect["mass"])
    assert np.allclose(ev3["angle"], expect["angle"])
    f1.free(); f3.free()


def test_load_npz_max_events_takes_first_rows(tmp_path):
    from tabpwa import Fitter
    rs = np.random.RandomState(0)
    n = 60
    npz = tmp_path / "arr.npz"
    np.savez(npz, mass=rs.rand(n, 1), q=rs.rand(n, 2),
             angle=rs.rand(n, 1, 3), weight=rs.rand(n))
    d, ne = Fitter.load_npz(str(npz), max_events=25, n_angle_comp=3)
    assert ne == 25
    full, _ = Fitter.load_npz(str(npz), n_angle_comp=3)
    assert d["mass"].shape[0] == 25
    assert np.allclose(d["mass"], full["mass"][:25])
    assert np.allclose(d["weight"], full["weight"][:25])


def test_load_dataset_explicit_path_and_max_events(cfg, tmp_path):
    """path= overrides the config file; max_events keeps the first rows."""
    f = _fitter(cfg)
    full = f.load_dataset("data")                         # from the config
    n = full["mass"].shape[0]
    assert n == 40

    # a different kernel-array file for the same prefix
    arr = tmp_path / "other.npz"
    np.savez(arr, mass=full["mass"][:7], q=full["q"][:7],
             angle=full["angle"][:7], weight=full["weight"][:7])
    got = f.load_dataset("data", path=str(arr))
    assert got["mass"].shape[0] == 7
    assert np.allclose(got["mass"], full["mass"][:7])

    # max_events on the config path and on the explicit path
    cut = f.load_dataset("data", max_events=11)
    assert cut["mass"].shape[0] == 11
    assert np.allclose(cut["mass"], full["mass"][:11])
    cut2 = f.load_dataset("data", path=str(arr), max_events=4)
    assert cut2["mass"].shape[0] == 4
    f.free()


def test_load_dataset_path_momenta_slices_sidecars(cfg, tmp_path):
    """A momenta .npy override is converted; weight/bg follow max_events."""
    f = _fitter(cfg)
    conf = f.config.dic["data"]
    mom = np.load(str(conf["data"]))
    small = tmp_path / "mom_small.npy"
    np.save(small, mom[:15])

    got = f.load_dataset("data", path=str(small))
    assert got["mass"].shape[0] == 15
    # the config's sidecars are longer than the new sample: sliced, not error
    assert got["weight"].shape[0] == 15
    assert got["bkg"].shape[0] == 15
    f.free()


def test_load_dataset_path_rejects_unknown_suffix(cfg, tmp_path):
    f = _fitter(cfg)
    bad = tmp_path / "x.txt"
    bad.write_text("nope")
    with pytest.raises(ValueError, match="expected a"):
        f.load_dataset("data", path=str(bad))
    with pytest.raises(FileNotFoundError):
        f.load_dataset("data", path=str(tmp_path / "missing.npz"))
    f.free()


def test_load_dataset_unconfigured_prefix_returns_none(cfg):
    f = _fitter(cfg)
    assert f.load_dataset("phsp_rec") is None
    f.free()

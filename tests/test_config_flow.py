"""Config round-trip: ``to_dict`` -> assign items -> reload a new Fitter.

The reload flow is deliberately plain dict manipulation — no mutation API on
the fitter/config: get the config dict, assign items (add/remove resonances),
then build a new :class:`~tabpwa.Fitter` from it.
"""
from tabpwa import Fitter

CFG = "tests/config_pwa.yml"


def _reload(dic):
    return Fitter(dic, backend="numpy_pwa")


def test_untouched_reload_is_identical():
    f = Fitter(CFG, backend="numpy_pwa")
    g = _reload(f.config.to_dict())
    assert g.all_comb == f.all_comb
    assert g.param_names() == f.param_names()


def test_add_resonance_by_assignment():
    f = Fitter(CFG, backend="numpy_pwa")
    n0 = len(f.all_comb)
    dic = f.config.to_dict()
    dic["particle"]["NR"] = dict(J=1, P=-1, mass=0.9, width=0.1, model="BW")
    dic["particle"]["pipi"].append("NR")
    g = _reload(dic)
    assert len(g.all_comb) > n0                    # new partial wave(s)
    x = g.initial_values(seed=1)
    params, _ = g.build_params(x)
    assert len(params["ck"]) == len(g.all_comb)


def test_remove_resonance_restores():
    f = Fitter(CFG, backend="numpy_pwa")
    n0 = len(f.all_comb)
    dic = f.config.to_dict()
    dic["particle"]["NR"] = dict(J=1, P=-1, mass=0.9, width=0.1, model="BW")
    dic["particle"]["pipi"].append("NR")
    del dic["particle"]["NR"]
    dic["particle"]["pipi"] = [p for p in dic["particle"]["pipi"] if p != "NR"]
    g = _reload(dic)
    assert len(g.all_comb) == n0


def test_dump_reload_roundtrip(tmp_path):
    f = Fitter(CFG, backend="numpy_pwa")
    path = f.config.dump(str(tmp_path / "cfg.yml"))
    g = Fitter(path, backend="numpy_pwa")
    assert g.all_comb == f.all_comb
    assert g.param_names() == f.param_names()


def test_parity_forbidden_resonance_adds_no_wave():
    # J=0, P=+1 -> pip pi needs L=0 but the parity filter requires odd L,
    # so the chain is empty and the wave count is unchanged.
    f = Fitter(CFG, backend="numpy_pwa")
    n0 = len(f.all_comb)
    dic = f.config.to_dict()
    dic["particle"]["NR0p"] = dict(J=0, P=1, mass=0.9, width=0.1,
                                   model="BW")
    dic["particle"]["pipi"].append("NR0p")
    g = _reload(dic)
    assert len(g.all_comb) == n0

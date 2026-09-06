"""Tests for the ``python -m ampfit`` entry point."""

import importlib
import os


def _main_mod():
    return importlib.import_module("ampfit.__main__")


def test_resolve_run_fit_and_scripts():
    m = _main_mod()
    rp = m._resolve("run_fit")
    assert rp and os.path.basename(rp) == "run_fit.py"
    pp = m._resolve("plot_pwa_groups")
    assert pp and pp.endswith(os.path.join("scripts", "plot_pwa_groups.py"))
    assert m._resolve("does_not_exist_xyz") is None


def test_available_includes_run_fit():
    m = _main_mod()
    avail = m._available()
    assert "run_fit" in avail
    assert "plot_pwa_groups" in avail


def test_main_lists_commands(capsys):
    m = _main_mod()
    rc = m.main(["--list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "run_fit" in out


def test_main_unknown_command(capsys):
    m = _main_mod()
    rc = m.main(["definitely_not_a_command_zz"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown command" in err

"""The ``mpi_service`` backend: the PROGRAM runs once; MPI lives in a child.

What is pinned: the parent imports no ``mpi4py`` and has no rank; the results
match a serial backend; the child is started once and reused; a killed child
raises instead of hanging; nesting and a non-leaf ``local`` are refused.
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import create_backend

CFG = "tests/config_pwa.yml"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPIEXEC = shutil.which("mpiexec") or shutil.which("mpirun")
#: only the gates that actually START a child need the launcher; the boundary
#: rules (nesting, n_procs, a non-leaf local, no mpi4py in the parent) must run
#: on a box without MPI too, which is where they matter most
needs_launcher = pytest.mark.skipif(MPIEXEC is None, reason="no mpiexec on PATH")


@pytest.fixture(scope="module")
def sample():
    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    x0 = f.initial_values(seed=1)
    return f, x0, f.generate_phsp(240, seed=11), f.generate_toy(x0, 160, seed=5)


def _service(**kw):
    spec = {"name": "mpi_service", "local": "numpy_pwa", "n_procs": 2,
            "device": None, "timeout": 30.0}
    spec.update(kw)
    return spec


@needs_launcher
def test_the_program_runs_once_and_matches_serial(sample):
    f, x0, phsp, data = sample
    ref = Fitter(CFG, backend="numpy_pwa")
    ref.apply_constrains()
    ref.set_phsp(phsp)
    ref.set_data(data)
    q_ref, g_ref = ref.get_nll(x0)

    s = Fitter(CFG, backend=_service())
    s.apply_constrains()
    try:
        s.set_phsp(phsp)
        s.set_data(data)
        pid = s.backend._proc.pid
        q, g = s.get_nll(x0)
        assert q == pytest.approx(q_ref, rel=1e-12)
        assert np.allclose(g, g_ref, rtol=1e-12, atol=1e-12)
        # the child is reused for the next evaluation, not restarted
        s.get_nll(x0)
        assert s.backend._proc.pid == pid
        assert "running" in s.backend.describe()
    finally:
        s.free()
    assert s.backend._proc is None


def test_the_parent_never_imports_mpi4py(tmp_path):
    """No mpi4py, no rank in the program — the whole MPI job is the child."""
    code = f"""
import sys
import tabpwa
assert "mpi4py" not in sys.modules, "import tabpwa pulled in mpi4py"
from tabpwa import Fitter
f = Fitter({CFG!r}, backend={{"name": "mpi_service", "local": "numpy_pwa",
                              "n_procs": 2, "device": None}})
f.apply_constrains()
assert "mpi4py" not in sys.modules, "constructing the backend pulled in mpi4py"
assert not hasattr(f.backend, "rank"), "the driver must not expose a rank"
print("OK")
"""
    env = dict(os.environ)
    path = str(os.path.join(ROOT, "src"))
    if env.get("PYTHONPATH"):
        path += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = path
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "OK" in out.stdout


@needs_launcher
def test_a_killed_child_raises_instead_of_hanging(sample):
    f, x0, phsp, data = sample
    s = Fitter(CFG, backend=_service())
    s.apply_constrains()
    try:
        s.set_phsp(phsp)
        params, _ = s.build_params(x0)
        be = s.backend
        be._proc.kill()
        be._proc.wait()
        with pytest.raises(RuntimeError, match="worker"):
            be.compute(params, s._phsp_holder, norm=None)
    finally:
        s.free()


@needs_launcher
def test_p_and_a_reload_survive_the_service(sample):
    """``return_p`` comes back in order, and a freed handle can be re-loaded.

    The per-event ``P`` is packed only when asked (the fitter never asks), and
    ``handle.free()`` drops ONE sample without stopping the child: the next
    ``load_data`` reuses it, which is the restart-on-demand path the ``Fitter``
    needs when a dataset is replaced.
    """
    f, x0, phsp, data = sample
    ref = Fitter(CFG, backend="numpy_pwa")
    ref.apply_constrains()
    params, _ = ref.build_params(x0)
    q_ref, _g, p_ref = ref.backend.compute(
        params, ref.backend.load_data(dict(phsp)), norm=None, return_p=True)

    s = Fitter(CFG, backend=_service())
    s.apply_constrains()
    try:
        be = s.backend
        h = be.load_data(dict(phsp))
        pid = be._proc.pid
        q1, _g1, p1 = be.compute(params, h, norm=None, return_p=True)
        assert q1 == pytest.approx(q_ref, rel=1e-12)
        assert np.array_equal(p1, p_ref)
        h.free()                                  # drop the sample, keep the job
        h2 = be.load_data(dict(phsp))
        q2, _g2, p2 = be.compute(params, h2, norm=None, return_p=True)
        assert be._proc.pid == pid                # the child was reused
        assert q2 == pytest.approx(q1, rel=1e-15)
        assert np.array_equal(p2, p1)
        h2.free()
    finally:
        s.free()
    # after free() there is no silent restart: the backend is CLOSED
    with pytest.raises(RuntimeError, match="shut down"):
        s.backend.compute(params, s.backend.load_data(dict(phsp)), norm=None)


@needs_launcher
def test_a_second_load_with_another_config_is_refused(sample):
    """One service serves ONE model: a changed kernel config aborts loudly."""
    f, x0, phsp, data = sample
    s = Fitter(CFG, backend=_service())
    s.apply_constrains()
    try:
        be = s.backend
        be.load_data(dict(phsp))
        be.kernel_config = dict(be.kernel_config)
        be.kernel_config["n_blocks"] = int(be.kernel_config["n_blocks"]) + 1
        with pytest.raises(RuntimeError, match="DIFFERENT kernel config"):
            be.load_data(dict(phsp))
    finally:
        s.free()


def test_nesting_inside_a_step_is_refused(monkeypatch):
    monkeypatch.setenv("PMI_RANK", "1")
    with pytest.raises(RuntimeError, match="already inside"):
        Fitter(CFG, backend=_service())


def test_n_procs_is_required(tmp_path):
    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    with pytest.raises(ValueError, match="n_procs is required"):
        create_backend({"name": "mpi_service", "local": "numpy_pwa"},
                       f.kernel_config, model="pwa")


def test_a_composer_cannot_be_the_local_backend(sample):
    f, x0, phsp, data = sample
    with pytest.raises(ValueError, match="ONE concrete kernel"):
        create_backend({"name": "mpi_service", "local": "auto", "n_procs": 2},
                       f.kernel_config, model="pwa")
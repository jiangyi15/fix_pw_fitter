"""The ``mpi`` backend: one slice per rank, reduced with ``allreduce``.

Single rank (in-process, mpi4py singleton) must be BIT-identical to the local
backend; two ranks under ``mpiexec`` must match the serial result up to
association; the per-event ``P`` is gathered to rank 0 only; and the launch
guards (device index, nested ``shard``) refuse loudly.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

pytest.importorskip("mpi4py")

from tabpwa import Fitter
from tabpwa.backends import create_backend
from tabpwa.backends.mpi_backend import device_index

CFG = "tests/config_pwa.yml"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPIEXEC = shutil.which("mpiexec") or shutil.which("mpirun")


@pytest.fixture(scope="module")
def sample():
    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    x0 = f.initial_values(seed=1)
    return f, x0, f.build_params(x0)[0], f.generate_phsp(240, seed=11)


def test_device_index_rules():
    assert device_index(0, None, 4, None) is None
    assert device_index(2, "auto", 4, None) == "2"
    assert device_index(3, "auto", 4, "") is None          # hides every device
    assert device_index(3, "auto", 4, "0") is None         # this task's own GPU
    assert device_index(1, "auto", 8, "0,2,3") == "2"      # node-level list
    assert device_index(3, "auto", 0, None) is None        # no GPU on this host
    assert device_index(1, 0, 4, None) == "0"              # explicit device
    with pytest.raises(ValueError, match="only 2 GPU"):
        device_index(2, "auto", 2, None)
    with pytest.raises(ValueError, match="has no device"):
        device_index(3, "auto", 8, "0,1")


def test_a_node_level_gpu_list_binds_the_node_local_rank(sample, monkeypatch):
    """Slurm's `--gres=gpu:N` exports every device to every task; index it."""
    f, x0, params, phsp = sample
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    create_backend({"name": "mpi", "local": "numpy_pwa", "device": "auto"},
                   f.kernel_config, model="pwa")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"     # local rank 0 -> 3


def test_one_rank_is_bit_identical_to_the_local_backend(sample):
    f, x0, params, phsp = sample
    local = create_backend("numpy_pwa", f.kernel_config, model="pwa")
    mpi = create_backend({"name": "mpi", "local": "numpy_pwa",
                          "device": None}, f.kernel_config, model="pwa")
    assert mpi.size == 1
    h_local = local.load_data(phsp)
    h_mpi = mpi.load_data(phsp)
    assert (h_mpi.lo, h_mpi.hi) == (0, len(phsp["mass"]))

    qa, ga, pa = local.compute(params, h_local, norm=None, return_p=True)
    qb, gb, pb = mpi.compute(params, h_mpi, norm=None, return_p=True)
    assert qa == qb
    assert np.array_equal(pa, pb)
    for key in ga:
        assert np.array_equal(np.asarray(ga[key]), np.asarray(gb[key])), key


def test_align_follows_the_local_grouped_objective(sample):
    """A group must never straddle two ranks: derived, not guessed."""
    f, x0, params, phsp = sample
    grouped = Fitter("tests/config_pwa_gen.yml", backend="numpy_flat")
    grouped.apply_constrains()
    be = create_backend({"name": "mpi", "device": None,
                         "local": {"name": "numpy_flat",
                                   "resolution_size": 4}},
                        grouped.kernel_config, model="pwa_gen")
    assert be.align == 4

    # the v5 family stores only `_resolution_size` (no public attribute): the
    # derivation must still see it, or a grouped fit is silently 1.6 % wrong
    v5 = create_backend({"name": "mpi", "device": None,
                         "local": {"name": "cpu_v5_pwa",
                                   "resolution_size": 4}},
                        f.kernel_config, model="pwa")
    assert v5.align == 4
    flat = create_backend({"name": "mpi", "local": "numpy_pwa",
                           "device": None}, f.kernel_config, model="pwa")
    assert flat.align == 1


def test_an_mpi_rank_inside_a_shard_worker_is_refused(sample, monkeypatch):
    f, x0, params, phsp = sample
    monkeypatch.setenv("TABPWA_SHARD_WORKER", "1")
    with pytest.raises(ValueError, match="inside a `shard` worker"):
        create_backend({"name": "mpi", "local": "numpy_pwa", "device": None},
                       f.kernel_config, model="pwa")


def test_a_composer_cannot_be_the_local_backend(sample):
    """One concrete kernel per rank: `auto`/`integrated*`/`fixed_coeff` refused."""
    f, x0, params, phsp = sample
    for spec in ("auto", {"name": "integrated_pwa"},
                 {"name": "fixed_coeff",
                  "base": "numpy_pwa"}):
        with pytest.raises(ValueError, match="ONE concrete kernel"):
            create_backend({"name": "mpi", "local": spec, "device": None},
                           f.kernel_config, model="pwa")


def test_nesting_shard_is_refused(sample):
    f, x0, params, phsp = sample
    with pytest.raises(ValueError, match="nesting `shard`"):
        create_backend({"name": "mpi", "device": None,
                        "local": {"name": "shard", "backends": "numpy_pwa"}},
                       f.kernel_config, model="pwa")


@pytest.mark.skipif(MPIEXEC is None, reason="no mpiexec on PATH")
def test_two_ranks_match_the_serial_result(tmp_path, sample):
    script = tmp_path / "run_mpi.py"
    script.write_text(textwrap.dedent(f'''
        import json
        import numpy as np
        from tabpwa import Fitter
        from tabpwa.backends import create_backend

        f = Fitter({CFG!r}, backend="numpy_pwa")
        f.apply_constrains()
        x0 = f.initial_values(seed=1)
        params, _ = f.build_params(x0)
        phsp = f.generate_phsp(240, seed=11)
        be = create_backend({{"name": "mpi", "local": "numpy_pwa",
                              "device": None}}, f.kernel_config, model="pwa")
        h = be.load_data(phsp)
        q, g, p = be.compute(params, h, norm=None, return_p=True)
        if be.rank == 0:
            qr, gr, pr = f.backend.compute(
                params, f.backend.load_data(phsp), norm=None, return_p=True)
            print("JSON" + json.dumps({{
                "size": be.size, "n": int(h.n), "lo": h.lo, "hi": h.hi,
                "dq": abs(float(q) - float(qr)),
                "dg": {{k: float(np.max(np.abs(np.asarray(g[k])
                                              - np.asarray(gr[k]))))
                        for k in gr}},
                "dp": float(np.max(np.abs(np.asarray(p) - np.asarray(pr)))),
                "np": int(len(p)),
            }}))
    '''))
    env = dict(os.environ, OMPI_MCA_rmaps_base_oversubscribe="1")
    out = subprocess.run([MPIEXEC, "-n", "2", sys.executable, str(script)],
                         cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("JSON")][-1]
    res = json.loads(line[4:])
    assert res["size"] == 2 and res["np"] == res["n"]
    assert res["dq"] < 1e-10
    assert max(res["dg"].values()) < 1e-10
    assert res["dp"] < 1e-12
    # rank 0 owns a strict subset: the shard boundaries are real
    assert res["hi"] - res["lo"] < res["n"]


def _mpi_run(tmp_path, name, body, timeout=120, extra_env=None):
    script = tmp_path / f"{name}.py"
    script.write_text(textwrap.dedent(body))
    env = dict(os.environ, OMPI_MCA_rmaps_base_oversubscribe="1")
    if extra_env:
        env.update(extra_env)
    start = time.monotonic()
    out = subprocess.run([MPIEXEC, "-n", "2", sys.executable, str(script)],
                         cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=timeout)
    return out, time.monotonic() - start


@pytest.mark.skipif(MPIEXEC is None, reason="no mpiexec on PATH")
def test_a_grouped_objective_matches_the_serial_fit(tmp_path):
    """`align` must follow the v5 family's group size, not default to 1."""
    out, _ = _mpi_run(tmp_path, "grouped", f'''
        import json
        import numpy as np
        from tabpwa import Fitter

        cfg = {CFG!r}
        spec = {{"name": "cpu_v5_pwa", "resolution_size": 4}}
        gen = Fitter(cfg, backend=spec)
        gen.apply_constrains()
        x0 = gen.initial_values(seed=1)
        phsp = gen.generate_phsp(30, seed=11)
        data = gen.generate_toy(x0, 14, seed=5)

        f = Fitter(cfg, backend={{"name": "mpi", "local": spec,
                                  "device": None}})
        f.apply_constrains()
        f.set_phsp(phsp)
        f.set_data(data)
        q, g = f.get_nll(x0)
        if f.backend.rank == 0:
            ref = Fitter(cfg, backend=spec)
            ref.apply_constrains()
            ref.set_phsp(phsp)
            ref.set_data(data)
            qr, gr = ref.get_nll(x0)
            print("JSON" + json.dumps({{
                "align": f.backend.align, "size": f.backend.size,
                "dq": abs(float(q) - float(qr)),
                "dg": float(np.max(np.abs(np.asarray(g) - np.asarray(gr)))),
            }}))
    ''')
    assert out.returncode == 0, out.stderr[-3000:]
    res = json.loads([ln for ln in out.stdout.splitlines()
                      if ln.startswith("JSON")][-1][4:])
    assert res["align"] == 4                    # the v5 private attribute
    assert res["size"] == 2
    assert res["dq"] < 1e-9 and res["dg"] < 1e-9


@pytest.mark.skipif(MPIEXEC is None, reason="no mpiexec on PATH")
def test_the_no_gpu_refusal_ends_the_job_fast(tmp_path):
    """A refused rank must Abort, not leave the peers in a Barrier (was 45 s).

    The GPU count is PATCHED in the child: the test is about the abort path, not
    about how many devices this box has (under a loaded suite ``nvidia-smi`` can
    report none, in which case `auto` rightly leaves the binding alone).
    """
    out, elapsed = _mpi_run(tmp_path, "device", f'''
        import tabpwa.backends.mpi_backend as mb
        mb._gpu_count = lambda: 1          # two ranks, one device -> rank 1 refused
        from tabpwa import Fitter
        from tabpwa.backends import create_backend
        f = Fitter({CFG!r}, backend="numpy_pwa")
        f.apply_constrains()
        be = create_backend({{"name": "mpi", "local": "numpy_pwa",
                              "device": "auto"}}, f.kernel_config, model="pwa")
        print("PROCEEDED")
    ''', timeout=60)
    assert out.returncode != 0
    assert "asks for device 1" in out.stderr
    # rank 0 may proceed and finish; what matters is that the JOB ends fast
    # instead of waiting in a Barrier (measured 45 s before the abort wrapper)
    assert elapsed < 30, elapsed


@pytest.mark.skipif(MPIEXEC is None, reason="no mpiexec on PATH")
def test_an_empty_slice_is_refused_loudly(tmp_path):
    """A rank with no rows cannot compute; it must not hang the others."""
    out, elapsed = _mpi_run(tmp_path, "empty", f'''
        from tabpwa import Fitter
        from tabpwa.backends import create_backend
        f = Fitter({CFG!r}, backend="numpy_pwa")
        f.apply_constrains()
        be = create_backend({{"name": "mpi", "local": "numpy_pwa",
                              "device": None, "weights": [1.0, 1000.0]}},
                            f.kernel_config, model="pwa")
        be.load_data(f.generate_phsp(10, seed=11))
        print("PROCEEDED")
    ''', timeout=60)
    assert out.returncode != 0
    assert "owns no rows" in out.stderr
    assert elapsed < 30, elapsed


@pytest.mark.skipif(MPIEXEC is None, reason="no mpiexec on PATH")
def test_an_unseeded_start_is_broadcast_from_rank_0(tmp_path):
    """`initial_values(seed=None)` differs per process; the backend must fix it."""
    out, _ = _mpi_run(tmp_path, "seedless", f'''
        from tabpwa import Fitter

        cfg = {CFG!r}
        gen = Fitter(cfg, backend="numpy_pwa")
        gen.apply_constrains()
        phsp = gen.generate_phsp(60, seed=11)
        data = gen.generate_toy(gen.initial_values(seed=1), 40, seed=5)

        f = Fitter(cfg, backend={{"name": "mpi", "local": "numpy_pwa",
                                  "device": None}})
        f.apply_constrains()
        x0 = f.initial_values(seed=None)        # unseeded: differs per rank
        f.set_phsp(phsp)
        f.set_data(data)
        q, _g = f.get_nll(x0)
        print("Q" + str(f.backend.rank) + "=" + f"{{float(q):.12f}}")
    ''')
    assert out.returncode == 0, out.stderr[-3000:]
    qs = {ln.split("=")[0][1:]: ln.split("=")[1]
          for ln in out.stdout.splitlines() if ln.startswith("Q")}
    assert set(qs) == {"0", "1"}, out.stdout
    assert len(set(qs.values())) == 1, qs


@pytest.mark.skipif(MPIEXEC is None, reason="no mpiexec on PATH")
def test_a_generated_toy_is_identical_on_every_rank(tmp_path):
    """`P` is allgathered, so the accept-reject decisions stay in lockstep."""
    script = tmp_path / "run_toy.py"
    script.write_text(textwrap.dedent(f'''
        import hashlib
        import numpy as np
        from tabpwa import Fitter

        cfg = {CFG!r}
        f = Fitter(cfg, backend={{"name": "mpi", "local": "numpy_pwa",
                                  "device": None}})
        f.apply_constrains()
        x0 = f.initial_values(seed=1)
        d = f.generate_toy(x0, 30, seed=5)
        h = hashlib.sha256(np.ascontiguousarray(d["mass"]).tobytes()).hexdigest()
        print("HASH" + str(f.backend.rank) + "=" + h)
    '''))
    env = dict(os.environ, OMPI_MCA_rmaps_base_oversubscribe="1")
    out = subprocess.run([MPIEXEC, "-n", "2", sys.executable, str(script)],
                         cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    hashes = {ln.split("=")[0][4:]: ln.split("=")[1]
              for ln in out.stdout.splitlines() if ln.startswith("HASH")}
    assert set(hashes) == {"0", "1"}, out.stdout
    assert len(set(hashes.values())) == 1, hashes


@pytest.mark.skipif(MPIEXEC is None, reason="no mpiexec on PATH")
def test_a_fit_reads_the_same_nll_as_a_serial_fitter(tmp_path, sample):
    """The whole Fitter path: norm + data, `grads["norm"]` included."""
    script = tmp_path / "run_fit.py"
    script.write_text(textwrap.dedent(f'''
        import json
        import numpy as np
        from tabpwa import Fitter

        cfg = {CFG!r}
        # sample generation is a SINGLE-process operation: build it with the
        # reference backend, then hand the arrays to the MPI fitter
        gen = Fitter(cfg, backend="numpy_pwa")
        gen.apply_constrains()
        x0 = gen.initial_values(seed=1)
        phsp = gen.generate_phsp(240, seed=11)
        data = gen.generate_toy(x0, 160, seed=5)

        f = Fitter(cfg, backend={{"name": "mpi", "local": "numpy_pwa",
                                  "device": None}})
        f.apply_constrains()
        f.set_phsp(phsp)
        f.set_data(data)
        q, g = f.get_nll(x0)
        if f.backend.rank == 0:
            ref = Fitter(cfg, backend="numpy_pwa")
            ref.apply_constrains()
            ref.set_phsp(phsp)
            ref.set_data(data)
            qr, gr = ref.get_nll(x0)
            print("JSON" + json.dumps({{
                "size": f.backend.size,
                "dq": abs(float(q) - float(qr)),
                "dg": float(np.max(np.abs(np.asarray(g) - np.asarray(gr)))),
            }}))
    '''))
    env = dict(os.environ, OMPI_MCA_rmaps_base_oversubscribe="1")
    out = subprocess.run([MPIEXEC, "-n", "2", sys.executable, str(script)],
                         cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    res = json.loads([ln for ln in out.stdout.splitlines()
                      if ln.startswith("JSON")][-1][4:])
    assert res["size"] == 2
    assert res["dq"] < 1e-9           # NLL ~ -160, so this is ~1e-11 relative
    assert res["dg"] < 1e-9
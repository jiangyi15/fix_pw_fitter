"""ShardBackend end-to-end worker compute (numpy_pwa workers, no GPU)."""

import os
import signal
import subprocess
import sys
import textwrap
import time

import pathlib

import numpy as np
import pytest

from tabpwa.config_loader import Config
from tabpwa.backends import create_backend


def _arrays(ne, kc, seed):
    rs = np.random.RandomState(seed)
    nang = int(kc["angle_k"].shape[-1])
    return {"mass": 2.2 + 0.5 * rs.randn(ne, 1),
            "q": np.abs(rs.randn(ne, 2)) + 0.3,
            "angle": rs.uniform(-np.pi, np.pi, size=(ne, 1, nang)),
            "weight": np.ones(ne), "bkg": np.ones(ne)}


def _params(kc, seed=1):
    rs = np.random.RandomState(seed)
    n_proj = int(kc.get("n_proj", 1))
    N = kc["matrix_angle"].shape[1] // n_proj
    n_m0 = int(np.max(kc["m0_index"])) + 1
    n_g0 = int(np.max(kc["g0_index"])) + 1
    return {"ck": rs.randn(N) + 1j * rs.randn(N),
            "m0": 0.7 + 0.4 * np.abs(rs.randn(n_m0)),
            "g0": 0.1 + 0.3 * np.abs(rs.randn(n_g0))}


def test_shard_numpy_matches_single():
    cfg = Config("tests/config_pwa.yml")
    kc = cfg.build_all_index()
    params = _params(kc)
    data = _arrays(300, kc, 0)
    phsp = _arrays(600, kc, 2)

    single = create_backend("numpy_pwa", kc)
    # spawn/forkserver are safe in multi-threaded test runners; fork is
    # deprecated there (Python >= 3.14)
    shard = create_backend(
        {"name": "shard", "backends": ["numpy_pwa", "numpy_pwa"],
         "start_method": "spawn"}, kc)
    try:
        hs = single.load_data(data)
        hps = single.load_data(phsp)
        hd = shard.load_data(data)
        hpd = shard.load_data(phsp)

        norm_s = float(single.compute(params, hps, norm=None)[0])
        norm_d = float(shard.compute(params, hpd, norm=None)[0])
        assert abs(norm_s - norm_d) < 1e-6   # sums split across workers
        Q_s, g_s, P_s = single.compute(params, hs, norm=norm_s)
        Q_d, g_d, P_d = shard.compute(params, hd, norm=norm_s)

        assert abs(Q_s - Q_d) < 1e-9
        for key in ("ck", "m0", "g0", "norm"):
            assert np.allclose(np.asarray(g_d[key]), np.asarray(g_s[key]),
                               atol=1e-8), key
        # P rows are concatenated in worker (contiguous chunk) order
        assert np.allclose(P_d, P_s, atol=1e-8)

        # handle.free() drops the dataset but keeps the pool usable
        hd.free()
        hd2 = shard.load_data(data)
        Q2, _g2, _P2 = shard.compute(params, hd2, norm=norm_s)
        assert abs(Q2 - Q_s) < 1e-9
    finally:
        shard.free()
        single.free()


def test_worker_error_raises_instead_of_hanging():
    """A worker exception must surface as RuntimeError, not a blocked get()."""
    import pytest

    cfg = Config("tests/config_pwa.yml")
    kc = cfg.build_all_index()
    data = _arrays(50, kc, 0)
    shard = create_backend(
        {"name": "shard", "backends": "numpy_pwa", "n_workers": 1,
         "start_method": "spawn"}, kc)
    try:
        h = shard.load_data(data)
        with pytest.raises(RuntimeError, match="shard worker 0 failed"):
            shard.compute({}, h)          # missing params -> worker KeyError
    finally:
        shard.free()


def test_a_worker_that_dies_is_reported_instead_of_hanging():
    """A worker that never replies must raise, not block forever.

    ``worker_timeout=None`` (the default) means "wait as long as the worker
    needs" and cannot cover a worker that is GONE: its queue stays empty, so a
    blocking ``get()`` hangs — on a batch system, a burnt allocation.  The
    death is detected by polling the process, and the message names the common
    cause (a ``spawn`` worker re-imports the calling program).
    """
    import pytest

    cfg = Config("tests/config_pwa.yml")
    kc = cfg.build_all_index()
    data = _arrays(50, kc, 0)
    # a spec that cannot build: the child raises before its task loop starts
    shard = create_backend(
        {"name": "shard", "backends": "no_such_backend", "n_workers": 1,
         "start_method": "spawn"}, kc)
    import faulthandler
    try:
        # a wall-clock bound: if the death detection were removed this gate
        # would HANG instead of failing, which is the failure mode it exists to
        # prevent — so the hang itself must be a failure
        faulthandler.dump_traceback_later(60, exit=True)
        # `load_data` queues and returns (only a COMPUTE replies), so the dead
        # worker is detected by the first call that waits for a result
        h = shard.load_data(data)
        with pytest.raises(RuntimeError) as ei:
            shard.compute(_params(kc), h)
        msg = str(ei.value)
        assert "died (exitcode=" in msg and "without returning a result" in msg
        assert "if __name__ == '__main__':" in msg      # the spawn hint
    finally:
        faulthandler.cancel_dump_traceback_later()
        shard.free()


def test_spawn_re_executes_the_calling_program_so_it_must_be_guarded(tmp_path):
    """Without ``if __name__ == '__main__':`` a spawn worker cannot start.

    ``spawn`` has no memory to inherit, so it reconstructs ``__main__`` by
    RE-EXECUTING the calling script (``runpy.run_path(…, "__mp_main__")``) — the
    ``Process`` object is pickled and its target may live in ``__main__``, so
    the child needs that module.  An unguarded program re-runs its own work
    there, starts another pool and aborts the bootstrap; a bare
    ``multiprocessing`` program then exits **0** with a dead worker, while the
    shard backend raises and names the guard.
    """
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    cfg = str(Path("tests/config_pwa.yml").resolve())
    script = tmp_path / "unguarded.py"
    script.write_text(textwrap.dedent(f"""
        import numpy as np
        from tabpwa.config_loader import Config
        from tabpwa.backends import create_backend

        kc = Config({cfg!r}).build_all_index()
        be = create_backend({{"name": "shard", "backends": "numpy_pwa",
                             "n_workers": 1, "start_method": "spawn"}}, kc)
        be.load_data({{"mass": np.array([[2.2]])}})
        """))
    r = subprocess.run([sys.executable, str(script)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0, r.stdout + r.stderr
    # CPython's own bootstrap text also names the idiom, so THIS line passes on
    # the unfixed code too; the discriminating assertion is the next one
    assert "if __name__ == '__main__':" in r.stderr
    # OUR translation, not only Python's bootstrap text; which of the two
    # branches reports it depends on the parent/child race
    assert ("shard worker 0 could not be started" in r.stderr
            or "shard worker 0 died" in r.stderr), r.stderr


@pytest.mark.skipif(os.name != "posix", reason="os.kill(pid, 0) is POSIX-only")
def test_the_liveness_probe_answers_about_a_pid_and_kills_nothing():
    """`_pid_alive` asks, it does not signal.

    Off POSIX the helper answers "alive" WITHOUT probing, because there
    `os.kill(pid, 0)` is `TerminateProcess` (it would kill the driver) — and
    the parent watchdog is not implemented there at all, since Windows never
    reparents and `getppid()` is frozen.  This gate pins the POSIX semantics
    only: a live pid reads alive, a REAPED child reads gone, and the caller
    still exists afterwards.  (That signal 0 never signals cannot be observed;
    the non-POSIX refusal is a branch this box cannot execute.)
    """
    from tabpwa.backends.shard_backend import _pid_alive

    assert _pid_alive(os.getpid()) is True
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    pid = child.pid
    child.wait()
    assert _pid_alive(os.getpid()) is True      # still here: nothing signalled
    assert _pid_alive(pid) is False             # reaped: ESRCH


def test_two_datasets_in_one_pool_do_not_cross():
    """Two datasets, two handles: each compute gets its OWN dataset back.

    Two datasets are loaded before either is computed (the harder order — the
    second load sits between the first load and the first compute), so this
    pins that a handle carries its dataset's identity, not just a reply slot:
    doubling the weights doubles `Q` exactly, and freeing one handle leaves the
    other's numbers untouched.
    """
    kc = Config("tests/config_pwa.yml").build_all_index()
    params = _params(kc)
    d1 = _arrays(60, kc, 0)
    d2 = {**d1, "weight": np.full(60, 2.0)}       # a different sample
    shard = create_backend({"name": "shard", "backends": "numpy_pwa",
                            "n_workers": 2, "start_method": "spawn"}, kc)
    try:
        h1 = shard.load_data(d1)
        h2 = shard.load_data(d2)
        Q1, _g1, _ = shard.compute(params, h1, norm=None)
        Q2, _g2, _ = shard.compute(params, h2, norm=None)
        # the weights differ by 2 → a weight-sum norm differs, P does not
        assert Q2 == pytest.approx(2.0 * Q1, rel=1e-12), (Q1, Q2)
        h1.free()
        Q3, _g3, _ = shard.compute(params, h2, norm=None)   # h2 unaffected
        assert Q3 == pytest.approx(Q2, rel=1e-12)
    finally:
        shard.free()


def test_free_stops_every_worker():
    """`free()` must leave NO worker alive — the pool is the job's lifetime.

    They are daemonic, so a parent that exits takes them with it; a parent that
    KEEPS RUNNING (a library user building one pool per fit) would keep a GPU
    worker forever if `free()` returned early.
    """
    kc = Config("tests/config_pwa.yml").build_all_index()
    data = _arrays(30, kc, 1)
    shard = create_backend({"name": "shard", "backends": "numpy_pwa",
                            "n_workers": 2, "start_method": "spawn"}, kc)
    try:
        shard.load_data(data)
        procs = list(shard._procs)
        assert len(procs) == 2 and all(p.is_alive() for p in procs)
    finally:
        shard.free()
    assert not any(p.is_alive() for p in procs), "a worker outlived free()"


def test_free_kills_the_workers_when_the_exit_command_cannot_be_delivered():
    """A failed `put(("exit",))` (a closed queue) must still take the workers down.

    MEASURED before the fix: with the task queues closed, `free()` raised on the
    first delivery attempt and BOTH workers were still alive — a daemonic leak
    that only ends when the parent does.
    """
    kc = Config("tests/config_pwa.yml").build_all_index()
    data = _arrays(30, kc, 1)
    shard = create_backend({"name": "shard", "backends": "numpy_pwa",
                            "n_workers": 2, "start_method": "spawn"}, kc)
    try:
        shard.load_data(data)
        procs = list(shard._procs)
        for q in shard._task_queues:
            q.close()
            q.join_thread()
    finally:
        shard.free()                      # must not raise
    assert not any(p.is_alive() for p in procs), "a worker outlived free()"


@pytest.mark.skipif(os.name != "posix", reason="needs SIGKILL and kill(2)")
@pytest.mark.parametrize("method", ["spawn", "fork", "forkserver"])
def test_a_worker_does_not_outlive_a_SIGKILLed_driver(tmp_path, method):
    """A driver killed with SIGKILL runs no `atexit`: the worker must notice.

    Nothing terminates a daemonic child on `SIGKILL` and no `("exit",)` command
    can arrive, so a blocking `get()` would hold a GPU context forever.  The
    worker re-checks `os.getppid()` each idle slice — reparenting happens when
    the parent dies, and unlike the multiprocessing sentinel it cannot be
    inherited from a SIBLING worker (`fork` copies the earlier workers' write
    ends, which made a sentinel-based pool stop one worker per slice) — and the
    DRIVER's pid is checked too, because under `forkserver` the worker's direct
    parent is the forkserver, which outlives the driver.  All three start
    methods run here; `spawn` alone never sees either problem.

    The wait below is `n_workers` slices, derived from the backend's own
    constant: a bound on how long the TEST waits, never a claim about how long
    any work takes.
    """
    from tabpwa.backends.shard_backend import _WORKER_POLL

    n = 3
    script = tmp_path / "killed.py"
    pids_file = tmp_path / "pids.txt"
    fs_file = tmp_path / "fs.txt"
    script.write_text(textwrap.dedent(f"""
        import os, signal, sys
        sys.path.insert(0, {str(pathlib.Path("tests").resolve())!r})
        from tabpwa import Fitter
        from tabpwa.backends import create_backend
        from _flat_grid import _in_grid

        def main():
            f = Fitter({str(pathlib.Path("tests/config_pwa_gen_km.yml").resolve())!r})
            f.apply_constrains()
            kc = f.kernel_config
            params, _ = f.build_params(f.initial_values(seed=1))
            data = _in_grid(kc, 40, seed=2)
            be = create_backend({{"name": "shard", "backends": "numpy_flat",
                                  "n_workers": {n},
                                  "start_method": {method!r}}}, kc, model="pwa_gen")
            h = be.load_data(data)
            be.compute(params, h, norm=None)
            with open({str(pids_file)!r}, "w") as fh:
                fh.write(" ".join(str(p.pid) for p in be._procs))
            try:                      # a private CPython handle; cleanup only
                from multiprocessing import forkserver as _fs
                fs_pid = _fs._forkserver._forkserver_pid
            except Exception:
                fs_pid = None
            with open({str(fs_file)!r}, "w") as fh:
                fh.write(str(fs_pid) if fs_pid else "")
            sys.stdout.flush()
            os.kill(os.getpid(), signal.SIGKILL)

        if __name__ == "__main__":
            main()
        """))
    # the driver's OUTPUT goes to a FILE, not a pipe: a `forkserver` child
    # inherits the driver's stdio, and `subprocess.run(capture_output=True)`
    # would block on the pipe's EOF long after the workers are gone
    log = tmp_path / "driver.log"
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, str(script)], stdout=fh, stderr=fh,
                           timeout=4 * (n + 1) * _WORKER_POLL)
    detail = log.read_text()[-800:]
    assert r.returncode == -signal.SIGKILL, (r.returncode, detail)
    pids = [int(x) for x in pids_file.read_text().split()]
    assert len(pids) == n

    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False

    deadline = time.time() + (n + 1) * _WORKER_POLL
    while time.time() < deadline and any(alive(p) for p in pids):
        time.sleep(0.2)
    try:
        assert not any(alive(p) for p in pids), \
            f"[{method}] worker(s) outlived the SIGKILLed driver: " \
            f"{[p for p in pids if alive(p)]}"
    finally:
        # a forkserver is CPython's own helper process: OUR contract is about
        # the workers, so it is cleaned up here rather than asserted on
        fs = fs_file.read_text().strip() if fs_file.exists() else ""
        if fs:
            try:
                os.kill(int(fs), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_a_worker_failure_names_its_phase_and_stops_reuse():
    """A failed worker leaves the reply stream shifted: refuse, don't reuse.

    `compute` queues one task per worker and reads one reply per worker.  When
    a worker fails it replies with `(_ERROR, phase, traceback)`, and with the
    failure consumed the NEXT reply in its queue belongs to the attempt that
    just failed — so a retry would return the previous call's numbers.  The
    pool therefore reports the phase and refuses further work.

    The failure is triggered in the compute (a coefficient vector of the wrong
    length), which is the phase every kernel can be made to fail in.
    """
    kc = Config("tests/config_pwa.yml").build_all_index()
    bad = _params(kc)
    bad["ck"] = bad["ck"][:-3]              # wrong length: the kernel raises
    shard = create_backend({"name": "shard", "backends": "numpy_pwa",
                            "n_workers": 2, "start_method": "spawn"}, kc)
    try:
        h = shard.load_data(_arrays(40, kc, 0))
        with pytest.raises(RuntimeError, match="failed during compute"):
            shard.compute(bad, h, norm=None)
        with pytest.raises(RuntimeError, match="desynchronised"):
            shard.compute(_params(kc), h, norm=None)   # not the old numbers
        with pytest.raises(RuntimeError, match="desynchronised"):
            shard.load_data(_arrays(40, kc, 1))
    finally:
        shard.free()


def test_compute_after_free_names_the_lifecycle_error():
    """A compute with no pool is a stated error, never an `IndexError`.

    `free()`/`kill()` detach the process list and the queues, so a later
    `compute` has nothing to send to and nothing to read from.  Say that;
    an index error out of `_result_queues[i]` would look like a kernel bug.
    A stale handle is a no-op (its dataset is already gone with the pool).
    """
    kc = Config("tests/config_pwa.yml").build_all_index()
    shard = create_backend({"name": "shard", "backends": "numpy_pwa",
                            "n_workers": 2, "start_method": "spawn"}, kc)
    h = shard.load_data(_arrays(40, kc, 0))
    shard.free()
    with pytest.raises(RuntimeError, match="pool is not running"):
        shard.compute(_params(kc), h, norm=None)
    h.free()                      # the dataset went with the pool: no-op
    assert shard._procs == [] and shard._task_queues == []


def test_worker_dict_spec_keeps_kwargs():
    """A dict worker spec must keep its kwargs (only device is stripped)."""
    from tabpwa.backends.shard_backend import ShardBackend

    cfg = Config("tests/config_pwa.yml")
    kc = cfg.build_all_index()
    be = ShardBackend(kc, backends=[
        {"name": "cuda_v5_pwa", "resolution_size": 20, "batch_size": 50000}])
    assert be._specs[0][0] == {"name": "cuda_v5_pwa",
                               "resolution_size": 20, "batch_size": 50000}

    be = ShardBackend(kc, backends=[
        {"name": "cuda_v5_pwa", "resolution_size": 20, "device": 1}])
    assert be._specs[0] == ({"name": "cuda_v5_pwa", "resolution_size": 20}, 1)

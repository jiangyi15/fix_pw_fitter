"""ShardBackend end-to-end worker compute (numpy_pwa workers, no GPU)."""

import numpy as np

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
        h = shard.load_data(data)
        # a wall-clock bound: if the death detection were removed this gate
        # would HANG instead of failing, which is the failure mode it exists to
        # prevent — so the hang itself must be a failure
        faulthandler.dump_traceback_later(60, exit=True)
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

"""Tests for :mod:`tabpwa.monitor`.

The GPU sampler is one ``nvidia-smi`` child; the host fields come from
``/proc/self/status``.  Nothing here may import ``torch``:
``torch.cuda.init()`` slows every later raw-CUDA call 5.1x, permanently.  The
suite-wide gate lives in ``_flat_compute.py``; this module keeps its own local
before/after form (see ``test_a_monitor_cycle_does_not_import_torch``).
"""
import io
import sys
import time

import pytest

from tabpwa import monitor as monitor_mod
from tabpwa.monitor import (FIELDS, GPU_FIELDS, Results, create_monitor,
                            free_bytes, has_device, sync)


def _started(what=None, interval=0.01, timeout=5.0):
    mon = create_monitor(what=what, interval=interval)
    mon.wait_ready(timeout)
    return mon


def _reaped(proc):
    return proc is None or proc.poll() is not None


# ── lifecycle / orphans ────────────────────────────────────────────────

def test_start_stop_leaves_no_child():
    mon = _started()
    proc = mon._proc
    res = mon.close()
    assert isinstance(res, Results)
    assert _reaped(proc)
    assert mon._proc is None
    assert mon._thread is None or not mon._thread.is_alive()


def test_close_twice_is_safe_and_returns_the_same_snapshot():
    mon = _started()
    a = mon.close()
    b = mon.close()
    assert a is b
    assert mon.get_results() is b


def test_close_after_exception_in_with_body():
    held = {}
    with pytest.raises(RuntimeError, match="boom"):
        with create_monitor(interval=0.01) as mon:
            held["mon"] = mon
            held["proc"] = mon._proc
            mon.wait_ready(5.0)
            raise RuntimeError("boom")
    mon = held["mon"]
    assert mon._closed
    assert _reaped(held["proc"])
    assert isinstance(mon.get_results(), Results)


def test_mark_after_close_raises():
    mon = _started()
    mon.close()
    with pytest.raises(RuntimeError):
        mon.mark("late")


# ── phases / slicing ───────────────────────────────────────────────────

def test_mark_and_phase_slicing():
    mon = _started()
    try:
        mon.mark("alpha")
        time.sleep(0.08)
        mon.mark("beta")
        time.sleep(0.08)
        mon.mark("gamma")
        time.sleep(0.08)
        res = mon.close()
    finally:
        mon.close()
    alpha = res.series("rss", phase="alpha")
    beta = res.series("rss", phase="beta")
    gamma = res.series("rss", phase="gamma")
    assert alpha and beta and gamma
    # windows are disjoint and ordered
    assert max(t for t, _ in alpha) <= min(t for t, _ in beta)
    assert max(t for t, _ in beta) <= min(t for t, _ in gamma)
    # an UNKNOWN phase is an error, not an empty window: a typo must not pass
    with pytest.raises(ValueError, match="unknown phase 'does-not-exist'"):
        res.series("rss", phase="does-not-exist")
    with pytest.raises(ValueError, match="marks are"):
        res.peak("rss", phase="does-not-exist")
    # the phase-less call covers everything
    assert res.peak("rss") == max(v for _, v in res.series("rss"))


def test_peak_ignores_none():
    res = Results(
        [(0.0, {"x": None}), (1.0, {"x": 3.0}),
         (2.0, {"x": None}), (3.0, {"x": 1.0})],
        [], 3.0, requested=("x",))
    assert res.peak("x") == 3.0
    empty = Results([(0.0, {"x": None}), (1.0, {"x": None})], [], 1.0,
                    requested=("x",))
    assert empty.peak("x") is None
    assert empty.series("x") == []
    with pytest.raises(ValueError, match="was not requested"):
        res.peak("missing")
    # a monitor that requested nothing: every peak is trivially None
    nothing = Results([], [], 0.0, requested=())
    assert nothing.peak("missing") is None
    assert nothing.series("missing") == []


def test_phase_bounds_are_half_open_and_an_existing_phase_can_be_empty():
    """The boundary sample belongs to the NEXT phase; an empty phase is an
    empty window, an unknown one is an error."""
    res = Results(
        [(1.0, {"x": 1.0}), (2.0, {"x": 2.0}), (3.0, {"x": 3.0})],
        [(0.0, "a"), (2.0, "b"), (4.0, "empty")], 4.0, requested=("x",))
    assert res.series("x", phase="a") == [(1.0, 1.0)]
    assert res.series("x", phase="b") == [(2.0, 2.0), (3.0, 3.0)]
    assert res.series("x", phase="empty") == []      # exists, caught nothing
    assert res.peak("x", phase="empty") is None
    with pytest.raises(ValueError, match="unknown phase 'nope'; marks are"):
        res.peak("x", phase="nope")
    # a REPEATED label is reachable only at its FIRST occurrence
    rep = Results([(1.0, {"x": 1.0}), (3.0, {"x": 3.0})],
                  [(0.0, "p"), (2.5, "p")], 3.0, requested=("x",))
    assert rep.series("x", phase="p") == [(1.0, 1.0)]


# ── snapshots ──────────────────────────────────────────────────────────

def test_get_results_mid_run_then_after_close():
    mon = _started()
    time.sleep(0.05)
    mid = mon.get_results()
    assert isinstance(mid, Results)
    time.sleep(0.08)
    res = mon.close()
    assert res.n_samples >= mid.n_samples
    assert mon.get_results() is res


def test_snapshot_is_not_mutated_by_later_samples():
    mon = _started()
    time.sleep(0.04)
    snap = mon.get_results()
    n = snap.n_samples
    n_series = len(snap.series("rss"))
    time.sleep(0.10)
    mon.close()
    assert snap.n_samples == n
    assert len(snap.series("rss")) == n_series


# ── host / GPU field behaviour ─────────────────────────────────────────

def test_host_fields_present_on_this_box():
    mon = _started()
    time.sleep(0.06)
    res = mon.close()
    rss = res.peak("rss")
    hwm = res.peak("hwm")
    assert rss is not None and rss > 0
    assert hwm is not None and hwm >= rss


def test_no_nvidia_smi_degrades_to_none(monkeypatch):
    monkeypatch.setenv("PATH", "")
    mon = create_monitor(interval=0.01)
    try:
        mon.wait_ready(5.0)
        time.sleep(0.06)
        res = mon.close()
    finally:
        mon.close()
    for field in GPU_FIELDS:
        assert res.peak(field) is None, field
    assert res.peak("gpu_used") is None
    assert res.peak("rss") is not None


def test_summary_is_printable():
    mon = _started()
    mon.mark("a")
    time.sleep(0.03)
    mon.mark("b")
    time.sleep(0.03)
    res = mon.close()
    text = res.summary()
    assert "gpu_used" in text and "rss" in text
    assert "\n" in text


# ── the field selection (`what=`) ──────────────────────────────────────

def test_unknown_field_raises_and_lists_valid_names():
    with pytest.raises(ValueError) as ei:
        create_monitor(what=("gpu_usd",))
    msg = str(ei.value)
    assert "gpu_usd" in msg
    for name in FIELDS:
        assert name in msg, name


def test_host_only_request_starts_no_child():
    mon = create_monitor(what=("rss", "hwm"), interval=0.01)
    try:
        assert mon._proc is None
        assert mon._thread is None
        assert mon._want_gpu is False
        assert mon.wait_ready(2.0)
        res = mon.get_results()
        assert res.peak("rss") is not None
        with pytest.raises(ValueError, match="was not requested"):
            res.peak("gpu_used")
    finally:
        mon.close()


def test_unrequested_raises_while_unavailable_is_none(monkeypatch):
    mon = create_monitor(what=("rss",), interval=0.01)
    try:
        res = mon.get_results()
        with pytest.raises(ValueError, match="was not requested"):
            res.peak("gpu_used")
        with pytest.raises(ValueError, match="was not requested"):
            res.series("gpu_total")
        assert res.peak("rss") is not None
    finally:
        mon.close()

    monkeypatch.setenv("PATH", "")        # requested gpu field, box cannot give it
    mon = create_monitor(what=("gpu_used",), interval=0.01)
    try:
        mon.wait_ready(2.0)
        res = mon.get_results()
        assert res.peak("gpu_used") is None
        assert res.series("gpu_used") == []
    finally:
        mon.close()


class _FakeProc:
    """A Popen stand-in: records argv, emits no lines, reaps cleanly."""

    def __init__(self, argv):
        self.argv = list(argv)
        self.stdout = io.StringIO("")
        self.returncode = 0

    def poll(self):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


def test_gpu_query_requests_exactly_the_selected_columns(monkeypatch):
    seen = {}

    def fake_popen(argv, **kw):
        seen["argv"] = list(argv)
        return _FakeProc(argv)

    monkeypatch.setattr(monitor_mod.shutil, "which",
                        lambda name: "/fake/nvidia-smi")
    monkeypatch.setattr(monitor_mod.subprocess, "Popen", fake_popen)
    mon = create_monitor(what=("gpu_used", "gpu_total"), interval=0.01)
    mon.close()
    argv = seen["argv"]
    queries = [a for a in argv if a.startswith("--query-gpu=")]
    assert queries == ["--query-gpu=memory.used,memory.total"], argv
    assert not any("memory.free" in a for a in argv)
    # the device is PINNED: without -i, --query-gpu emits one line per GPU
    assert argv[argv.index("-i") + 1] == "0", argv


def test_device_is_a_recorded_keyword_and_lands_in_the_command(monkeypatch):
    seen = {}

    def fake_popen(argv, **kw):
        seen["argv"] = list(argv)
        return _FakeProc(argv)

    monkeypatch.setattr(monitor_mod.shutil, "which",
                        lambda name: "/fake/nvidia-smi")
    monkeypatch.setattr(monitor_mod.subprocess, "Popen", fake_popen)
    for dev in (0, 2, "3"):
        mon = create_monitor(what=("gpu_used",), interval=0.01, device=dev)
        try:
            assert mon.device == int(dev)
            argv = seen["argv"]
            assert argv[argv.index("-i") + 1] == str(int(dev)), argv
        finally:
            mon.close()
    for bad in (-1, "gpu", None):
        with pytest.raises(ValueError, match="device"):
            create_monitor(what=("gpu_used",), interval=0.01, device=bad)


def test_a_child_that_dies_mid_run_falls_back_to_host_samples():
    """The reader's fallback: the child is killed, host samples keep coming.

    Exercises the ``for line in proc.stdout`` exit path (a dead child ends the
    iteration) followed by the host-only loop — and ``close()`` still returns
    even though the child was killed by someone else.
    """
    mon = create_monitor(interval=0.02)
    try:
        proc = mon._proc
        if proc is None:
            pytest.skip("no nvidia-smi child to kill on this box")
        proc.kill()
        before = mon.get_results().n_samples
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if mon.get_results().n_samples > before:
                break
            time.sleep(0.02)
        assert mon.get_results().n_samples > before, (
            "host samples stopped when the child died")
        res = mon.close()
    finally:
        mon.close()
    assert isinstance(res, Results)
    assert res.peak("rss") is not None
    assert _reaped(proc)


def test_empty_selection_is_legal():
    mon = create_monitor(what=(), interval=0.01)
    try:
        assert mon._proc is None and mon._thread is None
        assert mon.wait_ready(0.2)
        res = mon.close()
    finally:
        mon.close()
    assert res.n_samples == 0
    assert "nothing was sampled" in res.summary()
    assert res.peak("gpu_used") is None       # nothing requested at all


# ── the ctypes probe (the ONE copy: tabpwa.monitor) ────────────────────

def test_cuda_probe_helpers():
    if not has_device():
        pytest.skip("no CUDA device")
    sync()
    value = free_bytes()
    assert isinstance(value, int) and value > 0


# ── no torch ───────────────────────────────────────────────────────────

def test_a_monitor_cycle_does_not_import_torch():
    """The HONEST local form: the monitor must not PULL torch in.

    A bare ``assert "torch" not in sys.modules`` would duplicate
    ``_flat_compute.py``'s suite-wide gate and could only fail for a reason
    unrelated to this module (some earlier test importing torch).  So this
    records the state before and after a full cycle on THIS process and only
    fails if the cycle is what made it true.
    """
    was_loaded = "torch" in sys.modules
    mon = create_monitor(interval=0.01)
    mon.mark("cycle")
    time.sleep(0.05)
    mon.get_results()
    res = mon.close()
    assert isinstance(res, Results)
    assert ("torch" in sys.modules) == was_loaded, (
        "the monitor imported torch (it would slow every later raw-CUDA call)")

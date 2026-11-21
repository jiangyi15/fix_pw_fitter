"""System monitor: GPU memory/utilisation + host RSS, sampled out of process.

The GPU numbers come from ONE long-lived ``nvidia-smi`` child::

    nvidia-smi --query-gpu=<requested gpu columns> \\
        --format=csv,noheader,nounits -lms <interval_ms> -i <device>

``-lms`` emits one CSV line per interval, so the measured process never forks
per sample and never asks the CUDA runtime anything.  In particular this
module **never imports ``torch``** (``torch.cuda.init()`` makes every later
raw-CUDA call in the process 5.1x slower, permanently) and the sampler **never
calls ``cudaMemGetInfo``** (an in-process probe creates/initialises a context
and would inflate the very memory it reports).  ``nvidia-smi`` reads the
driver, so it cannot perturb the number.

**The sampled device is pinned with ``-i <device>``** (``create_monitor``'s
``device=`` keyword, default ``0``).  Without ``-i``, ``--query-gpu`` emits one
line per GPU every interval, and a reader that recorded every line would
alternate devices — ``series()`` interleaved and ``peak()`` a cross-device
maximum.  Note the two device selectors do NOT have to agree: the sampler's
``device=`` is an ``nvidia-smi`` ordinal, while :func:`free_bytes` reads
CUDA's CURRENT device (whatever ``CUDA_VISIBLE_DEVICES`` / ``cudaSetDevice``
left in place), so under ``CUDA_VISIBLE_DEVICES`` the two can point at
different physical GPUs.

The parent keeps exactly ONE reader thread, and only when a ``gpu_*`` field
was requested.  It *only drains the child's stdout* (and reads this process's
``/proc/self/status`` at the same cadence, because the host high-water mark
lives here, not in the driver).  The GPU sampling itself is the child; the
thread is not a sampler.

**The selection decides the mechanism, not just the filtering.**  A request
with any ``gpu_*`` field starts the child; a request with no ``gpu_*`` field
(``what=("rss", "hwm")``) starts NO child, NO pipe and NO drain thread — host
status is read in-process (at construction, at each ``mark``, at
``get_results`` and at ``close``).

Public API::

    from tabpwa.monitor import create_monitor

    with create_monitor(what=("gpu_used", "gpu_total", "rss", "hwm"),
                        interval=0.05, device=0) as mon:
        mon.mark("load")
        ...
        snap = mon.get_results()        # legal mid-run (immutable snapshot)
    res = mon.close()                   # stop + reap; idempotent
    res.peak("gpu_used")                # peak(field, phase=...)
    res.series("gpu_used")              # [(t, value), ...]
    print(res.summary())

``what=None`` (the default) requests every field.  Unknown names raise
``ValueError`` listing the valid ones.  Degrade, never raise: with no
``nvidia-smi`` (or a non-Linux box) the requested GPU fields are ``None`` and
the host fields still work; a monitor that cannot even start still returns a
usable (empty) :class:`Results`.

The module also owns the ONE ``libcudart``-via-``ctypes`` point probe
(:func:`free_bytes` / :func:`sync` / :func:`has_device`) used by the leak
tests.  That probe is a separate, explicit call — it is NOT part of the
sampler (its FIRST call initialises the primary CUDA context, which allocates
device memory) — and no test skip logic lives here (a missing runtime returns
``None``/``False`` so callers can skip themselves).
"""

from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import shutil
import subprocess
import threading
import time
import weakref

__all__ = [
    "create_monitor", "Monitor", "Results",
    "GPU_FIELDS", "HOST_FIELDS", "FIELDS", "UNITS",
    "cuda_runtime", "has_device", "sync", "free_bytes",
]

#: GPU fields, in ``nvidia-smi --query-gpu`` order (all MiB / % / degC / W)
_GPU_QUERY = (
    "memory.used", "memory.free", "memory.total",
    "utilization.gpu", "temperature.gpu", "power.draw",
)
GPU_FIELDS = ("gpu_used", "gpu_free", "gpu_total",
              "gpu_util", "gpu_temp", "gpu_power")
HOST_FIELDS = ("rss", "hwm")
FIELDS = GPU_FIELDS + HOST_FIELDS

#: field -> the ``nvidia-smi --query-gpu`` column it reads
_FIELD_QUERY = dict(zip(GPU_FIELDS, _GPU_QUERY))

#: printable unit per field (the GPU memory numbers are MiB, host MiB)
UNITS = {
    "gpu_used": "MiB", "gpu_free": "MiB", "gpu_total": "MiB",
    "gpu_util": "%", "gpu_temp": "C", "gpu_power": "W",
    "rss": "MiB", "hwm": "MiB",
}

_INF = float("inf")


def _resolve_fields(what):
    """Validate a ``what=`` argument into an ordered tuple of field names."""
    if what is None:
        return FIELDS
    if isinstance(what, str):
        what = (what,)
    fields = tuple(dict.fromkeys(what))
    bad = [f for f in fields if f not in FIELDS]
    if bad:
        raise ValueError(
            f"unknown monitor field(s) {bad}; valid names: {list(FIELDS)}")
    return fields


def _device_index(device):
    """Validate ``device=`` into a non-negative ``nvidia-smi`` GPU index."""
    try:
        dev = int(device)
    except (TypeError, ValueError):
        raise ValueError(
            f"device must be a GPU index (int), got {device!r}") from None
    if dev < 0:
        raise ValueError(f"device must be a non-negative GPU index, got {dev}")
    return dev


def _num(text):
    """Parse one ``nvidia-smi`` CSV cell, or ``None`` (``[N/A]`` etc.)."""
    text = text.strip()
    if not text or text.startswith("["):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _host_mib():
    """``(VmRSS, VmHWM)`` in MiB from ``/proc/self/status`` (``None`` off Linux)."""
    rss = hwm = None
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = float(line.split()[1]) / 1024.0
                elif line.startswith("VmHWM:"):
                    hwm = float(line.split()[1]) / 1024.0
                if rss is not None and hwm is not None:
                    break
    except OSError:
        pass
    return rss, hwm


# ── libcudart point probe (the ONE ctypes implementation) ───────────────
_LIB = None
_TRIED = False


def _load_runtime():
    for cand in ("libcudart.so", "libcudart.so.13", "libcudart.so.12",
                 ctypes.util.find_library("cudart")):
        if not cand:
            continue
        try:
            lib = ctypes.CDLL(cand)
        except OSError:
            continue
        lib.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t),
                                       ctypes.POINTER(ctypes.c_size_t)]
        lib.cudaMemGetInfo.restype = ctypes.c_int
        lib.cudaDeviceSynchronize.restype = ctypes.c_int
        lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cudaGetDeviceCount.restype = ctypes.c_int
        return lib
    return None


def cuda_runtime():
    """The ``libcudart`` handle, or ``None``.  Loaded once, never raises."""
    global _LIB, _TRIED
    if not _TRIED:
        _TRIED = True
        _LIB = _load_runtime()
    return _LIB


def has_device():
    """True iff a CUDA runtime with at least one device is available."""
    lib = cuda_runtime()
    if lib is None:
        return False
    n = ctypes.c_int(0)
    if lib.cudaGetDeviceCount(ctypes.byref(n)) != 0:
        return False
    return n.value > 0


def sync():
    """Wait for every queued kernel.  ``None`` when no runtime is available.

    The FIRST call initialises this process's primary CUDA context (any
    ``libcudart`` call does), which allocates device memory — which is exactly
    why this probe is NOT part of the sampler: it would inflate the number it
    exists to measure.  It acts on CUDA's CURRENT device, not on the
    ``device=`` ordinal the sampler was given.
    """
    lib = cuda_runtime()
    if lib is None:
        return None
    rc = lib.cudaDeviceSynchronize()
    if rc != 0:
        raise RuntimeError(f"cudaDeviceSynchronize failed: rc={rc}")
    return rc


def free_bytes():
    """Free device memory in BYTES, or ``None`` when no runtime is available.

    The FIRST call initialises this process's primary CUDA context (any
    ``libcudart`` call does), which allocates device memory — which is exactly
    why this probe is NOT part of the sampler: it would inflate the number it
    exists to measure.  It reads CUDA's CURRENT device, not the ``device=``
    ordinal the sampler was given; under ``CUDA_VISIBLE_DEVICES`` the visible
    device 0 is a different physical GPU from the one ``nvidia-smi -i 0``
    reports.
    """
    lib = cuda_runtime()
    if lib is None:
        return None
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    rc = lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    if rc != 0:
        raise RuntimeError(f"cudaMemGetInfo failed: rc={rc}")
    return free.value


# ── results ─────────────────────────────────────────────────────────────

class Results:
    """An immutable snapshot of a :class:`Monitor` (samples + phase marks).

    ``requested`` is the field selection the monitor was built with; a field
    outside it is an error, which is distinct from a requested field the box
    cannot provide (that one is simply ``None``).

    The phase accessors (:meth:`series` / :meth:`peak`) take a ``mark`` label:
    an unknown label is a ``ValueError``, a known one that caught no sample is
    empty/``None``, and the window is ``start <= t < end``.
    """

    __slots__ = ("_samples", "_marks", "_duration", "_requested")

    def __init__(self, samples, marks, duration, requested=None):
        object.__setattr__(self, "_samples",
                           tuple((float(t), dict(d)) for t, d in samples))
        object.__setattr__(self, "_marks",
                           tuple((float(t), str(lab)) for t, lab in marks))
        object.__setattr__(self, "_duration", float(duration))
        object.__setattr__(self, "_requested",
                           FIELDS if requested is None else tuple(requested))

    def __setattr__(self, name, value):        # pragma: no cover - trivial
        raise AttributeError("Results is immutable")

    def __delattr__(self, name):               # pragma: no cover - trivial
        raise AttributeError("Results is immutable")

    # -- introspection ---------------------------------------------------
    @property
    def n_samples(self):
        return len(self._samples)

    @property
    def n_marks(self):
        return len(self._marks)

    @property
    def duration(self):
        """Seconds covered by the samples."""
        return self._duration

    @property
    def fields(self):
        """The fields this monitor was asked to sample, in request order."""
        return self._requested

    @property
    def marks(self):
        return tuple((t, lab) for t, lab in self._marks)

    def __repr__(self):
        return (f"Results(n_samples={self.n_samples}, "
                f"n_marks={self.n_marks}, duration={self._duration:.3f}s, "
                f"fields={self._requested})")

    # -- phase slicing ---------------------------------------------------
    def _bounds(self, phase):
        """Half-open ``[start, end)`` of a phase.  Unknown ``phase`` raises."""
        if phase is None:
            return -_INF, _INF
        labels = [lab for _, lab in self._marks]
        try:
            i = labels.index(phase)
        except ValueError:
            raise ValueError("unknown phase %r; marks are %s"
                             % (phase, labels)) from None
        start = self._marks[i][0]
        end = self._marks[i + 1][0] if i + 1 < len(self._marks) else _INF
        return start, end

    def _slice(self, field, phase):
        start, end = self._bounds(phase)
        return ((t, d.get(field)) for t, d in self._samples
                if start <= t < end)

    def _require(self, field):
        """True if sampled; False if nothing was requested at all; else raise."""
        if field in self._requested:
            return True
        if not self._requested:
            return False
        raise ValueError(f"`{field}` was not requested; this monitor "
                         f"samples {self._requested}")

    def series(self, field, phase=None):
        """``[(t, value), ...]`` for ``field``, skipping absent (``None``) points.

        ``phase=None`` (default) is the whole run.  A label that exists but
        caught no sample returns ``[]``; an UNKNOWN ``phase`` raises
        ``ValueError`` — a typo must not look like an empty phase.

        The window is ``start <= t < end``: the sample stamped exactly on a
        boundary belongs to the NEXT phase, and a ``mark()`` is not a barrier
        (it never waits for the child), so the phase can contain a sample
        whose *measurement* is up to one ``interval`` older than its stamp.
        """
        if not self._require(field):
            return []
        return [(t, v) for t, v in self._slice(field, phase) if v is not None]

    def peak(self, field, phase=None):
        """Largest value of ``field`` in the phase, ignoring ``None``s.

        ``phase=None`` (default) is the whole run.  ``None`` when the phase
        exists but caught no usable sample; an UNKNOWN ``phase`` raises
        ``ValueError`` (see :meth:`series`, which also documents the
        ``start <= t < end`` window and the drain-time stamp).
        """
        if not self._require(field):
            return None
        values = [v for _, v in self._slice(field, phase) if v is not None]
        return max(values) if values else None

    def summary(self):
        """One printable table: only the requested fields, plus phase columns.

        A label repeated by ``mark`` appears (and is peaked) ONCE, at its
        first occurrence; the phase-based accessors cannot reach the repeat.
        """
        if not self._requested:
            return (repr(self) + "\n"
                    "(no fields requested: nothing was sampled, every peak is None)")
        labels = []
        for _, lab in self._marks:
            if lab not in labels:
                labels.append(lab)
        name_w = max(len(f) for f in self._requested)
        header = f"{'field':<{name_w}} {'unit':<4} {'n':>5} {'peak':>12}"
        for lab in labels:
            header += f"  {lab[:12]:>12}"
        lines = [repr(self), header]
        for field in self._requested:
            vals = self.series(field)
            pk = self.peak(field)
            pk_s = "-" if pk is None else f"{pk:.2f}"
            row = f"{field:<{name_w}} {UNITS[field]:<4} {len(vals):>5} {pk_s:>12}"
            for lab in labels:
                p = self.peak(field, phase=lab)
                row += f"  {('-' if p is None else f'{p:.2f}'):>12}"
            lines.append(row)
        return "\n".join(lines)


# ── monitor ─────────────────────────────────────────────────────────────

#: every live monitor, so one atexit hook can reap them all
_LIVE = weakref.WeakSet()
_ATEXIT = False


def _close_all():
    for mon in list(_LIVE):
        try:
            mon.close()
        except BaseException:
            pass


def _ensure_atexit():
    global _ATEXIT
    if not _ATEXIT:
        _ATEXIT = True
        atexit.register(_close_all)


class Monitor:
    """Sample the requested GPU + host fields from a background child.

    Build it through :func:`create_monitor` (which validates ``what`` and
    ``device``); ``with`` and ``__del__`` both close it, and :func:`atexit`
    reaps any that a process forgets.

    ``device`` is the ``nvidia-smi`` ordinal passed as ``-i`` — ONE device is
    sampled, so a ``series`` is a single card's trace and ``peak`` is that
    card's maximum.  :func:`free_bytes` is a separate probe on CUDA's CURRENT
    device and can name a different card under ``CUDA_VISIBLE_DEVICES``.
    """

    def __init__(self, what=None, interval=0.05, device=0):
        self.requested = _resolve_fields(what)
        self.interval = max(0.001, float(interval))
        self.device = _device_index(device)
        self._gpu_cols = tuple(f for f in GPU_FIELDS if f in self.requested)
        self._want_gpu = bool(self._gpu_cols)
        self._want_host = any(f in self.requested for f in HOST_FIELDS)
        self._host_only = (not self._want_gpu) and self._want_host
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._samples = []
        self._marks = []
        self._closed = False
        self._results = None
        self._proc = None
        self._thread = None
        self._stop = threading.Event()
        self._first_sample = threading.Event()
        _LIVE.add(self)
        _ensure_atexit()
        try:
            self._start()
        except Exception:                 # a monitor that cannot sample is empty
            try:
                self._reap()
            except Exception:
                pass
            self._proc = None

    # -- lifecycle -------------------------------------------------------
    def _start(self):
        proc = None
        if self._want_gpu:
            exe = shutil.which("nvidia-smi")
            if exe:
                ms = max(1, int(round(self.interval * 1000.0)))
                cols = ",".join(_FIELD_QUERY[f] for f in self._gpu_cols)
                cmd = [exe, f"--query-gpu={cols}",
                       "--format=csv,noheader,nounits", "-lms", str(ms),
                       "-i", str(self.device)]
                try:
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, text=True, bufsize=1,
                        close_fds=True, start_new_session=True)
                except OSError:
                    proc = None
        self._proc = proc
        if proc is not None or (self._want_gpu and self._want_host):
            # A child to drain, or a GPU request that failed and still wants
            # host samples: one reader thread, never a per-sample fork.
            th = threading.Thread(target=self._reader, name="tabpwa-monitor",
                                  daemon=True)
            th.start()
            self._thread = th
        elif self._host_only:
            self._record(None)            # in-process: no child, no pipe, no thread
        else:
            self._first_sample.set()      # nothing will ever be sampled

    def _reader(self):
        """Drain the child's stdout; read host status at the same cadence.

        This is ONE reader thread, not a sampler: the GPU cadence is the
        child's ``-lms`` loop.  If the child is absent or dies, the same
        thread falls back to host-only sampling on ``self.interval``.
        """
        proc = self._proc
        if proc is not None:
            try:
                for line in proc.stdout:
                    self._record_gpu_line(line)
                    if self._stop.is_set():
                        break
            except (OSError, ValueError):
                pass
        if not self._want_host:
            return
        while not self._stop.wait(self.interval):
            self._record(None)

    def _record_gpu_line(self, line):
        parts = line.split(",")
        cols = self._gpu_cols
        if len(parts) < len(cols):
            return False
        gpu = dict(zip(cols, (_num(p) for p in parts)))
        if all(v is None for v in gpu.values()):
            return False
        self._record(gpu)
        return True

    def _make_sample(self, gpu):
        sample = {f: None for f in self.requested}
        if gpu:
            for key, value in gpu.items():
                if key in sample:
                    sample[key] = value
        if self._want_host:
            rss, hwm = _host_mib()
            if "rss" in sample:
                sample["rss"] = rss
            if "hwm" in sample:
                sample["hwm"] = hwm
        return sample

    def _record(self, gpu):
        t = self._now()
        sample = self._make_sample(gpu)
        with self._lock:
            if self._results is not None:
                return
            self._samples.append((t, sample))
        self._first_sample.set()

    def _reap(self):
        """terminate → wait → kill → close, then drop the child reference LAST.

        The ``try/finally`` is the interrupt safety: if a ``BaseException``
        (``KeyboardInterrupt``, ``SystemExit``) lands anywhere in the sequence,
        ``self._proc`` is still cleared, so no later ``close()``/atexit pass
        can find a half-reaped child (and the reader thread is still joined).
        """
        proc = self._proc
        try:
            if proc is not None:
                try:
                    if proc.poll() is None:
                        proc.terminate()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
                try:
                    if proc.stdout is not None:
                        proc.stdout.close()
                except OSError:
                    pass
        finally:
            th, self._thread = self._thread, None
            if th is not None and th is not threading.current_thread():
                th.join(timeout=2.0)
            self._proc = None

    def close(self):
        """Stop sampling, reap the child, return the final :class:`Results`.

        Safe on normal exit, after an exception, and on a second call (which
        returns the same snapshot).
        """
        with self._close_lock:
            if self._results is not None:
                return self._results
            with self._lock:
                self._closed = True
            if self._host_only:
                self._record(None)        # final settled HWM/RSS for this mode
            self._stop.set()
            self._first_sample.set()
            self._reap()
            with self._lock:
                self._results = self._snapshot()
            try:
                _LIVE.discard(self)
            except TypeError:              # pragma: no cover - unhashable
                pass
            return self._results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False                    # never swallow the body's exception

    def __del__(self):                  # pragma: no cover - timing dependent
        try:
            self.close()
        except BaseException:
            pass

    # -- recording -------------------------------------------------------
    def _now(self):
        return time.monotonic() - self._t0

    def mark(self, label):
        """Record a phase boundary ``(t, label)``.  Errors after ``close()``.

        TIMING.  The boundary is stamped NOW, but a sample's timestamp is its
        **drain** time — when the reader thread picked the line up — so the
        measurement behind a sample can be up to one ``interval`` older than
        its stamp.  And ``peak``/``series`` slice ``start <= t < end``, so a
        sample stamped exactly on a boundary belongs to the NEXT phase.  A
        ``mark()`` is NOT a barrier: it never waits for the child.

        A label is used ONCE.  A repeated label starts a NEW phase that the
        phase-based accessors do not reach (``_bounds`` finds the first
        occurrence, ``summary`` lists each label once), so samples after the
        repeat are only visible through the phase-less calls.
        """
        if self._host_only:
            self._record(None)            # a sample just before the boundary
        with self._lock:
            if self._closed:
                raise RuntimeError("monitor is closed; mark() is not allowed")
            self._marks.append((self._now(), str(label)))

    def wait_ready(self, timeout=5.0):
        """Block until a sample exists (or none ever will, or ``timeout``).

        True means "a sample has been taken, OR this monitor can never take
        one" (``what=()``, or a request with no GPU child and no host field) —
        it is a readiness probe, NOT proof that a value arrived.  So a gate
        that needs a number still checks it after: ``peak(...) is None``.
        """
        return self._first_sample.wait(timeout)

    # -- snapshots -------------------------------------------------------
    def _snapshot(self):
        duration = self._samples[-1][0] if self._samples else self._now()
        return Results(self._samples, self._marks, duration, self.requested)

    def get_results(self):
        """An immutable snapshot.  Legal mid-run; after ``close()`` the final one."""
        with self._lock:
            if self._results is not None:
                return self._results
        if self._host_only:
            self._record(None)            # refresh RSS/HWM for the snapshot
        with self._lock:
            if self._results is not None:
                return self._results
            return self._snapshot()


def create_monitor(what=None, interval=0.05, device=0):
    """Start a :class:`Monitor` over ``what`` (``None`` = every field).

    ``what`` is an iterable of field names; unknown names raise ``ValueError``.
    A request with no ``gpu_*`` field runs host-only in-process (no child, no
    thread).  Start failures degrade to an empty monitor — never raise.

    ``device`` (keyword, default ``0``) is the GPU the sampler reads: it
    becomes ``nvidia-smi -i <device>``, so exactly ONE card is sampled (without
    it, ``--query-gpu`` emits a line per GPU every interval and the series
    would interleave devices).  Note :func:`free_bytes` is a different
    selector — it reads CUDA's CURRENT device, so under
    ``CUDA_VISIBLE_DEVICES`` the two can name different physical GPUs.
    """
    fields = _resolve_fields(what)          # ValueError must escape
    dev = _device_index(device)             # ... and so must this one
    try:
        return Monitor(what=fields, interval=interval, device=dev)
    except Exception:                   # pragma: no cover - defensive
        mon = Monitor.__new__(Monitor)
        mon.requested = tuple(fields)
        mon.interval = max(0.001, float(interval))
        mon.device = dev
        mon._gpu_cols = ()
        mon._want_gpu = False
        mon._want_host = any(f in fields for f in HOST_FIELDS)
        mon._host_only = mon._want_host
        mon._t0 = time.monotonic()
        mon._lock = threading.Lock()
        mon._close_lock = threading.Lock()
        mon._samples = []
        mon._marks = []
        mon._closed = True
        mon._results = Results((), (), 0.0, fields)
        mon._proc = None
        mon._thread = None
        mon._stop = threading.Event()
        mon._first_sample = threading.Event()
        mon._first_sample.set()
        return mon

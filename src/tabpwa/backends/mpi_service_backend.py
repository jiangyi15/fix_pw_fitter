"""``mpi_service`` backend — the program runs ONCE; the MPI job is a child.

    backend: {name: mpi_service, local: cuda_v4_pwa, n_procs: 4}

The driver process is MPI-free: this module imports no ``mpi4py`` and reads no
rank.  It starts ``mpiexec -n N python -m tabpwa.mpi_worker`` lazily on the first
``load_data``, talks to rank 0 over a Unix SOCKET (whose path travels in argv —
the launcher owns the child's stdio, so it cannot carry the protocol) with the
frames of :mod:`tabpwa._ipc`, and forwards ``load`` / ``compute`` / ``free`` /
``shutdown``.  Inside the child, every rank builds the ORDINARY ``mpi`` backend,
so slicing, device binding, the first-build barrier and the reduction exist
once (`docs/mpi_backend_design.md`).

Because there is one program process, output is written once and the package
needs no rank or writer concept (`docs/mpi_service_design.md`).

Not usable INSIDE an MPI/scheduler step: launching another launcher there would
nest.  That case uses the all-ranks ``mpi`` backend instead; this backend raises
and says so.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from tabpwa._ipc import join_tree, read_msg, split_tree, write_msg

from .core import ComputeBackend, register_backend

PROTOCOL = 1

#: environment that proves we are already inside an MPI/scheduler step
_NESTED_ENV = ("PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK",
               "MV2_COMM_WORLD_RANK", "SLURM_STEP_ID")

_TAIL_LINES = 200


class _ServiceHandle:
    """One server-side sample: an integer id and the event count."""

    __slots__ = ("id", "n", "_backend")

    def __init__(self, handle_id, n, backend):
        self.id = int(handle_id)
        self.n = int(n)
        self._backend = backend

    def free(self):
        self._backend._free_handle(self.id)

    def __repr__(self):
        return f"ServiceHandle(id={self.id}, n={self.n})"


@register_backend("mpi_service", tags=("delegate", "external"))
class MPIServiceBackend(ComputeBackend):
    """Driver half: a child `mpiexec` job, driven over a pipe."""

    def __init__(self, kernel_config, local=None, model=None, n_procs=None,
                 launcher="auto", timeout=600.0, device="auto", align=None,
                 weights=None):
        from .mpi_backend import MPIBackend

        if n_procs is None:
            raise ValueError(
                "mpi_service: n_procs is required — the driver cannot infer how "
                "many GPUs this job may take")
        self.n_procs = int(n_procs)
        if self.n_procs < 1:
            raise ValueError(f"mpi_service: n_procs={self.n_procs} must be >= 1")

        nested = [name for name in _NESTED_ENV if os.environ.get(name)]
        if nested:
            raise RuntimeError(
                "mpi_service: already inside an MPI/scheduler step "
                f"({', '.join(nested)}) — starting another launcher would nest. "
                "Use the all-ranks backend `mpi` (launched by srun/mpiexec) "
                "inside an allocation, or run this service outside the step.")

        self.kernel_config = kernel_config
        self.local = {"name": "default"} if local is None else local
        MPIBackend._refuse_shard(self.local)
        MPIBackend._require_leaf(self.local, model)
        self.model = model
        self.launcher = launcher
        self.timeout = float(timeout)
        self.device = device
        self.align = align
        self.weights = weights

        self._proc = None
        self._sock = None
        self._stream = None
        self._dir = None
        self._next_id = 0
        self._closed = False
        self._dead = None
        self._tail = []
        self._tail_lock = threading.Lock()
        self._reader = None
        self._exit_hook = None

    # ── child lifecycle ───────────────────────────────────────────────
    def _resolve_launcher(self):
        choice = self.launcher
        if choice == "auto":
            choice = os.environ.get("TABPWA_MPI_LAUNCHER")
        if choice in (None, "", "auto"):
            for name in ("mpiexec", "mpirun"):
                path = shutil.which(name)
                if path:
                    return path
            raise RuntimeError(
                "mpi_service: no `mpiexec`/`mpirun` on PATH (set "
                "TABPWA_MPI_LAUNCHER or the launcher= keyword)")
        path = shutil.which(choice) or (choice if os.path.exists(choice) else None)
        if not path:
            raise RuntimeError(f"mpi_service: launcher {choice!r} not found")
        return path

    def _argv(self, launcher, addr):
        device = self.device
        if device is None:
            device_txt = "none"
        else:
            device_txt = "auto" if device == "auto" else str(int(device))
        argv = [launcher, "-n", str(self.n_procs), sys.executable,
                "-m", "tabpwa.mpi_worker",
                "--local", json.dumps(self.local),
                "--device", device_txt,
                "--protocol", str(PROTOCOL), "--addr", addr]
        if self.align is not None:
            argv += ["--align", str(int(self.align))]
        if self.weights is not None:
            argv += ["--weights", json.dumps([float(w) for w in self.weights])]
        return argv

    def _ensure_started(self):
        if self._proc is not None:
            return
        if self._closed:
            raise RuntimeError(
                "mpi_service: this backend was shut down (free()) — a silent "
                "restart would hide the failure; build a new backend")
        if self._dead is not None:
            raise RuntimeError(
                f"mpi_service: the channel to the worker is broken "
                f"({self._dead}); build a new backend")
        try:
            self._start()
        except BaseException:
            # a failure BETWEEN mkdtemp and a live Popen leaks the directory
            # (and a listening socket) unless it is cleaned here
            self._kill()
            raise

    def _start(self):
        launcher = self._resolve_launcher()
        # Hydra/orted own the child's stdio, so the channel is a Unix socket
        # whose PATH travels in argv (no fd inheritance through mpiexec)
        self._dir = tempfile.mkdtemp(prefix="tabpwa-svc-")
        addr = os.path.join(self._dir, "io.sock")
        if len(addr) > 100:
            raise RuntimeError(
                f"mpi_service: socket path too long ({addr}); set TMPDIR to a "
                f"short directory")
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(addr)
        self._sock.listen(1)
        self._sock.settimeout(self.timeout)
        self._proc = subprocess.Popen(
            self._argv(launcher, addr), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._reader.start()
        self._exit_hook = atexit.register(self._kill)
        self._sock.settimeout(1.0)
        deadline = time.monotonic() + min(60.0, self.timeout)
        while True:
            try:
                conn, _ = self._sock.accept()
                break
            except socket.timeout:
                if self._proc.poll() is not None:
                    self._kill()
                    raise RuntimeError(
                        "mpi_service: the worker exited before connecting\n"
                        f"{self._stderr_tail()}") from None
                if time.monotonic() > deadline:
                    self._kill()
                    raise TimeoutError(
                        "mpi_service: the worker did not connect within "
                        f"{min(60.0, self.timeout):.0f}s\n{self._stderr_tail()}")
        conn.settimeout(self.timeout)
        self._stream = conn.makefile("rwb")
        hello, _hello_arrays = self._read_reply()
        if hello.get("protocol") != PROTOCOL:
            self._kill()
            raise RuntimeError(
                f"mpi_service: worker protocol {hello.get('protocol')!r}, "
                f"driver speaks {PROTOCOL}")
        if int(hello.get("n_ranks", 0)) != self.n_procs:
            self._kill()
            raise RuntimeError(
                f"mpi_service: worker reports {hello.get('n_ranks')} ranks, "
                f"expected {self.n_procs}")
        if hello.get("local") != self.local:
            self._kill()
            raise RuntimeError(
                f"mpi_service: the worker runs {hello.get('local')!r}, this "
                f"backend asked for {self.local!r}")

    def _drain_stderr(self):
        stream = self._proc.stderr
        for line in iter(stream.readline, b""):
            with self._tail_lock:
                self._tail.append(line.decode(errors="replace").rstrip())
                del self._tail[:-_TAIL_LINES]

    def _stderr_tail(self, limit=20):
        with self._tail_lock:
            return "\n".join(self._tail[-limit:])

    def _broken(self, why):
        """Mark the channel unusable: a partial frame cannot be resynchronised."""
        self._dead = why
        return why

    def _read_reply(self):
        if self._dead is not None:
            raise RuntimeError(
                f"mpi_service: the channel to the worker is broken "
                f"({self._dead}); build a new backend")
        try:
            msg = read_msg(self._stream)
        except socket.timeout as exc:
            why = self._broken(f"no reply within {self.timeout}s")
            raise TimeoutError(
                f"mpi_service: {why}\n{self._stderr_tail()}") from exc
        except (EOFError, OSError) as exc:
            why = self._broken(f"the worker closed the channel: {exc}")
            raise RuntimeError(
                f"mpi_service: {why}\n{self._stderr_tail()}") from exc
        if msg is None:
            raise RuntimeError(
                f"mpi_service: worker exited\n{self._stderr_tail()}")
        meta, arrays = msg
        if "error" in meta:
            raise RuntimeError(f"mpi_service: worker failed\n{meta['error']}")
        return meta, arrays

    def _call(self, meta, arrays=None):
        self._ensure_started()
        try:
            write_msg(self._stream, meta, arrays)
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(
                f"mpi_service: cannot reach the worker ({exc})\n"
                f"{self._stderr_tail()}") from exc
        return self._read_reply()

    def _kill(self):
        """Stop the child (shutdown → terminate → kill) and remove its artefacts.

        A healthy worker answers ``shutdown`` in ~0.06 s, so the courtesy wait is
        SHORT: the long one this replaces cost 40 s on a FROZEN child (a worker
        blocked in a CUDA compute cannot read its socket) at ``__del__``/Ctrl+C
        time.  Every step also reaps, so no zombie is left to the next spawn.
        """
        proc, self._proc = self._proc, None
        try:
            if proc is not None and proc.poll() is None:
                try:
                    write_msg(self._stream, {"op": "shutdown"})
                except Exception:
                    pass
                for action in (None, proc.terminate, proc.kill):
                    if action is not None:
                        try:
                            action()
                        except Exception:
                            pass
                    try:
                        proc.wait(timeout=2.0 if action is None else 5.0)
                        break
                    except subprocess.TimeoutExpired:
                        continue
        finally:
            self._cleanup(proc)
            if self._exit_hook is not None:
                try:
                    atexit.unregister(self._exit_hook)
                except Exception:
                    pass
                self._exit_hook = None

    def _cleanup(self, proc=None):
        """Close the channel and the socket, and delete the private directory."""
        for stream in (self._stream, self._sock,
                       None if proc is None else proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        self._stream = None
        self._sock = None
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None

    # ── ComputeBackend ────────────────────────────────────────────────
    def load_data(self, data_np):
        self._ensure_started()
        handle_id = self._next_id
        self._next_id += 1

        skeleton, arrays = split_tree(self.kernel_config, prefix="kc")
        for key, value in data_np.items():
            arrays[str(key)] = np.asarray(value)
        meta, _ = self._call({
            "op": "load", "id": handle_id, "keys": list(data_np),
            "kc": skeleton, "model": self.model}, arrays)
        return _ServiceHandle(handle_id, meta["n"], self)

    def compute(self, params, data_handle, norm=None, return_p=True):
        arrays = {"p__" + str(k): np.asarray(v) for k, v in params.items()}
        meta, out = self._call({
            "op": "compute", "id": data_handle.id,
            "norm": None if norm is None else float(norm),
            "return_p": bool(return_p)}, arrays)
        grads = {k[3:]: v for k, v in out.items() if k.startswith("g__")}
        for name, value in meta.get("grads_scalars", {}).items():
            grads[name] = float(value)
        for name in meta.get("none", ()):
            grads[name] = None
        P = out.get("P") if return_p else None
        return float(meta["Q"]), grads, P

    def _free_handle(self, handle_id):
        if self._proc is None:
            return
        try:
            self._call({"op": "free", "id": int(handle_id)})
        except RuntimeError:
            pass

    def free(self):
        self._closed = True
        self._kill()

    def describe(self):
        state = "running" if self._proc is not None else "not started"
        return (f"mpi_service -> {self.local} x{self.n_procs} "
                f"({state}, program is a single process)")

    def __del__(self):
        try:
            self._kill()
        except Exception:
            pass
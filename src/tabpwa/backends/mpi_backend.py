"""``mpi`` backend — N ranks, one slice each, summed by ``allreduce``.

    mpiexec -n <#GPUs> python run_fit.py --config … --backend mpi

Every rank is a full tabpwa process holding ONE contiguous slice of the sample
(``split_offsets``, the same weighted/align-aware split ``shard`` uses).  A
compute runs on that slice and the result is reduced:

    Q, grads, P = local.compute(params, slice, norm, return_p)
    Q           = allreduce(Q)                  # SUM
    grads[k]    = allreduce(grads[k])           # SUM, incl. grads["norm"]
    P           = allgather(P)                  # ONLY when return_p

``Q``, every gradient block and ``grads["norm"]`` are sums over disjoint events,
so the reduced value is the full-sample objective (association only).  The
per-event ``P`` is never requested by a fit (``get_nll_raw`` and
``_compute_norm_batched`` pass ``return_p=False``); when asked it is allgathered
because every rank must see the same full array (a gather desynchronises a
``generate_toy`` driven through the fitter).

Rank → device: the node-local rank (``COMM_TYPE_SHARED``) selects
``CUDA_VISIBLE_DEVICES`` BEFORE the local backend is built; a launcher-set
``CUDA_VISIBLE_DEVICES`` wins (no double mapping); a rank whose node-local rank
exceeds the visible GPU count raises (on a host with no NVIDIA device ``auto``
leaves the binding alone, so a CPU local backend stays usable).  The first build
is serialised (rank 0 builds, then a barrier) so N ranks sharing an install do
not compile the same kernel concurrently.

The FIRST ``compute`` of a process broadcasts the parameter dict from rank 0
(one small collective): ``initial_values()`` is unseeded per process, and every
rank must reduce the SAME point or the summed gradient is meaningless.

A failure on one rank outside ``compute`` (a bad device index, a slice with no
rows, a broken local build) would leave the others blocked in a collective, so
every such site replays it through ``comm.Abort`` — the job dies with the rank's
message instead of hanging.

Design: ``docs/mpi_backend_design.md``; as built and measured:
``docs/mpi_backend.md``; Slurm: ``docs/slurm.md``.  This is the CO-LOCATED
deployment; the remote (heterogeneous) transport is a draft standard,
``docs/remote_protocol_design.md``.

    backend: {name: mpi, local: cuda_v4_pwa}

``local`` MUST name ONE concrete kernel (a leaf): a delegate/composer is
refused — ``auto`` may resolve to a different kernel per rank, and a wrapper's
per-handle state (a fixed-coefficient mask, a Gram cache) is not forwarded to
the ranks.  ``local=None`` resolves the model's ``default``, which must itself
be a leaf.

Every rank MUST construct the backend (``Split_type``/``Barrier`` are
collective), and every rank must call ``compute`` the same number of times.
``align`` defaults to the local backend's ``resolution_size`` (a grouped
objective's group must never straddle two ranks); pass it for a kernel that does
not expose one.  Rank-0-only I/O (saving, plotting) is the CALLER's job.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from .core import ComputeBackend, register_backend


def _mpi():
    try:
        from mpi4py import MPI
    except ImportError as exc:          # optional dependency, named
        raise RuntimeError(
            "the mpi backend needs mpi4py: pip install 'tabpwa[mpi]'") from exc
    return MPI


_GPU_COUNT = None


def _gpu_count():
    """Number of NVIDIA devices visible to this process (0 when unknown).

    A transient probe failure is NOT cached, so a later call can retry.
    """
    global _GPU_COUNT
    if _GPU_COUNT is None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10)
            if out.returncode != 0:
                return 0
            _GPU_COUNT = len([ln for ln in (out.stdout or "").splitlines()
                              if ln.strip()])
        except Exception:
            return 0
    return _GPU_COUNT


def device_index(local_rank, device, n_gpu, visible_env):
    """``CUDA_VISIBLE_DEVICES`` value for a rank, or None to leave it alone.

    * a SINGLE rank (a process not launched by MPI) with ``device="auto"``
      binds nothing: device 0 is already the default, and setting the variable
      would hide the other GPUs from the caller's process;
    * a launcher-set ``CUDA_VISIBLE_DEVICES`` is respected as follows
      (Slurm sets it for every job):
        - empty             -> "hide every device", leave it;
        - ONE entry         -> the launcher assigned this task its own device
                               (``--gpus-per-task=1``), leave it;
        - SEVERAL entries   -> a NODE-level list (``--gres=gpu:4``): pick
                               ``entries[local_rank]``, so the ranks do not all
                               land on the first visible device; a rank past the
                               list raises;
    * ``device=None`` means "do not bind" (CPU ranks, tests, or a launcher that
      did the binding);
    * ``device="auto"`` binds the node-local rank and REFUSES a rank whose index
      exceeds the visible GPU count — the alternative is a CUDA driver failure
      far from the cause (a host with NO GPU leaves the binding alone);
    * an int binds that device on every rank (oversubscription/testing).
    """
    if visible_env is not None:
        entries = [e.strip() for e in str(visible_env).split(",") if e.strip()]
        if len(entries) <= 1:
            # "" hides every device; one entry is this task's own assignment
            return None
        if local_rank >= len(entries):
            raise ValueError(
                f"mpi: rank {local_rank} has no device: CUDA_VISIBLE_DEVICES="
                f"{visible_env!r} lists {len(entries)} — allocate one GPU per "
                f"task (--gpus-per-task) or launch fewer ranks per node")
        return entries[local_rank]
    if device is None:
        return None
    if n_gpu <= 0:
        return None                     # no GPU here: a CPU local backend is fine
    idx = local_rank if device == "auto" else int(device)
    if idx >= n_gpu:
        raise ValueError(
            f"mpi: a rank asks for device {idx} but only {n_gpu} GPU(s) are "
            f"visible — launch with -n <= {n_gpu}, or pass device=None/int")
    return str(idx)


class MPIDataHandle:
    """The local handle plus the slice this rank owns."""

    __slots__ = ("local", "n", "lo", "hi")

    def __init__(self, local, n, lo, hi):
        self.local = local
        self.n = int(n)
        self.lo = int(lo)
        self.hi = int(hi)

    def free(self):
        free = getattr(self.local, "free", None)
        if free is not None:
            free()

    def __repr__(self):
        return f"MPIDataHandle(n={self.n}, slice=[{self.lo}, {self.hi}))"


@register_backend("mpi", tags=("delegate", "external"))
class MPIBackend(ComputeBackend):
    """One slice per rank; the sums come back through ``allreduce``."""

    def __init__(self, kernel_config, local=None, model=None, device="auto",
                 align=None, weights=None, build_first=True):
        MPI = _mpi()
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self._params_ready = False
        self._align_arg = align

        if os.environ.get("TABPWA_SHARD_WORKER"):
            raise ValueError(
                "mpi: an MPI backend inside a `shard` worker is unsafe (fork "
                "after MPI_Init) — use one MPI rank per GPU, or shard without "
                "MPI")

        self.local_spec = {"name": "default"} if local is None else local
        self._refuse_shard(self.local_spec)
        self._require_leaf(self.local_spec, model)

        # node-local rank: rank r on a 4-GPU host binds device r
        node = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
        self.local_rank = node.Get_rank()
        node.Free()

        try:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if self.size == 1 and device == "auto" and visible is None:
                # one rank already means device 0; binding would only HIDE the
                # other GPUs from a process that is not a real MPI job
                idx = None
            else:
                idx = device_index(self.local_rank, device, _gpu_count(),
                                   visible)
            if idx is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = idx
            self.weights = self._weights(weights)

            from .core import create_backend
            self.local = None
            if build_first and self.rank == 0:
                # serialise the FIRST build: N ranks sharing an install must
                # not compile the same kernel concurrently
                self.local = create_backend(self.local_spec, kernel_config,
                                            model=model)
            if build_first:
                self.comm.Barrier()
            if self.local is None:
                self.local = create_backend(self.local_spec, kernel_config,
                                            model=model)
            if build_first:
                self.comm.Barrier()
        except BaseException as exc:    # the peers would wait in Barrier above
            self._abort(exc)

        self.align = self._derive_align()

    # -- construction helpers -----------------------------------------
    def _abort(self, exc):
        """Report a rank-local failure and end the JOB (never hang the peers).

        With ONE rank there are no peers to deadlock, so the failure is RAISED
        (a plain script calling ``create_backend("mpi", device=99)`` must get a
        catchable error, not have its interpreter taken down).
        """
        print(f"tabpwa mpi: rank {self.rank}/{self.size} failed: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.stderr.flush()
        if self.size == 1:
            raise exc
        self.comm.Abort(1)

    @staticmethod
    def _refuse_shard(spec):
        name = spec.get("name") if isinstance(spec, dict) else spec
        if name == "shard":
            raise ValueError(
                "mpi: nesting `shard` inside an MPI rank is unsafe (fork after "
                "MPI_Init, and CUDA in the parent) — use one MPI rank per GPU, "
                "or shard without MPI")

    @staticmethod
    def _require_leaf(spec, model):
        """`local` is ONE concrete kernel: a composer cannot be replicated.

        `auto` may choose a different kernel per rank (availability differs) and
        a wrapper's per-handle state (mask, Gram) is not forwarded, so the run
        would silently be a mixture.  The capability tag is the rule, not a
        name list.
        """
        from .core import backend_tags

        if not isinstance(spec, (str, dict)):
            raise ValueError(
                f"mpi: local must be a backend NAME or spec dict, got "
                f"{type(spec).__name__} — every rank must build its own")
        name = spec.get("name") if isinstance(spec, dict) else spec
        if name is not None and "delegate" in backend_tags(name, model):
            raise ValueError(
                f"mpi: local backend {name!r} is a delegate/composer — an MPI "
                f"run must name ONE concrete kernel per rank (auto may resolve "
                f"differently per rank; a wrapper's per-handle state is not "
                f"forwarded).  Use e.g. cuda_v4_pwa / numpy_flat.")

    def _weights(self, weights):
        if weights is None:
            return [1.0] * self.size
        w = [float(v) for v in weights]
        if len(w) != self.size:
            raise ValueError(
                f"mpi: weights has {len(w)} entries for {self.size} ranks")
        if any(v < 0 for v in w) or sum(w) <= 0:
            raise ValueError("mpi: weights must be >= 0 with a positive sum")
        return w

    def _derive_align(self):
        """The local objective's group size, from the widest source available.

        ``shard`` makes the caller pass ``align``; here it can be derived.  Some
        kernels expose ``resolution_size``, the v5 family only stores
        ``_resolution_size``, and a flat kernel may expose neither — then the
        SPEC is the record (a grouped local must be asked for it there).
        """
        if self._align_arg is not None:
            return int(self._align_arg)
        rs = getattr(self.local, "resolution_size", None)
        if rs is None:
            rs = getattr(self.local, "_resolution_size", None)
        if rs is None and isinstance(self.local_spec, dict):
            rs = self.local_spec.get("resolution_size")
        return int(rs or 1)

    # -- ComputeBackend ------------------------------------------------
    def load_data(self, data_np):
        from .shard_backend import split_offsets

        try:
            n = int(np.asarray(data_np["mass"]).shape[0])
            bounds = split_offsets(n, self.weights, self.align)
            lo, hi = int(bounds[self.rank]), int(bounds[self.rank + 1])
            if hi <= lo:
                raise ValueError(
                    f"mpi: rank {self.rank}/{self.size} owns no rows (n={n}, "
                    f"align={self.align}, weights={self.weights}) — the pwa "
                    f"kernels do not load an empty sample; lower align/"
                    f"resolution_size, launch fewer ranks, or rebalance weights")
            part = {}
            for k, v in data_np.items():
                a = np.asarray(v)
                part[k] = a[lo:hi] if a.ndim and a.shape[0] == n else v
            return MPIDataHandle(self.local.load_data(part), n, lo, hi)
        except BaseException as exc:    # a raise on one rank would hang the rest
            self._abort(exc)

    def compute(self, params, data_handle, norm=None, return_p=True):
        MPI = self.MPI
        try:
            if not self._params_ready:
                # rank 0's point is THE point (initial_values() is unseeded per
                # process; a divergence makes the summed gradient meaningless)
                params = self.comm.bcast(params, root=0)
                self._params_ready = True

            q, grads, p = self.local.compute(
                params, data_handle.local, norm=norm, return_p=return_p)

            # participation must be rank-invariant, so agree on `None` blocks
            # once rather than letting one rank skip a collective
            local_none = sorted(k for k, v in grads.items() if v is None)
            none_keys = self.comm.bcast(local_none, root=0)
            if local_none != none_keys:
                raise RuntimeError(
                    f"mpi: gradient blocks with a null value differ across "
                    f"ranks: {local_none} vs {none_keys}")

            q = self.comm.allreduce(float(q), op=MPI.SUM)
            total = {}
            for name, value in grads.items():
                if value is None:
                    total[name] = None
                elif np.isscalar(value):
                    total[name] = self.comm.allreduce(float(value), op=MPI.SUM)
                else:
                    total[name] = self.comm.allreduce(np.asarray(value),
                                                       op=MPI.SUM)

            if return_p:
                if p is None:
                    raise RuntimeError(
                        "mpi: return_p=True but this rank produced no P (the "
                        "allgather must be collective)")
                # object allgather: slices differ in length, contiguous in rank
                # order, so concatenation restores the sample order
                p = np.concatenate(self.comm.allgather(np.asarray(p)))
        except BaseException as exc:    # every other rank is inside a collective
            self._abort(exc)
        return q, total, p


    def free(self):
        try:
            self.local.free()
        except Exception:
            pass

    def describe(self):
        dev = os.environ.get("CUDA_VISIBLE_DEVICES")
        return (f"mpi[{self.rank}/{self.size}] -> {self.local_spec} "
                f"(visible device {dev if dev not in (None, '') else 'unbound'}, "
                f"align {self.align})")

    def __del__(self):
        self.free()
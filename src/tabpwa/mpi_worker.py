"""Worker entry for the ``mpi_service`` backend.

Launched by the driver as::

    mpiexec -n N python -m tabpwa.mpi_worker --local '{"name": "cuda_v4_pwa"}' \
            --addr /tmp/tabpwa-svc-XXXX/io.sock

Every rank builds the ORDINARY ``mpi`` backend of
``docs/mpi_backend_design.md`` — the slice, the rank→device rule, the serialised
first build, the ``allreduce``/``allgather`` reduction and the ``Abort`` policy
are that backend's.  This module adds only the command loop: rank 0 reads one
framed message from the driver over a Unix SOCKET (``--addr``), the whole group
executes it, and rank 0 writes the reply on the same connection.  The kernel
config arrives with the first ``load``, because the inner backend cannot be built
before it exists; a later ``load`` must carry the SAME one (this service serves
one model).

The launcher (Hydra/orted) owns the ranks' stdio, so it cannot carry the
protocol — and fd 1 is redirected to stderr on EVERY rank, because all ranks
inherit the launcher's pipe.  On a clean exit rank 0 also removes its socket and
the (empty) directory, so a killed DRIVER leaves no artefact behind.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import os
import socket
import sys
import time
import traceback

from tabpwa._ipc import join_tree, read_msg, write_msg

PROTOCOL = 1


def _device_from_text(text):
    if text is None or str(text).lower() in ("none", "null"):
        return None
    if text == "auto":
        return "auto"
    return int(text)


def _digest(obj):
    """A stable digest of the kernel config (a changed one is refused)."""
    import hashlib

    h = hashlib.sha1()

    def feed(v):
        if isinstance(v, dict):
            h.update(b"{")
            for k in sorted(v, key=str):
                h.update(str(k).encode())
                feed(v[k])
            h.update(b"}")
        elif isinstance(v, (list, tuple)):
            h.update(b"[")
            for x in v:
                feed(x)
            h.update(b"]")
        elif isinstance(v, np.ndarray):
            h.update(f"{v.dtype}{v.shape}".encode())
            if v.size <= 100_000:
                h.update(np.ascontiguousarray(v).tobytes())
        else:
            h.update(repr(v).encode())

    feed(obj)
    return h.hexdigest()


def main(argv=None):
    # BEFORE argparse: a bad argv must be visible, and stdout is about to be
    # redirected (argparse would otherwise write its usage to DEVNULL)
    os.dup2(2, 1)
    ap = argparse.ArgumentParser(prog="python -m tabpwa.mpi_worker")
    ap.add_argument("--local", default='{"name": "default"}',
                    help="inner backend spec (a LEAF)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--align", type=int, default=None)
    ap.add_argument("--weights", default=None, help="JSON list, or omitted")
    ap.add_argument("--protocol", type=int, default=PROTOCOL)
    ap.add_argument("--addr", required=True,
                    help="Unix socket the driver listens on (path)")
    args = ap.parse_args(argv)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    stream = None
    if rank == 0:
        deadline = time.monotonic() + 30.0
        while True:
            sock = None
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(args.addr)
                stream = sock.makefile("rwb")
                break
            except OSError:
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
                if time.monotonic() > deadline:
                    comm.Abort(1)
                time.sleep(0.05)
    local = json.loads(args.local)
    weights = json.loads(args.weights) if args.weights else None
    device = _device_from_text(args.device)

    from tabpwa._version import __version__
    if rank == 0:
        try:
            write_msg(stream, {"protocol": args.protocol, "n_ranks": size,
                               "tabpwa": __version__, "local": local})
        except BaseException:                  # nobody can be told; end the job
            traceback.print_exc()
            comm.Abort(1)

    inner = None
    kc_digest = None
    handles = {}
    while True:
        try:
            # INSIDE the try: a truncated frame (a driver killed mid-write)
            # raises HERE, and rank 0 would otherwise die while every peer waits
            # in the bcast below — an N-GPU job with no error anywhere
            msg = read_msg(stream) if rank == 0 else None
            payload = comm.bcast(msg, root=0)      # every rank sees the same op
            if payload is None:                    # driver closed the socket
                break
            meta, arrays = payload
            op = meta.get("op")

            if op == "load":
                kc = join_tree(meta["kc"], arrays)
                digest = _digest(kc)
                if inner is None:
                    from tabpwa.backends import create_backend
                    kc_digest = digest
                    inner = create_backend(
                        {"name": "mpi", "local": local, "device": device,
                         "align": args.align, "weights": weights},
                        kc, model=meta.get("model"))
                elif digest != kc_digest:
                    raise ValueError(
                        "mpi_service: a second load carried a DIFFERENT kernel "
                        "config — one service backend serves ONE model; build a "
                        "new backend for another config")
                data = {k: arrays[k] for k in meta["keys"]}
                handle = inner.load_data(data)
                handles[meta["id"]] = handle
                if rank == 0:
                    write_msg(stream, {"ok": True, "n": int(handle.n)})

            elif op == "compute":
                handle = handles[meta["id"]]
                params = {k[3:]: v for k, v in arrays.items()
                          if k.startswith("p__")}
                q, grads, p = inner.compute(
                    params, handle, norm=meta.get("norm"),
                    return_p=bool(meta.get("return_p")))
                if rank == 0:
                    out = {}
                    scalars = {}
                    none = []
                    for name, value in grads.items():
                        if value is None:
                            none.append(name)
                        elif np.isscalar(value) or getattr(value, "shape", None) == ():
                            scalars[name] = float(value)
                        else:
                            out["g__" + name] = value
                    reply = {"ok": True, "Q": float(q),
                             "grads_scalars": scalars, "none": none}
                    if meta.get("return_p") and p is not None:
                        out["P"] = p
                    write_msg(stream, reply, out)

            elif op == "free":
                handle = handles.pop(meta["id"], None)
                if handle is not None:
                    handle.free()
                if rank == 0:
                    write_msg(stream, {"ok": True})

            elif op == "shutdown":
                if rank == 0:
                    write_msg(stream, {"ok": True})
                break

            else:
                raise ValueError(f"unknown op {op!r}")

        except BaseException:
            # a rank-local failure: report (rank 0) and end the JOB, or the
            # peers stay blocked in a collective
            traceback.print_exc()
            if rank == 0:
                try:
                    write_msg(stream, {"error": traceback.format_exc()})
                except Exception:
                    pass
            comm.Abort(1)

    for handle in handles.values():
        try:
            handle.free()
        except Exception:
            pass
    if stream is not None:
        try:
            stream.close()
        except Exception:
            pass
    if rank == 0:
        # a driver killed without a shutdown cannot clean up; the empty socket
        # directory is ours, so remove it on the way out
        try:
            os.unlink(args.addr)
        except OSError:
            pass
        try:
            os.rmdir(os.path.dirname(args.addr))
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""PEAK device memory of one backend vs sample size (the PWACG memory wall).

    python mem.py --backend cuda_flat_pwa_gen2                 # 10k/100k/300k/1M
    python mem.py --backend cuda_flat_pwa_gen2 --sizes 3000000
    python mem.py --backend cuda_flat_pwa_gen2 --max-phsp 1000000 --max-data 1000000

The reference framework exceeds 16 GB of accelerator memory at 3x10^6
integration events, which is its practical wall; this box has 12 GB.  The sweep
is one child process per size; each PEAK belongs to its size alone only when
the sizes are invoked SEPARATELY — in a combined run the previous child's
allocation is still counted at the next child's baseline (measured 0.77 vs
0.60 GiB for the same 1 M cell), and a negative phase is reported as a stale
baseline rather than printed as a number (no
carry-over from a previous, larger allocation) and a failure at one size does
not lose the smaller cells.

Two kinds of peak, because the two kernel families live in different memories:

* GPU (``nvidia-smi`` sampled OUT OF PROCESS by :mod:`tabpwa.monitor` at
  10 ms): a backend can allocate a staging/cache buffer inside one pass and
  free it before the next sample, so polling misses the peak.  The phases
  ``context`` / ``load`` / ``nll`` / ``resident`` are recorded with
  ``mark()``, plus the overall ``PEAK``.
* Host RSS (``VmHWM``, the kernel's own high-water mark, exact): reported for
  every backend, and it is the number that matters for the CPU kernels, whose
  caches are RAM.

A ``cpu_*`` backend requests ``("rss", "hwm")`` only: the monitor then starts
NO ``nvidia-smi`` child at all, so "skip the GPU part" is structural rather
than a hand-written condition.

**The baseline is taken before ``Fitter(...)``**, so ``context`` is the CUDA
context this backend creates and ``PEAK`` includes it.

The samples are the `make_data.py` files, capped with ``--max-data`` /
``--max-phsp``; the printed ``ne`` / ``nphsp`` are what was actually loaded, so
a request larger than the file shows up as a smaller line instead of a silent
"small sample".  A 3x10^6 point needs ``make_data.py --n-phsp 3000000`` first.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import extra_amp  # noqa: E402,F401  registers the benchmark line shapes

PHASES = ("context", "load", "nll", "resident")


def one_size(a):
    """Measure one (data, phsp) size in THIS process."""
    from tabpwa.monitor import GPU_FIELDS, create_monitor

    if a.backend == "shard":
        raise SystemExit("mem.py: a `shard` backend mixes GPU and CPU memory, and "
                         "the GPU worker lives in a child process — this script "
                         "would report the parent's RSS only.")
    cpu_family = ("cpu", "numpy", "onnx_cpu")
    gpu_backend = not a.backend.startswith(cpu_family)

    # A GPU backend asks for the GPU columns + the host high-water mark; a CPU
    # backend asks for host fields only, so the monitor starts NO child.
    if gpu_backend:
        mon = create_monitor(what=GPU_FIELDS + ("hwm",), interval=0.01,
                             device=a.device)
    else:
        mon = create_monitor(what=("rss", "hwm"), interval=0.01)
    mon.wait_ready(5.0)
    first = mon.get_results().peak("gpu_used") if gpu_backend else None
    gpu_ok = gpu_backend and first is not None
    baseline = first if first is not None else 0.0

    from tabpwa import Fitter
    spec = {"name": a.backend}
    if a.base:
        spec["base"] = a.base
    if a.rsize:
        spec["resolution_size"] = a.rsize

    mon.mark("context")
    f = Fitter(a.config, backend=spec)

    mon.mark("load")
    phsp = f.load_dataset("phsp", **({"max_events": a.max_phsp} if a.max_phsp else {}))
    data = f.load_dataset("data", **({"max_events": a.max_data} if a.max_data else {}))
    f.set_phsp(phsp)
    f.set_data(data)
    f.apply_constrains()
    if a.fix_mw or a.free_mw:
        keep = {s.strip() for s in a.free_mw.split(",") if s.strip()}
        known = set(dict.fromkeys(f.var_registry.flat_names))
        unknown = sorted(keep - known)
        if unknown:
            avail = sorted(n for n in known if n.endswith(("_mass", "_width")))
            raise SystemExit(f"mem.py: unknown --free-mw name(s): {unknown}\n"
                             f"  available: {avail}")
        mw = {n: float(f.defaults[n])
              for n in dict.fromkeys(f.var_registry.flat_names)
              if n.endswith(("_mass", "_width")) and n not in keep
              and n in f.defaults}
        f.set_fixed(mw)
    x0 = f.initial_values(seed=1)

    mon.mark("nll")
    for _ in range(a.repeat):
        f.get_nll(x0)

    mon.mark("resident")
    time.sleep(0.1)
    res = mon.close()

    per = {p: res.peak("gpu_used", phase=p) for p in PHASES} if gpu_ok else {}

    def fmt(v):
        # `-` for no sample: a phase shorter than one 10 ms interval must not
        # read as "used nothing".
        return "    -" if v is None else f"{v:5.2f}"

    if gpu_ok:
        def gib(mib):
            return None if mib is None else (mib - baseline) / 1024.0

        vals = [v for v in per.values() if v is not None]
        # A baseline taken in a machine that still holds a previous child's
        # allocation makes every phase NEGATIVE.  That is not a measurement, it
        # is a stale baseline: say so instead of printing "-0.13 GiB".
        if any(v is not None and v < 0 for v in vals):
            print(f"{a.backend:26s} ne={len(data['weight']):>9,} "
                  f"nphsp={len(phsp['weight']):>9,}  BASELINE STALE — a phase "
                  f"read negative ({min(v for v in vals if v is not None):+.2f} "
                  f"GiB); run one size per invocation (--sizes <n>)", flush=True)
            return
        peak = max(vals) if vals else None
        s = (f"context {fmt(gib(per['context']))} | load {fmt(gib(per['load']))}"
             f" | nll {fmt(gib(per['nll']))} | resident {fmt(gib(per['resident']))} | "
             f"PEAK {fmt(gib(peak))} GiB")
    elif gpu_backend:
        if shutil.which("nvidia-smi"):
            s = "no parseable nvidia-smi sample (every cell was [N/A])"
        else:
            s = "gpu monitoring unavailable (no nvidia-smi)"
    else:
        s = "cpu backend (no GPU sampling)"
    hwm = res.peak("hwm")
    hwm_s = "    -" if hwm is None else f"{hwm / 1024.0:5.2f}"
    ne, nb = len(data["weight"]), len(phsp["weight"])
    print(f"{a.backend:26s} x={len(x0):3d} ne={ne:>9,} nphsp={nb:>9,}  {s}"
          f"  | host RSS {hwm_s} GiB", flush=True)
    if a.max_phsp and nb < a.max_phsp:
        print(f"{'':26s} note: asked for {a.max_phsp:,} phase-space events, the "
              f"file holds {nb:,} — run make_data.py with a larger --n-phsp",
              flush=True)
    f.backend.free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config_gen.yml"))
    ap.add_argument("--backend", default="cuda_flat_pwa_gen_s0e2")
    ap.add_argument("--base", default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--rsize", type=int, default=0)
    ap.add_argument("--fix-mw", action="store_true")
    ap.add_argument("--free-mw", default="")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max-data", type=int, default=0)
    ap.add_argument("--max-phsp", type=int, default=0)
    ap.add_argument("--sizes", default="10000,100000,300000,1000000",
                    # NOTE: several sizes in ONE invocation share the sampler's
                    # baseline — the previous child's GPU allocation is still
                    # counted at the next child's start (measured: a 1 M cell read
                    # 0.77 GiB inside a combined run vs 0.60 GiB alone).  For a
                    # PEAK that belongs to its size, invoke one size at a time.
                    help="comma-separated sample sizes; one child process each "
                         "(default: the brief's 10k/100k/300k/1M).  Pass an "
                         "empty string to run the single --max-data/--max-phsp "
                         "cell in this process.")
    a = ap.parse_args()

    sizes = [int(s) for s in a.sizes.split(",") if s.strip()]
    if not sizes:
        one_size(a)
        return
    for n in sizes:
        cmd = [sys.executable, os.path.abspath(__file__),
               "--config", a.config, "--backend", a.backend,
               "--device", str(a.device), "--repeat", str(a.repeat),
               "--max-data", str(n), "--max-phsp", str(n), "--sizes", ""]
        for flag, val in (("--base", a.base), ("--free-mw", a.free_mw),
                          ("--rsize", a.rsize if a.rsize else None)):
            if val:
                cmd += [flag, str(val)]
        if a.fix_mw:
            cmd.append("--fix-mw")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"{n:>9,} events: child exited {r.returncode}", flush=True)


if __name__ == "__main__":
    main()

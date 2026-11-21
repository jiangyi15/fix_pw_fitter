#!/usr/bin/env python3
"""PEAK memory of one backend at the benchmark size.

    python mem.py --backend cuda_v4_pwa_s0e --free-mw a,b,c [...]
    python mem.py --config config_gen.yml --backend cpu_flat_pwa_gen_cached2

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
context this backend creates and ``PEAK`` includes it.  That is the honest
reading of the card's use.  README §8's historical "peak (in-process)" column
used the old 1 ms in-process ``cudaMemGetInfo`` thread, which had already
created a context before taking its baseline and therefore EXCLUDED it, so
those numbers are 0.13-0.16 GiB lower; compare a re-run against that table's
"external max" column instead.
"""
import argparse
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from tabpwa.monitor import GPU_FIELDS, create_monitor    # noqa: E402

PHASES = ("context", "load", "nll", "resident")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yml"))
    ap.add_argument("--backend", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--fix-mw", action="store_true")
    ap.add_argument("--free-mw", default="")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max-data", type=int, default=0)
    ap.add_argument("--max-phsp", type=int, default=0)
    a = ap.parse_args()

    if a.backend == "shard":
        raise SystemExit("mem.py: a `shard` backend mixes GPU and CPU memory, and "
                         "the GPU worker lives in a child process -- use mem_shard.py "
                         "(this script would report the parent's RSS only).")
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
    print(f"{a.backend:28s} x={len(x0):3d}  {s}  | host RSS {hwm_s} GiB",
          flush=True)
    f.backend.free()


if __name__ == "__main__":
    main()

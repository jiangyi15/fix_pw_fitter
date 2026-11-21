#!/usr/bin/env python3
"""Peak memory of a mixed CPU+GPU `shard`.

    python mem_shard.py --workers cuda_flat_pwa_gen_s0e2,cpu_flat_pwa_gen_cached \
                        --weights 57,43

Two memories, one instrument each:

* **VRAM** — device-wide ``nvidia-smi`` sampled OUT OF PROCESS by
  :mod:`tabpwa.monitor` at 10 ms (MiB granularity), max over the run.  The
  baseline is taken before ``Fitter(...)``, so it is the card's idle level and
  the reported number INCLUDES every worker's CUDA context.  The parent never
  creates one (the `shard` only spawns workers), which is why there is no
  parent-context term to subtract: `nvidia-smi` is device-wide, so any other
  process on the card would be included too — run on an idle device.
* **RAM** — per-process `VmHWM` (kernel high-water mark) for the parent and for
  every spawned worker, via psutil.  The shard's workers are persistent
  processes, so they are still alive when this is read after a `get_nll`.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import psutil                                                       # noqa: E402

from tabpwa.monitor import GPU_FIELDS, create_monitor                # noqa: E402

PHASES = ("context", "load", "nll", "resident")


def vm_hwm_gib(pid):
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmHWM"):
                    return float(line.split()[1]) / 2**20
    except OSError:
        return 0.0
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config_gen.yml"))
    ap.add_argument("--workers", default="cuda_flat_pwa_gen_s0e2,cpu_flat_pwa_gen_cached")
    ap.add_argument("--weights", default="57,43")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()

    mon = create_monitor(what=GPU_FIELDS, interval=0.01, device=a.device)
    mon.wait_ready(5.0)
    first = mon.get_results().peak("gpu_used")
    gpu_ok = first is not None
    baseline = first if first is not None else 0.0

    from tabpwa import Fitter
    spec = {"name": "shard", "start_method": "spawn",
            "backends": [w.strip() for w in a.workers.split(",")],
            "weights": [float(w) for w in a.weights.split(",")]}

    mon.mark("context")
    f = Fitter(a.config, backend=spec)

    mon.mark("load")
    phsp = f.load_dataset("phsp")
    data = f.load_dataset("data")
    f.set_phsp(phsp)
    f.set_data(data)
    f.apply_constrains()
    mw = {n: float(f.defaults[n])
          for n in dict.fromkeys(f.var_registry.flat_names)
          if n.endswith(("_mass", "_width")) and n in f.defaults}
    f.set_fixed(mw)

    mon.mark("nll")
    x0 = f.initial_values(seed=1)
    for _ in range(a.repeat):
        f.get_nll(x0)

    mon.mark("resident")
    time.sleep(0.1)
    res = mon.close()

    def fmt(v):
        # `-` for no sample, as Results.summary does.
        return "    -" if v is None else f"{v:5.2f}"

    if gpu_ok:
        def gib(mib):
            return None if mib is None else (mib - baseline) / 1024.0

        per = {p: res.peak("gpu_used", phase=p) for p in PHASES}
        vals = [v for v in per.values() if v is not None]
        peak = max(vals) if vals else None
        vram = (f"VRAM (device-wide, peak) {fmt(gib(peak))} GiB   "
                f"[context {fmt(gib(per['context']))} | load {fmt(gib(per['load']))} | "
                f"nll {fmt(gib(per['nll']))}]")
    else:
        vram = "VRAM (device-wide, peak)     -  (no parseable nvidia-smi sample)"

    me = psutil.Process()
    kids = me.children(recursive=True)
    print(f"workers {a.workers} | weights {a.weights}")
    print(vram)
    print(f"RAM  parent (VmHWM)      {vm_hwm_gib(me.pid):5.2f} GiB")
    tot = vm_hwm_gib(me.pid)
    for k in kids:
        h = vm_hwm_gib(k.pid)
        tot += h
        print(f"RAM  worker pid {k.pid:<7d}   {h:5.2f} GiB")
    print(f"RAM  total (parent+workers) {tot:5.2f} GiB")
    f.backend.free()


if __name__ == "__main__":
    main()

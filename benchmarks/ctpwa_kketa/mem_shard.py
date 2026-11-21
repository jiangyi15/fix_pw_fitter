#!/usr/bin/env python3
"""Peak memory of a mixed CPU+GPU `shard`.

    python mem_shard.py --workers cuda_flat_pwa_gen_s0e2,cpu_flat_pwa_gen_cached \
                        --weights 57,43

Two memories, two sources:

* **VRAM** — device-wide `cudaMemGetInfo` sampled from a thread (~1 ms), max
  over the run.  The parent's own CUDA primary context (`torch.cuda.init()`, ~0.15 GiB) is in
  the baseline and excluded, so this is what the GPU *worker* holds -- but that
  context is a real device consumer and its exclusion is why the parent process
  shows up in neither column of the VRAM figure.
* **RAM** — per-process `VmHWM` (kernel high-water mark) for the parent and for
  every spawned worker, via psutil.  The shard's workers are persistent
  processes, so they are still alive when this is read after a `get_nll`.
"""
import argparse
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import psutil                                                       # noqa: E402
import torch                                                        # noqa: E402

torch.cuda.init()
PHASE = ["context"]
SAMPLES = []
_STOP = threading.Event()


def used_bytes():
    free, total = torch.cuda.mem_get_info()
    return total - free


def _sampler(interval=0.001):
    while not _STOP.is_set():
        SAMPLES.append((PHASE[0], used_bytes()))
        time.sleep(interval)


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
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()

    baseline = used_bytes()
    th = threading.Thread(target=_sampler, daemon=True)
    th.start()

    from tabpwa import Fitter
    PHASE[0] = "context"
    spec = {"name": "shard", "start_method": "spawn",
            "backends": [w.strip() for w in a.workers.split(",")],
            "weights": [float(w) for w in a.weights.split(",")]}
    f = Fitter(a.config, backend=spec)

    PHASE[0] = "load"
    phsp = f.load_dataset("phsp")
    data = f.load_dataset("data")
    f.set_phsp(phsp)
    f.set_data(data)
    f.apply_constrains()
    mw = {n: float(f.defaults[n])
          for n in dict.fromkeys(f.var_registry.flat_names)
          if n.endswith(("_mass", "_width")) and n in f.defaults}
    f.set_fixed(mw)

    PHASE[0] = "nll"
    x0 = f.initial_values(seed=1)
    for _ in range(a.repeat):
        f.get_nll(x0)

    PHASE[0] = "resident"
    time.sleep(0.1)
    resident = used_bytes()
    _STOP.set()
    th.join(timeout=1.0)

    per = {}
    for ph, u in SAMPLES:
        per[ph] = max(per.get(ph, 0), u)
    peak = max(per.values()) if per else resident
    g = lambda b: (b - baseline) / 2**30                       # noqa: E731

    me = psutil.Process()
    kids = me.children(recursive=True)
    print(f"workers {a.workers} | weights {a.weights}")
    print(f"VRAM (device-wide, peak) {g(peak):5.2f} GiB   "
          f"[context {g(per.get('context', 0)):4.2f} | load {g(per.get('load', 0)):4.2f} | "
          f"nll {g(per.get('nll', 0)):4.2f}]")
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

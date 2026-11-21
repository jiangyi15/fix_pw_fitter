#!/usr/bin/env python3
"""PEAK memory of one backend at the benchmark size.

    python mem.py --backend cuda_v4_pwa_s0e --free-mw a,b,c [...]
    python mem.py --config config_gen.yml --backend cpu_flat_pwa_gen_cached2

Two kinds of peak, because the two kernel families live in different memories:

* GPU (`cudaMemGetInfo` via torch, sampled from a thread at ~1 ms): a backend
  can allocate a staging/cache buffer inside one pass and free it before the
  next sample, so four-point polling misses the peak.  Phases:
  `context` / `load` / `nll` / `resident`, plus `PEAK`.
* Host RSS (`VmHWM`, the kernel's own high-water mark, exact): reported for
  every backend, and it is the number that matters for the CPU kernels, whose
  caches are RAM.

A `cpu_*` backend skips the GPU sampler entirely.
"""
import argparse
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

PHASE = ["context"]
SAMPLES = []
_STOP = threading.Event()
HAS_GPU = False


def rss_peak_gib():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM"):
                    return float(line.split()[1]) / 2**20
    except OSError:
        pass
    return 0.0


def _init_gpu():
    global HAS_GPU
    try:
        import torch
        torch.cuda.init()
        torch.cuda.mem_get_info()
        HAS_GPU = True
        return torch
    except Exception:
        return None


_TORCH = None


def used_bytes():
    free, total = _TORCH.cuda.mem_get_info()
    return total - free


def _sampler(interval=0.001):
    while not _STOP.is_set():
        SAMPLES.append((PHASE[0], used_bytes()))
        time.sleep(interval)


def main():
    global _TORCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yml"))
    ap.add_argument("--backend", required=True)
    ap.add_argument("--base", default=None)
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
    gpu = not a.backend.startswith(cpu_family)
    _TORCH = _init_gpu() if gpu else None
    if _TORCH is not None and not HAS_GPU:
        _TORCH = None
    baseline = used_bytes() if _TORCH is not None else 0

    if _TORCH is not None:
        th = threading.Thread(target=_sampler, daemon=True)
        th.start()

    from tabpwa import Fitter
    spec = {"name": a.backend}
    if a.base:
        spec["base"] = a.base

    PHASE[0] = "context"
    f = Fitter(a.config, backend=spec)

    PHASE[0] = "load"
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

    PHASE[0] = "nll"
    for _ in range(a.repeat):
        f.get_nll(x0)

    PHASE[0] = "resident"
    time.sleep(0.05)
    resident = used_bytes() if _TORCH is not None else 0
    _STOP.set()
    if _TORCH is not None:
        th.join(timeout=1.0)

    per = {}
    for ph, u in SAMPLES:
        per[ph] = max(per.get(ph, 0), u)
    g = lambda b: (b - baseline) / 2**30                       # noqa: E731
    if _TORCH is not None:
        peak = max(per.values()) if per else resident
        s = (f"context {g(per.get('context', 0)):5.2f} | load {g(per.get('load', 0)):5.2f}"
             f" | nll {g(per.get('nll', 0)):5.2f} | resident {g(resident):5.2f} | "
             f"PEAK {g(peak):5.2f} GiB")
    else:
        s = "cpu backend (no GPU sampling)"
    print(f"{a.backend:28s} x={len(x0):3d}  {s}  | host RSS {rss_peak_gib():5.2f} GiB",
          flush=True)
    f.backend.free()


if __name__ == "__main__":
    main()

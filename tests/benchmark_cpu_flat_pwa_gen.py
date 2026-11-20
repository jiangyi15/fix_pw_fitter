"""Benchmark ``cpu_flat_pwa_gen`` (and a GPU sibling) on a real config.

ONE backend per process (so the GPU and CPU numbers are not measured while
competing with each other), end-to-end through ``Fitter``: the config-driven
loader, ``set_phsp``/``set_data``, then the data pass alone and the full
``get_nll``.  ``--stage`` additionally reports the kernel's own ``rdtsc``
split (``w``, grid+bins, ka, fa, fl, num, den, fwd+bwd, accumulate), scaled
by the measured wall time — that is the number that says WHICH part a CPU
implementation pays for and the GPU does not (there it is a load-time cache).

Usage::

    python tests/benchmark_cpu_flat_pwa_gen.py CONFIG.yml cpu_flat_pwa_gen --stage
    OMP_NUM_THREADS=32 python tests/benchmark_cpu_flat_pwa_gen.py CONFIG.yml \\
        cpu_flat_pwa_gen --rsize 17
    python tests/benchmark_cpu_flat_pwa_gen.py CONFIG.yml cuda_flat_pwa_gen_s0e

Notes
-----
* ``ms``/``Mev/s`` are min-of-N; the data pass is reported against the DATA
  event count, ``get_nll`` against both the data-only and the all-events
  (data + phsp) counts, because ``get_nll`` integrates the phase space too.
* the CPU kernel ignores ``batch_size`` by construction, so there is no
  batching knob to sweep here.
"""
import argparse
import time

import numpy as np

from tabpwa import Fitter


def timeit(fn, n=3):
    best = None
    out = None
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t
        best = dt if best is None else min(best, dt)
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("backend")
    ap.add_argument("--rsize", type=int, default=17)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--stage", action="store_true",
                    help="also report the kernel's per-stage rdtsc split")
    args = ap.parse_args()

    f = Fitter(args.config,
               backend={"name": args.backend, "resolution_size": args.rsize})
    f.apply_constrains()
    data = f.load_dataset("data")
    phsp = f.load_dataset("phsp")
    print(f"[{args.backend}] data {data['mass'].shape[0]} events, "
          f"phsp {phsp['mass'].shape[0]}", flush=True)
    t0 = time.perf_counter()
    f.set_phsp(phsp)
    f.set_data(data)
    print(f"[{args.backend}] load+set {time.perf_counter() - t0:.2f} s",
          flush=True)

    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    kernel = getattr(f.backend, "kernel", f.backend)
    dh = f._data_holder
    ne_data = dh.ne
    ne_all = dh.ne + f._phsp_holder.ne
    norm = f._compute_norm_batched(params)[0]

    if args.stage and hasattr(kernel, "_lib"):
        stages = {}
        t = time.perf_counter()
        kernel.compute(params, dh, norm=norm, return_p=True,
                       stage_ns=stages)
        dt = time.perf_counter() - t
        cyc = float(sum(stages.values())) or 1.0
        print(f"[{args.backend}] stage split (data pass {dt * 1e3:.1f} ms):")
        for name, c in stages.items():
            if c:
                print(f"    {name:16s} {dt * 1e3 * c / cyc:8.2f} ms "
                      f"{100.0 * c / cyc:5.1f} %")
        ms = dt
    else:
        ms, _ = timeit(lambda: kernel.compute(params, dh, norm=norm,
                                              return_p=True),
                       n=args.repeat)
    print(f"[{args.backend}] data pass    {ms * 1e3:8.1f} ms "
          f"{ne_data / ms / 1e6:7.2f} Mev/s", flush=True)

    ms2, nll = timeit(lambda: f.get_nll(x), n=max(2, args.repeat - 1))
    print(f"[{args.backend}] full get_nll {ms2 * 1e3:7.1f} ms "
          f"data-only {ne_data / ms2 / 1e6:7.2f} Mev/s  "
          f"all-events {ne_all / ms2 / 1e6:7.2f} Mev/s  "
          f"NLL={float(nll[0]):.6f}", flush=True)


if __name__ == "__main__":
    main()

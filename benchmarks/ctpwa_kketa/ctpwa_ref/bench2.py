#!/usr/bin/env python3
"""Time CTPWA's own init + NLL + gradient (their scripts/bench_free_params.py,
with the analysis construction timed separately).

    python bench2.py config.yml --n 3

Prints: model dimension (partial waves, nvec, ntheta), init seconds (their
amplitude precomputation), and NLL / NLL+grad ms per call.  The paper's
"NLL + gradient per iteration" is `forward+grad`.
"""
import argparse
import time

import torch

import ctpwa


def _sync(dev):
    """their calls only ENQUEUE work; without this the timer reads the host."""
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def timeit(fn, n, dev, warmup=2):
    for _ in range(warmup):
        fn()
    _sync(dev)
    ts = []
    for _ in range(n):
        _sync(dev)
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--fit-mode", type=int, default=0)
    a = ap.parse_args()

    t0 = time.perf_counter()
    ana = ctpwa.analysis(a.config, a.fit_mode)
    if torch.cuda.is_available() and ctpwa.backend() == "cuda":
        torch.cuda.synchronize()          # their precompute is async too
    t_init = time.perf_counter() - t0

    dev = torch.device("cuda:0" if ctpwa.backend() == "cuda" else "cpu")
    nv, nt = ana.getNVector(), ana.getNFreeTheta()
    p = torch.zeros(2 * nv + nt, dtype=torch.float64, device=dev)
    p[0] = 1.0                      # their gauge: first coupling fixed
    if nt:
        p[2 * nv:] = ana.getFreeResParams()[0].to(device=dev)

    def fwd():
        with torch.no_grad():
            ana.getNLL(p)

    def full():
        q = p.clone().requires_grad_(True)
        torch.autograd.grad(ana.getNLL(q), q)[0]

    tf = timeit(fwd, a.n, dev)
    tg = timeit(full, a.n, dev)
    print(f"RESULT config={a.config} backend={ctpwa.backend()} nvec={nv} ntheta={nt} "
          f"init={t_init:.1f}s forward={tf:.2f}ms forward+grad={tg:.2f}ms", flush=True)


if __name__ == "__main__":
    main()

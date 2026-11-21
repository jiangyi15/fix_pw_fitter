"""Time their getNLL(+grad) with theta held vs perturbed, on the CUDA backend.

Usage: python ctpwa_vary.py <config.yml> [n_rep]
Prints the min over repeats for both modes, so a value-keyed fast path shows up.
"""
import sys
import time

import numpy as np
import torch

import ctpwa

cfg = sys.argv[1]
n_rep = int(sys.argv[2]) if len(sys.argv) > 2 else 4

ana = ctpwa.analysis(cfg)
nv, nt = ana.getNVector(), ana.getNFreeTheta()
dev = torch.device("cuda")
base = torch.zeros(2 * nv + nt, dtype=torch.float64, device=dev)
base[0] = 1.0
if nt:
    base[2 * nv:] = ana.getFreeResParams()[0].to(device=dev).double()
rng = np.random.default_rng(5)


def tgrad(p):
    q = p.clone().requires_grad_(True)
    torch.cuda.synchronize()
    a = time.perf_counter()
    torch.autograd.grad(ana.getNLL(q), q)[0]
    torch.cuda.synchronize()
    return (time.perf_counter() - a) * 1e3


def run(vary):
    ts = []
    for _ in range(n_rep):
        p = base.clone()
        if vary:
            p[:nv] = torch.from_numpy(rng.normal(size=nv)).to(dev)
            p[0] = 1.0            # the gauge slot stays pinned (as in bench2.py)
            if nt:
                th = p[2 * nv:]
                p[2 * nv:] = th + 1e-3 * (1 + th.abs()) * torch.from_numpy(
                    rng.normal(size=nt)).to(dev)
        tgrad(p)
        ts.append(tgrad(p))
    return min(ts)


print(f"ntheta {nt}", flush=True)
print(f"  same theta : {run(False):7.2f} ms", flush=True)
print(f"  vary theta : {run(True):7.2f} ms", flush=True)

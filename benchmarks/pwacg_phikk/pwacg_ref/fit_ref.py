#!/usr/bin/env python3
"""Run the reference's OWN fitting pipeline (`Control.run`) and time it.

    python pwacg_ref/fit_ref.py --pwacg /path/to/PWACG \
        --npz ../data/Momentum_kk.npz --mc ../data/Momentum_kk_phsp.npz \
        --maxiter 5

`bench_ref.py` times one likelihood+gradient evaluation.  This script times the
thing the reference actually runs: `Control.run`, i.e.
`scipy.optimize.minimize(method="Newton-CG", jac=thread_grad_likelihood,
hessp=thread_hvp)` over their multi-thread wrappers.  That step is NOT one
evaluation: Newton-CG calls the value, the gradient and Hessian-vector products
several times per iteration, so its cost per *iteration* is larger than the
per-evaluation numbers in `README.md` — this is the end-to-end figure.

Shims, all outside the vendored sources, each listed where it is installed:

* `config = jax.config` — the generated module calls `config.update(...)` in
  `Control.run` without importing `config` (a codegen artefact);
* a `pynvml` no-op — their post-fit memory log would otherwise hit the strict
  stub and abort the run.  It prints `GPU memory used: 0`: that number is a
  placeholder, NOT a measurement (use `tabpwa.monitor` for the real figure);
* a `minimize` wrapper — injects `maxiter` (they pass none, so the run is
  unbounded) and records `nit`/`nfev`/`njev`/`nhev`/wall; scipy is untouched;
* `save_in_json = no-op` — their post-fit writer reads a template the pipeline
  is supposed to have produced, and is not part of the fit cost;
* the GPU/threading attributes on `mod.args()` (`max_processes`, `thread_gpus`,
  `total_gpu_id`, …) — this harness runs one process, one thread, one chunk, so
  the numbers are single-GPU and do not include their multi-worker path.

`Control.run` also writes into `output/fit/fit_result_kk/` and calls `pynvml`;
the first must exist, the second is stubbed (`stubs/pynvml.py`).
"""
import argparse
import importlib.util
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "stubs"))

import jax_legacy_config  # noqa: E402,F401  installs the legacy jax.config alias


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pwacg", required=True)
    ap.add_argument("--npz", default=os.path.join(HERE, "..", "data", "Momentum_kk.npz"))
    ap.add_argument("--mc", default=os.path.join(HERE, "..", "data", "Momentum_kk_phsp.npz"))
    ap.add_argument("--maxiter", type=int, default=5)
    ap.add_argument("--max-events", type=int, default=10000,
                    help="cap both samples (0 = all)")
    ap.add_argument("--slices", type=int, default=1,
                    help="their CacheTensor chunking; >1 avoids the 12 GB OOM "
                         "above ~10^5 events")
    ap.add_argument("--method", default="newton-cg",
                    choices=("newton-cg", "bfgs", "fd"),
                    help="newton-cg = their shipped path (jac + hessp/HVP); "
                         "bfgs = the GRADIENT-only path (jac, no Hessian) -- what "
                         "the paper's iMinuit+AutoDiff drives; fd = no jac at all "
                         "(their iMinuit+FDM shape).  iMinuit itself is not "
                         "installable on this box, but it calls the same "
                         "thread_likelihood/thread_grad_likelihood pair, so the "
                         "bfgs row IS that path's cost.")
    ap.add_argument("--reuse", action="store_true",
                    help="trust the clone's existing per-event tensors")
    return ap.parse_args()


def main():
    a = parse()
    root = os.path.abspath(a.pwacg)
    os.chdir(root)
    sys.path.insert(0, root)

    import jax
    import numpy as onp

    jax.config.update("jax_enable_x64", True)

    from Tensor import CacheTensor as calc

    def prep(src, sub, n):
        os.makedirs(os.path.join(root, "data", sub), exist_ok=True)
        dst = os.path.join(root, "data", sub, "Momentum_kk.npz")
        shutil.copyfile(src, dst)
        if a.reuse:
            return n
        cdl = calc.data_info()
        cdl.address = f"data/{sub}/"
        cdl.filename = "Momentum_kk.npz"
        cdl.id = "kk"
        cdl.size = n
        cdl.begain = 0
        cdl.end = n
        cdl.slices = max(1, a.slices)
        mycdl = calc.CompoundDataForTensor(cdl)
        mycdl.g_id = 0
        mycdl.gpu_id = 0
        fn = calc.pwa_func()
        fn.fill_func()
        t0 = time.time()
        calc.Calculate(fn, mycdl)
        print(f"tensors[{sub}] {n} events: {time.time() - t0:.2f} s", flush=True)
        return n

    with onp.load(a.npz) as d:
        n_data = len(d["Kp"])
    with onp.load(a.mc) as d:
        n_mc = len(d["Kp"])
    if a.max_events:
        n_data = min(n_data, a.max_events)
        n_mc = min(n_mc, a.max_events)
    prep(a.npz, "real_data", n_data)
    prep(a.mc, "mc_truth", n_mc)

    # ── the generated module ───────────────────────────────────────────────
    path = os.path.join(root, "rendered_scripts", "fit_object_kk.py")
    spec = importlib.util.spec_from_file_location("fit_object_kk_fit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fit_object_kk_fit"] = mod
    spec.loader.exec_module(mod)
    mod.config = jax.config          # shim 1: Control.run's unimported `config`

    # shim 1b: `Control.run` logs GPU memory through pynvml AFTER the fit.  The
    # strict stub in stubs/ raises (it is dead code for bench_ref); here the call
    # is on the path, so give it a no-op that reports 0 and say so: the real
    # device memory belongs to tabpwa.monitor, not to this number.
    import types
    mod.pynvml = types.SimpleNamespace(
        nvmlInit=lambda *a, **k: None,
        nvmlDeviceGetHandleByIndex=lambda *a, **k: None,
        nvmlDeviceGetMemoryInfo=lambda *a, **k: types.SimpleNamespace(used=0))

    # ── shim 2: bound the optimiser and record what it did ─────────────────
    real_minimize = mod.minimize
    record = {}

    def minimize(*args, **kw):
        opts = dict(kw.get("options", {}))
        opts["maxiter"] = a.maxiter
        kw["options"] = opts
        if a.method == "bfgs":
            # the gradient path: keep jac, drop the Hessian-vector product.
            kw.pop("hessp", None)
            kw["method"] = "BFGS"
        elif a.method == "fd":
            # no gradient at all (finite differences) — the iMinuit+FDM shape.
            kw.pop("hessp", None)
            kw.pop("jac", None)
            kw["method"] = "BFGS"
        else:
            kw["method"] = "Newton-CG"
        t0 = time.perf_counter()
        res = real_minimize(*args, **kw)
        record["wall"] = time.perf_counter() - t0
        record["nit"] = int(getattr(res, "nit", -1))
        record["nfev"] = int(getattr(res, "nfev", -1))
        record["njev"] = int(getattr(res, "njev", -1))
        record["nhev"] = int(getattr(res, "nhev", -1))
        record["fun"] = float(getattr(res, "fun", float("nan")))
        return res

    mod.minimize = minimize

    # ── the two objects Control.run expects ────────────────────────────────
    def load(name, sub):
        return onp.load(os.path.join(root, "data", sub, f"{name}_kk.npy"))

    class CDL:
        def __init__(self):
            self.data_phi_kk = load("phi", "real_data")
            self.data_f_kk = load("f", "real_data")
            self.data_phif0_kk = load("phif0", "real_data")
            self.data_phif2_kk = load("phif2", "real_data")
            self.mc_phi_kk = load("phi", "mc_truth")
            self.mc_f_kk = load("f", "mc_truth")
            self.mc_phif0_kk = load("phif0", "mc_truth")
            self.mc_phif2_kk = load("phif2", "mc_truth")
            self.wt_data_kk = onp.ones_like(self.data_phi_kk)

    class Returns:
        def __init__(self):
            self.d = {}

        def set(self, k, v):
            self.d[k] = v

        def info(self):
            pass

    for d in ("output/fit/fit_result_kk", "output/fit/fit_result_combine",
              "output/fit/fit_result_pipi"):
        os.makedirs(os.path.join(root, d), exist_ok=True)


    args_o = mod.args()
    args_o.max_processes = 1
    args_o.max_processes_memory = 0.9
    args_o.processes_gpus = 1
    args_o.thread_gpus = 1
    args_o.threads_in_one_gpu = 1
    args_o.total_gpu_id = [0]
    args_o.data_slices = 1
    args_o.mc_slices = 1
    args_o.mini_run = 1
    ctrl = mod.Control(args_o)

    # `Control.run` expects each initializer to be the data object WITH the
    # three attributes `run_multiprocess` attaches (their object does both jobs).
    cdl = CDL()
    cdl.gpu_id = 0
    cdl.total_gpu_id = [0]
    cdl.gpu_memory_limit_percentage = 0.9
    init = [[cdl]]            # [thread_gpus][threads_in_one_gpu]

    # `Control.run` finishes by writing its result through `save_in_json`, which
    # reads a template the pipeline is supposed to have produced.  This harness
    # times the FIT, so the writer is a no-op here — stated rather than hidden.
    ctrl.save_in_json = lambda *args, **kw: None
    ret = Returns()
    t0 = time.perf_counter()
    ctrl.run(init, ret)
    total = time.perf_counter() - t0

    nit = record.get("nit", -1)
    print(f"\ntheir Control.run [{a.method}]: maxiter={a.maxiter}  nit={nit}  "
          f"nfev={record.get('nfev')}  njev={record.get('njev')}  "
          f"nhev={record.get('nhev')}")
    nfev, njev, nhev = record.get("nfev", -1), record.get("njev", -1), record.get("nhev", -1)
    print(f"  optimiser wall = {record['wall']*1e3:.1f} ms   "
          f"whole run (jit + setup + fit) = {total*1e3:.1f} ms")
    print(f"  evaluations: {nfev} value + {njev} gradient + {nhev} HVP, "
          f"i.e. {nfev/max(nit,1):.2f} + {njev/max(nit,1):.2f} + {nhev/max(nit,1):.2f} "
          f"per Newton-CG iteration")
    if nit > 0:
        print(f"  per iteration = {record['wall']/nit*1e3:.1f} ms "
              f"({(n_data + n_mc)*nit/record['wall']/1e6:.2f} Mev/s over data+MC)")
    evals = max(nfev, 1) + max(njev, 1) + max(nhev, 0)
    print(f"  so one cost unit = {record['wall']/evals*1e3:.1f} ms of optimiser time "
          f"({n_data + n_mc:,} events per sample, {evals} evaluations total)")
    print(f"  final NLL = {record.get('fun')}")


if __name__ == "__main__":
    main()

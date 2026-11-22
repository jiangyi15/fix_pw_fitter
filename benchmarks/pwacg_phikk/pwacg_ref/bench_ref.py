#!/usr/bin/env python3
"""Time the PWACG-generated fit object on the samples this benchmark uses.

    python pwacg_ref/bench_ref.py --pwacg /path/to/PWACG \
        --npz ../data/Momentum_kk.npz --mc ../data/Momentum_kk_phsp.npz

What is timed, all `block_until_ready`, min-of-N, steady state (a first call
includes XLA compilation, 0.3–1.1 s per signature):
  * `PWAFunc` construction                       (the resident per-event tensors)
  * `data_likelihood_kk` eager                   (no JIT — 45 ms at 10 k, why
                                                  the JIT is the whole design)
  * `jit_data_likelihood_kk`
  * `jit_grad_data_likelihood_kk` / `value_and_grad`  (one fit iteration)
  * the MC integral term, and its `value_and_grad`, at the same event counts

The samples are the four-momenta this benchmark generates (`make_data.py`) or
the reference's own release data (`make_data_from_ref.py`) — either way the
per-event tensors are computed here by THEIR `Tensor/CacheTensor.py`, so the
timing is their code on our event list, not a reimplementation.

`--reuse` skips the tensor step when the `.npy` files for the requested event
count already exist in the clone.  Nothing in the clone is edited: the legacy
`jax.config` import is served by `jax_legacy_config.py`, and `iminuit`/`pynvml`
by `stubs/` (both dead on the timed path).
"""
import argparse
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "stubs"))

import jax_legacy_config  # noqa: E402,F401  installs the legacy jax.config alias


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pwacg", required=True, help="the PWACG clone")
    ap.add_argument("--npz", default=os.path.join(HERE, "..", "data",
                                                 "Momentum_kk.npz"))
    ap.add_argument("--mc", default=os.path.join(HERE, "..", "data",
                                                "Momentum_kk_phsp.npz"))
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--slices", type=int, default=1,
                    help="their CacheTensor `slices`: chunk the tensor "
                         "precompute. 1 allocates the whole sample's tensors at "
                         "once and OOMs above ~10^5 events on a 12 GB card "
                         "(measured: 12.22 GiB request at 100 k)")
    ap.add_argument("--reuse", action="store_true",
                    help="trust the clone's existing per-event tensors")
    ap.add_argument("--mc-sizes", default="10000,100000,1000000",
                    help="event counts for the MC term (the 10 k set is tiled)")
    ap.add_argument("--data-sizes", default="",
                    help="optional extra event counts for the DATA term (tiled); "
                         "the first data timing is always the untitled sample")
    return ap.parse_args()


def main():
    a = parse()
    root = os.path.abspath(a.pwacg)
    os.chdir(root)
    sys.path.insert(0, root)

    import jax
    import numpy as onp

    jax.config.update("jax_enable_x64", True)

    for sub in ("real_data", "mc_truth"):
        os.makedirs(os.path.join(root, "data", sub), exist_ok=True)

    # ── the per-event tensors, computed by their own CacheTensor path ──────
    import shutil

    for src, sub in ((a.npz, "real_data"), (a.mc, "mc_truth")):
        dst = os.path.join(root, "data", sub, "Momentum_kk.npz")
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
    n_data = len(onp.load(os.path.join(root, "data/real_data/Momentum_kk.npz"))
                 ["Kp"])
    n_mc = len(onp.load(os.path.join(root, "data/mc_truth/Momentum_kk.npz"))
               ["Kp"])

    if not a.reuse:
        from Tensor import CacheTensor as calc
        for sub, n in (("real_data", n_data), ("mc_truth", n_mc)):
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
            print(f"tensors[{sub}] {n} events: {time.time() - t0:.2f} s",
                  flush=True)

    # ── the generated fit object ────────────────────────────────────────────
    # ORDER MATTERS: `create_all_scripts.py` reads data/real_data/{phi,f}_kk.npy
    # while it prepares, so the tensors above must exist first.  Offline it
    # returns the `.llm_cache/golden` artefact verbatim — there is no non-LLM
    # renderer in the current tree, so only the shipped kk model is reachable.
    path = os.path.join(root, "rendered_scripts", "fit_object_kk.py")
    if not os.path.exists(path):
        import subprocess
        print("no rendered_scripts/ — running their create_all_scripts.py",
              flush=True)
        subprocess.run([sys.executable, "create_all_scripts.py"], check=True)
    spec = importlib.util.spec_from_file_location("fit_object_kk_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fit_object_kk_ref"] = mod
    spec.loader.exec_module(mod)

    def load(name, sub):
        return onp.load(os.path.join(root, "data", sub, f"{name}_kk.npy"))

    class CDL:
        """The nine arrays `PWAFunc` reads (this is `ProcessInitializers`).

        ``ds``/``scale`` tile the data / MC sides along the event axis.  Tiling
        repeats events, which changes the *value* of a log-likelihood sum but not
        the per-event work, so it is how the reference's own harness measures
        their scaling (their `Tensor/RunCacheTensor.py` uses real 1.4 M MC sets;
        we do not have those here).
        """

        def __init__(self, scale=1, ds=1):
            self.data_phi_kk = onp.tile(load("phi", "real_data"), ds)
            self.data_f_kk = onp.tile(load("f", "real_data"), ds)
            self.data_phif0_kk = onp.tile(load("phif0", "real_data"), (1, ds, 1))
            self.data_phif2_kk = onp.tile(load("phif2", "real_data"), (1, ds, 1))
            self.mc_phi_kk = onp.tile(load("phi", "mc_truth"), scale)
            self.mc_f_kk = onp.tile(load("f", "mc_truth"), scale)
            self.mc_phif0_kk = onp.tile(load("phif0", "mc_truth"), (1, scale, 1))
            self.mc_phif2_kk = onp.tile(load("phif2", "mc_truth"), (1, scale, 1))
            self.wt_data_kk = onp.ones_like(self.data_phi_kk)

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
    args = onp.asarray(ctrl.args_list)[ctrl.float_list]

    def timed(fn):
        for _ in range(a.warm):
            jax.block_until_ready(fn())
        ts = []
        for _ in range(a.repeat):
            t = time.perf_counter()
            out = fn()
            jax.block_until_ready(out)
            ts.append(time.perf_counter() - t)
        return min(ts), out

    def first(fn):
        t = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        return time.perf_counter() - t, out

    rows = []
    cdl = CDL(1)
    t0 = time.perf_counter()
    pwa = mod.PWAFunc(cdl=cdl, device_id=None)
    jax.block_until_ready(pwa.data_phi_kk)
    print(f"PWAFunc init ({n_data} data + {n_mc} mc): "
          f"{time.perf_counter() - t0:.3f} s", flush=True)
    used = jax.devices()[0].memory_stats() or {}
    print(f"resident after init: {used.get('bytes_in_use', 0)/1e6:.1f} MB "
          f"(peak {used.get('peak_bytes_in_use', 0)/1e6:.1f} MB)", flush=True)

    f1, v = first(lambda: pwa.data_likelihood_kk(args))
    s, v = timed(lambda: pwa.data_likelihood_kk(args))
    rows.append(("data_likelihood_kk eager", n_data, f1, s, float(v)))

    pwa.jit_request()
    f1, v = first(lambda: pwa.jit_data_likelihood_kk(args))
    s_data_v, v = timed(lambda: pwa.jit_data_likelihood_kk(args))
    rows.append(("jit_data_likelihood_kk", n_data, f1, s_data_v, float(v)))

    f1, g = first(lambda: pwa.jit_grad_data_likelihood_kk(args))
    s_data_g, g = timed(lambda: pwa.jit_grad_data_likelihood_kk(args))
    assert onp.shape(g) == args.shape, (
        f"data gradient {onp.shape(g)} does not span the parameter vector "
        f"{args.shape}")
    rows.append(("jit_grad_data_likelihood_kk", n_data, f1, s_data_g,
                 float(onp.linalg.norm(g))))

    # "value_and_grad" here is ONE call PER SAMPLE, and the two are SUMMED by the
    # caller — NOT a single call over a combined objective.  Their code has no
    # such combined call either: `data_likelihood_kk` and `mc_likelihood_kk` are
    # separate jitted functions, and the sum  data + 100·log(Σmc/1000)  is
    # assembled in NumPy by `thread_likelihood` / `thread_grad_likelihood`
    # (whose gradient is  Σdata_grad + 100·Σmc_grad/Σmc — the chain rule with a
    # constant factor, no cross term, since the data term does not depend on Σmc).
    vg = jax.jit(jax.value_and_grad(pwa.data_likelihood_kk))
    f1, (v, _g) = first(lambda: vg(args))
    s, (v, _g) = timed(lambda: vg(args))
    # WHAT IS TIMED: one scalar likelihood and its gradient over the SAME
    # parameter vector.  Not the value alone, and NOT the Hessian-vector
    # product -- the generated module exposes `hvp_*`, and this harness never
    # calls it (grep: no `hvp` below).
    assert onp.ndim(v) == 0, f"expected a scalar likelihood, got {onp.shape(v)}"
    assert onp.shape(_g) == args.shape, (
        f"gradient {onp.shape(_g)} does not span the parameter vector {args.shape}")
    print(f"timed quantity: value_and_grad -> scalar + gradient{onp.shape(_g)}")
    rows.append(("jit value_and_grad(data)", n_data, f1, s, float(v)))

    # THE one-call path for THEIR objective: differentiate the combined scalar
    #     L(x) = Σ_data lh + 100·log( Σ_mc lh / 1000 )
    # in a single jit.  Summing two per-sample gradients is NOT this: the 100·log
    # chain factor (100/Σmc) is missing, so that sum is not ∇L.  The assertion
    # below compares against their own assembly (Σdata_grad + 100·Σmc_grad/Σmc)
    # so "same gradient" is checked, not assumed.
    import jax.numpy as jnp

    def combined(args_):
        return (pwa.data_likelihood_kk(args_)
                + 100.0 * jnp.log(pwa.mc_likelihood_kk(args_) / 1000.0))

    vgc = jax.jit(jax.value_and_grad(combined))
    f1, (v_c, g_c) = first(lambda: vgc(args))
    s_c, (v_c, g_c) = timed(lambda: vgc(args))
    rows.append(("jit value_and_grad(combined L)", n_data, f1, s_c, float(v_c)))
    # it must BE their gradient: compare with their own NumPy assembly
    g_d = jax.jit(jax.grad(pwa.data_likelihood_kk))(args)
    _vm = jax.jit(jax.value_and_grad(pwa.mc_likelihood_kk))(args)
    v_m, g_m = _vm[0], _vm[1]          # (Σmc, ∇Σmc); indexed, not unpacked
    assembled = g_d + 100.0 * g_m / v_m
    assert onp.allclose(onp.asarray(g_c), onp.asarray(assembled),
                        rtol=1e-10, atol=1e-8), (
        "the combined-objective gradient does not equal their assembly "
        f"max|diff| {onp.abs(onp.asarray(g_c) - onp.asarray(assembled)).max():.3e}")
    del pwa

    # ── the data term at larger (tiled) event counts ────────────────────────
    for n in [int(x) for x in a.data_sizes.split(",") if x.strip()]:
        ds = max(1, n // n_data)
        cdl = CDL(1, ds=ds)
        pwa = mod.PWAFunc(cdl=cdl, device_id=None)
        pwa.jit_request()
        jax.block_until_ready(pwa.data_phi_kk)
        nev = int(len(cdl.data_phi_kk))
        vg_d = jax.jit(jax.value_and_grad(pwa.data_likelihood_kk))
        f1, (v, _g) = first(lambda: vg_d(args))
        s, (v, _g) = timed(lambda: vg_d(args))
        rows.append(("jit value_and_grad(data)", nev, f1, s, float(v)))
        del pwa, cdl

    # ── the MC integral term at the requested event counts ──────────────────
    sizes = [int(x) for x in a.mc_sizes.split(",") if x.strip()]
    for n in sizes:
        scale = max(1, n // n_mc)
        cdl = CDL(scale)
        pwa = mod.PWAFunc(cdl=cdl, device_id=None)
        pwa.jit_request()
        jax.block_until_ready(pwa.mc_phi_kk)
        nev = int(len(cdl.mc_phi_kk))
        f1, v = first(lambda: pwa.jit_mc_likelihood_kk(args))
        s_mc_v, v = timed(lambda: pwa.jit_mc_likelihood_kk(args))
        rows.append(("jit_mc_likelihood_kk", nev, f1, s_mc_v, float(v)))
        vgm = jax.jit(jax.value_and_grad(pwa.mc_likelihood_kk))
        f1, (v, _g) = first(lambda: vgm(args))
        s, (v, _g) = timed(lambda: vgm(args))
        rows.append(("jit value_and_grad(mc)", nev, f1, s, float(v)))
        f1, (v, _g) = first(lambda: pwa.jit_grad_mc_likelihood_kk(args))
        s_mc_g, (v, _g) = timed(lambda: pwa.jit_grad_mc_likelihood_kk(args))
        assert onp.shape(_g) == args.shape, (
            f"mc gradient {onp.shape(_g)} does not span {args.shape}")
        rows.append(("jit_grad_mc_likelihood_kk", nev, f1, s_mc_g, float(v)))
        # their optimiser (`thread_likelihood` + `thread_grad_likelihood`) calls
        # the VALUE and the GRADIENT as separate jitted calls per sample, so the
        # amplitudes are walked twice per iteration.  The fused
        # value_and_grad above is the lower bound, not what their code does.
        # the label must not claim "N / N" when the data side was left at its
        # own size: with the default `--mc-sizes` tiling, `nev` is the MC count
        assert n_data == nev, (
            f"this row sums the data calls ({n_data} events) and the MC calls "
            f"({nev}); pass --data-sizes {nev} to make it one N / N iteration")
        rows.append(("= THEIR iteration (value+grad, separate calls)", nev, 0.0,
                     s_data_v + s_data_g + s_mc_v + s_mc_g, 0.0))
        del pwa, cdl

    print(f"\n{'quantity':32s} {'events':>10s} {'first ms':>10s} "
          f"{'steady ms':>10s} {'ns/event':>10s} {'value':>14s}")
    for name, nev, f1, s, v in rows:
        print(f"{name:32s} {nev:10d} {f1*1e3:10.3f} {s*1e3:10.3f} "
              f"{s/nev*1e9:10.2f} {v:14.6f}")
    used = jax.devices()[0].memory_stats() or {}
    print(f"\npeak device memory (jax): "
          f"{used.get('peak_bytes_in_use', 0)/1e6:.1f} MB")


if __name__ == "__main__":
    main()

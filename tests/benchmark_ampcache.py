#!/usr/bin/env python3
"""Time ``cuda_flat_pwa_gen_ampcache`` against its siblings, and its GPU Gram.

Four backends on the SAME sample and the SAME config, interleaved and min-of-N
(the box throttles, so sequential numbers are meaningless):

  * ``cuda_flat_pwa_gen_s0e2`` — the parent: no cache, every wave's `D` walked
    every iteration;
  * ``cuda_flat_pwa_gen_s0f`` — the s0f cache with EVERY column declared fixed,
    driven through the ``fixed_coeff`` wrapper (declaration, then the base);
  * ``cuda_flat_pwa_gen_ampcache`` — the same cache with the mask specialised
    away (no permutation, no F/T/B, no scratch, no mask at all);
  * ``integrated_pwa_gen`` over that same base — the SAME data kernel behind
    the wave-Gram norm, whose ``D`` is reduced from the base's own ``Afix``
    cache on the device (built once, then O(N²) per iteration).

All are timed the same way: ms/iteration of ``Fitter.get_nll``
(norm + data + gradients) over data + phase space.  ``bytes/event`` is the
PER-HANDLE cache each one carries (the data and the phase-space handle each
carry their own):

    s0e2          0                      (no cache)
    s0f all-fixed 8·n_proj·n_fix  = 8·n_proj·n_wave
    ampcache      8·n_proj·n_wave        (the same bytes — see below)

``load`` NEVER fills: the ONE lifecycle fills a handle on its FIRST compute, so
``first`` is the call that carries the ``Afix`` fill (and, for ``integrated/*``,
the Gram build), while ``steady`` carries neither.  This family has already
been misread once on exactly that point (s0f's ``load+fill`` vs ``first``), so
the two are printed separately.

The Gram section at the end times the numpy oracle against the streamed GPU
build, checks they agree, reports the device's peak for the build (ONE
``mc_batch`` batch, not the sample) and recomputes the break-even against the
per-iteration norm each build replaces.

The default is the CTPWA 34-wave shape with every fitted mass/width pinned
(``bench.py --fix-mw``'s convention), which is the configuration where the
whole cache is legitimate: nothing left free moves a ``coeff`` column, which
the harness CHECKS by perturbation before it times anything.

    python tests/benchmark_ampcache.py                       # CTPWA, 0.6 M
    python tests/benchmark_ampcache.py --n-data 1000000 --n-phsp 5000000

HONESTY NOTE 1: ampcache and s0f-all-fixed carry the SAME bytes (544 B/event at
34 waves x 2 projections), and on the fused path s0f already computes exactly
this arithmetic — so no speedup over s0f is expected or claimed.  What
ampcache buys is the CONTRACT (a pinned-coefficient kernel with no mask
machinery and no free-cache path at all), one fewer kernel shape, and the
absence of s0f's split scratch.  The script prints the numbers it measures and
says which one won.

HONESTY NOTE 2: the GPU Gram streams, so its time includes a load+fill per
``mc_batch`` batch — that is the price of never holding the sample on the
device, and it is what its break-even is computed from.  If it does not beat
the numpy build, the script says so.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from tabpwa import Fitter
from tabpwa.backends import create_backend
from benchmark_flat_vs_pwa import _events
from benchmark_s0f import _moving_columns, _pin_other_mw


def _load_sample(fitter, cfg, n_data, n_phsp, seed):
    """The config's own sample when it exists, else synthetic in-grid events."""
    data_path = fitter.config.dic.get("data", {}).get("data_arr")
    phsp_path = fitter.config.dic.get("data", {}).get("phsp_arr")
    base = os.path.dirname(os.path.abspath(cfg))
    if data_path:
        data_path = os.path.join(base, data_path)
    if phsp_path:
        phsp_path = os.path.join(base, phsp_path)
    if data_path and os.path.exists(data_path):
        data = fitter.load_dataset(
            os.path.basename(data_path).split("_arr")[0], path=data_path,
            max_events=n_data or None)
        phsp = fitter.load_dataset(
            os.path.basename(phsp_path).split("_arr")[0], path=phsp_path,
            max_events=n_phsp or None)
        return data, phsp
    kc = fitter.kernel_config
    return (_events(kc, n_data or 4000, seed),
            _events(kc, n_phsp or 4000, seed + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="benchmarks/ctpwa_kketa/config_gen.yml")
    ap.add_argument("--rsize", type=int, default=1)
    ap.add_argument("--n-data", type=int, default=600000)
    ap.add_argument("--n-phsp", type=int, default=600000)
    ap.add_argument("--repeat", type=int, default=5,
                    help="timed calls per backend per sweep (steady state)")
    ap.add_argument("--sweeps", type=int, default=2,
                    help="whole passes over the backends; the steady row is "
                         "the min over sweeps (this GPU throttles)")
    ap.add_argument("--batch-size", type=int, default=50000)
    ap.add_argument("--mc-batch", type=int, default=5000,
                    help="integrated_pwa_gen's Gram batch: the device holds "
                         "ONE batch of the phase space at a time")
    ap.add_argument("--keep-free-mw", default="",
                    help="comma-separated mass/width NAMES left free; the "
                         "default pins every one, which is what makes the "
                         "all-pinned cache legitimate.  A free mass/width "
                         "moves coeff, so this only works when the remaining "
                         "x does not touch a column (the harness checks).")
    args = ap.parse_args()

    f = Fitter(args.config)
    f.apply_constrains()
    free_mw = tuple(s.strip() for s in args.keep_free_mw.split(",") if s.strip())
    _pin_other_mw(f, free_mw)
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])

    moving = _moving_columns(f, x)
    if moving.size:
        den = [str(v) for v in np.asarray(kc["den_names"]).ravel()]
        names = [den[int(i)] for i in moving[:6]]
        raise SystemExit(
            "benchmark_ampcache.py: the coefficients are not pinned — "
            f"{moving.size}/{n_basis} columns still move ({names}…); pin the "
            "parameters that move them (the default pins every mass/width) or "
            "this comparison is not an all-pinned one")

    data, phsp = _load_sample(f, args.config, args.n_data, args.n_phsp, seed=7)
    coeff0 = np.array(f.build_params(x)[0]["coeff"], np.float64, copy=True)
    all_cols = list(range(n_basis))

    specs = [("s0e2", "cuda_flat_pwa_gen_s0e2", {}),
             # s0f with EVERY column declared fixed, through the mask
             # lifecycle's own home (declaration -> base rebuild)
             ("s0f/all-fixed",
              {"name": "fixed_coeff", "base": "cuda_flat_pwa_gen_s0f"},
              {"__mask__": (all_cols, coeff0)}),
             ("ampcache", "cuda_flat_pwa_gen_ampcache", {}),
             # the SAME data kernel behind the Gram norm: the norm is built
             # ONCE (on the first call), then O(N²) per iteration
             ("integrated/ampcache",
              {"name": "integrated_pwa_gen",
               "base": "cuda_flat_pwa_gen_ampcache"}, {})]

    def build(name, kw):
        kw = dict(kw)
        mask = kw.pop("__mask__", None)
        be = create_backend(name, kc, model="pwa_gen",
                            resolution_size=args.rsize,
                            batch_size=args.batch_size, **kw)
        if mask is not None:
            cols, vals = mask
            be.set_fixed_cols(cols, {int(c): float(vals[int(c)])
                                     for c in cols})
        return be

    def measure(name, kw):
        """Load ONCE per backend, timing the LOAD (which carries the ``Afix``
        fill), then the first COMPUTE (no fill) and the steady state."""
        be = build(name, kw)
        try:
            f.backend = be
            # Fitter.set_data overwrites `_data_holder` without freeing it, so
            # a previous backend's handle would leak ~GB; release it here.
            if f._data_holder is not None:
                f._data_holder.free()
                f._data_holder = None
            t0 = time.perf_counter()
            f.set_phsp(phsp)              # frees the previous phsp handle
            f.set_data(data)              # LOAD: both samples + their cache
            load = time.perf_counter() - t0
            t1 = time.perf_counter()
            f.get_nll(x)                  # first COMPUTE (no fill any more)
            first = time.perf_counter() - t1
            f.get_nll(x)                  # steady-state warm-up (not timed)
            ts = []
            for _ in range(args.repeat):
                t = time.perf_counter()
                f.get_nll(x)
                ts.append(time.perf_counter() - t)
            return load, first, ts
        finally:
            if f._data_holder is not None:
                f._data_holder.free()
                f._data_holder = None
            if f._phsp_holder is not None:
                f._phsp_holder.free()
                f._phsp_holder = None
            f._phsp_scratch = None
            be.free()

    ne_d = len(np.asarray(data["weight"]))
    ne_p = len(np.asarray(phsp["weight"]))
    n_ev = ne_d + ne_p
    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    n_proj = max(1, int(np.asarray(kc["matrix_angle"]).shape[1]) // n_wave)

    print(f"config      {args.config}")
    print(f"shape       n_wave {n_wave}  n_proj {n_proj}  n_basis {n_basis}  "
          f"rsize {args.rsize}")
    print(f"events      data {ne_d}  phsp {ne_p}")
    print(f"pinned      every coefficient (moving columns 0/{n_basis} by x "
          f"perturbation); x has {len(x)} free slot(s)")

    first, load_t = {}, {}
    steady = {label: float("inf") for label, _n, _kw in specs}
    for sweep in range(args.sweeps):
        for label, name, kw in specs:
            ld, fi, ts = measure(name, kw)
            if sweep == 0:
                load_t[label], first[label] = ld, fi
            steady[label] = min(steady[label], min(ts))

    print()
    best = min(steady.values())
    for label, _n, _kw in specs:
        per_ev = {"s0e2": 0,
                  "s0f/all-fixed": 8 * n_proj * n_wave,
                  "ampcache": 8 * n_proj * n_wave,
                  "integrated/ampcache": 8 * n_proj * n_wave}[label]
        fms, sms = first[label] * 1e3, steady[label] * 1e3
        print(f"{label:20s} load {load_t[label] * 1e3:8.1f} ms   "
              f"first {fms:7.1f} ms   steady {sms:8.1f} ms   "
              f"{n_ev / steady[label] / 1e6:7.2f} Mev/s   "
              f"{sms * 1e6 / n_ev:6.2f} ns/event   "
              f"{per_ev:5d} B/event   "
              f"{sms / (best * 1e3):5.2f}x")
    print("  `load` never fills: the ONE lifecycle fills a handle on its "
          "FIRST compute.")
    print("  `first` therefore CARRIES the Afix fill (ampcache) and the Gram "
          "build (integrated/*); `steady` carries neither, and for the "
          "integrated rows it is the data pass + the O(N^2) Gram norm.")

    amp = steady["ampcache"]
    s0f = steady["s0f/all-fixed"]
    par = steady["s0e2"]
    ratio = s0f / amp
    verdict = ("faster" if ratio > 1.02
               else "SLOWER" if ratio < 0.98 else "equal (within 2 %)")
    print()
    print(f"ampcache vs s0f/all-fixed: {ratio:.2f}x -> {verdict}, at the SAME "
          f"{8 * n_proj * n_wave} B/event")
    print(f"ampcache vs s0e2 (no cache): {par / amp:.2f}x")
    if ratio <= 1.02:
        print("  -> no speedup over s0f, and none was expected: on the fused "
              "path s0f already computes exactly this arithmetic (and its own "
              "all-fixed ladder picks the same tile this kernel measured), so "
              "the value here is the CONTRACT — pinned coefficients, no mask, "
              "no F/T/B, no split scratch — not the rate.")
    # the cache's real footprint, both handles
    kb = 8 * n_proj * n_wave
    print(f"cache       {kb} B/event -> {kb * ne_d / 1e6:.0f} MB data + "
          f"{kb * ne_p / 1e6:.0f} MB phsp ({2 * kb * ne_d / 1e6:.0f} MB at "
          f"equal sizes)")

    # ── the Gram build: numpy vs the GPU reduction of the same cache ─────
    # The GPU build STREAMS (one mc_batch batch on the device at a time), so
    # its time is the reduction PLUS a load+fill per batch — the honest cost
    # of not holding the sample on the device.
    from tabpwa.integrated_pwa_gen import IntegratedPWAFlat
    from tabpwa.monitor import create_monitor

    params = f.build_params(x)[0]
    coeff0 = np.array(params["coeff"], np.float64, copy=True)
    oracle = IntegratedPWAFlat(kc, chunk=args.mc_batch)
    t = time.perf_counter()
    D_np = oracle.gram(phsp, coeff0)
    t_np = time.perf_counter() - t

    be = create_backend({"name": "integrated_pwa_gen",
                         "base": "cuda_flat_pwa_gen_ampcache"},
                        kc, model="pwa_gen", mc_batch=args.mc_batch)
    t, peak = float("inf"), None
    for _ in range(2):
        mon = create_monitor(what=("gpu_used",), interval=0.01)
        b = be.load_data(phsp)
        try:
            t0 = time.perf_counter()
            D_gpu = be._ensure_gram(b, coeff0)
            t_gpu = time.perf_counter() - t0
        finally:
            b.free()
        res = mon.close()
        t = min(t, t_gpu)
        peak = res.peak("gpu_used")
    rel = np.abs(D_gpu - D_np).max() / np.abs(D_np).max()
    be.free()

    # the number the FIT consumes: the norm, against the base's own integral
    # over the phase space (a free-coefficient kernel, so nothing is shared
    # with either Gram builder beyond the config)
    ck = np.asarray(params["ck"], np.complex128)
    q_gpu, q_np = oracle.norm(ck, D_gpu), oracle.norm(ck, D_np)
    ref = create_backend("cuda_flat_pwa_gen_s0e2", kc, model="pwa_gen")
    try:
        q_ref = ref.compute(params, ref.load_data(phsp), norm=None,
                            return_p=False)[0]
    finally:
        ref.free()

    save = steady["s0e2"] - steady["integrated/ampcache"]
    print()
    print(f"gram        numpy {t_np * 1e3:8.1f} ms   GPU {t * 1e3:8.1f} ms "
          f"({t_np / t:5.2f}x)   |Dgpu - Dnumpy|/|Dnumpy| {rel:.2e}")
    print(f"            phsp {ne_p} events, mc_batch {args.mc_batch}: the "
          f"device holds ONE batch ({args.mc_batch * kb / 1e6:.1f} MB), not "
          f"the sample ({ne_p * kb / 1e6:.0f} MB)")
    print(f"            MEASURED peak gpu_used during the streamed build: "
          f"{peak} MiB")
    print(f"            norm Re(ck+ D ck) vs s0e2's sum w P ({q_ref:.6g}): "
          f"GPU {abs(q_gpu - q_ref) / abs(q_ref):.2e}   "
          f"numpy {abs(q_np - q_ref) / abs(q_ref):.2e}")
    # TWO break-evens, because they answer different questions:
    #   vs ampcache — the SAME data kernel, so the difference IS the norm the
    #                 Gram replaces (the honest "what the Gram buys");
    #   vs s0e2      — the whole integrated path against the no-cache world.
    save_amp = steady["ampcache"] - steady["integrated/ampcache"]
    for why, sv in (("vs ampcache (norm only)", save_amp),
                    ("vs s0e2 (whole path)", save)):
        print(f"break-even  {why:23s} saving {sv * 1e3:6.1f} ms/iter -> "
              f"{(t / sv if sv > 0 else float('inf')):.0f} iterations")
    if save <= 0:
        print("  -> the numpy build is cheaper than the whole saving: no "
              "break-even on this shape")
    if t_np < t:
        print("  -> the numpy build WON: streaming pays a load+fill per batch,")
        print("     and on this shape that costs more than the reduction "
              "saves.")


if __name__ == "__main__":
    main()

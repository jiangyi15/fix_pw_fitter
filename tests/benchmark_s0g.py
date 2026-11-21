#!/usr/bin/env python3
"""Time ``cuda_flat_pwa_gen_s0g`` against ``…s0f`` and the parent, mask on.

Three backends on the SAME sample and the SAME config, interleaved and min-of-N
(the box throttles, so sequential numbers are meaningless):

  * ``cuda_flat_pwa_gen_s0e2`` — the parent, no cache;
  * ``cuda_flat_pwa_gen_s0f`` — the fixed cache with F/T/B (the split);
  * ``cuda_flat_pwa_gen_s0g`` — the fixed cache with the free block UN-STAGED
    (one kernel per chunk, no F/B, no scratch).

The mask is every column a free ``x`` slot does NOT move (``--free-mw``
pins every other mass/width, ``bench.py``'s convention).

Reports ms/iteration of ``Fitter.get_nll`` (norm + data + gradients), Mev/s
over data + phase space, the mask's split (n_fix / n_free / moving columns /
fixed-block non-zeros) and the cache's memory (``8·n_proj·n_fix`` B/event).

    python tests/benchmark_s0g.py                          # km fixture, synthetic
    python tests/benchmark_s0g.py --config benchmarks/ctpwa_kketa/config_gen.yml \\
        --rsize 17 --n-data 200000 --n-phsp 200000
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

#: the ten mass/width names the CTPWA benchmark frees (its README §4), spelled
#: the way `bench.py --free-mw` spells them.  Pass them explicitly:
#:     --free-mw phi1680_mass,phi1680_width,...
CTPWA_FREE_MW = ("phi1680_mass", "phi1680_width", "X1750_mass", "X1750_width",
                 "omega2220_mass", "omega2220_width", "K3st1780p_mass",
                 "K3st1780p_width", "K4st2045p_mass", "K4st2045p_width")


def _pin_other_mw(f, keep):
    """``bench.py --free-mw``: pin every mass/width whose NAME is not in keep."""
    known = set(dict.fromkeys(f.var_registry.flat_names))
    unknown = sorted(set(keep) - known)
    if unknown:
        avail = sorted(n for n in known if n.endswith(("_mass", "_width")))
        raise SystemExit(
            f"benchmark_s0g.py: unknown --free-mw name(s): {unknown}\n"
            f"  the fitted mass/width names are:\n  {avail}")
    mw = {n: float(f.defaults[n])
          for n in dict.fromkeys(f.var_registry.flat_names)
          if n.endswith(("_mass", "_width")) and n not in keep
          and n in f.defaults}
    f.set_fixed(mw)


def _moving_columns(f, x, step=1e-3):
    """The ``coeff`` columns a perturbation of ANY free ``x`` slot moves.

    Ground truth, so an ALIASED parameter is included through the columns it
    really moves — `K3st1780p_mass` also moves `K3st1780m_s0`/`_width_t`,
    whose names do not share its prefix.  A name-prefix mask misses those and
    declares a column fixed that still moves (the cache then goes stale).
    """
    names = list(f.fwd_names())
    if len(names) != len(x):
        raise RuntimeError("fwd_names and x disagree — cannot derive the mask")
    c0 = np.asarray(f.build_params(x)[0]["coeff"], np.float64)
    moving = set()
    for i in range(len(x)):
        h = step * max(abs(float(x[i])), 1.0)
        for sgn in (1.0, -1.0):
            xp = np.asarray(x, np.float64).copy()
            xp[i] += sgn * h
            c = np.asarray(f.build_params(xp)[0]["coeff"], np.float64)
            moving.update(int(j) for j in np.nonzero(c != c0)[0])
    return np.asarray(sorted(moving), np.int64)


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
    return _events(kc, n_data or 4000, seed), _events(kc, n_phsp or 4000,
                                                      seed + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tests/config_pwa_gen_km.yml")
    ap.add_argument("--rsize", type=int, default=1)
    ap.add_argument("--n-data", type=int, default=0,
                    help="cap/create the data sample (0 = config's / 4000)")
    ap.add_argument("--n-phsp", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=5,
                    help="timed calls per backend per sweep (steady state)")
    ap.add_argument("--sweeps", type=int, default=2,
                    help="whole passes over the backends; the steady row is "
                         "the min over sweeps (this GPU throttles)")
    ap.add_argument("--batch-size", type=int, default=50000)
    ap.add_argument("--free-mw", default="",
                    help="comma-separated mass/width NAMES left free (e.g. "
                         "benchmarks/ctpwa_kketa's ten: " +
                         ",".join(CTPWA_FREE_MW) + "); every other "
                         "mass/width is pinned at its default")
    ap.add_argument("--fix-mw", action="store_true",
                    help="pin EVERY fitted mass/width (the paper's fixed "
                         "column); the mask is then derived from whatever is "
                         "still free (couplings, extra basis params)")
    ap.add_argument("--amp-model", default="",
                    help="override the config's amp_model (e.g. pwa_gen on a "
                         "plain-pwa production config)")
    ap.add_argument("--mask-all", action="store_true",
                    help="declare EVERY denominator column fixed (a config "
                         "whose every mass/width is pinned); overrides "
                         "--free-mw")
    args = ap.parse_args()

    if args.amp_model:
        import yaml
        raw = yaml.safe_load(open(args.config))
        raw["amp_model"] = args.amp_model
        # the timed backends replace `f.backend`; the construction-time spec
        # must merely be valid for the new model
        raw["backend"] = {"name": "numpy_flat"}
        f = Fitter(raw)
    else:
        f = Fitter(args.config)
    f.apply_constrains()
    free_mw = tuple(s.strip() for s in args.free_mw.split(",") if s.strip())
    if args.fix_mw:
        _pin_other_mw(f, ())
    if free_mw:
        _pin_other_mw(f, free_mw)
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    data, phsp = _load_sample(f, args.config, args.n_data, args.n_phsp, seed=7)
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])
    if args.mask_all:
        # mask-all is only honest when NOTHING that is still free moves a
        # column: check the ground truth, not just the mass/width names.
        still = _moving_columns(f, x)
        if still.size:
            den = [str(v) for v in np.asarray(kc["den_names"]).ravel()]
            names = [den[int(i)] for i in still[:6]]
            raise SystemExit(
                "benchmark_s0g.py: --mask-all needs every parameter that "
                f"moves a coeff column pinned, but {still.size}/{n_basis} "
                f"columns still move ({names}…); pin them or drop --mask-all")
        fixed_cols = list(range(n_basis))
        moving = np.zeros(0, np.int64)
    else:
        # the moving set by PERTURBATION (aliases included), never by name.
        # With no --free-mw this is the CONFIG's own free `x` — e.g. the
        # production config's `particle_float` list — so the mask is honest.
        moving = _moving_columns(f, x)
        fixed_cols = [i for i in range(n_basis) if i not in set(moving.tolist())]

    # the mask is declared WITH its values: the coefficients at the point the
    # fit starts from (the same `x` every `get_nll` below uses)
    vals_at_x = np.array(f.build_params(x)[0]["coeff"], np.float64, copy=True)
    specs = [("s0e2", "cuda_flat_pwa_gen_s0e2", {}),
             ("s0g/off", "cuda_flat_pwa_gen_s0g", {}),
             # the same mask through both cache kernels: F/T/B vs one pass
             ("s0f/on split", "cuda_flat_pwa_gen_s0f",
              {"fixed_cols": fixed_cols, "fixed_values": vals_at_x}),
             ("s0g/on", "cuda_flat_pwa_gen_s0g",
              {"fixed_cols": fixed_cols, "fixed_values": vals_at_x})]
    backends = []
    for label, name, kw in specs:
        be = create_backend(name, kc, model="pwa_gen",
                            resolution_size=args.rsize,
                            batch_size=args.batch_size, **kw)
        backends.append((label, be))

    # ── the mask, as the kernel sees it ────────────────────────────────
    k = backends[2][1].kernel
    nnz = int(k.n_nz_full)
    fixed_nnz = int(sum((k.gs_hi_full - k.gs_lo_full)[np.nonzero(
        k.fixed_wave)[0]]))
    print(f"config      {args.config}")
    print(f"shape       n_wave {k.n_wave}  n_proj {k.n_proj}  n_basis "
          f"{k.n_basis}  n_res {k.n_res}  rsize {args.rsize}")
    print(f"events      data {len(np.asarray(data['weight']))}  phsp "
          f"{len(np.asarray(phsp['weight']))}")
    source = ("mask-all" if args.mask_all
              else ("pinned --free-mw, moved columns by x perturbation"
                    if args.free_mw else
                    "config's free x, moved columns by perturbation"))
    print(f"mask        moving columns {len(moving)}/{k.n_basis} "
          f"({source})  n_fix {k.n_fix}  n_free {k.n_free}  "
          f"fixed nnz {fixed_nnz}/{nnz}")
    ne_d = len(np.asarray(data["weight"]))
    ne_p = len(np.asarray(phsp["weight"]))
    print(f"cache       8·P·n_fix·ne = {k.fix_bytes(ne_d) / 1e6:.1f} MB "
          f"data + {k.fix_bytes(ne_p) / 1e6:.1f} MB phsp")
    g = backends[3][1].kernel
    print(f"plan        s0f te {k.te} (one group)   s0g te {g.te} "
          f"({g.groups_per_block()} group(s)/block, {g.smem_bytes / 1024:.1f} "
          f"KiB shared, no scratch)")

    def measure(name, kw):
        """Load ONCE per backend, timing the LOAD (which carries the ``Afix``
        fill — the values are declared with the mask), then the first COMPUTE
        (no fill) and the steady state, and release the handles."""
        be = create_backend(name, kc, model="pwa_gen",
                            resolution_size=args.rsize,
                            batch_size=args.batch_size, **kw)
        try:
            f.backend = be
            # Fitter.set_data overwrites `_data_holder` without freeing it, so
            # a previous backend's handle would leak ~GB; release it here.
            if f._data_holder is not None:
                f._data_holder.free()
                f._data_holder = None
            t0 = time.perf_counter()
            f.set_phsp(phsp)              # frees the previous phsp handle
            f.set_data(data)              # LOAD: both samples + their Afix fill
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

    def mask_change():
        """The cost of a mask change with the samples ALREADY loaded.

        Uses the ``fixed_coeff`` wrapper (the mask lifecycle's home): a mask
        change rebuilds the base and the retained samples re-load on their next
        compute.  Alternates two honest masks (A = the derived fixed columns;
        B = the columns of the first half of A's fixed waves, a subset, so
        every B wave is still genuinely fixed) and reports the two halves
        separately: the base swap (``set_fixed_cols``) and the following
        ``get_nll`` (the per-sample re-prepare + one iteration).
        """
        if not fixed_cols:
            print("\nmask change: no fixed columns here — nothing to change")
            return
        # the wrapper's mask is declared WITH its values (eager fill at load)
        coeff0 = np.asarray(f.build_params(x)[0]["coeff"], np.float64)

        def set_cols(be, m):
            return be.set_fixed_cols(
                m, {int(c): float(coeff0[int(c)]) for c in m})
        be = create_backend({"name": "fixed_coeff",
                             "base": "cuda_flat_pwa_gen_s0g"},
                            kc, model="pwa_gen",
                            resolution_size=args.rsize,
                            batch_size=args.batch_size)
        try:
            set_cols(be, fixed_cols)                  # masked base before load
            kk = be.base.kernel
            if kk.n_fix == 0:
                print("\nmask change: no fully-fixed wave here — nothing to change")
                return
            fw = np.asarray(kk.fixed_waves)
            half = fw[: max(1, fw.size // 2)]
            cols_b = sorted({int(c) for w in half
                             for c in kk.nz_m_full[kk.gs_lo_full[w]:
                                                   kk.gs_hi_full[w]]})
            mask_a = list(fixed_cols)
            f.backend = be
            if f._data_holder is not None:
                f._data_holder.free()
                f._data_holder = None
            f.set_phsp(phsp)
            f.set_data(data)
            f.get_nll(x)                     # load both samples on the base
            stable = (be.creates, be.loads)
            f.get_nll(x)                     # UNCHANGED mask: nothing rebuilt
            assert (be.creates, be.loads) == stable
            ts_set, ts_call, te_of = [], [], {}
            for i in range(args.repeat):
                m = cols_b if i % 2 == 0 else mask_a
                t = time.perf_counter()
                set_cols(be, m)                      # base swap only
                ts_set.append(time.perf_counter() - t)
                t = time.perf_counter()
                f.get_nll(x)                         # samples re-load here
                ts_call.append(time.perf_counter() - t)
                te_of[len(m)] = (be.base.kernel.te, be.base.kernel.n_fix)
            print()
            print(f"mask change  samples loaded: set_fixed_cols "
                  f"{min(ts_set) * 1e3:.2f} ms (base swap) + "
                  f"next get_nll {min(ts_call) * 1e3:.2f} ms (both samples "
                  f"re-prepared)  -> {1e3 * (min(ts_set) + min(ts_call)):.2f}"
                  f" ms per change+call")
            print(f"             fixed_coeff counters: creates {be.creates}, "
                  f"loads {be.loads}, live records {len(be._records)}; an "
                  f"unchanged-mask compute adds none")
            for ncols, (te, nf) in sorted(te_of.items()):
                print(f"             mask with {ncols} fixed cols -> "
                      f"n_fix {nf}, te {te}")
        finally:
            if f._data_holder is not None:
                f._data_holder.free()
                f._data_holder = None
            if f._phsp_holder is not None:
                f._phsp_holder.free()
                f._phsp_holder = None
                f._phsp_scratch = None
            be.free()

    n_ev = ne_d + ne_p
    first = {}
    load_t = {}
    steady = {label: float("inf") for label, _n, _kw in specs}
    for sweep in range(args.sweeps):
        for label, name, kw in specs:
            ld, fi, ts = measure(name, kw)
            if sweep == 0:
                load_t[label], first[label] = ld, fi
            steady[label] = min(steady[label], min(ts))
    # interleaving all four backends at once does not fit: at the CTPWA
    # sample each handle set is ~5.6 GB, so the handles are loaded one backend
    # at a time and the min spans two whole sweeps to absorb the throttle.
    print()
    best = min(steady.values())
    for label, _n, _kw in specs:
        fms, sms = first[label] * 1e3, steady[label] * 1e3
        print(f"{label:15s} load+fill {load_t[label] * 1e3:7.2f} ms   "
              f"first {fms:7.2f} ms   steady {sms:8.2f} ms   "
              f"{n_ev / steady[label] / 1e6:7.2f} Mev/s   "
              f"{sms * 1e6 / n_ev:6.2f} ns/event   "
              f"{sms / (best * 1e3):5.2f}x")
    mask_change()

if __name__ == "__main__":
    main()

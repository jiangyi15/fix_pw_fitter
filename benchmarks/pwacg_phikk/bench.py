#!/usr/bin/env python3
"""Time one `Fitter.get_nll` (norm + data NLL + gradients) on the PWACG model.

    python bench.py --config config_gen.yml    --backend cuda_flat_pwa_gen_s0e2 --max-events 10000 --repeat 10 --vary
    python bench.py --config config_pwacg.yml  --backend auto --max-events 10000 --repeat 5 --vary

Two models, two questions:

* **`config_gen.yml`** — the MATCHED model: 24 waves, 2 polarisations, the
  paper's six resonances and Table 1 masses/widths, with the φ declared
  `model: one`.  That collapses the denominator's factor axis to 1
  (`n_res = 1`), which is the only way a fast tile (`s0e*`, `s0f`, `s0g`) applies
  — and is the config to quote when the question is "how fast is a kernel at
  this wave content".  Line shapes: `BW` (constant width) for four resonances,
  `FlatteC` with the ππ/K-K channel axis for f₀(980), `BWR` for f₂(1270).
* **`config_pwacg.yml`** — the FAITHFUL model: the φ is kept (their
  `BW(1.02, 0.004)` multiplies every wave) and every line shape is theirs
  verbatim, from `extra_amp.py` (`PWACG_phi_BW`, `PWACG_flatte980`,
  `PWACG_flatte1270`).  `check_model.py` verifies each denominator against their
  generated formula to machine precision.  Keeping the φ makes the chain
  `n_res = 2`, so this model runs on `cuda_flat_pwa_gen2` and the fast tiles
  refuse it — the honest cost of the faithful model.

Both carry `x = 59` (23 free complex couplings + 13 lineshape parameters), the
paper's count.

**What the numbers are compared against** (PWACG itself, on this box, on the
same events, min-of-5 steady state, XLA compile excluded):

| reference quantity | their time |
|---|---|
| `data_likelihood_kk`, value + gradient, 10 000 events | 0.394 ms |
| `mc_likelihood_kk`, value + gradient, 10 000 events | 0.388 ms |
| one fit iteration (data + MC, 10 k / 10 k) | **0.782 ms** |
| the same at 100 k / 100 k, 1 M / 1 M | 2.12 ms, 34.80 ms |

so `--max-events N` (N data + N phase space) is the counterpart of their
iteration at N.  Run theirs with `pwacg_ref/bench_ref.py` (see
`pwacg_ref/BUILD.md`).  Their `--mc-sizes` and `--data-sizes` accept the same
counts, so the two tables are produced on identical events.

Prints ms/iteration, Mev/s over data + phase space, the NLL and `x` (free
parameters), plus the kernel's `n_wave` / `n_basis` / `n_proj` / `n_res`, so a
backend cannot look fast by computing less.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import extra_amp                     # noqa: E402,F401  registers the line shapes
from tabpwa import Fitter            # noqa: E402
from tabpwa.backends import backend_tags   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config_gen.yml"))
    ap.add_argument("--backend", default="cuda_flat_pwa_gen2",
                    help="a backend registered for --config's model.  The "
                         "`s0e*`/`s0f`/`s0g` tile family refuses this topology "
                         "(n_res = 2); `cuda_flat_pwa_gen2` is the equivalent "
                         "general leaf")
    ap.add_argument("--rsize", type=int, default=0,
                    help="0 = do not pass resolution_size (per-event NLL)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--base", default=None,
                    help="composer base, e.g. an n_res1 kernel for fixed_coeff")
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--max-data", type=int, default=0,
                    help="cap the data sample only (default: --max-events)")
    ap.add_argument("--max-phsp", type=int, default=0,
                    help="cap the phase-space sample only (default: --max-events)")
    ap.add_argument("--fix-mw", action="store_true",
                    help="fix every mass/width at its config value (the paper's "
                         "scenario: phi fixed, only the couplings float)")
    ap.add_argument("--vary", action="store_true",
                    help="perturb x on every iteration, so couplings and any free "
                         "mass/width change as they do in a real fit (default: the "
                         "same x, which can hide a value-keyed cache)")
    ap.add_argument("--vary-eps", type=float, default=1e-3,
                    help="relative step of --vary (per slot)")
    ap.add_argument("--free-mw", default="",
                    help="comma-separated mass/width names to leave FREE "
                         "(implies --fix-mw for the rest)")
    ap.add_argument("--mask", action="store_true",
                    help="apply the constraint-derived fixed-column mask to a "
                         "`fixed_coeff` base (`--backend fixed_coeff --base "
                         "<kernel>`).  No-op on this topology: the mask kernels "
                         "are n_res1-only (see the module docstring).")
    args = ap.parse_args()

    t0 = time.perf_counter()
    spec = {"name": args.backend}
    label = args.backend
    if args.base:
        spec["base"] = args.base
    if args.rsize:
        spec["resolution_size"] = args.rsize
    f = Fitter(args.config, backend=spec)
    chosen = getattr(f.backend, "chosen", None)
    if chosen and chosen != args.backend:
        label = f"{args.backend}->{chosen}"
    t_init = time.perf_counter()
    nd = args.max_data or args.max_events
    np_ = args.max_phsp or args.max_events
    phsp = f.load_dataset("phsp", **({"max_events": np_} if np_ else {}))
    data = f.load_dataset("data", **({"max_events": nd} if nd else {}))
    t_io = time.perf_counter()
    f.set_phsp(phsp)
    f.set_data(data)
    t_upload = time.perf_counter()
    ne, nb = len(data["weight"]), len(phsp["weight"])
    f.apply_constrains()
    if args.fix_mw or args.free_mw:
        keep = {s.strip() for s in args.free_mw.split(",") if s.strip()}
        known = set(dict.fromkeys(f.var_registry.flat_names))
        unknown = sorted(keep - known)
        if unknown:
            avail = sorted(n for n in known if n.endswith(("_mass", "_width")))
            raise SystemExit(
                f"bench.py: unknown --free-mw name(s): {unknown}\n"
                f"  the fitted mass/width names are:\n  {avail}")
        mw = {n: float(f.defaults[n])
              for n in dict.fromkeys(f.var_registry.flat_names)
              if n.endswith(("_mass", "_width")) and n not in keep
              and n in f.defaults}
        f.set_fixed(mw)
    if args.mask:
        if not hasattr(f.backend, "set_fixed_coeffs"):
            ap.error("--mask needs --backend fixed_coeff (--base names the "
                     "kernel); the delegate is the mask's lifecycle")
        f.backend.set_fixed_coeffs(f)                # same x0 as the timing
        if args.base:
            label = f"{args.base} (masked)"
    x0 = f.initial_values(seed=1)
    t_prep = time.perf_counter()

    n_x = len(x0)
    names = [spec.get("name", "")] + ([spec["base"]] if spec.get("base") else [])
    if any("integrated" in n for n in names):
        free_mw = [n for n in dict.fromkeys(f.var_registry.flat_names)
                   if n.endswith(("_mass", "_width"))]
        if free_mw:
            raise SystemExit(
                f"bench.py: {spec} is an integrated (Gram) backend, which zeroes "
                f"the m0/g0 gradient blocks, but {len(free_mw)} mass/width "
                f"parameters are free: the printed ms would look fine and the "
                f"gradient would be wrong in {len(free_mw)} directions.  Pass "
                f"--fix-mw, or use a per-event kernel.")
    notes = []
    if "coeff_pinned" in backend_tags(spec.get("name", ""), f.model.name):
        notes.append("coeff pinned: grads['coeff'] == 0 by contract, NOT a fit")
    if args.vary and notes:
        raise SystemExit(
            "bench.py: --vary moves `coeff`, and a `coeff_pinned` backend raises "
            "from its second call (the cache is built once per handle).  Drop "
            "--vary to time the cache, or use a free-coefficient kernel.")

    f.get_nll(x0)                                   # warm-up (lazy precompute)
    t_warm = time.perf_counter()
    ts = []
    nll = float("nan")
    rng = np.random.default_rng(12345)
    x_cur = x0.copy()
    for _ in range(args.repeat):
        if args.vary:
            # a real fit moves every free slot; scale the step by the slot itself
            x_cur = x0 + args.vary_eps * (1.0 + np.abs(x0)) * rng.standard_normal(x0.size)
        a = time.perf_counter()
        nll, _g = f.get_nll(x_cur)
        # the contract the comparison relies on: ONE scalar and a gradient over
        # the SAME parameter vector — no Hessian, nothing of a different shape
        assert np.ndim(nll) == 0, f"get_nll returned a non-scalar {np.shape(nll)}"
        assert np.shape(_g) == (len(x_cur),), (
            f"gradient {np.shape(_g)} does not span x ({len(x_cur)}); the "
            f"per-iteration comparison assumes one value + one gradient")
        ts.append(time.perf_counter() - a)
    t = min(ts)
    mask_note = ""
    if args.mask:
        _k = getattr(getattr(f.backend, "base", None), "kernel", None)
        if getattr(_k, "n_fix", None) is not None:
            mask_note = f"  [mask n_fix {_k.n_fix} / n_free {_k.n_free}]"
    k = getattr(f.backend, "kernel", None) or getattr(
        getattr(f.backend, "base", None), "kernel", None)
    shape = ""
    if k is not None and getattr(k, "n_wave", None) is not None:
        shape = (f"  [n_wave {k.n_wave} n_basis {getattr(k, 'n_basis', '?')} "
                 f"n_proj {getattr(k, 'n_proj', '?')} n_res {getattr(k, 'n_res', '?')}]")
    note = ("  [" + "; ".join(notes) + "]") if notes else ""
    print(f"{label:34s} x={n_x:3d} ne={ne:>9,} nphsp={nb:>9,}  "
          f"{t*1e3:8.1f} ms/iter  {(ne+nb)/t/1e6:6.2f} Mev/s  NLL={nll:+.6f}"
          + ("  [vary]" if args.vary else "") + mask_note + note, flush=True)
    if shape:
        print(f"{'':34s} shape{shape}")
    print(f"{'':34s} prep: init {t_init - t0:5.1f} | file i/o {t_io - t_init:5.1f} | "
          f"upload+caches {t_upload - t_io:5.1f} | constrain {t_prep - t_upload:5.1f} | "
          f"first NLL {t_warm - t_prep:6.1f} | total {t_warm - t0:5.1f} s",
          flush=True)
    f.backend.free()


if __name__ == "__main__":
    main()

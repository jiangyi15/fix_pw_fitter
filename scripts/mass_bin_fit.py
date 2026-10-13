#!/usr/bin/env python3
"""Mass-binned partial-wave amplitudes ck from a direct per-bin fit.

For each invariant-mass bin, subset the data AND the phase space to the same
edges, set them on a single ``Fitter`` (single-chain config), and fit — with
only the ``ck`` amplitudes free (``m0``/``g0`` and the ``g_ls`` fixed by
``apply_constrains``).  Read back the fitted ``ck`` and their fit errors, and
plot ``|ck|`` (and the phase relative to a reference wave) versus mass.

    python scripts/mass_bin_fit.py --config tests/config_pwa.yml \\
        --data data_pwa.npz --phsp phsp.npz --bin-width 0.15 \\
        -o fit_ck.csv --plot plots/

The overall ``|ck|`` scale is a **gauge** (the likelihood uses ``P/∫P``, so it
is scale-invariant — see ``--free-ref``).  The scale-free, well-defined output
is the per-wave **fit fraction** ``R_m = Σ w|A_m|² / Σ w|Σ A|²`` and its
uncertainty (propagated from the fit covariance via ``AmplitudeFractions``).

Notes
-----
* Both data and phsp must be binned on the same edges (``--phsp`` required):
  the normalisation is ``∫P`` over the phase space you pass.
* The ``MI1m``/``MI3m`` angular degeneracy is not removed by fitting; check
  ``res.success``/``res.fun`` per bin.
"""
import argparse
import contextlib
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tabpwa import Fitter
from tabpwa.amp_frac import AmplitudeFractions
from mass_bin_moments import _edges, _wave_names


def _load(f, path, n_comp):
    """Kernel event dict from a .npz (precomputed) or a (N, n_finals, 4) .npy."""
    if path is None:
        return None
    if str(path).endswith(".npz"):
        return f.load_npz(path, n_angle_comp=n_comp)[0]
    conf = f.config.dic.get("data") or {}
    order = conf.get("dat_order") or list(f.model.finals)
    perm = [list(order).index(x) for x in f.model.finals]
    mom = np.asarray(np.load(path), float)[:, perm]
    return f.model.build_event_data(mom, f.kernel_config)


def _subset(d, mask):
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.shape[:1] == mask.shape else v)
            for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default=None, help=".npz or momentum .npy")
    ap.add_argument("--phsp", default=None, help=".npz or momentum .npy (required)")
    ap.add_argument("--backend", default="numpy_pwa")
    ap.add_argument("--mass-col", type=int, default=0)
    ap.add_argument("--bin-width", type=float, default=0.05)
    ap.add_argument("--nbins", type=int, default=None)
    ap.add_argument("--m-range", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--min-counts", type=float, default=0.0)
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--verbose", action="store_true",
                    help="let the fit print its per-iteration tracker")
    ap.add_argument("--ref", type=int, default=0,
                    help="reference wave index (its phase is fixed)")
    ap.add_argument("--free-ref", action="store_true",
                    help="also release the reference magnitude (the overall "
                         "scale is FLAT under the normalised likelihood — "
                         "expect erratic |ck|)")
    ap.add_argument("--no-phase", action="store_true")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    f = Fitter(args.config, backend=args.backend)
    f.apply_constrains()
    kc = f.kernel_config
    n_comp = len([k for _, k in kc.get("variables", [])])
    n_wave = len(kc["ck_map"])
    waves = _wave_names(f, n_wave)
    total = [f"{e[0]}" for e in kc["ck_map"]]        # *total_0 slot names

    if args.free_ref:
        f.set_free(f"{total[args.ref]}r")

    data = _load(f, args.data, n_comp) or f.load_dataset("data")
    phsp = _load(f, args.phsp, n_comp) or f.load_dataset("phsp")
    if data is None or phsp is None:
        raise SystemExit("need both data and phsp (config paths or --data/--phsp)")
    M = np.asarray(data["mass"], float)[:, args.mass_col]
    Mp = np.asarray(phsp["mass"], float)[:, args.mass_col]
    w = data.get("weight", np.ones(len(M)))

    edges = _edges(M, args)
    print(f"config={args.config}  n_events={len(M)}  waves={waves}  "
          f"free={f.free_param_names()}")
    print(f"mass bins: {edges[0]:.4g} .. {edges[-1]:.4g} ({len(edges) - 1} bins)")

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        dm = (M >= lo) & (M < hi)
        pm = (Mp >= lo) & (Mp < hi)
        nw = float(w[dm].sum())
        if nw <= args.min_counts or dm.sum() == 0 or pm.sum() == 0:
            continue
        t0 = time.time()
        f.set_phsp(_subset(phsp, pm))
        f.set_data(_subset(data, dm))
        if args.verbose:
            res = f.fit(f.initial_values(), maxiter=args.maxiter)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                res = f.fit(f.initial_values(), maxiter=args.maxiter)
        ck = f.build_params(res.x)[0]["ck"]
        unc = f.get_uncertainties(res)
        mag = np.abs(ck)
        dphi = np.angle(ck) - np.angle(ck[args.ref])
        merr = np.array([unc.get(f"{total[i]}r", (0.0, 0.0))[1] for i in range(n_wave)])
        perr = np.array([unc.get(f"{total[i]}i", (0.0, 0.0))[1] for i in range(n_wave)])
        # scale-free output: per-wave fit fraction + propagated uncertainty
        fv, fe = AmplitudeFractions(f, res).fractions([[i] for i in range(n_wave)])
        frac, ferr = np.asarray(fv, float), np.asarray(fe, float)
        ntot = float(dm.sum())                  # data sample size in the bin
        yld = frac * ntot                       # scale the fractions -> yields
        # error: shape (N·d(frac)) and the data Poisson(N) both propagated
        yerr = np.sqrt((ntot * ferr) ** 2 + frac ** 2 * ntot)
        rows.append((lo, hi, nw, mag, merr, dphi, perr, frac, ferr, yld, yerr))
        cells = "  ".join(f"|{waves[i]}|={mag[i]:6.3f}  Y={yld[i]:7.1f}±{yerr[i]:.1f}"
                          for i in range(n_wave))
        print(f"{lo:>7.3f} {hi:>7.3f} {nw:>7.0f}  nit={getattr(res, 'nit', -1):>3} "
              f"{'ok' if res.success else 'FAIL'}  {cells}   [{time.time()-t0:.1f}s]")

    if not rows:
        raise SystemExit("no bins passed --min-counts")

    if args.output:
        cols = ["m_lo", "m_hi", "n_events"]
        for wv in waves:
            cols += [f"|ck_{wv}|", f"|ck_{wv}|_err", f"frac_{wv}", f"frac_{wv}_err",
                     f"yield_{wv}", f"yield_{wv}_err"]
        if not args.no_phase:
            for wv in waves:
                cols += [f"arg_ck_{wv}", f"arg_ck_{wv}_err"]
        arr = []
        for lo, hi, nw, mag, merr, dphi, perr, frac, ferr, yld, yerr in rows:
            row = [lo, hi, nw, *[x for p in zip(mag, merr, frac, ferr, yld, yerr) for x in p]]
            if not args.no_phase:
                row += [x for p in zip(dphi, perr) for x in p]
            arr.append(row)
        np.savetxt(args.output, np.array(arr), delimiter=",", header=",".join(cols),
                   comments="", fmt="%.6g")
        print(f"\nwrote {args.output} ({len(rows)} bins x {len(cols)} cols)")

    if args.plot and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.plot, exist_ok=True)
        centers = np.array([0.5 * (lo + hi) for lo, hi, *_ in rows])
        halfw = np.array([0.5 * (hi - lo) for lo, hi, *_ in rows])
        for tag, iy, ie in (("mag", 3, 4), ("frac", 7, 8), ("yield", 9, 10),
                            ("phase", 5, 6)):
            if tag == "phase" and args.no_phase:
                continue
            fig, ax = plt.subplots(figsize=(6.4, 4.4))
            for m, wv in enumerate(waves):
                ys = np.array([r[iy][m] for r in rows], float)
                es = np.array([r[ie][m] for r in rows], float)
                ax.errorbar(centers, ys, xerr=halfw, yerr=es, fmt="o-",
                            ms=4, lw=1.2, capsize=2.5, label=wv)
            ax.set_xlabel("mass")
            ax.set_ylabel({"mag": "|ck|", "frac": "fit fraction",
                           "yield": "yield (events)",
                           "phase": "arg(ck) - arg(ck_ref)"}[tag])
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fn = os.path.join(args.plot, f"fit_ck_{tag}.png")
            fig.savefig(fn, dpi=130)
            plt.close(fig)
            print("plot:", fn)

    f.free()


if __name__ == "__main__":
    main()

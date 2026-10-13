#!/usr/bin/env python3
"""Mass-binned partial-wave amplitudes ck from the angular moment basis.

Cut the data into invariant-mass bins and, for each bin, solve the wave-pair
products ``kappa = ck ck'*`` with the linear moment method, then take the
rank-1 factorisation (dominant eigenpair) to obtain the complex amplitudes
``ck``.  Saves ``|ck_m|`` and the phase relative to a reference wave, with
uncertainties (Monte-Carlo over the moment covariance), versus mass.

    python scripts/mass_bin_ck.py --config tests/config_pwa.yml \\
        --data data_pwa.npz --bin-width 0.15 -o ck.csv --plot plots/

Caveat: ``kappa`` comes from the min-norm solve, so it is only approximately
rank-1 (the truly-null directions are zeroed, the partly-null ones are a
representative).  The factorised ``ck`` is the best rank-1 projection of the
measured ``kappa``; the mass-bin observables of ``mass_bin_moments.py`` stay
the robust output.  ``ck`` also carries a global phase per bin — only the
magnitudes and the *relative* phases are physical.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tabpwa import Fitter
from mass_bin_moments import _edges, _load_event_data, _wave_names


def _rank1_ck(kappa, ref):
    """Dominant rank-1 factor ``ck`` of the Hermitian ``kappa = ck ck'*``,
    global phase fixed so ``ck[ref]`` is real and non-negative."""
    w, v = np.linalg.eigh(np.asarray(kappa, complex))
    k = int(np.argmax(w))
    ck = np.sqrt(max(float(w[k]), 0.0)) * v[:, k]
    if abs(ck[ref]) > 0:
        ck = ck * np.exp(-1j * np.angle(ck[ref]))
    return ck


def _ck_errors(am, angles_bin, weight_bin, ref, n_mc, rng):
    """MC over the moment covariance: sample the moment sums
    ``nu ~ N(nu_hat, Cov)`` with ``Cov = sum_e w_e^2 W_e W_e'``, re-solve the
    coefficients and re-factorise; return the magnitude and phase spreads."""
    Bw = am.basis(angles_bin) * np.asarray(weight_bin, float)[:, None]
    nu = Bw.sum(0)
    cov = Bw.T @ Bw
    m = len(nu)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(m) * np.trace(cov) / m)
    samps = np.empty((n_mc, am.n_wave), complex)
    for s in range(n_mc):
        c = am.solver @ (nu + L @ rng.standard_normal(m))
        samps[s] = _rank1_ck(am.kappa(c), ref)
    return np.abs(samps).std(0), np.angle(samps).std(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default=None,
                    help=".npz kernel arrays or (N, n_finals, 4) .npy; "
                         "default = the config's data file")
    ap.add_argument("--weight", default=None,
                    help="optional (N,) event-weight .npy (4-momentum path)")
    ap.add_argument("--backend", default="numpy_pwa")
    ap.add_argument("--topo", type=int, default=0)
    ap.add_argument("--mass-col", type=int, default=0)
    ap.add_argument("--bin-width", type=float, default=0.05)
    ap.add_argument("--nbins", type=int, default=None)
    ap.add_argument("--m-range", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"))
    ap.add_argument("--min-counts", type=float, default=0.0)
    ap.add_argument("--ref", type=int, default=None,
                    help="reference wave whose phase is fixed (default: the "
                         "largest-|ck| wave per bin)")
    ap.add_argument("--mc", type=int, default=200,
                    help="MC samples per bin for the uncertainties (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-phase", action="store_true",
                    help="skip the phase plots/columns")
    ap.add_argument("-o", "--output", default=None, help="output CSV path")
    ap.add_argument("--plot", default=None, help="directory for the plots")
    args = ap.parse_args()

    f = Fitter(args.config, backend=args.backend)
    f.apply_constrains()
    kc = f.kernel_config
    kinds = [k for _, k in kc.get("variables", [])]
    n_comp = len(kinds)

    out = _load_event_data(f, args.data, n_comp, args.weight)
    angles = np.asarray(out["angle"], float)[:, args.topo, :n_comp]
    mass = np.asarray(out["mass"], float)[:, args.mass_col]
    weight = out.get("weight")
    weight = np.ones(len(mass)) if weight is None else np.asarray(weight, float)

    am = f.angular_moments()
    waves = _wave_names(f, am.n_wave)
    rng = np.random.default_rng(args.seed)

    edges = _edges(mass, args)
    print(f"config={args.config}  data={args.data or '<config>'}  "
          f"n_events={len(mass)}  waves={waves}  mc={args.mc}")
    print(f"mass bins: {edges[0]:.4g} .. {edges[-1]:.4g} "
          f"({len(edges) - 1} bins)")

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (mass >= lo) & (mass < hi)
        nw = float(weight[sel].sum())
        if nw <= args.min_counts:
            continue
        kappa = am.solve(angles[sel], weights=weight[sel])
        ref = args.ref if args.ref is not None else int(np.argmax(np.abs(np.diag(kappa))))
        ck = _rank1_ck(kappa, ref)
        mag, dphi = np.abs(ck), np.angle(ck) - np.angle(ck[ref])
        merr = perr = np.zeros_like(mag)
        if args.mc:
            merr, perr = _ck_errors(am, angles[sel], weight[sel], ref,
                                    args.mc, rng)
        rows.append((lo, hi, nw, mag, merr, dphi, perr))
        if len(rows) <= 12:
            cells = "  ".join(f"|{w}|={m:7.3f}±{e:.3f}"
                              for w, m, e in zip(waves, mag, merr))
            print(f"{lo:>7.3f} {hi:>7.3f} {nw:>8.0f}  {cells}")

    if not rows:
        raise SystemExit("no bins passed --min-counts")

    if args.output:
        cols = ["m_lo", "m_hi", "n_events"]
        for w in waves:
            cols += [f"|ck_{w}|", f"|ck_{w}|_err"]
        if not args.no_phase:
            for w in waves:
                cols += [f"arg_ck_{w}", f"arg_ck_{w}_err"]
        arr = []
        for lo, hi, nw, mag, merr, dphi, perr in rows:
            row = [lo, hi, nw, *[x for p in zip(mag, merr) for x in p]]
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
        for tag, iy, ie in (("mag", 3, 4), ("phase", 5, 6)):
            if tag == "phase" and args.no_phase:
                continue
            fig, ax = plt.subplots(figsize=(6.4, 4.4))
            for m, w in enumerate(waves):
                ys = np.array([r[iy][m] for r in rows], float)
                es = np.array([r[ie][m] for r in rows], float)
                ax.errorbar(centers, ys, xerr=halfw, yerr=es, fmt="o-",
                            ms=4, lw=1.2, capsize=2.5, label=w)
            ax.set_xlabel("mass")
            ax.set_ylabel("|ck|" if tag == "mag" else "arg(ck) - arg(ck_ref)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fn = os.path.join(args.plot, f"ck_{tag}.png")
            fig.savefig(fn, dpi=130)
            plt.close(fig)
            print("plot:", fn)

    f.free()


if __name__ == "__main__":
    main()

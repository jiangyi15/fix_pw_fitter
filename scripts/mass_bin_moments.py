#!/usr/bin/env python3
"""Mass-binned physical observables via the angular moment basis.

Cut the data into invariant-mass bins and, for each bin, evaluate the
physical ``ck ck'*`` observables through the moment basis on that bin's
canonical angles — no fit, just the linear moment method:

    physical_basis_observables(angles_bin, weights=weight_bin)

i.e. the partial-wave content (``|ck_m|^2``, ``Re``/``Im`` cross terms)
per mass bin, each with its exact Poisson(1) error.

Examples
--------
    # binned toy (kernel-format .npz data)
    python scripts/gen_toy_pwa.py --config tests/config_pwa.yml
    python scripts/mass_bin_moments.py --config tests/config_pwa.yml \\
        --data data_pwa.npz --bin-width 0.05 --m-range 0.3 1.3 -o bins.csv

    # 4-momentum .npy data, explicit bin count
    python scripts/mass_bin_moments.py --config tests/config_pwa.yml \\
        --data slice.npy --nbins 20
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tabpwa import Fitter


def _load_event_data(f, path, n_comp, weight_path=None):
    """Return the kernel event-data dict (mass/q/angle/weight).

    ``.npz`` -> precomputed kernel arrays; otherwise a 4-momentum
    ``.npy`` of shape ``(N, n_finals, 4)`` reordered by ``dat_order``
    and converted on the fly.  ``path=None`` uses the config's ``data``.
    """
    kc = f.kernel_config
    if path is None:
        out = f.load_dataset("data")
        if out is None:
            raise SystemExit("no --data given and config has no data file")
        return out
    if str(path).endswith(".npz"):
        return f.load_npz(path, n_angle_comp=n_comp)[0]
    momenta = np.load(path)
    order = (f.config.dic.get("data") or {}).get("dat_order") \
        or list(f.model.finals)
    perm = [list(order).index(x) for x in f.model.finals]
    momenta = np.asarray(momenta, float)[:, perm]
    if weight_path:
        weight = np.load(weight_path)
    else:
        weight = np.ones(momenta.shape[0])
    return f.model.build_event_data(momenta, kc, weight=weight)


def _wave_names(f, n):
    """Short partial-wave names in ck/``matrix_angle`` order (MI1m, ...)."""
    waves = []
    for entry in f.kernel_config["ck_map"]:
        label = entry[2] if len(entry) > 2 else entry[0]
        waves.append(label.split("->")[0])
    return waves[:n]


def _pretty(name, waves):
    """Replace 'ck_<i>' with the wave name in a physical item name."""
    return re.sub(r"ck_(\d+)", lambda m: waves[int(m.group(1))], name)


def _fmt(terms, waves):
    """Format (coefficient, item) terms into a display string."""
    if len(terms) == 1 and abs(terms[0][0] - 1.0) < 1e-6:
        return _pretty(terms[0][1], waves)          # a pure item
    parts = []
    for idx, (c, name) in enumerate(terms):
        nm = _pretty(name, waves)
        parts.append(f"{c:.3f}{nm}" if idx == 0 else f"{c:+.3f}{nm}")
    return "".join(parts) or "0"


def _edges(mass, args):
    if args.nbins:
        if args.m_range:
            lo, hi = args.m_range
        else:
            lo, hi = float(mass.min()), float(mass.max())
        return np.linspace(lo, hi, args.nbins + 1)
    if args.m_range:
        lo, hi = args.m_range
    else:
        lo, hi = float(mass.min()), float(mass.max())
    n = max(1, int(np.ceil((hi - lo) / args.bin_width)))
    return lo + args.bin_width * np.arange(n + 1)


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
    ap.add_argument("--topo", type=int, default=0,
                    help="topology index whose angles are used")
    ap.add_argument("--mass-col", type=int, default=0,
                    help="mass column in the (N, n_topo*n_res) array")
    ap.add_argument("--bin-width", type=float, default=0.05)
    ap.add_argument("--nbins", type=int, default=None)
    ap.add_argument("--m-range", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"))
    ap.add_argument("--min-counts", type=float, default=0.0,
                    help="skip bins with fewer (weighted) events")
    ap.add_argument("-o", "--output", default=None, help="output CSV path")
    ap.add_argument("--plot", default=None,
                    help="directory to save one plot per observable")
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

    terms, _rows = am.physical_basis()
    labels = [_fmt(t, waves) for t in terms]

    edges = _edges(mass, args)
    print(f"config={args.config}  data={args.data or '<config>'}  "
          f"n_events={len(mass)}  angles={kinds}")
    print(f"waves={waves} ({len(labels)})")
    print("basis elements:")
    for k, t in enumerate(terms):
        parts = "  ".join(f"{c:+.3f} {_pretty(nm, waves)}" for c, nm in t)
        print(f"  basis_{k}: {parts}")
    print(f"mass bins: {edges[0]:.4g} .. {edges[-1]:.4g} "
          f"({len(edges) - 1} bins)")
    header = f"{'m_lo':>8} {'m_hi':>8} {'N':>8}  "
    header += "  ".join(f"{lab:>18}" for lab in labels)
    print(header)

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (mass >= lo) & (mass < hi)
        n = float(weight[sel].sum())
        if n <= args.min_counts:
            continue
        a_bin, w_bin = angles[sel], weight[sel]
        _labs, vals, errs = am.physical_basis_observables(
            a_bin, weights=w_bin)
        cells = "  ".join(f"{v:>9.4f}±{e:<7.4f}" for v, e in zip(vals, errs))
        print(f"{lo:>8.4f} {hi:>8.4f} {n:>8.0f}  {cells}")
        rows.append((lo, hi, n, vals, errs))

    if args.output:
        cols = ["m_lo", "m_hi", "n_events"]
        for lab in labels:
            cols += [lab, lab + "_err"]
        arr = [[lo, hi, n, *[x for pair in zip(v, e) for x in pair]]
               for lo, hi, n, v, e in rows]
        np.savetxt(args.output, np.array(arr), delimiter=",", header=",".join(cols),
                   comments="", fmt="%.6g")
        print(f"\nwrote {args.output} ({len(rows)} bins x {len(cols)} cols)")

    if args.plot and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.plot, exist_ok=True)
        centers = np.array([0.5 * (lo + hi) for lo, hi, _, _, _ in rows])
        halfw = np.array([0.5 * (hi - lo) for lo, hi, _, _, _ in rows])
        for k, lab in enumerate(labels):
            ys = np.array([r[3][k] for r in rows], float)
            es = np.array([r[4][k] for r in rows], float)
            fig, ax = plt.subplots(figsize=(6.2, 4.2))
            ax.errorbar(centers, ys, xerr=halfw, yerr=es, fmt="o", ms=4,
                        lw=1.2, capsize=2.5, color="C0")
            ax.set_xlabel("mass")
            ax.set_ylabel("value")
            lines = [(_pretty(nm, waves) if abs(c - 1.0) < 1e-6
                      else f"{c:.3f} {_pretty(nm, waves)}")
                     for c, nm in terms[k]]
            ax.text(0.97, 0.97, "\n".join(lines),
                    transform=ax.transAxes, ha="right", va="top", fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fn = os.path.join(args.plot, f"basis_{k}.png")
            fig.savefig(fn, dpi=130)
            plt.close(fig)
            print(f"plot: {fn}  =  {lab}")

    f.free()


if __name__ == "__main__":
    main()

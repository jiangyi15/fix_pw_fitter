#!/usr/bin/env python3
"""
Plot partial-wave group contributions to mass distributions.

Each group (Rx → 3π, B → R1 R2) is plotted as a line on top of the
data points.  B0 and B0bar are combined (no flavour distinction).

Usage:
    python scripts/plot_pw_groups.py fit_results.json -o plots/
    python scripts/plot_pw_groups.py fit_results.json --max-events 5000
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ampfit import Fitter
from ampfit.amp_frac import AmplitudeFractions
from run_fit import build_constraints


def discover_groups(config):
    """Return list of (label, ck_indices) for each physical group.

    Merges B0 and B0bar (g_ls + g_lsbar blocks). Groups:
      * Rx → 3π  — resonance name (charge-conjugate merged, a1(1260)p/m separate)
      * B → R1 R2 — sorted tuple of child names
    """
    idx = 0
    chain_ranges = []
    for chain in config.full_decay.chains:
        n = len(chain.get_gls_combination())
        chain_ranges.append((idx, idx + n, chain))
        idx += n
    n_base = idx

    inner_set = set()
    for chain in config.full_decay.chains:
        inner_set.update(chain.inner)

    def _charge_stem(name):
        if name.startswith("a1(1260)"):
            return name
        if len(name) > 1 and name[-1] in ("p", "m"):
            return name[:-1]
        return name

    raw_3pi = {}
    raw_B = {}

    for start, end, chain in chain_ranges:
        d1 = chain.decays[1]
        d2 = chain.decays[2]
        d1_inner = any(o.name in inner_set for o in d1.outs)

        if d1_inner:
            res = _charge_stem(d1.core.name)
            raw_3pi.setdefault(res, []).extend(range(start, end))
        elif not d1_inner and not any(o.name in inner_set for o in d2.outs):
            r1, r2 = d1.core.name, d2.core.name
            key = tuple(sorted([r1, r2]))
            raw_B.setdefault(key, []).extend(range(start, end))

    # Expand base indices to all 8 blocks (merge B0 and B0bar together)
    def expand(base):
        ck = []
        for block in range(8):
            offset = block * n_base
            for i in base:
                ck.append(offset + i)
        return ck

    groups = {}
    for k, v in raw_3pi.items():
        groups[k] = expand(v)
    for k, v in raw_B.items():
        groups["+".join(k)] = expand(v)

    return groups


def main():
    ap = argparse.ArgumentParser(description="Plot partial-wave group distributions")
    ap.add_argument("fit_json", help="JSON from Fitter.save_params()")
    ap.add_argument("--config", default="config_amp.yml", help="Config YAML")
    ap.add_argument("--data", default="data/data_arrays.npz", help="Data NPZ")
    ap.add_argument("--phsp", default="data/phsp_arrays.npz", help="Phase-space NPZ")
    ap.add_argument("--max-events", type=int, default=None,
                    help="Limit events (faster testing)")
    ap.add_argument("--backend", default="cuda_v3", help="Compute backend")
    ap.add_argument("-o", "--output", default="plots/",
                    help="Output directory")
    ap.add_argument("--n-bins", type=int, default=100, help="Histogram bins")
    args = ap.parse_args()

    # ── Setup fitter ──────────────────────────────────────────────
    f = Fitter(args.config, backend=args.backend)

    constraints_path = os.path.splitext(args.fit_json)[0] + "_constraints.json"
    if os.path.exists(constraints_path):
        f.load_constraints(constraints_path)
    else:
        fs, sp, sc = build_constraints(f.all_comb)
        f.set_fixed(fs); f.set_same(sp); f.set_scale(sc)

    # ── Load data ─────────────────────────────────────────────────
    data_np, n_data = Fitter.load_npz(args.data, max_events=args.max_events)
    phsp_np, n_phsp = Fitter.load_npz(args.phsp, max_events=args.max_events)
    print(f"  Loaded {n_data:,} data + {n_phsp:,} phsp events")

    f.set_phsp(phsp_np)
    f.set_data(data_np)

    # ── Load fit result ───────────────────────────────────────────
    fit_result = f.load_results(args.fit_json)
    if fit_result.x is None or len(fit_result.x) == 0:
        print("ERROR: could not reconstruct x from", args.fit_json)
        sys.exit(1)
    x0 = fit_result.x

    # ── Build groups ──────────────────────────────────────────────
    groups = discover_groups(f.config)
    labels = sorted(groups.keys())

    # ── Compute norm and P values ─────────────────────────────────
    params, _, _, _ = f._build_params(x0)
    norm, _ = f._compute_norm_batched(params)
    norm = float(norm)
    print(f"  Norm = {norm:.4f}")

    # Total P (full model)
    _, _, P_total = f.backend.compute(params, f._phsp_holder, norm=norm)

    # P per group (masked ck, B0+B0bar combined)
    P_groups = []
    for label in labels:
        p_group = dict(params)
        p_group["ck"] = params["ck"].copy()
        mask = groups[label]
        for i in range(len(p_group["ck"])):
            if i not in mask:
                p_group["ck"][i] = 0.0j
        _, _, Pg = f.backend.compute(p_group, f._phsp_holder, norm=None)
        P_groups.append(Pg)

    # ── Normalisation ─────────────────────────────────────────────
    purity = f._purity if f._purity is not None else 1.0
    dw = data_np["weight"]
    pw = phsp_np["weight"]
    data_total = float(np.sum(dw))
    scale = data_total * purity / norm

    # ── Mass histogram (first column only for simplicity) ─────────
    import matplotlib.pyplot as plt
    os.makedirs(args.output, exist_ok=True)

    mass_data = data_np["mass"][:, 0]
    mass_phsp = phsp_np["mass"][:, 0]

    lo, hi = 0.2, 5.2
    bins = np.linspace(lo, hi, args.n_bins + 1)
    bin_c = (bins[:-1] + bins[1:]) / 2

    # Data
    data_y, _ = np.histogram(mass_data, bins=bins, weights=dw)
    data_w2, _ = np.histogram(mass_data, bins=bins, weights=dw ** 2)

    # Total fit
    total_y, _ = np.histogram(mass_phsp, bins=bins, weights=pw * P_total * scale)

    # Per group
    group_ys = []
    for Pg in P_groups:
        gy, _ = np.histogram(mass_phsp, bins=bins, weights=pw * Pg * scale)
        group_ys.append(gy)

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data
    ax.errorbar(bin_c, data_y, yerr=np.sqrt(data_w2), fmt='o',
                color='k', label='data', markersize=3, capsize=2)

    # Total fit
    ax.plot(bin_c, total_y, '-', color='grey', linewidth=2, label='total fit')

    # Groups
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
    for label, gy, c in zip(labels, group_ys, colors):
        ax.plot(bin_c, gy, '-', color=c, linewidth=1.5, label=label)

    ax.set_xlabel("mass [GeV]")
    ax.set_ylabel("Events")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(lo, hi)

    path = os.path.join(args.output, "pw_groups.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


if __name__ == "__main__":
    main()

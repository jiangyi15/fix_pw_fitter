#!/usr/bin/env python3
"""
Plot partial-wave contributions grouped by (L, S) coupling.

Groups waves by the orbital angular momentum L and total spin S of
the first resonance decay in each chain (e.g., rhoA→pip1.pim1).
B0 and B0bar are combined.

Usage:
    python scripts/plot_ls_groups.py fit_results.json -o plots/
"""

import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ampfit import Fitter
from ampfit.plot_pw_groups import PWGroupPlotter, plot_samesign


def discover_ls_groups(config):
    """Return dict mapping topo + full LS → CK indices.

    For each chain, groups by the full LS tuple across ALL decays:
        (L1,S1)-(L2,S2)-(L3,S3)
    with a topology prefix:
        R.pi — decays[1] decays to a resonance + π (3π topology)
        R.R  — decays[1] decays directly to two pions (dipion)
    """
    # First pass: compute n_base and collect chain info
    chain_info = []
    n_base = 0
    pip_names = {"pip1", "pim1", "pip2", "pim2"}
    for chain in config.full_decay.chains:
        ls_lists = [d.get_ls_list() for d in chain.decays]
        n_entries = len(chain.get_gls_combination())
        # Topology from the B decay (decays[0]):
        # R.pi if one B daughter is a bare pion, R.R if both are resonances
        d0_outs = {o.name for o in chain.decays[0].outs}
        topo = "R.R" if not any(n in pip_names for n in d0_outs) else "R.pi"
        chain_info.append((n_base, n_entries, ls_lists, topo))
        n_base += n_entries

    # Second pass: build groups
    groups = {}
    for base_start, n_entries, ls_lists, topo in chain_info:
        stride2 = len(ls_lists[2]) if ls_lists[2] else 1
        stride1 = len(ls_lists[1]) * stride2 if ls_lists[1] else stride2

        for i0, (l0, s0) in enumerate(ls_lists[0] if ls_lists[0] else [(0,0)]):
            for i1, (l1, s1) in enumerate(ls_lists[1] if ls_lists[1] else [(0,0)]):
                for i2, (l2, s2) in enumerate(ls_lists[2] if ls_lists[2] else [(0,0)]):
                    base_i = base_start + i0 * stride1 + i1 * stride2 + i2
                    ck_indices = [block * n_base + base_i for block in range(8)]
                    label = f"{topo} ({l0},{s0})-({l1},{s1})-({l2},{s2})"
                    groups.setdefault(label, []).extend(ck_indices)

    return groups


def main():
    ap = argparse.ArgumentParser(
        description="Plot partial-wave contributions grouped by (L, S) coupling")
    ap.add_argument("fit_json")
    ap.add_argument("--config", default="config_amp.yml")
    ap.add_argument("--data", default="data/data_arrays.npz")
    ap.add_argument("--phsp", default="data/phsp_arrays.npz")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--backend", default="cuda_v3")
    ap.add_argument("-o", "--output", default="plots/")
    ap.add_argument("--format", default="png",
                    help="Image format (default: png)")
    args = ap.parse_args()

    out_fmt = args.format

    f = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.fit_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)

    # Load data
    if os.path.exists(args.data) and os.path.exists(args.phsp):
        data_np, nd = Fitter.load_npz(args.data, max_events=args.max_events)
        phsp_np, np_ = Fitter.load_npz(args.phsp, max_events=args.max_events)
        print(f"  Loaded {nd:,} data + {np_:,} phsp events")
        f.set_phsp(phsp_np)
        f.set_data(data_np)
    else:
        f.load_all_data()

    r = f.load_results(args.fit_json)
    if r.x is None or len(r.x) == 0:
        sys.exit(1)

    groups = discover_ls_groups(f.config)
    plotter = PWGroupPlotter(f, r, groups).compute()
    print(f"  {len(plotter.labels)} (L,S) groups, scale={plotter._scale:.4f}")

    os.makedirs(args.output, exist_ok=True)

    # ── Mass ──────────────────────────────────────────────────────
    nm = f._data_np["mass"].shape[1] // 8
    plotter.plot_var(
        lambda x: [x["mass"][:, i] for i in range(nm)],
        [f"mass[{i}]" for i in range(nm)],
         0.2, 5.2, 0.05, "mass", output=args.output, fmt=out_fmt, smooth_sigma=1.0)

    # ── Angles ────────────────────────────────────────────────────
    def angle_var(x):
        a = x["angle"].reshape(x["angle"].shape[0], -1, 3)
        out = []
        for pos in range(3):
            out.append((a[:, pos, 0] + np.pi) % (2 * np.pi) - np.pi)
            out.append(np.cos(a[:, pos, 1]))
            out.append(np.cos(a[:, pos, 2]))
        return out
    ar = [(-np.pi, np.pi), (-1, 1), (-1, 1)] * 3
    al = [f"angle[{p},{c}]" for p in range(3) for c in range(3)]
    plotter.plot_var(angle_var, al, 0, 1, 0.1, "angles",
                     ranges=ar, output=args.output, fmt=out_fmt, unit="")

    # Diff-histogram: hist(cos θ₁) − hist(cos θ₂) for rows 1 and 2
    def diff_cos_theta(x, row):
        a = x["angle"].reshape(x["angle"].shape[0], -1, 3)
        return [np.cos(a[:, row, 1]), np.cos(a[:, row, 2])]
    for row in (1, 2):
        plotter.plot_stacked_perm(
            lambda x, r=row: diff_cos_theta(x, r),
            rf"$\cos\theta_1 - \cos\theta_2$ (row {row})",
            -1, 1, 0.05, f"cos_theta_diff_row{row}",
            output=args.output, fmt=out_fmt, scales=[1, -1],
            smooth_sigma=1.0, show_pull=True, legend=True)

    # Same-charge-pair variables: B → (π⁺π⁺)(π⁻π⁻) — one figure each
    plot_samesign(plotter, output=args.output, fmt=out_fmt)

    # ── Time ──────────────────────────────────────────────────────
    plotter.plot_var(lambda x: [x["time"]], ["time"], 0, 10, 0.2, "time",
                     output=args.output, fmt=out_fmt, unit="ps", legend=True, show_pull=True)

    # ── Stacked permutation plots ─────────────────────────────────
    def _mass_idx(x, cols):
        return [x["mass"].reshape(-1, 24, 2)[:, i, 0] for i in cols]

    plotter.plot_stacked_perm(
        lambda x: _mass_idx(x, [0, 3, 6, 9]),
        "m(π⁺π⁻)", 0.2, 5.2, 0.05, "m_pipi", output=args.output, fmt=out_fmt,
        smooth_sigma=1.0, show_pull=True, legend=True)
    plotter.plot_stacked_perm(
        lambda x: _mass_idx(x, [1, 4]),
        "m(π⁺π⁺π⁻)", 0.2, 5.2, 0.05, "m_pipipip", output=args.output, fmt=out_fmt,
        smooth_sigma=1.0, show_pull=True)
    plotter.plot_stacked_perm(
        lambda x: _mass_idx(x, [2, 8]),
        "m(π⁺π⁻π⁻)", 0.2, 5.2, 0.05, "m_pipipim", output=args.output, fmt=out_fmt,
        smooth_sigma=1.0, show_pull=True)

    # Sorted ππ
    def _sorted_pipi(x):
        m = x["mass"].reshape(-1, 24, 2)
        a = np.column_stack([m[:, 0, 0], m[:, 9, 0]])
        b = np.column_stack([m[:, 3, 0], m[:, 6, 0]])
        a_min = a.min(1); a_max = a.max(1)
        b_min = b.min(1); b_max = b.max(1)
        mask = a_min < b_min
        out = np.zeros((len(m), 4))
        out[mask,0]=a_min[mask]; out[mask,1]=a_max[mask]
        out[mask,2]=b_min[mask]; out[mask,3]=b_max[mask]
        out[~mask,0]=b_min[~mask]; out[~mask,1]=b_max[~mask]
        out[~mask,2]=a_min[~mask]; out[~mask,3]=a_max[~mask]
        return [out[:, i] for i in range(4)]

    _ranges = [(0.2, 1.5, 0.015), (0.2, 5.2, 0.05), (0.2, 3.0, 0.03), (0.2, 5.0, 0.05)]
    _xlabels = [
        r"$m(\pi\pi)^{\rm min}_{\rm low}$",
        r"$m(\pi\pi)^{\rm max}_{\rm low}$",
        r"$m(\pi\pi)^{\rm min}_{\rm high}$",
        r"$m(\pi\pi)^{\rm max}_{\rm high}$",
    ]
    for i in range(4):
        lo, hi, bw = _ranges[i]
        plotter.plot_var(
            lambda x, idx=i: [_sorted_pipi(x)[idx]],
            [_xlabels[i]], lo, hi, bw,
            f"m_pipi_sorted_{['pp1_min','pp1_max','pp2_min','pp2_max'][i]}",
            output=args.output, fmt=out_fmt, smooth_sigma=1.0, legend=(i in (0, 2)), show_pull=True)

    # Sorted pipipip/pipipim
    def _sorted_pair(x, idx_a, idx_b):
        m = x["mass"].reshape(-1, 24, 2)
        a, b = m[:, idx_a, 0], m[:, idx_b, 0]
        lo = np.minimum(a, b); hi = np.maximum(a, b)
        return [lo, hi]

    for prefix, idx_a, idx_b, r_min, r_max, xl_min, xl_max in [
            ("m_pipipip_sorted", 1, 4, (0.2, 5.2), (1.4, 5.2),
             r"$m(\pi^+\pi^+\pi^-)^{\rm min}$", r"$m(\pi^+\pi^+\pi^-)^{\rm max}$"),
            ("m_pipipim_sorted", 2, 8, (0.2, 5.2), (1.4, 5.2),
             r"$m(\pi^+\pi^-\pi^-)^{\rm min}$", r"$m(\pi^+\pi^-\pi^-)^{\rm max}$")]:
        for j, (rj, label_j) in enumerate([(r_min, xl_min), (r_max, xl_max)]):
            plotter.plot_var(
                lambda x, a=idx_a, b=idx_b, jj=j: [_sorted_pair(x, a, b)[jj]],
                [label_j], rj[0], rj[1], 0.05,
                f"{prefix}_{['min','max'][j]}", output=args.output, fmt=out_fmt,
                smooth_sigma=1.0, show_pull=True, legend=True)

    # pipipip - pipipim difference
    def diff_pipipi(x):
        m = x["mass"].reshape(-1, 24, 2)
        return [m[:, 1, 0], m[:, 4, 0], m[:, 2, 0], m[:, 8, 0]]

    plotter.plot_stacked_perm(
        diff_pipipi, r"$m(3\pi)$",
        0.2, 5.2, 0.05, "m_pipipi_diff", output=args.output, fmt=out_fmt,
        scales=[1, 1, -1, -1], smooth_sigma=1.0, show_pull=True)


if __name__ == "__main__":
    main()

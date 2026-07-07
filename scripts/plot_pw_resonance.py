#!/usr/bin/env python3
"""
Plot partial-wave groups, separating 3π mode and d1·d2 mode.

In each mode, groups are merged by the resonance that directly
produces [pip1, pim1] (π⁺π⁻).

3π mode (B → R·π, R → resonance·π, resonance → ππ):
  All chains with the same pipi resonance merged across parent R:
    - rhoA:   a1→ρπ, a2→ρπ, π1→ρπ, π2→ρπ, π1300→ρπ, π1600→ρπ
    - f0(500):  a1→f0π, π1300→f0π, π1600→f0π, π2→f0π
    - f0(980):  same
    - f2(1270): a1→f2π, a2→f2π, π2→f2π

d1·d2 mode (B → d1·d2, both → ππ):
  Groups by d1+d2 pair.

Usage:
    python scripts/plot_pw_resonance.py fit_results.json -o plots/
"""

import sys, os, argparse
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ampfit import Fitter
from ampfit.plot_pw_groups import PWGroupPlotter
from run_fit import build_constraints


def idx_total(config):
    """Total CK indices per CP (= n_wave / 2)."""
    return sum(len(chain.get_gls_combination()) for chain in config.full_decay.chains)


def get_groups(config):
    """Return dict mapping label → CK indices for 3π and d1·d2 modes."""
    inner_set = set()
    for chain in config.full_decay.chains:
        inner_set.update(chain.inner)

    n_base = idx_total(config)
    groups_3pi = defaultdict(list)
    groups_d1d2 = defaultdict(list)

    ck_start = 0
    for chain in config.full_decay.chains:
        d1 = chain.decays[1]
        d2 = chain.decays[2]
        n = len(chain.get_gls_combination())
        d1_inner = any(o.name in inner_set for o in d1.outs)

        pipi_res = "other"
        for d in chain.decays:
            outs = [o.name for o in d.outs]
            if "pip1" in outs and "pim1" in outs:
                pipi_res = d.core.name
                break

        ck_all = []
        for block in range(8):
            offset = block * n_base
            ck_all.extend(range(offset + ck_start, offset + ck_start + n))

        if d1_inner:
            groups_3pi[pipi_res].extend(ck_all)
        else:
            label = f"{d1.core.name}+{d2.core.name}"
            groups_d1d2[label].extend(ck_all)
        ck_start += n

    groups = {}
    for k, v in groups_3pi.items():
        groups[k] = sorted(set(v))
    for k, v in groups_d1d2.items():
        groups[k] = sorted(set(v))
    return groups


def _sorted_pipi(x):
    """Sorted ππ: groups [0,9] and [3,6], each sorted within, then by group min."""
    m = x["mass"].reshape(-1, 24, 2)
    a = np.column_stack([m[:, 0, 0], m[:, 9, 0]])
    b = np.column_stack([m[:, 3, 0], m[:, 6, 0]])
    a_min = a.min(1); a_max = a.max(1)
    b_min = b.min(1); b_max = b.max(1)
    mask = a_min < b_min
    out = np.zeros((len(m), 4))
    out[mask, 0] = a_min[mask]; out[mask, 1] = a_max[mask]
    out[mask, 2] = b_min[mask]; out[mask, 3] = b_max[mask]
    out[~mask, 0] = b_min[~mask]; out[~mask, 1] = b_max[~mask]
    out[~mask, 2] = a_min[~mask]; out[~mask, 3] = a_max[~mask]
    return [out[:, i] for i in range(4)]


def _sorted_pair(x, idx_a, idx_b):
    """Two-perm sorted pair: min, max."""
    m = x["mass"].reshape(-1, 24, 2)
    a, b = m[:, idx_a, 0], m[:, idx_b, 0]
    lo = np.minimum(a, b); hi = np.maximum(a, b)
    return [lo, hi]


def main():
    ap = argparse.ArgumentParser(
        description="Plot PW groups separated by 3π / d1·d2 mode")
    ap.add_argument("fit_json")
    ap.add_argument("--config", default="config_amp.yml")
    ap.add_argument("--data", default="data/data_arrays.npz")
    ap.add_argument("--phsp", default="data/phsp_arrays.npz")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--backend", default="cuda_v3")
    ap.add_argument("-o", "--output", default="plots/")
    ap.add_argument("--format", default="png", choices=["png", "pdf"],
                    help="output format (png or pdf)")
    args = ap.parse_args()
    ft = args.format

    f = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.fit_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    else:
        fs, sp, sc = build_constraints(f.all_comb)
        f.set_fixed(fs); f.set_same(sp); f.set_scale(sc)

    data_np, nd = Fitter.load_npz(args.data, max_events=args.max_events)
    phsp_np, np_ = Fitter.load_npz(args.phsp, max_events=args.max_events)
    print(f"  Loaded {nd:,} data + {np_:,} phsp events")
    f.set_phsp(phsp_np)
    f.set_data(data_np)

    r = f.load_results(args.fit_json)
    if r.x is None or len(r.x) == 0:
        sys.exit(1)

    groups = get_groups(f.config)
    groups = {k: v for k, v in groups.items() if len(v) >= 8}
    if not groups:
        print("No groups found")
        sys.exit(1)

    plotter = PWGroupPlotter(f, r, groups).compute()
    print(f"  {len(plotter.labels)} groups: {plotter.labels}")

    os.makedirs(args.output, exist_ok=True)

    # ── Mass overview ───────────────────────────────────────────
    nm = f._data_np["mass"].shape[1] // 8
    plotter.plot_var(
        lambda x: [x["mass"][:, i] for i in range(nm)],
        [f"mass[{i}]" for i in range(nm)],
        0.2, 5.2, 0.05, "mass", output=args.output, smooth_sigma=1.0,
        fmt=ft)

    # ── Angles ──────────────────────────────────────────────────
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
                     ranges=ar, output=args.output, unit="", fmt=ft)

    # ── π⁺π⁻ mass (columns 0,3,6,9) ─────────────────────────────
    def _pipi_mass(x):
        m = x["mass"].reshape(-1, 24, 2)
        return [m[:, i, 0] for i in [0, 3, 6, 9]]
    plotter.plot_stacked_perm(
        _pipi_mass, r"$m(\pi^+\pi^-)$", 0.2, 5.2, 0.05,
        "m_pipi", output=args.output,
        smooth_sigma=1.0, show_pull=True, legend=True, fmt=ft)

    # ── π⁺π⁺π⁻ mass ────────────────────────────────────────────
    def _pipipip_mass(x):
        m = x["mass"].reshape(-1, 24, 2)
        return [m[:, i, 0] for i in [1, 4, 7, 10]]
    plotter.plot_stacked_perm(
        _pipipip_mass, r"$m(\pi^+\pi^+\pi^-)$", 0.2, 5.2, 0.05,
        "m_pipipip", output=args.output,
        smooth_sigma=1.0, show_pull=True, fmt=ft)

    # ── π⁺π⁻π⁻ mass ────────────────────────────────────────────
    def _pipipim_mass(x):
        m = x["mass"].reshape(-1, 24, 2)
        return [m[:, i, 0] for i in [2, 5, 8, 11]]
    plotter.plot_stacked_perm(
        _pipipim_mass, r"$m(\pi^+\pi^-\pi^-)$", 0.2, 5.2, 0.05,
        "m_pipipim", output=args.output,
        smooth_sigma=1.0, show_pull=True, fmt=ft)

    # ── Sorted ππ ───────────────────────────────────────────────
    _sorted = _sorted_pipi(f._data_np)
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
            output=args.output, smooth_sigma=1.0, legend=(i == 0), fmt=ft, show_pull=True)

    # ── Sorted pipipip and pipipim ───────────────────────────────
    for prefix, idx_a, idx_b, r_min, r_max, xl_min, xl_max in [
            ("m_pipipip_sorted", 1, 4, (0.2, 5.2), (1.4, 5.2),
             r"$m(\pi^+\pi^+\pi^-)^{\rm min}$", r"$m(\pi^+\pi^+\pi^-)^{\rm max}$"),
            ("m_pipipim_sorted", 2, 8, (0.2, 5.2), (1.4, 5.2),
             r"$m(\pi^+\pi^-\pi^-)^{\rm min}$", r"$m(\pi^+\pi^-\pi^-)^{\rm max}$")]:
        for j, (rj, label_j) in enumerate([(r_min, xl_min), (r_max, xl_max)]):
            plotter.plot_var(
                lambda x, a=idx_a, b=idx_b, jj=j: [_sorted_pair(x, a, b)[jj]],
                [label_j], rj[0], rj[1], 0.05,
                f"{prefix}_{['min','max'][j]}", output=args.output,
                smooth_sigma=1.0, show_pull=True, fmt=ft)

    # ── Difference pipipi (B0 - B0bar) ───────────────────────────
    def diff_pipipi(x):
        m = x["mass"].reshape(-1, 24, 2)
        return [m[:, 1, 0], m[:, 4, 0], m[:, 2, 0], m[:, 8, 0]]
    plotter.plot_stacked_perm(
        diff_pipipi,
        r"$m(3\pi)$",
        0.2, 5.2, 0.05, "m_pipipi_diff", output=args.output,
        scales=[1, 1, -1, -1], smooth_sigma=1.0, show_pull=True, fmt=ft)

    # ── Time ────────────────────────────────────────────────────
    plotter.plot_var(lambda x: [x["time"]], ["time"], 0, 10, 0.2,
                     "time", output=args.output, unit="ps",
                     legend=True, show_pull=True, fmt=ft)


if __name__ == "__main__":
    main()

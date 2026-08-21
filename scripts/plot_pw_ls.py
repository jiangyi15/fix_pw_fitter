#!/usr/bin/env python3
"""
Plot partial-wave contributions grouped by (L, S) coupling.

Groups waves by the orbital angular momentum L and total spin S of
the first resonance decay in each chain (e.g., rhoA→pip1.pim1).
B0 and B0bar are combined.

Usage:
    python scripts/plot_pw_ls.py fit_results.json -o plots/
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ampfit.plot_pw_common import run


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

        for i0, (l0, s0) in enumerate(ls_lists[0] if ls_lists[0] else [(0, 0)]):
            for i1, (l1, s1) in enumerate(ls_lists[1] if ls_lists[1] else [(0, 0)]):
                for i2, (l2, s2) in enumerate(ls_lists[2] if ls_lists[2] else [(0, 0)]):
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

    run(args.fit_json, args.config, args.data, args.phsp,
        args.max_events, args.backend, args.output, args.format,
        discover_ls_groups, "(L,S) groups")


if __name__ == "__main__":
    main()

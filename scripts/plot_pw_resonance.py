#!/usr/bin/env python3
"""
Plot PW groups separated by 3π / d1·d2 mode.

Each chain is grouped either by its 3π inner resonance (R → πππ) or by
the two direct di-pion resonances (d1·d2).  B0 and B0bar are combined.

Usage:
    python scripts/plot_pw_resonance.py fit_results.json -o plots/
"""

import sys, os, argparse
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ampfit.plot_pw_common import run


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
                pipi_res = d.core.display
                break

        ck_all = []
        for block in range(8):
            offset = block * n_base
            ck_all.extend(range(offset + ck_start, offset + ck_start + n))

        if d1_inner:
            groups_3pi[pipi_res].extend(ck_all)
        else:
            label = f"{d1.core.display}+{d2.core.display}"
            groups_d1d2[label].extend(ck_all)
        ck_start += n

    groups = {}
    for k, v in groups_3pi.items():
        groups[k] = sorted(set(v))
    for k, v in groups_d1d2.items():
        groups[k] = sorted(set(v))
    return groups


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

    def groups(config):
        g = get_groups(config)
        return {k: v for k, v in g.items() if len(v) >= 8}

    run(args.fit_json, args.config, args.data, args.phsp,
        args.max_events, args.backend, args.output, args.format,
        groups, "resonance groups")


if __name__ == "__main__":
    main()

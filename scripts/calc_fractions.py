#!/usr/bin/env python3
"""
Compute amplitude fractions from a fit result and save to CSV.

Two separate categories (no cross-talk):

  * **Rx → 3π**: ``Rx → Ry + π, Ry → π + π``
  * **B → R1 R2**: ``B → R1 + R2, R1 → ππ, R2 → ππ``

Each entry reports the B0 (g_ls) and B0bar (g_lsbar) fraction side by side.

Usage:
    python scripts/calc_fractions.py fit_results.json -o fractions.csv
    python scripts/calc_fractions.py fit_results.json --max-events 5000
"""

import sys, os, argparse, csv
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ampfit import Fitter
from ampfit.amp_frac import AmplitudeFractions
from run_fit import build_constraints, load_npz


def _charge_stem(name):
    """Strip trailing ``p``/``m`` charge suffix to merge charge-conjugate pairs.

    ``a1(1260)`` is kept separate (different dynamics for p vs m).
    """
    if name.startswith("a1(1260)"):
        return name
    if len(name) > 1 and name[-1] in ("p", "m"):
        return name[:-1]
    return name


def discover_groups(config):
    """Return list of ``(label, ls_mask, lsbar_mask)`` for each category.

    Two lists returned: ``groups_3pi`` and ``groups_B_R1R2``.
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

    raw_3pi = {}
    raw_B = {}

    for start, end, chain in chain_ranges:
        d1 = chain.decays[1]
        d2 = chain.decays[2]

        d1_child_inner = any(o.name in inner_set for o in d1.outs)
        d2_child_inner = any(o.name in inner_set for o in d2.outs)

        if d1_child_inner:
            res = _charge_stem(d1.core.name)
            raw_3pi.setdefault(res, []).extend(range(start, end))
        elif not d1_child_inner and not d2_child_inner:
            r1 = d1.core.name
            r2 = d2.core.name
            key = tuple(sorted([r1, r2]))
            raw_B.setdefault(key, []).extend(range(start, end))

    def split_ls(base_idxs):
        ls, lsbar = [], []
        for i in base_idxs:
            for block in range(4):
                ls.append(block * n_base + i)
                lsbar.append((block + 4) * n_base + i)
        return ls, lsbar

    groups_3pi = []
    for k in sorted(raw_3pi):
        ls, lsbar = split_ls(raw_3pi[k])
        groups_3pi.append((k, ls, lsbar))

    groups_B = []
    for k in sorted(raw_B):
        ls, lsbar = split_ls(raw_B[k])
        label = "+".join(k)
        groups_B.append((label, ls, lsbar))

    return groups_3pi, groups_B


def compute_section(af, groups, title):
    """Compute fractions for a list of (label, ls_mask, lsbar_mask) tuples.

    Returns list of ``(label, v_ls, e_ls, v_lsbar, e_lsbar)``.
    """
    if not groups:
        return []

    # Single batch call: interleave ls and lsbar masks
    all_masks = []
    for _, ls, lsbar in groups:
        all_masks.append(ls)
        all_masks.append(lsbar)
    vals, errs = af.fractions(all_masks)

    rows = []
    for i, (label, _, _) in enumerate(groups):
        v_ls, v_lsbar = vals[2 * i], vals[2 * i + 1]
        e_ls, e_lsbar = errs[2 * i], errs[2 * i + 1]
        rows.append((label, v_ls, e_ls, v_lsbar, e_lsbar))

    # Print
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print(f"  {'Label':30s}  {'B0':>22s}  {'B0bar':>22s}")
    print("  " + "-" * 75)
    for label, v0, e0, v1, e1 in rows:
        print(f"  {label:30s}  {v0:10.6f} ± {e0:10.6f}  {v1:10.6f} ± {e1:10.6f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Amplitude fractions from fit result")
    ap.add_argument("fit_json", help="JSON from Fitter.save_params()")
    ap.add_argument("--config", default="config_amp.yml", help="Config YAML")
    ap.add_argument("--phsp", default="data/phsp_arrays.npz", help="Phase-space NPZ")
    ap.add_argument("--max-events", type=int, default=None,
                    help="Limit phsp events (faster testing)")
    ap.add_argument("--backend", default="cuda_v3", help="Compute backend")
    ap.add_argument("-o", "--output", help="CSV output path")
    args = ap.parse_args()

    # ── Setup fitter ──────────────────────────────────────────────
    f = Fitter(args.config, backend=args.backend)
    fs, sp, sc = build_constraints(f.all_comb)
    f.set_fixed(fs)
    f.set_same(sp)
    f.set_scale(sc)

    phsp, _ = load_npz(args.phsp, max_events=args.max_events)
    f.set_phsp(phsp)

    # ── Load fit result (auto-detects _error_matrix.npy) ──────────
    fit_result = f.load_results(args.fit_json)
    if fit_result.x is None or len(fit_result.x) == 0:
        print("ERROR: could not reconstruct x from", args.fit_json)
        sys.exit(1)

    # ── Build groups ──────────────────────────────────────────────
    groups_3pi, groups_B = discover_groups(f.config)

    # ── AmplitudeFractions ────────────────────────────────────────
    fit_ns = SimpleNamespace(x=fit_result.x,
                              hess_inv=getattr(fit_result, 'hess_inv', None))
    af = AmplitudeFractions(f, fit_ns)

    # ── Compute ───────────────────────────────────────────────────
    rows_3pi = compute_section(af, groups_3pi,
                               "Resonance → 3π  (Rx → Ry+π, Ry → ππ)")
    if groups_B:
        print()
    rows_B = compute_section(af, groups_B,
                             "B → R1 R2  (R1, R2 → ππ)")

    # ── CSV output ────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w", newline="") as fout:
            w = csv.writer(fout)
            w.writerow(["Category", "Label",
                        "B0_value", "B0_error",
                        "B0bar_value", "B0bar_error"])
            for row in rows_3pi:
                w.writerow(["Rx→3π", *row])
            for row in rows_B:
                w.writerow(["B→R1R2", *row])
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

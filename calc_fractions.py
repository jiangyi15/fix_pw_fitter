#!/usr/bin/env python3
"""
Compute amplitude fractions from a fit result.

Groups:
  * B → R1 R2 / B → R π  — fraction by B-meson decay children
  * Rx → intermediate     — fraction by the probe resonance (decays[1].core)

Usage:
    python calc_fractions.py fit_results.json --hessian hessian.npy
    python calc_fractions.py fit_results.json --hessian hessian.npy --max-events 5000
"""

import sys, os, argparse
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ampfit import Fitter
from ampfit.amp_frac import AmplitudeFractions
from run_fit import build_constraints, load_npz


def discover_groups(config):
    """Return (b_groups, res_groups) dicts mapping labels → ck indices.

    *b_groups*  — sorted tuple of B-decay children → list of ck indices.
    *res_groups* — probe resonance name → list of ck indices.
    """
    chain_ranges = []
    idx = 0
    for chain in config.full_decay.chains:
        n = len(chain.get_gls_combination())
        chain_ranges.append((idx, idx + n, chain))
        idx += n
    n_base = idx

    def base_to_ck(base_set):
        out = []
        for block in range(8):
            offset = block * n_base
            for i in sorted(base_set):
                out.append(offset + i)
        return out

    # --- B → R1 R2 groups ---
    b_groups = {}   # sorted-tuple of child names → [base indices]
    for start, end, chain in chain_ranges:
        b_outs = tuple(sorted(o.name for o in chain.decays[0].outs))
        b_groups.setdefault(b_outs, []).extend(range(start, end))

    # --- probe resonance groups (decays[1].core) ---
    res_groups = {}  # resonance name → [base indices]
    for start, end, chain in chain_ranges:
        res = chain.decays[1].core.name
        res_groups.setdefault(res, []).extend(range(start, end))

    # Convert base → ck for all
    b_ck = {k: base_to_ck(v) for k, v in b_groups.items()}
    res_ck = {k: base_to_ck(v) for k, v in res_groups.items()}

    return b_ck, res_ck


def main():
    ap = argparse.ArgumentParser(description="Amplitude fractions from fit result")
    ap.add_argument("fit_json", help="JSON from Fitter.save_params()")
    ap.add_argument("--hessian", help="Hessian .npy from Fitter.save_hessian()")
    ap.add_argument("--config", default="config_amp.yml", help="Config YAML")
    ap.add_argument("--phsp", default="data/phsp_arrays.npz", help="Phase-space NPZ")
    ap.add_argument("--max-events", type=int, default=None,
                    help="Limit phsp events (faster testing)")
    ap.add_argument("--backend", default="numpy", help="Compute backend")
    args = ap.parse_args()

    # ── Setup fitter ──────────────────────────────────────────────
    f = Fitter(args.config, backend=args.backend)
    fs, sp, sc = build_constraints(f.all_comb)
    f.set_fixed(fs)
    f.set_same(sp)
    f.set_scale(sc)

    phsp, _ = load_npz(args.phsp, max_events=args.max_events)
    f.set_phsp(phsp)

    # ── Load fit result ───────────────────────────────────────────
    fit_result = f.load_results(args.fit_json, args.hessian)
    if fit_result.x is None or len(fit_result.x) == 0:
        print("ERROR: could not reconstruct x from", args.fit_json)
        sys.exit(1)
    # ── Build groups ──────────────────────────────────────────────
    b_groups, res_groups = discover_groups(f.config)

    # ── AmplitudeFractions ────────────────────────────────────────
    fit_ns = SimpleNamespace(x=fit_result.x, hess_inv=fit_result.hess_inv)
    af = AmplitudeFractions(f, fit_ns)

    # ── Compute B-group fractions ─────────────────────────────────
    print("=" * 60)
    print("  B → R1 R2  /  B → R  +  π   amplitude fractions")
    print("=" * 60)
    b_masks = list(b_groups.values())
    b_labels = ["B→" + "+".join(k) for k in b_groups.keys()]
    if b_masks:
        vals, errs = af.fractions(b_masks)
        for label, v, e in zip(b_labels, vals, errs):
            print(f"  {label:35s}  {v:10.6f} ± {e:10.6f}")

    # ── Compute resonance-group fractions ─────────────────────────
    print()
    print("=" * 60)
    print("  Resonance (decays[1].core) amplitude fractions")
    print("=" * 60)
    res_masks = list(res_groups.values())
    res_labels = list(res_groups.keys())
    if res_masks:
        vals, errs = af.fractions(res_masks)
        for label, v, e in zip(res_labels, vals, errs):
            print(f"  {label:25s}  {v:10.6f} ± {e:10.6f}")

if __name__ == "__main__":
    main()

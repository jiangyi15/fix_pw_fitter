#!/usr/bin/env python3
"""
Compute sub-decay branching ratios for 3π resonances.

For each resonance Rx that decays to 3π (e.g. a₁, a₂, π₁, π₂),
compute the ratio of each sub-channel (e.g. Rx → ρπ, Rx → f₀π)
to the total Rx → 3π amplitude.

These are the sub-decay branching fractions within the model:
  R = Σ w·|A(Rx → Xπ)|² / Σ w·|A(Rx → all 3π)|²

Usage:
    python scripts/calc_sub_decay_ratios.py fit_results.json
    python scripts/calc_sub_decay_ratios.py fit_results.json -o ratios.csv
"""

import sys, os, argparse, csv, re, subprocess
from collections import OrderedDict

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPT_DIR)


def _default_path(name):
    return os.path.join(_SCRIPT_DIR, name)


def _cl_name(stem):
    """Human-readable short resonance label."""
    nice = {
        "rhoA": "ρπ", "rhoB": "ρπ",
        "f0(980)": "f₀(980)π", "f0(500)": "f₀(500)π",
        "f2(1270)": "f₂(1270)π",
    }
    return nice.get(stem, stem)


def _latex_name(name):
    """Convert Unicode label to LaTeX math mode."""
    map_ = {
        "ρπ": r"\rho\pi",
        "f₀(980)π": r"f_{0}(980)\pi",
        "f₀(500)π": r"f_{0}(500)\pi",
        "f₂(1270)π": r"f_{2}(1270)\pi",
    }
    return map_.get(name, name)


def _latex_rx(name):
    """Convert resonance name to LaTeX math mode (no charge suffix)."""
    map_ = {
        "NR0": r"{\rm NR}",
        "a1(1260)": r"a_{1}(1260)",
        "a1(1640)": r"a_{1}(1640)",
        "a2(1320)": r"a_{2}(1320)",
        "π1(1600)": r"\pi_{1}(1600)",
        "π2(1670)": r"\pi_{2}(1670)",
        "π(1300)": r"\pi(1300)",
        "π(1600)": r"\pi(1600)",
        "pi1(1600)": r"\pi_{1}(1600)",
        "pi2(1670)": r"\pi_{2}(1670)",
        "pi1300": r"\pi(1300)",
        "pi1600": r"\pi(1600)",
    }
    return map_.get(name, name)


def discover_sub_decays(config, merge_cp=False):
    """Group base ck indices by resonance → sub-decay channel.

    Only includes resonances that decay through a cascade
    (Rx → Ry + π, Ry → ππ), i.e. true 3π decays.

    Args:
        config: Config object.
        merge_cp: if True, merge charge-conjugate pairs (a1⁺/a1⁻ → a1).

    Returns list of ``(res_label, [(sub_label, [base_idxs]), ...])``.
    """
    from ampfit import Fitter
    pws = config.full_decay.get_partial_waves_params()

    chains = OrderedDict()
    for i, pw in enumerate(pws):
        name = pw[1]
        bm = re.match(r'^B->(.+)_g_ls_\d+$', name)
        if not bm:
            continue
        b_stem = bm.group(1)
        parts = b_stem.split(".")
        if len(parts) != 2:
            continue
        rx_name = parts[0]

        sub_name = pw[2] if len(pw) > 2 else ""
        sm = re.match(r'^[^>]+->(.+?)\.', sub_name)
        sub_channel = sm.group(1) if sm else ""

        if not sub_channel or sub_channel in ("pim1", "pim2", "pip1", "pip2"):
            continue

        chains.setdefault(rx_name, OrderedDict())
        chains[rx_name].setdefault(sub_channel, []).append(i)

    # Optionally merge charge-conjugate pairs
    if merge_cp:
        merged = OrderedDict()
        for rx_name, subs in chains.items():
            base = rx_name[:-1] if len(rx_name) > 1 and rx_name[-1] in "pm" else rx_name
            for sub_name, idxs in subs.items():
                merged.setdefault(base, OrderedDict())
                merged[base].setdefault(sub_name, []).extend(idxs)
        chains = merged

    # Convert to results, sorted by name
    result = []
    for rx_name in sorted(chains, key=lambda x: _cl_name(x)):
        subs = []
        for sub_name, idxs in chains[rx_name].items():
            subs.append((_cl_name(sub_name), sorted(set(idxs))))
        result.append((_cl_name(rx_name), subs))
    return result


def _setup(args):
    """Create Fitter, load data and fit result."""
    from ampfit import Fitter
    from run_fit import build_constraints

    f = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.fit_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    else:
        fs, sp, sc = build_constraints(f.all_comb)
        f.set_fixed(fs); f.set_same(sp); f.set_scale(sc)
    phsp, _ = Fitter.load_npz(args.phsp, max_events=args.max_events)
    f.set_phsp(phsp)
    return f


def _output_terminal(groups, csv_rows, f, af, n_base):
    """Terminal output (existing behavior)."""
    def split_ls(idxs):
        ls, lsbar = [], []
        for i in idxs:
            for block in range(4):
                ls.append(block * n_base + i)
                lsbar.append((block + 4) * n_base + i)
        return ls, lsbar

    print("\n" + "=" * 80)
    print("  Sub-decay ratios: Rx → Xπ / Rx → 3π")
    print("=" * 80)

    for rx_label, subs in groups:
        rx_all_idxs = sum([idxs for _, idxs in subs], [])
        ls_all, lsbar_all = split_ls(rx_all_idxs)
        v_tot, e_tot = af.fractions([ls_all, lsbar_all])
        f0_tot, f1_tot = v_tot[0], v_tot[1]
        e0_tot, e1_tot = e_tot[0], e_tot[1]

        print(f"\n  ── {rx_label} ──")
        print(f"  {'Sub-channel':30s}  {'B0 ratio':>22s}  {'B0bar ratio':>22s}")
        print("  " + "-" * 75)

        # Ratio = sub / total within B0 and B0bar separately
        # Use af.fractions with denominator for proper gradient-based uncertainty
        sub_masks_b0 = [split_ls(idxs)[0] for _, idxs in subs]
        sub_masks_b1 = [split_ls(idxs)[1] for _, idxs in subs]
        sub_names = [sub_label for sub_label, _ in subs]
        v_b0, e_b0 = af.fractions(sub_masks_b0, denominator=ls_all)
        v_b1, e_b1 = af.fractions(sub_masks_b1, denominator=lsbar_all)

        n_subs = len(sub_names)
        for j, sub_label in enumerate(sub_names):
            r0, s0 = v_b0[j], e_b0[j]
            r1, s1 = v_b1[j], e_b1[j]
            print(f"  {sub_label:30s}  {r0:8.5f} ± {s0:8.5f}  "
                  f"{r1:8.5f} ± {s1:8.5f}")
            csv_rows.append((rx_label, sub_label, r0, s0, r1, s1))


def _fmt(v, e):
    if e is None:
        return f"{v:.4f}"
    return f"{v:.4f}\\pm{max(e, 0.0):.4f}"


def _output_latex(groups, af, n_base, output_path, compile_pdf):
    """LaTeX/PDF output."""

    def split_ls(idxs):
        ls, lsbar = [], []
        for i in idxs:
            for block in range(4):
                ls.append(block * n_base + i)
                lsbar.append((block + 4) * n_base + i)
        return ls, lsbar

    L = []
    L.append(r"\documentclass[11pt,border=2pt]{standalone}")
    L.append(r"\usepackage{booktabs}")
    L.append(r"\usepackage{array}")
    L.append(r"\usepackage{multirow}")
    L.append(r"\begin{document}")
    L.append(r"\begin{tabular}{llrr}")
    L.append(r"\toprule")
    L.append(r"Resonance & Sub-channel & B$^{0}$ & $\overline{\rm B}{}^{0}$ \\")
    L.append(r"\midrule")

    # Base name without charge suffix (to group charge-conjugate pairs)
    def _base(name):
        for s in ("⁺", "⁻", "p", "m"):
            if name.endswith(s):
                return name[:-1]
        return name

    prev_base = None
    for rx_label, subs in groups:
        cur_base = _base(rx_label)
        if prev_base is not None and cur_base != prev_base:
            L.append(r"  \midrule")
        prev_base = cur_base

        rx_all_idxs = sum([idxs for _, idxs in subs], [])
        ls_all, lsbar_all = split_ls(rx_all_idxs)
        v_tot, e_tot = af.fractions([ls_all, lsbar_all])
        f0_tot, f1_tot = v_tot[0], v_tot[1]
        e0_tot, e1_tot = e_tot[0], e_tot[1]

        # Ratio = sub / total within B0 and B0bar separately (gradient-based)
        sub_masks_b0 = [split_ls(idxs)[0] for _, idxs in subs]
        sub_masks_b1 = [split_ls(idxs)[1] for _, idxs in subs]
        sub_names = [sub_label for sub_label, _ in subs]
        v_b0, e_b0 = af.fractions(sub_masks_b0, denominator=ls_all)
        v_b1, e_b1 = af.fractions(sub_masks_b1, denominator=lsbar_all)

        n_subs = len(sub_names)
        rx_ltx = _latex_rx(rx_label)

        # First sub-channel on the resonance line (multirow span)
        r0, s0 = v_b0[0], e_b0[0]
        r1, s1 = v_b1[0], e_b1[0]
        sub_ltx = _latex_name(sub_names[0])
        span = f"\\multirow{{{n_subs}}}{{*}}{{$\\ {rx_ltx}$}}"
        L.append(f"  {span} & ${sub_ltx}$ & ${_fmt(r0,s0)}$ & ${_fmt(r1,s1)}$ \\\\")

        # Remaining sub-channels (no resonance name)
        for j in range(1, n_subs):
            r0, s0 = v_b0[j], e_b0[j]
            r1, s1 = v_b1[j], e_b1[j]
            sub_ltx = _latex_name(sub_names[j])
            L.append(f"   & ${sub_ltx}$ & ${_fmt(r0,s0)}$ & ${_fmt(r1,s1)}$ \\\\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{document}")

    tex = "\n".join(L)
    if output_path:
        with open(output_path, "w") as f:
            f.write(tex)
        print(f"  Saved {output_path}")
    else:
        print(tex)

    if compile_pdf:
        outdir = os.path.dirname(output_path) if output_path else "/tmp"
        texpath = output_path if output_path else "/tmp/sub_decay.tex"
        if not output_path:
            with open(texpath, "w") as f:
                f.write(tex)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "-output-directory=" + outdir, texpath],
                           capture_output=True, text=True, timeout=60)
        pdfpath = os.path.splitext(texpath)[0] + ".pdf"
        if os.path.exists(pdfpath):
            print(f"  PDF: {pdfpath}")
        else:
            for ln in (r.stderr or "").split("\n"):
                if "Error" in ln:
                    print(f"  pdflatex: {ln}")


def main():
    ap = argparse.ArgumentParser(
        description="Sub-decay branching ratios for 3π resonances")
    ap.add_argument("fit_json")
    ap.add_argument("--config", default=_default_path("config_angle.yml"))
    ap.add_argument("--phsp", default=_default_path("data/phsp_arrays.npz"))
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--backend", default="cuda32_v3")
    ap.add_argument("-o", "--output", help="CSV output path")
    ap.add_argument("--latex", action="store_true",
                    help="Output LaTeX table instead of terminal")
    ap.add_argument("--latex-output", default=None,
                    help="LaTeX .tex path (default: stdout)")
    ap.add_argument("--pdf", action="store_true",
                    help="Compile LaTeX to PDF (requires --latex-output)")
    ap.add_argument("--no-merge-cp", action="store_true",
                    help="Show charge-conjugate pairs separately (default: merged)")
    args = ap.parse_args()

    # ── Setup ──────────────────────────────────────────────────────
    from ampfit import Fitter
    from ampfit.amp_frac import AmplitudeFractions

    f = _setup(args)
    fit_result = f.load_results(args.fit_json)
    if fit_result.x is None or len(fit_result.x) == 0:
        print("ERROR: could not reconstruct x from", args.fit_json)
        sys.exit(1)

    af = AmplitudeFractions(f, fit_result)

    # ── Discover sub-decay groups ──────────────────────────────────
    groups = discover_sub_decays(f.config, merge_cp=not args.no_merge_cp)
    n_base = len(f.config.full_decay.get_partial_waves_params())

    # ── Output ─────────────────────────────────────────────────────
    if args.latex:
        _output_latex(groups, af, n_base,
                      args.latex_output, args.pdf)
    else:
        csv_rows = []
        _output_terminal(groups, csv_rows, f, af, n_base)

        if args.output:
            with open(args.output, "w", newline="") as fout:
                w = csv.writer(fout)
                w.writerow(["Resonance", "SubChannel",
                            "B0_ratio", "B0_ratio_err",
                            "B0bar_ratio", "B0bar_ratio_err"])
                for row in csv_rows:
                    w.writerow(row)
            print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()

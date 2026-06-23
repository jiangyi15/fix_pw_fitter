#!/usr/bin/env python3
"""
Compute fit fractions grouped by first-decay topology.

Uses the ck parameter names (which encode the full decay chain) to
group ck indices by first decay.  Each entry reports B0 (g_ls) and
B0bar (g_lsbar) fractions separately.

Usage:
    python scripts/calc_first_decay_fractions.py fit_results.json
    python scripts/calc_first_decay_fractions.py fit_results.json -o table.csv
    python scripts/calc_first_decay_fractions.py fit_results.json --latex
    python scripts/calc_first_decay_fractions.py fit_results.json --latex --latex-output table.tex
"""

import sys, os, argparse, csv, re, subprocess
from collections import OrderedDict

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPT_DIR)


def _default_path(name):
    """Return absolute path relative to project root."""
    return os.path.join(_SCRIPT_DIR, name)
from ampfit import Fitter
from ampfit.amp_frac import AmplitudeFractions
from run_fit import build_constraints


def _display_name(stem):
    """Convert internal stem to human-readable first-decay label.

    Examples:
      'rhoA.rhoB'    → 'ρρ'
      'f0(980).rhoB' → 'f₀(980)ρ'
      'a1(1260)p.pim2' → 'a₁(1260)⁺π⁻'
    """
    nice = {
        "rhoA": "ρ⁰", "rhoB": "ρ⁰",
        "pip1": "π⁺", "pip2": "π⁺",
        "pim1": "π⁻", "pim2": "π⁻",
        "a1(1260)p": "a₁(1260)⁺", "a1(1260)m": "a₁(1260)⁻",
        "a1(1640)p": "a₁(1640)⁺", "a1(1640)m": "a₁(1640)⁻",
        "a2(1320)p": "a₂(1320)⁺", "a2(1320)m": "a₂(1320)⁻",
        "pi1(1600)p": "π₁(1600)⁺", "pi1(1600)m": "π₁(1600)⁻",
        "pi2(1670)p": "π₂(1670)⁺", "pi2(1670)m": "π₂(1670)⁻",
        "pi1300p": "π(1300)⁺", "pi1300m": "π(1300)⁻",
        "pi1600p": "π(1600)⁺", "pi1600m": "π(1600)⁻",
        "f0(980)": "f₀(980)", "f0(500)": "f₀(500)",
        "NR0": "NR",
    }
    def _nice(p):
        return nice.get(p, p)

    parts = stem.split(".")
    if len(parts) == 2:
        r1, r2 = _nice(parts[0]), _nice(parts[1])
        if r2 in ("π⁺", "π⁻"):
            return f"B → {r1}{r2}"
        if r1 in ("π⁺", "π⁻"):
            return f"B → {r2}{r1}"
        return f"B → {r1} {r2}"
    return f"B → {stem}"


def group_ck_stems(config):
    """Group base ck indices by their first-decay stem.

    Returns list of (stem, label, [base_ck_indices], [wave_labels], has_ls_waves).
    ``has_ls_waves`` is True when the stem has multiple distinct L values
    at the B → first-decay level (e.g. ρρ S/P/D).
    """
    pws = config.full_decay.get_partial_waves_params()
    raw = OrderedDict()
    for i, pw in enumerate(pws):
        name = pw[1]
        m = re.match(r'^B->(.+)_g_ls_(\d+)$', name)
        if not m:
            continue
        raw.setdefault(m.group(1), []).append((i, int(m.group(2))))

    result = []
    for stem, entries in sorted(raw.items(), key=lambda x: _display_name(x[0])):
        label = _display_name(stem)
        base_idxs = [e[0] for e in entries]
        ls_vals = sorted(set(e[1] for e in entries))
        has_ls = len(ls_vals) > 1
        if has_ls:
            by_l = OrderedDict()
            for i, lv in entries:
                by_l.setdefault(lv, []).append(i)
            waves = [(f"L={lv}", idxs) for lv, idxs in by_l.items()]
        else:
            waves = [(f"L={i}", [base_idxs[i]]) for i in range(len(base_idxs))]
        result.append((stem, label, base_idxs, waves, has_ls))
    return result


def _setup_fitter(args):
    """Create and configure Fitter from command-line args."""
    f = Fitter(args.config, backend=args.backend)
    constraints_path = os.path.splitext(args.fit_json)[0] + "_constraints.json"
    if os.path.exists(constraints_path):
        f.load_constraints(constraints_path)
        print(f"  Loaded constraints from {constraints_path}")
    else:
        fs, sp, sc = build_constraints(f.all_comb)
        f.set_fixed(fs)
        f.set_same(sp)
        f.set_scale(sc)
    phsp, _ = Fitter.load_npz(args.phsp, max_events=args.max_events)
    f.set_phsp(phsp)
    return f


def _compute_groups(f, stems, af):
    """Build masks, compute fractions for all groups."""
    n_base = len(f.config.full_decay.get_partial_waves_params())

    def split_ls(idxs):
        ls, lsbar = [], []
        for i in idxs:
            for block in range(4):
                ls.append(block * n_base + i)
                lsbar.append((block + 4) * n_base + i)
        return ls, lsbar

    groups = [(stem, label, split_ls(idxs), waves, has_ls)
              for stem, label, idxs, waves, has_ls in stems]

    all_masks = []
    for stem, label, (ls, lsbar), waves, _ in groups:
        all_masks.append(ls)
        all_masks.append(lsbar)

    vals, errs = af.fractions(all_masks)
    return groups, vals, errs, split_ls, n_base


# ── Output: terminal ───────────────────────────────────────────────────────

def _output_terminal(groups, vals, errs, af, split_ls):
    print("\n" + "=" * 80)
    print("  Amplitude fractions by first decay")
    print("=" * 80)
    print(f"  {'First decay':50s}  {'B0':>20s}  {'B0bar':>20s}")
    print("  " + "-" * 93)

    csv_rows = []
    for i, (stem, label, (ls, lsbar), waves, has_ls) in enumerate(groups):
        v0, v1 = vals[2 * i], vals[2 * i + 1]
        e0, e1 = errs[2 * i], errs[2 * i + 1]
        b0 = f"{v0:8.5f} ± {e0:8.5f}"
        b1 = f"{v1:8.5f} ± {e1:8.5f}"

        if has_ls:
            print(f"  {label:50s}  -- sum of {len(waves)} LS waves --")
            sub_masks = []
            for wave_lbl, sub_idxs in waves:
                sub_ls, sub_lsbar = split_ls(sub_idxs)
                sub_masks.append(sub_ls)
                sub_masks.append(sub_lsbar)
            sub_vals, sub_errs = af.fractions(sub_masks)
            for j, (wave_lbl, _) in enumerate(waves):
                sv, se = sub_vals[2 * j], sub_errs[2 * j]
                svb, seb = sub_vals[2 * j + 1], sub_errs[2 * j + 1]
                print(f"    {wave_lbl:48s}  {sv:8.5f} ± {se:8.5f}  {svb:8.5f} ± {seb:8.5f}")
            print(f"    {'Total':48s}  {b0:>20s}  {b1:>20s}")
        else:
            if len(waves) > 1:
                print(f"  {label:50s}  {b0:>20s}  {b1:>20s}  ({len(waves)} sub-waves)")
            else:
                print(f"  {label:50s}  {b0:>20s}  {b1:>20s}")

        csv_rows.append((label, v0, e0, v1, e1))

    return csv_rows


# ── Output: LaTeX / PDF ────────────────────────────────────────────────────

_LATEX_MAP = {
    "rhoA": r"\rho^{0}", "rhoB": r"\rho^{0}",
    "pip1": r"\pi^{+}", "pip2": r"\pi^{+}",
    "pim1": r"\pi^{-}", "pim2": r"\pi^{-}",
    "a1(1260)p": r"a_{1}(1260)^{+}", "a1(1260)m": r"a_{1}(1260)^{-}",
    "a1(1640)p": r"a_{1}(1640)^{+}", "a1(1640)m": r"a_{1}(1640)^{-}",
    "a2(1320)p": r"a_{2}(1320)^{+}", "a2(1320)m": r"a_{2}(1320)^{-}",
    "pi1(1600)p": r"\pi_{1}(1600)^{+}", "pi1(1600)m": r"\pi_{1}(1600)^{-}",
    "pi2(1670)p": r"\pi_{2}(1670)^{+}", "pi2(1670)m": r"\pi_{2}(1670)^{-}",
    "pi1300p": r"\pi(1300)^{+}", "pi1300m": r"\pi(1300)^{-}",
    "pi1600p": r"\pi(1600)^{+}", "pi1600m": r"\pi(1600)^{-}",
    "f0(980)": r"f_{0}(980)", "f0(500)": r"f_{0}(500)",
    "NR0": r"{\rm NR}",
}

def _latex_label(stem):
    parts = stem.split(".")
    if len(parts) != 2:
        return stem
    r1 = _LATEX_MAP.get(parts[0], parts[0])
    r2 = _LATEX_MAP.get(parts[1], parts[1])
    if r2 in (r"\pi^{+}", r"\pi^{-}"):
        return r"$B\to " + r1 + r2 + r"$"
    if r1 in (r"\pi^{+}", r"\pi^{-}"):
        return r"$B\to " + r2 + r1 + r"$"
    return r"$B\to " + r1 + r"\," + r2 + r"$"


def _fmt(v, e):
    if e is None or e < 1e-10:
        return f"{v:.4f}"
    return f"{v:.4f}\\pm{e:.4f}"


def _output_latex(groups, vals, errs, af, split_ls, output_path, compile_pdf):
    L = []
    L.append(r"\documentclass[11pt,border=2pt]{standalone}")
    L.append(r"\usepackage{booktabs}")
    L.append(r"\usepackage{array}")
    L.append(r"\begin{document}")
    L.append(r"\begin{tabular}{lrr}")
    L.append(r"\toprule")
    L.append(r"First decay & B$^{0}$ & $\overline{\rm B}{}^{0}$ \\")
    L.append(r"\midrule")

    for i, (stem, label_unicode, (ls, lsbar), waves, has_ls) in enumerate(groups):
        v0, v1 = vals[2*i], vals[2*i+1]
        e0, e1 = errs[2*i], errs[2*i+1]
        ltx_label = _latex_label(stem)

        if has_ls:
            L.append(f"  {ltx_label} & ${_fmt(v0,e0)}$ & ${_fmt(v1,e1)}$ \\\\")
            sub_masks = []
            for wave_lbl, sub_idxs in waves:
                sub_ls, sub_lsbar = split_ls(sub_idxs)
                sub_masks.append(sub_ls)
                sub_masks.append(sub_lsbar)
            sub_vals, sub_errs = af.fractions(sub_masks)
            for j, (wave_lbl, _) in enumerate(waves):
                sv, se = sub_vals[2*j], sub_errs[2*j]
                sbv, sbe = sub_vals[2*j+1], sub_errs[2*j+1]
                L.append(f"  \\quad {wave_lbl} & ${_fmt(sv,se)}$ & ${_fmt(sbv,sbe)}$ \\\\")
            L.append(r"  \addlinespace")
        else:
            L.append(f"  {ltx_label} & ${_fmt(v0,e0)}$ & ${_fmt(v1,e1)}$ \\\\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{document}")

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
        texpath = output_path if output_path else "/tmp/fractions.tex"
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
            errs_ = [ln for ln in (r.stderr or "").split("\n") if "Error" in ln]
            for ln in errs_[:5]:
                print(f"  pdflatex: {ln}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Fit fractions by first decay (B → X + Y)")
    ap.add_argument("fit_json", help="JSON from Fitter.save_params()")
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
    args = ap.parse_args()

    # ── Setup ──────────────────────────────────────────────────────
    f = _setup_fitter(args)
    fit_result = f.load_results(args.fit_json)
    if fit_result.x is None or len(fit_result.x) == 0:
        print("ERROR: could not reconstruct x from", args.fit_json)
        sys.exit(1)

    af = AmplitudeFractions(f, fit_result)

    stems = group_ck_stems(f.config)
    groups, vals, errs, split_ls, _ = _compute_groups(f, stems, af)

    # ── Output ─────────────────────────────────────────────────────
    if args.latex:
        _output_latex(groups, vals, errs, af, split_ls,
                      args.latex_output, args.pdf)
    else:
        csv_rows = _output_terminal(groups, vals, errs, af, split_ls)

        if args.output:
            with open(args.output, "w", newline="") as fout:
                w = csv.writer(fout)
                w.writerow(["FirstDecay", "B0_value", "B0_error",
                            "B0bar_value", "B0bar_error"])
                for row in csv_rows:
                    w.writerow(row)
            print(f"\n  Saved CSV to {args.output}")


if __name__ == "__main__":
    main()

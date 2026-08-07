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
    return os.path.join(_SCRIPT_DIR, name)


def _cl_name(stem, name_map):
    """Human-readable short resonance label (raw name for terminal)."""
    return name_map.get(stem, stem).strip("$")


def _latex_name(name, name_map):
    """Convert a name to LaTeX using the name map."""
    ls_part = ""
    if " L=" in name:
        base, ls = name.split(" L=", 1)
        ls_part = f" L={ls}"
    else:
        base = name
    return name_map.get(base, base).strip("$") + ls_part


def discover_sub_decays(config, name_map, merge_cp=False):
    """Group base ck indices by resonance → sub-decay channel."""
    pws = config.full_decay.get_partial_waves_params()

    idx_to_ls = {}
    for start, end, chain in config._chain_ranges():
        if len(chain.decays) > 1:
            ls_list = chain.decays[1].get_ls_list()
            for j, base_idx in enumerate(range(start, end)):
                if j < len(ls_list):
                    idx_to_ls[base_idx] = ls_list[j]

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

        ls_info = idx_to_ls.get(i, (0, 0))
        ls_key = (sub_channel, ls_info)

        chains.setdefault(rx_name, OrderedDict())
        chains[rx_name].setdefault(ls_key, []).append(i)

    if merge_cp:
        merged = OrderedDict()
        for rx_name, subs in chains.items():
            base = rx_name[:-1] if len(rx_name) > 1 and rx_name[-1] in "pm" else rx_name
            if base not in merged:
                merged[base] = (rx_name, OrderedDict())
            for ls_key, idxs in subs.items():
                merged[base][1].setdefault(ls_key, []).extend(idxs)
        chains = OrderedDict()
        for base, (full_name, subs) in merged.items():
            chains[full_name] = subs

    result = []
    for rx_name in sorted(chains, key=lambda x: _cl_name(x, name_map)):
        subs = []
        for (sub_name, (L, S)), idxs in sorted(chains[rx_name].items()):
            ls_counts = sum(1 for k in chains[rx_name] if k[0] == sub_name)
            wave_label = f" L={L}" if ls_counts > 1 else ""
            subs.append((_cl_name(sub_name, name_map) + wave_label, sorted(set(idxs))))
        result.append((_cl_name(rx_name, name_map), subs))
    return result


def _setup(args):
    from ampfit import Fitter

    f = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.fit_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    else:
        print(f"WARNING: no constraints file at {cp}")
    phsp, _ = Fitter.load_npz(args.phsp, max_events=args.max_events)
    f.set_phsp(phsp)
    return f


def _output_terminal(stems, csv_rows, af, n_base, merge_cp):
    """Terminal output. Denominator and numerator results are cached inside ``af``."""
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

    for rx_label, subs in stems:
        print(f"\n  ── {rx_label} ──")
        print(f"  {'Sub-channel':30s}  {'B0 ratio':>22s}  {'B0bar ratio':>22s}")
        print("  " + "-" * 75)

        if merge_cp:
            pi_p_all = [idxs[0] for _, idxs in subs]
            ls_p_total, lsbar_p_total = split_ls(pi_p_all)
        else:
            pi_p_all = [idx for _, idxs in subs for idx in idxs]
            ls_p_total, lsbar_p_total = split_ls(pi_p_all)

        sub_groups = []
        for sub_label, idxs in subs:
            base = sub_label.split(" L=")[0] if " L=" in sub_label else sub_label
            if sub_groups and sub_groups[-1][0] == base:
                sub_groups[-1][1].append((sub_label, idxs[0] if merge_cp else idxs))
            else:
                sub_groups.append((base, [(sub_label, idxs[0] if merge_cp else idxs)]))

        for base_label, entries in sub_groups:
            for sub_label, pi_p_idx in entries:
                ls_p, lsbar_p = split_ls([pi_p_idx])
                # Denominator cached internally after first call
                v_b0, e_b0 = af.fractions([ls_p], denominator=ls_p_total)
                v_b1, e_b1 = af.fractions([lsbar_p], denominator=lsbar_p_total)
                r0, s0 = v_b0[0], e_b0[0]
                r1, s1 = v_b1[0], e_b1[0]
                print(f"  {sub_label:30s}  {r0:8.5f} ± {s0:8.5f}  "
                      f"{r1:8.5f} ± {s1:8.5f}")
                csv_rows.append((rx_label, sub_label, r0, s0, r1, s1))

            if len(entries) > 1:
                all_pi_p = [e[1] for e in entries]
                ls_all, lsbar_all = split_ls(all_pi_p)
                v_tot, e_tot = af.fractions([ls_all], denominator=ls_p_total)
                v_tot1, e_tot1 = af.fractions([lsbar_all], denominator=lsbar_p_total)
                r0, s0 = v_tot[0], e_tot[0]
                r1, s1 = v_tot1[0], e_tot1[0]
                print(f"  {'  Total':28s}  {r0:8.5f} ± {s0:8.5f}  "
                      f"{r1:8.5f} ± {s1:8.5f}")
                csv_rows.append((rx_label, f"  Total {base_label}", r0, s0, r1, s1))


def _fmt(v, e):
    if e is None:
        return f"{v:.4f}"
    return f"{v:.4f}\\pm{max(e, 0.0):.4f}"


def _output_latex(stems, af, n_base, name_map, output_path, compile_pdf=True, merge_cp=True):
    """LaTeX/PDF output.  Results cached inside ``af`` from terminal run."""
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

    first_group = True
    for rx_label, subs in stems:
        if not first_group:
            L.append(r"  \midrule")
        first_group = False
        rx_ltx = name_map.get(rx_label)
        if rx_ltx is None:
            for sfx in ("p", "m"):
                key = rx_label + sfx
                if key in name_map:
                    rx_ltx = name_map[key]
                    break
        if rx_ltx is None:
            rx_ltx = rx_label
        else:
            rx_ltx = rx_ltx.strip("$")

        if merge_cp:
            pi_p_all = [idxs[0] for _, idxs in subs]
            ls_p_total, lsbar_p_total = split_ls(pi_p_all)
        else:
            pi_p_all = [idx for _, idxs in subs for idx in idxs]
            ls_p_total, lsbar_p_total = split_ls(pi_p_all)

        sub_groups = []
        for sub_label, idxs in subs:
            base = sub_label.split(" L=")[0] if " L=" in sub_label else sub_label
            if sub_groups and sub_groups[-1][0] == base:
                sub_groups[-1][1].append((sub_label, idxs[0] if merge_cp else idxs))
            else:
                sub_groups.append((base, [(sub_label, idxs[0] if merge_cp else idxs)]))

        n_subs = sum(len(e) + (1 if len(e) > 1 else 0) for _, e in sub_groups)
        first = True
        for base_label, entries in sub_groups:
            for k, (sub_label, pi_p_idx) in enumerate(entries):
                ls_p, lsbar_p = split_ls([pi_p_idx])
                # Denominator and numerators cached from terminal run
                v_b0, e_b0 = af.fractions([ls_p], denominator=ls_p_total)
                v_b1, e_b1 = af.fractions([lsbar_p], denominator=lsbar_p_total)
                r0, s0 = v_b0[0], e_b0[0]
                r1, s1 = v_b1[0], e_b1[0]
                sub_ltx = _latex_name(sub_label, name_map)
                if first:
                    span = f"\\multirow{{{n_subs}}}{{*}}{{$\\ {rx_ltx}$}}"
                    L.append(f"  {span} & ${sub_ltx}$ & ${_fmt(r0,s0)}$ & ${_fmt(r1,s1)}$ \\\\")
                    first = False
                else:
                    L.append(f"   & ${sub_ltx}$ & ${_fmt(r0,s0)}$ & ${_fmt(r1,s1)}$ \\\\")

            if len(entries) > 1:
                all_pi_p = [e[1] for e in entries]
                ls_all, lsbar_all = split_ls(all_pi_p)
                vt, et = af.fractions([ls_all], denominator=ls_p_total)
                vt1, et1 = af.fractions([lsbar_all], denominator=lsbar_p_total)
                r0, s0 = vt[0], et[0]
                r1, s1 = vt1[0], et1[0]
                L.append(f"   & ${_latex_name(base_label, name_map)}$ & ${_fmt(r0,s0)}$ & ${_fmt(r1,s1)}$ \\\\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{document}")
    tex = "\n".join(L)
    if output_path:
        with open(output_path, "w") as f:
            f.write(tex)
        print(f"  Saved {output_path}")

    if compile_pdf:
        outdir = os.path.dirname(output_path) if output_path else "/tmp"
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "-output-directory=" + outdir, output_path],
                           capture_output=True, text=True, timeout=60)
        pdfpath = os.path.splitext(output_path)[0] + ".pdf"
        if os.path.exists(pdfpath):
            print(f"  PDF: {pdfpath}")


def main():
    ap = argparse.ArgumentParser(
        description="Sub-decay branching ratios for 3π resonances")
    ap.add_argument("fit_json")
    ap.add_argument("--config", default=_default_path("config_angle.yml"))
    ap.add_argument("--phsp", default=_default_path("data/phsp_noeff_sym_arrays.npz"))
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--backend", default="cuda_v3_sparse",
                    help="Compute backend (use an f64 backend — f32 "
                         "backends like cuda32_v3 give noisy gradients "
                         "for fraction uncertainties)")
    ap.add_argument("-o", "--output", default=None,
                    help="Output prefix for CSV, LaTeX, and PDF (e.g. /path/to/prefix)")
    ap.add_argument("--no-merge-cp", action="store_true",
                    help="Show charge-conjugate pairs separately")
    args = ap.parse_args()

    from ampfit import Fitter
    from ampfit.amp_frac import AmplitudeFractions

    f = _setup(args)
    fit_result = f.load_results(args.fit_json)
    if fit_result.x is None or len(fit_result.x) == 0:
        print("ERROR: could not reconstruct x from", args.fit_json)
        sys.exit(1)

    af = AmplitudeFractions(f, fit_result)

    name_map = f.config.name_display_map()

    stems = discover_sub_decays(f.config, name_map, merge_cp=not args.no_merge_cp)
    n_base = len(f.config.full_decay.get_partial_waves_params())

    # ── Terminal output (populates ``af`` cache) ────────────────────
    csv_rows = []
    _output_terminal(stems, csv_rows, af, n_base, merge_cp=not args.no_merge_cp)

    # ── CSV output ─────────────────────────────────────────────────
    if args.output:
        csv_path = args.output + ".csv"
        with open(csv_path, "w", newline="") as fout:
            w = csv.writer(fout)
            w.writerow(["Resonance", "SubChannel",
                        "B0_ratio", "B0_ratio_err",
                        "B0bar_ratio", "B0bar_ratio_err"])
            for row in csv_rows:
                w.writerow(row)
        print(f"\n  Saved CSV to {csv_path}")

    # ── LaTeX + PDF output (reuses ``af`` cache from terminal) ──────
    if args.output:
        tex_path = args.output + ".tex"
        _output_latex(stems, af, n_base, name_map, tex_path,
                      compile_pdf=True, merge_cp=not args.no_merge_cp)


if __name__ == "__main__":
    main()
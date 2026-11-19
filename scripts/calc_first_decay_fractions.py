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
from tabpwa import Fitter
from tabpwa.amp_frac import AmplitudeFractions


def _make_label_fn(name_map):
    """Create a display-label function from a name→display map.

    The returned function converts a dot-separated stem (e.g.
    ``"rhoA.rhoB"``) to a human-readable label.

    With ``latex=False`` (default) returns Unicode for terminal output.
    With ``latex=True`` returns a full LaTeX string (with ``$``).
    """
    def _nice(p):
        return name_map.get(p, p).strip("$")

    def label(stem, latex=False):
        parts = stem.split(".")
        if len(parts) != 2:
            return stem
        r1, r2 = _nice(parts[0]), _nice(parts[1])
        is_pion = {r"\pi^{+}", r"\pi^{-}", "π⁺", "π⁻"}
        if r2 in is_pion:
            body = f"{r1}{r2}"
        elif r1 in is_pion:
            body = f"{r2}{r1}"
        else:
            body = f"{r1}{r'\,' if latex else ' '}{r2}"
        if latex:
            return rf"$B\to {body}$"
        return f"B → {body}"
    return label


def _wave_expansion(model):
    """``[(base_index, offset, count)]``: geometry wave → FLAT ck slots.

    ``AmplitudeModel.wave_expansion()`` is basis-aware: a basis particle model
    expands one geometry wave into ``count`` contiguous flat ck slots (the
    numerator rows), so ``sum(count) == len(model.get_partial_waves_params())``
    is the block stride of the flat ck vector.  The base ``AmplitudeModel``
    answers with the identity mapping (one slot per geometry wave), so a plain
    ``pwa`` config needs no special case.
    """
    return list(model.wave_expansion())


def _n_flat_waves(model):
    """Block stride of the FLAT ck vector (expanded, not geometry)."""
    return sum(cnt for _b, _off, cnt in _wave_expansion(model))


def group_ck_stems(config, label_fn):
    """Group base ck indices by their first-decay stem.

    Returns list of (stem, label, [flat_ck_indices], [wave_labels], has_ls_waves).
    ``has_ls_waves`` is True when the stem has multiple distinct L values
    at the B → first-decay level (e.g. ρρ S/P/D).

    A geometry wave can own several FLAT ck slots (basis particle model): the
    stem regex reads the geometry list — the numerator head is appended as the
    LAST tuple element, so ``pw[1]`` is still ``B->..._g_ls_L`` — while the
    recorded indices come from ``wave_expansion`` (all slots of the wave).
    """
    pws = config.full_decay.get_partial_waves_params()
    spans = {b: (off, count) for b, off, count in _wave_expansion(config)}
    raw = OrderedDict()
    for i, pw in enumerate(pws):
        name = pw[1]
        m = re.match(r'^B->(.+)_g_ls_(\d+)$', name)
        if not m:
            continue
        off, count = spans[i]
        raw.setdefault(m.group(1), []).append(
            (int(m.group(2)), list(range(off, off + count))))

    result = []
    for stem, entries in sorted(raw.items(), key=lambda x: label_fn(x[0])):
        label = label_fn(stem)
        base_idxs = [j for _lv, idxs in entries for j in idxs]
        ls_vals = sorted(set(lv for lv, _idxs in entries))
        has_ls = len(ls_vals) > 1
        if has_ls:
            by_l = OrderedDict()
            for lv, idxs in entries:
                by_l.setdefault(lv, []).extend(idxs)
            waves = [(f"L={lv}", idxs) for lv, idxs in by_l.items()]
        else:
            waves = [(f"L={i}", idxs) for i, (_lv, idxs) in enumerate(entries)]
        result.append((stem, label, base_idxs, waves, has_ls))
    return result


def _setup_fitter(args):
    """Create and configure Fitter from command-line args."""
    f = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.fit_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
        print(f"  Loaded constraints from {cp}")
    else:
        print(f"WARNING: no constraints file at {cp}")
    # the sample comes from the fitter's generic loader (config angle layout,
    # path map and sidecars), not the legacy load_npz default
    # a fraction is a ratio of integrals, so the unweighted (efficiency-free)
    # phase space is the right measure; ``phsp`` is the fallback
    phsp = (f.load_dataset("phsp_noeff", path=args.phsp,
                           max_events=args.max_events)
            or f.load_dataset("phsp", path=args.phsp,
                              max_events=args.max_events))
    if phsp is None:
        raise SystemExit(f"no phase space: data.phsp is not configured and "
                         f"--phsp was not given")
    f.set_phsp(phsp)
    return f


def _compute_groups(f, stems, af, batch=None):
    """Build masks, compute fractions for all groups.

    ``n_wave`` is the FLAT ck block stride (one slot per expanded wave), not
    the geometry count — a basis model owns several slots per geometry wave.
    """
    n_wave = _n_flat_waves(f.model)

    def split_ls(idxs):
        ls, lsbar = [], []
        for i in idxs:
            for block in range(4):
                ls.append(block * n_wave + i)
                lsbar.append((block + 4) * n_wave + i)
        return ls, lsbar

    groups = [(stem, label, split_ls(idxs), waves, has_ls)
              for stem, label, idxs, waves, has_ls in stems]

    all_masks = []
    for stem, label, (ls, lsbar), waves, _ in groups:
        all_masks.append(ls)
        all_masks.append(lsbar)

    vals, errs = af.fractions(all_masks, batch=batch)
    return groups, vals, errs, split_ls, n_wave


# ── Output: terminal ───────────────────────────────────────────────────────

def _output_terminal(groups, vals, errs, af, split_ls, batch=None):
    """Terminal output."""
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
            sub_vals, sub_errs = af.fractions(sub_masks, batch=batch)
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


def _output_latex(groups, vals, errs, af, split_ls, label_fn, output_path,
                  compile_pdf=True, batch=None):
    """LaTeX/PDF output."""

    def _fmt(v, e):
        if e is None:
            return f"{v:.4f}"
        return f"{v:.4f}\\pm{max(e, 0.0):.4f}"

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
        ltx_label = label_fn(stem, latex=True)

        if has_ls:
            L.append(f"  {ltx_label} & ${_fmt(v0,e0)}$ & ${_fmt(v1,e1)}$ \\\\")
            sub_masks = []
            for wave_lbl, sub_idxs in waves:
                sub_ls, sub_lsbar = split_ls(sub_idxs)
                sub_masks.append(sub_ls)
                sub_masks.append(sub_lsbar)
            sub_vals, sub_errs = af.fractions(sub_masks, batch=batch)
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
    ap.add_argument("--phsp", default=None,
                    help="Phase-space file (.npz kernel arrays or momentum "
                         ".npy); default: the config's data.phsp_noeff "
                         "(efficiency-free), falling back to data.phsp")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None,
                    help="Events per chunk: stream the phase space through "
                         "sum_stream (peak memory is one chunk)")
    ap.add_argument("--backend", default=None,
                    help="Compute backend (use an f64 backend — f32 "
                         "backends like cuda32_v3 give noisy gradients "
                         "for fraction uncertainties) "
                         " (default: config's config.backend)")
    ap.add_argument("-o", "--output", default=None,
                    help="Output prefix for CSV, LaTeX, and PDF (e.g. /path/to/prefix)")
    args = ap.parse_args()

    # ── Setup ──────────────────────────────────────────────────────
    f = _setup_fitter(args)
    fit_result = f.load_results(args.fit_json)
    if fit_result.x is None or len(fit_result.x) == 0:
        print("ERROR: could not reconstruct x from", args.fit_json)
        sys.exit(1)

    af = AmplitudeFractions(f, fit_result)

    name_map = f.name_display_map()
    label_fn = _make_label_fn(name_map)

    stems = group_ck_stems(f.model, label_fn)
    groups, vals, errs, split_ls, _ = _compute_groups(f, stems, af,
                                                      batch=args.batch)

    # ── Terminal output (always) ───────────────────────────────────
    csv_rows = _output_terminal(groups, vals, errs, af, split_ls,
                                batch=args.batch)

    # ── CSV output ─────────────────────────────────────────────────
    if args.output:
        csv_path = args.output + ".csv"
        with open(csv_path, "w", newline="") as fout:
            w = csv.writer(fout)
            w.writerow(["FirstDecay", "B0_value", "B0_error",
                        "B0bar_value", "B0bar_error"])
            for row in csv_rows:
                w.writerow(row)
        print(f"\n  Saved CSV to {csv_path}")

    # ── LaTeX + PDF output ─────────────────────────────────────────
    if args.output:
        tex_path = args.output + ".tex"
        _output_latex(groups, vals, errs, af, split_ls, label_fn,
                      tex_path, compile_pdf=True, batch=args.batch)


if __name__ == "__main__":
    main()

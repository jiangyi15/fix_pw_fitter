#!/usr/bin/env python3
"""Compute sub-decay ratios in PDG comparison format (LaTeX table) with uncertainties.

Partial-wave indices are resolved at runtime by searching the config's
decay chain structure — no hardcoded numbers.

Output: .tex file alongside the input JSON (or --output).
"""
import sys, os, numpy as np
_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPT_DIR)
from ampfit import Fitter
from ampfit.amp_frac import AmplitudeFractions
from ampfit.utils import fmt_meas, fmt_particle
from run_fit import build_constraints


# ── PDG reference data ──────────────────────────────────────────────
# Structured as resonance → {"den": <denominator pairs>, "modes": [...]}.
# Each mode:
#   label  : LaTeX label
#   pdg    : PDG reference value string
#   tag    : optional tag, "*" → single measurement (no PDG avg)
#   num    : numerator decay pairs (list of (parent,child[,wave_idx]))
#   den    : optional custom denominator (list or str key).
#            If omitted, uses the section-level den.
#   fmt    : "pct" (percentage) or "rat" (raw ratio)
#
# Decay pairs that don't exist in the config are silently ignored
# (amplitude assumed zero), so entries for missing modes automatically
# show "$-$" with no manual None needed.

PDG = {
    "a1(1260)": {
        "den": [
            ("a1(1260)p", "rhoA"), ("a1(1260)p", "f0(980)"),
            ("a1(1260)p", "f0(500)"), ("a1(1260)p", "f0(1370)"),
            ("a1(1260)p", "f2(1270)"),
        ],
        "modes": [
            {"label": r"$\Gamma(a_1\to[\rho\pi]_S)/\Gamma_{\mathrm{tot}}$",
             "pdg": r"60.19\%", "tag": "*",
             "num": [("a1(1260)p", "rhoA", 0)], "fmt": "pct"},
            {"label": r"$\Gamma(a_1\to[\rho\pi]_D)/\Gamma_{\mathrm{tot}}$",
             "pdg": r"1.30\pm0.60\pm0.22\%", "tag": "*",
             "num": [("a1(1260)p", "rhoA", 1)], "fmt": "pct"},
            {"label": r"$\Gamma(a_1\to[\rho(1450)\pi]_S)/\Gamma_{\mathrm{tot}}$",
             "pdg": r"0.56\pm0.84\pm0.32\%", "tag": "*",
             "num": [("a1(1260)p", "rho(1450)", 0)], "fmt": "pct"},
            {"label": r"$\Gamma(a_1\to[\rho(1450)\pi]_D)/\Gamma_{\mathrm{tot}}$",
             "pdg": r"2.04\pm1.20\pm0.28\%", "tag": "*",
             "num": [("a1(1260)p", "rho(1450)", 1)], "fmt": "pct"},
            {"label": r"$\Gamma(a_1\to f_0(500)\pi)/\Gamma_{\mathrm{tot}}$",
             "pdg": r"18.76\pm4.29\pm1.48\%", "tag": "*",
             "num": [("a1(1260)p", "f0(500)")], "fmt": "pct"},
            {"label": r"$\Gamma(a_1\to f_0(1370)\pi)/\Gamma_{\mathrm{tot}}$",
             "pdg": r"7.40\pm2.71\pm1.26\%", "tag": "*",
             "num": [("a1(1260)p", "f0(1370)")], "fmt": "pct"},
            {"label": r"$\Gamma(a_1\to f_2(1270)\pi)/\Gamma_{\mathrm{tot}}$",
             "pdg": r"1.19\pm0.49\pm0.17\%", "tag": "*",
             "num": [("a1(1260)p", "f2(1270)")], "fmt": "pct"},
        ],
    },
    "a1(1640)": {
        "den": [("a1(1640)p", "f2(1270)"), ("a1(1640)p", "f0(500)")],
        "modes": [
            {"label": r"$\Gamma(a_1\to f_2(1270)\pi)/\Gamma(a_1\to\sigma\pi)$",
             "pdg": r"0.24\pm0.07",
             "num": [("a1(1640)p", "f2(1270)")],
             "den": [("a1(1640)p", "f0(500)")], "fmt": "rat"},
        ],
    },
    "a2(1320)": {
        "den": None,
        "modes": [
            {"label": r"$\Gamma(a_2\to\rho\pi)/\Gamma(a_2\to f_2(1270)\pi)$",
             "pdg": r"16.5^{+1.2}_{-2.4}",
             "num": [("a2(1320)p", "rhoA")],
             "den": [("a2(1320)p", "f2(1270)")], "fmt": "rat"},
        ],
    },
    "pi2(1670)": {
        "den": [
            ("pi2(1670)p", "f2(1270)"), ("pi2(1670)p", "f0(980)"),
            ("pi2(1670)p", "f0(500)"), ("pi2(1670)p", "rhoA"),
        ],
        "modes": [
            {"label": r"$\Gamma(\pi_2\to f_2(1270)\pi)/\Gamma(\pi_2\to 3\pi)$",
             "pdg": r"56.3\pm3.2\%",
             "num": [("pi2(1670)p", "f2(1270)")], "fmt": "pct"},
            {"label": r"$\Gamma(\pi_2\to\rho\pi)/\Gamma(\pi_2\to 3\pi)$",
             "pdg": r"31\pm4\%",
             "num": [("pi2(1670)p", "rhoA")], "fmt": "pct"},
            {"label": r"$\Gamma(\pi_2\to f_0(500)\pi)/\Gamma(\pi_2\to 3\pi)$",
             "pdg": r"10\pm4\%",
             "num": [("pi2(1670)p", "f0(500)")], "fmt": "pct"},
            {"label": r"$\Gamma(\pi_2\to[\pi\pi]_S\pi)/\Gamma(\pi_2\to 3\pi)$",
             "pdg": r"8.7\pm3.4\%",
             "num": [("pi2(1670)p", "f0(500)"), ("pi2(1670)p", "f0(980)")],
             "fmt": "pct"},
        ],
    },
    "pi(1300)": {
        "den": [("pi1300p", "rhoA")],
        "modes": [
            {"label": r"$\Gamma(\pi_0\to[\pi\pi]_S\pi)/\Gamma(\pi_0\to\rho\pi)$",
             "pdg": r"2.2\pm0.4", "tag": "*",
             "num": [("pi1300p", "f0(500)"), ("pi1300p", "f0(980)")],
             "fmt": "rat"},
        ],
    },
    "pi(1800)": {
        "den": None,
        "modes": [
            {"label": r"$\Gamma(\pi_0\to f_0(980)\pi)/\Gamma(\pi_0\to f_0(500)\pi)$",
             "pdg": r"0.44\pm0.08\pm0.38",
             "num": [("pi1600p", "f0(980)")],
             "den": [("pi1600p", "f0(500)")], "fmt": "rat"},
            {"label": r"$\Gamma(\pi_0\to\rho\pi)/\Gamma(\pi_0\to f_0(980)\pi)$",
             "pdg": r"<0.25", "tag": "*",
             "num": [("pi1600p", "rhoA")],
             "den": [("pi1600p", "f0(980)")], "fmt": "rat"},
        ],
    },
}


# ── Helpers ─────────────────────────────────────────────────────────

def _get(cfg, pairs):
    """Return expanded ck indices for a list of (parent, child) tuples.

    Each entry can be:
      - ``(parent, child)`` — resolves all partial waves
      - ``(parent, child, wave_idx)`` — resolves only the *wave_idx*-th
        partial wave (e.g. ``wave_idx=0`` for S-wave, ``1`` for D-wave)

    Each tuple is resolved independently via ``get_decay_ck_indices``,
    then unioned (OR semantics).
    """
    all_idx = []
    for entry in pairs:
        if len(entry) == 3:
            parent, child, wave_idx = entry
        else:
            parent, child = entry
            wave_idx = None
        all_idx.extend(cfg.get_decay_ck_indices([(parent, child)], wave_idx=wave_idx))
    return list(set(all_idx))


def compute(af, cfg, den_pairs, num_pairs, fmt):
    """Compute a fraction or ratio.

    *den_pairs* is a list of ``(parent, child[, wave_idx])`` tuples or
    ``None`` (no denominator → ``N/0`` is undefined).

    Returns ``(value, error)`` or ``None`` when denominator is empty
    or undefined.  When only numerator is empty, returns 0.
    """
    n = _get(cfg, num_pairs)
    if den_pairs is None:
        return None       # no denominator specified
    d = _get(cfg, den_pairs)
    if not d:
        return None       # N/0 is undefined
    if not n:
        return (0.0, 0.0)  # 0/D = 0
    v, e = af.fractions([n], denominator=d)
    if fmt == "pct":
        return v[0] * 100, e[0] * 100
    return v[0], e[0]


def fmt_val(result, fmt):
    """Format computed result for LaTeX output."""
    if result is None:
        return r"$-$"
    return fmt_meas(result[0], result[1], pct=(fmt == "pct"))


def fmt_pdg(entry):
    """Format the PDG reference value from an entry."""
    pdg = entry.get("pdg", "")
    if not pdg:
        return r"$-$"
    tag = entry.get("tag", "")
    if tag == "*":
        return rf"${pdg}^{{\ast}}$"
    return f"${pdg}$"


def emit_section(tex, label, entries, sec_den, af, cfg):
    """Emit LaTeX for one resonance section."""
    P = tex.write
    P(f"\\hline\\multicolumn{{3}}{{|l|}}{{{label}}}\\\\\\hline\n")
    for entry in entries:
        # Resolve denominator: mode-level overrides section-level
        den = entry.get("den", sec_den)
        # String key → look up in PDG
        if isinstance(den, str):
            den = PDG[den]["den"]
        result = compute(af, cfg, den, entry["num"], entry["fmt"])
        ours = fmt_val(result, entry["fmt"])
        P(f"  {entry['label']} & {fmt_pdg(entry)} & {ours}\\\\\n")


# ── Main ────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("fit_json")
    ap.add_argument("--config", default=_SCRIPT_DIR + "/config_amp.yml")
    ap.add_argument("--phsp", default=_SCRIPT_DIR + "/data/phsp_noeff_sym_arrays.npz")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--backend", default="cuda32_v3")
    ap.add_argument("-o", "--output", default=None,
                    help="Output .tex file (default: same basename as fit_json)")
    args = ap.parse_args()

    f = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.fit_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    else:
        fs, sp, sc = build_constraints(f.all_comb)
        f.set_fixed(fs); f.set_same(sp); f.set_scale(sc)

    phsp, _ = Fitter.load_npz(args.phsp, max_events=args.max_events)
    phsp['time'] = np.zeros(phsp['mass'].shape[0])
    f.set_phsp(phsp)

    r = f.load_results(args.fit_json)
    if r.x is None or len(r.x) == 0:
        print("ERROR: no fit results")
        sys.exit(1)
    af = AmplitudeFractions(f, r)
    cfg = f.config

    # ── Build LaTeX ────────────────────────────────────────────────
    out_path = args.output or (os.path.splitext(args.fit_json)[0] + ".tex")
    with open(out_path, "w") as tex:
        def P(s):
            tex.write(s + "\n")
        P(r"\documentclass[11pt]{article}")
        P(r"\usepackage[landscape,margin=1.5cm]{geometry}")
        P(r"\usepackage[utf8]{inputenc}")
        P(r"\begin{document}")
        P(r"\begin{table}[h]\centering")
        P(r"\caption{\label{tab:br_pdg}Compare with PDG. $^{\ast}$ single measurement, no PDG average.}")
        P(r"\begin{tabular}{|c|c|c|}\hline")
        P(r" & PDG & ours\\\hline")

        # Emit each resonance section
        for key in PDG:
            sec = PDG[key]
            emit_section(tex, fmt_particle(key), sec["modes"], sec["den"], af, cfg)

        P(r"\hline\end{tabular}\end{table}\end{document}")

    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()

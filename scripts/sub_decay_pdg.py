#!/usr/bin/env python3
"""Compute sub-decay ratios in PDG comparison format (LaTeX table) with uncertainties."""
import sys, os, numpy as np
_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPT_DIR)
from ampfit import Fitter
from ampfit.amp_frac import AmplitudeFractions
from run_fit import build_constraints

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("fit_json"); ap.add_argument("--config", default=_SCRIPT_DIR+"/config_angle.yml")
    ap.add_argument("--phsp", default=_SCRIPT_DIR+"/data/phsp_noeff_sym_arrays.npz")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--backend", default="cuda32_v3")
    args = ap.parse_args()

    f = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.fit_json)[0] + "_constraints.json"
    if os.path.exists(cp): f.load_constraints(cp)
    else: fs,sp,sc = build_constraints(f.all_comb); f.set_fixed(fs); f.set_same(sp); f.set_scale(sc)
    phsp, _ = Fitter.load_npz(args.phsp, max_events=args.max_events)
    phsp['time'] = np.zeros(phsp['mass'].shape[0]); f.set_phsp(phsp)
    r = f.load_results(args.fit_json)
    if r.x is None or len(r.x) == 0: print("ERROR"); return
    af = AmplitudeFractions(f, r)

    n_base = 56
    def split(idxs):
        ls, lsbar = [], []
        for i in idxs:
            for b in range(4):
                ls.append(b * n_base + i)
                lsbar.append((b + 4) * n_base + i)
        return ls, lsbar

    def pct(sub_idxs, denom_idxs):
        ls, _ = split(sub_idxs); den, _ = split(denom_idxs)
        v, e = af.fractions([ls], denominator=den)
        return v[0]*100, e[0]*100

    def rat(num_idxs, den_idxs):
        n, _ = split(num_idxs); d, _ = split(den_idxs)
        v, e = af.fractions([n], denominator=d)
        return v[0], e[0]

    print(r"\documentclass[11pt]{article}")
    print(r"\usepackage[landscape,margin=1.5cm]{geometry}")
    print(r"\begin{document}")
    print(r"\begin{table}[h]\centering")
    print(r"\caption{\label{tab:br_pdg}Compare with PDG. $*$ newest, no PDG avg.}")
    print(r"\begin{tabular}{|c|c|c|}\hline")
    print(r" & PDG & ours\\\hline")

    den_a1 = [10,11,18,19,14,15,22,23]
    print(r"\multicolumn{3}{|l|}{$a_1(1260)$}\\\hline")
    v,e = pct([10,14], den_a1)
    print(f"$\\Gamma(a_1\\to[\\rho\\pi]_S)/\\Gamma_{{tot}}$ & $60.19$\\%$*$ & ${v:.1f}\\pm{e:.1f}$\\%\\\\")
    v,e = pct([11,15], den_a1)
    print(f"$\\Gamma(a_1\\to[\\rho\\pi]_D)/\\Gamma_{{tot}}$ & $1.30\\pm0.60\\pm0.22$\\%$*$ & ${v:.1f}\\pm{e:.1f}$\\%\\\\")
    print(r"$\Gamma(a_1\to[\rho(1450)\pi]_S)/\Gamma_{tot}$ & $0.56\pm0.84\pm0.32$\%$*$ & $-$\\")
    print(r"$\Gamma(a_1\to[\rho(1450)\pi]_D)/\Gamma_{tot}$ & $2.04\pm1.20\pm0.28$\%$*$ & $-$\\")
    v,e = pct([19,23], den_a1)
    print(f"$\\Gamma(a_1\\to f_0(500)\\pi)/\\Gamma_{{tot}}$ & $18.76\\pm4.29\\pm1.48$\\%$*$ & ${v:.1f}\\pm{e:.1f}$\\%\\\\")
    print(r"$\Gamma(a_1\to f_0(1370)\pi)/\Gamma_{tot}$ & $7.40\pm2.71\pm1.26$\%$*$ & $-$\\")
    print(r"$\Gamma(a_1\to f_2(1270)\pi)/\Gamma_{tot}$ & $1.19\pm0.49\pm0.17$\%$*$ & $-$\\")

    print(r"\hline\multicolumn{3}{|l|}{$a_1(1640)$}\\\hline")
    v,e = rat([26,27,31,32], [21,25])
    print(f"$\\Gamma(a_1\\to f_2(1270)\\pi)/\\Gamma(a_1\\to\\sigma\\pi)$ & $0.24\\pm0.07$ & ${v:.1f}\\pm{e:.1f}$\\\\")

    print(r"\hline\multicolumn{3}{|l|}{$a_2(1320)$}\\\hline")
    print(r"$\Gamma(a_2\to\rho\pi)/\Gamma(a_2\to f_2(1270)\pi)$ & $16.5^{+1.2}_{-2.4}$ & $-$\\")

    den_p2 = [28,29,30,42,43,44,45,33,34,35,52,53,54,55]
    print(r"\hline\multicolumn{3}{|l|}{$\pi_2(1670)$}\\\hline")
    v,e = pct([28,29,30,33,34,35], den_p2)
    print(f"$\\Gamma(\\pi_2\\to f_2(1270)\\pi)/\\Gamma(\\pi_2\\to 3\\pi)$ & $56.3\\pm3.2$\\% & ${v:.1f}\\pm{e:.1f}$\\%\\\\")
    v,e = pct([44,45,54,55], den_p2)
    print(f"$\\Gamma(\\pi_2\\to\\rho\\pi)/\\Gamma(\\pi_2\\to 3\\pi)$ & $31\\pm4$\\% & ${v:.1f}\\pm{e:.1f}$\\%\\\\")
    v,e = pct([43,53], den_p2)
    print(f"$\\Gamma(\\pi_2\\to f_0(500)\\pi)/\\Gamma(\\pi_2\\to 3\\pi)$ & $10\\pm4$\\% & ${v:.1f}\\pm{e:.1f}$\\%\\\\")
    # [ππ]S π = f₀(500)π + f₀(980)π
    v,e = pct([42,43,52,53], den_p2)
    print(f"$\\Gamma(\\pi_2\\to[\\pi\\pi]_S\\pi)/\\Gamma(\\pi_2\\to 3\\pi)$ & $8.7\\pm3.4$\\% & ${v:.1f}\\pm{e:.1f}$\\%\\\\")

    print(r"\hline\multicolumn{3}{|l|}{$\pi(1300)$}\\\hline")
    # [ππ]S π / ρπ = (f₀(500)π + f₀(980)π) / ρπ
    v, e = rat([36,37,46,47], [38,48])
    print(f"$\\Gamma(\\pi_0\\to[\\pi\\pi]_S\\pi)/\\Gamma(\\pi_0\\to\\rho\\pi)$ & $2.2\\pm0.4$ $*$ & ${v:.2f}\\pm{e:.2f}$\\\\")

    print(r"\hline\multicolumn{3}{|l|}{$\pi(1800)$}\\\hline")
    v1, e1 = rat([39,49], [40,50])
    print(f"$\\Gamma(\\pi_0\\to f_0(980)\\pi)/\\Gamma(\\pi_0\\to f_0(500)\\pi)$ & $0.44\\pm0.08\\pm0.38$ & ${v1:.2f}\\pm{e1:.2f}$\\\\")
    v2, e2 = rat([41,51], [39,49])
    print(f"$\\Gamma(\\pi_0\\to\\rho\\pi)/\\Gamma(\\pi_0\\to f_0(980)\\pi)$ & $<0.25$ $*$ & ${v2:.2f}\\pm{e2:.2f}$\\\\")

    print(r"\hline\end{tabular}\end{table}\end{document}")

if __name__ == "__main__":
    main()

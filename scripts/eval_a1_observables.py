#!/usr/bin/env python3
"""
Evaluate a1(1260)± weak-phase observables with uncertainty propagation.

The B → a1(1260)π decay has a single wave (L = 1), so per charge state
there is exactly one ratio of the B̄ (g_lsbar) to B (g_ls) coupling:

      α₊ = ½ arg(ā₊/a₊)   for a1(1260)⁺  (B → a1⁺ π⁻)
      α₋ = ½ arg(ā₋/a₋)   for a1(1260)⁻  (B → a1⁻ π⁺)
      λ₊ = |ā₊/a₊|,  λ₋ = |ā₋/a₋|    (the B̄/B magnitude ratios)

Uncertainties come from the fit covariance sub-matrix via finite-
difference gradients on the flat x vector.

Usage:
    python scripts/eval_a1_observables.py fit_results.json \\
        --config config_angle.yml --constraints results_constraints.json \\
        [--tex a1_params.tex]
"""
import sys, os, math
import numpy as np

_SRC = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _SRC)

import argparse
from ampfit import Fitter
from ampfit.rho_observables import read_polar

# ── a1(1260)± B-decay coupling names (single wave, L = 1) ─────────
A1_NAMES = {
    "a_plus":    "B->a1(1260)p.pim2_g_ls_0",
    "abar_plus": "B->a1(1260)p.pim2_g_lsbar_0",
    "a_minus":   "B->a1(1260)m.pip2_g_ls_0",
    "abar_minus": "B->a1(1260)m.pip2_g_lsbar_0",
}


def get_a1_couplings(resolved):
    """Extract the complex a1(1260)± B-decay couplings."""
    return {k: read_polar(resolved, name) for k, name in A1_NAMES.items()}


def _weak_phase(ratio):
    """½·arg(ratio) mod π, in radians (the weak phase)."""
    return (math.atan2(ratio.imag, ratio.real) / 2.0) % math.pi


def _weak_phase_deg(ratio):
    return _weak_phase(ratio) * 180.0 / math.pi


def get_error(f, x, cov, eps=1e-5):
    """Scalar observable → (value, error) via finite differences."""
    f0 = f(x)
    g = np.empty(len(x))
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    var = g @ cov @ g
    return float(f0), math.sqrt(max(var, 0.0))


def fmt_val_err(v, e):
    if e == 0 or not np.isfinite(e):
        return f"${v:.3f}$"
    h = -int(math.floor(math.log10(abs(e)))) if e > 0 else 2
    nd = max(h + 1, 0)
    return f"${{{v:.{nd}f}}}\\pm{{{e:.{nd}f}}}$"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a1(1260)± observables from fit results")
    parser.add_argument("results", help="Path to save_params JSON file")
    parser.add_argument("--config", default="config_angle.yml")
    parser.add_argument("--constraints", default=None)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--tex", default=None)
    args = parser.parse_args()

    fitter = Fitter(args.config)
    fitter.apply_constrains()
    import json as _json
    with open(args.results) as _fh:
        fitter.load_fixed_from_dict(_json.load(_fh))
    fitter.set_fixed({"delta_gamma": 0.0, "delta_m": 0.506, "A_prod": 0.0,
                      "poqr": 1.0, "poqi": 0.0}, reset=False)
    if args.constraints:
        fitter.load_constraints(args.constraints)

    res = fitter.load_results(args.results)
    x = res.x
    cov = res.hess_inv

    def _c(xf):
        _, resolved = fitter.build_params(xf)
        return get_a1_couplings(resolved)

    obs = {}
    obs["alpha_plus"]  = lambda xf: _weak_phase_deg(_c(xf)["abar_plus"] / _c(xf)["a_plus"])
    obs["lambda_plus"] = lambda xf: abs(_c(xf)["abar_plus"] / _c(xf)["a_plus"])
    obs["alpha_minus"] = lambda xf: _weak_phase_deg(_c(xf)["abar_minus"] / _c(xf)["a_minus"])
    obs["lambda_minus"] = lambda xf: abs(_c(xf)["abar_minus"] / _c(xf)["a_minus"])

    print("=" * 70)
    print(f"a1(1260)± observables from {args.results}")
    print("=" * 70)

    results = {}
    # (key, console label, console desc, tex label, tex desc)
    rows = [
        ("lambda_plus",  "|λ₊|",    "a1(1260)⁺  B→a1π  |ā/a|",
         r"$|\lambda_{+}|$",
         r"$B\to a_1(1260)^{+}\pi$, $|\bar{a}/a|$"),
        ("alpha_plus",   "α₊ (deg)", "a1(1260)⁺  B→a1π  ½arg(ā/a)",
         r"$\alpha_{+}$ (deg)",
         r"$B\to a_1(1260)^{+}\pi$, "
         r"$\frac{1}{2}\arg(\bar{a}/a)$"),
        ("lambda_minus", "|λ₋|",    "a1(1260)⁻  B→a1π  |ā/a|",
         r"$|\lambda_{-}|$",
         r"$B\to a_1(1260)^{-}\pi$, $|\bar{a}/a|$"),
        ("alpha_minus",  "α₋ (deg)", "a1(1260)⁻  B→a1π  ½arg(ā/a)",
         r"$\alpha_{-}$ (deg)",
         r"$B\to a_1(1260)^{-}\pi$, "
         r"$\frac{1}{2}\arg(\bar{a}/a)$"),
    ]
    for key, label, desc, *_ in rows:
        fn = obs[key]
        if cov is not None:
            val, err = get_error(fn, x, cov, eps=args.eps)
        else:
            val, err = fn(x), 0.0
        results[key] = (val, err)
        print(f"  {label:8s} = {val:9.4f} ± {err:9.4f}   ({desc})")

    if args.tex:
        lines = [r"\documentclass{standalone}", r"\begin{document}",
                 r"\begin{tabular}{|c|c|c|}", r"\hline",
                 "observable & value & description \\\\", r"\hline"]
        for key, label, desc, tex_label, tex_desc in rows:
            v, e = results[key]
            lines.append(f"{tex_label} & {fmt_val_err(v, e)} & {tex_desc} \\\\")
        lines += [r"\hline", r"\end{tabular}", r"\end{document}"]
        with open(args.tex, "w") as f:
            f.write("\n".join(lines))
        print(f"\n  LaTeX table saved to {args.tex}")

    print("=" * 70)


if __name__ == "__main__":
    main()


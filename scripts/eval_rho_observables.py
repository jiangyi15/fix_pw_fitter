#!/usr/bin/env python3
"""
Evaluate B→ρA.ρB observables with uncertainty propagation.

Matching ``eval_ratio32.py`` reference: computes helicity fractions,
CP asymmetries, weak phases, and λ ratios — all with uncertainties
from the fit covariance sub-matrix.

Usage:
    python scripts/eval_rho_observables.py fit_results.json \\
        --config config_angle.yml --constraints results_constraints.json

    # Output LaTeX table
    python scripts/eval_rho_observables.py fit_results.json \\
        --config config_angle.yml --constraints results_constraints.json \\
        --tex rho_params.tex
"""
import sys, os, math, json
import numpy as np

_SRC = os.path.join(os.path.dirname(__file__), '..')
_SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)
sys.path.insert(0, _SCR)

import argparse
from ampfit import Fitter
from ampfit.bw_form_factor import form_factor as bw_form_factor
from ampfit.rho_observables import helicity_amplitudes, get_rho_couplings


# ═══════════════════════════════════════════════════════════════════
# Finite-difference gradient helpers
# ═══════════════════════════════════════════════════════════════════

def get_error(f, x, cov, eps=1e-5):
    """Scalar observable.   Returns (value, error)."""
    f0 = f(x)
    g = np.empty(len(x))
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    var = g @ cov @ g
    return float(f0), math.sqrt(max(var, 0.0))


def get_error_matrix(f, x, cov, eps=1e-5):
    """Vector observable.   Returns (values, covariance)."""
    f0 = f(x)
    n_obs = len(f0)
    # Gradient matrix: (n_obs, n_params)
    G = np.empty((n_obs, len(x)))
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        G[:, i] = (f(xp) - f(xm)) / (2 * eps)
    cov_obs = G @ cov @ G.T
    return f0, cov_obs


# ═══════════════════════════════════════════════════════════════════
# Observable builders
# ═══════════════════════════════════════════════════════════════════

def build_observables_x(fitter, R=3.0):
    """Return a dict of observable functions ``f(x_flat) → scalar/array``.

    All use finite-difference gradients on the flat x vector through
    the full constraint pipeline.
    """

    def _resolve(xf):
        _, resolved, _, _ = fitter._build_params(xf)
        return helicity_amplitudes(resolved, R=R)

    # ── λ ratios  (|ā/a|) ──────────────────────────────────────────
    def lambda_S(xf):
        h = _resolve(xf); return abs(h["ab0"] / h["a0"])
    def lambda_P(xf):
        h = _resolve(xf); return abs(h["aperpb"] / h["aperp"])
    def lambda_D(xf):
        h = _resolve(xf); return abs(h["aparab"] / h["apara"])

    # ── λ phases (degrees) ─────────────────────────────────────────
    def _phase_ratio(num, den):
        ratio = num / den
        # Add π if real part negative, to match reference
        extra = 0.0 if ratio.real >= 0 else math.pi
        return ((math.atan2(ratio.imag, ratio.real) + extra) % (2*math.pi)) * 180 / math.pi / 2

    def angle_S(xf):
        h = _resolve(xf); return _phase_ratio(h["ab0"], h["a0"])
    def angle_P(xf):
        h = _resolve(xf); return _phase_ratio(h["aperpb"], h["aperp"])
    def angle_D(xf):
        h = _resolve(xf); return _phase_ratio(h["aparab"], h["apara"])

    # ── Individual fractions ───────────────────────────────────────
    def Abarall(xf):
        h = _resolve(xf)
        return np.array([
            abs(h["a0"])**2 / h["norm"],
            abs(h["aperp"])**2 / h["norm"],
            abs(h["apara"])**2 / h["norm"],
            abs(h["ab0"])**2 / h["normb"],
            abs(h["aperpb"])**2 / h["normb"],
            abs(h["aparab"])**2 / h["normb"],
            (abs(h["ab0"])**2 + abs(h["a0"])**2) / (h["normb"] + h["norm"]),
        ])

    # ── Combined fractions (B + Bbar) ──────────────────────────────
    def Aratioall(xf):
        h = _resolve(xf)
        total = h["norm"] + h["normb"]
        paratot = (abs(h["apara"])**2 + abs(h["aparab"])**2 +
                   abs(h["aperp"])**2 + abs(h["aperpb"])**2)
        return np.array([
            (abs(h["a0"])**2 + abs(h["ab0"])**2) / total,       # 0: helicity 0
            (abs(h["aperp"])**2 + abs(h["aperpb"])**2) / total, # 1: helicity ⟂
            (abs(h["apara"])**2 + abs(h["aparab"])**2) / total, # 2: helicity ∥
            (abs(h["gs"])**2 + abs(h["gsb"])**2) / total,       # 3: LS S-wave
            (abs(h["gp"])**2 + abs(h["gpb"])**2) / total,       # 4: LS P-wave
            (abs(h["gd"])**2 + abs(h["gdb"])**2) / total,       # 5: LS D-wave
            (abs(h["apara"])**2 + abs(h["aparab"])**2) /
            (paratot + 1e-30),                                   # 6: ∥/(⟂+∥)
        ])

    # ── λ vector (matching Lambdaall) ──────────────────────────────
    def Lambdaall(xf):
        h = _resolve(xf)
        return np.array([
            abs(h["ab0"] / h["a0"]),           # 0: |λ_0|
            abs(h["aperpb"] / h["aperp"]),     # 1: |λ_⟂|
            abs(h["aparab"] / h["apara"]),     # 2: |λ_∥|
            abs(h["gsb"] / h["gs"]),           # 3: |λ_S|
            abs(h["gpb"] / h["gp"]),           # 4: |λ_P|
            abs(h["gdb"] / h["gd"]),           # 5: |λ_D|
        ])

    # ── Angles (degrees, matching Angleall) ────────────────────────
    def Angleall(xf):
        h = _resolve(xf)
        def angle_deg(z):
            return (math.atan2(z.imag, z.real) % (2*math.pi)) * 180 / math.pi / 2
        return np.array([
            angle_deg(h["ab0"] / h["a0"]),     # 0: α_0
            angle_deg(h["aperpb"] / h["aperp"]),   # 1: α_⟂
            angle_deg(h["aparab"] / h["apara"]),   # 2: α_∥
            angle_deg(h["gsb"] / h["gs"]),     # 3: α_S
            angle_deg(h["gpb"] / h["gp"]),     # 4: α_P
            angle_deg(h["gdb"] / h["gd"]),     # 5: α_D
        ])

    # ── CP asymmetry ───────────────────────────────────────────────
    def Acpall(xf):
        h = _resolve(xf)
        def acp(sq_b, sq_bbar):
            return (sq_bbar - sq_b) / (sq_bbar + sq_b + 1e-30)
        return np.array([
            acp(abs(h["a0"])**2, abs(h["ab0"])**2),         # 0: A^CP_0
            acp(abs(h["aperp"])**2, abs(h["aperpb"])**2),   # 1: A^CP_⟂
            acp(abs(h["apara"])**2, abs(h["aparab"])**2),   # 2: A^CP_∥
            acp(abs(h["gs"])**2, abs(h["gsb"])**2),         # 3: A^CP_S
            acp(abs(h["gp"])**2, abs(h["gpb"])**2),         # 4: A^CP_P
            acp(abs(h["gd"])**2, abs(h["gdb"])**2),         # 5: A^CP_D
            (h["normb"] - h["norm"]) / (h["normb"] + h["norm"] + 1e-30),  # 6: total
        ])

    # ── CS parameters (cos2β, sin2β from λ₀) ───────────────────────
    def CSall(xf):
        h = _resolve(xf)
        l0 = h["ab0"] / h["a0"]
        lab = abs(l0)
        return np.array([
            (1 - lab**2) / (1 + lab**2),
            2 * l0.imag / (1 + lab**2),
        ])

    # ── Phase differences relative to gs (degrees) ─────────────────
    # [0-2] helicity amplitudes relative to gs (LS S-wave)
    # [3-5] LS couplings relative to gs
    # [6-7] (apara/a0) mod 2π for B and Bbar
    def Deltaall(xf):
        h = _resolve(xf)
        def rel_phase(num, den):
            z = num / den
            return math.atan2(z.imag, z.real) * 180 / math.pi
        return np.array([
            rel_phase(h["a0"], h["gs"]),           # 0: arg(a0/gs)
            rel_phase(h["aperp"], h["gs"]),        # 1: arg(aperp/gs)
            rel_phase(h["apara"], h["gs"]),        # 2: arg(apara/gs)
            rel_phase(h["gs"], h["gs"]),           # 3: arg(gs/gs) = 0
            rel_phase(h["gp"], h["gs"]),           # 4: arg(gp/gs)
            rel_phase(h["gd"], h["gs"]),           # 5: arg(gd/gs)
            rel_phase(h["apara"], h["a0"]) % 360,  # 6: δ_∥ (B)
            rel_phase(h["aparab"], h["ab0"]) % 360, # 7: δ_∥bar (Bbar)
        ])

    # ── Amplitude ratios |A_i / A_S| ────────────────────────────────
    # [0-2] helicity amplitudes |A_i|/|gs|
    # [3-5] LS couplings |g_i|/|gs|
    def Aall(xf):
        h = _resolve(xf)
        gs_abs = abs(h["gs"]) + 1e-30
        return np.array([
            abs(h["a0"]) / gs_abs,           # 0: |a0/gs|
            abs(h["aperp"]) / gs_abs,        # 1: |aperp/gs|
            abs(h["apara"]) / gs_abs,        # 2: |apara/gs|
            1.0,                              # 3: |gs/gs| = 1
            abs(h["gp"]) / gs_abs,           # 4: |gp/gs|
            abs(h["gd"]) / gs_abs,           # 5: |gd/gs|
        ])

    return dict(
        lambda_S=lambda_S, lambda_P=lambda_P, lambda_D=lambda_D,
        angle_S=angle_S, angle_P=angle_P, angle_D=angle_D,
        Abarall=Abarall, Aratioall=Aratioall,
        Lambdaall=Lambdaall, Angleall=Angleall,
        Acpall=Acpall, CSall=CSall,
        Deltaall=Deltaall, Aall=Aall,
    )


# ═══════════════════════════════════════════════════════════════════
# LaTeX formatting
# ═══════════════════════════════════════════════════════════════════

def fmt_val_err(v, e):
    """Format value±error with adaptive decimal places."""
    if e == 0:
        return f"${v:.3f}$(fixed)"
    if not np.isfinite(e) or e <= 0:
        return f"${v:.3f}$"
    h = -int(math.floor(math.log10(abs(e)))) if e > 0 else 2
    nd = max(h + 1, 0)
    return f"${{{v:.{nd}f}}}\\pm{{{e:.{nd}f}}}$"


def generate_latex_table(obs, name_list, labels, output_path):
    """Generate a LaTeX table with the observables."""
    lines = []
    lines.append(r"\documentclass{standalone}")
    lines.append(r"\begin{document}")
    lines.append(r"\begin{tabular}{|c|c|c|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append("basis & $|\\lambda_{i}|$ & $\\alpha_{i}$ & "
                 "$A^{{CP}}_{{i}}$ & ratio & $|A_{i}/A_{{S}}|$ & "
                 "$\\arg(A_{i}/A_{{S}})$ \\\\")
    lines.append(r"\hline")

    axis_labels = ["0", "\\perp", "\\parallel", "S", "P", "D"]

    for i, idx in enumerate(name_list):
        if idx == 3:
            lines.append(r"\hline")
        lam = fmt_val_err(obs["Lambdaall"][0][idx], obs["Lambdaall"][1][idx])
        ang = fmt_val_err(obs["Angleall"][0][idx], obs["Angleall"][1][idx])
        acp = fmt_val_err(obs["Acpall"][0][idx], obs["Acpall"][1][idx])
        rat = fmt_val_err(obs["Aratioall"][0][idx], obs["Aratioall"][1][idx])
        amp = fmt_val_err(obs["Aall"][0][idx], obs["Aall"][1][idx])
        del_ang = fmt_val_err(obs["Deltaall"][0][idx], obs["Deltaall"][1][idx])
        lines.append(f"${axis_labels[i]}$ & {lam} & {ang} & {acp} & "
                     f"{rat} & {amp} & {del_ang} \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{document}")

    tex = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(tex)
    print(f"  LaTeX table saved to {output_path}")
    return tex


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate B→ρA.ρB observables from fit results")
    parser.add_argument("results", help="Path to save_params JSON file")
    parser.add_argument("--config", default="config_angle.yml")
    parser.add_argument("--constraints", default=None,
                        help="Constraints JSON from save_constraints()")
    parser.add_argument("--radius", type=float, default=3.0,
                        help="Meson radius (GeV⁻¹)")
    parser.add_argument("--eps", type=float, default=1e-5,
                        help="FD step size")
    parser.add_argument("--tex", default=None,
                        help="Output LaTeX table path")
    args = parser.parse_args()

    # ── 1. Setup fitter ───────────────────────────────────────
    fitter = Fitter(args.config)
    from run_fit import build_constraints
    fixed_slots, same_params, scale_params = build_constraints(fitter.all_comb)
    for name in ["delta_gamma", "delta_m", "A_prod", "poqr", "poqi"]:
        fixed_slots[name] = 0.0 if name != "delta_m" else 0.506
    fixed_slots["poqr"] = 1.0
    fitter.set_fixed(fixed_slots)
    fitter.set_same(same_params)
    fitter.set_scale(scale_params)
    if args.constraints:
        fitter.load_constraints(args.constraints)

    # ── 2. Load results ────────────────────────────────────────
    res = fitter.load_results(args.results)
    x = res.x
    cov = res.hess_inv

    # Try to use rhorho sub-matrix from status
    with open(args.results) as f:
        raw = json.load(f)
    rhorho_info = raw.get("status", {}).get("rhorho")

    # Build observable functions
    obs_builder = build_observables_x(fitter, R=args.radius)

    scalar_obs = [
        ("lambda_S", "λ_S", r"$|\lambda_S|$"),
        ("lambda_P", "λ_P", r"$|\lambda_P|$"),
        ("lambda_D", "λ_D", r"$|\lambda_D|$"),
        ("angle_S", "φ_S (deg)", r"$\phi_S$"),
        ("angle_P", "φ_P (deg)", r"$\phi_P$"),
        ("angle_D", "φ_D (deg)", r"$\phi_D$"),
    ]

    print("=" * 70)
    print(f"B→ρA.ρB observables from {args.results}")
    print("=" * 70)

    results = {}

    # ── Scalar observables ─────────────────────────────────────
    for key, label, _ in scalar_obs:
        fn = obs_builder[key]
        val, err = get_error(fn, x, cov, eps=args.eps) if cov is not None else (fn(x), 0.0)
        results[key] = (val, err)
        if cov is not None:
            print(f"  {label:12s} = {val:.4f} ± {err:.4f}")
        else:
            print(f"  {label:12s} = {val:.4f}  (no error)")

    print()

    # ── Vector observables (6-component: 0, ⟂, ∥, S, P, D) ────
    axis = ["0", "\\perp", "\\parallel", "S", "P", "D"]
    vector_obs = [
        ("Abarall",    "Fraction"),
        ("Aratioall",  "Ratio"),
        ("Lambdaall",  "|λ|"),
        ("Angleall",   "α (deg)"),
        ("Acpall",     "A_CP"),
        ("Deltaall",   "δ (deg)"),
        ("Aall",       "|A/A_S|"),
    ]
    # CSall is 2-component
    cs_names = ["cos(2β)", "sin(2β)"]

    for key, label in vector_obs:
        fn = obs_builder[key]
        if cov is not None:
            vals, cov_obs = get_error_matrix(fn, x, cov, eps=args.eps)
            errs = np.sqrt(np.maximum(np.diag(cov_obs), 0))
        else:
            vals = fn(x); errs = np.zeros_like(vals)

        results[key] = (vals, errs)
        for i in range(len(axis)):
            if cov is not None:
                print(f"  {label:10s} {axis[i]:8s} = {vals[i]:.6f} ± {errs[i]:.6f}")
            else:
                print(f"  {label:10s} {axis[i]:8s} = {vals[i]:.6f}")

    # ── CSall (2-component) ────────────────────────────────────
    fn = obs_builder["CSall"]
    if cov is not None:
        cs_vals, cs_cov = get_error_matrix(fn, x, cov, eps=args.eps)
        cs_errs = np.sqrt(np.maximum(np.diag(cs_cov), 0))
    else:
        cs_vals = fn(x); cs_errs = np.zeros_like(cs_vals)
    results["CSall"] = (cs_vals, cs_errs)
    for i in range(2):
        if cov is not None:
            print(f"  CS         {cs_names[i]:8s} = {cs_vals[i]:.6f} ± {cs_errs[i]:.6f}")
        else:
            print(f"  CS         {cs_names[i]:8s} = {cs_vals[i]:.6f}")

    # ── λ summary matching eval_ratio32 ────────────────────────
    print()
    print("  λ summary:")
    for key, label, _ in scalar_obs[:3]:
        val, err = results[key]
        ang_key = key.replace("lambda_", "angle_")
        ang_val, ang_err = results[ang_key]
        if cov is not None:
            print(f"    {label} = ({val:.2f}±{err:.2f})·exp(i({ang_val:.1f}±{ang_err:.1f})°)")
        else:
            print(f"    {label} = ({val:.2f})·exp(i({ang_val:.1f})°)")

    # ── Save JSON ──────────────────────────────────────────────
    out_json = os.path.splitext(args.results)[0] + "_rho_obs.json"
    json_out = {}
    for key, (vals, errs) in results.items():
        if np.ndim(vals) == 0:
            json_out[key] = {"value": float(vals), "error": float(errs)}
        else:
            json_out[key] = {"value": vals.tolist(), "error": errs.tolist()}
    # Also save rhorho info
    if rhorho_info:
        json_out["rhorho_params"] = rhorho_info[0]
        json_out["rhorho_cov"] = rhorho_info[1]
    with open(out_json, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n  Saved JSON to {out_json}")

    # ── LaTeX table ────────────────────────────────────────────
    if args.tex:
        # Use the 6-component vector observables
        obs_for_table = {}
        for key in ["Lambdaall", "Angleall", "Acpall", "Aratioall", "Aall", "Deltaall"]:
            vals, errs = results[key]
            obs_for_table[key] = (vals, errs)

        generate_latex_table(obs_for_table, list(range(6)), axis, args.tex)

    print("=" * 70)


if __name__ == "__main__":
    main()

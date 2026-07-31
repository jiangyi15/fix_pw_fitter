#!/usr/bin/env python3
"""
Compute B→ρA.ρB observables from fit results with uncertainty propagation.

Reads a ``save_params()`` JSON (via ``Fitter.load_results()``) and
evaluates:

  fL   = S-wave (longitudinal) fraction
  phi  = weak phase  φ = ½ arg(ā₀/a₀)  (mod π)
  acp  = CP asymmetry  (|ā₀|² − |a₀|²) / (|ā₀|² + |a₀|²)

Uncertainties are propagated from the fit covariance matrix via
finite-difference gradients on the flat x vector.

Usage:
    python scripts/calc_observables.py fit_results.json \\
        --config config_angle.yml

    # With a custom constraint file
    python scripts/calc_observables.py fit_results.json \\
        --config config_angle.yml --constraints my_cons.json

    # Also save as JSON
    python scripts/calc_observables.py fit_results.json \\
        --output observables.json
"""
import sys, os, math, json
import numpy as np

_SRC = os.path.join(os.path.dirname(__file__), '..')
_SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)
sys.path.insert(0, _SCR)

from ampfit import Fitter
from ampfit.rho_observables import (helicity_amplitudes,
    obs_fL, obs_weak_phase, obs_cp_asym)


def finite_diff_gradient(f, x, eps=1e-5):
    """Compute df/dx via central finite differences."""
    g = np.empty(len(x))
    f0 = f(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    return g


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute B→ρA.ρB observables from fit results")
    parser.add_argument("results", help="Path to save_params JSON file")
    parser.add_argument("--config", default="config_angle.yml",
                        help="Config YAML (must match the fit)")
    parser.add_argument("--constraints", default=None,
                        help="Constraints JSON (use if not in fitter defaults)")
    parser.add_argument("--radius", type=float, default=3.0,
                        help="Meson radius for BW barrier factors (GeV⁻¹)")
    parser.add_argument("--output", default=None,
                        help="Save observables to JSON")
    parser.add_argument("--eps", type=float, default=1e-5,
                        help="Step size for finite-difference gradients")
    args = parser.parse_args()

    # ── 1. Setup fitter with same constraints ──────────────────────
    fitter = Fitter(args.config)

    fitter.apply_constrains()
    fitter.set_fixed({"delta_gamma": 0.0, "delta_m": 0.506, "A_prod": 0.0,
                      "poqr": 1.0, "poqi": 0.0}, reset=False)

    if args.constraints:
        fitter.load_constraints(args.constraints)

    # ── 2. Load results via Fitter.load_results ─────────────────────
    res = fitter.load_results(args.results)
    x = res.x
    cov = res.hess_inv  # None if not found
    has_errors = cov is not None

    # ── 3. Compute observables ─────────────────────────────────────
    obs_list = [("fL",  obs_fL,  "longitudinal fraction"),
                ("phi", obs_weak_phase, "weak phase"),
                ("acp", obs_cp_asym,   "CP asymmetry")]

    def _make_fn(obs_func):
        return lambda xf: obs_func(fitter.build_params(xf)[1], R=args.radius)

    print("=" * 70)
    print(f"Observables from {args.results}")
    if not has_errors:
        print("  (no error matrix — uncertainties unavailable)")
    print("=" * 70)

    output = {}

    for name, obs_func, label in obs_list:
        fn = _make_fn(obs_func)
        value = fn(x)

        if has_errors:
            grad = finite_diff_gradient(fn, x, eps=args.eps)
            var = grad @ cov @ grad
            err = math.sqrt(max(var, 0.0))
        else:
            err = 0.0

        output[name] = {"value": float(value), "error": float(err)}

        if has_errors:
            print(f"  {name:8s} = {value:+.8f} ± {err:.6f}   ({label})")
        else:
            print(f"  {name:8s} = {value:+.8f}   ({label}, no error)")

    # ── 4. Helicity amplitudes ─────────────────────────────────────
    _, resolved = fitter.build_params(x)
    h = helicity_amplitudes(resolved, R=args.radius)
    a0, ap, am = h["a0"], h["aperp"], h["apara"]
    ab0, abp, abm = h["ab0"], h["aperpb"], h["aparab"]

    print(f"\n  Helicity amplitudes at best fit:")
    for name_h, val_h in [("a₀", a0), ("a₊", ap), ("a₋", am),
                          ("ā₀", ab0), ("ā₊", abp), ("ā₋", abm)]:
        print(f"    |{name_h}| = {abs(val_h):.6f},  arg = {math.atan2(val_h.imag, val_h.real):+.6f}")

    # ── 5. Save ────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved to {args.output}")

    print("=" * 70)


if __name__ == "__main__":
    main()

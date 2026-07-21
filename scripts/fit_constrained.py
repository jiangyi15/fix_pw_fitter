#!/usr/bin/env python3
"""
Constrained fit: minimize NLL subject to f(x) = 0.

Two-step approach matching pw_cfit5_td6_fix29_cons.py:
  1. Penalty step:  minimize  NLL + w * (c(x) - target)²
  2. SLSQP step:    minimize  NLL  s.t.  c(x) - target = 0

Usage:
    # Single constraint: S-wave fraction = 0.7
    python scripts/fit_constrained.py --constraint fL:0.7

    # Single constraint: weak phase = 1.2
    python scripts/fit_constrained.py --constraint phi:1.2

    # Single constraint: CP asymmetry = 0.0
    python scripts/fit_constrained.py --constraint acp:0.0

    # Multi-constraint: S-wave fraction + weak phase (matching cons2/3)
    python scripts/fit_constrained.py --constraint fL:0.7,phi:1.2

    # Multi-constraint: CP asymmetry + weak phase (matching cons6)
    python scripts/fit_constrained.py --constraint acp:0.1,phi:1.2

    # Multi-constraint: CP asymmetry + S-wave fraction (matching cons7)
    python scripts/fit_constrained.py --constraint acp:0.1,fL:0.7

    # Short form with --target (single obs only)
    python scripts/fit_constrained.py --constraint fL --target 0.7

    # S-wave LS fraction (|gs|²/total)
    python scripts/fit_constrained.py --constraint fS:0.35

Constraints use the Blatt-Weisskopf form factor from ampfit's
``build_fl_table`` (F_L(q) = q^L · B'_L(q), normalised so that
F_L(1 GeV) = 1).  The Jacobian is computed via finite differences.
"""
import sys, os, json, time, argparse, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ampfit import Fitter
from ampfit.rho_observables import (helicity_amplitudes,
    obs_fL, obs_fS, obs_weak_phase, obs_cp_asym)


# Map observable name → (fun, description)
_OBSERVABLES = {
    "fL": (obs_fL, "longitudinal fraction f_L"),
    "fS": (obs_fS, "S-wave LS fraction f_S"),
    "phi": (obs_weak_phase, "weak phase"),
    "acp": (obs_cp_asym, "CP asymmetry"),
}


# ======================================================================
# Constraint builder
# ======================================================================

def build_constraint(fitter, obs_name, target, R=3.0):
    """Return (fun, jac) for a single-observable equality constraint.

    ``fun(x) = obs(resolved) - target``
    ``jac(x) = d(fun)/dx`` via finite differences.

    Args:
        fitter: Fitter instance (needed for ``build_params``).
        obs_name: observable name ('fL', 'phi', 'acp').
        target: target value (scalar).
        R: meson radius for BW barrier factors (GeV⁻¹).

    Returns:
        (fun, jac) callables for ``scipy.optimize.minimize``.
    """
    if obs_name not in _OBSERVABLES:
        raise ValueError(f"Unknown observable '{obs_name}'. "
                         f"Choose from: {list(_OBSERVABLES)}")

    obs_fn = _OBSERVABLES[obs_name][0]

    def fun(x):
        _, resolved = fitter.build_params(x)
        return obs_fn(resolved, R=R) - target

    def jac(x):
        eps = 1e-5
        f0 = fun(x)
        g = np.empty(len(x))
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            g[i] = (fun(xp) - f0) / eps
        return g

    return fun, jac


def build_multi_constraint(fitter, obs_targets, R=3.0):
    """Return (fun, jac) for a multi-observable constraint.

    ``obs_targets`` is a list of ``(obs_name, target)`` tuples, e.g.::

        [("fL", 0.7), ("phi", 1.2)]

    ``fun(x)`` returns an array ``[fL(x)-t1, phi(x)-t2, ...]``.
    ``jac(x)`` returns the Jacobian matrix ``(n_obs, n_params)``.
    """
    obs_list = []
    for obs_name, target in obs_targets:
        if obs_name not in _OBSERVABLES:
            raise ValueError(f"Unknown observable '{obs_name}'")
        obs_list.append((_OBSERVABLES[obs_name][0], target))

    def fun(x):
        _, resolved = fitter.build_params(x)
        return np.array([fn(resolved, R=R) - t for fn, t in obs_list])

    def jac(x):
        eps = 1e-5
        f0 = fun(x)
        n_obs = len(obs_list)
        g = np.empty((n_obs, len(x)))
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            g[:, i] = (fun(xp) - f0) / eps
        return g if n_obs > 1 else g[0]

    return fun, jac


def build_penalty_objective(fitter, constraint_fun, constraint_jac,
                            weight=10.0):
    """Return (fun, jac) for penalty-method objective.

    The constraint function already returns ``c(x) = observable(x) - target``
    (i.e. f(x) = 0 for the equality constraint).  The penalty is simply::

        F(x) = NLL(x) + weight * c(x)²
        dF/dx = dNLL/dx + 2 * weight * c(x) * dc/dx
    """
    def fun_jac(x):
        nll, grad = fitter.get_nll(x)
        c = constraint_fun(x)
        penalty = weight * c**2
        grad_c = constraint_jac(x)
        grad_total = grad + 2.0 * weight * c * grad_c
        return nll + penalty, grad_total.astype(np.float64)

    return fun_jac


# ======================================================================
# Main driver
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Constrained fit with penalty + SLSQP")
    parser.add_argument("--config", default="config_angle.yml")
    parser.add_argument("--data", default="data/data_arrays.npz")
    parser.add_argument("--phsp", default="data/phsp_arrays.npz")
    parser.add_argument("--backend", default="cuda_v3")
    parser.add_argument("--constraint", default="fL",
                        help="Constraint (comma-sep list of obs[:target]). "
                             "Observables: fL (longitudinal frac), "
                             "fS (S-wave LS frac), phi (weak phase), "
                             "acp (CP asym).  Examples: 'fL:0.7', "
                             "'fL:0.7,phi:1.2'")
    parser.add_argument("--target", type=float, default=None,
                        help="Single-observable target (alternative to :target in --constraint)")
    parser.add_argument("--radius", type=float, default=3.0,
                        help="Meson radius for BW barrier factors (GeV⁻¹)")
    parser.add_argument("--penalty-weight", type=float, default=10.0,
                        help="Weight for penalty step")
    parser.add_argument("--n-starts", type=int, default=10,
                        help="Number of random starts")
    parser.add_argument("--maxiter", type=int, default=500,
                        help="Max iterations per optimizer call")
    parser.add_argument("--maxiter-penalty", type=int, default=None,
                        help="Max iterations for penalty step (default: maxiter)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--output", default=None,
                        help="Output dir (default: ./fit_constrained/)")
    parser.add_argument("--format", default="png",
                        help="Image format for NLL progress plot (default: png)")
    parser.add_argument("--debug", action="store_true",
                        help="Use 1K data / 10K phsp")
    args = parser.parse_args()

    # Default output directory from constraint spec
    if args.output is None:
        safe_name = args.constraint.replace(",", "_").replace(":", "_")
        args.output = f"fit_{safe_name}"
    os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    maxiter_penalty = args.maxiter_penalty or args.maxiter

    # ================================================================
    # 1. Setup fitter with base constraints (same as run_fit.py)
    # ================================================================
    print("=" * 70)
    print("SETUP")
    print("=" * 70)

    # Import and use the constraint builder from run_fit
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from run_fit import build_constraints

    fitter = Fitter(args.config, backend=args.backend)
    fixed_slots, same_params, scale_params = build_constraints(fitter.all_comb)

    # gamma is free; fix other time params
    for name in ["delta_gamma", "delta_m", "A_prod", "poqr", "poqi"]:
        fixed_slots[name] = 0.0 if name != "delta_m" else 0.506
    fixed_slots["poqr"] = 1.0

    fitter.set_fixed(fixed_slots)
    fitter.set_same(same_params)
    fitter.set_scale(scale_params)

    n_free = len(fitter.free_param_names())
    print(f"Fixed: {len(fixed_slots)} slots, Same: {len(same_params)} groups")
    print(f"Free params: {n_free}")

    # ================================================================
    # 2. Load data
    # ================================================================
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    max_data = 1000 if args.debug else None
    max_phsp = 10000 if args.debug else None

    data_np, n_data = Fitter.load_npz(args.data, max_events=max_data)
    phsp_np, n_phsp = Fitter.load_npz(args.phsp, max_events=max_phsp)
    print(f"Loaded {n_data:,} data + {n_phsp:,} phsp")

    fitter.set_phsp(phsp_np)
    fitter.set_data(data_np)

    # ================================================================
    # 3. Parse constraint specification
    # ================================================================
    obs_targets = []
    parts = args.constraint.split(",")
    for part in parts:
        part = part.strip()
        if ":" in part:
            obs_name, target_str = part.split(":", 1)
            obs_targets.append((obs_name.strip(), float(target_str)))
        else:
            if args.target is None:
                raise ValueError(f"--target required for single-observable constraint '{part}'")
            obs_targets.append((part, args.target))

    obs_repr = "+".join(f"{n}={t:.4g}" for n, t in obs_targets)
    print("\n" + "=" * 70)
    print(f"CONSTRAINT: {obs_repr}")
    print("=" * 70)

    if len(obs_targets) == 1:
        cons_fun, cons_jac = build_constraint(fitter, obs_targets[0][0],
                                               obs_targets[0][1], R=args.radius)
    else:
        cons_fun, cons_jac = build_multi_constraint(fitter, obs_targets,
                                                     R=args.radius)

    penalty_fun_jac = build_penalty_objective(
        fitter, cons_fun, cons_jac, weight=args.penalty_weight)

    # ================================================================
    # 4. Multi-start constrained fit
    # ================================================================
    print("\n" + "=" * 70)
    print(f"MULTI-START FIT ({args.n_starts} starts)")
    print("=" * 70)

    from scipy.optimize import minimize

    constraint = [{"type": "eq", "fun": cons_fun, "jac": cons_jac}]

    best_nll = None
    all_nll = []
    all_results = []

    for idx in range(args.n_starts):
        t_start = time.time()
        print(f"\n--- Start {idx + 1}/{args.n_starts} ---")

        # Random initial point
        x = fitter.initial_values(seed=args.seed + idx)

        # ---- Step 1: Penalty method ----
        ret_penalty = minimize(
            penalty_fun_jac, x, jac=True, method="BFGS",
            options={"maxiter": maxiter_penalty, "gtol": 1e-3, "disp": False})
        x_penalty = ret_penalty.x

        # ---- Step 2: SLSQP constrained ----
        ret = fitter.fit_constrained(
            x_penalty, maxiter=args.maxiter, constraints=constraint, disp=False)

        t_elapsed = time.time() - t_start

        # Evaluate results
        nll_final = float(ret.fun)
        cons_val = cons_fun(ret.x)
        if np.ndim(cons_val) == 0:
            cons_val_str = f"{cons_val:.8f}"
            cons_val_list = [float(cons_val)]
        else:
            cons_val_str = str([f"{v:.6f}" for v in cons_val])
            cons_val_list = [float(v) for v in cons_val]

        print(f"  NLL = {nll_final:.6f}, c(x) = {cons_val_str}, "
              f"success = {ret.success}, time = {t_elapsed:.1f}s")

        # Save individual result
        prefix = os.path.join(args.output, f"start_{idx:03d}")
        fitter.save_params(ret, prefix + ".json")

        # Also save constraints value in status
        with open(prefix + ".json") as f:
            saved = json.load(f)
        saved["status"]["constraint_obs"] = [(n, float(t)) for n, t in obs_targets]
        saved["status"]["constraint_value"] = cons_val_list
        with open(prefix + ".json", "w") as f:
            json.dump(saved, f, indent=2)

        all_nll.append(nll_final)
        all_results.append({
            "index": idx,
            "NLL": nll_final,
            "constraint": cons_val_list,
            "success": bool(ret.success),
            "message": str(ret.message),
        })

        # Track best
        if best_nll is None or nll_final < best_nll - 0.1:
            best_nll = nll_final
            print(f"  ★ New best NLL: {best_nll:.6f}")

        # NLL progress plot (like reference)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.clf()
            plt.scatter(range(len(all_nll)), all_nll,
                        c=["green" if r["success"] else "red"
                          for r in all_results])
            plt.axhline(best_nll, color="gray", linestyle="--", alpha=0.5)
            plt.xlabel("Start index")
            plt.ylabel("NLL")
            plt.title(f"Constrained fit: {args.constraint} = {args.target}")
            plt.ylim(best_nll - 0.5, best_nll + 50)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output, f"nll_progress.{args.format}"),
                        format=args.format, dpi=150, bbox_inches="tight")
            plt.close()
        except ImportError:
            pass

    # ================================================================
    # 5. Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    success_count = sum(1 for r in all_results if r["success"])
    print(f"  Best NLL:     {best_nll:.6f}")
    print(f"  Starts:       {args.n_starts}")
    print(f"  Successful:   {success_count}/{args.n_starts}")
    print(f"  Output dir:   {args.output}")

    summary = {
        "config": args.config,
        "backend": args.backend,
        "constraint": args.constraint,
        "target": args.target,
        "penalty_weight": args.penalty_weight,
        "n_starts": args.n_starts,
        "n_free": n_free,
        "n_data": n_data,
        "n_phsp": n_phsp,
        "best_nll": best_nll,
        "all_results": all_results,
    }
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save best result separately
    best_idx = int(np.argmin(all_nll))
    best_src = os.path.join(args.output, f"start_{best_idx:03d}.json")
    best_dst = os.path.join(args.output, "best_result.json")
    if os.path.exists(best_src):
        import shutil
        shutil.copy(best_src, best_dst)
        print(f"  Best result:  {best_dst}")

    fitter.free()
    print("Done.")


if __name__ == "__main__":
    main()

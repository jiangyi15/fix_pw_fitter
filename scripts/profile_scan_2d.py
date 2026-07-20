#!/usr/bin/env python3
"""Two-parameter likelihood profiling with Gaussian Processes.

Phase 1 -- Conditional scan (fast, ~6s/pt):
  Coarse grid of conditional NLL evaluations (other params fixed).
  Provides initial surface estimate.

Phase 2 -- GP active learning (expensive, ~60-120s/pt):
  GP learns from conditional scan, contour acquisition suggests
  points for profiled fits.  Repeats until convergence.

Usage:
    cd /path/to/analysis
    python scripts/profile_likelihood_2d.py results.json config.yml \\
        --param1 mass --param2 width --output profile
"""

import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ampfit import Fitter


# ── Gaussian Process ───────────────────────────────────────────────

class GP:
    """Gaussian Process with squared-exponential (RBF) kernel."""

    def __init__(self, length_scale=1.0, sigma_f=1.0, sigma_n=1e-4):
        self.length_scale = np.atleast_1d(np.asarray(length_scale, float))
        self.sigma_f = float(sigma_f)
        self.sigma_n = float(sigma_n)
        self._X = None; self._y = None; self._L = None; self._alpha = None

    def _k(self, x1, x2):
        sq = 0.0
        for d in range(x1.shape[1]):
            diff = x1[:, d:d+1] / self.length_scale[d] - x2[:, d:d+1].T / self.length_scale[d]
            sq += diff ** 2
        return self.sigma_f ** 2 * np.exp(-0.5 * sq)

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float).ravel()
        self._X = X; self._y = y
        K = self._k(X, X) + self.sigma_n ** 2 * np.eye(len(X))
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))
        return self

    def predict(self, X, return_std=True):
        X = np.atleast_2d(np.asarray(X, float))
        K_s = self._k(X, self._X)
        mu = K_s @ self._alpha
        if not return_std:
            return mu
        v = np.linalg.solve(self._L, K_s.T)
        var = self._k(X, X) - v.T @ v
        return mu, np.sqrt(np.maximum(np.diag(var), 0))


# ── Acquisition ────────────────────────────────────────────────────

def acquire_contour(mu, std, levels=(2.30, 6.18, 11.83)):
    """Contour-focused acquisition: pick points near likelihood contours.
    acq = max_i  std · φ(|mu − c_i| / std)
    """
    acq = np.zeros_like(mu)
    for c in levels:
        z = np.abs(mu - c) / np.maximum(std, 1e-10)
        acq = np.maximum(acq, std * np.exp(-0.5 * z**2))
    return acq


def quadratic_from_hessian(cov22, v0_1, v0_2):
    """ΔNLL(v) ≈ ½·(v−v₀)ᵀ·H₂₂·(v−v₀) where H₂₂ = cov₂₂⁻¹"""
    H22 = np.linalg.inv(cov22)
    def q(v1, v2):
        dv = np.array([v1 - v0_1, v2 - v0_2])
        return 0.5 * dv @ H22 @ dv
    return q


# ── I/O helpers ────────────────────────────────────────────────────

def load_fit_result(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("value", data), data.get("error", {}), data.get("status", {}).get("NLL")


def phys_to_opt(pname, pval, fitter):
    """Convert physical value to optimizer space (full constraint inverse, not just bounds).

    Matches how values_from_dict reconstructs x_best: runs through cm.inverse()
    so that scale, same, and bound transforms are all properly reversed.
    """
    raw = fitter.cm.inverse({pname: pval})
    return raw.get(pname, pval)


def clamp(pname, val, fitter):
    if pname in fitter.cm.bounds:
        bt = fitter.cm.bounds[pname]
        val = max(bt.a, min(bt.b, val))
    return val


def run_profiled_fit(fitter, param1, param2, v1, v2,
                     x_best_all, free_names_all, maxiter=200, hess_inv_full=None):
    """Fix two params, evaluate NLL at point (no profiling fit — GP phase does that)."""
    fitter.set_fixed({param1: v1, param2: v2})
    free_now = fitter.free_param_names()
    x = np.array([x_best_all[free_names_all.index(n)] for n in free_now])
    try:
        nll, _grad = fitter.get_nll(x)
        return nll, None
    except Exception:
        return np.nan, None


# ── Main ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="2-param likelihood profiling")
    ap.add_argument("fit_json", help="Fit result JSON")
    ap.add_argument("config", help="Fitter config YAML")
    ap.add_argument("--param1", required=True)
    ap.add_argument("--param2", required=True)
    ap.add_argument("--grid", type=int, default=30)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--output", default="profile")
    ap.add_argument("--backend", default="cuda_v3_sparse")
    args = ap.parse_args()

    base = os.path.splitext(args.output)[0]
    pts_dir = base + "_points"
    os.makedirs(pts_dir, exist_ok=True)

    # ── Load fitter / data / constraints ──────────────────────────
    print(f"Loading: {args.fit_json}")
    fitter = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.fit_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        fitter.load_constraints(cp)
        print(f"  constraints: {cp}")
    val_dict, err_dict, nll0_saved = load_fit_result(args.fit_json)
    res = fitter.load_results(args.fit_json)
    x_best = res.x
    hess_inv_full = getattr(res, 'hess_inv', None)
    try:
        fitter.load_all_data()
        print("  data loaded")
    except Exception as e:
        print(f"  WARNING: no data ({e}) — profiled fits will fail")

    free_names = fitter.free_param_names()
    for p in [args.param1, args.param2]:
        if p not in free_names:
            print(f"Error: {p} not free. Available: {free_names[:6]}...")
            sys.exit(1)
    idx1, idx2 = free_names.index(args.param1), free_names.index(args.param2)
    v0_1 = float(val_dict.get(args.param1, 0.0))
    v0_2 = float(val_dict.get(args.param2, 0.0))
    nll0 = nll0_saved or 0.0
    # Verify NLL at best-fit point
    nll_at_best, _ = fitter.get_nll(x_best)
    print(f"  nll0 (from status) = {nll0:.4f}")
    print(f"  nll at x_best      = {nll_at_best:.4f}")
    print(f"  match: {abs(nll_at_best - nll0) < 1.0}")

    # ── Phase 1: Conditional scan ─────────────────────────────────
    # Initial step from conditional Hessian (valid near minimum).
    # The scan is adaptive — step outward with 1.2^k, bracket the contour.
    if hess_inv_full is not None:
        H_full = np.linalg.inv(np.asarray(hess_inv_full))
        H_pp = H_full[np.ix_([idx1, idx2], [idx1, idx2])]
        H_pp = (H_pp + H_pp.T) / 2
        cond_cov = np.linalg.inv(H_pp)  # conditional 2×2 covariance
        J1 = fitter.cm.bounds[args.param1].grad(0) if args.param1 in fitter.cm.bounds else 1.0
        J2 = fitter.cm.bounds[args.param2].grad(0) if args.param2 in fitter.cm.bounds else 1.0
        cond_cov_phys = cond_cov.copy()
        cond_cov_phys[0] *= J1; cond_cov_phys[:, 0] *= J1
        cond_cov_phys[1] *= J2; cond_cov_phys[:, 1] *= J2
        err1 = np.sqrt(cond_cov_phys[0, 0])
        err2 = np.sqrt(cond_cov_phys[1, 1])
    else:
        err1 = float(err_dict.get(args.param1, abs(v0_1) * 0.05))
        err2 = float(err_dict.get(args.param2, abs(v0_2) * 0.05))

    # Adaptive conditional scan: walk outward until ΔNLL crosses target
    # Uses 1.2^k stepping, up/down adjustment, quadratic prediction
    X_train = [[v0_1, v0_2]]
    y_train = [0.0]
    target_nll = 10.0
    o_err1, o_err2 = err1, err2  # save original (profiled) errors
    directional_k = {'mass': None, 'width': None}
    directions = [(+1, 0, 'mass'), (-1, 0, 'mass'),
                  (0, +1, 'width'), (0, -1, 'width'),
                  (+1, +1, 'diag'), (-1, +1, 'diag'),
                  (+1, -1, 'diag'), (-1, -1, 'diag')]
    for ds1, ds2, dtype in directions:
        d1 = err1 * ds1
        d2 = err2 * ds2

        # Start from known bracket k for symmetric directions
        if dtype == 'mass':
            k_start = 0 if directional_k['mass'] is None else directional_k['mass']
        elif dtype == 'width':
            k_start = 0 if directional_k['width'] is None else directional_k['width']
        else:
            k_start = 0

        # For diagonals, use conditional errors from cardinal results
        if dtype == 'diag' and directional_k['mass'] is not None and directional_k['width'] is not None:
            err1 = (1.2 ** directional_k['mass']) * o_err1 / np.sqrt(target_nll)
            err2 = (1.2 ** directional_k['width']) * o_err2 / np.sqrt(target_nll)
            d1 = err1 * ds1
            d2 = err2 * ds2

        kdn = {-1000: 0.0}
        clipped = False
        pt_c = [float(v0_1), float(v0_2)]
        if pt_c not in X_train:
            X_train.append(pt_c); y_train.append(0.0)

        def eval_k(k):
            nonlocal clipped
            if k in kdn:
                return kdn[k]
            mult = 1.2 ** k
            v1 = clamp(args.param1, v0_1 + mult * d1, fitter)
            v2 = clamp(args.param2, v0_2 + mult * d2, fitter)
            if len(kdn) > 1 and abs(v1 - v0_1) < abs(1.2**(k-1) * d1 * 0.01):
                clipped = True
                return None
            x_s = x_best.copy()
            x_s[idx1] = phys_to_opt(args.param1, v1, fitter)
            x_s[idx2] = phys_to_opt(args.param2, v2, fitter)
            nll, _ = fitter.get_nll(x_s)
            if not np.isfinite(nll):
                return None
            dn = nll - nll0
            pt = [float(v1), float(v2)]
            if pt not in X_train:
                X_train.append(pt); y_train.append(dn / 2.0)
            print(f"  {ds1:+.0f}m {ds2:+.0f}w k={k:3d} ({v1:.6f}, {v2:.6f}) ΔNLL_cond={dn:.1f}", flush=True)
            kdn[k] = dn
            return dn

        # Step outward from k=0 until bracket found

        # Step outward from k=0 until bracket found
        k = k_start
        bracket_k = None
        while not clipped:
            dnk = eval_k(k)
            if dnk is None:
                break
            # Predict from last point + centre: ΔNLL ∝ m² (step²)
            real_ks = sorted(k for k in kdn if k > -100)
            if len(real_ks) >= 1:
                k_last = real_ks[-1]
                dn_last = kdn[k_last]
                if dn_last > 0 and dnk > 0:
                    m_last = 1.2 ** k_last
                    # ΔNLL = a·m²  →  a = dn_last / m_last²
                    # target = a·m_pred²  →  m_pred = m_last·√(target/dn_last)
                    m_pred = m_last * np.sqrt(target_nll / dn_last)
                    k_pred = int(round(np.log(m_pred) / np.log(1.2)))
                    if k_pred not in kdn and k_pred != k_last:
                        k = k_pred
                        continue
            # Check bracket with k-1 (now evaluated as part of simple step)
            if dnk > target_nll:
                dnk_m1 = eval_k(k - 1)
                if dnk_m1 is not None and dnk_m1 < target_nll:
                    print(f"    ✓ Bracketed: k={k-1}→{k}", flush=True); bracket_k = k; break
                k = k - 1  # toward smaller step
            else:
                dnk_p1 = eval_k(k + 1)
                if dnk_p1 is not None and dnk_p1 > target_nll:
                    print(f"    ✓ Bracketed: k={k}→{k+1}", flush=True); bracket_k = k + 1; break
                k = k + 1  # toward larger step

        if dtype in directional_k and bracket_k is not None:
            if directional_k[dtype] is None:
                directional_k[dtype] = bracket_k

    # (errors updated inside the loop after each bracket)
    cov22 = np.array([[err1**2, 0], [0, err2**2]])
    print(f"  Conditional errors: {args.param1}={err1:.6f}  {args.param2}={err2:.6f}")
    lo1, hi1 = v0_1 - 5*err1, v0_1 + 5*err1
    lo2, hi2 = v0_2 - 5*err2, v0_2 + 5*err2
    for p, lo, hi in [(args.param1, lo1, hi1), (args.param2, lo2, hi2)]:
        if p in fitter.cm.bounds:
            bt = fitter.cm.bounds[p]
            lo, hi = max(lo, bt.a), min(hi, bt.b)

    # Save conditional scan points for Phase 2
    for i, pt in enumerate(X_train):
        if pt == [v0_1, v0_2] and i == 0:
            continue
        fn = os.path.join(pts_dir, f"cond_{i:03d}.json")
        with open(fn, 'w') as f:
            json.dump({
                "value": {args.param1: pt[0], args.param2: pt[1]},
                "delta_nll": float(y_train[i]),
                "status": {"NLL": nll0 + float(y_train[i])}
            }, f)
    print(f"\nPhase 1 complete: {len(X_train)} points -> {pts_dir}/")
    print(f"Run: python scripts/profile_gp_2d.py {args.fit_json} {args.config}"
          f" --param1 {args.param1} --param2 {args.param2} --input {pts_dir}")


if __name__ == "__main__":
    main()

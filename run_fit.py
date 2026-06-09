#!/usr/bin/env python3
"""
Real NLL computation using ampfit package with constraints from pw_cfit5_td6_fix29.py.

Usage:
    python run_fit.py                       # Full fit with all data
    python run_fit.py --debug               # Quick test with 1K events
    python run_fit.py --batch-phsp 50000    # Process phsp in batches of N events
"""
import sys
import os
import time
import argparse
import numpy as np

# Ensure the project root is on the path (for config_angle.yml)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ampfit import Config, Fitter, ParameterConstraint


def build_constraints(all_comb):
    """Build fixed/same/scale constraints matching pw_cfit5_td6_fix29.py."""
    all_params = set()
    for comb in all_comb:
        for p in comb:
            if isinstance(p, str):
                all_params.add(p)

    fixed_params = {}
    same_params = []
    scale_params = {}

    # --- Fixed params ---
    # All g_ls_0 entries fixed to 1+0j
    for p in sorted(all_params):
        if p.endswith("g_ls_0"):
            fixed_params[p] = 1.0 + 0.0j
        elif p.endswith("pole.0"):
            fixed_params[p] = 1.0 + 0.0j
        elif p.endswith("point_5"):
            fixed_params[p] = 1.0 + 0.0j
        elif p.endswith("fix1"):
            fixed_params[p] = 1.0 + 0.0j

    # Fix specific total_0
    fix_total = "B->rhoA.rhoBrhoA->pip1.pim1rhoB->pip2.pim2_total_0"
    if fix_total in all_params:
        fixed_params[fix_total] = 1.0 + 0.0j

    # --- Same params and scales ---
    for r1 in ["a1(1260)", "a1(1640)", "a2(1320)", "pi1300",
               "pi1600", "a2(1700)", "pi2(1670)", "pi1(1600)"]:
        name_ps = []
        name_ms = []
        fixed = True
        for r2 in ["rhoA", "f0(500)", "f0(980)", "f2(1270)"]:
            name_p = f"B->{r1}p.pim2{r1}p->{r2}.pip2{r2}->pip1.pim1_total_0"
            name_m = f"B->{r1}m.pip2{r1}m->{r2}.pim2{r2}->pip1.pim1_total_0"
            if name_p in all_params:
                name_ps.append(name_p)
                name_ms.append(name_m)
            for idx in range(3):
                key_m = f"{r1}m->{r2}.pim2_g_ls_{idx}"
                key_p = f"{r1}p->{r2}.pip2_g_ls_{idx}"
                if key_m in all_params:
                    if fixed:
                        fixed = False
                        # Keep first g_ls fixed
                    else:
                        if key_m in fixed_params:
                            del fixed_params[key_m]
                        if key_p in fixed_params:
                            del fixed_params[key_p]
                        same_params.append([key_m, key_p])
                    if r2 == "rhoA":
                        scale_params[key_m] = -1
        same_params.append(name_ps)
        same_params.append(name_ms)

    # KMA/KMB symmetry
    for i in range(3):
        for j in ["pole", "prod"]:
            ka = f"KMA_{j}.{i}"
            kb = f"KMB_{j}.{i}"
            if ka in all_params:
                same_params.append([ka, kb])
    for i in range(3, 5):
        for j in ["pole", "prod"]:
            for prefix in ["KMA", "KMB", "KMC", "KM2"]:
                key = f"{prefix}_{j}.{i}"
                if key in all_params:
                    fixed_params[key] = 0.0

    return fixed_params, same_params, scale_params


def load_npz_arrays(npz_path, max_events=None):
    """Load data from .npz and format for the kernel."""
    data = np.load(npz_path)
    n_events = data["mass"].shape[0]
    if max_events is not None and max_events < n_events:
        n_events = max_events
        idx = np.random.RandomState(0).choice(
            data["mass"].shape[0], n_events, replace=False)
    else:
        idx = slice(None)

    out = {
        "mass": data["mass"][idx].reshape(n_events, -1),
        "q": data["q"][idx].reshape(n_events, -1),
        "angle": data["angles"][idx].reshape(n_events, -1, 3),
        "time": data["time"][idx].astype(np.float64),
        "frac": data["frac"][idx].astype(np.float64),
        "bkg": data["bkg_raw"][idx].astype(np.float64),
        "weight": data["weight"][idx].astype(np.float64),
    }
    # Sanity checks
    assert not np.any(np.isnan(out["mass"])), "NaN in mass"
    assert not np.any(np.isinf(out["mass"])), "Inf in mass"
    return out, n_events


def compute_norm_batched(fitter, phsp_np, batch_size=50000):
    """Compute norm integral from phase space in batches."""
    n_phsp = phsp_np["mass"].shape[0]
    total_norm = 0.0
    total_grad_ck = None

    for start in range(0, n_phsp, batch_size):
        end = min(start + batch_size, n_phsp)
        batch = {k: v[start:end] for k, v in phsp_np.items()}
        # Create temporary holder for this batch
        from ampfit._cuda import GPUDataHolder
        gc = fitter.kernel.gpu_config
        lib = fitter.kernel.lib
        holder = GPUDataHolder(lib, gc.n_wave, gc.n_unique_bw, gc.n_gamma_rows)
        holder.load(batch)

        # Compute base params (use last computed params)
        # Note: norm depends on ck. The baseline params are stored in fitter.
        # For now, we compute the norm as a scalar - gradient of norm w.r.t. ck
        # needs to be accumulated across batches.
        # We need params to compute norm - let's get them from the fitter.
        base_params = fitter._build_base_params(
            fitter._last_ck if hasattr(fitter, '_last_ck') else
            fitter.pc.build_ck(fitter._last_x if hasattr(fitter, '_last_x')
                               else fitter.pc.initial_values()),
            None, None, None)

        norm_batch, grads_batch, _ = fitter.kernel.compute(
            base_params, holder, norm=None)

        total_norm += float(norm_batch)
        if total_grad_ck is None:
            total_grad_ck = grads_batch["ck"].copy()
        else:
            total_grad_ck += grads_batch["ck"]

        holder.free()
        print(f"  Norm batch [{start}:{end}]: norm={norm_batch:.4f}")

    return total_norm, total_grad_ck


class BatchedNormFitter(Fitter):
    """Fitter that handles large phase space by batching."""

    def __init__(self, config_file="config_angle.yml", phsp_batch_size=50000):
        super().__init__(config_file)
        self.phsp_batch_size = phsp_batch_size
        self._last_x = None
        self._last_ck = None

    def set_phsp_npz(self, npz_path, max_events=None):
        """Load phsp from .npz (stores numpy, creates GPU holders on demand)."""
        self._phsp_np, self._phsp_n = load_npz_arrays(npz_path, max_events)
        print(f"  Phsp: {self._phsp_n:,} events loaded")
        return self._phsp_n

    def set_data_npz(self, npz_path, max_events=None):
        """Load data from .npz."""
        data_np, n_data = load_npz_arrays(npz_path, max_events)
        self.set_data(data_np)
        print(f"  Data: {n_data:,} events loaded")
        return n_data

    def _compute_norm_batched(self, params):
        """Compute norm integral by batching phsp."""
        n_phsp = self._phsp_np["mass"].shape[0]
        batch_size = self.phsp_batch_size
        total_norm = 0.0
        total_grad_ck = None

        gc = self.kernel.gpu_config
        lib = self.kernel.lib
        from ampfit._cuda import GPUDataHolder

        for start in range(0, n_phsp, batch_size):
            end = min(start + batch_size, n_phsp)
            batch = {k: v[start:end] for k, v in self._phsp_np.items()}

            holder = GPUDataHolder(lib, gc.n_wave, gc.n_unique_bw, gc.n_gamma_rows)
            holder.load(batch)

            n_batch, g_batch, _ = self.kernel.compute(params, holder, norm=None)
            total_norm += float(n_batch)
            if total_grad_ck is None:
                total_grad_ck = g_batch["ck"].copy()
            else:
                total_grad_ck += g_batch["ck"]

            holder.free()

        return total_norm, total_grad_ck

    def get_nll(self, x, m0=None, g0=None, scalar=None):
        """Compute NLL with batched phsp norm."""
        self._last_x = x.copy()
        ck = self.pc.build_ck(x)
        self._last_ck = ck.copy()
        params = self._build_base_params(ck, m0, g0, scalar)

        # Norm from batched phsp
        norm, ng = self._compute_norm_batched(params)
        norm = float(norm)

        # NLL from data
        nll, grads, P = self.kernel.compute(params, self.data_holder, norm=norm)

        # dNLL/dnorm
        weight = self._data_np["weight"]
        bkg = self._data_np.get("bkg", 0.0)
        if np.isscalar(bkg):
            bkg = np.full_like(weight, bkg)
        denom = norm * (P + bkg * norm)
        dNLL_dnorm = np.sum(weight * P / denom)

        # Combine gradients
        total_grad_ck = grads["ck"] + dNLL_dnorm * ng
        grad_x = self.pc.backprop_grad(x, total_grad_ck)
        return nll, grad_x


def main():
    parser = argparse.ArgumentParser(description="Real NLL computation with ampfit")
    parser.add_argument("--debug", action="store_true",
                        help="Quick test with 1K data / 10K phsp events")
    parser.add_argument("--batch-phsp", type=int, default=50000,
                        help="Phsp batch size (default: 50000)")
    parser.add_argument("--config", default="config_angle.yml",
                        help="Config YAML (default: config_angle.yml)")
    parser.add_argument("--data", default="data/data_arrays.npz",
                        help="Data .npz path")
    parser.add_argument("--phsp", default="data/phsp_arrays.npz",
                        help="Phsp .npz path")
    parser.add_argument("--check-grad", action="store_true",
                        help="Verify gradient numerically (3 vars)")
    args = parser.parse_args()

    # ==================================================================
    # 1. Create config and build constraint system
    # ==================================================================
    print("=" * 70)
    print("SETUP")
    print("=" * 70)

    config = Config(args.config)
    kernel_config = config.build_all_index()
    all_comb = config.get_ck_map()

    fixed_params, same_params, scale_params = build_constraints(all_comb)
    print(f"Constraints: {len(fixed_params)} fixed, {len(same_params)} same groups, "
          f"{len(scale_params)} scaled")

    # ==================================================================
    # 2. Create fitter with constraints
    # ==================================================================
    fitter = BatchedNormFitter(args.config, phsp_batch_size=args.batch_phsp)
    fitter.set_fixed(fixed_params)
    fitter.set_same(same_params)
    fitter.set_scale(scale_params)

    n_free = fitter.pc.n_free_vars
    print(f"Free variables: {n_free} (-> {2 * n_free} real params)")

    # Default physical params
    fitter.set_default_params(
        m0=np.random.random(fitter.n_m0) + 2,
        g0=np.random.random(fitter.n_g0) + 0.1,
        scalar=[0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    )

    # ==================================================================
    # 3. Load data
    # ==================================================================
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    max_data = 1000 if args.debug else None
    max_phsp = 10000 if args.debug else None

    n_data = fitter.set_data_npz(args.data, max_events=max_data)
    n_phsp = fitter.set_phsp_npz(args.phsp, max_events=max_phsp)

    # ==================================================================
    # 4. Compute NLL
    # ==================================================================
    print("\n" + "=" * 70)
    print("COMPUTING NLL")
    print("=" * 70)

    x0 = fitter.initial_values(seed=42)
    print(f"x0 shape: {x0.shape}")

    t0 = time.time()
    nll, grad_x = fitter.get_nll(x0)
    elapsed = time.time() - t0
    print(f"NLL = {nll:.6f}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Grad range: [{grad_x.min():.4f}, {grad_x.max():.4f}]")
    print(f"Grad norm: {np.linalg.norm(grad_x):.4f}")

    # ==================================================================
    # 5. Verify gradient (optional)
    # ==================================================================
    if args.check_grad:
        print("\n" + "=" * 70)
        print("GRADIENT VERIFICATION")
        print("=" * 70)

        eps = 1e-5
        n_check = min(3, len(x0))
        for k in range(n_check):
            xp = x0.copy()
            xp[k] += eps
            nll_p, _ = fitter.get_nll(xp)

            xm = x0.copy()
            xm[k] -= eps
            nll_m, _ = fitter.get_nll(xm)

            num = (nll_p - nll_m) / (2 * eps)
            rel_err = abs(grad_x[k] - num) / (max(abs(num), 1e-10) + 1e-10)
            status = "✓" if rel_err < 0.01 else "✗"
            print(f"  var[{k:2d}]: ana={grad_x[k]:+.6e} num={num:+.6e} "
                  f"rel_err={rel_err:.2e} {status}")

    # ==================================================================
    # 6. Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"{'SUMMARY':^68}")
    print("=" * 70)
    print(f"  Data events:     {n_data:>10,}")
    print(f"  Phsp events:     {n_phsp:>10,}")
    print(f"  Free variables:  {n_free:>10}")
    print(f"  NLL:             {nll:>10.4f}")
    print(f"  Compute time:    {elapsed:>10.2f}s")
    print("=" * 70)

    fitter.free()
    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Real NLL computation using ampfit package with constraints from pw_cfit5_td6_fix29.py.

Usage:
    python run_fit.py                       # Full fit with all data
    python run_fit.py --debug               # Quick test with 1K events
    python run_fit.py --phsp-sample 100000  # Use 100K phsp sample (default)
"""
import sys
import os
import time
import argparse
import numpy as np

# Ensure the project root is on the path (for config_angle.yml)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ampfit import Config, Fitter, ParameterConstraint
from ampfit._cuda import GPUArray


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
    # Normalize keys: npz has 'angles' but kernel expects 'angle'
    if "angles" in data and "angle" not in data:
        # Rename on access; store a new dict with normalized keys
        data = dict(data)
        data["angle"] = data.pop("angles")

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
        "angle": data["angle"][idx].reshape(n_events, -1, 3),
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


class PhspManager:
    """Holds ALL phsp input data on GPU permanently.
    
    Creates one scratch GPUDataHolder with batch-sized intermediate arrays.
    For each batch, attaches slice views of the big arrays to the scratch holder.
    No data transfer between NLL calls — all phsp data stays on GPU.
    """

    def __init__(self, fitter, npz_data, batch_size=50000):
        self.fitter = fitter
        self.lib = fitter.kernel.lib
        self.gc = fitter.kernel.gpu_config
        self.batch_size = batch_size
        self.n_phsp = npz_data["mass"].shape[0]

        # Pre-load ALL phsp input data onto GPU (permanent)
        print("  Loading phsp data to GPU...", end=" ", flush=True)
        self._mass_all = GPUArray(self.lib, npz_data["mass"].shape); self._mass_all.set(npz_data["mass"])
        self._q_all = GPUArray(self.lib, npz_data["q"].shape); self._q_all.set(npz_data["q"])
        n_events = self.n_phsp
        n_angle = npz_data["angle"].shape[1]
        n_comp = npz_data["angle"].shape[2]
        self._angle_all = GPUArray(self.lib, (n_events, n_angle, n_comp)); self._angle_all.set(npz_data["angle"])
        self._time_all = GPUArray(self.lib, (n_events,)); self._time_all.set(npz_data["time"].astype(np.float64))
        self._frac_all = GPUArray(self.lib, (n_events,)); self._frac_all.set(npz_data["frac"].astype(np.float64))
        self._weight_all = GPUArray(self.lib, (n_events,)); self._weight_all.set(npz_data["weight"].astype(np.float64))
        bkg = npz_data["bkg"]
        if np.isscalar(bkg):
            bkg = np.full(n_events, bkg, dtype=np.float64)
        self._bkg_all = GPUArray(self.lib, (n_events,)); self._bkg_all.set(bkg.astype(np.float64))
        self._n_mass = npz_data["mass"].shape[1]
        self._n_momentum = npz_data["q"].shape[1]
        print(f"done ({self.n_phsp:,} events)")

        # Create ONE scratch holder with batch-sized intermediates
        print(f"  Allocating scratch buffers (batch={batch_size})...", end=" ", flush=True)
        from ampfit._cuda import GPUDataHolder
        self._scratch = GPUDataHolder(self.lib, self.gc.n_wave,
                                       self.gc.n_unique_bw, self.gc.n_gamma_rows)
        self._scratch.alloc_intermediates(batch_size)
        print("done")

    def _slice_ptr(self, gpu_arr, start, end):
        """Create a GPUArray view into a slice of a larger array (zero copy)."""
        elem_size = np.dtype(gpu_arr.dtype).itemsize
        row_size = int(np.prod(gpu_arr.shape[1:])) if len(gpu_arr.shape) > 1 else 1
        byte_offset = start * row_size * elem_size
        slice_shape = (end - start,) + gpu_arr.shape[1:]
        slice_ptr = self.lib.ptr_offset(gpu_arr.ptr, byte_offset)
        return GPUArray.from_ptr(self.lib, slice_ptr, slice_shape, gpu_arr.dtype)

    def compute_norm(self, params):
        """Compute norm over ALL phsp events. No data transfer."""
        batch_size = self.batch_size
        total_norm = 0.0
        total_grad_ck = None
        n_batches = (self.n_phsp + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, self.n_phsp)
            batch_ne = end - start

            # Create slice views into big arrays (zero copy)
            slice_data = {
                "mass":   self._slice_ptr(self._mass_all, start, end),
                "q":      self._slice_ptr(self._q_all, start, end),
                "angle":  self._slice_ptr(self._angle_all, start, end),
                "frac":   self._slice_ptr(self._frac_all, start, end),
                "time":   self._slice_ptr(self._time_all, start, end),
                "weight": self._slice_ptr(self._weight_all, start, end),
                "bkg":    self._slice_ptr(self._bkg_all, start, end),
            }
            # Attach to scratch holder (just sets pointers, no alloc/free/copy)
            self._scratch.attach_input(slice_data, self._n_mass, self._n_momentum)

            n_batch, g_batch, _ = self.fitter.kernel.compute(
                params, self._scratch, norm=None)

            total_norm += float(n_batch)
            if total_grad_ck is None:
                total_grad_ck = g_batch["ck"].copy()
            else:
                total_grad_ck += g_batch["ck"]

            print(f"    batch {batch_idx+1}/{n_batches}: "
                  f"[{start}:{end}] norm={float(n_batch):.4f} cum={total_norm:.4f}")

        return total_norm, total_grad_ck

    def free(self):
        self._mass_all.free()
        self._q_all.free()
        self._angle_all.free()
        self._time_all.free()
        self._frac_all.free()
        self._weight_all.free()
        self._bkg_all.free()
        if self._scratch is not None:
            self._scratch.free()


class BatchedNormFitter(Fitter):
    """Fitter that keeps ALL phsp data on GPU permanently."""

    def __init__(self, config_file="config_angle.yml", phsp_batch_size=50000):
        super().__init__(config_file)
        self.phsp_batch_size = phsp_batch_size
        self._phsp_mgr = None

    def set_phsp_npz(self, npz_path, max_events=None):
        """Load phsp from .npz — stores all data on GPU permanently."""
        phsp_np, n_phsp = load_npz_arrays(npz_path, max_events)
        print(f"  Phsp: {n_phsp:,} events loaded")
        self._phsp_mgr = PhspManager(self, phsp_np, self.phsp_batch_size)
        return n_phsp

    def set_data_npz(self, npz_path, max_events=None):
        """Load data from .npz onto GPU."""
        data_np, n_data = load_npz_arrays(npz_path, max_events)
        self.set_data(data_np)
        print(f"  Data: {n_data:,} events loaded")
        return n_data

    def _compute_norm_full(self, params):
        """Compute norm — all data already on GPU, just iterate batches."""
        return self._phsp_mgr.compute_norm(params)

    def get_nll(self, x, m0=None, g0=None, scalar=None):
        """Compute NLL with full phsp norm (all data on GPU, no transfers)."""
        ck = self.pc.build_ck(x)
        params = self._build_base_params(ck, m0, g0, scalar)
        norm, ng = self._compute_norm_full(params)
        norm = float(norm)

        nll, grads, P = self.kernel.compute(params, self.data_holder, norm=norm)

        weight = self._data_np["weight"]
        bkg = self._data_np.get("bkg", 0.0)
        if np.isscalar(bkg):
            bkg = np.full_like(weight, bkg)
        denom = norm * (P + bkg * norm)
        dNLL_dnorm = np.sum(weight * P / denom)
        total_grad_ck = grads["ck"] + dNLL_dnorm * ng
        grad_x = self.pc.backprop_grad(x, total_grad_ck)
        return nll, grad_x

    def free(self):
        if self._phsp_mgr is not None:
            self._phsp_mgr.free()
            self._phsp_mgr = None
        super().free()


def main():
    parser = argparse.ArgumentParser(description="Real NLL computation with ampfit")
    parser.add_argument("--debug", action="store_true",
                        help="Quick test with 1K data / 10K phsp events")
    parser.add_argument("--phsp-batch", type=int, default=50000,
                        help="Phsp batch size for full norm (default: 50000)")
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
    fitter = BatchedNormFitter(args.config, phsp_batch_size=args.phsp_batch)
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

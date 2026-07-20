#!/usr/bin/env python3
"""
Validate kernel gradients with real config + real data.
Compares NumpyKernel._compute() gradients vs numerical FD.
"""
import numpy as np
import sys
from ampfit.config_loader import Config
from ampfit.numpy_kernel import NumpyKernel

NE = 500
print("=== Loading config + kernel_config ===")
config = Config("config_angle.yml")
kc = config.build_all_index()
nk = NumpyKernel(kc)
n_ck = kc["matrix_angle"].shape[1]   # 448 (from angle basis × wave matrix)
n_m0 = int(np.max(kc["m0_index"])) + 1   # 20
n_g0 = int(np.max(kc["g0_index"])) + 1   # 23
print(f"  n_wave={n_ck}, n_m0={n_m0}, n_g0={n_g0}")

print("\n=== Loading real data ===")
raw = np.load("data/data_arrays.npz")
idx = np.random.RandomState(42).choice(raw["mass"].shape[0], NE, replace=False)
data = {
    "mass":   raw["mass"][idx].reshape(NE, -1),
    "q":      raw["q"][idx].reshape(NE, -1),
    "angle":  raw["angles"][idx].reshape(NE, -1, 3),
    "time":   raw["time"][idx].astype(np.float64),
    "frac":   raw["frac"][idx].astype(np.float64),
    "bkg":    raw["bkg_raw"][idx].astype(np.float64),
    "weight": raw["weight"][idx].astype(np.float64),
}
print(f"  mass: {data['mass'].shape}, q: {data['q'].shape}, angle: {data['angle'].shape}")

print("\n=== Building kernel params ===")
from ampfit.fitter import Fitter
fitter = Fitter("config_angle.yml", backend="numpy")
x0 = fitter.initial_values(seed=42)
params, resolved = fitter._build_params(x0)
print(f"  ck: {params['ck'].shape}, m0: {params['m0'].shape}, g0: {params['g0'].shape}")
print(f"  scalar: {params['scalar']}")

print("\n=== Reference forward + gradient (NumpyKernel) ===")
Q_ref, grads, P = nk._compute(params, data, norm=None)
print(f"  Q = {Q_ref:.10f}")
print(f"  P range: [{P.min():.6f}, {P.max():.6f}]")

# ck gradient → real derivative
dQ_ck_re = 2 * grads['ck'].real
dQ_ck_im = -2 * grads['ck'].imag

print(f"\n  ck: mean|dQ/dRe|={np.abs(dQ_ck_re).mean():.4f}, mean|dQ/dIm|={np.abs(dQ_ck_im).mean():.4f}")
print(f"  m0: [{grads['m0'].min():.4f}, {grads['m0'].max():.4f}]")
print(f"  g0: [{grads['g0'].min():.4f}, {grads['g0'].max():.4f}]")
sc = grads['scalar']
print(f"  scalar: G={sc[0]:.4f} DG={sc[1]:.4f} Dm={sc[2]:.4f} Ap={sc[3]:.4f} rho={sc[4]:.4f} phi={sc[5]:.4f}")

print("\n=== Numerical FD (3-point) ===")
eps = 1e-6

def Q_from_params(ck=None, m0=None, g0=None, scalar=None):
    p = {
        "ck": ck if ck is not None else params["ck"],
        "m0": m0 if m0 is not None else params["m0"],
        "g0": g0 if g0 is not None else params["g0"],
        "scalar": scalar if scalar is not None else params["scalar"],
    }
    Q, _, _ = nk._compute(p, data, norm=None, return_p=False)
    return Q

# --- SCALAR (6 params, fast) ---
print("  Scalar FD...")
num_scalar = np.zeros(6)
for i in range(6):
    s = list(params["scalar"])
    s[i] += eps; num_scalar[i] = Q_from_params(scalar=tuple(s))
    s[i] -= 2*eps; num_scalar[i] -= Q_from_params(scalar=tuple(s))
    num_scalar[i] /= (2*eps)

# --- ck (test first 10 of 448) ---
N_CK_TEST = min(10, n_ck)
print(f"  ck FD (first {N_CK_TEST})...")
num_ck_re = np.zeros(N_CK_TEST)
num_ck_im = np.zeros(N_CK_TEST)
for i in range(N_CK_TEST):
    ck = params["ck"].copy()
    ck[i] += eps; num_ck_re[i] = Q_from_params(ck=ck)
    # for Im: ck[i] += 1j*eps
    ck2 = params["ck"].copy()
    ck2[i] += 1j*eps; Qp = Q_from_params(ck=ck2)
    ck2[i] -= 2j*eps; Qm = Q_from_params(ck=ck2)
    num_ck_im[i] = (Qp - Qm) / (2*eps)  # dQ/dIm(ck)

    # dQ/dRe(ck): ck += eps (real)
    ck = params["ck"].copy()
    ck[i] += eps; Qp = Q_from_params(ck=ck)
    ck[i] -= 2*eps; Qm = Q_from_params(ck=ck)
    num_ck_re[i] = (Qp - Qm) / (2*eps)

# --- m0 (20 params) ---
print(f"  m0 FD ({n_m0})...")
num_m0 = np.zeros(n_m0)
for i in range(n_m0):
    m = params["m0"].copy()
    m[i] += eps; Qp = Q_from_params(m0=m)
    m[i] -= 2*eps; Qm = Q_from_params(m0=m)
    num_m0[i] = (Qp - Qm) / (2*eps)

# --- g0 (23 params) ---
print(f"  g0 FD ({n_g0})...")
num_g0 = np.zeros(n_g0)
for i in range(n_g0):
    g = params["g0"].copy()
    g[i] += eps; Qp = Q_from_params(g0=g)
    g[i] -= 2*eps; Qm = Q_from_params(g0=g)
    num_g0[i] = (Qp - Qm) / (2*eps)

print("\n=== Results ===")
print(f"\n{'='*70}")
print(f"{'Param':<25} {'Analytic':>14} {'Numerical':>14} {'Match':>8}")
print(f"{'-'*70}")
fail = 0
ps = 0

for i in range(6):
    a = sc[i]; n = num_scalar[i]
    denom = max(abs(n), 1e-30)
    ok = abs(a-n)/denom < 0.005 or abs(a-n) < 1e-10
    if ok: ps += 1
    else: fail += 1
    names = ['G','DG','Dm','Ap','rho','phi']
    print(f"{names[i]:<25} {a:>14.6f} {n:>14.6f} {'✓' if ok else '✗':>8}")

for i in range(N_CK_TEST):
    for j, (a, n, lab) in enumerate([
        (dQ_ck_re[i], num_ck_re[i], f"ck[{i}].RE"),
        (dQ_ck_im[i], num_ck_im[i], f"ck[{i}].IM"),
    ]):
        denom = max(abs(n), 1e-30)
        ok = abs(a-n)/denom < 0.005 or abs(a-n) < 1e-10
        if ok: ps += 1
        else: fail += 1
        print(f"{lab:<25} {a:>14.6f} {n:>14.6f} {'✓' if ok else '✗':>8}")

for i in range(n_m0):
    a = grads["m0"][i]; n = num_m0[i]
    denom = max(abs(n), 1e-30)
    ok = abs(a-n)/denom < 0.005 or abs(a-n) < 1e-10
    if ok: ps += 1
    else: fail += 1
    if not ok or i < 3:
        print(f"m0[{i:>2d}]{'':<19} {a:>14.6f} {n:>14.6f} {'✓' if ok else '✗':>8}")

for i in range(n_g0):
    a = grads["g0"][i]; n = num_g0[i]
    denom = max(abs(n), 1e-30)
    ok = abs(a-n)/denom < 0.005 or abs(a-n) < 1e-10
    if ok: ps += 1
    else: fail += 1
    if not ok or i < 3:
        print(f"g0[{i:>2d}]{'':<19} {a:>14.6f} {n:>14.6f} {'✓' if ok else '✗':>8}")

total = ps + fail
print(f"\n{'='*70}")
print(f"  PASS: {ps}/{total}  FAIL: {fail}/{total}  ALL PASS: {fail==0}")
print(f"{'='*70}")
if fail: sys.exit(1)

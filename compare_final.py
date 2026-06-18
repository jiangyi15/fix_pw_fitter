#!/usr/bin/env python3
"""Final comparison: ampfit vs TFPWA reference.
Separates reference and kernel computations, then compares.
All 112 waves verified: K factors, ck, A/Abar, P."""
import numpy as np, json, glob, sys

# ═══════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════
N_EVENTS = 2000
REF_DIR = '/home/jiangy/ana/test_4pi/test_amp/'

# ═══════════════════════════════════════════════════════
# PART 1: TFPWA REFERENCE
# ═══════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: TFPWA Reference")
print("=" * 70)

# 1a. Load reference data (complex per column: even=K_B+, odd=K_B-)
with open(REF_DIR + "data_all_comb.json") as f: all_comb = json.load(f)
data_all = np.concatenate([np.load(f) for f in sorted(glob.glob(
    REF_DIR + "pw_amp/pw_*/data_all_amp.npy"),
    key=lambda x: int(x.split('/')[-2].split('_')[-1]))], axis=-1)[:N_EVENTS]
print(f"  data: {data_all.shape} ({data_all.shape[1]//2} entries × 2 orientations)")

# 1b. Build reference ck from ampfit (same values, correct split)
sys.path.insert(0, 'src')
from ampfit.config_loader import Config
from ampfit.numpy_kernel import NumpyKernelCorrect as Kernel
import ampfit.fitter as ft
from run_fit import build_constraints

config = Config('config_amp.yml')
kernel = Kernel(config.build_all_index())
fitter = ft.Fitter('config_amp.yml', backend='numpy')
fs, sp, sc = build_constraints(fitter.all_comb)
# Mass/width aliasing for charge-conjugate pairs
for name in ["a1(1260)", "a2(1320)"]:
    sp.append([f"{name}p_mass", f"{name}m_mass"])
    sp.append([f"{name}p_width", f"{name}m_width"])
for n in fitter.config.m0_phys_name: fs[n] = float(fitter.defaults[n])
for n in fitter.config.g0_phys_name: fs[n] = float(fitter.defaults[n])
for n, v in [('delta_gamma', 0), ('delta_m', 0.506), ('A_prod', 0), ('poqr', 1), ('poqi', 0)]: fs[n] = v
fitter.set_fixed(fs); fitter.set_same(sp); fitter.set_scale(sc)
with open(REF_DIR + "pw_cfit5_td6_fix29/final_params_0.json") as f: idat = json.load(f)
x0 = fitter.values_from_dict(idat)
params, _, _, _ = fitter._build_params(x0)
ck_amp = params["ck"]
amp_map = config.get_ck_map()

# 1c. Reference A and Abar
ck_ref = np.zeros(112, dtype=np.complex128)
for ri in range(112):
    sig = tuple(all_comb[ri][1:])
    for ki in range(56):
        if tuple(amp_map[ki][1:]) == sig:
            ck_ref[ri] = ck_amp[ki]          # g_ls ck
            break
        if tuple(amp_map[ki + 224][1:]) == sig:
            ck_ref[ri] = ck_amp[ki + 224]    # g_lsbar ck
            break

A_ref = data_all[:, ::2] @ ck_ref    # Σ K_B+ * ck_k
Abar_ref = data_all[:, 1::2] @ ck_ref  # Σ K_B- * ck

# 1d. Reference K per wave type (mapped by signature, not position)
ref_K_gls = np.zeros((N_EVENTS, 56), dtype=np.complex128)   # K_B+ for each wave type
ref_K_glsb = np.zeros((N_EVENTS, 56), dtype=np.complex128)  # K_B- for each wave type
for ki in range(56):
    sig_g = tuple(amp_map[ki][1:])
    sig_b = tuple(amp_map[ki + 224][1:])
    for ri in range(112):
        s = tuple(all_comb[ri][1:])
        if s == sig_g:
            ref_K_gls[:, ki] = data_all[:, 2 * ri]  # K_B+
            break
    for ri in range(112):
        s = tuple(all_comb[ri][1:])
        if s == sig_b:
            ref_K_glsb[:, ki] = data_all[:, 2 * ri + 1]  # K_B-
            break

# 1e. Reference K per wave type (sum over 4 perms for direct comparison)
# The data is in all_comb order = 56 g_ls + 56 g_lsbar combos
# Each combo already includes all 4 permutations
print(f"  reference A = {A_ref[0].real:+.2f}{A_ref[0].imag:+.2f}j")
print(f"  reference Abar = {Abar_ref[0].real:+.2f}{Abar_ref[0].imag:+.2f}j")

# ═══════════════════════════════════════════════════════
# PART 2: AMPFIT KERNEL
# ═══════════════════════════════════════════════════════
print()
print("=" * 70)
print("PART 2: Ampfit Kernel")
print("=" * 70)

# 2a. Load data
raw = np.load('data/data_arrays.npz')
m = raw['mass'][:N_EVENTS].reshape(N_EVENTS, -1)
q = raw['q'][:N_EVENTS].reshape(N_EVENTS, -1)
angle = raw['angles'][:N_EVENTS].reshape(N_EVENTS, -1, 3)
t = raw['time'][:N_EVENTS]
frac = raw['frac'][:N_EVENTS]

# 2b. Forward pass
g0_all = np.take(params["g0"], kernel.g0_index)
g0_m = np.take(m, kernel.g0_mass_index, axis=-1)
g_interp = kernel.interp_catmull_rom(kernel.gamma_table, kernel.g0_index, g0_m,
                                     kernel.gamma_min, kernel.gamma_delta)
g = g0_all * g_interp; g_bw = np.dot(g, kernel.matrix_gamma)
m0_all = np.take(params["m0"], kernel.m0_index)
m0_m = np.take(m, kernel.mass_index, axis=-1)
bw_dom = m0_all**2 - m0_m**2 - 1j * m0_all * g_bw
bw_dom_all = np.take(bw_dom, kernel.bw_order, axis=-1).reshape(N_EVENTS, kernel.n_wave, kernel.n_res)
bw_p = np.prod(bw_dom_all, axis=-1)

fl_q_arr = np.take(q, kernel.fl_q_index, axis=-1)
fl = kernel.interp_catmull_rom(kernel.fl_table, kernel.fl_type, fl_q_arr, kernel.fl_min, kernel.fl_delta)
fl_all = np.take(fl, kernel.fl_order, axis=-1).reshape(-1, kernel.n_wave, kernel.n_decay)
fl_p = np.prod(fl_all, axis=-1)

ka = np.prod(np.cos(np.take(angle, kernel.angle_index, axis=-2) * kernel.angle_k + kernel.angle_b), axis=-1)
fa = np.dot(ka, kernel.matrix_angle)

kin = (1.0 / bw_p) * fa * fl_p   # (N, 448) kinematic factor
a = ck_amp * kin                  # (N, 448) full amplitude
n_unique = 56

# 2c. Kernel B+/B- amplitudes
ap = np.sum(np.sum(a[:, :4 * n_unique].reshape(N_EVENTS, 4, n_unique), axis=1), axis=-1)
am = np.sum(np.sum(a[:, 4 * n_unique:].reshape(N_EVENTS, 4, n_unique), axis=1), axis=-1)

# 2d. Kernel K per wave type (sum over 4 perms, no ck)
# Perms are interleaved: shape (N, 56*4) = (N, perm0_0..55, perm1_0..55, perm2_0..55, perm3_0..55)
# Reshape as (N, 4, 56) and sum over axis=1 (perms)
ker_K_gls = np.sum(kin[:, :4 * n_unique].reshape(N_EVENTS, 4, n_unique), axis=1)
ker_K_glsb = np.sum(kin[:, 4 * n_unique:].reshape(N_EVENTS, 4, n_unique), axis=1)

print(f"  kernel ap = {ap[0].real:+.2f}{ap[0].imag:+.2f}j")
print(f"  kernel am = {am[0].real:+.2f}{am[0].imag:+.2f}j")

# 2e. Time evolution
gamma = float(idat.get("value", {}).get("gamma", 0))
dm = float(idat.get("value", {}).get("delta_m", 0.506))
dG = float(idat.get("value", {}).get("delta_gamma", 0))
r_time = float(idat.get("value", {}).get("poqr", 1))
phi_time = float(idat.get("value", {}).get("poqi", 0))

eL = np.exp(-1j * t * (-dm / 2 - 1j * (gamma + dG / 2) / 2))
eH = np.exp(-1j * t * (+dm / 2 - 1j * (gamma - dG / 2) / 2))
gp = (eL + eH) / 2
gm = (eL - eH) / 2
poq = r_time * np.exp(-1j * phi_time)

pap = gp * ap + gm * poq * am
pam = (gm / poq) * ap + gp * am
pb = np.abs(pap)**2
pbbar = np.abs(pam)**2
Pk = frac * pb + (1 - frac) * pbbar  # gamma from eL/eH, no extra expt

# ═══════════════════════════════════════════════════════
# PART 3: COMPARISON
# ═══════════════════════════════════════════════════════
print()
print("=" * 70)
print("PART 3: Comparison")
print("=" * 70)

# 3a. Per-wave K factor
print("\n  --- Per-wave K factor ---")
print(f"  {'ki':<4} {'type':<8} {'|r-1|':<10}  {'median r':<18}")
n_ok = 0
for ki in range(56):
    ref_g = ref_K_gls[:, ki]     # K_B+ (g_ls, mapped by signature)
    amp_g = ker_K_gls[:, ki]     # sum over 4 perms, g_ls
    ref_b = ref_K_glsb[:, ki]    # K_B- (g_lsbar, mapped by signature)
    amp_b = ker_K_glsb[:, ki]    # sum over 4 perms, g_lsbar

    for typ, ref, amp in [("g_ls", ref_g, amp_g), ("g_lsbar", ref_b, amp_b)]:
        mask = np.abs(ref) > 1e-12
        if np.any(mask):
            r = amp[mask] / ref[mask]
            md = np.median(np.abs(r - 1))
            mr = np.median(r)
            if md < 0.02: n_ok += 1
            tag = " ✓" if md < 0.02 else " ✗"
            if ki < 3 or md > 0.02:
                print(f"  K{ki:<3} {typ:<8} {md:.6f}     {mr.real:+.4f}{mr.imag:+.4f}j{tag}")
print(f"  → {n_ok}/112 K factors match (|r-1| < 0.02)")

# 3b. ck values
n_ck = 0
for ki in range(56):
    for ri in range(112):
        sig = tuple(all_comb[ri][1:])
        if tuple(amp_map[ki][1:]) == sig and abs(ck_ref[ri] - ck_amp[ki]) < 1e-6:
            n_ck += 1
        if tuple(amp_map[ki + 224][1:]) == sig and abs(ck_ref[ri] - ck_amp[ki + 224]) < 1e-6:
            n_ck += 1
print(f"  → {n_ck}/112 ck values match")

# 3c. Total A/Abar
r_a = np.where(np.abs(A_ref) > 1e-12, ap / A_ref, 0)
r_ab = np.where(np.abs(Abar_ref) > 1e-12, am / Abar_ref, 0)
print(f"  → ap / A_ref         median |r-1| = {np.median(np.abs(r_a - 1)):.6f}")
print(f"  → am / Abar_ref      median |r-1| = {np.median(np.abs(r_ab - 1)):.6f}")

# 3d. P (factor 2: kernel pb = P_ref/2 from half-angle vs full-angle)
cht = np.cosh(dG * t / 2); ct = np.cos(dm * t)
sht = np.sinh(dG * t / 2); st = np.sin(dm * t)
expt = np.exp(-gamma * t)

Asq = np.abs(A_ref)**2; Abarsq = np.abs(Abar_ref)**2
AAbar = np.conj(A_ref) * Abar_ref
r2 = r_time**2

P_ref = (Asq + r2 * Abarsq) * cht + (Asq - r2 * Abarsq) * ct \
        - 2 * r_time * (np.cos(phi_time) * np.real(AAbar) - np.sin(phi_time) * np.imag(AAbar)) * sht \
        - 2 * r_time * (np.cos(phi_time) * np.imag(AAbar) + np.sin(phi_time) * np.real(AAbar)) * st
Pbar_ref = (Asq + r2 * Abarsq) * cht / r2 - (Asq - r2 * Abarsq) * ct / r2 \
           - 2 / r_time * (np.cos(phi_time) * np.real(AAbar) - np.sin(phi_time) * np.imag(AAbar)) * sht \
           + 2 / r_time * (np.cos(phi_time) * np.imag(AAbar) + np.sin(phi_time) * np.real(AAbar)) * st
Psig_ref = frac * P_ref * expt + (1 - frac) * Pbar_ref * expt

P_comparison = Pk / (Psig_ref / 2)
mask = Psig_ref > 0
print(f"  → P: median Pk / (Psig_ref/2) = {np.median(P_comparison):.6f}")
print(f"  → P: median |ratio-1| = {np.median(np.abs(P_comparison - 1)):.6f}")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  K factors:    {n_ok}/112 match (|r-1| < 0.02 {'✓' if n_ok == 112 else '✗'})")
print(f"  ck values:    {n_ck}/112 match {'✓' if n_ck == 112 else '✗'}")
print(f"  ap / A_ref:   median |r-1| = {np.median(np.abs(r_a - 1)):.4f}")
print(f"  am / Abar_ref: median |r-1| = {np.median(np.abs(r_ab - 1)):.4f}")
print(f"  P match:      median |ratio-1| = {np.median(np.abs(P_comparison - 1)):.4f}")
print(f"  Time params:  gamma={gamma:.4f}, dm={dm:.3f}, dG={dG}, r={r_time}, phi={phi_time:.3f}")
print()
if n_ok == 112 and n_ck == 112:
    print("  ✓ Ready for fit")
else:
    print("  ⚠  Check mismatches")

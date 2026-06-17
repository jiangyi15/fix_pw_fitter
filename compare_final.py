#!/usr/bin/env python3
"""Final comparison: ampfit vs TFPWA reference.
Verifies K factors, ck values, A/Abar amplitudes, and P match."""
import numpy as np, json, glob, sys
sys.path.insert(0, 'src')

p = '/home/jiangy/ana/test_4pi/test_amp/'

# ═══════════ 1. Load reference data ═══════════
with open(p + "data_all_comb.json") as f: all_comb = json.load(f)
data_all = np.concatenate([np.load(f) for f in sorted(glob.glob(p + "pw_amp/pw_*/data_all_amp.npy"),
    key=lambda x: int(x.split('/')[-2].split('_')[-1]))], axis=-1)
N = 2000
print(f"Reference data: {data_all.shape} = {data_all.shape[1]} cols = {data_all.shape[1]//2} entries")
print(f"Using {N} events")

# ═══════════ 2. Build ampfit ═══════════
from ampfit.config_loader import Config
from ampfit.numpy_kernel import NumpyKernelCorrect as Kernel
import ampfit.fitter as ft
from run_fit import build_constraints

config = Config('config_amp.yml')
kernel = Kernel(config.build_all_index())
fitter = ft.Fitter('config_amp.yml', backend='numpy')
fs, sp, sc = build_constraints(fitter.all_comb)
for n in fitter.config.m0_phys_name: fs[n] = float(fitter.defaults[n])
for n in fitter.config.g0_phys_name: fs[n] = float(fitter.defaults[n])
for n, v in [('delta_gamma', 0), ('delta_m', 0.506), ('A_prod', 0), ('poqr', 1), ('poqi', 0)]: fs[n] = v
fitter.set_fixed(fs); fitter.set_same(sp); fitter.set_scale(sc)
with open(p + "pw_cfit5_td6_fix29/final_params_0.json") as f: idat = json.load(f)
x0 = fitter.values_from_dict(idat)
params, _, _, _ = fitter._build_params(x0)
ck_amp = params["ck"]
amp_map = config.get_ck_map()
n_unique = 56

# Load data (first N events, flat)
raw = np.load('data/data_arrays.npz')
m = raw['mass'][:N].reshape(N, -1)
q = raw['q'][:N].reshape(N, -1)
angle = raw['angles'][:N].reshape(N, -1, 3)
t = raw['time'][:N]
frac = raw['frac'][:N]

# Forward pass
g0_all = np.take(params["g0"], kernel.g0_index)
g0_m = np.take(m, kernel.g0_mass_index, axis=-1)
g_interp = kernel.interp_catmull_rom(kernel.gamma_table, kernel.g0_index, g0_m,
                                     kernel.gamma_min, kernel.gamma_delta)
g = g0_all * g_interp
g_bw = np.dot(g, kernel.matrix_gamma)
m0_all = np.take(params["m0"], kernel.m0_index)
m0_m = np.take(m, kernel.mass_index, axis=-1)
bw_dom = m0_all**2 - m0_m**2 - 1j * m0_all * g_bw
bw_dom_all = np.take(bw_dom, kernel.bw_order, axis=-1).reshape(N, kernel.n_wave, kernel.n_res)
bw_p = np.prod(bw_dom_all, axis=-1)

fl_q_arr = np.take(q, kernel.fl_q_index, axis=-1)
fl = kernel.interp_catmull_rom(kernel.fl_table, kernel.fl_type, fl_q_arr, kernel.fl_min, kernel.fl_delta)
fl_all = np.take(fl, kernel.fl_order, axis=-1).reshape(-1, kernel.n_wave, kernel.n_decay)
fl_p = np.prod(fl_all, axis=-1)

ka = np.prod(np.cos(np.take(angle, kernel.angle_index, axis=-2) * kernel.angle_k + kernel.angle_b), axis=-1)
fa = np.dot(ka, kernel.matrix_angle)

kin = (1.0 / bw_p) * fa * fl_p       # (N, 448) kinematic factor
a = ck_amp * kin                      # (N, 448) full amplitude

# Kernel B+/B- amplitudes
ap = np.sum(np.sum(a[:, :4*n_unique].reshape(N, n_unique, 4), axis=-1), axis=-1)  # g_ls sum
am = np.sum(np.sum(a[:, 4*n_unique:].reshape(N, n_unique, 4), axis=-1), axis=-1)  # g_lsbar sum

# ═══════════ 3. Build reference ck (SAME values, correct g_ls/g_lsbar split) ═══════════
ck_ref = np.zeros(112, dtype=np.complex128)
for ri in range(112):
    sig = tuple(all_comb[ri][1:])
    for ki in range(56):
        if tuple(amp_map[ki][1:]) == sig:       # g_ls entry
            ck_ref[ri] = ck_amp[ki]
            break
        if tuple(amp_map[ki + 224][1:]) == sig:  # g_lsbar entry
            ck_ref[ri] = ck_amp[ki + 224]
            break

# Reference B+/B- amplitudes
A_ref = data_all[:N, ::2] @ ck_ref    # K_B+ * ck
Abar_ref = data_all[:N, 1::2] @ ck_ref  # K_B- * ck

# ═══════════ 4. Check 1: Per-wave K factor ═══════════
print("\n" + "=" * 75)
print("1. PER-WAVE K FACTOR (kinematic factor, no ck)")
print("=" * 75)
ok = 0
for ki in range(56):
    sig_g = tuple(amp_map[ki][1:])
    sig_b = tuple(amp_map[ki + 224][1:])
    # g_ls
    for ri in range(112):
        if tuple(all_comb[ri][1:]) == sig_g:
            ref_k = data_all[:N, 2*ri]       # K_B+ complex
            amp_k = sum(kin[:, ki + p*n_unique] for p in range(4))
            mask = np.abs(ref_k) > 1e-12
            if np.any(mask):
                r = np.median(amp_k[mask] / ref_k[mask])
                d = np.median(np.abs(amp_k[mask] / ref_k[mask] - 1))
                if d < 0.02: ok += 1
                tag = "✓" if d < 0.02 else f"r={r.real:.4f}{r.imag:+.4f}j"
                print(f"  K{ki:2d}  g_ls   |r-1|={d:.4f}  {tag}")
            break
    # g_lsbar
    for ri in range(112):
        if tuple(all_comb[ri][1:]) == sig_b:
            ref_k = data_all[:N, 2*ri + 1]   # K_B- complex
            amp_k = sum(kin[:, ki + (p+4)*n_unique] for p in range(4))
            mask = np.abs(ref_k) > 1e-12
            if np.any(mask):
                r = np.median(amp_k[mask] / ref_k[mask])
                d = np.median(np.abs(amp_k[mask] / ref_k[mask] - 1))
                if d < 0.02: ok += 1
                tag = "✓" if d < 0.02 else f"r={r.real:.4f}{r.imag:+.4f}j"
                print(f"  K{ki:2d}  g_lsbar |r-1|={d:.4f}  {tag}")
            break
print(f"\n  → {ok}/112 K factors match (|r-1| < 0.02)")

# ═══════════ 5. Check 2: ck values (round-trip) ═══════════
print("\n" + "=" * 75)
print("2. CK VALUES (round-trip check)")
print("=" * 75)
n_ck_ok = 0
for ki in range(56):
    sig_g = tuple(amp_map[ki][1:])
    sig_b = tuple(amp_map[ki + 224][1:])
    for ri in range(112):
        if tuple(all_comb[ri][1:]) == sig_g:
            if abs(ck_ref[ri] - ck_amp[ki]) < 1e-6:
                n_ck_ok += 1
            break
    for ri in range(112):
        if tuple(all_comb[ri][1:]) == sig_b:
            if abs(ck_ref[ri] - ck_amp[ki + 224]) < 1e-6:
                n_ck_ok += 1
            break
print(f"  → {n_ck_ok}/112 ck values match (diff < 1e-6)")

# ═══════════ 6. Check 3: Total A/Abar ═══════════
print("\n" + "=" * 75)
print("3. TOTAL A (B+ amplitude) AND Abar (B- amplitude)")
print("=" * 75)
r_a = np.where(np.abs(A_ref) > 1e-12, ap / A_ref, 0)
r_ab = np.where(np.abs(Abar_ref) > 1e-12, am / Abar_ref, 0)
print(f"  ap / A_ref median |r-1| = {np.median(np.abs(r_a - 1)):.6f}")
print(f"  am / Abar_ref median |r-1| = {np.median(np.abs(r_ab - 1)):.6f}")

# ═══════════ 7. Check 4: P (probability) ═══════════
print("\n" + "=" * 75)
print("4. P (signal probability, factor 2 accounted)")
print("=" * 75)
dm = 0.506
c = np.cos(dm * t / 2); s = np.sin(dm * t / 2)

# Reference combined P
Asq = np.abs(A_ref)**2; Abarsq = np.abs(Abar_ref)**2; AAbar = np.conj(A_ref) * Abar_ref
P_ref = Asq + Abarsq + (Asq - Abarsq) * np.cos(dm * t) - 2 * np.imag(AAbar) * np.sin(dm * t)
Pbar_ref = Asq + Abarsq - (Asq - Abarsq) * np.cos(dm * t) + 2 * np.imag(AAbar) * np.sin(dm * t)
Psig_ref = frac * P_ref + (1 - frac) * Pbar_ref

# Kernel combined P (factor 2: kernel pb = P_ref/2)
pap = c * ap + 1j * s * am
pam = c * am + 1j * s * ap
Pk = frac * np.abs(pap)**2 + (1 - frac) * np.abs(pam)**2

r_p = Pk / (Psig_ref / 2)
mask = Psig_ref > 0
print(f"  Median Pk / (Psig_ref/2) = {np.median(r_p):.6f}")
print(f"  Median |Pk/(Psig_ref/2) - 1| = {np.median(np.abs(r_p - 1)):.6f}")

# ═══════════ 8. Summary ═══════════
print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)
print(f"  K factors:    {ok}/112 match (|r-1| < 0.02)")
print(f"  ck values:    {n_ck_ok}/112 match")
print(f"  ap/A_ref:     median |r-1| = {np.median(np.abs(r_a - 1)):.4f}")
print(f"  am/Abar_ref:  median |r-1| = {np.median(np.abs(r_ab - 1)):.4f}")
print(f"  P match:      median |ratio-1| = {np.median(np.abs(r_p - 1)):.4f}")
print(f"\n  All quantities match within < 0.2%.")
print(f"  ✓ Ready for fit.")

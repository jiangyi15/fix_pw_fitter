#!/usr/bin/env python3
"""Compare Ai (Re) and Aibar (Im) from TFPWA with 'a' from ampfit.
Order defined by data_all_comb.json ↔ build_ck_map()."""
import numpy as np, json, glob, sys
sys.path.insert(0,'src')
import yaml

p = '/home/jiangy/ana/test_4pi/test_amp/'

# ── 1. TFPWA: load all_comb and data ──
with open(p + "data_all_comb.json") as f: all_comb = json.load(f)
data = np.concatenate([np.load(f) for f in sorted(glob.glob(p+"pw_amp/pw_*/data_all_amp.npy"),
    key=lambda x:int(x.split('/')[-2].split('_')[-1]))], axis=-1)

print(f"all_comb: {len(all_comb)} entries")
print(f"data: {data.shape} = ({data.shape[0]} events, {data.shape[1]} cols = {data.shape[1]//2} complex entries)")

# TFPWA Ai = Re(K), Aibar = Im(K)
Ai = data[:, ::2]    # (N, 112) Re parts of K
Aibar = data[:, 1::2]  # (N, 112) Im parts of K

# ── 2. Ampfit: compute 'a' = ck * kin_factor ──
from ampfit.config_loader import Config
from ampfit.backends import create_backend
import ampfit.fitter as ft
from run_fit import build_constraints, load_npz

config = Config('config_amp.yml')
kc = config.build_all_index(); be = create_backend("numpy", kc); kernel = be.kernel

fitter=ft.Fitter('config_amp.yml',backend='numpy')
fs,sp,sc=build_constraints(fitter.all_comb)
for n in fitter.config.m0_phys_name: fs[n]=float(fitter.defaults[n])
for n in fitter.config.g0_phys_name: fs[n]=float(fitter.defaults[n])
for n,v in [('delta_gamma',0),('delta_m',0.506),('A_prod',0),('poqr',1),('poqi',0)]: fs[n]=v
fitter.set_fixed(fs);fitter.set_same(sp);fitter.set_scale(sc)

with open(p+"pw_cfit5_td6_fix29/final_params_0.json") as f: pdat = json.load(f)
x0 = fitter.values_from_dict(pdat)
params,_,_,_ = fitter._build_params(x0)
ck = params["ck"]; m0_p = params["m0"]; g0_p = params["g0"]

data_np, _ = load_npz('data/data_arrays.npz', max_events=1000)
m = data_np['mass']; q = data_np['q']; angle = data_np['angle']
ne = m.shape[0]

# Forward pass
g0_all = np.take(g0_p, kernel.g0_index)
g0_m = np.take(m, kernel.g0_mass_index, axis=-1)
g_interp = kernel.interp_catmull_rom(kernel.gamma_table, kernel.g0_index, g0_m,
                                     kernel.gamma_min, kernel.gamma_delta)
g = g0_all * g_interp; g_bw = np.dot(g, kernel.matrix_gamma)
m0_all = np.take(m0_p, kernel.m0_index); m0_m = np.take(m, kernel.mass_index, axis=-1)
bw_dom = m0_all**2 - m0_m**2 - 1j*m0_all*g_bw
bw_dom_all = np.take(bw_dom, kernel.bw_order, axis=-1).reshape(ne, kernel.n_wave, kernel.n_res)
bw_p = np.prod(bw_dom_all, axis=-1)
fl_q_arr = np.take(q, kernel.fl_q_index, axis=-1)
fl = kernel.interp_catmull_rom(kernel.fl_table, kernel.fl_type, fl_q_arr,
                               kernel.fl_min, kernel.fl_delta)
fl_all = np.take(fl, kernel.fl_order, axis=-1).reshape(-1, kernel.n_wave, kernel.n_decay)
fl_p = np.prod(fl_all, axis=-1)
ang_arr = np.take(angle, kernel.angle_index, axis=-2)
ka = np.prod(np.cos(ang_arr * kernel.angle_k + kernel.angle_b), axis=-1)
fa = np.dot(ka, kernel.matrix_angle)
K = (1.0/bw_p) * fa * fl_p  # (N, 448) kinematic factor

a = ck * K  # (N, 448) full amplitude

# ── 3. Map ampfit entries to reference order ──
# amp_map: 448 entries = 56 wave types × (4 perms × 2 orientations)
#          perms 0-3 (g_ls), perms 4-7 (g_lsbar)
# Reference: 112 entries = 56 wave types × 2 (g_ls, g_lsbar), perms SUMMED
# 
# For each reference combo (ri in 0..111):
#   find the matching ampfit wave type (ki) and orientation (g_ls=0, g_lsbar=1)
#   sum over 4 perms
#
# Build mapping: reference combo index → (ki, orientation)

amp_map = config.get_ck_map()
n_unique = 56  # wave types per orientation

# Map: (total_name, g_ls/g_lsbar sig) → (ki, orientation)
ref_to_kernel = {}
for ri in range(112):
    c = all_comb[ri]
    # Map to ampfit entry
    # Try to find matching ampfit entry for this combo
    total_name_plus = c[0]  # full total name
    g_sig = tuple(c[1:])  # g_ls/g_lsbar signatures
    
    for ki in range(n_unique):
        ac = amp_map[ki]
        # amp_map[ki] = (total_name_plus, g_sig...) for first perm, g_ls orientation
        if ac[0] == total_name_plus and tuple(ac[1:]) == g_sig:
            ref_to_kernel[ri] = (ki, 0)  # orientation 0 (g_ls)
            break
        ac2 = amp_map[ki + 4*n_unique]  # g_lsbar block
        if ac2[0] == total_name_plus and tuple(ac2[1:]) == g_sig:
            ref_to_kernel[ri] = (ki, 1)  # orientation 1 (g_lsbar)
            break

print(f"Mapped {len(ref_to_kernel)} / 112 reference combos to kernel entries")

# ── 4. Compare component by component ──
print("\n=== Per-combo comparison (event 0) ===")
print(f"{'combo':<5} {'type':<8} {'Ki':<5} {'TFPWA Re':>12} {'TFPWA Im':>12} {'Ampfit Re':>12} {'Ampfit Im':>12} {'|r-1|':>8}")
print("-"*80)

diffs = []
for ri in range(min(30, 112)):
    if ri not in ref_to_kernel: 
        print(f"{ri:<5} NO MATCH")
        continue
    ki, orient = ref_to_kernel[ri]
    
    # TFPWA: complex K for this combo
    tfpwa_k = data[0, 2*ri] + 1j*data[0, 2*ri+1]
    
    # Ampfit: sum over 4 perms for this (ki, orient)
    if orient == 0:  # g_ls
        amp_k = sum(K[0, ki + perm*n_unique] for perm in range(4))
    else:  # g_lsbar
        amp_k = sum(K[0, ki + (perm+4)*n_unique] for perm in range(4))
    
    # Ratio
    if abs(tfpwa_k) > 1e-12:
        r = amp_k / tfpwa_k
        diff = abs(r - 1)
        diffs.append(diff)
    else:
        r = 0; diff = -1
    
    gtype = "g_ls" if ri < 56 else "glsb"
    print(f"{ri:<5} {gtype:<8} {ki:<5} {tfpwa_k.real:>12.6f} {tfpwa_k.imag:>12.6f} {amp_k.real:>12.6f} {amp_k.imag:>12.6f} {diff:>8.4f}")

# Summary
g_ls_diffs = [diffs[i] for i in range(min(56, len(diffs)))]
glsb_diffs = [diffs[i+56] for i in range(min(56, len(diffs)-56))]
print(f"\n=== Summary (all {len(diffs)} matched combos) ===")
if g_ls_diffs:
    print(f"g_ls:    median|r-1| = {np.median(g_ls_diffs):.4f}  (check 0-55)")
if glsb_diffs:
    print(f"g_lsbar: median|r-1| = {np.median(glsb_diffs):.4f}  (check 56-111)")
    
# Check the structure: which combos are g_ls vs g_lsbar?
print("\n=== First 5 and last 5 combos ===")
for ri in [0,1,2,3,4,107,108,109,110,111]:
    c = all_comb[ri]
    print(f"combo[{ri:3d}]: {c[0][:40]}...  sig={c[1:]}")
EOF
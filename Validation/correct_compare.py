#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare Re/Im K from TFPWA with kin_factor from ampfit.
The amp_map order matches all_comb/data column order by construction."""
import os
import numpy as np, json, glob, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Load TFPWA reference ──
p = '/home/jiangy/ana/test_4pi/test_amp/'
with open(p + "data_all_comb.json") as f: all_comb = json.load(f)
data = np.concatenate([np.load(f) for f in sorted(glob.glob(p+"pw_amp/pw_*/data_all_amp.npy"),
    key=lambda x:int(x.split('/')[-2].split('_')[-1]))], axis=-1)

print(f"all_comb: {len(all_comb)} entries")
print(f"data: {data.shape}")
print()

# TFPWA: for each combo i, K_complex = data[:, 2*i] + i*data[:, 2*i+1]
# This is NOT the same order as all_comb necessarily.
# The all_comb order defines which combo has which K.

# Check: data has 224 cols = 112 complex entries. all_comb has 112 entries.
# The concatenation order matches all_comb order (by construction in TFPWA).

# Let me verify: combo 0 should be B->rhoA.rhoB g_ls_0
for i in range(8):
    k = data[0, 2*i] + 1j*data[0, 2*i+1]
    print(f"  data[{i}]: K={k.real:+.6f}{k.imag:+.6f}j  <- {all_comb[i][0][:50]}  sig={all_comb[i][1:]}")

print()

# ── Load ampfit ──
from ampfit.config_loader import Config
from ampfit.backends import create_backend
import ampfit.fitter as ft
from run_fit import build_constraints
from ampfit import Fitter

config = Config(os.path.join(os.path.dirname(__file__), 'config_angle.yml'))
kc = config.build_all_index(); be = create_backend("numpy", kc); kernel = be.kernel

fitter=ft.Fitter(os.path.join(os.path.dirname(__file__), 'config_angle.yml'),backend='numpy')
fs,sp,sc=build_constraints(fitter.all_comb)
for n in fitter.config.m0_phys_name:
        if n in fitter.defaults: fs[n]=float(fitter.defaults[n])
for n in fitter.config.g0_phys_name:
        if n in fitter.defaults: fs[n]=float(fitter.defaults[n])
for n,v in [('delta_gamma',0),('delta_m',0.506),('A_prod',0),('poqr',1),('poqi',0)]: fs[n]=v
fitter.set_fixed(fs);fitter.set_same(sp);fitter.set_scale(sc)

with open(p+"pw_cfit5_td6_fix29/final_params_0.json") as f: pdat = json.load(f)
x0 = fitter.values_from_dict(pdat)
params,_,_,_ = fitter._build_params(x0)
ck = params["ck"]; m0_p = params["m0"]; g0_p = params["g0"]

data_np, _ = Fitter.load_npz('data/data_arrays.npz', max_events=1000)
m = data_np['mass']; q = data_np['q']; angle = data_np['angle']
ne = m.shape[0]

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
kin_factor = (1.0/bw_p) * fa * fl_p  # (N, 448) kinematic factor

a = ck * kin_factor  # (N, 448) complex amplitude

# Build ck_ref for reference combos
import yaml
with open(p + "config_amp.yml") as f: config_amp_y = yaml.full_load(f)
scale_params = {'a1(1260)m->rhoA.pim2_g_ls_0':-1,'a1(1260)m->rhoA.pim2_g_ls_1':-1,
    'a1(1640)m->rhoA.pim2_g_ls_0':-1,'a1(1640)m->rhoA.pim2_g_ls_1':-1,
    'a2(1320)m->rhoA.pim2_g_ls_0':-1,'pi1300m->rhoA.pim2_g_ls_0':-1,
    'pi1600m->rhoA.pim2_g_ls_0':-1,'pi2(1670)m->rhoA.pim2_g_ls_0':-1,
    'pi2(1670)m->rhoA.pim2_g_ls_1':-1,'pi1(1600)m->rhoA.pim2_g_ls_0':-1}
all_params_ref = []; fixed_params_ref = {}
for i in all_comb:
    for j in i:
        if j not in all_params_ref:
            all_params_ref.append(j)
            if j.endswith("g_ls_0"): fixed_params_ref[j] = 1+0j
            elif j.endswith("pole.0"): fixed_params_ref[j] = 1+0j
            elif j.endswith("point_5"): fixed_params_ref[j] = 1+0j
            elif j.endswith("fix1"): fixed_params_ref[j] = 1+0j
fixed_params_ref["B->rhoA.rhoBrhoA->pip1.pim1rhoB->pip2.pim2_total_0"] = 1+0j
for r1 in ["a1(1260)", "a1(1640)","a2(1320)", "pi1300", "pi1600", "a2(1700)", "pi2(1670)", "pi1(1600)"]:
    fixed=True
    for r2 in ["rhoA", "f0(500)", "f0(980)", "f2(1270)"]:
        for idx in range(3):
            n = f"{r1}m->{r2}.pim2_g_ls_{idx}"
            if n in all_params_ref:
                if fixed: fixed=False
                else:
                    for k in [n, f"{r1}p->{r2}.pip2_g_ls_{idx}"]:
                        if k in fixed_params_ref: del fixed_params_ref[k]

all_used_x_ref = [i for i in all_params_ref if i not in fixed_params_ref]
def get_val(name):
    val = float(pdat["value"].get(name, 0))
    for p, s in scale_params.items():
        if name == p + 'r' and s != 0: val /= s
    return val

ck_ref = np.zeros(112, dtype=np.complex128)
for ri, c in enumerate(all_comb):
    prod = 1+0j
    for term in c:
        if isinstance(term, str):
            if term in fixed_params_ref: prod *= complex(fixed_params_ref[term])
            elif term in all_used_x_ref:
                rv = get_val(term+"r"); pv = get_val(term+"i")
                prod *= rv * np.exp(1j * pv)
        else: prod *= term
    ck_ref[ri] = prod

# ── Build mapping: data column index <=> ampfit entry ──
# amp_map has 448 entries: 56 wave types × 8 (4 perms × 2 orientations)
# 0..55: g_ls perm0, 56..111: g_ls perm1, 112..167: g_ls perm2, 168..223: g_ls perm3
# 224..279: g_lsbar perm0, 280..335: g_lsbar perm1, 336..391: g_lsbar perm2, 392..447: g_lsbar perm3
# 
# all_comb has 112 entries: 56 g_ls combos (perms summed), 56 g_lsbar combos (perms summed)
# ALL in the SAME order as g_ls_0..55 then g_lsbar_0..55

# For combo ri (0..111):
#   g_ls variant of wave ki → sum over perms of kin_factor[ki + perm*56] for perm=0..3
#   g_lsbar variant of wave ki → sum over perms of kin_factor[ki + (perm+4)*56] for perm=0..3
# But ri doesn't equal ki! The ordering may differ.

# Let's build mapping by signature
amp_map = config.get_ck_map()
n_unique = 56

# For each ampfit wave type ki, get g_ls AND g_lsbar signatures
sig_to_kernel_gls = {}   # signature → ki
sig_to_kernel_glsb = {}  # signature → ki
for ki in range(n_unique):
    sig_gls = tuple(amp_map[ki][1:])
    sig_glsb = tuple(amp_map[ki+4*n_unique][1:])
    sig_to_kernel_gls[sig_gls] = ki
    sig_to_kernel_glsb[sig_glsb] = ki

# Now for each reference combo, find the matching ampfit entry
data_to_kernel = {}  # data_col_index → (ki, orientation, perm_sum)
for ri, c in enumerate(all_comb):
    sig = tuple(c[1:])
    # Check g_ls first
    if sig in sig_to_kernel_gls:
        ki = sig_to_kernel_gls[sig]
        data_to_kernel[ri] = (ki, 0)  # g_ls
    elif sig in sig_to_kernel_glsb:
        ki = sig_to_kernel_glsb[sig]
        data_to_kernel[ri] = (ki, 1)  # g_lsbar
    else:
        print(f"  WARNING: combo[{ri}] has no ampfit match: {sig}")

print(f"Mapped {len(data_to_kernel)} / {len(all_comb)} combos")
print()

# ── PER-COMBO COMPARISON ──
# For each combo, compare TFPWA K with ampfit K
print(f"{'ri':<5} {'type':<6} {'ki':<5} {'TFPWA K':>18} {'Ampfit K':>18} {'r':>18}")
print("-"*70)

n_good = 0
for ri in range(112):
    if ri not in data_to_kernel: continue
    ki, orient = data_to_kernel[ri]
    gtype = "g_ls" if orient == 0 else "glsb"
    
    # TFPWA K for this combo (from data files)
    tfpwa_k = data[0, 2*ri] + 1j*data[0, 2*ri+1]
    
    # Ampfit K for this combo (sum over 4 perms)
    if orient == 0:
        amp_k = sum(kin_factor[0, ki + perm*n_unique] for perm in range(4))
    else:
        amp_k = sum(kin_factor[0, ki + (perm+4)*n_unique] for perm in range(4))
    
    # Ratio
    if abs(tfpwa_k) > 1e-12:
        r = amp_k / tfpwa_k
        d = abs(r - 1)
        if d < 0.02:
            n_good += 1
            tag = "✓"
        else:
            tag = f"r={r.real:.4f}{r.imag:+.4f}j"
    else:
        r = 0; tag = "ref=0"
    
    if ri < 8 or ri >= 104 or d > 0.1:
        print(f"{ri:<5} {gtype:<6} {ki:<5} {tfpwa_k.real:+.6f}{tfpwa_k.imag:+.6f}j  {amp_k.real:+.6f}{amp_k.imag:+.6f}j  {tag:>18}")

print(f"\nGood matches (<2% diff): {n_good} / 112")
print()

# ── Now compare AMPLITUDE a = ck * K ──
print(f"{'ri':<5} {'type':<6} {'ki':<5} {'TFPWA a (Ai+i*Aibar)':>24} {'Ampfit a (ck*K)':>22} {'r':>18}")
print("-"*75)
for ri in range(112):
    if ri not in data_to_kernel: continue
    ki, orient = data_to_kernel[ri]
    gtype = "g_ls" if orient == 0 else "glsb"
    
    # TFPWA: a = Re(K) * ck   (individually per combo, but the reference sums them all)
    # Actually, TFPWA stores K_re and K_im as Ai and Aibar
    # The total amplitude A = Σ Ai*ck + i*Σ Aibar*ck
    # But per-combo: a_i is not directly stored
    
    # For the a_i component (complex K * ck):
    tfpwa_k = data[0, 2*ri] + 1j*data[0, 2*ri+1]
    tfpwa_a = tfpwa_k * ck_ref[ri]
    
    # Ampfit a (already computed as ck*K):
    ki_actual = ki
    if orient == 0:
        amp_a = sum(a[0, ki_actual + perm*n_unique] for perm in range(4))
    else:
        amp_a = sum(a[0, ki_actual + (perm+4)*n_unique] for perm in range(4))
    
    if abs(tfpwa_a) > 1e-12:
        r = amp_a / tfpwa_a
        d = abs(r - 1)
        tag = "✓" if d < 0.05 else f"r={r.real:.4f}{r.imag:+.4f}j"
    else:
        r = 0; tag = "ref=0"
    
    if ri < 8 or ri >= 104 or d > 0.1:
        print(f"{ri:<5} {gtype:<6} {ki:<5} {tfpwa_a.real:+.4f}{tfpwa_a.imag:+.4f}j  {amp_a.real:+.4f}{amp_a.imag:+.4f}j  {tag:>18}")

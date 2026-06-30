#!/usr/bin/env python3
"""Compare Ai/Aibar (TFPWA) vs a (ampfit), mapping by signature, not position."""
import numpy as np, json, glob, sys
sys.path.insert(0,'src')

p = '/home/jiangy/ana/test_4pi/test_amp/'
with open(p + "data_all_comb.json") as f: all_comb = json.load(f)
data_all = np.concatenate([np.load(f) for f in sorted(glob.glob(p+"pw_amp/pw_*/data_all_amp.npy"),
    key=lambda x:int(x.split('/')[-2].split('_')[-1]))], axis=-1)
print(f"Data: {data_all.shape} = {data_all.shape[1]//2} complex entries")

# Truncate to match kernel's event count (1000)
n_test = 1000
data = data_all[:n_test]

# ── Build correct matching: signature → (ri, data_col) ──
# Reference data columns correspond to all_comb in order.
# So combo ri corresponds to data cols (2*ri, 2*ri+1).
ref_by_sig = {}
for ri in range(112):
    sig = tuple(all_comb[ri][1:])
    ref_by_sig[sig] = ri

# ── Ampfit ──
from ampfit.config_loader import Config
from ampfit.backends import create_backend
import ampfit.fitter as ft
from run_fit import build_constraints

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

# Load first N events directly (no random subsampling)
raw = np.load('data/data_arrays.npz')
n_test = 1000
m = raw['mass'][:n_test].reshape(n_test, -1)
q = raw['q'][:n_test].reshape(n_test, -1)
angle = raw['angles'][:n_test].reshape(n_test, -1, 3)
# frac, time also needed but not for K factor
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
kin_factor = (1.0/bw_p) * fa * fl_p
a = ck * kin_factor

amp_map = config.get_ck_map()
n_unique = 56

# Build signature → kernel entry
ker_by_sig = {}
for ki in range(n_unique):
    sig = tuple(amp_map[ki][1:])  # g_ls
    ker_by_sig[sig] = (ki, 0)
    sig2 = tuple(amp_map[ki+4*n_unique][1:])  # g_lsbar
    ker_by_sig[sig2] = (ki, 1)

# ── Compare per combo, across ALL events ──
print(f"\n{'ri':<5} {'type':<6} {'ki':<5} {'median r':>18} {'med|r-1|':>10}  name")
print("-"*70)
stats_gls, stats_glsb = [], []
for ri in range(112):
    sig = tuple(all_comb[ri][1:])
    if sig not in ker_by_sig:
        print(f"{ri:<5} NO MATCH")
        continue
    ki, orient = ker_by_sig[sig]
    gtype = "g_ls" if orient == 0 else "glsb"
    
    # TFPWA K across all events
    ref_k = data[:, 2*ri] + 1j*data[:, 2*ri+1]
    
    # Ampfit K across all events (sum over 4 perms)
    if orient == 0:
        amp_k = sum(kin_factor[:, ki + perm*n_unique] for perm in range(4))
    else:
        amp_k = sum(kin_factor[:, ki + (perm+4)*n_unique] for perm in range(4))
    
    mask = np.abs(ref_k) > 1e-12
    if np.any(mask):
        r = amp_k[mask] / ref_k[mask]
        mr = np.median(r)
        md = np.median(np.abs(r - 1))
        name = all_comb[ri][0][:35]
        tag = " ✓" if md < 0.02 else ""
        print(f"{ri:<5} {gtype:<6} {ki:<5} {mr.real:+.4f}{mr.imag:+.4f}j  {md:.4f}{tag}  {name}")
        if orient == 0:
            stats_gls.append(md)
        else:
            stats_glsb.append(md)

print(f"\n=== Summary ===")
print(f"g_ls:    {len(stats_gls)} entries, median|r-1|={np.median(stats_gls):.4f}")
print(f"g_lsbar: {len(stats_glsb)} entries, median|r-1|={np.median(stats_glsb):.4f}")
EOF
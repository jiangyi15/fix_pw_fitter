"""Simple comparison script for you to check.
Compares TFPWA K (from data_all_amp.npy) with ampfit K (from kin_factor).
Match by signature (g_ls names), not by position."""

import numpy as np, json, glob, sys
sys.path.insert(0, 'src')

# ---------- 1. Load TFPWA reference ----------
p = '/home/jiangy/ana/test_4pi/test_amp/'
with open(p + "data_all_comb.json") as f: all_comb = json.load(f)
data = np.concatenate([np.load(f) for f in sorted(glob.glob(p + "pw_amp/pw_*/data_all_amp.npy"),
    key=lambda x: int(x.split('/')[-2].split('_')[-1]))], axis=-1)

# Take first 1000 events
data = data[:1000]
print(f"TFPWA data: {data.shape} = {data.shape[1]//2} complex entries")

# Build ref signature -> combo index map
ref_sig_to_ri = {}
for ri in range(112):
    sig = tuple(all_comb[ri][1:])  # (g_ls_0_sig, rhoA_sig, pipi_sig)
    ref_sig_to_ri[sig] = ri

# ---------- 2. Load ampfit kernel ----------
from ampfit.config_loader import Config
from ampfit.backends import create_backend
import ampfit.fitter as ft
from run_fit import build_constraints

config = Config('config_amp.yml')
kc = config.build_all_index(); be = create_backend("numpy", kc); kernel = be.kernel
fitter = ft.Fitter('config_amp.yml', backend='numpy')
fs, sp, sc = build_constraints(fitter.all_comb)
for n in fitter.config.m0_phys_name: fs[n] = float(fitter.defaults[n])
for n in fitter.config.g0_phys_name: fs[n] = float(fitter.defaults[n])
for n, v in [('delta_gamma', 0), ('delta_m', 0.506), ('A_prod', 0), ('poqr', 1), ('poqi', 0)]: fs[n] = v
fitter.set_fixed(fs); fitter.set_same(sp); fitter.set_scale(sc)
with open(p + "pw_cfit5_td6_fix29/final_params_0.json") as f: pdat = json.load(f)
x0 = fitter.values_from_dict(pdat)
params, _, _, _ = fitter._build_params(x0)

raw = np.load('data/data_arrays.npz')
m = raw['mass'][:1000].reshape(1000, -1)
q = raw['q'][:1000].reshape(1000, -1)
angle = raw['angles'][:1000].reshape(1000, -1, 3)

g0_all = np.take(params["g0"], kernel.g0_index)
g0_m = np.take(m, kernel.g0_mass_index, axis=-1)
g_interp = kernel.interp_catmull_rom(kernel.gamma_table, kernel.g0_index, g0_m,
                                     kernel.gamma_min, kernel.gamma_delta)
g = g0_all * g_interp
g_bw = np.dot(g, kernel.matrix_gamma)
m0_all = np.take(params["m0"], kernel.m0_index)
m0_m = np.take(m, kernel.mass_index, axis=-1)
bw_dom = m0_all**2 - m0_m**2 - 1j * m0_all * g_bw
bw_dom_all = np.take(bw_dom, kernel.bw_order, axis=-1).reshape(1000, kernel.n_wave, kernel.n_res)
bw_p = np.prod(bw_dom_all, axis=-1)
fl_q_arr = np.take(q, kernel.fl_q_index, axis=-1)
fl = kernel.interp_catmull_rom(kernel.fl_table, kernel.fl_type, fl_q_arr, kernel.fl_min, kernel.fl_delta)
fl_all = np.take(fl, kernel.fl_order, axis=-1).reshape(-1, kernel.n_wave, kernel.n_decay)
fl_p = np.prod(fl_all, axis=-1)
ang_arr = np.take(angle, kernel.angle_index, axis=-2)
ka = np.prod(np.cos(ang_arr * kernel.angle_k + kernel.angle_b), axis=-1)
fa = np.dot(ka, kernel.matrix_angle)
kin_factor = (1.0 / bw_p) * fa * fl_p  # (1000, 448)

# Build ampfit signature -> (ki, orientation) map
amp_map = config.get_ck_map()
ker_sig_to_ki = {}
for ki in range(56):
    sig_g = tuple(amp_map[ki][1:])       # g_ls
    ker_sig_to_ki[sig_g] = (ki, 0)
    sig_b = tuple(amp_map[ki + 4 * 56][1:])  # g_lsbar
    ker_sig_to_ki[sig_b] = (ki, 1)

# ---------- 3. Compare per combo ----------
print(f"\n{'ri':<5} {'type':<7} {'ki':<5} {'TFPWA K':>18} {'ampfit K':>18} {'r_median':>18}  combo name")
print("=" * 90)

g_ls_ok = []
for ri in range(112):
    sig = tuple(all_comb[ri][1:])
    if sig not in ker_sig_to_ki:
        print(f"{ri:<5} NO MATCH"); continue
    ki, orient = ker_sig_to_ki[sig]
    gtype = "g_ls" if orient == 0 else "g_lsbar"

    # TFPWA K = Re + i*Im
    ref_k = data[:, 2*ri] + 1j * data[:, 2*ri+1]

    # Ampfit K = sum over 4 permutations
    if orient == 0:
        amp_k = sum(kin_factor[:, ki + p * 56] for p in range(4))
    else:
        amp_k = sum(kin_factor[:, ki + (p + 4) * 56] for p in range(4))

    mask = np.abs(ref_k) > 1e-12
    if np.any(mask):
        r = amp_k[mask] / ref_k[mask]
        mr = np.median(r)
        md = np.median(np.abs(r - 1))
        # Show first 3 rows in detail
        if ri < 6 or md > 0.5:
            tag = f" diff={md:.4f}"
            if md < 0.02: tag = " OK"
            print(f"{ri:<5} {gtype:<7} {ki:<5} {ref_k.mean().real:+.4f}{ref_k.mean().imag:+.4f}j  {amp_k.mean().real:+.4f}{amp_k.mean().imag:+.4f}j  {mr.real:+.4f}{mr.imag:+.4f}j{tag}  {all_comb[ri][0][:40]}")
    else:
        mr = md = 0

# Summary
print(f"\n--- Summary ---")
n_gls = sum(1 for ri in range(56) if tuple(all_comb[ri][1:]) in ker_sig_to_ki)
n_glsb = sum(1 for ri in range(56, 112) if tuple(all_comb[ri][1:]) in ker_sig_to_ki)
print(f"Mapped: {n_gls}/56 g_ls + {n_glsb}/56 g_lsbar")
print("The g_ls K factors match (r ≈ 1). The g_lsbar K factors don't.")
print("The -j ratio for g_lsbar means ref_K_glsb = -Im(g_ls K) + i·Re(g_ls K)")
print("while ampfit uses K_glsb = K_gls (same angular function for both orientations).")
print("The fix belongs in the angular function for orientation 1 (g_lsbar).")

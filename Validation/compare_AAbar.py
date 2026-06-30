#!/usr/bin/env python3
"""Compare A/Abar between TFPWA reference and ampfit kernel"""
import numpy as np, json, glob, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── 1. TFPWA A/Abar ──
p = '/home/jiangy/ana/test_4pi/test_amp/'
with open(p + "data_all_comb.json") as f: all_comb = json.load(f)
import yaml
with open(p + "config_amp.yml") as f: config_amp = yaml.full_load(f)

# Setup params (minimal)
all_params = []; fixed_params = {}
for i in all_comb:
    for j in i:
        if j not in all_params:
            all_params.append(j)
            if j.endswith("g_ls_0"): fixed_params[j] = 1+0j
            elif j.endswith("pole.0"): fixed_params[j] = 1+0j
            elif j.endswith("point_5"): fixed_params[j] = 1+0j
            elif j.endswith("fix1"): fixed_params[j] = 1+0j
fix_total = "B->rhoA.rhoBrhoA->pip1.pim1rhoB->pip2.pim2_total_0"
fixed_params[fix_total] = 1+0j

for r1 in ["a1(1260)", "a1(1640)","a2(1320)", "pi1300", "pi1600", "a2(1700)", "pi2(1670)", "pi1(1600)"]:
    name_ps=[]; name_ms=[]; fixed=True
    for r2 in ["rhoA", "f0(500)", "f0(980)", "f2(1270)"]:
        name_p=f"B->{r1}p.pim2{r1}p->{r2}.pip2{r2}->pip1.pim1_total_0"
        name_m=f"B->{r1}m.pip2{r1}m->{r2}.pim2{r2}->pip1.pim1_total_0"
        if name_p in all_params: name_ps.append(name_p); name_ms.append(name_m)
        for idx in range(3):
            n = f"{r1}m->{r2}.pim2_g_ls_{idx}"
            if n in all_params:
                if fixed: fixed=False
                else:
                    for k in [n, f"{r1}p->{r2}.pip2_g_ls_{idx}"]:
                        if k in fixed_params: del fixed_params[k]
    same_params = []  # simplified

all_used_x = [i for i in all_params if i not in fixed_params]

# Load data
data = np.concatenate([np.load(f) for f in sorted(glob.glob(p+"pw_amp/pw_*/data_all_amp.npy"),
    key=lambda x:int(x.split('/')[-2].split('_')[-1]))], axis=-1)

with open(p+"pw_cfit5_td6_fix29/final_params_0.json") as f: pdat = json.load(f)

# Scale params from reference
scale_params = {'a1(1260)m->rhoA.pim2_g_ls_0':-1,'a1(1260)m->rhoA.pim2_g_ls_1':-1,
    'a1(1640)m->rhoA.pim2_g_ls_0':-1,'a1(1640)m->rhoA.pim2_g_ls_1':-1,
    'a2(1320)m->rhoA.pim2_g_ls_0':-1,'pi1300m->rhoA.pim2_g_ls_0':-1,
    'pi1600m->rhoA.pim2_g_ls_0':-1,'pi2(1670)m->rhoA.pim2_g_ls_0':-1,
    'pi2(1670)m->rhoA.pim2_g_ls_1':-1,'pi1(1600)m->rhoA.pim2_g_ls_0':-1}

def get_val(name):
    val = float(pdat["value"].get(name, 0))
    for p, s in scale_params.items():
        if name == p + 'r' and s != 0: val /= s
    return val

# Build ck from JSON params
ck_ref = np.zeros(112, dtype=np.complex128)
for ri, c in enumerate(all_comb):
    prod = 1+0j
    for term in c:
        if isinstance(term, str):
            if term in fixed_params:
                prod *= complex(fixed_params[term])
            elif term in all_used_x:
                r = get_val(term+"r")
                phi = get_val(term+"i")
                prod *= r * np.exp(1j * phi)
            # else: one of the total names — skip (handled by all_comb)
        else:
            prod *= term  # scale factors
    ck_ref[ri] = prod

# Reference A, Abar
n_test = 1000
Ai = data[:n_test, ::2]   # (N, 112) Re parts
Aibar = data[:n_test, 1::2]  # (N, 112) Im parts
A_ref = Ai @ ck_ref  # complex
Abar_ref = Aibar @ ck_ref  # complex

print("=== TFPWA A/Abar (first 5 events) ===")
for i in range(5):
    print(f"  evt{i}: A={A_ref[i].real:.6f}{A_ref[i].imag:+.6f}j  Abar={Abar_ref[i].real:.6f}{Abar_ref[i].imag:+.6f}j")

# ── 2. Ampfit ap/am ──
from ampfit.config_loader import Config as AmpConfig
from ampfit.backends import create_backend
import ampfit.fitter as ft
from run_fit import build_constraints

config_ampfit = AmpConfig(os.path.join(os.path.dirname(__file__), 'config_angle.yml'))
kc = config_ampfit.build_all_index()
be = create_backend("numpy", kc); kernel = be.kernel

fitter=ft.Fitter(os.path.join(os.path.dirname(__file__), 'config_angle.yml'),backend='numpy')
fs,sp,sc=build_constraints(fitter.all_comb)
for n in fitter.config.m0_phys_name:
        if n in fitter.defaults: fs[n]=float(fitter.defaults[n])
for n in fitter.config.g0_phys_name:
        if n in fitter.defaults: fs[n]=float(fitter.defaults[n])
for n,v in [('delta_gamma',0),('delta_m',0.506),('A_prod',0),('poqr',1),('poqi',0)]: fs[n]=v
fitter.set_fixed(fs);fitter.set_same(sp);fitter.set_scale(sc)

# Load data the ampfit way
from ampfit import Fitter; load_npz = Fitter.load_npz
data_np, _ = load_npz('data/data_arrays.npz', max_events=n_test)
m = data_np['mass']; q = data_np['q']; angle = data_np['angle']

# Get kernel params from the same JSON
x0 = fitter.values_from_dict(pdat)
params,_,_,_ = fitter._build_params(x0)
ck = params["ck"]; m0_p = params["m0"]; g0_p = params["g0"]

# Kernel forward pass
g0_all = np.take(g0_p, kernel.g0_index)
g0_m = np.take(m, kernel.g0_mass_index, axis=-1)
g_interp = kernel.interp_catmull_rom(kernel.gamma_table, kernel.g0_index, g0_m,
                                     kernel.gamma_min, kernel.gamma_delta)
g = g0_all * g_interp; g_bw = np.dot(g, kernel.matrix_gamma)
m0_all = np.take(m0_p, kernel.m0_index); m0_m = np.take(m, kernel.mass_index, axis=-1)
bw_dom = m0_all**2 - m0_m**2 - 1j*m0_all*g_bw
bw_dom_all = np.take(bw_dom, kernel.bw_order, axis=-1).reshape(n_test, kernel.n_wave, kernel.n_res)
bw_p = np.prod(bw_dom_all, axis=-1)
fl_q_arr = np.take(q, kernel.fl_q_index, axis=-1)
fl = kernel.interp_catmull_rom(kernel.fl_table, kernel.fl_type, fl_q_arr,
                               kernel.fl_min, kernel.fl_delta)
fl_all = np.take(fl, kernel.fl_order, axis=-1).reshape(-1, kernel.n_wave, kernel.n_decay)
fl_p = np.prod(fl_all, axis=-1)
ang_arr = np.take(angle, kernel.angle_index, axis=-2)
ka = np.prod(np.cos(ang_arr * kernel.angle_k + kernel.angle_b), axis=-1)
fa = np.dot(ka, kernel.matrix_angle)
K = (1.0/bw_p) * fa * fl_p  # (N, 448)

# Compute kernel's total amplitude a = ck * K
# ck has shape (448,) - one per entry
# K has shape (N, 448)
amp = ck * K  # (N, 448)

# Split into ap (g_ls group, first 224 entries) and am (g_lsbar group, last 224 entries)
# But also need to sum over 4 perms within each group
# n_wave = 448, perms = 4, n_unique = 56, 2 orientations
n_unique = 56
ap_ker = np.sum(amp[:, :4*n_unique].reshape(n_test, n_unique, 4), axis=-1)  # sum over 4 perms
am_ker = np.sum(amp[:, 4*n_unique:].reshape(n_test, n_unique, 4), axis=-1)

# Total ap = sum over all 56 wave types
ap_ker_total = np.sum(ap_ker, axis=-1)  # (N,)
am_ker_total = np.sum(am_ker, axis=-1)

print("\n=== Ampfit ap/am (first 5 events) ===")
for i in range(5):
    print(f"  evt{i}: ap={ap_ker_total[i].real:.6f}{ap_ker_total[i].imag:+.6f}j  am={am_ker_total[i].real:.6f}{am_ker_total[i].imag:+.6f}j")

# ── 3. Compare ──
# Reference: A = Σ Re(K)*ck, Abar = Σ Im(K)*ck
# Kernel: ap_total = Σ K_gls*ck (complex), am_total = Σ K_glsb*ck (complex)
# 
# Since the reference stores complex K as (Re, Im) pairs:
# A_ref = Σ Re(K_i)*ck_i
# Abar_ref = Σ Im(K_i)*ck_i
# 
# While the kernel computes:
# ap_ker_total = Σ K_gls_i * ck_i  where K is complex
# am_ker_total = Σ K_glsb_i * ck_i where K is complex
#
# The reference uses A and Abar in the P formula (see lines 532-539):
# P1 = (|A|^2 + r^2|Abar|^2) * cosh
# P2 = (|A|^2 - r^2|Abar|^2) * ct
# ...
# P = P1 + P2 - 2r*P3 - 2r*P4
# Pbar = P1/r2 - P2/r2 - 2/r*P3 + 2/r*P4
#
# The kernel uses ap and am in nearly the same formula (lines 172-186):
# pap = gp*ap + gm*poq*am
# pam = gm/poq*ap + gp*am
# pb = |pap|^2
# pbbar = |pam|^2
# P = frac * pb * (1-Ap) + (1-frac) * pbbar * (1+Ap)
#
# These are structurally similar but with different variable mappings

print("\n=== Pointwise comparison (first 5 events) ===")
for i in range(5):
    A_r, A_i = A_ref[i].real, A_ref[i].imag
    Abar_r, Abar_i = Abar_ref[i].real, Abar_ref[i].imag
    ap_r, ap_i = ap_ker_total[i].real, ap_ker_total[i].imag
    am_r, am_i = am_ker_total[i].real, am_ker_total[i].imag
    
    print(f"\n  evt{i}:")
    print(f"    TFPWA: A={A_r:+.4f}{A_i:+.4f}j  |A|={abs(A_ref[i]):.4f}")
    print(f"           Abar={Abar_r:+.4f}{Abar_i:+.4f}j  |Abar|={abs(Abar_ref[i]):.4f}")
    print(f"    KERNEL: ap={ap_r:+.4f}{ap_i:+.4f}j  |ap|={abs(ap_ker_total[i]):.4f}")
    print(f"            am={am_r:+.4f}{am_i:+.4f}j  |am|={abs(am_ker_total[i]):.4f}")

# The reference A/Abar combine g_ls and g_lsbar contributions via Re/Im split
# The kernel separates them into ap/am
# So we need to compare:
#   TFPWA A  vs  kernel(ap for g_ls parts in Re + am for g_lsbar parts in Re?)
# 
# Actually, let me check: in TFPWA the data stores complex K values.
# For each of the 112 entries, data has 2 columns: Re, Im.
# Ai = Re parts for ALL 112 entries (both g_ls and g_lsbar)
# Aibar = Im parts for ALL 112 entries
# 
# So A uses ALL Re parts (both g_ls and g_lsbar), Abar uses ALL Im parts.
# This is different from the kernel which separates by g_ls/g_lsbar.

# Let me also check: what if the kernel's ap (g_ls complex) maps to TFPWA's A, 
# and kernel's am (g_lsbar complex) maps to TFPWA's Abar?

# Check mapping: for a given event, what's the relationship?
print("\n=== Relationship check ===")
for i in range(5):
    # Try: kernel_total = ap + am (sum g_ls + g_lsbar)
    ker_total = ap_ker_total[i] + am_ker_total[i]
    # Reference complex = 
    ref_complex = A_ref[i] + 1j * Abar_ref[i]  # this would be Σ (Re + i*Im)*ck
    print(f"  evt{i}: ker_total={ker_total.real:.4f}{ker_total.imag:+.4f}j  ref_cplx={ref_complex.real:.4f}{ref_complex.imag:+.4f}j")
    print(f"         r={ker_total/ref_complex if abs(ref_complex)>1e-12 else 0:.4f}")

# Also compare the P (probability) from both frameworks
print("\n=== P value comparison ===")
time_ref = np.load(p+config_amp["data"]["data_time"][0])[:n_test]
eta_ref = np.load(p+config_amp["data"]["data_com_eta1"][0])[:n_test]
tag_ref = np.load(p+config_amp["data"]["data_com_tag1"][0])[:n_test]
bbar_frac = np.where(tag_ref>0, eta_ref, 1-eta_ref)

# Reference time-dep formula (from their code lines 532-539)
r = 1.0; phi = 0.0; dm = 0.506; dG = 0.0; gamma = 0.0; Ap = 0.0
fa = (1+Ap); fabar = (1-Ap)
r2 = r**2

cht = np.cosh(dG * time_ref / 2)
ct = np.cos(dm * time_ref)
sht = np.sinh(dG * time_ref / 2)
st = np.sin(dm * time_ref)
expt = np.exp(-gamma * time_ref)

Asq_ref = np.abs(A_ref)**2
Abarsq_ref = np.abs(Abar_ref)**2
AAbar_ref = np.conj(A_ref) * Abar_ref

P1 = (Asq_ref + r2 * Abarsq_ref) * cht
P2 = (Asq_ref - r2 * Abarsq_ref) * ct
P3 = (np.cos(phi)*np.real(AAbar_ref) - np.sin(phi)*np.imag(AAbar_ref)) * sht
P4 = (np.cos(phi)*np.imag(AAbar_ref) + np.sin(phi)*np.real(AAbar_ref)) * st
P_ref = P1 + P2 - 2*r*P3 - 2*r*P4
Pbar_ref = P1/r2 - P2/r2 - 2/r*P3 + 2/r*P4
Psig_ref = ((1-bbar_frac)*fa*P_ref + bbar_frac*fabar*Pbar_ref) * expt

# Kernel time-dep formula
Gamma=0.0; Delta_Gamma=0.0; Delta_m=0.506; A_p=0.0; poq_rho=1.0; pop_phi=0.0
poq = poq_rho * np.exp(-1j * pop_phi)

eL = np.exp(-1j * time_ref * (-Delta_m/2 - 1j*(Gamma + Delta_Gamma/2)/2))
eH = np.exp(-1j * time_ref * (+Delta_m/2 - 1j*(Gamma - Delta_Gamma/2)/2))
gp = (eL + eH) / 2
gm = (eL - eH) / 2

pap = gp * ap_ker_total + gm * poq * am_ker_total
pam = (gm / poq) * ap_ker_total + gp * am_ker_total
pb = np.abs(pap)**2
pbbar = np.abs(pam)**2

frac_arr = data_np['frac'][:n_test]
Psig_ker = frac_arr * pb * (1 - A_p) + (1 - frac_arr) * pbbar * (1 + A_p)

# Both should be similar after normalization
for i in range(5):
    s_ref = Psig_ref[i]
    s_ker = Psig_ker[i]
    print(f"  evt{i}: Psig_ref={s_ref:.4e}  Psig_ker={s_ker:.4e}  ratio={s_ker/s_ref:.4f}" if s_ref>0 else f"  evt{i}: Psig_ref={s_ref:.4e}  Psig_ker={s_ker:.4e}")

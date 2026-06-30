"""Round-trip test: load params from JSON, build ck, save and compare"""
import numpy as np, json, sys
sys.path.insert(0, 'src')
import tensorflow as tf
import yaml

p = '/home/jiangy/ana/test_4pi/test_amp/'

with open(p + "data_all_comb.json") as f:
    all_comb = json.load(f)
with open(p + "config_amp.yml") as f:
    config_amp = yaml.full_load(f)

all_time_params = {
    "gamma": 0.0, "delta_m": 0.506, "delta_gamma": 0.0,
    "poqr": 1.0, "poqi": 0.0, "A_prod": 0.0,
}
fix_time_params = ["A_prod", "delta_gamma", "delta_m", "poqr", "poqi"]
free_time_params = [i for i in all_time_params if i not in fix_time_params]

all_params = []
same_params = []
scale_params = {}
fixed_params = {}
for i in all_comb:
    for j in i:
        if j not in all_params:
            all_params.append(j)
            if j.endswith("g_ls_0"):
                fixed_params[j] = 1+0j
            elif j.endswith("pole.0"):
                fixed_params[j] = 1+0j
            elif j.endswith("point_5"):
                fixed_params[j] = 1+0j
            elif j.endswith("fix1"):
                fixed_params[j] = 1+0j

fix_total = "B->rhoA.rhoBrhoA->pip1.pim1rhoB->pip2.pim2_total_0"
fixed_params[fix_total] = 1+0j

for r1 in ["a1(1260)", "a1(1640)","a2(1320)", "pi1300", "pi1600", "a2(1700)", "pi2(1670)", "pi1(1600)"]:
    name_ps = []; name_ms = []; fixed = True
    for r2 in ["rhoA", "f0(500)", "f0(980)", "f2(1270)"]:
        name_p = f"B->{r1}p.pim2{r1}p->{r2}.pip2{r2}->pip1.pim1_total_0"
        name_m = f"B->{r1}m.pip2{r1}m->{r2}.pim2{r2}->pip1.pim1_total_0"
        if name_p in all_params:
            name_ps.append(name_p); name_ms.append(name_m)
        for idx in range(3):
            if f"{r1}m->{r2}.pim2_g_ls_{idx}" in all_params:
                if fixed:
                    fixed = False
                else:
                    if f"{r1}m->{r2}.pim2_g_ls_{idx}" in fixed_params:
                        del fixed_params[f"{r1}m->{r2}.pim2_g_ls_{idx}"]
                    if f"{r1}p->{r2}.pip2_g_ls_{idx}" in fixed_params:
                        del fixed_params[f"{r1}p->{r2}.pip2_g_ls_{idx}"]
                    same_params.append([f"{r1}m->{r2}.pim2_g_ls_{idx}", f"{r1}p->{r2}.pip2_g_ls_{idx}"])
                if r2 == "rhoA":
                    scale_params[f"{r1}m->{r2}.pim2_g_ls_{idx}"] = -1
    same_params.append(name_ps); same_params.append(name_ms)

for i in range(3):
    for j in ["pole", "prod"]:
        if f"KMA_{j}.{i}" in all_params:
            same_params.append([f"KMA_{j}.{i}", f"KMB_{j}.{i}"])
for i in range(3,5):
    for j in ["pole", "prod"]:
        for k in ["KMA", "KMB", "KMC", "KM2"]:
            if f"{k}_{j}.{i}" in all_params:
                fixed_params[f"{k}_{j}.{i}"] = 0.0

all_used_x = [i for i in all_params if i not in fixed_params]
new_name = {}
for i in same_params:
    for j in i[1:]:
        if j in all_used_x: all_used_x.remove(j)
        new_name[j] = i[0]

# Build new_all_comb with scale factors
new_all_comb = []
for i in all_comb:
    tmp = []
    for j in i:
        tmp.append(new_name.get(j, j))
        if j in scale_params:
            tmp.append(scale_params[j])
    new_all_comb.append(tmp)
all_comb = new_all_comb

# Load JSON
with open(p + "pw_cfit5_td6_fix29/final_params_0.json") as f:
    pdat = json.load(f)

print(f"Total free params: {len(all_used_x)}")
print(f"Fixed params: {len(fixed_params)}")
print(f"Scale params: {len(scale_params)}")
print(f"All combos: {len(all_comb)}")
print()

# Build x vector — polar form (r, phi)
x_y = np.array([[pdat["value"].get(i+"r", 0), pdat["value"].get(i+"i", 0)] for i in all_used_x]).reshape(-1)
tp = np.array([float(pdat["value"].get(n, all_time_params[n])) for n in free_time_params])

# Build ck using build_par
x_tf = tf.constant(x_y.reshape(-1, 2))
xc = tf.complex(x_tf[:,0]*tf.cos(x_tf[:,1]), x_tf[:,0]*tf.sin(x_tf[:,1]))
lst = tf.unstack(xc)
new_params = {n: lst[i] for i, n in enumerate(all_used_x)}

ck = []
for i in all_comb:
    tmp = tf.cast(1.0, tf.complex128)
    for j in i:
        if not isinstance(j, str):
            tmp = tmp * tf.cast(j, tf.complex128)
        elif j not in fixed_params:
            tmp = tmp * new_params[j]
        else:
            tmp = tmp * tf.cast(fixed_params[j], tf.complex128)
    ck.append(tmp)
ck_tf = tf.stack(ck)

print(f"ck vector: {ck_tf.numpy().shape}")
print(f"First 5 ck values:")
for i in range(5):
    print(f"  [{i}] {ck_tf.numpy()[i].real:.6f}{ck_tf.numpy()[i].imag:+.6f}j")

# Compare with kernel's ck for verification
sys.path.insert(0, 'src')
from ampfit.config_loader import Config
from ampfit.numpy_kernel import NumpyKernel as Kernel
from run_fit import build_constraints
import ampfit.fitter as ft

config = Config('config_amp.yml')
fitter=ft.Fitter('config_amp.yml',backend='numpy')
fs,sp,sc=build_constraints(fitter.all_comb)
for n in fitter.config.m0_phys_name: fs[n]=float(fitter.defaults[n])
for n in fitter.config.g0_phys_name: fs[n]=float(fitter.defaults[n])
for n,v in [('delta_gamma',0),('delta_m',0.506),('A_prod',0),('poqr',1),('poqi',0)]: fs[n]=v
fitter.set_fixed(fs);fitter.set_same(sp);fitter.set_scale(sc)
x0=fitter.values_from_dict(pdat)
params,_,_,_=fitter._build_params(x0)
ck_kernel = params["ck"]

# The kernel's ck is for 56 wave types (not 112 combos)
# The reference's ck is for 112 combos
# They should match when summing over 4 perms

amp_map = config.get_ck_map()
print("\nComparing kernel ck vs reference build_par ck:")
for ki in range(5):
    # Find which ref entries match this kernel wave
    sig_gls = tuple(amp_map[ki][1:])
    for ri, c in enumerate(all_comb):
        # Compare: ref ck[ri] (all params product) vs kernel's ck[ki]
        pass

# Compare specific: first few ref ck entries directly
print("\nbuild_par ck by comb index:")
for ri in range(5):
    print(f"  comb[{ri}]: {all_comb[ri]}")
    print(f"          ck = {ck_tf.numpy()[ri].real:.6f}{ck_tf.numpy()[ri].imag:+.6f}j")

# Save ck and reload to check round-trip
# Reference save format: value["name_r"] = r, value["name_i"] = phi
# where r = abs(ck), phi = phase(ck)
print("\nRound-trip check (first 5 free params):")
for i in range(min(5, len(all_used_x))):
    name = all_used_x[i]
    stored_r = pdat["value"].get(name+"r", 0)
    stored_phi = pdat["value"].get(name+"i", 0)
    # build_par: ck = r * exp(i*phi)
    ck_from_load = stored_r * np.cos(stored_phi) + 1j * stored_r * np.sin(stored_phi)
    idx = all_comb[0].index(name) if name in all_comb[0] else -1
    print(f"  {name}: stored(r={stored_r:.4f}, φ={stored_phi:.4f})  ck={ck_from_load.real:.4f}{ck_from_load.imag:+.4f}j")

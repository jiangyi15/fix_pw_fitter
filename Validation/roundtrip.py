#!/usr/bin/env python3
"""Load params from JSON, build ck, save back, compare — verify round-trip"""
import numpy as np, json, yaml, os
import tensorflow as tf

os.chdir('/home/jiangy/ana/test_4pi/test_amp')

with open("data_all_comb.json") as f: all_comb = json.load(f)

# ── Setup (matching reference exactly) ──
all_params = []; same_params = []; scale_params = {}; fixed_params = {}
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
                    same_params.append([n, f"{r1}p->{r2}.pip2_g_ls_{idx}"])
                if r2=="rhoA": scale_params[n] = -1
    same_params.append(name_ps); same_params.append(name_ms)

for i in range(3):
    for j in ["pole", "prod"]:
        if f"KMA_{j}.{i}" in all_params:
            same_params.append([f"KMA_{j}.{i}", f"KMB_{j}.{i}"])
for i in range(3,5):
    for j in ["pole", "prod"]:
        for k in ["KMA","KMB","KMC","KM2"]:
            if f"{k}_{j}.{i}" in all_params: fixed_params[f"{k}_{j}.{i}"]=0.0

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
        if j in scale_params: tmp.append(scale_params[j])
    new_all_comb.append(tmp)
all_comb = new_all_comb

print(f"Free params: {len(all_used_x)}")
print(f"Scale params: {scale_params}")

# ── build_par: complex ck from polar (r, phi) ──
def build_par(x):
    x = tf.reshape(x, (-1, 2))
    xc = tf.complex(x[:,0]*tf.cos(x[:,1]), x[:,0]*tf.sin(x[:,1]))
    lst = tf.unstack(xc)
    new_params = {n: lst[i] for i,n in enumerate(all_used_x)}
    ret = []
    for i in all_comb:
        tmp = tf.cast(tf.constant(1.0), tf.complex128)
        for j in i:
            if not isinstance(j, str):
                tmp = tmp * tf.cast(j, tf.complex128)
            elif j not in fixed_params:
                tmp = tmp * new_params[j]
            else:
                tmp = tmp * tf.cast(fixed_params[j], tf.complex128)
        ret.append(tmp)
    return tf.stack(ret)

# ── Load params (with scale reversal, matching ampfit values_from_dict) ──
with open("pw_cfit5_td6_fix29/final_params_0.json") as f: pdat = json.load(f)

def get_val(name):
    val = float(pdat["value"].get(name, 0))
    # Reverse scale: JSON stores SCALED values, optimizer needs unscaled
    for p, s in scale_params.items():
        if name == p + 'r' and s != 0:
            val /= s
    return val

x_y = np.array([[get_val(i+"r"), get_val(i+"i")] for i in all_used_x]).reshape(-1)

# ── Build ck ──
ck_ref = build_par(tf.constant(x_y)).numpy()
print(f"ck vector: {ck_ref.shape}")

# ── Round-trip: save ck back using reference format ──
# Reference stores all_used_x params as polar (r, phi)
# And also saves fixed_params, new_name aliases, scale_params
# Let me rebuild what the reference save does:

# Step 1: Build the output dict like the reference
save_dict = {}

# For each all_used_x param, store from input
for i, name in enumerate(all_used_x):
    r = x_y[2*i]
    phi = x_y[2*i+1] if 2*i+1 < len(x_y) else 0
    save_dict[name+"r"] = float(r)
    save_dict[name+"i"] = float(phi)

# Fixed params: r = abs(value), phi = phase(value)
for name, val in fixed_params.items():
    save_dict[name+"r"] = abs(complex(val))
    save_dict[name+"i"] = np.angle(complex(val))

# Same params via new_name: copy from aliased source
for alias, source in new_name.items():
    save_dict[alias+"r"] = save_dict[source+"r"]
    save_dict[alias+"i"] = save_dict[source+"i"]

# Scale params: apply scale (matching reference save)
for name, scale in scale_params.items():
    save_dict[name+"r"] *= scale

# ── Compare with original ──
print("\n=== Round-trip check ===")
n_diff = 0
for name in list(save_dict.keys())[:50]:
    orig_name = name
    # Check if original JSON has this key
    if name in pdat["value"]:
        orig = pdat["value"][name]
        saved = save_dict[name]
        if abs(orig - saved) > 1e-6:
            print(f"  DIFF {name}: orig={orig:.6f} saved={saved:.6f} delta={abs(orig-saved):.6f}")
            n_diff += 1
    else:
        print(f"  MISS {name}: not in original JSON")

print(f"\nTotal differences found: {n_diff}")

# If no differences, the loading is correct
if n_diff == 0:
    print("✓ ROUND-TRIP VERIFIED — no differences")
else:
    print("⚠ Round-trip has differences — check loading logic")

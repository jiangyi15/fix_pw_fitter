#!/usr/bin/env python3
"""Reproduce reference NLL using TFPWA — MUST give -29656.14"""
import numpy as np, json, glob, yaml, os, sys
import tensorflow as tf

p = '/home/jiangy/ana/test_4pi/test_amp/'
os.chdir(p)

with open("data_all_comb.json") as f: all_comb = json.load(f)
with open("config_amp.yml") as f: config_amp = yaml.full_load(f)

# ── Fixed/free params (exact copy from reference script) ──
all_time_params = {
    "gamma": tf.convert_to_tensor(0.0, tf.float64),
    "delta_m": tf.Variable(tf.convert_to_tensor(0.506, tf.float64)),
    "delta_gamma": tf.convert_to_tensor(0.0, tf.float64),
    "poqr": tf.convert_to_tensor(1.0, tf.float64),
    "poqi": tf.convert_to_tensor(0.0, tf.float64),
    "A_prod": tf.convert_to_tensor(0.0, tf.float64),
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
            if j.endswith("g_ls_0"): fixed_params[j] = 1+0j
            elif j.endswith("pole.0"): fixed_params[j] = 1+0j
            elif j.endswith("point_5"): fixed_params[j] = 1+0j
            elif j.endswith("fix1"): fixed_params[j] = 1+0j

fix_total = "B->rhoA.rhoBrhoA->pip1.pim1rhoB->pip2.pim2_total_0"
fixed_params[fix_total] = 1+0j

for r1 in ["a1(1260)", "a1(1640)","a2(1320)", "pi1300", "pi1600", "a2(1700)", "pi2(1670)", "pi1(1600)"]:
    name_ps = []; name_ms = []; fixed = True
    for r2 in ["rhoA", "f0(500)", "f0(980)", "f2(1270)"]:
        name_p = f"B->{r1}p.pim2{r1}p->{r2}.pip2{r2}->pip1.pim1_total_0"
        name_m = f"B->{r1}m.pip2{r1}m->{r2}.pim2{r2}->pip1.pim1_total_0"
        if name_p in all_params: name_ps.append(name_p); name_ms.append(name_m)
        for idx in range(3):
            if f"{r1}m->{r2}.pim2_g_ls_{idx}" in all_params:
                if fixed: fixed = False
                else:
                    for k in [f"{r1}m->{r2}.pim2_g_ls_{idx}", f"{r1}p->{r2}.pip2_g_ls_{idx}"]:
                        if k in fixed_params: del fixed_params[k]
                    same_params.append([f"{r1}m->{r2}.pim2_g_ls_{idx}", f"{r1}p->{r2}.pip2_g_ls_{idx}"])
                if r2 == "rhoA": scale_params[f"{r1}m->{r2}.pim2_g_ls_{idx}"] = -1
    same_params.append(name_ps); same_params.append(name_ms)

for i in range(3):
    for j in ["pole", "prod"]:
        if f"KMA_{j}.{i}" in all_params:
            same_params.append([f"KMA_{j}.{i}", f"KMB_{j}.{i}"])
for i in range(3,5):
    for j in ["pole", "prod"]:
        for k in ["KMA", "KMB", "KMC", "KM2"]:
            if f"{k}_{j}.{i}" in all_params: fixed_params[f"{k}_{j}.{i}"] = 0.0

all_used_x = [i for i in all_params if i not in fixed_params]
new_name = {}
for i in same_params:
    for j in i[1:]:
        if j in all_used_x: all_used_x.remove(j)
        new_name[j] = i[0]

new_all_comb = []
for i in all_comb:
    tmp = []
    for j in i:
        tmp.append(new_name.get(j, j))
        if j in scale_params: tmp.append(scale_params[j])
    new_all_comb.append(tmp)
all_comb = new_all_comb

# ── build_par (handles scale factors) ──
fix_params_value_vec = np.array(list(fixed_params.values()))
fix_params_name_vec = list(fixed_params.keys())
max_comb_depth = max([len(i) for i in all_comb])
comb_matrix = []
for i in all_comb:
    for j in range(max_comb_depth):
        if len(i) <= j: comb_matrix.append(0)
        else:
            name = i[j]
            if not isinstance(name, str):
                pass  # scale factors handled by build_par, not build_par_prod
            elif name in all_used_x:
                comb_matrix.append(1+all_used_x.index(name))
            else:
                comb_matrix.append(1+len(all_used_x)+fix_params_name_vec.index(name))
comb_matrix = np.array(comb_matrix)

def build_par_prod(x):
    """Used by build_par_jac (reference uses this)"""
    x = tf.reshape(x, (-1, 2))
    xc = tf.complex(x[:,0]*tf.cos(x[:,1]), x[:,0]*tf.sin(x[:,1]))
    empty = tf.complex(tf.constant(1.0), tf.constant(0.0))
    all_xc = tf.concat([[empty], xc, tf.cast(fix_params_value_vec, tf.complex128)], axis=-1)
    cv = tf.gather(all_xc, tf.cast(comb_matrix, tf.int32))
    cv = tf.reshape(cv, (len(all_comb), max_comb_depth))
    ret = tf.reduce_prod(cv, axis=-1)
    return tf.stack([tf.math.real(ret), tf.math.imag(ret)], axis=-1)

def build_par_simple(x):
    """build_par matching reference: handles scale factors by iterating all_comb"""
    x = tf.reshape(x, (-1, 2))
    xc = tf.complex(x[:,0]*tf.cos(x[:,1]), x[:,0]*tf.sin(x[:,1]))
    lst = tf.unstack(xc)
    new_params = {n: lst[i] for i, n in enumerate(all_used_x)}
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
    ret = tf.stack(ret)
    return tf.stack([tf.math.real(ret), tf.math.imag(ret)], axis=-1)

# ── Load data ──
data = []
for f in sorted(glob.glob("pw_amp/pw_*/data_all_amp.npy"), key=lambda x: int(x.split("/")[-2].split("_")[-1])):
    data.append(np.load(f))
data = np.concatenate(data, axis=-1)

sw = np.ones(data.shape[0])
Bi = np.load(config_amp["data"]["data_bg_value"][0]).flatten().astype(np.float64)
NB = np.mean(np.load(config_amp["data"]["phsp_bg_value"][0]).astype(np.float64))
Bi = tf.convert_to_tensor(Bi/NB)

data_time = np.load(config_amp["data"]["data_time"][0])
data_eta = np.load(config_amp["data"]["data_com_eta1"][0])
data_tag = np.load(config_amp["data"]["data_com_tag1"][0])
data_bbar_frac = np.where(data_tag>0, data_eta, 1-data_eta)

phsp_time = np.load(config_amp["data"]["phsp_time"][0])
phsp_time_weight = np.load(config_amp["data"]["phsp_weight"][0]).astype(np.float64)
phsp_time_weight = phsp_time_weight / np.sum(phsp_time_weight)
phsp_time_weight = tf.convert_to_tensor(phsp_time_weight)

purity = config_amp["data"]["bg_frac"]
sw_scale = float(np.sum(sw) / np.sum(sw**2))

# ── Norm integral (M matrix) ──
M_path = "phsp_pw_M2_td3.npy"
if os.path.exists(M_path):
    M = np.load(M_path)
    print(f"Loaded M matrix: {M.shape}, trace(M0)={np.trace(M[:,:,0]):.2f}")
else:
    raise FileNotFoundError(f"Need {M_path}")

# ── inte_sig (matching reference exactly) ──
def inte_sig(y, Mall, time_params):
    cht = tf.math.cosh(phsp_time * time_params["delta_gamma"] / 2)
    ct = tf.math.cos(phsp_time * time_params["delta_m"])
    sht = tf.math.sinh(phsp_time * time_params["delta_gamma"] / 2)
    st = tf.math.sin(phsp_time * time_params["delta_m"])
    expt = tf.math.exp(-phsp_time * time_params["gamma"])
    icht = tf.reduce_sum(phsp_time_weight * cht * expt)
    ict = tf.reduce_sum(phsp_time_weight * ct * expt)
    isht = tf.reduce_sum(phsp_time_weight * sht * expt)
    ist = tf.reduce_sum(phsp_time_weight * st * expt)
    idcht = tf.reduce_sum(phsp_time_weight * phsp_time * expt * sht) / 2
    idct = -tf.reduce_sum(phsp_time_weight * phsp_time * expt * st)
    idsht = tf.reduce_sum(phsp_time_weight * phsp_time * expt * cht) / 2
    idst = tf.reduce_sum(phsp_time_weight * phsp_time * expt * ct)

    MM = Mall[:,:,0]; Mbar = Mall[:,:,1]; Mf = Mall[:,:,2]
    M_a = tf.reduce_sum(MM * tf.math.conj(y), axis=-1)
    Mbar_a = tf.reduce_sum(Mbar * tf.math.conj(y), axis=-1)
    Mf_a = tf.reduce_sum(Mf * tf.math.conj(y), axis=-1)
    Mfbar_a = tf.reduce_sum(tf.math.conj(Mf) * tf.math.conj(y)[:,None], axis=0)
    M_s = tf.reduce_sum(M_a * y); Mbar_s = tf.reduce_sum(Mbar_a * y); Mf_s = tf.reduce_sum(Mf_a * y)
    r = time_params["poqr"]; r2 = r**2; phi = time_params["poqi"]; A_prod = time_params["A_prod"]
    fa = (1+A_prod); fabar = (1-A_prod)

    int1_x = 0.5*((fa+fabar/r2)*tf.math.real(M_s) + (fabar+fa*r2)*tf.math.real(Mbar_s))
    int2_x = 0.5*((fa-fabar/r2)*tf.math.real(M_s) + (fabar-fa*r2)*tf.math.real(Mbar_s))
    int3_x = (-(fa*r+fabar/r)*tf.cos(phi)*tf.math.real(Mf_s) + (fa*r+fabar/r)*tf.sin(phi)*tf.math.imag(Mf_s))
    int4_x = (-(fa*r-fabar/r)*tf.cos(phi)*tf.math.imag(Mf_s) - (fa*r-fabar/r)*tf.sin(phi)*tf.math.real(Mf_s))
    int_all = int1_x*icht + int2_x*ict + int3_x*isht + int4_x*ist

    # dN/dy
    def c(x): return tf.cast(x, M_a.dtype)
    dNdy = c((fa+fabar/r2)*icht/2)*M_a + c((fabar+fa*r2)*icht/2)*Mbar_a
    dNdy += c((fa-fabar/r2)*ict/2)*M_a + c((fabar-fa*r2)*ict/2)*Mbar_a
    Mf1_a = (Mf_a + Mfbar_a)/2
    Mf2_a = (Mf_a - Mfbar_a)/(2j)
    dNdy += c(-(fa*r+fabar/r)*tf.cos(phi)*isht)*Mf1_a + c((fa*r+fabar/r)*tf.sin(phi)*isht)*Mf2_a
    dNdy += c(-(fa*r-fabar/r)*tf.cos(phi)*ist)*Mf2_a + c(-(fa*r-fabar/r)*tf.sin(phi)*ist)*Mf1_a

    grad_time_params = {}
    invr2=1/r2; invr3=2/r**3
    dint1_dr = 0.5*(-fabar*invr3*tf.math.real(M_s) + 2*fa*r*tf.math.real(Mbar_s))*icht
    dint2_dr = 0.5*(fabar*invr3*tf.math.real(M_s) - 2*fa*r*tf.math.real(Mbar_s))*ict
    dint3_dr = (-(fa-fabar*invr2)*tf.cos(phi)*tf.math.real(Mf_s)+(fa-fabar*invr2)*tf.sin(phi)*tf.math.imag(Mf_s))*isht
    dint4_dr = (-(fa+fabar*invr2)*tf.cos(phi)*tf.math.imag(Mf_s)-(fa+fabar*invr2)*tf.sin(phi)*tf.math.real(Mf_s))*ist
    grad_time_params["poqr"] = dint1_dr + dint2_dr + dint3_dr + dint4_dr
    dint3_dphi = ((fa*r+fabar/r)*tf.sin(phi)*tf.math.real(Mf_s)+(fa*r+fabar/r)*tf.cos(phi)*tf.math.imag(Mf_s))*isht
    dint4_dphi = ((fa*r-fabar/r)*tf.sin(phi)*tf.math.imag(Mf_s)-(fa*r-fabar/r)*tf.cos(phi)*tf.math.real(Mf_s))*ist
    grad_time_params["poqi"] = dint3_dphi + dint4_dphi
    grad_time_params["delta_gamma"] = int1_x*idcht + int3_x*idsht
    grad_time_params["delta_m"] = int2_x*idct + int4_x*idst
    grad_time_params["gamma"] = -(int1_x*idsht*2 + int2_x*idst + int3_x*idcht*2 - int4_x*idct)
    # grad_time_params["A_prod"] not needed for NLL value
    # Skip A_prod grad definition (not needed for NLL value)

    return int_all, dNdy, grad_time_params

# ── nll_batch (matching reference) ──
def nll_batch(x, datavar, NdN):
    x_y = x[:-len(free_time_params)]
    time_params = {n: x[i+x_y.shape[0]] for i,n in enumerate(free_time_params)}
    for i in fix_time_params: time_params[i] = all_time_params[i]
    r = time_params["poqr"]; phi = time_params["poqi"]
    delta_m = time_params["delta_m"]; delta_gamma = time_params["delta_gamma"]
    gamma = time_params["gamma"]; A_prod = time_params["A_prod"]
    fa = (1+A_prod); fabar = (1-A_prod)

    with tf.device("CPU"):
        y = build_par_simple(tf.constant(x_y))
    y = tf.reshape(y, (-1,2))
    y = tf.complex(y[:,0], y[:,1])

    reN, dNdy, _ = NdN
    data, data_time_v, data_bbar_frac_v, Bi_v, sw_v = datavar
    Ai = data[:,::2]; Aibar = data[:,1::2]
    A = tf.reduce_sum(Ai*y, axis=-1); Abar = tf.reduce_sum(Aibar*y, axis=-1)
    Asq = tf.abs(A)**2; Abarsq = tf.abs(Abar)**2
    AAbar = tf.math.conj(A)*Abar

    cht = tf.math.cosh(delta_gamma*data_time_v/2)
    ct = tf.math.cos(delta_m*data_time_v)
    sht = tf.math.sinh(delta_gamma*data_time_v/2)
    st = tf.math.sin(delta_m*data_time_v)
    expt = tf.exp(-gamma*data_time_v)
    r2 = r**2

    P1 = (Asq + r2*Abarsq)*cht; P2 = (Asq - r2*Abarsq)*ct
    P3 = (tf.cos(phi)*tf.math.real(AAbar) - tf.sin(phi)*tf.math.imag(AAbar))*sht
    P4 = (tf.cos(phi)*tf.math.imag(AAbar) + tf.sin(phi)*tf.math.real(AAbar))*st
    P = P1 + P2 - 2*r*P3 - 2*r*P4
    Pbar = P1/r2 - P2/r2 - 2/r*P3 + 2/r*P4
    Psig_num = ((1-data_bbar_frac_v)*fa*P + data_bbar_frac_v*fabar*Pbar)*expt
    Psig = Psig_num/reN
    L = purity*Psig + (1-purity)*Bi_v
    lnL = -tf.reduce_sum(sw_v * tf.math.log(tf.maximum(L, 1e-300)))
    return lnL

# ── Main ──
n_batch = 2; batch_size = data.shape[0] // n_batch
batch_data = []
for batch in range(n_batch):
    sl = slice(batch*batch_size, min((batch+1)*batch_size, data.shape[0]))
    batch_data.append((data[sl], data_time[sl], data_bbar_frac[sl], Bi[sl], sw[sl]))

# Load params (with scale reversal matching ampfit values_from_dict)
with open("pw_cfit5_td6_fix29/final_params_0.json") as f: pdat = json.load(f)

def get_val(name):
    val = float(pdat["value"].get(name, 0))
    for p, s in scale_params.items():
        if name == p + 'r' and s != 0:
            val /= s
    return val

x_y = np.array([[get_val(i+"r"), get_val(i+"i")] for i in all_used_x]).reshape(-1)
tp = np.array([float(pdat["value"].get(n, float(all_time_params[n]))) for n in free_time_params])
x = np.concatenate([x_y, tp])

print(f"Params: {len(all_used_x)} free, x={len(x)} ({len(x_y)} ck + {len(free_time_params)} time)")

# Compute NdN
with tf.device("CPU"):
    y0 = build_par_simple(tf.constant(x_y))
y0 = tf.reshape(y0, (-1,2))
y0 = tf.complex(y0[:,0], y0[:,1])

time_params = {n: x[len(x_y)+i] for i,n in enumerate(free_time_params)}
for n in fix_time_params: time_params[n] = all_time_params[n]
NdN = inte_sig(y0, M, time_params)
print(f"Norm integral: {float(NdN[0]):.2f}")

# Compute NLL
lnL_total = 0.0
for batch in range(n_batch):
    lnL = nll_batch(x, batch_data[batch], NdN)
    lnL_total += lnL
    print(f"  Batch {batch}: lnL={float(lnL)*sw_scale:.2f}")

nll_final = float(lnL_total) * sw_scale
print(f"\n=== RESULTS ===")
print(f"NLL (reproduced): {nll_final:.6f}")
print(f"NLL (JSON):       {pdat['status']['NLL']:.6f}")
print(f"Diff:             {nll_final - pdat['status']['NLL']:.6f}")

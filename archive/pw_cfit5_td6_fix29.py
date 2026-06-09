import numpy as np
import json
import tensorflow as tf
import glob
from tqdm import tqdm
import cmath
import os
import yaml
from tf_pwa.amp.time_dep import cal_gp_gm
import sys

prefix=sys.argv[0][:-3]

with open("data_all_comb.json") as f:
    all_comb = json.load(f)
with open("config_amp.yml") as f:
    config_amp = yaml.full_load(f)

all_time_params = {
    "gamma": tf.convert_to_tensor(0.0, tf.float64),
    "delta_m": tf.Variable(tf.convert_to_tensor(0.506, tf.float64)),
    "delta_gamma": tf.convert_to_tensor(0.0, tf.float64),
    "poqr": tf.convert_to_tensor(1.0, tf.float64),
    "poqi": tf.convert_to_tensor(0.0, tf.float64),
    "A_prod": tf.convert_to_tensor(0.0, tf.float64),
}
fix_time_params = [ "A_prod", "delta_gamma", "delta_m", "poqr", "poqi"]
free_time_params = [i for i in all_time_params if i not in fix_time_params]

all_params = []
same_params = []
scale_params = {}
fixed_params = {}
for i in all_comb:
    for j in i:
        if j not in all_params:
            all_params.append(j)
            # if j.endswith("total_0"):
            #    fixed_params[j] = 1+0j
            if j.endswith("g_ls_0"):
                fixed_params[j] = 1+0j
            elif j.endswith("pole.0"):
                fixed_params[j] = 1+0j
            elif j.endswith("point_5"):
                fixed_params[j] = 1+0j
            elif j.endswith("fix1"):
                fixed_params[j] = 1+0j
            else:
                pass # all_params.append(j)
# all_total = [i for i in all_used_x if "total_0" in i]
fix_total = "B->rhoA.rhoBrhoA->pip1.pim1rhoB->pip2.pim2_total_0" # "B->a1(1260)p.pim2a1(1260)p->rhoA.pip2rhoA->pip1.pim1_total_0"
# fix_total = "B->rhoA.rhoB_g_ls_0"
fixed_params[fix_total] = 1+0j
# fixed_params["B->rhoA.rhoB_g_lsbar_0"] = 1+0j
# fixed_params["rhoB->pip2.pim2_g_ls_0"] = 1+0j
# del fixed_params["KMC->pip1.pim1_g_ls_0"]

# all_used_x.remove(fix_total)
# fixed_params["B->rhoA.rhoB_g_ls_1"] = 1+0j
# fixed_params["B->rhoA.rhoB_g_lsbar_1"] = 1+0j
# del fixed_params["B->rhoA.rhoB_g_ls_0"]
# fixed_params["B->a1(1260)p.pim2a1(1260)p->KM2.pip2KM2->pip1.pim1_total_0"] = 1.+0j
for r1 in ["a1(1260)", "a1(1640)","a2(1320)", "pi1300", "pi1600", "a2(1700)", "pi2(1670)", "pi1(1600)"]:
    name_ps = []
    name_ms = []
    fixed = True
    for r2 in ["rhoA", "f0(500)", "f0(980)", "f2(1270)"]:
        name_p = f"B->{r1}p.pim2{r1}p->{r2}.pip2{r2}->pip1.pim1_total_0"
        name_m = f"B->{r1}m.pip2{r1}m->{r2}.pim2{r2}->pip1.pim1_total_0"
        if name_p in all_params:
            name_ps.append(name_p)
            name_ms.append(name_m)
        for idx in range(3):
            if f"{r1}m->{r2}.pim2_g_ls_{idx}" in all_params:
                if fixed:
                    fixed = False
                    print(f"{r1}m->{r2}.pim2_g_ls_{idx}")
                else:
                    if f"{r1}m->{r2}.pim2_g_ls_{idx}" in fixed_params:
                        del fixed_params[f"{r1}m->{r2}.pim2_g_ls_{idx}"]
                    if f"{r1}p->{r2}.pip2_g_ls_{idx}" in fixed_params:
                        del fixed_params[f"{r1}p->{r2}.pip2_g_ls_{idx}"]
                    same_params.append([
                        f"{r1}m->{r2}.pim2_g_ls_{idx}",
                        f"{r1}p->{r2}.pip2_g_ls_{idx}"
                        ])
                if r2 == "rhoA":
                    scale_params[f"{r1}m->{r2}.pim2_g_ls_{idx}"] = -1
    same_params.append(name_ps)
    same_params.append(name_ms)


#fixed_params["B->rhoA.rhoB_g_lsbar_0"] = 1.0+0j
# same_params.append(["B->rhoA.rhoB_g_ls_1", "B->rhoA.rhoB_g_lsbar_1"])
# same_params.append(["B->rhoA.rhoB_g_ls_2", "B->rhoA.rhoB_g_lsbar_2"])
# same_params.append(["a1(1260)p->rhoA.pip2_g_ls_1", "a1(1260)m->rhoA.pim2_g_ls_1"])
# same_params.append(["rhoA_frac", "rhoB_frac"])



for i in range(3):
    for j in ["pole", "prod"]:
        if f"KMA_{j}.{i}" in all_params:
            same_params.append([f"KMA_{j}.{i}", f"KMB_{j}.{i}"])
for i in range(3,5):
    for j in ["pole", "prod"]:
        fixed_params[f"KMA_{j}.{i}"] = 0.0
        fixed_params[f"KMB_{j}.{i}"] = 0.0
        fixed_params[f"KMC_{j}.{i}"] = 0.0
        fixed_params[f"KM2_{j}.{i}"] = 0.0

# fixed_params["rhoA_frac"] = 0
# fixed_params["rhoB_frac"] = 0

print("B->a1(1640)m.pip2a1(1640)m->rhoA.pim2rhoA->pip1.pim1_total_0" in fixed_params)
print("B->a1(1640)m.pip2a1(1640)m->rhoA.pim2rhoA->pip1.pim1_total_0" in all_params)

# print(fixed_params)

all_used_x = [i for i in all_params if i not in fixed_params]


new_name = {}
for i in same_params:
    for j in i[1:]:
        if j in all_used_x:
            all_used_x.remove(j)
        new_name[j] = i[0]

print(all_used_x)

print(new_name)
print(fixed_params)

new_all_comb = []
for i in all_comb:
    tmp = []
    for j in i:
        if j in new_name:
            tmp.append(new_name[j])
        else:
            tmp.append(j)
        if j in scale_params:
            tmp.append(scale_params[j])
    new_all_comb.append(tmp)
all_comb = new_all_comb



#@tf.function
def build_par(x):
    x = tf.reshape(x,(-1,2))
    # print(x, x[:,0]*tf.cos(x[:,1]), x[:,0]*tf.sin(x[:,1]))
    xc = tf.complex(x[:,0]*tf.cos(x[:,1]), x[:,0]*tf.sin(x[:,1]))
    list_x = tf.unstack(xc)
    new_params = {}
    for i, name in enumerate(all_used_x):
        new_params[name] = xc[i]
    ret = []
    for i in all_comb:
        tmp = 1.0
        for j in i:
            if not isinstance(j, str):
                tmp = tmp * j
            elif j not in fixed_params:
                tmp = tmp * new_params[j]
            else:
                tmp = tmp * fixed_params[j]
        ret.append(tmp)
    ret = tf.stack(ret)
    ret = tf.stack([tf.math.real(ret),
    tf.math.imag(ret)], axis=-1)
    return tf.reshape(ret, (-1,))


fix_params_value_vec = np.array(list(fixed_params.values()))
fix_params_name_vec = list(fixed_params.keys())

comb_matrix = []
max_comb_depth = max([len(i) for i in all_comb])
for i in all_comb:
    for j in range(max_comb_depth):
        if len(i) <= j:
            comb_matrix.append(0)
        else:
            name = i[j]
            if not isinstance(name, str):
                pass
            elif name in all_used_x:
                comb_matrix.append(1+all_used_x.index(name))
            else:
                comb_matrix.append(1+len(all_used_x)+fix_params_name_vec.index(name))
comb_matrix = np.array(comb_matrix)


def build_par_prod(x):
    x = tf.reshape(x,(-1,2))
    # print(x, x[:,0]*tf.cos(x[:,1]), x[:,0]*tf.sin(x[:,1]))
    xc = tf.complex(x[:,0]*tf.cos(x[:,1]), x[:,0]*tf.sin(x[:,1]))

    all_xc = tf.concat([np.array([1+0j]), xc, fix_params_value_vec], axis=-1)
    comb_matrix_value = tf.gather(all_xc, comb_matrix)
    comb_matrix_value = tf.reshape(comb_matrix_value, (len(all_comb), max_comb_depth))
    ret = tf.reduce_prod(comb_matrix_value, axis=-1)
    ret = tf.stack(ret)
    ret = tf.stack([tf.math.real(ret),
    tf.math.imag(ret)], axis=-1)
    return tf.reshape(ret, (-1,))


@tf.function
def build_par_jac_ana(x):

    """all_used_x -> pw"""
    x = tf.unstack(x)
    params_map = {all_used_x[i]: x[i] for i in range(len(all_used_x))}
    new_pw = []
    jac = []
    zero = tf.zeros_like(x[0])
    one = tf.ones_like(x[0])
    for i in all_comb:
        not_fix = [j for j in i if j not in fixed_params]
        fix_i = [j for j in i if j in fixed_params]
        scale = 1.0+0j
        if len(fix_i) != 0:
            for i in fix_i:
                scale = scale * fixed_params[i]
        mul_all = [params_map[j] for j in not_fix]
        if len(mul_all) == 0:
            new_pw.append(scale*one)
        else:
            new_pw.append(scale*tf.reduce_prod(mul_all))
        tmp = []
        for j in all_used_x:
            if j in not_fix:
                order = len([k for k in not_fix if k == j])
                mul = [params_map[k] for k in not_fix if k !=j]
                if len(mul) == 0:
                    frac = one
                else:
                    frac = tf.reduce_prod(mul)
                if order == 1:
                    tmp.append( scale*frac )
                else:
                    tmp.append( scale*frac * order * params_map[j]**(order-1))
            else:
                tmp.append( zero )
        jac.append(tf.stack(tmp))
    # print(tf.stack(jac))
    return tf.stack(new_pw), tf.stack(jac)


@tf.function
def build_par_jac(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = build_par(x) # build_par_prod(x) # y = build_par(x)
    jac = tape.jacobian(y, x)
    return y, jac





data = []
for i in sorted(glob.glob("pw_amp/pw_*/data_all_amp.npy"), key=lambda x: int(x.split("/")[-2].split("_")[-1])):
     data.append(np.load(i))
data = np.concatenate(data, axis=-1)

sw = np.ones(data.shape[0]) # np.load("data_sigbg_w.npy")[:]
Bi = np.load(config_amp["data"]["data_bg_value"][0]).flatten().astype(np.float64)
NB = np.mean(np.load(config_amp["data"]["phsp_bg_value"][0]).astype(np.float64))
Bi = tf.convert_to_tensor(Bi/NB)

data_time = np.load(config_amp["data"]["data_time"][0])
data_eta = np.load(config_amp["data"]["data_com_eta1"][0])
data_tag = np.load(config_amp["data"]["data_com_tag1"][0])

phsp_time = np.load(config_amp["data"]["phsp_time"][0])
phsp_eta = np.load(config_amp["data"]["phsp_eta1"][0])
phsp_tag = np.load(config_amp["data"]["phsp_tag1"][0])
phsp_time_weight = np.load(config_amp["data"]["phsp_weight"][0]).astype(np.float64)
phsp_time_weight = phsp_time_weight / np.sum(phsp_time_weight)
phsp_time_weight = tf.convert_to_tensor(phsp_time_weight)

phsp_l0_weight = np.load(config_amp["data"]["phsp_l0_weight"][0]).astype(np.float64)
phsp_l0_weight = phsp_l0_weight / np.sum(phsp_l0_weight)
phsp_l0_weight = tf.convert_to_tensor(phsp_l0_weight)

purity = config_amp["data"]["bg_frac"]

sw_scale = np.sum(sw)/np.sum(sw**2)

update_M2 = False

if os.path.exists("phsp_pw_M2_td3.npy"):
    for i in glob.glob("pw_amp/pw_*/phsp_all_amp_*.npy"):
        if os.path.getmtime(i) > os.path.getmtime("phsp_pw_M2_td3.npy"):
            update_M2 = True

if update_M2 or not os.path.exists("phsp_pw_M2_td3.npy"):
    M = 0
    Mbar = 0
    Mcov = 0
    N_count = 0
    N_split = 100
    all_files = list(glob.glob("pw_amp/pw_0/phsp_all_amp_*.npy"))
    all_files = sorted(all_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    print(all_files)
    total_idx = 0
    for i in tqdm(glob.glob("pw_amp/pw_0/phsp_all_amp_*.npy")):
        total_tensor = []
        for j in sorted(glob.glob(i.replace("pw_0", "pw_*")), key=lambda x: int(x.split("/")[-2].split("_")[-1])):
            total_tensor.append(np.load(j))
        total_tensor = np.concatenate(total_tensor, axis=-1)
        file_size = total_tensor.shape[0]
        split_size = (file_size+N_split-1)//N_split
        phsp_wall = phsp_l0_weight[total_idx:total_idx+file_size]
        total_idx += file_size

        for j in range(N_split):
            split_slice = slice(split_size*j,min(split_size*(j+1), file_size))
            tmp_x_all = tf.convert_to_tensor(total_tensor[split_slice], dtype=total_tensor.dtype)
            tmp_x = tf.reshape(tmp_x_all, (tmp_x_all.shape[0], -1,2))
            phsp_w = phsp_wall[split_slice][:,None,None]
            N_count += tf.reduce_sum(phsp_w) # tmp_x.shape[0]
            A = tmp_x[:,:,0]
            Abar  = tmp_x[:,:,1]
            temp_M1 = tf.reduce_sum(A[:,:,None] * tf.math.conj(A[:,None,:]) * tf.cast(phsp_w, A.dtype), axis=0)
            temp_M2 = tf.reduce_sum(Abar[:,:,None] * tf.math.conj(Abar[:,None,:] * tf.cast(phsp_w, A.dtype)), axis=0)
            temp_M3 = tf.reduce_sum(Abar[:,:,None] * tf.math.conj(A[:,None,:]) * tf.cast(phsp_w, A.dtype), axis=0)
            M = M + tf.stack([temp_M1, temp_M2, temp_M3], axis=-1).numpy()
    M = M / tf.cast(N_count, M.dtype)

    np.save("phsp_pw_M2_td3.npy", M)
    # M = np.sum(M, axis=-1)
else:
    M = np.load("phsp_pw_M2_td3.npy")
    # M = np.sum(M, axis=-1)

nsig = np.sum(sw)

print(M.shape, M)
# exit()

# data_gp, data_gm = cal_gp_gm(data_time, time_params["gamma"], time_params["delta_m"],  time_params["delta_gamma"])
data_bbar_frac = np.where(data_tag>0, data_eta, 1- data_eta)


def inte_sig(y, Mall, time_params):
    """
    N = \int\int 1/2[ P(t) + Pbar(t)] dx dt
      = 1/2 { \int [ (1+1/r^2)|A|^2 + (1+r^2)|Abar|^2    ] dx\int cosh(dG t/2) dt
              + \int [ (1-1/r^2)|A|^2 + (1-r^2)|Abar|^2    ] dx\int cos(dmt) dt
              + \int [ -2(r+1/r) cosphi   Re[A*Abar] + 2(r+1/r) sin phi Im[A*Abar] ] dx\int sinh(dG t/2) dt
              + \int [ -2(r-1/r) cosphi   Im[A*Abar] - 2(r-1/r) sin phi Re[A*Abar] ] dx\int sin(dm t) dt}
      = 1/2 { (1+1/r^2) cMc* + (1+r^2) cbar Mbar cbar* \int cosh(dG t/2) dt
              + (1-1/r^2)cMc* + (1-r^2) cbar Mbar cbar*    ] dx\int cos(dmt) dt
              -2(r+1/r) cosphi Re(cbar T c*)  + 2(r+1/r) sin phi Im(cbar T c*)\int sinh(dG t/2) dt
              -2(r-1/r) cosphi Im(cbar T c*) - 2(r-1/r) sin phi Re(cbar T c*)\int sin(dm t) dt}

    d Re(yMy*)/dy = 1/2 d (yMy* + y*M*y)/dy = 1/2 (My* + y*M*)/dy
    d Im(yMy*)/dy = 1/2i d (yMy* - y*M*y)/dy = 1/2i (My* - y*M*)/dy


    """

    cht = tf.math.cosh(phsp_time * time_params["delta_gamma"] / 2)
    ct = tf.math.cos(phsp_time * time_params["delta_m"])
    sht = tf.math.sinh(phsp_time * time_params["delta_gamma"] / 2)
    st = tf.math.sin(phsp_time * time_params["delta_m"])
    expt = tf.math.exp(-phsp_time * time_params["gamma"])

    icht = tf.reduce_sum( phsp_time_weight * cht *expt )
    ict = tf.reduce_sum( phsp_time_weight * ct *expt )
    isht = tf.reduce_sum( phsp_time_weight * sht *expt )
    ist = tf.reduce_sum( phsp_time_weight * st *expt )
    idcht = tf.reduce_sum( phsp_time_weight * phsp_time *expt * sht )/2
    idct = -tf.reduce_sum( phsp_time_weight * phsp_time *expt* st )
    idsht = tf.reduce_sum( phsp_time_weight * phsp_time *expt* cht )/2
    idst = tf.reduce_sum( phsp_time_weight * phsp_time *expt* ct )

    M = Mall[:,:,0]
    Mbar = Mall[:,:,1]
    Mf = Mall[:,:,2]

    M_a = tf.reduce_sum( M * tf.math.conj(y), axis=-1)
    Mbar_a = tf.reduce_sum( Mbar * tf.math.conj(y) , axis=-1)
    Mf_a = tf.reduce_sum( Mf * tf.math.conj(y), axis=-1 )
    Mfbar_a = tf.reduce_sum( tf.math.conj(Mf) * tf.math.conj(y)[:,None], axis=0)
    M_s = tf.reduce_sum(M_a * y)
    Mbar_s = tf.reduce_sum(Mbar_a * y)
    Mf_s = tf.reduce_sum(Mf_a * y)
    r = time_params["poqr"]
    r2 = r**2
    phi = time_params["poqi"]
    A_prod = time_params["A_prod"]

    fa = (1+A_prod)
    fabar = (1-A_prod)

    int1_x = 0.5*((fa+fabar/r2)* tf.math.real(M_s) + (fabar+fa*r2) * tf.math.real(Mbar_s))
    int1 = int1_x * icht
    int2_x = 0.5*((fa-fabar/r2)* tf.math.real(M_s) + (fabar-fa*r2) * tf.math.real(Mbar_s))
    int2 = int2_x * ict

    int3_x = (-(fa*r+fabar/r) *tf.cos(phi) * tf.math.real(Mf_s)  + (fa*r+fabar/r)* tf.sin(phi)* tf.math.imag(Mf_s) )
    int3 = int3_x * isht

    int4_x = (-(fa*r-fabar/r) * tf.cos(phi) * tf.math.imag(Mf_s)  - (fa*r-fabar/r)* tf.sin(phi)* tf.math.real(Mf_s) )
    int4 = int4_x * ist

    int_all = (int1 + int2 + int3 + int4)
    dint1_dy = tf.cast( (fa+fabar/r2)* icht / 2, M_a.dtype) * M_a + tf.cast( (fabar+fa*r2)* icht / 2, Mbar_a.dtype) * Mbar_a
    dint2_dy = tf.cast( (fa-fabar/r2)* ict / 2, M_a.dtype) * M_a + tf.cast( (fabar-fa*r2)* ict / 2, Mbar_a.dtype) * Mbar_a
    Mf1_a = (Mf_a + Mfbar_a)/2
    Mf2_a = (Mf_a - Mfbar_a)/2j
    dint3_dy = tf.cast( -(fa*r+fabar/r)*tf.cos(phi)* isht, Mf1_a.dtype) * Mf1_a + tf.cast( (fa*r+fabar/r)*tf.sin(phi)* isht, Mf2_a.dtype) * Mf2_a
    dint4_dy = tf.cast( -(fa*r-fabar/r)*tf.cos(phi)* ist, Mf2_a.dtype) * Mf2_a + tf.cast( -(fa*r-fabar/r)*tf.sin(phi)* ist, Mf1_a.dtype) * Mf1_a
    dint_dy = dint1_dy + dint2_dy + dint3_dy + dint4_dy

    grad_time_params = {}
    invr2 = 1/r**2
    invr3 = 2/r**3
    dint1_dr = 0.5*(-fabar*invr3 * tf.math.real(M_s) + 2*fa*r*tf.math.real(Mbar_s)) * icht
    dint2_dr = 0.5*(fabar*invr3* tf.math.real(M_s) - 2*fa*r*tf.math.real(Mbar_s)) * ict
    dint3_dr = (-(fa-fabar*invr2) *tf.cos(phi) * tf.math.real(Mf_s)  + (fa-fabar*invr2)* tf.sin(phi)* tf.math.imag(Mf_s) ) * isht
    dint4_dr = (-(fa+fabar*invr2) * tf.cos(phi) * tf.math.imag(Mf_s)  - (fa+fabar*invr2)* tf.sin(phi)* tf.math.real(Mf_s) ) * ist

    grad_time_params["poqr"] = dint1_dr + dint2_dr + dint3_dr + dint4_dr

    dint3_dphi = ((fa*r+fabar/r) *tf.sin(phi) * tf.math.real(Mf_s)  + (fa*r+fabar/r)* tf.cos(phi)* tf.math.imag(Mf_s) ) * isht
    dint4_dphi = ((fa*r-fabar/r) * tf.sin(phi) * tf.math.imag(Mf_s)  - (fa*r-fabar/r)* tf.cos(phi)* tf.math.real(Mf_s) ) * ist


    dint1_dap = 0.5*((1-1/r2)* tf.math.real(M_s) + (-1+r2) * tf.math.real(Mbar_s)) * icht
    dint2_dap = 0.5*((1+1/r2)* tf.math.real(M_s) + (-1-r2) * tf.math.real(Mbar_s))* ict
    dint3_dap = (-(r-1/r) *tf.cos(phi) * tf.math.real(Mf_s)  + (r-1/r)* tf.sin(phi)* tf.math.imag(Mf_s) ) * isht
    dint4_dap = (-(r+1/r) * tf.cos(phi) * tf.math.imag(Mf_s)  - (r+1/r)* tf.sin(phi)* tf.math.real(Mf_s) ) * ist

    grad_time_params["poqi"] = dint3_dphi + dint4_dphi
    grad_time_params["delta_gamma"] = int1_x * idcht + int3_x * idsht
    grad_time_params["delta_m"] = int2_x * idct + int4_x * idst
    grad_time_params["gamma"] =  -(int1_x * idsht*2 + int2_x * idst + int3_x * idcht*2 - int4_x * idct)
    grad_time_params["A_prod"] = dint1_dap + dint2_dap + dint3_dap + dint4_dap

    return int_all, dint_dy, grad_time_params


@tf.function
def nll_batch(x, datavar, NdN):
    """

    P(t) =   (|A|^2 + |r exp(iphi) Abar|^2)cosh(dG t/2)
           + (|A|^2 - |r exp(iphi) Abar|^2)cos(dm t)
           - 2r Re[exp(iphi) A* Abar] sinh (dG t/2)
           - 2r Im[exp(iphi) A* Abar] sinh (dG t/2)
         =   (|A|^2 + r |Abar|^2)cosh(dG t/2)
           + (|A|^2 - r |Abar|^2)cos(dm t)
           - 2r [cosphi Re[A* Abar] - sin phi Im[A* Abar]] sinh (dG t/2)
           - 2r [cosphi Im[A* Abar] + sin phi Re[A* Abar]] sin (dmt )


    Pbar(t) =  (|Abar|^2 + |1/r exp(-iphi) A|^2)cosh(dG t/2)
           + (|Abar|^2 - |1/r exp(-iphi) A|^2)cos(dm t)
           - 2Re[1/r exp(-iphi) Abar* A] sinh (dG t/2)
           - 2Im[1/r exp(-iphi) Abar* A] sin (dmt)

            =  (|Abar|^2 + |1/r exp(-iphi) A|^2)cosh(dG t/2)
           + (|Abar|^2 - |1/r exp(-iphi) A|^2)cos(dm t)
           - 2/r Re[exp(iphi) A*Abar] sinh (dG t/2)
           + 2/r Im[exp(iphi)* A*Abar] sin (dmt)
            =  (|Abar|^2 + 1/r|A|^2)cosh(dG t/2)
           + (|Abar|^2 - 1/r|A|^2)cos(dm t)
           - 2/r [ cosphi Re[A*Abar] - sin phi Im[A*Abar]] sinh (dG t/2)
           + 2/r [ cosphi Im[A*Abar] + sinphi Re[A*Abar]] sin (dmt)



    """

    x_y = x[:-len(free_time_params)]
    time_params = {name: x[i+x_y.shape[0]] for i, name in enumerate(free_time_params)}
    for i in fix_time_params:
        time_params[i] = all_time_params[i]
    r = time_params["poqr"]
    phi = time_params["poqi"]
    delta_m = time_params["delta_m"]
    delta_gamma = time_params["delta_gamma"]
    gamma = time_params["gamma"]
    A_prod = time_params["A_prod"]

    fa = (1+A_prod)
    fabar = (1-A_prod)

    with tf.device("CPU"):
        y, jac = build_par_jac(x_y)

    y = tf.reshape(y, (-1,2))

    y = tf.complex(y[:,0],y[:,1])

    reN, dNdy, dNdtime = NdN # inte_sig(y, M, time_params)


    data, data_time, data_bbar_frac, Bi, sw = datavar
    Ai = data[:,::2]
    Aibar = data[:,1::2]

    A = tf.reduce_sum(Ai*y, axis=-1)
    Abar = tf.reduce_sum(Aibar*y, axis=-1)

    Asq = tf.abs(A)**2
    Abarsq = tf.abs(Abar)**2
    AAbar = tf.math.conj(A)*Abar

    GA = Ai * tf.math.conj(A)[:,None]
    GAbar = Aibar * tf.math.conj(Abar)[:,None]
    GAAbar = Aibar * tf.math.conj(A)[:,None]
    GAbarA = Ai * tf.math.conj(Abar)[:,None]

    cht = tf.math.cosh(time_params["delta_gamma"]*data_time / 2)
    ct = tf.math.cos(time_params["delta_m"]*data_time)
    sht = tf.math.sinh(time_params["delta_gamma"]*data_time / 2)
    st = tf.math.sin(time_params["delta_m"]*data_time)
    expt = tf.exp(-time_params["gamma"]*data_time)

    r2 = r**2
    invr2 = 1/r2
    r4 = r2**2

    P1 = (Asq + r2 * Abarsq) * cht
    P2 = (Asq - r2 * Abarsq) * ct

    P3 = (tf.cos(phi)* tf.math.real(AAbar) - tf.math.sin(phi)* tf.math.imag(AAbar)) * sht
    P4 = (tf.math.cos(phi)*tf.math.imag(AAbar) + tf.math.sin(phi) * tf.math.real(AAbar)) * st
    P = P1  + P2  - 2 * r * P3 - 2*r*P4

    Pbar = P1/r2 - P2/r2 - 2 / r *P3 + 2/r*P4

    Psig_num = ((1-data_bbar_frac)*fa*P+data_bbar_frac*fabar*Pbar )* expt
    Psig = Psig_num/reN

    L = (purity) * Psig + (1-purity) * Bi

    # cut = P < 1e-7
    # P = tf.where(cut, 1e-7*tf.ones_like(P), P)
    lnL = - tf.reduce_sum(sw * tf.math.log(L))

    C = lambda x: tf.cast(x, GA.dtype)

    G1 = (GA + C(r2)* GAbar)*C(cht[:,None])
    G2 = (GA - C(r2) * GAbar)*C(ct[:,None])
    GrAAbar = (GAAbar + GAbarA)/2
    GiAAbar = (GAAbar - GAbarA)/2j
    G3 = (C(tf.cos(phi))* GrAAbar - C(tf.math.sin(phi))* GiAAbar)*C(sht[:,None])
    G4 = (C(tf.cos(phi))* GiAAbar + C(tf.math.sin(phi))* GrAAbar)*C(st[:,None])

    Gb = G1 + G2 - C(2 * r)*G3 - C(2 * r) * G4
    Gbbar = G1/C(r2) - G2/C(r2) - C(2 / r)*G3 + C(2 / r) * G4

    # dPsig/dy
    G_num = (C((1-data_bbar_frac) * fa)[:,None]*Gb + C(data_bbar_frac * fabar)[:,None] * Gbbar)*C(expt[:,None])

    G = G_num/C(reN) - C(Psig_num/reN**2)[:,None] * dNdy
    ret_g = -tf.reduce_sum(purity * tf.complex(sw/L,tf.zeros_like(L))[:,None] * G, axis=0)
    ret_x = 2 * tf.math.real(ret_g)
    ret_y = -2 * tf.math.imag(ret_g)
    g = tf.reshape(tf.stack([ret_x, ret_y], axis=-1), (-1,))
    # print(g.shape, jac.shape)
    ret_g = tf.reduce_sum(g[:,None] * jac, axis=0)

    Gb_r = 2*r*(Abarsq * cht - Abarsq * ct) - 2*(P3 + P4)
    Gbbar_r = -2/r**3 *(Asq * cht - Asq * ct) +  2/r**2 * ( P3 - P4)

    G3_phi = (-tf.sin(phi)* tf.math.real(AAbar) - tf.math.cos(phi)* tf.math.imag(AAbar)) * sht
    G4_phi = (-tf.math.sin(phi)*tf.math.imag(AAbar) + tf.math.cos(phi) * tf.math.real(AAbar)) * st
    Gb_phi = - 2 * r * G3_phi - 2*r*G4_phi
    Gbbar_phi = - 2 / r *G3_phi + 2/r*G4_phi

    G1_dg = (Asq + r2 * Abarsq) * sht *data_time/2
    G2_dm = (Asq - r2 * Abarsq) * -st * data_time
    G3_dg = (tf.cos(phi)* tf.math.real(AAbar) - tf.math.sin(phi)* tf.math.imag(AAbar)) * cht * data_time/2
    G4_dm = (tf.math.cos(phi)*tf.math.imag(AAbar) + tf.math.sin(phi) * tf.math.real(AAbar)) * ct * data_time
    Gb_dg = G1_dg  - 2 * r * G3_dg
    Gbbar_dg = G1_dg/r2 - 2 / r *G3_dg
    Gb_dm = G2_dm - 2*r*G4_dm
    Gbbar_dm = -G2_dm /r2 + 2/r*G4_dm
    Gb_dgamma = - data_time * P
    Gbbar_dgamma = - data_time * Pbar
    Gb_dap = P
    Gbbar_dap = - Pbar

    Gb_time_dic = {"poqr": fa*Gb_r, "poqi": fa*Gb_phi, "delta_m": fa*Gb_dm, "delta_gamma": fa*Gb_dg, "gamma": fa*Gb_dgamma, "A_prod": Gb_dap}
    Gb_time = tf.stack([Gb_time_dic[i] for i in free_time_params], axis=-1) * expt[:,None]

    Gbbar_time_dic = {"poqr": fabar*Gbbar_r, "poqi": fabar*Gbbar_phi, "delta_m": fabar*Gbbar_dm, "delta_gamma": fabar*Gbbar_dg, "gamma": fabar*Gbbar_dgamma, "A_prod": Gbbar_dap}
    Gbbar_time = tf.stack([Gbbar_time_dic[i] for i in free_time_params], axis=-1)  * expt[:,None]

    dn_dtime = tf.stack([dNdtime[i] for i in free_time_params])
    Gnum_time = (1-data_bbar_frac)[:,None]*Gb_time + data_bbar_frac[:,None]*Gbbar_time
    Gsig_time = Gnum_time/reN - (Psig_num/reN**2)[:,None] * dn_dtime

    dret_time = - tf.reduce_sum((purity *sw/L)[:,None] * Gsig_time, axis=0)

   #  lnL = tf.reduce_sum(Pbar, axis=0)
   # dret_time = tf.reduce_sum(Gbbar_time, axis=0)

    ret_g = tf.concat([ret_g, dret_time], axis=0)


    return lnL, ret_g


n_batch = 2
batch_size = data.shape[0]//n_batch
batch_data = []
for batch in range(n_batch):
    cur_slice = slice(batch * batch_size, min((batch+1)*batch_size, data.shape[0]))
    data_i = data[cur_slice]
    data_time_i = data_time[cur_slice]
    bbfrac = data_bbar_frac[cur_slice]
    Bi_i = Bi[cur_slice]
    sw_i = sw[cur_slice]
    batch_data.append((data_i, data_time_i, bbfrac, Bi_i, sw_i))


@tf.function
def nll(x):

    x_y = x[:-len(free_time_params)]
    time_params = {name: x[i+x_y.shape[0]] for i, name in enumerate(free_time_params)}
    for i in fix_time_params:
        time_params[i] = all_time_params[i]
    with tf.device("CPU"):
        y, jac = build_par_jac(x_y)
    y = tf.reshape(y, (-1,2))
    y = tf.complex(y[:,0],y[:,1])
    NdN = inte_sig(y, M, time_params)

    lnL = 0.
    ret_g = 0.
    for batch in range(n_batch):
        tmp_lnL, tmp_ret_g = nll_batch(x, batch_data[batch], NdN)
        lnL = lnL + tmp_lnL
        ret_g = ret_g + tmp_ret_g
    return lnL, ret_g



combine_idx = {}
for idx, i in enumerate(all_comb):
    total = tuple(j for j in i if "total" in str(j) or "B->rhoA.rhoB" in str(j))
    combine_idx[total] = combine_idx.get(total, []) + [idx]


@tf.function
def get_fraction(x):

    with tf.device("CPU"):
        y, jac = build_par_jac(x)

    y = tf.reshape(y, (2,-1,2))
    yp = tf.complex(y[0,:,0],y[0,:,1])
    ym = tf.complex(y[1,:,0],y[1,:,1])

    # print(M.shape, yp.shape)
    N_a1 = M[:,:,0] * tf.math.conj(yp)/2
    N_a2 = M[:,:,1] * tf.math.conj(ym)/2

    N1 = N_a1 * yp[:,None]
    N2 = N_a2 * ym[:,None]
    N = N1 + N2
    frac = tf.math.real(N)
    size = frac.shape[0]

    # dN/dyp = dN1/dyp = N_a1 # (N, N)
    # dN/dym = dN1/dyp = N_a2 # (N, N)

    # dN/dy = dN/dyp dyp/dy + dN/dym dym/dy
    #       = N_a1 dyp/dy + N_a2 dym/dy
    #       = (N, N) (N, 2N) + (N, N) (N, 2N)
    #       = (N, 2N, N) + (N, 2N, N)
    #       = (N, 2N, N)
    # dN1/dy[i,j,k] = N_a1[i,j]dypdy[i,k]
    # dN/dy[i,j,k] = N_a1[i,j]dypdy[i,k] + N_a2[i,j]dymdy[i,k]

    ret_g = tf.concat([N_a1, N_a2], axis=0) # (2N, N)
    ret_x = 2 * tf.math.real(ret_g)
    ret_y = -2 * tf.math.imag(ret_g)

    g = tf.stack([ret_x, ret_y], axis=1) # (2N, 2, N)
    g = tf.reshape(g, (-1, size))
    g = g[:,:,None] * jac[:,None, :]
    g = tf.reshape(g, (2, size, 2, size, -1))
    g = tf.reduce_sum(g, axis=0)
    g = tf.reduce_sum(g, axis=1)
    return frac, tf.reshape(g, (size, size, -1))

def fraction(x, V):
    frac, g = get_fraction(x)
    frac, g = frac.numpy(), g.numpy()
    sum_g = np.sum(np.sum(g, axis=0), axis=0)
    scale = np.sum(frac)
    ret = {}
    for k, v in combine_idx.items():
        v = np.array(v)
        part = np.sum(frac[v][:,v])
        part_g = np.sum(np.sum(g[v][:,v], axis=0), axis=0)
        jac = part_g/scale-part/scale**2*sum_g
        print(k, part/scale, np.sqrt(np.dot(np.dot(V, jac), jac)))
        ret[str(k)] = [part/scale, np.sqrt(np.dot(np.dot(V, jac), jac))]
    return ret


class Trans:
    def __init__(self, a, b):
        self.a = min(a, b)
        self.b = max(a, b)
        self.k = (self.b - self.a)/2
        self.bias = (self.b + self.a)/2
    def __call__(self, x):
        y = self.k * np.sin( x / self.k) + self.bias
        return y
    def grad(self, x):
        return np.cos( x / self.k )
    def inv(self, y):
        y = (y - self.a) % (self.b - self.a) + self.a
        x = np.arcsin( ((y - self.bias)/self.k + 1) % 2 - 1 ) * self.k
        return x
    def trans_err(self, x, e):
        g = self.grad(x)
        return np.abs(g) * e


time_params_bound = {
    "delta_m": [0.3, 0.8],
    "delta_gamma": [-0.3, 0.3],
    "A_prod": [-0.5, 0.5],
    "gamma": [-0.3, 0.3]
}

bound_trans = {}

for k, v in time_params_bound.items():
    if k in free_time_params:
        idx = len(all_used_x) * 2 + free_time_params.index(k)
        bound_trans[idx] = Trans(v[0], v[1])
# bound_trans = {}

params_order = {}
for i,xi in enumerate(all_used_x):
    params_order[2*i] = xi+"r"
    params_order[2*i+1] = xi +"i"
for i, xi in enumerate(free_time_params):
    params_order[i + len(all_used_x)*2] = xi


grad_sacle = 1e-3
def nll_numpy(x):
    # x2 = np.insert(x, all_used_x.index(fix_total)*2+1,0)
    # print(x2.shape, x.shape)
    import time
    now = time.time()
    new_x = x.copy()
    for k, v in bound_trans.items():
        new_x[k] = v(x[k])
    lnL, g = nll(new_x)
    lnL = float(lnL)*grad_sacle
    g = g.numpy()*grad_sacle
    new_g = g.copy()
    for k, v in bound_trans.items():
        new_g[k] = v.grad(x[k]) * g[k]
        # print(k, new_g[k], g[k], v.grad(x[k]))
    # print(lnL, new_x[-4:], time.time()-now)
    # g2 = np.delete(g, all_used_x.index(fix_total)*2+1,0)
    return lnL*sw_scale, new_g*sw_scale # float(lnL)*grad_sacle, g.numpy()*grad_sacle

x = np.random.random(len(all_used_x)*2+len(free_time_params))

ret_y, ref_g = nll_numpy(x)
for i in range(x.shape[0]):
    x[i] += 1e-5
    ret_y1, ref_g1 = nll_numpy(x)
    x[i] -= 2e-5
    ret_y2, ref_g2 = nll_numpy(x)
    x[i] += 2e-5
    print(params_order[i], i, (ret_y1 - ret_y2)/2e-5, ref_g[i])

# exit()
# exit()
from scipy.optimize import minimize
all_nll = []
all_success = []
os.makedirs(prefix, exist_ok=True)
best_nll = None
best_count = 0
for idx in tqdm(range(50)):
    x = np.random.random(len(all_used_x)*2)*2*np.pi
    x = np.concatenate([x, [float(all_time_params[i]) for i in free_time_params]])
    for k, v in bound_trans.items():
        x[k] = v.inv(x[k])
    # bnds = [(None,None) for _ in range(x.shape[0])]
    # if "delta_m" in free_time_params:
    #     idx_d = free_time_params.index("delta_m") + len(all_used_x) * 2
    #     bnds[idx_d] = (0.3, 0.8)
    ret = minimize(nll_numpy, jac=True, x0=x) # , bounds=bnds)
    print(ret)
    final_params = {}
    final_params["value"] = {v: ret.x[k] for k, v in params_order.items()}
    for k, v in bound_trans.items():
        name = params_order[k]
        final_params["value"][name] = v(final_params["value"][name])
    for j in fixed_params:
        final_params["value"][j + "r"]  =  abs(fixed_params[j])
        final_params["value"][j + "i"]  =  cmath.phase(fixed_params[j])
    for j, jv in new_name.items():
        final_params["value"][j + "r"]  =  final_params["value"][jv + "r"]
        final_params["value"][j + "i"]  =  final_params["value"][jv + "i"]
    for k in fix_time_params:
        final_params["value"][k] = float(all_time_params[k])
    for j, jv in scale_params.items():
        final_params["value"][j + "r"] = jv*final_params["value"][j + "r"]
    final_params["error"] = {v: np.sqrt(ret.hess_inv[k,k]*grad_sacle) for k, v in params_order.items()}
    for k, v in bound_trans.items():
        name = params_order[k]
        x = final_params["value"][name]
        final_params["error"][name] = v.trans_err(v.inv(x), final_params["error"][name])

    final_params["status"]  = {}
    final_params["status"]["NLL"] = ret.fun/grad_sacle
    if best_nll is None or final_params["status"]["NLL"] < best_nll - 0.1:
        best_nll = final_params["status"]["NLL"]
        best_count = 1
    elif abs(final_params["status"]["NLL"] - best_nll) < 0.1:
        best_count += 1
    final_params["status"]["Ndf"] = len(ret.x)
    final_params["status"]["jac"] = list(ret.jac)
    final_params["status"]["success"] = ret.success
    final_params["status"]["message"] = ret.message
    # final_params["status"]["frac"] = fraction(ret.x, ret.hess_inv*grad_sacle)

    corr_name = [[i, n] for i, n in params_order.items() if "B->rhoA.rhoB" in n]
    corr_idx = np.array([i[0] for i in corr_name])
    corr_order = [i[1] for i in corr_name]
    corr_mat = ret.hess_inv[corr_idx][:, corr_idx] *grad_sacle
    final_params["status"]["rhorho"] = [corr_order, corr_mat.tolist()]

    with open(prefix+f"/final_params_{idx}.json", "w") as f:
        json.dump(final_params, f, indent=2)
    all_nll.append(ret.fun/grad_sacle)
    all_success.append(1 if ret.success else 0)

    import matplotlib.pyplot as plt
    plt.clf()
    plt.scatter(range(len(all_nll)), all_nll, c=all_success)
    plt.ylim((min(all_nll)-0.5, min(all_nll)+1000))
    plt.savefig(prefix+"_nll.png")

    if idx > 10 and best_count > 2:
         break




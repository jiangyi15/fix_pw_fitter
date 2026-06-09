"""
Detailed comparison of intermediate CUDA vs NumPy arrays.
"""

import numpy as np
from config_loader import Config
from numpy_kernel import NumpyKernelCorrect


def compare_step_by_step():
    """Compare each computation step"""
    print("="*70)
    print("DETAILED STEP-BY-STEP COMPARISON")
    print("="*70)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    # Single event for simplicity
    n_events = 1
    np.random.seed(42)
    
    data = {
        "mass": np.random.random((n_events, 2*3*8)),
        "q": np.random.random((n_events, 3*3*8)),
        "angle": np.random.random((n_events, 3*8, 3)),
        "frac": np.random.random((n_events,)),
        "time": np.random.random((n_events,)),
        "bkg": np.random.random((n_events,)) * 0.01,
        "weight": np.ones((n_events,)),
    }
    
    ck_map = config.get_ck_map()
    params = {
        "ck": np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
        "m0": np.random.random(len(config.m0_phys_name)) + 2,
        "g0": np.random.random(len(config.g0_phys_name)) + 0.1,
        "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }
    
    # Extract parameters
    ck = params["ck"]
    m0 = params["m0"]
    g0 = params["g0"]
    Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi = params["scalar"]
    
    # Extract data
    mass = data["mass"]
    momentum = data["q"]
    angle = data["angle"]
    frac = data["frac"]
    time = data["time"]
    weight = data["weight"]
    bkg = data.get("bkg", 0.0)
    
    print("\n" + "="*70)
    print("STEP 1: g_interp computation")
    print("="*70)
    
    # NumPy computation
    def interp(table, types, x, xmin, xdelta):
        diff = (x - xmin) / xdelta
        xbin = np.floor(diff).astype(np.intp)
        n_bins = table.shape[-1]
        xbin = np.clip(xbin, 0, n_bins - 2)
        delta = diff - xbin
        idx = types * n_bins + xbin
        left = np.take(table.flatten(), idx)
        right = np.take(table.flatten(), idx + 1)
        return (right - left) * delta + left
    
    g0_all = np.take(g0, kernel_config["g0_index"])
    g0_m = np.take(mass, kernel_config["g0_mass_index"], axis=-1)
    g_interp = interp(kernel_config["gamma_table"], kernel_config["g0_index"], g0_m,
                      kernel_config["gamma_min"], kernel_config["gamma_delta"])
    
    print(f"g_interp shape: {g_interp.shape}")
    print(f"g_interp[0, 0]: {g_interp[0, 0]}")
    print(f"g_interp.real[0, 0]: {g_interp.real[0, 0]}")
    print(f"g_interp.imag[0, 0]: {g_interp.imag[0, 0]}")
    
    print("\n" + "="*70)
    print("STEP 2: g and g_bw computation")
    print("="*70)
    
    g = g0_all * g_interp
    g_bw = np.dot(g, kernel_config["matrix_gamma"])
    
    print(f"g shape: {g.shape}")
    print(f"g[0, 0]: {g[0, 0]}")
    print(f"g_bw shape: {g_bw.shape}")
    print(f"g_bw[0, 0]: {g_bw[0, 0]}")
    print(f"g_bw.real[0, 0]: {g_bw.real[0, 0]}")
    print(f"g_bw.imag[0, 0]: {g_bw.imag[0, 0]}")
    
    print("\n" + "="*70)
    print("STEP 3: bw_dom computation")
    print("="*70)
    
    m0_all = np.take(m0, kernel_config["m0_index"])
    m0_m = np.take(mass, kernel_config["mass_index"], axis=-1)
    bw_dom = m0_all**2 - m0_m**2 - 1j * m0_all * g_bw
    
    print(f"m0_all shape: {m0_all.shape}")
    print(f"m0_all[0]: {m0_all[0]}")
    print(f"m0_m[0, 0]: {m0_m[0, 0]}")
    print(f"bw_dom shape: {bw_dom.shape}")
    print(f"bw_dom[0, 0]: {bw_dom[0, 0]}")
    
    print("\n" + "="*70)
    print("STEP 4: bw_p computation")
    print("="*70)
    
    bw_dom_all = np.take(bw_dom, kernel_config["bw_order"], axis=-1)
    n_wave = kernel_config["matrix_angle"].shape[1]
    n_res = kernel_config["bw_order"].size // n_wave
    
    print(f"bw_order size: {kernel_config['bw_order'].size}")
    print(f"n_wave: {n_wave}")
    print(f"n_res: {n_res}")
    print(f"bw_dom_all shape: {bw_dom_all.shape}")
    
    bw_dom_all_reshaped = bw_dom_all.reshape(n_events, n_wave, n_res)
    bw_p = np.prod(bw_dom_all_reshaped, axis=-1)
    
    print(f"bw_p shape: {bw_p.shape}")
    print(f"bw_p[0, 0]: {bw_p[0, 0]}")
    print(f"bw_p[0, 1]: {bw_p[0, 1]}")
    
    print("\n" + "="*70)
    print("STEP 5: Angular factors")
    print("="*70)
    
    ang = np.take(angle, kernel_config["angle_index"], axis=-2)
    ka = np.prod(np.cos(ang * kernel_config["angle_k"] + kernel_config["angle_b"]), axis=-1)
    fa = np.dot(ka, kernel_config["matrix_angle"])
    
    print(f"ang shape: {ang.shape}")
    print(f"ka shape: {ka.shape}")
    print(f"fa shape: {fa.shape}")
    print(f"ka[0, 0:5]: {ka[0, 0:5]}")
    print(f"fa[0, 0:5]: {fa[0, 0:5]}")
    
    print("\n" + "="*70)
    print("STEP 6: FL factors")
    print("="*70)
    
    fl_q = np.take(momentum, kernel_config["fl_q_index"], axis=-1)
    fl = interp(kernel_config["fl_table"], kernel_config["fl_type"], fl_q,
                kernel_config["fl_min"], kernel_config["fl_delta"])
    fl_all = np.take(fl, kernel_config["fl_order"], axis=-1)
    n_decay = kernel_config["fl_order"].size // n_wave
    fl_p = np.prod(fl_all.reshape(-1, n_wave, n_decay), axis=-1)
    
    print(f"fl_q shape: {fl_q.shape}")
    print(f"fl shape: {fl.shape}")
    print(f"fl_p shape: {fl_p.shape}")
    print(f"fl[0, 0]: {fl[0, 0]}")
    print(f"fl_p[0, 0]: {fl_p[0, 0]}")
    
    print("\n" + "="*70)
    print("STEP 7: common_amp_factor")
    print("="*70)
    
    one_over_bw = 1.0 / bw_p
    fa_times_fl = fa * fl_p
    common_amp_factor = one_over_bw * fa_times_fl
    
    print(f"one_over_bw[0, 0]: {one_over_bw[0, 0]}")
    print(f"fa_times_fl[0, 0]: {fa_times_fl[0, 0]}")
    print(f"common_amp_factor[0, 0]: {common_amp_factor[0, 0]}")
    
    print("\n" + "="*70)
    print("STEP 8: Amplitude ap, am")
    print("="*70)
    
    a = ck * common_amp_factor
    a_reshaped = a.reshape(-1, 2, n_wave // 2)
    ap = np.sum(a_reshaped[:, 0, :], axis=-1)
    am = np.sum(a_reshaped[:, 1, :], axis=-1)
    
    print(f"a shape: {a.shape}")
    print(f"ap shape: {ap.shape}")
    print(f"am shape: {am.shape}")
    print(f"ap[0]: {ap[0]}")
    print(f"am[0]: {am[0]}")
    
    print("\n" + "="*70)
    print("STEP 9: Time evolution and probabilities")
    print("="*70)
    
    eL = np.exp(-1j * time * (-Delta_m/2 - 1j * (Gamma + Delta_Gamma/2)/2))
    eH = np.exp(-1j * time * (+Delta_m/2 - 1j * (Gamma - Delta_Gamma/2)/2))
    gp = (eL + eH) / 2
    gm = (eL - eH) / 2
    
    print(f"gp[0]: {gp[0]}")
    print(f"gm[0]: {gm[0]}")
    
    poq = poq_rho * np.exp(1j * pop_phi)
    pap = gp * ap + gm * poq * am
    pam = (gm / poq) * ap + gp * am
    
    print(f"poq: {poq}")
    print(f"pap[0]: {pap[0]}")
    print(f"pam[0]: {pam[0]}")
    
    pb = np.abs(pap)**2
    pbbar = np.abs(pam)**2
    P = frac * pb * (1 - A_p) + (1 - frac) * pbbar * (1 + A_p)
    
    print(f"\npb[0]: {pb[0]}")
    print(f"pbbar[0]: {pbbar[0]}")
    print(f"P[0]: {P[0]}")
    
    Q = np.sum(weight * P)
    print(f"\nQ: {Q}")


if __name__ == "__main__":
    compare_step_by_step()

"""
Build gamma_table and bf_table for the kernel from a parsed config.

Gamma table: stores mass-dependent width Gamma(m) extracted via:
    gamma = 1j * (1/amplitude(m) + m^2 - m0^2) / m0
This formula works for ALL resonance models (BW, GS_rho, Bugg, Flatte, one...).

Usage:
  from pwa_gpu.parse_config import parse_config
  from pwa_gpu.build_tables import build_tables
  
  cfg, pw_list, kw_list = parse_config("config.yml")
  cfg = build_tables(cfg, pw_list, kw_list, "config.yml")
"""

import os
import numpy as np
from collections import OrderedDict


# ====================================================================
# Physics helpers
# ====================================================================

def blatt_weisskopf(q, L, d=3.0):
    """Blatt-Weisskopf barrier factor for orbital angular momentum L."""
    z = (q * d) ** 2
    if L == 0: return 1.0
    if L == 1: return np.sqrt(np.maximum(0., 1.0 / (1.0 + z)))
    if L == 2: return np.sqrt(np.maximum(0., 1.0 / (z**2 + 3.0*z + 9.0)))
    if L == 3: return np.sqrt(np.maximum(0., 1.0 / (z**3 + 6.0*z**2 + 45.0*z + 225.0)))
    if L == 4: return np.sqrt(np.maximum(0., 1.0 / (z**4 + 10.0*z**3 + 135.0*z**2 + 1575.0*z + 11025.0)))
    return 1.0


def breakup_momentum(m, m1, m2):
    """Breakup momentum for decay m -> m1 + m2 (GeV)."""
    s = m**2
    mabp = (m1 + m2)**2
    mabm = (m1 - m2)**2
    p2 = np.maximum(0., (s - mabp) * (s - mabm))
    return np.sqrt(p2) / (2.0 * np.maximum(m, 1e-10))


# ====================================================================
# Amplitude computation per model
# ====================================================================

def amp_BW(m, m0, gamma0, L=0, d=3.0, m1=0.139, m2=0.139):
    """Standard relativistic Breit-Wigner with mass-dependent width."""
    q = breakup_momentum(m, m1, m2)
    q0 = breakup_momentum(m0, m1, m2)
    bf = blatt_weisskopf(q, L, d)
    bf0 = blatt_weisskopf(q0, L, d)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(q0 > 0, q / q0, 0.0)
        gamma_m = gamma0 * (ratio ** (2 * L + 1)) * (m0 / np.maximum(m, 1e-10)) * (bf / np.maximum(bf0, 1e-10))**2
        gamma_m = np.nan_to_num(gamma_m, nan=0.0, posinf=0.0, neginf=0.0)
    bw = 1.0 / (m0**2 - m**2 - 1j * m0 * gamma_m)
    return bw


def amp_constant(m, m0):
    """Constant amplitude = 1.0 (model='one')."""
    return np.ones_like(m, dtype=np.complex128)


def amp_from_file(m, m0, file_gamma, file_mass):
    """BW with mass-dependent width loaded from file (width_linear_npy)."""
    gamma_m = np.interp(m, file_mass, file_gamma)
    bw = 1.0 / (m0**2 - m**2 - 1j * m0 * gamma_m)
    return bw


def amp_Bugg(m, m0, params=None):
    """Bugg lineshape (from bugg_model.py)."""
    # Bugg parameters (defaults from bugg_model.py)
    b1 = params.get('b1', 1.302)
    b2 = params.get('b2', 0.340)
    A = params.get('A', 2.426)
    g_4pi = params.get('g_4pi', 0.011)
    g_2K = params.get('g_2K', 0.6)
    g_2eta = params.get('g_2eta', 0.2)
    alpha = params.get('alpha', 1.3)
    sA = params.get('sA', 0.41) * 0.13957039**2
    s0_4pi = params.get('s0_4pi', 7.082 / 2.845)
    lambda_4pi = params.get('lambda_4pi', 2.845)
    mPiPlus = 0.13957039
    mKPlus = 0.493677
    mEta = 0.547863
    M = m0  # use the resonance mass as M

    s = m**2
    M2 = M * M
    sA_val = sA

    def rho_2(s, s0):
        r2 = np.maximum(0., 1. - 4. * s0 / s)
        return np.sqrt(r2) + 0j

    def rho_4pi(s, lam, s0):
        return 1.0 / (1.0 + np.exp(lam * (s0 - s)))

    def Buggj1(s, m0):
        rho_pipi = np.sqrt(np.maximum(0., rho_2(s, m0 * m0).real))
        t1 = rho_pipi * np.log(np.maximum(1e-10, (1. - rho_pipi) / (1. + rho_pipi)))
        t1 = np.where(rho_pipi > 0, t1, np.zeros_like(rho_pipi))
        return (2. + t1) / np.pi

    z = Buggj1(s, mPiPlus) - Buggj1(M2, mPiPlus)
    g1sg = M * (b1 + b2 * s) * np.exp(-(s - M2) / A)
    adlerZero = (s - sA_val) / (M2 - sA_val)

    gamma_2pi = g1sg * adlerZero * rho_2(s, mPiPlus**2)
    gamma_2K = g_2K * g1sg * s / M2 * np.exp(-alpha * np.abs(s - 4. * mKPlus**2)) * rho_2(s, mKPlus**2)
    gamma_2eta = g_2eta * g1sg * s / M2 * np.exp(-alpha * np.abs(s - 4. * mEta**2)) * rho_2(s, mEta**2)
    gamma_4pi = M * g_4pi * rho_4pi(s, lambda_4pi, s0_4pi)
    Gamma_tot = gamma_2pi + gamma_2K + gamma_2eta + gamma_4pi
    
    iBW = M2 - s - adlerZero * g1sg * z - 1j * Gamma_tot
    BW = 1.0 / iBW
    return BW


def amp_Flatte(m, m0, g_vals, mass_list):
    """Flatte coupled-channel lineshape.
    
    amplitude = 1 / (m0^2 - m^2 - i * m0 * sum_i g_i * rho_i(m))
    where rho_i(m) = breakup_momentum(m, m_i1, m_i2) / m  (complex below threshold)
    """
    total_width = 0j
    for i, (g_val, (m1, m2)) in enumerate(zip(g_vals, mass_list)):
        if g_val == 0:
            continue
        s = m**2
        mabp = (m1 + m2)**2
        mabm = (m1 - m2)**2
        p2 = (s - mabp) * (s - mabm) / (4.0 * np.maximum(s, 1e-10))
        # Complex sqrt: real above threshold, imaginary below
        rho = np.where(p2 >= 0, np.sqrt(np.maximum(0., p2)) + 0j,
                       1j * np.sqrt(np.maximum(0., -p2)))
        total_width += g_val * rho

    bw = 1.0 / (m0**2 - m**2 - 1j * m0 * total_width)
    return bw


# ====================================================================
# Gamma table builder
# ====================================================================

N_GAMMA_POINTS = 500
M_PI = 0.13957039
M_B = 5.279
G_MIN = 2.0 * M_PI
G_MAX = M_B


def compute_gamma_table(cfg, pw_list, kw_list, config_path):
    """
    Compute gamma_table for each g0 entry using the universal formula:
      gamma = 1j * (1/amplitude(m) + m^2 - m0^2) / m0
    
    Returns (gamma_table, g_min, g_delta, n_gamma_points).
    """
    n_g0 = cfg['n_g0']
    mass_grid = np.linspace(G_MIN, G_MAX, N_GAMMA_POINTS)
    g_delta = (G_MAX - G_MIN) / (N_GAMMA_POINTS - 1)
    cfg_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else "."
    from pwa_gpu.parse_config import get_particle

    # Build resonance info
    res_info = OrderedDict()
    for pw in pw_list:
        for res in pw.resonances:
            if res.name not in res_info:
                res_info[res.name] = res

    gamma_table = np.zeros((n_g0, N_GAMMA_POINTS), dtype=np.complex128)
    gi = 0

    for res_name, res in res_info.items():
        props = get_particle(cfg, res_name) or {}
        extra = res.extra
        m0 = res.mass
        width = res.width if res.width > 0 else 0.1

        if res.model == 'FlatteC':
            mass_list = props.get('mass_list', [[M_PI, M_PI]] * 4) if isinstance(props, dict) else [[M_PI, M_PI]] * 4
            g_vals = []
            for i in range(4):
                gk = props.get(f'g_{i}', 0.0) if isinstance(props, dict) else 0.0
                g_vals.append(gk)
            for sub_g in range(4):
                # For Flatte, each coupling g_i contributes to the total width
                # We create separate g0 entries per channel, each with its own rho_i(m)
                g_only = [0]*4
                g_only[sub_g] = g_vals[sub_g]
                amp = amp_Flatte(mass_grid, m0, g_only, mass_list)
                gamma_val = 1j * (1.0/amp + mass_grid**2 - m0**2) / m0
                gamma_table[gi] = gamma_val
                gi += 1

        elif res.model == 'one':
            amp = amp_constant(mass_grid, m0)
            gamma_table[gi] = 1j * (1.0/amp + mass_grid**2 - m0**2) / m0
            gi += 1

        elif res.model == 'Bugg':
            bugg_params = {k: v for k, v in (props.items() if isinstance(props, dict) else {})}
            amp = amp_Bugg(mass_grid, m0, bugg_params)
            gamma_table[gi] = 1j * (1.0/amp + mass_grid**2 - m0**2) / m0
            gi += 1

        elif res.model == 'width_linear_npy' and 'file' in extra:
            fpath = os.path.join(cfg_dir, extra['file'])
            if os.path.exists(fpath):
                data = np.load(fpath)
                file_mass = data[:, 0]
                file_gamma = data[:, 1] + 1j * data[:, 2]
                amp = amp_from_file(mass_grid, m0, file_gamma, file_mass)
            else:
                amp = amp_BW(mass_grid, m0, width)
            gamma_table[gi] = 1j * (1.0/amp + mass_grid**2 - m0**2) / m0
            gi += 1

        else:
            # Default: mass-dependent BW with L from the particle's J
            L = res.J if res.J > 0 else 0
            m1 = M_PI  # assume ππ decay
            m2 = M_PI
            amp = amp_BW(mass_grid, m0, width, L, d=3.0, m1=m1, m2=m2)
            gamma_table[gi] = 1j * (1.0/amp + mass_grid**2 - m0**2) / m0
            gi += 1

    return gamma_table, G_MIN, g_delta, N_GAMMA_POINTS


# ====================================================================
# BF table builder
# ====================================================================

def compute_bf_table(cfg, pw_list, kw_list):
    """Barrier factor interpolation for each bf_type."""
    n_bf_types = cfg['n_bf_types']
    n_bf_points = 500
    q_grid = np.linspace(0.0, 3.0, n_bf_points)
    q_delta = 3.0 / (n_bf_points - 1)

    bf_type_to_L = {}
    idx = 0
    for kw in kw_list:
        for L in kw.bf_order_entries:
            if idx not in bf_type_to_L:
                bf_type_to_L[idx] = int(L)
            idx += 1

    bf_table = np.zeros((n_bf_types, n_bf_points), dtype=np.float64)
    for bf_idx in range(n_bf_types):
        base_idx = bf_idx % (max(bf_type_to_L.keys()) + 1) if bf_type_to_L else 0
        L = bf_type_to_L.get(base_idx, 0)
        bf_table[bf_idx] = blatt_weisskopf(q_grid, L, 3.0)

    return bf_table, 0.0, q_delta, n_bf_points


# ====================================================================
# Main entry point
# ====================================================================

def build_tables(cfg, pw_list, kw_list, config_path=None):
    """Build gamma_table and bf_table, add them to cfg."""
    gamma_table, g_min, g_delta, n_gp = compute_gamma_table(cfg, pw_list, kw_list, config_path)
    bf_table, q_min, q_delta, n_bp = compute_bf_table(cfg, pw_list, kw_list)

    cfg['gamma_table'] = gamma_table
    cfg['g_min'] = g_min
    cfg['g_delta'] = g_delta
    cfg['n_gamma_points'] = n_gp
    cfg['bf_table'] = bf_table
    cfg['q_min'] = q_min
    cfg['q_delta'] = q_delta
    cfg['n_bf_points'] = n_bp

    print(f"  gamma_table: {gamma_table.shape}, g_min={g_min:.4f}, g_delta={g_delta:.6f}")
    print(f"  bf_table:    {bf_table.shape}, q_min={q_min:.4f}, q_delta={q_delta:.6f}")

    return cfg


if __name__ == '__main__':
    import sys
    from pwa_gpu.parse_config import parse_config
    if len(sys.argv) < 2:
        print("Usage: python -m pwa_gpu.build_tables <config.yml>")
        sys.exit(1)
    cfg, pw_list, kw_list = parse_config(sys.argv[1])
    cfg = build_tables(cfg, pw_list, kw_list, sys.argv[1])
    print(f"  gamma_table[0,:3]: {cfg['gamma_table'][0,:3]}")
    print(f"  bf_table[0,:3]:    {cfg['bf_table'][0,:3]}")

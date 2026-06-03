"""
Build gamma_table and bf_table for the kernel from a parsed config.

Usage:
  from pwa_gpu.parse_config import parse_config
  from pwa_gpu.build_tables import build_tables
  
  cfg, pw_list, kw_list = parse_config("config.yml")
  cfg = build_tables(cfg, pw_list, kw_list, "config.yml")
"""

import os
import numpy as np
from collections import OrderedDict


def blatt_weisskopf(q, L, d=3.0):
    """Blatt-Weisskopf barrier factor for orbital angular momentum L."""
    z = (q * d) ** 2
    if L == 0:
        return 1.0
    if L == 1:
        return np.sqrt(1.0 / (1.0 + z))
    if L == 2:
        return np.sqrt(1.0 / (z ** 2 + 3.0 * z + 9.0))
    if L == 3:
        return np.sqrt(1.0 / (z ** 3 + 6.0 * z ** 2 + 45.0 * z + 225.0))
    if L == 4:
        return np.sqrt(1.0 / (z ** 4 + 10.0 * z ** 3 + 135.0 * z ** 2 + 1575.0 * z + 11025.0))
    return 1.0


def breakup_momentum(m, m1, m2):
    """Breakup momentum for decay m → m1 + m2 (GeV)."""
    m2_sum = (m1 + m2) ** 2
    m2_diff = (m1 - m2) ** 2
    val = (m ** 2 - m2_sum) * (m ** 2 - m2_diff)
    return np.sqrt(np.maximum(0.0, val)) / (2.0 * m)


def compute_gamma_table(cfg, pw_list, kw_list, config_path):
    """
    Compute gamma_table (mass-dependent width) for each g0 entry.
    
    Returns (gamma_table, g_min, g_delta, n_gamma_points).
    """
    n_g0 = cfg['n_g0']
    n_gamma_points = 500
    m_pi = 0.13957039
    m_B = 5.279
    g_min = 2.0 * m_pi
    g_max = m_B
    g_delta = (g_max - g_min) / (n_gamma_points - 1)
    mass_grid = np.linspace(g_min, g_max, n_gamma_points)
    cfg_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else "."

    # Collect unique (res_name, model) from the config's particle section
    # Using the bwall_key_to_idx / res_name_to_bwall from the parser
    from pwa_gpu.parse_config import get_particle

    # We need res_name_to_bwall — it's not in cfg. Reconstruct from wave_info.
    res_name_to_bwall = {}
    for wi in cfg.get('wave_info', []):
        for entry in wi.get('ck_formula', []):
            if entry[0] in ('g_ls', 'g_lsbar'):
                name = entry[1]
                if name not in res_name_to_bwall:
                    res_name_to_bwall[name] = len(res_name_to_bwall)

    # Build resonance info dict
    res_info = OrderedDict()
    for pw in pw_list:
        for res in pw.resonances:
            if res.name not in res_info:
                res_info[res.name] = res

    gamma_table = np.zeros((n_g0, n_gamma_points), dtype=np.complex128)
    gi = 0
    for res_name in res_info:
        res = res_info[res_name]
        props = get_particle(cfg, res_name)
        n_g = 4 if res.model == 'FlatteC' else 1

        for sub_g in range(n_g):
            model = res.model
            extra = res.extra

            if model == 'width_linear_npy' and 'file' in extra:
                # Load precomputed gamma file
                fpath = os.path.join(cfg_dir, extra['file'])
                if os.path.exists(fpath):
                    data = np.load(fpath)
                    file_mass = data[:, 0]
                    file_gamma = data[:, 1] + 1j * data[:, 2]
                    gamma_table[gi] = np.interp(mass_grid, file_mass, file_gamma.real) + \
                                      1j * np.interp(mass_grid, file_mass, file_gamma.imag)
                else:
                    gamma_table[gi] = (res.width if res.width > 0 else 0.1) + 0.0j
                gi += 1

            elif model == 'BW':
                gamma_table[gi] = (res.width if res.width > 0 else 0.1) + 0.0j
                gi += 1

            elif model == 'GS_rho':
                # Mass-dependent width: Gamma(m) = Gamma0 * (q/q0)^(2L+1) * (m0/m) * Bl(q)^2/Bl(q0)^2
                m0 = res.mass
                Gamma0 = res.width
                d = 3.0
                L = 1  # rho(770) is spin-1 → ππ (both spin-0)
                q = breakup_momentum(mass_grid, m_pi, m_pi)
                q0 = breakup_momentum(m0, m_pi, m_pi)
                bf = blatt_weisskopf(q, L, d)
                bf0 = blatt_weisskopf(q0, L, d)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(q0 > 0, q / q0, 0.0)
                    mass_dep = np.where(mass_grid > 0, m0 / mass_grid, 0.0)
                    gamma_val = Gamma0 * (ratio ** (2 * L + 1)) * mass_dep * (bf / bf0) ** 2
                    gamma_val = np.nan_to_num(gamma_val, nan=0.0, posinf=0.0, neginf=0.0)
                gamma_table[gi] = gamma_val.astype(np.complex128)
                gi += 1

            elif model in ('Bugg', 'FlatteC', 'one', 'pipi_Swave', 'spline_c_idx'):
                # Placeholder: needs tf_pwa or dedicated implementation
                gamma_table[gi] = (res.width if res.width > 0 else 0.1) + 0.0j
                gi += 1

            else:
                gamma_table[gi] = (res.width if res.width > 0 else 0.1) + 0.0j
                gi += 1

    return gamma_table, g_min, g_delta, n_gamma_points


def compute_bf_table(cfg, pw_list, kw_list):
    """
    Compute bf_table (barrier factor interpolation) for each bf_type.
    Each bf_type has (L, d, qkey). The barrier function Bl(q) only depends on L.
    """
    n_bf_types = cfg.get('n_bf_types_base', len(cfg['bf_index']))
    n_bf_points = 500
    q_min = 0.0
    q_max = 3.0
    q_delta = q_max / (n_bf_points - 1)
    q_grid = np.linspace(q_min, q_max, n_bf_points)
    d = 3.0

    # Build L→bf_type index mapping from kw bf_order_entries
    # Each kw stores bf_order_entries as L values in chain order
    # We need to know which bf_type index corresponds to which L
    # Reconstruct from the kw_list and the config's unique_bf (which we need to pass)
    # For now, build a map from the bf_order and unique_bf data that's in cfg
    # Actually simpler: reassign bf_table to match the actual bf_type order
    # The bf_type ordering is determined by unique_bf in the parser
    # For now, use unique L values from bf_order
    unique_Ls = set()
    for kw in kw_list:
        for L in kw.bf_order_entries:
            unique_Ls.add(int(L))

    # For each bf_type index, determine L from the kw_list
    # Walk all kw's bf_order_entries and build mapping
    bf_type_to_L = {}
    idx = 0
    for kw in kw_list[:]:
        for L in kw.bf_order_entries:
            if idx not in bf_type_to_L:
                bf_type_to_L[idx] = L
            idx += 1
    n_base = max(bf_type_to_L.keys()) + 1 if bf_type_to_L else n_bf_types

    bf_table = np.zeros((n_bf_types, n_bf_points), dtype=np.float64)
    for bf_idx in range(n_bf_types):
        L = bf_type_to_L.get(bf_idx % n_base if n_base > 0 else 0, 0)
        bf_table[bf_idx] = blatt_weisskopf(q_grid, int(L), d)

    return bf_table, q_min, q_delta, n_bf_points


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

    print(f"  gamma_table[0,:5]: {cfg['gamma_table'][0,:5]}")
    print(f"  bf_table[0,:5]:    {cfg['bf_table'][0,:5]}")

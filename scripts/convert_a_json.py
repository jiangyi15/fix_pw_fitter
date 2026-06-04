#!/usr/bin/env python3
"""
Scale g_ls values in a.json by 1/BF(q₀) for each decay.

tf_pwa amplitude includes BF(q)/BF(q₀) per decay (normalized barrier factor).
Our GPU kernel uses raw BF(q). To match tf_pwa's fitted couplings, each decay
coupling g_ls must be divided by BF(q₀) — the Blatt-Weisskopf factor at the
parent resonance's nominal mass.

Only sub-decay couplings (keys containing "->") are scaled.
B-level couplings (keys starting with "B->") are NOT scaled.

Usage:
    python convert_a_json.py config_angle.yml a.json -o a_scaled.json
"""

import os, sys, json, yaml, argparse
import numpy as np


def blatt_weisskopf(q, L, d=3.0):
    """Blatt-Weisskopf barrier factor F_L(q) = sqrt(P_L(0) / P_L(z)), z=(q*d)².
    
    Normalized so F_L(0) = 1.0 for all L.
    """
    z = (q * d) ** 2
    if L == 0: return 1.0
    if L == 1: return np.sqrt(max(0., 1.0 / (1.0 + z)))
    if L == 2: return np.sqrt(max(0., 9.0 / (9.0 + 3.0*z + z**2)))
    if L == 3: return np.sqrt(max(0., 225.0 / (225.0 + 45.0*z + 6.0*z**2 + z**3)))
    if L == 4: return np.sqrt(max(0., 11025.0 / (11025.0 + 1575.0*z + 135.0*z**2 + 10.0*z**3 + z**4)))
    return 1.0


def breakup_momentum(m, m1, m2):
    s = m**2
    mabp = (m1 + m2)**2
    mabm = (m1 - m2)**2
    p2 = max(0., (s - mabp) * (s - mabm))
    return np.sqrt(p2) / (2.0 * max(m, 1e-10))


def load_particle_masses(config_path):
    """Extract {name: mass} from config.yml particle section."""
    with open(config_path) as f:
        ycfg = yaml.safe_load(f)
    masses = {}
    particle = ycfg.get('particle', {})
    for name, props in particle.items():
        if isinstance(props, dict) and 'mass' in props:
            masses[name] = float(props['mass'])
    # Also get finals for π masses
    finals = particle.get('$finals', [])
    for f in finals:
        fp = particle.get(f, {})
        if isinstance(fp, dict) and 'mass' in fp:
            masses[f] = float(fp['mass'])
    return masses


def get_L_for_decay(config_path, parent, daughters):
    """Look up orbital angular momentum L for a decay from config wave definitions."""
    with open(config_path) as f:
        ycfg = yaml.safe_load(f)
    finals = set(ycfg.get('particle', {}).get('$finals', []))
    # Try to find the L from wave chain definitions
    # Walk through decay chains to find the L for parent->daughters
    decay = ycfg.get('decay', {})
    for dname, dlist in decay.items():
        for item in dlist:
            if isinstance(item, list):
                # Wave chain: look for parent at this level
                pass
    # Default: L=0 if not found (most sub-decays are S-wave or P-wave)
    return 0


def get_L_from_full_prefix(full_prefix):
    """Try to determine L from the decay name pattern."""
    # Common patterns based on known decays
    # rhoA/f0/f2 → pip1.pim1: varies
    return None  # unknown — caller should handle


def convert(config_path, a_json_path, output_path=None):
    """Scale a.json g_ls values by 1/BF(q₀)."""
    with open(a_json_path) as f:
        data = json.load(f)
    
    masses = load_particle_masses(config_path)
    
    # Parse config for L info per decay
    with open(config_path) as f:
        ycfg = yaml.safe_load(f)
    finals = set(ycfg.get('particle', {}).get('$finals', []))
    
    # Build decay → L map from physical wave chain definitions
    decay_L = {}
    from pwa_gpu.parse_config import expand_physical_waves, expand_to_kernel_waves
    pw_list = expand_physical_waves(ycfg)
    kw_list = expand_to_kernel_waves(pw_list, ycfg)
    for kw in kw_list:
        pw = next((p for p in pw_list if p.id == kw.pw_id), None)
        if pw:
            d_idx = 0
            for step in pw.chain:
                if step.daughters and len(step.daughters) >= 2:
                    dkey = (step.parent, tuple(step.daughters[:2]))
                    if dkey not in decay_L:
                        L = kw.bf_order_entries[d_idx] if d_idx < len(kw.bf_order_entries) else 0
                        decay_L[dkey] = int(L)
                    d_idx += 1
    
    # Process all r/i pairs
    scaled = {}
    keys = list(data.keys())
    n_scaled = 0
    
    for k in keys:
        v = data[k]
        # Only scale sub-decay g_ls/g_lsbar keys (contain "->")
        if '->' in k and not k.startswith('B->'):
            # Extract prefix before _g_ls_{idx}r or _g_lsbar_{idx}r
            # e.g. "rhoA->pip1.pim1_g_ls_0r" → prefix="rhoA->pip1.pim1"
            # e.g. "a1(1260)m->rhoA.pim2_g_ls_1r" → prefix="a1(1260)m->rhoA.pim2"
            prefix = k.split('_g_ls_')[0].split('_g_lsbar_')[0]
            if '->' in prefix and not prefix.startswith('B->'):
                # Parse parent and first daughter
                parent, rest = prefix.split('->', 1)
                daus = rest.split('.')
                if len(daus) >= 2:
                    d1, d2 = daus[0], daus[1]
                    # Get masses
                    m0 = masses.get(parent, 0.769)
                    m1 = masses.get(d1, 0.13957)
                    m2 = masses.get(d2, 0.13957)
                    # Get L from the key's _g_ls_{idx} suffix
                    import re
                    m_ls = re.search(r'_g_ls_(\d+)', k)
                    if not m_ls:
                        m_ls = re.search(r'_g_lsbar_(\d+)', k)
                    L = int(m_ls.group(1)) if m_ls else 0
                    # Also try decay_L lookup
                    dkey = (parent, (d1, d2))
                    if dkey in decay_L:
                        L = decay_L[dkey]
                    # Compute BF(q₀)
                    q0 = breakup_momentum(m0, m1, m2)
                    bf0 = blatt_weisskopf(q0, L, 3.0)
                    if bf0 > 1e-10 and abs(bf0 - 1.0) > 1e-6:
                        scaled[k] = v / bf0
                        n_scaled += 1
                    else:
                        scaled[k] = v
                else:
                    scaled[k] = v
            else:
                scaled[k] = v
        else:
            scaled[k] = v
    
    print(f"Scaled {n_scaled} entries by 1/BF(q₀)", flush=True)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(scaled, f, indent=2)
        print(f"Saved: {output_path}", flush=True)
    
    return scaled


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Scale a.json by 1/BF(q₀)')
    p.add_argument('config', help='config_angle.yml')
    p.add_argument('a_json', help='input a.json')
    p.add_argument('-o', '--output', default=None, help='output path')
    args = p.parse_args()
    convert(args.config, args.a_json, args.output)

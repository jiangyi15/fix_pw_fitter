#!/usr/bin/env python3
"""
Standalone converter: raw 4‑momenta → pwa_gpu arrays using tf_pwa.

Reads data file paths from the ``data:`` section of config.yml,
resolves them relative to the config file, loads everything,
converts masses/q/angles, and saves .npz files for the GPU pipeline.

Masses & q‑values: pure numpy.
Angles: tf_pwa's cal_angle_from_momentum (GPU accelerated).

Usage:
    python convert_with_tfpwa.py config_angle.yml --out converted/
    python convert_with_tfpwa.py config_angle.yml --data-max 50000
"""

import os, sys, time, yaml, json, itertools
import numpy as np
from collections import OrderedDict


# =====================================================================
# Lorentz helpers
# =====================================================================

def inv_mass(p):
    m2 = p[:,0]*p[:,0] - p[:,1]*p[:,1] - p[:,2]*p[:,2] - p[:,3]*p[:,3]
    return np.sqrt(np.clip(m2, 0., None))

def decay_q(m0, m1, m2):
    m12s, m12d = m1+m2, m1-m2
    p2 = (m0-m12s)*(m0+m12s)*(m0-m12d)*(m0+m12d)
    return np.sqrt(np.clip(p2, 0., None)) / np.clip(2.*m0, 1e-15, None)


# =====================================================================
# Config utilities
# =====================================================================

def resolve_path(path, config_dir):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(config_dir, path))

def get_data_files(config_path):
    """Read ``data:`` section of config.yml → dict of resolved file lists."""
    config_dir = os.path.dirname(os.path.abspath(config_path))
    with open(config_path) as f:
        sec = yaml.safe_load(f).get('data', {})

    def get(key):
        val = sec.get(key)
        if val is None: return []
        if isinstance(val, str): return [resolve_path(val, config_dir)]
        return [resolve_path(v, config_dir) for v in val]

    bg_frac = sec.get('bg_frac', 0.0)

    return dict(
        p4_data        = get('data') or get('dataall'),
        p4_phsp        = get('phsp'),
        time_data      = get('data_time'),
        time_phsp      = get('phsp_time'),
        tag_data       = get('data_tag1'),
        tag_phsp       = get('phsp_tag1'),
        eta_data       = get('data_eta1'),
        eta_phsp       = get('phsp_eta1'),
        bkg_data       = get('data_bg_value'),
        bkg_phsp       = get('phsp_bg_value'),
        weight_phsp    = get('phsp_weight'),
        bg_frac        = bg_frac,
    )


def load_momenta(filename, order):
    p4 = np.load(filename).astype(np.float64)
    if p4.ndim == 3 and p4.shape[1] == len(order):
        return {name: p4[:, i, :] for i, name in enumerate(order)}
    raise ValueError(f"Shape {p4.shape}, expected (n,{len(order)},4)")


# =====================================================================
# Parse config structure (matching pwa_gpu.parse_config conventions)
# =====================================================================

def parse_structure(config_path):
    """
    Return all structural info needed for conversion.
    
    Returns: (finals_list, top_name, dat_order,
              res_daughters,      # resonance → [final daughters]
              q_entries,          # (parent, daughters) → q column
              res_name_to_bwall,  # resonance name → mass column
              n_m0_base, q_stride_base, perms, inter_tree, pw_list)
    """
    from pwa_gpu.parse_config import (
        parse_config, get_finals, get_top, get_mass_key
    )

    with open(config_path) as f:
        ycfg = yaml.safe_load(f)

    finals_list = get_finals(ycfg)
    finals_set  = set(finals_list)
    top_name    = get_top(ycfg)
    dat_order   = ycfg.get('data', {}).get('dat_order', finals_list)
    ident_particles = ycfg.get('data', {}).get('identical_particles', [])

    _, pw_list, _ = parse_config(config_path)

    # Permutations
    perms = [{}]
    if ident_particles:
        groups = [list(itertools.permutations(g)) for g in ident_particles]
        for combo in itertools.product(*groups):
            swap = {}
            for c, g in zip(combo, ident_particles):
                for ci, pi in zip(c, g):
                    if ci != pi: swap[ci] = pi
            if swap: perms.append(swap)

    # Column mapping (matches build_kernel_config)
    res_name_to_bwall = OrderedDict()
    bwall_key_to_idx  = OrderedDict()
    for pw in pw_list:
        for res in pw.resonances:
            if res.name in res_name_to_bwall: continue
            mk = get_mass_key(res.name, pw, finals_set, ident_groups=None)
            key = (mk, res.mass, res.width, res.model)
            if key not in bwall_key_to_idx:
                bwall_key_to_idx[key] = len(bwall_key_to_idx)
            res_name_to_bwall[res.name] = bwall_key_to_idx[key]
    n_m0_base = len(bwall_key_to_idx)

    # Q column mapping
    q_entries = OrderedDict()
    for pw in pw_list:
        for step in pw.chain:
            if step.daughters:
                qkey = (step.parent, tuple(sorted(step.daughters)))
                if qkey not in q_entries:
                    q_entries[qkey] = len(q_entries)
    q_stride_base = len(q_entries)

    # Resonance → final-daughters
    particle_sec = ycfg.get('particle', {})
    decay_sec    = ycfg.get('decay', {})

    def get_final_daus(name, visited=None):
        if visited is None: visited = set()
        if name in visited or name in finals_set:
            return [name] if name in finals_set else []
        visited.add(name)
        if name in decay_sec:
            daus = [x for x in decay_sec[name] if isinstance(x, str)]
            if daus:
                out = []
                for d in daus: out.extend(get_final_daus(d, visited))
                return out
        return []

    inter_of_res = {}
    for name, props in particle_sec.items():
        if isinstance(props, list):
            for p in props:
                if isinstance(p, str):
                    inter_of_res[p] = name

    res_daughters = {}
    for pw in pw_list:
        for res in pw.resonances:
            if res.name in res_daughters: continue
            inter = inter_of_res.get(res.name)
            src = inter or res.name
            fd = get_final_daus(src)
            if fd: res_daughters[res.name] = sorted(set(fd))

    # Intermediate tree
    inter_tree = {}
    for dname, dlist in decay_sec.items():
        daus = [x for x in dlist if isinstance(x, str)]
        if daus and dname != top_name:
            inter_tree[dname] = daus
    inter_tree[top_name] = list(finals_list)

    return (finals_list, top_name, dat_order, res_daughters,
            q_entries, res_name_to_bwall, n_m0_base, q_stride_base,
            perms, inter_tree, pw_list)


# =====================================================================
# 4‑momentum propagation
# =====================================================================

def propagate_momenta(raw_p4, inter_tree, top_name, finals_list, res_daughters):
    all_p4 = dict(raw_p4)
    done   = set(raw_p4.keys())

    changed = True
    while changed:
        changed = False
        for parent, daus in inter_tree.items():
            if parent in done: continue
            if all(d in done for d in daus):
                all_p4[parent] = sum(all_p4[d] for d in daus)
                done.add(parent); changed = True
        if top_name not in done and all(f in done for f in finals_list):
            all_p4[top_name] = sum(all_p4[f] for f in finals_list)
            done.add(top_name); changed = True

    for rname, fd in res_daughters.items():
        if rname not in done:
            all_p4[rname] = sum(all_p4[d] for d in fd)
            done.add(rname)
    return all_p4


# =====================================================================
# Mass and q-value computation
# =====================================================================

def compute_mass_block(raw_p4, inter_tree, top_name, finals_list,
                       res_daughters, res_name_to_bwall, n_m0_base):
    all_p4 = propagate_momenta(raw_p4, inter_tree, top_name,
                                finals_list, res_daughters)
    block = np.zeros((len(raw_p4[list(raw_p4)[0]]), n_m0_base), dtype=np.float64)
    for rname, col in res_name_to_bwall.items():
        if rname in all_p4:
            block[:, col] = inv_mass(all_p4[rname])
    return block


def compute_q_block(raw_p4, inter_tree, top_name, finals_list,
                    res_daughters, q_entries, q_stride_base):
    all_p4 = propagate_momenta(raw_p4, inter_tree, top_name,
                                finals_list, res_daughters)
    n_ev = len(raw_p4[list(raw_p4)[0]])

    def get_p4(name):
        if name in all_p4: return all_p4[name]
        if name in res_daughters: return sum(raw_p4[d] for d in res_daughters[name])
        if name in inter_tree: return sum(get_p4(d) for d in inter_tree[name])
        return np.zeros((n_ev, 4))

    block = np.zeros((n_ev, q_stride_base), dtype=np.float64)
    for (parent, daughters), col in q_entries.items():
        pp = get_p4(parent)
        p1 = get_p4(daughters[0])
        p2 = get_p4(daughters[1]) if len(daughters) > 1 else np.zeros((n_ev, 4))
        block[:, col] = decay_q(inv_mass(pp), inv_mass(p1),
                                inv_mass(p2) if len(daughters) > 1 else 0.)
    return block


# =====================================================================
# Build tf_pwa DecayGroup
# =====================================================================

def build_decay_group(ycfg):
    from tf_pwa.particle import BaseParticle, BaseDecay, DecayChain, DecayGroup

    finals = set(ycfg.get('particle', {}).get('$finals', []))
    top    = ycfg.get('particle', {}).get('$top', 'B')
    decay_sec = ycfg.get('decay', {})
    b_decays = list(decay_sec.get(top, []))

    def collect_chain(b_line):
        b_daus = [x for x in b_line if isinstance(x, str)]
        BP, BD = BaseParticle(top), [BaseParticle(d) for d in b_daus]
        steps = [BaseDecay(BP, BD)]

        def add_decays(parent_name, parent_bp):
            if parent_name in finals: return
            if parent_name not in decay_sec: return
            raw = decay_sec[parent_name]
            daus = [x for x in raw if isinstance(x, str)]
            if not daus: return
            d_bp = [BaseParticle(d) for d in daus]
            steps.append(BaseDecay(parent_bp, d_bp))
            for dn, dp in zip(daus, d_bp):
                add_decays(dn, dp)

        for d, bp in zip(b_daus, BD):
            add_decays(d, bp)
        return DecayChain(steps)

    chains = [collect_chain(bl) for bl in b_decays
              if any(isinstance(x, str) for x in bl)]
    return DecayGroup(chains)


# =====================================================================
# tf_pwa angle computation
# =====================================================================

def compute_angles_tfpwa(raw_p4, dg, dat_order):
    """Use tf_pwa cal_angle_from_momentum → θ₁, θ₂, φ."""
    from tf_pwa.particle import BaseParticle
    from tf_pwa.cal_angle import cal_angle_from_momentum
    from tf_pwa.data import data_to_numpy
    import tensorflow as tf

    n_ev = len(raw_p4[list(raw_p4)[0]])
    p4_tf = {BaseParticle(k): np.ascontiguousarray(v)
             for k, v in raw_p4.items()}

    with tf.device('cpu'):
        ad = cal_angle_from_momentum(
            p4_tf, dg, center_mass=True, r_boost=True,
            random_z=True, align_ref='center_mass',
        )
    np_data = data_to_numpy(ad)

    # Extract all (alpha, beta) pairs from decay data
    ang_vals = []
    for dc_val in np_data.get('decay', {}).values():
        for dec_key, dec_val in dc_val.items():
            out0 = dec_key.outs[0]
            if out0 in dec_val and isinstance(dec_val[out0], dict) \
               and 'ang' in dec_val[out0]:
                a = float(np.atleast_1d(dec_val[out0]['ang'].get('alpha', 0))[0])
                b = float(np.atleast_1d(dec_val[out0]['ang'].get('beta', 0))[0])
                ang_vals.append((a, b))

    theta1 = np.zeros(n_ev)
    theta2 = np.zeros(n_ev)
    phi    = np.zeros(n_ev)

    # Likely order: [B→S1, B→S2, S1→d1, S2→d2] or [S1→d1, S2→d2]
    # Sub-decay angles are at indices 2/3 or 0/1
    if len(ang_vals) >= 4:
        i1, i2 = 2, 3
    elif len(ang_vals) >= 2:
        i1, i2 = 0, 1
    else:
        return theta1, theta2, phi

    if ang_vals[i1][1] != 0: theta1 = np.full(n_ev, ang_vals[i1][1])
    if ang_vals[i2][1] != 0: theta2 = np.full(n_ev, ang_vals[i2][1])
    p = ang_vals[i1][0] - ang_vals[i2][0]
    phi = np.full(n_ev, np.arctan2(np.sin(p), np.cos(p)))

    return theta1, theta2, phi


# =====================================================================
# Main conversion
# =====================================================================

def convert(config_path, out_dir='converted',
            n_data_max=None, n_phsp_max=None):
    """Read config, load data, convert, save .npz."""

    files = get_data_files(config_path)

    (finals_list, top_name, dat_order, res_daughters,
     q_entries, res_name_to_bwall, n_m0_base, q_stride_base,
     perms, inter_tree, pw_list) = parse_structure(config_path)

    n_perm = len(perms)
    print(f"n_m0={n_m0_base}*{n_perm}={n_m0_base*n_perm}, "
          f"n_q={q_stride_base}*{n_perm}={q_stride_base*n_perm}")

    # Build DecayGroup for tf_pwa
    with open(config_path) as f:
        ycfg = yaml.safe_load(f)
    dg = build_decay_group(ycfg)

    # ---- Process data/phsp ----
    def process(p4_files, n_max, label):
        if not p4_files:
            print(f"  No {label} files")
            return None, None, None, 0

        all_p4 = None
        for f in p4_files:
            d = load_momenta(f, dat_order)
            if all_p4 is None:
                all_p4 = d
            else:
                for k in d:
                    all_p4[k] = np.concatenate([all_p4[k], d[k]])

        n = len(all_p4[list(all_p4)[0]])
        if n_max and n > n_max:
            for k in all_p4:
                all_p4[k] = all_p4[k][:n]
            n = n_max

        print(f"  {label}: {n} events, {n_perm} permutations...")
        t0 = time.time()
        m_blocks, q_blocks = [], []
        angles_0 = None

        for pi, swap in enumerate(perms):
            t1 = time.time()
            sw = dict(all_p4)
            for src, dst in swap.items():
                if src in sw and dst in sw:
                    sw[src], sw[dst] = sw[dst], sw[src]

            m_blocks.append(compute_mass_block(
                sw, inter_tree, top_name, finals_list,
                res_daughters, res_name_to_bwall, n_m0_base))
            q_blocks.append(compute_q_block(
                sw, inter_tree, top_name, finals_list,
                res_daughters, q_entries, q_stride_base))

            if angles_0 is None:
                th1, th2, ph = compute_angles_tfpwa(sw, dg, dat_order)
                angles_0 = np.column_stack([th1, th2, ph])

            print(f"    perm {pi}: {time.time()-t1:.1f}s"
                  f" ({n/max(time.time()-t1,0.01):.0f} ev/s)")

        dt = time.time() - t0
        mass   = np.concatenate(m_blocks, axis=1)
        q      = np.concatenate(q_blocks, axis=1)
        angles = angles_0
        print(f"    total: {dt:.1f}s  ({n/max(dt,0.01):.0f} ev/s)")
        return mass, q, angles, n

    m_d, q_d, a_d, nd  = process(files['p4_data'],  n_data_max, 'signal')
    m_p, q_p, a_p, np_ = process(files['p4_phsp'], n_phsp_max, 'phsp')

    # ---- Aux arrays ----
    def load_aux(paths, n, default=0.):
        if not paths:
            return np.full(n, default, dtype=np.float64)
        d = np.load(paths[0]).astype(np.float64)
        return d[:n] if len(d) > n else d

    dt    = load_aux(files['time_data'], nd, 1.)
    dtag  = load_aux(files['tag_data'],  nd, 1.)
    deta  = load_aux(files['eta_data'],  nd, 0.5)
    dbraw = load_aux(files['bkg_data'],  nd, 0.)

    pt    = load_aux(files['time_phsp'], np_, 0.)
    ptag  = load_aux(files['tag_phsp'],  np_, 1.)
    peta  = load_aux(files['eta_phsp'],  np_, 0.5)
    pw    = load_aux(files['weight_phsp'], np_, 1.)
    pbraw = load_aux(files['bkg_phsp'],   np_, 0.)

    bg_frac = files['bg_frac']

    # ---- frac: flavor-tag fraction ----
    #   tag == 0   -> 0.5  (untagged)
    #   tag >  0   -> 1 - eta  (B)
    #   tag <  0   -> eta      (Bbar)
    dfrac = np.where(dtag == 0, 0.5,
                     np.where(dtag > 0, 1.0 - deta, deta))
    pfrac = np.where(ptag == 0, 0.5,
                     np.where(ptag > 0, 1.0 - peta, peta))

    # ---- bkg: normalized background ----
    # Nb = average background over phsp
    if np_ > 0:
        Nb = np.sum(pbraw * pw) / np_
    else:
        Nb = 1.0
    # bkg = bkg_raw * bg_frac / (1 - bg_frac) / Nb
    scale = bg_frac / max(1 - bg_frac, 1e-10) / max(Nb, 1e-30)
    db = dbraw * scale
    pb = pbraw * scale  # also compute for phsp (useful for validation)

    print(f"\nAux computed:")
    print(f"  bg_frac={bg_frac}, Nb={Nb:.6e}, scale={scale:.6e}")
    print(f"  bkg range: data [{db.min():.4e}, {db.max():.4e}], "
          f"phsp [{pb.min():.4e}, {pb.max():.4e}]")

    # ---- Save ----
    os.makedirs(out_dir, exist_ok=True)
    data_npz = os.path.join(out_dir, 'data_arrays.npz')
    np.savez(data_npz,
             mass=m_d, q=q_d, angles=a_d,
             time=dt, frac=dfrac, bkg=db)
    phsp_npz = os.path.join(out_dir, 'phsp_arrays.npz')
    np.savez(phsp_npz,
             mass=m_p, q=q_p, angles=a_p,
             time=pt, frac=pfrac, weight=pw, bkg=pb)
    meta = dict(n_m0=n_m0_base*n_perm, n_q=q_stride_base*n_perm,
                n_angles=3, n_data=nd, n_phsp=np_,
                bg_frac=bg_frac, Nb=float(Nb),
                config_path=os.path.abspath(config_path))
    with open(os.path.join(out_dir, 'convert_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved:")
    print(f"  {data_npz}  ({nd} events)  {m_d.shape if nd else '-'}")
    print(f"  {phsp_npz}  ({np_} events) {m_p.shape if np_ else '-'}")
    return data_npz, phsp_npz


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--out', default='converted')
    p.add_argument('--data-max', type=int)
    p.add_argument('--phsp-max', type=int)
    args = p.parse_args()
    convert(config_path=args.config, out_dir=args.out,
            n_data_max=args.data_max, n_phsp_max=args.phsp_max)

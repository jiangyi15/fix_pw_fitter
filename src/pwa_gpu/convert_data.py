#!/usr/bin/env python3
"""
Convert raw 4-momenta to pwa_gpu kernel arrays.

Uses tf_pwa's cal_angle_from_momentum for helicity angles.
Masses and q-values computed with pure numpy.

Saves .npz files for fast loading without tf_pwa.

Usage:
    python -m pwa_gpu.convert_data config_angle.yml data_sig.npy --out out/ \\
        --phsp phsp_sig.npy --bg data_sig_bg_model.npy \\
        --data-time data_sig_time.npy
"""

import os, sys, yaml, json, time, itertools
import numpy as np
from collections import OrderedDict


def load_momenta(filename, order):
    p4 = np.load(filename).astype(np.float64)
    if p4.ndim == 3 and p4.shape[1] == len(order):
        return {name: p4[:, i, :] for i, name in enumerate(order)}
    raise ValueError(f"Shape {p4.shape}, expected (n,{len(order)},4)")


# ===== Lorentz helpers =====
def inv_mass(p):
    m2 = p[:,0]*p[:,0] - p[:,1]*p[:,1] - p[:,2]*p[:,2] - p[:,3]*p[:,3]
    return np.sqrt(np.clip(m2, 0., None))

def decay_q(m0, m1, m2):
    m12s, m12d = m1+m2, m1-m2
    p2 = (m0-m12s)*(m0+m12s)*(m0-m12d)*(m0+m12d)
    return np.sqrt(np.clip(p2, 0., None)) / np.clip(2.*m0, 1e-15, None)


def build_decay_group(ycfg):
    """Build tf_pwa DecayGroup from config (no model loading)."""
    from tf_pwa.particle import BaseParticle, BaseDecay, DecayChain, DecayGroup
    
    finals = set(ycfg.get('particle', {}).get('$finals', []))
    top = ycfg.get('particle', {}).get('$top', 'B')
    decay_sec = ycfg.get('decay', {})
    b_decays = list(decay_sec.get(top, []))
    
    def collect_chain(b_line):
        """Build a single decay chain from B decay line."""
        b_daus = [x for x in b_line if isinstance(x, str)]
        BP, BD = BaseParticle(top), [BaseParticle(d) for d in b_daus]
        steps = [BaseDecay(BP, BD)]
        
        def add_decays(parent_name, parent_bp):
            if parent_name in finals:
                return
            if parent_name not in decay_sec:
                return
            raw = decay_sec[parent_name]
            daus = [x for x in raw if isinstance(x, str)]
            if not daus:
                return
            d_bp = [BaseParticle(d) for d in daus]
            steps.append(BaseDecay(parent_bp, d_bp))
            for dn, dp in zip(daus, d_bp):
                add_decays(dn, dp)
        
        for d, bp in zip(b_daus, BD):
            add_decays(d, bp)
        
        return DecayChain(steps)
    
    chains = [collect_chain(bl) for bl in b_decays if any(isinstance(x, str) for x in bl)]
    return DecayGroup(chains), top, list(finals)


def convert(config_path, data_files, phsp_files, out_dir,
            data_time=None, phsp_time=None, phsp_weight=None, bg_file=None,
            data_tag=None, data_eta=None, phsp_tag=None, phsp_eta=None,
            n_data_max=None, n_phsp_max=None, batch=50000):
    """Convert 4-momenta → pwa_gpu arrays. Saves .npz."""
    
    from pwa_gpu.parse_config import parse_config, get_finals, get_top, get_mass_key
    
    with open(config_path) as f:
        ycfg = yaml.safe_load(f)
    
    finals_list = get_finals(ycfg)
    finals_set = set(finals_list)
    top_name = get_top(ycfg)
    dat_order = ycfg.get('data', {}).get('dat_order', finals_list)
    ident_particles = ycfg.get('data', {}).get('identical_particles', [])
    cp_particles = ycfg.get('data', {}).get('cp_particles', [])
    
    cfg, pw_list, kw_list = parse_config(config_path)
    
    # ===== Permutations =====
    perms = []
    if ident_particles:
        groups = [list(itertools.permutations(g)) for g in ident_particles]
        for combo in itertools.product(*groups):
            swap = {}
            for c, g in zip(combo, ident_particles):
                for ci, pi in zip(c, g):
                    if ci != pi:
                        swap[ci] = pi
            perms.append(swap)
    if not perms:
        perms = [{}]
    
    n_perm = len(perms)
    
    # ===== Column mapping =====
    res_name_to_bwall = OrderedDict()
    bwall_key_to_idx = OrderedDict()
    for pw in pw_list:
        for res in pw.resonances:
            if res.name in res_name_to_bwall:
                continue
            mk = get_mass_key(res.name, pw, finals_set, ident_groups=None)
            key = (mk, res.mass, res.width, res.model)
            if key not in bwall_key_to_idx:
                bwall_key_to_idx[key] = len(bwall_key_to_idx)
            res_name_to_bwall[res.name] = bwall_key_to_idx[key]
    n_m0_base = len(bwall_key_to_idx)
    n_m0 = n_m0_base * n_perm
    
    q_entries = OrderedDict()
    for pw in pw_list:
        for step in pw.chain:
            if step.daughters:
                qkey = (step.parent, tuple(sorted(step.daughters)))
                if qkey not in q_entries:
                    q_entries[qkey] = len(q_entries)
    q_stride_base = len(q_entries)
    q_stride = q_stride_base * n_perm
    
    print(f"n_m0={n_m0} ({n_m0_base}*{n_perm}), n_q={q_stride} ({q_stride_base}*{n_perm}), n_angles=3")

    # ===== Build resonance→final-daughters map =====
    particle_sec = ycfg.get('particle', {})
    decay_sec = ycfg.get('decay', {})
    
    def get_final_daus(name, visited=None):
        if visited is None:
            visited = set()
        if name in visited or name in finals_set:
            return [name] if name in finals_set else []
        visited.add(name)
        if name in decay_sec:
            raw = decay_sec[name]
            daus = [x for x in raw if isinstance(x, str)]
            if daus:
                result = []
                for d in daus:
                    result.extend(get_final_daus(d, visited))
                return result
        return []
    
    res_daughters = {}
    # Build from particle section: name → resonance mapping
    inter_of_res = {}
    for name, props in particle_sec.items():
        if isinstance(props, list):
            for p in props:
                if isinstance(p, str):
                    inter_of_res[p] = name
    
    for pw in pw_list:
        for res in pw.resonances:
            if res.name in res_daughters:
                continue
            inter = inter_of_res.get(res.name)
            src = inter or res.name
            fd = get_final_daus(src)
            if fd:
                res_daughters[res.name] = sorted(set(fd))
    
    # ===== Build intermediate tree =====
    inter_tree = {}
    for dname, dlist in decay_sec.items():
        daus = [x for x in dlist if isinstance(x, str)]
        if daus and dname != top_name:
            inter_tree[dname] = daus
    inter_tree[top_name] = finals_list
    
    # ===== All-particle 4-momentum computation =====
    def compute_all_p4(raw_p4):
        all_p4 = dict(raw_p4)
        done = set(raw_p4.keys())
        changed = True
        while changed:
            changed = False
            for parent, daus in inter_tree.items():
                if parent in done:
                    continue
                if all(d in done for d in daus):
                    all_p4[parent] = sum(all_p4[d] for d in daus)
                    done.add(parent)
                    changed = True
            # Top = sum of finals
            if top_name not in done and all(f in done for f in finals_list):
                all_p4[top_name] = sum(all_p4[f] for f in finals_list)
                done.add(top_name)
                changed = True
        # Add resonance 4-momenta
        for rname, fd in res_daughters.items():
            if rname not in done:
                all_p4[rname] = sum(all_p4[d] for d in fd)
                done.add(rname)
        return all_p4
    
    # ===== Process one permutation (with swapped momenta) =====
    def process_perm(swap_map):
        """Process one particle permutation. Returns (mass_block, q_block, angles)."""
        def apply_swap(raw_p4):
            if not swap_map:
                return raw_p4
            swapped = {}
            for k, v in raw_p4.items():
                target = swap_map.get(k, k)
                swapped[target] = v
            # Fill missing keys
            for k in raw_p4:
                if k not in swapped:
                    swapped[k] = raw_p4[k]
            return swapped
        
        def batch_func(raw_p4):
            n_ev = len(raw_p4[dat_order[0]])
            
            # 4-momenta
            all_p4 = compute_all_p4(raw_p4)
            
            # Masses
            m_block = np.zeros((n_ev, n_m0_base), dtype=np.float64)
            for rname, col in res_name_to_bwall.items():
                if rname in all_p4:
                    m_block[:, col] = inv_mass(all_p4[rname])
            
            # Q-values
            q_block = np.zeros((n_ev, q_stride_base), dtype=np.float64)
            for (parent, daughters), col in q_entries.items():
                def get_p4(name):
                    if name in all_p4:
                        return all_p4[name]
                    if name in res_daughters:
                        return sum(raw_p4[d] for d in res_daughters[name])
                    if name in inter_tree:
                        return sum(get_p4(d) for d in inter_tree[name])
                    return np.zeros((n_ev, 4))
                
                p_parent = get_p4(parent)
                p_d1 = get_p4(daughters[0])
                p_d2 = get_p4(daughters[1]) if len(daughters) > 1 else np.zeros((n_ev, 4))
                m0 = inv_mass(p_parent)
                m1 = inv_mass(p_d1)
                m2 = inv_mass(p_d2)
                q_block[:, col] = decay_q(m0, m1, m2)
            
            # Angles via tf_pwa
            from tf_pwa.particle import BaseParticle
            from tf_pwa.cal_angle import cal_angle_from_momentum
            from tf_pwa.data import data_to_numpy
            import tensorflow as tf
            
            p4_tf = {BaseParticle(k): np.ascontiguousarray(v) 
                     for k, v in raw_p4.items()}
            
            with tf.device('cpu'):
                angle_data = cal_angle_from_momentum(
                    p4_tf, dg, center_mass=True, r_boost=True,
                    random_z=True, align_ref='center_mass',
                )
            np_data = data_to_numpy(angle_data)
            
            theta1 = np.zeros(n_ev)
            theta2 = np.zeros(n_ev)
            phi = np.zeros(n_ev)
            
            if pw_list and len(pw_list[0].chain) >= 2:
                # Get parent names for the two sub-decays
                sub_parents = []
                for step in pw_list[0].chain[1:]:
                    if step.daughters:
                        sub_parents.append(step.parent)
                
                # Find angles in tf_pwa data using composite names
                ang_vals = []
                for parent in sub_parents[:2]:
                    for dc_val in np_data.get('decay', {}).values():
                        for dec_key, dec_val in dc_val.items():
                            if str(dec_key.core) == parent:
                                # Try finding under topology composite name
                                pass
                            out0 = dec_key.outs[0]
                            if out0 in dec_val and 'ang' in dec_val[out0]:
                                ang = dec_val[out0]['ang']
                                ang_vals.append((
                                    float(ang.get('alpha', 0)[0]),
                                    float(ang.get('beta', 0)[0])
                                ))
                                break
                        if len(ang_vals) > len(ang_vals) if ang_vals else 0:
                            break
                
                # Fallback: extract from first chain's decay data
                if not ang_vals and np_data.get('decay'):
                    for dc_val in np_data['decay'].values():
                        for dec_key, dec_val in dc_val.items():
                            for out_key, out_val in dec_val.items():
                                if isinstance(out_val, dict) and 'ang' in out_val:
                                    ang = out_val['ang']
                                    ang_vals.append((
                                        float(np.atleast_1d(ang.get('alpha', 0))[0]),
                                        float(np.atleast_1d(ang.get('beta', 0))[0])
                                    ))
                                    if len(ang_vals) >= 2:
                                        break
                        if len(ang_vals) >= 2:
                            break
                
                if len(ang_vals) >= 1 and ang_vals[0][1] != 0:
                    theta1 = np.full(n_ev, ang_vals[0][1])
                if len(ang_vals) >= 2 and ang_vals[1][1] != 0:
                    theta2 = np.full(n_ev, ang_vals[1][1])
                if len(ang_vals) >= 2:
                    p = ang_vals[0][0] - ang_vals[1][0]
                    phi = np.full(n_ev, np.arctan2(np.sin(p), np.cos(p)))
            
            return m_block, q_block, np.column_stack([theta1, theta2, phi])
        
        return batch_func
    
    # Build DecayGroup once (needed for angle computation)
    from tf_pwa.particle import BaseParticle, BaseDecay, DecayChain, DecayGroup
    from tf_pwa.cal_angle import cal_angle_from_momentum
    from tf_pwa.data import data_to_numpy
    import tensorflow as tf
    
    dg, _, _ = build_decay_group(ycfg)
    
    # ===== Load and process =====
    def load_and_process(files, n_max=None, aux_loader=None):
        all_p4 = None
        for f in files:
            d = load_momenta(f, dat_order)
            if all_p4 is None:
                all_p4 = d
            else:
                for k in d:
                    all_p4[k] = np.concatenate([all_p4[k], d[k]])
        n_total = len(all_p4[dat_order[0]])
        if n_max and n_total > n_max:
            for k in all_p4:
                all_p4[k] = all_p4[k][:n_max]
            n_total = n_max
        
        print(f"  {n_total} events, {len(perms)} permutations...")
        t0 = time.time()
        
        m_blocks, q_blocks, a_blocks = [], [], []
        for pi, swap in enumerate(perms):
            t1 = time.time()
            swapped = {k: v for k, v in all_p4.items()}
            for src, dst in swap.items():
                if src in swapped and dst in swapped:
                    swapped[src], swapped[dst] = swapped[dst], swapped[src]
            
            proc = process_perm(swap)
            m, q, a = proc(swapped)
            m_blocks.append(m)
            q_blocks.append(q)
            a_blocks.append(a)
            dt = time.time() - t1
            print(f"    perm {pi}: {dt:.1f}s ({n_total/max(dt,0.01):.0f} ev/s)")
        
        mass = np.concatenate(m_blocks, axis=1)
        q = np.concatenate(q_blocks, axis=1)
        angles = np.concatenate(a_blocks if len(a_blocks[0].shape) == 2 else [a.reshape(-1,3) for a in a_blocks], axis=1)
        angles = angles[:, :3] if angles.shape[1] >= 3 else np.pad(angles, ((0,0), (0,3-angles.shape[1])))
        
        print(f"    total: {time.time()-t0:.1f}s")
        return mass, q, angles[:, :3], n_total
    
    m_d, q_d, a_d, n_data = load_and_process(data_files, n_data_max)
    m_p, q_p, a_p, n_phsp = load_and_process(phsp_files, n_phsp_max)
    
    # ===== Aux arrays =====
    def load_arr(p, n, default=1.0):
        if p is None:
            return np.full(n, default, dtype=np.float64)
        d = np.load(p).astype(np.float64)
        return d[:n] if len(d) > n else d
    
    dt = load_arr(data_time, n_data, 1.0) if data_time else np.ones(n_data)
    pt = load_arr(phsp_time, n_phsp, 0.0) if phsp_time else np.zeros(n_phsp)
    pw = load_arr(phsp_weight, n_phsp, 1.0) if phsp_weight else np.ones(n_phsp)
    db = load_arr(bg_file, n_data, 0.0) if bg_file else np.zeros(n_data)
    dtag = load_arr(data_tag, n_data, 1.0) if data_tag else np.ones(n_data)
    deta = load_arr(data_eta, n_data, 0.5) if data_eta else np.full(n_data, 0.5)
    ptag = load_arr(phsp_tag, n_phsp, 1.0) if phsp_tag else np.ones(n_phsp)
    peta = load_arr(phsp_eta, n_phsp, 0.5) if phsp_eta else np.full(n_phsp, 0.5)
    
    # ===== Save =====
    os.makedirs(out_dir, exist_ok=True)
    data_npz = os.path.join(out_dir, 'data_arrays.npz')
    np.savez(data_npz, mass=m_d, q=q_d, angles=a_d,
             time=dt, tag=dtag, eta=deta, bkg=db)
    phsp_npz = os.path.join(out_dir, 'phsp_arrays.npz')
    np.savez(phsp_npz, mass=m_p, q=q_p, angles=a_p,
             time=pt, tag=ptag, eta=peta, weight=pw)
    meta = {'n_m0': n_m0, 'n_q': q_stride, 'n_angles': 3,
            'n_data': n_data, 'n_phsp': n_phsp}
    with open(os.path.join(out_dir, 'convert_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved: {data_npz} ({n_data}), {phsp_npz} ({n_phsp})")
    return data_npz, phsp_npz


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('data', nargs='+')
    p.add_argument('--phsp', nargs='*', default=[])
    p.add_argument('--out', default='converted')
    p.add_argument('--data-time'); p.add_argument('--phsp-time')
    p.add_argument('--phsp-weight'); p.add_argument('--bg')
    p.add_argument('--data-tag'); p.add_argument('--data-eta')
    p.add_argument('--phsp-tag'); p.add_argument('--phsp-eta')
    p.add_argument('--data-max', type=int); p.add_argument('--phsp-max', type=int)
    p.add_argument('--batch', type=int, default=50000)
    args = p.parse_args()
    convert(**vars(args))

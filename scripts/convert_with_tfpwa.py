#!/usr/bin/env python3
"""
Convert data via tf_pwa's config loader — uses config.get_data(name).

Reads data file paths from config.yml, loads/preprocesses via
ConfigLoader, extracts masses/q/angles from the preprocessed data dict
using tensor.numpy(), saves .npz for the GPU pipeline.

Usage:
    python convert_with_tfpwa.py config_angle.yml --out converted/
"""

import os, sys, time, yaml, json, itertools
import numpy as np
from collections import OrderedDict


def convert(config_path, out_dir='converted',
            n_data_max=None, n_phsp_max=None):
    """Load via config.get_data(), extract & save as .npz."""
    import importlib

    config_dir = os.path.dirname(os.path.abspath(config_path))
    sys.path.insert(0, config_dir)
    sys.path.insert(0, '/mnt/e/github/tf-pwa')

    # ---- Register strip preprocessor if needed ----
    try:
        import create_simple_angles
        importlib.reload(create_simple_angles)
    except ImportError:
        from tf_pwa.amp.preprocess import register_preprocessor, BasePreProcessor
        @register_preprocessor("strip")
        class _Strip(BasePreProcessor):
            def call(self, x, **kwargs):
                return x

    from tf_pwa.config_loader import ConfigLoader
    from tf_pwa.data import data_to_numpy, data_shape, data_split, data_merge, data_index
    import tensorflow as tf

    # ---- Parse YAML & pwa_gpu structure ----
    with open(config_path) as f:
        ycfg = yaml.safe_load(f)

    from pwa_gpu.parse_config import (
        parse_config, get_finals, get_top, get_mass_key
    )
    finals_list = get_finals(ycfg)
    finals_set  = set(finals_list)
    top_name    = get_top(ycfg)
    dat_order   = ycfg.get('data', {}).get('dat_order', finals_list)
    ident_particles = ycfg.get('data', {}).get('identical_particles', [])
    bg_frac     = ycfg.get('data', {}).get('bg_frac', 0.0)
    data_sec    = ycfg.get('data', {})

    cfg, pw_list, kw_list = parse_config(config_path)

    # ---- Permutations from config ----
    perms = [{}]
    if ident_particles:
        groups = [list(itertools.permutations(g)) for g in ident_particles]
        for combo in itertools.product(*groups):
            swap = {}
            for c, g in zip(combo, ident_particles):
                for ci, pi in zip(c, g):
                    if ci != pi: swap[ci] = pi
            if swap: perms.append(swap)
    n_perm = len(perms)

    # ---- Column mapping (matching build_kernel_config) ----
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

    q_entries = OrderedDict()
    for pw in pw_list:
        for step in pw.chain:
            if step.daughters:
                qkey = (step.parent, tuple(sorted(step.daughters)))
                if qkey not in q_entries:
                    q_entries[qkey] = len(q_entries)
    q_stride_base = len(q_entries)

    print(f"n_m0={n_m0_base}×{n_perm}={n_m0_base*n_perm}, "
          f"n_q={q_stride_base}×{n_perm}={q_stride_base*n_perm}", flush=True)

    # ---- Resonance → final-daughters map ----
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
                if isinstance(p, str): inter_of_res[p] = name

    res_daughters = {}
    for pw in pw_list:
        for res in pw.resonances:
            if res.name in res_daughters: continue
            inter = inter_of_res.get(res.name)
            src = inter or res.name
            fd = get_final_daus(src)
            if fd: res_daughters[res.name] = sorted(set(fd))

    # ---- Build tf_pwa name -> our resonance map ----
    # tf_pwa composites: "(d1, d2, ...)" with sorted daughters
    tfname_to_res = {}
    for rname, fd in res_daughters.items():
        tfname_to_res["(" + ", ".join(sorted(fd)) + ")"] = rname
    for f in finals_list:
        tfname_to_res[f] = f
    tfname_to_res[top_name] = top_name

    # ---- Initialize ConfigLoader (this loads data) ----
    print("Initializing ConfigLoader...", flush=True)
    t0 = time.time()
    # Use CPU for large datasets
    with tf.device('cpu'):
        config = ConfigLoader(config_path)
    print(f"  ready ({time.time()-t0:.1f}s)", flush=True)

    # ---- Extract arrays from tf_pwa data dict ----
    def extract_arrays(np_data, n_ev):
        """Return (mass_block, q_block, angles_block) from tf_pwa data."""

        # mass_flat: match composite particle keys
        m_block = np.zeros((n_ev, n_m0_base), dtype=np.float64)
        for pk in np_data.get('particle', {}):
            pname = str(pk)
            if pname in tfname_to_res:
                rname = tfname_to_res[pname]
                if rname in res_name_to_bwall:
                    col = res_name_to_bwall[rname]
                    m_block[:, col] = np.asarray(
                        np_data['particle'][pk]['m']).ravel()

        # q_flat: match by daughter set
        q_block = np.zeros((n_ev, q_stride_base), dtype=np.float64)
        for (parent, daughters), col in q_entries.items():
            dau_set = frozenset(daughters)
            for dc_val in np_data.get('decay', {}).values():
                for dec_key, dec_val in dc_val.items():
                    if frozenset(str(o) for o in dec_key.outs) == dau_set \
                       and '|q|2' in dec_val:
                        q2 = np.asarray(dec_val['|q|2']).ravel()
                        q_block[:, col] = np.sqrt(np.clip(q2, 0, None))
                        break
                else: continue
                break

        # angles: helicity angles from sub-decay daughters
        ab_pairs = []
        for dc_val in np_data.get('decay', {}).values():
            for dec_key, dec_val in dc_val.items():
                out0 = dec_key.outs[0]
                if out0 in dec_val and isinstance(dec_val[out0], dict) \
                   and 'ang' in dec_val[out0]:
                    ang = dec_val[out0]['ang']
                    a = np.asarray(ang['alpha']).ravel()
                    b = np.asarray(ang['beta']).ravel()
                    if a.size == n_ev and b.size == n_ev:
                        ab_pairs.append((a, b))

        theta1 = np.zeros(n_ev)
        theta2 = np.zeros(n_ev)
        phi    = np.zeros(n_ev)

        # Sub-decay angles are usually at indices 2/3 (after B→S1, B→S2)
        if len(ab_pairs) >= 4:
            i1, i2 = 2, 3
        elif len(ab_pairs) >= 2:
            i1, i2 = 0, 1
        else:
            return m_block, q_block, np.zeros((n_ev, 3))

        theta1 = ab_pairs[i1][1]
        theta2 = ab_pairs[i2][1]
        p_diff = ab_pairs[i1][0] - ab_pairs[i2][0]
        phi    = np.arctan2(np.sin(p_diff), np.cos(p_diff))

        return m_block, q_block, np.column_stack([theta1, theta2, phi])

    # ---- Load data via config.get_data() ----
    def load_from_config(name, n_max):
        """Use config.get_data() to load and preprocess."""
        print(f"\nLoading '{name}' via config.get_data()...", flush=True)
        t0 = time.time()
        data_list = config.get_data(name)
        print(f"  got {len(data_list)} files, "
              f"converting to numpy ({time.time()-t0:.1f}s)", flush=True)

        n_total = sum(data_shape(d) for d in data_list)
        if n_max and n_total > n_max:
            # Take first n_max events
            all_splits = []
            remaining = n_max
            for d in data_list:
                n = data_shape(d)
                if n <= remaining:
                    all_splits.append(d)
                    remaining -= n
                else:
                    all_splits.extend(data_split(d)[:remaining])
                    break
            data_merged = data_merge(*all_splits) if len(all_splits) > 1 else all_splits[0]
            n_total = n_max
        else:
            data_merged = data_merge(*data_list) if len(data_list) > 1 else data_list[0]

        # Convert to numpy (this calls .numpy() on all tensors)
        print(f"  converting to numpy ({n_total} events)...", flush=True)
        t1 = time.time()
        np_data = data_to_numpy(data_merged)
        print(f"  numpy conversion: {time.time()-t1:.1f}s", flush=True)

        print(f"  extracting arrays ({time.time()-t0:.1f}s)...", flush=True)
        m, q, a = extract_arrays(np_data, n_total)

        # ---- Extract aux arrays via data_index ----
        aux = {}
        for aux_key in ('time', 'tag', 'eta1', 'bg_value', 'weight'):
            try:
                idx = data_index(np_data, (aux_key,))
                arr = np.asarray(idx).ravel()
                if len(arr) > n_total: arr = arr[:n_total]
                aux[aux_key] = arr
            except Exception:
                aux[aux_key] = None

        print(f"  done ({time.time()-t0:.1f}s)", flush=True)
        return m, q, a, n_total, aux

    # Load signal
    m_d, q_d, a_d, nd, aux_d = load_from_config('data', n_data_max)
    # Load phsp
    m_p, q_p, a_p, np_, aux_p = load_from_config('phsp', n_phsp_max)

    # ---- Handle id_swap and cp_swap also ----
    # For identical particles, each permutation gives the same physical
    # masses/q/angles but swapped columns. We replicate the blocks.

    def replicate_blocks(m, q, a):
        """Replicate arrays for n_perm identical blocks."""
        if m is None: return None, None, None
        return (np.tile(m, (1, n_perm)),
                np.tile(q, (1, n_perm)),
                a)  # angles same for all perms

    m_d, q_d, a_d = replicate_blocks(m_d, q_d, a_d)
    m_p, q_p, a_p = replicate_blocks(m_p, q_p, a_p)

    # ---- Assemble aux arrays ----
    dt    = np.asarray(aux_d.get('time', np.ones(nd))).ravel()[:nd] if nd else np.ones(0)
    dtag  = np.asarray(aux_d.get('tag',  np.ones(nd))).ravel()[:nd] if nd else np.ones(0)
    deta  = np.asarray(aux_d.get('eta1', np.full(nd, 0.5))).ravel()[:nd] if nd else np.ones(0)
    dbraw = np.asarray(aux_d.get('bg_value', np.zeros(nd))).ravel()[:nd] if nd else np.zeros(0)

    pt    = np.asarray(aux_p.get('time', np.zeros(np_))).ravel()[:np_] if np_ else np.zeros(0)
    ptag  = np.asarray(aux_p.get('tag',  np.ones(np_))).ravel()[:np_] if np_ else np.ones(0)
    peta  = np.asarray(aux_p.get('eta1', np.full(np_, 0.5))).ravel()[:np_] if np_ else np.ones(0)
    pw    = np.asarray(aux_p.get('weight', np.ones(np_))).ravel()[:np_] if np_ else np.ones(0)
    pbraw = np.asarray(aux_p.get('bg_value', np.zeros(np_))).ravel()[:np_] if np_ else np.zeros(0)

    # ---- frac ----
    dfrac = np.where(dtag == 0, 0.5,
                     np.where(dtag > 0, 1.0 - deta, deta))
    pfrac = np.where(ptag == 0, 0.5,
                     np.where(ptag > 0, 1.0 - peta, peta))

    # ---- bkg normalization ----
    Nb = np.sum(pbraw * pw) / max(np_, 1) if np_ > 0 else 1.0
    scale = bg_frac / max(1 - bg_frac, 1e-10) / max(Nb, 1e-30)
    db = dbraw * scale
    pb = pbraw * scale

    print(f"\nAux: bg_frac={bg_frac}, Nb={Nb:.6e}, scale={scale:.6e}",
          flush=True)
    print(f"  bkg: data [{db.min():.4e},{db.max():.4e}]  "
          f"phsp [{pb.min():.4e},{pb.max():.4e}]", flush=True)

    # ---- Save ----
    os.makedirs(out_dir, exist_ok=True)
    data_npz = os.path.join(out_dir, 'data_arrays.npz')
    np.savez(data_npz, mass=m_d, q=q_d, angles=a_d,
             time=dt, frac=dfrac, bkg=db)
    phsp_npz = os.path.join(out_dir, 'phsp_arrays.npz')
    np.savez(phsp_npz, mass=m_p, q=q_p, angles=a_p,
             time=pt, frac=pfrac, weight=pw, bkg=pb)
    meta = dict(n_m0=n_m0_base*n_perm, n_q=q_stride_base*n_perm,
                n_angles=3, n_data=nd, n_phsp=np_,
                bg_frac=bg_frac, Nb=float(Nb),
                config_path=os.path.abspath(config_path))
    with open(os.path.join(out_dir, 'convert_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: {data_npz} ({nd})  {phsp_npz} ({np_})", flush=True)
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

#!/usr/bin/env python3
"""
Convert raw 4‑momenta → pwa_gpu arrays via tf_pwa.

Uses ConfigLoader for model init (extra_amp, strip preprocessor),
then loads raw .npy 4‑momenta in batches and calls
cal_angle_from_momentum + data_to_numpy per batch to avoid OOM.

Usage:
    python convert_with_tfpwa.py config_angle.yml --out converted/
"""

import os, sys, time, yaml, json, itertools
import numpy as np
from collections import OrderedDict


def convert(config_path, out_dir='converted',
            n_data_max=None, n_phsp_max=None, batch_size=10000):
    """Load 4‑momenta in batches → extract arrays → save .npz."""

    config_dir = os.path.dirname(os.path.abspath(config_path))
    sys.path.insert(0, config_dir)
    sys.path.insert(0, '/mnt/e/github/tf-pwa')

    import extra_amp  # registers gls_cpv_aabar etc.
    from tf_pwa.amp.preprocess import register_preprocessor, BasePreProcessor
    @register_preprocessor("strip")
    class _StripStub(BasePreProcessor):
        def call(self, x, **kwargs): return x

    from tf_pwa.config_loader import ConfigLoader
    from tf_pwa.cal_angle import cal_angle_from_momentum
    from tf_pwa.data import data_to_numpy, data_shape
    from tf_pwa.particle import BaseParticle
    import tensorflow as tf

    # ---- Parse YAML & structure ----
    with open(config_path) as f:
        ycfg = yaml.safe_load(f)

    from pwa_gpu.parse_config import (
        parse_config, get_finals, get_top, get_mass_key
    )
    finals_list = get_finals(ycfg); finals_set = set(finals_list)
    top_name = get_top(ycfg)
    dat_order = ycfg.get('data', {}).get('dat_order', finals_list)
    ident_particles = ycfg.get('data', {}).get('identical_particles', [])
    bg_frac = ycfg.get('data', {}).get('bg_frac', 0.0)
    data_sec = ycfg.get('data', {})

    cfg, pw_list, kw_list = parse_config(config_path)

    # ---- Permutations ----
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

    # ---- Column mapping ----
    res_name_to_bwall = OrderedDict()
    for pw in pw_list:
        for res in pw.resonances:
            if res.name in res_name_to_bwall: continue
            mk = get_mass_key(res.name, pw, finals_set, ident_groups=None)
            res_name_to_bwall[res.name] = len(res_name_to_bwall)
    n_m0_base = len(res_name_to_bwall)

    q_entries = OrderedDict()
    for pw in pw_list:
        for step in pw.chain:
            if step.daughters:
                q_entries[(step.parent, tuple(sorted(step.daughters)))] = \
                    len(q_entries)
    q_stride_base = len(q_entries)

    print(f"n_m0={n_m0_base}*{n_perm}={n_m0_base*n_perm}, "
          f"n_q={q_stride_base}*{n_perm}={q_stride_base*n_perm}", flush=True)

    # ---- Resonance → final-daughters ----
    def get_final_daus(name, visited=None):
        if visited is None: visited = set()
        if name in visited or name in finals_set:
            return [name] if name in finals_set else []
        visited.add(name)
        daus = [x for x in ycfg['decay'].get(name, []) if isinstance(x, str)]
        if daus:
            out = []
            for d in daus: out.extend(get_final_daus(d, visited))
            return out
        return []

    inter_of_res = {}
    for name, props in ycfg.get('particle', {}).items():
        if isinstance(props, list):
            for p in props:
                if isinstance(p, str): inter_of_res[p] = name

    res_daughters = {}
    for pw in pw_list:
        for res in pw.resonances:
            if res.name in res_daughters: continue
            inter = inter_of_res.get(res.name)
            fd = get_final_daus(inter or res.name)
            if fd: res_daughters[res.name] = sorted(set(fd))

    # ---- tf_pwa composite name → our resonance ----
    tfname_to_res = {}
    for rname, fd in res_daughters.items():
        tfname_to_res["(" + ", ".join(sorted(fd)) + ")"] = rname
    for f in finals_list: tfname_to_res[f] = f
    tfname_to_res[top_name] = top_name

    # ---- Initialize ConfigLoader & get decay_group ----
    print("Initializing ConfigLoader...", flush=True)
    t0 = time.time()
    config = ConfigLoader(config_path)
    decay_group = config.get_decay()
    print(f"  ready ({time.time()-t0:.1f}s)", flush=True)

    # ---- Extract arrays from a single batch ----
    def extract_from_batch(np_data, n_ev):
        """Return (mass_block, q_block, angles_block) from one batch."""
        m = np.zeros((n_ev, n_m0_base), dtype=np.float64)
        for pk in np_data.get('particle', {}):
            pname = str(pk)
            if pname in tfname_to_res:
                rn = tfname_to_res[pname]
                if rn in res_name_to_bwall:
                    m[:, res_name_to_bwall[rn]] = \
                        np.asarray(np_data['particle'][pk]['m']).ravel()

        q = np.zeros((n_ev, q_stride_base), dtype=np.float64)
        for (parent, daughters), col in q_entries.items():
            ds = frozenset(daughters)
            for dc in np_data.get('decay', {}).values():
                for dk, dv in dc.items():
                    if frozenset(str(o) for o in dk.outs) == ds \
                       and '|q|2' in dv:
                        q[:, col] = np.sqrt(np.clip(
                            np.asarray(dv['|q|2']).ravel(), 0, None))
                        break
                else: continue
                break

        ab = []
        for dc in np_data.get('decay', {}).values():
            for dk, dv in dc.items():
                o0 = dk.outs[0]
                if o0 in dv and isinstance(dv[o0], dict) and 'ang' in dv[o0]:
                    a = np.asarray(dv[o0]['ang']['alpha']).ravel()
                    b = np.asarray(dv[o0]['ang']['beta']).ravel()
                    if a.size == n_ev and b.size == n_ev:
                        ab.append((a, b))

        t1 = np.zeros(n_ev); t2 = np.zeros(n_ev); ph = np.zeros(n_ev)
        if len(ab) >= 4: i1, i2 = 2, 3
        elif len(ab) >= 2: i1, i2 = 0, 1
        else: return m, q, np.zeros((n_ev, 3))
        t1[:] = ab[i1][1]; t2[:] = ab[i2][1]
        p = ab[i1][0] - ab[i2][0]
        ph[:] = np.arctan2(np.sin(p), np.cos(p))
        return m, q, np.column_stack([t1, t2, ph])

    # ---- Load & process one dataset ----
    def process_dataset(field, n_max):
        """Load raw .npy files, batch through cal_angle_from_momentum."""
        flist = data_sec.get(field) or data_sec.get(
            field.replace('data','dataall'), [])
        if isinstance(flist, str): flist = [flist]

        p4_data = None
        for fn in flist:
            fn = os.path.normpath(os.path.join(config_dir, fn))
            p4 = np.load(fn).astype(np.float64)
            if p4.ndim == 3 and p4.shape[1] == len(dat_order):
                d = {n: p4[:, i, :] for i, n in enumerate(dat_order)}
            else:
                raise ValueError(f"Bad shape {p4.shape}")
            if p4_data is None: p4_data = d
            else:
                for k in d: p4_data[k] = np.concatenate([p4_data[k], d[k]])

        n_total = len(p4_data[dat_order[0]])
        if n_max and n_total > n_max:
            for k in p4_data: p4_data[k] = p4_data[k][:n_max]
            n_total = n_max

        n_perm = len(perms)
        print(f"\n{field}: {n_total} events, {n_perm} permutations, "
              f"batch={batch_size}...", flush=True)

        m_all, q_all, a_all = [], [], []
        # Aux accumulators for raw file data
        aux_raw = {k: [] for k in
                   (f'{field}_time', f'{field}_tag1', f'{field}_eta1',
                    f'{field}_bg_value', f'{field}_weight')}

        # Process in batches
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            t1 = time.time()

            batch_p4 = {k: v[start:end] for k, v in p4_data.items()}
            p4_tf = {BaseParticle(k): np.ascontiguousarray(v)
                     for k, v in batch_p4.items()}

            with tf.device('cpu'):
                result = cal_angle_from_momentum(
                    p4_tf, decay_group, center_mass=True, r_boost=True,
                    random_z=True, align_ref='center_mass',
                )
            np_batch = data_to_numpy(result)
            n_b = data_shape(result)

            m_b, q_b, a_b = extract_from_batch(np_batch, n_b)
            m_all.append(m_b); q_all.append(q_b); a_all.append(a_b)

            # Also accumulate aux from raw files (time, tag, etc.)
            # These are loaded separately below
            print(f"  [{start}:{end}] {time.time()-t1:.1f}s "
                  f"({n_b/max(time.time()-t1,0.01):.0f} ev/s)", flush=True)

        # Permutation replication: tile mass and q blocks
        mass   = np.tile(np.concatenate(m_all, axis=0), (1, n_perm))
        q_arr  = np.tile(np.concatenate(q_all, axis=0), (1, n_perm))
        angles = np.concatenate(a_all, axis=0)

        return mass, q_arr, angles, n_total

    # ---- Process signal & phsp ----
    data_files = {}
    for key, field in [('p4_data', 'data'), ('p4_phsp', 'phsp')]:
        flist = data_sec.get(field) or data_sec.get(
            field.replace('data','dataall'), [])
        if isinstance(flist, str): flist = [flist]
        data_files[key] = [os.path.normpath(os.path.join(config_dir, f))
                           for f in flist]

    m_d, q_d, a_d, nd = process_dataset('data', n_data_max)
    if data_files.get('p4_phsp'):
        m_p, q_p, a_p, np_ = process_dataset('phsp', n_phsp_max)
    else:
        m_p = q_p = a_p = None; np_ = 0

    # ---- Aux arrays from raw files ----
    def load_raw(key_fragment, n, default):
        path = data_sec.get(key_fragment)
        if not path: return np.full(n, default, dtype=np.float64)
        if isinstance(path, list): path = path[0]
        path = os.path.normpath(os.path.join(config_dir, path))
        d = np.load(path).astype(np.float64)
        return d[:n] if len(d) > n else d

    dt    = load_raw('data_time',    nd, 1.)
    dtag  = load_raw('data_tag1',    nd, 1.)
    deta  = load_raw('data_eta1',    nd, 0.5)
    dbraw = load_raw('data_bg_value', nd, 0.)

    pt    = load_raw('phsp_time',    np_, 0.) if np_ else np.ones(0)
    ptag  = load_raw('phsp_tag1',    np_, 1.) if np_ else np.ones(0)
    peta  = load_raw('phsp_eta1',    np_, 0.5) if np_ else np.ones(0)
    pw    = load_raw('phsp_weight',  np_, 1.) if np_ else np.ones(0)
    pbraw = load_raw('phsp_bg_value', np_, 0.) if np_ else np.zeros(0)

    # ---- frac ----
    dfrac = np.where(dtag == 0, 0.5,
                     np.where(dtag > 0, 1.0 - deta, deta))
    pfrac = np.where(ptag == 0, 0.5,
                     np.where(ptag > 0, 1.0 - peta, peta)) if np_ else np.ones(0)

    # ---- bkg ----
    Nb = np.sum(pbraw * pw) / max(np_, 1) if np_ > 0 else 1.0
    scale = bg_frac / max(1 - bg_frac, 1e-10) / max(Nb, 1e-30)
    db = dbraw * scale
    pb = pbraw * scale if np_ else np.zeros(0)

    print(f"\nAux: bg_frac={bg_frac}, Nb={Nb:.6e}, scale={scale:.6e}",
          flush=True)

    # ---- Save ----
    os.makedirs(out_dir, exist_ok=True)
    data_npz = os.path.join(out_dir, 'data_arrays.npz')
    np.savez(data_npz, mass=m_d, q=q_d, angles=a_d,
             time=dt, frac=dfrac, bkg=db)
    phsp_npz = os.path.join(out_dir, 'phsp_arrays.npz')
    if np_:
        np.savez(phsp_npz, mass=m_p, q=q_p, angles=a_p,
                 time=pt, frac=pfrac, weight=pw, bkg=pb)
    else:
        np.savez(phsp_npz, mass=np.zeros((0,1)), q=np.zeros((0,1)),
                 angles=np.zeros((0,3)))

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
    p.add_argument('--batch', type=int, default=10000)
    args = p.parse_args()
    convert(config_path=args.config, out_dir=args.out,
            n_data_max=args.data_max, n_phsp_max=args.phsp_max,
            batch_size=args.batch)

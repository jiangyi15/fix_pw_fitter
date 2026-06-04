#!/usr/bin/env python3
"""
Convert via config.get_data(name). Extract per-chain masses/q/angles.

Output: mass (n, 24, 2), q (n, 24, 3), angles (n, 24, 3)
24 topologies = 3 chains × 4 id perms × 2 CP.
"""
import os, sys, time, yaml, json, itertools
import numpy as np

def convert(config_path, out_dir='converted'):
    config_dir = os.path.dirname(os.path.abspath(config_path))
    sys.path.insert(0, config_dir); sys.path.insert(0, '/mnt/e/github/tf-pwa')
    import extra_amp
    from tf_pwa.amp.preprocess import register_preprocessor, BasePreProcessor
    @register_preprocessor("strip")
    class _S(BasePreProcessor):
        def call(self, x, **kwargs):
            d = self.strip(x)
            for k in d.get("id_swap",{}): d["id_swap"][k]=self.strip(d["id_swap"][k])
            if "cp_swap" in d:
                d["cp_swap"]=self.strip(d["cp_swap"])
                for k2 in d["cp_swap"].get("id_swap",{}): d["cp_swap"]["id_swap"][k2]=self.strip(d["cp_swap"]["id_swap"][k2])
            return d
        def strip(self, x):
            for k in x["particle"]: del x["particle"][k]["p"]
            for k in list(x.get("decay",{})):
                for k2 in list(x["decay"][k]):
                    dv=x["decay"][k][k2]; o0=k2.outs[0]
                    if o0 in dv and 'ang' in dv.get(o0,{}):
                        if 'gamma' in dv[o0]['ang']: del dv[o0]['ang']['gamma']
                        if len(str(k2.core).split(", "))==3 and 'alpha' in dv[o0]['ang']: del dv[o0]['ang']['alpha']
                    if k2.outs[1] in dv: dv[k2.outs[1]]={}
                    if k2.core==self.decay_struct.top and o0 in dv and 'ang' in dv.get(o0,{}): dv[o0]['ang']={}
                    if o0 in dv and 'aligned_angle' in dv.get(o0,{}): del dv[o0]['aligned_angle']
                    if '|q|2' in dv: del dv['|q|2']
            return x
    from tf_pwa.config_loader import ConfigLoader
    from tf_pwa.data import data_to_numpy, data_shape
    import tensorflow as tf
    with open(config_path) as f: ycfg=yaml.safe_load(f)
    from pwa_gpu.parse_config import parse_config, get_finals, get_top
    finals_list=get_finals(ycfg); finals_set=set(finals_list)
    top_name=get_top(ycfg); dat_order=ycfg.get('data',{}).get('dat_order',finals_list)
    data_sec=ycfg.get('data',{}); bg_frac=data_sec.get('bg_frac',0.)
    cfg, pw_list, _ = parse_config(config_path)

    # Init ConfigLoader
    print("Initializing ConfigLoader...",flush=True)
    t0=time.time()
    with tf.device('cpu'): config=ConfigLoader(config_path)
    dg=config.get_decay()
    print(f"  ready ({time.time()-t0:.1f}s)",flush=True)

    # Get topology structure (chains list)
    chains = dg.topology_structure()
    n_chain = len(chains)  # should be 3

    # ---- Extract 2 masses, 3 q, 3 angles from ONE data dict ----
    def extract_topo(np_data, ci, n_ev):
        """Return (2 masses, 3 q, 3 angles) for chain ci from np_data.
        
        Angles follow fixed_ls_chain.py convention:
          VV:   phi = ang1["alpha"] + ang2["alpha"], theta1=ang1["beta"], theta2=ang2["beta"]
          Cascade: phi = ang2["alpha"], theta1=ang1["beta"], theta2=ang2["beta"]
        """
        chain = chains[ci]
        # Find decay data matching this chain by core names
        dc_val = None
        for k in np_data.get('decay', {}):
            if len(k) == len(chain):
                ok = True
                for di in range(len(chain)):
                    if str(k[di].core) != str(chain[di].core): ok=False; break
                if ok: dc_val=np_data['decay'][k]; break
        if dc_val is None:
            return np.zeros((n_ev,2)), np.zeros((n_ev,3)), np.zeros((n_ev,3))

        all_m = {}
        for pk in np_data.get('particle',{}): all_m[str(pk)]=np.asarray(np_data['particle'][pk]['m']).ravel()

        # 2 masses: composites from non-B decays in order
        comps = []
        for dk,dv in dc_val.items():
            c=str(dk.core)
            if c!=top_name and c in all_m: comps.append(c)
        m0=all_m[comps[0]] if len(comps)>0 else np.zeros(n_ev)
        m1=all_m[comps[1]] if len(comps)>1 else np.zeros(n_ev)

        # 3 q-values: B→S1+S2, S1→a+b, S2→c+d
        def _q(core, out1, out2):
            if core in all_m and out1 in all_m and out2 in all_m:
                mo,m1q,m2q = all_m[core],all_m[out1],all_m[out2]
                ms,md=m1q+m2q,m1q-m2q
                p2=(mo-ms)*(mo+ms)*(mo-md)*(mo+md)
                return np.sqrt(np.clip(p2,0,None))/np.clip(2*mo,1e-15,None)
            return np.zeros(n_ev)
        q_vals=[]
        for dk,dv in dc_val.items():
            c=str(dk.core); os=[str(o) for o in dk.outs]
            if len(os)>=2: q_vals.append(_q(c, os[0], os[1]))
        while len(q_vals)<3: q_vals.append(np.zeros(n_ev))

        # 3 angles: theta1=ang1["beta"], theta2=ang2["beta"], phi per topology
        # ang1 = helicity angle of chain[1]'s first daughter
        # ang2 = helicity angle of chain[2]'s first daughter
        ang1=None; ang2=None
        for dk,dv in dc_val.items():
            c=str(dk.core)
            if c==top_name: continue
            o0=dk.outs[0]
            if o0 in dv and isinstance(dv[o0],dict) and 'ang' in dv[o0]:
                ang=dv[o0]['ang']
                if ang1 is None:
                    ang1=ang
                else:
                    ang2=ang
        t1=np.zeros(n_ev); t2=np.zeros(n_ev); ph=np.zeros(n_ev)
        if ang1 is not None:
            t1[:]=np.asarray(ang1.get('beta',np.zeros(n_ev))).ravel()
        if ang2 is not None:
            t2[:]=np.asarray(ang2.get('beta',np.zeros(n_ev))).ravel()
        # Determine phi from topology:
        # VV: "(pim1, pip1)+(pim2, pip2)" → phi = ang1["alpha"] + ang2["alpha"]
        # Cascade: "(pim1, pip1, pip2)" or "(pim1, pim2, pip1)" → phi = ang2["alpha"]
        chain_str=str(chains[ci])
        is_vv = "(pim1, pip1)+(pim2, pip2)" in chain_str
        if is_vv:
            if ang1 is not None and ang2 is not None:
                a1=np.asarray(ang1.get('alpha',np.zeros(n_ev))).ravel()
                a2=np.asarray(ang2.get('alpha',np.zeros(n_ev))).ravel()
                ph[:]=a1+a2
        else:
            if ang2 is not None:
                ph[:]=np.asarray(ang2.get('alpha',np.zeros(n_ev))).ravel()

        return np.column_stack([m0,m1]), np.column_stack(q_vals[:3]), np.column_stack([t1,t2,ph])

    # ---- Process data ----
    def process_field(field):
        flist=data_sec.get(field) or data_sec.get(field.replace('data','dataall'),[])
        if isinstance(flist,str): flist=[flist]
        if not flist: print(f"  No {field}"); return None,None,None,0
        print(f"  Loading '{field}'...",flush=True)
        t0=time.time()
        with tf.device('cpu'): raw=config.get_data(field)[0]
        n_ev=data_shape(raw)
        print(f"    loaded {n_ev} events ({time.time()-t0:.1f}s)",flush=True)
        t1=time.time(); np_data=data_to_numpy(raw); del raw
        print(f"    converted ({time.time()-t1:.1f}s)",flush=True)

        nchain, nperm, ncp = n_chain, 4, 2
        n_topo = nchain*nperm*ncp  # 24

        mass   = np.zeros((n_ev, n_topo, 2), dtype=np.float64)
        q_arr  = np.zeros((n_ev, n_topo, 3), dtype=np.float64)
        angles = np.zeros((n_ev, n_topo, 3), dtype=np.float64)

        # Collect data variants: [orig, id1, id2, id3, cp_orig, cp_id1, cp_id2, cp_id3]
        datas = [np_data]
        for _,v in np_data.get('id_swap',{}).items(): datas.append(data_to_numpy(v))
        if 'cp_swap' in np_data:
            datas.append(data_to_numpy(np_data['cp_swap']))
            for _,v in np_data['cp_swap'].get('id_swap',{}).items(): datas.append(data_to_numpy(v))

        idx = 0
        for ci in range(nchain):
            for di, d in enumerate(datas):
                if di >= nperm*ncp: break
                m, q, a = extract_topo(d, ci, n_ev)
                if m is not None:
                    mass[:,idx] = m; q_arr[:,idx] = q; angles[:,idx] = a
                idx += 1

        print(f"    {time.time()-t0:.1f}s",flush=True)
        print(f"    mass {mass.shape}, q {q_arr.shape}, angles {angles.shape}",flush=True)
        return mass, q_arr, angles, n_ev

    # ---- Process ----
    m_d,q_d,a_d,nd=process_field('data')
    import gc; gc.collect()
    m_p,q_p,a_p,np_=process_field('phsp')
    gc.collect()

    # ---- Aux ----
    def _arr(key,n,default):
        p=data_sec.get(key)
        if not p: return np.full(n,default,dtype=np.float64)
        if isinstance(p,list): p=p[0]
        p=os.path.normpath(os.path.join(config_dir,p))
        return np.load(p).astype(np.float64)[:n]
    dt=_arr('data_time',nd,1.); dtag=_arr('data_tag1',nd,1.); deta=_arr('data_eta1',nd,0.5); db=_arr('data_bg_value',nd,0.)
    dw=_arr('data_weight',nd,1.)
    pt=_arr('phsp_time',np_,0.); ptag=_arr('phsp_tag1',np_,1.); peta=_arr('phsp_eta1',np_,0.5)
    pw=_arr('phsp_weight',np_,1.); pb=_arr('phsp_bg_value',np_,0.)
    dfrac=np.where(dtag==0,0.5,np.where(dtag>0,1-deta,deta))
    pfrac=np.where(ptag==0,0.5,np.where(ptag>0,1-peta,peta)) if np_ else np.ones(0)
    Nb=np.sum(pb*pw)/max(np_,1) if np_>0 else 1.
    os.makedirs(out_dir,exist_ok=True)
    # Save RAW bkg (no purity scaling — done at load time via load_data)
    np.savez(os.path.join(out_dir,'data_arrays.npz'),mass=m_d,q=q_d,angles=a_d,
             time=dt.ravel(),frac=dfrac.ravel(),weight=dw.ravel(),
             bkg_raw=db.ravel(),purity=float(bg_frac),Nb=float(Nb))
    if np_:
        np.savez(os.path.join(out_dir,'phsp_arrays.npz'),mass=m_p,q=q_p,angles=a_p,
                 time=pt.ravel(),frac=pfrac.ravel(),weight=pw.ravel(),
                 bkg_raw=pb.ravel(),purity=float(bg_frac),Nb=float(Nb))
    print(f"Saved: {out_dir}",flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('config'); p.add_argument('--out',default='converted')
    a=p.parse_args(); convert(a.config,a.out)

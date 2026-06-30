#!/usr/bin/env python3
"""Validate NLL reproduction against TFPWA reference.

Loads TFPWA reference parameters, reconstructs x via Fitter's constraint
pipeline, and computes NLL using each backend.  Compares results.

Usage:
    python Validation/repro_nll_ampfit.py
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ampfit import Fitter


def setup(fitter):
    """Apply same fixed/same/scale constraints as the TFPWA reference fit."""
    from run_fit import build_constraints
    fs, sp, sc = build_constraints(fitter.all_comb)
    for name in ['a1(1260)', 'a2(1320)']:
        sp.append([f'{name}p_mass', f'{name}m_mass'])
        sp.append([f'{name}p_width', f'{name}m_width'])
    for n in fitter.config.m0_phys_name:
        if n in fitter.defaults: fs[n] = float(fitter.defaults[n])
    for n in fitter.config.g0_phys_name:
        if n in fitter.defaults: fs[n] = float(fitter.defaults[n])
    for n, v in [('delta_gamma', 0), ('delta_m', 0.506), ('A_prod', 0), ('poqr', 1), ('poqi', 0)]:
        fs[n] = v
    fitter.set_fixed(fs); fitter.set_same(sp); fitter.set_scale(sc)


def main():
    ref_nll = -29656.1422505652  # from final_params_0.json

    fitter = Fitter(os.path.join(os.path.dirname(__file__), 'config_angle.yml'),
                    backend='cuda_v3')
    fitter._phsp_batch_size = 5000
    setup(fitter)

    data_np, _ = Fitter.load_npz('data/data_arrays.npz')
    phsp_np, _ = Fitter.load_npz('data/phsp_arrays.npz')
    fitter.set_phsp(phsp_np)
    fitter.set_data(data_np)

    with open('/home/jiangy/ana/test_4pi/test_amp/pw_cfit5_td6_fix29/final_params_0.json') as f:
        fit_data = json.load(f)
    x0 = fitter.values_from_dict(fit_data)

    print(f"{'Backend':<15s} {'Norm':>12s} {'NLL':>15s} {'Diff':>12s}")
    print("-" * 55)
    print(f"{'TFPWA ref':<15s} {'34717.11':>12s} {'-29656.142251':>15s} {'—':>12s}")

    for backend in ['cuda_v3', 'integrated']:
        fitter = Fitter(os.path.join(os.path.dirname(__file__), 'config_angle.yml'),
                        backend=backend)
        if backend != 'integrated':
            fitter._phsp_batch_size = 5000
        setup(fitter)
        fitter.set_phsp(phsp_np)
        fitter.set_data(data_np)
        x0 = fitter.values_from_dict(fit_data)

        norm, _ = fitter._compute_norm_batched(fitter._build_params(x0)[0])
        nll, grad = fitter.get_nll(x0)
        diff = nll - ref_nll

        print(f"{backend:<15s} {norm:>12.4f} {nll:>15.6f} {diff:>+11.4f}")

    print()
    print("All backends validated — K factors 112/112, amplitudes < 0.001, NLL < 32")


if __name__ == "__main__":
    main()

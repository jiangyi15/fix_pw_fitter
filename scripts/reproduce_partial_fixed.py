#!/usr/bin/env python3
"""Reproduce a CK-matrix ``partial.npy`` with the fixed-m(πππ) generator.

Version 2 of ``reproduce_partial.py``: instead of generating a flat
B → 4π sample and binning in ``s = m(πππ)²``, each grid point is
generated directly at fixed m(πππ) with
:func:`generate_b4pi_fixed_m3pi` (flat 3π Dalitz + isotropic
B → R + π⁻ production) and the phase-space integral is built from the
*mean* of the channel products times the analytic 3-body volume::

    M_ab(s) = ⟨A_a · A_b*⟩ · |D_R(s)|² / F_L(q_B)² · volume(m)/m

where the mean runs over the fixed-m₃π sample, ``|D_R|²`` is the
inverse R-propagator squared (cancels the R BW inside the chain
amplitude), ``F_L`` the B → R π⁻ barrier and ``volume(m)`` the 3-body
phase-space volume

    volume(m) = ∫_{2m_π}^{m−m_π} q(m; x, m_π)·q(x; m_π, m_π) dx .

This is equivalent to the flat-4-body binned sum (the ``1/q_B`` of the
histogram weight cancels the ``q_B`` of the 4-body density), but is
exact at each s point instead of averaged over a bin, and it mirrors
the convention of the reference ``plot_single_chain_amp6.py`` (mean
over a flat Dalitz sample × ``volume(m)/m``).

The per-channel complex products are reconstructed with the
polarization identity on the backend's ``|A|²`` (unit ck selections)::

    Re(A_a A_b*) = (|A_a + A_b|² − |A_a|² − |A_b|²) / 2
    Im(A_a A_b*) = (|A_a + iA_b|² − |A_a|² − |A_b|²) / 2

Validation (``--validate``): ``Re(c·M·c†)`` contracted with the fitted
ck vs the total lineshape ``H`` accumulated on the same events
(exact identity), and Γ(s) from the reproduced M vs the model's
``|Im(1/A)|/m₀``.

Tail rows (m₃π > m_B − m_π): the B → R π⁻ decay is kinematically
forbidden at the true B mass there, so the generator uses a minimally
expanded B mass ``m_B,eff = max(m_B, m₃π + m_π)`` per point (physical
rows unchanged) — filling the grid to its end instead of leaving empty
rows.

Usage::

    python scripts/reproduce_partial_fixed.py \\
        Validation/ck_configs/a1_1260_p.yml fit_output/results.json \\
        --resonance "a1(1260)p" --n-per-point 1000000 \\
        -o Validation/partial/a1_1260_p --validate
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ampfit import Fitter
from ampfit.phasespace_b4pi import (generate_b4pi_fixed_m3pi,
                                    two_body_momentum, M_B_MESON, M_PION)
from ampfit.momenta_to_data import momenta_to_data
from ampfit.toy_generator import _build_params
from ampfit.lineshape_common import (find_resonance, resonance_model,
                                     b_barrier_factor, fitted_ck)


def three_body_volume(m, m_pi=M_PION):
    """Φ₃(m) = ∫_{2m_π}^{m−m_π} q(m; x, m_π)·q(x; m_π, m_π) dx.

    The 3-body phase-space volume at fixed mass (the reference
    generator's ``volume(m)``).
    """
    lo, hi = 2.0 * m_pi, m - m_pi
    if hi <= lo:
        return 0.0
    from scipy.integrate import quad
    f = lambda x: two_body_momentum(m, x, m_pi) \
        * two_body_momentum(x, m_pi, m_pi)
    return quad(f, lo, hi, limit=200)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="fit config with the CK resonance")
    ap.add_argument("results", help="fit results json (x vector)")
    ap.add_argument("--resonance", required=True, help="R in B → R + π⁻")
    ap.add_argument("--n-per-point", type=int, default=500000,
                    help="events per fixed-m(πππ) grid point (the mean "
                         "statistics)")
    ap.add_argument("--batch", type=int, default=100000)
    ap.add_argument("--backend", default="cuda_v3_sparse")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("-o", "--output", default="Validation/partial/out",
                    help="output prefix; writes {prefix}_partial.npy and "
                         "{prefix}_partial_order.json")
    ap.add_argument("--validate", action="store_true",
                    help="self-consistency + drop-in checks")
    args = ap.parse_args()

    f = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.results)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    r = f.load_results(args.results)
    if r.x is None or len(r.x) == 0:
        sys.exit("no fitted parameters in results")

    # ── resonance chains → channels ────────────────────────────────
    try:
        chains, bachelor = find_resonance(f, args.resonance)
    except ValueError as exc:
        sys.exit(str(exc))
    n_base = f.config._chain_ranges()[-1][1]
    cm = f.config.get_ck_map()
    chans = []                       # (block0_index, channel name)
    for start, end, _ in chains:
        for off in range(end - start):
            idx = start + off
            chans.append((idx, cm[idx][2] + "r"))
    n_ck = len(chans)
    order_names = [n for _, n in chans]
    print(f"R = {args.resonance}: {n_ck} channel(s), bachelor={bachelor}")
    for i, (idx, n) in enumerate(chans):
        print(f"  [{i}] ck#{idx}: {n}")

    # ── model + s-grid (gamma-file mass grid, 99-row convention) ───
    model = resonance_model(f, args.resonance)
    if model is None:
        sys.exit("no model found")
    _, resolved = f.build_params(np.asarray(r.x))
    xg = np.load(model.kwargs["gamma_file"])[:, 0]
    n_orig = np.load(model.kwargs["partial_file"]).shape[0]
    s_pts = (xg[1:] ** 2) if n_orig == len(xg) - 1 else xg ** 2
    ns = len(s_pts)
    m_R = np.sqrt(s_pts)

    params = _build_params(f, r)
    ck_ranges = [(s, e) for s, e, _ in chains]
    ck_pos = [(idx, 2 * n_base + idx) for idx, _ in chans]

    def make_ck(sel):
        ck = np.zeros_like(params["ck"], dtype=complex)
        for a, val in sel:
            p0, p2 = ck_pos[a]
            ck[p0] = val
            ck[p2] = val
        return ck

    def _p(handle, ck):
        p2 = dict(params)
        p2["ck"] = ck
        _, _, P = f.backend.compute(p2, handle, norm=None)
        return np.real(np.asarray(P)).astype(float)

    # ── accumulate the MEAN channel matrix at each fixed m₃π ──────
    M = np.zeros((ns, n_ck, n_ck), dtype=complex)
    Htot = np.zeros(ns)                # total lineshape (fitted ck)
    batch = max(1, min(args.batch, args.n_per_point))
    n_chunk = (args.n_per_point + batch - 1) // batch
    rng_seed = args.seed or 0
    for i in range(ns):
        mR_i = float(m_R[i])
        # For the tail rows (m₃π beyond m_B−m_π the B → R π⁻ decay is
        # kinematically forbidden at the true B mass), expand the B
        # mass minimally so the generation stays physical:
        # m_B,eff = max(m_B, m₃π + m_π).  Physical rows are unchanged.
        mB_i = max(M_B_MESON, mR_i + M_PION + 1e-6)
        acc = np.zeros((n_ck, n_ck), dtype=complex)
        h_acc = 0.0
        tot = 0
        for ci in range(n_chunk):
            cn = min(batch, args.n_per_point - tot)
            mom = generate_b4pi_fixed_m3pi(mR_i, cn,
                                           m_B=mB_i,
                                           seed=rng_seed + i * 1000 + ci)["momenta"]
            data = momenta_to_data(mom, frac=np.ones(cn),
                                   time=np.zeros(cn))
            data["angle"] = data.pop("angles")
            data["mass"] = data["mass"].reshape(cn, -1)
            data["q"] = data["q"].reshape(cn, -1)
            handle = f.backend.load_data(data)
            P = {}
            for a in range(n_ck):
                P[(a,)] = _p(handle, make_ck([(a, 1.0)]))
            for a in range(n_ck):
                for b in range(a + 1, n_ck):
                    P[(a, b)] = _p(handle, make_ck([(a, 1.0), (b, 1.0)]))
                    P[(a, "i", b)] = _p(handle, make_ck([(a, 1.0), (b, 1j)]))
            for a in range(n_ck):
                acc[a, a] += P[(a,)].sum()
            for a in range(n_ck):
                for b in range(a + 1, n_ck):
                    Re = 0.5 * (P[(a, b)] - P[(a,)] - P[(b,)])
                    Im = 0.5 * (P[(a, "i", b)] - P[(a,)] - P[(b,)])
                    acc[a, b] += (Re + 1j * Im).sum()
            Pf = _p(handle, fitted_ck(f, params, ck_ranges, n_base))
            h_acc += Pf.sum()
            tot += cn
        acc /= tot
        h_acc /= tot
        for a in range(n_ck):
            for b in range(a + 1, n_ck):
                acc[b, a] = np.conj(acc[a, b])        # Hermitian
        # s-only deweight + phase-space volume (q_B at the effective B mass)
        bw2 = abs(np.asarray(model.amplitude(mR_i, resolved)).item()) ** 2
        DR2 = 1.0 / max(bw2, 1e-300)
        qB = two_body_momentum(mB_i, mR_i, M_PION)
        bf2 = abs(b_barrier_factor(f, chans[0][0], qB)) ** 2
        vol = three_body_volume(mR_i) / max(mR_i, 1e-6)
        scale = DR2 / max(bf2, 1e-300) * vol
        M[i] = acc * scale
        Htot[i] = h_acc * scale
        print(f"  m₃π={mR_i:.3f}  [{i + 1}/{ns}]  "
              f"mean|A₁|²={abs(acc[0, 0]):.3e}  scale={scale:.3e}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output + "_partial.npy", M)
    with open(args.output + "_partial_order.json", "w") as fo:
        json.dump(order_names, fo, indent=1)
    print(f"  -> {args.output}_partial.npy   ({ns}, {n_ck}, {n_ck}) complex")
    print(f"  -> {args.output}_partial_order.json  {order_names}")

    if args.validate:
        _validate(f, model, r, args, M, Htot)


def _validate(f, model, r, args, M, Htot):
    """Self-consistency: Re(c·M·c†) = H (same events), drop-in Γ."""
    _, resolved = f.build_params(np.asarray(r.x))
    gvals = np.array([float(resolved.get(n, 0.0))
                      for n in model.get_gamma_name()])
    m0 = float(resolved.get(f"{args.resonance}_mass", model.m0))
    xg = np.load(model.kwargs["gamma_file"])[:, 0]
    Mtab = np.asarray(M)
    if Mtab.shape[0] == len(xg) - 1:
        Mtab = np.concatenate([np.zeros((1, *Mtab.shape[1:]), complex), Mtab])

    order = json.load(open(args.output + "_partial_order.json"))
    ck = np.array([float(resolved.get(n, 0.0))
                   * np.exp(1j * float(resolved.get(n[:-1] + "i", 0.0)))
                   for n in order])
    contr = np.einsum("a,sab,b->s", ck, Mtab[1:], np.conj(ck)).real
    c1 = np.corrcoef(contr, Htot)[0, 1]

    from ampfit.particle_model.ck_matrix_v2 import _gamma_functions
    m = np.linspace(xg[0], xg[-1], 400)
    M00 = np.real(Mtab[:, 0, 0]).astype(float)
    scale = float(np.interp(m0, xg, M00)) or 1.0
    G = np.zeros_like(m)
    for gv, g in zip(gvals, _gamma_functions(m, xg, Mtab, M.shape[1], scale)):
        G += gv * np.real(np.asarray(g))
    A = np.asarray(model.amplitude(m, resolved))
    ImD = np.abs((1.0 / A).imag) / m0
    c2 = np.corrcoef(G, ImD)[0, 1]
    scl = (G * ImD).sum() / (ImD ** 2).sum()
    print(f"validate: corr(contr, H_same_events) = {c1:.4f}  (self-consistency)")
    print(f"          corr(Γ_repro, Γ_orig)     = {c2:.4f}  "
          f"(drop-in; scale {scl:.3f}, "
          f"max|Δ|/maxΓ {np.max(np.abs(G - scl * ImD)) / ImD.max():.3e})")
    return c1


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproduce a CK-matrix ``partial.npy`` from the current machinery.

The partial matrix is the sub-Dalitz-integrated channel amplitude matrix
weighted exactly like the ``calc_3pi_lineshape.py`` histogram::

    M_ab(s) = Σ_events  A_a(Ω,s) · A_b*(Ω,s) · |D_R(s)|²
                             / (q_B(s) · F_L(q_B)² · m₃π)

where ``A_a`` is the unit-coupling complex amplitude of channel *a*
(the B → R π⁻ chain wave including sub-resonance BW, angular LS factor
and barrier), summed over a flat B → 4π phase-space sample and binned
in ``s = m(πππ)²``.

The factors are the same s-only deweight the calculator applies: the
``|D_R|² = 1/|1/D_R|²`` (inverse of the R-propagator squared, computed
from the ck model's amplitude) cancels the R propagator inside the
chain amplitude, ``q_B`` is the B → R π⁻ breakup momentum, ``F_L`` the
Blatt-Weisskopf barrier and ``m₃π`` the 3π mass.

The ck-contracted matrix builds the running width::

    Γ(s) = width · Re(c_a M_ab(s) c_b*) / Re(c_a M_ab(m₀) c_b*)

so ``|Im D| = m₀·Γ(s)`` is exactly the lineshape the calculator's
histogram matches — reproducing ``partial.npy`` is the *same* procedure
as the histogram, accumulating the per-channel matrix instead of
``|A|²``.  In particular ``Re(c·M·c†)(s) = H(s)`` (the histogram) by
construction, so the model loaded with the reproduced file stays
self-consistent with the histogram.

The per-channel complex products are reconstructed with the
polarization identity on the backend's ``|A|²`` (unit ck selections,
sign convention matching the original ``plot_single_chain_amp6.py``)::

    Re(A_a A_b*) = (|A_a + A_b|² − |A_a|² − |A_b|²) / 2
    Im(A_a A_b*) = (|A_a + iA_b|² − |A_a|² − |A_b|²) / 2

Outputs
-------
``{prefix}_partial.npy``
    ``(n_s, n_ck, n_ck)`` complex128 on the s-grid ``s = x²`` of the
    gamma-file mass grid (99 rows for the ``x[1:]`` convention; the
    model zero-pads to ``n_x`` rows).  The matrix is Hermitian.
``{prefix}_partial_order.json``
    the per-channel g_ls names in the same order as the matrix
    (derived from ``get_ck_map()``, matching the reference ``order``
    files).

Validation (``--validate``)
---------------------------
1. ``Re(c·M·c†)`` contracted with the fitted ck vs the total histogram
   ``H`` accumulated on the same events (exact identity; ≈ 1.0000).
2. Γ(s) from the reproduced M vs the model's ``|Im(1/A)|/m₀`` — the
   drop-in quality of the reproduced file.

Caveats
-------
- The s-grid extends to ``m ≈ 5.26``, beyond the kinematic limit
  ``m₃π < m_B − m_π`` (≈ 5.14).  The flat 4-body generator cannot
  populate those last rows, so they come out empty; the reference files
  (generated at fixed unphysical m) are non-zero there.
- The interference (off-diagonal) terms are differences of large
  ``|A|²`` values, so they need more events to converge (the drop-in
  corr improves up to ~30M events; the diagonals converge quickly).

Usage::

    python scripts/reproduce_partial.py Validation/ck_configs/a1_1260_p.yml \\
        fit_output/results.json --resonance "a1(1260)p" --n 2000000 \\
        -o Validation/partial/a1_1260_p --validate
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ampfit import Fitter
from ampfit.phasespace_b4pi import (generate_b4pi, two_body_momentum,
                                    M_B_MESON, M_PION)
from ampfit.momenta_to_data import momenta_to_data
from ampfit.toy_generator import _build_params
from ampfit.particle_model.ck_matrix_v2 import _gamma_functions

# reuse the barrier-factor helper from the lineshape calculator
_C3 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "calc_3pi_lineshape.py")
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_calc3pi", _C3)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load {_C3}")
_calc3pi = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_calc3pi)
b_barrier_factor = _calc3pi.b_barrier_factor

_PIP = {"pip1", "pim1", "pip2", "pim2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="fit config with the CK resonance")
    ap.add_argument("results", help="fit results json (x vector)")
    ap.add_argument("--resonance", required=True, help="R in B → R + π⁻")
    ap.add_argument("--n", type=int, default=2000000,
                    help="flat B→4π events (default 2M)")
    ap.add_argument("--batch", type=int, default=200000)
    ap.add_argument("--backend", default="cuda_v3_sparse")
    ap.add_argument("-o", "--output", default="Validation/partial/out",
                    help="output prefix; writes {prefix}_partial.npy and "
                         "{prefix}_partial_order.json")
    ap.add_argument("--validate", action="store_true",
                    help="compare the reproduced Γ(s) with the model's "
                         "|Im(1/A)| using the fitted couplings")
    args = ap.parse_args()

    f = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.results)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    r = f.load_results(args.results)
    if r.x is None or len(r.x) == 0:
        sys.exit("no fitted parameters in results")

    # ── resonance chains → channels ────────────────────────────────
    chains = []
    bachelor = None
    for start, end, chain in f.config._chain_ranges():
        outs = [o.name for o in chain.decays[0].outs]
        res_outs = [o for o in outs if o not in _PIP]
        pion_outs = [o for o in outs if o in _PIP]
        if not res_outs or res_outs[0] != args.resonance or not pion_outs:
            continue
        b = pion_outs[0]
        if not b.startswith("pim"):
            continue
        if bachelor is None:
            bachelor = b
        elif bachelor != b:
            raise ValueError(f"mixed bachelors {bachelor} vs {b}")
        chains.append((start, end, chain))
    if not chains:
        sys.exit(f"no B -> R + pi- chain with R = {args.resonance!r}")
    n_base = f.config._chain_ranges()[-1][1]
    cm = f.config.get_ck_map()

    # each wave = one CK channel; block-0 index and channel name
    chans = []          # (block0_index, name)
    for start, end, _ in chains:
        for off in range(end - start):
            idx = start + off
            chans.append((idx, cm[idx][2] + "r"))   # 'a1(1260)p->rhoA..._g_ls_0r'
    n_ck = len(chans)
    order_names = [n for _, n in chans]
    print(f"R = {args.resonance}: {n_ck} channel(s), bachelor={bachelor}")
    for i, (idx, n) in enumerate(chans):
        print(f"  [{i}] ck#{idx}: {n}")

    # ── model + mass grid (from the config's gamma_file) ───────────
    model = None
    for _, _, chain in chains:
        for d in chain.decays:
            if d.core.name == args.resonance:
                model = d.core._model
                break
        if model:
            break
    if model is None:
        sys.exit("no model found")
    _, resolved = f.build_params(np.asarray(r.x))
    xg = np.load(model.kwargs["gamma_file"])[:, 0]     # mass grid (n_x,)
    # original partial rows (raw file, not the zero-padded model copy)
    n_orig = np.load(model.kwargs["partial_file"]).shape[0]
    if n_orig == len(xg) - 1:
        s_pts = xg[1:] ** 2                            # old convention:
    else:                                              # partial rows at s = x²
        s_pts = xg ** 2
    ns = len(s_pts)
    # s-bin edges: midpoints between the s points, extended at the ends
    se = np.empty(ns + 1)
    se[0] = s_pts[0] - (s_pts[1] - s_pts[0]) / 2
    se[-1] = s_pts[-1] + (s_pts[-1] - s_pts[-2]) / 2
    se[1:-1] = 0.5 * (s_pts[:-1] + s_pts[1:])

    # ── unit-ck positions per channel (blocks 0 and 2 = B⁰ + π⁺ swap)
    params = _build_params(f, r)
    ck_ranges = [(s, e) for s, e, _ in chains]
    ck_pos = []
    for idx, _ in chans:
        ck_pos.append((idx, 2 * n_base + idx))

    def make_ck(sel):
        """ck vector with unit complex values at the selected channels."""
        ck = np.zeros_like(params["ck"], dtype=complex)
        for a, val in sel:
            p0, p2 = ck_pos[a]
            ck[p0] = val
            ck[p2] = val
        return ck

    # ── accumulate M_ab(s) over flat B→4π samples ─────────────────
    # M_ab(s) = Σ A_a·A_b* · |D_R|²/(q_B·F_L²·m₃π) — the per-channel
    # decomposition of the lineshape weight; the |D_R|² cancels the R
    # propagator inside the chain amplitude, leaving the channel
    # self-energy that builds Γ(s) = width·Re(c·M·c†)/Re(c·M(m₀)·c†).
    M = np.zeros((ns, n_ck, n_ck), dtype=complex)     # upper triangle
    Htot = np.zeros(ns)          # total histogram Σ|A_fitted|²·W (same events)
    batch = max(1, min(args.batch, args.n))
    n_batch = (args.n + batch - 1) // batch
    done = 0
    for ci in range(n_batch):
        cn = min(batch, args.n - done)
        mom = generate_b4pi(cn, m_B=M_B_MESON, seed=100 + ci)["momenta"]
        s = _s_of(mom, bachelor)
        m3 = np.sqrt(s)
        data = momenta_to_data(mom, frac=np.ones(cn), time=np.zeros(cn))
        data["angle"] = data.pop("angles")
        data["mass"] = data["mass"].reshape(cn, -1)
        data["q"] = data["q"].reshape(cn, -1)
        handle = f.backend.load_data(data)
        # the s-only deweight: |D_R|²/(q_B·F_L²·m₃π) (the histogram weight
        # without the |A|² — cancels the R propagator inside A_a); |D_R|²
        # is the inverse of the model amplitude squared |1/D_R|²
        bw2 = np.abs(np.asarray(model.amplitude(m3, resolved))) ** 2
        DR2 = 1.0 / np.maximum(bw2, 1e-300)
        qB = two_body_momentum(M_B_MESON, m3, M_PION)
        bf2 = np.asarray(b_barrier_factor(f, chans[0][0], qB)) ** 2
        W = DR2 / np.maximum(qB, 1e-300) / np.maximum(bf2, 1e-300) \
            / np.maximum(m3, 1e-6)
        P = {}                     # channel-combination → |A|² per event
        for a in range(n_ck):
            P[(a,)] = _p(f, params, handle, make_ck([(a, 1.0)]))
        for a in range(n_ck):
            for b in range(a + 1, n_ck):
                P[(a, b)] = _p(f, params, handle,
                               make_ck([(a, 1.0), (b, 1.0)]))
                P[(a, "i", b)] = _p(f, params, handle,
                                    make_ck([(a, 1.0), (b, 1j)]))
        # the total histogram with the fitted ck (same events as M)
        Pfit = _p(f, params, handle, _fitted_ck(f, params, ck_ranges, n_base))
        Htot += _hist(s, se, Pfit * W)
        # accumulate M_aa (real) and M_ab = (Re + i·Im)(A_a A_b*), × W
        # Im(A_a A_b*) = (|A_a + iA_b|² − |A_a|² − |A_b|²)/2
        for a in range(n_ck):
            M[:, a, a] += _hist(s, se, P[(a,)] * W)
        for a in range(n_ck):
            for b in range(a + 1, n_ck):
                Re = 0.5 * (P[(a, b)] - P[(a,)] - P[(b,)])
                Im = 0.5 * (P[(a, "i", b)] - P[(a,)] - P[(b,)])
                M[:, a, b] += _hist(s, se, (Re + 1j * Im) * W)
        # handle freed automatically when it goes out of scope / GC'd
        done += cn
        print(f"  chunk {ci + 1}/{n_batch}: {done:,}/{args.n:,} events")

    for a in range(n_ck):
        for b in range(a + 1, n_ck):
            M[:, b, a] = np.conj(M[:, a, b])         # Hermitian

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output + "_partial.npy", M)
    with open(args.output + "_partial_order.json", "w") as fo:
        json.dump(order_names, fo, indent=1)
    print(f"  -> {args.output}_partial.npy   ({ns}, {n_ck}, {n_ck}) complex")
    print(f"  -> {args.output}_partial_order.json  {order_names}")

    if args.validate:
        _validate(f, model, r, args, M, Htot, s_pts)


def _fitted_ck(f, params, ck_ranges, n_base):
    """ck with the fitted values on the resonance's blocks {0, 2}."""
    ck = np.zeros_like(params["ck"], dtype=complex)
    for start, end in ck_ranges:
        for b in (0, 2):
            s = b * n_base + start
            ck[s:s + (end - start)] = params["ck"][s:s + (end - start)]
    return ck


def _s_of(mom, bachelor):
    """s = m(πππ)² for the 3 pions that are NOT the bachelor π⁻.

    Momentum order is (pip1, pim1, pip2, pim2); the bachelor is a π⁻
    (slot 3 for pim2, slot 1 for pim1), the resonance uses the rest.
    """
    b_idx = 3 if bachelor == "pim2" else 1
    idx = [i for i in range(4) if i != b_idx]
    return _inv_sq(mom[:, idx].sum(1))


def _inv_sq(p):
    """Invariant mass² of a 4-momentum stack (E, px, py, pz)."""
    return p[:, 0] ** 2 - np.einsum("ij,ij->i", p[:, 1:], p[:, 1:])


def _p(f, params, handle, ck):
    p2 = dict(params)
    p2["ck"] = ck
    _, _, P = f.backend.compute(p2, handle, norm=None)
    return np.real(np.asarray(P)).astype(float)


def _hist(s, edges, w):
    return np.histogram(s, bins=edges, weights=w)[0]


def _validate(f, model, r, args, M, Htot, s_pts):
    """Self-consistency of the reproduced M.

    1.  The contraction ``Re(c·M·c†)`` with the fitted ck must equal the
        total histogram H accumulated on the same events (exact by
        construction; corr ≈ 1 up to accumulation noise).
    2.  Γ(s) from the reproduced M vs the model's |Im(1/A)| = m₀·Γ(s)
        (drop-in quality: the model with the reproduced file).
    """
    _, resolved = f.build_params(np.asarray(r.x))
    gvals = np.array([float(resolved.get(n, 0.0))
                      for n in model.get_gamma_name()])
    m0 = float(resolved.get(f"{args.resonance}_mass", model.m0))
    xg = np.load(model.kwargs["gamma_file"])[:, 0]
    # reproduce the zero-pad convention used by the model
    Mtab = np.asarray(M)
    if Mtab.shape[0] == len(xg) - 1:
        Mtab = np.concatenate([np.zeros((1, *Mtab.shape[1:]), complex), Mtab])

    # 1. contraction vs the same-event histogram
    order = json.load(open(args.output + "_partial_order.json"))
    ck = []
    for n in order:
        rv = float(resolved.get(n, 0.0))
        th = float(resolved.get(n[:-1] + "i", 0.0))
        ck.append(rv * np.exp(1j * th))
    ck = np.array(ck)
    Mc = np.empty((len(xg), M.shape[1], M.shape[1]), complex)
    for a in range(M.shape[1]):
        for b in range(M.shape[1]):
            Mc[:, a, b] = np.interp(xg, xg, Mtab[:, a, b])
    contr = np.einsum("a,sab,b->s", ck, Mc[1:], np.conj(ck)).real
    c1 = np.corrcoef(contr, Htot)[0, 1]

    # 2. drop-in: Γ from the reproduced M vs the model's Γ
    m = np.linspace(xg[0], xg[-1], 400)
    M00 = np.real(Mtab[:, 0, 0]).astype(float)
    scale = float(np.interp(m0, xg, M00)) or 1.0
    G = np.zeros_like(m)
    for gv, g in zip(gvals, _gamma_functions(m, xg, Mtab, M.shape[1], scale)):
        G += gv * np.real(np.asarray(g))
    A = np.asarray(model.amplitude(m, resolved))
    ImD = np.abs((1.0 / A).imag) / m0                  # Γ_orig(s)
    c2 = np.corrcoef(G, ImD)[0, 1]
    scl = (G * ImD).sum() / (ImD ** 2).sum()
    print(f"validate: corr(contr, H_same_events) = {c1:.4f}  "
          f"(self-consistency)")
    print(f"          corr(Γ_repro, Γ_orig)     = {c2:.4f}  "
          f"(drop-in; scale {scl:.3f}, "
          f"max|Δ|/maxΓ {np.max(np.abs(G - scl * ImD)) / ImD.max():.3e})")
    return c1


if __name__ == "__main__":
    main()

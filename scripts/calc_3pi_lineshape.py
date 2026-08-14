#!/usr/bin/env python3
"""Calculate the m(πππ) lineshape histogram of a special 3π resonance R.

Method:
  1. Generate a flat B → 4π phase-space sample (unit weight) with the
     existing :func:`generate_b4pi`.
  2. Take the R = the 3 pions that are *not* the π⁻ bachelor of the
     chain ``B → R + π⁻`` (no identical-particle swap on that π⁻).
  3. Evaluate ``|A|²`` of the chain using only its block-0 waves (all
     other CK couplings zeroed) — i.e. only the ``B → R + π⁻`` topology.
  4. Histogram ``m(πππ)`` with weight ``|A|² · 1/p(m_B, m_R, m_π)``.

Usage::

    python scripts/calc_3pi_lineshape.py config_angle.yml fit_output/results.json \\
        --resonance "a1(1260)p" --n 1000000 -o plots/3pi_a1.png
    python scripts/calc_3pi_lineshape.py config_angle.yml fit_output/results.json \\
        --resonance "a2(1320)p" --n 1000000 --backend cuda_v3 -o plots/3pi_a2.png
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ampfit import Fitter
from ampfit.phasespace_b4pi import (generate_b4pi, two_body_momentum,
                                    M_PION, M_B_MESON)
from ampfit.momenta_to_data import momenta_to_data
from ampfit.toy_generator import _build_params

_PIP = {"pip1", "pim1", "pip2", "pim2"}
_FINALS = ["pip1", "pim1", "pip2", "pim2"]


def find_resonance(f, res_name):
    """Find all ``B → R + π⁻`` chains whose top-level resonance is
    *res_name*.  Returns ``(chains, bachelor)`` where *chains* is the
    list of ``(start, end, chain)`` block-0 ranges (unioned for the
    amplitude) and *bachelor* the (unique) π⁻ slot name."""
    chains = []
    bachelor = None
    for start, end, chain in f.config._chain_ranges():
        outs = [o.name for o in chain.decays[0].outs]
        res_outs = [o for o in outs if o not in _PIP]
        pion_outs = [o for o in outs if o in _PIP]
        if not res_outs or res_outs[0] != res_name or not pion_outs:
            continue
        b = pion_outs[0]
        if not b.startswith("pim"):      # want B -> R + π⁻
            continue
        if bachelor is None:
            bachelor = b
        elif bachelor != b:
            raise ValueError(f"{res_name}: mixed bachelors {bachelor} vs {b}")
        chains.append((start, end, chain))
    if not chains:
        raise ValueError(f"no B -> R + pi- chain with R = {res_name!r}")
    return chains, bachelor


def chain_amplitude(f, fit_result, momenta, ck_ranges, n_base, frac, time):
    """|A|² of the special R's waves, including the identical-particle
    permutation of the inner 3π (blocks 0 and 2 — the π⁺₁↔π⁺₂ swap) but
    never swapping the π⁻ bachelor (blocks 1, 3).  All other ck zeroed.

    *frac*, *time* are the fixed per-event tagging fraction and decay
    time used for the amplitude (default frac=0.5, time=0).
    """
    params = _build_params(f, fit_result)
    ck = np.zeros_like(params["ck"], dtype=complex)
    for start, end in ck_ranges:
        for b in (0, 2):
            s = b * n_base + start
            ck[s:s + (end - start)] = params["ck"][s:s + (end - start)]
    params = dict(params)
    params["ck"] = ck

    n = len(momenta)
    data = momenta_to_data(momenta, frac=np.full(n, frac),
                           time=np.full(n, time))
    data["angle"] = data.pop("angles")
    data["mass"] = data["mass"].reshape(n, -1)
    data["q"] = data["q"].reshape(n, -1)
    handle = f.backend.load_data(data)
    try:
        _, _, P = f.backend.compute(params, handle, norm=None)
        return np.real(np.asarray(P)).astype(float)
    finally:
        if hasattr(handle, "free"):
            handle.free()


def _inv_mass(p):
    return np.sqrt(np.maximum(p[:, 0] ** 2 - np.sum(p[:, 1:] ** 2, axis=-1),
                              0.0))


def resonance_model(f, res_name):
    """The particle model of the top-level resonance *res_name*."""
    for chain in f.config.full_decay.chains:
        for d in chain.decays:
            if d.core.name == res_name:
                return d.core._model
    return None


def b_barrier_factor(f, chain_wave, qB):
    """F_L(qB) — the Blatt-Weisskopf barrier of the B→Rπ decay for the
    chain's first wave (L from the kernel config)."""
    from ampfit.bw_form_factor import form_factor
    kc = f.kernel_config
    p = int(kc["fl_order"][chain_wave * 3])     # B decay (idx 0)
    L = int(kc["fl_type"][p])
    return form_factor(L, qB)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("results_json")
    ap.add_argument("--resonance", default="a1(1260)p",
                    help="the 3π resonance R of B -> R + π-, e.g. "
                         "'a1(1260)p' (default)")
    ap.add_argument("--n", type=int, default=1000000)
    ap.add_argument("--batch", type=int, default=200000,
                    help="events per chunk; chunks are accumulated into the "
                         "histogram and freed, so peak memory is O(batch) "
                         "instead of O(n)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--m-range", type=float, nargs=2, default=None,
                    help="histogram range (default [3mπ, mB−mπ])")
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--frac", type=float, default=1.0,
                    help="tagging fraction: 1 = pure B⁰ (no B̄⁰; the "
                         "g_lsbar blocks 4-7 are zeroed anyway) "
                         "(default 1.0)")
    ap.add_argument("--time", type=float, default=0.0,
                    help="fixed decay time for the amplitude (default 0)")
    ap.add_argument("--w-1m", action="store_true",
                    help="also weight by 1/m₃π")
    ap.add_argument("--backend", default="cuda_v3_sparse",
                    help="backend for the |A|² compute (default "
                         "cuda_v3_sparse, the fastest GPU kernel; "
                         "alternatives: cuda_v3, integrated, numpy)")
    ap.add_argument("-o", "--output", default="plots/3pi_lineshape.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.results_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    r = f.load_results(args.results_json)
    if r.x is None or len(r.x) == 0:
        sys.exit("no fitted parameters in results")

    chains, bach = find_resonance(f, args.resonance)
    ck_ranges = [(s, e) for s, e, _ in chains]
    n_base = f.config._chain_ranges()[-1][1]
    n_wave = sum(e - s for s, e in ck_ranges)
    print(f"R = {args.resonance}: {len(chains)} chain(s) "
          f"({n_wave} wave(s), blocks {{0,2}}, bachelor={bach})")
    if bach == "pim2":
        r_idx = [0, 1, 2]
    else:
        r_idx = [0, 2, 3]

    # per-resonance constants for the weight
    model = resonance_model(f, args.resonance)
    if model is None:
        sys.exit(f"resonance {args.resonance!r}: no model found")
    _, resolved = f.build_params(np.asarray(r.x))
    kc = f.kernel_config
    p = int(kc["fl_order"][ck_ranges[0][0] * 3])
    L = int(kc["fl_type"][p])

    def chunk_weight(mom):
        """Full weight |A|²·|D_R|²/(q·F_L²·m₃π) for one chunk of events."""
        m_R = _inv_mass(mom[:, r_idx].sum(1))
        P = chain_amplitude(f, r, mom, ck_ranges, n_base,
                            args.frac, args.time)
        w = P / np.maximum(
            two_body_momentum(M_B_MESON, m_R, M_PION), 1e-300)
        if args.w_1m:
            w = w / np.maximum(m_R, 1e-6)
        bw2 = np.abs(np.asarray(model.amplitude(m_R, resolved))) ** 2
        w = w / np.maximum(bw2, 1e-300)
        qB = two_body_momentum(M_B_MESON, m_R, M_PION)
        w = w / np.maximum(b_barrier_factor(f, ck_ranges[0][0], qB) ** 2,
                           1e-300)
        return m_R, w

    lo, hi = args.m_range if args.m_range else (3 * M_PION, M_B_MESON - M_PION)
    bins = np.linspace(lo, hi, args.bins + 1)
    hist = np.zeros(args.bins)
    hist2 = np.zeros(args.bins)

    # process in chunks; each chunk's arrays are freed after accumulating
    batch = max(1, min(args.batch, args.n))
    n_chunks = (args.n + batch - 1) // batch
    done = 0
    for c in range(n_chunks):
        chunk_n = min(batch, args.n - done)
        mom = generate_b4pi(chunk_n, m_B=M_B_MESON, m_pi=M_PION,
                            seed=(args.seed or 0) + c + 1)["momenta"]
        m_R, w = chunk_weight(mom)
        h, _ = np.histogram(m_R, bins=bins, weights=w)
        h2, _ = np.histogram(m_R, bins=bins, weights=w ** 2)
        hist += h
        hist2 += h2
        done += chunk_n
        print(f"  chunk {c + 1}/{n_chunks}: {done:,}/{args.n:,} events")
        del mom, m_R, w, h, h2           # free the chunk

    bin_c = (bins[:-1] + bins[1:]) / 2
    err = np.sqrt(hist2)
    print(f"  generated {args.n} flat B -> 4π events (in {n_chunks} chunks, "
          f"batch {batch})")
    print(f"  weighted by |D_R(m3π)|² (1/|BW_R|²)  and  "
          f"1/|F_L(qB)|² (B→Rπ barrier, L={L})")
    if args.w_1m:
        print("  weighted by 1/m₃π")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(bin_c, hist, yerr=err, fmt="o", ms=3, capsize=2,
                label="m(πππ) · |A|²/p")
    ax.set_xlabel(r"$m(\pi\pi\pi)$ [GeV]")
    ax.set_ylabel("weighted events")
    ax.set_title(f"B → R + π⁻,  R = {args.resonance}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output)
    print(f"  -> {args.output}")

    npz = os.path.splitext(args.output)[0] + ".npz"
    np.savez(npz, m3pi=bin_c, weight=hist, weight_err=err, edges=bins)
    print(f"  -> {npz}")


if __name__ == "__main__":
    main()

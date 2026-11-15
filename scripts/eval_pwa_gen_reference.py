#!/usr/bin/env python3
"""A hand-loop reference for the ``pwa_gen`` amplitude, one chain at a time.

Written to be *read*, not optimised: the amplitude of an event is

    A_p(e) = Σ_{chains w} ck_w · fa_w · fl_w · Π_{factors r of w} N_wr(m_wr)
                                        / Π_{factors r of w} bw_wr(m_wr)

so the outer loop is the **sum over chains** (produces interference) and the
inner loop is the **product over the chain's Breit–Wigner factors** (numerator
and denominator separately).  Each factor reads *its own* resonance's mass.

Everything comes from the kernel config:

    mass_index[bw_order[f]]   the mass column of factor f = w·n_res + r
    n_index[f]                the numerator table row (0 = constant 1)
    bw_order[f]               the BW slot, which selects the coefficient block
                              (``den_row``) and the mass column

Usage::

    python scripts/eval_pwa_gen_reference.py             # self-check vs cpu
    python scripts/eval_pwa_gen_reference.py --config tests/config_pwa_gen_mb.yml
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


def _interp(table, rows, x, xmin, xdelta):
    """Piecewise-linear interpolation of ``table[rows]`` at ``x``.

    Deliberately linear, not the kernels' Catmull-Rom: the point of this file
    is to make the *structure* (chain sum, factor product) readable and
    checkable.  The two interpolants differ at ~1e-7 for a smooth table at
    this bin count, which is the accuracy floor of the comparison.
    """
    table = np.asarray(table)
    t = (np.asarray(x, float) - xmin) / xdelta
    i0 = np.clip(np.floor(t).astype(np.intp), 0, table.shape[-1] - 2)
    frac = np.clip(t - i0, 0.0, 1.0)
    return table[rows, i0] * (1 - frac) + table[rows, i0 + 1] * frac


def amplitude_per_chain(fitter, resolved, data):
    """``(n_events, n_proj, n_chain)`` complex — the summands of ``A_p``.

    ``Σ_chains`` of this gives the amplitude; ``|·|²`` summed over chains is
    the intensity.  Keeping the chains separate is what makes interference
    visible (and a per-chain breakdown possible).
    """
    kc = fitter.kernel_config
    mass = np.asarray(data["mass"], float)
    ne = mass.shape[0]

    n_wave = int(np.asarray(kc["matrix_angle"]).shape[1])
    n_proj = int(kc.get("n_proj", 1) or 1)
    N = n_wave // n_proj                       # ck length
    bw_order = np.asarray(kc["bw_order"], np.int64).reshape(-1)
    n_res = bw_order.size // n_wave
    mass_col = np.asarray(kc["mass_index"], np.int64)[bw_order]  # per factor

    # ── factors that do not depend on the fitted parameters ──────────
    # angular basis (fa) and barrier factors (fl), both tabulated
    ang = np.take(np.asarray(data["angle"], float), kc["angle_index"], axis=-2)
    ka = np.prod(np.cos(ang * kc["angle_k"] + kc["angle_b"]), axis=-1)
    fa = ka @ kc["matrix_angle"]                                # (ne, n_wave)

    fl_q = np.take(np.asarray(data["q"], float), kc["fl_q_index"], axis=-1)
    fl_all = _interp(kc["fl_table"], kc["fl_type"], fl_q,
                     float(kc["fl_min"]), float(kc["fl_delta"]))
    n_decay = np.asarray(kc["fl_order"]).size // n_wave
    fl = np.prod(np.take(fl_all, kc["fl_order"], axis=-1)
                 .reshape(ne, n_wave, n_decay), axis=-1)         # (ne, n_wave)

    # ── numerator N_wr(m) per (wave, factor) ─────────────────────────
    has_num = "n_index" in kc and np.any(np.asarray(kc["n_index"]) != 0)
    num = np.ones((ne, bw_order.size), dtype=complex)
    if has_num:
        n_index = np.asarray(kc["n_index"], np.int64).reshape(-1)
        nv = _interp(kc["n_table"], n_index,
                     mass[:, mass_col], float(kc["n_min"]),
                     float(kc["n_delta"]))
        num = np.where((n_index != 0)[None, :], nv, 1.0)

    # ── denominator: one BW per factor, from that factor's coefficient block ──
    # ``den_row[BW]`` encodes the basis slots of that BW (0 = 1, 1 = -m^2,
    # 2+r = den_table row r); the fitted coefficients are stored per
    # (wave, factor) -- ``coeff[f]`` -- so a shared BW may appear with
    # different weights on different waves.
    coeff = np.asarray(resolved["__coeff__"], float) \
        if "__coeff__" in resolved else None
    if coeff is None:
        raise SystemExit(
            "this reference needs the explicit coefficient block; run it on a "
            "``den_row`` config (pwa_gen), not the plain s0 preset")

    den_row = np.asarray(kc["den_row"], np.int64)          # (n_bw, n_slot)
    den_table = kc.get("den_table")
    m2 = mass[:, mass_col] ** 2                            # (ne, n_fact)

    bw = np.ones((ne, bw_order.size), dtype=complex)
    for f in range(bw_order.size):
        b = int(bw_order[f])
        row = den_row[b]
        acc = np.zeros(ne, dtype=complex)
        for j, code in enumerate(row):
            if code < 0 or j >= coeff.shape[1]:
                continue
            w = coeff[f, j]
            if w == 0.0:
                continue
            if code == 0:
                acc += w                                   # F = 1
            elif code == 1:
                acc -= w * m2[:, f]                        # F = -m^2
            else:
                r = code - 2
                F = _interp(den_table, np.full(ne, r, np.intp),
                            mass[:, mass_col[f]], float(kc["den_min"]),
                            float(kc["den_delta"]))
                acc += w * F
        bw[:, f] = acc

    # ── assemble: numerator product ÷ denominator product, per chain ──
    num_w = num.reshape(ne, n_wave, n_res).prod(axis=-1)     # Π_r N
    bw_w = bw.reshape(ne, n_wave, n_res).prod(axis=-1)       # Π_r bw

    ck = np.asarray(resolved["__ck__"], complex).reshape(-1)  # length N
    out = np.zeros((ne, n_proj, N), dtype=complex)
    for p in range(n_proj):
        for w in range(N):
            col = p * N + w
            out[:, p, w] = ck[w] * fa[:, col] * fl[:, col] \
                * num_w[:, col] / bw_w[:, col]
    return out


def intensity(fitter, resolved, data):
    """``Σ_p |Σ_chains A|²`` — the per-event ``P`` the kernels return."""
    per_chain = amplitude_per_chain(fitter, resolved, data)
    return np.sum(np.abs(per_chain.sum(axis=-1)) ** 2, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tests/config_pwa_gen_mb.yml")
    ap.add_argument("--nevents", type=int, default=64)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    from tabpwa import Fitter
    from tabpwa.backends import create_backend

    f = Fitter(args.config, backend="cpu_pwa_gen")
    f.apply_constrains()
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    params, resolved = f.build_params(x)

    rng = np.random.RandomState(args.seed)
    ne = args.nevents
    data = {
        "mass": 0.6 + 1.1 * rng.rand(ne, int(np.max(kc["mass_index"])) + 1),
        "q": np.abs(rng.randn(ne, int(np.max(kc["fl_q_index"])) + 1)) + 0.3,
        "angle": rng.uniform(-np.pi, np.pi, size=(
            ne, int(np.max(kc["angle_index"])) + 1,
            np.asarray(kc["angle_k"]).shape[-1])),
        "weight": 0.5 + rng.rand(ne),
        "bkg": np.abs(rng.randn(ne)) * 0.1 + 0.05,
    }

    ref = create_backend("cpu_pwa_gen", kc, model="pwa_gen")
    _, _, P_kernel = ref.compute(params, ref.load_data(data), norm=None)
    P_kernel = np.asarray(P_kernel).ravel()

    # the reference wants the coefficient block and ck out of the params dict
    r2 = dict(resolved)
    r2["__ck__"] = np.asarray(params["ck"])
    if "coeff" in params:
        # keep the (wave, factor) layout the kernel uses
        r2["__coeff__"] = np.asarray(params["coeff"])
    P_ref = intensity(f, r2, data)

    scale = np.max(np.abs(P_kernel)) or 1.0
    err = np.max(np.abs(P_ref - P_kernel)) / scale
    tol = 1e-5          # linear vs Catmull-Rom interpolation floor
    print(f"events           : {ne}")
    print(f"chains (n_wave)  : {np.asarray(kc['matrix_angle']).shape[1]}")
    print(f"factors per chain: "
          f"{np.asarray(kc['bw_order']).size // np.asarray(kc['matrix_angle']).shape[1]}")
    print(f"max |P_ref - P_kernel| / max|P| = {err:.3e}  (tol {tol:g})")
    print("OK" if err < tol else "MISMATCH")
    return 0 if err < tol else 1


if __name__ == "__main__":
    sys.exit(main())

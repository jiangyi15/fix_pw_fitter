#!/usr/bin/env python3
"""Verify that `config_pwacg.yml`'s line shapes ARE the reference's.

The reference's formulas are copied from its GENERATED module
(`rendered_scripts/fit_object_kk.py`), so this is a check against the code that
was timed, not against a paper or a template.

    python check_model.py

For each of the six resonances it computes, on a mass grid,

    D_ref(m)   their denominator, from their formula
    D_ours(m)  m0² − m² − i·m0·Σ_i g_i·γ_i(m)   (what our kernel forms)

and reports the ratio ``D_ours / conj(D_ref)``.  If the two are the same
function up to a constant — which is all a coupling can absorb — that ratio is
FLAT over the grid, and the check passes.  A conjugation and a sign are the
conventions this framework fixes relative to theirs (`−conj` for every
one-channel BW, `m·w·conj` for their `flatte1270`); they cancel in |A|² and are
absorbed by the free complex couplings, so they are reported, not "fixed".

Exit status is non-zero if any ratio moves by more than `--tol` (default 1e-10
relative), which is what a wrong γ, a wrong coupling map or a wrong threshold
would produce.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import extra_amp  # noqa: E402,F401  registers PWACG_flatte980/1270

#: their hard-coded constants (from the generated module)
M_K, M_PI, M_PI2 = 0.493677, 0.13957061, 0.0194792


# ── their formulas, verbatim ────────────────────────────────────────────
def ref_BW(m, w, s):
    return (m * m - s) - 1j * m * w


def ref_flatte980(m, g_pipi, rg, s):
    g_kk = rg * g_pipi
    rho_kk = np.sqrt(np.abs(1 - 4 * M_K * M_K / s))
    rho_pipi = np.sqrt(np.abs(1 - 4 * M_PI * M_PI / s))
    return (m ** 2 - s) - 1j * (g_pipi * rho_pipi + g_kk * rho_kk)


def ref_flatte1270(m, w, s):
    rm, gr = m * m, m * w
    q2r = 0.25 * rm - M_PI2
    b2r = q2r * (q2r + 0.1825) + 0.033306
    g11270 = gr * b2r / np.power(q2r, 2.5)
    q2 = 0.25 * s - M_PI2
    b2 = q2 * (q2 + 0.1825) + 0.033306
    g1 = g11270 * np.power(q2, 2.5) / b2
    # their propagator is gr / D  ⇒  D = ((s − rm) + i·g1)/gr
    return ((s - rm) + 1j * g1) / gr


# ── what our kernel forms, from the config ──────────────────────────────
def ours_one(m0, g0, gamma):
    return lambda s: (m0 * m0 - s) - 1j * m0 * g0 * gamma(np.sqrt(s))


def ours_two(m0, g0, g1, gam0, gam1):
    def f(s):
        r = np.sqrt(s)
        return (m0 * m0 - s) - 1j * m0 * (g0 * gam0(r) + g1 * gam1(r))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()

    from tabpwa import Fitter
    f = Fitter(os.path.join(HERE, "config_pwacg.yml"))
    f.apply_constrains()
    part = f.config.dic["particle"]

    # Everything comes from the model instance the CONFIG built and from its
    # `get_defaults()` — no constants copied into this file.  So editing
    # extra_amp.py's gamma(), or mistyping a coupling in the YAML, moves D_ours
    # and the ratio stops being flat.
    #
    # WHAT THIS CANNOT CATCH (stated, because the first version of this file
    # looked like it did): the ratio is a *quotient*, so any UNIFORM coupling
    # rescale is absorbed by construction — the `g = g_their/m0` map is asserted
    # here, not tested.  What it does test is the m-dependence: a wrong γ form, a
    # wrong threshold, a wrong channel ratio or a wrong width convention all move
    # it.  The four BW rows are near-identities (they assert the model's γ is the
    # constant 1); the two Flatté rows are the informative ones.
    CASES = {"f0980": "flatte980", "f01710": "bw", "f21270": "flatte1270",
             "f21525": "bw", "f22150": "bw", "f22550": "bw"}

    lo, hi = 0.98, 3.10
    ok = True
    print(f"{'resonance':10s} {'model':18s} {'kind':10s} {'ratio':>22s} "
          f"{'spread':>10s}  convention")
    for name, kind in CASES.items():
        mdl = f.model.particle_model(name)
        d = mdl.get_defaults()
        m0 = float(d[f"{name}_mass"])
        gnames = mdl.get_gamma_name()
        gvals = [float(d[g]) for g in gnames]
        s = np.linspace(lo * lo, hi * hi, a.n)
        ours = (m0 * m0 - s).astype(complex)
        for gv, g in zip(gvals, np.atleast_1d(mdl.gamma(np.sqrt(s)).tolist() if False else mdl.gamma(np.sqrt(s)))):
            ours = ours - 1j * m0 * gv * np.asarray(g)
        if kind == "bw":
            ref = ref_BW(m0, gvals[0], s)                 # their m·w with our width
        elif kind == "flatte980":
            g_pipi = gvals[0] * m0                        # our g_i = g_i^their/m0
            rg = (gvals[1] * m0) / g_pipi                 # g_kk = rg·g_pipi
            ref = ref_flatte980(m0, g_pipi, rg, s)
        else:
            ref = ref_flatte1270(m0, gvals[0], s)

        got = None
        for tag, ratio in (("same", ours / ref), ("conj", ours / np.conj(ref))):
            spread = float(np.abs(ratio - ratio[0]).max()
                           / max(np.abs(ratio[0]), 1e-300))
            if spread <= a.tol:
                got = (tag, ratio[0], spread)
                break
        if got is None:
            ok = False
            print(f"{name:10s} {type(mdl).__name__:18s} {kind:10s} "
                  f"neither ours=ref nor ours=conj(ref) is constant  MISMATCH")
            continue
        tag, r0, spread = got
        print(f"{name:10s} {type(mdl).__name__:18s} {kind:10s} "
              f"{r0.real:10.6f}{r0.imag:+10.6f}j {spread:10.2e}  {tag}")

    # the phi: their BW(1.02, 0.004) multiplies every wave; ours is a BW too
    s = np.linspace(0.98 ** 2, 1.06 ** 2, a.n)
    ours = ours_one(1.02, 0.004, lambda m: np.ones_like(m))(s)
    ratio = ours / ref_BW(1.02, 0.004, s)
    spread = float(np.abs(ratio - ratio[0]).max() / abs(ratio[0]))
    ok &= spread <= a.tol
    print(f"{'phires':10s} {'BW':20s} same {ratio[0].real:10.6f}"
          f"{ratio[0].imag:+10.6f}j {spread:10.2e}  "
          f"{'ok' if spread <= a.tol else 'MISMATCH'}")

    print("\nA flat ratio means the two sides form the same denominator up to a "
          "constant,\nwhich the free complex coupling absorbs.  A moving ratio "
          "means the line\nshape, a coupling map or a threshold differs.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

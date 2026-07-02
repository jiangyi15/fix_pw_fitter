"""
B→ρA.ρB helicity amplitude observables.

Shared by ``scripts/fit_constrained.py``, ``scripts/calc_observables.py``,
and ``scripts/eval_rho_observables.py``.

All functions operate on the *resolved* parameter dict from
``Fitter._build_params(x)`` and use Blatt-Weisskopf form factors
matching ``ampfit``'s ``build_fl_table`` convention.
"""
import math
import numpy as np
from ampfit.bw_form_factor import form_factor as bw_form_factor
from ampfit.particle_model.bwr_model import _two_body_cm_mom


# ── B→ρρ coupling names ──────────────────────────────────────────
RHO_LS_NAMES = {
    "gs":  "B->rhoA.rhoB_g_ls_0",
    "gp":  "B->rhoA.rhoB_g_ls_1",
    "gd":  "B->rhoA.rhoB_g_ls_2",
    "gsb": "B->rhoA.rhoB_g_lsbar_0",
    "gpb": "B->rhoA.rhoB_g_lsbar_1",
    "gdb": "B->rhoA.rhoB_g_lsbar_2",
}


# ── Helpers ──────────────────────────────────────────────────────

def read_polar(resolved, base):
    """Read a complex value from the resolved dict (polar convention).

    ``resolved[base + 'r']`` = magnitude *r*
    ``resolved[base + 'i']`` = phase *θ*

    Returns ``r·exp(i·θ)``.
    """
    r = float(resolved[base + "r"])
    theta = float(resolved[base + "i"])
    return r * np.exp(1j * theta)


def bw_form_factor_polar(L, q_phys=None, R=3.0):
    """Convenience: ``bw_form_factor(L, q_phys, q0_ref=1.0, d=R)``.

    If *q_phys* is ``None``, computes the B→ρρ breakup momentum.
    """
    if q_phys is None:
        q_phys = _two_body_cm_mom(5.279, 0.769, 0.769)
    return float(bw_form_factor(L, q_phys, q0_ref=1.0, d=R))


def get_rho_couplings(resolved, R=3.0):
    """Extract B→ρA.ρB LS couplings from resolved dict with form factors.

    Returns:
        ``(gs, gp, gd, gsb, gpb, gdb)`` — complex LS couplings with
        Blatt-Weisskopf barrier factors applied to P- and D-waves.
    """
    _F1 = bw_form_factor_polar(1, R=R)
    _F2 = bw_form_factor_polar(2, R=R)
    gs  = read_polar(resolved, RHO_LS_NAMES["gs"])
    gp  = read_polar(resolved, RHO_LS_NAMES["gp"]) * _F1
    gd  = read_polar(resolved, RHO_LS_NAMES["gd"]) * _F2
    gsb = read_polar(resolved, RHO_LS_NAMES["gsb"])
    gpb = read_polar(resolved, RHO_LS_NAMES["gpb"]) * _F1
    gdb = read_polar(resolved, RHO_LS_NAMES["gdb"]) * _F2
    return gs, gp, gd, gsb, gpb, gdb


def helicity_amplitudes(resolved, R=3.0):
    """Full helicity amplitude decomposition.

    Returns a dict with keys:

        a0, aperp, apara  — B helicity amplitudes (0, ⟂, ∥)
        norm              — |gs|² + |gp|² + |gd|²
        ab0, aperpb, aparab — Bbar helicity amplitudes
        normb             — |gsb|² + |gpb|² + |gdb|²
        gs, gp, gd, gsb, gpb, gdb — raw LS couplings (with F_L)
    """
    gs, gp, gd, gsb, gpb, gdb = get_rho_couplings(resolved, R=R)

    a0 = -math.sqrt(1/3) * gs  + math.sqrt(2/3) * gd
    ap =  math.sqrt(1/3) * gs  - math.sqrt(1/2) * gp + math.sqrt(1/6) * gd
    am =  math.sqrt(1/3) * gs  + math.sqrt(1/2) * gp + math.sqrt(1/6) * gd
    aperp = (ap - am) / math.sqrt(2)
    apara = (ap + am) / math.sqrt(2)
    norm = abs(gs)**2 + abs(gp)**2 + abs(gd)**2

    ab0 = -math.sqrt(1/3) * gsb + math.sqrt(2/3) * gdb
    apb =  math.sqrt(1/3) * gsb - math.sqrt(1/2) * gpb + math.sqrt(1/6) * gdb
    amb =  math.sqrt(1/3) * gsb + math.sqrt(1/2) * gpb + math.sqrt(1/6) * gdb
    aperpb = (apb - amb) / math.sqrt(2)
    aparab = (apb + amb) / math.sqrt(2)
    normb = abs(gsb)**2 + abs(gpb)**2 + abs(gdb)**2

    return dict(a0=a0, aperp=aperp, apara=apara, norm=norm,
                ab0=ab0, aperpb=aperpb, aparab=aparab, normb=normb,
                gs=gs, gp=gp, gd=gd, gsb=gsb, gpb=gpb, gdb=gdb)


# ── Basic observables ─────────────────────────────────────────────

def obs_fL(resolved, R=3.0):
    """Longitudinal (helicity-0) fraction f_L = (|a0|² + |ā0|²) / Σ|g_LS|²."""
    h = helicity_amplitudes(resolved, R=R)
    return float((abs(h["a0"])**2 + abs(h["ab0"])**2) /
                 (h["norm"] + h["normb"]))


def obs_fS(resolved, R=3.0):
    """S-wave LS fraction f_S = (|gs|² + |gsb|²) / Σ|g_LS|²."""
    h = helicity_amplitudes(resolved, R=R)
    return float((abs(h["gs"])**2 + abs(h["gsb"])**2) /
                 (h["norm"] + h["normb"]))


def obs_weak_phase(resolved, R=3.0):
    """Weak phase φ = ½ arg(ā₀/a₀) mod π."""
    h = helicity_amplitudes(resolved, R=R)
    ratio = h["ab0"] / h["a0"]
    return (math.atan2(ratio.imag, ratio.real) / 2) % math.pi


def obs_cp_asym(resolved, R=3.0):
    """CP asymmetry A_CP = (|ā₀|² - |a₀|²) / (|ā₀|² + |a₀|²)."""
    h = helicity_amplitudes(resolved, R=R)
    a0s = abs(h["a0"])**2
    ab0s = abs(h["ab0"])**2
    return float((ab0s - a0s) / (ab0s + a0s + 1e-30))

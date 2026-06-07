"""Helicity amplitude T_{λ,LS}(φ,θ) for decay chains.

Output is organized by (helicity, LS) keys, each value is a list of
Fourier terms: ``coeff · sin(θ/2)^p · cos(θ/2)^q · cos(kφ)``
where ``coeff`` may include ``I`` for imaginary parts (``I·sin(kφ)``).
"""

from __future__ import annotations

import math
import sympy as sp
from dataclasses import dataclass, field
from itertools import product as iproduct


# ============================================================================
#  Constants
# ============================================================================

I = sp.I  # imaginary unit


# ============================================================================
#  Spin/topology helpers
# ============================================================================

def _helicities(J: float):
    """Return list of helicity values for spin J (step 1)."""
    vals = []
    h = -J
    while h <= J + 1e-10:
        vals.append(h)
        h += 1.0
    return vals


# ============================================================================
#  Structured Fourier term
# ============================================================================

@dataclass
class Factor:
    """``func(k·var/2)`` where var is ``theta_i`` or ``phi_i``."""
    name: str      # e.g. "theta_0", "phi_1"
    func: str      # "cos" or "sin"
    k: int         # multiplier of var/2


@dataclass
class AmpTerm:
    """A single term in the helicity amplitude: ``coeff · Π Factor``.

    ``coeff`` is a sympy expression (may include ``I`` for imaginary parts).
    Complex multiplication (``I·I = -1``) is handled automatically by sympy.
    """
    coeff: sp.Expr = sp.Integer(0)
    factors: list[Factor] = field(default_factory=list)


# ============================================================================
#  Output structure
# ============================================================================

HelicityKey = str   # e.g. "0,0,0" for lambda_a, lambda_b, lambda_c
LSKey = str         # e.g. "1,0.5;2,1.0" for L,S pairs per vertex


# ============================================================================
#  Vertex amplitude builder (to be implemented)
# ============================================================================

# (placeholder — will be filled in the next step)

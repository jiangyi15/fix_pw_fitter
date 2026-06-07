"""Helicity amplitude T_{λ,LS}(φ,θ) for decay chains.

Output is organized by (helicity, LS) keys, each value is a list of
Fourier terms: ``coeff · sin(θ/2)^p · cos(θ/2)^q · cos(kφ)``
or imaginary ``· sin(kφ)`` (tracked by im flag).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product as iproduct


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


def _half_int(v: float) -> int:
    return round(2 * v)


def _to_frac_str(v: float) -> str:
    """Format a half-integer spin as a fraction string for computeCGExact."""
    h = _half_int(v)
    if h % 2 == 0:
        return str(h // 2)
    return f"{h}/2" if h > 0 else f"{h}/2"


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
    """A single term in the helicity amplitude.

    The full amplitude is ``coeff · Π Factor``.
    ``im=True`` means this term contributes to the imaginary part.
    """
    coeff: object = 0  # sympy expression (exact)
    im: bool = False
    factors: list[Factor] = field(default_factory=list)


# ============================================================================
#  Output structure
# ============================================================================

HelicityKey = str   # e.g. "0,0,0" for lambda_a, lambda_b, lambda_c
LSKey = str         # e.g. "1,0.5;2,1.0" for L,S pairs per vertex


def build_amplitude_dict() -> dict[HelicityKey, dict[LSKey, list[AmpTerm]]]:
    """Create an empty amplitude dict."""
    return {}


# ============================================================================
#  Vertex amplitude builder (to be implemented)
# ============================================================================

# (placeholder — will be filled in the next step)

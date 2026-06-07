"""Resonance model registry for energy-dependent width computation.

Each resonance model is a class that exposes:
- ``n_channels`` → number of gamma channels
- ``gamma_table(particle, child_masses, n_int, mass_min, mass_max)``
  → ``(n_channels, n_int)`` complex phase-space array

Register a model::

    @register_model("MyModel")
    class MyModel(Model):
        ...
"""

from __future__ import annotations

import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------

_model_registry: dict[str, type] = {}


def register_model(name: str):
    """Decorator that registers a Model subclass under *name*."""
    def decorator(cls):
        _model_registry[name] = cls
        return cls
    return decorator


def get_model(name: str):
    """Look up a model class by name.  Returns None if not found."""
    return _model_registry.get(name)


def list_models():
    """List all registered model names."""
    return list(_model_registry.keys())


# ---------------------------------------------------------------------------
#  Base class
# ---------------------------------------------------------------------------

class Model:
    """Base class for resonance width models.

    Subclasses should override ``n_channels`` and ``gamma_table``.
    The particle's ``props`` dict provides model-specific parameters
    (e.g. daughter masses for ``BW``, channel list for ``Flatte``).
    """

    @staticmethod
    def n_channels(particle: Any) -> int:
        """Number of gamma channels for this resonance."""
        raise NotImplementedError

    @staticmethod
    def gamma_table(particle: Any, child_masses: list[float],
                    n_int: int, mass_min: float, mass_max: float
                    ) -> np.ndarray:
        """Return ``(n_channels, n_int)`` complex phase-space table.

        Parameters
        ----------
        particle : Particle
            The resonance particle with its ``props`` dict.
        child_masses : list[float]
            Masses of the daughters (for phase-space calculation).
        n_int : int
            Number of interpolation bins.
        mass_min, mass_max : float
            Mass grid range.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
#  Phase-space helper
# ---------------------------------------------------------------------------

def phase_space(m: np.ndarray, m1: float, m2: float) -> np.ndarray:
    """Two-body phase space factor ρ(m) = √((1−s₋/m²)(1−s₊/m²)).

    where s₊ = (m₁+m₂)² and s₋ = (m₁−m₂)².  Returns 0 below threshold.
    """
    s_plus = (m1 + m2) ** 2
    s_minus = (m1 - m2) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.sqrt(np.clip((1.0 - s_minus / m ** 2) * (1.0 - s_plus / m ** 2),
                                 0.0, None))
    result[m < m1 + m2] = 0.0
    result[~np.isfinite(result)] = 0.0
    return result


# ---------------------------------------------------------------------------
#  Built-in models
# ---------------------------------------------------------------------------

@register_model("one")
class OneModel(Model):
    """Unity width model — gamma_table = 1 everywhere.

    The propagator reduces to a constant-width Breit-Wigner:
        A = 1 / (m₀² − m² − i·m₀·g₀)
    """

    @staticmethod
    def n_channels(particle) -> int:
        return 1

    @staticmethod
    def gamma_table(particle, child_masses, n_int, mass_min, mass_max):
        return np.ones((1, n_int), dtype=np.complex64)


@register_model("BW")
class BWModel(Model):
    """Constant Breit-Wigner — width does NOT run (Γ = Γ₀ constant).

    gamma_table returns all ones; the width is just ``g0 * 1``.
    """

    @staticmethod
    def n_channels(particle) -> int:
        return 1

    @staticmethod
    def gamma_table(particle, child_masses, n_int, mass_min, mass_max):
        return np.ones((1, n_int), dtype=np.complex64)


@register_model("BWR")
class BWRModel(Model):
    """Running-width Breit-Wigner — energy-dependent phase-space width.

    gamma_table returns the two-body phase space factor ρ(m) for the
    given daughter masses.  The running width is ``g0 · ρ(m)``.
    """

    @staticmethod
    def n_channels(particle) -> int:
        return 1

    @staticmethod
    def gamma_table(particle, child_masses, n_int, mass_min, mass_max):
        m_grid = np.linspace(mass_min, mass_max, n_int, endpoint=False)
        m1, m2 = child_masses[0], child_masses[1] if len(child_masses) > 1 else child_masses[0]
        rho = phase_space(m_grid, m1, m2)
        return rho.astype(np.complex64).reshape(1, -1)


@register_model("Flatte")
class FlatteModel(Model):
    """Flatté model — multiple channels each with its own phase space."""

    @staticmethod
    def n_channels(particle) -> int:
        return len(particle.props.get("channels", []))

    @staticmethod
    def gamma_table(particle, child_masses, n_int, mass_min, mass_max):
        channels = particle.props.get("channels", [])
        if not channels:
            return np.zeros((0, n_int), dtype=np.complex64)
        m_grid = np.linspace(mass_min, mass_max, n_int, endpoint=False)
        table = np.zeros((len(channels), n_int), dtype=np.complex64)
        for i, ch in enumerate(channels):
            m1, m2 = ch.get("mass1", 0.0), ch.get("mass2", 0.0)
            table[i] = phase_space(m_grid, m1, m2)
        return table

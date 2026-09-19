"""
Barrier factors (form factors) — pluggable and id-keyed.

A barrier factor is a :class:`BarrierFactor` instance carrying its orbital
angular momentum *L* and shape parameters (reference momentum ``q0_ref``,
radius ``d``).  :meth:`BarrierFactor.get_params` returns the constructor
kwargs that reconstruct it; :meth:`BarrierFactor.get_id` hashes those.  The
id is used both for de-duplication in the kernel config (the ``fl_table``
rows) and as the ``fl_type`` row index — so two barriers with different ``d``
(or any extra parameter) are two distinct ids and get two table rows.

New types are registered as classes; add parameters via ``get_params``::

    from ampfit.bw_form_factor import BarrierFactor, register_barrier
    import numpy as np

    @register_barrier("myname")
    class MyBarrier(BarrierFactor):
        def __init__(self, L=0, q0_ref=1.0, d=3.0, alpha=1.0):
            super().__init__(L, q0_ref=q0_ref, d=d)
            self.alpha = alpha

        def get_params(self):          # feeds get_id + kc fl_specs
            return {**super().get_params(), "alpha": self.alpha}

        def factor(self, q):
            return np.exp(-self.alpha * (q * self.d) ** 2)

The default type is ``"bw"`` — Blatt-Weisskopf,
``F_L(q) = q^L · B'_L(q, q0_ref, d)`` with the TFPWA normalisation
``F_L(q0_ref) = q0_ref^L``.  ``"exp"`` is a Gaussian barrier
``F_L(q) = exp(-(q·d)²/2)`` (no ``q^L``).
"""
import numpy as np

__all__ = [
    "BarrierFactor", "BARRIER_MODELS", "DEFAULT_BARRIER", "register_barrier",
    "build_barrier", "barrier_names", "form_factor", "barrier_ratio",
    "blatt_weisskopf", "exponential",
]

BARRIER_MODELS = {}
DEFAULT_BARRIER = "bw"


# ── Registry ──────────────────────────────────────────────────────

def register_barrier(name):
    """Class decorator: register a :class:`BarrierFactor` subclass as *name*."""
    def _f(cls):
        if name in BARRIER_MODELS and BARRIER_MODELS[name] is not cls:
            raise ValueError(f"barrier type {name!r} already registered")
        if not issubclass(cls, BarrierFactor):
            raise TypeError(f"{cls!r} is not a BarrierFactor subclass")
        cls.name = name
        BARRIER_MODELS[name] = cls
        return cls
    return _f


def build_barrier(name=DEFAULT_BARRIER, L=0, q0_ref=1.0, d=3.0, **kwargs):
    """Instantiate the registered barrier type *name*.

    Extra *kwargs* go to the class constructor (custom barriers may add
    parameters; fold them into :meth:`BarrierFactor.get_id`).
    """
    try:
        cls = BARRIER_MODELS[name]
    except KeyError:
        raise KeyError(
            f"unknown barrier type {name!r}; "
            f"registered: {barrier_names()}") from None
    return cls(L, q0_ref=q0_ref, d=d, **kwargs)


def barrier_names():
    """Registered barrier type names (sorted)."""
    return sorted(BARRIER_MODELS)


# ── Base class ────────────────────────────────────────────────────

class BarrierFactor:
    """Base barrier shape for orbital angular momentum *L*.

    Subclasses override :meth:`factor`.  Anything that changes the shape must
    enter :meth:`get_id` so it maps to its own table row.
    """
    name = DEFAULT_BARRIER

    def __init__(self, L=0, q0_ref=1.0, d=3.0):
        self.L = int(L)
        self.q0_ref = float(q0_ref)
        self.d = float(d)

    # identity — hashable, used as the kernel-config table key
    def get_params(self):
        """Constructor kwargs (beyond ``L``) reconstructing this barrier.

        Subclasses that add parameters must extend this::

            def get_params(self):
                return {**super().get_params(), "alpha": self.alpha}

        It feeds both :meth:`get_id` and the kernel-config ``fl_specs``, so
        distinct parameters get distinct ``fl_table`` rows.
        """
        return {"type": self.name, "q0_ref": self.q0_ref, "d": self.d}

    def get_id(self):
        """Hashable identity of this barrier variant."""
        return (self.L, tuple(sorted(self.get_params().items())))

    def __eq__(self, other):
        return (isinstance(other, BarrierFactor)
                and self.get_id() == other.get_id())

    def __hash__(self):
        return hash(self.get_id())

    def __repr__(self):
        return (f"{type(self).__name__}(L={self.L}, "
                f"{ {k: v for k, v in self.get_params().items() if k != 'type'} })")

    # shape
    def factor(self, q):
        raise NotImplementedError(
            f"{type(self).__name__} must implement factor(q)")

    def __call__(self, q):
        return self.factor(q)


# ── Blatt-Weisskopf helpers ───────────────────────────────────────

def _bprime_poly(L, z):
    """Blatt-Weisskopf barrier polynomial at order *L*.

    ``z = (q·R)²`` where *q* is the breakup momentum and *R* the
    meson radius.
    """
    coeff = {
        0: [1.0],
        1: [1.0, 1.0],
        2: [1.0, 3.0, 9.0],
        3: [1.0, 6.0, 45.0, 225.0],
        4: [1.0, 10.0, 135.0, 1575.0, 11025.0],
        5: [1.0, 15.0, 315.0, 6300.0, 99225.0, 893025.0],
    }
    c = coeff.get(L, [1.0])
    val = np.zeros_like(z)
    for ci in c:
        val = val * z + ci
    return val


def barrier_ratio(L, q, q0, d=3.0):
    r"""Blatt-Weisskopf barrier ratio :math:`B'_L(q, q_0, d)`.

    .. math::

        B'_L(q, q_0, d) = \sqrt{\frac{P_L(z_0)}{P_L(z)}}
        \qquad z = (q d)^2

    where :math:`P_L` is the barrier polynomial.

    Args:
        L: orbital angular momentum (0 … 5).
        q: breakup momentum (GeV), scalar or array.
        q0: reference breakup momentum (GeV).
        d: meson radius (GeV\ :sup:`-1`).  Default 3.0.

    Returns:
        Barrier ratio :math:`B'_L` (dimensionless, = 1 at q = q0).
    """
    z = (q * d) ** 2
    z0 = (q0 * d) ** 2
    return np.sqrt(_bprime_poly(L, z0)) / np.sqrt(_bprime_poly(L, z))


# ── Built-in barrier types ────────────────────────────────────────

@register_barrier("bw")
class BlattWeisskopf(BarrierFactor):
    r"""Blatt-Weisskopf :math:`F_L(q) = q^L \, B'_L(q, q_0^{\mathrm{ref}}, d)`.

    ``F_0(q) = 1``; normalised so that
    :math:`F_L(q_0^{\mathrm{ref}}) = (q_0^{\mathrm{ref}})^L`.
    """
    def factor(self, q):
        scalar_in = np.ndim(q) == 0
        qa = np.asarray(q, dtype=np.float64)
        if self.L == 0:
            out = np.ones_like(qa)
        else:
            out = (qa ** self.L) * barrier_ratio(self.L, qa, self.q0_ref, self.d)
        return float(out) if scalar_in else out


@register_barrier("exp")
class ExponentialBarrier(BarrierFactor):
    r"""Gaussian barrier :math:`F_L(q) = \exp(-(q\,d)^2 / 2)` (no ``q^L``)."""
    def factor(self, q):
        scalar_in = np.ndim(q) == 0
        qa = np.asarray(q, dtype=np.float64)
        out = np.exp(-0.5 * (qa * self.d) ** 2)
        return float(out) if scalar_in else out


# ── Functional convenience (backward compatible) ─────────────────

blatt_weisskopf = BlattWeisskopf
exponential = ExponentialBarrier


def form_factor(L, q, q0_ref=1.0, d=3.0, kind=DEFAULT_BARRIER):
    """``build_barrier(kind, L, q0_ref, d).factor(q)`` (default ``"bw"``)."""
    return build_barrier(kind, L=L, q0_ref=q0_ref, d=d).factor(q)

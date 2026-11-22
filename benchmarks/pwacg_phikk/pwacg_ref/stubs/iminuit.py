"""Import stub for `iminuit`, which PWACG's generated fit object imports at module
scope.  Nothing on the TIMED path uses it (the reference harness times
`PWAFunc.data_likelihood_kk` / `mc_likelihood_kk` and their `jax.grad`), and
installing minuit for a timing benchmark would add a dependency that can move the
numbers we are not measuring.  A fit that used `iminuit` would fail loudly here
rather than silently return a wrong number.
"""


def _unavailable(*_a, **_kw):
    raise RuntimeError(
        "iminuit is stubbed in this benchmark: PWACG imports it at module scope "
        "but only its Migrad driver uses it, and this harness times the JAX "
        "likelihood, not the optimiser")


class Minuit:
    def __init__(self, *a, **kw):
        _unavailable()


def minimize(*a, **kw):
    """``from iminuit import minimize`` — imported at module scope, used by
    nothing on the timed path."""
    _unavailable()


class Migrad:
    def __init__(self, *a, **kw):
        _unavailable()

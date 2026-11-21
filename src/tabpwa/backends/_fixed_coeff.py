"""``fixed_coeff`` — the coefficient-mask lifecycle as a delegating backend.

A mask-aware kernel is immutable per mask: the mask decides the wave
permutation, the free-only ``gamma``/CSR and the context, so changing it means
building a new kernel.  This wrapper owns that lifecycle, in ONE place, for any
kernel backend that accepts ``fixed_cols`` + ``fixed_values``:

* it takes the mask as COLUMNS + VALUES (``set_fixed_cols(cols, values)``;
  the name helper is ``fixed_coeff_spec.resolve_fixed_cols``) and swaps the
  base kernel when it changes.  The declaration travels as the base's
  CONSTRUCTOR kwargs (``fixed_cols`` + ``fixed_values``) — never through the
  kernel config, which cannot carry the values; a base that does not take them
  is mask-agnostic and simply gets neither.  The base is rebuilt on every
  ``set_fixed_cols`` call (one reload; constraints are applied once);
* it retains the loaded samples so a rebuilt kernel gets them back without the
  caller re-supplying anything — each handle is re-loaded lazily on its first
  compute under the new mask;
* the wrapper starts on the PARENT path (no mask), which is the state a
  ``Fitter`` is constructed in; the mask is set afterwards, once constraints
  are known, with ``set_fixed_cols`` — or, in one call, with
  ``set_fixed_coeffs(fitter)``, which derives the columns from the fitter's
  constraint state and MODEL (``model.fixed_coeff_columns``) and then applies
  them; the explicit-columns form stays ``set_fixed_cols``.  The name says
  COEFFICIENTS on purpose: ``Fitter.set_fixed`` is the parameter setter;
* a failed rebuild rolls back to the previous base, so a rejected mask cannot
  leave the fitter unusable.

The mask is deliberately NOT a constructor argument (passing one is rejected):
there would then be two ways to set it.  ``set_fixed_cols`` is the only one.

This module knows nothing about any particular kernel: it sees a base spec and
the same ``den_names`` vocabulary the mask is declared in.
"""
import inspect
import warnings as _warnings
import weakref as _weakref

import numpy as np

from tabpwa.fixed_coeff_spec import (MaskNotEngaged,
                                     resolve_mask_and_values,
                                     warn_unfixed_mask)

from .core import ComputeBackend, create_backend, register_backend

#: constructor kwargs that would be a second way to set the mask
_MASK_KW = ("mask", "fixed_cols", "fixed_values", "fixed_names",
            "fixed_param_map")

#: "the caller said nothing" — distinct from an explicit ``values=None``
_VALUES_UNSET = object()

#: raised when a non-empty mask is declared without its values: the wrapper's
#: cache is filled at `load_data`, so there is nothing to read them from later
_NEEDS_VALUES = (
    "fixed_coeff.set_fixed_cols needs the values of the columns it declares "
    "(the coefficients are read from these, not from the first compute): pass "
    "values=<{col: value} | coeff vector>, or use set_fixed_coeffs(fitter), "
    "which derives both. To clear the mask pass None.")


def _accepts_mask_kwargs(base_spec, model=None):
    """Whether the base's constructor takes ``fixed_cols`` / ``fixed_values``.

    A base that does not (``numpy_flat``, …) is mask-AGNOSTIC: the mask is a
    documented no-op there, so the wrapper does not pass the kwargs.  The
    signature is inspected once, never the behaviour.
    """
    if isinstance(base_spec, dict):
        name = base_spec.get("name")
    else:
        name = base_spec
    if not isinstance(name, str):
        return False
    try:
        from .core import backend_class
        cls = backend_class(name, model)
        params = inspect.signature(cls.__init__).parameters
    except Exception:
        return False
    if "fixed_cols" in params or "fixed_values" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in params.values())


def _base_n_fix(base):
    """The base's ``n_fix`` (directly, or on its ``.kernel``), else None.

    It is how the wrapper asks "did this base UNDERSTAND the mask?" — a
    mask-agnostic base reports neither and is a documented no-op (no warning).
    """
    for obj in (base, getattr(base, "kernel", None)):
        n = getattr(obj, "n_fix", None)
        if n is not None:
            return int(n)
    return None


def _free_any(real):
    """Release an opaque base handle; a plain dict (the numpy backend) has
    nothing to release."""
    free = getattr(real, "free", None)
    if callable(free):
        free()


def mask_equal(a, b):
    """``None``-aware mask equality (``None`` = the parent path)."""
    if a is None or b is None:
        return a is None and b is None
    return bool(np.array_equal(a, b))


class _FixedCoeffHandle:
    """A retained sample + its base handle for the CURRENT mask.

    ``owner`` is a WEAK reference to the wrapper: ``free`` pops this handle's
    record there, so the retained sample (its numpy dict included) is
    collectable as soon as the caller releases the handle — otherwise every
    ``set_phsp``/``set_data`` cycle would leave a record (and its sample)
    behind, GBs at the production size.
    """

    __slots__ = ("data", "real", "stamp", "freed", "owner", "__weakref__")

    def __init__(self, data, real, stamp, owner=None):
        self.data = data
        self.real = real
        self.stamp = None if stamp is None else stamp.copy()
        self.freed = False
        self.owner = None if owner is None else _weakref.ref(owner)

    def free(self):
        if self.real is not None:
            _free_any(self.real)
            self.real = None
        self.freed = True
        self.data = None            # release the retained sample too
        owner = self.owner() if self.owner is not None else None
        if owner is not None:
            owner._records.pop(self, None)
        self.owner = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


@register_backend("fixed_coeff", model=None, tags=("delegate",))
class FixedCoeffBackend(ComputeBackend):
    """The mask lifecycle over any base backend that accepts ``fixed_cols``.

    ``base`` defaults to the model's ``"default"`` backend (pass a
    mask-aware kernel spec to make the mask do something).  The base is built
    MASK-LESS here, so a ``Fitter`` starts on the parent path exactly as
    before; ``set_fixed_cols(cols, values)`` rebuilds it with those two as
    constructor kwargs.  A backend that does not take them simply ignores the
    mask — that is a valid base too, and the mask is then a NO-OP (reported by
    ``wrapper.mask`` and the base's own name; no warning is emitted).  ``set_fixed_cols`` is the only way to set a
    mask; it frees the base and builds a new one for that mask, and every
    retained sample is re-loaded lazily on its next compute.
    """

    def __init__(self, kernel_config, base="default", model=None, **kw):
        bad = [k for k in _MASK_KW if k in kw]
        if bad:
            raise ValueError(
                "fixed_coeff: the mask is not a constructor argument "
                f"({bad}); the backend starts on the parent path — call "
                "set_fixed_cols(...) after construction (constraints are "
                "applied after the backend is built)")
        if not isinstance(base, (str, dict)):
            raise ValueError(
                "fixed_coeff: base must be a backend name or spec dict "
                "(the model's \"default\" is used when omitted)")
        self._kc = kernel_config
        self._base_spec = base
        self.model = model
        self._kw = dict(kw)
        self.mask = None
        #: the mask's values (a full `coeff` vector).  ALWAYS set together with
        #: a mask by this wrapper — the eager path is the only wrapper path —
        #: and None only on the parent path.
        self.values = None
        self.creates = 0            # base instances built (attempts)
        self.loads = 0              # base.load_data calls (sample, mask)
        self._records = _weakref.WeakKeyDictionary()   # handle -> None
        self._base_gen = 0
        self._accepts_values = _accepts_mask_kwargs(self._base_spec, model)
        self.base = self._make_base(None)

    # ── the base ────────────────────────────────────────────────────────
    def _mask_kwargs(self, mask, values):
        """The base's constructor kwargs for a mask (and its values).

        The mask travels as ARGUMENTS (``fixed_cols=`` + ``fixed_values=``),
        never through the kernel config: a config key cannot carry the values,
        and keeping both spellings is exactly the second API this replaces.
        A base whose constructor does not take them is mask-AGNOSTIC — a
        documented no-op, so the kwargs are simply not passed.
        """
        if mask is None or not self._accepts_values:
            return {}
        return {"fixed_cols": [int(i) for i in np.nonzero(mask)[0]],
                "fixed_values": np.asarray(values, np.float64)}

    def _make_base(self, mask, values=None):
        self.creates += 1           # attempts, so a rollback is visible
        # the base's own "mask cached nothing" warning would duplicate the
        # wrapper's (below), so it is suppressed here and re-emitted once
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", MaskNotEngaged)
            return create_backend(self._base_spec, self._kc,
                                  model=self.model,
                                  **{**self._kw,
                                     **self._mask_kwargs(mask, values)})

    @property
    def resolution_size(self):
        return self.base.resolution_size

    def __getattr__(self, name):
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    # ── the mask: the only way to set one ────────────────────────────────
    def set_fixed_cols(self, fixed_cols=None, values=_VALUES_UNSET):
        """``set_fixed_cols(cols, values)`` — declare the columns AND their values.

        The two go together: the base fills its ``Afix`` cache at ``load_data``
        from the values, so a mask without them has no coefficient to read and
        the FIRST compute is guarded from the start.  ``values`` is
        ``{column: value}``, the full ``coeff`` vector, or one value per
        declared column.  :meth:`set_fixed_coeffs` derives both from a fitter
        (``resolve_fixed_cols`` in :mod:`tabpwa.fixed_coeff_spec` is the name
        helper when the caller speaks names).

        ``set_fixed_cols(None)`` (or ``[]``) is the one exception — clearing
        needs no values.

        The mask travels as the base's CONSTRUCTOR kwargs (``fixed_cols`` +
        ``fixed_values``), never through the kernel config: a config key cannot
        carry the values.  EVERY call rebuilds the base (one reload) —
        constraints are applied once, so that is the right trade; the
        ``creates`` counter makes it visible.  No sample is touched here: the
        retained samples re-load (and fill their cache) on their next compute.
        """
        basis = self._kc.get("basis_table")
        if basis is None:
            raise ValueError(
                "fixed_coeff: the kernel config has no `basis_table` — a "
                "coefficient mask needs the flat basis, i.e. the model's "
                "coefficients; use `amp_model: pwa_gen`")
        n_basis = int(np.asarray(basis).shape[0])
        # "values but no columns" is checked BEFORE the empty-cols branch, so
        # `set_fixed_cols(None, values)` and `set_fixed_cols([], values)` agree
        if (fixed_cols is None or np.asarray(
                list(fixed_cols), np.int64).ravel().size == 0):
            if values is not _VALUES_UNSET and values is not None:
                raise ValueError(
                    "fixed_coeff: values were given but no columns — nothing "
                    "to fill")
            new_mask, vec = None, None
        else:
            cols = np.asarray(list(fixed_cols), np.int64).ravel()
            if values is _VALUES_UNSET or values is None:
                raise ValueError(_NEEDS_VALUES)
            new_mask, vec = resolve_mask_and_values(
                cols, values, n_basis, name="fixed_coeff")
        old_mask, old_values = self.mask, self.values
        self._unload_records()              # their handles belong to the base
        self.base.free()
        self.base = None
        try:
            self.base = self._make_base(new_mask, vec)
        except BaseException:
            # rollback: the previous mask (and its values) must stay usable
            self.base = self._make_base(old_mask, old_values)
            raise
        self.mask = None if new_mask is None else new_mask.copy()
        self.values = None if vec is None else vec.copy()
        self._base_gen += 1
        # a mask that caches nothing is a silent no-op on every kernel that
        # honours it (the base reports `n_fix`): say so ONCE, here, naming the
        # wave that still reads a free column.  A mask-agnostic base has no
        # `n_fix` and is a documented no-op, so it stays quiet.
        if self.mask is not None and _base_n_fix(self.base) is not None:
            warn_unfixed_mask(self._kc["gamma_matrix"], self.mask,
                              self._kc.get("den_names"), name="fixed_coeff",
                              stacklevel=3)

    def set_fixed_coeffs(self, fitter, *, how="transform", x=None):
        """Derive the FIXED coefficient columns from the fitter and apply them.

        This declares which coefficient COLUMNS are fixed — a **performance
        hint** telling the base kernel which basis weights can never move —
        and NOT parameters: :meth:`Fitter.set_fixed` is that one, and pins
        parameter VALUES.  The name says what is set, so the two are never
        confused.

        ``cols = fitter.model.fixed_coeff_columns(fitter.cm, x=..., how=how)``,
        then :meth:`set_fixed_cols` with those columns — the wrapper knows
        nothing about the model.  A model without a flat coefficient basis
        raises its own "use ``amp_model: pwa_gen``" error, and an unknown
        ``how`` raises the model's ``ValueError``; both propagate unchanged,
        before the base is touched.

        It ALSO supplies the VALUES: ``build_params(x)["coeff"][cols]``, so the
        base caches ``Afix`` at ``load_data`` and the first compute is guarded
        from the start.  ``x`` defaults to ``fitter.initial_values(seed=1)``
        for BOTH ``how`` forms (the jacobian walk always needed a point; the
        transform walk now needs one for the values).  Pass ``x`` explicitly
        when the fit will start somewhere else — the values ARE that point.

        ``how="transform"`` (default) is the constraint-graph walk; an unknown
        ``how`` raises the model's ``ValueError``.

        An empty derived set returns to the parent path (mask-less), exactly
        like ``set_fixed_cols(fixed_cols=[])``; ``set_fixed_cols(None)`` stays
        the explicit "no mask" form.
        """
        if x is None:
            x = fitter.initial_values(seed=1)
        cols = fitter.model.fixed_coeff_columns(fitter.cm, x=x, how=how)
        if not len(cols):
            self.set_fixed_cols(None)
            return
        coeff = np.asarray(fitter.build_params(x)[0]["coeff"], np.float64)
        self.set_fixed_cols(cols,
                            {int(c): float(coeff[int(c)]) for c in cols})

    # ── data ────────────────────────────────────────────────────────────
    def _unload_records(self):
        for rec in list(self._records):
            if rec.real is not None:
                _free_any(rec.real)
                rec.real = None
            rec.stamp = None

    def load_data(self, data):
        """Retain the sample and load it on the current base."""
        real = self.base.load_data(data)
        self.loads += 1
        h = _FixedCoeffHandle(data, real, self.mask, owner=self)
        # The record IS the handle, so it must be the KEY only: a
        # WeakKeyDictionary holds its VALUE strongly, and a value that is also
        # the key would keep that key alive forever (the very leak this store
        # exists to avoid).  Hence key -> None and the handle carries the
        # record; dropping the caller's reference then collects it, runs
        # `_FixedCoeffHandle.__del__` -> `free()`, and releases the device copy.
        self._records[h] = None
        return h

    def _ensure(self, handle):
        rec = handle
        if handle not in self._records or rec.freed:
            raise ValueError(
                "fixed_coeff: unknown or freed handle (this handle was not "
                "loaded by this backend, or it was freed)")
        if rec.real is None or not mask_equal(rec.stamp, self.mask):
            if rec.real is not None:
                _free_any(rec.real)
            rec.real = self.base.load_data(rec.data)
            self.loads += 1
            rec.stamp = None if self.mask is None else self.mask.copy()
        return rec

    def compute(self, params, handle, norm=None, return_p=True):
        rec = self._ensure(handle)
        return self.base.compute(params, rec.real, norm=norm,
                                 return_p=return_p)

    def free(self):
        """Release every retained sample's handle and the base; idempotent."""
        for rec in list(self._records):
            rec.free()
        self._records.clear()
        if getattr(self, "base", None) is not None:
            self.base.free()
            self.base = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

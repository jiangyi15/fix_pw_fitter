"""The FIXED coefficient declaration — ONE spelling, shared by the kernels and
the ``fixed_coeff`` wrapper.

A fixed-coefficient mask is declared as **columns + their values**:

    CUDAFlatPwaGenS0F(kc, fixed_cols=cols, fixed_values=vals)   # both, or
    CUDAFlatPwaGenS0F(kc)                                       # neither
    set_fixed_cols(cols, values)      # the wrapper; set_fixed_cols(None) clears

The values live in the kernel context (which is per mask), ``load_data`` fills
the ``Afix`` cache from them and every ``compute`` compares the call's
``coeff`` against them — so a mask without values (or values without a mask) is
an error, at both levels.  There is no int8-array constructor, no name
constructor and no config key: those were three more spellings of the same
state, and a config key cannot carry the values.

NAMES are a helper, not an API: :func:`resolve_fixed_cols` maps ``den_names``
(optionally through a ``{parameter: [column, ...]}`` map, with the alias guard)
to column indices — the caller then passes those as ``fixed_cols`` and the
values still come from its own ``coeff``.

This module imports numpy only and knows no kernel.
"""
import warnings

import numpy as np


class MaskNotEngaged(UserWarning):
    """A mask was declared but it cannot cache anything.

    Every wave's denominator row still reads a column the mask left free, so
    the wave axis has no fixed block at all: the kernel runs its parent path
    and the declaration is a silent no-op.  Its own class (rather than a bare
    ``UserWarning``) lets the ``fixed_coeff`` wrapper suppress the kernel's
    duplicate and emit exactly ONE warning per ``set_fixed_cols`` call.
    """


def mask_from_cols(cols, n_basis, prefix="mask"):
    """``int8 [n_basis]`` from column indices, range-checked.

    ``fixed_cols`` is a list of COLUMN INDICES.  A length-``n_basis`` boolean
    (or ``int8`` 0/1) array is the MASK in another spelling, and reading it as
    indices would silently install a different mask — with a full-length values
    vector nothing would raise — so it is refused with the spelling it needs.
    """
    arr = np.asarray(list(cols) if not isinstance(cols, np.ndarray) else cols)
    if arr.dtype.kind == "b":
        raise ValueError(
            f"{prefix}fixed_cols is a list of COLUMN INDICES, got a boolean "
            "array — pass the column indices (or a {col: value} mapping via "
            "fixed_values)")
    if (arr.dtype.kind in "iu" and arr.size == int(n_basis)
            and np.all((arr == 0) | (arr == 1))):
        raise ValueError(
            f"{prefix}fixed_cols is a list of COLUMN INDICES, got a "
            f"length-{int(n_basis)} 0/1 vector — that is the MASK, not the "
            "columns; pass np.nonzero(mask)[0]")
    a = np.asarray(arr, np.int64).ravel()
    bad = a[(a < 0) | (a >= int(n_basis))]
    if bad.size:
        raise ValueError(
            f"{prefix}fixed columns outside [0, n_basis = {int(n_basis)}): "
            f"{bad.tolist()}")
    m = np.zeros(int(n_basis), np.int8)
    m[a] = 1
    return m


def resolve_fixed_cols(den_names, names=None, param_map=None):
    """Column indices for a NAME declaration (a helper, not the API).

    ``names`` are entries of ``den_names`` — a repeated name fixes EVERY column
    carrying it — or keys of ``param_map = {parameter: [column name, ...]}``,
    which says which columns a fitted parameter drives.

    Without a map a name is a COLUMN, which is unsafe for an ALIASED parameter
    (one fitted mass that moves two charge sectors' columns): the declaration
    looks complete and the cache goes stale.  So a name-only declaration warns,
    and a *partial* declaration of a mapped parameter RAISES
    (:func:`_check_alias_cover`).

    Returns a sorted ``int64`` array of COLUMN INDICES.  It cannot supply the
    VALUES — those come from the caller's ``coeff`` — so the caller passes both
    to the kernel::

        cols = resolve_fixed_cols(kc["den_names"], names=["rho_mass"])
        params = fitter.build_params(x)[0]
        set_fixed_cols(cols, {int(c): params["coeff"][c] for c in cols})
    """
    den = [str(n) for n in np.asarray(den_names).ravel()]
    if names is None and param_map is None:
        return np.zeros(0, np.int64)
    if names is None and param_map is not None:
        names = list(param_map)
    groups = (None if param_map is None else
              _normalise_param_map(param_map, den, "resolve_fixed_cols: "))
    if groups is None:
        warnings.warn(
            "resolve_fixed_cols: a name mask has no param_map, so it declares "
            "COLUMNS, not fitted parameters — an aliased parameter (e.g. a "
            "mass that moves both charge sectors' columns) can be "
            "under-covered. Pass param_map=, or use explicit column indices.",
            stacklevel=2)
    idx = []
    for n in map(str, names):
        if n in den:
            idx.extend(i for i, d in enumerate(den) if d == n)
        elif groups is not None and n in groups:
            idx.extend(groups[n])
        else:
            raise ValueError(
                f"resolve_fixed_cols: unknown names {[n]}; the available "
                f"den_names are {den}"
                + ("" if groups is None else
                   f" and the parameters are {sorted(groups)}"))
    idx = sorted(set(int(i) for i in idx))
    if groups is not None:
        _check_alias_cover(idx, groups, den, "resolve_fixed_cols: ")
    return np.asarray(idx, np.int64)


def _normalise_param_map(param_map, den, prefix):
    if not hasattr(param_map, "items"):
        raise ValueError(
            prefix + "param_map must be a {parameter name: [column name, ...]} "
            "mapping")
    col_of = {}
    for i, d in enumerate(den):
        col_of.setdefault(d, []).append(i)
    groups = {}
    for param, cols in param_map.items():
        cols = [str(c) for c in cols]
        unknown = [c for c in cols if c not in col_of]
        if unknown:
            raise ValueError(
                prefix + f"param_map entry {param!r} names unknown columns "
                f"{unknown}")
        got = []
        for c in cols:
            got.extend(col_of[c])
        groups[str(param)] = sorted(set(got))
    return groups


def _check_alias_cover(fixed_idx, groups, den, prefix):
    F = set(int(i) for i in fixed_idx)
    for param, cols in groups.items():
        C = set(cols)
        hit = F & C
        if hit and hit != C:
            missing = sorted(C - F)
            raise ValueError(
                prefix + "the declaration covers only part of the columns "
                f"driven by {param!r}: it moves {[den[i] for i in cols]} but "
                f"{[den[i] for i in missing]} are not declared fixed. They "
                "share that fitted parameter (an aliased charge/mirror "
                "sector), so the cache would go stale. Declare every alias, "
                "name the parameter, or pass fixed_cols.")


def fixed_values_vector(values, cols, n_basis, prefix="mask"):
    """Validate a VALUES declaration and expand it to a full coeff vector.

    ``cols`` are the declared column indices.  Accepted forms:

    * ``{column: value}`` — every key must be a declared column;
    * a length-``n_basis`` vector — the full coefficient vector, read at the
      declared columns;
    * a length-``len(cols)`` vector — one value per declared column, in
      ascending column order.

    Returns ``np.float64 [n_basis]``, so the kernel can be handed one pointer
    and read it exactly like ``coeff`` — no index remap, no forked fill
    kernel.  Values for an EMPTY declaration are an error (nothing to fill).
    """
    c = np.asarray(list(cols), np.int64).ravel()
    if values is None:
        raise ValueError(
            prefix + "the coefficient values are required with the columns "
            "they belong to (the cache is filled at `load_data`)")
    if c.size == 0:
        raise ValueError(
            prefix + "fixed VALUES were given but no column is declared — "
            "declare the columns they belong to")
    if hasattr(values, "items"):
        out = np.zeros(int(n_basis), np.float64)
        given = set()
        for k, v in values.items():
            col = int(k)
            if not (0 <= col < int(n_basis)):
                raise ValueError(
                    prefix + f"fixed value column {col} is outside "
                    f"[0, n_basis = {int(n_basis)})")
            if not np.isin(col, c):
                raise ValueError(
                    prefix + f"fixed value column {col} is not declared fixed "
                    f"— the declaration is {c.tolist()}")
            out[col] = float(v)
            given.add(col)
        # a mapping is the declaration written twice: EVERY declared column
        # needs its value (an omitted one would silently become 0.0 and the
        # strict guard would later fail with the wrong message)
        missing = [int(col) for col in c if int(col) not in given]
        if missing:
            raise ValueError(
                prefix + f"fixed values do not cover the declaration: no value "
                f"for column(s) {missing} of {c.tolist()}")
    else:
        arr = np.asarray(values, np.float64).ravel()
        if arr.size == int(n_basis):
            out = arr.copy()
        elif arr.size == c.size:
            out = np.zeros(int(n_basis), np.float64)
            out[c] = arr
        else:
            raise ValueError(
                prefix + f"fixed values have {arr.size} entries; expected "
                f"n_basis = {int(n_basis)} or the {c.size} declared column(s)")
    if not np.all(np.isfinite(out[c])):
        raise ValueError(
            prefix + "fixed values must be finite (a NaN/inf would make the "
            "strict guard fail on every compute)")
    return out


def resolve_mask_and_values(fixed_cols, fixed_values, n_basis,
                            name="masked tile"):
    """THE one declaration: ``(int8 mask or None, full values vector or None)``.

    ``fixed_cols`` and ``fixed_values`` go together: neither (the parent path)
    or both.  A mask without values raises (the cache is filled at
    ``load_data``, so there is no later coefficient read to fall back on), and
    values without a mask raise too.
    """
    if fixed_cols is None and fixed_values is None:
        return None, None
    if fixed_cols is None:
        raise ValueError(
            f"{name}: fixed_values were given but no column is declared fixed "
            "— declare the mask too, or drop the values")
    m = mask_from_cols(fixed_cols, n_basis, prefix=f"{name}: ")
    if not m.any():
        raise ValueError(
            f"{name}: fixed_cols is empty — pass no mask at all for the "
            "parent path")
    if fixed_values is None:
        raise ValueError(
            f"{name}: a coefficient mask must be declared WITH the values of "
            "its columns (the cache is filled at `load_data`, so there is no "
            "later coefficient read to fall back on): pass "
            "fixed_values=<{col: value} | coeff vector>, or leave both out "
            "for the parent path")
    vec = fixed_values_vector(fixed_values, np.nonzero(m)[0], n_basis,
                              prefix=f"{name}: ")
    return m, vec


def unfixed_waves(gamma_matrix, mask, den_names=None):
    """``(n_fix, missing)`` — how many waves the mask can cache, and why not.

    The rule is the wave plan's, stated once here (a kernel and the wrapper
    must not disagree about it): a wave is CACHED iff **every** column its
    denominator row reads is declared fixed:

        fixed_wave[k] = all(mask[nonzero(gamma[k])])

    ``n_fix`` is the number of cached waves.  ``missing`` is ``None`` when the
    mask does cache something: the message is only useful for the "mask
    declares columns but caches nothing" case, and there it names the CLOSEST
    wave — the one with the fewest free columns, lowest index on a tie — as
    ``(wave_index, [column names still read])``.  A partial mask therefore
    returns ``(n, None)`` and never warns.

    Returns ``(None, None)`` for a ``gamma_matrix`` this rule cannot read.
    """
    g = np.asarray(gamma_matrix, np.float64)
    if g.ndim == 3 and g.shape[1] == 1:
        g = g[:, 0, :]
    if g.ndim != 2:
        return None, None
    m = np.asarray(mask, np.int8).ravel()
    if m.size != g.shape[1]:
        raise ValueError(
            f"unfixed_waves: mask has {m.size} entries, expected "
            f"n_basis = {g.shape[1]}")
    names = (None if den_names is None
             else [str(n) for n in np.asarray(den_names).ravel()])
    n_fix = 0
    best = None                      # (n_free, wave, [columns])
    for k in range(g.shape[0]):
        cols = np.nonzero(g[k])[0]
        if cols.size == 0:
            continue                 # an all-zero row is not a cacheable wave
        free = cols[m[cols] == 0]
        if free.size == 0:
            n_fix += 1
            continue
        if best is None or free.size < best[0]:
            best = (int(free.size), k, [int(c) for c in free])
    if n_fix:
        return n_fix, None
    if best is None:
        return 0, None
    _n, wave, cols = best
    if names is None:
        return 0, (wave, [f"column {c}" for c in cols])
    return 0, (wave, [names[c] for c in cols])


def warn_unfixed_mask(gamma_matrix, mask, den_names=None, name="mask",
                      stacklevel=2):
    """Warn (once per call) when a NON-EMPTY mask caches nothing.

    Returns True when it warned.  ``mask=None`` (the parent path) and a mask
    that caches at least one wave are both silent — a partial mask is the
    normal production state and must not nag.
    """
    if mask is None:
        return False
    m = np.asarray(mask, np.int8).ravel()
    if not m.any():
        return False
    n_fix, missing = unfixed_waves(gamma_matrix, m, den_names)
    if n_fix is None or n_fix > 0 or missing is None:
        return False
    wave, cols = missing
    warnings.warn(
        f"{name}: mask declares {int(m.sum())} column(s) but no wave's row is "
        f"fully fixed — wave {wave} still reads {cols}; the cache will not "
        "engage", MaskNotEngaged, stacklevel=stacklevel)
    return True

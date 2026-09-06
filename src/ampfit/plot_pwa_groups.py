"""
Plot partial-wave group contributions for the pure-PWA mode.

Pure-PWA model (any config that builds through the generic single-config
builder with ``C == 1``)::

    P(e) = Σ_p |A_p(e)|²          A_p(e) = Σ_k ck_k·a_{p,k}(e)
    n_wave = n_proj · N           entries p-major, shared ck of length N

Every *group* is a subset of the N shared base waves.  A group curve is the
reduced model obtained by zeroing every ck outside the group and recomputing
P over the phase space — exactly the convention of the legacy
``plot_pw_groups`` (coherent within the group, overlaid on the data, scaled
to the data with the global scale from the full model).  Groups are derived
from ``config.full_decay.get_partial_waves()``, the ordered list
``(ls, chain)`` at base-wave index k, so all policies share the same ground
truth as the kernels:

* ``by="chain"``      — one group per decay chain (whole LS path)
* ``by="resonance"``  — chains merged by their set of intermediate
                        resonances (default; like ``discover_groups``)
* ``by="ls"``         — one group per base wave k, labelled by resonance
                        and the orbital-L letters of its decays

The plotting panels are generic on the event arrays: every ``mass`` column
and every per-position/per-component ``angle`` variable (canonical φ-first:
φ columns are wrapped to [−π, π]).  No B→4π / time / mixing layout is
assumed.
"""

import os
import re

import numpy as np

from ampfit.plot_pw_groups import PWGroupPlotter

_L_LETTERS = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H"}


def _ls_tag(ls):
    """Orbital-L letters of a chain's LS tuple, one per decay in order."""
    return "".join(_L_LETTERS.get(l, f"L{l}") for l, _s in ls)


def _display(config, name):
    """LaTeX display name when the config knows the particle, else raw."""
    dm = config.name_display_map()
    if name in dm:
        return dm[name]
    return name


def _base_wave_units(config, waves):
    """Number of block duplications of the base waves in the full ck list."""
    n_base = max(1, len(waves))
    n_full = max(n_base, len(config.get_ck_map()))
    return max(1, n_full // n_base)


def _expand(base, n_blk):
    out = []
    for b in range(n_blk):
        out.extend(b * len(base) + i for i in base)
    return out


def discover_pwa_groups(config, by="resonance", merge=None):
    """Return ``{display label: ck indices}`` for the pure-PWA mode.

    The k-axis ground truth is ``full_decay.get_partial_waves()``:
    ``waves[k] = (ls_tuple, chain)``.  Group keys are built from the
    intermediate (non-top) resonance cores of each chain, formatted with
    ``config.name_display_map()``.  The per-base indices are duplicated over
    the full ck list (blocks × CP/permutation copies) the same way the
    legacy plot groups expand over the 8 blocks, so masks always index the
    real parameter vector handed to the backends.

    Args:
        config: the ampfit Config.
        by: 'chain' | 'resonance' (default) | 'ls'.
        merge: optional ``[(regex, label)]`` — group keys matching *regex*
            are merged into *label* before display conversion.

    Returns:
        ``{label: [ck_indices]}``.
    """
    if not hasattr(config, "full_decay"):
        raise ValueError("plot_pwa_groups needs a config with full_decay")
    waves = list(config.full_decay.get_partial_waves())
    n_blk = _base_wave_units(config, waves)

    raw = {}  # internal key -> base k indices

    def _key_for(policy, chain, ls):
        cores = []
        for d in chain.decays:
            if d.core.name != config.top and d.core.name not in cores:
                cores.append(d.core.name)
        res = " + ".join(cores)
        if policy == "chain":
            return f"{res} [{_ls_tag(ls)}]"
        if policy == "ls":
            return f"{res} {_ls_tag(ls)}"
        return res  # resonance: chains merge on their resonance set

    for k, (ls, chain) in enumerate(waves):
        key = _key_for(by, chain, ls)
        if merge:
            for pat, newlabel in merge:
                if re.match(pat, key):
                    key = newlabel
                    break
        raw.setdefault(key, []).append(k)

    groups = {}
    for key, inds in raw.items():
        display = " + ".join(_display(config, p) for p in key.split(" + "))
        groups.setdefault(display, []).extend(_expand(inds, n_blk))
    return groups


# ── generic panels (mass columns + canonical φ-first angle variables) ──────

def pwa_mass_varfun(x):
    """One variable per ``mass`` column of a pwa event dict."""
    return [x["mass"][:, i] for i in range(x["mass"].shape[1])]


def pwa_angle_varfun(x):
    """Per (position, component) angle variables, φ wrapped to [−π, π].

    Assumes the canonical φ-first variable order: the first ``n_comp//2``
    columns are the per-vertex azimuths, the rest the polar angles.
    """
    a = np.asarray(x["angle"])
    ne = a.shape[0]
    if a.ndim == 2:
        a = a.reshape(ne, -1, 1)
    elif a.ndim != 3:
        a = a.reshape(ne, -1, a.shape[-1])
    n_comp = a.shape[2]
    n_phi = n_comp // 2 if n_comp > 3 else (1 if n_comp == 3 else n_comp // 2)
    out = []
    n_pos = a.shape[1]
    for r in range(n_pos):
        for c in range(n_comp):
            v = a[:, r, c]
            if n_comp == 3 and c == 0 or (n_comp != 3 and c < n_phi):
                v = (v + np.pi) % (2 * np.pi) - np.pi
            out.append(v)
    return out


def angle_variable_labels(x):
    """Titles for :func:`pwa_angle_varfun` panels."""
    a = np.asarray(x["angle"])
    n_comp = a.shape[-1]
    n_pos = 1 if a.ndim == 2 else a.shape[1]
    n_phi = n_comp // 2 if n_comp > 3 else (1 if n_comp == 3 else n_comp // 2)
    labels = []
    for r in range(n_pos):
        for c in range(n_comp):
            if n_comp == 3 and c == 0 or (n_comp != 3 and c < n_phi):
                labels.append(rf"$\phi_{r}^{{({c})}}$")
            else:
                labels.append(rf"$\theta_{r}^{{({c})}}$")
    return labels


def var_ranges(data, phsp, varfun):
    """Per-panel (lo, hi) ranges from the data ∪ phsp values."""
    vd = varfun(data)
    vp = varfun(phsp)
    out = []
    for d, p in zip(vd, vp):
        d = np.asarray(d, dtype=float)
        p = np.asarray(p, dtype=float)
        lo = float(np.min([d.min(), p.min()]))
        hi = float(np.max([d.max(), p.max()]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            lo, hi = -1.0, 1.0
        out.append((lo, hi))
    return out


# ── config-plot-driven panels (via the generic ReadVar readers) ────────────

def config_panels(cfg, data_np, phsp_np):
    """Plot panels assembled from the config's ``plot:`` section.

    Uses :func:`ampfit.read_var.vars_from_config` — every variable keeps a
    :meth:`~ampfit.read_var.ReadVar.read` on the raw kernel event dict, so
    data and phsp histograms come from the exact same resolution.

    Returns ``{'mass': pg, 'angles': pg, ...}`` with each panel group *pg*
    holding the declared variable keys/labels/units and ready-made
    varfun/ranges/bin-width for ``PWGroupPlotter.plot_var``.  Variables
    whose topology is not present in the loaded arrays are skipped.
    """
    from ampfit.read_var import vars_from_config

    items = [(k, v) for k, v in vars_from_config(cfg)
             if _readable(v, data_np)]

    groups = {}

    def _group(name, vs):
        vrs = [v for _k, v in vs]
        sel = [it for it in items if it[1] in vrs]
        keys = [k for k, v in sel]
        labels = [getattr(v, "display", k) for k, v in sel]
        units = [getattr(v, "unit", "") for k, v in sel]

        def varfun(x):
            return [v.read(x) for v in vrs]

        ranges = []
        for v in vrs:
            rng = getattr(v, "range", None)
            if rng is None:
                lo = float(min(np.min(v.read(data_np)),
                               np.min(v.read(phsp_np))))
                hi = float(max(np.max(v.read(data_np)),
                               np.max(v.read(phsp_np))))
                if hi - lo < 1e-12 or not np.all(np.isfinite([lo, hi])):
                    lo, hi = -1.0, 1.0
                rng = (lo, hi)
            ranges.append(rng)
        spans = np.array([r[1] - r[0] for r in ranges])
        span = float(np.median(spans)) if len(spans) else 1.0
        width = span / 60.0 if span > 0 else 1.0
        return {"keys": keys, "labels": labels, "units": units,
                "varfun": varfun, "ranges": ranges, "width": width}

    for key, v in items:
        if getattr(v, "kind", None) == "mass":
            groups.setdefault("mass", []).append((key, v))
        else:
            groups.setdefault("angles", []).append((key, v))
    return {name: _group(name, vs) for name, vs in groups.items()}


def _readable(v, data_np):
    try:
        v.read(data_np)
        return True
    except (IndexError, ValueError):
        return False

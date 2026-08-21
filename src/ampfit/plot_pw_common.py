"""Common plotting pipeline for the ``plot_pw_*`` scripts.

All ``plot_pw*`` scripts produce the same set of distributions — mass
overview, angles, cos θ₁−θ₂ diff-histograms, same-charge-pair
variables, ππ/3π masses, sorted masses and differences — with the same
styles.  The **only** thing that differs between scripts is how the
partial-wave *groups* are defined (topology groups, (L,S) groups,
resonance groups, …).

So each script implements a ``groups(config) -> {label: ck_indices}``
function and this module provides the shared setup + common plotting::

    from ampfit.plot_pw_common import run
    def groups(config): ...            # script-specific
    run(fit_json, config, data, phsp, max_events, backend,
        output, fmt, groups, description)
"""

import os
import sys

import numpy as np

from ampfit import Fitter
from ampfit.plot_pw_groups import PWGroupPlotter, plot_samesign

_PIP_NAMES = {"pip1", "pim1", "pip2", "pim2"}

# stacked-mass permutation columns (rows of the 24-row mass array)
# 24 rows = 8 blocks × 3 topologies; block b covers rows 3b..3b+2
_MASS_TOP1 = [1, 4, 7, 10]      # topo 1 across the four B-blocks (3π⁺)
_MASS_TOP2 = [2, 5, 8, 11]      # topo 2 across the four B-blocks (3π⁻)
_MASS_PIPI = [0, 3, 6, 9]       # topo 0 across the four B-blocks (ππ)


def _mass_cols(x, cols):
    """The first mass column of the given rows."""
    return [x["mass"].reshape(-1, 24, 2)[:, i, 0] for i in cols]


def _angle_var(x):
    """φ, cos θ₁, cos θ₂ for the first three rows."""
    a = x["angle"].reshape(x["angle"].shape[0], -1, 3)
    out = []
    for pos in range(3):
        out.append((a[:, pos, 0] + np.pi) % (2 * np.pi) - np.pi)
        out.append(np.cos(a[:, pos, 1]))
        out.append(np.cos(a[:, pos, 2]))
    return out


def _diff_cos_theta(x, row):
    """[cos θ₁, cos θ₂] of *row* — for a diff-histogram."""
    a = x["angle"].reshape(x["angle"].shape[0], -1, 3)
    return [np.cos(a[:, row, 1]), np.cos(a[:, row, 2])]


def _sorted_pipi(x):
    """Sorted ππ: groups [0,9] and [3,6], each sorted within, then by
    group min."""
    m = x["mass"].reshape(-1, 24, 2)
    a = np.column_stack([m[:, 0, 0], m[:, 9, 0]])
    b = np.column_stack([m[:, 3, 0], m[:, 6, 0]])
    a_min = a.min(1); a_max = a.max(1)
    b_min = b.min(1); b_max = b.max(1)
    mask = a_min < b_min
    out = np.zeros((len(m), 4))
    out[mask, 0] = a_min[mask]; out[mask, 1] = a_max[mask]
    out[mask, 2] = b_min[mask]; out[mask, 3] = b_max[mask]
    out[~mask, 0] = b_min[~mask]; out[~mask, 1] = b_max[~mask]
    out[~mask, 2] = a_min[~mask]; out[~mask, 3] = a_max[~mask]
    return [out[:, i] for i in range(4)]


def _sorted_pair(x, idx_a, idx_b):
    """Two-perm sorted pair: min, max."""
    m = x["mass"].reshape(-1, 24, 2)
    a, b = m[:, idx_a, 0], m[:, idx_b, 0]
    lo = np.minimum(a, b); hi = np.maximum(a, b)
    return [lo, hi]


def _diff_pipipi(x):
    """m(3π) columns for the (π⁺π⁺π⁻) − (π⁺π⁻π⁻) difference."""
    m = x["mass"].reshape(-1, 24, 2)
    return [m[:, 1, 0], m[:, 4, 0], m[:, 2, 0], m[:, 8, 0]]


def plot_common(plotter, fitter, output="plots/", fmt="png"):
    """Plot all the common distributions (same order, same styles).

    Args:
        plotter: a computed :class:`~ampfit.plot_pw_groups.PWGroupPlotter`.
        fitter: the :class:`~ampfit.Fitter` holding the data.
        output: output directory.
        fmt: image format (png/pdf).
    """
    os.makedirs(output, exist_ok=True)

    # ── mass overview ─────────────────────────────────────────────
    nm = fitter._data_np["mass"].shape[1] // 8
    plotter.plot_var(
        lambda x: [x["mass"][:, i] for i in range(nm)],
        [f"mass[{i}]" for i in range(nm)],
        0.2, 5.2, 0.05, "mass", output=output, fmt=fmt, smooth_sigma=1.0)

    # ── angles ────────────────────────────────────────────────────
    ar = [(-np.pi, np.pi), (-1, 1), (-1, 1)] * 3
    al = [f"angle[{p},{c}]" for p in range(3) for c in range(3)]
    plotter.plot_var(_angle_var, al, 0, 1, 0.1, "angles",
                     ranges=ar, output=output, fmt=fmt, unit="")

    # ── cos θ₁ − cos θ₂ diff-histograms (rows 1, 2) ──────────────
    for row in (1, 2):
        plotter.plot_stacked_perm(
            lambda x, r=row: _diff_cos_theta(x, r),
            rf"$\cos\theta_1 - \cos\theta_2$ (row {row})",
            -1, 1, 0.05, f"cos_theta_diff_row{row}",
            output=output, fmt=fmt, scales=[1, -1],
            smooth_sigma=1.0, show_pull=True, legend=True)

    # ── same-charge-pair variables (B → (π⁺π⁺)(π⁻π⁻)) ────────────
    plot_samesign(plotter, output=output, fmt=fmt)

    # ── ππ / 3π masses (stacked permutations) ─────────────────────
    plotter.plot_stacked_perm(
        lambda x: _mass_cols(x, _MASS_PIPI),
        "m(π⁺π⁻)", 0.2, 5.2, 0.05, "m_pipi", output=output, fmt=fmt,
        smooth_sigma=1.0, show_pull=True, legend=True)
    plotter.plot_stacked_perm(
        lambda x: _mass_cols(x, _MASS_TOP1),
        "m(π⁺π⁺π⁻)", 0.2, 5.2, 0.05, "m_pipipip", output=output, fmt=fmt,
        smooth_sigma=1.0, show_pull=True)
    plotter.plot_stacked_perm(
        lambda x: _mass_cols(x, _MASS_TOP2),
        "m(π⁺π⁻π⁻)", 0.2, 5.2, 0.05, "m_pipipim", output=output, fmt=fmt,
        smooth_sigma=1.0, show_pull=True)

    # ── sorted ππ ─────────────────────────────────────────────────
    _ranges = [(0.2, 1.5, 0.015), (0.2, 5.2, 0.05), (0.2, 3.0, 0.03),
               (0.2, 5.0, 0.05)]
    _xlabels = [
        r"$m(\pi\pi)^{\rm min}_{\rm low}$",
        r"$m(\pi\pi)^{\rm max}_{\rm low}$",
        r"$m(\pi\pi)^{\rm min}_{\rm high}$",
        r"$m(\pi\pi)^{\rm max}_{\rm high}$",
    ]
    for i in range(4):
        lo, hi, bw = _ranges[i]
        plotter.plot_var(
            lambda x, idx=i: [_sorted_pipi(x)[idx]],
            [_xlabels[i]], lo, hi, bw,
            f"m_pipi_sorted_{['pp1_min', 'pp1_max', 'pp2_min', 'pp2_max'][i]}",
            output=output, fmt=fmt, smooth_sigma=1.0,
            legend=(i in (0, 2)), show_pull=True)

    # ── sorted 3π (min/max of the two charge permutations) ────────
    for prefix, idx_a, idx_b, r_min, r_max, xl_min, xl_max in [
            ("m_pipipip_sorted", 1, 4, (0.2, 5.2), (1.4, 5.2),
             r"$m(\pi^+\pi^+\pi^-)^{\rm min}$",
             r"$m(\pi^+\pi^+\pi^-)^{\rm max}$"),
            ("m_pipipim_sorted", 2, 8, (0.2, 5.2), (1.4, 5.2),
             r"$m(\pi^+\pi^-\pi^-)^{\rm min}$",
             r"$m(\pi^+\pi^-\pi^-)^{\rm max}$")]:
        for j, (rj, label_j) in enumerate([(r_min, xl_min), (r_max, xl_max)]):
            plotter.plot_var(
                lambda x, a=idx_a, b=idx_b, jj=j: [_sorted_pair(x, a, b)[jj]],
                [label_j], rj[0], rj[1], 0.05,
                f"{prefix}_{['min', 'max'][j]}", output=output, fmt=fmt,
                smooth_sigma=1.0, show_pull=True, legend=True)

    # ── m(3π): (π⁺π⁺π⁻) − (π⁺π⁻π⁻) difference ───────────────────
    plotter.plot_stacked_perm(
        _diff_pipipi, r"$m(3\pi)$",
        0.2, 5.2, 0.05, "m_pipipi_diff", output=output, fmt=fmt,
        scales=[1, 1, -1, -1], smooth_sigma=1.0, show_pull=True)

    # ── time ──────────────────────────────────────────────────────
    plotter.plot_var(lambda x: [x["time"]], ["time"], 0, 10, 0.2, "time",
                     output=output, fmt=fmt, unit="ps",
                     legend=True, show_pull=True)


def run(fit_json, config, data, phsp, max_events, backend, output, fmt,
        groups_fn, description):
    """Full pipeline: setup fitter → build groups → plot everything.

    Args:
        fit_json, config, data, phsp, max_events, backend, output, fmt:
            the common CLI arguments.
        groups_fn: callable(config) → ``{label: ck_indices}`` — the only
            script-specific part.
        description: short label printed for the group summary.
    """
    f = Fitter(config, backend=backend)
    cp = os.path.splitext(fit_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)

    data_np, nd = Fitter.load_npz(data, max_events=max_events)
    phsp_np, np_ = Fitter.load_npz(phsp, max_events=max_events)
    print(f"  Loaded {nd:,} data + {np_:,} phsp events")
    f.set_phsp(phsp_np)
    f.set_data(data_np)

    r = f.load_results(fit_json)
    if r.x is None or len(r.x) == 0:
        sys.exit(1)

    groups = groups_fn(f.config)
    plotter = PWGroupPlotter(f, r, groups).compute()
    print(f"  {len(plotter.labels)} {description}: {plotter.labels}")
    plot_common(plotter, f, output=output, fmt=fmt)
    return plotter

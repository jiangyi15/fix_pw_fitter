#!/usr/bin/env python3
"""Plot the combined topology+LS group shape as a 1D profile along one
resonance mass (``--axis m1|m2``), with the per-resonance components.

For a partial-wave group (a topology + full (L,S) coupling) the shape
depends only on the two resonance masses and is the coherent sum of the
group's waves::

    A(m1, m2) = Σ_chains  C_chain · BW₁(m1) · BW₂(m2),   C_chain = Σ ck_i

The plot fixes the other mass at *--slice* and shows the profile
``|A|²`` with a 1σ band (covariance propagation by central differences
over the flat fit vector).  For every distinct resonance pair
(res1 × res2) in the group the coupling-weighted product component
``|C·BW(res1)·BW(res2)|²`` is overlaid, so the individual lineshape
envelopes (scaled by their fitted couplings) are visible against the
coherent sum.

Usage::

    python scripts/plot_group_shape.py config_angle.yml fit_output/results.json \\
        --group "R.pi (1,1)-(1,0)-(0,0)" --slice 0.475 -o plots/      # m1 profile
    python scripts/plot_group_shape.py config_angle.yml fit_output/results.json \\
        --group "R.pi (1,1)-(1,0)-(0,0)" --axis m2 --slice 1.234      # m2 profile
    python scripts/plot_group_shape.py config_angle.yml fit_output/results.json \\
        --group "R.R"                                                 # default slices
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ampfit import Fitter

M_PI = 0.13957
M_B = 5.279
_PIP = {"pip1", "pim1", "pip2", "pim2"}


def _load_ls_groups():
    """Import ``discover_ls_groups`` from scripts/plot_pw_ls.py."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "plot_pw_ls.py")
    spec = importlib.util.spec_from_file_location("plot_pw_ls", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.discover_ls_groups


def _particle_names(f):
    """All resonance particle names in the config (for charge-conjugate
    detection)."""
    names = set()
    for chain in f.config.full_decay.chains:
        for d in chain.decays:
            names.add(d.core.name)
    return names


def _is_rminus(f, res):
    """True if *res* is the R⁻ charge conjugate (name ends with ``m``
    and has a ``+``-charge counterpart)."""
    name = res.name
    return name.endswith("m") and (name[:-1] + "p") in _particle_names(f)


def chain_resonances(chain):
    """Return ``(res1, res2)`` — the two resonance Particles whose masses
    are the topology's (m1, m2).  ``res1`` is the non-pion daughter of the
    B decay, ``res2`` the second resonance (the other B daughter for R.R,
    the inner resonance for R.pi chains)."""
    outs = chain.decays[0].outs
    res1 = next(o for o in outs if o.name not in _PIP)
    if any(o.name not in _PIP for o in outs[1:]):
        # R.R: both B daughters are resonances
        res2 = next(o for o in outs if o.name != res1.name
                    and o.name not in _PIP)
    else:
        # R.pi: the inner resonance is the non-pion daughter of decays[1]
        res2 = next(o for o in chain.decays[1].outs
                    if o.name not in _PIP and o.name != res1.name)
    return res1, res2


def group_wave_map(f, groups, chain_index, label):
    """Map each (res1, res2) pair of the group to its base ck indices
    (block 0) in that chain."""
    ranges = chain_index          # list of (start, end, chain)
    mask = set(groups[label])

    group_waves = {}
    for start, end, chain in ranges:
        base_idx = [i for i in mask if start <= i < end]
        if not base_idx:
            continue
        res1, res2 = chain_resonances(chain)
        group_waves[(res1, res2)] = base_idx
    return group_waves


def build_profile_observable(f, group_waves, grid, fix, along, obs="abs2"):
    """Return ``obs_both(x) -> (obs_Rplus, obs_Rminus)`` — the 1D profile
    of the requested observable along the profiled axis (other mass fixed
    at *fix*), split into the R⁺ and R⁻ charge-conjugate totals.

    The two totals are the coherent sums of the group's waves over the
    R⁺ and R⁻ chains separately (no backend, pure BW propagators).

    ``along`` = ``"m1"`` profiles ``BW₁(grid)·BW₂(fix)``, ``"m2"``
    profiles ``BW₁(fix)·BW₂(grid)``.  *obs* selects the real observable:
    ``"abs2"`` (|A|²), ``"re"``, ``"im"``, or ``"phase"`` (arg A)."""

    def obs_both(x):
        params, resolved = f.build_params(x)
        ck = params["ck"]
        Ap = np.zeros(len(grid), dtype=complex)
        Am = np.zeros(len(grid), dtype=complex)
        for (res1, res2), idx in group_waves.items():
            C = ck[idx].sum()
            if along == "m1":
                bw1 = res1._model.amplitude_raw(grid, resolved)
                bw2 = res2._model.amplitude_raw(np.array([fix]), resolved)[0]
            else:
                bw1 = res1._model.amplitude_raw(np.array([fix]), resolved)[0]
                bw2 = res2._model.amplitude_raw(grid, resolved)
            comp = C * bw1 * bw2
            if _is_rminus(f, res1) or _is_rminus(f, res2):
                Am += comp
            else:
                Ap += comp

        def red(A):
            if obs == "abs2":
                return np.abs(A) ** 2
            if obs == "re":
                return A.real
            if obs == "im":
                return A.imag
            return np.angle(A)

        return red(Ap), red(Am)

    return obs_both


OBS_LABEL = {"abs2": r"$|A|^2$", "re": r"$\mathrm{Re}\,A$",
             "im": r"$\mathrm{Im}\,A$", "phase": r"$\arg A$ [rad]"}


def shape_errors(f, fit_result, obs_flat):
    """1σ errors of ``obs_flat`` from the fit covariance.

    Central differences over the flat optimizer vector *x* (the space
    ``hess_inv`` lives in), avoiding the (broken) constraint backprop.
    """
    x0 = np.asarray(fit_result.x)
    hess_inv = getattr(fit_result, "hess_inv", None)
    if hess_inv is None:
        return np.zeros(len(obs_flat(x0)))
    hess_inv = np.asarray(hess_inv, dtype=float)
    n_obs = len(obs_flat(x0))
    G = np.zeros((n_obs, len(x0)))
    for j in range(len(x0)):
        eps = 1e-4 * max(1.0, abs(float(x0[j])))
        xp = x0.copy(); xp[j] += eps
        xm = x0.copy(); xm[j] -= eps
        G[:, j] = (obs_flat(xp) - obs_flat(xm)) / (2 * eps)
    var = np.sum(G * (G @ hess_inv), axis=1)
    return np.sqrt(np.maximum(var, 0.0))


def product_component(f, res1, res2, C, grid, fix, x, along="m1", obs="abs2"):
    """The per-pair component ``C·BW₁·BW₂`` of one res1 × res2 pair at
    the fitted parameters *x* (``C = Σ ck`` over the pair's group waves),
    reduced to the requested observable *obs* (abs2/re/im/phase).

    ``along`` selects which variable runs over *grid* while the other is
    fixed at the scalar *fix*: ``"m1"`` → ``BW₁(grid)·BW₂(fix)``,
    ``"m2"`` → ``BW₁(fix)·BW₂(grid)``.
    """
    _, resolved = f.build_params(x)
    if along == "m1":
        bw1 = res1._model.amplitude_raw(grid, resolved)
        bw2 = res2._model.amplitude_raw(np.array([fix]), resolved)[0]
    else:
        bw1 = res1._model.amplitude_raw(np.array([fix]), resolved)[0]
        bw2 = res2._model.amplitude_raw(grid, resolved)
    A = C * bw1 * bw2
    if obs == "abs2":
        return np.abs(A) ** 2
    if obs == "re":
        return A.real
    if obs == "im":
        return A.imag
    return np.angle(A)


def resonance_mass(f, res, x):
    """Fitted (or config) mass of a resonance, for the default slice."""
    _, resolved = f.build_params(x)
    return float(resolved.get(f"{res.name}_mass", res._model.kwargs.get("mass", 0.775)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("results_json")
    ap.add_argument("--group", default=None,
                    help="substring filter on the (L,S) group label; "
                         "default: all groups")
    ap.add_argument("--m1", type=float, nargs=2, default=None,
                    help="range of the profiled mass (default: [2mπ, mB−mπ])")
    ap.add_argument("--axis", choices=("m1", "m2"), default="m1",
                    help="variable to profile (default m1)")
    ap.add_argument("--obs", choices=("abs2", "re", "im", "phase"),
                    default="abs2",
                    help="observable to plot (default abs2)")
    ap.add_argument("--slice", type=float, default=None,
                    help="fixed value of the other mass (default: the "
                         "dominant chain's partner-resonance mass)")
    ap.add_argument("--n", type=int, default=300,
                    help="grid points along the profiled axis (default 300)")
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("-o", "--output", default="plots/group_shape/")
    ap.add_argument("--format", default="png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.results_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    r = f.load_results(args.results_json)
    if r.x is None or len(r.x) == 0:
        sys.exit("no fitted parameters in results")
    x0 = np.asarray(r.x)

    discover_ls_groups = _load_ls_groups()
    groups = discover_ls_groups(f.config)
    if args.group:
        groups = {k: v for k, v in groups.items() if args.group in k}
    if not groups:
        sys.exit(f"no groups matching {args.group!r}")
    chain_ranges = f.config._chain_ranges()

    lo1, hi1 = args.m1 if args.m1 else (2 * M_PI, M_B - M_PI)
    grid = np.linspace(lo1, hi1, args.n)

    os.makedirs(args.output, exist_ok=True)

    for label in sorted(groups):
        group_waves = group_wave_map(f, groups, chain_ranges, label)
        print(f"{label}: {len(groups[label])//8} wave(s), "
              f"{len(group_waves)} chain(s)")

        # dominant chain sets the default slice (the partner mass)
        params, _ = f.build_params(x0)
        ck = params["ck"]
        (res1d, res2d), _ = max(
            group_waves.items(), key=lambda kv: abs(ck[kv[1]].sum()))
        if args.axis == "m1":
            fixname, fixres = "m2", res2d
            slice_default = resonance_mass(f, res2d, x0)
        else:
            fixname, fixres = "m1", res1d
            slice_default = resonance_mass(f, res1d, x0)
        fix = args.slice if args.slice is not None else slice_default

        obs_both = build_profile_observable(f, group_waves, grid, fix,
                                            args.axis, args.obs)
        vp, vm = obs_both(x0)
        ep = shape_errors(f, r, lambda x: obs_both(x)[0])
        em = shape_errors(f, r, lambda x: obs_both(x)[1])
        # only split into R⁺/R⁻ totals when the group actually contains
        # charge-conjugate chains; neutral groups (e.g. R.R ρρ) have no
        # R⁻ and would otherwise show a meaningless zero curve
        has_rminus = any(_is_rminus(f, r1) or _is_rminus(f, r2)
                         for (r1, r2) in group_waves)
        if has_rminus:
            print(f"  axis={args.axis}  {fixname}-slice={fix:.3f} "
                  f"({fixres.name})  {OBS_LABEL[args.obs]} "
                  f"R+: {vp.min():.2e}..{vp.max():.2e}  "
                  f"R-: {vm.min():.2e}..{vm.max():.2e}")
        else:
            print(f"  axis={args.axis}  {fixname}-slice={fix:.3f} "
                  f"({fixres.name})  {OBS_LABEL[args.obs]} "
                  f"{vp.min():.2e}..{vp.max():.2e}")

        base = f"{label.replace(' ', '_').replace('(', '').replace(')', '')}"
        base = base.replace(",", "_")
        obs_label = OBS_LABEL[args.obs]
        fig, ax = plt.subplots(figsize=(7, 5))
        if has_rminus:
            ax.plot(grid, vp, "k-", lw=1.5,
                    label=rf"$\sum_{{R^+}} C\cdot BW_1(m_1) BW_2(m_2)$ "
                          rf"({fixname}={fix:.2f})")
            ax.fill_between(grid, vp - ep, vp + ep, color="k", alpha=0.2,
                            label="1σ (R⁺)")
            ax.plot(grid, vm, "r-", lw=1.5,
                    label=rf"$\sum_{{R^-}} C\cdot BW_1(m_1) BW_2(m_2)$ "
                          rf"({fixname}={fix:.2f})")
            ax.fill_between(grid, vm - em, vm + em, color="r", alpha=0.2,
                            label="1σ (R⁻)")
        else:
            ax.plot(grid, vp, "k-", lw=1.5,
                    label=rf"$\sum C\cdot BW_1(m_1) BW_2(m_2)$ "
                          rf"({fixname}={fix:.2f})")
            ax.fill_between(grid, vp - ep, vp + ep, color="k", alpha=0.2,
                            label="1σ")
        # partial-wave components: only the R⁺ chains (the R⁻ conjugates
        # carry the same wave shape)
        for (res1, res2), idx in group_waves.items():
            if _is_rminus(f, res1) or _is_rminus(f, res2):
                continue
            C = ck[idx].sum()
            comp = product_component(f, res1, res2, C, grid, fix, x0,
                                     along=args.axis, obs=args.obs)
            if np.max(np.abs(comp)) > 0:
                r1 = res1.display.strip("$")
                r2 = res2.display.strip("$")
                if args.obs == "abs2":
                    lab = rf"$|C*BW({r1})*BW({r2})|^2$"
                elif args.obs == "re":
                    lab = rf"$\mathrm{{Re}}\,C*BW({r1})*BW({r2})$"
                elif args.obs == "im":
                    lab = rf"$\mathrm{{Im}}\,C*BW({r1})*BW({r2})$"
                else:
                    lab = rf"$\arg\,C*BW({r1})*BW({r2})$"
                ax.plot(grid, comp, "--", lw=1.2, label=lab)
        ax.set_xlabel(rf"${args.axis}$ [GeV]")
        ax.set_ylabel(rf"{obs_label}")
        ax.set_title(f"{label}   ({fixname} = {fix:.3f} GeV)   {obs_label}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        out = os.path.join(args.output,
                           f"slice_{args.obs}_{base}.{args.format}")
        fig.savefig(out)
        plt.close(fig)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot FixedShape basis amplitude contributions (GaussianBasis / BSplineBasis).

Shows each basis particle's CK-weighted amplitude per sub-decay channel
(e.g. MI00p->rhoA, MI00p->f0(980), MI00p->f0(500)) and the total.

Usage::

    # All particles
    python scripts/plot_gaussian_basis.py config.yml results.json -o plot.pdf

    # Filter by particle name pattern
    python scripts/plot_gaussian_basis.py config.yml results.json --pattern MI00
    python scripts/plot_gaussian_basis.py config.yml results.json --pattern "MI0[12]"
    python scripts/plot_gaussian_basis.py config.yml results.json --pattern "p$"
"""

import sys, os, re, argparse
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description="Plot GaussianBasis lineshape")
    ap.add_argument("config")
    ap.add_argument("results_json")
    ap.add_argument("--mass-lo", type=float, default=0.28)
    ap.add_argument("--mass-hi", type=float, default=5.14)
    ap.add_argument("--n-points", type=int, default=500)
    ap.add_argument("-o", "--output", default="gaussian_basis.pdf")
    ap.add_argument("--pattern", default=None,
                    help="Regex filter on particle name (e.g. 'MI00', 'p$', 'MI0[12]')")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ampfit import Fitter

    # ── Load ────────────────────────────────────────────────────
    os.chdir(os.path.dirname(os.path.abspath(args.config)))
    f = Fitter(args.config, backend="numpy")
    cp = os.path.splitext(args.results_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    r = f.load_results(args.results_json)

    m_grid = np.linspace(args.mass_lo, args.mass_hi, args.n_points)

    # ── Build CK and collect per-particle amplitudes ────────────
    seen_models = set()
    particle_map = {}  # name → (model, mu, sigma)
    from ampfit.particle_model.models_builtin import GaussianBasisModel, BSplineBasisModel
    basis_types = (GaussianBasisModel, BSplineBasisModel)
    for chain in f.config.full_decay.chains:
        for decay in chain.decays[1:]:
            model = decay.core._model
            if not isinstance(model, basis_types):
                continue
            mid = id(model)
            if mid in seen_models:
                continue
            seen_models.add(mid)
            name = decay.core.name
            mu = float(model.kwargs.get("mu", 0.775))
            sigma = float(model.kwargs.get("sigma", 0.1))
            particle_map[name] = (model, mu, sigma)

    if not particle_map:
        print("No GaussianBasis particles found")
        sys.exit(1)

    # Filter by pattern if given
    if args.pattern:
        filtered = {n: v for n, v in particle_map.items()
                    if re.search(args.pattern, n)}
        if not filtered:
            print(f"Pattern '{args.pattern}' matched no particles. Available: {list(particle_map)}")
            sys.exit(1)
        print(f"Filtered {len(particle_map)} → {len(filtered)} particles by '{args.pattern}'")
        particle_map = filtered

    print(f"Found {len(particle_map)} GaussianBasis/BSplineBasis particles")

    # Extract the sub-decay daughter from a comb's pname term:
    #   "MI00p->f0(980).pip2_g_ls_0" → "f0(980)"
    def sub_decay_of(pname, comb):
        for t in comb:
            if isinstance(t, str) and t.startswith(pname + "->"):
                return t[len(pname) + 2:].split(".", 1)[0]
        return None

    # ── Standalone loop: map each (variant, daughter) channel to the
    # list of (basis_name, model, wave_index) that contribute ────
    basis_idx = {}   # (variant, daughter) → [(basis_name, model, idx), ...]
    for bname, (model, _, _) in particle_map.items():
        for i, comb in enumerate(f.all_comb):
            d = sub_decay_of(bname, comb)
            if d:
                basis_idx.setdefault((bname[-1], d), []).append((bname, model, i))

    # ── Channel amplitude: A(m) = Σ_basis ck[i_b] · ak_b(m) ─────
    # ck[i_b]: INDEX the CK wave array (one wave per basis particle).
    # ak_b:    (n_basis, n_m) — the basis particle's amplitude shape.
    def channel_amp(phys, variant, daughter):
        ck_all = f.pc.build_ck(phys)                 # (n_wave,)
        # ak_b depends only on the basis particle, not the daughter
        ak_all = {name: model.amplitude_raw(m_grid, phys)
                  for name, (model, _, _) in particle_map.items()}
        entries = basis_idx[(variant, daughter)]
        # Standalone loop: indexed ck per basis particle
        ck_b = [ck_all[idx] for _b, _m, idx in entries]
        # Standalone loop: basis amplitude shape (daughter-independent)
        ak_b = [ak_all[b] for b, _m, _i in entries]
        ck_b = np.asarray(ck_b)                      # (n_basis,)
        ak_b = np.asarray(ak_b)                      # (n_basis, n_m)
        assert ck_b.shape[0] == ak_b.shape[0]
        return np.sum(ck_b[:, None] * ak_b, axis=0)  # (n_m,)

    _, resolved = f.build_params(r.x)

    # Unique (variant, daughter) channels for coloring
    channels = sorted({(p[-1], d) for p in particle_map
                       for d in (sub_decay_of(p, c) for c in f.all_comb) if d})
    daughters = sorted({d for (_v, d) in channels})
    _cmap = plt.get_cmap("tab10")
    dcolors = {d: _cmap(i % 10) for i, d in enumerate(daughters)}

    # ── Per-(variant, daughter) totals ──────────────────────────
    def make_obs_tot(variant, daughter):
        def obs(phys):
            A = channel_amp(phys, variant, daughter)
            out = [0.0] * (3 * len(m_grid))
            out[0::3] = A.real
            out[1::3] = A.imag
            out[2::3] = np.abs(A) ** 2
            return out
        return obs

    def relevant_params_tot(variant, daughter):
        names = set()
        for comb in f.all_comb:
            if not any(p[-1] == variant and sub_decay_of(p, comb) == daughter
                       for p in particle_map):
                continue
            for t in comb:
                if isinstance(t, str):
                    names.add(t + "r")
                    names.add(t + "i")
        for pname, (model, _, _) in particle_map.items():
            if pname[-1] != variant:
                continue
            for gn in model.get_gamma_name():
                names.add(gn)
            names.add(f"{pname}_mass")
        return [n for n in names if n in resolved]

    tot_var = {ch: channel_amp(resolved, *ch) for ch in channels}

    err_var = {}
    has_err = hasattr(r, "hess_inv") and r.hess_inv is not None
    for ch in channels:
        if not has_err:
            err_var[ch] = tuple(np.zeros(len(m_grid)) for _ in range(3))
            continue
        param_names = relevant_params_tot(*ch)
        values, errors = f.cal_uncertainties_multi_vec(
            make_obs_tot(*ch), param_names, r, return_cov=False)
        errors = np.asarray(errors)
        err_var[ch] = (errors[0::3], errors[1::3], errors[2::3])
    print(f"Uncertainties: {'covariance band' if has_err else 'no hess_inv — no band'}")

    # ── Plot: one curve per sub-decay channel (sum over all basis
    # particles), with covariance-propagated uncertainty band ─────
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.08)
    ax_revm = fig.add_subplot(gs[0, 0])                # x=Re, y=m
    ax_sq = fig.add_subplot(gs[0, 1])                   # x=m, y=|A|²
    ax_ar = fig.add_subplot(gs[1, 0], sharex=ax_revm)   # Argand: share Re
    ax_im = fig.add_subplot(gs[1, 1], sharex=ax_sq, sharey=ax_ar)  # share m + Im

    for (variant, daughter) in sorted(channels):
        A = tot_var[(variant, daughter)]
        e_re, e_im, e_sq = err_var[(variant, daughter)]
        c = dcolors[daughter]
        ls = '-' if variant == 'p' else '--'
        lbl = f'Σ MI*{variant}->{daughter}'
        ax_revm.fill_betweenx(m_grid, A.real - e_re, A.real + e_re,
                              color=c, alpha=0.25, linewidth=0)
        ax_sq.fill_between(m_grid, np.abs(A) ** 2 - e_sq, np.abs(A) ** 2 + e_sq,
                           color=c, alpha=0.25, linewidth=0)
        ax_im.fill_between(m_grid, A.imag - e_im, A.imag + e_im,
                           color=c, alpha=0.25, linewidth=0)
        ax_revm.plot(A.real, m_grid, ls, color=c, linewidth=2.5, label=lbl)
        ax_sq.plot(m_grid, np.abs(A) ** 2, ls, color=c, linewidth=2.5, label=lbl)
        ax_im.plot(m_grid, A.imag, ls, color=c, linewidth=2.5, label=lbl)
        ax_ar.plot(A.real, A.imag, ls, color=c, linewidth=2.0, label=lbl)

    for ax in [ax_revm, ax_im, ax_sq]:
        ax.grid(True, alpha=0.3)
    # ax_revm's y-axis is mass — no axhline(0) (it would pull ymin to 0)
    for ax in [ax_im, ax_sq]:
        ax.axhline(0, color="grey", linewidth=0.5)

    # Legend: one entry per sub-decay channel
    handles, labels = ax_sq.get_legend_handles_labels()
    ax_sq.legend(handles, labels, fontsize=6, ncol=3, loc='upper right')

    # Hide redundant tick labels on shared axes
    ax_revm.tick_params(labelbottom=False)   # x shared with Argand
    ax_sq.tick_params(labelbottom=False)     # x shared with bottom-right
    ax_im.tick_params(labelleft=False)       # y shared with Argand

    ax_revm.set_xlabel("Re(amplitude)")
    ax_revm.set_ylabel("m (GeV)")
    ax_sq.set_xlabel("m (GeV)")
    ax_sq.set_ylabel("|amplitude|²")
    ax_sq.yaxis.set_label_position("right")
    ax_sq.tick_params(labelright=True, labelleft=False)
    ax_im.set_xlabel("m (GeV)")
    ax_im.set_ylabel("Im(amplitude)")

    # Argand
    ax_ar.axhline(0, color="grey", linewidth=0.5)
    ax_ar.axvline(0, color="grey", linewidth=0.5)
    ax_ar.grid(True, alpha=0.3)
    ax_ar.set_xlabel("Re(amplitude)")
    ax_ar.set_ylabel("Im(amplitude)")

    # Symmetric range for Argand; sharex/sharey align the rest
    r = max(np.abs(ax_ar.get_xlim()).max(), np.abs(ax_ar.get_ylim()).max()) * 1.1
    ax_ar.set_xlim(-r, r)
    ax_ar.set_ylim(-r, r)

    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {args.output}")


if __name__ == "__main__":
    main()

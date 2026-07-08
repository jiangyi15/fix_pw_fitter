#!/usr/bin/env python3
"""Plot BW lineshape with 1σ uncertainty band for a chosen resonance.

Uses :meth:`Fitter.cal_uncertainties_multi_vec` to propagate fit
parameter uncertainties through the lineshape at each mass point.

Usage::

    python scripts/plot_bw_lineshape_uncert.py config_amp.yml results.json --resonance rhoA
    python scripts/plot_bw_lineshape_uncert.py config_amp.yml results.json --resonance rhoA -o bw_rhoA.pdf
"""

import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(
        description="Plot BW lineshape with uncertainty band")
    ap.add_argument("config")
    ap.add_argument("results_json")
    ap.add_argument("--resonance", required=True,
                    help="Resonance name (e.g. rhoA, f0(980))")
    ap.add_argument("--mass-lo", type=float, default=None,
                    help="Lower mass limit (default: 2*m_pi ≈ 0.28)")
    ap.add_argument("--mass-hi", type=float, default=None,
                    help="Upper mass limit (default: m_B - m_pi ≈ 5.14)")
    ap.add_argument("--n-points", type=int, default=500,
                    help="Number of mass points (default 500)")
    ap.add_argument("-o", "--output", default="bw_lineshape_uncert.pdf")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from ampfit import Fitter

    # ── Load fitter and results ──────────────────────────────────
    f = Fitter(args.config, backend="numpy")
    # Try constraints alongside results; fall back to stripping
    # _converted suffix (convert_ck_names.py creates _converted.json
    # but the constraints still use the original filename)
    cp = os.path.splitext(args.results_json)[0] + "_constraints.json"
    if not os.path.exists(cp):
        base = os.path.splitext(args.results_json)[0]
        if base.endswith("_converted"):
            cp = base[:-len("_converted")] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    result = f.load_results(args.results_json)

    # ── Find the resonance model ─────────────────────────────────
    model = None
    res_name = None
    res_disp = None
    for chain in f.config.full_decay.chains:
        for decay in chain.decays[1:]:
            m = decay.core._model
            nm = decay.core.name
            if nm == args.resonance:
                model = m
                res_name = nm
                res_disp = decay.core.display
                break
        if model:
            break

    if model is None:
        print(f"Resonance '{args.resonance}' not found")
        sys.exit(1)

    mass_key = f"{res_name}_mass"
    gamma_names = list(model.get_gamma_name())
    param_names = [mass_key] + gamma_names

    # ── Mass grid ────────────────────────────────────────────────
    m_pi = 0.13957
    m_B = 5.279
    mass_lo = args.mass_lo if args.mass_lo is not None else 2 * m_pi
    mass_hi = args.mass_hi if args.mass_hi is not None else m_B - m_pi
    m_grid = np.linspace(mass_lo, mass_hi, args.n_points)

    # ── Observable: Re(1/D) and Im(1/D) at all mass points ─────
    gamma_fn = model.gamma

    def lineshape_obs(phys):
        m0 = float(phys[mass_key])
        g0_vals = [float(phys[gn]) for gn in gamma_names]
        gamma_vals = np.asarray(gamma_fn(m_grid), dtype=complex)
        Gamma = np.zeros_like(m_grid, dtype=complex)
        for i in range(min(len(g0_vals), gamma_vals.shape[0])):
            Gamma += g0_vals[i] * gamma_vals[i]
        D = (m0**2 - m_grid**2) - 1j * m0 * Gamma
        invD = 1.0 / D
        # Interleave Re and Im: [Re_0, Im_0, Re_1, Im_1, ...]
        out = [0.0] * (2 * len(m_grid))
        out[0::2] = invD.real
        out[1::2] = invD.imag
        return out

    print(f"  Computing Re(1/D) and Im(1/D) + uncertainties for {res_name} ({args.n_points} points)...")
    values, errors = f.cal_uncertainties_multi_vec(
        lineshape_obs, param_names, result, return_cov=False)
    values = np.asarray(values)
    errors = np.asarray(errors)

    # Separate Re and Im components
    re_vals = values[0::2]
    im_vals = values[1::2]
    re_errs = errors[0::2]
    im_errs = errors[1::2]

    # ── Plot ─────────────────────────────────────────────────────
    fig, (ax_re, ax_im) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Re(1/D)
    ax_re.plot(m_grid, re_vals, "r-", linewidth=1.5, label=r"Re($1/D$)")
    ax_re.fill_between(m_grid, re_vals - re_errs, re_vals + re_errs,
                       alpha=0.25, color="r", label=r"1$\sigma$")
    ax_re.axhline(0, color="grey", linewidth=0.5)
    ax_re.set_ylabel(r"Re($1/D$)")
    ax_re.legend(fontsize=10)
    ax_re.grid(True, alpha=0.3)

    # Im(1/D)
    ax_im.plot(m_grid, im_vals, "b-", linewidth=1.5, label=r"Im($1/D$)")
    ax_im.fill_between(m_grid, im_vals - im_errs, im_vals + im_errs,
                       alpha=0.25, color="b", label=r"1$\sigma$")
    ax_im.axhline(0, color="grey", linewidth=0.5)
    ax_im.set_xlabel(r"$m$ (GeV)")
    ax_im.set_ylabel(r"Im($1/D$)")
    ax_im.legend(fontsize=10)
    ax_im.grid(True, alpha=0.3)

    fig.suptitle(f"{res_disp}  —  Re($1/D$) and Im($1/D$) with uncertainty band",
                 fontsize=11)
    fig.subplots_adjust(hspace=0.08)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {args.output}")


if __name__ == "__main__":
    main()

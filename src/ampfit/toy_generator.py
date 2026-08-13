"""Toy B → 4π event generator: sample phase space ∝ |A|².

Pipeline::

    1. generate flat phase space        (ampfit.phasespace_b4pi)
    2. convert 4-momenta → data arrays  (ampfit.momenta_to_data)
    3. compute the model probability P  (fitter backend, |A|² with
       per-event decay time and tagging fraction)
    4. rejection sampling with P/P_max  (max estimated from a
       calibration batch)

The accepted events are distributed like the model amplitude squared,
i.e. a "signal MC" sample of the configured decay model.  By default
the decay time follows the PDG B⁰ exponential
(τ = 1.519 ps → Γ = 0.658 ps⁻¹) and the tagging fraction is a random
B⁰ / B̄⁰ tag ∈ {0, 1}.

Usage::

    from ampfit import Fitter
    from ampfit.toy_generator import toy_generate

    f = Fitter("config_amp.yml", backend="cuda_v3_sparse")
    ev = toy_generate(f, 100000, fit_result=r, seed=7)   # from a fit
    ev = toy_generate(f, 100000, seed=7)                 # config defaults

    # ev["momenta"] : (n, 4, 4) accepted 4-momenta
    # ev["data"]    : the mass/q/angles dict (usable with Fitter.set_phsp)
"""

import numpy as np

from ampfit.phasespace_b4pi import generate_b4pi, M_B_MESON, M_PION
from ampfit.momenta_to_data import momenta_to_data

# PDG B⁰ mean lifetime and decay width (τ = 1.519 ps, Γ = 1/τ in ps⁻¹).
TAU_B0_PDG = 1.519
GAMMA_B0_PDG = 1.0 / TAU_B0_PDG


def _make_frac_time(n, frac, time, rng):
    """Per-event tagging fraction and decay time for a batch.

    *frac* = ``"random"`` → {0, 1} (B⁰ / B̄⁰ tag); *time* = ``"exp"`` →
    exponential with the PDG B⁰ width.  Scalars/arrays are passed
    through (broadcast to length *n*).
    """
    if frac == "random":
        frac_a = rng.choice([0.0, 1.0], size=n)
    elif frac is None:
        frac_a = np.full(n, 0.5)
    else:
        frac_a = np.asarray(frac, dtype=float)
        frac_a = np.full(n, float(frac_a)) if frac_a.ndim == 0 else frac_a
    if time == "exp":
        time_a = rng.exponential(TAU_B0_PDG, size=n)
    elif time is None:
        time_a = np.zeros(n)
    else:
        time_a = np.asarray(time, dtype=float)
        time_a = np.full(n, float(time_a)) if time_a.ndim == 0 else time_a
    return frac_a, time_a


def _build_params(fitter, fit_result=None, param_dict=None):
    """Kernel params from a fit result, a resolved dict, or defaults."""
    if fit_result is not None:
        params, _ = fitter.build_params(fit_result.x)
        return params
    # Ensure the constraint pipeline is set up (the config defaults give
    # non-trivial CK reference values only after apply_constrains).
    fitter.apply_constrains()
    if param_dict is not None:
        resolved = fitter.cm.resolve(dict(param_dict))
    else:
        resolved = fitter.cm.resolve({})
    return fitter._kernel_builder.forward(resolved)


def toy_generate(fitter, n_events, fit_result=None, param_dict=None,
                 seed=None, oversample=20, compute_batch=50000,
                 p_max_margin=1.3,
                 m_B=M_B_MESON, m_pi=M_PION,
                 frac="random", time="exp"):
    """Generate *n_events* B → 4π events distributed like |A|².

    Parameters
    ----------
    fitter : Fitter
        Configured fitter (the backend's `load_data`/`compute` are
        used, so a GPU backend is recommended for speed).
    n_events : int
        Number of accepted events to produce.
    fit_result : OptimizeResult, optional
        Fit result whose parameters define the amplitude.  If None and
        *param_dict* is None, the config defaults are used.
    param_dict : dict, optional
        Resolved physical parameter values (overrides defaults).
    seed : int, optional
        Random seed.
    oversample : int
        Kept for compatibility (flat events per batch); the actual batch
        size is *compute_batch*.
    compute_batch : int
        Flat events per GPU compute batch (also the calibration batch
        used to estimate P_max).
    p_max_margin : float
        Safety factor on the estimated P_max (default 1.3) — guards
        against under-estimating the true |A|² maximum of a peaked
        amplitude; a larger margin costs acceptance efficiency.
    m_B, m_pi : float
        B and pion masses for the flat phase-space generation.
    frac, time : "random" | "exp" | array-like
        Tagging fraction and decay time written into the data arrays.
        Default ``frac="random"`` → {0, 1} (B⁰ / B̄⁰ tag);
        ``time="exp"`` → exponential with the PDG B⁰ width
        Γ = 1/τ = 1/1.519 ps⁻¹.  Scalars/arrays pass through.

    Returns
    -------
    dict with:
        momenta : (n, 4, 4) accepted pion 4-momenta
        data    : the mass/q/angles dict (kernel-ready)
        P       : (n,) the model probability |A|² per accepted event
        n_flat  : total flat phase-space events generated
    """
    params = _build_params(fitter, fit_result, param_dict)
    rng = np.random.default_rng(seed)

    def get_P(momenta):
        n = len(momenta)
        frac_a, time_a = _make_frac_time(n, frac, time, rng)
        data = momenta_to_data(momenta, frac=frac_a, time=time_a)
        data = dict(data)
        data["angle"] = data.pop("angles")       # kernel expects "angle"
        data["mass"] = data["mass"].reshape(n, -1)   # (n, 48)
        data["q"] = data["q"].reshape(n, -1)         # (n, 72)
        handle = fitter.backend.load_data(data)
        try:
            _, _, P = fitter.backend.compute(params, handle, norm=None)
            return data, np.real(np.asarray(P)).astype(float)
        finally:
            if hasattr(handle, "free"):
                handle.free()

    # ── Calibration: estimate P_max from a compute_batch ───────────
    cal = generate_b4pi(compute_batch, m_B=m_B, m_pi=m_pi, seed=seed)
    _, P_cal = get_P(cal["momenta"])
    # safety margin against under-estimating the true max of a peaked
    # amplitude (events with P > P_max would be over-accepted)
    P_max = float(np.max(P_cal)) * p_max_margin

    accepted_mom, accepted_P, accepted_frac, accepted_time = [], [], [], []
    n_flat = compute_batch                       # calibration batch
    got = 0
    while got < n_events:
        batch_n = min(compute_batch, max(n_events - got, 1000))
        batch = generate_b4pi(batch_n, m_B=m_B, m_pi=m_pi,
                              seed=rng.integers(0, 2 ** 31))
        data, P = get_P(batch["momenta"])
        n_flat += len(P)
        u = rng.uniform(0.0, P_max, len(P))
        keep = np.flatnonzero(u < P)
        keep = keep[: n_events - got]
        if len(keep):
            accepted_mom.append(batch["momenta"][keep])
            accepted_P.append(P[keep])
            accepted_frac.append(data["frac"][keep])
            accepted_time.append(data["time"][keep])
            got += len(keep)

    momenta = np.concatenate(accepted_mom)
    P = np.concatenate(accepted_P)
    data = momenta_to_data(momenta, frac=np.concatenate(accepted_frac),
                           time=np.concatenate(accepted_time))
    data["angle"] = data.pop("angles")
    n = len(momenta)
    data["mass"] = data["mass"].reshape(n, -1)
    data["q"] = data["q"].reshape(n, -1)

    P_final = np.real(np.asarray(P)).astype(float)
    print(f"  accepted {len(momenta)} / {n_flat} flat "
          f"(eff {100 * len(momenta) / n_flat:.2f}%), "
          f"P_max bound = {P_max:.3f}, "
          f"max |A|² observed = {P_final.max():.3f}")
    return {"momenta": momenta, "data": data, "P": P_final,
            "n_flat": n_flat, "p_max": P_max}


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, json, os
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config_amp.yml")
    ap.add_argument("--n-events", type=int, default=100000)
    ap.add_argument("--backend", default="cuda_v3_sparse")
    ap.add_argument("--fit-result", default=None,
                    help="fit results.json to define the amplitude")
    ap.add_argument("-o", "--output", default="toy_signal.npz")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    from ampfit import Fitter
    f = Fitter(args.config, backend=args.backend)
    r = None
    if args.fit_result:
        cp = os.path.splitext(args.fit_result)[0] + "_constraints.json"
        if os.path.exists(cp):
            f.load_constraints(cp)
        r = f.load_results(args.fit_result)

    ev = toy_generate(f, args.n_events, fit_result=r, seed=args.seed)
    print(f"accepted {len(ev['momenta'])} events "
          f"(from {ev['n_flat']} flat, eff "
          f"{100 * len(ev['momenta']) / ev['n_flat']:.1f}%)")
    data = ev["data"]
    np.savez(args.output, mass=data["mass"], q=data["q"], angle=data["angle"],
             frac=data["frac"], time=data["time"], bkg_raw=data["bkg_raw"],
             weight=data["weight"])
    print(f"wrote {args.output}")

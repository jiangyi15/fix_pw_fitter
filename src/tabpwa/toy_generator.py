"""Toy B → 4π event generator: sample phase space ∝ |A|².

Pipeline::

    1. generate flat phase space        (tabpwa.phasespace_b4pi)
    2. convert 4-momenta → data arrays  (tabpwa.momenta_to_data)
    3. compute the model probability P  (fitter backend, |A|² with
       per-event decay time and tagging fraction)
    4. rejection sampling with P/P_max  (max estimated from a
       calibration batch)

The accepted events are distributed like the model amplitude squared,
i.e. a "signal MC" sample of the configured decay model.  The decay time and the
tagging fraction are the SAMPLE's law and come from ``data.extra_var``
(``random: {dist: …, **kwargs}``, one numpy call per column); an UNDECLARED
column falls back to a uniform time on ``[0, 10]`` ps (:data:`TIME_MAX`, a toy
window, not a lifetime) and a tagging fraction that is a choice of {0, 1}
(B⁰ / B̄⁰).  A config that wants the PDG lifetime declares
``{dist: exponential, scale: 1.519}``, and one that wants a constant declares
``{default: 1.0}``.

Usage::

    from tabpwa import Fitter
    from tabpwa.toy_generator import toy_generate

    f = Fitter("config_amp.yml", backend="cuda_v3_sparse")
    ev = toy_generate(f, 100000, fit_result=r, seed=7)   # from a fit
    ev = toy_generate(f, 100000, seed=7)                 # config defaults

    # ev["momenta"] : (n, 4, 4) accepted 4-momenta
    # ev["data"]    : the mass/q/angles dict (usable with Fitter.set_phsp)
"""

import numpy as np

from tabpwa.phasespace_b4pi import generate_b4pi, M_B_MESON, M_PION
from tabpwa.momenta_to_data import momenta_to_data

# PDG B⁰ mean lifetime and decay width (τ = 1.519 ps, Γ = 1/τ in ps⁻¹).
TAU_B0_PDG = 1.519
GAMMA_B0_PDG = 1.0 / TAU_B0_PDG


#: The fallback decay-time window (ps): an undeclared ``time`` draws U(0, TIME_MAX).
TIME_MAX = 10.0


def draw_extra(n, rng, spec=None, default=0.0):
    """Values for ONE extra variable, from its config ``random:`` spec.

    ``spec`` is the declaration in ``data.extra_var.<name>.random`` and is a
    NUMPY call, not a private mini-language::

        random: {dist: uniform, low: 0, high: 10}
        random: {dist: choice, a: [0, 1]}
        random: {dist: exponential, scale: 1.519}
        random: {dist: standard_normal}          # no extra arguments

    i.e. ``getattr(rng, spec["dist"])(size=n, **{the rest})``.  A bare string is
    read as ``{dist: <string>}``; ``None``/absent falls back to the ``default``
    (a constant, which is what a config declaring only ``default:`` means).
    """
    if spec is None or spec == "":
        return np.full(n, float(default))
    if isinstance(spec, str):
        spec = {"dist": spec}
    if not isinstance(spec, dict) or "dist" not in spec:
        raise ValueError(
            f"toy: extra_var random spec {spec!r} must be a mapping with a "
            f"'dist' key naming a numpy RandomState method")
    kwargs = {k: v for k, v in spec.items() if k != "dist"}
    fn = getattr(rng, str(spec["dist"]), None)
    if fn is None or not callable(fn):
        raise ValueError(
            f"toy: extra_var random spec {spec['dist']!r} is not a numpy "
            f"RandomState method (e.g. uniform / choice / exponential / "
            f"standard_normal)")
    try:
        return np.asarray(fn(size=n, **kwargs), float)
    except TypeError as exc:
        raise ValueError(
            f"toy: extra_var random spec {spec!r} rejected by numpy: {exc}")


def _build_params(fitter, fit_result=None, param_dict=None, x0=None):
    """Kernel params from a flat ``x0``, a fit result, a resolved dict, or defaults.

    ``x0`` is the FLAT parameter vector and goes through ``Fitter.build_params``
    exactly as a fit does, so a toy is generated at the point the caller asked
    for.  (A ``param_dict`` is a RESOLVED dict and is re-resolved by the
    constraint manager, which is why passing an already resolved one moved the
    point by ~2x on the masses.)
    """
    if x0 is not None:
        params, _ = fitter.build_params(x0)
        return params
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
                 frac=None, time=None, x0=None):
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
    frac, time : float, optional
        OVERRIDES for the two extra variables.  By default they come from the
        config's declaration, one entry per column::

            data:
              extra_var:
                time: {default: 1.0, random: "uniform(0, 10)"}
                frac: {default: 0.5, random: "choice(0, 1)"}

        — the same declaration a background sample is filled from, so the law is
        defined once, for the SAMPLE.  ``random:`` accepts
        ``uniform(a, b)``, ``choice(v, ...)``, ``const(v)``, ``exp(mean)``; a
        column declared with only ``default:`` is a constant.

    Returns
    -------
    dict with:
        momenta : (n, 4, 4) accepted pion 4-momenta
        data    : the mass/q/angles dict (kernel-ready)
        P       : (n,) the model probability |A|² per accepted event
        n_flat  : total flat phase-space events generated
    """
    params = _build_params(fitter, fit_result, param_dict, x0=x0)
    rng = np.random.RandomState(seed)

    def _override(val, n, name):
        arr = np.asarray(val, float)
        if arr.size == 1:
            return np.full(n, float(arr))
        if arr.size == n:
            return arr
        raise ValueError(
            f"toy: the {name}= override has {arr.size} values for a batch of "
            f"{n}; give a scalar, or declare the law in data.extra_var.{name}")

    def get_P(momenta):
        n = len(momenta)
        extra = fitter.model.mixing_inputs(n, rng)
        frac_a, time_a = extra.get("frac"), extra.get("time")
        if frac_a is None:                       # a model without extra columns
            frac_a = np.full(n, 0.5)
        if time_a is None:
            time_a = np.zeros(n)
        if frac is not None:                     # explicit override (scalar)
            frac_a = _override(frac, n, "frac")
        if time is not None:
            time_a = _override(time, n, "time")
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
                              seed=rng.randint(0, 2 ** 31))
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

    from tabpwa import Fitter
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

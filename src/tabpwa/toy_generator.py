"""Toy B → 4π event generator: sample phase space ∝ |A|².

Pipeline::

    1. generate flat phase space        (tabpwa.phasespace_b4pi)
    2. convert 4-momenta → data arrays  (tabpwa.momenta_to_data)
    3. compute the model probability P  (fitter backend, |A|² with
       per-event decay time and tagging fraction)
    4. rejection sampling with P/P_max  (max estimated from a
       calibration batch, RAISED when a later batch exceeds it and the
       accepted pool re-thinned by ``old/new`` — an under-estimate costs
       time, never correctness)

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
# the ceiling raise + pool thinning, the SAME helper the generic engine and its
# background use (a second copy could drift in the exact old/new weight)
from tabpwa.toy_pwa import ToyCache, ToyProgress, raise_ceiling, toy_cache_signature

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
                 p_max_margin=1.3, max_loops=None, stall_loops=1000,
                 m_B=M_B_MESON, m_pi=M_PION,
                 frac=None, time=None, x0=None, progress=None,
                 cache_dir=None):
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
        Safety factor on the estimated P_max (default 1.3).  MARGIN ONLY: a
        later batch that exceeds the ceiling raises it (and re-thins the
        accepted pool by ``old/new``), so an under-estimate costs time, never
        correctness — the same rule as the generic engine's ``maxp_scale``.
        Must be >= 1.
    max_loops : int, optional
        An OPTIONAL hard cap on proposal batches (``None`` = no cap).
    stall_loops : int
        The runaway guard: raise when this many CONSECUTIVE batches accept
        nothing (``compute_batch · stall_loops`` flat events with no
        acceptance; 50 M at the defaults).  A slow but PROGRESSING model has
        no cap, so a low-efficiency (peaked) amplitude runs as long as it
        keeps accepting — a total candidate budget would be wrong for it.
        Raise this for a genuinely tiny acceptance, or give ``max_loops``.
    progress : bool, optional
        Live progress lines while generating: ``True`` forces them, ``False``
        silences them, ``None`` (default) is AUTO — on when stdout is a
        terminal.  One throttled line per second (the generic engine's
        :class:`tabpwa.toy_pwa.ToyProgress`), so a long run does not look like
        a hang.
    cache_dir : str, optional
        Checkpoint the accepted pool here and RESUME from it: each accepted
        batch is appended as ``chunk_NNNNNN.npz``, and the counters, the
        ceiling and the RNG state land in ``state.json`` after it, so a run
        killed by a wall clock or a crash continues where it stopped on the next
        call with the same directory and the same (target, seed, generating
        point, model config) — see :class:`tabpwa.toy_pwa.ToyCache`.  A
        different run in the same directory raises.  Needs an EXPLICIT seed.
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
    if cache_dir is not None and seed is None:
        raise ValueError(
            "toy: cache_dir needs an explicit seed — the seed is part of what a "
            "resume must match, and an unseeded run would draw a different "
            "sample into the same directory")
    params = _build_params(fitter, fit_result, param_dict, x0=x0)
    rng = np.random.RandomState(seed)
    cache = (None if cache_dir is None else ToyCache(
        cache_dir, toy_cache_signature(fitter, "toy_generator", n_events,
                                       -1 if seed is None else seed, params)))

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
    if int(compute_batch) < 1:
        raise ValueError(f"toy: compute_batch = {compute_batch!r} must be >= 1")
    if int(stall_loops) < 0:
        raise ValueError(f"toy: stall_loops = {stall_loops!r} must be >= 0")
    if not p_max_margin >= 1.0:
        raise ValueError(
            f"toy: p_max_margin = {p_max_margin!r} < 1 — the guessed maximum "
            f"must be an over-estimate (use >= 1, e.g. 1.3)")
    limit = float("inf") if max_loops is None else int(max_loops)
    stall = int(stall_loops)

    cal = generate_b4pi(compute_batch, m_B=m_B, m_pi=m_pi, seed=seed)
    _, P_cal = get_P(cal["momenta"])
    P_max = float(np.max(P_cal)) * p_max_margin

    # the accepted pool, in the columns the result is rebuilt from; a dict of
    # lists so the ceiling raise can re-thin it with the ONE shared helper
    parts = {"momenta": [], "P": [], "frac": [], "time": []}
    n_flat = compute_batch                       # calibration batch
    got = loops = last_accept = 0
    if cache is not None:
        restored = cache.resume()
        if restored is not None:
            for k, v in restored["parts"].items():   # empty => keep the 4 keys
                parts[k] = list(v)
            got, n_flat = restored["have"], restored["tried"]
            loops, last_accept = restored["loops"], restored["last_accept"]
            if restored["guess"] is not None:
                P_max = float(restored["guess"])
            if restored["rng_state"] is not None:
                rng.set_state(restored["rng_state"])
            print(f"toy: resuming from {cache.dir} — {got}/{n_events} accepted, "
                  f"{loops} batches, {n_flat} flat")
    prog = ToyProgress(n_events, enabled=progress)
    while got < n_events and loops < limit:
        batch_n = min(compute_batch, max(n_events - got, 1000))
        batch = generate_b4pi(batch_n, m_B=m_B, m_pi=m_pi,
                              seed=rng.randint(0, 2 ** 31))
        data, P = get_P(batch["momenta"])
        n_flat += len(P)
        raised = False
        bmax = float(P.max()) if P.size else 0.0
        if bmax > P_max:                     # exceeded -> raise + thin
            new = p_max_margin * bmax
            got = raise_ceiling(rng, P_max, new, parts, got)
            P_max = new
            raised = True
        u = rng.uniform(0.0, P_max, len(P))
        keep = np.flatnonzero(u < P)
        keep = keep[: n_events - got]
        chunk = None
        if len(keep):
            chunk = {"momenta": batch["momenta"][keep], "P": P[keep],
                     "frac": data["frac"][keep], "time": data["time"][keep]}
            for k, v in chunk.items():
                parts[k].append(v)
            got += len(keep)
        if len(keep) or raised:
            last_accept = loops + 1
        loops += 1
        if cache is not None:
            cache.checkpoint(chunk, have=got, tried=n_flat, loops=loops,
                             last_accept=last_accept, guess=P_max, rng=rng)
        prog.update(got, n_flat, loops, P_max)
        if stall and loops - last_accept >= stall:
            raise RuntimeError(
                f"toy: no event accepted for {stall} consecutive batches "
                f"({n_flat} flat events tried, {got}/{n_events} accepted) — "
                f"the ceiling (p_max={P_max:.6g}) may be far above the true "
                f"|A|² maximum, or the density is zero / has a very narrow "
                f"support.  Raise stall_loops for a genuinely tiny acceptance.")
    if got < n_events:
        raise RuntimeError(
            f"accepted only {got} of {n_events} events in {loops} loops "
            f"({n_flat} flat); raise max_loops or compute_batch")

    prog.done(got, n_flat, loops)
    if cache is not None:
        cache.checkpoint(None, have=got, tried=n_flat, loops=loops,
                         last_accept=last_accept, guess=P_max, rng=rng, done=True)
    momenta = np.concatenate(parts["momenta"])
    P = np.concatenate(parts["P"])
    data = momenta_to_data(momenta, frac=np.concatenate(parts["frac"]),
                           time=np.concatenate(parts["time"]))
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

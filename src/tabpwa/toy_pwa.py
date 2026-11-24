"""Pure-PWA toy data: accept-reject sampling of the model density.

The core of ``scripts/gen_toy_pwa.py`` as a reusable function: stream
flat phase space (the config masses, two-body product + inverse boost
chain), accept with probability ``P(x)/guess`` — the guess of the
maximum P starts at ``maxp_scale · max(batch P)`` and is raised (with
retrospective thinning of the already-accepted events) only when a
batch exceeds it — and stop as soon as ``n_data`` are accepted.

The density uses the **Fitter parameter layer** (constraints/defaults
via ``initial_values``/``build_params``) and the fitter's configured
backend — never the raw kernel directly.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tabpwa.pwa_build import generate_pwa_phsp


@dataclass(frozen=True)
class ToyStats:
    """What ONE generation run did — everything a caller may want to reuse.

    ``maxp`` is the acceptance ceiling (a property of the parameter point and the
    topology), so ``Fitter.generate_toy(..., maxp=stats.maxp)`` skips the first-batch
    estimate; ``truth`` is the parameter point the events were generated AT, so
    ``f.save_params(stats.truth, ...)`` is the start point that fits back to it.
    ``maxp``/``loops`` are ``None`` where the generator cannot report them (the
    time-dependent model owns its own accept-reject).
    """

    seed: int
    n_events: int
    tried: int
    acceptance: float
    truth: np.ndarray
    maxp: Optional[float] = None
    loops: Optional[int] = None


def _raise_ceiling(rng, guess, new, parts, have):
    """Re-thin an accepted pool when the ceiling rises from ``guess`` to ``new``.

    Every pool event was accepted under ``<= guess`` (the raise happens BEFORE its
    own batch is accepted), so the exact keep factor a_sample(new)/a_sample(old)
    is the constant ``guess/new`` — no event above ``guess`` can be in the pool.
    MEASURED on P = x^2 from a starved start: mean 0.7515 against the exact 0.75,
    KS p = 0.19.  ONE implementation, because the SIGNAL and the BACKGROUND both
    thin a pool of accepted events this way and a second copy could drift.
    """
    if not have:
        return have
    mask = rng.random(have) < (guess / new)
    for k in parts:
        parts[k] = [np.concatenate(parts[k])[mask]]
    return int(mask.sum())


def toy_pwa_generate(fitter, n_data, seed=None, batch_size=20000,
                     maxp_scale=1.1, max_loops=1000, x0=None, top_p=None,
                     maxp=None, verbose=False):
    """Generate toy data distributed like the model |A|².

    Args:
        fitter: configured Fitter (``apply_constrains`` already called);
            its backend computes the density.
        n_data: the number of accepted events to produce.
        seed: master seed; sub-seeds are seed+1+k (proposal batches),
            seed+2 (the parameter point when ``x0`` is None) and seed+3
            (event selection).
        batch_size: flat events generated per loop.
        maxp_scale: the safety factor on the estimated max P.
        max_loops: the runaway-loop guard (None = unlimited).
        x0: the generating parameter point (flat); default: the Fitter's
            ``initial_values(seed + 2)``.
        top_p: the top's momentum (GeV/c) as a ``(n, rng) -> (n, 3)``
            callable (e.g. ``phasespace.fix_beam``); ``None`` = rest frame.
        maxp: an initial ceiling, e.g. the ``maxp`` a previous call RETURNED.
            The ceiling is a property of the parameter point and the topology,
            so reusing it skips the first-batch estimate; it is still raised
            (with thinning) if a batch exceeds it, so an under-estimate costs
            time and never correctness.
        verbose: print the guess/acceptance progress.

    Returns:
        ``(data, stats)`` — the kernel-format data dict and a :class:`ToyStats`
        carrying the truth, the ceiling actually used (feed it back as
        ``maxp=stats.maxp``) and the run's counts.

    NO norm sample: the accept-reject needs only ``P`` and its ceiling (a
    constant factor cancels), so the normalisation integral is the FIT's
    business — ``Fitter.generate_phsp`` gives it.
    """
    if seed is None:
        s = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        s %= 2**31 - 1
    else:
        s = int(seed)
    if verbose:
        print(f"toy seed = {s}")

    model, kc = fitter.model, fitter.kernel_config
    chain = fitter.decay_tree.partial_waves()[0][1]

    def events(n, seed):
        """Momenta -> this model's event buffer (its own hook, so every model
        shape filled the same way the fit will read it)."""
        mom = generate_pwa_phsp(model, chain, n, seed=seed, top_p=top_p)
        return model.build_event_data(mom, kc)

    if not maxp_scale >= 1.0:
        # a ceiling below the maximum can never become an upper bound by
        # raising: new = scale * bmax < bmax, so the peak stays capped and the
        # sample is biased no matter how the pool is thinned
        raise ValueError(
            f"toy: maxp_scale = {maxp_scale!r} < 1 — the guessed maximum must "
            f"be an over-estimate (use >= 1, e.g. 1.1)")
    if x0 is None:
        x0 = fitter.initial_values(seed=s + 2)
    params, _resolved = fitter.build_params(x0)

    backend = fitter.backend
    rng = np.random.RandomState(s + 3)
    # the layout is the MODEL's: every column of its event buffer is carried
    # (mass/q/angle/weight and whatever the model adds — bkg, frac, time, …),
    # so the toy is a drop-in for set_data and can be combined with a background
    cols = None
    have = tried = loops = 0
    guess = None if maxp is None else float(maxp)   # ceiling of P
    limit = max_loops if max_loops is not None else float("inf")
    scale = float(maxp_scale)
    while have < n_data and loops < limit:
        ev = events(batch_size, s + 1 + loops)
        if cols is None:
            cols = {k: [] for k in ev}
        _, _, P = backend.compute(params, backend.load_data(ev),
                                  norm=None, return_p=True)
        if P is None:
            raise RuntimeError("backend compute returned no P")
        P = np.asarray(P, dtype=float)
        bmax = float(P.max()) if P.size else 0.0

        if guess is None:                          # first batch, or maxp=
            guess = scale * bmax
            if verbose:
                print(f"  loop {loops}: guess max P = {guess:.6g}")
        elif bmax > guess:                     # exceeded -> raise + thin
            new = scale * bmax
            before = have
            have = _raise_ceiling(rng, guess, new, cols, have)
            guess = new
            if verbose:
                print(f"  loop {loops}: guess max P -> {guess:.6g} "
                      f"(kept {have}/{before})")

        keep = ((rng.random(batch_size) * guess <= P) if guess and guess > 0
                else np.zeros(batch_size, dtype=bool))
        tried += batch_size
        nk = int(keep.sum())
        if nk:
            idx = np.flatnonzero(keep)[:n_data - have]
            for k in cols:
                cols[k].append(np.asarray(ev[k])[idx])
            have += len(idx)
        loops += 1
    if have < n_data:
        raise RuntimeError(
            f"accepted only {have} of {n_data} events in {loops} loops "
            f"({tried} tried); raise max_loops or batch_size")
    data = {k: np.concatenate(cols[k]) for k in cols}
    if verbose:
        print(f"accepted {have} events in {loops} loops "
              f"({have / max(1, tried):.1%}), max P = {guess:.6g}")
    stats = ToyStats(seed=s, n_events=have, tried=tried,
                     acceptance=have / max(1, tried), truth=x0,
                     maxp=float(guess), loops=loops)
    return data, stats


@dataclass(frozen=True)
class ToyBgStats(ToyStats):
    """A signal+background toy: the same counters, plus the split and the purity.

    ``tried``/``acceptance``/``loops`` describe the SIGNAL draw only (they are
    0/0.0/None when ``n_signal == 0``); ``bg_tried`` counts the flat candidates
    the background sampler consumed.
    """

    n_signal: int = 0
    n_background: int = 0
    purity: float = 1.0
    bg_tried: int = 0


def bkg_values(ev, bg):
    """The per-event background column ``b(x)`` for an event buffer."""
    n = len(ev["mass"])
    if bg is None:
        return np.ones(n)
    v = np.asarray(bg(ev), float)
    if v.shape != (n,):
        raise ValueError(
            f"toy: the background function must give {n} values, got {v.shape}")
    if not np.isfinite(v).all() or (v < 0).any():
        raise ValueError("toy: the background function must be finite and >= 0")
    return v


def generate_toy_bg(fitter, x0, n_data, *, purity=None, bg=None, seed=None,
                    top_p=None, maxp=None, stats=False, **kw):
    """Toy data from the MIXTURE ``p·P/norm + (1-p)·b(x)/N_b``, counts Poisson.

    ``n_sig ~ Poisson(n_data·p)`` and ``n_bg ~ Poisson(n_data·(1-p))``; the signal
    events come from the model (``toy_pwa_generate``), the background events from
    ``bg`` sampled over flat phase space.  ``purity`` defaults to the CONFIG's
    ``purity`` — the value ``set_data`` will use — and ``bg=None`` means a UNIFORM
    background (all ones).

    The fitter's ``bkg`` column is ADDITIVE at every event (``set_data`` scales it
    by ``(1-p)/(p·N_b)`` with ``N_b`` the phase-space mean), so this sets
    ``bkg = b(x)`` on EVERY event, signal ones included.  The norm sample is a
    SEPARATE call — build it with the SAME ``b`` so ``N_b`` is the mean of the same
    function::

        data, st = f.generate_toy_bg(x0, 4000, purity=0.8, bg=b, stats=True)
        f.set_phsp(f.generate_phsp(40_000, seed=8, bkg=b))   # same column
        f.set_data(data)

    The extra per-event inputs of the sample (a time-dependent model's
    ``frac``/``time``) are declared in the CONFIG — ``data.extra_var.<name>`` with
    a ``random:`` spec — and the SAME declaration fills the signal (through the
    model's generator) and the background rows: the law is a property of the
    sample, not of either component.

    Returns ``data`` (``(data, stats)`` with ``stats=True``); ``stats`` is a
    :class:`ToyBgStats` with ``n_signal``, ``n_background`` and the realised
    purity.

    ``bg`` is called on the EVENT BUFFER (the same dict the fit consumes), so it
    can be any function of ``mass``/``q``/``angle``.
    """
    if n_data < 1:
        raise ValueError(
            f"toy: n_data must be >= 1, got {n_data!r}")
    if purity is None:
        purity = fitter.config_purity()
    purity = float(purity)
    if not 0.0 <= purity <= 1.0:
        raise ValueError(f"toy: purity = {purity!r} is not in [0, 1]")

    if seed is None:
        s = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        s %= 2 ** 31 - 1
    else:
        s = int(seed)
    rng = np.random.RandomState(s)
    n_sig = int(rng.poisson(n_data * purity))
    n_bg = int(rng.poisson(n_data * (1.0 - purity)))
    if n_sig + n_bg == 0:
        raise ValueError(
            f"toy: the Poisson split realised 0 events "
            f"(n_sig={n_sig}, n_bg={n_bg}) at n_data={n_data}, purity={purity} "
            f"— use a larger n_data or another seed")

    # ── signal: the MODEL's own generator when it declares one ────────────
    hook = getattr(fitter.model, "toy_generator", None)
    hook = None if hook is None else hook()
    if n_sig:
        if hook is not None:
            data, st = hook(fitter, n_sig, seed=s, x0=x0, top_p=top_p,
                            maxp=maxp, **kw)
        else:
            data, st = toy_pwa_generate(fitter, n_sig, seed=s, x0=x0,
                                        top_p=top_p, maxp=maxp, verbose=False,
                                        **kw)
    else:
        data, st = {}, ToyStats(seed=s, n_events=0, tried=0,
                                acceptance=0.0, truth=np.asarray(x0, float))

    # ── background: flat phase space, accepted on b(x)/b_max ─────────────
    # The SAME algorithm as the signal (raise + thin): the raise happens before
    # its own batch is accepted, so every kept event has v <= its ceiling and the
    # constant factor old/new is exact.  A batch that merely MISSES a narrow
    # support must not be read as "b == 0 everywhere", so the hard error comes
    # only after a cumulative budget of all-zero draws.
    bg_parts = {}
    have, guess, loops, zero_seen, bg_tried = 0, None, 0, 0, 0
    chain = fitter.decay_tree.partial_waves()[0][1]
    while have < n_bg:
        loops += 1
        if loops > 100_000:
            raise RuntimeError("toy: the background function rejects everything")
        m = int(max(64, 4 * (n_bg - have)) if bg is not None else (n_bg - have))
        bg_tried += m
        # flat phase space, then the MODEL's background rows (its own layout,
        # and its own flat law for any extra per-event input such as frac/time)
        mom = generate_pwa_phsp(fitter.model, chain, m,
                               seed=int(rng.randint(2**31 - 1)), top_p=top_p)
        # the SAME per-event inputs as the signal: the model only declares which
        # columns it takes, the toy defines their law
        ev = fitter.model.build_event_data(
            mom, fitter.kernel_config,
            **fitter.model.mixing_inputs(m, rng))
        v = bkg_values(ev, bg)
        if bg is not None:
            vmax = float(v.max()) if v.size else 0.0
            if vmax <= 0.0:                    # this batch says nothing; retry
                zero_seen += m
                if have == 0 and zero_seen >= 1_000_000:
                    raise RuntimeError(
                        "toy: the background function is zero over "
                        f"{zero_seen} sampled events — its support is too small "
                        "for this n_data (raise n_data or widen the support)")
                continue
            if guess is None:
                guess = vmax * 1.0001
            elif vmax > guess:
                new = vmax * 1.0001
                have = _raise_ceiling(rng, guess, new, bg_parts, have)
                guess = new
            keep = rng.random(m) * guess <= v
        else:
            keep = np.ones(m, bool)
        take = np.flatnonzero(keep)[:n_bg - have]
        if take.size:
            have += take.size
            for k, arr in ev.items():
                bg_parts.setdefault(k, []).append(np.asarray(arr)[take])
    bg_data = ({k: np.concatenate(v)[:n_bg] for k, v in bg_parts.items()}
               if n_bg else {})

    # ── combine, with b(x) on EVERY event (the column is additive) ────────
    # The additive background column is OWNED here (``bkg = b(x)``), so the
    # model's own ``bkg``/``bkg_raw`` are dropped from both sides — the two
    # builders of one model do not necessarily agree on that name.
    drop = {"bkg", "bkg_raw"}
    if n_sig and n_bg:                       # signal first, then background
        keys = [k for k in data if k in bg_data and k not in drop]
        only = ((set(data) | set(bg_data)) - drop) - set(keys)
        if only:
            raise ValueError(
                "toy: the signal and background rows carry different columns "
                f"({sorted(only)}); the model's build_bkg_event_data must "
                "produce the same layout as its toy")
        bkg_col = np.concatenate([bkg_values(data, bg),
                                  bkg_values(bg_data, bg)])
        data = {k: np.concatenate([np.asarray(data[k], float),
                                   np.asarray(bg_data[k], float)])
                for k in keys}
    else:                                    # one side is empty: keep its shape
        src = data if n_sig else bg_data
        bkg_col = bkg_values(src, bg) if (n_sig or n_bg) else np.empty(0)
        data = {k: np.asarray(v, float) for k, v in src.items()
                if k not in drop}
    order = rng.permutation(n_sig + n_bg)
    data = {k: data[k][order] for k in data}
    data["bkg"] = bkg_col[order]

    out_stats = ToyBgStats(seed=s, n_events=n_sig + n_bg, tried=st.tried,
                           acceptance=st.acceptance, truth=st.truth,
                           maxp=st.maxp, loops=st.loops, n_signal=n_sig,
                           n_background=n_bg,
                           purity=n_sig / max(1, n_sig + n_bg),
                           bg_tried=bg_tried)

    return (data, out_stats) if stats else data

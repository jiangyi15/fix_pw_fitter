"""Toy data from a fitted model: accept–reject on ``|A|²``, plus the mixture.

MODEL-GENERAL despite the module name: every function here takes its rows from
the model's own event builder, so the same code serves ``pwa``, ``pwa_gen``,
``polarised_pwa`` and the time-dependent model.

``toy_pwa_generate`` is the engine: stream flat phase space (the config masses,
two-body product + inverse boost chain), accept with probability
``P(x)/guess`` — the guess of the maximum ``P`` starts at
``maxp_scale · max(batch P)`` and is raised (with retrospective thinning of the
already-accepted pool) only when a batch exceeds it — and stop as soon as
``n_data`` are accepted.  The density uses the **Fitter parameter layer**
(constraints/defaults via ``initial_values``/``build_params``) and the fitter's
configured backend, never the raw kernel.

``generate_toy_bg`` combines that with a background drawn from flat phase space
and accepted on ``b(x)``; ``bkg_values`` is the ONE evaluator of the shape
(``b(x)`` per event, UNIFORM when the function is ``None``), shared with the norm
sample so ``N_b`` is the mean of the same function.

The user-facing entry points are ``Fitter.generate_toy``,
``Fitter.generate_toy_bg`` and ``Fitter.generate_phsp``;
``scripts/gen_toy_pwa.py`` is the CLI over the engine.
"""

from dataclasses import dataclass
import hashlib
import json
import os
import sys
import time
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


def raise_ceiling(rng, guess, new, parts, have):
    """Re-thin an accepted pool when the ceiling rises from ``guess`` to ``new``.

    Every pool event was accepted under ``<= guess`` (the raise happens BEFORE its
    own batch is accepted), so the exact keep factor a_sample(new)/a_sample(old)
    is the constant ``guess/new`` — no event above ``guess`` can be in the pool.
    MEASURED on P = x^2 from a starved start: mean 0.7515 against the exact 0.75,
    KS p = 0.19.  ONE implementation: this engine's SIGNAL and BACKGROUND and the
    time-dependent engine (``toy_generator.toy_generate``) all thin a pool of
    accepted events this way, and a second copy could drift.
    """
    if not have:
        return have
    mask = rng.random(have) < (guess / new)
    for k in parts:
        parts[k] = [np.concatenate(parts[k])[mask]]
    return int(mask.sum())


def _digest_arrays(d):
    """A stable digest of a params/config dict (arrays by bytes, the rest by repr)."""
    h = hashlib.sha1()
    for k in sorted(d, key=str):
        v = d[k]
        h.update(str(k).encode())
        if isinstance(v, np.ndarray):
            h.update(f"{v.dtype}{v.shape}".encode())
            if v.size <= 100_000:          # the physics tables, not a sample
                h.update(np.ascontiguousarray(v).tobytes())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()


def toy_cache_signature(fitter, engine, target, seed, params):
    """What a toy cache belongs to: a resume must match ALL of it.

    A cache continued under different physics is silently wrong, and nothing
    downstream can notice (the events look fine), so the signature covers the
    engine (its chunk layout), the target, the seed, the GENERATING POINT (the
    resolved parameters — the ceiling and the density are properties of it),
    the model name and the kernel config's own tables.
    """
    return {
        "engine": str(engine),
        "target": int(target),
        "seed": int(seed),
        "params": _digest_arrays(params),
        "model": str(getattr(getattr(fitter, "model", None), "name", "")),
        "config": _digest_arrays(getattr(fitter, "kernel_config", {}) or {}),
    }


class ToyCache:
    """Checkpoint an accept-reject toy to a directory, and RESUME from it.

    A toy that needs 10^8 candidates is minutes-to-hours of a GPU; a wall-clock
    kill or a crash would otherwise throw all of it away.  Every accepted batch
    is its own ``chunk_NNNNNN.npz`` (append-only, so a torn write cannot damage
    an earlier one — and it is written ``.tmp`` then renamed) and ``state.json``
    records the counters, the ceiling and the RNG state, replaced atomically
    after every batch.  The next call with the SAME ``cache_dir`` and the SAME
    :func:`toy_cache_signature` CONTINUES that stream: restoring the RNG state
    and the loop counter keeps the sub-seeds — and therefore the sample law —
    identical, so a resumed run IS the run that was interrupted, not a fresh
    sample grafted onto an old one.

    A different signature RAISES (reusing a directory for another point would
    blend two samples); a directory whose run already finished starts over.
    ``state.json`` also gives a killed run's progress, loaded by ``resume()``.
    """

    STATE = "state.json"
    PREFIX = "chunk_"

    def __init__(self, path, signature):
        self.dir = os.path.abspath(os.fspath(path))
        os.makedirs(self.dir, exist_ok=True)
        self.sig = dict(signature)
        self.n_chunks = 0
        self._state = self._read_state()

    # ── files ─────────────────────────────────────────────────────────
    def _state_path(self):
        return os.path.join(self.dir, self.STATE)

    def _chunk_path(self, i):
        return os.path.join(self.dir, f"{self.PREFIX}{i:06d}.npz")

    def _read_state(self):
        path = self._state_path()
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        got = state.get("signature") or {}
        bad = sorted(k for k in set(got) | set(self.sig)
                     if got.get(k) != self.sig.get(k))
        if bad:
            raise ValueError(
                f"toy cache {self.dir}: this directory holds another run "
                f"({', '.join(f'{k}: {got.get(k)!r} != {self.sig.get(k)!r}' for k in bad)}) "
                f"— use a different cache_dir, or delete it to start over")
        if state.get("done"):
            # the SAME run already completed here (same signature == same
            # sample); keeping it would only replay it, so start clean
            for name in os.listdir(self.dir):
                if name.startswith(self.PREFIX) and name.endswith(".npz"):
                    os.unlink(os.path.join(self.dir, name))
            os.unlink(path)
            return None
        self.n_chunks = int(state.get("n_chunks", 0))
        return state

    def resume(self):
        """The restored pool and counters, or ``None`` when starting fresh."""
        if self._state is None:
            return None
        parts, seen = {}, None
        for i in range(self.n_chunks):
            path = self._chunk_path(i)
            if not os.path.exists(path):
                raise ValueError(
                    f"toy cache {self.dir}: {self.STATE} claims {self.n_chunks} "
                    f"chunks but {os.path.basename(path)} is missing — delete "
                    f"the directory and regenerate")
            with np.load(path, allow_pickle=False) as npz:
                keys = sorted(npz.files)
                if seen is None:
                    seen = keys
                elif keys != seen:
                    raise ValueError(
                        f"toy cache {self.dir}: chunk {i} has columns {keys}, "
                        f"chunk 0 has {seen}")
                for k in keys:
                    parts.setdefault(k, []).append(np.asarray(npz[k]))
        st = self._state
        rng_state = st.get("rng")
        if rng_state is not None:
            rng_state = (str(rng_state[0]), np.asarray(rng_state[1], dtype=np.uint32),
                         int(rng_state[2]), int(rng_state[3]), float(rng_state[4]))
        return {"parts": parts, "have": int(st["have"]), "tried": int(st["tried"]),
                "loops": int(st["loops"]),
                "last_accept": int(st.get("last_accept", 0)),
                "guess": st.get("guess"), "rng_state": rng_state}

    def checkpoint(self, chunk, have, tried, loops, last_accept, guess, rng,
                   done=False):
        """Append ``chunk`` (or nothing) and publish the counters atomically."""
        n_new = 0 if chunk is None else len(next(iter(chunk.values())))
        if n_new:
            tmp = self._chunk_path(self.n_chunks) + ".tmp"
            with open(tmp, "wb") as fh:
                np.savez(fh, **chunk)
            os.replace(tmp, self._chunk_path(self.n_chunks))
            self.n_chunks += 1
        rng_json = None
        if rng is not None:
            name, keys, pos, has_gauss, cached = rng.get_state()
            rng_json = [str(name), np.asarray(keys).tolist(), int(pos),
                        int(has_gauss), float(cached)]
        state = {
            "signature": self.sig, "have": int(have), "tried": int(tried),
            "loops": int(loops), "last_accept": int(last_accept),
            "guess": None if guess is None else float(guess),
            "n_chunks": int(self.n_chunks), "done": bool(done),
            "rng": rng_json,
        }
        tmp = self._state_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp, self._state_path())
        self._state = state


class ToyProgress:
    """Throttled progress reporting for the toy generators (opt-in).

    A run of 10^8 candidates is minutes of a silent process, which reads as a
    hang; this prints ONE line per ``interval`` seconds (and one at the end):
    accepted/target, batches so far, flat events tried, acceptance and the
    ceiling.  On a terminal the line is rewritten in place (``\r``), in a log it
    is a new line each time — so it is usable from a script and from a job
    file.  ``enabled=None`` means AUTO: on when the stream is a terminal, off
    in a pipeline or under a test runner.

    ONE implementation: both engines (``toy_pwa_generate`` and
    ``toy_generator.toy_generate``) report through this, so the two paths show
    the same fields.
    """

    def __init__(self, target, enabled=None, interval=1.0, stream=None):
        self.stream = sys.stdout if stream is None else stream
        self.target = max(1, int(target))
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.enabled = self.tty if enabled is None else bool(enabled)
        self.interval = max(0.0, float(interval))
        self.t0 = time.monotonic()
        self.t_last = None
        self.have_last = 0
        self.width = 0
        self.wrote = False

    def update(self, have, tried, loops, guess=None, force=False):
        if not self.enabled:
            return
        now = time.monotonic()
        if (not force and self.t_last is not None
                and now - self.t_last < self.interval):
            return
        rate = ((have - self.have_last) / (now - self.t_last)
                if self.t_last is not None and now > self.t_last else float("nan"))
        self.t_last, self.have_last = now, have
        msg = (f"toy: {have}/{self.target} accepted "
               f"({100.0 * have / self.target:5.1f}%), "
               f"{loops} batches, {tried} flat, "
               f"eff {100.0 * have / tried if tried else 0.0:.2f}%"
               + (f", {rate:.0f}/s" if rate == rate else "")
               + f", {now - self.t0:.0f} s")
        if guess is not None:
            msg += f", ceiling {guess:.6g}"
        if self.tty:
            pad = " " * max(0, self.width - len(msg))
            print(msg + pad, end="\r", file=self.stream, flush=True)
            self.width = len(msg)
        else:
            print(msg, file=self.stream, flush=True)
        self.wrote = True

    def done(self, have, tried, loops):
        if not self.enabled:
            return
        self.update(have, tried, loops, None, force=True)
        if self.tty and self.wrote:
            print(file=self.stream, flush=True)


def toy_pwa_generate(fitter, n_data, seed=None, batch_size=20000,
                     maxp_scale=1.1, max_loops=None, stall_loops=1000,
                     x0=None, top_p=None, maxp=None, verbose=False,
                     progress=None, cache_dir=None):
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
        max_loops: an OPTIONAL hard cap on proposal batches (None = no cap).
        stall_loops: the runaway guard — raise when this many CONSECUTIVE
            batches accept NOTHING (`batch_size · stall_loops` flat events
            without a single acceptance — 20 M at the defaults).  That is a
            no-progress window, not a total budget: a peaked amplitude that
            needs 10^9 candidates runs as long as it keeps accepting.
        x0: the generating parameter point (flat); default: the Fitter's
            ``initial_values(seed + 2)``.
        top_p: the top's momentum (GeV/c) as a ``(n, rng) -> (n, 3)``
            callable (e.g. ``phasespace.fix_beam``); ``None`` = rest frame.
        maxp: an initial ceiling, e.g. the ``maxp`` a previous call RETURNED.
            The ceiling is a property of the parameter point and the topology,
            so reusing it skips the first-batch estimate; it is still raised
            (with thinning) if a batch exceeds it, so an under-estimate costs
            time and never correctness.
        verbose: print a line for EVERY batch (the ceiling moves).
        cache_dir: checkpoint the accepted pool here, and RESUME from it.  Every
            accepted batch is appended as ``chunk_NNNNNN.npz`` and the counters,
            the ceiling and the RNG state are published in ``state.json`` after
            it, so a run killed by a wall clock or a crash continues where it
            stopped on the next call with the same directory and the same
            (target, seed, generating point, model config) — see
            :class:`ToyCache`.  A different run in the same directory raises.
        progress: live progress lines while generating — ``True`` forces them,
            ``False`` silences them, ``None`` (default) is AUTO: on when stdout
            is a terminal.  One throttled line per second (:class:`ToyProgress`)
            with accepted/target, batches, flat tried, efficiency, rate,
            elapsed and the ceiling, so a long run does not look like a hang.

    Returns:
        ``(data, stats)`` — the kernel-format data dict and a :class:`ToyStats`
        carrying the truth, the ceiling actually used (feed it back as
        ``maxp=stats.maxp``) and the run's counts.

    The returned buffer is the MODEL's own layout; its ``bkg`` column is
    whatever that model's builder writes (a data file's flat shape, or ``0`` for
    a signal-MC fill), so the background SHAPE is not this function's business:
    ``Fitter.generate_toy(fbg=…)`` sets it, in one evaluation that also covers a
    model's own generator.

    NO norm sample: the accept-reject needs only ``P`` and its ceiling (a
    constant factor cancels), so the normalisation integral is the FIT's
    business — ``Fitter.generate_phsp`` gives it.
    """
    if cache_dir is not None and seed is None:
        raise ValueError(
            "toy: cache_dir needs an explicit seed — the seed is part of what a "
            "resume must match, and an unseeded run would draw a different "
            "sample into the same directory")
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
    cache = (None if cache_dir is None else ToyCache(
        cache_dir, toy_cache_signature(fitter, "toy_pwa", n_data, s, _resolved)))
    # the layout is the MODEL's: every column of its event buffer is carried
    # (mass/q/angle/weight and whatever the model adds — bkg, frac, time, …),
    # so the toy is a drop-in for set_data and can be combined with a background
    cols = None
    have = tried = loops = 0
    guess = None if maxp is None else float(maxp)   # ceiling of P
    if int(batch_size) < 1:
        raise ValueError(f"toy: batch_size = {batch_size!r} must be >= 1")
    if int(stall_loops) < 0:
        raise ValueError(f"toy: stall_loops = {stall_loops!r} must be >= 0")
    limit = float("inf") if max_loops is None else int(max_loops)
    stall = int(stall_loops)
    last_accept = 0            # loop after which the pool last grew (or raised)
    if cache is not None:
        restored = cache.resume()
        if restored is not None:
            # an EMPTY cache (killed before the first acceptance) has no column
            # set; keep ``None`` so the first batch defines it
            cols = restored["parts"] or None
            have, tried = restored["have"], restored["tried"]
            loops, last_accept = restored["loops"], restored["last_accept"]
            if restored["guess"] is not None:
                guess = float(restored["guess"])
            if restored["rng_state"] is not None:
                rng.set_state(restored["rng_state"])
            print(f"toy: resuming from {cache.dir} — {have}/{n_data} accepted, "
                  f"{loops} batches, {tried} flat")
    scale = float(maxp_scale)
    prog = ToyProgress(n_data, enabled=progress)
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

        raised = False
        if guess is None:                          # first batch, or maxp=
            guess = scale * bmax
            if verbose:
                print(f"  loop {loops}: guess max P = {guess:.6g}")
        elif bmax > guess:                     # exceeded -> raise + thin
            new = scale * bmax
            before = have
            have = raise_ceiling(rng, guess, new, cols, have)
            guess = new
            raised = True
            if verbose:
                print(f"  loop {loops}: guess max P -> {guess:.6g} "
                      f"(kept {have}/{before})")

        keep = ((rng.random(batch_size) * guess <= P) if guess and guess > 0
                else np.zeros(batch_size, dtype=bool))
        tried += batch_size
        nk = int(keep.sum())
        chunk = None
        if nk:
            idx = np.flatnonzero(keep)[:n_data - have]
            chunk = {k: np.asarray(ev[k])[idx] for k in cols}
            for k in cols:
                cols[k].append(chunk[k])
            have += len(idx)
        if nk or raised:
            last_accept = loops + 1
        loops += 1
        if cache is not None:
            cache.checkpoint(chunk, have=have, tried=tried, loops=loops,
                             last_accept=last_accept, guess=guess, rng=rng)
        prog.update(have, tried, loops, guess)
        if stall and loops - last_accept >= stall:
            raise RuntimeError(
                f"toy: no event accepted for {stall} consecutive batches "
                f"({tried} flat events tried, {have}/{n_data} accepted) — the "
                f"ceiling (maxp={guess:.6g}) may be far above the true |A|² "
                f"maximum, or the density is zero / has a very narrow support.  "
                f"Raise stall_loops for a genuinely tiny acceptance, or pass "
                f"maxp= from a previous run at the same point.")
    if have < n_data:
        raise RuntimeError(
            f"accepted only {have} of {n_data} events in {loops} loops "
            f"({tried} tried); raise max_loops or batch_size")
    prog.done(have, tried, loops)
    if cache is not None:
        cache.checkpoint(None, have=have, tried=tried, loops=loops,
                         last_accept=last_accept, guess=guess, rng=rng, done=True)
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
    can be any function of ``mass``/``q``/``angle`` — and, for a time-dependent
    model, of ``frac``/``time``.  ``Fitter.generate_toy(fbg=…)`` takes the SAME
    function for a signal-only sample (uniform by default): one quantity, the
    background shape, under the name each entry point inherited.
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
                                        top_p=top_p, maxp=maxp, **kw)
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
                have = raise_ceiling(rng, guess, new, bg_parts, have)
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

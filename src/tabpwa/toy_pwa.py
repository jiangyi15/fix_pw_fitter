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

import numpy as np

from tabpwa.pwa_build import build_tree_event_data, generate_pwa_phsp

_STORE = ("mass", "q", "angle", "weight")


def toy_pwa_generate(fitter, n_data, seed=None, n_phsp=0, batch_size=20000,
                     maxp_scale=1.1, max_loops=1000, x0=None, verbose=False):
    """Generate toy data distributed like the model |A|².

    Args:
        fitter: configured Fitter (``apply_constrains`` already called);
            its backend computes the density.
        n_data: the number of accepted events to produce.
        seed: master seed; sub-seeds are seed (phsp/norm), seed+1+k
            (proposal batches), seed+2 (the parameter point when ``x0``
            is None), seed+3 (event selection).
        n_phsp: flat phase-space (norm) events to also produce
            (0 = skip).
        batch_size: flat events generated per loop.
        maxp_scale: the safety factor on the estimated max P.
        max_loops: the runaway-loop guard (None = unlimited).
        x0: the generating parameter point (flat); default: the Fitter's
            ``initial_values(seed + 2)``.
        verbose: print the guess/acceptance progress.

    Returns:
        ``(data, phsp, x0)`` — the kernel-format data dict, the flat
        phase-space dict (``None`` when ``n_phsp == 0``), and the
        generating parameter point (the truth).
    """
    if seed is None:
        s = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        s %= 2**31 - 1
    else:
        s = int(seed)
    if verbose:
        print(f"toy seed = {s}")

    model, kc = fitter.model, fitter.kernel_config
    tree = fitter.decay_tree
    chain = tree.partial_waves()[0][1]
    byt = {tree.topo_index[ch.topo_id()]: ch
           for _, ch in tree.partial_waves()}

    phsp = None
    if n_phsp > 0:
        phsp = build_tree_event_data(
            model, kc, byt,
            generate_pwa_phsp(model, chain, n_phsp, seed=s))

    if x0 is None:
        x0 = fitter.initial_values(seed=s + 2)
    params, _resolved = fitter.build_params(x0)

    backend = fitter.backend
    rng = np.random.RandomState(s + 3)
    cols = {k: [] for k in _STORE}
    have = tried = loops = 0
    guess = None
    limit = max_loops if max_loops is not None else float("inf")
    scale = float(maxp_scale)
    while have < n_data and loops < limit:
        ev = build_tree_event_data(
            model, kc, byt,
            generate_pwa_phsp(model, chain, batch_size, seed=s + 1 + loops))
        _, _, P = backend.compute(params, backend.load_data(ev),
                                  norm=None, return_p=True)
        if P is None:
            raise RuntimeError("backend compute returned no P")
        P = np.asarray(P, dtype=float)
        bmax = float(P.max()) if P.size else 0.0

        if guess is None:
            guess = scale * bmax
            if verbose:
                print(f"  loop {loops}: guess max P = {guess:.6g}")
        elif bmax > guess:                     # exceeded -> raise + thin
            new = scale * bmax
            before = have
            if have:
                mask = rng.random(have) < (guess / new)
                for k in _STORE:
                    cols[k] = [np.concatenate(cols[k])[mask]]
                have = int(mask.sum())
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
            for k in _STORE:
                cols[k].append(ev[k][idx])
            have += len(idx)
        loops += 1
    if have < n_data:
        raise RuntimeError(
            f"accepted only {have} of {n_data} events in {loops} loops "
            f"({tried} tried); raise max_loops or batch_size")
    data = {k: np.concatenate(cols[k]) for k in _STORE}
    if verbose:
        print(f"accepted {have} events in {loops} loops "
              f"({have / max(1, tried):.1%})")
    return data, phsp, x0

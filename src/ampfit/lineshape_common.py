"""Shared helpers for the B → R π⁻, R → 3π lineshape scripts.

Common code between ``scripts/calc_3pi_lineshape.py`` and
``scripts/reproduce_partial.py``: locating the resonance's chains in
the fit config, the particle model, the B → R π⁻ Blatt-Weisskopf
barrier, and the chain amplitude on a flat B → 4π sample restricted
to the resonance's waves (blocks 0 and 2 = B⁰ with the
π⁺₁↔π⁺₂ identical-particle swap; the π⁻ bachelor is never swapped).
"""

import numpy as np

from ampfit.momenta_to_data import momenta_to_data
from ampfit.toy_generator import _build_params

PION_NAMES = {"pip1", "pim1", "pip2", "pim2"}   # final-state pion slots
_PIP = PION_NAMES                                # back-compat alias


def find_resonance(f, res_name):
    """Find all ``B → R + π⁻`` chains whose top-level resonance is
    *res_name*.

    Returns ``(chains, bachelor)`` where *chains* is the list of
    ``(start, end, chain)`` block-0 wave ranges (unioned for the
    amplitude) and *bachelor* the (unique) π⁻ slot name.
    """
    chains = []
    bachelor = None
    for start, end, chain in f.config._chain_ranges():
        outs = [o.name for o in chain.decays[0].outs]
        res_outs = [o for o in outs if o not in PION_NAMES]
        pion_outs = [o for o in outs if o in PION_NAMES]
        if not res_outs or res_outs[0] != res_name or not pion_outs:
            continue
        b = pion_outs[0]
        if not b.startswith("pim"):            # want B -> R + π⁻
            continue
        if bachelor is None:
            bachelor = b
        elif bachelor != b:
            raise ValueError(f"{res_name}: mixed bachelors {bachelor} vs {b}")
        chains.append((start, end, chain))
    if not chains:
        raise ValueError(f"no B -> R + pi- chain with R = {res_name!r}")
    return chains, bachelor


def resonance_model(f, res_name):
    """The particle model of the top-level resonance *res_name*."""
    for chain in f.config.full_decay.chains:
        for d in chain.decays:
            if d.core.name == res_name:
                return d.core._model
    return None


def b_barrier_factor(f, chain_wave, qB):
    """F_L(qB) — the Blatt-Weisskopf barrier of the B→Rπ decay for the
    chain's first wave (L read from the kernel config)."""
    from ampfit.bw_form_factor import form_factor
    kc = f.kernel_config
    p = int(kc["fl_order"][chain_wave * 3])     # B decay (idx 0)
    L = int(kc["fl_type"][p])
    return form_factor(L, qB)


def inv_mass2(p):
    """Invariant mass² of a stack of 4-momenta ``(E, px, py, pz)``."""
    return p[:, 0] ** 2 - np.sum(p[:, 1:] ** 2, axis=-1)


def inv_mass(p):
    """Invariant mass of a stack of 4-momenta (0 above threshold)."""
    return np.sqrt(np.maximum(inv_mass2(p), 0.0))


def resonance_indices(bachelor):
    """Momentum-slot indices of the 3 pions forming R (excluding the
    bachelor π⁻).

    The momentum order is ``(pip1, pim1, pip2, pim2)``; the bachelor
    is a π⁻ (slot 3 for ``pim2``, slot 1 for ``pim1``).
    """
    b_idx = 3 if bachelor == "pim2" else 1
    return [i for i in range(4) if i != b_idx]


def fitted_ck(f, params, ck_ranges, n_base):
    """ck vector with the *fitted* values on the resonance's blocks
    {0, 2} (the π⁺₁↔π⁺₂ swap), zero elsewhere."""
    ck = np.zeros_like(params["ck"], dtype=complex)
    for start, end in ck_ranges:
        for b in (0, 2):
            s = b * n_base + start
            ck[s:s + (end - start)] = params["ck"][s:s + (end - start)]
    return ck


def chain_amplitude(f, fit_result, momenta, ck_ranges, n_base, frac, time):
    """|A|² of the special R's waves on a flat B → 4π sample.

    Includes the identical-particle permutation of the inner 3π
    (blocks 0 and 2) but never swaps the π⁻ bachelor (blocks 1, 3);
    all other CK couplings are zeroed.  *frac*, *time* are the fixed
    per-event tagging fraction and decay time (default frac=0.5,
    time=0).
    """
    params = _build_params(f, fit_result)
    params = dict(params)
    params["ck"] = fitted_ck(f, params, ck_ranges, n_base)

    n = len(momenta)
    data = momenta_to_data(momenta, frac=np.full(n, frac),
                           time=np.full(n, time))
    data["angle"] = data.pop("angles")
    data["mass"] = data["mass"].reshape(n, -1)
    data["q"] = data["q"].reshape(n, -1)
    handle = f.backend.load_data(data)
    try:
        _, _, P = f.backend.compute(params, handle, norm=None)
        return np.real(np.asarray(P)).astype(float)
    finally:
        if hasattr(handle, "free"):
            handle.free()

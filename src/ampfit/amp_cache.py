"""
amp_cache — minimal-set angular-amplitude cache layout.

For the ``cuda_v3_ampcache`` backend the per-event per-wave amplitude is
factorized as ``BW_w(m) · Amp_w(q, angles)`` where the angular factor

    Amp_w(e) = fa_w(e) · fl_w(e)
      fa_w = Σ_k ka_k · matrix_angle[k, w]      (angles only)
      fl_w = Π_d interp(fl_table, q[...])        (breakup momentum q only)

is pure per-event kinematics and can be cached once per data handle,
while ``BW_w(m)`` (the Breit-Wigner propagator, depends on the fitted
m0/g0 and per-event mass m) is recomputed every fit iteration.

``Amp_w`` is identical for waves that share the same matrix_angle column
*and* the same fl_order row.  Those waves map to one cache slot, so the
per-event cache stores only the minimal unique set (272 for the current
configs) instead of one value per wave (448/496).

Functions
---------
build_amp_cache_layout : map waves to minimal cache slots
fill_amp_cache         : compute the cached Amp values for a batch
amp_factor_cached      : full spatial factor from a filled cache (numpy reference)
"""

import numpy as np


def build_amp_cache_layout(config):
    """Wave → minimal angular-cache-slot mapping.

    Two waves share a slot iff they have the identical matrix_angle
    column (same angular basis combination → same ``fa``) AND the
    identical fl_order row (same form-factor momentum slots → same
    ``fl``).  This is the minimal set of ``Amp`` values per event.

    Args:
        config: kernel config dict (from Config.build_all_index()).

    Returns:
        ``(n_uniq, slot_of_wave, rep_of_slot)`` where *slot_of_wave* is
        int32[ n_wave ] (wave → slot), *rep_of_slot* is int32[ n_uniq ]
        (slot → a representative wave used to compute the cached value),
        and *n_uniq* is the number of unique (column, fl-row) pairs.
    """
    n_wave = config["matrix_angle"].shape[1]
    n_decay = config["fl_order"].size // n_wave
    cols = config["matrix_angle"].T                    # (n_wave, n_basis)
    fl_rows = np.asarray(config["fl_order"]).reshape(n_wave, n_decay)
    sig = [(cols[w].tobytes(), fl_rows[w].tobytes()) for w in range(n_wave)]
    keys = {}
    slot_of_wave = np.zeros(n_wave, dtype=np.int32)
    rep_of_slot = []
    for w in range(n_wave):
        s = sig[w]
        if s not in keys:
            keys[s] = len(rep_of_slot)
            rep_of_slot.append(w)
        slot_of_wave[w] = keys[s]
    return len(rep_of_slot), slot_of_wave, np.array(rep_of_slot, dtype=np.int32)


def fill_amp_cache(data, config, kernel, rep_of_slot):
    """Compute the cached angular amplitudes ``Amp = fa · fl`` per event.

    ``fa`` is evaluated for all waves in one matmul over the angular
    basis and ``fl`` for all waves in one gather; only the representative
    wave of each slot is kept, giving the minimal per-event cache.  Group
    members are byte-identical by construction (same matrix_angle column
    and same fl_order row), so the representative value is exact for the
    whole slot.

    Args:
        data: dict with keys ``q``, ``angle`` (batched arrays).
        config: kernel config dict.
        kernel: a NumpyKernel instance (for the Catmull-Rom interp).
        rep_of_slot: int32[ n_uniq ] slot → representative wave.

    Returns:
        complex ndarray (n_events, n_uniq) of cached Amp values.
    """
    momentum = data["q"]
    angle = data["angle"]
    n_events = momentum.shape[0]
    n_uniq = len(rep_of_slot)
    n_wave = config["matrix_angle"].shape[1]
    n_decay = config["fl_order"].size // n_wave

    # fa for ALL waves (vectorised over the 336 angular basis functions)
    ang = np.take(angle, config["angle_index"], axis=-2)
    ka = np.prod(np.cos(ang * config["angle_k"] + config["angle_b"]), axis=-1)
    fa = ka @ config["matrix_angle"]                     # (n_events, n_wave)

    # fl for ALL waves (form factors at the per-decay momentum slots)
    fl_q = np.take(momentum, config["fl_q_index"], axis=-1)
    fl = kernel.interp_catmull_rom(
        config["fl_table"], config["fl_type"], fl_q,
        config["fl_min"], config["fl_delta"])
    fl_all = np.take(fl, config["fl_order"], axis=-1)
    fl_p = np.prod(fl_all.reshape(n_events, n_wave, n_decay), axis=-1)

    return (fa * fl_p)[:, rep_of_slot]                   # (n_events, n_uniq)


def amp_factor_cached(data, config, kernel, m0, g0, slot_of_wave,
                      amp_cache=None):
    """Spatial factor ``common_amp = (fa·fl)/bw_p`` from a filled cache.

    Numpy reference for the cached forward: read the angular amplitude
    from the cache by wave→slot index and recompute only the BW
    propagator denominator per wave.

    Args:
        data: dict with keys ``mass``, ``q``, ``angle``.
        config: kernel config dict.
        kernel: NumpyKernel instance (interp helpers).
        m0, g0: fitted mass/width parameter arrays — must be UNIQUE-length
            (``max(index)+1``), the kernel-style scatter convention, not
            the expanded length.
        slot_of_wave: int32[ n_wave ].
        amp_cache: complex (n_events, n_uniq) from :func:`fill_amp_cache`;
            if None it is computed here (touching ``q``/``angle``).

    Returns:
        complex ndarray (n_events, n_wave) of common_amp_factor.
    """
    mass = data["mass"]
    n_events = mass.shape[0]
    n_wave = config["matrix_angle"].shape[1]

    # BW propagator denominators (recomputed every iteration)
    g0_all = np.take(g0, config["g0_index"])
    g0_m = np.take(mass, config["g0_mass_index"], axis=-1)
    g_interp = kernel.interp_catmull_rom(
        config["gamma_table"], config["g0_index"], g0_m,
        config["gamma_min"], config["gamma_delta"])
    g_bw = (g0_all * g_interp) @ config["matrix_gamma"]

    m0_all = np.take(m0, config["m0_index"])
    m0_m = np.take(mass, config["mass_index"], axis=-1)
    bw_dom = m0_all ** 2 - m0_m ** 2 - 1j * m0_all * g_bw
    bd = np.take(bw_dom, config["bw_order"], axis=-1)
    n_res = config["bw_order"].size // n_wave
    bw_p = np.prod(bd.reshape(n_events, n_wave, n_res), axis=-1)

    if amp_cache is None:
        from ampfit.numpy_kernel import NumpyKernel
        _, slot, rep = build_amp_cache_layout(config)
        amp_cache = fill_amp_cache(data, config, NumpyKernel(config), rep)
    amp_w = amp_cache[:, slot_of_wave]                   # (n_events, n_wave)
    return amp_w / bw_p

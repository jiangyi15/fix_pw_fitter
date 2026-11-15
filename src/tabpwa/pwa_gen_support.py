"""Backend support for the ``pwa_gen`` (basis) amplitude model.

``pwa_gen`` extends the projection-sum PWA by two *optional* pieces of
kernel-config data.  Both are **additive**: a config without them behaves
exactly like ``pwa`` (the model only emits them when a resonance actually
carries a free numerator basis / an extended denominator basis).

1. **Numerator table** ``n_table`` / ``n_index``
   One extra table row per free numerator basis function.  Every *(wave,
   factor)* gets a numerator row index ``n_index[f]`` (int32, ``0`` = the
   constant row; ``f = w·n_res + r``, the ``bw_order`` layout), so the
   per-event amplitude factor of a wave is::

       amp_w(e) = fa_w(e) · fl_w(e) · Π_r N_{n_index[w·n_res+r]}(m_{c_r}(e))

   where ``c_r = mass_index[bw_order[w·n_res+r]]`` is that factor's **own**
   mass column — the same one its denominator propagator reads, so a chain
   carrying a numerator on two BWs is evaluated at two different masses
   (``B → R1 x, R1 → R2 y`` gives ``N_R1(m_R1)·N_R2(m_R2)``).

   ``N_0 ≡ 1``: row 0 is built as exactly ``1`` and the factor is applied
   **only when some ``n_index != 0``**, so an all-zero index array is
   bit-identical to the plain PWA path.

2. **Generic denominator coefficients** ``den_row`` / ``den_table``
   The kernel denominator is the opaque basis sum
   ``BW = Σ_j coeff_j · F_j``.  The plain preset hard-codes the slot layout
   ``F = [1, −m², −i·γ_1, …]`` (coefficients derived from ``m0``/``g0``);
   ``den_row`` replaces that with an explicit per-BW row mapping and an
   explicit ``coeff`` parameter (``params["coeff"]``, returned gradients in
   ``grads["coeff"]``), which is what a ``BasisModel`` denominator
   (``[1, m², m⁴, −i·ρ, +i·s·ρ]`` …) needs.

   ``den_row`` is ``int32 (n_unique_bw, n_basis)`` — one row per **unique
   base BW**, indexed per (wave, factor) through ``bw_order`` — and the entry
   encodes the basis function of that slot:

   ======  =========================================================
   ``-1``  ``F = 0``            (padding — coefficient 0, gradient 0)
   ``0``   ``F = 1``
   ``1``   ``F = −m²``         (the s0c sign convention: coefficient
                                is the *negated* m² weight)
   ``2+r`` ``F = den_table[r]`` (a tabulated finished basis function)
   ======  =========================================================

   The tables live on the ``gamma_table`` mass grid
   (``den_min`` / ``den_delta`` are stored explicitly for clarity), so a
   same-named grid is used for every interpolated basis function.
"""

import numpy as np

#: den_row slot kinds (see the module docstring).  ``ZERO_KIND`` is the
#: *negative* padding sentinel: every non-negative encoding is a real
#: basis selector (``2+r`` = ``TABLE_KIND`` row ``r``), so a distinct,
#: non-colliding value is required (``3`` used to mean ``den_table[1]``).
CONST_KIND = 0
MSQ_KIND = 1
TABLE_KIND = 2
ZERO_KIND = -1


# ── numerator ────────────────────────────────────────────────────────

def has_numerator(config):
    """True when some wave carries a non-constant numerator row."""
    idx = config.get("n_index")
    return idx is not None and bool(np.any(np.asarray(idx) != 0))


def numerator_values(config, kernel, mass):
    """Per-factor numerators ``N_{n_index[f]}(m_{n_mass_index[f]})``.

    ``f = w·n_res + r`` (the ``bw_order`` layout), so a chain carrying a
    numerator on more than one BW factor is evaluated at each factor's **own**
    mass.  Returns ``(n_events, n_fact)`` complex.
    """
    idx = np.asarray(config["n_index"])
    # each factor reads its OWN BW's mass column -- the same
    # ``mass_index[bw_order]`` the denominator propagator uses
    cols = np.asarray(config["mass_index"])[np.asarray(config["bw_order"])]
    m = np.take(mass, cols, axis=-1)
    return kernel.interp_catmull_rom(
        config["n_table"], idx, m, config["n_min"], config["n_delta"])


def numerator_product(config, kernel, mass):
    """``Π_r N_{n_index[w·n_res+r]}`` — ``(n_events, n_wave)`` complex.

    The product runs over a wave's factors, each at its own mass (row 0 is
    the constant 1, so an all-constant config gives ``1``).
    """
    vals = numerator_values(config, kernel, mass)
    idx = np.asarray(config["n_index"]).reshape(-1)
    # ``n_wave`` comes from the angular matrix, ``n_res`` is then implied
    n_wave = int(np.asarray(config["matrix_angle"]).shape[1])
    n_res = max(1, idx.size // n_wave)
    # (n_events, n_wave, n_res): multiply across the factor axis
    vals = vals[:, :n_wave * n_res].reshape(vals.shape[0], n_wave, n_res)
    live = np.asarray(idx, bool).reshape(n_wave, n_res)[None, :, :]
    return np.prod(np.where(live, vals, 1.0), axis=-1)


# ── generic denominator ──────────────────────────────────────────────

def has_generic_den(config):
    """True when the config carries an explicit coefficient basis."""
    return "den_row" in config


def build_den_slots(enc, mass_col):
    """Unique-slot table for the generic denominator layout.

    Args:
        enc: ``(n_fact, n_basis)`` int32 slot encodings (see the module
            docstring).
        mass_col: ``(n_fact,)`` event-mass column of each factor.

    Returns:
        ``(kinds, rows, cols, slot_map, n_uslot)`` — ``slot_map`` is the
        flat ``int32 (n_fact * n_basis,)`` index into the unique slots;
        ``kinds``/``rows``/``cols`` describe each unique slot for
        :func:`fill_den_F`.
    """
    enc = np.asarray(enc, dtype=np.int64)
    mass_col = np.asarray(mass_col, dtype=np.int64).reshape(-1)
    n_fact, n_basis = enc.shape
    uniq = {}
    kinds, rows, cols = [], [], []

    def _u(kind, row, col):
        key = (int(kind), int(row), int(col))
        u = uniq.get(key)
        if u is None:
            u = len(kinds)
            uniq[key] = u
            kinds.append(int(kind))
            rows.append(int(row))
            cols.append(int(col))
        return u

    slot_map = np.zeros((n_fact, n_basis), dtype=np.int32)
    for f in range(n_fact):
        col = int(mass_col[f])
        for j in range(n_basis):
            e = int(enc[f, j])
            if e < 0:
                slot_map[f, j] = _u(ZERO_KIND, 0, 0)
            elif e == 0:
                slot_map[f, j] = _u(CONST_KIND, 0, 0)
            elif e == 1:
                slot_map[f, j] = _u(MSQ_KIND, 0, col)
            else:
                slot_map[f, j] = _u(TABLE_KIND, e - 2, col)
    return kinds, rows, cols, slot_map.reshape(-1), len(kinds)


def fill_den_F(config, kernel, mass, kinds, rows, cols):
    """Finished basis values ``F[u](e)`` per unique denominator slot.

    Returns ``(F_r, F_i)`` float64 ``(n_events, n_uslot)`` arrays (the
    same ready-to-use convention the CPU kernels store).
    """
    ne = mass.shape[0]
    nu = len(kinds)
    F_r = np.zeros((ne, nu), dtype=np.float64)
    F_i = np.zeros((ne, nu), dtype=np.float64)
    for u in range(nu):
        kind = kinds[u]
        if kind == CONST_KIND:
            F_r[:, u] = 1.0
        elif kind == MSQ_KIND:
            F_r[:, u] = -mass[:, int(cols[u])] ** 2
        elif kind == TABLE_KIND:
            m = np.take(mass, int(cols[u]), axis=-1)
            g = kernel.interp_catmull_rom(
                config["den_table"], int(rows[u]), m,
                config["den_min"], config["den_delta"])
            F_r[:, u] = np.real(g)
            F_i[:, u] = np.imag(g)
        # ZERO_KIND: leave at 0
    return F_r, F_i


def den_domain(mass, coeff, slots, F):
    """``bw_b(e) = Σ_j coeff[b, j] · F_{slots[b, j]}(e)`` per slot block.

    A *block* is one row of ``slots``; the two layouts in use are one row per
    **unique base BW** (the CUDA ``gen_bw``/``gen_s0e`` addressing) and one
    row per **(wave, factor)** (the flat/CPU addressing, ``b = f``).  This
    helper is layout-agnostic: pass matching ``coeff``/``slots``.

    Args:
        mass: ``(n_events, n_mass)`` mass columns.
        coeff: ``(n_blk, n_basis)`` real coefficients.
        slots: ``(n_blk, n_basis)`` **unique-slot indices** (the reshaped
            ``slot_map`` from :func:`build_den_slots`), not the raw
            ``den_row`` encoding.  Padding (``ZERO_KIND``) is a real slot
            whose basis value is 0, so it never needs a special case here.
        F: ``(n_events, n_uslot)`` complex basis values (from
            :func:`fill_den_F`, combined here).

    Returns:
        complex ``(n_events, n_blk)``.
    """
    coeff = np.asarray(coeff, dtype=np.float64)
    slots = np.asarray(slots, dtype=np.int64)
    n_bw = slots.shape[0]
    out = np.zeros((mass.shape[0], n_bw), dtype=complex)
    # one reused gather buffer: the per-slot (n_events, n_blk) block is the
    # denominator's own size, so avoid re-allocating it for every slot
    buf = np.empty((mass.shape[0], n_bw), dtype=complex)
    for j in range(slots.shape[1]):
        np.take(F, slots[:, j], axis=1, out=buf)
        out += coeff[:, j] * buf
    return out


def den_domain_grad(dQ_dbw, slots, F):
    """``dQ/dcoeff[b, j] = 2·Re Σ_e dQ/dbw_b(e) · F_{slots[b,j]}(e)``.

    ``slots`` is the reshaped unique-slot map and ``b`` the same slot block
    :func:`den_domain` used (a unique BW, or a (wave, factor)); the caller
    passes ``dQ/dbw`` in that same layout.
    """
    dQ_dbw = np.asarray(dQ_dbw)
    slots = np.asarray(slots, dtype=np.int64)
    out = np.zeros((slots.shape[0], slots.shape[1]), dtype=np.float64)
    buf = np.empty((dQ_dbw.shape[0], slots.shape[0]), dtype=complex)
    for j in range(slots.shape[1]):
        np.take(F, slots[:, j], axis=1, out=buf)
        out[:, j] = 2.0 * np.real(np.sum(dQ_dbw * buf, axis=0))
    return out


# ── backend capability check ─────────────────────────────────────────

def check_backend_support(config, backend_name):
    """Refuse a pwa_gen config a backend cannot evaluate yet.

    This is the guard on the CUDA backends' single construction point: a
    CUDA kernel that reads ``m0``/``g0`` directly (all of them except the
    ``gen_*`` family, which are case-1 copies carrying their own combined
    basis table and the numerator in their amp-cache fill) cannot evaluate
    either extension.  The CPU/NumPy pwa_gen backends never reach this
    guard — they are separate registrations, not shared kernels.  An
    ``n ≡ 1``, default-basis ``pwa_gen`` config is exactly the plain PWA
    config and therefore works everywhere.
    """
    if has_numerator(config) or has_generic_den(config):
        # ``_CUDABackend`` passes its CLASS name here (the backend is built
        # through the class); accept both spellings of the gen kernels.
        allow = {"CUDABackendV5PWAGenS0E", "cuda_v5_pwa_gen_s0e",
                 "CUDABackendV5PWAGenFlat", "cuda_v5_pwa_gen_flat",
                 "CUDABackendV5PWAGenBW", "cuda_v5_pwa_gen_bw",
                 "CUDABackendV5PWAGenBWT", "cuda_v5_pwa_gen_bwt"}
        if backend_name in allow:
            return
        raise ValueError(
            f"pwa_gen numerator not supported by {backend_name} yet; "
            f"use a CPU backend, cuda_v5_pwa_gen_s0e, "
            f"cuda_v5_pwa_gen_flat, cuda_v5_pwa_gen_bw, "
            f"cuda_v5_pwa_gen_bwt, or n≡1 (and the default denominator "
            f"basis, i.e. no extended basis rows)")

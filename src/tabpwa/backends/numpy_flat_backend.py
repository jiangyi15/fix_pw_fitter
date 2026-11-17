"""``numpy_flat`` — a minimal reference backend for the flat kernel config.

This is the "simple example" backend for :mod:`tabpwa.pwa_gen_config`: it
consumes the config **exactly as the builder emits it** (the full, non-deduped
form) and evaluates it with plain loops over the tables.  No index arrays are
pre-folded, no cache is built, nothing is shared between (wave, factor)
entries — every table row is read at the index the config names.

The point is to be obviously correct rather than fast:

    for each wave k:
        fl_k  = Π_d  fl_table[k, d][ q_index[k, d] ]
        num_k = Π_r  num_table[k, r][ m_index[k, r] ]
        D_k   = Π_l  Σ_m gamma_matrix[k, l, m] · basis_table[m][ m_index[m] ]
        common_k = fa_k · fl_k · num_k / D_k
    P_p = |Σ_k ck_k · common_k · matrix_angle[·, p·n_wave + k]|²

Any dedup a real backend wants (shared barrier rows, a shared basis block, an
amplitude cache) happens **inside** the backend, on its own copy.  This one
deliberately does none: it is the reference the deduped config must agree
with, and the shape contract every other backend is measured against.

Because it reads the config directly, it also *checks* the config: every
shape it relies on is asserted once in :meth:`__init__`, so a builder that
emits a half-filled array fails here immediately instead of producing
plausible wrong numbers.
"""
import numpy as np

from .core import ComputeBackend, register_backend


@register_backend("numpy_flat", model="pwa_gen")
@register_backend("default", model="pwa_gen")
class NumpyFlatBackend(ComputeBackend):
    """Flat-loop reference over the emitted ``pwa_gen`` config.

    ``params`` is ``{"ck", "coeff"}``:

    * ``ck`` — complex ``(n_wave,)``, the shared coupling per wave;
    * ``coeff`` — real ``(n_basis,)``, one weight per basis row.  This config
      is the **not-deduped** one: the builder appends each factor's basis rows
      where that factor needs them, so every column belongs to exactly one
      ``(wave, res)`` and a flat vector is the whole parameter block.
      ``gamma_matrix`` selects which columns a factor reads; it never
      re-indexes the weights.

    Returns ``(Q, grads, P)`` with ``grads = {"ck", "coeff"}``, both in the
    shapes accepted above.

    ``resolution_size`` selects the objective:

    * ``1`` (default) — the per-event NLL ``-Σ_e w_e·log(P_e/norm + bkg_e)``;
    * ``> 1`` — the grouped log-sum, one log per group of consecutive events
      with the group's TOTAL weight, ``Q = -Σ_g W_g·log(S_g/W_g)`` where
      ``S_g = Σ_{e∈g} w_e·(P_e/norm + bkg_e)`` and ``W_g = Σ_{e∈g} w_e``.
      A zero-weight group is inert; the last group is short when ``ne`` is
      not a multiple of ``resolution_size``.

    ``P`` is the per-event intensity in BOTH modes — grouping changes the
    objective, never the per-event quantity.
    """

    def __init__(self, kernel_config, batch_size=20000, resolution_size=1):
        self.config = kernel_config
        self._check_shapes()
        self._last_dnorm = None
        #: events per grouped-log-sum group; 1 = the per-event NLL
        self.resolution_size = max(1, int(resolution_size))

    # ── the shape contract ──────────────────────────────────────────
    def _check_shapes(self):
        """Assert the arrays agree, deriving every count from a shape.

        Two table layouts are accepted and must be self-consistent:

        * the **full** form from :func:`build_from_groups` — ``fl_table`` is
          ``(n_wave, n_decay, n_interp)`` and ``num_table`` is
          ``(n_wave, n_res, n_interp)``;
        * the **deduped** form from :func:`dedup` — ``fl_table`` is
          ``(n_fl_rows, n_interp)`` and ``num_table`` is
          ``(n_num_rows, n_interp)``, with ``fl_index`` / ``num_index``
          mapping each use onto a row.
        """
        c = self.config
        fl = np.asarray(c["fl_table"])
        if fl.ndim == 3:
            n_wave, n_decay, n_interp = fl.shape
        elif fl.ndim == 2:
            n_interp = fl.shape[1]
            n_wave, n_decay = np.asarray(c["fl_q_index"]).shape
        else:
            raise ValueError(
                "numpy_flat: fl_table must be (n_wave, n_decay, n_interp) "
                f"(full) or (n_rows, n_interp) (deduped), got {fl.shape}")
        num = np.asarray(c["num_table"])
        if num.ndim == 3:
            n_res = num.shape[1]
        elif num.ndim == 2:
            n_res = np.asarray(c["num_m_index"]).shape[1]
        else:
            raise ValueError(
                f"numpy_flat: num_table must be 3-D (full) or 2-D (deduped), "
                f"got {num.shape}")
        n_basis = np.asarray(c["basis_table"]).shape[0]
        # ``n_proj`` is encoded in matrix_angle's column count (p-major),
        # not in a separate axis: columns are ``p*n_wave + k``.
        _ma_cols = int(np.asarray(c["matrix_angle"]).shape[1])
        n_proj = max(1, _ma_cols // n_wave)
        n_ab = np.asarray(c["angle_k"]).shape[0]
        checks = [
            ("fl_q_index", c["fl_q_index"], (n_wave, n_decay)),
            ("num_table", c["num_table"], (n_wave, n_res, n_interp)
             if num.ndim == 3 else (num.shape[0], n_interp)),
            ("num_m_index", c["num_m_index"], (n_wave, n_res)),
            ("basis_table", c["basis_table"], (n_basis, n_interp)),
            ("basis_m_index", c["basis_m_index"], (n_basis,)),
            ("gamma_matrix", c["gamma_matrix"], (n_wave, n_res, n_basis)),
            ("angle_k", c["angle_k"], (n_ab, np.asarray(c["angle_k"]).shape[1])),
            ("angle_b", c["angle_b"], np.asarray(c["angle_k"]).shape),
            ("angle_index", c["angle_index"], (n_ab,)),
            ("matrix_angle", c["matrix_angle"], (n_ab, n_proj * n_wave)),
        ]
        for name in ("fl_index", "num_index"):
            if name in c:
                arr = np.asarray(c[name])
                rows = np.asarray(c["fl_table" if name == "fl_index"
                                      else "num_table"]).shape[0]
                if arr.size and (arr.min() < 0 or arr.max() >= rows):
                    raise ValueError(
                        f"numpy_flat: {name} indexes row {arr.max()} of a "
                        f"{rows}-row table")
        for name, arr, want in checks:
            got = np.asarray(arr).shape
            if got != tuple(want):
                raise ValueError(
                    f"numpy_flat: {name} has shape {got}, expected {tuple(want)} "
                    f"from the other arrays (n_wave={n_wave}, n_decay={n_decay}, "
                    f"n_res={n_res}, n_basis={n_basis}, n_proj={n_proj}, "
                    f"n_anglebasis={n_ab})")
        if len(c.get("den_names", [])) != n_basis:
            raise ValueError(
                f"numpy_flat: den_names has {len(c.get('den_names', []))} "
                f"entries, expected n_basis = {n_basis}")

    # ── data ────────────────────────────────────────────────────────
    def load_data(self, data):
        return {"data": data}

    def free(self):
        pass

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    # ── helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _lin(tab, x, lo, d):
        """Linear interpolation of a 1-D table at ``x`` (same shape out)."""
        pos = (np.asarray(x, np.float64) - lo) / d
        i0 = np.clip(pos.astype(np.int64), 0, tab.size - 2)
        fr = pos - i0
        return tab[i0] * (1.0 - fr) + tab[i0 + 1] * fr

    def _coeff(self, params):
        """The coefficient weights, ``(n_basis,)``, one per basis row.

        Every basis column belongs to exactly one ``(wave, res)`` factor — the
        builder appends a factor's rows where that factor needs them and never
        shares a column — so a *flat* vector indexed by the basis list is the
        whole parameter block.  ``gamma_matrix`` selects which columns a factor
        reads; it never re-indexes the weights.
        """
        coeff = np.asarray(params["coeff"], np.float64)
        n_basis = np.asarray(self.config["basis_table"]).shape[0]
        if coeff.shape != (n_basis,):
            raise ValueError(
                f"numpy_flat: coeff has shape {coeff.shape}, expected "
                f"(n_basis,) = ({n_basis},) — this config is the not-deduped "
                f"one, so every basis entry carries its own weight")
        return np.broadcast_to(coeff[None, None, :],
                               (np.asarray(self.config["fl_table"]).shape[0],
                                np.asarray(self.config["num_table"]).shape[1],
                                n_basis))

    # ── compute ─────────────────────────────────────────────────────
    def compute(self, params, data_handle, norm=None, return_p=True):
        c = self.config
        d = data_handle["data"]
        mass = np.asarray(d["mass"], np.float64)
        mom = np.asarray(d["q"], np.float64)
        ang = np.asarray(d["angle"], np.float64)
        ne = mass.shape[0]

        fl_tab = np.asarray(c["fl_table"])
        fl_q = np.asarray(c["fl_q_index"])
        # The deduped layout maps each (wave, decay) onto a shared row and its
        # table is already flat ``(n_rows, n_interp)``.  The full layout has
        # one row per use inside a 3-D table, so flatten it and use the
        # identity index — the loop below then needs no branch.
        if fl_tab.ndim == 3:
            fl_tab = fl_tab.reshape(-1, fl_tab.shape[-1])
        fl_idx = np.asarray(c["fl_index"]) if "fl_index" in c \
            else np.arange(np.asarray(fl_q).size).reshape(np.asarray(fl_q).shape)
        num_tab = np.asarray(c["num_table"])
        num_m = np.asarray(c["num_m_index"])
        if num_tab.ndim == 3:
            num_tab = num_tab.reshape(-1, num_tab.shape[-1])
        num_idx = np.asarray(c["num_index"]) if "num_index" in c \
            else np.arange(np.asarray(num_m).size).reshape(np.asarray(num_m).shape)
        bas = np.asarray(c["basis_table"])
        bmi = np.asarray(c["basis_m_index"])
        gam = np.asarray(c["gamma_matrix"])
        ak = np.asarray(c["angle_k"])
        ab = np.asarray(c["angle_b"])
        ai = np.asarray(c["angle_index"])
        am = np.asarray(c["matrix_angle"])          # (n_ab, n_proj*n_wave)

        # counts from the SHAPES of the index arrays (both layouts), never
        # from the table's row count (dedup changes that, the use-count not)
        n_wave, n_decay = np.asarray(fl_q).shape
        n_res = np.asarray(num_m).shape[1]
        n_basis = bas.shape[0]
        # p-major columns: p*n_wave + k
        n_proj = max(1, am.shape[1] // n_wave)
        m0, md = float(c["m_min"]), float(c["m_delta"])
        q0, qd = float(c["q_min"]), float(c["q_delta"])

        ck = np.asarray(params["ck"], complex)
        if ck.size != n_wave:
            raise ValueError(f"numpy_flat: ck has {ck.size} entries, expected "
                             f"n_wave = {n_wave}")
        weights = self._coeff(params)

        # ── fa: (ne, n_wave, n_proj), via the historical matrix_angle ──
        # ka @ matrix_angle is the long-standing idiom; the result is p-major
        # (`p*n_wave + k`), so it is reshaped to (ne, n_proj, n_wave) and
        # moved to the (ne, n_wave, n_proj) order the rest of this kernel uses.
        ka = np.zeros((ne, ak.shape[0]), np.complex128)
        for a in range(ak.shape[0]):
            ka[:, a] = np.prod(np.cos(ang[:, ai[a], :] * ak[a] + ab[a]),
                               axis=-1)
        fa = (ka @ am).reshape(ne, n_proj, n_wave).transpose(0, 2, 1)

        # ── per-wave barrier × numerator / denominator ──────────────
        # The per-factor denominator products are kept: the backward pass
        # needs both ``s_l`` and their product ``D`` (``common = F/D``), and
        # recomputing them there would evaluate every basis row twice.
        common = np.zeros((ne, n_wave), np.complex128)
        factors = np.zeros((ne, n_wave, max(1, n_res)), np.complex128)
        for k in range(n_wave):
            fl = np.ones(ne)
            for dec in range(n_decay):
                q = mom[:, fl_q[k, dec]]
                row = fl_idx[k, dec]
                fl *= self._lin(fl_tab[row], q, q0, qd)
            num = np.ones(ne, np.complex128)
            for r in range(n_res):
                xx = mass[:, num_m[k, r]]
                tab = num_tab[num_idx[k, r]]
                num *= (self._lin(tab.real, xx, m0, md)
                        + 1j * self._lin(tab.imag, xx, m0, md))
            den = np.ones(ne, np.complex128)
            for r in range(n_res):
                s = self._den_factor(k, r, mass, weights)
                factors[:, k, r] = s
                den *= s
            common[:, k] = fl * num / den

        # ── projection sum ──────────────────────────────────────────
        A = np.zeros((ne, n_proj), np.complex128)
        for p in range(n_proj):
            for k in range(n_wave):
                A[:, p] += ck[k] * common[:, k] * fa[:, k, p]
        P_e = np.sum(A.real ** 2 + A.imag ** 2, axis=-1)

        weight = np.asarray(d.get("weight", np.ones(ne)), np.float64)
        bkg = d.get("bkg", 1.0)
        bv = bkg if not np.isscalar(bkg) else np.full(ne, float(bkg))
        if norm is None:
            Q = float(np.sum(weight * P_e))
            dQ_dP = weight
        elif self.resolution_size == 1:
            Q = float(-np.sum(weight * np.log(P_e / norm + bv)))
            dQ_dP = -weight / (P_e + bv * norm)
        else:
            # ── grouped log-sum with the group TOTAL weight ──────────
            #   Q = -Σ_g W_g·log(S_g/W_g),  S_g = Σ_{e∈g} w_e·(P_e/norm + bkg_e)
            #   dQ/dP_e = -(W_g/S_g)·(w_e/norm)
            # Groups are consecutive events aligned to the GLOBAL index; the
            # last group is short when ne is not a multiple of rs, so it is
            # zero-PADDED to a rectangle and the pad rows carry w = 0 (they
            # add nothing to S_g, Pb_g or W_g).
            rs = self.resolution_size
            ng = (ne + rs - 1) // rs
            pad = ng * rs - ne
            w_g = np.concatenate([weight, np.zeros(pad)]).reshape(ng, rs)
            p_g = np.concatenate([P_e, np.zeros(pad)]).reshape(ng, rs)
            b_g = np.concatenate([bv, np.zeros(pad)]).reshape(ng, rs)
            W_g = w_g.sum(axis=1)
            S_g = (w_g * (p_g / norm + b_g)).sum(axis=1)
            safe = W_g > 0.0
            # a zero-weight group is inert: log() of it is skipped, not -inf
            Q = float(-np.sum(np.where(
                safe,
                W_g * np.log(np.where(safe, S_g, 1.0)
                             / np.where(safe, W_g, 1.0)),
                0.0)))
            scale = np.where(safe, -W_g / np.where(safe, S_g, 1.0), 0.0)
            dQ_dP = (scale[:, None] * (w_g / norm)).reshape(-1)[:ne]

        # ── backward ────────────────────────────────────────────────
        # dQ/dA_p = dQ/dP · conj(A_p);  every gradient below is the same
        # algebra the kernel uses, written out per table entry.
        dQ_dA = dQ_dP[:, None] * np.conj(A)              # (ne, n_proj)

        grad_ck = np.zeros(n_wave, np.complex128)
        grad_coeff = np.zeros((n_wave, n_res, n_basis), np.float64)
        for k in range(n_wave):
            # d/dcommon_k, summed over projections
            d_common = np.zeros(ne, np.complex128)
            for p in range(n_proj):
                d_common += dQ_dA[:, p] * ck[k] * fa[:, k, p]
            # The kernel convention (matching ``numpy_pwa``): the raw
            # Wirtinger half ∂Q/∂ck = Σ_p dQ/dA_p·common_k·fa_pk, with NO
            # factor 2 and NO conjugation.  ``CKProduct.backprop_grad``
            # applies the 2·Re real-parameter convention downstream — adding
            # either here double-counts.
            grad_ck[k] = np.sum(
                [np.sum(dQ_dA[:, p] * common[:, k] * fa[:, k, p])
                 for p in range(n_proj)])
            # common = F / D  =>  dQ/dD = -dQ/dcommon · common / D
            den = np.prod(factors[:, k, :n_res], axis=1)
            dD = -d_common * common[:, k] / den
            # D = Π_l s_l  =>  dQ/ds_l = dQ/dD · Π_{l'≠l} s_l'  (no division)
            for r in range(n_res):
                others = np.ones(ne, np.complex128)
                for r2 in range(n_res):
                    if r2 == r:
                        continue
                    others *= factors[:, k, r2]
                ds = dD * others                          # (ne,)
                # s_l = Σ_m gamma·coeff·basis_m  =>  dQ/dcoeff = 2 Re Σ_e ds·basis
                for m in range(n_basis):
                    w = gam[k, r, m]
                    if w == 0.0:
                        continue
                    xx = mass[:, bmi[m]]
                    tab = bas[m]
                    b = (self._lin(tab.real, xx, m0, md)
                         + 1j * self._lin(tab.imag, xx, m0, md))
                    grad_coeff[k, r, m] += 2.0 * np.real(np.sum(ds * w * b))

        grads = {"ck": grad_ck, "coeff": grad_coeff}
        if norm is not None:
            # the fitter's contract: a normed compute must return the native
            # d(NLL)/d(norm) alongside the parameter blocks.
            #   per-event : Σ w·P / (norm·(P + b·norm))
            #   grouped   : Σ_g W_g·Pb_g / (S_g·norm²),  Pb_g = Σ w·P
            if self.resolution_size == 1:
                self._last_dnorm = float(np.sum(
                    weight * P_e / (norm * (P_e + bv * norm))))
            else:
                rs = self.resolution_size
                ng = (ne + rs - 1) // rs
                pad = ng * rs - ne
                w_g = np.concatenate([weight, np.zeros(pad)]).reshape(ng, rs)
                p_g = np.concatenate([P_e, np.zeros(pad)]).reshape(ng, rs)
                b_g = np.concatenate([bv, np.zeros(pad)]).reshape(ng, rs)
                W_g = w_g.sum(axis=1)
                S_g = (w_g * (p_g / norm + b_g)).sum(axis=1)
                Pb_g = (w_g * p_g).sum(axis=1)
                safe = W_g > 0.0
                self._last_dnorm = float(np.sum(np.where(
                    safe, W_g * Pb_g / (np.where(safe, S_g, 1.0) * norm * norm),
                    0.0)))
            grads["norm"] = self._last_dnorm
        else:
            self._last_dnorm = None

        # ``grad_coeff`` is accumulated per (wave, res) because that is how the
        # loop reads the basis rows; with private columns exactly one entry is
        # non-zero per column, so the sum back to the flat (n_basis,) block the
        # caller supplied is exact — not an approximation.
        grads["coeff"] = grad_coeff.sum(axis=(0, 1))

        return Q, grads, (P_e if return_p else None)

    def _den_factor(self, k, r, mass, weights):
        """``s_{k,r} = Σ_m gamma[k,r,m]·coeff[k,r,m]·basis_m(m)``.

        One factor of the denominator: the basis rows this (wave, factor)
        selects, weighted by the coefficient at that slot.  The selection is
        the config's flat ``gamma_matrix`` — no index is folded or cached.
        """
        c = self.config
        bas = np.asarray(c["basis_table"])
        bmi = np.asarray(c["basis_m_index"])
        gam = np.asarray(c["gamma_matrix"])
        m0, md = float(c["m_min"]), float(c["m_delta"])
        ne = mass.shape[0]
        s = np.zeros(ne, np.complex128)
        for m in range(bas.shape[0]):
            w = gam[k, r, m]
            if w == 0.0:
                continue
            xx = mass[:, bmi[m]]
            tab = bas[m]
            s += w * weights[k, r, m] * (
                self._lin(tab.real, xx, m0, md)
                + 1j * self._lin(tab.imag, xx, m0, md))
        return s

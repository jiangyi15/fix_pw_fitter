"""``numpy_pwa_gen`` — NumPy reference for the ``pwa_gen`` (basis) amplitude model.

**Case-1 copy** of :mod:`tabpwa.numpy_pwa`'s projection-sum engine, specialised
to the basis model.  ``numpy_pwa`` itself is a working kernel for the plain
``pwa`` model and is left untouched (kernel policy, case 1: a new variant gets
its own file and its own symbols).

What differs from ``NumpyPWA``:

* the denominator may be an **explicit coefficient basis**
  ``bw_f = Σ_i coeff_{f,i} · F_i`` (``f = w·n_res + r``, one block per
  (wave, factor)) read from ``den_row`` / ``den_table`` / ``gamma_table``,
  instead of the closed-form ``m0² − m² − i·m0·Σ g0·γ(m)``.  A config
  without ``den_row`` keeps the plain ``m0``/``g0`` propagator, so this
  kernel covers the whole model (default-basis configs included) and is the
  NumPy oracle for every ``pwa_gen`` config;
* the amplitude carries the tabulated **numerator** ``N_{n_index[f]}(m)``
  per (wave, factor), folded into ``fa·fl``;
* on the basis path ``params`` is ``{"ck", "coeff"}`` and ``grads`` is
  ``{"ck", "coeff"}`` — there is no ``m0``/``g0`` surface (the basis weights
  already are the parameters; see
  :class:`tabpwa.kernel_params.PWAGenKernelParams`).

``dQ/dcoeff`` is reported per (wave, factor), the layout ``cpu_pwa_gen``
uses and the layout ``PWAGenKernelParams.backward`` sums over its copies.

Everything else — the projection sum ``P_e = Σ_p |Σ_k ck_k·common_{p·N+k}|²``,
the ``ck`` gradient, the NLL and its ``dNLL/dnorm`` — is the same algebra as
``numpy_pwa``, and the two agree to fp64 round-off on a config where the basis
is the default one (the generic path sums ``Σ coeff·F`` where the plain path
evaluates ``m0² − m² − i·m0·Σ g0·γ``: same terms, different order).
"""

import numpy as np

from .numpy_kernel import NumpyKernel


class NumpyPWAGen:
    """NumPy projection-sum PWA on the ``pwa_gen`` basis (reference).

    ``compute(params, data_handle, norm=None, return_p=True)`` returns
    ``(Q, grads, P)`` with ``params = {"ck", "coeff"}`` and
    ``grads = {"ck", "coeff"}``.
    """

    def __init__(self, config, cp_block=False):
        from tabpwa.pwa_gen_support import has_generic_den, has_numerator

        self.config = config
        self.cp_block = bool(cp_block)
        self._n_on = has_numerator(config)
        self._generic = has_generic_den(config)

        self.n_wave = config["matrix_angle"].shape[1]
        self.n_proj_base = int(config.get("n_proj", 1) or 1)
        self.n_proj = self.n_proj_base * (2 if self.cp_block else 1)
        if self.n_wave % self.n_proj != 0:
            raise ValueError(
                f"numpy_pwa_gen: n_wave={self.n_wave} not divisible by "
                f"n_proj_eff={self.n_proj} (n_proj={self.n_proj_base}, "
                f"cp_block={self.cp_block})")
        self.n_wave_base = self.n_wave // self.n_proj    # N (shared ck length)

        self._nk = NumpyKernel(config)
        self._last_dnorm = None

        # ── denominator basis layout, per (wave, factor) ────────────────
        # ``coeff`` is stored per (wave, factor) -- the kernel vocabulary
        # ``f = w·n_res + r`` -- so the slot map is expanded the same way:
        # ``den_row`` is per BW, indexed through ``bw_order``.
        if self._generic:
            from tabpwa.pwa_gen_support import build_den_slots
            bow = np.asarray(self._nk.bw_order).reshape(-1)
            enc_f = np.asarray(config["den_row"])[bow]  # [n_fact, n_basis]
            mass_col_f = np.asarray(self._nk.mass_index)[bow]
            kinds, rows, cols, slot_flat, n_uslot = build_den_slots(
                enc_f, mass_col_f)
            self._den_kinds = kinds
            self._den_rows = rows
            self._den_cols = cols
            # ``F`` is indexed by unique-slot id, not by the raw encoding
            self._den_slots = slot_flat.reshape(enc_f.shape)
            self.n_uslot = n_uslot
            self.n_fact = int(enc_f.shape[0])

    # ── data ────────────────────────────────────────────────────────────
    def load_data(self, data):
        return {"data": data}

    def free(self):
        self._nk = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    # ── compute ─────────────────────────────────────────────────────────
    def compute(self, params, data_handle, norm=None, return_p=True):
        from tabpwa.pwa_gen_support import (
            den_domain, den_domain_grad, fill_den_F, numerator_product)

        k = self._nk
        n_wave = self.n_wave
        P = self.n_proj
        N = self.n_wave_base
        kc = self.config

        ck = np.asarray(params["ck"], dtype=complex)
        if len(ck) != N:
            raise ValueError(
                f"numpy_pwa_gen: ck length {len(ck)} != n_wave/n_proj_eff = "
                f"{N} ({n_wave}/{P})")
        m0 = np.asarray(params["m0"]) if "m0" in params else None
        g0 = np.asarray(params["g0"]) if "g0" in params else None

        data = data_handle["data"]
        mass = data["mass"]
        ne = mass.shape[0]

        # ── denominator: explicit basis, or the plain m0/g0 propagator ──
        if self._generic:
            if "coeff" not in params:
                raise ValueError(
                    "numpy_pwa_gen: the basis model needs params['coeff'] "
                    "(the per-(wave, factor) basis weights); there is no "
                    "m0/g0 surface")
            coeff = np.asarray(params["coeff"], dtype=float)
            F_r, F_i = fill_den_F(kc, k, mass, self._den_kinds,
                                  self._den_rows, self._den_cols)
            F = F_r + 1j * F_i
            # one block per (wave, factor): bw_f = Σ_i coeff_f,i·F_i
            bw_f = den_domain(mass, coeff, self._den_slots, F)
        else:
            if "coeff" in params:
                raise ValueError(
                    "numpy_pwa_gen: this config uses the plain m0/g0 "
                    "propagator; pass params['m0']/['g0'], not ['coeff']")
            g0_all = np.take(g0, k.g0_index)
            g0_m = np.take(mass, k.g0_mass_index, axis=-1)
            g_interp = k.interp_catmull_rom(
                k.gamma_table, k.g0_index, g0_m, k.gamma_min, k.gamma_delta)
            g_bw = (g0_all * g_interp) @ k.matrix_gamma
            m0_all = np.take(m0, k.m0_index)
            m0_m = np.take(mass, k.mass_index, axis=-1)
            bw_dom = m0_all ** 2 - m0_m ** 2 - 1j * m0_all * g_bw
            # expand to the per-(wave, factor) tile the gradient path uses
            bw_f = np.take(bw_dom, np.asarray(k.bw_order).reshape(-1), axis=-1)

        bw_dom_r = bw_f.reshape(ne, n_wave, k.n_res)
        bw_p = np.prod(bw_dom_r, axis=-1)

        # ── numerator × barrier × angular ───────────────────────────────
        fl_q = np.take(data["q"], k.fl_q_index, axis=-1)
        fl = k.interp_catmull_rom(k.fl_table, k.fl_type, fl_q,
                                  k.fl_min, k.fl_delta)
        fl_p = np.prod(np.take(fl, k.fl_order, axis=-1).reshape(
            ne, n_wave, k.n_decay), axis=-1)

        ang = np.take(data["angle"], k.angle_index, axis=-2)
        ka = np.prod(np.cos(ang * k.angle_k + k.angle_b), axis=-1)
        fa = ka @ k.matrix_angle

        amp = fa * fl_p
        if self._n_on:
            amp = amp * numerator_product(kc, k, mass)
        common = amp / bw_p

        cm = common.reshape(ne, P, N)
        A = (cm.reshape(ne * P, N) @ ck).reshape(ne, P)
        P_e = np.sum(A.real ** 2 + A.imag ** 2, axis=-1)

        weight = data.get("weight", np.ones(ne))
        bkg = data.get("bkg", 1.0)
        if norm is None:
            Q = float(np.sum(weight * P_e))
            dQ_dP = weight
        else:
            Q = float(-np.sum(weight * np.log(P_e / norm + bkg)))
            dQ_dP = -weight / (P_e + bkg * norm)

        # ── backward: ck and the denominator parameters ─────────────────
        dQ_dA = dQ_dP[:, None] * np.conj(A)              # (ne, P)
        S_ek = np.einsum("ep,epk->ek", dQ_dA, cm)        # (ne, N)
        grad_ck = S_ek.sum(axis=0)

        # dQ/dbw_{w,r} = dBW_w / bw_{w,r} with dBW_w = −(dQ/dcommon_w)·common_w.
        # ``dQ/dcommon`` is PER PROJECTION (``dQ/dA_p·ck_k``), not the p-sum
        # ``S_ek`` — the reference kernel computes it inside the wave loop, so
        # each projection carries its own dBW.  ``w = p·N + k`` is p-major, so
        # the per-wave dBW is ``dQ_dA[p,k]·ck_k·common[p,k]`` written as the
        # (ne, n_wave) tile below.
        dQ_dcommon = (dQ_dA[:, :, None] * ck[None, None, :])   # (ne, P, N)
        dBW_w = -(dQ_dcommon * cm).reshape(ne, n_wave)          # (ne, n_wave)
        dBW_f = np.repeat(dBW_w, k.n_res, axis=1)               # (ne, n_fact)
        dQ_dbw_f = dBW_f / bw_f

        if self._generic:
            # ∂Q/∂coeff[f, i] = 2·Re Σ_e dQ/dbw_f · F_{slots[f,i]}(e)
            grad_coeff = den_domain_grad(dQ_dbw_f, self._den_slots, F)
            grads = {"ck": grad_ck, "coeff": grad_coeff}
        else:
            # the p-reduced chain of the plain m0/g0 parameters: the
            # propagator is duplicated over projections, so sum there first
            boN = np.asarray(k.bw_order).reshape(n_wave, k.n_res)[:N]
            bw_dom_t = np.take(bw_dom, boN, axis=-1)         # (ne, N, n_res)
            V = -ck[None, :] * S_ek
            dQ_dbw_dom = ((V[:, :, None] / bw_dom_t)
                          .reshape(ne, N * k.n_res)
                          @ k._bw_scatter[:N * k.n_res])     # (ne, n_unique_bw)
            dbw_dom_dm0 = 2 * m0_all - 1j * g_bw
            grad_m0 = np.sum(2 * np.real(dQ_dbw_dom * dbw_dom_dm0),
                             axis=0) @ k._m0_scatter
            dQ_dg_bw = dQ_dbw_dom * (-1j * m0_all)
            dQ_dg = dQ_dg_bw @ k.matrix_gamma.T
            grad_g0 = np.sum(2 * np.real(dQ_dg * g_interp),
                             axis=0) @ k._g0_scatter
            grads = {"ck": grad_ck, "m0": grad_m0, "g0": grad_g0}

        if norm is not None:
            b = bkg if not np.isscalar(bkg) else np.full(ne, float(bkg))
            self._last_dnorm = float(np.sum(
                weight * P_e / (norm * (P_e + b * norm))))
        else:
            self._last_dnorm = None

        return Q, grads, (P_e if return_p else None)

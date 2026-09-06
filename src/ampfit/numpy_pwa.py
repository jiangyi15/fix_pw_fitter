"""
numpy_pwa — pure-PWA reference kernel derived from the original NumPy kernel.

The PWA model is the original ``ampfit.numpy_kernel.NumpyKernel._compute``
**with the time / D0-D0bar-mixing / scalar parts removed** and the two
flavour blocks promoted to an incoherent *sum over projections*:

    P(e) = Σ_p |A_p(e)|² ,   A_p(e) = Σ_k  ck_k · a_{p,k}(e)

exactly the same amplitude chain as the original kernel (BW/gamma running
width, per-decay Blatt-Weisskopf form factors, matrix_angle angular basis)
and the same gradient back-propagation for ``ck`` / ``m0`` / ``g0`` — the
only structural change is the projection sum.

Wave-block / projection layout follows the original kernel convention:
the first half of the wave space (blocks 0-3, ``g_ls``) is projection 0 and
the second half (blocks 4-7, ``g_lsbar``) is projection 1:

    a = ck * common_amp_factor
    a.reshape(n_events, P, n_wave // P)   with P = 2
    A_p  = Σ_k a[e, p, k]
    P(e) = |A_0|² + |A_1|²                (no time, no scalars)

Gradients use the original Wirtinger convention:
``∂Q/∂Re(ck) = 2·Re(grad_ck)``, ``∂Q/∂Im(ck) = −2·Im(grad_ck)`` and for real
``m0``/``g0`` ``dQ/dx = 2·Re(∂Q/∂z·∂z/∂x)``.
"""

import numpy as np


class NumpyPWA:
    """PWA (projection-sum) variant of the original NumPy kernel.

    Drop-in for ``NumpyKernel`` on the forward/backward path, but:

    * no ``scalar`` parameters (Gamma/ΔΓ/Δm/A_p/poq),
    * no event ``time`` / ``frac`` / mixing evolution,
    * ``P(e) = Σ_p |A_p(e)|²`` over the two flavour projections.

    ``compute(params, data, norm=None, return_p=True) -> (Q, grads, P)``
    with ``params = {"ck", "m0", "g0"}`` and ``grads = {"ck", "m0", "g0"}``.
    """

    def __init__(self, config):
        from ampfit.numpy_kernel import NumpyKernel
        self.config = config
        self.n_wave = config["matrix_angle"].shape[1]
        # Number of incoherent projections (flavour blocks, 0-3 vs 4-7).
        self.n_proj = int(config.get("n_proj", 1) or 1)
        if self.n_wave % 2 != 0:
            raise ValueError("numpy_pwa: n_wave must be even (two flavour "
                             "projection halves)")
        if self.n_proj == 1:
            # Original layout is already block-paired; force the 2-flavour
            # projection split used by the amplitude model.
            self.n_proj = 2
        self._kernel = NumpyKernel(config)
        self._nk = self._kernel

    # -- data -------------------------------------------------------------
    def load_data(self, data):
        """No one-time cache on the CPU path — store data as the handle."""
        return {"data": data}

    def free(self):
        pass

    def __del__(self):
        self.free()

    # -- forward ----------------------------------------------------------
    def compute(self, params, data_handle, norm=None, return_p=True):
        k = self._nk
        n_wave = self.n_wave
        P = self.n_proj
        M = n_wave // P

        ck = np.asarray(params["ck"], dtype=complex)
        if len(ck) != n_wave:
            raise ValueError(
                f"numpy_pwa: ck length {len(ck)} != n_wave {n_wave}")
        m0 = np.asarray(params["m0"])
        g0 = np.asarray(params["g0"])

        data = data_handle["data"]
        mass = data["mass"]
        ne = mass.shape[0]

        # ── forward amplitude chain (identical to the original kernel) ────
        g0_all = np.take(g0, k.g0_index)
        g0_m = np.take(mass, k.g0_mass_index, axis=-1)
        g_interp = k.interp_catmull_rom(
            k.gamma_table, k.g0_index, g0_m, k.gamma_min, k.gamma_delta)
        g_bw = (g0_all * g_interp) @ k.matrix_gamma

        m0_all = np.take(m0, k.m0_index)
        m0_m = np.take(mass, k.mass_index, axis=-1)
        bw_dom = m0_all ** 2 - m0_m ** 2 - 1j * m0_all * g_bw
        bw_dom_all = np.take(bw_dom, k.bw_order, axis=-1)
        bw_dom_r = bw_dom_all.reshape(ne, n_wave, k.n_res)
        bw_p = np.prod(bw_dom_r, axis=-1)

        fl_q = np.take(data["q"], k.fl_q_index, axis=-1)
        fl = k.interp_catmull_rom(k.fl_table, k.fl_type, fl_q,
                                  k.fl_min, k.fl_delta)
        fl_all = np.take(fl, k.fl_order, axis=-1)
        fl_p = np.prod(fl_all.reshape(ne, n_wave, k.n_decay), axis=-1)

        ang = np.take(data["angle"], k.angle_index, axis=-2)
        ka = np.prod(np.cos(ang * k.angle_k + k.angle_b), axis=-1)
        fa = ka @ k.matrix_angle

        one_over_bw = 1.0 / bw_p
        common = one_over_bw * (fa * fl_p)                # (ne, n_wave)

        a = ck * common
        ar = a.reshape(ne, P, M)
        A = ar.sum(axis=-1)                               # (ne, P) = A_p
        P_e = np.sum(A.real ** 2 + A.imag ** 2, axis=-1)

        weight = data.get("weight", np.ones(ne))
        bkg = data.get("bkg", 0.0)
        if norm is None:
            Q = float(np.sum(weight * P_e))
            dQ_dP = weight
        else:
            Q = float(-np.sum(weight * np.log(P_e / norm + bkg)))
            dQ_dP = -weight / (P_e + bkg * norm)

        # ── back-prop through the projection sum ──────────────────────────
        # ∂Q/∂A_p = dQ/dP · conj(A_p)   (Wirtinger, cf. the original kernel)
        dQ_dA = dQ_dP[:, None] * np.conj(A)               # (ne, P)
        dQ_da = np.repeat(dQ_dA, M, axis=-1)              # (ne, n_wave)

        grad_ck = np.sum(dQ_da * common, axis=0)

        # BW chain gradients — verbatim formulas of the original kernel.
        dQ_dbw_p = dQ_da * (-ck * one_over_bw * common)
        # d(bw_p)/d(bw_dom_r) = bw_p / bw_dom_r  (product rule, any n_res)
        dQ_dbw_dom_all = dQ_dbw_p[:, :, None] * (
            bw_p[:, :, None] / bw_dom_r)          # (ne, n_wave, n_res)
        dQ_dbw_dom = (dQ_dbw_dom_all.reshape(ne, -1)
                      @ k._bw_scatter)

        dbw_dom_dm0 = 2 * m0_all - 1j * g_bw
        grad_m0 = np.sum(2 * np.real(dQ_dbw_dom * dbw_dom_dm0),
                         axis=0) @ k._m0_scatter

        dQ_dg_bw = dQ_dbw_dom * (-1j * m0_all)
        dQ_dg = dQ_dg_bw @ k.matrix_gamma.T
        grad_g0 = np.sum(2 * np.real(dQ_dg * g_interp),
                         axis=0) @ k._g0_scatter

        grads = {"ck": grad_ck, "m0": grad_m0, "g0": grad_g0}
        return Q, grads, (P_e if return_p else None)

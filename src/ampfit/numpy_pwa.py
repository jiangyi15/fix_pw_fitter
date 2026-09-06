"""
numpy_pwa — pure-PWA NumPy kernel: P(e) = Σ_p |A_p(e)|², shared ck.

Derived from the original ``ampfit.numpy_kernel.NumpyKernel`` amplitude
chain (running width g0·γ(m), per-resonance BW denominators, per-decay
Blatt-Weisskopf form factors, matrix_angle angular basis) with the time /
D0-D0bar-mixing / scalar parts removed, and replaced by a generic sum over
``n_proj`` incoherent projections.

Layout (identical to ``cuda_v4_pwa``):

    n_wave = n_proj · N        entries are p-major, w = p·N + k
    A_p(e) = Σ_k  ck_k · a_{p,k}(e)      all projections SHARE ck (len N)
    P(e)   = Σ_p |A_p(e)|²

``n_proj`` is taken directly from the kernel config; ``cp_block=False`` by
default.  When ``cp_block=True`` the charge-conjugate (flavour-partner)
block is counted as an additional set of projections, doubling the
projection count:

    n_proj_eff = n_proj · (2 if cp_block else 1)

(the angular rows for the CP partner must already be present in the
p-major-duplicated per-wave arrays — as with the flavour halves of the
legacy 8-block layout, blocks 0-3 vs 4-7).

Gradients keep the original kernel's Wirtinger convention:
``∂Q/∂Re(ck) = 2·Re(grad_ck)``, ``∂Q/∂Im(ck) = −2·Im(grad_ck)`` and for
real m0/g0 ``dQ/dx = 2·Re(∂Q/∂z · ∂z/∂x)``.
"""

import numpy as np


class NumpyPWA:
    """PWA (projection-sum, shared-ck) variant of the original NumPy kernel.

    API mirrors ``NumpyKernel``/``cuda_v4_pwa.CUDAKernelV4PWA``:

    * ``load_data(data) -> handle``  (CPU: no cache, data stored as handle)
    * ``compute(params, handle, norm=None, return_p=True)
        -> (Q, grads, P)``
      with ``params = {"ck", "m0", "g0"}`` (ck length N = n_wave/n_proj_eff)
      and ``grads = {"ck", "m0", "g0"}``.
    """

    def __init__(self, config, cp_block=False):
        from ampfit.numpy_kernel import NumpyKernel
        self.config = config
        self.cp_block = bool(cp_block)
        self.n_wave = config["matrix_angle"].shape[1]
        self.n_proj_base = int(config.get("n_proj", 1) or 1)
        self.n_proj = self.n_proj_base * (2 if self.cp_block else 1)
        if self.n_wave % self.n_proj != 0:
            raise ValueError(
                f"numpy_pwa: n_wave={self.n_wave} not divisible by "
                f"n_proj_eff={self.n_proj} (n_proj={self.n_proj_base}, "
                f"cp_block={self.cp_block})")
        self.n_wave_base = self.n_wave // self.n_proj    # N (shared ck length)
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

    # -- forward + gradients ----------------------------------------------
    def compute(self, params, data_handle, norm=None, return_p=True):
        k = self._nk
        n_wave = self.n_wave
        P = self.n_proj
        N = self.n_wave_base

        ck = np.asarray(params["ck"], dtype=complex)
        if len(ck) != N:
            raise ValueError(
                f"numpy_pwa: ck length {len(ck)} != n_wave/n_proj_eff = "
                f"{N} ({n_wave}/{P})")
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
        common = one_over_bw * (fa * fl_p)               # (ne, n_wave) p-major

        cm = common.reshape(ne, P, N)
        # A_p = Σ_k ck_k·cm[e,p,k]: stacked BLAS gemv over (ne·P, N) rows is
        # ~10x faster than materialising ck per entry (numpy: (ne,P,N)@ck
        # loops per event, this does one contiguous matvec).
        A = (cm.reshape(ne * P, N) @ ck).reshape(ne, P)  # (ne, P) = A_p
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
        # ∂Q/∂A_p = dQ/dP · conj(A_p);  entry w=(p,k) takes the A_p gradient
        dQ_dA = dQ_dP[:, None] * np.conj(A)              # (ne, P)
        # S[e,k] = Σ_p dQ_dA[e,p]·a_{p,k}(e) — project P away up front:
        # the BW propagators (and fl_q/mass) are identical across projections
        # (p-major duplication), so only the angular part differs and every
        # m0/g0 gradient contribution is linear in this p-reduced sum.
        S_ek = np.einsum("ep,epk->ek", dQ_dA, cm)       # (ne, N)
        grad_ck = S_ek.sum(axis=0)

        # BW chain gradient, reduced to the N base waves:
        #   dQ/dbw_dom[e,bw] = Σ_{k,r→bw} −ck_k·S[e,k] / bw_dom[e,bw]
        V = -ck[None, :] * S_ek                          # (ne, N)
        boN = k.bw_order.reshape(n_wave, k.n_res)[:N]
        bw_dom_t = np.take(bw_dom, boN, axis=-1)         # (ne, N, n_res)
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
        return Q, grads, (P_e if return_p else None)

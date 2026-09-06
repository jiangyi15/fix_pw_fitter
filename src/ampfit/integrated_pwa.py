"""
integrated_pwa — Gram-matrix (pre-integrated) phase-space norm for the
projection-sum PWA model.

For the shared-ck projection-sum model

    P(e) = Σ_p |A_p(e)|² ,   A_p(e) = Σ_k ck_k · a_{p,k}(e)

the phase-space integral of the unnormalised density over N_phsp weighted
events factorises into a per-wave Gram matrix:

    ∫ dΦ |A|² ≈ Σ_e w_e Σ_p |A_p(e)|²
               = Σ_{k,k'}  ck_k · conj(ck_k') · D_{k,k'}

with the **wave Gram matrix**

    D_{k,k'} = Σ_e w_e Σ_p  conj(a_{p,k}(e)) · a_{p,k'}(e)      (k,k' < N)

i.e. ``D = C^H C``-style over events/projections where
``C[e·P+p, k] = a_{p,k}(e)``.  ``D`` is Hermitian positive semi-definite,
and its size is N×N (independent of the number of phsp events or of the
projection count P).  Evaluating the norm therefore costs O(N²) per
iteration once ``D`` has been built for the current m0/g0 — the same idea
as the legacy ``IntegratedBackend`` (Gram matrices) but for the pure-PWA
shared-ck layout with per-projection angular entries.

    norm(ck)   = Σ_{k,k'} ck_k · conj(ck_k') · D_{k,k'}   = Re(ck^T D conj(ck)?)
    (convention below: norm = Σ_k ck_k * (D @ conj(ck))_k )

Amplitudes ``a_{p,k} = common_{p·N+k}`` use the exact forward chain of
``numpy_pwa`` (running width gamma, per-resonance BW denominators, fl
form factors, matrix_angle angular basis), fp64 — correctness reference.
"""

import numpy as np

from ampfit.numpy_kernel import NumpyKernel


class IntegratedPWA:
    """Gram-matrix phase-space integration for the shared-ck PWA model.

    ``params`` = {"ck", "m0", "g0"}; ck length N = n_wave/n_proj.
    """

    def __init__(self, config):
        from ampfit.amp_cache import build_amp_cache_layout  # noqa: F401
        self.config = config
        self.n_wave = config["matrix_angle"].shape[1]
        self.n_proj = int(config.get("n_proj", 1) or 1)
        if self.n_wave % self.n_proj != 0:
            raise ValueError(
                f"integrated_pwa: n_wave={self.n_wave} not divisible by "
                f"n_proj={self.n_proj}")
        self.n_wave_base = self.n_wave // self.n_proj     # N = ck length
        self._kernel = NumpyKernel(config)
        self._k = self._kernel

    # ------------------------------------------------------------------
    def _common(self, data, m0, g0):
        """Per-event spatial amplitude factors common (ne, n_wave), p-major.

        Verbatim forward chain of the numpy_pwa kernel.
        """
        k = self._k
        n_wave = self.n_wave
        mass = data["mass"]
        ne = mass.shape[0]

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

        return (1.0 / bw_p) * (fa * fl_p)                  # (ne, n_wave)

    # ------------------------------------------------------------------
    def gram(self, data, m0, g0, weight=None):
        """Wave Gram matrix D (N, N) for the phsp sample at m0/g0.

        D[k,k'] = Σ_e w_e Σ_p conj(common[e,pN+k])·common[e,pN+k'].
        """
        common = self._common(data, m0, g0)
        ne = common.shape[0]
        P = self.n_proj
        N = self.n_wave_base
        cm = common.reshape(ne, P, N)
        w = np.asarray(weight, dtype=float) if weight is not None \
            else np.ones(ne)
        sw = np.sqrt(w)
        # einsum over events & projections: D = Σ_e w_e Σ_p conj(cm)·cm
        D = np.einsum("e,epk,epj->kj", w, cm.conj(), cm, optimize=True)
        # Hermitian average (numerical; same value, exactly HPSD in exact math)
        D = (D + D.conj().T) / 2.0
        return D

    def norm(self, ck, D):
        """∫dΦ |A|² = Σ_{k,k'} ck_k·conj(ck_k')·D[k,k']  (= real ≥ 0)."""
        ck = np.asarray(ck)
        return float(np.real(ck.conj() @ (D @ ck)))

    def norm_from_data(self, data, params, weight=None):
        """Direct (un-integrated) norm: Σ_e w_e P(e) — for validation."""
        from ampfit.numpy_pwa import NumpyPWA
        npw = NumpyPWA(self.config)
        h = npw.load_data(data)
        _, _, P = npw.compute(params, h)
        if P is None:
            raise RuntimeError("norm_from_data: per-event P not returned")
        w = np.asarray(weight, dtype=float) if weight is not None \
            else np.ones(len(P))
        return float(np.sum(w * P))

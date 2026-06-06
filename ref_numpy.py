import numpy as np


"""
A = ck [\\prod_i BW_i(m_i)] [\\prod_j F_j(q_j)] \\sum_n basis_n(angle) M_ang_nw

For each wave w:
  amp_w[w] = ck[w] * prod_r bw[w,r] * prod_d fl[w,d] * sum_n basis_n * M_ang[n,w]

The full amplitude is split into two CP-conjugate groups:
  amp = (amp_waves.reshape(nevt, 2, -1)).sum(axis=-1) -> (nevt, 2)

Time-dependent mixing:
  PB = |ep*A0 + poq*em*A1|^2
  PBbar = |em/poq*A0 + ep*A1|^2
  P = (1-frac)*(1-ap)*PB + frac*(1+ap)*PBbar

Objective:
  norm=None: Q = sum(weight * P)
  norm=...:  Q = -sum(weight * log(P/norm + bkg))
"""


class Kernel:
    def __init__(self, config):
        self.config = config
        # Expected config keys (set by subclass/loader):
        # gamma_table     (gamma_types, n_int) complex
        # fl_table        (fl_types, n_int) real
        # matrix_gamma    (n_m0, n_gamma) real
        # matrix_ang      (nbasis, nwaves) complex
        # g0_index        (n_gamma,) int
        # gamma_index     (n_gamma,) int
        # gamma_type      (n_gamma,) int
        # gamma_min       scalar
        # gamma_delta     scalar
        # m0_index        (n_bw,) int
        # bw_index        (n_bw,) int
        # bw_gamma_index  (n_bw,) int
        # bw_order        (nwaves * nres,) int
        # q_index         (n_fl,) int
        # fl_type         (n_fl,) int
        # fl_min          scalar
        # fl_delta        scalar
        # fl_order        (nwaves * ndecays,) int
        # angle_index     (n_ang,) int
        # angle_k         (n_ang,) real
        # angle_b         (n_ang,) real
        # ang_order       (nbasis, n_per_basis) int
        ...

    def compute(self, params, data, norm=None):
        ck = params["ck"]                          # (nwaves,) complex
        m0 = params["m0"]                          # (n_m0,) real
        g0 = params["g0"]                          # (n_g0,) real
        tp = params["time_params"]                 # [gamma, dg, dm, poqr, poqi, ap]
        gamma_tp, delta_gamma, delta_m, poqr, poqi, ap = tp
        poq = poqr * np.exp(1j * poqi)

        mass = data["mass"]                        # (nevt, ndim_mass)
        q_data = data["q"]                         # (nevt, ndim_q)
        angle = data["angle"]                      # (nevt, ndim_angle)
        time = data["time"]                        # (nevt,)
        weight = data["weight"]                    # (nevt,)
        frac = np.broadcast_to(np.asarray(data.get("frac", 0.0)), (mass.shape[0],))
        bkg = np.broadcast_to(np.asarray(data.get("bkg", 0.0)), (mass.shape[0],))

        nevt = mass.shape[0]
        nwaves = ck.shape[0]

        # ============================================================
        #  1) Gamma — energy-dependent width via interpolation
        # ============================================================
        g0a = np.take(g0, self.g0_index, axis=-1)                            # (n_gamma,)
        mass_for_gamma = np.take(mass, self.gamma_index, axis=-1)             # (nevt, n_gamma)
        gamma_interp = self.interp(mass_for_gamma, self.gamma_table,
                                   self.gamma_type, self.gamma_min,
                                   self.gamma_delta)                          # (nevt, n_gamma) complex
        gamma_val = g0a * gamma_interp                                        # (nevt, n_gamma) complex
        gamma_for_mass = np.einsum('ij,...j->...i',
                                   self.matrix_gamma, gamma_val)              # (nevt, n_m0) complex

        # ============================================================
        #  2) Breit-Wigner
        # ============================================================
        m0a = np.take(m0, self.m0_index, axis=-1)                             # (n_bw,)
        mass_for_bw = np.take(mass, self.bw_index, axis=-1)                   # (nevt, n_bw)
        gamma_for_bw = np.take(gamma_for_mass,
                               self.bw_gamma_index, axis=-1)                  # (nevt, n_bw) complex
        bwdom = m0a**2 - mass_for_bw**2 - 1j * m0a * gamma_for_bw            # (nevt, n_bw)
        bw = 1.0 / bwdom                                                     # (nevt, n_bw)

        bw_ordered = np.take(bw, self.bw_order, axis=-1)                     # (nevt, nwaves * nres)
        bw_reshaped = bw_ordered.reshape(nevt, nwaves, -1)                   # (nevt, nwaves, nres)
        bwa = np.prod(bw_reshaped, axis=-1)                                  # (nevt, nwaves) complex

        # ============================================================
        #  3) Form factors
        # ============================================================
        fl_q = np.take(q_data, self.q_index, axis=-1)                        # (nevt, n_fl)
        fl = self.interp(fl_q, self.fl_table,
                         self.fl_type, self.fl_min, self.fl_delta)           # (nevt, n_fl) real
        fl_ordered = np.take(fl, self.fl_order, axis=-1)                     # (nevt, nwaves*ndec)
        fl_reshaped = fl_ordered.reshape(nevt, nwaves, -1)                   # (nevt, nwaves, ndec)
        fla = np.prod(fl_reshaped, axis=-1)                                  # (nevt, nwaves) real

        # ============================================================
        #  4) Angular basis
        # ============================================================
        ang = np.take(angle, self.angle_index, axis=-1)                      # (nevt, n_ang)
        ang_a = self.angle_k * ang + self.angle_b
        cosang = np.cos(ang_a)                                               # (nevt, n_ang)
        cos_ordered = np.take(cosang, self.ang_order, axis=-1)               # (nevt, nbasis, n_per)
        cosa = np.prod(cos_ordered, axis=-1)                                 # (nevt, nbasis) real
        fa = cosa @ self.matrix_ang                                          # (nevt, nwaves) complex
        # NB: matrix_ang is (nbasis, nwaves)

        # ============================================================
        #  5) Wave amplitude -> two CP groups
        # ============================================================
        T = bwa * fla * fa                                                   # (nevt, nwaves) complex
        amp_waves = ck[np.newaxis, :] * T                                    # (nevt, nwaves) complex
        amp = amp_waves.reshape(nevt, 2, -1).sum(axis=-1)                   # (nevt, 2)
        amp0, amp1 = amp[:, 0], amp[:, 1]

        # ============================================================
        #  6) Time-dependent mixing
        # ============================================================
        eL = np.exp(1j * time * (-delta_m / 2 + 1j * (gamma_tp + delta_gamma / 2) / 2))
        eH = np.exp(1j * time * (delta_m / 2 + 1j * (gamma_tp - delta_gamma / 2) / 2))
        ep = (eL + eH) / 2
        em = (eL - eH) / 2

        X = ep * amp0 + poq * em * amp1                                      # (nevt,) complex
        Y = em / poq * amp0 + ep * amp1                                      # (nevt,) complex
        PB = np.abs(X) ** 2
        PBbar = np.abs(Y) ** 2
        P = (1 - frac) * (1 - ap) * PB + frac * (1 + ap) * PBbar

        # ============================================================
        #  7) Objective
        # ============================================================
        if norm is None:
            Q = np.sum(weight * P)
            dQ_dP = weight
            dQ_dnorm = None
        else:
            Pnorm = P / norm + bkg
            Q = -np.sum(weight * np.log(Pnorm))
            dQ_dPnorm = -weight / Pnorm
            dQ_dP = dQ_dPnorm / norm
            dQ_dnorm = np.sum(dQ_dPnorm * (-P / norm ** 2))

        # ============================================================
        #  8) Gradients
        # ============================================================
        # --- Adjoints for |X|^2, |Y|^2 ---
        # P = (1-f)*(1-ap)*X*conj(X) + f*(1+ap)*Y*conj(Y)
        # ∂P/∂X = (1-f)*(1-ap)*conj(X)   (Wirtinger)
        # ∂P/∂Y = f*(1+ap)*conj(Y)
        adjX = (1 - frac) * (1 - ap) * np.conj(X)
        adjY = frac * (1 + ap) * np.conj(Y)

        # ∂Q/∂X = dQ/dP * ∂P/∂X   (Wirtinger)
        dQ_dX = dQ_dP * adjX
        dQ_dY = dQ_dP * adjY

        # --- Propagate to amp0, amp1 ---
        # X = ep*A0 + poq*em*A1,  Y = em/poq*A0 + ep*A1
        # ∂Q/∂A0 = ∂Q/∂X*ep + ∂Q/∂Y*em/poq   (Wirtinger)
        dQ_damp0 = dQ_dX * ep + dQ_dY * em / poq
        dQ_damp1 = dQ_dX * poq * em + dQ_dY * ep

        # --- Propagate to amp_waves ---
        n_per_group = amp_waves.shape[1] // 2
        dQ_damp_waves = np.zeros_like(amp_waves)
        dQ_damp_waves[:, :n_per_group] = dQ_damp0[:, np.newaxis]
        dQ_damp_waves[:, n_per_group:] = dQ_damp1[:, np.newaxis]

        # --- ck gradient ---
        # amp_waves[:,w] = ck[w] * T[:,w]
        # ∂Q/∂ck[w] = Σ_e ∂Q/∂amp_waves[e,w] * T[e,w]   (Wirtinger)
        dQ_dck_wirt = np.sum(dQ_damp_waves * T, axis=0)                     # (nwaves,) complex
        # For optimisation (Re + Im coordinates):
        #   ∂Q/∂Re(ck) + i*∂Q/∂Im(ck) = 2 * conj(∂Q/∂ck̄) = 2 * conj(∂Q/∂ck)
        grads_ck = 2 * np.conj(dQ_dck_wirt)

        # --- m0 gradient ---
        # bw = 1/(m0a² - s - i*m0a*Γ(s))
        # dbw/dm0a = -bw² * (2*m0a - i*Γ(s))     (Γ(s) does not depend on m0)
        dbw_dm0a = -(bw ** 2) * (2 * m0a - 1j * gamma_for_bw)               # (nevt, n_bw)

        # d(bwa[w])/d(bw[d]) = bwa[w]/bw[d] if d used in wave w
        dBwa_dbw = bwa[..., np.newaxis] / bw_reshaped                        # (nevt, nwaves, nres)
        dQ_dT = dQ_damp_waves * ck[np.newaxis, :]                            # (nevt, nwaves)
        dQ_dbwa = dQ_dT * fla * fa                                           # (nevt, nwaves)
        dQ_dbw_reshaped = dQ_dbwa[..., np.newaxis] * dBwa_dbw                # (nevt, nwaves, nres)
        dQ_dbw = np.zeros((nevt, bw.shape[1]), dtype=complex)
        np.add.at(dQ_dbw, (slice(None), self.bw_order),
                  dQ_dbw_reshaped.reshape(nevt, -1))

        # Full real derivative: dQ/dm0a = 2*Re(Σ dQ_dbw * dbw/dm0a)
        dQ_dm0a = 2 * np.sum(dQ_dbw * dbw_dm0a, axis=0).real                # (n_bw,)
        dQ_dm0 = np.zeros(m0.shape, dtype=float)
        np.add.at(dQ_dm0, self.m0_index, dQ_dm0a)

        # --- g0 gradient ---
        # gamma_for_bw[d] = Σ_j g0a[j] * gamma_interp[:,j] * mat_gamma[bwΓ_idx[d], j]
        # dbw/dΓ = -bw² * (-i*m0a)
        dbw_dgamma = -(bw ** 2) * (-1j * m0a)                                # (nevt, n_bw)
        dQ_dgamma_bw = dQ_dbw * dbw_dgamma                                    # (nevt, n_bw)

        dQ_dgamma_mass = np.zeros((nevt, gamma_for_mass.shape[1]), dtype=complex)
        np.add.at(dQ_dgamma_mass, (slice(None), self.bw_gamma_index),
                  dQ_dgamma_bw)

        # gamma_for_mass[:,i] = Σ_j gamma_val[:,j] * mat_gamma[i,j]
        # ∂Q/∂gamma_val[:,j] = Σ_i ∂Q/∂gamma_for_mass[:,i] * mat_gamma[i,j]
        dQ_dgamma_val = np.einsum('...i,ij->...j', dQ_dgamma_mass,
                                  self.matrix_gamma)                         # (nevt, n_gamma)

        # gamma_val[:,j] = g0a[j] * gamma_interp[:,j]
        # dQ/dg0a[j] = 2*Re(Σ_e dQ_dgamma_val[e,j] * gamma_interp[e,j])
        dQ_dg0a = 2 * np.sum((dQ_dgamma_val * gamma_interp).real, axis=0)   # (n_gamma,)
        dQ_dg0 = np.zeros(g0.shape, dtype=float)
        np.add.at(dQ_dg0, self.g0_index, dQ_dg0a)

        # --- time_params gradients ---
        # Derivatives of eL, eH w.r.t. time_params
        deL_dgamma = eL * (-time / 2)
        deH_dgamma = eH * (-time / 2)
        deL_ddg = eL * (-time / 4)
        deH_ddg = eH * (time / 4)
        deL_ddm = eL * (-1j * time / 2)
        deH_ddm = eH * (1j * time / 2)

        dep_dgamma = (deL_dgamma + deH_dgamma) / 2
        dem_dgamma = (deL_dgamma - deH_dgamma) / 2
        dep_ddg = (deL_ddg + deH_ddg) / 2
        dem_ddg = (deL_ddg - deH_ddg) / 2
        dep_ddm = (deL_ddm + deH_ddm) / 2
        dem_ddm = (deL_ddm - deH_ddm) / 2

        # dX/d(param) = d(ep)/d(param)*A0 + poq*d(em)/d(param)*A1
        dX_dgamma = dep_dgamma * amp0 + poq * dem_dgamma * amp1
        dY_dgamma = dem_dgamma / poq * amp0 + dep_dgamma * amp1
        dX_ddg = dep_ddg * amp0 + poq * dem_ddg * amp1
        dY_ddg = dem_ddg / poq * amp0 + dep_ddg * amp1
        dX_ddm = dep_ddm * amp0 + poq * dem_ddm * amp1
        dY_ddm = dem_ddm / poq * amp0 + dep_ddm * amp1

        dX_dpoqr = np.exp(1j * poqi) * em * amp1
        dY_dpoqr = -em / (poq ** 2) * np.exp(1j * poqi) * amp0
        dX_dpoqi = 1j * poq * em * amp1
        dY_dpoqi = -1j * em / poq * amp0
        dP_dap = -(1 - frac) * PB + frac * PBbar

        # Full real derivative: dQ/dp = 2*Re(Σ dQ_dX*dX/dp + dQ_dY*dY/dp)
        def real_grad(dX, dY):
            return 2 * np.real(np.sum(dQ_dX * dX + dQ_dY * dY))

        grads_tp = np.array([
            real_grad(dX_dgamma, dY_dgamma),
            real_grad(dX_ddg, dY_ddg),
            real_grad(dX_ddm, dY_ddm),
            real_grad(dX_dpoqr, dY_dpoqr),
            real_grad(dX_dpoqi, dY_dpoqi),
            np.sum(dQ_dP * dP_dap),
        ])

        grads = {
            "ck": grads_ck,
            "m0": dQ_dm0,
            "g0": dQ_dg0,
            "time_params": grads_tp,
        }
        if norm is not None:
            grads["norm"] = dQ_dnorm

        return P, Q, grads

    @staticmethod
    def interp(x, table, types, xmin, xdelta):
        diff = (x - xmin) / xdelta
        xbin = np.floor(diff).astype(np.int64)
        idx = types * table.shape[-1] + xbin
        left = np.take(table.ravel(), idx)
        right = np.take(table.ravel(), idx + 1)
        return (right - left) * (diff - xbin) + left

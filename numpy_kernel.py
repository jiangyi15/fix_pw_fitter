import numpy as np

class NumpyKernel:
    def __init__(self, config):
        self.m0_index = config["m0_index"]
        self.g0_index = config["g0_index"]
        self.fl_type = config["fl_type"]
        self.angle_k = config["angle_k"]
        self.angle_b = config["angle_b"]
        self.fl_type = config["fl_type"]
        self.mass_index = config["mass_index"]
        self.g0_mass_index = config["g0_mass_index"]
        self.fl_q_index = config["fl_q_index"]
        self.bw_order = config["bw_order"]
        self.fl_order = config["fl_order"]
        self.angle_index = config["angle_index"]
        self.matrix_angle = config["matrix_angle"]
        self.matrix_gamma = config["matrix_gamma"]
        self.gamma_table = config["gamma_table"]
        self.fl_table = config["fl_table"]
        self.gamma_min = config["gamma_min"]
        self.fl_min = config["fl_min"]
        self.gamma_delta = config["gamma_delta"]
        self.fl_delta = config["fl_delta"]


        self.n_basis = self.angle_k.shape[0]
        self.n_angle = self.angle_k.shape[1]
        self.n_wave = self.matrix_angle.shape[1]
        self.n_res = self.bw_order.size//self.n_wave
        self.n_decay = self.fl_order.size//self.n_wave





    def _compute(self, params, data, norm=None):

        ck = params["ck"]
        m0 = params["m0"]
        g0 = params["g0"]
        Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi = params["scalar"]

        mass = data["mass"]
        momentum = data["q"]
        angle = data["angle"]
        frac = data["frac"]

        # bw
        g0_all = np.take(g0, self.g0_index)
        g0_m = np.take(mass, self.g0_mass_index, axis=-1)

        g = g0_all * self.interp(self.gamma_table, self.g0_index, g0_m, self.gamma_min, self.gamma_delta)
        g_bw = np.dot(g, self.matrix_gamma)
        m0_all = np.take(m0, self.m0_index)
        m0_m = np.take(mass, self.mass_index, axis=-1)
        bw_dom = m0_all**2 - m0_m**2 - 1j * m0_all * g_bw
        bw_dom_all = np.take(bw_dom, self.bw_order, axis=-1)
        bw_p = np.prod(np.reshape(bw_dom_all, (-1, self.n_wave, self.n_res)), axis=-1)

        # fl
        fl_q = np.take(momentum, self.fl_q_index, axis=-1)
        fl = self.interp(self.fl_table, self.fl_type, fl_q, self.fl_min, self.fl_delta)
        fl_all = np.take(fl, self.fl_order)
        fl_p = np.prod(np.reshape(fl_all, (-1, self.n_wave, self.n_decay)), axis=-1)

        # angle
        ang = np.take(angle, self.angle_index, axis=-2)
        ka = np.prod(np.cos(ang * self.angle_k + self.angle_b), axis=-1)
        fa = np.dot(ka, self.matrix_angle)
        a = ck * 1/bw_p * fa * fl_p
        a = np.reshape(a, (-1, 2, self.n_wave//2))
        ap = np.sum(a[:,0], axis=-1)
        am = np.sum(a[:,1], axis=-1)
        eL= np.exp( -1j * data["time"] * (-Delta_m/2  - 1j * (Gamma +  Delta_Gamma/2)/2 ))
        eH= np.exp( -1j * data["time"] * (+Delta_m/2  - 1j * (Gamma -  Delta_Gamma/2)/2 ))
        gp = (eL + eH)/2
        gm = (eL - eH)/2

        poq = poq_rho * np.exp(1j * pop_phi)

        pb = np.abs( gp * ap + gm * poq * am  )**2
        pbbar = np.abs( gm/poq * ap + gp * am  )**2

        P = frac * pb * (1-A_p) + (1-frac) * pbbar * (1+A_p)
        if norm is None:
            Q = np.sum(data["weight"] * P)
        else:
            Q = -np.sum(data["weight"] *  np.log(P /norm + data["bkg"]))

        # Compute gradients
        # dQ/dP
        if norm is None:
            dQ_dP = data["weight"]
        else:
            dQ_dP = -data["weight"] / (P / norm + data["bkg"])

        # dP/d(pb, pbbar, A_p)
        dP_dpb = frac * (1 - A_p)
        dP_dpbbar = (1 - frac) * (1 + A_p)
        dP_dAp = -frac * pb + (1 - frac) * pbbar

        # Gradient for A_p
        dQ_dAp = np.sum(dQ_dP * dP_dAp)

        # d(pb)/d(ap, am) and d(pbbar)/d(ap, am)
        pap = gp * ap + gm * poq * am
        pam = gm/poq * ap + gp * am

        dQ_dpb_bar = dQ_dP * (dP_dpb if norm is None else dP_dpb)
        dQ_dpbbar_bar = dQ_dP * (dP_dpbbar if norm is None else dP_dpbbar)

        # d(pb)/d(ap), d(pb)/d(am)
        d_pb_dap = 2 * np.real(gp * np.conj(pap))
        d_pb_dam = 2 * np.real(gm * poq * np.conj(pap))

        # d(pbbar)/d(ap), d(pbbar)/d(am)
        d_pbbar_dap = 2 * np.real(gm/poq * np.conj(pam))
        d_pbbar_dam = 2 * np.real(gp * np.conj(pam))

        # Chain rule for ap, am
        dQ_dap = dQ_dpb_bar * d_pb_dap + dQ_dpbbar_bar * d_pbbar_dap
        dQ_dam = dQ_dpb_bar * d_pb_dam + dQ_dpbbar_bar * d_pbbar_dam

        # Backprop through reshape: a -> (ap, am)
        # a shape: (-1, 2, n_wave//2)
        # ap = sum(a[:,0], axis=-1), am = sum(a[:,1], axis=-1)
        dQ_da = np.zeros_like(a)
        dQ_da[:, 0, :] = dQ_dap[:, np.newaxis]
        dQ_da[:, 1, :] = dQ_dam[:, np.newaxis]

        # Backprop through a = ck * 1/bw_p * fa * fl_p
        a_reshaped = a.reshape(-1, self.n_wave)

        dQ_dck = dQ_da.reshape(-1, self.n_wave) * (1/bw_p * fa * fl_p)
        dQ_dbw_p = dQ_da.reshape(-1, self.n_wave) * ck * (-1/bw_p**2) * fa * fl_p
        dQ_dfa = dQ_da.reshape(-1, self.n_wave) * ck * (1/bw_p) * fl_p
        dQ_dfl_p = dQ_da.reshape(-1, self.n_wave) * ck * (1/bw_p) * fa

        # Gradient for ck
        grad_ck = np.sum(dQ_dck, axis=0)

        # Backprop through fa = dot(ka, matrix_angle)
        # ka shape: (-1, n_basis), matrix_angle: (n_basis, n_wave)
        dQ_dka = np.dot(dQ_dfa, self.matrix_angle.T)

        # Backprop through ka = prod(cos(ang * angle_k + angle_b), axis=-1)
        # ang shape: (n_events, n_basis, n_angle), angle_k: (n_basis, n_angle)
        ang = np.take(angle, self.angle_index, axis=-2)
        cos_term = np.cos(ang * self.angle_k + self.angle_b)
        sin_term = np.sin(ang * self.angle_k + self.angle_b)

        # Gradient w.r.t ka
        # ka = prod(cos_term, axis=-1), so d(ka)/d(cos_term[:, :, i]) = prod(cos_term except i)
        dQ_dcos_term = np.zeros_like(cos_term)
        for i in range(self.n_angle):
            mask = np.ones(self.n_angle, dtype=bool)
            mask[i] = False
            prod_except_i = np.prod(cos_term[:, :, mask], axis=-1)
            dQ_dcos_term[:, :, i] = prod_except_i * dQ_dka

        # d(cos_term)/d(ang) = -sin_term * angle_k
        # angle_k shape: (n_basis, n_angle)
        dQ_dang = -sin_term * self.angle_k[np.newaxis, :, :] * dQ_dcos_term

        # Gradient for angle_k and angle_b (if needed as parameters)
        # These appear to be fixed config parameters, so no gradient

        # Backprop through fl_p
        # fl_p = prod(fl_all, axis=-1) where fl_all shape: (-1, n_wave, n_decay)
        # fl comes from interpolation of fixed table, so no gradient parameters here

        # Backprop through bw_p
        # bw_p = prod(bw_dom_all, axis=-1) where bw_dom_all shape: (n_events, n_wave, n_res)
        # bw_dom shape: (n_events, n_unique_bw), bw_dom_all shape: (n_events, n_wave * n_res)
        bw_dom_all = np.take(bw_dom, self.bw_order, axis=-1)
        n_events = bw_dom_all.shape[0]
        bw_dom_all_reshaped = bw_dom_all.reshape(n_events, self.n_wave, self.n_res)

        dQ_dbw_dom_all = np.zeros_like(bw_dom_all_reshaped)
        for i in range(self.n_res):
            mask = np.ones(self.n_res, dtype=bool)
            mask[i] = False
            prod_except_i = np.prod(bw_dom_all_reshaped[:, :, mask], axis=-1)
            dQ_dbw_dom_all[:, :, i] = dQ_dbw_p * prod_except_i

        # dQ_dbw_dom_all shape: (n_events, n_wave, n_res)
        # Need to map back to bw_dom through bw_order
        # bw_dom_all = take(bw_dom, bw_order), so we need to scatter-add gradients
        dQ_dbw_dom = np.zeros_like(bw_dom)  # shape: (n_events, n_unique_bw)

        for wave_idx in range(self.n_wave):
            for res_idx in range(self.n_res):
                order_idx = wave_idx * self.n_res + res_idx
                bw_idx = self.bw_order[order_idx]
                dQ_dbw_dom[:, bw_idx] += dQ_dbw_dom_all[:, wave_idx, res_idx]

        # Backprop through bw_dom = m0_all**2 - m0_m**2 - 1j * m0_all * g_bw
        # bw_dom shape: (n_events, n_unique_bw)
        # m0_all shape: (n_unique_bw,), g_bw shape: (n_events, n_unique_bw)
        # For m0_all: d(bw_dom)/d(m0_all) = 2*m0_all - 1j*g_bw (broadcasted)
        # For g_bw: d(bw_dom)/d(g_bw) = -1j*m0_all (broadcasted)

        grad_m0 = np.zeros_like(m0)
        grad_g0 = np.zeros_like(g0)

        # Gradient for m0
        # d(bw_dom[:, i])/d(m0_all[i]) = 2*m0_all[i] - 1j*g_bw[:, i]
        # Need to sum over events and accumulate over all bw_idx that use same m0
        for bw_idx in range(len(self.m0_index)):
            m0_param_idx = self.m0_index[bw_idx]
            d_bw_dom_dm0 = 2 * m0_all[bw_idx] - 1j * g_bw[:, bw_idx]
            grad_m0[m0_param_idx] += np.sum(np.real(dQ_dbw_dom[:, bw_idx] * d_bw_dom_dm0))

        # Backprop through g_bw = dot(g, matrix_gamma)
        # g shape: (n_events, n_gamma), matrix_gamma: (n_gamma, n_unique_bw)
        # g_bw shape: (n_events, n_unique_bw)
        # d(g_bw)/d(g) = matrix_gamma.T
        dQ_dg = np.dot(np.real(dQ_dbw_dom * (-1j * m0_all)), self.matrix_gamma.T)

        # Backprop through g = g0_all * interp(...)
        # g shape: (n_events, n_gamma)
        # g0_all shape: (n_gamma,), interp_result shape: (n_events, n_gamma)
        # d(g)/d(g0_all[i]) = interp_result[:, i]

        for gamma_idx in range(len(self.g0_index)):
            g0_param_idx = self.g0_index[gamma_idx]
            # Get interpolated value for this gamma index across all events
            interp_val = self.interp(self.gamma_table, self.g0_index[gamma_idx],
                                     g0_m[:, gamma_idx], self.gamma_min, self.gamma_delta)
            # d(g[:, gamma_idx])/d(g0_all[gamma_idx]) = interp_val
            # And g0_all[gamma_idx] = g0[g0_index[gamma_idx]]
            grad_g0[g0_param_idx] += np.sum(np.real(dQ_dg[:, gamma_idx] * interp_val))

        # Backprop for scalar parameters (Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi)
        # eL= exp(-1j * time * (-Delta_m/2 - 1j*(Gamma + Delta_Gamma/2)/2))
        # eH= exp(-1j * time * (+Delta_m/2 - 1j*(Gamma - Delta_Gamma/2)/2))
        # gp = (eL + eH)/2, gm = (eL - eH)/2

        time = data["time"]

        # Derivatives of eL and eH
        # d(eL)/d(Gamma) = -1j * time * (-1j/2) * eL = -time/2 * eL
        # d(eL)/d(Delta_Gamma) = -1j * time * (-1j/4) * eL = -time/4 * eL
        # d(eL)/d(Delta_m) = -1j * time * (1/2) * eL = -1j*time/2 * eL

        deL_dGamma = -time/2 * eL
        deL_dDeltaGamma = -time/4 * eL
        deL_dDeltaM = -1j * time/2 * eL

        deH_dGamma = -time/2 * eH
        deH_dDeltaGamma = time/4 * eH
        deH_dDeltaM = 1j * time/2 * eH

        # Derivatives of gp and gm
        dgp_dGamma = (deL_dGamma + deH_dGamma) / 2
        dgp_dDeltaGamma = (deL_dDeltaGamma + deH_dDeltaGamma) / 2
        dgp_dDeltaM = (deL_dDeltaM + deH_dDeltaM) / 2

        dgm_dGamma = (deL_dGamma - deH_dGamma) / 2
        dgm_dDeltaGamma = (deL_dDeltaGamma - deH_dDeltaGamma) / 2
        dgm_dDeltaM = (deL_dDeltaM - deH_dDeltaM) / 2

        # Recompute d(pb)/d(gp, gm) and d(pbbar)/d(gp, gm) with chain rule
        # pb = |gp * ap + gm * poq * am|^2
        # d(pb)/d(gp) = 2 * real(conj(gp*ap + gm*poq*am) * ap)
        # d(pb)/d(gm) = 2 * real(conj(gp*ap + gm*poq*am) * poq * am)
        d_pb_dgp = 2 * np.real(np.conj(pap) * ap)
        d_pb_dgm = 2 * np.real(np.conj(pap) * poq * am)

        d_pbbar_dgp = 2 * np.real(np.conj(pam) * am)
        d_pbbar_dgm = 2 * np.real(np.conj(pam) * ap / poq)

        # Gradient for Gamma
        dQ_dGamma = np.sum(dQ_dpb_bar * d_pb_dgp * dgp_dGamma +
                          dQ_dpb_bar * d_pb_dgm * dgm_dGamma +
                          dQ_dpbbar_bar * d_pbbar_dgp * dgp_dGamma +
                          dQ_dpbbar_bar * d_pbbar_dgm * dgm_dGamma)

        # Gradient for Delta_Gamma
        dQ_dDeltaGamma = np.sum(dQ_dpb_bar * d_pb_dgp * dgp_dDeltaGamma +
                               dQ_dpb_bar * d_pb_dgm * dgm_dDeltaGamma +
                               dQ_dpbbar_bar * d_pbbar_dgp * dgp_dDeltaGamma +
                               dQ_dpbbar_bar * d_pbbar_dgm * dgm_dDeltaGamma)

        # Gradient for Delta_m
        dQ_dDeltaM = np.sum(dQ_dpb_bar * d_pb_dgp * dgp_dDeltaM +
                           dQ_dpb_bar * d_pb_dgm * dgm_dDeltaM +
                           dQ_dpbbar_bar * d_pbbar_dgp * dgp_dDeltaM +
                           dQ_dpbbar_bar * d_pbbar_dgm * dgm_dDeltaM)

        # Gradient for poq_rho and pop_phi
        # poq = poq_rho * exp(1j * pop_phi)
        # d(pb)/d(poq_rho) = 2 * real(conj(pap) * gm * am * exp(1j*pop_phi))
        # d(pb)/d(pop_phi) = 2 * real(conj(pap) * gm * poq_rho * am * 1j * exp(1j*pop_phi))
        d_pb_dpoq_rho = 2 * np.real(np.conj(pap) * gm * am * np.exp(1j * pop_phi))
        d_pb_dpop_phi = 2 * np.real(np.conj(pap) * gm * poq_rho * am * 1j * np.exp(1j * pop_phi))

        # d(pbbar)/d(poq_rho) = 2 * real(conj(pam) * (-gm/poq_rho^2 * ap))
        # d(pbbar)/d(pop_phi) = 2 * real(conj(pam) * gp * am * 1j * exp(1j*pop_phi) + ...)
        d_pbbar_dpoq_rho = 2 * np.real(np.conj(pam) * (-gm / (poq_rho**2) * ap))
        d_pbbar_dpop_phi = 2 * np.real(np.conj(pam) * (gm / poq_rho * ap * (-1j) + gp * am * 1j) * poq_rho * np.exp(1j * pop_phi))

        dQ_dpoq_rho = np.sum(dQ_dpb_bar * d_pb_dpoq_rho + dQ_dpbbar_bar * d_pbbar_dpoq_rho)
        dQ_dpop_phi = np.sum(dQ_dpb_bar * d_pb_dpop_phi + dQ_dpbbar_bar * d_pbbar_dpop_phi)

        # Gradient for norm (if applicable)
        if norm is None:
            grad_norm = None
        else:
            grad_norm = np.sum(data["weight"] / (P / norm + data["bkg"]) * (P / norm**2))

        grads = {
            "ck": grad_ck,
            "m0": grad_m0,
            "g0": grad_g0,
            "scalar": (dQ_dGamma, dQ_dDeltaGamma, dQ_dDeltaM, dQ_dAp, dQ_dpoq_rho, dQ_dpop_phi),
            "norm": grad_norm,
        }
        return Q, grads, P


    def interp(self, table, types, x, xmin, xdelta):
        diff = (x - xmin)/xdelta
        xbin = np.floor(diff).astype(np.intp)
        n_bins = table.shape[-1]
        xbin = np.clip(xbin, 0, n_bins - 2)
        delta = diff - xbin
        idx = types * n_bins + xbin
        left = np.take(table.flatten(), idx)
        right = np.take(table.flatten(), idx + 1)
        return (right - left) * delta + left












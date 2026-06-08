"""
Optimized kernel with merged index operations.
Precompute composed indices to reduce np.take overhead.
"""
import numpy as np


class NumpyKernelMergedIndices:
    def __init__(self, config):
        # Store config arrays
        self.m0_index = config["m0_index"]
        self.g0_index = config["g0_index"]
        self.fl_type = config["fl_type"]
        self.angle_k = config["angle_k"]
        self.angle_b = config["angle_b"]
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

        # Compute dimensions
        self.n_basis = self.angle_k.shape[0]
        self.n_angle = self.angle_k.shape[1]
        self.n_wave = self.matrix_angle.shape[1]
        self.n_res = self.bw_order.size // self.n_wave
        self.n_decay = self.fl_order.size // self.n_wave
        
        # Precompute composed indices
        self._precompute_composed_indices()
        
    def _precompute_composed_indices(self):
        """
        Precompute composed indices to merge sequential np.take operations.
        
        Key insight: Can merge indexing with matrix multiplication!
        Instead of: g_bw = dot(g, matrix_gamma), then g_bw[bw_order]
        We do:      g_bw_direct = dot(g, matrix_gamma[:, bw_order])
        """
        # Compose m0 indices: m0[m0_index] then [bw_order] → m0[m0_index[bw_order]]
        self.m0_composed = self.m0_index[self.bw_order]
        
        # Compose mass indices similarly
        self.mass_composed = self.mass_index[self.bw_order]
        
        # OPTIMIZATION: Compose matrix_gamma with bw_order
        # Instead of: g_bw = dot(g, matrix_gamma), then g_bw_direct = g_bw[bw_order]
        # We do: matrix_gamma_direct = matrix_gamma[:, bw_order], then g_bw_direct = dot(g, matrix_gamma_direct)
        # This eliminates the intermediate g_bw array and one np.take operation!
        self.matrix_gamma_direct = np.take(self.matrix_gamma, self.bw_order, axis=1)
        # shape: (n_gamma, n_wave * n_res)
        
        # Count occurrences for gradient scatter
        self.m0_param_counts = np.bincount(self.m0_index, minlength=len(np.unique(self.m0_index)))
        
    def interp(self, table, types, x, xmin, xdelta):
        """Vectorized linear interpolation"""
        diff = (x - xmin) / xdelta
        xbin = np.floor(diff).astype(np.intp)
        n_bins = table.shape[-1]
        xbin = np.clip(xbin, 0, n_bins - 2)
        delta = diff - xbin
        idx = types * n_bins + xbin
        left = np.take(table.flatten(), idx)
        right = np.take(table.flatten(), idx + 1)
        return (right - left) * delta + left

    def _compute(self, params, data, norm=None):
        """
        Compute forward pass and gradients with merged index operations.
        """
        
        # Extract parameters
        ck = params["ck"]
        m0 = params["m0"]
        g0 = params["g0"]
        Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi = params["scalar"]
        
        # Extract data
        mass = data["mass"]
        momentum = data["q"]
        angle = data["angle"]
        frac = data["frac"]
        time = data["time"]
        weight = data["weight"]
        bkg = data.get("bkg", 0.0)
        
        n_events = mass.shape[0]
        
        # ==================== FORWARD PASS ====================
        
        # BW propagators - FULLY OPTIMIZED
        g0_all = np.take(g0, self.g0_index)
        g0_m = np.take(mass, self.g0_mass_index, axis=-1)
        g_interp = self.interp(self.gamma_table, self.g0_index, g0_m, 
                               self.gamma_min, self.gamma_delta)
        g = g0_all * g_interp
        
        # OPTIMIZED: Use precomputed matrix_gamma_direct
        # Instead of: g_bw = dot(g, matrix_gamma), then g_bw_direct = g_bw[bw_order]
        # We do: g_bw_direct = dot(g, matrix_gamma_direct)
        # This eliminates one np.take and the intermediate g_bw array!
        g_bw_direct = np.dot(g, self.matrix_gamma_direct)
        # shape: (n_events, n_wave * n_res)
        
        # OPTIMIZED: Use composed indices for m0
        m0_direct = np.take(m0, self.m0_composed)  # Single take! shape: (n_wave * n_res,)
        mass_direct = np.take(mass, self.mass_composed, axis=-1)  # shape: (n_events, n_wave * n_res)
        
        # Compute bw_dom directly in final order
        bw_dom_all = m0_direct**2 - mass_direct**2 - 1j * m0_direct * g_bw_direct
        # shape: (n_events, n_wave * n_res)
        
        bw_dom_all_reshaped = bw_dom_all.reshape(n_events, self.n_wave, self.n_res)
        bw_p = np.prod(bw_dom_all_reshaped, axis=-1)
        
        # FL factors - similar optimization
        fl_q = np.take(momentum, self.fl_q_index, axis=-1)
        fl = self.interp(self.fl_table, self.fl_type, fl_q, 
                        self.fl_min, self.fl_delta)
        fl_all = np.take(fl, self.fl_order)
        fl_p = np.prod(fl_all.reshape(-1, self.n_wave, self.n_decay), axis=-1)
        
        # Angular factors
        ang = np.take(angle, self.angle_index, axis=-2)
        ka = np.prod(np.cos(ang * self.angle_k + self.angle_b), axis=-1)
        fa = np.dot(ka, self.matrix_angle)
        
        # Amplitudes with merged calculations
        one_over_bw = 1.0 / bw_p
        fa_times_fl = fa * fl_p
        common_amp_factor = one_over_bw * fa_times_fl
        
        a = ck * common_amp_factor
        a_reshaped = a.reshape(-1, 2, self.n_wave // 2)
        ap = np.sum(a_reshaped[:, 0, :], axis=-1)
        am = np.sum(a_reshaped[:, 1, :], axis=-1)
        
        # Time evolution
        eL = np.exp(-1j * time * (-Delta_m/2 - 1j * (Gamma + Delta_Gamma/2)/2))
        eH = np.exp(-1j * time * (+Delta_m/2 - 1j * (Gamma - Delta_Gamma/2)/2))
        gp = (eL + eH) / 2
        gm = (eL - eH) / 2
        
        # Probabilities
        poq = poq_rho * np.exp(1j * pop_phi)
        pap = gp * ap + gm * poq * am
        pam = (gm / poq) * ap + gp * am
        
        pb = np.abs(pap)**2
        pbbar = np.abs(pam)**2
        
        P = frac * pb * (1 - A_p) + (1 - frac) * pbbar * (1 + A_p)
        
        # Loss
        if norm is None:
            Q = np.sum(weight * P)
            dQ_dP = weight
        else:
            Q = -np.sum(weight * np.log(P / norm + bkg))
            dQ_dP = -weight / (P / norm + bkg)
        
        # ==================== BACKWARD PASS ====================
        
        # Probability gradients
        dP_dpb = frac * (1 - A_p)
        dP_dpbbar = (1 - frac) * (1 + A_p)
        dP_dAp = -frac * pb + (1 - frac) * pbbar
        dQ_dAp = np.sum(dQ_dP * dP_dAp)
        
        dQ_dpb = dQ_dP * dP_dpb
        dQ_dpbbar = dQ_dP * dP_dpbbar
        
        # Amplitude gradients
        d_pb_dap = 2 * np.real(gp * np.conj(pap))
        d_pb_dam = 2 * np.real(gm * poq * np.conj(pap))
        d_pbbar_dap = 2 * np.real((gm / poq) * np.conj(pam))
        d_pbbar_dam = 2 * np.real(gp * np.conj(pam))
        
        dQ_dap = dQ_dpb * d_pb_dap + dQ_dpbbar * d_pbbar_dap
        dQ_dam = dQ_dpb * d_pb_dam + dQ_dpbbar * d_pbbar_dam
        
        dQ_da = np.zeros_like(a_reshaped)
        dQ_da[:, 0, :] = dQ_dap[:, np.newaxis]
        dQ_da[:, 1, :] = dQ_dam[:, np.newaxis]
        dQ_da_flat = dQ_da.reshape(n_events, self.n_wave)
        
        # Gradients with merged calculations
        grad_ck = np.sum(dQ_da_flat * common_amp_factor, axis=0)
        dQ_dbw_p = dQ_da_flat * ck * (-one_over_bw) * common_amp_factor
        dQ_dfa = dQ_da_flat * ck * one_over_bw * fl_p
        dQ_dka = np.dot(dQ_dfa, self.matrix_angle.T)
        
        # Angular gradient
        cos_term = np.cos(ang * self.angle_k + self.angle_b)
        dQ_dcos_term = np.zeros_like(cos_term)
        for i in range(self.n_angle):
            mask = np.ones(self.n_angle, dtype=bool)
            mask[i] = False
            prod_except_i = np.prod(cos_term[:, :, mask], axis=-1)
            dQ_dcos_term[:, :, i] = prod_except_i * dQ_dka
        
        # BW gradients - OPTIMIZED with direct indexing
        dQ_dbw_dom_all = np.zeros_like(bw_dom_all_reshaped)
        for i in range(self.n_res):
            mask = np.ones(self.n_res, dtype=bool)
            mask[i] = False
            prod_except_i = np.prod(bw_dom_all_reshaped[:, :, mask], axis=-1)
            dQ_dbw_dom_all[:, :, i] = dQ_dbw_p * prod_except_i
        
        dQ_dbw_dom_all_flat = dQ_dbw_dom_all.reshape(n_events, -1)
        
        # m0 gradient - SCATTER using composed indices
        grad_m0 = np.zeros_like(m0)
        d_bw_dom_dm0_direct = 2 * m0_direct - 1j * g_bw_direct
        
        # Accumulate gradients using composed indices
        np.add.at(grad_m0, self.m0_composed, 
                  np.sum(np.real(dQ_dbw_dom_all_flat * d_bw_dom_dm0_direct), axis=0))
        
        # g_bw gradient - OPTIMIZED with matrix_gamma_direct
        # dQ_dg = dot(dQ_dbw_dom * (-1j * m0), matrix_gamma.T)
        # We use matrix_gamma_direct.T instead for direct gradient
        dQ_dg = np.dot(np.real(dQ_dbw_dom_all_flat * (-1j * m0_direct)), self.matrix_gamma_direct.T)
        
        # g0 gradient
        grad_g0 = np.zeros_like(g0)
        for gamma_idx in range(len(self.g0_index)):
            g0_param_idx = self.g0_index[gamma_idx]
            grad_g0[g0_param_idx] += np.sum(np.real(dQ_dg[:, gamma_idx] * g_interp[:, gamma_idx]))
        
        # Time gradients (merged)
        time_half = time / 2
        time_quarter = time / 4
        
        d_pb_dgp = 2 * np.real(np.conj(pap) * ap)
        d_pb_dgm = 2 * np.real(np.conj(pap) * poq * am)
        d_pbbar_dgp = 2 * np.real(np.conj(pam) * am)
        d_pbbar_dgm = 2 * np.real(np.conj(pam) * ap / poq)
        
        grad_common_gp = dQ_dpb * d_pb_dgp + dQ_dpbbar * d_pbbar_dgp
        grad_common_gm = dQ_dpb * d_pb_dgm + dQ_dpbbar * d_pbbar_dgm
        
        dgp_dGamma = -time_half * gp
        dgm_dGamma = -time_half * gm
        
        dgp_dDeltaGamma = -time_quarter * gm
        dgm_dDeltaGamma = -time_quarter * gp
        
        dgp_dDeltaM = -1j * time_half * gm
        dgm_dDeltaM = -1j * time_half * gp
        
        dQ_dGamma = np.sum(grad_common_gp * dgp_dGamma + grad_common_gm * dgm_dGamma)
        dQ_dDeltaGamma = np.sum(grad_common_gp * dgp_dDeltaGamma + grad_common_gm * dgm_dDeltaGamma)
        dQ_dDeltaM = np.sum(grad_common_gp * dgp_dDeltaM + grad_common_gm * dgm_dDeltaM)
        
        # poq gradients (merged)
        conj_pap_gm_am = np.conj(pap) * gm * am
        conj_pam = np.conj(pam)
        exp_phi = np.exp(1j * pop_phi)
        
        d_pb_dpoq_rho = 2 * np.real(conj_pap_gm_am * exp_phi)
        d_pb_dpop_phi = 2 * np.real(conj_pap_gm_am * poq_rho * 1j * exp_phi)
        
        conj_pam_gm_ap = conj_pam * gm * ap
        conj_pam_gp_am = conj_pam * gp * am
        
        d_pbbar_dpoq_rho = 2 * np.real(-conj_pam_gm_ap / (poq_rho**2))
        d_pbbar_dpop_phi = 2 * np.real(
            (-conj_pam_gm_ap * 1j / poq_rho + conj_pam_gp_am * 1j) * poq_rho * exp_phi
        )
        
        dQ_dpoq_rho = np.sum(dQ_dpb * d_pb_dpoq_rho + dQ_dpbbar * d_pbbar_dpoq_rho)
        dQ_dpop_phi = np.sum(dQ_dpb * d_pb_dpop_phi + dQ_dpbbar * d_pbbar_dpop_phi)
        
        # norm gradient
        if norm is None:
            grad_norm = None
        else:
            grad_norm = np.sum(weight / (P / norm + bkg) * (P / norm**2))
        
        grads = {
            "ck": grad_ck,
            "m0": grad_m0,
            "g0": grad_g0,
            "scalar": (dQ_dGamma, dQ_dDeltaGamma, dQ_dDeltaM, dQ_dAp, dQ_dpoq_rho, dQ_dpop_phi),
            "norm": grad_norm,
        }
        
        return Q, grads, P

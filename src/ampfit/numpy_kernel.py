"""
CORRECT gradient computation using proper Wirtinger calculus.

Key insight: For real-valued Q with complex variables, we need BOTH:
1. ∂Q/∂z (Wirtinger derivative w.r.t. z)
2. ∂Q/∂z* (Wirtinger derivative w.r.t. conjugate)

Then for REAL parameters like m0, Γ: ∂Q/∂x = 2*Re(∂Q/∂z * ∂z/∂x)
And for COMPLEX parameters like ck: gradient descent uses ∂Q/∂z*
"""
import numpy as np


class NumpyKernelCorrect:
    """Gradient computation with correct Wirtinger calculus throughout"""
    
    def __init__(self, config):
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

        self.n_basis = self.angle_k.shape[0]
        self.n_angle = self.angle_k.shape[1]
        self.n_wave = self.matrix_angle.shape[1]
        self.n_res = self.bw_order.size // self.n_wave
        self.n_decay = self.fl_order.size // self.n_wave

        # Precompute scatter matrices for gradient accumulation
        n_unique_bw = len(self.m0_index)
        n_wave_n_res = self.bw_order.size
        self._bw_scatter = np.zeros((n_wave_n_res, n_unique_bw), dtype=np.float64)
        for k, bw_idx in enumerate(self.bw_order):
            self._bw_scatter[k, bw_idx] = 1.0

        n_m0_unique = int(np.max(self.m0_index)) + 1
        self._m0_scatter = np.zeros((n_unique_bw, n_m0_unique), dtype=np.float64)
        for i, m_idx in enumerate(self.m0_index):
            self._m0_scatter[i, m_idx] = 1.0

        n_gamma_rows = len(self.g0_index)
        n_g0_unique = int(np.max(self.g0_index)) + 1
        self._g0_scatter = np.zeros((n_gamma_rows, n_g0_unique), dtype=np.float64)
        for i, g_idx in enumerate(self.g0_index):
            self._g0_scatter[i, g_idx] = 1.0

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
        """Compute forward and gradients with correct complex calculus"""
        
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
        
        # ==================== FORWARD PASS ====================
        
        # BW propagators
        g0_all = np.take(g0, self.g0_index)
        g0_m = np.take(mass, self.g0_mass_index, axis=-1)
        g_interp = self.interp(self.gamma_table, self.g0_index, g0_m, 
                               self.gamma_min, self.gamma_delta)
        g = g0_all * g_interp
        g_bw = np.dot(g, self.matrix_gamma)
        
        m0_all = np.take(m0, self.m0_index)
        m0_m = np.take(mass, self.mass_index, axis=-1)
        bw_dom = m0_all**2 - m0_m**2 - 1j * m0_all * g_bw
        
        bw_dom_all = np.take(bw_dom, self.bw_order, axis=-1)
        n_events = bw_dom_all.shape[0]
        bw_dom_all_reshaped = bw_dom_all.reshape(n_events, self.n_wave, self.n_res)
        bw_p = np.prod(bw_dom_all_reshaped, axis=-1)
        
        # FL factors
        fl_q = np.take(momentum, self.fl_q_index, axis=-1)
        fl = self.interp(self.fl_table, self.fl_type, fl_q, 
                        self.fl_min, self.fl_delta)
        fl_all = np.take(fl, self.fl_order, axis=-1)
        fl_p = np.prod(fl_all.reshape(-1, self.n_wave, self.n_decay), axis=-1)
        
        # Angular factors
        ang = np.take(angle, self.angle_index, axis=-2)
        ka = np.prod(np.cos(ang * self.angle_k + self.angle_b), axis=-1)
        fa = np.dot(ka, self.matrix_angle)
        
        # Amplitude
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
            # dQ/dP = -w / (P/norm + bkg) * (1/norm) = -w / (P + bkg*norm)
            dQ_dP = -weight / (P + bkg * norm)
        
        # ==================== BACKWARD PASS ====================
        # Use Wirtinger calculus consistently
        
        # Probability gradients (REAL)
        dP_dpb = frac * (1 - A_p)
        dP_dpbbar = (1 - frac) * (1 + A_p)
        dP_dAp = -frac * pb + (1 - frac) * pbbar
        dQ_dAp = np.sum(dQ_dP * dP_dAp)
        
        dQ_dpb = dQ_dP * dP_dpb
        dQ_dpbbar = dQ_dP * dP_dpbbar
        
        # ==================== GRADIENTS FOR COMPLEX AMPLITUDES ====================
        # For pb = |pap|² = pap * pap*, the Wirtinger derivatives are:
        # ∂pb/∂pap = pap* (derivative w.r.t. pap)
        # ∂pb/∂pap* = pap (derivative w.r.t. conjugate)
        #
        # For chain rule through pap = gp*ap + gm*poq*am:
        # ∂pap/∂ap = gp, ∂pap/∂ap* = 0
        # ∂pap*/∂ap = 0, ∂pap*/∂ap* = gp*
        #
        # Therefore:
        # ∂pb/∂ap = ∂pb/∂pap * ∂pap/∂ap + ∂pb/∂pap* * ∂pap*/∂ap
        #          = pap* * gp + pap * 0 = pap* * gp ✓
        # ∂pb/∂ap* = pap * gp* ✓
        
        # Wirtinger gradients (COMPLEX):
        d_pb_dap = np.conj(pap) * gp
        d_pb_dam = np.conj(pap) * gm * poq
        d_pbbar_dap = np.conj(pam) * (gm / poq)
        d_pbbar_dam = np.conj(pam) * gp
        
        # Complex gradients for ap and am:
        dQ_dap_Wirtinger = dQ_dpb * d_pb_dap + dQ_dpbbar * d_pbbar_dap
        dQ_dam_Wirtinger = dQ_dpb * d_pb_dam + dQ_dpbbar * d_pbbar_dam
        
        # ==================== BACKPROP THROUGH SUM ====================
        # ap = sum(a[:, 0, :]), am = sum(a[:, 1, :])
        # For Wirtinger calculus: ∂Q/∂a = ∂Q/∂ap (broadcasted)
        
        dQ_da_Wirtinger = np.zeros_like(a, dtype=complex)
        dQ_da_Wirtinger_reshaped = dQ_da_Wirtinger.reshape(-1, 2, self.n_wave // 2)
        
        # Assign Wirtinger gradients
        dQ_da_Wirtinger_reshaped[:, 0, :] = dQ_dap_Wirtinger[:, np.newaxis]
        dQ_da_Wirtinger_reshaped[:, 1, :] = dQ_dam_Wirtinger[:, np.newaxis]
        dQ_da_flat = dQ_da_Wirtinger
        
        # ==================== GRADIENT FOR ck (COMPLEX PARAMETER) ====================
        # a = ck * common_amp_factor
        # ∂a/∂ck = common_amp_factor (holomorphic)
        # ∂a/∂ck* = 0
        
        # ∂Q/∂ck = ∂Q/∂a * common_amp_factor
        # ∂Q/∂ck* = ∂Q/∂a* * common_amp_factor* = conj(∂Q/∂ck) (for real Q)
        
        grad_ck = np.sum(dQ_da_flat * common_amp_factor, axis=0)
        
        # For numerical gradient comparison:
        # ∂Q/∂Re(ck) = 2*Re(∂Q/∂ck)
        # ∂Q/∂Im(ck) = -2*Im(∂Q/∂ck)
        
        # ==================== GRADIENT FOR bw_p (COMPLEX) ====================
        # a = ck * (1/bw_p) * constant
        # ∂a/∂bw_p = -ck / bw_p² * constant = -ck * one_over_bw * common_amp_factor

        dQ_dbw_p = dQ_da_flat * (-ck * one_over_bw * common_amp_factor)

        # BW gradients through product (vectorized for n_res=2)
        dQ_dbw_dom_all = np.stack([
            dQ_dbw_p * bw_dom_all_reshaped[:, :, 1],
            dQ_dbw_p * bw_dom_all_reshaped[:, :, 0],
        ], axis=-1)

        # Scatter gradients via MatMul with scatter matrix
        dQ_dbw_flat = dQ_dbw_dom_all.reshape(n_events, -1)
        dQ_dbw_dom = dQ_dbw_flat @ self._bw_scatter

        # ==================== GRADIENT FOR m0 (REAL PARAMETER) ====================
        # bw_dom = m0² - m² - 1j*m0*g_bw
        # ∂bw_dom/∂m0 = 2*m0 - 1j*g_bw (complex derivative)

        dbw_dom_dm0 = 2 * m0_all - 1j * g_bw  # shape: (n_events, n_unique_bw)

        # Scatter by m0_index via MatMul
        dm0_raw = 2 * np.real(dQ_dbw_dom * dbw_dom_dm0)  # (n_events, n_unique_bw)
        dm0_sum = np.sum(dm0_raw, axis=0)                  # (n_unique_bw,)
        grad_m0 = dm0_sum @ self._m0_scatter               # (n_m0_unique,)

        # ==================== GRADIENT FOR g0 (REAL PARAMETER) ====================
        # First, compute ∂Q/∂g_bw:  ∂bw_dom/∂g_bw = -1j*m0
        dQ_dg_bw = dQ_dbw_dom * (-1j * m0_all)

        # Then ∂Q/∂g = ∂Q/∂g_bw @ matrix_gamma.T
        dQ_dg = dQ_dg_bw @ self.matrix_gamma.T

        # Scatter by g0_index via MatMul
        dg0_raw = 2 * np.real(dQ_dg * g_interp)  # (n_events, n_gamma_rows)
        dg0_sum = np.sum(dg0_raw, axis=0)          # (n_gamma_rows,)
        grad_g0 = dg0_sum @ self._g0_scatter       # (n_g0_unique,)
        
        # ==================== GRADIENTS FOR TIME EVOLUTION PARAMETERS ====================
        # Gamma, Delta_Gamma, Delta_m are REAL parameters
        # They affect gp and gm which are complex
        #
        # Using Wirtinger calculus for REAL parameters:
        # ∂Q/∂Γ = 2*Re(∂Q/∂gp * ∂gp/∂Γ + ∂Q/∂gm * ∂gm/∂Γ)
        
        # Wirtinger gradients for gp and gm:
        # For pb = |pap|² with pap = gp*ap + gm*poq*am:
        # ∂pb/∂gp = ∂pb/∂pap* * ∂pap*/∂gp = pap * ap*
        # ∂pb/∂gp* = pap* * ap
        
        d_pb_dgp = np.conj(pap) * np.conj(ap)
        d_pb_dgm = np.conj(pap) * np.conj(poq) * np.conj(am)
        d_pbbar_dgp = np.conj(pam) * np.conj(am)
        d_pbbar_dgm = np.conj(pam) * np.conj(ap) / np.conj(poq)
        
        # Wait, let me recalculate properly:
        # pap = gp*ap + gm*poq*am
        # ∂pap/∂gp = ap, ∂pap*/∂gp = 0
        # ∂pb/∂gp = ∂pb/∂pap * ∂pap/∂gp + ∂pb/∂pap* * ∂pap*/∂gp
        #         = pap* * ap + pap * 0 = pap* * ap
        
        d_pb_dgp = np.conj(pap) * ap
        d_pb_dgm = np.conj(pap) * poq * am
        d_pbbar_dgp = np.conj(pam) * am
        d_pbbar_dgm = np.conj(pam) * ap / poq
        
        # Complex gradients for gp and gm:
        dQ_dgp = dQ_dpb * d_pb_dgp + dQ_dpbbar * d_pbbar_dgp
        dQ_dgm = dQ_dpb * d_pb_dgm + dQ_dpbbar * d_pbbar_dgm
        
        # Time derivatives (complex):
        # eL = exp(-i*t*(-Δm/2 - i*(Γ + ΔΓ/2)/2))
        # ∂eL/∂Δm = eL * (-i*t) * (-1/2) = i*t/2 * eL
        # eH = exp(-i*t*(+Δm/2 - i*(Γ - ΔΓ/2)/2))
        # ∂eH/∂Δm = eH * (-i*t) * (+1/2) = -i*t/2 * eH
        # ∂gp/∂Δm = (∂eL/∂Δm + ∂eH/∂Δm)/2 = (i*t/2*eL - i*t/2*eH)/2 = i*t/4*(eL-eH) = i*t/2*gm
        # ∂gm/∂Δm = (∂eL/∂Δm - ∂eH/∂Δm)/2 = (i*t/2*eL + i*t/2*eH)/2 = i*t/4*(eL+eH) = i*t/2*gp
        
        dgp_dGamma = -time/2 * gp
        dgm_dGamma = -time/2 * gm
        
        dgp_dDeltaGamma = -time/4 * gm
        dgm_dDeltaGamma = -time/4 * gp
        
        dgp_dDeltaM = 1j * time/2 * gm  # FIXED: was -1j * time/2 * gm
        dgm_dDeltaM = 1j * time/2 * gp  # FIXED: was -1j * time/2 * gp
        
        # Gradient for REAL parameters using Wirtinger:
        dQ_dGamma = np.sum(2 * np.real(dQ_dgp * dgp_dGamma + dQ_dgm * dgm_dGamma))
        dQ_dDeltaGamma = np.sum(2 * np.real(dQ_dgp * dgp_dDeltaGamma + dQ_dgm * dgm_dDeltaGamma))
        dQ_dDeltaM = np.sum(2 * np.real(dQ_dgp * dgp_dDeltaM + dQ_dgm * dgm_dDeltaM))
        
        # ==================== GRADIENTS FOR poq (COMPLEX IN POLAR FORM) ====================
        # poq = poq_rho * exp(i*pop_phi), both parameters are REAL
        #
        # For pap = gp*ap + gm*poq*am:
        # ∂pap/∂poq = gm*am
        # ∂pb/∂poq = ∂pb/∂pap* * ∂pap*/∂poq = pap * gm* * am*
        
        # Wait, let me recalculate properly:
        # pap = gp*ap + gm*poq*am
        # ∂pap/∂poq = gm*am, ∂pap*/∂poq = 0
        # ∂pb/∂poq = pap* * gm * am
        
        d_pb_dpoq = np.conj(pap) * gm * am
        
        # For pam = (gm/poq)*ap + gp*am:
        # ∂pam/∂poq = -gm/poq² * ap
        # ∂pbbar/∂poq = pam* * (-gm/poq²) * ap
        
        d_pbbar_dpoq = np.conj(pam) * (-gm / poq**2) * ap
        
        # Complex gradient for poq:
        dQ_dpoq = dQ_dpb * d_pb_dpoq + dQ_dpbbar * d_pbbar_dpoq
        
        # Gradient for REAL parameters using Wirtinger:
        # ∂Q/∂poq_rho = 2*Re(∂Q/∂poq * ∂poq/∂poq_rho)
        # where ∂poq/∂poq_rho = exp(i*pop_phi)
        
        exp_phi = np.exp(1j * pop_phi)
        dQ_dpoq_rho = np.sum(2 * np.real(dQ_dpoq * exp_phi))
        
        # ∂Q/∂pop_phi = 2*Re(∂Q/∂poq * ∂poq/∂pop_phi)
        # where ∂poq/∂pop_phi = poq_rho * i * exp(i*pop_phi)
        
        dQ_dpop_phi = np.sum(2 * np.real(dQ_dpoq * poq_rho * 1j * exp_phi))
        
        # ==================== RETURN GRADIENTS ====================
        grads = {
            "ck": grad_ck,
            "m0": grad_m0,
            "g0": grad_g0,
            "scalar": (dQ_dGamma, dQ_dDeltaGamma, dQ_dDeltaM, dQ_dAp, dQ_dpoq_rho, dQ_dpop_phi),
            "norm": None if norm is None else np.sum(weight / (P / norm + bkg) * (P / norm**2)),
        }
        
        return Q, grads, P

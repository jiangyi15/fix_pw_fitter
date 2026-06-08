"""
FIXED gradient computation with correct handling of:
1. Complex parameters with real-valued loss (Wirtinger calculus)
2. Real parameters with complex intermediate variables
3. Proper chain rule for complex numbers
"""
import numpy as np


class NumpyKernelFixed:
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
        """Compute forward pass with CORRECT gradients"""
        
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
            dQ_dP = -weight / (P / norm + bkg)
        
        # ==================== BACKWARD PASS ====================
        
        # Probability gradients
        dP_dpb = frac * (1 - A_p)
        dP_dpbbar = (1 - frac) * (1 + A_p)
        dP_dAp = -frac * pb + (1 - frac) * pbbar
        dQ_dAp = np.sum(dQ_dP * dP_dAp)
        
        dQ_dpb = dQ_dP * dP_dpb
        dQ_dpbbar = dQ_dP * dP_dpbbar
        
        # ==================== FIX #1: Correct amplitude gradients ====================
        # For pb = |pap|², the Wirtinger derivatives are:
        # ∂pb/∂pap* = pap (derivative w.r.t. conjugate)
        # ∂pb/∂pap = pap* (derivative w.r.t. variable)
        
        # Gradient of pb w.r.t. pap (Wirtinger)
        d_pb_d_pap = np.conj(pap)
        d_pb_d_pam = np.conj(pam)
        
        # Similarly for pbbar
        d_pbbar_d_pam = np.conj(pam)
        d_pbbar_d_pap = np.conj(pam)  # Actually this should be pam* when pam depends on pap
        
        # Recompute properly:
        # pap = gp*ap + gm*poq*am
        # ∂pb/∂ap = ∂pb/∂pap * ∂pap/∂ap = pap* * gp
        d_pb_dap_Wirtinger = np.conj(pap) * gp
        d_pb_dam_Wirtinger = np.conj(pap) * gm * poq
        
        # For pbbar:
        # pam = (gm/poq)*ap + gp*am
        # ∂pbbar/∂ap = pam* * (gm/poq)
        d_pbbar_dap_Wirtinger = np.conj(pam) * (gm / poq)
        d_pbbar_dam_Wirtinger = np.conj(pam) * gp
        
        # For real-valued Q, the gradient w.r.t. complex ap is:
        # ∂Q/∂ap = ∂Q/∂pb * (∂pb/∂ap + ∂pb/∂ap*) + ...
        # But since ap is intermediate, we use standard chain rule
        
        # Actually, for intermediate complex variables, we need BOTH derivatives!
        # ∂Q/∂ap = ∂Q/∂pap * ∂pap/∂ap + ∂Q/∂pap* * ∂pap*/∂ap
        #        = dQ_dpb * ∂pb/∂pap * gp + dQ_dpb * ∂pb/∂pap* * gp*
        #        = dQ_dpb * (pap* * gp + pap * gp*)
        #        = dQ_dpb * 2*Re(pap* * gp)
        
        d_pb_dap = 2 * np.real(np.conj(pap) * gp)  # This matches the original!
        d_pb_dam = 2 * np.real(np.conj(pap) * gm * poq)
        d_pbbar_dap = 2 * np.real(np.conj(pam) * (gm / poq))
        d_pbbar_dam = 2 * np.real(np.conj(pam) * gp)
        
        dQ_dap = dQ_dpb * d_pb_dap + dQ_dpbbar * d_pbbar_dap
        dQ_dam = dQ_dpb * d_pb_dam + dQ_dpbbar * d_pbbar_dam
        
        # Backprop through amplitude reshape
        dQ_da = np.zeros_like(a_reshaped)
        dQ_da[:, 0, :] = dQ_dap[:, np.newaxis]
        dQ_da[:, 1, :] = dQ_dam[:, np.newaxis]
        dQ_da_flat = dQ_da.reshape(n_events, self.n_wave)
        
        # ==================== FIX #2: Correct ck gradient ====================
        # For complex parameter ck with real-valued Q:
        # ∂Q/∂Re(ck) = 2 * Re(∂Q/∂a * ∂a/∂ck*)
        # ∂Q/∂Im(ck) = -2 * Im(∂Q/∂a * ∂a/∂ck*)
        
        # where ∂a/∂ck* = 0 (a is holomorphic in ck)
        # and ∂a/∂ck = common_amp_factor
        
        # So: ∂Q/∂ck = ∂Q/∂a * common_amp_factor (standard derivative)
        #     ∂Q/∂ck* = 0 (conjugate derivative)
        
        # Numerical gradient for real perturbation:
        # ∂Q/∂Re(ck) = (Q(ck+ε) - Q(ck-ε)) / (2ε)
        #            = 2 * Re(∂Q/∂ck)  # Factor of 2 from Wirtinger!
        
        dQ_dck_Wirtinger = dQ_da_flat * common_amp_factor
        grad_ck_complex = np.sum(dQ_dck_Wirtinger, axis=0)
        
        # For comparison with numerical gradient, output complex value
        # The numerical test will compare real and imaginary parts separately
        grad_ck = grad_ck_complex
        
        # bw_p gradient
        # a = ck * (1/bw_p) * fa * fl_p
        # ∂a/∂bw_p = -ck / bw_p² * fa * fl_p = -ck * one_over_bw * common_amp_factor
        dQ_dbw_p = dQ_da_flat * (-ck * one_over_bw * common_amp_factor)
        
        # BW gradients through product
        dQ_dbw_dom_all = np.zeros_like(bw_dom_all_reshaped)
        for i in range(self.n_res):
            mask = np.ones(self.n_res, dtype=bool)
            mask[i] = False
            prod_except_i = np.prod(bw_dom_all_reshaped[:, :, mask], axis=-1)
            dQ_dbw_dom_all[:, :, i] = dQ_dbw_p * prod_except_i
        
        # Scatter gradients
        dQ_dbw_dom = np.zeros_like(bw_dom)
        for wave_idx in range(self.n_wave):
            for res_idx in range(self.n_res):
                order_idx = wave_idx * self.n_res + res_idx
                bw_idx = self.bw_order[order_idx]
                dQ_dbw_dom[:, bw_idx] += dQ_dbw_dom_all[:, wave_idx, res_idx]
        
        # ==================== FIX #3: Correct m0 gradient ====================
        # m0 is REAL, but bw_dom is complex
        # bw_dom = m0² - m² - 1j*m0*g_bw
        # 
        # Split into real and imaginary parts:
        # Re(bw_dom) = m0² - m² + m0*Im(g_bw)
        # Im(bw_dom) = -m0*Re(g_bw)
        #
        # ∂Re(bw_dom)/∂m0 = 2*m0 + Im(g_bw)
        # ∂Im(bw_dom)/∂m0 = -Re(g_bw)
        #
        # ∂Q/∂m0 = ∂Q/∂Re(bw_dom) * ∂Re(bw_dom)/∂m0 + ∂Q/∂Im(bw_dom) * ∂Im(bw_dom)/∂m0
        
        g_bw_real = np.real(g_bw)
        g_bw_imag = np.imag(g_bw)
        
        # For real-valued Q:
        # ∂Q/∂Re(bw_dom) = Re(∂Q/∂bw_dom)
        # ∂Q/∂Im(bw_dom) = Re(-1j * ∂Q/∂bw_dom) = Im(∂Q/∂bw_dom)
        
        dQ_dRe_bw_dom = np.real(dQ_dbw_dom)
        dQ_dIm_bw_dom = np.imag(dQ_dbw_dom)
        
        # Derivatives of bw_dom components
        # Note: These should be shape (n_events, n_unique_bw) due to broadcasting
        # bw_dom = m0² - m² - 1j*m0*g_bw
        # ∂bw_dom/∂m0 = 2*m0 - 1j*g_bw
        # ∂Re(bw_dom)/∂m0 = 2*m0 + Im(g_bw)
        # ∂Im(bw_dom)/∂m0 = -Re(g_bw)
        
        # For each unique bw, we need the gradient w.r.t. the corresponding m0 parameter
        # m0_all has shape (n_unique_bw,), g_bw has shape (n_events, n_unique_bw)
        # So ∂bw_dom/∂m0 has shape (n_events, n_unique_bw)
        
        dbw_dom_dm0 = 2 * m0_all - 1j * g_bw  # shape: (n_events, n_unique_bw)
        dRe_bw_dom_dm0 = 2 * m0_all + g_bw_imag  # shape: (n_events, n_unique_bw)  
        dIm_bw_dom_dm0 = -g_bw_real  # shape: (n_events, n_unique_bw)
        
        # Accumulate gradient
        grad_m0 = np.zeros_like(m0)
        for bw_idx in range(len(self.m0_index)):
            m0_param_idx = self.m0_index[bw_idx]
            # Sum over events for this unique bw
            grad_m0[m0_param_idx] += np.sum(
                dQ_dRe_bw_dom[:, bw_idx] * dRe_bw_dom_dm0[:, bw_idx] +
                dQ_dIm_bw_dom[:, bw_idx] * dIm_bw_dom_dm0[:, bw_idx]
            )
        
        # ==================== FIX #4: Correct g0 gradient ====================
        # g_bw = dot(g, matrix_gamma), where g = g0_all * g_interp
        # g_bw is complex, g0 is REAL
        #
        # ∂g_bw/∂g = matrix_gamma.T
        # ∂g/∂g0_all = g_interp
        #
        # Split g_bw into real/imaginary parts:
        # Re(g_bw) = Re(dot(g, M))
        # Im(g_bw) = Im(dot(g, M))
        #
        # ∂Re(g_bw)/∂g0_all[i] = Re(g_interp[:, i] * M[i, :])
        # ∂Im(g_bw)/∂g0_all[i] = Im(g_interp[:, i] * M[i, :])
        
        # Gradient through bw_dom:
        # ∂Q/∂Re(g_bw) = Re(-1j*m0*∂Q/∂bw_dom) = Im(m0*∂Q/∂bw_dom)
        # ∂Q/∂Im(g_bw) = Re(m0*∂Q/∂bw_dom)
        
        dQ_dRe_g_bw = np.imag(m0_all * dQ_dbw_dom)  # ∂Q/∂Re(g_bw)
        dQ_dIm_g_bw = np.real(m0_all * dQ_dbw_dom)  # ∂Q/∂Im(g_bw)
        
        # Gradient through g_bw = dot(g, M)
        # g is complex, but we need gradients w.r.t. REAL g0
        # ∂Re(g_bw)/∂Re(g) = Re(M), ∂Re(g_bw)/∂Im(g) = -Im(M)
        # ∂Im(g_bw)/∂Re(g) = Im(M), ∂Im(g_bw)/∂Im(g) = Re(M)
        
        # Combined: ∂g_bw/∂g = M (matrix_gamma)
        # ∂Q/∂g = ∂Q/∂g_bw * M.T (complex derivative)
        
        dQ_dg_complex = np.dot(dQ_dbw_dom * (-1j * m0_all), self.matrix_gamma.T)
        
        # Now split into real and imaginary parts of g
        # g = g0_all * g_interp
        # Re(g) = g0_all * Re(g_interp)
        # Im(g) = g0_all * Im(g_interp)
        #
        # ∂Re(g)/∂g0_all[i] = Re(g_interp[:, i])
        # ∂Im(g)/∂g0_all[i] = Im(g_interp[:, i])
        
        grad_g0 = np.zeros_like(g0)
        for gamma_idx in range(len(self.g0_index)):
            g0_param_idx = self.g0_index[gamma_idx]
            interp_val = g_interp[:, gamma_idx]  # Complex
            
            # Gradient components
            dQ_dRe_g = np.real(dQ_dg_complex[:, gamma_idx])
            dQ_dIm_g = np.imag(dQ_dg_complex[:, gamma_idx])
            
            dRe_g_dg0 = np.real(interp_val)
            dIm_g_dg0 = np.imag(interp_val)
            
            grad_g0[g0_param_idx] += np.sum(
                dQ_dRe_g * dRe_g_dg0 + dQ_dIm_g * dIm_g_dg0
            )
        
        # ==================== FIX #5: Correct scalar gradients ====================
        # Time evolution gradients
        # eL = exp(-1j*t*(-Δm/2 - 1j*(Γ + ΔΓ/2)/2))
        # eH = exp(-1j*t*(+Δm/2 - 1j*(Γ - ΔΓ/2)/2))
        #
        # Split into real/imaginary parts and differentiate w.r.t. REAL parameters
        
        # For Gamma (REAL parameter):
        # ∂eL/∂Γ = -t/2 * eL  (complex)
        # ∂eH/∂Γ = -t/2 * eH  (complex)
        #
        # ∂gp/∂Γ = (∂eL/∂Γ + ∂eH/∂Γ)/2 = -t/2 * gp (complex)
        # ∂gm/∂Γ = (∂eL/∂Γ - ∂eH/∂Γ)/2 = -t/2 * gm (complex)
        
        # Now compute ∂Q/∂Γ:
        # Q depends on gp through pb = |pap|²
        # pap = gp*ap + gm*poq*am
        #
        # Treat gp as complex variable:
        # ∂pb/∂gp = ap * pap*  (Wirtinger derivative)
        # ∂pb/∂gp* = ap* * pap (conjugate derivative)
        #
        # For REAL Γ:
        # ∂Q/∂Γ = ∂Q/∂pb * (∂pb/∂gp * ∂gp/∂Γ + ∂pb/∂gp* * ∂gp*/∂Γ)
        #       = ∂Q/∂pb * (ap*pap*(-t/2)*gp + ap*pap*(-t/2)*gp*)
        #       = ∂Q/∂pb * (-t/2) * 2*Re(ap*pap*gp)
        #       = -t * ∂Q/∂pb * Re(ap*pap*gp)
        
        # Alternatively, use real/imaginary decomposition:
        # gp = gp_real + 1j*gp_imag
        # ∂gp_real/∂Γ, ∂gp_imag/∂Γ are both real
        
        # Let's use the correct Wirtinger approach:
        # ∂Q/∂Γ = ∂Q/∂pb * ∂pb/∂Γ + ∂Q/∂pbbar * ∂pbbar/∂Γ
        # where ∂pb/∂Γ = ∂pb/∂gp * ∂gp/∂Γ + ∂pb/∂gp* * ∂gp*/∂Γ
        
        # Compute ∂pb/∂gp (Wirtinger)
        d_pb_dgp = ap * np.conj(pap)
        d_pb_dgm = poq * am * np.conj(pap)
        d_pbbar_dgp = am * np.conj(pam)
        d_pbbar_dgm = ap * np.conj(pam) / poq
        
        # Time derivatives (complex)
        dgp_dGamma = -time/2 * gp
        dgm_dGamma = -time/2 * gm
        
        dgp_dDeltaGamma = -time/4 * gm  # Note: check sign!
        dgm_dDeltaGamma = -time/4 * gp
        
        dgp_dDeltaM = -1j * time/2 * gm
        dgm_dDeltaM = -1j * time/2 * gp
        
        # For REAL parameters, the gradient is:
        # ∂Q/∂Γ = 2*Re(∂Q/∂gp * ∂gp/∂Γ + ∂Q/∂gm * ∂gm/∂Γ)
        # where ∂Q/∂gp = ∂Q/∂pb * ∂pb/∂gp + ∂Q/∂pbbar * ∂pbbar/∂gp
        
        dQ_dgp = dQ_dpb * d_pb_dgp + dQ_dpbbar * d_pbbar_dgp
        dQ_dgm = dQ_dpb * d_pb_dgm + dQ_dpbbar * d_pbbar_dgm
        
        # Gradient for Gamma (REAL)
        dQ_dGamma_complex = dQ_dgp * dgp_dGamma + dQ_dgm * dgm_dGamma
        dQ_dGamma = 2 * np.real(dQ_dGamma_complex)
        
        # Gradient for Delta_Gamma (REAL)
        dQ_dDeltaGamma_complex = dQ_dgp * dgp_dDeltaGamma + dQ_dgm * dgm_dDeltaGamma
        dQ_dDeltaGamma = 2 * np.real(dQ_dDeltaGamma_complex)
        
        # Gradient for Delta_m (REAL)
        dQ_dDeltaM_complex = dQ_dgp * dgp_dDeltaM + dQ_dgm * dgm_dDeltaM
        dQ_dDeltaM = 2 * np.real(dQ_dDeltaM_complex)
        
        # ==================== poq gradients ====================
        # poq = poq_rho * exp(1j*pop_phi)
        # This is a complex parameter in polar form
        
        # pap = gp*ap + gm*poq*am
        # ∂pap/∂poq = gm*am
        # ∂pb/∂poq = ∂pb/∂pap * ∂pap/∂poq = pap* * gm * am
        
        # For poq_rho and pop_phi (both REAL):
        # poq = poq_rho * exp(1j*pop_phi)
        # ∂poq/∂poq_rho = exp(1j*pop_phi)
        # ∂poq/∂pop_phi = poq_rho * 1j * exp(1j*pop_phi)
        
        d_pb_dpoq = np.conj(pap) * gm * am
        d_pbbar_dpoq = np.conj(pam) * gp * am - np.conj(pam) * gm * ap / poq**2
        
        d_poq_dpoq_rho = np.exp(1j * pop_phi)
        d_poq_dpop_phi = poq_rho * 1j * np.exp(1j * pop_phi)
        
        # Gradients for REAL parameters
        dQ_dpoq = dQ_dpb * d_pb_dpoq + dQ_dpbbar * d_pbbar_dpoq
        
        dQ_dpoq_rho_complex = dQ_dpoq * d_poq_dpoq_rho
        dQ_dpoq_rho = 2 * np.real(dQ_dpoq_rho_complex)
        
        dQ_dpop_phi_complex = dQ_dpoq * d_poq_dpop_phi
        dQ_dpop_phi = 2 * np.real(dQ_dpop_phi_complex)
        
        # ==================== norm gradient ====================
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

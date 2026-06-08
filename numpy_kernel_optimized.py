"""
Optimized NumPy kernel with reused computations and vectorized operations.
Reference implementation: numpy_kernel.py
"""
import numpy as np


class NumpyKernelOptimized:
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
        
        # Precompute bw_order indices mapping for gradient scatter
        self._precompute_bw_order_mapping()
        # Precompute angle masks for gradient computation
        self._precompute_angle_masks()
        
    def _precompute_bw_order_mapping(self):
        """Precompute indices for efficient gradient scatter-add"""
        # Create mapping: (wave_idx, res_idx) -> list of bw_idx positions
        self.bw_order_wave_res_idx = []
        for wave_idx in range(self.n_wave):
            for res_idx in range(self.n_res):
                order_idx = wave_idx * self.n_res + res_idx
                self.bw_order_wave_res_idx.append(self.bw_order[order_idx])
        self.bw_order_wave_res_idx = np.array(self.bw_order_wave_res_idx)
        
    def _precompute_angle_masks(self):
        """Precompute boolean masks for angle gradient computation"""
        self.angle_masks = []
        for i in range(self.n_angle):
            mask = np.ones(self.n_angle, dtype=bool)
            mask[i] = False
            self.angle_masks.append(mask)
            
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

    def _forward(self, params, data):
        """
        Forward pass: compute all intermediate values.
        Returns cache dict for reuse in backward pass.
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
        
        # Cache dict to store all intermediate values
        cache = {}
        cache['ck'] = ck
        cache['m0'] = m0
        cache['g0'] = g0
        cache['Gamma'] = Gamma
        cache['Delta_Gamma'] = Delta_Gamma
        cache['Delta_m'] = Delta_m
        cache['A_p'] = A_p
        cache['poq_rho'] = poq_rho
        cache['pop_phi'] = pop_phi
        cache['frac'] = frac
        cache['time'] = time
        
        # ==================== BW propagators ====================
        # g = g0_all * interp(gamma_table, ...)
        g0_all = np.take(g0, self.g0_index)
        g0_m = np.take(mass, self.g0_mass_index, axis=-1)
        g_interp = self.interp(self.gamma_table, self.g0_index, g0_m, 
                               self.gamma_min, self.gamma_delta)
        g = g0_all * g_interp  # shape: (n_events, n_gamma)
        
        # g_bw = dot(g, matrix_gamma)
        g_bw = np.dot(g, self.matrix_gamma)  # shape: (n_events, n_unique_bw)
        
        # bw_dom = m0_all**2 - m0_m**2 - 1j * m0_all * g_bw
        m0_all = np.take(m0, self.m0_index)
        m0_m = np.take(mass, self.mass_index, axis=-1)
        bw_dom = m0_all**2 - m0_m**2 - 1j * m0_all * g_bw  # shape: (n_events, n_unique_bw)
        
        # bw_p = prod(bw_dom_all, axis=-1)
        bw_dom_all = np.take(bw_dom, self.bw_order, axis=-1)
        n_events = bw_dom_all.shape[0]
        bw_dom_all_reshaped = bw_dom_all.reshape(n_events, self.n_wave, self.n_res)
        bw_p = np.prod(bw_dom_all_reshaped, axis=-1)  # shape: (n_events, n_wave)
        
        cache['g'] = g
        cache['g_interp'] = g_interp
        cache['g_bw'] = g_bw
        cache['m0_all'] = m0_all
        cache['m0_m'] = m0_m
        cache['bw_dom'] = bw_dom
        cache['bw_dom_all_reshaped'] = bw_dom_all_reshaped
        cache['bw_p'] = bw_p
        cache['n_events'] = n_events
        
        # ==================== FL factors ====================
        fl_q = np.take(momentum, self.fl_q_index, axis=-1)
        fl = self.interp(self.fl_table, self.fl_type, fl_q, 
                        self.fl_min, self.fl_delta)
        fl_all = np.take(fl, self.fl_order)
        fl_all_reshaped = fl_all.reshape(-1, self.n_wave, self.n_decay)
        fl_p = np.prod(fl_all_reshaped, axis=-1)  # shape: (n_events, n_wave)
        
        cache['fl_all_reshaped'] = fl_all_reshaped
        cache['fl_p'] = fl_p
        
        # ==================== Angular factors ====================
        ang = np.take(angle, self.angle_index, axis=-2)
        cos_term = np.cos(ang * self.angle_k + self.angle_b)  # shape: (n_events, n_basis, n_angle)
        sin_term = np.sin(ang * self.angle_k + self.angle_b)
        ka = np.prod(cos_term, axis=-1)  # shape: (n_events, n_basis)
        fa = np.dot(ka, self.matrix_angle)  # shape: (n_events, n_wave)
        
        cache['cos_term'] = cos_term
        cache['sin_term'] = sin_term
        cache['ka'] = ka
        cache['fa'] = fa
        
        # ==================== Amplitudes ====================
        # a = ck * (1/bw_p) * fa * fl_p
        a = ck * (1.0 / bw_p) * fa * fl_p  # shape: (n_events, n_wave)
        a_reshaped = a.reshape(-1, 2, self.n_wave // 2)
        ap = np.sum(a_reshaped[:, 0, :], axis=-1)  # shape: (n_events,)
        am = np.sum(a_reshaped[:, 1, :], axis=-1)
        
        cache['a'] = a
        cache['a_reshaped'] = a_reshaped
        cache['ap'] = ap
        cache['am'] = am
        
        # ==================== Time evolution ====================
        eL = np.exp(-1j * time * (-Delta_m/2 - 1j * (Gamma + Delta_Gamma/2)/2))
        eH = np.exp(-1j * time * (+Delta_m/2 - 1j * (Gamma - Delta_Gamma/2)/2))
        gp = (eL + eH) / 2
        gm = (eL - eH) / 2
        
        cache['eL'] = eL
        cache['eH'] = eH
        cache['gp'] = gp
        cache['gm'] = gm
        
        # ==================== Probabilities ====================
        poq = poq_rho * np.exp(1j * pop_phi)
        pap = gp * ap + gm * poq * am
        pam = (gm / poq) * ap + gp * am
        
        pb = np.abs(pap)**2
        pbbar = np.abs(pam)**2
        
        cache['poq'] = poq
        cache['pap'] = pap
        cache['pam'] = pam
        cache['pb'] = pb
        cache['pbbar'] = pbbar
        
        # ==================== Total probability ====================
        P = frac * pb * (1 - A_p) + (1 - frac) * pbbar * (1 + A_p)
        cache['P'] = P
        
        return cache

    def _backward(self, cache, data, norm):
        """
        Backward pass: compute gradients using cached forward values.
        """
        weight = data["weight"]
        bkg = data["bkg"]
        n_events = cache['n_events']
        
        # ==================== Loss gradient ====================
        P = cache['P']
        if norm is None:
            Q = np.sum(weight * P)
            dQ_dP = weight
        else:
            Q = -np.sum(weight * np.log(P / norm + bkg))
            dQ_dP = -weight / (P / norm + bkg)
            
        cache['Q'] = Q
        
        # ==================== Probability gradients ====================
        frac = cache['frac']
        A_p = cache['A_p']
        pb = cache['pb']
        pbbar = cache['pbbar']
        
        dP_dpb = frac * (1 - A_p)
        dP_dpbbar = (1 - frac) * (1 + A_p)
        dP_dAp = -frac * pb + (1 - frac) * pbbar
        
        dQ_dAp = np.sum(dQ_dP * dP_dAp)
        
        # ==================== Amplitude gradients ====================
        gp = cache['gp']
        gm = cache['gm']
        ap = cache['ap']
        am = cache['am']
        poq = cache['poq']
        pap = cache['pap']
        pam = cache['pam']
        
        dQ_dpb_bar = dQ_dP * dP_dpb
        dQ_dpbbar_bar = dQ_dP * dP_dpbbar
        
        # d(pb)/d(ap, am)
        d_pb_dap = 2 * np.real(gp * np.conj(pap))
        d_pb_dam = 2 * np.real(gm * poq * np.conj(pap))
        
        # d(pbbar)/d(ap, am)
        d_pbbar_dap = 2 * np.real((gm / poq) * np.conj(pam))
        d_pbbar_dam = 2 * np.real(gp * np.conj(pam))
        
        # Chain rule
        dQ_dap = dQ_dpb_bar * d_pb_dap + dQ_dpbbar_bar * d_pbbar_dap
        dQ_dam = dQ_dpb_bar * d_pb_dam + dQ_dpbbar_bar * d_pbbar_dam
        
        # ==================== ck gradients ====================
        a = cache['a']
        a_reshaped = cache['a_reshaped']
        bw_p = cache['bw_p']
        fa = cache['fa']
        fl_p = cache['fl_p']
        ck = cache['ck']
        
        # Backprop through reshape
        dQ_da = np.zeros_like(a_reshaped)
        dQ_da[:, 0, :] = dQ_dap[:, np.newaxis]
        dQ_da[:, 1, :] = dQ_dam[:, np.newaxis]
        dQ_da_flat = dQ_da.reshape(n_events, self.n_wave)
        
        # Backprop through a = ck * (1/bw_p) * fa * fl_p
        dQ_dck = dQ_da_flat * (1.0 / bw_p) * fa * fl_p
        dQ_dbw_p = dQ_da_flat * ck * (-1.0 / bw_p**2) * fa * fl_p
        dQ_dfa = dQ_da_flat * ck * (1.0 / bw_p) * fl_p
        dQ_dfl_p = dQ_da_flat * ck * (1.0 / bw_p) * fa
        
        grad_ck = np.sum(dQ_dck, axis=0)
        
        # ==================== Angle gradients ====================
        ka = cache['ka']
        cos_term = cache['cos_term']
        sin_term = cache['sin_term']
        
        # fa = dot(ka, matrix_angle)
        dQ_dka = np.dot(dQ_dfa, self.matrix_angle.T)
        
        # ka = prod(cos_term, axis=-1) - vectorized gradient computation
        # Use cumprod trick: prod_except_i = total_prod / cos_term[:,:,i]
        # But we need to handle zeros, so use cumsum approach
        dQ_dcos_term = self._vectorized_prod_gradient(cos_term, dQ_dka)
        
        # d(cos_term)/d(ang) = -sin_term * angle_k
        dQ_dang = -sin_term * self.angle_k[np.newaxis, :, :] * dQ_dcos_term
        
        # ==================== BW gradients ====================
        bw_dom_all_reshaped = cache['bw_dom_all_reshaped']
        
        # bw_p = prod(bw_dom_all_reshaped, axis=-1) - vectorized gradient
        dQ_dbw_dom_all = self._vectorized_prod_gradient(bw_dom_all_reshaped, dQ_dbw_p)
        
        # Scatter gradients back to bw_dom using bw_order
        bw_dom = cache['bw_dom']
        dQ_dbw_dom = np.zeros_like(bw_dom)
        
        # Vectorized scatter-add using index array
        # Create event indices for each position
        dQ_dbw_dom_all_flat = dQ_dbw_dom_all.reshape(n_events, -1)
        event_indices = np.arange(n_events).reshape(-1, 1)
        
        # Use advanced indexing for scatter-add
        np.add.at(dQ_dbw_dom, (event_indices, self.bw_order_wave_res_idx[np.newaxis, :]), 
                  dQ_dbw_dom_all_flat)
        
        # ==================== m0 gradients ====================
        m0_all = cache['m0_all']
        g_bw = cache['g_bw']
        grad_m0 = np.zeros_like(cache['m0'])
        
        # bw_dom = m0_all**2 - m0_m**2 - 1j * m0_all * g_bw
        # d(bw_dom)/d(m0_all) = 2*m0_all - 1j*g_bw
        d_bw_dom_dm0 = 2 * m0_all - 1j * g_bw  # shape: (n_events, n_unique_bw)
        
        # Accumulate gradients for each m0 parameter
        np.add.at(grad_m0, self.m0_index, 
                  np.sum(np.real(dQ_dbw_dom * d_bw_dom_dm0), axis=0))
        
        # ==================== g0 gradients ====================
        # d(g_bw)/d(g) = matrix_gamma.T
        dQ_dg = np.dot(np.real(dQ_dbw_dom * (-1j * m0_all)), self.matrix_gamma.T)
        
        # g = g0_all * g_interp
        g_interp = cache['g_interp']
        grad_g0 = np.zeros_like(cache['g0'])
        
        # Accumulate gradients for each g0 parameter
        np.add.at(grad_g0, self.g0_index,
                  np.sum(np.real(dQ_dg * g_interp), axis=0))
        
        # ==================== Scalar gradients ====================
        time = cache['time']
        eL = cache['eL']
        eH = cache['eH']
        Delta_m = cache['Delta_m']
        Gamma = cache['Gamma']
        Delta_Gamma = cache['Delta_Gamma']
        poq_rho = cache['poq_rho']
        pop_phi = cache['pop_phi']
        
        # Derivatives of eL and eH
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
        
        # Derivatives of pb and pbbar w.r.t gp and gm
        d_pb_dgp = 2 * np.real(np.conj(pap) * ap)
        d_pb_dgm = 2 * np.real(np.conj(pap) * poq * am)
        
        d_pbbar_dgp = 2 * np.real(np.conj(pam) * am)
        d_pbbar_dgm = 2 * np.real(np.conj(pam) * ap / poq)
        
        # Gamma gradient
        dQ_dGamma = np.sum(
            dQ_dpb_bar * d_pb_dgp * dgp_dGamma +
            dQ_dpb_bar * d_pb_dgm * dgm_dGamma +
            dQ_dpbbar_bar * d_pbbar_dgp * dgp_dGamma +
            dQ_dpbbar_bar * d_pbbar_dgm * dgm_dGamma
        )
        
        # Delta_Gamma gradient
        dQ_dDeltaGamma = np.sum(
            dQ_dpb_bar * d_pb_dgp * dgp_dDeltaGamma +
            dQ_dpb_bar * d_pb_dgm * dgm_dDeltaGamma +
            dQ_dpbbar_bar * d_pbbar_dgp * dgp_dDeltaGamma +
            dQ_dpbbar_bar * d_pbbar_dgm * dgm_dDeltaGamma
        )
        
        # Delta_m gradient
        dQ_dDeltaM = np.sum(
            dQ_dpb_bar * d_pb_dgp * dgp_dDeltaM +
            dQ_dpb_bar * d_pb_dgm * dgm_dDeltaM +
            dQ_dpbbar_bar * d_pbbar_dgp * dgp_dDeltaM +
            dQ_dpbbar_bar * d_pbbar_dgm * dgm_dDeltaM
        )
        
        # poq_rho and pop_phi gradients
        d_pb_dpoq_rho = 2 * np.real(np.conj(pap) * gm * am * np.exp(1j * pop_phi))
        d_pb_dpop_phi = 2 * np.real(np.conj(pap) * gm * poq_rho * am * 1j * np.exp(1j * pop_phi))
        
        d_pbbar_dpoq_rho = 2 * np.real(np.conj(pam) * (-gm / (poq_rho**2) * ap))
        d_pbbar_dpop_phi = 2 * np.real(
            np.conj(pam) * (gm / poq_rho * ap * (-1j) + gp * am * 1j) * poq_rho * np.exp(1j * pop_phi)
        )
        
        dQ_dpoq_rho = np.sum(dQ_dpb_bar * d_pb_dpoq_rho + dQ_dpbbar_bar * d_pbbar_dpoq_rho)
        dQ_dpop_phi = np.sum(dQ_dpb_bar * d_pb_dpop_phi + dQ_dpbbar_bar * d_pbbar_dpop_phi)
        
        # ==================== Norm gradient ====================
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

    def _vectorized_prod_gradient(self, arr, grad_output):
        """
        Compute gradient of product operation efficiently.
        
        For prod(arr, axis=-1), compute d(prod)/d(arr[..., i]) efficiently.
        
        Args:
            arr: Input array of shape (..., n)
            grad_output: Gradient of output w.r.t. product, shape (...)
            
        Returns:
            Gradient w.r.t. arr, same shape as arr
        """
        n = arr.shape[-1]
        
        # Simple and efficient approach using division
        # d(prod)/d(arr[..., i]) = prod / arr[..., i]
        total_prod = np.prod(arr, axis=-1, keepdims=True)
        
        # Use safe division - where arr is zero, the gradient is zero for that element
        # but the product gradients for other elements are unaffected
        with np.errstate(divide='ignore', invalid='ignore'):
            grad = np.where(arr != 0, total_prod / arr, 0.0)
        
        # Multiply by output gradient
        grad = grad * grad_output[..., np.newaxis]
        
        return grad

    def _compute(self, params, data, norm=None):
        """
        Combined forward and backward pass.
        """
        # Forward pass - compute and cache all intermediate values
        cache = self._forward(params, data)
        
        # Backward pass - compute gradients using cached values
        Q, grads, P = self._backward(cache, data, norm)
        
        return Q, grads, P

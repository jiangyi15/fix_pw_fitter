import numpy as np

r"""


A_{i,k} = c_k [\prod_{r} BW_{k,r}(m,m0,g)] [\prod_{d} F_{k,d}(q)] \sum_{j} M_{k,k,j} [\prod_{l} \cos(k_{jl} \theta_{l} + b_{jl})]


use index to get m, m0, g, q, theta_{l}, BW, F
"""



class PWAFitter:
    def __init__(self, config):
        # Initialize indices and tables from config
        self.bw_index = config.get('bw_index')           # indices to get m for BW
        self.gamma_index = config.get('gamma_index')     # indices to get m for gamma interpolation
        self.bw_order = config.get('bw_order')           # ordering of BW terms
        self.bf_index = config.get('bf_index')           # indices for barrier factor q
        self.bf_order = config.get('bf_order')           # ordering of BF terms
        self.ang_index = config.get('ang_index')         # indices for angles
        self.ang_k = config.get('ang_k')                 # angular coefficients k
        self.ang_b = config.get('ang_b')                 # angular offsets b
        self.matrix_gamma = config.get('matrix_gamma')   # gamma coupling matrix
        self.matrix_ang = config.get('matrix_ang')       # angular basis matrix
        self.gamma_table = config.get('gamma_table')     # interpolation table for gamma
        self.bf_table = config.get('bf_table')           # interpolation table for barrier factors
        self.g_min = config.get('g_min')                 # min value for gamma interpolation
        self.g_delta = config.get('g_delta')             # step for gamma interpolation
        self.q_min = config.get('q_min')                 # min value for BF interpolation
        self.q_delta = config.get('q_delta')             # step for BF interpolation

    def compute(self, params, data, N=None):
        # ck (waves,)          complex
        # m0 (n_m0,)           real
        # g0 (n_g0,)           real
        # detal_m, delta_g, g, ap, lam, phi           real
        ck, m0, g0, delta_m, delta_g, g, ap, lam, phi  = params
        # mass   (events, topo, res)    real
        # q      (events, topo, decays)  real
        # angles (events, topo, ang)     real
        # weights, bkg, time, frac (events,)         real
        mass, q, angles, time, frac, weights, bkg = data

        n_events = mass.shape[0]
        n_waves = len(ck)

        # --- Breit-Wigner ---
        mass_flat = mass.reshape(mass.shape[0], -1)                 # (E, T*R)
        m_bw = np.take(mass_flat, self.bw_index, axis=1)            # (E, n_m0)
        m_gamma = np.take(mass_flat, self.gamma_index, axis=1)      # (E, n_g0)

        gi = self.interp(self.gamma_table, m_gamma, self.g_min, self.g_delta)  # (E, n_g0)
        g_vals = g0 * gi                                                  # (E, n_g0)
        gamma = np.einsum("mg,eg->em", self.matrix_gamma, g_vals)         # (E, n_m0)
        bwall = self.bw(m_bw, m0, gamma)                             # (E, n_m0)
        bw = np.take(bwall, self.bw_order, axis=1)                   # (E, W*R)

        # --- Barrier Factors ---
        q_flat = q.reshape(mass.shape[0], -1)                        # (E, T*D)
        q_bf = np.take(q_flat, self.bf_index, axis=1)                # (E, n_bf_types)
        bfall = self.interp(self.bf_table, q_bf, self.q_min, self.q_delta)  # (E, n_bf_types)
        bf = np.take(bfall, self.bf_order, axis=1)                   # (E, W*D)

        # --- Angular Part ---
        angles_flat = angles.reshape(angles.shape[0], -1)            # (E, T*A)
        ang = np.take(angles_flat, self.ang_index, axis=1)           # (E, n_basis, n_ang_per_basis)
        ang = ang * self.ang_k + self.ang_b
        ang_f = np.prod(np.cos(ang), axis=-1)                        # (E, n_basis)

        ag = np.einsum("wb,eb->ew", self.matrix_ang, ang_f)        # (E, W)

        # --- Combine ---
        n_res_per_wave = len(self.bw_order) // n_waves
        n_decays_per_wave = len(self.bf_order) // n_waves
        bw_r = bw.reshape(bw.shape[0], n_waves, n_res_per_wave)      # (E, W, R)
        bf_r = bf.reshape(bf.shape[0], n_waves, n_decays_per_wave)   # (E, W, D)
        bwprod = np.prod(bw_r, axis=-1)                              # (E, W)
        bfprod = np.prod(bf_r, axis=-1)                              # (E, W)
        inv_bwprod = 1.0 / bwprod                                    # (E, W)

        # Amplitude scale (independent of bw); safe divisor-free later
        a_full = ck * inv_bwprod * bfprod * ag                                  # (E, W)
        a_full_reshaped = np.reshape(a_full, (a_full.shape[0], 2, -1))
        amp = np.sum(a_full_reshaped, axis=-1)                                 # (E, 2)  sum over waves

        # Intensity:  |amplitude|^2  summed over projections
        gL = np.exp(-1j * time * (-delta_m/2 - 1j * (g + delta_g)/2))
        gH = np.exp(-1j * time * (+delta_m/2 - 1j * (g - delta_g)/2))
        gp = (gL + gH) / 2
        gm = (gL - gH) / 2
        pq = lam * np.exp(1j * phi)

        # Time-dependent amplitudes
        amp_p = gp * amp[:, 0] + gm * pq * amp[:, 1]    # (E,)
        amp_m = gm / pq * amp[:, 0] + gp * amp[:, 1]    # (E,)

        pt_p = np.abs(amp_p) ** 2    # (E,)
        pt_m = np.abs(amp_m) ** 2    # (E,)

        p = (1 - frac) * pt_p * (1 - ap) + frac * pt_m * (1 + ap)    # (E,)

        if N is None:
            q_val = np.sum(weights * p)
            # For chi-square, gradients are straightforward
            grad_p = weights    # d(q_val)/dp = weights
        else:
            q = p / N + bkg
            q_val = np.sum(weights * np.log(q))
            # For likelihood: q = p/N + bkg
            # d(q_val)/dp = weights * d(log(q))/dp = weights * (1/q) * (1/N)
            grad_p = weights / (N * q)    # d(q_val)/dp = weights / (N*q)

        # ========== Gradient Computations ==========

        # --- Gradient w.r.t. ap ---
        # p = (1-frac)*pt_p*(1-ap) + frac*pt_m*(1+ap)
        # dp/dap = -(1-frac)*pt_p + frac*pt_m
        dp_dap = -(1 - frac) * pt_p + frac * pt_m
        grad_ap = np.real(np.sum(grad_p * dp_dap))

        # --- Gradient w.r.t. frac ---
        # dp/d(frac) = -pt_p*(1-ap) + pt_m*(1+ap)
        grad_frac_factor = -pt_p * (1 - ap) + pt_m * (1 + ap)
        # Note: frac is data, not a parameter in current setup

        # --- Gradients w.r.t. pt_p and pt_m ---
        # These are intermediate quantities needed for other gradients
        grad_pt_p = grad_p * (1 - frac) * (1 - ap)    # d(q_val)/d(pt_p)
        grad_pt_m = grad_p * frac * (1 + ap)    # d(q_val)/d(pt_m)

        # --- Gradient w.r.t. pq (lam, phi) ---
        # pt_p = |gp*amp0 + gm*pq*amp1|^2
        # pt_m = |gm/pq*amp0 + gp*amp1|^2
        amp0, amp1 = amp[:, 0], amp[:, 1]

        # Wirtinger derivatives:
        # ∂(pt_p)/∂(pq*) = amp_p * conj(gm * amp1)
        # ∂(pt_m)/∂(pq*) = amp_m * conj(gm * amp0 / pq^2) = -amp_m * conj(gm * amp0) / conj(pq)^2
        #                  = -amp_m * conj(gm * amp0) / pq*^2 (note: careful with signs)

        # Actually for pt_m = |gm/pq * amp0 + gp*amp1|^2, let's compute ∂(pt_m)/∂(pq*):
        # Let u = gm/pq * amp0, then ∂u/∂(pq*) = 0 (since u is holomorphic in pq)
        # Let v = gp*amp1, ∂v/∂(pq*) = 0
        # ∂(|u+v|^2)/∂(pq*) = (u+v) * ∂(conj(u+v))/∂(pq*)
        # ∂conj(u)/∂(pq*) = ∂(conj(gm)*conj(amp0)/conj(pq))/∂(pq*) = -conj(gm)*conj(amp0)/pq*^2

        # So ∂(pt_m)/∂(pq*) = amp_m * (-conj(gm) * conj(amp0) / pq**2)
        #                   = -amp_m * conj(gm * amp0) / conj(pq)**2

        # Wait, let me redo this more carefully using the chain rule for practical gradients
        # pq is complex, so we need ∇_{pq} J = ∂J/∂Re(pq) + i∂J/∂Im(pq) = 2∂J/∂(pq*)

        # For pt_p = |amp_p|^2 where amp_p = gp*amp0 + gm*pq*amp1:
        # ∂(pt_p)/∂(pq*) = amp_p * conj(gm * amp1)

        # For pt_m = |amp_m|^2 where amp_m = gm/pq * amp0 + gp*amp1:
        # amp_m = gm * amp0 * (1/pq) + gp*amp1
        # ∂amp_m/∂(pq*) = 0 (since 1/pq is holomorphic)
        # ∂conj(amp_m)/∂(pq*) = conj(gm * amp0) * ∂(1/pq*)/∂(pq*) = conj(gm * amp0) * (-1/pq*^2)
        # So ∂(pt_m)/∂(pq*) = amp_m * conj(gm * amp0) * (-1/conj(pq)**2)

        dpt_p_dpq_star = amp_p * np.conj(gm * amp1)
        dpt_m_dpq_star = -amp_m * np.conj(gm * amp0) / np.conj(pq)**2

        # Practical gradient: ∇_{pq} J = 2 * ∂J/∂(pq*)
        grad_pq = 2 * (grad_pt_p * dpt_p_dpq_star + grad_pt_m * dpt_m_dpq_star)

        # pq = lam * exp(i*phi)
        # For real parameters lam and phi:
        # dJ/d(lam) = 2 * Re(∂J/∂(pq*) * ∂pq*/∂(lam)) = 2 * Re(grad_pq/2 * exp(-i*phi))
        #           = Re(grad_pq * exp(-i*phi))
        # dJ/d(phi) = 2 * Re(∂J/∂(pq*) * ∂pq*/∂(phi)) = 2 * Re(grad_pq/2 * (-i*lam*exp(-i*phi)))
        #           = Re(grad_pq * (-i*pq*)) = Re(-i * grad_pq * pq*)

        grad_lam = np.real(np.sum(grad_pq * np.exp(-1j * phi)))
        grad_phi = np.real(np.sum(-1j * grad_pq * np.conj(pq)))

        # --- Gradients w.r.t. time mixing parameters ---
        # gL = exp(-i*t*(-dm/2 - i*(g+dg)/2)) = exp(i*t*dm/2 - t*(g+dg)/2)
        # gH = exp(-i*t*(+dm/2 - i*(g-dg)/2)) = exp(-i*t*dm/2 - t*(g-dg)/2)

        # Derivatives of gL:
        # d(gL)/d(delta_m) = i*t/2 * gL
        # d(gL)/d(delta_g) = -t/2 * gL
        # d(gL)/d(g) = -t/2 * gL

        # Derivatives of gH:
        # d(gH)/d(delta_m) = -i*t/2 * gH
        # d(gH)/d(delta_g) = +t/2 * gH
        # d(gH)/d(g) = -t/2 * gH

        dgL_ddelta_m = 1j * time / 2 * gL
        dgH_ddelta_m = -1j * time / 2 * gH
        dgL_ddelta_g = -time / 2 * gL
        dgH_ddelta_g = time / 2 * gH
        dgL_dg = -time / 2 * gL
        dgH_dg = -time / 2 * gH

        dgp_ddelta_m = (dgL_ddelta_m + dgH_ddelta_m) / 2
        dgm_ddelta_m = (dgL_ddelta_m - dgH_ddelta_m) / 2
        dgp_ddelta_g = (dgL_ddelta_g + dgH_ddelta_g) / 2
        dgm_ddelta_g = (dgL_ddelta_g - dgH_ddelta_g) / 2
        dgp_dg = (dgL_dg + dgH_dg) / 2
        dgm_dg = (dgL_dg - dgH_dg) / 2

        # pt_p = |gp*amp0 + gm*pq*amp1|^2 = |amp_p|^2
        # For complex parameters, we compute ∂f/∂(gp*) which equals ∂(|amp_p|^2)/∂(gp*)
        # Using chain rule: ∂(|amp_p|^2)/∂(gp*) = amp_p * conj(amp0)
        dpt_p_dgp_star = amp_p * np.conj(amp0)      # ∂(pt_p)/∂(gp*)
        dpt_p_dgm_star = amp_p * np.conj(pq * amp1)  # ∂(pt_p)/∂(gm*)
        dpt_m_dgp_star = amp_m * np.conj(amp1)       # ∂(pt_m)/∂(gp*)
        dpt_m_dgm_star = amp_m * np.conj(amp0 / pq)  # ∂(pt_m)/∂(gm*)

        # For the gradient w.r.t. real parameters delta_m, delta_g, g:
        # d(pt_p)/d(dm) = ∂(pt_p)/∂(gp*) * ∂(gp*)/∂(dm) + ∂(pt_p)/∂(gp) * ∂(gp)/∂(dm) + similar for gm
        # Since dm is real: ∂(gp*)/∂(dm) = (dgp/dm)*
        # And ∂(pt_p)/∂(gp) = conj(∂(pt_p)/∂(gp*))
        # So: d(pt_p)/d(dm) = A * (dgp/dm)* + A* * (dgp/dm) = 2 * Re(A * conj(dgp/dm))
        # where A = dpt_p_dgp_star

        grad_delta_m = np.sum(
            grad_pt_p * 2 * np.real(dpt_p_dgp_star * np.conj(dgp_ddelta_m) +
                                    dpt_p_dgm_star * np.conj(dgm_ddelta_m)) +
            grad_pt_m * 2 * np.real(dpt_m_dgp_star * np.conj(dgp_ddelta_m) +
                                    dpt_m_dgm_star * np.conj(dgm_ddelta_m))
        )

        grad_delta_g = np.sum(
            grad_pt_p * 2 * np.real(dpt_p_dgp_star * np.conj(dgp_ddelta_g) +
                                    dpt_p_dgm_star * np.conj(dgm_ddelta_g)) +
            grad_pt_m * 2 * np.real(dpt_m_dgp_star * np.conj(dgp_ddelta_g) +
                                    dpt_m_dgm_star * np.conj(dgm_ddelta_g))
        )
        grad_g = np.sum(
            grad_pt_p * 2 * np.real(dpt_p_dgp_star * np.conj(dgp_dg) +
                                    dpt_p_dgm_star * np.conj(dgm_dg)) +
            grad_pt_m * 2 * np.real(dpt_m_dgp_star * np.conj(dgp_dg) +
                                    dpt_m_dgm_star * np.conj(dgm_dg))
        )

        # --- Gradient w.r.t. amplitudes amp0, amp1 ---
        # Using Wirtinger calculus: ∂(pt_p)/∂(amp0*) = amp_p * conj(gp)
        # Numerical gradient computes ∂f/∂Re(amp0) + i∂f/∂Im(amp0) = 2*∂f/∂(amp0*)
        dpt_p_damp0_star = amp_p * np.conj(gp)
        dpt_p_damp1_star = amp_p * np.conj(gm * pq)
        dpt_m_damp0_star = amp_m * np.conj(gm / pq)
        dpt_m_damp1_star = amp_m * np.conj(gp)

        dpt_p_damp0 = 2 * dpt_p_damp0_star
        dpt_p_damp1 = 2 * dpt_p_damp1_star
        dpt_m_damp0 = 2 * dpt_m_damp0_star
        dpt_m_damp1 = 2 * dpt_m_damp1_star

        grad_amp0 = grad_pt_p * dpt_p_damp0 + grad_pt_m * dpt_m_damp0
        grad_amp1 = grad_pt_p * dpt_p_damp1 + grad_pt_m * dpt_m_damp1

        # --- Gradient w.r.t. a_full (amplitudes per wave) ---
        # amp0 = sum over waves in first group, amp1 = sum over waves in second group
        # a_full has shape (E, W), reshaped to (E, 2, W//2)
        n_waves_per_group = n_waves // 2
        grad_a_full_reshaped = np.zeros((n_events, 2, n_waves_per_group), dtype=a_full.dtype)
        grad_a_full_reshaped[:, 0, :] = grad_amp0[:, np.newaxis]
        grad_a_full_reshaped[:, 1, :] = grad_amp1[:, np.newaxis]
        grad_a_full = grad_a_full_reshaped.reshape(n_events, n_waves)

        # --- Gradient w.r.t. ck ---
        # a_full = ck * prefactor, where prefactor = inv_bwprod * bfprod * ag
        # For practical gradient: ∇_{ck} J = ∂J/∂Re(ck) + i∂J/∂Im(ck)
        # Using chain rule: ∇_{ck[w]} J = sum_e grad_a_full[e,w] * conj(prefactor[e,w])
        prefactor = inv_bwprod * bfprod * ag    # (E, W)
        grad_ck = np.sum(grad_a_full * np.conj(prefactor), axis=0)    # (W,)

        # --- Gradient w.r.t. m0 ---
        # Need to propagate through Breit-Wigner
        # bw(m, m0, gamma) = m0^2 - m^2 - m0*gamma
        # dbw/dm0 = 2*m0 - gamma

        # First, gradient w.r.t. inv_bwprod
        # a_full = ck * inv_bwprod * bfprod * ag (holomorphic in inv_bwprod)
        # For practical gradient: ∇_{inv_bwprod} J = grad_a_full * conj(ck * bfprod * ag)
        grad_inv_bwprod = grad_a_full * np.conj(ck * bfprod * ag)    # (E, W)

        # d(inv_bwprod)/d(bw) for each bw in the product
        # For complex variables, using Wirtinger calculus:
        # inv_bwprod = 1/bwprod, where bwprod = prod(bw_i)
        # ∂(inv_bwprod)/∂(bw_i) = -inv_bwprod^2 * (bwprod/bw_i) = -inv_bwprod / bw_i
        # ∂(inv_bwprod)/∂(bw_i*) = 0
        # ∇_{bw_i} J = 2 * ∂J/∂(bw_i*) = 2 * (∂J/∂(inv_bwprod*)) * (∂(inv_bwprod*)/∂(bw_i*))
        #            = 2 * (∇_{inv_bwprod} J / 2) * (-1/bwprod*^2) * (bwprod*/bw_i*)
        #            = -∇_{inv_bwprod} J * inv_bwprod*^2 * bwprod*/bw_i*
        #            = -∇_{inv_bwprod} J * conj(inv_bwprod / bw_i)

        # bw has shape (E, W*R), need to compute gradient for each
        grad_bw = np.zeros_like(bw)
        for i in range(n_res_per_wave):
            bw_i = bw_r[:, :, i]    # (E, W)
            grad_bw_r = -grad_inv_bwprod * np.conj(inv_bwprod / bw_i)    # (E, W)
            grad_bw[:, i::n_res_per_wave] = grad_bw_r

        # bw = take(bwall, bw_order)
        # Need to scatter gradient back to bwall
        # bwall has shape (E, n_m0), bw has shape (E, W*R)
        grad_bwall = np.zeros((n_events, len(m0)), dtype=bw.dtype)
        for i, idx in enumerate(self.bw_order):
            grad_bwall[:, idx] += grad_bw[:, i]

        # bwall = bw(m_bw, m0, gamma)
        # bw = m0^2 - m^2 - i*m0*gamma (complex)
        # dbwall/dm0 = 2*m0 - i*gamma
        dbwall_dm0 = 2 * m0 - 1j * gamma    # (E, n_m0)
        grad_m0 = np.sum(np.real(grad_bwall * np.conj(dbwall_dm0)), axis=0)

        # --- Gradient w.r.t. g0 ---
        # bw(m, m0, gamma) = m0^2 - m^2 - i*m0*gamma
        # dbwall/dgamma = -i*m0
        dbwall_dgamma = -1j * m0[np.newaxis, :]    # (1, n_m0)

        grad_gamma = grad_bwall * np.conj(dbwall_dgamma)    # (E, n_m0)

        # gamma = einsum("mg,eg->em", matrix_gamma, g_vals)
        # dgamma/dg_vals = matrix_gamma.T
        grad_g_vals = np.einsum("mg,em->eg", self.matrix_gamma, grad_gamma)    # (E, n_g0)

        # g_vals = g0 * gi (gi is complex from gamma_table)
        # grad_g0 = sum over events of Re(grad_g_vals * conj(gi))
        grad_g0 = np.sum(np.real(grad_g_vals * np.conj(gi)), axis=0)    # (n_g0,)

        # --- Gradient w.r.t. N (normalization) ---
        if N is not None:
            # q = p/N + bkg
            # q_val = sum(w * log(q))
            # dq_val/dN = sum(w * (p/N^2) * (1/q)) = sum(w * p / (N^2 * q))
            grad_N = -np.sum(weights * p / (N * q)) / N
        else:
            grad_N = None

        grad = (grad_ck, grad_m0, grad_g0, grad_N, grad_delta_m, grad_delta_g,
                grad_g, grad_ap, grad_lam, grad_phi)

        return q_val, grad

    def interp(self, tables, value, x_min, delta_x):
        # tables  (N_types, interp_points)
        # value   (events, N_types)  or  (events,) for single type
        # returns (events, N_types) or (events,)
        diff = (value - x_min) / delta_x
        idx = np.clip(diff.astype(np.int32), 0, tables.shape[1] - 2)
        delta = diff - idx
        if value.ndim == 1:
            fl = tables[0, idx]
            fr = tables[0, idx + 1]
        else:
            # For multi-type tables, index each type separately
            n_events, n_types = idx.shape
            if tables.shape[0] == 1:
                # Single type, broadcast
                fl = tables[0, idx]
                fr = tables[0, idx + 1]
            else:
                # Multiple types: select from each row
                fl = np.empty((n_events, n_types), dtype=tables.dtype)
                fr = np.empty((n_events, n_types), dtype=tables.dtype)
                for j in range(n_types):
                    fl[:, j] = tables[j, idx[:, j]]
                    fr[:, j] = tables[j, idx[:, j] + 1]
        return (fr - fl) * delta + fl

    def bw(self, m, m0, gamma):
        """Relativistic Breit-Wigner lineshape denominator (complex)

        BW = 1 / (m0^2 - m^2 - i*m0*Gamma)
        This returns the complex denominator: m0^2 - m^2 - i*m0*gamma
        where gamma is the width parameter.
        """
        return m0 ** 2 - m ** 2 - 1j * m0 * gamma


def numerical_gradient(fitter, params, data, N=None, eps=1e-5):
    """Compute numerical gradients for testing"""
    q_val, _ = fitter.compute(params, data, N)

    ck, m0, g0, delta_m, delta_g, g, ap, lam, phi = params

    grads = []

    # Gradient for complex ck - use the "practical" Wirtinger derivative
    # For real-valued f(z), the gradient used in optimization is ∂f/∂z*
    # ∂f/∂z* = ∂f/∂Re(z) + i * ∂f/∂Im(z)
    grad_ck_num = np.zeros(len(ck), dtype=np.complex128)
    for i in range(len(ck)):
        # Perturb real part
        params_plus = list(params)
        params_plus[0] = ck.copy()
        params_plus[0][i] = ck[i] + eps
        q_plus, _ = fitter.compute(tuple(params_plus), data, N)

        params_minus = list(params)
        params_minus[0] = ck.copy()
        params_minus[0][i] = ck[i] - eps
        q_minus, _ = fitter.compute(tuple(params_minus), data, N)

        grad_re = (q_plus - q_minus) / (2 * eps)

        # Perturb imaginary part
        params_plus = list(params)
        params_plus[0] = ck.copy()
        params_plus[0][i] = ck[i] + 1j * eps
        q_plus, _ = fitter.compute(tuple(params_plus), data, N)

        params_minus = list(params)
        params_minus[0] = ck.copy()
        params_minus[0][i] = ck[i] - 1j * eps
        q_minus, _ = fitter.compute(tuple(params_minus), data, N)

        grad_im = (q_plus - q_minus) / (2 * eps)

        # Practical gradient: ∂f/∂z* = grad_re + i*grad_im
        grad_ck_num[i] = grad_re + 1j * grad_im
    grads.append(grad_ck_num)

    # Gradient for m0 (index 1)
    grad_m0_num = np.zeros_like(m0, dtype=np.float64)
    for i in range(len(m0)):
        params_plus = list(params)
        params_plus[1] = m0.copy()
        params_plus[1][i] += eps
        q_plus, _ = fitter.compute(tuple(params_plus), data, N)

        params_minus = list(params)
        params_minus[1] = m0.copy()
        params_minus[1][i] -= eps
        q_minus, _ = fitter.compute(tuple(params_minus), data, N)

        grad_m0_num[i] = (q_plus - q_minus) / (2 * eps)
    grads.append(grad_m0_num)

    # Gradient for g0 (index 2)
    grad_g0_num = np.zeros_like(g0, dtype=np.float64)
    for i in range(len(g0)):
        params_plus = list(params)
        params_plus[2] = g0.copy()
        params_plus[2][i] += eps
        q_plus, _ = fitter.compute(tuple(params_plus), data, N)

        params_minus = list(params)
        params_minus[2] = g0.copy()
        params_minus[2][i] -= eps
        q_minus, _ = fitter.compute(tuple(params_minus), data, N)

        grad_g0_num[i] = (q_plus - q_minus) / (2 * eps)
    grads.append(grad_g0_num)

    # Gradient for N
    if N is not None:
        q_plus, _ = fitter.compute(params, data, N + eps)
        q_minus, _ = fitter.compute(params, data, N - eps)
        grad_N_num = (q_plus - q_minus) / (2 * eps)
        grads.append(grad_N_num)
    else:
        grads.append(None)

    # Gradients for scalar parameters: delta_m, delta_g, g, ap, lam, phi (indices 3-8)
    scalar_indices = [3, 4, 5, 6, 7, 8]
    for param_idx in scalar_indices:
        param = params[param_idx]
        params_plus = list(params)
        params_plus[param_idx] = param + eps
        q_plus, _ = fitter.compute(tuple(params_plus), data, N)

        params_minus = list(params)
        params_minus[param_idx] = param - eps
        q_minus, _ = fitter.compute(tuple(params_minus), data, N)

        grad_num = (q_plus - q_minus) / (2 * eps)
        grads.append(grad_num)

    return grads


def test_gradients():
    """Test analytical gradients against numerical gradients"""
    np.random.seed(42)

    # Use more complex test case
    n_events = 20
    n_waves = 4
    n_m0 = 3
    n_g0 = 2
    n_res_per_wave = 2
    n_decays_per_wave = 2
    n_bf_types = 4
    n_basis = 5
    n_ang_per_basis = 2

    # Mass: (E, topo, res) -> mass_flat: (E, T*R)
    n_mass_cols = 3 * 3  # 9
    n_q_cols = 3 * 4  # 12
    n_ang_cols = 2 * 5  # 10

    config = {
        'bw_index': np.random.randint(0, n_mass_cols, n_m0),
        'gamma_index': np.random.randint(0, n_mass_cols, n_g0),
        'bw_order': np.random.randint(0, n_m0, n_waves * n_res_per_wave),
        'bf_index': np.random.randint(0, n_q_cols, n_bf_types),
        'bf_order': np.random.randint(0, n_bf_types, n_waves * n_decays_per_wave),
        'ang_index': np.random.randint(0, n_ang_cols, (n_basis, n_ang_per_basis)),
        'ang_k': np.random.randn(n_basis, n_ang_per_basis),
        'ang_b': np.random.randn(n_basis, n_ang_per_basis),
        'matrix_gamma': np.random.randn(n_m0, n_g0),
        'matrix_ang': np.random.randn(n_waves, n_basis) + 1j * np.random.randn(n_waves, n_basis),
        'gamma_table': np.exp(1j * np.linspace(0, np.pi/2, 100).reshape(1, -1) * np.random.randn(1, 1)),
        'bf_table': np.exp(np.linspace(0, 1, 100).reshape(1, -1)),
        'g_min': 0.0,
        'g_delta': 0.01,
        'q_min': 0.0,
        'q_delta': 0.01,
    }

    fitter = PWAFitter(config)

    # Generate test data
    mass = np.random.rand(n_events, 3, 3) * 0.5 + 0.5
    q = np.random.rand(n_events, 3, 4) * 0.5 + 0.25
    angles = np.random.rand(n_events, 2, 5) * np.pi
    time = np.random.rand(n_events) * 5 + 0.5
    frac = np.random.rand(n_events) * 0.3 - 0.15
    weights = np.ones(n_events)
    bkg = np.random.rand(n_events) * 0.05 + 0.05

    data = (mass, q, angles, time, frac, weights, bkg)

    # Test parameters
    ck = np.random.randn(n_waves) + 1j * np.random.randn(n_waves)
    m0 = np.random.rand(n_m0) + 1.5
    g0 = np.random.rand(n_g0) * 0.1 + 0.05
    delta_m = np.random.rand() * 0.5
    delta_g = np.random.rand() * 0.05
    g = np.random.rand() * 0.5 + 0.5
    ap = np.random.rand() * 0.02
    lam = np.random.rand() * 0.3 + 0.6
    phi = np.random.rand() * 0.3

    params = (ck, m0, g0, delta_m, delta_g, g, ap, lam, phi)
    N = 500.0

    # Compute analytical gradients
    q_val, grad_analytical = fitter.compute(params, data, N)

    print(f"Function value: {q_val:.6f}")
    print()

    # Compute numerical gradients
    grad_numerical = numerical_gradient(fitter, params, data, N)

    param_names = ['ck', 'm0', 'g0', 'N', 'delta_m', 'delta_g', 'g', 'ap', 'lam', 'phi']

    print("Gradient comparison (analytical vs numerical):")
    print("=" * 60)

    all_passed = True
    for i, (name, ga, gn) in enumerate(zip(param_names, grad_analytical, grad_numerical)):
        if ga is None or gn is None:
            print(f"{name}: skipped (None)")
            continue

        ga_flat = np.atleast_1d(ga).flatten()
        gn_flat = np.atleast_1d(gn).flatten()

        # Relative error
        rel_err = np.abs(ga_flat - gn_flat) / (np.abs(gn_flat) + 1e-10)
        max_rel_err = np.max(rel_err)

        passed = max_rel_err < 1e-4
        status = "PASS" if passed else "FAIL"

        if not passed:
            all_passed = False

        print(f"{name:12s}: max_rel_err = {max_rel_err:.2e}  [{status}]")

    print("=" * 60)
    if all_passed:
        print("All gradient tests PASSED!")
    else:
        print("Some gradient tests FAILED!")

    return all_passed
    n_events = 10
    n_waves = 4
    n_m0 = 2
    n_g0 = 2
    n_res_per_wave = 2
    n_decays_per_wave = 2
    n_bf_types = 3
    n_basis = 5
    n_ang_per_basis = 2

    # Mass: (E, topo, res) -> mass_flat: (E, T*R)
    # We have (n_events, 3, 2) so mass_flat has shape (n_events, 6)
    n_mass_cols = 3 * 2  # 6
    # q: (E, topo, decays) -> q_flat: (E, T*D)
    n_q_cols = 2 * 3  # 6
    # angles: (E, topo, ang) -> angles_flat: (E, T*A)
    n_ang_cols = 2 * 4  # 8

    config = {
        # Indices must be within mass_flat bounds (0 to n_mass_cols-1)
        'bw_index': np.random.randint(0, n_mass_cols, n_m0),
        'gamma_index': np.random.randint(0, n_mass_cols, n_g0),
        # bw_order indexes into bwall which has shape (E, n_m0)
        'bw_order': np.random.randint(0, n_m0, n_waves * n_res_per_wave),
        # Indices must be within q_flat bounds (0 to n_q_cols-1)
        'bf_index': np.random.randint(0, n_q_cols, n_bf_types),
        # bf_order indexes into bfall which has shape (E, n_bf_types)
        'bf_order': np.random.randint(0, n_bf_types, n_waves * n_decays_per_wave),
        # Indices must be within angles_flat bounds (0 to n_ang_cols-1)
        'ang_index': np.random.randint(0, n_ang_cols, (n_basis, n_ang_per_basis)),
        'ang_k': np.random.randn(n_basis, n_ang_per_basis),
        'ang_b': np.random.randn(n_basis, n_ang_per_basis),
        'matrix_gamma': np.random.randn(n_m0, n_g0),
        'matrix_ang': np.random.randn(n_waves, n_basis) + 1j * np.random.randn(n_waves, n_basis),  # complex
        'gamma_table': np.exp(1j * np.linspace(0, np.pi, 100).reshape(1, -1)),  # complex
        'bf_table': np.exp(np.linspace(0, 1, 100).reshape(1, -1)),
        'g_min': 0.0,
        'g_delta': 0.01,
        'q_min': 0.0,
        'q_delta': 0.01,
    }

    fitter = PWAFitter(config)

    # Generate test data - values must be in interpolation table range
    # mass values should give gamma in reasonable range
    mass = np.random.rand(n_events, 3, 2) * 0.5 + 0.5  # range [0.5, 1.0]
    # q values should be in BF table range
    q = np.random.rand(n_events, 2, 3) * 0.5 + 0.25  # range [0.25, 0.75]
    # angle values
    angles = np.random.rand(n_events, 2, 4) * np.pi
    time = np.random.rand(n_events) * 10
    frac = np.random.rand(n_events) * 0.4 - 0.2
    weights = np.ones(n_events)
    bkg = np.random.rand(n_events) * 0.1

    data = (mass, q, angles, time, frac, weights, bkg)

    # Generate test parameters
    ck = np.random.randn(n_waves) + 1j * np.random.randn(n_waves)
    m0 = np.random.rand(n_m0) + 1.5
    g0 = np.random.rand(n_g0) * 0.1 + 0.05
    delta_m = 0.5
    delta_g = 0.02
    g = 0.65
    ap = 0.01
    lam = 0.7
    phi = 0.1

    params = (ck, m0, g0, delta_m, delta_g, g, ap, lam, phi)
    N = 1000.0

    # Compute analytical gradients
    q_val, grad_analytical = fitter.compute(params, data, N)

    print(f"Function value: {q_val:.6f}")
    print()

    # Compute numerical gradients
    grad_numerical = numerical_gradient(fitter, params, data, N)

    param_names = ['ck', 'm0', 'g0', 'N', 'delta_m', 'delta_g', 'g', 'ap', 'lam', 'phi']

    print("Gradient comparison (analytical vs numerical):")
    print("=" * 60)

    all_passed = True
    for i, (name, ga, gn) in enumerate(zip(param_names, grad_analytical, grad_numerical)):
        if ga is None or gn is None:
            print(f"{name}: skipped (None)")
            continue

        ga_flat = np.atleast_1d(ga).flatten()
        gn_flat = np.atleast_1d(gn).flatten()

        # Relative error
        rel_err = np.abs(ga_flat - gn_flat) / (np.abs(gn_flat) + 1e-10)
        max_rel_err = np.max(rel_err)

        passed = max_rel_err < 1e-4
        status = "PASS" if passed else "FAIL"

        if not passed:
            all_passed = False

        print(f"{name:12s}: max_rel_err = {max_rel_err:.2e}  [{status}]")
        if not passed and len(ga_flat) <= 4:
            print(f"              analytical: {ga_flat}")
            print(f"              numerical:  {gn_flat}")

    print("=" * 60)
    if all_passed:
        print("All gradient tests PASSED!")
    else:
        print("Some gradient tests FAILED!")

    return all_passed


if __name__ == "__main__":
    test_gradients()

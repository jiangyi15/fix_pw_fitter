"""
CUDA kernel using CuPy - NumPy-compatible GPU arrays.

Features:
1. Persistent GPU data (load once, compute many times)
2. Correct Wirtinger calculus for gradients
3. NumPy-compatible API - easy to maintain
4. Automatic GPU memory management
"""

import numpy as np

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CuPy not available. Install with: pip install cupy-cuda11x or cupy-cuda12x")


class GPUData:
    """
    Manages persistent GPU data for amplitude analysis.

    Usage:
        gpu_data = GPUData(config)
        gpu_data.load_data(data)  # Load once
        Q, grads = gpu_data.compute(params)  # Compute many times
    """

    def __init__(self, config):
        """Initialize GPU memory with configuration"""
        if not CUDA_AVAILABLE:
            raise RuntimeError("CuPy not available. Cannot use GPU acceleration.")

        self.config = config

        # Allocate config arrays on GPU (persistent)
        self._alloc_config_gpu()

        # Data arrays (allocated when load_data is called)
        self.mass_gpu = None
        self.momentum_gpu = None
        self.angle_gpu = None
        self.frac_gpu = None
        self.time_gpu = None
        self.weight_gpu = None
        self.bkg_gpu = None

        self.n_events = 0
        self.data_loaded = False

        # Extract config dimensions
        self.n_wave = config["matrix_angle"].shape[1]
        self.n_res = config["bw_order"].size // self.n_wave
        self.n_decay = config["fl_order"].size // self.n_wave

    def _alloc_config_gpu(self):
        """Allocate and copy configuration arrays to GPU"""
        # These stay on GPU for the lifetime of GPUData
        self.m0_index_gpu = cp.array(self.config["m0_index"], dtype=cp.int32)
        self.g0_index_gpu = cp.array(self.config["g0_index"], dtype=cp.int32)
        self.fl_type_gpu = cp.array(self.config["fl_type"], dtype=cp.int32)
        self.mass_index_gpu = cp.array(self.config["mass_index"], dtype=cp.int32)
        self.g0_mass_index_gpu = cp.array(self.config["g0_mass_index"], dtype=cp.int32)
        self.fl_q_index_gpu = cp.array(self.config["fl_q_index"], dtype=cp.int32)
        self.bw_order_gpu = cp.array(self.config["bw_order"], dtype=cp.int32)
        self.fl_order_gpu = cp.array(self.config["fl_order"], dtype=cp.int32)
        self.angle_index_gpu = cp.array(self.config["angle_index"], dtype=cp.int32)

        self.angle_k_gpu = cp.array(self.config["angle_k"], dtype=cp.float64)
        self.angle_b_gpu = cp.array(self.config["angle_b"], dtype=cp.float64)
        self.matrix_angle_gpu = cp.array(self.config["matrix_angle"], dtype=cp.float64)
        self.matrix_gamma_gpu = cp.array(self.config["matrix_gamma"], dtype=cp.float64)
        self.gamma_table_gpu = cp.array(self.config["gamma_table"], dtype=cp.float64)
        self.fl_table_gpu = cp.array(self.config["fl_table"], dtype=cp.float64)

        # Extract scalars
        self.gamma_min = self.config["gamma_min"]
        self.gamma_delta = self.config["gamma_delta"]
        self.fl_min = self.config["fl_min"]
        self.fl_delta = self.config["fl_delta"]

    def load_data(self, data):
        """
        Load data to GPU memory (call once).

        Args:
            data: dict with keys 'mass', 'q', 'angle', 'frac', 'time', 'weight', 'bkg'
        """
        self.n_events = data["mass"].shape[0]

        # Copy data to GPU (these persist until load_data is called again)
        print(f"Loading {self.n_events} events to GPU...")
        self.mass_gpu = cp.asarray(data["mass"], dtype=cp.float64)
        self.momentum_gpu = cp.asarray(data["q"], dtype=cp.float64)
        self.angle_gpu = cp.asarray(data["angle"], dtype=cp.float64)
        self.frac_gpu = cp.asarray(data["frac"], dtype=cp.float64)
        self.time_gpu = cp.asarray(data["time"], dtype=cp.float64)
        self.weight_gpu = cp.asarray(data["weight"], dtype=cp.float64)

        bkg = data.get("bkg", 0.0)
        if np.isscalar(bkg):
            bkg = np.full(self.n_events, bkg, dtype=np.float64)
        self.bkg_gpu = cp.asarray(bkg, dtype=cp.float64)

        self.data_loaded = True
        print(f"✓ Data loaded to GPU ({self.n_events} events)")

    def interp(self, table, types, x, xmin, xdelta):
        """Vectorized linear interpolation on GPU"""
        diff = (x - xmin) / xdelta
        xbin = cp.floor(diff).astype(cp.intp)
        n_bins = table.shape[-1]
        xbin = cp.clip(xbin, 0, n_bins - 2)
        delta = diff - xbin
        idx = types * n_bins + xbin
        left = table.ravel()[idx]
        right = table.ravel()[idx + 1]
        return (right - left) * delta + left

    def compute(self, params, norm=None):
        """
        Compute Q and gradients with current parameters.

        Args:
            params: dict with 'ck', 'm0', 'g0', 'scalar'
            norm: optional normalization factor

        Returns:
            Q: scalar loss
            grads: dict of gradients
            P: probability array
        """
        if not self.data_loaded:
            raise RuntimeError("Must call load_data() before compute()")

        # Transfer parameters to GPU (small arrays, fast transfer)
        ck = cp.asarray(params["ck"], dtype=cp.complex128)
        m0 = cp.asarray(params["m0"], dtype=cp.float64)
        g0 = cp.asarray(params["g0"], dtype=cp.float64)
        Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi = params["scalar"]

        # Reference persistent GPU data
        mass = self.mass_gpu
        momentum = self.momentum_gpu
        angle = self.angle_gpu
        frac = self.frac_gpu
        time = self.time_gpu
        weight = self.weight_gpu
        bkg = self.bkg_gpu

        # ==================== FORWARD PASS ====================

        # BW propagators
        g0_all = g0[self.g0_index_gpu]
        g0_m = mass[:, self.g0_mass_index_gpu]
        g_interp = self.interp(self.gamma_table_gpu, self.g0_index_gpu, g0_m,
                               self.gamma_min, self.gamma_delta)
        g = g0_all * g_interp
        g_bw = cp.dot(g, self.matrix_gamma_gpu)

        m0_all = m0[self.m0_index_gpu]
        m0_m = mass[:, self.mass_index_gpu]
        bw_dom = m0_all**2 - m0_m**2 - 1j * m0_all * g_bw

        bw_dom_all = bw_dom[:, self.bw_order_gpu]
        n_events = bw_dom_all.shape[0]
        bw_dom_all_reshaped = bw_dom_all.reshape(n_events, self.n_wave, self.n_res)
        bw_p = cp.prod(bw_dom_all_reshaped, axis=-1)

        # FL factors
        fl_q = momentum[:, self.fl_q_index_gpu]
        fl = self.interp(self.fl_table_gpu, self.fl_type_gpu, fl_q,
                        self.fl_min, self.fl_delta)
        fl_all = fl[:, self.fl_order_gpu]
        fl_p = cp.prod(fl_all.reshape(-1, self.n_wave, self.n_decay), axis=-1)

        # Angular factors
        ang = angle[:, self.angle_index_gpu, :]
        ka = cp.prod(cp.cos(ang * self.angle_k_gpu + self.angle_b_gpu), axis=-1)
        fa = cp.dot(ka, self.matrix_angle_gpu)

        # Amplitude
        one_over_bw = 1.0 / bw_p
        fa_times_fl = fa * fl_p
        common_amp_factor = one_over_bw * fa_times_fl

        a = ck * common_amp_factor
        a_reshaped = a.reshape(-1, 2, self.n_wave // 2)
        ap = cp.sum(a_reshaped[:, 0, :], axis=-1)
        am = cp.sum(a_reshaped[:, 1, :], axis=-1)

        # Time evolution
        eL = cp.exp(-1j * time * (-Delta_m/2 - 1j * (Gamma + Delta_Gamma/2)/2))
        eH = cp.exp(-1j * time * (+Delta_m/2 - 1j * (Gamma - Delta_Gamma/2)/2))
        gp = (eL + eH) / 2
        gm = (eL - eH) / 2

        # Probabilities
        poq = poq_rho * cp.exp(1j * pop_phi)
        pap = gp * ap + gm * poq * am
        pam = (gm / poq) * ap + gp * am

        pb = cp.abs(pap)**2
        pbbar = cp.abs(pam)**2

        P = frac * pb * (1 - A_p) + (1 - frac) * pbbar * (1 + A_p)

        # Loss
        if norm is None:
            Q = cp.sum(weight * P)
            dQ_dP = weight
        else:
            Q = -cp.sum(weight * cp.log(P / norm + bkg))
            dQ_dP = -weight / (P / norm + bkg)

        # ==================== BACKWARD PASS ====================
        # Use Wirtinger calculus consistently

        # Probability gradients (REAL)
        dP_dpb = frac * (1 - A_p)
        dP_dpbbar = (1 - frac) * (1 + A_p)
        dP_dAp = -frac * pb + (1 - frac) * pbbar
        dQ_dAp = cp.sum(dQ_dP * dP_dAp)

        dQ_dpb = dQ_dP * dP_dpb
        dQ_dpbbar = dQ_dP * dP_dpbbar

        # Wirtinger gradients (COMPLEX):
        d_pb_dap = cp.conj(pap) * gp
        d_pb_dam = cp.conj(pap) * gm * poq
        d_pbbar_dap = cp.conj(pam) * (gm / poq)
        d_pbbar_dam = cp.conj(pam) * gp

        # Complex gradients for ap and am:
        dQ_dap_Wirtinger = dQ_dpb * d_pb_dap + dQ_dpbbar * d_pbbar_dap
        dQ_dam_Wirtinger = dQ_dpb * d_pb_dam + dQ_dpbbar * d_pbbar_dam

        # BACKPROP THROUGH SUM
        dQ_da_Wirtinger = cp.zeros_like(a, dtype=complex)
        dQ_da_Wirtinger_reshaped = dQ_da_Wirtinger.reshape(-1, 2, self.n_wave // 2)

        # Assign Wirtinger gradients
        dQ_da_Wirtinger_reshaped[:, 0, :] = dQ_dap_Wirtinger[:, cp.newaxis]
        dQ_da_Wirtinger_reshaped[:, 1, :] = dQ_dam_Wirtinger[:, cp.newaxis]
        dQ_da_flat = dQ_da_Wirtinger

        # GRADIENT FOR ck (COMPLEX PARAMETER)
        grad_ck = cp.sum(dQ_da_flat * common_amp_factor, axis=0)

        # GRADIENT FOR bw_p (COMPLEX)
        dQ_dbw_p = dQ_da_flat * (-ck * one_over_bw * common_amp_factor)

        # BW gradients through product
        dQ_dbw_dom_all = cp.zeros_like(bw_dom_all_reshaped)
        for i in range(self.n_res):
            mask = cp.ones(self.n_res, dtype=bool)
            mask[i] = False
            prod_except_i = cp.prod(bw_dom_all_reshaped[:, :, mask], axis=-1)
            dQ_dbw_dom_all[:, :, i] = dQ_dbw_p * prod_except_i

        # Scatter gradients
        dQ_dbw_dom = cp.zeros_like(bw_dom)
        for wave_idx in range(self.n_wave):
            for res_idx in range(self.n_res):
                order_idx = wave_idx * self.n_res + res_idx
                bw_idx = self.bw_order_gpu[order_idx]
                dQ_dbw_dom[:, bw_idx] += dQ_dbw_dom_all[:, wave_idx, res_idx]

        # GRADIENT FOR m0 (REAL PARAMETER)
        dbw_dom_dm0 = 2 * m0_all - 1j * g_bw

        grad_m0 = cp.zeros_like(m0)
        for bw_idx in range(len(self.m0_index_gpu)):
            m0_param_idx = self.m0_index_gpu[bw_idx]
            # Wirtinger gradient for real parameter:
            # ∂Q/∂m0 = 2*Re(∂Q/∂bw_dom * ∂bw_dom/∂m0)
            grad_m0[m0_param_idx] += 2 * cp.sum(cp.real(
                dQ_dbw_dom[:, bw_idx] * dbw_dom_dm0[:, bw_idx]
            ))

        # GRADIENT FOR g0 (REAL PARAMETER)
        dQ_dg_bw = dQ_dbw_dom * (-1j * m0_all)
        dQ_dg = cp.dot(dQ_dg_bw, self.matrix_gamma_gpu.T)

        grad_g0 = cp.zeros_like(g0)
        for gamma_idx in range(len(self.g0_index_gpu)):
            g0_param_idx = self.g0_index_gpu[gamma_idx]
            # ∂Q/∂g0 = 2*Re(∂Q/∂g * ∂g/∂g0) = 2*Re(∂Q/∂g * g_interp)
            grad_g0[g0_param_idx] += 2 * cp.sum(cp.real(
                dQ_dg[:, gamma_idx] * g_interp[:, gamma_idx]
            ))

        # GRADIENTS FOR TIME EVOLUTION PARAMETERS
        d_pb_dgp = cp.conj(pap) * ap
        d_pb_dgm = cp.conj(pap) * poq * am
        d_pbbar_dgp = cp.conj(pam) * am
        d_pbbar_dgm = cp.conj(pam) * ap / poq

        dQ_dgp = dQ_dpb * d_pb_dgp + dQ_dpbbar * d_pbbar_dgp
        dQ_dgm = dQ_dpb * d_pb_dgm + dQ_dpbbar * d_pbbar_dgm

        # Time evolution gradients using Wirtinger calculus
        deL_dGamma = -1j * time * (-1j/2) * eL
        deH_dGamma = -1j * time * (-1j/2) * eH
        deL_dDeltaGamma = -1j * time * (-1j/4) * eL
        deH_dDeltaGamma = -1j * time * (+1j/4) * eH
        deL_dDeltam = -1j * time * (-1/2) * eL
        deH_dDeltam = -1j * time * (+1/2) * eH

        dgp_dGamma = (deL_dGamma + deH_dGamma) / 2
        dgm_dGamma = (deL_dGamma - deH_dGamma) / 2
        dgp_dDeltaGamma = (deL_dDeltaGamma + deH_dDeltaGamma) / 2
        dgm_dDeltaGamma = (deL_dDeltaGamma - deH_dDeltaGamma) / 2
        dgp_dDeltam = (deL_dDeltam + deH_dDeltam) / 2
        dgm_dDeltam = (deL_dDeltam - deH_dDeltam) / 2

        # For real parameters: ∂Q/∂x = 2*Re(∂Q/∂gp * ∂gp/∂x + ∂Q/∂gm * ∂gm/∂x)
        grad_Gamma = 2 * cp.sum(cp.real(dQ_dgp * dgp_dGamma + dQ_dgm * dgm_dGamma))
        grad_Delta_Gamma = 2 * cp.sum(cp.real(dQ_dgp * dgp_dDeltaGamma + dQ_dgm * dgm_dDeltaGamma))
        grad_Delta_m = 2 * cp.sum(cp.real(dQ_dgp * dgp_dDeltam + dQ_dgm * dgm_dDeltam))

        # GRADIENT FOR poq (complex parameter)
        d_pap_dpoq = gm * am
        d_pam_dpoq = -gm * ap / (poq**2)

        dQ_dpoq = dQ_dpb * cp.conj(pap) * d_pap_dpoq + dQ_dpbbar * cp.conj(pam) * d_pam_dpoq

        # poq = poq_rho * exp(i * pop_phi)
        # ∂poq/∂poq_rho = exp(i * pop_phi)
        # ∂poq/∂pop_phi = i * poq
        d_poq_dpoq_rho = cp.exp(1j * pop_phi)
        d_poq_dpop_phi = 1j * poq

        # For real parameters poq_rho and pop_phi:
        grad_poq_rho = 2 * cp.sum(cp.real(dQ_dpoq * d_poq_dpoq_rho))
        grad_pop_phi = 2 * cp.sum(cp.real(dQ_dpoq * d_poq_dpop_phi))

        # Transfer results back to CPU
        Q_cpu = float(Q.get())
        P_cpu = P.get()
        grads_cpu = {
            "ck": grad_ck.get(),
            "m0": grad_m0.get(),
            "g0": grad_g0.get(),
            "scalar": np.array([
                float(grad_Gamma.get()),
                float(grad_Delta_Gamma.get()),
                float(grad_Delta_m.get()),
                float(dQ_dAp.get()),
                float(grad_poq_rho.get()),
                float(grad_pop_phi.get())
            ])
        }

        return Q_cpu, grads_cpu, P_cpu

    def free_data(self):
        """Free GPU data memory"""
        self.mass_gpu = None
        self.momentum_gpu = None
        self.angle_gpu = None
        self.frac_gpu = None
        self.time_gpu = None
        self.weight_gpu = None
        self.bkg_gpu = None
        self.data_loaded = False
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()


class CUDAKernel:
    """
    CUDA kernel with persistent GPU data.

    Provides the same interface as NumPy kernel but runs on GPU.

    Usage:
        kernel = CUDAKernel(config)
        kernel.load_data(data)  # Load once
        Q, grads, P = kernel.compute(params, norm=None)  # Compute many times
    """

    def __init__(self, config):
        """Initialize with config"""
        if not CUDA_AVAILABLE:
            raise RuntimeError("CuPy not available. Cannot use GPU acceleration.")

        self.gpu_data = GPUData(config)
        self.config = config

    def load_data(self, data):
        """Load data to GPU (persistent)"""
        self.gpu_data.load_data(data)

    def compute(self, params, norm=None):
        """
        Compute with current parameters.

        Args:
            params: parameter dict
            norm: optional normalization

        Returns:
            Q: scalar
            grads: dict of gradients
            P: probability array
        """
        return self.gpu_data.compute(params, norm)

    def free_data(self):
        """Free GPU data"""
        self.gpu_data.free_data()


if __name__ == "__main__":
    # Test the CUDA kernel
    from config_loader import Config
    import time

    if not CUDA_AVAILABLE:
        print("CuPy not available. Install with: pip install cupy-cuda11x or cupy-cuda12x")
        exit(1)

    print("="*70)
    print("CUDA KERNEL TEST WITH PERSISTENT GPU DATA")
    print("="*70)

    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()

    # Create CUDA kernel
    kernel = CUDAKernel(kernel_config)

    # Create test data
    n_events = 1000
    np.random.seed(42)
    data = {
        "mass": np.random.random((n_events, 2*3*8)),
        "q": np.random.random((n_events, 3*3*8)),
        "angle": np.random.random((n_events, 3*8, 3)),
        "frac": np.random.random((n_events,)),
        "time": np.random.random((n_events,)),
        "bkg": np.random.random((n_events,)) * 0.01,
        "weight": np.ones((n_events,)),
    }

    # Load data to GPU ONCE
    kernel.load_data(data)

    # Create parameters
    ck_map = config.get_ck_map()
    params = {
        "ck": np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
        "m0": np.random.random(len(config.m0_phys_name)) + 2,
        "g0": np.random.random(len(config.g0_phys_name)) + 0.1,
        "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }

    # Compute multiple times without reloading data
    print("\nComputing with different parameters (data stays on GPU)...")
    times = []
    for i in range(10):
        # Modify parameters slightly
        params["m0"] = params["m0"] + np.random.random(len(params["m0"])) * 0.01

        # Compute - data stays on GPU
        start = time.time()
        Q, grads, P = kernel.compute(params, norm=None)
        elapsed = time.time() - start
        times.append(elapsed)

        if i < 3:
            print(f"  Iteration {i+1}: Q = {Q:.6f}, grad_m0[0] = {grads['m0'][0]:.6f}, time = {elapsed*1000:.2f} ms")

    avg_time = np.mean(times)
    print(f"\nAverage time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {n_events/avg_time:.0f} events/sec")

    print("\n" + "="*70)
    print("✓ CUDA kernel test complete")
    print("✓ Data loaded ONCE, computed MANY times")
    print("✓ All gradients computed with Wirtinger calculus")
    print("="*70)

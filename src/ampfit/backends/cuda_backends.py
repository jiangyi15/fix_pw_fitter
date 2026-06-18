"""CUDA backends — GPU-accelerated computation (f64 and f32)."""
import numpy as np
from .core import ComputeBackend, register_backend


# ── v2 backends (C-managed memory, linear interpolation) ──────

@register_backend("cuda_v2")
@register_backend("cuda64_v2")
class CUDABackendV2(ComputeBackend):
    """v2 CUDA backend — C-managed memory."""
    def __init__(self, kernel_config, batch_size=50000):
        from ampfit._cuda_v2 import CUDAKernelV2 as _K
        self.kernel = _K(kernel_config, batch_size=batch_size)
        self._phsp_np = None; self._phsp_scratch = None; self._phsp_n = 0

    def load_data(self, data_np): return self.kernel.load_data(data_np)
    def compute(self, params, data_handle, norm=None):
        return self.kernel.compute(params, data_handle, norm=norm)
    def prepare_phsp_batched(self, phsp_np, n_events):
        self._phsp_np = phsp_np; self._phsp_n = n_events
        self._phsp_scratch = self.kernel.load_data(phsp_np)
    def compute_norm_batched(self, params):
        if self._phsp_scratch is None: raise RuntimeError("No phsp loaded")
        n_b, g_b, _ = self.kernel.compute(params, self._phsp_scratch, norm=None)
        return n_b, g_b
    def free_phsp_batched(self):
        self._phsp_np = None
        if self._phsp_scratch is not None: self._phsp_scratch.free(); self._phsp_scratch = None
    def free(self): self.free_phsp_batched(); self.kernel.free()


@register_backend("cuda32_v2")
class CUDABackendV2F32(ComputeBackend):
    """v2 CUDA backend — float32."""
    dtype = np.float32
    def __init__(self, kernel_config, batch_size=50000):
        from ampfit._cuda_v2_f32 import CUDAKernelV2F32 as _K
        self.kernel = _K(kernel_config, batch_size=batch_size)
        self._phsp_np = None; self._phsp_scratch = None; self._phsp_n = 0
    def load_data(self, data_np): return self.kernel.load_data(data_np)
    def compute(self, params, data_handle, norm=None):
        return self.kernel.compute(params, data_handle, norm=norm)
    def prepare_phsp_batched(self, phsp_np, n_events):
        self._phsp_np = phsp_np; self._phsp_n = n_events
        self._phsp_scratch = self.kernel.load_data(phsp_np)
    def compute_norm_batched(self, params):
        if self._phsp_scratch is None: raise RuntimeError("No phsp loaded")
        n_b, g_b, _ = self.kernel.compute(params, self._phsp_scratch, norm=None)
        return n_b, g_b
    def free_phsp_batched(self):
        self._phsp_np = None
        if self._phsp_scratch is not None: self._phsp_scratch.free(); self._phsp_scratch = None
    def free(self): self.free_phsp_batched(); self.kernel.free()


# ── v3 backends (Catmull-Rom interpolation) ───────────────────

@register_backend("cuda")
@register_backend("cuda64")
@register_backend("cuda_v3")
@register_backend("cuda64_v3")
class CUDABackendV3(ComputeBackend):
    """v3 CUDA backend — Catmull-Rom interpolation."""
    def __init__(self, kernel_config, batch_size=50000):
        from ampfit._cuda_v3 import CUDAKernelV3 as _K
        self.kernel = _K(kernel_config, batch_size=batch_size)
        self._phsp_np = None; self._phsp_scratch = None; self._phsp_n = 0
    def load_data(self, data_np): return self.kernel.load_data(data_np)
    def compute(self, params, data_handle, norm=None):
        return self.kernel.compute(params, data_handle, norm=norm)
    def prepare_phsp_batched(self, phsp_np, n_events):
        self._phsp_np = phsp_np; self._phsp_n = n_events
        self._phsp_scratch = self.kernel.load_data(phsp_np)
    def compute_norm_batched(self, params):
        if self._phsp_scratch is None: raise RuntimeError("No phsp loaded")
        n_b, g_b, _ = self.kernel.compute(params, self._phsp_scratch, norm=None)
        return n_b, g_b
    def free_phsp_batched(self):
        self._phsp_np = None
        if self._phsp_scratch is not None: self._phsp_scratch.free(); self._phsp_scratch = None
    def free(self): self.free_phsp_batched(); self.kernel.free()


@register_backend("cuda32_v3")
class CUDABackendV3F32(ComputeBackend):
    """v3 CUDA backend — float32 with Catmull-Rom."""
    dtype = np.float32
    def __init__(self, kernel_config, batch_size=50000):
        from ampfit._cuda_v3_f32 import CUDAKernelV3F32 as _K
        self.kernel = _K(kernel_config, batch_size=batch_size)
        self._phsp_np = None; self._phsp_scratch = None; self._phsp_n = 0
    def load_data(self, data_np): return self.kernel.load_data(data_np)
    def compute(self, params, data_handle, norm=None):
        return self.kernel.compute(params, data_handle, norm=norm)
    def prepare_phsp_batched(self, phsp_np, n_events):
        self._phsp_np = phsp_np; self._phsp_n = n_events
        self._phsp_scratch = self.kernel.load_data(phsp_np)
    def compute_norm_batched(self, params):
        if self._phsp_scratch is None: raise RuntimeError("No phsp loaded")
        n_b, g_b, _ = self.kernel.compute(params, self._phsp_scratch, norm=None)
        return n_b, g_b
    def free_phsp_batched(self):
        self._phsp_np = None
        if self._phsp_scratch is not None: self._phsp_scratch.free(); self._phsp_scratch = None
    def free(self): self.free_phsp_batched(); self.kernel.free()

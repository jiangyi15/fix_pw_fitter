"""NumPy backend — pure CPU computation (float64, reference)."""
import numpy as np
from .core import ComputeBackend, register_backend


@register_backend("numpy")
class NumpyBackend(ComputeBackend):
    """Pure NumPy computation (f64, CPU)."""
    def __init__(self, kernel_config):
        from ampfit.numpy_kernel import NumpyKernelCorrect
        self.kernel = NumpyKernelCorrect(kernel_config)

    def load_data(self, data_np):
        return data_np  # numpy dict is its own handle

    def compute(self, params, data_handle, norm=None):
        return self.kernel._compute(params, data_handle, norm=norm)

"""
PWA GPU - CUDA-accelerated partial wave analysis.

Architecture:
  PWAGPU   - GPU fitter holding configuration (indices, tables, matrixes on GPU)
  PWAData  - per-event data on GPU (one per dataset)
  PWAFitter - reference numpy implementation (from pwa_gpu.numpy_ref)

Usage:
  from pwa_gpu import PWAGPU, PWAData

  fitter = PWAGPU(config)
  data = PWAData(fitter, mass, q, angles, time, frac, weights, bkg)
  q_val, grads = fitter.compute(params, data, N)
"""

from pwa_gpu.core import PWAGPU, PWAData, compare_with_numpy

__all__ = ["PWAGPU", "PWAData", "compare_with_numpy"]

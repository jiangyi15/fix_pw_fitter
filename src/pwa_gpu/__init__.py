"""
PWA GPU - CUDA-accelerated partial wave analysis.

Architecture:
  PWAGPU    - GPU fitter holding configuration on GPU
  PWAData   - per-event data on GPU
  PWAFitter - unified entry point (config → tables → GPU → compute)

Usage:
  from pwa_gpu import PWAFitter

  fitter = PWAFitter("config.yml")
  fitter.build_tables()
  fitter.load_params("a.json")
  data = fitter.load_data(mass, q, angles, time, frac, weights, bkg)
  q_val, grads = fitter.compute(data, N=50000)
"""

from pwa_gpu.core import PWAGPU, PWAData, compare_with_numpy
from pwa_gpu.fitter import PWAFitter

__all__ = ["PWAGPU", "PWAData", "compare_with_numpy", "PWAFitter"]

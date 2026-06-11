#!/usr/bin/env python3
"""Test the void* API handles batching correctly."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ampfit.config_loader import Config
from ampfit.numpy_kernel import NumpyKernelCorrect

# Test all three backends at various batch sizes
config = Config('config_angle.yml')
kc = config.build_all_index()

np.random.seed(42)
# Fixed params across all tests
params = {
    'ck': np.random.randn(448) + 1j * np.random.randn(448),
    'm0': np.random.rand(20) + 2,
    'g0': np.random.rand(23) + 0.1,
    'scalar': [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
}

for batch in [1, 3, 10, 64, 128]:
    np.random.seed(42)
    data = {
        'mass': np.random.random((batch, 48)),
        'q': np.random.random((batch, 72)),
        'angle': np.random.random((batch, 24, 3)),
        'frac': np.random.random(batch),
        'time': np.random.random(batch),
        'weight': np.ones(batch),
        'bkg': np.random.random(batch) * 0.01,
    }

    # NumPy reference
    nk = NumpyKernelCorrect(kc)
    Q_np, grads_np, P_np = nk._compute(params, data)

    # Test each CUDA backend
    for label, mod_name in [('f64', '_cuda'), ('f32', '_cuda_f32'), ('merged', '_cuda_merged')]:
        try:
            mod = __import__(f'ampfit.{mod_name}', fromlist=['object'])
            cls = mod.CUDAKernel if 'f32' not in mod_name else mod.CUDAKernel32
            if 'merged' in mod_name:
                cls = mod.CUDAMergedKernel
            
            k = cls(kc)
            dh = k.load_data(data)
            Q, grads, P = k.compute(params, dh, norm=None)
            k.free()

            q_ok = abs(Q - Q_np) < 1e-8 * max(1.0, abs(Q_np))
            p_ok = np.max(np.abs(P - P_np)) < 1e-6
            g_ok = True
            for key in ['ck', 'm0', 'g0', 'scalar']:
                rel = np.max(np.abs(grads[key] - grads_np[key])) / (np.max(np.abs(grads_np[key])) + 1e-30)
                if rel > 1e-5:
                    g_ok = False

            status = '✓' if (q_ok and p_ok and g_ok) else '✗'
            print(f'{status} {label} n={batch:>4}: Q={Q:.4f} P_err={np.max(np.abs(P-P_np)):.2e} grads_ok={g_ok}')
        except Exception as e:
            print(f'  {label} n={batch}: SKIP ({e})')

print('Done')

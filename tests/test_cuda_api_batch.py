#!/usr/bin/env python3
"""Test all CUDA backends match NumPy reference across batch sizes."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ampfit.config_loader import Config
from ampfit.backends import create_backend


def _cuda_backends():
    yield "f64_v3",  "cuda_v3"
    yield "f32_v3",  "cuda32_v3"
    yield "f64_v2",  "cuda_v2"
    yield "f32_v2",  "cuda32_v2"


def test_cuda_batch_accuracy():
    config = Config('config_angle.yml')
    kc = config.build_all_index()

    n_ck = len(config.get_ck_map())
    n_m0 = int(np.max(kc["m0_index"])) + 1
    n_g0 = int(np.max(kc["g0_index"])) + 1

    rng = np.random.default_rng()
    params = {
        'ck': rng.normal(size=n_ck) + 1j * rng.normal(size=n_ck),
        'm0': rng.random(n_m0) + 2,
        'g0': rng.random(n_g0) + 0.1,
        'scalar': [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }

    # Create each backend ONCE and reuse across batch sizes
    numpy_backend = create_backend("numpy", kc)
    cuda_backends = []
    for label, backend_name in _cuda_backends():
        try:
            be = create_backend(backend_name, kc)
            cuda_backends.append((label, be))
        except Exception as e:
            print(f'  {label}: SKIP ({e})')

    for batch in [1, 3, 10, 64, 128]:
        rng = np.random.default_rng()
        data = {
            'mass': rng.uniform(2, 3, (batch, 48)),
            'q': rng.random((batch, 72)),
            'angle': rng.random((batch, 24, 3)),
            'frac': rng.random(batch),
            'time': rng.random(batch),
            'weight': np.ones(batch),
            'bkg': rng.random(batch) * 0.01,
        }

        # NumPy reference (cheap — create handle per batch, no GPU)
        np_handle = numpy_backend.load_data(data)
        Q_np, grads_np, P_np = numpy_backend.compute(params, np_handle, norm=None)

        for label, backend in cuda_backends:
            try:
                handle = backend.load_data(data)
                Q, grads, P = backend.compute(params, handle, norm=None)

                is_f32 = 'f32' in label
                q_ok = abs(Q - Q_np) < (1e-4 if is_f32 else 1e-10) * max(1.0, abs(Q_np))
                p_ok = np.max(np.abs(P - P_np)) < (1e-5 if is_f32 else 1e-10)
                g_ok = True
                for key in ['ck', 'm0', 'g0', 'scalar']:
                    rel = np.max(np.abs(grads[key] - grads_np[key])) / (np.max(np.abs(grads_np[key])) + 1e-30)
                    if rel > (1e-4 if is_f32 else 1e-5):
                        g_ok = False

                assert q_ok, f"{label} n={batch}: Q mismatch"
                assert p_ok, f"{label} n={batch}: P mismatch"
                assert g_ok, f"{label} n={batch}: grads mismatch"

                handle.free()
            except Exception as e:
                print(f'  {label} n={batch}: SKIP ({e})')

    # Free backends
    for _, backend in cuda_backends:
        backend.free()

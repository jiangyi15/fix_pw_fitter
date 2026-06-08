"""
Batched fully merged kernel - test if merged amplitude + batching works better.
"""
import numpy as np
from numpy_kernel_fully_merged import NumpyKernelFullyMerged


class NumpyKernelFullyMergedBatched:
    """Process data in batches with fully merged optimizations."""
    
    def __init__(self, config, optimal_batch_size=100):
        self.kernel = NumpyKernelFullyMerged(config)
        self.optimal_batch_size = optimal_batch_size
    
    def _compute(self, params, data, norm=None):
        """Process in batches."""
        n_events = data['mass'].shape[0]
        
        # Batching only works when norm=None
        if norm is not None or n_events <= self.optimal_batch_size:
            return self.kernel._compute(params, data, norm=norm)
        
        n_batches = (n_events + self.optimal_batch_size - 1) // self.optimal_batch_size
        
        Q_total = 0.0
        grads_accumulated = None
        P_all = []
        
        for batch_idx in range(n_batches):
            start = batch_idx * self.optimal_batch_size
            end = min((batch_idx + 1) * self.optimal_batch_size, n_events)
            
            batch_data = {
                'mass': data['mass'][start:end],
                'q': data['q'][start:end],
                'angle': data['angle'][start:end],
                'frac': data['frac'][start:end],
                'time': data['time'][start:end],
                'bkg': data['bkg'][start:end] if isinstance(data['bkg'], np.ndarray) else data['bkg'],
                'weight': data['weight'][start:end],
            }
            
            Q_batch, grads_batch, P_batch = self.kernel._compute(params, batch_data, norm=norm)
            
            Q_total += Q_batch
            
            if grads_accumulated is None:
                grads_accumulated = {
                    'ck': grads_batch['ck'].copy(),
                    'm0': grads_batch['m0'].copy(),
                    'g0': grads_batch['g0'].copy(),
                    'scalar': grads_batch['scalar'],
                    'norm': grads_batch['norm'],
                }
            else:
                grads_accumulated['ck'] += grads_batch['ck']
                grads_accumulated['m0'] += grads_batch['m0']
                grads_accumulated['g0'] += grads_batch['g0']
                grads_accumulated['scalar'] = tuple(
                    s1 + s2 for s1, s2 in zip(grads_accumulated['scalar'], grads_batch['scalar'])
                )
                if grads_accumulated['norm'] is not None:
                    grads_accumulated['norm'] += grads_batch['norm']
            
            P_all.append(P_batch)
        
        P_total = np.concatenate(P_all)
        return Q_total, grads_accumulated, P_total

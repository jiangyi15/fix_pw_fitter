"""
Batched NumPy kernel - process data in small batches for better cache performance.

Key insight: Small batches (100 events) get 1.79x speedup vs 1.33x for large batches.
Solution: Split data into batches, process each, sum results.

Mathematical justification:
- Loss: Q = Σ weight_i * P_i is additive across batches
- Gradients: ∇Q = Σ ∇Q_i is additive (chain rule)
"""
import numpy as np
from numpy_kernel_selective_cache import NumpyKernelSelectiveCache


class NumpyKernelBatched:
    """Process data in optimal batch sizes for better cache performance."""
    
    def __init__(self, config, optimal_batch_size=100):
        """
        Args:
            config: Kernel configuration
            optimal_batch_size: Batch size with best cache performance (default: 100)
        """
        self.kernel = NumpyKernelSelectiveCache(config)
        self.optimal_batch_size = optimal_batch_size
    
    def _compute(self, params, data, norm=None):
        """
        Process data in batches for better cache performance.
        
        NOTE: Batching only works correctly when norm=None because:
        - norm=None: Q = sum(weight * P) → additive across batches ✓
        - norm≠None: Q = -sum(log(P/norm + bkg)) → NOT additive ✗
        
        When norm is provided, processes all data at once (no batching).
        
        Returns:
            Q: Total loss (sum across batches)
            grads: Combined gradients (sum across batches)
            P: All probabilities concatenated
        """
        n_events = data['mass'].shape[0]
        
        # Batching only works when norm=None (additive loss)
        # When norm is provided, must process all data together
        if norm is not None or n_events <= self.optimal_batch_size:
            return self.kernel._compute(params, data, norm=norm)
        
        # Process in batches
        n_batches = (n_events + self.optimal_batch_size - 1) // self.optimal_batch_size
        
        Q_total = 0.0
        grads_accumulated = None
        P_all = []
        
        for batch_idx in range(n_batches):
            start = batch_idx * self.optimal_batch_size
            end = min((batch_idx + 1) * self.optimal_batch_size, n_events)
            
            # Extract batch data
            batch_data = {
                'mass': data['mass'][start:end],
                'q': data['q'][start:end],
                'angle': data['angle'][start:end],
                'frac': data['frac'][start:end],
                'time': data['time'][start:end],
                'bkg': data['bkg'][start:end] if isinstance(data['bkg'], np.ndarray) else data['bkg'],
                'weight': data['weight'][start:end],
            }
            
            # Process batch
            Q_batch, grads_batch, P_batch = self.kernel._compute(params, batch_data, norm=norm)
            
            # Accumulate loss
            Q_total += Q_batch
            
            # Accumulate gradients (gradients are additive)
            if grads_accumulated is None:
                grads_accumulated = {
                    'ck': grads_batch['ck'].copy(),
                    'm0': grads_batch['m0'].copy(),
                    'g0': grads_batch['g0'].copy(),
                    'scalar': grads_batch['scalar'],  # Tuple, no need to copy
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
        
        # Concatenate all probabilities
        P_total = np.concatenate(P_all)
        
        return Q_total, grads_accumulated, P_total

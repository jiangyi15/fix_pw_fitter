"""cpu_v5_pwa backend — CPU (C + OpenMP) general pure-PWA with the v5
grouped log-sum data NLL.

CPU counterpart of ``cuda_v5_pwa``: the same general (any ``n_res``)
projection-sum PWA as ``cpu_pwa_gen``, but in the data-NLL branch events
are grouped into contiguous blocks of ``resolution_size`` and one log is
taken per group,

    Q = -Σ_groups log Σ_{e∈g} w_e·(P_e/norm + bkg_e)

The phase-space path (``norm=None``) and the per-event returned ``P`` are
unchanged.  ``resolution_size <= 1`` takes the exact v4 per-event path, so
it is bit-identical to ``cpu_pwa_gen`` (which is this same kernel at
``resolution_size=1``) for any weights.  ``d(NLL)/d(norm)`` is accumulated
per group on the C path and exposed through ``grads["norm"]``
(``dnorm_on_gpu``).
"""
from .core import ComputeBackend, register_backend


@register_backend("cpu_v5_pwa", model="pwa")
@register_backend("cpu_v5_pwa_gen", model="pwa")
class CPUPWAV5Backend(ComputeBackend):
    """CPU PWA grouped-log-sum backend (C + OpenMP, float64).

    ``resolution_size`` (kwarg, default 1) is the number of contiguous
    events per log-sum group.  At ``resolution_size == 1`` the NLL and
    gradients are bit-identical to ``cpu_pwa_gen`` (the v4 per-event
    objective) for any weights.
    """

    def __init__(self, kernel_config, batch_size=50000, resolution_size=1):
        self.kernel = None   # free/__del__ must be safe if the build fails
        self._resolution_size = int(resolution_size)
        from tabpwa.cpu._pwa_v5 import CPUKernelPWAV5 as K
        self.kernel = K(kernel_config, batch_size=batch_size,
                        resolution_size=self._resolution_size)

    @property
    def dnorm_on_gpu(self):
        return True

    def load_data(self, data_np):
        return self.kernel.load_data(data_np)

    def compute(self, params, data_handle, norm=None, return_p=True):
        Q, grads, P = self.kernel.compute(params, data_handle, norm=norm,
                                          return_p=return_p)
        if norm is not None:
            dn = self.kernel._last_dnorm
            if dn is None:
                raise RuntimeError(
                    "cpu_v5_pwa kernel did not provide _last_dnorm")
            grads = dict(grads)
            grads["norm"] = float(dn)
        return Q, grads, (P if return_p else None)

    def free(self):
        if self.kernel is not None:
            self.kernel.free()
            self.kernel = None

    def __del__(self):
        self.free()

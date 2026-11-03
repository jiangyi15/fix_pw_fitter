"""
CPU pure-PWA kernel on an opaque basis with a general BW product per wave
(C + OpenMP, cffi) — **v4 preset of the v5 kernel at ``resolution_size=1``**.

This module is a THIN ALIAS.  The whole implementation (data handle, the
opaque-basis ``common``/``A_p`` tile, both gradient phases and the host
chain dQ/ds0, dQ/dmult → dQ/dm0, dQ/dg0) lives in
:mod:`tabpwa.cpu._pwa_v5`, whose ``resolution_size <= 1`` branch is the
exact v4 per-event NLL

    Q        = −Σ_e w_e·log(P_e/norm + b_e)
    dQ/dP_e  = −w_e/(norm·(P_e/norm + b_e))
    dQ/dnorm = Σ_e w_e·(P_e/norm²)/(P_e/norm + b_e)

— bit-identical to the former standalone ``kernels_cpu_pwa_gen.c``
(kernel policy case 3: a green case-1 copy supersedes the old kernel; the
old source is archived at ``cpu/archive/kernels_cpu_pwa_gen_v1.c`` for
rollback and is no longer built).  ``cpu_pwa_gen`` and ``cpu_v5_pwa``
therefore share ONE C kernel and ONE shared library.

Model (any ``n_res >= 1``):

    P(e) = Σ_p |A_p(e)|² ,   A_p(e) = Σ_k ck_k·common[e, p·N+k]
    common[e,w] = amp[e,w] / BW_w(e),   BW_w(e) = Π_r bw_{w,r}(e)

The C kernel works on an **opaque basis**: it knows nothing about s0 or
about what a coefficient means.  Each (wave, factor) contributes
``n_basis`` slots

    bw_{w,r}(e) = Σ_i coeff[(w·n_res+r)·n_basis + i]·F[slot_map[… i]](e)
    ∂Q/∂coeff[(w,r,i)] = 2·Re( dBW_w·(1/bw_{w,r})·F_i )
    dBW_w = −(dQ/dcommon_w)·common_w

so a wave can carry one basis function (a plain coefficient) or a whole
denominator expansion — the kernel code is identical.  The table stores
the finished function F (the −i·γ rotation, the −m² sign and the constant
1 are all baked in at load), so the kernel never rotates anything.

The Python side builds the **s0c preset of every factor** of that basis

    slot 0   : F =  1          coeff = m0_{w,r}²       (the pole term s0)
    slot 1   : F = −m²         coeff = +1              (a pure indicator)
    slot 2+j : F = −i·γ_j(m)   coeff = m0_{w,r}·g0_j   (running-width terms)

and chains dQ/dcoeff back to the fitted m0/g0 (exactly like the CUDA
siblings), so no m0/g0 algebra enters the kernel.  At ``n_res == 1`` this
module is **bit-identical** to :mod:`tabpwa.cpu._pwa_s0c` (the single
factor folds exactly through the product).

Iteration-invariant per-event inputs are precomputed once at load time in
NumPy (the CPU analogue of the CUDA τ-u-cache) — see :mod:`_pwa_v5`:

    amp[e,w] = (ka·matrix_angle)·Π fl        (angular basis × form factors)
    F[u,e]   = basis function at unique slot u (1, −m², −i·γ interp)

Performance note: on SMT CPUs set ``OMP_NUM_THREADS`` to the number of
**physical** cores (16 on a 16C/32T Ryzen 9); using all logical threads was
measured slower (memory/FP64 contention).
"""

from tabpwa.cpu._pwa_v5 import (          # noqa: F401 — re-exported
    DataHandle,
    CPUKernelPWAV5,
    _ffi,
)


class CPUKernelPWAGen(CPUKernelPWAV5):
    """General (any n_res) projection-sum PWA — the v4 per-event NLL.

    Identical to :class:`tabpwa.cpu._pwa_v5.CPUKernelPWAV5` at
    ``resolution_size=1``; kept as a named class so existing callers
    (``cpu_pwa_gen`` / ``cpu_v4_pwa``) need no change.
    """

    def __init__(self, config, batch_size=50000, lib_path=None):
        super().__init__(config, batch_size=batch_size, resolution_size=1,
                         lib_path=lib_path)

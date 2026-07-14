"""Pre‑integrated hyper backend — Gram matrix norm + base backend for data NLL.

A *hyper backend* that pre‑computes the phase‑space normalization integral
as Gram matrices (fast O(n²) per iteration) while delegating the data
negative log‑likelihood computation to a configurable *base backend*
(e.g. ``"numpy"``, ``"cuda64"``, ``"onnx_cpu"``).

Usage::

    # Create directly
    backend = create_backend("integrated", kernel_config,
                             base_backend="cuda64")

    # Or via Fitter (defaults to numpy base)
    fitter = Fitter("config.yml", backend="integrated")
"""
import numpy as np
from .core import ComputeBackend, register_backend, create_backend


@register_backend("integrated")
class IntegratedBackend(ComputeBackend):
    """Hyper backend: Gram‑matrix norm + base backend for data NLL.

    Parameters
    ----------
    kernel_config : dict
        Kernel configuration from ``Config.build_all_index()``.
    base_backend : str or ComputeBackend, optional
        Backend used for data NLL computation.  A string is resolved
        via :func:`create_backend`; an instance is used directly.
        Default ``"numpy"``.

    Attributes
    ----------
    base : ComputeBackend
        The underlying backend for data NLL.
    M_pp, M_mm, M_pm : (n_basis, n_basis) complex
        Pre‑computed Gram matrices.
    int_Ap2, int_Am2, int_ApAm:
        Spatial integrals for the latest call (user‑accessible).
    """

    def __init__(self, kernel_config, base="cuda_v3_cache", strict_gram=True):
        from ampfit.numpy_kernel import NumpyKernel
        self.kernel = NumpyKernel(kernel_config)
        self._kernel_config = kernel_config
        self.strict_gram = strict_gram

        # ── Base backend for data NLL ─────────────────────────────
        if isinstance(base, ComputeBackend):
            self.base = base
        else:
            from . import create_backend
            self.base = create_backend(base, kernel_config)

        # ── Reduced Gram matrices (from 4-group structure: n_wave//8 groups)
        self._ng = self.kernel.n_wave // 8
        self._groups_B0 = [list(range(k, self.kernel.n_wave // 2, self._ng))
                           for k in range(self._ng)]
        self._groups_B0bar = [list(range(k, self.kernel.n_wave // 2, self._ng))
                              for k in range(self._ng)]

        # ── Auto‑detect gram backend from base's kernel ──────────
        # If the base backend's kernel has a compute_gram() method,
        # use it for GPU‑accelerated Gram matrix pre‑computation.
        # Otherwise fall back to the numpy batched path.
        base_kernel = getattr(self.base, "kernel", None)
        self._gram_compute = getattr(base_kernel, "compute_gram", None)

        # User‑accessible integral values
        self.int_Ap2 = None
        self.int_Am2 = None
        self.int_ApAm = None

        # ── Time average caching (per-array + scalar result) ──
        self._cached_Gam = None
        self._cached_DG = None
        self._cached_Dm = None
        self._cache_expt = None
        self._cache_cht = None
        self._cache_sht = None
        self._cache_ct = None
        self._cache_st = None
        self._ta_key = None
        self._ta_result = None

    # ═══════════════════════════════════════════════════════════════
    #  Phsp pre‑integration
    # ═══════════════════════════════════════════════════════════════

    def _gp_gm(self, t, scalar):
        """Compute gp(t), gm(t) arrays.  (N,) complex128 each."""
        Gam, DG, Dm, Ap, r, phi = scalar
        eL = np.exp(-1j * t * (-Dm/2 - 1j * (Gam + DG/2) / 2))
        eH = np.exp(-1j * t * (+Dm/2 - 1j * (Gam - DG/2) / 2))
        return (eL + eH) / 2, (eL - eH) / 2

    def _compute_time_averages(self, scalar, pb):
        """Compute time averages via GPU if available, else CPU.
        *pb* is the :class:`_PhspBundle` containing the phsp data."""
        base_kernel = getattr(self.base, "kernel", None)
        gpu_ta = getattr(base_kernel, "compute_time_averages", None)
        if gpu_ta is not None:
            return gpu_ta(pb.handle, scalar)
        return self._time_averages_cpu(scalar, pb)[:11]

    def _time_averages_cpu(self, scalar, pb):
        """Time integrals with per-array caching keyed by scalar params."""
        Gam, DG, Dm = scalar[0], scalar[1], scalar[2]
        t = pb.time
        w = pb.weight
        ws = np.sum(w)

        # Per-array cache (recompute only changed params)
        eps = 1e-15
        if self._cached_Gam is None or abs(Gam - self._cached_Gam) > eps:
            self._cache_expt = np.exp(-Gam * t)
            self._cached_Gam = Gam
        if self._cached_DG is None or abs(DG - self._cached_DG) > eps:
            self._cache_cht = np.cosh(DG * t / 2)
            self._cache_sht = np.sinh(DG * t / 2)
            self._cached_DG = DG
        if self._cached_Dm is None or abs(Dm - self._cached_Dm) > eps:
            self._cache_ct = np.cos(Dm * t)
            self._cache_st = np.sin(Dm * t)
            self._cached_Dm = Dm

        # Check if final scalar averages are cached
        key = (Gam, DG, Dm)
        if self._ta_key is not None and self._ta_key == key:
            return self._ta_result

        expt = self._cache_expt
        cht = self._cache_cht
        sht = self._cache_sht
        ct = self._cache_ct
        st = self._cache_st

        def avg(f): return float(np.sum(w * f) / ws)
        def avg_c(f): return complex(np.sum(w * f) / ws)

        icht = avg(cht * expt)
        ict = avg(ct * expt)
        isht = avg(sht * expt)
        ist = avg(st * expt)

        gp2_avg = (icht + ict) / 2
        gm2_avg = (icht - ict) / 2
        gpgm_avg = (-isht + 1j * ist) / 2

        # Derivative time integrals (for scalar gradient chain)
        idcht = avg(t * expt * sht) / 2       # d(icht)/dDG
        idct = -avg(t * expt * st)            # d(ict)/dDm
        idsht = avg(t * expt * cht) / 2       # d(isht)/dDG
        idst = avg(t * expt * ct)             # d(ist)/dDm

        self._ta_result = (gp2_avg, gm2_avg, gpgm_avg,
                           icht, ict, isht, ist,
                           idcht, idct, idsht, idst,
                           expt, cht, sht, ct, st, t, w, ws)
        self._ta_key = key
        return self._ta_result

    def _load_phsp_matrices(self, phsp, m0, g0, phsp_handle=None):
        """Compute ng×ng reduced Gram matrices directly (no full matrix).

        Groups of 4 basis functions (spaced ng apart) are summed
        before the outer product, reducing 4·ng×4·ng → ng×ng.

        If *phsp_handle* is provided, reuses it instead of uploading
        the phsp data again (avoids redundant GPU memory allocation).
        """
        n_events = phsp["mass"].shape[0]
        n_wave = self.kernel.n_wave
        n = n_wave // 2
        ng = self._ng  # number of groups

        # ── CUDA accelerated path (via base backend's kernel) ─────
        if self._gram_compute is not None:
            w = np.asarray(phsp.get("weight", np.ones(n_events)), dtype=np.float64)
            # Reuse existing GPU handle if available, otherwise upload
            if phsp_handle is not None:
                dh = phsp_handle
            else:
                cuda_phsp = {k: v for k, v in phsp.items() if isinstance(v, np.ndarray)}
                cuda_phsp.setdefault("frac", np.ones(n_events) * 0.5)
                cuda_phsp.setdefault("time", np.zeros(n_events))
                cuda_phsp.setdefault("bkg_raw", np.zeros(n_events))
                dh = self.base.load_data(cuda_phsp)
            Mpp, Mmm, Mpm = self._gram_compute(dh, m0, g0)
            if phsp_handle is None:
                dh.free()
            self._Mpp_r = (Mpp + Mpp.conj().T) / 2
            self._Mmm_r = (Mmm + Mmm.conj().T) / 2
            self._Mpm_r = Mpm
            self.phsp_n = n_events
            self._phsp_weights = w
            self._phsp_frac = np.asarray(
                phsp.get("frac", np.ones(n_events)), dtype=np.float64)
            self._phsp_time = np.asarray(
                phsp.get("time", np.zeros(n_events)), dtype=np.float64)
            return

        # ── NumPy fallback (batched) ───────────────────────────────
        w = np.asarray(phsp.get("weight", np.ones(n_events)), dtype=np.float64)
        sw = np.sqrt(w)
        bs = 500
        n_batches = (n_events + bs - 1) // bs
        import sys, time as _time
        _t0 = _time.time()
        Mpp = np.zeros((ng, ng), dtype=complex)
        Mmm = np.zeros((ng, ng), dtype=complex)
        Mpm = np.zeros((ng, ng), dtype=complex)
        for b_start in range(0, n_events, bs):
            b_idx = b_start // bs
            b_end = min(b_start + bs, n_events)
            batch = {k: v[b_start:b_end] for k, v in phsp.items()
                     if isinstance(v, np.ndarray)}
            ba = self.kernel._compute_common_amp_factor(batch, m0=m0, g0=g0)
            # Project onto groups: sum 4 blocks of ng → (batch, ng)
            A0 = ba[:, :n].reshape(-1, 4, ng).sum(axis=1) * sw[b_start:b_end, np.newaxis]
            A1 = ba[:, n:].reshape(-1, 4, ng).sum(axis=1) * sw[b_start:b_end, np.newaxis]
            Mpp += A0.T.conj() @ A0
            Mmm += A1.T.conj() @ A1
            Mpm += A0.T.conj() @ A1
            _elapsed = _time.time() - _t0
            _eta = _elapsed / (b_idx + 1) * (n_batches - b_idx - 1)
            sys.stdout.write(
                f"\r  Gram matrix: batch {b_idx + 1}/{n_batches}  "
                f"[{_elapsed:.1f}s"
                f"{f', {_eta:.1f}s ETA' if _eta > 1 else ''}]   ")
            sys.stdout.flush()
        sys.stdout.write("\n")
        self._Mpp_r = (Mpp + Mpp.conj().T) / 2
        self._Mmm_r = (Mmm + Mmm.conj().T) / 2
        self._Mpm_r = Mpm  # not necessarily symmetric
        self.phsp_n = n_events
        self._phsp_weights = w
        self._phsp_frac = np.asarray(
            phsp.get("frac", np.ones(n_events)), dtype=np.float64)
        self._phsp_time = np.asarray(
            phsp.get("time", np.zeros(n_events)), dtype=np.float64)

    def _n_m0(self):
        """Number of unique BW mass parameters."""
        return int(np.max(self.kernel.m0_index)) + 1

    def _n_g0(self):
        """Number of unique BW width parameters."""
        return int(np.max(self.kernel.g0_index)) + 1

    # ═══════════════════════════════════════════════════════════════
    #  ComputeBackend interface
    # ═══════════════════════════════════════════════════════════════

    class _PhspBundle:
        """Wrapper bundling a GPU data handle with the numpy arrays needed for
        Gram matrices and time averages.  Returned by :meth:`load_data`."""
        def __init__(self, handle, data_np):
            self.handle = handle
            n = data_np["mass"].shape[0]
            self.n_events = n
            self.mass = data_np.get("mass")
            self.q = data_np.get("q")
            self.angle = data_np.get("angle")
            self.weight = np.asarray(
                data_np.get("weight", np.ones(n)), dtype=np.float64)
            self.frac = np.asarray(
                data_np.get("frac", np.ones(n) * 0.5), dtype=np.float64)
            self.time = np.asarray(
                data_np.get("time", np.zeros(n)), dtype=np.float64)
            self.bkg = np.asarray(
                data_np.get("bkg_raw",
                            data_np.get("bkg", np.zeros(n))), dtype=np.float64)
            # Gram matrices (lazily built by IntegratedBackend.ensure_gram)
            self.m0 = self.g0 = None
            self.Mpp = self.Mmm = self.Mpm = None

        def free(self):
            if hasattr(self.handle, 'free'):
                self.handle.free()
            self.Mpp = self.Mmm = self.Mpm = None

    def ensure_gram(self, bundle, m0, g0):
        """Build Gram matrices on *bundle* for *m0*, *g0* if not cached."""
        if bundle.Mpp is not None:
            if (np.array_equal(m0, bundle.m0) and
                np.array_equal(g0, bundle.g0)):
                return
        from ampfit.numpy_kernel import NumpyKernel
        nk = NumpyKernel(self._kernel_config)
        ng = nk.n_wave // 8
        n = nk.n_wave // 2
        if self._gram_compute is not None:
            Mpp, Mmm, Mpm = self._gram_compute(bundle.handle, m0, g0)
        else:
            ne = bundle.n_events
            sw = np.sqrt(bundle.weight)
            A0_all = np.empty((ne, ng), dtype=complex)
            A1_all = np.empty((ne, ng), dtype=complex)
            for b_start in range(0, ne, 500):
                b_end = min(b_start + 500, ne)
                batch = {k: getattr(bundle, k)[b_start:b_end]
                         for k in ("mass", "q", "angle")}
                ba = nk._compute_common_amp_factor(batch, m0=m0, g0=g0)
                A0_all[b_start:b_end] = (ba[:, :n].reshape(-1, 4, ng).sum(axis=1)
                                         * sw[b_start:b_end, None])
                A1_all[b_start:b_end] = (ba[:, n:].reshape(-1, 4, ng).sum(axis=1)
                                         * sw[b_start:b_end, None])
            Mpp = A0_all.T.conj() @ A0_all
            Mmm = A1_all.T.conj() @ A1_all
            Mpm = A0_all.T.conj() @ A1_all
        bundle.Mpp = (Mpp + Mpp.conj().T) / 2
        bundle.Mmm = (Mmm + Mmm.conj().T) / 2
        bundle.Mpm = Mpm
        bundle.m0 = m0.copy()
        bundle.g0 = g0.copy()

    def load_data(self, data_np):
        """Load data.  Returns a :class:`_PhspBundle` holding the GPU handle
        together with the raw numpy arrays.

        The Fitter calls this twice — once for phsp (via *set_phsp*)
        and once for data (via *set_data*).  Both return a bundle; the
        :meth:`compute` method uses the bundle from the *phsp* call for
        Gram‑matrix normalisation and the bundle from the *data* call for
        negative log‑likelihood computation.
        """
        h = self.base.load_data(data_np)
        return self._PhspBundle(h, data_np)

    def compute(self, params, data_handle, norm=None, return_p=True):
        """Forward / backward pass.

        * *norm=float* — data NLL → delegate to base backend.
        * *norm=None, return_p=True*  — per-event P → delegate to base (plot).
        * *norm=None, return_p=False* — fast norm from Gram matrices.

        m0/g0 gradients are always zero (fixed at Gram pre‑computation).
        """
        pb = data_handle  # always a _PhspBundle from load_data
        dh = pb.handle

        if norm is not None or return_p:
            Q, grads, P = self.base.compute(params, dh, norm=norm,
                                            return_p=return_p)
            grads["m0"] = np.zeros_like(grads["m0"])
            grads["g0"] = np.zeros_like(grads["g0"])
            return Q, grads, P

        # ── Fast norm from pre‑integrated Gram matrices ──────────
        if pb is None:
            raise RuntimeError("IntegratedBackend: phsp not loaded.")

        m0 = np.asarray(params["m0"])
        g0 = np.asarray(params["g0"])
        self.ensure_gram(pb, m0, g0)

        ck = np.asarray(params["ck"], dtype=complex)
        n = len(ck) // 2
        ck_B0, ck_B0bar = ck[:n], ck[n:]

        Gamma, Delta_Gamma, Delta_m, A_prod, poqr, poqi = params["scalar"]
        poq = complex(poqr, poqi)
        poq2 = poq * poq.conjugate()

        # ── Spatial integrals (reduced 56×56) ─────────────────────
        g_B0 = np.array([ck_B0[grp[0]] for grp in self._groups_B0])
        g_B1 = np.array([ck_B0bar[grp[0]] for grp in self._groups_B0bar])
        I_pp = g_B0.conj() @ pb.Mpp @ g_B0
        I_mm = g_B1.conj() @ pb.Mmm @ g_B1
        I_pm = g_B0.conj() @ pb.Mpm @ g_B1

        self.int_Ap2 = float(I_pp.real)
        self.int_Am2 = float(I_mm.real)
        self.int_ApAm = complex(I_pm)

        # Time averages via unified method (GPU if available, else CPU)
        (gp2_avg, gm2_avg, gpgm_avg,
         icht, ict, isht, ist,
         idcht, idct, idsht, idst) = self._compute_time_averages(params["scalar"], pb)

        # Combine
        frac_avg = float(np.sum(pb.weight * pb.frac))
        omf = 1.0 - frac_avg
        A_co = frac_avg * (1.0 - A_prod)
        B_co = omf * (1.0 + A_prod)

        norm_val = (A_co * gp2_avg * I_pp.real
                    + A_co * poq2 * gm2_avg * I_mm.real
                    + A_co * 2.0 * (poq * gpgm_avg * I_pm).real
                    + B_co * gm2_avg / poq2 * I_pp.real
                    + B_co * gp2_avg * I_mm.real
                    + B_co * 2.0 * (gpgm_avg.conjugate()
                                    / poq.conjugate() * I_pm).real)

        # ── ck gradient (reduced: each member gets 1/n of the group gradient) ──
        C_pp = A_co * gp2_avg + B_co * gm2_avg / poq2
        C_mm = A_co * poq2 * gm2_avg + B_co * gp2_avg
        z_total = complex(A_co * poq * gpgm_avg
                          + B_co * gpgm_avg.conjugate() / poq.conjugate())

        grad_ck = np.empty_like(ck, dtype=complex)
        g_B0 = np.array([ck_B0[grp[0]] for grp in self._groups_B0])
        g_B1 = np.array([ck_B0bar[grp[0]] for grp in self._groups_B0bar])
        grad_B0_red = (C_pp * (pb.Mpp @ g_B0)
                       + z_total * (pb.Mpm @ g_B1)).conj()
        grad_B1_red = (C_mm * (pb.Mmm @ g_B1)).conj() \
                      + z_total * (pb.Mpm.T @ g_B0.conj())
        for gi, grp in enumerate(self._groups_B0):
            w = 1.0 / len(grp)
            for idx in grp:
                grad_ck[idx] = grad_B0_red[gi] * w
        for gi, grp in enumerate(self._groups_B0bar):
            w = 1.0 / len(grp)
            for idx in grp:
                grad_ck[n + idx] = grad_B1_red[gi] * w

        # ── scalar gradients (using derivative time integrals) ──────
        # Spatial combination coefficients (matching reference's int1..4)
        poq_c = poq.conjugate()
        int1_x = 0.5*(A_co + B_co/poq2)*I_pp.real + 0.5*(B_co + A_co*poq2)*I_mm.real
        int2_x = 0.5*(A_co - B_co/poq2)*I_pp.real + 0.5*(B_co - A_co*poq2)*I_mm.real
        int3_x = -(A_co*(poq*I_pm) + B_co*I_pm/poq_c).real
        int4_x = -(A_co*(poq*I_pm)).imag + (B_co*I_pm/poq_c).imag

        # Gradient via reference's derivative-time-integral formula
        dG = float(-(int1_x*2*idsht + int2_x*idst + int3_x*2*idcht - int4_x*idct).real)
        dDG = float((int1_x*idcht + int3_x*idsht).real)
        dDM = float((int2_x*idct + int4_x*idst).real)

        # A_prod
        d_B0_dens = (gp2_avg * I_pp.real + poq2 * gm2_avg * I_mm.real
                     + 2.0 * (poq * gpgm_avg * I_pm).real)
        d_B0bar_dens = (gm2_avg / poq2 * I_pp.real + gp2_avg * I_mm.real
                        + 2.0 * (gpgm_avg.conjugate() / poq_c * I_pm).real)
        dAp = float((-frac_avg * d_B0_dens + omf * d_B0bar_dens).real)

        # poqr, poqi (Wirtinger)
        dn_dpoq = (A_co * poq_c * gm2_avg * I_mm.real
                   - B_co * gm2_avg * poq_c / poq2**2 * I_pp.real
                   + A_co * gpgm_avg * I_pm
                   - B_co * gpgm_avg * I_pm.conjugate() / (poq * poq))
        dpoqr = float(2.0 * dn_dpoq.real)
        dpoqi = float(-2.0 * dn_dpoq.imag)

        scalar_grads = (dG, dDG, dDM, dAp, dpoqr, dpoqi)

        # m0/g0 gradients are zero (fixed at pre‑computation)
        _z_m0 = np.zeros(self._n_m0())
        _z_g0 = np.zeros(self._n_g0())

        grads = {"ck": grad_ck, "m0": _z_m0, "g0": _z_g0,
                 "scalar": scalar_grads, "norm": None}
        P = None if not return_p else np.zeros(pb.n_events if pb else 0)

        return float(norm_val.real if hasattr(norm_val, 'real')
                     else norm_val), grads, P

    def free(self):
        """Release all resources (matrices + base backend + time avg cache)."""
        self._cached_Gam = self._cached_DG = self._cached_Dm = None
        self._cache_expt = self._cache_cht = self._cache_sht = None
        self._cache_ct = self._cache_st = None
        self._ta_key = self._ta_result = None
        try:
            self.base.free()
        except Exception:
            pass

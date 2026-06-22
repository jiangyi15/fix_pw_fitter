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

    def __init__(self, kernel_config, base="numpy"):
        from ampfit.numpy_kernel import NumpyKernelCorrect
        self.kernel = NumpyKernelCorrect(kernel_config)

        # ── Base backend for data NLL ─────────────────────────────
        if isinstance(base, ComputeBackend):
            self.base = base
        else:
            from . import create_backend
            self.base = create_backend(base, kernel_config)

        # ── Gram matrices (pre‑computed from phsp) ────────────────
        self.M_pp = None   # (n, n) B0 × B0
        self.M_mm = None   # (n, n) B0bar × B0bar
        self.M_pm = None   # (n, n) B0 × B0bar
        self.phsp_n = 0
        self._gram_m0 = None   # m0 snapshot used for current Gram matrices
        self._gram_g0 = None

        # Raw phsp arrays stored by prepare_phsp_batched — used for
        # deferred Gram matrix computation in _ensure_gram().
        self._phsp_data = None   # dict with mass, q, angle arrays

        # Time/frac/weight arrays (for per‑NLL time averages)
        self._phsp_weights = None
        self._phsp_frac = None
        self._phsp_time = None

        # User‑accessible integral values
        self.int_Ap2 = None
        self.int_Am2 = None
        self.int_ApAm = None

    # ═══════════════════════════════════════════════════════════════
    #  Phsp pre‑integration
    # ═══════════════════════════════════════════════════════════════

    def _gp_gm(self, t, scalar):
        """Compute gp(t), gm(t) arrays.  (N,) complex128 each."""
        Gam, DG, Dm, Ap, r, phi = scalar
        eL = np.exp(-1j * t * (-Dm/2 - 1j * (Gam + DG/2) / 2))
        eH = np.exp(-1j * t * (+Dm/2 - 1j * (Gam - DG/2) / 2))
        return (eL + eH) / 2, (eL - eH) / 2

    def _time_averages(self, scalar):
        """⟨|gp|²⟩, ⟨|gm|²⟩, ⟨gp*·gm⟩ from stored phsp time/weights."""
        w = self._phsp_weights
        gp, gm = self._gp_gm(self._phsp_time, scalar)
        gp2 = (gp * gp.conj()).real
        gm2 = (gm * gm.conj()).real
        gpgm = gp.conj() * gm
        ws = np.sum(w)
        return (float(np.sum(w * gp2) / ws),
                float(np.sum(w * gm2) / ws),
                complex(np.sum(w * gpgm) / ws))

    def _ensure_gram(self, params):
        """Build Gram matrices from stored phsp data if needed.

        Uses *m0*, *g0* from *params* — called on first
        ``compute_norm_batched`` or when m0/g0 change.
        """
        if self._phsp_data is None:
            raise RuntimeError("IntegratedBackend: phsp not loaded. "
                               "Call prepare_phsp_batched() first.")
        m0 = np.asarray(params["m0"])
        g0 = np.asarray(params["g0"])
        # Check if already computed with matching m0/g0
        if (self.M_pp is not None
                and np.array_equal(m0, self._gram_m0)
                and np.array_equal(g0, self._gram_g0)):
            return
        # Build (or rebuild) from stored phsp data
        phsp = {**self._phsp_data, "weight": self._phsp_weights,
                "frac": self._phsp_frac, "time": self._phsp_time}
        self._load_phsp_matrices(phsp, m0, g0)
        self._gram_m0 = m0.copy()
        self._gram_g0 = g0.copy()

    def _load_phsp_matrices(self, phsp, m0, g0):
        """Compute Gram matrices from phsp (batched, ~350 MB peak)."""
        n_events = phsp["mass"].shape[0]
        n_wave = self.kernel.n_wave
        n = n_wave // 2

        w = np.asarray(phsp.get("weight", np.ones(n_events)), dtype=np.float64)
        sw = np.sqrt(w)

        bs = 50000
        n_batches = (n_events + bs - 1) // bs
        import sys, time as _time
        _t0 = _time.time()
        M = np.zeros((n_wave, n_wave), dtype=complex)
        for b_start in range(0, n_events, bs):
            b_idx = b_start // bs
            b_end = min(b_start + bs, n_events)
            batch = {k: v[b_start:b_end] for k, v in phsp.items()
                     if isinstance(v, np.ndarray)}
            ba = self.kernel._compute_common_amp_factor(batch, m0=m0, g0=g0)
            Ab = ba * sw[b_start:b_end, np.newaxis]
            M += Ab.T.conj() @ Ab
            _elapsed = _time.time() - _t0
            _eta = _elapsed / (b_idx + 1) * (n_batches - b_idx - 1)
            sys.stdout.write(
                f"\r  Gram matrix: batch {b_idx + 1}/{n_batches}  "
                f"[{_elapsed:.1f}s"
                f"{f', {_eta:.1f}s ETA' if _eta > 1 else ''}]   ")
            sys.stdout.flush()
        sys.stdout.write("\n")
        M = (M + M.conj().T) / 2

        self.M_pp = M[:n, :n].copy()
        self.M_mm = M[n:, n:].copy()
        self.M_pm = M[:n, n:].copy()
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

    def load_data(self, data_np):
        """Delegate data loading to the base backend."""
        return self.base.load_data(data_np)

    def compute(self, params, data_handle, norm=None):
        """Forward / backward pass.

        *norm=None* + *data_handle=None* — norm from Gram matrices
                     (used by :meth:`compute_norm_batched`).
        *norm=None* + *data_handle provided* — delegate to base backend
                     (fallback for plotting, etc.).
        *norm=float* — data NLL → delegate to base backend.
        """
        if norm is not None:
            return self.base.compute(params, data_handle, norm=norm)

        # ── norm mode: use Gram matrices if no data_handle ────────
        if data_handle is not None:
            # Called from plot() — delegate to base backend for per‑event P
            return self.base.compute(params, data_handle, norm=None)

        # Build Gram matrices on first call using m0/g0 from params
        self._ensure_gram(params)

        # ── Fast norm from pre‑computed Gram matrices ─────────────
        if self.M_pp is None:
            raise RuntimeError("IntegratedBackend: phsp not loaded. "
                               "Call prepare_phsp_batched() first.")

        ck = np.asarray(params["ck"], dtype=complex)
        n = len(ck) // 2
        ck_B0, ck_B0bar = ck[:n], ck[n:]

        Gamma, Delta_Gamma, Delta_m, A_prod, poqr, poqi = params["scalar"]
        poq = complex(poqr, poqi)
        poq2 = poq * poq.conjugate()

        # Spatial integrals
        I_pp = ck_B0.conj() @ self.M_pp @ ck_B0
        I_mm = ck_B0bar.conj() @ self.M_mm @ ck_B0bar
        I_pm = ck_B0.conj() @ self.M_pm @ ck_B0bar

        self.int_Ap2 = float(I_pp.real)
        self.int_Am2 = float(I_mm.real)
        self.int_ApAm = complex(I_pm)

        # Time averages
        gp2_avg, gm2_avg, gpgm_avg = self._time_averages(params["scalar"])

        # Combine
        frac_avg = float(np.sum(self._phsp_weights * self._phsp_frac))
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

        # ── ck gradient ───────────────────────────────────────────
        C_pp = A_co * gp2_avg + B_co * gm2_avg / poq2
        C_mm = A_co * poq2 * gm2_avg + B_co * gp2_avg
        z_total = complex(A_co * poq * gpgm_avg
                          + B_co * gpgm_avg.conjugate() / poq.conjugate())

        grad_ck = np.empty_like(ck, dtype=complex)
        Mp = self.M_pp @ ck_B0
        Mm = self.M_mm @ ck_B0bar
        Mpm = self.M_pm @ ck_B0bar
        MpmT_ckbar = self.M_pm.T @ ck_B0.conj()
        grad_ck[:n] = (C_pp * Mp + z_total * Mpm).conj()
        grad_ck[n:] = (C_mm * Mm).conj() + z_total * MpmT_ckbar

        # ── scalar gradients ──────────────────────────────────────
        # ∂norm/∂{gp2,gm2,gpgm}
        d_gp2 = A_co * I_pp.real + B_co * I_mm.real
        d_gm2 = A_co * poq2 * I_mm.real + B_co / poq2 * I_pp.real
        # ∂norm/∂Re(gpgm_avg) = 2·A_co·Re(poq·I_pm) + 2·B_co·Re(I_pm/poq*)
        # ∂norm/∂Im(gpgm_avg) = -2·A_co·Im(poq·I_pm) + 2·B_co·Im(I_pm/poq*)
        # Note: the B term sign differs from the Re case because
        # the B0bar cross term involves conj(gpgm)/poq* · I_pm.
        d_Re = (2.0 * (A_co * (poq * I_pm) + B_co * (I_pm / poq.conjugate())).real)
        d_Im = (-2.0 * (A_co * (poq * I_pm)).imag
                + 2.0 * (B_co * I_pm / poq.conjugate()).imag)

        t = self._phsp_time
        w = self._phsp_weights
        ws = np.sum(w)
        gp, gm = self._gp_gm(t, params["scalar"])

        def avg(f): return float(np.sum(w * f) / ws)
        def avg_c(f): return complex(np.sum(w * f) / ws)

        def chain(dgp2, dgm2, dgpgm):
            return (d_gp2 * dgp2 + d_gm2 * dgm2
                    + d_Re * dgpgm.real + d_Im * dgpgm.imag)

        # Γ
        dg = -0.5 * t * gp; dgm = -0.5 * t * gm
        dgp2_G = avg(2.0 * (gp.conj() * dg).real)
        dgm2_G = avg(2.0 * (gm.conj() * dgm).real)
        dgpgm_G = avg_c(dg.conj() * gm + gp.conj() * dgm)
        dG = float(chain(dgp2_G, dgm2_G, dgpgm_G).real)

        # ΔΓ
        dg = -0.25 * t * gm; dgm = -0.25 * t * gp
        dgp2_DG = avg(2.0 * (gp.conj() * dg).real)
        dgm2_DG = avg(2.0 * (gm.conj() * dgm).real)
        dgpgm_DG = avg_c(dg.conj() * gm + gp.conj() * dgm)
        dDG = float(chain(dgp2_DG, dgm2_DG, dgpgm_DG).real)

        # Δm
        dg = 0.5j * t * gm; dgm = 0.5j * t * gp
        dgp2_DM = avg(2.0 * (gp.conj() * dg).real)
        dgm2_DM = avg(2.0 * (gm.conj() * dgm).real)
        dgpgm_DM = avg_c(dg.conj() * gm + gp.conj() * dgm)
        dDM = float(chain(dgp2_DM, dgm2_DM, dgpgm_DM).real)

        # A_prod
        d_B0_dens = (gp2_avg * I_pp.real + poq2 * gm2_avg * I_mm.real
                     + 2.0 * (poq * gpgm_avg * I_pm).real)
        d_B0bar_dens = (gm2_avg / poq2 * I_pp.real + gp2_avg * I_mm.real
                        + 2.0 * (gpgm_avg.conjugate()
                                 / poq.conjugate() * I_pm).real)
        dAp = float((-frac_avg * d_B0_dens + omf * d_B0bar_dens).real)

        # poqr, poqi (Wirtinger)
        poq_c = poq.conjugate()
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
        P = np.zeros(self.phsp_n)

        return float(norm_val.real if hasattr(norm_val, 'real')
                     else norm_val), grads, P

    # ═══════════════════════════════════════════════════════════════
    #  Batched phsp interface (for fitter compatibility)
    # ═══════════════════════════════════════════════════════════════

    def prepare_phsp_batched(self, phsp_np, n_events):
        """Store phsp data for deferred Gram matrix computation.

        The actual Gram matrix build happens on the first
        :meth:`compute_norm_batched` call when m0/g0 from *params*
        are available.

        Also loads the phsp data to the base backend (e.g. CUDA) so
        that :meth:`compute` with ``norm=None`` and a ``DataHandle``
        (called from :meth:`~ampfit.fitter.Fitter.plot`) can use the
        accelerated backend for per‑event P computation.
        """
        import numpy as np
        w = np.asarray(phsp_np.get("weight", np.ones(n_events)), dtype=np.float64)
        self._phsp_weights = w
        self._phsp_frac = np.asarray(
            phsp_np.get("frac", np.ones(n_events)), dtype=np.float64)
        self._phsp_time = np.asarray(
            phsp_np.get("time", np.zeros(n_events)), dtype=np.float64)
        # Store raw spatial arrays for Gram matrix build
        self._phsp_data = {k: np.asarray(v)
                           for k, v in phsp_np.items()
                           if isinstance(v, np.ndarray)
                           and k in ("mass", "q", "angle")}
        self.phsp_n = n_events

        # Load phsp into the base backend for downstream use (plot, etc.)
        # Ensure 'bkg' key exists (required by CUDA backend)
        if "bkg" not in phsp_np:
            phsp_np["bkg"] = np.zeros(n_events, dtype=np.float64)
        self._phsp_scratch = self.base.load_data(phsp_np)

    def compute_norm_batched(self, params):
        """Fast norm + gradients from Gram matrices.

        Returns ``(norm_scalar, grads_dict)``.
        """
        n, g, _ = self.compute(params, None, norm=None)
        return n, g

    def free(self):
        """Release all resources (matrices + base backend)."""
        self.M_pp = self.M_mm = self.M_pm = None
        self._gram_m0 = self._gram_g0 = None
        self._phsp_data = None
        self._phsp_weights = self._phsp_frac = self._phsp_time = None
        self.phsp_n = 0
        try:
            self.base.free()
        except Exception:
            pass

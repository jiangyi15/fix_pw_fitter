"""Profile likelihood via quadratic prediction + Newton correction.

The Minos-style fast profile: at the minimum ``x̂`` with inverse Hessian
``S`` (the BFGS ``hess_inv``), the quadratic model predicts the profile
path analytically — fixing parameter *i* at ``x̂_i + Δ``:

    nuisance shift:  δx = (S_{·i}/S_ii − e_i)·Δ
    profile NLL:     Q_prof(Δ) = Q(x̂) + Δ²/(2·S_ii)

The real NLL drifts from the quadratic as Δ grows; each scan point
applies the prediction and then CORRECTS with Newton steps in the
nuisance subspace (``δx_n = −S_nn·g_n``, x_i fixed), so the path tracks
the true profile.  1–3 NLL+grad evaluations per scan point instead of a
full re-minimization — the fast NLL+grad (the s0c kernels) makes the
corrections cheap.

The scan operates on the FLAT parameters (the optimizer space); the
physical parameters are the same likelihood with the transform removed.
"""

import numpy as np

from .numerics import fd_hess


class ProfileResult:
    """One profile scan: the corrected path and the ΔQ crossings.

    Attributes:
        idx, name: the scanned flat parameter (index and name).
        q_min: the NLL at the start point.
        delta_q: the target ΔQ (1.0 → the 1σ Minos interval).
        delta: signed scan offsets (x̂_i + delta = the parameter value).
        q: the corrected NLL at each point.
        x: the corrected flat parameter vectors.
        cross_lo, cross_hi: the signed Δ where Q − q_min = delta_q
            (None when the scan did not reach the target).
    """

    def __init__(self, idx, name, x0, q_min, delta_q,
                 delta, q, xs, cross_lo, cross_hi):
        self.idx = idx
        self.name = name
        self.x0 = np.asarray(x0, float)
        self.q_min = float(q_min)
        self.delta_q = float(delta_q)
        self.delta = np.asarray(delta, float)
        self.q = np.asarray(q, float)
        self.x = list(xs)
        self.cross_lo = cross_lo
        self.cross_hi = cross_hi

    @property
    def interval(self):
        """The (lo, hi) parameter values at the ΔQ crossing."""
        lo = None if self.cross_lo is None else self.x0[self.idx] + self.cross_lo
        hi = None if self.cross_hi is None else self.x0[self.idx] + self.cross_hi
        return lo, hi


def fd_hessian(fitter, x0, damp=1e-8):
    """Damped inverse Hessian at ``x0`` — the profile-scan curvature.

    The raw Hessian comes from :func:`fd_hess` on ``fitter.get_nll``'s
    gradient part.

    Flat directions (zero curvature) are regularized: the returned
    inverse Hessian is of ``H + damp·max|diag(H)|·I`` — the well-determined
    directions are untouched; the flat ones get a tiny curvature so the
    prediction/correction algebra stays finite (their components are
    effectively frozen, which the corrections handle).

    Returns:
        ``(h_inv, diag_H)`` — the regularized inverse Hessian and the
        raw curvature diagonal (for diagnostics).
    """
    H = fd_hess(lambda z: fitter.get_nll(z)[1], x0)
    diag = np.diag(H).copy()
    lam = damp * max(float(np.max(np.abs(diag))), 1e-300)
    h_inv = np.linalg.inv(H + lam * np.eye(len(diag)))
    return h_inv, diag


def _crossing(pts, q_min, delta_q):
    """Linear interpolation of the signed Δ where q − q_min = delta_q."""
    ds = np.array([p[0] for p in pts], float)
    dq = np.array([p[1] for p in pts], float) - q_min
    hit = np.nonzero(dq >= delta_q)[0]
    if not len(hit):
        return None
    k = int(hit[0])
    if k == 0 or dq[k] == dq[k - 1]:
        return float(ds[k])
    t = (delta_q - dq[k - 1]) / (dq[k] - dq[k - 1])
    return float(ds[k - 1] + t * (ds[k] - ds[k - 1]))


def profile_scan(fitter, x0, h_inv, idx, delta_q=1.0,
                 max_points=30, max_corr=15, grow=1.6, h_inv_fd=False,
                 g_tol=1e-6):
    """Predictor-corrector profile scan of flat parameter ``idx``.

    Args:
        fitter: provides ``get_nll(x) -> (nll, grad)`` in the flat space.
        x0: the start point (the fit minimum).
        h_inv: the inverse Hessian at ``x0`` (the BFGS ``hess_inv``).
        idx: the flat parameter index to scan.
        delta_q: the target ΔQ (1.0 → the 1σ interval).
        max_points: the per-direction scan-point budget.
        max_corr: the Newton-correction budget per point (the loop
            exits early once the nuisance gradient is below ``g_tol``
            — the definition of on-profile).
        g_tol: the nuisance-gradient norm at which a point is considered
            on-profile.
        grow: the step-growth factor per point (the corrections keep
            the path on the true profile, so the step can grow
            steadily; a rare over-correction just wastes a point).

    Returns:
        A :class:`ProfileResult`.
    """
    x0 = np.asarray(x0, float)
    n = len(x0)
    if h_inv_fd or h_inv is None:
        # the FD Hessian (2N gradient evals, once) — far more accurate
        # than a young BFGS hess_inv; the prediction and the corrections
        # then converge in 1-2 steps
        h_inv, diag_H = fd_hessian(fitter, x0)
    S = np.asarray(h_inv, float)
    if S.shape != (n, n):
        raise ValueError(f"h_inv shape {S.shape} != ({n}, {n})")
    if not np.isfinite(S).all():
        raise ValueError("h_inv has non-finite entries")

    nll0, _g0 = fitter.get_nll(x0)
    q_min = float(nll0)
    Sii = float(S[idx, idx])
    if Sii <= 0:
        raise ValueError(f"h_inv[{idx},{idx}] = {Sii} — not a minimum "
                         f"direction (run the fit longer?)")

    # the quadratic prediction: the FULL profile shift per unit Δ —
    # the i-component of S_{·i}/S_ii is 1, so x_i moves by Δ and the
    # nuisances by their conditional-minimum response
    pred = S[:, idx] / Sii
    d0 = 0.5 * np.sqrt(2.0 * delta_q * abs(Sii))

    mask = np.ones(n, bool)
    mask[idx] = False
    S_nn = S[np.ix_(mask, mask)]

    def point(direction, delta):
        """Predict the profile point, then correct in the nuisance
        subspace.  Returns (q, x, corr_norm) — corr_norm is the total
        correction magnitude (the prediction-quality measure: small
        means the quadratic prediction was good)."""
        x = x0 + pred * (direction * delta)
        q, g = fitter.get_nll(x)
        corr_norm = 0.0
        for _ in range(max_corr):
            if np.linalg.norm(g[mask]) < g_tol:
                break                      # on-profile already
            dx_n = -(S_nn @ g[mask])
            step = 1.0
            improved = False
            for _ in range(3):                       # backtracking
                x_try = x.copy()
                x_try[mask] += step * dx_n
                q_try, g_try = fitter.get_nll(x_try)
                if q_try < q:
                    x, q, g = x_try, q_try, g_try
                    corr_norm += step * float(np.linalg.norm(dx_n))
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break
        return q, x, corr_norm

    def scan(direction):
        pts = []
        delta = d0
        while len(pts) < max_points:
            q, x, corr_norm = point(direction, delta)
            pts.append((direction * delta, q, x))
            if q - q_min >= delta_q:
                break
            # the corrections keep the point on the true profile — the
            # step can grow steadily (the correction magnitude is NOT a
            # quality measure: an imperfect S needs large corrections
            # even on-profile, and the ck-scale corrections dwarf a
            # mass-scale step by nature)
            delta *= grow
        return pts

    pts_plus = scan(+1.0)
    pts_minus = scan(-1.0)
    pts = pts_minus[::-1] + pts_plus
    delta = [p[0] for p in pts]
    q = [p[1] for p in pts]
    xs = [p[2] for p in pts]
    return ProfileResult(idx, None, x0, q_min, delta_q, delta, q, xs,
                         _crossing(pts_minus, q_min, delta_q),
                         _crossing(pts_plus, q_min, delta_q))

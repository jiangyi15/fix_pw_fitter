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

from .numerics import fd_hess, interp_inverse


class ProfileResult:
    """One profile scan: the corrected path — a curve.  The ΔQ level
    and its crossings are queried on demand (:meth:`crossing`,
    :meth:`interval`), not baked in.

    Attributes:
        idx, name: the scanned flat parameter (index and name).
        q_min: the NLL at the start point.
        delta: signed scan offsets (x̂_i + delta = the parameter value);
            the path reads left edge → minimum anchor (Δ=0) → right
            edge.
        q: the corrected NLL at each point.
        x: the corrected flat parameter vectors.
    """

    def __init__(self, idx, name, x0, q_min, delta, q, xs):
        self.idx = idx
        self.name = name
        self.x0 = np.asarray(x0, float)
        self.q_min = float(q_min)
        self.delta = np.asarray(delta, float)
        self.q = np.asarray(q, float)
        self.x = list(xs)

    @property
    def x_vals(self):
        """The scanned parameter values (``x0[idx] + delta``) — the
        plot x-axis; the minimum anchor sits at ``x0[idx]``."""
        return self.x0[self.idx] + self.delta

    @property
    def dq(self):
        """The scaled NLL (``q − q_min``) — the plot y-axis; the
        minimum anchor sits at 0."""
        return self.q - self.q_min

    def crossing(self, delta_q=0.5):
        """The signed ``(Δ_lo, Δ_hi)`` where the scaled NLL crosses
        ``delta_q`` — PCHIP (monotone cubic) inversion of the sampled
        curve, per side; a side never reaching the level gives None.
        """
        d = self.delta
        dq = self.dq
        minus = d <= 0
        plus = d >= 0
        # each side outward from the anchor (dq growing)
        lo = interp_inverse(d[minus][::-1], dq[minus][::-1], delta_q)
        hi = interp_inverse(d[plus], dq[plus], delta_q)
        return lo, hi

    def interval(self, delta_q=0.5):
        """The (lo, hi) parameter values at the ΔQ crossing."""
        lo, hi = self.crossing(delta_q)
        return (None if lo is None else self.x0[self.idx] + lo,
                None if hi is None else self.x0[self.idx] + hi)


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


def profile_scan(fitter, x0, h_inv, idx, delta_q=0.5,
                 max_points=30, max_corr=15, grow=1.6, h_inv_fd=False,
                 g_tol=1e-6, edges=None, step=None):
    """Predictor-corrector profile scan of flat parameter ``idx``.

    Args:
        fitter: provides ``get_nll(x) -> (nll, grad)`` in the flat space.
        x0: the start point (the fit minimum).
        h_inv: the inverse Hessian at ``x0`` (the BFGS ``hess_inv``).
        idx: the flat parameter index to scan.
        delta_q: the target ΔQ (0.5 → the 1σ Minos interval:
            ΔNLL = ½·χ²₁(68.3%) — MINUIT's ERRORDEF for a likelihood).
        max_points: the per-direction scan-point budget.
        max_corr: the Newton-correction budget per point (the loop
            exits early once the nuisance gradient is below ``g_tol``
            — the definition of on-profile).
        g_tol: the nuisance-gradient norm at which a point is considered
            on-profile.
        grow: the step-growth factor per point (the corrections keep
            the path on the true profile, so the step can grow
            steadily; a rare over-correction just wastes a point).
        edges: optional ``(lo, hi)`` scan EDGES as signed offsets from
            ``x0[idx]`` (lo < 0 < hi; a side whose edge is None is
            skipped).  When given, each present side scans from the
            minimum out to its edge — the last point sits exactly on
            the edge — instead of stopping at ΔQ.  The returned path
            always carries the minimum anchor (Δ=0, q=q_min) between
            the two sides.
        step: uniform scan-point spacing (in parameter units) for the
            edge-driven sides — the interval resolution.  Requires the
            edges; ``None`` keeps the adaptive scheme (``d0`` then
            ``grow`` steps), where ``delta_q`` is the stop level.

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
        edge = (None if edges is None
                else edges[0] if direction < 0 else edges[1])
        edge = None if edge is None else abs(float(edge))
        if step is not None:
            if edge is None:
                raise ValueError(
                    "profile_scan: step requires an edge on that side")
            # full steps strictly inside the edge (integer count —
            # arange accumulates float error and can emit the edge)
            k = int(np.floor(edge / step + 1e-9))
            ds = [step * i for i in range(1, k + 1)]
            if ds and edge - ds[-1] <= 1e-12 * max(step, edge):
                ds[-1] = edge            # the last step IS the edge
            else:
                ds.append(edge)          # partial step → exact edge
            if len(ds) > max_points:
                raise ValueError(
                    f"profile_scan: step={step} to edge={edge:.4g} gives "
                    f"{len(ds)} points > max_points={max_points}")
            pts = []
            for d in ds:
                q, x, _cn = point(direction, d)
                pts.append((direction * d, q, x))
            return pts
        pts = []
        delta = d0
        while len(pts) < max_points:
            if edge is not None and delta > edge:
                delta = edge             # the last point exactly on the edge
            q, x, corr_norm = point(direction, delta)
            pts.append((direction * delta, q, x))
            # an explicit edge owns the stopping: the side scans to the
            # edge even past ΔQ (the crossings use whatever it reached);
            # without an edge the ΔQ rule stops the side
            if edge is not None:
                if delta >= edge:
                    break
            elif q - q_min >= delta_q:
                break
            # the corrections keep the point on the true profile — the
            # step can grow steadily (the correction magnitude is NOT a
            # quality measure: an imperfect S needs large corrections
            # even on-profile, and the ck-scale corrections dwarf a
            # mass-scale step by nature)
            delta *= grow
        return pts

    edge_lo = None if edges is None else edges[0]
    edge_hi = None if edges is None else edges[1]
    pts_plus = scan(+1.0) if edge_hi is not None or edges is None else []
    pts_minus = scan(-1.0) if edge_lo is not None or edges is None else []
    # the minimum anchor between the sides (Δ=0, q=q_min) — the path
    # reads "min to left/right edge" as one array
    pts = pts_minus[::-1] + [(0.0, q_min, x0.copy())] + pts_plus
    delta = [p[0] for p in pts]
    q = [p[1] for p in pts]
    xs = [p[2] for p in pts]
    return ProfileResult(idx, None, x0, q_min, delta, q, xs)

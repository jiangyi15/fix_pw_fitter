"""B → 4π (π⁺π⁻π⁺π⁻) phase-space generation via sequential decay.

The decay is treated as two sequential two-body decays::

    B → (ππ)₁ (ππ)₂ → π⁺π⁻π⁺π⁻

Sampling variables (all 5 phase-space degrees of freedom):

    m₁, m₂        di-pion invariant masses, importance-sampled from
                  ``p(m_B; m₁,m₂) · p(m₁; m_π,m_π) · p(m₂; m_π,m_π)``
                  (rejection sampling), where ``p(M; a, b) = 2q/M`` is
                  the two-body phase-space factor and *q* the breakup
                  momentum.
    cos θ₁, cos θ₂  flat in [-1, 1]
    φ               flat in [0, 2π) — azimuth between the two decay
                    planes

Geometry (B rest frame)::

    (ππ)₁ flies along +z, (ππ)₂ along −z.      (B → 12 34 axis = z)
    In the (ππ)₁ rest frame, π₁ makes polar angle θ₁ with +z.
    In the (ππ)₂ rest frame, π₃ makes polar angle θ₂ with −z.
    The (ππ)₁ decay plane is the xz-plane; the (ππ)₂ plane is rotated
    about z by the azimuth φ.  Both planes contain the z-axis.

Because m₁, m₂ are sampled from the product of the two-body phase-space
factors and cos θ, φ are flat, the generated points are uniformly
distributed over the 5-dimensional B → 4π phase space (unit weight).

:func:`generate_b4pi_fixed_m3pi` instead generates the event at a
*fixed* m(πππ) (bottom-up: flat 3π Dalitz of R in its rest frame,
isotropic B → R + π⁻ production angles, boosted into the B rest frame)
— the approach of the reference lineshape generator.

Usage::

    from ampfit.phasespace_b4pi import generate_b4pi
    ev = generate_b4pi(10000)
    mom = ev["momenta"]            # (n, 4, 4)  (E, px, py, pz) per pion

CLI (write an ``.npz`` file)::

    python -m ampfit.phasespace_b4pi -n 200000 -o phsp_b4pi.npz
"""

import functools

import numpy as np

M_PION = 0.1396
M_B_MESON = 5.279


# ═══════════════════════════════════════════════════════════════════
# Two-body kinematics
# ═══════════════════════════════════════════════════════════════════

def two_body_momentum(M, m1, m2):
    """Breakup momentum *q* in the rest frame of ``M → m1 + m2``."""
    M, m1, m2 = np.asarray(M), np.asarray(m1), np.asarray(m2)
    arg = np.maximum((M ** 2 - (m1 + m2) ** 2) * (M ** 2 - (m1 - m2) ** 2),
                     0.0)
    return np.sqrt(arg) / (2.0 * M)


# ═══════════════════════════════════════════════════════════════════
# Mass sampling
# ═══════════════════════════════════════════════════════════════════

def _mass_weight(m1, m2, m_B, m_pi):
    """w = q(m_B; m1,m2) · q(m1; mπ,mπ) · q(m2; mπ,mπ) — the mass
    sampling weight in the breakup-momentum convention."""
    return (two_body_momentum(m_B, m1, m2)
            * two_body_momentum(m1, m_pi, m_pi)
            * two_body_momentum(m2, m_pi, m_pi))


def _find_w_max(m_B, m_pi):
    """Maximum of the mass weight over the physical region.

    The weight is smooth and unimodal, so it is maximised directly via
    ``scipy.optimize.minimize`` on ``-log w`` (several starts for
    robustness), subject to ``2mπ ≤ m₁, m₂ ≤ m_B−2mπ`` and
    ``m₁+m₂ ≤ m_B``.  A tiny safety margin is added so the returned
    value is a guaranteed upper bound for the rejection sampling.
    Cached per ``(m_B, m_pi)``.
    """
    return _find_w_max_cached(float(m_B), float(m_pi))


@functools.lru_cache(maxsize=None)
def _find_w_max_cached(m_B, m_pi):
    from scipy.optimize import minimize

    lo = 2.0 * m_pi
    hi = m_B - 2.0 * m_pi

    def neg_log_w(x):
        m1, m2 = x
        if m1 <= lo or m2 <= lo or m1 >= hi or m2 >= hi or m1 + m2 > m_B:
            return 1e6
        w = _mass_weight(np.array([m1]), np.array([m2]), m_B, m_pi)[0]
        return -np.log(w) if w > 0 else 1e6

    bounds = [(lo + 1e-4, hi - 1e-4)] * 2
    cons = {"type": "ineq", "fun": lambda x: m_B - x[0] - x[1]}

    best = None
    for x0 in [(1.0, 1.0), (0.5 * hi, 0.5 * hi), (lo + 0.25 * (hi - lo),
               lo + 0.25 * (hi - lo)), (lo + 0.75 * (hi - lo),
               lo + 0.75 * (hi - lo))]:
        res = minimize(neg_log_w, x0, method="SLSQP", bounds=bounds,
                       constraints=cons)
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("w_max minimisation failed")
    return float(np.exp(-best.fun)) * 1.0001   # safety margin


def sample_masses(n, m_B, m_pi, rng):
    """Importance-sample ``(m₁, m₂)`` from the phase-space weight.

    Rejection sampling with an independent uniform proposal: m₁, m₂
    both uniform in ``[2mπ, m_B−2mπ]`` (independent bounds), points with
    ``m₁+m₂ > m_B`` rejected, and acceptance ``u < w/w_max`` where
    ``w_max`` is the weight maximum (found by direct minimisation).
    """
    lo = 2.0 * m_pi
    hi = m_B - 2.0 * m_pi

    w_max = _find_w_max(m_B, m_pi)

    m1s = np.empty(n)
    m2s = np.empty(n)
    got = 0
    while got < n:
        batch = max(n - got, 1024)
        m1 = rng.uniform(lo, hi, batch)
        m2 = rng.uniform(lo, hi, batch)            # independent bounds
        w = _mass_weight(m1, m2, m_B, m_pi)
        w = np.where(m1 + m2 <= m_B, w, 0.0)       # kinematic cut
        u = rng.uniform(0.0, w_max, batch)
        keep = u < w
        k = min(int(keep.sum()), n - got)
        if k:
            m1s[got:got + k] = m1[keep][:k]
            m2s[got:got + k] = m2[keep][:k]
            got += k
    return m1s, m2s


# ═══════════════════════════════════════════════════════════════════
# 4-momentum construction
# ═══════════════════════════════════════════════════════════════════

def _boost(p, beta):
    """Lorentz boost a stack of 4-momenta (n, 4) by velocities (n, 3)."""
    b = np.linalg.norm(beta, axis=-1)
    b = np.maximum(b, 1e-15)
    nhat = beta / b[:, None]
    gam = 1.0 / np.sqrt(1.0 - b ** 2)
    pvec = p[:, 1:]
    ppar = np.sum(pvec * nhat, axis=-1)              # component along n̂
    pperp = pvec - ppar[:, None] * nhat              # transverse part
    E = p[:, 0]
    Enew = gam * (E + b * ppar)
    ppar_new = gam * (ppar + b * E)
    pnew = ppar_new[:, None] * nhat + pperp
    return np.stack([Enew, pnew[:, 0], pnew[:, 1], pnew[:, 2]], axis=-1)


def generate_b4pi_fixed_m3pi(m3pi, n, m_B=M_B_MESON, m_pi=M_PION, seed=None,
                             cos_theta13=None, phi13=None,
                             cos_theta_B=None, phi_B=None):
    """Generate B → R + π⁻, R → π⁺π⁻π⁺ at a **fixed** m(πππ) = *m3pi*.

    Builds the event bottom-up at fixed resonance mass (the approach of
    the reference lineshape generator, ``plot_single_chain_amp6.py``)::

        1. R rest frame: a *flat* 3-body Dalitz sample of the 3π —
           m₁₂ drawn from the mass marginal
           ``P(m₁₂) ∝ q(m_R;m₁₂,m_π)·q(m₁₂;m_π,m_π)`` (the q·q is the
           pdf of the *mass* m₁₂; the s₁₂ marginal carries an extra
           m_R/m₁₂ factor), cosθ₁₃ and φ₁₃ uniform, constructed with
           the (12) axis along +z.
        2. B → R + π⁻: the R carries the fixed breakup momentum
           q_B = q(m_B; m_R, m_π) in a uniformly random direction
           (cos θ_B, φ_B flat).
        3. Boost the 3π from the R rest frame into the B rest frame
           along the R flight axis; the π⁻ bachelor balances momentum.

    The 4 final momenta are returned in the B (lab) rest frame,
    ordered ``[pip1, pim1, pip2, pim2]`` (pip1/pip2 are the two
    identical π⁺ of R, pim1 the π⁻ of R, pim2 the π⁻ bachelor).  The
    sample is uniform over the full 5-dim phase space *at* the fixed
    m(πππ) shell.

    Parameters
    ----------
    m3pi : float or ndarray (n,)
        Fixed 3π mass(es); a scalar is broadcast to all events.
    n : int
        Number of events (ignored if ``m3pi`` is an array).
    m_B, m_pi : float
        B meson and pion masses.
    seed : int, optional
        Random seed.
    cos_theta13, phi13 : float or ndarray (n,), optional
        Fixed R → 3π decay angles (in the R rest frame, particle 1 of
        the (12) pair w.r.t. the flight axis).  Default: random.
    cos_theta_B, phi_B : float or ndarray (n,), optional
        Fixed B → R + π⁻ production angles (direction of R in the B
        rest frame).  Default: random.

    Returns
    -------
    dict with:
        momenta   : (n, 4, 4) — [pip1, pim1, pip2, pim2] 4-momenta in
                    the B rest frame.
        m_R       : (n,) — the fixed resonance mass(es).
        m12       : (n,) — the (π⁺π⁻) invariant mass of the R decay.
        cos_theta13, phi13 : (n,) — R-rest-frame decay angles.
        cos_theta_B, phi_B : (n,) — B → R π⁻ production angles.
    """
    m3pi = np.asarray(m3pi, dtype=float)
    if m3pi.ndim == 0:
        m_R = np.full(n, float(m3pi))
    else:
        m_R = np.asarray(m3pi)
        n = len(m_R)
    rng = np.random.default_rng(seed)

    def _angles(v, key):
        """Broadcast an optional fixed angle (scalar or (n,) array)."""
        if v is None:
            return None
        a = np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy()
        if key.startswith("cos"):
            if np.any(np.abs(a) > 1.0):
                raise ValueError(f"{key} must lie in [-1, 1]")
        return a

    ct13_f = _angles(cos_theta13, "cos_theta13")
    phi13_f = _angles(phi13, "phi13")
    ctB_f = _angles(cos_theta_B, "cos_theta_B")
    phiB_f = _angles(phi_B, "phi_B")

    lo, hi = 2.0 * m_pi, m_R - m_pi
    if np.any(hi <= lo):
        raise ValueError("m3pi must satisfy 3 m_pi < m3pi < m_B - m_pi")

    # ── 1. flat 3-body Dalitz of R in its rest frame ───────────────
    # Sample m12 from the MASS marginal P(m12) ∝ q(m_R;m12,m_π)·
    # q(m12;m_π,m_π) — this IS the flat-Dalitz marginal (q·q is the
    # pdf of m12; the s12 marginal carries an extra m_R/m12 factor).
    # cosθ₁₃ uniform then makes (s12, s13) flat.
    m12 = np.empty(n)
    # vectorised rejection: one proposal pass per distinct mass value.
    # w_max bound: q(m_R; m12, m_π) ≤ q at smallest m12 (=2m_π) and
    # q(m12; m_π, m_π) ≤ q at largest m12 (=m_R−m_π), so the product is
    # bounded by q_max·q_max (no numerical scan needed).
    for mR_i in np.unique(m_R):
        mask = m_R == mR_i
        cnt = int(mask.sum())
        lo_i, hi_i = lo, hi[mask][0]
        wmax = (two_body_momentum(mR_i, lo_i, m_pi)
                * two_body_momentum(hi_i, m_pi, m_pi)) * 1.0001
        got = 0
        buf = np.empty(cnt)
        while got < cnt:
            cand = rng.uniform(lo_i, hi_i, max(cnt - got, 1024))
            ww = (two_body_momentum(mR_i, cand, m_pi)
                  * two_body_momentum(cand, m_pi, m_pi))
            keep = rng.uniform(0.0, wmax, len(cand)) < ww
            k = min(int(keep.sum()), cnt - got)
            if k:
                buf[got:got + k] = cand[keep][:k]
                got += k
        m12[mask] = buf

    s12 = m12 ** 2
    ct13 = ct13_f if ct13_f is not None else rng.uniform(-1.0, 1.0, n)
    phi13 = phi13_f if phi13_f is not None \
        else rng.uniform(0.0, 2.0 * np.pi, n)
    st13 = np.sqrt(np.maximum(1.0 - ct13 ** 2, 0.0))

    # (12) rest frame: particles 1, 2 back-to-back along n̂13, then boost
    q1 = two_body_momentum(m12, m_pi, m_pi)
    E1 = np.sqrt(q1 ** 2 + m_pi ** 2)
    q3 = two_body_momentum(m_R, m12, m_pi)    # p of particle 3 in R frame
    E3 = np.sqrt(q3 ** 2 + m_pi ** 2)
    E12 = np.sqrt(q3 ** 2 + m12 ** 2)         # (12) energy in R frame
    beta12 = q3 / E12                          # (12) frame → R frame, +z
    gam12 = 1.0 / np.sqrt(1.0 - beta12 ** 2)

    p1x, p1z = q1 * st13 * np.cos(phi13), q1 * ct13
    p1y = q1 * st13 * np.sin(phi13)
    E1_R = gam12 * (E1 + beta12 * p1z)
    pz1_R = gam12 * (p1z + beta12 * E1)
    E2_R = gam12 * (E1 - beta12 * p1z)
    pz2_R = gam12 * (-p1z + beta12 * E1)

    pR_rf = np.empty((n, 3, 4))
    pR_rf[:, 0] = np.stack([E1_R, p1x, p1y, pz1_R], axis=1)
    pR_rf[:, 1] = np.stack([E2_R, -p1x, -p1y, pz2_R], axis=1)
    pR_rf[:, 2] = np.stack([E3, np.zeros(n), np.zeros(n), -q3], axis=1)

    # ── 2. B → R + π⁻: production angles (fixed or random) ────────
    qB = two_body_momentum(m_B, m_R, m_pi)
    ER = np.sqrt(qB ** 2 + m_R ** 2)
    Epi = np.sqrt(qB ** 2 + m_pi ** 2)
    ctB = ctB_f if ctB_f is not None else rng.uniform(-1.0, 1.0, n)
    phiB = phiB_f if phiB_f is not None \
        else rng.uniform(0.0, 2.0 * np.pi, n)
    stB = np.sqrt(np.maximum(1.0 - ctB ** 2, 0.0))
    nhat = np.stack([stB * np.cos(phiB), stB * np.sin(phiB), ctB], axis=1)

    # ── 3. boost the 3π R rest → B rest along the R flight axis ───
    betaR = (qB / ER)[:, None] * nhat
    pB = _boost(pR_rf.reshape(n * 3, 4), np.repeat(betaR, 3, axis=0))
    pB = pB.reshape(n, 3, 4)

    mom = np.empty((n, 4, 4))
    mom[:, 0] = pB[:, 0]                                  # pip1
    mom[:, 1] = pB[:, 1]                                  # pim1 (π⁻ of R)
    mom[:, 2] = pB[:, 2]                                  # pip2
    mom[:, 3] = np.stack([Epi, -qB * nhat[:, 0],
                          -qB * nhat[:, 1], -qB * nhat[:, 2]], axis=1)  # pim2

    return {"momenta": mom, "m_R": m_R, "m12": m12,
            "cos_theta13": ct13, "phi13": phi13,
            "cos_theta_B": ctB, "phi_B": phiB}


def generate_b4pi(n_events, m_B=M_B_MESON, m_pi=M_PION, seed=None):
    """Generate *n_events* B → 4π phase-space points (B rest frame).

    Parameters
    ----------
    n_events : int
        Number of events.
    m_B, m_pi : float
        B meson and pion masses.
    seed : int, optional
        Random seed (``np.random.default_rng``).

    Returns
    -------
    dict with:
        momenta    : ndarray (n, 4, 4)  — per event, the four pion
                     4-momenta (E, px, py, pz) ordered [π₁, π₂, π₃, π₄],
                     where (π₁,π₂) form (ππ)₁ and (π₃,π₄) form (ππ)₂.
        m1, m2     : ndarray (n,) — di-pion invariant masses.
        cos_theta1 : ndarray (n,) — cos of θ₁ (π₁ w.r.t. +z).
        cos_theta2 : ndarray (n,) — cos of θ₂ (π₃ w.r.t. −z).
        phi        : ndarray (n,) — azimuth between the decay planes.
    """
    rng = np.random.default_rng(seed)

    m1, m2 = sample_masses(n_events, m_B, m_pi, rng)
    ct1 = rng.uniform(-1.0, 1.0, n_events)
    ct2 = rng.uniform(-1.0, 1.0, n_events)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_events)
    st1 = np.sqrt(np.maximum(1.0 - ct1 ** 2, 0.0))
    st2 = np.sqrt(np.maximum(1.0 - ct2 ** 2, 0.0))

    # B → (ππ)₁ (ππ)₂ in the B rest frame
    q_B = two_body_momentum(m_B, m1, m2)
    E1 = np.sqrt(m1 ** 2 + q_B ** 2)      # (ππ)₁ energy
    E2 = np.sqrt(m2 ** 2 + q_B ** 2)      # (ππ)₂ energy
    beta1 = q_B / E1                      # (ππ)₁ moves along +z
    beta2 = q_B / E2                      # (ππ)₂ moves along −z
    gam1 = 1.0 / np.sqrt(1.0 - beta1 ** 2)
    gam2 = 1.0 / np.sqrt(1.0 - beta2 ** 2)

    # (ππ)₁ rest frame: π₁ at azimuth 0 (decay plane = xz), θ₁ from +z
    q1 = two_body_momentum(m1, m_pi, m_pi)
    e1p = np.sqrt(q1 ** 2 + m_pi ** 2)    # pion energy in (ππ)₁ frame
    p1x, p1z = q1 * st1, q1 * ct1         # π₁: (e1p, p1x, 0, p1z)

    # (ππ)₂ rest frame: π₃ at azimuth φ, θ₂ from −z
    q2 = two_body_momentum(m2, m_pi, m_pi)
    e2p = np.sqrt(q2 ** 2 + m_pi ** 2)    # pion energy in (ππ)₂ frame
    p3x = q2 * st2 * np.cos(phi)
    p3y = q2 * st2 * np.sin(phi)
    p3z = -q2 * ct2

    # Boost π₁, π₂ from (ππ)₁ frame → B frame (+z by β1)
    Eo1 = gam1 * (e1p + beta1 * p1z)
    pzo1 = gam1 * (p1z + beta1 * e1p)
    Eo2 = gam1 * (e1p - beta1 * p1z)
    pzo2 = gam1 * (-p1z + beta1 * e1p)

    # Boost π₃, π₄ from (ππ)₂ frame → B frame (velocity −β2 along z)
    Eo3 = gam2 * (e2p - beta2 * p3z)
    pzo3 = gam2 * (p3z - beta2 * e2p)
    Eo4 = gam2 * (e2p + beta2 * p3z)
    pzo4 = gam2 * (-p3z - beta2 * e2p)

    mom = np.empty((n_events, 4, 4))
    mom[:, 0] = np.stack([Eo1, p1x, np.zeros(n_events), pzo1], axis=1)
    mom[:, 1] = np.stack([Eo2, -p1x, np.zeros(n_events), pzo2], axis=1)
    mom[:, 2] = np.stack([Eo3, p3x, p3y, pzo3], axis=1)
    mom[:, 3] = np.stack([Eo4, -p3x, -p3y, pzo4], axis=1)

    return {"momenta": mom, "m1": m1, "m2": m2,
            "cos_theta1": ct1, "cos_theta2": ct2, "phi": phi}


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def _write_npz(path, ev):
    np.savez(path, momenta=ev["momenta"], m1=ev["m1"], m2=ev["m2"],
             cos_theta1=ev["cos_theta1"], cos_theta2=ev["cos_theta2"],
             phi=ev["phi"])
    print(f"wrote {len(ev['m1'])} events -> {path}")


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--n-events", type=int, default=100000)
    ap.add_argument("-o", "--output", default="phsp_b4pi.npz")
    ap.add_argument("--mB", type=float, default=M_B_MESON)
    ap.add_argument("--mpi", type=float, default=M_PION)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    ev = generate_b4pi(args.n_events, m_B=args.mB, m_pi=args.mpi,
                       seed=args.seed)
    print(f"generated {args.n_events} events in {time.time()-t0:.2f}s")
    _write_npz(args.output, ev)

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

"""Flat phase space for a decay tree of ANY shape — TFPWA's mass-tree API.

The signature is TFPWA's: a PARENT MASS, a mass tree of its daughters, then
the event count.  Nothing here knows about configs, amplitudes or kernel
slots::

    from tabpwa.phasespace import generate_phsp

    # flat 3-body: list the FINAL masses
    mom = generate_phsp(3.0, [0.139, 0.139, 0.547], 10_000, seed=1)

    # flat 5-body — the same rule, however the decay is grouped
    mom = generate_phsp(3.0, [0.139, 0.139, 0.493, 0.493, 0.547], 10_000, seed=1)

    # a resonance PINNED at 0.9 (TFPWA's nested form): m(ef) is then a delta
    mom = generate_phsp(3.0, [(0.9, [0.139, 0.139]), 0.547], 10_000, seed=1)

A child is one of

* ``float`` — a FINAL particle of that mass;
* ``(m, [...])`` — an intermediate **pinned** at ``m`` (TFPWA's form: the node
  decays, but its invariant mass is not integrated).

Flat phase space is a FLAT list of the final masses: the grouping does not
enter the measure (the density is invariant under the two-body fold, which
``tests/test_pwa_phsp.py`` gates by reordering the daughters).  A mass-less
nested list is therefore rejected rather than silently treated as a structure
— it would mean the same thing as the flat list.

``generate_phsp`` returns one ``(n, 4)`` array per final, in depth-first leaf
order, i.e. the order the tree lists them.

``top_p`` moves the decaying particle off rest (a produced resonance is
never exactly at rest): :func:`fix_beam` returns the top's momentum as the
``(n, rng) -> (n, 3)`` callable ``top_p`` wants, and a plain callable
does the same when the momentum varies event by event.

Why this is exact
-----------------
``dΦ_n = Π_k dΦ_2(S_{k-1}; m_k, m_Sk) dm_Sk²/2π`` and
``dΦ_2 = q/(16π²M) dΩ`` (angle-integrated: ``q/(4πM)``),
so each split contributes ``q_k · dm_k`` (the ``1/m`` cancels the ``2m`` from
``dm²``): flat phase space is "uniform in the masses, weighted by ``Π q``".
A multi-daughter vertex is folded into two-body splits

    M -> d_0 + S_1,  S_1 -> d_1 + S_2,  …,  S_{n-2} -> d_{n-2} + d_{n-1}

with ``S_k`` the cluster ``{d_k … d_{n-1}}``.  The SAMPLED variables are the
slack ``δ_i ≥ 0`` subject to the ONE constraint ``Σ δ_i = Δ = M − Σ m_i`` — a
simplex.  Uniform on it (``Dirichlet(1, …, 1)``) gives

    S_k = suffix[k+1] + Σ_{i>k} δ_i

which satisfies ``S_k ≥ Σ_{i>k} m_i`` and ``S_k ≤ S_{k-1} − m_k`` automatically,
and the map ``δ → S`` is triangular with unit diagonal: volume preserving, so
the proposal density over the feasible region is CONSTANT and the importance
weight is exactly ``Π q``.

That is why no window factor appears.  With NESTED uniform draws instead
(``S_k ~ U(suffix[k+1], S_{k-1} − m_k)``) the proposal density is ``Π 1/W_k``
and the weight has to carry ``Π W`` back; omitting it is not a constant rescale
and biases the sample — measured on a 4-body with four identical masses,
``m(e,f) = 1.38`` against ``m(g,h) = 1.29`` where symmetry demands equality.  The
simplex proposal also rejects far less: MEASURED acceptance 0.170 → 0.460 at 5
daughters and 0.012 → 0.302 at 7, i.e. 2.7x and 25x fewer candidate mass draws
for the same law.

The rejection envelope is that weight's SUPREMUM, solved level by level from
the mass ranges (:func:`_fold_envelope`) rather than estimated from a pilot —
so no event is capped and the sample does not depend on the seed.  Each VERTEX
is accepted on its own envelope: the vertices are independent (their variables
are disjoint), so a per-vertex acceptance reproduces the product law exactly
while paying ``1/r`` per vertex instead of the whole event's ``Π 1/r``.

The weight depends on the MASSES ALONE (the directions are isotropic and
unweighted), and masses are far cheaper than a boost, so the two are separate
passes: every candidate draws only its masses and is accepted or rejected on
them, and the angles and boosts are paid ONCE, for the survivors
(:func:`_vertex_masses` / :func:`_vertex_momenta`).  Each vertex draws its own
masses independently; the only coupling between decays is the boost chain, and
the total weight is the product of the per-vertex ones — which is exact
because their variables are disjoint.
"""

import numpy as np

__all__ = ["generate_phsp", "PhaseSpace", "two_body_momentum", "boost",
           "fix_beam"]


# ════════════════════════════════════════════════════════════════════
# kinematics
# ════════════════════════════════════════════════════════════════════

def two_body_momentum(M, m1, m2):
    """Breakup momentum of ``M -> m1 + m2`` (0 below threshold or at M = 0).

    A MASSLESS parent has no phase space: ``0/0`` here would poison every
    weight that touches it (a grid starting at ``s_min = 0`` then yields
    ``nan``), so ``M == 0`` returns 0 rather than ``nan``.

    ``phasespace_b4pi`` and ``pwa_build._two_body_p``/``_boost_vec`` keep their
    own frozen siblings of this pair.  They agree with this copy BIT FOR BIT
    wherever a decay is physical; outside it they differ on purpose — their
    clamped Källén product is positive again when BOTH factors are negative, so
    they return a spurious momentum where this copy returns 0 (and ``nan`` at
    ``M == 0``).  The siblings stay frozen, so this copy must not be unified
    into them.
    """
    M, m1, m2 = np.broadcast_arrays(np.asarray(M, float), np.asarray(m1, float),
                                    np.asarray(m2, float))
    # The Källén product is POSITIVE again when BOTH factors are negative, i.e.
    # for an impossible decay, so clamping the product alone hands back a
    # spurious momentum (measured q(3.0, 0.139, 3.5) = 0.520).  Gate on the
    # physical condition itself; M >= m1 + m2 also implies M >= |m1 - m2|.
    lam = (M * M - (m1 + m2) ** 2) * (M * M - (m1 - m2) ** 2)
    physical = (M >= m1 + m2) & (M > 0.0)
    num = np.sqrt(np.maximum(lam, 0.0))
    out = np.zeros(M.shape)
    np.divide(num, 2.0 * M, out=out, where=physical)
    return float(out) if out.shape == () else out


def boost(p4, beta):
    """Boost 4-momenta ``p4`` (…, 4) by velocity ``beta`` (…, 3), rest -> moving."""
    p4 = np.asarray(p4, float)
    beta = np.asarray(beta, float)
    b2 = np.sum(beta * beta, axis=-1)
    if np.any(b2 >= 1.0):
        raise ValueError(
            "phase space: a boost needs |beta| < 1 "
            f"(max |beta|^2 = {float(np.max(b2)):.6g})")
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-300))
    bp = np.einsum("...i,...i->...", beta, p4[..., 1:])
    safe = np.where(b2 > 1e-300, b2, 1.0)
    coef = np.where(b2 > 1e-300, (gamma - 1.0) / safe, 0.0)
    out = np.empty_like(p4)
    out[..., 0] = gamma * (p4[..., 0] + bp)
    out[..., 1:] = (p4[..., 1:]
                    + coef[..., None] * beta * bp[..., None]
                    + gamma[..., None] * p4[..., 0:1] * beta)
    return out


def _unit_vec(rng, n):
    """Isotropic unit vectors (n, 3)."""
    z = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


# ════════════════════════════════════════════════════════════════════
# the mass tree
# ════════════════════════════════════════════════════════════════════

class _Tree:
    """Topology + masses, read off a mass tree.

    ``dmap``   name -> [daughter names]   (a name not in it is a final)
    ``fixed``  name -> mass

    EVERY node's mass is given — a final IS its mass and ``(m, [...])`` pins an
    intermediate — so nothing here is "sampled": flat phase space integrates
    the finals' masses through the fold's cluster variables
    (``_vertex_masses``),
    never through a node's own mass.
    """

    def __init__(self, top, dmap, fixed):
        self.top = top
        self.dmap = dmap
        self.fixed = fixed
        self._lo = {}

    def is_final(self, name):
        return name not in self.dmap

    def lo(self, name):
        """Smallest invariant mass ``name`` can have (the sum of its minima)."""
        if name not in self._lo:
            if name == self.top:
                self._lo[name] = float(sum(self.lo(o) for o in self.dmap[name]))
            else:
                self._lo[name] = float(self.fixed[name])
        return self._lo[name]


def _parse_tree(m0, children):
    """``m0`` + ``[children]`` -> ``(top, dmap, fixed, leaves)``.

    Names are index tuples (``()`` is the top), so the tree needs no strings
    and a repeated mass (two pions) is still two distinct nodes.
    """
    if not isinstance(children, (tuple, list)) or len(children) < 2:
        raise ValueError(
            f"phase space: m0={m0!r} must decay to >= 2 daughters, got "
            f"{children!r}")

    top, dmap, fixed, leaves = (), {}, {(): float(m0)}, []

    def walk(kids, name):
        dmap[name] = []
        for i, ch in enumerate(kids):
            cn = name + (i,)
            if isinstance(ch, (tuple, list)):
                if not (len(ch) == 2 and not isinstance(ch[0], (tuple, list))
                        and isinstance(ch[1], (tuple, list))):
                    raise ValueError(
                        f"phase space: a nested node must carry its mass as "
                        f"(m, [daughters]), got {ch!r}.  For FLAT phase space "
                        f"list the final masses directly — the grouping does "
                        f"not change the measure.")
                if len(ch[1]) < 1:
                    raise ValueError(
                        f"phase space: pinned node {ch[0]!r} has no daughters")
                fixed[cn] = float(ch[0])            # TFPWA: (m, [daughters])
                walk(list(ch[1]), cn)
            else:
                if float(ch) < 0:
                    raise ValueError(f"phase space: negative mass {ch!r}")
                fixed[cn] = float(ch)               # a final
            dmap[name].append(cn)

    walk(list(children), top)

    def collect(name):
        if name != top and not dmap.get(name):
            leaves.append(name)
            return
        for ch in dmap[name]:
            collect(ch)

    collect(top)
    return top, dmap, fixed, leaves


# ════════════════════════════════════════════════════════════════════
# masses, then momenta, at one vertex
# ════════════════════════════════════════════════════════════════════

class _VertexPlan:
    """Everything about ONE vertex that does not depend on the events.

    Built once when the topology is compiled: the daughters' masses, their
    cumulative sums (``suffix``), the slack ``Δ`` the simplex splits, and the
    rejection ceiling.  A regenerating
    caller therefore pays NEITHER the parse NOR the envelope solve again.
    """

    def __init__(self, m, kids, tree):
        self.m = float(m)
        self.kids = list(kids)
        if len(self.kids) < 2:
            raise ValueError(
                f"phase space: a decay needs >= 2 daughters, got {len(self.kids)}")
        self.md = [float(tree.fixed[d]) for d in self.kids]
        self.lo = [tree.lo(d) for d in self.kids]
        self.suffix = [0.0] * (len(self.kids) + 1)
        for k in range(len(self.kids) - 1, -1, -1):
            self.suffix[k] = self.suffix[k + 1] + self.lo[k]
        #: the whole phase space is the slack split among the daughters
        self.slack = self.m - self.suffix[0]
        value, exact = _fold_envelope(self.m, self.kids, tree)
        self.ceiling = value * (_ENV_EXACT_MARGIN if exact else _ENV_MARGIN)


def _vertex_masses(plan, rng, ne):
    """Sample one vertex's MASSES: ``(steps, weight)``, no angles, no boosts.

    The variables are the SLACK: every daughter takes ``m_i + δ_i`` with
    ``δ_i ≥ 0``, so the only constraint is ``Σ δ_i = Δ = M − Σ m_i`` — a simplex.
    A uniform draw on it (``Dirichlet(1, …, 1)``) makes the cluster masses

        ``S_k = suffix[k+1] + Σ_{i>k} δ_i``

    which automatically satisfies both ``S_k ≥ Σ_{i>k} m_i`` and
    ``S_k ≤ S_{k−1} − m_k``.  The map ``δ → S`` is triangular with unit diagonal,
    so it is volume preserving and the proposal density over the feasible region
    is CONSTANT: the importance weight is exactly ``Π q`` — no window factors.

    ``weight`` depends on the masses ALONE (the directions are isotropic), so it
    is the whole rejection criterion and the cheap half.  ``steps`` is one
    ``(m_left, m_right, q, None)`` per fold step for :func:`_vertex_momenta`.

    MEASURED against the older nested-uniform proposal (which needs ``Π W``):
    the acceptance against the solved ceiling rises 0.170 → 0.460 at 5 daughters
    and 0.012 → 0.302 at 7, i.e. 2.7x and 25x fewer candidate draws for the same
    law (KS p = 0.35 on a pair mass).
    """
    n = len(plan.kids)
    parent = np.broadcast_to(np.asarray(plan.m, float), (ne,)).astype(float)
    weight = np.ones(ne)
    if n == 3:
        # Dirichlet(1, 1) IS one uniform, and the call overhead is 5x that of a
        # plain draw (MEASURED 0.51 vs 0.10 ms per 20 k): take the short path.
        u = rng.uniform(0.0, 1.0, ne) * plan.slack
        above = np.stack([np.full(ne, plan.slack), u], axis=1)
    elif n > 3:
        # n-1 parts: daughter i's share of the slack (the last daughter's
        # cluster is pinned, so its part only shifts the earlier clusters)
        share = rng.dirichlet(np.ones(n - 1), size=ne) * plan.slack
        above = np.cumsum(share[:, ::-1], axis=1)[:, ::-1]   # Σ_{j>=i} share
    steps = []
    for k in range(n - 1):
        m_left = np.full(ne, plan.md[k])
        if k == n - 2:
            m_right = np.full(ne, plan.md[n - 1])
        else:
            m_right = plan.suffix[k + 1] + above[:, k + 1]
        q = two_body_momentum(parent, m_left, m_right)
        weight *= q
        steps.append((m_left, m_right, q))
        parent = m_right
    return steps, weight


def _vertex_momenta(plan, steps, rng, ne, m=None):
    """Angles and boosts for one vertex — run only for ACCEPTED masses.

    ``m`` is the parent's mass IN THE FRAME IT WAS PRODUCED IN: for a sub-decay
    that is the invariant mass reconstructed from the parent's own 4-vector
    (``m_inv``), not the declared pin, so the sub-daughters sum back to exactly
    the parent's momentum.  The two agree to ~1e-16 and the MASS pass always
    uses the declared value (that is what the windows and the envelope are
    built from).
    """
    daughters = plan.kids
    mom, P = {}, np.zeros((ne, 4))
    P[:, 0] = np.broadcast_to(np.asarray(plan.m if m is None else m, float),
                              (ne,))
    for k in range(len(steps)):
        m_left, m_right, q = steps[k]
        beta = P[:, 1:] / P[:, 0:1]
        u = _unit_vec(rng, ne)
        p_l = np.concatenate([np.sqrt(q * q + m_left * m_left)[:, None],
                              q[:, None] * u], axis=-1)
        mom[daughters[k]] = boost(p_l, beta)
        if k == len(steps) - 1:
            p_r = np.concatenate([np.sqrt(q * q + m_right * m_right)[:, None],
                                  -q[:, None] * u], axis=-1)
            mom[daughters[k + 1]] = boost(p_r, beta)
        else:
            P = P - mom[daughters[k]]
    return mom


def _vertex_masses_accepted(plan, rng, n):
    """``n`` ACCEPTED mass records for ONE vertex, against its own envelope.

    Per-vertex acceptance: the vertices are independent, so each can consume
    its own weight and hand back a flat sample — no global ``Π w`` and no
    sibling's work thrown away.  Cost is ``1/r`` draws for THIS vertex rather
    than the whole event's ``Π 1/r``.
    """
    ceiling = plan.ceiling
    if not ceiling > 0:
        raise RuntimeError(
            "phase space: this vertex has zero weight everywhere — its parent "
            "is at or below the daughters' threshold")
    parts, got, block = [], 0, max(4096, n)
    while got < n:
        steps, w = _vertex_masses(plan, rng, block)
        keep = w / ceiling >= rng.uniform(0.0, 1.0, block)
        got_i = int(keep.sum())
        if got_i:
            parts.append(_subset_steps(steps, keep))
            got += got_i
        else:                      # a ceiling far above the truth must not spin
            block = min(4 * block, 1 << 23)
    return _concat_steps(parts, n)


def _node_masses(space, name, rng, n):
    """Accepted mass records for every vertex under ``name``, each independently."""
    records = {name: _vertex_masses_accepted(space._plan[name], rng, n)}
    for d in space._plan[name].kids:
        if not space.tree.is_final(d):
            records.update(_node_masses(space, d, rng, n))
    return records


def _node_momenta(space, name, records, rng, ne, m=None):
    """4-momenta of every leaf under ``name``, in ``name``'s rest frame."""
    mom = _vertex_momenta(space._plan[name], records[name], rng, ne, m)
    out = {}
    for d in space._plan[name].kids:
        p4 = mom[d]
        if space.tree.is_final(d):
            out[d] = p4
        else:
            beta = p4[:, 1:] / p4[:, 0:1]
            m_inv = np.sqrt(np.maximum(
                p4[:, 0] ** 2 - np.sum(p4[:, 1:] ** 2, axis=-1), 0.0))
            sub = _node_momenta(space, d, records, rng, ne, m_inv)
            for k2, v in sub.items():
                out[k2] = boost(v, beta)
    return out


def _subset_steps(steps, keep):
    """One vertex's accepted candidates."""
    return [(a[keep], b[keep], c[keep]) for (a, b, c) in steps]


def _concat_steps(parts, n):
    """Join the accepted blocks of one vertex and trim to ``n`` events."""
    out = []
    for i in range(len(parts[0])):
        cols = [p[i] for p in parts]
        out.append(tuple(np.concatenate([c[j] for c in cols])[:n]
                         for j in range(3)))
    return out


# ════════════════════════════════════════════════════════════════════
# public entry
# ════════════════════════════════════════════════════════════════════

def fix_beam(px, py=0.0, pz=0.0):
    """The TOP's momentum (GeV/c) as a fixed ``top_p`` callable.

    ``top_p`` is a per-event contract (``(n, rng) -> (n, 3)``) and the
    value it returns is a MOMENTUM, not a velocity — the readable quantity, and
    the one the sampler can use without you restating ``m0``: it forms
    ``E = sqrt(p^2 + m0^2)`` and boosts by ``p/E``::

        from tabpwa.phasespace import generate_phsp, fix_beam

        # BESIII at the J/psi: the +/-11 mrad beam crossing gives the
        # resonance a fixed p_x = 34 MeV/c = 0.034 GeV/c
        mom = generate_phsp(3.097, [0.139, 0.139, 0.547], 10_000,
                            top_p=fix_beam(0.034, 0.0, 0.0))

    so the boosted event sums to exactly this momentum and to
    ``E = sqrt(p^2 + m0^2)`` — the two numbers you supplied, nothing to invert.

    ``rng`` is accepted and ignored: a fixed momentum is not a draw.
    """
    p = np.array([px, py, pz], float)
    if p.shape != (3,):
        raise ValueError(
            f"phase space: fix_beam needs 3 momentum components, got {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError(f"phase space: fix_beam got a non-finite momentum {p!r}")

    def fixed_beam(n, rng=None):
        return np.tile(p, (int(n), 1))

    return fixed_beam


# ════════════════════════════════════════════════════════════════════
# the rejection envelope
# ════════════════════════════════════════════════════════════════════

#: Nodes of the parent-mass table ``C_k(p)``.  MEASURED convergence — the
#: envelope SHRINKS as this grows (the coarse grid's linear interpolation
#: over-estimates): 3.3e-4 relative between 64 and 512 nodes at 5 daughters,
#: 2.2e-3 at 7.  With 256 nodes the residual is 4e-7 / 1.4e-5, covered by the
#: 1.001 in ``_weight_envelope``.
_ENV_NGRID = 256
#: Coarse points used to BRACKET the inner maximum before the ternary search.
_ENV_BRACKET = 48
#: Ternary iterations on that bracket (1e-9 of its width).
_ENV_ITER = 45
#: Margin for a vertex whose supremum is TABULATED in the parent (4+ daughters):
#: the solve is exact in the daughter mass and interpolated in the parent, so it
#: can sit BELOW the true supremum by the interpolation error — MEASURED 6e-7
#: relative (a 3-body draw above the raw value by that much).  1e-3 covers it
#: with room, and the cost is acceptance rate, not correctness.
_ENV_MARGIN = 1.001
#: Margin for an EXACT supremum (2-3 daughters): only the arithmetic slack of a
#: closed form, no interpolation to cover.
_ENV_EXACT_MARGIN = 1.0 + 1e-9


def _three_body_max(M, md):
    """Exact ``max_x q(M, m0, x) · q(x, m1, m2)`` — the stationary quartic.

    With ``u = x²``, ``g(u) = P(u)·Q(u)/u`` where ``P = u² − 2Au + C``
    (``A = M²+m0²``, ``C = (M²−m0²)²``) and ``Q = u² − 2Bu + D``
    (``B = m1²+m2²``, ``D = (m1²−m2²)²``), so ``g' = 0`` is the quartic

        3u⁴ − 4(A+B)u³ + (C + D + 4AB)u² − CD = 0

    Its real roots inside the window, plus the window's ends, are the only
    candidates.  MEASURED against a 2 000 001-point scan: ≤ 1e-13 relative,
    degenerate cases (``m0 = 0``, ``m1 = 0``, equal masses) included.
    """
    m0, m1, m2 = md
    A, C = M * M + m0 * m0, (M * M - m0 * m0) ** 2
    B, D = m1 * m1 + m2 * m2, (m1 * m1 - m2 * m2) ** 2
    u_lo, u_hi = (m1 + m2) ** 2, (M - m0) ** 2
    if u_hi <= u_lo:
        return 0.0

    def at(u):
        x = np.sqrt(u)
        return float(two_body_momentum(M, m0, x)
                     * two_body_momentum(x, m1, m2))

    best = max(at(u_lo), at(u_hi))
    try:
        roots = np.roots([3.0, -4.0 * (A + B), C + D + 4.0 * A * B, 0.0,
                          -C * D])
    except np.linalg.LinAlgError:                 # LAPACK edge case: scan
        roots = []
        best = max(best, max(at(u) for u in np.linspace(u_lo, u_hi, 65)))
    for r in roots:
        if abs(r.imag) <= 1e-9 * max(1.0, abs(r.real)):
            u = float(r.real)
            if u_lo < u < u_hi:
                best = max(best, at(u))
    return best


def _fold_envelope(m, kids, tree):
    """``(supremum, exact)`` of ``Π q`` over ONE vertex's fold.

    ``exact`` says whether a margin for grid interpolation is needed (see
    ``_ENV_MARGIN`` / ``_ENV_EXACT_MARGIN``): 2 and 3 daughters are closed form.

    ``q(A, m1, m2)`` is monotone: DECREASING in the daughter mass (so a factor's
    own maximum is at its window's low edge) and INCREASING in the parent.
    The fold's maximum is therefore the peak of a decreasing factor against an
    increasing remainder, not a product of independent maxima — that product
    is exact but useless here (MEASURED 66x above the real maximum at 5
    daughters and 8.6e3x at 7 for the shapes used in this repo, because each
    factor's maximum needs its cluster mass at 0, which zeroes the next one;
    acceptance 8e-4 / ~0).

    With the simplex proposal (see ``_vertex_masses``) the weight is ``Π q``
    alone, so the maximum is taken level by level::

        C_last(p) = q(p, m_d, m_last)                      (increasing in p)
        C_k(p)    = max_x q(p, m_d, x) · C_{k+1}(x)

    ``C_{k+1}`` is tabulated on the parent grid and interpolated, the peak
    over the window is bracketed coarsely and refined by ternary search — the
    product is unimodal, and the bracket keeps a non-unimodal case honest.
    Each vertex is accepted on its OWN envelope, so this deliberately covers
    one vertex only — the sub-decays have their own.
    """
    n = len(kids)
    md = np.array([tree.fixed[d] for d in kids], float)
    suffix = np.zeros(n + 1)
    for k in range(n - 1, -1, -1):
        suffix[k] = suffix[k + 1] + md[k]

    if n == 2:
        return float(two_body_momentum(m, md[0], md[1])), True
    if n == 3:
        return _three_body_max(m, md), True

    def grid(k):
        lo, hi = suffix[k], m - (suffix[0] - suffix[k])
        return np.linspace(lo, hi, _ENV_NGRID) if hi > lo else np.array([lo])

    p_prev = grid(n - 2)
    C = np.asarray(two_body_momentum(p_prev, md[n - 2], md[n - 1]), float)
    fraction = np.linspace(0.0, 1.0, _ENV_BRACKET)[None, :]
    for k in range(n - 3, -1, -1):
        p_grid = np.array([m], float) if k == 0 else grid(k)
        x_lo = np.full(p_grid.shape, suffix[k + 1], float)
        x_hi = p_grid - md[k]
        live = x_hi > x_lo

        def value(x, pg):
            """``q(pg, m_d, x) · C(x)`` — the factor times the remainder."""
            return np.asarray(two_body_momentum(pg, md[k], x), float) \
                * np.interp(x, p_prev, C, left=0.0, right=0.0)

        xs = x_lo[:, None] + (x_hi - x_lo)[:, None] * fraction
        vals = np.where(live[:, None], value(xs, p_grid[:, None]), 0.0)
        j = np.argmax(vals, axis=1)
        rows = np.arange(p_grid.shape[0])
        a = np.where(live, xs[rows, np.maximum(j - 1, 0)], x_lo)
        b = np.where(live, xs[rows, np.minimum(j + 1, _ENV_BRACKET - 1)], x_lo)
        for _ in range(_ENV_ITER):                 # ternary, vectorised over p
            m1 = a + (b - a) / 3.0
            m2 = b - (b - a) / 3.0
            take = value(m1, p_grid) < value(m2, p_grid)
            a = np.where(take, m1, a)
            b = np.where(take, b, m2)
        refined = value(0.5 * (a + b), p_grid)
        best = np.where(live, np.maximum(vals[rows, j], refined), 0.0)
        C = best
        p_prev = p_grid
    return float(C[0]), False


class PhaseSpace:
    """A mass-tree topology COMPILED ONCE — parse, windows and ceilings included.

    ``__init__`` parses the tree, validates it (the threshold at every pinned
    node) and solves each vertex's rejection envelope.  ``generate`` can then be
    called repeatedly — per batch, per loop, per toy run — without paying any of
    that again.  MEASURED fixed cost: 1.3 / 4.3 / 8.1 ms for 3 / 5 / 7
    daughters, i.e. a third of a whole call at ``n = 400``.

    ``generate_phsp(m0, mass_tree, n_events, ...)`` is the one-shot wrapper.

    ``top_p`` is fixed at construction (it belongs to the frame you generate
    into) and is a CALLABLE ``(n, rng) -> (n, 3)`` of the top's MOMENTUM in
    GeV/c, or ``None`` for the rest frame; :func:`fix_beam` builds a fixed one::

        # BESIII at the J/psi: the +/-11 mrad crossing gives a fixed
        # p_x = 34 MeV/c; E = sqrt(p^2 + m0^2) is formed here
        space = PhaseSpace(3.097, [0.139, 0.139, 0.547], top_p=fix_beam(0.034))
        mom = space.generate(10_000, seed=1)
        more = space.generate(10_000, rng=...)      # continues the same stream

    Momentum, not velocity: the sampler already knows ``m0``, so a boosted
    event sums to exactly the momentum you gave and to that ``E``.
    """

    def __init__(self, m0, mass_tree, top_p=None):
        if top_p is not None and not callable(top_p):
            raise ValueError(
                "phase space: top_p is None or a CALLABLE (n, rng) -> (n, 3) "
                "of the top's MOMENTUM in GeV/c — for a fixed one use "
                f"fix_beam(...); got {top_p!r}")
        self.top, self.dmap, self.fixed, self.leaves = _parse_tree(m0, mass_tree)
        self.tree = _Tree(self.top, self.dmap, self.fixed)
        self.m0 = float(self.fixed[self.top])
        self.top_p = top_p
        if self.m0 <= self.tree.lo(self.top) + 1e-12:
            raise ValueError(
                f"phase space: m0 = {self.m0:.6g} is at or below the sum of the "
                f"final masses {self.tree.lo(self.top):.6g}; there is no phase "
                f"space to sample")
        for name, kids in self.dmap.items():
            # a PINNED node can be infeasible on its own; say so now rather than
            # thousands of draws later as a zero weight with no node named
            need = float(sum(self.tree.lo(c) for c in kids))
            # AT threshold is as empty as below it: the daughters are at rest
            # and every weight is 0 — the same rule the top gets above
            if name != self.top and need >= self.fixed[name] - 1e-12:
                raise ValueError(
                    f"phase space: the pinned node {name!r} has mass "
                    f"{self.fixed[name]:.6g} but its daughters need {need:.6g}; "
                    f"there is no phase space for that node")
        self._plan = {}
        self._compile(self.m0, self.top)

    def _compile(self, m, name):
        """Solve this vertex once: slack, lower edges, rejection ceiling."""
        self._plan[name] = _VertexPlan(m, self.dmap[name], self.tree)
        for d in self.dmap[name]:
            if not self.tree.is_final(d):
                self._compile(self.fixed[d], d)

    def describe(self):
        return (f"phase space {self.m0:.6g} -> {len(self.leaves)} finals, "
                f"{len(self._plan)} vertex/vertices, ceilings "
                + ", ".join(f"{n!r}:{p.ceiling:.4g}"
                            for n, p in self._plan.items()))

    def generate(self, n_events, seed=None, weights_out=False, batch=200_000,
                 rng=None, top_p=None):
        """Generate ``n_events`` events for the compiled topology.

        ``seed`` starts a fresh stream; ``rng=`` continues an existing one, so a
        second call after ``generate(1000, seed=1)`` can keep drawing from the
        same stream instead of replaying the first 1000.

        Returns a LIST of ``(n_events, 4)`` arrays, one per final, in
        depth-first leaf order (or ``(momenta, ones)`` with ``weights_out``).
        ``batch`` bounds the working set per chunk, not the result.

        The sample is generated in the top rest frame and then boosted by
        ``top_p`` (fixed at construction), so the momenta come back in the
        frame that momentum names; ``dΦ`` is Lorentz invariant, so the weights
        stay 1.  The event-buffer builders take ``cm_boost=True`` to put each
        event back in its own CM.
        """
        if n_events < 0:
            raise ValueError("phase space: n_events must be >= 0")
        if seed is not None and rng is not None:
            raise ValueError(
                "phase space: give seed= (a fresh stream) or rng= (continue "
                "one), not both")
        beam = self.top_p if top_p is None else top_p
        if beam is not None and not callable(beam):
            raise ValueError(
                "phase space: top_p is None or a CALLABLE (n, rng) -> (n, 3) "
                "of the top's MOMENTUM in GeV/c — for a fixed one use "
                f"fix_beam(...); got {beam!r}")
        if batch < 1:
            raise ValueError(f"phase space: batch must be >= 1, got {batch!r}")
        if n_events == 0:
            empty = [np.empty((0, 4)) for _ in self.leaves]
            return (empty, np.ones(0)) if weights_out else empty
        if rng is None:
            rng = np.random.RandomState(seed) if seed is not None \
                else np.random.RandomState()

        M, top, leaves = self.m0, self.top, self.leaves
        chunks, got = [], 0
        while got < n_events:
            n = min(batch, max(4096, n_events - got))
            # every vertex accepts its own masses against its own solved
            # envelope; the masses are the cheap half, and the angles/boosts
            # are paid once, for the events that survived
            records = _node_masses(self, top, rng, n)
            mom = _node_momenta(self, top, records, rng, n)
            if beam is None:
                chunks.append([mom[f] for f in leaves])
            else:
                p_top = np.asarray(beam(n, rng), float)
                if p_top.shape != (n, 3):
                    raise ValueError(
                        f"phase space: top_p must give ({n}, 3) momenta, "
                        f"got {p_top.shape}")
                p2 = np.einsum("ij,ij->i", p_top, p_top)
                if not np.isfinite(p2).all():
                    raise ValueError("phase space: top_p must be finite")
                # m0 is known, so the momentum IS the boost: E = sqrt(p^2+m0^2)
                beta = p_top / np.sqrt(p2 + M * M)[:, None]
                chunks.append([boost(mom[f], beta) for f in leaves])
            got += n
        out = [np.concatenate([c[j] for c in chunks], axis=0)[:n_events]
               for j in range(len(leaves))]
        if weights_out:
            return out, np.ones(n_events)
        return out


def generate_phsp(m0, mass_tree, n_events, seed=None, weights_out=False,
                  batch=200_000, top_p=None):
    """One-shot flat phase space — ``PhaseSpace(...).generate(...)``.

    Use :class:`PhaseSpace` directly when you will generate more than once: it
    parses the tree and solves the rejection envelopes once, in ``__init__``.
    """
    return PhaseSpace(m0, mass_tree, top_p=top_p).generate(
        n_events, seed=seed, weights_out=weights_out, batch=batch)

"""
helicity_angle — numerical helicity-amplitude angular engine.

Implements the angular part of a sequential two-body decay chain in the
helicity formalism, mirroring the "Angular Formula Calculator" tool
(ptoject61/tools/get-angle-calculator.html and its assets/js engine):

    T^{l,s}_{λa}(φ,θ) = √((2l+1)/(2J_a+1))
                       · ⟨J_b,λb; J_c,−λc | s,δ⟩     (Clebsch–Gordan)
                       · ⟨l,0; s,δ | J_a,δ⟩           (Clebsch–Gordan)
                       · D^{J_a}*_{λ_a,δ}(φ,θ)

with  δ = λb − λc  and D the (conjugate) Wigner D-matrix in the physics
convention  D^{J}_{m,m'}(φ,θ,0) = e^{−imφ} d^{J}_{m,m'}(θ).

Everything is NUMERICAL (double precision, no sympy): the CG coefficients
and Wigner d-functions are evaluated directly, and cascades are combined by
summing over the internal (intermediate) helicities.  This gives the
per-event angular amplitude for a fixed *external helicity configuration*
(a projection) and a fixed partial wave (one (l,s) per decay vertex):

    A(e) = Σ_{internal λ}  ∏_{vertices v}  T^{l_v,s_v}_{λ_v}(θ_v(e), φ_v(e))

Angle variables are indexed by decay order: vertex v (0-based, in the order
the decays appear in the chain) owns the pair (θ_v, φ_v).

Spins may be integer or half-integer.
"""

import math
from fractions import Fraction

import numpy as np

# ---------------------------------------------------------------------------
# Spin helpers (values kept as Fraction)
# ---------------------------------------------------------------------------

def to_spin(x):
    """int/float/Fraction/'1/2'-like → Fraction spin."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, str):
        return Fraction(x) if '/' in x else Fraction(int(x), 1)
    if isinstance(x, (int, np.integer)):
        return Fraction(int(x), 1)
    if isinstance(x, float):
        return Fraction(x).limit_denominator(200)
    return Fraction(x)


def helicity_values(j, spins=None):
    """Allowed helicities of a particle of spin *j*.

    If *spins* is given (a list like ``[-1, 1]`` or ``["-1/2","1/2"]``) those
    values are used verbatim (e.g. transverse-only spin-1 → ±1).  Otherwise
    the full set −j, −j+1, …, j is returned.
    """
    j = to_spin(j)
    if spins is not None:
        return [to_spin(x) for x in spins]
    out = []
    j2 = int(2 * j)
    for m2 in range(-j2, j2 + 1, 2):
        out.append(Fraction(m2, 2))
    return out


def ls_combinations(J_a, J_b, J_c):
    """All (l, s) partial waves for decay a → b + c (triangle conditions).

    l (orbital) integer, s = j_b ⊕ j_c may be half-integer.
    """
    Ja, Jb, Jc = to_spin(J_a), to_spin(J_b), to_spin(J_c)
    Ja2, Jb2, Jc2 = int(2 * Ja), int(2 * Jb), int(2 * Jc)
    out = []
    for s2 in range(abs(Jb2 - Jc2), Jb2 + Jc2 + 1, 2):
        s = Fraction(s2, 2)
        # l integer with |l−s| ≤ Ja ≤ l+s  ⇒  l ∈ [|Ja−s|, Ja+s]
        lmin = int(math.ceil(abs(Ja2 - s2) / 2.0))
        lmax = (Ja2 + s2) // 2
        for l in range(max(0, lmin), lmax + 1):
            if abs(2 * l - s2) <= Ja2 <= 2 * l + s2:
                out.append((l, s))
    return out


# ---------------------------------------------------------------------------
# Clebsch–Gordan coefficient  ⟨j1,m1; j2,m2 | j3,m3⟩  (real, numeric)
# ---------------------------------------------------------------------------
def _lf(x2):
    """ln( (x2/2)! ) for an integer *x2* (factorial of an integer or of a
    half-integer argument).  Negative → None (caller skips the term)."""
    if x2 < 0:
        return None
    return math.lgamma(x2 / 2.0 + 1.0)


def cg(j1, m1, j2, m2, j3, m3):
    """Numeric Clebsch–Gordan ⟨j1,m1; j2,m2 | j3,m3⟩ (Racah form).

    All spins handled in units of 1/2; factorials of (possibly half-integer)
    arguments are evaluated with ``math.lgamma``.  Validated against the
    closed-form trivial couplings and against orthonormality (Σ|CG|² = 1).
    """
    j1, m1, j2, m2, j3, m3 = (to_spin(x) for x in (j1, m1, j2, m2, j3, m3))
    J1, M1 = int(2 * j1), int(2 * m1)
    J2, M2 = int(2 * j2), int(2 * m2)
    J3, M3 = int(2 * j3), int(2 * m3)
    # projection legality: |m| ≤ j and m shares j's integer/half-integer
    # character (2j − 2m must be an even integer → J−M even).
    if (J1 - M1) % 2 or (J1 + M1) % 2 or (J2 - M2) % 2 or (J2 + M2) % 2 \
            or (J3 - M3) % 2 or (J3 + M3) % 2:
        return 0.0
    if M1 + M2 != M3:
        return 0.0
    if not (abs(J1 - J2) <= J3 <= J1 + J2):
        return 0.0

    log_delta = (_lf(J1 + J2 - J3) + _lf(J1 - J2 + J3) + _lf(-J1 + J2 + J3)
                 - _lf(J1 + J2 + J3 + 2))
    log_num = (_lf(J3 + M3) + _lf(J3 - M3) + _lf(J1 + M1) + _lf(J1 - M1)
               + _lf(J2 + M2) + _lf(J2 - M2))
    prefac = math.exp(0.5 * (math.log(J3 + 1) + log_delta + log_num))

    # factorial-argument bounds in units of 1/2 (J* = 2j etc.):
    #   (j1−m1−k), (j2+m2−k), (j1+j2−j3−k) ≥ 0 ⇒ k ≤ those/2 (floor)
    #   (j3−j2+m1+k), (j3−j1−m2+k) ≥ 0       ⇒ k ≥ −those/2 (ceil)
    from math import floor as _fl, ceil as _cl
    kmin = max(0,
               _cl((J2 - J3 - M1) / 2.0),
               _cl((J1 - J3 + M2) / 2.0))
    kmax = min(_fl((J1 + J2 - J3) / 2.0),
               _fl((J1 - M1) / 2.0),
               _fl((J2 + M2) / 2.0))
    s = 0.0
    for k in range(kmin, kmax + 1):
        den = (_lf(2 * k) + _lf(J1 + J2 - J3 - 2 * k) + _lf(J1 - M1 - 2 * k)
               + _lf(J2 + M2 - 2 * k) + _lf(J3 - J2 + M1 + 2 * k)
               + _lf(J3 - J1 - M2 + 2 * k))
        s += (-1.0 if k % 2 else 1.0) * math.exp(-den)
    return prefac * s


# ---------------------------------------------------------------------------
# Wigner small-d  d^{j}_{m,m'}(β)   (integer & half-integer j)
# ---------------------------------------------------------------------------
# d^j_{m,m'}(β) = ⟨j,m| e^{−iβ J_y} |j,m'⟩.  J_y is Hermitian (pure-imaginary,
# symmetric in the Condon–Shortley |j,m⟩ basis), so e^{−iβJ_y} is computed by
# diagonalising J_y = Q·Λ·Q† and exponentiating the eigenvalues.
_jy_cache = {}
_dy_cache = {}


def _jy_hermitian(j):
    """Hermitian matrix of J_y in the |j,m⟩ basis (Condon–Shortley)."""
    j = to_spin(j)
    if j in _jy_cache:
        return _jy_cache[j]
    dim = int(2 * j) + 1
    ms = [j - Fraction(k, 1) for k in range(dim)]
    J = np.zeros((dim, dim), dtype=np.complex128)
    for a, m in enumerate(ms):
        # raising |m⟩ → |m+1⟩ : J_y = (J_+ − J_−)/(2i)
        # ⟨m+1|J_y|m⟩ = −i/2·√[(j−m)(j+m+1)]   (row index of m+1 is a−1)
        f_up = (j - m) * (j + m + 1)
        if f_up > 0 and a - 1 >= 0:
            J[a - 1, a] = -0.5j * math.sqrt(float(f_up))
        # lowering |m⟩ → |m−1⟩ : ⟨m−1|J_y|m⟩ = +i/2·√[(j+m)(j−m+1)]
        f_dn = (j + m) * (j - m + 1)
        if f_dn > 0 and a + 1 < dim:
            J[a + 1, a] = 0.5j * math.sqrt(float(f_dn))
    _jy_cache[j] = J
    return J


def wigner_d(j, m, mp, beta):
    """d^j_{m,m'}(β), matching sympy's ``wigner_d`` convention.

    sympy's small-d matrix is the transpose of ``⟨j,m|e^{−iβJ_y}|j,m'⟩``
    (i.e. e^{−iβJ_y} with the m-index order flipped), so we return the
    transposed element.
    """
    j = to_spin(j)
    m, mp = to_spin(m), to_spin(mp)
    J, M, Mp = int(2 * j), int(2 * m), int(2 * mp)
    if abs(M) > J or abs(Mp) > J or (J - M) % 2 or (J - Mp) % 2:
        return 0.0
    a = int(j - m)
    b = int(j - mp)
    key = (j, round(float(beta), 12))
    if key not in _dy_cache:
        Jy = _jy_hermitian(j)
        evals, evecs = np.linalg.eigh(Jy)          # Jy = evecs·diag·evecs†
        phase = np.exp(-1j * float(beta) * evals)
        U = (evecs * phase) @ evecs.conj().T
        _dy_cache[key] = U
    return float(np.real(_dy_cache[key][b, a]))     # transposed indexing


def wigner_D(j, m, mp, phi, theta):
    """D^j_{m,mp}(φ,θ,0) = e^{−imφ} d^j_{m,mp}(θ)."""
    m = to_spin(m)
    ph = complex(math.cos(-float(m) * phi), math.sin(-float(m) * phi))
    return ph * wigner_d(j, m, mp, theta)


def wigner_D_conj(j, m, mp, phi, theta):
    """D^{j*} = e^{+imφ} d^j_{m,mp}(θ) (conjugate of wigner_D)."""
    m = to_spin(m)
    ph = complex(math.cos(float(m) * phi), math.sin(float(m) * phi))
    return ph * wigner_d(j, m, mp, theta)


# ---------------------------------------------------------------------------
# Per-vertex helicity amplitude (one (l,s) partial wave)
# ---------------------------------------------------------------------------
def vertex_amplitude(J_a, J_b, J_c, la, lb, lc, l, s, phi, theta):
    """T^{l,s} for decay a→b+c with helicities (la,lb,lc).

    Returns the complex angular factor (JS "T" of the calculator).
    """
    delta = to_spin(lb) - to_spin(lc)
    la = to_spin(la)
    if abs(delta) > to_spin(J_a):
        return 0.0
    cg1 = cg(J_b, lb, J_c, -to_spin(lc), s, delta)
    if cg1 == 0.0:
        return 0.0
    cg2 = cg(l, 0, s, delta, J_a, delta)
    if cg2 == 0.0:
        return 0.0
    l, s = int(l), to_spin(s)
    Ja = to_spin(J_a)
    coeff = math.sqrt((2 * l + 1) / float(2 * Ja + 1)) * cg1 * cg2
    return coeff * wigner_D_conj(J_a, la, delta, phi, theta)


# ---------------------------------------------------------------------------
# Decay-tree helpers
# ---------------------------------------------------------------------------
# A decay tree is a nested structure:
#     (J_parent, [child0, child1])   — a two-body decay vertex
#     J_parent                       — a stable (final) particle
# e.g.  (0, [(1, [0, 0]), (1, [0, 0])])   (top J=0 → ρ₁(1)+ρ₂(1), each → 0+0)
#       (1, [(Jres, [0, 0]), 0])          (J/ψ-like: top 1 → res(J)+0, res→0+0)
#
# Vertex v (0-based, DFS pre-order over the *decaying* nodes, child-0 subtree
# first) owns the angle pair (φ_v, θ_v).  Final-state (leaf) particles are
# ordered likewise in pre-order.

def tree_vertices(node):
    """[(Ja, Jb, Jc), ...] of every two-body vertex, in pre-order."""
    out = []
    def walk(n):
        if isinstance(n, tuple):
            Ja, children = n
            Jb = children[0][0] if isinstance(children[0], tuple) else children[0]
            Jc = children[1][0] if isinstance(children[1], tuple) else children[1]
            out.append((to_spin(Ja), to_spin(Jb), to_spin(Jc)))
            for c in children:
                walk(c)
    walk(node)
    return out


def tree_leaves(node):
    """[J, ...] of the final-state particles, in pre-order."""
    out = []
    def walk(n):
        if isinstance(n, tuple):
            for c in n[1]:
                walk(c)
        else:
            out.append(to_spin(n))
    walk(node)
    return out


def tree_info(node):
    """Return (vertices, leaves, vmap, leaf_order) describing *node*.

    *vertices* : [(Ja,Jb,Jc)] per two-body vertex (pre-order)
    *leaves*   : [J] per final particle (pre-order)
    *vmap*     : path (tuple of 0/1 child indices from root) → vertex id
    *leaf_ids* : {path: leaf index} for the final particles
    """
    vertices = []
    leaves = []
    vmap = {}
    leaf_ids = {}

    def walk(n, path):
        if isinstance(n, tuple):
            Ja, children = n
            Jb = children[0][0] if isinstance(children[0], tuple) else children[0]
            Jc = children[1][0] if isinstance(children[1], tuple) else children[1]
            vmap[path] = len(vertices)
            vertices.append((to_spin(Ja), to_spin(Jb), to_spin(Jc)))
            walk(children[0], path + (0,))
            walk(children[1], path + (1,))
        else:
            leaf_ids[path] = len(leaves)
            leaves.append(to_spin(n))

    walk(node, ())
    return vertices, leaves, vmap, leaf_ids


def wave_ls_lists(node):
    """Per-vertex (l, s) partial-wave lists (pre-order)."""
    return [ls_combinations(Ja, Jb, Jc) for (Ja, Jb, Jc) in tree_vertices(node)]


# ---------------------------------------------------------------------------
# Numeric cascade amplitude (internal helicities summed numerically)
# ---------------------------------------------------------------------------
def amplitude(node, lss, angles, lambda_top, lambda_leaves):
    """Angular amplitude of a full cascade for one partial wave + projection.

    Args:
        node:  decay tree (see above).
        lss:   one (l, s) per vertex (pre-order) — the chain's partial wave.
        angles: dict vertex-id → (φ, θ).
        lambda_top: helicity of the top particle (a projection).
        lambda_leaves: list of helicities of the final particles (pre-order;
            the projection index enumerates the external helicity combos).

    Returns the complex amplitude; intermediate helicities are summed.
    """
    vertices, leaves, vmap, leaf_ids = tree_info(node)

    def rec(n, path, lam):
        if not isinstance(n, tuple):
            # final particle — survives iff lam equals the chosen projection
            return 1.0 if lam == lambda_leaves[leaf_ids[path]] else 0.0
        Ja, children = n
        Jb = children[0][0] if isinstance(children[0], tuple) else children[0]
        Jc = children[1][0] if isinstance(children[1], tuple) else children[1]
        vid = vmap[path]
        phi, theta = angles[vid]
        l, s = lss[vid]
        tot = 0.0 + 0.0j
        for lb in helicity_values(Jb):
            for lc in helicity_values(Jc):
                t = vertex_amplitude(Ja, Jb, Jc, lam, lb, lc, l, s,
                                     phi, theta)
                if t == 0.0:
                    continue
                f0 = rec(children[0], path + (0,), lb)
                f1 = rec(children[1], path + (1,), lc)
                if f0 == 0.0 or f1 == 0.0:
                    continue
                tot += t * f0 * f1
        return tot

    return rec(node, (), to_spin(lambda_top))


def external_helicities(node, top_spins=None, leaf_spins=None):
    """All external helicity configurations (the incoherent projections).

    Args:
        top_spins: optional list of allowed helicities of the top particle
            (the config ``spins`` entry, e.g. ``[-1, 1]``); default the full
            −J…J set.
        leaf_spins: optional list of lists, per final particle (pre-order);
            default the full −J…J set of each final.

    Returns a list of ``(lambda_top, tuple_of_leaf_lambdas)``.
    """
    _, leaves, _, _ = tree_info(node)
    top = to_spin(node[0]) if isinstance(node, tuple) else to_spin(node)
    tops = [to_spin(x) for x in (top_spins if top_spins is not None
                                 else helicity_values(top))]
    if leaf_spins is None:
        leaf_sets = [helicity_values(J) for J in leaves]
    else:
        leaf_sets = []
        for s, J in zip(leaf_spins, leaves):
            if s is None:
                leaf_sets.append(helicity_values(J))
            else:
                leaf_sets.append([to_spin(x) for x in s])
    import itertools as _it
    out = []
    for lt in tops:
        for combo in _it.product(*leaf_sets):
            out.append((lt, tuple(combo)))
    return out


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Exact, event-independent monomial expansion
# ---------------------------------------------------------------------------
# For a fixed (projection = external-helicity configuration, wave = one (l,s)
# per vertex) the angular amplitude is expanded EXACTLY into single-frequency
# trig monomials
#        ∏_v  cos/sin(f^φ_v·φ_v) · cos/sin(f^θ_v·θ_v)
# whose coefficients are pure numbers (CG × normalisation × exact Wigner-d
# single-frequency coefficients) — no event dependence anywhere.
#
# Variables are ordered per vertex as (φ_v, θ_v): variable 2v → φ_v and
# 2v+1 → θ_v.  A monomial key is a tuple of one ``(kind, freq)`` per
# variable; the identity factor is ('c', 0.0).  Frequencies are stored in
# units of 1/2 so integer AND half-integer spins share one representation.
#
# The Wigner-d frequency coefficients are themselves obtained once per
# (J, m, m′) by solving a small, event-independent linear system on the
# single-frequency basis; the cascade is then a numeric tensor contraction
# over the internal helicities of these sparse per-vertex monomials.

def _d_single_freq_parts(J, m, mp):
    """Single-frequency parts of d^J_{m,mp}(θ):
    list of (freq, kind, coeff) with d = Σ coeff·trig(freq·θ)."""
    J = to_spin(J)
    m, mp = to_spin(m), to_spin(mp)
    J2 = int(2 * J)
    funcs = [('c', 0.0)]
    for q in range(1, J2 + 1):
        funcs.append(('c', q / 2.0))
        funcs.append(('s', q / 2.0))
    F = len(funcs)
    M = 2 * J2 + 1
    xs = np.linspace(0.0, 4.0 * np.pi, M, endpoint=False)
    A = np.zeros((M, F))
    for j, (kind, f) in enumerate(funcs):
        A[:, j] = np.cos(f * xs) if kind == 'c' else np.sin(f * xs)
    y = np.array([wigner_d(J, m, mp, float(x)) for x in xs])
    coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return [(f, kind, float(c)) for (kind, f), c in zip(funcs, coefs)
            if abs(c) > 1e-10]


def _vertex_monomials(Ja, Jb, Jc, lam, lb, lc, l, s):
    """Sparse monomials of one vertex factor (only its φ,θ variables set).

    Returns a list of (coeff, phi_el, theta_el) with *el* = (kind, freq).
    """
    delta = to_spin(lb) - to_spin(lc)
    lam = to_spin(lam)
    if abs(delta) > to_spin(Ja):
        return []
    cg1 = cg(Jb, lb, Jc, -to_spin(lc), s, delta)
    cg2 = cg(l, 0, s, delta, Ja, delta)
    if cg1 == 0.0 or cg2 == 0.0:
        return []
    Ja = to_spin(Ja)
    C = math.sqrt((2 * int(l) + 1) / float(2 * Ja + 1)) * cg1 * cg2
    lamf = float(lam)
    phi_parts = [(1.0, ('c', abs(lamf)))]
    if lamf != 0.0:
        phi_parts.append((1j if lamf > 0 else -1j, ('s', abs(lamf))))
    theta_parts = _d_single_freq_parts(Ja, lam, delta)
    out = []
    for pc, (pk, pf) in phi_parts:
        for tf, tk, tc in theta_parts:
            out.append((C * pc * tc, (pk, pf), (tk, tf)))
    return out


_ID = lambda V: tuple(('c', 0.0) for _ in range(V))


def amplitude_monomials(node, lss, lambda_top, lambda_leaves, tol=1e-10):
    """Exact monomial expansion of one (projection, wave) amplitude.

    Args:
        node: decay tree.
        lss: (l, s) per vertex (pre-order).
        lambda_top, lambda_leaves: external helicity configuration.
        tol: coefficients with |c| < tol·max|c| are dropped.

    Returns ``(var_order, {key: complex})`` where *var_order* describes the
    variables: [(variable_id, vertex, 'phi'|'theta'), ...] in phi,theta order
    and each *key* is a tuple of ``(kind, freq)`` for those variables.
    """
    vertices, leaves, vmap, leaf_ids = tree_info(node)
    V = 2 * len(vertices)
    ident = _ID(V)
    var_order = []
    for v in range(len(vertices)):
        var_order.append((2 * v, v, 'phi'))
        var_order.append((2 * v + 1, v, 'theta'))

    def compose(phi_el, theta_el, k0, k1, vid):
        key = []
        for i in range(V):
            if i == 2 * vid:
                key.append(phi_el)
            elif i == 2 * vid + 1:
                key.append(theta_el)
            else:
                a = k0[i]
                b = k1[i]
                key.append(a if a[1] != 0.0 else (b if b[1] != 0.0
                                                  else ('c', 0.0)))
        return tuple(key)

    def add(acc, key, val):
        old = acc.get(key)
        acc[key] = val if old is None else old + val

    def rec(n, path, lam):
        if not isinstance(n, tuple):
            if lam == lambda_leaves[leaf_ids[path]]:
                return {ident: 1.0}
            return {}
        Ja, children = n
        Jb = children[0][0] if isinstance(children[0], tuple) else children[0]
        Jc = children[1][0] if isinstance(children[1], tuple) else children[1]
        vid = vmap[path]
        l, s = lss[vid]
        out = {}
        for lb in helicity_values(Jb):
            for lc in helicity_values(Jc):
                vm = _vertex_monomials(Ja, Jb, Jc, lam, lb, lc, l, s)
                if not vm:
                    continue
                c0 = rec(children[0], path + (0,), lb)
                c1 = rec(children[1], path + (1,), lc)
                if not c0 or not c1:
                    continue
                for (coef, pel, tel) in vm:
                    for k0, v0 in c0.items():
                        for k1, v1 in c1.items():
                            add(out, compose(pel, tel, k0, k1, vid),
                                coef * v0 * v1)
        return out

    raw = rec(node, (), to_spin(lambda_top))
    maxc = max((abs(x) for x in raw.values()), default=0.0)
    thr = tol * maxc
    keep = {k: v for k, v in raw.items() if abs(v) > thr}
    return var_order, keep


# ---------------------------------------------------------------------------
# ampfit DecayChain adapters
# ---------------------------------------------------------------------------
# ampfit's own decay model already is the tree: a chain's ``.decays`` list is
# a DFS pre-order of its two-body vertices (parent before children, child-0
# subtree before child-1 subtree), so vertex v ↔ ``chain.decays[v]`` and the
# per-vertex partial waves are ``Decay.get_ls_list()`` (parity / ``p_break``
# already applied).  We adapt it directly — no separate tree construction.

def _spin_of(particle):
    """Numeric spin of an ampfit Particle (int/float/Fraction-like)."""
    from fractions import Fraction as _Fr
    j = particle.J
    if isinstance(j, _Fr):
        return j
    if hasattr(j, 'numerator') and hasattr(j, 'denominator'):   # sympy Rational
        return _Fr(int(j.numerator), int(j.denominator))
    return to_spin(j)


def decay_chain_to_tree(chain):
    """DecayChain → nested (J, [child0, child1]) spin tree (pre-order).

    The recursion consumes ``chain.decays`` in list order; this is exactly
    the DFS pre-order the tree helpers assume, so vertex index == position in
    ``chain.decays``.
    """
    decays = chain.decays
    by_core = {}
    for i, d in enumerate(decays):
        by_core.setdefault(d.core.name, []).append(i)
    used = [False] * len(decays)

    def build(name):
        idxs = by_core.get(name)
        if not idxs:
            return _spin_of(_particle_by_name(chain, name))
        i = idxs.pop(0)
        used[i] = True
        d = decays[i]
        return (_spin_of(d.core),
                [build(o.name) for o in d.outs])

    tree = build(chain.top)
    if not all(used):
        raise ValueError(
            "DecayChain.decays is not a DFS pre-order of the tree")
    return tree


def _particle_by_name(chain, name):
    for d in chain.decays:
        if d.core.name == name:
            return d.core
        for o in d.outs:
            if o.name == name:
                return o
    raise KeyError(name)


def decay_chain_leaves(chain):
    """Final-state ampfit Particles of a chain, in pre-order."""
    decays = chain.decays
    by_core = {}
    for i, d in enumerate(decays):
        by_core.setdefault(d.core.name, []).append(i)
    used = [False] * len(decays)
    leaves = []

    def collect(name):
        idxs = by_core.get(name)
        if not idxs:
            leaves.append(_particle_by_name(chain, name))
            return
        i = idxs.pop(0)
        used[i] = True
        for o in decays[i].outs:
            collect(o.name)

    collect(chain.top)
    if not all(used):
        raise ValueError("DecayChain.decays is not a DFS pre-order of the tree")
    return leaves


def decay_chain_ls_sets(chain):
    """Per-vertex (l, s) wave lists, in chain.decays order (parity applied)."""
    return [list(d.get_ls_list()) for d in chain.decays]


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Canonical angle layout + gauge fixing
# ---------------------------------------------------------------------------
# Natural per-vertex (φ_v, θ_v) order has 2·Nv slots:
#     [φ0, θ0, φ1, θ1, φ2, θ2, …]      (slot 2v = φ_v, 2v+1 = θ_v)
#
# Gauge rule (top J = 0): the overall rotation of the event is unobservable,
# so the first three slots (φ0, θ0 of the top vertex and φ1 of the first
# sub-vertex, fixed to 0 by the global rotation) are dropped.  Cos factors on
# a dropped slot collapse to 1; sin factors kill the whole monomial.
#
# The canonical storage layout is then the grouped "phi first, then theta":
#     [φ…, θ…]   (each block ordered by vertex)
# which reproduces the old cache ordering (e.g. ρρ: [φ₂, θ₁, θ₂]).

def canonical_variables(n_vertices, top_j0=True):
    """Gauge-fixed canonical variable description.

    Returns a list of (vertex, 'phi'|'theta') describing the per-event angle
    columns (phi block first, then theta block, each by vertex order).
    """
    natural = [(v, k) for v in range(n_vertices) for k in ('phi', 'theta')]
    if top_j0:
        natural = natural[3:]          # drop φ0, θ0, φ1
    phis = [(v, k) for (v, k) in natural if k == 'phi']
    thetas = [(v, k) for (v, k) in natural if k == 'theta']
    return phis + thetas


def _reduce_layout(mono, n_vertices, drop, phi_first=True):
    """Reduce a full-2Nv monomial dict by fixing *drop* slots to 0 and
    (optionally) regrouping the rest as phi-then-theta.

    Returns ``(layout, {key: coeff})`` with keys over the reduced layout.
    """
    full = 2 * n_vertices
    natural = [(v, k) for v in range(n_vertices) for k in ('phi', 'theta')]
    drop = set(drop)
    keep = [i for i in range(full) if i not in drop]
    layout0 = [natural[i] for i in keep]
    out = {}
    for key, coef in mono.items():
        if any(key[i][0] == 's' for i in drop):
            continue
        nk = tuple(key[i] for i in keep)
        out[nk] = out.get(nk, 0.0) + coef
    if not phi_first:
        return layout0, out
    order = sorted(range(len(layout0)),
                   key=lambda j: (layout0[j][1] != 'phi', layout0[j][0]))
    layout = [layout0[j] for j in order]
    out2 = {}
    for key, coef in out.items():
        out2[tuple(key[j] for j in order)] = coef
    return layout, out2


def gauge_fix_top0(mono, n_vertices, regroup=True):
    """Gauge-fix a monomial dict for a spinless top (drop slots 0,1,2)."""
    return _reduce_layout(mono, n_vertices, (0, 1, 2), phi_first=regroup)


# ---------------------------------------------------------------------------
# DecayChain → angular table (per chain)
# ---------------------------------------------------------------------------
# For one DecayChain the table contains:
#   * variables   : canonical per-event angle columns [(vertex, kind), …]
#   * basis       : list of monomial keys — one (kind, freq) per variable
#   * waves       : list of {'proj': (λtop, leaf-λ…), 'ls': per-vertex ls,
#                             'cols': {basis_index: complex}}
#   * evaluator   : matrix-multiply amplitude recovery from per-event angles
# The basis is the UNION of the (projection × wave) nonzero monomials, so
# matrix_angle[basis, (proj,wave)] is exactly the sparse coefficients.

def chain_top_j0(chain):
    return to_spin(chain.decays[0].core.J) == 0


def chain_angular_table(chain, verbose=False):
    """Build the canonical angular table for an ampfit DecayChain."""
    import itertools as _it
    from fractions import Fraction as _Fr
    tree = decay_chain_to_tree(chain)
    nv = len(tree_vertices(tree))
    leaves = decay_chain_leaves(chain)           # pre-order Particles

    # external-helicity configurations (projections) from the config spins
    def spins_of(part):
        s = getattr(part, 'spins', None)
        return None if s is None else [to_spin(x) for x in s]

    top = chain.decays[0].core
    top_spins = spins_of(top)
    leaf_spins = [spins_of(o) for o in leaves]
    exts = external_helicities(tree, top_spins=top_spins,
                               leaf_spins=leaf_spins)

    top_j0 = to_spin(top.J) == 0
    drop = (0, 1, 2) if top_j0 else ()
    variables = canonical_variables(nv, top_j0)

    # per-vertex partial waves (parity filtered by the config chain)
    lsets = decay_chain_ls_sets(chain)
    wave_list = list(_it.product(*lsets))

    # gather basis (union of keys) and per-wave coefficients
    key_set = {}
    waves = []
    for wave in wave_list:
        for (lt, lls) in exts:
            _, mono = amplitude_monomials(tree, wave, lt, lls)
            if top_j0:
                _, mono = gauge_fix_top0(mono, nv)
            else:
                layout, mono = _reduce_layout(mono, nv, ())
                del layout
            for k in mono:
                if k not in key_set:
                    key_set[k] = len(key_set)
            waves.append({'proj': (lt, lls), 'ls': wave,
                          'cols': [(key_set[k], c) for k, c in mono.items()]})
    basis = [None] * len(key_set)
    for k, i in key_set.items():
        basis[i] = k

    table = {
        'n_vertices': nv,
        'top_j0': top_j0,
        'top_spin': to_spin(top.J),
        'top_spins': top_spins,
        'variables': variables,
        'basis': basis,
        'waves': waves,
        'n_projections': len(exts),
        'n_waves': len(wave_list),
    }
    if verbose:
        print(f"chain {chain}")
        print("  variables (angle columns):", variables)
        print(f"  projections {len(exts)}  waves {len(wave_list)}  "
              f"basis {len(basis)}")
    return table


def evaluate_table(table, angles):
    """Per-event amplitude matrix via basis multiply.

    *angles*: ordered values for the table['variables'] columns
    (or a dict {vertex: (φ, θ)} in vertex order — then only the columns listed
    in ``variables`` are used).

    Returns complex array shape (n_proj_wave_entries,) giving A for each row
    of ``table['waves']``.
    """
    vars_ = table['variables']
    # resolve per-column values
    if isinstance(angles, dict):
        vals = []
        for (v, kind) in vars_:
            pair = angles[v]
            vals.append(pair[0] if kind == 'phi' else pair[1])
    else:
        vals = list(angles)
    if len(vals) != len(vars_):
        raise ValueError(f"need {len(vars_)} angle values, got {len(vals)}")
    nb = len(table['basis'])
    ka = np.ones(nb, dtype=np.complex128)
    for b, key in enumerate(table['basis']):
        v = 1.0
        for (kind, f), x in zip(key, vals):
            v *= math.cos(f * x) if kind == 'c' else math.sin(f * x)
        ka[b] = v
    out = np.zeros(len(table['waves']), dtype=np.complex128)
    for wi, w in enumerate(table['waves']):
        acc = 0j
        for bi, c in w['cols']:
            acc += ka[bi] * c
        out[wi] = acc
    return out


# ---------------------------------------------------------------------------
# Multi-chain combine → global angular model
# ---------------------------------------------------------------------------
# Chains are merged by their canonical variable layout (event chains that
# share the same (φ…, θ…) column structure, i.e. the same number of vertices
# after the top-J=0 gauge).  Within one layout the global basis is the UNION
# of the per-chain nonzero monomials; every wave entry becomes one column of
# the sparse matrix_angle.

def angular_model(chains, verbose=False):
    """Combine several DecayChain angular tables into global models.

    Args:
        chains: iterable of ampfit DecayChain objects.

    Returns a dict ``layout_key → model`` where each model is::

        {'variables': [(vertex, 'phi'|'theta'), ...],   # event angle columns
         'basis':     [monomial key over the variables, ...],
         'entries':   [{'chain': i, 'proj': (λtop, leafλ…), 'ls': (l,s)…,
                        'cols': [(basis_index, complex), ...]}, ...]}

    Only chains with at least one partial wave contribute entries.
    """
    groups = {}          # layout_key -> model skeleton
    order = []
    for ci, chain in enumerate(chains):
        tbl = chain_angular_table(chain)
        if tbl['n_waves'] == 0 or not tbl['waves']:
            continue
        key = tuple(tbl['variables'])
        if key not in groups:
            groups[key] = {'variables': tbl['variables'], 'entries': [],
                           'n_chains': 0}
            order.append(key)
        groups[key]['n_chains'] += 1
        # per-chain cols are indices into tbl['basis']; translate to keys
        for row in tbl['waves']:
            groups[key]['entries'].append(
                {'chain': ci, 'proj': row['proj'], 'ls': row['ls'],
                 'cols': [(tbl['basis'][bi], c) for bi, c in row['cols']]})

    models = {}
    for key in order:
        g = groups[key]
        # union basis over all tables of this layout (remap per-entry cols)
        bas = {}
        entries = []
        for e in g['entries']:
            newcols = []
            for k, c in e['cols']:
                # e['cols'] keys are the per-chain basis keys (tuples)
                if k not in bas:
                    bas[k] = len(bas)
                newcols.append((bas[k], c))
            entries.append({'chain': e['chain'], 'proj': e['proj'],
                            'ls': e['ls'], 'cols': newcols})
        basis = [None] * len(bas)
        for k, i in bas.items():
            basis[i] = k
        models[key] = {'variables': g['variables'], 'basis': basis,
                       'entries': entries, 'n_chains': g['n_chains']}
        if verbose:
            print(f"layout {list(key)}: chains {g['n_chains']}  "
                  f"entries {len(entries)}  global basis {len(basis)}")
    return models


def evaluate_model(model, angles):
    """Matrix-multiply amplitudes for all entries of one merged model.

    *angles*: per-event values for ``model['variables']`` (list) or
    {vertex: (φ, θ)}.
    Returns a complex array over ``model['entries']``.
    """
    vars_ = model['variables']
    if isinstance(angles, dict):
        vals = [(angles[v][0] if k == 'phi' else angles[v][1])
                for (v, k) in vars_]
    else:
        vals = list(angles)
    nb = len(model['basis'])
    ka = np.ones(nb, dtype=np.complex128)
    for b, key in enumerate(model['basis']):
        v = 1.0
        for (kind, f), x in zip(key, vals):
            v *= math.cos(f * x) if kind == 'c' else math.sin(f * x)
        ka[b] = v
    out = np.zeros(len(model['entries']), dtype=np.complex128)
    for i, e in enumerate(model['entries']):
        acc = 0j
        for bi, c in e['cols']:
            acc += ka[bi] * c
        out[i] = acc
    return out

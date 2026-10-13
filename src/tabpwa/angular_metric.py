"""Analytic angular metric and moment-method extraction for one chain.

The kernels' angular basis elements are

    Z_i(angles) = Prod_c cos(angle_k[i, c] . x_c + angle_b[i, c])

(``kernels_v*_pwa_s0c.cu`` phase 1), where the columns ``x`` are the
canonical helicity angles of the chain — per vertex
one ``phi`` angle in ``[0, 2pi)`` (flat measure ``dphi``) and one
``theta`` in ``[0, pi)`` (flat measure ``dcos(theta) = sin(theta)
dtheta``).  The ``angle_kinds`` input names each column's integration
prescription: ``'phi'`` or ``'theta'`` (cos-theta vs theta is only how
the column is integrated).

Because every element is a product of cosines, every angular integral
reduces per column to ``Int cos(k_1 x + b_1) ... cos(k_m x + b_m) dx``,
which has a CLOSED FORM for any real frequencies (integer or
half-integer): expand each cosine into exponentials (2^m terms) and
use the elementary antiderivatives.  No Monte Carlo is needed on the
model side.

The user-facing interface is :class:`AngularMoments`: built once from
the kernel-config arrays, it caches the analytic tensors and provides

    am = AngularMoments(matrix_angle, n_proj,
                        angle_k, angle_b, angle_kinds)
    kappa = am.solve(data_angles, counts=n_data)   # ck ck'* absolute
    P = am.intensity(kappa, angles)                # the angular intensity

The low-level pieces remain module functions: :func:`column_integral`,
:func:`angular_metric` (pair metric ``Int Z_i Z_j``), :func:`quad_metric`
(the 4-Z tensor on the Z-pair basis), :func:`solve_quad` (a generic
inverse-matrix step).

All metrics are real symmetric.  WITHIN a block everything is exact,
and the phi selection rules make most entries exactly zero
(angular-momentum conservation), not approximately.  Entries between
DIFFERENT blocks (topologies) are NOT computed: the topologies'
canonical coordinates are kinematically correlated through the same
phase space, so those integrals have no analytic factorisation — this
module is the one-chain (single-topology) metric; cross-block terms
would need Monte Carlo over real phase space.
"""

import numpy as np

__all__ = [
    "column_integral", "pair_index", "angular_metric", "quad_metric",
    "coeff_matrix", "intensity_norm", "quad_weighted_norm", "solve_quad",
    "AngularMoments",
]


def _exp_column(kind, a):
    """``Int e^{i a x}`` over the column domain (complex)."""
    if kind == "phi":                     # x in [0, 2pi), measure dx
        if abs(a) < 1e-12:
            return 2.0 * np.pi + 0.0j
        return (np.exp(1j * a * 2.0 * np.pi) - 1.0) / (1j * a)
    if kind == "theta":                   # x in [0, pi), measure sin(x) dx
        def prim(s):                      # Int_0^pi e^{i s x} dx
            if abs(s) < 1e-12:
                return np.pi + 0.0j
            return (np.exp(1j * s * np.pi) - 1.0) / (1j * s)
        # sin x = (e^{ix} - e^{-ix}) / (2i)
        return (prim(a + 1.0) - prim(a - 1.0)) / (2.0j)
    raise ValueError(f"unknown angle kind {kind!r} (phi|theta)")


def column_integral(kind, ks, bs):
    """``Int cos(k_1 x+b_1) ... cos(k_m x+b_m) dx`` for one column.

    Exact for any real frequencies: each cosine expands into 2
    exponentials, the product into ``2^m`` closed-form terms.

    Args:
        kind: 'phi' (domain [0, 2pi), flat) or 'theta' (domain
              [0, pi), measure sin(x) dx — the dcos(theta) measure).
        ks, bs: the per-factor frequencies and phases.

    Returns:
        float — the exact integral.
    """
    m = len(ks)
    total = 0.0 + 0.0j
    for mask in range(1 << m):
        a = 0.0
        phase = 0.0
        for t in range(m):
            s = 1.0 if (mask >> t) & 1 else -1.0
            a += s * float(ks[t])
            phase += s * float(bs[t])
        total += _exp_column(kind, a) * np.exp(1j * phase)
    return float(np.real(total)) / (2.0 ** m)


def pair_index(n):
    """The Hermitian pair index ``[(i, j) for i <= j]`` of n elements."""
    return [(i, j) for i in range(n) for j in range(i, n)]


def _tensor(angle_k, angle_b, angle_kinds, idx):
    """``Int Prod_t Z_{idx[t]}`` — exact product over the elements."""
    angle_k = np.asarray(angle_k, float)
    angle_b = np.asarray(angle_b, float)
    v = 1.0
    for c, kind in enumerate(angle_kinds):
        v *= column_integral(kind,
                             tuple(angle_k[t, c] for t in idx),
                             tuple(angle_b[t, c] for t in idx))
        if v == 0.0:
            return 0.0
    return v


def angular_metric(angle_k, angle_b, angle_kinds):
    """Pair metric ``M[i, j] = Int Z_i Z_j dOmega`` — exact.

    Returns:
        (n_basis, n_basis) float64 — real symmetric.
    """
    n = np.asarray(angle_k).shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v = _tensor(angle_k, angle_b, angle_kinds, (i, j))
            M[i, j] = M[j, i] = v
    return M


def quad_metric(angle_k, angle_b, angle_kinds, pairs=None):
    """Metric on the Z-pair basis: ``G[a, b] = Int g_a g_b dOmega`` with
    ``g_(ij) = Z_i Z_j`` — the 4-Z tensor, exact."""
    n = np.asarray(angle_k).shape[0]
    if pairs is None:
        pairs = pair_index(n)
    npair = len(pairs)
    G = np.zeros((npair, npair))
    for a in range(npair):
        ia, ja = pairs[a]
        for b in range(a, npair):
            ib, jb = pairs[b]
            v = _tensor(angle_k, angle_b, angle_kinds,
                        (ia, ja, ib, jb))
            G[a, b] = G[b, a] = v
    return G


def coeff_matrix(matrix_angle):
    """``C[i, j] = sum_p MA[p, i] MA[p, j]*`` — the (Hermitian PSD)
    coefficients of ``P = sum_ij C_ij Z_i Z_j``."""
    ma = np.asarray(matrix_angle)
    return ma.conj().T @ ma


def intensity_norm(coeff, metric):
    """``Int P dOmega = Re sum_ij C_ij M_ij``."""
    return float(np.real(np.einsum("ij,ij->", coeff, metric)))


def quad_weighted_norm(coeff, b, G, pairs=None):
    """``Int w P`` with ``w = sum_b b_b g_b`` — ``a^T G b`` on the pair
    basis (``a`` = the flattened Hermitian *coeff*: the Re(C_ij) count
    twice)."""
    n = coeff.shape[0]
    if pairs is None:
        pairs = pair_index(n)
    a = np.array([coeff[i, j] if i == j else 2.0 * coeff[i, j]
                  for i, j in pairs])
    return float(np.real(a @ np.asarray(G) @ np.asarray(b, float)))


def solve_quad(G, nu):
    """The inverse-matrix step: pair coefficients from data moments.

    Solves ``nu_a = sum_b G[a, b] c_b`` in the least-squares sense.
    Tiny singular values are truncated (overcomplete/dependent pair
    bases have a null space; the solution lives in its range).
    """
    G = np.asarray(G)
    nu = np.asarray(nu, float)
    c, *_ = np.linalg.lstsq(G, nu, rcond=1e-10)
    return c


class AngularMoments:
    """The one-chain angular moment-method interface.

    Built once from the kernel-config arrays; the analytic tensors
    (the 4-Z metric, the weight-function map, the counts integrals)
    are computed lazily and cached.

    The moment method: the intensity of the chain is the quadratic form

        P(x) = sum_mn kappa_mn w_mn(x),
        w_mn(x) = sum_p fa_{p,m}(x) fa*_{p,n}(x),
        fa_p(x) = matrix_angle-block . Z(x),   kappa_mn = ck_m ck*_n

    so the data moments — the DIRECT SUMS ``nu_a = sum_data W_a(e)``
    over the real weight functions ``W`` built from the Re/Im parts of
    ``w_mn`` — satisfy ``nu = G . kappa . (N_data/Int P)`` with the
    ANALYTIC metric ``G_ab = Int W_a W_b``.  :meth:`solve` inverts that
    and the normalization is automatic: ``sum kappa Int w == sum_data W
    == N_data`` — no separate normalization step exists.

    Args:
        matrix_angle: (n_basis, n_proj * N) p-major angular
            coefficients.
        n_proj: the number of projections P.
        angle_k, angle_b: (n_basis, n_comp) float — per-column
            frequency and phase of each element.
        angle_kinds: (n_comp,) of 'phi' | 'theta' — the per-column
            integration prescription.
    """

    def __init__(self, matrix_angle, n_proj, angle_k,
                 angle_b, angle_kinds):
        self.matrix_angle = np.asarray(matrix_angle, complex)
        self.n_proj = int(n_proj)
        self.angle_k = np.asarray(angle_k, float)
        self.angle_b = np.asarray(angle_b, float)
        self.angle_kinds = list(angle_kinds)
        self.n_basis = self.angle_k.shape[0]
        self.n_wave = self.matrix_angle.shape[1] // self.n_proj
        self.zpairs = pair_index(self.n_basis)
        # the real weight functions: diag (Re), off-diag (Re), off-diag (Im)
        self.wpairs = ([(m, m, "re") for m in range(self.n_wave)]
                       + [(m, n, "re") for m in range(self.n_wave)
                          for n in range(m + 1, self.n_wave)]
                       + [(m, n, "im") for m in range(self.n_wave)
                          for n in range(m + 1, self.n_wave)])
        self._E = None
        self._T = None
        self._G_wave = None
        self._integrals = None
        self._metric = None

    # -- analytic pieces (lazy, cached) -------------------------------

    @property
    def E(self):
        """(n_zpairs, n_wpairs) — the Z-pair expansion of each real
        weight function ``W_a = sum_ij E[a, ij] Z_i Z_j`` (both
        (i, j) and (j, i) orderings: the wave columns differ)."""
        if self._E is None:
            R, I = self.matrix_angle.real, self.matrix_angle.imag
            pc = np.arange(self.n_proj) * self.n_wave
            E = np.zeros((len(self.zpairs), len(self.wpairs)))
            for a, (i, j) in enumerate(self.zpairs):
                for b_idx, (m, n, kind) in enumerate(self.wpairs):
                    # term (i, j): monomial i from wave m, j from wave n
                    r1 = np.sum(R[i, pc + m] * R[j, pc + n]
                                + I[i, pc + m] * I[j, pc + n])
                    i1 = np.sum(I[i, pc + m] * R[j, pc + n]
                                - R[i, pc + m] * I[j, pc + n])
                    if i == j:
                        E[a, b_idx] = r1 if kind == "re" else 0.0
                        continue
                    # term (j, i): monomial j from wave m, i from wave n
                    r2 = np.sum(R[j, pc + m] * R[i, pc + n]
                                + I[j, pc + m] * I[i, pc + n])
                    i2 = np.sum(I[j, pc + m] * R[i, pc + n]
                                - R[j, pc + m] * I[i, pc + n])
                    E[a, b_idx] = (r1 + r2) if kind == "re" else (i1 + i2)
            self._E = E
        return self._E

    @property
    def T(self):
        """The 4-Z tensor on the Z-pair basis (exact)."""
        if self._T is None:
            self._T = quad_metric(self.angle_k,
                                  self.angle_b, self.angle_kinds,
                                  pairs=self.zpairs)
        return self._T

    def _prod_tensor(self, idx):
        """``Int Prod_t Z_{idx[t]}`` — exact (flat measure)."""
        return _tensor(self.angle_k, self.angle_b, self.angle_kinds, idx)

    @property
    def metric(self):
        """The (n_basis, n_basis) pair metric ``M[i, j] = Int Z_i Z_j``
        (exact)."""
        if self._metric is None:
            self._metric = angular_metric(self.angle_k,
                                          self.angle_b, self.angle_kinds)
        return self._metric

    @property
    def wave_metric(self):
        """The analytic metric on the weight functions:
        ``G_wave[a, b] = Int W_a W_b dOmega`` (exact, flat measure),
        computed as ``E^T T E`` through the verified weight-function
        map.  The efficiency correction is NOT here: weight the DATA
        by 1/eps(x) (:meth:`moments`) — the eps cancels in the moment
        relation and this flat metric stays valid."""
        return self.E.T @ self.T @ self.E

    @property
    def integrals(self):
        """``Int W_a dOmega`` — the linear integrals; with the SUM data
        moments ``sum kappa Int W == sum_data W == N_data`` is
        inherent."""
        if self._integrals is None:
            self._integrals = np.array(
                [self._prod_tensor((i, j)) for i, j in self.zpairs]) @ self.E
        return self._integrals

    # -- data-side pieces ---------------------------------------------

    def _z(self, angles):
        """The per-event monomial values Z_i(e): (n_ev, n_basis)."""
        ang = np.asarray(angles, float)[:, :len(self.angle_kinds)]
        return np.prod(np.cos(ang[:, None, :] * self.angle_k[None, :, :]
                              + self.angle_b[None, :, :]), axis=-1)

    def fa(self, angles):
        """The per-wave angular amplitudes: (n_ev, n_proj, n_wave)."""
        return (self._z(angles) @ self.matrix_angle).reshape(
            -1, self.n_proj, self.n_wave)

    def moments(self, data_angles, weights=None):
        """The data moments ``nu_a = sum_events w_e W_a(e)`` — the
        DIRECT sums (the projection sum over P included).  With the
        per-event ``weights`` (sWeights, efficiency weights) the sums
        are weighted and the solve applies the efficiency correction
        through them; the normalization stays ``sum nu == sum kappa
        Int eps w == N_data`` (N_data = the sum of the weights).

        Returns:
            (n_wpairs,) float64.
        """
        fp = self.fa(data_angles)
        fd = fp.reshape(fp.shape[0], self.n_proj * self.n_wave)
        g = []
        for m, n, kind in self.wpairs:
            cm = [p * self.n_wave + m for p in range(self.n_proj)]
            cn = [p * self.n_wave + n for p in range(self.n_proj)]
            v = np.sum(fd[:, cm] * fd[:, cn].conj(), axis=1)
            g.append(v.real if kind == "re" else v.imag)
        g = np.stack(g, axis=1)
        if weights is not None:
            g = g * np.asarray(weights, float)[:, None]
        return np.sum(g, axis=0)

    def intensity(self, kappa, angles):
        """The angular intensity ``P(e)`` for a Hermitian coefficient
        matrix *kappa*: per-event ``Re sum_mn kappa_mn B_mn(e)`` with
        ``B_mn = sum_p fa_{p,m} fa*_{p,n}``."""
        fp = self.fa(angles)
        B = fp.conj().transpose(0, 2, 1) @ fp
        return np.real(np.sum(np.asarray(kappa, complex)[None] * B,
                              axis=(1, 2)))

    def norm(self, kappa):
        """``Int P dOmega = sum kappa Int w`` — for a :meth:`solve`
        result this IS the observed data size (sum_data W)."""
        kappa = np.asarray(kappa, complex)
        vec = []
        for m, n, kind in self.wpairs:
            if m == n:
                vec.append(kappa[m, m].real)
            elif kind == "re":
                vec.append(2.0 * kappa[m, n].real)     # the ordered-pair 2x
            else:
                vec.append(-2.0 * kappa[m, n].imag)
        return float(np.asarray(vec) @ self.integrals)

    # -- the solve -----------------------------------------------------

    def solve(self, data_angles, weights=None, rcond=1e-10):
        """Wave-pair coefficients ``ck ck'*`` from toy data — no
        likelihood fit, no normalization step.

        Solves the linear moment system ``nu = G_wave . c`` in the
        least-squares sense (tiny singular values truncated —
        angularly degenerate waves leave directions the moments
        cannot determine) and unstacks the coefficients to the
        Hermitian ``ck ck'*`` matrix.  With the SUM moments the
        normalization is inherent: ``sum kappa Int w == sum_data W ==
        N_data`` — the overall scale is fixed by the observed counts.

        Args:
            data_angles: (n_data, n_comp) toy angles.
            rcond: the singular-value cutoff of the solve.

        Returns:
            (n_wave, n_wave) complex Hermitian — ``ck ck'*``.
        """
        nu = self.moments(data_angles, weights=weights)
        c, *_ = np.linalg.lstsq(self.wave_metric, nu, rcond=rcond)

        # unstack to the Hermitian kappa.  The intensity sums over
        # ORDERED pairs: the m < n contributions enter as
        # 2 Re(kappa_mn) W_(mn,re) and -2 Im(kappa_mn) W_(mn,im) — the
        # solve's unknown vector carries that factor 2, the Hermitian
        # kappa does not.  The re and im slots map to the SAME
        # kappa[m, n] entry — ADD, never overwrite.
        kappa = np.zeros((self.n_wave, self.n_wave), complex)
        for b_idx, (m, n, kind) in enumerate(self.wpairs):
            v = c[b_idx]
            if m == n:
                kappa[m, m] = v
            elif kind == "re":
                kappa[m, n] += v / 2
                kappa[n, m] += v / 2
            else:
                kappa[m, n] += -1j * v / 2
                kappa[n, m] += 1j * v / 2
        return kappa

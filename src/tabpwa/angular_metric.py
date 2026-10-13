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
    kappa = am.solve(data_angles, weights=1/eff)   # ck ck'* (one representative)
    P = am.intensity(kappa, angles)                # the angular intensity

The linear algebra can also be precomputed once (per model) and the
data side reduced to plain per-event WEIGHTS:

    c = am.weight_vectors(angles).sum(axis=0)   # no solve downstream
    kappa = am.kappa(c)                         # unstack
    # am.solver is the fixed pseudo-inverse; am.basis(angles) the
    # per-event weight-function values.  Save the weight matrix and
    # any subsample's answer is a column sum.

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
        self._solver = None
        self._spectrum = None

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
        if self._G_wave is None:
            self._G_wave = self.E.T @ self.T @ self.E
        return self._G_wave

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

    def basis(self, data_angles):
        """The per-event values of the real weight functions:
        (n_events, n_wpairs) with column ``a`` = ``W_a(e)``.

        This is everything the data side needs — the moment sums are
        its column sums and the reduced per-event weights are
        :meth:`weight_vectors`."""
        fp = self.fa(data_angles)
        fd = fp.reshape(fp.shape[0], self.n_proj * self.n_wave)
        g = []
        for m, n, kind in self.wpairs:
            cm = [p * self.n_wave + m for p in range(self.n_proj)]
            cn = [p * self.n_wave + n for p in range(self.n_proj)]
            v = np.sum(fd[:, cm] * fd[:, cn].conj(), axis=1)
            g.append(v.real if kind == "re" else v.imag)
        return np.stack(g, axis=1)

    @property
    def solver(self):
        """The FIXED linear solver: the minimum-norm pseudo-inverse of
        the analytic ``wave_metric`` (cached — the linear algebra is
        done once per model, never per dataset).

        The coefficient vector is ``c = solver @ sum_events W(e)`` —
        equivalently ``c = sum_events (solver @ W(e))``, i.e. a plain
        sum of the precomputed :meth:`weight_vectors`.  Consumers can
        store the weight vectors and skip this class entirely."""
        if self._solver is None:
            Ur, s2r = self.spectrum
            self._solver = (Ur / s2r) @ Ur.T
        return self._solver

    @property
    def spectrum(self):
        """The truncated eigendecomposition of ``wave_metric``:
        ``(U_r, s2_r)`` with ``r = rank`` — the identifiable directions
        (the singular value cutoff matches the pseudo-inverse, 1e-10
        relative on the singular values).

        Its meaning: ``G = U_r diag(s2_r) U_r^T``, so the solver is the
        rank-r sum ``sum_k u_k u_k^T / s2_k`` and the per-event reduced
        weights collapse to the ``r`` scalar observables
        ``Phi_k(e) = u_k . W(e)`` (:meth:`mode_weights`)."""
        if self._spectrum is None:
            s2, U = np.linalg.eigh(self.wave_metric)
            floor = max(1e-10 ** 2, np.finfo(float).eps * len(s2) * 16)
            keep = s2 > floor * s2.max()
            self._spectrum = (U[:, keep], s2[keep])
        return self._spectrum

    @property
    def rank(self):
        """The number of independent angular observables —
        ``dim span{W_a}``.  The pair basis has ``n(n+1)/2`` functions;
        the rank is smaller whenever the wave functions' products
        collide (integer spins do), and the difference is the
        unidentifiable null space."""
        return len(self.spectrum[1])

    def mode_weights(self, data_angles, weights=None):
        """The per-event SCALAR observable weights
        ``Phi_k(e) = u_k . W(e)`` — (n_events, rank) float64.

        These are the reduced, independent weights: their column sums
        are the observables ``gamma_k`` (:meth:`modes`), and the
        coefficient vector is the fixed recombination
        ``c = sum_k (gamma_k / s2_k) u_k``
        (:meth:`coeffs_from_modes`).  No linear algebra
        downstream — a plain sum of ``rank`` numbers per event.

        Args:
            data_angles: (n_data, n_comp) canonical angle columns.
            weights: optional (n_data,) per-event efficiency weights.

        Returns:
            (n_data, rank) float64.
        """
        Ur = self.spectrum[0]
        ow = self.basis(data_angles) @ Ur
        if weights is not None:
            ow = ow * np.asarray(weights, float)[:, None]
        return ow

    def modes(self, data_angles, weights=None):
        """The independent angular observables
        ``gamma_k = sum_events w_e Phi_k(e)`` — (rank,) float64.

        These are ALL a data sample can say about the angular
        coefficients: the (n(n+1)/2 - rank) null directions are
        invisible to every moment (any test function), so no basis
        beyond the model span can add information.

        Uncertainties: see :meth:`mode_errors` (each event is a
        unit Poisson count, so each weighted sum carries the
        sqrt-sum-of-squares error)."""
        return self.mode_weights(data_angles, weights=weights).sum(
            axis=0)

    def mode_errors(self, data_angles, weights=None):
        """The Poisson(1) standard errors of :meth:`modes`:
        ``sigma_k = sqrt(sum_events (w_e Phi_k(e))^2)``.

        With unit-count (Poisson(1)) events, each weighted sum
        ``sum_e w_e Phi_k(e)`` has variance ``sum_e (w_e Phi_k(e))^2``.
        This diagonal IS the uncertainty of ``gamma_k`` — exact, no
        assumption about correlations.  The correlations are a
        separate object: :meth:`mode_covariance`."""
        ow = self.mode_weights(data_angles, weights=weights)
        return np.sqrt((ow ** 2).sum(axis=0))

    def mode_covariance(self, data_angles, weights=None):
        """The observable-observable CORRELATIONS (not the
        uncertainties): ``Cov(gamma_k, gamma_l) = sum_e (w Phi_k)(w
        Phi_l)`` — (rank, rank) float64.

        Its diagonal squared equals :meth:`mode_errors`; the
        off-diagonal is the correlation information, needed only when
        COMBINING observables (e.g. to propagate exactly into the
        coefficients — though :meth:`weight_errors` gets that directly
        from the per-event weights with no covariance bookkeeping)."""
        ow = self.mode_weights(data_angles, weights=weights)
        return ow.T @ ow

    def coeff_errors(self, cov_or_sigma):
        """The coefficient-vector standard errors from either the
        observable diagonal errors (1-D) or the full observable
        covariance (2-D, exact):

        ``sigma_c = sqrt(diag(A Cov(gamma) A^T))`` with ``A = U/s2``
        — for a 1-D argument the off-diagonal is taken as zero (the
        diagonal-only shortcut; the results then differ)."""
        Ur, s2r = self.spectrum
        A = Ur / s2r
        v = np.asarray(cov_or_sigma, float)
        if v.ndim == 1:
            return np.sqrt(((A ** 2) * v ** 2).sum(axis=1))
        return np.sqrt(((A @ v) * A).sum(axis=1))

    def weight_errors(self, data_angles, weights=None):
        """The EXACT coefficient-vector standard errors:
        ``sigma_c[b] = sqrt(sum_events w(e)[b]^2)``.

        ``c = sum_e w(e)`` is a plain sum of per-event contributions,
        so its variance is the per-event square sum — exact, with no
        correlation bookkeeping.  Identical to
        ``coeff_errors(mode_covariance(...))``."""
        wv = self.weight_vectors(data_angles, weights=weights)
        return np.sqrt((wv ** 2).sum(axis=0))

    def kappa_weights(self, data_angles, weights=None):
        """The per-event contributions to :meth:`kappa` (before the
        imaginary unit factors are folded in): ``(n_events, n_wave,
        n_wave)`` complex = ``kappa(w(e))`` per event."""
        wv = self.weight_vectors(data_angles, weights=weights)
        m = len(self.wpairs)
        K = np.stack([self.kappa(np.eye(m)[b]) for b in range(m)])
        return np.tensordot(wv, K, axes=([1], [0]))

    def kappa_errors(self, data_angles, weights=None):
        """The EXACT per-entry standard errors of :meth:`kappa`:
        ``sqrt(sum_events |kappa(w(e))[m, n]|^2)``.

        ``kappa`` is linear in ``c`` and ``c`` is a sum of per-event
        weights, so every entry is a per-event sum — the diagonal
        square sum is exact."""
        kw = self.kappa_weights(data_angles, weights=weights)
        return np.sqrt((np.abs(kw) ** 2).sum(axis=0))

    def coeffs_from_modes(self, gamma):
        """The coefficient vector ``c = sum_k (gamma_k / s2_k) u_k``
        from the observables (the fixed recombination; tiny)."""
        Ur, s2r = self.spectrum
        return Ur @ (np.asarray(gamma, float) / s2r)

    def mode_forms(self):
        """Export the independent observable FUNCTIONS as compact
        quadratic forms — a self-contained "simple basis".

        Returns:
            (rank, n_basis, n_basis) real symmetric — the coefficient
            matrices ``Q_k`` with

                ``Phi_k(x) = sum_ij Q_k[i,j] Z_i(x) Z_j(x)``,
                ``Z_i(x) = prod_c cos(angle_k[i,c] x_c + angle_b[i,c])``

            i.e. the k-th observable's per-event weight is a plain
            quadratic form in the cos-monomials.  The symmetric
            (half-off-diagonal) convention absorbs the ordered-pair
            factor, so the evaluation is a plain double sum — no
            module machinery needed downstream.
        """
        Ur = self.spectrum[0]
        g = Ur.T @ self.E.T                # (rank, n_zpairs) coefficients
        n = self.n_basis
        Q = np.zeros((g.shape[0], n, n))
        for k in range(g.shape[0]):
            for a, (i, j) in enumerate(self.zpairs):
                if i == j:
                    Q[k, i, i] = g[k, a]
                else:
                    Q[k, i, j] = g[k, a] / 2
                    Q[k, j, i] = g[k, a] / 2
        return Q

    def mode_kappa(self):
        """The observable→``ck ck'*`` instrument matrices: each
        observable is an explicit combination of the wave-pair
        products,

            ``gamma_k = (s2_k / Int P) . Tr(kappa . K_k)``
            ``        = (s2_k / Int P) . [ sum_m K_k[m,m] |ck_m|^2
            ``            + 2 sum_{m<n} Re(kappa_mn conj(K_k[m,n])) ]``

        Returns:
            (rank, n_wave, n_wave) complex Hermitian ``K_k`` — the few
            entries the k-th observable measures.  Together with the
            per-event basis (:meth:`mode_forms`) this is the
            explicit dictionary between the measurements and the
            physical ``ck_m ck_n*`` combinations.
        """
        Ur = self.spectrum[0]
        K = np.zeros((self.rank, self.n_wave, self.n_wave), complex)
        for k in range(self.rank):
            u = Ur[:, k]
            for b_idx, (i, j, kind) in enumerate(self.wpairs):
                if i == j:
                    K[k, i, i] += u[b_idx]
                elif kind == "re":
                    K[k, i, j] += u[b_idx]
                    K[k, j, i] += u[b_idx]
                else:                     # the c vector holds -2 Im(kappa)
                    K[k, i, j] -= 1j * u[b_idx]
                    K[k, j, i] += 1j * u[b_idx]
        return K

    def wave_groups(self, n_probe=16, seed=0):
        """Group the waves by PROPORTIONAL angular functions.

        Members of a group are degenerate in the angular data: if
        ``f_j = alpha_j . f_rep`` then the amplitudes only enter through
        the group's coherent sum ``G = sum_j alpha_j c_j`` — the
        individual ``c_j`` inside a group cannot be separated by any
        angular test function.  A rank-raising extension of the basis
        (e.g. mass-dependent test functions, whose Breit-Wigner shapes
        differ) splits the groups.

        Returns:
            ``(groups, ratios)`` — the list of member-index lists and,
            per group, the coefficients ``alpha_j`` relative to the
            first member (``alpha_rep = 1``).
        """
        rng = np.random.default_rng(seed)
        ang = rng.uniform(0.0, 1.0, size=(n_probe, len(self.angle_kinds)))
        fa = self.fa(ang)                       # (n_probe, P, N)
        V = [fa[:, :, w].reshape(-1).copy() for w in range(self.n_wave)]
        groups, ratios, assign = [], [], [-1] * self.n_wave
        for w in range(self.n_wave):
            if assign[w] >= 0:
                continue
            members, alphas = [w], [1.0 + 0.0j]
            assign[w] = len(groups)
            for w2 in range(w + 1, self.n_wave):
                if assign[w2] >= 0:
                    continue
                v1, v2 = V[w], V[w2]
                nz = np.abs(v1) > 1e-12
                if not nz.any():
                    continue
                r = v2[nz] / v1[nz]
                M = np.stack([v1, v2])
                scale = max(1.0, np.abs(M).max())
                if (np.abs(r - r[0]).max() < 1e-9 * scale
                        and np.linalg.matrix_rank(M, tol=1e-9 * scale) <= 1):
                    members.append(w2)
                    alphas.append(complex(r[0]))
                    assign[w2] = len(groups)
            groups.append(members)
            ratios.append(alphas)
        return groups, ratios

    def physical_basis(self):
        """A basis of the observable space made of PHYSICAL items: each
        element is an explicit sum of ``ck_m ck_n*`` products and is
        individually identified (residual 0) — ``rank`` elements, no
        leftover.

        The construction: every physical item (``|ck_m|^2``, ``Re``,
        ``Im``) is projected onto the identifiable subspace; the exact
        singles are taken first, then the projections of the rest are
        completed by a rank-revealing QR, giving a basis of the whole
        observable space.

        Returns:
            ``(labels, coeffs, residual)`` — for each basis element
            the label, its coefficients over the physical items (the
            slot coordinates: |ck|^2 / Re / Im in ``wpairs`` order) and
            the residual (0 for every kept element — anything with a
            non-zero out-of-span part is part of the discarded halos,
            not of the basis).
        """
        Ur = self.spectrum[0]
        P = Ur @ Ur.T

        # the physical items in SLOT (wpairs) order — index-aligned with
        # the returned coefficient vectors
        wpos = {w: i for i, w in enumerate(self.wpairs)}
        items = []
        kind_rank = []
        for (m, n, kind) in self.wpairs:
            if kind == "re" and m == n:
                label, rank_k = f"|ck_{m}|^2", 0
            elif kind == "re":
                label, rank_k = f"Re(ck_{m} ck_{n}*)", 1
            else:
                label, rank_k = f"Im(ck_{m} ck_{n}*)", 2
            items.append((label, wpos[(m, n, kind)]))
            kind_rank.append(rank_k)
        # order: KIND first — the diagonal |ck_m|^2 terms are kept in
        # preference (they are the physical wave magnitudes), then Re,
        # then Im; within each kind the identified items first, then
        # slot order.  The QR below reads the columns in this order, so
        # the kept basis prefers the diagonals wherever they can be
        # resolved.
        def _resid(a):
            e = np.eye(len(self.wpairs))[items[a][1]]
            return np.linalg.norm(e - P @ e)
        order = sorted(range(len(items)),
                       key=lambda a: (kind_rank[a], round(_resid(a), 6), a))

        # rank-revealing QR over the projected items (columns in the
        # priority order): the QR diagonal is the independence measure
        # — |R_kk| ~ 0 marks a dependent (or unmeasurable) item — so
        # the kept basis needs no hand-rolled Gram-Schmidt
        m = len(self.wpairs)
        A = np.stack([P @ np.eye(m)[items[a][1]] for a in order], axis=1)
        _Q, R = np.linalg.qr(A)
        dR = np.abs(np.diag(R))
        tol = 1e-9 * max(float(dR.max()), 1e-300)
        keep = [k for k in range(len(order)) if dR[k] > tol][:self.rank]
        labels = [items[order[k]][0] for k in keep]
        rows = [A[:, k] for k in keep]
        return labels, np.array(rows), np.zeros(len(rows))

    def physical_basis_observables(self, data_angles, weights=None):
        """Measure the :meth:`physical_basis` elements from data.

        Returns:
            ``(labels, values, errors, residual)`` — the same shape as
            :meth:`physical_functionals`; residual is 0 for the basis
            (every element is identified).
        """
        labels, rows, _ = self.physical_basis()
        Ur = self.spectrum[0]
        alpha = (rows @ Ur)                 # (n_basis, rank) recombos
        ow = self.mode_weights(data_angles, weights=weights)
        values = alpha @ ow.sum(axis=0)
        errors = np.sqrt(((ow @ alpha.T) ** 2).sum(axis=0))
        return labels, values, errors, np.zeros(len(labels))

    def physical_functionals(self, items):
        """Represent physical ``ck_m ck_n*`` products as combinations of
        the observables — the general form of "a·gamma_1 + b·gamma_3
        = Re(ck_1 ck_2*)".

        Args:
            items: iterable of ``(kind, m, n)`` with kind in
                ``'abs'`` (``|ck_m|^2``, requires m == n), ``'re'``
                (``Re(ck_m ck_n*)``) or ``'im'`` (``Im(ck_m ck_n*)``).

        Returns:
            ``(labels, alpha, residual)`` — for each item the label
            string, the observable combination ``alpha`` (length
            ``rank``) with ``value = alpha @ gamma``, and ``residual``
            the *relative* unidentifiable part (0 = exactly fixed by
            the angular data; >0 = that fraction of the item lives in
            the null space and no data of this kind can measure it).
        """
        Ur, s2r = self.spectrum
        wpos = {w: i for i, w in enumerate(self.wpairs)}
        labels, alphas, res = [], [], []
        for kind, m, n in items:
            v = np.zeros(len(self.wpairs))
            if kind == "abs":
                assert m == n
                v[wpos[(m, n, "re")]] = 1.0
                label = f"|ck_{m}|^2"
            elif kind == "re":
                v[wpos[(m, n, "re")]] = 0.5
                label = f"Re(ck_{m} ck_{n}*)"
            elif kind == "im":
                v[wpos[(m, n, "im")]] = -0.5
                label = f"Im(ck_{m} ck_{n}*)"
            else:
                raise ValueError(f"unknown kind {kind!r} (abs|re|im)")
            alpha = (Ur.T @ v) / s2r
            resid = np.linalg.norm(v - Ur @ (Ur.T @ v)) / np.linalg.norm(v)
            labels.append(label)
            alphas.append(alpha)
            res.append(float(resid))
        return labels, np.array(alphas), np.array(res)

    def physical_observables(self, items, data_angles, weights=None):
        """Measure the requested physical ``ck_m ck_n*`` products
        directly from data — the observable sums recombined into
        readable physics, with the exact Poisson(1) errors.

        Returns:
            ``(labels, values, errors, residual)`` — see
            :meth:`physical_functionals` for the items and *residual*.
        """
        labels, alpha, residual = self.physical_functionals(items)
        ow = self.mode_weights(data_angles, weights=weights)
        values = alpha @ ow.sum(axis=0)
        errors = np.sqrt(((ow @ alpha.T) ** 2).sum(axis=0))
        return labels, values, errors, residual

    def _ck_reference(self, ck):
        """The global-phase reference slot: the largest-|ck| wave
        (slot 0 when *ck* is None).  Robust to a zero wave, unlike a
        fixed ``ck_0`` reference."""
        if ck is None:
            return 0
        return int(np.argmax(np.abs(np.asarray(ck, complex))))

    def ck_labels(self, ck=None):
        """The real ck degrees of freedom, global phase fixed by the
        largest-|ck| wave: ``['Re ck_ref', 'Re ck_m', 'Im ck_m', ...]``
        over the remaining slots in order.  ``ck=None`` -> reference 0."""
        ref = self._ck_reference(ck)
        order = [ref] + [m for m in range(self.n_wave) if m != ref]
        labels = [f"Re ck_{order[0]}"]
        for m in order[1:]:
            labels += [f"Re ck_{m}", f"Im ck_{m}"]
        return labels

    def _coeff_from_ck(self, ck):
        """The pair-coefficient vector of the density for amplitudes
        *ck* (the inverse of :meth:`kappa`'s convention: the ordered
        pair factor 2 is carried here)."""
        ck = np.asarray(ck, complex)
        wpos = {w: i for i, w in enumerate(self.wpairs)}
        c = np.zeros(len(self.wpairs))
        for m in range(self.n_wave):
            c[wpos[(m, m, "re")]] = float(abs(ck[m]) ** 2)
        for m in range(self.n_wave):
            for n in range(m + 1, self.n_wave):
                g = ck[m] * np.conjugate(ck[n])
                c[wpos[(m, n, "re")]] = 2.0 * g.real
                c[wpos[(m, n, "im")]] = -2.0 * g.imag
        return c

    def ck_jacobian(self, ck=None):
        """``d(nu)/d(ck dof)`` — (n_wpairs, 2N-1): the EXACT derivative of
        the pair moments ``nu(ck) = G . c(ck) / (c(ck) . integrals)``.

        ``nu`` is quadratic in the ck (the pair coefficients are
        ``ck_m ck_n*``) with a simple closed form, so the Jacobian is
        written out directly (no finite differences): with
        ``kappa = ck ck'*``,
        ``d kappa_mn / d(Re ck_t) = delta_mt ck_n* + ck_m delta_nt`` and
        ``d kappa_mn / d(Im ck_t) = 1j (delta_mt ck_n* - ck_m delta_nt)``;
        map each to the coefficient vector and apply the quotient rule.
        The global phase is fixed by making the largest-|ck| wave real
        (robust to a zero wave), leaving ``2N-1`` real dofs.

        No internal ``gamma`` (observable eigenbasis) is involved — that
        basis is a full-rank recombination of the moments, and a
        full-rank left factor does not change which ck directions are
        independent (the QR column dependencies are identical).
        """
        if ck is None:
            rng = np.random.default_rng(0)
            ck = (rng.standard_normal(self.n_wave)
                  + 1j * rng.standard_normal(self.n_wave))
        ck = np.asarray(ck, complex)
        ref = self._ck_reference(ck)
        ck = ck * np.exp(-1j * np.angle(ck[ref]))     # ck[ref] -> real
        rest = [m for m in range(self.n_wave) if m != ref]

        G = self.wave_metric
        integ = self.integrals
        c = self._coeff_from_ck(ck)
        cint = c @ integ
        Gc = G @ c
        wpos = {w: i for i, w in enumerate(self.wpairs)}

        def dcoeff(dkappa):
            """The density-derivative coefficient vector from dkappa
            (the inverse of :meth:`kappa`'s element mapping)."""
            dc = np.zeros(len(self.wpairs))
            for (m, n, kind) in self.wpairs:
                if m == n:
                    dc[wpos[(m, n, "re")]] = dkappa[m, m].real
                elif kind == "re":
                    dc[wpos[(m, n, "re")]] = 2.0 * dkappa[m, n].real
                else:
                    dc[wpos[(m, n, "im")]] = -2.0 * dkappa[m, n].imag
            return dc

        # dofs in the SAME order as :meth:`ck_labels`: Re ref, then
        # (Re m, Im m) for each remaining wave
        dofs = [(ref, False)]
        for m in rest:
            dofs += [(m, False), (m, True)]

        J = np.zeros((len(self.wpairs), len(dofs)))
        for j, (t, is_imag) in enumerate(dofs):
            u = np.zeros(self.n_wave)
            u[t] = 1.0
            A = np.outer(u, ck.conj())                 # delta_mt ck_n*
            B = A.conj().T                             # ck_m delta_nt
            dkap = 1j * (A - B) if is_imag else (A + B)
            dc = dcoeff(dkap)
            J[:, j] = (G @ dc * cint - Gc * (dc @ integ)) / cint ** 2
        return J

    def ck_basis(self, ck=None):
        """Rank-revealing QR in the ck space: the INDEPENDENT ck
        combinations versus the unmeasurable ones.

        With the pivoted QR of :meth:`ck_jacobian` the R diagonal gives
        the independence directly: ``|R_kk| > 0`` keeps a direction
        (its dof is fixed by the observables), ``|R_kk| ~ 0`` marks a
        near-null one (that ck combination cannot be measured from the angular data).

        Returns:
            ``(keep, drop, dR)`` — the kept/dropped dof labels (in the
            pivot order) and the raw R diagonal.
        """
        from scipy.linalg import qr as _qr
        if ck is None:
            rng = np.random.default_rng(0)
            ck = (rng.standard_normal(self.n_wave)
                  + 1j * rng.standard_normal(self.n_wave))
        J = self.ck_jacobian(ck)
        _Q, R, piv = _qr(J, pivoting=True, mode="economic")
        dR = np.abs(np.diag(R))
        tol = 1e-6 * max(float(dR.max()), 1e-300)
        labels = self.ck_labels(ck)
        keep = [labels[p] for k, p in enumerate(piv[:len(dR)])
                if dR[k] > tol]
        drop = [labels[p] for k, p in enumerate(piv[:len(dR)])
                if dR[k] <= tol]
        drop += [labels[p] for p in piv[len(dR):]]
        return keep, drop, dR

    @property
    def mode_recombination(self):
        """``(n_wpairs, rank)`` — the FIXED map from the observable sums
        to the coefficient vector: ``c = mode_recombination @
        gamma``.  Together with :meth:`mode_forms` this makes the
        whole reduction exportable (no live :class:`AngularMoments`
        needed downstream)."""
        Ur, s2r = self.spectrum
        return Ur / s2r

    def weight_vectors(self, data_angles, weights=None):
        """The per-event REDUCED weights ``w(e) = solver @ W(e)``:
        (n_events, n_wpairs) float64.

        ``w(e).sum(axis=0)`` IS the coefficient vector ``c`` of
        :meth:`solve` (before the unstack) — no linear algebra needed
        downstream: save the weight matrix (or the per-event rows) and
        any subsample's coefficient vector is a sum.  The optional
        per-event ``weights`` (e.g. ``1/eps(x)``) multiply in as
        usual.

        Args:
            data_angles: (n_data, n_comp) canonical angle columns.
            weights: optional (n_data,) per-event efficiency weights.

        Returns:
            (n_data, n_wpairs) float64 — sum over events for ``c``.
        """
        wv = self.basis(data_angles) @ self.solver.T
        if weights is not None:
            wv = wv * np.asarray(weights, float)[:, None]
        return wv

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
        g = self.basis(data_angles)
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

    def kappa(self, c):
        """Unstack a coefficient vector *c* (the pair-basis unknowns,
        e.g. ``weight_vectors(...).sum(axis=0)``) into the Hermitian
        ``ck ck'*`` matrix.

        The intensity sums over ORDERED pairs: the m < n contributions
        enter as ``2 Re(kappa_mn) W_(mn,re)`` and ``-2 Im(kappa_mn)
        W_(mn,im)`` — the coefficient vector carries that factor 2,
        the Hermitian kappa does not.  The re and im slots map to the
        SAME ``kappa[m, n]`` entry — ADD, never overwrite.
        """
        c = np.asarray(c)
        out = np.zeros((self.n_wave, self.n_wave), complex)
        for b_idx, (m, n, kind) in enumerate(self.wpairs):
            v = c[b_idx]
            if m == n:
                out[m, m] = v
            elif kind == "re":
                out[m, n] += v / 2
                out[n, m] += v / 2
            else:
                out[m, n] += -1j * v / 2
                out[n, m] += 1j * v / 2
        return out

    def solve(self, data_angles, weights=None):
        """Wave-pair coefficients ``ck ck'*`` from toy data — no
        likelihood fit, no normalization step.

        Solves the linear moment system ``nu = G_wave . c`` with the
        cached minimum-norm :attr:`solver` (tiny singular values
        truncated — angularly degenerate waves leave directions the
        moments cannot determine; the returned kappa is one
        representative, its INTENSITY is unique) and unstacks the
        coefficients to the Hermitian ``ck ck'*`` matrix.  With the
        SUM moments the normalization is inherent: ``sum kappa Int w
        == sum_data W == N_data`` — the overall scale is fixed by the
        observed counts.

        For data distributed as ``eps(x) . P(x)`` pass
        ``weights = 1/eps(x)`` per event — the efficiency cancels.

        Args:
            data_angles: (n_data, n_comp) toy angles.
            weights: optional (n_data,) per-event weights.

        Returns:
            (n_wave, n_wave) complex Hermitian — ``ck ck'*``.
        """
        c = self.weight_vectors(data_angles, weights=weights).sum(axis=0)
        return self.kappa(c)

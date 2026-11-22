"""integrated_pwa_gen — Gram-matrix (pre-integrated) phase-space norm for the
FLAT ``pwa_gen`` config.

For the flat projection-sum intensity

    P(e) = Σ_p |A_p(e)|² ,   A_p(e) = Σ_k ck_k · c_{e,p,k}
    c_{e,p,k} = common[e,k] · fa[e,k,p]
              = ( Π_d fl · Π_r num / D_{e,k} ) · fa_{e,p,k}
    D_{e,k}   = Π_l Σ_m gamma[k,l,m]·basis_m(m_e)·coeff[m]

``c`` is LINEAR in ``ck`` and, with ``coeff`` PINNED, is a per-event number — so
the phase-space integral factorises into a wave Gram matrix

    Σ_e w_e P(e) = Σ_{k,k'} conj(ck_k)·D[k,k']·ck_{k'}
    D[k,k']      = Σ_e w_e Σ_p conj(c_{e,p,k})·c_{e,p,k'}      (N×N, N = n_wave)

``D`` is Hermitian positive semi-definite and independent of the number of
phase-space events, so an iteration costs O(N²) instead of O(N_events).  That is
the same trade ``IntegratedPWA`` makes for the ``pwa`` model; the restriction is
different and stronger in one way: there it is ``m0/g0`` pinned, here it is the
whole ``coeff`` vector (every basis-transform input), because ``coeff`` is
inside ``D_{e,k}``.

The amplitude is NOT re-derived here: :meth:`NumpyFlatBackend.wave_amplitudes`
is the reference implementation's own forward chain (``numpy_flat``), so the
Gram and the reference cannot drift.  That is why ``numpy_flat`` grew
``_parse`` / ``_coeff_block`` / ``_wave_amplitudes`` as one shared path.

This class is the ORACLE and the FALLBACK, not the fast path.
``integrated_pwa_gen_backend`` prefers the base kernel's own ``compute_gram``
when it has one — ``cuda_flat_pwa_gen_ampcache`` caches the finished per-wave
amplitude ``Afix = Amp·conj(D)/|D|²``, so its Gram is a pure device reduction
of a cache the data NLL needs anyway (no interpolation, no basis row, no
parameter), and it streams the phase space one batch at a time.  What remains
here is the fp64 reference that GPU Gram is checked against, plus the builder
for every base without one.

Conventions follow :mod:`tabpwa.integrated_pwa` exactly:

    norm(ck, D)  = Re( conj(ck) @ (D @ ck) )
    dnorm/dck    = conj(D @ ck)

which is what ``integrated_pwa_backend`` publishes as ``grads["ck"]``.
"""

import numpy as np

from tabpwa.backends.numpy_flat_backend import NumpyFlatBackend

#: events per Gram slice: peak memory is one slice of ``(ne, n_wave, n_proj)``
#: complex arrays plus the N×N accumulator, not the whole sample
DEFAULT_CHUNK = 5000


class IntegratedPWAFlat:
    """Wave Gram matrix for the flat ``pwa_gen`` config at a PINNED ``coeff``."""

    def __init__(self, config, chunk=DEFAULT_CHUNK):
        self.config = config
        self.chunk = int(chunk) if chunk else DEFAULT_CHUNK
        self.n_wave = int(np.asarray(config["gamma_matrix"]).shape[0])
        self.n_basis = int(np.asarray(config["basis_table"]).shape[0])
        # derived from the shape the builder emits, exactly as the two kernels
        # derive it — never a stored count (the flat config's rule)
        self.n_proj = max(
            1, int(np.asarray(config["matrix_angle"]).shape[1]) // self.n_wave)
        # the reference forward chain, used as a library (no device state)
        self._ref = NumpyFlatBackend(config)

    # ------------------------------------------------------------------
    def gram(self, data, coeff, weight=None):
        """``D`` (``n_wave × n_wave``) over ``data`` at the given ``coeff``.

        Chunked: one slice of the sample is turned into amplitudes at a time, so
        the peak is ``chunk·n_wave·n_proj`` complex numbers plus ``D``.
        """
        coeff = np.asarray(coeff, np.float64)
        if coeff.shape != (self.n_basis,):
            raise ValueError(
                f"integrated_pwa_gen: coeff has shape {coeff.shape}, expected "
                f"(n_basis,) = ({self.n_basis},)")
        ne = len(np.asarray(data["weight"])) if "weight" in data \
            else len(np.asarray(data["mass"]))
        w_all = np.asarray(data["weight"], np.float64) if "weight" in data \
            else np.ones(ne, np.float64)
        if weight is not None:
            w_all = w_all * np.asarray(weight, np.float64)

        D = np.zeros((self.n_wave, self.n_wave), np.complex128)
        for start in range(0, ne, self.chunk):
            stop = min(start + self.chunk, ne)
            sl = {k: (v[start:stop] if isinstance(v, np.ndarray)
                      and v.ndim >= 1 and v.shape[0] == ne else v)
                  for k, v in data.items()}
            common, fa, _factors = self._ref.wave_amplitudes(sl, coeff)
            if fa.shape[2] != self.n_proj:
                raise ValueError(
                    f"integrated_pwa_gen: fa has {fa.shape[2]} projections, "
                    f"but matrix_angle encodes n_proj = {self.n_proj}")
            # c[e,k,p] = common[e,k]·fa[e,k,p];  the SAME pair numpy_flat's
            # compute multiplies by ck, so the Gram cannot drift from it.
            c = common[:, :, None] * fa
            w = w_all[start:stop]
            D += np.einsum("e,ekp,ejp->kj", w, c.conj(), c, optimize=True)
        return (D + D.conj().T) / 2.0

    # ------------------------------------------------------------------
    def norm(self, ck, D):
        """``Σ_e w_e P(e)`` from the Gram — real, ≥ 0 by construction."""
        ck = np.asarray(ck, np.complex128)
        if ck.size != self.n_wave:
            raise ValueError(
                f"integrated_pwa_gen: ck has {ck.size} entries, expected "
                f"n_wave = {self.n_wave}")
        return float(np.real(ck.conj() @ (D @ ck)))

    def dnorm(self, ck, D):
        """``∂norm/∂ck`` — the convention ``integrated_pwa`` publishes."""
        ck = np.asarray(ck, np.complex128)
        return np.conj(D @ ck)

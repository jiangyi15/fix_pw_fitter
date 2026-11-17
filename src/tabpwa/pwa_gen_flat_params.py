"""Parameter transform for the flat (non-deduped) ``pwa_gen`` kernel config.

The kernel input of a flat-config backend is exactly two arrays:

    ck     complex (n_wave,)    the shared coupling per wave
    coeff  real    (n_basis,)   one weight per basis row of ``basis_table``

Unlike :class:`tabpwa.kernel_params.PWAGenKernelParams` (which fills a padded
``(n_fact, n_slots)`` block from ``den_row`` and shares it across the factor
copies of a BW), the flat config **appends a factor's basis rows where that
factor needs them**, so every basis column has exactly one owning
``(wave, res)``.  ``coeff`` is therefore a flat vector with no slot padding and
no copy-sum: entry ``m`` is the weight of ``basis_table[m]``, and
``gamma_matrix[k, r, m]`` says whether factor ``(k, r)`` reads it.

Sign convention (inherited from the s0c encoding): the basis row for ``m²`` is
stored as ``F = −m²``, so its weight enters negated.  This transform applies
that once, here, so the kernel never sees a sign flip.

Usage::

    from tabpwa.pwa_gen_config import build_from_groups
    from tabpwa.pwa_gen_flat_params import FlatKernelParams

    config = build_from_groups(model)
    tr = FlatKernelParams(model, config)
    params = tr.forward(resolved)        # {"ck": ..., "coeff": ...}
    grads = tr.backward(kernel_grads, resolved)   # back to resolved names
"""
import numpy as np

from .kernel_params import BuildKernelParams


class FlatKernelParams(BuildKernelParams):
    """Resolved dict ↔ ``{"ck", "coeff"}`` for the flat kernel config.

    Args:
        model: the amplitude model (its ``ck_map`` and physical names come
            from the base class).
        config: the dict from
            :func:`tabpwa.pwa_gen_config.build_from_groups`.

    The coefficient names come from ``config["den_names"]``; each is a model
    parameter produced by the resonance's basis transform, so the gradient
    chains to the physical parameters there.
    """

    def __init__(self, model, config):
        super().__init__(model)
        self.config = config
        self._n_basis = int(np.asarray(config["basis_table"]).shape[0])
        names = list(config["den_names"])
        if len(names) != self._n_basis:
            raise ValueError(
                f"FlatKernelParams: den_names has {len(names)} entries but "
                f"basis_table has {self._n_basis} rows")
        self._den_names = names
        # ``basis_m_index`` is the event mass column each row reads; the
        # kernel uses it directly, so nothing here needs it — kept for
        # introspection only.
        self._does_not_use_m0_g0 = True

    # -- parameter surface ---------------------------------------------
    def param_names(self):
        """ck slots plus the basis-weight names (deduplicated, order stable)."""
        base = super().param_names()
        # the flat config's weights replace m0/g0 for any resonance that has
        # a basis transform; keep whatever the base class already exposes and
        # add the ones the coefficient vector actually consumes
        out = [n for n in base if n not in set(self.model.m0_phys_name)
               and n not in set(self.model.g0_phys_name)]
        out.extend(self._den_names)
        return list(dict.fromkeys(out))

    # -- forward --------------------------------------------------------
    def forward(self, resolved):
        """Resolved dict → ``{"ck": (n_wave,), "coeff": (n_basis,)}``.

        A name may appear several times in ``den_names`` — the builder appends
        a factor's rows per ``(wave, res)``, so one physical weight can own
        more than one basis column.  Those copies are the SAME parameter, so
        they are filled from one resolved value (fan-out); the backward pass
        folds them back (fan-in).
        """
        out = {"ck": self._pc.build_ck(resolved)}
        coeff = np.zeros(self._n_basis, np.float64)
        for m, name in enumerate(self._den_names):
            coeff[m] = float(np.real(resolved[name]))
        # No sign correction here.  The model's `den_basis` rows are used
        # verbatim and the shared m² row carries the literal ``"one"`` weight
        # (fixed to 1), so the kernel sees exactly the expression the model
        # wrote.  `_m2_rows()` used to SNIFF for a row equal to -m² and negate
        # its weight to compensate for a builder-side override; both are gone.
        out["coeff"] = coeff
        # this parameterisation is ck + coeff only
        out.pop("m0", None)
        out.pop("g0", None)
        return out

    def _m2_rows(self):
        """Indices of the basis rows that encode ``F = −m²``.

        They are identified by value on the mass grid (the row equals
        ``−m²``), so this does not depend on a positional convention.
        """
        bas = np.asarray(self.config["basis_table"])
        m_min = float(self.config["m_min"])
        m_delta = float(self.config["m_delta"])
        grid = m_min + m_delta * np.arange(bas.shape[1])
        target = -(grid ** 2)
        rows = [m for m in range(bas.shape[0])
                if np.allclose(bas[m].real, target, atol=1e-9)
                and np.allclose(bas[m].imag, 0.0, atol=1e-12)]
        return rows

    # -- backward -------------------------------------------------------
    def backward(self, total_grads, resolved):
        """Kernel grads → per-resolved-name gradient dict."""
        grads = dict(total_grads)
        coeff_g = grads.pop("coeff", None)
        # this parameterisation is ck + coeff; there is no m0/g0 block, so
        # the base class's positional chaining is not used here
        out = dict(self._pc.backprop_grad(resolved, grads.pop("ck")))
        if coeff_g is not None:
            g = np.asarray(coeff_g, np.float64).reshape(-1)
            if g.size != self._n_basis:
                raise ValueError(
                    f"FlatKernelParams: grads['coeff'] has {g.size} entries, "
                    f"expected n_basis = {self._n_basis}")
            # fan-in: every basis column contributes to its parameter, so the
            # copies the forward pass fanned out are summed back here
            for m, name in enumerate(self._den_names):
                out[name] = out.get(name, 0.0) + float(g[m])
        return out

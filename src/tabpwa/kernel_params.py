"""Kernel parameter transforms (resolved params ↔ kernel arrays).

Each amplitude model owns a concrete transform: the model decides which
parameter groups exist, so there is no runtime "is scalar present?" branch.

* :class:`BuildKernelParams` — common machinery: ck (Wirtinger backprop),
  m0 and g0 (real, positional).
* :class:`PWAKernelParams` — the pure projection-sum PWA model: ck/m0/g0.
* :class:`FlavourTagMixKernelParams` — the legacy time/mixing model: adds
  the six flavour-tagged scalars.
"""
import numpy as np

from tabpwa.param_constraint import CKProduct, complex_tail


class BuildKernelParams:
    """Resolved dict ↔ kernel params for ck/m0/g0 (base class).

    Constructed by the amplitude model (``model.build_params_transform()``);
    the model is the only thing that knows this constructor's input.
    """

    def __init__(self, model):
        self.model = model
        self.tail = complex_tail()
        self._pc = CKProduct(model.get_ck_map(), tail=self.tail)
        self._m0_names = list(model.m0_phys_name)
        self._g0_names = list(model.g0_phys_name)

    @property
    def pc(self):
        """The :class:`~tabpwa.param_constraint.CKProduct` for CK."""
        return self._pc

    # -- parameter surface (generic: whatever the model declares) --------
    def param_names(self):
        """All resolved parameter names of this model.

        Order matches the legacy flat list: the CK real/imag names are
        sorted *together* (interleaved ``.._i, .._r``), then m0, then g0 —
        so seeded initial values are reproducible across the refactor.
        """
        bases = sorted({p for comb in self.model.get_ck_map()
                        for p in comb if isinstance(p, str)})
        names = (sorted([b + self.tail[0] for b in bases]
                        + [b + self.tail[1] for b in bases])
                 + list(self.model.m0_phys_name)
                 + list(self.model.g0_phys_name))
        return list(dict.fromkeys(names))

    def param_defaults(self):
        """Default values for the parameters this model adds (base: none)."""
        return {}

    # -- forward --------------------------------------------------------
    def forward(self, resolved):
        """Resolved dict → kernel params dict (ck/m0/g0)."""
        return {
            "ck": self._pc.build_ck(resolved),
            "m0": np.array([resolved[n] for n in self._m0_names]),
            "g0": np.array([resolved[n] for n in self._g0_names]),
        }

    # -- backward -------------------------------------------------------
    def backward(self, total_grads, resolved):
        """Kernel grads dict → per-resolved-name gradient dict."""
        grad_dict = self._pc.backprop_grad(resolved, total_grads["ck"])
        for names_list, key in ((self._m0_names, "m0"),
                                (self._g0_names, "g0")):
            arr = np.asarray(total_grads[key])
            for i, name in enumerate(names_list):
                if i < len(arr):
                    grad_dict[name] = grad_dict.get(name, 0.0) + arr[i]
        return grad_dict


class PWAKernelParams(BuildKernelParams):
    """Pure projection-sum PWA: ck / m0 / g0 only — no scalar group."""


class PWAGenKernelParams(PWAKernelParams):
    """``pwa_gen``: ck/m0/g0 plus the explicit denominator coefficients.

    The model may carry an extended denominator basis (extra basis rows of a
    :class:`~tabpwa.particle_model.base.BasisModel`) and a free numerator
    basis.  In that case the kernel gets an explicit ``coeff`` array
    (``(n_fact, n_slots)``, the host slot layout of the kernel config's
    ``den_row``) instead of ``m0``/``g0``, and the raw per-slot gradient
    ``grads["coeff"]`` is chained back to the basis *weight* names.  Those
    weights are produced by the model's combined physical → weight transform
    (registered through ``Fitter.setup_mass_width_transforms``), so the
    gradient continues to the physical parameters there.

    Without an extended basis the class is exactly :class:`PWAKernelParams`
    (``m0``/``g0``, no ``coeff``) — the ``pwa`` numbers, bit for bit.
    """

    def __init__(self, model):
        super().__init__(model)
        kc = model.build_kernel_config()
        self._generic = "den_row" in kc
        if not self._generic:
            return
        den_row = np.asarray(kc["den_row"])
        bow = np.asarray(kc["bw_order"]).reshape(-1)
        n_bw = den_row.shape[0]
        self._den_row = den_row
        self._n_slots = den_row.shape[1]
        self._n_fact = int(bow.size)
        self._bw_of_fact = bow
        self._base_bw = bow % n_bw
        self._n_bw = n_bw
        self._den_names = list(model._pg_den_names)
        self._facts_of_bw = [np.nonzero(self._base_bw == b)[0]
                             for b in range(n_bw)]
        self._inputs = list(model._coef_transform().input_names)

    # -- parameter surface ----------------------------------------------
    def param_names(self):
        base = super().param_names()
        if not self._generic:
            return base
        return list(dict.fromkeys(base + self._inputs))

    # -- forward --------------------------------------------------------
    def forward(self, resolved):
        out = super().forward(resolved)
        if not self._generic:
            return out
        coeff = np.zeros((self._n_fact, self._n_slots))
        for b in range(self._n_bw):
            facts = self._facts_of_bw[b]
            if facts.size == 0:
                continue
            names = self._den_names[b]
            vals = [float(np.real(resolved[n])) for n in names]
            coeff[facts, 0] = vals[0]
            # slot 1 is F = −m² (the s0c sign convention): the coefficient
            # is the NEGATED weight of the m² basis row.
            coeff[facts, 1] = -vals[1]
            for j in range(2, len(names)):
                coeff[facts, j] = vals[j]
        out["coeff"] = coeff
        return out

    # -- backward -------------------------------------------------------
    def backward(self, total_grads, resolved):
        total_grads = dict(total_grads)
        if self._generic:
            total_grads.setdefault("m0", np.zeros(len(self._m0_names)))
            total_grads.setdefault("g0", np.zeros(len(self._g0_names)))
        grad_dict = super().backward(total_grads, resolved)
        if not self._generic:
            return grad_dict
        gc = total_grads.get("coeff")
        if gc is not None:
            gc = np.asarray(gc, dtype=float).reshape(self._n_fact,
                                                     self._n_slots)
            for b in range(self._n_bw):
                facts = self._facts_of_bw[b]
                if facts.size == 0:
                    continue
                g = gc[facts].sum(axis=0)
                names = self._den_names[b]
                grad_dict[names[0]] = grad_dict.get(names[0], 0.0) + g[0]
                grad_dict[names[1]] = grad_dict.get(names[1], 0.0) - g[1]
                for j in range(2, len(names)):
                    grad_dict[names[j]] = (grad_dict.get(names[j], 0.0)
                                           + g[j])
        return grad_dict


class PolarisedKernelParams(BuildKernelParams):
    """Polarised PWA: ck / m0 / g0 plus the spin-density parameters.

    The spin parameters are model-declared scalars (``model.scalar_names``
    / ``scalar_defaults`` — the same mechanism as the legacy mixing
    model).  The backend transforms them into the intensity contraction.
    """

    def __init__(self, model):
        super().__init__(model)
        self.spin_names = list(model.scalar_names)   # policy from the model

    def param_names(self):
        # Dedup again: a spin name could collide with a ck/m0/g0 name.
        return list(dict.fromkeys(
            super().param_names() + list(self.spin_names)))

    def param_defaults(self):
        model_defaults = self.model.scalar_defaults or {}
        return {n: float(model_defaults.get(n, 0.0))
                for n in self.spin_names}

    def forward(self, resolved):
        out = super().forward(resolved)
        out["scalar"] = np.array([resolved[n] for n in self.spin_names])
        return out

    def backward(self, total_grads, resolved):
        grad_dict = super().backward(total_grads, resolved)
        scalar_g = total_grads.get("scalar")
        if scalar_g is not None:
            arr = np.asarray(scalar_g)
            for i, name in enumerate(self.spin_names):
                if i < len(arr):
                    grad_dict[name] = grad_dict.get(name, 0.0) + arr[i]
        return grad_dict


class FlavourTagMixKernelParams(BuildKernelParams):
    """Legacy time/mixing model: adds the config's scalar parameters."""

    def __init__(self, model):
        super().__init__(model)
        self.scalar_names = list(model.scalar_names)   # policy from the model

    def param_names(self):
        # Dedup again: a scalar name could collide with a ck/m0/g0 name.
        return list(dict.fromkeys(
            super().param_names() + list(self.scalar_names)))

    def param_defaults(self):
        model_defaults = self.model.scalar_defaults or {}
        return {n: float(model_defaults.get(n, 0.0))
                for n in self.scalar_names}

    def forward(self, resolved):
        out = super().forward(resolved)
        out["scalar"] = np.array([resolved[n] for n in self.scalar_names])
        return out

    def backward(self, total_grads, resolved):
        grad_dict = super().backward(total_grads, resolved)
        scalar_g = total_grads.get("scalar")
        if scalar_g is not None:
            arr = np.asarray(scalar_g)
            for i, name in enumerate(self.scalar_names):
                if i < len(arr):
                    grad_dict[name] = grad_dict.get(name, 0.0) + arr[i]
        return grad_dict

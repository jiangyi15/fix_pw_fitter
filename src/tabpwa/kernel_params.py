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

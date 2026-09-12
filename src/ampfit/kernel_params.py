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

from ampfit.param_constraint import CKProduct


class BuildKernelParams:
    """Resolved dict ↔ kernel params for ck/m0/g0 (base class)."""

    def __init__(self, config):
        self.config = config
        self._pc = CKProduct(config.get_ck_map())
        self._m0_names = list(config.m0_phys_name)
        self._g0_names = list(config.g0_phys_name)

    @property
    def pc(self):
        """The :class:`~ampfit.param_constraint.CKProduct` for CK."""
        return self._pc

    # -- parameter surface (generic: whatever the model declares) --------
    def param_names(self):
        """All resolved parameter names of this model."""
        bases = sorted({p for comb in self.config.get_ck_map()
                        for p in comb if isinstance(p, str)})
        names = ([b + "r" for b in bases] + [b + "i" for b in bases]
                 + list(self.config.m0_phys_name)
                 + list(self.config.g0_phys_name))
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

    def __init__(self, config):
        super().__init__(config)
        self.scalar_names = list(config.scalar_names)

    def param_names(self):
        return super().param_names() + list(self.scalar_names)

    def param_defaults(self):
        cfg_defaults = self.config.scalar_defaults or {}
        return {n: float(cfg_defaults.get(n, 0.0))
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

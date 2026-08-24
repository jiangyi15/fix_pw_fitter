"""
Independent pipeline stages for parameter constraints.

Each stage is a small object with ``forward(input_dict)`` (transform)
and ``backward(grad_dict)`` (gradient backpropagation):

    raw = registry.to_dict(x)
    d   = fixed_tr.forward(raw)       # inject constant values
    d   = name_res.forward(d)         # alias → canonical (same)
    for scale in scale_transforms:    # multiply by scale factors
        d = scale.forward(d)
    ck  = pc.build_ck(d)              # partial-wave amplitudes
    ...
    grad = pc.backprop_grad(d, grad_ck)
    for scale in reversed(scale_transforms):
        grad = scale.backward(grad)
    grad = name_res.backward(grad, d)
    grad = fixed_tr.backward(grad, raw)
    flat = registry.flat_gradient(x, grad)
"""

import numpy as np


class CKProduct:
    """Pure combinatorics: named values → partial-wave amplitudes (ck).

    ::

        ck[i] = ∏_{term ∈ comb[i]} term_value
    """

    def __init__(self, all_comb):
        self.all_comb = list(all_comb)
        self.n_wave = len(all_comb)

    def _get_val(self, slot_dict, name):
        """Get the complex value for a term from slot-level dict.
        
        If the dict has ``name + 'r'`` / ``name + 'i'`` slots, treat
        as a complex ck parameter: ``r·exp(j·θ)``.
        Otherwise use ``name`` directly (real/m0/g0/scalar).
        
        NOTE: coupling parameters not in slot_dict default to 0+0j.
        """
        if name + 'r' in slot_dict:
            r = slot_dict[name + 'r']
            theta = slot_dict.get(name + 'i', 0.0)
            return r * np.exp(1j * theta)
        return slot_dict.get(name, 0.0 + 0.0j)

    def build_ck(self, slot_dict):
        """Compute ck from slot-level real dict.

        Slot dict has ``{name_r: mag, name_i: phase}`` for complex
        ck parameters and ``{name: value}`` for real parameters.
        """
        ck = np.empty(self.n_wave, dtype=complex)
        for i, comb in enumerate(self.all_comb):
            prod = 1.0 + 0.0j
            for term in comb:
                if isinstance(term, str):
                    prod *= self._get_val(slot_dict, term)
                else:
                    prod *= term
            ck[i] = prod
        return ck

    def free_param_names(self):
        """All canonical parameter names from the comb structure (compat)."""
        names = set()
        for comb in self.all_comb:
            for p in comb:
                if isinstance(p, str):
                    names.add(p)
        return sorted(names)

    def backprop_grad(self, slot_dict, grad_ck):
        """Gradient w.r.t. each slot from kernel's ``grad_ck``.

        Returns ``{name_r: dQ/dr, name_i: dQ/dθ}`` for complex ck
        params, ``{name: dQ/dval}`` for real params.

        Uses Wirtinger calculus: for a real-valued Q, the chain rule
        through ``ck = r·exp(j·θ)`` gives::

            dQ/dr = 2 · Re( Σ grad_ck · order · ck / r )
            dQ/dθ = 2 · Re( Σ grad_ck · j · order · ck )
        """
        ck = self.build_ck(slot_dict)
        grads = {}
        for i, comb in enumerate(self.all_comb):
            counts = {}
            for term in comb:
                if isinstance(term, str):
                    counts[term] = counts.get(term, 0) + 1
            for name, order in counts.items():
                dck = order * ck[i]
                if name + 'r' in slot_dict:
                    r = slot_dict[name + 'r']
                    grads[name + 'r'] = grads.get(name + 'r', 0.0) + 2.0 * np.real(grad_ck[i] * dck / r)
                    grads[name + 'i'] = grads.get(name + 'i', 0.0) + 2.0 * np.real(grad_ck[i] * dck * 1j)
                else:
                    v = slot_dict.get(name, 1.0)
                    grads[name] = grads.get(name, 0.0) + 2.0 * np.real(grad_ck[i] * dck / v)
        return grads


# ================================================================
# VariableRegistry — flat vector ↔ named dict
# ================================================================

class VariableRegistry:
    """Maps named variables to the flat real vector ``x``.

    Every entry represents one slot (no complex/real distinction).
    Complex coupling parameters occupy two slots (``name_r``, ``name_i``)
    registered as separate entries.
    """

    def __init__(self):
        self._entries = []
        self._name_to_entry = {}

    def add(self, name):
        self._entries.append(name)
        self._name_to_entry[name] = name

    def add_real(self, name):
        """Convenience: real-valued parameter (single slot)."""
        self.add(name)

    def add_complex(self, base):
        """Convenience: complex parameter → two slots ``base + 'r'``, ``base + 'i'``."""
        self.add(base + 'r')
        self.add(base + 'i')

    @property
    def names(self):
        return list(self._name_to_entry.keys())

    @property
    def flat_names(self):
        return list(self._entries)

    @property
    def n_flat(self):
        return len(self._entries)

    def flat_index(self, name):
        for i, n in enumerate(self._entries):
            if n == name:
                return (i, i + 1)
        raise KeyError(f"Unknown variable: {name}")

    def build_initial(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        x = np.empty(self.n_flat)
        for i in range(self.n_flat):
            x[i] = np.random.uniform(-np.pi, np.pi)
        return x

    def build_initial_deterministic(self):
        return np.zeros(self.n_flat)

    # ── forward: flat x → {slot_name: real_value} ──────────────

    def to_dict(self, x):
        return dict(zip(self._entries, x))

    # ── backward: {slot_name: grad} → flat gradient ────────────

    def flat_gradient(self, x, grad_dict):
        flat = np.zeros(self.n_flat)
        for i, n in enumerate(self._entries):
            flat[i] = grad_dict.get(n, 0.0)
        return flat

    # ── compat shims ────────────────────────────────────────────

    def extract_real_dict(self, x):
        return self.to_dict(x)

    def extract_by_target(self, x, target_type):
        return np.array([])

    def backprop_grad(self, x, grad_dict):
        return self.flat_gradient(x, grad_dict)


# ================================================================
# NameResolution — alias→canonical (same-parameter groups)
# ================================================================

class NameResolution:
    """Resolves alias names to canonical names.

    Forward: ``d[canon] = raw[alias]`` (last alias wins for each canon).
    Backward: ``grad[alias] = grad_out[canon]``.

    Keys are exact slot names — no suffix manipulation.
    """

    def __init__(self):
        self.map = {}          # {alias: canonical}

    @property
    def input_names(self):
        # Root canons: canons that are NOT themselves aliases
        all_aliases = set(self.map.keys())
        return list(set(self.map.values()) - all_aliases)

    @property
    def output_names(self):
        return list(self.map.keys())

    def set_same(self, groups, reset=False):
        # Additive union-find: subsequent calls extend the equal groups,
        # reset=True starts fresh.  The first element of a group (its
        # canonical) is the root, so the canonical convention is kept
        # (e.g. charge pairs [[pn, mn]] keep "p" as the root, not "m"):
        #   [["a","b"],["b","c"]] ⇒ a = b = c, map = {b: a, c: a}
        #   [["a","b"],["b","a"]] ⇒ cycle collapses to {b: a}
        if reset or not hasattr(self, "_parent"):
            self._parent = {}
        parent = self._parent
        old_map = dict(getattr(self, "map", {}))

        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            # *a* is the group canonical — its root stays the root
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # roots established before this call
        established = set(old_map.values())
        for group in groups:
            if not group:
                continue
            # roots established before *this* group (previous calls and
            # earlier groups of this call)
            before = set(established)
            for alias in group[1:]:
                union(group[0], alias)
            for alias in group[1:]:
                root = find(alias)
                if root in before and root != alias:
                    print(f"  same: {alias} unified into previous "
                          f"group {root}")
            established.add(find(group[0]))

        self.map = {n: find(n) for n in parent if find(n) != n}

        # log existing aliases whose root changed vs the previous call
        for alias, root in self.map.items():
            if alias in old_map and old_map[alias] != root:
                print(f"  same: {alias} re-unified into {root} "
                      f"(was {old_map[alias]})")

    def _resolve_slot(self, key):
        return self.map.get(key, key)

    def apply(self, d):
        # Canon values are already in *d* (they're the free params).
        # Alias keys, if present, must mirror the canon — one pass.
        result = dict(d)
        for alias, canon in self.map.items():
            if canon in result:
                result[alias] = result[canon]
        return result

    def inverse(self, d):
        """Reverse of :meth:`apply`: no-op (all keys already present)."""
        return dict(d)

    def chain_grad(self, grad_out, d_in):
        """Reverse of :meth:`apply` — maps resolved grads back to raw keys."""
        result = {}
        for key in d_in:
            resolved = self._resolve_slot(key)
            g = grad_out.get(resolved, 0.0)
            # Also accumulate contributions from aliases that map here
            for alias, canon in self.map.items():
                if canon == resolved and alias in grad_out:
                    g += grad_out[alias]
            if key == resolved:  # only keep canonical keys
                result[key] = g
        return result


# ================================================================
# Transform — general parameter transform base class
# ================================================================

class Transform:
    """General base class for parameter-space transforms.

    A ``Transform`` maps from a set of input parameters to a set of
    output parameters, with differentiable forward/backward passes.

    ── ``forward(input_dict) → output_dict``
    ── ``backward(grad_output, d_input=None) → grad_input``
    ── ``inverse(output_dict) → input_dict``  (optional — set
        ``has_inverse = False`` on subclasses that don't support it)

    Subclasses that need save/load support should also implement
    :meth:`to_dict` and :classmethod:`from_dict` and be registered
    with :func:`_register_transform`.

    Parameters
    ----------
    input_names : list of str, optional
        Names of input parameters (used for registry introspection).
    output_names : list of str, optional
        Names of output parameters.
    """

    _has_inverse = True

    def __init__(self, input_names=None, output_names=None):
        self._input_names = list(input_names) if input_names is not None else None
        self._output_names = list(output_names) if output_names is not None else None

    @property
    def input_names(self):
        return list(self._input_names) if self._input_names is not None else []

    @property
    def output_names(self):
        return list(self._output_names) if self._output_names is not None else []

    @property
    def has_inverse(self):
        """Whether this transform supports ``inverse()``."""
        return self._has_inverse

    def forward(self, d):
        """Transform input dict → output dict.

        Should only modify entries in ``output_names``.  Other entries
        are passed through unchanged.
        """
        raise NotImplementedError

    def apply_forward(self, d):
        """Apply :meth:`forward` and return the updated dict.

        Only entries in ``output_names`` may change; values for other keys
        are preserved from the input *d*.  Only ``input_names`` entries are
        passed to :meth:`forward` — transforms should not read values they
        didn't declare as inputs.

        Stores ``self._last_input`` for :meth:`apply_backward` so subclasses
        can read the exact pre-transform parameter values during backward.
        """
        saved = {k: v for k, v in d.items() if k not in self.output_names}
        inputs = {k: v for k, v in d.items() if k in self.input_names}
        self._last_input = dict(inputs)
        result = self.forward(inputs)
        for k, v in saved.items():
            result[k] = v
        return result

    def backward(self, grad_out, d_in=None):
        """Backpropagate gradient from output space to input space.

        Subclasses implement this.  Returns a dict mapping **only**
        ``input_names`` to their gradients.

        Args
        ----
        grad_out : dict
            Gradients w.r.t. output parameters ``{name: value}``.
        d_in : dict or None
            Input dict passed to forward (may be needed for chain rule).

        Returns
        -------
        dict
            Gradients w.r.t. input parameters (only ``input_names`` keys).
        """
        raise NotImplementedError

    def apply_backward(self, grad_out, d_in=None):
        """Apply backward and merge into *grad_out*.

        Uses ``self._last_input`` (stored by :meth:`apply_forward`) as the
        ``d_in`` argument to :meth:`backward` — never the post-transform
        resolved dict, guaranteeing correct gradients even when transforms
        chain.

        Rules
        -----
        * Input **and** output: **replace** — the transform's backward
          gives the complete derivative for parameters it both reads
          and writes (e.g. pass-through like ``re_00``).
        * Input **only**: **accumulate** — adds the gradient through the
          transform's gamma path to any existing kernel gradient.
        * Output **only**: **remove** — derived quantities that should
          not be in the optimizer space.
        """
        d_in = getattr(self, '_last_input', {})
        back = self.backward(grad_out, d_in=d_in)
        for name in self.input_names:
            if name in back:
                if name in self.output_names:
                    grad_out[name] = back[name]          # replace
                else:
                    grad_out[name] = grad_out.get(name, 0.0) + back[name]  # accumulate
        for name in self.output_names:
            if name not in self.input_names:
                grad_out.pop(name, None)
        return grad_out

    def inverse(self, d):
        """Reverse of :meth:`forward`: output dict → input dict.

        Should only modify entries in ``input_names``.
        Raise :class:`NotImplementedError` if unsupported.
        """
        raise NotImplementedError

    def apply_inverse(self, d):
        """Apply :meth:`inverse` and return the updated dict.

        Only entries in ``input_names`` may change; values for other keys
        are preserved from the input *d*.
        """
        saved = {k: v for k, v in d.items() if k not in self.input_names}
        try:
            result = self.inverse(d)
        except NotImplementedError:
            return d
        for k, v in saved.items():
            result[k] = v
        return result

    def to_dict(self):
        """Serialize transform configuration to a JSON-compatible dict.

        The dict must include a ``"type"`` key matching the name registered
        via :func:`_register_transform`.

        Raises :class:`NotImplementedError` if the transform does not
        support serialisation.
        """
        raise NotImplementedError


# Registry + dispatch for transform save/load
_transform_registry = {}


def _register_transform(cls):
    """Register a Transform subclass for deserialisation by ``transform_from_dict``."""
    _transform_registry[cls.__name__] = cls
    return cls


def transform_from_dict(d, **kwargs):
    """Reconstruct a Transform from its ``to_dict()`` serialisation.

    Args:
        d: dict with ``"type"`` key and subclass-specific fields.
        **kwargs: forwarded to ``cls.from_dict(d, **kwargs)``.
                  Each transform type documents what kwargs it expects.

    Returns:
        A :class:`Transform` instance.
    """
    cls = _transform_registry.get(d["type"])
    if cls is None:
        raise ValueError(f"Unknown transform type '{d['type']}'. "
                         f"Registered: {list(_transform_registry)}")
    return cls.from_dict(d, **kwargs)


# ================================================================
# LinearTransform — scale + bias a single parameter
# ================================================================

class LinearTransform(Transform):
    """Apply a linear transform to a single named parameter.

    ``forward``:  ``d[name] = factor * d[name] + bias``
    ``backward``: ``grad[name] *= factor``  (bias doesn't affect gradient)
    ``inverse``:  ``d[name] = (d[name] - bias) / factor``

    Input and output share the same parameter name; only the value
    is transformed.  Multiple independent ``LinearTransform`` instances
    are composed as a list in :class:`ConstraintManager`.
    """

    def __init__(self, name, factor, bias=0.0):
        super().__init__(input_names=[name], output_names=[name])
        self.name = name
        self.factor = float(factor)
        self.bias = float(bias)

    def forward(self, d):
        d = dict(d)
        if self.name in d:
            d[self.name] = self.factor * d[self.name] + self.bias
        return d

    def backward(self, grad_out, d_in=None):
        grad = dict(grad_out)
        if self.name in grad:
            grad[self.name] = grad[self.name] * self.factor
        return grad

    def inverse(self, d):
        d = dict(d)
        if self.name in d and self.factor != 0:
            d[self.name] = (d[self.name] - self.bias) / self.factor
        return d


# Backward-compatible alias
ScaleTransform = LinearTransform


# ================================================================
# BWParamsTransform — get_bw_params as a Transform
# ================================================================

@_register_transform
class BWParamsTransform(Transform):
    """Transform: model parameters → Breit-Wigner mass and width.

    ``forward`` reads the model's physical parameters (mass + gamma
    couplings) from the resolved dict and writes
    ``"{model.name}_mass_bw"`` and ``"{model.name}_width_bw"``.

    ``backward`` propagates gradients from the BW names back to the
    input parameters via centered finite differences through
    ``model.get_bw_params()``.

    ``has_inverse`` is ``False``.

    Parameters
    ----------
    model : object
        Particle model with ``get_bw_params(dict)``,
        ``get_gamma_name()``, and a ``name`` attribute.
    """

    _has_inverse = False

    def __init__(self, model):
        self.model = model
        in_names = [f"{model.name}_mass"] + list(model.get_gamma_name())
        out_names = [f"{model.name}_mass_bw", f"{model.name}_width_bw"]
        super().__init__(input_names=in_names, output_names=out_names)

    def forward(self, d):
        bw = self.model.get_bw_params(d)
        return {self.output_names[0]: bw["mass_bw"],
                self.output_names[1]: bw["width_bw"]}

    def backward(self, grad_out, d_in=None):
        resolved = dict(d_in) if d_in else {}
        d_mass = grad_out.get(self.output_names[0], 0.0)
        d_width = grad_out.get(self.output_names[1], 0.0)
        if d_mass == 0.0 and d_width == 0.0:
            return {name: 0.0 for name in self.input_names}

        eps = 1e-6
        bw0 = self.model.get_bw_params(resolved)
        grads = {}
        for name in self.input_names:
            rp = dict(resolved)
            rp[name] = resolved[name] + eps
            bw_p = self.model.get_bw_params(rp)

            rm = dict(resolved)
            rm[name] = resolved[name] - eps
            bw_m = self.model.get_bw_params(rm)

            dm_dx = (bw_p["mass_bw"] - bw_m["mass_bw"]) / (2 * eps)
            dw_dx = (bw_p["width_bw"] - bw_m["width_bw"]) / (2 * eps)
            grads[name] = d_mass * dm_dx + d_width * dw_dx
        return grads

    def to_dict(self):
        """Serialize to a JSON-compatible dict.

        The serialised form includes ``particle_name``, ``model_type``
        (registered name), and ``model_kwargs`` (constructor kwargs)
        so the model can be reconstructed standalone via
        ``build_particle()``.
        """
        from ampfit.particle_model.base import ALL_MODELS
        # Find the registered name for this model class
        model_type = next((n for n, c in ALL_MODELS.items()
                          if isinstance(self.model, c)),
                         type(self.model).__name__)
        # Only serialise JSON-safe kwargs (exclude numpy arrays etc.)
        safe_kwargs = {k: v for k, v in self.model.kwargs.items()
                       if isinstance(v, (str, int, float, bool, list, dict))}
        return {"type": "BWParamsTransform",
                "particle_name": self.model.name,
                "model_type": model_type,
                "model_kwargs": safe_kwargs}

    @classmethod
    def from_dict(cls, d, **kwargs):
        """Reconstruct from *to_dict()* output.

        Resolution order (first wins):
        1. ``model`` keyword argument — direct model instance.
        2. ``fitter`` keyword argument — calls ``fitter.get_particle_model()``.
        3. Standalone — uses ``build_particle()`` from saved
           ``model_type`` and ``model_kwargs``.

        Args:
            d: dict from ``to_dict()``.
            **kwargs: may include ``model`` (instance) or ``fitter``.

        Returns:
            A new :class:`BWParamsTransform` instance.
        """
        model = kwargs.get("model")
        if model is None and "fitter" in kwargs:
            model = kwargs["fitter"].get_particle_model(d["particle_name"])
        if model is None:
            from ampfit.particle_model.base import build_particle
            model = build_particle(d["particle_name"],
                                   model=d["model_type"],
                                   **d.get("model_kwargs", {}))
        return cls(model)


# ================================================================
# Prior — additive NLL penalty
# ================================================================

# Registry for prior deserialization
_prior_registry = {}


def _register_prior(cls):
    """Register a Prior subclass for deserialization by ``prior_from_dict``."""
    _prior_registry[cls.__name__] = cls
    return cls


class Prior:
    """Base class for additive NLL priors (penalties).

    A prior contributes a scalar to the NLL and provides per-parameter
    gradients w.r.t. resolved parameter names, identified by
    ``input_names``.  Unlike :class:`Transform`, there is no
    ``inverse()`` — the penalty is just added to the loss.

    Subclasses must implement :meth:`fun`, :meth:`gradients`, and
    :meth:`to_dict` (for JSON serialisation).
    """

    input_names = []

    def fun(self, resolved):
        """Prior NLL contribution, scalar, from resolved parameter dict."""
        raise NotImplementedError

    def gradients(self, resolved):
        """Per-parameter gradients ``{name: value}``."""
        raise NotImplementedError

    def to_dict(self):
        """Serialize prior configuration to a JSON-compatible dict.

        The dict must include a ``"type"`` key matching the class name
        registered via :func:`_register_prior` (or the class ``__name__``).
        """
        raise NotImplementedError


def prior_from_dict(d):
    """Reconstruct a Prior from its ``to_dict()`` serialisation.

    Args:
        d: dict with ``"type"`` key (class name) and subclass-specific fields.

    Returns:
        A :class:`Prior` instance.
    """
    cls = _prior_registry.get(d["type"])
    if cls is None:
        raise ValueError(f"Unknown prior type '{d['type']}'. "
                         f"Registered: {list(_prior_registry)}")
    return cls.from_dict(d)


@_register_prior
class GaussianPrior(Prior):
    """Gaussian (quadratic) penalty on one or more parameters.

    Adds ``½ Σ ((xᵢ − μᵢ) / σᵢ)²`` to the NLL.

    Parameters
    ----------
    input_names : str or list of str
        Parameter name(s) in the resolved dict.
    mu : float or array-like
        Mean value(s).  Broadcast to match *input_names*.
    sigma : float or array-like
        Width(s).  Broadcast to match *input_names*.
    """

    def __init__(self, input_names, mu, sigma):
        if isinstance(input_names, str):
            input_names = [input_names]
        self.input_names = list(input_names)
        n = len(self.input_names)
        self.mu = np.broadcast_to(np.atleast_1d(np.asarray(mu, float)), n).copy()
        self.sigma = np.broadcast_to(np.atleast_1d(np.asarray(sigma, float)), n).copy()

    def fun(self, resolved):
        dx = np.array([resolved[name] for name in self.input_names]) - self.mu
        return float(0.5 * np.sum((dx / self.sigma) ** 2))

    def gradients(self, resolved):
        dx = np.array([resolved[name] for name in self.input_names]) - self.mu
        return {name: float(dx[i] / self.sigma[i] ** 2)
                for i, name in enumerate(self.input_names)}

    def to_dict(self):
        return {
            "type": "GaussianPrior",
            "input_names": list(self.input_names),
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["input_names"], mu=d["mu"], sigma=d["sigma"])


# ================================================================
# FixedOverride — inject constant parameter values
# ================================================================
# FixedOverride — inject constant parameter values
# ================================================================

class FixedOverride:
    """Injects fixed (constant) values into the slot-level param dict.

    Works with slot-level names (``name_r``, ``name_i`` for complex ck
    params, ``name`` for real params).  Only overrides keys that exist
    in the dict — fully-fixed ck params are already excluded from the
    registry, so their slot names won't appear.
    """

    def __init__(self):
        self.values = {}       # {slot_name: value}

    @property
    def input_names(self):
        return []

    @property
    def output_names(self):
        return list(self.values.keys())

    def set_fixed(self, fixed):
        self.values = dict(fixed)

    def apply(self, d):
        d = dict(d)
        for name, val in self.values.items():
            d[name] = val  # inject fixed value even if not in raw dict
        return d

    def inverse(self, d):
        """Reverse of :meth:`apply`: no-op (forward re-injects fixed values)."""
        return dict(d)

    def chain_grad(self, grad_out, d_in):
        grad = dict(grad_out)
        for name in self.values:
            grad.pop(name, None)
        return grad


# ================================================================
# Bound constraint helper (re-export)
# ================================================================
from ampfit.boundary import Boundary, BoundTransform  # noqa: F401


# ================================================================
# ConstraintManager — coordinates the pipeline stages
# ================================================================

# Scalar parameter names (time evolution, production — not per-particle).


class ConstraintManager:
    """Owns the constraint pipeline stages (:class:`NameResolution`,
    :class:`ScaleTransform`, :class:`FixedOverride`).  Does **not**
    own a :class:`CKProduct` — that lives in ``BuildKernelParams``.

    Every ``set_*`` method updates the relevant stage and rebuilds
    the registry — trivial for ~200 parameters.

    Parameter names are passed as a flat ``all_names`` list — no
    type distinction (ck / m0 / g0 / scalar separation only exists
    in :class:`Fitter`).
    """

    def __init__(self, all_names):
        self._all_names = list(all_names)

        # Default values for complete resolved dict (set by Fitter)
        self._defaults = {}

        # Pipeline stages (independent objects)
        self.name_res = NameResolution()
        self.scale_transforms = []    # list of ScaleTransform (applied in order)
        self.mass_width_transforms = []  # list of Transform from particle models
        self.custom_transforms = []   # list of Transform (applied after mass/width)
        self.fixed_tr = FixedOverride()

        # Bound transforms — stored by name in a Boundary collection.
        # No registry resolution at set time; applied at the dict level.
        self.bounds = Boundary()

        # Variable registry (rebuilt on constraint change)
        self._var_registry = None

        self._rebuild()

    # ── public read access ──────────────────────────────────────

    @property
    def var_registry(self):
        return self._var_registry

    @property
    def defaults(self):
        """Read-only view of default parameter values."""
        return dict(self._defaults)

    @property
    def alias_to_canon(self):
        return self.name_res.map

    @property
    def bound_transforms(self):
        """The :class:`~ampfit.boundary.Boundary` collection."""
        return self.bounds

    # Backward-compat property aliases
    @property
    def fixed_slots(self):
        return self.fixed_tr.values

    @property
    def same_params(self):
        return self.name_res.map

    @property
    def scale_params(self):
        return {tr.name: tr.factor for tr in self.scale_transforms}

    def free_param_names(self):
        return [n for n in self.var_registry.flat_names
                if n not in self.fixed_tr.values]

    def initial_values(self, seed=None):
        """Flat vector with random values for all free variables."""
        return self.var_registry.build_initial(seed=seed)

    # ── modifiers ──────────────────────────────────────────────

    def set_fixed(self, fixed_slots, reset=False):
        if reset:
            self.fixed_tr.values = {}
        self.fixed_tr.values.update({k: float(v) for k, v in fixed_slots.items()})
        self._rebuild()

    def set_same(self, same_params, reset=False):
        self.name_res.set_same(same_params, reset=reset)
        self._rebuild()

    def set_scale(self, scale_params, reset=False):
        if reset:
            self.scale_transforms.clear()
        for name, val in scale_params.items():
            if isinstance(val, (list, tuple)):
                factor, bias = val[0], val[1] if len(val) > 1 else 0.0
            else:
                factor, bias = val, 0.0
            self.scale_transforms.append(LinearTransform(name, factor, bias))
        self._rebuild()

    def set_mass_width_transforms(self, transforms, reset=True):
        if reset:
            self.mass_width_transforms.clear()
            # Don't clear _all_names — it also holds m0/g0/scalar names.
            # Only remove names that were added by a previous call.
            self._all_names = [n for n in self._all_names
                               if not any(n in tr.input_names
                                          for tr in self.mass_width_transforms)]
        for tr in transforms:
            if tr is not None:
                self.mass_width_transforms.append(tr)
                input_set = set(tr.input_names)
                for name in tr.input_names:
                    if name not in self._all_names:
                        self._all_names.append(name)
                for name in tr.output_names:
                    if name not in input_set and name in self._all_names:
                        self._all_names.remove(name)
        self._rebuild()

    def add_transform(self, tr):
        """Register a custom Transform at the end of the resolve pipeline.

        The transform is applied after mass/width transforms in
        :meth:`resolve` and in reverse order in :meth:`chain_gradient`.
        Output names are added to the variable registry.

        Args:
            tr: A :class:`Transform` instance.
        """
        self.custom_transforms.append(tr)
        for name in tr.output_names:
            if name not in self._all_names:
                self._all_names.append(name)
        self._rebuild()

    def set_free(self, name):
        self.fixed_tr.values.pop(name, None)
        self._rebuild()

    def set_range(self, name, lo, hi):
        self.bounds.set(name, lo, hi)
        self._rebuild()

    def unset_range(self, name):
        self.bounds.unset(name)
        self._rebuild()

    # ── forward pipeline ───────────────────────────────────────

    def set_defaults(self, defaults):
        """Set default parameter values for complete resolved dict."""
        self._defaults = dict(defaults) if defaults else {}

    def from_flat(self, x):
        """First stage: flat array → raw dict (name→value, no constraints)."""
        return self._var_registry.to_dict(x)

    def flat_resolve(self, x, stop_after=None):
        """Flat array → resolved dict, with defaults for completeness.

        Args:
            stop_after: ``'bounds'``, ``'fixed'``, ``'same'``, ``'scale'``,
                       ``'transforms'``, or ``None`` for full pipeline.
        """
        return self.resolve(self.from_flat(x), stop_after=stop_after)

    def resolve(self, raw_dict, stop_after=None):
        """Full forward pipeline: bounds → fixed → same → scale → mass/width → custom.
        
        First applies bound transforms (unbounded → bounded), then
        runs the standard constraint pipeline.  Returns a complete dict
        with defaults for any parameters not produced by the pipeline.

        Args:
            stop_after: ``'bounds'``, ``'fixed'``, ``'same'``, ``'scale'``,
                       ``'transforms'`` (all transforms but no defaults),
                       or ``None`` for full pipeline.
        """
        d = dict(raw_dict)
        self.bounds.apply(d)                     # unbounded → bounded
        if stop_after == 'bounds':
            return d
        d = self.fixed_tr.apply(d)
        if stop_after == 'fixed':
            return d
        d = self.name_res.apply(d)
        if stop_after == 'same':
            return d
        for tr in self.scale_transforms:
            d = tr.apply_forward(d)
        if stop_after == 'scale':
            return d
        for tr in self.mass_width_transforms:
            d = tr.apply_forward(d)
        if stop_after == 'transforms':
            return d
        for tr in self.custom_transforms:
            d = tr.apply_forward(d)
        if stop_after == 'custom':
            return d
        # Merge defaults at the end of the full pipeline
        full = dict(self._defaults)
        full.update(d)
        return full

    def inverse(self, resolved, stop_before=None):
        """Full inverse: custom⁻¹ → mass/width⁻¹ → scale⁻¹ → same⁻¹ → fixed⁻¹ → bounds⁻¹.

        Args:
            stop_before: stop BEFORE applying the inverse of this stage.
                ``'custom'``, ``'transforms'``, ``'scale'``, ``'same'``,
                ``'fixed'``, ``'bounds'``, or ``None`` for full pipeline.
                E.g. ``stop_before='fixed'`` undoes custom/scale/same but
                leaves fixed values intact.
        """
        d = resolved
        if stop_before == 'custom':
            return d
        for tr in reversed(self.custom_transforms):
            d = tr.apply_inverse(d)
        if stop_before == 'transforms':
            return d
        for tr in reversed(self.mass_width_transforms):
            d = tr.apply_inverse(d)
        if stop_before == 'scale':
            return d
        for tr in reversed(self.scale_transforms):
            d = tr.apply_inverse(d)
        if stop_before == 'same':
            return d
        d = self.name_res.inverse(d)
        if stop_before == 'fixed':
            return d
        d = self.fixed_tr.inverse(d)
        if stop_before == 'bounds':
            return d
        # Invert bounds: return unbounded values
        for name in list(d):
            d[name] = self.bounds.inverse(name, d[name])
        return d

    def full_gradient(self, grad_resolved, resolved, x):
        """Backprop through pipeline: chain_gradient → flat → bound correction.

        Returns flat gradient w.r.t. unbounded optimizer variables.
        """
        raw = self.from_flat(x)
        grad_raw = self.chain_gradient(grad_resolved, resolved, raw)
        grad_flat = self._var_registry.flat_gradient(x, grad_raw)
        self.bounds.correct_gradient(grad_flat, x, self._var_registry.flat_names, raw)
        return grad_flat

    def apply_bound_dict(self, d, errors=None):
        """Apply forward bounds to dict *d* (in-place).

        If *errors* dict given, propagates uncertainties through bounds.
        """
        self.bounds.apply_dict(d)
        if errors is not None:
            for name, bt in self.bounds.items():
                if name in errors:
                    errors[name] = bt.trans_err(d[name], errors[name])
        return d

    def to_dict(self):
        """Serialize all constraints (fixed, same, scale, bounds) to dict.

        Returns a JSON-compatible dict.
        """
        from collections import defaultdict
        alias_map = self.same_params
        canon_groups = defaultdict(list)
        for alias, canon in alias_map.items():
            canon_groups[canon].append(alias)
        same = [[canon] + sorted(aliases) for canon, aliases in canon_groups.items()]
        return {
            "fixed": dict(self.fixed_slots),
            "same": same,
            "scale": dict(self.scale_params),
            "bounds": self.bounds.to_dict(),
        }

    # ── backward pipeline ──────────────────────────────────────

    def chain_gradient(self, grad_resolved, resolved, raw):
        """Reverse of :meth:`resolve` (custom → mass/width → scale → same → fixed).

        Uses :meth:`Transform.apply_backward` on each stage — only
        ``input_names`` are updated; output-only and unrelated gradients
        pass through unchanged.
        """
        grad = grad_resolved
        for tr in reversed(self.custom_transforms):
            grad = tr.apply_backward(grad)
        for tr in reversed(self.mass_width_transforms):
            grad = tr.apply_backward(grad)
        for tr in reversed(self.scale_transforms):
            grad = tr.apply_backward(grad)
        grad = self.name_res.chain_grad(grad, resolved)
        grad = self.fixed_tr.chain_grad(grad, raw)
        return grad

    # ── rebuild ─────────────────────────────────────────────────

    def _rebuild(self):
        # Collect names that are genuinely produced by a transform
        # (output where the name is NOT also an input of the same transform).
        # In-place transforms (LinearTransform: same name in/out) are excluded.
        produced = set()
        for tr in self._all_transforms():
            for name in tr.output_names:
                if name not in tr.input_names:
                    produced.add(name)

        self._var_registry = VariableRegistry()
        for name in self._all_names:
            if name not in produced:
                self._var_registry.add(name)

    def _all_transforms(self):
        """Yield all constraint transforms in forward order."""
        yield self.fixed_tr
        yield self.name_res
        yield from self.scale_transforms
        yield from self.mass_width_transforms
        yield from self.custom_transforms


# ================================================================
# ParameterizedObjective wrapper
# ================================================================

class ParameterizedObjective:
    """Wraps a kernel compute function with parameter constraints."""

    def __init__(self, compute_fn, param_constraint, data_holder,
                 m0=None, g0=None, scalar=None, norm=None):
        self.pc = param_constraint
        self.compute_fn = compute_fn
        self.data_holder = data_holder
        self.m0 = m0
        self.g0 = g0
        self.scalar = scalar
        self.norm = norm

    def build_params(self, x):
        ck = self.pc.build_ck(x)
        params = {"ck": ck}
        if self.m0 is not None:
            params["m0"] = self.m0
        if self.g0 is not None:
            params["g0"] = self.g0
        if self.scalar is not None:
            params["scalar"] = self.scalar
        return params

    def __call__(self, x):
        params = self.build_params(x)
        Q, grads, P = self.compute_fn(params, self.data_holder, norm=self.norm)
        grad_ck = grads["ck"]
        grad_x = self.pc.backprop_grad(x, grad_ck)
        return Q, grad_x


# ================================================================
# Self-test
# ================================================================
if __name__ == "__main__":
    from ampfit.config_loader import Config
    config = Config("config_angle.yml")
    all_comb = config.get_ck_map()
    ck_names = sorted({p for comb in all_comb for p in comb if isinstance(p, str)})

    pc = CKProduct(all_comb)
    name_res = NameResolution()
    fixed_tr = FixedOverride()

    # Build a slot-level dict (as produced by VariableRegistry.to_dict)
    # ck params: {name_r: mag, name_i: phase}
    np.random.seed(42)
    raw = {}
    for n in ck_names:
        raw[n + 'r'] = np.random.uniform(0.5, 2.0)
        raw[n + 'i'] = np.random.uniform(-np.pi, np.pi)

    # Test resolve pipeline on slot-level dict
    d = name_res.apply(raw)
    d = fixed_tr.apply(d)
    ck = pc.build_ck(d)
    print(f"ck shape: {ck.shape}, ck[:3]: {ck[:3]}")

    # Gradient test: verify backprop_grad against numerical diff
    test_vec = np.random.randn(pc.n_wave) + 1j * np.random.randn(pc.n_wave)

    def Q(slot_dict):
        return np.real(np.sum(pc.build_ck(slot_dict) * np.conj(test_vec)))

    grad_ck_exact = 0.5 * np.conj(test_vec)

    # Analytical gradient w.r.t. the resolved slot dict
    grad_resolved = pc.backprop_grad(d, grad_ck_exact)
    # Chain back through fixed → same → raw
    grad = fixed_tr.chain_grad(grad_resolved, d)
    grad = name_res.chain_grad(grad, raw)

    eps = 1e-6
    errs = []
    for key in list(raw.keys())[:10]:   # test first 10 slots
        sd = raw.copy()
        sd[key] += eps
        Qp = Q(fixed_tr.apply(name_res.apply(sd)))
        sd[key] -= 2 * eps
        Qm = Q(fixed_tr.apply(name_res.apply(sd)))
        num = (Qp - Qm) / (2 * eps)
        ana = grad.get(key, 0.0)
        err = abs(ana - num)
        errs.append(err)
        print(f"  {key:35s} ana={ana:+.6e} num={num:+.6e} err={err:.2e}")
    print(f"Max error (first 10): {max(errs):.2e}")
    assert max(errs) < 1e-5
    print("✓ Full pipeline gradient verified")

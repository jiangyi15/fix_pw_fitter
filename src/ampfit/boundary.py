"""
boundary — Variable bound transformations for optimization.

Maps unbounded optimizer variables to bounded physical ranges using a
smooth, bijective sin transform. Provides forward, inverse, gradient,
and error propagation.

Usage:
    b = BoundTransform(0.3, 0.8)
    
    x_unbounded = 0.0          # optimizer works in unbounded space
    y_bounded   = b(x)         # 0.55  — maps to physical range
    dy_dx       = b.grad(x)    # gradient for chain rule
    x_back      = b.inv(y)     # invert back to unbounded
    
    # With error propagation:
    err_on_y = b.trans_err(x, err_on_x)
"""

import numpy as np


class BoundTransform:
    """Transform unbounded variables to a bounded [a, b] range.
    
    Uses: y = k * sin(x/k) + bias  where k = (b-a)/2, bias = (b+a)/2
    This is smooth, bijective, and has a simple gradient: cos(x/k).
    
    The inverse maps bounded values back to the unbounded range.
    trans_err propagates covariance through the transform.
    """

    def __init__(self, a, b):
        self.a = float(min(a, b))
        self.b = float(max(a, b))
        self.k = (self.b - self.a) / 2.0
        self.bias = (self.b + self.a) / 2.0

    def forward(self, x):
        """Unbounded x -> bounded y in [a, b]."""
        return self.k * np.sin(x / self.k) + self.bias

    def __call__(self, x):
        """Alias for forward()."""
        return self.forward(x)

    def grad(self, x):
        """Gradient dy/dx at x (for chain rule in gradient backprop)."""
        return np.cos(x / self.k)

    def inverse(self, y):
        """Bounded y in [a, b] -> unbounded x.
        
        Handles periodic wrapping: values outside [a, b] are folded back in.
        """
        y_clipped = np.clip(y, self.a, self.b)
        t = (y_clipped - self.bias) / self.k
        t = np.clip(t, -1.0, 1.0)
        return np.arcsin(t) * self.k

    def trans_err(self, x, error):
        """Propagate error through the transform.
        
        Args:
            x: unbounded variable value.
            error: standard deviation (or error) on x.
        
        Returns:
            error on y (bounded value) = |grad(x)| * error
        """
        return np.abs(self.grad(x)) * error


# Commonly used bounds for time-dependent amplitude parameters
TIME_PARAM_BOUNDS = {
    "delta_m":       [0.3, 0.8],
    "delta_gamma":  [-0.3, 0.3],
    "A_prod":       [-0.5, 0.5],
    "gamma":        [-0.3, 0.3],
}

# Fixed default values (used when parameters are not free)
TIME_PARAM_DEFAULTS = {
    "gamma":        0.0,
    "delta_m":      0.506,
    "delta_gamma":  0.0,
    "poqr":         1.0,
    "poqi":         0.0,
    "A_prod":       0.0,
}


def build_bound_trans(free_params, bounds=None, n_ck_free_vars=None):
    """Build a dict mapping variable indices to BoundTransform instances.
    
    Args:
        free_params: list of free time parameter names (e.g. ["gamma"]).
        bounds: dict of {name: [lo, hi]}, defaults to TIME_PARAM_BOUNDS.
        n_ck_free_vars: number of free complex parameters (ck vars).
                        If None, the index offset is inferred from the
                        number of free params (zero ck vars).
    
    Returns:
        dict of {index_in_flat_x: BoundTransform}
    """
    if bounds is None:
        bounds = TIME_PARAM_BOUNDS
    if n_ck_free_vars is None:
        n_ck_free_vars = 0

    bound_trans = {}
    offset = n_ck_free_vars * 2  # 2 real values per complex var
    for k in free_params:
        if k in bounds:
            idx = offset + free_params.index(k)
            bound_trans[idx] = BoundTransform(bounds[k][0], bounds[k][1])
    return bound_trans


def apply_bounds(x, bound_trans):
    """Transform x in-place: apply BoundTransform to specified indices.
    
    Args:
        x: variable vector to transform (modified in place).
        bound_trans: dict of {index: BoundTransform}.
    
    Returns:
        new_x (modified copy).
    """
    x = x.copy()
    for k, v in bound_trans.items():
        x[k] = v(x[k])
    return x


def apply_bound_grads(grad, x, bound_trans):
    """Correct gradients for bound transforms (chain rule).
    
    The gradient from the NLL is w.r.t. the bounded variable y.
    This converts it to w.r.t. the unbounded variable x.
    
    Args:
        grad: gradient w.r.t. bounded variables.
        x: unbounded variable values.
        bound_trans: dict of {index: BoundTransform}.
    
    Returns:
        grad_corrected (copy, modified at bound-transformed indices).
    """
    grad = grad.copy()
    for k, v in bound_trans.items():
        grad[k] = v.grad(x[k]) * grad[k]
    return grad

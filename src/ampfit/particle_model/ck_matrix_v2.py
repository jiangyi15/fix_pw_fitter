"""CK matrix model v2: Gamma(x) = c_a M_{ab}(x) c_b*.

The total running width at the pole mass is normalised to a
standalone ``{res}_width`` parameter::

    Σ g_i · gamma_i(m₀) = {res}_width

The transform reads ``{res}_width`` and ck g_ls parameters, computes
raw outer-product values from ck, normalises them, and writes the
full gamma set to the param dict.

``{res}_width`` is NOT a gamma parameter (it is not ``re_ab``).
It is registered separately by the fitter.

Config::

    particle:
      my_res:
        mass: 1.716
        width: 0.480
        model: ck_matrix_v2
        gamma_file: /path/to/gamma.npy
        partial_file: /path/to/partial.npy
        order_file: /path/to/order.json
"""

import json
import numpy as np
from .base import BaseModel, register_model
from ampfit.param_constraint import Transform


# ── reduced gamma computation ────────────────────────────────────

def _gamma_functions(m, x_table, M_table, n_ck, scale=1.0):
    """Compute reduced gamma functions at masses *m*.

    Results are divided by *scale* (so that the first diagonal
    ``re_00 = 1`` when *m* = *m₀*).

    Returns list of complex arrays in gamma-name order::

        diag:   M_aa(m)/scale         (real)
        re_ab:  Re(M_ab(m))/scale     (real)
        im_ab:  -Im(M_ab(m))/scale    (real)
    """
    inv = 1.0 / scale if scale != 0 else 1.0
    out = []
    for a in range(n_ck):
        Mab = np.interp(m, x_table, M_table[:, a, a])
        out.append(Mab.real * inv + 0j)            # re_aa
        for b in range(a + 1, n_ck):
            Mab = np.interp(m, x_table, M_table[:, a, b])
            out.append(Mab.real * inv + 0j)        # re_ab = Re(M_ab)/scale
            out.append(-Mab.imag * inv + 0j)       # im_ab = -Im(M_ab)/scale
    return out


def _gamma_at_m0(x_table, M_table, n_ck, scale, m0):
    """Compute gamma_i(m₀) for all reduced channels.

    Returns a list of real values in gamma-name order::

        a=a:       M_aa(m₀) / scale
        a<b:       Re(M_ab(m₀)) / scale   (for re_ab)
                  -Im(M_ab(m₀)) / scale  (for im_ab)
    """
    inv = 1.0 / scale if scale != 0 else 1.0
    vals = []
    for a in range(n_ck):
        M_aa = float(np.interp(m0, x_table, M_table[:, a, a].real))
        vals.append(M_aa * inv)
        for b in range(a + 1, n_ck):
            Mab = np.interp(m0, x_table, M_table[:, a, b])
            vals.append(Mab.real * inv)
            vals.append(-Mab.imag * inv)
    return vals


def _reduced_spec(n_ck, name):
    """Build gamma names (no {res}_width — separate parameter).

    Names::

        {name}_re_00, {name}_re_01, {name}_im_01, …
    """
    names = []
    for a in range(n_ck):
        names.append(f"{name}_re_{a}{a}")
        for b in range(a + 1, n_ck):
            names.append(f"{name}_re_{a}{b}")
            names.append(f"{name}_im_{a}{b}")
    return names


def _raw_expanded(ck):
    """Compute raw expanded values from ck vector (list of real)."""
    n = len(ck)
    raw = []
    for a in range(n):
        raw.append((ck[a] * np.conj(ck[a])).real)       # |c_a|²
        for b in range(a + 1, n):
            cab = ck[a] * np.conj(ck[b])
            raw.append(2.0 * cab.real)                   # 2·Re
            raw.append(2.0 * cab.imag)                   # 2·Im
    return np.array(raw, dtype=float)


class _CKWidthTransform(Transform):
    """Transform {res}_width + ck → normalised gamma values.

    Computes raw expanded values from ck, then scales them so that
    ``Σ g_i · gamma_i(m₀) = {res}_width``.
    """

    _has_inverse = False

    def __init__(self, name, order_names, gamma_names,
                 ck_r0, ck_i0, gamma_at_m0, width_name, default_width):
        self.width_name = width_name
        in_names = [width_name] + list(order_names) + \
                   [n.rstrip('r') + 'i' for n in order_names]
        super().__init__(input_names=in_names, output_names=gamma_names)
        self.name = name
        self.order_names = list(order_names)
        self.gamma_names = list(gamma_names)
        self.n_ck = len(order_names)
        self.ck_r0 = np.array(ck_r0, dtype=float)
        self.ck_i0 = np.array(ck_i0, dtype=float)
        self.gamma_at_m0 = np.array(gamma_at_m0, dtype=float)
        self._default_width = float(default_width)

    def forward(self, d):
        d = dict(d)

        width = d.get(self.width_name, self._default_width)

        ck = np.zeros(self.n_ck, dtype=complex)
        for a in range(self.n_ck):
            r = d.get(self.order_names[a], self.ck_r0[a])
            i = d.get(self.order_names[a].rstrip('r') + 'i', self.ck_i0[a])
            ck[a] = r + 1j * i

        raw = _raw_expanded(ck)
        N = np.dot(raw, self.gamma_at_m0)
        scale = width / N if N != 0 else 0.0

        idx = 0
        for a in range(self.n_ck):
            d[self.gamma_names[idx]] = raw[idx] * scale     # re_aa
            idx += 1
            for b in range(a + 1, self.n_ck):
                d[self.gamma_names[idx]]     = raw[idx] * scale      # re_ab
                d[self.gamma_names[idx + 1]] = raw[idx + 1] * scale  # im_ab
                idx += 2
        return d

    def backward(self, grad_out, d_in=None):
        d = dict(d_in) if d_in else {}

        # 1. Read inputs
        width = d.get(self.width_name, self._default_width)
        ck = np.zeros(self.n_ck, dtype=complex)
        for a in range(self.n_ck):
            r = d.get(self.order_names[a], self.ck_r0[a])
            i = d.get(self.order_names[a].rstrip('r') + 'i', self.ck_i0[a])
            ck[a] = r + 1j * i

        # 2. Raw, N, scale
        raw = _raw_expanded(ck)
        N = np.dot(raw, self.gamma_at_m0)
        if N == 0:
            return {}
        scale = width / N

        # 3. B = Σ grad_out[name] · raw[i]
        B = 0.0
        for i, name in enumerate(self.gamma_names):
            B += grad_out.get(name, 0.0) * raw[i]

        grad = {}

        # 4. Gradient w.r.t. width
        grad[self.width_name] = B / N

        # 5. Gradients w.r.t. each ck component
        n = self.n_ck
        gammas = self.gamma_at_m0

        # Pre-compute: for each ck component a, collect all raw indices
        # that depend on it, with their derivatives.
        # Build index-to-(a,b) mapping once if not cached
        if not hasattr(self, '_raw_idx_map'):
            self._raw_idx_map = []  # (a, b, type) for each raw index
            ri = 0
            for aa in range(n):
                self._raw_idx_map.append((aa, aa, 're'))  # diagonal
                ri += 1
                for bb in range(aa + 1, n):
                    self._raw_idx_map.append((aa, bb, 're'))
                    self._raw_idx_map.append((aa, bb, 'im'))
                    ri += 2

        for a in range(n):
            dN_dr = 0.0; dN_di = 0.0  # dN/d(ck_r[a]), dN/d(ck_i[a])
            A_re  = 0.0; A_im  = 0.0   # Σ grad · ∂raw/∂ck

            for ri, (aa, bb, rtype) in enumerate(self._raw_idx_map):
                g_out = grad_out.get(self.gamma_names[ri], 0.0)
                gamma_val = gammas[ri]

                if aa == bb and aa == a:
                    # Diagonal: |c_a|²
                    # ∂/∂r: 2·r_a, ∂/∂i: 2·i_a
                    dr_dr = 2.0 * ck[a].real
                    dr_di = 2.0 * ck[a].imag
                    dN_dr += dr_dr * gamma_val
                    dN_di += dr_di * gamma_val
                    A_re  += g_out * dr_dr
                    A_im  += g_out * dr_di

                elif aa == a and bb > a:
                    if rtype == 're':
                        # 2·Re(c_a c_b*) = 2(r_a·r_b + i_a·i_b)
                        # ∂/∂r_a: 2·r_b, ∂/∂i_a: 2·i_b
                        dr_dr = 2.0 * ck[bb].real
                        dr_di = 2.0 * ck[bb].imag
                    else:
                        # 2·Im(c_a c_b*) = 2(i_a·r_b - r_a·i_b)
                        # ∂/∂r_a: -2·i_b, ∂/∂i_a: 2·r_b
                        dr_dr = -2.0 * ck[bb].imag
                        dr_di =  2.0 * ck[bb].real
                    dN_dr += dr_dr * gamma_val
                    dN_di += dr_di * gamma_val
                    A_re  += g_out * dr_dr
                    A_im  += g_out * dr_di

                elif bb == a and aa < a:
                    # re_ba or im_ba where b<a (stored as pair (b,a))
                    if rtype == 're':
                        # 2·Re(c_b c_a*) = 2(r_b·r_a + i_b·i_a)
                        # ∂/∂r_a: 2·r_b, ∂/∂i_a: 2·i_b
                        dr_dr = 2.0 * ck[aa].real
                        dr_di = 2.0 * ck[aa].imag
                    else:
                        # 2·Im(c_b c_a*) = 2(i_b·r_a - r_b·i_a)
                        # ∂/∂r_a: 2·i_b, ∂/∂i_a: -2·r_b
                        dr_dr = 2.0 * ck[aa].imag
                        dr_di = -2.0 * ck[aa].real
                    dN_dr += dr_dr * gamma_val
                    dN_di += dr_di * gamma_val
                    A_re  += g_out * dr_dr
                    A_im  += g_out * dr_di

            # ∂loss/∂ck[a]_r = scale · A_re − (scale/N) · dN_dr · B
            grad[self.order_names[a]] = scale * A_re - (scale / N) * dN_dr * B
            iname_i = self.order_names[a].rstrip('r') + 'i'
            grad[iname_i] = scale * A_im - (scale / N) * dN_di * B

        return grad


@register_model("ck_matrix_v2")
class CKMatrixModelV2(BaseModel):
    """Running width from CK outer product, normalised to ``{res}_width``.

    The ``{res}_width`` parameter is NOT a gamma parameter — it is
    registered separately.  The gamma names are only the reduced
    expanded set ``re_aa``, ``re_ab``, ``im_ab`` for all a≤b.

    The transform scales the raw ck expanded components so that the
    sum of all gamma contributions at m₀ equals ``{res}_width``.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        # ── load files ───────────────────────────────────────────
        gamma_data = np.load(kwargs["gamma_file"])
        self.x_table = gamma_data[:, 0].copy()

        M = np.load(kwargs["partial_file"])
        self.M_table = np.asarray(M, dtype=complex)
        self.n_ck = self.M_table.shape[1]

        with open(kwargs["order_file"]) as f:
            self.order_names = json.load(f)
        if len(self.order_names) != self.n_ck:
            raise ValueError(...)

        n_x = len(self.x_table)
        if self.M_table.shape[0] == n_x - 1:
            zero_pad = np.zeros((1, self.n_ck, self.n_ck), dtype=self.M_table.dtype)
            self.M_table = np.concatenate([zero_pad, self.M_table], axis=0)

        # ── config values ─────────────────────────────────────────
        self.m0 = float(kwargs.get("mass", 0.775))

        # ── gamma scale so re_00(m₀) = 1 in unscaled form ────────
        self._gamma_scale = float(np.interp(self.m0, self.x_table,
                                              self.M_table[:, 0, 0].real))
        if self._gamma_scale == 0:
            self._gamma_scale = 1.0

        # ── pre-compute gamma_i(m₀) for the reduced set ──────────
        self._gamma_m0 = _gamma_at_m0(
            self.x_table, self.M_table, self.n_ck,
            self._gamma_scale, self.m0)

        # ── ck defaults ───────────────────────────────────────────
        ck_raw = kwargs.get("ck", None)
        if ck_raw is not None:
            n_parts = 2 * self.n_ck
            if len(ck_raw) != n_parts:
                raise ValueError(
                    f"ck expects {n_parts} values for n_ck={self.n_ck}, "
                    f"got {len(ck_raw)}")
            self._ck0_r = np.array(ck_raw[0::2], dtype=float)
            self._ck0_i = np.array(ck_raw[1::2], dtype=float)
        else:
            self._ck0_r = np.zeros(self.n_ck, dtype=float)
            self._ck0_r[0] = 1.0
            self._ck0_i = np.zeros(self.n_ck, dtype=float)

        # ── gamma names & defaults (no {res}_width) ──────────────
        self._g_names = _reduced_spec(self.n_ck, self.name)

        # Defaults: only-first-channel active
        ck0 = self._ck0_r + 1j * self._ck0_i
        raw0 = _raw_expanded(ck0)
        N0 = np.dot(raw0, self._gamma_m0)
        self._g_defaults = [float(self.kwargs.get("width", 0.1)) * r / N0 if N0 != 0 else 0.0
                            for r in raw0]

    # ── gamma interface ──────────────────────────────────────────

    def get_defaults(self):
        """All physical defaults: mass + reduced gamma values."""
        mass = float(self.kwargs.get("mass", 0.775))
        return {f"{self.name}_mass": mass,
                **dict(zip(self._g_names, self._g_defaults))}

    def get_gamma_count(self):
        return self.n_ck + self.n_ck * (self.n_ck - 1)  # re_aa + (re+im)*upper

    def get_gamma_name(self):
        return list(self._g_names)

    def get_gamma_defaults(self):
        return list(self._g_defaults)

    def gamma(self, m):
        """Gamma functions: M_ab(m)/M_00(m₀) for each reduced channel."""
        return _gamma_functions(m, self.x_table, self.M_table,
                                self.n_ck, self._gamma_scale)

    # ── transform ────────────────────────────────────────────────

    def make_mass_width_transform(self):
        width_name = f"{self.name}_width"
        return _CKWidthTransform(
            self.name, self.order_names, self._g_names,
            self._ck0_r, self._ck0_i,
            self._gamma_m0, width_name, float(self.kwargs.get("width", 0.1)),
        )

    # ── get_bw_params ────────────────────────────────────────────

    def get_bw_params(self, params=None):
        from scipy.optimize import root_scalar

        def _p(key, fallback=None):
            if params and key in params:
                return float(params[key])
            if key.startswith(self.name + "_"):
                bare = key[len(self.name) + 1:]
            else:
                bare = key
            if bare in self.kwargs:
                return float(self.kwargs[bare])
            return fallback

        m0 = _p(f"{self.name}_mass", 0.775)
        gamma_names = self.get_gamma_name()
        defaults = self.get_gamma_defaults()
        g0_vals = [_p(gamma_names[i], defaults[i])
                   for i in range(self.get_gamma_count())]

        # Read total width from params or kwargs
        total_width = _p(f"{self.name}_width", float(self.kwargs.get("width", 0.1)))

        def g_bw_re(m):
            g_list = self.gamma(np.array([float(m)]))
            s = 0.0
            for i, gv in enumerate(g0_vals):
                s += float(gv) * float(g_list[i][0].real)
            return s

        def sum_gamma_im(m):
            g_list = self.gamma(np.array([float(m)]))
            s = 0.0
            for i, gv in enumerate(g0_vals):
                s += float(gv) * float(g_list[i][0].imag)
            return s

        def f(m):
            return m0**2 - m**2 + m0 * sum_gamma_im(m)

        sol = root_scalar(f, x0=m0, x1=m0 * 1.1, method='secant', xtol=1e-8)
        if not sol.converged:
            raise RuntimeError(f"get_bw_params: root finding failed for {self.name}")

        mass_bw = float(sol.root)
        # The total width at the peak should equal {res}_width
        width_bw = total_width
        return {"mass_bw": mass_bw, "width_bw": width_bw}

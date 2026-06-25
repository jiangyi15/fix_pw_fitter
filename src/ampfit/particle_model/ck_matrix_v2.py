"""CK matrix model v2: Gamma(x) = c_a M_{ab}(x) c_b*.

The total running width at the pole mass is normalized to a
standalone ``{res}_width`` parameter::

    Σ g_i · gamma_i(m₀) = {res}_width

The transform reads ``{res}_width`` and ck g_ls parameters, computes
raw outer-product values from ck, normalises them, and writes the
full gamma set to the param dict.

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


def _gamma_at_m0(x_table, M_table, n_ck, scale, m0):
    """Compute gamma_i(m₀) for all reduced channels.

    Returns a list of real values in gamma-name order::

        for a:        M_aa(m₀) / scale
        for a<b:      Re(M_ab(m₀)) / scale   (for re_ab)
                      -Im(M_ab(m₀)) / scale  (for im_ab)
    """
    inv = 1.0 / scale if scale != 0 else 1.0
    vals = []
    for a in range(n_ck):
        M_aa = float(np.interp(m0, x_table, M_table[:, a, a].real))
        vals.append(M_aa * inv)       # gamma_re_aa(m₀)
        for b in range(a + 1, n_ck):
            Mab = float(np.interp(m0, x_table, M_table[:, a, b]))
            vals.append(Mab.real * inv)   # gamma_re_ab(m₀)
            vals.append(-Mab.imag * inv)  # gamma_im_ab(m₀)
    return vals


def _reduced_spec(n_ck, name):
    """Build gamma names and raw-expanded-value function.

    Gamma names::

        [{name}_width, {name}_re_00, {name}_re_01, {name}_im_01, …]

    The raw expanded values from ck are::

        for a:        |c_a|²
        for a<b:      2·Re(c_a c_b*)    (re)
                      2·Im(c_a c_b*)    (im)
    """
    names = [f"{name}_width"]
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
                 ck_r0, ck_i0, gamma_at_m0, width0):
        re_00_name = gamma_names[0]   # {res}_width
        in_names = [re_00_name] + list(order_names) + \
                   [n.rstrip('r') + 'i' for n in order_names]
        super().__init__(input_names=in_names, output_names=gamma_names)
        self.name = name
        self.order_names = list(order_names)
        self.gamma_names = list(gamma_names)
        self.n_ck = len(order_names)
        self.re_00_name = re_00_name
        self.ck_r0 = np.array(ck_r0, dtype=float)
        self.ck_i0 = np.array(ck_i0, dtype=float)
        self.gamma_at_m0 = np.array(gamma_at_m0, dtype=float)  # pre-computed
        self.width0 = float(width0)

    def forward(self, d):
        d = dict(d)

        # 1. Read {res}_width (standalone global scale)
        width = d.get(self.re_00_name, self.width0)

        # 2. Read ck
        ck = np.zeros(self.n_ck, dtype=complex)
        for a in range(self.n_ck):
            r = d.get(self.order_names[a], self.ck_r0[a])
            i = d.get(self.order_names[a].rstrip('r') + 'i', self.ck_i0[a])
            ck[a] = r + 1j * i

        # 3. Raw expanded values
        raw = _raw_expanded(ck)

        # 4. Normalisation: N = Σ raw_i · gamma_i(m₀)
        N = np.dot(raw, self.gamma_at_m0)
        scale = width / N if N != 0 else 0.0

        # 5. Write outputs
        d[self.re_00_name] = width     # pass-through
        idx = 1
        for a in range(self.n_ck):
            d[self.gamma_names[idx]] = raw[idx - 1] * scale  # re_aa = width·|c_a|²/N
            idx += 1
            for b in range(a + 1, self.n_ck):
                d[self.gamma_names[idx]]     = raw[idx - 1] * scale      # re_ab
                d[self.gamma_names[idx + 1]] = raw[idx] * scale          # im_ab
                idx += 2
        return d

    def backward(self, grad_out, d_in=None):
        eps = 1e-6
        d = dict(d_in) if d_in else {}
        grad = dict(grad_out)
        for name in self.input_names:
            d_p = dict(d); d_p[name] = d.get(name, 0.0) + eps
            out_p = self.forward(d_p)
            d_m = dict(d); d_m[name] = d.get(name, 0.0) - eps
            out_m = self.forward(d_m)
            g = 0.0
            for on in self.output_names:
                g += grad_out.get(on, 0.0) * (
                    out_p.get(on, 0.0) - out_m.get(on, 0.0)) / (2 * eps)
            grad[name] = g
        return grad


@register_model("ck_matrix_v2")
class CKMatrixModelV2(BaseModel):
    """Running width from CK outer product, normalised to ``{res}_width``.

    The gamma parameter list starts with ``{res}_width`` (the total width
    at the pole), followed by the reduced expanded set ``re_aa``, ``re_ab``,
    ``im_ab`` for all a≤b.

    The transform scales the raw ck expanded components so that the sum
    of all gamma contributions at m₀ equals ``{res}_width``.
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
        self.width0 = float(kwargs.get("width", 0.1))

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

        # ── gamma names & defaults ────────────────────────────────
        self._g_names = _reduced_spec(self.n_ck, self.name)

        # Only-first-channel active -> {res}_width, re_00=|c₀|², then 0
        ck0 = self._ck0_r + 1j * self._ck0_i
        raw0 = _raw_expanded(ck0)
        N0 = np.dot(raw0, self._gamma_m0)
        self._g_defaults = [self.width0]
        for r in raw0:
            self._g_defaults.append(self.width0 * r / N0 if N0 != 0 else 0.0)

    # ── gamma interface ──────────────────────────────────────────

    def get_gamma_count(self):
        return 1 + self.n_ck + self.n_ck * (self.n_ck - 1)  # width + re_aa + (re+im)*upper

    def get_gamma_name(self):
        return list(self._g_names)

    def get_gamma_defaults(self):
        return list(self._g_defaults)

    def gamma(self, m):
        """Gamma functions: first entry 0 (for {res}_width), then the
        normalised M_ab(m)/M_00(m₀) as in ck_matrix."""
        from .ck_matrix_model import _gamma_functions
        gf = _gamma_functions(m, self.x_table, self.M_table,
                              self.n_ck, self._gamma_scale)
        return [np.zeros_like(m, dtype=complex)] + gf

    # ── transform ────────────────────────────────────────────────

    def make_mass_width_transform(self):
        return _CKWidthTransform(
            self.name, self.order_names, self._g_names,
            self._ck0_r, self._ck0_i,
            self._gamma_m0, self.width0,
        )

    # ── get_bw_params ────────────────────────────────────────────

    def get_bw_params(self, params=None):
        """BW peak mass and width.

        With the normalisation, the total width at the pole is simply
        the ``{res}_width`` parameter value.  The peak mass is found
        by solving Re(m₀² - m² - i·m₀·g_bw) = 0.  Since the gamma
        functions are purely real for the CK matrix model, the peak
        is at m = m₀ and width_bw = {res}_width.
        """
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

        # The total running width at any m = Σ g_i · gamma_i(m)
        # gamma_0 = 0 (for the width param itself), others from M table
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
        width_bw = g_bw_re(mass_bw)
        return {"mass_bw": mass_bw, "width_bw": width_bw}

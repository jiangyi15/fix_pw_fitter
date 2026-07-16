"""CK matrix model with dispersive self-energy (``ck_matrix_disp_v2``).

Adds a real-part correction to the Breit-Wigner denominator from the
subtracted dispersion integral of the absorptive part.

The kernel formula is unchanged::

    bw_dom = m0^2 - m^2  -  i * m0 * sum(g_i * gamma_i(m))

But ``gamma_i(m)`` now returns a complex number where:

* **.real** = gamma_i(m) -- unchanged running width from partial_file
* **.imag** = RePi_i(m^2) - RePi_i(m0^2) -- dispersion correction

    where RePi is computed from the subtracted dispersion relation::

                      s     inf        Im Pi(s')
        RePi(s) =  --------  int  -----------------  ds'
                      pi    s_th    (s' - s) * s'

    with s_0 = 0, s_th = (3*m_pi)^2, and ImPi(sqrt(s')) = gamma_i(sqrt(s')).

This gives the propagator::

    g_bw   = sum g_i*gamma_i(m)  +  i * sum g_i*(RePi - RePi_0)
           = Gamma(m)  +  i * ReSigma(m^2)

    bw_dom = (m0^2 - m^2 + m0*ReSigma) - i*m0*Gamma(m)

At m = m_0: ReSigma = 0  -->  bw_dom = -i*m0*Gamma(m0)  -->  peak at m_0.
"""

import json
import numpy as np
from .base import BaseModel, register_model
from ampfit.param_constraint import Transform


# -- gamma table from M_ab matrix -----------------------------------

def _gamma_table(m_grid, M_table, n_ck, scale):
    """Compute gamma_i at ALL grid points.

    Returns (n_gamma, n_x) array -- the running-width shape values
    (same as ``ck_matrix_v2.gamma(m).real``).
    """
    n_x = len(m_grid)
    n_ch = n_ck + n_ck * (n_ck - 1)
    inv = 1.0 / scale if scale != 0 else 1.0
    out = np.zeros((n_ch, n_x))
    idx = 0
    for a in range(n_ck):
        out[idx, :] = M_table[:, a, a].real * inv
        idx += 1
        for b in range(a + 1, n_ck):
            Mab = M_table[:, a, b]
            out[idx, :]     = Mab.real * inv
            out[idx + 1, :] = -Mab.imag * inv
            idx += 2
    return out


def _gamma_at_m0(m_grid, M_table, n_ck, scale, m0):
    """gamma_i(m_0) for normalisation (same as ck_matrix_v2)."""
    inv = 1.0 / scale if scale != 0 else 1.0
    vals = []
    for a in range(n_ck):
        M_aa = float(np.interp(m0, m_grid, M_table[:, a, a].real))
        vals.append(M_aa * inv)
        for b in range(a + 1, n_ck):
            Mab = np.interp(m0, m_grid, M_table[:, a, b])
            vals.append(Mab.real * inv)
            vals.append(-Mab.imag * inv)
    return vals


def _reduced_spec(n_ck, name):
    """Gamma names -- same as ck_matrix_v2."""
    names = []
    for a in range(n_ck):
        names.append(f"{name}_re_{a}_{a}")
        for b in range(a + 1, n_ck):
            names.append(f"{name}_re_{a}_{b}")
            names.append(f"{name}_im_{a}_{b}")
    return names


def _raw_expanded(ck):
    """Raw outer-product expanded values from ck vector."""
    n = len(ck)
    raw = []
    for a in range(n):
        raw.append((ck[a] * np.conj(ck[a])).real)
        for b in range(a + 1, n):
            cab = ck[a] * np.conj(ck[b])
            raw.append(2.0 * cab.real)
            raw.append(2.0 * cab.imag)
    return np.array(raw, dtype=float)


# -- dispersion integral --------------------------------------------

def _compute_re_dispersion(m_grid, gamma_table, m_pi=0.1396):
    """Compute RePi_i(s_k) via exact analytical per-segment integration.

    gamma_i(m) is piecewise-linear in m (from np.interp).  The
    dispersion integral is done analytically for each segment.

    * Intervals not touching the pole: standard antiderivative.
    * Combined PV region [s_{k-1}, s_{k+1}]: regularised integrand
      [gamma(m) - gamma(m_k)]/((s'-s_k)*s') + gamma(m_k)*PV known term.

    The tail [s_max, inf] extends gamma = gamma_max constant.
    """
    m = m_grid
    s = m ** 2
    n_x = len(m)
    n_ch = gamma_table.shape[0]
    s_th = (3.0 * m_pi) ** 2
    m_th = np.sqrt(s_th)

    # Zero below threshold
    gamma_work = gamma_table.copy()
    gamma_work[:, m < m_th] = 0.0

    # Weight matrix: RePi_i(s_k) = sum_j W[k, j] * gamma_i(m_j)
    W = np.zeros((n_x, n_x))

    for k in range(n_x):
        sk = s[k]
        if sk < s_th:
            continue
        mk = m[k]

        # -- antiderivatives ------------------------------------
        def _F1(sj):
            if sk == 0 or sj == sk:
                return 0.0
            return np.log(abs((sj - sk) / sj)) / sk

        def _F2(sj):
            mj = np.sqrt(sj)
            if mk == 0 or mj == mk:
                return 0.0
            return np.log(abs((mj - mk) / (mj + mk))) / mk

        def _I1(sa, sb):
            return _F1(sb) - _F1(sa)
        def _I2(sa, sb):
            return _F2(sb) - _F2(sa)

        # -- 1. Standard intervals (no PV) ----------------------
        for j in range(n_x - 1):
            if j == k - 1 or j == k:
                continue
            m_a, m_b = m[j], m[j+1]
            s_a, s_b = s[j], s[j+1]
            dm = m_b - m_a
            I1 = _I1(s_a, s_b)
            I2 = _I2(s_a, s_b)
            pref = sk / np.pi
            w_a = pref * (m_b * I1 - I2) / dm
            w_b = pref * (I2 - m_a * I1) / dm
            W[k, j]     += w_a
            W[k, j + 1] += w_b

        # -- 2. Combined PV region [s_{k-1}, s_{k+1}] ----------
        s_left  = s[k - 1] if k > 0 else s_th
        s_right = s[k + 1] if k < n_x - 1 else s[k]
        PV_val = _I1(s_left, s_right)

        # right half [s_k, s_{k+1}]
        if k < n_x - 1:
            m_r = m[k + 1]
            dm_r = m_r - mk
            if mk != 0:
                ln_r = np.log(2.0 * m_r / (m_r + mk))
                w_r = (sk / np.pi) * (2.0 / mk) * ln_r / dm_r
                W[k, k]     -= w_r
                W[k, k + 1] += w_r

        # left half [s_{k-1}, s_k]
        if k > 0:
            m_l = m[k - 1]
            dm_l = mk - m_l
            if mk != 0:
                ln_l = np.log((m_l + mk) / (2.0 * m_l))
                w_l = (sk / np.pi) * (2.0 / mk) * ln_l / dm_l
                W[k, k - 1] -= w_l
                W[k, k]     += w_l

        # gamma_k * PV term
        W[k, k] += (sk / np.pi) * PV_val

        # -- 3. Tail [s_max, inf] with gamma = gamma_max -------
        I1_tail = -_F1(s[-1])  # F1(inf) = 0
        W[k, n_x - 1] += (sk / np.pi) * I1_tail

    # Apply weight matrix to each channel
    Re_Pi = W @ gamma_work.T
    return Re_Pi.T

# -- transform (identical to ck_matrix_v2) --------------------------

class _CKWidthDispTransform(Transform):
    """Transform {res}_width + ck -> normalised gamma values.

    Identical to ck_matrix_v2 -- normalisation uses the real-part (gamma)
    values at m_0.  The dispersion correction lives in gamma's .imag
    and does not affect the normalisation.
    """
    _has_inverse = False

    def __init__(self, name, order_names, gamma_names,
                 ck_r0, ck_i0, gamma_at_m0, width_name, default_width,
                 use_ref=False):
        self.width_name = width_name
        self.use_ref = use_ref
        if use_ref:
            in_names = [width_name]
        else:
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
        if self.use_ref:
            ck = self.ck_r0 * np.exp(1j * self.ck_i0)
        else:
            ck = np.zeros(self.n_ck, dtype=complex)
            for a in range(self.n_ck):
                r = d.get(self.order_names[a], self.ck_r0[a])
                theta = d.get(self.order_names[a].rstrip('r') + 'i', self.ck_i0[a])
                ck[a] = r * np.exp(1j * theta)
        raw = _raw_expanded(ck)
        N = np.dot(raw, self.gamma_at_m0)
        scale = width / N if N != 0 else 0.0
        idx = 0
        for a in range(self.n_ck):
            d[self.gamma_names[idx]] = raw[idx] * scale
            idx += 1
            for b in range(a + 1, self.n_ck):
                d[self.gamma_names[idx]] = raw[idx] * scale
                d[self.gamma_names[idx + 1]] = raw[idx + 1] * scale
                idx += 2
        return d

    def backward(self, grad_out, d_in=None):
        d = dict(d_in) if d_in else {}
        width = d.get(self.width_name, self._default_width)
        if self.use_ref:
            ck = self.ck_r0 * np.exp(1j * self.ck_i0)
        else:
            ck = np.zeros(self.n_ck, dtype=complex)
            for a in range(self.n_ck):
                r = d.get(self.order_names[a], self.ck_r0[a])
                theta = d.get(self.order_names[a].rstrip('r') + 'i', self.ck_i0[a])
                ck[a] = r * np.exp(1j * theta)
        raw = _raw_expanded(ck)
        N = np.dot(raw, self.gamma_at_m0)
        if N == 0:
            return {}
        scale = width / N
        B = 0.0
        for i, name in enumerate(self.gamma_names):
            B += grad_out.get(name, 0.0) * raw[i]
        grad = {}
        grad[self.width_name] = B / N
        if not self.use_ref:
            n = self.n_ck
            gammas = self.gamma_at_m0
            if not hasattr(self, '_raw_idx_map'):
                self._raw_idx_map = []
                ri = 0
                for aa in range(n):
                    self._raw_idx_map.append((aa, aa, 're'))
                    ri += 1
                    for bb in range(aa + 1, n):
                        self._raw_idx_map.append((aa, bb, 're'))
                        self._raw_idx_map.append((aa, bb, 'im'))
                        ri += 2
            r_signed = np.array(
                [d.get(self.order_names[a], self.ck_r0[a]) for a in range(n)])
            theta_vals = np.array(
                [d.get(self.order_names[a].rstrip('r') + 'i', self.ck_i0[a]) for a in range(n)])
            for a in range(n):
                dN_dr = 0.0; dN_dt = 0.0
                A_r = 0.0; A_t = 0.0
                ra = r_signed[a]
                ta = theta_vals[a]
                for ri, (aa, bb, rtype) in enumerate(self._raw_idx_map):
                    g_out = grad_out.get(self.gamma_names[ri], 0.0)
                    gamma_val = gammas[ri]
                    if aa == bb and aa == a:
                        dr = 2.0 * ra
                        dt = 0.0
                    elif aa == a and bb > a:
                        rb = r_signed[bb]; tb = theta_vals[bb]
                        dth = ta - tb
                        if rtype == 're':
                            dr = 2.0 * rb * np.cos(dth)
                            dt = -2.0 * ra * rb * np.sin(dth)
                        else:
                            dr = 2.0 * rb * np.sin(dth)
                            dt = 2.0 * ra * rb * np.cos(dth)
                    elif bb == a and aa < a:
                        rb = r_signed[aa]; tb = theta_vals[aa]
                        dth = ta - tb
                        if rtype == 're':
                            dr = 2.0 * rb * np.cos(dth)
                            dt = -2.0 * ra * rb * np.sin(dth)
                        else:
                            dr = -2.0 * rb * np.sin(dth)
                            dt = -2.0 * ra * rb * np.cos(dth)
                    else:
                        continue
                    dN_dr += dr * gamma_val
                    dN_dt += dt * gamma_val
                    A_r += g_out * dr
                    A_t += g_out * dt
                grad[self.order_names[a]] = scale * A_r - (scale / N) * dN_dr * B
                iname = self.order_names[a].rstrip('r') + 'i'
                grad[iname] = scale * A_t - (scale / N) * dN_dt * B
        return grad


# -- model ----------------------------------------------------------

@register_model("ck_matrix_disp_v2")
class CKMatrixDispModelV2(BaseModel):
    """Running width with dispersive real-part correction.

    gamma_i(m).real = gamma_i(m)                    -- unchanged running width
    gamma_i(m).imag = RePi_i(m^2) - RePi_i(m_0^2)  -- dispersion correction

    The dispersion integral::

                      s     inf       gamma_i(sqrt(s'))
        RePi_i(s) = ----- int   ---------------------  ds'
                      pi   s_th      (s' - s) * s'
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        # -- load files --------------------------------------------
        gamma_data = np.load(kwargs["gamma_file"])
        self.x_table = gamma_data[:, 0].copy()

        M = np.load(kwargs["partial_file"])
        self.M_table = np.asarray(M, dtype=complex)
        self.n_ck = self.M_table.shape[1]

        with open(kwargs["order_file"]) as f:
            self.order_names = json.load(f)
        if len(self.order_names) != self.n_ck:
            raise ValueError(
                f"order_file has {len(self.order_names)} names, "
                f"but M_table has n_ck={self.n_ck}")

        n_x = len(self.x_table)
        if self.M_table.shape[0] == n_x - 1:
            zero_pad = np.zeros((1, self.n_ck, self.n_ck), dtype=self.M_table.dtype)
            self.M_table = np.concatenate([zero_pad, self.M_table], axis=0)

        # -- config values -----------------------------------------
        self.m0 = float(kwargs.get("mass", 0.775))
        self.m_pi = float(kwargs.get("m_pi", 0.1396))

        # -- gamma scale so re_00(m_0) = 1 -------------------------
        self._gamma_scale = float(np.interp(self.m0, self.x_table,
                                              self.M_table[:, 0, 0].real))
        if self._gamma_scale == 0:
            self._gamma_scale = 1.0

        # -- gamma_i at all grid points (running width shape) ------
        self._gamma_table = _gamma_table(
            self.x_table, self.M_table, self.n_ck, self._gamma_scale)

        # -- gamma_i at m_0 (for normalisation) --------------------
        self._gamma_m0 = _gamma_at_m0(
            self.x_table, self.M_table, self.n_ck,
            self._gamma_scale, self.m0)

        # -- RePi from dispersion integral -------------------------
        self._Re_Pi_table = _compute_re_dispersion(
            self.x_table, self._gamma_table, m_pi=self.m_pi)

        # -- RePi at m_0 (for subtraction, so dispersion vanishes) -
        self._Re_Pi_at_m0 = np.array([
            float(np.interp(self.m0, self.x_table, self._Re_Pi_table[i]))
            for i in range(self._Re_Pi_table.shape[0])])

        # -- ck defaults -------------------------------------------
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

        # -- reference file mode -----------------------------------
        self.ref_file = kwargs.get("ref_file", None)
        self._ref_ck = None
        if self.ref_file:
            with open(self.ref_file) as f:
                ref_data = json.load(f)
            ref = ref_data.get("value") or ref_data
            ck_r = np.array([float(ref.get(n, self._ck0_r[i]))
                             for i, n in enumerate(self.order_names)])
            ck_i = np.array([float(ref.get(n.rstrip('r') + 'i', self._ck0_i[i]))
                             for i, n in enumerate(self.order_names)])
            self._ref_ck = ck_r * np.exp(1j * ck_i)

        # -- gamma names & defaults --------------------------------
        self._g_names = _reduced_spec(self.n_ck, self.name)
        ck0 = self._ck0_r * np.exp(1j * self._ck0_i)
        raw0 = _raw_expanded(ck0)
        N0 = np.dot(raw0, self._gamma_m0)
        self._g_defaults = [
            float(self.kwargs.get("width", 0.1)) * r / N0 if N0 != 0 else 0.0
            for r in raw0]

    # -- gamma interface -------------------------------------------

    def get_defaults(self):
        mass = float(self.kwargs.get("mass", 0.775))
        width = float(self.kwargs.get("width", 0.1))
        return {f"{self.name}_mass": mass, f"{self.name}_width": width}

    def get_gamma_count(self):
        return self.n_ck + self.n_ck * (self.n_ck - 1)

    def get_gamma_name(self):
        return list(self._g_names)

    def gamma(self, m):
        """Complex values at masses *m*.

        Returns list of complex arrays, one per channel::

            gamma_i(m).real =  gamma_i(m)             (running width, unchanged)
            gamma_i(m).imag = -(RePi_i(m^2) - RePi_0) (dispersion, same sign as width)
        """
        m_arr = np.asarray(m, dtype=float)
        out = []
        n_ch = self._Re_Pi_table.shape[0]
        for i in range(n_ch):
            re = np.interp(m_arr, self.x_table, self._gamma_table[i])
            re_pi = np.interp(m_arr, self.x_table, self._Re_Pi_table[i])
            im = -(re_pi - self._Re_Pi_at_m0[i])
            out.append(re + 1j * im)
        return out

    # -- transform ------------------------------------------------

    def make_mass_width_transform(self):
        width_name = f"{self.name}_width"
        if self._ref_ck is not None:
            ck_r0 = np.abs(self._ref_ck)
            ck_i0 = np.angle(self._ref_ck)
        else:
            ck_r0 = self._ck0_r
            ck_i0 = self._ck0_i
        return _CKWidthDispTransform(
            self.name, self.order_names, self._g_names,
            ck_r0, ck_i0,
            self._gamma_m0, width_name, float(self.kwargs.get("width", 0.1)),
            use_ref=self.ref_file is not None,
        )

    # -- get_bw_params --------------------------------------------

    def get_bw_params(self, params=None):
        """Breit-Wigner peak position from the complex gamma.

        The pole equation: f(m) = m0^2 - m^2 + m0 * sum_gamma_im
        With dispersion: sum_gamma_im = sum g_i*(RePi-RePi_0)
        At m = m_0: dispersion vanishes --> f(m_0) = 0 --> pole at m_0.
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

        if self._ref_ck is not None:
            ck = self._ref_ck
            raw_ref = _raw_expanded(ck)
            N_ref = np.dot(raw_ref, self._gamma_m0)
            width = _p(f"{self.name}_width", self.kwargs.get("width", 0.1))
            scale = width / N_ref if N_ref != 0 else 0.0
            g0_vals = [float(r * scale) for r in raw_ref]
        else:
            gamma_names = self.get_gamma_name()
            g0_vals = [_p(gamma_names[i], self._g_defaults[i])
                       for i in range(self.get_gamma_count())]

        total_width = _p(f"{self.name}_width",
                         float(self.kwargs.get("width", 0.1)))

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
            raise RuntimeError(
                f"get_bw_params: root finding failed for {self.name}")

        mass_bw = float(sol.root)
        width_bw = total_width
        return {"mass_bw": mass_bw, "width_bw": width_bw}

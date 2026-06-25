"""CK matrix model: Gamma(x) = c_a M_{ab}(x) c_b*.

The running width is a quadratic form in complex parameters c_a and
an energy-dependent Hermitian M matrix loaded from file::

    Gamma(m) = Σ_{a,b}  c_a · M_{ab}(m) · c_b*

Using Hermitian symmetry (M_ba = conj(M_ab), M_aa ∈ ℝ) the expanded
form reduces to independent terms::

    Gamma(m) = Σ_a  |c_a|² · M_aa(m)
             + Σ_{a<b}  2·Re(c_a c_b*) · Re(M_ab(m))
                        -2·Im(c_a c_b*) · Im(M_ab(m))

This gives the reduced gamma parameter set (no redundant im_aa = 0
or duplicated re_ba = re_ab pairs).

Config::

    particle:
      my_res:
        mass: 1.23
        model: ck_matrix
        gamma_file: /path/to/gamma.npy        # (n_x, 3)  — gamma[:,0] is x
        partial_file: /path/to/partial.npy    # (n_x, n_ck, n_ck) complex
        order_file: /path/to/order.json       # list of n_ck names
        ck: [1.0, 0.0, ...]                  # [r0, i0, r1, i1, ...]
"""

import json
import numpy as np
from .base import BaseModel, register_model
from ampfit.param_constraint import Transform


# ── helper: build reduced gamma-name lists ───────────────────────

def _reduced_gamma_spec(n_ck, name):
    """Build (gamma_names, gamma_defaults) for the reduced set.

    Returns:
        names:  list of gamma slot names
        re_idx: (a,b) pairs for the ``re`` entries
        im_idx: (a,b) pairs for the ``im`` entries
    """
    names = []
    re_pairs = []
    im_pairs = []
    for a in range(n_ck):
        # diagonal: re_aa only (im_aa = 0 always)
        names.append(f"{name}_re_{a}{a}")
        re_pairs.append((a, a))
        for b in range(a + 1, n_ck):
            # upper triangle: re_ab and im_ab
            names.append(f"{name}_re_{a}{b}")
            names.append(f"{name}_im_{a}{b}")
            re_pairs.append((a, b))
            im_pairs.append((a, b))
    return names, re_pairs, im_pairs


def _gamma_defaults(n_ck, ck, re_pairs, im_pairs):
    """Initial gamma values from ck (normalised by |c₀|²)."""
    re_00 = (ck[0] * np.conj(ck[0])).real
    inv = 1.0 / re_00 if re_00 != 0 else 0.0
    vals = []
    for a, b in re_pairs:
        cab = ck[a] * np.conj(ck[b])
        if a == b:
            if a == 0:
                vals.append(cab.real)          # |c₀|² (absolute)
            else:
                vals.append(cab.real * inv)    # |c_a|² / |c₀|²
        else:
            vals.append(2.0 * cab.real * inv)  # 2·Re(c_a c_b*) / |c₀|²
    for a, b in im_pairs:
        cab = ck[a] * np.conj(ck[b])
        vals.append(2.0 * cab.imag * inv)      # 2·Im(c_a c_b*) / |c₀|²
    return vals


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


# ── Transform: ck → reduced gamma values ─────────────────────────

class _CKReducedTransform(Transform):
    """Transform re_00 + ck → full reduced gamma set.

    ``re_00`` is the global scale (pass-through).  All other gamma
    values are::

        re_00 × raw_ck_value / |c₀|²

    so that the whole running width scales linearly with re_00.
    """

    _has_inverse = False

    def __init__(self, name, order_names, gamma_names,
                 ck_r0, ck_i0, re_00_name):
        re_00_name_in = re_00_name
        # Inputs: re_00 + ck real parts + ck imag parts
        in_names = [re_00_name_in] + list(order_names) + \
                   [n.rstrip('r') + 'i' for n in order_names]
        # Outputs: all gamma names
        super().__init__(input_names=in_names, output_names=gamma_names)
        self.name = name
        self.order_names = list(order_names)
        self.n_ck = len(order_names)
        self.gamma_names = list(gamma_names)
        self.re_00_name = re_00_name_in
        self.ck_r0 = np.array(ck_r0, dtype=float)
        self.ck_i0 = np.array(ck_i0, dtype=float)

    def forward(self, d):
        d = dict(d)

        # Read re_00 (standalone global scale)
        scale = d.get(self.re_00_name, 1.0)

        # Read ck
        ck = np.zeros(self.n_ck, dtype=complex)
        for a in range(self.n_ck):
            r = d.get(self.order_names[a], self.ck_r0[a])
            i = d.get(self.order_names[a].rstrip('r') + 'i', self.ck_i0[a])
            ck[a] = r + 1j * i

        # Normalisation denominator = |c₀|²
        norm = (ck[0] * np.conj(ck[0])).real
        inv_norm = 1.0 / norm if norm != 0 else 0.0

        # Write all gamma values
        idx = 0
        for a in range(self.n_ck):
            # re_aa (diagonal)
            if a == 0:
                d[self.gamma_names[idx]] = scale              # re_00 pass-through
            else:
                d[self.gamma_names[idx]] = scale * (ck[a] * np.conj(ck[a])).real * inv_norm
            idx += 1
            for b in range(a + 1, self.n_ck):
                cab = ck[a] * np.conj(ck[b])
                d[self.gamma_names[idx]]     = scale * 2.0 * cab.real * inv_norm
                d[self.gamma_names[idx + 1]] = scale * 2.0 * cab.imag * inv_norm
                idx += 2
        return d


# ── Model class ──────────────────────────────────────────────────

@register_model("ck_matrix")
class CKMatrixModel(BaseModel):
    """Running width from a CK outer product with M(x) matrix.

    Files
    -----
    * ``gamma_file`` — (n_x, 3)  ``gamma[:, 0]`` is the energy grid
    * ``partial_file`` — (n_x, n_ck, n_ck) complex Hermitian matrix
    * ``order_file`` — JSON list of ``n_ck`` parameter names

    Config params
    -------------
    * ``ck`` — initial complex params ``[r0, i0, r1, i1, …]``

    Gamma parameters (reduced set using Hermitian symmetry)::

        diag:  re_00 (= |c₀|² absolute, global scale)
               re_aa (= |c_a|²/|c₀|² for a>0)
        a<b:   re_ab (= 2·Re(c_a c_b*)/|c₀|²)
               im_ab (= 2·Im(c_a c_b*)/|c₀|²)
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        # ── load files ───────────────────────────────────────────
        gamma_data = np.load(kwargs["gamma_file"])              # (n_x, 3)
        self.x_table = gamma_data[:, 0].copy()                  # x values

        M = np.load(kwargs["partial_file"])                     # (n_x, n_ck, n_ck)
        self.M_table = np.asarray(M, dtype=complex)
        self.n_ck = self.M_table.shape[1]

        with open(kwargs["order_file"]) as f:
            self.order_names = json.load(f)

        if len(self.order_names) != self.n_ck:
            raise ValueError(
                f"order_file has {len(self.order_names)} names "
                f"but M_table has n_ck={self.n_ck}"
            )

        # Pad M_table if shorter than x_table (insert zeros at start)
        n_x = len(self.x_table)
        if self.M_table.shape[0] == n_x - 1:
            zero_pad = np.zeros((1, self.n_ck, self.n_ck), dtype=self.M_table.dtype)
            self.M_table = np.concatenate([zero_pad, self.M_table], axis=0)

        # ── parse ck from config (optional, default: c₀=√width) ──
        ck_raw = kwargs.get("ck", None)
        if ck_raw is not None:
            n_parts = 2 * self.n_ck
            if len(ck_raw) != n_parts:
                raise ValueError(
                    f"ck expects {n_parts} values [r0,i0,…] "
                    f"for n_ck={self.n_ck}, got {len(ck_raw)}"
                )
            self._ck0_r = np.array(ck_raw[0::2], dtype=float)
            self._ck0_i = np.array(ck_raw[1::2], dtype=float)
        else:
            # Default: scale from config width, only first channel active
            width = float(kwargs.get("width", 0.1))
            c0 = np.sqrt(width)
            self._ck0_r = np.zeros(self.n_ck, dtype=float)
            self._ck0_r[0] = c0
            self._ck0_i = np.zeros(self.n_ck, dtype=float)
        self._ck0 = self._ck0_r + 1j * self._ck0_i

        # ── global scale so re_00(m₀) = 1 ────────────────────────
        m0 = float(kwargs.get("mass", 0.775))
        self._gamma_scale = float(np.interp(m0, self.x_table,
                                              self.M_table[:, 0, 0].real))
        if self._gamma_scale == 0:
            self._gamma_scale = 1.0  # fallback

        # ── build reduced gamma spec ─────────────────────────────
        self._g_names, self._re_pairs, self._im_pairs = \
            _reduced_gamma_spec(self.n_ck, self.name)
        self._g_defaults = _gamma_defaults(
            self.n_ck, self._ck0, self._re_pairs, self._im_pairs)

    # ── gamma parameter interface ────────────────────────────────

    def get_gamma_count(self):
        return len(self._g_names)

    def get_gamma_name(self):
        return list(self._g_names)

    def get_gamma_defaults(self):
        return list(self._g_defaults)

    def gamma(self, m):
        """Return reduced gamma functions::

            diag:  M_aa(m)               (real)
            a<b:   Re(M_ab(m))           (real, for re_ab)
                   -Im(M_ab(m))          (real, for im_ab)

        Each is purely real (as complex with 0 imag part).
        The total running width matches ``c_a M_ab c_b*`` via::

            Σ_a |c_a|²·M_aa
            + Σ_{a<b} 2·Re(c_a c_b*)·Re(M_ab) - 2·Im(c_a c_b*)·Im(M_ab)
        """
        return _gamma_functions(m, self.x_table, self.M_table, self.n_ck,
                                 self._gamma_scale)

    # ── transform ────────────────────────────────────────────────

    def make_mass_width_transform(self):
        """Transform ck → reduced gamma set, normalised by |c₀|²."""
        return _CKReducedTransform(
            self.name, self.order_names, self._g_names,
            self._ck0_r, self._ck0_i,
            self._g_names[0],
        )

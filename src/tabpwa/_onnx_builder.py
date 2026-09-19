#!/usr/bin/env python3
"""
Build a PWA ONNX model directly (no PyTorch dependency).

The graph is constructed node-by-node using the ``onnx`` library.
Complex numbers are split into (real, imag) float pairs throughout.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def float_type():
    return TensorProto.FLOAT


def int64():
    return TensorProto.INT64


class PWAONNXBuilder:
    """Builds a PWA forward-computation ONNX graph.

    All complex values are stored as pairs ``(name_r, name_i)``.
    The model takes event data + parameters and returns ``(Q, P)``.
    """

    def __init__(self, kernel_config):
        kc = kernel_config

        self.n_wave = kc["matrix_angle"].shape[1]
        self.n_res = kc["bw_order"].size // self.n_wave
        self.n_decay = kc["fl_order"].size // self.n_wave
        self.n_unique_bw = len(kc["m0_index"])
        self.n_gamma_rows = len(kc["g0_index"])
        self.n_angle = kc["angle_k"].shape[0]

        # Store config constants
        self._consts = {}
        self._nodes = []
        self._value_info = []
        self._graph_name = "pwa_forward"
        self._counter = 0
        self.N = None  # set in build()
        self._name("_init")  # seed counter
        self._embed("m0_index", kc["m0_index"].astype(np.int64))
        self._embed("g0_index", kc["g0_index"].astype(np.int64))
        self._embed("fl_type", kc["fl_type"].astype(np.int64))
        self._embed("mass_index", kc["mass_index"].astype(np.int64))
        self._embed("g0_mass_index", kc["g0_mass_index"].astype(np.int64))
        self._embed("fl_q_index", kc["fl_q_index"].astype(np.int64))
        self._embed("bw_order", kc["bw_order"].astype(np.int64))
        self._embed("fl_order", kc["fl_order"].astype(np.int64))
        self._embed("angle_index", kc["angle_index"].astype(np.int64))
        self._embed("matrix_angle_real", np.real(kc["matrix_angle"]).astype(np.float32))
        self._embed("matrix_angle_imag", np.imag(kc["matrix_angle"]).astype(np.float32))
        self._embed("matrix_gamma", kc["matrix_gamma"].astype(np.float32))
        self._embed("gamma_table_real", np.real(kc["gamma_table"]).astype(np.float32))
        self._embed("gamma_table_imag", np.imag(kc["gamma_table"]).astype(np.float32))
        self._embed("fl_table", kc["fl_table"].astype(np.float32))
        self._embed("angle_k", kc["angle_k"])
        self._embed("angle_b", kc["angle_b"].astype(np.float32))

        # Interpolation configs
        self.gamma_min = float(kc["gamma_min"])
        self.gamma_delta = float(kc["gamma_delta"])
        self.fl_min = float(kc["fl_min"])
        self.fl_delta = float(kc["fl_delta"])
        self.kc = kc

        # Gather shorthands
        self._n_bins_gamma = kc["gamma_table"].shape[-1]
        self._n_bins_fl = kc["fl_table"].shape[-1]

    # ── helpers ─────────────────────────────────────────────────

    def _name(self, prefix):
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def _embed(self, name, arr):
        t = numpy_helper.from_array(np.asarray(arr), name=name)
        self._consts[name] = t

    def _input(self, name, shape, dtype=TensorProto.FLOAT):
        vi = helper.make_tensor_value_info(name, dtype, shape)
        self._value_info.append(vi)
        return name

    def _node(self, op_type, inputs, outputs=None, **attrs):
        if outputs is None:
            outputs = [self._name(op_type.lower())]
        elif isinstance(outputs, str):
            outputs = [outputs]
        named = attrs.pop('named_as', None)
        if named:
            # Use identity to create a named alias
            aid = self._name("alias")
            self._nodes.append(helper.make_node("Identity", [outputs[0]], [named]))
            return named
        if outputs is None:
            outputs = [self._name(op_type.lower())]
        elif isinstance(outputs, str):
            outputs = [outputs]
        # Opset 11: Squeeze/Unsqueeze/Reduce* use axes as attributes.
        # Only Slice uses axes as tensor input[3] - handled by _slice_axis.
        node = helper.make_node(op_type, inputs, outputs, **attrs)
        self._nodes.append(node)
        return outputs[0]

    def _c(self, name):
        return name

    def _scalar(self, value, dtype=TensorProto.FLOAT):
        name = self._name(f"c{value}")
        if dtype == TensorProto.FLOAT:
            self._embed(name, np.array([value], dtype=np.float32))
        else:
            self._embed(name, np.array([value], dtype=np.int64))
        # Wrap in Identity to prevent ATC from detecting this as a scalar
        # constant in Mul ops (which triggers unsupported Muls conversion).
        id_name = self._name("scid")
        self._nodes.append(helper.make_node("Identity", [name], [id_name]))
        return id_name

    # ── complex arithmetic on (r, i) pairs ──────────────────────

    def _complex_mul(self, ar, ai, br, bi, prefix):
        """(a_r + j*a_i) * (b_r + j*b_i) → (r, i)."""
        m1 = self._node("Mul", [ar, br])
        m2 = self._node("Mul", [ai, bi])
        m3 = self._node("Mul", [ar, bi])
        m4 = self._node("Mul", [ai, br])
        r = self._node("Sub", [m1, m2])
        i = self._node("Add", [m3, m4])
        return r, i

    def _complex_mul_real(self, ar, ai, b, prefix):
        """(a_r + j*a_i) * b_real → (r, i)."""
        r = self._node("Mul", [ar, b])
        i = self._node("Mul", [ai, b])
        return r, i

    def _complex_abs2(self, ar, ai, prefix):
        """|a|² = a_r² + a_i²."""
        sr = self._node("Mul", [ar, ar])
        si = self._node("Mul", [ai, ai])
        return self._node("Add", [sr, si])

    def _complex_div(self, ar, ai, br, bi, prefix):
        """(a_r + j*a_i) / (b_r + j*b_i) = (a+jb)(c-jd)/(c²+d²)."""
        denom = self._complex_abs2(br, bi, "den")
        # num = (a*c + b*d) + j(b*c - a*d)
        ac = self._node("Mul", [ar, br])
        bd = self._node("Mul", [ai, bi])
        bc = self._node("Mul", [ai, br])
        ad = self._node("Mul", [ar, bi])
        num_r = self._node("Add", [ac, bd])
        num_i = self._node("Sub", [bc, ad])
        r = self._node("Div", [num_r, denom])
        i = self._node("Div", [num_i, denom])
        return r, i

    # ── slice helper (handles axes tensor for opset 13) ─────────

    def _slice_axis(self, data, axis, start, end, prefix):
        """Slice along one axis: data[:,.., start:end, ..,:].
        Uses opset-9 compatible approach via Gather.
        For axis=2 (last dim): gather indices range(start, end) along axis.
        For axis=1: gather indices range(start, end) along axis.
        """
        indices = np.arange(start, end, dtype=np.int64)
        idx_name = self._name("idx")
        self._embed(idx_name, indices)
        o = self._node("Gather", [data, idx_name], axis=axis)
        # Squeeze the gathered dim if it's size 1
        if end - start == 1:
            o = self._node("Squeeze", [o], axes=[axis])
        return o

    # ── gather/indexing helpers ─────────────────────────────────

    def _gather_1d(self, data_1d, indices_1d, prefix):
        """Gather elements from 1D tensor by 1D indices."""
        return self._node("Gather", [data_1d, indices_1d], axis=0)

    def _gather_cols(self, data_2d, col_indices, prefix):
        """Gather columns from 2D tensor (index_select on last axis).
        Uses Gather with axis=1 instead of GatherElements to avoid
        the rank-matching requirement.
        """
        return self._node("Gather", [data_2d, col_indices], axis=1)

    def _interp(self, table, types, x, xmin, xdelta, n_bins, n_cols, prefix):
        """Linear interpolation: table[types, x] → values.

        table: (n_types, n_bins) float
        types: (n_rows,) int64 — which type for each row
        x: (N, n_rows) float
        n_cols: number of columns (must match types.shape[0])
        """
        c_min = self._scalar(xmin)
        c_delta = self._scalar(xdelta)
        diff = self._node("Sub", [x, c_min])
        xbin_f = self._node("Div", [diff, c_delta])
        xbin_floor = self._node("Floor", [xbin_f])

        # Clamp xbin to [0, n_bins-2] via Min/Max on float
        xbin_float = self._node("Cast", [xbin_floor], to=TensorProto.FLOAT)
        zero_f = self._scalar(0.0)
        nbm2_f = self._scalar(float(n_bins - 2))
        xbin_max = self._node("Max", [xbin_float, zero_f])
        xbin_clip_f = self._node("Min", [xbin_max, nbm2_f])
        xbin_clip = self._node("Cast", [xbin_clip_f], to=TensorProto.INT64)

        delta = self._node("Sub", [xbin_f, xbin_floor])

        # Reshape table to 1D for gather
        table_flat = self._node("Reshape", [table, self._shape_1d(-1)])
        n_bins_i = self._scalar(n_bins, TensorProto.INT64)
        types_off = self._node("Mul", [types, n_bins_i])
        # Expand types_off to match batch dim
        types_off_2d = self._node("Reshape", [types_off, self._shape_1d(-1)])
        types_off_3d = self._node("Unsqueeze", [types_off_2d], axes=[0])
        lin_idx = self._node("Add", [xbin_clip, types_off_3d])

        # Gather left and right
        lin_idx_flat = self._node("Reshape", [lin_idx, self._shape_1d(-1)])
        left = self._node("Gather", [table_flat, lin_idx_flat], axis=0)
        right_idx = self._node("Add", [lin_idx_flat, self._scalar(1, TensorProto.INT64)])
        right = self._node("Gather", [table_flat, right_idx], axis=0)

        # Reshape back to (N, n_cols)
        left_r = self._node("Reshape", [left, self._shape_2d(-1, n_cols)])
        right_r = self._node("Reshape", [right, self._shape_2d(-1, n_cols)])

        return self._node("Add", [
            left_r, self._node("Mul", [self._node("Sub", [right_r, left_r]), delta])
        ])

    def _shape_1d(self, d0):
        name = self._name("sh1d")
        self._embed(name, np.array([d0], dtype=np.int64))
        return name

    def _shape_2d(self, d0, d1):
        name = self._name("sh2d")
        self._embed(name, np.array([d0, d1], dtype=np.int64))
        return name

    def _shape_3d(self, d0, d1, d2):
        name = self._name("sh3d")
        self._embed(name, np.array([d0, d1, d2], dtype=np.int64))
        return name

    # ── build the full graph ────────────────────────────────────

    def build(self, batch_size=4, norm_model=False):
        """Build the ONNX graph.

        Parameters
        ----------
        batch_size : int
            Fixed batch dimension for event-level inputs.  The backend
            splits larger datasets into chunks of this size and sums
            the results (batch summation).
        norm_model : bool
            If True, build a norm-only model (Q = sum(P*weight)) suitable
            for computing the normalisation integral.  The forward model (default)
            computes the full NLL with background and norm scaling.
        """
        # Reset per-build state but KEEP constants (embedded once in __init__).
        # The counter keeps incrementing to guarantee unique node names across
        # multiple build() calls.
        self._nodes = []
        self._value_info = []

        N = batch_size
        n_wave = self.n_wave
        n_res = self.n_res
        n_decay = self.n_decay
        n_unique_bw = self.n_unique_bw
        n_gamma_rows = self.n_gamma_rows
        n_angle = self.n_angle

        # ──── inputs ────
        ck_r = self._input("ck_real", [n_wave])
        ck_i = self._input("ck_imag", [n_wave])
        n_m0_unique = len(np.unique(self.kc["m0_index"]))
        m0 = self._input("m0", [n_m0_unique])
        n_g0_unique = len(np.unique(self.kc["g0_index"]))
        g0 = self._input("g0", [n_g0_unique])
        mass = self._input("mass", [N, 48])
        q = self._input("q", [N, 72])
        angle = self._input("angle", [N, 24, 3])
        frac = self._input("frac", [N])
        time = self._input("time", [N])
        weight = self._input("weight", [N])
        if not norm_model:
            bkg = self._input("bkg", [N])
            norm = self._input("norm", [1])
        Gamma = self._input("Gamma", [1])
        Delta_Gamma = self._input("Delta_Gamma", [1])
        Delta_m = self._input("Delta_m", [1])
        A_prod = self._input("A_prod", [1])
        poq_rho = self._input("poq_rho", [1])
        pop_phi = self._input("pop_phi", [1])

        # ──── 1. Gamma interpolation → g_bw ────
        # g0_all = g0[g0_index]
        gather_g0 = self._gather_1d(g0, "g0_index", "g0_all")
        g0_all_2d = self._node("Unsqueeze", [gather_g0], axes=[0])
        # g0_mass from mass by g0_mass_index
        g0_mass = self._gather_cols(mass, "g0_mass_index", "g0_mass")
        # interpolate gamma table
        n_gamma_rows = len(self.kc["g0_index"])
        g_interp_r = self._interp("gamma_table_real", "g0_index", g0_mass,
                                   self.gamma_min, self.gamma_delta,
                                   self._n_bins_gamma, n_gamma_rows, "g_interp_r")
        g_interp_i = self._interp("gamma_table_imag", "g0_index", g0_mass,
                                   self.gamma_min, self.gamma_delta,
                                   self._n_bins_gamma, n_gamma_rows, "g_interp_i")
        # g = g0_all * g_interp  (complex)
        g_r, g_i = self._complex_mul_real(g_interp_r, g_interp_i, g0_all_2d, "g")
        # g_bw = g @ matrix_gamma
        g_bw_r = self._node("MatMul", [g_r, "matrix_gamma"])
        g_bw_i = self._node("MatMul", [g_i, "matrix_gamma"])

        # ──── 2. BW propagators ────
        m0_all_2d = self._node("Unsqueeze", [self._gather_1d(m0, "m0_index", "m0_all")], axes=[0])
        m0m = self._gather_cols(mass, "mass_index", "m0m")
        m0_sq = self._node("Mul", [m0_all_2d, m0_all_2d])
        m_sq = self._node("Mul", [m0m, m0m])
        # bw_dom = m0² - m² - 1j * m0 * g_bw
        # Real: m0² - m² + m0 * g_bw_i   (since -1j*(r+ji) = -j*r + i)
        # Imag: -m0 * g_bw_r
        bw_dom_r = self._node("Sub", [m0_sq, m_sq])
        gbw_i_m0 = self._node("Mul", [m0_all_2d, g_bw_i])
        bw_dom_r = self._node("Add", [bw_dom_r, gbw_i_m0])
        gbw_r_m0 = self._node("Mul", [m0_all_2d, g_bw_r])
        bw_dom_i = self._node("Neg", [gbw_r_m0])

        # ──── 3. Product over resonances ────
        # Gather by bw_order then reshape (N, n_wave, n_res) → prod over last dim
        bw_all_r = self._gather_cols(bw_dom_r, "bw_order", "bw_all_r")
        bw_all_i = self._gather_cols(bw_dom_i, "bw_order", "bw_all_i")
        bw_3d_r = self._node("Reshape", [bw_all_r, self._shape_3d(N, n_wave, n_res)])
        bw_3d_i = self._node("Reshape", [bw_all_i, self._shape_3d(N, n_wave, n_res)])

        # Integer constants for slicing
        c_s0 = self._scalar(0, TensorProto.INT64)
        c_1 = self._scalar(1, TensorProto.INT64)

        if n_res == 1:
            bw_p_r = self._node("Squeeze", [bw_3d_r], axes=[2])
            bw_p_i = self._node("Squeeze", [bw_3d_i], axes=[2])
        else:
            # Complex product: multiply sequentially over n_res
            p_r = self._slice_axis(bw_3d_r, 2, 0, 1, "p0")
            p_i = self._slice_axis(bw_3d_i, 2, 0, 1, "p0i")
            for k in range(1, n_res):
                f_r = self._slice_axis(bw_3d_r, 2, k, k + 1, f"f{k}r")
                f_i = self._slice_axis(bw_3d_i, 2, k, k + 1, f"f{k}i")
                p_r, p_i = self._complex_mul(p_r, p_i, f_r, f_i, f"bw_p{k}")
            bw_p_r, bw_p_i = p_r, p_i

        # ──── 4. FL factors ────
        fl_q = self._gather_cols(q, "fl_q_index", "fl_q")
        n_fl_cols = len(self.kc["fl_type"])
        fl = self._interp("fl_table", "fl_type", fl_q,
                          self.fl_min, self.fl_delta, self._n_bins_fl, n_fl_cols, "fl")
        fl_all = self._gather_cols(fl, "fl_order", "fl_all")
        fl_3d = self._node("Reshape", [fl_all, self._shape_3d(N, n_wave, n_decay)])
        if n_decay == 1:
            fl_p = self._node("Squeeze", [fl_3d], axes=[2])
        else:
            fl_p = self._node("ReduceProd", [fl_3d], axes=[2], keepdims=0)

        # ──── 5. Angular factors ────
        # numpy: ang = np.take(angle, angle_index, axis=-2)
        # ONNX Gather with axis=1 on (N,24,3) with 1D index (336,) → (N,336,3)
        ang = self._node("Gather", [angle, "angle_index"], axis=1)

        # ka = cos(ang * angle_k + angle_b).prod(axis=-1)
        angle_k_f = self._node("Cast", ["angle_k"], to=TensorProto.FLOAT)
        angle_k_3d = self._node("Unsqueeze", [angle_k_f], axes=[0])
        ang_k_mul = self._node("Mul", [ang, angle_k_3d])
        angle_b_3d = self._node("Unsqueeze", ["angle_b"], axes=[0])
        ang_kb = self._node("Add", [ang_k_mul, angle_b_3d])
        cos_a = self._node("Cos", [ang_kb])
        ka = self._node("ReduceProd", [cos_a], axes=[2], keepdims=0)

        # fa = ka @ matrix_angle (complex matmul)
        fa_r = self._node("MatMul", [ka, "matrix_angle_real"])
        fa_i = self._node("MatMul", [ka, "matrix_angle_imag"])

        # ──── 6. Amplitude ────
        # inv_bw = 1/bw_p (complex division)
        one_r = self._scalar(1.0)
        one_i = self._scalar(0.0)
        # Expand one_r/one_i to (1, n_wave) for proper 2D broadcasting
        one_r_2d = self._node("Expand", [one_r, self._shape_2d(1, n_wave)])
        one_i_2d = self._node("Expand", [one_i, self._shape_2d(1, n_wave)])
        inv_r, inv_i = self._complex_div(one_r_2d, one_i_2d, bw_p_r, bw_p_i, "inv")
        # common = inv_bw * fa * fl_p
        ca_r, ca_i = self._complex_mul_real(inv_r, inv_i, fl_p, "c_fl")
        cf_r, cf_i = self._complex_mul(ca_r, ca_i, fa_r, fa_i, "cf")
        # a = ck * common
        ck_r_2d = self._node("Unsqueeze", [ck_r], axes=[0])
        ck_i_2d = self._node("Unsqueeze", [ck_i], axes=[0])
        a_r, a_i = self._complex_mul(ck_r_2d, ck_i_2d, cf_r, cf_i, "a")

        # ──── 7. Split CP ────
        n_half = n_wave // 2
        c_nhalf = self._scalar(n_half, TensorProto.INT64)
        ap_r = self._node("ReduceSum",
                          [self._slice_axis(a_r, 1, 0, n_half, "ap_r")],
                          axes=[1], keepdims=0)
        ap_i = self._node("ReduceSum",
                          [self._slice_axis(a_i, 1, 0, n_half, "ap_i")],
                          axes=[1], keepdims=0)
        am_r = self._node("ReduceSum",
                          [self._slice_axis(a_r, 1, n_half, n_wave, "am_r")],
                          axes=[1], keepdims=0)
        am_i = self._node("ReduceSum",
                          [self._slice_axis(a_i, 1, n_half, n_wave, "am_i")],
                          axes=[1], keepdims=0)

        # ──── 8. Time evolution ────
        # poq = poq_rho * exp(j*pop_phi)
        cos_phi = self._node("Cos", [pop_phi])
        sin_phi = self._node("Sin", [pop_phi])
        poq_r = self._node("Mul", [poq_rho, cos_phi])
        poq_i = self._node("Mul", [poq_rho, sin_phi])

        half = self._scalar(0.5)
        neg_half = self._scalar(-0.5)
        dhalf = self._scalar(0.25)

        # Compute eL and eH
        # eL = exp(-j*t*(-Δm/2 - j*(Γ + ΔΓ/2)/2))
        # Let: a = -Δm/2, b = -(Γ + ΔΓ/2)/2
        # Then: -j*t*(a + j*b) = -j*t*a + t*b
        # So: eL = exp(t*b) * (cos(t*a) - j*sin(t*a))
        t_t = time  # rename

        # aL = -Δm/2,  bL = -(Γ + ΔΓ/2)/2
        # aH = +Δm/2,  bH = -(Γ - ΔΓ/2)/2
        aL = self._node("Mul", [neg_half, Delta_m])
        aH = self._node("Mul", [half, Delta_m])
        dg2 = self._node("Mul", [half, Delta_Gamma])
        bL = self._node("Mul", [neg_half, self._node("Add", [Gamma, dg2])])
        bH = self._node("Mul", [neg_half, self._node("Sub", [Gamma, dg2])])

        def _exp_minus_j(a, b):
            """exp(-j*t*(a + j*b)) = exp(t*b) * (cos(t*a) - j*sin(t*a))."""
            ta = self._node("Mul", [t_t, a])
            tb = self._node("Mul", [t_t, b])
            exp_tb = self._node("Exp", [tb])
            cos_ta = self._node("Cos", [ta])
            sin_ta = self._node("Sin", [ta])
            return (self._node("Mul", [exp_tb, cos_ta]),
                    self._node("Neg", [self._node("Mul", [exp_tb, sin_ta])]))

        eL_r, eL_i = _exp_minus_j(aL, bL)
        eH_r, eH_i = _exp_minus_j(aH, bH)

        gp_r = self._node("Mul", [self._node("Add", [eL_r, eH_r]), half])
        gp_i = self._node("Mul", [self._node("Add", [eL_i, eH_i]), half])
        gm_r = self._node("Mul", [self._node("Sub", [eL_r, eH_r]), half])
        gm_i = self._node("Mul", [self._node("Sub", [eL_i, eH_i]), half])

        # pap = gp*ap + gm*poq*am
        gp_ap_r, gp_ap_i = self._complex_mul(gp_r, gp_i, ap_r, ap_i, "gp_ap")
        gm_poq_r, gm_poq_i = self._complex_mul(gm_r, gm_i, poq_r, poq_i, "gm_poq")
        gm_poq_am_r, gm_poq_am_i = self._complex_mul(gm_poq_r, gm_poq_i, am_r, am_i, "gm_poq_am")
        pap_r = self._node("Add", [gp_ap_r, gm_poq_am_r])
        pap_i = self._node("Add", [gp_ap_i, gm_poq_am_i])

        # pam = (gm/poq)*ap + gp*am
        gm_poq_r, gm_poq_i2 = self._complex_div(gm_r, gm_i, poq_r, poq_i, "gm_d_poq")
        gmdp_ap_r, gmdp_ap_i = self._complex_mul(gm_poq_r, gm_poq_i2, ap_r, ap_i, "gmdp_ap")
        gp_am_r, gp_am_i = self._complex_mul(gp_r, gp_i, am_r, am_i, "gp_am")
        pam_r = self._node("Add", [gmdp_ap_r, gp_am_r])
        pam_i = self._node("Add", [gmdp_ap_i, gp_am_i])

        # Probabilities
        pb = self._complex_abs2(pap_r, pap_i, "pb")
        pbbar = self._complex_abs2(pam_r, pam_i, "pbbar")

        c1 = self._scalar(1.0)
        term1 = self._node("Mul", [
            self._node("Mul", [frac, pb]),
            self._node("Sub", [c1, A_prod])
        ])
        term2 = self._node("Mul", [
            self._node("Mul", [self._node("Sub", [c1, frac]), pbbar]),
            self._node("Add", [c1, A_prod])
        ])
        P = self._node("Add", [term1, term2])

        if norm_model:
            # Norm model: Q = sum(P * weight)
            pw = self._node("Mul", [P, weight])
            Q = self._node("ReduceSum", [pw], axes=[0], keepdims=0, outputs="Q")
            self._node("Identity", [P], outputs="P")
            # dQ_dP = weight  (Q = sum(P*weight) → dQ/dP = weight)
            dQ_dP = weight
        else:
            # NLL
            eps = self._scalar(1e-30)
            Pnorm = self._node("Div", [P, norm])
            Pbkg = self._node("Add", [Pnorm, bkg])
            clamped = self._node("Max", [Pbkg, eps])
            logP = self._node("Log", [clamped])
            wlog = self._node("Mul", [weight, logP])
            wsum = self._node("ReduceSum", [wlog], axes=[0], keepdims=0)
            Q = self._node("Neg", [wsum], outputs="Q")
            self._node("Identity", [P], outputs="P")

            # dQ_dP = -weight / (P + bkg * norm)
            bkg_norm = self._node("Mul", [bkg, norm])
            P_plus_bn = self._node("Add", [P, bkg_norm])
            dQ_dP = self._node("Neg", [self._node("Div", [weight, P_plus_bn])])

        # dP/dpb, dP/dpbbar, dP/dAp
        dP_dpb = self._node("Mul", [frac, self._node("Sub", [c1, A_prod])])
        dP_dpbbar = self._node("Mul", [self._node("Sub", [c1, frac]), self._node("Add", [c1, A_prod])])
        neg_frac = self._node("Neg", [frac])
        dP_dAp = self._node("Add", [
            self._node("Mul", [neg_frac, pb]),
            self._node("Mul", [self._node("Sub", [c1, frac]), pbbar]),
        ])

        dQ_dpb = self._node("Mul", [dQ_dP, dP_dpb])
        dQ_dpbbar = self._node("Mul", [dQ_dP, dP_dpbbar])
        dQ_dAp = self._node("ReduceSum", [self._node("Mul", [dQ_dP, dP_dAp])], axes=[0], keepdims=0)

        # Wirtinger gradients for |pap|² and |pam|²
        # d_pb_dpap = conj(pap)  — Wirtinger: ∂|pap|²/∂pap = pap̄
        # d_pbbar_dpam = conj(pam)
        conj_pap_r = pap_r
        conj_pap_i = self._node("Neg", [pap_i])
        conj_pam_r = pam_r
        conj_pam_i = self._node("Neg", [pam_i])

        # dQ/dpap (Wirtinger) = dQ/dpb * d_pb_dpap
        dQ_dpap_r, dQ_dpap_i = self._complex_mul_real(conj_pap_r, conj_pap_i, dQ_dpb, "dQ_dpap")
        dQ_dpam_r, dQ_dpam_i = self._complex_mul_real(conj_pam_r, conj_pam_i, dQ_dpbbar, "dQ_dpam")

        # Chain through pap = gp*ap + gm*poq*am
        # Wirtinger chain: dQ/d(ap) = dQ/d(pap)·gp + dQ/d(pam)·(gm/poq)
        #                   dQ/d(am) = dQ/d(pap)·(gm·poq) + dQ/d(pam)·gp
        # Note: these use DIRECT derivatives, not conjugates.
        d1_r, d1_i = self._complex_mul(dQ_dpap_r, dQ_dpap_i, gp_r, gp_i, "d1")
        cgm_cpoq_r, cgm_cpoq_i = self._complex_div(gm_r, gm_i, poq_r, poq_i, "cgm_cpoq")
        d2_r, d2_i = self._complex_mul(dQ_dpam_r, dQ_dpam_i, cgm_cpoq_r, cgm_cpoq_i, "d2")
        dQ_dap_r = self._node("Add", [d1_r, d2_r])
        dQ_dap_i = self._node("Add", [d1_i, d2_i])

        gm_poq_r2, gm_poq_i2 = self._complex_mul(gm_r, gm_i, poq_r, poq_i, "gm_poq")
        d3_r, d3_i = self._complex_mul(dQ_dpap_r, dQ_dpap_i, gm_poq_r2, gm_poq_i2, "d3")
        d4_r, d4_i = self._complex_mul(dQ_dpam_r, dQ_dpam_i, gp_r, gp_i, "d4")
        dQ_dam_r = self._node("Add", [d3_r, d4_r])
        dQ_dam_i = self._node("Add", [d3_i, d4_i])

        # Expand dQ_dap, dQ_dam from (N,) to (N, n_half)
        # ap = sum(a[:, :n_half], axis=-1), am = sum(a[:, n_half:], axis=-1)
        # dQ/da[:, :n_half] = dQ_dap (broadcast)
        dQ_dap_r_2d = self._node("Unsqueeze", [dQ_dap_r], axes=[1])
        dQ_dap_i_2d = self._node("Unsqueeze", [dQ_dap_i], axes=[1])
        dQ_dam_r_2d = self._node("Unsqueeze", [dQ_dam_r], axes=[1])
        dQ_dam_i_2d = self._node("Unsqueeze", [dQ_dam_i], axes=[1])

        # Tile to (N, n_half)
        sh_half = self._shape_2d(batch_size, n_half)
        dQ_dap_r_e = self._node("Expand", [dQ_dap_r_2d, sh_half])
        dQ_dap_i_e = self._node("Expand", [dQ_dap_i_2d, sh_half])
        dQ_dam_r_e = self._node("Expand", [dQ_dam_r_2d, sh_half])
        dQ_dam_i_e = self._node("Expand", [dQ_dam_i_2d, sh_half])

        # dQ/da = concat(dQ_dap, dQ_dam) along axis=1
        dQ_da_r_full = self._node("Concat", [dQ_dap_r_e, dQ_dam_r_e], axis=1)
        dQ_da_i_full = self._node("Concat", [dQ_dap_i_e, dQ_dam_i_e], axis=1)

        # Gradient for ck: a = ck * common → dQ/d(ck) = sum(dQ/da * common, axis=0)
        # (Wirtinger: d(ck·common)/d(ck) = common since it's holomorphic)
        g_ck_r, g_ck_i = self._complex_mul(dQ_da_r_full, dQ_da_i_full, cf_r, cf_i, "g_ck")
        gck_r_sum = self._node("ReduceSum", [g_ck_r], axes=[0], keepdims=0)
        gck_i_sum = self._node("ReduceSum", [g_ck_i], axes=[0], keepdims=0)
        # Output the Wirtinger derivative dQ/d(ck) components directly.
        # The fitter's backprop_grad() expects ∂Q/∂ck, NOT the real partials.
        # Wirtinger: dQ/d(ck) = Re(G) + j·Im(G)  where G = sum(dQ_da * common).
        grad_ck_real = self._node("Identity", [gck_r_sum])
        grad_ck_imag = self._node("Identity", [gck_i_sum])
        self._node("Identity", [grad_ck_real], outputs="grad_ck_real")
        self._node("Identity", [grad_ck_imag], outputs="grad_ck_imag")

        # ════════════════════════════════════════════════════════
        # dQ/dbw_p — backprop through BW product and scatter
        # ════════════════════════════════════════════════════════

        # dQ/dbw_p = -dQ/da * a / bw_p  (Wirtinger derivative)
        # a = ck * cf, da/d(1/bw_p) = ck * fa * fl_p
        # From numpy: dQ_dbw_p = dQ/da * (-ck/bw_p²*fa*fl_p) = -dQ/da*a/bw_p
        neg_da_r, neg_da_i = self._node("Neg", [dQ_da_r_full]), self._node("Neg", [dQ_da_i_full])
        a_r, a_i = self._complex_mul(ck_r_2d, ck_i_2d, cf_r, cf_i, "a_bw")
        abw_r, abw_i = self._complex_div(a_r, a_i, bw_p_r, bw_p_i, "abw")
        dQ_dbw_p_r, dQ_dbw_p_i = self._complex_mul(neg_da_r, neg_da_i, abw_r, abw_i, "dQ_dbw_p")

        # Backprop through product: for n_res=2
        # bw_p = bw_dom_0 * bw_dom_1 → dQ/d(bw_dom_0) = dQ/d(bw_p) * bw_dom_1
        bw_f1_r = self._slice_axis(bw_3d_r, 2, 1, 2, "bf1r")
        bw_f1_i = self._slice_axis(bw_3d_i, 2, 1, 2, "bf1i")
        dQ_dd0_r, dQ_dd0_i = self._complex_mul(dQ_dbw_p_r, dQ_dbw_p_i, bw_f1_r, bw_f1_i, "dd0")
        bw_f0_r = self._slice_axis(bw_3d_r, 2, 0, 1, "bf0r")
        bw_f0_i = self._slice_axis(bw_3d_i, 2, 0, 1, "bf0i")
        dQ_dd1_r, dQ_dd1_i = self._complex_mul(dQ_dbw_p_r, dQ_dbw_p_i, bw_f0_r, bw_f0_i, "dd1")

        # Stack dQ_dd0 and dQ_dd1 → (N, n_wave, 2)
        d0_3d_r = self._node("Unsqueeze", [dQ_dd0_r], axes=[2])
        d0_3d_i = self._node("Unsqueeze", [dQ_dd0_i], axes=[2])
        d1_3d_r = self._node("Unsqueeze", [dQ_dd1_r], axes=[2])
        d1_3d_i = self._node("Unsqueeze", [dQ_dd1_i], axes=[2])
        dQ_dbw_3d_r = self._node("Concat", [d0_3d_r, d1_3d_r], axis=2)
        dQ_dbw_3d_i = self._node("Concat", [d0_3d_i, d1_3d_i], axis=2)
        # Reshape to (N, n_wave * n_res)
        dQ_dbw_flat_r = self._node("Reshape", [dQ_dbw_3d_r, self._shape_2d(N, n_wave * n_res)])
        dQ_dbw_flat_i = self._node("Reshape", [dQ_dbw_3d_i, self._shape_2d(N, n_wave * n_res)])

        # Scatter by bw_order using MatMul with a constant scatter matrix
        # S: (n_wave * n_res, n_unique_bw) where S[k, j] = 1 if bw_order[k] == j
        scatter_S = np.zeros((n_wave * n_res, n_unique_bw), dtype=np.float32)
        for k, bw_idx in enumerate(self.kc["bw_order"]):
            scatter_S[k, bw_idx] = 1.0
        self._embed("scatter_S", scatter_S)

        dQ_dbd_r = self._node("MatMul", [dQ_dbw_flat_r, "scatter_S"])
        dQ_dbd_i = self._node("MatMul", [dQ_dbw_flat_i, "scatter_S"])

        # ════════════════════════════════════════════════════════
        # grad_m0
        # ════════════════════════════════════════════════════════
        # dQ/dm0 = 2·Re(Σ dQ/dbw_dom · (2·m0 - 1j·g_bw), axis=0)
        # (2·m0 - 1j·g_bw) complex derivative of bw_dom w.r.t m0
        # -1j*(g_bw_r + j*g_bw_i) = -j*g_bw_r + g_bw_i
        # So dbw_dm0 = (2*m0 + g_bw_i) + j*(-g_bw_r)
        two_m0 = self._node("Mul", [m0_all_2d, self._scalar(2.0)])
        # ∂bw_dom/∂m0 = 2*m0 - j*g_bw = (2*m0 + g_bw_i) + j*(-g_bw_r)
        # Note: g_bw_i and g_bw_r are NOT multiplied by m0 again
        dbw_dm0_r = self._node("Add", [two_m0, g_bw_i])
        dbw_dm0_i = self._node("Neg", [g_bw_r])

        # ∂Q/∂m0 = 2·Re(dQ/d(bw_dom) · ∂bw_dom/∂m0)  (Wirtinger for real param)
        n_m0_unique = len(np.unique(self.kc["m0_index"]))
        m0_scatter = np.zeros((n_unique_bw, n_m0_unique), dtype=np.float32)
        for i, m_idx in enumerate(self.kc["m0_index"]):
            m0_scatter[i, m_idx] = 1.0
        self._embed("m0_scatter", m0_scatter)

        dm0_r, dm0_i = self._complex_mul(dQ_dbd_r, dQ_dbd_i, dbw_dm0_r, dbw_dm0_i, "dm0")
        dm0_sum = self._node("ReduceSum", [dm0_r], axes=[0], keepdims=0)
        # Scatter from (N, n_unique_bw) to (N, n_m0_unique) via m0_index
        dm0_scat = self._node("MatMul", [dm0_sum, "m0_scatter"])
        grad_m0 = self._node("Mul", [dm0_scat, self._scalar(2.0)])

        # ════════════════════════════════════════════════════════
        # grad_g0
        # ════════════════════════════════════════════════════════
        # dQ/dg_bw = dQ/dbw_dom * (-1j·m0)  (chain through bw_dom)
        # -1j·m0 = (0 + j*(-m0))
        # (dQ_r + j*dQ_i) * (0 - j*m0) = (dQ_i*m0) + j*(-dQ_r*m0)
        dQ_dgbw_r = self._node("Mul", [dQ_dbd_i, m0_all_2d])
        neg_m0 = self._node("Neg", [m0_all_2d])
        dQ_dgbw_i = self._node("Mul", [dQ_dbd_r, neg_m0])

        # dQ/dg = dQ/dg_bw @ matrix_gamma.T  (chain through g_bw = g @ matrix_gamma)
        mg_t = self._node("Transpose", ["matrix_gamma"], perm=[1, 0])
        dQ_dg_r = self._node("MatMul", [dQ_dgbw_r, mg_t])
        dQ_dg_i = self._node("MatMul", [dQ_dgbw_i, mg_t])

        # ∂Q/∂g0 = 2·Re(Σ dQ/dg · ∂g/∂g0, axis=0)  (Wirtinger for real param)
        # ∂g/∂g0 = g_interp (g = g0 * g_interp)
        n_g0_unique = len(np.unique(self.kc["g0_index"]))
        g0_scatter = np.zeros((n_gamma_rows, n_g0_unique), dtype=np.float32)
        for i, g_idx in enumerate(self.kc["g0_index"]):
            g0_scatter[i, g_idx] = 1.0
        self._embed("g0_scatter", g0_scatter)

        gg_r, gg_i = self._complex_mul(dQ_dg_r, dQ_dg_i, g_interp_r, g_interp_i, "gg")
        gg_sum = self._node("ReduceSum", [gg_r], axes=[0], keepdims=0)
        gg_scat = self._node("MatMul", [gg_sum, "g0_scatter"])
        grad_g0 = self._node("Mul", [gg_scat, self._scalar(2.0)])

        # ════════════════════════════════════════════════════════
        # Scalar gradients
        # ════════════════════════════════════════════════════════
        # dQ/dgp = dQ/dpb·pap̄·ap + dQ/dpbbar·pam̄·am  (Wirtinger)
        dpg_r, dpg_i = self._complex_mul_real(conj_pap_r, conj_pap_i, dQ_dpb, "dpg")
        dpg2_r, dpg2_i = self._complex_mul(dpg_r, dpg_i, ap_r, ap_i, "dpg2")
        dpg3_r, dpg3_i = self._complex_mul_real(conj_pam_r, conj_pam_i, dQ_dpbbar, "dpg3")
        dpg4_r, dpg4_i = self._complex_mul(dpg3_r, dpg3_i, am_r, am_i, "dpg4")
        dQ_dgp_r = self._node("Add", [dpg2_r, dpg4_r])
        dQ_dgp_i = self._node("Add", [dpg2_i, dpg4_i])

        # dQ/dgm = dQ/dpb·pap̄·poq·am + dQ/dpbbar·pam̄·ap/poq
        # dpg already includes dQ_dpb factor
        dmg_r, dmg_i = self._complex_mul(dpg_r, dpg_i, poq_r, poq_i, "dmg")
        dmg2_r, dmg2_i = self._complex_mul(dmg_r, dmg_i, am_r, am_i, "dmg2")
        # dpg3 = conj(pam) * dQ_dpbbar (real*complex, dpg3 already has dQ_dpbbar)
        dmg5_r, dmg5_i = self._complex_div(dpg3_r, dpg3_i, poq_r, poq_i, "dmg5")
        dmg6_r, dmg6_i = self._complex_mul(dmg5_r, dmg5_i, ap_r, ap_i, "dmg6")
        dQ_dgm_r = self._node("Add", [dmg2_r, dmg6_r])
        dQ_dgm_i = self._node("Add", [dmg2_i, dmg6_i])

        half_t = self._node("Mul", [time, half])
        qua_t = self._node("Mul", [time, self._scalar(0.25)])

        # dG = 2·Re(Σ dQ/dgp·(-t/2·gp) + dQ/dgm·(-t/2·gm))
        # But for scalar param: dQ/dΓ = 2·Re(Σ dQ/dgp · dgp/dΓ + dQ/dgm · dgm/dΓ)
        def _grad_time(dgp_r, dgp_i, dgm_r, dgm_i):
            p1_r, p1_i = self._complex_mul(dQ_dgp_r, dQ_dgp_i, dgp_r, dgp_i, "gt1")
            p2_r, p2_i = self._complex_mul(dQ_dgm_r, dQ_dgm_i, dgm_r, dgm_i, "gt2")
            sr = self._node("ReduceSum", [self._node("Add", [p1_r, p2_r])], axes=[0], keepdims=0)
            return self._node("Mul", [sr, self._scalar(2.0)])

        nht = self._node("Neg", [half_t])
        nqt = self._node("Neg", [qua_t])
        ggpr_r, ggpr_i = self._complex_mul_real(gp_r, gp_i, nht, "ggpr")
        ggmr_r, ggmr_i = self._complex_mul_real(gm_r, gm_i, nht, "ggmr")
        grad_Gamma = _grad_time(ggpr_r, ggpr_i, ggmr_r, ggmr_i)

        gdgr_r, gdgr_i = self._complex_mul_real(gm_r, gm_i, nqt, "gdgr")
        gdgm_r, gdgm_i = self._complex_mul_real(gp_r, gp_i, nqt, "gdgm")
        grad_DeltaGamma = _grad_time(gdgr_r, gdgr_i, gdgm_r, gdgm_i)

        # dgp/dΔm = j·t/2·gm, dgm/dΔm = j·t/2·gp
        # j*gm = (0+1j)*(gm_r+j*gm_i) = -gm_i + j*gm_r
        htgm_r = self._node("Neg", [self._node("Mul", [gm_i, half_t])])
        htgm_i = self._node("Mul", [gm_r, half_t])
        htgp_r = self._node("Neg", [self._node("Mul", [gp_i, half_t])])
        htgp_i = self._node("Mul", [gp_r, half_t])
        grad_DeltaM = _grad_time(htgm_r, htgm_i, htgp_r, htgp_i)

        # A_prod gradient
        grad_A_prod = dQ_dAp

        # poq gradients: dQ/dpoq = dQ/dpb·pap̄·gm·am + dQ/dpbbar·pam̄·(-gm/poq²)·ap
        pd_r, pd_i = self._complex_mul(conj_pap_r, conj_pap_i, gm_r, gm_i, "pd")
        pd_r, pd_i = self._complex_mul(pd_r, pd_i, am_r, am_i, "pd2")
        pd_r = self._node("Mul", [pd_r, dQ_dpb]); pd_i = self._node("Mul", [pd_i, dQ_dpb])
        one_r = self._scalar(1.0); one_i = self._scalar(0.0)
        inv_poq_r, inv_poq_i = self._complex_div(one_r, one_i, poq_r, poq_i, "ipoq")
        inv_poq_sq_r, inv_poq_sq_i = self._complex_mul(inv_poq_r, inv_poq_i, inv_poq_r, inv_poq_i, "ipoq_sq")
        # Expand scalar poq terms to (N,) for proper ONNX broadcasting
        inv_poq_sq_r = self._node("Expand", [inv_poq_sq_r, self._shape_1d(N)])
        inv_poq_sq_i = self._node("Expand", [inv_poq_sq_i, self._shape_1d(N)])
        gm_poq_sq_r, gm_poq_sq_i = self._complex_mul(gm_r, gm_i, inv_poq_sq_r, inv_poq_sq_i, "gps")
        neg_gps_r = self._node("Neg", [gm_poq_sq_r])
        neg_gps_i = self._node("Neg", [gm_poq_sq_i])
        pdd_r, pdd_i = self._complex_mul(conj_pam_r, conj_pam_i, neg_gps_r, neg_gps_i, "pdd")
        pdd_r, pdd_i = self._complex_mul(pdd_r, pdd_i, ap_r, ap_i, "pdd2")
        pdd_r = self._node("Mul", [pdd_r, dQ_dpbbar]); pdd_i = self._node("Mul", [pdd_i, dQ_dpbbar])
        dQ_dpoq_r = self._node("Add", [pd_r, pdd_r])
        dQ_dpoq_i = self._node("Add", [pd_i, pdd_i])
        # Debug: save dQ_dpoq mean for comparison

        # dQ/dρ = 2·Re(Σ dQ/dpoq · exp(j·ϕ))
        cos_phi2 = self._node("Cos", [self._node("Reshape", [pop_phi, self._shape_1d(1)])])
        sin_phi2 = self._node("Sin", [self._node("Reshape", [pop_phi, self._shape_1d(1)])])
        # Expand to per-event rank
        cphi = self._node("Expand", [cos_phi2, self._shape_1d(N)])
        sphi = self._node("Expand", [sin_phi2, self._shape_1d(N)])
        # dQ_dpoq * exp(jϕ) → complex mul
        er, ei = self._complex_mul(dQ_dpoq_r, dQ_dpoq_i, cphi, sphi, "er")
        grad_poq_rho = self._node("Mul", [self._node("ReduceSum", [er], axes=[0], keepdims=0), self._scalar(2.0)])

        # dQ/dϕ = 2·Re(Σ dQ/dpoq · ρ·j·exp(j·ϕ))
        # j·exp(jϕ) = j·cos ϕ - sin ϕ = (-sin ϕ) + j·cos ϕ
        jcphi_r = self._node("Neg", [sphi])
        jcphi_i = cphi
        pr_1d = self._node("Reshape", [poq_rho, self._shape_1d(1)])
        jcphi_pr_r, jcphi_pr_i = self._complex_mul_real(jcphi_r, jcphi_i, pr_1d, "jc")
        jcphi_pr_r = self._node("Expand", [jcphi_pr_r, self._shape_1d(N)])
        jcphi_pr_i = self._node("Expand", [jcphi_pr_i, self._shape_1d(N)])
        pr_r, pr_i = self._complex_mul(dQ_dpoq_r, dQ_dpoq_i, jcphi_pr_r, jcphi_pr_i, "pr")
        grad_pop_phi = self._node("Mul", [self._node("ReduceSum", [pr_r], axes=[0], keepdims=0), self._scalar(2.0)])

        # ── output with Identity ──
        self._node("Identity", [grad_m0], outputs="grad_m0")
        self._node("Identity", [grad_g0], outputs="grad_g0")
        gsc_list = [self._node("Reshape", [g, self._shape_1d(1)]) for g in
                    [grad_Gamma, grad_DeltaGamma, grad_DeltaM, grad_A_prod, grad_poq_rho, grad_pop_phi]]
        self._node("Identity", [self._node("Concat", gsc_list, axis=0)], outputs="grad_scalar")

        # ── outputs ──
        Q_vi = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [])
        P_vi = helper.make_tensor_value_info("P", TensorProto.FLOAT, [batch_size])
        gck_r_vi = helper.make_tensor_value_info("grad_ck_real", TensorProto.FLOAT, [n_wave])
        gck_i_vi = helper.make_tensor_value_info("grad_ck_imag", TensorProto.FLOAT, [n_wave])
        n_m0_unique = len(np.unique(self.kc["m0_index"]))
        gm0_vi = helper.make_tensor_value_info("grad_m0", TensorProto.FLOAT, [n_m0_unique])
        n_g0_unique = len(np.unique(self.kc["g0_index"]))
        gg0_vi = helper.make_tensor_value_info("grad_g0", TensorProto.FLOAT, [n_g0_unique])
        gsc_vi = helper.make_tensor_value_info("grad_scalar", TensorProto.FLOAT, [6])
        self._value_info.extend([Q_vi, P_vi, gck_r_vi, gck_i_vi, gm0_vi, gg0_vi, gsc_vi])
        graph_outputs = [Q_vi, P_vi, gck_r_vi, gck_i_vi, gm0_vi, gg0_vi, gsc_vi]

        input_names = set(["ck_real", "ck_imag", "m0", "g0",
                       "mass", "q", "angle", "frac", "time", "weight",
                       "Gamma", "Delta_Gamma", "Delta_m", "A_prod", "poq_rho", "pop_phi"])
        if not norm_model:
            input_names |= {"bkg", "norm"}
        inputs_vi = [vi for vi in self._value_info if vi.name in input_names]

        graph = helper.make_graph(self._nodes, self._graph_name,
                                  inputs_vi,
                                  graph_outputs,
                                  list(self._consts.values()))
        model = helper.make_model(graph, opset_imports=[
            helper.make_opsetid("", 11)
        ])
        onnx.checker.check_model(model)
        # Strip unused initializers (keeps onnxruntime from spamming warnings)
        used = set()
        for n in model.graph.node: used.update(n.input); used.update(n.output)
        kept = [i for i in model.graph.initializer if i.name in used]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
        return model

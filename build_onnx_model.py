#!/usr/bin/env python3
"""
Build a PWA ONNX model directly (no PyTorch dependency).

The graph is constructed node-by-node using the `onnx` library.
Complex numbers are split into (real, imag) float pairs throughout.
"""
import sys, os, argparse, numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ampfit.config_loader import Config

OP = onnx.helper.make_node


def float_type():
    return TensorProto.DOUBLE


def int64():
    return TensorProto.INT64


class PWAONNXBuilder:
    """Builds a PWA forward-computation ONNX graph.

    All complex values are stored as pairs ``(name_r, name_i)``.
    The model takes event data + parameters and returns ``(Q, P)``.
    """

    def __init__(self, config_file="config_angle.yml"):
        self.config = Config(config_file)
        kc = self.config.build_all_index()

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
        self._embed("matrix_angle_real", np.real(kc["matrix_angle"]))
        self._embed("matrix_angle_imag", np.imag(kc["matrix_angle"]))
        self._embed("matrix_gamma", kc["matrix_gamma"])
        self._embed("gamma_table_real", np.real(kc["gamma_table"]))
        self._embed("gamma_table_imag", np.imag(kc["gamma_table"]))
        self._embed("fl_table", kc["fl_table"])
        self._embed("angle_k", kc["angle_k"])
        self._embed("angle_b", kc["angle_b"])

        # Interpolation configs
        self.gamma_min = float(kc["gamma_min"])
        self.gamma_delta = float(kc["gamma_delta"])
        self.fl_min = float(kc["fl_min"])
        self.fl_delta = float(kc["fl_delta"])

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

    def _input(self, name, shape, dtype=TensorProto.DOUBLE):
        vi = helper.make_tensor_value_info(name, dtype, shape)
        self._value_info.append(vi)
        return name

    def _scalar(self, value, dtype=TensorProto.DOUBLE):
        name = self._name(f"c{value}")
        if dtype == TensorProto.DOUBLE:
            self._embed(name, np.array(value, dtype=np.float64))
        else:
            self._embed(name, np.array(value, dtype=np.int64))
        return name

    def _node(self, op_type, inputs, outputs=None, **attrs):
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
        self._embed(name, np.array(value, dtype=np.float64 if dtype == TensorProto.FLOAT else np.int64))
        return name

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
        """(a_r + j*a_i) / (b_r + j*b_i)."""
        denom = self._complex_abs2(br, bi, "den")
        conj_r = br
        conj_i = self._node("Neg", [bi])
        num_r = self._node("Add", [self._node("Mul", [ar, conj_r]),
                                   self._node("Mul", [ai, conj_i])])
        num_i = self._node("Sub", [self._node("Mul", [ai, conj_r]),
                                   self._node("Mul", [ar, conj_i])])
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
        xbin_float = self._node("Cast", [xbin_floor], to=TensorProto.DOUBLE)
        zero_d = self._scalar(0.0)
        nbm2_d = self._scalar(float(n_bins - 2))
        xbin_max = self._node("Max", [xbin_float, zero_d])
        xbin_clip_f = self._node("Min", [xbin_max, nbm2_d])
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

    def build(self, batch_size=4):
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
        m0 = self._input("m0", [self.n_unique_bw])
        g0 = self._input("g0", [n_gamma_rows])
        mass = self._input("mass", [N, 48])
        q = self._input("q", [N, 72])
        angle = self._input("angle", [N, 24, 3])
        frac = self._input("frac", [N])
        time = self._input("time", [N])
        weight = self._input("weight", [N])
        bkg = self._input("bkg", [N])
        norm = self._input("norm", [])
        Gamma = self._input("Gamma", [])
        Delta_Gamma = self._input("Delta_Gamma", [])
        Delta_m = self._input("Delta_m", [])
        A_prod = self._input("A_prod", [])
        poq_rho = self._input("poq_rho", [])
        pop_phi = self._input("pop_phi", [])

        # ──── 1. Gamma interpolation → g_bw ────
        # g0_all = g0[g0_index]
        gather_g0 = self._gather_1d(g0, "g0_index", "g0_all")
        g0_all_2d = self._node("Unsqueeze", [gather_g0], axes=[0])
        # g0_mass from mass by g0_mass_index
        g0_mass = self._gather_cols(mass, "g0_mass_index", "g0_mass")
        # interpolate gamma table
        n_gamma_rows = len(self.config.build_all_index()["g0_index"])
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
        n_fl_cols = len(self.config.build_all_index()["fl_type"])
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
        angle_k_f = self._node("Cast", ["angle_k"], to=TensorProto.DOUBLE)
        ang_k_mul = self._node("Mul", [ang, angle_k_f])
        ang_kb = self._node("Add", [ang_k_mul, "angle_b"])
        # Cos/Sin not implemented for DOUBLE in some ORT builds; cast to FLOAT
        ang_kb_f = self._node("Cast", [ang_kb], to=TensorProto.FLOAT)
        cos_a_f = self._node("Cos", [ang_kb_f])
        cos_a = self._node("Cast", [cos_a_f], to=TensorProto.DOUBLE)
        ka = self._node("ReduceProd", [cos_a], axes=[2], keepdims=0)

        # fa = ka @ matrix_angle (complex matmul)
        fa_r = self._node("MatMul", [ka, "matrix_angle_real"])
        fa_i = self._node("MatMul", [ka, "matrix_angle_imag"])

        # ──── 6. Amplitude ────
        # inv_bw = 1/bw_p (complex division)
        one_r = self._scalar(1.0)
        one_i = self._scalar(0.0)
        inv_r, inv_i = self._complex_div(
            self._node("Expand", [one_r, self._shape_1d(n_wave)]),
            self._node("Expand", [one_i, self._shape_1d(n_wave)]),
            bw_p_r, bw_p_i, "inv"
        )
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
        phi_f = self._node("Cast", [pop_phi], to=TensorProto.FLOAT)
        cos_phi_f = self._node("Cos", [phi_f])
        sin_phi_f = self._node("Sin", [phi_f])
        cos_phi = self._node("Cast", [cos_phi_f], to=TensorProto.DOUBLE)
        sin_phi = self._node("Cast", [sin_phi_f], to=TensorProto.DOUBLE)
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
            # Exp/Cos/Sin on FLOAT for ORT compatibility
            tb_f = self._node("Cast", [tb], to=TensorProto.FLOAT)
            ta_f = self._node("Cast", [ta], to=TensorProto.FLOAT)
            exp_tb_f = self._node("Exp", [tb_f])
            cos_ta_f = self._node("Cos", [ta_f])
            sin_ta_f = self._node("Sin", [ta_f])
            exp_tb = self._node("Cast", [exp_tb_f], to=TensorProto.DOUBLE)
            cos_ta = self._node("Cast", [cos_ta_f], to=TensorProto.DOUBLE)
            sin_ta = self._node("Cast", [sin_ta_f], to=TensorProto.DOUBLE)
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

        # NLL
        eps = self._scalar(1e-30)
        Pnorm = self._node("Div", [P, norm])
        Pbkg = self._node("Add", [Pnorm, bkg])
        clamped = self._node("Max", [Pbkg, eps])
        logP = self._node("Log", [clamped])
        wlog = self._node("Mul", [weight, logP])
        wsum = self._node("ReduceSum", [wlog], axes=[0], keepdims=0)
        Q = self._node("Neg", [wsum], outputs="Q")
        P_out = self._name("P_final")
        self._node("Identity", [P], outputs="P")

        # ──── outputs ────
        Q_vi = helper.make_tensor_value_info("Q", TensorProto.DOUBLE, [])
        P_vi = helper.make_tensor_value_info("P", TensorProto.DOUBLE, [batch_size])
        self._value_info.append(Q_vi)
        self._value_info.append(P_vi)

        # Inputs list (excluding constants)
        input_names = ["ck_real", "ck_imag", "m0", "g0",
                       "mass", "q", "angle", "frac", "time", "weight", "bkg",
                       "norm",
                       "Gamma", "Delta_Gamma", "Delta_m", "A_prod", "poq_rho", "pop_phi"]
        inputs_vi = [vi for vi in self._value_info if vi.name in input_names]

        graph = helper.make_graph(self._nodes, self._graph_name,
                                  inputs_vi,
                                  [Q_vi, P_vi],
                                  list(self._consts.values()))
        model = helper.make_model(graph, opset_imports=[
            helper.make_opsetid("", 11)
        ])
        onnx.checker.check_model(model)
        return model


def main():
    parser = argparse.ArgumentParser(description="Build PWA ONNX model")
    parser.add_argument("--config", default="config_angle.yml")
    parser.add_argument("--output", default="pwa_forward.onnx")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    print("Building ONNX graph...")
    builder = PWAONNXBuilder(args.config)
    model = builder.build()
    onnx.save(model, args.output)
    print(f"✓ Saved to {args.output}")
    print(f"  Inputs: {len(model.graph.input)}")
    print(f"  Outputs: {len(model.graph.output)}")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Constants: {len(model.graph.initializer)}")

    if args.validate:
        print("\nValidating with onnxruntime...")
        import onnxruntime as ort
        sess = ort.InferenceSession(args.output)
        for i in sess.get_inputs():
            print(f"  Input {i.name}: {i.shape}")
        for o in sess.get_outputs():
            print(f"  Output {o.name}: {o.shape}")
        print("✓ Model loads successfully")


if __name__ == "__main__":
    main()

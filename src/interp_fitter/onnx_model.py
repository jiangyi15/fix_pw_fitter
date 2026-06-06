"""Build the Kernel forward pass directly as an ONNX model.

Complex numbers are kept as *separate* real / imag tensors throughout
so that every ONNX op stays scalar (no ``(..., 2)`` dimension trickery).
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


# ---------------------------------------------------------------------------
#  GraphBuilder  —  thin wrapper around ``helper.make_node``
# ---------------------------------------------------------------------------

class GraphBuilder:
    """Accumulates ONNX nodes and tracks tensor / initializer names.

    Parameters
    ----------
    opset : int
        Target ONNX opset version.  Squeeze / Unsqueeze / ReduceSum
        use attribute-based axes when ``opset <= 12`` (opset 11
        compatible), and tensor-based axes otherwise (opset 13+).
    """

    def __init__(self, name: str = "Kernel", opset: int = 11):
        self.name = name
        self.opset = opset
        self._inputs: list[helper.ValueInfoProto] = []
        self._outputs: list[helper.ValueInfoProto] = []
        self._init: list[TensorProto] = []
        self._nodes: list[helper.NodeProto] = []
        self._counter = 0

    def _uid(self, tag: str) -> str:
        self._counter += 1
        return f"{tag}_{self._counter}"

    # -- inputs / outputs / constants ---------------------------------------

    def input(self, name: str, dtype: int, shape):
        self._inputs.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    def output(self, name: str, dtype: int, shape):
        self._outputs.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    def const(self, name: str, arr: np.ndarray) -> str:
        self._init.append(numpy_helper.from_array(arr, name))
        return name

    # -- node factory -------------------------------------------------------

    def node(self, op_type: str, inputs: list[str], outputs: list[str],
             **attrs):
        node = helper.make_node(op_type, inputs, outputs,
                                name=self._uid(op_type), **attrs)
        self._nodes.append(node)
        return outputs[0] if len(outputs) == 1 else outputs

    # -- scalar helpers -----------------------------------------------------

    def scalar(self, tag: str, val, dtype=np.float32) -> str:
        name = self._uid(tag)
        self.const(name, np.array(val, dtype=dtype))
        return name

    def int_scalar(self, tag: str, val: int) -> str:
        """Scalar int64 constant (for Gather indices etc.)."""
        name = self._uid(tag)
        self.const(name, np.array(val, dtype=np.int64))
        return name

    # -- unary ops ----------------------------------------------------------

    def neg(self, x: str) -> str:
        return self.node("Neg", [x], [self._uid("neg")])

    def exp(self, x: str) -> str:
        return self.node("Exp", [x], [self._uid("exp")])

    def log(self, x: str) -> str:
        return self.node("Log", [x], [self._uid("log")])

    def cos(self, x: str) -> str:
        return self.node("Cos", [x], [self._uid("cos")])

    def sin(self, x: str) -> str:
        return self.node("Sin", [x], [self._uid("sin")])

    def floor(self, x: str) -> str:
        return self.node("Floor", [x], [self._uid("floor")])

    # -- binary ops ---------------------------------------------------------

    def add(self, a: str, b: str) -> str:
        return self.node("Add", [a, b], [self._uid("add")])

    def sub(self, a: str, b: str) -> str:
        return self.node("Sub", [a, b], [self._uid("sub")])

    def mul(self, a: str, b: str) -> str:
        return self.node("Mul", [a, b], [self._uid("mul")])

    def div(self, a: str, b: str) -> str:
        return self.node("Div", [a, b], [self._uid("div")])

    # -- reductions ---------------------------------------------------------

    def reduce_sum(self, x: str, axes: list[int], keepdims: int = 0) -> str:
        if self.opset <= 12:
            return self.node("ReduceSum", [x], [self._uid("rsum")],
                             axes=axes, keepdims=keepdims)
        ax_name = self._uid("rsax")
        self.const(ax_name, np.array(axes, dtype=np.int64))
        return self.node("ReduceSum", [x, ax_name], [self._uid("rsum")],
                         keepdims=keepdims)

    def reduce_prod(self, x: str, axes: list[int], keepdims: int = 0) -> str:
        return self.node("ReduceProd", [x], [self._uid("rprod")],
                         axes=axes, keepdims=keepdims)

    # -- shape ops ----------------------------------------------------------

    def reshape(self, x: str, shape: str | list[int]) -> str:
        if isinstance(shape, list):
            s = self._uid("shape")
            self.const(s, np.array(shape, dtype=np.int64))
            shape = s
        return self.node("Reshape", [x, shape], [self._uid("rs")])

    def unsqueeze(self, x: str, axes: list[int]) -> str:
        if self.opset <= 12:
            return self.node("Unsqueeze", [x], [self._uid("unsq")], axes=axes)
        ax_name = self._uid("uax")
        self.const(ax_name, np.array(axes, dtype=np.int64))
        return self.node("Unsqueeze", [x, ax_name], [self._uid("unsq")])

    def squeeze(self, x: str, axes: list[int]) -> str:
        if self.opset <= 12:
            return self.node("Squeeze", [x], [self._uid("sq")], axes=axes)
        ax_name = self._uid("sax")
        self.const(ax_name, np.array(axes, dtype=np.int64))
        return self.node("Squeeze", [x, ax_name], [self._uid("sq")])

    def gather(self, data: str, indices: str, axis: int = 0) -> str:
        return self.node("Gather", [data, indices], [self._uid("gth")],
                         axis=axis)

    def transpose(self, x: str, perm: list[int]) -> str:
        return self.node("Transpose", [x], [self._uid("tpose")], perm=perm)

    def matmul(self, a: str, b: str) -> str:
        return self.node("MatMul", [a, b], [self._uid("mm")])

    def cast(self, x: str, to: int) -> str:
        return self.node("Cast", [x], [self._uid("cst")], to=to)

    # -- stack two tensors along a new axis via Unsqueeze + Concat ----------

    def stack2(self, a: str, b: str, axis: int = -1) -> str:
        """Stack *a* and *b* along *axis*."""
        # For axis=-1: unsqueeze both at last axis, then concat
        a2 = self.unsqueeze(a, [axis])
        b2 = self.unsqueeze(b, [axis])
        return self.node("Concat", [a2, b2], [self._uid("stk")], axis=axis)

    # -- complex numbers as pairs (re, im) ----------------------------------

    def c_mul(self, ar: str, ai: str, br: str, bi: str):
        """Return ``(re, im)`` of *a × b*."""
        rr = self.sub(self.mul(ar, br), self.mul(ai, bi))
        ri = self.add(self.mul(ar, bi), self.mul(ai, br))
        return rr, ri

    def c_rmul(self, r: str, zr: str, zi: str):
        """Real scalar * complex."""
        return self.mul(r, zr), self.mul(r, zi)

    def c_abs2(self, zr: str, zi: str) -> str:
        return self.add(self.mul(zr, zr), self.mul(zi, zi))

    def c_conj(self, zr: str, zi: str):
        return zr, self.neg(zi)

    def c_exp(self, zr: str, zi: str):
        e = self.exp(zr)
        return self.mul(e, self.cos(zi)), self.mul(e, self.sin(zi))

    def c_div(self, ar: str, ai: str, br: str, bi: str):
        denom = self.add(self.mul(br, br), self.mul(bi, bi))
        rr = self.div(self.add(self.mul(ar, br), self.mul(ai, bi)), denom)
        ri = self.div(self.sub(self.mul(ai, br), self.mul(ar, bi)), denom)
        return rr, ri

    def c_inv(self, zr: str, zi: str):
        denom = self.add(self.mul(zr, zr), self.mul(zi, zi))
        return self.div(zr, denom), self.div(self.neg(zi), denom)

    def c_add(self, ar: str, ai: str, br: str, bi: str):
        return self.add(ar, br), self.add(ai, bi)

    def c_sub(self, ar: str, ai: str, br: str, bi: str):
        return self.sub(ar, br), self.sub(ai, bi)

    def c_prod_seq(self, zr: str, zi: str, n: int, dim: int = 1):
        """Complex product along *dim*.

        ``zr, zi`` have shape ``(..., n, ...)``.
        The product is over ``n`` elements along *dim* via unrolled
        pairwise multiplication.

        Note: ``Gather`` with a scalar index already removes the
        indexed dimension, so no extra ``Squeeze`` is needed.
        """
        slices_r, slices_i = [], []
        for i in range(n):
            idx_name = self.int_scalar(f"cps_{i}", i)
            sr = self.gather(zr, idx_name, axis=dim)  # dim removed by scalar gather
            si = self.gather(zi, idx_name, axis=dim)
            slices_r.append(sr)
            slices_i.append(si)
        if n == 0:
            raise ValueError("empty product")
        rr, ri = slices_r[0], slices_i[0]
        for i in range(1, n):
            rr, ri = self.c_mul(rr, ri, slices_r[i], slices_i[i])
        return rr, ri

    def c_gather(self, dr: str, di: str, idx_name: str, axis: int = 1):
        """Gather complex slices along *axis* by index array."""
        return (self.gather(dr, idx_name, axis=axis),
                self.gather(di, idx_name, axis=axis))

    # -- build ---------------------------------------------------------------

    def build(self) -> onnx.ModelProto:
        graph = helper.make_graph(self._nodes, self.name,
                                  self._inputs, self._outputs, self._init)
        model = helper.make_model(graph, producer_name="interp_fitter",
                                  opset_imports=[
                                      helper.make_operatorsetid("", self.opset)])
        return model


# ---------------------------------------------------------------------------
#  Interpolation helpers  (pure ONNX nodes)
# ---------------------------------------------------------------------------

def _interp_real(g: GraphBuilder, x: str,
                 tbl_name: str, n_int: int,
                 types_name: str,
                 xmin_name: str, xdelta_name: str) -> str:
    """Linear interpolation for a real table. Returns interpolant."""
    diff = g.div(g.sub(x, xmin_name), xdelta_name)
    xbin = g.floor(diff)
    xbin_i = g.cast(xbin, TensorProto.INT64)
    t1d = g.unsqueeze(g.cast(types_name, TensorProto.INT64), [0])
    off = g.mul(t1d, g.int_scalar("nir", n_int))
    flat_idx = g.add(off, xbin_i)

    flat_tbl = g.reshape(tbl_name, [-1])
    left = g.gather(flat_tbl, flat_idx, axis=0)
    idx_p1 = g.add(flat_idx, g.int_scalar("oner", 1))
    right = g.gather(flat_tbl, idx_p1, axis=0)
    frac = g.sub(diff, xbin)
    return g.add(left, g.mul(g.sub(right, left), frac))


# ---------------------------------------------------------------------------
#  Builder — full model
# ---------------------------------------------------------------------------

def build_onnx_model(config: dict, with_norm: bool = True,
                     opset: int = 11) -> onnx.ModelProto:
    """Build an ONNX model for ``Kernel.compute(...)``.

    Parameters
    ----------
    with_norm : bool
        If True, ``norm`` is an input and ``Q = -Σ w·log(P/norm + bkg)``.
        If False, no ``norm`` input and ``Q = Σ w·P``.
    opset : int
        ONNX opset version.  Use 11 for maximum CANN / Ascend
        compatibility;  Use 13 / 17 for more recent runtimes.

    Inputs (all ``FLOAT``):
      ck_re (nwaves,), ck_im (nwaves,), m0 (n_m0,), g0 (n_g0,),
      time_params (6,),
      mass (nevt, ndim_mass), q (nevt, ndim_q), angle (nevt, ndim_angle),
      time (nevt,), weight (nevt,), frac (nevt,), bkg (nevt,)
      norm ()                                          — only if *with_norm*

    Outputs: P (nevt,), Q ()
    """
    g = GraphBuilder("Kernel", opset=opset)

    # -- static dimensions --------------------------------------------------
    cfg = config
    n_int_g = cfg["gamma_table"].shape[-1]
    n_int_f = cfg["fl_table"].shape[-1]
    n_gamma = len(cfg["g0_index"])
    n_m0 = cfg["matrix_gamma"].shape[0]
    n_bw = len(cfg["m0_index"])
    nwaves = cfg["matrix_ang"].shape[-1]
    nres = len(cfg["bw_order"]) // nwaves
    ndec = len(cfg["fl_order"]) // nwaves
    nbasis = cfg["ang_order"].shape[0]

    F = TensorProto.FLOAT

    # ---- inputs -----------------------------------------------------------
    ck_re = g.input("ck_re", F, ("nwaves",))
    ck_im = g.input("ck_im", F, ("nwaves",))
    m0_in = g.input("m0", F, ("n_m0",))
    g0_in = g.input("g0", F, ("n_g0",))
    tp_in = g.input("time_params", F, (6,))
    mass = g.input("mass", F, ("nevt", "ndim_mass"))
    q_in = g.input("q", F, ("nevt", "ndim_q"))
    angle = g.input("angle", F, ("nevt", "ndim_angle"))
    time = g.input("time", F, ("nevt",))
    weight = g.input("weight", F, ("nevt",))
    frac = g.input("frac", F, ("nevt",))
    bkg = g.input("bkg", F, ("nevt",))
    if with_norm:
        norm = g.input("norm", F, ())

    # ---- constants --------------------------------------------------------
    gt = np.asarray(cfg["gamma_table"], dtype=np.complex64)
    g.const("gt_re", np.ascontiguousarray(gt.real))
    g.const("gt_im", np.ascontiguousarray(gt.imag))

    g.const("ft", np.asarray(cfg["fl_table"], dtype=np.float32))
    g.const("mat_gamma", np.asarray(cfg["matrix_gamma"], dtype=np.float32))
    ma = np.asarray(cfg["matrix_ang"], dtype=np.complex64)
    g.const("ma_re", np.ascontiguousarray(ma.real))
    g.const("ma_im", np.ascontiguousarray(ma.imag))

    for name in ("g0_index", "gamma_index", "gamma_type",
                 "m0_index", "bw_index", "bw_gamma_index", "bw_order",
                 "q_index", "fl_type", "fl_order",
                 "angle_index"):
        g.const(name, np.asarray(cfg[name], dtype=np.int64))

    # ang_order flattened
    g.const("ang_order_f",
            np.asarray(cfg["ang_order"], dtype=np.int64).ravel())

    for name in ("gamma_min", "gamma_delta", "fl_min", "fl_delta"):
        g.const(name, np.array(cfg[name], dtype=np.float32))

    g.const("angle_k", np.asarray(cfg["angle_k"], dtype=np.float32))
    g.const("angle_b", np.asarray(cfg["angle_b"], dtype=np.float32))

    # ---- helper for complex interpolation (using split re/im tables) ------

    def interp_c(xx, tbl_re, tbl_im, types_name, n_int, xmin_n, xdelta_n):
        diff = g.div(g.sub(xx, xmin_n), xdelta_n)
        xbin = g.floor(diff)
        xbin_i = g.cast(xbin, TensorProto.INT64)
        t1d = g.unsqueeze(g.cast(types_name, TensorProto.INT64), [0])
        off = g.mul(t1d, g.int_scalar("nc", n_int))
        flat_idx = g.add(off, xbin_i)

        flat_re = g.reshape(tbl_re, [-1])
        flat_im = g.reshape(tbl_im, [-1])
        left_re = g.gather(flat_re, flat_idx, axis=0)
        left_im = g.gather(flat_im, flat_idx, axis=0)
        idx_p1 = g.add(flat_idx, g.int_scalar("onec", 1))
        right_re = g.gather(flat_re, idx_p1, axis=0)
        right_im = g.gather(flat_im, idx_p1, axis=0)
        frac = g.sub(diff, xbin)
        res_r = g.add(left_re, g.mul(g.sub(right_re, left_re), frac))
        res_i = g.add(left_im, g.mul(g.sub(right_im, left_im), frac))
        return res_r, res_i

    # ---- 1) Gamma interpolation ------------------------------------------
    g0a = g.gather(g0_in, "g0_index", axis=0)              # (n_gamma,)
    mg = g.gather(mass, "gamma_index", axis=1)             # (nevt, n_gamma)
    gi_r, gi_i = interp_c(mg, "gt_re", "gt_im", "gamma_type",
                          n_int_g, "gamma_min", "gamma_delta")
    gv_r, gv_i = g.c_rmul(g0a, gi_r, gi_i)

    # gamma_for_mass = mat_gamma @ gamma_val  (einsum ij,...j->...i)
    # (n_m0, n_gamma) @ (nevt, n_gamma) -> (nevt, n_m0)
    # We need: for each event e: gamma_for_mass[e] = mat_gamma @ gamma_val[e]
    # gv_r/e is (nevt, n_gamma), mat_gamma is (n_m0, n_gamma)
    # We want (nevt, n_m0) = (nevt, n_gamma) @ (n_gamma, n_m0)
    mg_t = g.transpose("mat_gamma", [1, 0])                 # (n_gamma, n_m0)
    gfm_r = g.matmul(gv_r, mg_t)                            # (nevt, n_m0)
    gfm_i = g.matmul(gv_i, mg_t)

    # ---- 2) Breit-Wigner ------------------------------------------------
    m0a = g.gather(m0_in, "m0_index", axis=0)               # (n_bw,)
    mbw = g.gather(mass, "bw_index", axis=1)                # (nevt, n_bw)
    gbw_r = g.gather(gfm_r, "bw_gamma_index", axis=1)       # (nevt, n_bw)
    gbw_i = g.gather(gfm_i, "bw_gamma_index", axis=1)

    # bwdom = m0a² - s² - i*m0a*Gamma
    #   = (m0a² - s² + m0a*Im(Γ))  +  i*(-m0a*Re(Γ))
    bwdom_r = g.add(g.sub(g.mul(m0a, m0a), g.mul(mbw, mbw)),
                    g.mul(m0a, gbw_i))
    bwdom_i = g.neg(g.mul(m0a, gbw_r))
    bw_r, bw_i = g.c_inv(bwdom_r, bwdom_i)                  # (nevt, n_bw)

    # Product over resonances per wave
    bw_ord_r, bw_ord_i = g.c_gather(bw_r, bw_i, "bw_order", axis=1)
    bw_rsh_r = g.reshape(bw_ord_r, [0, nwaves, nres])
    bw_rsh_i = g.reshape(bw_ord_i, [0, nwaves, nres])
    bwa_r, bwa_i = g.c_prod_seq(bw_rsh_r, bw_rsh_i, nres, dim=2)

    # ---- 3) Form factors ------------------------------------------------
    fl_q = g.gather(q_in, "q_index", axis=1)                 # (nevt, n_fl)
    fl = _interp_real(g, fl_q, "ft", n_int_f,
                      "fl_type", "fl_min", "fl_delta")       # (nevt, n_fl)
    fl_ord = g.gather(fl, "fl_order", axis=1)                # (nevt, Nw*ndec)
    fl_rsh = g.reshape(fl_ord, [0, nwaves, ndec])
    fla = g.reduce_prod(fl_rsh, [2])                          # (nevt, nwaves)

    # ---- 4) Angular basis -----------------------------------------------
    ang = g.gather(angle, "angle_index", axis=1)             # (nevt, n_ang)
    ang_a = g.add(g.mul(ang, "angle_k"), "angle_b")
    cosang = g.cos(ang_a)                                    # (nevt, n_ang)
    cos_ord = g.gather(cosang, "ang_order_f", axis=1)        # (nevt, Nb*n_per)
    nbasis = int(cfg["ang_order"].shape[0])
    n_per = int(cfg["ang_order"].shape[1])
    cos_rsh = g.reshape(cos_ord, [0, nbasis, n_per])
    cosa = g.reduce_prod(cos_rsh, [2])                        # (nevt, nbasis)

    # fa = cosa @ matrix_ang  (complex)
    fa_r = g.matmul(cosa, "ma_re")
    fa_i = g.matmul(cosa, "ma_im")

    # ---- 5) Amplitude ---------------------------------------------------
    # T = bwa * fla * fa   (bwa complex, fla real, fa complex)
    # First: bwa * fa  (complex * complex)
    bf_r, bf_i = g.c_mul(bwa_r, bwa_i, fa_r, fa_i)
    # Then * fla (real)
    T_r = g.mul(bf_r, fla)
    T_i = g.mul(bf_i, fla)

    # amp_waves = ck * T
    aw_r, aw_i = g.c_mul(ck_re, ck_im, T_r, T_i)             # (nevt, nwaves)

    # Split into 2 CP groups
    n_per_group = nwaves // 2
    aw_rsh_r = g.reshape(aw_r, [0, 2, n_per_group])
    aw_rsh_i = g.reshape(aw_i, [0, 2, n_per_group])

    idx0 = g.int_scalar("i0", 0)
    idx1 = g.int_scalar("i1", 1)

    g0_r = g.gather(aw_rsh_r, idx0, axis=1)   # (nevt, n_per_group) — scalar index removes dim
    g0_i = g.gather(aw_rsh_i, idx0, axis=1)
    g1_r = g.gather(aw_rsh_r, idx1, axis=1)
    g1_i = g.gather(aw_rsh_i, idx1, axis=1)

    amp0_r = g.reduce_sum(g0_r, [1])   # (nevt,)
    amp0_i = g.reduce_sum(g0_i, [1])
    amp1_r = g.reduce_sum(g1_r, [1])
    amp1_i = g.reduce_sum(g1_i, [1])

    # ---- 6) Time-dependent mixing ----------------------------------------
    gt = g.gather(tp_in, g.int_scalar("ti0", 0), axis=0)
    dg = g.gather(tp_in, g.int_scalar("ti1", 1), axis=0)
    dm = g.gather(tp_in, g.int_scalar("ti2", 2), axis=0)
    poqr = g.gather(tp_in, g.int_scalar("ti3", 3), axis=0)
    poqi = g.gather(tp_in, g.int_scalar("ti4", 4), axis=0)
    ap = g.gather(tp_in, g.int_scalar("ti5", 5), axis=0)

    half = g.scalar("h", 0.5)
    argL_re = g.neg(g.mul(g.mul(time, g.add(gt, g.mul(dg, half))), half))
    argL_im = g.neg(g.mul(g.mul(time, dm), half))
    argH_re = g.neg(g.mul(g.mul(time, g.sub(gt, g.mul(dg, half))), half))
    argH_im = g.mul(g.mul(time, dm), half)

    eL_r, eL_i = g.c_exp(argL_re, argL_im)
    eH_r, eH_i = g.c_exp(argH_re, argH_im)
    ep_r, ep_i = g.c_rmul(half, *g.c_add(eL_r, eL_i, eH_r, eH_i))
    em_r, em_i = g.c_rmul(half, *g.c_sub(eL_r, eL_i, eH_r, eH_i))

    poq_r = g.mul(poqr, g.cos(poqi))
    poq_i = g.mul(poqr, g.sin(poqi))

    # X = ep*amp0 + poq*em*amp1
    t1_r, t1_i = g.c_mul(ep_r, ep_i, amp0_r, amp0_i)
    t2_r, t2_i = g.c_mul(poq_r, poq_i, em_r, em_i)
    t3_r, t3_i = g.c_mul(t2_r, t2_i, amp1_r, amp1_i)
    X_r, X_i = g.c_add(t1_r, t1_i, t3_r, t3_i)

    # Y = em/poq * amp0 + ep * amp1
    poq_div_r, poq_div_i = g.c_inv(poq_r, poq_i)
    em_poq_r, em_poq_i = g.c_mul(em_r, em_i, poq_div_r, poq_div_i)
    u1_r, u1_i = g.c_mul(em_poq_r, em_poq_i, amp0_r, amp0_i)
    u2_r, u2_i = g.c_mul(ep_r, ep_i, amp1_r, amp1_i)
    Y_r, Y_i = g.c_add(u1_r, u1_i, u2_r, u2_i)

    PB = g.c_abs2(X_r, X_i)
    PBbar = g.c_abs2(Y_r, Y_i)

    one = g.scalar("o", 1.0)
    P = g.add(
        g.mul(g.mul(g.sub(one, frac), g.sub(one, ap)), PB),
        g.mul(g.mul(frac, g.add(one, ap)), PBbar),
    )

    # ---- 7) Objective ---------------------------------------------------
    if with_norm:
        Pnorm = g.add(g.div(P, norm), bkg)
        Q = g.neg(g.reduce_sum(g.mul(weight, g.log(Pnorm)), [0]))
    else:
        Q = g.reduce_sum(g.mul(weight, P), [0])

    g.node("Identity", [P], ["P"])
    g.node("Identity", [Q], ["Q"])
    g.output("P", F, ("nevt",))
    g.output("Q", F, ())
    return g.build()


def export_to_onnx(config: dict, onnx_path: str, with_norm: bool = True,
                   opset: int = 11):
    """Build, check, and save the ONNX model."""
    model = build_onnx_model(config, with_norm=with_norm, opset=opset)
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    return model

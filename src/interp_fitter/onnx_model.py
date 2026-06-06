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
        """Scalar int32 constant (for Gather indices etc.)."""
        name = self._uid(tag)
        self.const(name, np.array(val, dtype=np.int32))
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

    # -- Concat / stack ----------------------------------------------------

    def concat(self, tensors: list[str], axis: int) -> str:
        return self.node("Concat", tensors, [self._uid("cat")], axis=axis)

    def stack2(self, a: str, b: str, axis: int = -1) -> str:
        """Stack *a* and *b* along *axis*."""
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

    def c_pow2(self, zr: str, zi: str):
        """Complex square: z² = (zr² - zi²) + i*(2*zr*zi)."""
        rr = self.sub(self.mul(zr, zr), self.mul(zi, zi))
        two = self.scalar("two", 2.0)
        ri = self.mul(self.mul(two, zr), zi)
        return rr, ri

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
    xbin_i = g.cast(xbin, TensorProto.INT32)
    t1d = g.unsqueeze(g.cast(types_name, TensorProto.INT32), [0])
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
                     opset: int = 11, nevt: int | None = None,
                     with_gradients: bool = True) -> onnx.ModelProto:
    """Build an ONNX model for ``Kernel.compute(...)``.

    Parameters
    ----------
    nevt : int or None
        If given, all shapes are static (fixed batch size).
        If ``None``, the batch dimension is symbolic (``"nevt"``).

    For other parameters see :func:`export_to_onnx`.

    Inputs (all ``FLOAT``):
      ck_re (nwaves,), ck_im (nwaves,), m0 (n_m0,), g0 (n_g0,),
      time_params (6,),
      mass (nevt, ndim_mass), q (nevt, ndim_q), angle (nevt, ndim_angle),
      time (nevt,), weight (nevt,), frac (nevt,), bkg (nevt,)
      norm ()                                          — only if *with_norm*

    Outputs: P (nevt,), Q ()
    """
    g = GraphBuilder("Kernel", opset=opset)

    cfg = config

    # inner dims (always concrete — derived from config)
    ndim_mass = int(max(np.max(cfg["gamma_index"]), np.max(cfg["bw_index"])) + 1)
    ndim_q    = int(np.max(cfg["q_index"]) + 1)
    ndim_angle = int(np.max(cfg["angle_index"]) + 1)
    n_int_g = cfg["gamma_table"].shape[-1]
    n_int_f = cfg["fl_table"].shape[-1]
    n_gamma = len(cfg["g0_index"])
    n_g0 = int(np.max(cfg["g0_index"]) + 1)
    n_m0 = cfg["matrix_gamma"].shape[0]
    n_bw = len(cfg["m0_index"])
    nwaves = cfg["matrix_ang"].shape[-1]
    nres = len(cfg["bw_order"]) // nwaves
    ndec = len(cfg["fl_order"]) // nwaves
    nbasis = cfg["ang_order"].shape[0]

    # batch dim
    _N = nevt if nevt else "nevt"

    def RS(*dims: int) -> list[int]:
        """Reshape shape — replace ``0`` with *nevt* when static."""
        return [nevt if d == 0 else d for d in dims] if nevt else list(dims)

    F = TensorProto.FLOAT

    # ---- inputs (all concrete for small dims) -----------------------------
    ck_re = g.input("ck_re", F, (nwaves,))
    ck_im = g.input("ck_im", F, (nwaves,))
    m0_in = g.input("m0", F, (n_m0,))
    g0_in = g.input("g0", F, (n_g0,))
    tp_in = g.input("time_params", F, (6,))
    mass = g.input("mass", F, (_N, ndim_mass))
    q_in = g.input("q", F, (_N, ndim_q))
    angle = g.input("angle", F, (_N, ndim_angle))
    time = g.input("time", F, (_N,))
    weight = g.input("weight", F, (_N,))
    frac = g.input("frac", F, (_N,))
    bkg = g.input("bkg", F, (_N,))
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
        g.const(name, np.asarray(cfg[name], dtype=np.int32))

    # ang_order flattened
    g.const("ang_order_f",
            np.asarray(cfg["ang_order"], dtype=np.int32).ravel())

    for name in ("gamma_min", "gamma_delta", "fl_min", "fl_delta"):
        g.const(name, np.array(cfg[name], dtype=np.float32))

    g.const("angle_k", np.asarray(cfg["angle_k"], dtype=np.float32))
    g.const("angle_b", np.asarray(cfg["angle_b"], dtype=np.float32))

    # ---- helper for complex interpolation (using split re/im tables) ------

    def interp_c(xx, tbl_re, tbl_im, types_name, n_int, xmin_n, xdelta_n):
        diff = g.div(g.sub(xx, xmin_n), xdelta_n)
        xbin = g.floor(diff)
        xbin_i = g.cast(xbin, TensorProto.INT32)
        t1d = g.unsqueeze(g.cast(types_name, TensorProto.INT32), [0])
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
    bw_rsh_r = g.reshape(bw_ord_r, RS(0, nwaves, nres))
    bw_rsh_i = g.reshape(bw_ord_i, RS(0, nwaves, nres))
    bwa_r, bwa_i = g.c_prod_seq(bw_rsh_r, bw_rsh_i, nres, dim=2)

    # ---- 3) Form factors ------------------------------------------------
    fl_q = g.gather(q_in, "q_index", axis=1)                 # (nevt, n_fl)
    fl = _interp_real(g, fl_q, "ft", n_int_f,
                      "fl_type", "fl_min", "fl_delta")       # (nevt, n_fl)
    fl_ord = g.gather(fl, "fl_order", axis=1)                # (nevt, Nw*ndec)
    fl_rsh = g.reshape(fl_ord, RS(0, nwaves, ndec))
    fla = g.reduce_prod(fl_rsh, [2])                          # (nevt, nwaves)

    # ---- 4) Angular basis -----------------------------------------------
    ang = g.gather(angle, "angle_index", axis=1)             # (nevt, n_ang)
    ang_a = g.add(g.mul(ang, "angle_k"), "angle_b")
    cosang = g.cos(ang_a)                                    # (nevt, n_ang)
    cos_ord = g.gather(cosang, "ang_order_f", axis=1)        # (nevt, Nb*n_per)
    nbasis = int(cfg["ang_order"].shape[0])
    n_per = int(cfg["ang_order"].shape[1])
    cos_rsh = g.reshape(cos_ord, RS(0, nbasis, n_per))
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
    aw_rsh_r = g.reshape(aw_r, RS(0, 2, n_per_group))
    aw_rsh_i = g.reshape(aw_i, RS(0, 2, n_per_group))

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

    # ================================================================
    #  8) Gradients
    # ================================================================

    # --- dQ_dP --------------------------------------------------------
    if with_norm:
        dQ_dP_norm = g.neg(g.div(weight, Pnorm))
        dQ_dP = g.div(dQ_dP_norm, norm)
        grad_norm = g.reduce_sum(
            g.mul(dQ_dP_norm, g.neg(g.div(P, g.mul(norm, norm)))), [0])
    else:
        dQ_dP = weight

    # --- Wirtinger adjoints |X|², |Y|² --------------------------------
    cX_r, cX_i = g.c_conj(X_r, X_i)
    cY_r, cY_i = g.c_conj(Y_r, Y_i)

    def _adj(r, i, coeff):
        return g.mul(coeff, r), g.mul(coeff, i)

    adjX_r, adjX_i = _adj(cX_r, cX_i, g.mul(g.sub(one, frac), g.sub(one, ap)))
    adjY_r, adjY_i = _adj(cY_r, cY_i, g.mul(frac, g.add(one, ap)))

    dX_r = g.mul(dQ_dP, adjX_r)
    dX_i = g.mul(dQ_dP, adjX_i)
    dY_r = g.mul(dQ_dP, adjY_r)
    dY_i = g.mul(dQ_dP, adjY_i)

    # --- ∂Q/∂amp0, ∂Q/∂amp1 -------------------------------------------
    t_r, t_i = g.c_mul(dX_r, dX_i, ep_r, ep_i)
    u_r, u_i = g.c_mul(dY_r, dY_i, em_poq_r, em_poq_i)   # em/poq from forward
    dA0_r, dA0_i = g.c_add(t_r, t_i, u_r, u_i)

    p1_r, p1_i = g.c_mul(poq_r, poq_i, em_r, em_i)        # poq * em
    t_r, t_i = g.c_mul(dX_r, dX_i, p1_r, p1_i)
    u_r, u_i = g.c_mul(dY_r, dY_i, ep_r, ep_i)
    dA1_r, dA1_i = g.c_add(t_r, t_i, u_r, u_i)

    # scatter → (nevt, nwaves) — unsqueeze each 1D column first
    dAw_pairs = [(dA0_r, dA0_i)] * n_per_group + [(dA1_r, dA1_i)] * n_per_group
    dAw_r = g.concat(
        [g.reshape(r, RS(0, 1)) for r, _ in dAw_pairs], axis=1)
    dAw_i = g.concat(
        [g.reshape(i, RS(0, 1)) for _, i in dAw_pairs], axis=1)

    # --- grad ck -------------------------------------------------------
    td_r, td_i = g.c_mul(dAw_r, dAw_i, T_r, T_i)
    dck_w_r = g.reduce_sum(td_r, [0])
    dck_w_i = g.reduce_sum(td_i, [0])
    dck_cr, dck_ci = g.c_conj(dck_w_r, dck_w_i)
    two_f = g.scalar("tcf", 2.0)
    grad_ck_re = g.mul(two_f, dck_cr)
    grad_ck_im = g.mul(two_f, dck_ci)

    # --- grad m0 -------------------------------------------------------
    # Backprop: amp_waves → T → bwa → bw → m0a → m0
    # ∂Q/∂T = dAw * ck   (Wirtinger)
    dT_r, dT_i = g.c_mul(dAw_r, dAw_i, ck_re, ck_im)

    # ∂Q/∂bwa = ∂Q/∂T * fla * fa   (Wirtinger, T holomorphic in bwa)
    bf_r, bf_i = g.c_mul(dT_r, dT_i, fa_r, fa_i)
    dbwa_r, dbwa_i = g.c_rmul(fla, bf_r, bf_i)

    # d(bwa[e,w]) / d(bw[e,w,r]) = bwa[e,w] / bw_reshaped[e,w,r]
    # ∂Q/∂bw_reshaped = ∂Q/∂bwa * bwa / bw_reshaped
    # Unsqueeze bwa/dbwa to (nevt, nwaves, 1) for broadcasting
    us = [1]  # unsqueeze axis for complex (nevt, nwaves) → (nevt, nwaves, 1)
    # c_rmul handles broadcasting of fla already, but bwa and dbwa need unsqueeze
    # Let me use reshape instead
    bwa_us_r = g.reshape(bwa_r, RS(0, nwaves, 1))
    bwa_us_i = g.reshape(bwa_i, RS(0, nwaves, 1))
    dbwa_us_r = g.reshape(dbwa_r, RS(0, nwaves, 1))
    dbwa_us_i = g.reshape(dbwa_i, RS(0, nwaves, 1))

    # ratio = bwa / bw_reshaped
    ratio_r, ratio_i = g.c_div(bwa_us_r, bwa_us_i, bw_rsh_r, bw_rsh_i)
    # dQ/dbw_reshaped = dQ/dbwa * ratio
    drsh_r, drsh_i = g.c_mul(dbwa_us_r, dbwa_us_i, ratio_r, ratio_i)

    # Scatter by bw_order: (nevt, nwaves*nres) → (nevt, n_bw)
    drsh_flat_r = g.reshape(drsh_r, RS(0, nwaves * nres))
    drsh_flat_i = g.reshape(drsh_i, RS(0, nwaves * nres))
    dQ_dbw_r = g.gather(drsh_flat_r, "bw_order", axis=1)
    dQ_dbw_i = g.gather(drsh_flat_i, "bw_order", axis=1)
    # Hmm, this is the reverse: we need to SCATTER, not gather.
    # The bw_order tells us which bw entry each (w,r) maps to.
    # We need to add contributions from all (w,r) that map to the same d.
    # In numpy this is np.add.at. In ONNX we use ReduceSum on a scattered tensor.

    # Build scatter matrix: one-hot for each (w,r) → n_bw
    # Simpler: just use a manually constructed scatter via Gather on the
    # transposed lookup. Actually the easiest is to construct the
    # mapping explicitly. Let me use a matrix multiplication.
    # Create one-hot: (nwaves*nres, n_bw) where one-hot[i, bw_order[i]] = 1
    # In ONNX: create the one-hot as an initializer.

    # Actually, let me just use the transpose of the bw_order relationship.
    # drsh_flat: (nevt, nwaves*nres) — the gradient for each (w,r) position
    # We need to sum over (w,r) positions that have the same underlying bw index.
    # bw_order maps (w,r) → bw_idx, so we need the REVERSE mapping.
    # Build a scatter matrix S of shape (nwaves*nres, n_bw) where S[i, bw_order[i]] = 1.
    # Then dQ_dbw = drsh_flat @ S  (matrix multiply)
    scatter_mat = np.zeros((nwaves * nres, n_bw), dtype=np.float32)
    for i, d in enumerate(cfg["bw_order"]):
        scatter_mat[i, d] = 1.0
    g.const("scatter_mat", scatter_mat)

    dQ_dbw_r = g.matmul(drsh_flat_r, "scatter_mat")
    dQ_dbw_i = g.matmul(drsh_flat_i, "scatter_mat")

    # dbw/dm0a = -bw² * (2*m0a - i*Γ)
    # (2*m0a - i*Γ) = (2*m0a + Im(Γ)) + i*(-Re(Γ))
    bw2_r, bw2_i = g.c_pow2(bw_r, bw_i)
    two_m0a = g.mul(g.scalar("tm", 2.0), m0a)
    num_r = g.add(two_m0a, gbw_i)
    num_i = g.neg(gbw_r)
    t_r, t_i = g.c_mul(bw2_r, bw2_i, num_r, num_i)
    dbw_dm0a_r = g.neg(t_r)
    dbw_dm0a_i = g.neg(t_i)

    # dQ/dm0a[d] = 2 * Re(Σ_e dQ_dbw * dbw/dm0a)
    # The real part of the complex product is Re(dQ_dbw * dbw/dm0a)
    prod_r = g.sub(g.mul(dQ_dbw_r, dbw_dm0a_r), g.mul(dQ_dbw_i, dbw_dm0a_i))
    sum_r = g.reduce_sum(prod_r, [0])
    grad_m0a = g.mul(two_f, sum_r)  # 2 * Re(Σ)

    # scatter m0a → m0
    scatter_m0 = np.zeros((n_bw, n_m0), dtype=np.float32)
    for d, k in enumerate(cfg["m0_index"]):
        scatter_m0[d, k] = 1.0
    g.const("scatter_m0", scatter_m0)
    grad_m0_r2 = g.reshape(grad_m0a, [1, n_bw])                    # (1, n_bw)
    grad_m0_r2 = g.matmul(grad_m0_r2, "scatter_m0")                # (1, n_m0)
    grad_m0 = g.squeeze(grad_m0_r2, [0])                           # (n_m0,)

    # --- grad g0 -------------------------------------------------------
    # dQ/dΓ_bw = dQ/dbw * dbw/dΓ     (Wirtinger)
    # dbw/dΓ = -bw² * (-i*m0a)   where Γ = gamma_for_bw (complex)
    # -i*m0a * dΓ  ...  Actually: bw = 1/(m0a² - s² - i*m0a*Γ)
    # dbw/dΓ = -bw² * (-i*m0a) = bw² * i*m0a
    # = (bw_r + i*bw_i)² * i * m0a
    # = m0a * i * (bw_r² - bw_i² + 2i*bw_r*bw_i)
    # = m0a * (i*(bw_r² - bw_i²) - 2*bw_r*bw_i)
    # So: d(bw_r + i*bw_i)/d(Γ_r + i*Γ_i) = m0a * (-2*bw_r*bw_i + i*(bw_r² - bw_i²))
    # Wait, this is the Wirtinger derivative of bw w.r.t. Γ.
    # bw = 1/D where D = m0a² - s² - i*m0a*Γ
    # d(bw)/dΓ = -bw² * dD/dΓ = -bw² * (-i*m0a) = i*m0a*bw²
    # So d(bw)/dΓ (Wirtinger) = i * m0a * bw²

    # i * bw² = i * ((bw_r² - bw_i²) + i*(2*bw_r*bw_i))
    #          = i*(bw_r² - bw_i²) - 2*bw_r*bw_i
    #          = -2*bw_r*bw_i + i*(bw_r² - bw_i²)
    bw2_r, bw2_i = g.c_pow2(bw_r, bw_i)   # already computed above
    # i * bw²:
    ibw2_r = g.neg(g.mul(g.scalar("t", 2.0), g.mul(bw_r, bw_i)))  # = -2*bw_r*bw_i
    # Wait, I already have bw2_r = bw_r² - bw_i², bw2_i = 2*bw_r*bw_i
    # i * bw² = -bw2_i + i*bw2_r
    ibw2_r = g.neg(bw2_i)
    ibw2_i = bw2_r
    # dbw/dΓ = m0a * i * bw²
    dbw_dG_r = g.mul(m0a, ibw2_r)
    dbw_dG_i = g.mul(m0a, ibw2_i)

    # ∂Q/∂Γ_bw = dQ_dbw * dbw_dG   (Wirtinger)
    dG_bw_r, dG_bw_i = g.c_mul(dQ_dbw_r, dQ_dbw_i, dbw_dG_r, dbw_dG_i)

    # Map Γ_bw → Γ_mass via bw_gamma_index (reverse mapping)
    # Γ_mass has shape (nevt, n_m0).  Γ_bw = Γ_mass[:, bw_gamma_index]
    # Need to scatter dQ/dΓ_bw back to dQ/dΓ_mass
    scatter_gbw = np.zeros((n_bw, n_m0), dtype=np.float32)
    for d, k in enumerate(cfg["bw_gamma_index"]):
        scatter_gbw[d, k] = 1.0
    g.const("scatter_gbw", scatter_gbw)
    dG_mass_r = g.matmul(dG_bw_r, "scatter_gbw")                 # (nevt, n_m0)
    dG_mass_i = g.matmul(dG_bw_i, "scatter_gbw")

    # Γ_mass[:,i] = Σ_j gamma_val[:,j] * matrix_gamma[i,j]
    # ∂Q/∂gamma_val[:,j] = Σ_i ∂Q/∂Γ_mass[:,i] * matrix_gamma[i,j]
    # (nevt, n_m0) @ (n_m0, n_gamma) = (nevt, n_gamma)
    dG_val_r = g.matmul(dG_mass_r, "mat_gamma")
    dG_val_i = g.matmul(dG_mass_i, "mat_gamma")

    # gamma_val = g0a * gamma_interp
    # dQ/dg0a = 2 * Re(Σ_e dQ/dgamma_val * conj(gamma_interp))
    # Wait: gamma_val[:,j] = g0a[j] * gamma_interp[:,j]
    # gamma_val is complex, g0a is real.
    # ∂Q/∂g0a = ∂Q/∂gamma_val * ∂gamma_val/∂g0a + ∂Q/∂gamma_val̄ * ∂gamma_val̄/∂g0a
    # = dG_val * gamma_interp + conj(dG_val) * conj(gamma_interp)
    # = 2 * Re(dG_val * gamma_interp)

    # Problem: dG_val is Wirtinger, gamma_interp is complex.
    # The product dG_val * gamma_interp is complex.
    # I need 2 * Re(Σ_e dG_val[e,j] * gamma_interp[e,j])

    # dG_val_r + i*dG_val_i and gi_r + i*gi_i
    # (dG_val_r * gi_r - dG_val_i * gi_i) + i*(dG_val_r * gi_i + dG_val_i * gi_r)
    # Real part = dG_val_r * gi_r - dG_val_i * gi_i
    g0a_prod_r = g.sub(g.mul(dG_val_r, gi_r), g.mul(dG_val_i, gi_i))
    g0a_sum = g.reduce_sum(g0a_prod_r, [0])
    grad_g0a = g.mul(two_f, g0a_sum)                              # (n_gamma,)

    # scatter g0a → g0 via g0_index
    scatter_g0 = np.zeros((n_gamma, n_g0), dtype=np.float32)
    for j, k in enumerate(cfg["g0_index"]):
        scatter_g0[j, k] = 1.0
    g.const("scatter_g0", scatter_g0)
    grad_g0_r2 = g.reshape(grad_g0a, [1, n_gamma])                # (1, n_gamma)
    grad_g0_r2 = g.matmul(grad_g0_r2, "scatter_g0")               # (1, n_g0)
    grad_g0 = g.squeeze(grad_g0_r2, [0])                          # (n_g0,)

    # --- grad time_params ---------------------------------------------
    # Derivatives of eL, eH
    deL_dgt = g.c_rmul(g.neg(g.div(time, two_f)), eL_r, eL_i)
    deH_dgt = g.c_rmul(g.neg(g.div(time, two_f)), eH_r, eH_i)
    deL_ddg = g.c_rmul(g.neg(g.div(time, g.scalar("f4", 4.0))), eL_r, eL_i)
    deH_ddg = g.c_rmul(g.div(time, g.scalar("f4b", 4.0)), eH_r, eH_i)
    deL_ddm = g.c_rmul(g.neg(g.mul(time, half)), eL_r, eL_i)
    deH_ddm = g.c_rmul(g.mul(time, half), eH_r, eH_i)

    half_f = g.scalar("hf", 0.5)
    dep_dgt = (g.c_rmul(half_f, *deL_dgt), g.c_rmul(half_f, *deH_dgt))
    # Hmm, this is getting messy with tuple unpacking. Let me use a helper.

    def _c_avg(a_r, a_i, b_r, b_i):
        return g.c_rmul(half_f, *g.c_add(a_r, a_i, b_r, b_i))

    def _c_diff(a_r, a_i, b_r, b_i):
        return g.c_rmul(half_f, *g.c_sub(a_r, a_i, b_r, b_i))

    dep_dgt_r, dep_dgt_i = _c_avg(*deL_dgt, *deH_dgt)
    dem_dgt_r, dem_dgt_i = _c_diff(*deL_dgt, *deH_dgt)
    dep_ddg_r, dep_ddg_i = _c_avg(*deL_ddg, *deH_ddg)
    dem_ddg_r, dem_ddg_i = _c_diff(*deL_ddg, *deH_ddg)
    dep_ddm_r, dep_ddm_i = _c_avg(*deL_ddm, *deH_ddm)
    dem_ddm_r, dem_ddm_i = _c_diff(*deL_ddm, *deH_ddm)

    # dX/dp = dep/dp * amp0 + poq * dem/dp * amp1
    # dY/dp = dem/dp / poq * amp0 + dep/dp * amp1
    def _dXdY(dep_r, dep_i, dem_r, dem_i):
        t1_r, t1_i = g.c_mul(dep_r, dep_i, amp0_r, amp0_i)
        p1_r, p1_i = g.c_mul(poq_r, poq_i, dem_r, dem_i)
        t2_r, t2_i = g.c_mul(p1_r, p1_i, amp1_r, amp1_i)
        dx_r, dx_i = g.c_add(t1_r, t1_i, t2_r, t2_i)

        # em/poq already computed as em_poq_r, em_poq_i
        u1_r, u1_i = g.c_mul(dem_r, dem_i, poq_div_r, poq_div_i)
        # Actually Y = em/poq * amp0 + ep * amp1
        # dY/dp = d(em/poq)/dp * amp0 + dep/dp * amp1
        # where d(em/poq)/dp = (dem/dp * poq - em * dpoq/dp) / poq²
        # For time params (not poq/poqi): dpoq/dp = 0, so d(em/poq)/dp = dem/dp / poq
        u1_r, u1_i = g.c_mul(dem_r, dem_i, poq_div_r, poq_div_i)
        u2_r, u2_i = g.c_mul(dep_r, dep_i, amp1_r, amp1_i)
        dy_r, dy_i = g.c_add(u1_r, u1_i, u2_r, u2_i)
        return dx_r, dx_i, dy_r, dy_i

    def _grad_tp(dx_r, dx_i, dy_r, dy_i):
        """2 * Re(Σ dQ_dX*dx + dQ_dY*dy)"""
        p_r = g.add(g.sub(g.mul(dX_r, dx_r), g.mul(dX_i, dx_i)),
                    g.sub(g.mul(dY_r, dy_r), g.mul(dY_i, dy_i)))
        return g.mul(two_f, g.reduce_sum(p_r, [0]))

    dx_dgt_r, dx_dgt_i, dy_dgt_r, dy_dgt_i = _dXdY(
        dep_dgt_r, dep_dgt_i, dem_dgt_r, dem_dgt_i)
    dx_ddg_r, dx_ddg_i, dy_ddg_r, dy_ddg_i = _dXdY(
        dep_ddg_r, dep_ddg_i, dem_ddg_r, dem_ddg_i)
    dx_ddm_r, dx_ddm_i, dy_ddm_r, dy_ddm_i = _dXdY(
        dep_ddm_r, dep_ddm_i, dem_ddm_r, dem_ddm_i)

    # poq = poqr * exp(i*poqi)
    dpoq_dpoqr_r = g.cos(poqi)
    dpoq_dpoqr_i = g.sin(poqi)
    dpoq_dpoqi_r = g.neg(g.mul(poqr, g.sin(poqi)))
    dpoq_dpoqi_i = g.mul(poqr, g.cos(poqi))

    def _dXdY_poq(dpoq_r, dpoq_i):
        dX_r = g.mul(g.sub(g.mul(dpoq_r, em_r), g.mul(dpoq_i, em_i)), amp1_r)
        dX_i = g.mul(g.add(g.mul(dpoq_r, em_i), g.mul(dpoq_i, em_r)), amp1_r)
        # Wait, dX/dpoq * dpoq/dp where dX/dpoq = em * amp1
        # dX = d(poq)*em*amp1, so dX/dpoq = em*amp1 (this is a complex derivative)
        # ∂X/∂poq_r = em * amp1  (Wirtinger)
        # ∂X/∂poq_i = i * em * amp1
        # But dpoq/dp is a real derivative, so full chain:
        # dX/dp = dX/dpoq * dpoq/dp (complex * complex for poq)
        # Actually dX = d(poq*em*amp1) = dpoq * em * amp1
        # dpoq = dpoq_dp * dp where dpoq_dp = (dpoq_r/dp + i*dpoq_i/dp)
        # So dX/dp = (dpoq_r/dp + i*dpoq_i/dp) * em * amp1
        return g.c_mul(dpoq_r, dpoq_i, g.c_mul(em_r, em_i, amp1_r, amp1_i))

    # Hmm this is getting really tangled. Let me simplify.

    # For X = ep*amp0 + poq*em*amp1:
    # dX/dpoqr = d(poq)/dpoqr * em * amp1 = (cos(poqi) + i*sin(poqi)) * em * amp1
    # dX/dpoqi = d(poq)/dpoqi * em * amp1 = (-poqr*sin(poqi) + i*poqr*cos(poqi)) * em * amp1

    # For Y = em/poq * amp0 + ep * amp1:
    # dY/dpoqr = em * d(1/poq)/dpoqr * amp0
    # d(1/poq)/dpoqr = -1/poq² * dpoq/dpoqr
    # dY/dpoqi = em * d(1/poq)/dpoqi * amp0
    # d(1/poq)/dpoqi = -1/poq² * dpoq/dpoqi

    # This is a lot. Let me use a simpler approach for poq-related derivatives.

    # For the time params not involving poq, use _dXdY.
    # For poqr, poqi, use separate formulas.

    # grad_gt
    g_gt = _grad_tp(dx_dgt_r, dx_dgt_i, dy_dgt_r, dy_dgt_i)
    g_dg = _grad_tp(dx_ddg_r, dx_ddg_i, dy_ddg_r, dy_ddg_i)
    g_dm = _grad_tp(dx_ddm_r, dx_ddm_i, dy_ddm_r, dy_ddm_i)

    # grad_poqr: dX = dpoq * em * amp1, dY = -em/poq² * dpoq * amp0
    em_amp1_r, em_amp1_i = g.c_mul(em_r, em_i, amp1_r, amp1_i)
    dX_poqr_r, dX_poqr_i = g.c_mul(dpoq_dpoqr_r, dpoq_dpoqr_i,
                                    em_amp1_r, em_amp1_i)
    # For Y: dY = -em/poq² * dpoq * amp0 = -(em/poq) * dpoq/poq * amp0
    # em/poq we already have as em_poq_r, em_poq_i
    # dpoq/poq = dpoq * 1/poq = dpoq * poq_div
    dpoq_dpoqr_poq_r, dpoq_dpoqr_poq_i = g.c_mul(
        dpoq_dpoqr_r, dpoq_dpoqr_i, poq_div_r, poq_div_i)
    # -em/poq * dpoq/poq * amp0
    t_r, t_i = g.c_mul(em_poq_r, em_poq_i, dpoq_dpoqr_poq_r, dpoq_dpoqr_poq_i)
    dY_poqr_r = g.neg(g.mul(t_r, amp0_r))  # Wait, this is wrong. t_r * amp0_r doesn't respect complex mult.
    # Actually: (-em/poq * dpoq/poq) * amp0
    # Let me do: c_mul(c_mul(-one, c_mul(em_poq, dpoq_poq)), amp0)
    neg_em_poq_r, neg_em_poq_i = g.c_rmul(g.neg(one), em_poq_r, em_poq_i)
    t_r, t_i = g.c_mul(neg_em_poq_r, neg_em_poq_i, dpoq_dpoqr_poq_r, dpoq_dpoqr_poq_i)
    dY_poqr_r, dY_poqr_i = g.c_mul(t_r, t_i, amp0_r, amp0_i)

    g_poqr = _grad_tp(dX_poqr_r, dX_poqr_i, dY_poqr_r, dY_poqr_i)

    # grad_poqi — same structure with dpoq_dpoqi
    dX_poqi_r, dX_poqi_i = g.c_mul(dpoq_dpoqi_r, dpoq_dpoqi_i,
                                    em_amp1_r, em_amp1_i)
    dpoq_poqi_poq_r, dpoq_poqi_poq_i = g.c_mul(
        dpoq_dpoqi_r, dpoq_dpoqi_i, poq_div_r, poq_div_i)
    t_r, t_i = g.c_mul(neg_em_poq_r, neg_em_poq_i, dpoq_poqi_poq_r, dpoq_poqi_poq_i)
    dY_poqi_r, dY_poqi_i = g.c_mul(t_r, t_i, amp0_r, amp0_i)
    g_poqi = _grad_tp(dX_poqi_r, dX_poqi_i, dY_poqi_r, dY_poqi_i)

    # grad_ap: P = (1-frac)*(1-ap)*PB + frac*(1+ap)*PBbar
    # dP_dap = -(1-frac)*PB + frac*PBbar
    dP_ap = g.add(g.neg(g.mul(g.sub(one, frac), PB)),
                  g.mul(frac, PBbar))
    g_ap = g.reduce_sum(g.mul(dQ_dP, dP_ap), [0])

    # ================================================================
    #  Outputs
    # ================================================================
    outputs = {"P": P, "Q": Q}
    if with_gradients:
        outputs.update({
            "grad_ck_re": grad_ck_re,
            "grad_ck_im": grad_ck_im,
            "grad_m0": grad_m0,
            "grad_g0": grad_g0,
            "grad_time_params": g.concat(
                [g.reshape(g_gt, [1]), g.reshape(g_dg, [1]),
                 g.reshape(g_dm, [1]), g.reshape(g_poqr, [1]),
                 g.reshape(g_poqi, [1]), g.reshape(g_ap, [1])], axis=0),
        })
        if with_norm:
            outputs["grad_norm"] = grad_norm

    _out_shape = {"P": (_N,), "Q": ()}
    if with_gradients:
        _out_shape.update({
            "grad_ck_re": (nwaves,),
            "grad_ck_im": (nwaves,),
            "grad_m0": (n_m0,),
            "grad_g0": (n_g0,),
            "grad_time_params": (6,),
        })
        if with_norm:
            _out_shape["grad_norm"] = ()
    for name, tname in outputs.items():
        g.node("Identity", [tname], [name])
        g.output(name, F, _out_shape[name])
    return g.build()


def export_to_onnx(config: dict, onnx_path: str, with_norm: bool = True,
                   opset: int = 11, nevt: int | None = None,
                   with_gradients: bool = True):
    """Build, check, and save the ONNX model.

    Parameters
    ----------
    config : dict
        Kernel config dict.
    onnx_path : str
        Output path.
    with_norm : bool
        If True, include ``norm`` input and ``Q = -Σ w·log(P/norm + bkg)``.
    opset : int
        ONNX opset version.  Use 11 for CANN / Ascend compatibility.
    nevt : int or None
        If given, all shapes are fixed (static batch dimension).
        If ``None``, the batch dimension is symbolic.
    """
    model = build_onnx_model(config, with_norm=with_norm, opset=opset,
                             nevt=nevt, with_gradients=with_gradients)
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    return model

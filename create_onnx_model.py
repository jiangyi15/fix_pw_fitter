"""
create_onnx_model.py  –  Build ONNX model for FPW Fitter optimized algorithm.

Implements the optimized computation from DERIVATION.tex (Sec 7, page 8):

The key geometric optimisation: reshape F from (N, j, k) → (N*j, k) so that a
* single * complex MatMul processes all projections simultaneously — no
per‑projection unrolling, works for any j.

Usage:
    python create_onnx_model.py                        # build + save
    python create_onnx_model.py --validate             # + validate vs NumpyFitter
    python create_onnx_model.py --infer                # + run sample inference
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EPS = np.float32(1e-30)
_name_counter = [0]


def _uid(prefix: str) -> str:
    _name_counter[0] += 1
    return f"{prefix}_{_name_counter[0]}"


def _f32_scalar(val: float) -> np.ndarray:
    return np.array(val, dtype=np.float32)


def _const_init(name: str, arr: np.ndarray):
    return numpy_helper.from_array(arr, name)


FLOAT = TensorProto.FLOAT


# ======================================================================
# ONNX graph builder
# ======================================================================

def build_fpwfitter_graph(n_comp: int, n_proj: int, n_events: int | None = None):
    """Build ONNX graph for the FPW fitter  (flattened‑projection).

    When *n_events* is ``None`` the batch dimension is dynamic (symbolic ``N``).
    When it is an integer all shapes are fixed, which lets ONNX Runtime apply
    more aggressive constant-folding and memory-planning.

    Inputs (all float32):
        F_real  : (N, n_proj, n_comp)    F_imag  : (N, n_proj, n_comp)
        w       : (N,)                   B       : (N,)
        M_real  : (n_comp, n_comp)       M_imag  : (n_comp, n_comp)
        N_b     : ()                     purity  : ()
        c_real  : (n_comp,)              c_imag  : (n_comp,)

    Outputs:
        nll       : ()                     grad_real : (n_comp,)
        grad_imag : (n_comp,)
    """
    _name_counter[0] = 0
    dynamic = n_events is None

    # ---- symbolic or fixed N ----
    N_dim = "N" if dynamic else n_events

    # ---- tensor value-info helper ----
    def vi(name, shape, dtype=FLOAT):
        return helper.make_tensor_value_info(name, dtype, shape)

    # ---- node-builder helpers ----
    nodes = []
    def N(op_type, inputs, outputs, **attrs):
        nodes.append(helper.make_node(op_type, inputs, outputs, **attrs))

    def bin_op(op, a, b):
        o = _uid(f"t_{op.lower()}")
        N(op, [a, b], [o]); return o

    def add(a, b):  return bin_op("Add", a, b)
    def sub(a, b):  return bin_op("Sub", a, b)
    def mul(a, b):  return bin_op("Mul", a, b)
    def div(a, b):  return bin_op("Div", a, b)
    def neg(x):
        o = _uid("neg"); N("Neg", [x], [o]); return o
    def log(x):
        o = _uid("log"); N("Log", [x], [o]); return o

    def unsqueeze(x, axes):
        o = _uid("unsq"); N("Unsqueeze", [x], [o], axes=axes); return o
    def squeeze(x, axes):
        o = _uid("sq");   N("Squeeze",   [x], [o], axes=axes); return o

    def rsum(x, axes=None, keepdims=0):
        o = _uid("rsum")
        kw = {"keepdims": keepdims}
        if axes is not None: kw["axes"] = axes
        N("ReduceSum", [x], [o], **kw); return o

    # ---- reshape shapes — use -1 for dynamic, explicit for fixed ----
    f_flat_0 = -1 if dynamic else (n_events * n_proj)
    a_0      = -1 if dynamic else n_events
    g_flat_0 = -1 if dynamic else (n_events * n_proj)

    shape_f_flat = np.array([f_flat_0, n_comp], dtype=np.int64)
    shape_a      = np.array([a_0,      n_proj], dtype=np.int64)
    shape_g_flat = np.array([g_flat_0, 1],      dtype=np.int64)

    initializers = [
        _const_init("eps",          _f32_scalar(_EPS)),
        _const_init("one",          _f32_scalar(1.0)),
        _const_init("shape_f_flat", shape_f_flat),
        _const_init("shape_a",      shape_a),
        _const_init("shape_g_flat", shape_g_flat),
    ]

    # ---- graph inputs / outputs ----
    inputs = [
        vi("F_real",  [N_dim, n_proj, n_comp]),
        vi("F_imag",  [N_dim, n_proj, n_comp]),
        vi("w",       [N_dim]),
        vi("B",       [N_dim]),
        vi("M_real",  [n_comp, n_comp]),
        vi("M_imag",  [n_comp, n_comp]),
        vi("N_b",     []),
        vi("purity",  []),
        vi("c_real",  [n_comp]),
        vi("c_imag",  [n_comp]),
    ]
    outputs = [
        vi("nll",       []),
        vi("grad_real", [n_comp]),
        vi("grad_imag", [n_comp]),
    ]

    # ================================================================
    # 1.  Flatten F: (N, j, k) → (N*j, k)    — single MatMul for all j
    # ================================================================
    F_flat_r = _uid("Ff_r");  N("Reshape", ["F_real", "shape_f_flat"], [F_flat_r])
    F_flat_i = _uid("Ff_i");  N("Reshape", ["F_imag", "shape_f_flat"], [F_flat_i])

    # ================================================================
    # 2.  A = F @ c   (complex MatVec, flat)
    #     A_flat_r = F_flat_r @ c_r  –  F_flat_i @ c_i     → (N*j, 1)
    #     A_flat_i = F_flat_r @ c_i  +  F_flat_i @ c_r     → (N*j, 1)
    # ================================================================
    c_r_2d = unsqueeze("c_real", [1])          # (k, 1)
    c_i_2d = unsqueeze("c_imag", [1])

    m0 = _uid("a_m0"); N("MatMul", [F_flat_r, c_r_2d], [m0])
    m1 = _uid("a_m1"); N("MatMul", [F_flat_i, c_i_2d], [m1])
    m2 = _uid("a_m2"); N("MatMul", [F_flat_r, c_i_2d], [m2])
    m3 = _uid("a_m3"); N("MatMul", [F_flat_i, c_r_2d], [m3])

    A_flat_r = _uid("Af_r"); N("Sub", [m0, m1], [A_flat_r])
    A_flat_i = _uid("Af_i"); N("Add", [m2, m3], [A_flat_i])

    # Unflatten A: (N*j, 1) → (N, j)
    A_r = _uid("A_r"); N("Reshape", [A_flat_r, "shape_a"], [A_r])
    A_i = _uid("A_i"); N("Reshape", [A_flat_i, "shape_a"], [A_i])

    # ================================================================
    # 3.  S = sum_j |A|²   →  (N,)
    # ================================================================
    S = rsum(add(mul(A_r, A_r), mul(A_i, A_i)), axes=[1])

    # ================================================================
    # 4.  N_s = real(c^H M c)   — scalar
    # ================================================================
    m4 = _uid("m_m0"); N("MatMul", ["M_real", c_r_2d], [m4])
    m5 = _uid("m_m1"); N("MatMul", ["M_imag", c_i_2d], [m5])
    m6 = _uid("m_m2"); N("MatMul", ["M_real", c_i_2d], [m6])
    m7 = _uid("m_m3"); N("MatMul", ["M_imag", c_r_2d], [m7])

    mc_r_2d = _uid("mc_r"); N("Sub", [m4, m5], [mc_r_2d])
    mc_i_2d = _uid("mc_i"); N("Add", [m6, m7], [mc_i_2d])

    mc_r = squeeze(mc_r_2d, axes=[1])          # (k,)
    mc_i = squeeze(mc_i_2d, axes=[1])

    Ns = rsum(add(mul("c_real", mc_r), mul("c_imag", mc_i)), keepdims=0)

    # ================================================================
    # 5.  P  and  nll   (with clipping for stability)
    # ================================================================
    Ns_c = _uid("Ns_c"); N("Clip", [Ns, "eps"], [Ns_c])
    Nb_c = _uid("Nb_c"); N("Clip", ["N_b", "eps"], [Nb_c])

    S_div_Ns = div(S, Ns_c)
    B_div_Nb = div("B", Nb_c)
    one_m_p  = sub("one", "purity")
    P = add(mul("purity", S_div_Ns), mul(one_m_p, B_div_Nb))

    P_c = _uid("P_c"); N("Clip", [P, "eps"], [P_c])

    log_P  = log(P_c)
    w_log_P = mul("w", log_P)
    nll    = neg(rsum(w_log_P, keepdims=0))

    # ================================================================
    # 6.  Gradient — ratio & weighted A
    # ================================================================
    ratio   = div("w", P_c)                   # (N,)
    ratio_2d = unsqueeze(ratio, [1])           # (N, 1)
    G_r = mul(ratio_2d, A_r)                  # (N, j)
    G_i = mul(ratio_2d, A_i)

    # ================================================================
    # 7.  g_data = F_flat^H @ G_flat   (single batched GEMM over all j)
    #
    #     F_flat_r  (N*j, k)  →  F_flat_r^T  (k, N*j)
    #     G_flat_r  (N*j, 1)
    #     ─────────────────────────────────────────────────
    #     g_data_r = F_flat_r^T @ G_flat_r  +  F_flat_i^T @ G_flat_i   (k,1)
    #     g_data_i = F_flat_r^T @ G_flat_i  –  F_flat_i^T @ G_flat_r   (k,1)
    # ================================================================
    G_flat_r = _uid("Gf_r"); N("Reshape", [G_r, "shape_g_flat"], [G_flat_r])
    G_flat_i = _uid("Gf_i"); N("Reshape", [G_i, "shape_g_flat"], [G_flat_i])

    Ff_T_r = _uid("FfT_r"); N("Transpose", [F_flat_r], [Ff_T_r], perm=[1, 0])
    Ff_T_i = _uid("FfT_i"); N("Transpose", [F_flat_i], [Ff_T_i], perm=[1, 0])

    g_r1 = _uid("g_r1"); N("MatMul", [Ff_T_r, G_flat_r], [g_r1])
    g_r2 = _uid("g_r2"); N("MatMul", [Ff_T_i, G_flat_i], [g_r2])
    g_i1 = _uid("g_i1"); N("MatMul", [Ff_T_r, G_flat_i], [g_i1])
    g_i2 = _uid("g_i2"); N("MatMul", [Ff_T_i, G_flat_r], [g_i2])

    g_data_r_2d = _uid("gd_r"); N("Add", [g_r1, g_r2], [g_data_r_2d])
    g_data_i_2d = _uid("gd_i"); N("Sub", [g_i1, g_i2], [g_data_i_2d])

    g_data_r = squeeze(g_data_r_2d, axes=[1])   # (k,)
    g_data_i = squeeze(g_data_i_2d, axes=[1])

    # ================================================================
    # 8.  dN_s / dc*  (reuse mc_r / mc_i)
    #     S_corr = sum(w * S / P)
    # ================================================================
    S_div_P   = div(S, P_c)
    w_S_div_P = mul("w", S_div_P)
    S_corr    = rsum(w_S_div_P, keepdims=0)

    # ================================================================
    # 9.  Assemble gradient
    #     grad = –p/Ns · g_data  +  p/Ns² · dNs · S_corr
    # ================================================================
    p_over_Ns   = div("purity", Ns_c)
    neg_p_over  = neg(p_over_Ns)
    Ns_sq       = mul(Ns_c, Ns_c)
    p_over_Ns2  = div("purity", Ns_sq)

    dNs_Sc_r = mul(mc_r, S_corr)       # (k,)
    dNs_Sc_i = mul(mc_i, S_corr)

    grad_r = add(mul(neg_p_over, g_data_r), mul(p_over_Ns2, dNs_Sc_r))
    grad_i = add(mul(neg_p_over, g_data_i), mul(p_over_Ns2, dNs_Sc_i))

    # ================================================================
    # 10.  Rename outputs  (Identity for fixed output names)
    # ================================================================
    N("Identity", [nll],    ["nll"])
    N("Identity", [grad_r], ["grad_real"])
    N("Identity", [grad_i], ["grad_imag"])

    # ================================================================
    # Assemble
    # ================================================================
    graph = helper.make_graph(nodes, "FpwFitterONNX",
                              inputs, outputs, initializer=initializers)
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", 11)],
                              producer_name="fpwfitter", producer_version="0.1.0")
    model.ir_version = onnx.IR_VERSION_2021_7_30
    return model


# ======================================================================
# Validation helpers
# ======================================================================

def numpy_reference(F_data, w_data, B_data, M, N_b, c, purity):
    """Standard NumPy reference (double precision)."""
    A = np.einsum("ijk,k->ij", F_data, c)
    S = np.sum(np.abs(A) ** 2, axis=1)
    Ns = np.real(np.vdot(c, M @ c))
    if Ns < 1e-300:
        Ns = 1e-300
    if N_b < 1e-300:
        N_b = 1e-300
    P = S / Ns * purity + B_data / N_b * (1 - purity)
    P = np.maximum(P, 1e-300)
    nll = -float(np.dot(w_data, np.log(P)))

    ratio = w_data / P
    G = A * ratio[:, None]
    g_data = np.einsum("ijk,ij->k", np.conj(F_data), G)
    dNs_dc = M @ c
    S_corr = np.sum(w_data * S / P)
    grad = -purity / Ns * g_data + purity / Ns ** 2 * dNs_dc * S_corr
    return np.float64(nll), grad


def run_ort_session(model, feeds):
    """Run ONNX Runtime session and return outputs dict."""
    import onnxruntime as ort
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    return dict(zip(out_names, sess.run(out_names, feeds)))


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Build FPW Fitter ONNX model")
    parser.add_argument("--n-comp", type=int, default=10)
    parser.add_argument("--n-proj", type=int, default=2)
    parser.add_argument("--n-events", "--fixed-n", type=int, default=None,
                        help="Fix N (events).  Omit or 0/negative for dynamic N.")
    parser.add_argument("--output", "-o", type=str, default="fpwfitter.onnx")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--infer", action="store_true")
    args = parser.parse_args()

    n_events = args.n_events if args.n_events is not None and args.n_events > 0 else None
    model = build_fpwfitter_graph(args.n_comp, args.n_proj, n_events)
    onnx.save(model, args.output)
    mb = Path(args.output).stat().st_size / 1e6
    print(f"Model saved to {args.output}  ({mb:.1f} MB)")
    print(f"  Inputs:  {[i.name for i in model.graph.input]}")
    print(f"  Outputs: {[o.name for o in model.graph.output]}")

    onnx.checker.check_model(model)
    print("  ONNX checker: passed")

    # Count ops
    from collections import Counter
    ops = Counter(n.op_type for n in model.graph.node)
    print(f"  Total nodes: {len(model.graph.node)}")
    for op, cnt in sorted(ops.items(), key=lambda x: -x[1]):
        print(f"    {op}: {cnt}")

    if args.validate or args.infer:
        print("\n--- Validation ---")
        rng = np.random.RandomState(42)
        k, j, N = args.n_comp, args.n_proj, n_events or 1000

        # Build M from MC data = guaranteed PSD
        F_mc = (rng.randn(N * 5, j, k).astype(np.float32) +
                1j * rng.randn(N * 5, j, k).astype(np.float32))
        w_mc = np.abs(rng.randn(N * 5).astype(np.float32))
        M = sum((F_mc[:, p, :].conj().T @ (w_mc[:, None] * F_mc[:, p, :]))
                for p in range(j))

        F = (rng.randn(N, j, k).astype(np.float32) +
             1j * rng.randn(N, j, k).astype(np.float32))
        w = np.abs(rng.randn(N).astype(np.float32))
        B = np.abs(rng.randn(N).astype(np.float32))
        N_b = float(np.dot(w, B))
        c = (rng.randn(k).astype(np.float32) +
             1j * rng.randn(k).astype(np.float32))
        purity = np.float32(0.8)

        # NumPy reference (FP64)
        nll_ref, grad_ref = numpy_reference(
            F.astype(np.complex128), w.astype(np.float64), B.astype(np.float64),
            M.astype(np.complex128), N_b, c.astype(np.complex128), float(purity))

        # ONNX Runtime (FP32)
        feeds = {
            "F_real": F.real, "F_imag": F.imag, "w": w, "B": B,
            "M_real": M.real, "M_imag": M.imag,
            "N_b": np.array(N_b, dtype=np.float32),
            "purity": np.array(purity),
            "c_real": c.real, "c_imag": c.imag,
        }
        ort_out = run_ort_session(model, feeds)
        nll_ort = ort_out["nll"]
        grad_ort = ort_out["grad_real"] + 1j * ort_out["grad_imag"]

        nll_ok = abs(nll_ort - nll_ref) / max(abs(nll_ref), 1.0) < 1e-4
        g_err = (np.linalg.norm(grad_ort - grad_ref) /
                 max(np.linalg.norm(grad_ref), 1e-30))
        g_ok = g_err < 1e-2

        print(f"  N_ref={nll_ref:.6f}  N_ort={nll_ort:.6f}  "
              f"diff={abs(nll_ort-nll_ref):.2e}")
        print(f"  Grad rel err: {g_err:.2e}")
        print(f"  {'PASS' if (nll_ok and g_ok) else 'FAIL'}")

        if args.infer:
            print(f"\n  n_data={N}, n_comp={k}, n_proj={j}")
            print(f"  NLL={nll_ort:.6f}  ||grad||={np.linalg.norm(grad_ort):.4e}")


if __name__ == "__main__":
    main()

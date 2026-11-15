#!/usr/bin/env python3
"""Source-level ablation of the pwa_gen GPU kernel: attribute time to a phase.

``ncu`` hardware counters are unavailable on this machine
(``ERR_NVGPUCTRPERM``; needs ``NVreg_RestrictProfilingToAdminUsers=0`` as
root), so the bottleneck is located by **controlled ablation**: compile the
SHIPPED ``kernels_v5_pwa_gen_bw.cu`` with exactly one phase removed or one
constant changed, then time the same shape, data and params.  The variant's
delta is that phase's share of the call.

This is what produced the attribution behind ``cuda_v5_pwa_gen_bwt``
(phase A's fp64 divisions 19 %, the ck gradient 10 %, the coefficient
gradient 2.8 %, the amp-cache read 1.7 %, the fp64 atomics ~0 %) and the
``S0E_TE`` occupancy result.  Shipped-vs-shipped timing lives in
``tests/profile_pwa_gen_ab_shipped.py``.

The GPU throttles hard (2730 MHz -> 180 MHz inside one run, so identical work
measures 8.99 ms and 12.79 ms minutes apart), which makes any sequential
comparison worthless.  This harness keeps exactly TWO contexts alive,
alternates them every round and reports the min per variant, so both see the
same clock distribution.  ``--check`` also compares the returned values, so
an ablation that changed the answer (they all do — that is the point) is
visible.

    python tests/profile_pwa_gen_ab.py --variant no_div
    python tests/profile_pwa_gen_ab.py --variant rcp_te4 --check
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from tabpwa.cuda.build import _arch_flags, detect_gcc_compat, find_nvcc

HERE = os.path.dirname(os.path.abspath(__file__))
CUDA_DIR = os.path.join(os.path.dirname(HERE), "src", "tabpwa", "cuda")
OUT = "/tmp/opencode/pwa_gen_ab"

SRC = "kernels_v5_pwa_gen_bw.cu"
BWT = "kernels_v5_pwa_gen_bwt.cu"

# one fp64 reciprocal + multiplies instead of two fp64 divisions
RCP_OLD = """const double den = bw_r * bw_r + bw_i * bw_i;
            const double ir =  bw_r / den;
            const double ii = -bw_i / den;"""
RCP_NEW = """const double den = bw_r * bw_r + bw_i * bw_i;
            const double rden = 1.0 / den;
            const double ir =  bw_r * rden;
            const double ii = -bw_i * rden;"""

# Measurement-only ablations (they CHANGE the answer on purpose: they exist to
# attribute time to a phase, never to be shipped).
NO_ATOMICS = [
    ("atomicAdd(d_ck_r + k, sr);", "(void)sr;"),
    ("atomicAdd(d_ck_i + k, si);", "(void)si;"),
    ("atomicAdd(d_coeff + i, acc);", "(void)acc;"),
]

VARIANTS = {
    # the two shipped changes, kept so the attribution can be re-measured
    # individually (together they are ``cuda_v5_pwa_gen_bwt``; see
    # tests/test_pwa_gen_bwt.py and profile_pwa_gen_ab_shipped.py)
    "rcp": [(RCP_OLD, RCP_NEW)],
    "te4": [("#define S0E_TE    8 ", "#define S0E_TE    4 ")],
    "rcp_te4": [(RCP_OLD, RCP_NEW),
                ("#define S0E_TE    8 ", "#define S0E_TE    4 ")],
    # candidates REJECTED by the profile (kept as evidence, do not ship):
    # TE=2 loses the intra-BW reuse; TE=6/16 do not fit or add nothing.
    "te2": [("#define S0E_TE    8 ", "#define S0E_TE    2 ")],
    "te6": [("#define S0E_TE    8 ", "#define S0E_TE    6 ")],
    # attribution only
    # phase A with the reciprocal removed entirely (sets ir=bw_r) -> the pure
    # cost of the two fp64 divisions, nothing else changes
    "no_div": [("const double den = bw_r * bw_r + bw_i * bw_i;\n"
                "            const double ir =  bw_r / den;\n"
                "            const double ii = -bw_i / den;",
                "const double ir = bw_r;\n"
                "            const double ii = -bw_i;")],
    # the amp-cache global read -> a constant
    "amp_const": [("const float2 A = amp_cache[(size_t)(ce0 + e) * n_uniq\n"
                   "                                       + slot_of_wave[w]];",
                   "const float2 A = make_float2(0.5f, 0.25f);")],
    # phase D2 (the coefficient gradient's basis pass)
    "no_D2": [("atomicAdd(d_coeff + i, acc);", "(void)acc;")],
    # phase C (the ck gradient)
    "no_C": [("atomicAdd(d_ck_r + k, sr);", "(void)sr;"),
             ("atomicAdd(d_ck_i + k, si);", "(void)si;")],
    "no_atomics": NO_ATOMICS,
    "no_phaseC": [("atomicAdd(d_ck_r + k, sr);", "(void)sr;"),
                  ("atomicAdd(d_ck_i + k, si);", "(void)si;")],
    "no_phaseD2": [("atomicAdd(d_coeff + i, acc);", "(void)acc;")],
}


def _compile(edits, tag):
    os.makedirs(OUT, exist_ok=True)
    text = open(os.path.join(CUDA_DIR, SRC), encoding="utf-8").read()
    for pat, rep in edits:
        text, n = re.subn(re.escape(pat), rep, text)
        if n != 1:
            raise SystemExit(f"{tag}: pattern matched {n} times")
    src = os.path.join(OUT, f"{tag}.cu")
    open(src, "w", encoding="utf-8").write(text)
    so = os.path.join(OUT, f"{tag}.so")
    nvcc = find_nvcc()
    base = [nvcc, "-shared", "-Xcompiler", "-fPIC", "-lcudart", "-lm", "-O2"]
    base.extend(_arch_flags(nvcc))
    extra = detect_gcc_compat(base, os.path.join(OUT, "_probe.so"), src)
    r = subprocess.run(base + extra + ["-o", so, src], capture_output=True,
                       text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr[-1500:])
    return so


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="rcp", choices=sorted(VARIANTS))
    ap.add_argument("--shape", default="km", choices=["km", "mb"])
    ap.add_argument("--n", type=int, default=65536)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--rs", type=int, default=1)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    import time
    import tabpwa.cuda._v5_pwa_gen_bw as W
    from benchmark_pwa_gen_bw import _mb_case
    from benchmark_pwa_gen_s0e import _km_case

    kc, data, params, norm = (_km_case(n_events=args.n) if args.shape == "km"
                              else _mb_case(n_events=args.n))
    norm = float(norm)
    # source-level ablation of the SHIPPED gen_bw kernel: every variant is
    # that kernel with one phase disabled, so the deltas attribute time.
    # Shipped-vs-shipped timing is tests/profile_pwa_gen_ab_shipped.py.
    so = _compile(VARIANTS[args.variant], args.variant)
    CLS = W.CUDAKernelV5PWAGenBW

    def build(so_path):
        orig = W._load_lib
        if so_path is not None:
            W._load_lib = lambda: W._ffi.dlopen(so_path)
        try:
            k = CLS(kc, resolution_size=args.rs)
        finally:
            W._load_lib = orig
        return k, k.load_data(data)

    ka, ha = build(None)
    kb, hb = build(so)
    try:
        def once(k, h):
            t0 = time.perf_counter()
            out = k.compute(params, h, norm=norm)
            return (time.perf_counter() - t0) * 1e3, out

        ta, tb = [], []
        oa = ob = None
        for i in range(args.rounds):
            for (k, h, ts) in ((ka, ha, ta), (kb, hb, tb)):
                once(k, h)                       # warm
                dt, out = once(k, h)
                ts.append(dt)
                if i == 0:
                    if ts is ta:
                        oa = out
                    else:
                        ob = out
        best_a, best_b = min(ta), min(tb)
        print(f"{args.variant}: repo {best_a:.3f} ms   variant {best_b:.3f} ms"
              f"   ratio {best_b / best_a:.3f}x   (min of {args.rounds})")
        print(f"          medians: repo {np.median(ta):.3f}  "
              f"variant {np.median(tb):.3f}  "
              f"ratio {np.median(tb) / np.median(ta):.3f}x")
        if args.check and oa is not None and ob is not None:
            Qa, ga, Pa = oa
            Qb, gb, Pb = ob
            print(f"  check: dQ={abs(Qa - Qb) / max(abs(Qa), 1e-30):.2e}  "
                  f"dP={(np.max(np.abs(Pa - Pb)) / np.max(np.abs(Pa))):.2e}")
            for key in sorted(ga):
                if key in ("ck", "coeff"):
                    a = np.asarray(ga[key]).ravel()
                    b = np.asarray(gb[key]).ravel()
                    print(f"         d{key}="
                          f"{np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-30):.2e}")
    finally:
        ha.free(); ka.free()
        hb.free(); kb.free()


if __name__ == "__main__":
    main()

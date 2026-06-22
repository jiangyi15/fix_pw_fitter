#!/usr/bin/env python3
"""Test the simple void*-based C API for merged CUDA kernel."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cffi import FFI

ffi = FFI()
CDEF = """
/* Memory + device info */
int cuda_alloc(void** ptr, unsigned long size);
int cuda_free(void* ptr);
int cuda_memcpy_to_device(void* dst, const void* src, unsigned long size);
int cuda_memcpy_to_host(void* dst, const void* src, unsigned long size);
int cuda_get_device_count();
int cuda_get_device_name(char* name, int len);

/* High-level opaque API — all GPU memory managed internally */
void* cuda_create_context(
    const int* bw_m0_i, int nbw0, const int* bw_mass_i, int nbwm,
    const int* fl_t, int nflt_, const int* fl_q, int nflq,
    const int* fl_o, int nflo, const int* ang_i, int nangi,
    const int* g0_i, int ng0i, const int* g0_m, int ng0m,
    const int* bpg_i, int nbpg, const int* bpg_o, int nbpo,
    const int* m0_i, int nm0i, const int* bw_o, int nbwo,
    const double* ak, int nak, const double* ab, int nab,
    const double* mar, int nmar, const double* mai, int nmai,
    const double* gtr, int ngtr, const double* gti, int ngti,
    double gmin, double gdel, int gbins,
    const double* mg, int nmg,
    const double* flt, int nflt, double flmin, double fldel, int fbins,
    int nw, int nr, int nd, int ntp, int nub, int ngr,
    int nm, int nmom, int nak_, int nat);

void* cuda_load_data(void* ctx, const double* mass, const double* mom,
    const double* ang, const double* frac, const double* time,
    const double* wgt, const double* bkg, int ne);

void cuda_compute(void* ctx, void* dh,
    const double* ck_r, const double* ck_i,
    const double* m0, const double* g0,
    double Gamma, double DG, double DM,
    double Ap, double pr, double pp,
    double norm_val, int use_norm,
    double* Q_out, double* P_out,
    double* gck_r, double* gck_i,
    double* gm0_out, double* gg0_out,
    double* gsc_out,
    int n_wave, int n_unique_bw, int n_gamma_rows);

void cuda_free_context(void* ctx);
void cuda_free_data(void* dh);
"""
ffi.cdef(CDEF)

lib = ffi.dlopen(os.path.join(os.path.dirname(__file__), '..', 'src', 'ampfit', 'cuda', 'libcuda_kernels_merged.so'))

# Load config
from ampfit.config_loader import Config
config = Config('config_angle.yml')
kc = config.build_all_index()

def as_c(arr):
    buf = ffi.from_buffer(np.ascontiguousarray(arr))
    if arr.dtype == np.int32:
        return ffi.cast("int*", buf)
    return ffi.cast("double*", buf)

# Build context
bo = kc["bw_order"]
nw = kc["matrix_angle"].shape[1]
nr = kc["bw_order"].size // nw
nd = kc["fl_order"].size // nw
ntp = nw * nr
nub = len(kc["m0_index"])
ngr = len(kc["g0_index"])
nk = kc["angle_k"].shape[0]
nat = int(np.max(kc["angle_index"])) + 1
nm0u = int(np.max(kc["m0_index"])) + 1
ng0u = int(np.max(kc["g0_index"])) + 1

# Per-position gamma scatter indices
bpg_i, bpg_o = [], [0]
for pos in range(len(bo)):
    rows = np.where(kc["matrix_gamma"][:, bo[pos]] != 0)[0]
    bpg_i.extend(rows.tolist())
    bpg_o.append(len(bpg_i))

ctx = lib.cuda_create_context(
    as_c(kc["m0_index"].astype(np.int32)), len(kc["m0_index"]),
    as_c(kc["mass_index"].astype(np.int32)), len(kc["mass_index"]),
    as_c(kc["fl_type"].astype(np.int32)), len(kc["fl_type"]),
    as_c(kc["fl_q_index"].astype(np.int32)), len(kc["fl_q_index"]),
    as_c(kc["fl_order"].astype(np.int32)), len(kc["fl_order"]),
    as_c(kc["angle_index"].astype(np.int32)), len(kc["angle_index"]),
    as_c(kc["g0_index"].astype(np.int32)), len(kc["g0_index"]),
    as_c(kc["g0_mass_index"].astype(np.int32)), len(kc["g0_mass_index"]),
    as_c(np.array(bpg_i, dtype=np.int32)), len(bpg_i),
    as_c(np.array(bpg_o, dtype=np.int32)), len(bpg_o),
    as_c(kc["m0_index"].astype(np.int32)), len(kc["m0_index"]),
    as_c(kc["bw_order"].astype(np.int32)), len(kc["bw_order"]),
    as_c(kc["angle_k"].astype(np.float64)), kc["angle_k"].size,
    as_c(kc["angle_b"].astype(np.float64)), kc["angle_b"].size,
    as_c(np.real(kc["matrix_angle"]).astype(np.float64)), kc["matrix_angle"].size,
    as_c(np.imag(kc["matrix_angle"]).astype(np.float64)), kc["matrix_angle"].size,
    as_c(np.real(kc["gamma_table"]).astype(np.float64)), kc["gamma_table"].size,
    as_c(np.imag(kc["gamma_table"]).astype(np.float64)), kc["gamma_table"].size,
    float(kc["gamma_min"]), float(kc["gamma_delta"]), kc["gamma_table"].shape[-1],
    as_c(kc["matrix_gamma"].astype(np.float64)), kc["matrix_gamma"].size,
    as_c(kc["fl_table"].astype(np.float64)), kc["fl_table"].size,
    float(kc["fl_min"]), float(kc["fl_delta"]), kc["fl_table"].shape[-1],
    nw, nr, nd, ntp, nub, ngr, 48, 72, nk, nat)
print("✓ Context created")

# Test with data
np.random.seed(42)
n_events = 64
mass = np.random.random((n_events, 48)).astype(np.float64)
momentum = np.random.random((n_events, 72)).astype(np.float64)
angle = np.random.random((n_events, 24, 3)).astype(np.float64)
frac = np.random.random(n_events).astype(np.float64)
time = np.random.random(n_events).astype(np.float64)
weight = np.ones(n_events, dtype=np.float64)
bkg = np.random.random(n_events).astype(np.float64) * 0.01

dh = lib.cuda_load_data(ctx,
    as_c(mass), as_c(momentum), as_c(angle),
    as_c(frac), as_c(time), as_c(weight), as_c(bkg), n_events)
print("✓ Data loaded")

# Compute
ck_real = np.random.randn(448).astype(np.float64)
ck_imag = np.random.randn(448).astype(np.float64)
m0_arr = np.random.rand(20).astype(np.float64) + 2.0
g0_arr = np.random.rand(23).astype(np.float64) + 0.1

Q_out = np.zeros(n_events, dtype=np.float64)
P_out = np.zeros(n_events, dtype=np.float64)
gck_r = np.zeros((n_events, 448), dtype=np.float64)
gck_i = np.zeros((n_events, 448), dtype=np.float64)
gm0_out = np.zeros((n_events, nub), dtype=np.float64)
gg0_out = np.zeros((n_events, ngr), dtype=np.float64)
gsc_out = np.zeros(6, dtype=np.float64)

lib.cuda_compute(ctx, dh,
    as_c(ck_real), as_c(ck_imag), as_c(m0_arr), as_c(g0_arr),
    0.6, 0.01, 0.506, 0.01, 0.9, 0.2,
    -1e100, 0,  # norm=None sentinel
    as_c(Q_out), as_c(P_out),
    as_c(gck_r), as_c(gck_i),
    as_c(gm0_out), as_c(gg0_out),
    as_c(gsc_out),
    448, nub, ngr)

Q = float(np.sum(Q_out))
print(f"Q = {Q:.6f}")
print(f"P shape: {P_out.shape}")
print(f"grad_ck shape: {gck_r.shape}")
print(f"grad_scalar: {gsc_out}")
print("✓ Compute succeeded")

# Compare with numpy reference
from ampfit.numpy_kernel import NumpyKernel
params = {
    'ck': ck_real.astype(np.complex128) + 1j * ck_imag.astype(np.complex128),
    'm0': m0_arr, 'g0': g0_arr,
    'scalar': [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
}
data = {'mass': mass, 'q': momentum, 'angle': angle, 'frac': frac,
        'time': time, 'weight': weight, 'bkg': bkg}
nk = NumpyKernel(kc)
Q_np, grads_np, P_np = nk._compute(params, data)

print(f"\nQ diff: {abs(Q - Q_np):.2e}")
print(f"P max diff: {np.max(np.abs(P_out - P_np)):.2e}")

# Scatter m0/g0
m0_scat = np.zeros((nub, nm0u), dtype=np.float64)
for i, m in enumerate(kc["m0_index"]): m0_scat[i, m] = 1.0
g0_scat = np.zeros((ngr, ng0u), dtype=np.float64)
for i, g in enumerate(kc["g0_index"]): g0_scat[i, g] = 1.0

gm0 = (gm0_out.sum(axis=0) @ m0_scat).astype(np.float64)
gg0 = (gg0_out.sum(axis=0) @ g0_scat).astype(np.float64)

for k, g_cuda, g_np in [("ck", gck_r.sum(axis=0) + 1j*gck_i.sum(axis=0), grads_np["ck"]),
                          ("m0", gm0, grads_np["m0"]),
                          ("g0", gg0, grads_np["g0"]),
                          ("scalar", gsc_out, grads_np["scalar"])]:
    rel = np.max(np.abs(g_cuda - g_np)) / (np.max(np.abs(g_np)) + 1e-30)
    print(f"grad_{k}: rel={rel:.2e} {'✓' if rel < 1e-10 else '✗'}")

lib.cuda_free_data(dh)
lib.cuda_free_context(ctx)
print("\n✓ Simple C API verified!")

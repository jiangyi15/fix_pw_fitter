"""
CUDA kernel ``cuda_polarised`` — projection-sum PWA with a fitted
spin-density matrix (standalone sibling of ``cuda_v4_pwa``).

Same ampcache infrastructure, BW/m0/g0 chains and batching as
``cuda_v4_pwa``; shared-ck forward with ``n_wave = n_proj·N`` (NO config
duplication), but the reduce contracts the per-projection amplitudes with
the spin-density matrix:

    P(e) = Re Σ_{p,p'} rho[p,p'] · A_p(e) · conj(A_p')
    dQ/dA_p = dQ/dP · Σ_{p'} rho[p,p'] · conj(A_p')
    dQ/dp_l = Σ_e dQ/dP · Re Σ_{j,j'} G_l[j,j']·A_j·conj(A_j')

``rho = I + Σ_l scalar[l]·gen_l`` is built HOST-side per call from the
polarisation parameters; the generator basis ``gen`` (n_gen = P²−1
Hermitian P×P matrices) is uploaded once at context creation.  The kernel
returns the FINISHED per-parameter scalar gradients — no raw dQ/dρ.

With ``scalar = 0`` (rho = I) this reproduces ``cuda_v4_pwa`` exactly.
No Gram-matrix entry point: the ρ model normalises via the full-forward
phase-space pass.
"""

import os
import numpy as np
from cffi import FFI

_ffi = FFI()
_ffi.cdef("""
void* cuda_create_context_polarised(
    const int* m0_i,int n1, const int* g0_i,int n2,
    const int* g0_m,int n3, const int* mass_i,int n4,
    const int* fl_t,int n5, const int* fl_q,int n6,
    const int* bw_o,int n7, const int* fl_o,int n8,
    const int* ang_i,int n9,
    const double* ak,int n10, const double* ab,int n11,
    const double* mar,int n12, const double* mai,int n13,
    const double* gtr,int n14, const double* gti,int n15,
    double gmin,double gdel,int gbins,
    const double* mg,int n16,
    const int* gci,int ngci,
    const double* ft,int n17, double flmin,double fldel,int fbins,
    int nw,int nr,int nd,int nub,int ngr,
    int nm,int nmom,int nak_,int nat,int nac,
    int n_m0p,int n_g0p,
    int batch_size,
    const int* slot_of_wave,int n_slot,
    const int* rep_of_slot,int n_rep,
    int n_uniq,int n_proj,
    const double* gen_re,int n_re_,
    const double* gen_im,int n_im_,
    int n_gen);
void cuda_free_context_polarised(void*);
void* cuda_load_data_polarised(void*,const double*,int,const double*,int,
    const double*,int,const double*,const double*,int);
void cuda_free_data_polarised(void*);
void cuda_compute_polarised(void*,void*,
    const double*,const double*,const double*,const double*,
    const double*,const double*,
    double,int,
    double*,double*,double*,double*,double*,double*,double*,
    double*);
int cuda_get_device_count();
int cuda_get_device_name(char*,int);
""")


def _load_lib():
    from tabpwa.cuda.build import ensure
    lib_path = ensure("kernels_polarised.cu", "libcuda_polarised.so")
    if not lib_path:
        raise RuntimeError(
            f"polarised CUDA library not found at {lib_path}.")
    return _ffi.dlopen(lib_path)


class DataHandle:
    def __init__(self, ptr, lib, ne):
        self.ptr = ptr
        self._lib = lib
        self.ne = ne
        self._keep = []

    def free(self):
        if self.ptr is not None:
            self._lib.cuda_free_data_polarised(self.ptr)
            self.ptr = None
            self._keep.clear()

    def __del__(self):
        self.free()


class CUDAKernelPolarised:
    """Projection-sum PWA kernel with a fitted spin-density matrix.

    Standalone sibling of ``CUDAKernelV4PWA`` for the ``polarised_pwa``
    model: shared-ck forward (ck length N = n_wave/n_proj, no config
    duplication), reduce contracts the per-projection amplitudes with
    ``rho`` (built host-side from ``params["scalar"]`` + the generator
    basis *gen* given here) and returns the FINISHED per-parameter
    ``scalar`` gradients.  ``params`` needs ``ck``/``m0``/``g0``/
    ``scalar`` (length P²−1).
    """

    def __init__(self, config, gen, batch_size=50000, lib_path=None):
        self._ctx = None
        self._lib = None
        self._ka = []
        self._lib = _load_lib()
        n_dev = self._lib.cuda_get_device_count()
        if n_dev > 0:
            name_buf = _ffi.new("char[256]")
            self._lib.cuda_get_device_name(name_buf, 256)
            print(f"polarised GPU: {_ffi.string(name_buf).decode()}")
        else:
            raise RuntimeError("No CUDA devices found")

        gen = np.asarray(gen, dtype=complex)
        if gen.ndim != 3 or gen.shape[0] == 0:
            raise ValueError(
                "cuda_polarised: gen must be a (n_gen, P, P) complex array, "
                f"got shape {gen.shape}")

        c = config
        self.config = c
        self.gen = gen                      # (n_gen, P, P) rho basis
        self.n_gen = int(gen.shape[0])
        self.n_wave = c["matrix_angle"].shape[1]     # P·N entries per event
        self.n_proj = int(c.get("n_proj", 1))
        if self.n_wave % self.n_proj != 0:
            raise ValueError(
                f"cuda_polarised: n_wave={self.n_wave} not divisible by "
                f"n_proj={self.n_proj}")
        self.n_wave_base = self.n_wave // self.n_proj  # N = ck length
        if gen.shape[1:] != (self.n_proj, self.n_proj):
            raise ValueError(
                f"cuda_polarised: gen matrices have shape {gen.shape[1:]}, "
                f"does not match n_proj = {self.n_proj}")
        self.n_res = c["bw_order"].size // self.n_wave
        self.n_decay = c["fl_order"].size // self.n_wave
        self.n_unique_bw = len(c["m0_index"])
        self.n_gamma_rows = c["matrix_gamma"].shape[0]
        self.n_mass = int(np.max(c["mass_index"])) + 1
        self.n_momentum = int(np.max(c["fl_q_index"])) + 1
        self.n_angle_k = c["angle_k"].shape[0]
        self.n_angle_total = (int(np.max(c["angle_index"])) + 1
                              if len(c["angle_index"]) > 0 else 0)
        self.n_angle_comp = c["angle_k"].shape[-1]
        self.n_m0_params = int(np.max(c["m0_index"])) + 1
        self.n_g0_params = int(np.max(c["g0_index"])) + 1

        from tabpwa.amp_cache import build_amp_cache_layout
        self.n_uniq, self.slot_of_wave, self.rep_of_slot = \
            build_amp_cache_layout(c)
        self._ka = []
        self._last_dnorm = None

        def _db(a):
            arr = np.ascontiguousarray(a, np.float64)
            buf = _ffi.from_buffer(arr)
            self._ka.append(buf)
            return _ffi.cast("double*", buf)

        def _ib(a):
            arr = np.ascontiguousarray(a, np.int32)
            buf = _ffi.from_buffer(arr)
            self._ka.append(buf)
            return _ffi.cast("int*", buf)

        ma = c["matrix_angle"]
        gt = c["gamma_table"]
        mg = c["matrix_gamma"]
        gamma_col_idx = np.argmax(mg, axis=1).astype(np.int32)

        gen_r = np.ascontiguousarray(self.gen.real, np.float64)
        gen_i = np.ascontiguousarray(self.gen.imag, np.float64)

        self._ctx = self._lib.cuda_create_context_polarised(
            _ib(c["m0_index"]), len(c["m0_index"]),
            _ib(c["g0_index"]), len(c["g0_index"]),
            _ib(c["g0_mass_index"]), len(c["g0_mass_index"]),
            _ib(c["mass_index"]), len(c["mass_index"]),
            _ib(c["fl_type"]), len(c["fl_type"]),
            _ib(c["fl_q_index"]), len(c["fl_q_index"]),
            _ib(c["bw_order"]), len(c["bw_order"]),
            _ib(c["fl_order"]), len(c["fl_order"]),
            _ib(c["angle_index"]), len(c["angle_index"]),
            _db(c["angle_k"].flatten()), c["angle_k"].size,
            _db(c["angle_b"].flatten()), c["angle_b"].size,
            _db(np.real(ma).flatten()), ma.size,
            _db(np.imag(ma).flatten()), ma.size,
            _db(np.real(gt).flatten()), gt.size,
            _db(np.imag(gt).flatten()), gt.size,
            float(c["gamma_min"]), float(c["gamma_delta"]), gt.shape[-1],
            _db(mg.flatten()), mg.size,
            _ib(gamma_col_idx), len(gamma_col_idx),
            _db(c["fl_table"].flatten()), c["fl_table"].size,
            float(c["fl_min"]), float(c["fl_delta"]), c["fl_table"].shape[-1],
            self.n_wave, self.n_res, self.n_decay,
            self.n_unique_bw, self.n_gamma_rows,
            self.n_mass, self.n_momentum,
            self.n_angle_k, self.n_angle_total, self.n_angle_comp,
            self.n_m0_params, self.n_g0_params, batch_size,
            _ib(self.slot_of_wave), len(self.slot_of_wave),
            _ib(self.rep_of_slot), len(self.rep_of_slot),
            self.n_uniq, self.n_proj,
            _db(gen_r), gen_r.size,
            _db(gen_i), gen_i.size,
            self.n_gen)

    def load_data(self, data):
        ka = []

        def _db(a):
            arr = np.ascontiguousarray(a, np.float64)
            buf = _ffi.from_buffer(arr)
            ka.append(buf)
            return _ffi.cast("double*", buf)

        ne = data["mass"].shape[0]
        mass = data["mass"].reshape(ne, -1)[:, :self.n_mass]
        mom = data["q"].reshape(ne, -1)[:, :self.n_momentum]
        ang = data["angle"].reshape(ne, -1)[:, :self.n_angle_total
                                            * self.n_angle_comp]
        nang = self.n_angle_total
        bkg_key = "bkg_raw" if "bkg_raw" in data else "bkg"

        dh = DataHandle(self._lib.cuda_load_data_polarised(
            self._ctx, _db(mass), mass.shape[1],
            _db(mom), mom.shape[1], _db(ang), nang,
            _db(data["weight"]), _db(data[bkg_key]), ne), self._lib, ne)
        dh._keep = ka
        return dh

    def compute(self, params, data_handle, rho=None, norm=None,
                return_p=True):
        """Forward + gradients for the polarised projection-sum model.

        Args:
            params: dict with 'ck' (complex length N), 'm0', 'g0' and
                'scalar' (real length P²−1).  ``rho`` is built host-side
                as ``I + Σ_l scalar[l]·gen_l`` unless explicitly given.
            rho: optional explicit P×P Hermitian spin-density matrix
                (overrides the scalar-built one); passed to the kernel as
                two contiguous P² float64 arrays (row-major re / im).
            norm: optional float normalization; None → unnormalized.
            return_p: if True return per-event P (always returned here).

        Returns:
            (Q, grads, P) with grads keys ck/m0/g0/scalar — 'scalar' is
            the finished per-parameter polarisation gradient.
        """
        ck = params["ck"]
        nw = self.n_wave
        nbase = self.n_wave_base           # N — ck / gradient buffer size
        nu_ = self.n_unique_bw             # m0 gradient buffer size
        ng_ = self.n_gamma_rows            # g0 gradient buffer size
        nm_ = self.n_m0_params
        gg_ = self.n_g0_params
        use_norm = 0 if norm is None else 1
        norm_val = norm if norm is not None else 0.0

        if len(ck) != nbase:
            raise ValueError(
                f"cuda_polarised: ck length {len(ck)} != n_wave/n_proj = "
                f"{nbase} ({self.n_wave}/{self.n_proj})")

        scalar = np.asarray(params.get("scalar", np.zeros(self.n_gen)),
                            dtype=np.float64)
        if len(scalar) != self.n_gen:
            raise ValueError(
                f"cuda_polarised: scalar length {len(scalar)} != P²−1 = "
                f"{self.n_gen} (n_proj={self.n_proj})")

        # rho = I + Σ_l scalar[l]·gen_l  (host-side, per call)
        P = self.n_proj
        if rho is None:
            rho = np.eye(P, dtype=complex) + np.einsum(
                "l,lij->ij", scalar, self.gen)
        else:
            rho = np.asarray(rho, dtype=complex)
            if rho.shape != (P, P):
                raise ValueError(
                    f"cuda_polarised: rho must be ({P}, {P}), got "
                    f"{rho.shape}")
        rho_r = np.ascontiguousarray(rho.real, np.float64).reshape(-1)
        rho_i = np.ascontiguousarray(rho.imag, np.float64).reshape(-1)

        ck_r = np.real(ck).astype(np.float64)
        ck_i = np.imag(ck).astype(np.float64)
        m0 = np.zeros(nm_, np.float64)
        m0[:len(params["m0"])] = np.asarray(params["m0"])
        g0 = np.zeros(gg_, np.float64)
        g0[:len(params["g0"])] = np.asarray(params["g0"])

        oQ = _ffi.new("double*")
        oDn = _ffi.new("double*")
        oP = (np.zeros(data_handle.ne, np.float64)
              if return_p else None)
        ogck_r = np.zeros(nbase, np.float64)
        ogck_i = np.zeros(nbase, np.float64)
        ogm0 = np.zeros(nu_, np.float64)
        ogg0 = np.zeros(ng_, np.float64)
        ogsc = np.zeros(self.n_gen, np.float64)

        ka = []

        def _db(a):
            arr = np.ascontiguousarray(a, np.float64)
            buf = _ffi.from_buffer(arr)
            ka.append(buf)
            return _ffi.cast("double*", buf)

        self._lib.cuda_compute_polarised(
            self._ctx, data_handle.ptr,
            _db(ck_r), _db(ck_i), _db(m0), _db(g0),
            _db(rho_r), _db(rho_i),
            norm_val, use_norm,
            oQ, oDn, (_db(oP) if oP is not None else _ffi.NULL), _db(ogck_r), _db(ogck_i),
            _db(ogm0), _db(ogg0), _db(ogsc))
        self._last_dnorm = float(oDn[0]) if use_norm else None

        m0_idx = self.config["m0_index"]
        g0_idx = self.config["g0_index"]
        grad_m0 = np.zeros(self.n_m0_params, np.float64)
        grad_g0 = np.zeros(self.n_g0_params, np.float64)
        for i, idx in enumerate(m0_idx):
            grad_m0[idx] += ogm0[i]
        for i, idx in enumerate(g0_idx):
            grad_g0[idx] += ogg0[i]

        grads = {"ck": ogck_r + 1j * ogck_i,
                 "m0": grad_m0, "g0": grad_g0,
                 "scalar": ogsc}
        return oQ[0], grads, (oP if return_p else None)

    def free(self):
        if self._ctx is not None:
            self._lib.cuda_free_context_polarised(self._ctx)
            self._ctx = None
        self._ka.clear()

    def __del__(self):
        self.free()

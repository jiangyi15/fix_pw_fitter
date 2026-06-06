"""PyTorch implementation of the Kernel — exportable to ONNX.

Complex numbers are represented as ``(..., 2)`` float tensors
where ``[..., 0] = real``, ``[..., 1] = imag``.
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
#  Complex helpers
# ---------------------------------------------------------------------------

def c_make(re, im):
    return torch.stack([re, im], dim=-1)


def c_mul(a, b):
    return torch.stack([
        a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
        a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
    ], dim=-1)


def c_abs2(z):
    return z[..., 0] ** 2 + z[..., 1] ** 2


def c_conj(z):
    return torch.stack([z[..., 0], -z[..., 1]], dim=-1)


def c_exp(z):
    e = torch.exp(z[..., 0])
    return torch.stack([e * torch.cos(z[..., 1]),
                        e * torch.sin(z[..., 1])], dim=-1)


def c_div(a, b):
    denom = b[..., 0] ** 2 + b[..., 1] ** 2
    return torch.stack([
        (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]) / denom,
        (a[..., 1] * b[..., 0] - a[..., 0] * b[..., 1]) / denom,
    ], dim=-1)


def c_inv(z):
    return c_div(torch.ones_like(z), z)


def c_rmul(r, z):
    """Real scalar * complex."""
    return torch.stack([r * z[..., 0], r * z[..., 1]], dim=-1)


def c_prod_seq(tensor, dim):
    """Complex product along *dim* via unrolled pairwise multiplies.

    ``tensor.shape = (..., N, 2)`` → ``(..., 2)`` after multiplying
    all ``N`` complex numbers together.
    """
    n = tensor.shape[dim]
    if n == 0:
        return torch.ones_like(tensor.select(dim, 0))
    result = tensor.select(dim, 0)
    for i in range(1, n):
        result = c_mul(result, tensor.select(dim, i))
    return result


# ---------------------------------------------------------------------------
#  Interpolation
# ---------------------------------------------------------------------------

def interp_cplx(x, table, types, xmin, xdelta):
    """Linear interpolation for complex tables.

    Parameters
    ----------
    x : (nevt, n_pts) float
    table : (n_types, n_int, 2) float
    types : (n_pts,) int
    """
    diff = (x - xmin) / xdelta
    xbin = torch.floor(diff).to(torch.int64)
    flat_idx = types * table.shape[1] + xbin
    flat_tbl = table.reshape(-1, 2)
    left = torch.gather(flat_tbl, 0,
                        flat_idx.unsqueeze(-1).expand(-1, -1, 2))
    right = torch.gather(flat_tbl, 0,
                         (flat_idx + 1).unsqueeze(-1).expand(-1, -1, 2))
    frac = (diff - xbin).unsqueeze(-1)
    return left + (right - left) * frac


def interp_real(x, table, types, xmin, xdelta):
    """Linear interpolation for real tables.

    Parameters
    ----------
    x : (nevt, n_pts) float
    table : (n_types, n_int) float
    types : (n_pts,) int
    """
    diff = (x - xmin) / xdelta
    xbin = torch.floor(diff).to(torch.int64)
    flat_idx = types * table.shape[1] + xbin
    flat_tbl = table.reshape(-1)
    left = torch.gather(flat_tbl, 0, flat_idx)
    right = torch.gather(flat_tbl, 0, flat_idx + 1)
    return left + (right - left) * (diff - xbin)


# ---------------------------------------------------------------------------
#  TorchModel
# ---------------------------------------------------------------------------

class TorchModel(nn.Module):
    """ONNX-exportable forward model.

    All complex config arrays are stored as ``(..., 2)`` float buffers.
    Index arrays are int buffers.
    """

    def __init__(self, config: dict):
        super().__init__()

        gamma_table = np.asarray(config["gamma_table"])
        self.register_buffer(
            "gamma_table",
            torch.tensor(np.stack([gamma_table.real, gamma_table.imag], axis=-1)),
        )

        fl_table = np.asarray(config["fl_table"])
        self.register_buffer("fl_table", torch.tensor(fl_table))

        mat_gamma = np.asarray(config["matrix_gamma"], dtype=float)
        self.register_buffer("matrix_gamma", torch.tensor(mat_gamma))

        mat_ang = np.asarray(config["matrix_ang"], dtype=complex)
        self.register_buffer(
            "matrix_ang",
            torch.tensor(np.stack([mat_ang.real, mat_ang.imag], axis=-1)),
        )

        for name in ("g0_index", "gamma_index", "gamma_type",
                     "m0_index", "bw_index", "bw_gamma_index", "bw_order",
                     "q_index", "fl_type", "fl_order",
                     "angle_index", "ang_order"):
            arr = np.asarray(config[name], dtype=np.int64)
            self.register_buffer(name, torch.tensor(arr))

        self.gamma_min = float(config["gamma_min"])
        self.gamma_delta = float(config["gamma_delta"])
        self.fl_min = float(config["fl_min"])
        self.fl_delta = float(config["fl_delta"])
        self.angle_k = torch.tensor(np.asarray(config["angle_k"], dtype=float))
        self.angle_b = torch.tensor(np.asarray(config["angle_b"], dtype=float))

        self._nwaves = mat_ang.shape[-1]

    def forward(self, ck_re, ck_im, m0, g0, time_params,
                mass, q_data, angle, time, weight, frac, bkg, norm):
        nevt = mass.shape[0]
        nwaves = self._nwaves

        # ---- 1) Gamma interpolation ----
        g0a = torch.index_select(g0, 0, self.g0_index)
        mg = torch.index_select(mass, 1, self.gamma_index)
        gi = interp_cplx(mg, self.gamma_table, self.gamma_type,
                         self.gamma_min, self.gamma_delta)
        gamma_val = c_rmul(g0a, gi)
        gamma_for_mass = torch.einsum('ij,...jk->...ik',
                                      self.matrix_gamma, gamma_val)

        # ---- 2) Breit-Wigner ----
        m0a = torch.index_select(m0, 0, self.m0_index)
        mbw = torch.index_select(mass, 1, self.bw_index)
        gbw = torch.index_select(gamma_for_mass, 1, self.bw_gamma_index)

        # bwdom = m0a² - s² - 1j*m0a*gamma_for_bw
        # -1j*m0a*(Γr + iΓi) = m0a*Γi - i*m0a*Γr
        bwdom = torch.stack([
            m0a ** 2 - mbw ** 2 + m0a * gbw[..., 1],
            -m0a * gbw[..., 0],
        ], dim=-1)

        bw = c_inv(bwdom)

        nres = self.bw_order.shape[0] // nwaves
        bw_ordered = torch.index_select(bw, 1, self.bw_order)
        bw_reshaped = bw_ordered.reshape(nevt, nwaves, nres, 2)
        bwa = c_prod_seq(bw_reshaped, dim=-2)

        # ---- 3) Form factors ----
        fl_q = torch.index_select(q_data, 1, self.q_index)
        fl = interp_real(fl_q, self.fl_table, self.fl_type,
                         self.fl_min, self.fl_delta)
        ndec = self.fl_order.shape[0] // nwaves
        fl_ordered = torch.index_select(fl, 1, self.fl_order)
        fl_reshaped = fl_ordered.reshape(nevt, nwaves, ndec)
        fla = torch.prod(fl_reshaped, dim=-1)

        # ---- 4) Angular basis ----
        ang = torch.index_select(angle, 1, self.angle_index)
        ang_a = self.angle_k * ang + self.angle_b
        cosang = torch.cos(ang_a)
        nbasis, n_per = self.ang_order.shape
        cos_ordered = torch.index_select(cosang, 1,
                                         self.ang_order.reshape(-1))
        cos_reshaped = cos_ordered.reshape(nevt, nbasis, n_per)
        cosa = torch.prod(cos_reshaped, dim=-1)

        fa = torch.stack([
            cosa @ self.matrix_ang[..., 0],
            cosa @ self.matrix_ang[..., 1],
        ], dim=-1)

        # ---- 5) Amplitude ----
        ck = torch.stack([ck_re, ck_im], dim=-1)
        T = c_mul(bwa, c_rmul(fla, fa))
        amp_waves = c_mul(ck.unsqueeze(0), T)
        amp = amp_waves.reshape(nevt, 2, -1, 2).sum(dim=-2)
        amp0, amp1 = amp[:, 0], amp[:, 1]

        # ---- 6) Time-dependent mixing ----
        gt, dg, dm, poqr, poqi, ap = time_params.unbind()

        argL = torch.stack([
            -time * (gt + dg / 2) / 2,
            -time * dm / 2,
        ], dim=-1)
        argH = torch.stack([
            -time * (gt - dg / 2) / 2,
            time * dm / 2,
        ], dim=-1)

        eL = c_exp(argL)
        eH = c_exp(argH)
        ep = (eL + eH) / 2
        em = (eL - eH) / 2
        poq = c_make(poqr * torch.cos(poqi), poqr * torch.sin(poqi))

        X = c_add_f(c_mul(ep, amp0), c_mul(c_mul(poq, em), amp1))
        Y = c_add_f(c_mul(c_div(em, poq), amp0), c_mul(ep, amp1))

        PB = c_abs2(X)
        PBbar = c_abs2(Y)

        P = (1 - frac) * (1 - ap) * PB + frac * (1 + ap) * PBbar

        # ---- 7) Objective ----
        Pnorm = P / norm + bkg
        Q = -torch.sum(weight * torch.log(Pnorm))

        return P, Q

    def forward_no_norm(self, ck_re, ck_im, m0, g0, time_params,
                        mass, q_data, angle, time, weight, frac, bkg):
        P, _ = self.forward(ck_re, ck_im, m0, g0, time_params,
                            mass, q_data, angle, time, weight, frac, bkg,
                            torch.ones_like(weight))
        Q = torch.sum(weight * P)
        return P, Q


def c_add_f(a, b):
    """Faster complex addition (just delegates to ``+``)."""
    return a + b


# ---------------------------------------------------------------------------
#  ONNX export
# ---------------------------------------------------------------------------

def export_torch_model(model, onnx_path, nevt=10):
    """Trace *model* and export to ONNX.

    Parameters
    ----------
    model : TorchModel
    onnx_path : str
    nevt : int   batch size for the dummy trace input
    """
    model.eval()
    device = next(model.parameters()).device
    rng = np.random.default_rng(42)

    dummy = {
        "ck_re": torch.tensor(rng.normal(size=2), dtype=torch.float, device=device),
        "ck_im": torch.tensor(rng.normal(size=2), dtype=torch.float, device=device),
        "m0": torch.tensor(rng.normal(size=1), dtype=torch.float, device=device),
        "g0": torch.tensor(rng.normal(size=1), dtype=torch.float, device=device),
        "time_params": torch.tensor(rng.normal(size=6), dtype=torch.float, device=device),
        "mass": torch.tensor(rng.normal(size=(nevt, 2)), dtype=torch.float, device=device),
        "q_data": torch.tensor(rng.normal(size=(nevt, 1)), dtype=torch.float, device=device),
        "angle": torch.tensor(rng.normal(size=(nevt, 1)), dtype=torch.float, device=device),
        "time": torch.tensor(rng.normal(size=nevt), dtype=torch.float, device=device),
        "weight": torch.ones(nevt, device=device),
        "frac": torch.zeros(nevt, device=device),
        "bkg": torch.zeros(nevt, device=device),
        "norm": torch.tensor(1.0, device=device),
    }

    torch.onnx.export(
        model,
        tuple(dummy[k] for k in dummy),
        onnx_path,
        input_names=list(dummy.keys()),
        output_names=["P", "Q"],
        dynamic_axes={k: {0: "nevt"} for k in ("mass", "q_data", "angle",
                                                "time", "weight", "frac",
                                                "bkg", "P")},
        opset_version=17,
    )

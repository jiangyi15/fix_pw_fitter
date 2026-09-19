"""SU(2) 2x2 complex matrices (vectorized over events), ported faithfully
from tf-pwa ``tf_pwa/angle.py`` ``class SU2M`` so that alignment euler
angles computed here are identical in convention to the reference used by
``tf_pwa/cal_angle.py``.

Every element has a leading batch shape ``(..., 2, 2)`` of complex128.
Rotation/boost conventions (copy of tf-pwa):

* ``Rotation_z(alpha)``   = diag(e^{-i alpha/2}, e^{+i alpha/2})
* ``Rotation_y(beta)``    = [[cos b/2, -sin b/2], [sin b/2, cos b/2]]
* ``Boost_z(omega)``      = diag(e^{-omega/2}, e^{+omega/2})     (rapidity)
* ``Boost_z_from_p(p4)``  = Boost_z(acosh(E/m))  with p4 = (E, px, py, pz)

Composition ``A * B`` is ordinary matrix multiplication (left factor applied
first for column-vector transformations), matching tf-pwa ``__mul__``.
``inv`` is the SU(2) inverse [[bb,-ab],[-ba,aa]].

``get_euler_angle`` returns ``(alpha, beta, gamma)`` in the z-y-z convention
``Rz(alpha) Ry(beta) Rz(gamma)`` using tf-pwa's exact branch choice
(beta in [0, pi]).
"""
import numpy as np

_I = 1j


def _c(x):
    return np.asarray(x, dtype=np.complex128)


def _scalar_or_batch(x):
    return np.asarray(x, dtype=float)


def Rotation_z(alpha):
    """diag(e^{-i alpha/2}, e^{i alpha/2}); alpha shape (...) -> (...,2,2)."""
    a = _scalar_or_batch(alpha)
    ph = np.exp(_I * a / 2.0)
    sh = ph.shape
    out = np.zeros(sh + (2, 2), dtype=np.complex128)
    out[..., 0, 0] = 1.0 / ph
    out[..., 1, 1] = ph
    return out


def Rotation_y(beta):
    b = _scalar_or_batch(beta)
    sh = b.shape
    s = np.sin(b / 2.0)
    c = np.cos(b / 2.0)
    out = np.zeros(sh + (2, 2), dtype=np.complex128)
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    return out


def Boost_z(omega):
    """diag(e^{-omega/2}, e^{omega/2}); omega = rapidity along +z."""
    o = _scalar_or_batch(omega)
    sh = o.shape
    a = np.exp(o / 2.0)
    out = np.zeros(sh + (2, 2), dtype=np.complex128)
    out[..., 0, 0] = 1.0 / a
    out[..., 1, 1] = a
    return out


def Boost_z_from_p(p4):
    """p4 shape (..., 4) = (E, px, py, pz).  omega = acosh(E/m)."""
    p4 = np.asarray(p4, dtype=float)
    E = p4[..., 0]
    m2 = np.clip(E * E - np.sum(p4[..., 1:] ** 2, axis=-1), 1e-30, None)
    m = np.sqrt(m2)
    omega = np.arccosh(np.clip(E / m, 1.0, None))
    return Boost_z(omega)


def _mul(A, B):
    """Batched matrix product of (...,2,2) complex arrays (tf-pwa __mul__)."""
    return np.einsum("...ij,...jk->...ik", A, B)


def inv(A):
    """SU(2) inverse [[bb,-ab],[-ba,aa]] of (...,2,2) array."""
    A = _c(A)
    out = np.zeros_like(A)
    out[..., 0, 0] = A[..., 1, 1]
    out[..., 0, 1] = -A[..., 0, 1]
    out[..., 1, 0] = -A[..., 1, 0]
    out[..., 1, 1] = A[..., 0, 0]
    return out


def _su2_to_quat(A):
    """SU(2) ``U = [[q0 - i q3, -q2 - i q1], [q2 - i q1, q0 + i q3]]`` -> quaternion."""
    A = _c(A)
    q0 = np.real(A[..., 0, 0])
    q1 = -np.imag(A[..., 1, 0])
    q2 = np.real(A[..., 1, 0])
    q3 = -np.imag(A[..., 0, 0])
    return q0, q1, q2, q3


def _quat_to_rot(q0, q1, q2, q3):
    """Unit quaternion (active vector rotation) -> (...,3,3) rotation matrix."""
    M = np.zeros(q0.shape + (3, 3))
    M[..., 0, 0] = 1 - 2 * (q2 * q2 + q3 * q3)
    M[..., 0, 1] = 2 * (q1 * q2 - q0 * q3)
    M[..., 0, 2] = 2 * (q1 * q3 + q0 * q2)
    M[..., 1, 0] = 2 * (q1 * q2 + q0 * q3)
    M[..., 1, 1] = 1 - 2 * (q1 * q1 + q3 * q3)
    M[..., 1, 2] = 2 * (q2 * q3 - q0 * q1)
    M[..., 2, 0] = 2 * (q1 * q3 - q0 * q2)
    M[..., 2, 1] = 2 * (q2 * q3 + q0 * q1)
    M[..., 2, 2] = 1 - 2 * (q1 * q1 + q2 * q2)
    return M


def get_euler_angle(A):
    """z-y-z euler ``(alpha, beta, gamma)`` with exact SU(2) equality
    ``A = Rz(alpha) Ry(beta) Rz(gamma)`` (sign included).

    alpha, gamma come out in ``(-2 pi, 2 pi]``; when the principal phases
    (which carry (alpha +/- gamma)/2) reconstruct ``-A``, gamma is shifted by
    2 pi so the returned triple reproduces the matrix phase exactly - required
    for half-integer-spin D matrices, which change sign under +2 pi.
    """
    A = _c(A)
    cosbeta = np.real(A[..., 0, 0] * A[..., 1, 1]
                      + A[..., 0, 1] * A[..., 1, 0])
    cosbeta = np.clip(cosbeta, -1.0, 1.0)
    beta = np.arccos(cosbeta)
    p = np.angle(A[..., 1, 1])          # (alpha+gamma)/2
    m = np.angle(A[..., 1, 0])          # (alpha-gamma)/2
    alpha = p + m
    gamma = p - m
    # singular beta (s = sin(beta/2) ~ 0): only alpha+gamma is physical; put
    # the whole rotation into alpha (gamma = 0).
    s = np.sin(beta / 2.0)
    small = np.abs(s) < 1e-9
    if np.any(small):
        ph = np.angle(A[..., 1, 0]) if False else np.zeros_like(beta)
        # Rz(alpha) cos-part: alpha = phase of A00 (e^{-i alpha/2}) * 2 with sign
        tot = 2.0 * (-np.angle(np.where(small, A[..., 0, 0], 1.0)))
        alpha = np.where(small, tot, alpha)
        gamma = np.where(small, 0.0, gamma)
    # exact SU(2) sign: if the triple reconstructs -A, add 2 pi to gamma.
    U0 = _mul(_mul(Rotation_z(alpha), Rotation_y(beta)), Rotation_z(gamma))
    overlap = np.sum(np.conj(A) * U0, axis=(-2, -1))
    neg = np.real(overlap) < 0
    gamma = np.where(neg, gamma + 2.0 * np.pi, gamma)
    return alpha, beta, gamma


def Identity(shape):
    """(...,2,2) identity SU(2) for batch shape *shape* (numpy broadcast)."""
    sh = tuple(np.atleast_1d(np.asarray(shape, dtype=int))) + (2, 2)
    I = np.zeros(sh, dtype=np.complex128)
    I[..., 0, 0] = 1.0
    I[..., 1, 1] = 1.0
    return I


def product(seq):
    """Product of a sequence of (...,2,2) SU(2) arrays, left applied first."""
    out = None
    for U in seq:
        U = _c(U)
        out = U if out is None else _mul(out, U)
    return out


import numpy as np
import pytest

from ampfit.su2 import (Rotation_z, Rotation_y, Boost_z, Boost_z_from_p,
                        inv, _mul, get_euler_angle)


def rand_angles(n=5):
    rng = np.random.default_rng(0)
    return (rng.uniform(-2 * np.pi, 2 * np.pi, n),
            rng.uniform(0.0, np.pi, n),
            rng.uniform(-2 * np.pi, 2 * np.pi, n))


def test_euler_roundtrip():
    a, b, g = rand_angles()
    U = _mul(_mul(Rotation_z(a), Rotation_y(b)), Rotation_z(g))
    a2, b2, g2 = get_euler_angle(U)
    # reconstruct and compare
    U2 = _mul(_mul(Rotation_z(a2), Rotation_y(b2)), Rotation_z(g2))
    assert np.allclose(U, U2, atol=1e-9)
    assert np.allclose(b2, b, atol=1e-9)


def test_inverse():
    a, b, g = rand_angles()
    U = _mul(_mul(Rotation_z(a), Rotation_y(b)), Rotation_z(g))
    Id = _mul(U, inv(U))
    target = np.tile(np.eye(2), (a.shape[0], 1, 1))
    assert np.allclose(Id, target, atol=1e-12)


def test_boost_z_from_p():
    # 4-vectors with known rapidity: E = m cosh(w), pz = m sinh(w)
    m = 1.186                       # Lambda-like mass
    rng = np.random.default_rng(1)
    w = rng.uniform(0.1, 2.0, 4)
    E = m * np.cosh(w)
    pz = m * np.sinh(w)
    p4 = np.stack([E, np.zeros_like(E), np.zeros_like(E), pz], axis=-1)
    B = Boost_z_from_p(p4)
    # eigen-decomposition check: B diag phases exp(±w/2)
    assert np.allclose(np.abs(B[..., 0, 1]), 0.0, atol=1e-12)
    assert np.allclose(np.log(np.abs(B[..., 0, 0])), -w / 2, atol=1e-9)
    assert np.allclose(np.log(np.abs(B[..., 1, 1])), w / 2, atol=1e-9)


def test_boost_z_scalar_matches_batch():
    w = 0.8
    B1 = Boost_z(np.array(w))
    B2 = Boost_z(np.array([w]))[0]
    assert np.allclose(B1, B2)


def test_determinant_one():
    a, b, g = rand_angles(3)
    U = _mul(_mul(Rotation_z(a), Rotation_y(b)), Rotation_z(g))
    det = U[..., 0, 0] * U[..., 1, 1] - U[..., 0, 1] * U[..., 1, 0]
    assert np.allclose(det, 1.0, atol=1e-12)

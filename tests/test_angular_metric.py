"""Analytic angular metric: closed-form integrals vs flat-angle MC.

The metric integrates products of ``cos(k x + b)`` monomials exactly.
Validation: generate flat angles DIRECTLY from the canonical measures
(phi uniform on [0, 2pi); theta = arccos(uniform[-1, 1]), i.e. flat
dcos(theta)), evaluate the basis per event exactly as the kernels do,
and compare Monte-Carlo averages with the analytic values (scaled by
the domain volume).  No model or phsp generator involved.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from tabpwa.angular_metric import (
    angular_metric, quad_metric, coeff_matrix, intensity_norm,
    quad_weighted_norm, solve_quad, pair_index, column_integral,
)

KINDS = ("phi", "theta", "theta")          # canonical 3-column layout
VOLUME = 2.0 * np.pi * 2.0 * 2.0           # phi: 2pi; theta: 2 each
N = 400_000


def _flat_angles(rng, n):
    """The canonical one-chain angles, flat in their measures:
    phi ~ U[0, 2pi); theta = arccos(U[-1, 1]) (flat dcos(theta))."""
    ang = np.empty((n, 3))
    ang[:, 0] = rng.uniform(0.0, 2.0 * np.pi, n)
    ang[:, 1] = np.arccos(rng.uniform(-1.0, 1.0, n))
    ang[:, 2] = np.arccos(rng.uniform(-1.0, 1.0, n))
    return ang


@pytest.fixture(scope="module")
def basis():
    rng = np.random.default_rng(11)
    n = 12                                  # one chain: one block
    ks = rng.choice([0.0, 0.5, 1.0, 1.5, 2.0], size=(n, 3))
    bs = rng.choice([0.0, -np.pi / 2, 0.3], size=(n, 3))
    ang = _flat_angles(rng, N)
    Z = np.prod(np.cos(ang[:, None, :] * ks[None, :, :] + bs[None, :, :]),
                axis=-1)                     # (N, n)
    return ks, bs, ang, Z


def test_column_integral_vs_quad():
    """Closed-form column integrals == numerical integration."""
    from scipy.integrate import quad
    rng = np.random.default_rng(3)
    for _ in range(30):
        m = rng.integers(1, 4)
        ks = rng.choice([0.0, 0.5, 1.0, 1.5, 2.0, 3.0], size=m)
        bs = rng.uniform(-np.pi, np.pi, size=m)
        for kind, dom in (("phi", (0.0, 2.0 * np.pi)),
                          ("theta", (0.0, np.pi))):
            ana = column_integral(kind, ks, bs)
            f = lambda x: np.prod(np.cos(np.asarray(ks) * x
                                         + np.asarray(bs)), axis=0)
            w = (lambda x: 1.0) if kind == "phi" else np.sin
            num = quad(f, *dom, weight="sin" if kind == "theta" else None,
                       wvar=1.0 if kind == "theta" else None,
                       limit=200)[0]
            assert ana == pytest.approx(num, rel=1e-9, abs=1e-11), (kind, ks, bs)


def test_metric_vs_mc(basis):
    ks, bs, ang, Z = basis
    M = angular_metric(ks, bs, KINDS)
    # batched MC mean + per-entry standard error (25 batches)
    parts = np.array_split(Z, 25)
    means = [p.T @ p / len(p) for p in parts]
    mc = np.mean(means, axis=0)
    sigma = np.std(means, axis=0, ddof=1) / np.sqrt(len(means))
    d = np.abs(M / VOLUME - mc)
    assert np.all(d < 6.0 * sigma + 1e-12), np.max(d / (sigma + 1e-12))


def test_quad_metric_and_solve(basis):
    ks, bs, ang, Z = basis
    Z0 = Z
    pairs = pair_index(ks.shape[0])
    g = Z[:, [i for i, _ in pairs]] * Z[:, [j for _, j in pairs]]
    G = quad_metric(ks, bs, KINDS)
    G_mc = g.T @ g / N
    d = np.abs(G / VOLUME - G_mc)
    assert np.max(d / (0.05 * (1.0 + np.abs(G_mc)))) < 1.0, np.max(d)

    # the inverse-matrix step: the moments must be reproduced.  (The
    # synthetic pair basis is overcomplete — pairs whose exponent sums
    # coincide are the SAME function — so c itself is not unique; real
    # applications dedup the monomials first.)
    c_true = np.abs(np.random.default_rng(5).normal(size=len(pairs))) + 0.1
    c_fit = solve_quad(G, G @ c_true)
    assert np.max(np.abs(G @ c_fit - G @ c_true)) < 1e-8 * np.max(
        np.abs(G @ c_true))


def test_intensity_norm_vs_mc(basis):
    ks, bs, ang, Z = basis
    Z0 = Z
    M = angular_metric(ks, bs, KINDS)
    rng = np.random.default_rng(7)
    ma = rng.normal(size=(3, ks.shape[0])) + 1j * rng.normal(
        size=(3, ks.shape[0]))
    C = coeff_matrix(ma)
    ana = intensity_norm(C, M)
    A = Z0 @ ma.T
    mc = np.mean(np.sum(np.abs(A) ** 2, axis=1)) * VOLUME
    assert ana == pytest.approx(mc, rel=0.02)


def test_quad_weighted_norm_vs_mc(basis):
    ks, bs, ang, Z = basis
    Z0 = Z
    pairs = pair_index(ks.shape[0])
    G = quad_metric(ks, bs, KINDS)
    rng = np.random.default_rng(9)
    c = np.abs(rng.normal(size=len(pairs))) + 0.1
    b = np.abs(rng.normal(size=len(pairs))) + 0.1
    ana = float(c @ G @ b)
    g = Z0[:, [i for i, _ in pairs]] * Z0[:, [j for _, j in pairs]]
    w = g @ b                                 # w(e) = sum b_a g_a(e)
    p = g @ c                                 # P(e) = sum a_a g_a(e)
    mc = np.mean(w * p) * VOLUME
    assert ana == pytest.approx(mc, rel=0.02)

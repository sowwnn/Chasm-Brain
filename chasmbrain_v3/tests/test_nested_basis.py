"""Invariants from spec §4 for every NestedBasis implementation."""
import numpy as np
import pytest

from chasmbrain_v3.basis.laplacian import LaplacianBasis


def _random_laplacian_basis(n=64, K=16, seed=0):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(n, K)))
    lambdas = np.sort(rng.uniform(0, 2, size=K))
    return LaplacianBasis(U, lambdas)


@pytest.mark.parametrize("t", [0.25, 0.5, 1.0])
def test_project_lift_roundtrip(t):
    b = _random_laplacian_basis()
    c = np.random.default_rng(1).normal(size=b.k(t))
    np.testing.assert_allclose(b.project(b.lift(c, t), t), c, atol=1e-6)


def test_lift_is_nested():
    b = _random_laplacian_basis()
    t_lo, t_hi = 0.25, 0.75
    c_lo = np.random.default_rng(2).normal(size=b.k(t_lo))
    c_hi = np.concatenate([c_lo, np.zeros(b.k(t_hi) - b.k(t_lo))])
    np.testing.assert_allclose(b.lift(c_hi, t_hi), b.lift(c_lo, t_lo), atol=1e-6)


def test_band_is_the_difference():
    b = _random_laplacian_basis()
    idx = b.band(0.25, 0.75)
    assert len(idx) == b.k(0.75) - b.k(0.25)

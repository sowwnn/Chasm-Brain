"""Spec §5: for every t_prev < t, step(...)[:, :k(t_prev)] must equal c_prev exactly."""
import numpy as np
import pytest
import torch

from chasmbrain_v3.basis.laplacian import LaplacianBasis
from chasmbrain_v3.model.encoder import DualPathEncoder
from chasmbrain_v3.model.refiner import Refiner


@pytest.mark.parametrize("t_prev,t", [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0)])
def test_step_keeps_c_prev(t_prev, t):
    n, K, d, B = 64, 16, 32, 4
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.normal(size=(n, K)))
    basis = LaplacianBasis(U, np.linspace(0, 2, K))
    model = Refiner(DualPathEncoder(d, 64), basis, d_hidden=64)
    cls, patch = torch.randn(B, d), torch.randn(B, 256, d)
    c_prev = torch.randn(B, basis.k(t_prev))
    c_full = model.step(cls, patch, c_prev, t_prev, t)
    assert c_full.shape == (B, basis.k(t))
    assert torch.equal(c_full[:, : basis.k(t_prev)], c_prev)

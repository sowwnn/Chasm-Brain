"""Spec §3 checks that need no dataset files: repeat averaging and noise ceiling on synthetic data."""
import numpy as np
import pytest

from chasmbrain_v3.data.normalize import fit_zscore, apply_zscore, noise_ceiling, reliability_weights
from chasmbrain_v3.data.split import average_repeats


def test_average_repeats():
    b = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    X = average_repeats(b)
    assert X.shape == (2, 4)
    np.testing.assert_allclose(X[0], b[0].mean(0), atol=1e-6)


def test_zscore_roundtrip():
    X = np.random.default_rng(1).normal(3, 2, size=(100, 5)).astype(np.float32)
    mu, s = fit_zscore(X)
    Z = apply_zscore(X, mu, s)
    np.testing.assert_allclose(Z.mean(0), 0, atol=1e-5)
    np.testing.assert_allclose(Z.std(0), 1, atol=1e-5)


@pytest.mark.parametrize("snr", [0.5, 2.0])
def test_noise_ceiling_recovers_known_snr(snr):
    """signal ~ N(0, snr^2), noise ~ N(0, 1), both already unit-variance-normalised inputs:
    ncsnr = snr, NC = snr^2 / (snr^2 + 1/3)."""
    rng = np.random.default_rng(2)
    n_img, n_rep, n_vox = 8000, 3, 8
    sig = rng.normal(0, snr, size=(n_img, 1, n_vox))
    raw = sig + rng.normal(size=(n_img, n_rep, n_vox))
    # huwng betas are pre-z-scored per voxel; emulate that so total variance is 1
    raw = (raw - raw.reshape(-1, n_vox).mean(0)) / raw.reshape(-1, n_vox).std(0)
    nc = noise_ceiling(raw.astype(np.float32), n_reps=3)
    expected = snr**2 / (snr**2 + 1 / 3)
    np.testing.assert_allclose(nc, expected, atol=0.05)
    w = reliability_weights(raw.astype(np.float32), n_reps=3)
    np.testing.assert_allclose(w, expected, atol=0.05)

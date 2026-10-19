"""Noise ceiling and reliability weights from huwng's single-trial betas (spec §3).
huwng betas are already z-scored per voxel, so fit_zscore/apply_zscore are identity-like helpers
kept only for API symmetry with the spec; most callers can skip straight to noise_ceiling."""
import numpy as np

NC_FORMULA = ("ncsnr = sqrt(max(signal_var, 0)) / sqrt(noise_var), signal_var = 1 - mean per-image unbiased "
              "variance across the 3 repeats of huwng's already-z-scored single-trial betas; "
              "NC = ncsnr^2 / (ncsnr^2 + 1/n), n = repeats averaged (Allen et al. 2022)")


def fit_zscore(X_train: np.ndarray):
    mu = X_train.mean(axis=0)
    sigma = np.maximum(X_train.std(axis=0), 1e-6)
    return mu.astype(np.float32), sigma.astype(np.float32)


def apply_zscore(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((X - mu) / sigma).astype(np.float32)


def _signal_noise_var(betas_3rep: np.ndarray):
    """betas_3rep: [n_images, 3, n_voxels], already z-scored per voxel (huwng convention).
    Returns per-voxel (signal_var, noise_var) with total variance normalised to 1."""
    b = np.asarray(betas_3rep, dtype=np.float64)
    noise_var = b.var(axis=1, ddof=1).mean(axis=0)     # unbiased within-image variance, averaged over images
    signal_var = np.clip(1.0 - noise_var, 0.0, None)
    return signal_var, noise_var


def noise_ceiling(betas_3rep: np.ndarray, n_reps: int = 3) -> np.ndarray:
    """NSD-style noise ceiling per voxel, in [0, 1], for targets averaged over n_reps repeats.
    Compute on train rows only. Write NC_FORMULA into metrics.json alongside the value."""
    signal_var, noise_var = _signal_noise_var(betas_3rep)
    ncsnr2 = signal_var / np.maximum(noise_var, 1e-6)
    return (ncsnr2 / (ncsnr2 + 1.0 / n_reps)).astype(np.float32)


def reliability_weights(betas_3rep: np.ndarray, n_reps: int = 3) -> np.ndarray:
    """Signal variance / total variance of the n_reps-averaged target, per voxel (spec §8 calibration weight w)."""
    signal_var, noise_var = _signal_noise_var(betas_3rep)
    return (signal_var / np.maximum(signal_var + noise_var / n_reps, 1e-6)).astype(np.float32)

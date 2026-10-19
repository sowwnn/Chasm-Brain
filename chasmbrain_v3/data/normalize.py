"""Per-voxel z-score, noise ceiling, reliability weights. All statistics on train only (spec §3)."""
import numpy as np


def fit_zscore(X_train: np.ndarray):
    """Return (mu, sigma) per voxel."""
    raise NotImplementedError


def apply_zscore(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def noise_ceiling(betas_single_trial: np.ndarray, image_ids: np.ndarray) -> np.ndarray:
    """NSD formula: NC = ncsnr^2 / (ncsnr^2 + 1/n), from the 3 train repeats. Formula is also written to metrics.json."""
    raise NotImplementedError


def reliability_weights(betas_single_trial: np.ndarray, image_ids: np.ndarray) -> np.ndarray:
    """Signal variance / total variance per voxel, used by calibration (spec §8)."""
    raise NotImplementedError

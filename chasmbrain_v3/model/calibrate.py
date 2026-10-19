"""Analytic amplitude calibration (spec §8). No learned parameters; applied identically to every model.
Fit on train only. Report MSE raw and calibrated; Pearson is invariant under per-voxel affine."""
import numpy as np


def fit_affine(v_pred_train: np.ndarray, v_true_train: np.ndarray):
    """Per-voxel least squares v_true ≈ a*v_pred + b. Returns (a [n_voxels], b [n_voxels]). Reviewer t4Xm's question."""
    raise NotImplementedError


def fit_shrinkage(v_pred_train: np.ndarray, v_true_train: np.ndarray, w: np.ndarray) -> np.ndarray:
    """s_i = w_i * cov(pred_i, true_i) / var(pred_i), w = reliability weights."""
    raise NotImplementedError


def apply(v_pred: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    raise NotImplementedError

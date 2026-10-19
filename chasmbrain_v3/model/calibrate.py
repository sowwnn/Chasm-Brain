"""Analytic amplitude calibration (spec §8). No learned parameters; applied identically to every model.
Fit on train only. Report MSE raw and calibrated; Pearson is invariant under per-voxel affine."""
import numpy as np


def fit_affine(v_pred_train: np.ndarray, v_true_train: np.ndarray):
    """Per-voxel least squares v_true ~= a*v_pred + b. Returns (a [n_voxels], b [n_voxels]).
    Reviewer t4Xm's question. Degenerate (zero-variance) predicted voxels get a=0, b=mean(v_true)."""
    pred_mean = v_pred_train.mean(axis=0)
    true_mean = v_true_train.mean(axis=0)
    pred_c = v_pred_train - pred_mean
    true_c = v_true_train - true_mean
    var_pred = (pred_c**2).mean(axis=0)
    cov = (pred_c * true_c).mean(axis=0)
    a = np.divide(cov, var_pred, out=np.zeros_like(cov), where=var_pred > 1e-12)
    b = true_mean - a * pred_mean
    return a.astype(np.float32), b.astype(np.float32)


def fit_shrinkage(v_pred_train: np.ndarray, v_true_train: np.ndarray, w: np.ndarray) -> np.ndarray:
    """s_i = w_i * cov(pred_i, true_i) / var(pred_i), w = reliability weights [n_voxels]."""
    pred_c = v_pred_train - v_pred_train.mean(axis=0)
    true_c = v_true_train - v_true_train.mean(axis=0)
    var_pred = (pred_c**2).mean(axis=0)
    cov = (pred_c * true_c).mean(axis=0)
    ratio = np.divide(cov, var_pred, out=np.zeros_like(cov), where=var_pred > 1e-12)
    return (w * ratio).astype(np.float32)


def apply(v_pred: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * v_pred + b).astype(np.float32)

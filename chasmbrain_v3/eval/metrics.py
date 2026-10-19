"""Metrics (spec §9)."""
import numpy as np


def voxel_pearson(v_pred: np.ndarray, v_true: np.ndarray) -> np.ndarray:
    """Pearson across images, per voxel -> r [n_voxels]. Report the mean over the mask."""
    raise NotImplementedError


def nc_normalized(r: np.ndarray, nc: np.ndarray) -> float:
    """mean(r^2) / mean(nc), Algonauts convention."""
    raise NotImplementedError


def voxel_mse(v_pred: np.ndarray, v_true: np.ndarray):
    raise NotImplementedError


def per_roi(r: np.ndarray, roi_labels: dict) -> dict:
    raise NotImplementedError


def per_band(c_pred, c_true, basis, n_bands: int = 10) -> np.ndarray:
    """Pearson per frequency band -> r [n_bands]. The new axis of the paper."""
    raise NotImplementedError

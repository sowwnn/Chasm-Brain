"""Metrics (spec §9)."""
import numpy as np


def voxel_pearson(v_pred: np.ndarray, v_true: np.ndarray) -> np.ndarray:
    """v_pred, v_true: [n_images, n_voxels]. Pearson across images, per voxel -> r [n_voxels].
    Report the mean over the mask. Voxels with zero variance in either array get r=0."""
    pred_c = v_pred - v_pred.mean(axis=0, keepdims=True)
    true_c = v_true - v_true.mean(axis=0, keepdims=True)
    num = (pred_c * true_c).sum(axis=0)
    denom = np.sqrt((pred_c**2).sum(axis=0) * (true_c**2).sum(axis=0))
    r = np.divide(num, denom, out=np.zeros_like(num), where=denom > 1e-12)
    return r.astype(np.float32)


def nc_normalized(r: np.ndarray, nc: np.ndarray) -> float:
    """mean(r^2) / mean(nc), Algonauts convention. nc in [0,1] per voxel, matching r's voxels."""
    return float(np.mean(r**2) / np.mean(nc))


def voxel_mse(v_pred: np.ndarray, v_true: np.ndarray):
    """Returns (mse_per_voxel [n_voxels], mean_mse)."""
    mse = ((v_pred - v_true) ** 2).mean(axis=0)
    return mse.astype(np.float32), float(mse.mean())


def per_roi(r: np.ndarray, roi_labels: dict) -> dict:
    """r: [n_voxels]. roi_labels: {name: bool [n_voxels]}. Returns {name: mean r over that ROI}."""
    return {name: float(r[mask].mean()) for name, mask in roi_labels.items() if mask.any()}


def per_band(c_pred: np.ndarray, c_true: np.ndarray, basis, n_bands: int = 10) -> np.ndarray:
    """c_pred, c_true: [n_images, K] full coefficient arrays (already projected via basis.project).
    Splits the K dimensions into n_bands contiguous, equal-width-in-t chunks (so each band corresponds
    to a slice of the nested basis, consistent with basis.band's meaning) and computes Pearson across
    images for all voxels reconstructed from just that band, averaged over images.
    Returns r [n_bands]: mean per-voxel Pearson of basis.lift(band coefficients) vs the equivalent
    lift of the true coefficients, restricted to that band's own incremental voxel reconstruction
    being compared the same way for pred and true."""
    K = basis.K
    edges = np.linspace(0, K, n_bands + 1).astype(int)
    r_bands = np.zeros(n_bands, dtype=np.float32)
    for b in range(n_bands):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            r_bands[b] = np.nan
            continue
        # Reconstruct voxels using ONLY this band's coefficients (zero elsewhere), for pred and true alike,
        # so the comparison isolates what that frequency band contributes.
        cp = np.zeros_like(c_pred); cp[:, lo:hi] = c_pred[:, lo:hi]
        ct = np.zeros_like(c_true); ct[:, lo:hi] = c_true[:, lo:hi]
        vp = cp @ basis.U.T
        vt = ct @ basis.U.T
        r = voxel_pearson(vp, vt)
        r_bands[b] = np.nanmean(r)
    return r_bands

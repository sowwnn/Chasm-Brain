"""Statistics (spec §9). Every table: mean over 4 subjects ± SD, bootstrap CI, p vs TBEn."""
import numpy as np
import torch
from scipy import stats as sstats


def paired_subject_test(metric_A: np.ndarray, metric_B: np.ndarray) -> dict:
    """metric_A, metric_B: [n_subjects] (one scalar metric per subject, e.g. mean Pearson). Paired
    t-test and Wilcoxon signed-rank over subjects. Returns {mean_diff, t_stat, t_p, wilcoxon_stat, wilcoxon_p}."""
    metric_A, metric_B = np.asarray(metric_A, dtype=np.float64), np.asarray(metric_B, dtype=np.float64)
    diff = metric_A - metric_B
    t_stat, t_p = sstats.ttest_rel(metric_A, metric_B)
    try:
        w_stat, w_p = sstats.wilcoxon(metric_A, metric_B)
    except ValueError:   # all differences zero, or n too small
        w_stat, w_p = np.nan, np.nan
    return {"mean_diff": float(diff.mean()), "t_stat": float(t_stat), "t_p": float(t_p),
            "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p)}


def bootstrap_ci(v_pred: np.ndarray, v_true: np.ndarray, n: int = 10000, seed: int = 0,
                  voxel_subsample: int = 2000, device: str = None) -> dict:
    """v_pred, v_true: [n_images, n_voxels]. Bootstrap over images (resample rows with replacement);
    each resample's statistic is the mean-over-voxels Pearson and mean-over-voxels MSE. Returns 95% CI
    for both. Uses the percentile method.

    Computing exact per-voxel Pearson for every one of n resamples over all voxels is O(n * n_images *
    n_voxels); a pure-numpy loop took ~4 min for n=2000 on 1000x15724 (verified during development), and
    a vectorised CPU version that materialises [chunk, n_images, n_voxels] float32 arrays either spiked
    to several GB of system RAM at a loose chunk budget or needed a tight one to stay safe. This is
    pure repeated matrix arithmetic, so it runs on the GPU instead (falls back to CPU if no CUDA):
    resampling, centering and reduction all happen as batched tensor ops in VRAM, which both avoids the
    system-RAM pressure and is faster (RTX 4090 vs a numpy loop) -- device defaults to cuda if available."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    n_images, n_voxels = v_pred.shape
    if n_voxels > voxel_subsample:
        vox_idx = rng.choice(n_voxels, voxel_subsample, replace=False)
        v_pred, v_true = v_pred[:, vox_idx], v_true[:, vox_idx]
        n_voxels = voxel_subsample

    vp = torch.tensor(v_pred, dtype=torch.float32, device=device)
    vt = torch.tensor(v_true, dtype=torch.float32, device=device)

    # Chunk over n so each chunk's [m, n_images, n_voxels] tensors fit comfortably in VRAM (a few
    # hundred MB per live tensor at m~200, n_images~1000, n_voxels~2000).
    chunk = max(1, min(n, int(3e8 // max(1, n_images * n_voxels * 4 * 4))))
    pearsons = torch.empty(n, device=device)
    mses = torch.empty(n, device=device)
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        idx = torch.randint(0, n_images, (m, n_images), device=device,
                             generator=torch.Generator(device=device).manual_seed(int(rng.integers(0, 2**31))))
        pred_r = vp[idx]   # [m, n_images, n_voxels]
        true_r = vt[idx]
        pred_c = pred_r - pred_r.mean(dim=1, keepdim=True)
        true_c = true_r - true_r.mean(dim=1, keepdim=True)
        num = (pred_c * true_c).sum(dim=1)
        denom = torch.sqrt((pred_c**2).sum(dim=1) * (true_c**2).sum(dim=1)).clamp(min=1e-12)
        r = (num / denom)   # [m, n_voxels]
        pearsons[start:start + m] = r.nanmean(dim=1)
        mses[start:start + m] = ((pred_r - true_r) ** 2).mean(dim=(1, 2))

    pearsons, mses = pearsons.cpu().numpy(), mses.cpu().numpy()
    return {
        "pearson_mean": float(pearsons.mean()), "pearson_ci95": [float(np.percentile(pearsons, 2.5)), float(np.percentile(pearsons, 97.5))],
        "mse_mean": float(mses.mean()), "mse_ci95": [float(np.percentile(mses, 2.5)), float(np.percentile(mses, 97.5))],
    }

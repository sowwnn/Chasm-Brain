"""Losses (spec §6). total = band_mse + gamma * voxel_pearson(lift(c_full, t), v_true), gamma = 0.3. No InfoNCE, no KL."""
import torch


def band_mse(c_pred: torch.Tensor, c_true: torch.Tensor, lambdas: torch.Tensor, weighting: str = "uniform"):
    """c_pred, c_true: [B, |band|]. lambdas: [|band|] eigenvalues for this band (ignored if weighting='uniform').
    weighting='spectral': each dimension weighted by 1/(1+lambda) (low-frequency dimensions matter more).
    Both weightings normalise so the loss scale is comparable across bands of different width, which
    matters here because raw per-dimension variance differs by >10x between low- and high-frequency
    bands of a Laplacian basis (verified empirically on subj01); training on raw unweighted MSE summed
    across heterogeneous-scale bands makes the optimisation unstable."""
    sq = (c_pred - c_true) ** 2
    if weighting == "uniform":
        w = torch.ones_like(lambdas)
    elif weighting == "spectral":
        w = 1.0 / (1.0 + lambdas)
    else:
        raise ValueError("weighting must be 'uniform' or 'spectral'")
    w = w / w.sum().clamp(min=1e-12) * len(w)   # keep w's mean at 1 so the loss scale doesn't shrink with band width
    return (sq * w[None, :]).mean()


def voxel_pearson(v_pred: torch.Tensor, v_true: torch.Tensor) -> torch.Tensor:
    """v_pred, v_true: [B, n_voxels]. Mean per-voxel Pearson across the batch, used as an auxiliary
    loss (so callers should pass -voxel_pearson(...) as the term to minimise, or subtract it -- this
    function returns the correlation itself, not a loss)."""
    pred_c = v_pred - v_pred.mean(dim=0, keepdim=True)
    true_c = v_true - v_true.mean(dim=0, keepdim=True)
    num = (pred_c * true_c).sum(dim=0)
    denom = torch.sqrt((pred_c**2).sum(dim=0) * (true_c**2).sum(dim=0)).clamp(min=1e-8)
    return (num / denom).mean()

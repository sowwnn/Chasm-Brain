"""Losses (spec §6). total = band_mse + gamma * voxel_pearson(lift(c_full, t), v_true), gamma = 0.3. No InfoNCE, no KL."""


def band_mse(c_pred, c_true, lambdas, weighting: str = "uniform"):
    """MSE over the band. weighting='spectral' multiplies each dimension by 1/(1+lambda) (ablation)."""
    raise NotImplementedError


def voxel_pearson(v_pred, v_true):
    """Mean per-voxel Pearson, used as auxiliary loss."""
    raise NotImplementedError

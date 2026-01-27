"""
Loss functions and metrics for 3D Flow Matching fMRI generation.

Features:
- Flow Matching loss (velocity matching)
- Peak-focused loss for high activation voxels
- Combined loss with Pearson correlation
- Evaluation metrics for both 3D and 1D fMRI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from typing import Optional, Dict, Tuple


class FlowMatching3DLoss(nn.Module):
    """
    Flow Matching loss for 3D fMRI generation.

    Components:
    1. Velocity MSE: ||v_pred - v_target||^2
    2. Peak-focused loss: Weighted MSE on high activation voxels
    3. Pearson correlation loss (optional)
    """
    def __init__(
        self,
        alpha: float = 10.0,  # Weight for peak loss
        tau: float = 0.5,     # Threshold percentile for peaks
        pearson_weight: float = 0.1,  # Weight for Pearson loss
        mask_weight: float = 2.0,  # Extra weight for brain voxels
    ):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.pearson_weight = pearson_weight
        self.mask_weight = mask_weight

    def forward(
        self,
        pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        pred_fmri: torch.Tensor,
        target_fmri: torch.Tensor,
        mean_fmri: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute flow matching loss.

        Args:
            pred_velocity: [B, 1, D, H, W] predicted velocity
            target_velocity: [B, 1, D, H, W] target velocity
            pred_fmri: [B, 1, D, H, W] predicted fMRI (x_0 + v_pred)
            target_fmri: [B, 1, D, H, W] target fMRI
            mean_fmri: [B, 1, D, H, W] mean fMRI baseline
            mask: [D, H, W] or [1, 1, D, H, W] brain mask

        Returns:
            Combined loss
        """
        # 1. Velocity MSE
        velocity_loss = F.mse_loss(pred_velocity, target_velocity)

        # 2. Peak-focused loss on predicted fMRI
        peak_loss = self._peak_focused_loss(pred_fmri, target_fmri, mean_fmri, mask)

        # 3. Pearson correlation loss
        pearson_loss = self._pearson_loss(pred_fmri, target_fmri, mask)

        # Combine losses
        total_loss = (
            velocity_loss +
            self.alpha * peak_loss +
            self.pearson_weight * pearson_loss
        )

        return total_loss

    def _peak_focused_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mean_fmri: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Peak-focused loss: higher weight on high activation voxels.

        Args:
            pred: [B, 1, D, H, W]
            target: [B, 1, D, H, W]
            mean_fmri: [B, 1, D, H, W]
            mask: [D, H, W] or [1, 1, D, H, W]

        Returns:
            Weighted MSE loss
        """
        # Deviation from mean
        target_dev = torch.abs(target - mean_fmri)

        # Threshold for peaks
        if mask is not None:
            # Only consider brain voxels for threshold
            B = target_dev.shape[0]
            if mask.dim() == 3:
                mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1, -1)
            else:
                mask_expanded = mask
            target_dev_masked = target_dev * mask_expanded
            threshold = torch.quantile(
                target_dev_masked[mask_expanded > 0.5],
                self.tau
            )
        else:
            threshold = torch.quantile(target_dev, self.tau)

        # Weight: higher for peaks
        weights = torch.where(
            target_dev > threshold,
            torch.ones_like(target_dev) * 2.0,  # 2x weight for peaks
            torch.ones_like(target_dev)
        )

        # Apply mask weight if provided
        if mask is not None:
            B = target_dev.shape[0]
            if mask.dim() == 3:
                mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1, -1).float()
            else:
                mask_expanded = mask.float()
            weights = weights * (1.0 + (self.mask_weight - 1.0) * mask_expanded)

        # Weighted MSE
        mse = (pred - target) ** 2
        weighted_mse = (mse * weights).sum() / weights.sum()

        return weighted_mse

    def _pearson_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pearson correlation loss: 1 - correlation

        Args:
            pred: [B, 1, D, H, W]
            target: [B, 1, D, H, W]
            mask: [D, H, W] or [1, 1, D, H, W]

        Returns:
            1 - mean(pearson_correlation)
        """
        B = pred.shape[0]

        if mask is not None:
            if mask.dim() == 3:
                mask_flat = mask.view(-1)
            else:
                mask_flat = mask.view(-1)
        else:
            mask_flat = None

        correlations = []
        for b in range(B):
            pred_b = pred[b].view(-1)
            target_b = target[b].view(-1)

            # Apply mask
            if mask_flat is not None:
                pred_b = pred_b[mask_flat > 0.5]
                target_b = target_b[mask_flat > 0.5]

            # Compute correlation
            if pred_b.std() > 1e-6 and target_b.std() > 1e-6:
                pred_centered = pred_b - pred_b.mean()
                target_centered = target_b - target_b.mean()
                corr = (pred_centered * target_centered).sum() / (
                    pred_centered.norm() * target_centered.norm() + 1e-8
                )
                correlations.append(corr)

        if len(correlations) > 0:
            mean_corr = torch.stack(correlations).mean()
            return 1.0 - mean_corr
        else:
            return torch.tensor(0.0, device=pred.device)


class Metrics3D1D:
    """
    Evaluation metrics for 3D fMRI that can be computed on both
    3D volumes and 1D (converted back) representations.
    """

    @staticmethod
    def compute_mse_3d(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute MSE on 3D volume.

        Args:
            pred: [B, 1, D, H, W] or [D, H, W]
            target: [B, 1, D, H, W] or [D, H, W]
            mask: [D, H, W] brain mask (optional)

        Returns:
            MSE value
        """
        if mask is not None:
            if pred.dim() == 5:
                B = pred.shape[0]
                mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1, -1)
            else:
                mask_expanded = mask

            pred_masked = pred[mask_expanded > 0.5]
            target_masked = target[mask_expanded > 0.5]
            mse = F.mse_loss(pred_masked, target_masked).item()
        else:
            mse = F.mse_loss(pred, target).item()

        return mse

    @staticmethod
    def compute_mse_1d(
        pred_1d: np.ndarray,
        target_1d: np.ndarray
    ) -> float:
        """
        Compute MSE on 1D fMRI.

        Args:
            pred_1d: [B, N] or [N]
            target_1d: [B, N] or [N]

        Returns:
            MSE value
        """
        return np.mean((pred_1d - target_1d) ** 2)

    @staticmethod
    def compute_pearson_3d(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute Pearson correlation on 3D volume.

        Args:
            pred: [B, 1, D, H, W]
            target: [B, 1, D, H, W]
            mask: [D, H, W] brain mask

        Returns:
            dict with 'mean', 'std', 'min', 'max' correlations
        """
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        B = pred_np.shape[0]
        correlations = []

        for b in range(B):
            pred_b = pred_np[b].flatten()
            target_b = target_np[b].flatten()

            # Apply mask
            if mask is not None:
                mask_flat = mask.cpu().numpy().flatten()
                pred_b = pred_b[mask_flat > 0.5]
                target_b = target_b[mask_flat > 0.5]

            # Compute correlation
            if pred_b.std() > 1e-6 and target_b.std() > 1e-6:
                corr, _ = pearsonr(pred_b, target_b)
                correlations.append(corr)

        if len(correlations) > 0:
            return {
                'mean': np.mean(correlations),
                'std': np.std(correlations),
                'min': np.min(correlations),
                'max': np.max(correlations)
            }
        else:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

    @staticmethod
    def compute_pearson_1d(
        pred_1d: np.ndarray,
        target_1d: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute Pearson correlation on 1D fMRI (per sample).

        Args:
            pred_1d: [B, N] or [N]
            target_1d: [B, N] or [N]

        Returns:
            dict with 'mean', 'std', 'min', 'max' correlations
        """
        if pred_1d.ndim == 1:
            pred_1d = pred_1d.reshape(1, -1)
            target_1d = target_1d.reshape(1, -1)

        B = pred_1d.shape[0]
        correlations = []

        for b in range(B):
            if pred_1d[b].std() > 1e-6 and target_1d[b].std() > 1e-6:
                corr, _ = pearsonr(pred_1d[b], target_1d[b])
                correlations.append(corr)

        if len(correlations) > 0:
            return {
                'mean': np.mean(correlations),
                'std': np.std(correlations),
                'min': np.min(correlations),
                'max': np.max(correlations)
            }
        else:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

    @staticmethod
    def compute_all_metrics(
        pred_3d: torch.Tensor,
        target_3d: torch.Tensor,
        pred_1d: Optional[np.ndarray] = None,
        target_1d: Optional[np.ndarray] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """
        Compute all metrics on both 3D and 1D representations.

        Args:
            pred_3d: [B, 1, D, H, W] predicted 3D fMRI
            target_3d: [B, 1, D, H, W] target 3D fMRI
            pred_1d: [B, N] predicted 1D fMRI (optional)
            target_1d: [B, N] target 1D fMRI (optional)
            mask: [D, H, W] brain mask

        Returns:
            dict with all metrics
        """
        metrics = {}

        # 3D metrics
        metrics['mse_3d'] = Metrics3D1D.compute_mse_3d(pred_3d, target_3d, mask)
        pearson_3d = Metrics3D1D.compute_pearson_3d(pred_3d, target_3d, mask)
        metrics['pearson_3d_mean'] = pearson_3d['mean']
        metrics['pearson_3d_std'] = pearson_3d['std']

        # 1D metrics (if provided)
        if pred_1d is not None and target_1d is not None:
            metrics['mse_1d'] = Metrics3D1D.compute_mse_1d(pred_1d, target_1d)
            pearson_1d = Metrics3D1D.compute_pearson_1d(pred_1d, target_1d)
            metrics['pearson_1d_mean'] = pearson_1d['mean']
            metrics['pearson_1d_std'] = pearson_1d['std']

            # Verify lossless conversion
            mse_diff = abs(metrics['mse_3d'] - metrics['mse_1d'])
            pearson_diff = abs(metrics['pearson_3d_mean'] - metrics['pearson_1d_mean'])
            metrics['conversion_mse_diff'] = mse_diff
            metrics['conversion_pearson_diff'] = pearson_diff

        return metrics


def test_lossless_conversion():
    """
    Test that 3D ↔ 1D conversion is lossless for metrics.
    """
    print("Testing lossless 3D ↔ 1D conversion...")

    # Create dummy data
    compact_shape = (61, 46, 42)
    n_voxels = 15724
    B = 4

    # Create mask
    mask = torch.zeros(compact_shape, dtype=torch.bool)
    coords = torch.randint(0, min(compact_shape), (n_voxels, 3))
    for coord in coords:
        mask[coord[0], coord[1], coord[2]] = True

    # Create 1D data
    pred_1d = torch.randn(B, n_voxels)
    target_1d = torch.randn(B, n_voxels)

    # Convert to 3D
    pred_3d = torch.zeros(B, 1, *compact_shape)
    target_3d = torch.zeros(B, 1, *compact_shape)

    for b in range(B):
        for i, coord in enumerate(coords):
            pred_3d[b, 0, coord[0], coord[1], coord[2]] = pred_1d[b, i]
            target_3d[b, 0, coord[0], coord[1], coord[2]] = target_1d[b, i]

    # Compute metrics
    metrics = Metrics3D1D.compute_all_metrics(
        pred_3d, target_3d,
        pred_1d.numpy(), target_1d.numpy(),
        mask
    )

    print("\nMetrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")

    # Check lossless
    assert metrics['conversion_mse_diff'] < 1e-6, "MSE should match!"
    assert metrics['conversion_pearson_diff'] < 1e-4, "Pearson should match!"

    print("\n✓ Conversion is lossless! Metrics match on 3D and 1D.")


if __name__ == '__main__':
    test_lossless_conversion()

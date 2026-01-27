"""
Loss functions for Dual-Stream CNN fMRI model.

Features:
- Direct MSE loss on 1D fMRI predictions
- Peak-focused loss for high activation voxels
- Pearson correlation loss
- Optional consistency loss between streams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from typing import Optional, Dict


class DualStreamLoss(nn.Module):
    """
    Loss function for Dual-Stream CNN.

    Components:
    1. MSE loss on predicted fMRI
    2. Peak-focused loss (higher weight on high-activation voxels)
    3. Pearson correlation loss
    4. Optional: consistency loss between 3D and 1D predictions
    """
    def __init__(
        self,
        alpha: float = 10.0,  # Weight for peak loss
        tau: float = 0.5,  # Threshold percentile for peaks
        pearson_weight: float = 0.1,  # Weight for Pearson loss
        consistency_weight: float = 0.0,  # Weight for stream consistency (0 to disable)
    ):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.pearson_weight = pearson_weight
        self.consistency_weight = consistency_weight

    def forward(
        self,
        pred_fmri: torch.Tensor,
        target_fmri: torch.Tensor,
        mean_fmri: torch.Tensor,
        feat_3d: Optional[torch.Tensor] = None,
        feat_1d: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute dual-stream loss.

        Args:
            pred_fmri: [B, N] predicted 1D fMRI
            target_fmri: [B, N] target 1D fMRI
            mean_fmri: [B, N] mean fMRI baseline
            feat_3d: [B, C_3d, D, H, W] 3D features (optional, for consistency)
            feat_1d: [B, C_1d, N'] 1D features (optional, for consistency)

        Returns:
            Combined loss
        """
        # 1. MSE loss
        mse_loss = F.mse_loss(pred_fmri, target_fmri)

        # 2. Peak-focused loss
        peak_loss = self._peak_focused_loss(pred_fmri, target_fmri, mean_fmri)

        # 3. Pearson correlation loss
        pearson_loss = self._pearson_loss(pred_fmri, target_fmri)

        # 4. Consistency loss (optional)
        if self.consistency_weight > 0 and feat_3d is not None and feat_1d is not None:
            consistency_loss = self._consistency_loss(feat_3d, feat_1d)
        else:
            consistency_loss = torch.tensor(0.0, device=pred_fmri.device)

        # Combine losses
        total_loss = (
            mse_loss +
            self.alpha * peak_loss +
            self.pearson_weight * pearson_loss +
            self.consistency_weight * consistency_loss
        )

        return total_loss

    def _peak_focused_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mean_fmri: torch.Tensor
    ) -> torch.Tensor:
        """
        Peak-focused loss: higher weight on high activation voxels.

        Args:
            pred: [B, N]
            target: [B, N]
            mean_fmri: [B, N]

        Returns:
            Weighted MSE loss
        """
        # Deviation from mean
        target_dev = torch.abs(target - mean_fmri)

        # Threshold for peaks (per sample)
        threshold = torch.quantile(target_dev, self.tau, dim=1, keepdim=True)

        # Weight: higher for peaks
        weights = torch.where(
            target_dev > threshold,
            torch.ones_like(target_dev) * 2.0,  # 2x weight for peaks
            torch.ones_like(target_dev)
        )

        # Weighted MSE
        mse = (pred - target) ** 2
        weighted_mse = (mse * weights).sum() / weights.sum()

        return weighted_mse

    def _pearson_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Pearson correlation loss: 1 - correlation

        Args:
            pred: [B, N]
            target: [B, N]

        Returns:
            1 - mean(pearson_correlation)
        """
        B = pred.shape[0]

        correlations = []
        for b in range(B):
            pred_b = pred[b]
            target_b = target[b]

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

    def _consistency_loss(
        self,
        feat_3d: torch.Tensor,
        feat_1d: torch.Tensor
    ) -> torch.Tensor:
        """
        Consistency loss between 3D and 1D stream features.

        Encourages both streams to learn similar representations.

        Args:
            feat_3d: [B, C_3d, D, H, W]
            feat_1d: [B, C_1d, N]

        Returns:
            Consistency loss (cosine distance)
        """
        # Global average pooling on both streams
        feat_3d_pooled = F.adaptive_avg_pool3d(feat_3d, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # [B, C_3d]
        feat_1d_pooled = F.adaptive_avg_pool1d(feat_1d, 1).squeeze(-1)  # [B, C_1d]

        # Normalize
        feat_3d_norm = F.normalize(feat_3d_pooled, p=2, dim=1)
        feat_1d_norm = F.normalize(feat_1d_pooled, p=2, dim=1)

        # Project to common dimension if needed
        if feat_3d_norm.shape[1] != feat_1d_norm.shape[1]:
            min_dim = min(feat_3d_norm.shape[1], feat_1d_norm.shape[1])
            feat_3d_norm = feat_3d_norm[:, :min_dim]
            feat_1d_norm = feat_1d_norm[:, :min_dim]

        # Cosine similarity
        cosine_sim = (feat_3d_norm * feat_1d_norm).sum(dim=1).mean()

        # Loss: maximize similarity = minimize negative similarity
        return 1.0 - cosine_sim


class DualStreamMetrics:
    """
    Evaluation metrics for Dual-Stream CNN.
    """

    @staticmethod
    def compute_mse(
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """
        Compute MSE.

        Args:
            pred: [B, N] or [N]
            target: [B, N] or [N]

        Returns:
            MSE value
        """
        return F.mse_loss(pred, target).item()

    @staticmethod
    def compute_pearson(
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute Pearson correlation (per sample).

        Args:
            pred: [B, N]
            target: [B, N]

        Returns:
            dict with 'mean', 'std', 'min', 'max' correlations
        """
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        if pred_np.ndim == 1:
            pred_np = pred_np.reshape(1, -1)
            target_np = target_np.reshape(1, -1)

        B = pred_np.shape[0]
        correlations = []

        for b in range(B):
            if pred_np[b].std() > 1e-6 and target_np[b].std() > 1e-6:
                corr, _ = pearsonr(pred_np[b], target_np[b])
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
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            pred: [B, N] predicted fMRI
            target: [B, N] target fMRI

        Returns:
            dict with all metrics
        """
        metrics = {}

        # MSE
        metrics['mse'] = DualStreamMetrics.compute_mse(pred, target)

        # Pearson
        pearson = DualStreamMetrics.compute_pearson(pred, target)
        metrics['pearson_mean'] = pearson['mean']
        metrics['pearson_std'] = pearson['std']
        metrics['pearson_min'] = pearson['min']
        metrics['pearson_max'] = pearson['max']

        return metrics


def test_loss():
    """Test loss function"""
    print("Testing DualStreamLoss...")

    # Config
    batch_size = 4
    n_voxels = 15724

    # Create dummy data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pred_fmri = torch.randn(batch_size, n_voxels).to(device)
    target_fmri = torch.randn(batch_size, n_voxels).to(device)
    mean_fmri = torch.randn(batch_size, n_voxels).to(device)

    # Optional features
    feat_3d = torch.randn(batch_size, 256, 8, 6, 5).to(device)
    feat_1d = torch.randn(batch_size, 512, 1000).to(device)

    # Create loss
    criterion = DualStreamLoss(
        alpha=10.0,
        tau=0.5,
        pearson_weight=0.1,
        consistency_weight=0.1
    )

    # Compute loss
    loss = criterion(pred_fmri, target_fmri, mean_fmri, feat_3d, feat_1d)

    print(f"Loss: {loss.item():.4f}")

    # Test without consistency
    criterion_no_cons = DualStreamLoss(
        alpha=10.0,
        tau=0.5,
        pearson_weight=0.1,
        consistency_weight=0.0
    )
    loss_no_cons = criterion_no_cons(pred_fmri, target_fmri, mean_fmri)
    print(f"Loss (no consistency): {loss_no_cons.item():.4f}")

    # Test metrics
    print("\nTesting metrics...")
    metrics = DualStreamMetrics.compute_all_metrics(pred_fmri, target_fmri)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n✓ DualStreamLoss test passed!")


if __name__ == '__main__':
    test_loss()

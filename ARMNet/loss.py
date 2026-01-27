import torch
import torch.nn as nn
import torch.nn.functional as F

class PeakFocusedLoss(nn.Module):
    """
    Peak-Focused Adaptive Loss (PFAL)
    
    Ý tưởng: 
    - Phạt nhẹ nếu sai ở gần giá trị Mean (Baseline).
    - Phạt nặng nếu không khớp được các đỉnh (Peak) của Ground Truth.
    - Thưởng (giảm loss nhanh) nếu dự đoán đúng xu hướng của Peak.
    """
    def __init__(self, alpha=3.0, tau=0.5, pearson_weight=0.5, std_weight=1.0):
        """
        Args:
            alpha: Hệ số khuếch đại cho các vùng Peak.
            tau: Ngưỡng chênh lệch so với Mean để bắt đầu coi là Peak.
            pearson_weight: Trọng số cho hàm loss dựa trên tương quan Pearson.
            std_weight: Trọng số ép biên độ (std) của dự đoán khớp với thực tế.
        """
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.pearson_weight = pearson_weight
        self.std_weight = std_weight

    def forward(self, pred, target, mean_fmri):
        """
        Args:
            pred: [B, N] dự đoán của mô hình.
            target: [B, N] ground truth.
            mean_fmri: [B, N] hoặc [1, N] giá trị trung bình voxel.
        """
        # 1. Base MSE per voxel
        mse_loss = (pred - target) ** 2
        
        # 2. Tính độ lệch của GT so với Mean để xác định đâu là Peak
        with torch.no_grad():
            deviation = torch.abs(target - mean_fmri)
            # Trọng số tăng mạnh khi deviation > tau
            # Trọng số min là 1.0 (cho vùng gần Mean)
            weights = 1.0 + self.alpha * torch.relu(deviation - self.tau)
        
        # 3. Weighted MSE
        weighted_mse = (mse_loss * weights).mean()

        # 4. Pearson Correlation Loss (để khớp "hình dáng" tín hiệu)
        # 1 - corr (giá trị từ 0 đến 2, mong muốn về 0)
        p_loss = 0
        if self.pearson_weight > 0:
            # Centering
            pred_c = pred - pred.mean(dim=1, keepdim=True)
            target_c = target - target.mean(dim=1, keepdim=True)

            # Cosine similarity của centered vectors = Pearson correlation
            sim = F.cosine_similarity(pred_c, target_c, dim=1)
            p_loss = (1 - sim).mean()

        # 5. Amplitude (STD) Matching Loss
        # Ép độ lệch chuẩn của dự đoán phải tương đồng với target
        std_loss = 0
        if self.std_weight > 0:
            pred_std = pred.std(dim=1)
            target_std = target.std(dim=1)
            std_loss = F.mse_loss(pred_std, target_std)

        # Total Loss
        total_loss = (1 - self.pearson_weight) * weighted_mse + \
                     self.pearson_weight * p_loss + \
                     self.std_weight * std_loss

        return total_loss

class FlowMatchingLoss(nn.Module):
    """
    Loss cho Flow Matching, kết hợp giữa Velocity MSE, Peak Focused Loss và Diversity Loss.
    """
    def __init__(self, alpha=3.0, tau=0.5, pearson_weight=0.5):
        super().__init__()
        self.peak_loss_fn = PeakFocusedLoss(alpha=alpha, tau=tau, pearson_weight=pearson_weight)

    def forward(self, v_pred, v_target, x_pred_final, x_target, mean_fmri):
        """
        Args:
            v_pred: Vận tốc dự đoán [B, N]
            v_target: Vận tốc thực (x_1 - x_0) [B, N]
            x_pred_final: Ước lượng x_1 từ v_pred (x_0 + v_pred) [B, N]
            x_target: Target fMRI thực tế (x_1) [B, N]
            mean_fmri: Baseline (x_0) [B, N]
        """
        # 1. Velocity MSE Loss
        velocity_loss = F.mse_loss(v_pred, v_target)
        
        # 2. Peak Focused Loss
        peak_loss = self.peak_loss_fn(x_pred_final, x_target, mean_fmri)
        
        # 3. INNER-SAMPLE STD (Độ biến thiên trong 1 người)
        target_std_inner = x_target.std(dim=1).mean().detach()
        pred_std_inner = x_pred_final.std(dim=1).mean()
        std_inner_loss = F.mse_loss(pred_std_inner, target_std_inner) 

        # 4. CROSS-SAMPLE DIVERSITY (Độ biến thiên giữa các bức ảnh khác nhau)
        # Ép độ lệch chuẩn trên từng voxel qua các mẫu trong batch phải lớn
        target_std_cross = x_target.std(dim=0).mean().detach()
        pred_std_cross = x_pred_final.std(dim=0).mean()
        diversity_loss = F.mse_loss(pred_std_cross, target_std_cross)

        # Trọng số cho Diversity phải đủ lớn để phá vỡ collapse
        return velocity_loss + 1.0 * peak_loss + 0.5 * std_inner_loss + 2.0 * diversity_loss

class CVAELoss(nn.Module):
    """
    Loss for CVAE: Reconstruction (PeakFocusedLoss) + KL Divergence.
    """
    def __init__(self, alpha=3.0, tau=0.5, pearson_weight=0.5, kld_weight=0.01):
        super().__init__()
        self.peak_loss_fn = PeakFocusedLoss(alpha=alpha, tau=tau, pearson_weight=pearson_weight)
        self.kld_weight = kld_weight

    def forward(self, pred, target, mean_fmri, mu, logvar):
        """
        Args:
            pred: [B, N]
            target: [B, N]
            mean_fmri: [B, N]
            mu: [B, L]
            logvar: [B, L]
        """
        # 1. Reconstruction Loss (using existing PeakFocusedLoss)
        recon_loss = self.peak_loss_fn(pred, target, mean_fmri)

        # 2. KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = kld_loss / pred.size(0) # Average over batch

        return recon_loss + self.kld_weight * kld_loss, recon_loss, kld_loss


class DiffusionPeakFocusedLoss(nn.Module):
    """
    Peak-Focused Loss cho Diffusion 3D fMRI Generation.

    Thiết kế cho context diffusion:
    - Phạt nặng lỗi tại Peak regions (voxels có activation cao)
    - Thêm Diversity Loss để tránh mode collapse
    - Hỗ trợ cả noise prediction và x0 prediction targets

    Ý tưởng:
    - Peak voxels (deviation > tau từ mean) quan trọng hơn cho neural decoding
    - Diffusion dễ bị mode collapse → cần diversity regularization
    - Cross-sample diversity đảm bảo model học được variation giữa các images
    """
    def __init__(
        self,
        alpha: float = 2.0,           # Peak amplification (thấp hơn cho diffusion vì noise)
        tau: float = 0.3,             # Threshold để xác định peak (normalized)
        pearson_weight: float = 0.3,  # Correlation loss weight
        std_weight: float = 0.5,      # Within-sample amplitude matching
        diversity_weight: float = 1.0, # Cross-sample diversity (anti-collapse)
        l1_weight: float = 0.1,       # L1 sparsity regularization
    ):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.pearson_weight = pearson_weight
        self.std_weight = std_weight
        self.diversity_weight = diversity_weight
        self.l1_weight = l1_weight

    def forward(self, pred, target, mean_fmri=None):
        """
        Args:
            pred: [B, N] predicted fMRI (denoised or x0 estimate)
            target: [B, N] ground truth fMRI
            mean_fmri: [B, N] or [1, N] baseline mean fMRI.
                       If None, uses per-sample mean of target.
        Returns:
            total_loss: scalar tensor
            loss_dict: dict with individual loss components for logging
        """
        batch_size = pred.shape[0]

        # Fallback mean if not provided
        if mean_fmri is None:
            mean_fmri = target.mean(dim=0, keepdim=True)

        # ========== 1. Peak-Weighted MSE Loss ==========
        mse_per_voxel = (pred - target) ** 2

        with torch.no_grad():
            # Compute deviation from baseline to identify peaks
            deviation = torch.abs(target - mean_fmri)
            # Normalize deviation per sample for stable thresholding
            dev_std = deviation.std(dim=1, keepdim=True).clamp(min=1e-6)
            deviation_norm = deviation / dev_std
            # Weights: minimum 1.0, increases for peaks
            weights = 1.0 + self.alpha * torch.relu(deviation_norm - self.tau)

        weighted_mse = (mse_per_voxel * weights).mean()

        # ========== 2. Pearson Correlation Loss ==========
        pearson_loss = torch.tensor(0.0, device=pred.device)
        if self.pearson_weight > 0:
            pred_centered = pred - pred.mean(dim=1, keepdim=True)
            target_centered = target - target.mean(dim=1, keepdim=True)

            # Cosine similarity of centered vectors = Pearson correlation
            sim = F.cosine_similarity(pred_centered, target_centered, dim=1)
            pearson_loss = (1 - sim).mean()

        # ========== 3. Within-Sample STD Matching ==========
        std_loss = torch.tensor(0.0, device=pred.device)
        if self.std_weight > 0:
            pred_std = pred.std(dim=1)
            target_std = target.std(dim=1).detach()
            std_loss = F.mse_loss(pred_std, target_std)

        # ========== 4. Cross-Sample Diversity Loss (Anti-Collapse) ==========
        diversity_loss = torch.tensor(0.0, device=pred.device)
        if self.diversity_weight > 0 and batch_size > 1:
            # STD across samples for each voxel
            pred_cross_std = pred.std(dim=0).mean()
            target_cross_std = target.std(dim=0).mean().detach()
            diversity_loss = F.mse_loss(pred_cross_std, target_cross_std)

        # ========== 5. L1 Sparsity (optional) ==========
        l1_loss = torch.tensor(0.0, device=pred.device)
        if self.l1_weight > 0:
            # L1 on the difference from mean (encourage sparse deviations)
            l1_loss = torch.abs(pred - mean_fmri).mean()

        # ========== Total Loss ==========
        total_loss = (
            weighted_mse +
            self.pearson_weight * pearson_loss +
            self.std_weight * std_loss +
            self.diversity_weight * diversity_loss +
            self.l1_weight * l1_loss
        )

        # Return loss dict for logging
        loss_dict = {
            'total': total_loss.item(),
            'weighted_mse': weighted_mse.item(),
            'pearson': pearson_loss.item(),
            'std': std_loss.item(),
            'diversity': diversity_loss.item(),
            'l1': l1_loss.item(),
        }

        return total_loss, loss_dict

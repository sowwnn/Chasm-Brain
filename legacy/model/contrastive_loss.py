
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for Contrastive Learning.
    Maximizes similarity between positive pairs (pred_i, target_i)
    while minimizing similarity with negative pairs (pred_i, target_j).
    """
    def __init__(self, temperature=0.07, learnable_temp=False):
        super().__init__()
        self.learnable_temp = learnable_temp
        if learnable_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        else:
            self.temperature = temperature

    def forward(self, pred, target):
        """
        Args:
            pred: [B, D] Predicted features
            target: [B, D] Target features (Ground Truth)
        """
        # Normalize with epsilon for stability
        pred_norm = F.normalize(pred, dim=-1, eps=1e-8)
        target_norm = F.normalize(target, dim=-1, eps=1e-8)

        if self.learnable_temp:
            # Clamp logit_scale to prevent extreme temperatures
            scale = torch.clamp(self.logit_scale.exp(), max=100.0)
            logits = (pred_norm @ target_norm.T) * scale
        else:
            logits = (pred_norm @ target_norm.T) / self.temperature

        # Clamp logits to prevent overflow in exp() for fp16
        # FP16 max: 65504, but exp(11) ≈ 59874, so we clamp to ±10
        logits = torch.clamp(logits, min=-10.0, max=10.0)

        # Targets are diagonal (0, 1, 2...)
        labels = torch.arange(logits.shape[0], device=logits.device)

        # Cross Entropy (symmetric or asymmetric)
        # CLIP uses symmetric: (loss_i2t + loss_t2i) / 2
        loss_p2t = F.cross_entropy(logits, labels)
        loss_t2p = F.cross_entropy(logits.T, labels)

        return (loss_p2t + loss_t2p) / 2

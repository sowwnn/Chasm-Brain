"""CLS + patch pathways, MLP only (spec §5). No Mamba, no attention."""
import torch
import torch.nn as nn


class DualPathEncoder(nn.Module):
    """CLS: 2-layer MLP + GELU. Patch: linear projection + learned positional embedding + 2-layer MLP
    + mean-pool. n_patches must be known at construction time (DINOv2 gives 256; CLIP/ResNet in E6
    give a different count and need their own encoder instance) -- the positional embedding is a
    real nn.Parameter from __init__ so it is always registered with the optimizer, never created
    lazily inside forward (a lazily-created nn.Parameter silently misses any optimizer built before
    the first forward call, which starves it of gradient updates)."""

    def __init__(self, d_in: int, d_hidden: int, n_patches: int = 256):
        super().__init__()
        self.d_in, self.d_hidden, self.n_patches = d_in, d_hidden, n_patches
        self.cls_mlp = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden))
        self.patch_proj = nn.Linear(d_in, d_hidden)
        self.patch_mlp = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_hidden) * 0.02)

    def forward(self, cls, patch):
        """cls [B, d], patch [B, n_patches, d] -> (h_cls [B, dh], h_patch [B, dh], h_patch_tokens [B, n_patches, dh])."""
        if patch.shape[1] != self.n_patches:
            raise ValueError(f"expected {self.n_patches} patches, got {patch.shape[1]}")
        h_cls = self.cls_mlp(cls)
        h_patch_tokens = self.patch_mlp(self.patch_proj(patch) + self.pos_embed)
        h_patch = h_patch_tokens.mean(dim=1)
        return h_cls, h_patch, h_patch_tokens


class FusionGate(nn.Module):
    """Softmax over the two pathways; kept for the H1 analysis (spec §10 E4)."""

    def __init__(self, d_hidden: int):
        super().__init__()
        self.linear = nn.Linear(2 * d_hidden, 2)

    def forward(self, h_cls, h_patch):
        """-> alpha [B, 1], weight on h_cls (1 - alpha is the weight on h_patch)."""
        logits = self.linear(torch.cat([h_cls, h_patch], dim=-1))
        return torch.softmax(logits, dim=-1)[:, :1]

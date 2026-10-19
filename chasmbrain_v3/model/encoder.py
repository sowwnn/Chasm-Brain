"""CLS + patch pathways, MLP only (spec §5). No Mamba, no attention."""
import torch.nn as nn


class DualPathEncoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, cls, patch):
        """cls [B, d], patch [B, 256, d] -> (h_cls [B, dh], h_patch [B, dh], h_patch_tokens [B, 256, dh])."""
        raise NotImplementedError


class FusionGate(nn.Module):
    """Softmax over the two pathways; kept for the H1 analysis (spec §10 E4)."""

    def __init__(self, d_hidden: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, h_cls, h_patch):
        """-> alpha [B, 1]."""
        raise NotImplementedError

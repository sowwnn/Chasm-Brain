"""Matched two-stage MLP-Mixer control for HierarchicalARM V3.

The hierarchy, tokenizers, VAE, ROI conditioning, global/local heads, and output
fusion are identical to the Mamba model. Only the sequence-processing blocks in
both stages are replaced by residual token/channel MLP mixers.
"""

import torch
import torch.nn as nn

from .hierarchical_v3_1 import (
    DualStreamMambaStage1,
    HierarchicalARM_V3,
    MambaStage2,
)


class MLPMixerDelta(nn.Module):
    """Return the residual delta of a pre-norm MLP-Mixer block."""

    def __init__(self, sequence_length, embed_dim, channel_expansion=3, token_hidden=64):
        super().__init__()
        self.token_norm = nn.LayerNorm(embed_dim)
        self.token_mlp = None
        if sequence_length > 1 and token_hidden > 0:
            self.token_mlp = nn.Sequential(
                nn.Linear(sequence_length, token_hidden),
                nn.GELU(),
                nn.Linear(token_hidden, sequence_length),
            )
        channel_hidden = channel_expansion * embed_dim
        self.channel_norm = nn.LayerNorm(embed_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(embed_dim, channel_hidden),
            nn.GELU(),
            nn.Linear(channel_hidden, embed_dim),
        )

    def forward(self, x):
        token_delta = torch.zeros_like(x)
        if self.token_mlp is not None:
            token_delta = self.token_mlp(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        channel_delta = self.channel_mlp(self.channel_norm(x + token_delta))
        return token_delta + channel_delta


class DualStreamMLPStage1(DualStreamMambaStage1):
    """Stage 1 with matched residual MLP blocks in both visual streams."""

    def __init__(self, *args, depth_what=2, depth_where=4, **kwargs):
        super().__init__(*args, depth_what=depth_what, depth_where=depth_where, **kwargs)
        self.what_mamba = nn.ModuleList([
            MLPMixerDelta(1, self.hidden_dim, token_hidden=0)
            for _ in range(depth_what)
        ])
        self.where_mamba = nn.ModuleList([
            MLPMixerDelta(self.num_patches, self.hidden_dim, token_hidden=128)
            for _ in range(depth_where)
        ])
        self._init_weights()


class MLPStage2(MambaStage2):
    """Stage 2 with MLP-Mixer blocks and otherwise unchanged decoding."""

    def __init__(self, *args, depth=6, **kwargs):
        super().__init__(*args, depth=depth, **kwargs)
        memory_length = self.num_rois + 1
        global_length = memory_length + self.num_global_queries
        self.backbone = nn.ModuleList([
            MLPMixerDelta(memory_length, self.embed_dim, token_hidden=64)
            for _ in range(depth)
        ])
        self.global_mamba = nn.ModuleList([
            MLPMixerDelta(global_length, self.embed_dim, token_hidden=64)
            for _ in range(2)
        ])


class HierarchicalARM_V3_MLP(HierarchicalARM_V3):
    """Two-stage coarse-to-fine MLP control matched to HierarchicalARM V3."""

    def __init__(
        self,
        cls_dim=768,
        patch_dim=768,
        num_patches=256,
        fmri_dim=15724,
        voxels_per_cluster=30,
        hidden_dim_1=512,
        depth_what=2,
        depth_where=4,
        contrastive_dim=256,
        embed_dim_2=384,
        latent_dim=128,
        depth_2=6,
        num_global_queries=128,
        dropout=0.1,
        roi_top_k_percent=0.7,
        vae_mode="stochastic",
    ):
        nn.Module.__init__(self)
        self.fmri_dim = fmri_dim
        self.voxels_per_cluster = voxels_per_cluster
        self.num_rois = fmri_dim // voxels_per_cluster

        self.stage1 = DualStreamMLPStage1(
            cls_dim=cls_dim,
            patch_dim=patch_dim,
            num_patches=num_patches,
            fmri_dim=fmri_dim,
            voxels_per_cluster=voxels_per_cluster,
            hidden_dim=hidden_dim_1,
            depth_what=depth_what,
            depth_where=depth_where,
            contrastive_dim=contrastive_dim,
            dropout=dropout,
            roi_top_k_percent=roi_top_k_percent,
        )

        visual_dim_2 = cls_dim + hidden_dim_1 // 2
        self.stage2 = MLPStage2(
            visual_dim=visual_dim_2,
            num_rois=self.num_rois,
            fmri_dim=fmri_dim,
            voxels_per_cluster=voxels_per_cluster,
            embed_dim=embed_dim_2,
            latent_dim=latent_dim,
            depth=depth_2,
            num_global_queries=num_global_queries,
            dropout=dropout,
            vae_mode=vae_mode,
        )
        self.patch_pooler = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim_1 // 2),
            nn.LayerNorm(hidden_dim_1 // 2),
            nn.GELU(),
        )


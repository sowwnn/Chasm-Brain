"""Matched two-stage Transformer control for HierarchicalARM V3.

The hierarchy, ROI conditioning, VAE path, prediction heads, and fusion are
unchanged. Only Mamba sequence blocks are replaced by Transformer encoder
layers with a parameter budget close to the original blocks.
"""

import torch.nn as nn

from .hierarchical_v3_1 import (
    DualStreamMambaStage1,
    HierarchicalARM_V3,
    MambaStage2,
)


class TransformerDelta(nn.Module):
    """Adapt a residual Transformer layer to the model's outer residual loop."""

    def __init__(self, embed_dim, num_heads=8, ffn_multiplier=1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ffn_multiplier,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x):
        return self.layer(x) - x


class DualStreamTransformerStage1(DualStreamMambaStage1):
    """Stage 1 with matched Transformer blocks in the what/where streams."""

    def __init__(self, *args, depth_what=2, depth_where=4, **kwargs):
        super().__init__(*args, depth_what=depth_what, depth_where=depth_where, **kwargs)
        self.what_mamba = nn.ModuleList([
            TransformerDelta(self.hidden_dim) for _ in range(depth_what)
        ])
        self.where_mamba = nn.ModuleList([
            TransformerDelta(self.hidden_dim) for _ in range(depth_where)
        ])
        self._init_weights()


class TransformerStage2(MambaStage2):
    """Stage 2 with Transformer blocks and unchanged coarse-to-fine decoding."""

    def __init__(self, *args, depth=6, **kwargs):
        super().__init__(*args, depth=depth, **kwargs)
        self.backbone = nn.ModuleList([
            TransformerDelta(self.embed_dim) for _ in range(depth)
        ])
        self.global_mamba = nn.ModuleList([
            TransformerDelta(self.embed_dim) for _ in range(2)
        ])


class HierarchicalARM_V3_Transformer(HierarchicalARM_V3):
    """Two-stage coarse-to-fine Transformer matched to HierarchicalARM V3."""

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

        self.stage1 = DualStreamTransformerStage1(
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
        self.stage2 = TransformerStage2(
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


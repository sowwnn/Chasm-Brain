"""
HierarchicalARM V3: Dual-Stream Mamba Architecture

Key changes from V2:
1. Stage 1: Dual-stream processing (What + Where branches)
   - What branch: CLS token → object identity → ROIs liên quan object chính
   - Where branch: Patch tokens → spatial features → ROIs chi tiết
2. voxels_per_cluster parameter thay vì fix num_clusters
3. Chỉ dùng Mamba từ mamba-ssm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mamba_ssm import Mamba


class ContrastiveHead(nn.Module):
    """Projector for Contrastive Learning (CLIP-style)."""
    def __init__(self, input_dim, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class DualStreamMambaStage1(nn.Module):
    """
    Stage 1 V3: Dual-Stream Mamba Architecture.

    Input: CLS token (what) + Patch tokens (where)
    Output: ROI means

    Design rationale:
    - What branch (CLS): DINOv2 CLS captures global semantic = "what object"
      → Maps to ROIs directly related to the main object
    - Where branch (Patches): DINOv2 patches capture spatial features = "where/details"
      → Maps to ROIs with fine-grained spatial information
    """
    def __init__(
        self,
        cls_dim=768,
        patch_dim=768,
        num_patches=256,
        fmri_dim=15724,
        voxels_per_cluster=30,
        hidden_dim=512,
        depth_what=2,
        depth_where=4,
        contrastive_dim=256,
        dropout=0.1,
        roi_top_k_percent=0.7,
    ):
        super().__init__()

        self.cls_dim = cls_dim
        self.patch_dim = patch_dim
        self.num_patches = num_patches
        self.fmri_dim = fmri_dim
        self.voxels_per_cluster = voxels_per_cluster
        self.hidden_dim = hidden_dim
        self.roi_top_k_percent = roi_top_k_percent

        # Compute number of ROIs
        self.num_rois = fmri_dim // voxels_per_cluster
        self.remainder_voxels = fmri_dim % voxels_per_cluster

        print(f"DualStreamMambaStage1: {fmri_dim} voxels → {self.num_rois} ROIs "
              f"({voxels_per_cluster} voxels/ROI, {self.remainder_voxels} remainder)")

        # ========== What Branch (CLS → Object Identity) ==========
        self.what_proj = nn.Linear(cls_dim, hidden_dim)
        self.what_norm = nn.LayerNorm(hidden_dim)
        self.what_mamba = nn.ModuleList([
            Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(depth_what)
        ])
        self.what_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.num_rois),
        )

        # ========== Where Branch (Patches → Spatial Details) ==========
        self.where_proj = nn.Linear(patch_dim, hidden_dim)
        self.where_pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
        self.where_norm = nn.LayerNorm(hidden_dim)
        self.where_mamba = nn.ModuleList([
            Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(depth_where)
        ])
        self.where_pool = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.where_head = nn.Linear(hidden_dim // 2, self.num_rois)

        # ========== Fusion ==========
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )

        # ========== Contrastive Alignment ==========
        # combined_feat = what_feat (hidden_dim) + where_pooled (hidden_dim // 2)
        self.img_contrastive_head = ContrastiveHead(hidden_dim + hidden_dim // 2, contrastive_dim)
        self.brain_contrastive_head = ContrastiveHead(self.num_rois, contrastive_dim)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cls_token, patch_tokens, gt_roi_means=None):
        """
        Args:
            cls_token: [B, cls_dim] - CLS token from DINOv2
            patch_tokens: [B, num_patches, patch_dim] - Patch tokens from DINOv2
            gt_roi_means: [B, num_rois] - Ground truth ROI means (for contrastive)

        Returns:
            pred_roi_means: [B, num_rois]
            img_proj: [B, contrastive_dim]
            brain_proj: [B, contrastive_dim]
        """
        # ===== What Branch =====
        # Run Mamba in fp32 to avoid NaN (selective scan is unstable in fp16)
        what_x = self.what_proj(cls_token).unsqueeze(1)  # [B, 1, D]
        what_x = self.what_norm(what_x)
        with torch.amp.autocast('cuda', enabled=False):
            what_x_fp32 = what_x.float()
            for mamba in self.what_mamba:
                residual = mamba(what_x_fp32)
                # Clamp to prevent explosion
                residual = torch.clamp(residual, -10, 10)
                what_x_fp32 = what_x_fp32 + self.dropout(residual)
            what_x = what_x_fp32.to(what_x.dtype)
        what_feat = what_x.squeeze(1)  # [B, D]
        what_roi = self.what_head(what_feat)  # [B, num_rois]

        # ===== Where Branch =====
        where_x = self.where_proj(patch_tokens)  # [B, num_patches, D]
        where_x = where_x + self.where_pos_embed
        where_x = self.where_norm(where_x)
        with torch.amp.autocast('cuda', enabled=False):
            where_x_fp32 = where_x.float()
            for mamba in self.where_mamba:
                residual = mamba(where_x_fp32)
                residual = torch.clamp(residual, -10, 10)
                where_x_fp32 = where_x_fp32 + self.dropout(residual)
            where_x = where_x_fp32.to(where_x.dtype)
        where_pooled = self.where_pool(where_x.mean(dim=1))  # [B, D//2]
        where_roi = self.where_head(where_pooled)  # [B, num_rois]

        # ===== Fusion =====
        combined_feat = torch.cat([what_feat, where_pooled], dim=-1)
        gate_input = torch.cat([what_feat, F.pad(where_pooled, (0, self.hidden_dim - self.hidden_dim // 2))], dim=-1)
        gate_weights = self.fusion_gate(gate_input)  # [B, 2]

        pred_roi_means = (gate_weights[:, 0:1] * what_roi +
                         gate_weights[:, 1:2] * where_roi)

        # ===== Contrastive Alignment =====
        img_proj = self.img_contrastive_head(combined_feat)
        target_brain = gt_roi_means if gt_roi_means is not None else pred_roi_means
        brain_proj = self.brain_contrastive_head(target_brain)

        return pred_roi_means, img_proj, brain_proj

    def get_fusion_weights(self, cls_token, patch_tokens):
        """Get fusion weights for visualization."""
        what_x = self.what_proj(cls_token).unsqueeze(1)
        what_x = self.what_norm(what_x)
        with torch.amp.autocast('cuda', enabled=False):
            what_x_fp32 = what_x.float()
            for mamba in self.what_mamba:
                residual = torch.clamp(mamba(what_x_fp32), -10, 10)
                what_x_fp32 = what_x_fp32 + residual
            what_x = what_x_fp32.to(what_x.dtype)
        what_feat = what_x.squeeze(1)

        where_x = self.where_proj(patch_tokens)
        where_x = where_x + self.where_pos_embed
        where_x = self.where_norm(where_x)
        with torch.amp.autocast('cuda', enabled=False):
            where_x_fp32 = where_x.float()
            for mamba in self.where_mamba:
                residual = torch.clamp(mamba(where_x_fp32), -10, 10)
                where_x_fp32 = where_x_fp32 + residual
            where_x = where_x_fp32.to(where_x.dtype)
        where_pooled = self.where_pool(where_x.mean(dim=1))

        gate_input = torch.cat([what_feat, F.pad(where_pooled, (0, self.hidden_dim - self.hidden_dim // 2))], dim=-1)
        return self.fusion_gate(gate_input)


class MambaStage2(nn.Module):
    """
    Stage 2 V3: Mamba-VAE Fine Stage with Hybrid Global + Local.

    Key features:
    - VAE latent space for capturing fMRI variability
    - Global branch: learnable queries for holistic prediction
    - Local branch: ROI-wise residual prediction
    """
    def __init__(
        self,
        visual_dim=768 + 256,
        num_rois=524,
        fmri_dim=15724,
        voxels_per_cluster=30,
        embed_dim=384,
        latent_dim=128,
        depth=6,
        dropout=0.1,
        num_global_queries=128,
    ):
        super().__init__()

        self.num_rois = num_rois
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.fmri_dim = fmri_dim
        self.voxels_per_cluster = voxels_per_cluster
        self.num_global_queries = num_global_queries

        # 1. Tokenizers
        self.vis_tokenizer = nn.Linear(visual_dim, embed_dim)
        self.roi_tokenizer = nn.Linear(1, embed_dim)
        self.roi_pos_embed = nn.Parameter(torch.randn(1, num_rois, embed_dim) * 0.02)

        # 2. Mamba Encoder
        self.backbone_norm = nn.LayerNorm(embed_dim)
        self.backbone = nn.ModuleList([
            Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(depth)
        ])

        # 3. VAE Latent Space
        # Encode pooled features to μ and log(σ²)
        self.vae_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(embed_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim // 2, latent_dim)
        # Project latent back to embed_dim for injection
        self.latent_proj = nn.Linear(latent_dim, embed_dim)

        # 4. Global Branch
        self.global_queries = nn.Parameter(torch.randn(1, num_global_queries, embed_dim) * 0.02)
        self.global_mamba = nn.ModuleList([
            Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(2)
        ])
        self.global_head = nn.Sequential(
            nn.Linear(num_global_queries * embed_dim, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4096, fmri_dim)
        )

        # 5. Local Branch
        self.local_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, voxels_per_cluster)
        )

        self.remainder_voxels = fmri_dim - (num_rois * voxels_per_cluster)
        if self.remainder_voxels > 0:
            self.remainder_head = nn.Linear(embed_dim, self.remainder_voxels)

        # 6. Fusion & Scale
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.scale_layer = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.global_scale = nn.Parameter(torch.tensor(2.5))
        self.dropout = nn.Dropout(dropout)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, 1)"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, use mean (deterministic)
            return mu

    def forward(self, vis_feat, roi_means):
        """
        Returns:
            pred_voxels: [B, fmri_dim] predicted voxel activations
            kl_loss: scalar, KL divergence loss for VAE
        """
        B = vis_feat.shape[0]

        # Tokenize
        vis_tokens = self.vis_tokenizer(vis_feat).unsqueeze(1)
        roi_tokens = self.roi_tokenizer(roi_means.unsqueeze(-1))
        roi_tokens = roi_tokens + self.roi_pos_embed

        # Encode - Run Mamba in fp32 to avoid NaN
        x = torch.cat([vis_tokens, roi_tokens], dim=1)
        x = self.backbone_norm(x)
        with torch.amp.autocast('cuda', enabled=False):
            x_fp32 = x.float()
            for mamba in self.backbone:
                residual = torch.clamp(mamba(x_fp32), -10, 10)
                x_fp32 = x_fp32 + self.dropout(residual)
            memory = x_fp32.to(x.dtype)

        # VAE: compute latent from pooled memory
        pooled_memory = memory.mean(dim=1)  # [B, embed_dim]
        vae_hidden = self.vae_encoder(pooled_memory)
        mu = self.fc_mu(vae_hidden)
        logvar = self.fc_logvar(vae_hidden)
        z = self.reparameterize(mu, logvar)  # [B, latent_dim]

        # KL divergence loss: D_KL(q(z|x) || p(z)) where p(z) = N(0, 1)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Inject latent into memory
        z_proj = self.latent_proj(z).unsqueeze(1)  # [B, 1, embed_dim]
        memory = memory + z_proj  # Broadcast add to all tokens

        # Global Branch - also in fp32
        queries = self.global_queries.expand(B, -1, -1)
        global_input = torch.cat([memory, queries], dim=1)
        with torch.amp.autocast('cuda', enabled=False):
            global_input_fp32 = global_input.float()
            for mamba in self.global_mamba:
                residual = torch.clamp(mamba(global_input_fp32), -10, 10)
                global_input_fp32 = global_input_fp32 + self.dropout(residual)
            global_input = global_input_fp32.to(memory.dtype)
        global_output = global_input[:, -self.num_global_queries:, :]
        global_pred = self.global_head(global_output.flatten(1))

        # Local Branch
        roi_tokens_out = memory[:, 1:, :]
        deltas = self.local_head(roi_tokens_out)
        roi_means_expanded = roi_means.unsqueeze(-1).expand(-1, -1, self.voxels_per_cluster)
        local_voxels = roi_means_expanded + deltas
        local_pred = local_voxels.flatten(1)

        if self.remainder_voxels > 0:
            vis_token = memory[:, 0, :]
            remainder_pred = self.remainder_head(vis_token)
            local_pred = torch.cat([local_pred, remainder_pred], dim=1)

        # Fusion & Scale
        alpha = torch.sigmoid(self.fusion_weight)
        pred_voxels = alpha * global_pred + (1 - alpha) * local_pred
        scale_val = self.scale_layer(memory[:, 0, :]) * self.global_scale
        pred_voxels = pred_voxels * scale_val

        return pred_voxels, kl_loss


class HierarchicalARM_V3(nn.Module):
    """
    Hierarchical ARM V3: Dual-Stream Mamba Architecture with VAE

    Key features:
    - Stage 1: Dual-stream (Global + Local) with Mamba + Contrastive
    - Stage 2: Mamba-VAE with Hybrid Global + Local branches
    - voxels_per_cluster parameter for flexible ROI granularity
    """
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
    ):
        super().__init__()

        self.fmri_dim = fmri_dim
        self.voxels_per_cluster = voxels_per_cluster
        self.num_rois = fmri_dim // voxels_per_cluster

        self.stage1 = DualStreamMambaStage1(
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
        self.stage2 = MambaStage2(
            visual_dim=visual_dim_2,
            num_rois=self.num_rois,
            fmri_dim=fmri_dim,
            voxels_per_cluster=voxels_per_cluster,
            embed_dim=embed_dim_2,
            latent_dim=latent_dim,
            depth=depth_2,
            num_global_queries=num_global_queries,
            dropout=dropout,
        )

        self.patch_pooler = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim_1 // 2),
            nn.LayerNorm(hidden_dim_1 // 2),
            nn.GELU(),
        )

    def forward(self, cls_token, patch_tokens, stage=2, gt_roi_means=None):
        """
        Forward pass.

        Returns:
            If stage 1: (pred_roi_means, img_proj, brain_proj)
            If stage 2: (guidance, pred_voxels, kl_loss)
        """
        pred_roi_means, img_proj, brain_proj = self.stage1(
            cls_token, patch_tokens, gt_roi_means
        )

        if stage == 1:
            return pred_roi_means, img_proj, brain_proj

        guidance = gt_roi_means if gt_roi_means is not None else pred_roi_means
        pooled_patches = self.patch_pooler(patch_tokens.mean(dim=1))
        vis_feat = torch.cat([cls_token, pooled_patches], dim=-1)
        pred_voxels, kl_loss = self.stage2(vis_feat, guidance)

        return guidance, pred_voxels, kl_loss

    def get_stage1_weights(self, cls_token, patch_tokens):
        return self.stage1.get_fusion_weights(cls_token, patch_tokens)


HierarchicalARM = HierarchicalARM_V3


if __name__ == "__main__":
    # Test on CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")

    B = 4
    model = HierarchicalARM_V3(
        cls_dim=768,
        patch_dim=768,
        num_patches=256,
        fmri_dim=15724,
        voxels_per_cluster=30,
        latent_dim=128,
    ).to(device)

    cls_token = torch.randn(B, 768).to(device)
    patch_tokens = torch.randn(B, 256, 768).to(device)
    gt_roi_means = torch.randn(B, 15724 // 30).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Num ROIs: {model.num_rois}")
    print(f"Latent dim: {model.stage2.latent_dim}")

    # Test Stage 1
    pred_roi, img_proj, brain_proj = model(cls_token, patch_tokens, stage=1, gt_roi_means=gt_roi_means)
    print(f"Stage 1: pred_roi={pred_roi.shape}, img_proj={img_proj.shape}, brain_proj={brain_proj.shape}")

    # Test Stage 2 (with VAE)
    model.train()  # Enable sampling
    guidance, pred_voxels, kl_loss = model(cls_token, patch_tokens, stage=2)
    print(f"Stage 2 (train): guidance={guidance.shape}, pred_voxels={pred_voxels.shape}, kl_loss={kl_loss.item():.4f}")

    model.eval()  # Deterministic (use mean)
    with torch.no_grad():
        guidance, pred_voxels, kl_loss = model(cls_token, patch_tokens, stage=2)
    print(f"Stage 2 (eval): guidance={guidance.shape}, pred_voxels={pred_voxels.shape}, kl_loss={kl_loss.item():.4f}")

    weights = model.get_stage1_weights(cls_token, patch_tokens)
    print(f"Fusion weights mean: global={weights[:, 0].mean():.3f}, local={weights[:, 1].mean():.3f}")

    print("✓ All tests passed!")

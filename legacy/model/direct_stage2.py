"""One-stage DINOv2-to-voxel baseline using the CHASMBrain Stage 2 decoder.

The baseline removes both uses of coarse cluster values:
1. ROI inputs are learned tokens rather than tokenized cluster predictions.
2. The local branch predicts absolute voxel values rather than cluster + residual.
"""

import torch
import torch.nn as nn

from model.hierarchical_v3_1 import MambaStage2
from model.hierarchical_v3_mlp import MLPStage2
from model.hierarchical_v3_transformer import TransformerStage2

STAGE2_BACKBONES = {
    "mamba": MambaStage2,
    "mlp": MLPStage2,
    "transformer": TransformerStage2,
}


class DirectStage2Only(nn.Module):
    """Predict voxels directly from DINOv2 CLS and mean-pooled patch features."""

    def __init__(
        self,
        cls_dim=768,
        patch_dim=768,
        fmri_dim=15724,
        voxels_per_cluster=30,
        hidden_dim=512,
        embed_dim=384,
        latent_dim=128,
        depth=6,
        num_global_queries=128,
        dropout=0.1,
        vae_mode="stochastic",
        backbone="mamba",
    ):
        super().__init__()
        self.fmri_dim = fmri_dim
        self.num_rois = fmri_dim // voxels_per_cluster

        self.patch_pooler = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        self.decoder = STAGE2_BACKBONES[backbone](
            visual_dim=cls_dim + hidden_dim // 2,
            num_rois=self.num_rois,
            fmri_dim=fmri_dim,
            voxels_per_cluster=voxels_per_cluster,
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            depth=depth,
            dropout=dropout,
            num_global_queries=num_global_queries,
            vae_mode=vae_mode,
        )
        # Cluster values are absent. roi_pos_embed itself serves as the learned
        # ROI-token bank; remove the unused scalar-to-token projection.
        self.decoder.roi_tokenizer = nn.Identity()

    def forward(self, cls_token, patch_tokens):
        pooled_patches = self.patch_pooler(patch_tokens.mean(dim=1))
        visual_features = torch.cat([cls_token, pooled_patches], dim=-1)
        return self._decode_without_clusters(visual_features)

    def _decode_without_clusters(self, visual_features):
        decoder = self.decoder
        batch_size = visual_features.shape[0]

        visual_token = decoder.vis_tokenizer(visual_features).unsqueeze(1)
        roi_tokens = decoder.roi_pos_embed.expand(batch_size, -1, -1)
        x = decoder.backbone_norm(torch.cat([visual_token, roi_tokens], dim=1))

        with torch.amp.autocast("cuda", enabled=False):
            memory_fp32 = x.float()
            for mamba in decoder.backbone:
                residual = torch.clamp(mamba(memory_fp32), -10, 10)
                memory_fp32 = memory_fp32 + decoder.dropout(residual)
        memory = memory_fp32.to(x.dtype)

        kl_loss = memory.new_zeros(())
        if decoder.vae_mode != "none":
            latent_hidden = decoder.vae_encoder(memory.mean(dim=1))
            mu = decoder.fc_mu(latent_hidden)
            if decoder.vae_mode == "deterministic":
                latent = mu
            else:
                # Keep variance math in fp32 and bound exp(logvar). Without
                # cluster conditioning, the early latent distribution can be
                # poorly scaled and overflow under fp16 autocast.
                mu_fp32 = mu.float()
                logvar = decoder.fc_logvar(latent_hidden).float().clamp(-10.0, 10.0)
                if decoder.training:
                    std = torch.exp(0.5 * logvar)
                    latent = mu_fp32 + std * torch.randn_like(std)
                else:
                    latent = mu_fp32
                kl_loss = -0.5 * torch.mean(
                    1 + logvar - mu_fp32.square() - logvar.exp()
                )
            memory = memory + decoder.latent_proj(latent.to(memory.dtype)).unsqueeze(1)

        queries = decoder.global_queries.expand(batch_size, -1, -1)
        global_input = torch.cat([memory, queries], dim=1)
        with torch.amp.autocast("cuda", enabled=False):
            global_fp32 = global_input.float()
            for mamba in decoder.global_mamba:
                residual = torch.clamp(mamba(global_fp32), -10, 10)
                global_fp32 = global_fp32 + decoder.dropout(residual)
        global_output = global_fp32.to(memory.dtype)[:, -decoder.num_global_queries:, :]
        global_prediction = decoder.global_head(global_output.flatten(1))

        # Absolute local prediction: no cluster-value residual shortcut.
        local_prediction = decoder.local_head(memory[:, 1:, :]).flatten(1)
        if decoder.remainder_voxels > 0:
            remainder = decoder.remainder_head(memory[:, 0, :])
            local_prediction = torch.cat([local_prediction, remainder], dim=1)

        alpha = torch.sigmoid(decoder.fusion_weight)
        prediction = alpha * global_prediction + (1 - alpha) * local_prediction
        prediction = (
            prediction
            * decoder.scale_layer(memory[:, 0, :])
            * decoder.global_scale
        )
        return prediction, kl_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectStage2Only(
        cls_dim=16,
        patch_dim=16,
        fmri_dim=23,
        voxels_per_cluster=5,
        hidden_dim=16,
        embed_dim=16,
        latent_dim=4,
        depth=1,
        num_global_queries=2,
        dropout=0.0,
    ).to(device)
    prediction, kl_loss = model(
        torch.randn(2, 16, device=device),
        torch.randn(2, 4, 16, device=device),
    )
    assert prediction.shape == (2, 23)
    assert kl_loss.ndim == 0
    print("DirectStage2Only smoke test passed")

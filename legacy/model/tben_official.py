"""TBEn readout controls built on the authors' own transformer code.

The decoder here is `third_party/transformer_brain_encoder/transformer.py`,
vendored verbatim from https://github.com/Hosseinadeli/transformer_brain_encoder
(Adeli, Sun & Kriegeskorte 2025). Only the parts of `models/brain_encoder.py`
that assume the Algonauts fsaverage layout are reimplemented, because our
protocol predicts a single volumetric NSD mask instead of left/right hemisphere
surface vertices. Each correspondence is noted below.

Their readout, for an ROI-resolution run (`--readout_res rois_all`):

    output_tokens = transformer(src, mask, query_embed.weight, pos, ...)[-1]
    lh_f_pred = self.lh_embed(output_tokens[:, :n//2, :])      # shared Linear
    lh_f_pred = torch.movedim(lh_f_pred, 1, -1)                # [B, V, K]
    lh_f_pred = sum(roi_mask * lh_f_pred, dim=2)               # pick each
                                                              # voxel's own ROI

Two adaptations, both exact:

1. One volumetric mask instead of two hemispheres, so there is one shared
   readout Linear rather than `lh_embed` and `rh_embed`.
2. Their masked sum materialises [B, V, K]. With our K=476 clusters and
   V=14278 voxels that tensor is ~868 MB per batch and the matmul behind it is
   ~166 TFLOP, which is why they only ever run it with a handful of ROIs. Our
   cluster mask is one-hot (voxel v belongs to exactly one cluster), so
   `sum_k mask[k,v] * pred[b,v,k]` collapses to `pred[b, v, cluster(v)]`, and
   we compute only that entry. `verify_masked_sum_equivalence` below asserts
   this against a literal transcription of their expression.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))

from third_party.transformer_brain_encoder.position_encoding import build_position_encoding
from third_party.transformer_brain_encoder.transformer import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
)

from .hierarchical_v3_1 import ContrastiveHead, HierarchicalARM_V3, MambaStage2


class TBEnDecoder(nn.Module):
    """The authors' transformer driven by precomputed DINOv2 patch tokens.

    Their backbone emits a [B, C, H, W] feature map; DINOv2 ViT-B/14 at 224px
    gives 256 patch tokens, which reshape to a 16x16 map exactly, so their
    `Transformer.forward` and sine position encoding run unmodified.
    """

    def __init__(
        self,
        patch_dim=768,
        num_patches=256,
        num_queries=476,
        hidden_dim=768,
        nheads=16,
        dim_feedforward=1024,
        enc_layers=0,
        dec_layers=1,
        dropout=0.1,
        pre_norm=True,
        enc_output_layer=1,
        position_embedding="sine",
    ):
        super().__init__()
        grid = int(round(num_patches ** 0.5))
        if grid * grid != num_patches:
            raise ValueError(f"num_patches must be square to form a feature map, got {num_patches}")
        self.grid = grid
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # Their pipeline projects backbone channels to hidden_dim only for
        # ResNet; for DINOv2 they set hidden_dim=768 so the tokens pass
        # straight through. We keep that, and only add a projection if the
        # dimensions actually differ.
        self.input_proj = (
            nn.Identity() if patch_dim == hidden_dim
            else nn.Conv2d(patch_dim, hidden_dim, kernel_size=1)
        )
        self.position_embedding = build_position_encoding(position_embedding, hidden_dim // 2)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_enc=True,
            return_intermediate_dec=True,
            enc_output_layer=enc_output_layer,
        )

    def forward(self, patch_tokens):
        batch = patch_tokens.shape[0]
        feature_map = patch_tokens.transpose(1, 2).reshape(
            batch, -1, self.grid, self.grid
        )
        feature_map = self.input_proj(feature_map)
        # No padding: our feature maps are fixed size, so the mask is all-False
        # exactly as it would be for uniformly sized images.
        mask = torch.zeros(
            (batch, self.grid, self.grid), dtype=torch.bool, device=patch_tokens.device
        )
        pos_embed = self.position_embedding(feature_map, mask).to(feature_map.dtype)

        hs = self.transformer(
            feature_map, mask, self.query_embed.weight, pos_embed, False
        )
        return hs[-1]


class TBEnOneStageOfficial(nn.Module):
    """Variant A: their readout, one stage, no coarse-to-fine refinement.

    Returns (prediction, kl_loss) so it drops into the one-stage trainer; the
    KL term is always zero because TBEn has no latent path.
    """

    def __init__(
        self,
        cls_dim=768,
        patch_dim=768,
        num_patches=256,
        fmri_dim=14278,
        voxels_per_cluster=30,
        hidden_dim=768,
        nheads=16,
        dim_feedforward=1024,
        enc_layers=0,
        dec_layers=1,
        dropout=0.1,
        pre_norm=True,
        enc_output_layer=1,
    ):
        super().__init__()
        self.fmri_dim = fmri_dim
        self.voxels_per_cluster = voxels_per_cluster
        self.full_clusters = fmri_dim // voxels_per_cluster
        self.remainder_voxels = fmri_dim - self.full_clusters * voxels_per_cluster
        # One query per cluster, plus one for the leftover voxels so that every
        # voxel is owned by exactly one query, as their ROI masks require.
        self.num_queries = self.full_clusters + (1 if self.remainder_voxels > 0 else 0)

        self.decoder = TBEnDecoder(
            patch_dim=patch_dim, num_patches=num_patches, num_queries=self.num_queries,
            hidden_dim=hidden_dim, nheads=nheads, dim_feedforward=dim_feedforward,
            enc_layers=enc_layers, dec_layers=dec_layers, dropout=dropout,
            pre_norm=pre_norm, enc_output_layer=enc_output_layer,
        )
        # Their `lh_embed`: one Linear shared by every query, mapping a query
        # token to the whole voxel vector.
        self.voxel_embed = nn.Linear(hidden_dim, fmri_dim)

    def forward(self, cls_token, patch_tokens):
        tokens = self.decoder(patch_tokens)
        return self._masked_readout(tokens), tokens.new_zeros(())

    def _masked_readout(self, tokens):
        """Each voxel takes the prediction of the query that owns it."""
        weight = self.voxel_embed.weight
        bias = self.voxel_embed.bias

        cluster_voxels = self.full_clusters * self.voxels_per_cluster
        grouped_weight = weight[:cluster_voxels].view(
            self.full_clusters, self.voxels_per_cluster, -1
        )
        prediction = torch.einsum(
            "bkd,kvd->bkv", tokens[:, :self.full_clusters, :], grouped_weight
        ).flatten(1)
        prediction = prediction + bias[:cluster_voxels]

        if self.remainder_voxels > 0:
            remainder = (
                tokens[:, -1, :] @ weight[cluster_voxels:].T + bias[cluster_voxels:]
            )
            prediction = torch.cat([prediction, remainder], dim=1)
        return prediction

    @torch.no_grad()
    def verify_masked_sum_equivalence(self, tokens, tolerance=1e-4):
        """Check our gather against a literal copy of the authors' expression.

        Theirs (engine.py):
            f_pred = movedim(embed(output_tokens), 1, -1)     # [B, V, K]
            rois_b = tile(rois[:, :, None], (1, 1, B)).permute(2, 1, 0)
            f_pred = sum(mul(rois_b, f_pred), dim=2)
        """
        voxel_to_query = torch.arange(self.fmri_dim, device=tokens.device)
        voxel_to_query = torch.clamp(
            voxel_to_query // self.voxels_per_cluster, max=self.num_queries - 1
        )
        rois = torch.zeros(
            (self.num_queries, self.fmri_dim), device=tokens.device, dtype=tokens.dtype
        )
        rois[voxel_to_query, torch.arange(self.fmri_dim, device=tokens.device)] = 1.0

        reference = torch.movedim(self.voxel_embed(tokens), 1, -1)
        rois_b = torch.tile(rois[:, :, None], (1, 1, tokens.shape[0])).permute(2, 1, 0)
        reference = torch.sum(torch.mul(rois_b, reference), dim=2)

        ours = self._masked_readout(tokens)
        deviation = (ours - reference).abs().max().item()
        if deviation > tolerance:
            raise AssertionError(f"gather deviates from the masked sum by {deviation}")
        return deviation


class TBEnStage1Official(nn.Module):
    """Variant B Stage 1: the same decoder, emitting one value per cluster.

    Their `readout_res='voxels'` mode reads one scalar per query by taking the
    diagonal of a shared Linear's output. We reuse that verbatim at cluster
    resolution. The signature matches DualStreamMambaStage1 so the existing
    Stage 1 losses and Stage 2 run unchanged.
    """

    def __init__(
        self,
        cls_dim=768,
        patch_dim=768,
        num_patches=256,
        fmri_dim=14278,
        voxels_per_cluster=30,
        hidden_dim=768,
        nheads=16,
        dim_feedforward=1024,
        enc_layers=0,
        dec_layers=1,
        dropout=0.1,
        pre_norm=True,
        enc_output_layer=1,
        contrastive_dim=256,
    ):
        super().__init__()
        self.fmri_dim = fmri_dim
        self.voxels_per_cluster = voxels_per_cluster
        # Stage 2 conditions on one value per full cluster, so the query count
        # here follows num_rois rather than adding a remainder query.
        self.num_rois = fmri_dim // voxels_per_cluster

        self.decoder = TBEnDecoder(
            patch_dim=patch_dim, num_patches=num_patches, num_queries=self.num_rois,
            hidden_dim=hidden_dim, nheads=nheads, dim_feedforward=dim_feedforward,
            enc_layers=enc_layers, dec_layers=dec_layers, dropout=dropout,
            pre_norm=pre_norm, enc_output_layer=enc_output_layer,
        )
        self.cluster_embed = nn.Linear(hidden_dim, self.num_rois)
        self.img_contrastive_head = ContrastiveHead(hidden_dim, contrastive_dim)
        self.brain_contrastive_head = ContrastiveHead(self.num_rois, contrastive_dim)

    def forward(self, cls_token, patch_tokens, gt_roi_means=None):
        tokens = self.decoder(patch_tokens)
        pred_roi_means = torch.diagonal(self.cluster_embed(tokens), dim1=-2, dim2=-1)

        img_proj = self.img_contrastive_head(tokens.mean(dim=1))
        target_brain = gt_roi_means if gt_roi_means is not None else pred_roi_means
        brain_proj = self.brain_contrastive_head(target_brain)
        return pred_roi_means, img_proj, brain_proj


class TBEnStage2(nn.Module):
    """A second TBEn pass that refines Stage 1's clusters into voxels.

    This is our Stage 2 mechanism with the backbone swapped for TBEn's, so the
    two ways Stage 2 uses coarse values are preserved exactly as in
    `MambaStage2`:

      1. Conditioning. `MambaStage2` tokenizes the cluster scalars with
         `nn.Linear(1, embed_dim)` and adds them to `roi_pos_embed` before the
         backbone runs. Here the learned query embedding plays the role of
         `roi_pos_embed`, and the tokenized cluster value is added to it, so
         the cluster value is visible to cross-attention rather than only to
         the output head.
      2. Residual output. `MambaStage2` predicts `roi_means + delta`; this does
         the same, so the second stage only has to model the deviation from the
         coarse prediction.

    The attention itself is the authors' `TransformerDecoderLayer`, used
    verbatim. Only query preparation is ours, because their `Transformer`
    wrapper builds queries from a fixed `[K, D]` table and cannot accept the
    per-sample cluster conditioning this stage needs. Their default
    `enc_layers=0` means no encoder is involved either way.
    """

    def __init__(
        self,
        patch_dim=768,
        num_patches=256,
        fmri_dim=14278,
        voxels_per_cluster=30,
        hidden_dim=768,
        nheads=16,
        dim_feedforward=1024,
        dec_layers=1,
        dropout=0.1,
        pre_norm=True,
    ):
        super().__init__()
        self.fmri_dim = fmri_dim
        self.voxels_per_cluster = voxels_per_cluster
        self.num_rois = fmri_dim // voxels_per_cluster
        self.remainder_voxels = fmri_dim - self.num_rois * voxels_per_cluster
        self.num_queries = self.num_rois + (1 if self.remainder_voxels > 0 else 0)

        grid = int(round(num_patches ** 0.5))
        self.grid = grid
        self.hidden_dim = hidden_dim

        self.input_proj = (
            nn.Identity() if patch_dim == hidden_dim
            else nn.Conv2d(patch_dim, hidden_dim, kernel_size=1)
        )
        self.position_embedding = build_position_encoding("sine", hidden_dim // 2)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.cluster_tokenizer = nn.Linear(1, hidden_dim)

        layer = TransformerDecoderLayer(
            hidden_dim, nheads, dim_feedforward, dropout, "relu", pre_norm
        )
        self.decoder = TransformerDecoder(
            layer, dec_layers, nn.LayerNorm(hidden_dim), return_intermediate=False
        )
        self.voxel_embed = nn.Linear(hidden_dim, fmri_dim)
        if self.remainder_voxels > 0:
            self.remainder_head = nn.Linear(hidden_dim, self.remainder_voxels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, patch_tokens, roi_means):
        batch = patch_tokens.shape[0]
        feature_map = self.input_proj(
            patch_tokens.transpose(1, 2).reshape(batch, -1, self.grid, self.grid)
        )
        mask = torch.zeros(
            (batch, self.grid, self.grid), dtype=torch.bool, device=patch_tokens.device
        )
        pos = self.position_embedding(feature_map, mask).to(feature_map.dtype)

        memory = feature_map.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)

        # Queries carry the coarse prediction into the attention, mirroring
        # roi_tokenizer(roi_means) + roi_pos_embed in MambaStage2.
        cluster_values = roi_means
        if self.num_queries > self.num_rois:
            padding = roi_means.new_zeros((batch, self.num_queries - self.num_rois))
            cluster_values = torch.cat([roi_means, padding], dim=1)
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch, 1)
        query_pos = query_pos + self.cluster_tokenizer(
            cluster_values.unsqueeze(-1)
        ).permute(1, 0, 2)

        tgt = torch.zeros_like(query_pos)
        tokens = self.decoder(
            tgt, memory, memory_key_padding_mask=mask.flatten(1),
            pos=pos, query_pos=query_pos,
        )[-1].transpose(0, 1)

        # Residual on top of the coarse values, as in MambaStage2.
        weight, bias = self.voxel_embed.weight, self.voxel_embed.bias
        cluster_voxels = self.num_rois * self.voxels_per_cluster
        grouped = weight[:cluster_voxels].view(
            self.num_rois, self.voxels_per_cluster, -1
        )
        delta = torch.einsum("bkd,kvd->bkv", tokens[:, :self.num_rois, :], grouped)
        delta = delta + bias[:cluster_voxels].view(self.num_rois, self.voxels_per_cluster)
        prediction = (roi_means.unsqueeze(-1) + delta).flatten(1)

        if self.remainder_voxels > 0:
            prediction = torch.cat(
                [prediction, self.remainder_head(tokens[:, -1, :])], dim=1
            )
        return prediction, prediction.new_zeros(())


class HierarchicalARM_TBEn_Both(nn.Module):
    """Two-stage coarse-to-fine with TBEn as the backbone in both stages."""

    def __init__(
        self,
        cls_dim=768,
        patch_dim=768,
        num_patches=256,
        fmri_dim=14278,
        voxels_per_cluster=30,
        hidden_dim_1=512,
        contrastive_dim=256,
        dropout=0.1,
        tben_hidden_dim=768,
        tben_nheads=16,
        tben_dim_feedforward=1024,
        tben_enc_layers=0,
        tben_dec_layers=1,
        tben_pre_norm=True,
        tben_enc_output_layer=1,
        **unused,
    ):
        super().__init__()
        self.fmri_dim = fmri_dim
        self.voxels_per_cluster = voxels_per_cluster
        self.num_rois = fmri_dim // voxels_per_cluster

        self.stage1 = TBEnStage1Official(
            cls_dim=cls_dim, patch_dim=patch_dim, num_patches=num_patches,
            fmri_dim=fmri_dim, voxels_per_cluster=voxels_per_cluster,
            hidden_dim=tben_hidden_dim, nheads=tben_nheads,
            dim_feedforward=tben_dim_feedforward, enc_layers=tben_enc_layers,
            dec_layers=tben_dec_layers, dropout=dropout, pre_norm=tben_pre_norm,
            enc_output_layer=tben_enc_output_layer, contrastive_dim=contrastive_dim,
        )
        self.stage2 = TBEnStage2(
            patch_dim=patch_dim, num_patches=num_patches, fmri_dim=fmri_dim,
            voxels_per_cluster=voxels_per_cluster, hidden_dim=tben_hidden_dim,
            nheads=tben_nheads, dim_feedforward=tben_dim_feedforward,
            dec_layers=tben_dec_layers, dropout=dropout, pre_norm=tben_pre_norm,
        )
        # Unused by this architecture, but the Stage 2 optimizer in the training
        # script collects its parameters alongside stage2's.
        self.patch_pooler = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim_1 // 2),
            nn.LayerNorm(hidden_dim_1 // 2),
            nn.GELU(),
        )

    def forward(self, cls_token, patch_tokens, stage=2, gt_roi_means=None):
        pred_roi_means, img_proj, brain_proj = self.stage1(
            cls_token, patch_tokens, gt_roi_means
        )
        if stage == 1:
            return pred_roi_means, img_proj, brain_proj
        guidance = gt_roi_means if gt_roi_means is not None else pred_roi_means
        pred_voxels, kl_loss = self.stage2(patch_tokens, guidance)
        return guidance, pred_voxels, kl_loss


class HierarchicalARM_V3_TBEn(HierarchicalARM_V3):
    """Variant B: their readout as Stage 1, our Mamba Stage 2 untouched."""

    def __init__(
        self,
        cls_dim=768,
        patch_dim=768,
        num_patches=256,
        fmri_dim=14278,
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
        tben_hidden_dim=768,
        tben_nheads=16,
        tben_dim_feedforward=1024,
        tben_enc_layers=0,
        tben_dec_layers=1,
        tben_pre_norm=True,
        tben_enc_output_layer=1,
    ):
        nn.Module.__init__(self)
        self.fmri_dim = fmri_dim
        self.voxels_per_cluster = voxels_per_cluster
        self.num_rois = fmri_dim // voxels_per_cluster

        self.stage1 = TBEnStage1Official(
            cls_dim=cls_dim, patch_dim=patch_dim, num_patches=num_patches,
            fmri_dim=fmri_dim, voxels_per_cluster=voxels_per_cluster,
            hidden_dim=tben_hidden_dim, nheads=tben_nheads,
            dim_feedforward=tben_dim_feedforward, enc_layers=tben_enc_layers,
            dec_layers=tben_dec_layers, dropout=dropout, pre_norm=tben_pre_norm,
            enc_output_layer=tben_enc_output_layer, contrastive_dim=contrastive_dim,
        )

        visual_dim_2 = cls_dim + hidden_dim_1 // 2
        self.stage2 = MambaStage2(
            visual_dim=visual_dim_2, num_rois=self.num_rois, fmri_dim=fmri_dim,
            voxels_per_cluster=voxels_per_cluster, embed_dim=embed_dim_2,
            latent_dim=latent_dim, depth=depth_2,
            num_global_queries=num_global_queries, dropout=dropout, vae_mode=vae_mode,
        )

        self.patch_pooler = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim_1 // 2),
            nn.LayerNorm(hidden_dim_1 // 2),
            nn.GELU(),
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, fmri_dim = 2, 14278
    cls_token = torch.randn(batch, 768, device=device)
    patch_tokens = torch.randn(batch, 256, 768, device=device)

    one_stage = TBEnOneStageOfficial(fmri_dim=fmri_dim).to(device).eval()
    prediction, kl_loss = one_stage(cls_token, patch_tokens)
    assert prediction.shape == (batch, fmri_dim)
    assert kl_loss.item() == 0.0
    deviation = one_stage.verify_masked_sum_equivalence(
        one_stage.decoder(patch_tokens)
    )
    print(f"gather vs authors' masked sum: max abs deviation {deviation:.2e}")
    print(f"TBEnOneStageOfficial parameters: "
          f"{sum(p.numel() for p in one_stage.parameters()):,} "
          f"(queries {one_stage.num_queries})")

    two_stage = HierarchicalARM_V3_TBEn(fmri_dim=fmri_dim).to(device)
    pred_roi, img_proj, brain_proj = two_stage(cls_token, patch_tokens, stage=1)
    assert pred_roi.shape == (batch, fmri_dim // 30)
    _, pred_voxels, _ = two_stage(cls_token, patch_tokens, stage=2)
    assert pred_voxels.shape == (batch, fmri_dim)
    print(f"HierarchicalARM_V3_TBEn parameters: "
          f"{sum(p.numel() for p in two_stage.parameters()):,} "
          f"(Stage 1 {sum(p.numel() for p in two_stage.stage1.parameters()):,})")
    print("TBEn official-port smoke test passed")

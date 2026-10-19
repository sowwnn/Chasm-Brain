"""
HierarchicalARM V3 Multi-Subject: Shared backbone + per-subject output heads.
Exp 2 (rebuttal): train on subj01+02, adapt+test on subj05.

Shared: Stage1 backbone (what/where branches, fusion, img_contrastive_head)
        Stage2 backbone (tokenizers, Mamba, VAE, global_queries, local_head)
Per-subject: Stage1 heads (what_head, where_head, brain_contrastive_head)
             Stage2 heads (roi_pos_embed, global_head, remainder_head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class ContrastiveHead(nn.Module):
    def __init__(self, input_dim, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────
# Shared Stage 1 backbone
# ─────────────────────────────────────────

class SharedStage1(nn.Module):
    """What + Where branches, fusion gate, img_contrastive_head. No output heads."""

    def __init__(self, cls_dim=768, patch_dim=768, num_patches=256,
                 hidden_dim=512, depth_what=2, depth_where=4,
                 contrastive_dim=256, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.what_proj = nn.Linear(cls_dim, hidden_dim)
        self.what_norm = nn.LayerNorm(hidden_dim)
        self.what_mamba = nn.ModuleList([
            Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(depth_what)
        ])

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

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )
        self.img_contrastive_head = ContrastiveHead(hidden_dim + hidden_dim // 2, contrastive_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cls_token, patch_tokens, branch_mode: str = "both"):
        """Returns (what_feat, where_pooled, gate_weights, img_proj).

        branch_mode:
          "both"       : normal (paper)            — what←CLS, where←patches
          "swap"       : swap inputs               — what←mean(patches), where←broadcast(cls)
          "what_only"  : disable where branch      — gate forced [1, 0], where_pooled=0
          "where_only" : disable what branch       — gate forced [0, 1], what_feat=0
        Architecture/params unchanged across modes (cls_dim must equal patch_dim for swap).
        """
        if branch_mode == "swap":
            num_patches = patch_tokens.shape[1]
            cls_input = patch_tokens.mean(dim=1)
            patches_input = cls_token.unsqueeze(1).expand(-1, num_patches, -1).contiguous()
        else:
            cls_input = cls_token
            patches_input = patch_tokens

        # What branch
        what_x = self.what_norm(self.what_proj(cls_input).unsqueeze(1))
        with torch.amp.autocast('cuda', enabled=False):
            x = what_x.float()
            for m in self.what_mamba:
                x = x + self.dropout(torch.clamp(m(x), -10, 10))
            what_x = x.to(what_x.dtype)
        what_feat = what_x.squeeze(1)  # [B, D]

        # Where branch
        where_x = self.where_norm(self.where_proj(patches_input) + self.where_pos_embed)
        with torch.amp.autocast('cuda', enabled=False):
            x = where_x.float()
            for m in self.where_mamba:
                x = x + self.dropout(torch.clamp(m(x), -10, 10))
            where_x = x.to(where_x.dtype)
        where_pooled = self.where_pool(where_x.mean(dim=1))  # [B, D//2]

        # Disable branches if requested (zero out features; force gate one-hot)
        if branch_mode == "what_only":
            where_pooled = torch.zeros_like(where_pooled)
        elif branch_mode == "where_only":
            what_feat = torch.zeros_like(what_feat)

        # Fusion
        pad = F.pad(where_pooled, (0, self.hidden_dim - self.hidden_dim // 2))
        gate_weights = self.fusion_gate(torch.cat([what_feat, pad], dim=-1))  # [B, 2]
        if branch_mode == "what_only":
            gate_weights = torch.zeros_like(gate_weights); gate_weights[:, 0] = 1.0
        elif branch_mode == "where_only":
            gate_weights = torch.zeros_like(gate_weights); gate_weights[:, 1] = 1.0

        img_proj = self.img_contrastive_head(torch.cat([what_feat, where_pooled], dim=-1))

        return what_feat, where_pooled, gate_weights, img_proj


# ─────────────────────────────────────────
# Shared Stage 2 backbone
# ─────────────────────────────────────────

class SharedStage2(nn.Module):
    """Stage 2 backbone without roi_pos_embed, global_head, remainder_head."""

    def __init__(self, visual_dim, embed_dim=384, latent_dim=128,
                 depth=6, num_global_queries=128, voxels_per_cluster=30, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_global_queries = num_global_queries
        self.voxels_per_cluster = voxels_per_cluster

        self.vis_tokenizer = nn.Linear(visual_dim, embed_dim)
        self.roi_tokenizer = nn.Linear(1, embed_dim)

        self.backbone_norm = nn.LayerNorm(embed_dim)
        self.backbone = nn.ModuleList([
            Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(depth)
        ])

        self.vae_encoder = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.GELU())
        self.fc_mu = nn.Linear(embed_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim // 2, latent_dim)
        self.latent_proj = nn.Linear(latent_dim, embed_dim)

        self.global_queries = nn.Parameter(torch.randn(1, num_global_queries, embed_dim) * 0.02)
        self.global_mamba = nn.ModuleList([
            Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(2)
        ])

        # local_head is fully shared (outputs voxels_per_cluster, same for all subjects)
        self.local_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, voxels_per_cluster),
        )

        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.scale_layer = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, 1), nn.Sigmoid()
        )
        self.global_scale = nn.Parameter(torch.tensor(2.5))
        self.dropout = nn.Dropout(dropout)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def forward(self, vis_feat, roi_means, subj_head):
        """
        subj_head: SubjectHead instance (provides roi_pos_embed, global_head, remainder_head)
        Returns (pred_voxels, kl_loss)
        """
        B = vis_feat.shape[0]
        num_rois = roi_means.shape[1]

        vis_tokens = self.vis_tokenizer(vis_feat).unsqueeze(1)
        roi_tokens = self.roi_tokenizer(roi_means.unsqueeze(-1)) + subj_head.s2_roi_pos_embed
        x = self.backbone_norm(torch.cat([vis_tokens, roi_tokens], dim=1))

        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            for m in self.backbone:
                x = x + self.dropout(torch.clamp(m(x), -10, 10))
        memory = x.to(vis_feat.dtype)

        pooled = memory.mean(dim=1)
        vae_h = self.vae_encoder(pooled)
        mu, logvar = self.fc_mu(vae_h), self.fc_logvar(vae_h)
        z = self.reparameterize(mu, logvar)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        memory = memory + self.latent_proj(z).unsqueeze(1)

        # Global branch
        queries = self.global_queries.expand(B, -1, -1)
        g = torch.cat([memory, queries], dim=1)
        with torch.amp.autocast('cuda', enabled=False):
            g = g.float()
            for m in self.global_mamba:
                g = g + self.dropout(torch.clamp(m(g), -10, 10))
        global_output = g.to(memory.dtype)[:, -self.num_global_queries:, :]
        global_pred = subj_head.s2_global_head(global_output.flatten(1))  # [B, fmri_dim]

        # Local branch
        roi_out = memory[:, 1:, :]  # [B, num_rois, embed_dim]
        deltas = self.local_head(roi_out)  # [B, num_rois, voxels_per_cluster]
        local_pred = (roi_means.unsqueeze(-1) + deltas).flatten(1)

        if subj_head.remainder_voxels > 0:
            remainder = subj_head.s2_remainder_head(memory[:, 0, :])
            local_pred = torch.cat([local_pred, remainder], dim=1)

        alpha = torch.sigmoid(self.fusion_weight)
        pred = alpha * global_pred + (1 - alpha) * local_pred
        pred = pred * self.scale_layer(memory[:, 0, :]) * self.global_scale

        return pred, kl_loss


# ─────────────────────────────────────────
# Per-subject head
# ─────────────────────────────────────────

class SubjectHead(nn.Module):
    """All subject-specific parameters for Stage 1 + Stage 2."""

    def __init__(self, fmri_dim, voxels_per_cluster=30, hidden_dim_1=512,
                 contrastive_dim=256, embed_dim_2=384, num_global_queries=128, dropout=0.1):
        super().__init__()
        num_rois = fmri_dim // voxels_per_cluster
        self.num_rois = num_rois
        self.fmri_dim = fmri_dim
        self.remainder_voxels = fmri_dim % voxels_per_cluster

        # Stage 1
        self.s1_what_head = nn.Sequential(
            nn.LayerNorm(hidden_dim_1),
            nn.Linear(hidden_dim_1, num_rois),
        )
        self.s1_where_head = nn.Linear(hidden_dim_1 // 2, num_rois)
        self.s1_brain_contrastive = ContrastiveHead(num_rois, contrastive_dim)

        # Stage 2
        self.s2_roi_pos_embed = nn.Parameter(torch.randn(1, num_rois, embed_dim_2) * 0.02)
        self.s2_global_head = nn.Sequential(
            nn.Linear(num_global_queries * embed_dim_2, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4096, fmri_dim),
        )
        if self.remainder_voxels > 0:
            self.s2_remainder_head = nn.Linear(embed_dim_2, self.remainder_voxels)
        else:
            self.s2_remainder_head = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def stage1_forward(self, what_feat, where_pooled, gate_weights, gt_roi_means=None):
        """Returns (pred_roi_means, brain_proj)."""
        what_roi = self.s1_what_head(what_feat)
        where_roi = self.s1_where_head(where_pooled)
        pred_roi = gate_weights[:, 0:1] * what_roi + gate_weights[:, 1:2] * where_roi
        target = gt_roi_means if gt_roi_means is not None else pred_roi
        brain_proj = self.s1_brain_contrastive(target)
        return pred_roi, brain_proj


# ─────────────────────────────────────────
# Full Multi-Subject Model
# ─────────────────────────────────────────

class HierarchicalARM_MultiSubject(nn.Module):
    """
    Shared backbone with per-subject heads.
    subject_fmri_dims: dict mapping subject_id (int) -> fmri_dim
    """

    def __init__(
        self,
        subject_fmri_dims: dict,  # {1: 15724, 2: 14278, ...}
        cls_dim=768, patch_dim=768, num_patches=256,
        voxels_per_cluster=30,
        hidden_dim_1=512, depth_what=2, depth_where=4,
        contrastive_dim=256,
        embed_dim_2=384, latent_dim=128, depth_2=6,
        num_global_queries=128,
        dropout=0.4,
        branch_mode: str = "both",
    ):
        super().__init__()
        self.voxels_per_cluster = voxels_per_cluster
        valid_modes = {"both", "swap", "what_only", "where_only"}
        assert branch_mode in valid_modes, f"branch_mode must be one of {valid_modes}"
        self.branch_mode = branch_mode
        if branch_mode == "swap":
            assert cls_dim == patch_dim, "swap requires cls_dim == patch_dim"
        if branch_mode != "both":
            print(f"  [BRANCH_MODE={branch_mode}]")

        self.shared_s1 = SharedStage1(
            cls_dim=cls_dim, patch_dim=patch_dim, num_patches=num_patches,
            hidden_dim=hidden_dim_1, depth_what=depth_what, depth_where=depth_where,
            contrastive_dim=contrastive_dim, dropout=dropout,
        )

        visual_dim_2 = cls_dim + hidden_dim_1 // 2
        self.shared_s2 = SharedStage2(
            visual_dim=visual_dim_2, embed_dim=embed_dim_2, latent_dim=latent_dim,
            depth=depth_2, num_global_queries=num_global_queries,
            voxels_per_cluster=voxels_per_cluster, dropout=dropout,
        )

        self.patch_pooler = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim_1 // 2),
            nn.LayerNorm(hidden_dim_1 // 2),
            nn.GELU(),
        )

        self.subject_heads = nn.ModuleDict({
            str(sid): SubjectHead(
                fmri_dim=fmri_dim,
                voxels_per_cluster=voxels_per_cluster,
                hidden_dim_1=hidden_dim_1,
                contrastive_dim=contrastive_dim,
                embed_dim_2=embed_dim_2,
                num_global_queries=num_global_queries,
                dropout=dropout,
            )
            for sid, fmri_dim in subject_fmri_dims.items()
        })

        for sid, fmri_dim in subject_fmri_dims.items():
            num_rois = fmri_dim // voxels_per_cluster
            print(f"  Subject {sid}: {fmri_dim} voxels → {num_rois} ROIs "
                  f"(remainder={fmri_dim % voxels_per_cluster})")

    def get_head(self, subject_id: int) -> SubjectHead:
        return self.subject_heads[str(subject_id)]

    def forward(self, cls_token, patch_tokens, subject_id: int, stage=2, gt_roi_means=None):
        """
        Returns:
            stage 1: (pred_roi_means, img_proj, brain_proj)
            stage 2: (pred_roi_means, pred_voxels, kl_loss)
        """
        head = self.get_head(subject_id)
        what_feat, where_pooled, gate_weights, img_proj = self.shared_s1(
            cls_token, patch_tokens, branch_mode=self.branch_mode
        )
        pred_roi_means, brain_proj = head.stage1_forward(what_feat, where_pooled, gate_weights, gt_roi_means)

        if stage == 1:
            return pred_roi_means, img_proj, brain_proj

        guidance = gt_roi_means if gt_roi_means is not None else pred_roi_means
        pooled_patches = self.patch_pooler(patch_tokens.mean(dim=1))
        vis_feat = torch.cat([cls_token, pooled_patches], dim=-1)
        pred_voxels, kl_loss = self.shared_s2(vis_feat, guidance, head)

        return pred_roi_means, pred_voxels, kl_loss

    def freeze_backbone(self):
        """Freeze all shared parameters. Only subject heads remain trainable."""
        for p in self.shared_s1.parameters():
            p.requires_grad = False
        for p in self.shared_s2.parameters():
            p.requires_grad = False
        for p in self.patch_pooler.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.shared_s1.parameters():
            p.requires_grad = True
        for p in self.shared_s2.parameters():
            p.requires_grad = True
        for p in self.patch_pooler.parameters():
            p.requires_grad = True

    def add_subject(self, subject_id: int, fmri_dim: int, device=None):
        """Add a new subject head (e.g., for adaptation to subj05)."""
        sid = str(subject_id)
        if sid in self.subject_heads:
            print(f"Subject {subject_id} already exists.")
            return
        cfg = self.shared_s1  # borrow config values
        head = SubjectHead(
            fmri_dim=fmri_dim,
            voxels_per_cluster=self.voxels_per_cluster,
            hidden_dim_1=cfg.hidden_dim,
            contrastive_dim=cfg.img_contrastive_head.net[-1].normalized_shape[0],
            embed_dim_2=self.shared_s2.embed_dim,
            num_global_queries=self.shared_s2.num_global_queries,
        )
        if device is not None:
            head = head.to(device)
        self.subject_heads[sid] = head
        num_rois = fmri_dim // self.voxels_per_cluster
        print(f"Added subject {subject_id}: {fmri_dim} voxels → {num_rois} ROIs")

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

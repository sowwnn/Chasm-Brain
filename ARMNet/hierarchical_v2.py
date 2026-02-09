
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Try to import Mamba, if fails, use Transformer fallback
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

class ContrastiveHead(nn.Module):
    """
    Projector for Contrastive Learning (CLIP-style).
    Maps features to a shared latent space for loss calculation.
    """
    def __init__(self, input_dim, embed_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class MambaStage1(nn.Module):
    """
    Stage 1 V3: Mamba-based Selective Coarse Stage.
    Image Features -> ROI Means.
    Sử dụng Mamba để chọn lọc đặc trưng quan trọng từ ảnh nhằm ánh xạ chính xác sang ROI.
    """
    def __init__(self, visual_dim=768, num_clusters=518, hidden_dim=768, depth=4, dropout=0.1, contrastive_dim=256):
        super().__init__()
        self.num_clusters = num_clusters
        
        # Tokenizer: Biến image features thành chuỗi các tokens (vùng đặc trưng)
        # Ở đây ta giả lập một chuỗi bằng cách chiếu visual_dim sang hidden_dim và reshape
        # hoặc dùng image feature như một token duy nhất (nhưng Mamba tốt hơn với sequence).
        # Cách tiếp cận hay nhất: Project sang hidden_dim và dùng 1-step Mamba hoặc biến thành chuỗi nhỏ.
        
        self.input_proj = nn.Linear(visual_dim, hidden_dim)
        
        if HAS_MAMBA:
            print("Using Mamba for Stage 1 Selection mechanism.")
            self.backbone = nn.ModuleList([
                Mamba(
                    d_model=hidden_dim,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                ) for _ in range(depth)
            ])
            self.is_mamba = True
        else:
            print("Mamba not found, fallback to Transformer for Stage 1.")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            )
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=depth)
            self.is_mamba = False
            
        self.dropout = nn.Dropout(dropout)
        self.norm_final = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_clusters)
        
        # Contrastive Path
        self.img_contrastive_head = ContrastiveHead(visual_dim, contrastive_dim)
        self.brain_contrastive_head = ContrastiveHead(num_clusters, contrastive_dim)
        
    def forward(self, vis_feat, gt_roi_means=None):
        """
        Args:
            vis_feat: [B, D_vis]
        Returns:
            pred_roi_means: [B, N_roi]
            img_proj: [B, D_cont]
            brain_proj: [B, D_cont]
        """
        # 1. Prediction Path
        x = self.input_proj(vis_feat).unsqueeze(1) # [B, 1, D] - xử lý như sequence length=1
        
        if self.is_mamba:
            for layer in self.backbone:
                x = layer(x)
        else:
            x = self.backbone(x)
            
        x = self.dropout(x.squeeze(1)) # [B, D]
        x = self.norm_final(x)
        pred_roi_means = self.head(x)
        
        # 2. Contrastive Alignment
        img_proj = self.img_contrastive_head(vis_feat)
        target_brain = gt_roi_means if gt_roi_means is not None else pred_roi_means
        brain_proj = self.brain_contrastive_head(target_brain)
        
        return pred_roi_means, img_proj, brain_proj

class TransformerStage2(nn.Module):
    """
    Stage 2 V2: Hybrid Global + Local Transformer/Mamba-based Fine Stage.
    Image + ROI Means -> Voxels.

    Combines:
    - Global Branch: Attention pooling with learnable queries for long-range dependencies
    - Local Branch: ROI-specific predictions to preserve hierarchical structure
    - Learnable fusion weight to balance both branches
    """
    def __init__(self, visual_dim=768, num_clusters=518, fmri_dim=15724,
                 embed_dim=384, depth=4, num_heads=8, dropout=0.1, use_mamba=False,
                 num_global_queries=128):
        super().__init__()

        self.num_clusters = num_clusters
        self.embed_dim = embed_dim
        self.fmri_dim = fmri_dim
        self.num_global_queries = num_global_queries

        # 1. Tokenizers
        self.img_tokenizer = nn.Linear(visual_dim, embed_dim)
        self.roi_tokenizer = nn.Linear(1, embed_dim)

        # Positional Embeddings for ROIs
        self.roi_pos_embed = nn.Parameter(torch.randn(1, num_clusters, embed_dim) * 0.02)

        # 2. Encoder Backbone
        if use_mamba and HAS_MAMBA:
            print("Initializing Stage 2 with Mamba Backbone.")
            self.backbone = nn.ModuleList([
                Mamba(
                    d_model=embed_dim,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                ) for _ in range(depth)
            ])
            self.is_mamba = True
        else:
            print("Initializing Stage 2 with Transformer Backbone.")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim*4,
                dropout=dropout,
                batch_first=True
            )
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=depth)
            self.is_mamba = False

        # 3. Global Branch: Learnable queries + Cross-Attention Decoder
        self.global_queries = nn.Parameter(torch.randn(1, num_global_queries, embed_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.global_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Global head: [B, num_queries, D] -> [B, fmri_dim]
        self.global_head = nn.Sequential(
            nn.Linear(num_global_queries * embed_dim, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4096, fmri_dim)
        )

        # 4. Local Branch: ROI-specific predictions
        # Each ROI token predicts its corresponding voxels
        voxels_per_roi = fmri_dim // num_clusters  # ~30 voxels per ROI
        self.local_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, voxels_per_roi)
        )

        # Handle remainder voxels
        self.remainder_voxels = fmri_dim - (num_clusters * voxels_per_roi)
        if self.remainder_voxels > 0:
            self.remainder_head = nn.Linear(embed_dim, self.remainder_voxels)

        # 5. Learnable fusion weight (initialized at 0.5)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        # 6. Amplitude Scale Recovery
        # Dự đoán hệ số scale để điều chỉnh biên độ, giúp giảm MSE
        self.scale_layer = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.global_scale = nn.Parameter(torch.tensor(2.5))

    def encode(self, vis_feat, roi_means):
        """Shared encoder for both branches"""
        B = vis_feat.shape[0]

        # Tokenize Image
        img_tokens = self.img_tokenizer(vis_feat).unsqueeze(1)  # [B, 1, D]

        # Tokenize ROIs
        roi_tokens = self.roi_tokenizer(roi_means.unsqueeze(-1))  # [B, N, D]
        roi_tokens = roi_tokens + self.roi_pos_embed  # Add position

        # Concat: [Image, ROIs] -> [B, N+1, D]
        x = torch.cat([img_tokens, roi_tokens], dim=1)

        # Backbone encoding
        if self.is_mamba:
            for layer in self.backbone:
                x = layer(x)
        else:
            x = self.backbone(x)

        return x

    def forward(self, vis_feat, roi_means):
        """
        Args:
            vis_feat: [B, D_vis]
            roi_means: [B, N_roi] (Guidance from Stage 1)
        Returns:
            pred_voxels: [B, fmri_dim]
        """
        B = vis_feat.shape[0]

        # Shared encoding
        memory = self.encode(vis_feat, roi_means)  # [B, N+1, D]

        # ===== Global Branch =====
        queries = self.global_queries.expand(B, -1, -1)
        global_output = self.global_decoder(queries, memory)
        global_pred = self.global_head(global_output.flatten(1))

        # ===== Local Branch (Residual Learning) =====
        # Mỗi token ROI dự đoán delta (sai lệch) so với ROI Mean của nó
        roi_tokens = memory[:, 1:, :]  # [B, N, D]
        deltas = self.local_head(roi_tokens)  # [B, N, voxels_per_roi]
        
        # Expand ROI Means: [B, N] -> [B, N, voxels_per_roi]
        # Giả định phân bổ voxel đều cho các ROI (đây là base line)
        roi_means_expanded = roi_means.unsqueeze(-1).expand(-1, -1, deltas.shape[-1])
        local_voxels = roi_means_expanded + deltas # Residual formula
        local_pred = local_voxels.flatten(1)

        # Xử lý voxel dư thừa nếu có
        if self.remainder_voxels > 0:
            img_token = memory[:, 0, :]
            remainder_pred = self.remainder_head(img_token)
            local_pred = torch.cat([local_pred, remainder_pred], dim=1)

        # ===== Fusion & Scale Recovery =====
        alpha = torch.sigmoid(self.fusion_weight)
        pred_voxels = alpha * global_pred + (1 - alpha) * local_pred
        
        # Scale Recovery dùng đặc trưng ảnh để điều chỉnh biên độ
        scale_val = self.scale_layer(memory[:, 0, :]) * self.global_scale
        pred_voxels = pred_voxels * scale_val

        return pred_voxels

class HierarchicalARM_V2(nn.Module):
    """
    Hierarchical ARM V2
    - Stage 1: Contrastive + Regression
    - Stage 2: Hybrid Transformer/Mamba (Global + Local)
    """
    def __init__(self, visual_dim=768, num_clusters=518, fmri_dim=15724,
                 hidden_dim_1=1024, # Stage 1 Params
                 embed_dim_2=384, depth_2=4, num_heads_2=8, # Stage 2 Params
                 num_global_queries=128,
                 contrastive_dim=256,
                 use_mamba=False):
        super().__init__()

        self.stage1 = MambaStage1(
            visual_dim=visual_dim,
            num_clusters=num_clusters,
            hidden_dim=hidden_dim_1,
            contrastive_dim=contrastive_dim
        )

        self.stage2 = TransformerStage2(
            visual_dim=visual_dim,
            num_clusters=num_clusters,
            fmri_dim=fmri_dim,
            embed_dim=embed_dim_2,
            depth=depth_2,
            num_heads=num_heads_2,
            num_global_queries=num_global_queries,
            use_mamba=use_mamba
        )
        
    def forward(self, vis_feat, stage=2, gt_roi_means=None):
        """
        Forward pass.
        If stage 1: returns (pred_roi_means, img_proj, brain_proj)
        If stage 2: returns (guidance, pred_voxels)
        """
        # Stage 1
        pred_roi_means, img_proj, brain_proj = self.stage1(vis_feat, gt_roi_means)
        
        if stage == 1:
            return pred_roi_means, img_proj, brain_proj
            
        # Stage 2 requires ROI means as guidance
        guidance = gt_roi_means if gt_roi_means is not None else pred_roi_means
        pred_voxels = self.stage2(vis_feat, guidance)
        
        return guidance, pred_voxels



import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleResBlock(nn.Module):
    """
    Standard Residual Block without Time Conditioning.
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        h = F.gelu(h)
        h = self.linear1(h)
        h = self.dropout(h)
        
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.linear2(h)
        h = self.dropout(h)
        
        return x + h

class CoarseStage(nn.Module):
    """
    Stage 1: Image Features -> Sub-ROI Means (Coarse Voxel Map)
    Predicts mean activity for each sub-ROI cluster.
    """
    def __init__(self, visual_dim=768, num_clusters=500, hidden_dim=1024, num_layers=4, dropout=0.2):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.GELU()
        )
        
        self.layers = nn.ModuleList([
            SimpleResBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_clusters)
        )
        
    def forward(self, vis_feat):
        h = self.projector(vis_feat)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)

class FineStage(nn.Module):
    """
    Stage 2: Image Features + Coarse Map -> Fine Voxel Map
    Predicts full voxel activity using image features and coarse guidance.
    """
    def __init__(self, visual_dim=768, num_clusters=500, fmri_dim=15724, hidden_dim=2048, num_layers=4, dropout=0.2):
        super().__init__()
        
        # Project Visual
        self.vis_proj = nn.Linear(visual_dim, hidden_dim // 2)
        
        # Project Coarse Map (Input is ROI means)
        # We assume the input to this stage is the vector of ROI means.
        # Ideally, we should "scatter" this to voxels, but for MLP, simple concat is fine.
        # However, concatenating 500 dims to 768 dims is small.
        # Maybe we should project it.
        self.roi_proj = nn.Linear(num_clusters, hidden_dim // 2)
        
        # Combine
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.layers = nn.ModuleList([
            SimpleResBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, fmri_dim)
        )
        
    def forward(self, vis_feat, roi_means):
        v = self.vis_proj(vis_feat) # [B, H/2]
        r = self.roi_proj(roi_means) # [B, H/2]
        
        h = torch.cat([v, r], dim=-1) # [B, H]
        h = self.combiner(h)
        
        for layer in self.layers:
            h = layer(h)
            
        return self.head(h)

class HierarchicalARM(nn.Module):
    """
    Two-Stage Hierarchical Model.
    Stage 1: Coarse Prediction (Sub-ROI Means)
    Stage 2: Fine Prediction (Voxels) conditioned on Stage 1.
    """
    def __init__(self, visual_dim=768, num_clusters=518, fmri_dim=15724, 
                 hidden_dim_1=1024, layers_1=4,
                 hidden_dim_2=2048, layers_2=6,
                 dropout=0.2):
        super().__init__()
        
        self.stage1 = CoarseStage(visual_dim, num_clusters, hidden_dim_1, layers_1, dropout)
        self.stage2 = FineStage(visual_dim, num_clusters, fmri_dim, hidden_dim_2, layers_2, dropout)
        
    def forward(self, vis_feat, stage=2, gt_roi_means=None):
        """
        Forward pass.
        If stage=1: Returns (pred_roi_means, None)
        If stage=2: Returns (pred_roi_means, pred_voxels)
        
        Args:
            vis_feat: Visual embeddings
            stage: 1 or 2
            gt_roi_means: Optional ground truth ROI means for Teacher Forcing in Stage 2.
                          If None, uses Stage 1 predictions.
        """
        # Always run stage 1 (or we could skip if we have GT)
        pred_roi_means = self.stage1(vis_feat)
        
        if stage == 1:
            return pred_roi_means, None
            
        # Stage 2
        # Use GT if provided (Teacher Forcing), else use pred
        # User said "frozen model step 1 use result".
        # But for training stage 2 efficiently, maybe mixed is good?
        # Let's support both.
        
        guidance = gt_roi_means if gt_roi_means is not None else pred_roi_means
        
        # Detach guidance if it comes from stage 1 to verify "frozen" concept
        # But if we want e2e finetuning later, keep gradients.
        # User said "frozen", so normally we detach.
        # But if gt_roi_means is passed, gradients don't flow to stage 1 anyway.
        # If pred_roi_means is passed, we might want gradients or not.
        # I'll modify the training loop to handle freezing/detaching.
        # Here I just pass tensors.
        
        pred_voxels = self.stage2(vis_feat, guidance)
        
        return pred_roi_means, pred_voxels

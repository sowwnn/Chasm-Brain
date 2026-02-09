
import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelRefiner(nn.Module):
    """
    Shared MLP that processes each voxel independently.
    Uses FiLM to inject global visual context.
    """
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # FiLM: Global visual context projects to scale/shift for all voxels
        self.film_proj = nn.Linear(512, hidden_dim * 2)

    def forward(self, x, global_cond):
        """
        x: [B, N, in_dim] where N=15724
        global_cond: [B, 512]
        """
        # FiLM parameters
        scale, shift = self.film_proj(global_cond).chunk(2, dim=-1) # [B, hidden_dim]
        scale = scale.unsqueeze(1) # [B, 1, hidden_dim]
        shift = shift.unsqueeze(1) 

        # Block 1
        h = self.norm1(x)
        h = self.linear1(h)
        h = h * (1 + scale) + shift # Modulate all voxels
        h = F.gelu(h)
        h = self.dropout(h)
        
        # Block 2
        identity = h
        h = self.norm2(h)
        h = self.linear2(h)
        h = F.gelu(h + identity)
        
        return h

class SpatialDenoisingARM(nn.Module):
    """
    Spatial-Aware 1D Denoising Model.
    Processes 15k voxels in parallel using shared weights and spatial features.
    
    Advantages:
    - Speed: 1D operations are much faster than 3D Conv on empty space.
    - Precision: Every voxel knows its (X, Y, Z) and ROI membership.
    - Consistency: Learnable scale and Tanh head for stable amplitude.
    """
    def __init__(self, visual_dim=768, spatial_dim=15, hidden_dim=256, num_layers=4, dropout=0.2):
        """
        Args:
            spatial_dim: Number of spatial features (3 coords + N ROI maps)
            hidden_dim: Dimension of hidden representations for EACH voxel
        """
        super().__init__()
        
        # 1. Global Visual Projector
        self.vis_projector = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )
        
        # 2. Voxel Input Projector
        # Input features per voxel: [Noised_Val, Mean_Val, Coord_X, Coord_Y, Coord_Z, ROI_G1, ROI_G2...]
        # in_dim = 1 (noised) + 1 (mean) + spatial_dim
        self.voxel_input_proj = nn.Linear(2 + spatial_dim, hidden_dim)
        
        # 3. Refinement Layers (Shared Weights)
        self.layers = nn.ModuleList([
            VoxelRefiner(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # 4. Final Head
        self.final_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh() # Constrain delta
        )
        
        self.output_scale = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x_noised, visual_feat, mean_fmri, spatial_features):
        """
        x_noised: [B, N]
        visual_feat: [B, visual_dim]
        mean_fmri: [B, N]
        spatial_features: [N, spatial_dim] (Constant spatial maps)
        """
        B, N = x_noised.shape
        
        # 1. Global conditioning
        cond = self.vis_projector(visual_feat) # [B, 512]
        
        # 2. Prepare Point-wise features
        # Expand spatial features to batch: [N, D] -> [B, N, D]
        spatial_b = spatial_features.unsqueeze(0).expand(B, -1, -1)
        
        # Combine everything: [B, N, 2 + D]
        x_input = torch.stack([x_noised, mean_fmri], dim=-1) # [B, N, 2]
        x_full = torch.cat([x_input, spatial_b], dim=-1)
        
        # 3. Initial projection
        h = self.voxel_input_proj(x_full) # [B, N, hidden_dim]
        
        # 4. Refine through layers
        for layer in self.layers:
            h = layer(h, cond)
            
        # 5. Predict Delta
        delta = self.final_head(h).squeeze(-1) # [B, N]
        
        # 6. Final Signal = Mean + Delta * Scale
        return mean_fmri + delta * self.output_scale

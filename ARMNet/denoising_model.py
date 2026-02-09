import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion_model import ResBlock, CrossAttentionBlock

class DenoisingARM(nn.Module):
    """
    Unified Conditional Denoising Model (Denoising Refiner).
    Maps (mean_fMRI + noise) conditioned by visual features to clear_fMRI.

    Supports any fMRI dimension (1000 parcels, 15724 voxels, etc.)
    """
    def __init__(self, visual_dim=768, fmri_dim=15724, hidden_dim=1024, num_layers=4, num_heads=8, dropout=0.2,
                 output_clamp=None):
        super().__init__()
        self.fmri_dim = fmri_dim
        self.output_clamp = output_clamp

        # 1. Global Condition Projector (DINOv2 + potential ROI metadata)
        self.vis_projector = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # 2. Main Input Projection (Global Connectivity)
        self.input_proj = nn.Linear(fmri_dim, hidden_dim)

        # 3. Global Feature Modulation Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'res_block': ResBlock(hidden_dim, hidden_dim, dropout=dropout),
                'cross_attn': CrossAttentionBlock(hidden_dim, hidden_dim, num_heads, dropout)
            }))

        # 4. Refinement head (Predicts DELTA)
        self.refinement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, fmri_dim),
            nn.Tanh() # Stability barrier
        )

        # Learnable scaling factor (Initialize small to let Mean dominate initially)
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x_noised, visual_feat, mean_fmri):
        """
        x_noised: [B, fmri_dim] (Pure Noise or Noisy Baseline)
        visual_feat: [B, visual_dim]
        mean_fmri: [B, fmri_dim]
        """
        # 1. Project global condition
        cond = self.vis_projector(visual_feat)
        
        # 2. Project full brain noise pattern
        h = self.input_proj(x_noised)
        
        # 3. Process through global layers
        for layer in self.layers:
            h = layer['res_block'](h, cond)
            h = layer['cross_attn'](h, cond)
            
        # 4. Predict Delta
        delta = self.refinement_head(h)

        # 5. Output = mean_fmri + delta * learnable_scale
        output = mean_fmri + delta * self.output_scale

        if self.output_clamp is not None:
            output = torch.clamp(output, self.output_clamp[0], self.output_clamp[1])

        return output



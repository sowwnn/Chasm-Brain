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
    def __init__(self, visual_dim=768, fmri_dim=15724, hidden_dim=1024, num_layers=4, num_heads=8, dropout=0.2):
        super().__init__()
        self.fmri_dim = fmri_dim
        
        # Project visual features
        self.vis_projector = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Initial projection of (mean_fMRI + noise)
        self.input_proj = nn.Linear(fmri_dim, hidden_dim)
        
        # Feature modulation layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'res_block': ResBlock(hidden_dim, hidden_dim, dropout=dropout),
                'cross_attn': CrossAttentionBlock(hidden_dim, hidden_dim, num_heads, dropout)
            }))
            
        # Refinement head (predicts DELTA to be added to mean_fMRI)
        self.refinement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, fmri_dim)
        )
        
        # Learnable scaling factor to help the model reach higher amplitudes
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x_noised, visual_feat, mean_fmri):
        """
        Args:
            x_noised: [B, fmri_dim] (mean_fMRI + gaussian noise)
            visual_feat: [B, visual_dim] (DINOv2 embeddings)
            mean_fmri: [B, fmri_dim] (The base mean signal)
        """
        # 1. Project condition
        cond = self.vis_projector(visual_feat) # [B, hidden_dim]
        
        # 2. Initial projection of input
        h = self.input_proj(x_noised) # [B, hidden_dim]
        
        # 3. Process through layers
        # Using a dummy 't' for ResBlock compatibility (constant 0)
        t_dummy = torch.zeros(x_noised.size(0), device=x_noised.device)
        
        for layer in self.layers:
            # FiLM modulated ResBlock (using visual cond instead of time)
            h = layer['res_block'](h, cond) 
            # Cross-Attention to let visual features guide the peaks
            h = layer['cross_attn'](h, cond)
            
        # 4. Predict Delta
        delta = self.refinement_head(h)
        
        # 5. Output = mean_fMRI + predicted delta
        # Using mean_fmri as the anchor ensures we stay in the right brain space
        return mean_fmri + delta * self.output_scale



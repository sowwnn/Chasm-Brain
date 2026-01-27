"""
Dual-Stream CNN Model for fMRI Generation using ConvNeXt Blocks

Architecture:
1. 3D Stream (Mask-Aware):
   - Processes 3D volume (61x46x42) with mask conditioning
   - Uses ConvNeXt 3D blocks
   - Only learns brain regions inside mask

2. 1D Stream:
   - Processes 15k voxel vector directly
   - Uses ConvNeXt 1D blocks
   - Learns on raw voxel sequence

3. Fusion Module:
   - Combines features from both streams
   - Produces final fMRI prediction

Key Features:
- Pure CNN, no flow matching
- ConvNeXt block design (depth-wise conv, layer norm, GELU, inverted bottleneck)
- Mask-aware 3D processing
- Direct 1D voxel modeling
- Dual-stream fusion for robust predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


# ============================================================================
# ConvNeXt Building Blocks
# ============================================================================

class LayerNorm3d(nn.Module):
    """Layer normalization for 3D tensors (channel-first)"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # x: [B, C, D, H, W]
        # Normalize over C dimension
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class ConvNeXt3DBlock(nn.Module):
    """
    ConvNeXt block for 3D data.

    Architecture:
    1. Depthwise conv 7x7x7 (spatial mixing)
    2. LayerNorm
    3. Pointwise conv 1x1x1 to 4*C (expansion)
    4. GELU
    5. Pointwise conv 1x1x1 to C (projection)
    6. Residual connection
    """
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()

        # Depthwise conv (spatial mixing)
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # Layer norm
        self.norm = LayerNorm3d(dim)

        # Pointwise convs (channel mixing - inverted bottleneck)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)

        # Layer scale
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """
        x: [B, C, D, H, W]
        """
        residual = x

        # Depthwise conv
        x = self.dwconv(x)

        # Norm
        x = self.norm(x)

        # Inverted bottleneck
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Layer scale
        if self.gamma is not None:
            x = self.gamma[:, None, None, None] * x

        # Residual + drop path
        x = residual + self.drop_path(x)

        return x


class ConvNeXt1DBlock(nn.Module):
    """
    ConvNeXt block for 1D sequence data.

    Similar to 3D version but operates on 1D sequences.
    """
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()

        # Depthwise conv (temporal mixing)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # Layer norm
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Pointwise (channel mixing - inverted bottleneck)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Layer scale
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """
        x: [B, C, N] where N is sequence length
        """
        residual = x

        # Depthwise conv
        x = self.dwconv(x)

        # Permute for LayerNorm: [B, C, N] -> [B, N, C]
        x = x.permute(0, 2, 1)

        # Norm
        x = self.norm(x)

        # Inverted bottleneck
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Layer scale
        if self.gamma is not None:
            x = self.gamma * x

        # Permute back: [B, N, C] -> [B, C, N]
        x = x.permute(0, 2, 1)

        # Residual
        x = residual + self.drop_path(x)

        return x


# ============================================================================
# 3D Stream (Mask-Aware)
# ============================================================================

class MaskAwareConv3D(nn.Module):
    """
    Mask-aware convolution that focuses learning on brain regions.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, mask=None):
        """
        x: [B, C, D, H, W]
        mask: [D, H, W] or [B, 1, D, H, W]
        """
        x = self.conv(x)

        # Apply mask to zero out background regions
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
            if mask.shape[0] == 1:
                mask = mask.expand(x.shape[0], 1, -1, -1, -1)
            if mask.shape[1] == 1:
                mask = mask.expand(-1, x.shape[1], -1, -1, -1)

            # Resize mask if spatial dims don't match
            if mask.shape[2:] != x.shape[2:]:
                mask = F.interpolate(
                    mask.float(),
                    size=x.shape[2:],
                    mode='nearest'
                )

            x = x * mask.float()

        return x


class Stream3D(nn.Module):
    """
    3D Stream with ConvNeXt blocks and mask conditioning.

    Processes 3D volume with mask to focus on brain regions.
    """
    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        depths=[2, 2, 4, 2],  # Number of blocks at each stage
        drop_path_rate=0.1,
    ):
        super().__init__()

        self.depths = depths
        self.num_stages = len(depths)

        # Stem: initial projection
        self.stem = MaskAwareConv3D(in_channels, base_channels, kernel_size=4, stride=2, padding=1)

        # Build stages
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur_channels = base_channels
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        cur_idx = 0
        for i in range(self.num_stages):
            # Stage: stack of ConvNeXt blocks
            stage_blocks = nn.Sequential(*[
                ConvNeXt3DBlock(
                    dim=cur_channels,
                    drop_path=dp_rates[cur_idx + j]
                )
                for j in range(depths[i])
            ])
            self.stages.append(stage_blocks)
            cur_idx += depths[i]

            # Downsampling between stages (except last)
            if i < self.num_stages - 1:
                next_channels = cur_channels * 2
                downsample = nn.Sequential(
                    LayerNorm3d(cur_channels),
                    nn.Conv3d(cur_channels, next_channels, kernel_size=2, stride=2)
                )
                self.downsamples.append(downsample)
                cur_channels = next_channels
            else:
                self.downsamples.append(nn.Identity())

        # Final norm
        self.norm = LayerNorm3d(cur_channels)
        self.out_channels = cur_channels

    def forward(self, x, mask=None):
        """
        x: [B, 1, D, H, W] - 3D fMRI volume
        mask: [D, H, W] - brain mask

        Returns:
            features: [B, C, D', H', W'] - 3D features
        """
        # Stem
        x = self.stem(x, mask)

        # Stages with downsampling
        for stage, downsample in zip(self.stages, self.downsamples):
            x = stage(x)
            x = downsample(x)

            # Apply mask at each stage
            if mask is not None:
                if mask.dim() == 3:
                    mask_expanded = mask.unsqueeze(0).unsqueeze(0)
                else:
                    mask_expanded = mask

                # Resize mask to current resolution
                if mask_expanded.shape[2:] != x.shape[2:]:
                    mask_resized = F.interpolate(
                        mask_expanded.float(),
                        size=x.shape[2:],
                        mode='nearest'
                    )
                    x = x * mask_resized.expand_as(x)

        # Final norm
        x = self.norm(x)

        return x


# ============================================================================
# 1D Stream
# ============================================================================

class Stream1D(nn.Module):
    """
    1D Stream with ConvNeXt blocks.

    Processes 1D voxel sequence (15k voxels) directly.
    """
    def __init__(
        self,
        n_voxels=15724,
        base_channels=128,
        depths=[2, 2, 4, 2],
        drop_path_rate=0.1,
    ):
        super().__init__()

        self.depths = depths
        self.num_stages = len(depths)

        # Stem: project voxels to feature space
        self.stem = nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3)

        # Build stages
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur_channels = base_channels
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        cur_idx = 0
        for i in range(self.num_stages):
            # Stage: stack of ConvNeXt blocks
            stage_blocks = nn.Sequential(*[
                ConvNeXt1DBlock(
                    dim=cur_channels,
                    drop_path=dp_rates[cur_idx + j]
                )
                for j in range(depths[i])
            ])
            self.stages.append(stage_blocks)
            cur_idx += depths[i]

            # Downsampling between stages (except last)
            if i < self.num_stages - 1:
                next_channels = cur_channels * 2
                downsample = nn.Sequential(
                    nn.LayerNorm(cur_channels),
                    # Permute wrapper for LayerNorm
                    nn.Conv1d(cur_channels, next_channels, kernel_size=2, stride=2)
                )
                self.downsamples.append(downsample)
                cur_channels = next_channels
            else:
                self.downsamples.append(nn.Identity())

        # Final norm
        self.norm = nn.LayerNorm(cur_channels)
        self.out_channels = cur_channels

    def forward(self, x):
        """
        x: [B, N] - 1D fMRI voxel sequence (N = 15724)

        Returns:
            features: [B, C, N'] - 1D features
        """
        # Add channel dimension: [B, N] -> [B, 1, N]
        x = x.unsqueeze(1)

        # Stem
        x = self.stem(x)  # [B, C, N/2]

        # Stages with downsampling
        for stage, downsample in zip(self.stages, self.downsamples):
            x = stage(x)

            # Apply downsample (need to handle LayerNorm)
            if isinstance(downsample, nn.Sequential):
                # Permute for LayerNorm, apply conv
                if isinstance(downsample[0], nn.LayerNorm):
                    x_perm = x.permute(0, 2, 1)  # [B, C, N] -> [B, N, C]
                    x_perm = downsample[0](x_perm)
                    x = x_perm.permute(0, 2, 1)  # [B, N, C] -> [B, C, N]
                    x = downsample[1](x)  # Conv1d
                else:
                    x = downsample(x)
            else:
                x = downsample(x)

        # Final norm: [B, C, N] -> [B, N, C] -> norm -> [B, C, N]
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)

        return x


# ============================================================================
# Fusion Module
# ============================================================================

class FusionModule(nn.Module):
    """
    Fuses features from 3D and 1D streams.

    Strategy:
    1. Project 3D features to 1D space (using mask to extract brain voxels)
    2. Concatenate with 1D features
    3. Apply cross-attention or MLP fusion
    4. Output final fMRI prediction
    """
    def __init__(
        self,
        channels_3d,
        channels_1d,
        n_voxels=15724,
        fusion_type='attention',  # 'attention' or 'mlp'
        fusion_dim: int = 128,
    ):
        super().__init__()

        self.fusion_type = fusion_type
        self.n_voxels = n_voxels

        # Project to reduced common (fusion) dimension to limit memory
        common_dim = fusion_dim
        self.proj_3d = nn.Linear(channels_3d, common_dim)
        self.proj_1d = nn.Linear(channels_1d, common_dim)

        if fusion_type == 'attention':
            # Cross-attention
            # Use fewer heads to reduce memory when attention is enabled
            num_heads = max(1, min(8, common_dim // 16))
            self.attn = nn.MultiheadAttention(common_dim, num_heads=num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(common_dim)
            self.norm2 = nn.LayerNorm(common_dim)

            # Feed-forward
            self.ff = nn.Sequential(
                nn.Linear(common_dim, common_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(common_dim * 4, common_dim)
            )
        else:
            # Simple MLP fusion
            self.fusion_mlp = nn.Sequential(
                nn.Linear(common_dim * 2, common_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(common_dim * 2, common_dim)
            )

        # Output head: predict fMRI values
        # Output head: maps fused features per-voxel to scalar fMRI value
        self.output_head = nn.Sequential(
            nn.Linear(common_dim, max(32, common_dim // 4)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(max(32, common_dim // 4), 1)
        )

    def extract_3d_to_1d(self, feat_3d, mask, coords_compact):
        """
        Extract brain voxel features from 3D volume using mask (vectorized).

        Args:
            feat_3d: [B, C, D, H, W]
            mask: [D, H, W]
            coords_compact: [N, 3] coordinates of brain voxels

        Returns:
            feat_1d: [B, N, C]
        """
        B, C, D, H, W = feat_3d.shape
        N = coords_compact.shape[0]
        device = feat_3d.device

        # Get coordinates
        coords = coords_compact.long()  # [N, 3]

        # Scale coordinates to feature map size
        # Note: feat_3d might have smaller spatial dims than original mask due to pooling/strides
        scale_d = D / mask.shape[0]
        scale_h = H / mask.shape[1]
        scale_w = W / mask.shape[2]

        coords_scaled = coords.float() * torch.tensor(
            [scale_d, scale_h, scale_w],
            device=device
        )
        coords_scaled = coords_scaled.long().clamp(
            min=torch.zeros(3, device=device).long(),
            max=torch.tensor([D-1, H-1, W-1], device=device).long()
        )
        
        # Vectorized gather:
        # coords_scaled is [N, 3] -> x, y, z are [N]
        # We want feat_1d[b, i, :] = feat_3d[b, :, x[i], y[i], z[i]]
        # Transpose feat_3d to [B, D, H, W, C] for easier gathering? No, can index directly.
        
        x, y, z = coords_scaled[:, 0], coords_scaled[:, 1], coords_scaled[:, 2]
        
        # Advanced indexing:
        # feat_3d[:, :, x, y, z] will result in shape [B, C, N]
        # We want [B, N, C], so we permute.
        
        feat_1d = feat_3d[:, :, x, y, z].permute(0, 2, 1) # [B, N, C]

        return feat_1d

    def forward(self, feat_3d, feat_1d, mask, coords_compact):
        """
        Fuse 3D and 1D features.

        Args:
            feat_3d: [B, C_3d, D', H', W']
            feat_1d: [B, C_1d, N']
            mask: [D, H, W] original mask
            coords_compact: [N, 3] brain voxel coordinates

        Returns:
            pred_fmri: [B, N] predicted 1D fMRI
        """
        B = feat_3d.shape[0]

        # Extract 3D features to 1D (using mask and coordinates)
        feat_3d_1d = self.extract_3d_to_1d(feat_3d, mask, coords_compact)  # [B, N, C_3d]

        # Global average pooling on 1D features to get [B, N, C_1d]
        # Need to upsample or adapt feat_1d to match N voxels
        feat_1d_perm = feat_1d.permute(0, 2, 1)  # [B, N', C_1d]

        # Interpolate to match N voxels if needed
        if feat_1d_perm.shape[1] != feat_3d_1d.shape[1]:
            # Use adaptive pooling or interpolation
            feat_1d_perm = F.interpolate(
                feat_1d.unsqueeze(-1),  # [B, C_1d, N', 1]
                size=(self.n_voxels, 1),
                mode='bilinear',
                align_corners=False
            ).squeeze(-1).permute(0, 2, 1)  # [B, N, C_1d]

        # Project to common dimension
        feat_3d_proj = self.proj_3d(feat_3d_1d)  # [B, N, D]
        feat_1d_proj = self.proj_1d(feat_1d_perm)  # [B, N, D]

        # Fusion
        if self.fusion_type == 'attention':
            # Cross-attention: 1D attends to 3D
            attn_out, _ = self.attn(feat_1d_proj, feat_3d_proj, feat_3d_proj)
            feat_fused = self.norm1(feat_1d_proj + attn_out)

            # Feed-forward
            ff_out = self.ff(feat_fused)
            feat_fused = self.norm2(feat_fused + ff_out)
        else:
            # MLP fusion
            feat_concat = torch.cat([feat_3d_proj, feat_1d_proj], dim=-1)  # [B, N, 2D]
            feat_fused = self.fusion_mlp(feat_concat)  # [B, N, D]

        # Output head: predict fMRI
        pred_fmri = self.output_head(feat_fused).squeeze(-1)  # [B, N]

        return pred_fmri


# ============================================================================
# Dual-Stream CNN Model
# ============================================================================

class DualStreamCNN(nn.Module):
    """
    Dual-Stream CNN for fMRI generation.

    Architecture:
    - 3D Stream: ConvNeXt on 3D volume with mask
    - 1D Stream: ConvNeXt on 1D voxel sequence
    - Fusion: Combines both streams for final prediction

    Input:
    - Visual features (DINOv2 embeddings)
    - 3D fMRI volume (for 3D stream)
    - 1D fMRI vector (for 1D stream)
    - Brain mask

    Output:
    - Predicted 1D fMRI vector (15k voxels)
    """
    def __init__(
        self,
        vis_dim=768,
        compact_shape=(61, 46, 42),
        n_voxels=15724,
        base_channels_3d=64,
        base_channels_1d=128,
        depths=[2, 2, 4, 2],
        drop_path_rate=0.1,
        fusion_type='attention',
        fusion_dim: int = 128,
        use_coarse_supervision: bool = True,
    ):
        super().__init__()

        self.compact_shape = compact_shape
        self.n_voxels = n_voxels
        self.use_coarse_supervision = use_coarse_supervision

        # 0. Coarse Generator (Image -> Initial fMRI guess)
        # This initializes the streams with a rough prediction based on visual features
        self.coarse_gen = nn.Sequential(
            nn.Linear(vis_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, n_voxels)
        )

        # Visual feature projection (for semantic conditioning)
        hidden_dim = max(base_channels_3d, base_channels_1d) * 2
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 3D Stream
        self.stream_3d = Stream3D(
            in_channels=1,
            base_channels=base_channels_3d,
            depths=depths,
            drop_path_rate=drop_path_rate
        )

        # 1D Stream
        self.stream_1d = Stream1D(
            n_voxels=n_voxels,
            base_channels=base_channels_1d,
            depths=depths,
            drop_path_rate=drop_path_rate
        )

        # Visual conditioning for both streams
        self.cond_3d = nn.Linear(hidden_dim, self.stream_3d.out_channels)
        self.cond_1d = nn.Linear(hidden_dim, self.stream_1d.out_channels)

        # Fusion module
        self.fusion = FusionModule(
            channels_3d=self.stream_3d.out_channels,
            channels_1d=self.stream_1d.out_channels,
            n_voxels=n_voxels,
            fusion_type=fusion_type,
            fusion_dim=fusion_dim,
        )

        # Mean fMRI baseline (for training)
        self.register_buffer('mean_fmri_1d', torch.zeros(1, n_voxels))

    def _to_3d(self, fmri_1d: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Convert 1D fMRI to 3D volume using coordinates (vectorized)."""
        B = fmri_1d.shape[0]
        D, H, W = self.compact_shape
        # Ensure volume has same dtype as input for mixed precision compatibility
        vol = torch.zeros(B, 1, D, H, W, device=fmri_1d.device, dtype=fmri_1d.dtype)
        
        # Vectorized scatter: vol[:, 0, x, y, z] selects shape (B, N)
        # Note: coords is [N, 3], so x,y,z are [N]
        # We index vol with [:, 0, N_x, N_y, N_z] -> broadcasts to (B, N)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        vol[:, 0, x, y, z] = fmri_1d
            
        return vol

    def forward(
        self,
        vis_feat: torch.Tensor,
        mask: torch.Tensor,
        coords_compact: torch.Tensor,
        fmri_3d: Optional[torch.Tensor] = None, # Deprecated, kept for compat if needed but unused
        fmri_1d: Optional[torch.Tensor] = None, # Deprecated
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Now generates fMRI prediction purely from visual features.
        
        Args:
            vis_feat: [B, vis_dim] - Visual features (DINOv2)
            mask: [D, H, W] - Brain mask
            coords_compact: [N, 3] - Voxel coordinates
            
        Returns:
            dict with 'pred_fmri_1d', 'coarse_1d', ...
        """
        # 0. Generate initial guess (Coarse Prediction)
        coarse_1d = self.coarse_gen(vis_feat)  # [B, N]
        coarse_3d = self._to_3d(coarse_1d, coords_compact)  # [B, 1, D, H, W]

        # 1. Visual Semantic Conditioning
        vis_emb = self.vis_proj(vis_feat)  # [B, hidden_dim]

        # 2. 3D Stream (Refining spatial structure)
        feat_3d = self.stream_3d(coarse_3d, mask)  # [B, C_3d, D', H', W']

        # Add visual conditioning to 3D features
        cond_3d = self.cond_3d(vis_emb)  # [B, C_3d]
        feat_3d = feat_3d + cond_3d[:, :, None, None, None]

        # 3. 1D Stream (Refining voxel values)
        feat_1d = self.stream_1d(coarse_1d)  # [B, C_1d, N']

        # Add visual conditioning to 1D features
        cond_1d = self.cond_1d(vis_emb)  # [B, C_1d]
        feat_1d = feat_1d + cond_1d[:, :, None]

        # Fusion
        pred_fmri_1d = self.fusion(feat_3d, feat_1d, mask, coords_compact)  # [B, N]

        return {
            'pred_fmri_1d': pred_fmri_1d,
            'coarse_1d': coarse_1d,
            'feat_3d': feat_3d,
            'feat_1d': feat_1d
        }

    @torch.no_grad()
    def generate(
        self,
        vis_feat: torch.Tensor,
        mask: torch.Tensor,
        coords_compact: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate fMRI from visual features.
        """
        outputs = self.forward(vis_feat, mask, coords_compact)
        return outputs['pred_fmri_1d']


# ============================================================================
# Utility functions
# ============================================================================

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model
    print("Testing DualStreamCNN...")

    # Config
    vis_dim = 768
    compact_shape = (61, 46, 42)
    n_voxels = 15724
    batch_size = 2

    # Create dummy data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fmri_3d = torch.randn(batch_size, 1, *compact_shape).to(device)
    fmri_1d = torch.randn(batch_size, n_voxels).to(device)
    vis_feat = torch.randn(batch_size, vis_dim).to(device)

    # Create mask and coordinates
    mask = torch.zeros(compact_shape, dtype=torch.bool)
    coords_list = []
    for _ in range(n_voxels):
        coord = (
            torch.randint(0, compact_shape[0], (1,)).item(),
            torch.randint(0, compact_shape[1], (1,)).item(),
            torch.randint(0, compact_shape[2], (1,)).item()
        )
        mask[coord] = True
        coords_list.append(coord)

    coords_compact = torch.tensor(coords_list, dtype=torch.long).to(device)
    mask = mask.to(device)

    # Create model
    model = DualStreamCNN(
        vis_dim=vis_dim,
        compact_shape=compact_shape,
        n_voxels=n_voxels,
        base_channels_3d=32,  # Smaller for testing
        base_channels_1d=64,
        depths=[1, 1, 2, 1],  # Smaller for testing
        fusion_type='attention'
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Forward pass
    outputs = model(fmri_3d, fmri_1d, vis_feat, mask, coords_compact)

    print(f"\nOutput shapes:")
    print(f"  pred_fmri_1d: {outputs['pred_fmri_1d'].shape}")
    print(f"  feat_3d: {outputs['feat_3d'].shape}")
    print(f"  feat_1d: {outputs['feat_1d'].shape}")

    # Test generation
    print("\nTesting generation...")
    model.mean_fmri_1d.copy_(fmri_1d.mean(0, keepdim=True))
    pred = model.generate(vis_feat, mask, coords_compact)
    print(f"  Generated shape: {pred.shape}")

    print("\n✓ DualStreamCNN test passed!")

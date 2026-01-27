"""
Hybrid GNN+CNN Flow Matching Model for 3D fMRI Generation

Architecture:
1. Input: Visual features (DINOv2) → 3D fMRI volume (61x46x42)
2. Optional mask conditioning to guide brain region learning
3. Hybrid approach:
   - 3D CNN backbone for spatial features
   - Graph layers for long-range voxel interactions
   - Flow Matching for generation (faster than diffusion)
4. Output: Predicted 3D fMRI volume

Key Features:
- Compact 3D volume representation (61x46x42)
- Mask-aware conditioning (brain vs background)
- OT Flow Matching for stable training
- Classifier-Free Guidance (CFG) support
- Evaluation on both 3D and 1D (converted back)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict

try:
    from torch_geometric.nn import GATConv
    from torch_geometric.utils import dense_to_sparse
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("Warning: torch_geometric not available. Graph layers will be disabled.")
    TORCH_GEOMETRIC_AVAILABLE = False


def simple_radius_graph(coords: torch.Tensor, r: float, max_num_neighbors: int = 32) -> torch.Tensor:
    """
    Memory-efficient radius graph implementation using pure PyTorch.

    Args:
        coords: [N, D] normalized coordinates
        r: radius threshold
        max_num_neighbors: maximum neighbors per node

    Returns:
        edge_index: [2, num_edges] edge indices
    """
    N = coords.shape[0]
    device = coords.device

    # For large graphs, use a chunked approach to avoid OOM
    chunk_size = min(512, N)  # Process in chunks

    edges_src = []
    edges_dst = []

    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        chunk_coords = coords[i:end_i]  # [chunk_size, D]

        # Compute distances from this chunk to all nodes
        dist_chunk = torch.cdist(chunk_coords, coords, p=2)  # [chunk_size, N]

        # Find neighbors within radius (excluding self-loops)
        mask = (dist_chunk < r) & (dist_chunk > 0)

        # Get indices
        src_indices, dst_indices = torch.where(mask)
        src_indices = src_indices + i  # Offset by chunk start

        # Limit neighbors per node if needed
        if max_num_neighbors > 0 and len(src_indices) > 0:
            # Group by source node and keep top-k
            for node_offset in range(end_i - i):
                node_id = i + node_offset
                node_mask = (src_indices == node_id)
                node_neighbors = dst_indices[node_mask]

                if len(node_neighbors) > max_num_neighbors:
                    # Get distances for this node
                    node_dists = dist_chunk[node_offset, node_neighbors]
                    # Keep top-k closest
                    _, topk_idx = torch.topk(node_dists, k=max_num_neighbors, largest=False)
                    node_neighbors = node_neighbors[topk_idx]

                    # Add edges
                    edges_src.extend([node_id] * len(node_neighbors))
                    edges_dst.extend(node_neighbors.tolist())
                elif len(node_neighbors) > 0:
                    edges_src.extend([node_id] * len(node_neighbors))
                    edges_dst.extend(node_neighbors.tolist())
        else:
            edges_src.extend(src_indices.tolist())
            edges_dst.extend(dst_indices.tolist())

    if len(edges_src) == 0:
        # Return empty edge index if no edges
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=device)
    return edge_index


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for flow time t ∈ [0,1]"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Conv3DBlock(nn.Module):
    """3D Convolution block with time conditioning via FiLM"""
    def __init__(self, in_channels, out_channels, time_dim, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

        # FiLM conditioning from time embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2)
        )

    def forward(self, x, time_emb):
        """
        x: [B, C, D, H, W]
        time_emb: [B, time_dim]
        """
        h = self.conv(x)
        h = self.norm(h)

        # FiLM modulation
        time_mod = self.time_mlp(time_emb)  # [B, 2C]
        scale, shift = time_mod.chunk(2, dim=1)
        scale = scale.view(-1, h.shape[1], 1, 1, 1)
        shift = shift.view(-1, h.shape[1], 1, 1, 1)

        h = h * (1 + scale) + shift
        h = self.act(h)
        return h


class ResBlock3D(nn.Module):
    """3D Residual block with time conditioning"""
    def __init__(self, channels, time_dim, dropout=0.1):
        super().__init__()
        self.block1 = Conv3DBlock(channels, channels, time_dim)
        self.block2 = Conv3DBlock(channels, channels, time_dim)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x, time_emb):
        h = self.block1(x, time_emb)
        h = self.dropout(h)
        h = self.block2(h, time_emb)
        return x + h


class SparseGraphLayer(nn.Module):
    """
    Graph layer operating on sparse brain voxels.
    Converts 3D volume → graph → apply GNN → back to volume
    """
    def __init__(self, channels, num_heads=4, dropout=0.1):
        super().__init__()
        if not TORCH_GEOMETRIC_AVAILABLE:
            print("Warning: torch_geometric not available, using identity layer")
            self.enabled = False
            return

        self.enabled = True
        self.gat = GATConv(channels, channels // num_heads, heads=num_heads,
                          dropout=dropout, concat=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, mask, coords_compact=None, original_shape=None, radius=3.0):
        """
        x: [B, C, D, H, W] 3D volume (possibly downsampled)
        mask: [D_orig, H_orig, W_orig] boolean mask for brain voxels in ORIGINAL space
        coords_compact: [N_brain, 3] coordinates in ORIGINAL compact space (optional, cached)
        original_shape: (D_orig, H_orig, W_orig) original shape before downsampling
        radius: radius for graph connectivity

        Returns: [B, C, D, H, W] updated volume
        """
        if not self.enabled:
            return x

        B, C, D, H, W = x.shape
        device = x.device

        # Extract brain voxels
        brain_mask = mask.to(device)

        # Get coordinates if not provided
        if coords_compact is None:
            coords_compact = torch.stack(torch.where(brain_mask), dim=1).float()  # [N, 3]

        # Scale coordinates to current feature map size
        if original_shape is not None:
            D_orig, H_orig, W_orig = original_shape
            scale_factors = torch.tensor([D / D_orig, H / H_orig, W / W_orig],
                                        device=device, dtype=torch.float32)
            coords_scaled = coords_compact * scale_factors
        else:
            coords_scaled = coords_compact

        N = coords_scaled.shape[0]

        # Extract features at brain voxels: [B, C, D, H, W] → [B*N, C]
        # Create index tensors - clamp to ensure valid indices
        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N).reshape(-1)
        z_idx = coords_scaled[:, 0].long().clamp(0, D-1).unsqueeze(0).expand(B, -1).reshape(-1)
        y_idx = coords_scaled[:, 1].long().clamp(0, H-1).unsqueeze(0).expand(B, -1).reshape(-1)
        x_idx = coords_scaled[:, 2].long().clamp(0, W-1).unsqueeze(0).expand(B, -1).reshape(-1)

        # Correct indexing: reshape to [B, N] first, then use gather
        # Option 1: Use permute and advanced indexing
        x_perm = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
        node_feat = x_perm[batch_idx, z_idx, y_idx, x_idx, :]  # [B*N, C]

        # Build graph (using normalized coordinates for radius)
        coords_norm = coords_scaled / torch.tensor([D, H, W], device=device, dtype=torch.float32)
        edge_index = simple_radius_graph(coords_norm, r=radius/max(D,H,W), max_num_neighbors=32)

        # Apply graph convolution
        h = self.gat(node_feat, edge_index)
        h = self.norm(h)

        # Add residual
        node_feat = node_feat + h

        # Put back to volume
        x_out = x.clone()
        x_out_perm = x_out.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
        x_out_perm[batch_idx, z_idx, y_idx, x_idx, :] = node_feat
        x_out = x_out_perm.permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]

        return x_out


class MaskAttention(nn.Module):
    """
    Mask-aware attention to help model focus on brain regions.
    Uses mask as additional conditioning signal.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Learnable mask embedding
        self.mask_embed = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=1)
        )

        # Attention weights
        self.attention = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        """
        x: [B, C, D, H, W] features
        mask: [D, H, W] or [1, 1, D, H, W] brain mask

        Returns: [B, C, D, H, W] mask-attended features
        """
        B, C, D, H, W = x.shape

        # Expand and resize mask to match feature size
        if mask.dim() == 3:
            mask = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]

        # Resize mask if needed
        if mask.shape[-3:] != (D, H, W):
            mask = F.interpolate(mask, size=(D, H, W), mode='nearest')

        mask = mask.expand(B, 1, D, H, W)

        # Embed mask
        mask_emb = self.mask_embed(mask)  # [B, C, D, H, W]

        # Compute attention
        attn_input = torch.cat([x, mask_emb], dim=1)  # [B, 2C, D, H, W]
        attn_weights = self.attention(attn_input)  # [B, C, D, H, W]

        # Apply attention
        out = x * attn_weights + x  # Residual connection

        return out


class HybridCNN3DUNet(nn.Module):
    """
    Hybrid 3D CNN U-Net with optional graph layers and mask conditioning.

    Architecture:
    - Encoder: 3D conv blocks + optional graph layers
    - Bottleneck: ResBlocks + graph attention
    - Decoder: 3D conv blocks with skip connections
    - Mask conditioning at multiple scales
    """
    def __init__(
        self,
        time_dim=256,
        vis_dim=768,
        hidden_channels=[64, 128, 256, 512],
        use_graph_layers=True,
        use_mask_conditioning=True,
        dropout=0.1,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.use_graph_layers = use_graph_layers and TORCH_GEOMETRIC_AVAILABLE
        self.use_mask_conditioning = use_mask_conditioning

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # Visual embedding projection
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, time_dim * 2),
            nn.LayerNorm(time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )

        # Learnable null embedding for CFG
        self.null_vis_emb = nn.Parameter(torch.randn(1, time_dim) * 0.02)

        # Input: [fMRI_value + optional_mask] → channels
        in_channels = 2 if use_mask_conditioning else 1
        self.input_conv = Conv3DBlock(in_channels, hidden_channels[0], time_dim)

        # Encoder
        self.enc1 = ResBlock3D(hidden_channels[0], time_dim, dropout)
        self.down1 = nn.Conv3d(hidden_channels[0], hidden_channels[1], 3, stride=2, padding=1)

        self.enc2 = ResBlock3D(hidden_channels[1], time_dim, dropout)
        self.down2 = nn.Conv3d(hidden_channels[1], hidden_channels[2], 3, stride=2, padding=1)

        self.enc3 = ResBlock3D(hidden_channels[2], time_dim, dropout)
        self.down3 = nn.Conv3d(hidden_channels[2], hidden_channels[3], 3, stride=2, padding=1)

        # Optional graph layers
        if self.use_graph_layers:
            self.graph1 = SparseGraphLayer(hidden_channels[1], num_heads=4, dropout=dropout)
            self.graph2 = SparseGraphLayer(hidden_channels[2], num_heads=4, dropout=dropout)
            self.graph3 = SparseGraphLayer(hidden_channels[3], num_heads=4, dropout=dropout)

        # Mask attention
        if self.use_mask_conditioning:
            self.mask_attn1 = MaskAttention(hidden_channels[1])
            self.mask_attn2 = MaskAttention(hidden_channels[2])
            self.mask_attn3 = MaskAttention(hidden_channels[3])

        # Bottleneck
        self.mid1 = ResBlock3D(hidden_channels[3], time_dim, dropout)
        self.mid2 = ResBlock3D(hidden_channels[3], time_dim, dropout)

        # Decoder
        self.up3 = nn.ConvTranspose3d(hidden_channels[3], hidden_channels[2], 4, stride=2, padding=1)
        self.dec3 = ResBlock3D(hidden_channels[2] * 2, time_dim, dropout)  # *2 for skip

        self.up2 = nn.ConvTranspose3d(hidden_channels[2] * 2, hidden_channels[1], 4, stride=2, padding=1)
        self.dec2 = ResBlock3D(hidden_channels[1] * 2, time_dim, dropout)

        self.up1 = nn.ConvTranspose3d(hidden_channels[1] * 2, hidden_channels[0], 4, stride=2, padding=1)
        self.dec1 = ResBlock3D(hidden_channels[0] * 2, time_dim, dropout)

        # Output: hidden → 1 (velocity for fMRI)
        self.output_conv = nn.Sequential(
            nn.Conv3d(hidden_channels[0] * 2, hidden_channels[0], 3, padding=1),
            nn.GroupNorm(8, hidden_channels[0]),
            nn.SiLU(),
            nn.Conv3d(hidden_channels[0], 1, 1)
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        vis_feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        coords_compact: Optional[torch.Tensor] = None,
        use_cfg_dropout: bool = False,
        cfg_dropout: float = 0.1
    ):
        """
        x_t: [B, 1, D, H, W] fMRI at flow time t (compact 3D volume)
        t: [B] flow time ∈ [0, 1]
        vis_feat: [B, vis_dim] visual features
        mask: [D, H, W] brain mask (optional but recommended)
        coords_compact: [N_brain, 3] cached brain coordinates (optional)
        use_cfg_dropout: Apply CFG dropout during training
        cfg_dropout: Dropout rate for CFG

        Returns: [B, 1, D, H, W] predicted velocity
        """
        B = x_t.shape[0]
        device = x_t.device

        # Time embedding
        t_emb = self.time_mlp(t)  # [B, time_dim]

        # Visual embedding with optional CFG dropout
        v_emb = self.vis_proj(vis_feat)  # [B, time_dim]

        if self.training and use_cfg_dropout:
            mask_cfg = torch.rand(B, 1, device=device) > cfg_dropout
            null_emb = self.null_vis_emb.expand(B, -1)
            v_emb = torch.where(mask_cfg, v_emb, null_emb)

        # Combined conditioning
        cond = t_emb + v_emb  # [B, time_dim]

        # Input: optionally concat mask
        if self.use_mask_conditioning and mask is not None:
            if mask.dim() == 3:
                mask_input = mask.unsqueeze(0).unsqueeze(0).float().expand(B, 1, -1, -1, -1)
            else:
                mask_input = mask.expand(B, 1, -1, -1, -1)
            x_input = torch.cat([x_t, mask_input], dim=1)  # [B, 2, D, H, W]
        else:
            x_input = x_t

        # Input projection
        h = self.input_conv(x_input, cond)  # [B, C0, D, H, W]

        # Store original shape for graph layers
        original_shape = (x_t.shape[2], x_t.shape[3], x_t.shape[4])  # (D, H, W)

        # Encoder with skip connections
        h1 = self.enc1(h, cond)  # [B, C0, D, H, W]
        h = self.down1(h1)  # [B, C1, D/2, H/2, W/2]

        h2 = self.enc2(h, cond)
        if self.use_mask_conditioning and mask is not None:
            # MaskAttention will auto-resize mask to match h2 shape
            h2 = self.mask_attn1(h2, mask)
        if self.use_graph_layers and mask is not None:
            h2 = self.graph1(h2, mask, coords_compact, original_shape)
        h = self.down2(h2)  # [B, C2, D/4, H/4, W/4]

        h3 = self.enc3(h, cond)
        if self.use_mask_conditioning and mask is not None:
            h3 = self.mask_attn2(h3, mask)
        if self.use_graph_layers and mask is not None:
            h3 = self.graph2(h3, mask, coords_compact, original_shape)
        h = self.down3(h3)  # [B, C3, D/8, H/8, W/8]

        # Bottleneck
        if self.use_mask_conditioning and mask is not None:
            h = self.mask_attn3(h, mask)
        if self.use_graph_layers and mask is not None:
            h = self.graph3(h, mask, coords_compact, original_shape)

        h = self.mid1(h, cond)
        h = self.mid2(h, cond)

        # Decoder with skip connections
        h = self.up3(h)  # [B, C2, D/4, H/4, W/4]
        # Resize h to match h3 if needed
        if h.shape != h3.shape:
            h = F.interpolate(h, size=h3.shape[2:], mode='trilinear', align_corners=False)
        h = torch.cat([h, h3], dim=1)
        h = self.dec3(h, cond)

        h = self.up2(h)  # [B, C1, D/2, H/2, W/2]
        if h.shape != h2.shape:
            h = F.interpolate(h, size=h2.shape[2:], mode='trilinear', align_corners=False)
        h = torch.cat([h, h2], dim=1)
        h = self.dec2(h, cond)

        h = self.up1(h)  # [B, C0, D, H, W]
        if h.shape != h1.shape:
            h = F.interpolate(h, size=h1.shape[2:], mode='trilinear', align_corners=False)
        h = torch.cat([h, h1], dim=1)
        h = self.dec1(h, cond)

        # Output: velocity
        velocity = self.output_conv(h)  # [B, 1, D, H, W]

        return velocity


class FlowMatching3DManager:
    """
    Flow Matching manager for 3D volumes using Optimal Transport.

    OT Flow: x_t = (1-t) * x_0 + t * x_1 + σ * ε
    Target velocity: v = x_1 - x_0
    """
    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def get_train_tuple(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample training tuple for flow matching.

        Args:
            x_0: [B, 1, D, H, W] source (mean_fmri)
            x_1: [B, 1, D, H, W] target (ground truth)

        Returns:
            x_t: [B, 1, D, H, W] interpolated state
            t: [B] flow time
            target_v: [B, 1, D, H, W] target velocity
        """
        B = x_0.shape[0]
        device = x_0.device

        # Sample random time
        t = torch.rand(B, device=device)

        # OT interpolation
        t_expanded = t.view(B, 1, 1, 1, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

        # Add noise if sigma > 0
        if self.sigma > 0:
            noise = torch.randn_like(x_t) * self.sigma
            x_t = x_t + noise

        # Target velocity (constant for OT)
        target_v = x_1 - x_0

        return x_t, t, target_v

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        vis_feat: torch.Tensor,
        x_0: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        coords_compact: Optional[torch.Tensor] = None,
        steps: int = 20,
        cfg_scale: float = 1.0,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate 3D fMRI by integrating velocity field.

        Args:
            model: HybridCNN3DUNet
            vis_feat: [B, vis_dim] visual features
            x_0: [B, 1, D, H, W] initial state (mean_fmri)
            mask: [D, H, W] brain mask
            coords_compact: [N_brain, 3] brain coordinates
            steps: Number of integration steps
            cfg_scale: Classifier-free guidance scale

        Returns:
            [B, 1, D, H, W] generated 3D fMRI
        """
        B = vis_feat.shape[0]
        dt = 1.0 / steps

        x = x_0.clone()

        for i in range(steps):
            t = torch.full((B,), i / steps, device=device, dtype=torch.float32)

            if cfg_scale > 1.0:
                # Conditional prediction
                v_cond = model(x, t, vis_feat, mask, coords_compact, use_cfg_dropout=False)

                # Unconditional prediction
                null_vis = model.null_vis_emb.expand(B, -1)
                v_uncond = model(x, t, null_vis, mask, coords_compact, use_cfg_dropout=False)

                # Guided velocity
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model(x, t, vis_feat, mask, coords_compact, use_cfg_dropout=False)

            # Euler integration
            x = x + v * dt

        return x


class Flow3DHybridARM(nn.Module):
    """
    Complete hybrid GNN+CNN Flow Matching model for 3D fMRI generation.

    Features:
    - 3D CNN U-Net backbone
    - Optional graph layers for long-range interactions
    - Optional mask conditioning
    - Flow Matching (OT path)
    - CFG support
    - Evaluation on both 3D and 1D (lossless conversion)
    """
    def __init__(
        self,
        input_dim: int = 768,  # DINOv2 embedding
        compact_shape: Tuple[int, int, int] = (61, 46, 42),
        hidden_dim: int = 256,
        time_dim: int = 256,
        use_graph_layers: bool = True,
        use_mask_conditioning: bool = True,
        dropout: float = 0.1,
        cfg_dropout: float = 0.1,
        flow_sigma: float = 0.0,
    ):
        super().__init__()
        self.compact_shape = compact_shape
        self.use_mask_conditioning = use_mask_conditioning

        # 3D U-Net
        self.unet = HybridCNN3DUNet(
            time_dim=time_dim,
            vis_dim=input_dim,
            hidden_channels=[hidden_dim // 4, hidden_dim // 2, hidden_dim, hidden_dim * 2],
            use_graph_layers=use_graph_layers,
            use_mask_conditioning=use_mask_conditioning,
            dropout=dropout,
        )

        # Flow manager
        self.flow_manager = FlowMatching3DManager(sigma=flow_sigma)

        # Mean fMRI baseline (will be set during training)
        self.register_buffer('mean_fmri_3d', torch.zeros(1, 1, *compact_shape))

        # CFG dropout
        self.cfg_dropout = cfg_dropout

    def forward(
        self,
        clean_fmri_3d: torch.Tensor,
        vis_feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        coords_compact: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            clean_fmri_3d: [B, 1, D, H, W] ground truth 3D fMRI
            vis_feat: [B, vis_dim] visual features
            mask: [D, H, W] brain mask
            coords_compact: [N_brain, 3] brain coordinates

        Returns:
            dict with 'pred_velocity', 'target_velocity', 't', 'x_pred'
        """
        B = clean_fmri_3d.shape[0]

        # x_0 = mean, x_1 = target
        x_0 = self.mean_fmri_3d.expand(B, -1, -1, -1, -1)
        x_1 = clean_fmri_3d

        # Get training tuple
        x_t, t, target_v = self.flow_manager.get_train_tuple(x_0, x_1)

        # Predict velocity
        pred_v = self.unet(x_t, t, vis_feat, mask, coords_compact,
                          use_cfg_dropout=True, cfg_dropout=self.cfg_dropout)

        return {
            'pred_velocity': pred_v,
            'target_velocity': target_v,
            't': t,
            'x_pred': x_0 + pred_v,  # For loss computation
        }

    @torch.no_grad()
    def generate(
        self,
        vis_feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        coords_compact: Optional[torch.Tensor] = None,
        steps: int = 20,
        cfg_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Generate 3D fMRI from visual features.

        Args:
            vis_feat: [B, vis_dim] visual features
            mask: [D, H, W] brain mask
            coords_compact: [N_brain, 3] brain coordinates
            steps: Number of integration steps
            cfg_scale: Classifier-free guidance scale

        Returns:
            [B, 1, D, H, W] generated 3D fMRI
        """
        device = vis_feat.device
        B = vis_feat.shape[0]

        x_0 = self.mean_fmri_3d.expand(B, -1, -1, -1, -1)

        x = self.flow_manager.sample(
            self.unet, vis_feat, x_0, mask, coords_compact,
            steps=steps, cfg_scale=cfg_scale, device=device
        )

        return x


# Utility functions for 3D ↔ 1D conversion

def fmri_1d_to_3d_compact(
    fmri_1d: np.ndarray,
    coords_compact: np.ndarray,
    compact_shape: Tuple[int, int, int],
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Convert fMRI 1D vector to 3D compact volume.

    Args:
        fmri_1d: [N_voxels] or [B, N_voxels] fMRI data
        coords_compact: [N_voxels, 3] coordinates in compact space
        compact_shape: (D, H, W) shape of compact volume
        fill_value: value for background voxels

    Returns:
        [D, H, W] or [B, D, H, W] 3D volume
    """
    if fmri_1d.ndim == 1:
        volume_3d = np.full(compact_shape, fill_value, dtype=fmri_1d.dtype)
        for idx, coord in enumerate(coords_compact):
            volume_3d[tuple(coord)] = fmri_1d[idx]
        return volume_3d
    else:
        B = fmri_1d.shape[0]
        volumes = np.full((B, *compact_shape), fill_value, dtype=fmri_1d.dtype)
        for b in range(B):
            for idx, coord in enumerate(coords_compact):
                volumes[b, tuple(coord)] = fmri_1d[b, idx]
        return volumes


def fmri_3d_compact_to_1d(
    volume_3d: np.ndarray,
    compact_mask: np.ndarray
) -> np.ndarray:
    """
    Convert 3D compact volume back to 1D fMRI vector.

    Args:
        volume_3d: [D, H, W] or [B, D, H, W] 3D volume
        compact_mask: [D, H, W] boolean mask

    Returns:
        [N_voxels] or [B, N_voxels] fMRI data
    """
    if volume_3d.ndim == 3:
        return volume_3d[compact_mask]
    else:
        B = volume_3d.shape[0]
        N = compact_mask.sum()
        fmri_1d = np.zeros((B, N), dtype=volume_3d.dtype)
        for b in range(B):
            fmri_1d[b] = volume_3d[b][compact_mask]
        return fmri_1d

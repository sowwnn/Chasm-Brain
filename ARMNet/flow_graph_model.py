"""
Graph Flow Matching Model for Image-to-fMRI Generation

Combines Graph Neural Networks with Flow Matching for fMRI prediction.
Based on Optimal Transport (OT) flow matching framework.

Key features:
- Graph U-Net architecture (from DiffusionGraphARM)
- Flow matching instead of diffusion (faster sampling)
- Velocity prediction with OT path: x_0 (mean) -> x_1 (target)
- Point cloud representation with KNN graph structure
- Classifier-Free Guidance (CFG) support

Flow Matching vs Diffusion:
- Simpler: Direct velocity prediction instead of noise prediction
- Faster: Fewer sampling steps needed (10-20 vs 50-1000)
- Stable: Deterministic OT path, no noise schedule needed
- Performance: Often matches or exceeds diffusion quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from torch_geometric.nn import GATConv, GCNConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("Warning: torch_geometric not installed. Install with: pip install torch-geometric")
    TORCH_GEOMETRIC_AVAILABLE = False


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps (actually flow time t ∈ [0,1])"""
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


class GraphConvBlock(nn.Module):
    """
    Graph convolution block with time conditioning.
    Reused from DiffusionGraphARM.
    """
    def __init__(self, in_channels, out_channels, time_dim, num_heads=4, gnn_type='gat', dropout=0.1):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_heads = num_heads

        if gnn_type == 'gat':
            self.conv = GATConv(
                in_channels,
                out_channels // num_heads,
                heads=num_heads,
                dropout=dropout,
                concat=True,
            )
        elif gnn_type == 'gcn':
            self.conv = GCNConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()

        # Time embedding projection (FiLM conditioning)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2)
        )

    def forward(self, x, edge_index, time_emb, batch=None):
        """
        x: [N, in_channels] node features
        edge_index: [2, E] graph edges
        time_emb: [B, time_dim] time embedding
        batch: [N] batch assignment
        """
        h = self.conv(x, edge_index)
        h = self.norm(h)

        # FiLM conditioning
        if batch is not None:
            time_emb_node = time_emb[batch]
        else:
            time_emb_node = time_emb.expand(x.shape[0], -1)

        time_mod = self.time_mlp(time_emb_node)
        scale, shift = time_mod.chunk(2, dim=-1)

        h = h * (1 + scale) + shift
        h = self.act(h)
        return h


class GraphResBlock(nn.Module):
    """Graph residual block with time conditioning"""
    def __init__(self, channels, time_dim, num_heads=4, gnn_type='gat', dropout=0.1):
        super().__init__()
        self.block1 = GraphConvBlock(channels, channels, time_dim, num_heads, gnn_type, dropout)
        self.block2 = GraphConvBlock(channels, channels, time_dim, num_heads, gnn_type, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, time_emb, batch=None):
        h = self.block1(x, edge_index, time_emb, batch)
        h = self.dropout(h)
        h = self.block2(h, edge_index, time_emb, batch)
        return x + h


class GraphCrossAttention(nn.Module):
    """
    Cross-attention between graph nodes and visual features.
    Each node attends to the visual embedding.
    """
    def __init__(self, node_dim, vis_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_node = nn.LayerNorm(node_dim)
        self.norm_vis = nn.LayerNorm(vis_dim)

        self.to_q = nn.Linear(node_dim, node_dim)
        self.to_k = nn.Linear(vis_dim, node_dim)
        self.to_v = nn.Linear(vis_dim, node_dim)
        self.to_out = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.Dropout(dropout)
        )

        # Zero-initialize for residual learning
        nn.init.zeros_(self.to_out[0].weight)
        nn.init.zeros_(self.to_out[0].bias)

    def forward(self, node_feat, vis_feat, batch=None):
        """
        node_feat: [N, node_dim]
        vis_feat: [B, vis_dim]
        batch: [N] batch assignment
        """
        residual = node_feat

        node_feat = self.norm_node(node_feat)
        vis_feat = self.norm_vis(vis_feat)

        # Broadcast visual features to nodes
        if batch is not None:
            vis_feat_node = vis_feat[batch]
        else:
            vis_feat_node = vis_feat.expand(node_feat.shape[0], -1)

        N = node_feat.shape[0]

        q = self.to_q(node_feat).view(N, self.num_heads, self.head_dim)
        k = self.to_k(vis_feat_node).view(N, self.num_heads, self.head_dim)
        v = self.to_v(vis_feat_node).view(N, self.num_heads, self.head_dim)

        attn = torch.einsum('nhd,nhd->nh', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('nh,nhd->nhd', attn, v)
        out = out.reshape(N, -1)
        out = self.to_out(out)

        return residual + out


class GraphFlowUNet(nn.Module):
    """
    Graph U-Net for Flow Matching with visual conditioning.

    Predicts velocity field v(x_t, t, vis_feat) where:
    - x_t: Current fMRI state at flow time t
    - t: Flow time ∈ [0, 1]
    - vis_feat: Visual conditioning (DINOv2 embedding)

    Output: Velocity v such that x_1 ≈ x_0 + ∫v dt
    """
    def __init__(
        self,
        time_dim=256,
        vis_dim=768,
        hidden_channels=[64, 128, 256],
        num_heads=8,
        gnn_type='gat',
        dropout=0.1,
        cfg_dropout=0.1,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.cfg_dropout = cfg_dropout
        self.hidden_channels = hidden_channels

        # Time embedding (flow time t ∈ [0,1])
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # Visual feature projection
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, time_dim * 2),
            nn.LayerNorm(time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim * 2),
            nn.LayerNorm(time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )

        # Learnable null embedding for CFG
        self.null_vis_emb = nn.Parameter(torch.randn(1, time_dim) * 0.02)

        # Input projection: [fMRI value, x, y, z] -> hidden
        self.input_proj = nn.Sequential(
            nn.Linear(4, hidden_channels[0]),
            nn.LayerNorm(hidden_channels[0]),
            nn.SiLU()
        )

        # Encoder
        self.enc1 = GraphConvBlock(hidden_channels[0], hidden_channels[0], time_dim, num_heads, gnn_type, dropout)
        self.enc2 = GraphConvBlock(hidden_channels[0], hidden_channels[1], time_dim, num_heads, gnn_type, dropout)
        self.enc3 = GraphConvBlock(hidden_channels[1], hidden_channels[2], time_dim, num_heads, gnn_type, dropout)

        self.enc2_attn = GraphCrossAttention(hidden_channels[1], time_dim, num_heads, dropout)
        self.enc3_attn = GraphCrossAttention(hidden_channels[2], time_dim, num_heads, dropout)

        # Middle
        self.mid1 = GraphResBlock(hidden_channels[2], time_dim, num_heads, gnn_type, dropout)
        self.mid_attn = GraphCrossAttention(hidden_channels[2], time_dim, num_heads, dropout)
        self.mid2 = GraphResBlock(hidden_channels[2], time_dim, num_heads, gnn_type, dropout)

        # Decoder
        self.dec3 = GraphConvBlock(hidden_channels[2] * 2, hidden_channels[1], time_dim, num_heads, gnn_type, dropout)
        self.dec2 = GraphConvBlock(hidden_channels[1] * 2, hidden_channels[0], time_dim, num_heads, gnn_type, dropout)
        self.dec1 = GraphConvBlock(hidden_channels[0], hidden_channels[0], time_dim, num_heads, gnn_type, dropout)

        self.dec3_attn = GraphCrossAttention(hidden_channels[1], time_dim, num_heads, dropout)
        self.dec2_attn = GraphCrossAttention(hidden_channels[0], time_dim, num_heads, dropout)

        # Output projection: hidden -> 1 (predicted velocity for fMRI)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels[0], hidden_channels[0]),
            nn.SiLU(),
            nn.Linear(hidden_channels[0], 1)
        )

    def forward(self, x_t, coords, edge_index, vis_feat, t, batch=None, use_cfg_dropout=True):
        """
        x_t: [B, N] or [N] fMRI at flow time t
        coords: [B, N, 3] or [N, 3] spatial coordinates
        edge_index: [2, E] graph edges
        vis_feat: [B, vis_dim] visual features
        t: [B] flow time ∈ [0, 1]
        batch: [N] batch assignment
        use_cfg_dropout: Apply CFG dropout during training

        Returns: [B, N] predicted velocity
        """
        B = vis_feat.shape[0]

        # Handle input shapes
        if x_t.dim() == 2:
            x_t = x_t.view(-1)
        if coords.dim() == 3:
            N = coords.shape[1]
            coords = coords.view(-1, 3)
        else:
            N = coords.shape[0] // B

        # Create batch assignment if not provided
        if batch is None:
            batch = torch.arange(B, device=x_t.device).repeat_interleave(N)

        # Time embedding
        t_emb = self.time_mlp(t)  # [B, time_dim]

        # Visual embedding with CFG dropout
        if vis_feat.shape[-1] != self.time_dim:
            v_emb = self.vis_proj(vis_feat)
        else:
            v_emb = vis_feat

        # Apply CFG dropout during training
        if self.training and use_cfg_dropout and self.cfg_dropout > 0:
            mask = torch.rand(B, 1, device=vis_feat.device) > self.cfg_dropout
            null_emb = self.null_vis_emb.expand(B, -1)
            v_emb = torch.where(mask, v_emb, null_emb)

        # Combined conditioning
        cond = t_emb + v_emb  # [B, time_dim]

        # Build node features: [fMRI value, x, y, z]
        node_features = torch.cat([x_t.unsqueeze(-1), coords], dim=-1)  # [B*N, 4]

        # Input projection
        h = self.input_proj(node_features)

        # Encoder
        h1 = self.enc1(h, edge_index, cond, batch)

        h2 = self.enc2(h1, edge_index, cond, batch)
        h2 = self.enc2_attn(h2, v_emb, batch)

        h3 = self.enc3(h2, edge_index, cond, batch)
        h3 = self.enc3_attn(h3, v_emb, batch)

        # Middle
        h = self.mid1(h3, edge_index, cond, batch)
        h = self.mid_attn(h, v_emb, batch)
        h = self.mid2(h, edge_index, cond, batch)

        # Decoder with skip connections
        h = torch.cat([h, h3], dim=-1)
        h = self.dec3(h, edge_index, cond, batch)
        h = self.dec3_attn(h, v_emb, batch)

        h = torch.cat([h, h2], dim=-1)
        h = self.dec2(h, edge_index, cond, batch)
        h = self.dec2_attn(h, v_emb, batch)

        h = self.dec1(h, edge_index, cond, batch)

        # Output: velocity
        pred_velocity = self.output_proj(h).squeeze(-1)  # [B*N]
        pred_velocity = pred_velocity.view(B, N)

        return pred_velocity


class FlowMatchingManager:
    """
    Flow Matching manager using Optimal Transport (OT) path.

    OT Flow: x_t = (1-t) * x_0 + t * x_1 + σ * ε
    where:
    - x_0: source (mean_fmri)
    - x_1: target (ground truth fMRI)
    - t ∈ [0, 1]: flow time
    - σ: noise scale (small)
    - ε: Gaussian noise

    Target velocity: v = x_1 - x_0 (constant along OT path)
    """
    def __init__(self, sigma=0.0):
        """
        sigma: Noise scale for stochastic flow (0.0 = deterministic OT)
        """
        self.sigma = sigma

    def get_train_tuple(self, x_0, x_1):
        """
        Sample training tuple (x_t, t, target_velocity).

        Args:
            x_0: [B, N] source (mean_fmri)
            x_1: [B, N] target (ground truth)

        Returns:
            x_t: [B, N] interpolated state
            t: [B] flow time
            target_v: [B, N] target velocity
        """
        B = x_0.shape[0]
        device = x_0.device

        # Sample random time
        t = torch.rand(B, device=device)  # [B]

        # OT interpolation
        t_expanded = t.view(B, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

        # Add noise if sigma > 0
        if self.sigma > 0:
            noise = torch.randn_like(x_t) * self.sigma
            x_t = x_t + noise

        # Target velocity (constant for OT)
        target_v = x_1 - x_0

        return x_t, t, target_v

    @torch.no_grad()
    def sample(self, model, coords, edge_index, vis_feat, x_0, steps=20, batch=None, cfg_scale=1.0, device='cuda'):
        """
        Generate fMRI by integrating velocity field.

        Uses Euler integration: x_{t+dt} = x_t + v(x_t, t) * dt

        Args:
            model: GraphFlowUNet
            coords: [B, N, 3] or [N, 3] coordinates
            edge_index: [2, E] edges
            vis_feat: [B, vis_dim] visual features
            x_0: [B, N] initial state (mean_fmri)
            steps: Number of integration steps
            batch: [N] batch assignment
            cfg_scale: Classifier-free guidance scale

        Returns:
            [B, N] generated fMRI
        """
        B = vis_feat.shape[0]
        dt = 1.0 / steps

        x = x_0.clone()

        for i in range(steps):
            t = torch.full((B,), i / steps, device=device, dtype=torch.float32)

            if cfg_scale > 1.0:
                # CFG: interpolate between conditional and unconditional
                v_cond = model(x, coords, edge_index, vis_feat, t, batch, use_cfg_dropout=False)

                # Unconditional prediction
                null_vis = model.null_vis_emb.expand(B, -1) if hasattr(model, 'null_vis_emb') else vis_feat * 0
                v_uncond = model(x, coords, edge_index, null_vis, t, batch, use_cfg_dropout=False)

                # Guided velocity
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model(x, coords, edge_index, vis_feat, t, batch, use_cfg_dropout=False)

            # Euler integration
            x = x + v * dt

        return x


class FlowGraphARM(nn.Module):
    """
    Complete Flow Matching Graph ARM model.
    Combines GraphFlowUNet with FlowMatchingManager.
    """
    def __init__(
        self,
        input_dim=768,
        output_dim=15724,
        hidden_dim=256,
        time_dim=256,
        num_heads=8,
        gnn_type='gat',
        dropout=0.1,
        cfg_dropout=0.1,
        flow_sigma=0.0,
    ):
        super().__init__()
        self.output_dim = output_dim

        # Graph Flow U-Net
        self.unet = GraphFlowUNet(
            time_dim=time_dim,
            vis_dim=input_dim,
            hidden_channels=[hidden_dim // 2, hidden_dim, hidden_dim * 2],
            num_heads=num_heads,
            gnn_type=gnn_type,
            dropout=dropout,
            cfg_dropout=cfg_dropout,
        )

        # Flow manager
        self.flow_manager = FlowMatchingManager(sigma=flow_sigma)

        # Mean fMRI baseline
        self.register_buffer('mean_fmri', torch.zeros(output_dim))

    def forward(self, clean_fmri, coords, edge_index, vis_feat, batch=None):
        """
        Training forward pass.

        Returns: dict with 'velocity_loss', 'pred_velocity', 'target_velocity', 't'
        """
        B = clean_fmri.shape[0]
        device = clean_fmri.device

        # x_0 = mean, x_1 = target
        x_0 = self.mean_fmri.unsqueeze(0).expand(B, -1)
        x_1 = clean_fmri

        # Get training tuple
        x_t, t, target_v = self.flow_manager.get_train_tuple(x_0, x_1)

        # Predict velocity
        pred_v = self.unet(x_t, coords, edge_index, vis_feat, t, batch, use_cfg_dropout=True)

        return {
            'pred_velocity': pred_v,
            'target_velocity': target_v,
            't': t,
            'x_pred': x_0 + pred_v,  # For loss computation
        }

    @torch.no_grad()
    def generate(self, coords, edge_index, vis_feat, batch=None, steps=20, cfg_scale=1.0):
        """
        Generate fMRI from visual features.

        Args:
            steps: Number of integration steps (10-20 typical)
            cfg_scale: Classifier-free guidance scale

        Returns: [B, N] generated fMRI
        """
        device = vis_feat.device
        B = vis_feat.shape[0]

        x_0 = self.mean_fmri.unsqueeze(0).expand(B, -1)

        x = self.flow_manager.sample(
            self.unet, coords, edge_index, vis_feat, x_0,
            steps=steps, batch=batch, cfg_scale=cfg_scale, device=device
        )

        return x


# Check dependencies
if not TORCH_GEOMETRIC_AVAILABLE:
    raise ImportError(
        "This module requires torch_geometric. Install with:\n"
        "pip install torch-geometric torch-scatter torch-sparse"
    )

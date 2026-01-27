"""
Graph Neural Network Diffusion Model for Image-to-fMRI Generation

Adapts DiffusionARM3D to use Graph Neural Networks instead of 3D CNN.
Processes fMRI as a point cloud with graph structure (KNN edges).

Key features:
- Graph U-Net architecture with pooling/unpooling
- Graph attention layers (GAT) for spatial relationships
- Cross-attention between graph nodes and visual features
- Diffusion process with DDPM/DDIM sampling
- Classifier-Free Guidance (CFG) support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_add_pool
    from torch_geometric.utils import to_dense_batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("Warning: torch_geometric not installed. Install with: pip install torch-geometric")
    TORCH_GEOMETRIC_AVAILABLE = False


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps"""
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

    Uses Graph Attention Network (GAT) by default for learning edge weights.
    """
    def __init__(self, in_channels, out_channels, time_dim, num_heads=4, gnn_type='gat', dropout=0.1):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_heads = num_heads

        if gnn_type == 'gat':
            # GAT with multi-head attention
            self.conv = GATConv(
                in_channels,
                out_channels // num_heads,  # head_dim
                heads=num_heads,
                dropout=dropout,
                concat=True,
            )
        elif gnn_type == 'gcn':
            # Simple GCN
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
        batch: [N] batch assignment (optional for batched graphs)

        Returns: [N, out_channels]
        """
        # Apply graph convolution
        h = self.conv(x, edge_index)
        h = self.norm(h)

        # FiLM conditioning with time
        # Need to broadcast time_emb to all nodes in the batch
        if batch is not None:
            # Map batch-level time_emb to node-level
            time_emb_node = time_emb[batch]  # [N, time_dim]
        else:
            # Assume single graph (batch size 1)
            time_emb_node = time_emb.expand(x.shape[0], -1)

        time_mod = self.time_mlp(time_emb_node)  # [N, out_channels * 2]
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

        # Zero-initialize output for residual learning
        nn.init.zeros_(self.to_out[0].weight)
        nn.init.zeros_(self.to_out[0].bias)

    def forward(self, node_feat, vis_feat, batch=None):
        """
        node_feat: [N, node_dim] node features
        vis_feat: [B, vis_dim] visual features
        batch: [N] batch assignment for mapping nodes to batch indices

        Returns: [N, node_dim]
        """
        residual = node_feat

        # Normalize
        node_feat = self.norm_node(node_feat)
        vis_feat = self.norm_vis(vis_feat)

        # Broadcast visual features to all nodes
        if batch is not None:
            vis_feat_node = vis_feat[batch]  # [N, vis_dim]
        else:
            # Single graph
            vis_feat_node = vis_feat.expand(node_feat.shape[0], -1)

        # Multi-head attention
        N = node_feat.shape[0]

        q = self.to_q(node_feat).view(N, self.num_heads, self.head_dim)  # [N, heads, head_dim]
        k = self.to_k(vis_feat_node).view(N, self.num_heads, self.head_dim)
        v = self.to_v(vis_feat_node).view(N, self.num_heads, self.head_dim)

        # Attention: [N, heads]
        attn = torch.einsum('nhd,nhd->nh', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention: [N, heads, head_dim]
        out = torch.einsum('nh,nhd->nhd', attn, v)
        out = out.reshape(N, -1)
        out = self.to_out(out)

        return residual + out


class GraphUNet(nn.Module):
    """
    Graph U-Net for fMRI diffusion with visual conditioning.

    Note: This is a simplified version without graph pooling/unpooling.
    All operations are at the full graph resolution (15724 nodes).
    For hierarchical design, would need to add graph coarsening (TopKPooling, SAGPooling).
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

        # Time embedding
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

        # Input projection: node features (fMRI + coords) -> hidden
        # Input: 1 (fMRI value) + 3 (spatial coords) = 4 features per node
        self.input_proj = nn.Sequential(
            nn.Linear(4, hidden_channels[0]),
            nn.LayerNorm(hidden_channels[0]),
            nn.SiLU()
        )

        # Encoder blocks
        self.enc1 = GraphConvBlock(hidden_channels[0], hidden_channels[0], time_dim, num_heads, gnn_type, dropout)
        self.enc2 = GraphConvBlock(hidden_channels[0], hidden_channels[1], time_dim, num_heads, gnn_type, dropout)
        self.enc3 = GraphConvBlock(hidden_channels[1], hidden_channels[2], time_dim, num_heads, gnn_type, dropout)

        # Cross-attention in encoder
        self.enc2_attn = GraphCrossAttention(hidden_channels[1], time_dim, num_heads, dropout)
        self.enc3_attn = GraphCrossAttention(hidden_channels[2], time_dim, num_heads, dropout)

        # Middle blocks
        self.mid1 = GraphResBlock(hidden_channels[2], time_dim, num_heads, gnn_type, dropout)
        self.mid_attn = GraphCrossAttention(hidden_channels[2], time_dim, num_heads, dropout)
        self.mid2 = GraphResBlock(hidden_channels[2], time_dim, num_heads, gnn_type, dropout)

        # Decoder blocks (with skip connections)
        self.dec3 = GraphConvBlock(hidden_channels[2] * 2, hidden_channels[1], time_dim, num_heads, gnn_type, dropout)
        self.dec2 = GraphConvBlock(hidden_channels[1] * 2, hidden_channels[0], time_dim, num_heads, gnn_type, dropout)
        self.dec1 = GraphConvBlock(hidden_channels[0], hidden_channels[0], time_dim, num_heads, gnn_type, dropout)

        # Cross-attention in decoder
        self.dec3_attn = GraphCrossAttention(hidden_channels[1], time_dim, num_heads, dropout)
        self.dec2_attn = GraphCrossAttention(hidden_channels[0], time_dim, num_heads, dropout)

        # Output projection: hidden -> 1 (predicted noise for fMRI)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels[0], hidden_channels[0]),
            nn.SiLU(),
            nn.Linear(hidden_channels[0], 1)
        )

    def forward(self, noisy_fmri, coords, edge_index, vis_feat, t, batch=None, use_cfg_dropout=True):
        """
        noisy_fmri: [N] or [B, N] noisy fMRI values
        coords: [N, 3] or [B, N, 3] spatial coordinates
        edge_index: [2, E] graph edges
        vis_feat: [B, vis_dim] visual features
        t: [B] timestep
        batch: [N] batch assignment (required for batched graphs)
        use_cfg_dropout: bool, whether to apply CFG dropout

        Returns: [N] or [B, N] predicted noise
        """
        B = vis_feat.shape[0]

        # Handle input shapes
        if noisy_fmri.dim() == 2:
            # [B, N] -> [B*N]
            noisy_fmri = noisy_fmri.view(-1)
        if coords.dim() == 3:
            # [B, N, 3] -> [B*N, 3]
            N = coords.shape[1]
            coords = coords.view(-1, 3)
        else:
            N = coords.shape[0] // B

        # Create batch assignment if not provided
        if batch is None:
            batch = torch.arange(B, device=noisy_fmri.device).repeat_interleave(N)

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
        node_features = torch.cat([noisy_fmri.unsqueeze(-1), coords], dim=-1)  # [B*N, 4]

        # Input projection
        h = self.input_proj(node_features)  # [B*N, hidden_channels[0]]

        # Encoder with cross-attention
        h1 = self.enc1(h, edge_index, cond, batch)  # [B*N, hidden_channels[0]]

        h2 = self.enc2(h1, edge_index, cond, batch)  # [B*N, hidden_channels[1]]
        h2 = self.enc2_attn(h2, v_emb, batch)

        h3 = self.enc3(h2, edge_index, cond, batch)  # [B*N, hidden_channels[2]]
        h3 = self.enc3_attn(h3, v_emb, batch)

        # Middle with cross-attention
        h = self.mid1(h3, edge_index, cond, batch)
        h = self.mid_attn(h, v_emb, batch)
        h = self.mid2(h, edge_index, cond, batch)

        # Decoder with skip connections
        h = torch.cat([h, h3], dim=-1)  # Skip connection
        h = self.dec3(h, edge_index, cond, batch)
        h = self.dec3_attn(h, v_emb, batch)

        h = torch.cat([h, h2], dim=-1)  # Skip connection
        h = self.dec2(h, edge_index, cond, batch)
        h = self.dec2_attn(h, v_emb, batch)

        h = self.dec1(h, edge_index, cond, batch)

        # Output projection
        pred_noise = self.output_proj(h).squeeze(-1)  # [B*N]

        # Reshape back to [B, N] if needed
        pred_noise = pred_noise.view(B, N)

        return pred_noise


class DiffusionManager:
    """
    Diffusion process manager (DDPM/DDIM).
    Handles noise schedule, forward/reverse process.

    Same as DiffusionARM3D - works with any representation.
    """
    def __init__(self, timesteps=1000, beta_schedule='cosine'):
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule

        # Create noise schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(0.0001, 0.02, timesteps)
        elif beta_schedule == 'cosine':
            self.betas = self.cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute values for diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def to(self, device):
        """Move all precomputed tensors to device"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        """Cosine schedule from Improved DDPM"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: add noise to x_start at timestep t
        x_start: [B, N] clean fMRI
        t: [B] timestep indices
        noise: [B, N] optional pre-generated noise

        Returns: [B, N] noisy fMRI
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Ensure tensors are on same device for indexing
        device = t.device
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(device)[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(device)[t].view(-1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, x_t, coords, edge_index, vis_feat, t, batch=None, cfg_scale=1.0):
        """
        Reverse diffusion: denoise x_t at timestep t
        Uses Classifier-Free Guidance if cfg_scale > 1.0
        """
        B = x_t.shape[0]

        if cfg_scale > 1.0:
            # CFG: predict with and without conditioning
            noise_cond = model(x_t, coords, edge_index, vis_feat, t, batch, use_cfg_dropout=False)

            # Unconditional prediction (null embedding)
            null_vis_feat = model.module.null_vis_emb.expand(B, -1) if hasattr(model, 'module') else model.null_vis_emb.expand(B, -1)
            noise_uncond = model(x_t, coords, edge_index, null_vis_feat, t, batch, use_cfg_dropout=False)

            # Guided prediction
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = model(x_t, coords, edge_index, vis_feat, t, batch, use_cfg_dropout=False)

        # Compute x_{t-1}
        betas_t = self.betas[t].view(-1, 1).to(x_t.device)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1).to(x_t.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1).to(x_t.device)

        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

        if t[0] > 0:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1).to(x_t.device)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return model_mean

    @torch.no_grad()
    def sample(self, model, coords, edge_index, vis_feat, batch=None, cfg_scale=4.0, device='cuda'):
        """
        Generate fMRI from visual features using DDPM sampling.

        coords: [N, 3] or [B, N, 3] spatial coordinates
        edge_index: [2, E] graph edges
        vis_feat: [B, vis_dim] visual features
        batch: [N] batch assignment
        cfg_scale: CFG guidance scale

        Returns: [B, N] generated fMRI
        """
        B = vis_feat.shape[0]
        if coords.dim() == 2:
            N = coords.shape[0] // B
        else:
            N = coords.shape[1]

        # Start from pure noise
        x = torch.randn(B, N, device=device)

        # Iterative denoising
        for i in reversed(range(self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, coords, edge_index, vis_feat, t, batch, cfg_scale)

        return x

    @torch.no_grad()
    def ddim_sample(self, model, coords, edge_index, vis_feat, ddim_steps=50, batch=None, cfg_scale=4.0, device='cuda'):
        """
        Generate fMRI using DDIM sampling (faster).

        ddim_steps: Number of sampling steps (< timesteps for speedup)
        """
        B = vis_feat.shape[0]
        if coords.dim() == 2:
            N = coords.shape[0] // B
        else:
            N = coords.shape[1]

        # Create DDIM timestep sequence
        c = self.timesteps // ddim_steps
        ddim_timesteps = torch.arange(0, self.timesteps, c, device=device)

        # Start from noise
        x = torch.randn(B, N, device=device)

        # DDIM sampling
        for i in reversed(range(len(ddim_timesteps))):
            t = torch.full((B,), ddim_timesteps[i], device=device, dtype=torch.long)

            # Predict noise
            if cfg_scale > 1.0:
                noise_cond = model(x, coords, edge_index, vis_feat, t, batch, use_cfg_dropout=False)
                null_vis_feat = model.module.null_vis_emb.expand(B, -1) if hasattr(model, 'module') else model.null_vis_emb.expand(B, -1)
                noise_uncond = model(x, coords, edge_index, null_vis_feat, t, batch, use_cfg_dropout=False)
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = model(x, coords, edge_index, vis_feat, t, batch, use_cfg_dropout=False)

            # DDIM update
            alpha_t = self.alphas_cumprod[t].view(-1, 1).to(x.device)
            if i > 0:
                alpha_t_prev = self.alphas_cumprod[ddim_timesteps[i - 1]].view(-1, 1).to(x.device)
            else:
                alpha_t_prev = torch.ones_like(alpha_t)

            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev) * noise_pred

            # DDIM update
            x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt

        return x


class DiffusionGraphARM(nn.Module):
    """
    Complete Diffusion Graph ARM model.
    Combines GraphUNet with DiffusionManager.
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
        timesteps=1000,
        beta_schedule='cosine',
    ):
        super().__init__()
        self.output_dim = output_dim

        # Graph U-Net
        self.unet = GraphUNet(
            time_dim=time_dim,
            vis_dim=input_dim,
            hidden_channels=[hidden_dim // 2, hidden_dim, hidden_dim * 2],
            num_heads=num_heads,
            gnn_type=gnn_type,
            dropout=dropout,
            cfg_dropout=cfg_dropout,
        )

        # Diffusion manager
        self.diffusion = DiffusionManager(timesteps=timesteps, beta_schedule=beta_schedule)

        # Mean fMRI baseline (to be loaded from training data)
        self.register_buffer('mean_fmri', torch.zeros(output_dim))

    def forward(self, clean_fmri, coords, edge_index, vis_feat, batch=None):
        """
        Training forward pass.

        clean_fmri: [B, N] clean fMRI
        coords: [B, N, 3] or [N, 3] spatial coordinates
        edge_index: [2, E] graph edges
        vis_feat: [B, 768] visual features
        batch: [N] batch assignment

        Returns: dict with 'loss', 'pred_noise', 'noise', 't'
        """
        B = clean_fmri.shape[0]
        device = clean_fmri.device

        # Sample timesteps
        t = torch.randint(0, self.diffusion.timesteps, (B,), device=device).long()

        # Add noise
        noise = torch.randn_like(clean_fmri)
        noisy_fmri = self.diffusion.q_sample(clean_fmri, t, noise)

        # Predict noise
        pred_noise = self.unet(noisy_fmri, coords, edge_index, vis_feat, t, batch, use_cfg_dropout=True)

        return {
            'pred_noise': pred_noise,
            'noise': noise,
            't': t,
        }

    @torch.no_grad()
    def generate(self, coords, edge_index, vis_feat, batch=None, method='ddim', ddim_steps=25, cfg_scale=4.0):
        """
        Generate fMRI from visual features.

        method: 'ddpm' or 'ddim'
        ddim_steps: Number of DDIM steps (ignored for DDPM)
        cfg_scale: Classifier-free guidance scale

        Returns: [B, N] generated fMRI
        """
        device = vis_feat.device

        if method == 'ddim':
            x = self.diffusion.ddim_sample(
                self.unet, coords, edge_index, vis_feat,
                ddim_steps=ddim_steps, batch=batch, cfg_scale=cfg_scale, device=device
            )
        else:
            x = self.diffusion.sample(
                self.unet, coords, edge_index, vis_feat,
                batch=batch, cfg_scale=cfg_scale, device=device
            )

        # Add mean back
        x = x + self.mean_fmri.unsqueeze(0)

        return x


# Check if torch_geometric is available
if not TORCH_GEOMETRIC_AVAILABLE:
    raise ImportError(
        "This module requires torch_geometric. Install with:\n"
        "pip install torch-geometric torch-scatter torch-sparse"
    )

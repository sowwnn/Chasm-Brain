"""
Improved 3D Diffusion Model for Image-to-fMRI Generation
Combines Diffusion process with 3D CNN for better spatial modeling

Key improvements:
- Classifier-Free Guidance (CFG) for stronger visual conditioning
- Multi-scale cross-attention at encoder/decoder levels
- AdaLN-Zero for better conditioning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEmb(nn.Module):
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


class Reshape3D(nn.Module):
    """Reshape 1D fMRI to 3D volume"""
    def __init__(self, shape_3d=(26, 26, 24)):
        super().__init__()
        self.shape_3d = shape_3d
        self.total_size = shape_3d[0] * shape_3d[1] * shape_3d[2]
        
    def forward(self, x):
        """x: [B, 15724] -> [B, 1, 26, 26, 24]"""
        batch_size = x.shape[0]
        # Pad to 16224
        padded = torch.zeros(batch_size, self.total_size, device=x.device, dtype=x.dtype)
        padded[:, :15724] = x
        # Reshape to 3D
        return padded.view(batch_size, 1, *self.shape_3d)
    
    def inverse(self, x):
        """x: [B, C, 26, 26, 24] -> [B, 15724]"""
        batch_size = x.shape[0]
        # Flatten
        flat = x.view(batch_size, -1)
        # Unpad
        return flat[:, :15724]


class Conv3DBlock(nn.Module):
    """3D Convolution block with GroupNorm and SiLU"""
    def __init__(self, in_channels, out_channels, time_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # Time embedding projection
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
        
        # FiLM conditioning
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb.view(time_emb.shape[0], -1, 1, 1, 1)
        scale, shift = time_emb.chunk(2, dim=1)
        
        h = h * (1 + scale) + shift
        h = self.act(h)
        return h


class ResBlock3D(nn.Module):
    """3D Residual block with time conditioning"""
    def __init__(self, channels, time_dim, dropout=0.1):
        super().__init__()
        self.block1 = Conv3DBlock(channels, channels, time_dim)
        self.block2 = Conv3DBlock(channels, channels, time_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, time_emb):
        h = self.block1(x, time_emb)
        h = self.dropout(h)
        h = self.block2(h, time_emb)
        return x + h


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Norm Zero - Used in DiT for better conditioning
    Learns to modulate features based on conditioning signal
    """
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim)  # scale, shift, gate for two sublayers
        )
        # Initialize to zero for residual learning
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, cond):
        """
        x: [B, N, dim] or [B, dim]
        cond: [B, cond_dim]
        Returns: shift1, scale1, gate1, shift2, scale2, gate2
        """
        return self.adaLN_modulation(cond).chunk(6, dim=-1)


class CrossAttention3D(nn.Module):
    """Cross-attention between 3D fMRI and visual features"""
    def __init__(self, fmri_dim, vis_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = fmri_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(fmri_dim, fmri_dim)
        self.to_k = nn.Linear(vis_dim, fmri_dim)
        self.to_v = nn.Linear(vis_dim, fmri_dim)
        self.to_out = nn.Linear(fmri_dim, fmri_dim)

        # Layer norm for better stability
        self.norm_q = nn.LayerNorm(fmri_dim)
        self.norm_kv = nn.LayerNorm(vis_dim)

    def forward(self, fmri_feat, vis_feat):
        """
        fmri_feat: [B, fmri_dim]
        vis_feat: [B, vis_dim]
        """
        B = fmri_feat.shape[0]

        # Normalize inputs
        fmri_feat = self.norm_q(fmri_feat)
        vis_feat = self.norm_kv(vis_feat)

        q = self.to_q(fmri_feat).view(B, self.num_heads, -1)
        k = self.to_k(vis_feat).view(B, self.num_heads, -1)
        v = self.to_v(vis_feat).view(B, self.num_heads, -1)

        # Compute attention scores: Q @ K^T
        attn = torch.einsum('bhd,bhd->bh', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.einsum('bh,bhd->bhd', attn, v)
        out = out.reshape(B, -1)
        out = self.to_out(out)

        return out


class SpatialCrossAttention3D(nn.Module):
    """
    Spatial cross-attention for 3D feature maps
    Allows each spatial location to attend to visual features
    """
    def __init__(self, channels, vis_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(8, channels)
        self.to_q = nn.Conv3d(channels, channels, 1)
        self.to_k = nn.Linear(vis_dim, channels)
        self.to_v = nn.Linear(vis_dim, channels)
        self.to_out = nn.Conv3d(channels, channels, 1)

        # Zero-initialize output for residual learning
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x, vis_feat):
        """
        x: [B, C, D, H, W]
        vis_feat: [B, vis_dim]
        """
        B, C, D, H, W = x.shape
        residual = x

        x = self.norm(x)

        # Query from spatial features
        q = self.to_q(x)  # [B, C, D, H, W]
        q = q.view(B, self.num_heads, self.head_dim, -1)  # [B, heads, head_dim, D*H*W]
        q = q.permute(0, 1, 3, 2)  # [B, heads, D*H*W, head_dim]

        # Key and Value from visual features
        k = self.to_k(vis_feat).view(B, self.num_heads, self.head_dim, 1)  # [B, heads, head_dim, 1]
        v = self.to_v(vis_feat).view(B, self.num_heads, self.head_dim, 1)  # [B, heads, head_dim, 1]

        # Attention: [B, heads, D*H*W, 1]
        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention: [B, heads, D*H*W, head_dim]
        out = torch.matmul(attn, v.permute(0, 1, 3, 2))
        out = out.permute(0, 1, 3, 2)  # [B, heads, head_dim, D*H*W]
        out = out.reshape(B, C, D, H, W)
        out = self.to_out(out)

        return residual + out


class UNet3D(nn.Module):
    """
    3D U-Net for diffusion process with enhanced visual conditioning

    Improvements:
    - Multi-scale cross-attention at encoder, middle, and decoder
    - Stronger visual feature projection
    - CFG dropout support
    """
    def __init__(self, time_dim=256, vis_dim=768, base_channels=32, cfg_dropout=0.1):
        super().__init__()
        self.time_dim = time_dim
        self.cfg_dropout = cfg_dropout

        # Time embedding (deeper MLP)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # Visual feature projection (stronger, deeper)
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, time_dim * 2),
            nn.LayerNorm(time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim * 2),
            nn.LayerNorm(time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )

        # Learnable null embedding for CFG (unconditional)
        self.null_vis_emb = nn.Parameter(torch.randn(1, time_dim) * 0.02)

        # Encoder
        self.enc1 = Conv3DBlock(1, base_channels, time_dim)
        self.enc2 = Conv3DBlock(base_channels, base_channels * 2, time_dim)
        self.enc3 = Conv3DBlock(base_channels * 2, base_channels * 4, time_dim)

        # Multi-scale cross-attention in encoder
        self.enc2_attn = SpatialCrossAttention3D(base_channels * 2, time_dim, num_heads=4)
        self.enc3_attn = SpatialCrossAttention3D(base_channels * 4, time_dim, num_heads=4)

        # Middle
        self.mid1 = ResBlock3D(base_channels * 4, time_dim)
        self.mid_attn = CrossAttention3D(base_channels * 4, time_dim, num_heads=8)
        self.mid_spatial_attn = SpatialCrossAttention3D(base_channels * 4, time_dim, num_heads=8)
        self.mid2 = ResBlock3D(base_channels * 4, time_dim)

        # Decoder (input channels = h_channels + skip_channels)
        self.dec3 = Conv3DBlock(base_channels * 6, base_channels * 2, time_dim)
        self.dec2 = Conv3DBlock(base_channels * 3, base_channels, time_dim)
        self.dec1 = Conv3DBlock(base_channels, base_channels, time_dim)

        # Multi-scale cross-attention in decoder
        self.dec3_attn = SpatialCrossAttention3D(base_channels * 2, time_dim, num_heads=4)
        self.dec2_attn = SpatialCrossAttention3D(base_channels, time_dim, num_heads=4)

        # Output
        self.out = nn.Conv3d(base_channels, 1, 3, padding=1)

        self.pool = nn.MaxPool3d(2)

    def forward(self, x, t, vis_feat, use_cfg_dropout=True):
        """
        x: [B, 1, 26, 26, 24] noisy fMRI
        t: [B] timestep
        vis_feat: [B, time_dim] projected visual features (or [B, 768] raw)
        use_cfg_dropout: bool, whether to apply CFG dropout during training
        """
        B = x.shape[0]

        # Time embedding
        t_emb = self.time_mlp(t)

        # Visual embedding with CFG dropout
        if vis_feat.shape[-1] != self.time_dim:
            v_emb = self.vis_proj(vis_feat)
        else:
            v_emb = vis_feat

        # Apply CFG dropout during training
        if self.training and use_cfg_dropout and self.cfg_dropout > 0:
            # Randomly drop visual conditioning for some samples
            mask = torch.rand(B, 1, device=x.device) > self.cfg_dropout
            null_emb = self.null_vis_emb.expand(B, -1)
            v_emb = torch.where(mask, v_emb, null_emb)

        # Combined conditioning
        cond = t_emb + v_emb

        # Encoder with cross-attention
        h1 = self.enc1(x, cond)

        h2 = self.enc2(self.pool(h1), cond)
        h2 = self.enc2_attn(h2, v_emb)  # Cross-attention with visual

        h3 = self.enc3(self.pool(h2), cond)
        h3 = self.enc3_attn(h3, v_emb)  # Cross-attention with visual

        # Middle with enhanced cross-attention
        h = self.mid1(h3, cond)

        # Global cross-attention
        B, C, D, H, W = h.shape
        h_flat = h.mean(dim=[2, 3, 4])  # [B, C]
        h_attn = self.mid_attn(h_flat, v_emb)
        h_attn = h_attn.view(B, C, 1, 1, 1).expand_as(h)
        h = h + h_attn

        # Spatial cross-attention
        h = self.mid_spatial_attn(h, v_emb)

        h = self.mid2(h, cond)

        # Decoder with skip connections and cross-attention
        h = F.interpolate(h, size=h2.shape[2:], mode='trilinear', align_corners=True)
        h = torch.cat([h, h2], dim=1)
        h = self.dec3(h, cond)
        h = self.dec3_attn(h, v_emb)  # Cross-attention

        h = F.interpolate(h, size=h1.shape[2:], mode='trilinear', align_corners=True)
        h = torch.cat([h, h1], dim=1)
        h = self.dec2(h, cond)
        h = self.dec2_attn(h, v_emb)  # Cross-attention

        h = self.dec1(h, cond)
        h = self.out(h)

        return h

    def forward_with_cfg(self, x, t, vis_feat, cfg_scale=1.0):
        """
        Forward pass with Classifier-Free Guidance for inference

        cfg_scale: guidance scale (1.0 = no guidance, >1.0 = stronger conditioning)
        """
        B = x.shape[0]

        # Project visual features
        if vis_feat.shape[-1] != self.time_dim:
            v_emb = self.vis_proj(vis_feat)
        else:
            v_emb = vis_feat

        if cfg_scale == 1.0:
            # No CFG, just conditional prediction
            return self.forward(x, t, v_emb, use_cfg_dropout=False)

        # CFG: compute both conditional and unconditional predictions
        # Conditional
        t_emb = self.time_mlp(t)
        cond = t_emb + v_emb
        pred_cond = self._forward_core(x, cond, v_emb)

        # Unconditional (use null embedding)
        null_emb = self.null_vis_emb.expand(B, -1)
        uncond = t_emb + null_emb
        pred_uncond = self._forward_core(x, uncond, null_emb)

        # CFG formula: pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)

    def _forward_core(self, x, cond, v_emb):
        """Core forward pass used by forward_with_cfg"""
        # Encoder with cross-attention
        h1 = self.enc1(x, cond)

        h2 = self.enc2(self.pool(h1), cond)
        h2 = self.enc2_attn(h2, v_emb)

        h3 = self.enc3(self.pool(h2), cond)
        h3 = self.enc3_attn(h3, v_emb)

        # Middle
        h = self.mid1(h3, cond)

        B, C, D, H, W = h.shape
        h_flat = h.mean(dim=[2, 3, 4])
        h_attn = self.mid_attn(h_flat, v_emb)
        h_attn = h_attn.view(B, C, 1, 1, 1).expand_as(h)
        h = h + h_attn
        h = self.mid_spatial_attn(h, v_emb)
        h = self.mid2(h, cond)

        # Decoder
        h = F.interpolate(h, size=h2.shape[2:], mode='trilinear', align_corners=True)
        h = torch.cat([h, h2], dim=1)
        h = self.dec3(h, cond)
        h = self.dec3_attn(h, v_emb)

        h = F.interpolate(h, size=h1.shape[2:], mode='trilinear', align_corners=True)
        h = torch.cat([h, h1], dim=1)
        h = self.dec2(h, cond)
        h = self.dec2_attn(h, v_emb)

        h = self.dec1(h, cond)
        return self.out(h)


class DiffusionARM3D(nn.Module):
    """
    3D Diffusion Model for Image-to-fMRI Generation

    Architecture:
        Visual Features (768) -> Conditioning
        Noisy fMRI (15724) -> 3D (26x26x24) -> U-Net 3D -> Denoised 3D -> 1D (15724)

    Features:
        - Classifier-Free Guidance (CFG) for stronger visual conditioning
        - Multi-scale cross-attention
    """
    def __init__(self, vis_dim=768, fmri_dim=15724, time_dim=256, base_channels=32,
                 cfg_dropout=0.1, **kwargs):
        super().__init__()
        self.fmri_dim = fmri_dim
        self.time_dim = time_dim
        self.reshape_3d = Reshape3D()

        # 3D U-Net with CFG support
        self.unet = UNet3D(time_dim, vis_dim, base_channels, cfg_dropout)

        # 1D refinement network (bottleneck design to reduce params)
        # Uses smaller hidden dim to avoid parameter explosion
        refiner_hidden = 1024  # Much smaller than fmri_dim * 2
        self.refiner = nn.Sequential(
            nn.Linear(fmri_dim, refiner_hidden),
            nn.LayerNorm(refiner_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(refiner_hidden, refiner_hidden),
            nn.LayerNorm(refiner_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(refiner_hidden, fmri_dim)
        )

    def forward(self, vis_feat, noisy_fmri, t, mean_fmri=None, return_3d=False,
                use_cfg_dropout=True):
        """
        vis_feat: [B, 768] visual features
        noisy_fmri: [B, 15724] noisy fMRI at timestep t
        t: [B] timestep
        mean_fmri: [B, 15724] baseline (optional)
        return_3d: bool, if True return 3D representation for training loss
        use_cfg_dropout: bool, whether to apply CFG dropout

        Returns:
            If return_3d=False: pred_x0: [B, 15724] predicted clean fMRI (1D)
            If return_3d=True: pred_3d for computing 3D loss
        """
        # Reshape to 3D
        noisy_3d = self.reshape_3d(noisy_fmri)

        # Predict in 3D space
        pred_3d = self.unet(noisy_3d, t, vis_feat, use_cfg_dropout=use_cfg_dropout)

        # If return_3d is True, return 3D representation for training
        if return_3d:
            return pred_3d

        # Reshape back to 1D
        pred_1d = self.reshape_3d.inverse(pred_3d)

        # Optional refinement
        pred_1d = self.refiner(pred_1d)

        # Add mean back if provided
        if mean_fmri is not None:
            pred_1d = pred_1d + mean_fmri

        return pred_1d

    def forward_with_cfg(self, vis_feat, noisy_fmri, t, mean_fmri=None, cfg_scale=3.0):
        """
        Forward pass with Classifier-Free Guidance for inference

        cfg_scale: guidance scale (1.0 = no guidance, >1.0 = stronger conditioning)
                   Typical values: 2.0 - 7.0
        """
        # Reshape to 3D
        noisy_3d = self.reshape_3d(noisy_fmri)

        # Predict with CFG in 3D space
        pred_3d = self.unet.forward_with_cfg(noisy_3d, t, vis_feat, cfg_scale)

        # Reshape back to 1D
        pred_1d = self.reshape_3d.inverse(pred_3d)

        # Refinement
        pred_1d = self.refiner(pred_1d)

        # Add mean back if provided
        if mean_fmri is not None:
            pred_1d = pred_1d + mean_fmri

        return pred_1d


class DiffusionManager3D:
    """
    Manager for training and sampling with CFG support

    Features:
    - Classifier-Free Guidance (CFG) for inference
    - DDIM fast sampling
    - Cosine noise schedule
    """
    def __init__(self, model, timesteps=50, inference_timesteps=10, device='cuda'):
        self.model = model
        self.timesteps = timesteps
        self.inference_timesteps = inference_timesteps
        self.device = device

        # Cosine schedule
        betas = self.cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor.to(self.device))

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(self, x_start, t):
        """Add noise to clean signal"""
        noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise

    def get_train_tuple(self, x_0, vis_feat, mean_fmri):
        """Prepare training data"""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)

        # Subtract mean
        x_0_centered = x_0 - mean_fmri

        # Add noise
        noisy_x, noise = self.add_noise(x_0_centered, t)

        return noisy_x, t, x_0_centered

    @torch.no_grad()
    def sample_ddim(self, vis_feat, mean_fmri, clip_value=5.0, eta=0.0, cfg_scale=1.0):
        """
        DDIM sampling with optional Classifier-Free Guidance

        Args:
            vis_feat: [B, 768] visual features
            mean_fmri: [B, 15724] baseline fMRI
            clip_value: clipping value for predictions
            eta: DDIM stochasticity (0=deterministic)
            cfg_scale: CFG guidance scale (1.0=no guidance, >1.0=stronger visual conditioning)
                       Recommended: 2.0-5.0 for good results
        """
        batch_size = vis_feat.shape[0]

        # Start from noise
        x_t = torch.randn(batch_size, self.model.fmri_dim, device=self.device)

        # DDIM timesteps
        timesteps = torch.linspace(
            self.timesteps - 1, 0, self.inference_timesteps,
            dtype=torch.long, device=self.device
        )

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Predict x_0 with optional CFG
            if cfg_scale != 1.0:
                pred_x0 = self.model.forward_with_cfg(
                    vis_feat, x_t, t_batch, mean_fmri=None, cfg_scale=cfg_scale
                )
            else:
                pred_x0 = self.model(
                    vis_feat, x_t, t_batch, mean_fmri=None, use_cfg_dropout=False
                )

            pred_x0 = torch.clamp(pred_x0, -clip_value, clip_value)

            if i < len(timesteps) - 1:
                # DDIM update
                alpha_t = self.alphas_cumprod[t]
                alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]]

                # Predict noise from x_0 prediction
                pred_noise = (x_t - torch.sqrt(alpha_t) * pred_x0) / (torch.sqrt(1 - alpha_t) + 1e-8)

                # DDIM formula with optional stochasticity
                if eta > 0:
                    sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                    noise = torch.randn_like(x_t)
                    x_t = (
                        torch.sqrt(alpha_t_prev) * pred_x0 +
                        torch.sqrt(1 - alpha_t_prev - sigma**2) * pred_noise +
                        sigma * noise
                    )
                else:
                    x_t = (
                        torch.sqrt(alpha_t_prev) * pred_x0 +
                        torch.sqrt(1 - alpha_t_prev) * pred_noise
                    )
            else:
                x_t = pred_x0

        # Add mean back
        return x_t + mean_fmri

    @torch.no_grad()
    def sample_ddim_with_intermediate(self, vis_feat, mean_fmri, clip_value=5.0,
                                       cfg_scale=3.0, return_all=False):
        """
        DDIM sampling with CFG, optionally returning intermediate steps

        Useful for debugging and visualization
        """
        batch_size = vis_feat.shape[0]
        intermediates = []

        x_t = torch.randn(batch_size, self.model.fmri_dim, device=self.device)

        timesteps = torch.linspace(
            self.timesteps - 1, 0, self.inference_timesteps,
            dtype=torch.long, device=self.device
        )

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            if cfg_scale != 1.0:
                pred_x0 = self.model.forward_with_cfg(
                    vis_feat, x_t, t_batch, mean_fmri=None, cfg_scale=cfg_scale
                )
            else:
                pred_x0 = self.model(vis_feat, x_t, t_batch, use_cfg_dropout=False)

            pred_x0 = torch.clamp(pred_x0, -clip_value, clip_value)

            if return_all:
                intermediates.append(pred_x0.clone() + mean_fmri)

            if i < len(timesteps) - 1:
                alpha_t = self.alphas_cumprod[t]
                alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]]
                pred_noise = (x_t - torch.sqrt(alpha_t) * pred_x0) / (torch.sqrt(1 - alpha_t) + 1e-8)
                x_t = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * pred_noise
            else:
                x_t = pred_x0

        final = x_t + mean_fmri

        if return_all:
            return final, intermediates
        return final

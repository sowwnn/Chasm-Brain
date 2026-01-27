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

class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention block for attending to conditioning signals (visual or mean_fmri).
    Uses multi-head attention to capture rich interactions between queries and context.
    """
    def __init__(self, dim, context_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Layer norms
        self.norm_x = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)

        # Q, K, V projections
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        """
        Args:
            x: [B, dim] query features (current fMRI representation)
            context: [B, context_dim] or [B, seq_len, context_dim] key/value features
        Returns:
            [B, dim] attended features
        """
        # Normalize inputs
        x_norm = self.norm_x(x)

        # Handle both 2D and 3D context
        if context.dim() == 2:
            context = context.unsqueeze(1)  # [B, 1, context_dim]
        context_norm = self.norm_context(context)

        B, seq_len, _ = context_norm.shape

        # Compute Q, K, V
        q = self.to_q(x_norm).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, 1, head_dim]
        k = self.to_k(context_norm).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, seq_len, head_dim]
        v = self.to_v(context_norm).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, seq_len, head_dim]

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, 1, seq_len]
        attn = F.softmax(attn, dim=-1)

        # Attend to values
        out = torch.matmul(attn, v)  # [B, num_heads, 1, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, -1)  # [B, dim]

        # Output projection
        out = self.to_out(out)

        # Residual connection
        return x + out


class ResBlock(nn.Module):
    """
    Residual block with FiLM (Feature-wise Linear Modulation) conditioning.
    Injects time embedding via scale and shift parameters.
    """
    def __init__(self, dim, cond_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # FiLM conditioning: project condition to scale and shift
        self.cond_proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, condition):
        """
        Args:
            x: [B, dim] hidden state
            condition: [B, cond_dim] conditioning vector (usually time embedding)
        Returns:
            [B, dim] residual output
        """
        # Get scale and shift from condition
        scale, shift = self.cond_proj(condition).chunk(2, dim=-1)

        # First block with FiLM
        h = self.norm1(x)
        h = h * (1 + scale) + shift  # FiLM modulation
        h = F.gelu(h)
        h = self.linear1(h)
        h = self.dropout(h)

        # Second block
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.linear2(h)
        h = self.dropout(h)

        return x + h
    

class DiffusionARM(nn.Module):
    """
    Diffusion-based ARM model for fMRI prediction with Cross-Attention.
    Uses DDPM with x_0 prediction, FiLM time conditioning, and Cross-Attention for visual/mean_fmri.

    Architecture (per layer):
    1. ResBlock with FiLM (time conditioning)
    2. Cross-Attention to visual features
    3. Cross-Attention to mean_fmri (baseline guidance)

    This allows the model to:
    - Attend to relevant visual features
    - Use mean_fmri as a denoising guide/anchor
    - Learn complex visual-fMRI relationships
    """
    def __init__(self, input_dim=768, fmri_dim=1000, hidden_dim=2048,
                 output_dim=None, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        if output_dim is None:
            output_dim = fmri_dim
        self.fmri_dim = fmri_dim
        self.num_layers = num_layers

        # Time embedding (sinusoidal + MLP)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )

        # Deep visual feature projector
        self.vis_projector = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.GELU()
        )

        # Mean fMRI projector (for attention)
        self.mean_projector = nn.Sequential(
            nn.Linear(fmri_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Input projection (noisy fMRI)
        self.input_proj = nn.Linear(fmri_dim, hidden_dim)

        # Interleaved layers: ResBlock → CrossAttn(visual) → CrossAttn(mean_fmri)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'res_block': ResBlock(hidden_dim, 512, dropout=dropout),  # time conditioning
                'cross_attn_visual': CrossAttentionBlock(hidden_dim, 768, num_heads, dropout),
                'cross_attn_mean': CrossAttentionBlock(hidden_dim, hidden_dim, num_heads, dropout)
            }))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, vis_feat, mean_fmri, noisy_fmri=None, t=None):
        """
        Forward pass with Cross-Attention to visual and mean_fmri.

        Args:
            vis_feat: [B, input_dim] visual features
            mean_fmri: [B, fmri_dim] or [1, fmri_dim] baseline fMRI (used for attention guidance)
            noisy_fmri: [B, fmri_dim] noisy input (required during training/inference)
            t: [B] timestep (required during training/inference)
        Returns:
            [B, fmri_dim] predicted clean signal (x_0)
        """
        if noisy_fmri is None or t is None:
            raise ValueError("DiffusionARM requires noisy_fmri and t during forward pass. Use DiffusionManager.sample() for inference.")

        batch_size = noisy_fmri.shape[0]

        # Expand mean_fmri to batch size if needed
        if mean_fmri.shape[0] == 1:
            mean_fmri = mean_fmri.expand(batch_size, -1)

        # Embed time, visual, and mean_fmri
        t_emb = self.time_mlp(t.float())  # [B, 512]
        v_emb = self.vis_projector(vis_feat)  # [B, 768]
        m_emb = self.mean_projector(mean_fmri)  # [B, hidden_dim]

        # Project noisy input to hidden dimension
        h = self.input_proj(noisy_fmri)  # [B, hidden_dim]

        # Apply interleaved layers: ResBlock → CrossAttn(visual) → CrossAttn(mean)
        for layer in self.layers:
            # 1. Time conditioning via FiLM
            h = layer['res_block'](h, t_emb)

            # 2. Attend to visual features
            h = layer['cross_attn_visual'](h, v_emb)

            # 3. Attend to mean_fmri (baseline guidance)
            h = layer['cross_attn_mean'](h, m_emb)

        # Output projection (predict clean signal)
        x_0_pred = self.output_proj(h)

        return x_0_pred

class DiffusionManager:
    """
    Manager for DDPM training and sampling with x_0 prediction.
    Uses cosine beta schedule for stable diffusion process.
    """
    def __init__(self, model, timesteps=10, inference_timesteps=None, device='cuda'):
        self.model = model
        self.timesteps = timesteps
        self.inference_timesteps = inference_timesteps if inference_timesteps is not None else timesteps
        self.device = device

        # Cosine schedule
        self.betas = self.cosine_beta_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in Improved DDPM."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9)

    def get_train_tuple(self, x_0, vis_feat):
        """
        Prepare training data for diffusion.

        Args:
            x_0: [B, N] clean fMRI (target - mean)
            vis_feat: [B, input_dim] visual features

        Returns:
            noisy_x: [B, N] noisy input at time t
            t: [B] random timesteps
            x_0: [B, N] clean target (for x_0 prediction)
        """
        batch_size = x_0.shape[0]

        # Random timestep for each sample
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

        # Add noise to x_0
        noisy_x, noise = self.add_noise(x_0, t)

        return noisy_x, t, x_0

    def add_noise(self, x_start, t):
        """Add noise to clean signal according to forward diffusion process."""
        noise = torch.randn_like(x_start)
        noisy_x = (self.sqrt_alphas_cumprod[t][:, None] * x_start +
                   self.sqrt_one_minus_alphas_cumprod[t][:, None] * noise)
        return noisy_x, noise

    @torch.no_grad()
    def sample(self, vis_feat, mean_fmri, clip_value=5.0):
        """
        DDPM sampling with x_0 prediction.

        Args:
            vis_feat: [B, input_dim] visual features
            mean_fmri: [B, fmri_dim] or [1, fmri_dim] baseline

        Returns:
            [B, fmri_dim] sampled fMRI signals
        """
        batch_size = vis_feat.shape[0]

        # Expand mean_fmri to batch size if needed
        if mean_fmri.shape[0] == 1:
            mean_fmri = mean_fmri.expand(batch_size, -1)

        # Start from pure noise
        x = torch.randn(batch_size, self.model.fmri_dim, device=self.device)

        # Reverse diffusion process
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)

            # Predict x_0 directly
            x_0_pred = self.model(vis_feat, mean_fmri, noisy_fmri=x, t=t)

            # Clip x_0 to maintain stability
            x_0_pred = torch.clamp(x_0_pred, -clip_value, clip_value)

            # DDPM posterior sampling
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alphas_cumprod[i]

            if i > 0:
                alpha_bar_prev = self.alphas_cumprod[i-1]
                beta_t = self.betas[i]
                sigma_t = torch.sqrt(beta_t * (1. - alpha_bar_prev) / (1. - alpha_bar_t))

                # Compute mean of posterior q(x_{t-1} | x_t, x_0)
                mu_t = (torch.sqrt(alpha_bar_prev) * beta_t / (1. - alpha_bar_t) * x_0_pred +
                        torch.sqrt(alpha_t) * (1. - alpha_bar_prev) / (1. - alpha_bar_t) * x)

                # Add noise
                x = mu_t + sigma_t * torch.randn_like(x)
            else:
                # Final step: x_0 = x_0_pred
                x = x_0_pred

        # Add back mean
        return x + mean_fmri

    @torch.no_grad()
    def sample_ddim(self, vis_feat, mean_fmri, clip_value=5.0, eta=0.0):
        """
        DDIM sampling with x_0 prediction - faster inference with fewer steps.

        Args:
            vis_feat: [B, input_dim] visual features
            mean_fmri: [B, fmri_dim] or [1, fmri_dim] baseline
            clip_value: clipping value for x_0 prediction
            eta: stochasticity parameter (0 = deterministic DDIM, 1 = DDPM)

        Returns:
            [B, fmri_dim] sampled fMRI signals
        """
        batch_size = vis_feat.shape[0]

        # Expand mean_fmri to batch size if needed
        if mean_fmri.shape[0] == 1:
            mean_fmri = mean_fmri.expand(batch_size, -1)

        # Start from pure noise
        x = torch.randn(batch_size, self.model.fmri_dim, device=self.device)

        # Create timestep sequence for inference
        # Uniformly sample inference_timesteps from the full timesteps
        step_size = self.timesteps // self.inference_timesteps
        timestep_seq = list(range(0, self.timesteps, step_size))[:self.inference_timesteps]
        timestep_seq = list(reversed(timestep_seq))

        # DDIM reverse process
        for i, timestep in enumerate(timestep_seq):
            t = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)

            # Predict x_0 directly
            x_0_pred = self.model(vis_feat, mean_fmri, noisy_fmri=x, t=t)

            # Clip x_0 to maintain stability
            x_0_pred = torch.clamp(x_0_pred, -clip_value, clip_value)

            # Get alpha values
            alpha_bar_t = self.alphas_cumprod[timestep]

            if i < len(timestep_seq) - 1:
                # Not the final step
                timestep_prev = timestep_seq[i + 1]
                alpha_bar_prev = self.alphas_cumprod[timestep_prev]

                # DDIM update rule
                # Predict noise from x_0
                sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_alpha_bar_t = torch.sqrt(1. - alpha_bar_t)
                pred_noise = (x - sqrt_alpha_bar_t * x_0_pred) / sqrt_one_minus_alpha_bar_t

                # Compute x_{t-1}
                sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
                sqrt_one_minus_alpha_bar_prev = torch.sqrt(1. - alpha_bar_prev)

                # Direction pointing to x_t
                dir_xt = sqrt_one_minus_alpha_bar_prev * pred_noise

                # Stochastic component (eta controls randomness)
                if eta > 0:
                    sigma_t = eta * torch.sqrt((1. - alpha_bar_prev) / (1. - alpha_bar_t) * (1. - alpha_bar_t / alpha_bar_prev))
                    noise = torch.randn_like(x)
                    x = sqrt_alpha_bar_prev * x_0_pred + dir_xt + sigma_t * noise
                else:
                    # Deterministic DDIM
                    x = sqrt_alpha_bar_prev * x_0_pred + dir_xt
            else:
                # Final step: x_0 = x_0_pred
                x = x_0_pred

        # Add back mean
        return x + mean_fmri

import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM3D(nn.Module):
    def __init__(self, cond_dim, num_features):
        super().__init__()
        self.proj = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):
        gamma, beta = self.proj(cond).chunk(2, dim=1)
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1, 1)
        return x * (1 + gamma) + beta

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(out_channels)
        self.film = FiLM3D(cond_dim, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, cond):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.film(out, cond)
        out = F.gelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        return F.gelu(out + residual)

class VisualProjector(nn.Module):
    def __init__(self, visual_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(visual_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class Denoising3DUNet(nn.Module):
    def __init__(self, visual_dim=768, hidden_dim=32, dropout=0.2, in_channels=5):
        super().__init__()
        
        # Enhanced Visual Projector
        self.cond_dim = 512
        self.vis_proj = VisualProjector(visual_dim, self.cond_dim)
        
        # Input Channels: [Noisy_Input, Global_Mean, Coord_X, Coord_Y, Coord_Z] + optional ROI maps
        self.enc1 = ResBlock3D(in_channels, hidden_dim, self.cond_dim, dropout)
        self.down1 = nn.Conv3d(hidden_dim, hidden_dim*2, kernel_size=3, stride=2, padding=1)
        
        self.enc2 = ResBlock3D(hidden_dim*2, hidden_dim*4, self.cond_dim, dropout)
        self.down2 = nn.Conv3d(hidden_dim*4, hidden_dim*8, kernel_size=3, stride=2, padding=1)
        
        # Deeper Bottleneck
        self.bottleneck1 = ResBlock3D(hidden_dim*8, hidden_dim*8, self.cond_dim, dropout)
        self.bottleneck2 = ResBlock3D(hidden_dim*8, hidden_dim*8, self.cond_dim, dropout)
        
        self.up2 = nn.ConvTranspose3d(hidden_dim*8, hidden_dim*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = ResBlock3D(hidden_dim*8, hidden_dim*4, self.cond_dim, dropout)
        
        self.up1 = nn.ConvTranspose3d(hidden_dim*4, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec1 = ResBlock3D(hidden_dim*2, hidden_dim, self.cond_dim, dropout)
        
        self.final_head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, 1, kernel_size=1),
            nn.Tanh() # Constrain delta
        )

        # Learnable scaling factor for delta (Initial=1.0)
        self.output_scale = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x_noised, visual_feat, global_mean, spatial_priors):
        """
        x_noised: [B, 1, D, H, W]
        global_mean: [B, 1, D, H, W]
        spatial_priors: [B, C_spatial, D, H, W] -> e.g., Coords + ROI Gaussian maps
        """
        cond = self.vis_proj(visual_feat)
        
        # Cat channels
        x_input = torch.cat([x_noised, global_mean, spatial_priors], dim=1)
        
        e1 = self.enc1(x_input, cond)
        d1 = self.down1(e1)
        
        e2 = self.enc2(d1, cond)
        d2 = self.down2(e2)
        
        b = self.bottleneck1(d2, cond)
        b = self.bottleneck2(b, cond)
        
        u2 = self.up2(b)
        u2 = F.interpolate(u2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        res2 = self.dec2(torch.cat([u2, e2], dim=1), cond)
        
        u1 = self.up1(res2)
        u1 = F.interpolate(u1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        res1 = self.dec1(torch.cat([u1, e1], dim=1), cond)
        
        delta = self.final_head(res1)
        
        return global_mean + delta * self.output_scale


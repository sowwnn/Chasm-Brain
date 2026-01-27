import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder for CVAE: Maps target fMRI + condition (visual) to latent space.
    """
    def __init__(self, fmri_dim, cond_dim, latent_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fmri_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, cond):
        # Concatenate fMRI target and condition
        h = torch.cat([x, cond], dim=-1)
        h = self.net(h)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        return mu, logvar

class Decoder(nn.Module):
    """
    Stage 1 Decoder: Generates predicted parcels from latent z + condition.
    """
    def __init__(self, latent_dim, cond_dim, fmri_dim, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, fmri_dim)
        )

    def forward(self, z, cond):
        h = torch.cat([z, cond], dim=-1)
        return self.net(h)

class Refiner(nn.Module):
    """
    Stage 2 Refiner: Refines parcels to full voxels.
    """
    def __init__(self, parcel_dim, cond_dim, voxel_dim, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(parcel_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, voxel_dim)
        )

    def forward(self, parcels, cond):
        h = torch.cat([parcels, cond], dim=-1)
        return self.net(h)

class HierarchicalCVAE(nn.Module):
    """
    Hierarchical Conditional VAE for fMRI reconstruction.
    Stage 1: Predict 1k parcels.
    Stage 2: Refine 1k parcels to 15k voxels.
    """
    def __init__(self, visual_dim=768, parcel_dim=1000, voxel_dim=15724, latent_dim=128, hidden_dim=1024):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Condition projector
        self.cond_proj = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )

        # Encoder (only used during training)
        self.encoder = Encoder(parcel_dim, 512, latent_dim, hidden_dim=512)
        
        # Decoder Stage 1 (Parcels)
        self.decoder_s1 = Decoder(latent_dim, 512, parcel_dim, hidden_dim=hidden_dim)
        
        # Refiner Stage 2 (Voxels - Optional, can be used if needed)
        self.refiner_s2 = Refiner(parcel_dim, 512, voxel_dim, hidden_dim=hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, visual_feat, target_parcels=None):
        """
        visual_feat: [B, 768]
        target_parcels: [B, 1000] (Ground truth parcels, only for training)
        """
        cond = self.cond_proj(visual_feat)

        if target_parcels is not None:
            # Training mode: use encoder
            mu, logvar = self.encoder(target_parcels, cond)
            z = self.reparameterize(mu, logvar)
            
            # Predict parcels (Stage 1)
            pred_parcels = self.decoder_s1(z, cond)
            
            # Refine to voxels (Stage 2)
            # pred_voxels = self.refiner_s2(pred_parcels, cond)
            
            return pred_parcels, mu, logvar
        else:
            # Inference mode: sample z from N(0, 1)
            z = torch.randn(visual_feat.size(0), self.latent_dim, device=visual_feat.device)
            pred_parcels = self.decoder_s1(z, cond)
            return pred_parcels

    def generate(self, visual_feat, num_samples=1):
        """
        Generate diverse fMRI samples for the same visual features.
        """
        cond = self.cond_proj(visual_feat)
        batch_size = visual_feat.size(0)
        
        all_samples = []
        for _ in range(num_samples):
            z = torch.randn(batch_size, self.latent_dim, device=visual_feat.device)
            pred_parcels = self.decoder_s1(z, cond)
            all_samples.append(pred_parcels)
            
        return torch.stack(all_samples) # [num_samples, B, parcel_dim]

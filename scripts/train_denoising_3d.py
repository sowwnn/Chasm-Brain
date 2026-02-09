import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ARMNet import Denoising3DUNet, PeakFocusedLoss, MaskedPeakFocusedLoss
from data.neuroflux_dataset import create_dataloaders

class FMRI3DConverter:
    def __init__(self, mapping_path, roi_mapping_path=None, device='cuda'):
        m = np.load(mapping_path)
        self.compact_shape = tuple(m['compact_shape'])
        self.coords_compact = torch.tensor(m['coords_compact'], dtype=torch.long).to(device)
        self.n_voxels = len(self.coords_compact)
        self.device = device
        
        # Pre-calculate normalized coordinate maps [3, D, H, W]
        d, h, w = self.compact_shape
        self.coord_maps = torch.zeros((3,) + self.compact_shape, device=device)
        z_coords = torch.linspace(-1, 1, d, device=device)
        y_coords = torch.linspace(-1, 1, h, device=device)
        x_coords = torch.linspace(-1, 1, w, device=device)
        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        self.coord_maps[0] = zz
        self.coord_maps[1] = yy
        self.coord_maps[2] = xx
        
        # Pre-calculate ROI Gaussian maps if requested
        self.roi_maps = None
        if roi_mapping_path:
            self._init_roi_priors(roi_mapping_path)

    def _init_roi_priors(self, roi_mapping_path):
        data = np.load(roi_mapping_path)
        # Use 'streams' as primary ROI mapping
        roi_1d = torch.tensor(data['streams'], dtype=torch.long, device=self.device)
        unique_rois = torch.unique(roi_1d)
        unique_rois = unique_rois[unique_rois > 0] # Skip 0 and -1
        
        num_rois = len(unique_rois)
        self.roi_maps = torch.zeros((num_rois,) + self.compact_shape, device=self.device)
        
        d_coords, h_coords, w_coords = self.coords_compact[:, 0], self.coords_compact[:, 1], self.coords_compact[:, 2]
        
        # Sigma for Gaussian (normalized to compact_shape)
        sigma = 2.0 
        
        for i, roi_val in enumerate(unique_rois):
            mask = (roi_1d == roi_val)
            if mask.sum() == 0: continue
            
            # Find centroid in 3D
            roi_z = d_coords[mask].float().mean()
            roi_y = h_coords[mask].float().mean()
            roi_x = w_coords[mask].float().mean()
            
            # Create Gaussian volume
            z_idx = torch.arange(self.compact_shape[0], device=self.device).view(-1, 1, 1)
            y_idx = torch.arange(self.compact_shape[1], device=self.device).view(1, -1, 1)
            x_idx = torch.arange(self.compact_shape[2], device=self.device).view(1, 1, -1)
            
            dist_sq = (z_idx - roi_z)**2 + (y_idx - roi_y)**2 + (x_idx - roi_x)**2
            self.roi_maps[i] = torch.exp(-dist_sq / (2 * sigma**2))
        
        print(f"Initialized {num_rois} ROI Gaussian priors.")

    def to_3d(self, fmri_1d):
        B = fmri_1d.shape[0]
        fmri_3d = torch.zeros((B, 1) + self.compact_shape, dtype=fmri_1d.dtype, device=fmri_1d.device)
        d, h, w = self.coords_compact[:, 0], self.coords_compact[:, 1], self.coords_compact[:, 2]
        fmri_3d[:, 0, d, h, w] = fmri_1d
        return fmri_3d

    def to_1d(self, fmri_3d):
        if fmri_3d.ndim == 5:
            fmri_3d = fmri_3d.squeeze(1)
        d, h, w = self.coords_compact[:, 0], self.coords_compact[:, 1], self.coords_compact[:, 2]
        return fmri_3d[:, d, h, w]

    def get_spatial_priors(self, batch_size):
        # Cat [Coord_X, Coord_Y, Coord_Z, ROI_Map1, ...]
        priors = self.coord_maps.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        if self.roi_maps is not None:
            roi_expanded = self.roi_maps.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            priors = torch.cat([priors, roi_expanded], dim=1)
        return priors

    @property
    def num_prior_channels(self):
        return 3 + (self.roi_maps.shape[0] if self.roi_maps is not None else 0)

def compute_pearson_batch(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    corrs = []
    for i in range(pred_np.shape[0]):
        if pred_np[i].std() > 1e-6 and target_np[i].std() > 1e-6:
            r = np.corrcoef(pred_np[i], target_np[i])[0, 1]
            corrs.append(r)
    return np.mean(corrs) if corrs else 0.0

def train_epoch(model, converter, dataloader, criterion, optimizer, scaler, device, config, global_mean_1d):
    model.train()
    total_loss, total_pearson = 0, 0
    
    global_mean_3d = converter.to_3d(global_mean_1d.unsqueeze(0)).to(device)

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        target_1d = batch['fmri'].to(device)
        vis_feat = batch['embedding'].to(device)
        B = target_1d.size(0)
        
        batch_global_mean_1d = global_mean_1d.unsqueeze(0).expand(B, -1)
        batch_global_mean_3d = global_mean_3d.expand(B, -1, -1, -1, -1)
        batch_spatial_priors = converter.get_spatial_priors(B)
        
        # Denoising Strategy: Noise added to baseline
        noise_std = config['data'].get('noise_std', 0.15)
        noise = torch.randn_like(target_1d) * noise_std
        x_noised_1d = batch_global_mean_1d + noise
        x_noised_3d = converter.to_3d(x_noised_1d)

        vis_noise = torch.randn_like(vis_feat) * config['training'].get('vis_noise_std', 0.2)
        vis_feat_noised = vis_feat + vis_noise

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=config['training']['precision'] == 'fp16'):
            pred_3d = model(x_noised_3d, vis_feat_noised, batch_global_mean_3d, batch_spatial_priors)
            pred_1d = converter.to_1d(pred_3d)
            
            loss = criterion(pred_1d, target_1d, batch_global_mean_1d)

        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # GRAD CLIPPING
        
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        curr_p = compute_pearson_batch(pred_1d, target_1d)
        total_pearson += curr_p
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'pearson': f"{curr_p:.4f}"})

    return total_loss / len(dataloader), total_pearson / len(dataloader)

@torch.no_grad()
def save_validation_plot(pred_1d, target_1d, global_mean_1d, epoch, save_dir, writer):
    """
    Visualize 1D fMRI signals for comparison.
    """
    pred_np = pred_1d[0].detach().cpu().numpy()
    target_np = target_1d[0].detach().cpu().numpy()
    mean_np = global_mean_1d.detach().cpu().numpy()
    
    plt.figure(figsize=(20, 6))
    plt.plot(target_np, label='Target (Ground Truth)', alpha=0.7, color='green')
    plt.plot(pred_np, label='Predicted (Denoised)', alpha=0.8, color='blue')
    plt.plot(mean_np, label='Global Mean (Baseline)', alpha=0.5, color='gray', linestyle='--')
    
    plt.title(f'1D fMRI Reconstruction - Epoch {epoch}')
    plt.xlabel('Voxel Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(save_dir, f'val_plot_epoch_{epoch}.png')
    plt.savefig(plot_path)
    
    # Log to TensorBoard
    img = plt.imread(plot_path)
    writer.add_image('Validation/Reconstruction_1D', img, epoch, dataformats='HWC')
    plt.close()

@torch.no_grad()
def validate(model, converter, dataloader, device, config, global_mean_1d, epoch, save_dir, writer):
    model.eval()
    total_pearson, total_mse = 0, 0
    global_mean_3d = converter.to_3d(global_mean_1d.unsqueeze(0))

    # Keep track for one plot
    sample_pred_1d, sample_target_1d = None, None

    for i, batch in enumerate(dataloader):
        target_1d = batch['fmri'].to(device)
        vis_feat = batch['embedding'].to(device)
        B = target_1d.size(0)
        
        batch_global_mean_1d = global_mean_1d.unsqueeze(0).expand(B, -1)
        batch_global_mean_3d = global_mean_3d.expand(B, -1, -1, -1, -1)
        batch_spatial_priors = converter.get_spatial_priors(B)
        
        # Validation Strategy: Denoise the baseline mean (cleanest start)
        x_noised_1d = batch_global_mean_1d.clone()
        x_noised_3d = converter.to_3d(x_noised_1d)
        
        pred_3d = model(x_noised_3d, vis_feat, batch_global_mean_3d, batch_spatial_priors)
        pred_1d = converter.to_1d(pred_3d)
        total_pearson += compute_pearson_batch(pred_1d, target_1d)
        total_mse += F.mse_loss(pred_1d, target_1d).item()

        if i == 0: # Save first batch's first sample for plotting
            sample_pred_1d = pred_1d
            sample_target_1d = target_1d

    if sample_pred_1d is not None:
        plot_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        save_validation_plot(sample_pred_1d, sample_target_1d, global_mean_1d, epoch, plot_dir, writer)

    n = len(dataloader)
    return {'pearson': total_pearson / n, 'mse': total_mse / n}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_denoising_3d.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / 'logs'))

    # Enhanced Converter with ROI Priors
    converter = FMRI3DConverter(
        config['data']['mapping_path'], 
        roi_mapping_path=config['data'].get('roi_mapping_path'),
        device=device
    )

    train_loader, val_loader = create_dataloaders(
        datalist_path=config['data']['datalist_path'],
        fmri_path=config['data']['fmri_path'],
        train_embeddings_path=config['data']['train_embeddings_path'],
        test_embeddings_path=config['data']['test_embeddings_path'],
        subjects=[config['data']['subject']],
        batch_size=config['training']['batch_size'],
        average_trials_train=config['data']['average_trials_train'],
        average_trials_val=config['data']['average_trials_val'],
        augment_noise_train=config['data']['augment_noise_train'],
        noise_std=config['data'].get('noise_std', 0.15)
    )

    # Compute Global Mean of the entire subject 01 dataset
    print("Computing Global Mean fMRI...")
    all_fmri = torch.cat([b['fmri'] for b in train_loader], dim=0)
    global_mean_1d = all_fmri.mean(dim=0).to(device)

    # Initial Model with ROI-aware channel count
    model = Denoising3DUNet(
        visual_dim=config['model']['visual_dim'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout'],
        in_channels=2 + converter.num_prior_channels  # [Noisy, Mean] + [X, Y, Z, ROIs...]
    ).to(device)
    print(f"Model initialized with {2+converter.num_prior_channels} input channels.")

    use_mask = config['loss'].get('use_mask', False)
    if use_mask and config['data'].get('roi_mapping_path'):
        criterion = MaskedPeakFocusedLoss.from_roi_mapping(
            roi_mapping_path=config['data']['roi_mapping_path'],
            mask_field=config['loss'].get('mask_field', 'streams'),
            alpha=config['loss']['alpha'],
            tau=config['loss']['tau'],
            pearson_weight=config['loss']['pearson_weight']
        ).to(device)
    else:
        criterion = PeakFocusedLoss(
            alpha=config['loss']['alpha'],
            tau=config['loss']['tau'],
            pearson_weight=config['loss']['pearson_weight']
        )

    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['precision'] == 'fp16')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['max_epochs'])

    best_pearson = -1.0
    for epoch in range(1, config['training']['max_epochs'] + 1):
        t_loss, t_pearson = train_epoch(model, converter, train_loader, criterion, optimizer, scaler, device, config, global_mean_1d)
        v_metrics = validate(model, converter, val_loader, device, config, global_mean_1d, epoch, str(save_dir), writer)
        
        print(f"Epoch {epoch} | Loss: {t_loss:.4f} | Val Pearson: {v_metrics['pearson']:.4f}")
        
        writer.add_scalar('Loss/Train', t_loss, epoch)
        writer.add_scalar('Pearson/Val', v_metrics['pearson'], epoch)
        
        if v_metrics['pearson'] > best_pearson:
            best_pearson = v_metrics['pearson']
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            print(f"Saved new best model: {best_pearson:.4f}")
        
        scheduler.step()

    writer.close()

if __name__ == '__main__':
    main()

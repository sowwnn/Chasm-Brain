
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
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ARMNet import SpatialDenoisingARM, PeakFocusedLoss, MaskedPeakFocusedLoss
from data.neuroflux_dataset import create_dataloaders

class SpatialFeatureManager:
    """Pre-calculates and manages spatial features (Coords + ROI Gaussian maps)"""
    def __init__(self, mapping_path, roi_mapping_path, device='cuda'):
        # 1. Load Coords
        m = np.load(mapping_path)
        # coords_compact is [15724, 3] in Z, Y, X
        coords = torch.tensor(m['coords_compact'], dtype=torch.float32).to(device)
        
        # Normalize coords to [-1, 1] for stable learning
        max_vals = coords.max(dim=0)[0]
        min_vals = coords.min(dim=0)[0]
        coords_norm = 2.0 * (coords - min_vals) / (max_vals - min_vals + 1e-6) - 1.0
        
        # 2. Load ROI Priors
        roi_data = np.load(roi_mapping_path)
        roi_1d = torch.tensor(roi_data['streams'], dtype=torch.long, device=device)
        unique_rois = torch.unique(roi_1d)
        unique_rois = unique_rois[unique_rois > 0] # Skip 0 and -1
        
        roi_priors = []
        sigma = 5.0 # Spatial smoothness scale
        
        for roi_val in unique_rois:
            mask = (roi_1d == roi_val)
            centroid = coords[mask].mean(dim=0)
            
            # Distance squared in 3D
            dist_sq = torch.sum((coords - centroid)**2, dim=1)
            gaussian = torch.exp(-dist_sq / (2 * sigma**2))
            roi_priors.append(gaussian)
            
        # Cat everything: [N, 3 + num_rois]
        self.features = torch.cat([coords_norm] + [p.unsqueeze(1) for p in roi_priors], dim=1)
        self.spatial_dim = self.features.shape[1]
        print(f"Spatial Features Ready: {self.spatial_dim} channels (3 coords + {len(roi_priors)} ROIs)")

    def get_features(self):
        return self.features

def compute_pearson_batch(pred, target):
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    corrs = []
    for i in range(p.shape[0]):
        if p[i].std() > 1e-6 and t[i].std() > 1e-6:
            r = np.corrcoef(p[i], t[i])[0, 1]
            corrs.append(r)
    return np.mean(corrs) if corrs else 0.0

def train_epoch(model, spatial_feat, dataloader, criterion, optimizer, scaler, device, config, global_mean):
    model.train()
    total_loss, total_pearson = 0, 0
    
    # 1. Global Mean Baseline
    batch_mean_base = global_mean.unsqueeze(0)

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        target = batch['fmri'].to(device)
        vis_feat = batch['embedding'].to(device)
        B = target.size(0)
        
        batch_mean = batch_mean_base.expand(B, -1)
        
        # Denoising Strategy: Predict from mean + small noise
        noise = torch.randn_like(target) * config['data'].get('noise_std', 0.1)
        x_noised = batch_mean + noise

        # Visual Regularization
        vis_noise = torch.randn_like(vis_feat) * config['training'].get('vis_noise_std', 0.2)
        vis_feat_noised = vis_feat + vis_noise

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=config['training']['precision'] == 'fp16'):
            pred = model(x_noised, vis_feat_noised, batch_mean, spatial_feat)
            loss = criterion(pred, target, batch_mean)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        curr_p = compute_pearson_batch(pred, target)
        total_pearson += curr_p
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'pearson': f"{curr_p:.4f}"})

    return total_loss / len(dataloader), total_pearson / len(dataloader)

@torch.no_grad()
def validate(model, spatial_feat, dataloader, device, config, global_mean, epoch, save_dir, writer):
    model.eval()
    total_pearson, total_mse = 0, 0
    batch_mean_base = global_mean.unsqueeze(0)

    for i, batch in enumerate(dataloader):
        target = batch['fmri'].to(device)
        vis_feat = batch['embedding'].to(device)
        B = target.size(0)
        batch_mean = batch_mean_base.expand(B, -1)
        
        # Inference Strategy: Clean start (Mean only)
        pred = model(batch_mean.clone(), vis_feat, batch_mean, spatial_feat)
        
        total_pearson += compute_pearson_batch(pred, target)
        total_mse += nn.functional.mse_loss(pred, target).item()

    n = len(dataloader)
    pearson = total_pearson / n
    writer.add_scalar('Pearson/Val', pearson, epoch)
    return {'pearson': pearson, 'mse': total_mse / n}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_spatial_1d.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / 'logs'))

    # Load spatial features
    spatial_manager = SpatialFeatureManager(
        config['data']['mapping_path'],
        config['data']['roi_mapping_path'],
        device=device
    )
    spatial_feat = spatial_manager.get_features()

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
        noise_std=config['data'].get('noise_std', 0.1)
    )

    # Global Mean
    all_fmri = torch.cat([b['fmri'] for b in train_loader], dim=0)
    global_mean = all_fmri.mean(dim=0).to(device)

    model = SpatialDenoisingARM(
        visual_dim=config['model']['visual_dim'],
        spatial_dim=spatial_manager.spatial_dim,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    criterion = MaskedPeakFocusedLoss.from_roi_mapping(
        roi_mapping_path=config['data']['roi_mapping_path'],
        alpha=config['loss']['alpha'],
        tau=config['loss']['tau'],
        pearson_weight=config['loss']['pearson_weight']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['precision'] == 'fp16')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['max_epochs'])

    best_pearson = -1.0
    for epoch in range(1, config['training']['max_epochs'] + 1):
        t_loss, t_p = train_epoch(model, spatial_feat, train_loader, criterion, optimizer, scaler, device, config, global_mean)
        v_metrics = validate(model, spatial_feat, val_loader, device, config, global_mean, epoch, str(save_dir), writer)
        
        print(f"Epoch {epoch} | Loss: {t_loss:.4f} | Train P: {t_p:.4f} | Val P: {v_metrics['pearson']:.4f}")
        
        if v_metrics['pearson'] > best_pearson:
            best_pearson = v_metrics['pearson']
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            print(f"Saved new best: {best_pearson:.4f}")
        
        scheduler.step()

    writer.close()

if __name__ == '__main__':
    main()

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

from ARMNet import DenoisingARM, PeakFocusedLoss
# from ARMNet.darm_model import DeMARM
from data.neuroflux_dataset import create_dataloaders
from data.parcel_utils import ParcelMapper


def compute_pearson_batch(pred, target):
    if torch.is_tensor(pred): pred_np = pred.detach().cpu().numpy()
    else: pred_np = pred
    if torch.is_tensor(target): target_np = target.detach().cpu().numpy()
    else: target_np = target
    
    corrs = []
    for i in range(pred_np.shape[0]):
        if pred_np[i].std() > 1e-6 and target_np[i].std() > 1e-6:
            r = np.corrcoef(pred_np[i], target_np[i])[0, 1]
            corrs.append(r)
    return np.mean(corrs) if corrs else 0.0

def compute_mse_batch(pred, target):
    if torch.is_tensor(pred): pred_np = pred.detach().cpu().numpy()
    else: pred_np = pred
    if torch.is_tensor(target): target_np = target.detach().cpu().numpy()
    else: target_np = target
    return np.mean((pred_np - target_np)**2)

def plot_results(pred_parcels, target_parcels, mean_parcels, epoch, save_dir, parcel_mapper=None):
    num_samples = min(3, pred_parcels.shape[0])
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 5 * num_samples))
    if num_samples == 1: axes = axes.reshape(1, -1)
    
    pred_np = pred_parcels.detach().cpu().numpy()
    target_np = target_parcels.detach().cpu().numpy()
    mean_np = mean_parcels.detach().cpu().numpy()
    
    for i in range(num_samples):
        # 1k parcels
        axes[i, 0].plot(target_np[i], label='Target (GT)', alpha=0.5, color='blue')
        axes[i, 0].plot(mean_np[i], label='Mean (Input)', alpha=0.5, color='gray', linestyle='--')
        axes[i, 0].plot(pred_np[i], label='Pred', alpha=0.8, color='red')
        axes[i, 0].set_title(f'Sample {i+1} - 1000 Parcels')
        axes[i, 0].legend()
        
        if parcel_mapper:
            pred_v = parcel_mapper.reconstruct(pred_np[i])
            target_v = parcel_mapper.reconstruct(target_np[i])
            axes[i, 1].plot(target_v, label='GT (15k)', alpha=0.4, color='blue')
            axes[i, 1].plot(pred_v, label='Pred (15k)', alpha=0.7, color='red')
            axes[i, 1].set_title(f'Sample {i+1} - 15,724 Voxels')
            axes[i, 1].legend()
            
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'val_epoch_{epoch:03d}.png'))
    plt.close()

def compute_mean_fmri(train_loader, device) -> torch.Tensor:
    """Compute mean fMRI from training data."""
    print("Computing mean fMRI from training data...")
    all_fmri = []
    for batch in train_loader:
        all_fmri.append(batch['fmri'])

    all_fmri = torch.cat(all_fmri, dim=0)
    mean_fmri = all_fmri.mean(dim=0).to(device)
    print(f"Mean fMRI: shape={mean_fmri.shape}, mean={mean_fmri.mean():.4f}, std={mean_fmri.std():.4f}")
    return mean_fmri

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, config, mean_fmri):
    model.train()
    total_loss, total_p1k, total_mse = 0, 0, 0
    sigma = config['training'].get('sigma', 0.1)
    
    # Expand global mean to batch size
    batch_mean_base = mean_fmri.unsqueeze(0)
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        target = batch['fmri'].to(device) # [B, 1000]
        vis_feat = batch['embedding'].to(device) # [B, visual_dim]
        batch_mean = batch_mean_base.expand(target.size(0), -1)

        # 100% Pure Noise Strategy
        x_input = torch.randn_like(target)
        
        # Additional Visual Regularization:
        # Add noise to visual features during training to prevent memorization
        vis_noise = torch.randn_like(vis_feat) * config['training'].get('vis_noise_std', 0.2)
        vis_feat_noised = vis_feat + vis_noise
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=config['training']['precision'] == 'fp16'):
            pred = model(x_input, vis_feat_noised, batch_mean)
            loss = criterion(pred, target, batch_mean)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        p1k = compute_pearson_batch(pred, target)
        mse = compute_mse_batch(pred, target)
        total_p1k += p1k
        total_mse += mse
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'p1k': f"{p1k:.4f}"})
        
    return total_loss / len(dataloader), total_p1k / len(dataloader), total_mse / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, device, config, mean_fmri, parcel_mapper=None):
    model.eval()
    total_p1k, total_mse1k, total_p15k, total_mse15k = 0, 0, 0, 0
    sigma = config['training'].get('sigma', 0.1)
    batch_mean_base = mean_fmri.unsqueeze(0)
    
    pbar = tqdm(dataloader, desc="Validation")
    for batch in pbar:
        target_1k = batch['fmri'].to(device)
        vis_feat = batch['embedding'].to(device)
        batch_mean = batch_mean_base.expand(target_1k.size(0), -1)
        
        # STRICT VALIDATION: Always use PURE NOISE.
        # This reflects real inference where we only have visual features.
        x_input = torch.randn_like(target_1k)
        pred_1k = model(x_input, vis_feat, batch_mean)
        
        p1k = compute_pearson_batch(pred_1k, target_1k)
        mse1k = compute_mse_batch(pred_1k, target_1k)
        total_p1k += p1k
        total_mse1k += mse1k
        
        if parcel_mapper:
            pred_v = parcel_mapper.reconstruct(pred_1k.cpu().numpy())
            target_v = parcel_mapper.reconstruct(target_1k.cpu().numpy())
            p15k = compute_pearson_batch(pred_v, target_v)
            mse15k = compute_mse_batch(pred_v, target_v)
            total_p15k += p15k
            total_mse15k += mse15k
            
    n = len(dataloader)
    return {
        'p1k': total_p1k / n, 'mse1k': total_mse1k / n,
        'p15k': total_p15k / n, 'mse15k': total_mse15k / n
    }

def main(args):
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / 'logs'))
    
    # Loaders
    use_parcellation = config['data'].get('use_parcellation', False)
    parcel_path = config['data'].get('parcel_labels_path') if use_parcellation else None
    
    train_loader, val_loader = create_dataloaders(
        datalist_path=config['data']['datalist_path'],
        fmri_path=config['data']['fmri_path'],
        train_embeddings_path=config['data']['train_embeddings_path'],
        test_embeddings_path=config['data']['test_embeddings_path'],
        subjects=[config['data']['subject']],
        batch_size=config['training']['batch_size'],
        parcel_labels_path=parcel_path,
        average_trials_train=config['data']['average_trials_train'],
        average_trials_val=config['data']['average_trials_val'],
        augment_noise_train=config['data']['augment_noise_train'],
        noise_std=config['data'].get('noise_std', 0.15)
    )
    
    # Only load parcel_mapper if using parcellation
    parcel_mapper = None
    if config['data'].get('use_parcellation', False) and config['data'].get('parcel_labels_path'):
        parcel_mapper = ParcelMapper.from_files(config['data']['parcel_labels_path'])
        print(f"Using parcellation: {config['model']['fmri_dim']} parcels")
    else:
        print(f"Using full voxels: {config['model']['fmri_dim']} voxels")
    
    # Compute global mean fMRI
    mean_fmri = compute_mean_fmri(train_loader, device)
    
    model = DenoisingARM(
        visual_dim=config['model']['visual_dim'],
        fmri_dim=config['model']['fmri_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout']
    ).to(device)
    
    criterion = PeakFocusedLoss(
        alpha=config['loss']['alpha'],
        tau=config['loss']['tau'],
        pearson_weight=config['loss']['pearson_weight'],
        std_weight=config['loss'].get('std_weight', 1.0)
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['precision'] == 'fp16')
    
    best_p1k = -1.0
    for epoch in range(1, config['training']['max_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['max_epochs']}")
        t_loss, t_p1k, t_mse = train_epoch(model, train_loader, criterion, optimizer, scaler, device, config, mean_fmri)
        v_metrics = validate(model, val_loader, device, config, mean_fmri, parcel_mapper)
        
        # Step scheduler based on Val Pearson
        # scheduler.step(v_metrics['p1k'])
        
        print(f"Train P1k: {t_p1k:.4f} | Val P1k: {v_metrics['p1k']:.4f} | Val P15k: {v_metrics['p15k']:.4f}")
        
        # Plotting
        with torch.no_grad():
            batch = next(iter(val_loader))
            fmri = batch['fmri'][:3].to(device)
            vis = batch['embedding'][:3].to(device)
            batch_mean = mean_fmri.unsqueeze(0).expand(fmri.size(0), -1)
            noise = torch.randn_like(batch_mean) * config['training'].get('sigma', 0.1)
            pred = model(batch_mean + noise, vis, batch_mean)
            plot_results(pred, fmri, batch_mean, epoch, save_dir / 'plots', parcel_mapper)

        writer.add_scalar('Pearson/Train_1k', t_p1k, epoch)
        writer.add_scalar('Pearson/Val_1k', v_metrics['p1k'], epoch)
        writer.add_scalar('Pearson/Val_15k', v_metrics['p15k'], epoch)
        
        if v_metrics['p1k'] > best_p1k:
            best_p1k = v_metrics['p1k']
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            print(f"✓ New best: {best_p1k:.4f}")
            
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_denoising.yaml')
    args = parser.parse_args()
    main(args)

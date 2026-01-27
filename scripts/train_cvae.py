import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from ARMNet import HierarchicalCVAE, CVAELoss
from data.neuroflux_dataset import load_neuroflux_data, create_dataloaders
from data.parcel_utils import ParcelMapper

def plot_results(pred_parcels, target_parcels, epoch, save_dir, parcel_mapper=None):
    """
    Visualize ground truth vs prediction for comparison.
    """
    num_samples = min(3, pred_parcels.shape[0])
    
    # 1k parcels visualization
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 5 * num_samples))
    if num_samples == 1: axes = axes.reshape(1, -1)
    
    pred_np = pred_parcels.detach().cpu().numpy()
    target_np = target_parcels.detach().cpu().numpy()
    
    for i in range(num_samples):
        # 1k parcels
        axes[i, 0].plot(target_np[i], label='GT (1k)', alpha=0.7)
        axes[i, 0].plot(pred_np[i], label='Pred (1k)', alpha=0.7)
        axes[i, 0].set_title(f'Sample {i+1} - 1000 Parcels')
        axes[i, 0].legend()
        
        # 15k voxels (if mapper exists)
        if parcel_mapper:
            pred_v = parcel_mapper.reconstruct(pred_np[i])
            target_v = parcel_mapper.reconstruct(target_np[i])
            axes[i, 1].plot(target_v, label='GT (15k)', alpha=0.5)
            axes[i, 1].plot(pred_v, label='Pred (15k)', alpha=0.5)
            axes[i, 1].set_title(f'Sample {i+1} - 15,724 Voxels')
            axes[i, 1].legend()
            
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'val_epoch_{epoch:03d}.png'))
    plt.close()

def compute_pearson_batch(pred, target):
    """
    Compute Pearson correlation for each sample in the batch.
    """
    if torch.is_tensor(pred):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = pred
        
    if torch.is_tensor(target):
        target_np = target.detach().cpu().numpy()
    else:
        target_np = target
    
    corrs = []
    for i in range(pred_np.shape[0]):
        if pred_np[i].std() > 1e-6 and target_np[i].std() > 1e-6:
            r = np.corrcoef(pred_np[i], target_np[i])[0, 1]
            corrs.append(r)
    return np.mean(corrs) if corrs else 0.0

def compute_mse_batch(pred, target):
    """
    Compute MSE for each sample in the batch.
    """
    if torch.is_tensor(pred):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = pred
        
    if torch.is_tensor(target):
        target_np = target.detach().cpu().numpy()
    else:
        target_np = target
        
    return np.mean((pred_np - target_np)**2)

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, config):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kld = 0
    total_pearson = 0
    total_mse = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        fmri = batch['fmri'].to(device) # target parcels [B, 1000]
        embeddings = batch['embedding'].to(device) # [B, 768]
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=config['training']['precision'] == 'fp16'):
            # Forward pass: CVAE takes target_parcels during training
            pred_parcels, mu, logvar = model(embeddings, target_parcels=fmri)
            
            # Loss: pred, target, mean_fmri
            loss, recon_loss, kld_loss = criterion(pred_parcels, fmri, fmri, mu, logvar)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kld += kld_loss.item()
        
        pearson = compute_pearson_batch(pred_parcels, fmri)
        mse = compute_mse_batch(pred_parcels, fmri)
        total_pearson += pearson
        total_mse += mse
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}", 
            'mse': f"{mse:.4f}",
            'p': f"{pearson:.4f}"
        })
        
    return total_loss / len(dataloader), total_pearson / len(dataloader), total_mse / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, criterion, device, config, parcel_mapper=None):
    model.eval()
    total_p1k = 0
    total_mse1k = 0
    total_p15k = 0
    total_mse15k = 0
    
    pbar = tqdm(dataloader, desc="Validation")
    for batch in pbar:
        fmri_1k = batch['fmri'].to(device) # [B, 1000]
        embeddings = batch['embedding'].to(device)
        
        # Inference: sample from latent space
        pred_parcels = model(embeddings) # random sample
        
        # 1k Metrics
        p1k = compute_pearson_batch(pred_parcels, fmri_1k)
        mse1k = compute_mse_batch(pred_parcels, fmri_1k)
        total_p1k += p1k
        total_mse1k += mse1k
        
        # 15k Metrics
        if parcel_mapper is not None:
            pred_voxels = parcel_mapper.reconstruct(pred_parcels.cpu().numpy())
            target_voxels = parcel_mapper.reconstruct(fmri_1k.cpu().numpy())
            
            p15k = compute_pearson_batch(pred_voxels, target_voxels)
            mse15k = compute_mse_batch(pred_voxels, target_voxels)
            total_p15k += p15k
            total_mse15k += mse15k
        else:
            p15k, mse15k = 0.0, 0.0
            
        pbar.set_postfix({
            'p1k': f"{p1k:.3f}", 
            'p15k': f"{p15k:.3f}",
            'mse15k': f"{mse15k:.4f}"
        })
        
    n = len(dataloader)
    return {
        'p1k': total_p1k / n,
        'mse1k': total_mse1k / n,
        'p15k': total_p15k / n,
        'mse15k': total_mse15k / n
    }

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Save directory
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger
    writer = SummaryWriter(log_dir=str(save_dir / 'logs'))
    
    # Data
    train_loader, val_loader = create_dataloaders(
        datalist_path=config['data']['datalist_path'],
        fmri_path=config['data']['fmri_path'],
        train_embeddings_path=config['data']['train_embeddings_path'],
        test_embeddings_path=config['data']['test_embeddings_path'],
        subjects=[config['data']['subject']],
        batch_size=config['training']['batch_size'],
        parcel_labels_path=config['data']['parcel_labels_path'],
        average_trials_train=config['data']['average_trials_train'],
        average_trials_val=config['data']['average_trials_val'],
        augment_noise_train=config['data']['augment_noise_train'],
        noise_std=config['data']['noise_std']
    )
    
    # Parcel Mapper for 15k reconstruction
    parcel_mapper = ParcelMapper.from_files(config['data']['parcel_labels_path'])
    
    # Model
    model = HierarchicalCVAE(
        visual_dim=config['model']['visual_dim'],
        parcel_dim=config['model']['parcel_dim'],
        voxel_dim=config['model']['voxel_dim'],
        latent_dim=config['model']['latent_dim'],
        hidden_dim=config['model']['hidden_dim']
    ).to(device)
    
    # Criterion
    criterion = CVAELoss(
        alpha=config['loss']['alpha'],
        tau=config['loss']['tau'],
        pearson_weight=config['loss']['pearson_weight'],
        kld_weight=config['loss']['kld_weight']
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['training']['lr'], 
        weight_decay=config['training']['weight_decay']
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['precision'] == 'fp16')
    
    best_pearson = -1.0
    
    print(f"Starting training for {config['training']['max_epochs']} epochs...")
    for epoch in range(1, config['training']['max_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['max_epochs']}")
        
        train_loss, train_p1k, train_mse1k = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config
        )
        
        val_metrics = validate(
            model, val_loader, criterion, device, config, parcel_mapper=parcel_mapper
        )
        
        val_p1k = val_metrics['p1k']
        val_p15k = val_metrics['p15k']
        val_mse15k = val_metrics['mse15k']
        
        print(f"Train Loss: {train_loss:.4f} | Train P1k: {train_p1k:.4f}")
        print(f"Val P1k: {val_p1k:.4f} | Val P15k: {val_p15k:.4f} | Val MSE15k: {val_mse15k:.6f}")
        
        # Plotting
        with torch.no_grad():
            batch = next(iter(val_loader))
            fmri = batch['fmri'][:3].to(device)
            embeddings = batch['embedding'][:3].to(device)
            pred = model(embeddings)
            plot_results(pred, fmri, epoch, save_dir / 'plots', parcel_mapper=parcel_mapper)
        
        # Logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Pearson/Train_1k', train_p1k, epoch)
        writer.add_scalar('Pearson/Val_1k', val_p1k, epoch)
        writer.add_scalar('Pearson/Val_15k', val_p15k, epoch)
        writer.add_scalar('MSE/Val_15k', val_mse15k, epoch)
        
        # Save best model (based on 1k Pearson for consistency)
        if val_p1k > best_pearson:
            best_pearson = val_p1k
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            print(f"✓ Saved new best model (P1k: {val_p1k:.4f})")
            
        # Save last model
        torch.save(model.state_dict(), save_dir / 'last_model.pth')
        
    writer.close()
    print("\nTraining completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_cvae.yaml')
    args = parser.parse_args()
    main(args)

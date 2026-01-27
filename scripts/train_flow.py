"""
Training script for FlowARM (Flow Matching Version)
Specialized for Flow Matching with Optimal Transport path
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch.utils.tensorboard import SummaryWriter


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ARMNet.flow_model import FlowARM, FlowMatchingManager
from ARMNet.loss import FlowMatchingLoss
from data.neuroflux_dataset import create_dataloaders
from data.parcel_utils import ParcelMapper


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mean_fmri(train_loader, device) -> torch.Tensor:
    print("Computing mean fMRI from training data...")
    all_fmri = []
    for batch in train_loader:
        all_fmri.append(batch['fmri'])

    all_fmri = torch.cat(all_fmri, dim=0)
    mean_fmri = all_fmri.mean(dim=0).to(device)

    print(f"Mean fMRI: shape={mean_fmri.shape}, mean={mean_fmri.mean():.4f}, std={mean_fmri.std():.4f}")
    return mean_fmri


def compute_pearson(pred: torch.Tensor, target: torch.Tensor, parcel_mapper=None) -> dict:
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    correlations_parcel = []
    for i in range(pred_np.shape[0]):
        if pred_np[i].std() > 0 and target_np[i].std() > 0:
            corr, _ = pearsonr(pred_np[i], target_np[i])
            correlations_parcel.append(corr)

    pearson_parcel = np.mean(correlations_parcel) if correlations_parcel else 0.0
    result = {'parcel': pearson_parcel}

    if parcel_mapper is not None:
        pred_recon = parcel_mapper.reconstruct(pred_np)
        target_recon = parcel_mapper.reconstruct(target_np)

        correlations_recon = []
        for i in range(pred_recon.shape[0]):
            if pred_recon[i].std() > 0 and target_recon[i].std() > 0:
                corr, _ = pearsonr(pred_recon[i], target_recon[i])
                correlations_recon.append(corr)

        result['reconstructed'] = np.mean(correlations_recon) if correlations_recon else 0.0

    return result


def train_epoch_flow(model, flow_manager, train_loader, criterion, optimizer,
                     scaler, device, mean_fmri, grad_clip_val, use_amp, parcel_mapper=None):
    model.train()
    total_loss = 0.0
    total_pearson_parcel = 0.0
    total_pearson_recon = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        embeddings = batch['embedding'].to(device)
        fmri = batch['fmri'].to(device)

        batch_mean = mean_fmri.unsqueeze(0).expand(fmri.size(0), -1)

        # In FlowARM, x_0 = mean_fmri, x_1 = target_fmri
        x_0 = batch_mean
        x_1 = fmri

        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                # Get training tuple
                x_t, t, target_v = flow_manager.get_train_tuple(x_0, x_1)
                
                # Predict velocity with CFG (20% dropout on visual features)
                v_pred = model(x_t, t, embeddings, batch_mean, vis_dropout=0.2)
                
                # In OT, x_pred_final = x_0 + v_pred
                x_pred_final = x_0 + v_pred
                
                loss = criterion(v_pred, target_v, x_pred_final, x_1, batch_mean)

            scaler.scale(loss).backward()
            if grad_clip_val > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            scaler.step(optimizer)
            scaler.update()
        else:
            x_t, t, target_v = flow_manager.get_train_tuple(x_0, x_1)
            v_pred = model(x_t, t, embeddings, batch_mean, vis_dropout=0.2)
            x_pred_final = x_0 + v_pred
            
            loss = criterion(v_pred, target_v, x_pred_final, x_1, batch_mean)
            loss.backward()

            if grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            optimizer.step()

        # Metrics based on prediction at t=1 (approx)
        with torch.no_grad():
            pearson_dict = compute_pearson(x_pred_final, x_1, parcel_mapper)

        total_loss += loss.item()
        total_pearson_parcel += pearson_dict['parcel']
        if 'reconstructed' in pearson_dict:
            total_pearson_recon += pearson_dict['reconstructed']
        num_batches += 1

        display_pearson = pearson_dict.get('reconstructed', pearson_dict['parcel'])
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'pearson': f'{display_pearson:.4f}'
        })

    avg_pearson_parcel = total_pearson_parcel / num_batches
    avg_pearson_recon = total_pearson_recon / num_batches if parcel_mapper else None

    return total_loss / num_batches, avg_pearson_recon if avg_pearson_recon is not None else avg_pearson_parcel


@torch.no_grad()
def validate_flow(model, flow_manager, val_loader, criterion, device, mean_fmri, steps=10, parcel_mapper=None):
    model.eval()
    total_loss = 0.0
    total_pearson_parcel = 0.0
    total_pearson_recon = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    pbar = tqdm(val_loader, desc='Validation')
    for batch in pbar:
        embeddings = batch['embedding'].to(device)
        fmri = batch['fmri'].to(device)
        batch_mean = mean_fmri.unsqueeze(0).expand(fmri.size(0), -1)

        # Generate via sampling
        pred = flow_manager.sample(embeddings, batch_mean, steps=steps)

        # We need a dummy v_pred/v_target for FlowMatchingLoss if we use it for validation
        # or we can just use PeakFocusedLoss directly
        loss = criterion.peak_loss_fn(pred, fmri, batch_mean)
        pearson_dict = compute_pearson(pred, fmri, parcel_mapper)

        total_loss += loss.item()
        total_pearson_parcel += pearson_dict['parcel']
        if 'reconstructed' in pearson_dict:
            total_pearson_recon += pearson_dict['reconstructed']
        num_batches += 1

        if num_batches == 1:
            all_preds.append(pred.cpu())
            all_targets.append(fmri.cpu())

        display_pearson = pearson_dict.get('reconstructed', pearson_dict['parcel'])
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'pearson': f'{display_pearson:.4f}'
        })

    avg_pearson_parcel = total_pearson_parcel / num_batches
    avg_pearson_recon = total_pearson_recon / num_batches if parcel_mapper else None

    return {
        'loss': total_loss / num_batches,
        'pearson': avg_pearson_recon if avg_pearson_recon is not None else avg_pearson_parcel,
        'preds': all_preds[0] if all_preds else None,
        'targets': all_targets[0] if all_targets else None
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, save_path)


def visualize_predictions(preds, targets, epoch, save_dir, parcel_mapper=None, num_samples=3):
    if preds is None or targets is None:
        return

    preds_np = preds.numpy()
    targets_np = targets.numpy()

    if parcel_mapper is not None:
        preds_np = parcel_mapper.reconstruct(preds_np)
        targets_np = parcel_mapper.reconstruct(targets_np)
        title_suffix = " (Reconstructed)"
    else:
        title_suffix = ""

    num_samples = min(num_samples, preds_np.shape[0])
    epoch_dir = save_dir / f'epoch_{epoch:04d}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        axes[i, 0].plot(targets_np[i], linewidth=0.5, alpha=0.7)
        axes[i, 0].set_title(f'Sample {i+1}: Ground Truth')
        axes[i, 1].plot(preds_np[i], linewidth=0.5, alpha=0.7, color='orange')
        axes[i, 1].set_title(f'Sample {i+1}: Pred (Flow Sampled)')
        corr, _ = pearsonr(preds_np[i], targets_np[i])
        axes[i, 1].text(0.02, 0.98, f'Pearson: {corr:.4f}', transform=axes[i, 1].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Epoch {epoch}: FlowARM Predictions{title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(epoch_dir / 'gt_vs_pred.png', dpi=150)
    plt.close()


def main(args):
    config = load_config(args.config)
    set_seed(config['experiment']['seed'])

    save_dir = Path(config['experiment']['save_dir']) / config['experiment']['name'] / config['experiment']['version']
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    viz_dir = save_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = create_dataloaders(
        datalist_path=config['data']['datalist_path'],
        fmri_path=config['data']['fmri_path'],
        train_embeddings_path=config['data']['train_embeddings_path'],
        test_embeddings_path=config['data']['test_embeddings_path'],
        subjects=config['data']['subjects'],
        batch_size=config['data']['batch_size'],
        average_trials_train=config['data'].get('average_trials_train', False),
        average_trials_val=config['data'].get('average_trials_val', True),
        augment_noise_train=config['data'].get('augment_noise_train', False),
        noise_std=config['data'].get('noise_std', 0.1),
        parcel_labels_path=config['data'].get('parcel_labels_path'),
    )

    parcel_mapper = None
    if config['data'].get('use_parcellation', False) and config['data'].get('parcel_labels_path'):
        parcel_mapper = ParcelMapper.from_files(config['data']['parcel_labels_path'])

    output_dim = config['model']['output_dim']
    if config['data'].get('use_parcellation', False):
        sample_batch = next(iter(train_loader))
        output_dim = sample_batch['fmri'].shape[1]

    mean_fmri = compute_mean_fmri(train_loader, device)

    model = FlowARM(
        input_dim=config['model']['input_dim'],
        fmri_dim=output_dim,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    flow_manager = FlowMatchingManager(
        model=model, 
        device=device, 
        sigma=config['model']['flow'].get('sigma', 0.1)
    )
    
    criterion = FlowMatchingLoss(
        alpha=config['model']['loss']['alpha'],
        tau=config['model']['loss']['tau'],
        pearson_weight=config['model']['loss']['pearson_weight']
    )

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['training']['lr'], 
        weight_decay=0.05  # Tăng lên để chống overfitting
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['max_epochs'])

    use_amp = config['training'].get('precision', '32') == '16-mixed'
    scaler = GradScaler('cuda') if use_amp else None
    writer = SummaryWriter(save_dir / 'logs')

    best_metric = -float('inf')
    for epoch in range(config['training']['max_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['max_epochs']}")
        
        train_loss, train_pearson = train_epoch_flow(
            model, flow_manager, train_loader, criterion, optimizer, scaler,
            device, mean_fmri, config['training'].get('gradient_clip_val', 1.0), use_amp, parcel_mapper
        )
        
        val_metrics = validate_flow(model, flow_manager, val_loader, criterion, device, mean_fmri, steps=config['model']['flow'].get('sample_steps', 10), parcel_mapper=parcel_mapper)
        
        scheduler.step()
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Pearson/train', train_pearson, epoch)
        writer.add_scalar('Pearson/val', val_metrics['pearson'], epoch)
        
        print(f"Train Loss: {train_loss:.4f} | Val Pearson: {val_metrics['pearson']:.4f}")

        if val_metrics['pearson'] > best_metric:
            best_metric = val_metrics['pearson']
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, checkpoint_dir / 'best.pt')

        if epoch % config['logging'].get('viz_every_n_epochs', 5) == 0:
            visualize_predictions(val_metrics['preds'], val_metrics['targets'], epoch, viz_dir, parcel_mapper)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_flow.yaml')
    args = parser.parse_args()
    main(args)

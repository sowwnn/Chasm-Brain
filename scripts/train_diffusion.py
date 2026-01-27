"""
Training script for DiffusionARM
Specialized for diffusion-based fMRI prediction with x_0 parameterization
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

from ARMNet.diffusion_model import DiffusionARM, DiffusionManager
from ARMNet.loss import PeakFocusedLoss
from data.neuroflux_dataset import create_dataloaders
from data.parcel_utils import ParcelMapper


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


# def compute_pearson(pred: torch.Tensor, target: torch.Tensor) -> float:
#     """Compute average Pearson correlation across batch."""
#     pred_np = pred.detach().cpu().numpy()
#     target_np = target.detach().cpu().numpy()

#     correlations = []
#     for i in range(pred_np.shape[0]):
#         if pred_np[i].std() > 0 and target_np[i].std() > 0:
#             corr, _ = pearsonr(pred_np[i], target_np[i])
#             correlations.append(corr)

#     return np.mean(correlations) if correlations else 0.0

def compute_pearson(pred: torch.Tensor, target: torch.Tensor, parcel_mapper=None) -> dict:
    """
    Compute average Pearson correlation across batch.

    Args:
        pred: Predicted fMRI [B, N]
        target: Ground truth fMRI [B, N]
        parcel_mapper: Optional ParcelMapper for reconstruction to voxels

    Returns:
        Dictionary with 'parcel' and optionally 'reconstructed' pearson correlations
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    # Compute on parcels/original dimension
    correlations_parcel = []
    for i in range(pred_np.shape[0]):
        if pred_np[i].std() > 0 and target_np[i].std() > 0:
            corr, _ = pearsonr(pred_np[i], target_np[i])
            correlations_parcel.append(corr)

    pearson_parcel = np.mean(correlations_parcel) if correlations_parcel else 0.0

    result = {'parcel': pearson_parcel}

    # Compute on reconstructed voxels if parcel_mapper is provided
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


def train_epoch_diffusion(model, diffusion_manager, train_loader, criterion, optimizer,
                         scaler, device, mean_fmri, grad_clip_val, use_amp, parcel_mapper=None):
    """Train one epoch with diffusion process."""
    model.train()
    total_loss = 0.0
    total_pearson_parcel = 0.0
    total_pearson_recon = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        embeddings = batch['embedding'].to(device)
        fmri = batch['fmri'].to(device)

        # Expand mean_fmri to batch size
        batch_mean = mean_fmri.unsqueeze(0).expand(fmri.size(0), -1)

        # Get residual (target - mean)
        x_0 = fmri - batch_mean

        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                # Get training tuple from diffusion manager
                noisy_x, t, target_x_0 = diffusion_manager.get_train_tuple(x_0, embeddings)

                # Model predicts clean x_0 from noisy input
                pred_x_0 = model(embeddings, batch_mean, noisy_fmri=noisy_x, t=t)

                # Compute loss on predicted x_0 vs clean x_0
                # Reconstruct full fMRI for loss computation
                pred_fmri = pred_x_0 + batch_mean
                target_fmri = target_x_0 + batch_mean

                loss = criterion(pred_fmri, target_fmri, batch_mean)

            scaler.scale(loss).backward()
            if grad_clip_val > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Get training tuple
            noisy_x, t, target_x_0 = diffusion_manager.get_train_tuple(x_0, embeddings)

            # Model predicts clean x_0
            pred_x_0 = model(embeddings, batch_mean, noisy_fmri=noisy_x, t=t)

            # Reconstruct full fMRI
            pred_fmri = pred_x_0 + batch_mean
            target_fmri = target_x_0 + batch_mean

            loss = criterion(pred_fmri, target_fmri, batch_mean)
            loss.backward()

            if grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            optimizer.step()

        # Metrics
        with torch.no_grad():
            pearson_dict = compute_pearson(pred_fmri, target_fmri, parcel_mapper)

        total_loss += loss.item()
        total_pearson_parcel += pearson_dict['parcel']
        if 'reconstructed' in pearson_dict:
            total_pearson_recon += pearson_dict['reconstructed']
        num_batches += 1

        # Display reconstructed pearson if available, otherwise parcel pearson
        display_pearson = pearson_dict.get('reconstructed', pearson_dict['parcel'])
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'pearson': f'{display_pearson:.4f}'
        })

    avg_pearson_parcel = total_pearson_parcel / num_batches
    avg_pearson_recon = total_pearson_recon / num_batches if parcel_mapper else None

    return total_loss / num_batches, avg_pearson_recon if avg_pearson_recon is not None else avg_pearson_parcel


@torch.no_grad()
def validate_diffusion(model, diffusion_manager, val_loader, criterion, device, mean_fmri, use_ddim=False, parcel_mapper=None):
    """Validate model using diffusion sampling (DDPM or DDIM)."""
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

        # Generate via diffusion sampling (DDIM or DDPM)
        if use_ddim:
            pred = diffusion_manager.sample_ddim(embeddings, batch_mean)
        else:
            pred = diffusion_manager.sample(embeddings, batch_mean)

        # Compute loss
        loss = criterion(pred, fmri, batch_mean)
        pearson_dict = compute_pearson(pred, fmri, parcel_mapper)

        total_loss += loss.item()
        total_pearson_parcel += pearson_dict['parcel']
        if 'reconstructed' in pearson_dict:
            total_pearson_recon += pearson_dict['reconstructed']
        num_batches += 1

        # Store first batch for visualization
        if num_batches == 1:
            all_preds.append(pred.cpu())
            all_targets.append(fmri.cpu())

        # Display reconstructed pearson if available, otherwise parcel pearson
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
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, save_path)


def visualize_predictions(preds, targets, epoch, save_dir, parcel_mapper=None, num_samples=3):
    """Generate visualization plots."""
    if preds is None or targets is None:
        return

    preds_np = preds.numpy()
    targets_np = targets.numpy()

    # Reconstruct from parcels if using parcellation
    if parcel_mapper is not None:
        print(f"Reconstructing {preds_np.shape[1]} parcels → {parcel_mapper.n_voxels} voxels for visualization")
        preds_np = parcel_mapper.reconstruct(preds_np)
        targets_np = parcel_mapper.reconstruct(targets_np)
        title_suffix = " (Reconstructed from Parcels)"
    else:
        title_suffix = ""

    num_samples = min(num_samples, preds_np.shape[0])
    epoch_dir = save_dir / f'epoch_{epoch:04d}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    # Ground truth vs Prediction
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Ground truth
        axes[i, 0].plot(targets_np[i], linewidth=0.5, alpha=0.7)
        axes[i, 0].set_title(f'Sample {i+1}: Ground Truth')
        axes[i, 0].set_xlabel('Voxel Index')
        axes[i, 0].set_ylabel('fMRI Signal')
        axes[i, 0].grid(True, alpha=0.3)

        # Prediction
        axes[i, 1].plot(preds_np[i], linewidth=0.5, alpha=0.7, color='orange')
        axes[i, 1].set_title(f'Sample {i+1}: Prediction (Diffusion Sampled)')
        axes[i, 1].set_xlabel('Voxel Index')
        axes[i, 1].set_ylabel('fMRI Signal')
        axes[i, 1].grid(True, alpha=0.3)

        # Compute correlation
        corr, _ = pearsonr(preds_np[i], targets_np[i])
        axes[i, 1].text(
            0.02, 0.98, f'Pearson: {corr:.4f}',
            transform=axes[i, 1].transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    plt.suptitle(f'Epoch {epoch}: DiffusionARM Predictions{title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(epoch_dir / 'gt_vs_pred.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Correlation scatter plots
    fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 4))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        axes[i].scatter(targets_np[i], preds_np[i], alpha=0.3, s=1, c='blue')

        min_val = min(targets_np[i].min(), preds_np[i].min())
        max_val = max(targets_np[i].max(), preds_np[i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

        corr, _ = pearsonr(preds_np[i], targets_np[i])
        mse = np.mean((preds_np[i] - targets_np[i]) ** 2)

        axes[i].set_xlabel('Ground Truth')
        axes[i].set_ylabel('Prediction')
        axes[i].set_title(f'Sample {i+1}\nPearson: {corr:.4f}, MSE: {mse:.4f}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(f'Epoch {epoch}: Correlation Plots{title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(epoch_dir / 'correlation_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualizations to {epoch_dir}")


def main(args):
    """Main training function for DiffusionARM."""
    # Load config
    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Override from CLI
    if args.batch_size: config['data']['batch_size'] = args.batch_size
    if args.lr: config['training']['lr'] = args.lr
    if args.max_epochs: config['training']['max_epochs'] = args.max_epochs
    if args.num_res_blocks: config['model']['num_res_blocks'] = args.num_res_blocks
    if args.timesteps: config['model']['diffusion']['timesteps'] = args.timesteps

    # Seed
    print(f"Seed set to {config['experiment']['seed']}")
    set_seed(config['experiment']['seed'])

    # Save directory
    save_dir = Path(config['experiment']['save_dir']) / config['experiment']['name'] / config['experiment']['version']
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    viz_dir = save_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved: {save_dir / 'config.yaml'}\n")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and config['hardware']['accelerator'] == 'gpu' else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)

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
        apply_zscore_train=config['data'].get('apply_zscore_train', False),
        apply_zscore_val=config['data'].get('apply_zscore_val', False),
    )

    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # Load parcel_mapper if using parcellation
    parcel_mapper = None
    if config['data'].get('use_parcellation', False) and config['data'].get('parcel_labels_path'):
        parcel_mapper = ParcelMapper.from_files(config['data']['parcel_labels_path'])

    # Auto-detect output_dim from parcellation
    output_dim = config['model']['output_dim']
    if config['data'].get('use_parcellation', False):
        sample_batch = next(iter(train_loader))
        output_dim = sample_batch['fmri'].shape[1]
        config['model']['output_dim'] = output_dim
        print(f"Using parcellated dimension: {output_dim}")

    # Compute mean fMRI
    mean_fmri = compute_mean_fmri(train_loader, device)

    # Create model
    print("\n" + "="*60)
    print("Creating DiffusionARM Model")
    print("="*60)

    model = DiffusionARM(
        input_dim=config['model']['input_dim'],
        fmri_dim=output_dim,
        output_dim=output_dim,
        hidden_dim=config['model']['hidden_dim'],
    # num_res_blocks=config['model']['num_res_blocks'],
        dropout=config['model']['dropout']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: DiffusionARM")
    print(f"Hidden dim: {config['model']['hidden_dim']}")
    print(f"Num ResBlocks: {config['model']['num_res_blocks']}")
    print(f"Dropout: {config['model']['dropout']}")
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Create diffusion manager
    diffusion_config = config['model']['diffusion']
    diffusion_manager = DiffusionManager(
        model=model,
        timesteps=diffusion_config['timesteps'],
        inference_timesteps=diffusion_config.get('inference_timesteps', diffusion_config['timesteps']),
        device=device
    )
    print(f"Diffusion timesteps (training): {diffusion_config['timesteps']}")
    print(f"Diffusion timesteps (inference): {diffusion_manager.inference_timesteps}")
    print(f"Clip value: {diffusion_config['clip_value']}")
    print(f"Use DDIM: {diffusion_config.get('use_ddim', False)}")

    # Loss function
    loss_config = config['model']['loss']
    criterion = PeakFocusedLoss(
        alpha=loss_config['alpha'],
        tau=loss_config['tau'],
        pearson_weight=loss_config['pearson_weight']
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999)
    )

    # Scheduler
    scheduler_type = config['training'].get('scheduler', 'cosine')
    if scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

    # AMP scaler
    use_amp = config['training'].get('precision', '32') == '16-mixed'
    scaler = GradScaler('cuda') if use_amp else None

    # TensorBoard
    writer = SummaryWriter(save_dir / 'logs')

    # Training setup
    max_epochs = config['training']['max_epochs']
    grad_clip_val = config['training'].get('gradient_clip_val', 1.0)
    early_stopping_patience = config['validation'].get('early_stopping_patience', None)
    best_metric = -float('inf')
    patience_counter = 0

    print("\n" + "="*60)
    print("Training DiffusionARM")
    print("="*60)
    print(f"Epochs: {max_epochs} | Batch: {config['data']['batch_size']} | LR: {config['training']['lr']}")
    print(f"Precision: {config['training'].get('precision', '32')} | Grad Clip: {grad_clip_val}")

    # Training loop
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        print("-" * 60)

        # Train
        train_loss, train_pearson = train_epoch_diffusion(
            model, diffusion_manager, train_loader, criterion, optimizer, scaler,
            device, mean_fmri, grad_clip_val, use_amp, parcel_mapper
        )

        # Validate
        use_ddim = config['model']['diffusion'].get('use_ddim', False)
        val_metrics = validate_diffusion(model, diffusion_manager, val_loader, criterion, device, mean_fmri, use_ddim, parcel_mapper)

        # Scheduler step
        if scheduler_type == 'plateau':
            scheduler.step(val_metrics['pearson'])
        else:
            scheduler.step()

        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Pearson/train', train_pearson, epoch)
        writer.add_scalar('Pearson/val', val_metrics['pearson'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"Train Loss: {train_loss:.4f} | Train Pearson: {train_pearson:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Pearson: {val_metrics['pearson']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Visualize
        if config['logging'].get('visualize', True) and (epoch + 1) % config['logging'].get('viz_every_n_epochs', 1) == 0:
            visualize_predictions(
                val_metrics['preds'],
                val_metrics['targets'],
                epoch,
                viz_dir,
                parcel_mapper,
                config['logging'].get('viz_num_samples', 3)
            )

        # Save best checkpoint
        current_metric = val_metrics['pearson']
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_loss': val_metrics['loss'], 'val_pearson': val_metrics['pearson']},
                checkpoint_dir / 'best.pt'
            )
            print(f"✓ Saved best checkpoint (Pearson: {best_metric:.4f})")
        else:
            patience_counter += 1

        # Save last checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {'val_loss': val_metrics['loss'], 'val_pearson': val_metrics['pearson']},
            checkpoint_dir / 'last.pt'
        )

        # Early stopping
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {early_stopping_patience} epochs without improvement")
            break

    writer.close()

    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    print(f"Best Val Pearson: {best_metric:.4f}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {save_dir / 'logs'}")
    print(f"Visualizations: {viz_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DiffusionARM for I2fMRI')
    parser.add_argument('--config', type=str, default='configs/train_diffusion.yaml', help='Config file')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--max_epochs', type=int, default=None, help='Override max epochs')
    parser.add_argument('--num_res_blocks', type=int, default=None, help='Override num ResBlocks')
    parser.add_argument('--timesteps', type=int, default=None, help='Override diffusion timesteps')

    args = parser.parse_args()
    main(args)

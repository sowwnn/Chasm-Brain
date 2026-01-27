"""
Training script for I2fMRI using Pure PyTorch
Supports multiple model architectures and flexible configuration
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

from ARMNet.model import ARMNet
from ARMNet.sde_model import ARMSDE
from ARMNet.diffusion_model import DiffusionARM, DiffusionManager
from ARMNet.flow_model import FlowARM, FlowMatchingManager
from ARMNet.loss import PeakFocusedLoss, FlowMatchingLoss
from data.neuroflux_dataset import create_dataloaders
from data.parcel_utils import ParcelMapper


class ModelWrapper(nn.Module):
    """Wrapper to store mean_fmri with the model."""
    def __init__(self, model, mean_fmri):
        super().__init__()
        self.model = model
        self.register_buffer('mean_fmri', mean_fmri)

    def forward(self, x):
        return self.model(x, self.mean_fmri)


class LossWrapper(nn.Module):
    """Wrapper to store mean_fmri with the loss function."""
    def __init__(self, loss_fn, mean_fmri):
        super().__init__()
        self.loss_fn = loss_fn
        self.register_buffer('mean_fmri', mean_fmri)

    def forward(self, pred, target):
        return self.loss_fn(pred, target, self.mean_fmri)


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


def create_model(config: dict, mean_fmri: torch.Tensor) -> nn.Module:
    """Create model based on configuration."""
    model_name = config['model']['name']
    input_dim = config['model']['input_dim']
    output_dim = config['model']['output_dim']
    hidden_dim = config['model']['hidden_dim']
    dropout = config['model'].get('dropout', 0.3)

    if model_name == "ARMNet":
        model = ARMNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    elif model_name == "ARMSDE":
        model = ARMSDE(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim
        )
    elif model_name == "DiffusionARM":
        num_res_blocks = config['model'].get('num_res_blocks', 4)
        model = DiffusionARM(
            input_dim=input_dim,
            fmri_dim=output_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            dropout=dropout
        )
    elif model_name == "FlowARM":
        model = FlowARM(
            input_dim=input_dim,
            fmri_dim=output_dim,
            hidden_dim=hidden_dim
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def create_loss_fn(config: dict):
    """Create loss function based on configuration."""
    loss_type = config['model']['loss']['type']

    if loss_type == "PeakFocusedLoss":
        return PeakFocusedLoss(
            alpha=config['model']['loss']['alpha'],
            tau=config['model']['loss']['tau'],
            pearson_weight=config['model']['loss']['pearson_weight']
        )
    elif loss_type == "FlowMatchingLoss":
        return FlowMatchingLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_type}")


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


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, grad_clip_val, use_amp, parcel_mapper=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_pearson_parcel = 0.0
    total_pearson_recon = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        embeddings = batch['embedding'].to(device)
        fmri = batch['fmri'].to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                pred = model(embeddings)
                loss = criterion(pred, fmri)

            scaler.scale(loss).backward()
            if grad_clip_val > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(embeddings)
            loss = criterion(pred, fmri)
            loss.backward()
            if grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            optimizer.step()

        # Metrics
        with torch.no_grad():
            pearson_dict = compute_pearson(pred, fmri, parcel_mapper)

        total_loss += loss.item()
        total_pearson_parcel += pearson_dict['parcel']
        if 'reconstructed' in pearson_dict:
            total_pearson_recon += pearson_dict['reconstructed']
        num_batches += 1

        postfix = {
            'loss': f'{loss.item():.4f}',
            'pearson_p': f'{pearson_dict["parcel"]:.4f}'
        }
        if 'reconstructed' in pearson_dict:
            postfix['pearson_r'] = f'{pearson_dict["reconstructed"]:.4f}'
        pbar.set_postfix(postfix)

    result = {
        'loss': total_loss / num_batches,
        'pearson_parcel': total_pearson_parcel / num_batches
    }
    if parcel_mapper is not None:
        result['pearson_reconstructed'] = total_pearson_recon / num_batches

    return result


@torch.no_grad()
def validate(model, val_loader, criterion, device, parcel_mapper=None):
    """Validate model."""
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

        pred = model(embeddings)
        loss = criterion(pred, fmri)
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

        postfix = {
            'loss': f'{loss.item():.4f}',
            'pearson_p': f'{pearson_dict["parcel"]:.4f}'
        }
        if 'reconstructed' in pearson_dict:
            postfix['pearson_r'] = f'{pearson_dict["reconstructed"]:.4f}'
        pbar.set_postfix(postfix)

    result = {
        'loss': total_loss / num_batches,
        'pearson_parcel': total_pearson_parcel / num_batches,
        'preds': all_preds[0] if all_preds else None,
        'targets': all_targets[0] if all_targets else None
    }
    if parcel_mapper is not None:
        result['pearson_reconstructed'] = total_pearson_recon / num_batches

    return result


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
        title_suffix = " (Reconstructed from 1000 Parcels)"
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
        axes[i, 1].set_title(f'Sample {i+1}: Prediction')
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

    plt.suptitle(f'Epoch {epoch}: Ground Truth vs Prediction{title_suffix}', fontsize=14)
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
    """Main training function."""
    # Load config
    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Override from CLI
    if args.batch_size: config['data']['batch_size'] = args.batch_size
    if args.lr: config['training']['lr'] = args.lr
    if args.max_epochs: config['training']['max_epochs'] = args.max_epochs

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
    print("Creating Model")
    print("="*60)

    base_model = create_model(config, mean_fmri)
    model = ModelWrapper(base_model, mean_fmri).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config['model']['name']}")
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Loss function
    base_criterion = create_loss_fn(config)
    criterion = LossWrapper(base_criterion, mean_fmri).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999)
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
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
    print("Training")
    print("="*60)
    print(f"Epochs: {max_epochs} | Batch: {config['data']['batch_size']} | LR: {config['training']['lr']}")
    print(f"Precision: {config['training'].get('precision', '32')} | Grad Clip: {grad_clip_val}")

    # Training loop
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        print("-" * 60)

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, grad_clip_val, use_amp, parcel_mapper
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, parcel_mapper)

        # Scheduler step
        scheduler.step()

        # Log metrics
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Pearson_Parcel/train', train_metrics['pearson_parcel'], epoch)
        writer.add_scalar('Pearson_Parcel/val', val_metrics['pearson_parcel'], epoch)
        if 'pearson_reconstructed' in train_metrics:
            writer.add_scalar('Pearson_Reconstructed/train', train_metrics['pearson_reconstructed'], epoch)
            writer.add_scalar('Pearson_Reconstructed/val', val_metrics['pearson_reconstructed'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Pearson (Parcel): {train_metrics['pearson_parcel']:.4f}")
        if 'pearson_reconstructed' in train_metrics:
            print(f"Train Pearson (Reconstructed 15k): {train_metrics['pearson_reconstructed']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Pearson (Parcel): {val_metrics['pearson_parcel']:.4f}")
        if 'pearson_reconstructed' in val_metrics:
            print(f"Val Pearson (Reconstructed 15k): {val_metrics['pearson_reconstructed']:.4f}")
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

        # Save best checkpoint (use reconstructed pearson if available, else parcel pearson)
        current_metric = val_metrics.get('pearson_reconstructed', val_metrics['pearson_parcel'])
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            save_metrics = {
                'val_loss': val_metrics['loss'],
                'val_pearson_parcel': val_metrics['pearson_parcel']
            }
            if 'pearson_reconstructed' in val_metrics:
                save_metrics['val_pearson_reconstructed'] = val_metrics['pearson_reconstructed']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                save_metrics,
                checkpoint_dir / 'best.pt'
            )
            metric_type = "Reconstructed" if 'pearson_reconstructed' in val_metrics else "Parcel"
            print(f"✓ Saved best checkpoint (Pearson {metric_type}: {best_metric:.4f})")
        else:
            patience_counter += 1

        # Save last checkpoint
        save_metrics = {
            'val_loss': val_metrics['loss'],
            'val_pearson_parcel': val_metrics['pearson_parcel']
        }
        if 'pearson_reconstructed' in val_metrics:
            save_metrics['val_pearson_reconstructed'] = val_metrics['pearson_reconstructed']
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            save_metrics,
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
    parser = argparse.ArgumentParser(description='Train I2fMRI with Pure PyTorch')
    parser.add_argument('--config', type=str, default='configs/train_armnet.yaml', help='Config file')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--max_epochs', type=int, default=None, help='Override max epochs')

    args = parser.parse_args()
    main(args)

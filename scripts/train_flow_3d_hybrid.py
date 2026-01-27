"""
Training script for Flow3DHybridARM
Hybrid GNN+CNN Flow Matching for 3D fMRI generation with optional mask conditioning
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: tensorboard not available. Logging will be limited.")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# Add parent directory to path to import ARMNet modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ARMNet.flow_3d_hybrid_model import (
    Flow3DHybridARM,
    fmri_1d_to_3d_compact,
    fmri_3d_compact_to_1d
)
from ARMNet.flow_3d_loss import FlowMatching3DLoss, Metrics3D1D


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_3d_mapping(mapping_path: str) -> Dict[str, Any]:
    """Load 3D compact mapping from npz file"""
    data = np.load(mapping_path)
    return {
        'compact_shape': tuple(data['compact_shape']),
        'compact_mask': data['compact_mask'],
        'coords_compact': data['coords_compact'],
        'n_voxels': int(data['n_voxels'])
    }


class fMRI3DDataset(Dataset):
    def __init__(
        self, 
        fmri_data: np.ndarray, 
        embeddings: np.ndarray, 
        coords_compact: np.ndarray, 
        compact_shape: Tuple[int, int, int]
    ):
        self.fmri_1d = fmri_data  # [N, n_voxels]
        self.embeddings = embeddings  # [N, embed_dim]
        self.coords_compact = coords_compact
        self.compact_shape = compact_shape

    def __len__(self) -> int:
        return len(self.fmri_1d)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fmri_1d = self.fmri_1d[idx]
        embedding = self.embeddings[idx]

        # Convert to 3D
        fmri_3d = fmri_1d_to_3d_compact(
            fmri_1d, self.coords_compact, self.compact_shape, fill_value=0.0
        )

        return {
            'fmri_3d': torch.from_numpy(fmri_3d).unsqueeze(0).float(),  # [1, D, H, W]
            'fmri_1d': torch.from_numpy(fmri_1d).float(),  # [N]
            'embedding': torch.from_numpy(embedding).float()
        }


def create_dataloaders_3d(
    fmri_path: str,
    embeddings_train_path: str,
    embeddings_test_path: str,
    mapping_info: Dict[str, Any],
    batch_size: int = 16,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders that return 3D fMRI volumes.

    Returns both 3D and 1D for evaluation purposes.
    """
    # Load embeddings first to know how many samples we have
    print(f"Loading embeddings...")
    embeddings_train = np.load(embeddings_train_path)
    embeddings_test = np.load(embeddings_test_path)

    print(f"Embeddings: train={len(embeddings_train)}, test={len(embeddings_test)}")

    # Load fMRI data
    print(f"Loading fMRI data from {fmri_path}...")
    with h5py.File(fmri_path, 'r') as f:
        # Load only the samples we have embeddings for
        n_train_samples = len(embeddings_train)
        n_test_samples = len(embeddings_test)

        fmri_train = f['betas'][:n_train_samples]
        fmri_val = f['betas'][n_train_samples:n_train_samples + n_test_samples]

    print(f"fMRI loaded: train={len(fmri_train)}, val={len(fmri_val)}")

    # Verify alignment
    assert len(fmri_train) == len(embeddings_train), \
        f"Train size mismatch: fmri={len(fmri_train)}, embeddings={len(embeddings_train)}"
    assert len(fmri_val) == len(embeddings_test), \
        f"Val size mismatch: fmri={len(fmri_val)}, embeddings={len(embeddings_test)}"

    print(f"Train samples: {len(fmri_train)}, Val samples: {len(fmri_val)}")

    # Create datasets
    train_dataset = fMRI3DDataset(
        fmri_train, embeddings_train,
        mapping_info['coords_compact'],
        mapping_info['compact_shape']
    )

    val_dataset = fMRI3DDataset(
        fmri_val, embeddings_test,
        mapping_info['coords_compact'],
        mapping_info['compact_shape']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


def compute_mean_fmri_3d(train_loader, device, compact_shape):
    """Compute mean 3D fMRI from training data"""
    print("Computing mean 3D fMRI from training data...")
    all_fmri_3d = []
    for batch in tqdm(train_loader, desc='Computing mean'):
        all_fmri_3d.append(batch['fmri_3d'])

    all_fmri_3d = torch.cat(all_fmri_3d, dim=0)
    mean_fmri_3d = all_fmri_3d.mean(dim=0, keepdim=True)  # [1, 1, D, H, W]

    print(f"Mean 3D fMRI: shape={mean_fmri_3d.shape}, "
          f"mean={mean_fmri_3d.mean():.4f}, std={mean_fmri_3d.std():.4f}")

    return mean_fmri_3d.to(device)


def train_epoch(
    model, train_loader, criterion, optimizer, scaler,
    device, mask, coords_compact, grad_clip_val, use_amp
):
    model.train()
    total_loss = 0.0
    total_mse_3d = 0.0
    total_pearson_3d = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        fmri_3d = batch['fmri_3d'].to(device)  # [B, 1, D, H, W]
        embeddings = batch['embedding'].to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                outputs = model(fmri_3d, embeddings, mask, coords_compact)

                loss = criterion(
                    outputs['pred_velocity'],
                    outputs['target_velocity'],
                    outputs['x_pred'],
                    fmri_3d,
                    model.mean_fmri_3d,
                    mask
                )

            scaler.scale(loss).backward()
            if grad_clip_val > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(fmri_3d, embeddings, mask, coords_compact)
            loss = criterion(
                outputs['pred_velocity'],
                outputs['target_velocity'],
                outputs['x_pred'],
                fmri_3d,
                model.mean_fmri_3d,
                mask
            )
            loss.backward()

            if grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            optimizer.step()

        # Metrics
        with torch.no_grad():
            mse = Metrics3D1D.compute_mse_3d(outputs['x_pred'], fmri_3d, mask)
            pearson = Metrics3D1D.compute_pearson_3d(outputs['x_pred'], fmri_3d, mask)

        total_loss += loss.item()
        total_mse_3d += mse
        total_pearson_3d += pearson['mean']
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mse': f'{mse:.4f}',
            'pearson': f'{pearson["mean"]:.4f}'
        })

    return {
        'loss': total_loss / num_batches,
        'mse_3d': total_mse_3d / num_batches,
        'pearson_3d': total_pearson_3d / num_batches
    }


@torch.no_grad()
def validate(
    model, val_loader, criterion, device, mask, coords_compact,
    compact_mask_np, steps=20, cfg_scale=1.0
):
    model.eval()
    total_loss = 0.0
    all_metrics_3d = []
    all_metrics_1d = []
    num_batches = 0

    # For visualization
    first_batch_pred_3d = None
    first_batch_target_3d = None
    first_batch_pred_1d = None
    first_batch_target_1d = None

    pbar = tqdm(val_loader, desc='Validation')
    for batch in pbar:
        fmri_3d = batch['fmri_3d'].to(device)
        fmri_1d = batch['fmri_1d'].numpy()
        embeddings = batch['embedding'].to(device)

        # Generate via sampling
        pred_3d = model.generate(
            embeddings, mask, coords_compact,
            steps=steps, cfg_scale=cfg_scale
        )

        # Convert to 1D for comparison
        pred_1d = fmri_3d_compact_to_1d(
            pred_3d.squeeze(1).cpu().numpy(),
            compact_mask_np
        )

        # Compute loss
        loss = criterion(
            pred_3d - model.mean_fmri_3d,  # velocity
            fmri_3d - model.mean_fmri_3d,  # target velocity
            pred_3d,
            fmri_3d,
            model.mean_fmri_3d,
            mask
        )

        # Compute metrics on both 3D and 1D
        metrics = Metrics3D1D.compute_all_metrics(
            pred_3d, fmri_3d, pred_1d, fmri_1d, mask
        )

        total_loss += loss.item()
        all_metrics_3d.append({
            'mse': metrics['mse_3d'],
            'pearson': metrics['pearson_3d_mean']
        })
        all_metrics_1d.append({
            'mse': metrics['mse_1d'],
            'pearson': metrics['pearson_1d_mean']
        })
        num_batches += 1

        # Save first batch for visualization
        if num_batches == 1:
            first_batch_pred_3d = pred_3d.cpu()
            first_batch_target_3d = fmri_3d.cpu()
            first_batch_pred_1d = pred_1d
            first_batch_target_1d = fmri_1d

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'pearson_3d': f'{metrics["pearson_3d_mean"]:.4f}',
            'pearson_1d': f'{metrics["pearson_1d_mean"]:.4f}'
        })

    # Aggregate metrics
    avg_metrics_3d = {
        'mse': np.mean([m['mse'] for m in all_metrics_3d]),
        'pearson': np.mean([m['pearson'] for m in all_metrics_3d])
    }

    avg_metrics_1d = {
        'mse': np.mean([m['mse'] for m in all_metrics_1d]),
        'pearson': np.mean([m['pearson'] for m in all_metrics_1d])
    }

    return {
        'loss': total_loss / num_batches,
        'metrics_3d': avg_metrics_3d,
        'metrics_1d': avg_metrics_1d,
        'pred_3d': first_batch_pred_3d,
        'target_3d': first_batch_target_3d,
        'pred_1d': first_batch_pred_1d,
        'target_1d': first_batch_target_1d
    }


def visualize_3d_slices(pred_3d, target_3d, mask, epoch, save_dir, sample_idx=0):
    """Visualize middle slices of 3D volume"""
    pred_np = pred_3d[sample_idx, 0].numpy()
    target_np = target_3d[sample_idx, 0].numpy()
    mask_np = mask.cpu().numpy()

    D, H, W = pred_np.shape
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Axial (z-slice)
    axes[0, 0].imshow(target_np[mid_d, :, :], cmap='hot', aspect='auto')
    axes[0, 0].set_title(f'Target - Axial (z={mid_d})')
    axes[1, 0].imshow(pred_np[mid_d, :, :], cmap='hot', aspect='auto')
    axes[1, 0].set_title(f'Predicted - Axial (z={mid_d})')

    # Coronal (y-slice)
    axes[0, 1].imshow(target_np[:, mid_h, :], cmap='hot', aspect='auto')
    axes[0, 1].set_title(f'Target - Coronal (y={mid_h})')
    axes[1, 1].imshow(pred_np[:, mid_h, :], cmap='hot', aspect='auto')
    axes[1, 1].set_title(f'Predicted - Coronal (y={mid_h})')

    # Sagittal (x-slice)
    axes[0, 2].imshow(target_np[:, :, mid_w], cmap='hot', aspect='auto')
    axes[0, 2].set_title(f'Target - Sagittal (x={mid_w})')
    axes[1, 2].imshow(pred_np[:, :, mid_w], cmap='hot', aspect='auto')
    axes[1, 2].set_title(f'Predicted - Sagittal (x={mid_w})')

    plt.suptitle(f'Epoch {epoch}: 3D fMRI Slices (Sample {sample_idx})')
    plt.tight_layout()

    epoch_dir = save_dir / f'epoch_{epoch:04d}'
    epoch_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(epoch_dir / 'slices_3d.png', dpi=150)
    plt.close()


def visualize_1d_comparison(pred_1d, target_1d, epoch, save_dir, num_samples=3):
    """Visualize 1D fMRI comparison"""
    from scipy.stats import pearsonr

    num_samples = min(num_samples, pred_1d.shape[0])

    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        axes[i, 0].plot(target_1d[i], linewidth=0.5, alpha=0.7, label='Target')
        axes[i, 0].set_title(f'Sample {i}: Target 1D fMRI')
        axes[i, 0].legend()

        axes[i, 1].plot(pred_1d[i], linewidth=0.5, alpha=0.7, color='orange', label='Predicted')
        axes[i, 1].set_title(f'Sample {i}: Predicted 1D fMRI')

        corr, _ = pearsonr(pred_1d[i], target_1d[i])
        axes[i, 1].text(0.02, 0.98, f'Pearson: {corr:.4f}',
                       transform=axes[i, 1].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[i, 1].legend()

    plt.suptitle(f'Epoch {epoch}: 1D fMRI Comparison')
    plt.tight_layout()

    epoch_dir = save_dir / f'epoch_{epoch:04d}'
    epoch_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(epoch_dir / 'comparison_1d.png', dpi=150)
    plt.close()


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, save_path)


def main(args):
    config = load_config(args.config)
    set_seed(config['experiment']['seed'])

    # Setup directories
    save_dir = Path(config['experiment']['save_dir']) / config['experiment']['name'] / config['experiment']['version']
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    viz_dir = save_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load 3D mapping
    print("Loading 3D compact mapping...")
    mapping_info = load_3d_mapping(config['data']['mapping_path'])
    compact_shape = mapping_info['compact_shape']
    compact_mask = torch.from_numpy(mapping_info['compact_mask']).to(device)
    coords_compact = torch.from_numpy(mapping_info['coords_compact']).float().to(device)

    print(f"Compact shape: {compact_shape}")
    print(f"Number of brain voxels: {mapping_info['n_voxels']}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders_3d(
        fmri_path=config['data']['fmri_path'],
        embeddings_train_path=config['data']['embeddings_train_path'],
        embeddings_test_path=config['data']['embeddings_test_path'],
        mapping_info=mapping_info,
        batch_size=config['data']['batch_size'],
        num_workers=config['data'].get('num_workers', 4)
    )

    # Compute mean fMRI
    mean_fmri_3d = compute_mean_fmri_3d(train_loader, device, compact_shape)

    # Create model
    print("Creating model...")
    model = Flow3DHybridARM(
        input_dim=config['model']['input_dim'],
        compact_shape=compact_shape,
        hidden_dim=config['model']['hidden_dim'],
        time_dim=config['model']['time_dim'],
        use_graph_layers=config['model'].get('use_graph_layers', True),
        use_mask_conditioning=config['model'].get('use_mask_conditioning', True),
        dropout=config['model']['dropout'],
        cfg_dropout=config['model'].get('cfg_dropout', 0.1),
        flow_sigma=config['model'].get('flow_sigma', 0.0)
    ).to(device)

    # Set mean fMRI
    model.mean_fmri_3d.copy_(mean_fmri_3d)

    # Loss and optimizer
    criterion = FlowMatching3DLoss(
        alpha=config['model']['loss']['alpha'],
        tau=config['model']['loss']['tau'],
        pearson_weight=config['model']['loss']['pearson_weight'],
        mask_weight=config['model']['loss'].get('mask_weight', 2.0)
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 0.05)
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['max_epochs']
    )

    use_amp = config['training'].get('precision', '32') == '16-mixed'
    scaler = GradScaler('cuda') if use_amp else None
    writer = SummaryWriter(save_dir / 'logs') if TENSORBOARD_AVAILABLE else None

    best_pearson = -float('inf')

    print("\nStarting training...")
    for epoch in range(config['training']['max_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['training']['max_epochs']}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, compact_mask, coords_compact,
            config['training'].get('gradient_clip_val', 1.0),
            use_amp
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device,
            compact_mask, coords_compact, mapping_info['compact_mask'],
            steps=config['model'].get('sample_steps', 20),
            cfg_scale=config['model'].get('cfg_scale', 1.0)
        )

        scheduler.step()

        # Log metrics
        if writer is not None:
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('MSE_3D/train', train_metrics['mse_3d'], epoch)
            writer.add_scalar('MSE_3D/val', val_metrics['metrics_3d']['mse'], epoch)
            writer.add_scalar('Pearson_3D/train', train_metrics['pearson_3d'], epoch)
            writer.add_scalar('Pearson_3D/val', val_metrics['metrics_3d']['pearson'], epoch)
            writer.add_scalar('MSE_1D/val', val_metrics['metrics_1d']['mse'], epoch)
            writer.add_scalar('Pearson_1D/val', val_metrics['metrics_1d']['pearson'], epoch)

        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Pearson: {train_metrics['pearson_3d']:.4f}")
        print(f"Val   - Pearson 3D: {val_metrics['metrics_3d']['pearson']:.4f}, "
              f"Pearson 1D: {val_metrics['metrics_1d']['pearson']:.4f}")
        print(f"        (Conversion verified: MSE diff = "
              f"{abs(val_metrics['metrics_3d']['mse'] - val_metrics['metrics_1d']['mse']):.6f})")

        # Save best model
        if val_metrics['metrics_3d']['pearson'] > best_pearson:
            best_pearson = val_metrics['metrics_3d']['pearson']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics, checkpoint_dir / 'best.pt'
            )
            print(f"✓ Saved best model (Pearson: {best_pearson:.4f})")

        # Visualize
        if epoch % config['logging'].get('viz_every_n_epochs', 5) == 0:
            visualize_3d_slices(
                val_metrics['pred_3d'], val_metrics['target_3d'],
                compact_mask, epoch, viz_dir
            )
            visualize_1d_comparison(
                val_metrics['pred_1d'], val_metrics['target_1d'],
                epoch, viz_dir
            )

    if writer is not None:
        writer.close()
    print("\nTraining completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    args = parser.parse_args()
    main(args)

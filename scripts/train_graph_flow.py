"""
Training script for FlowGraphARM (Graph Flow Matching)
Combines Graph Neural Networks with Flow Matching for fMRI prediction.

Key features:
- Point cloud + graph representation of fMRI
- Flow matching with OT path (faster than diffusion)
- Metrics on both point cloud and voxel space
- 3D visualization support
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

# Try plotly for 3D visualization
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. 3D visualizations disabled.")


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ARMNet.flow_graph_model import FlowGraphARM
from ARMNet.loss import FlowMatchingLoss
from data.point_cloud_dataset import create_point_cloud_dataloaders


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


def custom_collate_fn(batch):
    """
    Custom collate for batching graph data.

    Since all samples share the same graph structure (same brain),
    we can batch them efficiently.
    """
    fmri = torch.stack([item['fmri'] for item in batch])
    coords = torch.stack([item['coords'] for item in batch])
    embeddings = torch.stack([item['embedding'] for item in batch])

    # Edge index is shared (same for all samples)
    edge_index = batch[0]['edge_index']

    # Create batch assignment for graph operations
    B = len(batch)
    N = coords.shape[1]
    batch_idx = torch.arange(B).repeat_interleave(N)

    return {
        'fmri': fmri,
        'coords': coords,
        'embedding': embeddings,
        'edge_index': edge_index,
        'batch': batch_idx,
        'image_id': [item['image_id'] for item in batch],
        'subject_id': [item['subject_id'] for item in batch],
    }


def compute_mean_fmri(train_loader, device) -> torch.Tensor:
    """Compute mean fMRI from training data"""
    print("Computing mean fMRI from training data...")
    all_fmri = []
    for batch in train_loader:
        all_fmri.append(batch['fmri'])

    all_fmri = torch.cat(all_fmri, dim=0)
    mean_fmri = all_fmri.mean(dim=0).to(device)

    print(f"Mean fMRI: shape={mean_fmri.shape}, mean={mean_fmri.mean():.4f}, std={mean_fmri.std():.4f}")
    return mean_fmri


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute metrics on both point cloud and voxel space.

    Since we're working with full voxel representation,
    point cloud metrics = voxel metrics.

    Returns:
        dict with MSE, Pearson correlation, per-sample metrics
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    # MSE
    mse = np.mean((pred_np - target_np) ** 2)

    # Pearson correlation (per sample, then average)
    correlations = []
    for i in range(pred_np.shape[0]):
        if pred_np[i].std() > 0 and target_np[i].std() > 0:
            corr, _ = pearsonr(pred_np[i], target_np[i])
            correlations.append(corr)

    pearson_mean = np.mean(correlations) if correlations else 0.0
    pearson_std = np.std(correlations) if correlations else 0.0

    # Peak correlation (high activation voxels only)
    peak_threshold = 0.5  # STD threshold
    peak_correlations = []
    for i in range(pred_np.shape[0]):
        mask = np.abs(target_np[i] - target_np[i].mean()) > peak_threshold * target_np[i].std()
        if mask.sum() > 10:  # Need enough voxels
            pred_peak = pred_np[i][mask]
            target_peak = target_np[i][mask]
            if pred_peak.std() > 0 and target_peak.std() > 0:
                corr, _ = pearsonr(pred_peak, target_peak)
                peak_correlations.append(corr)

    peak_pearson = np.mean(peak_correlations) if peak_correlations else 0.0

    return {
        'mse': mse,
        'pearson_mean': pearson_mean,
        'pearson_std': pearson_std,
        'peak_pearson': peak_pearson,
        'num_samples': pred_np.shape[0],
    }


def train_epoch(model, train_loader, criterion, optimizer, scaler, device,
                grad_clip_val, use_amp):
    """Training loop for one epoch"""
    model.train()
    total_loss = 0.0
    total_velocity_loss = 0.0
    total_peak_loss = 0.0
    total_pearson = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        fmri = batch['fmri'].to(device)
        coords = batch['coords'].to(device)
        embeddings = batch['embedding'].to(device)
        edge_index = batch['edge_index'].to(device)
        batch_idx = batch['batch'].to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                outputs = model(fmri, coords, edge_index, embeddings, batch_idx)

                # Compute loss
                v_pred = outputs['pred_velocity']
                v_target = outputs['target_velocity']
                x_pred = outputs['x_pred']
                x_target = fmri
                mean_fmri = model.mean_fmri.unsqueeze(0).expand(fmri.size(0), -1)

                loss = criterion(v_pred, v_target, x_pred, x_target, mean_fmri)

                # Individual losses for logging
                velocity_loss = nn.functional.mse_loss(v_pred, v_target)
                peak_loss = criterion.peak_loss_fn(x_pred, x_target, mean_fmri)

            scaler.scale(loss).backward()
            if grad_clip_val > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(fmri, coords, edge_index, embeddings, batch_idx)

            v_pred = outputs['pred_velocity']
            v_target = outputs['target_velocity']
            x_pred = outputs['x_pred']
            x_target = fmri
            mean_fmri = model.mean_fmri.unsqueeze(0).expand(fmri.size(0), -1)

            loss = criterion(v_pred, v_target, x_pred, x_target, mean_fmri)
            velocity_loss = nn.functional.mse_loss(v_pred, v_target)
            peak_loss = criterion.peak_loss_fn(x_pred, x_target, mean_fmri)

            loss.backward()
            if grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            optimizer.step()

        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(x_pred, x_target)

        total_loss += loss.item()
        total_velocity_loss += velocity_loss.item()
        total_peak_loss += peak_loss.item()
        total_pearson += metrics['pearson_mean']
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'v_loss': f'{velocity_loss.item():.4f}',
            'pearson': f'{metrics["pearson_mean"]:.4f}'
        })

    return {
        'loss': total_loss / num_batches,
        'velocity_loss': total_velocity_loss / num_batches,
        'peak_loss': total_peak_loss / num_batches,
        'pearson': total_pearson / num_batches,
    }


@torch.no_grad()
def validate(model, val_loader, criterion, device, steps=20):
    """Validation loop with flow sampling"""
    model.eval()
    total_loss = 0.0
    total_pearson = 0.0
    total_peak_pearson = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []
    all_coords = []

    pbar = tqdm(val_loader, desc='Validation')
    for batch in pbar:
        fmri = batch['fmri'].to(device)
        coords = batch['coords'].to(device)
        embeddings = batch['embedding'].to(device)
        edge_index = batch['edge_index'].to(device)
        batch_idx = batch['batch'].to(device)

        # Generate via flow sampling
        pred = model.generate(
            coords=coords,
            edge_index=edge_index,
            vis_feat=embeddings,
            batch=batch_idx,
            steps=steps,
            cfg_scale=1.0  # No CFG during validation for fair comparison
        )

        # Compute loss
        mean_fmri = model.mean_fmri.unsqueeze(0).expand(fmri.size(0), -1)
        loss = criterion.peak_loss_fn(pred, fmri, mean_fmri)

        # Compute metrics
        metrics = compute_metrics(pred, fmri)

        total_loss += loss.item()
        total_pearson += metrics['pearson_mean']
        total_peak_pearson += metrics['peak_pearson']
        num_batches += 1

        # Save first batch for visualization
        if num_batches == 1:
            all_preds.append(pred.cpu())
            all_targets.append(fmri.cpu())
            all_coords.append(coords.cpu())

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'pearson': f'{metrics["pearson_mean"]:.4f}',
            'peak_r': f'{metrics["peak_pearson"]:.4f}'
        })

    return {
        'loss': total_loss / num_batches,
        'pearson': total_pearson / num_batches,
        'peak_pearson': total_peak_pearson / num_batches,
        'preds': all_preds[0] if all_preds else None,
        'targets': all_targets[0] if all_targets else None,
        'coords': all_coords[0] if all_coords else None,
    }


def visualize_predictions(preds, targets, coords, epoch, save_dir, num_samples=2):
    """
    Visualize predictions in both 1D and 3D.

    Creates:
    - 1D line plots (pred vs target)
    - 3D point cloud visualizations (if plotly available)
    """
    if preds is None or targets is None:
        return

    preds_np = preds.numpy()
    targets_np = targets.numpy()

    num_samples = min(num_samples, preds_np.shape[0])
    epoch_dir = save_dir / f'epoch_{epoch:04d}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    # 1. 1D Line plots
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Ground truth
        axes[i, 0].plot(targets_np[i], linewidth=0.5, alpha=0.7, label='Ground Truth')
        axes[i, 0].set_title(f'Sample {i+1}: Ground Truth')
        axes[i, 0].set_xlabel('Voxel Index')
        axes[i, 0].set_ylabel('fMRI Activation')

        # Prediction
        axes[i, 1].plot(preds_np[i], linewidth=0.5, alpha=0.7, color='orange', label='Prediction')
        axes[i, 1].set_title(f'Sample {i+1}: Prediction (Flow Sampled)')
        axes[i, 1].set_xlabel('Voxel Index')
        axes[i, 1].set_ylabel('fMRI Activation')

        # Compute correlation
        corr, _ = pearsonr(preds_np[i], targets_np[i])
        axes[i, 1].text(0.02, 0.98, f'Pearson: {corr:.4f}',
                       transform=axes[i, 1].transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Epoch {epoch}: Graph Flow Matching Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig(epoch_dir / 'predictions_1d.png', dpi=150)
    plt.close()

    # 2. 3D Point cloud visualization
    if PLOTLY_AVAILABLE and coords is not None:
        coords_np = coords.numpy()

        for i in range(num_samples):
            # Ground truth point cloud
            fig_gt = go.Figure(data=[go.Scatter3d(
                x=coords_np[i, :, 0],
                y=coords_np[i, :, 1],
                z=coords_np[i, :, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=targets_np[i],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="fMRI"),
                ),
                text=[f'Voxel {j}: {targets_np[i, j]:.3f}' for j in range(len(targets_np[i]))],
                hoverinfo='text'
            )])
            fig_gt.update_layout(
                title=f'Sample {i+1}: Ground Truth Point Cloud',
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                width=800, height=800
            )
            fig_gt.write_html(str(epoch_dir / f'pointcloud_gt_sample{i+1}.html'))

            # Prediction point cloud
            fig_pred = go.Figure(data=[go.Scatter3d(
                x=coords_np[i, :, 0],
                y=coords_np[i, :, 1],
                z=coords_np[i, :, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=preds_np[i],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="fMRI"),
                ),
                text=[f'Voxel {j}: {preds_np[i, j]:.3f}' for j in range(len(preds_np[i]))],
                hoverinfo='text'
            )])

            corr, _ = pearsonr(preds_np[i], targets_np[i])
            fig_pred.update_layout(
                title=f'Sample {i+1}: Prediction (Pearson: {corr:.4f})',
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                width=800, height=800
            )
            fig_pred.write_html(str(epoch_dir / f'pointcloud_pred_sample{i+1}.html'))

    print(f"Visualizations saved to {epoch_dir}")


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, save_path)
    print(f"Checkpoint saved: {save_path}")


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

    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders with custom collate
    print("\nLoading datasets...")
    train_loader, val_loader = create_point_cloud_dataloaders(
        datalist_path=config['data']['datalist_path'],
        fmri_path=config['data']['fmri_path'],
        train_embeddings_path=config['data']['train_embeddings_path'],
        test_embeddings_path=config['data']['test_embeddings_path'],
        point_cloud_path=config['data']['point_cloud_path'],
        subjects=config['data']['subjects'],
        batch_size=config['data']['batch_size'],
        average_trials_train=config['data'].get('average_trials_train', False),
        average_trials_val=config['data'].get('average_trials_val', True),
        augment_noise_train=config['data'].get('augment_noise_train', False),
        noise_std=config['data'].get('noise_std', 0.1),
        augment_position_jitter_train=config['data'].get('augment_position_jitter', False),
        jitter_std=config['data'].get('jitter_std', 0.01),
        parcel_labels_path=config['data'].get('parcel_labels_path'),
        apply_zscore_train=config['data'].get('apply_zscore_train', False),
        apply_zscore_val=config['data'].get('apply_zscore_val', False),
        num_workers=config['data'].get('num_workers', 4),
        collate_fn=custom_collate_fn,
    )

    # Compute mean fMRI
    mean_fmri = compute_mean_fmri(train_loader, device)

    # Create model
    print("\nInitializing model...")
    model = FlowGraphARM(
        input_dim=config['model']['input_dim'],
        output_dim=config['model']['output_dim'],
        hidden_dim=config['model']['hidden_dim'],
        time_dim=config['model'].get('time_dim', 256),
        num_heads=config['model']['num_heads'],
        gnn_type=config['model']['gnn_type'],
        dropout=config['model']['dropout'],
        cfg_dropout=config['model']['flow'].get('cfg_dropout', 0.1),
        flow_sigma=config['model']['flow'].get('sigma', 0.0),
    ).to(device)

    # Set mean fMRI
    model.mean_fmri.copy_(mean_fmri)

    # Loss function
    criterion = FlowMatchingLoss(
        alpha=config['model']['loss']['alpha'],
        tau=config['model']['loss']['tau'],
        pearson_weight=config['model']['loss']['pearson_weight']
    )

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['max_epochs'],
        eta_min=config['training'].get('lr_min', 1e-6)
    )

    # Mixed precision
    use_amp = config['training'].get('precision', '32') == '16-mixed'
    scaler = GradScaler('cuda') if use_amp else None

    # Tensorboard
    writer = SummaryWriter(save_dir / 'logs')

    # Training loop
    print(f"\nStarting training for {config['training']['max_epochs']} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    best_pearson = -float('inf')
    patience_counter = 0
    patience = config['validation'].get('early_stopping_patience', 20)

    for epoch in range(config['training']['max_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['training']['max_epochs']}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            config['training'].get('gradient_clip_val', 1.0), use_amp
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device,
            steps=config['model']['flow'].get('sample_steps', 20)
        )

        # Scheduler step
        scheduler.step()

        # Log metrics
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Pearson/train', train_metrics['pearson'], epoch)
        writer.add_scalar('Pearson/val', val_metrics['pearson'], epoch)
        writer.add_scalar('Pearson/val_peak', val_metrics['peak_pearson'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | Train Pearson: {train_metrics['pearson']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Pearson: {val_metrics['pearson']:.4f} | Peak Pearson: {val_metrics['peak_pearson']:.4f}")

        # Save best model
        if val_metrics['pearson'] > best_pearson:
            best_pearson = val_metrics['pearson']
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, checkpoint_dir / 'best.pt')
            print(f"✓ New best model! Pearson: {best_pearson:.4f}")
        else:
            patience_counter += 1

        # Visualize
        if epoch % config['validation'].get('viz_every_n_epochs', 1) == 0:
            visualize_predictions(
                val_metrics['preds'],
                val_metrics['targets'],
                val_metrics['coords'],
                epoch,
                viz_dir,
                num_samples=config['validation'].get('viz_num_samples', 2)
            )

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break

    writer.close()
    print(f"\nTraining completed! Best Pearson: {best_pearson:.4f}")
    print(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Graph Flow Matching ARM')
    parser.add_argument('--config', type=str, default='configs/train_graph_flow.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    main(args)

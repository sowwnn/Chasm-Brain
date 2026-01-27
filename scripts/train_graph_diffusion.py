"""
Training script for Graph Diffusion ARM (DiffusionGraphARM)

Features:
- Point cloud + graph neural network architecture
- Diffusion-based fMRI generation
- 1D metrics computation (Pearson, MSE, MAE)
- 3D point cloud visualization during validation
- Mixed precision training with AMP
- TensorBoard logging
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

# Plotly for 3D visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ARMNet.diffusion_graph_model import DiffusionGraphARM
from ARMNet.loss import DiffusionPeakFocusedLoss
from data.point_cloud_dataset import create_point_cloud_dataloaders


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
    for batch in tqdm(train_loader, desc="Computing mean"):
        all_fmri.append(batch['fmri'])

    all_fmri = torch.cat(all_fmri, dim=0)
    mean_fmri = all_fmri.mean(dim=0).to(device)

    print(f"Mean fMRI: shape={mean_fmri.shape}, mean={mean_fmri.mean():.4f}, std={mean_fmri.std():.4f}")
    return mean_fmri


def create_model(config: dict, mean_fmri: torch.Tensor, device) -> nn.Module:
    """Create DiffusionGraphARM model."""
    model_cfg = config['model']

    model = DiffusionGraphARM(
        input_dim=model_cfg['input_dim'],
        output_dim=model_cfg['output_dim'],
        hidden_dim=model_cfg['hidden_dim'],
        time_dim=model_cfg.get('time_dim', 256),
        num_heads=model_cfg['num_heads'],
        gnn_type=model_cfg.get('gnn_type', 'gat'),
        dropout=model_cfg['dropout'],
        cfg_dropout=model_cfg.get('diffusion', {}).get('cfg_dropout', 0.1),
        timesteps=model_cfg.get('diffusion', {}).get('timesteps', 1000),
        beta_schedule=model_cfg.get('diffusion', {}).get('beta_schedule', 'cosine'),
    )

    # Load mean fMRI
    model.mean_fmri.copy_(mean_fmri)

    # Move model to device
    model = model.to(device)

    # Move diffusion manager tensors to device
    model.diffusion.to(device)

    return model


def create_loss_fn(config: dict):
    """Create loss function."""
    loss_cfg = config['model']['loss']

    return DiffusionPeakFocusedLoss(
        alpha=loss_cfg.get('alpha', 2.0),
        tau=loss_cfg.get('tau', 0.3),
        pearson_weight=loss_cfg.get('pearson_weight', 0.5),
        diversity_weight=loss_cfg.get('diversity_weight', 3.0),
        l1_weight=loss_cfg.get('l1_weight', 0.05),
    )


def collate_graph_batch(batch_list):
    """
    Custom collate function for graph data.

    Handles batching of point cloud + graph data.
    Note: edge_index is shared across batch, so we don't need to offset indices.
    """
    fmri = torch.stack([item['fmri'] for item in batch_list])  # [B, N]
    coords = torch.stack([item['coords'] for item in batch_list])  # [B, N, 3]
    embedding = torch.stack([item['embedding'] for item in batch_list])  # [B, 768]
    edge_index = batch_list[0]['edge_index']  # Shared across batch
    image_id = torch.tensor([item['image_id'] for item in batch_list])
    subject_id = torch.tensor([item['subject_id'] for item in batch_list])

    # Create batch assignment for nodes
    B, N = fmri.shape
    batch = torch.arange(B).repeat_interleave(N)  # [B*N]

    result = {
        'fmri': fmri,
        'coords': coords,
        'edge_index': edge_index,
        'embedding': embedding,
        'batch': batch,
        'image_id': image_id,
        'subject_id': subject_id,
    }

    # Add ROI labels if available
    if 'roi_labels' in batch_list[0]:
        result['roi_labels'] = batch_list[0]['roi_labels']

    return result


def compute_metrics_1d(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute 1D metrics: Pearson, MSE, MAE.

    Args:
        pred: [B, N] predicted fMRI
        target: [B, N] target fMRI

    Returns:
        dict with 'pearson', 'mse', 'mae'
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    # Pearson correlation (per-sample then average)
    correlations = []
    for i in range(pred_np.shape[0]):
        if pred_np[i].std() > 0 and target_np[i].std() > 0:
            corr, _ = pearsonr(pred_np[i], target_np[i])
            correlations.append(corr)
        else:
            correlations.append(0.0)

    pearson = np.mean(correlations)

    # MSE and MAE
    mse = ((pred - target) ** 2).mean().item()
    mae = (pred - target).abs().mean().item()

    return {
        'pearson': pearson,
        'mse': mse,
        'mae': mae,
        'per_sample_pearson': correlations,  # For visualization
    }


def visualize_point_cloud_predictions(
    coords,
    fmri_gt,
    fmri_pred,
    save_path,
    roi_labels=None,
    sample_indices=None,
    num_samples=2
):
    """
    Create interactive 3D point cloud visualization.

    coords: [B, N, 3] or [N, 3] coordinates
    fmri_gt: [B, N] or [N] ground truth fMRI
    fmri_pred: [B, N] or [N] predicted fMRI
    save_path: Path to save HTML file
    roi_labels: [N] ROI labels (optional)
    sample_indices: [num_samples] which samples to visualize
    num_samples: Number of samples to visualize
    """
    # Convert to numpy
    if coords.dim() == 3:
        coords = coords.cpu().numpy()  # [B, N, 3]
        fmri_gt = fmri_gt.cpu().numpy()  # [B, N]
        fmri_pred = fmri_pred.cpu().numpy()  # [B, N]
        B, N, _ = coords.shape
    else:
        coords = coords.cpu().numpy()  # [N, 3]
        fmri_gt = fmri_gt.cpu().numpy()  # [N]
        fmri_pred = fmri_pred.cpu().numpy()  # [N]
        B = 1
        N = coords.shape[0]
        coords = coords[None, ...]  # [1, N, 3]
        fmri_gt = fmri_gt[None, ...]  # [1, N]
        fmri_pred = fmri_pred[None, ...]  # [1, N]

    if sample_indices is None:
        sample_indices = list(range(min(num_samples, B)))

    # Create subplot titles ahead of time
    subplot_titles = []
    for sample_idx in sample_indices:
        subplot_titles.append(f"Sample {sample_idx}: Ground Truth")
        subplot_titles.append(f"Sample {sample_idx}: Prediction")

    # Create subplots: rows = samples, cols = 2 (GT vs Pred)
    fig = make_subplots(
        rows=len(sample_indices),
        cols=2,
        subplot_titles=subplot_titles,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}] for _ in sample_indices],
        vertical_spacing=0.05,
        horizontal_spacing=0.02,
    )

    # Color scale limits (based on typical fMRI range after preprocessing)
    vmin, vmax = -2.0, 2.0

    for idx, sample_idx in enumerate(sample_indices):
        row = idx + 1

        # Get data for this sample
        coords_sample = coords[sample_idx]  # [N, 3]
        gt_sample = fmri_gt[sample_idx]  # [N]
        pred_sample = fmri_pred[sample_idx]  # [N]

        # Compute correlation for this sample
        if gt_sample.std() > 0 and pred_sample.std() > 0:
            corr, _ = pearsonr(gt_sample, pred_sample)
        else:
            corr = 0.0

        # Ground truth
        fig.add_trace(
            go.Scatter3d(
                x=coords_sample[:, 0],
                y=coords_sample[:, 1],
                z=coords_sample[:, 2],
                mode='markers',
                marker=dict(
                    size=1.5,
                    color=gt_sample,
                    colorscale='RdBu_r',
                    cmin=vmin,
                    cmax=vmax,
                    showscale=(row == 1),  # Show colorbar only for first row
                    colorbar=dict(title="fMRI", x=0.45) if row == 1 else None,
                ),
                name=f'Sample {sample_idx} GT',
                showlegend=False,
            ),
            row=row, col=1
        )

        # Prediction
        fig.add_trace(
            go.Scatter3d(
                x=coords_sample[:, 0],
                y=coords_sample[:, 1],
                z=coords_sample[:, 2],
                mode='markers',
                marker=dict(
                    size=1.5,
                    color=pred_sample,
                    colorscale='RdBu_r',
                    cmin=vmin,
                    cmax=vmax,
                    showscale=(row == 1),  # Show colorbar only for first row
                    colorbar=dict(title="fMRI", x=1.0) if row == 1 else None,
                ),
                name=f'Sample {sample_idx} Pred',
                showlegend=False,
            ),
            row=row, col=2
        )

        # Update prediction title with correlation
        try:
            if fig.layout.annotations and len(fig.layout.annotations) > idx * 2 + 1:
                fig.layout.annotations[idx * 2 + 1].update(text=f"Sample {sample_idx}: Prediction (r={corr:.3f})")
        except (IndexError, AttributeError, TypeError):
            # If annotations update fails, skip
            pass

    # Update layout
    fig.update_layout(
        title="3D Point Cloud: Ground Truth vs Prediction",
        height=400 * len(sample_indices),
        showlegend=False,
    )

    # Save as HTML
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))

    print(f"  Saved 3D visualization: {save_path.name}")


def visualize_1d_signals(
    fmri_gt,
    fmri_pred,
    save_path,
    num_samples=3
):
    """
    Visualize 1D fMRI signals: GT vs Pred.

    fmri_gt: [B, N] ground truth
    fmri_pred: [B, N] prediction
    save_path: Path to save PNG file
    num_samples: Number of samples to visualize
    """
    fmri_gt = fmri_gt.cpu().numpy()
    fmri_pred = fmri_pred.cpu().numpy()

    B, N = fmri_gt.shape
    num_samples = min(num_samples, B)

    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        gt = fmri_gt[i]
        pred = fmri_pred[i]

        # Compute correlation
        if gt.std() > 0 and pred.std() > 0:
            corr, _ = pearsonr(gt, pred)
        else:
            corr = 0.0

        axes[i].plot(gt, label='Ground Truth', alpha=0.7, linewidth=0.5)
        axes[i].plot(pred, label='Prediction', alpha=0.7, linewidth=0.5)
        axes[i].set_title(f'Sample {i} - Pearson: {corr:.3f}')
        axes[i].set_xlabel('Voxel Index')
        axes[i].set_ylabel('fMRI Value')
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved 1D signal plot: {save_path.name}")


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, grad_clip_val, use_amp, accumulate_grad_batches=1):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_pearson = 0.0
    total_mse = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        clean_fmri = batch['fmri'].to(device)  # [B, N]
        coords = batch['coords'].to(device)  # [B, N, 3]
        edge_index = batch['edge_index'].to(device)  # [2, E]
        embedding = batch['embedding'].to(device)  # [B, 768]
        batch_assignment = batch['batch'].to(device)  # [B*N]

        # Only zero gradients at the start of accumulation cycle
        if batch_idx % accumulate_grad_batches == 0:
            optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                # Forward pass
                output = model(clean_fmri, coords, edge_index, embedding, batch_assignment)
                pred_noise = output['pred_noise']
                noise = output['noise']

                # Compute loss
                loss, loss_dict = criterion(pred_noise, noise, None)  # No mean_fmri needed

                # Scale loss by accumulation steps
                loss = loss / accumulate_grad_batches

            scaler.scale(loss).backward()

            # Only step optimizer after accumulating enough gradients
            if (batch_idx + 1) % accumulate_grad_batches == 0:
                if grad_clip_val > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)

                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(clean_fmri, coords, edge_index, embedding, batch_assignment)
            pred_noise = output['pred_noise']
            noise = output['noise']

            loss, loss_dict = criterion(pred_noise, noise, None)

            # Scale loss by accumulation steps
            loss = loss / accumulate_grad_batches

            loss.backward()

            # Only step optimizer after accumulating enough gradients
            if (batch_idx + 1) % accumulate_grad_batches == 0:
                if grad_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)

                optimizer.step()

        # Compute metrics on predicted noise vs actual noise
        metrics = compute_metrics_1d(pred_noise, noise)

        # Accumulate metrics (multiply back by accumulate steps for proper averaging)
        total_loss += loss.item() * accumulate_grad_batches
        total_pearson += metrics['pearson']
        total_mse += metrics['mse']
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item() * accumulate_grad_batches:.4f}",
            'pearson': f"{metrics['pearson']:.3f}",
            'mse': f"{metrics['mse']:.4f}"
        })

        # Periodically clear CUDA cache to prevent fragmentation
        if (batch_idx + 1) % (accumulate_grad_batches * 10) == 0:
            torch.cuda.empty_cache()

    return {
        'loss': total_loss / num_batches,
        'pearson': total_pearson / num_batches,
        'mse': total_mse / num_batches,
    }


@torch.no_grad()
def validate(model, val_loader, criterion, device, visualize_dir=None, epoch=0, viz_num_samples=2, visualize_pointcloud=True, num_gen_batches=5):
    """
    Validate model.

    Args:
        num_gen_batches: Number of batches to generate fMRI for evaluation (default: 5)
    """
    model.eval()
    total_loss = 0.0
    total_pearson = 0.0
    total_mse = 0.0
    total_mae = 0.0
    num_gen_batches_processed = 0

    # Store first batch for visualization
    first_batch_data = None

    pbar = tqdm(val_loader, desc='Validation')
    for i, batch in enumerate(pbar):
        # Move data to device
        clean_fmri = batch['fmri'].to(device)
        coords = batch['coords'].to(device)
        edge_index = batch['edge_index'].to(device)
        embedding = batch['embedding'].to(device)
        batch_assignment = batch['batch'].to(device)

        # Forward pass (for loss computation only)
        output = model(clean_fmri, coords, edge_index, embedding, batch_assignment)
        pred_noise = output['pred_noise']
        noise = output['noise']

        # Compute loss
        loss, loss_dict = criterion(pred_noise, noise, None)

        # Generate clean fMRI prediction using DDIM sampling
        # Only generate for first N batches to save time while getting accurate metrics
        if i < num_gen_batches:
            fmri_generated = model.generate(
                coords, edge_index, embedding, batch_assignment,
                method='ddim', ddim_steps=25, cfg_scale=4.0
            )
            # Compute metrics on reconstructed fMRI vs ground truth
            metrics = compute_metrics_1d(fmri_generated, clean_fmri)

            total_loss += loss.item()
            total_pearson += metrics['pearson']
            total_mse += metrics['mse']
            total_mae += metrics.get('mae', 0.0)
            num_gen_batches_processed += 1
        else:
            # Skip generation for remaining batches
            fmri_generated = None
            metrics = None

        # Store first batch for visualization
        if i == 0 and visualize_dir is not None:
            first_batch_data = {
                'clean_fmri': clean_fmri.cpu(),
                'fmri_generated': fmri_generated.cpu() if fmri_generated is not None else None,
                'coords': coords.cpu(),
                'roi_labels': batch.get('roi_labels', None),
            }

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'pearson': f"{metrics['pearson']:.3f}",
            'mse': f"{metrics['mse']:.4f}"
        })

    # Visualize first batch
    if first_batch_data is not None and visualize_dir is not None and first_batch_data['fmri_generated'] is not None:
        viz_dir = Path(visualize_dir) / f"epoch_{epoch:04d}"
        viz_dir.mkdir(parents=True, exist_ok=True)

        # 1D signal visualization
        visualize_1d_signals(
            first_batch_data['clean_fmri'],
            first_batch_data['fmri_generated'],
            viz_dir / 'fmri_1d_signals.png',
            num_samples=min(viz_num_samples, first_batch_data['clean_fmri'].shape[0])
        )

        # 3D point cloud visualization
        if visualize_pointcloud:
            visualize_point_cloud_predictions(
                first_batch_data['coords'],
                first_batch_data['clean_fmri'],
                first_batch_data['fmri_generated'],
                viz_dir / 'pointcloud_3d.html',
                roi_labels=first_batch_data['roi_labels'],
                num_samples=min(viz_num_samples, first_batch_data['clean_fmri'].shape[0])
            )

    return {
        'loss': total_loss / num_batches,
        'pearson': total_pearson / num_batches,
        'mse': total_mse / num_batches,
        'mae': total_mae / num_batches,
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics,
    }, save_path)


def main(config_path: str):
    """Main training function."""
    # Load config
    config = load_config(config_path)
    print("=" * 60)
    print("Graph Diffusion ARM Training")
    print("=" * 60)
    print(f"Config: {config_path}")

    # Set seed
    set_seed(config['experiment']['seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create save directory
    save_dir = Path(config['experiment']['save_dir']) / config['experiment']['name'] / config['experiment']['version']
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save directory: {save_dir}")

    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Create dataloaders
    print("\nLoading data...")
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
        parcel_labels_path=config['data'].get('parcel_labels_path', None),
        apply_zscore_train=config['data'].get('apply_zscore_train', False),
        apply_zscore_val=config['data'].get('apply_zscore_val', False),
        num_workers=config['data'].get('num_workers', 4),
    )

    # Use custom collate function
    train_loader.collate_fn = collate_graph_batch
    val_loader.collate_fn = collate_graph_batch

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Compute mean fMRI
    mean_fmri = compute_mean_fmri(train_loader, device)

    # Create model
    print("\nCreating model...")
    model = create_model(config, mean_fmri, device)
    epoch_st = 0

    if config['model'].get('resume_path', ''):
        resume_path = config['model']['resume_path']
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_st = checkpoint.get('epoch', 0)
        print("Checkpoint loaded.")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Create loss function
    criterion = create_loss_fn(config)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['max_epochs'],
        eta_min=config['training'].get('lr_min', 1e-6)
    )

    # AMP
    use_amp = config['training'].get('precision', 'fp32') == '16-mixed'
    scaler = GradScaler('cuda') if use_amp else None
    print(f"Mixed precision: {use_amp}")

    # TensorBoard
    writer = SummaryWriter(save_dir / 'logs')

    # Training loop
    best_val_pearson = -float('inf')
    patience_counter = 0
    patience = config['validation']['early_stopping_patience']

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(epoch_st, config['training']['max_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['max_epochs']}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            config['training']['gradient_clip_val'], use_amp,
            config['training'].get('accumulate_grad_batches', 1)
        )

        # Validate
        visualize = (epoch % config['validation'].get('viz_every_n_epochs', 1) == 0)
        val_metrics = validate(
            model, val_loader, criterion, device,
            visualize_dir=save_dir / 'visualizations' if visualize else None,
            epoch=epoch,
            viz_num_samples=config['validation'].get('viz_num_samples', 2),
            visualize_pointcloud=config['validation'].get('visualize_pointcloud', True)
        )

        # Scheduler step
        scheduler.step()

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Pearson/train', train_metrics['pearson'], epoch)
        writer.add_scalar('Pearson/val', val_metrics['pearson'], epoch)
        writer.add_scalar('MSE/train', train_metrics['mse'], epoch)
        writer.add_scalar('MSE/val', val_metrics['mse'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Pearson: {train_metrics['pearson']:.3f}, MSE: {train_metrics['mse']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Pearson: {val_metrics['pearson']:.3f}, MSE: {val_metrics['mse']:.4f}")

        # Save checkpoint
        checkpoint_dir = save_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        # Save last checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, checkpoint_dir / 'last.pt')

        # Save best checkpoint
        if val_metrics['pearson'] > best_val_pearson:
            best_val_pearson = val_metrics['pearson']
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, checkpoint_dir / 'best.pt')
            print(f"✓ New best model! Pearson: {best_val_pearson:.3f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch + 1} epochs (patience: {patience})")
            break

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation Pearson: {best_val_pearson:.3f}")
    print(f"Model saved to: {save_dir}")
    print("=" * 60)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Graph Diffusion ARM')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    main(args.config)

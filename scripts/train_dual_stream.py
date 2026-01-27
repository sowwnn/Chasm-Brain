"""
Training script for Dual-Stream CNN (3D mask-aware + 1D voxel stream)

This script mirrors conventions used in other training scripts in the repo.
"""

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ARMNet.dual_stream_cnn_model import DualStreamCNN
from ARMNet.loss import PeakFocusedLoss
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from data.neuroflux_dataset import create_dataloaders
from data.parcel_utils import ParcelMapper


def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_mapping(mapping_path: str):
    data = np.load(mapping_path)
    return {
        'compact_shape': tuple(data['compact_shape']),
        'compact_mask': data['compact_mask'],
        'coords_compact': data['coords_compact'],
        'n_voxels': int(data['n_voxels'])
    }


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mean_fmri_1d(train_loader, device):
    all_fmri = []
    print('Computing mean fMRI (1D) from training data...')
    for batch in tqdm(train_loader, desc="Computing Mean"):
        all_fmri.append(batch['fmri'])
    all_fmri = torch.cat(all_fmri, dim=0)
    mean_1d = all_fmri.mean(dim=0, keepdim=True).to(device)
    print(f"Mean 1D fmri shape: {mean_1d.shape}")
    return mean_1d


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, mapping, grad_clip, use_amp, writer, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        fmri_1d = batch['fmri'].to(device)  # [B, N]
        embeddings = batch['embedding'].to(device)

        # Prepare inputs
        mask = torch.from_numpy(mapping['compact_mask']).to(device)
        coords = torch.from_numpy(mapping['coords_compact']).long().to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                # Forward pass - only visual features needed
                outputs = model(embeddings, mask, coords)
                
                # Main loss (PeakFocusedLoss)
                loss = criterion(outputs['pred_fmri_1d'], fmri_1d, model.mean_fmri_1d)
                
                # Auxiliary loss for coarse predictor (Deep Supervision)
                if 'coarse_1d' in outputs:
                    loss += 0.5 * nn.functional.mse_loss(outputs['coarse_1d'], fmri_1d)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(embeddings, mask, coords)
            loss = criterion(outputs['pred_fmri_1d'], fmri_1d, model.mean_fmri_1d)
            
            if 'coarse_1d' in outputs:
                loss += 0.5 * nn.functional.mse_loss(outputs['coarse_1d'], fmri_1d)
                
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    writer.add_scalar('Loss/train', avg_loss, epoch)
    return avg_loss


@torch.no_grad()
def validate(model, val_loader, criterion, device, mapping, writer, epoch, config=None):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []
    all_coarse = []

    # Randomly select a batch to visualize
    viz_batch_idx = np.random.randint(0, len(val_loader))
    viz_data = None

    for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
        fmri_1d = batch['fmri'].to(device)
        embeddings = batch['embedding'].to(device)

        mask = torch.from_numpy(mapping['compact_mask']).to(device)
        coords = torch.from_numpy(mapping['coords_compact']).long().to(device)

        outputs = model(embeddings, mask, coords)
        
        # PeakFocusedLoss requires mean_fmri for calculation
        loss = criterion(outputs['pred_fmri_1d'], fmri_1d, model.mean_fmri_1d)

        if 'coarse_1d' in outputs:
            loss += 0.5 * nn.functional.mse_loss(outputs['coarse_1d'], fmri_1d)

        total_loss += loss.item()
        num_batches += 1
        
        # Collect for metrics
        all_preds.append(outputs['pred_fmri_1d'].cpu().numpy())
        all_targets.append(fmri_1d.cpu().numpy())
        if 'coarse_1d' in outputs:
            all_coarse.append(outputs['coarse_1d'].cpu().numpy())

        # Save visualization data
        if batch_idx == viz_batch_idx:
            viz_data = {
                'pred': outputs['pred_fmri_1d'].cpu().numpy(),
                'target': fmri_1d.cpu().numpy(),
                'coarse': outputs['coarse_1d'].cpu().numpy() if 'coarse_1d' in outputs else None
            }

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Compute Pearson Correlation
    all_preds = np.concatenate(all_preds, axis=0) # [N_samples, N_voxels]
    all_targets = np.concatenate(all_targets, axis=0)
    if all_coarse:
        all_coarse = np.concatenate(all_coarse, axis=0)
    
    correlations = []
    for i in range(len(all_preds)):
        corr, _ = pearsonr(all_preds[i], all_targets[i])
        correlations.append(corr)
    mean_pearson = np.mean(correlations)

    writer.add_scalar('Loss/val', avg_loss, epoch)
    writer.add_scalar('Metrics/Pearson_val', mean_pearson, epoch)
    
    # Enhanced Visualization
    if config and config.get('logging', {}).get('visualize', True):
        viz_every = config.get('logging', {}).get('viz_every_n_epochs', 1)
        if (epoch + 1) % viz_every == 0 and viz_data is not None:
            # Create visualization directory
            from pathlib import Path
            save_dir = Path(config['experiment']['save_dir']) / config['experiment']['name'] / config['experiment']['version']
            viz_dir = save_dir / 'visualizations' / f'epoch_{epoch+1:03d}'
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            create_validation_visualizations(
                viz_data, 
                all_preds, 
                all_targets, 
                all_coarse,
                writer, 
                epoch,
                viz_dir,
                num_samples=config.get('logging', {}).get('viz_num_samples', 4)
            )
    
    return avg_loss, mean_pearson


def create_validation_visualizations(viz_data, all_preds, all_targets, all_coarse, writer, epoch, viz_dir, num_samples=4):
    """Create comprehensive validation visualizations and save as PNG files"""
    
    # Limit number of samples to visualize
    num_viz = min(num_samples, len(viz_data['pred']))
    
    for idx in range(num_viz):
        pred = viz_data['pred'][idx]
        target = viz_data['target'][idx]
        coarse = viz_data['coarse'][idx] if viz_data['coarse'] is not None else None
        
        # 1. Voxel-wise comparison (first 500 voxels)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Plot full comparison
        axes[0].plot(target[:500], label='Target', alpha=0.7, linewidth=1.5)
        axes[0].plot(pred[:500], label='Prediction', alpha=0.7, linewidth=1.5)
        if coarse is not None:
            axes[0].plot(coarse[:500], label='Coarse', alpha=0.5, linewidth=1, linestyle='--')
        axes[0].set_xlabel('Voxel Index')
        axes[0].set_ylabel('fMRI Value')
        axes[0].set_title(f'Voxel-wise Comparison (First 500) - Sample {idx+1}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot error
        error = pred[:500] - target[:500]
        axes[1].plot(error, color='red', alpha=0.6, linewidth=1)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1].set_xlabel('Voxel Index')
        axes[1].set_ylabel('Prediction Error')
        axes[1].set_title('Prediction Error')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Save PNG
        fig.savefig(viz_dir / f'voxel_comparison_sample_{idx+1}.png', dpi=150, bbox_inches='tight')
        # Log to TensorBoard
        writer.add_figure(f'Val/Voxel_Comparison_Sample_{idx+1}', fig, epoch)
        plt.close(fig)
        
        # 2. Scatter plot (Predicted vs Target)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter for refined prediction
        axes[0].scatter(target, pred, alpha=0.3, s=1)
        axes[0].plot([target.min(), target.max()], [target.min(), target.max()], 
                     'r--', linewidth=2, label='Perfect Prediction')
        corr, _ = pearsonr(target, pred)
        axes[0].set_xlabel('Target fMRI')
        axes[0].set_ylabel('Predicted fMRI')
        axes[0].set_title(f'Prediction vs Target (Pearson: {corr:.4f}) - Sample {idx+1}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Scatter for coarse prediction
        if coarse is not None:
            axes[1].scatter(target, coarse, alpha=0.3, s=1, color='orange')
            axes[1].plot([target.min(), target.max()], [target.min(), target.max()], 
                         'r--', linewidth=2, label='Perfect Prediction')
            corr_coarse, _ = pearsonr(target, coarse)
            axes[1].set_xlabel('Target fMRI')
            axes[1].set_ylabel('Coarse Predicted fMRI')
            axes[1].set_title(f'Coarse Prediction vs Target (Pearson: {corr_coarse:.4f})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Save PNG
        fig.savefig(viz_dir / f'scatter_plot_sample_{idx+1}.png', dpi=150, bbox_inches='tight')
        # Log to TensorBoard
        writer.add_figure(f'Val/Scatter_Plot_Sample_{idx+1}', fig, epoch)
        plt.close(fig)
        
        # 3. Distribution comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].hist(target, bins=50, alpha=0.6, label='Target', color='blue')
        axes[0].hist(pred, bins=50, alpha=0.6, label='Prediction', color='orange')
        axes[0].set_xlabel('fMRI Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Value Distribution - Sample {idx+1}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Error distribution
        error = pred - target
        axes[1].hist(error, bins=50, alpha=0.7, color='red')
        axes[1].axvline(x=0, color='black', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Prediction Error')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Error Distribution (Mean: {error.mean():.4f}, Std: {error.std():.4f})')
        axes[1].grid(True, alpha=0.3)
        
        # Statistics comparison
        stats_labels = ['Mean', 'Std', 'Min', 'Max']
        target_stats = [target.mean(), target.std(), target.min(), target.max()]
        pred_stats = [pred.mean(), pred.std(), pred.min(), pred.max()]
        
        x = np.arange(len(stats_labels))
        width = 0.35
        axes[2].bar(x - width/2, target_stats, width, label='Target', alpha=0.7)
        axes[2].bar(x + width/2, pred_stats, width, label='Prediction', alpha=0.7)
        axes[2].set_ylabel('Value')
        axes[2].set_title('Statistics Comparison')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(stats_labels)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        # Save PNG
        fig.savefig(viz_dir / f'distribution_sample_{idx+1}.png', dpi=150, bbox_inches='tight')
        # Log to TensorBoard
        writer.add_figure(f'Val/Distribution_Sample_{idx+1}', fig, epoch)
        plt.close(fig)
    
    # 4. Overall correlation distribution across all validation samples
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    correlations = []
    for i in range(len(all_preds)):
        corr, _ = pearsonr(all_targets[i], all_preds[i])
        correlations.append(corr)
    
    axes[0].hist(correlations, bins=30, alpha=0.7, color='green')
    axes[0].axvline(x=np.mean(correlations), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(correlations):.4f}')
    axes[0].set_xlabel('Pearson Correlation')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Correlation Distribution (All {len(all_preds)} Samples)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MSE distribution
    mse_values = []
    for i in range(len(all_preds)):
        mse = np.mean((all_preds[i] - all_targets[i]) ** 2)
        mse_values.append(mse)
    
    axes[1].hist(mse_values, bins=30, alpha=0.7, color='purple')
    axes[1].axvline(x=np.mean(mse_values), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(mse_values):.4f}')
    axes[1].set_xlabel('MSE')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'MSE Distribution (All {len(all_preds)} Samples)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Save PNG
    fig.savefig(viz_dir / 'overall_metrics.png', dpi=150, bbox_inches='tight')
    # Log to TensorBoard
    writer.add_figure('Val/Overall_Metrics', fig, epoch)
    plt.close(fig)
    
    print(f"  ✓ Saved {num_viz * 3 + 1} visualization plots to {viz_dir}")


def save_checkpoint(state, path: Path):
    torch.save(state, str(path))


def main(args):
    config = load_config(args.config)
    set_seed(config['experiment'].get('seed', 42))

    save_dir = Path(config['experiment']['save_dir']) / config['experiment']['name'] / config['experiment']['version']
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mapping = load_mapping(config['data']['mapping_path'])

    train_loader, val_loader = create_dataloaders(
        datalist_path=config['data']['datalist_path'],
        fmri_path=config['data']['fmri_path'],
        train_embeddings_path=config['data']['train_embeddings_path'],
        test_embeddings_path=config['data']['test_embeddings_path'],
        subjects=config['data'].get('subjects', [1]),
        batch_size=config['data'].get('batch_size', 16),
        average_trials_train=config['data'].get('average_trials_train', False),
        average_trials_val=config['data'].get('average_trials_val', True),
        augment_noise_train=config['data'].get('augment_noise_train', False),
        noise_std=config['data'].get('noise_std', 0.1),
        parcel_labels_path=config['data'].get('parcel_labels_path'),
        apply_zscore_train=config['data'].get('apply_zscore_train', False),
        apply_zscore_val=config['data'].get('apply_zscore_val', False)
    )

    parcel_mapper = None
    if config['data'].get('use_parcellation', False) and config['data'].get('parcel_labels_path'):
        parcel_mapper = ParcelMapper.from_files(config['data']['parcel_labels_path'])

    mean_fmri_1d = compute_mean_fmri_1d(train_loader, device)

    model = DualStreamCNN(
        vis_dim=config['model'].get('vis_dim', 768),
        compact_shape=tuple(config['model'].get('compact_shape', mapping['compact_shape'])),
        n_voxels=int(config['model'].get('n_voxels', mapping['n_voxels'])),
        base_channels_3d=config['model'].get('base_channels_3d', 64),
        base_channels_1d=config['model'].get('base_channels_1d', 128),
        depths=config['model'].get('depths', [2,2,4,2]),
        drop_path_rate=config['model'].get('drop_path_rate', 0.1),
        fusion_type=config['model'].get('fusion_type', 'attention'),
        fusion_dim=int(config['model'].get('fusion_dim', 128)),
    ).to(device)

    # set mean baseline
    model.mean_fmri_1d.copy_(mean_fmri_1d)

    model.mean_fmri_1d.copy_(mean_fmri_1d)

    # Use PeakFocusedLoss as requested
    alpha = float(config['model']['loss'].get('alpha', 10.0))
    tau = float(config['model']['loss'].get('tau', 0.5))
    criterion = PeakFocusedLoss(alpha=alpha, tau=tau)

    # Coerce numeric hyperparameters to proper types (robust to YAML strings)
    lr = float(config['training'].get('lr', 1e-4))
    weight_decay = float(config['training'].get('weight_decay', 0.01))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    t_max = int(config['training'].get('max_epochs', 100))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    use_amp = config['training'].get('precision', '32') == '16-mixed'
    scaler = GradScaler('cuda') if use_amp else None
    writer = SummaryWriter(save_dir / 'logs')

    best_val = float('inf')
    max_epochs = t_max
    grad_clip_val = float(config['training'].get('gradient_clip_val', 1.0))

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            mapping, grad_clip_val, use_amp, writer, epoch
        )

        val_loss, val_pearson = validate(model, val_loader, criterion, device, mapping, writer, epoch, config)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Pearson: {val_pearson:.4f}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss': val_loss
            }, checkpoint_dir / 'best_dual_stream.pt')

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_dual_stream.yaml')
    args = parser.parse_args()
    main(args)

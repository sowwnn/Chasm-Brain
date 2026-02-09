
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
import sys
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.hierarchical_v2 import HierarchicalARM_V2
from model.contrastive_loss import InfoNCELoss
from model.loss import PeakFocusedLoss
from data.dataset import create_dataloaders
import matplotlib.pyplot as plt

def compute_pearson_batch(pred, target):
    if torch.isnan(pred).any() or torch.isnan(target).any():
        return 0.0
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    
    corrs = []
    for i in range(p.shape[0]):
        ps, ts = p[i], t[i]
        # Skip samples with no variance to avoid division by zero/NaN
        if ps.std() > 1e-9 and ts.std() > 1e-9:
            try:
                r = np.corrcoef(ps, ts)[0, 1]
                if not np.isnan(r):
                    corrs.append(r)
            except:
                continue
    return np.mean(corrs) if corrs else 0.0

def train_stage_1(model, dataloader, optimizer, scaler, device, config, epoch):
    model.train()
    total_loss, total_mse, total_cont, total_pearson = 0, 0, 0, 0
    # Stage 1: Peak Focused Loss (helps with standard deviation/amplitude)
    peak_criterion = PeakFocusedLoss(
        alpha=config['loss'].get('peak_alpha', 3.0),
        tau=config['loss'].get('peak_tau', 0.4),
        pearson_weight=config['loss'].get('pearson_weight', 0.5),
        std_weight=config['loss'].get('std_weight', 0.5)
    ).to(device)
    
    # InfoNCE with higher temperature for stability (lower = harder negatives = higher gradients)
    cont_criterion = InfoNCELoss(temperature=0.15).to(device)  # Increased from 0.07
    cont_weight = config['loss'].get('contrastive_weight', 0.1)
    
    # Pre-calculate global mean if not provided (assume 0 for normalized data or use batch mean)
    # Here we'll use batch-wise target mean as baseline for peak detection
    
    pbar = tqdm(dataloader, desc=f"Stage 1 Train {epoch}")
    for batch in pbar:
        vis_feat = batch['embedding'].to(device)
        gt_roi_means = batch['roi_means'].to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=True):
            # Forward
            pred_roi_means, img_proj, brain_proj = model(vis_feat, stage=1, gt_roi_means=gt_roi_means)
            
            # Baseline mean for peak focused loss (mean across dimensions for each sample)
            # Or use global mean across batch
            batch_mean = gt_roi_means.mean(dim=0, keepdim=True) 

            # Peak Focused Loss includes MSE + Pearson + STD matching
            loss_peak = peak_criterion(pred_roi_means, gt_roi_means, batch_mean)
            loss_cont = cont_criterion(img_proj, brain_proj)
            
            loss = loss_peak + cont_weight * loss_cont

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced from 1.0
        scaler.step(optimizer)
        scaler.update()

        p = compute_pearson_batch(pred_roi_means, gt_roi_means)
        
        total_loss += loss.item()
        total_mse += loss_peak.item() if hasattr(loss_peak, 'item') else loss_peak
        total_cont += loss_cont.item()
        total_pearson += p
        
        pbar.set_postfix({
            'L_Tot': f"{loss.item():.4f}",
            'L_Peak': f"{loss_peak.item():.4f}",
            'L_Cont': f"{loss_cont.item():.4f}",
            'P': f"{p:.4f}"
        })

    n = len(dataloader)
    return total_loss/n, total_mse/n, total_cont/n, total_pearson/n

def train_stage_2(model, dataloader, optimizer, scaler, device, config, epoch):
    """
    Train V2 Stage 2: Transformer Residual Learning
    """
    model.train()
    model.stage1.eval()
    for param in model.stage1.parameters():
        param.requires_grad = False

    total_loss, total_mse, total_std, total_pearson = 0, 0, 0, 0
    mse_criterion = nn.MSELoss()
    pearson_weight = config['loss'].get('pearson_weight', 0.5)
    std_weight = config['loss'].get('std_weight', 0.2)

    pbar = tqdm(dataloader, desc=f"Stage 2 Train {epoch}")
    for batch in pbar:
        vis_feat = batch['embedding'].to(device)
        gt_voxels = batch['fmri'].to(device)
        gt_roi_means = batch['roi_means'].to(device)

        # Stage 2 requires ROI means as guidance
        # During Stage 2 training, we use sampled/predicted ROI means from Stage 1
        with torch.no_grad():
            guidance, _, _ = model.stage1(vis_feat)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=True):
            # Forward Stage 2
            _, pred_voxels = model(vis_feat, stage=2, gt_roi_means=guidance)

            # MSE Loss
            loss_mse = mse_criterion(pred_voxels, gt_voxels)

            # Pearson Loss
            pred_centered = pred_voxels - pred_voxels.mean(dim=1, keepdim=True)
            target_centered = gt_voxels - gt_voxels.mean(dim=1, keepdim=True)
            pred_std = pred_centered.std(dim=1, keepdim=True) + 1e-8
            target_std = target_centered.std(dim=1, keepdim=True) + 1e-8
            
            pearson_corr = (pred_centered * target_centered).mean(dim=1) / (pred_std.squeeze() * target_std.squeeze())
            loss_pearson = (1 - pearson_corr).mean()

            # Std Loss: Ép biên độ (vốn là nguyên nhân gây MSE cao)
            loss_std = F.mse_loss(pred_std, target_std)

            # Combined Loss
            loss = loss_mse + pearson_weight * loss_pearson + std_weight * loss_std

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        p = compute_pearson_batch(pred_voxels, gt_voxels)
        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_std += loss_std.item()
        total_pearson += p

        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'MSE': f"{loss_mse.item():.4f}",
            'Std': f"{loss_std.item():.4f}",
            'P': f"{p:.4f}"
        })

    n = len(dataloader)
    return total_loss/n, total_mse/n, total_std/n, total_pearson/n

@torch.no_grad()
def validate(model, dataloader, device, stage):
    model.eval()
    total_pearson = 0
    total_mse = 0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        vis_feat = batch['embedding'].to(device)

        if stage == 1:
            target = batch['roi_means'].to(device)
            pred, _, _ = model(vis_feat, stage=1)
        else:
            target = batch['fmri'].to(device)
            # Full pass: S1 -> S2 Transformer
            _, pred = model(vis_feat, stage=2)

        # Compute MSE
        mse = F.mse_loss(pred, target)
        total_mse += mse.item()

        total_pearson += compute_pearson_batch(pred, target)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    avg_mse = total_mse / len(dataloader)
    avg_pearson = total_pearson / len(dataloader)
    return avg_pearson, avg_mse, all_preds, all_targets


def visualize_validation(all_preds, all_targets, epoch, stage, save_dir, writer=None):
    """
    Create visualizations for validation results.

    Args:
        all_preds: [N, D] predictions
        all_targets: [N, D] targets
        epoch: current epoch
        stage: 1 or 2
        save_dir: directory to save images
        writer: tensorboard writer (optional)
    """
    # Compute per-sample correlations
    sample_corrs = []
    for i in range(len(all_preds)):
        p_val, t_val = all_preds[i], all_targets[i]
        if np.isfinite(p_val).all() and np.isfinite(t_val).all():
            if p_val.std() > 1e-6 and t_val.std() > 1e-6:
                try:
                    r = np.corrcoef(p_val, t_val)[0, 1]
                    if not np.isnan(r):
                        sample_corrs.append(r)
                except:
                    continue

    sample_corrs = np.array(sample_corrs)
    if len(sample_corrs) == 0:
        print("Warning: No valid samples for visualization (std too low or NaNs)")
        return
        
    mean_corr = sample_corrs.mean()
    median_corr = np.median(sample_corrs)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Pearson distribution histogram
    axes[0, 0].hist(sample_corrs, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(mean_corr, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_corr:.3f}')
    axes[0, 0].axvline(median_corr, color='g', linestyle='--', linewidth=2, label=f'Median: {median_corr:.3f}')
    axes[0, 0].set_xlabel('Pearson Correlation', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title(f'Stage {stage} Epoch {epoch} - Pearson Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)

    # 2. Scatter plot: Pred vs Target (random 3 samples)
    sample_indices = np.random.choice(len(all_preds), min(3, len(all_preds)), replace=False)
    pred_samples = all_preds[sample_indices].flatten()
    target_samples = all_targets[sample_indices].flatten()

    axes[0, 1].scatter(target_samples, pred_samples, alpha=0.4, s=2, c='steelblue')
    axes[0, 1].plot([target_samples.min(), target_samples.max()],
                     [target_samples.min(), target_samples.max()],
                     'r--', linewidth=2, label='Perfect prediction')
    axes[0, 1].set_xlabel('Ground Truth', fontsize=11)
    axes[0, 1].set_ylabel('Prediction', fontsize=11)
    axes[0, 1].set_title(f'Pred vs GT (3 random samples)', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)

    # 3 & 4. Example predictions (2 samples: best and worst)
    best_idx = sample_corrs.argmax()
    worst_idx = sample_corrs.argmin()

    for idx, (sample_idx, title_prefix) in enumerate([(best_idx, 'Best'), (worst_idx, 'Worst')]):
        ax = axes[1, idx]
        pred_sample = all_preds[sample_idx]
        target_sample = all_targets[sample_idx]

        # Subsample for clarity if too many points
        step = max(1, len(pred_sample) // 500)
        x = np.arange(len(pred_sample))

        ax.plot(x[::step], target_sample[::step], 'b-', alpha=0.6, label='Ground Truth', linewidth=1.5)
        ax.plot(x[::step], pred_sample[::step], 'r-', alpha=0.6, label='Prediction', linewidth=1.5)

        corr = sample_corrs[sample_idx]
        ax.set_title(f'{title_prefix} Sample - Pearson: {corr:.3f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Voxel/ROI Index', fontsize=11)
        ax.set_ylabel('Activation', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save figure
    viz_path = save_dir / f'val_stage{stage}_epoch{epoch:03d}.png'
    plt.savefig(viz_path, dpi=100, bbox_inches='tight')

    # Log to tensorboard if available
    if writer is not None:
        writer.add_figure(f'Validation/Stage{stage}', fig, epoch)
        writer.add_histogram(f'Validation/Stage{stage}/Pearson_Distribution', sample_corrs, epoch)

    plt.close(fig)

    return mean_corr, median_corr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_hierarchical_v2.yaml')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=None,
                        help='Train specific stage only: 1=Stage1 only, 2=Stage2 only, None=both stages')
    parser.add_argument('--load_stage1', type=str, default=None,
                        help='Path to Stage 1 checkpoint (required when --stage 2)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to full model checkpoint to resume training (Stage 1 + Stage 2)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'visualizations').mkdir(exist_ok=True)  # Create viz directory
    writer = SummaryWriter(log_dir=str(save_dir / 'logs'))

    print(f"Training Hierarchical V2 on device: {device}")
    
    # Load Data
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
        noise_std=config['data'].get('noise_std', 0.1),
        sub_roi_cluster_path=config['data']['sub_roi_cluster_path'],
        roi_top_k_percent=config['data'].get('roi_top_k_percent', None)
    )

    # Initialize Model
    model = HierarchicalARM_V2(
        visual_dim=config['model']['visual_dim'],
        num_clusters=config['model']['num_clusters'],
        fmri_dim=config['model']['fmri_dim'],
        hidden_dim_1=config['model']['hidden_dim_1'],
        embed_dim_2=config['model']['embed_dim_2'],
        depth_2=config['model']['depth_2'],
        num_heads_2=config['model']['num_heads_2'],
        num_global_queries=config['model'].get('num_global_queries', 128),
        contrastive_dim=config['model']['contrastive_dim'],
        use_mamba=config['model']['use_mamba']
    ).to(device)

    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['precision'] == 'fp16')

    stage1_epochs = config['training'].get('stage_1_epochs', 50)
    total_epochs = config['training']['max_epochs']

    # Handle stage-specific training
    train_stage = args.stage  # None, 1, or 2

    # Validate arguments
    if train_stage == 2 and args.load_stage1 is None and args.resume is None:
        print("ERROR: --load_stage1 or --resume is required when training Stage 2 only")
        print("Example: python scripts/train_hierarchical_v2.py --stage 2 --load_stage1 checkpoints/hierarchical_v2_subj01/best_stage1.pth")
        print("Or: python scripts/train_hierarchical_v2.py --stage 2 --resume checkpoints/hierarchical_v21_subj01/best_model.pth")
        return

    # Load full checkpoint if resuming
    if args.resume is not None:
        print(f"Resuming training from full checkpoint: {args.resume}")
        if not Path(args.resume).exists():
            print(f"ERROR: Checkpoint not found at {args.resume}")
            return
        state_dict = torch.load(args.resume, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print("Full model (Stage 1 + Stage 2) weights loaded successfully.")
    # Load Stage 1 checkpoint if training Stage 2 only (and not resuming)
    elif train_stage == 2 and args.load_stage1 is not None:
        print(f"Loading Stage 1 checkpoint from: {args.load_stage1}")
        if not Path(args.load_stage1).exists():
            print(f"ERROR: Checkpoint not found at {args.load_stage1}")
            return
        state_dict = torch.load(args.load_stage1, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("Stage 1 weights loaded successfully.")

    # Optimizers
    optimizer_s1 = optim.AdamW(model.stage1.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    optimizer_s2 = optim.AdamW(model.stage2.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])

    best_p_s1, best_p_s2 = -1.0, -1.0

    print("=== Starting Training V2 ===")
    print(f"Structure: Contrastive+MLP (Stage 1) -> Transformer/Mamba (Stage 2)")
    if train_stage == 1:
        print("MODE: Training Stage 1 ONLY")
    elif train_stage == 2:
        print("MODE: Training Stage 2 ONLY (Stage 1 frozen)")
    else:
        print("MODE: Training BOTH stages sequentially")

    for epoch in range(1, total_epochs + 1):
        # Determine which stage to train based on --stage argument or epoch
        if train_stage == 1:
            # Train Stage 1 only
            should_train_s1 = True
            should_train_s2 = False
        elif train_stage == 2:
            # Train Stage 2 only
            should_train_s1 = False
            should_train_s2 = True
        else:
            # Auto mode: switch based on epoch
            should_train_s1 = epoch <= stage1_epochs
            should_train_s2 = epoch > stage1_epochs

        if should_train_s1:
            # S1: Mamba Selection Stage
            l_tot, l_peak, l_cont, p = train_stage_1(model, train_loader, optimizer_s1, scaler, device, config, epoch)
            val_p, val_mse, val_preds, val_targets = validate(model, val_loader, device, stage=1)

            writer.add_scalar('Stage1/LossTotal', l_tot, epoch)
            writer.add_scalar('Stage1/LossPeak', l_peak, epoch)
            writer.add_scalar('Stage1/LossCont', l_cont, epoch)
            writer.add_scalar('Stage1/TrainPearson', p, epoch)
            writer.add_scalar('Stage1/ValPearson', val_p, epoch)
            writer.add_scalar('Stage1/ValMSE', val_mse, epoch)

            print(f"S1 Ep {epoch} | L_Tot:{l_tot:.3f} L_Peak:{l_peak:.3f} L_Cont:{l_cont:.3f} | Tr P:{p:.3f} Val P:{val_p:.3f} MSE:{val_mse:.3f}")

            # Visualize every N epochs (configurable)
            viz_interval = config['training'].get('viz_interval', 5)
            if epoch % viz_interval == 0 or val_p > best_p_s1:
                visualize_validation(val_preds, val_targets, epoch, stage=1,
                                   save_dir=save_dir / 'visualizations', writer=writer)

            if val_p > best_p_s1:
                best_p_s1 = val_p
                torch.save(model.state_dict(), save_dir / 'best_stage1.pth')
                print(f"✓ Saved best Stage 1: {val_p:.4f}")

        if should_train_s2:
            # S2
            # Auto mode: reload best S1 when transitioning
            if train_stage is None and epoch == stage1_epochs + 1:
                print(">>> Transitioning to Stage 2 (Transformer/Mamba)...")
                # Reload best S1
                s1_path = save_dir / 'best_stage1.pth'
                if s1_path.exists():
                    model.load_state_dict(torch.load(s1_path), strict=False)
                    print("Loaded best Stage 1 weights.")

            loss, loss_mse, loss_std, p = train_stage_2(model, train_loader, optimizer_s2, scaler, device, config, epoch)
            val_p, val_mse, val_preds, val_targets = validate(model, val_loader, device, stage=2)

            writer.add_scalar('Stage2/Loss', loss, epoch)
            writer.add_scalar('Stage2/TrainMSE', loss_mse, epoch)
            writer.add_scalar('Stage2/LossStd', loss_std, epoch)
            writer.add_scalar('Stage2/TrainPearson', p, epoch)
            writer.add_scalar('Stage2/ValPearson', val_p, epoch)
            writer.add_scalar('Stage2/ValMSE', val_mse, epoch)

            print(f"S2 Ep {epoch} | Loss:{loss:.3f} Tr MSE:{loss_mse:.3f} Std:{loss_std:.3f} | Tr P:{p:.3f} | Val P:{val_p:.3f} MSE:{val_mse:.3f}")

            # Visualize every N epochs (configurable)
            viz_interval = config['training'].get('viz_interval', 5)
            if epoch % viz_interval == 0 or val_p > best_p_s2:
                visualize_validation(val_preds, val_targets, epoch, stage=2,
                                   save_dir=save_dir / 'visualizations', writer=writer)

            if val_p > best_p_s2:
                best_p_s2 = val_p
                torch.save(model.state_dict(), save_dir / 'best_model.pth')
                print(f"✓ Saved best Stage 2: {val_p:.4f}")

    print("Training Complete.")

    # ===== Final Evaluation =====
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    # Load best model
    best_model_path = save_dir / 'best_model.pth'
    best_s1_path = save_dir / 'best_stage1.pth'

    if best_model_path.exists():
        print("Loading best full model...")
        model.load_state_dict(torch.load(best_model_path))
        stage_to_eval = 2
    elif best_s1_path.exists():
        print("Loading best Stage 1 model...")
        model.load_state_dict(torch.load(best_s1_path))
        stage_to_eval = 1
    else:
        print("No checkpoint found, evaluating current model...")
        stage_to_eval = 2 if epoch > stage1_epochs else 1

    # Evaluate on validation set
    model.eval()
    all_preds = []
    all_targets = []

    print(f"\nEvaluating Stage {stage_to_eval} on validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            vis_feat = batch['embedding'].to(device)

            if stage_to_eval == 1:
                target = batch['roi_means'].to(device)
                pred, _, _ = model(vis_feat, stage=1)
            else:
                target = batch['fmri'].to(device)
                _, pred = model(vis_feat, stage=2)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    print("\nComputing metrics...")

    # Per-sample Pearson correlation
    sample_corrs = []
    for i in range(len(all_preds)):
        if all_preds[i].std() > 1e-6 and all_targets[i].std() > 1e-6:
            r = np.corrcoef(all_preds[i], all_targets[i])[0, 1]
            sample_corrs.append(r)

    mean_corr = np.mean(sample_corrs)
    median_corr = np.median(sample_corrs)
    std_corr = np.std(sample_corrs)

    # MSE metric
    overall_mse = np.mean((all_preds - all_targets) ** 2)

    print(f"\nFinal Validation Results (Stage {stage_to_eval}):")
    print(f"  Mean Pearson: {mean_corr:.4f} ± {std_corr:.4f}")
    print(f"  Median Pearson: {median_corr:.4f}")
    print(f"  Min: {np.min(sample_corrs):.4f}, Max: {np.max(sample_corrs):.4f}")
    print(f"  MSE: {overall_mse:.4f}")

    # ===== Visualization =====
    print("\nGenerating visualizations...")

    # 1. Pearson distribution histogram
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(sample_corrs, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(mean_corr, color='r', linestyle='--', label=f'Mean: {mean_corr:.3f}')
    axes[0, 0].axvline(median_corr, color='g', linestyle='--', label=f'Median: {median_corr:.3f}')
    axes[0, 0].set_xlabel('Pearson Correlation')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Stage {stage_to_eval} - Pearson Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Scatter plot: Pred vs Target (first 5 samples averaged)
    sample_indices = np.random.choice(len(all_preds), min(5, len(all_preds)), replace=False)
    pred_avg = all_preds[sample_indices].flatten()
    target_avg = all_targets[sample_indices].flatten()

    axes[0, 1].scatter(target_avg, pred_avg, alpha=0.3, s=1)
    axes[0, 1].plot([target_avg.min(), target_avg.max()],
                     [target_avg.min(), target_avg.max()], 'r--', label='Perfect prediction')
    axes[0, 1].set_xlabel('Ground Truth')
    axes[0, 1].set_ylabel('Prediction')
    axes[0, 1].set_title(f'Stage {stage_to_eval} - Pred vs GT (5 samples)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Example predictions (first 3 samples)
    for i in range(min(3, len(all_preds))):
        ax = axes[1, 0] if i == 0 else (axes[1, 1] if i == 1 else None)
        if ax is None:
            continue

        pred_sample = all_preds[i]
        target_sample = all_targets[i]

        x = np.arange(len(pred_sample))
        ax.plot(x[::10], target_sample[::10], 'b-', alpha=0.5, label='Ground Truth', linewidth=1)
        ax.plot(x[::10], pred_sample[::10], 'r-', alpha=0.5, label='Prediction', linewidth=1)

        corr = np.corrcoef(pred_sample, target_sample)[0, 1]
        ax.set_title(f'Sample {i+1} - Pearson: {corr:.3f}')
        ax.set_xlabel('Voxel/ROI Index (subsampled)')
        ax.set_ylabel('Activation')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save figure
    viz_path = save_dir / f'final_evaluation_stage{stage_to_eval}.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {viz_path}")

    plt.close()

    # Save metrics to file
    metrics_path = save_dir / f'final_metrics_stage{stage_to_eval}.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Stage {stage_to_eval} - Final Validation Results\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Mean Pearson: {mean_corr:.6f}\n")
        f.write(f"Median Pearson: {median_corr:.6f}\n")
        f.write(f"Std Pearson: {std_corr:.6f}\n")
        f.write(f"Min Pearson: {np.min(sample_corrs):.6f}\n")
        f.write(f"Max Pearson: {np.max(sample_corrs):.6f}\n")
        f.write(f"MSE: {overall_mse:.6f}\n")
        f.write(f"\nTotal validation samples: {len(all_preds)}\n")

    print(f"✓ Metrics saved to {metrics_path}")
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()

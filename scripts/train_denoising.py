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

from ARMNet import DenoisingARM, PeakFocusedLoss, MaskedPeakFocusedLoss
from data.neuroflux_dataset import create_dataloaders


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

def plot_results(pred, target, mean_fmri, epoch, save_dir):
    """Plot validation results for 15k voxels."""
    num_samples = min(3, pred.shape[0])
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4 * num_samples))
    if num_samples == 1: axes = [axes]

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    mean_np = mean_fmri.detach().cpu().numpy()

    for i in range(num_samples):
        axes[i].plot(target_np[i], label='Target (GT)', alpha=0.5, color='blue')
        axes[i].plot(mean_np[i], label='Mean (Input)', alpha=0.5, color='gray', linestyle='--')
        axes[i].plot(pred_np[i], label='Pred', alpha=0.8, color='red')
        axes[i].set_title(f'Sample {i+1} - 15,724 Voxels')
        axes[i].legend()
        axes[i].set_xlabel('Voxel Index')
        axes[i].set_ylabel('Activation')

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
    total_loss, total_pearson, total_mse = 0, 0, 0

    # Expand global mean to batch size
    batch_mean_base = mean_fmri.unsqueeze(0)

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        target = batch['fmri'].to(device)  # [B, 15724]
        vis_feat = batch['embedding'].to(device)  # [B, visual_dim]
        batch_mean = batch_mean_base.expand(target.size(0), -1)

        # Pure Noise Strategy (What gave 0.2 Pearson previously)
        x_input = torch.randn_like(target)

        # Additional Visual Regularization
        vis_noise = torch.randn_like(vis_feat) * config['training'].get('vis_noise_std', 0.2)
        vis_feat_noised = vis_feat + vis_noise

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=config['training']['precision'] == 'fp16'):
            pred = model(x_input, vis_feat_noised, batch_mean)
            loss = criterion(pred, target, batch_mean)

        scaler.scale(loss).backward()
        
        # Stability: Grad Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pearson = compute_pearson_batch(pred, target)
        total_mse += compute_mse_batch(pred, target)
        total_pearson += pearson

        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'pearson': f"{pearson:.4f}"})

    return total_loss / len(dataloader), total_pearson / len(dataloader), total_mse / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, device, config, mean_fmri):
    model.eval()
    total_pearson, total_mse = 0, 0
    batch_mean_base = mean_fmri.unsqueeze(0)

    pbar = tqdm(dataloader, desc="Validation")
    for batch in pbar:
        target = batch['fmri'].to(device)
        vis_feat = batch['embedding'].to(device)
        batch_mean = batch_mean_base.expand(target.size(0), -1)

        # Inference: Reconstruct from pure noise
        x_input = torch.randn_like(target)
        pred = model(x_input, vis_feat, batch_mean)

        pearson = compute_pearson_batch(pred, target)
        total_pearson += pearson
        total_mse += compute_mse_batch(pred, target)

    n = len(dataloader)
    return {'pearson': total_pearson / n, 'mse': total_mse / n}

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / 'logs'))

    # Loaders - Full 15k voxels (no parcellation)
    train_loader, val_loader = create_dataloaders(
        datalist_path=config['data']['datalist_path'],
        fmri_path=config['data']['fmri_path'],
        train_embeddings_path=config['data']['train_embeddings_path'],
        test_embeddings_path=config['data']['test_embeddings_path'],
        subjects=[config['data']['subject']],
        batch_size=config['training']['batch_size'],
        parcel_labels_path=None,  # No parcellation
        average_trials_train=config['data']['average_trials_train'],
        average_trials_val=config['data']['average_trials_val'],
        augment_noise_train=config['data']['augment_noise_train'],
        noise_std=config['data'].get('noise_std', 0.15)
    )

    print(f"Using full voxels: {config['model']['fmri_dim']} voxels")

    # Compute global mean fMRI
    mean_fmri = compute_mean_fmri(train_loader, device)

    # Parse output_clamp from config
    output_clamp = config['model'].get('output_clamp')
    if output_clamp is not None:
        output_clamp = tuple(output_clamp)
        print(f"Output clamping enabled: [{output_clamp[0]}, {output_clamp[1]}]")

    model = DenoisingARM(
        visual_dim=config['model']['visual_dim'],
        fmri_dim=config['model']['fmri_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        output_clamp=output_clamp
    ).to(device)

    # Setup Loss Function
    use_mask = config['loss'].get('use_mask', False)
    if use_mask and config['data'].get('roi_mapping_path'):
        criterion = MaskedPeakFocusedLoss.from_roi_mapping(
            roi_mapping_path=config['data']['roi_mapping_path'],
            mask_field=config['loss'].get('mask_field', 'streams'),
            alpha=config['loss']['alpha'],
            tau=config['loss']['tau'],
            pearson_weight=config['loss']['pearson_weight'],
            std_weight=config['loss'].get('std_weight', 1.0)
        ).to(device)
        print(f"Using MaskedPeakFocusedLoss with {criterion.n_masked}/{criterion.n_total} voxels")
    else:
        criterion = PeakFocusedLoss(
            alpha=config['loss']['alpha'],
            tau=config['loss']['tau'],
            pearson_weight=config['loss']['pearson_weight'],
            std_weight=config['loss'].get('std_weight', 1.0)
        )
        print("Using PeakFocusedLoss (no mask)")

    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['precision'] == 'fp16')

    # Cosine Annealing LR Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['max_epochs'],
        eta_min=config['training'].get('lr_min', 1e-6)
    )
    print(f"Using CosineAnnealingLR: lr={config['training']['lr']} -> {config['training'].get('lr_min', 1e-6)}")

    best_pearson = -1.0
    for epoch in range(1, config['training']['max_epochs'] + 1):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch}/{config['training']['max_epochs']} | LR: {current_lr:.2e}")
        t_loss, t_pearson, t_mse = train_epoch(model, train_loader, criterion, optimizer, scaler, device, config, mean_fmri)
        v_metrics = validate(model, val_loader, device, config, mean_fmri)

        print(f"Train Pearson: {t_pearson:.4f} | Val Pearson: {v_metrics['pearson']:.4f} | Val MSE: {v_metrics['mse']:.4f}")

        # Plotting
        with torch.no_grad():
            batch = next(iter(val_loader))
            fmri = batch['fmri'][:3].to(device)
            vis = batch['embedding'][:3].to(device)
            batch_mean = mean_fmri.unsqueeze(0).expand(fmri.size(0), -1)
            x_input = torch.randn_like(fmri)
            pred = model(x_input, vis, batch_mean)
            plot_results(pred, fmri, batch_mean, epoch, save_dir / 'plots')

        writer.add_scalar('Pearson/Train', t_pearson, epoch)
        writer.add_scalar('Pearson/Val', v_metrics['pearson'], epoch)
        writer.add_scalar('MSE/Val', v_metrics['mse'], epoch)
        writer.add_scalar('Loss/Train', t_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        # Step scheduler
        scheduler.step()

        if v_metrics['pearson'] > best_pearson:
            best_pearson = v_metrics['pearson']
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            print(f"New best: {best_pearson:.4f}")

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_denoising.yaml')
    args = parser.parse_args()
    main(args)


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


sys.path.append(str(Path(__file__).resolve().parent.parent))

from ARMNet import HierarchicalARM, PeakFocusedLoss
from data.neuroflux_dataset import create_dataloaders
import matplotlib.pyplot as plt

def visualize_samples(model, dataloader, device, stage, epoch, save_dir, num_samples=3):
    """
    Visualize ground truth vs prediction for a few samples.
    """
    model.eval()
    save_path = Path(save_dir) / 'visualizations' / f'epoch_{epoch:04d}'
    save_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        # Get one batch
        batch = next(iter(dataloader))
        vis_feat = batch['embedding'].to(device)
        
        if stage == 1:
            target = batch['roi_means'].to(device)
            pred, _ = model(vis_feat, stage=1)
            title_prefix = "Stage 1 (ROI Means)"
        else:
            target = batch['fmri'].to(device)
            pred_coarse, _ = model(vis_feat, stage=1)
            _, pred = model.stage2(vis_feat, pred_coarse)
            title_prefix = "Stage 2 (Voxels)"
            
        target = target.cpu().numpy()
        pred = pred.cpu().numpy()
        
        # Plot
        for i in range(min(num_samples, target.shape[0])):
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(target[i], label='Ground Truth', alpha=0.7)
            ax.plot(pred[i], label='Prediction', alpha=0.7)
            
            # Corr
            if pred[i].std() > 0 and target[i].std() > 0:
                corr = np.corrcoef(pred[i], target[i])[0, 1]
            else:
                corr = 0.0
                
            ax.set_title(f"{title_prefix} Sample {i+1} | Pearson: {corr:.4f}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.savefig(save_path / f'sample_{i}.png')
            plt.close(fig)


def compute_pearson_batch(pred, target):
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    corrs = []
    for i in range(p.shape[0]):
        # Avoid division by zero
        if p[i].std() > 1e-6 and t[i].std() > 1e-6:
            r = np.corrcoef(p[i], t[i])[0, 1]
            corrs.append(r)
    return np.mean(corrs) if corrs else 0.0

def train_stage_1(model, dataloader, optimizer, scaler, device, epoch):
    """
    Train Coarse Stage (Image -> ROI Means)
    """
    model.train()
    total_loss, total_pearson = 0, 0
    
    criterion = nn.MSELoss()
    
    pbar = tqdm(dataloader, desc=f"Stage 1 Train {epoch}")
    for batch in pbar:
        vis_feat = batch['embedding'].to(device)
        gt_roi_means = batch['roi_means'].to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            # Forward stage 1 only
            pred_roi_means, _ = model(vis_feat, stage=1)
            loss = criterion(pred_roi_means, gt_roi_means)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        p = compute_pearson_batch(pred_roi_means, gt_roi_means)
        total_loss += loss.item()
        total_pearson += p
        
        pbar.set_postfix({'Layer1_MSE': f"{loss.item():.4f}", 'P': f"{p:.4f}"})
        
    return total_loss / len(dataloader), total_pearson / len(dataloader)

def train_stage_2(model, dataloader, optimizer, scaler, device, config, epoch):
    """
    Train Fine Stage (Image + Coarse -> Voxels)
    Stage 1 is frozen (no grad).
    """
    model.train()
    # Ensure stage 1 is in eval mode or at least detached
    model.stage1.eval() 
    
    total_loss, total_pearson = 0, 0
    
    # Loss for fine tuning
    criterion = PeakFocusedLoss(
        alpha=config['loss']['alpha'], 
        tau=config['loss']['tau'],
        pearson_weight=config['loss']['pearson_weight']
    ).to(device)
    
    pbar = tqdm(dataloader, desc=f"Stage 2 Train {epoch}")
    for batch in pbar:
        vis_feat = batch['embedding'].to(device)
        gt_voxels = batch['fmri'].to(device)
        
        # We can use GT structure for "Mean" baseline if needed by loss
        # Batch Mean for anti-collapse:
        batch_mean_target = gt_voxels.mean(dim=0, keepdim=True).expand(gt_voxels.size(0), -1)

        vis_noise = torch.randn_like(vis_feat) * config['training'].get('vis_noise_std', 0.1)
        vis_feat_noised = vis_feat + vis_noise
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                # Get coarse prediction from Stage 1
                pred_coarse, _ = model(vis_feat_noised, stage=1)
            
            # Forward Stage 2 using predicted coarse maps
            # We treat Stage 1 as fixed feature extractor
            _, pred_voxels = model.stage2(vis_feat_noised, pred_coarse)
            
            loss = criterion(pred_voxels, gt_voxels, batch_mean_target)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        p = compute_pearson_batch(pred_voxels, gt_voxels)
        total_loss += loss.item()
        total_pearson += p
        
        pbar.set_postfix({'Stage2_Loss': f"{loss.item():.4f}", 'P': f"{p:.4f}"})
        
    return total_loss / len(dataloader), total_pearson / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, device, stage):
    model.eval()
    total_pearson = 0
    
    for batch in dataloader:
        vis_feat = batch['embedding'].to(device)
        
        if stage == 1:
            target = batch['roi_means'].to(device)
            pred, _ = model(vis_feat, stage=1)
        else:
            target = batch['fmri'].to(device)
            # Full pass
            pred_coarse, _ = model(vis_feat, stage=1)
            _, pred = model.stage2(vis_feat, pred_coarse)
            
        total_pearson += compute_pearson_batch(pred, target)
        
    return total_pearson / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_hierarchical.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / 'logs'))

    print(f"Training on device: {device}")
    
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
        sub_roi_cluster_path=config['data']['sub_roi_cluster_path']
    )

    # Initialize Model
    model = HierarchicalARM(
        visual_dim=config['model']['visual_dim'],
        fmri_dim=config['model']['fmri_dim'],
        num_clusters=config['model']['num_clusters'],
        hidden_dim_1=config['model']['hidden_dim_1'],
        hidden_dim_2=config['model']['hidden_dim_2'],
        layers_1=config['model']['layers_1'],
        layers_2=config['model']['layers_2'],
        dropout=config['model']['dropout']
    ).to(device)

    start_epoch = 1
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint, strict=False) 
            # Note: We don't restore optimizer state/epoch here simplicity as stage transition might be manual
            # But normally we should. Here we assume user might want to continue training or switch stages.
            # If resume file name contains 'stage1', we might infer? 
            # Let's keep it simple: Load weights, start from config epochs.
        else:
            print(f"Checkpoint not found: {args.resume}")

    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['precision'] == 'fp16')
    
    # Training Loop
    # If Stage 2 is requested, we assume we might need to train Stage 1 first
    # Strategy: Train S1 for N epochs, then S2 for remaining.
    
    stage1_epochs = config['training'].get('stage_1_epochs', 30)
    total_epochs = config['training']['max_epochs']
    
    # Stage 1 Optimizer
    optimizer_s1 = optim.AdamW(model.stage1.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scheduler_s1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=stage1_epochs)
    
    # Stage 2 Optimizer
    optimizer_s2 = optim.AdamW(model.stage2.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=total_epochs - stage1_epochs)

    print("=== Model Summary and Training Plan ===")
    print(f"Model: HierarchicalARM (Coarse-to-Fine)")
    print(f"  - Stage 1: Maps Image -> {config['model']['num_clusters']} ROI Means (Coarse Map)")
    print(f"  - Stage 2: Maps Image + Coarse Map -> {config['model']['fmri_dim']} Voxels")
    print("-" * 40)
    print(f"Training Schedule:")
    print(f"  - Stage 1: Epochs 1-{stage1_epochs} (Goal: Learn spatial activation map)")
    print(f"  - Stage 2: Epochs {stage1_epochs+1}-{total_epochs} (Goal: Refine details from coarse map)")
    print("-" * 40)
    print(f"Expected Results:")
    print(f"  - Stage 1 Pearson: Should reach high correlation (>0.6) quickly as task is simple.")
    print(f"  - Stage 2 Pearson: Should surpass direct methods (>0.3 - 0.4) by avoiding mean collapse.")
    print("==========================================")

    best_p_s1 = -1.0
    best_p_s2 = -1.0
    
    for epoch in range(1, total_epochs + 1):
        if epoch <= stage1_epochs:
            # --- STAGE 1 ---
            loss, p = train_stage_1(model, train_loader, optimizer_s1, scaler, device, epoch)
            val_p = validate(model, val_loader, device, stage=1)
            
            writer.add_scalar('Stage1/Loss', loss, epoch)
            writer.add_scalar('Stage1/TrainPearson', p, epoch)
            writer.add_scalar('Stage1/ValPearson', val_p, epoch)
            
            print(f"S1 Epoch {epoch} | Loss: {loss:.4f} | Train P: {p:.4f} | Val P: {val_p:.4f}")
            
            if val_p > best_p_s1:
                best_p_s1 = val_p
                torch.save(model.state_dict(), save_dir / 'best_stage1.pth')
            
            # Visualize every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                visualize_samples(model, val_loader, device, stage=1, epoch=epoch, save_dir=save_dir)

            scheduler_s1.step()
            
        else:
            # --- STAGE 2 ---
            # Load best stage 1 weights if transitioning
            if epoch == stage1_epochs + 1:
                print("Transitioning to Stage 2... Frozen Stage 1.")
                s1_path = save_dir / 'best_stage1.pth'
                if s1_path.exists():
                    model.load_state_dict(torch.load(s1_path), strict=False)
                    print("Loaded best Stage 1 weights.")
            
            loss, p = train_stage_2(model, train_loader, optimizer_s2, scaler, device, config, epoch)
            val_p = validate(model, val_loader, device, stage=2)
            
            writer.add_scalar('Stage2/Loss', loss, epoch)
            writer.add_scalar('Stage2/TrainPearson', p, epoch)
            writer.add_scalar('Stage2/ValPearson', val_p, epoch)
            
            print(f"S2 Epoch {epoch} | Loss: {loss:.4f} | Train P: {p:.4f} | Val P: {val_p:.4f}")
            
            if val_p > best_p_s2:
                best_p_s2 = val_p
                torch.save(model.state_dict(), save_dir / 'best_model.pth')
                
            # Visualize every 5 epochs
            if epoch % 5 == 0 or epoch == stage1_epochs + 1:
                visualize_samples(model, val_loader, device, stage=2, epoch=epoch, save_dir=save_dir)

            scheduler_s2.step()

    writer.close()
    print("Training Complete.")

if __name__ == '__main__':
    main()

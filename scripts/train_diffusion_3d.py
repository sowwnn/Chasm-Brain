"""
Training script for 3D Diffusion Model (Image-to-fMRI Generation)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
import os

# Import custom modules
from ARMNet.diffusion_3d_model import DiffusionARM3D, DiffusionManager3D
from ARMNet.loss import DiffusionPeakFocusedLoss
from dataset_diffusion import ImagefMRIDataset, collate_fn


class PearsonCorrelationLoss(nn.Module):
    """Pearson correlation loss (legacy, used for simple loss mode)"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        pred: [B, N]
        target: [B, N]
        """
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)

        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt((pred_centered ** 2).sum(dim=1) * (target_centered ** 2).sum(dim=1))

        corr = numerator / (denominator + 1e-8)
        return 1 - corr.mean()  # Loss = 1 - correlation


class Trainer:
    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup device
        self.device = torch.device(self.config['hardware']['device'])

        # Set seed
        if self.config['hardware']['deterministic']:
            torch.manual_seed(self.config['hardware']['seed'])
            np.random.seed(self.config['hardware']['seed'])
            torch.backends.cudnn.deterministic = True

        # Create directories
        self.checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        self.log_dir = Path(self.config['logging']['log_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Get CFG config (with defaults for backward compatibility)
        cfg_config = self.config.get('cfg', {})
        cfg_dropout = cfg_config.get('dropout', 0.1)
        self.cfg_scale = cfg_config.get('inference_scale', 3.0)

        # Build model with CFG support
        model_config = dict(self.config['model'])
        model_config['cfg_dropout'] = cfg_dropout
        self.model = DiffusionARM3D(**model_config).to(self.device)

        # Build diffusion manager
        self.diffusion_manager = DiffusionManager3D(
            model=self.model,
            timesteps=self.config['diffusion']['timesteps'],
            inference_timesteps=self.config['diffusion']['inference_timesteps'],
            device=self.device
        )
        
        # Build datasets
        self.train_dataset = ImagefMRIDataset(
            fmri_path=self.config['dataset']['fmri_path'],
            image_dir=self.config['dataset']['image_dir'],
            visual_features_path=self.config['dataset']['visual_features_path'],
            mean_fmri_path=self.config['dataset']['mean_fmri_path'],
            split='train',
            preload_fmri=self.config['dataset']['preload_fmri']
        )
        
        self.val_dataset = ImagefMRIDataset(
            fmri_path=self.config['dataset']['fmri_path'],
            image_dir=self.config['dataset']['image_dir'],
            visual_features_path=self.config['dataset']['visual_features_path'],
            mean_fmri_path=self.config['dataset']['mean_fmri_path'],
            split='val',
            preload_fmri=self.config['dataset']['preload_fmri']
        )
        
        # Build dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory'],
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory'],
            collate_fn=collate_fn
        )
        
        # Build optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=self.config['training']['betas'],
            eps=self.config['training']['eps']
        )
        
        # Build scheduler
        if self.config['training']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=self.config['training']['min_lr']
            )
        else:
            self.scheduler = None
        
        # Build losses
        loss_config = self.config['training'].get('loss', {})
        self.loss_type = loss_config.get('type', 'simple')

        # Always create simple losses for validation metrics
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.pearson_loss = PearsonCorrelationLoss()

        if self.loss_type == 'peak_focused':
            # Use DiffusionPeakFocusedLoss for training
            self.peak_focused_loss = DiffusionPeakFocusedLoss(
                alpha=loss_config.get('alpha', 2.0),
                tau=loss_config.get('tau', 0.3),
                pearson_weight=loss_config.get('pearson_weight', 0.3),
                std_weight=loss_config.get('std_weight', 0.5),
                diversity_weight=loss_config.get('diversity_weight', 1.0),
                l1_weight=loss_config.get('l1_weight', 0.1),
            )
            print(f"Using DiffusionPeakFocusedLoss (alpha={loss_config.get('alpha', 2.0)}, "
                  f"tau={loss_config.get('tau', 0.3)}, diversity_weight={loss_config.get('diversity_weight', 1.0)})")
        else:
            print("Using simple loss (MSE + L1 + Pearson)")
        
        # Mixed precision
        self.use_amp = self.config['training']['use_amp']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_metrics = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            vis_feat = batch['visual_feat'].to(self.device)
            fmri = batch['fmri'].to(self.device)
            mean_fmri = batch['mean_fmri'].to(self.device)

            # Get training tuple
            noisy_fmri, t, target_fmri = self.diffusion_manager.get_train_tuple(
                fmri, vis_feat, mean_fmri
            )

            # Forward pass - Get 3D representation for training loss
            with autocast(enabled=self.use_amp):
                # Get 3D prediction
                pred_3d = self.model(vis_feat, noisy_fmri, t, return_3d=True)

                # Convert target to 3D
                target_3d = self.model.reshape_3d(target_fmri)

                # Flatten and extract valid voxels (first 15724 of 16224)
                batch_size = pred_3d.shape[0]
                pred_3d_flat = pred_3d.view(batch_size, -1)
                target_3d_flat = target_3d.view(batch_size, -1)

                # Only compute loss on valid (non-padded) voxels
                pred_valid = pred_3d_flat[:, :self.model.fmri_dim]
                target_valid = target_3d_flat[:, :self.model.fmri_dim]

                if self.loss_type == 'peak_focused':
                    # Use DiffusionPeakFocusedLoss
                    # NOTE: target_valid is already centered (x_0 - mean)
                    # Pass mean_fmri=None to use per-sample mean for peak detection
                    # (since target is centered, its per-sample mean represents deviation from 0)
                    loss, loss_dict = self.peak_focused_loss(
                        pred_valid, target_valid, mean_fmri=None
                    )
                    # Accumulate metrics
                    for k, v in loss_dict.items():
                        total_metrics[k] = total_metrics.get(k, 0) + v
                else:
                    # Legacy simple loss
                    mse = self.mse_loss(pred_valid, target_valid)
                    l1 = self.l1_loss(pred_valid, target_valid)
                    pearson = self.pearson_loss(pred_valid, target_valid)

                    loss = (
                        self.config['training']['loss_weights']['mse'] * mse +
                        self.config['training']['loss_weights']['l1'] * l1 +
                        self.config['training']['loss_weights']['pearson'] * pearson
                    )
                    total_metrics['mse'] = total_metrics.get('mse', 0) + mse.item()
                    total_metrics['pearson'] = total_metrics.get('pearson', 0) + pearson.item()
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config['training']['gradient_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                if self.config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )

                self.optimizer.step()

            self.optimizer.zero_grad()

            # Update total loss
            total_loss += loss.item()

            # Update progress bar
            if self.loss_type == 'peak_focused':
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'wmse': f"{loss_dict['weighted_mse']:.4f}",
                    'pear': f"{loss_dict['pearson']:.3f}",
                    'div': f"{loss_dict['diversity']:.4f}"
                })
            else:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'mse': f"{total_metrics.get('mse', 0) / (batch_idx + 1):.4f}",
                    'pearson': f"{total_metrics.get('pearson', 0) / (batch_idx + 1):.4f}"
                })

            self.global_step += 1

        # Average metrics
        n_batches = len(self.train_loader)
        avg_metrics = {'loss': total_loss / n_batches}
        for k, v in total_metrics.items():
            avg_metrics[k] = v / n_batches

        return avg_metrics
    
    @torch.no_grad()
    def validate(self):
        """Validate model with CFG"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_pearson = 0

        # Track diversity metrics
        all_preds = []
        all_targets = []

        for batch in tqdm(self.val_loader, desc="Validation"):
            vis_feat = batch['visual_feat'].to(self.device)
            fmri = batch['fmri'].to(self.device)
            mean_fmri = batch['mean_fmri'].to(self.device)

            # Sample using DDIM with CFG
            pred_fmri = self.diffusion_manager.sample_ddim(
                vis_feat,
                mean_fmri,
                clip_value=self.config['diffusion']['clip_value'],
                eta=self.config['diffusion']['eta'],
                cfg_scale=self.cfg_scale  # Use CFG for stronger conditioning
            )

            # Compute losses
            mse = self.mse_loss(pred_fmri, fmri)
            pearson_loss = self.pearson_loss(pred_fmri, fmri)
            loss = mse + pearson_loss

            # Compute actual Pearson correlation (for monitoring)
            pred_c = pred_fmri - pred_fmri.mean(dim=1, keepdim=True)
            target_c = fmri - fmri.mean(dim=1, keepdim=True)
            pearson_corr = F.cosine_similarity(pred_c, target_c, dim=1).mean()

            total_loss += loss.item()
            total_mse += mse.item()
            total_pearson += pearson_corr.item()  # Log actual correlation, not loss

            # Collect for diversity analysis
            all_preds.append(pred_fmri.cpu())
            all_targets.append(fmri.cpu())

        avg_loss = total_loss / len(self.val_loader)
        avg_mse = total_mse / len(self.val_loader)
        avg_pearson = total_pearson / len(self.val_loader)

        # Compute diversity metrics (check for mode collapse)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        pred_std = all_preds.std(dim=0).mean().item()
        target_std = all_targets.std(dim=0).mean().item()

        return {
            'loss': avg_loss,
            'mse': avg_mse,
            'pearson': avg_pearson,
            'pred_std': pred_std,
            'target_std': target_std,
            'std_ratio': pred_std / (target_std + 1e-8)  # Should be close to 1.0
        }
    
    def save_checkpoint(self, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
        
        # Keep only last N checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if len(checkpoints) > self.config['checkpoint']['keep_last_n']:
            for old_ckpt in checkpoints[:-self.config['checkpoint']['keep_last_n']]:
                old_ckpt.unlink()
    
    def train(self):
        """Main training loop"""
        print("=" * 80)
        print("Starting training...")
        print(f"Total epochs: {self.config['training']['num_epochs']}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print("=" * 80)
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            if self.loss_type == 'peak_focused':
                print(f"\n[Epoch {epoch + 1}] Train - Loss: {train_metrics['loss']:.4f}, "
                      f"wMSE: {train_metrics.get('weighted_mse', 0):.4f}, "
                      f"Pearson: {train_metrics.get('pearson', 0):.4f}, "
                      f"Diversity: {train_metrics.get('diversity', 0):.4f}")
            else:
                print(f"\n[Epoch {epoch + 1}] Train - Loss: {train_metrics['loss']:.4f}, "
                      f"MSE: {train_metrics.get('mse', 0):.4f}, "
                      f"Pearson: {train_metrics.get('pearson', 0):.4f}")
            
            # Validate
            if (epoch + 1) % self.config['validation']['val_interval'] == 0:
                val_metrics = self.validate()
                print(f"[Epoch {epoch + 1}] Val - Loss: {val_metrics['loss']:.4f}, "
                      f"MSE: {val_metrics['mse']:.4f}, Corr: {val_metrics['pearson']:.4f}")
                print(f"  Diversity - Pred STD: {val_metrics['pred_std']:.4f}, "
                      f"Target STD: {val_metrics['target_std']:.4f}, "
                      f"Ratio: {val_metrics['std_ratio']:.4f} (should be ~1.0)")

                # Check if best
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                    print(f"New best validation loss: {self.best_val_loss:.4f}")
            else:
                is_best = False
            
            # Save checkpoint
            if (epoch + 1) % self.config['checkpoint']['save_interval'] == 0:
                self.save_checkpoint(is_best=is_best)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train 3D Diffusion Model for Image-to-fMRI")
    parser.add_argument('--config', type=str, default='configs/train_diffusion_3d.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Create trainer and train
    trainer = Trainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()

"""
Training script for HierarchicalARM V3: Dual-Stream Mamba Architecture
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import sys
import torch.nn.functional as F
import random
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.hierarchical_v3_1 import HierarchicalARM_V3
from model.hierarchical_v3_mlp import HierarchicalARM_V3_MLP
from model.hierarchical_v3_transformer import HierarchicalARM_V3_Transformer
from model.tben_official import HierarchicalARM_TBEn_Both, HierarchicalARM_V3_TBEn
from model.contrastive_loss import InfoNCELoss
from model.loss import PeakFocusedLoss
from data.dataset_v7_h import create_dataloaders_v7h
import matplotlib.pyplot as plt


SUBJECT_SPECIFIC_PREFIXES = (
    'stage1.what_head.',
    'stage1.where_head.',
    'stage1.brain_contrastive_head.',
    'stage2.roi_pos_embed',
    'stage2.global_head.4.',
    'stage2.local_head.',
    'stage2.remainder_head.',
)

HEAD_ONLY_PREFIXES = {
    1: (
        'stage1.what_head.',
        'stage1.where_head.',
        'stage1.brain_contrastive_head.',
    ),
    2: (
        'stage2.roi_pos_embed',
        'stage2.global_head.4.',
        'stage2.local_head.',
        'stage2.remainder_head.',
    ),
}


def load_cross_dataset_backbone(model, checkpoint_path):
    """Load only shape-compatible, non-subject-specific NSD parameters."""
    source = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    target = model.state_dict()
    transferred = {}
    skipped_subject = []
    skipped_shape = []
    for name, value in source.items():
        if name not in target:
            continue
        if name.startswith(SUBJECT_SPECIFIC_PREFIXES):
            skipped_subject.append(name)
        elif value.shape != target[name].shape:
            skipped_shape.append(name)
        else:
            transferred[name] = value
    model.load_state_dict(transferred, strict=False)
    return {
        'checkpoint': checkpoint_path,
        'transferred_tensors': len(transferred),
        'transferred_parameters': int(sum(value.numel() for value in transferred.values())),
        'skipped_subject_specific_tensors': len(skipped_subject),
        'skipped_shape_tensors': len(skipped_shape),
        'subject_specific_prefixes': list(SUBJECT_SPECIFIC_PREFIXES),
    }


def apply_head_only_policy(model, stage):
    """Freeze the transferred representation and expose only BOLD-specific heads."""
    prefixes = HEAD_ONLY_PREFIXES[stage]
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_names = []
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad = True
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError(f'head-only policy selected no parameters for Stage {stage}')
    return {
        'stage': stage,
        'trainable_tensors': len(trainable_names),
        'trainable_parameters': int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        'total_parameters': int(sum(p.numel() for p in model.parameters())),
        'trainable_prefixes': list(prefixes),
    }


def compute_pearson_batch(pred, target):
    if torch.isnan(pred).any() or torch.isnan(target).any():
        return 0.0
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()

    corrs = []
    for i in range(p.shape[0]):
        ps, ts = p[i], t[i]
        if ps.std() > 1e-9 and ts.std() > 1e-9:
            try:
                r = np.corrcoef(ps, ts)[0, 1]
                if not np.isnan(r):
                    corrs.append(r)
            except:
                continue
    return np.mean(corrs) if corrs else 0.0


def select_stage2_guidance(predicted, ground_truth, mode):
    """Select Stage 2 cluster conditioning for controlled ablations."""
    if mode == 'predicted':
        return predicted
    if mode == 'none':
        return torch.zeros_like(predicted)
    if mode == 'shuffle':
        if predicted.shape[0] < 2:
            return predicted
        return predicted[torch.randperm(predicted.shape[0], device=predicted.device)]
    if mode == 'oracle':
        return ground_truth
    raise ValueError(f"Unknown guidance_mode: {mode!r}")


def train_stage_1(model, dataloader, optimizer, scaler, device, config, epoch):
    """Train Stage 1: Dual-Stream Mamba (What + Where)"""
    model.train()
    total_loss, total_peak, total_cont, total_pearson = 0, 0, 0, 0

    peak_criterion = PeakFocusedLoss(
        alpha=config['loss'].get('peak_alpha', 3.0),
        tau=config['loss'].get('peak_tau', 0.4),
        pearson_weight=config['loss'].get('pearson_weight', 0.5),
        std_weight=config['loss'].get('std_weight', 0.5)
    ).to(device)

    cont_criterion = InfoNCELoss(temperature=0.15).to(device)
    cont_weight = config['loss'].get('contrastive_weight', 0.3)

    pbar = tqdm(dataloader, desc=f"Stage 1 Train {epoch}")
    for batch in pbar:
        cls_token = batch['cls_token'].to(device)
        patch_tokens = batch['patch_tokens'].to(device)
        gt_roi_means = batch['roi_means'].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=True):
            pred_roi_means, img_proj, brain_proj = model(
                cls_token, patch_tokens, stage=1, gt_roi_means=gt_roi_means
            )

            batch_mean = gt_roi_means.mean(dim=0, keepdim=True)
            loss_peak = peak_criterion(pred_roi_means, gt_roi_means, batch_mean)
            loss_cont = cont_criterion(img_proj, brain_proj)

            loss = loss_peak + cont_weight * loss_cont

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

        p = compute_pearson_batch(pred_roi_means, gt_roi_means)

        total_loss += loss.item()
        total_peak += loss_peak.item()
        total_cont += loss_cont.item()
        total_pearson += p

        pbar.set_postfix({
            'L_Tot': f"{loss.item():.4f}",
            'L_Peak': f"{loss_peak.item():.4f}",
            'L_Cont': f"{loss_cont.item():.4f}",
            'P': f"{p:.4f}"
        })

    n = len(dataloader)
    return total_loss/n, total_peak/n, total_cont/n, total_pearson/n


def train_stage_2(model, dataloader, optimizer, scaler, device, config, epoch):
    """Train Stage 2: Mamba with Global + Local branches"""
    model.train()
    model.stage1.eval()
    for param in model.stage1.parameters():
        param.requires_grad = False

    total_loss, total_mse, total_std, total_pearson = 0, 0, 0, 0
    mse_criterion = nn.MSELoss()
    pearson_weight = config['loss'].get('pearson_weight', 0.5)
    std_weight = config['loss'].get('std_weight', 0.2)
    kl_weight = config['loss'].get('kl_weight', 0.001)
    guidance_mode = config.get('experiment', {}).get('guidance_mode', 'predicted')

    pbar = tqdm(dataloader, desc=f"Stage 2 Train {epoch}")
    for batch in pbar:
        cls_token = batch['cls_token'].to(device)
        patch_tokens = batch['patch_tokens'].to(device)
        gt_voxels = batch['fmri'].to(device)
        gt_roi_means = batch['roi_means'].to(device)

        with torch.no_grad():
            predicted_guidance, _, _ = model.stage1(cls_token, patch_tokens)
            guidance = select_stage2_guidance(predicted_guidance, gt_roi_means, guidance_mode)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=True):
            _, pred_voxels, kl_loss = model(cls_token, patch_tokens, stage=2, gt_roi_means=guidance)

            loss_mse = mse_criterion(pred_voxels, gt_voxels)

            pred_centered = pred_voxels - pred_voxels.mean(dim=1, keepdim=True)
            target_centered = gt_voxels - gt_voxels.mean(dim=1, keepdim=True)
            pred_std = pred_centered.std(dim=1, keepdim=True) + 1e-8
            target_std = target_centered.std(dim=1, keepdim=True) + 1e-8

            pearson_corr = (pred_centered * target_centered).mean(dim=1) / (pred_std.squeeze() * target_std.squeeze())
            loss_pearson = (1 - pearson_corr).mean()
            loss_std = F.mse_loss(pred_std, target_std)

            loss = loss_mse + pearson_weight * loss_pearson + std_weight * loss_std + kl_weight * kl_loss

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
def validate(model, dataloader, device, stage, guidance_mode='predicted'):
    model.eval()
    total_pearson = 0
    total_mse = 0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        cls_token = batch['cls_token'].to(device)
        patch_tokens = batch['patch_tokens'].to(device)

        if stage == 1:
            target = batch['roi_means'].to(device)
            pred, _, _ = model(cls_token, patch_tokens, stage=1)
        else:
            target = batch['fmri'].to(device)
            predicted_guidance, _, _ = model.stage1(cls_token, patch_tokens)
            guidance = select_stage2_guidance(
                predicted_guidance, batch['roi_means'].to(device), guidance_mode
            )
            _, pred, _ = model(cls_token, patch_tokens, stage=2, gt_roi_means=guidance)

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
        return None, None

    mean_corr = sample_corrs.mean()
    median_corr = np.median(sample_corrs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(sample_corrs, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(mean_corr, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_corr:.3f}')
    axes[0, 0].axvline(median_corr, color='g', linestyle='--', linewidth=2, label=f'Median: {median_corr:.3f}')
    axes[0, 0].set_xlabel('Pearson Correlation')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Stage {stage} Epoch {epoch} - Pearson Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    sample_indices = np.random.choice(len(all_preds), min(3, len(all_preds)), replace=False)
    pred_samples = all_preds[sample_indices].flatten()
    target_samples = all_targets[sample_indices].flatten()

    axes[0, 1].scatter(target_samples, pred_samples, alpha=0.4, s=2, c='steelblue')
    axes[0, 1].plot([target_samples.min(), target_samples.max()],
                     [target_samples.min(), target_samples.max()], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('Ground Truth')
    axes[0, 1].set_ylabel('Prediction')
    axes[0, 1].set_title('Pred vs GT (3 random samples)')
    axes[0, 1].grid(alpha=0.3)

    best_idx = sample_corrs.argmax()
    worst_idx = sample_corrs.argmin()

    for idx, (sample_idx, title_prefix) in enumerate([(best_idx, 'Best'), (worst_idx, 'Worst')]):
        ax = axes[1, idx]
        pred_sample = all_preds[sample_idx]
        target_sample = all_targets[sample_idx]
        step = max(1, len(pred_sample) // 500)
        x = np.arange(len(pred_sample))

        ax.plot(x[::step], target_sample[::step], 'b-', alpha=0.6, label='Ground Truth')
        ax.plot(x[::step], pred_sample[::step], 'r-', alpha=0.6, label='Prediction')
        ax.set_title(f'{title_prefix} Sample - Pearson: {sample_corrs[sample_idx]:.3f}')
        ax.set_xlabel('Voxel/ROI Index')
        ax.set_ylabel('Activation')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    viz_path = save_dir / f'val_stage{stage}_epoch{epoch:03d}.png'
    plt.savefig(viz_path, dpi=100, bbox_inches='tight')

    if writer is not None:
        writer.add_figure(f'Validation/Stage{stage}', fig, epoch)

    plt.close(fig)
    return mean_corr, median_corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_hierarchical_v3.yaml')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=None)
    parser.add_argument('--load_stage1', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--guidance_mode', choices=['predicted', 'none', 'shuffle', 'oracle'])
    parser.add_argument('--vae_mode', choices=['none', 'deterministic', 'stochastic'])
    parser.add_argument('--kl_weight', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--backbone', choices=['mamba', 'mlp', 'transformer', 'tben', 'tben_both'])
    parser.add_argument('--pretrained', type=str,
                        help='Cross-dataset source checkpoint; loads compatible backbone tensors only')
    parser.add_argument('--adaptation', choices=['full', 'head_only'], default='full')
    parser.add_argument('--lr', type=float)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    experiment = config.setdefault('experiment', {})
    if args.guidance_mode is not None:
        experiment['guidance_mode'] = args.guidance_mode
    if args.vae_mode is not None:
        experiment['vae_mode'] = args.vae_mode
    if args.kl_weight is not None:
        config['loss']['kl_weight'] = args.kl_weight
    if args.save_dir is not None:
        config['training']['save_dir'] = args.save_dir
    if args.backbone is not None:
        experiment['backbone'] = args.backbone
    if args.lr is not None:
        config['training']['lr'] = args.lr

    seed = args.seed if args.seed is not None else experiment.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CLI --stage overrides config; config 'stage' key is fallback
    train_stage = args.stage if args.stage is not None else config['training'].get('stage')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'visualizations').mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / 'logs'))

    print(f"Training HierarchicalARM V3 on device: {device}")

    # Load Data with HUWNG dataset format
    train_loader, val_loader = create_dataloaders_v7h(
        huwng_base_path=config['data']['huwng_base_path'],
        train_embeddings_path=config['data'].get('train_embeddings_path'),  # Optional
        test_embeddings_path=config['data'].get('test_embeddings_path'),    # Optional
        subjects=[config['data']['subject']],
        batch_size=config['training']['batch_size'],
        voxels_per_cluster=config['model']['voxels_per_cluster'],
        roi_top_k_percent=config['data'].get('roi_top_k_percent', 0.7),
        average_trials_train=config['data']['average_trials_train'],
        average_trials_val=config['data']['average_trials_val'],
        augment_noise_train=config['data']['augment_noise_train'],
        noise_std=config['data'].get('noise_std', 0.1),
        train_image_ids_path=config['data'].get('train_image_ids_path'),
        test_image_ids_path=config['data'].get('test_image_ids_path'),
        concat_train_val=train_stage == 1,
        dinov2_layer=config['data'].get('dinov2_layer', -1),  # Default to last layer
        train_fmri_path=config['data'].get('train_fmri_path'),
        test_fmri_path=config['data'].get('validation_fmri_path', config['data'].get('test_fmri_path')),
        train_sample_indices_path=config['data'].get('train_sample_indices_path'),
        test_sample_indices_path=config['data'].get('validation_sample_indices_path'),
    )

    # Initialize the requested matched sequence backbone.
    backbone = experiment.get('backbone', 'mamba')
    model_classes = {
        'mamba': HierarchicalARM_V3,
        'mlp': HierarchicalARM_V3_MLP,
        'transformer': HierarchicalARM_V3_Transformer,
        'tben': HierarchicalARM_V3_TBEn,
        'tben_both': HierarchicalARM_TBEn_Both,
    }
    model_class = model_classes[backbone]
    model = model_class(
        cls_dim=config['model']['cls_dim'],
        patch_dim=config['model']['patch_dim'],
        num_patches=config['model']['num_patches'],
        fmri_dim=config['model']['fmri_dim'],
        voxels_per_cluster=config['model']['voxels_per_cluster'],
        hidden_dim_1=config['model']['hidden_dim_1'],
        depth_what=config['model']['depth_what'],
        depth_where=config['model']['depth_where'],
        contrastive_dim=config['model']['contrastive_dim'],
        embed_dim_2=config['model']['embed_dim_2'],
        depth_2=config['model']['depth_2'],
        num_global_queries=config['model']['num_global_queries'],
        dropout=config['model'].get('dropout', 0.1),
        latent_dim=config['model'].get('latent_dim', 128),
        vae_mode=experiment.get('vae_mode', 'stochastic'),
        **({'global_hidden': config['model']['global_hidden']}
           if 'global_hidden' in config['model'] and backbone == 'mamba' else {}),
        **({
            'tben_hidden_dim': config['model'].get('tben_hidden_dim', 768),
            'tben_nheads': config['model'].get('tben_nheads', 16),
            'tben_dim_feedforward': config['model'].get('tben_dim_feedforward', 1024),
            'tben_enc_layers': config['model'].get('tben_enc_layers', 0),
            'tben_dec_layers': config['model'].get('tben_dec_layers', 1),
            'tben_pre_norm': config['model'].get('tben_pre_norm', True),
            'tben_enc_output_layer': config['model'].get('tben_enc_output_layer', 1),
        } if backbone in ('tben', 'tben_both') else {}),
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Num ROIs: {model.num_rois}")
    print(f"Experiment: guidance={experiment.get('guidance_mode', 'predicted')}, "
          f"vae={experiment.get('vae_mode', 'stochastic')}, "
          f"backbone={backbone}, kl_weight={config['loss'].get('kl_weight', 0.001)}, seed={seed}")

    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['precision'] == 'fp16')

    stage1_epochs = config['training'].get('stage_1_epochs', 150)
    total_epochs = config['training']['max_epochs']

    if train_stage == 1:
        total_epochs = stage1_epochs

    if train_stage == 2 and args.load_stage1 is None and args.resume is None:
        print("ERROR: --load_stage1 or --resume required for Stage 2")
        return

    transfer_manifest = None
    if args.resume is not None:
        print(f"Resuming from: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
    elif train_stage == 2 and args.load_stage1 is not None:
        print(f"Loading Stage 1 from: {args.load_stage1}")
        model.load_state_dict(torch.load(args.load_stage1, map_location=device), strict=False)
    elif args.pretrained is not None:
        print(f"Loading cross-dataset backbone from: {args.pretrained}")
        transfer_manifest = load_cross_dataset_backbone(model, args.pretrained)
        print(json.dumps(transfer_manifest, indent=2))

    adaptation_manifest = None
    if args.adaptation == 'head_only':
        adaptation_manifest = apply_head_only_policy(model, train_stage)
        print(json.dumps(adaptation_manifest, indent=2))
    manifest = {
        'adaptation': args.adaptation,
        'transfer': transfer_manifest,
        'policy': adaptation_manifest,
    }
    (save_dir / f'transfer_manifest_stage{train_stage}.json').write_text(
        json.dumps(manifest, indent=2) + '\n'
    )

    optimizer_s1 = optim.AdamW(model.stage1.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    optimizer_s2 = optim.AdamW(list(model.stage2.parameters()) + list(model.patch_pooler.parameters()),
                               lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])

    best_p_s1, best_p_s2 = -1.0, -1.0
    no_improve_s1, no_improve_s2 = 0, 0
    patience = config['training'].get('patience', 15)

    print("=== Starting Training V3 ===")
    print(f"Architecture: Dual-Stream {backbone.upper()} (Stage 1) -> "
          f"{backbone.upper()} Global+Local (Stage 2)")

    for epoch in range(1, total_epochs + 1):
        if train_stage == 1:
            should_train_s1 = True
            should_train_s2 = False
        elif train_stage == 2:
            should_train_s1 = False
            should_train_s2 = True
        else:
            should_train_s1 = epoch <= stage1_epochs
            should_train_s2 = epoch > stage1_epochs

        if should_train_s1:
            l_tot, l_peak, l_cont, p = train_stage_1(model, train_loader, optimizer_s1, scaler, device, config, epoch)
            val_p, val_mse, val_preds, val_targets = validate(model, val_loader, device, stage=1)

            writer.add_scalar('Stage1/LossTotal', l_tot, epoch)
            writer.add_scalar('Stage1/LossPeak', l_peak, epoch)
            writer.add_scalar('Stage1/LossCont', l_cont, epoch)
            writer.add_scalar('Stage1/TrainPearson', p, epoch)
            writer.add_scalar('Stage1/ValPearson', val_p, epoch)
            writer.add_scalar('Stage1/ValMSE', val_mse, epoch)

            # Log fusion weights (dual-stream backbones only; TBEn has no gate)
            if hasattr(model.stage1, 'get_fusion_weights'):
                with torch.no_grad():
                    sample_batch = next(iter(val_loader))
                    cls_t = sample_batch['cls_token'][:4].to(device)
                    patch_t = sample_batch['patch_tokens'][:4].to(device)
                    weights = model.get_stage1_weights(cls_t, patch_t)
                    writer.add_scalar('Stage1/FusionWeight_What', weights[:, 0].mean().item(), epoch)
                    writer.add_scalar('Stage1/FusionWeight_Where', weights[:, 1].mean().item(), epoch)

            print(f"S1 Ep {epoch} | L_Tot:{l_tot:.3f} L_Peak:{l_peak:.3f} L_Cont:{l_cont:.3f} | Tr P:{p:.3f} Val P:{val_p:.3f} MSE:{val_mse:.3f}")

            viz_interval = config['training'].get('viz_interval', 5)
            if epoch % viz_interval == 0 or val_p > best_p_s1:
                visualize_validation(val_preds, val_targets, epoch, stage=1,
                                   save_dir=save_dir / 'visualizations', writer=writer)

            if val_p > best_p_s1:
                best_p_s1 = val_p
                no_improve_s1 = 0
                torch.save(model.state_dict(), save_dir / 'best_stage1.pth')
                print(f"✓ Saved best Stage 1: {val_p:.4f}")
            else:
                no_improve_s1 += 1
                if no_improve_s1 >= patience:
                    print(f"Early stopping Stage 1 at epoch {epoch} (no improvement for {patience} epochs)")
                    break

        if should_train_s2:
            if train_stage is None and epoch == stage1_epochs + 1:
                print(">>> Transitioning to Stage 2...")
                s1_path = save_dir / 'best_stage1.pth'
                if s1_path.exists():
                    model.load_state_dict(torch.load(s1_path), strict=False)
                    print("Loaded best Stage 1 weights.")

            loss, loss_mse, loss_std, p = train_stage_2(model, train_loader, optimizer_s2, scaler, device, config, epoch)
            val_p, val_mse, val_preds, val_targets = validate(
                model, val_loader, device, stage=2,
                guidance_mode=experiment.get('guidance_mode', 'predicted')
            )

            writer.add_scalar('Stage2/Loss', loss, epoch)
            writer.add_scalar('Stage2/TrainMSE', loss_mse, epoch)
            writer.add_scalar('Stage2/LossStd', loss_std, epoch)
            writer.add_scalar('Stage2/TrainPearson', p, epoch)
            writer.add_scalar('Stage2/ValPearson', val_p, epoch)
            writer.add_scalar('Stage2/ValMSE', val_mse, epoch)

            print(f"S2 Ep {epoch} | Loss:{loss:.3f} MSE:{loss_mse:.3f} Std:{loss_std:.3f} | Tr P:{p:.3f} Val P:{val_p:.3f} MSE:{val_mse:.3f}")

            viz_interval = config['training'].get('viz_interval', 5)
            if epoch % viz_interval == 0 or val_p > best_p_s2:
                visualize_validation(val_preds, val_targets, epoch, stage=2,
                                   save_dir=save_dir / 'visualizations', writer=writer)

            if val_p > best_p_s2:
                best_p_s2 = val_p
                no_improve_s2 = 0
                torch.save(model.state_dict(), save_dir / 'best_model.pth')
                print(f"✓ Saved best Stage 2: {val_p:.4f}")
            else:
                no_improve_s2 += 1
                if no_improve_s2 >= patience:
                    print(f"Early stopping Stage 2 at epoch {epoch} (no improvement for {patience} epochs)")
                    break

    print("Training Complete.")
    writer.close()


if __name__ == '__main__':
    main()

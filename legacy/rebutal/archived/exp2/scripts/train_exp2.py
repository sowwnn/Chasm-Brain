"""
Exp 2 Phase 1: Train shared backbone on subj01 + subj02.

Each iteration alternates one batch from subj01 and one from subj02.
Both subjects use the same shared backbone; subject-specific heads route output.

Usage:
  python rebutal/exp2/scripts/train_exp2.py --config rebutal/exp2/configs/exp2.yaml --stage 1
  python rebutal/exp2/scripts/train_exp2.py --config rebutal/exp2/configs/exp2.yaml --stage 2 \
      --load_stage1 rebutal/exp2/checkpoints/phase1/best_stage1.pth
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from model.hierarchical_v3_multisubj import HierarchicalARM_MultiSubject
from model.contrastive_loss import InfoNCELoss
from model.loss import PeakFocusedLoss
from data.dataset_v7_h import load_neuroflux_data_v7h
from torch.utils.data import DataLoader


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def compute_pearson(pred, target):
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    corrs = []
    for i in range(p.shape[0]):
        if p[i].std() > 1e-9 and t[i].std() > 1e-9:
            r = np.corrcoef(p[i], t[i])[0, 1]
            if np.isfinite(r):
                corrs.append(r)
    return float(np.mean(corrs)) if corrs else 0.0


def make_loader(cfg, subject_id, split, batch_size):
    avg = cfg['data']['average_trials_val'] if split == 'test' else cfg['data']['average_trials_train']
    noise = cfg['data']['augment_noise_train'] if split == 'train' else False
    ds = load_neuroflux_data_v7h(
        huwng_base_path=cfg['data']['huwng_base_path'],
        subjects=[subject_id],
        split=split,
        average_trials=avg,
        voxels_per_cluster=cfg['model']['voxels_per_cluster'],
        roi_top_k_percent=cfg['data']['roi_top_k_percent'],
        augment_noise=noise,
        noise_std=cfg['data'].get('noise_std', 0.1),
        dinov2_layer=cfg['data'].get('dinov2_layer', -1),
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=(split == 'train'),
        num_workers=4, pin_memory=True,
        persistent_workers=True, drop_last=(split == 'train'),
    )
    return loader


# ─────────────────────────────────────────
# Train / Val steps
# ─────────────────────────────────────────

def train_step_s1(model, batch, subject_id, optimizer, scaler, device, cfg):
    cls_token = batch['cls_token'].to(device)
    patch_tokens = batch['patch_tokens'].to(device)
    gt_roi = batch['roi_means'].to(device)

    peak_criterion = PeakFocusedLoss(
        alpha=cfg['loss']['peak_alpha'], tau=cfg['loss']['peak_tau'],
        pearson_weight=cfg['loss']['pearson_weight'], std_weight=cfg['loss']['std_weight'],
    ).to(device)
    cont_criterion = InfoNCELoss(temperature=0.15).to(device)

    with torch.amp.autocast(device_type='cuda'):
        pred_roi, img_proj, brain_proj = model(cls_token, patch_tokens, subject_id, stage=1, gt_roi_means=gt_roi)
        batch_mean = gt_roi.mean(dim=0, keepdim=True)
        loss_peak = peak_criterion(pred_roi, gt_roi, batch_mean)
        loss_cont = cont_criterion(img_proj, brain_proj)
        loss = loss_peak + cfg['loss']['contrastive_weight'] * loss_cont

    scaler.scale(loss).backward()
    return loss.item(), compute_pearson(pred_roi, gt_roi)


def train_step_s2(model, batch, subject_id, optimizer, scaler, device, cfg):
    cls_token = batch['cls_token'].to(device)
    patch_tokens = batch['patch_tokens'].to(device)
    gt_voxels = batch['fmri'].to(device)

    with torch.no_grad():
        guidance, _, _ = model(cls_token, patch_tokens, subject_id, stage=1)

    with torch.amp.autocast(device_type='cuda'):
        _, pred_voxels, kl_loss = model(cls_token, patch_tokens, subject_id, stage=2, gt_roi_means=guidance)

        loss_mse = F.mse_loss(pred_voxels, gt_voxels)
        pc = pred_voxels - pred_voxels.mean(dim=1, keepdim=True)
        tc = gt_voxels - gt_voxels.mean(dim=1, keepdim=True)
        ps = pc.std(dim=1, keepdim=True) + 1e-8
        ts = tc.std(dim=1, keepdim=True) + 1e-8
        loss_pearson = (1 - (pc * tc).mean(dim=1) / (ps.squeeze() * ts.squeeze())).mean()
        loss_std = F.mse_loss(ps, ts)
        loss = (loss_mse
                + cfg['loss']['pearson_weight'] * loss_pearson
                + cfg['loss']['std_weight'] * loss_std
                + cfg['loss']['kl_weight'] * kl_loss)

    scaler.scale(loss).backward()
    return loss.item(), compute_pearson(pred_voxels, gt_voxels)


@torch.no_grad()
def validate(model, loader, subject_id, device, stage):
    model.eval()
    total_p, total_mse, n = 0.0, 0.0, 0
    for batch in loader:
        cls_token = batch['cls_token'].to(device)
        patch_tokens = batch['patch_tokens'].to(device)
        if stage == 1:
            target = batch['roi_means'].to(device)
            pred, _, _ = model(cls_token, patch_tokens, subject_id, stage=1)
        else:
            target = batch['fmri'].to(device)
            _, pred, _ = model(cls_token, patch_tokens, subject_id, stage=2)
        total_p += compute_pearson(pred, target)
        total_mse += F.mse_loss(pred, target).item()
        n += 1
    return total_p / max(n, 1), total_mse / max(n, 1)


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='rebutal/exp2/configs/exp2.yaml')
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True)
    parser.add_argument('--load_stage1', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(cfg['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / 'logs'))

    train_subjects = cfg['data']['train_subjects']
    subject_fmri_dims = {sid: cfg['model']['subject_fmri_dims'][sid] for sid in train_subjects}
    bs = cfg['training']['batch_size']

    print(f"Exp 2 Phase 1 | Stage {args.stage} | Subjects: {train_subjects} | Device: {device}")

    # Build dataloaders per subject
    train_loaders = {sid: make_loader(cfg, sid, 'train', bs) for sid in train_subjects}
    val_loaders   = {sid: make_loader(cfg, sid, 'test',  bs) for sid in train_subjects}

    # Model
    model = HierarchicalARM_MultiSubject(
        subject_fmri_dims=subject_fmri_dims,
        cls_dim=cfg['model']['cls_dim'],
        patch_dim=cfg['model']['patch_dim'],
        num_patches=cfg['model']['num_patches'],
        voxels_per_cluster=cfg['model']['voxels_per_cluster'],
        hidden_dim_1=cfg['model']['hidden_dim_1'],
        depth_what=cfg['model']['depth_what'],
        depth_where=cfg['model']['depth_where'],
        contrastive_dim=cfg['model']['contrastive_dim'],
        embed_dim_2=cfg['model']['embed_dim_2'],
        latent_dim=cfg['model']['latent_dim'],
        depth_2=cfg['model']['depth_2'],
        num_global_queries=cfg['model']['num_global_queries'],
        dropout=cfg['model']['dropout'],
        branch_mode=cfg['model'].get('branch_mode', 'both'),
    ).to(device)

    total, _ = model.param_count()
    print(f"Model params: {total:,}")

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")
    elif args.stage == 2 and args.load_stage1:
        model.load_state_dict(torch.load(args.load_stage1, map_location=device), strict=False)
        print(f"Loaded Stage 1 weights from {args.load_stage1}")
    elif args.stage == 2:
        print("ERROR: --load_stage1 required for Stage 2"); return

    scaler = torch.amp.GradScaler('cuda', enabled=cfg['training']['precision'] == 'fp16')

    if args.stage == 1:
        # Optimize shared Stage1 backbone + all subject heads Stage1 params
        s1_params = (list(model.shared_s1.parameters())
                     + [p for h in model.subject_heads.values() for p in h.s1_what_head.parameters()]
                     + [p for h in model.subject_heads.values() for p in h.s1_where_head.parameters()]
                     + [p for h in model.subject_heads.values() for p in h.s1_brain_contrastive.parameters()])
        optimizer = optim.AdamW(s1_params, lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
        num_epochs = cfg['training']['stage_1_epochs']
    else:
        # Optimize shared Stage2 + patch_pooler + all subject heads Stage2 params
        # Stage1 is frozen during Stage2 training
        for p in model.shared_s1.parameters():
            p.requires_grad = False
        for h in model.subject_heads.values():
            for p in list(h.s1_what_head.parameters()) + list(h.s1_where_head.parameters()) + list(h.s1_brain_contrastive.parameters()):
                p.requires_grad = False

        # Collect all params with requires_grad=True (shared S2 + per-subject S2 heads)
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'],
        )
        num_epochs = cfg['training']['stage_2_epochs']

    best_val_p = -1.0
    best_val_mse = float('inf')
    best_val_per_subj = {}
    best_epoch = 0
    no_improve = 0
    patience = cfg['training']['patience']

    for epoch in range(1, num_epochs + 1):
        model.train()
        if args.stage == 2:
            model.shared_s1.eval()

        # Iterate alternating batches from each subject
        iters = {sid: iter(train_loaders[sid]) for sid in train_subjects}
        n_batches = min(len(train_loaders[sid]) for sid in train_subjects)

        epoch_loss, epoch_p = 0.0, 0.0
        pbar = tqdm(range(n_batches), desc=f"S{args.stage} Ep{epoch}")
        for _ in pbar:
            optimizer.zero_grad()
            step_loss, step_p = 0.0, 0.0
            for sid in train_subjects:
                try:
                    batch = next(iters[sid])
                except StopIteration:
                    continue
                if args.stage == 1:
                    l, p = train_step_s1(model, batch, sid, optimizer, scaler, device, cfg)
                else:
                    l, p = train_step_s2(model, batch, sid, optimizer, scaler, device, cfg)
                step_loss += l
                step_p += p

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += step_loss / len(train_subjects)
            epoch_p    += step_p    / len(train_subjects)
            pbar.set_postfix({'L': f'{step_loss/len(train_subjects):.4f}',
                              'P': f'{step_p/len(train_subjects):.4f}'})

        epoch_loss /= n_batches
        epoch_p    /= n_batches

        # Validate on all training subjects
        val_ps, val_mses = [], []
        for sid in train_subjects:
            vp, vm = validate(model, val_loaders[sid], sid, device, args.stage)
            val_ps.append(vp)
            val_mses.append(vm)
            writer.add_scalar(f'Val/S{args.stage}_P_subj{sid:02d}', vp, epoch)
        mean_val_p   = np.mean(val_ps)
        mean_val_mse = np.mean(val_mses)

        writer.add_scalar(f'Train/S{args.stage}_Loss', epoch_loss, epoch)
        writer.add_scalar(f'Train/S{args.stage}_Pearson', epoch_p, epoch)
        writer.add_scalar(f'Val/S{args.stage}_Pearson_mean', mean_val_p, epoch)
        writer.add_scalar(f'Val/S{args.stage}_MSE_mean', mean_val_mse, epoch)

        subj_str = ' '.join(f's{sid}:{vp:.3f}' for sid, vp in zip(train_subjects, val_ps))
        print(f"S{args.stage} Ep{epoch:3d} | L:{epoch_loss:.4f} TrP:{epoch_p:.3f} | "
              f"Val {subj_str} mean:{mean_val_p:.3f} MSE:{mean_val_mse:.3f}")

        if mean_val_p > best_val_p:
            best_val_p = mean_val_p
            best_val_mse = mean_val_mse
            best_val_per_subj = {int(sid): {'pearson': float(vp), 'mse': float(vm)}
                                 for sid, vp, vm in zip(train_subjects, val_ps, val_mses)}
            best_epoch = epoch
            no_improve = 0
            ckpt = save_dir / f'best_stage{args.stage}.pth'
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ Saved best Stage {args.stage}: {mean_val_p:.4f} → {ckpt}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Done. Best val Pearson (mean over {train_subjects}): {best_val_p:.4f}")
    writer.close()

    # Dump summary JSON to results/ next to checkpoints
    results_dir = save_dir.parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = save_dir.name  # e.g. swap_subj1
    summary = {
        'tag': tag,
        'config': args.config,
        'stage': args.stage,
        'branch_mode': cfg['model'].get('branch_mode', 'both'),
        'train_subjects': list(train_subjects),
        'best_epoch': best_epoch,
        'best_val_pearson_mean': float(best_val_p),
        'best_val_mse_mean': float(best_val_mse),
        'best_val_per_subject': best_val_per_subj,
        'checkpoint': str(save_dir / f'best_stage{args.stage}.pth'),
    }
    out = results_dir / f'{tag}_stage{args.stage}_summary.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Wrote summary → {out}")


if __name__ == '__main__':
    main()

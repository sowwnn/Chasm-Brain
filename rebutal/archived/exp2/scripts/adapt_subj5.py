"""
Exp 2 Phase 2: Freeze shared backbone, train only subj05 head.
Then evaluate on subj05 test set and save results.

Usage:
  python rebutal/exp2/scripts/adapt_subj5.py \
      --config rebutal/exp2/configs/exp2.yaml \
      --checkpoint rebutal/exp2/checkpoints/phase1/best_stage2.pth
"""

import sys
import yaml
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from model.hierarchical_v3_multisubj import HierarchicalARM_MultiSubject
from model.loss import PeakFocusedLoss
from model.contrastive_loss import InfoNCELoss
from data.dataset_v7_h import load_neuroflux_data_v7h
from torch.utils.data import DataLoader


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


@torch.no_grad()
def evaluate(model, loader, subject_id, device):
    model.eval()
    all_preds, all_targets = [], []
    for batch in tqdm(loader, desc='Evaluating'):
        cls_token    = batch['cls_token'].to(device)
        patch_tokens = batch['patch_tokens'].to(device)
        target       = batch['fmri'].to(device)
        _, pred, _   = model(cls_token, patch_tokens, subject_id, stage=2)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Per-sample Pearson
    corrs = []
    for i in range(len(preds)):
        if preds[i].std() > 1e-9 and targets[i].std() > 1e-9:
            r = np.corrcoef(preds[i], targets[i])[0, 1]
            if np.isfinite(r):
                corrs.append(r)
    pearson_mean = float(np.mean(corrs))
    pearson_std  = float(np.std(corrs))
    mse          = float(np.mean((preds - targets) ** 2))
    return pearson_mean, pearson_std, mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='rebutal/exp2/configs/exp2.yaml')
    parser.add_argument('--checkpoint', required=True, help='Phase 1 best_stage2.pth')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(cfg['adapt']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    writer   = SummaryWriter(log_dir=str(save_dir / 'logs'))

    adapt_sid  = cfg['adapt']['subject_id']   # 5
    adapt_fdim = cfg['adapt']['fmri_dim']     # 13039
    bs         = cfg['adapt']['batch_size']
    train_subjects = cfg['data']['train_subjects']

    print(f"Phase 2: Adapt to subj{adapt_sid:02d} | backbone frozen: {cfg['adapt']['backbone_frozen']}")

    # ── Build model with Phase 1 subjects (need their heads to load checkpoint) ──
    subject_fmri_dims = {sid: cfg['model']['subject_fmri_dims'][sid] for sid in train_subjects}
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
    ).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Add subj05 head ──
    model.add_subject(adapt_sid, adapt_fdim, device=device)

    # ── Optionally freeze backbone ──
    if cfg['adapt']['backbone_frozen']:
        model.freeze_backbone()
    total, trainable = model.param_count()
    print(f"Backbone frozen={cfg['adapt']['backbone_frozen']} | trainable: {trainable:,} / {total:,}")

    # ── Data ──
    train_ds = load_neuroflux_data_v7h(
        huwng_base_path=cfg['data']['huwng_base_path'],
        subjects=[adapt_sid],
        split='train',
        average_trials=cfg['data']['average_trials_train'],
        voxels_per_cluster=cfg['model']['voxels_per_cluster'],
        roi_top_k_percent=cfg['data']['roi_top_k_percent'],
        augment_noise=cfg['data']['augment_noise_train'],
        noise_std=cfg['data'].get('noise_std', 0.1),
        dinov2_layer=cfg['data'].get('dinov2_layer', -1),
    )
    val_ds = load_neuroflux_data_v7h(
        huwng_base_path=cfg['data']['huwng_base_path'],
        subjects=[adapt_sid],
        split='test',
        average_trials=cfg['data']['average_trials_val'],
        voxels_per_cluster=cfg['model']['voxels_per_cluster'],
        roi_top_k_percent=cfg['data']['roi_top_k_percent'],
        augment_noise=False,
        dinov2_layer=cfg['data'].get('dinov2_layer', -1),
    )
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Optimizer ──
    head5 = model.subject_heads[str(adapt_sid)]
    if cfg['adapt']['backbone_frozen']:
        opt_params = list(head5.parameters())
        optimizer = optim.AdamW(opt_params, lr=cfg['adapt']['lr'],
                                weight_decay=cfg['adapt']['weight_decay'])
    else:
        backbone_lr = cfg['adapt'].get('backbone_lr', cfg['adapt']['lr'] * 0.1)
        optimizer = optim.AdamW([
            {'params': head5.parameters(),               'lr': cfg['adapt']['lr']},
            {'params': model.shared_s1.parameters(),     'lr': backbone_lr},
            {'params': model.shared_s2.parameters(),     'lr': backbone_lr},
            {'params': model.patch_pooler.parameters(),  'lr': backbone_lr},
        ], weight_decay=cfg['adapt']['weight_decay'])
        opt_params = [p for g in optimizer.param_groups for p in g['params']]
        print(f"Unfrozen mode | head_lr={cfg['adapt']['lr']} backbone_lr={backbone_lr}")
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['training']['precision'] == 'fp16')

    best_val_p = -1.0
    no_improve = 0
    patience = cfg['training']['patience']
    peak_criterion = PeakFocusedLoss(
        alpha=cfg['loss']['peak_alpha'], tau=cfg['loss']['peak_tau'],
        pearson_weight=cfg['loss']['pearson_weight'], std_weight=cfg['loss']['std_weight'],
    ).to(device)
    cont_criterion = InfoNCELoss(temperature=0.15).to(device)

    for epoch in range(1, cfg['adapt']['epochs'] + 1):
        model.train()
        if cfg['adapt']['backbone_frozen']:
            model.shared_s1.eval()
            model.shared_s2.eval()
            model.patch_pooler.eval()

        epoch_loss, epoch_p = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Adapt Ep{epoch}")
        for batch in pbar:
            cls_token    = batch['cls_token'].to(device)
            patch_tokens = batch['patch_tokens'].to(device)
            gt_voxels    = batch['fmri'].to(device)
            gt_roi       = batch['roi_means'].to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                guidance, _, _ = model(cls_token, patch_tokens, adapt_sid, stage=1)

            with torch.amp.autocast(device_type='cuda'):
                _, pred_voxels, kl_loss = model(cls_token, patch_tokens, adapt_sid,
                                                 stage=2, gt_roi_means=guidance)
                loss_mse = F.mse_loss(pred_voxels, gt_voxels)
                pc = pred_voxels - pred_voxels.mean(1, keepdim=True)
                tc = gt_voxels   - gt_voxels.mean(1, keepdim=True)
                ps = pc.std(1, keepdim=True) + 1e-8
                ts = tc.std(1, keepdim=True) + 1e-8
                loss_pearson = (1 - (pc * tc).mean(1) / (ps.squeeze() * ts.squeeze())).mean()
                loss_std     = F.mse_loss(ps, ts)
                loss = (loss_mse
                        + cfg['loss']['pearson_weight'] * loss_pearson
                        + cfg['loss']['std_weight']     * loss_std
                        + cfg['loss']['kl_weight']      * kl_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(opt_params, cfg['training']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            p = compute_pearson(pred_voxels, gt_voxels)
            epoch_loss += loss.item()
            epoch_p    += p
            pbar.set_postfix({'L': f'{loss.item():.4f}', 'P': f'{p:.4f}'})

        epoch_loss /= len(train_loader)
        epoch_p    /= len(train_loader)

        # Validate
        val_p, val_p_std, val_mse = evaluate(model, val_loader, adapt_sid, device)
        writer.add_scalar('Adapt/TrainLoss',   epoch_loss, epoch)
        writer.add_scalar('Adapt/TrainPearson', epoch_p,   epoch)
        writer.add_scalar('Adapt/ValPearson',   val_p,     epoch)
        writer.add_scalar('Adapt/ValMSE',       val_mse,   epoch)

        print(f"Ep{epoch:3d} | L:{epoch_loss:.4f} TrP:{epoch_p:.3f} | "
              f"ValP:{val_p:.4f}±{val_p_std:.4f} MSE:{val_mse:.4f}")

        if val_p > best_val_p:
            best_val_p = val_p
            no_improve = 0
            torch.save(model.state_dict(), save_dir / 'best_adapted.pth')
            print(f"  ✓ Saved best adapted model: {val_p:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # ── Final evaluation ──
    model.load_state_dict(torch.load(save_dir / 'best_adapted.pth', map_location=device))
    final_p, final_p_std, final_mse = evaluate(model, val_loader, adapt_sid, device)

    results = {
        'experiment': 'Exp2_CrossSubject',
        'setup': f'train_subj{train_subjects}_test_subj{adapt_sid:02d}',
        'backbone_frozen': cfg['adapt']['backbone_frozen'],
        f'subj{adapt_sid:02d}_test': {
            'pearson_mean': final_p,
            'pearson_std':  final_p_std,
            'mse':          final_mse,
        }
    }

    setup_tag = ''.join(str(s) for s in train_subjects) + f'-{adapt_sid}'
    results_path = Path(f'rebutal/exp2/results/exp2_results_{setup_tag}.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print(f"FINAL  subj{adapt_sid:02d} test | Pearson: {final_p:.4f} ± {final_p_std:.4f} | MSE: {final_mse:.4f}")
    print(f"Results saved → {results_path}")
    writer.close()


if __name__ == '__main__':
    main()

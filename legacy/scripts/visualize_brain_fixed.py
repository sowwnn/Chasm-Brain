"""
Fixed Brain Visualization Script
=================================

Visualize fMRI predictions with CORRECTED affine matrix for proper anatomical alignment.

Key fix: Use reference NIfTI affine + bbox offset to ensure brain alignment.
"""

import sys
sys.path.append('/home/sowwn/Workspace/ws/2026/I2fMRI')

import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from pathlib import Path
import yaml

from model.hierarchical_v2 import HierarchicalARM_V2
from data.dataset import create_dataloaders


def voxel_values_to_volume(voxel_values, coords_3d, volume_shape):
    """Convert flat voxel values to 3D volume."""
    volume = np.zeros(volume_shape, dtype=np.float32)
    x = coords_3d[:, 0].astype(int)
    y = coords_3d[:, 1].astype(int)
    z = coords_3d[:, 2].astype(int)
    volume[x, y, z] = voxel_values
    return volume


def create_nifti_image(volume, bbox_min, ref_nii_path='/home/sowwn/Workspace/ws/2026/I2fMRI/dataset/nsd/subj01/schaefer1000_nsd_space.nii.gz'):
    """
    Create NIfTI with corrected affine.

    Args:
        volume: [H, W, D] compact volume
        bbox_min: [3] bounding box offset
        ref_nii_path: Path to reference NIfTI for affine

    Returns:
        nib.Nifti1Image with corrected affine
    """
    # Load reference affine
    ref_nii = nib.load(ref_nii_path)
    affine = ref_nii.affine.copy()

    # Adjust for bbox offset
    affine[0, 3] += bbox_min[0] * affine[0, 0]
    affine[1, 3] += bbox_min[1] * affine[1, 1]
    affine[2, 3] += bbox_min[2] * affine[2, 2]

    return nib.Nifti1Image(volume, affine)


def get_predictions(model, dataloader, num_samples=5, device='cuda'):
    """Get predictions from model."""
    model.eval()
    all_gt = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:
            vis_feat = batch['embedding'].to(device)
            fmri = batch['fmri'].to(device)
            roi_means_gt = batch['roi_means'].to(device)

            _, voxel_pred = model(vis_feat, stage=2, gt_roi_means=roi_means_gt)

            all_gt.append(fmri.cpu())
            all_pred.append(voxel_pred.cpu())

            if len(all_gt) * fmri.shape[0] >= num_samples:
                break

    gt = torch.cat(all_gt, dim=0)[:num_samples]
    pred = torch.cat(all_pred, dim=0)[:num_samples]

    return gt.numpy(), pred.numpy()


def main():
    print("🧠 Brain Visualization with Fixed Affine\n")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = Path('/home/sowwn/Workspace/ws/2026/I2fMRI')
    output_dir = base_path / 'outputs'
    output_dir.mkdir(exist_ok=True)

    # Load config
    config_path = base_path / 'configs/train_hierarchical_v2.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"✓ Using device: {device}")

    # Load model
    print("Loading model...")
    model = HierarchicalARM_V2(
        visual_dim=config['model']['visual_dim'],
        num_clusters=config['model']['num_clusters'],
        fmri_dim=config['model']['fmri_dim'],
        hidden_dim_1=config['model']['hidden_dim_1'],
        embed_dim_2=config['model']['embed_dim_2'],
        depth_2=config['model']['depth_2'],
        num_heads_2=config['model']['num_heads_2'],
        num_global_queries=config['model']['num_global_queries'],
        contrastive_dim=config['model']['contrastive_dim'],
        use_mamba=config['model']['use_mamba']
    ).to(device)

    checkpoint = torch.load(base_path / 'checkpoints/hierarchical_v21_subj01/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()
    print("✓ Model loaded\n")

    # Load data
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        datalist_path=str(base_path / config['data']['datalist_path']),
        fmri_path=str(base_path / config['data']['fmri_path']),
        train_embeddings_path=str(base_path / config['data']['train_embeddings_path']),
        test_embeddings_path=str(base_path / config['data']['test_embeddings_path']),
        subjects=[config['data']['subject']],
        batch_size=8,
        average_trials_train=config['data']['average_trials_train'],
        average_trials_val=config['data']['average_trials_val'],
        augment_noise_train=False,
        sub_roi_cluster_path=str(base_path / config['data']['sub_roi_cluster_path']),
        roi_top_k_percent=config['data']['roi_top_k_percent']
    )
    print(f"✓ Data loaded: {len(val_loader)} val batches\n")

    # Load 3D mapping
    print("Loading 3D mapping...")
    mapping = np.load(base_path / 'data/fmri_3d_compact_mapping_subj01.npz')
    coords_3d = mapping['coords_compact']
    volume_shape = tuple(mapping['compact_shape'])
    bbox_min = mapping['bbox_min']

    print(f"  Volume shape: {volume_shape}")
    print(f"  BBox offset: {bbox_min}")
    print(f"  Num voxels: {len(coords_3d)}\n")

    # Get predictions
    print("Getting predictions...")
    gt_voxels, pred_voxels = get_predictions(model, val_loader, num_samples=1, device=device)
    print(f"✓ Predictions shape: {pred_voxels.shape}\n")

    # Visualize
    sample_idx = 0
    print(f"Visualizing sample {sample_idx}...")

    # Create volumes
    gt_vol = voxel_values_to_volume(gt_voxels[sample_idx], coords_3d, volume_shape)
    pred_vol = voxel_values_to_volume(pred_voxels[sample_idx], coords_3d, volume_shape)
    error_vol = np.abs(gt_vol - pred_vol)

    # Create NIfTI with CORRECTED affine
    gt_nii = create_nifti_image(gt_vol, bbox_min)
    pred_nii = create_nifti_image(pred_vol, bbox_min)
    error_nii = create_nifti_image(error_vol, bbox_min)

    # Compute metrics
    corr = np.corrcoef(gt_voxels[sample_idx], pred_voxels[sample_idx])[0, 1]
    mae = error_vol[error_vol != 0].mean()

    print(f"  Correlation: {corr:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Affine origin: {gt_nii.affine[:3, 3]}\n")

    # 1. Orthogonal slices
    print("1️⃣ Creating orthogonal slices...")
    fig, axes = plt.subplots(2, 1, figsize=(20, 16))

    plotting.plot_stat_map(
        gt_nii,
        title=f'Ground Truth (r={corr:.3f})',
        cut_coords=(0, 0, 0),
        display_mode='ortho',
        cmap='cold_hot',
        symmetric_cbar=True,
        axes=axes[0],
        figure=fig
    )

    plotting.plot_stat_map(
        pred_nii,
        title=f'Predictions (r={corr:.3f})',
        cut_coords=(0, 0, 0),
        display_mode='ortho',
        cmap='cold_hot',
        symmetric_cbar=True,
        axes=axes[1],
        figure=fig
    )

    plt.tight_layout()
    out_path = output_dir / 'brain_ortho_FIXED.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {out_path}")

    # 2. Glass brain
    print("2️⃣ Creating glass brain...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    plotting.plot_glass_brain(
        gt_nii,
        title='Ground Truth Glass Brain',
        colorbar=True,
        cmap='cold_hot',
        plot_abs=False,
        axes=axes[0],
        figure=fig
    )

    plotting.plot_glass_brain(
        pred_nii,
        title=f'Predictions Glass Brain (r={corr:.3f})',
        colorbar=True,
        cmap='cold_hot',
        plot_abs=False,
        axes=axes[1],
        figure=fig
    )

    plt.tight_layout()
    out_path = output_dir / 'brain_glass_FIXED.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {out_path}")

    # 3. Error map
    print("3️⃣ Creating error map...")
    fig = plt.figure(figsize=(18, 6))

    plotting.plot_stat_map(
        error_nii,
        title=f'Prediction Error (MAE={mae:.3f})',
        cut_coords=(0, 0, 0),
        display_mode='ortho',
        cmap='hot',
        colorbar=True,
        vmin=0,
        figure=fig
    )

    out_path = output_dir / 'brain_error_FIXED.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {out_path}")

    print("\n✅ Done! All visualizations saved with corrected affine.")
    print(f"   Check: {output_dir}")


if __name__ == '__main__':
    main()

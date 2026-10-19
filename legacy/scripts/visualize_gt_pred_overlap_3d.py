"""
Visualize GT vs Pred fMRI Overlap in 3D Web Viewer

Colors:
- PURPLE: Overlap (both GT and Pred active)
- GREEN: Miss (GT active only - model missed)
- ORANGE: False Positive (Pred active only)

Usage:
    python scripts/visualize_gt_pred_overlap_3d.py --port 8051 --sample 0
"""

import sys
sys.path.insert(0, '.')

import argparse
import torch
import numpy as np
import json
import nibabel as nib
import cortex

from model.hierarchical_v3 import HierarchicalARM_V3


def load_data():
    """Load all necessary data."""
    print("Loading data...")

    # Load NIfTI
    nsd_roi_path = '/mnt/HDD/dataset/NSD_raw/nsddata/ppdata/subj01/func1pt8mm/roi/'
    streams_nii = nib.load(nsd_roi_path + 'streams.nii.gz')
    streams_data = streams_nii.get_fdata()

    # Load test data
    cls_test = np.load('dataset/nsd/embeddings_mindeye2/dinov2_test_sub01.npy')
    patches_test = np.load('dataset/nsd/embeddings_patches/dinov2_patches_test_sub01.npy')[:, 1:, :]
    gt_fmri = np.load('dataset/nsd/mixed_1000_sub01/test_fmri.npy')

    print(f"  Test samples: {len(cls_test)}")
    print(f"  Volume shape: {streams_data.shape}")

    return {
        'streams_data': streams_data,
        'streams_nii': streams_nii,
        'cls_test': cls_test,
        'patches_test': patches_test,
        'gt_fmri': gt_fmri
    }


def load_model(device):
    """Load trained model."""
    print("Loading model...")

    model = HierarchicalARM_V3(
        cls_dim=768, patch_dim=768, num_patches=256,
        fmri_dim=15724, voxels_per_cluster=30,
    ).to(device)

    checkpoint_path = 'checkpoints/hierarchical_v3_subj01_30/best_stage1.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"  Model: {model.num_rois} ROIs, {model.voxels_per_cluster} voxels/ROI")
    return model


def get_flat_to_volume_mapping(streams_data):
    """Create mapping from flat fMRI indices to 3D volume coordinates."""
    nonzero_coords = np.array(np.where(streams_data != 0)).T
    if len(nonzero_coords) >= 15724:
        return nonzero_coords[:15724]
    else:
        flat_indices = np.zeros((15724, 3), dtype=int)
        flat_indices[:len(nonzero_coords)] = nonzero_coords
        return flat_indices


def flat_to_volume(flat_fmri, flat_indices, volume_shape):
    """Convert flat fMRI array to 3D volume."""
    volume = np.zeros(volume_shape, dtype=np.float32)
    for i, (x, y, z) in enumerate(flat_indices):
        if i < len(flat_fmri):
            volume[int(x), int(y), int(z)] = flat_fmri[i]
    return volume


def roi_to_voxel(roi_pred, n_rois, vpc, total_voxels=15724):
    """Expand ROI predictions to voxel-level."""
    voxel_pred = np.zeros(total_voxels)
    for i in range(n_rois):
        start = i * vpc
        end = min((i + 1) * vpc, total_voxels)
        voxel_pred[start:end] = roi_pred[i]
    return voxel_pred


def get_prediction(model, cls_token, patch_tokens, device):
    """Get model prediction for a single sample."""
    with torch.no_grad():
        cls_b = torch.tensor(cls_token).unsqueeze(0).to(device)
        pat_b = torch.tensor(patch_tokens).unsqueeze(0).to(device)
        pred_roi, _ = model(cls_b, pat_b)
        pred_roi = pred_roi.cpu().numpy()[0]

    # Expand to voxel level
    vpc = model.voxels_per_cluster
    n_rois = model.num_rois
    pred_voxel = roi_to_voxel(pred_roi, n_rois, vpc)
    return pred_voxel


def create_overlap_volumes(gt_vol, pred_vol, threshold_percentile=75):
    """
    Create RGB volumes for overlap visualization.

    Returns:
        red, green, blue, alpha channels
        stats: overlap, miss, fp counts
    """
    # Threshold for "active" voxels
    gt_nonzero = gt_vol[gt_vol != 0]
    pred_nonzero = pred_vol[pred_vol != 0]

    gt_thresh = np.percentile(np.abs(gt_nonzero), threshold_percentile) if len(gt_nonzero) > 0 else 0
    pred_thresh = np.percentile(np.abs(pred_nonzero), threshold_percentile) if len(pred_nonzero) > 0 else 0

    gt_active = np.abs(gt_vol) > gt_thresh
    pred_active = np.abs(pred_vol) > pred_thresh

    # Classify voxels
    overlap = gt_active & pred_active       # Both active
    gt_only = gt_active & ~pred_active      # GT only (miss)
    pred_only = ~gt_active & pred_active    # Pred only (false positive)

    # Create RGB channels
    red = np.zeros_like(gt_vol, dtype=np.float32)
    green = np.zeros_like(gt_vol, dtype=np.float32)
    blue = np.zeros_like(gt_vol, dtype=np.float32)

    # Purple for overlap (Red + Blue)
    red[overlap] = 0.8
    blue[overlap] = 0.8

    # Green for miss
    green[gt_only] = 0.9

    # Orange for false positive (Red + some Green)
    red[pred_only] = 1.0
    green[pred_only] = 0.5

    # Alpha channel
    alpha = np.maximum(red, np.maximum(green, blue))

    stats = {
        'overlap': int(overlap.sum()),
        'miss': int(gt_only.sum()),
        'fp': int(pred_only.sum())
    }

    return red, green, blue, alpha, stats


def launch_3d_viewer(data, sample_idx, model, device, port=8051):
    """Launch Pycortex 3D web viewer for GT vs Pred overlap."""
    print(f"\nProcessing sample #{sample_idx}...")

    streams_data = data['streams_data']
    volume_shape = streams_data.shape

    # Get flat-to-volume mapping
    flat_indices = get_flat_to_volume_mapping(streams_data)

    # Get GT and Pred
    gt = data['gt_fmri'][sample_idx]
    pred = get_prediction(model, data['cls_test'][sample_idx],
                         data['patches_test'][sample_idx], device)

    # Convert to volumes
    gt_vol = flat_to_volume(gt, flat_indices, volume_shape)
    pred_vol = flat_to_volume(pred, flat_indices, volume_shape)

    print(f"  GT volume: min={gt_vol.min():.3f}, max={gt_vol.max():.3f}")
    print(f"  Pred volume: min={pred_vol.min():.3f}, max={pred_vol.max():.3f}")

    # Create overlap visualization
    red, green, blue, alpha, stats = create_overlap_volumes(gt_vol, pred_vol)

    total = stats['overlap'] + stats['miss'] + stats['fp']
    if total > 0:
        overlap_pct = stats['overlap'] / total * 100
        miss_pct = stats['miss'] / total * 100
        fp_pct = stats['fp'] / total * 100
    else:
        overlap_pct = miss_pct = fp_pct = 0

    print(f"\n  Overlap Statistics:")
    print(f"    Purple (Overlap): {stats['overlap']:,} voxels ({overlap_pct:.1f}%)")
    print(f"    Green (Miss):     {stats['miss']:,} voxels ({miss_pct:.1f}%)")
    print(f"    Orange (FP):      {stats['fp']:,} voxels ({fp_pct:.1f}%)")

    # Pycortex setup - use identity transform
    SUBJECT = 'subj01'
    TRANSFORM = 'identity'
    print(f"\n  Pycortex: {SUBJECT}, {TRANSFORM}")

    # Normalize for visualization
    vmax = max(np.abs(gt_vol).max(), np.abs(pred_vol).max())
    if vmax == 0:
        vmax = 1

    # Create pycortex volumes
    gt_cortex = cortex.Volume(
        gt_vol.T, subject=SUBJECT, xfmname=TRANSFORM,
        cmap='RdBu_r', vmin=-vmax, vmax=vmax
    )

    pred_cortex = cortex.Volume(
        pred_vol.T, subject=SUBJECT, xfmname=TRANSFORM,
        cmap='RdBu_r', vmin=-vmax, vmax=vmax
    )

    diff_vol = pred_vol - gt_vol
    diff_cortex = cortex.Volume(
        diff_vol.T, subject=SUBJECT, xfmname=TRANSFORM,
        cmap='PiYG', vmin=-vmax, vmax=vmax
    )

    # Create Volume for each RGB channel
    red_vol = cortex.Volume(red.T, subject=SUBJECT, xfmname=TRANSFORM)
    green_vol = cortex.Volume(green.T, subject=SUBJECT, xfmname=TRANSFORM)
    blue_vol = cortex.Volume(blue.T, subject=SUBJECT, xfmname=TRANSFORM)

    overlap_rgb = cortex.VolumeRGB(
        red_vol,    # channel1 = Red
        green_vol,  # channel2 = Green
        blue_vol,   # channel3 = Blue
        subject=SUBJECT,
        xfmname=TRANSFORM
    )

    # Create dataset
    dataset = cortex.Dataset(
        gt_fmri=gt_cortex,
        pred_fmri=pred_cortex,
        difference=diff_cortex,
        overlap=overlap_rgb
    )

    print("\n" + "="*60)
    print(f"GT vs PRED OVERLAP - Sample #{sample_idx}")
    print("="*60)
    print("\nViews available:")
    print("  - gt_fmri: Ground truth fMRI activation")
    print("  - pred_fmri: Model predicted fMRI activation")
    print("  - difference: Pred - GT (green=over, pink=under)")
    print("  - overlap: RGB overlap visualization")
    print("\nOverlap Colors:")
    print("  - PURPLE: Both GT & Pred active (correct)")
    print("  - GREEN:  GT only (model missed)")
    print("  - ORANGE: Pred only (false positive)")
    print("\nControls:")
    print("  - Mouse drag: Rotate brain")
    print("  - Scroll: Zoom")
    print("  - Double-click: Reset view")
    print(f"\nOpening browser at http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")

    # Launch viewer
    cortex.webgl.show(dataset, open_browser=True, port=port)


def main():
    parser = argparse.ArgumentParser(description='Visualize GT vs Pred overlap in 3D')
    parser.add_argument('--port', type=int, default=8051, help='Port for web viewer')
    parser.add_argument('--sample', type=int, default=0, help='Sample index to visualize')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data and model
    data = load_data()
    model = load_model(device)

    # Validate sample index
    max_samples = len(data['gt_fmri'])
    if args.sample < 0 or args.sample >= max_samples:
        print(f"Error: Sample index must be between 0 and {max_samples-1}")
        return

    # Launch viewer
    launch_3d_viewer(data, args.sample, model, device, args.port)


if __name__ == '__main__':
    main()

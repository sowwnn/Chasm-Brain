"""
Demo Script: Load and Visualize Inference Results
==================================================

Example script showing how to load and work with inference results.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from PIL import Image

def demo_basic_loading():
    """Demo: Load basic arrays and metadata."""
    print("=" * 80)
    print("Demo 1: Basic Loading")
    print("=" * 80)

    results_dir = Path('/home/sowwn/Workspace/ws/2026/I2fMRI/inference_results_hierarchical')
    sample_dir = results_dir / 'sample_000'

    # Load numpy arrays
    pred = np.load(sample_dir / 'predict.npy')
    gt = np.load(sample_dir / 'groundtruth.npy')

    print(f"\nPrediction array:")
    print(f"  Shape: {pred.shape}")
    print(f"  Range: [{pred.min():.3f}, {pred.max():.3f}]")
    print(f"  Mean: {pred.mean():.3f}")
    print(f"  Std: {pred.std():.3f}")

    print(f"\nGround truth array:")
    print(f"  Shape: {gt.shape}")
    print(f"  Range: [{gt.min():.3f}, {gt.max():.3f}]")
    print(f"  Mean: {gt.mean():.3f}")
    print(f"  Std: {gt.std():.3f}")

    # Load metadata
    with open(sample_dir / 'metadata.json') as f:
        meta = json.load(f)

    print(f"\nMetadata:")
    print(f"  Image ID: {meta['image_id']}")
    print(f"  Correlation: {meta['correlation']:.4f}")
    print(f"  MAE: {meta['mae']:.4f}")
    print(f"  MSE: {meta['mse']:.4f}")


def demo_summary_stats():
    """Demo: Load and analyze summary statistics."""
    print("\n" + "=" * 80)
    print("Demo 2: Summary Statistics")
    print("=" * 80)

    results_dir = Path('/home/sowwn/Workspace/ws/2026/I2fMRI/inference_results_hierarchical')

    # Load summary
    with open(results_dir / 'summary.json') as f:
        summary = json.load(f)

    print(f"\nOverall Performance:")
    print(f"  Number of samples: {summary['num_samples']}")
    print(f"  Mean correlation: {summary['mean_correlation']:.4f} ± {summary['std_correlation']:.4f}")
    print(f"  Mean MAE: {summary['mean_mae']:.4f} ± {summary['std_mae']:.4f}")
    print(f"  Mean MSE: {summary['mean_mse']:.4f}")

    # Find best and worst samples
    samples = summary['samples']
    best = max(samples, key=lambda x: x['correlation'])
    worst = min(samples, key=lambda x: x['correlation'])

    print(f"\nBest sample:")
    print(f"  Sample index: {best['sample_idx']}")
    print(f"  Correlation: {best['correlation']:.4f}")
    print(f"  MAE: {best['mae']:.4f}")

    print(f"\nWorst sample:")
    print(f"  Sample index: {worst['sample_idx']}")
    print(f"  Correlation: {worst['correlation']:.4f}")
    print(f"  MAE: {worst['mae']:.4f}")


def demo_compute_metrics():
    """Demo: Compute additional metrics."""
    print("\n" + "=" * 80)
    print("Demo 3: Compute Additional Metrics")
    print("=" * 80)

    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import r2_score, mean_absolute_percentage_error

    results_dir = Path('/home/sowwn/Workspace/ws/2026/I2fMRI/inference_results_hierarchical')
    sample_dir = results_dir / 'sample_000'

    # Load data
    pred = np.load(sample_dir / 'predict.npy')
    gt = np.load(sample_dir / 'groundtruth.npy')

    # Pearson correlation
    corr, p_value = pearsonr(pred, gt)
    print(f"\nPearson correlation: {corr:.4f} (p={p_value:.4e})")

    # Spearman correlation
    spearman, p_value_sp = spearmanr(pred, gt)
    print(f"Spearman correlation: {spearman:.4f} (p={p_value_sp:.4e})")

    # R² score
    r2 = r2_score(gt, pred)
    print(f"R² score: {r2:.4f}")

    # MAE
    mae = np.mean(np.abs(pred - gt))
    print(f"MAE: {mae:.4f}")

    # RMSE
    rmse = np.sqrt(np.mean((pred - gt)**2))
    print(f"RMSE: {rmse:.4f}")

    # Normalized RMSE
    nrmse = rmse / (gt.max() - gt.min())
    print(f"Normalized RMSE: {nrmse:.4f}")

    # MAPE (careful with zeros)
    # Add small epsilon to avoid division by zero
    mape = np.mean(np.abs((gt - pred) / (np.abs(gt) + 1e-8))) * 100
    print(f"MAPE: {mape:.2f}%")


def demo_visualize_comparison():
    """Demo: Create comparison plots."""
    print("\n" + "=" * 80)
    print("Demo 4: Visualize Comparison")
    print("=" * 80)

    results_dir = Path('/home/sowwn/Workspace/ws/2026/I2fMRI/inference_results_hierarchical')
    sample_dir = results_dir / 'sample_000'

    # Load data
    pred = np.load(sample_dir / 'predict.npy')
    gt = np.load(sample_dir / 'groundtruth.npy')

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))

    # 1. Line comparison
    voxel_idx = np.arange(len(pred))
    axes[0].plot(voxel_idx, gt, label='Ground Truth', alpha=0.6, color='blue', linewidth=1)
    axes[0].plot(voxel_idx, pred, label='Prediction', alpha=0.6, color='red', linewidth=1)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Voxel Index')
    axes[0].set_ylabel('Activation')
    axes[0].set_title('Voxel-wise Comparison')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Scatter plot
    axes[1].scatter(gt, pred, alpha=0.3, s=1)
    axes[1].plot([gt.min(), gt.max()], [gt.min(), gt.max()], 'r--', linewidth=2, label='Perfect prediction')
    axes[1].set_xlabel('Ground Truth')
    axes[1].set_ylabel('Prediction')
    axes[1].set_title('Scatter Plot: Prediction vs Ground Truth')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 3. Error distribution
    error = pred - gt
    axes[2].hist(error, bins=100, alpha=0.7, color='purple', edgecolor='black')
    axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[2].axvline(x=error.mean(), color='green', linestyle='--', linewidth=2,
                   label=f'Mean error: {error.mean():.3f}')
    axes[2].set_xlabel('Prediction Error')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Error Distribution')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    output_path = results_dir / 'demo_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to: {output_path}")
    plt.close()


def demo_load_nifti():
    """Demo: Load and inspect NIfTI files."""
    print("\n" + "=" * 80)
    print("Demo 5: Load NIfTI Files")
    print("=" * 80)

    results_dir = Path('/home/sowwn/Workspace/ws/2026/I2fMRI/inference_results_hierarchical')
    sample_dir = results_dir / 'sample_000'

    # Load NIfTI files
    pred_nii = nib.load(sample_dir / 'predict_brain.nii.gz')
    gt_nii = nib.load(sample_dir / 'groundtruth_brain.nii.gz')

    print(f"\nPrediction NIfTI:")
    print(f"  Shape: {pred_nii.shape}")
    print(f"  Data type: {pred_nii.get_data_dtype()}")
    print(f"  Affine matrix:\n{pred_nii.affine}")

    print(f"\nGround truth NIfTI:")
    print(f"  Shape: {gt_nii.shape}")
    print(f"  Data type: {gt_nii.get_data_dtype()}")

    # Get data
    pred_vol = pred_nii.get_fdata()
    gt_vol = gt_nii.get_fdata()

    print(f"\nVolume statistics:")
    print(f"  Prediction: {pred_vol.min():.3f} to {pred_vol.max():.3f}")
    print(f"  Ground truth: {gt_vol.min():.3f} to {gt_vol.max():.3f}")
    print(f"  Non-zero voxels (pred): {np.count_nonzero(pred_vol)}")
    print(f"  Non-zero voxels (gt): {np.count_nonzero(gt_vol)}")


def demo_batch_analysis():
    """Demo: Analyze all samples."""
    print("\n" + "=" * 80)
    print("Demo 6: Batch Analysis")
    print("=" * 80)

    results_dir = Path('/home/sowwn/Workspace/ws/2026/I2fMRI/inference_results_hierarchical')

    # Load summary
    with open(results_dir / 'summary.json') as f:
        summary = json.load(f)

    samples = summary['samples']

    # Create analysis table
    print(f"\n{'Sample':<8} {'Corr':<8} {'MAE':<8} {'MSE':<8} {'Pred Std':<10} {'GT Std':<10}")
    print("-" * 70)

    for s in samples:
        print(f"{s['sample_idx']:<8} "
              f"{s['correlation']:<8.4f} "
              f"{s['mae']:<8.4f} "
              f"{s['mse']:<8.4f} "
              f"{s['pred_std']:<10.4f} "
              f"{s['gt_std']:<10.4f}")

    # Correlation analysis
    correlations = [s['correlation'] for s in samples]
    print(f"\nCorrelation statistics:")
    print(f"  Mean: {np.mean(correlations):.4f}")
    print(f"  Median: {np.median(correlations):.4f}")
    print(f"  Std: {np.std(correlations):.4f}")
    print(f"  Min: {np.min(correlations):.4f}")
    print(f"  Max: {np.max(correlations):.4f}")
    print(f"  Range: {np.max(correlations) - np.min(correlations):.4f}")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("I2fMRI Inference Results - Demo Script")
    print("=" * 80)

    # Check if results exist
    results_dir = Path('/home/sowwn/Workspace/ws/2026/I2fMRI/inference_results_hierarchical')
    if not results_dir.exists():
        print(f"\nERROR: Results directory not found: {results_dir}")
        print("Please run inference script first:")
        print("  python scripts/inference_hierarchical_samples.py")
        return

    # Run demos
    demo_basic_loading()
    demo_summary_stats()
    demo_compute_metrics()
    demo_visualize_comparison()
    demo_load_nifti()
    demo_batch_analysis()

    print("\n" + "=" * 80)
    print("✅ All demos completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()

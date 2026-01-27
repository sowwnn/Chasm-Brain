"""
Quick test script for visualization functions without running training.
This creates dummy data to test the visualization pipeline.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import visualization functions from train script
from train_graph_diffusion import visualize_point_cloud_predictions, visualize_1d_signals


def create_dummy_data(num_samples=2, num_points=1000):
    """Create dummy data that mimics real fMRI data."""
    # Create random 3D coordinates (mimicking brain voxel coordinates)
    coords = torch.randn(num_samples, num_points, 3) * 50  # Scale to brain-like coordinates

    # Create random fMRI signals (mimicking preprocessed fMRI data)
    fmri_gt = torch.randn(num_samples, num_points)
    fmri_pred = fmri_gt + torch.randn(num_samples, num_points) * 0.3  # Add some noise

    # Create dummy ROI labels
    roi_labels = torch.randint(0, 10, (num_samples, num_points))

    return coords, fmri_gt, fmri_pred, roi_labels


def test_visualizations():
    """Test all visualization functions."""
    print("Creating dummy data...")
    coords, fmri_gt, fmri_pred, roi_labels = create_dummy_data(num_samples=2, num_points=1000)

    # Create output directory
    output_dir = Path("test_viz_output")
    output_dir.mkdir(exist_ok=True)

    print("\n1. Testing 1D signal visualization...")
    try:
        visualize_1d_signals(
            fmri_gt,
            fmri_pred,
            output_dir / 'test_1d_signals.png',
            num_samples=2
        )
        print("   ✓ 1D signal visualization successful!")
    except Exception as e:
        print(f"   ✗ 1D signal visualization failed: {e}")

    print("\n2. Testing 3D point cloud visualization...")
    try:
        visualize_point_cloud_predictions(
            coords,
            fmri_gt,
            fmri_pred,
            output_dir / 'test_pointcloud_3d.html',
            roi_labels=roi_labels,
            num_samples=2
        )
        print("   ✓ 3D point cloud visualization successful!")
    except Exception as e:
        print(f"   ✗ 3D point cloud visualization failed: {e}")

    print(f"\n✓ All tests completed! Check output in: {output_dir.absolute()}")


if __name__ == "__main__":
    test_visualizations()

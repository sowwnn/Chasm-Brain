"""
Test script for Flow3DHybridARM model and utilities.

Tests:
1. 3D ↔ 1D conversion (lossless)
2. Model forward pass
3. Flow Matching sampling
4. Metrics computation on 3D and 1D
5. Mask conditioning
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ARMNet.flow_3d_hybrid_model import (
    Flow3DHybridARM,
    fmri_1d_to_3d_compact,
    fmri_3d_compact_to_1d
)
from ARMNet.flow_3d_loss import FlowMatching3DLoss, Metrics3D1D


def test_3d_1d_conversion():
    """Test lossless 3D ↔ 1D conversion"""
    print("\n" + "="*60)
    print("TEST 1: 3D ↔ 1D Conversion (Lossless)")
    print("="*60)

    compact_shape = (61, 46, 42)
    n_voxels = 15724

    # Create random mask
    mask = np.zeros(compact_shape, dtype=bool)
    coords = np.random.randint(0, [61, 46, 42], size=(n_voxels, 3))
    for coord in coords:
        mask[tuple(coord)] = True

    # Update n_voxels to actual count
    n_voxels_actual = mask.sum()
    coords_actual = np.array(np.where(mask)).T

    # Create 1D data
    fmri_1d = np.random.randn(n_voxels_actual).astype(np.float32)

    # Convert to 3D
    fmri_3d = fmri_1d_to_3d_compact(fmri_1d, coords_actual, compact_shape)

    # Convert back to 1D
    fmri_1d_recon = fmri_3d_compact_to_1d(fmri_3d, mask)

    # Verify
    max_diff = np.max(np.abs(fmri_1d - fmri_1d_recon))
    print(f"Original 1D shape: {fmri_1d.shape}")
    print(f"3D shape: {fmri_3d.shape}")
    print(f"Reconstructed 1D shape: {fmri_1d_recon.shape}")
    print(f"Max difference: {max_diff:.2e}")

    assert max_diff < 1e-6, f"Conversion is NOT lossless! Max diff: {max_diff}"
    print("✓ Conversion is 100% lossless!")

    return coords_actual, mask


def test_model_forward():
    """Test model forward pass"""
    print("\n" + "="*60)
    print("TEST 2: Model Forward Pass")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    compact_shape = (61, 46, 42)
    B = 2
    vis_dim = 768

    # Create model
    model = Flow3DHybridARM(
        input_dim=vis_dim,
        compact_shape=compact_shape,
        hidden_dim=64,
        time_dim=128,
        use_graph_layers=True,
        use_mask_conditioning=True,
        dropout=0.1
    ).to(device)

    # Create dummy data
    fmri_3d = torch.randn(B, 1, *compact_shape).to(device)
    vis_feat = torch.randn(B, vis_dim).to(device)

    # Create mask
    mask = torch.zeros(compact_shape, dtype=torch.bool)
    n_voxels = 15724
    coords = torch.randint(0, min(compact_shape), (n_voxels, 3))
    for coord in coords:
        mask[coord[0], coord[1], coord[2]] = True
    mask = mask.to(device)

    # Get actual coords
    coords_compact = torch.stack(torch.where(mask), dim=1).float().to(device)

    # Forward pass
    print("\nForward pass...")
    outputs = model(fmri_3d, vis_feat, mask, coords_compact)

    print(f"Input 3D fMRI shape: {fmri_3d.shape}")
    print(f"Visual features shape: {vis_feat.shape}")
    print(f"Predicted velocity shape: {outputs['pred_velocity'].shape}")
    print(f"Target velocity shape: {outputs['target_velocity'].shape}")
    print(f"Predicted fMRI shape: {outputs['x_pred'].shape}")

    assert outputs['pred_velocity'].shape == fmri_3d.shape
    assert outputs['target_velocity'].shape == fmri_3d.shape
    assert outputs['x_pred'].shape == fmri_3d.shape

    print("✓ Forward pass successful!")

    return model, mask, coords_compact, device


def test_flow_sampling(model, mask, coords_compact, device):
    """Test Flow Matching sampling"""
    print("\n" + "="*60)
    print("TEST 3: Flow Matching Sampling")
    print("="*60)

    B = 2
    vis_dim = 768

    vis_feat = torch.randn(B, vis_dim).to(device)

    # Generate
    print("Generating with 10 steps...")
    pred = model.generate(vis_feat, mask, coords_compact, steps=10, cfg_scale=1.0)

    print(f"Generated fMRI shape: {pred.shape}")
    print(f"Mean: {pred.mean():.4f}, Std: {pred.std():.4f}")
    print(f"Min: {pred.min():.4f}, Max: {pred.max():.4f}")

    assert pred.shape == (B, 1, 61, 46, 42)
    print("✓ Sampling successful!")

    return pred


def test_metrics():
    """Test metrics computation on 3D and 1D"""
    print("\n" + "="*60)
    print("TEST 4: Metrics Computation (3D and 1D)")
    print("="*60)

    compact_shape = (61, 46, 42)
    B = 4
    n_voxels = 15724

    # Create mask
    mask = torch.zeros(compact_shape, dtype=torch.bool)
    coords = torch.randint(0, min(compact_shape), (n_voxels, 3))
    for coord in coords:
        mask[coord[0], coord[1], coord[2]] = True

    coords_actual = np.array(np.where(mask.numpy())).T

    # Create 3D data
    pred_3d = torch.randn(B, 1, *compact_shape)
    target_3d = torch.randn(B, 1, *compact_shape)

    # Convert to 1D
    pred_1d = fmri_3d_compact_to_1d(pred_3d.squeeze(1).numpy(), mask.numpy())
    target_1d = fmri_3d_compact_to_1d(target_3d.squeeze(1).numpy(), mask.numpy())

    # Compute metrics
    metrics = Metrics3D1D.compute_all_metrics(
        pred_3d, target_3d, pred_1d, target_1d, mask
    )

    print("\nMetrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")

    # Verify lossless
    mse_diff = abs(metrics['mse_3d'] - metrics['mse_1d'])
    pearson_diff = abs(metrics['pearson_3d_mean'] - metrics['pearson_1d_mean'])

    print(f"\nConversion verification:")
    print(f"  MSE difference: {mse_diff:.2e}")
    print(f"  Pearson difference: {pearson_diff:.2e}")

    assert mse_diff < 1e-5, f"MSE mismatch: {mse_diff}"
    assert pearson_diff < 1e-3, f"Pearson mismatch: {pearson_diff}"

    print("✓ Metrics computation successful and lossless!")


def test_loss_function():
    """Test FlowMatching3DLoss"""
    print("\n" + "="*60)
    print("TEST 5: Loss Function")
    print("="*60)

    compact_shape = (61, 46, 42)
    B = 2

    # Create mask
    mask = torch.zeros(compact_shape, dtype=torch.bool)
    n_voxels = 15724
    coords = torch.randint(0, min(compact_shape), (n_voxels, 3))
    for coord in coords:
        mask[coord[0], coord[1], coord[2]] = True

    # Create data
    pred_v = torch.randn(B, 1, *compact_shape)
    target_v = torch.randn(B, 1, *compact_shape)
    pred_fmri = torch.randn(B, 1, *compact_shape)
    target_fmri = torch.randn(B, 1, *compact_shape)
    mean_fmri = torch.zeros(B, 1, *compact_shape)

    # Create loss
    criterion = FlowMatching3DLoss(
        alpha=10.0,
        tau=0.5,
        pearson_weight=0.1,
        mask_weight=2.0
    )

    # Compute loss
    loss = criterion(pred_v, target_v, pred_fmri, target_fmri, mean_fmri, mask)

    print(f"Loss value: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be inf"

    print("✓ Loss computation successful!")


def test_mask_conditioning():
    """Test mask conditioning effect"""
    print("\n" + "="*60)
    print("TEST 6: Mask Conditioning")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    compact_shape = (61, 46, 42)
    B = 2
    vis_dim = 768

    # Create mask
    mask = torch.zeros(compact_shape, dtype=torch.bool)
    n_voxels = 15724
    coords = torch.randint(0, min(compact_shape), (n_voxels, 3))
    for coord in coords:
        mask[coord[0], coord[1], coord[2]] = True
    mask = mask.to(device)
    coords_compact = torch.stack(torch.where(mask), dim=1).float().to(device)

    # Create two models: with and without mask conditioning
    model_with_mask = Flow3DHybridARM(
        input_dim=vis_dim,
        compact_shape=compact_shape,
        hidden_dim=64,
        time_dim=128,
        use_mask_conditioning=True
    ).to(device)

    model_without_mask = Flow3DHybridARM(
        input_dim=vis_dim,
        compact_shape=compact_shape,
        hidden_dim=64,
        time_dim=128,
        use_mask_conditioning=False
    ).to(device)

    # Test forward pass
    fmri_3d = torch.randn(B, 1, *compact_shape).to(device)
    vis_feat = torch.randn(B, vis_dim).to(device)

    print("Testing model WITH mask conditioning...")
    out_with = model_with_mask(fmri_3d, vis_feat, mask, coords_compact)
    print(f"  Output shape: {out_with['pred_velocity'].shape}")

    print("Testing model WITHOUT mask conditioning...")
    out_without = model_without_mask(fmri_3d, vis_feat, None, coords_compact)
    print(f"  Output shape: {out_without['pred_velocity'].shape}")

    print("✓ Mask conditioning test successful!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*20 + "FLOW 3D HYBRID ARM TESTS")
    print("="*70)

    try:
        # Test 1: Conversion
        coords, mask = test_3d_1d_conversion()

        # Test 2: Forward pass
        model, mask_torch, coords_compact, device = test_model_forward()

        # Test 3: Sampling
        pred = test_flow_sampling(model, mask_torch, coords_compact, device)

        # Test 4: Metrics
        test_metrics()

        # Test 5: Loss
        test_loss_function()

        # Test 6: Mask conditioning
        test_mask_conditioning()

        print("\n" + "="*70)
        print(" "*25 + "ALL TESTS PASSED!")
        print("="*70)
        print("\nSummary:")
        print("  ✓ 3D ↔ 1D conversion is lossless")
        print("  ✓ Model forward pass works correctly")
        print("  ✓ Flow Matching sampling works")
        print("  ✓ Metrics can be computed on both 3D and 1D")
        print("  ✓ Loss function works correctly")
        print("  ✓ Mask conditioning works")
        print("\nReady for training!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

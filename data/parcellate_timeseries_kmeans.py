"""
Parcellate fMRI data using K-means clustering on VOXEL TIME-SERIES.

This approach follows the methodology from analy.ipynb which preserves
information better than spatial-only clustering:

1. Use RAW fMRI data (not z-scored) for clustering
2. Cluster voxels based on their TIME-SERIES patterns (not just spatial location)
3. Aggregate voxels into parcels via mean pooling
4. Can reconstruct back to original voxel space
5. Z-score AFTER reconstruction for comparison

Key difference from parcellate_kmeans.py:
- That file clusters on spatial coordinates only (X,Y,Z)
- This file clusters on voxel time-series, capturing functional similarity

Usage:
    python parcellate_timeseries_kmeans.py --input dataset/nsd/subj01/betas_all_subj01_fp32_renorm.hdf5 --n_parcels 1000
"""

import os
import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_fmri_subset(fmri_path: str, n_samples: int = None):
    """
    Load fMRI data (optionally just a subset for clustering).

    Args:
        fmri_path: Path to HDF5 file with 'betas' dataset
        n_samples: Number of samples to use (None = use all)

    Returns:
        fmri_data: [n_samples, n_voxels] array
    """
    print(f"\n{'='*60}")
    print(f"Loading fMRI data from {Path(fmri_path).name}...")
    print(f"{'='*60}")

    with h5py.File(fmri_path, 'r') as f:
        if n_samples is None:
            fmri_data = f['betas'][:]
        else:
            total_samples = f['betas'].shape[0]
            n_samples = min(n_samples, total_samples)
            fmri_data = f['betas'][:n_samples]

    print(f"  Shape: {fmri_data.shape}")
    print(f"  dtype: {fmri_data.dtype}")
    print(f"  Range: [{fmri_data.min():.4f}, {fmri_data.max():.4f}]")
    print(f"  Mean: {fmri_data.mean():.4f}, Std: {fmri_data.std():.4f}")

    return fmri_data


def cluster_voxels_by_timeseries(
    fmri_data: np.ndarray,
    n_parcels: int = 1000,
    random_state: int = 42,
    use_minibatch: bool = False,
):
    """
    Cluster voxels based on their time-series patterns using K-means.

    Args:
        fmri_data: [n_samples, n_voxels] fMRI data
        n_parcels: Number of parcels to create
        random_state: Random seed
        use_minibatch: Use MiniBatchKMeans for large datasets

    Returns:
        labels: [n_voxels] parcel assignment for each voxel (0 to n_parcels-1)
        kmeans: Fitted K-means model
    """
    n_samples, n_voxels = fmri_data.shape

    print(f"\n{'='*60}")
    print(f"K-means clustering on voxel time-series...")
    print(f"{'='*60}")
    print(f"  Input shape: {fmri_data.shape}")
    print(f"  Clustering: {n_voxels} voxels → {n_parcels} parcels")
    print(f"  Each voxel has {n_samples} time points")

    # Transpose: [n_voxels, n_samples]
    # Each row is a voxel's time-series
    voxel_timeseries = fmri_data.T
    print(f"  Voxel timeseries shape: {voxel_timeseries.shape}")

    # Choose KMeans variant
    if use_minibatch:
        print(f"  Using MiniBatchKMeans (faster for large datasets)...")
        kmeans = MiniBatchKMeans(
            n_clusters=n_parcels,
            random_state=random_state,
            batch_size=min(2000, n_voxels),
            max_iter=300,
            verbose=1,
            n_init=3,
        )
    else:
        print(f"  Using standard KMeans...")
        kmeans = KMeans(
            n_clusters=n_parcels,
            random_state=random_state,
            max_iter=300,
            verbose=1,
            n_init=10,
        )

    # Fit and predict
    labels = kmeans.fit_predict(voxel_timeseries)

    print(f"\n  Clustering complete!")
    print(f"  Iterations: {kmeans.n_iter_}")
    print(f"  Inertia: {kmeans.inertia_:.2f}")

    # Statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\n  Parcel statistics:")
    print(f"    Number of parcels: {len(unique_labels)}")
    print(f"    Voxels per parcel:")
    print(f"      Min: {counts.min()}")
    print(f"      Max: {counts.max()}")
    print(f"      Mean: {counts.mean():.1f}")
    print(f"      Median: {np.median(counts):.1f}")
    print(f"      Std: {counts.std():.1f}")

    percentiles = [10, 25, 50, 75, 90]
    print(f"    Percentiles:")
    for p in percentiles:
        print(f"      {p}th: {np.percentile(counts, p):.1f}")

    return labels, kmeans


def parcellate_data(
    fmri_data: np.ndarray,
    labels: np.ndarray,
    n_parcels: int,
    aggregation: str = 'mean',
):
    """
    Aggregate voxels into parcels based on labels.

    Args:
        fmri_data: [n_samples, n_voxels] fMRI data
        labels: [n_voxels] parcel assignments
        n_parcels: Number of parcels
        aggregation: 'mean' or 'median'

    Returns:
        parcellated: [n_samples, n_parcels] aggregated data
    """
    n_samples, n_voxels = fmri_data.shape

    print(f"\n{'='*60}")
    print(f"Parcellating data...")
    print(f"{'='*60}")
    print(f"  Input: {fmri_data.shape}")
    print(f"  Output: ({n_samples}, {n_parcels})")
    print(f"  Aggregation: {aggregation}")

    parcellated = np.zeros((n_samples, n_parcels), dtype=np.float32)

    for parcel_id in tqdm(range(n_parcels), desc="Aggregating parcels"):
        voxel_indices = np.where(labels == parcel_id)[0]

        if len(voxel_indices) == 0:
            print(f"    Warning: Parcel {parcel_id} has no voxels!")
            continue

        voxels_in_parcel = fmri_data[:, voxel_indices]

        if aggregation == 'mean':
            parcellated[:, parcel_id] = voxels_in_parcel.mean(axis=1)
        elif aggregation == 'median':
            parcellated[:, parcel_id] = np.median(voxels_in_parcel, axis=1)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    print(f"\n  Parcellated data statistics:")
    print(f"    Range: [{parcellated.min():.4f}, {parcellated.max():.4f}]")
    print(f"    Mean: {parcellated.mean():.4f}")
    print(f"    Std: {parcellated.std():.4f}")

    print(f"\n  Compression ratio: {n_voxels / n_parcels:.2f}x")

    return parcellated


def reconstruct_from_parcels(
    parcellated: np.ndarray,
    labels: np.ndarray,
    n_voxels: int,
):
    """
    Reconstruct full voxel space from parcellated data.

    Each voxel gets the value of its parcel.

    Args:
        parcellated: [n_samples, n_parcels] parcellated data
        labels: [n_voxels] parcel assignments
        n_voxels: Original number of voxels

    Returns:
        reconstructed: [n_samples, n_voxels] reconstructed data
    """
    n_samples, n_parcels = parcellated.shape

    print(f"\n{'='*60}")
    print(f"Reconstructing voxel space...")
    print(f"{'='*60}")
    print(f"  Input: {parcellated.shape}")
    print(f"  Output: ({n_samples}, {n_voxels})")

    reconstructed = np.zeros((n_samples, n_voxels), dtype=np.float32)

    for parcel_id in tqdm(range(n_parcels), desc="Reconstructing"):
        voxel_indices = np.where(labels == parcel_id)[0]

        if len(voxel_indices) > 0:
            # Broadcast parcel value to all voxels in the parcel
            reconstructed[:, voxel_indices] = parcellated[:, parcel_id:parcel_id+1]

    print(f"\n  Reconstructed data statistics:")
    print(f"    Range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
    print(f"    Mean: {reconstructed.mean():.4f}")
    print(f"    Std: {reconstructed.std():.4f}")

    return reconstructed


def zscore_normalize(data: np.ndarray) -> np.ndarray:
    """
    Z-score normalize across voxel/parcel dimension (per sample).

    Args:
        data: [n_samples, n_features] array

    Returns:
        Z-scored array
    """
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return (data - mean) / (std + 1e-6)


def evaluate_reconstruction_quality(
    original: np.ndarray,
    reconstructed: np.ndarray,
):
    """
    Evaluate reconstruction quality by comparing z-scored versions.

    Args:
        original: [n_samples, n_voxels] original data
        reconstructed: [n_samples, n_voxels] reconstructed data
    """
    print(f"\n{'='*60}")
    print(f"Evaluating reconstruction quality...")
    print(f"{'='*60}")

    # Z-score both
    print(f"  Z-scoring original and reconstructed data...")
    original_z = zscore_normalize(original)
    reconstructed_z = zscore_normalize(reconstructed)

    # Global metrics
    mse = np.mean((original_z - reconstructed_z) ** 2)
    mae = np.mean(np.abs(original_z - reconstructed_z))

    # Flatten for correlation
    corr = np.corrcoef(original_z.flatten(), reconstructed_z.flatten())[0, 1]

    print(f"\n  Global metrics (on z-scored data):")
    print(f"    MSE: {mse:.6f}")
    print(f"    MAE: {mae:.6f}")
    print(f"    Pearson correlation: {corr:.6f}")

    # Per-sample correlation
    n_samples = original.shape[0]
    per_sample_corr = []

    for i in range(n_samples):
        c = np.corrcoef(original_z[i], reconstructed_z[i])[0, 1]
        per_sample_corr.append(c)

    per_sample_corr = np.array(per_sample_corr)

    print(f"\n  Per-sample correlation:")
    print(f"    Mean: {per_sample_corr.mean():.6f}")
    print(f"    Std: {per_sample_corr.std():.6f}")
    print(f"    Min: {per_sample_corr.min():.6f}")
    print(f"    Max: {per_sample_corr.max():.6f}")

    return {
        'mse': mse,
        'mae': mae,
        'global_corr': corr,
        'per_sample_corr_mean': per_sample_corr.mean(),
        'per_sample_corr_std': per_sample_corr.std(),
    }


def parcellate_with_timeseries_kmeans(
    input_path: str,
    output_path: str = None,
    n_parcels: int = 1000,
    n_samples_for_clustering: int = None,
    aggregation: str = 'mean',
    random_state: int = 42,
    use_minibatch: bool = False,
    save_reconstructed: bool = False,
    evaluate: bool = True,
):
    """
    Main function: Parcellate fMRI data using time-series K-means clustering.

    Args:
        input_path: Path to input HDF5 file with 'betas' dataset
        output_path: Path to output HDF5 file
        n_parcels: Number of parcels
        n_samples_for_clustering: Number of samples to use for clustering (None = all)
        aggregation: Aggregation method ('mean' or 'median')
        random_state: Random seed
        use_minibatch: Use MiniBatchKMeans
        save_reconstructed: Save reconstructed data to HDF5
        evaluate: Evaluate reconstruction quality
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_tskm{n_parcels}.hdf5"
    else:
        output_path = Path(output_path)

    print(f"\n{'='*80}")
    print(f"TIME-SERIES K-MEANS PARCELLATION")
    print(f"{'='*80}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Parcels: {n_parcels}")
    print(f"Aggregation: {aggregation}")

    # Step 1: Load subset for clustering
    clustering_data = load_fmri_subset(input_path, n_samples_for_clustering)
    n_samples_cluster, n_voxels = clustering_data.shape

    # Step 2: Cluster voxels by time-series
    labels, kmeans = cluster_voxels_by_timeseries(
        clustering_data,
        n_parcels=n_parcels,
        random_state=random_state,
        use_minibatch=use_minibatch,
    )

    # Step 3: Load ALL data and parcellate
    print(f"\n{'='*60}")
    print(f"Loading full dataset for parcellation...")
    print(f"{'='*60}")

    with h5py.File(input_path, 'r') as f:
        full_fmri = f['betas'][:]

    n_samples_total, n_voxels_check = full_fmri.shape
    assert n_voxels == n_voxels_check, "Voxel count mismatch!"

    print(f"  Full dataset: {full_fmri.shape}")

    parcellated = parcellate_data(full_fmri, labels, n_parcels, aggregation)

    # Step 4: Optionally reconstruct and evaluate
    if evaluate or save_reconstructed:
        # Reconstruct from parcels
        reconstructed = reconstruct_from_parcels(parcellated, labels, n_voxels)

        if evaluate:
            # Use subset for evaluation (faster)
            eval_samples = min(1000, n_samples_total)
            print(f"\n  Evaluating on {eval_samples} samples...")
            metrics = evaluate_reconstruction_quality(
                full_fmri[:eval_samples],
                reconstructed[:eval_samples],
            )

    # Step 5: Save results
    print(f"\n{'='*60}")
    print(f"Saving results...")
    print(f"{'='*60}")

    with h5py.File(output_path, 'w') as f:
        # Save parcellated data
        f.create_dataset('betas', data=parcellated, compression='gzip', compression_opts=4)

        # Save labels (voxel → parcel mapping)
        f.create_dataset('voxel_to_parcel', data=labels.astype(np.int32))

        # Save parcel centroids (time-series patterns)
        f.create_dataset('parcel_centroids', data=kmeans.cluster_centers_.astype(np.float32))

        # Optionally save reconstructed data
        if save_reconstructed:
            f.create_dataset('reconstructed', data=reconstructed, compression='gzip', compression_opts=4)

        # Metadata
        f.attrs['original_shape'] = full_fmri.shape
        f.attrs['parcellated_shape'] = parcellated.shape
        f.attrs['n_parcels'] = n_parcels
        f.attrs['method'] = 'timeseries_kmeans'
        f.attrs['aggregation'] = aggregation
        f.attrs['random_state'] = random_state
        f.attrs['n_samples_for_clustering'] = n_samples_cluster

        if evaluate:
            for k, v in metrics.items():
                f.attrs[f'eval_{k}'] = v

    print(f"  Saved to: {output_path}")

    # Save labels separately
    labels_path = output_path.parent / f"{output_path.stem}_labels.npy"
    np.save(labels_path, labels)
    print(f"  Saved labels: {labels_path.name}")

    # Save model
    model_path = output_path.parent / f"{output_path.stem}_model.pkl"
    joblib.dump(kmeans, model_path)
    print(f"  Saved K-means model: {model_path.name}")

    print(f"\n{'='*80}")
    print(f"✓ Time-series K-means parcellation complete!")
    print(f"{'='*80}")
    print(f"\nCompression: {n_voxels} voxels → {n_parcels} parcels ({n_voxels/n_parcels:.1f}x)")

    original_size = input_path.stat().st_size / 1e9
    new_size = output_path.stat().st_size / 1e9
    print(f"File size: {original_size:.2f} GB → {new_size:.2f} GB ({new_size/original_size*100:.1f}%)")

    if evaluate:
        print(f"\nReconstruction quality:")
        print(f"  Global correlation: {metrics['global_corr']:.6f}")
        print(f"  Per-sample correlation: {metrics['per_sample_corr_mean']:.6f} ± {metrics['per_sample_corr_std']:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Time-series K-means parcellation for fMRI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input HDF5 file with betas')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output HDF5 file')
    parser.add_argument('--n_parcels', type=int, default=1000,
                        help='Number of parcels')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of samples for clustering (None = all)')
    parser.add_argument('--aggregation', type=str, default='mean',
                        choices=['mean', 'median'],
                        help='Aggregation method')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_minibatch', action='store_true',
                        help='Use MiniBatchKMeans (faster)')
    parser.add_argument('--save_reconstructed', action='store_true',
                        help='Save reconstructed data to HDF5')
    parser.add_argument('--no_evaluate', action='store_true',
                        help='Skip reconstruction evaluation')

    args = parser.parse_args()

    parcellate_with_timeseries_kmeans(
        input_path=args.input,
        output_path=args.output,
        n_parcels=args.n_parcels,
        n_samples_for_clustering=args.n_samples,
        aggregation=args.aggregation,
        random_state=args.random_state,
        use_minibatch=args.use_minibatch,
        save_reconstructed=args.save_reconstructed,
        evaluate=not args.no_evaluate,
    )

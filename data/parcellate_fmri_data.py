"""
Parcellate fMRI data from 15724 voxels to 1000 parcels.

This script reduces the dimensionality of fMRI data by grouping voxels
into parcels using K-means clustering or spatial averaging.

Usage:
    python parcellate_fmri_data.py --input dataset/nsd/subj01/betas_all_subj01_fp32_renorm.hdf5 --n_parcels 1000
"""

import os
import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import joblib


def load_mask_coordinates(mask_path: str = None):
    """
    Load mask and get 3D coordinates of voxels.

    Args:
        mask_path: Path to nsdgeneral.nii.gz mask file

    Returns:
        coordinates: [n_voxels, 3] array of (x, y, z) coordinates
    """
    if mask_path is None or not Path(mask_path).exists():
        print("Warning: Mask file not provided or not found.")
        print("Using dummy spatial coordinates based on voxel indices.")
        # Create dummy 3D coordinates
        n_voxels = 15724
        # Arrange voxels in a roughly cubic grid
        side = int(np.ceil(n_voxels ** (1/3)))
        coords = []
        for i in range(n_voxels):
            x = i % side
            y = (i // side) % side
            z = i // (side * side)
            coords.append([x, y, z])
        return np.array(coords, dtype=np.float32)

    import nibabel as nib
    print(f"Loading mask from {mask_path}...")
    mask = nib.load(mask_path).get_fdata()

    # Get coordinates of non-zero voxels
    coords = np.array(np.where(mask > 0)).T  # [n_voxels, 3]
    print(f"  Found {len(coords)} voxels in mask")

    return coords.astype(np.float32)


def create_spatial_parcellation(coordinates, n_parcels=1000, method='kmeans'):
    """
    Create spatial parcellation using K-means clustering.

    Args:
        coordinates: [n_voxels, 3] spatial coordinates
        n_parcels: Number of parcels to create
        method: 'kmeans' or 'hierarchical'

    Returns:
        labels: [n_voxels] parcel assignment for each voxel
    """
    print(f"\nCreating spatial parcellation with {n_parcels} parcels using {method}...")

    if method == 'kmeans':
        # Use MiniBatchKMeans for efficiency
        kmeans = MiniBatchKMeans(
            n_clusters=n_parcels,
            random_state=42,
            batch_size=1000,
            max_iter=100,
            verbose=1
        )
        labels = kmeans.fit_predict(coordinates)
        print(f"  K-means converged in {kmeans.n_iter_} iterations")

    else:
        raise NotImplementedError(f"Method {method} not implemented")

    # Check label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nParcel statistics:")
    print(f"  Number of parcels: {len(unique_labels)}")
    print(f"  Voxels per parcel: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    return labels, kmeans


def apply_parcellation(fmri_data, labels, aggregation='mean'):
    """
    Apply parcellation to fMRI data.

    Args:
        fmri_data: [n_trials, n_voxels] fMRI data
        labels: [n_voxels] parcel assignment
        aggregation: 'mean' or 'median'

    Returns:
        parcellated_data: [n_trials, n_parcels] aggregated data
    """
    n_trials = fmri_data.shape[0]
    n_parcels = len(np.unique(labels))

    print(f"\nApplying parcellation to fMRI data...")
    print(f"  Input shape: {fmri_data.shape}")
    print(f"  Output shape: ({n_trials}, {n_parcels})")
    print(f"  Aggregation method: {aggregation}")

    parcellated_data = np.zeros((n_trials, n_parcels), dtype=np.float32)

    for parcel_id in tqdm(range(n_parcels), desc="Parcels"):
        # Get voxels belonging to this parcel
        voxel_mask = (labels == parcel_id)
        voxels_in_parcel = fmri_data[:, voxel_mask]

        # Aggregate
        if aggregation == 'mean':
            parcellated_data[:, parcel_id] = voxels_in_parcel.mean(axis=1)
        elif aggregation == 'median':
            parcellated_data[:, parcel_id] = np.median(voxels_in_parcel, axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    # Print statistics
    print(f"\nParcellated data statistics:")
    print(f"  Shape: {parcellated_data.shape}")
    print(f"  Min: {parcellated_data.min():.4f}")
    print(f"  Max: {parcellated_data.max():.4f}")
    print(f"  Mean: {parcellated_data.mean():.4f}")
    print(f"  Std: {parcellated_data.std():.4f}")

    return parcellated_data


def parcellate_hdf5_file(
    input_path: str,
    output_path: str = None,
    mask_path: str = None,
    n_parcels: int = 1000,
    method: str = 'kmeans',
    aggregation: str = 'mean',
    save_labels: bool = True,
):
    """
    Parcellate fMRI data from HDF5 file.

    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file (default: auto-generated)
        mask_path: Path to nsdgeneral.nii.gz mask file
        n_parcels: Number of parcels
        method: Parcellation method ('kmeans')
        aggregation: Aggregation method ('mean' or 'median')
        save_labels: Whether to save parcel labels
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_parcel{n_parcels}.hdf5"
    else:
        output_path = Path(output_path)

    print(f"\n{'='*60}")
    print(f"Parcellating fMRI Data")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Parcels: {n_parcels}")
    print(f"Method: {method}")
    print(f"Aggregation: {aggregation}")

    # Load input data
    print(f"\nLoading fMRI data from {input_path.name}...")
    with h5py.File(input_path, 'r') as f:
        fmri_data = f['betas'][:]
        print(f"  Shape: {fmri_data.shape}")
        print(f"  dtype: {fmri_data.dtype}")
        print(f"  Min: {fmri_data.min():.4f}, Max: {fmri_data.max():.4f}")
        print(f"  Mean: {fmri_data.mean():.4f}, Std: {fmri_data.std():.4f}")

    n_trials, n_voxels = fmri_data.shape

    # Load or create spatial coordinates
    coordinates = load_mask_coordinates(mask_path)

    if len(coordinates) != n_voxels:
        print(f"\nWarning: Number of voxels in mask ({len(coordinates)}) "
              f"doesn't match data ({n_voxels})")
        print("Creating dummy spatial coordinates...")
        coordinates = load_mask_coordinates(None)

    # Create parcellation
    labels, model = create_spatial_parcellation(coordinates, n_parcels, method)

    # Apply parcellation
    parcellated_data = apply_parcellation(fmri_data, labels, aggregation)

    # Save parcellated data
    print(f"\nSaving parcellated data to {output_path.name}...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('betas', data=parcellated_data, compression='gzip', compression_opts=4)
        f.create_dataset('parcel_labels', data=labels)

        # Save metadata
        f.attrs['original_shape'] = fmri_data.shape
        f.attrs['n_parcels'] = n_parcels
        f.attrs['method'] = method
        f.attrs['aggregation'] = aggregation
        f.attrs['n_voxels'] = n_voxels

    print(f"  Saved parcellated data: {parcellated_data.shape}")

    # Save parcel labels separately for reference
    if save_labels:
        labels_path = output_path.parent / f"{output_path.stem}_labels.npy"
        np.save(labels_path, labels)
        print(f"  Saved parcel labels: {labels_path.name}")

        # Save k-means model
        if method == 'kmeans':
            model_path = output_path.parent / f"{output_path.stem}_kmeans.pkl"
            joblib.dump(model, model_path)
            print(f"  Saved K-means model: {model_path.name}")

    print(f"\n{'='*60}")
    print(f"✓ Parcellation complete!")
    print(f"{'='*60}")
    print(f"\nReduction factor: {n_voxels}/{n_parcels} = {n_voxels/n_parcels:.1f}x")
    print(f"File size: {input_path.stat().st_size / 1e9:.2f} GB -> "
          f"{output_path.stat().st_size / 1e9:.2f} GB")


def verify_parcellation(output_path: str):
    """
    Verify the parcellated data.
    """
    print(f"\n{'='*60}")
    print(f"Verifying Parcellated Data")
    print(f"{'='*60}")

    output_path = Path(output_path)

    with h5py.File(output_path, 'r') as f:
        print(f"\nDatasets:")
        for key in f.keys():
            print(f"  {key}: {f[key].shape}, dtype={f[key].dtype}")

        print(f"\nMetadata:")
        for key in f.attrs.keys():
            print(f"  {key}: {f.attrs[key]}")

        # Load and check data
        betas = f['betas'][:]
        labels = f['parcel_labels'][:]

        print(f"\nData statistics:")
        print(f"  Betas shape: {betas.shape}")
        print(f"  Min: {betas.min():.4f}")
        print(f"  Max: {betas.max():.4f}")
        print(f"  Mean: {betas.mean():.4f}")
        print(f"  Std: {betas.std():.4f}")

        print(f"\nParcel labels:")
        print(f"  Shape: {labels.shape}")
        print(f"  Unique parcels: {len(np.unique(labels))}")
        print(f"  Min label: {labels.min()}")
        print(f"  Max label: {labels.max()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parcellate fMRI data')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input HDF5 file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output HDF5 file (default: auto-generated)')
    parser.add_argument('--mask', type=str, default=None,
                        help='Path to nsdgeneral.nii.gz mask file')
    parser.add_argument('--n_parcels', type=int, default=1000,
                        help='Number of parcels (default: 1000)')
    parser.add_argument('--method', type=str, default='kmeans',
                        choices=['kmeans'],
                        help='Parcellation method (default: kmeans)')
    parser.add_argument('--aggregation', type=str, default='mean',
                        choices=['mean', 'median'],
                        help='Aggregation method (default: mean)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the output file after parcellation')
    args = parser.parse_args()

    # Parcellate data
    parcellate_hdf5_file(
        input_path=args.input,
        output_path=args.output,
        mask_path=args.mask,
        n_parcels=args.n_parcels,
        method=args.method,
        aggregation=args.aggregation,
    )

    # Verify if requested
    if args.verify:
        output_path = args.output
        if output_path is None:
            input_path = Path(args.input)
            output_path = input_path.parent / f"{input_path.stem}_parcel{args.n_parcels}.hdf5"
        verify_parcellation(output_path)

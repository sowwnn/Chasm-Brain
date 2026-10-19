"""
Utility functions for applying parcellation to fMRI data.

This module provides functions to:
1. Load pre-computed parcellation labels
2. Apply parcellation to new fMRI data
3. Reconstruct from parcels back to voxels
4. Z-score normalization

Usage in training pipeline:
    from data.parcel_utils import ParcelMapper

    # Load parcellation
    mapper = ParcelMapper.from_files(
        labels_path='path/to/labels.npy',
        model_path='path/to/model.pkl'  # optional
    )

    # During training: parcellate raw fMRI
    parcellated = mapper.parcellate(raw_fmri)

    # During inference: reconstruct and z-score
    reconstructed = mapper.reconstruct(predicted_parcels)
    reconstructed_z = mapper.zscore(reconstructed)
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Union


class ParcelMapper:
    """
    Handles parcellation and reconstruction operations.

    Attributes:
        labels: [n_voxels] parcel assignments (0 to n_parcels-1)
        n_voxels: Number of voxels
        n_parcels: Number of parcels
        aggregation: Aggregation method ('mean' or 'median')
        kmeans: Optional fitted KMeans model
    """

    def __init__(
        self,
        labels: np.ndarray,
        n_parcels: int = None,
        aggregation: str = 'mean',
        kmeans=None,
    ):
        """
        Initialize ParcelMapper.

        Args:
            labels: [n_voxels] parcel assignments
            n_parcels: Number of parcels (inferred from labels if None)
            aggregation: Aggregation method ('mean' or 'median')
            kmeans: Optional KMeans model
        """
        self.labels = labels.astype(np.int32)
        self.n_voxels = len(labels)
        self.n_parcels = n_parcels if n_parcels is not None else len(np.unique(labels))
        self.aggregation = aggregation
        self.kmeans = kmeans

        # Pre-compute voxel indices for each parcel (for efficiency)
        self._parcel_to_voxels = {}
        for parcel_id in range(self.n_parcels):
            self._parcel_to_voxels[parcel_id] = np.where(labels == parcel_id)[0]

        print(f"ParcelMapper initialized:")
        print(f"  {self.n_voxels} voxels → {self.n_parcels} parcels")
        print(f"  Aggregation: {self.aggregation}")
        print(f"  Compression: {self.n_voxels / self.n_parcels:.2f}x")

    @classmethod
    def from_files(
        cls,
        labels_path: str,
        model_path: Optional[str] = None,
        aggregation: str = 'mean',
    ):
        """
        Load ParcelMapper from saved files.

        Args:
            labels_path: Path to .npy file with labels
            model_path: Optional path to .pkl file with KMeans model
            aggregation: Aggregation method

        Returns:
            ParcelMapper instance
        """
        print(f"Loading parcellation from files...")
        print(f"  Labels: {Path(labels_path).name}")

        labels = np.load(labels_path)

        kmeans = None
        if model_path is not None:
            print(f"  Model: {Path(model_path).name}")
            kmeans = joblib.load(model_path)

        return cls(labels=labels, aggregation=aggregation, kmeans=kmeans)

    @classmethod
    def from_hdf5(cls, hdf5_path: str, aggregation: str = 'mean'):
        """
        Load ParcelMapper from HDF5 file with metadata.

        Args:
            hdf5_path: Path to HDF5 file with 'voxel_to_parcel' dataset
            aggregation: Aggregation method

        Returns:
            ParcelMapper instance
        """
        import h5py

        print(f"Loading parcellation from HDF5: {Path(hdf5_path).name}")

        with h5py.File(hdf5_path, 'r') as f:
            labels = f['voxel_to_parcel'][:]
            n_parcels = f.attrs.get('n_parcels', None)

        return cls(labels=labels, n_parcels=n_parcels, aggregation=aggregation)

    def parcellate(self, fmri_data: np.ndarray) -> np.ndarray:
        """
        Parcellate fMRI data by aggregating voxels into parcels.

        Args:
            fmri_data: [n_samples, n_voxels] or [n_voxels] array

        Returns:
            parcellated: [n_samples, n_parcels] or [n_parcels] array
        """
        is_1d = (fmri_data.ndim == 1)

        if is_1d:
            fmri_data = fmri_data[np.newaxis, :]  # [1, n_voxels]

        n_samples, n_voxels = fmri_data.shape

        if n_voxels != self.n_voxels:
            raise ValueError(
                f"Input has {n_voxels} voxels but expected {self.n_voxels}"
            )

        parcellated = np.zeros((n_samples, self.n_parcels), dtype=np.float32)

        for parcel_id in range(self.n_parcels):
            voxel_indices = self._parcel_to_voxels[parcel_id]

            if len(voxel_indices) == 0:
                continue

            voxels_in_parcel = fmri_data[:, voxel_indices]

            if self.aggregation == 'mean':
                parcellated[:, parcel_id] = voxels_in_parcel.mean(axis=1)
            elif self.aggregation == 'median':
                parcellated[:, parcel_id] = np.median(voxels_in_parcel, axis=1)
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

        if is_1d:
            parcellated = parcellated[0]  # [n_parcels]

        return parcellated

    def reconstruct(self, parcellated: np.ndarray) -> np.ndarray:
        """
        Reconstruct full voxel space from parcellated data.

        Each voxel gets the value of its assigned parcel.

        Args:
            parcellated: [n_samples, n_parcels] or [n_parcels] array

        Returns:
            reconstructed: [n_samples, n_voxels] or [n_voxels] array
        """
        is_1d = (parcellated.ndim == 1)

        if is_1d:
            parcellated = parcellated[np.newaxis, :]  # [1, n_parcels]

        n_samples, n_parcels = parcellated.shape

        if n_parcels != self.n_parcels:
            raise ValueError(
                f"Input has {n_parcels} parcels but expected {self.n_parcels}"
            )

        reconstructed = np.zeros((n_samples, self.n_voxels), dtype=np.float32)

        for parcel_id in range(self.n_parcels):
            voxel_indices = self._parcel_to_voxels[parcel_id]

            if len(voxel_indices) > 0:
                # Broadcast parcel value to all voxels
                reconstructed[:, voxel_indices] = parcellated[:, parcel_id:parcel_id+1]

        if is_1d:
            reconstructed = reconstructed[0]  # [n_voxels]

        return reconstructed

    @staticmethod
    def zscore(data: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """
        Z-score normalize across feature dimension (per sample).

        Args:
            data: [n_samples, n_features] or [n_features] array
            epsilon: Small constant to avoid division by zero

        Returns:
            Z-scored array with same shape
        """
        is_1d = (data.ndim == 1)

        if is_1d:
            mean = data.mean()
            std = data.std()
            return (data - mean) / (std + epsilon)
        else:
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True)
            return (data - mean) / (std + epsilon)

    def parcellate_and_zscore(self, fmri_data: np.ndarray) -> np.ndarray:
        """
        Parcellate raw fMRI, then z-score the parcellated data.

        This is the recommended pipeline for training:
        Raw fMRI → Parcellate → Z-score

        Args:
            fmri_data: [n_samples, n_voxels] raw fMRI data

        Returns:
            [n_samples, n_parcels] z-scored parcellated data
        """
        parcellated = self.parcellate(fmri_data)
        return self.zscore(parcellated)

    def reconstruct_and_zscore(self, parcellated: np.ndarray) -> np.ndarray:
        """
        Reconstruct from parcels to voxels, then z-score.

        This is the recommended pipeline for evaluation:
        Predicted parcels → Reconstruct → Z-score → Compare with ground truth

        Args:
            parcellated: [n_samples, n_parcels] parcellated data

        Returns:
            [n_samples, n_voxels] z-scored reconstructed data
        """
        reconstructed = self.reconstruct(parcellated)
        return self.zscore(reconstructed)

    def evaluate_reconstruction(
        self,
        original: np.ndarray,
        predicted_parcels: np.ndarray,
    ) -> dict:
        """
        Evaluate reconstruction quality.

        Pipeline:
        1. Reconstruct predicted parcels to voxels
        2. Z-score both original and reconstructed
        3. Compute metrics

        Args:
            original: [n_samples, n_voxels] original raw fMRI
            predicted_parcels: [n_samples, n_parcels] predicted parcels

        Returns:
            dict with metrics: mse, mae, global_corr, per_sample_corr_mean/std
        """
        # Reconstruct and z-score
        reconstructed_z = self.reconstruct_and_zscore(predicted_parcels)
        original_z = self.zscore(original)

        # Global metrics
        mse = np.mean((original_z - reconstructed_z) ** 2)
        mae = np.mean(np.abs(original_z - reconstructed_z))
        corr = np.corrcoef(original_z.flatten(), reconstructed_z.flatten())[0, 1]

        # Per-sample correlation
        n_samples = original_z.shape[0]
        per_sample_corr = []

        for i in range(n_samples):
            c = np.corrcoef(original_z[i], reconstructed_z[i])[0, 1]
            per_sample_corr.append(c)

        per_sample_corr = np.array(per_sample_corr)

        return {
            'mse': float(mse),
            'mae': float(mae),
            'global_corr': float(corr),
            'per_sample_corr_mean': float(per_sample_corr.mean()),
            'per_sample_corr_std': float(per_sample_corr.std()),
            'per_sample_corr_min': float(per_sample_corr.min()),
            'per_sample_corr_max': float(per_sample_corr.max()),
        }

    def get_info(self) -> dict:
        """
        Get information about this parcellation.

        Returns:
            dict with metadata
        """
        unique_labels, counts = np.unique(self.labels, return_counts=True)

        return {
            'n_voxels': self.n_voxels,
            'n_parcels': self.n_parcels,
            'compression_ratio': self.n_voxels / self.n_parcels,
            'aggregation': self.aggregation,
            'voxels_per_parcel_min': int(counts.min()),
            'voxels_per_parcel_max': int(counts.max()),
            'voxels_per_parcel_mean': float(counts.mean()),
            'voxels_per_parcel_median': float(np.median(counts)),
            'voxels_per_parcel_std': float(counts.std()),
            'has_kmeans_model': self.kmeans is not None,
        }


# Convenience functions for common operations

def apply_parcellation_to_hdf5(
    input_hdf5: str,
    output_hdf5: str,
    labels_path: str,
    aggregation: str = 'mean',
    zscore_output: bool = False,
):
    """
    Apply parcellation to an HDF5 file and save result.

    Args:
        input_hdf5: Input HDF5 with 'betas' dataset
        output_hdf5: Output HDF5 path
        labels_path: Path to .npy file with labels
        aggregation: Aggregation method
        zscore_output: Whether to z-score the output
    """
    import h5py

    print(f"Applying parcellation to HDF5 file...")
    print(f"  Input: {Path(input_hdf5).name}")
    print(f"  Output: {Path(output_hdf5).name}")

    # Load mapper
    mapper = ParcelMapper.from_files(labels_path, aggregation=aggregation)

    # Load data
    with h5py.File(input_hdf5, 'r') as f:
        fmri_data = f['betas'][:]

    print(f"  Loaded data: {fmri_data.shape}")

    # Parcellate
    if zscore_output:
        parcellated = mapper.parcellate_and_zscore(fmri_data)
    else:
        parcellated = mapper.parcellate(fmri_data)

    print(f"  Parcellated: {parcellated.shape}")

    # Save
    with h5py.File(output_hdf5, 'w') as f:
        f.create_dataset('betas', data=parcellated, compression='gzip', compression_opts=4)
        f.attrs['original_shape'] = fmri_data.shape
        f.attrs['parcellated_shape'] = parcellated.shape
        f.attrs['n_parcels'] = mapper.n_parcels
        f.attrs['aggregation'] = aggregation
        f.attrs['zscore'] = zscore_output

    print(f"  ✓ Saved to: {output_hdf5}")


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("ParcelMapper Example")
    print("=" * 60)

    # Create example data
    n_samples = 100
    n_voxels = 15724
    n_parcels = 1000

    print(f"\nGenerating example data:")
    print(f"  Samples: {n_samples}")
    print(f"  Voxels: {n_voxels}")
    print(f"  Parcels: {n_parcels}")

    # Random labels
    labels = np.random.randint(0, n_parcels, size=n_voxels)
    fmri_data = np.random.randn(n_samples, n_voxels).astype(np.float32)

    print(f"\nInitializing ParcelMapper...")
    mapper = ParcelMapper(labels, n_parcels=n_parcels)

    # Test parcellation
    print(f"\nTesting parcellation...")
    parcellated = mapper.parcellate(fmri_data)
    print(f"  Input: {fmri_data.shape}")
    print(f"  Output: {parcellated.shape}")

    # Test reconstruction
    print(f"\nTesting reconstruction...")
    reconstructed = mapper.reconstruct(parcellated)
    print(f"  Input: {parcellated.shape}")
    print(f"  Output: {reconstructed.shape}")

    # Test z-score
    print(f"\nTesting z-score...")
    zscored = mapper.zscore(reconstructed)
    print(f"  Mean: {zscored.mean():.6f}")
    print(f"  Std: {zscored.std():.6f}")

    # Test evaluation
    print(f"\nTesting evaluation...")
    metrics = mapper.evaluate_reconstruction(fmri_data, parcellated)
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    print(f"\n✓ All tests passed!")

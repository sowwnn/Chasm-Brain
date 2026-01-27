"""
Point Cloud Dataset for Graph Neural Network fMRI Prediction.

This dataset loads fMRI data in point cloud representation with graph structure.
Based on NeuroFluxDataset but returns graph data instead of flat 1D vectors.

Key features:
- Loads point cloud coordinates and graph edges from pre-computed NPZ file
- Returns node features, spatial coordinates, and edge indices
- Supports same preprocessing as NeuroFluxDataset (clipping, z-score, parcellation)
- Adds optional position jitter augmentation for robustness
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Union
import h5py
import json
from data.parcel_utils import ParcelMapper


# Reuse clipping function from NeuroFluxDataset
def clip_minmax(data, min_val=-2.36, max_val=2.41):
    """
    Clip data to the [min_val, max_val] range (from global dataset stats).
    """
    return np.clip(data, min_val, max_val)


class PointCloudDataset(Dataset):
    """
    Dataset for fMRI prediction using point cloud + graph representation.

    Returns:
        dict with keys:
        - 'fmri': torch.Tensor [N_voxels] - node features (fMRI values)
        - 'coords': torch.Tensor [N_voxels, 3] - spatial coordinates
        - 'edge_index': torch.LongTensor [2, N_edges] - graph edges
        - 'embedding': torch.Tensor [768] - visual features (DINOv2)
        - 'image_id': int - index into embedding array
        - 'subject_id': int - subject number
        - 'roi_labels': torch.LongTensor [N_voxels] - ROI assignments (optional)
    """

    def __init__(
        self,
        fmri_data: Dict[int, np.ndarray],
        embeddings: np.ndarray,
        image_ids: Dict[int, np.ndarray],
        point_cloud_path: str,
        augment_noise: bool = False,          # Embedding noise augmentation
        noise_std: float = 0.1,
        augment_position_jitter: bool = False, # Position jitter augmentation
        jitter_std: float = 0.01,
        parcel_mapper: Optional[ParcelMapper] = None,
        apply_zscore: bool = False,
        **kwargs,
    ):
        """
        Initialize Point Cloud Dataset.

        Args:
            fmri_data: Dict mapping subject_id to fMRI data [n_samples, n_voxels]
            embeddings: DINOv2 embeddings [n_images, 768]
            image_ids: Dict mapping subject_id to image indices [n_samples]
            point_cloud_path: Path to NPZ file with point cloud data
            augment_noise: Enable Gaussian noise on embeddings
            noise_std: Std for embedding noise
            augment_position_jitter: Enable position jitter augmentation
            jitter_std: Std for position jitter (in normalized coordinate space)
            parcel_mapper: Optional parcellation mapper
            apply_zscore: Apply z-score normalization to fMRI
        """
        self.embeddings = embeddings
        self.augment_noise = augment_noise
        self.noise_std = noise_std
        self.augment_position_jitter = augment_position_jitter
        self.jitter_std = jitter_std
        self.parcel_mapper = parcel_mapper
        self.apply_zscore = apply_zscore

        # Load point cloud data
        print(f"Loading point cloud data from {Path(point_cloud_path).name}...")
        pc_data = np.load(point_cloud_path)

        # Spatial coordinates (normalized to [0, 1])
        self.coords_normalized = pc_data['coords_normalized'].astype(np.float32)  # [15724, 3]

        # Graph structure (KNN edges)
        self.edge_index = torch.from_numpy(pc_data['edge_list']).long()  # [2, N_edges]

        # Optional ROI labels
        if 'visual_roi_labels' in pc_data:
            self.roi_labels = torch.from_numpy(pc_data['visual_roi_labels']).long()
        else:
            self.roi_labels = None

        print(f"  Point cloud: {self.coords_normalized.shape[0]} nodes")
        print(f"  Graph edges: {self.edge_index.shape[1]} edges")
        print(f"  Coordinates range: [{self.coords_normalized.min():.3f}, {self.coords_normalized.max():.3f}]")

        # Process fMRI data (same as NeuroFluxDataset)
        self.fmri_list = []
        self.image_id_list = []
        self.subject_ids = []

        subjects = sorted(fmri_data.keys())
        for subj in subjects:
            fmri_s = fmri_data[subj]
            ids_s = image_ids[subj]
            # Clip fMRI
            fmri_s = clip_minmax(fmri_s)
            self.fmri_list.append(fmri_s)
            self.image_id_list.append(ids_s)
            self.subject_ids.extend([subj] * len(fmri_s))

        # Concatenate all subjects
        if len(self.fmri_list) > 0:
            self.fmri_data = np.concatenate(self.fmri_list, axis=0).astype(np.float32)
            self.image_id_data = np.concatenate(self.image_id_list, axis=0).astype(np.int32)
            self.subject_ids = np.array(self.subject_ids)
        else:
            self.fmri_data = np.array([])
            self.image_id_data = np.array([])
            self.subject_ids = np.array([])

        # Verify voxel count matches
        n_voxels_fmri = self.fmri_data.shape[1]
        n_voxels_pc = self.coords_normalized.shape[0]
        assert n_voxels_fmri == n_voxels_pc, \
            f"fMRI voxels ({n_voxels_fmri}) != point cloud nodes ({n_voxels_pc})"

        # Apply parcellation if mapper is provided
        if self.parcel_mapper is not None:
            print(f"Applying parcellation: {self.fmri_data.shape[1]} voxels → {self.parcel_mapper.n_parcels} parcels")
            self.fmri_data = self.parcel_mapper.parcellate(self.fmri_data)
            print(f"Parcellated shape: {self.fmri_data.shape}")
            # Note: When using parcellation, coords will still be per-voxel
            # Model needs to handle aggregation from voxels to parcels

        # Apply z-score if requested
        if self.apply_zscore:
            print(f"Applying z-score normalization...")
            self.fmri_data = ParcelMapper.zscore(self.fmri_data)
            print(f"Z-scored - Mean: {self.fmri_data.mean():.6f}, Std: {self.fmri_data.std():.6f}")

        # Summary
        aug_str = f"with noise augmentation (std={noise_std})" if augment_noise else "without augmentation"
        jitter_str = f", position jitter (std={jitter_std})" if augment_position_jitter else ""
        parcel_str = f", parcellated to {self.parcel_mapper.n_parcels} parcels" if parcel_mapper else ""
        zscore_str = ", z-scored" if apply_zscore else ""

        print(f"PointCloudDataset: {len(self.fmri_data)} samples, "
              f"{len(np.unique(self.image_id_data))} unique images, "
              f"embeddings shape: {self.embeddings.shape}, {aug_str}{jitter_str}{parcel_str}{zscore_str}")

    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            dict with:
            - fmri: [N_voxels] or [N_parcels] if parcellated
            - coords: [N_voxels, 3] spatial coordinates
            - edge_index: [2, N_edges] graph edges (shared across batch)
            - embedding: [768] visual features
            - image_id: int
            - subject_id: int
            - roi_labels: [N_voxels] (optional)
        """
        fmri = self.fmri_data[idx]
        image_id_in_split = self.image_id_data[idx]
        subject_id = self.subject_ids[idx]

        # Get DINOv2 embedding
        embedding = self.embeddings[image_id_in_split].astype(np.float32)

        # Apply noise augmentation to embedding (if enabled)
        if self.augment_noise:
            noise = np.random.normal(0, self.noise_std, embedding.shape).astype(np.float32)
            embedding = embedding + noise

        # Get coordinates (copy to allow augmentation)
        coords = self.coords_normalized.copy()

        # Apply position jitter augmentation (if enabled)
        if self.augment_position_jitter:
            jitter = np.random.normal(0, self.jitter_std, coords.shape).astype(np.float32)
            coords = coords + jitter
            # Clip to valid range
            coords = np.clip(coords, 0, 1)

        # Prepare return dict
        sample = {
            'fmri': torch.from_numpy(fmri).float(),
            'coords': torch.from_numpy(coords).float(),
            'edge_index': self.edge_index,  # Shared across batch
            'embedding': torch.from_numpy(embedding).float(),
            'image_id': image_id_in_split,
            'subject_id': subject_id,
        }

        # Add ROI labels if available
        if self.roi_labels is not None:
            sample['roi_labels'] = self.roi_labels

        return sample


def load_point_cloud_data(
    datalist_path: str,
    fmri_path: str,
    embeddings_path: str,
    point_cloud_path: str,
    subjects: List[int] = [1],
    split: str = 'train',
    average_trials: bool = False,
    augment_noise: bool = False,
    noise_std: float = 0.1,
    augment_position_jitter: bool = False,
    jitter_std: float = 0.01,
    parcel_labels_path: Optional[str] = None,
    apply_zscore: bool = False,
) -> PointCloudDataset:
    """
    Load point cloud dataset for a given split.

    Args:
        datalist_path: Path to datalist JSON file
        fmri_path: Path to fMRI HDF5 file
        embeddings_path: Path to embeddings .npy file for the split
        point_cloud_path: Path to point cloud NPZ file
        subjects: List of subject IDs
        split: 'train' or 'test'
        average_trials: Whether to average 3 trials per image
        augment_noise: Enable embedding noise augmentation
        noise_std: Std for embedding noise
        augment_position_jitter: Enable position jitter augmentation
        jitter_std: Std for position jitter
        parcel_labels_path: Path to parcellation labels (optional)
        apply_zscore: Apply z-score normalization

    Returns:
        PointCloudDataset instance
    """
    # Load datalist
    print(f"Loading datalist from {Path(datalist_path).name}...")
    with open(datalist_path, 'r') as f:
        datalist = json.load(f)

    samples = datalist['samples']

    # Load embeddings
    print(f"Loading {split} embeddings from {Path(embeddings_path).name}...")
    split_embeddings = np.load(embeddings_path)

    # Load parcellation mapper if provided
    parcel_mapper = None
    if parcel_labels_path is not None:
        print(f"Loading parcellation from {Path(parcel_labels_path).name}...")
        parcel_mapper = ParcelMapper.from_files(parcel_labels_path)

    # Load fMRI data
    print(f"Loading fMRI data from {Path(fmri_path).name}...")
    with h5py.File(fmri_path, 'r') as hf:
        full_fmri_data = hf['betas'][:]

    fmri_data = {}
    image_ids = {}

    for subj in subjects:
        print(f"\nProcessing Subject {subj} for {split} split...")

        # Collect trial indices and image indices
        trial_indices = []
        image_indices_in_split = []

        # Create mapping from nsd_id to embedding index
        nsd_id_to_embed_idx = {}
        embed_idx = 0

        for nsd_id_str, nsd_data in samples.items():
            if split in nsd_data['splits']:
                nsd_id_to_embed_idx[nsd_id_str] = embed_idx
                embed_idx += 1

        # Collect all samples
        for nsd_id_str, nsd_data in samples.items():
            if split in nsd_data['splits']:
                split_samples = nsd_data['splits'][split]
                embed_idx = nsd_id_to_embed_idx[nsd_id_str]

                for sample in split_samples:
                    sample_id = sample['sample_id']
                    trial_idx = int(sample_id.replace('sample', ''))
                    trial_indices.append(trial_idx)
                    image_indices_in_split.append(embed_idx)

        if not trial_indices:
            print(f"Warning: No samples found for subject {subj} in split {split}.")
            continue

        # Select fMRI data
        trial_indices = np.array(trial_indices)
        raw_fmri = full_fmri_data[trial_indices]
        image_indices_in_split = np.array(image_indices_in_split)

        # Average trials if requested
        if average_trials:
            print(f"  Averaging trials for {split} split...")
            unique_img_indices = np.unique(image_indices_in_split)
            avg_fmri = []
            final_img_indices = []

            for img_idx in unique_img_indices:
                mask = (image_indices_in_split == img_idx)
                avg_fmri.append(np.mean(raw_fmri[mask], axis=0))
                final_img_indices.append(img_idx)

            fmri_data[subj] = np.stack(avg_fmri)
            image_ids[subj] = np.array(final_img_indices)
        else:
            fmri_data[subj] = raw_fmri
            image_ids[subj] = image_indices_in_split

        print(f"  {split.capitalize()} fMRI: {fmri_data[subj].shape}, "
              f"{len(np.unique(image_ids[subj]))} unique images")

    # Create point cloud dataset
    dataset = PointCloudDataset(
        fmri_data=fmri_data,
        embeddings=split_embeddings,
        image_ids=image_ids,
        point_cloud_path=point_cloud_path,
        augment_noise=augment_noise,
        noise_std=noise_std,
        augment_position_jitter=augment_position_jitter,
        jitter_std=jitter_std,
        parcel_mapper=parcel_mapper,
        apply_zscore=apply_zscore,
    )

    return dataset


def create_point_cloud_dataloaders(
    datalist_path: str,
    fmri_path: str,
    train_embeddings_path: str,
    test_embeddings_path: str,
    point_cloud_path: str,
    subjects: List[int] = [1],
    batch_size: int = 16,
    average_trials_train: bool = False,
    average_trials_val: bool = True,
    augment_noise_train: bool = False,
    noise_std: float = 0.1,
    augment_position_jitter_train: bool = False,
    jitter_std: float = 0.01,
    parcel_labels_path: Optional[str] = None,
    apply_zscore_train: bool = False,
    apply_zscore_val: bool = False,
    num_workers: int = 4,
    collate_fn = None,
) -> tuple:
    """
    Create train and validation dataloaders for point cloud data.

    Note: Unlike NeuroFluxDataset, we don't use SubjectSampler here
    because graph batching is more complex and typically handled
    by PyTorch Geometric's custom collate functions.

    Returns:
        (train_loader, val_loader)
    """
    # Training dataset with augmentation
    train_dataset = load_point_cloud_data(
        datalist_path=datalist_path,
        fmri_path=fmri_path,
        embeddings_path=train_embeddings_path,
        point_cloud_path=point_cloud_path,
        subjects=subjects,
        split='train',
        average_trials=average_trials_train,
        augment_noise=augment_noise_train,
        noise_std=noise_std,
        augment_position_jitter=augment_position_jitter_train,
        jitter_std=jitter_std,
        parcel_labels_path=parcel_labels_path,
        apply_zscore=apply_zscore_train,
    )

    # Validation dataset (no augmentation)
    val_dataset = load_point_cloud_data(
        datalist_path=datalist_path,
        fmri_path=fmri_path,
        embeddings_path=test_embeddings_path,
        point_cloud_path=point_cloud_path,
        subjects=subjects,
        split='test',
        average_trials=average_trials_val,
        augment_noise=False,
        noise_std=0.0,
        augment_position_jitter=False,
        jitter_std=0.0,
        parcel_labels_path=parcel_labels_path,
        apply_zscore=apply_zscore_val,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


# Test script
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test Point Cloud Dataset')
    parser.add_argument('--datalist', type=str, required=True, help='Path to datalist JSON')
    parser.add_argument('--fmri', type=str, required=True, help='Path to fMRI HDF5')
    parser.add_argument('--train_emb', type=str, required=True, help='Path to train embeddings')
    parser.add_argument('--test_emb', type=str, required=True, help='Path to test embeddings')
    parser.add_argument('--point_cloud', type=str, required=True, help='Path to point cloud NPZ')
    parser.add_argument('--subjects', type=str, default='[1]', help='List of subject IDs')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    subjects = json.loads(args.subjects)

    print("=" * 60)
    print("Testing Point Cloud Dataset")
    print("=" * 60)

    train_loader, val_loader = create_point_cloud_dataloaders(
        datalist_path=args.datalist,
        fmri_path=args.fmri,
        train_embeddings_path=args.train_emb,
        test_embeddings_path=args.test_emb,
        point_cloud_path=args.point_cloud,
        subjects=subjects,
        batch_size=args.batch_size,
        augment_position_jitter_train=True,
        jitter_std=0.01,
    )

    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")

    # Test one batch
    for batch in train_loader:
        print(f"\nSample batch:")
        print(f"  fMRI shape: {batch['fmri'].shape}")
        print(f"  Coords shape: {batch['coords'].shape}")
        print(f"  Edge index shape: {batch['edge_index'].shape}")
        print(f"  Embedding shape: {batch['embedding'].shape}")
        print(f"  Image IDs: {batch['image_id'][:5]}...")
        print(f"  Subject IDs: {batch['subject_id'][:5]}...")
        if 'roi_labels' in batch:
            print(f"  ROI labels shape: {batch['roi_labels'].shape}")
        break

    print("\n✓ Point Cloud Dataset test passed!")

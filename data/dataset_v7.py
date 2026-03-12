"""
NeuroFlux Dataset V7: Support for DINOv2 CLS + Patches embeddings.

Key changes from V6:
1. Support loading (N, 257, 768) embeddings: CLS + 256 patches
2. Returns both cls_token and patch_tokens separately
3. Dynamic ROI computation based on voxels_per_cluster
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
from pathlib import Path
from typing import Dict, List, Optional




class NeuroFluxDatasetV7(Dataset):
    """
    Dataset for HierarchicalARM V3 with DINOv2 CLS + Patches.

    Returns:
        - fmri: [fmri_dim] full voxels
        - cls_token: [768] CLS token
        - patch_tokens: [256, 768] patch tokens
        - roi_means: [num_rois] ROI aggregated values
    """

    def __init__(
        self,
        fmri_data: Dict[int, np.ndarray],
        embeddings: np.ndarray,  # Shape: [N, 257, 768] or [N, 768]
        image_ids: Dict[int, np.ndarray],
        voxels_per_cluster: int = 30,
        roi_top_k_percent: float = 0.7,
        augment_noise: bool = False,
        noise_std: float = 0.1,
        use_float16: bool = False,  # Keep float32 by default for precision
        **kwargs,
    ):
        """
        Args:
            fmri_data: Dict[subject_id -> fMRI array]
            embeddings: DINOv2 embeddings, shape [N, 257, 768] (CLS + patches) or [N, 768] (CLS only)
            image_ids: Dict[subject_id -> image indices in embeddings]
            voxels_per_cluster: Number of voxels per ROI cluster
            roi_top_k_percent: Use top-k% voxels for ROI aggregation
            augment_noise: Add noise to embeddings during training
            noise_std: Noise standard deviation
        """
        # Check embeddings format (keep original array, convert in __getitem__)
        if embeddings.ndim == 2:
            # Old format: [N, 768] - CLS only
            print(f"Warning: Embeddings are CLS-only format [N, {embeddings.shape[1]}]. "
                  "Consider using CLS+patches format for V3 model.")
            self.embeddings_format = "cls_only"
            self.embeddings = embeddings  # Keep original (may be mmap)
        elif embeddings.ndim == 3 and embeddings.shape[1] == 257:
            # New format: [N, 257, 768] - CLS + 256 patches
            self.embeddings_format = "cls_patches"
            self.embeddings = embeddings  # Keep original (may be mmap)
            print(f"Using CLS + patches format: {embeddings.shape}")
        else:
            raise ValueError(f"Unsupported embeddings shape: {embeddings.shape}. "
                           "Expected [N, 768] or [N, 257, 768]")

        self.voxels_per_cluster = voxels_per_cluster
        self.roi_top_k_percent = roi_top_k_percent
        self.augment_noise = augment_noise
        self.noise_std = noise_std
        self.use_float16 = use_float16

        # Process fMRI data
        self.fmri_list = []
        self.image_id_list = []
        self.subject_ids = []

        for subj in sorted(fmri_data.keys()):
            # No preprocessing - use raw fMRI data
            fmri_s = fmri_data[subj]
            ids_s = image_ids[subj]
            self.fmri_list.append(fmri_s)
            self.image_id_list.append(ids_s)
            self.subject_ids.extend([subj] * len(fmri_s))

        if len(self.fmri_list) > 0:
            dtype = np.float16 if use_float16 else np.float32
            self.fmri_data = np.concatenate(self.fmri_list, axis=0).astype(dtype)
            self.image_id_data = np.concatenate(self.image_id_list, axis=0).astype(np.int32)
            self.subject_ids = np.array(self.subject_ids)
        else:
            self.fmri_data = np.array([])
            self.image_id_data = np.array([])
            self.subject_ids = np.array([])

        # Compute ROI info
        self.fmri_dim = self.fmri_data.shape[1] if len(self.fmri_data) > 0 else 15724
        self.num_rois = self.fmri_dim // voxels_per_cluster
        self.remainder_voxels = self.fmri_dim % voxels_per_cluster

        print(f"NeuroFluxDatasetV7: {len(self.fmri_data)} samples, "
              f"{self.fmri_dim} voxels → {self.num_rois} ROIs "
              f"({voxels_per_cluster} voxels/ROI, top-{int(roi_top_k_percent*100)}%) "
              f"[dtype={'float16' if use_float16 else 'float32'}]")

    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        fmri = self.fmri_data[idx].astype(np.float32)
        image_id = self.image_id_data[idx]
        subject_id = self.subject_ids[idx]

        # Get embeddings
        if self.embeddings_format == "cls_patches":
            emb = self.embeddings[image_id].astype(np.float32)  # [257, 768]
            cls_token = emb[0]  # [768]
            patch_tokens = emb[1:]  # [256, 768]
        else:
            # CLS only - create dummy patches
            cls_token = self.embeddings[image_id].astype(np.float32)  # [768]
            patch_tokens = np.zeros((256, 768), dtype=np.float32)

        # Add noise augmentation
        if self.augment_noise:
            noise_cls = np.random.normal(0, self.noise_std, cls_token.shape).astype(np.float32)
            noise_patch = np.random.normal(0, self.noise_std, patch_tokens.shape).astype(np.float32)
            cls_token = cls_token + noise_cls
            patch_tokens = patch_tokens + noise_patch

        # Compute ROI means on-the-fly
        roi_means = self._compute_roi_means(fmri)

        return {
            'fmri': torch.from_numpy(fmri).float(),
            'cls_token': torch.from_numpy(cls_token).float(),
            'patch_tokens': torch.from_numpy(patch_tokens).float(),
            'roi_means': torch.from_numpy(roi_means).float(),
            'image_id': image_id,
            'subject_id': subject_id,
        }

    def _compute_roi_means(self, fmri):
        """
        Compute ROI means by sequential grouping (not KMeans).

        Groups voxels sequentially into clusters of voxels_per_cluster size.
        Uses top-k% aggregation within each cluster.
        """
        roi_means = np.zeros(self.num_rois, dtype=np.float32)

        for i in range(self.num_rois):
            start_idx = i * self.voxels_per_cluster
            end_idx = start_idx + self.voxels_per_cluster
            cluster_voxels = fmri[start_idx:end_idx]

            if self.roi_top_k_percent is not None and 0 < self.roi_top_k_percent < 1:
                # Top-k% aggregation
                k = max(1, int(len(cluster_voxels) * self.roi_top_k_percent))
                top_k_values = np.partition(cluster_voxels, -k)[-k:]
                roi_means[i] = top_k_values.mean()
            else:
                # Simple mean
                roi_means[i] = cluster_voxels.mean()

        return roi_means


class SubjectSampler(Sampler):
    """Sampler that ensures each batch contains samples from only one subject."""

    def __init__(self, dataset, batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.subject_ids = np.array(dataset.subject_ids)
        self.unique_subjects = np.unique(self.subject_ids)
        self.drop_last = drop_last

    def __iter__(self):
        batches = []
        for subject in self.unique_subjects:
            indices = np.where(self.subject_ids == subject)[0]
            np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        np.random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class NeuroFluxConcatDataset(ConcatDataset):
    """
    Custom ConcatDataset that exposes subject_ids for SubjectSampler.
    Used to combine train + val datasets for training.
    """
    def __init__(self, datasets):
        super().__init__(datasets)
        # Concatenate subject_ids from all datasets
        self.subject_ids = np.concatenate([ds.subject_ids for ds in datasets])


def load_neuroflux_data_v7(
    datalist_path: str,
    fmri_path: str,
    embeddings_path: str,
    subjects: List[int] = [1],
    split: str = 'train',
    average_trials: bool = False,
    voxels_per_cluster: int = 30,
    roi_top_k_percent: float = 0.7,
    augment_noise: bool = False,
    noise_std: float = 0.1,
    image_ids_path: Optional[str] = None,  # Path to image_ids_{split}_sub01.npy
    use_float16: bool = False,  # Keep float32 by default for reproducibility
) -> NeuroFluxDatasetV7:
    """
    Load data for HierarchicalARM V3.

    Args:
        datalist_path: Path to datalist JSON
        fmri_path: Path to fMRI HDF5
        embeddings_path: Path to embeddings .npy (CLS+patches format)
        subjects: List of subject IDs
        split: 'train' or 'test'
        average_trials: Average trials per image
        voxels_per_cluster: Voxels per ROI
        roi_top_k_percent: Top-k% for ROI aggregation
        augment_noise: Add noise during training
        noise_std: Noise std

    Returns:
        NeuroFluxDatasetV7
    """
    import json
    import h5py

    print(f"Loading datalist from {datalist_path}...")
    with open(datalist_path, 'r') as f:
        datalist = json.load(f)

    samples = datalist['samples']

    print(f"Loading {split} embeddings from {embeddings_path}...")
    # Use memory mapping to avoid loading all embeddings into RAM at once
    split_embeddings = np.load(embeddings_path, mmap_mode='r')

    # Load image_ids mapping if provided (authoritative source for embedding order)
    if image_ids_path is not None:
        print(f"Loading image IDs mapping from {image_ids_path}...")
        emb_nsd_ids = np.load(image_ids_path)
        # Create mapping: NSD ID -> embedding index
        nsd_id_to_embed_idx = {int(nsd_id): idx for idx, nsd_id in enumerate(emb_nsd_ids)}
        print(f"  Loaded {len(nsd_id_to_embed_idx)} NSD ID -> embedding index mappings")
    else:
        # Fallback: assume JSON iteration order (legacy behavior)
        print("  Warning: No image_ids_path provided, using JSON iteration order")
        nsd_id_to_embed_idx = {}
        embed_idx = 0
        for nsd_id_str, nsd_data in samples.items():
            if split in nsd_data['splits']:
                nsd_id_to_embed_idx[int(nsd_id_str)] = embed_idx
                embed_idx += 1

    print(f"Loading fMRI data from {fmri_path}...")
    with h5py.File(fmri_path, 'r') as hf:
        full_fmri_data = hf['betas'][:]

    fmri_data = {}
    image_ids = {}

    for subj in subjects:
        print(f"\nProcessing Subject {subj} for {split} split...")

        trial_indices = []
        image_indices_in_split = []

        # Collect samples using the NSD ID -> embedding index mapping
        for nsd_id_str, nsd_data in samples.items():
            if split in nsd_data['splits']:
                nsd_id = int(nsd_id_str)
                embed_idx = nsd_id_to_embed_idx.get(nsd_id)

                if embed_idx is None:
                    print(f"  Warning: NSD ID {nsd_id} not found in embedding mapping, skipping...")
                    continue

                split_samples = nsd_data['splits'][split]
                for sample in split_samples:
                    sample_id = sample['sample_id']
                    trial_idx = int(sample_id.replace('sample', ''))
                    trial_indices.append(trial_idx)
                    image_indices_in_split.append(embed_idx)

        if not trial_indices:
            print(f"Warning: No samples found for subject {subj} in split {split}.")
            continue

        trial_indices = np.array(trial_indices)
        raw_fmri = full_fmri_data[trial_indices]
        image_indices_in_split = np.array(image_indices_in_split)

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

    return NeuroFluxDatasetV7(
        fmri_data=fmri_data,
        embeddings=split_embeddings,
        image_ids=image_ids,
        voxels_per_cluster=voxels_per_cluster,
        roi_top_k_percent=roi_top_k_percent,
        augment_noise=augment_noise,
        noise_std=noise_std,
        use_float16=use_float16,
    )


def create_dataloaders_v7(
    datalist_path: str,
    fmri_path: str,
    train_embeddings_path: str,
    test_embeddings_path: str,
    subjects: List[int] = [1],
    batch_size: int = 32,
    voxels_per_cluster: int = 30,
    roi_top_k_percent: float = 0.7,
    average_trials_train: bool = False,
    average_trials_val: bool = True,
    augment_noise_train: bool = False,
    noise_std: float = 0.1,
    num_workers: int = 8,
    train_image_ids_path: Optional[str] = None,
    test_image_ids_path: Optional[str] = None,
    use_float16: bool = False,  # Keep float32 by default for reproducibility
):
    """Create train and validation dataloaders for V3 model."""

    train_dataset = load_neuroflux_data_v7(
        datalist_path=datalist_path,
        fmri_path=fmri_path,
        embeddings_path=train_embeddings_path,
        subjects=subjects,
        split='train',
        average_trials=average_trials_train,
        voxels_per_cluster=voxels_per_cluster,
        roi_top_k_percent=roi_top_k_percent,
        augment_noise=augment_noise_train,
        noise_std=noise_std,
        image_ids_path=train_image_ids_path,
        use_float16=use_float16,
    )

    val_dataset = load_neuroflux_data_v7(
        datalist_path=datalist_path,
        fmri_path=fmri_path,
        embeddings_path=test_embeddings_path,
        subjects=subjects,
        split='test',
        average_trials=average_trials_val,
        voxels_per_cluster=voxels_per_cluster,
        roi_top_k_percent=roi_top_k_percent,
        augment_noise=False,
        noise_std=0.0,
        image_ids_path=test_image_ids_path,
        use_float16=use_float16,
    )
    
    # train_dataset = NeuroFluxConcatDataset([train_dataset, val_dataset])
    train_sampler = SubjectSampler(train_dataset, batch_size=batch_size, drop_last=True)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # Test with patches embeddings
    print("=" * 60)
    print("Testing NeuroFluxDatasetV7 with CLS + Patches")
    print("=" * 60)

    # Create dummy data
    N = 100
    fmri_dim = 15724
    voxels_per_cluster = 30

    dummy_fmri = {1: np.random.randn(N, fmri_dim).astype(np.float32)}
    dummy_embeddings = np.random.randn(N, 257, 768).astype(np.float32)  # CLS + patches
    dummy_ids = {1: np.arange(N)}

    dataset = NeuroFluxDatasetV7(
        fmri_data=dummy_fmri,
        embeddings=dummy_embeddings,
        image_ids=dummy_ids,
        voxels_per_cluster=voxels_per_cluster,
        roi_top_k_percent=0.7,
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Num ROIs: {dataset.num_rois}")

    # Test one sample
    sample = dataset[0]
    print(f"\nSample:")
    print(f"  fmri: {sample['fmri'].shape}")
    print(f"  cls_token: {sample['cls_token'].shape}")
    print(f"  patch_tokens: {sample['patch_tokens'].shape}")
    print(f"  roi_means: {sample['roi_means'].shape}")

    print("\n✓ Dataset test passed!")

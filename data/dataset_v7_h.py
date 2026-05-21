"""
NeuroFlux Dataset V7-H: Dataset loader for NSD data format

Dataset loader for Natural Scenes Dataset (NSD) with multi-trial fMRI data.
Key features:
1. Loads fMRI data from .npy files
2. No datalist JSON required - uses simple image indexing
3. Compatible with DINOv2 embedding and training pipeline

NSD Dataset structure:
- dataset/nsd/subjXX/nsd_train_fmri_zscore_subX.npy: shape (N_train, 3, 15724)
- dataset/nsd/subjXX/nsd_test_fmri_zscore_subX.npy: shape (N_test, 3, 15724)
- Where: N is number of images, 3 is number of trials, 15724 is number of voxels

Returns:
    - fmri: [fmri_dim] full voxels
    - cls_token: [768] CLS token
    - patch_tokens: [256, 768] patch tokens
    - roi_means: [num_rois] ROI aggregated values
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
from pathlib import Path
from typing import Dict, List, Optional


class NeuroFluxDatasetV7H(Dataset):
    """
    Dataset for HierarchicalARM V3 with DINOv2 CLS + Patches.
    Works with NSD dataset format (.npy files).

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
        use_float16: bool = False,
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
        # Check embeddings format
        if embeddings.ndim == 2:
            # Old format: [N, 768] - CLS only
            print(f"Warning: Embeddings are CLS-only format [N, {embeddings.shape[1]}]. "
                  "Consider using CLS+patches format for V3 model.")
            self.embeddings_format = "cls_only"
            self.embeddings = embeddings
        elif embeddings.ndim == 3 and embeddings.shape[1] == 257:
            # New format: [N, 257, 768] - CLS + 256 patches
            self.embeddings_format = "cls_patches"
            self.embeddings = embeddings
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

        print(f"NeuroFluxDatasetV7H: {len(self.fmri_data)} samples, "
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
        Compute ROI means by sequential grouping.

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



def load_neuroflux_data_v7h(
    nsd_base_path: str,
    embeddings_path: Optional[str] = None,  # Optional - will auto-load from NSD folder if None
    subjects: List[int] = [1],
    split: str = 'train',
    average_trials: bool = False,
    voxels_per_cluster: int = 30,
    roi_top_k_percent: float = 0.7,
    augment_noise: bool = False,
    noise_std: float = 0.1,
    image_ids_path: Optional[str] = None,  # Deprecated - embeddings are pre-aligned
    use_float16: bool = False,
    dinov2_layer: int = -1,  # Which layer to use from multilayer features (-1 = last)
) -> NeuroFluxDatasetV7H:
    """
    Load data from NSD dataset format for HierarchicalARM V3.

    Args:
        nsd_base_path: Path to NSD dataset base (e.g., "dataset/nsd")
        embeddings_path: Path to embeddings .npy. If None, auto-loads from NSD folder
        subjects: List of subject IDs (1, 2, 5, 7 for NSD dataset)
        split: 'train' or 'test'
        average_trials: Average trials per image (NSD has 3 trials per image)
        voxels_per_cluster: Voxels per ROI
        roi_top_k_percent: Top-k% for ROI aggregation
        augment_noise: Add noise during training
        noise_std: Noise std
        image_ids_path: Deprecated - embeddings are already aligned with fMRI
        use_float16: Use float16 for memory efficiency
        dinov2_layer: Which DINOv2 layer to use (0-3, or -1 for last layer)

    Returns:
        NeuroFluxDatasetV7H
    """
    nsd_path = Path(nsd_base_path)

    # Auto-detect embeddings path if not provided
    if embeddings_path is None:
        # Use built-in NSD embeddings
        if subjects and len(subjects) > 0:
            subj = subjects[0]
            subj_str = f"sub{subj}"
            embeddings_path = str(nsd_path / f"subj0{subj}" / f"nsd_dinov2_vitb14_multilayer_{split}_{subj_str}.npy")
            print(f"Auto-detected embeddings path: {embeddings_path}")
    
    print(f"Loading {split} embeddings from {embeddings_path}...")
    # Use memory mapping to avoid loading all embeddings into RAM
    split_embeddings_raw = np.load(embeddings_path, mmap_mode='r')
    
    # Handle multilayer format: (N, 4, 257, 768) -> (N, 257, 768)
    if split_embeddings_raw.ndim == 4:
        print(f"  Multilayer embeddings detected: {split_embeddings_raw.shape}")
        print(f"  Using layer {dinov2_layer} (0-indexed, -1=last)")
        split_embeddings = split_embeddings_raw[:, dinov2_layer, :, :]
        print(f"  Selected shape: {split_embeddings.shape}")
    else:
        split_embeddings = split_embeddings_raw
        print(f"  Single layer embeddings: {split_embeddings.shape}")
    
    # Deprecated warning for image_ids_path
    if image_ids_path is not None:
        print(f"  WARNING: image_ids_path is deprecated.")
        print(f"  Embeddings are pre-aligned with fMRI. Ignoring image_ids_path.")

    fmri_data = {}
    image_ids = {}

    for subj in subjects:
        print(f"\nProcessing Subject {subj} for {split} split...")

        # Construct file path for NSD dataset
        # Format: dataset/nsd/subjXX/nsd_{split}_fmri_zscore_subX.npy
        if subj < 10:
            subj_str = f"sub{subj}"
        else:
            subj_str = f"sub{subj}"

        fmri_file = nsd_path / f"subj0{subj}" / f"nsd_{split}_fmri_zscore_{subj_str}.npy"
        
        if not fmri_file.exists():
            print(f"Warning: File {fmri_file} not found, skipping subject {subj}")
            continue
        
        # Load fMRI data: shape (N, 3, 15724)
        print(f"  Loading fMRI from {fmri_file}...")
        raw_fmri = np.load(fmri_file)  # (N_images, 3_trials, 15724_voxels)
        print(f"  Loaded shape: {raw_fmri.shape}")
        
        n_images = raw_fmri.shape[0]
        n_trials = raw_fmri.shape[1]
        n_voxels = raw_fmri.shape[2]
        
        if average_trials:
            # Average across trials: (N, 3, 15724) -> (N, 15724)
            print(f"  Averaging {n_trials} trials per image...")
            fmri_s = np.mean(raw_fmri, axis=1).astype(np.float32)
            # Image IDs: simple sequential from 0 to N-1
            image_ids_s = np.arange(n_images)
            
            print(f"  After averaging: {fmri_s.shape}")
        else:
            # Keep all trials: (N, 3, 15724) -> (N*3, 15724)
            print(f"  Keeping all {n_trials} trials...")
            fmri_s = raw_fmri.reshape(-1, n_voxels).astype(np.float32)
            # Each image has 3 trials, so repeat image indices
            image_ids_s = np.repeat(np.arange(n_images), n_trials)
            
            print(f"  After flattening trials: {fmri_s.shape}")
        
        fmri_data[subj] = fmri_s
        image_ids[subj] = image_ids_s
        
        print(f"  {split.capitalize()} fMRI: {fmri_data[subj].shape}, "
              f"{len(np.unique(image_ids[subj]))} unique images")

    return NeuroFluxDatasetV7H(
        fmri_data=fmri_data,
        embeddings=split_embeddings,
        image_ids=image_ids,
        voxels_per_cluster=voxels_per_cluster,
        roi_top_k_percent=roi_top_k_percent,
        augment_noise=augment_noise,
        noise_std=noise_std,
        use_float16=use_float16,
    )


def create_dataloaders_v7h(
    nsd_base_path: str,
    train_embeddings_path: Optional[str] = None,  # Optional - auto-loads from NSD
    test_embeddings_path: Optional[str] = None,   # Optional - auto-loads from NSD
    subjects: List[int] = [1],
    batch_size: int = 32,
    voxels_per_cluster: int = 30,
    roi_top_k_percent: float = 0.7,
    average_trials_train: bool = False,
    average_trials_val: bool = True,
    augment_noise_train: bool = False,
    noise_std: float = 0.1,
    num_workers: int = 8,
    train_image_ids_path: Optional[str] = None,  # Deprecated
    test_image_ids_path: Optional[str] = None,   # Deprecated
    use_float16: bool = False,
    concat_train_val: bool = False,
    dinov2_layer: int = -1,  # Which DINOv2 layer to use
):
    """
    Create train and validation dataloaders for V3 model using NSD dataset.

    Args:
        nsd_base_path: Path to NSD dataset (e.g., "dataset/nsd")
        train_embeddings_path: Optional path to train embeddings (auto-loads if None)
        test_embeddings_path: Optional path to test embeddings (auto-loads if None)
        dinov2_layer: Which layer to use from multilayer features (-1 = last)
    """
    train_dataset = load_neuroflux_data_v7h(
        nsd_base_path=nsd_base_path,
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
        dinov2_layer=dinov2_layer,
    )

    val_dataset = load_neuroflux_data_v7h(
        nsd_base_path=nsd_base_path,
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
        dinov2_layer=dinov2_layer,
    )
    
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
    # Test with NSD dataset
    print("=" * 60)
    print("Testing NeuroFluxDatasetV7H with NSD dataset")
    print("=" * 60)

    # Test with dummy data first
    N = 100
    fmri_dim = 15724
    voxels_per_cluster = 30

    dummy_fmri = {1: np.random.randn(N, fmri_dim).astype(np.float32)}
    dummy_embeddings = np.random.randn(N, 257, 768).astype(np.float32)  # CLS + patches
    dummy_ids = {1: np.arange(N)}

    dataset = NeuroFluxDatasetV7H(
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

    print("\nDataset test passed!")

    # Test with real NSD data if available
    nsd_path = Path("dataset/nsd")
    if nsd_path.exists():
        print("\n" + "=" * 60)
        print("Testing with real NSD data")
        print("=" * 60)

        try:
            # Test loading without embeddings (just structure test)
            test_file = nsd_path / "subj01" / "nsd_test_fmri_zscore_sub1.npy"
            if test_file.exists():
                data = np.load(test_file)
                print(f"Loaded {test_file.name}: {data.shape}")
                print(f"Expected format: (N_images, 3_trials, 15724_voxels)")
                print("NSD data structure verified!")
        except Exception as e:
            print(f"Error testing NSD data: {e}")

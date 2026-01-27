"""
NeuroFlux Dataset for NeuroSDE with DINOv2 Embeddings.

Adapted from SynBrain's fMRI_feature_Dataset with key changes:
- Uses DINOv2 embeddings instead of CLIP
- Global 73k embeddings indexed by NSD Image ID
- Supports virtual indexing for 3-trial fMRI repetition
- Supports parcellation via ParcelMapper (following analy.ipynb approach)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
from pathlib import Path
from typing import Dict, List, Optional, Union
import scipy.io as spio
from data.parcel_utils import ParcelMapper

# Robust clipping function for preprocessing
def clip_minmax(data, min_val=-2.36, max_val=2.41):
    """
    Clip data to the [min_val, max_val] range (from global dataset stats).
    Args:
        data: np.ndarray (1D or 2D)
        min_val: lower bound (default -2.36)
        max_val: upper bound (default 2.41)
    Returns:
        np.ndarray with values clipped to [min_val, max_val]
    """
    return np.clip(data, min_val, max_val)


def load_nsd_expdesign(expdesign_path: str):
    """
    Load NSD experiment design file to get image ID mappings.
    
    Returns:
        dict with keys:
        - masterordering: [750*40] trial order across sessions
        - subjectim: [8, 10000] subject-specific image presentation
    """
    def _check_keys(d):
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d
    
    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = elem
            else:
                d[strg] = elem
        return d
    
    data = spio.loadmat(expdesign_path, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def get_image_ids_for_subject(expdesign: dict, subject_id: int, num_sessions: int = 40) -> dict:
    """
    Get NSD image IDs for train and test splits for a subject.
    
    Args:
        expdesign: Loaded nsd_expdesign.mat
        subject_id: Subject number (1-8)
        num_sessions: Number of sessions (default 40)
    
    Returns:
        dict with:
        - train_ids: list of (trial_idx, image_id) for training
        - test_ids: list of (trial_idx, image_id) for test (shared1000)
    """
    sig_train = {}
    sig_test = {}
    num_trials = num_sessions * 750
    
    subjectim = expdesign['subjectim']
    masterordering = expdesign['masterordering']
    
    for trial_idx in range(num_trials):
        # NSD ID: 1-based index into 73k images
        nsd_id = subjectim[subject_id - 1, masterordering[trial_idx] - 1] - 1  # Convert to 0-based
        
        if masterordering[trial_idx] > 1000:
            # Training data
            if nsd_id not in sig_train:
                sig_train[nsd_id] = []
            sig_train[nsd_id].append(trial_idx)
        else:
            # Test data (shared1000)
            if nsd_id not in sig_test:
                sig_test[nsd_id] = []
            sig_test[nsd_id].append(trial_idx)
    
    return {
        'train': sig_train,  # {image_id: [trial_indices]}
        'test': sig_test
    }


class NeuroFluxDataset(Dataset):
    """
    Dataset for NeuroFlux-SDE with DINOv2 embeddings.

    This version is simplified to work with pre-filtered fMRI and embedding data.
    Supports data augmentation via noise injection during training.
    Supports parcellation following analy.ipynb approach:
    - Raw fMRI → Parcellate → (optionally Z-score) → Model
    """

    def __init__(
        self,
        fmri_data: Dict[int, np.ndarray],
        embeddings: np.ndarray,
        image_ids: Dict[int, np.ndarray],
        augment_noise: bool = False,      # Enable noise augmentation on embeddings
        noise_std: float = 0.1,           # Std of Gaussian noise
        parcel_mapper: Optional[ParcelMapper] = None,  # NEW: Parcellation mapper
        apply_zscore: bool = False,       # NEW: Apply z-score to fMRI
        **kwargs, # Absorb unused arguments
    ):
        self.embeddings = embeddings
        self.augment_noise = augment_noise
        self.noise_std = noise_std
        self.parcel_mapper = parcel_mapper  # NEW
        self.apply_zscore = apply_zscore    # NEW

        # Storage
        self.fmri_list = []
        self.image_id_list = []  # Index into the split-specific embedding file
        self.subject_ids = []
        subjects = sorted(fmri_data.keys())
        for subj in subjects:
            fmri_s = fmri_data[subj]
            ids_s = image_ids[subj]
            # Luôn clip fMRI theo min/max toàn dataset
            fmri_s = clip_minmax(fmri_s)
            self.fmri_list.append(fmri_s)
            self.image_id_list.append(ids_s)
            self.subject_ids.extend([subj] * len(fmri_s))
        # Concatenate all
        if len(self.fmri_list) > 0:
            self.fmri_data = np.concatenate(self.fmri_list, axis=0).astype(np.float32)
            self.image_id_data = np.concatenate(self.image_id_list, axis=0).astype(np.int32)
            self.subject_ids = np.array(self.subject_ids)
        else:
            self.fmri_data = np.array([])
            self.image_id_data = np.array([])
            self.subject_ids = np.array([])

        # Apply parcellation if mapper is provided
        if self.parcel_mapper is not None:
            print(f"Applying parcellation: {self.fmri_data.shape[1]} voxels → {self.parcel_mapper.n_parcels} parcels")
            self.fmri_data = self.parcel_mapper.parcellate(self.fmri_data)
            print(f"Parcellated shape: {self.fmri_data.shape}")

        # Apply z-score if requested
        if self.apply_zscore:
            print(f"Applying z-score normalization...")
            self.fmri_data = ParcelMapper.zscore(self.fmri_data)
            print(f"Z-scored - Mean: {self.fmri_data.mean():.6f}, Std: {self.fmri_data.std():.6f}")

        aug_str = f"with noise augmentation (std={noise_std})" if augment_noise else "without augmentation"
        parcel_str = f", parcellated to {self.parcel_mapper.n_parcels} parcels" if parcel_mapper else ""
        zscore_str = ", z-scored" if apply_zscore else ""
        print(f"NeuroFluxDataset: {len(self.fmri_data)} samples, "
              f"{len(np.unique(self.image_id_data))} unique images, "
              f"embeddings shape: {self.embeddings.shape}, {aug_str}{parcel_str}{zscore_str}")
    
    def __len__(self):
        return len(self.fmri_data)
    
    def __getitem__(self, idx):
        fmri = self.fmri_data[idx]
        image_id_in_split = self.image_id_data[idx]
        subject_id = self.subject_ids[idx]

        # Get DINOv2 embedding from the split-specific file
        embedding = self.embeddings[image_id_in_split].astype(np.float32)

        # Apply noise augmentation to embedding during training (if enabled)
        if self.augment_noise:
            noise = np.random.normal(0, self.noise_std, embedding.shape).astype(np.float32)
            embedding = embedding + noise

        return {
            'fmri': torch.from_numpy(fmri).float(),
            'embedding': torch.from_numpy(embedding),
            'image_id': image_id_in_split, # This is now an index, not a global ID
            'subject_id': subject_id
        }


class NeuroFluxConcatDataset(ConcatDataset):
    """
    Custom ConcatDataset that exposes subject_ids for SubjectSampler.
    """
    def __init__(self, datasets):
        super().__init__(datasets)
        # Concatenate subject_ids from all datasets
        self.subject_ids = np.concatenate([ds.subject_ids for ds in datasets])


class SubjectSampler(Sampler):
    """
    Sampler that ensures each batch contains samples from only one subject.
    From SynBrain.
    """

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
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def load_neuroflux_data(
    datalist_path: str,
    fmri_path: str,
    embeddings_path: str,
    subjects: List[int] = [1],
    split: str = 'train',
    average_trials: bool = False,
    augment_noise: bool = False,      # Enable noise augmentation on embeddings
    noise_std: float = 0.1,           # Noise std for augmentation
    parcel_labels_path: Optional[str] = None,  # NEW: Path to parcellation labels
    apply_zscore: bool = False,       # NEW: Apply z-score to fMRI
) -> NeuroFluxDataset:
    """
    Load data for NeuroFlux-SDE from preprocessed MindEye2-style data.

    Args:
        datalist_path: Path to the datalist JSON file.
        fmri_path: Path to the fMRI HDF5 file.
        embeddings_path: Path to DINOv2 embeddings .npy file for the split.
        subjects: List of subject IDs to load.
        split: 'train' or 'test'.
        average_trials: Whether to average 3 trials per image.
        augment_noise: Enable noise augmentation on embeddings.
        noise_std: Noise std for augmentation.
        parcel_labels_path: Path to .npy file with parcellation labels (optional).
        apply_zscore: Apply z-score normalization to fMRI after parcellation.

    Returns:
        NeuroFluxDataset instance
    """
    import json
    import h5py

    # Load the datalist
    print(f"Loading datalist from {datalist_path}...")
    with open(datalist_path, 'r') as f:
        datalist = json.load(f)

    # Extract samples from the datalist
    samples = datalist['samples']

    # Load embeddings for the current split
    print(f"Loading {split} embeddings from {embeddings_path}...")
    split_embeddings = np.load(embeddings_path)

    # Load parcellation mapper if provided
    parcel_mapper = None
    if parcel_labels_path is not None:
        print(f"Loading parcellation from {parcel_labels_path}...")
        parcel_mapper = ParcelMapper.from_files(parcel_labels_path)

    # Load the full fMRI data
    print(f"Loading fMRI data from {fmri_path}...")
    with h5py.File(fmri_path, 'r') as hf:
        full_fmri_data = hf['betas'][:]

    fmri_data = {}
    image_ids = {}
    
    for subj in subjects:
        print(f"\nProcessing Subject {subj} for {split} split...")
        
        # Collect all samples for this split from all NSD images
        trial_indices = []
        image_indices_in_split = []
        
        # The datalist is organized by nsd_id
        # First, create a mapping from nsd_id to its index in the embeddings array
        nsd_id_to_embed_idx = {}
        embed_idx = 0
        
        for nsd_id_str, nsd_data in samples.items():
            if split in nsd_data['splits']:
                nsd_id_to_embed_idx[nsd_id_str] = embed_idx
                embed_idx += 1
        
        # Now collect all samples and map them to the correct embedding index
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

        if average_trials:
            print(f"  Averaging trials for {split} split...")
            unique_img_indices = np.unique(image_indices_in_split)
            avg_fmri = []
            final_img_indices = []
            
            for img_idx in unique_img_indices:
                # Find all trials for this specific image
                mask = (image_indices_in_split == img_idx)
                # Average the trials (shape [N, 15724] -> [15724])
                avg_fmri.append(np.mean(raw_fmri[mask], axis=0))
                final_img_indices.append(img_idx)
                
            fmri_data[subj] = np.stack(avg_fmri)
            image_ids[subj] = np.array(final_img_indices)
        else:
            fmri_data[subj] = raw_fmri
            image_ids[subj] = image_indices_in_split

        print(f"  {split.capitalize()} fMRI: {fmri_data[subj].shape}, {len(np.unique(image_ids[subj]))} unique images")

    # Create dataset with augmentation and parcellation options
    dataset = NeuroFluxDataset(
        fmri_data=fmri_data,
        embeddings=split_embeddings,
        image_ids=image_ids,
        augment_noise=augment_noise,
        noise_std=noise_std,
        parcel_mapper=parcel_mapper,
        apply_zscore=apply_zscore,
    )

    return dataset


def create_dataloaders(
    datalist_path: str,
    fmri_path: str,
    train_embeddings_path: str,
    test_embeddings_path: str,
    subjects: List[int] = [1],
    batch_size: int = 32,
    average_trials_train: bool = False,
    average_trials_val: bool = True,
    augment_noise_train: bool = False,    # Enable augmentation for training
    noise_std: float = 0.1,               # Augmentation noise std
    parcel_labels_path: Optional[str] = None,  # NEW: Parcellation labels
    apply_zscore_train: bool = False,     # NEW: Z-score for training
    apply_zscore_val: bool = False,       # NEW: Z-score for validation
) -> tuple:
    """
    Create train and validation dataloaders from preprocessed MindEye2-style data.

    Args:
        datalist_path: Path to datalist JSON
        fmri_path: Path to fMRI HDF5
        train_embeddings_path: Path to train embeddings
        test_embeddings_path: Path to test embeddings
        subjects: List of subject IDs
        batch_size: Batch size
        average_trials_train: Average trials for training
        average_trials_val: Average trials for validation
        augment_noise_train: Enable noise augmentation for training
        noise_std: Augmentation noise std
        parcel_labels_path: Path to parcellation labels (optional)
        apply_zscore_train: Apply z-score to training data
        apply_zscore_val: Apply z-score to validation data

    Returns:
        (train_loader, val_loader)
    """
    # Training data with augmentation and parcellation
    train_dataset = load_neuroflux_data(
        datalist_path=datalist_path,
        fmri_path=fmri_path,
        embeddings_path=train_embeddings_path,
        subjects=subjects,
        split='train',
        average_trials=average_trials_train,
        augment_noise=augment_noise_train,
        noise_std=noise_std,
        parcel_labels_path=parcel_labels_path,
        apply_zscore=apply_zscore_train,
    )


    

    # Validation data (no augmentation, but can have parcellation and z-score)
    val_dataset = load_neuroflux_data(
        datalist_path=datalist_path,
        fmri_path=fmri_path,
        embeddings_path=test_embeddings_path,
        subjects=subjects,
        split='test',
        average_trials=average_trials_val,
        augment_noise=False,  # Never augment validation data
        noise_std=0.0,
        parcel_labels_path=parcel_labels_path,
        apply_zscore=apply_zscore_val,
    )

    # Use custom concat dataset to preserve subject_ids
    # train_sampler = SubjectSampler(combined_dataset, batch_size=batch_size, drop_last=True)
    # train_loader = DataLoader(combined_dataset, batch_sampler=train_sampler, num_workers=4)

    train_sampler = SubjectSampler(train_dataset, batch_size=batch_size, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader


# Test script
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/media/hung/data1/codes/img2fmriii/NSD/data/nsd')
    parser.add_argument('--embeddings_path', type=str, 
                        default='/media/hung/data1/codes/img2fmriii/NSD/data/embeddings/dinov2_vitb14_embeddings.npy')
    parser.add_argument('--expdesign_path', type=str,
                        default='/media/hung/data1/codes/img2fmriii/NSD/data/nsddata/experiments/nsd/nsd_expdesign.mat')
    parser.add_argument('--subjects', type=str, default='[1]')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    import json
    subjects = json.loads(args.subjects)
    
    # Test loading
    print("=" * 60)
    print("Testing NeuroFlux Dataset with DINOv2")
    print("=" * 60)
    
    train_loader, val_loader = create_dataloaders(
        data_path=args.data_path,
        embeddings_path=args.embeddings_path,
        expdesign_path=args.expdesign_path,
        subjects=subjects,
        batch_size=args.batch_size,
        average_trials_train=False,
        average_trials_val=True
    )
    
    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Test one batch
    for batch in train_loader:
        print(f"\nSample batch:")
        print(f"  fMRI shape: {batch['fmri'].shape}")
        print(f"  Embedding shape: {batch['embedding'].shape}")
        print(f"  Image IDs: {batch['image_id'][:5]}...")
        print(f"  Subject IDs: {batch['subject_id'][:5]}...")
        break
    
    print("\n✓ Dataset loading test passed!")

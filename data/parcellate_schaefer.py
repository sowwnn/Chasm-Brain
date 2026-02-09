"""
Create a 1000-parcel parcellation using Schaefer atlas as a base.

Since the Schaefer-1000 atlas only covers cortical areas and the NSD mask 
might overlap with only a subset of these parcels (and include non-cortical voxels),
this script implements a hybrid approach:
1. Assign voxels to Schaefer parcels where they overlap.
2. For voxels not covered by Schaefer, use K-means to create the remaining 
   parcels to reach exactly 1000.

This ensures:
- Functional boundaries from Schaefer are respected.
- All 15724 voxels are included.
- Output dimensionality is exactly 1000.
"""

import os
import argparse
import numpy as np
import h5py
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import joblib

def load_schaefer_and_mask(atlas_path, mask_path):
    print(f"Loading Schaefer atlas from {atlas_path}...")
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    
    print(f"Loading NSD mapping from {mask_path}...")
    mapping = np.load(mask_path)
    mask = mapping['original_mask']
    
    # Extract atlas labels for voxels in mask
    # 15724 voxels
    labels = atlas_data[mask > 0]
    
    return labels, mapping['coords_3d_original']

def create_hybrid_parcellation(voxel_atlas_labels, coordinates, total_parcels=1000):
    """
    Create a hybrid parcellation: Schaefer parcels + K-means for the rest.
    """
    n_voxels = len(voxel_atlas_labels)
    print(f"Processing {n_voxels} voxels...")
    
    # 1. Identify which Schaefer parcels are represented
    unique_schaefer = np.unique(voxel_atlas_labels)
    unique_schaefer = unique_schaefer[unique_schaefer > 0] # Exclude background
    n_schaefer = len(unique_schaefer)
    
    print(f"Found {n_schaefer} Schaefer parcels in the mask.")
    
    # 2. Identify voxels not covered by Schaefer (label 0)
    uncovered_mask = (voxel_atlas_labels == 0)
    n_uncovered = np.sum(uncovered_mask)
    print(f"Found {n_uncovered} voxels not covered by Schaefer.")
    
    if n_uncovered == 0:
        print("All voxels are covered by Schaefer (unlikely for NSD).")
        # Just use Schaefer labels and map to 0..n_parcels-1
        final_labels = np.zeros(n_voxels, dtype=np.int32)
        for i, b_id in enumerate(unique_schaefer):
            final_labels[voxel_atlas_labels == b_id] = i
        return final_labels, n_schaefer

    # 3. Use K-means for the remaining parcels
    n_remaining_parcels = total_parcels - n_schaefer
    if n_remaining_parcels <= 0:
        print(f"Warning: Already have {n_schaefer} parcels, which is >= {total_parcels}.")
        print("Using only Schaefer parcels.")
        # Map to 0..n_schaefer-1
        final_labels = np.zeros(n_voxels, dtype=np.int32)
        for i, b_id in enumerate(unique_schaefer):
            final_labels[voxel_atlas_labels == b_id] = i
        # Assign uncovered voxels to a new "garbage" parcel if any
        if n_uncovered > 0:
            final_labels[uncovered_mask] = n_schaefer 
            return final_labels, n_schaefer + 1
        return final_labels, n_schaefer

    print(f"Creating {n_remaining_parcels} additional parcels using K-means on {n_uncovered} voxels...")
    
    # Clustering based on spatial coordinates for the uncovered voxels
    kmeans = MiniBatchKMeans(
        n_clusters=n_remaining_parcels,
        random_state=42,
        batch_size=1000,
        n_init=3
    )
    uncovered_coords = coordinates[uncovered_mask]
    kmeans_labels = kmeans.fit_predict(uncovered_coords)
    
    # 4. Merge labels
    # Label mapping:
    # 0 to n_schaefer-1: Schaefer parcels
    # n_schaefer to 999: K-means parcels
    
    final_labels = np.zeros(n_voxels, dtype=np.int32) - 1
    
    # Map Schaefer
    for i, b_id in enumerate(unique_schaefer):
        final_labels[voxel_atlas_labels == b_id] = i
        
    # Map K-means
    final_labels[uncovered_mask] = kmeans_labels + n_schaefer
    
    assert np.all(final_labels >= 0), "Some voxels were not assigned!"
    
    return final_labels, total_parcels

def apply_parcellation(input_path, labels, n_parcels, output_path):
    print(f"Applying parcellation to {input_path}...")
    with h5py.File(input_path, 'r') as f:
        fmri_data = f['betas'][:]
    
    n_samples, n_voxels = fmri_data.shape
    parcellated = np.zeros((n_samples, n_parcels), dtype=np.float32)
    
    for i in tqdm(range(n_parcels), desc="Aggregating"):
        mask = (labels == i)
        if np.any(mask):
            parcellated[:, i] = fmri_data[:, mask].mean(axis=1)
            
    print(f"Saving to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('betas', data=parcellated, compression='gzip', compression_opts=4)
        f.create_dataset('voxel_to_parcel', data=labels)
        f.attrs['n_parcels'] = n_parcels
        f.attrs['method'] = 'schaefer_hybrid'
        
    # Save labels separately
    labels_path = output_path.parent / f"{output_path.stem}_labels.npy"
    np.save(labels_path, labels)
    print(f"Labels saved to {labels_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input HDF5 file")
    parser.add_argument("--atlas", type=str, default="dataset/nsd/subj01/schaefer1000_nsd_space.nii.gz")
    parser.add_argument("--mask_info", type=str, default="data/fmri_3d_compact_mapping_subj01.npz")
    parser.add_argument("--n_parcels", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_schaefer{args.n_parcels}.hdf5"
    else:
        output_path = Path(args.output)
        
    voxel_atlas_labels, coords = load_schaefer_and_mask(args.atlas, args.mask_info)
    
    final_labels, n_actual = create_hybrid_parcellation(voxel_atlas_labels, coords, args.n_parcels)
    
    apply_parcellation(input_path, final_labels, n_actual, output_path)
    
    print("\nDone!")

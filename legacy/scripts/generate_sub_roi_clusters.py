
import numpy as np
import nibabel as nib
import os
from sklearn.cluster import KMeans
import argparse
import sys

# Default paths
DEFAULT_MAPPING_PATH = '/home/sowwn/Workspace/ws/2026/I2fMRI/data/fmri_3d_compact_mapping_subj01.npz'
DEFAULT_ROI_DIR = '/mnt/HDD/dataset/NSD_raw/nsddata/ppdata/subj01/func1pt8mm/roi/'
DEFAULT_OUTPUT_PATH = '/home/sowwn/Workspace/ws/2026/I2fMRI/data/sub_roi_labels.npy'

def load_roi_1d(fname, roi_dir, mask_3d):
    path = os.path.join(roi_dir, fname)
    if not os.path.exists(path):
        print(f"Warning: {fname} not found, skipping.")
        return None
    # Load nii volume, get data, apply mask
    vol = nib.load(path).get_fdata()
    return vol[mask_3d > 0] # Flatten according to mask

def generate_clusters(mapping_path, roi_dir, output_path, voxels_per_cluster=30):
    print(f"Loading mapping from {mapping_path}")
    mapping = np.load(mapping_path)
    mask_3d = mapping['original_mask']
    coords = mapping['coords_compact']
    n_voxels = len(coords)
    
    print(f"Total voxels: {n_voxels}")
    
    # Load ROIs
    print(f"Loading ROIs from {roi_dir}")
    prf = load_roi_1d('prf-visualrois.nii.gz', roi_dir, mask_3d)
    faces = load_roi_1d('floc-faces.nii.gz', roi_dir, mask_3d)
    places = load_roi_1d('floc-places.nii.gz', roi_dir, mask_3d)
    bodies = load_roi_1d('floc-bodies.nii.gz', roi_dir, mask_3d)
    words = load_roi_1d('floc-words.nii.gz', roi_dir, mask_3d)

    if prf is None: prf = np.zeros(n_voxels)
    if faces is None: faces = np.zeros(n_voxels)
    if places is None: places = np.zeros(n_voxels)
    if bodies is None: bodies = np.zeros(n_voxels)
    if words is None: words = np.zeros(n_voxels)
    
    # Define primary regions logic matches notebook
    # Priority: V1-V4 > Face/Place/Body/Word > Other
    
    # Init Masks
    idx_v1 = (prf == 1) | (prf == 2)
    idx_v2 = (prf == 3) | (prf == 4)
    idx_v3 = (prf == 5) | (prf == 6)
    idx_hv4 = (prf == 7)
    
    idx_face = (faces > 0)
    idx_place = (places > 0)
    idx_body = (bodies > 0)
    idx_word = (words > 0)
    
    # Resolve overlaps naturally by processing order or exclusion
    # We follow the notebook logic: iterate regions, exclude from 'Other', 
    # but also ensure a voxel belongs to exactly one main group for clustering.
    # Actually, notebook code allowed overlap in region_priors definition but handled it via 'idx_other'.
    # For sub-roi labels, we need strict partition. Every voxel must have exactly one sub-roi label.
    
    # Let's define a strict partition array first
    partition_map = np.zeros(n_voxels, dtype=int) 
    # 0: Other, 1: V1, 2: V2, 3: V3, 4: hV4, 5: Face, 6: Place, 7: Body, 8: Word
    
    # Apply in reverse priority so higher priority overwrites lower?
    # Or just use `if/elif`.
    # Let's say: V1-V4 are exclusive and highest priority?
    # Floc regions often overlap.
    # The notebook visualized them by simple overwrite.
    # Let's use specific priority: V1 > V2 > V3 > hV4 > Face > Place > Body > Word > Other
    
    # We can assign codes:
    partition_map[:] = 0 # Other
    partition_map[idx_word] = 8
    partition_map[idx_body] = 7
    partition_map[idx_place] = 6
    partition_map[idx_face] = 5
    partition_map[idx_hv4] = 4
    partition_map[idx_v3] = 3
    partition_map[idx_v2] = 2
    partition_map[idx_v1] = 1
    
    region_names = {0: 'Other', 1: 'V1', 2: 'V2', 3: 'V3', 4: 'hV4', 
                    5: 'Face', 6: 'Place', 7: 'Body', 8: 'Word'}
    
    sub_roi_labels = np.zeros(n_voxels, dtype=int)
    current_cluster_id = 0
    
    print("Clustering sub-rois...")
    print(f"{'Region':<10} {'Voxels':<10} {'Clusters':<10} {'StartID':<10}")
    
    for r_code in range(1, 9): # 1 to 8
        mask = (partition_map == r_code)
        n_pts = np.sum(mask)
        if n_pts == 0: continue
        
        k = max(1, int(n_pts / voxels_per_cluster))
        region_coords = coords[mask]
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        local_labels = kmeans.fit_predict(region_coords)
        
        global_labels = local_labels + current_cluster_id
        sub_roi_labels[mask] = global_labels
        
        print(f"{region_names[r_code]:<10} {n_pts:<10} {k:<10} {current_cluster_id:<10}")
        current_cluster_id += k
        
    # Handle 'Other' last (code 0)
    mask = (partition_map == 0)
    n_pts = np.sum(mask)
    if n_pts > 0:
        k = max(1, int(n_pts / voxels_per_cluster))
        region_coords = coords[mask]
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        local_labels = kmeans.fit_predict(region_coords)
        global_labels = local_labels + current_cluster_id
        sub_roi_labels[mask] = global_labels
        print(f"{region_names[0]:<10} {n_pts:<10} {k:<10} {current_cluster_id:<10}")
        current_cluster_id += k

    print("-" * 40)
    print(f"Total Clusters: {current_cluster_id}")
    
    # Save
    print(f"Saving labels to {output_path}")
    np.save(output_path, sub_roi_labels)
    print("Done.")

if __name__ == "__main__":
    generate_clusters(DEFAULT_MAPPING_PATH, DEFAULT_ROI_DIR, DEFAULT_OUTPUT_PATH)

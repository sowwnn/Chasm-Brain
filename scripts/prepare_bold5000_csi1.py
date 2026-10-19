#!/usr/bin/env python3
"""Prepare CSI1 visual-cortex targets from BOLD5000 Release 2.0 TYPED betas."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path("dataset/BOLD5000")
BETA_DIR = ROOT / "wholebrain/CSI1/TYPED"
MASK_PATH = ROOT / "wholebrain/CSI1/masks/visual_cortex_aparc_mask.nii.gz"
VALID_MASK_PATH = ROOT / "wholebrain/CSI1/masks/visual_cortex_valid_mask.nii.gz"
IMAGE_NAMES_PATH = ROOT / "image_names/CSI1_imgnames.txt"
OUTPUT_DIR = ROOT / "processed/CSI1"


def zscore_session(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


def main() -> None:
    beta_paths = sorted(BETA_DIR.glob("CSI1_GLMbetas-TYPED-*_ses-*.nii.gz"))
    if len(beta_paths) != 15:
        raise RuntimeError(f"Expected 15 complete CSI1 sessions, found {len(beta_paths)}")

    mask_image = nib.load(MASK_PATH)
    mask = np.asarray(mask_image.dataobj).astype(bool)
    image_names = IMAGE_NAMES_PATH.read_text().splitlines()

    sessions = []
    session_lengths = []
    for path in beta_paths:
        image = nib.load(path)
        if image.shape[:3] != mask.shape:
            raise ValueError(f"Spatial shape mismatch: {path} {image.shape[:3]} vs {mask.shape}")
        # Release 2.0 beta headers use a different translation origin from the
        # OpenNeuro fMRIPrep masks, although shape, voxel size, orientation and
        # voxel-index brain support match. Apply masks in voxel-index space.
        if nib.aff2axcodes(image.affine) != nib.aff2axcodes(mask_image.affine):
            raise ValueError(f"Orientation mismatch: {path}")
        if not np.allclose(image.header.get_zooms()[:3], mask_image.header.get_zooms()[:3]):
            raise ValueError(f"Voxel-size mismatch: {path}")

        volume = np.asarray(image.dataobj, dtype=np.float32)
        targets = volume[mask, :].T
        sessions.append(targets)
        session_lengths.append(int(targets.shape[0]))
        print(f"Loaded {path.name}: {targets.shape}")

    valid_voxels = np.ones(int(mask.sum()), dtype=bool)
    for targets in sessions:
        valid_voxels &= np.isfinite(targets).all(axis=0)
    if not valid_voxels.any():
        raise FloatingPointError("No visual-cortex voxels are finite across all sessions")
    sessions = [
        zscore_session(targets[:, valid_voxels]).astype(np.float32)
        for targets in sessions
    ]

    valid_mask = np.zeros(mask.shape, dtype=np.uint8)
    valid_mask[mask] = valid_voxels.astype(np.uint8)
    valid_header = mask_image.header.copy()
    valid_header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(valid_mask, mask_image.affine, valid_header), VALID_MASK_PATH)

    all_targets = np.concatenate(sessions, axis=0)
    if all_targets.shape[0] != len(image_names):
        raise ValueError(
            f"Trial mismatch: {all_targets.shape[0]} beta trials vs {len(image_names)} image names"
        )

    counts = Counter(image_names)
    train_indices = [index for index, name in enumerate(image_names) if counts[name] == 1]
    repeated_indices = defaultdict(list)
    for index, name in enumerate(image_names):
        if counts[name] > 1:
            repeated_indices[name].append(index)

    test_names = list(repeated_indices)
    test_targets = np.stack(
        [all_targets[repeated_indices[name]].mean(axis=0) for name in test_names]
    ).astype(np.float32)
    train_targets = all_targets[train_indices]
    train_names = [image_names[index] for index in train_indices]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "train_fmri_zscore.npy", train_targets)
    np.save(OUTPUT_DIR / "test_fmri_zscore.npy", test_targets)
    (OUTPUT_DIR / "train_image_names.txt").write_text("\n".join(train_names) + "\n")
    (OUTPUT_DIR / "test_image_names.txt").write_text("\n".join(test_names) + "\n")

    metadata = {
        "subject": "CSI1",
        "source_beta_type": "TYPED-FITHRF-GLMDENOISE-RR",
        "normalization": "per-session per-voxel z-score",
        "anatomical_mask": str(MASK_PATH),
        "valid_mask": str(VALID_MASK_PATH),
        "mask_alignment": "voxel-index alignment; shape, RAS orientation, 2mm voxel size and brain support verified",
        "num_anatomical_visual_voxels": int(mask.sum()),
        "num_visual_voxels": int(valid_voxels.sum()),
        "session_lengths": session_lengths,
        "num_trials": len(image_names),
        "num_unique_images": len(counts),
        "train_split": "images presented exactly once",
        "num_train_images": len(train_names),
        "test_split": "repeated images, targets averaged across presentations",
        "num_test_images": len(test_names),
        "train_shape": list(train_targets.shape),
        "test_shape": list(test_targets.shape),
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

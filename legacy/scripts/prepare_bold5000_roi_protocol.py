#!/usr/bin/env python3
"""Prepare BOLD5000 CSI1 functional-ROI targets for encoding benchmarks.

The output supports the leakage-free repeated-image protocol used by
Delavari et al. (Vision Research, 2025).  Single-presentation images form
the model-development pool; every individual response to a repeated image
is retained for final evaluation.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("dataset/BOLD5000")
ROI_DIR = ROOT / "BOLD5000_GLMsingle_ROI_betas/py"
PROCESSED_DIR = ROOT / "processed/CSI1"
IMAGE_NAMES_PATH = ROOT / "image_names/CSI1_imgnames.txt"
ROIS = ("PPA", "LOC", "EarlyVis", "OPA", "RSC")
HEMISPHERES = ("LH", "RH")


def session_zscore(values: np.ndarray, lengths: list[int]) -> np.ndarray:
    normalized = np.empty_like(values, dtype=np.float32)
    start = 0
    for length in lengths:
        stop = start + length
        session = values[start:stop].astype(np.float32)
        mean = session.mean(axis=0, keepdims=True)
        std = session.std(axis=0, keepdims=True)
        normalized[start:stop] = (session - mean) / np.maximum(std, 1e-6)
        start = stop
    if start != len(values):
        raise ValueError(f"Session lengths cover {start} rows, expected {len(values)}")
    return normalized


def main() -> None:
    base_metadata = json.loads((PROCESSED_DIR / "metadata.json").read_text())
    session_lengths = base_metadata["session_lengths"]
    image_names = IMAGE_NAMES_PATH.read_text().splitlines()
    counts = Counter(image_names)

    roi_arrays = []
    boundaries = {}
    offset = 0
    for roi in ROIS:
        hemispheres = []
        hemisphere_sizes = {}
        for hemisphere in HEMISPHERES:
            path = ROI_DIR / (
                f"CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_allses_{hemisphere}{roi}.npy"
            )
            values = np.load(path)
            if values.shape[0] != len(image_names):
                raise ValueError(f"Trial mismatch for {path}: {values.shape[0]}")
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Non-finite values in {path}")
            hemispheres.append(values)
            hemisphere_sizes[hemisphere] = int(values.shape[1])
        roi_values = session_zscore(np.concatenate(hemispheres, axis=1), session_lengths)
        roi_arrays.append(roi_values)
        boundaries[roi] = {
            "start": offset,
            "stop": offset + int(roi_values.shape[1]),
            "num_voxels": int(roi_values.shape[1]),
            "hemisphere_sizes": hemisphere_sizes,
        }
        offset += int(roi_values.shape[1])

    all_trials = np.concatenate(roi_arrays, axis=1).astype(np.float32)
    single_indices = np.array(
        [i for i, name in enumerate(image_names) if counts[name] == 1], dtype=np.int64
    )
    repeated_indices = np.array(
        [i for i, name in enumerate(image_names) if counts[name] > 1], dtype=np.int64
    )
    train_names = (PROCESSED_DIR / "train_image_names.txt").read_text().splitlines()
    test_names = (PROCESSED_DIR / "test_image_names.txt").read_text().splitlines()
    if [image_names[i] for i in single_indices] != train_names:
        raise ValueError("Single-presentation order does not match the existing training embeddings")
    test_name_to_index = {name: i for i, name in enumerate(test_names)}
    repeated_embedding_indices = np.array(
        [test_name_to_index[image_names[i]] for i in repeated_indices], dtype=np.int64
    )

    np.save(PROCESSED_DIR / "roi_all_trials_zscore.npy", all_trials)
    np.save(PROCESSED_DIR / "roi_train_unique_zscore.npy", all_trials[single_indices])
    np.save(PROCESSED_DIR / "roi_test_repetitions_zscore.npy", all_trials[repeated_indices])
    np.save(PROCESSED_DIR / "roi_single_trial_indices.npy", single_indices)
    np.save(PROCESSED_DIR / "roi_repeated_trial_indices.npy", repeated_indices)
    np.save(PROCESSED_DIR / "roi_test_repetition_embedding_indices.npy", repeated_embedding_indices)

    metadata = {
        "subject": "CSI1",
        "source": "BOLD5000 Release 2.0 TYPED-FITHRF-GLMDENOISE-RR ROI beta arrays",
        "normalization": "per-session, per-voxel z-score",
        "roi_order": list(ROIS),
        "roi_boundaries": boundaries,
        "important_overlap_note": (
            "Functional ROI arrays are concatenated for joint training. Some anatomical voxels may "
            "occur in more than one public ROI array; report each ROI separately for paper comparison."
        ),
        "num_trials": len(image_names),
        "num_unique_images": len(counts),
        "num_single_presentation_images": int(len(single_indices)),
        "num_repeated_unique_images": len(test_names),
        "num_repeated_presentations": int(len(repeated_indices)),
        "train_shape": list(all_trials[single_indices].shape),
        "test_repetitions_shape": list(all_trials[repeated_indices].shape),
        "protocol": (
            "Leakage-free repeated-image evaluation: develop on images shown once; predict each of "
            "113 repeated images once and correlate against every held-out measured repetition."
        ),
    }
    (PROCESSED_DIR / "roi_protocol_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

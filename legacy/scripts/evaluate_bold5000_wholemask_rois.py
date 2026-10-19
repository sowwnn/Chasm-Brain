#!/usr/bin/env python3
"""Evaluate a 13,197-voxel CHASMBrain checkpoint on mapped functional ROIs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_bold5000_roi_protocol import bootstrap_interval, row_pearson
from scripts.evaluate_guidance_sensitivity import build_model


class UniqueEmbeddingDataset(Dataset):
    def __init__(self, path, layer):
        raw = np.load(path, mmap_mode="r")
        self.embeddings = raw[:, layer] if raw.ndim == 4 else raw

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        embedding = self.embeddings[index].astype(np.float32)
        return torch.from_numpy(embedding[0]), torch.from_numpy(embedding[1:])


def map_roi_columns(whole_targets, roi_targets):
    """Recover exact column identities from stable response fingerprints."""
    fingerprint_rows = np.array([0, 1, 7, 31, 127, 509, 1021, 2047, 4095, 4802])
    whole_fingerprints = np.asarray(whole_targets[fingerprint_rows], dtype=np.float32).T
    roi_fingerprints = np.asarray(roi_targets[fingerprint_rows], dtype=np.float32).T
    lookup = {}
    for index, values in enumerate(whole_fingerprints):
        lookup.setdefault(tuple(np.round(values, 3)), []).append(index)

    mapping = np.full(len(roi_fingerprints), -1, dtype=np.int64)
    for roi_index, values in enumerate(roi_fingerprints):
        candidates = lookup.get(tuple(np.round(values, 3)), [])
        if len(candidates) == 1:
            mapping[roi_index] = candidates[0]

    # Verify every recovered identity across all 4,803 development images.
    mapped_roi = np.flatnonzero(mapping >= 0)
    max_errors = np.array([
        np.max(np.abs(
            np.asarray(whole_targets[:, mapping[index]], dtype=np.float32)
            - np.asarray(roi_targets[:, index], dtype=np.float32)
        ))
        for index in mapped_roi
    ])
    if len(max_errors) == 0 or max_errors.max() > 2e-5:
        raise ValueError(f"Functional-ROI mapping verification failed: {max_errors.max()}")
    return mapping, float(max_errors.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_hierarchical_v3_bold5000_csi1.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--roi_metadata", default="dataset/BOLD5000/processed/CSI1/roi_protocol_metadata.json")
    parser.add_argument("--output", default="results/wacv_round2/bold5000_csi1_wholemask_to_rois_seed42.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    data_cfg = config["data"]
    processed = Path(data_cfg["huwng_base_path"])
    roi_metadata = json.loads(Path(args.roi_metadata).read_text())
    mapping, max_mapping_error = map_roi_columns(
        np.load(processed / "train_fmri_zscore.npy", mmap_mode="r"),
        np.load(processed / "roi_train_unique_zscore.npy", mmap_mode="r"),
    )

    dataset = UniqueEmbeddingDataset(
        data_cfg["evaluation_embeddings_path"], data_cfg.get("dinov2_layer", -1)
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    unique_predictions = []
    with torch.inference_mode():
        for cls_token, patch_tokens in tqdm(loader, desc="13,197-voxel inference"):
            cls_token = cls_token.to(device, non_blocking=True)
            patch_tokens = patch_tokens.to(device, non_blocking=True)
            guidance, _, _ = model(cls_token, patch_tokens, stage=1)
            _, prediction, _ = model(cls_token, patch_tokens, stage=2, gt_roi_means=guidance)
            unique_predictions.append(prediction.cpu().numpy())
    unique_predictions = np.concatenate(unique_predictions)

    repetition_embedding_indices = np.load(processed / "roi_test_repetition_embedding_indices.npy")
    repeated_predictions = unique_predictions[repetition_embedding_indices]
    repeated_targets = np.load(processed / "roi_test_repetitions_zscore.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    regions = {}
    region_vectors = []
    mapped_predictions = []
    mapped_targets = []
    for roi in roi_metadata["roi_order"]:
        boundary = roi_metadata["roi_boundaries"][roi]
        roi_columns = np.arange(boundary["start"], boundary["stop"])
        valid_roi_columns = roi_columns[mapping[roi_columns] >= 0]
        whole_columns = mapping[valid_roi_columns]
        prediction = repeated_predictions[:, whole_columns]
        target = np.asarray(repeated_targets[:, valid_roi_columns], dtype=np.float32)
        values = row_pearson(prediction, target)
        region_vectors.append(values)
        mapped_predictions.append(prediction)
        mapped_targets.append(target)
        regions[roi] = {
            "pearson_mean": float(values.mean()),
            "pearson_sd": float(values.std(ddof=1)),
            "pearson_95_ci": bootstrap_interval(values, rng, args.resamples),
            "mapped_voxels": int(len(valid_roi_columns)),
            "public_roi_voxels": int(len(roi_columns)),
            "coverage": float(len(valid_roi_columns) / len(roi_columns)),
        }

    macro = np.stack(region_vectors).mean(axis=0)
    concatenated_prediction = np.concatenate(mapped_predictions, axis=1)
    concatenated_target = np.concatenate(mapped_targets, axis=1)
    concatenated = row_pearson(concatenated_prediction, concatenated_target)
    result = {
        "dataset": "BOLD5000",
        "subject": "CSI1",
        "model_output_shape": ["B", int(config["model"]["fmri_dim"])],
        "evaluation": "functional ROI intersection with the trained 13,197-voxel anatomical visual mask",
        "checkpoint": args.checkpoint,
        "num_unique_test_images": len(dataset),
        "num_test_presentations": int(len(repetition_embedding_indices)),
        "mapped_entries": int((mapping >= 0).sum()),
        "mapped_unique_wholemask_voxels": int(len(np.unique(mapping[mapping >= 0]))),
        "public_roi_entries": int(len(mapping)),
        "mapping_max_absolute_error": max_mapping_error,
        "regions": regions,
        "macro_mean_over_five_rois": float(macro.mean()),
        "macro_pearson_95_ci": bootstrap_interval(macro, rng, args.resamples),
        "concatenated_mapped_entries_pearson": float(concatenated.mean()),
        "comparability_note": (
            "This preserves CHASMBrain's full 13,197-voxel two-stage output. ROI scores use only "
            "functional-ROI voxels present in that fixed anatomical mask, so voxel counts differ "
            "from published full-ROI arrays."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        output.with_suffix(".npz"), mapping=mapping, macro_pearson=macro,
        concatenated_pearson=concatenated,
        **{f"pearson_{roi}": values for roi, values in zip(roi_metadata["roi_order"], region_vectors)},
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

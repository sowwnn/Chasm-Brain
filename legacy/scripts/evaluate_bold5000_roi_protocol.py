#!/usr/bin/env python3
"""Evaluate CHASMBrain on BOLD5000's repeated-image functional-ROI protocol."""

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

from scripts.evaluate_guidance_sensitivity import build_model


class RepeatedROIDataset(Dataset):
    def __init__(self, embeddings_path, targets_path, embedding_indices_path, layer):
        raw = np.load(embeddings_path, mmap_mode="r")
        self.embeddings = raw[:, layer] if raw.ndim == 4 else raw
        self.targets = np.load(targets_path, mmap_mode="r")
        self.embedding_indices = np.load(embedding_indices_path)
        if len(self.targets) != len(self.embedding_indices):
            raise ValueError("Target and embedding-index lengths differ")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        embedding = self.embeddings[self.embedding_indices[index]].astype(np.float32)
        return (
            torch.from_numpy(embedding[0]),
            torch.from_numpy(embedding[1:]),
            torch.from_numpy(self.targets[index].astype(np.float32)),
        )


def row_pearson(prediction, target):
    prediction = prediction - prediction.mean(axis=1, keepdims=True)
    target = target - target.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
    return (prediction * target).sum(axis=1) / np.maximum(denominator, 1e-12)


def bootstrap_interval(values, rng, resamples):
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    return [float(x) for x in np.quantile(values[indices].mean(axis=1), [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_hierarchical_v3_bold5000_csi1_roi.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/wacv_round2/bold5000_csi1_roi_repeated_seed42.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    data_cfg = config["data"]
    metadata = json.loads(Path(data_cfg["roi_metadata_path"]).read_text())
    dataset = RepeatedROIDataset(
        data_cfg["evaluation_embeddings_path"], data_cfg["evaluation_fmri_path"],
        data_cfg["evaluation_embedding_indices_path"], data_cfg.get("dinov2_layer", -1),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for cls_token, patch_tokens, target in tqdm(loader, desc="BOLD5000 ROI evaluation"):
            cls_token = cls_token.to(device, non_blocking=True)
            patch_tokens = patch_tokens.to(device, non_blocking=True)
            guidance, _, _ = model(cls_token, patch_tokens, stage=1)
            _, prediction, _ = model(cls_token, patch_tokens, stage=2, gt_roi_means=guidance)
            predictions.append(prediction.cpu().numpy())
            targets.append(target.numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    rng = np.random.default_rng(args.seed)
    concatenated_pearson = row_pearson(predictions, targets)
    concatenated_mse = np.square(predictions - targets).mean(axis=1)
    roi_results = {}
    roi_vectors = {}
    for roi in metadata["roi_order"]:
        boundary = metadata["roi_boundaries"][roi]
        values = row_pearson(
            predictions[:, boundary["start"]:boundary["stop"]],
            targets[:, boundary["start"]:boundary["stop"]],
        )
        roi_vectors[roi] = values
        roi_results[roi] = {
            "pearson_mean": float(values.mean()),
            "pearson_sd": float(values.std(ddof=1)),
            "pearson_95_ci": bootstrap_interval(values, rng, args.resamples),
            "num_voxels": boundary["num_voxels"],
        }

    macro_per_presentation = np.stack(list(roi_vectors.values())).mean(axis=0)
    result = {
        "dataset": "BOLD5000",
        "subject": "CSI1",
        "protocol": metadata["protocol"],
        "metric": "Pearson across voxels per presentation, then averaged",
        "checkpoint": args.checkpoint,
        "num_repeated_unique_images": metadata["num_repeated_unique_images"],
        "num_test_presentations": len(dataset),
        "concatenated_roi_array": {
            "pearson_mean": float(concatenated_pearson.mean()),
            "pearson_sd": float(concatenated_pearson.std(ddof=1)),
            "pearson_95_ci": bootstrap_interval(concatenated_pearson, rng, args.resamples),
            "mse_mean": float(concatenated_mse.mean()),
        },
        "regions": roi_results,
        "macro_mean_over_five_rois": float(macro_per_presentation.mean()),
        "macro_pearson_95_ci": bootstrap_interval(macro_per_presentation, rng, args.resamples),
        "comparability_note": (
            "ROI-wise values match the Delavari et al. metric and repeated-image split. The macro "
            "average is not their deduplicated all-ROI union because public ROI arrays can overlap."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        output.with_suffix(".npz"), predictions=predictions, targets=targets,
        predicted_pearson=concatenated_pearson, predicted_mse=concatenated_mse,
        macro_pearson=macro_per_presentation,
        **{f"pearson_{roi}": values for roi, values in roi_vectors.items()},
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

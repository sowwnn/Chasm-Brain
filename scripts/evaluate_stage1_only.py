"""Stage-1-only readout: broadcast predicted cluster values back to voxels.

This measures what the coarse stage alone achieves at voxel level, with no
learned refinement, so Stage 2's contribution can be read off directly.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.dataset_v7_h import load_neuroflux_data_v7h
from scripts.evaluate_guidance_sensitivity import build_model, sample_metrics


def upsample(clusters, num_voxels, voxels_per_cluster):
    out = clusters.repeat_interleave(voxels_per_cluster, dim=1)
    remainder = num_voxels - out.shape[1]
    if remainder > 0:
        fill = clusters.mean(dim=1, keepdim=True).expand(-1, remainder)
        out = torch.cat([out, fill], dim=1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_hierarchical_v3_huwng.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/wacv_round2/stage1_only_subj7_seed42.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    model_cfg = config["model"]
    data_cfg = config["data"]
    voxels_per_cluster = model_cfg["voxels_per_cluster"]

    dataset = load_neuroflux_data_v7h(
        huwng_base_path=data_cfg["huwng_base_path"],
        embeddings_path=data_cfg.get("test_embeddings_path"),
        subjects=[data_cfg["subject"]], split="test", average_trials=True,
        voxels_per_cluster=voxels_per_cluster,
        roi_top_k_percent=data_cfg.get("roi_top_k_percent", 0.7),
        augment_noise=False, dinov2_layer=data_cfg.get("dinov2_layer", -1),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {args.checkpoint} | missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()

    arrays = {
        "stage1_only_predicted": {"pearson": [], "mse": []},
        "stage1_only_oracle": {"pearson": [], "mse": []},
        "cluster_level": {"pearson": [], "mse": []},
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Stage-1 only"):
            cls_token = batch["cls_token"].to(device)
            patch_tokens = batch["patch_tokens"].to(device)
            target = batch["fmri"].to(device)
            gt_clusters = batch["roi_means"].to(device)

            pred_clusters, _, _ = model.stage1(cls_token, patch_tokens)
            num_voxels = target.shape[1]

            for name, clusters in [
                ("stage1_only_predicted", pred_clusters),
                ("stage1_only_oracle", gt_clusters),
            ]:
                prediction = upsample(clusters, num_voxels, voxels_per_cluster)
                pearson, mse = sample_metrics(prediction, target)
                arrays[name]["pearson"].append(pearson)
                arrays[name]["mse"].append(mse)

            pearson, mse = sample_metrics(pred_clusters, gt_clusters)
            arrays["cluster_level"]["pearson"].append(pearson)
            arrays["cluster_level"]["mse"].append(mse)

    rng = np.random.default_rng(args.seed)
    result = {
        "subject": data_cfg["subject"],
        "checkpoint": args.checkpoint,
        "num_images": len(dataset),
        "readout": "predicted cluster values repeated over their 30 voxels; "
                   "remainder voxels take the per-image cluster mean",
    }
    for name, values in arrays.items():
        pearson = np.concatenate(values["pearson"]).astype(np.float64)
        mse = np.concatenate(values["mse"]).astype(np.float64)
        idx = rng.integers(0, len(pearson), size=(args.resamples, len(pearson)))
        ci = [float(x) for x in np.quantile(pearson[idx].mean(axis=1), [0.025, 0.975])]
        result[name] = {
            "pearson_mean": float(pearson.mean()),
            "pearson_95_ci": ci,
            "mse_mean": float(mse.mean()),
        }
        print(f"{name:<24} Pearson {pearson.mean():.4f} {ci}  MSE {mse.mean():.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nSaved -> {output}")


if __name__ == "__main__":
    main()

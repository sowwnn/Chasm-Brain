"""Evaluate the direct one-stage baseline against CHASMBrain on identical samples."""

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
from model.direct_stage2 import DirectStage2Only
from scripts.evaluate_guidance_sensitivity import build_model, paired_tests, sample_metrics


def confidence_interval(values, rng, resamples):
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    return [float(x) for x in np.quantile(values[indices].mean(axis=1), [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_config", default="configs/train_hierarchical_v3_huwng.yaml")
    parser.add_argument("--direct_config", default="configs/train_direct_stage2_subj7.yaml")
    parser.add_argument("--full_checkpoint", required=True)
    parser.add_argument("--direct_checkpoint", required=True)
    parser.add_argument("--output", default="results/wacv_round2/direct_vs_coarse_to_fine_subj7.json")
    parser.add_argument("--per_image_output", default="results/wacv_round2/direct_vs_coarse_to_fine_subj7_per_image.npz")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.full_config, "r") as handle:
        full_config = yaml.safe_load(handle)
    with open(args.direct_config, "r") as handle:
        direct_config = yaml.safe_load(handle)

    full_cfg = full_config["model"]
    data_cfg = full_config["data"]
    direct_cfg = direct_config["model"]
    dataset = load_neuroflux_data_v7h(
        huwng_base_path=data_cfg["huwng_base_path"],
        embeddings_path=data_cfg.get("test_embeddings_path"),
        subjects=[data_cfg["subject"]], split="test", average_trials=True,
        voxels_per_cluster=full_cfg["voxels_per_cluster"],
        roi_top_k_percent=data_cfg.get("roi_top_k_percent", 0.7),
        augment_noise=False, dinov2_layer=data_cfg.get("dinov2_layer", -1),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_model = build_model(full_config)
    full_model.load_state_dict(torch.load(args.full_checkpoint, map_location="cpu", weights_only=True))
    full_model = full_model.to(device).eval()

    direct_model = DirectStage2Only(
        cls_dim=direct_cfg["cls_dim"], patch_dim=direct_cfg["patch_dim"],
        fmri_dim=direct_cfg["fmri_dim"],
        voxels_per_cluster=direct_cfg["voxels_per_cluster"],
        hidden_dim=direct_cfg["hidden_dim"], embed_dim=direct_cfg["embed_dim"],
        latent_dim=direct_cfg.get("latent_dim", 128), depth=direct_cfg["depth"],
        num_global_queries=direct_cfg["num_global_queries"],
        dropout=direct_cfg.get("dropout", 0.1), vae_mode=direct_cfg.get("vae_mode", "stochastic"),
        backbone=direct_cfg.get("backbone", "mamba"),
    )
    direct_model.load_state_dict(torch.load(args.direct_checkpoint, map_location="cpu", weights_only=True))
    direct_model = direct_model.to(device).eval()

    arrays = {
        "coarse_to_fine": {"pearson": [], "mse": []},
        "direct_one_stage": {"pearson": [], "mse": []},
    }
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Direct vs coarse-to-fine"):
            cls_token = batch["cls_token"].to(device, non_blocking=True)
            patch_tokens = batch["patch_tokens"].to(device, non_blocking=True)
            target = batch["fmri"].to(device, non_blocking=True)
            _, full_prediction, _ = full_model(cls_token, patch_tokens, stage=2)
            direct_prediction, _ = direct_model(cls_token, patch_tokens)
            for name, prediction in (
                ("coarse_to_fine", full_prediction),
                ("direct_one_stage", direct_prediction),
            ):
                pearson, mse = sample_metrics(prediction, target)
                arrays[name]["pearson"].append(pearson)
                arrays[name]["mse"].append(mse)

    arrays = {
        name: {metric: np.concatenate(chunks) for metric, chunks in metrics.items()}
        for name, metrics in arrays.items()
    }
    rng = np.random.default_rng(args.seed)
    full_pearson = arrays["coarse_to_fine"]["pearson"]
    direct_pearson = arrays["direct_one_stage"]["pearson"]
    result = {
        "subject": data_cfg["subject"],
        "num_images": len(dataset),
        "protocol": "identical NSD split, DINOv2 layer, trial averaging, voxel mask, and metrics",
        "checkpoints": {
            "coarse_to_fine": args.full_checkpoint,
            "direct_one_stage": args.direct_checkpoint,
        },
        "parameters": {
            "coarse_to_fine": sum(p.numel() for p in full_model.parameters()),
            "direct_one_stage": sum(p.numel() for p in direct_model.parameters()),
        },
        "coarse_to_fine": {
            "pearson_mean": float(full_pearson.mean()),
            "pearson_95_ci": confidence_interval(full_pearson, rng, args.resamples),
            "mse_mean": float(arrays["coarse_to_fine"]["mse"].mean()),
        },
        "direct_one_stage": {
            "pearson_mean": float(direct_pearson.mean()),
            "pearson_95_ci": confidence_interval(direct_pearson, rng, args.resamples),
            "mse_mean": float(arrays["direct_one_stage"]["mse"].mean()),
        },
        "direct_minus_coarse_to_fine": paired_tests(
            full_pearson, direct_pearson, rng, args.resamples
        ),
    }
    result["relative_pearson_drop_percent"] = float(
        100 * (full_pearson.mean() - direct_pearson.mean()) / full_pearson.mean()
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    per_image_output = Path(args.per_image_output)
    per_image_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        per_image_output,
        coarse_to_fine_pearson=full_pearson,
        coarse_to_fine_mse=arrays["coarse_to_fine"]["mse"],
        direct_one_stage_pearson=direct_pearson,
        direct_one_stage_mse=arrays["direct_one_stage"]["mse"],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

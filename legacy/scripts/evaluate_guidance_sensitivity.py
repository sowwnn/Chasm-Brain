"""Inference-only cluster-guidance sensitivity with paired uncertainty tests.

This is a fast diagnostic on one trained checkpoint. It does not replace retraining
each cluster-conditioning ablation from the same Stage 1 initialization.
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
from model.hierarchical_v3_1 import HierarchicalARM_V3
from model.hierarchical_v3_mlp import HierarchicalARM_V3_MLP
from model.hierarchical_v3_transformer import HierarchicalARM_V3_Transformer
from model.tben_official import HierarchicalARM_TBEn_Both, HierarchicalARM_V3_TBEn


def build_model(config):
    cfg = config["model"]
    backbone = config.get("experiment", {}).get("backbone", "mamba")
    model_classes = {
        "mamba": HierarchicalARM_V3,
        "mlp": HierarchicalARM_V3_MLP,
        "transformer": HierarchicalARM_V3_Transformer,
        "tben": HierarchicalARM_V3_TBEn,
        "tben_both": HierarchicalARM_TBEn_Both,
    }
    model_class = model_classes[backbone]
    tben_kwargs = {
        "tben_hidden_dim": cfg.get("tben_hidden_dim", 768),
        "tben_nheads": cfg.get("tben_nheads", 16),
        "tben_dim_feedforward": cfg.get("tben_dim_feedforward", 1024),
        "tben_enc_layers": cfg.get("tben_enc_layers", 0),
        "tben_dec_layers": cfg.get("tben_dec_layers", 1),
        "tben_pre_norm": cfg.get("tben_pre_norm", True),
        "tben_enc_output_layer": cfg.get("tben_enc_output_layer", 1),
    } if backbone in ("tben", "tben_both") else {}
    return model_class(
        cls_dim=cfg["cls_dim"], patch_dim=cfg["patch_dim"],
        num_patches=cfg["num_patches"], fmri_dim=cfg["fmri_dim"],
        voxels_per_cluster=cfg["voxels_per_cluster"],
        hidden_dim_1=cfg["hidden_dim_1"], depth_what=cfg["depth_what"],
        depth_where=cfg["depth_where"], contrastive_dim=cfg["contrastive_dim"],
        embed_dim_2=cfg["embed_dim_2"], latent_dim=cfg.get("latent_dim", 128),
        depth_2=cfg["depth_2"], num_global_queries=cfg["num_global_queries"],
        dropout=cfg.get("dropout", 0.1),
        vae_mode=config.get("experiment", {}).get("vae_mode", "stochastic"),
        **tben_kwargs,
    )


def sample_metrics(prediction, target):
    pred_centered = prediction - prediction.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = pred_centered.square().sum(dim=1).sqrt() * target_centered.square().sum(dim=1).sqrt()
    pearson = numerator / denominator.clamp_min(1e-12)
    mse = (prediction - target).square().mean(dim=1)
    return pearson.cpu().numpy(), mse.cpu().numpy()


def paired_tests(reference, variant, rng, samples):
    differences = variant - reference
    bootstrap_indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrap_means = differences[bootstrap_indices].mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, len(differences)))
    null_means = (signs * differences).mean(axis=1)
    observed = abs(differences.mean())
    return {
        "mean_delta": float(differences.mean()),
        "bootstrap_95_ci": [float(x) for x in np.quantile(bootstrap_means, [0.025, 0.975])],
        "paired_permutation_p": float((np.count_nonzero(np.abs(null_means) >= observed) + 1) / (samples + 1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_hierarchical_v3_huwng.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/wacv_round2/guidance_sensitivity_subj7.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    cfg, data_cfg = config["model"], config["data"]

    dataset = load_neuroflux_data_v7h(
        huwng_base_path=data_cfg["huwng_base_path"],
        embeddings_path=data_cfg.get("evaluation_embeddings_path", data_cfg.get("test_embeddings_path")),
        subjects=[data_cfg["subject"]], split="test", average_trials=True,
        voxels_per_cluster=cfg["voxels_per_cluster"],
        roi_top_k_percent=data_cfg.get("roi_top_k_percent", 0.7),
        augment_noise=False, dinov2_layer=data_cfg.get("dinov2_layer", -1),
        fmri_path=data_cfg.get("evaluation_fmri_path", data_cfg.get("test_fmri_path")),
        sample_indices_path=data_cfg.get("evaluation_sample_indices_path"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    modes = ("predicted", "none", "shuffle", "oracle")
    metrics = {mode: {"pearson": [], "mse": []} for mode in modes}
    generator = torch.Generator(device=device).manual_seed(args.seed)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Guidance sensitivity"):
            cls_token = batch["cls_token"].to(device, non_blocking=True)
            patch_tokens = batch["patch_tokens"].to(device, non_blocking=True)
            target = batch["fmri"].to(device, non_blocking=True)
            oracle = batch["roi_means"].to(device, non_blocking=True)
            predicted, _, _ = model.stage1(cls_token, patch_tokens)
            permutation = torch.randperm(predicted.shape[0], generator=generator, device=device)
            guidance_by_mode = {
                "predicted": predicted,
                "none": torch.zeros_like(predicted),
                "shuffle": predicted[permutation],
                "oracle": oracle,
            }
            for mode, guidance in guidance_by_mode.items():
                _, prediction, _ = model(cls_token, patch_tokens, stage=2, gt_roi_means=guidance)
                pearson, mse = sample_metrics(prediction, target)
                metrics[mode]["pearson"].append(pearson)
                metrics[mode]["mse"].append(mse)

    arrays = {
        mode: {metric: np.concatenate(values) for metric, values in mode_values.items()}
        for mode, mode_values in metrics.items()
    }
    rng = np.random.default_rng(args.seed)
    reference = arrays["predicted"]["pearson"]
    result = {
        "diagnostic": "inference-only; retrain variants for the primary ablation table",
        "subject": data_cfg["subject"],
        "num_images": len(dataset),
        "checkpoint": args.checkpoint,
        "modes": {},
    }
    for mode in modes:
        result["modes"][mode] = {
            "pearson_mean": float(arrays[mode]["pearson"].mean()),
            "pearson_95_ci": [
                float(x) for x in np.quantile(
                    arrays[mode]["pearson"][rng.integers(0, len(dataset), size=(args.resamples, len(dataset)))].mean(axis=1),
                    [0.025, 0.975],
                )
            ],
            "mse_mean": float(arrays[mode]["mse"].mean()),
        }
        if mode != "predicted":
            result["modes"][mode]["paired_vs_predicted"] = paired_tests(
                reference, arrays[mode]["pearson"], rng, args.resamples
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        output.with_suffix(".npz"),
        **{
            f"{mode}_{metric}": values
            for mode, mode_values in arrays.items()
            for metric, values in mode_values.items()
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

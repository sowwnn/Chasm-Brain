"""Compare CHASMBrain against the TBEn readout controls on identical samples.

Every model is scored on the same test batches with the same metric, so the
rows differ only in architecture:

  chasmbrain       two-stage, dual-stream Mamba Stage 1 + Stage 2
  tben_one_stage   structured readout, one stage (Variant A)
  tben_two_stage   same readout as Stage 1 + our unchanged Stage 2 (Variant B)
  direct_one_stage unstructured one-stage baseline, for reference

All paired tests are reported against CHASMBrain, plus the A-vs-B contrast that
isolates the refinement stage under a fixed readout mechanism.
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
from model.direct_stage2 import DirectStage2Only
from model.tben_official import TBEnOneStageOfficial
from scripts.evaluate_guidance_sensitivity import build_model, paired_tests, sample_metrics


def confidence_interval(values, rng, resamples):
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    return [float(x) for x in np.quantile(values[indices].mean(axis=1), [0.025, 0.975])]


def load_yaml(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chasmbrain_config", default="configs/train_hierarchical_v3_huwng.yaml")
    parser.add_argument("--chasmbrain_checkpoint", default="checkpoints/chasm_rerun_s7/best_model.pth")
    parser.add_argument("--tben_one_stage_config", default="configs/train_tben_one_stage_subj7.yaml")
    parser.add_argument("--tben_one_stage_checkpoint", required=True)
    parser.add_argument("--tben_two_stage_config", default="configs/train_tben_two_stage_subj7.yaml")
    parser.add_argument("--tben_two_stage_checkpoint", required=True)
    parser.add_argument("--direct_config", default="configs/train_direct_stage2_transformer_subj7.yaml")
    parser.add_argument("--direct_checkpoint")
    parser.add_argument("--output", default="results/wacv_round2/tben_comparison_subj7.json")
    parser.add_argument("--per_image_output", default="results/wacv_round2/tben_comparison_subj7_per_image.npz")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chasm_config = load_yaml(args.chasmbrain_config)
    data_cfg = chasm_config["data"]
    dataset = load_neuroflux_data_v7h(
        huwng_base_path=data_cfg["huwng_base_path"],
        embeddings_path=data_cfg.get("test_embeddings_path"),
        subjects=[data_cfg["subject"]], split="test", average_trials=True,
        voxels_per_cluster=chasm_config["model"]["voxels_per_cluster"],
        roi_top_k_percent=data_cfg.get("roi_top_k_percent", 0.7),
        augment_noise=False, dinov2_layer=data_cfg.get("dinov2_layer", -1),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(model, checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        return model.to(device).eval()

    # Two-stage models take (cls, patch, stage=2) and return three values; the
    # one-stage models return (prediction, kl). Normalise to a prediction here.
    models, checkpoints = {}, {}
    models["chasmbrain"] = load(build_model(chasm_config), args.chasmbrain_checkpoint)
    checkpoints["chasmbrain"] = args.chasmbrain_checkpoint

    tben_a_cfg = load_yaml(args.tben_one_stage_config)["model"]
    models["tben_one_stage"] = load(
        TBEnOneStageOfficial(
            cls_dim=tben_a_cfg["cls_dim"], patch_dim=tben_a_cfg["patch_dim"],
            num_patches=tben_a_cfg.get("num_patches", 256),
            fmri_dim=tben_a_cfg["fmri_dim"],
            voxels_per_cluster=tben_a_cfg["voxels_per_cluster"],
            hidden_dim=tben_a_cfg.get("hidden_dim", 768),
            nheads=tben_a_cfg.get("nheads", 16),
            dim_feedforward=tben_a_cfg.get("dim_feedforward", 1024),
            enc_layers=tben_a_cfg.get("enc_layers", 0),
            dec_layers=tben_a_cfg.get("dec_layers", 1),
            dropout=tben_a_cfg.get("dropout", 0.1),
            pre_norm=tben_a_cfg.get("pre_norm", True),
            enc_output_layer=tben_a_cfg.get("enc_output_layer", 1),
        ),
        args.tben_one_stage_checkpoint,
    )
    checkpoints["tben_one_stage"] = args.tben_one_stage_checkpoint

    models["tben_two_stage"] = load(
        build_model(load_yaml(args.tben_two_stage_config)), args.tben_two_stage_checkpoint
    )
    checkpoints["tben_two_stage"] = args.tben_two_stage_checkpoint

    if args.direct_checkpoint is not None:
        direct_cfg = load_yaml(args.direct_config)["model"]
        models["direct_one_stage"] = load(
            DirectStage2Only(
                cls_dim=direct_cfg["cls_dim"], patch_dim=direct_cfg["patch_dim"],
                fmri_dim=direct_cfg["fmri_dim"],
                voxels_per_cluster=direct_cfg["voxels_per_cluster"],
                hidden_dim=direct_cfg["hidden_dim"], embed_dim=direct_cfg["embed_dim"],
                latent_dim=direct_cfg.get("latent_dim", 128), depth=direct_cfg["depth"],
                num_global_queries=direct_cfg["num_global_queries"],
                dropout=direct_cfg.get("dropout", 0.1),
                vae_mode=direct_cfg.get("vae_mode", "stochastic"),
                backbone=direct_cfg.get("backbone", "mamba"),
            ),
            args.direct_checkpoint,
        )
        checkpoints["direct_one_stage"] = args.direct_checkpoint

    two_stage_names = {"chasmbrain", "tben_two_stage"}
    arrays = {name: {"pearson": [], "mse": []} for name in models}
    with torch.inference_mode():
        for batch in tqdm(loader, desc="TBEn comparison"):
            cls_token = batch["cls_token"].to(device, non_blocking=True)
            patch_tokens = batch["patch_tokens"].to(device, non_blocking=True)
            target = batch["fmri"].to(device, non_blocking=True)
            for name, model in models.items():
                if name in two_stage_names:
                    _, prediction, _ = model(cls_token, patch_tokens, stage=2)
                else:
                    prediction, _ = model(cls_token, patch_tokens)
                pearson, mse = sample_metrics(prediction, target)
                arrays[name]["pearson"].append(pearson)
                arrays[name]["mse"].append(mse)

    arrays = {
        name: {metric: np.concatenate(chunks) for metric, chunks in metrics.items()}
        for name, metrics in arrays.items()
    }
    rng = np.random.default_rng(args.seed)
    reference = arrays["chasmbrain"]["pearson"]
    result = {
        "subject": data_cfg["subject"],
        "num_images": len(dataset),
        "protocol": "identical NSD split, DINOv2 layer, trial averaging, voxel mask, and metrics",
        "note": (
            "tben_* rows are our reimplementation of the TBEn readout mechanism under our "
            "protocol; they are not a reproduction of the published TBEn numbers."
        ),
        "checkpoints": checkpoints,
        "parameters": {
            name: sum(p.numel() for p in model.parameters())
            for name, model in models.items()
        },
        "models": {},
    }
    for name, metrics in arrays.items():
        entry = {
            "pearson_mean": float(metrics["pearson"].mean()),
            "pearson_95_ci": confidence_interval(metrics["pearson"], rng, args.resamples),
            "mse_mean": float(metrics["mse"].mean()),
        }
        if name != "chasmbrain":
            entry["paired_vs_chasmbrain"] = paired_tests(
                reference, metrics["pearson"], rng, args.resamples
            )
        result["models"][name] = entry

    result["tben_two_stage_minus_tben_one_stage"] = paired_tests(
        arrays["tben_one_stage"]["pearson"], arrays["tben_two_stage"]["pearson"],
        rng, args.resamples,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    per_image = Path(args.per_image_output)
    per_image.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(per_image, **{
        f"{name}_{metric}": values
        for name, metrics in arrays.items()
        for metric, values in metrics.items()
    })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

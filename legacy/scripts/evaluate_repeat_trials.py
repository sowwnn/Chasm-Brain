"""Evaluate averaged vs single-trial targets and latent uncertainty alignment."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.dataset_v7_h import load_neuroflux_data_v7h
from model.hierarchical_v3_1 import HierarchicalARM_V3


def build_model(config):
    cfg = config["model"]
    return HierarchicalARM_V3(
        cls_dim=cfg["cls_dim"], patch_dim=cfg["patch_dim"], num_patches=cfg["num_patches"],
        fmri_dim=cfg["fmri_dim"], voxels_per_cluster=cfg["voxels_per_cluster"],
        hidden_dim_1=cfg["hidden_dim_1"], depth_what=cfg["depth_what"],
        depth_where=cfg["depth_where"], contrastive_dim=cfg["contrastive_dim"],
        embed_dim_2=cfg["embed_dim_2"], latent_dim=cfg.get("latent_dim", 128),
        depth_2=cfg["depth_2"], num_global_queries=cfg["num_global_queries"],
        dropout=cfg.get("dropout", 0.1), vae_mode="stochastic",
    )


def sample_pearson(prediction, target):
    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (prediction * target).sum(dim=1)
    denominator = prediction.square().sum(dim=1).sqrt() * target.square().sum(dim=1).sqrt()
    return (numerator / denominator.clamp_min(1e-12)).cpu().numpy()


def bootstrap_correlation(x, y, rng, resamples):
    indices = rng.integers(0, len(x), size=(resamples, len(x)))
    bx, by = x[indices], y[indices]
    bx = bx - bx.mean(axis=1, keepdims=True)
    by = by - by.mean(axis=1, keepdims=True)
    correlations = (bx * by).sum(axis=1) / np.sqrt((bx * bx).sum(axis=1) * (by * by).sum(axis=1))
    return [float(value) for value in np.quantile(correlations, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_hierarchical_v3_huwng.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/wacv_round2/repeat_trials_subj7.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    cfg, data_cfg = config["model"], config["data"]
    subject = data_cfg["subject"]

    dataset = load_neuroflux_data_v7h(
        huwng_base_path=data_cfg["huwng_base_path"],
        embeddings_path=data_cfg.get("test_embeddings_path"), subjects=[subject], split="test",
        average_trials=True, voxels_per_cluster=cfg["voxels_per_cluster"],
        roi_top_k_percent=data_cfg.get("roi_top_k_percent", 0.7), augment_noise=False,
        dinov2_layer=data_cfg.get("dinov2_layer", -1),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    raw_trials = np.load(
        Path(data_cfg["huwng_base_path"]) / f"subj0{subject}" /
        f"nsd_test_fmri_zscore_sub{subject}.npy", mmap_mode="r"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    captured = {}
    hook = model.stage2.fc_logvar.register_forward_hook(
        lambda module, inputs, output: captured.update(logvar=output.detach())
    )
    averaged_correlations, trial_correlations = [], [[], [], []]
    latent_sigmas, intertrial_variances = [], []

    try:
        with torch.inference_mode():
            for batch in tqdm(loader, desc="Repeat-trial validation"):
                cls_token = batch["cls_token"].to(device, non_blocking=True)
                patch_tokens = batch["patch_tokens"].to(device, non_blocking=True)
                averaged_target = batch["fmri"].to(device, non_blocking=True)
                _, prediction, _ = model(cls_token, patch_tokens, stage=2)

                ids = batch["image_id"].cpu().numpy()
                trials = torch.from_numpy(np.asarray(raw_trials[ids]).copy()).to(device)
                averaged_correlations.append(sample_pearson(prediction, averaged_target))
                for trial_index in range(trials.shape[1]):
                    trial_correlations[trial_index].append(
                        sample_pearson(prediction, trials[:, trial_index, :])
                    )

                latent_sigmas.append(torch.exp(0.5 * captured["logvar"]).mean(dim=1).cpu().numpy())
                intertrial_variances.append(
                    trials.var(dim=1, unbiased=False).mean(dim=1).cpu().numpy()
                )
    finally:
        hook.remove()

    averaged = np.concatenate(averaged_correlations)
    singles = [np.concatenate(values) for values in trial_correlations]
    sigma = np.concatenate(latent_sigmas)
    trial_variance = np.concatenate(intertrial_variances)
    pearson = pearsonr(sigma, trial_variance)
    spearman = spearmanr(sigma, trial_variance)
    rng = np.random.default_rng(args.seed)

    result = {
        "subject": subject,
        "num_images": len(dataset),
        "checkpoint": args.checkpoint,
        "prediction_target_pearson": {
            "trial_averaged_mean": float(averaged.mean()),
            "single_trial_means": [float(values.mean()) for values in singles],
            "single_trial_pooled_mean": float(np.concatenate(singles).mean()),
        },
        "latent_sigma_vs_intertrial_variance": {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "pearson_bootstrap_95_ci": bootstrap_correlation(
                sigma, trial_variance, rng, args.resamples
            ),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

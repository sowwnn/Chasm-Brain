#!/usr/bin/env python3
"""Fit a DINOv2 GAP-to-voxel ridge baseline on the leakage-free CSI1 split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_gap_features(path, layer=-1, chunk_size=128):
    embeddings = np.load(path, mmap_mode="r")
    if embeddings.ndim != 4:
        raise ValueError(f"expected multilayer embeddings [N,L,T,D], got {embeddings.shape}")
    features = np.empty((embeddings.shape[0], embeddings.shape[-1]), dtype=np.float32)
    for start in range(0, len(embeddings), chunk_size):
        stop = min(start + chunk_size, len(embeddings))
        # Match the paper's linear-probe protocol: GAP over patch tokens only.
        features[start:stop] = embeddings[start:stop, layer, 1:, :].mean(axis=1, dtype=np.float32)
    return features


def sample_metrics(prediction, target):
    pred_centered = prediction - prediction.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    pearson = (pred_centered * target_centered).sum(dim=1) / (
        pred_centered.square().sum(dim=1).sqrt()
        * target_centered.square().sum(dim=1).sqrt()
    ).clamp_min(1e-12)
    mse = (prediction - target).square().mean(dim=1)
    return pearson, mse


def bootstrap_ci(values, seed=42, samples=10000):
    rng = np.random.default_rng(seed)
    values = values.detach().cpu().numpy()
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/BOLD5000/processed/CSI1")
    parser.add_argument("--output", default="results/wacv_round2/bold5000_csi1_ridge_seed42.json")
    parser.add_argument("--train_targets", default="train_fmri_zscore.npy")
    parser.add_argument("--test_targets", default="test_fmri_zscore.npy")
    parser.add_argument("--test_embedding_indices")
    parser.add_argument("--roi_metadata")
    parser.add_argument(
        "--alphas", nargs="+", type=float,
        default=[100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0, 300000.0, 1000000.0],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the multi-output ridge solve")
    device = torch.device("cuda")
    data_dir = Path(args.data_dir)
    fit_indices = np.load(data_dir / "fit_indices_seed42.npy")
    validation_indices = np.load(data_dir / "validation_indices_seed42.npy")

    train_features = load_gap_features(data_dir / "dinov2_vitb14_multilayer_train.npy")
    test_features = load_gap_features(data_dir / "dinov2_vitb14_multilayer_test.npy")
    train_targets = np.load(data_dir / args.train_targets, mmap_mode="r")
    test_targets = np.load(data_dir / args.test_targets, mmap_mode="r")
    if args.test_embedding_indices is not None:
        test_embedding_indices = np.load(data_dir / args.test_embedding_indices)
        test_features = test_features[test_embedding_indices]

    x_fit = torch.from_numpy(train_features[fit_indices]).to(device)
    x_validation = torch.from_numpy(train_features[validation_indices]).to(device)
    x_test = torch.from_numpy(test_features).to(device)
    y_fit = torch.from_numpy(np.asarray(train_targets[fit_indices], dtype=np.float32)).to(device)
    y_validation = torch.from_numpy(np.asarray(train_targets[validation_indices], dtype=np.float32)).to(device)
    y_test = torch.from_numpy(np.asarray(test_targets, dtype=np.float32).copy()).to(device)

    x_mean = x_fit.mean(dim=0, keepdim=True)
    x_std = x_fit.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_fit = (x_fit - x_mean) / x_std
    x_validation = (x_validation - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    y_mean = y_fit.mean(dim=0, keepdim=True)
    y_fit_centered = y_fit - y_mean

    # One compact SVD supports all alpha candidates without recomputing X^T Y.
    u, singular_values, vh = torch.linalg.svd(x_fit, full_matrices=False)
    uty = u.transpose(0, 1) @ y_fit_centered
    history = []
    best = None
    best_weights = None
    for alpha in args.alphas:
        shrinkage = singular_values / (singular_values.square() + alpha)
        weights = vh.transpose(0, 1) @ (shrinkage[:, None] * uty)
        prediction = x_validation @ weights + y_mean
        pearson, mse = sample_metrics(prediction, y_validation)
        record = {
            "alpha": alpha,
            "pearson": float(pearson.mean()),
            "mse": float(mse.mean()),
        }
        history.append(record)
        print(json.dumps(record))
        if best is None or record["pearson"] > best["pearson"]:
            best = record
            best_weights = weights.clone()

    test_prediction = x_test @ best_weights + y_mean
    test_pearson, test_mse = sample_metrics(test_prediction, y_test)
    region_results = None
    if args.roi_metadata is not None:
        roi_metadata = json.loads((data_dir / args.roi_metadata).read_text())
        region_results = {}
        for roi in roi_metadata["roi_order"]:
            boundary = roi_metadata["roi_boundaries"][roi]
            start, stop = boundary["start"], boundary["stop"]
            roi_pearson, roi_mse = sample_metrics(
                test_prediction[:, start:stop], y_test[:, start:stop]
            )
            region_results[roi] = {
                "pearson_mean": float(roi_pearson.mean()),
                "pearson_95_ci": bootstrap_ci(roi_pearson, args.seed),
                "mse_mean": float(roi_mse.mean()),
                "num_voxels": int(stop - start),
            }
    result = {
        "status": "complete",
        "dataset": "BOLD5000",
        "subject": "CSI1",
        "baseline": "DINOv2 ViT-B/14 layer 12 patch GAP -> Ridge",
        "split": {
            "fit": int(len(fit_indices)),
            "validation": int(len(validation_indices)),
            "test_repeated": int(len(test_targets)),
            "test_used_for_model_selection": False,
        },
        "num_features": int(x_fit.shape[1]),
        "num_voxels": int(y_fit.shape[1]),
        "effective_parameters": int((x_fit.shape[1] + 1) * y_fit.shape[1]),
        "validation_history": history,
        "best_alpha": best["alpha"],
        "test": {
            "pearson_mean": float(test_pearson.mean()),
            "pearson_95_ci": bootstrap_ci(test_pearson, args.seed),
            "mse_mean": float(test_mse.mean()),
        },
        "regions": region_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        output.with_suffix(".npz"),
        pearson=test_pearson.detach().cpu().numpy(),
        mse=test_mse.detach().cpu().numpy(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

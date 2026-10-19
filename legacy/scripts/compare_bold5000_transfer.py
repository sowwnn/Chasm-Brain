#!/usr/bin/env python3
"""Paired tests for NSD-pretrained BOLD head adaptation."""

import argparse
import json
from pathlib import Path

import numpy as np


def paired_test(candidate, reference, rng, samples):
    difference = candidate - reference
    indices = rng.integers(0, len(difference), size=(samples, len(difference)))
    bootstrap = difference[indices].mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, len(difference)))
    null = (signs * difference).mean(axis=1)
    observed = abs(difference.mean())
    return {
        "mean_delta": float(difference.mean()),
        "bootstrap_95_ci": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
        "paired_permutation_p": float((np.count_nonzero(np.abs(null) >= observed) + 1) / (samples + 1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer", default="results/wacv_round2/bold5000_csi1_nsd_transfer_seed42.npz")
    parser.add_argument("--scratch", default="results/wacv_round2/bold5000_csi1_chasmbrain_seed42.npz")
    parser.add_argument("--ridge", default="results/wacv_round2/bold5000_csi1_ridge_seed42.npz")
    parser.add_argument("--output", default="results/wacv_round2/bold5000_csi1_transfer_comparison_seed42.json")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    transfer = np.load(args.transfer)
    scratch = np.load(args.scratch)
    ridge = np.load(args.ridge)
    rng = np.random.default_rng(args.seed)
    result = {
        "dataset": "BOLD5000",
        "subject": "CSI1",
        "num_paired_test_images": int(len(ridge["pearson"])),
        "transfer_vs_scratch": {
            "pearson": paired_test(transfer["predicted_pearson"], scratch["predicted_pearson"], rng, args.samples),
            "mse": paired_test(transfer["predicted_mse"], scratch["predicted_mse"], rng, args.samples),
        },
        "transfer_vs_ridge": {
            "pearson": paired_test(transfer["predicted_pearson"], ridge["pearson"], rng, args.samples),
            "mse": paired_test(transfer["predicted_mse"], ridge["mse"], rng, args.samples),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

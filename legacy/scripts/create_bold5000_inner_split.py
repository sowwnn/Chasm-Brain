#!/usr/bin/env python3
"""Create a deterministic fit/validation split while preserving repeated images for final test."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/BOLD5000/processed/CSI1")
    parser.add_argument("--validation_size", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    targets = np.load(data_dir / "train_fmri_zscore.npy", mmap_mode="r")
    if not 0 < args.validation_size < len(targets):
        raise ValueError("validation_size must be between zero and the number of training images")

    permutation = np.random.default_rng(args.seed).permutation(len(targets))
    validation = np.sort(permutation[: args.validation_size])
    fit = np.sort(permutation[args.validation_size :])
    np.save(data_dir / "fit_indices_seed42.npy", fit)
    np.save(data_dir / "validation_indices_seed42.npy", validation)

    metadata = {
        "seed": args.seed,
        "source_train_images": int(len(targets)),
        "fit_images": int(len(fit)),
        "validation_images": int(len(validation)),
        "final_test_images": int(np.load(data_dir / "test_fmri_zscore.npy", mmap_mode="r").shape[0]),
        "final_test_used_for_model_selection": False,
    }
    (data_dir / "inner_split_seed42.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

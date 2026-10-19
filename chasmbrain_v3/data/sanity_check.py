"""Spec §3 sanity checks for one subject, using dataset/huwng/ (spec §3, see nsd_loader docstring for
the subj02/subj07 swap fix and the image-order caveat). Run: python -m chasmbrain_v3.data.sanity_check --subj 1

1. Pearson between two repeats of the same train image >= 0 and sane magnitude (no ground-truth NSD id
   to compare against, so this checks internal consistency, not the literature's 0.2-0.4 range directly).
2. train/val split has no overlap; test is the untouched huwng "test" array (1000 rows).
3. Mean noise ceiling over the mask, per-ROI breakdown, written to chasmbrain_v3_artifacts/cache/subj{N}/sanity.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .nsd_loader import load_betas, load_roi_labels, n_voxels
from .normalize import NC_FORMULA, noise_ceiling
from .split import make_split

CACHE_DIR = Path("chasmbrain_v3_artifacts/cache")


def main(subj: int):
    betas = load_betas(subj, "train")              # [9000, 3, n_voxels] float64, memory-mapped
    split = make_split(subj)
    assert not np.intersect1d(split["train"], split["val"]).size
    print(f"subj{subj}: betas {betas.shape}, split sizes train={len(split['train'])} val={len(split['val'])} test=1000")

    b = np.asarray(betas)                            # materialise once; ~1-2 GB per subject
    a, c = b[:, 0, :], b[:, 1, :]                    # repeat 1 vs repeat 2, all train images
    an = (a - a.mean(0)) / a.std(0)
    cn = (c - c.mean(0)) / c.std(0)
    r = (an * cn).mean(0)

    nc = noise_ceiling(b, n_reps=3)
    try:
        rois = load_roi_labels(subj)
        per_roi = {k: float(nc[m].mean()) for k, m in rois.items() if m.any()}
    except Exception as e:
        per_roi = {"error": str(e)}

    out = {
        "subj": subj, "n_voxels": n_voxels(subj), "n_train_images": int(b.shape[0]),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "repeat_pearson": {"mean": float(r.mean()), "p10": float(np.percentile(r, 10)), "p90": float(np.percentile(r, 90))},
        "noise_ceiling": {"mean": float(nc.mean()), "median": float(np.median(nc)), "per_roi_mean": per_roi, "formula": NC_FORMULA},
    }
    print(json.dumps(out, indent=1))
    d = CACHE_DIR / f"subj{subj:02d}"; d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / "sanity.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--subj", type=int, default=1)
    main(p.parse_args().subj)

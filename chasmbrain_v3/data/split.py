"""Train / val split within huwng's private train set (spec §3).

huwng already fixes the test set (1,000 images, loaded separately via load_betas(subj, "test")) --
that IS the shared-1000 role here (see nsd_loader KNOWN ISSUES: these are not verified to be NSD's
actual shared-1000 ids, but they are a held-out 1,000 never touched by anything in this function).
val = 1,000 images held out from the 9,000 "train" rows with a fixed seed, written once.
"""
import json
from pathlib import Path

import numpy as np

from .nsd_loader import _CFG

SPLITS_DIR = Path(_CFG["splits_dir"])


def make_split(subj: int, seed: int = 0) -> dict:
    """Return {'train': idx[8000], 'val': idx[1000]}, 0-based row indices into the 9000-row
    "train" huwng array (not NSD image ids -- see nsd_loader KNOWN ISSUES #2). Test is simply all
    1000 rows of the "test" huwng array; it needs no split file. Written once to
    splits/subj{N}.json; the seed argument is ignored if that file already exists."""
    path = SPLITS_DIR / f"subj{subj:02d}.json"
    if path.exists():
        d = json.load(open(path))
        return {k: np.asarray(v, dtype=np.int64) for k, v in d.items()}
    rng = np.random.default_rng(seed)
    val = np.sort(rng.choice(9000, size=_CFG["n_val"], replace=False))
    train = np.setdiff1d(np.arange(9000), val)
    split = {"train": train, "val": val}
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({k: v.tolist() for k, v in split.items()}, open(path, "w"))
    return {k: v.astype(np.int64) for k, v in split.items()}


def average_repeats(betas: np.ndarray) -> np.ndarray:
    """[n_images, 3, n_voxels] -> [n_images, n_voxels], mean over the 3 repeats."""
    return np.asarray(betas).mean(axis=1).astype(np.float32)

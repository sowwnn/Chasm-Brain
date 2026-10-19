"""Train / val / test split by image id (spec §3). Written once to splits/subj{N}.json and never changed."""
import numpy as np


def make_split(subj: int, seed: int = 0) -> dict:
    """{train: 8000, val: 1000, test: 1000}. test = NSD shared-1000; val = 1000 random subject-specific images."""
    raise NotImplementedError


def average_repeats(betas: np.ndarray, image_ids: np.ndarray):
    """Return (X [n_images, n_voxels], n_reps [n_images])."""
    raise NotImplementedError

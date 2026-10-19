"""NSD loading (spec §3)."""
import numpy as np


def load_betas(subj: int, session_range) -> np.ndarray:
    """Return betas inside the mask, shape [n_trials, n_voxels], in original trial order."""
    raise NotImplementedError


def load_mask(subj: int):
    """Return (mask_bool [X, Y, Z], coords [n_voxels, 3]) with coords in beta column order."""
    raise NotImplementedError


def load_stimuli(subj: int):
    """Return (image_ids [n_trials], images) using NSD 73k ids."""
    raise NotImplementedError


def load_roi_labels(subj: int) -> dict:
    """Early / Midventral / Lateral / Parietal / Midlateral / Ventral. Analysis only, never training."""
    raise NotImplementedError

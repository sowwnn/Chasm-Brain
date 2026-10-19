"""NSD loading from the pre-extracted dataset/huwng/ cache (spec §3).

Source: dataset/huwng/{dir}/nsd_{train,test}_fmri_zscore_sub{k}.npy, shape [n_images, 3, n_voxels] float64,
already masked to nsdgeneral and z-scored, in a fixed image order private to this cache (does NOT match
NSD's 73k image ids or masterordering -- see KNOWN ISSUES). Features: nsd_dinov2_vitb14_multilayer_{split}_sub{k}.npy,
shape [n_images, 4, 257, 768] float16 (4 layers, CLS + 256 patch tokens, ViT-B/14).

KNOWN ISSUES (verified 2026-10-20, see docs/CHASMBrain_v3_spec.md and memory project-subj02-07-swapped):
  1. huwng/subj02 and huwng/subj07 directories are swapped: subj02's files have 12,682 voxels
     (= subj07's true nsdgeneral mask) and subj07's files have 14,278 (= subj02's). Corrected here
     via huwng_dir_for_subject in configs/data.yaml -- callers use the nominal subject id (1,2,5,7)
     and get the right data.
  2. The image order in huwng/ does not match /mnt/HDD/dataset/NSD_raw's masterordering/subjectim
     (RDM correlation ~0 against dataset/nsd_raw_subj01, which does use real NSD 73k ids). There is
     no recovered mapping to absolute NSD 73k image ids. Each subject's "train"/"test" rows are an
     internally consistent, subject-private image order (fMRI and features for row i are the same
     image, verified by ridge-decoding signal well above a shuffled control) -- fine for everything
     in this package (splits, training, noise ceiling), but NOT for anything that needs the actual
     stimulus image (ROI assignment by a fresh mask read is fine; nsd_stimuli.hdf5 lookup is not).
  3. ROI stream labels (load_roi_labels) are assigned from NSD_raw's streams.nii.gz using the mask's
     own voxel order; this assumes huwng's per-subject voxel order follows the standard nsdgeneral
     mask flattening used by the MindEye2 release, which has not been independently verified (voxel
     *counts* match exactly for all 4 subjects; the spatial *order* is the standard MindEye2
     convention but unconfirmed here). Treat per-ROI numbers as best-effort until spot-checked.
"""
from pathlib import Path

import numpy as np
import yaml

_CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "data.yaml"))
HUWNG_ROOT = Path(_CFG["huwng_root"])
DIR_FOR_SUBJECT = {int(k): v for k, v in _CFG["huwng_dir_for_subject"].items()}


def _suffix(subj: int) -> str:
    """huwng filenames use the directory's own digit (e.g. subj02's files end in _sub2, even though
    that directory actually holds subject 7's data -- see KNOWN ISSUES #1)."""
    d = DIR_FOR_SUBJECT[subj]
    return d.replace("subj", "").lstrip("0") or "0"


def _path(subj: int, kind: str, split: str) -> Path:
    d, suf = DIR_FOR_SUBJECT[subj], _suffix(subj)
    name = {
        "fmri": f"nsd_{split}_fmri_zscore_sub{suf}.npy",
        "dinov2": f"nsd_dinov2_vitb14_multilayer_{split}_sub{suf}.npy",
    }[kind]
    return HUWNG_ROOT / d / name


def load_betas(subj: int, split: str = "train") -> np.ndarray:
    """Single-trial betas [n_images, 3, n_voxels] float64, z-scored, masked to nsdgeneral.
    split: 'train' (9000 images) or 'test' (1000 images)."""
    return np.load(_path(subj, "fmri", split), mmap_mode="r")


def n_voxels(subj: int) -> int:
    return load_betas(subj, "test").shape[-1]


def load_dinov2(subj: int, split: str = "train", layer: int = -1):
    """Return (cls [n, 768] float32, patch [n, 256, 768] float16) at the given layer (0-3, -1 = last
    of the 4 cached layers; the cached layers are NOT necessarily contiguous transformer blocks --
    treat `layer` as an index into the 4 cached layers, not a raw block index)."""
    d = np.load(_path(subj, "dinov2", split), mmap_mode="r")   # [n, 4, 257, 768] float16
    d = d[:, layer]                                             # [n, 257, 768]
    cls = np.ascontiguousarray(d[:, 0]).astype(np.float32)
    patch = np.ascontiguousarray(d[:, 1:])
    return cls, patch


def load_roi_labels(subj: int) -> dict:
    """{stream name: bool [n_voxels]} from NSD streams.nii.gz on /mnt/HDD/dataset/NSD_raw, using the
    TRUE subject's mask (DIR_FOR_SUBJECT corrects for the swap). See KNOWN ISSUES #3 for the voxel-order
    caveat. Analysis only, never training (spec §3)."""
    import nibabel as nib
    root = Path("/mnt/HDD/dataset/NSD_raw/nsddata/ppdata")
    roi_dir = root / f"subj{subj:02d}" / "func1pt8mm" / "roi"
    mask = nib.load(roi_dir / "nsdgeneral.nii.gz").get_fdata() == 1
    streams = nib.load(roi_dir / "streams.nii.gz").get_fdata()[mask].astype(int)
    labels = {1: "Early", 2: "Midventral", 3: "Midlateral", 4: "Midparietal", 5: "Ventral", 6: "Lateral", 7: "Parietal"}
    return {name: streams == code for code, name in labels.items()}

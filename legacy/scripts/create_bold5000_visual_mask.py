#!/usr/bin/env python3
"""Create a CSI1 visual-cortex mask in BOLD5000 functional space."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np


MASK_DIR = Path("dataset/BOLD5000/wholebrain/CSI1/masks")
APARCASEG_PATH = MASK_DIR / "aparcaseg_func.nii.gz"
OUTPUT_PATH = MASK_DIR / "visual_cortex_aparc_mask.nii.gz"
METADATA_PATH = MASK_DIR / "visual_cortex_aparc_mask.json"

# Destrieux cortical labels covering occipital and ventral visual cortex.
# Left hemisphere labels are 111xx; right hemisphere labels are 121xx.
LEFT_LABELS = {
    11102: "G_and_S_occipital_inf",
    11111: "G_cuneus",
    11119: "G_occipital_middle",
    11120: "G_occipital_sup",
    11121: "G_oc-temp_lat-fusifor",
    11122: "G_oc-temp_med-Lingual",
    11123: "G_oc-temp_med-Parahip",
    11143: "Pole_occipital",
    11145: "S_calcarine",
    11151: "S_collat_transv_ant",
    11152: "S_collat_transv_post",
    11160: "S_occipital_ant",
    11161: "S_oc-temp_lat",
    11162: "S_oc-temp_med_and_Lingual",
    11166: "S_parieto_occipital",
}
RIGHT_LABELS = {label + 1000: name for label, name in LEFT_LABELS.items()}
LABELS = LEFT_LABELS | RIGHT_LABELS


def main() -> None:
    atlas_image = nib.load(APARCASEG_PATH)
    atlas = np.asarray(atlas_image.dataobj).astype(np.int32)
    mask = np.isin(atlas, list(LABELS)).astype(np.uint8)

    header = atlas_image.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask, atlas_image.affine, header), OUTPUT_PATH)

    counts = {str(label): int((atlas == label).sum()) for label in LABELS}
    metadata = {
        "source": str(APARCASEG_PATH),
        "output": str(OUTPUT_PATH),
        "shape": list(mask.shape),
        "voxel_count": int(mask.sum()),
        "voxel_size_mm": [float(value) for value in atlas_image.header.get_zooms()[:3]],
        "labels": {str(label): name for label, name in LABELS.items()},
        "label_voxel_counts": counts,
        "note": "Anatomical occipital and ventral visual cortex; reliability filtering must use training data only.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {OUTPUT_PATH} with {int(mask.sum()):,} voxels")


if __name__ == "__main__":
    main()

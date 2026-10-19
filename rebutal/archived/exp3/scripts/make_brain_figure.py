"""Posterior brain view: per-voxel Pearson under 4 branch variants (subj01).

4 panels (Both, Swap, Local only, Global only) × 2 hemispheres,
posterior view (occipital cortex facing camera).
"""
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn import surface, plotting

REPO = Path(__file__).resolve().parent.parent.parent.parent
RES = REPO / "rebutal/exp3/results"
NSD_PPDATA = "/mnt/HDD/dataset/NSD_raw/nsddata/ppdata/subj01/func1pt8mm"
FS_SURF    = "/mnt/HDD/dataset/NSD_raw/nsddata/freesurfer/subj01/surf"
VOX_PER_ROI = 30

mask_img = nib.load(f"{NSD_PPDATA}/roi/nsdgeneral.nii.gz")
mask = mask_img.get_fdata() > 0
mask_idx = np.where(mask)
n_vox = int(mask.sum())


def per_roi_to_volume(per_roi):
    per_roi = np.asarray(per_roi)
    vox = np.full(n_vox, np.nan, dtype=np.float32)
    for i in range(per_roi.shape[0]):
        vox[i * VOX_PER_ROI:(i + 1) * VOX_PER_ROI] = per_roi[i]
    vol = np.full(mask.shape, np.nan, dtype=np.float32)
    vol[mask_idx] = vox
    return nib.Nifti1Image(vol, mask_img.affine)


per_roi = {v: np.asarray(json.load(open(RES / f"per_region_{v}_subj1.json"))["per_roi_pearson"])
           for v in ["both", "swap", "whatonly", "whereonly"]}
imgs = {v: per_roi_to_volume(per_roi[v]) for v in per_roi}

variants = [("both", "Both (full)"),
            ("whereonly", "Local only (no CLS)"),
            ("swap", "Swap (CLS↔patches)"),
            ("whatonly", "Global only (no patches)")]

hemis = [("lh", f"{FS_SURF}/lh.inflated", f"{FS_SURF}/lh.pial", "left"),
         ("rh", f"{FS_SURF}/rh.inflated", f"{FS_SURF}/rh.pial", "right")]

vmin, vmax = 0.0, 0.7
cmap = "RdBu_r"

# fig, axes = plt.subplots(2, 4, figsize=(13, 6.5),
#                          subplot_kw={"projection": "3d"})

# for col, (vkey, vname) in enumerate(variants):
#     img = imgs[vkey]
#     for row, (hname, inflated, pial, side) in enumerate(hemis):
#         tex = surface.vol_to_surf(img, pial, radius=2.0,
#                                   interpolation="nearest_most_frequent")
#         ax = axes[row, col]
#         plotting.plot_surf_stat_map(
#             inflated, tex, hemi="both", view="posterior",
#             vmax=vmax, threshold=0.001, cmap=cmap,
#             bg_on_data=True, axes=ax, colorbar=False,
#         )
#         if row == 0:
#             ax.set_title(vname, fontsize=11, pad=2)
#         if col == 0:
#             ax.text2D(-0.02, 0.5, hname.upper(), transform=ax.transAxes,
#                       fontsize=11, ha="right", va="center", weight="bold")


# cax = fig.add_axes([0.92, 0.25, 0.012, 0.5])
# sm = mpl.cm.ScalarMappable(cmap=cmap,
#                            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
# cb = fig.colorbar(sm, cax=cax)
# cb.set_label("Pearson r", fontsize=10)

# 1. Chỉnh lại subplot thành 1 hàng, 4 cột
fig, axes = plt.subplots(1, 4, figsize=(16, 4), 
                         subplot_kw={"projection": "3d"})

for col, (vkey, vname) in enumerate(variants):
    img = imgs[vkey]
    ax = axes[col] # Mỗi biến thể (variant) chiếm 1 cột
    
    for hname, inflated, pial, side in hemis:
        # Sử dụng pial để có bề mặt nhăn giống ảnh mẫu
        tex = surface.vol_to_surf(img, pial, radius=2.0,
                                  interpolation="nearest_most_frequent")
        
        # Vẽ cả LH và RH vào cùng một trục 'ax'
        plotting.plot_surf_stat_map(
            pial, tex, hemi=side, view="posterior",
            vmax=vmax, threshold=0.001, cmap=cmap,
            bg_on_data=True, axes=ax, colorbar=False
        )
    
    ax.set_title(vname, fontsize=12, pad=10)

# Chỉnh lại vị trí colorbar vì chỉ còn 1 hàng
cax = fig.add_axes([0.93, 0.2, 0.01, 0.6])
sm = mpl.cm.ScalarMappable(cmap=cmap,
                           norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
cb = fig.colorbar(sm, cax=cax)
cb.set_label("Pearson r", fontsize=10)

plt.subplots_adjust(left=0.02, right=0.92, top=0.85, bottom=0.05, wspace=0.05)
 

fig.suptitle("Per-voxel prediction quality, posterior view (subj01)",
             fontsize=12, y=0.99)
plt.subplots_adjust(left=0.04, right=0.9, top=0.93, bottom=0.02,
                    wspace=0.0, hspace=0.0)
out = RES / "fig_brain_posterior.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"✓ {out}")

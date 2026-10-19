"""Per-region Pearson analysis for exp3 branch ablations.

For each (variant × subject):
  1. Load checkpoint, run inference on test set to get pred_roi_means.
  2. Compute Pearson per ROI across test samples (gt vs pred).
  3. Map each ROI cluster (sequential 30-voxel bins) to NSD stream label
     via majority vote (using nsdgeneral mask + streams.nii.gz from NSD raw).
  4. Aggregate Pearson per stream group (early / mid / higher).

Outputs:
  rebutal/exp3/results/per_region_<variant>_subj<N>.json   (per-ROI vector + groupings)
  rebutal/exp3/results/per_region_summary.csv              (aggregated table)

Stream IDs (NSD docs):
  -1/0: unknown/none, 1: early (V1/V2/V3), 2: midventral, 3: midlateral,
  4: midparietal, 5: ventral, 6: lateral, 7: parietal
"""
import sys
import json
import csv
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(REPO))

from model.hierarchical_v3_multisubj import HierarchicalARM_MultiSubject
from model.hierarchical_v3_1 import HierarchicalARM_V3
from data.dataset_v7_h import load_neuroflux_data_v7h


NSD_ROI_TMPL = "/mnt/HDD/dataset/NSD_raw/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/{}.nii.gz"

# Baseline ckpts (single-subject HierarchicalARM_V3); per-subject if available.
BOTH_CKPTS = {
    1: REPO / "checkpoints/chasm_rerun_s1/best_stage1.pth",
}

VARIANTS = {
    "both":      {"branch_mode": "both", "model": "v3_baseline"},
    "swap":      {"branch_mode": "swap"},
    "whatonly":  {"branch_mode": "what_only"},
    "whereonly": {"branch_mode": "where_only"},
}
# Per-region analysis only on subj01 (huwng voxel ordering verified vs nsdgeneral mask).
# subj02/05/07: huwng uses a different voxel subset → ordering not reconstructible from
# repo data alone. They appear in overall-only summaries elsewhere.
SUBJECTS = [1]
SUBJECT_FMRI_DIMS = {1: 15724, 2: 12682, 5: 13039, 7: 14278}

# Stream grouping for region-level summary
GROUPS = {
    "early":    [1],          # V1, V2, V3
    "mid":      [2, 3, 4],    # midventral, midlateral, midparietal
    "higher":   [5, 6, 7],    # ventral, lateral, parietal
}


def load_streams_per_voxel(subj: int) -> np.ndarray:
    """Return [n_voxels] stream label per voxel, ordered to match the fmri tensor."""
    mask = nib.load(NSD_ROI_TMPL.format(subj, "nsdgeneral")).get_fdata()  # (81,104,83)
    streams = nib.load(NSD_ROI_TMPL.format(subj, "streams")).get_fdata()
    assert mask.shape == streams.shape, f"shape mismatch: {mask.shape} vs {streams.shape}"
    mask_bool = mask > 0
    n_vox = int(mask_bool.sum())
    expected = SUBJECT_FMRI_DIMS[subj]
    assert n_vox == expected, f"subj{subj}: nsdgeneral has {n_vox}, fmri has {expected}"
    return streams[mask_bool].astype(np.int32)  # nifti-native ravel order


def assign_roi_streams(streams_per_voxel: np.ndarray, voxels_per_cluster: int = 30) -> np.ndarray:
    """Majority-vote stream label per ROI cluster (sequential bins)."""
    n_vox = streams_per_voxel.shape[0]
    n_rois = n_vox // voxels_per_cluster
    roi_labels = np.zeros(n_rois, dtype=np.int32)
    for i in range(n_rois):
        bin_ = streams_per_voxel[i * voxels_per_cluster:(i + 1) * voxels_per_cluster]
        # Majority vote, ignoring -1 (unknown). If all -1, label = -1.
        vals = bin_[bin_ >= 0]
        if vals.size == 0:
            roi_labels[i] = -1
        else:
            counts = np.bincount(vals)
            roi_labels[i] = int(counts.argmax())
    return roi_labels


def pearson_per_dim(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pearson correlation along axis=0 (samples), returned per dim. NaN for zero-std."""
    pc = pred - pred.mean(axis=0, keepdims=True)
    tc = target - target.mean(axis=0, keepdims=True)
    ps = pc.std(axis=0) + 1e-9
    ts = tc.std(axis=0) + 1e-9
    n = pred.shape[0]
    return (pc * tc).sum(axis=0) / (n * ps * ts)


def run_inference(ckpt_path: Path, subj: int, branch_mode: str, device: torch.device,
                  model_kind: str = "multisubj"):
    """Load checkpoint, run on test set, return (pred_rois [N,R], gt_rois [N,R]).

    model_kind: "multisubj" (HierarchicalARM_MultiSubject) or "v3_baseline" (HierarchicalARM_V3).
    """
    fmri_dim = SUBJECT_FMRI_DIMS[subj]
    if model_kind == "v3_baseline":
        model = HierarchicalARM_V3(
            cls_dim=768, patch_dim=768, num_patches=256, fmri_dim=fmri_dim,
            voxels_per_cluster=30, hidden_dim_1=512, depth_what=2, depth_where=4,
            contrastive_dim=256, embed_dim_2=384, latent_dim=128, depth_2=6,
            num_global_queries=128, dropout=0.4, roi_top_k_percent=0.7,
        ).to(device)
    else:
        model = HierarchicalARM_MultiSubject(
            subject_fmri_dims={subj: fmri_dim},
            cls_dim=768, patch_dim=768, num_patches=256,
            voxels_per_cluster=30, hidden_dim_1=512, depth_what=2, depth_where=4,
            contrastive_dim=256, embed_dim_2=384, latent_dim=128, depth_2=6,
            num_global_queries=128, dropout=0.4, branch_mode=branch_mode,
        ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = load_neuroflux_data_v7h(
        huwng_base_path="dataset/huwng",
        subjects=[subj], split="test",
        average_trials=True,
        voxels_per_cluster=30, roi_top_k_percent=0.7,
        augment_noise=False, dinov2_layer=-1,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            cls_t = batch["cls_token"].to(device)
            pat_t = batch["patch_tokens"].to(device)
            gt_roi = batch["roi_means"].to(device)
            if model_kind == "v3_baseline":
                pred_roi, _, _ = model(cls_t, pat_t, stage=1)
            else:
                pred_roi, _, _ = model(cls_t, pat_t, subj, stage=1)
            preds.append(pred_roi.cpu().numpy())
            gts.append(gt_roi.cpu().numpy())
    return np.concatenate(preds, 0), np.concatenate(gts, 0)


def aggregate_groups(per_roi_pearson: np.ndarray, roi_labels: np.ndarray) -> dict:
    out = {"overall_mean": float(np.nanmean(per_roi_pearson)),
           "overall_n_rois": int(per_roi_pearson.shape[0])}
    for gname, sids in GROUPS.items():
        mask = np.isin(roi_labels, sids)
        if mask.any():
            vals = per_roi_pearson[mask]
            out[gname] = {"mean": float(np.nanmean(vals)),
                          "std":  float(np.nanstd(vals)),
                          "n_rois": int(mask.sum())}
        else:
            out[gname] = {"mean": float("nan"), "std": float("nan"), "n_rois": 0}
    # Per-stream breakdown (1..7)
    out["per_stream"] = {}
    for sid in range(1, 8):
        mask = roi_labels == sid
        if mask.any():
            out["per_stream"][str(sid)] = {
                "mean": float(np.nanmean(per_roi_pearson[mask])),
                "n_rois": int(mask.sum()),
            }
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = REPO / "rebutal/exp3/results"
    ckpt_root   = REPO / "rebutal/exp3/checkpoints"
    csv_rows = []

    for subj in SUBJECTS:
        print(f"\n=== subj{subj} ===")
        streams_v = load_streams_per_voxel(subj)
        roi_labels = assign_roi_streams(streams_v, voxels_per_cluster=30)
        print(f"  ROI label distribution: {dict(zip(*np.unique(roi_labels, return_counts=True)))}")

        for variant, vcfg in VARIANTS.items():
            if vcfg.get("model") == "v3_baseline":
                ckpt = BOTH_CKPTS.get(subj)
                if ckpt is None or not ckpt.exists():
                    print(f"  [skip] no baseline ckpt for subj{subj}"); continue
            else:
                ckpt = ckpt_root / f"{variant}_subj{subj}/best_stage1.pth"
                if not ckpt.exists():
                    print(f"  [skip] no checkpoint: {ckpt}"); continue
            print(f"  → {variant}: inferring...")
            pred, gt = run_inference(
                ckpt, subj, vcfg["branch_mode"], device,
                model_kind=vcfg.get("model", "multisubj"),
            )
            per_roi = pearson_per_dim(pred, gt)
            agg = aggregate_groups(per_roi, roi_labels)

            out_json = results_dir / f"per_region_{variant}_subj{subj}.json"
            json.dump({
                "subj": subj, "variant": variant, "branch_mode": vcfg["branch_mode"],
                "per_roi_pearson": per_roi.tolist(),
                "roi_labels": roi_labels.tolist(),
                "groups": agg,
            }, open(out_json, "w"), indent=2)

            csv_rows.append({
                "subj": subj, "variant": variant,
                "overall": agg["overall_mean"],
                "early_mean":  agg["early"]["mean"],  "early_n":  agg["early"]["n_rois"],
                "mid_mean":    agg["mid"]["mean"],    "mid_n":    agg["mid"]["n_rois"],
                "higher_mean": agg["higher"]["mean"], "higher_n": agg["higher"]["n_rois"],
            })
            print(f"     overall={agg['overall_mean']:.4f}  "
                  f"early={agg['early']['mean']:.4f}({agg['early']['n_rois']})  "
                  f"mid={agg['mid']['mean']:.4f}({agg['mid']['n_rois']})  "
                  f"higher={agg['higher']['mean']:.4f}({agg['higher']['n_rois']})")

    csv_path = results_dir / "per_region_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n✓ Wrote {csv_path}")


if __name__ == "__main__":
    main()

"""Per-region Pearson for TBEn Stage 1, matched to the "Both (CLS + patch)" row.

Reuses, without modification, the two things that make the numbers comparable:

  1. The cluster -> NSD stream mapping (`roi_labels`, 524 entries for subj01),
     read back from `rebutal/archived/exp3/results/per_region_both_subj1.json`.
     It was produced by `analyze_per_region.py` from nsdgeneral + streams.nii.gz;
     the raw NSD volumes are no longer mounted, and reading the stored labels is
     in any case the exact mapping the "Both" row used.
  2. `pearson_per_dim` and `aggregate_groups` from that same script, imported by
     file path rather than copied, so the correlation is over test images per
     cluster and then averaged unweighted over clusters within a group.

Outputs rebutal/exp3/results/per_region_tben_subj1.json plus a printed table.
"""
import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))

from model.tben_official import HierarchicalARM_V3_TBEn
from data.dataset_v7_h import load_neuroflux_data_v7h

REF_JSON = REPO / "rebutal/archived/exp3/results/per_region_both_subj1.json"
ANALYZE_PER_REGION = REPO / "rebutal/archived/exp3/scripts/analyze_per_region.py"
CKPT = REPO / "checkpoints/wacv_round2/tben_two_stage/subj1_seed42/best_stage1.pth"
CONFIG = REPO / "configs/train_tben_two_stage_subj1.yaml"
OUT_JSON = REPO / "rebutal/exp3/results/per_region_tben_subj1.json"


def _load_reference_module():
    spec = importlib.util.spec_from_file_location("analyze_per_region", ANALYZE_PER_REGION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_inference(config: dict, device: torch.device):
    m = config["model"]
    model = HierarchicalARM_V3_TBEn(
        cls_dim=m["cls_dim"], patch_dim=m["patch_dim"], num_patches=m["num_patches"],
        fmri_dim=m["fmri_dim"], voxels_per_cluster=m["voxels_per_cluster"],
        hidden_dim_1=m["hidden_dim_1"], depth_what=m["depth_what"],
        depth_where=m["depth_where"], contrastive_dim=m["contrastive_dim"],
        embed_dim_2=m["embed_dim_2"], depth_2=m["depth_2"],
        num_global_queries=m["num_global_queries"], dropout=m["dropout"],
        roi_top_k_percent=config["data"]["roi_top_k_percent"],
        tben_hidden_dim=m["tben_hidden_dim"], tben_nheads=m["tben_nheads"],
        tben_dim_feedforward=m["tben_dim_feedforward"],
        tben_enc_layers=m["tben_enc_layers"], tben_dec_layers=m["tben_dec_layers"],
        tben_pre_norm=m["tben_pre_norm"], tben_enc_output_layer=m["tben_enc_output_layer"],
    ).to(device)
    missing, unexpected = model.load_state_dict(
        torch.load(CKPT, map_location=device), strict=False
    )
    stage1_missing = [k for k in missing if k.startswith("stage1.")]
    assert not stage1_missing, f"Stage 1 weights missing from checkpoint: {stage1_missing}"
    model.eval()

    ds = load_neuroflux_data_v7h(
        huwng_base_path=config["data"]["huwng_base_path"],
        subjects=[config["data"]["subject"]], split="test",
        average_trials=config["data"]["average_trials_val"],
        voxels_per_cluster=m["voxels_per_cluster"],
        roi_top_k_percent=config["data"]["roi_top_k_percent"],
        augment_noise=False, dinov2_layer=-1,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            pred_roi, _, _ = model(
                batch["cls_token"].to(device), batch["patch_tokens"].to(device), stage=1
            )
            preds.append(pred_roi.float().cpu().numpy())
            gts.append(batch["roi_means"].float().cpu().numpy())
    return np.concatenate(preds, 0), np.concatenate(gts, 0)


def main():
    ref_mod = _load_reference_module()
    reference = json.load(open(REF_JSON))
    roi_labels = np.array(reference["roi_labels"], dtype=np.int32)

    config = yaml.safe_load(open(CONFIG))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred, gt = run_inference(config, device)
    assert pred.shape[1] == roi_labels.shape[0], (
        f"cluster count mismatch: pred {pred.shape[1]} vs labels {roi_labels.shape[0]}"
    )
    print(f"test set: {pred.shape[0]} images x {pred.shape[1]} clusters")

    per_roi = ref_mod.pearson_per_dim(pred, gt)
    agg = ref_mod.aggregate_groups(per_roi, roi_labels)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "subj": 1, "variant": "tben_stage1",
        "checkpoint": str(CKPT.relative_to(REPO)),
        "per_roi_pearson": per_roi.tolist(),
        "roi_labels": roi_labels.tolist(),
        "groups": agg,
    }, open(OUT_JSON, "w"), indent=2)

    ref_per_roi = np.array(reference["per_roi_pearson"])
    print(f"\n{'group':<12}{'n':>5}{'TBEn-S1':>10}{'Both':>10}{'diff':>9}")
    rows = [("overall", np.ones_like(roi_labels, dtype=bool))]
    rows += [(g, np.isin(roi_labels, s)) for g, s in ref_mod.GROUPS.items()]
    for name, mask in rows:
        a, b = np.nanmean(per_roi[mask]), np.nanmean(ref_per_roi[mask])
        print(f"{name:<12}{int(mask.sum()):>5}{a:>10.4f}{b:>10.4f}{a - b:>+9.4f}")

    print(f"\nper-stream (1 early, 2 midventral, 3 midlateral, 4 midparietal, "
          f"5 ventral, 6 lateral, 7 parietal)")
    print(f"{'stream':<12}{'n':>5}{'TBEn-S1':>10}{'Both':>10}{'diff':>9}")
    for sid in range(1, 8):
        mask = roi_labels == sid
        if not mask.any():
            continue
        a, b = np.nanmean(per_roi[mask]), np.nanmean(ref_per_roi[mask])
        print(f"{sid:<12}{int(mask.sum()):>5}{a:>10.4f}{b:>10.4f}{a - b:>+9.4f}")

    print(f"\n✓ Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()

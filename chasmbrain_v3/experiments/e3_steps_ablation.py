"""E1: n_steps in {1,2,4,8}, Laplacian basis, subj01. Gate: n_steps>1 beats n_steps=1 with
non-overlapping bootstrap CI (spec §10).

Order: E0 -> E1 -> E2 are gates; E3 only after all three pass; E4-E8 only after E3 (spec §10).
"""
import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

from ..basis.graph import build_voxel_graph, connected_components
from ..basis.laplacian import LaplacianBasis, eigenmodes_per_component
from ..data.nsd_loader import load_betas, load_dinov2
from ..data.split import average_repeats, make_split
from ..eval.stats import bootstrap_ci
from ..model.encoder import DualPathEncoder
from ..model.refiner import Refiner
from ..model.schedule import fixed_schedule
from ..train.train_refiner import run

OUT_ROOT = Path("chasmbrain_v3_artifacts/experiments/e3_steps_ablation")
CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "model.yaml"))


def build_basis(subj: int, K: int = 512):
    mask = nib.load(f"/mnt/HDD/dataset/NSD_raw/nsddata/ppdata/subj{subj:02d}/func1pt8mm/roi/nsdgeneral.nii.gz").get_fdata() == 1
    coords = np.stack(np.where(mask), axis=1).astype(np.int32)
    A = build_voxel_graph(mask, coords, connectivity=26)
    labels = connected_components(A)
    lambdas, U = eigenmodes_per_component(A, labels, K=K)
    return LaplacianBasis(U, lambdas)


def load_subject_data(subj: int):
    betas_tr = np.asarray(load_betas(subj, "train"))
    X_tr_all = average_repeats(betas_tr)                 # [9000, n_voxels]
    cls_tr_all, patch_tr_all = load_dinov2(subj, "train", layer=-1)
    betas_te = np.asarray(load_betas(subj, "test"))
    X_test = average_repeats(betas_te)
    cls_test, patch_test = load_dinov2(subj, "test", layer=-1)
    split = make_split(subj)
    return {
        "X_tr_all": X_tr_all, "cls_tr_all": cls_tr_all, "patch_tr_all": patch_tr_all,
        "X_test": X_test, "cls_test": cls_test, "patch_test": patch_test,
        "train_idx": split["train"], "val_idx": split["val"],
    }


def run_one(n_steps: int, subj: int, sub, basis, device, max_epochs, patience, d_hidden, t_embed_dim):
    schedule = fixed_schedule(n_steps, kind="log")
    encoder = DualPathEncoder(sub["cls_tr_all"].shape[1], d_hidden, n_patches=sub["patch_tr_all"].shape[1])
    model = Refiner(encoder, basis, d_hidden=d_hidden, t_embed_dim=t_embed_dim).to(device)

    tr, val = sub["train_idx"], sub["val_idx"]
    X_tr_coef = sub["X_tr_all"][tr] @ basis.U            # [n_tr, K] full coefficients
    data = {
        "cls_tr": sub["cls_tr_all"][tr], "patch_tr": sub["patch_tr_all"][tr], "X_tr": X_tr_coef,
        "cls_val": sub["cls_tr_all"][val], "patch_val": sub["patch_tr_all"][val], "X_val": sub["X_tr_all"][val],
        "schedule": schedule,
    }
    out_dir = OUT_ROOT / f"subj{subj:02d}_n{n_steps}"
    model, metrics = run(model, basis, data, {**CFG, "n_steps": n_steps}, out_dir, device=device,
                          max_epochs=max_epochs, patience=patience, log_every=5)

    # test predictions, written once (spec §6)
    model.eval()
    with torch.no_grad():
        cls_t = torch.tensor(sub["cls_test"], dtype=torch.float32, device=device)
        patch_t = torch.tensor(sub["patch_test"].astype(np.float32), device=device)
        c = torch.zeros(cls_t.shape[0], 0, device=device, dtype=cls_t.dtype)
        t_prev = 0.0
        for t in schedule:
            c = model.step(cls_t, patch_t, c, t_prev, t)
            t_prev = t
        v_pred_test = c.cpu().numpy() @ basis.U[:, : c.shape[1]].T
    np.save(out_dir / "preds.npy", v_pred_test)
    ci = bootstrap_ci(v_pred_test, sub["X_test"], n=2000)
    json.dump(ci, open(out_dir / "test_bootstrap_ci.json", "w"), indent=1)
    return {"n_steps": n_steps, "schedule": schedule, "best_val_pearson": metrics["best_val_pearson"],
            "best_epoch": metrics["best_epoch"], "test_bootstrap_ci": ci}


def main(subj: int, steps_list, max_epochs: int, patience: int, d_hidden: int, t_embed_dim: int, K: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    basis = build_basis(subj, K=K)
    sub = load_subject_data(subj)
    print(f"subj{subj}: basis K={basis.K}, train={len(sub['train_idx'])}, val={len(sub['val_idx'])}, "
          f"test={sub['X_test'].shape[0]}  (setup {time.time()-t0:.0f}s)")

    results = []
    for n_steps in steps_list:
        t1 = time.time()
        res = run_one(n_steps, subj, sub, basis, device, max_epochs, patience, d_hidden, t_embed_dim)
        print(f"n_steps={n_steps}: val_pearson={res['best_val_pearson']:.4f} "
              f"test_pearson={res['test_bootstrap_ci']['pearson_mean']:.4f} "
              f"CI={res['test_bootstrap_ci']['pearson_ci95']}  ({time.time()-t1:.0f}s)")
        results.append(res)

    baseline = next(r for r in results if r["n_steps"] == 1)
    gate_results = []
    for r in results:
        if r["n_steps"] == 1:
            continue
        lo, hi = r["test_bootstrap_ci"]["pearson_ci95"]
        base_lo, base_hi = baseline["test_bootstrap_ci"]["pearson_ci95"]
        passed = lo > base_hi   # non-overlapping CI, multi-step strictly above single-shot
        gate_results.append({"n_steps": r["n_steps"], "beats_single_shot_ci_nonoverlap": passed})

    summary = {"subj": subj, "results": results, "gate": {"baseline_n_steps_1": baseline, "comparisons": gate_results},
               "gate_passed": any(g["beats_single_shot_ci_nonoverlap"] for g in gate_results)}
    json.dump(summary, open(OUT_ROOT / f"subj{subj:02d}_summary.json", "w"), indent=1)
    print(f"\nE1 gate (n_steps>1 beats n_steps=1, non-overlapping bootstrap CI): "
          f"{'PASS' if summary['gate_passed'] else 'FAIL'}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subj", type=int, default=1)
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--max_epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--d_hidden", type=int, default=256)
    p.add_argument("--t_embed_dim", type=int, default=32)
    p.add_argument("--K", type=int, default=512)
    a = p.parse_args()
    main(a.subj, a.steps, a.max_epochs, a.patience, a.d_hidden, a.t_embed_dim, a.K)

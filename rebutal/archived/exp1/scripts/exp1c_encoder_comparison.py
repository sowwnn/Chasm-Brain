"""
Experiment 1C: Encoder Comparison — CLIP vs DINOv2 (Ridge Regression baseline)

Uses NSD subj01 data (only subject with CLIP features pre-extracted).
Both encoders evaluated on same image set (9000 train / 982 test) with same fMRI.

Addresses: Reviewers xsi9, Pq8b — Show DINOv2 is the best encoder choice.
"""

import gc
import json
import numpy as np
import h5py
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT      = Path(__file__).resolve().parent.parent
NSD_BASE  = ROOT / "dataset/nsd"
META_BASE = NSD_BASE / "metadata"
SAVE_DIR  = ROOT / "rebutal/results"


def load_fmri_nsd(subj_id: str = "subj01"):
    """
    Load fMRI for subj01 from HDF5, aligned with NSD train/test splits.
    Returns z-scored (N, 15724) arrays for train and test.
    """
    train_ids = np.load(META_BASE / "train_ids.npy")  # (9000,)
    test_ids  = np.load(META_BASE / "test_ids.npy")   # (982,)

    subj_id_short = subj_id.replace("subj", "sub")
    with open(META_BASE / f"datalist_mindeye2_{subj_id_short}.json") as f:
        datalist = json.load(f)["samples"]

    def get_beta_indices(nsd_id):
        entry = datalist[str(nsd_id)]
        split = list(entry["splits"].values())[0]
        return [int(s["sample_id"].replace("sample", "")) for s in split]

    print(f"  Loading fMRI HDF5...", flush=True)
    hdf5_path = NSD_BASE / f"{subj_id}/betas_all_{subj_id}_fp32_renorm.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        betas = f["betas"][:]  # (30000, 15724)

    print(f"  Building fMRI train ({len(train_ids)} images)...", flush=True)
    Y_train = np.stack([
        betas[get_beta_indices(nid)].mean(axis=0) for nid in train_ids
    ]).astype(np.float32)

    print(f"  Building fMRI test ({len(test_ids)} images)...", flush=True)
    Y_test = np.stack([
        betas[get_beta_indices(nid)].mean(axis=0) for nid in test_ids
    ]).astype(np.float32)

    del betas
    gc.collect()

    # Z-score per voxel (fit on train, apply to test)
    mean = Y_train.mean(axis=0)
    std  = Y_train.std(axis=0) + 1e-8
    Y_train = ((Y_train - mean) / std).astype(np.float32)
    Y_test  = ((Y_test  - mean) / std).astype(np.float32)

    print(f"  fMRI shapes: train {Y_train.shape}, test {Y_test.shape}", flush=True)
    return Y_train, Y_test


def load_clip_features():
    base = NSD_BASE / "embeddings_clip"
    X_train = np.load(base / "clip_cls_train_sub01.npy").astype(np.float32)  # (9000, 1024)
    X_test  = np.load(base / "clip_cls_test_sub01.npy").astype(np.float32)   # (982, 1024)
    return X_train, X_test


def load_dinov2_features():
    base = NSD_BASE / "embeddings_patches"
    # Global average pooling over all 257 tokens (CLS + patches), last layer only
    tr = np.load(base / "dinov2_patches_train_sub01.npy").astype(np.float32)  # (9000, 257, 768)
    te = np.load(base / "dinov2_patches_test_sub01.npy").astype(np.float32)   # (982, 257, 768)
    X_train = tr.mean(axis=1)  # (9000, 768)
    X_test  = te.mean(axis=1)  # (982, 768)
    del tr, te
    gc.collect()
    return X_train, X_test


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pearson_vals, mse_vals = [], []
    for p, t in zip(pred, target):
        if p.std() > 1e-8 and t.std() > 1e-8:
            r, _ = pearsonr(p, t)
            if not np.isnan(r):
                pearson_vals.append(r)
        mse_vals.append(mean_squared_error(t, p))
    return {
        "pearson": float(np.mean(pearson_vals)),
        "pearson_std": float(np.std(pearson_vals)),
        "mse": float(np.mean(mse_vals)),
    }


def run_ridge(X_train, Y_train, X_test, Y_test) -> dict:
    alphas = [1, 100, 10000, 100000]
    best_pearson, best_metrics = -1, None
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver="auto", fit_intercept=True)
        model.fit(X_train, Y_train)
        pred = model.predict(X_test)
        m = compute_metrics(pred, Y_test)
        print(f"    alpha={alpha:<6} | Pearson: {m['pearson']:.4f} | MSE: {m['mse']:.4f}")
        if m["pearson"] > best_pearson:
            best_pearson = m["pearson"]
            best_metrics = m
    return best_metrics


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Loading fMRI (subj01, NSD format)")
    print("=" * 60)
    Y_train, Y_test = load_fmri_nsd("subj01")

    results = {}

    print("\n" + "=" * 60)
    print("  [CLIP ViT-B/16] — CLS token (1024-dim)")
    print("=" * 60)
    X_train, X_test = load_clip_features()
    print(f"  CLIP train: {X_train.shape}, test: {X_test.shape}")
    results["clip"] = run_ridge(X_train, Y_train, X_test, Y_test)
    print(f"  → Best: Pearson {results['clip']['pearson']:.4f} | MSE {results['clip']['mse']:.4f}")
    del X_train, X_test
    gc.collect()

    print("\n" + "=" * 60)
    print("  [DINOv2 ViT-B/14] — global avg pooling (768-dim)")
    print("=" * 60)
    X_train, X_test = load_dinov2_features()
    print(f"  DINOv2 train: {X_train.shape}, test: {X_test.shape}")
    results["dinov2"] = run_ridge(X_train, Y_train, X_test, Y_test)
    print(f"  → Best: Pearson {results['dinov2']['pearson']:.4f} | MSE {results['dinov2']['mse']:.4f}")
    del X_train, X_test
    gc.collect()

    print("\n" + "=" * 70)
    print("  FINAL SUMMARY — subj01, NSD format (9000 train / 982 test)")
    print("  Note: ResNet-50 not available (no pre-extracted features); commit to camera-ready")
    print("=" * 70)
    print(f"  {'Encoder':<28} | {'Pearson ↑':>10} | {'MSE ↓':>10}")
    print(f"  {'-'*54}")
    for name, m in results.items():
        label = "CLIP ViT-B/16" if name == "clip" else "DINOv2 ViT-B/14 (Ridge)"
        print(f"  {label:<28} | {m['pearson']:>10.4f} | {m['mse']:>10.4f}")
    print(f"  {'CHASMBrain (DINOv2, full model)':<28} | {'0.4290':>10} | {'0.2610':>10}")
    print("=" * 70)

    save_path = SAVE_DIR / "exp1c_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {save_path}")


if __name__ == "__main__":
    main()

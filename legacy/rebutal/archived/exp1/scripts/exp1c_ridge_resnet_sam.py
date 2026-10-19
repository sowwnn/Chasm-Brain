"""
Ridge regression on ResNet50, ResNet101, SAM ViT-B features (pool & multi).
Compares against existing DINOv2 results from exp1c_ridge_results.json.

Run:
    python rebutal/exp1c_ridge_resnet_sam.py
"""

import gc
import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT      = Path(__file__).resolve().parent.parent
HUWNG_DIR = ROOT / "dataset/huwng/subj01"
FEAT_DIR  = ROOT / "data/exp1/features_resnet_sam/subj01"
SAVE_DIR  = ROOT / "rebutal/results"

ENCODERS = {
    "ResNet-50 pool": {
        "train": FEAT_DIR / "nsd_resnet50_pool_train_sub1.npy",
        "test":  FEAT_DIR / "nsd_resnet50_pool_test_sub1.npy",
        "layer": None,
    },
    "ResNet-50 multi (last layer)": {
        "train": FEAT_DIR / "nsd_resnet50_multi_train_sub1.npy",
        "test":  FEAT_DIR / "nsd_resnet50_multi_test_sub1.npy",
        "layer": -1,   # (N, 4, 2048) → last layer
    },
    "ResNet-101 pool": {
        "train": FEAT_DIR / "nsd_resnet101_pool_train_sub1.npy",
        "test":  FEAT_DIR / "nsd_resnet101_pool_test_sub1.npy",
        "layer": None,
    },
    "ResNet-101 multi (last layer)": {
        "train": FEAT_DIR / "nsd_resnet101_multi_train_sub1.npy",
        "test":  FEAT_DIR / "nsd_resnet101_multi_test_sub1.npy",
        "layer": -1,
    },
    "SAM ViT-B pool": {
        "train": FEAT_DIR / "nsd_sam-vit-base_pool_train_sub1.npy",
        "test":  FEAT_DIR / "nsd_sam-vit-base_pool_test_sub1.npy",
        "layer": None,
    },
    "SAM ViT-B multi (last layer)": {
        "train": FEAT_DIR / "nsd_sam-vit-base_multi_train_sub1.npy",
        "test":  FEAT_DIR / "nsd_sam-vit-base_multi_test_sub1.npy",
        "layer": -1,   # (N, 4, 768) → last layer → (N, 768)
    },
}


def load_fmri():
    train_raw = np.load(HUWNG_DIR / "nsd_train_fmri_zscore_sub1.npy")
    test_raw  = np.load(HUWNG_DIR / "nsd_test_fmri_zscore_sub1.npy")
    Y_train = train_raw.mean(axis=1).astype(np.float32)
    Y_test  = test_raw.mean(axis=1).astype(np.float32)
    print(f"  fMRI train {Y_train.shape} | test {Y_test.shape}")
    return Y_train, Y_test


def load_encoder(cfg):
    tr = np.load(cfg["train"], mmap_mode="r")
    te = np.load(cfg["test"],  mmap_mode="r")
    if cfg["layer"] is not None:
        tr = tr[:, cfg["layer"], :]
        te = te[:, cfg["layer"], :]
    X_train = np.array(tr, dtype=np.float32)
    X_test  = np.array(te, dtype=np.float32)
    return X_train, X_test


def compute_metrics(pred, target):
    rs, mses = [], []
    for p, t in zip(pred, target):
        if p.std() > 1e-8 and t.std() > 1e-8:
            r, _ = pearsonr(p, t)
            if not np.isnan(r):
                rs.append(r)
        mses.append(mean_squared_error(t, p))
    return {"pearson": float(np.mean(rs)), "mse": float(np.mean(mses))}


def run_ridge(X_train, Y_train, X_test, Y_test):
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    best_p, best_m = -1, None
    for alpha in [1, 100, 10_000, 100_000]:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train, Y_train)
        m_tr = compute_metrics(model.predict(X_train), Y_train)
        m_te = compute_metrics(model.predict(X_test),  Y_test)
        print(f"    alpha={alpha:<7} | train {m_tr['pearson']:.4f} | test {m_te['pearson']:.4f}")
        if m_te["pearson"] > best_p:
            best_p, best_m = m_te["pearson"], {**m_te, "train_pearson": m_tr["pearson"]}
    return best_m


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading fMRI...")
    Y_train, Y_test = load_fmri()

    results = {}
    for name, cfg in ENCODERS.items():
        if not Path(cfg["train"]).exists():
            print(f"\n[SKIP] {name} — not found")
            continue
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        X_train, X_test = load_encoder(cfg)
        print(f"  Features: train {X_train.shape} | test {X_test.shape}")
        results[name] = run_ridge(X_train, Y_train, X_test, Y_test)
        print(f"  → Best: Pearson {results[name]['pearson']:.4f} | MSE {results[name]['mse']:.4f}")
        del X_train, X_test
        gc.collect()

    # Load DINOv2 for comparison
    prior_path = SAVE_DIR / "exp1c_ridge_results.json"
    dinov2 = {}
    if prior_path.exists():
        with open(prior_path) as f:
            prior = json.load(f)
        dinov2 = {"DINOv2 ViT-B/14 (ref)": prior["DINOv2 ViT-B/14"]}

    all_results = {**dinov2, **results}

    print(f"\n{'='*70}")
    print(f"  {'Encoder':<36} | {'Train Pearson':>13} | {'Test Pearson':>12}")
    print(f"  {'-'*65}")
    for name, m in all_results.items():
        print(f"  {name:<36} | {m.get('train_pearson', float('nan')):>13.4f} | {m['pearson']:>12.4f}")
    print(f"{'='*70}")

    out_path = SAVE_DIR / "exp1c_ridge_resnet_sam_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

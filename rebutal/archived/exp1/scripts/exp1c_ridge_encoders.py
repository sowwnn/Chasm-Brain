"""
EXP 1C (Step 1): Ridge Regression baseline for CLIP, ResNet-50, DINOv2.

Quick sanity check before training full CHASMBrain with each encoder.
Uses HUWNG dataset (9000 train / 1000 test, subj01).

Run:
    python rebutal/exp1c_ridge_encoders.py
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
CLIP_DIR  = ROOT / "dataset/clip/clip_features/subj01"
SAVE_DIR  = ROOT / "rebutal/results"


def load_fmri():
    """Return z-scored fMRI arrays (N, 15724) averaged over 3 trials."""
    train_raw = np.load(HUWNG_DIR / "nsd_train_fmri_zscore_sub1.npy")  # (9000, 3, 15724)
    test_raw  = np.load(HUWNG_DIR / "nsd_test_fmri_zscore_sub1.npy")   # (1000, 3, 15724)
    Y_train = train_raw.mean(axis=1).astype(np.float32)   # (9000, 15724)
    Y_test  = test_raw.mean(axis=1).astype(np.float32)    # (1000, 15724)
    print(f"  fMRI train {Y_train.shape} | test {Y_test.shape}")
    return Y_train, Y_test


def global_avg_pool(emb: np.ndarray) -> np.ndarray:
    """(N, T, D) → (N, D) via mean over token dim."""
    if emb.ndim == 3:
        return emb.mean(axis=1)
    return emb  # already (N, D)


ENCODERS = {
    "DINOv2 ViT-B/14": {
        "train": HUWNG_DIR / "nsd_dinov2_vitb14_multilayer_train_sub1.npy",
        "test":  HUWNG_DIR / "nsd_dinov2_vitb14_multilayer_test_sub1.npy",
        "layer": -1,   # multilayer (N,4,257,768) — use last layer
    },
    "CLIP ViT-B/16": {
        "train": HUWNG_DIR / "nsd_clip_vitb16_train_sub1.npy",
        "test":  HUWNG_DIR / "nsd_clip_vitb16_test_sub1.npy",
        "layer": None,
    },
    "ResNet-50": {
        "train": HUWNG_DIR / "nsd_resnet50_train_sub1.npy",
        "test":  HUWNG_DIR / "nsd_resnet50_test_sub1.npy",
        "layer": None,
    },
    # SDXL CLIP features (new)
    "CLIP SDXL pool": {
        "train": CLIP_DIR / "nsd_sdxl_clip_pool_train_sub1.npy",
        "test":  CLIP_DIR / "nsd_sdxl_clip_pool_test_sub1.npy",
        "layer": None,   # already (N, 1280)
    },
    "CLIP SDXL patch": {
        "train": CLIP_DIR / "nsd_sdxl_clip_train_sub1.npy",
        "test":  CLIP_DIR / "nsd_sdxl_clip_test_sub1.npy",
        "layer": None,   # (N, 256, 1664) → avg pool → (N, 1664)
    },
    "CLIP SDXL multi (last layer)": {
        "train": CLIP_DIR / "nsd_sdxl_clip_multi_train_sub1.npy",
        "test":  CLIP_DIR / "nsd_sdxl_clip_multi_test_sub1.npy",
        "layer": -1,     # (N, 4, 1664) → use last layer → (N, 1664)
    },
    "ConvNeXT-Large": {
        "train": HUWNG_DIR / "nsd_convnext_large_train_sub1.npy",
        "test":  HUWNG_DIR / "nsd_convnext_large_test_sub1.npy",
        "layer": None,   # (N, 1536)
    },
    "SAM ViT-B": {
        "train": HUWNG_DIR / "nsd_sam_vitb_train_sub1.npy",
        "test":  HUWNG_DIR / "nsd_sam_vitb_test_sub1.npy",
        "layer": None,   # (N, 256)
    },
}


def load_encoder(cfg: dict):
    tr = np.load(cfg["train"], mmap_mode="r")
    te = np.load(cfg["test"],  mmap_mode="r")
    if cfg["layer"] is not None:
        if tr.ndim == 4:
            # (N, L, T, D) → (N, T, D)
            tr = tr[:, cfg["layer"], :, :]
            te = te[:, cfg["layer"], :, :]
        elif tr.ndim == 3:
            # (N, L, D) → (N, D)
            tr = tr[:, cfg["layer"], :]
            te = te[:, cfg["layer"], :]
    X_train = global_avg_pool(np.array(tr, dtype=np.float32))
    X_test  = global_avg_pool(np.array(te, dtype=np.float32))
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
        print(f"    alpha={alpha:<7} | train Pearson {m_tr['pearson']:.4f} | test Pearson {m_te['pearson']:.4f}")
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
            print(f"\n[SKIP] {name} — features not found, run extract_encoder_features.py first")
            continue
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        X_train, X_test = load_encoder(cfg)
        print(f"  Features: train {X_train.shape} | test {X_test.shape}")
        results[name] = run_ridge(X_train, Y_train, X_test, Y_test)
        print(f"  → Best: Pearson {results[name]['pearson']:.4f} | MSE {results[name]['mse']:.4f}")
        del X_train, X_test
        gc.collect()

    print(f"\n{'='*70}")
    print(f"  {'Encoder':<28} | {'Train Pearson':>13} | {'Test Pearson':>12}")
    print(f"  {'-'*60}")
    for name, m in results.items():
        print(f"  {name:<28} | {m.get('train_pearson', float('nan')):>13.4f} | {m['pearson']:>12.4f}")
    print(f"{'='*70}")

    out_path = SAVE_DIR / "exp1c_ridge_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

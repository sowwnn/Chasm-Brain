"""
Ridge Regression baseline — DINOv2 ViT-B/14 multilayer, all 4 subjects.

Feature: (N, 4, 257, 768) → layer -1 → (N, 257, 768) → avg pool → (N, 768)
fMRI:    (N, 3, V) → mean over trials → (N, V)
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT     = Path(__file__).resolve().parent.parent.parent.parent
HUWNG    = ROOT / "dataset/huwng"
SAVE_DIR = ROOT / "rebutal/exp1/results"

SUBJECTS = {
    1: {"subj_dir": "subj01", "suffix": "sub1", "fmri_dim": 15724},
    2: {"subj_dir": "subj02", "suffix": "sub2", "fmri_dim": 12682},
    5: {"subj_dir": "subj05", "suffix": "sub5", "fmri_dim": 13039},
    7: {"subj_dir": "subj07", "suffix": "sub7", "fmri_dim": 14278},
}

ALPHAS = [1, 100, 10_000, 100_000]


def load_data(subj_id):
    cfg    = SUBJECTS[subj_id]
    d      = HUWNG / cfg["subj_dir"]
    sfx    = cfg["suffix"]

    emb_tr = np.load(d / f"nsd_dinov2_vitb14_multilayer_train_{sfx}.npy", mmap_mode="r")
    emb_te = np.load(d / f"nsd_dinov2_vitb14_multilayer_test_{sfx}.npy",  mmap_mode="r")

    # layer -1, avg pool over 257 tokens → (N, 768)
    X_tr = np.array(emb_tr[:, -1, :, :], dtype=np.float32).mean(axis=1)
    X_te = np.array(emb_te[:, -1, :, :], dtype=np.float32).mean(axis=1)

    fmri_tr = np.load(d / f"nsd_train_fmri_zscore_{sfx}.npy", mmap_mode="r")
    fmri_te = np.load(d / f"nsd_test_fmri_zscore_{sfx}.npy",  mmap_mode="r")

    # mean over 3 trials
    Y_tr = np.array(fmri_tr.mean(axis=1), dtype=np.float32)
    Y_te = np.array(fmri_te.mean(axis=1), dtype=np.float32)

    return X_tr, X_te, Y_tr, Y_te


def pearson_mean(pred, target):
    rs = []
    for p, t in zip(pred, target):
        if p.std() > 1e-8 and t.std() > 1e-8:
            r, _ = pearsonr(p, t)
            if np.isfinite(r):
                rs.append(r)
    return float(np.mean(rs)) if rs else 0.0


def run_ridge(X_tr, Y_tr, X_te, Y_te):
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    best_p, best_result = -1, None
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_tr, Y_tr)
        p_tr = pearson_mean(model.predict(X_tr), Y_tr)
        p_te = pearson_mean(model.predict(X_te), Y_te)
        print(f"    alpha={alpha:<7} | train={p_tr:.4f} | test={p_te:.4f}")
        if p_te > best_p:
            best_p      = p_te
            best_result = {"alpha": alpha, "train_pearson": p_tr, "test_pearson": p_te}

    return best_result


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for subj_id in SUBJECTS:
        print(f"\n{'='*55}")
        print(f"  Subject {subj_id}  (fmri_dim={SUBJECTS[subj_id]['fmri_dim']})")
        print(f"{'='*55}")
        X_tr, X_te, Y_tr, Y_te = load_data(subj_id)
        print(f"  X: train={X_tr.shape} test={X_te.shape}")
        print(f"  Y: train={Y_tr.shape} test={Y_te.shape}")
        result = run_ridge(X_tr, Y_tr, X_te, Y_te)
        all_results[f"subj{subj_id:02d}"] = result
        print(f"  → Best alpha={result['alpha']} | test Pearson={result['test_pearson']:.4f}")

    print(f"\n{'='*55}")
    print("  SUMMARY — Ridge + DINOv2 ViT-B/14 (layer -1, avg pool)")
    print(f"{'='*55}")
    print(f"  {'Subject':<10} | {'Test Pearson':>12} | {'Train Pearson':>13}")
    print(f"  {'-'*40}")
    for subj, r in all_results.items():
        print(f"  {subj:<10} | {r['test_pearson']:>12.4f} | {r['train_pearson']:>13.4f}")

    save_path = SAVE_DIR / "ridge_dinov2_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {save_path}")


if __name__ == "__main__":
    main()

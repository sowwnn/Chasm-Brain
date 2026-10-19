"""
Ridge baseline with the coarse-to-fine two-stage structure.

Stage 1: DINOv2 (layer -1, GAP over 257 tokens) -> sequential cluster means
         (30 voxels/cluster, top-70% aggregation, matching NeuroFluxDatasetV7H).
Stage 2: [DINOv2, cluster guidance] -> voxels.

Variants per subject:
  one_stage            X -> Y
  two_stage_predicted  [X, Stage-1 prediction] -> Y   (out-of-fold guidance at fit time)
  two_stage_oracle     [X, ground-truth cluster means] -> Y   (upper bound)

Alphas are selected on a held-out validation split, never on test.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HUWNG = ROOT / "dataset/huwng"

SUBJECTS = {
    1: "sub1",
    2: "sub2",
    5: "sub5",
    7: "sub7",
}

ALPHAS = [1.0, 1e2, 1e3, 1e4, 1e5, 1e6]
VOXELS_PER_CLUSTER = 30
ROI_TOP_K_PERCENT = 0.7
N_VAL = 1000
N_FOLDS = 5


def load_subject(subj_id):
    sfx = SUBJECTS[subj_id]
    d = HUWNG / f"subj{subj_id:02d}"

    emb_tr = np.load(d / f"nsd_dinov2_vitb14_multilayer_train_{sfx}.npy", mmap_mode="r")
    emb_te = np.load(d / f"nsd_dinov2_vitb14_multilayer_test_{sfx}.npy", mmap_mode="r")
    X_tr = np.array(emb_tr[:, -1, :, :], dtype=np.float32).mean(axis=1)
    X_te = np.array(emb_te[:, -1, :, :], dtype=np.float32).mean(axis=1)

    fmri_tr = np.load(d / f"nsd_train_fmri_zscore_{sfx}.npy", mmap_mode="r")
    fmri_te = np.load(d / f"nsd_test_fmri_zscore_{sfx}.npy", mmap_mode="r")
    Y_tr = np.array(fmri_tr.mean(axis=1), dtype=np.float32)
    Y_te = np.array(fmri_te.mean(axis=1), dtype=np.float32)

    return X_tr, Y_tr, X_te, Y_te


def compute_roi_means(fmri):
    """Sequential 30-voxel clusters aggregated by top-70% mean."""
    n, v = fmri.shape
    num_rois = v // VOXELS_PER_CLUSTER
    clusters = fmri[:, : num_rois * VOXELS_PER_CLUSTER].reshape(n, num_rois, VOXELS_PER_CLUSTER)
    k = max(1, int(VOXELS_PER_CLUSTER * ROI_TOP_K_PERCENT))
    top_k = np.partition(clusters, -k, axis=-1)[..., -k:]
    return top_k.mean(axis=-1).astype(np.float32)


def ridge_fit_predict(X_fit, Y_fit, alphas, X_eval_list):
    """Ridge with intercept for a list of alphas; returns {alpha: [pred per eval set]}."""
    x_mean = X_fit.mean(axis=0)
    x_std = X_fit.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_mean = Y_fit.mean(axis=0)

    Xc = ((X_fit - x_mean) / x_std).astype(np.float64)
    Yc = (Y_fit - y_mean).astype(np.float64)

    gram = Xc.T @ Xc
    xty = Xc.T @ Yc
    eye = np.eye(gram.shape[0])

    evals = [((Xe - x_mean) / x_std).astype(np.float64) for Xe in X_eval_list]

    out = {}
    for alpha in alphas:
        w = np.linalg.solve(gram + alpha * eye, xty)
        out[alpha] = [(Xe @ w + y_mean).astype(np.float32) for Xe in evals]
    return out


def upsample_clusters(clusters, num_voxels):
    """Stage-1-only readout: broadcast each cluster value to its 30 voxels.

    The trailing remainder voxels have no cluster; they take the per-image mean.
    """
    n, num_rois = clusters.shape
    out = np.repeat(clusters, VOXELS_PER_CLUSTER, axis=1)
    remainder = num_voxels - num_rois * VOXELS_PER_CLUSTER
    if remainder > 0:
        fill = np.repeat(clusters.mean(axis=1, keepdims=True), remainder, axis=1)
        out = np.concatenate([out, fill], axis=1)
    return out.astype(np.float32)


def sample_pearson(pred, target):
    """Per-image Pearson across voxels, averaged over images."""
    p = pred - pred.mean(axis=1, keepdims=True)
    t = target - target.mean(axis=1, keepdims=True)
    pn = np.linalg.norm(p, axis=1)
    tn = np.linalg.norm(t, axis=1)
    valid = (pn > 1e-8) & (tn > 1e-8)
    r = np.sum(p[valid] * t[valid], axis=1) / (pn[valid] * tn[valid])
    return r[np.isfinite(r)]


def metrics(pred, target):
    r = sample_pearson(pred, target)
    mse = float(np.mean((pred - target) ** 2))
    return float(r.mean()), mse


def select_alpha(preds_by_alpha, target, eval_index=0):
    best_alpha, best_r = None, -np.inf
    for alpha, preds in preds_by_alpha.items():
        r, _ = metrics(preds[eval_index], target)
        if r > best_r:
            best_alpha, best_r = alpha, r
    return best_alpha, best_r


def oof_stage1(X_fit, R_fit, alpha, rng):
    """Out-of-fold Stage-1 predictions so Stage 2 never sees in-sample guidance."""
    n = X_fit.shape[0]
    order = rng.permutation(n)
    folds = np.array_split(order, N_FOLDS)
    oof = np.zeros_like(R_fit)
    for held in folds:
        mask = np.ones(n, dtype=bool)
        mask[held] = False
        preds = ridge_fit_predict(X_fit[mask], R_fit[mask], [alpha], [X_fit[held]])
        oof[held] = preds[alpha][0]
    return oof


def run_subject(subj_id, seed):
    rng = np.random.default_rng(seed)
    X_tr, Y_tr, X_te, Y_te = load_subject(subj_id)

    perm = rng.permutation(X_tr.shape[0])
    val_idx, fit_idx = perm[:N_VAL], perm[N_VAL:]
    X_fit, Y_fit = X_tr[fit_idx], Y_tr[fit_idx]
    X_val, Y_val = X_tr[val_idx], Y_tr[val_idx]

    R_fit = compute_roi_means(Y_fit)
    R_val = compute_roi_means(Y_val)
    R_te = compute_roi_means(Y_te)

    result = {
        "subject": subj_id,
        "num_voxels": int(Y_tr.shape[1]),
        "num_clusters": int(R_fit.shape[1]),
        "n_fit": int(len(fit_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(X_te.shape[0]),
    }

    # --- one-stage reference -------------------------------------------------
    preds = ridge_fit_predict(X_fit, Y_fit, ALPHAS, [X_val, X_te])
    alpha1s, val_r = select_alpha(preds, Y_val)
    r, mse = metrics(preds[alpha1s][1], Y_te)
    result["one_stage"] = {"alpha": alpha1s, "val_pearson": val_r, "test_pearson": r, "test_mse": mse}
    print(f"  one_stage           alpha={alpha1s:<8.0f} val r={val_r:.4f}  test r={r:.4f}  mse={mse:.4f}")
    del preds

    # --- stage 1: features -> cluster means ----------------------------------
    s1 = ridge_fit_predict(X_fit, R_fit, ALPHAS, [X_val, X_te])
    alpha_s1, s1_val_r = select_alpha(s1, R_val)
    s1_r, s1_mse = metrics(s1[alpha_s1][1], R_te)
    result["stage1"] = {
        "alpha": alpha_s1,
        "val_pearson": s1_val_r,
        "test_pearson": s1_r,
        "test_mse": s1_mse,
    }
    print(f"  stage1 (clusters)   alpha={alpha_s1:<8.0f} val r={s1_val_r:.4f}  test r={s1_r:.4f}  mse={s1_mse:.4f}")
    G_val, G_te = s1[alpha_s1][0], s1[alpha_s1][1]
    del s1

    # --- stage-1-only readout: no learned refinement --------------------------
    num_voxels = Y_te.shape[1]
    for name, clusters in [
        ("stage1_only_predicted", G_te),
        ("stage1_only_oracle", R_te),
    ]:
        r, mse = metrics(upsample_clusters(clusters, num_voxels), Y_te)
        result[name] = {"test_pearson": r, "test_mse": mse}
        print(f"  {name:<20}{'':14} test r={r:.4f}  mse={mse:.4f}")

    # --- stage 2: [features, guidance] -> voxels ------------------------------
    G_fit = oof_stage1(X_fit, R_fit, alpha_s1, np.random.default_rng(seed + 1))

    for name, g_fit, g_val, g_te in [
        ("two_stage_predicted", G_fit, G_val, G_te),
        ("two_stage_oracle", R_fit, R_val, R_te),
    ]:
        Xf = np.concatenate([X_fit, g_fit], axis=1)
        Xv = np.concatenate([X_val, g_val], axis=1)
        Xt = np.concatenate([X_te, g_te], axis=1)
        preds = ridge_fit_predict(Xf, Y_fit, ALPHAS, [Xv, Xt])
        alpha, val_r = select_alpha(preds, Y_val)
        r, mse = metrics(preds[alpha][1], Y_te)
        result[name] = {"alpha": alpha, "val_pearson": val_r, "test_pearson": r, "test_mse": mse}
        print(f"  {name:<20}alpha={alpha:<8.0f} val r={val_r:.4f}  test r={r:.4f}  mse={mse:.4f}")
        del preds, Xf, Xv, Xt

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=list(SUBJECTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "results/wacv_round2/ridge_two_stage_seed42.json")
    args = parser.parse_args()

    per_subject = {}
    for subj_id in args.subjects:
        print(f"\n=== Subject {subj_id} ===")
        per_subject[f"subj{subj_id:02d}"] = run_subject(subj_id, args.seed)

    variants = [
        "one_stage",
        "stage1",
        "stage1_only_predicted",
        "stage1_only_oracle",
        "two_stage_predicted",
        "two_stage_oracle",
    ]
    aggregate = {}
    for variant in variants:
        rs = [per_subject[s][variant]["test_pearson"] for s in per_subject]
        ms = [per_subject[s][variant]["test_mse"] for s in per_subject]
        aggregate[variant] = {
            "pearson_mean": float(np.mean(rs)),
            "pearson_std": float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0,
            "mse_mean": float(np.mean(ms)),
            "mse_std": float(np.std(ms, ddof=1)) if len(ms) > 1 else 0.0,
        }

    out = {
        "seed": args.seed,
        "features": "DINOv2 ViT-B/14 layer -1, GAP over 257 tokens (768-d)",
        "clustering": f"{VOXELS_PER_CLUSTER} voxels/cluster, top-{int(ROI_TOP_K_PERCENT * 100)}% aggregation",
        "protocol": "9000 train images split into fit/val; alpha selected on val; 1000 held-out test images",
        "stage2_guidance": f"{N_FOLDS}-fold out-of-fold Stage-1 predictions at fit time",
        "alphas": ALPHAS,
        "per_subject": per_subject,
        "aggregate": aggregate,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== Aggregate over {len(per_subject)} subjects ===")
    for variant in variants:
        a = aggregate[variant]
        print(f"  {variant:<22} Pearson {a['pearson_mean']:.4f} +/- {a['pearson_std']:.4f}"
              f"   MSE {a['mse_mean']:.4f} +/- {a['mse_std']:.4f}")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()

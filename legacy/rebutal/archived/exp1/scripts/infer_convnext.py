"""
Inference: ConvNeXT-Large + HierarchicalARM_V3
Evaluates on huwng averaged fMRI (same as ridge) for fair comparison.
"""

import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from model.hierarchical_v3_1 import HierarchicalARM_V3

CONFIG_PATH  = ROOT / "configs/exp1c_convnext_large.yaml"
CKPT_PATH    = ROOT / "checkpoints/exp1c_convnext_large/best_model.pth"
FEAT_TEST    = ROOT / "dataset/nsd_raw_subj01/convnext_test.npy"
FMRI_TEST_RAW   = ROOT / "dataset/nsd_raw_subj01/fmri_test.npy"
FMRI_TEST_HUWNG = ROOT / "dataset/huwng/subj01/nsd_test_fmri_zscore_sub1.npy"
BATCH_SIZE   = 64


def compute_metrics(pred, target):
    rs, mses = [], []
    for p, t in zip(pred, target):
        if p.std() > 1e-8 and t.std() > 1e-8:
            r, _ = pearsonr(p, t)
            if not np.isnan(r):
                rs.append(r)
        mses.append(mean_squared_error(t, p))
    return float(np.mean(rs)), float(np.mean(mses))


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HierarchicalARM_V3(
        cls_dim=config["model"]["cls_dim"],
        patch_dim=config["model"]["patch_dim"],
        num_patches=config["model"]["num_patches"],
        fmri_dim=config["model"]["fmri_dim"],
        voxels_per_cluster=config["model"]["voxels_per_cluster"],
        hidden_dim_1=config["model"]["hidden_dim_1"],
        depth_what=config["model"]["depth_what"],
        depth_where=config["model"]["depth_where"],
        contrastive_dim=config["model"]["contrastive_dim"],
        embed_dim_2=config["model"]["embed_dim_2"],
        depth_2=config["model"]["depth_2"],
        num_global_queries=config["model"]["num_global_queries"],
        dropout=0.0,
    ).to(device)

    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {CKPT_PATH}")

    # Load features: (1000, 50, 1536) → cls=(1000,1536), patches=(1000,49,1536)
    feats = np.load(FEAT_TEST).astype(np.float32)
    cls_tokens   = feats[:, 0, :]       # (1000, 1536)
    patch_tokens = feats[:, 1:, :]      # (1000, 49, 1536)

    # Load raw fMRI (what model was trained on)
    Y_raw = np.load(FMRI_TEST_RAW).astype(np.float32)       # (1000, 15724)
    # Load huwng fMRI averaged (same as ridge)
    Y_huwng = np.load(FMRI_TEST_HUWNG).mean(axis=1).astype(np.float32)  # (1000, 15724)
    print(f"Features: {feats.shape} | fMRI raw: {Y_raw.shape} | fMRI huwng: {Y_huwng.shape}")

    all_preds = []
    N = len(cls_tokens)
    with torch.no_grad():
        for start in range(0, N, BATCH_SIZE):
            cls   = torch.from_numpy(cls_tokens[start:start+BATCH_SIZE]).to(device)
            patch = torch.from_numpy(patch_tokens[start:start+BATCH_SIZE]).to(device)
            _, pred, _ = model(cls, patch, stage=2)
            all_preds.append(pred.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)  # (1000, 15724)
    print(f"Predictions: {preds.shape}")

    p_raw,   mse_raw   = compute_metrics(preds, Y_raw)
    p_huwng, mse_huwng = compute_metrics(preds, Y_huwng)

    print(f"\n{'='*50}")
    print(f"  ConvNeXT-Large (HierarchicalARM_V3)")
    print(f"  vs raw fMRI (train target) : Pearson={p_raw:.4f}  MSE={mse_raw:.4f}")
    print(f"  vs huwng fMRI (same as ridge): Pearson={p_huwng:.4f}  MSE={mse_huwng:.4f}")
    print(f"{'='*50}")
    print(f"\nRidge baseline (huwng): Pearson=0.3148, MSE=0.4961")


if __name__ == "__main__":
    main()

"""Training loop (spec §6): sample (t_prev, t) per batch, teacher-force c_prev = project(v_true, t_prev) + N(0, 0.05),
penalise the new band. AdamW lr 3e-4, batch 64, early stopping on val (patience 15, max 150 epochs).
Writes config.yaml + metrics.json per run; preds.npy on test written once at the end."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..eval.metrics import voxel_pearson
from .losses import band_mse, voxel_pearson as voxel_pearson_torch


def sample_t_pair(schedule, continuous: bool = False, rng: np.random.Generator = None):
    """Sample one (t_prev, t) pair for a training batch. schedule: the fixed_schedule list (used to
    pick an adjacent pair when continuous=False). continuous=True ignores schedule and draws
    t_prev ~ U[0,1), t ~ U(t_prev, 1] (ablation, spec §6)."""
    rng = rng or np.random.default_rng()
    if continuous:
        t_prev = float(rng.uniform(0, 1))
        t = float(rng.uniform(t_prev, 1.0)) if t_prev < 1.0 else 1.0
        if t <= t_prev:
            t = min(1.0, t_prev + 1e-3)
        return t_prev, t
    i = rng.integers(0, len(schedule))
    t_prev = 0.0 if i == 0 else schedule[i - 1]
    return t_prev, schedule[i]


def make_batch(X_full_coef: np.ndarray, basis, t_prev: float, t: float, noise_sigma: float = 0.05,
               rng: np.random.Generator = None):
    """X_full_coef: [B, K] true coefficients at t=1 (already projected). Returns (c_prev [B, k(t_prev)]
    with Gaussian teacher-forcing noise, c_true_new [B, |band|] the clean target for this step)."""
    rng = rng or np.random.default_rng()
    kt_prev, kt = basis.k(t_prev), basis.k(t)
    c_prev = X_full_coef[:, :kt_prev].copy()
    if noise_sigma > 0 and kt_prev > 0:
        c_prev += rng.normal(0, noise_sigma, size=c_prev.shape).astype(c_prev.dtype)
    c_true_new = X_full_coef[:, kt_prev:kt]
    return c_prev, c_true_new


def scheduled_sampling(epoch: int, start_epoch: int = 50, p_max: float = 0.5, end_epoch: int = 150) -> float:
    """Probability of replacing teacher-forced c_prev with the model's own prediction. 0 before
    start_epoch, ramping linearly to p_max by end_epoch (spec §6: "sau epoch 50 ... tăng p từ 0 -> 0.5")."""
    if epoch < start_epoch:
        return 0.0
    return float(min(p_max, p_max * (epoch - start_epoch) / max(1, end_epoch - start_epoch)))


def _get_c_prev(model, cls, patch, X_full_coef_t, t_prev, t, p_scheduled, basis, device, rng_noise):
    """c_prev for one training step: with probability p_scheduled, roll out the model's own prediction
    from t=0 up to t_prev (no grad); otherwise teacher-force from the true coefficients."""
    kt_prev = basis.k(t_prev)
    if p_scheduled == 0.0 or np.random.default_rng().random() >= p_scheduled or kt_prev == 0:
        c_prev = X_full_coef_t[:, :kt_prev]
        if rng_noise > 0 and kt_prev > 0:
            c_prev = c_prev + torch.randn_like(c_prev) * rng_noise
        return c_prev
    with torch.no_grad():
        c = torch.zeros(cls.shape[0], 0, device=device, dtype=cls.dtype)
        tp = 0.0
        # Roll out on the SAME fixed schedule up to t_prev; this helper is only called with t_prev that
        # is itself a schedule value (or 0), so this terminates exactly at t_prev.
        for tt in model._rollout_schedule:
            if tt > t_prev:
                break
            c = model.step(cls, patch, c, tp, tt)
            tp = tt
    return c


def train_epoch(model, cls, patch, X_full_coef, basis, optim, cfg, epoch, device, rng, U_t=None, lambdas_t=None):
    """One epoch over the training set. cls/patch/X_full_coef are full numpy arrays (indexed per batch);
    cfg is the model.yaml dict. U_t/lambdas_t: basis.U/basis.lambdas as float32 tensors already on
    device (built once by the caller and reused across epochs/batches -- rebuilding them from numpy
    inside the batch loop was both the throughput bottleneck and a RAM leak during development: each
    batch allocated a fresh [n_voxels, K] float32 array on top of the autograd graph holding every
    batch's copy until backward(), which exhausted system RAM within a few hundred steps at K=512).
    Returns mean training loss for the epoch."""
    if U_t is None:
        U_t = torch.tensor(basis.U, dtype=torch.float32, device=device)
    if lambdas_t is None:
        lambdas_t = torch.tensor(basis.lambdas, dtype=torch.float32, device=device)

    N = cls.shape[0]
    idx_all = rng.permutation(N)
    batch_size = cfg["optim"]["batch_size"]
    schedule = model._rollout_schedule
    p_sched = scheduled_sampling(epoch, **cfg["scheduled_sampling"])
    losses = []
    for start in range(0, N - batch_size + 1, batch_size):
        idx = idx_all[start:start + batch_size]
        cls_b = torch.tensor(cls[idx], dtype=torch.float32, device=device)
        patch_b = torch.tensor(patch[idx].astype(np.float32), device=device)
        Xc_b = torch.tensor(X_full_coef[idx], dtype=torch.float32, device=device)

        t_prev, t = sample_t_pair(schedule, continuous=cfg.get("continuous", False), rng=rng)
        kt_prev, kt = basis.k(t_prev), basis.k(t)
        c_prev = _get_c_prev(model, cls_b, patch_b, Xc_b, t_prev, t, p_sched, basis, device,
                              cfg["teacher_noise_sigma"])
        c_true_new = Xc_b[:, kt_prev:kt]

        c_pred_new = model.forward(cls_b, patch_b, c_prev, t_prev, t)
        band_idx = basis.band(t_prev, t)
        loss_mse = band_mse(c_pred_new, c_true_new, lambdas_t[band_idx], weighting=cfg["band_weighting"])

        # Auxiliary loss term (spec §6): voxel_pearson(lift(c_full, t), v_true), using the FULL
        # coefficient vector up to t (c_prev + c_new), not just the new band -- lift is linear so this
        # is basis.U[:, :kt] @ c_full, computed via the pre-built U_t slice (no per-batch numpy copy).
        c_full_pred = torch.cat([c_prev, c_pred_new], dim=1)
        v_pred = c_full_pred @ U_t[:, :kt].T
        v_true = Xc_b[:, :kt] @ U_t[:, :kt].T
        loss_pearson = -voxel_pearson_torch(v_pred, v_true)
        loss = loss_mse + cfg["gamma_pearson"] * loss_pearson

        optim.zero_grad(); loss.backward(); optim.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def validate(model, cls, patch, X, basis, schedule, device, batch_size: int = 128):
    """Run the fixed schedule on the val set and return (mean voxel Pearson at t=1, full v_pred array)."""
    N = cls.shape[0]
    v_pred_chunks = []
    for start in range(0, N, batch_size):
        cls_b = torch.tensor(cls[start:start + batch_size], dtype=torch.float32, device=device)
        patch_b = torch.tensor(patch[start:start + batch_size].astype(np.float32), device=device)
        c = torch.zeros(cls_b.shape[0], 0, device=device, dtype=cls_b.dtype)
        t_prev = 0.0
        for t in schedule:
            c = model.step(cls_b, patch_b, c, t_prev, t)
            t_prev = t
        v_pred_chunks.append((c.cpu().numpy() @ basis.U[:, : c.shape[1]].T))
    v_pred = np.concatenate(v_pred_chunks, axis=0)
    r = voxel_pearson(v_pred, X)
    return float(np.nanmean(r)), v_pred


def run(model, basis, data, cfg, out_dir: Path, device: str = "cuda", max_epochs: int = None, patience: int = None,
        log_every: int = 1):
    """data: dict with cls_tr, patch_tr, X_tr (coefficients at t=1, [N,K]), cls_val, patch_val, X_val
    (voxel space, [N,n_voxels]), schedule (list of t values). Writes config.yaml + metrics.json to
    out_dir; preds.npy (test predictions) is the caller's responsibility once training is done, written
    exactly once (spec §6)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    max_epochs = max_epochs or cfg["optim"]["max_epochs"]
    patience = patience if patience is not None else cfg["optim"]["patience"]
    model._rollout_schedule = data["schedule"]
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"])
    rng = np.random.default_rng(0)
    U_t = torch.tensor(basis.U, dtype=torch.float32, device=device)
    lambdas_t = torch.tensor(basis.lambdas, dtype=torch.float32, device=device)

    yaml.safe_dump({**cfg, "schedule": data["schedule"]}, open(out_dir / "config.yaml", "w"))

    history = []
    best_val, best_epoch, best_state = -np.inf, -1, None
    t0 = time.time()
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, data["cls_tr"], data["patch_tr"], data["X_tr"], basis, optim, cfg, epoch, device, rng, U_t=U_t, lambdas_t=lambdas_t)
        val_pearson, _ = validate(model, data["cls_val"], data["patch_val"], data["X_val"], basis, data["schedule"], device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_pearson": val_pearson})
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_pearson={val_pearson:.4f}  ({time.time()-t0:.0f}s)")
        if val_pearson > best_val:
            best_val, best_epoch = val_pearson, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            print(f"early stopping at epoch {epoch} (best epoch {best_epoch}, val_pearson={best_val:.4f})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {"best_epoch": best_epoch, "best_val_pearson": best_val, "history": history, "wall_time_s": time.time() - t0}
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=1)
    return model, metrics


if __name__ == "__main__":
    raise NotImplementedError("use chasmbrain_v3/experiments/e*.py as entry points")

"""Train the direct Stage2-only DINOv2-to-voxel baseline."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.dataset_v7_h import create_dataloaders_v7h
from model.direct_stage2 import DirectStage2Only
from model.tben_official import TBEnOneStageOfficial


def build_one_stage_model(model_cfg):
    """Build the requested one-stage baseline; both return (prediction, kl)."""
    if model_cfg.get("name") == "TBEnOneStage":
        return TBEnOneStageOfficial(
            cls_dim=model_cfg["cls_dim"], patch_dim=model_cfg["patch_dim"],
            num_patches=model_cfg.get("num_patches", 256),
            fmri_dim=model_cfg["fmri_dim"],
            voxels_per_cluster=model_cfg["voxels_per_cluster"],
            hidden_dim=model_cfg.get("hidden_dim", 768),
            nheads=model_cfg.get("nheads", 16),
            dim_feedforward=model_cfg.get("dim_feedforward", 1024),
            enc_layers=model_cfg.get("enc_layers", 0),
            dec_layers=model_cfg.get("dec_layers", 1),
            dropout=model_cfg.get("dropout", 0.1),
            pre_norm=model_cfg.get("pre_norm", True),
            enc_output_layer=model_cfg.get("enc_output_layer", 1),
        )
    return DirectStage2Only(
        cls_dim=model_cfg["cls_dim"], patch_dim=model_cfg["patch_dim"],
        fmri_dim=model_cfg["fmri_dim"],
        voxels_per_cluster=model_cfg["voxels_per_cluster"],
        hidden_dim=model_cfg["hidden_dim"], embed_dim=model_cfg["embed_dim"],
        latent_dim=model_cfg.get("latent_dim", 128), depth=model_cfg["depth"],
        num_global_queries=model_cfg["num_global_queries"],
        dropout=model_cfg.get("dropout", 0.1),
        vae_mode=model_cfg.get("vae_mode", "stochastic"),
        backbone=model_cfg.get("backbone", "mamba"),
    )


def per_sample_pearson(prediction, target):
    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (prediction * target).sum(dim=1)
    denominator = prediction.square().sum(dim=1).sqrt() * target.square().sum(dim=1).sqrt()
    # Match the shared WACV-round2 evaluator exactly. Direct predictions can
    # be nearly constant, so the previous 1e-8 product clamp materially
    # changed checkpoint ranking even though MSE was identical.
    return numerator / denominator.clamp_min(1e-12)


def loss_components(prediction, target, kl_loss, config):
    loss_cfg = config["loss"]
    stability_eps = loss_cfg.get("stability_eps", 1e-6)
    mse = F.mse_loss(prediction, target)
    # Direct predictions can initially collapse to an almost constant vector.
    # Put epsilon inside the square root so Pearson/std derivatives remain
    # bounded in that regime; adding epsilon after std does not stabilize its
    # 1/sqrt(variance) backward term.
    pred_centered = prediction - prediction.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    pred_std = torch.sqrt(pred_centered.square().mean(dim=1, keepdim=True) + stability_eps)
    target_std = torch.sqrt(target_centered.square().mean(dim=1, keepdim=True) + stability_eps)
    pearson = (pred_centered * target_centered).mean(dim=1, keepdim=True) / (
        pred_std * target_std
    )
    pearson_loss = (1 - pearson).mean()
    std_loss = F.mse_loss(pred_std, target_std)
    total = (
        mse
        + loss_cfg.get("pearson_weight", 0.5) * pearson_loss
        + loss_cfg.get("std_weight", 0.2) * std_loss
        + loss_cfg.get("kl_weight", 0.001) * kl_loss
    )
    return total, mse, std_loss


def train_epoch(model, loader, optimizer, scaler, device, config, epoch):
    model.train()
    totals = {
        "loss": 0.0, "mse": 0.0, "std": 0.0,
        "pearson": 0.0, "grad_norm": 0.0,
    }
    successful_steps = 0
    skipped_nonfinite_steps = 0
    max_nonfinite_steps = config["training"].get("max_nonfinite_steps_per_epoch", 5)
    precision = config["training"].get("precision", "fp16")
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    autocast_enabled = precision in {"fp16", "bf16"}
    progress = tqdm(loader, desc=f"Direct Stage2 Train {epoch}")
    for batch_index, batch in enumerate(progress):
        cls_token = batch["cls_token"].to(device, non_blocking=True)
        patch_tokens = batch["patch_tokens"].to(device, non_blocking=True)
        target = batch["fmri"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled,
        ):
            prediction, kl_loss = model(cls_token, patch_tokens)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(
                    f"non-finite prediction at epoch={epoch}, batch={batch_index}"
                )
            if not torch.isfinite(kl_loss):
                raise FloatingPointError(
                    f"non-finite KL at epoch={epoch}, batch={batch_index}"
                )
            loss, mse, std_loss = loss_components(prediction, target, kl_loss, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch={epoch}, batch={batch_index}"
                )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["training"].get("grad_clip", 1.0),
                error_if_nonfinite=True,
            )
        except RuntimeError as error:
            if "non-finite" not in str(error).lower():
                raise
            # GradScaler has already recorded the overflow in unscale_. Drop
            # this update and let scaler.update() reduce the loss scale.
            nonfinite_parameters = []
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    bad_count = (~torch.isfinite(parameter.grad)).sum().item()
                    nonfinite_parameters.append(
                        f"{name} ({bad_count}/{parameter.grad.numel()})"
                    )
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            skipped_nonfinite_steps += 1
            print(
                f"Skipped non-finite gradient: epoch={epoch}, batch={batch_index}, "
                f"count={skipped_nonfinite_steps}; parameters="
                f"{nonfinite_parameters[:20]}"
            )
            if skipped_nonfinite_steps > max_nonfinite_steps:
                raise FloatingPointError(
                    f"more than {max_nonfinite_steps} non-finite gradient steps "
                    f"at epoch={epoch}, batch={batch_index}"
                )
            continue
        scaler.step(optimizer)
        scaler.update()
        successful_steps += 1

        pearson = per_sample_pearson(prediction.detach(), target).mean().item()
        values = {
            "loss": loss.item(), "mse": mse.item(),
            "std": std_loss.item(), "pearson": pearson,
            "grad_norm": grad_norm.item(),
        }
        for key, value in values.items():
            totals[key] += value
        progress.set_postfix({key: f"{value:.4f}" for key, value in values.items()})

    if successful_steps == 0:
        raise FloatingPointError(f"no finite optimization steps at epoch={epoch}")
    metrics = {key: value / successful_steps for key, value in totals.items()}
    metrics["skipped_nonfinite_steps"] = skipped_nonfinite_steps
    return metrics


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    correlations, squared_errors = [], []
    for batch in tqdm(loader, desc="Direct Stage2 Validation"):
        cls_token = batch["cls_token"].to(device, non_blocking=True)
        patch_tokens = batch["patch_tokens"].to(device, non_blocking=True)
        target = batch["fmri"].to(device, non_blocking=True)
        prediction, _ = model(cls_token, patch_tokens)
        correlations.append(per_sample_pearson(prediction, target).cpu())
        squared_errors.append((prediction - target).square().mean(dim=1).cpu())
    correlations = torch.cat(correlations)
    squared_errors = torch.cat(squared_errors)
    return {
        "pearson": correlations.mean().item(),
        "pearson_std": correlations.std().item(),
        "mse": squared_errors.mean().item(),
        "num_samples": len(correlations),
    }


def save_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_direct_stage2_subj7.yaml")
    parser.add_argument("--result", default="results/wacv_round2/direct_stage2_only_subj7_seed42.json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--save_dir")
    parser.add_argument("--vae_mode", choices=["none", "deterministic", "stochastic"])
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    seed = args.seed if args.seed is not None else config["experiment"].get("seed", 42)
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    if args.patience is not None:
        config["training"]["patience"] = args.patience
    if args.save_dir is not None:
        config["training"]["save_dir"] = args.save_dir
    if args.vae_mode is not None:
        config["model"]["vae_mode"] = args.vae_mode
    if args.precision is not None:
        config["training"]["precision"] = args.precision
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training configuration")
    device = torch.device("cuda")
    model_cfg, data_cfg, training_cfg = config["model"], config["data"], config["training"]

    train_loader, val_loader = create_dataloaders_v7h(
        huwng_base_path=data_cfg["huwng_base_path"],
        train_embeddings_path=data_cfg.get("train_embeddings_path"),
        test_embeddings_path=data_cfg.get("test_embeddings_path"),
        subjects=[data_cfg["subject"]],
        batch_size=training_cfg["batch_size"],
        voxels_per_cluster=model_cfg["voxels_per_cluster"],
        roi_top_k_percent=data_cfg.get("roi_top_k_percent", 0.7),
        average_trials_train=data_cfg.get("average_trials_train", False),
        average_trials_val=data_cfg.get("average_trials_val", True),
        augment_noise_train=data_cfg.get("augment_noise_train", True),
        noise_std=data_cfg.get("noise_std", 0.1),
        num_workers=data_cfg.get("num_workers", 8),
        dinov2_layer=data_cfg.get("dinov2_layer", -1),
    )

    model = build_one_stage_model(model_cfg).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_name = model_cfg.get("name", "DirectStage2Only")
    if model_name == "DirectStage2Only":
        model_name = f"{model_name} ({model_cfg.get('backbone', 'mamba')})"
    print(f"{model_name} parameters: {parameter_count:,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=training_cfg.get("precision") == "fp16",
        init_scale=training_cfg.get("amp_init_scale", 1024.0),
    )
    save_dir = Path(training_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    result_path = Path(args.result)
    start_time = time.time()
    best_pearson = -1.0
    best_epoch = None
    epochs_without_improvement = 0
    history = []

    result = {
        "status": "running",
        "experiment": model_cfg.get("name", "DirectStage2Only"),
        "description": model_cfg.get(
            "description",
            "DINOv2 to voxels with learned ROI tokens and absolute local prediction; no cluster values",
        ),
        "subject": data_cfg["subject"], "seed": seed,
        "vae_mode": model_cfg.get("vae_mode", "stochastic"),
        "parameters": parameter_count,
        "max_epochs": training_cfg["max_epochs"],
        "patience": training_cfg["patience"],
        "history": history,
    }
    save_result(result_path, result)

    try:
        for epoch in range(1, training_cfg["max_epochs"] + 1):
            train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, config, epoch)
            validation_metrics = validate(model, val_loader, device)
            if not all(np.isfinite(value) for key, value in validation_metrics.items() if key != "num_samples"):
                raise FloatingPointError(f"non-finite validation metrics at epoch={epoch}")
            history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
            print(
                f"Direct Ep {epoch} | Train P:{train_metrics['pearson']:.4f} "
                f"Val P:{validation_metrics['pearson']:.4f} MSE:{validation_metrics['mse']:.4f} "
                f"Grad:{train_metrics['grad_norm']:.4f}"
            )

            if validation_metrics["pearson"] > best_pearson:
                best_pearson = validation_metrics["pearson"]
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), save_dir / "best_model.pth")
                print(f"Saved best direct baseline: {best_pearson:.4f}")
            else:
                epochs_without_improvement += 1

            result.update({
                "latest_epoch": epoch, "best_epoch": best_epoch,
                "best_validation_pearson": best_pearson,
                "elapsed_seconds": time.time() - start_time,
            })
            save_result(result_path, result)
            if epochs_without_improvement >= training_cfg["patience"]:
                result["stopping_reason"] = "early_stopping"
                break
    except (FloatingPointError, RuntimeError) as error:
        if "non-finite" not in str(error).lower() and "nan" not in str(error).lower():
            raise
        result.update({
            "status": "failed_nonfinite",
            "failure": str(error),
            "elapsed_seconds": time.time() - start_time,
        })
        save_result(result_path, result)
        print(f"Training aborted safely: {error}")
        return

    result.update({
        "status": "complete",
        "completed_epochs": history[-1]["epoch"],
        "elapsed_seconds": time.time() - start_time,
        "checkpoint": str(save_dir / "best_model.pth"),
    })
    result.setdefault("stopping_reason", "max_epochs")
    save_result(result_path, result)
    print(f"Training complete. Best epoch={best_epoch}, Pearson={best_pearson:.4f}")


if __name__ == "__main__":
    main()

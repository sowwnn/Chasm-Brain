"""Profile CHASMBrain parameters, estimated FLOPs, memory, and inference speed."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from mamba_ssm import Mamba
from model.hierarchical_v3_1 import HierarchicalARM_V3
from model.tben_official import TBEnOneStageOfficial


def build_model(config, vae_mode):
    cfg = config["model"]
    if cfg.get("name") == "TBEnOneStage":
        return TBEnOneStageOfficial(
            cls_dim=cfg["cls_dim"],
            patch_dim=cfg["patch_dim"],
            num_patches=cfg.get("num_patches", 256),
            fmri_dim=cfg["fmri_dim"],
            voxels_per_cluster=cfg["voxels_per_cluster"],
            hidden_dim=cfg.get("hidden_dim", 768),
            nheads=cfg.get("nheads", 16),
            dim_feedforward=cfg.get("dim_feedforward", 1024),
            enc_layers=cfg.get("enc_layers", 0),
            dec_layers=cfg.get("dec_layers", 1),
            dropout=cfg.get("dropout", 0.1),
            pre_norm=cfg.get("pre_norm", True),
            enc_output_layer=cfg.get("enc_output_layer", 1),
        )
    return HierarchicalARM_V3(
        cls_dim=cfg["cls_dim"],
        patch_dim=cfg["patch_dim"],
        num_patches=cfg["num_patches"],
        fmri_dim=cfg["fmri_dim"],
        voxels_per_cluster=cfg["voxels_per_cluster"],
        hidden_dim_1=cfg["hidden_dim_1"],
        depth_what=cfg["depth_what"],
        depth_where=cfg["depth_where"],
        contrastive_dim=cfg["contrastive_dim"],
        embed_dim_2=cfg["embed_dim_2"],
        latent_dim=cfg.get("latent_dim", 128),
        depth_2=cfg["depth_2"],
        num_global_queries=cfg["num_global_queries"],
        global_hidden=cfg.get("global_hidden", 4096),
        dropout=cfg.get("dropout", 0.1),
        vae_mode=vae_mode,
    )


def count_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters())


class FlopEstimator:
    """Count dense ops and estimate Mamba selective-scan ops for one forward."""

    def __init__(self, model):
        self.flops = 0
        self.handles = []
        mamba_children = {
            id(child)
            for module in model.modules()
            if isinstance(module, Mamba)
            for child in module.modules()
            if child is not module
        }

        for module in model.modules():
            if isinstance(module, Mamba):
                self.handles.append(module.register_forward_hook(self._mamba_hook))
            elif isinstance(module, nn.Linear) and id(module) not in mamba_children:
                self.handles.append(module.register_forward_hook(self._linear_hook))

    def _linear_hook(self, module, inputs, output):
        vectors = math.prod(output.shape[:-1])
        self.flops += 2 * vectors * module.in_features * module.out_features

    def _mamba_hook(self, module, inputs, output):
        tokens = math.prod(inputs[0].shape[:-1])
        # Dense projections + depthwise convolution + approximate selective scan.
        per_token_macs = (
            module.d_model * (2 * module.d_inner)
            + module.d_inner * module.d_conv
            + module.d_inner * (module.dt_rank + 2 * module.d_state)
            + module.dt_rank * module.d_inner
            + 3 * module.d_inner * module.d_state
            + module.d_inner * module.d_model
        )
        self.flops += 2 * tokens * per_token_macs

    def close(self):
        for handle in self.handles:
            handle.remove()


def estimate_flops(module, forward):
    estimator = FlopEstimator(module)
    try:
        with torch.inference_mode():
            forward()
        return estimator.flops
    finally:
        estimator.close()


def benchmark(forward, device, warmup, repeats):
    for _ in range(warmup):
        forward()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        forward()
    end.record()
    torch.cuda.synchronize(device)

    latency_ms = start.elapsed_time(end) / repeats
    return latency_ms, torch.cuda.max_memory_allocated(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_hierarchical_v3_huwng.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="results/wacv_round2/efficiency.json")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--vae_mode", choices=["none", "deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for latency and peak-memory profiling")

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)

    device = torch.device("cuda")
    model = build_model(config, args.vae_mode)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model = model.to(device).eval()

    cfg = config["model"]
    batch = args.batch_size
    cls_token = torch.randn(batch, cfg["cls_dim"], device=device)
    patch_tokens = torch.randn(batch, cfg["num_patches"], cfg["patch_dim"], device=device)
    if not isinstance(model, TBEnOneStageOfficial):
        guidance = torch.randn(batch, model.num_rois, device=device)
        vis_feat = torch.randn(batch, cfg["cls_dim"] + cfg["hidden_dim_1"] // 2, device=device)

    autocast = lambda: torch.autocast("cuda", dtype=torch.float16, enabled=args.precision == "fp16")

    if isinstance(model, TBEnOneStageOfficial):
        def full_forward():
            with autocast():
                return model(cls_token, patch_tokens)

        total_flops = estimate_flops(model, full_forward)
        parameters = {"total": count_parameters(model)}
        flops = {
            "total": total_flops / batch,
            "note": "Dense Linear FLOPs only; attention matmuls, normalization and elementwise ops are excluded.",
        }
    else:
        def stage1_forward():
            with autocast():
                return model(cls_token, patch_tokens, stage=1)

        def stage2_forward():
            with autocast():
                return model.stage2(vis_feat, guidance)

        def full_forward():
            with autocast():
                return model(cls_token, patch_tokens, stage=2)

        stage1_flops = estimate_flops(model.stage1, stage1_forward)
        stage2_flops = estimate_flops(model.stage2, stage2_forward)
        stage1_params = count_parameters(model.stage1)
        stage2_params = count_parameters(model.stage2) + count_parameters(model.patch_pooler)
        parameters = {
            "stage1": stage1_params,
            "stage2_including_patch_pooler": stage2_params,
            "total": stage1_params + stage2_params,
        }
        flops = {
            "stage1": stage1_flops / batch,
            "stage2": stage2_flops / batch,
            "total": (stage1_flops + stage2_flops) / batch,
            "note": "Dense FLOPs plus an analytical Mamba estimate; normalization and elementwise ops are excluded.",
        }

    with torch.inference_mode():
        latency_ms, peak_memory = benchmark(full_forward, device, args.warmup, args.repeats)

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": torch.cuda.get_device_name(device),
        "batch_size": batch,
        "precision": args.precision,
        "parameters": parameters,
        "estimated_flops_per_image": flops,
        "inference": {
            "latency_ms_per_batch": latency_ms,
            "latency_ms_per_image": latency_ms / batch,
            "throughput_images_per_second": 1000 * batch / latency_ms,
            "peak_memory_bytes": peak_memory,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

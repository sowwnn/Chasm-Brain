#!/usr/bin/env python3
"""
Compute image quality metrics between reconstructed images.

Metrics:
    - SSIM: Structural Similarity (pixel-level)
    - LPIPS: Learned Perceptual Image Patch Similarity (perceptual)
    - CLIP: Vision-language similarity (high-level semantic)
    - DINOv2: Self-supervised visual similarity (semantic)

Folder structure (subfolders):
    images/
        gt/
            gt_01.png, gt_02.png, ...
        recongt/
            recongt_01.png, recongt_02.png, ...    (GT fMRI reconstruction)
        mindsim/
            mindsim_01.png, mindsim_02.png, ...    (MindSimulator)
        chasmbrain/
            chasmbrain_01.png, chasmbrain_02.png, ... (ChasmBrain - ours)

Usage:
    python scripts/compute_image_metrics.py --image_dir path/to/images
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from tqdm import tqdm

# For CLIP (using open_clip for better compatibility)
try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: open_clip not available. Install with: pip install open-clip-torch")

# For LPIPS (perceptual similarity)
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available. Install with: pip install lpips")

# For DINOv2 (semantic similarity)
DINO_AVAILABLE = True  # Uses torch.hub, always available with PyTorch


def load_image_rgb(path: str) -> np.ndarray:
    """Load image as RGB numpy array (0-255)."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two RGB images."""
    # Resize to same size if needed
    if img1.shape != img2.shape:
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = np.array(Image.fromarray(img1).resize((w, h)))
        img2 = np.array(Image.fromarray(img2).resize((w, h)))

    # Compute SSIM with multichannel
    return ssim(img1, img2, channel_axis=2, data_range=255)


class CLIPScorer:
    """CLIP-based image similarity scorer using open_clip."""

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        # Use ViT-B-32 with OpenAI pretrained weights
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, img_path: str) -> torch.Tensor:
        """Encode image to CLIP embedding."""
        image = self.preprocess(Image.open(img_path).convert("RGB"))
        image = image.unsqueeze(0).to(self.device)
        features = self.model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def compute_similarity(self, img1_path: str, img2_path: str) -> float:
        """Compute CLIP cosine similarity between two images."""
        feat1 = self.encode_image(img1_path)
        feat2 = self.encode_image(img2_path)
        similarity = (feat1 @ feat2.T).item()
        return similarity


class LPIPSScorer:
    """LPIPS perceptual similarity scorer."""

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = lpips.LPIPS(net="alex", verbose=False).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
        ])

    @torch.no_grad()
    def compute_distance(self, img1_path: str, img2_path: str) -> float:
        """Compute LPIPS distance (lower = more similar)."""
        img1 = self.transform(Image.open(img1_path).convert("RGB")).unsqueeze(0).to(self.device)
        img2 = self.transform(Image.open(img2_path).convert("RGB")).unsqueeze(0).to(self.device)
        distance = self.model(img1, img2).item()
        return distance


class DINOv2Scorer:
    """DINOv2-based semantic similarity scorer."""

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        # Load DINOv2 ViT-B/14
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def encode_image(self, img_path: str) -> torch.Tensor:
        """Encode image to DINOv2 embedding."""
        image = self.transform(Image.open(img_path).convert("RGB"))
        image = image.unsqueeze(0).to(self.device)
        features = self.model(image)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def compute_similarity(self, img1_path: str, img2_path: str) -> float:
        """Compute DINOv2 cosine similarity between two images."""
        feat1 = self.encode_image(img1_path)
        feat2 = self.encode_image(img2_path)
        similarity = (feat1 @ feat2.T).item()
        return similarity


def find_image_pairs(image_dir: str) -> dict:
    """
    Find image pairs from subfolders.

    Expected structure:
        image_dir/
            gt/           -> gt_01.png, gt_02.png, ...
            recongt/      -> recongt_01.png, ...
            mindsim/      -> mindsim_01.png, ...
            chasmbrain/   -> chasmbrain_01.png, ...

    Returns dict: {sample_id: {folder_key: filepath}}
    """
    image_dir = Path(image_dir)
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    # Folder mapping: folder_name -> (key, prefix_to_strip)
    folder_config = {
        "gt": ("gt", "gt_"),
        "recongt": ("recongt", "recongt_"),
        "mindsim": ("mindsim", "mindsim_"),
        "chasmbrain": ("chasmbrain", "chasmbrain_"),
    }

    # Collect all images
    samples = defaultdict(dict)

    for folder_name, (key, prefix) in folder_config.items():
        folder_path = image_dir / folder_name
        if not folder_path.exists():
            print(f"Warning: Folder not found: {folder_path}")
            continue

        for f in folder_path.iterdir():
            if f.suffix.lower() not in extensions:
                continue

            name = f.stem
            # Extract sample ID by removing prefix
            if name.startswith(prefix):
                sample_id = name[len(prefix):]
            else:
                sample_id = name  # Use full name if no prefix

            samples[sample_id][key] = str(f)

    # Filter to only complete sets (all 4 types present)
    complete_samples = {
        sid: paths for sid, paths in samples.items()
        if len(paths) == 4
    }

    return complete_samples


def compute_metrics(image_dir: str, device: str = "cuda") -> dict:
    """Compute all metrics for image pairs."""

    samples = find_image_pairs(image_dir)

    if not samples:
        print(f"No complete image sets found in {image_dir}")
        print("Expected subfolders: gt/, recongt/, mindsim/, chasmbrain/")
        return {}

    print(f"Found {len(samples)} complete image sets")

    # Initialize scorers
    clip_scorer = None
    lpips_scorer = None
    dino_scorer = None

    if CLIP_AVAILABLE:
        print("Loading CLIP model...")
        clip_scorer = CLIPScorer(device)

    if LPIPS_AVAILABLE:
        print("Loading LPIPS model...")
        lpips_scorer = LPIPSScorer(device)

    if DINO_AVAILABLE:
        print("Loading DINOv2 model...")
        dino_scorer = DINOv2Scorer(device)

    # Comparison pairs: (method1, method2, label)
    comparisons = [
        ("gt", "recongt", "GT vs ReconGT"),
        ("gt", "mindsim", "GT vs MindSimulator"),
        ("gt", "chasmbrain", "GT vs ChasmBrain (Ours)"),
    ]

    # Collect metrics
    results = {
        label: {"ssim": [], "lpips": [], "clip": [], "dino": []}
        for _, _, label in comparisons
    }

    for sample_id in tqdm(sorted(samples.keys()), desc="Computing metrics"):
        paths = samples[sample_id]

        for prefix1, prefix2, label in comparisons:
            img1 = load_image_rgb(paths[prefix1])
            img2 = load_image_rgb(paths[prefix2])

            # SSIM (pixel-level)
            ssim_val = compute_ssim(img1, img2)
            results[label]["ssim"].append(ssim_val)

            # LPIPS (perceptual)
            if lpips_scorer:
                lpips_val = lpips_scorer.compute_distance(paths[prefix1], paths[prefix2])
                results[label]["lpips"].append(lpips_val)

            # CLIP (high-level semantic)
            if clip_scorer:
                clip_val = clip_scorer.compute_similarity(paths[prefix1], paths[prefix2])
                results[label]["clip"].append(clip_val)

            # DINOv2 (semantic)
            if dino_scorer:
                dino_val = dino_scorer.compute_similarity(paths[prefix1], paths[prefix2])
                results[label]["dino"].append(dino_val)

    return results


def print_results(results: dict):
    """Print results as a formatted table."""

    print("\n" + "=" * 100)
    print("IMAGE QUALITY METRICS")
    print("=" * 100)

    # Header
    print(f"{'Comparison':<25} | {'SSIM ↑':>12} | {'LPIPS ↓':>12} | {'CLIP ↑':>12} | {'DINOv2 ↑':>12}")
    print("-" * 100)

    for label, metrics in results.items():
        ssim_vals = metrics["ssim"]
        lpips_vals = metrics["lpips"]
        clip_vals = metrics["clip"]
        dino_vals = metrics["dino"]

        ssim_mean = np.mean(ssim_vals) if ssim_vals else 0
        ssim_std = np.std(ssim_vals) if ssim_vals else 0

        lpips_mean = np.mean(lpips_vals) if lpips_vals else 0
        lpips_std = np.std(lpips_vals) if lpips_vals else 0

        clip_mean = np.mean(clip_vals) if clip_vals else 0
        clip_std = np.std(clip_vals) if clip_vals else 0

        dino_mean = np.mean(dino_vals) if dino_vals else 0
        dino_std = np.std(dino_vals) if dino_vals else 0

        ssim_str = f"{ssim_mean:.3f}±{ssim_std:.3f}"
        lpips_str = f"{lpips_mean:.3f}±{lpips_std:.3f}" if lpips_vals else "N/A"
        clip_str = f"{clip_mean:.3f}±{clip_std:.3f}" if clip_vals else "N/A"
        dino_str = f"{dino_mean:.3f}±{dino_std:.3f}" if dino_vals else "N/A"

        print(f"{label:<25} | {ssim_str:>12} | {lpips_str:>12} | {clip_str:>12} | {dino_str:>12}")

    print("=" * 100)
    print(f"Total samples: {len(list(results.values())[0]['ssim'])}")
    print("\nMetric interpretation:")
    print("  SSIM ↑    : Structural similarity (higher = better)")
    print("  LPIPS ↓   : Perceptual distance (lower = better)")
    print("  CLIP ↑    : High-level semantic similarity (higher = better)")
    print("  DINOv2 ↑  : Self-supervised semantic similarity (higher = better)")


def save_results_latex(results: dict, output_path: str):
    """Save results as LaTeX table."""

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Image reconstruction quality metrics. $\uparrow$ higher is better, $\downarrow$ lower is better.}",
        r"\label{tab:image_metrics}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{SSIM} $\uparrow$ & \textbf{LPIPS} $\downarrow$ & \textbf{CLIP} $\uparrow$ & \textbf{DINOv2} $\uparrow$ \\",
        r"\midrule",
    ]

    for label, metrics in results.items():
        ssim_mean = np.mean(metrics["ssim"])
        ssim_std = np.std(metrics["ssim"])

        lpips_mean = np.mean(metrics["lpips"]) if metrics["lpips"] else 0
        lpips_std = np.std(metrics["lpips"]) if metrics["lpips"] else 0

        clip_mean = np.mean(metrics["clip"]) if metrics["clip"] else 0
        clip_std = np.std(metrics["clip"]) if metrics["clip"] else 0

        dino_mean = np.mean(metrics["dino"]) if metrics["dino"] else 0
        dino_std = np.std(metrics["dino"]) if metrics["dino"] else 0

        # Bold for ChasmBrain (Ours)
        if "Ours" in label:
            ssim_str = f"\\textbf{{{ssim_mean:.3f}}}$\\pm${ssim_std:.3f}"
            lpips_str = f"\\textbf{{{lpips_mean:.3f}}}$\\pm${lpips_std:.3f}"
            clip_str = f"\\textbf{{{clip_mean:.3f}}}$\\pm${clip_std:.3f}"
            dino_str = f"\\textbf{{{dino_mean:.3f}}}$\\pm${dino_std:.3f}"
            label_str = f"\\textbf{{{label}}}"
        else:
            ssim_str = f"{ssim_mean:.3f}$\\pm${ssim_std:.3f}"
            lpips_str = f"{lpips_mean:.3f}$\\pm${lpips_std:.3f}"
            clip_str = f"{clip_mean:.3f}$\\pm${clip_std:.3f}"
            dino_str = f"{dino_mean:.3f}$\\pm${dino_std:.3f}"
            label_str = label

        lines.append(f"{label_str} & {ssim_str} & {lpips_str} & {clip_str} & {dino_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nLaTeX table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute image quality metrics (SSIM, LPIPS, CLIP, DINOv2)")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing subfolders: gt/, recongt/, mindsim/, chasmbrain/")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for CLIP model (cuda/cpu)")
    parser.add_argument("--save_latex", type=str, default=None,
                        help="Path to save LaTeX table")
    args = parser.parse_args()

    # Check directory exists
    if not os.path.isdir(args.image_dir):
        print(f"Error: Directory not found: {args.image_dir}")
        return

    # Compute metrics
    results = compute_metrics(args.image_dir, args.device)

    if results:
        # Print results
        print_results(results)

        # Save LaTeX if requested
        if args.save_latex:
            save_results_latex(results, args.save_latex)


if __name__ == "__main__":
    main()

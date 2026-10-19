#!/usr/bin/env python3
"""Extract DINOv2 ViT-B/14 multilayer tokens for the prepared CSI1 split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


ROOT = Path("dataset/BOLD5000")
IMAGE_ROOT = ROOT / "BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli"
PROCESSED_DIR = ROOT / "processed/CSI1"
LAYERS_ZERO_INDEXED = [2, 5, 8, 11]  # Paper notation: layers 3, 6, 9, 12.


class StimulusDataset(Dataset):
    def __init__(self, names_path: Path, transform):
        self.names = names_path.read_text().splitlines()
        paths_by_name = {}
        for path in IMAGE_ROOT.rglob("*"):
            if path.is_file():
                if path.name in paths_by_name:
                    raise ValueError(f"Duplicate presented stimulus basename: {path.name}")
                paths_by_name[path.name] = path
        missing = [name for name in self.names if name not in paths_by_name]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} stimuli, e.g. {missing[:5]}")
        self.paths = [paths_by_name[name] for name in self.names]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor


def extract_split(model, split: str, batch_size: int, num_workers: int) -> dict:
    names_path = PROCESSED_DIR / f"{split}_image_names.txt"
    output_path = PROCESSED_DIR / f"dinov2_vitb14_multilayer_{split}.npy"
    partial_path = output_path.with_suffix(".partial.npy")
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    dataset = StimulusDataset(names_path, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    output = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(dataset), 4, 257, 768),
    )

    offset = 0
    with torch.inference_mode():
        for images in tqdm(loader, desc=f"DINOv2 CSI1 {split}"):
            images = images.cuda(non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                layers = model.get_intermediate_layers(
                    images,
                    n=LAYERS_ZERO_INDEXED,
                    reshape=False,
                    return_class_token=True,
                    norm=True,
                )
                tokens = [
                    torch.cat((cls_token.unsqueeze(1), patch_tokens), dim=1)
                    for patch_tokens, cls_token in layers
                ]
                features = torch.stack(tokens, dim=1)
            if features.shape[1:] != (4, 257, 768):
                raise ValueError(f"Unexpected DINOv2 output shape: {tuple(features.shape)}")
            if not torch.isfinite(features).all():
                raise FloatingPointError(f"Non-finite DINOv2 features in {split} at {offset}")
            batch = features.cpu().numpy().astype(np.float16, copy=False)
            output[offset : offset + len(batch)] = batch
            offset += len(batch)

    output.flush()
    del output
    if offset != len(dataset):
        raise RuntimeError(f"Wrote {offset} rows, expected {len(dataset)}")
    partial_path.replace(output_path)
    return {
        "split": split,
        "num_images": len(dataset),
        "names": str(names_path),
        "output": str(output_path),
        "shape": [len(dataset), 4, 257, 768],
        "dtype": "float16",
        "size_bytes": output_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DINOv2 feature extraction")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    model = model.eval().cuda()

    outputs = [
        extract_split(model, "train", args.batch_size, args.num_workers),
        extract_split(model, "test", args.batch_size, args.num_workers),
    ]
    metadata = {
        "model": "DINOv2 ViT-B/14",
        "torch_hub": "facebookresearch/dinov2:dinov2_vitb14",
        "layers_paper_indexed": [3, 6, 9, 12],
        "layers_zero_indexed": LAYERS_ZERO_INDEXED,
        "token_order": "CLS followed by 256 patch tokens",
        "preprocessing": "RGB; resize 256 bicubic; center crop 224; ImageNet normalization",
        "outputs": outputs,
    }
    metadata_path = PROCESSED_DIR / "dinov2_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

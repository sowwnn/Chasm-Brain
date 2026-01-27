"""
Extract DINOv2 embeddings from NSD stimuli for MindEye2 splits.

This script:
1. Loads NSD stimuli (425x425 RGB uint8)
2. Preprocesses to 224x224 float16 (MindEye2 style)
3. Extracts DINOv2 embeddings for images in MindEye2 splits
4. Saves embeddings aligned with MindEye2 image IDs

Usage:
    # Extract for all splits
    python extract_dinov2_mindeye2.py --subject 1

    # Extract only specific splits
    python extract_dinov2_mindeye2.py --subject 1 --splits train test
"""

import os
import h5py
import numpy as np
import json
import torch
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse


class DINOv2Extractor:
    """Extract DINOv2 embeddings from images."""

    def __init__(
        self,
        model_name: str = 'dinov2_vitb14',
        device: str = 'cuda',
        batch_size: int = 64,
    ):
        self.device = device
        self.batch_size = batch_size

        print(f"Loading DINOv2 model: {model_name}...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(device)
        self.model.eval()

        # Preprocessing: resize to 224 and normalize
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"  Device: {device}")
        print(f"  Batch size: {batch_size}")

    def preprocess_image(self, img_array: np.ndarray) -> torch.Tensor:
        """
        Preprocess NSD image for DINOv2.

        Args:
            img_array: [425, 425, 3] uint8

        Returns:
            tensor: [3, 224, 224] normalized
        """
        # Convert to PIL
        img_pil = Image.fromarray(img_array)

        # Apply transforms
        img_tensor = self.transform(img_pil)

        return img_tensor

    def extract_batch(self, images: list) -> np.ndarray:
        """
        Extract embeddings for a batch of images.

        Args:
            images: List of [425, 425, 3] uint8 arrays

        Returns:
            embeddings: [batch, 768] float32
        """
        # Preprocess all images
        batch_tensors = [self.preprocess_image(img) for img in images]
        batch_tensor = torch.stack(batch_tensors).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model(batch_tensor)  # [batch, 768]

        return features.cpu().numpy()

    def extract_from_stimuli(
        self,
        stimuli_path: str,
        image_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Extract embeddings for specific NSD image IDs.

        Args:
            stimuli_path: Path to nsd_stimuli.hdf5
            image_ids: Array of NSD IDs to extract [N]

        Returns:
            embeddings: [N, 768] aligned with image_ids
        """
        num_images = len(image_ids)
        embeddings = np.zeros((num_images, 768), dtype=np.float32)

        print(f"\nExtracting embeddings for {num_images} images...")

        with h5py.File(stimuli_path, 'r') as f:
            img_brick = f['imgBrick']  # [73000, 425, 425, 3]

            # Process in batches
            for i in tqdm(range(0, num_images, self.batch_size), desc="Batches"):
                end_idx = min(i + self.batch_size, num_images)
                batch_ids = image_ids[i:end_idx]

                # Load batch of images
                batch_images = [img_brick[img_id] for img_id in batch_ids]

                # Extract embeddings
                batch_embeddings = self.extract_batch(batch_images)

                embeddings[i:end_idx] = batch_embeddings

        print(f"✓ Extracted embeddings: {embeddings.shape}")
        print(f"  Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")

        return embeddings


def extract_embeddings_for_splits(
    stimuli_path: str,
    metadata_dir: str,
    output_dir: str,
    subject: int,
    splits: list = ['train', 'test', 'new_test'],
    model_name: str = 'dinov2_vitb14',
    batch_size: int = 64,
    device: str = 'cuda',
):
    """
    Extract DINOv2 embeddings for all splits.

    Args:
        stimuli_path: Path to nsd_stimuli.hdf5
        metadata_dir: Directory with MindEye2 metadata
        output_dir: Directory to save embeddings
        subject: Subject number
        splits: List of splits to process
        model_name: DINOv2 model name
        batch_size: Batch size for extraction
        device: Device for inference
    """
    metadata_dir = Path(metadata_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractor
    extractor = DINOv2Extractor(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )

    # Process each split
    results = {}

    for split in splits:
        print(f"\n{'='*70}")
        print(f"Processing split: {split.upper()}")
        print(f"{'='*70}")

        # Load image IDs for this split
        ids_path = metadata_dir / f'{split}_ids.npy'

        if not ids_path.exists():
            print(f"  Skipping {split}: {ids_path} not found")
            continue

        image_ids = np.load(ids_path)
        print(f"  Images: {len(image_ids)}")

        # Extract embeddings
        embeddings = extractor.extract_from_stimuli(stimuli_path, image_ids)

        # Save embeddings
        emb_path = output_dir / f'dinov2_{split}_sub{subject:02d}.npy'
        np.save(emb_path, embeddings)
        print(f"\n✓ Saved to: {emb_path}")

        # Save image IDs for reference
        ids_out_path = output_dir / f'image_ids_{split}_sub{subject:02d}.npy'
        np.save(ids_out_path, image_ids)
        print(f"✓ Saved image IDs to: {ids_out_path}")

        results[split] = {
            'embeddings_path': str(emb_path),
            'image_ids_path': str(ids_out_path),
            'num_images': len(image_ids),
            'embedding_dim': embeddings.shape[1],
        }

    # Save summary
    summary = {
        'subject': subject,
        'model': model_name,
        'embedding_dim': 768,
        'splits': results
    }

    summary_path = output_dir / f'embedding_summary_sub{subject:02d}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Extract DINOv2 embeddings for MindEye2 splits')
    parser.add_argument('--stimuli_path', type=str,
                        default='/mnt/HDD/dataset/NSD_raw/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5',
                        help='Path to nsd_stimuli.hdf5')
    parser.add_argument('--metadata_dir', type=str,
                        default='dataset/nsd/metadata',
                        help='Directory with MindEye2 metadata')
    parser.add_argument('--output_dir', type=str,
                        default='dataset/nsd/embeddings_mindeye2',
                        help='Output directory for embeddings')
    parser.add_argument('--subject', type=int, default=1,
                        help='Subject number')
    parser.add_argument('--splits', nargs='+', default=['train', 'test', 'new_test'],
                        help='Splits to process')
    parser.add_argument('--model', type=str, default='dinov2_vitb14',
                        choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                        help='DINOv2 model variant')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for extraction')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference')
    args = parser.parse_args()

    print("=" * 70)
    print("DINOv2 Embedding Extraction for MindEye2 Splits")
    print("=" * 70)
    print(f"Subject: {args.subject}")
    print(f"Model: {args.model}")
    print(f"Splits: {args.splits}")
    print(f"Output: {args.output_dir}")

    results = extract_embeddings_for_splits(
        stimuli_path=args.stimuli_path,
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
        subject=args.subject,
        splits=args.splits,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )

    print("\n" + "=" * 70)
    print("✓ Embedding extraction complete!")
    print("=" * 70)
    print("\nGenerated files:")
    for split, info in results.items():
        print(f"\n  {split.upper()}:")
        print(f"    Embeddings: {info['embeddings_path']}")
        print(f"    Image IDs:  {info['image_ids_path']}")
        print(f"    Count:      {info['num_images']} images")


if __name__ == '__main__':
    main()

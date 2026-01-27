"""
Extract DINOv2 embeddings from NSD stimuli images.

Usage:
    # Extract all 73k images
    python extract_dinov2_embeddings.py
    
    # Quick test with 100 images
    python extract_dinov2_embeddings.py --num_images 100
    
Output:
    NSD/data/embeddings/dinov2_vitb14_embeddings.npy - [73000, 768] CLS tokens
"""

import os
import sys
import argparse
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

# Image transforms
from torchvision import transforms
from PIL import Image


class NSDStimuliDataset(Dataset):
    """
    Dataset for NSD stimuli images from HDF5 file.
    Memory-efficient: loads images on-demand, does not load entire HDF5 into RAM.
    """
    
    def __init__(self, hdf5_path: str, transform=None, num_images: int = None):
        """
        Args:
            hdf5_path: Path to nsd_stimuli.hdf5
            transform: Image transforms
            num_images: Number of images to load (None = all)
        """
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.hdf5_file = None  # Lazy open
        
        # Get total number of images (quick open/close)
        with h5py.File(hdf5_path, 'r') as f:
            total_images = f['imgBrick'].shape[0]
        
        self.num_images = num_images if num_images else total_images
        print(f"Dataset: {self.num_images} / {total_images} images")
        print(f"  Memory-efficient mode: loading images on-demand")
    
    def _open_hdf5(self):
        """Lazy open HDF5 file (needed for multiprocessing)."""
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        # Open file lazily (for multiprocessing compatibility)
        self._open_hdf5()
        
        # Load single image (memory-efficient)
        img = self.hdf5_file['imgBrick'][idx]  # [425, 425, 3] uint8
        
        # Convert to PIL Image
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return {'image': img, 'idx': idx}
    
    def __del__(self):
        """Close HDF5 file when dataset is deleted."""
        if self.hdf5_file is not None:
            self.hdf5_file.close()


def get_dinov2_model(model_name: str = 'dinov2_vitb14'):
    """
    Load DINOv2 model from torch hub.
    
    Args:
        model_name: Model variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
    
    Returns:
        model, transform
    """
    print(f"\nLoading {model_name}...")
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model.eval()
    
    # DINOv2 expects 224x224 images with ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    return model, transform


def extract_embeddings(
    stimuli_path: str,
    output_path: str,
    model_name: str = 'dinov2_vitb14',
    batch_size: int = 64,
    num_images: int = None,
    device: str = 'cuda',
):
    """
    Extract DINOv2 CLS token embeddings from NSD stimuli.
    
    Args:
        stimuli_path: Path to nsd_stimuli.hdf5
        output_path: Output .npy file path
        model_name: DINOv2 model variant
        batch_size: Batch size for extraction
        num_images: Number of images (None = all 73k)
        device: 'cuda' or 'cpu'
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, transform = get_dinov2_model(model_name)
    model = model.to(device)
    
    # Get embedding dimension
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        dummy_out = model(dummy)
        embed_dim = dummy_out.shape[-1]
    print(f"Embedding dimension: {embed_dim}")
    
    # Create dataset and dataloader
    dataset = NSDStimuliDataset(
        hdf5_path=stimuli_path,
        transform=transform,
        num_images=num_images,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # HDF5 not compatible with multiprocessing, also saves RAM
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # Extract embeddings
    print(f"\nExtracting embeddings...")
    all_embeddings = np.zeros((len(dataset), embed_dim), dtype=np.float32)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Batches"):
            images = batch['image'].to(device)
            indices = batch['idx'].numpy()
            
            # Get CLS token embeddings
            embeddings = model(images)  # [batch, embed_dim]
            
            all_embeddings[indices] = embeddings.cpu().numpy()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path, all_embeddings)
    
    print(f"\n{'='*60}")
    print(f"✓ Embeddings saved to: {output_path}")
    print(f"  Shape: {all_embeddings.shape}")
    print(f"  Mean: {all_embeddings.mean():.4f}")
    print(f"  Std: {all_embeddings.std():.4f}")
    print(f"{'='*60}")
    
    return all_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract DINOv2 embeddings from NSD stimuli')
    parser.add_argument('--stimuli_path', type=str,
                        default='/media/hung/data1/codes/img2fmriii/NSD/data/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5',
                        help='Path to nsd_stimuli.hdf5')
    parser.add_argument('--output_dir', type=str,
                        default='/media/hung/data1/codes/img2fmriii/NSD/data/embeddings',
                        help='Output directory for embeddings')
    parser.add_argument('--model', type=str, default='dinov2_vitb14',
                        choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                        help='DINOv2 model variant')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for extraction')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Number of images to process (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    args = parser.parse_args()
    
    output_path = Path(args.output_dir) / f'{args.model}_embeddings.npy'
    
    extract_embeddings(
        stimuli_path=args.stimuli_path,
        output_path=str(output_path),
        model_name=args.model,
        batch_size=args.batch_size,
        num_images=args.num_images,
        device=args.device,
    )

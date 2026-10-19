"""
Extract ConvNeXT-Large and SAM ViT-H features from NSD stimuli.

IDs are derived from nsd_stim_info_merged.csv (NSD_raw ground truth).
  subject1 & not shared1000 → 9000 train images
  shared1000                 → 1000 test images

ConvNeXT-Large (facebook/convnext-large-224-22k-1k):
  last_hidden_state (B, 1536, 7, 7) → global avg pool → (B, 1536)
  Output: nsd_convnext_large_train_sub1.npy  (9000, 1536)
           nsd_convnext_large_test_sub1.npy   (1000, 1536)

SAM ViT-H (facebook/sam-vit-huge):
  image_embeddings (B, 256, 64, 64) → global avg pool → (B, 256)
  Output: nsd_sam_vith_train_sub1.npy  (9000, 256)
           nsd_sam_vith_test_sub1.npy   (1000, 256)
"""

import numpy as np
import scipy.io
import h5py
import torch
from pathlib import Path
from PIL import Image
from transformers import ConvNextModel, ConvNextImageProcessor
from transformers import SamModel, SamProcessor

ROOT         = Path(__file__).resolve().parent.parent
STIMULI_PATH = Path("/mnt/HDD/dataset/NSD_raw/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")
EXPDESIGN    = Path("/mnt/HDD/dataset/NSD_raw/nsddata/experiments/nsd/nsd_expdesign.mat")
HUWNG_BASE   = ROOT / "dataset/huwng/subj01"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH        = 64
BATCH_SAM    = 8   # SAM ViT-H runs at 1024x1024, very VRAM-heavy


class NSDStimuli:
    def __init__(self, path):
        self._f    = h5py.File(path, "r")
        self._imgs = self._f["imgBrick"]

    def get_batch(self, nsd_ids):
        sorted_pos = np.argsort(nsd_ids)
        sorted_ids = nsd_ids[sorted_pos]
        raw = self._imgs[sorted_ids.tolist()]
        out = np.empty_like(raw)
        out[sorted_pos] = raw
        return out

    def close(self):
        self._f.close()


@torch.no_grad()
def extract_convnext(stimuli, nsd_ids, model, processor, save_path):
    N   = len(nsd_ids)
    out = np.lib.format.open_memmap(save_path, mode="w+", dtype=np.float32, shape=(N, 1536))

    for start in range(0, N, BATCH):
        end    = min(start + BATCH, N)
        raw_np = stimuli.get_batch(nsd_ids[start:end])
        imgs   = [Image.fromarray(raw_np[i]) for i in range(len(raw_np))]

        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
        feats  = model(**inputs).last_hidden_state   # (B, 1536, 7, 7)
        feats  = feats.mean(dim=[-2, -1])            # (B, 1536)
        out[start:end] = feats.float().cpu().numpy()
        out.flush()

        if start % (BATCH * 4) == 0:
            print(f"  {end:>5}/{N}", flush=True)

    return out.shape


@torch.no_grad()
def extract_sam(stimuli, nsd_ids, model, processor, save_path):
    N   = len(nsd_ids)
    out = np.lib.format.open_memmap(save_path, mode="w+", dtype=np.float32, shape=(N, 256))

    for start in range(0, N, BATCH_SAM):
        end    = min(start + BATCH_SAM, N)
        raw_np = stimuli.get_batch(nsd_ids[start:end])
        imgs   = [Image.fromarray(raw_np[i]) for i in range(len(raw_np))]

        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
        feats  = model.get_image_embeddings(inputs["pixel_values"])  # (B, 256, 64, 64)
        feats  = feats.mean(dim=[-2, -1])                             # (B, 256)
        out[start:end] = feats.float().cpu().numpy()
        out.flush()

        if start % (BATCH_SAM * 16) == 0:
            print(f"  {end:>5}/{N}", flush=True)

    return out.shape


def main():
    print(f"Device: {DEVICE}")
    mat       = scipy.io.loadmat(str(EXPDESIGN))
    subj01    = mat['subjectim'][0]          # (10000,) 1-indexed NSD IDs
    train_ids = subj01[1000:] - 1            # 9000 train → matches HUWNG fmri row order
    test_ids  = subj01[:1000] - 1            # 1000 test  → matches HUWNG fmri row order
    print(f"Train: {len(train_ids)} | Test: {len(test_ids)}")

    stimuli = NSDStimuli(STIMULI_PATH)

    # --- ConvNeXT-Large ---
    print("\n[ConvNeXT-Large] Loading model...")
    cnxt_processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-large-224-22k-1k")
    cnxt_model     = ConvNextModel.from_pretrained("facebook/convnext-large-224-22k-1k").eval().to(DEVICE)

    print("[ConvNeXT-Large] train")
    shape = extract_convnext(stimuli, train_ids, cnxt_model, cnxt_processor,
                             HUWNG_BASE / "nsd_convnext_large_train_sub1.npy")
    print(f"  saved {shape}")

    print("[ConvNeXT-Large] test")
    shape = extract_convnext(stimuli, test_ids, cnxt_model, cnxt_processor,
                             HUWNG_BASE / "nsd_convnext_large_test_sub1.npy")
    print(f"  saved {shape}")

    del cnxt_model, cnxt_processor
    torch.cuda.empty_cache()

    # --- SAM ViT-H ---
    print("\n[SAM ViT-H] Loading model...")
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    sam_model     = SamModel.from_pretrained("facebook/sam-vit-huge").eval().to(DEVICE)

    print("[SAM ViT-H] train")
    shape = extract_sam(stimuli, train_ids, sam_model, sam_processor,
                        HUWNG_BASE / "nsd_sam_vith_train_sub1.npy")
    print(f"  saved {shape}")

    print("[SAM ViT-H] test")
    shape = extract_sam(stimuli, test_ids, sam_model, sam_processor,
                        HUWNG_BASE / "nsd_sam_vith_test_sub1.npy")
    print(f"  saved {shape}")

    stimuli.close()
    print("\nDone.")


if __name__ == "__main__":
    main()

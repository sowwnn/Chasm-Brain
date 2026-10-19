"""
Build self-contained dataset from NSD_raw for subj01:
  - fMRI: process betas → (N_img, 15724) z-scored, averaged over reps
  - Features: ConvNeXT-Large, SAM ViT-H from same image set/ordering

Output dir: dataset/nsd_raw_subj01/
  fmri_train.npy        (9000, 15724) float32
  fmri_test.npy         (1000, 15724) float32
  ids_train.npy         (9000,)       int  — NSD IDs (0-indexed)
  ids_test.npy          (1000,)       int
  convnext_train.npy    (9000, 50, 1536)  float32  — token 0: global avg, tokens 1-49: spatial patches
  convnext_test.npy     (1000, 50, 1536)  float32
  sam_train.npy         (9000, 256)   float32
  sam_test.npy          (1000, 256)   float32

Index ordering: by NSD ID ascending (consistent across fMRI and features).
"""

import argparse
import numpy as np
import scipy.io
import h5py
import nibabel as nib
import torch
from pathlib import Path
from PIL import Image

ROOT       = Path(__file__).resolve().parent.parent
NSD_ROOT   = Path("/mnt/HDD/dataset/NSD_raw")
STIMULI    = NSD_ROOT / "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
EXPDESIGN  = NSD_ROOT / "nsddata/experiments/nsd/nsd_expdesign.mat"
BETAS_DIR  = NSD_ROOT / "nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR"
MASK_PATH  = NSD_ROOT / "nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz"
OUT_DIR    = ROOT / "dataset/nsd_raw_subj01"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_SESSIONS = 40
TRIALS_PER_SESSION = 750
BETA_SCALE = 300.0   # int16 → % BOLD


def build_fmri():
    """Process all 40 session betas → per-image averaged z-scored fMRI."""
    print(f"Loading mask {MASK_PATH.name}...")
    mask_nii = nib.load(MASK_PATH).get_fdata()         # (81,104,83), values in {-1,0,1}
    mask_bool = mask_nii > 0                            # 15724 True
    n_vox = int(mask_bool.sum())
    print(f"  {n_vox} voxels in nsdgeneral")

    # Beta volume axes are reversed vs NIfTI: beta(T,83,104,81) ↔ mask(81,104,83)
    mask_beta = np.transpose(mask_bool, (2, 1, 0))      # (83,104,81)

    print(f"Loading expdesign...")
    mat = scipy.io.loadmat(str(EXPDESIGN))
    masterordering = mat['masterordering'].flatten().astype(np.int32) - 1  # 0-indexed (0..9999)
    subjectim      = mat['subjectim'][0].astype(np.int32) - 1              # (10000,) 0-indexed NSD IDs
    sharedix       = mat['sharedix'].flatten().astype(np.int32) - 1        # (1000,) 0-indexed NSD IDs
    print(f"  masterordering: {masterordering.shape}, subjectim: {subjectim.shape}, sharedix: {sharedix.shape}")

    # Trial → 10k-image-index, repeated across 40 sessions = 30000 trials
    assert len(masterordering) == N_SESSIONS * TRIALS_PER_SESSION

    # Accumulate per-image betas (sum + count) for averaging
    sum_betas   = np.zeros((10000, n_vox), dtype=np.float64)
    count_betas = np.zeros(10000, dtype=np.int32)

    for sess in range(1, N_SESSIONS + 1):
        path = BETAS_DIR / f"betas_session{sess:02d}.hdf5"
        print(f"  [session {sess:02d}] loading {path.name}...")
        with h5py.File(path, 'r') as f:
            betas = np.array(f['betas'])              # (750,83,104,81) int16
        # Apply mask → (750, 15724), scale to % BOLD
        masked = betas[:, mask_beta].astype(np.float32) / BETA_SCALE
        # Trial indices for this session
        t0 = (sess - 1) * TRIALS_PER_SESSION
        for i in range(TRIALS_PER_SESSION):
            img_idx = masterordering[t0 + i]
            sum_betas[img_idx]   += masked[i]
            count_betas[img_idx] += 1

    # Average reps per image
    seen = count_betas > 0
    print(f"  Images with at least 1 trial: {seen.sum()} / 10000")
    avg_betas = np.zeros_like(sum_betas, dtype=np.float32)
    avg_betas[seen] = (sum_betas[seen] / count_betas[seen, None]).astype(np.float32)

    # Z-score per voxel across all seen images (using train+test together is fine,
    # since z-score is just per-voxel normalization, not data leakage of labels)
    mu  = avg_betas[seen].mean(axis=0, keepdims=True)
    sd  = avg_betas[seen].std(axis=0, keepdims=True) + 1e-8
    fmri_zs = (avg_betas - mu) / sd
    print(f"  Z-scored. mean≈0: {fmri_zs[seen].mean():.4f}, std≈1: {fmri_zs[seen].std():.4f}")

    # Map 10k-image-index → NSD ID, then split into train/test
    nsd_ids_all   = subjectim                                     # (10000,) 0-indexed NSD IDs
    is_test       = np.isin(nsd_ids_all, sharedix)                # (10000,) bool, ~1000 True
    test_mask10k  = is_test & seen
    train_mask10k = ~is_test & seen

    # Sort by NSD ID ascending → consistent ordering for features
    test_idx10k_sorted  = np.where(test_mask10k)[0]
    test_idx10k_sorted  = test_idx10k_sorted[np.argsort(nsd_ids_all[test_idx10k_sorted])]
    train_idx10k_sorted = np.where(train_mask10k)[0]
    train_idx10k_sorted = train_idx10k_sorted[np.argsort(nsd_ids_all[train_idx10k_sorted])]

    fmri_test  = fmri_zs[test_idx10k_sorted]
    fmri_train = fmri_zs[train_idx10k_sorted]
    ids_test   = nsd_ids_all[test_idx10k_sorted]
    ids_train  = nsd_ids_all[train_idx10k_sorted]

    print(f"  Train: fMRI {fmri_train.shape} | IDs {ids_train.shape}")
    print(f"  Test : fMRI {fmri_test.shape}  | IDs {ids_test.shape}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "fmri_train.npy", fmri_train)
    np.save(OUT_DIR / "fmri_test.npy",  fmri_test)
    np.save(OUT_DIR / "ids_train.npy",  ids_train)
    np.save(OUT_DIR / "ids_test.npy",   ids_test)
    print(f"  Saved to {OUT_DIR}")
    return ids_train, ids_test


def get_batch_imgs(stim_h5, nsd_ids):
    """Read a batch of images from nsd_stimuli.hdf5 (HDF5 needs sorted index access)."""
    pos = np.argsort(nsd_ids)
    raw = stim_h5[nsd_ids[pos].tolist()]
    out = np.empty_like(raw)
    out[pos] = raw
    return out


@torch.no_grad()
def extract_convnext(stim_h5, ids_train, ids_test):
    from transformers import ConvNextModel, ConvNextImageProcessor
    print("Loading ConvNeXT-Large...")
    proc  = ConvNextImageProcessor.from_pretrained("facebook/convnext-large-224-22k-1k")
    model = ConvNextModel.from_pretrained("facebook/convnext-large-224-22k-1k").eval().to(DEVICE)
    BATCH = 64

    def run(ids, save_path):
        N   = len(ids)
        # Shape (N, 50, 1536): token 0 = global avg (CLS), tokens 1-49 = spatial patches
        out = np.lib.format.open_memmap(save_path, mode='w+', dtype=np.float32, shape=(N, 50, 1536))
        for s in range(0, N, BATCH):
            e      = min(s + BATCH, N)
            raw    = get_batch_imgs(stim_h5, ids[s:e])
            pil    = [Image.fromarray(raw[i]) for i in range(len(raw))]
            inp    = proc(images=pil, return_tensors='pt').to(DEVICE)
            feat   = model(**inp).last_hidden_state.float()  # (B, 1536, 7, 7)
            cls    = feat.mean(dim=[-2, -1])                 # (B, 1536)
            patches = feat.permute(0, 2, 3, 1).reshape(feat.shape[0], 49, 1536)  # (B, 49, 1536)
            tokens = torch.cat([cls.unsqueeze(1), patches], dim=1)  # (B, 50, 1536)
            out[s:e] = tokens.cpu().numpy()
            out.flush()
            if s % (BATCH * 8) == 0:
                print(f"    {e:>5}/{N}", flush=True)

    print("  [convnext train]")
    run(ids_train, OUT_DIR / "convnext_train.npy")
    print("  [convnext test]")
    run(ids_test, OUT_DIR / "convnext_test.npy")
    del model, proc
    torch.cuda.empty_cache()


@torch.no_grad()
def extract_sam(stim_h5, ids_train, ids_test):
    from transformers import SamModel, SamProcessor
    print("Loading SAM ViT-H (fp16)...")
    proc  = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    model = SamModel.from_pretrained("facebook/sam-vit-huge", torch_dtype=torch.float16).eval().to(DEVICE)
    BATCH = 2

    def run(ids, save_path):
        N   = len(ids)
        out = np.lib.format.open_memmap(save_path, mode='w+', dtype=np.float32, shape=(N, 256))
        for s in range(0, N, BATCH):
            e   = min(s + BATCH, N)
            raw = get_batch_imgs(stim_h5, ids[s:e])
            pil = [Image.fromarray(raw[i]) for i in range(len(raw))]
            inp = proc(images=pil, return_tensors='pt').to(DEVICE)
            pix = inp['pixel_values'].to(torch.float16)
            f   = model.get_image_embeddings(pix).mean(dim=[-2, -1])
            out[s:e] = f.float().cpu().numpy()
            out.flush()
            if s % (BATCH * 16) == 0:
                print(f"    {e:>5}/{N}", flush=True)

    print("  [sam train]")
    run(ids_train, OUT_DIR / "sam_train.npy")
    print("  [sam test]")
    run(ids_test, OUT_DIR / "sam_test.npy")
    del model, proc
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fmri", action="store_true", help="reuse existing fmri/ids files")
    ap.add_argument("--only", choices=["fmri", "convnext", "sam"], default=None)
    args = ap.parse_args()

    print(f"Device: {DEVICE} | Output: {OUT_DIR}")

    if args.skip_fmri or (args.only and args.only != "fmri"):
        ids_train = np.load(OUT_DIR / "ids_train.npy")
        ids_test  = np.load(OUT_DIR / "ids_test.npy")
        print(f"Reusing IDs: train {ids_train.shape}, test {ids_test.shape}")
    else:
        ids_train, ids_test = build_fmri()
        if args.only == "fmri":
            return

    print("\nOpening stimuli HDF5...")
    with h5py.File(STIMULI, 'r') as f:
        stim = f['imgBrick']
        if args.only in (None, "convnext"):
            print("\n=== ConvNeXT-Large ===")
            extract_convnext(stim, ids_train, ids_test)
        if args.only in (None, "sam"):
            print("\n=== SAM ViT-H ===")
            extract_sam(stim, ids_train, ids_test)

    print("\nDone.")


if __name__ == "__main__":
    main()

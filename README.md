# ChasmBrain: Dual-Stream Mamba for fMRI Prediction

Image-to-fMRI prediction using Dual-Stream Mamba architecture with DINOv2 visual features.

## Project Structure

```
ChasmBrain/
├── preprocess/          # Data preprocessing scripts
├── data/                # Dataset loaders
├── model/               # Model architectures
├── scripts/             # Training scripts
├── configs/             # Training configurations
└── checkpoints/         # Model weights
```

## Quick Start

### Prerequisites

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.10
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Install PyTorch with CUDA (if not auto-detected)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Data Preprocessing Pipeline

Run scripts in order from `preprocess/` directory.

### Step 1: Download NSD Data

```bash
cd Data  # Create this folder outside ChasmBrain
python ../ChasmBrain/preprocess/download_nsddata.py
```

Downloads from AWS S3 (no authentication required):
- Experiment design files
- Stimuli images (HDF5)
- fMRI betas for subjects 1, 2, 5, 7
- ROI masks

**Output structure:**
```
Data/
├── nsddata/
│   ├── experiments/nsd/
│   │   ├── nsd_expdesign.mat
│   │   └── nsd_stim_info_merged.pkl
│   └── ppdata/subj0X/func1pt8mm/roi/
├── nsddata_betas/ppdata/subj0X/func1pt8mm/betas_fithrf_GLMdenoise_RR/
└── nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5
```

### Step 2: Prepare fMRI Data

```bash
python ../ChasmBrain/preprocess/prepare_nsddata_scale.py -sub 1 -session 40 -mode zscore
```

**Arguments:**
- `-sub`: Subject number (1, 2, 5, 7)
- `-session`: Number of sessions (default: 40)
- `-mode`: Normalization mode (`scale` or `zscore`)

**Output:**
```
Data/nsd/subj01/
├── nsd_train_fmri_zscore_sub1.npy   # (N_train, 3, 15724) - 3 trials per image
├── nsd_test_fmri_zscore_sub1.npy    # (N_test, 3, 15724)
├── nsd_train_stim_sub1.npy          # (N_train, 425, 425, 3) uint8
├── nsd_test_stim_sub1.npy           # (N_test, 425, 425, 3) uint8
├── nsd_train_cap_sub1.npy           # Captions (optional)
└── nsd_test_cap_sub1.npy
```

### Step 3: Extract PNG Images

```bash
python ../ChasmBrain/preprocess/save_images.py --sub 1
```

**Output:**
```
Data/nsd/subj01/
├── train_img/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── test_img/
│   ├── 0.png
│   └── ...
└── ../evals/all_images.pt  # (N_test, 3, 256, 256) for evaluation
```

### Step 4: Extract Visual Features

Choose one of the following:

#### Option A: DINOv2 Multi-layer (Recommended for ChasmBrain)

```bash
python ../ChasmBrain/preprocess/extract_features_dinov2_multilayer.py --subjects 1 --batch_size 128
```

Uses DINOv2 ViT-B/14, extracts layers [3, 6, 9, 12].

**Output:** `nsd_dinov2_vitb14_multilayer_{split}_sub{X}.npy`
- Shape: `(N, 4, 257, 768)` - 4 layers × (CLS + 256 patches) × 768-dim
- Size: ~14 GB train, ~1.5 GB test

#### Option B: DINOv2 Single-layer

```bash
python ../ChasmBrain/preprocess/extract_features_dinov2.py --subjects 1 --batch_size 128
```

Uses DINOv2 ViT-L/14, extracts final layer only.

**Output:** `nsd_dinov2_vitl14_{split}_sub{X}.npy`
- Shape: `(N, 257, 1024)` - (CLS + 256 patches) × 1024-dim

#### Option C: SDXL CLIP (Optional, for image reconstruction)

```bash
python ../ChasmBrain/preprocess/extract_features_sdxl_unclip.py
```

Requires SDXL model checkpoint. Used for image generation tasks.

---

## Training

### Configure Training

Edit `configs/train_hierarchical_v3_nsd.yaml`:

```yaml
data:
  subject: 1
  nsd_base_path: "../Data/nsd"  # Path to processed data
  dinov2_layer: -1                 # Use last layer from multilayer features

model:
  fmri_dim: 15724
  voxels_per_cluster: 30

training:
  batch_size: 32
  lr: 3.0e-5
  max_epochs: 450
  save_dir: "checkpoints/hierarchical_v3_subj01"
```

### Train Stage 1 (Dual-Stream Mamba)

```bash
python scripts/train_hierarchical_v3.py \
    --config configs/train_hierarchical_v3_nsd.yaml \
    --stage 1
```

Trains the What-stream (CLS token) and Where-stream (patch tokens) with:
- Peak-focused loss for ROI prediction
- Contrastive loss for image-brain alignment

### Train Stage 2 (Voxel Prediction)

```bash
python scripts/train_hierarchical_v3.py \
    --config configs/train_hierarchical_v3_nsd.yaml \
    --stage 2 \
    --load_stage1 checkpoints/hierarchical_v3_subj01/best_stage1.pth
```

Refines voxel-level predictions using guidance from Stage 1.

### Monitor Training

```bash
tensorboard --logdir checkpoints/hierarchical_v3_subj01/logs
```

---

## Model Architecture

```
Image → DINOv2 → [CLS, Patches]
                      │
                      ▼
         ┌────────────┴────────────┐
         │                         │
    What-Stream              Where-Stream
    (CLS token)              (Patches)
    Mamba blocks             Mamba blocks
         │                         │
         └─────────┬───────────────┘
                   │
            Adaptive Fusion
                   │
                   ▼
            ROI Predictions (Stage 1)
                   │
                   ▼
         ┌─────────┴─────────┐
         │                   │
    Global Branch       Local Branch
    (Mamba)             (Mamba)
         │                   │
         └─────────┬─────────┘
                   │
                   ▼
           Voxel Predictions (Stage 2)
```

---

## File Reference

### Preprocessing (keep these)
| File | Purpose |
|------|---------|
| `download_nsddata.py` | Download NSD from AWS S3 |
| `prepare_nsddata_scale.py` | Process fMRI betas, split train/test |
| `save_images.py` | Extract PNG images from stimuli |
| `extract_features_dinov2_multilayer.py` | DINOv2 ViT-B/14 multi-layer (recommended) |
| `extract_features_dinov2.py` | DINOv2 ViT-L/14 single layer |
| `extract_features_sdxl_unclip.py` | SDXL CLIP features (optional) |

### Data Loaders
| File | Purpose |
|------|---------|
| `data/dataset_v7_h.py` | Main dataset loader for NSD format |

### Models
| File | Purpose |
|------|---------|
| `model/hierarchical_v3_1.py` | Dual-Stream Mamba architecture |
| `model/loss.py` | Peak-focused loss |
| `model/contrastive_loss.py` | InfoNCE contrastive loss |

---

## Citation

If you use this code, please cite:
```bibtex
@misc{chasmbrain2024,
  title={ChasmBrain: Dual-Stream Mamba for fMRI Prediction},
  year={2024}
}
```

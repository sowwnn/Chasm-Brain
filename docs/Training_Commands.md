# Hướng Dẫn Chạy Training Các Model ARMNet

## 📋 Mục Lục
1. [DiffusionARM (1D - 1000 parcels)](#1-diffusionarm)
2. [DiffusionARM3D (3D CNN)](#2-diffusionarm3d)
3. [DiffusionGraphARM (GNN)](#3-diffusiongrapharm)
4. [DenoisingARM (Single-step)](#4-denoisingarm)
5. [Flow3DHybridARM (Flow Matching)](#5-flow3dhybridarm)
6. [FlowGraphARM (Graph Flow)](#6-flowgrapharm)
7. [DualStreamCNN (Dual-stream)](#7-dualstreamcnn)
8. [HierarchicalCVAE (VAE)](#8-hierarchicalcvae)
9. [ARMNet Baseline](#9-armnet-baseline)

---

## 1. DiffusionARM

### Mô tả
- **Model**: DDPM với x_0 prediction
- **Input**: Visual features (768) → 1000 parcels
- **Output**: 1000 parcels (có thể reconstruct về 15k voxels)

### Command
```bash
# Basic training
python scripts/train_diffusion.py --config configs/train_diffusion.yaml

# Override parameters
python scripts/train_diffusion.py \
    --config configs/train_diffusion.yaml \
    --batch_size 64 \
    --lr 1e-4 \
    --max_epochs 100 \
    --timesteps 50
```

### Config chính (configs/train_diffusion.yaml)
```yaml
model:
  hidden_dim: 2048        # Wide model
  num_layers: 6           # Số layers
  num_heads: 8            # Attention heads
  dropout: 0.15
  
  diffusion:
    timesteps: 50         # Training timesteps
    inference_timesteps: 10  # DDIM inference (faster)
    clip_value: 3.0
    
data:
  batch_size: 32
  use_parcellation: true  # Sử dụng 1000 parcels
  parcel_labels_path: "dataset/nsd/subj01/betas_all_subj01_fp32_renorm_kmeans1000_labels.npy"
```

### Output
- **Checkpoints**: `outputs/diffusion_arm/v1_wide/checkpoints/`
- **Logs**: `outputs/diffusion_arm/v1_wide/logs/`
- **Visualizations**: `outputs/diffusion_arm/v1_wide/visualizations/`

---

## 2. DiffusionARM3D

### Mô tả
- **Model**: 3D CNN U-Net với DDPM
- **Input**: Visual features → 3D volume (26×26×24)
- **Output**: 3D fMRI volume
- **Đặc điểm**: Spatial modeling, CFG support

### Command
```bash
python scripts/train_diffusion_3d.py --config configs/train_diffusion_3d.yaml
```

### Config chính
```yaml
model:
  hidden_channels: [64, 128, 256, 512]  # U-Net channels
  use_cfg: true                         # Classifier-free guidance
  cfg_dropout: 0.1
  
  diffusion:
    timesteps: 1000
    inference_timesteps: 50
```

---

## 3. DiffusionGraphARM

### Mô tả
- **Model**: Graph Neural Network với DDPM
- **Input**: Visual features + Graph structure
- **Output**: 15k voxels (as graph nodes)
- **Yêu cầu**: `torch_geometric`

### Command
```bash
# Cài đặt torch_geometric trước
pip install torch-geometric

# Training
python scripts/train_graph_diffusion.py --config configs/train_graph_diffusion.yaml
```

### Config chính
```yaml
model:
  gnn_type: 'gat'         # GAT hoặc GCN
  num_heads: 8
  hidden_channels: [64, 128, 256]
  
graph:
  k_neighbors: 8          # KNN graph
  # hoặc
  radius: 3.0            # Radius graph
```

---

## 4. DenoisingARM

### Mô tả
- **Model**: Single-step denoising (không phải diffusion)
- **Input**: Visual features → 1000 parcels
- **Output**: 1000 parcels → reconstruct về 15k
- **Đặc điểm**: Nhanh nhất (1 forward pass)

### ⚠️ Lưu Ý Quan Trọng
**DenoisingARM KHÔNG hỗ trợ trực tiếp 15k voxels!**
- Chỉ làm việc với **1000 parcels**
- Reconstruct về 15k thông qua `ParcelMapper`
- Nếu cần 15k trực tiếp → dùng **DualStreamCNN** hoặc **Flow3DHybridARM**

### Command
```bash
python scripts/train_denoising.py --config configs/train_denoising.yaml
```

### Config chính
```yaml
model:
  fmri_dim: 1000          # PHẢI là 1000 (parcels)
  hidden_dim: 512
  num_layers: 12
  dropout: 0.5

training:
  sigma: 1.0              # Noise level
  vis_noise_std: 0.3      # Visual feature noise (regularization)
  
data:
  parcel_labels_path: "..."  # BẮT BUỘC để reconstruct về 15k
```

### Cách Reconstruct về 15k
```python
from data.parcel_utils import ParcelMapper

# Load mapper
parcel_mapper = ParcelMapper.from_files('path/to/parcel_labels.npy')

# Predict 1k parcels
pred_1k = model(x_noised, visual_feat, mean_fmri)  # [B, 1000]

# Reconstruct to 15k
pred_15k = parcel_mapper.reconstruct(pred_1k.cpu().numpy())  # [B, 15724]
```

---

## 5. Flow3DHybridARM

### Mô tả
- **Model**: Hybrid CNN+GNN với Flow Matching
- **Input**: Visual features → 3D volume (61×46×42)
- **Output**: 3D fMRI (lossless conversion ↔ 15k)
- **Đặc điểm**: State-of-the-art, mask-aware

### Command
```bash
python scripts/train_flow_3d_hybrid.py --config configs/train_flow_3d_hybrid.yaml
```

### Config chính
```yaml
model:
  compact_shape: [61, 46, 42]  # Full 3D shape
  use_graph_layers: true       # Hybrid CNN+GNN
  use_mask_conditioning: true  # Mask-aware
  
  flow:
    sigma: 0.0                 # Deterministic OT flow
    
training:
  steps: 20                    # Integration steps
  cfg_scale: 1.0              # CFG scale
```

---

## 6. FlowGraphARM

### Mô tả
- **Model**: Pure GNN với Flow Matching
- **Input**: Visual features + Graph
- **Output**: 15k voxels (as graph)
- **Yêu cầu**: `torch_geometric`

### Command
```bash
python scripts/train_graph_flow.py --config configs/train_graph_flow.yaml
```

---

## 7. DualStreamCNN

### Mô tả
- **Model**: Dual-stream (3D + 1D) với ConvNeXt
- **Input**: Visual features
- **Output**: 15k voxels trực tiếp
- **Đặc điểm**: Coarse-to-fine, mask-aware

### Command
```bash
python scripts/train_dual_stream.py --config configs/train_dual_stream.yaml
```

### Config chính
```yaml
model:
  compact_shape: [61, 46, 42]
  n_voxels: 15724            # Output 15k trực tiếp!
  base_channels_3d: 64
  base_channels_1d: 128
  fusion_type: 'attention'   # concat, attention, gated
  use_coarse_supervision: true
```

### Đặc điểm
✅ **Hỗ trợ 15k voxels trực tiếp** (không cần parcellation)
✅ Kết hợp spatial (3D) và voxel-level (1D)
✅ Modern ConvNeXt architecture

---

## 8. HierarchicalCVAE

### Mô tả
- **Model**: Conditional VAE
- **Input**: Visual features → 1000 parcels
- **Output**: 1000 parcels (có thể generate diverse samples)

### Command
```bash
python scripts/train_cvae.py --config configs/train_cvae.yaml
```

---

## 9. ARMNet Baseline

### Mô tả
- **Model**: MLP baseline
- **Input**: Visual features → 15k voxels
- **Output**: 15k voxels trực tiếp

### Command
```bash
python scripts/train.py --config configs/train.yaml
```

---

## 📊 So Sánh Hỗ Trợ 15k Voxels

| Model | Hỗ trợ 15k trực tiếp? | Cách xử lý |
|-------|----------------------|------------|
| **ARMNet** | ✅ Có | Output trực tiếp 15k |
| **DiffusionARM** | ❌ Không | 1k parcels → reconstruct |
| **DiffusionARM3D** | ⚠️ Gián tiếp | 3D (26×26×24) ≈ 16k |
| **DiffusionGraphARM** | ✅ Có | Graph với 15k nodes |
| **HierarchicalCVAE** | ❌ Không | 1k parcels → reconstruct |
| **DenoisingARM** | ❌ Không | 1k parcels → reconstruct |
| **Flow3DHybridARM** | ✅ Có | 3D (61×46×42) → lossless 15k |
| **FlowGraphARM** | ✅ Có | Graph với 15k nodes |
| **DualStreamCNN** | ✅ Có | Output trực tiếp 15k |

---

## 🎯 Khuyến Nghị Theo Use Case

### Nếu cần 15k voxels trực tiếp:
1. **DualStreamCNN** - Modern, dual-stream, tốt nhất cho 15k
2. **Flow3DHybridARM** - State-of-the-art, lossless 3D↔1D
3. **FlowGraphARM** - Graph-based, flexible
4. **ARMNet** - Baseline đơn giản

### Nếu chấp nhận 1k parcels + reconstruct:
1. **DiffusionARM** - Quality cao, diffusion-based
2. **DenoisingARM** - Nhanh nhất (single-step)
3. **HierarchicalCVAE** - Diversity (multiple samples)

### Nếu cần spatial modeling:
1. **Flow3DHybridARM** - Hybrid CNN+GNN
2. **DiffusionARM3D** - 3D CNN U-Net
3. **DualStreamCNN** - Dual-stream 3D+1D

---

## 🔧 Troubleshooting

### DenoisingARM với 15k
**Lỗi**: `RuntimeError: size mismatch`

**Nguyên nhân**: DenoisingARM chỉ hỗ trợ `fmri_dim=1000`

**Giải pháp**:
```yaml
# configs/train_denoising.yaml
model:
  fmri_dim: 1000  # KHÔNG thể thay đổi thành 15724

data:
  use_parcellation: true
  parcel_labels_path: "..."  # BẮT BUỘC
```

### Muốn 15k trực tiếp?
**Chuyển sang DualStreamCNN hoặc Flow3DHybridARM**

```bash
# Thay vì DenoisingARM
python scripts/train_dual_stream.py --config configs/train_dual_stream.yaml
```

---

## 📁 File Structure

```
I2fMRI/
├── scripts/
│   ├── train.py                    # ARMNet baseline
│   ├── train_diffusion.py          # DiffusionARM
│   ├── train_diffusion_3d.py       # DiffusionARM3D
│   ├── train_graph_diffusion.py    # DiffusionGraphARM
│   ├── train_denoising.py          # DenoisingARM
│   ├── train_flow_3d_hybrid.py     # Flow3DHybridARM
│   ├── train_graph_flow.py         # FlowGraphARM
│   ├── train_dual_stream.py        # DualStreamCNN
│   └── train_cvae.py               # HierarchicalCVAE
│
├── configs/
│   ├── train.yaml
│   ├── train_diffusion.yaml
│   ├── train_diffusion_3d.yaml
│   ├── train_graph_diffusion.yaml
│   ├── train_denoising.yaml
│   ├── train_flow_3d_hybrid.yaml
│   ├── train_graph_flow.yaml
│   ├── train_dual_stream.yaml
│   └── train_cvae.yaml
│
└── outputs/
    ├── diffusion_arm/
    ├── diffusion_arm_3d/
    ├── denoising_arm/
    ├── flow_3d_hybrid/
    └── dual_stream_cnn/
```

---

## 🚀 Quick Start

### 1. Training DiffusionARM (1k parcels)
```bash
python scripts/train_diffusion.py --config configs/train_diffusion.yaml
```

### 2. Training DualStreamCNN (15k voxels)
```bash
python scripts/train_dual_stream.py --config configs/train_dual_stream.yaml
```

### 3. Training với custom parameters
```bash
python scripts/train_diffusion.py \
    --config configs/train_diffusion.yaml \
    --batch_size 64 \
    --lr 1e-4 \
    --max_epochs 100
```

---

## 📝 Notes

1. **Parcellation vs Full Voxels**:
   - Parcellation (1k): Nhanh hơn, ít memory, nhưng mất spatial detail
   - Full voxels (15k): Chậm hơn, nhiều memory, giữ full detail

2. **Reconstruction Quality**:
   - ParcelMapper reconstruction có thể mất một số detail
   - Nếu cần quality cao → dùng model hỗ trợ 15k trực tiếp

3. **Memory Requirements**:
   - 1k parcels: ~4GB VRAM
   - 15k voxels: ~8-12GB VRAM
   - 3D models: ~12-16GB VRAM

4. **Training Time**:
   - DenoisingARM: Nhanh nhất (~2-3h)
   - DiffusionARM: Trung bình (~5-8h)
   - Flow3DHybridARM: Chậm nhất (~10-15h)

# ARMNet - Bảng Tham Chiếu Nhanh

## Tóm Tắt Input/Output Của Các Kiến Trúc

### 1. ARMNet (Baseline)
```python
# Input
x: [B, 768]              # Visual features (DINOv2)
mean_fmri: [B, 15724]    # Baseline fMRI

# Output
output: [B, 15724]       # Predicted fMRI
```

### 2. DiffusionARM
```python
# Training
vis_feat: [B, 768]       # Visual features
mean_fmri: [B, 1000]     # Baseline fMRI
noisy_fmri: [B, 1000]    # Noisy input
t: [B]                   # Timestep

# Output
pred_x0: [B, 1000]       # Predicted clean signal
```

### 3. DiffusionARM3D
```python
# Training
vis_feat: [B, 768]                    # Visual features
noisy_fmri_3d: [B, 1, 26, 26, 24]    # 3D noisy fMRI
t: [B]                                # Timestep
mask: [26, 26, 24] (optional)         # Brain mask

# Output
pred_x0_3d: [B, 1, 26, 26, 24]       # Predicted clean 3D fMRI
```

### 4. DiffusionGraphARM
```python
# Training
noisy_fmri: [B, N] or [N]            # Noisy fMRI (N=15724)
coords: [B, N, 3] or [N, 3]          # Voxel coordinates
edge_index: [2, E]                    # Graph edges
vis_feat: [B, 768]                    # Visual features
t: [B]                                # Timestep
batch: [N] (optional)                 # Batch assignment

# Output
pred_noise: [B, N] or [N]            # Predicted noise
```

### 5. HierarchicalCVAE
```python
# Training
visual_feat: [B, 768]                # Visual features
target_parcels: [B, 1000]            # Ground truth parcels

# Output (Training)
pred_parcels: [B, 1000]              # Predicted parcels
mu: [B, 128]                         # Latent mean
logvar: [B, 128]                     # Latent log variance

# Inference
visual_feat: [B, 768]                # Visual features

# Output (Inference)
pred_parcels: [B, 1000]              # Predicted parcels
```

### 6. DenoisingARM
```python
# Input
x_noised: [B, 1000]                  # mean_fMRI + noise
visual_feat: [B, 768]                # Visual features
mean_fmri: [B, 1000]                 # Base mean signal

# Output
denoised: [B, 1000]                  # Denoised fMRI
```

### 7. Flow3DHybridARM
```python
# Training
clean_fmri_3d: [B, 1, D, H, W]       # Ground truth 3D (D=61, H=46, W=42)
vis_feat: [B, 768]                    # Visual features
mask: [D, H, W] (optional)            # Brain mask
coords_compact: [N_brain, 3] (opt)    # Brain coordinates

# Output (Training)
{
    'pred_velocity': [B, 1, D, H, W],
    'target_velocity': [B, 1, D, H, W],
    't': [B],
    'x_pred': [B, 1, D, H, W]
}

# Inference
vis_feat: [B, 768]                    # Visual features
mask, coords_compact (optional)

# Output (Inference)
generated_3d: [B, 1, D, H, W]        # Generated 3D fMRI
```

### 8. FlowGraphARM
```python
# Training
clean_fmri: [B, N]                   # Ground truth (N=15724)
coords: [B, N, 3] or [N, 3]          # Coordinates
edge_index: [2, E]                    # Graph edges
vis_feat: [B, 768]                    # Visual features
batch: [N] (optional)                 # Batch assignment

# Output (Training)
{
    'pred_velocity': [B, N],
    'target_velocity': [B, N],
    't': [B],
    'x_pred': [B, N]
}

# Inference
coords, edge_index, vis_feat, batch

# Output (Inference)
generated: [B, N]                     # Generated fMRI
```

### 9. DualStreamCNN
```python
# Input
vis_feat: [B, 768]                    # Visual features (DINOv2)
mask: [D, H, W]                       # Brain mask (D=61, H=46, W=42)
coords_compact: [N, 3]                # Voxel coordinates (N=15724)

# Output
{
    'pred_fmri_1d': [B, N],          # Final prediction
    'coarse_1d': [B, N],             # Coarse prediction
    'feat_3d': [B, C_3d, D', H', W'], # 3D features
    'feat_1d': [B, C_1d, N']         # 1D features
}
```

---

## Bảng So Sánh Nhanh

| Model | Input Dim | Output Dim | Representation | Inference Speed | Quality | GPU Memory |
|-------|-----------|------------|----------------|-----------------|---------|------------|
| ARMNet | 1D (15k) | 1D (15k) | MLP | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | 💾 |
| DiffusionARM | 1D (1k) | 1D (1k) | MLP+Attn | ⚡⚡ | ⭐⭐⭐⭐ | 💾💾 |
| DiffusionARM3D | 3D | 3D | 3D CNN | ⚡ | ⭐⭐⭐⭐⭐ | 💾💾💾 |
| DiffusionGraphARM | Graph | Graph | GNN | ⚡ | ⭐⭐⭐⭐ | 💾💾💾 |
| HierarchicalCVAE | 1D (1k) | 1D (1k) | VAE | ⚡⚡⚡⚡ | ⭐⭐⭐ | 💾💾 |
| DenoisingARM | 1D (1k) | 1D (1k) | MLP+Attn | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 💾💾 |
| Flow3DHybridARM | 3D | 3D | CNN+GNN | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 💾💾💾💾 |
| FlowGraphARM | Graph | Graph | GNN | ⚡⚡⚡ | ⭐⭐⭐⭐ | 💾💾💾 |
| DualStreamCNN | 3D+1D | 1D (15k) | Dual CNN | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 💾💾💾 |

---

## Các Thành Phần Chung

### Visual Features
Tất cả các model đều nhận **DINOv2 embeddings** với shape `[B, 768]`

### fMRI Representations
- **1D Full**: 15,724 voxels (toàn bộ brain voxels)
- **1D Parcels**: 1,000 parcels (vùng não được gom nhóm)
- **3D Compact**: 26×26×24 = 16,224 (padding để fit 3D)
- **3D Full**: 61×46×42 = 117,852 (lossless 3D representation)
- **Graph**: 15,724 nodes với edges (KNN hoặc radius-based)

### Coordinates
- **coords**: `[N, 3]` - Tọa độ (x, y, z) của mỗi voxel trong không gian 3D
- **coords_compact**: Coordinates trong compact 3D space (61×46×42)

### Brain Mask
- **mask**: `[D, H, W]` - Boolean mask chỉ ra voxels nào thuộc não
- Dùng để focus learning vào brain regions, ignore background

---

## Workflow Điển Hình

### Training
```python
# 1. Load data
visual_feat = dinov2_model(image)  # [B, 768]
fmri = load_fmri()                 # [B, N] or [B, 1, D, H, W]

# 2. Forward pass
output = model(visual_feat, fmri, ...)

# 3. Compute loss
loss = criterion(output, target)

# 4. Backward
loss.backward()
optimizer.step()
```

### Inference
```python
# 1. Extract visual features
visual_feat = dinov2_model(image)  # [B, 768]

# 2. Generate fMRI
with torch.no_grad():
    if isinstance(model, (DiffusionARM, DiffusionARM3D)):
        # Diffusion models
        pred_fmri = diffusion_manager.sample(visual_feat, mean_fmri)
    elif isinstance(model, (Flow3DHybridARM, FlowGraphARM)):
        # Flow matching models
        pred_fmri = model.generate(visual_feat, ...)
    else:
        # Direct models
        pred_fmri = model(visual_feat, mean_fmri)
```

---

## Dependencies

### Core
- `torch`
- `torch.nn`
- `torch.nn.functional`

### Optional
- `torch_geometric` - Required for:
  - DiffusionGraphARM
  - FlowGraphARM
  - Flow3DHybridARM (optional graph layers)

### Installation
```bash
# Core PyTorch
pip install torch torchvision

# Graph support (optional)
pip install torch-geometric
```

---

## File Structure

```
ARMNet/
├── __init__.py                    # Package exports
├── model.py                       # ARMNet (baseline)
├── diffusion_model.py            # DiffusionARM
├── diffusion_3d_model.py         # DiffusionARM3D
├── diffusion_graph_model.py      # DiffusionGraphARM
├── cvae_model.py                 # HierarchicalCVAE
├── denoising_model.py            # DenoisingARM
├── flow_3d_hybrid_model.py       # Flow3DHybridARM
├── flow_graph_model.py           # FlowGraphARM
├── dual_stream_cnn_model.py      # DualStreamCNN
├── loss.py                       # Loss functions
├── dual_stream_loss.py           # Dual stream losses
└── flow_3d_loss.py               # Flow 3D losses
```

---

## Import Examples

```python
# Basic models
from ARMNet import ARMNet, DenoisingARM, HierarchicalCVAE

# Diffusion models
from ARMNet import DiffusionARM, DiffusionManager
from ARMNet import DiffusionARM3D, DiffusionManager3D

# Flow models
from ARMNet import Flow3DHybridARM, FlowMatching3DLoss

# Graph models (requires torch_geometric)
from ARMNet import FlowGraphARM, DiffusionGraphARM

# Dual stream
from ARMNet.dual_stream_cnn_model import DualStreamCNN
from ARMNet.dual_stream_loss import DualStreamLoss

# Loss functions
from ARMNet import PeakFocusedLoss, CVAELoss
```

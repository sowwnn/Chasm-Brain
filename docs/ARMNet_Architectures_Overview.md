# Tổng Quan Các Kiến Trúc ARMNet

## Mục Lục
1. [ARMNet (Baseline)](#1-armnet-baseline)
2. [DiffusionARM](#2-diffusionarm)
3. [DiffusionARM3D](#3-diffusionarm3d)
4. [DiffusionGraphARM](#4-diffusiongrapharm)
5. [HierarchicalCVAE](#5-hierarchicalcvae)
6. [DenoisingARM](#6-denoisingarm)
7. [Flow3DHybridARM](#7-flow3dhybridarm)
8. [FlowGraphARM](#8-flowgrapharm)
9. [DualStreamCNN](#9-dualstreamcnn)

---

## 1. ARMNet (Baseline)

**File:** `ARMNet/model.py`

### Mô tả
Kiến trúc baseline đơn giản sử dụng MLP với residual connections và parcel-level guidance.

### Input
- `x`: **[B, 768]** - Visual features từ DINOv2
- `mean_fmri`: **[B, 15724]** - Baseline fMRI signal (mean của subject)

### Output
- **[B, 15724]** - Predicted fMRI signal

### Kiến trúc chính
1. **Feature Extractor**: Nâng cấp visual features lên hidden_dim (1024)
2. **Deep Features**: 2 ResidualBlocks để học sâu hơn
3. **Parcel Bottleneck**: Giảm xuống 1000 parcels (vùng não)
4. **Residual Head**: Dự đoán chi tiết ở mức voxel
5. **Gating Network**: Học vùng hoạt động của não
6. **Scale Network**: Học độ hưng phấn chung

### Công thức
```
output = mean_fmri + scale * residual * gate
```

---

## 2. DiffusionARM

**File:** `ARMNet/diffusion_model.py`

### Mô tả
Sử dụng DDPM (Denoising Diffusion Probabilistic Model) với x_0 prediction, FiLM time conditioning, và Cross-Attention.

### Input
- `vis_feat`: **[B, 768]** - Visual features
- `mean_fmri`: **[B, 1000]** hoặc **[1, 1000]** - Baseline fMRI
- `noisy_fmri`: **[B, 1000]** - Noisy fMRI tại timestep t (training)
- `t`: **[B]** - Timestep (0 đến T)

### Output
- **[B, 1000]** - Predicted clean signal (x_0)

### Kiến trúc chính
1. **Sinusoidal Time Embedding**: Encode timestep
2. **Visual/Mean Projection**: Project conditioning signals
3. **FiLM ResBlocks**: Feature-wise Linear Modulation với time
4. **Cross-Attention**: Attend to visual và mean_fmri
5. **Output Head**: Predict x_0

### Training
- Sử dụng `DiffusionManager` với cosine beta schedule
- Loss: MSE giữa predicted x_0 và ground truth

### Inference
- DDPM sampling hoặc DDIM (faster)
- Từ noise → clean signal qua T steps

---

## 3. DiffusionARM3D

**File:** `ARMNet/diffusion_3d_model.py`

### Mô tả
Mở rộng DiffusionARM sang 3D CNN để tận dụng cấu trúc không gian của não bộ.

### Input
- `vis_feat`: **[B, 768]** - Visual features
- `noisy_fmri_3d`: **[B, 1, 26, 26, 24]** - 3D noisy fMRI volume
- `t`: **[B]** - Timestep
- `mask`: **[26, 26, 24]** (optional) - Brain mask

### Output
- **[B, 1, 26, 26, 24]** - Predicted clean 3D fMRI

### Kiến trúc chính
1. **Reshape3D**: Convert 1D (15724) ↔ 3D (26×26×24)
2. **3D U-Net**: Encoder-Decoder với skip connections
3. **Conv3DBlock**: 3D conv + GroupNorm + FiLM time conditioning
4. **SpatialCrossAttention3D**: Cross-attention ở mức spatial
5. **AdaLN-Zero**: Adaptive LayerNorm cho conditioning tốt hơn

### Cải tiến
- **Classifier-Free Guidance (CFG)**: Tăng cường visual conditioning
- **Multi-scale cross-attention**: Ở nhiều levels của U-Net
- **3D spatial modeling**: Học cấu trúc không gian tốt hơn

---

## 4. DiffusionGraphARM

**File:** `ARMNet/diffusion_graph_model.py`

### Mô tả
Sử dụng Graph Neural Networks thay vì 3D CNN, xử lý fMRI như point cloud với graph structure.

### Input
- `noisy_fmri`: **[N]** hoặc **[B, N]** - Noisy fMRI values (N=15724)
- `coords`: **[N, 3]** hoặc **[B, N, 3]** - Spatial coordinates của voxels
- `edge_index`: **[2, E]** - Graph edges (KNN hoặc radius graph)
- `vis_feat`: **[B, 768]** - Visual features
- `t`: **[B]** - Timestep
- `batch`: **[N]** - Batch assignment cho batched graphs

### Output
- **[N]** hoặc **[B, N]** - Predicted noise

### Kiến trúc chính
1. **GraphConvBlock**: GAT hoặc GCN với time conditioning
2. **GraphResBlock**: Residual blocks cho graphs
3. **GraphCrossAttention**: Nodes attend to visual features
4. **GraphUNet**: U-Net architecture cho graphs

### Yêu cầu
- `torch_geometric` library
- Graph construction (KNN hoặc radius-based)

---

## 5. HierarchicalCVAE

**File:** `ARMNet/cvae_model.py`

### Mô tả
Conditional Variational Autoencoder với hierarchical structure (parcels → voxels).

### Input (Training)
- `visual_feat`: **[B, 768]** - Visual features
- `target_parcels`: **[B, 1000]** - Ground truth parcels (chỉ khi training)

### Input (Inference)
- `visual_feat`: **[B, 768]** - Visual features

### Output
- **Training**: `(pred_parcels, mu, logvar)` - **[B, 1000]**, **[B, 128]**, **[B, 128]**
- **Inference**: `pred_parcels` - **[B, 1000]**

### Kiến trúc chính
1. **Encoder**: Maps (target_parcels + visual) → latent (mu, logvar)
2. **Decoder Stage 1**: (latent z + visual) → predicted parcels
3. **Refiner Stage 2**: (parcels + visual) → voxels (optional)

### Loss
- Reconstruction loss (MSE)
- KL divergence: `KL(q(z|x,c) || p(z))`

### Đặc điểm
- Có thể generate diverse samples từ cùng visual features
- Hierarchical: 768 → 1000 parcels → 15724 voxels

---

## 6. DenoisingARM

**File:** `ARMNet/denoising_model.py`

### Mô tả
Unified Conditional Denoising Model - maps (mean_fMRI + noise) → clear_fMRI.

### Input
- `x_noised`: **[B, 1000]** - mean_fMRI + gaussian noise
- `visual_feat`: **[B, 768]** - DINOv2 embeddings
- `mean_fmri`: **[B, 1000]** - Base mean signal

### Output
- **[B, 1000]** - Denoised fMRI (mean_fmri + delta)

### Kiến trúc chính
1. **Visual Projector**: Project visual features
2. **Input Projection**: Project noisy input
3. **ResBlock + CrossAttention Layers**: FiLM modulation + cross-attention
4. **Refinement Head**: Predict delta
5. **Learnable Output Scale**: `output_scale` parameter

### Công thức
```
output = mean_fmri + delta * output_scale
```

---

## 7. Flow3DHybridARM

**File:** `ARMNet/flow_3d_hybrid_model.py`

### Mô tả
Hybrid GNN+CNN Flow Matching model cho 3D fMRI generation. Sử dụng Optimal Transport flow matching.

### Input (Training)
- `clean_fmri_3d`: **[B, 1, D, H, W]** - Ground truth 3D fMRI (D=61, H=46, W=42)
- `vis_feat`: **[B, 768]** - Visual features
- `mask`: **[D, H, W]** (optional) - Brain mask
- `coords_compact`: **[N_brain, 3]** (optional) - Brain voxel coordinates

### Input (Inference)
- `vis_feat`: **[B, 768]** - Visual features
- `mask`, `coords_compact` (optional)

### Output
- **Training**: Dict với `pred_velocity`, `target_velocity`, `t`, `x_pred`
- **Inference**: **[B, 1, D, H, W]** - Generated 3D fMRI

### Kiến trúc chính
1. **HybridCNN3DUNet**: 3D CNN U-Net + optional graph layers
2. **Conv3DBlock**: 3D conv với FiLM time conditioning
3. **SparseGraphLayer**: Graph layer trên sparse brain voxels
4. **MaskAttention**: Mask-aware attention
5. **FlowMatching3DManager**: OT flow matching

### Flow Matching
- **OT Path**: `x_t = (1-t) * x_0 + t * x_1 + σ * ε`
- **Velocity**: `v_t = x_1 - x_0 + σ * dε/dt`
- **Training**: Predict velocity tại random t
- **Inference**: Integrate từ x_0 → x_1

### Đặc điểm
- Kết hợp CNN (local) và GNN (long-range)
- Mask conditioning để focus vào brain regions
- CFG support
- Lossless 3D ↔ 1D conversion

---

## 8. FlowGraphARM

**File:** `ARMNet/flow_graph_model.py`

### Mô tả
Graph Flow Matching model - kết hợp GNN với Flow Matching framework.

### Input (Training)
- `clean_fmri`: **[B, N]** - Ground truth fMRI (N=15724)
- `coords`: **[B, N, 3]** hoặc **[N, 3]** - Spatial coordinates
- `edge_index`: **[2, E]** - Graph edges
- `vis_feat`: **[B, 768]** - Visual features
- `batch`: **[N]** (optional) - Batch assignment

### Input (Inference)
- `coords`, `edge_index`, `vis_feat`, `batch`

### Output
- **Training**: Dict với `pred_velocity`, `target_velocity`, `t`, `x_pred`
- **Inference**: **[B, N]** - Generated fMRI

### Kiến trúc chính
1. **GraphFlowUNet**: Graph U-Net cho flow matching
2. **GraphConvBlock**: GAT/GCN với time conditioning
3. **GraphCrossAttention**: Cross-attention với visual features
4. **FlowMatchingManager**: OT flow với σ noise

### Đặc điểm
- Tương tự Flow3DHybridARM nhưng pure graph-based
- Không cần reshape 1D ↔ 3D
- Yêu cầu `torch_geometric`

---

## 9. DualStreamCNN

**File:** `ARMNet/dual_stream_cnn_model.py`

### Mô tả
Dual-stream architecture với 3D CNN và 1D CNN, sử dụng ConvNeXt blocks.

### Input
- `vis_feat`: **[B, 768]** - Visual features (DINOv2)
- `mask`: **[D, H, W]** - Brain mask (D=61, H=46, W=42)
- `coords_compact`: **[N, 3]** - Voxel coordinates (N=15724)

### Output
- Dict với:
  - `pred_fmri_1d`: **[B, N]** - Final prediction
  - `coarse_1d`: **[B, N]** - Coarse prediction
  - `feat_3d`: **[B, C_3d, D', H', W']** - 3D features
  - `feat_1d`: **[B, C_1d, N']** - 1D features

### Kiến trúc chính
1. **Coarse Generator**: Visual → initial fMRI guess
2. **Stream3D**: ConvNeXt 3D blocks với mask conditioning
3. **Stream1D**: ConvNeXt 1D blocks trên voxel sequence
4. **Visual Conditioning**: Inject visual features vào cả 2 streams
5. **FusionModule**: Kết hợp 3D và 1D features

### ConvNeXt Blocks
- **3D**: Depthwise conv 7×7×7 + LayerNorm + expansion
- **1D**: Depthwise conv 7 + LayerNorm + expansion
- Layer scale + stochastic depth

### Fusion Types
- `'concat'`: Concatenate features
- `'attention'`: Cross-attention giữa 3D và 1D
- `'gated'`: Gated fusion

### Đặc điểm
- **Dual-stream**: Tận dụng cả spatial structure (3D) và voxel-level details (1D)
- **Mask-aware**: MaskAwareConv3D chỉ học brain regions
- **Coarse-to-fine**: Coarse prediction → refinement
- **ConvNeXt**: Modern CNN architecture với better performance

---

## So Sánh Các Kiến Trúc

| Model | Input Type | Output Type | Representation | Special Features |
|-------|-----------|-------------|----------------|------------------|
| **ARMNet** | 1D (15724) | 1D (15724) | MLP | Baseline, parcel guidance |
| **DiffusionARM** | 1D (1000) | 1D (1000) | MLP + Attention | DDPM, cross-attention |
| **DiffusionARM3D** | 3D (26×26×24) | 3D (26×26×24) | 3D CNN U-Net | Spatial modeling, CFG |
| **DiffusionGraphARM** | Graph (15724 nodes) | Graph | GNN U-Net | Point cloud, long-range |
| **HierarchicalCVAE** | 1D (1000) | 1D (1000) | VAE | Diversity, hierarchical |
| **DenoisingARM** | 1D (1000) | 1D (1000) | MLP + Attention | Single-step denoising |
| **Flow3DHybridARM** | 3D (61×46×42) | 3D (61×46×42) | Hybrid CNN+GNN | Flow matching, mask-aware |
| **FlowGraphARM** | Graph (15724 nodes) | Graph | GNN | Flow matching, pure graph |
| **DualStreamCNN** | 3D + 1D | 1D (15724) | Dual CNN | ConvNeXt, coarse-to-fine |

---

## Lựa Chọn Kiến Trúc

### Khi nào dùng gì?

1. **ARMNet**: Baseline đơn giản, nhanh, ít tài nguyên
2. **DiffusionARM**: Cần quality cao, có thời gian inference
3. **DiffusionARM3D**: Cần spatial modeling, có GPU mạnh
4. **DiffusionGraphARM**: Cần long-range interactions, có torch_geometric
5. **HierarchicalCVAE**: Cần generate diverse samples
6. **DenoisingARM**: Cần inference nhanh (single-step)
7. **Flow3DHybridARM**: Best of both worlds (CNN + GNN), state-of-the-art
8. **FlowGraphARM**: Pure graph approach, flexible
9. **DualStreamCNN**: Cần kết hợp spatial + voxel-level, modern architecture

### Xu hướng phát triển
- **Diffusion → Flow Matching**: Faster inference, better quality
- **1D → 3D/Graph**: Better spatial modeling
- **Single-stream → Dual-stream**: Complementary information
- **MLP → CNN/GNN**: Inductive biases cho brain structure

---

## Tài Liệu Tham Khảo

- **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
- **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
- **ConvNeXt**: Liu et al., "A ConvNet for the 2020s", CVPR 2022
- **GAT**: Veličković et al., "Graph Attention Networks", ICLR 2018

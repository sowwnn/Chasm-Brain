# DenoisingARM - Hướng Dẫn Sử Dụng với 15k Voxels

## ✅ Đã Cập Nhật

DenoisingARM giờ đây **hỗ trợ linh hoạt** cả 1000 parcels và 15,724 voxels!

## 🎯 Các Mode Hoạt Động

### Mode 1: 1000 Parcels (Original)
- **Ưu điểm**: Nhanh, ít memory, training ổn định
- **Nhược điểm**: Mất spatial detail, cần reconstruct về 15k

### Mode 2: 15,724 Voxels (Mới)
- **Ưu điểm**: Giữ full spatial detail, không cần reconstruct
- **Nhược điểm**: Chậm hơn, tốn memory hơn, khó train hơn

---

## 🚀 Cách Sử Dụng

### Option 1: Training với 1000 Parcels (Khuyến nghị cho bắt đầu)

```bash
python scripts/train_denoising.py --config configs/train_denoising.yaml
```

**Config**: `configs/train_denoising.yaml`
```yaml
model:
  fmri_dim: 1000           # 1000 parcels

data:
  use_parcellation: true   # BẮT BUỘC
  parcel_labels_path: "dataset/nsd/subj01/betas_all_subj01_fp32_renorm_kmeans1000_labels.npy"
```

**Reconstruction về 15k**:
```python
from data.parcel_utils import ParcelMapper

# Load mapper
parcel_mapper = ParcelMapper.from_files('path/to/labels.npy')

# Predict
pred_1k = model(x_noised, visual_feat, mean_fmri)  # [B, 1000]

# Reconstruct to 15k
pred_15k = parcel_mapper.reconstruct(pred_1k.cpu().numpy())  # [B, 15724]
```

---

### Option 2: Training với 15,724 Voxels (Mới)

```bash
python scripts/train_denoising.py --config configs/train_denoising_15k.yaml
```

**Config**: `configs/train_denoising_15k.yaml`
```yaml
model:
  fmri_dim: 15724          # 15k voxels trực tiếp
  hidden_dim: 1024         # Tăng capacity
  num_layers: 8            # Nhiều layers hơn
  dropout: 0.3             # Dropout cao hơn

data:
  use_parcellation: false  # QUAN TRỌNG: Tắt parcellation
  parcel_labels_path: null # Không cần
  
training:
  batch_size: 32           # Giảm batch size (15k lớn hơn)
```

**Inference**:
```python
# Predict trực tiếp 15k
pred_15k = model(x_noised, visual_feat, mean_fmri)  # [B, 15724]
# Không cần reconstruct!
```

---

## 📊 So Sánh Performance

| Metric | 1000 Parcels | 15724 Voxels |
|--------|--------------|--------------|
| **Training Speed** | ⚡⚡⚡⚡⚡ Fast | ⚡⚡⚡ Medium |
| **Memory Usage** | 💾💾 ~4GB | 💾💾💾💾 ~8-10GB |
| **Spatial Detail** | ⭐⭐⭐ Good (after reconstruct) | ⭐⭐⭐⭐⭐ Excellent |
| **Training Stability** | ⭐⭐⭐⭐⭐ Very Stable | ⭐⭐⭐ Moderate |
| **Batch Size** | 64 | 32 |
| **Recommended For** | Quick experiments | Final model |

---

## 🔧 Hyperparameters Khuyến Nghị

### Cho 1000 Parcels
```yaml
model:
  hidden_dim: 512
  num_layers: 12
  num_heads: 4
  dropout: 0.5

training:
  batch_size: 64
  lr: 0.0001
  weight_decay: 0.3
```

### Cho 15724 Voxels
```yaml
model:
  hidden_dim: 1024         # Tăng gấp đôi
  num_layers: 8            # Giảm một chút để tránh overfit
  num_heads: 8             # Nhiều heads hơn
  dropout: 0.3             # Dropout vừa phải

training:
  batch_size: 32           # Giảm batch size
  lr: 0.0001
  weight_decay: 0.1        # Weight decay vừa phải
```

---

## 💡 Tips & Tricks

### 1. Bắt đầu với Parcels
Nếu bạn mới bắt đầu, hãy train với **1000 parcels** trước:
- Nhanh hơn để experiment
- Ổn định hơn
- Dễ debug hơn

Sau khi có baseline tốt, chuyển sang **15k voxels**.

### 2. Gradient Accumulation cho 15k
Nếu GPU không đủ memory cho batch_size=32:

```yaml
training:
  batch_size: 16           # Giảm batch size
  gradient_accumulation: 2 # Accumulate 2 steps
  # Effective batch size = 16 * 2 = 32
```

### 3. Mixed Precision
Luôn dùng FP16 cho 15k để tiết kiệm memory:

```yaml
training:
  precision: "fp16"
```

### 4. Regularization cho 15k
15k voxels dễ overfit hơn, tăng regularization:

```yaml
model:
  dropout: 0.3-0.4         # Cao hơn

training:
  weight_decay: 0.1-0.2    # Cao hơn
  vis_noise_std: 0.2-0.3   # Nhiễu visual features
  
data:
  augment_noise_train: true
  noise_std: 0.15
```

---

## 🐛 Troubleshooting

### Lỗi: CUDA Out of Memory (15k)

**Giải pháp 1**: Giảm batch size
```yaml
training:
  batch_size: 16  # hoặc 8
```

**Giải pháp 2**: Giảm hidden_dim
```yaml
model:
  hidden_dim: 768  # thay vì 1024
```

**Giải pháp 3**: Giảm num_layers
```yaml
model:
  num_layers: 6  # thay vì 8
```

### Lỗi: Model không converge (15k)

**Nguyên nhân**: 15k khó train hơn

**Giải pháp**:
1. Giảm learning rate: `lr: 5e-5`
2. Tăng warmup: Thêm warmup scheduler
3. Tăng regularization: `dropout: 0.4`, `weight_decay: 0.2`
4. Kiểm tra data: Đảm bảo normalization đúng

### Lỗi: Pearson correlation thấp (15k)

**Giải pháp**:
1. Tăng `pearson_weight` trong loss: `0.8 → 1.0`
2. Tăng `std_weight`: `1.0 → 1.5`
3. Giảm `alpha`: `2.5 → 2.0`
4. Train lâu hơn: `max_epochs: 200 → 300`

---

## 📈 Monitoring Training

### Metrics quan trọng

**Cho 1000 Parcels**:
- `Pearson/Train_1k`: Correlation trên parcels
- `Pearson/Val_1k`: Validation correlation
- `Pearson/Val_15k`: Correlation sau reconstruct (quan trọng nhất!)

**Cho 15724 Voxels**:
- `Pearson/Train_15k`: Training correlation
- `Pearson/Val_15k`: Validation correlation
- `Loss/train`, `Loss/val`: Training/validation loss

### TensorBoard
```bash
tensorboard --logdir checkpoints/denoising_15k_subj01/logs
```

---

## 🎓 Best Practices

### 1. Progressive Training
```bash
# Step 1: Train với 1k parcels (nhanh, ổn định)
python scripts/train_denoising.py --config configs/train_denoising.yaml

# Step 2: Đánh giá kết quả reconstruct về 15k

# Step 3: Nếu cần quality cao hơn → train với 15k
python scripts/train_denoising.py --config configs/train_denoising_15k.yaml
```

### 2. Transfer Learning (Nâng cao)
Có thể load weights từ 1k model làm initialization cho 15k:

```python
# Load 1k model
model_1k = DenoisingARM(fmri_dim=1000, ...)
model_1k.load_state_dict(torch.load('best_1k.pth'))

# Create 15k model
model_15k = DenoisingARM(fmri_dim=15724, ...)

# Transfer shared weights (vis_projector, layers)
# Chỉ khác input_proj và refinement_head
```

### 3. Ensemble
Kết hợp cả 2 models:
```python
# Predict với 1k
pred_1k = model_1k(...)
pred_15k_from_1k = parcel_mapper.reconstruct(pred_1k)

# Predict với 15k
pred_15k_direct = model_15k(...)

# Ensemble
pred_final = 0.5 * pred_15k_from_1k + 0.5 * pred_15k_direct
```

---

## 📝 Example Commands

### Training 1k Parcels
```bash
python scripts/train_denoising.py \
    --config configs/train_denoising.yaml
```

### Training 15k Voxels
```bash
python scripts/train_denoising.py \
    --config configs/train_denoising_15k.yaml
```

### Resume Training
```bash
python scripts/train_denoising.py \
    --config configs/train_denoising_15k.yaml \
    --resume checkpoints/denoising_15k_subj01/best_model.pth
```

---

## 🔍 Code Changes Summary

### 1. Model (`ARMNet/denoising_model.py`)
```python
# Before
def __init__(self, visual_dim=768, fmri_dim=1000, ...):

# After
def __init__(self, visual_dim=768, fmri_dim=15724, ...):
# Giờ default là 15k, nhưng có thể set bất kỳ giá trị nào
```

### 2. Config (`configs/train_denoising_15k.yaml`)
```yaml
model:
  fmri_dim: 15724  # Có thể là 1000, 15724, hoặc bất kỳ số nào

data:
  use_parcellation: false  # false cho 15k, true cho 1k
  parcel_labels_path: null # null cho 15k
```

### 3. Training Script (`scripts/train_denoising.py`)
```python
# Auto-detect parcellation mode
parcel_mapper = None
if config['data'].get('use_parcellation', False):
    parcel_mapper = ParcelMapper.from_files(...)
```

---

## ✨ Kết Luận

DenoisingARM giờ đây **linh hoạt hoàn toàn**:
- ✅ Hỗ trợ 1000 parcels (nhanh, ổn định)
- ✅ Hỗ trợ 15,724 voxels (quality cao)
- ✅ Hỗ trợ bất kỳ dimension nào khác

**Khuyến nghị**:
- Bắt đầu với **1k parcels** để experiment nhanh
- Chuyển sang **15k voxels** khi cần quality cao nhất
- Sử dụng **ensemble** cả 2 models cho kết quả tốt nhất

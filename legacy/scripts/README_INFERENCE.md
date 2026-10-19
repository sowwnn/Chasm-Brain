# I2fMRI Inference Scripts

Các script inference để generate predictions và visualizations cho 10 mẫu test.

## Scripts Available

### 1. `inference_hierarchical_samples.py` ✅ (Recommended - có checkpoint)

Script inference cho **HierarchicalARM_V2** model (có checkpoint sẵn).

**Chạy:**
```bash
cd /home/sowwn/Workspace/ws/2026/I2fMRI
python scripts/inference_hierarchical_samples.py
```

**Yêu cầu:**
- Checkpoint: `checkpoints/hierarchical_v21_subj01/best_model.pth` ✅
- Config: `configs/train_hierarchical_v2.yaml` ✅
- 3D mapping: `data/fmri_3d_compact_mapping_subj01.npz` ✅
- Sub-ROI clusters: `data/sub_roi_labels.npy` ✅

**Output directory:** `inference_results_hierarchical/`

---

### 2. `inference_samples.py` (Cần train model trước)

Script inference cho **DenoisingARM** model.

**Chạy:**
```bash
cd /home/sowwn/Workspace/ws/2026/I2fMRI
python scripts/inference_samples.py
```

**Yêu cầu:**
- ❌ Checkpoint: `checkpoints/denoising_15k_subj01/best_model.pth` (chưa có - cần train)
- Config: `configs/train_denoising_15k.yaml` ✅
- 3D mapping: `data/fmri_3d_compact_mapping_subj01.npz` ✅

**Output directory:** `inference_results/`

**Nếu muốn sử dụng script này:**
```bash
# Train model trước
python scripts/train_denoising.py --config configs/train_denoising_15k.yaml

# Sau đó mới chạy inference
python scripts/inference_samples.py
```

---

## Output Structure

Mỗi sample sẽ được lưu trong thư mục `sample_XXX/` với các files sau:

```
inference_results_hierarchical/
├── sample_000/
│   ├── input_image.jpeg           # Input image từ NSD dataset
│   ├── predict.npy                # Predicted fMRI voxels [15724]
│   ├── groundtruth.npy            # Ground truth fMRI voxels [15724]
│   ├── predict_brain.nii.gz       # 3D brain prediction (NIfTI format)
│   ├── groundtruth_brain.nii.gz   # 3D brain ground truth (NIfTI format)
│   ├── error_brain.nii.gz         # 3D error map (NIfTI format)
│   ├── brain_ortho.png            # Orthogonal slices visualization
│   ├── brain_glass.png            # Glass brain visualization
│   ├── brain_error.png            # Error map visualization
│   ├── voxel_comparison_1d.png    # 1D voxel-wise comparison plot
│   └── metadata.json              # Metrics và thông tin mẫu
├── sample_001/
│   └── ...
├── ...
├── sample_009/
│   └── ...
├── summary.json                   # Tổng hợp metrics cho tất cả samples
└── summary_metrics.png            # Visualization của metrics tổng hợp
```

### File Descriptions

1. **input_image.jpeg**: Ảnh input từ NSD dataset (425x425 RGB)

2. **predict.npy**: NumPy array shape `[15724]` - predicted fMRI activations
   ```python
   pred = np.load('predict.npy')
   ```

3. **groundtruth.npy**: NumPy array shape `[15724]` - ground truth fMRI activations
   ```python
   gt = np.load('groundtruth.npy')
   ```

4. **predict_brain.nii.gz**: NIfTI file có thể load bằng `nibabel` hoặc view trong FSLeyes/MRIcroGL
   ```python
   import nibabel as nib
   nii = nib.load('predict_brain.nii.gz')
   ```

5. **metadata.json**: Thông tin và metrics
   ```json
   {
     "sample_idx": 0,
     "image_id": 123,
     "correlation": 0.5234,
     "mae": 0.1234,
     "mse": 0.0234,
     "pred_mean": 0.0123,
     "pred_std": 0.4567,
     "gt_mean": 0.0234,
     "gt_std": 0.4321
   }
   ```

6. **Visualizations**:
   - `brain_ortho.png`: 3 orthogonal slices (sagittal, coronal, axial)
   - `brain_glass.png`: Glass brain view showing all activations
   - `brain_error.png`: Absolute error heatmap
   - `voxel_comparison_1d.png`: Line plot comparing predictions vs ground truth

### Summary Files

1. **summary.json**: Tổng hợp metrics cho tất cả samples
   ```json
   {
     "num_samples": 10,
     "mean_correlation": 0.5234,
     "std_correlation": 0.0234,
     "mean_mae": 0.1234,
     "std_mae": 0.0123,
     "mean_mse": 0.0234,
     "samples": [...]
   }
   ```

2. **summary_metrics.png**: Bar charts showing correlation và MAE cho từng sample

---

## Usage Examples

### Load và analyze kết quả

```python
import numpy as np
import json
from pathlib import Path

# Load results
results_dir = Path('inference_results_hierarchical')

# Load summary
with open(results_dir / 'summary.json') as f:
    summary = json.load(f)

print(f"Mean correlation: {summary['mean_correlation']:.4f}")

# Load specific sample
sample_dir = results_dir / 'sample_000'
pred = np.load(sample_dir / 'predict.npy')
gt = np.load(sample_dir / 'groundtruth.npy')

with open(sample_dir / 'metadata.json') as f:
    meta = json.load(f)

print(f"Sample correlation: {meta['correlation']:.4f}")
```

### Visualize 3D brain

```python
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt

# Load NIfTI
nii_pred = nib.load('sample_000/predict_brain.nii.gz')
nii_gt = nib.load('sample_000/groundtruth_brain.nii.gz')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

plotting.plot_stat_map(
    nii_gt,
    title='Ground Truth',
    cut_coords=(0, 0, 0),
    display_mode='ortho',
    axes=axes[0],
    figure=fig
)

plotting.plot_stat_map(
    nii_pred,
    title='Prediction',
    cut_coords=(0, 0, 0),
    display_mode='ortho',
    axes=axes[1],
    figure=fig
)

plt.show()
```

### Compute additional metrics

```python
import numpy as np
from scipy.stats import pearsonr

pred = np.load('sample_000/predict.npy')
gt = np.load('sample_000/groundtruth.npy')

# Pearson correlation
corr, p_value = pearsonr(pred, gt)
print(f"Correlation: {corr:.4f} (p={p_value:.4e})")

# MAE
mae = np.mean(np.abs(pred - gt))
print(f"MAE: {mae:.4f}")

# RMSE
rmse = np.sqrt(np.mean((pred - gt)**2))
print(f"RMSE: {rmse:.4f}")

# R² score
from sklearn.metrics import r2_score
r2 = r2_score(gt, pred)
print(f"R² score: {r2:.4f}")
```

---

## Troubleshooting

### Lỗi: "Checkpoint not found"
- Đảm bảo bạn đã train model hoặc có checkpoint sẵn
- Kiểm tra đường dẫn trong config file

### Lỗi: "3D mapping not found"
- File `data/fmri_3d_compact_mapping_subj01.npz` bị thiếu
- Tạo lại bằng script trong `notebook/` (nếu có)

### Lỗi: "Image not found"
- NSD stimuli images chưa được download
- Script sẽ tự động tạo placeholder images màu xám

### Lỗi CUDA out of memory
- Giảm batch_size trong dataloader (dòng 129/131)
- Hoặc giảm số samples: `num_samples = 5`

---

## Notes

- Script tự động sử dụng GPU nếu có, fallback về CPU nếu không
- Inference time: ~30-60 giây cho 10 samples (tùy hardware)
- Total disk space cần: ~100-200 MB cho 10 samples
- NIfTI files có thể được view trong FSLeyes, MRIcroGL, hoặc SPM

---

## Contact

Nếu có vấn đề, kiểm tra:
1. Các file dependencies đã đủ chưa
2. Config paths đúng chưa
3. Python environment đã activate chưa

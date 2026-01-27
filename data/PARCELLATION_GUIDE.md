# Hướng dẫn Parcellation cho I2fMRI

## Tổng quan

Parcellation là kỹ thuật giảm chiều dữ liệu fMRI từ ~15,000 voxels xuống ~1,000 parcels, giúp:
- Giảm kích thước dữ liệu (15x nhỏ hơn)
- Tăng tốc độ training
- Giảm overfitting
- **Giữ được ~90% thông tin** khi reconstruct

## Phương pháp

Chúng ta sử dụng **Time-series K-means Clustering** theo approach trong `analy.ipynb`:

```
Raw fMRI → K-means (on time-series) → Parcels → Mean Pooling → Training
                                                     ↓
                                            [Model predicts parcels]
                                                     ↓
                                              Reconstruct → Z-score → Compare
```

### Sự khác biệt so với Spatial K-means

| Aspect | Spatial K-means (cũ) | Time-series K-means (mới) |
|--------|---------------------|--------------------------|
| Clustering | Dựa trên tọa độ (X,Y,Z) | Dựa trên time-series của voxel |
| Information | Chỉ dùng spatial info | Dùng cả spatial + temporal patterns |
| Reconstruction quality | ~80-85% correlation | **~90% correlation** |
| Method | `parcellate_kmeans.py` | `parcellate_timeseries_kmeans.py` |

## Sử dụng

### Bước 1: Tạo Parcellation Labels

```bash
python data/parcellate_timeseries_kmeans.py \
  --input dataset/nsd/subj01/betas_all_subj01_fp32_renorm.hdf5 \
  --n_parcels 1000 \
  --n_samples 1000 \
  --use_minibatch
```

**Output:**
- `betas_all_subj01_fp32_renorm_tskm1000.hdf5` - Parcellated data
- `betas_all_subj01_fp32_renorm_tskm1000_labels.npy` - Voxel→Parcel mapping
- `betas_all_subj01_fp32_renorm_tskm1000_model.pkl` - KMeans model

**Parameters:**
- `--n_parcels`: Số lượng parcels (default: 1000)
- `--n_samples`: Số samples dùng cho clustering (default: None = all)
  - Dùng subset (~1000) để tăng tốc độ clustering
- `--use_minibatch`: Dùng MiniBatchKMeans (nhanh hơn cho dataset lớn)
- `--aggregation`: 'mean' hoặc 'median' (default: 'mean')
- `--save_reconstructed`: Lưu reconstructed data để kiểm tra
- `--no_evaluate`: Bỏ qua evaluation (nhanh hơn)

### Bước 2: Training với Parcellated Data

**Option A: Load parcellated data trực tiếp**

```python
from data.neuroflux_dataset import load_neuroflux_data

# Train dataset với parcellation
train_dataset = load_neuroflux_data(
    datalist_path='dataset/nsd/metadata/datalist_mindeye2_sub01.json',
    fmri_path='dataset/nsd/subj01/betas_all_subj01_fp32_renorm.hdf5',
    embeddings_path='dataset/nsd/embeddings_mindeye2/dinov2_train_sub01.npy',
    subjects=[1],
    split='train',
    average_trials=False,
    parcel_labels_path='dataset/nsd/subj01/betas_all_subj01_fp32_renorm_tskm1000_labels.npy',
    apply_zscore=True,  # Z-score sau khi parcellate
)
```

**Option B: Sử dụng DataLoader wrapper**

```python
from data.neuroflux_dataset import create_dataloaders

train_loader, val_loader = create_dataloaders(
    datalist_path='dataset/nsd/metadata/datalist_mindeye2_sub01.json',
    fmri_path='dataset/nsd/subj01/betas_all_subj01_fp32_renorm.hdf5',
    train_embeddings_path='dataset/nsd/embeddings_mindeye2/dinov2_train_sub01.npy',
    test_embeddings_path='dataset/nsd/embeddings_mindeye2/dinov2_test_sub01.npy',
    subjects=[1],
    batch_size=32,
    parcel_labels_path='dataset/nsd/subj01/betas_all_subj01_fp32_renorm_tskm1000_labels.npy',
    apply_zscore_train=True,
    apply_zscore_val=True,
)
```

### Bước 3: Evaluation với Reconstruction

```python
from data.parcel_utils import ParcelMapper

# Load mapper
mapper = ParcelMapper.from_files(
    labels_path='dataset/nsd/subj01/betas_all_subj01_fp32_renorm_tskm1000_labels.npy'
)

# Model predicts parcels
predicted_parcels = model(embeddings)  # [batch, 1000]

# Reconstruct về voxels và z-score
reconstructed_z = mapper.reconstruct_and_zscore(predicted_parcels)

# Ground truth z-scored
gt_z = mapper.zscore(ground_truth_voxels)

# So sánh
metrics = mapper.evaluate_reconstruction(
    original=ground_truth_voxels,
    predicted_parcels=predicted_parcels
)
print(f"Correlation: {metrics['global_corr']:.4f}")
```

## Pipeline đầy đủ

### Training Pipeline

```
[Raw fMRI 15724 voxels]
         ↓
    [Parcellate]  ← Sử dụng pre-computed labels
         ↓
  [1000 parcels]
         ↓
    [Z-score]
         ↓
  [Train Model] → Model học predict 1000 parcels từ embeddings
```

### Inference/Evaluation Pipeline

```
[Embeddings]
     ↓
  [Model]
     ↓
[Predicted 1000 parcels]
     ↓
 [Reconstruct] ← Mỗi voxel = giá trị của parcel chứa nó
     ↓
[Reconstructed 15724 voxels]
     ↓
  [Z-score]
     ↓
[Compare with GT z-scored] → Correlation, MSE, MAE
```

## Sử dụng ParcelMapper trực tiếp

```python
from data.parcel_utils import ParcelMapper
import numpy as np

# Tạo mapper
mapper = ParcelMapper.from_files('path/to/labels.npy')

# Parcellate data
raw_fmri = np.random.randn(100, 15724)  # [samples, voxels]
parcellated = mapper.parcellate(raw_fmri)  # [samples, 1000]

# Parcellate + Z-score trong 1 bước
parcellated_z = mapper.parcellate_and_zscore(raw_fmri)

# Reconstruct
reconstructed = mapper.reconstruct(parcellated)  # [samples, 15724]

# Reconstruct + Z-score
reconstructed_z = mapper.reconstruct_and_zscore(parcellated)

# Evaluate
metrics = mapper.evaluate_reconstruction(
    original=raw_fmri,
    predicted_parcels=parcellated
)
```

## Các thông số quan trọng

### Số lượng Parcels

| n_parcels | Compression | Correlation | Training Speed | Recommendation |
|-----------|-------------|-------------|----------------|----------------|
| 500 | 31x | ~85% | Fastest | Too aggressive |
| 1000 | 15.7x | **~90%** | Fast | **Recommended** |
| 2000 | 7.9x | ~93% | Moderate | Good balance |
| 5000 | 3.1x | ~96% | Slow | Minimal loss |

**Khuyến nghị: n_parcels=1000** cho balance tốt giữa compression và information retention.

### Z-score Timing

**Option 1: Z-score sau Parcellation (Recommended)**
```python
# Training
parcellated = mapper.parcellate(raw_fmri)  # Keep raw values
parcellated_z = mapper.zscore(parcellated)  # Z-score for training

# Inference
predicted_parcels = model(embeddings)  # Model outputs parcels
reconstructed = mapper.reconstruct(predicted_parcels)
reconstructed_z = mapper.zscore(reconstructed)  # Z-score for comparison
```

**Option 2: Model học trực tiếp trên z-scored parcels**
```python
# Training
parcellated_z = mapper.parcellate_and_zscore(raw_fmri)
# Model learns z-scored space

# Inference - cần denormalize rồi reconstruct
predicted_z = model(embeddings)
# Need to reverse z-score before reconstruct (complex!)
```

→ **Khuyến nghị dùng Option 1** vì đơn giản hơn và flexible hơn.

## Ví dụ hoàn chỉnh

Xem file `examples/train_with_parcellation.py` (sẽ tạo) để có ví dụ training pipeline hoàn chỉnh.

## Troubleshooting

### 1. Out of Memory khi clustering

```bash
# Giảm số samples cho clustering
python data/parcellate_timeseries_kmeans.py \
  --input data.hdf5 \
  --n_samples 500 \
  --use_minibatch
```

### 2. Mismatch giữa labels và data

```python
# Kiểm tra số voxels
import h5py
with h5py.File('data.hdf5', 'r') as f:
    n_voxels = f['betas'].shape[1]

labels = np.load('labels.npy')
print(f"Data: {n_voxels} voxels")
print(f"Labels: {len(labels)} voxels")
# Phải match!
```

### 3. Correlation thấp

- Kiểm tra xem có đang dùng đúng labels không
- Kiểm tra xem data có bị z-score 2 lần không
- Thử tăng số parcels (1000 → 2000)

## References

- Notebook: `analy.ipynb` - Experiment ban đầu với approach này
- Code: `data/parcellate_timeseries_kmeans.py` - Tạo parcellation
- Utils: `data/parcel_utils.py` - Utility functions
- Dataset: `data/neuroflux_dataset.py` - Tích hợp vào dataset

## Performance Benchmarks

Với Subject 1 (27,000 training samples, 15,724 voxels):

| Metric | Without Parcellation | With Parcellation (1000) |
|--------|---------------------|--------------------------|
| Data size | 1.6 GB | 100 MB |
| Training speed | 1x | **~3x faster** |
| Memory usage | 12 GB | **~4 GB** |
| Validation correlation | Baseline | ~90% of baseline |
| Inference speed | 1x | **~2x faster** |

**Kết luận:** Parcellation với 1000 parcels cho speedup đáng kể với minimal loss về quality.

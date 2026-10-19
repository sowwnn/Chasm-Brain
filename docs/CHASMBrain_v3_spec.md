<!-- Source: https://claude.ai/artifact/CdxEjShZUBud9hVMgAPkTi (Claude Doc, rev 11, exported 2026-10-20). This file is the reference spec for CHASMBrain v3. -->

# CHASMBrain v3 — Tài liệu kỹ thuật triển khai

2026-10-19 · @Sho

## 1. Tổng quan

CHASMBrain v3 dự đoán fMRI vùng thị giác từ ảnh bằng **coarse-to-fine trên một basis lồng nhau**: một model duy nhất, điều kiện theo mức phân giải t ∈ [0, 1], đi từ thành phần thô (tần số không gian thấp) tới đủ ~15k voxel, mỗi bước chỉ **thêm** chi tiết, không sửa cái đã đoán.

Ba thứ thay đổi so với bản WACV:

- Đơn vị thô là **eigenmode của Laplacian trên đồ thị voxel** (hoặc tầng của cây cụm), không còn cụm 30 voxel + top-70%.
- Hai stage cứng thành **một model điều kiện theo t**. Mamba, Eq.7, VAE bị bỏ.
- Calibration biên độ là **công thức cố định**, áp cho mọi baseline như nhau.

Nguyên tắc thiết kế cho team:

1. Mọi thống kê (z-score, cây cụm, eigenmode, noise ceiling, hệ số calibration) tính **chỉ trên 9,000 ảnh train**.
2. Tách **1,000 ảnh validation** từ 9,000 ảnh train. Shared-1,000 là test,** không chọn hyperparameter trên đó**.
3. Mỗi thí nghiệm có **tiêu chí dừng** viết trước (mục 10). Rớt cổng nào thì dừng, báo lại, không chạy tiếp.
4. Mọi baseline chạy qua **cùng pipeline** (features, split, mask, metric, calibration). Không lấy số từ paper khác.
5. Code dùng PyTorch, config bằng YAML, mỗi run ghi `config.yaml` + `metrics.json` vào thư mục riêng.

## 2. Cấu trúc thư mục và file

Mỗi file làm một việc; tên hàm dưới đây là tên dùng trong code, các mục sau mô tả từng hàm.

```markdown
chasmbrain_v3/
├── configs/
│ ├── data.yaml # đường dẫn NSD, subject, split
│ ├── basis_laplacian.yaml # K mode, connectivity, cách xử lý thành phần rời
│ ├── basis_cluster.yaml # cây cụm, các tầng cắt
│ ├── model.yaml # hidden dim, số bước, lịch t
│ └── baselines/ # ridge.yaml, tben.yaml, ...
├── data/
│ ├── nsd_loader.py # load betas, mask, stimuli, trial ids
│ ├── split.py # train / val / test
│ ├── normalize.py # z-score, noise ceiling
│ └── features.py # DINOv2 / CLIP / ResNet, cache ra .npy
├── basis/
│ ├── graph.py # đồ thị voxel từ mask
│ ├── laplacian.py # eigenmode
│ ├── cluster_tree.py # hierarchical clustering
│ └── nested.py # interface chung: project / lift / band
├── model/
│ ├── encoder.py # CLS + patch pathways (MLP)
│ ├── refiner.py # model điều kiện theo t
│ ├── schedule.py # lịch t cố định và dynamic
│ └── calibrate.py # shrinkage giải tích
├── train/
│ ├── train_refiner.py
│ └── losses.py
├── eval/
│ ├── metrics.py # Pearson, NC-normalized, MSE, per-band
│ ├── stats.py # paired test, bootstrap CI
│ └── run_eval.py
├── baselines/
│ ├── ridge.py
│ ├── tben.py # reproduce, giữ nguyên code cũ
│ ├── single_shot_spectral.py
│ └── smoothed_ridge.py
├── experiments/
│ ├── e0_rescale_tben.py
│ ├── e1_main_comparison.py
│ ├── e2_basis_comparison.py
│ ├── e3_steps_ablation.py
│ ├── e4_band_contribution.py # H1
│ ├── e5_patch_shuffle.py # control retinotopy
│ └── e6_backbones.py
└── outputs/<run_name>/{config.yaml, metrics.json, preds.npy}
```

## 3. Dữ liệu và tiền xử lý

Dùng NSD subject 1, 2, 5, 7; mask vùng thị giác (nsdgeneral) ~15k voxel; target là beta trial-averaged. Mọi thống kê tính trên train.

### `data/nsd_loader.py`

- `load_betas(subj, session_range) -> np.ndarray [n_trials, n_voxels]` — đọc betas trong mask, giữ thứ tự trial id gốc.
- `load_mask(subj) -> (mask_bool [X,Y,Z], coords [n_voxels, 3])` — mask + tọa độ voxel theo thứ tự cột của betas.
- `load_stimuli(subj) -> (image_ids [n_trials], images)` — ảnh và id 73k của NSD.
- `load_roi_labels(subj) -> dict[str, np.ndarray]` — nhãn Early / Midventral / Lateral / Parietal / Midlateral / Ventral, dùng để phân tích,** không** dùng để train.

### `data/split.py`

- `make_split(subj, seed=0) -> dict` — trả về `{train: 8000, val: 1000, test: 1000}` theo image id. `test` = shared-1,000 của NSD. `val` = 1,000 ảnh rút ngẫu nhiên từ 9,000 ảnh subject-specific. Lưu ra `splits/subj{N}.json`,** không đổi** sau khi tạo.
- `average_repeats(betas, image_ids) -> (X [n_images, n_voxels], n_reps [n_images])` — trung bình các trial cùng ảnh.

### `data/normalize.py`

- `fit_zscore(X_train) -> (mu, sigma)` — per-voxel, chỉ trên train.
- `apply_zscore(X, mu, sigma)`.
- `noise_ceiling(betas_single_trial, image_ids) -> nc [n_voxels]` — từ 3 repeat trên **train**, theo công thức NSD (`ncsnr` → `NC = ncsnr² / (ncsnr² + 1/n)`), xuất kèm công thức vào `metrics.json`. Reviewer round 1 đã bắt vì thiếu cái này.
- `reliability_weights(betas_single_trial, image_ids) -> w [n_voxels]` — tỉ lệ variance tín hiệu / tổng, dùng cho calibration (mục 8).

### `data/features.py`

- `extract(backbone: str, images, layer: int) -> (cls [n, d], patch [n, 256, d])` — `backbone ∈ {dinov2_vitb14, clip_vitb16, resnet50}`, cache ra `features/{backbone}_L{layer}_subj{N}.npy`. Backbone **đóng băng**.

Kiểm tra đầu ra (team tự chạy trước khi đi tiếp):

- Pearson giữa hai repeat bất kỳ của cùng ảnh trên train ≈ 0.2–0.4 ở voxel tốt (sanity).
- Không ảnh nào của shared-1,000 xuất hiện trong train hoặc val.
- Trung bình noise ceiling trên mask có số cụ thể, ghi vào doc này.

## 4. Basis lồng nhau

Một basis lồng nhau là một dãy không gian con V₁ ⊂ V₂ ⊂ … ⊂ Vₘ = ℝⁿ. Mức t chọn một không gian con; đi từ t nhỏ lên t lớn là thêm chiều, không đổi chiều cũ. Hai cách dựng, cùng một interface.

### `basis/nested.py` — interface chung

Mọi basis cài ba hàm, model chỉ gọi ba hàm này:

- `project(v [n_voxels], t) -> c [k(t)]` — hệ số của v trong Vₜ.
- `lift(c [k(t)], t) -> v_hat [n_voxels]` — dựng lại từ hệ số (nghịch đảo Moore–Penrose).
- `band(t_lo, t_hi) -> indices` — các chiều được **thêm** khi đi từ t_lo lên t_hi.
- `k(t) -> int` — số chiều tại t.

Bất biến phải giữ: `project(lift(c, t), t) == c` và `lift` tại t_hi khi chỉ dùng hệ số của t_lo cho kết quả **bằng** `lift` tại t_lo. Viết unit test cho hai bất biến này.

### `basis/graph.py`

- `build_voxel_graph(mask, coords, connectivity=26) -> scipy.sparse.csr_matrix A` — cạnh giữa voxel kề nhau trong mask. Trọng số 1 (volume) hoặc exp(−d²/σ²) nếu dùng khoảng cách.
- `connected_components(A) -> labels` — tách bán cầu / đảo nhỏ.
- `handle_components(A, labels, policy)` — `policy ∈ {per_component, bridge}`. `per_component`: tính eigenmode riêng từng mảnh rồi ghép, số mode chia theo kích thước mảnh. `bridge`: nối hai bán cầu bằng cạnh ảo giữa các voxel gần midline nhất. **Mặc định `per_component`**, ghi rõ trong paper.

### `basis/laplacian.py`

- `laplacian(A, normalized=True) -> L` — L = I − D^{-1/2} A D^{-1/2}.
- `eigenmodes(L, K) -> (lambdas [K], U [n_voxels, K])` — K eigenvector nhỏ nhất bằng `scipy.sparse.linalg.eigsh(L, k=K, which='SM')`. Với n ≈ 15k, K ≈ 2,000 chạy trong vài phút. Lưu `basis/laplacian_subj{N}.npz`.
- `LaplacianBasis(U, lambdas)` cài interface `nested.py`: `k(t) = round(t · K)`, `project(v, t) = Uₜᵀ v`, `lift(c, t) = Uₜ c`. Vì U trực giao, `lift` là chuyển vị, bất biến lồng nhau đúng tự động.

Kiểm tra bằng mắt trước khi đi tiếp: vẽ 6 mode đầu lên mask. Kỳ vọng mode 1–3 là trái/phải, trước/sau, V1 vs vùng cao. Nếu mode trùng eccentricity hay polar angle, ghi lại — vừa là quan sát, vừa là confound cho mục 10.

### `basis/cluster_tree.py`

- `fit_tree(X_train [n_images, n_voxels], coords, linkage='ward', spatial_weight) -> tree` — hierarchical clustering trên tương quan response **của train**, có ràng buộc kề nhau qua `A` (dùng `sklearn.cluster.AgglomerativeClustering(connectivity=A)`).
- `cut(tree, n_clusters) -> labels [n_voxels]`.
- `ClusterBasis(tree, levels=[300, 1000, 5000, 15000])` cài interface: `k(t)` = tầng gần nhất, `project` = trung bình theo cụm, `lift` = gán lại giá trị cụm cho voxel. Cụm ở tầng thô là hợp của cụm tầng mịn theo cây, nên lồng nhau tự nhiên.

### `basis/pca_basis.py` (baseline)

- `PCABasis(X_train, K)` — SVD của X_train, cài cùng interface. So với Laplacian để trả lời "tần số hình học hay variance dữ liệu quan trọng hơn".

### Ghi chú cho paper

Low-pass Laplacian gần với spatial smoothing trên surface. Mục 9 có baseline `smoothed_ridge` để chặn lời phê "các anh phát hiện lại smoothing". Bản cuối nên thay đồ thị volume bằng đồ thị surface (FreeSurfer của NSD) vì đồ thị volume nối voxel qua rãnh não.

## 5. Model

Một model duy nhất `Refiner` nhận (features ảnh, hệ số hiện tại c_prev, mức t_prev, mức t) và xuất hệ số cho các chiều mới trong `band(t_prev, t)`. Nó **không** được xuất lại các chiều cũ.

### `model/encoder.py`

- `class DualPathEncoder(d_in, d_hidden)`
 - `forward(cls [B, d], patch [B, 256, d]) -> (h_cls [B, d_hidden], h_patch [B, d_hidden], h_patch_tokens [B, 256, d_hidden])`
 - CLS: 2-layer MLP + GELU. Patch: linear projection + learned positional embedding + 2-layer MLP + mean-pool. Không Mamba, không attention.
- `class FusionGate(d_hidden)` — `forward(h_cls, h_patch) -> alpha [B, 1]`, softmax hai nhánh, giữ lại để làm phân tích H1 (mục 10).

### `model/refiner.py`

- `class Refiner(encoder, basis, d_hidden, t_embed_dim=64)`
 - `forward(cls, patch, c_prev [B, k(t_prev)], t_prev, t) -> c_new [B, |band(t_prev, t)|]`
 - Luồng: `h = encoder(cls, patch)`; `e_t = sinusoidal(t) ⊕ sinusoidal(t_prev)` → MLP; `z = MLP([h_fused; e_t; proj(c_prev)])`; `head(z) -> c_new`. Head là một linear chung cho mọi chiều, cộng một **band embedding** học được cho từng chiều (giống positional embedding), để một head phục vụ mọi t.
 - `step(cls, patch, c_prev, t_prev, t) -> c_full [B, k(t)]` = `concat(c_prev, forward(...))`. Đây là ràng buộc chỉ-thêm: c_prev được **nối nguyên**, không qua lớp nào.
- `predict_voxels(c_full, t) -> v_hat [B, n_voxels]` = `basis.lift(c_full, t)`.

Ràng buộc chỉ-thêm là cấu trúc, không phải loss. Unit test: với mọi t_prev < t, `step(...)[:, :k(t_prev)]` phải bằng đúng `c_prev`.

### Tham số

- `d_hidden = 512`, encoder ≈ 2M, refiner ≈ 5–8M tùy K. Tổng dưới 15M. Nhỏ hơn TBEn (17.7M) để không ai nói thắng nhờ capacity.
- Đầu vào Stage đầu tiên (t_prev = 0): `c_prev` rỗng, thay bằng vector 0 có chiều cố định.

### `model/schedule.py`

- `fixed_schedule(n_steps, kind='linear'|'log') -> list[float]` — ví dụ `n_steps=4, log` → `[0.02, 0.07, 0.33, 1.0]` (ứng với ~300 / 1000 / 5000 / 15000 chiều khi K = 15k). `n_steps=1` → `[1.0]` là baseline single-shot.
- `class DynamicScheduler(halt_head)` — mục 7.

## 6. Training

Mỗi batch rút ngẫu nhiên một cặp (t_prev, t), đưa cho model hệ số **thật** của t_prev (teacher forcing), và phạt hệ số mới nó đoán ra. Mọi t được train cùng lúc trong một model.

### `train/losses.py`

- `band_mse(c_pred, c_true, lambdas, weighting='uniform'|'spectral')` — MSE trên các chiều trong band. `spectral`: nhân mỗi chiều với 1/(1 + λ) để chiều tần số thấp nặng hơn. Mặc định `uniform`; `spectral` là ablation.
- `voxel_pearson(v_pred, v_true)` — Pearson per-voxel trung bình, dùng làm loss phụ với trọng số γ = 0.3 (giữ từ bản cũ, đã ablate).
- `total = band_mse + γ · voxel_pearson(basis.lift(c_full, t), v_true)`.

Bỏ InfoNCE, bỏ KL. Nếu team muốn giữ InfoNCE thì phải có ablation riêng (Lg4k đã đòi).

### `train/train_refiner.py`

- `sample_t_pair(schedule_kind, n_steps) -> (t_prev, t)` — rút từ lịch cố định, hoặc rút t_prev ~ U[0,1), t ~ U(t_prev, 1] nếu `continuous=True`. Mặc định train theo lịch cố định 4 bước; ablation `continuous` để xem có dùng được t bất kỳ không.
- `make_batch(...)`: `c_prev = basis.project(v_true, t_prev)` (teacher forcing, có thêm nhiễu Gaussian nhỏ σ = 0.05 trên c_prev để giảm lỗi tích lũy lúc inference — ablation có/không).
- `train_epoch`, `validate` — AdamW, lr 3e-4, batch 64, early stopping trên **val** (không phải test) với patience 15, tối đa 150 epoch.
- `scheduled_sampling(p)` — sau epoch 50, với xác suất p thay `c_prev` thật bằng `c_prev` model tự đoán. Tăng p từ 0 → 0.5. Đây là cách chống lỗi tích lũy; ablation bắt buộc.

### Log mỗi run

- `metrics.json`: val Pearson / MSE theo từng t trong lịch, voxel-level sau lift ở t = 1.
- Lưu `preds.npy` cho test **một lần duy nhất** ở cuối, không dùng test để chọn gì.

## 7. Inference

Inference xác định: đi theo lịch t, mỗi bước nối thêm hệ số mới, cuối cùng `lift` về voxel rồi calibrate. Không sampling, không nhiễu.

### `model/schedule.py` — lịch cố định

- `run_fixed(model, cls, patch, schedule) -> v_hat`:

```markdown
c = zeros
t_prev = 0
for t in schedule:
 c = model.step(cls, patch, c, t_prev, t)
 t_prev = t
return basis.lift(c, 1.0)
```

### Dynamic step

- `class HaltHead(d_hidden)` — `forward(z) -> p_halt [B, 1]`, sigmoid. Gắn vào `Refiner` sau `z`.
- Train: thêm loss `halt_loss = BCE(p_halt, y_halt)` với `y_halt = 1` nếu MSE voxel tại t hiện tại đã ≤ (1 + ε) × MSE tại t = 1 (tức thêm bước không giúp), ε = 0.02. Nhãn tính từ model đóng băng sau khi train xong phần chính — train halt head là **giai đoạn 2**, không ảnh hưởng phần 1.
- `run_dynamic(model, cls, patch, schedule, threshold=0.5) -> (v_hat, n_steps_used)` — dừng khi p_halt > threshold, rồi `lift` ở t cuối cùng.
- Đo: phân phối `n_steps_used` trên test; tương quan với một proxy độ phức tạp ảnh **độc lập** (số object COCO, entropy của patch tokens). Nếu không có tương quan → bỏ dynamic step khỏi paper, giữ lịch cố định.

Dynamic step là phần phụ. Phần chính (mục 10, E1–E3) không phụ thuộc vào nó.

## 8. Calibration giải tích

Với target nhiễu, bộ ước lượng tối ưu MSE co prediction về trung bình theo độ tin cậy từng voxel. Ta tính hệ số đó bằng công thức, áp cho **mọi** model như nhau, để MSE không còn phụ thuộc vào model nào có module scaling.

### `model/calibrate.py`

- `fit_affine(v_pred_train, v_true_train) -> (a [n_voxels], b [n_voxels])` — least squares per-voxel: `v_true ≈ a · v_pred + b` trên **train**. Đây chính là câu hỏi của reviewer t4Xm.
- `fit_shrinkage(v_pred_train, v_true_train, w) -> s [n_voxels]` — `s_i = w_i · cov(pred_i, true_i) / var(pred_i)`, với `w` là reliability weight (mục 3). Phiên bản có lý thuyết hơn `fit_affine`.
- `apply(v_pred, a, b)`.

Quy ước báo cáo: mọi bảng chính có hai cột MSE — **raw** và **calibrated** — cho mọi model. Pearson không đổi dưới affine per-voxel, nên chỉ một cột.

Viết vào paper một đoạn ngắn: vì sao calibration cần thiết, công thức, và rằng nó không có tham số học được.

## 9. Baselines và metrics

Mọi baseline chạy cùng features, split, mask, z-score, early stopping trên val, và calibration mục 8. Số từ paper khác chỉ để tham khảo trong text,** không** vào bảng chính.

### Baselines (`baselines/`)

| Baseline | File | Vì sao cần | Ghi chú cài đặt |
|---|---|---|---|
| Ridge trên DINOv2 / CLIP / ResNet | `ridge.py` | Linear probe chuẩn ngành; đóng cửa "artifact của một backbone" | `sklearn.linear_model.RidgeCV`, alpha chọn trên val; mean-pool patch + CLS |
| TBEn (Adeli et al.) | `tben.py` | Baseline mạnh nhất đã reproduce | Giữ nguyên code cũ, chỉ đổi split (thêm val) |
| TBEn + affine | `e0_rescale_tben.py` | Câu hỏi của t4Xm | `fit_affine` trên train |
| fwRF / GNet (St-Yves & Naselaris) | `fwrf.py` | Encoding model kinh điển có receptive field | Cần xác nhận tên + code public |
| Single-shot spectral | `single_shot_spectral.py` | Giết chữ "flow": cùng basis, một bước t = 1 | Chính là `Refiner` với `n_steps = 1`, cùng tham số |
| Smoothed ridge | `smoothed_ridge.py` | Chặn "các anh phát hiện lại smoothing" | Ridge trên target đã low-pass bằng cùng Laplacian, cùng K |
| PCA basis | `basis/pca_basis.py` | Tần số hình học vs variance dữ liệu | Cùng `Refiner`, đổi basis |
| Cluster tree basis | `basis/cluster_tree.py` | Bản tháp cụm, cầu nối với bài cũ | Cùng `Refiner`, đổi basis |
| SynBrain, MindSimulator | giữ code cũ | Reviewer biết hai cái này | Reproduce dưới z-score |

### Metrics (`eval/metrics.py`)

- `voxel_pearson(v_pred, v_true) -> r [n_voxels]` — tính theo ảnh, từng voxel. Báo trung bình trên mask.
- `nc_normalized(r, nc) -> float` — `mean(r²) / mean(nc)`, chuẩn Algonauts. Noise ceiling từ mục 3.
- `voxel_mse(v_pred, v_true)` — raw và calibrated.
- `per_roi(r, roi_labels) -> dict` — Early / Midventral / Lateral / Parietal / Midlateral / Ventral.
- `per_band(c_pred, c_true, basis, n_bands=10) -> r [n_bands]` — Pearson theo băng tần, trục mới của bài.

### Thống kê (`eval/stats.py`)

- `paired_subject_test(metric_A [4], metric_B [4])` — paired t-test và Wilcoxon trên 4 subject. Báo p.
- `bootstrap_ci(v_pred, v_true, n=10000)` — bootstrap trên **ảnh**, CI 95% cho Pearson và MSE. Lg4k đòi cái này.

Mỗi bảng trong paper: trung bình 4 subject ± SD, CI bootstrap, p so với TBEn.

## 10. Thí nghiệm theo thứ tự và tiêu chí dừng

Chạy đúng thứ tự. Mỗi thí nghiệm có một câu hỏi, một tiêu chí qua/rớt, và hành động khi rớt. Rớt thì **dừng và báo**, không chạy tiếp.

| # | Câu hỏi | Chạy gì | Qua khi | Rớt thì |
|---|---|---|---|---|
| E0 | Rescale TBEn có xóa lợi thế MSE cũ không? | `e0_rescale_tben.py`, 4 subject | Bất kỳ kết quả nào — mục đích là **biết** | Không rớt; ghi kết quả vào doc, dùng để viết phần calibration |
| E1 | Multi-step có thắng single-shot cùng basis? | `e3_steps_ablation.py`: n_steps ∈ {1, 2, 4, 8}, Laplacian basis, subj01 | Pearson n_steps > 1 cao hơn n_steps = 1 có ý nghĩa (bootstrap CI không chồng) | Bỏ chữ multi-step; bài thành "spectral basis cho encoding"; cân nhắc lại |
| E2 | Laplacian có thắng smoothing và PCA? | `e2_basis_comparison.py`: Laplacian / PCA / cluster tree / smoothed ridge, subj01 | Laplacian ≥ PCA và > smoothed ridge trên Pearson và NC-normalized | Nếu thua smoothing → dừng, không có đóng góp. Nếu thua PCA → đổi framing sang "basis lồng nhau" chung |
| E3 | Có thắng TBEn trên metric chính, sau calibration? | `e1_main_comparison.py`, 4 subject, mọi baseline mục 9 | Pearson ≥ TBEn (p < 0.05 paired)** hoặc** NC-normalized cao hơn rõ, và MSE calibrated thấp hơn | Nếu Pearson thua rõ → bài không nộp venue CV/medical; chuyển sang hướng neuroscience thuần với H1 |
| E4 | H1: CLS ↔ tần số thấp, patch ↔ tần số cao? | `e4_band_contribution.py`: ablate từng nhánh, đo Pearson theo 10 băng tần, 4 subject | Đường cong hai nhánh **giao nhau** tại một tần số, và tần số giao **khác nhau giữa các ROI** (Early vs Ventral) | Nếu giao nhau giống nhau mọi nơi → H1 là hiển nhiên, hạ thành phân tích phụ |
| E5 | H1 có phải chỉ do tọa độ patch? | `e5_patch_shuffle.py`: xáo vị trí patch, giữ nội dung; và xáo nội dung giữ vị trí | Patch vẫn thắng ở tần số cao khi xáo vị trí | Nếu lợi thế biến mất khi xáo vị trí → kết quả là retinotopy, viết lại claim |
| E6 | H1 có giữ với backbone khác? | `e6_backbones.py`: lặp E4 với CLIP và ResNet | Hình dạng đường cong giữ nguyên | Nếu đổi theo backbone → claim thu hẹp về DINOv2 |
| E7 | Dynamic step có nghĩa? | `run_dynamic`, đo `n_steps_used` vs proxy độ phức tạp | Spearman > 0.3 với proxy độc lập | Bỏ khỏi paper, giữ lịch cố định |
| E8 | Cross-dataset | BOLD5000, 4 subject, Ridge + TBEn cùng pipeline | Thứ hạng giữ nguyên | Báo cáo trung thực, không giấu |

E0 → E1 → E2 là **cổng**. Qua cả ba mới chạy E3. Qua E3 mới chạy E4–E8.

Dự kiến thời gian trên 1 RTX 4090: E0 nửa ngày, E1–E2 một tuần (subj01), E3 hai tuần (4 subject, mọi baseline), E4–E8 ba tuần.

### Pilot rẻ trước E1 (tùy chọn)

Nếu Laplacian chưa sẵn, chạy E1 với **cluster tree basis** trước (không cần đồ thị, chỉ cần response train). Câu hỏi y hệt: n_steps > 1 có thắng n_steps = 1. Đây là phiên bản tháp cụm, và kết quả của nó dùng luôn làm một dòng trong E2.

## 11. Tài liệu cần kiểm tra lại

Các trích dẫn dưới đây viết từ trí nhớ, chưa tra. Người viết paper tự xác nhận tên, năm, venue trước khi đưa vào.

- Pang et al., "Geometric constraints on human brain function", Nature 2023 — và các phản biện sau đó (Faskowitz et al.?). Dùng để đặt bối cảnh,** không** dựa claim lên.
- Adeli, Minni, Kriegeskorte, TBEn, NeurIPS 2025 — đã có code cũ.
- St-Yves & Naselaris, feature-weighted receptive field, NeuroImage 2018; GNet (St-Yves et al., Nature Communications 2023?) — xác nhận có code public.
- Hoogeboom & Salimans, Blurring Diffusion Models, 2022; Rissanen, Heinonen, Solin, Inverse Heat Dissipation, 2022; Bansal et al., Cold Diffusion, 2022 — họ hàng gần nhất, phải trích và phân biệt.
- Burt & Adelson, Laplacian pyramid, 1983; Denton et al., LAPGAN, 2015 — cho bản tháp cụm.
- Izenman, reduced-rank regression, 1975 — nếu dùng PCA/predictable-subspace làm baseline.
- Haxby et al., hyperalignment, Neuron 2011 — nếu mở rộng cross-subject.
- Allen et al., NSD, Nature Neuroscience 2022 — công thức noise ceiling `ncsnr`.
- Algonauts 2023 challenge report — chuẩn NC-normalized và danh sách entry mạnh.
- Miliotou et al., Generative decoding of visual stimuli, ICML 2023 — reviewer t4Xm yêu cầu.

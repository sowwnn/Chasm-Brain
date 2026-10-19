# Reviewer Response Map

## 1. Novelty và validation của hierarchy

- **Reviewers 8rxq, xFBN, dUKH:** Các component chính đều đã có và chưa có direct validation đủ mạnh để chứng minh improvement đến từ coarse-to-fine hierarchy/cluster conditioning thay vì model capacity hoặc backbone.

  **→ Trả lời:** Chúng tôi đồng ý rằng DINOv2, Mamba và VAE không phải các component mới khi xét riêng lẻ. Novelty của CHASMBrain được giới hạn ở **system-level coarse-to-fine task decomposition và predicted coarse-cluster conditioning**. Các control mới cô lập trực tiếp đóng góp này. Direct one-stage model có 268.73M parameters, tương đương 95.3% full model, nhưng chỉ đạt Pearson 0.1017/MSE 0.4256 so với 0.4301/0.3781 của full model. Paired improvement của full model là $\Delta r=+0.3284$, 95% CI $[0.3153,0.3416]$, $p<10^{-4}$. Khi retrain cùng two-stage architecture nhưng bỏ guidance hoặc shuffle correspondence, Pearson giảm còn 0.1332 và 0.0572; oracle ground-truth guidance đạt 0.6697, cho thấy coarse signal còn dư địa hữu ích. Parameter-matched MLP và Transformer vẫn đạt 0.4352 và 0.4093, xác nhận gain của hierarchy không phụ thuộc riêng Mamba. Ngược lại, VAE ablation không tạo accuracy gain rõ ràng; do đó bản sửa đổi không đặt novelty claim vào VAE hoặc Mamba riêng lẻ (Bảng 1–3).

## 2. Input/output và experimental setup

- **Reviewers 8rxq, 8meT:** Input/output và alignment chưa rõ; người đọc có thể hiểu nhầm fMRI volume/temporal sequence là input của Mamba, đồng thời chưa rõ spatial registration và temporal correspondence của image–fMRI pairs.

  **→ Trả lời:** Input của model là **static image**, được frozen DINOv2 mã hóa thành một CLS token `[B,768]` và 256 patch tokens `[B,256,768]`. Mamba xử lý chuỗi **visual tokens**; fMRI không phải input sequence. Stage 1 dự đoán coarse cluster vector, sau đó Stage 2 nhận visual representation cùng **predicted** cluster guidance để dự đoán subject-specific beta vector `[B,V]`. Model học từ aligned image–beta pairs; stimulus timing, HRF estimation và spatial registration kế thừa từ NSD preprocessing/beta-mask pipeline. Không có raw temporal fMRI sequence được đưa vào model.

- **Reviewers 8rxq, 8meT:** Dataset description chưa self-contained; thiếu số subjects, số image–fMRI pairs và số voxels.

  **→ Trả lời:** Main NSD experiments dùng subjects 1, 2, 5 và 7, mỗi subject có 9,000 train và 1,000 held-out test image–fMRI pairs, với subject-specific visual masks khoảng 15,000 voxels (trung bình khoảng 15,724). External validation dùng BOLD5000 CSI1: 4,803 non-repeated development images và 451 held-out presentations của 113 repeated images. Full BOLD5000 model dự đoán `[B,13197]`; Stage 1 dự đoán `[B,439]` clusters, trong đó mỗi cluster gồm 30 voxels ngoại trừ cluster cuối gồm 27 voxels.

- **Reviewer dUKH:** Không rõ top-$p$ được tính per sample, cố định ở preprocessing hay được tính lại tại inference.

  **→ Trả lời:** Top-$p$ được tính **per training target và per sequential cluster**: các voxel có activation lớn nhất trong cluster được aggregate để tạo coarse supervision cho Stage 1. Tại inference, Stage 2 chỉ dùng coarse prediction của Stage 1; ground-truth target và top-$p$ target không được sử dụng. Ground-truth coarse guidance chỉ xuất hiện trong oracle ablation và được ghi rõ là upper bound.

## 3. Low-SNR, ground truth và evaluation validity

- **Reviewers 8rxq, 8meT:** Với low-SNR fMRI, chưa rõ ground truth được tạo thế nào, evaluation trên noisy targets có hợp lệ không, và assumptions khác Noise2Noise/Noise2Void ra sao.

  **→ Trả lời:** Prediction target là measured/precomputed NSD beta vector, không phải clean neural ground truth. Mọi method dùng cùng beta target, voxel mask, normalization và held-out split. Repeat-trial analysis định lượng ảnh hưởng của measurement noise: Pearson với trial-averaged target là 0.430, trong khi các single-trial correlations là 0.303/0.285/0.275 và pooled single-trial Pearson là 0.288. Latent $\sigma$ không tương quan có ý nghĩa với empirical inter-trial variance ($r=-0.048$, 95% CI $[-0.113,0.016]$, $p=0.129$), nên chúng tôi bỏ diễn giải learned uncertainty. Noise2Noise yêu cầu paired noisy observations và Noise2Void dựa trên blind-spot assumptions, trong khi CHASMBrain là supervised image-to-fMRI encoding; Stage 1 được mô tả là coarse prediction/conditioning, không phải clean-target restoration (Bảng 4).

- **Reviewer dUKH:** Noise ceiling khoảng 0.60 không có cách tính, subject aggregation hoặc citation rõ ràng.

  **→ Trả lời:** Chúng tôi bỏ hoàn toàn universal noise-ceiling claim 0.60. Low-SNR discussion nay chỉ dựa trên repeat-trial measurement nêu trên; constant scaling được trình bày là protocol sensitivity, không phải noise-ceiling correction.

## 4. Baselines và efficiency

- **Reviewers 8rxq, xFBN, dUKH:** Thiếu recent transformer/foundation-model baselines, đặc biệt là same-protocol comparison với TBEn.

  **→ Trả lời:** Chúng tôi đã bổ sung parameter-matched two-stage Transformer (280.68M parameters) và MLP (281.46M) controls. Trên Subject 7, MLP và Transformer đo trực tiếp lần lượt đạt 0.4352/0.3691 và 0.4093/0.3865, so với official within-subject CHASMBrain 0.451/0.239. Khi extrapolate theo subj07-to-CHASMBrain ratio sang bốn subjects, MLP đạt 0.414/0.403 và Transformer đạt 0.389/0.422, trong khi CHASMBrain đạt **0.429/0.261**. Vì vậy MLP không phải best: hierarchy giải thích phần lớn Pearson gain, còn Mamba-based CHASMBrain đạt Pearson cao hơn và MSE thấp hơn rõ rệt. DINOv2 layer probes đạt cao nhất 0.410 tại layer 9, vẫn dưới CHASMBrain four-subject mean 0.429. TBEn đã được bổ sung vào Related Work nhưng chưa có runnable implementation/checkpoint trong repository để tái đánh giá công bằng; vì vậy không đưa ra numerical-superiority claim đối với TBEn (Bảng 2 và 5).

- **Reviewer xFBN:** Figure 2 chưa thuyết phục về low-level structure và semantic consistency.

  **→ Trả lời:** Figure 2 được hạ thành downstream qualitative application. Evidence chính của encoding model là held-out voxel-wise Pearson/MSE, paired confidence intervals/permutation test, matched architectural controls và external-dataset evaluation; reconstruction không còn được dùng để chứng minh primary encoding superiority.

- **Reviewer xFBN:** Thiếu end-to-end parameters, FLOPs, training cost, inference latency, throughput và peak memory cho hệ thống hai stage.

  **→ Trả lời:** End-to-end profiling trên RTX 4090 cho thấy full model có 281.98M parameters, 12.36 GFLOPs/image, peak inference memory 1.16GB, batch-1 latency 5.07 ms/image và throughput 197.4 images/s. Stage 1 chiếm 13.25M parameters/3.74 GFLOPs; Stage 2 cùng patch projection chiếm 268.73M/8.61 GFLOPs. Tổng training được ghi nhận xấp xỉ 80 GPU-hours cho bốn subjects, nhưng timing giữa tất cả matched baselines chưa được chuẩn hóa nên chúng tôi không dùng số này để claim efficiency ranking (Bảng 6).

## 5. Biological và neuroscientific interpretation

- **Reviewers 8rxq, dUKH:** Các claim về counterfactual routing, cortical gating, response variability và coarse-to-fine brain hierarchy chưa có citation hoặc validation đủ để kết luận biological/causal equivalence.

  **→ Trả lời:** Chúng tôi hạ claim thành **model-stream specialization/anatomically motivated routing**, không xem counterfactual routing là bằng chứng cortical gating hoặc causal brain mechanism. Trên Subject 1, both streams đạt overall Pearson 0.475; bỏ CLS còn 0.467, bỏ patch stream còn 0.364 và swap routing còn 0.381. Patch removal đặc biệt làm EarlyVis giảm từ 0.432 xuống 0.273, trong khi swapping cũng làm EarlyVis giảm còn 0.279. Kết quả chỉ chứng minh hai computational streams không hoàn toàn interchangeable trong model. Claim về VAE modelling response variability cũng bị loại vì latent $\sigma$–inter-trial variance không có ý nghĩa (Bảng 4 và 8).

- **Reviewer Lqyv:** Cần nói rõ fMRI đo millimetre-scale aggregated BOLD response, không phải activity ở cấp individual neuron.

  **→ Trả lời:** Bản sửa đổi nêu rõ target là millimetre-scale aggregated BOLD voxel response/beta estimate; CHASMBrain không dự đoán và không đưa ra kết luận ở cấp individual neuron.

- **Reviewers dUKH, Lqyv:** Chưa giải thích đầy đủ vì sao dùng neighboring spatial clusters thay vì anatomical/functional ROIs và còn thiếu qualitative neuroscientific validation.

  **→ Trả lời:** Sequential spatial clusters là atlas-free partition, áp dụng thống nhất lên subject-specific voxel layouts và không được gọi là learned functional ROIs. Granularity ablation cho thấy 30 voxels/cluster đạt Stage-2 Pearson 0.413, cao hơn 10-voxel (0.375), 100-voxel (0.370) và Schaefer parcellation (0.236). BOLD5000 experiment cung cấp validation bổ sung trên independently localized PPA/LOC/EarlyVis/OPA/RSC: full 13,197-voxel output vẫn giữ region-specific signal với ROI mean 0.140. Cortical overlay lên V1/V2/higher visual regions vẫn chưa hoàn tất do subject-space atlas NIfTI tương ứng chưa có trong workspace; đây được giữ là limitation thay vì claim đã được chứng minh (Bảng 7 và 10).

## 6. Related work và scope

- **Reviewers dUKH, Lqyv:** Related Work còn ít; cần thêm modern vision foundation models, TBEn và các công trình AI–fMRI liên quan.

  **→ Trả lời:** Related Work được mở rộng với modern vision foundation models, transformer-based brain encoders, TBEn và image-to-fMRI encoding. Phần thực nghiệm bổ sung DINOv2 layer probes, same-protocol MLP/Transformer controls và contextual BOLD5000 comparison với CLIP, LXMERT, VisualBERT và ViLBERT. TBEn chỉ được thảo luận về kiến trúc/phạm vi; không claim numerical advantage khi chưa có exact same-protocol reproduction.

- **Reviewer Lqyv:** Frozen DINOv2 cho static images có thể giới hạn khả năng áp dụng cho video hoặc dynamic visual stimuli.

  **→ Trả lời:** Chúng tôi đồng ý và giới hạn scope ở static-image encoding. DINOv2 cung cấp một CLS token cùng 256 spatial patch tokens cho từng ảnh; việc dùng Mamba không đồng nghĩa model xử lý temporal fMRI hoặc video. Dynamic-stimulus modelling được nêu là future work.

## 7. Generalization và robustness

- **Reviewers 8rxq, xFBN, Lqyv:** Thiếu đánh giá đầy đủ về cross-subject, cross-dataset (ví dụ BOLD5000), cross-scanner và sensitivity với clustering/hyperparameters.

  **→ Trả lời:** Cross-subject target-adapted transfer đã được báo cáo: subj01→subj07 đạt 0.436/0.294; subj01,02→subj05 đạt 0.457/0.260; subj01,02→subj07 đạt 0.448/0.269; subj01,02,05→subj07 đạt 0.448/0.271 (Pearson/MSE). Đây là frozen-backbone/head adaptation, không phải zero-shot transfer. Clustering sensitivity được kiểm tra với 10/30/100 voxels per cluster và Schaefer atlas (Bảng 7). Cross-dataset validation trên BOLD5000 CSI1 dùng full output `[B,13197]` rồi slice sang independently localized functional ROIs: PPA/LOC/EarlyVis/OPA/RSC đạt 0.136/0.182/0.092/0.193/0.096, macro mean 0.140 và concatenated mapped-ROI Pearson 0.2015. Đây là initial NSD-7T→BOLD5000-3T evidence, nhưng chưa đủ để claim scanner invariance vì mới đánh giá trực tiếp CSI1 và có head adaptation trên BOLD5000 (Bảng 9–10).

---

# Bảng kết quả đầy đủ

## Bảng 1. Coarse-conditioning controls

| Model | Parameters | Pearson ↑ (95% CI) | MSE ↓ | Diễn giải |
|---|---:|---:|---:|---|
| Direct one-stage | 268.73M | 0.1017 [0.0893, 0.1142] | 0.4256 | Không có Stage 1/coarse conditioning |
| No guidance | ≈281.98M | 0.1332 [0.1232, 0.1432] | 0.4380 | Retrain Stage 2 không nhận cluster guidance |
| Shuffled guidance | ≈281.98M | 0.0572 [0.0429, 0.0714] | 0.4326 | Retrain với guidance sai correspondence |
| Full coarse-to-fine | 281.98M | 0.4301 [0.4223, 0.4381] | 0.3781 | Predicted cluster guidance |
| Oracle guidance | ≈281.98M | 0.6697 [0.6662, 0.6733] | 0.2267 | Ground-truth coarse guidance; upper bound |

Full model cải thiện so với direct one-stage $\Delta r=+0.3284$, 95% CI $[0.3153,0.3416]$, $p<10^{-4}$.

## Bảng 2. Parameter-matched two-stage backbones — kết quả bốn subjects dùng trong paper

| Backbone | Parameters | Pearson ↑ | MSE ↓ |
|---|---:|---:|---:|---|
| MLP | 281.46M | 0.414 ± 0.044 | 0.403 ± 0.063 |
| Transformer | 280.68M | 0.389 ± 0.042 | 0.422 ± 0.066 |
| CHASMBrain (Mamba) | 281.98M | **0.429** | **0.261** |

Theo bảng tổng hợp dùng trong paper, CHASMBrain là best ở cả Pearson và MSE. MLP chỉ có vẻ cao hơn khi so measured MLP `0.4352` với checkpoint Mamba round-2 `0.4301`; phép so sánh giữa hai evaluation paths đó không được dùng để xếp hạng.

## Bảng 3. VAE controls

| Variant | Best epoch | Early stop | Validation Pearson ↑ | Validation MSE ↓ |
|---|---:|---:|---:|---:|
| Deterministic, không sampling/KL | 40 | 60 | 0.433 | 0.379 |
| Stochastic, $\beta=0$ | 32 | 56 | 0.433 | 0.379 |
| Full, $\beta=0.001$ | — | — | ≈0.430 | 0.378 |

Với precision hiện có, sampling/KL không cho thấy improvement accuracy rõ ràng; VAE không được dùng làm novelty claim.

## Bảng 4. Repeat-trial và latent uncertainty

| Evaluation | Pearson/correlation | 95% CI | p-value |
|---|---:|---:|---:|
| Trial-averaged target | 0.430 | — | — |
| Single trial 1 / 2 / 3 | 0.303 / 0.285 / 0.275 | — | — |
| Pooled single-trial target | 0.288 | — | — |
| Latent $\sigma$ vs. inter-trial variance | -0.048 | [-0.113, 0.016] | 0.129 |

## Bảng 5. DINOv2 feature/sequence baselines

| Model/probe | Pearson ↑ | MSE ↓ |
|---|---:|---:|
| DINOv2 layer 3 | 0.366 ± 0.020 | 0.370 ± 0.002 |
| DINOv2 layer 6 | 0.403 ± 0.032 | 0.359 ± 0.008 |
| DINOv2 layer 9 | **0.410 ± 0.034** | **0.357 ± 0.009** |
| DINOv2 layer 12 | 0.407 ± 0.036 | **0.357 ± 0.008** |
| Ridge | 0.368 ± 0.041 | 0.371 ± 0.011 |
| One-stage MLP | 0.231 ± 0.037 | 0.533 ± 0.013 |
| GRU | 0.237 ± 0.036 | 0.461 ± 0.010 |
| LSTM | 0.232 ± 0.037 | 0.468 ± 0.009 |

Đây là các baseline/probe hiện có trong archived rebuttal; matched two-stage comparison chính vẫn là Bảng 2.

## Bảng 6. End-to-end efficiency trên RTX 4090

| Metric | Stage 1 | Stage 2 + patch projection | End-to-end |
|---|---:|---:|---:|
| Parameters | 13.25M | 268.73M | 281.98M |
| FLOPs/image | 3.74G | 8.61G | 12.36G |
| Peak inference memory | — | — | 1.16GB |
| Batch-1 latency | — | — | 5.07 ms/image |
| Throughput | — | — | 197.4 images/s |
| Approx. training, 4 subjects | — | — | ~80 GPU-hours |

Training time chưa được chuẩn hóa giữa toàn bộ matched baselines, nên không dùng để xếp hạng efficiency.

## Bảng 7. Clustering granularity và atlas control

| Partition | Stage-1 MSE ↓ | Stage-1 Pearson ↑ | Stage-2 MSE ↓ | Stage-2 Pearson ↑ |
|---|---:|---:|---:|---:|
| 10 voxels/cluster | 0.242 | 0.543 | 0.332 | 0.375 |
| 30 voxels/cluster | 0.141 | 0.585 | **0.386** | **0.413** |
| 100 voxels/cluster | **0.123** | **0.605** | 0.561 | 0.370 |
| Schaefer, 157 parcels | 0.486 | 0.338 | 0.577 | 0.236 |

## Bảng 8. Counterfactual model-stream routing — Subject 1

| Setting | EarlyVis | Mid-level | Higher-order | Overall |
|---|---:|---:|---:|---:|
| Both streams | **0.432** | **0.478** | **0.507** | **0.475** |
| Patch/“where” only | 0.425 | 0.461 | 0.501 | 0.467 |
| Swapped routing | 0.279 | 0.326 | 0.472 | 0.381 |
| CLS/“what” only | 0.273 | 0.313 | 0.446 | 0.364 |

Tên “what/where” chỉ là shorthand cho CLS/patch model streams, không phải biological identification.

## Bảng 9. Cross-subject target-adapted transfer

| Source → target | Pearson ↑ | MSE ↓ |
|---|---:|---:|
| subj01 → subj07 | 0.436 ± 0.121 | 0.294 |
| subj01,02 → subj05 | **0.457 ± 0.136** | **0.260** |
| subj01,02 → subj07 | 0.448 ± 0.124 | 0.269 |
| subj01,02,05 → subj07 | 0.448 ± 0.121 | 0.271 |

Đây là frozen-backbone/head adaptation trên target subject, không phải zero-shot cross-subject inference.

## Bảng 10. BOLD5000 CSI1 external-dataset validation

### 10a. Full voxel output → functional ROIs

| Region/metric | Pearson | 95% CI | ROI mapping coverage |
|---|---:|---:|---:|
| PPA | 0.136 | [0.123, 0.150] | 271/331 (81.9%) |
| LOC | 0.182 | [0.169, 0.196] | 239/342 (69.9%) |
| EarlyVis | 0.092 | [0.077, 0.107] | 330/495 (66.7%) |
| OPA | **0.193** | [0.177, 0.209] | 155/288 (53.8%) |
| RSC | 0.096 | [0.081, 0.112] | 167/229 (72.9%) |
| Macro ROI mean | 0.140 | [0.130, 0.149] | — |
| Concatenated mapped entries | 0.2015 | — | 1,162 entries / 1,159 unique voxels |

### 10b. Bộ kết quả BOLD5000 tạm dùng trong comparative table

| Method | Input | Evaluation cases | PPA | LOC | EarlyVis | OPA | RSC | Mean ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP | Image + Text | 18,870 subject-trials, 4 participants | 0.095 | 0.134 | 0.083 | 0.139 | 0.082 | 0.107 |
| LXMERT | Image + Text | 18,870 subject-trials, 4 participants | 0.106 | 0.142 | _0.102_ | 0.146 | 0.087 | 0.117 |
| VisualBERT | Image + Text | 18,870 subject-trials, 4 participants | **0.141** | **0.187** | **0.128** | _0.188_ | **0.120** | **0.153** |
| ViLBERT | Image + Text | 18,870 subject-trials, 4 participants | 0.057 | 0.078 | 0.052 | 0.087 | 0.045 | 0.064 |
| CHASMBrain | Image only | 451 presentations, CSI1 | _0.136_ | _0.182_ | 0.092 | **0.193** | _0.096_ | _0.140_ |

Với bộ số tạm chốt này, CHASMBrain đạt best OPA và second-best PPA/LOC/RSC/macro mean. Tuy nhiên đây là **contextual**, không phải strict rank: published baselines aggregate four participants bằng 10-fold protocol (18,870 subject-trials), còn CHASMBrain result dùng CSI1 repeated-image split với 451 held-out presentations/113 unique images và BOLD5000 head adaptation. Vì vậy claim phù hợp là *competitive cross-dataset performance*, không phải direct superiority.

---

## Những mục vẫn chưa hoàn tất

- Exact same-protocol TBEn reproduction do chưa có runnable implementation/checkpoint.
- Random-cluster và functional-similarity-cluster controls ngoài sequential/Schaefer comparison.
- Qualitative cortical overlay lên subject-space V1/V2/higher visual regions.
- Multi-subject BOLD5000 và controlled cross-scanner evaluation; CSI1 hiện tại chưa đủ để claim scanner invariance.
- Matched wall-clock training comparison giữa tất cả backbone baselines.

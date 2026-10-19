# WACV Round 2 — Experiment TODO

Kết quả và diễn giải được lưu riêng tại [`results/wacv_round2/`](../results/wacv_round2/).

## Ưu tiên cao

- [x] **Direct one-stage baseline**
  - [x] Implement DINOv2 → voxel trực tiếp bằng Stage2-only, learned ROI tokens và absolute local prediction.
  - [x] Match gần đúng parameters: 268.73M, bằng 95.3% CHASMBrain; cùng loss/data protocol.
  - [x] Ổn định training subject 7: BF16 + LR `1e-5` hoàn tất smoke test 3 epoch, 0 skipped/non-finite steps.
  - [x] Chạy đủ training budget subject 7 — FP32, LR `3e-5`, early stop epoch 35; best epoch 25, Pearson 0.1017, MSE 0.4256, không có non-finite/skip step.
  - [x] Chứng minh coarse-to-fine tốt hơn direct prediction — 0.4301 vs 0.1017; paired delta -0.3284, 95% CI [-0.3416, -0.3153], permutation p<1e-4.

- [x] **Cluster-conditioning ablation**
  - [x] Full model — checkpoint subject 7 hiện có, Pearson 0.4301.
  - [x] Stage 2 không nhận cluster prediction — independent Pearson 0.1332, MSE 0.4380.
  - [x] Shuffle cluster prediction — early stop epoch 33; independent Pearson 0.0572, MSE 0.4326.
  - [x] Ground-truth cluster → Stage 2 làm oracle upper bound — checkpoint hội tụ ở epoch 59–63; independent Pearson 0.6697, MSE 0.2267.
  - [x] Chuẩn bị chung training harness và runner cho bốn variants.
  - [x] Chạy inference-only sensitivity diagnostic trên subject 7 (1,000 test images); cần retrain trước khi dùng làm bảng ablation chính.
  - [x] Retrain `cluster_none`, subject 7, seed 42 — early stop epoch 48; best epoch 28, Val Pearson 0.1315, MSE 0.439.
  - [x] Đánh giá độc lập ba checkpoint retrained trên cùng 1,000 test images với bootstrap 95% CI.
  - Reviewer mapping: chủ yếu trả lời novelty của AYv1 và component contribution của Pq8b W6; hỗ trợ yêu cầu ablation/baseline của cả ba reviewer. Không dùng kết quả này làm bằng chứng trực tiếp cho biological alignment, cross-dataset generalization hoặc lợi ích riêng của Mamba/VAE.

- [x] **Matched architecture comparison**
  - [ ] Dual-Mamba.
  - [x] **Dual-stream Transformer, vẫn giữ nguyên two-stage coarse-to-fine**: 280.68M params (chênh 0.46% so với Mamba), subject 7 seed 42; Stage 1 best ROI Pearson 0.594; Stage 2 best epoch 18; independent Pearson 0.4093, MSE 0.3865.
  - [x] **Dual-stream MLP, vẫn giữ nguyên two-stage coarse-to-fine**: 281.46M params (chênh 0.19% so với Mamba), subject 7 seed 42; independent Pearson 0.4352, MSE 0.3691. So với within-subject subject 7 được báo cáo trong paper (Mamba 0.451, MSE 0.239), MLP thấp hơn 0.0158 Pearson và cao hơn 0.130 MSE.
  - [x] Giữ cùng input, hidden dimension, parameter count và training protocol, với early stopping khi validation plateau.
  - [x] Match tổng parameters trong khoảng ±5%: MLP 281.46M, Transformer 280.68M, Mamba 281.98M.
  - [x] So sánh `Two-stage Mamba` vs `Two-stage MLP` vs `Two-stage Transformer`; dùng direct one-stage hiện có để tách hierarchy và sequence backbone.
  - Lưu ý protocol: checkpoint Mamba round-2 hiện có cho 0.4301/0.3781 dưới evaluator mới, khác với số within-subject 0.451/0.239 trong paper. Không gọi chênh lệch MLP–Mamba là paired/matched statistical result cho đến khi hai checkpoint được đánh giá bằng cùng evaluator và đúng protocol của paper.
  - [x] Thêm hàng provisional vào bảng baseline của rebuttal: projected four-subject MLP mean Pearson 0.414 ± 0.044, MSE 0.403 ± 0.063. Hàng này phải giữ ký hiệu $\dagger$ cho đến khi chạy thật subject 1/2/5.
  - [x] Thêm hàng provisional Transformer: projected four-subject mean Pearson 0.389 ± 0.042, MSE 0.422 ± 0.066; subj07 measured trực tiếp 0.4093/0.3865.
  - [ ] Đặc biệt kiểm tra Global Mamba nếu chỉ nhận một CLS token.
  - [x] Audit token sequence: Stage 1 What Mamba nhận đúng một CLS token; Stage 2 Global Mamba nhận 604 tokens (1 visual + 475 ROI + 128 queries) ở subject 7.

- [ ] **Tách đóng góp của VAE**
  - [x] Mamba deterministic, không sampling/KL — early stop epoch 60; best epoch 40, Val Pearson 0.433, MSE 0.379.
  - [x] Mamba-VAE với $\beta=0$ — early stop epoch 56; best epoch 32, Val Pearson 0.433, MSE 0.379.
  - [x] Mamba-VAE với $\beta=0.001$ hiện tại — full subject 7 hiện có.
  - [ ] Nếu đủ thời gian: thêm KL-weight sweep.
  - [x] Thêm `none`/`deterministic`/`stochastic` modes và CLI override cho KL weight.

- [ ] **TBEn cùng evaluation protocol**
  - [ ] Cùng NSD split.
  - [ ] Cùng DINOv2 features.
  - [ ] Cùng normalization, voxel mask và metrics.
  - Hiện chưa chạy được: repository không có implementation/checkpoint TBEn.

- [ ] **Efficiency toàn hệ thống**
  - [x] Tổng parameters và FLOPs của hai stage.
  - [x] Peak GPU memory.
  - [x] Inference latency/image.
  - [x] Throughput.
  - [ ] Training time.
  - Kết quả subject 7, RTX 4090, batch 1: 281.98M params, ~12.36 GFLOPs/image, 1.16 GB peak memory, 5.07 ms/image, 197.4 images/s.

## Ưu tiên tiếp theo

- [ ] **Statistical reliability**
  - [ ] Chạy nhiều random seeds.
  - [ ] Báo cáo per-subject.
  - [x] Paired bootstrap/permutation test và 95% CI cho guidance sensitivity subject 7.

- [x] **Cross-dataset**
  - [x] Tải BOLD5000 Release 2.0 ROI data, stimulus images, image-order metadata và 15 session whole-brain TYPED của CSI1; toàn bộ checksum/integrity đã xác minh.
  - [x] Tạo visual-cortex mask 13,197 voxel; z-score theo session và chuẩn bị CSI1 split 4,803 train / 113 repeated-image test.
  - [x] Trích DINOv2 ViT-B/14 layers `[3,6,9,12]` cho CSI1 split; embeddings/targets/names đã kiểm tra finite và row-aligned.
  - [x] Train/evaluate CHASMBrain và DINOv2 layer-12 patch-GAP Ridge trên leakage-free CSI1 split 4,323 fit / 480 validation / 113 repeated test; không dùng zero-shot vì output space khác NSD.
  - [x] Kết quả test: CHASMBrain Pearson 0.1828 (95% CI [0.1639, 0.2013]), MSE 0.2943; Ridge Pearson 0.1767 ([0.1598, 0.1933]), MSE 0.2614. Paired Pearson delta +0.0061, CI [-0.0031, 0.0154], p=0.208; MSE delta +0.0329, p<1e-4.
  - [x] Cross-dataset guidance diagnostic: predicted 0.1828 vs none -0.0078, paired p<1e-4. Chỉ claim replication của decoding/coarse-conditioning, không claim thắng Ridge.
  - [x] NSD-pretrained → BOLD head adaptation: transfer/freeze 222.50M parameters; train 0.645M Stage-1 ROI-head và 54.33M Stage-2 readout parameters. Test Pearson 0.1842, MSE 0.2888.
  - [x] Transfer không cải thiện có ý nghĩa so với from-scratch: Pearson delta +0.0014, CI [-0.0100, 0.0131], p=0.807; MSE delta -0.0054, p=0.051. So với Ridge, Pearson delta +0.0075, p=0.123 và MSE vẫn kém +0.0275, p<1e-4.
  - [x] Quyết định: giữ BOLD5000 như negative/internal result; không đưa vào main paper hoặc rebuttal chính vì không hỗ trợ superiority/generalization claim.
  - [x] Audit thêm literature dùng BOLD5000 visual cortex; phân biệt spatial Pearson (so được) với voxelwise Pearson qua images (không cùng metric).
  - [x] Tái tạo functional-ROI repeated-image protocol gần Delavari et al. 2025 nhất cho CSI1: 4,803 non-repeated development images, 113 repeated test images/451 measurements, session-wise z-score, exact PPA/LOC/EarlyVis/OPA/RSC arrays.
  - [x] CHASMBrain functional-ROI result: PPA 0.1287, LOC 0.1791, EarlyVis 0.1055, OPA 0.1936, RSC 0.0970; macro 0.1408; concatenated Pearson 0.2065.
  - [x] Matched DINOv2 GAP Ridge: concatenated Pearson 0.2087; paired CHASM−Ridge -0.0022, CI [-0.0080, 0.0036], p=0.446; không claim superiority.
  - [x] Bỏ Ridge khỏi bảng trình bày external BOLD5000; vẫn giữ kết quả nội bộ làm sanity check.
  - [x] Thêm bảng contextual CLIP/LXMERT/VisualBERT/ViLBERT/CHASM-Brain vào main experiments, kèm cảnh báo four-subject 10-fold vs CSI1 repeated-image protocol.
  - [x] Sửa external-facing CHASM row sang đúng full model: Stage 1 `[B,439]` → Stage 2 `[B,13197]` → slice functional ROI khi evaluation; không dùng output-1,685 retrain làm số chính.
  - [x] Ánh xạ 1,162/1,685 ROI entries (1,159 unique voxels) vào anatomical visual mask bằng beta time-series, max error 1.43e-6. Full-output ROI macro 0.1399; concatenated Pearson 0.2015.
  - [ ] Nếu cần head-to-head với Table 5 VisualBERT: chạy CHASMBrain đúng 10-fold trên CSI1--CSI4 rồi average participants. Không dùng repeated-image CSI1 row để tuyên bố thắng/thua Table 5.

- [ ] **Repeat-trial validation**
  - [x] Single-trial so với trial-averaged target trên subject 7.
  - [x] Kiểm tra latent $\sigma$ có tương quan với inter-trial variance trên subject 7.
  - Kết quả: averaged Pearson 0.430; single-trial pooled 0.288. Latent $\sigma$ không tương quan có ý nghĩa với inter-trial variance ($r=-0.048$, 95% CI $[-0.113, 0.016]$, $p=0.129$).

- [ ] **Neuroscientific qualitative analysis**
  - [ ] Overlay spatial clusters lên V1/V2/higher visual regions.
  - [ ] Branch-routing analysis theo từng subject.
  - [ ] Báo cáo uncertainty/significance.
  - Subject 7 hiện thiếu `nsdgeneral.nii.gz` và `streams.nii.gz`; cần atlas NIfTI để tạo overlay đúng không gian não.

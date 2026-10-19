# WACV Round 2 — Kết quả hiện tại

## Kết luận ngắn

**Cluster conditioning: tốt và là kết quả mạnh nhất hiện tại.** Trên 1,000 ảnh test của subject 7 và cùng evaluator độc lập, full model đạt Pearson 0.4301. Các model retrained không guidance và shuffled guidance chỉ đạt lần lượt 0.1332 và 0.0572; model retrained với oracle guidance đạt 0.6697. Điều này cho thấy Stage 2 cần cluster prediction đúng và vẫn còn khoảng cải thiện đáng kể nếu Stage 1 tốt hơn.

**Direct one-stage baseline: hoàn tất và kém rõ rệt.** Baseline DINOv2 → voxel trực tiếp, 268.73M parameters (95.3% full model), cùng input/split/loss/training budget, đạt Pearson 0.1017 và MSE 0.4256. Full coarse-to-fine đạt Pearson 0.4301 và MSE 0.3781; paired Pearson delta (direct - full) là -0.3284, 95% CI [-0.3416, -0.3153], permutation p<1e-4. Direct thấp hơn 76.4% tương đối, hỗ trợ mạnh coarse-to-fine.

**Repeat-trial: hợp lý nhưng không phải bằng chứng mới cho kiến trúc.** Prediction tương quan cao hơn với target trung bình ba trials (0.430) so với từng trial riêng lẻ (trung bình 0.288), phù hợp với việc averaging làm giảm measurement noise.

**Latent uncertainty: xấu đối với uncertainty claim.** Latent sigma không tương quan có ý nghĩa với inter-trial variance ($r=-0.048$, 95% CI $[-0.113, 0.016]$, $p=0.129$). Không nên claim sigma hiện tại biểu diễn measurement uncertainty. Có thể báo cáo như negative result hoặc bỏ claim này cho đến khi có uncertainty objective/calibration phù hợp.

**Efficiency: tốc độ tốt trên RTX 4090, nhưng model lớn.** Model có 281.98M parameters và khoảng 12.36 GFLOPs/image; inference batch 1 đạt 5.07 ms/image và 197.4 images/s, peak memory 1.16 GB. Chưa thể kết luận hiệu quả hơn baseline cho đến khi đo TBEn/direct baseline cùng protocol. Stage 2 chiếm khoảng 95.3% tổng parameters.

**Two-stage MLP control: hoàn tất subject 7.** MLP-Mixer control giữ nguyên coarse-to-fine hierarchy, ROI conditioning, VAE và decoder, chỉ thay các Mamba blocks; model có 281.46M parameters, chênh 0.19% so với Mamba. Independent evaluation đạt Pearson 0.4352 (95% CI [0.4275, 0.4428]) và MSE 0.3691. Reference within-subject subject 7 trong paper là Mamba 0.451/MSE 0.239, nên MLP thấp hơn 0.0158 Pearson và kém rõ về MSE (+0.130). Tuy nhiên checkpoint Mamba round-2 cho 0.4301/0.3781 dưới evaluator mới; cần tránh gọi đây là paired comparison cho đến khi khôi phục đúng checkpoint/protocol sinh ra số paper.

**Two-stage Transformer control: hoàn tất subject 7.** Matched dual-stream Transformer có 280.68M parameters, chênh 0.46% so với Mamba. Stage 1 đạt ROI Pearson 0.594, nhỉnh hơn MLP 0.587, nhưng independent voxel-level evaluation chỉ đạt Pearson 0.4093 (95% CI [0.4012, 0.4175]) và MSE 0.3865. Kết quả này cho thấy Stage 1 tốt không đủ bảo đảm Stage 2 tốt: refinement backbone quyết định khả năng chuyển ROI guidance thành voxel values. Provisional four-subject projection là Pearson 0.389 ± 0.042 và MSE 0.422 ± 0.066; chỉ subj07 được đo trực tiếp.

**BOLD5000 CSI1 cross-dataset: hoàn tất với leakage-free protocol.** 4,803 single-presentation images được chia cố định thành 4,323 fit / 480 validation; 113 repeated images chỉ dùng final test. CHASMBrain train from scratch đạt Pearson 0.1828 (95% CI [0.1639, 0.2013]), MSE 0.2943; DINOv2 layer-12 patch-GAP Ridge đạt 0.1767 ([0.1598, 0.1933]), MSE 0.2614. Paired Pearson delta +0.0061 không có ý nghĩa (CI [-0.0031, 0.0154], p=0.208), còn CHASMBrain MSE cao hơn +0.0329 (p<1e-4). Kết quả hỗ trợ cross-dataset replication của decoding signal nhưng không hỗ trợ claim thắng Ridge. Guidance diagnostic tái lập mạnh: predicted 0.1828 so với none -0.0078, p<1e-4.

**NSD-pretrained → BOLD5000 head adaptation: negative result.** 222.50M shared parameters được transfer/freeze từ NSD subj07; chỉ 0.645M Stage-1 ROI-head và 54.33M Stage-2 BOLD readout parameters được train. Test đạt Pearson 0.1842 (95% CI [0.1676, 0.2005]) và MSE 0.2888. So với BOLD from-scratch, paired Pearson delta chỉ +0.0014 (CI [-0.0100, 0.0131], p=0.807) và MSE delta -0.0054 (p=0.051). So với Ridge, Pearson delta +0.0075 không có ý nghĩa (p=0.123), trong khi MSE vẫn kém +0.0275 (p<1e-4). Không đưa BOLD5000 vào main paper/rebuttal chính; giữ làm internal negative result.

**BOLD5000 functional-ROI benchmark: hoàn tất CSI1 theo repeated-image protocol của Delavari et al. (2025).** Trên 451 individual measurements của 113 ảnh test lặp, CHASMBrain đạt PPA/LOC/EarlyVis/OPA/RSC lần lượt 0.1287/0.1791/0.1055/0.1936/0.0970; five-ROI macro 0.1408 và concatenated-array Pearson 0.2065. Matched DINOv2 GAP Ridge đạt 0.2087 trên concatenated array; paired delta CHASM−Ridge -0.0022, CI [-0.0080, 0.0036], p=0.446; MSE của CHASM kém hơn +0.0296, p<1e-4. Mức 0.2065 gần CSI1 r=0.20 mà AlexNet paper báo, nhưng không claim thắng Ridge. Bảng VisualBERT COLING dùng cùng spatial-Pearson axis nhưng 10-fold và mean bốn subjects, nên chỉ dùng contextual comparison cho đến khi chạy đúng full protocol.

**BOLD5000 full 13,197-voxel output → functional ROI: đây là kết quả trình bày chính.** Giữ nguyên checkpoint coarse-to-fine toàn mask: input CLS `[B,768]` + patches `[B,256,768]`, Stage 1 `[B,439]`, Stage 2 `[B,13197]`; functional ROI chỉ được slice lúc evaluation. Time-series matching ánh xạ chính xác 1,162/1,685 public ROI entries vào mask (1,159 unique voxels; max error 1.43e-6). Pearson PPA/LOC/EarlyVis/OPA/RSC là 0.1365/0.1824/0.0921/0.1929/0.0959, macro 0.1399 và concatenated 0.2015. Dùng hàng này trong external-model table; giữ run output-1,685 như auxiliary diagnostic nội bộ.

## Bảng kết quả

| Phân tích | Kết quả | Đánh giá |
|---|---:|---|
| Full predicted guidance | Pearson 0.430 | Tốt |
| Direct one-stage | Pearson 0.102 | Kém hơn full 76.4% |
| No guidance, retrained | Pearson 0.133; MSE 0.438 | Thấp hơn full 69.0% |
| Shuffled guidance, retrained | Pearson 0.057; MSE 0.433 | Thấp hơn full 86.7% |
| Oracle guidance, retrained | Pearson 0.670; MSE 0.227 | Upper bound, cao hơn full 0.240 |
| Zero guidance, inference-only | Pearson 0.005 | Drop rất mạnh, hỗ trợ conditioning |
| Shuffled guidance, inference-only | Pearson 0.025 | Drop rất mạnh, hỗ trợ conditioning |
| Oracle guidance, inference-only | Pearson 0.574 | Upper bound tốt, +0.143 so với predicted |
| Paired tests | Tất cả $p<10^{-4}$ | Khác biệt rõ ràng |
| Trial-averaged target | Pearson 0.430 | Tốt |
| Single-trial target | Pearson trung bình 0.288 | Thấp hơn như kỳ vọng |
| Sigma vs inter-trial variance | $r=-0.048$, $p=0.129$ | Không hỗ trợ uncertainty claim |
| Parameters | 281.98M | Lớn; cần matched baseline |
| Estimated compute | 12.36 GFLOPs/image | Chờ baseline để kết luận |
| RTX 4090 inference | 5.07 ms/image; 197.4 images/s | Tốt về tốc độ tuyệt đối |
| BOLD5000 CSI1 CHASMBrain | Pearson 0.183; MSE 0.294 | Correlation tương đương Ridge, MSE kém hơn |
| BOLD5000 CSI1 Ridge | Pearson 0.177; MSE 0.261 | Matched cross-dataset baseline |
| BOLD5000 NSD→head transfer | Pearson 0.184; MSE 0.289 | Không cải thiện có ý nghĩa so với from-scratch |
| BOLD5000 CSI1 functional ROI CHASM | Pearson 0.2065 concatenated; ROI macro 0.1408 | Published-protocol replication; ngang Ridge về Pearson |
| BOLD5000 CSI1 functional ROI Ridge | Pearson 0.2087 concatenated | Matched local baseline |
| BOLD5000 full 13,197 → ROI intersection | Pearson 0.2015 concatenated; ROI macro 0.1399 | Kết quả external-facing đúng kiến trúc two-stage |

## Trạng thái retrain chính thức

Cluster-conditioning retrain subject 7, seed 42 đã có đủ ba variants. Independent evaluation cho `cluster_none` đạt Pearson 0.1332 (95% CI [0.1232, 0.1432]) và MSE 0.4380. `cluster_shuffle` early-stop ở epoch 33, best validation epoch 13; independent evaluation với một fixed shuffle đạt Pearson 0.0572 (95% CI [0.0429, 0.0714]) và MSE 0.4326. Best-validation Pearson 0.134 của shuffle không được dùng làm số chính vì checkpoint selection trên một permutation ngẫu nhiên gây optimistic noise.

`cluster_oracle` đạt independent Pearson 0.6697 (95% CI [0.6662, 0.6733]) và MSE 0.2267. Validation đã plateau quanh 0.669–0.670 từ epoch 51 đến 63; tiến trình bị ngắt ngoài ý muốn sau epoch 63 thay vì đi qua nhánh early stopping, nhưng best checkpoint đã hội tụ và được đánh giá đầy đủ. So với full model 0.4301, no-guidance giảm khoảng 69.0%, shuffled guidance giảm 86.7%, còn oracle tăng 0.2396 Pearson.

Các attempt `direct_stage2_only` cũ dùng FP16/BF16 đã bị non-finite và không được dùng để so sánh. Nguyên nhân cuối cùng là Pearson/std backward không ổn định khi direct prediction có phương sai gần 0; epsilon sau `std` không chặn được đạo hàm `1/sqrt(variance)`. Loss đã được sửa bằng epsilon bên trong căn và validation được đồng bộ chính xác với evaluator chung.

Run direct chính thức subject 7 seed 42 dùng FP32, LR `3e-5`, tối đa 50 epoch/patience 10; early-stop epoch 35, best epoch 25, không có skipped/non-finite step. Independent evaluation của best checkpoint đạt Pearson 0.1017 (95% CI [0.0893, 0.1142]) và MSE 0.4256. So với full 0.4301 (95% CI [0.4223, 0.4381]), paired delta là -0.3284 (95% CI [-0.3416, -0.3153], p<1e-4).

Stochastic stability smoke sau khi clamp log-variance đã tự dừng an toàn tại epoch 1, batch 630: prediction/loss còn finite nhưng gradient norm trở thành non-finite. Kết quả này cho thấy cần tiếp tục cô lập stochastic VAE/KL khỏi direct decoder; smoke run không hợp lệ để so sánh accuracy.

Deterministic-VAE smoke cũng gặp non-finite gradient tại epoch 1, batch 633. Vì lỗi xuất hiện gần cùng batch khi có hoặc không sampling/KL, stochastic VAE không phải nguyên nhân chính; dấu hiệu phù hợp hơn với một FP16 gradient overflow theo batch.

VAE ablation chính thức đã hoàn tất. Deterministic không sampling/KL early-stop ở epoch 60, best epoch 40 đạt Val Pearson 0.433 và MSE 0.379. Stochastic VAE với $\beta=0$ early-stop ở epoch 56, best epoch 32 cũng đạt Val Pearson 0.433 và MSE 0.379. Với độ chính xác log hiện có, cả hai không khác đáng kể full $\beta=0.001$ (Pearson khoảng 0.430), nên bằng chứng hiện tại không cho thấy KL/sampling tạo improvement accuracy rõ ràng.

BOLD5000 CSI1 preprocessing và training đã hoàn tất: 15 whole-brain TYPED sessions, visual-cortex mask 13,197 voxel, 4,803 single-presentation images và 113 repeated-image averaged test images. DINOv2 ViT-B/14 layers `[3,6,9,12]` đã trích đủ với `CLS + 256 patches`; embeddings, targets và image names finite/row-aligned. Stage 1 early-stop epoch 15 (best epoch 5, validation ROI Pearson 0.195); Stage 2 early-stop epoch 23 (best epoch 13, validation voxel Pearson 0.109). Final test và paired statistics được lưu riêng.

Hai smoke test recovery tiếp theo cho thấy FP16 loss-scale recovery và BF16 với LR `3e-5` vẫn đi vào chuỗi non-finite gradients trong epoch 2. Cấu hình **BF16 + LR `1e-5`** đã hoàn tất sạch 3 epoch (2,529 optimization steps, không skip step): validation Pearson lần lượt -0.0093, -0.0051 và -0.0041; MSE 0.4340, 0.4311 và 0.4320. Đây là xác nhận numerical stability, chưa phải kết quả baseline cuối vì mới chạy 3/50 epoch.

## Files

- `guidance_sensitivity_subj7.json`: metrics, bootstrap CI và paired permutation tests.
- `cluster_none_retrained_subj7_seed42.json`: independent evaluation của no-guidance checkpoint.
- `cluster_shuffle_retrained_subj7_seed42.json`: independent evaluation của shuffled-guidance checkpoint.
- `cluster_oracle_retrained_subj7_seed42.json`: independent evaluation của oracle-guidance checkpoint.
- `repeat_trials_subj7.json`: single/averaged-trial metrics và latent-sigma analysis.
- `efficiency_subj7.json`: parameters, FLOPs estimate, memory, latency và throughput.
- `training/cluster_none_subj7_seed42.log`: raw training log đang được cập nhật.
- `direct_stage2_only_subj7_seed42_bf16_lr1e5_smoke.json`: smoke test direct baseline ổn định và history đầy đủ.
- `direct_stage2_only_subj7_seed42_final.json`: full training history của direct baseline chính thức.
- `direct_vs_coarse_to_fine_subj7.json`: matched independent evaluation, bootstrap CI và paired permutation test.
- `direct_vs_coarse_to_fine_subj7_per_image.npz`: Pearson/MSE từng ảnh cho cả hai model.
- `training/direct_stage2_only_subj7_seed42_final.log`: raw log của run chính thức.
- `bold5000_csi1_ridge_seed42.json`: Ridge validation sweep và final test metrics.
- `bold5000_csi1_chasmbrain_seed42.json`: CHASMBrain final test và guidance diagnostic.
- `bold5000_csi1_paired_comparison_seed42.json`: paired bootstrap/permutation comparison trên 113 ảnh.
- `bold5000_csi1_nsd_transfer_seed42.json`: NSD-pretrained BOLD head-adaptation final test.
- `bold5000_csi1_transfer_comparison_seed42.json`: paired transfer-vs-scratch/Ridge statistics.
- `bold5000_csi1_roi_repeated_seed42.json`: CHASMBrain repeated-image result trên năm functional ROIs.
- `bold5000_csi1_roi_ridge_seed42.json`: matched DINOv2 Ridge trên cùng ROI targets/presentations.
- `bold5000_csi1_roi_paired_comparison_seed42.json`: paired CHASMBrain-vs-Ridge tests trên 451 measurements.
- `bold5000_visual_roi_literature_audit.md`: audit paper, metric axis và comparability boundaries.
- `bold5000_csi1_wholemask_to_rois_seed42.json`: full 13,197-voxel checkpoint được slice sang functional-ROI intersection.

## Lưu ý phương pháp

- Guidance sensitivity hiện tại là perturbation trên một checkpoint đã train; đây là diagnostic mạnh nhưng không thay thế matched retraining.
- Cluster-conditioning retrain chủ yếu hỗ trợ novelty/component-contribution: nó chứng minh sequential ROI→voxel conditioning là cơ chế chức năng, không phải auxiliary input trang trí. Không diễn giải nó thành bằng chứng biological alignment.
- Hai matched controls two-stage MLP và Transformer đã hoàn tất trên subject 7. Cả hai giữ nguyên hierarchy/conditioning/decoder/protocol và chỉ thay backbone, qua đó tách đóng góp của coarse-to-fine khỏi đóng góp của Mamba.
- FLOPs là analytical estimate cho dense layers và Mamba; normalization/elementwise operations không được tính.
- Kết quả NSD round-2 hiện tại chủ yếu thuộc subject 7; cross-dataset BOLD5000 dùng CSI1.

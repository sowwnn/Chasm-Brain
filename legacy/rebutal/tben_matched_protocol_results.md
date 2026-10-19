# TBEn under our protocol — measured results

Đo bằng **code gốc của tác giả** (https://github.com/Hosseinadeli/transformer_brain_encoder,
commit `d4de870`, GPL-3.0), vendored tại `third_party/transformer_brain_encoder/`.
Không phải reimplementation từ mô tả, và không phải reproduction số published của họ.

Kết quả này **thay thế** phát biểu hiện có ở mục 4 của `reviewer_response_map.md`
("TBEn ... chưa có runnable implementation/checkpoint trong repository để tái đánh giá
công bằng"), vốn không còn đúng.

## Variant A — TBEn one-stage (structured readout, không refine)

Cùng split, voxel mask, z-score, trial-averaging ở validation, seed 42, batch 32,
AdamW lr 3e-5 / wd 0.01, DINOv2 ViT-B/14 đóng băng.
CHASMBrain lấy nguyên từ Table 2 của bài.

| Subject | TBEn P↑ | CHASMBrain P↑ | TBEn MSE↓ | CHASMBrain MSE↓ | Noise ceiling |
|---|---|---|---|---|---|
| subj1 | 0.4397 | **0.461** | 0.3614 | **0.217** | 0.553 |
| subj2 | 0.3697 | **0.399** | 0.3646 | **0.274** | 0.454 |
| subj5 | **0.4907** | 0.429 | 0.3501 | **0.292** | 0.593 |
| subj7 | **0.4608** | 0.427 | 0.3627 | **0.261** | 0.574 |
| **Mean** | **0.4402** | 0.4293 | 0.3597 | **0.2610** | 0.543 |

- **Pearson: hoà 2–2.** TBEn thắng subj5 (+0.062) và subj7 (+0.034); CHASMBrain thắng
  subj1 (+0.021) và subj2 (+0.029). Chênh lệch trung bình +0.011 nằm trong độ lệch
  giữa các subject (SD 0.025–0.086), không phải khoảng cách thật.
- **MSE: CHASMBrain thắng cả 4 subject**, biên rộng và nhất quán (0.261 vs 0.360).
- **Tham số:** TBEn 16.4–18.8M (thay đổi theo số voxel vì readout là `768 × n_voxels`)
  so với CHASMBrain 282M — khoảng 6%.

## Noise ceiling

Tính từ tương quan giữa 3 trial lặp trên chính test set, hiệu chỉnh Spearman–Brown.
Thứ tự trần khớp đúng thứ tự điểm số của cả hai model.

| | subj1 | subj2 | subj5 | subj7 | Mean |
|---|---|---|---|---|---|
| Inter-trial r | 0.1423 | 0.1009 | 0.1717 | 0.1547 | 0.1424 |
| Noise ceiling | 0.553 | 0.454 | 0.593 | 0.574 | 0.543 |
| % trần — TBEn | 79.5 | 81.4 | 82.7 | 80.3 | 81.0 |
| % trần — CHASMBrain | 83.4 | 87.9 | 72.3 | 74.4 | 79.5 |

Số này đo trực tiếp trên dữ liệu của mình, dùng được để thay trích dẫn `~0.6` chung chung
ở đoạn phản biện Synbrain.

## Không có leakage

Đối chiếu CLS token của 1,000 ảnh test với 9,000 ảnh train ở cả 4 subject:
**0 cặp trùng**, cosine cao nhất 0.95–0.97 (mức của hai ảnh tự nhiên giống nhau).
subj5 cao vì noise ceiling cao nhất, không vì rò rỉ.

## Caveat (phải giữ khi trích dẫn)

1. **subj2 và subj5 dừng tay ở epoch 12** khi val Pearson vẫn tăng đơn điệu → là **cận dưới**.
   Chỉ subj7 chạy hết tới early-stopping (ep 40, best ep 20). Sai lệch này làm baseline
   TBEn yếu đi, tức nghiêng về phía có lợi cho bài.
2. **subj1** lấy từ một run đã bị ghi đè log/JSON/checkpoint; số còn đúng nhưng không còn artifact.
3. Pearson của TBEn đo bằng `validate()` trong train script, không cùng đường với Table 2.
   Chênh lệch đo được giữa hai evaluator là 0.0004. Số cuối nên chạy qua
   `scripts/evaluate_tben_comparison.py` (một `sample_metrics` chung, bootstrap CI,
   paired permutation test).
4. **Variant B chưa có kết quả** tại thời điểm ghi file này.

## Đường viết đề xuất

Bài đã có sẵn lập luận đúng cho hình dạng kết quả này (`manuscripts/experiments.tex`:
hierarchy tạo phần lớn correlation gain, Mamba quyết định độ chính xác biên độ).
TBEn rơi đúng khuôn: readout có cấu trúc lấy được correlation, thiếu tầng refine thì
MSE hỏng ở cả 4 subject. Điều kiện để dùng lập luận này: nâng MSE lên ngang hàng với
Pearson trong cách trình bày, vì ở Pearson kết quả là hoà với một model 6% kích thước.

## Tái tạo

```bash
# Variant A, một subject
.venv/bin/python scripts/train_direct_stage2.py \
  --config configs/train_tben_one_stage_subj7.yaml --seed 42 \
  --save_dir checkpoints/wacv_round2/tben_one_stage/subj7_seed42 \
  --result results/wacv_round2/tben_one_stage_subj7_seed42.json

# Cả hai variant, cả 4 subject
./scripts/run_tben_multisubject.sh
```

Model: `model/tben_official.py`. Tương đương giữa gather của ta và masked-sum của tác giả
được assert trong `TBEnOneStageOfficial.verify_masked_sum_equivalence` (lệch tối đa 2.4e-06).

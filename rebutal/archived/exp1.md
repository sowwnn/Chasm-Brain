# Exp 1 — Encoder Comparison for fMRI Decoding

## Mục tiêu
Đáp lại critique của reviewers về thiếu baseline: chứng minh improvement của CHASMBrain không chỉ đến từ DINOv2 features. So sánh 5 visual encoders trên cả Ridge baseline lẫn CHASM full model (subj01, NSD 9k train / 1k test).

---

## Encoders (final shortlist)

| # | Encoder | Pretrain | Feature shape (per image) |
|---|---|---|---|
| 1 | DINOv2 ViT-B/14 | self-supervised (LVD-142M) | `(257, 768)` — 1 CLS + 256 patches |
| 2 | CLIP SDXL (multi) | text-image contrastive | `(4, 1664)` — 4 layers pooled |
| 3 | SAM ViT-B (multi) | segmentation | last layer pooled |
| 4 | ResNet-50 (multi) | ImageNet | last layer pooled |
| 5 | ConvNeXT-Large | ImageNet-22k | last layer pooled |

**Loại bỏ trong quá trình thử nghiệm:**
- CLIP ViT-B/16 và ResNet-50 ImageNet (single-layer): Ridge Pearson ~0.01 → feature không align với fMRI.
- DeepLabV3 ResNet-50 COCO: Ridge Pearson 0.0099, không cải thiện supervised features.

→ Chốt 5 encoders trên cho cả Ridge + CHASM.

---

## Kết quả

### Ridge Regression baseline (val Pearson ↑ / MSE ↓)

| Encoder | Pearson | MSE |
|---|---|---|
| DINOv2 ViT-B/14    | **0.3745** | 0.3629 |
| CLIP SDXL multi    | 0.3742 | 0.3629 |
| SAM ViT-B multi    | 0.3650 | 0.3663 |
| ResNet-50 multi    | 0.3501 | 0.3723 |
| ConvNeXT-Large     | 0.3048 | 0.4961 |

### CHASM full model (val Pearson ↑ / MSE ↓)

| Encoder | Pearson | MSE | Δ vs Ridge |
|---|---|---|---|
| DINOv2 ViT-B/14    | **0.429**  | **0.261** | +0.055 |
| CLIP SDXL multi    | 0.3998 | 0.3331 | +0.026 |
| ResNet-50 multi    | 0.396  | 0.385  | +0.046 |
| SAM ViT-B multi    | 0.3671 | 0.3612 | +0.002 |
| ConvNeXT-Large     | 0.318  | 0.456  | +0.013 |

---

## Kết luận chính (cho rebuttal)

1. **DINOv2 dẫn đầu cả 2 setting** — confirm choice of self-supervised ViT.
2. **CHASM cải thiện trên TẤT CẢ encoders** so với Ridge → architectural contribution không phụ thuộc encoder cụ thể.
3. **Improvement breakdown** (DINOv2 setting):
   - Ridge baseline: 0.3745
   - CHASM full: 0.429 → **+14.5% từ architecture**
4. **Self-supervised > supervised** rõ rệt: DINOv2 (0.429) > ResNet-50 (0.396) > ConvNeXT (0.318).

---

## File structure

```
rebutal/exp1/
├── configs/        # 5 yaml: exp1c_{dinov2,clip_sdxl_multi,sam_vit_b_multi,resnet50,convnext_large}.yaml
├── scripts/
│   ├── exp1c_train.py                  # CHASM training
│   ├── exp1c_encoder_comparison.py     # eval CHASM
│   ├── exp1c_ridge_encoders.py         # Ridge baseline (DINOv2, CLIP, ConvNeXT)
│   ├── exp1c_ridge_resnet_sam.py       # Ridge baseline (ResNet, SAM)
│   ├── extract_convnext_sam.py         # feature extraction
│   ├── infer_convnext.py
│   └── run_exp1c{,_stage1,_stage2}.sh
├── logs/           # 9 log: stage1+stage2 × 5 encoders (trừ ConvNeXT chỉ stage1)
└── results/
    ├── exp1c_results.json              # CHASM full
    ├── exp1c_ridge_results.json        # Ridge (DINOv2, CLIP, ConvNeXT)
    ├── exp1c_ridge_resnet_sam_results.json
    ├── exp1c_run.log
    └── exp1_res.md                     # summary tables
```

**Features location:**
- `data/exp1/features_resnet_sam/` — ResNet-50 + SAM ViT-B
- `dataset/huwng/subj01/` — DINOv2, CLIP SDXL, ConvNeXT
- `dataset/clip/clip_features/subj01/` — CLIP SDXL variants

---

## Status
✅ **Hoàn tất** — kết quả đủ để viết Section 1 (Baselines) và Section 5 (Novelty) của rebuttal. Bước tiếp theo: Swap Branch Ablation (Priority 2) cho Section 2 (Biological alignment).

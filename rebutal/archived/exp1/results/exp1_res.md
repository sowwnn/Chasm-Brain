---
name: Encoder shortlist for rebuttal experiments
description: Final list of 5 encoders chosen for rebuttal comparison (Ridge + CHASM full model)
type: project
originSessionId: 0d6438d9-8933-416f-b75d-aef0454d7b95
---
Encoders chốt cho rebuttal (subj01, NSD 9k/1k train/test):

1. DINOv2 ViT-B/14
2. CLIP SDXL multi (last layer)
3. SAM ViT-B multi (last layer)
4. ResNet-50 multi (last layer)
5. ConvNeXT-Large

## Ridge baseline (val Pearson / MSE)

| Encoder | Pearson | MSE |
|---|---|---|
| DINOv2 ViT-B/14 | 0.3745 | 0.3629 |
| CLIP SDXL multi | 0.3742 | 0.3629 |
| SAM ViT-B multi | 0.3650 | 0.3663 |
| ResNet-50 multi | 0.3501 | 0.3723 |
| ConvNeXT-Large  | 0.3048 | 0.4961 |

Results saved: `rebutal/results/exp1c_ridge_results.json`

## CHASM full model (val Pearson / MSE)

| Encoder | Pearson | MSE |
|---|---|---|
| DINOv2 ViT-B/14 | 0.429 | 0.261 |
| CLIP SDXL multi | 0.3998 | 0.3331|
| SAM ViT-B multi | 0.3671 | 0.3612 |
| ResNet-50 multi | 0.396  | 0.385 |
| ConvNeXT-Large  | 0.318 | 0.456 |

Results saved: `rebutal/results/exp1c_results.json`

Features location: `data/exp1/features_resnet_sam/` (ResNet + SAM), `dataset/huwng/subj01/` (DINOv2, CLIP, ConvNeXT).
